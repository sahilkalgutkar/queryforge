use crate::array::{Array, ArrayBuilder};
use crate::bitmap::Bitmap;
use qf_common::{Error, Result, Schema, Value};
use std::sync::Arc;

/// The unit of work that moves between operators: a schema plus one array per
/// column, all of the same length.
///
/// Everything downstream is written to consume batches rather than rows. A
/// filter evaluates its predicate over a whole column and hands on a batch;
/// nothing in the engine has a "next row" method, because per-row dispatch
/// through the expression tree is the cost the vectorised model exists to
/// amortise.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordBatch {
    schema: Arc<Schema>,
    columns: Vec<Array>,
    num_rows: usize,
}

/// Rows per batch. Big enough to amortise per-batch overhead, small enough that
/// a batch's working set stays in L2 while an operator runs over it.
pub const DEFAULT_BATCH_SIZE: usize = 8192;

impl RecordBatch {
    pub fn try_new(schema: Arc<Schema>, columns: Vec<Array>) -> Result<RecordBatch> {
        if schema.len() != columns.len() {
            return Err(Error::internal(format!(
                "schema has {} fields but {} columns were supplied",
                schema.len(),
                columns.len()
            )));
        }
        let num_rows = columns.first().map_or(0, Array::len);
        for (i, col) in columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(Error::internal(format!(
                    "column {i} has {} rows, expected {num_rows}",
                    col.len()
                )));
            }
            let expected = schema.field(i)?.data_type;
            if col.data_type() != expected {
                return Err(Error::internal(format!(
                    "column {i} is {} but the schema says {expected}",
                    col.data_type()
                )));
            }
        }
        Ok(RecordBatch {
            schema,
            columns,
            num_rows,
        })
    }

    /// A batch with the right schema and no rows, which operators emit when a
    /// filter rejects everything.
    pub fn empty(schema: Arc<Schema>) -> Result<RecordBatch> {
        let columns = schema
            .fields()
            .iter()
            .map(|f| Array::nulls(f.data_type, 0))
            .collect::<Result<Vec<_>>>()?;
        RecordBatch::try_new(schema, columns)
    }

    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    pub fn columns(&self) -> &[Array] {
        &self.columns
    }

    pub fn column(&self, i: usize) -> Result<&Array> {
        self.columns
            .get(i)
            .ok_or_else(|| Error::internal(format!("no column at index {i}")))
    }

    pub fn column_by_name(&self, name: &str) -> Result<&Array> {
        self.column(self.schema.index_of(name)?)
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Materialises one row as scalars. Used by the CLI's printer and by tests;
    /// the operators never call it.
    pub fn row(&self, i: usize) -> Vec<Value> {
        self.columns.iter().map(|c| c.value(i)).collect()
    }

    pub fn rows(&self) -> impl Iterator<Item = Vec<Value>> + '_ {
        (0..self.num_rows).map(move |i| self.row(i))
    }

    pub fn filter(&self, mask: &Bitmap) -> Result<RecordBatch> {
        let columns = self
            .columns
            .iter()
            .map(|c| c.filter(mask))
            .collect::<Result<Vec<_>>>()?;
        // A batch with zero columns still has to remember how many rows
        // survived, or `SELECT count(*)` after a filter counts the wrong thing.
        if columns.is_empty() {
            return Ok(RecordBatch {
                schema: Arc::clone(&self.schema),
                columns,
                num_rows: mask.count_set(),
            });
        }
        RecordBatch::try_new(Arc::clone(&self.schema), columns)
    }

    pub fn take(&self, indices: &[usize]) -> Result<RecordBatch> {
        let columns = self
            .columns
            .iter()
            .map(|c| c.take(indices))
            .collect::<Result<Vec<_>>>()?;
        if columns.is_empty() {
            return Ok(RecordBatch {
                schema: Arc::clone(&self.schema),
                columns,
                num_rows: indices.len(),
            });
        }
        RecordBatch::try_new(Arc::clone(&self.schema), columns)
    }

    /// Keeps `columns` in the order given, producing the matching schema.
    pub fn project(&self, indices: &[usize]) -> Result<RecordBatch> {
        let schema = Arc::new(self.schema.project(indices)?);
        let mut columns = Vec::with_capacity(indices.len());
        for &i in indices {
            columns.push(self.column(i)?.clone());
        }
        if columns.is_empty() {
            return Ok(RecordBatch {
                schema,
                columns,
                num_rows: self.num_rows,
            });
        }
        RecordBatch::try_new(schema, columns)
    }

    /// Glues two batches side by side. This is how a join builds its output
    /// once both sides have been gathered to matching row counts.
    pub fn hstack(left: &RecordBatch, right: &RecordBatch) -> Result<RecordBatch> {
        if left.num_rows != right.num_rows {
            return Err(Error::internal(format!(
                "cannot stack a {}-row batch beside a {}-row batch",
                left.num_rows, right.num_rows
            )));
        }
        let schema = Arc::new(Schema::join(&left.schema, &right.schema));
        let mut columns = left.columns.clone();
        columns.extend(right.columns.iter().cloned());
        if columns.is_empty() {
            return Ok(RecordBatch {
                schema,
                columns,
                num_rows: left.num_rows,
            });
        }
        RecordBatch::try_new(schema, columns)
    }

    /// Concatenates batches that share a schema, which is what a collecting
    /// operator does before it sorts or returns.
    pub fn concat(schema: Arc<Schema>, batches: &[RecordBatch]) -> Result<RecordBatch> {
        if batches.is_empty() {
            return RecordBatch::empty(schema);
        }
        let mut builders: Vec<ArrayBuilder> = schema
            .fields()
            .iter()
            .map(|f| ArrayBuilder::new(f.data_type))
            .collect();
        for b in batches {
            if b.schema.as_ref() != schema.as_ref() {
                return Err(Error::internal(
                    "cannot concatenate batches with different schemas".to_string(),
                ));
            }
            for (ci, builder) in builders.iter_mut().enumerate() {
                let col = b.column(ci)?;
                for r in 0..b.num_rows {
                    builder.push_from(col, r);
                }
            }
        }
        let columns = builders
            .into_iter()
            .map(ArrayBuilder::finish)
            .collect::<Result<Vec<_>>>()?;
        if columns.is_empty() {
            let num_rows = batches.iter().map(|b| b.num_rows).sum();
            return Ok(RecordBatch {
                schema,
                columns,
                num_rows,
            });
        }
        RecordBatch::try_new(schema, columns)
    }

    /// Splits into batches of at most `size` rows, restoring the batch size
    /// after an operator (a join, say) has produced something much larger.
    pub fn chunks(&self, size: usize) -> Result<Vec<RecordBatch>> {
        if size == 0 {
            return Err(Error::internal("batch size must be positive".to_string()));
        }
        if self.num_rows <= size {
            return Ok(vec![self.clone()]);
        }
        let mut out = Vec::new();
        let mut start = 0;
        while start < self.num_rows {
            let end = (start + size).min(self.num_rows);
            let idx: Vec<usize> = (start..end).collect();
            out.push(self.take(&idx)?);
            start = end;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qf_common::{DataType, Field};

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn batch(ids: &[i64], names: &[&str]) -> RecordBatch {
        let ids: Vec<Value> = ids.iter().map(|i| Value::Int64(*i)).collect();
        let names: Vec<Value> = names.iter().map(|s| Value::Utf8(s.to_string())).collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Array::from_values(DataType::Int64, &ids).unwrap(),
                Array::from_values(DataType::Utf8, &names).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn a_batch_knows_its_shape() {
        let b = batch(&[1, 2, 3], &["a", "b", "c"]);
        assert_eq!(b.num_rows(), 3);
        assert_eq!(b.num_columns(), 2);
        assert!(!b.is_empty());
        assert_eq!(b.row(1), vec![Value::Int64(2), Value::Utf8("b".into())]);
    }

    #[test]
    fn columns_of_unequal_length_are_rejected() {
        let err = RecordBatch::try_new(
            schema(),
            vec![
                Array::from_values(DataType::Int64, &[Value::Int64(1)]).unwrap(),
                Array::from_values(DataType::Utf8, &[]).unwrap(),
            ],
        )
        .unwrap_err();
        assert!(matches!(err, Error::Internal(_)));
    }

    #[test]
    fn a_column_whose_type_contradicts_the_schema_is_rejected() {
        let err = RecordBatch::try_new(
            schema(),
            vec![
                Array::from_values(DataType::Utf8, &[Value::Utf8("x".into())]).unwrap(),
                Array::from_values(DataType::Utf8, &[Value::Utf8("y".into())]).unwrap(),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("schema says"));
    }

    #[test]
    fn a_column_count_mismatch_is_rejected() {
        assert!(RecordBatch::try_new(schema(), vec![]).is_err());
    }

    #[test]
    fn filter_and_take_agree() {
        let b = batch(&[1, 2, 3, 4], &["a", "b", "c", "d"]);
        let mask: Bitmap = [false, true, false, true].into_iter().collect();
        assert_eq!(b.filter(&mask).unwrap(), b.take(&[1, 3]).unwrap());
        assert_eq!(b.filter(&mask).unwrap().num_rows(), 2);
    }

    #[test]
    fn projection_reorders_columns_and_the_schema_with_them() {
        let b = batch(&[1], &["a"]);
        let p = b.project(&[1, 0]).unwrap();
        assert_eq!(p.schema().field(0).unwrap().name, "name");
        assert_eq!(p.row(0), vec![Value::Utf8("a".into()), Value::Int64(1)]);
    }

    #[test]
    fn a_zero_column_projection_still_remembers_its_row_count() {
        // `SELECT count(*) FROM t WHERE ...` projects away every column; if the
        // row count went with them the count would come back as zero.
        let b = batch(&[1, 2, 3], &["a", "b", "c"]);
        let p = b.project(&[]).unwrap();
        assert_eq!(p.num_columns(), 0);
        assert_eq!(p.num_rows(), 3);

        let mask: Bitmap = [true, false, true].into_iter().collect();
        assert_eq!(p.filter(&mask).unwrap().num_rows(), 2);
        assert_eq!(p.take(&[0, 0, 1, 2]).unwrap().num_rows(), 4);
    }

    #[test]
    fn hstack_joins_two_batches_side_by_side() {
        let l = batch(&[1, 2], &["a", "b"]);
        let r = batch(&[9, 8], &["x", "y"]);
        let s = RecordBatch::hstack(&l, &r).unwrap();
        assert_eq!(s.num_columns(), 4);
        assert_eq!(s.num_rows(), 2);
        assert_eq!(s.row(0)[2], Value::Int64(9));
    }

    #[test]
    fn hstack_refuses_mismatched_row_counts() {
        let l = batch(&[1, 2], &["a", "b"]);
        let r = batch(&[9], &["x"]);
        assert!(RecordBatch::hstack(&l, &r).is_err());
    }

    #[test]
    fn concat_preserves_order_across_batches() {
        let a = batch(&[1, 2], &["a", "b"]);
        let b = batch(&[3], &["c"]);
        let c = RecordBatch::concat(schema(), &[a, b]).unwrap();
        assert_eq!(c.num_rows(), 3);
        assert_eq!(
            c.column(0).unwrap().iter().collect::<Vec<_>>(),
            vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)]
        );
    }

    #[test]
    fn concat_of_nothing_yields_an_empty_batch_with_the_right_schema() {
        let c = RecordBatch::concat(schema(), &[]).unwrap();
        assert!(c.is_empty());
        assert_eq!(c.num_columns(), 2);
        assert_eq!(RecordBatch::empty(schema()).unwrap(), c);
    }

    #[test]
    fn concat_rejects_a_batch_with_a_different_schema() {
        let other = Arc::new(Schema::new(vec![Field::new("z", DataType::Int64, false)]));
        let odd = RecordBatch::empty(other).unwrap();
        assert!(RecordBatch::concat(schema(), &[batch(&[1], &["a"]), odd]).is_err());
    }

    #[test]
    fn chunks_splits_a_large_batch_and_loses_no_rows() {
        let ids: Vec<i64> = (0..2500).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let b = batch(&ids, &refs);
        let chunks = b.chunks(1000).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2].num_rows(), 500);
        let total: usize = chunks.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total, 2500);
        assert_eq!(RecordBatch::concat(schema(), &chunks).unwrap(), b);
    }

    #[test]
    fn a_batch_smaller_than_the_chunk_size_passes_through_whole() {
        let b = batch(&[1], &["a"]);
        assert_eq!(b.chunks(DEFAULT_BATCH_SIZE).unwrap().len(), 1);
        assert!(b.chunks(0).is_err());
    }

    #[test]
    fn columns_can_be_addressed_by_name_or_index() {
        let b = batch(&[7], &["a"]);
        assert_eq!(b.column_by_name("ID").unwrap().value(0), Value::Int64(7));
        assert!(b.column(5).is_err());
        assert!(b.column_by_name("nope").is_err());
        assert_eq!(b.columns().len(), 2);
        assert_eq!(b.rows().count(), 1);
    }
}
