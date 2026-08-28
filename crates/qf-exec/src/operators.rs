//! The streaming operators: filter, projection, limit, distinct and a literal
//! source. None of them buffer, so a `LIMIT 10` over a billion-row table stops
//! the scan after the first batch or two.

use crate::eval::{evaluate, evaluate_predicate};
use crate::operator::{Metrics, Operator};
use qf_common::{Field, Result, Schema, Value};
use qf_plan::expr::BoundExpr;
use qf_storage::batch::RecordBatch;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

pub struct FilterExec {
    input: Box<dyn Operator>,
    predicate: BoundExpr,
    metrics: Metrics,
}

impl FilterExec {
    pub fn new(input: Box<dyn Operator>, predicate: BoundExpr) -> FilterExec {
        FilterExec {
            input,
            predicate,
            metrics: Metrics::default(),
        }
    }
}

impl Operator for FilterExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(batch) = self.input.next_batch()? else {
            return Ok(None);
        };
        let start = Instant::now();
        let mask = evaluate_predicate(&self.predicate, &batch)?;
        let out = batch.filter(&mask)?;
        self.metrics.elapsed += start.elapsed();
        self.metrics.record(&out);
        Ok(Some(out))
    }

    fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn children(&self) -> Vec<&dyn Operator> {
        vec![self.input.as_ref()]
    }

    fn name(&self) -> &'static str {
        "Filter"
    }
}

pub struct ProjectionExec {
    input: Box<dyn Operator>,
    exprs: Vec<(BoundExpr, String)>,
    schema: Arc<Schema>,
    metrics: Metrics,
}

impl ProjectionExec {
    pub fn new(input: Box<dyn Operator>, exprs: Vec<(BoundExpr, String)>) -> ProjectionExec {
        let schema = Arc::new(Schema::new(
            exprs
                .iter()
                .map(|(e, n)| Field::new(n, e.data_type(), true))
                .collect(),
        ));
        ProjectionExec {
            input,
            exprs,
            schema,
            metrics: Metrics::default(),
        }
    }
}

impl Operator for ProjectionExec {
    fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(batch) = self.input.next_batch()? else {
            return Ok(None);
        };
        let start = Instant::now();
        let columns = self
            .exprs
            .iter()
            .map(|(e, _)| evaluate(e, &batch))
            .collect::<Result<Vec<_>>>()?;
        // A projection that produces no columns still has to carry the row
        // count, or `SELECT count(*)` above it counts nothing.
        let out = if columns.is_empty() {
            RecordBatch::empty(Arc::clone(&self.schema))?.take(&vec![0usize; batch.num_rows()])?
        } else {
            RecordBatch::try_new(Arc::clone(&self.schema), columns)?
        };
        self.metrics.elapsed += start.elapsed();
        self.metrics.record(&out);
        Ok(Some(out))
    }

    fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn children(&self) -> Vec<&dyn Operator> {
        vec![self.input.as_ref()]
    }

    fn name(&self) -> &'static str {
        "Projection"
    }
}

/// `LIMIT n OFFSET k`, applied while streaming.
pub struct LimitExec {
    input: Box<dyn Operator>,
    limit: Option<usize>,
    offset: usize,
    skipped: usize,
    emitted: usize,
    metrics: Metrics,
}

impl LimitExec {
    pub fn new(input: Box<dyn Operator>, limit: Option<usize>, offset: usize) -> LimitExec {
        LimitExec {
            input,
            limit,
            offset,
            skipped: 0,
            emitted: 0,
            metrics: Metrics::default(),
        }
    }

    fn finished(&self) -> bool {
        self.limit.is_some_and(|l| self.emitted >= l)
    }
}

impl Operator for LimitExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        loop {
            if self.finished() {
                // Stop pulling. This is the whole point of a pull model: the
                // scan below never reads the rest of the table.
                return Ok(None);
            }
            let Some(batch) = self.input.next_batch()? else {
                return Ok(None);
            };
            let rows = batch.num_rows();
            if rows == 0 {
                continue;
            }
            let start = Instant::now();

            let skip = self.offset.saturating_sub(self.skipped).min(rows);
            self.skipped += skip;
            let available = rows - skip;
            if available == 0 {
                self.metrics.elapsed += start.elapsed();
                continue;
            }
            let take = match self.limit {
                Some(l) => available.min(l - self.emitted),
                None => available,
            };
            let indices: Vec<usize> = (skip..skip + take).collect();
            let out = batch.take(&indices)?;
            self.emitted += take;
            self.metrics.elapsed += start.elapsed();
            self.metrics.record(&out);
            return Ok(Some(out));
        }
    }

    fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn children(&self) -> Vec<&dyn Operator> {
        vec![self.input.as_ref()]
    }

    fn name(&self) -> &'static str {
        "Limit"
    }
}

/// `SELECT DISTINCT`, streaming and order-preserving.
///
/// Keeping the first occurrence rather than sorting means a `DISTINCT` under
/// an `ORDER BY` does not undo the sort, and rows come out as soon as they are
/// seen instead of after the input is exhausted.
pub struct DistinctExec {
    input: Box<dyn Operator>,
    seen: HashSet<Vec<Value>>,
    metrics: Metrics,
}

impl DistinctExec {
    pub fn new(input: Box<dyn Operator>) -> DistinctExec {
        DistinctExec {
            input,
            seen: HashSet::new(),
            metrics: Metrics::default(),
        }
    }
}

impl Operator for DistinctExec {
    fn schema(&self) -> Arc<Schema> {
        self.input.schema()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(batch) = self.input.next_batch()? else {
            return Ok(None);
        };
        let start = Instant::now();
        let mut keep = Vec::new();
        for i in 0..batch.num_rows() {
            let key = batch.row(i);
            if self.seen.insert(key) {
                keep.push(i);
            }
        }
        let out = batch.take(&keep)?;
        self.metrics.elapsed += start.elapsed();
        self.metrics.record(&out);
        Ok(Some(out))
    }

    fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn children(&self) -> Vec<&dyn Operator> {
        vec![self.input.as_ref()]
    }

    fn name(&self) -> &'static str {
        "Distinct"
    }
}

/// Literal rows: `INSERT ... VALUES`, and the single empty row a `SELECT` with
/// no `FROM` runs over.
pub struct ValuesExec {
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
    next: usize,
    metrics: Metrics,
}

impl ValuesExec {
    pub fn new(schema: Arc<Schema>, batches: Vec<RecordBatch>) -> ValuesExec {
        ValuesExec {
            schema,
            batches,
            next: 0,
            metrics: Metrics::default(),
        }
    }

    /// One row with no columns — what `SELECT 1 + 1` evaluates against.
    pub fn single_row() -> Result<ValuesExec> {
        let schema = Arc::new(Schema::empty());
        let batch = RecordBatch::empty(Arc::clone(&schema))?.take(&[0usize; 1])?;
        Ok(ValuesExec::new(schema, vec![batch]))
    }
}

impl Operator for ValuesExec {
    fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let b = self.batches[self.next].clone();
        self.next += 1;
        self.metrics.record(&b);
        Ok(Some(b))
    }

    fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn name(&self) -> &'static str {
        "Values"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::{collect_one, Operator};
    use qf_common::DataType;
    use qf_sql::ast::BinaryOp;
    use qf_storage::array::Array;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8, false),
        ]))
    }

    fn source(values: &[&[i64]]) -> Box<dyn Operator> {
        let batches = values
            .iter()
            .map(|vs| {
                let a: Vec<Value> = vs.iter().map(|v| Value::Int64(*v)).collect();
                let b: Vec<Value> = vs.iter().map(|v| Value::Utf8(format!("v{v}"))).collect();
                RecordBatch::try_new(
                    schema(),
                    vec![
                        Array::from_values(DataType::Int64, &a).unwrap(),
                        Array::from_values(DataType::Utf8, &b).unwrap(),
                    ],
                )
                .unwrap()
            })
            .collect();
        Box::new(ValuesExec::new(schema(), batches))
    }

    fn column(op: &mut dyn Operator, index: usize) -> Vec<String> {
        collect_one(op)
            .unwrap()
            .column(index)
            .unwrap()
            .iter()
            .map(|v| v.to_string())
            .collect()
    }

    fn col(i: usize, t: DataType) -> BoundExpr {
        BoundExpr::column(i, format!("c{i}"), t)
    }

    #[test]
    fn a_filter_keeps_matching_rows_across_batches() {
        let mut op = FilterExec::new(
            source(&[&[1, 2, 3], &[4, 5]]),
            BoundExpr::binary(
                col(0, DataType::Int64),
                BinaryOp::Gt,
                BoundExpr::Literal(Value::Int64(2)),
            )
            .unwrap(),
        );
        assert_eq!(column(&mut op, 0), vec!["3", "4", "5"]);
        assert_eq!(op.name(), "Filter");
        assert_eq!(op.children().len(), 1);
        assert_eq!(op.metrics().rows_out, 3);
    }

    #[test]
    fn a_filter_that_rejects_a_whole_batch_keeps_going() {
        // An empty batch is not the end of the input.
        let mut op = FilterExec::new(
            source(&[&[1], &[9]]),
            BoundExpr::binary(
                col(0, DataType::Int64),
                BinaryOp::Gt,
                BoundExpr::Literal(Value::Int64(5)),
            )
            .unwrap(),
        );
        assert_eq!(column(&mut op, 0), vec!["9"]);
    }

    #[test]
    fn a_projection_computes_its_expressions_and_names_its_columns() {
        let mut op = ProjectionExec::new(
            source(&[&[1, 2]]),
            vec![(
                BoundExpr::binary(
                    col(0, DataType::Int64),
                    BinaryOp::Multiply,
                    BoundExpr::Literal(Value::Int64(10)),
                )
                .unwrap(),
                "tens".into(),
            )],
        );
        assert_eq!(op.schema().field(0).unwrap().name, "tens");
        assert_eq!(column(&mut op, 0), vec!["10", "20"]);
        assert_eq!(op.name(), "Projection");
        assert_eq!(op.children().len(), 1);
    }

    #[test]
    fn a_projection_with_no_columns_still_carries_the_row_count() {
        let mut op = ProjectionExec::new(source(&[&[1, 2, 3]]), vec![]);
        let out = collect_one(&mut op).unwrap();
        assert_eq!(out.num_columns(), 0);
        assert_eq!(out.num_rows(), 3);
    }

    #[test]
    fn a_limit_takes_the_first_rows_and_stops_pulling() {
        let mut op = LimitExec::new(source(&[&[1, 2, 3], &[4, 5, 6]]), Some(2), 0);
        assert_eq!(column(&mut op, 0), vec!["1", "2"]);
        assert_eq!(op.name(), "Limit");
        assert_eq!(op.children().len(), 1);
    }

    #[test]
    fn an_offset_can_span_several_batches() {
        let mut op = LimitExec::new(source(&[&[1, 2], &[3, 4], &[5, 6]]), Some(2), 3);
        assert_eq!(column(&mut op, 0), vec!["4", "5"]);
    }

    #[test]
    fn an_offset_past_the_end_returns_nothing() {
        let mut op = LimitExec::new(source(&[&[1, 2]]), None, 10);
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 0);
    }

    #[test]
    fn a_limit_of_zero_returns_nothing_and_a_limit_without_one_returns_everything() {
        let mut zero = LimitExec::new(source(&[&[1, 2]]), Some(0), 0);
        assert_eq!(collect_one(&mut zero).unwrap().num_rows(), 0);
        let mut all = LimitExec::new(source(&[&[1, 2], &[3]]), None, 0);
        assert_eq!(column(&mut all, 0), vec!["1", "2", "3"]);
    }

    #[test]
    fn a_limit_larger_than_the_input_returns_everything() {
        let mut op = LimitExec::new(source(&[&[1, 2]]), Some(50), 0);
        assert_eq!(column(&mut op, 0), vec!["1", "2"]);
    }

    #[test]
    fn distinct_removes_repeats_across_batch_boundaries_and_keeps_first_order() {
        let mut op = DistinctExec::new(source(&[&[3, 1, 3], &[1, 2]]));
        assert_eq!(column(&mut op, 0), vec!["3", "1", "2"]);
        assert_eq!(op.name(), "Distinct");
        assert_eq!(op.children().len(), 1);
    }

    #[test]
    fn values_replays_its_batches_once() {
        let mut op = ValuesExec::new(schema(), vec![]);
        assert!(op.next_batch().unwrap().is_none());
        assert_eq!(op.name(), "Values");
        assert!(op.children().is_empty());
        assert_eq!(op.schema().len(), 2);
    }

    #[test]
    fn the_single_empty_row_has_no_columns_but_one_row() {
        let mut op = ValuesExec::single_row().unwrap();
        let b = collect_one(&mut op).unwrap();
        assert_eq!(b.num_rows(), 1);
        assert_eq!(b.num_columns(), 0);
    }

    #[test]
    fn empty_input_batches_are_skipped_by_the_limit() {
        let mut op = LimitExec::new(source(&[&[], &[1, 2]]), Some(1), 0);
        assert_eq!(column(&mut op, 0), vec!["1"]);
    }
}
