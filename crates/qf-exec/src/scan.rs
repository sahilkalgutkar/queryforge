//! The scan, and the one place where predicate pushdown stops being a plan
//! rewrite and starts saving I/O.
//!
//! For a file-backed table the scan walks row groups. Before reading one it
//! asks each pushed predicate whether the group's zone map could possibly
//! satisfy it; if not, the group's column chunks are never read. What is left
//! is decoded — only the projected columns — and the predicates are applied to
//! the rows that survive.

use crate::eval::evaluate_predicate;
use crate::operator::{Metrics, Operator};
use qf_common::{Result, Schema, Value};
use qf_plan::expr::BoundExpr;
use qf_sql::ast::BinaryOp;
use qf_storage::batch::{RecordBatch, DEFAULT_BATCH_SIZE};
use qf_storage::format::QfcReader;
use qf_storage::stats::PruneOp;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

enum Source {
    /// Batches already in memory, still in the table's full schema.
    Memory {
        batches: Vec<RecordBatch>,
        next: usize,
    },
    File {
        reader: Box<QfcReader>,
        next_group: usize,
        /// Rows left over from a group larger than one batch.
        pending: Vec<RecordBatch>,
    },
}

pub struct ScanExec {
    source: Source,
    /// Source column positions this scan reads, in output order.
    projection: Vec<usize>,
    /// Predicates over the *projected* schema.
    filters: Vec<BoundExpr>,
    schema: Arc<Schema>,
    metrics: Metrics,
}

impl ScanExec {
    pub fn from_batches(
        batches: Vec<RecordBatch>,
        projection: Vec<usize>,
        filters: Vec<BoundExpr>,
        schema: Arc<Schema>,
    ) -> ScanExec {
        ScanExec {
            source: Source::Memory { batches, next: 0 },
            projection,
            filters,
            schema,
            metrics: Metrics::default(),
        }
    }

    pub fn from_file(
        path: impl AsRef<Path>,
        projection: Vec<usize>,
        filters: Vec<BoundExpr>,
        schema: Arc<Schema>,
    ) -> Result<ScanExec> {
        Ok(ScanExec {
            source: Source::File {
                reader: Box::new(QfcReader::open(path)?),
                next_group: 0,
                pending: Vec::new(),
            },
            projection,
            filters,
            schema,
            metrics: Metrics::default(),
        })
    }

    fn apply_filters(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let mut out = batch;
        for f in &self.filters {
            let mask = evaluate_predicate(f, &out)?;
            out = out.filter(&mask)?;
            if out.num_rows() == 0 {
                break;
            }
        }
        Ok(out)
    }
}

/// Can any row of this row group satisfy every pushed predicate?
///
/// The answer is only ever used to *skip* work, so it errs toward reading:
/// anything the zone maps cannot decide comes back `true`.
fn group_may_match(
    filters: &[BoundExpr],
    projection: &[usize],
    group: &qf_storage::format::RowGroupMeta,
) -> bool {
    filters
        .iter()
        .all(|f| conjunct_may_match(f, projection, group))
}

fn conjunct_may_match(
    expr: &BoundExpr,
    projection: &[usize],
    group: &qf_storage::format::RowGroupMeta,
) -> bool {
    match expr {
        // Both halves of an AND must be satisfiable.
        BoundExpr::Binary {
            left,
            op: BinaryOp::And,
            right,
            ..
        } => {
            conjunct_may_match(left, projection, group)
                && conjunct_may_match(right, projection, group)
        }
        // Either half of an OR is enough.
        BoundExpr::Binary {
            left,
            op: BinaryOp::Or,
            right,
            ..
        } => {
            conjunct_may_match(left, projection, group)
                || conjunct_may_match(right, projection, group)
        }
        BoundExpr::Binary {
            left, op, right, ..
        } if op.is_comparison() => {
            let Some((index, literal, op)) = column_literal(left, *op, right) else {
                return true;
            };
            let Some(source) = projection.get(index) else {
                return true;
            };
            let Ok(chunk) = group.column(*source) else {
                return true;
            };
            let Some(prune_op) = prune_op(op) else {
                return true;
            };
            chunk.stats.may_match(prune_op, literal)
        }
        _ => true,
    }
}

/// Recognises `column <op> literal`, mirroring the operator when the literal
/// is on the left.
fn column_literal<'a>(
    left: &'a BoundExpr,
    op: BinaryOp,
    right: &'a BoundExpr,
) -> Option<(usize, &'a Value, BinaryOp)> {
    match (left, right) {
        (BoundExpr::Column { index, .. }, BoundExpr::Literal(v)) => Some((*index, v, op)),
        (BoundExpr::Literal(v), BoundExpr::Column { index, .. }) => {
            Some((*index, v, op.swap_operands()?))
        }
        _ => None,
    }
}

fn prune_op(op: BinaryOp) -> Option<PruneOp> {
    Some(match op {
        BinaryOp::Eq => PruneOp::Eq,
        BinaryOp::NotEq => PruneOp::NotEq,
        BinaryOp::Lt => PruneOp::Lt,
        BinaryOp::LtEq => PruneOp::LtEq,
        BinaryOp::Gt => PruneOp::Gt,
        BinaryOp::GtEq => PruneOp::GtEq,
        _ => return None,
    })
}

impl Operator for ScanExec {
    fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let start = Instant::now();
        let result = match &mut self.source {
            Source::Memory { batches, next } => {
                if *next >= batches.len() {
                    None
                } else {
                    let b = batches[*next].project(&self.projection)?;
                    *next += 1;
                    Some(b)
                }
            }
            Source::File {
                reader,
                next_group,
                pending,
            } => {
                if !pending.is_empty() {
                    Some(pending.remove(0))
                } else {
                    let mut found = None;
                    while *next_group < reader.num_row_groups() {
                        let index = *next_group;
                        *next_group += 1;
                        let meta = reader.row_group(index)?.clone();
                        if !group_may_match(&self.filters, &self.projection, &meta) {
                            self.metrics.row_groups_pruned += 1;
                            continue;
                        }
                        self.metrics.row_groups_read += 1;
                        let batch = reader.read_row_group(index, &self.projection)?;
                        let mut chunks = batch.chunks(DEFAULT_BATCH_SIZE)?;
                        let first = chunks.remove(0);
                        pending.extend(chunks);
                        found = Some(first);
                        break;
                    }
                    found
                }
            }
        };

        let Some(batch) = result else {
            self.metrics.elapsed += start.elapsed();
            return Ok(None);
        };
        let out = self.apply_filters(batch)?;
        self.metrics.elapsed += start.elapsed();
        self.metrics.record(&out);
        Ok(Some(out))
    }

    fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn name(&self) -> &'static str {
        "Scan"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::collect_one;
    use qf_common::{DataType, Field};
    use qf_storage::array::Array;
    use qf_storage::format::write_file;
    use std::path::PathBuf;

    struct TempFile(PathBuf);

    impl TempFile {
        fn new(tag: &str) -> TempFile {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "queryforge-scan-{tag}-{}-{}.qfc",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            TempFile(p)
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("tag", DataType::Utf8, false),
        ]))
    }

    fn batch(range: std::ops::Range<i64>) -> RecordBatch {
        let ids: Vec<Value> = range.clone().map(Value::Int64).collect();
        let tags: Vec<Value> = range.map(|i| Value::Utf8(format!("t{i}"))).collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Array::from_values(DataType::Int64, &ids).unwrap(),
                Array::from_values(DataType::Utf8, &tags).unwrap(),
            ],
        )
        .unwrap()
    }

    fn id_col() -> BoundExpr {
        BoundExpr::column(0, "id", DataType::Int64)
    }

    fn cmp(op: BinaryOp, v: i64) -> BoundExpr {
        BoundExpr::binary(id_col(), op, BoundExpr::Literal(Value::Int64(v))).unwrap()
    }

    fn file_scan(path: &PathBuf, filters: Vec<BoundExpr>) -> ScanExec {
        ScanExec::from_file(path, vec![0, 1], filters, schema()).unwrap()
    }

    fn write(path: &PathBuf, rows: std::ops::Range<i64>, group: usize) {
        write_file(path, schema(), std::slice::from_ref(&batch(rows)), group).unwrap();
    }

    #[test]
    fn a_memory_scan_projects_and_filters() {
        let mut op = ScanExec::from_batches(
            vec![batch(0..10)],
            vec![0],
            vec![cmp(BinaryOp::Gt, 7)],
            Arc::new(schema().project(&[0]).unwrap()),
        );
        let out = collect_one(&mut op).unwrap();
        assert_eq!(out.num_columns(), 1);
        assert_eq!(out.num_rows(), 2);
        assert_eq!(op.name(), "Scan");
    }

    #[test]
    fn a_range_predicate_prunes_the_groups_it_cannot_reach() {
        let f = TempFile::new("range");
        write(&f.0, 0..500, 100);
        let mut op = file_scan(&f.0, vec![cmp(BinaryOp::GtEq, 400)]);
        let out = collect_one(&mut op).unwrap();
        assert_eq!(out.num_rows(), 100);
        assert_eq!(op.metrics().row_groups_read, 1);
        assert_eq!(op.metrics().row_groups_pruned, 4);
    }

    #[test]
    fn a_literal_on_the_left_prunes_just_as_well() {
        let f = TempFile::new("flipped");
        write(&f.0, 0..500, 100);
        // `400 <= id` must prune the same groups as `id >= 400`.
        let expr = BoundExpr::binary(
            BoundExpr::Literal(Value::Int64(400)),
            BinaryOp::LtEq,
            id_col(),
        )
        .unwrap();
        let mut op = file_scan(&f.0, vec![expr]);
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 100);
        assert_eq!(op.metrics().row_groups_pruned, 4);
    }

    #[test]
    fn a_conjunction_prunes_a_group_that_either_half_rules_out() {
        let f = TempFile::new("and");
        write(&f.0, 0..500, 100);
        let both = BoundExpr::binary(
            cmp(BinaryOp::GtEq, 150),
            BinaryOp::And,
            cmp(BinaryOp::Lt, 180),
        )
        .unwrap();
        let mut op = file_scan(&f.0, vec![both]);
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 30);
        assert_eq!(op.metrics().row_groups_read, 1);
        assert_eq!(op.metrics().row_groups_pruned, 4);
    }

    #[test]
    fn a_disjunction_only_prunes_a_group_that_neither_half_can_reach() {
        let f = TempFile::new("or");
        write(&f.0, 0..500, 100);
        let either = BoundExpr::binary(
            cmp(BinaryOp::Lt, 50),
            BinaryOp::Or,
            cmp(BinaryOp::GtEq, 450),
        )
        .unwrap();
        let mut op = file_scan(&f.0, vec![either]);
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 100);
        // The first and last groups survive; the middle three are ruled out by
        // both halves.
        assert_eq!(op.metrics().row_groups_read, 2);
        assert_eq!(op.metrics().row_groups_pruned, 3);
    }

    #[test]
    fn a_predicate_the_zone_maps_cannot_decide_leaves_every_group_readable() {
        let f = TempFile::new("undecidable");
        write(&f.0, 0..300, 100);
        // Neither side is a plain column against a literal.
        let expr = BoundExpr::binary(
            BoundExpr::binary(
                id_col(),
                BinaryOp::Plus,
                BoundExpr::Literal(Value::Int64(1)),
            )
            .unwrap(),
            BinaryOp::Gt,
            BoundExpr::Literal(Value::Int64(250)),
        )
        .unwrap();
        let mut op = file_scan(&f.0, vec![expr]);
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 50);
        assert_eq!(op.metrics().row_groups_pruned, 0);
        assert_eq!(op.metrics().row_groups_read, 3);
    }

    #[test]
    fn a_non_comparison_predicate_is_applied_but_never_prunes() {
        let f = TempFile::new("isnull");
        write(&f.0, 0..200, 100);
        let expr = BoundExpr::IsNull {
            expr: Box::new(id_col()),
            negated: true,
        };
        let mut op = file_scan(&f.0, vec![expr]);
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 200);
        assert_eq!(op.metrics().row_groups_pruned, 0);
    }

    #[test]
    fn pruning_everything_returns_no_rows_and_reads_nothing() {
        let f = TempFile::new("all-pruned");
        write(&f.0, 0..300, 100);
        let mut op = file_scan(&f.0, vec![cmp(BinaryOp::Gt, 10_000)]);
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 0);
        assert_eq!(op.metrics().row_groups_read, 0);
        assert_eq!(op.metrics().row_groups_pruned, 3);
    }

    #[test]
    fn a_scan_with_no_predicate_reads_every_group() {
        let f = TempFile::new("plain");
        write(&f.0, 0..250, 100);
        let mut op = file_scan(&f.0, vec![]);
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 250);
        assert_eq!(op.metrics().row_groups_read, 3);
        assert_eq!(op.metrics().row_groups_pruned, 0);
    }

    #[test]
    fn a_projected_scan_maps_predicate_positions_back_to_source_columns() {
        // Reading only `tag` puts it at position 0, but its statistics live at
        // source position 1.
        let f = TempFile::new("projected");
        write(&f.0, 0..300, 100);
        let tag = BoundExpr::column(0, "tag", DataType::Utf8);
        let expr = BoundExpr::binary(
            tag,
            BinaryOp::Eq,
            BoundExpr::Literal(Value::Utf8("t150".into())),
        )
        .unwrap();
        let mut op = ScanExec::from_file(
            &f.0,
            vec![1],
            vec![expr],
            Arc::new(schema().project(&[1]).unwrap()),
        )
        .unwrap();
        let out = collect_one(&mut op).unwrap();
        assert_eq!(out.num_rows(), 1);
        assert_eq!(out.num_columns(), 1);
    }

    #[test]
    fn a_row_group_larger_than_a_batch_is_split() {
        let f = TempFile::new("chunked");
        write(&f.0, 0..20_000, 20_000);
        let mut op = file_scan(&f.0, vec![]);
        let mut batches = 0;
        let mut rows = 0;
        while let Some(b) = op.next_batch().unwrap() {
            batches += 1;
            rows += b.num_rows();
        }
        assert_eq!(rows, 20_000);
        assert!(batches > 1, "one row group came back as a single batch");
        assert_eq!(op.metrics().row_groups_read, 1);
    }

    #[test]
    fn a_missing_file_is_reported_rather_than_panicking() {
        assert!(ScanExec::from_file("/nonexistent/nope.qfc", vec![0], vec![], schema()).is_err());
    }
}
