//! The operator interface and the counters `EXPLAIN ANALYZE` reports.
//!
//! Operators pull: each asks its input for the next batch and returns a batch
//! of its own, or `None` when it is finished. Pulling is what lets `LIMIT`
//! stop a scan early — a push model would have to run the scan to completion
//! and throw the extra rows away.
//!
//! Some operators cannot stream. A sort has to see every row before it can
//! emit the first one, and a hash join has to finish building before it can
//! probe. Those buffer internally and stream their *output*, so the operator
//! above them never has to know the difference.

use qf_common::{Result, Schema};
use qf_storage::batch::RecordBatch;
use std::sync::Arc;
use std::time::Duration;

/// What one operator did, for `EXPLAIN ANALYZE`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metrics {
    pub rows_out: usize,
    pub batches_out: usize,
    pub elapsed: Duration,
    /// Row groups actually read. Scans only.
    pub row_groups_read: usize,
    /// Row groups skipped because a zone map ruled the predicate out. This is
    /// the number that shows whether pushdown did anything.
    pub row_groups_pruned: usize,
    /// Runs written to disk by an external sort.
    pub spilled_runs: usize,
    pub spilled_rows: usize,
}

impl Metrics {
    pub fn record(&mut self, batch: &RecordBatch) {
        self.rows_out += batch.num_rows();
        self.batches_out += 1;
    }

    /// The parenthesised suffix `EXPLAIN ANALYZE` appends to a plan line.
    pub fn describe(&self) -> String {
        let mut parts = vec![
            format!("rows={}", self.rows_out),
            format!("time={:.2}ms", self.elapsed.as_secs_f64() * 1000.0),
        ];
        if self.row_groups_read + self.row_groups_pruned > 0 {
            parts.push(format!(
                "row groups={} read, {} pruned",
                self.row_groups_read, self.row_groups_pruned
            ));
        }
        if self.spilled_runs > 0 {
            parts.push(format!(
                "spilled={} runs, {} rows",
                self.spilled_runs, self.spilled_rows
            ));
        }
        parts.join(", ")
    }
}

/// A node in the running query.
pub trait Operator {
    fn schema(&self) -> Arc<Schema>;

    /// The next batch, or `None` when the operator is exhausted. Returning an
    /// empty batch is allowed and means "nothing this time, ask again" — a
    /// filter that rejected everything in one batch is not finished.
    fn next_batch(&mut self) -> Result<Option<RecordBatch>>;

    fn metrics(&self) -> &Metrics;

    /// This operator's children, so the driver can walk the tree for
    /// `EXPLAIN ANALYZE`.
    fn children(&self) -> Vec<&dyn Operator> {
        vec![]
    }

    /// A short label for `EXPLAIN ANALYZE`.
    fn name(&self) -> &'static str;
}

/// Drains an operator into batches. Used by the CLI and by the tests; the
/// operators themselves never call it.
pub fn collect(op: &mut dyn Operator) -> Result<Vec<RecordBatch>> {
    let mut out = Vec::new();
    while let Some(b) = op.next_batch()? {
        if b.num_rows() > 0 {
            out.push(b);
        }
    }
    Ok(out)
}

/// Drains an operator into a single batch.
pub fn collect_one(op: &mut dyn Operator) -> Result<RecordBatch> {
    let schema = op.schema();
    let batches = collect(op)?;
    RecordBatch::concat(schema, &batches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qf_common::{DataType, Field, Value};
    use qf_storage::array::Array;

    struct Fixed {
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
        metrics: Metrics,
    }

    impl Operator for Fixed {
        fn schema(&self) -> Arc<Schema> {
            Arc::clone(&self.schema)
        }
        fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
            if self.batches.is_empty() {
                return Ok(None);
            }
            let b = self.batches.remove(0);
            self.metrics.record(&b);
            Ok(Some(b))
        }
        fn metrics(&self) -> &Metrics {
            &self.metrics
        }
        fn name(&self) -> &'static str {
            "Fixed"
        }
    }

    fn fixed(values: &[&[i64]]) -> Fixed {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        let batches = values
            .iter()
            .map(|vs| {
                let vals: Vec<Value> = vs.iter().map(|v| Value::Int64(*v)).collect();
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Array::from_values(DataType::Int64, &vals).unwrap()],
                )
                .unwrap()
            })
            .collect();
        Fixed {
            schema,
            batches,
            metrics: Metrics::default(),
        }
    }

    #[test]
    fn collecting_drains_every_batch_and_drops_empty_ones() {
        let mut op = fixed(&[&[1, 2], &[], &[3]]);
        let batches = collect(&mut op).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(op.metrics().rows_out, 3);
        assert_eq!(op.metrics().batches_out, 3);
    }

    #[test]
    fn collecting_into_one_batch_concatenates_in_order() {
        let mut op = fixed(&[&[1, 2], &[3]]);
        let b = collect_one(&mut op).unwrap();
        assert_eq!(b.num_rows(), 3);
        assert_eq!(b.column(0).unwrap().value(2), Value::Int64(3));
    }

    #[test]
    fn an_exhausted_operator_keeps_returning_none() {
        let mut op = fixed(&[&[1]]);
        assert!(op.next_batch().unwrap().is_some());
        assert!(op.next_batch().unwrap().is_none());
        assert!(op.next_batch().unwrap().is_none());
        assert!(op.children().is_empty());
        assert_eq!(op.name(), "Fixed");
    }

    #[test]
    fn metrics_render_only_the_counters_that_apply() {
        let plain = Metrics {
            rows_out: 10,
            batches_out: 1,
            ..Metrics::default()
        };
        let text = plain.describe();
        assert!(text.contains("rows=10"));
        assert!(text.contains("time="));
        assert!(!text.contains("row groups"));
        assert!(!text.contains("spilled"));

        let scan = Metrics {
            rows_out: 5,
            row_groups_read: 2,
            row_groups_pruned: 8,
            ..Metrics::default()
        };
        assert!(scan.describe().contains("2 read, 8 pruned"));

        let sort = Metrics {
            spilled_runs: 3,
            spilled_rows: 300,
            ..Metrics::default()
        };
        assert!(sort.describe().contains("3 runs, 300 rows"));
    }
}
