//! Sorting, in three modes depending on what the query asks for.
//!
//! * **Top-k** — when the sort feeds a `LIMIT`, only the best `k` rows are ever
//!   held. Memory is bounded by `k` no matter how large the input, which is the
//!   difference between `ORDER BY ts DESC LIMIT 10` being instant and being
//!   impossible.
//! * **In memory** — when everything fits, sort once and emit.
//! * **External** — when it does not, sorted runs are written to spill files
//!   and merged. This is the path that lets a sort finish on an input larger
//!   than memory, which is the only reason a database sorts differently from
//!   `Vec::sort`.

use crate::eval::evaluate;
use crate::operator::{Metrics, Operator};
use qf_common::{Result, Schema, Value};
use qf_plan::logical::SortExpr;
use qf_storage::batch::{RecordBatch, DEFAULT_BATCH_SIZE};
use qf_storage::format::{write_file, QfcReader};
use std::cmp::Ordering;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Instant;

/// Rows held before spilling. Deliberately low so the external path is
/// exercised by real queries and by the tests, not just in theory.
pub const DEFAULT_SPILL_THRESHOLD: usize = 100_000;

static SPILL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A sorted run on disk. Removes its file when dropped, including when a query
/// fails partway through.
struct SpillFile {
    path: PathBuf,
}

impl SpillFile {
    fn create(schema: Arc<Schema>, batch: &RecordBatch) -> Result<SpillFile> {
        let n = SPILL_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!("queryforge-sort-{}-{n}.qfc", std::process::id()));
        write_file(
            &path,
            schema,
            std::slice::from_ref(batch),
            DEFAULT_BATCH_SIZE,
        )?;
        Ok(SpillFile { path })
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Reads one spill run in order, a batch at a time.
struct RunCursor {
    reader: QfcReader,
    columns: Vec<usize>,
    group: usize,
    batch: Option<RecordBatch>,
    keys: Vec<Vec<Value>>,
    row: usize,
    _file: SpillFile,
}

impl RunCursor {
    fn open(file: SpillFile, exprs: &[SortExpr]) -> Result<RunCursor> {
        let reader = QfcReader::open(&file.path)?;
        let columns = (0..reader.schema().len()).collect();
        let mut c = RunCursor {
            reader,
            columns,
            group: 0,
            batch: None,
            keys: Vec::new(),
            row: 0,
            _file: file,
        };
        c.advance_group(exprs)?;
        Ok(c)
    }

    fn advance_group(&mut self, exprs: &[SortExpr]) -> Result<()> {
        if self.group >= self.reader.num_row_groups() {
            self.batch = None;
            return Ok(());
        }
        let batch = self.reader.read_row_group(self.group, &self.columns)?;
        self.group += 1;
        self.keys = sort_keys(exprs, &batch)?;
        self.row = 0;
        self.batch = Some(batch);
        Ok(())
    }

    fn current(&self) -> Option<(&RecordBatch, usize)> {
        self.batch.as_ref().map(|b| (b, self.row))
    }

    fn key(&self) -> Option<&Vec<Value>> {
        self.keys.get(self.row)
    }

    fn step(&mut self, exprs: &[SortExpr]) -> Result<()> {
        self.row += 1;
        if let Some(b) = &self.batch {
            if self.row >= b.num_rows() {
                self.advance_group(exprs)?;
            }
        }
        Ok(())
    }
}

/// Materialises the sort key of every row.
fn sort_keys(exprs: &[SortExpr], batch: &RecordBatch) -> Result<Vec<Vec<Value>>> {
    let columns = exprs
        .iter()
        .map(|s| evaluate(&s.expr, batch))
        .collect::<Result<Vec<_>>>()?;
    Ok((0..batch.num_rows())
        .map(|r| columns.iter().map(|c| c.value(r)).collect())
        .collect())
}

/// Compares two keys under the sort specification.
///
/// NULL ordering is explicit rather than falling out of the value ordering,
/// because SQL's default differs by direction: ascending puts NULLs last,
/// descending puts them first.
pub fn compare_keys(a: &[Value], b: &[Value], exprs: &[SortExpr]) -> Ordering {
    for ((x, y), spec) in a.iter().zip(b.iter()).zip(exprs.iter()) {
        let ord = match (x.is_null(), y.is_null()) {
            (true, true) => Ordering::Equal,
            (true, false) => {
                if spec.nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (false, true) => {
                if spec.nulls_first {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            (false, false) => {
                let o = x.total_cmp(y);
                if spec.ascending {
                    o
                } else {
                    o.reverse()
                }
            }
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

pub struct SortExec {
    input: Box<dyn Operator>,
    exprs: Vec<SortExpr>,
    schema: Arc<Schema>,
    /// Rows the consumer will actually take, when known. Enables top-k.
    fetch: Option<usize>,
    spill_threshold: usize,

    buffered: Vec<RecordBatch>,
    buffered_rows: usize,
    runs: Vec<SpillFile>,
    /// Paths of every run this sort wrote, kept after the runs themselves are
    /// consumed so a caller can confirm they were cleaned up.
    spill_paths: Vec<PathBuf>,

    output: Vec<RecordBatch>,
    next: usize,
    done: bool,
    metrics: Metrics,
}

impl SortExec {
    pub fn new(input: Box<dyn Operator>, exprs: Vec<SortExpr>, fetch: Option<usize>) -> SortExec {
        SortExec::with_spill_threshold(input, exprs, fetch, DEFAULT_SPILL_THRESHOLD)
    }

    /// Every spill file this sort wrote, whether or not it still exists.
    pub fn spill_paths(&self) -> &[PathBuf] {
        &self.spill_paths
    }

    pub fn with_spill_threshold(
        input: Box<dyn Operator>,
        exprs: Vec<SortExpr>,
        fetch: Option<usize>,
        spill_threshold: usize,
    ) -> SortExec {
        let schema = input.schema();
        SortExec {
            input,
            exprs,
            schema,
            fetch,
            spill_threshold: spill_threshold.max(1),
            buffered: Vec::new(),
            buffered_rows: 0,
            runs: Vec::new(),
            spill_paths: Vec::new(),
            output: Vec::new(),
            next: 0,
            done: false,
            metrics: Metrics::default(),
        }
    }

    fn sort_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let keys = sort_keys(&self.exprs, batch)?;
        let mut order: Vec<usize> = (0..batch.num_rows()).collect();
        // A stable sort keeps rows that tie in the order they arrived, which
        // makes results reproducible run to run.
        order.sort_by(|a, b| compare_keys(&keys[*a], &keys[*b], &self.exprs));
        batch.take(&order)
    }

    fn drain_buffer(&mut self) -> Result<RecordBatch> {
        let combined = RecordBatch::concat(Arc::clone(&self.schema), &self.buffered)?;
        self.buffered.clear();
        self.buffered_rows = 0;
        self.sort_batch(&combined)
    }

    fn run(&mut self) -> Result<()> {
        while let Some(b) = self.input.next_batch()? {
            if b.num_rows() == 0 {
                continue;
            }
            self.buffered_rows += b.num_rows();
            self.buffered.push(b);

            match self.fetch {
                // Top-k: trim back to the best `k` whenever the buffer grows
                // past twice that, so memory stays O(k).
                Some(k) if self.buffered_rows > (k.max(1)).saturating_mul(2) => {
                    let sorted = self.drain_buffer()?;
                    let keep: Vec<usize> = (0..sorted.num_rows().min(k)).collect();
                    let trimmed = sorted.take(&keep)?;
                    self.buffered_rows = trimmed.num_rows();
                    self.buffered.push(trimmed);
                }
                None if self.buffered_rows >= self.spill_threshold => {
                    let sorted = self.drain_buffer()?;
                    self.metrics.spilled_rows += sorted.num_rows();
                    self.metrics.spilled_runs += 1;
                    let file = SpillFile::create(Arc::clone(&self.schema), &sorted)?;
                    self.spill_paths.push(file.path.clone());
                    self.runs.push(file);
                }
                _ => {}
            }
        }

        if self.runs.is_empty() {
            let sorted = if self.buffered.is_empty() {
                RecordBatch::empty(Arc::clone(&self.schema))?
            } else {
                self.drain_buffer()?
            };
            let sorted = match self.fetch {
                Some(k) if sorted.num_rows() > k => sorted.take(&(0..k).collect::<Vec<_>>())?,
                _ => sorted,
            };
            self.output = sorted.chunks(DEFAULT_BATCH_SIZE)?;
            return Ok(());
        }

        // Whatever is still buffered becomes one last run, so the merge has a
        // uniform set of inputs.
        if !self.buffered.is_empty() {
            let sorted = self.drain_buffer()?;
            self.metrics.spilled_rows += sorted.num_rows();
            self.metrics.spilled_runs += 1;
            let file = SpillFile::create(Arc::clone(&self.schema), &sorted)?;
            self.spill_paths.push(file.path.clone());
            self.runs.push(file);
        }
        self.merge()
    }

    /// Merges the sorted runs.
    ///
    /// The smallest head is found by scanning the cursors rather than with a
    /// heap. Runs are few — one per spill threshold's worth of rows — so the
    /// linear scan costs less than maintaining the heap would, and it is far
    /// easier to be sure it is correct.
    fn merge(&mut self) -> Result<()> {
        let files: Vec<SpillFile> = std::mem::take(&mut self.runs);
        let mut cursors = files
            .into_iter()
            .map(|f| RunCursor::open(f, &self.exprs))
            .collect::<Result<Vec<_>>>()?;

        let mut emitted = 0usize;
        let mut chunk: Vec<(usize, usize)> = Vec::new(); // (cursor, row)
        let mut sources: Vec<RecordBatch> = Vec::new();
        let mut out = Vec::new();

        loop {
            if self.fetch.is_some_and(|k| emitted >= k) {
                break;
            }
            let mut best: Option<usize> = None;
            for (i, c) in cursors.iter().enumerate() {
                let Some(k) = c.key() else { continue };
                match best {
                    None => best = Some(i),
                    Some(b) => {
                        let bk = cursors[b].key().expect("a cursor with a key");
                        if compare_keys(k, bk, &self.exprs) == Ordering::Less {
                            best = Some(i);
                        }
                    }
                }
            }
            let Some(i) = best else { break };

            let (batch, row) = cursors[i].current().expect("a cursor with a row");
            // Rows are copied out one at a time; keeping the source batch
            // alongside lets the gather below stay a single take per run.
            if sources.len() <= i {
                sources.resize(i + 1, batch.clone());
            }
            sources[i] = batch.clone();
            chunk.push((i, row));
            emitted += 1;
            cursors[i].step(&self.exprs)?;

            if chunk.len() >= DEFAULT_BATCH_SIZE {
                out.push(assemble(&self.schema, &sources, &chunk)?);
                chunk.clear();
            }
        }
        if !chunk.is_empty() {
            out.push(assemble(&self.schema, &sources, &chunk)?);
        }
        self.output = out;
        Ok(())
    }
}

/// Builds one output batch from rows picked out of several run batches.
fn assemble(
    schema: &Arc<Schema>,
    sources: &[RecordBatch],
    rows: &[(usize, usize)],
) -> Result<RecordBatch> {
    let mut builders: Vec<qf_storage::array::ArrayBuilder> = schema
        .fields()
        .iter()
        .map(|f| qf_storage::array::ArrayBuilder::new(f.data_type))
        .collect();
    for (cursor, row) in rows {
        let batch = &sources[*cursor];
        for (c, b) in builders.iter_mut().enumerate() {
            b.push_from(batch.column(c)?, *row);
        }
    }
    let columns = builders
        .into_iter()
        .map(qf_storage::array::ArrayBuilder::finish)
        .collect::<Result<Vec<_>>>()?;
    if columns.is_empty() {
        return RecordBatch::empty(Arc::clone(schema))?.take(&vec![0usize; rows.len()]);
    }
    RecordBatch::try_new(Arc::clone(schema), columns)
}

impl Operator for SortExec {
    fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let start = Instant::now();
        if !self.done {
            self.run()?;
            self.done = true;
        }
        if self.next >= self.output.len() {
            self.metrics.elapsed += start.elapsed();
            return Ok(None);
        }
        let b = self.output[self.next].clone();
        self.next += 1;
        self.metrics.elapsed += start.elapsed();
        self.metrics.record(&b);
        Ok(Some(b))
    }

    fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn children(&self) -> Vec<&dyn Operator> {
        vec![self.input.as_ref()]
    }

    fn name(&self) -> &'static str {
        "Sort"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::collect_one;
    use crate::operators::ValuesExec;
    use qf_common::{DataType, Field};
    use qf_plan::expr::BoundExpr;
    use qf_storage::array::Array;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("tag", DataType::Utf8, false),
        ]))
    }

    fn source(keys: &[Option<i64>], batch_size: usize) -> Box<dyn Operator> {
        let ks: Vec<Value> = keys
            .iter()
            .map(|k| k.map_or(Value::Null, Value::Int64))
            .collect();
        let tags: Vec<Value> = keys
            .iter()
            .enumerate()
            .map(|(i, _)| Value::Utf8(format!("r{i}")))
            .collect();
        let whole = RecordBatch::try_new(
            schema(),
            vec![
                Array::from_values(DataType::Int64, &ks).unwrap(),
                Array::from_values(DataType::Utf8, &tags).unwrap(),
            ],
        )
        .unwrap();
        let batches = whole.chunks(batch_size.max(1)).unwrap();
        Box::new(ValuesExec::new(schema(), batches))
    }

    fn spec(ascending: bool, nulls_first: bool) -> Vec<SortExpr> {
        vec![SortExpr {
            expr: BoundExpr::column(0, "k", DataType::Int64),
            ascending,
            nulls_first,
        }]
    }

    fn keys_of(batch: &RecordBatch) -> Vec<String> {
        batch
            .column(0)
            .unwrap()
            .iter()
            .map(|v| v.to_string())
            .collect()
    }

    #[test]
    fn a_small_input_sorts_in_memory_without_spilling() {
        let mut op = SortExec::new(
            source(&[Some(3), Some(1), Some(2)], 8),
            spec(true, false),
            None,
        );
        let out = collect_one(&mut op).unwrap();
        assert_eq!(keys_of(&out), vec!["1", "2", "3"]);
        assert_eq!(op.metrics().spilled_runs, 0);
    }

    #[test]
    fn nulls_go_last_ascending_and_first_descending_by_default() {
        let data = [Some(2), None, Some(1)];
        let mut asc = SortExec::new(source(&data, 8), spec(true, false), None);
        assert_eq!(
            keys_of(&collect_one(&mut asc).unwrap()),
            vec!["1", "2", "NULL"]
        );

        let mut desc = SortExec::new(source(&data, 8), spec(false, true), None);
        assert_eq!(
            keys_of(&collect_one(&mut desc).unwrap()),
            vec!["NULL", "2", "1"]
        );
    }

    #[test]
    fn an_input_larger_than_the_spill_threshold_still_comes_back_fully_sorted() {
        // The whole point of the external path: the answer must not depend on
        // whether the input fitted in memory.
        let data: Vec<Option<i64>> = (0..500).map(|i| Some((i * 37) % 500)).collect();
        let mut spilled =
            SortExec::with_spill_threshold(source(&data, 32), spec(true, false), None, 50);
        let out = collect_one(&mut spilled).unwrap();
        let keys = keys_of(&out);
        assert_eq!(keys.len(), 500);
        assert!(spilled.metrics().spilled_runs > 1, "nothing spilled");
        assert_eq!(spilled.metrics().spilled_rows, 500);

        let expected: Vec<String> = (0..500).map(|i| i.to_string()).collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn spilling_and_not_spilling_produce_the_same_rows_in_the_same_order() {
        let data: Vec<Option<i64>> = (0..200)
            .map(|i| {
                if i % 17 == 0 {
                    None
                } else {
                    Some((i * 13) % 200)
                }
            })
            .collect();
        let mut memory = SortExec::new(source(&data, 16), spec(true, false), None);
        let mut disk =
            SortExec::with_spill_threshold(source(&data, 16), spec(true, false), None, 25);
        let a = collect_one(&mut memory).unwrap();
        let b = collect_one(&mut disk).unwrap();
        assert!(disk.metrics().spilled_runs > 0);
        assert_eq!(a, b);
    }

    #[test]
    fn spill_files_are_removed_once_the_merge_has_consumed_them() {
        let data: Vec<Option<i64>> = (0..300).map(Some).collect();
        let mut op = SortExec::with_spill_threshold(source(&data, 16), spec(true, false), None, 40);
        collect_one(&mut op).unwrap();
        assert!(op.metrics().spilled_runs > 0);
        let paths: Vec<PathBuf> = op.spill_paths().to_vec();
        assert!(!paths.is_empty());
        // The merge takes ownership of each run and drops it as it finishes,
        // so by the time the output is drained nothing is left on disk.
        for p in &paths {
            assert!(!p.exists(), "{} was left behind", p.display());
        }
    }

    #[test]
    fn spill_files_are_removed_even_when_the_sort_is_abandoned_mid_query() {
        let data: Vec<Option<i64>> = (0..300).map(Some).collect();
        let paths;
        {
            let mut op =
                SortExec::with_spill_threshold(source(&data, 16), spec(true, false), None, 40);
            // Pull one batch, then drop the operator without draining it.
            op.next_batch().unwrap();
            paths = op.spill_paths().to_vec();
            assert!(!paths.is_empty());
        }
        for p in &paths {
            assert!(!p.exists(), "{} survived the drop", p.display());
        }
    }

    #[test]
    fn a_top_k_sort_returns_the_right_rows_without_holding_the_input() {
        let data: Vec<Option<i64>> = (0..1000).map(|i| Some((i * 91) % 1000)).collect();
        let mut op = SortExec::new(source(&data, 64), spec(true, false), Some(5));
        let out = collect_one(&mut op).unwrap();
        assert_eq!(keys_of(&out), vec!["0", "1", "2", "3", "4"]);
        // Top-k never spills, however large the input.
        assert_eq!(op.metrics().spilled_runs, 0);
    }

    #[test]
    fn top_k_agrees_with_a_full_sort_then_truncate() {
        let data: Vec<Option<i64>> = (0..300)
            .map(|i| {
                if i % 23 == 0 {
                    None
                } else {
                    Some((i * 7) % 300)
                }
            })
            .collect();
        let mut full = SortExec::new(source(&data, 32), spec(false, true), None);
        let all = collect_one(&mut full).unwrap();
        let expected = all.take(&(0..10).collect::<Vec<_>>()).unwrap();

        let mut top = SortExec::new(source(&data, 32), spec(false, true), Some(10));
        assert_eq!(collect_one(&mut top).unwrap(), expected);
    }

    #[test]
    fn a_top_k_larger_than_the_input_returns_everything() {
        let mut op = SortExec::new(source(&[Some(2), Some(1)], 8), spec(true, false), Some(50));
        assert_eq!(keys_of(&collect_one(&mut op).unwrap()), vec!["1", "2"]);
    }

    #[test]
    fn an_empty_input_sorts_to_nothing() {
        let mut op = SortExec::new(source(&[], 8), spec(true, false), None);
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 0);
        assert_eq!(op.name(), "Sort");
        assert_eq!(op.children().len(), 1);
        assert_eq!(op.schema().len(), 2);
    }

    #[test]
    fn ties_keep_their_input_order() {
        // Every key is 1, so a stable sort must leave the tags as they came.
        let data = vec![Some(1); 6];
        let mut op = SortExec::new(source(&data, 2), spec(true, false), None);
        let out = collect_one(&mut op).unwrap();
        let tags: Vec<String> = out
            .column(1)
            .unwrap()
            .iter()
            .map(|v| v.to_string())
            .collect();
        assert_eq!(tags, vec!["r0", "r1", "r2", "r3", "r4", "r5"]);
    }

    #[test]
    fn comparing_keys_follows_the_specification_field_by_field() {
        let asc = spec(true, false);
        assert_eq!(
            compare_keys(&[Value::Int64(1)], &[Value::Int64(2)], &asc),
            Ordering::Less
        );
        let desc = spec(false, false);
        assert_eq!(
            compare_keys(&[Value::Int64(1)], &[Value::Int64(2)], &desc),
            Ordering::Greater
        );
        assert_eq!(
            compare_keys(&[Value::Null], &[Value::Null], &asc),
            Ordering::Equal
        );
        assert_eq!(
            compare_keys(&[Value::Null], &[Value::Int64(1)], &asc),
            Ordering::Greater,
            "nulls last"
        );
        assert_eq!(
            compare_keys(&[Value::Int64(1)], &[Value::Null], &spec(true, true)),
            Ordering::Greater,
            "nulls first"
        );
        assert_eq!(compare_keys(&[], &[], &asc), Ordering::Equal);
    }
}
