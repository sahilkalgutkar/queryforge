//! Hash join.
//!
//! One side is collected and hashed; the other streams past it. Which side is
//! built is a real decision — the build side is what has to fit in memory —
//! and for an inner join the physical planner picks the smaller estimate. For
//! an outer join the choice is forced: the side that may be NULL-padded has to
//! be the one being built, so that unmatched rows can be found at the end.
//!
//! Output column order is always left-then-right, whichever side was built.

use crate::eval::evaluate;
use crate::operator::{collect_one, Metrics, Operator};
use qf_common::{Result, Schema, Value};
use qf_plan::expr::BoundExpr;
use qf_sql::ast::JoinType;
use qf_storage::array::Array;
use qf_storage::batch::{RecordBatch, DEFAULT_BATCH_SIZE};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Which input is loaded into the hash table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildSide {
    Left,
    Right,
}

pub struct HashJoinExec {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    join_type: JoinType,
    on: Vec<(BoundExpr, BoundExpr)>,
    filter: Option<BoundExpr>,
    build_side: BuildSide,
    schema: Arc<Schema>,

    built: Option<BuiltSide>,
    /// Which build rows found a partner, for the outer-join tail.
    matched: Vec<bool>,
    tail_emitted: bool,
    pending: Vec<RecordBatch>,
    metrics: Metrics,
}

struct BuiltSide {
    batch: RecordBatch,
    /// Key to the rows holding it. A vector because keys repeat.
    index: HashMap<Vec<Value>, Vec<usize>>,
}

impl HashJoinExec {
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        join_type: JoinType,
        on: Vec<(BoundExpr, BoundExpr)>,
        filter: Option<BoundExpr>,
        build_side: BuildSide,
    ) -> HashJoinExec {
        // An outer join has to be able to find the unmatched rows of the side
        // it preserves the *other* way round, so the choice is not free.
        let build_side = match join_type {
            JoinType::Left => BuildSide::Right,
            JoinType::Right => BuildSide::Left,
            JoinType::Full => BuildSide::Right,
            _ => build_side,
        };
        let schema = Arc::new(Schema::join(&left.schema(), &right.schema()));
        HashJoinExec {
            left,
            right,
            join_type,
            on,
            filter,
            build_side,
            schema,
            built: None,
            matched: Vec::new(),
            tail_emitted: false,
            pending: Vec::new(),
            metrics: Metrics::default(),
        }
    }

    pub fn build_side(&self) -> BuildSide {
        self.build_side
    }

    fn build(&mut self) -> Result<()> {
        if self.built.is_some() {
            return Ok(());
        }
        let batch = match self.build_side {
            BuildSide::Left => collect_one(self.left.as_mut())?,
            BuildSide::Right => collect_one(self.right.as_mut())?,
        };
        let keys: Vec<&BoundExpr> = self
            .on
            .iter()
            .map(|(l, r)| match self.build_side {
                BuildSide::Left => l,
                BuildSide::Right => r,
            })
            .collect();

        let mut index: HashMap<Vec<Value>, Vec<usize>> = HashMap::new();
        if !keys.is_empty() {
            let columns = keys
                .iter()
                .map(|e| evaluate(e, &batch))
                .collect::<Result<Vec<_>>>()?;
            for row in 0..batch.num_rows() {
                let key: Vec<Value> = columns.iter().map(|c| c.value(row)).collect();
                // A NULL key never joins — `NULL = NULL` is unknown, not true.
                if key.iter().any(Value::is_null) {
                    continue;
                }
                index.entry(key).or_default().push(row);
            }
        }
        self.matched = vec![false; batch.num_rows()];
        self.built = Some(BuiltSide { batch, index });
        Ok(())
    }

    fn probe_operator(&mut self) -> &mut dyn Operator {
        match self.build_side {
            BuildSide::Left => self.right.as_mut(),
            BuildSide::Right => self.left.as_mut(),
        }
    }

    /// Does the probe side hold rows that must survive without a match?
    fn probe_is_preserved(&self) -> bool {
        match self.join_type {
            JoinType::Left => self.build_side == BuildSide::Right,
            JoinType::Right => self.build_side == BuildSide::Left,
            JoinType::Full => true,
            _ => false,
        }
    }

    fn build_is_preserved(&self) -> bool {
        match self.join_type {
            JoinType::Left => self.build_side == BuildSide::Left,
            JoinType::Right => self.build_side == BuildSide::Right,
            JoinType::Full => true,
            _ => false,
        }
    }

    fn probe(&mut self, probe_batch: &RecordBatch) -> Result<RecordBatch> {
        // Candidate pairs first, with no padding. The ON clause decides what
        // counts as a match, and its non-equality part is as much a part of
        // that decision as the keys are — so it has to be applied *before*
        // asking whether a probe row matched anything.
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        {
            let built = self.built.as_ref().expect("built before probing");
            if self.on.is_empty() {
                for p in 0..probe_batch.num_rows() {
                    for b in 0..built.batch.num_rows() {
                        candidates.push((b, p));
                    }
                }
            } else {
                let probe_keys: Vec<&BoundExpr> = self
                    .on
                    .iter()
                    .map(|(l, r)| match self.build_side {
                        BuildSide::Left => r,
                        BuildSide::Right => l,
                    })
                    .collect();
                let columns = probe_keys
                    .iter()
                    .map(|e| evaluate(e, probe_batch))
                    .collect::<Result<Vec<_>>>()?;
                for p in 0..probe_batch.num_rows() {
                    let key: Vec<Value> = columns.iter().map(|c| c.value(p)).collect();
                    // A NULL key never joins: `NULL = NULL` is unknown.
                    if key.iter().any(Value::is_null) {
                        continue;
                    }
                    if let Some(rows) = built.index.get(&key) {
                        for b in rows {
                            candidates.push((*b, p));
                        }
                    }
                }
            }
        }

        // Apply the ON clause's non-equality part to the candidates.
        if let Some(filter) = self.filter.clone().filter(|_| !candidates.is_empty()) {
            let build_rows: Vec<Option<usize>> = candidates.iter().map(|(b, _)| Some(*b)).collect();
            let probe_rows: Vec<Option<usize>> = candidates.iter().map(|(_, p)| Some(*p)).collect();
            let assembled = self.assemble(&build_rows, &probe_rows, probe_batch)?;
            let mask = crate::eval::evaluate_predicate(&filter, &assembled)?;
            candidates = candidates
                .into_iter()
                .enumerate()
                .filter(|(i, _)| mask.get(*i))
                .map(|(_, c)| c)
                .collect();
        }

        // Only now is it settled which probe rows matched, so a preserved
        // probe row that had candidates but lost them all still comes back
        // padded rather than disappearing.
        let mut build_rows: Vec<Option<usize>> = Vec::with_capacity(candidates.len());
        let mut probe_rows: Vec<Option<usize>> = Vec::with_capacity(candidates.len());
        let mut probe_matched = vec![false; probe_batch.num_rows()];
        for (b, p) in &candidates {
            build_rows.push(Some(*b));
            probe_rows.push(Some(*p));
            probe_matched[*p] = true;
            if self.build_is_preserved() {
                self.matched[*b] = true;
            }
        }
        if self.probe_is_preserved() {
            for (p, matched) in probe_matched.iter().enumerate() {
                if !matched {
                    build_rows.push(None);
                    probe_rows.push(Some(p));
                }
            }
        }

        self.assemble(&build_rows, &probe_rows, probe_batch)
    }

    /// Gathers the chosen rows from both sides into one batch, always with the
    /// left input's columns first.
    fn assemble(
        &self,
        build_rows: &[Option<usize>],
        probe_rows: &[Option<usize>],
        probe_batch: &RecordBatch,
    ) -> Result<RecordBatch> {
        let built = self.built.as_ref().expect("built before assembling");
        let build_part = gather(&built.batch, build_rows)?;
        let probe_part = gather(probe_batch, probe_rows)?;
        match self.build_side {
            BuildSide::Left => RecordBatch::hstack(&build_part, &probe_part),
            BuildSide::Right => RecordBatch::hstack(&probe_part, &build_part),
        }
    }

    /// The build-side rows that never matched, padded with NULLs. Emitted once
    /// the probe side is exhausted.
    fn tail(&mut self) -> Result<Option<RecordBatch>> {
        if self.tail_emitted || !self.build_is_preserved() {
            return Ok(None);
        }
        self.tail_emitted = true;
        let built = self.built.as_ref().expect("built before the tail");
        let rows: Vec<Option<usize>> = (0..built.batch.num_rows())
            .filter(|i| !self.matched[*i])
            .map(Some)
            .collect();
        if rows.is_empty() {
            return Ok(None);
        }
        let build_part = gather(&built.batch, &rows)?;
        let other_schema = match self.build_side {
            BuildSide::Left => self.right.schema(),
            BuildSide::Right => self.left.schema(),
        };
        let nulls = null_batch(&other_schema, rows.len())?;
        let out = match self.build_side {
            BuildSide::Left => RecordBatch::hstack(&build_part, &nulls),
            BuildSide::Right => RecordBatch::hstack(&nulls, &build_part),
        }?;
        Ok(Some(out))
    }
}

/// Takes rows by index, turning `None` into a row of NULLs.
fn gather(batch: &RecordBatch, rows: &[Option<usize>]) -> Result<RecordBatch> {
    if rows.iter().all(Option::is_some) {
        let indices: Vec<usize> = rows.iter().map(|r| r.unwrap()).collect();
        return batch.take(&indices);
    }
    let schema = Arc::clone(batch.schema());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for c in 0..batch.num_columns() {
        let source = batch.column(c)?;
        let mut b = qf_storage::array::ArrayBuilder::new(source.data_type());
        b.reserve(rows.len());
        for r in rows {
            match r {
                Some(i) => b.push_from(source, *i),
                None => b.push_null(),
            }
        }
        columns.push(b.finish()?);
    }
    if columns.is_empty() {
        return RecordBatch::empty(schema)?.take(&vec![0usize; rows.len()]);
    }
    RecordBatch::try_new(schema, columns)
}

fn null_batch(schema: &Arc<Schema>, rows: usize) -> Result<RecordBatch> {
    let columns = schema
        .fields()
        .iter()
        .map(|f| Array::nulls(f.data_type, rows))
        .collect::<Result<Vec<_>>>()?;
    if columns.is_empty() {
        return RecordBatch::empty(Arc::clone(schema))?.take(&vec![0usize; rows]);
    }
    RecordBatch::try_new(Arc::clone(schema), columns)
}

impl Operator for HashJoinExec {
    fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let start = Instant::now();
        self.build()?;

        if let Some(b) = self.pending.pop() {
            self.metrics.elapsed += start.elapsed();
            self.metrics.record(&b);
            return Ok(Some(b));
        }

        while let Some(probe_batch) = self.probe_operator().next_batch()? {
            if probe_batch.num_rows() == 0 {
                continue;
            }
            let out = self.probe(&probe_batch)?;
            if out.num_rows() == 0 {
                continue;
            }
            // A join can multiply rows, so cut the result back to batch size.
            let mut chunks = out.chunks(DEFAULT_BATCH_SIZE)?;
            let first = chunks.remove(0);
            chunks.reverse();
            self.pending = chunks;
            self.metrics.elapsed += start.elapsed();
            self.metrics.record(&first);
            return Ok(Some(first));
        }

        let tail = self.tail()?;
        self.metrics.elapsed += start.elapsed();
        if let Some(b) = &tail {
            self.metrics.record(b);
        }
        Ok(tail)
    }

    fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn children(&self) -> Vec<&dyn Operator> {
        vec![self.left.as_ref(), self.right.as_ref()]
    }

    fn name(&self) -> &'static str {
        "HashJoin"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::collect_one;
    use crate::operators::ValuesExec;
    use qf_common::{DataType, Field};
    use qf_storage::array::Array;

    fn table(name: &str, ids: &[Option<i64>]) -> (Arc<Schema>, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(format!("{name}_id"), DataType::Int64, true),
            Field::new(format!("{name}_tag"), DataType::Utf8, false),
        ]));
        let id_values: Vec<Value> = ids
            .iter()
            .map(|i| i.map_or(Value::Null, Value::Int64))
            .collect();
        let tags: Vec<Value> = ids
            .iter()
            .enumerate()
            .map(|(i, _)| Value::Utf8(format!("{name}{i}")))
            .collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Array::from_values(DataType::Int64, &id_values).unwrap(),
                Array::from_values(DataType::Utf8, &tags).unwrap(),
            ],
        )
        .unwrap();
        (schema, batch)
    }

    fn operator(schema: Arc<Schema>, batch: RecordBatch) -> Box<dyn Operator> {
        Box::new(ValuesExec::new(schema, vec![batch]))
    }

    fn key(index: usize) -> BoundExpr {
        BoundExpr::column(index, "id", DataType::Int64)
    }

    fn run(join_type: JoinType, build_side: BuildSide, on: bool) -> RecordBatch {
        let (ls, lb) = table("l", &[Some(1), Some(2), None]);
        let (rs, rb) = table("r", &[Some(2), Some(3)]);
        let mut op = HashJoinExec::new(
            operator(ls, lb),
            operator(rs, rb),
            join_type,
            if on { vec![(key(0), key(0))] } else { vec![] },
            None,
            build_side,
        );
        collect_one(&mut op).unwrap()
    }

    fn pairs(batch: &RecordBatch) -> Vec<(String, String)> {
        batch
            .rows()
            .map(|r| (r[1].to_string(), r[3].to_string()))
            .collect()
    }

    #[test]
    fn an_inner_join_gives_the_same_answer_whichever_side_is_built() {
        let a = run(JoinType::Inner, BuildSide::Left, true);
        let b = run(JoinType::Inner, BuildSide::Right, true);
        assert_eq!(pairs(&a), vec![("l1".into(), "r0".into())]);
        assert_eq!(pairs(&a), pairs(&b));
        // Column order is left-then-right regardless of the build side.
        assert_eq!(a.schema().field(0).unwrap().name, "l_id");
        assert_eq!(b.schema().field(0).unwrap().name, "l_id");
    }

    #[test]
    fn a_null_key_row_never_matches() {
        let out = run(JoinType::Inner, BuildSide::Left, true);
        assert!(!pairs(&out).iter().any(|(l, _)| l == "l2"));
    }

    #[test]
    fn an_outer_join_forces_the_build_side_regardless_of_what_was_asked_for() {
        // A LEFT join has to be able to pad its right side, so it builds right
        // even when the planner suggested otherwise.
        let (ls, lb) = table("l", &[Some(1)]);
        let (rs, rb) = table("r", &[Some(1)]);
        let op = HashJoinExec::new(
            operator(ls, lb),
            operator(rs, rb),
            JoinType::Left,
            vec![(key(0), key(0))],
            None,
            BuildSide::Left,
        );
        assert_eq!(op.build_side(), BuildSide::Right);
    }

    #[test]
    fn a_left_join_pads_unmatched_left_rows() {
        let out = run(JoinType::Left, BuildSide::Right, true);
        let p = pairs(&out);
        assert_eq!(p.len(), 3);
        assert!(p.contains(&("l0".into(), "NULL".into())));
        assert!(p.contains(&("l2".into(), "NULL".into())));
    }

    #[test]
    fn a_right_join_pads_unmatched_right_rows() {
        let out = run(JoinType::Right, BuildSide::Left, true);
        let p = pairs(&out);
        assert_eq!(p.len(), 2);
        assert!(p.contains(&("NULL".into(), "r1".into())));
    }

    #[test]
    fn a_full_join_pads_both_sides() {
        let out = run(JoinType::Full, BuildSide::Right, true);
        let p = pairs(&out);
        assert_eq!(p.len(), 4);
        assert!(p.contains(&("l0".into(), "NULL".into())));
        assert!(p.contains(&("NULL".into(), "r1".into())));
        assert!(p.contains(&("l1".into(), "r0".into())));
    }

    #[test]
    fn a_cross_join_pairs_every_row_with_every_other() {
        let out = run(JoinType::Cross, BuildSide::Left, false);
        assert_eq!(out.num_rows(), 6);
        assert_eq!(out.num_columns(), 4);
    }

    #[test]
    fn a_residual_filter_removes_rows_after_matching() {
        let (ls, lb) = table("l", &[Some(1), Some(2)]);
        let (rs, rb) = table("r", &[Some(1), Some(2)]);
        let filter = BoundExpr::binary(
            BoundExpr::column(0, "l_id", DataType::Int64),
            qf_sql::ast::BinaryOp::Gt,
            BoundExpr::Literal(Value::Int64(1)),
        )
        .unwrap();
        let mut op = HashJoinExec::new(
            operator(ls, lb),
            operator(rs, rb),
            JoinType::Inner,
            vec![(key(0), key(0))],
            Some(filter),
            BuildSide::Right,
        );
        let out = collect_one(&mut op).unwrap();
        assert_eq!(out.num_rows(), 1);
        assert_eq!(out.row(0)[1], Value::Utf8("l1".into()));
    }

    #[test]
    fn a_left_join_whose_on_clause_rejects_a_match_pads_rather_than_drops() {
        // The bug this was written for: the ON clause's non-equality part was
        // applied *after* deciding what matched, so a preserved row whose only
        // candidate failed the condition vanished instead of coming back
        // NULL-padded — and so did rows that never had a candidate at all.
        let (ls, lb) = table("l", &[Some(1), Some(2)]);
        let (rs, rb) = table("r", &[Some(1)]);
        let never_true = BoundExpr::binary(
            BoundExpr::column(2, "r_id", DataType::Int64),
            qf_sql::ast::BinaryOp::Gt,
            BoundExpr::Literal(Value::Int64(1000)),
        )
        .unwrap();
        let mut op = HashJoinExec::new(
            operator(ls, lb),
            operator(rs, rb),
            JoinType::Left,
            vec![(key(0), key(0))],
            Some(never_true),
            BuildSide::Right,
        );
        let out = collect_one(&mut op).unwrap();
        assert_eq!(out.num_rows(), 2, "both left rows must survive");
        assert!(out.rows().all(|r| r[2].is_null()), "both must be padded");
    }

    #[test]
    fn a_full_join_with_a_rejecting_on_clause_preserves_both_sides() {
        let (ls, lb) = table("l", &[Some(1)]);
        let (rs, rb) = table("r", &[Some(1)]);
        let never_true = BoundExpr::Literal(Value::Boolean(false));
        let mut op = HashJoinExec::new(
            operator(ls, lb),
            operator(rs, rb),
            JoinType::Full,
            vec![(key(0), key(0))],
            Some(never_true),
            BuildSide::Right,
        );
        let out = collect_one(&mut op).unwrap();
        assert_eq!(out.num_rows(), 2);
        assert!(out.rows().any(|r| r[0].is_null()));
        assert!(out.rows().any(|r| r[2].is_null()));
    }

    #[test]
    fn a_join_reports_its_children_and_name() {
        let (ls, lb) = table("l", &[Some(1)]);
        let (rs, rb) = table("r", &[Some(1)]);
        let op = HashJoinExec::new(
            operator(ls, lb),
            operator(rs, rb),
            JoinType::Inner,
            vec![],
            None,
            BuildSide::Left,
        );
        assert_eq!(op.name(), "HashJoin");
        assert_eq!(op.children().len(), 2);
        assert_eq!(op.schema().len(), 4);
    }

    #[test]
    fn joining_against_an_empty_side_yields_nothing_or_padding() {
        let (ls, lb) = table("l", &[Some(1)]);
        let (rs, _) = table("r", &[]);
        let empty = RecordBatch::empty(Arc::clone(&rs)).unwrap();
        let mut inner = HashJoinExec::new(
            operator(Arc::clone(&ls), lb.clone()),
            operator(Arc::clone(&rs), empty.clone()),
            JoinType::Inner,
            vec![(key(0), key(0))],
            None,
            BuildSide::Right,
        );
        assert_eq!(collect_one(&mut inner).unwrap().num_rows(), 0);

        let mut left = HashJoinExec::new(
            operator(ls, lb),
            operator(rs, empty),
            JoinType::Left,
            vec![(key(0), key(0))],
            None,
            BuildSide::Right,
        );
        assert_eq!(collect_one(&mut left).unwrap().num_rows(), 1);
    }

    #[test]
    fn a_repeated_key_on_the_build_side_produces_a_row_per_match() {
        let (ls, lb) = table("l", &[Some(7)]);
        let (rs, rb) = table("r", &[Some(7), Some(7), Some(8)]);
        let mut op = HashJoinExec::new(
            operator(ls, lb),
            operator(rs, rb),
            JoinType::Inner,
            vec![(key(0), key(0))],
            None,
            BuildSide::Right,
        );
        assert_eq!(collect_one(&mut op).unwrap().num_rows(), 2);
    }
}
