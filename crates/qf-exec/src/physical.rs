//! Turning a logical plan into running operators.
//!
//! The logical plan says *what*; this decides *how*. Two choices are made here
//! rather than in the optimiser, because both depend on physical facts the
//! logical plan does not model:
//!
//! * **Which side of a join to build.** The build side is what has to fit in
//!   memory, so the smaller estimate wins — except on an outer join, where the
//!   preserved side has to be the one that can be scanned for non-matches.
//! * **Whether a sort can be a top-k.** A sort feeding a `LIMIT` never needs to
//!   hold more than `limit + offset` rows, so the limit is pushed into the sort
//!   as a fetch hint. Nothing about the logical plan changes; the sort just
//!   stops keeping rows it can prove it will throw away.

use crate::aggregate::HashAggregateExec;
use crate::eval::evaluate;
use crate::join::{BuildSide, HashJoinExec};
use crate::operator::Operator;
use crate::operators::{DistinctExec, FilterExec, LimitExec, ProjectionExec, ValuesExec};
use crate::scan::ScanExec;
use crate::sort::SortExec;
use qf_common::{Error, Result, Schema};
use qf_plan::cost::estimate_rows;
use qf_plan::logical::LogicalPlan;
use qf_sql::ast::JoinType;
use qf_storage::array::ArrayBuilder;
use qf_storage::batch::RecordBatch;
use qf_storage::catalog::{Catalog, TableSource};
use std::sync::Arc;

/// Builds the operator tree for a plan.
pub fn build(plan: &LogicalPlan, catalog: &Catalog) -> Result<Box<dyn Operator>> {
    build_with_fetch(plan, catalog, None)
}

fn build_with_fetch(
    plan: &LogicalPlan,
    catalog: &Catalog,
    fetch: Option<usize>,
) -> Result<Box<dyn Operator>> {
    Ok(match plan {
        LogicalPlan::Scan {
            table,
            source_schema,
            projection,
            pushed_filters,
            ..
        } => {
            let entry = catalog.get(table)?;
            let indices = projection
                .clone()
                .unwrap_or_else(|| (0..source_schema.len()).collect());
            let schema = Arc::new(source_schema.project(&indices)?);
            match &entry.source {
                TableSource::Memory(batches) => Box::new(ScanExec::from_batches(
                    batches.clone(),
                    indices,
                    pushed_filters.clone(),
                    schema,
                )),
                TableSource::File(path) => Box::new(ScanExec::from_file(
                    path,
                    indices,
                    pushed_filters.clone(),
                    schema,
                )?),
            }
        }

        LogicalPlan::Filter { input, predicate } => Box::new(FilterExec::new(
            build_with_fetch(input, catalog, None)?,
            predicate.clone(),
        )),

        // A projection neither adds nor removes rows, so a fetch hint from a
        // limit above it passes straight through to whatever is below.
        LogicalPlan::Projection { input, exprs } => Box::new(ProjectionExec::new(
            build_with_fetch(input, catalog, fetch)?,
            exprs.clone(),
        )),

        LogicalPlan::Aggregate {
            input,
            group_exprs,
            aggregates,
        } => Box::new(HashAggregateExec::new(
            build_with_fetch(input, catalog, None)?,
            group_exprs.clone(),
            aggregates.clone(),
        )),

        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => {
            let build_side = choose_build_side(left, right, *join_type);
            Box::new(HashJoinExec::new(
                build_with_fetch(left, catalog, None)?,
                build_with_fetch(right, catalog, None)?,
                *join_type,
                on.clone(),
                filter.clone(),
                build_side,
            ))
        }

        LogicalPlan::Sort { input, exprs } => Box::new(SortExec::new(
            build_with_fetch(input, catalog, None)?,
            exprs.clone(),
            fetch,
        )),

        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            // A sort below only ever needs the rows this limit can reach.
            let hint = limit.map(|l| l.saturating_add(*offset));
            Box::new(LimitExec::new(
                build_with_fetch(input, catalog, hint)?,
                *limit,
                *offset,
            ))
        }

        LogicalPlan::Distinct { input } => {
            Box::new(DistinctExec::new(build_with_fetch(input, catalog, None)?))
        }

        LogicalPlan::Values { schema, rows } => Box::new(ValuesExec::new(
            Arc::clone(schema),
            values_batches(schema, rows)?,
        )),
    })
}

/// The smaller input becomes the hash table. On an outer join the choice is
/// forced by which side has to survive without a match, so this only decides
/// the inner and cross cases.
fn choose_build_side(left: &LogicalPlan, right: &LogicalPlan, join_type: JoinType) -> BuildSide {
    match join_type {
        JoinType::Left | JoinType::Full => BuildSide::Right,
        JoinType::Right => BuildSide::Left,
        _ => {
            if estimate_rows(left) <= estimate_rows(right) {
                BuildSide::Left
            } else {
                BuildSide::Right
            }
        }
    }
}

/// Evaluates a `VALUES` list into a batch. The expressions are constant, so
/// they are evaluated against a single empty row.
fn values_batches(
    schema: &Arc<Schema>,
    rows: &[Vec<qf_plan::expr::BoundExpr>],
) -> Result<Vec<RecordBatch>> {
    if schema.is_empty() {
        // The single empty row a `SELECT` with no `FROM` runs against.
        let batch = RecordBatch::empty(Arc::clone(schema))?.take(&vec![0usize; rows.len()])?;
        return Ok(vec![batch]);
    }
    let seed = RecordBatch::empty(Arc::new(Schema::empty()))?.take(&[0usize; 1])?;
    let mut builders: Vec<ArrayBuilder> = schema
        .fields()
        .iter()
        .map(|f| ArrayBuilder::new(f.data_type))
        .collect();
    for row in rows {
        if row.len() != schema.len() {
            return Err(Error::plan(format!(
                "a row has {} values but the table has {} columns",
                row.len(),
                schema.len()
            )));
        }
        for (i, e) in row.iter().enumerate() {
            let column = evaluate(e, &seed)?;
            builders[i].push(column.value(0))?;
        }
    }
    let columns = builders
        .into_iter()
        .map(ArrayBuilder::finish)
        .collect::<Result<Vec<_>>>()?;
    Ok(vec![RecordBatch::try_new(Arc::clone(schema), columns)?])
}

/// Renders the running operator tree with its counters, for `EXPLAIN ANALYZE`.
pub fn explain_analyze(op: &dyn Operator) -> String {
    fn walk(op: &dyn Operator, depth: usize, out: &mut String) {
        for _ in 0..depth {
            out.push_str("  ");
        }
        out.push_str(op.name());
        out.push_str(" (");
        out.push_str(&op.metrics().describe());
        out.push_str(")\n");
        for c in op.children() {
            walk(c, depth + 1, out);
        }
    }
    let mut out = String::new();
    walk(op, 0, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::collect_one;
    use qf_common::{DataType, Field, Value};
    use qf_plan::binder::bind;
    use qf_plan::expr::BoundExpr;
    use qf_plan::optimizer::optimize;
    use qf_storage::array::Array;
    use qf_storage::stats::{ColumnStats, TableStats};

    fn catalog() -> Catalog {
        let mut c = Catalog::new();
        add(&mut c, "big", 100_000);
        add(&mut c, "small", 10);
        c
    }

    fn add(c: &mut Catalog, name: &str, rows: usize) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, true),
        ]));
        let values: Vec<Value> = (0..rows.min(10) as i64).map(Value::Int64).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Array::from_values(DataType::Int64, &values).unwrap(),
                Array::from_values(DataType::Int64, &values).unwrap(),
            ],
        )
        .unwrap();
        let stats = TableStats {
            row_count: rows,
            columns: vec![
                ColumnStats {
                    min: Value::Int64(0),
                    max: Value::Int64(rows as i64),
                    null_count: 0,
                    row_count: rows,
                    distinct_count: rows,
                };
                2
            ],
        };
        c.replace(qf_storage::catalog::Table {
            name: name.to_string(),
            schema,
            source: qf_storage::catalog::TableSource::Memory(vec![batch]),
            stats,
        });
    }

    fn plan(sql: &str, c: &Catalog) -> LogicalPlan {
        match qf_sql::parse(sql).unwrap() {
            qf_sql::ast::Statement::Query(q) => optimize(bind(&q, c).unwrap()).unwrap(),
            other => panic!("expected a query, got {other:?}"),
        }
    }

    fn find_join(plan: &LogicalPlan) -> (&LogicalPlan, &LogicalPlan, JoinType) {
        fn walk(p: &LogicalPlan) -> Option<(&LogicalPlan, &LogicalPlan, JoinType)> {
            if let LogicalPlan::Join {
                left,
                right,
                join_type,
                ..
            } = p
            {
                return Some((left, right, *join_type));
            }
            p.children().into_iter().find_map(walk)
        }
        walk(plan).expect("a join")
    }

    #[test]
    fn the_smaller_input_becomes_the_build_side() {
        let c = catalog();
        let p = plan("SELECT b.id FROM big b JOIN small s ON b.id = s.id", &c);
        let (l, r, t) = find_join(&p);
        assert_eq!(
            choose_build_side(l, r, t),
            BuildSide::Right,
            "small is right"
        );

        let p = plan("SELECT b.id FROM small s JOIN big b ON s.id = b.id", &c);
        let (l, r, t) = find_join(&p);
        assert_eq!(choose_build_side(l, r, t), BuildSide::Left, "small is left");
    }

    #[test]
    fn an_outer_join_s_build_side_is_forced_by_the_side_it_preserves() {
        let c = catalog();
        for (sql, want) in [
            (
                "SELECT b.id FROM big b LEFT JOIN small s ON b.id = s.id",
                BuildSide::Right,
            ),
            (
                "SELECT b.id FROM big b RIGHT JOIN small s ON b.id = s.id",
                BuildSide::Left,
            ),
            (
                "SELECT b.id FROM big b FULL JOIN small s ON b.id = s.id",
                BuildSide::Right,
            ),
        ] {
            let p = plan(sql, &c);
            let (l, r, t) = find_join(&p);
            assert_eq!(choose_build_side(l, r, t), want, "{sql}");
        }
    }

    #[test]
    fn a_limit_pushes_a_fetch_hint_through_a_projection_into_the_sort() {
        // The sort must see the limit, or `ORDER BY ... LIMIT 5` would hold
        // the whole table.
        let c = catalog();
        let p = plan("SELECT id FROM big ORDER BY id LIMIT 5 OFFSET 2", &c);
        let mut op = build(&p, &c).unwrap();
        let out = collect_one(op.as_mut()).unwrap();
        assert!(out.num_rows() <= 5);
        let text = explain_analyze(op.as_ref());
        assert!(text.contains("Sort"), "{text}");
        // The sort emitted at most limit + offset rows.
        let sort_line = text
            .lines()
            .find(|l| l.trim_start().starts_with("Sort"))
            .unwrap();
        assert!(
            sort_line.contains("rows=7") || sort_line.contains("rows=8"),
            "{sort_line}"
        );
    }

    #[test]
    fn a_values_plan_evaluates_its_constant_expressions() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let rows = vec![vec![
            BoundExpr::Literal(Value::Int64(1)),
            BoundExpr::Literal(Value::Utf8("x".into())),
        ]];
        let batches = values_batches(&schema, &rows).unwrap();
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[0].row(0)[1], Value::Utf8("x".into()));
    }

    #[test]
    fn a_values_row_of_the_wrong_width_is_refused() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let rows = vec![vec![
            BoundExpr::Literal(Value::Int64(1)),
            BoundExpr::Literal(Value::Int64(2)),
        ]];
        assert!(values_batches(&schema, &rows)
            .unwrap_err()
            .to_string()
            .contains("but the table has"));
    }

    #[test]
    fn a_schemaless_values_plan_produces_bare_rows() {
        let schema = Arc::new(Schema::empty());
        let batches = values_batches(&schema, &[vec![]]).unwrap();
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[0].num_columns(), 0);
    }

    #[test]
    fn every_logical_node_has_a_physical_counterpart() {
        let c = catalog();
        let sqls = [
            "SELECT id FROM big",
            "SELECT id FROM big WHERE id > 1",
            "SELECT DISTINCT id FROM big",
            "SELECT id FROM big ORDER BY id",
            "SELECT id FROM big LIMIT 1",
            "SELECT count(*) FROM big",
            "SELECT b.id FROM big b JOIN small s ON b.id = s.id",
            "SELECT 1",
        ];
        for sql in sqls {
            let p = plan(sql, &c);
            let mut op = build(&p, &c).unwrap();
            assert!(collect_one(op.as_mut()).is_ok(), "{sql}");
            assert!(!explain_analyze(op.as_ref()).is_empty(), "{sql}");
        }
    }

    #[test]
    fn explain_analyze_indents_children_under_their_parent() {
        let c = catalog();
        let p = plan("SELECT count(*) FROM big WHERE id > 1", &c);
        let mut op = build(&p, &c).unwrap();
        collect_one(op.as_mut()).unwrap();
        let text = explain_analyze(op.as_ref());
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("Projection ("));
        assert!(lines.last().unwrap().starts_with("    Scan ("));
    }

    #[test]
    fn scanning_a_table_that_is_not_registered_is_an_error() {
        let c = Catalog::new();
        let p = LogicalPlan::Scan {
            table: "ghost".into(),
            source_schema: Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            projection: None,
            pushed_filters: vec![],
            stats: TableStats::default(),
        };
        assert!(build(&p, &c).is_err());
    }
}
