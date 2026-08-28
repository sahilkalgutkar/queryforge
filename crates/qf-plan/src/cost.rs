//! Cardinality estimation.
//!
//! Everything here is an estimate, and the numbers it produces are only ever
//! used to *choose between* plans, never to decide what a query returns. A bad
//! estimate costs a slower plan; it cannot cost a wrong answer. That is why
//! the defaults below are allowed to be crude.
//!
//! Where a column's zone map is available — which is exactly at a scan — the
//! estimate uses real statistics. Above a scan it falls back to the standard
//! textbook constants, and says so.

use crate::expr::BoundExpr;
use crate::logical::LogicalPlan;
use qf_sql::ast::{BinaryOp, UnaryOp};
use qf_storage::stats::{ColumnStats, TableStats};

/// Fraction of rows a range comparison is assumed to keep when nothing better
/// is known. The classic value, and about as defensible as any constant.
const RANGE_SELECTIVITY: f64 = 1.0 / 3.0;
/// Fallback for a predicate whose shape the estimator does not model.
const UNKNOWN_SELECTIVITY: f64 = 0.25;
/// Fallback join selectivity when neither key has usable statistics.
const UNKNOWN_JOIN_SELECTIVITY: f64 = 0.1;

/// Estimated rows out of a plan node.
pub fn estimate_rows(plan: &LogicalPlan) -> f64 {
    match plan {
        LogicalPlan::Scan {
            pushed_filters,
            stats,
            ..
        } => {
            let base = stats.row_count as f64;
            let s: f64 = pushed_filters
                .iter()
                .map(|f| selectivity(f, Some(stats)))
                .product();
            (base * s).max(1.0)
        }
        LogicalPlan::Filter { input, predicate } => {
            let stats = source_stats(input);
            (estimate_rows(input) * selectivity(predicate, stats)).max(1.0)
        }
        LogicalPlan::Projection { input, .. } => estimate_rows(input),
        LogicalPlan::Aggregate {
            input, group_exprs, ..
        } => {
            if group_exprs.is_empty() {
                return 1.0;
            }
            // Grouping cannot produce more rows than it consumes; without a
            // distinct count for the group keys, assume a tenth.
            let rows = estimate_rows(input);
            (rows * 0.1).clamp(1.0, rows)
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            ..
        } => {
            let (l, r) = (estimate_rows(left), estimate_rows(right));
            let product = l * r;
            if on.is_empty() {
                // A cross join really does produce the product.
                return product.max(1.0);
            }
            // The textbook equi-join estimate: dividing by the larger distinct
            // count assumes the smaller key set is contained in the larger.
            let divisor: f64 = on
                .iter()
                .map(|(lk, rk)| {
                    let ld = key_distinct(left, lk);
                    let rd = key_distinct(right, rk);
                    match (ld, rd) {
                        (Some(a), Some(b)) => a.max(b).max(1.0),
                        (Some(a), None) | (None, Some(a)) => a.max(1.0),
                        (None, None) => 1.0 / UNKNOWN_JOIN_SELECTIVITY,
                    }
                })
                .product();
            let estimate = (product / divisor.max(1.0)).max(1.0);
            match join_type {
                // An outer join emits at least every row of its preserved side.
                qf_sql::ast::JoinType::Left => estimate.max(l),
                qf_sql::ast::JoinType::Right => estimate.max(r),
                qf_sql::ast::JoinType::Full => estimate.max(l).max(r),
                _ => estimate,
            }
        }
        LogicalPlan::Sort { input, .. } => estimate_rows(input),
        LogicalPlan::Distinct { input } => {
            let rows = estimate_rows(input);
            (rows * 0.5).clamp(1.0, rows)
        }
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            let rows = (estimate_rows(input) - *offset as f64).max(0.0);
            match limit {
                Some(l) => rows.min(*l as f64).max(1.0),
                None => rows.max(1.0),
            }
        }
        LogicalPlan::Values { rows, .. } => rows.len().max(1) as f64,
    }
}

/// The table statistics behind a subtree, if it is a scan (possibly with
/// filters or a projection stacked on it).
fn source_stats(plan: &LogicalPlan) -> Option<&TableStats> {
    match plan {
        LogicalPlan::Scan { stats, .. } => Some(stats),
        LogicalPlan::Filter { input, .. } | LogicalPlan::Sort { input, .. } => source_stats(input),
        _ => None,
    }
}

/// Distinct values of a join key, when the key is a plain column of a scan.
fn key_distinct(plan: &LogicalPlan, key: &BoundExpr) -> Option<f64> {
    let BoundExpr::Column { index, .. } = key else {
        return None;
    };
    let stats = source_stats(plan)?;
    let column = column_stats_for(plan, stats, *index)?;
    if column.distinct_count == 0 {
        return None;
    }
    Some(column.distinct_count as f64)
}

/// Maps an output column position back to the statistics of the source column,
/// accounting for a scan that has already been projected.
fn column_stats_for<'a>(
    plan: &LogicalPlan,
    stats: &'a TableStats,
    index: usize,
) -> Option<&'a ColumnStats> {
    let source_index = match scan_of(plan) {
        Some(LogicalPlan::Scan {
            projection: Some(p),
            ..
        }) => *p.get(index)?,
        _ => index,
    };
    stats.column(source_index)
}

fn scan_of(plan: &LogicalPlan) -> Option<&LogicalPlan> {
    match plan {
        LogicalPlan::Scan { .. } => Some(plan),
        LogicalPlan::Filter { input, .. } | LogicalPlan::Sort { input, .. } => scan_of(input),
        _ => None,
    }
}

/// The fraction of rows a predicate is expected to keep, in `[0, 1]`.
pub fn selectivity(expr: &BoundExpr, stats: Option<&TableStats>) -> f64 {
    match expr {
        BoundExpr::Literal(qf_common::Value::Boolean(true)) => 1.0,
        BoundExpr::Literal(qf_common::Value::Boolean(false)) => 0.0,
        BoundExpr::Binary {
            left, op, right, ..
        } => match op {
            BinaryOp::And => selectivity(left, stats) * selectivity(right, stats),
            // Assumes independence, which is wrong whenever the two sides
            // correlate — the usual failure mode of this whole approach.
            BinaryOp::Or => {
                let (a, b) = (selectivity(left, stats), selectivity(right, stats));
                a + b - a * b
            }
            BinaryOp::Eq => column_literal(left, right)
                .and_then(|i| stats?.column(i))
                .map_or(0.1, ColumnStats::selectivity_eq),
            BinaryOp::NotEq => {
                1.0 - column_literal(left, right)
                    .and_then(|i| stats?.column(i))
                    .map_or(0.1, ColumnStats::selectivity_eq)
            }
            BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => RANGE_SELECTIVITY,
            _ => UNKNOWN_SELECTIVITY,
        },
        BoundExpr::Unary {
            op: UnaryOp::Not,
            expr,
            ..
        } => 1.0 - selectivity(expr, stats),
        BoundExpr::IsNull { expr, negated } => {
            let fraction = match (expr.as_ref(), stats) {
                (BoundExpr::Column { index, .. }, Some(s)) => match s.column(*index) {
                    Some(c) if c.row_count > 0 => c.null_count as f64 / c.row_count as f64,
                    _ => 0.1,
                },
                _ => 0.1,
            };
            if *negated {
                1.0 - fraction
            } else {
                fraction
            }
        }
        BoundExpr::InList {
            expr,
            list,
            negated,
        } => {
            let one = match (expr.as_ref(), stats) {
                (BoundExpr::Column { index, .. }, Some(s)) => {
                    s.column(*index).map_or(0.1, ColumnStats::selectivity_eq)
                }
                _ => 0.1,
            };
            let s = (one * list.len() as f64).min(1.0);
            if *negated {
                1.0 - s
            } else {
                s
            }
        }
        // A LIKE pattern's selectivity depends on the pattern in ways this
        // estimator does not model; a leading-wildcard pattern in particular
        // may match everything.
        BoundExpr::Like { negated, .. } => {
            if *negated {
                1.0 - UNKNOWN_SELECTIVITY
            } else {
                UNKNOWN_SELECTIVITY
            }
        }
        _ => UNKNOWN_SELECTIVITY,
    }
}

/// Recognises `column <op> literal` in either order, returning the column
/// position.
fn column_literal(left: &BoundExpr, right: &BoundExpr) -> Option<usize> {
    match (left, right) {
        (BoundExpr::Column { index, .. }, r) if r.is_literal() => Some(*index),
        (l, BoundExpr::Column { index, .. }) if l.is_literal() => Some(*index),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qf_common::{DataType, Field, Schema, Value};
    use qf_sql::ast::JoinType;
    use std::sync::Arc;

    fn stats(row_count: usize, distinct: &[usize], nulls: &[usize]) -> TableStats {
        TableStats {
            row_count,
            columns: distinct
                .iter()
                .zip(nulls)
                .map(|(d, n)| ColumnStats {
                    min: Value::Int64(0),
                    max: Value::Int64(100),
                    null_count: *n,
                    row_count,
                    distinct_count: *d,
                })
                .collect(),
        }
    }

    fn schema(n: usize) -> Arc<Schema> {
        Arc::new(Schema::new(
            (0..n)
                .map(|i| Field::new(format!("c{i}"), DataType::Int64, true))
                .collect(),
        ))
    }

    fn scan(rows: usize, distinct: &[usize]) -> LogicalPlan {
        let nulls = vec![0; distinct.len()];
        LogicalPlan::Scan {
            table: "t".into(),
            source_schema: schema(distinct.len()),
            projection: None,
            pushed_filters: vec![],
            stats: stats(rows, distinct, &nulls),
        }
    }

    fn col(i: usize) -> BoundExpr {
        BoundExpr::column(i, format!("c{i}"), DataType::Int64)
    }

    fn eq(i: usize, v: i64) -> BoundExpr {
        BoundExpr::binary(col(i), BinaryOp::Eq, BoundExpr::Literal(Value::Int64(v))).unwrap()
    }

    #[test]
    fn a_scan_estimates_its_own_row_count() {
        assert_eq!(estimate_rows(&scan(1000, &[10])), 1000.0);
    }

    #[test]
    fn equality_on_a_ten_value_column_keeps_a_tenth() {
        let plan = LogicalPlan::Filter {
            input: Box::new(scan(1000, &[10])),
            predicate: eq(0, 5),
        };
        assert!((estimate_rows(&plan) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn a_pushed_filter_is_reflected_in_the_scan_s_own_estimate() {
        let mut s = scan(1000, &[10]);
        if let LogicalPlan::Scan { pushed_filters, .. } = &mut s {
            pushed_filters.push(eq(0, 5));
        }
        assert!((estimate_rows(&s) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn conjunctions_multiply_and_disjunctions_do_not() {
        let and = BoundExpr::binary(eq(0, 1), BinaryOp::And, eq(1, 2)).unwrap();
        let or = BoundExpr::binary(eq(0, 1), BinaryOp::Or, eq(1, 2)).unwrap();
        let s = stats(1000, &[10, 10], &[0, 0]);
        let sa = selectivity(&and, Some(&s));
        let so = selectivity(&or, Some(&s));
        assert!((sa - 0.01).abs() < 1e-9);
        assert!((so - 0.19).abs() < 1e-9);
        assert!(so > sa);
    }

    #[test]
    fn a_range_predicate_uses_the_standard_third() {
        let lt =
            BoundExpr::binary(col(0), BinaryOp::Lt, BoundExpr::Literal(Value::Int64(5))).unwrap();
        assert!((selectivity(&lt, None) - RANGE_SELECTIVITY).abs() < 1e-9);
    }

    #[test]
    fn negation_inverts_a_selectivity() {
        let e = eq(0, 1);
        let s = stats(100, &[4], &[0]);
        let not = BoundExpr::unary(UnaryOp::Not, e.clone()).unwrap();
        assert!((selectivity(&not, Some(&s)) + selectivity(&e, Some(&s)) - 1.0).abs() < 1e-9);
        let ne = BoundExpr::binary(col(0), BinaryOp::NotEq, BoundExpr::Literal(Value::Int64(1)))
            .unwrap();
        assert!((selectivity(&ne, Some(&s)) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn a_literal_predicate_is_all_or_nothing() {
        assert_eq!(
            selectivity(&BoundExpr::Literal(Value::Boolean(true)), None),
            1.0
        );
        assert_eq!(
            selectivity(&BoundExpr::Literal(Value::Boolean(false)), None),
            0.0
        );
    }

    #[test]
    fn is_null_uses_the_column_s_recorded_null_count() {
        let s = stats(100, &[10], &[25]);
        let is_null = BoundExpr::IsNull {
            expr: Box::new(col(0)),
            negated: false,
        };
        let not_null = BoundExpr::IsNull {
            expr: Box::new(col(0)),
            negated: true,
        };
        assert!((selectivity(&is_null, Some(&s)) - 0.25).abs() < 1e-9);
        assert!((selectivity(&not_null, Some(&s)) - 0.75).abs() < 1e-9);
        // With no statistics it falls back rather than guessing zero.
        assert!(selectivity(&is_null, None) > 0.0);
    }

    #[test]
    fn an_in_list_scales_with_its_length_and_saturates_at_one() {
        let s = stats(100, &[10], &[0]);
        let short = BoundExpr::InList {
            expr: Box::new(col(0)),
            list: vec![BoundExpr::Literal(Value::Int64(1))],
            negated: false,
        };
        let long = BoundExpr::InList {
            expr: Box::new(col(0)),
            list: (0..50)
                .map(|i| BoundExpr::Literal(Value::Int64(i)))
                .collect(),
            negated: false,
        };
        assert!((selectivity(&short, Some(&s)) - 0.1).abs() < 1e-9);
        assert_eq!(selectivity(&long, Some(&s)), 1.0);
        let negated = BoundExpr::InList {
            expr: Box::new(col(0)),
            list: vec![BoundExpr::Literal(Value::Int64(1))],
            negated: true,
        };
        assert!((selectivity(&negated, Some(&s)) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn like_is_treated_as_unknown_in_both_polarities() {
        let like = BoundExpr::Like {
            expr: Box::new(col(0)),
            pattern: Box::new(BoundExpr::Literal(Value::Utf8("a%".into()))),
            negated: false,
        };
        assert!((selectivity(&like, None) - UNKNOWN_SELECTIVITY).abs() < 1e-9);
        let not_like = BoundExpr::Like {
            expr: Box::new(col(0)),
            pattern: Box::new(BoundExpr::Literal(Value::Utf8("a%".into()))),
            negated: true,
        };
        assert!((selectivity(&not_like, None) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn an_equijoin_divides_by_the_larger_distinct_count() {
        // 1000 x 100 rows joined on a key with 100 distinct values on the
        // right and 10 on the left → 1000 * 100 / 100.
        let join = LogicalPlan::Join {
            left: Box::new(scan(1000, &[10])),
            right: Box::new(scan(100, &[100])),
            join_type: JoinType::Inner,
            on: vec![(col(0), col(0))],
            filter: None,
        };
        assert!((estimate_rows(&join) - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn a_cross_join_estimates_the_full_product() {
        let join = LogicalPlan::Join {
            left: Box::new(scan(50, &[5])),
            right: Box::new(scan(20, &[5])),
            join_type: JoinType::Cross,
            on: vec![],
            filter: None,
        };
        assert!((estimate_rows(&join) - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn an_outer_join_never_estimates_fewer_rows_than_its_preserved_side() {
        let mk = |t| LogicalPlan::Join {
            left: Box::new(scan(1000, &[1000])),
            right: Box::new(scan(2, &[2])),
            join_type: t,
            on: vec![(col(0), col(0))],
            filter: None,
        };
        assert!(estimate_rows(&mk(JoinType::Left)) >= 1000.0);
        assert!(estimate_rows(&mk(JoinType::Right)) >= 2.0);
        assert!(estimate_rows(&mk(JoinType::Full)) >= 1000.0);
        assert!(estimate_rows(&mk(JoinType::Inner)) < 1000.0);
    }

    #[test]
    fn a_join_key_that_is_not_a_plain_column_falls_back_to_a_constant() {
        let expr =
            BoundExpr::binary(col(0), BinaryOp::Plus, BoundExpr::Literal(Value::Int64(1))).unwrap();
        let join = LogicalPlan::Join {
            left: Box::new(scan(100, &[10])),
            right: Box::new(scan(100, &[10])),
            join_type: JoinType::Inner,
            on: vec![(expr.clone(), expr)],
            filter: None,
        };
        // 100 * 100 * 0.1
        assert!((estimate_rows(&join) - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn aggregates_collapse_and_a_global_aggregate_yields_one_row() {
        let grouped = LogicalPlan::Aggregate {
            input: Box::new(scan(1000, &[10])),
            group_exprs: vec![(col(0), "c0".into())],
            aggregates: vec![],
        };
        assert!(estimate_rows(&grouped) < 1000.0);
        let global = LogicalPlan::Aggregate {
            input: Box::new(scan(1000, &[10])),
            group_exprs: vec![],
            aggregates: vec![],
        };
        assert_eq!(estimate_rows(&global), 1.0);
    }

    #[test]
    fn limit_caps_the_estimate_and_offset_reduces_it() {
        let input = Box::new(scan(1000, &[10]));
        assert_eq!(
            estimate_rows(&LogicalPlan::Limit {
                input: input.clone(),
                limit: Some(10),
                offset: 0
            }),
            10.0
        );
        assert_eq!(
            estimate_rows(&LogicalPlan::Limit {
                input: input.clone(),
                limit: None,
                offset: 400
            }),
            600.0
        );
        assert_eq!(
            estimate_rows(&LogicalPlan::Limit {
                input,
                limit: Some(10),
                offset: 5000
            }),
            1.0
        );
    }

    #[test]
    fn projection_sort_and_distinct_pass_through_or_shrink() {
        let s = scan(100, &[10]);
        assert_eq!(
            estimate_rows(&LogicalPlan::Projection {
                input: Box::new(s.clone()),
                exprs: vec![(col(0), "c0".into())]
            }),
            100.0
        );
        assert_eq!(
            estimate_rows(&LogicalPlan::Sort {
                input: Box::new(s.clone()),
                exprs: vec![]
            }),
            100.0
        );
        assert!(estimate_rows(&LogicalPlan::Distinct { input: Box::new(s) }) < 100.0);
    }

    #[test]
    fn values_estimates_its_own_row_count() {
        assert_eq!(
            estimate_rows(&LogicalPlan::Values {
                schema: schema(1),
                rows: vec![vec![], vec![], vec![]]
            }),
            3.0
        );
    }

    #[test]
    fn an_estimate_never_drops_below_one_row() {
        // Stacking selective filters must not talk the planner into believing
        // a branch is empty and choosing a plan on that basis.
        let mut plan = scan(10, &[1000]);
        for i in 0..8 {
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: eq(0, i),
            };
        }
        assert!(estimate_rows(&plan) >= 1.0);
    }

    #[test]
    fn statistics_are_read_through_a_projected_scan() {
        // After projection pushdown the join key is position 0, but its
        // statistics still live at the source position.
        let s = LogicalPlan::Scan {
            table: "t".into(),
            source_schema: schema(3),
            projection: Some(vec![2]),
            pushed_filters: vec![],
            stats: stats(1000, &[1, 1, 500], &[0, 0, 0]),
        };
        let join = LogicalPlan::Join {
            left: Box::new(s),
            right: Box::new(scan(10, &[10])),
            join_type: JoinType::Inner,
            on: vec![(col(0), col(0))],
            filter: None,
        };
        // Divides by 500 (the real distinct count of the projected column),
        // not by 1 (the count of source column 0).
        assert!((estimate_rows(&join) - 20.0).abs() < 1e-6);
    }
}
