//! Plan rewrites.
//!
//! Each rule is a plan-to-plan function that must preserve two things: the
//! rows the plan produces, and the schema it produces them in. The second is
//! what makes the rules composable — a rule may move a filter or reorder a
//! join, but the node above it must not be able to tell, or every rule would
//! have to know about every other.
//!
//! The rules, in the order they run:
//!
//! 1. **Constant folding** — evaluate what is already known, so later rules see
//!    `false` rather than `1 = 2`.
//! 2. **Predicate pushdown** — move filters as close to the data as possible,
//!    ending inside the scan, where a zone map can skip whole row groups.
//! 3. **Join reordering** — order an inner-join chain by estimated size.
//! 4. **Projection pushdown** — stop reading columns nothing asks for.
//! 5. **Constant folding again** — the earlier rules leave new constants behind.

use crate::cost::estimate_rows;
use crate::expr::BoundExpr;
use crate::logical::LogicalPlan;
use qf_common::{DataType, Error, Result, Value};
use qf_sql::ast::{BinaryOp, JoinType, UnaryOp};
use std::collections::BTreeSet;

/// Runs every rule, in order.
pub fn optimize(plan: LogicalPlan) -> Result<LogicalPlan> {
    let plan = fold_constants(plan)?;
    let plan = push_down_predicates(plan)?;
    let plan = reorder_joins(plan)?;
    let plan = push_down_projections(plan)?;
    fold_constants(plan)
}

/// Names of the rules, for `EXPLAIN`'s header.
pub const RULES: &[&str] = &[
    "constant folding",
    "predicate pushdown",
    "join reordering",
    "projection pushdown",
];

// ---------------------------------------------------------------- folding

/// Evaluates subexpressions whose value is already known and applies the
/// boolean identities.
///
/// Worth doing on its own (`WHERE 1 = 1` should not cost a comparison per row)
/// but the real payoff is downstream: predicate pushdown can drop a conjunct
/// entirely once it folds to `true`.
pub fn fold_constants(plan: LogicalPlan) -> Result<LogicalPlan> {
    let plan = map_children(plan, fold_constants)?;
    Ok(match plan {
        LogicalPlan::Filter { input, predicate } => {
            let folded = fold_expr(&predicate)?;
            match folded.as_literal() {
                // `WHERE true` is not a filter at all.
                Some(Value::Boolean(true)) => *input,
                _ => LogicalPlan::Filter {
                    input,
                    predicate: folded,
                },
            }
        }
        LogicalPlan::Projection { input, exprs } => LogicalPlan::Projection {
            input,
            exprs: exprs
                .into_iter()
                .map(|(e, n)| Ok((fold_expr(&e)?, n)))
                .collect::<Result<Vec<_>>>()?,
        },
        LogicalPlan::Scan {
            table,
            source_schema,
            projection,
            pushed_filters,
            stats,
        } => LogicalPlan::Scan {
            table,
            source_schema,
            projection,
            pushed_filters: pushed_filters
                .iter()
                .map(fold_expr)
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .filter(|f| !matches!(f.as_literal(), Some(Value::Boolean(true))))
                .collect(),
            stats,
        },
        other => other,
    })
}

fn fold_expr(expr: &BoundExpr) -> Result<BoundExpr> {
    Ok(match expr {
        BoundExpr::Binary {
            left,
            op,
            right,
            data_type,
        } => {
            let (l, r) = (fold_expr(left)?, fold_expr(right)?);
            if let Some(v) = fold_binary(&l, *op, &r)? {
                return Ok(BoundExpr::Literal(v));
            }
            if let Some(simplified) = simplify_boolean(&l, *op, &r) {
                return Ok(simplified);
            }
            BoundExpr::Binary {
                left: Box::new(l),
                op: *op,
                right: Box::new(r),
                data_type: *data_type,
            }
        }
        BoundExpr::Unary {
            op,
            expr,
            data_type,
        } => {
            let inner = fold_expr(expr)?;
            match (op, inner.as_literal()) {
                (UnaryOp::Not, Some(Value::Boolean(b))) => BoundExpr::Literal(Value::Boolean(!b)),
                (UnaryOp::Neg, Some(Value::Int64(i))) => BoundExpr::Literal(Value::Int64(-i)),
                (UnaryOp::Neg, Some(Value::Float64(x))) => BoundExpr::Literal(Value::Float64(-x)),
                // Double negation cancels.
                (UnaryOp::Not, None) => match &inner {
                    BoundExpr::Unary {
                        op: UnaryOp::Not,
                        expr: nested,
                        ..
                    } => nested.as_ref().clone(),
                    _ => BoundExpr::Unary {
                        op: *op,
                        expr: Box::new(inner),
                        data_type: *data_type,
                    },
                },
                _ => BoundExpr::Unary {
                    op: *op,
                    expr: Box::new(inner),
                    data_type: *data_type,
                },
            }
        }
        BoundExpr::Cast { expr, data_type } => {
            let inner = fold_expr(expr)?;
            match inner.as_literal() {
                Some(v) => BoundExpr::Literal(v.cast_to(*data_type)?),
                None => BoundExpr::Cast {
                    expr: Box::new(inner),
                    data_type: *data_type,
                },
            }
        }
        BoundExpr::IsNull { expr, negated } => {
            let inner = fold_expr(expr)?;
            match inner.as_literal() {
                Some(v) => BoundExpr::Literal(Value::Boolean(v.is_null() != *negated)),
                None => BoundExpr::IsNull {
                    expr: Box::new(inner),
                    negated: *negated,
                },
            }
        }
        BoundExpr::InList {
            expr,
            list,
            negated,
        } => BoundExpr::InList {
            expr: Box::new(fold_expr(expr)?),
            list: list.iter().map(fold_expr).collect::<Result<Vec<_>>>()?,
            negated: *negated,
        },
        BoundExpr::Like {
            expr,
            pattern,
            negated,
        } => BoundExpr::Like {
            expr: Box::new(fold_expr(expr)?),
            pattern: Box::new(fold_expr(pattern)?),
            negated: *negated,
        },
        BoundExpr::Case {
            branches,
            else_result,
            data_type,
        } => {
            let mut kept = Vec::new();
            for (when, then) in branches {
                let w = fold_expr(when)?;
                let t = fold_expr(then)?;
                match w.as_literal() {
                    // A branch that can never fire is dead.
                    Some(Value::Boolean(false)) | Some(Value::Null) => continue,
                    // A branch that always fires makes everything after it dead.
                    Some(Value::Boolean(true)) => {
                        if kept.is_empty() {
                            return Ok(t);
                        }
                        kept.push((w, t));
                        break;
                    }
                    _ => kept.push((w, t)),
                }
            }
            let else_folded = match else_result {
                Some(e) => Some(fold_expr(e)?),
                None => None,
            };
            if kept.is_empty() {
                return Ok(else_folded.unwrap_or(BoundExpr::Literal(Value::Null)));
            }
            BoundExpr::Case {
                branches: kept,
                else_result: else_folded.map(Box::new),
                data_type: *data_type,
            }
        }
        other => other.clone(),
    })
}

fn fold_binary(left: &BoundExpr, op: BinaryOp, right: &BoundExpr) -> Result<Option<Value>> {
    let (Some(l), Some(r)) = (left.as_literal(), right.as_literal()) else {
        return Ok(None);
    };
    if l.is_null() || r.is_null() {
        // Any operation with NULL is NULL, except that AND/OR have their own
        // three-valued rules, which `simplify_boolean` handles.
        if op.is_logical() {
            return Ok(None);
        }
        return Ok(Some(Value::Null));
    }
    Ok(Some(match op {
        BinaryOp::And => {
            Value::Boolean(l.as_bool()?.unwrap_or(false) && r.as_bool()?.unwrap_or(false))
        }
        BinaryOp::Or => {
            Value::Boolean(l.as_bool()?.unwrap_or(false) || r.as_bool()?.unwrap_or(false))
        }
        op if op.is_comparison() => {
            let Some(ord) = l.sql_compare(r)? else {
                return Ok(Some(Value::Null));
            };
            Value::Boolean(match op {
                BinaryOp::Eq => ord.is_eq(),
                BinaryOp::NotEq => !ord.is_eq(),
                BinaryOp::Lt => ord.is_lt(),
                BinaryOp::LtEq => ord.is_le(),
                BinaryOp::Gt => ord.is_gt(),
                _ => ord.is_ge(),
            })
        }
        _ => return fold_arithmetic(l, op, r),
    }))
}

fn fold_arithmetic(l: &Value, op: BinaryOp, r: &Value) -> Result<Option<Value>> {
    // Division by zero is left alone rather than folded: whatever the runtime
    // does with it, folding must not change the answer at plan time.
    if let (Some(_), Some(0.0)) = (l.as_f64(), r.as_f64()) {
        if matches!(op, BinaryOp::Divide | BinaryOp::Modulo) {
            return Ok(None);
        }
    }
    Ok(Some(match (l, r) {
        (Value::Int64(a), Value::Int64(b)) => match op {
            // Overflow is left unfolded so the runtime's behaviour is the only
            // behaviour.
            BinaryOp::Plus => match a.checked_add(*b) {
                Some(v) => Value::Int64(v),
                None => return Ok(None),
            },
            BinaryOp::Minus => match a.checked_sub(*b) {
                Some(v) => Value::Int64(v),
                None => return Ok(None),
            },
            BinaryOp::Multiply => match a.checked_mul(*b) {
                Some(v) => Value::Int64(v),
                None => return Ok(None),
            },
            BinaryOp::Divide => Value::Float64(*a as f64 / *b as f64),
            BinaryOp::Modulo => Value::Int64(a % b),
            _ => return Ok(None),
        },
        _ => {
            let (Some(a), Some(b)) = (l.as_f64(), r.as_f64()) else {
                return Ok(None);
            };
            match op {
                BinaryOp::Plus => Value::Float64(a + b),
                BinaryOp::Minus => Value::Float64(a - b),
                BinaryOp::Multiply => Value::Float64(a * b),
                BinaryOp::Divide => Value::Float64(a / b),
                BinaryOp::Modulo => Value::Float64(a % b),
                _ => return Ok(None),
            }
        }
    }))
}

/// `x AND true`, `x OR false` and their absorbing counterparts.
fn simplify_boolean(left: &BoundExpr, op: BinaryOp, right: &BoundExpr) -> Option<BoundExpr> {
    let literal = |e: &BoundExpr| match e.as_literal() {
        Some(Value::Boolean(b)) => Some(*b),
        _ => None,
    };
    match op {
        BinaryOp::And => match (literal(left), literal(right)) {
            (Some(true), _) => Some(right.clone()),
            (_, Some(true)) => Some(left.clone()),
            (Some(false), _) | (_, Some(false)) => Some(BoundExpr::Literal(Value::Boolean(false))),
            _ => None,
        },
        BinaryOp::Or => match (literal(left), literal(right)) {
            (Some(false), _) => Some(right.clone()),
            (_, Some(false)) => Some(left.clone()),
            (Some(true), _) | (_, Some(true)) => Some(BoundExpr::Literal(Value::Boolean(true))),
            _ => None,
        },
        _ => None,
    }
}

// ------------------------------------------------------- predicate pushdown

/// Moves filters down the tree, ending inside the scan.
///
/// A filter left above a scan still reads every row. Pushed into the scan it
/// becomes a zone-map test, and whole row groups stop being read at all — the
/// difference between saving CPU and saving I/O.
pub fn push_down_predicates(plan: LogicalPlan) -> Result<LogicalPlan> {
    let plan = map_children(plan, push_down_predicates)?;
    let LogicalPlan::Filter { input, predicate } = plan else {
        return Ok(plan);
    };
    push_filter(*input, split_and(predicate))
}

fn push_filter(input: LogicalPlan, conjuncts: Vec<BoundExpr>) -> Result<LogicalPlan> {
    if conjuncts.is_empty() {
        return Ok(input);
    }
    match input {
        // Two stacked filters become one.
        LogicalPlan::Filter {
            input: inner,
            predicate,
        } => {
            let mut all = split_and(predicate);
            all.extend(conjuncts);
            push_filter(*inner, all)
        }

        LogicalPlan::Scan {
            table,
            source_schema,
            projection,
            mut pushed_filters,
            stats,
        } => {
            pushed_filters.extend(conjuncts);
            Ok(LogicalPlan::Scan {
                table,
                source_schema,
                projection,
                pushed_filters,
                stats,
            })
        }

        // Rewrite the predicate in terms of the projection's own input and
        // push it underneath. Only safe when every column it reads is a plain
        // column of the input — substituting a computed expression could
        // duplicate work rather than save it.
        LogicalPlan::Projection {
            input: inner,
            exprs,
        } => {
            let mut pushable = Vec::new();
            let mut stuck = Vec::new();
            for c in conjuncts {
                match substitute(&c, &exprs) {
                    Some(rewritten) => pushable.push(rewritten),
                    None => stuck.push(c),
                }
            }
            let below = push_filter(*inner, pushable)?;
            let projection = LogicalPlan::Projection {
                input: Box::new(below),
                exprs,
            };
            Ok(rebuild_filter(projection, stuck))
        }

        // Sorting and then filtering is the same as filtering and then
        // sorting, and the second reads fewer rows.
        LogicalPlan::Sort {
            input: inner,
            exprs,
        } => Ok(LogicalPlan::Sort {
            input: Box::new(push_filter(*inner, conjuncts)?),
            exprs,
        }),

        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => push_filter_into_join(*left, *right, join_type, on, filter, conjuncts),

        // Everything else keeps the filter above it. Pushing below a Limit
        // would change which rows the limit sees; pushing below an Aggregate
        // would filter rows the aggregate was supposed to see.
        other => Ok(rebuild_filter(other, conjuncts)),
    }
}

fn push_filter_into_join(
    left: LogicalPlan,
    right: LogicalPlan,
    join_type: JoinType,
    mut on: Vec<(BoundExpr, BoundExpr)>,
    join_filter: Option<BoundExpr>,
    conjuncts: Vec<BoundExpr>,
) -> Result<LogicalPlan> {
    let left_width = left.schema()?.len();
    // Which sides may receive a predicate. Pushing into the null-padded side
    // of an outer join changes the answer: a row that would have been padded
    // with NULLs disappears instead.
    let (can_push_left, can_push_right) = match join_type {
        JoinType::Inner | JoinType::Cross => (true, true),
        JoinType::Left => (true, false),
        JoinType::Right => (false, true),
        JoinType::Full => (false, false),
    };

    let mut to_left = Vec::new();
    let mut to_right = Vec::new();
    let mut stuck = Vec::new();
    let mut join_type = join_type;

    for c in conjuncts {
        let indices = c.column_indices();
        let touches_left = indices.iter().any(|i| *i < left_width);
        let touches_right = indices.iter().any(|i| *i >= left_width);

        if !touches_right && can_push_left {
            to_left.push(c);
            continue;
        }
        if !touches_left && can_push_right {
            to_right.push(c.remap_columns(&|i| i.checked_sub(left_width))?);
            continue;
        }
        // A cross join with an equality across it in the WHERE clause is an
        // inner join that was written the old way. Turning it into a join key
        // is the difference between a hash join and a full cross product.
        if join_type == JoinType::Cross && touches_left && touches_right {
            if let BoundExpr::Binary {
                left: l,
                op: BinaryOp::Eq,
                right: r,
                ..
            } = &c
            {
                let l_only_left = l.column_indices().iter().all(|i| *i < left_width);
                let l_only_right = l.column_indices().iter().all(|i| *i >= left_width);
                let r_only_left = r.column_indices().iter().all(|i| *i < left_width);
                let r_only_right = r.column_indices().iter().all(|i| *i >= left_width);
                if l_only_left && r_only_right {
                    on.push((
                        l.as_ref().clone(),
                        r.remap_columns(&|i| i.checked_sub(left_width))?,
                    ));
                    join_type = JoinType::Inner;
                    continue;
                }
                if r_only_left && l_only_right {
                    on.push((
                        r.as_ref().clone(),
                        l.remap_columns(&|i| i.checked_sub(left_width))?,
                    ));
                    join_type = JoinType::Inner;
                    continue;
                }
            }
        }
        stuck.push(c);
    }

    let new_left = push_filter(left, to_left)?;
    let new_right = push_filter(right, to_right)?;
    let join = LogicalPlan::Join {
        left: Box::new(new_left),
        right: Box::new(new_right),
        join_type,
        on,
        filter: join_filter,
    };
    Ok(rebuild_filter(join, stuck))
}

/// Rewrites a predicate expressed over a projection's output into one over its
/// input, or reports that it cannot.
fn substitute(expr: &BoundExpr, exprs: &[(BoundExpr, String)]) -> Option<BoundExpr> {
    let mut ok = true;
    expr.walk(&mut |e| {
        if let BoundExpr::Column { index, .. } = e {
            match exprs.get(*index) {
                Some((BoundExpr::Column { .. }, _)) => {}
                _ => ok = false,
            }
        }
    });
    if !ok {
        return None;
    }
    rewrite_columns(expr, &|index| exprs.get(index).map(|(e, _)| e.clone())).ok()
}

/// Replaces each column reference with whatever `lookup` returns for it.
fn rewrite_columns(
    expr: &BoundExpr,
    lookup: &dyn Fn(usize) -> Option<BoundExpr>,
) -> Result<BoundExpr> {
    if let BoundExpr::Column { index, name, .. } = expr {
        return lookup(*index)
            .ok_or_else(|| Error::internal(format!("cannot rewrite column `{name}`")));
    }
    // Every other shape just rebuilds with rewritten children, which
    // `remap_columns` already does structurally — but it can only renumber,
    // not substitute, so the walk is repeated here.
    Ok(match expr {
        BoundExpr::Binary {
            left,
            op,
            right,
            data_type,
        } => BoundExpr::Binary {
            left: Box::new(rewrite_columns(left, lookup)?),
            op: *op,
            right: Box::new(rewrite_columns(right, lookup)?),
            data_type: *data_type,
        },
        BoundExpr::Unary {
            op,
            expr,
            data_type,
        } => BoundExpr::Unary {
            op: *op,
            expr: Box::new(rewrite_columns(expr, lookup)?),
            data_type: *data_type,
        },
        BoundExpr::Cast { expr, data_type } => BoundExpr::Cast {
            expr: Box::new(rewrite_columns(expr, lookup)?),
            data_type: *data_type,
        },
        BoundExpr::IsNull { expr, negated } => BoundExpr::IsNull {
            expr: Box::new(rewrite_columns(expr, lookup)?),
            negated: *negated,
        },
        BoundExpr::InList {
            expr,
            list,
            negated,
        } => BoundExpr::InList {
            expr: Box::new(rewrite_columns(expr, lookup)?),
            list: list
                .iter()
                .map(|e| rewrite_columns(e, lookup))
                .collect::<Result<Vec<_>>>()?,
            negated: *negated,
        },
        BoundExpr::Like {
            expr,
            pattern,
            negated,
        } => BoundExpr::Like {
            expr: Box::new(rewrite_columns(expr, lookup)?),
            pattern: Box::new(rewrite_columns(pattern, lookup)?),
            negated: *negated,
        },
        BoundExpr::Case {
            branches,
            else_result,
            data_type,
        } => BoundExpr::Case {
            branches: branches
                .iter()
                .map(|(w, t)| Ok((rewrite_columns(w, lookup)?, rewrite_columns(t, lookup)?)))
                .collect::<Result<Vec<_>>>()?,
            else_result: match else_result {
                Some(e) => Some(Box::new(rewrite_columns(e, lookup)?)),
                None => None,
            },
            data_type: *data_type,
        },
        other => other.clone(),
    })
}

fn rebuild_filter(input: LogicalPlan, conjuncts: Vec<BoundExpr>) -> LogicalPlan {
    match join_and(conjuncts) {
        None => input,
        Some(predicate) => LogicalPlan::Filter {
            input: Box::new(input),
            predicate,
        },
    }
}

fn split_and(expr: BoundExpr) -> Vec<BoundExpr> {
    match expr {
        BoundExpr::Binary {
            left,
            op: BinaryOp::And,
            right,
            ..
        } => {
            let mut out = split_and(*left);
            out.extend(split_and(*right));
            out
        }
        other => vec![other],
    }
}

fn join_and(mut parts: Vec<BoundExpr>) -> Option<BoundExpr> {
    if parts.is_empty() {
        return None;
    }
    let first = parts.remove(0);
    Some(parts.into_iter().fold(first, |a, b| BoundExpr::Binary {
        left: Box::new(a),
        op: BinaryOp::And,
        right: Box::new(b),
        data_type: DataType::Boolean,
    }))
}

// ------------------------------------------------------- projection pushdown

/// Narrows every scan to the columns the query actually reads.
///
/// In a columnar format this is the single largest win available: a query
/// touching two of forty columns should read two column chunks per row group,
/// not forty. Doing it means renumbering every expression above the scan,
/// which is the part that has to be exactly right — a mistake here reads the
/// wrong column and reports a plausible wrong answer.
pub fn push_down_projections(plan: LogicalPlan) -> Result<LogicalPlan> {
    let all: BTreeSet<usize> = (0..plan.schema()?.len()).collect();
    let (pruned, mapping) = prune(plan, &all)?;
    // The top of the plan must still produce its columns in the original
    // order; if pruning reordered them, restore it.
    if mapping.iter().enumerate().all(|(i, m)| i == *m) {
        return Ok(pruned);
    }
    let schema = pruned.schema()?;
    let exprs = mapping
        .iter()
        .enumerate()
        .map(|(new, _)| {
            let f = schema.field(new)?;
            Ok((
                BoundExpr::column(new, f.name.clone(), f.data_type),
                f.name.clone(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LogicalPlan::Projection {
        input: Box::new(pruned),
        exprs,
    })
}

/// Rewrites `plan` to produce at least the columns in `needed`, returning the
/// new plan and the original output positions it now produces, in order.
fn prune(plan: LogicalPlan, needed: &BTreeSet<usize>) -> Result<(LogicalPlan, Vec<usize>)> {
    match plan {
        LogicalPlan::Scan {
            table,
            source_schema,
            projection,
            pushed_filters,
            stats,
        } => {
            let mut wanted: BTreeSet<usize> = needed.clone();
            for f in &pushed_filters {
                wanted.extend(f.column_indices());
            }
            // A scan of nothing still has to produce rows to count, so keep
            // one column rather than none.
            if wanted.is_empty() && !source_schema.is_empty() {
                wanted.insert(0);
            }
            let kept: Vec<usize> = wanted.into_iter().collect();
            let source_indices: Vec<usize> = match &projection {
                Some(p) => kept.iter().map(|i| p[*i]).collect(),
                None => kept.clone(),
            };
            let position = |old: usize| kept.iter().position(|k| *k == old);
            let new_filters = pushed_filters
                .iter()
                .map(|f| f.remap_columns(&position))
                .collect::<Result<Vec<_>>>()?;
            Ok((
                LogicalPlan::Scan {
                    table,
                    source_schema,
                    projection: Some(source_indices),
                    pushed_filters: new_filters,
                    stats,
                },
                kept,
            ))
        }

        LogicalPlan::Filter { input, predicate } => {
            let mut child_needs = needed.clone();
            child_needs.extend(predicate.column_indices());
            let (new_input, mapping) = prune(*input, &child_needs)?;
            let position = |old: usize| mapping.iter().position(|m| *m == old);
            Ok((
                LogicalPlan::Filter {
                    input: Box::new(new_input),
                    predicate: predicate.remap_columns(&position)?,
                },
                mapping,
            ))
        }

        LogicalPlan::Projection { input, exprs } => {
            let kept: Vec<usize> = (0..exprs.len()).filter(|i| needed.contains(i)).collect();
            let kept = if kept.is_empty() { vec![0] } else { kept };
            let mut child_needs = BTreeSet::new();
            for i in &kept {
                child_needs.extend(exprs[*i].0.column_indices());
            }
            let (new_input, mapping) = prune(*input, &child_needs)?;
            let position = |old: usize| mapping.iter().position(|m| *m == old);
            let new_exprs = kept
                .iter()
                .map(|i| Ok((exprs[*i].0.remap_columns(&position)?, exprs[*i].1.clone())))
                .collect::<Result<Vec<_>>>()?;
            Ok((
                LogicalPlan::Projection {
                    input: Box::new(new_input),
                    exprs: new_exprs,
                },
                kept,
            ))
        }

        LogicalPlan::Aggregate {
            input,
            group_exprs,
            aggregates,
        } => {
            // Every group column and aggregate argument is needed; the
            // aggregate's own output is not narrowed, because dropping an
            // aggregate would change what the groups mean to the node above.
            let mut child_needs = BTreeSet::new();
            for (e, _) in &group_exprs {
                child_needs.extend(e.column_indices());
            }
            for a in &aggregates {
                if let Some(arg) = &a.arg {
                    child_needs.extend(arg.column_indices());
                }
            }
            let (new_input, mapping) = prune(*input, &child_needs)?;
            let position = |old: usize| mapping.iter().position(|m| *m == old);
            let new_groups = group_exprs
                .iter()
                .map(|(e, n)| Ok((e.remap_columns(&position)?, n.clone())))
                .collect::<Result<Vec<_>>>()?;
            let new_aggs = aggregates
                .iter()
                .map(|a| {
                    let mut a = a.clone();
                    a.arg = match &a.arg {
                        Some(arg) => Some(arg.remap_columns(&position)?),
                        None => None,
                    };
                    Ok(a)
                })
                .collect::<Result<Vec<_>>>()?;
            let width = new_groups.len() + new_aggs.len();
            Ok((
                LogicalPlan::Aggregate {
                    input: Box::new(new_input),
                    group_exprs: new_groups,
                    aggregates: new_aggs,
                },
                (0..width).collect(),
            ))
        }

        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => {
            let left_width = left.schema()?.len();
            let mut left_needs = BTreeSet::new();
            let mut right_needs = BTreeSet::new();
            for i in needed {
                if *i < left_width {
                    left_needs.insert(*i);
                } else {
                    right_needs.insert(*i - left_width);
                }
            }
            for (l, r) in &on {
                left_needs.extend(l.column_indices());
                right_needs.extend(r.column_indices());
            }
            if let Some(f) = &filter {
                for i in f.column_indices() {
                    if i < left_width {
                        left_needs.insert(i);
                    } else {
                        right_needs.insert(i - left_width);
                    }
                }
            }
            let (new_left, left_map) = prune(*left, &left_needs)?;
            let (new_right, right_map) = prune(*right, &right_needs)?;
            let left_pos = |old: usize| left_map.iter().position(|m| *m == old);
            let right_pos = |old: usize| right_map.iter().position(|m| *m == old);

            let new_on = on
                .iter()
                .map(|(l, r)| Ok((l.remap_columns(&left_pos)?, r.remap_columns(&right_pos)?)))
                .collect::<Result<Vec<_>>>()?;
            let new_width = left_map.len();
            let new_filter = match &filter {
                Some(f) => Some(f.remap_columns(&|i| {
                    if i < left_width {
                        left_pos(i)
                    } else {
                        right_pos(i - left_width).map(|p| p + new_width)
                    }
                })?),
                None => None,
            };

            let mut mapping = left_map;
            mapping.extend(right_map.iter().map(|i| i + left_width));
            Ok((
                LogicalPlan::Join {
                    left: Box::new(new_left),
                    right: Box::new(new_right),
                    join_type,
                    on: new_on,
                    filter: new_filter,
                },
                mapping,
            ))
        }

        LogicalPlan::Sort { input, exprs } => {
            let mut child_needs = needed.clone();
            for s in &exprs {
                child_needs.extend(s.expr.column_indices());
            }
            let (new_input, mapping) = prune(*input, &child_needs)?;
            let position = |old: usize| mapping.iter().position(|m| *m == old);
            let new_exprs = exprs
                .iter()
                .map(|s| {
                    Ok(crate::logical::SortExpr {
                        expr: s.expr.remap_columns(&position)?,
                        ascending: s.ascending,
                        nulls_first: s.nulls_first,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((
                LogicalPlan::Sort {
                    input: Box::new(new_input),
                    exprs: new_exprs,
                },
                mapping,
            ))
        }

        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            let (new_input, mapping) = prune(*input, needed)?;
            Ok((
                LogicalPlan::Limit {
                    input: Box::new(new_input),
                    limit,
                    offset,
                },
                mapping,
            ))
        }

        // Narrowing the input of a DISTINCT would change which rows are
        // duplicates, so its input keeps every column.
        LogicalPlan::Distinct { input } => {
            let all: BTreeSet<usize> = (0..input.schema()?.len()).collect();
            let (new_input, mapping) = prune(*input, &all)?;
            Ok((
                LogicalPlan::Distinct {
                    input: Box::new(new_input),
                },
                mapping,
            ))
        }

        LogicalPlan::Values { schema, rows } => {
            let width = schema.len();
            Ok((LogicalPlan::Values { schema, rows }, (0..width).collect()))
        }
    }
}

// ---------------------------------------------------------- join reordering

/// Orders an inner-join chain so the smallest relations join first.
///
/// The rule is greedy: start from the smallest estimated relation, then keep
/// adding the smallest relation that has a join key to what is already joined.
/// Preferring a relation with a key matters more than preferring a small one —
/// joining an unrelated relation means a cross product, and no later choice
/// recovers from that.
///
/// Reordering permutes the join's output columns, so the rule wraps the
/// reordered join in a projection that puts them back. The node above cannot
/// tell the difference, which is what makes the rewrite safe to apply
/// anywhere in the tree.
pub fn reorder_joins(plan: LogicalPlan) -> Result<LogicalPlan> {
    let plan = map_children(plan, reorder_joins)?;
    let LogicalPlan::Join {
        join_type: JoinType::Inner,
        ..
    } = &plan
    else {
        return Ok(plan);
    };

    let mut relations = Vec::new();
    let mut conditions = Vec::new();
    let mut residuals = Vec::new();
    flatten_join(&plan, 0, &mut relations, &mut conditions, &mut residuals)?;
    if relations.len() < 3 {
        // Two relations have only one order, and the build-side choice that
        // does matter for them is made by the physical planner.
        return Ok(plan);
    }

    let order = choose_order(&relations, &conditions);
    if order.iter().enumerate().all(|(i, r)| i == *r) {
        return Ok(plan);
    }
    rebuild_join(plan, relations, conditions, residuals, order)
}

struct Relation {
    plan: LogicalPlan,
    /// Where this relation's columns start in the original flattened order.
    offset: usize,
    width: usize,
    rows: f64,
}

/// A join key in the flattened coordinate space.
struct Condition {
    left: BoundExpr,
    right: BoundExpr,
}

fn flatten_join(
    plan: &LogicalPlan,
    offset: usize,
    relations: &mut Vec<Relation>,
    conditions: &mut Vec<Condition>,
    residuals: &mut Vec<BoundExpr>,
) -> Result<usize> {
    match plan {
        LogicalPlan::Join {
            left,
            right,
            join_type: JoinType::Inner,
            on,
            filter,
        } => {
            let left_width = flatten_join(left, offset, relations, conditions, residuals)?;
            let right_offset = offset + left_width;
            let right_width = flatten_join(right, right_offset, relations, conditions, residuals)?;
            for (l, r) in on {
                conditions.push(Condition {
                    left: l.remap_columns(&|i| Some(i + offset))?,
                    right: r.remap_columns(&|i| Some(i + right_offset))?,
                });
            }
            if let Some(f) = filter {
                residuals.push(f.remap_columns(&|i| Some(i + offset))?);
            }
            Ok(left_width + right_width)
        }
        other => {
            let width = other.schema()?.len();
            relations.push(Relation {
                plan: other.clone(),
                offset,
                width,
                rows: estimate_rows(other),
            });
            Ok(width)
        }
    }
}

fn relation_of(relations: &[Relation], column: usize) -> Option<usize> {
    relations
        .iter()
        .position(|r| column >= r.offset && column < r.offset + r.width)
}

fn condition_relations(relations: &[Relation], c: &Condition) -> (Option<usize>, Option<usize>) {
    let side = |e: &BoundExpr| {
        let mut found = None;
        for i in e.column_indices() {
            match (found, relation_of(relations, i)) {
                (None, r) => found = r,
                (Some(a), Some(b)) if a != b => return None,
                _ => {}
            }
        }
        found
    };
    (side(&c.left), side(&c.right))
}

fn choose_order(relations: &[Relation], conditions: &[Condition]) -> Vec<usize> {
    let mut remaining: Vec<usize> = (0..relations.len()).collect();
    // Start from the smallest relation: the first join's build side is the one
    // that has to fit in memory.
    remaining.sort_by(|a, b| {
        relations[*a]
            .rows
            .partial_cmp(&relations[*b].rows)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut order = vec![remaining.remove(0)];

    while !remaining.is_empty() {
        let connected: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|r| {
                conditions.iter().any(|c| {
                    let (a, b) = condition_relations(relations, c);
                    matches!((a, b), (Some(x), Some(y))
                        if (order.contains(&x) && y == *r) || (order.contains(&y) && x == *r))
                })
            })
            .collect();
        // `remaining` is already sorted by size, so the first connected
        // relation is also the smallest connected one.
        let pick = connected.first().copied().unwrap_or(remaining[0]);
        order.push(pick);
        remaining.retain(|r| *r != pick);
    }
    order
}

fn rebuild_join(
    original: LogicalPlan,
    relations: Vec<Relation>,
    conditions: Vec<Condition>,
    residuals: Vec<BoundExpr>,
    order: Vec<usize>,
) -> Result<LogicalPlan> {
    // Where each original column ends up in the reordered join.
    let mut position = vec![usize::MAX; relations.iter().map(|r| r.width).sum()];
    let mut cursor = 0;
    for r in &order {
        let rel = &relations[*r];
        for k in 0..rel.width {
            position[rel.offset + k] = cursor + k;
        }
        cursor += rel.width;
    }
    let map = |i: usize| position.get(i).copied().filter(|p| *p != usize::MAX);

    let mut joined: Vec<usize> = vec![order[0]];
    let mut plan = relations[order[0]].plan.clone();
    let mut used = vec![false; conditions.len()];
    let mut width = relations[order[0]].width;

    for step in &order[1..] {
        let rel = &relations[*step];
        let mut keys = Vec::new();
        for (i, c) in conditions.iter().enumerate() {
            if used[i] {
                continue;
            }
            let (a, b) = condition_relations(&relations, c);
            let (Some(a), Some(b)) = (a, b) else { continue };
            let (left_expr, right_expr) = if joined.contains(&a) && b == *step {
                (&c.left, &c.right)
            } else if joined.contains(&b) && a == *step {
                (&c.right, &c.left)
            } else {
                continue;
            };
            used[i] = true;
            keys.push((
                left_expr.remap_columns(&map)?,
                // The right side's positions are relative to the new relation.
                right_expr.remap_columns(&|i| map(i).and_then(|p| p.checked_sub(width)))?,
            ));
        }
        plan = LogicalPlan::Join {
            left: Box::new(plan),
            right: Box::new(rel.plan.clone()),
            join_type: if keys.is_empty() {
                JoinType::Cross
            } else {
                JoinType::Inner
            },
            on: keys,
            filter: None,
        };
        joined.push(*step);
        width += rel.width;
    }

    // Conditions that spanned relations in a way the greedy order could not
    // consume become a filter above the join rather than being dropped.
    let mut leftover: Vec<BoundExpr> = Vec::new();
    for (i, c) in conditions.iter().enumerate() {
        if !used[i] {
            leftover.push(BoundExpr::Binary {
                left: Box::new(c.left.remap_columns(&map)?),
                op: BinaryOp::Eq,
                right: Box::new(c.right.remap_columns(&map)?),
                data_type: DataType::Boolean,
            });
        }
    }
    for r in &residuals {
        leftover.push(r.remap_columns(&map)?);
    }
    let plan = rebuild_filter(plan, leftover);

    // Restore the original column order so nothing above notices.
    let original_schema = original.schema()?;
    let exprs = (0..original_schema.len())
        .map(|i| {
            let f = original_schema.field(i)?;
            let p = position[i];
            Ok((
                BoundExpr::column(p, f.name.clone(), f.data_type),
                f.name.clone(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LogicalPlan::Projection {
        input: Box::new(plan),
        exprs,
    })
}

// ------------------------------------------------------------------ helpers

fn map_children(
    plan: LogicalPlan,
    f: impl Fn(LogicalPlan) -> Result<LogicalPlan> + Copy,
) -> Result<LogicalPlan> {
    let children: Vec<LogicalPlan> = plan.children().into_iter().cloned().collect();
    if children.is_empty() {
        return Ok(plan);
    }
    let new = children.into_iter().map(f).collect::<Result<Vec<_>>>()?;
    plan.with_children(new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binder::bind;
    use qf_common::{Field, Schema};
    use qf_storage::catalog::Catalog;
    use qf_storage::stats::{ColumnStats, TableStats};
    use std::sync::Arc;

    /// A catalog whose tables carry hand-set statistics, so the cost-based
    /// rules have something to reason about without building real data.
    fn catalog() -> Catalog {
        let mut c = Catalog::new();
        add(
            &mut c,
            "orders",
            &["id", "customer_id", "product_id", "amount"],
            100_000,
            &[100_000, 5_000, 900, 50_000],
        );
        add(
            &mut c,
            "customers",
            &["id", "name", "region"],
            5_000,
            &[5_000, 4_900, 8],
        );
        add(
            &mut c,
            "products",
            &["id", "title", "price"],
            900,
            &[900, 900, 400],
        );
        c
    }

    fn add(c: &mut Catalog, name: &str, columns: &[&str], rows: usize, distinct: &[usize]) {
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|n| Field::new(*n, DataType::Int64, true))
                .collect(),
        ));
        let table = qf_storage::catalog::Table {
            name: name.to_string(),
            schema,
            source: qf_storage::catalog::TableSource::Memory(vec![]),
            stats: TableStats {
                row_count: rows,
                columns: distinct
                    .iter()
                    .map(|d| ColumnStats {
                        min: Value::Int64(0),
                        max: Value::Int64(1_000_000),
                        null_count: 0,
                        row_count: rows,
                        distinct_count: *d,
                    })
                    .collect(),
            },
        };
        c.replace(table);
    }

    fn raw(sql: &str) -> LogicalPlan {
        match qf_sql::parse(sql).unwrap() {
            qf_sql::ast::Statement::Query(q) => bind(&q, &catalog()).unwrap(),
            other => panic!("expected a query, got {other:?}"),
        }
    }

    fn opt(sql: &str) -> LogicalPlan {
        optimize(raw(sql)).unwrap()
    }

    fn plan_text(sql: &str) -> String {
        opt(sql).explain()
    }

    fn find<'p>(
        plan: &'p LogicalPlan,
        pred: &dyn Fn(&LogicalPlan) -> bool,
    ) -> Option<&'p LogicalPlan> {
        if pred(plan) {
            return Some(plan);
        }
        plan.children().into_iter().find_map(|c| find(c, pred))
    }

    fn scans(plan: &LogicalPlan) -> Vec<&LogicalPlan> {
        let mut out = Vec::new();
        collect(plan, &mut out);
        fn collect<'p>(p: &'p LogicalPlan, out: &mut Vec<&'p LogicalPlan>) {
            if matches!(p, LogicalPlan::Scan { .. }) {
                out.push(p);
            }
            for c in p.children() {
                collect(c, out);
            }
        }
        out
    }

    /// The invariant every rule has to hold: optimising must not change the
    /// shape of the answer.
    fn assert_schema_preserved(sql: &str) {
        let before = raw(sql).schema().unwrap();
        let after = opt(sql).schema().unwrap();
        assert_eq!(before, after, "schema changed while optimising `{sql}`");
    }

    #[test]
    fn optimising_never_changes_the_output_schema() {
        for sql in [
            "SELECT id FROM orders",
            "SELECT * FROM orders WHERE amount > 10",
            "SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id",
            "SELECT c.region, count(*) FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.region",
            "SELECT DISTINCT customer_id FROM orders ORDER BY customer_id LIMIT 5",
            "SELECT o.id FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id",
            "SELECT o.amount * 2 AS doubled FROM orders o WHERE o.amount BETWEEN 1 AND 5",
            "SELECT 1 + 1",
        ] {
            assert_schema_preserved(sql);
        }
    }

    // ---- constant folding ----

    #[test]
    fn arithmetic_between_literals_is_evaluated_once_at_plan_time() {
        let text = plan_text("SELECT 2 * 3 + 1 FROM orders");
        assert!(text.contains("Projection: 7"), "{text}");
    }

    #[test]
    fn a_predicate_that_is_always_true_disappears_entirely() {
        let text = plan_text("SELECT id FROM orders WHERE 1 = 1");
        assert!(!text.contains("Filter"), "{text}");
        assert!(!text.contains("pushed"), "{text}");
    }

    #[test]
    fn a_predicate_that_is_always_false_survives_so_the_query_returns_nothing() {
        let text = plan_text("SELECT id FROM orders WHERE 1 = 2");
        assert!(text.contains("false"), "{text}");
    }

    #[test]
    fn boolean_identities_collapse() {
        assert!(plan_text("SELECT id FROM orders WHERE id = 1 AND true").contains("(id#0 = 1)"));
        let and_false = plan_text("SELECT id FROM orders WHERE id = 1 AND false");
        assert!(and_false.contains("false"), "{and_false}");
        assert!(!plan_text("SELECT id FROM orders WHERE id = 1 OR true").contains("id#0 = 1"));
        assert!(plan_text("SELECT id FROM orders WHERE id = 1 OR false").contains("(id#0 = 1)"));
    }

    #[test]
    fn double_negation_cancels_and_a_literal_negation_folds() {
        assert!(plan_text("SELECT id FROM orders WHERE NOT NOT (id = 1)").contains("(id#0 = 1)"));
        assert!(plan_text("SELECT -(-3) FROM orders").contains("Projection: 3"));
        assert!(plan_text("SELECT NOT true FROM orders").contains("Projection: false"));
    }

    #[test]
    fn casts_and_null_tests_over_literals_fold() {
        assert!(plan_text("SELECT CAST('42' AS INT) FROM orders").contains("Projection: 42"));
        assert!(plan_text("SELECT NULL IS NULL FROM orders").contains("Projection: true"));
        assert!(plan_text("SELECT 1 IS NOT NULL FROM orders").contains("Projection: true"));
    }

    #[test]
    fn a_case_whose_conditions_are_known_collapses_to_the_branch_that_fires() {
        assert!(
            plan_text("SELECT CASE WHEN true THEN 1 ELSE 2 END FROM orders")
                .contains("Projection: 1")
        );
        assert!(
            plan_text("SELECT CASE WHEN false THEN 1 ELSE 2 END FROM orders")
                .contains("Projection: 2")
        );
        assert!(
            plan_text("SELECT CASE WHEN false THEN 1 END FROM orders").contains("Projection: NULL")
        );
    }

    #[test]
    fn division_by_zero_and_overflow_are_left_for_the_runtime_to_decide() {
        // Folding these would mean the plan-time answer and the run-time
        // answer could differ, which is worse than not folding.
        let div = plan_text("SELECT 1 / 0 FROM orders");
        assert!(div.contains('/'), "{div}");
        let overflow = plan_text("SELECT 9223372036854775807 + 1 FROM orders");
        assert!(overflow.contains('+'), "{overflow}");
    }

    #[test]
    fn arithmetic_involving_null_folds_to_null() {
        assert!(plan_text("SELECT 1 + NULL FROM orders").contains("Projection: NULL"));
    }

    // ---- predicate pushdown ----

    #[test]
    fn a_filter_ends_up_inside_the_scan_where_a_zone_map_can_use_it() {
        let plan = opt("SELECT id FROM orders WHERE amount > 100");
        assert!(find(&plan, &|p| matches!(p, LogicalPlan::Filter { .. })).is_none());
        match scans(&plan)[0] {
            LogicalPlan::Scan { pushed_filters, .. } => {
                assert_eq!(pushed_filters.len(), 1);
                assert!(pushed_filters[0].to_string().contains("> 100"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_conjunction_is_split_so_each_part_can_travel_separately() {
        let plan = opt(
            "SELECT o.id FROM orders o JOIN customers c ON o.customer_id = c.id \
             WHERE o.amount > 100 AND c.region = 3",
        );
        // One predicate reached each side of the join, not both stuck above it.
        for s in scans(&plan) {
            match s {
                LogicalPlan::Scan {
                    table,
                    pushed_filters,
                    ..
                } => assert_eq!(pushed_filters.len(), 1, "{table} kept nothing"),
                _ => unreachable!(),
            }
        }
        assert!(find(&plan, &|p| matches!(p, LogicalPlan::Filter { .. })).is_none());
    }

    #[test]
    fn a_predicate_spanning_both_sides_of_a_join_stays_above_it() {
        let plan = opt(
            "SELECT o.id FROM orders o JOIN customers c ON o.customer_id = c.id \
             WHERE o.amount > c.id",
        );
        assert!(find(&plan, &|p| matches!(p, LogicalPlan::Filter { .. })).is_some());
    }

    #[test]
    fn a_cross_join_with_an_equality_in_where_becomes_a_hash_joinable_inner_join() {
        // The old-style comma join must not stay a cross product.
        let plan = opt("SELECT o.id FROM orders o, customers c WHERE o.customer_id = c.id");
        match find(&plan, &|p| matches!(p, LogicalPlan::Join { .. })).unwrap() {
            LogicalPlan::Join { join_type, on, .. } => {
                assert_eq!(*join_type, JoinType::Inner);
                assert_eq!(on.len(), 1);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_reversed_equality_in_where_also_becomes_a_join_key() {
        let plan = opt("SELECT o.id FROM orders o, customers c WHERE c.id = o.customer_id");
        match find(&plan, &|p| matches!(p, LogicalPlan::Join { .. })).unwrap() {
            LogicalPlan::Join { on, join_type, .. } => {
                assert_eq!(*join_type, JoinType::Inner);
                // The left key must be orders.customer_id and the right one
                // customers.id, even though the equality was written the other
                // way round.
                assert_eq!(on.len(), 1);
                assert!(on[0].0.to_string().starts_with("customer_id#"));
                assert!(on[0].1.to_string().starts_with("id#"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_predicate_is_never_pushed_into_the_padded_side_of_an_outer_join() {
        // Pushing `c.region = 3` into the right side of a LEFT JOIN would
        // delete rows that should have come back padded with NULLs.
        let plan = opt(
            "SELECT o.id FROM orders o LEFT JOIN customers c ON o.customer_id = c.id \
             WHERE c.region = 3",
        );
        let customers = scans(&plan)
            .into_iter()
            .find(|s| matches!(s, LogicalPlan::Scan { table, .. } if table == "customers"))
            .unwrap();
        match customers {
            LogicalPlan::Scan { pushed_filters, .. } => assert!(pushed_filters.is_empty()),
            _ => unreachable!(),
        }
        assert!(find(&plan, &|p| matches!(p, LogicalPlan::Filter { .. })).is_some());
    }

    #[test]
    fn a_predicate_on_the_preserved_side_of_a_left_join_does_push_down() {
        let plan = opt(
            "SELECT o.id FROM orders o LEFT JOIN customers c ON o.customer_id = c.id \
             WHERE o.amount > 5",
        );
        let orders = scans(&plan)
            .into_iter()
            .find(|s| matches!(s, LogicalPlan::Scan { table, .. } if table == "orders"))
            .unwrap();
        match orders {
            LogicalPlan::Scan { pushed_filters, .. } => assert_eq!(pushed_filters.len(), 1),
            _ => unreachable!(),
        }
    }

    #[test]
    fn a_full_outer_join_accepts_no_pushdown_at_all() {
        let plan = opt(
            "SELECT o.id FROM orders o FULL JOIN customers c ON o.customer_id = c.id \
             WHERE o.amount > 5",
        );
        for s in scans(&plan) {
            match s {
                LogicalPlan::Scan { pushed_filters, .. } => assert!(pushed_filters.is_empty()),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn a_filter_travels_through_a_projection_of_plain_columns() {
        let plan = opt("SELECT id FROM orders WHERE id > 5 ORDER BY id");
        match scans(&plan)[0] {
            LogicalPlan::Scan { pushed_filters, .. } => assert_eq!(pushed_filters.len(), 1),
            _ => unreachable!(),
        }
    }

    #[test]
    fn a_filter_does_not_push_below_a_limit_or_an_aggregate() {
        // Filtering before a LIMIT would change which rows the limit sees,
        // so the filter has to stay above it.
        let limited = optimize(LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Limit {
                input: Box::new(raw("SELECT id FROM orders")),
                limit: Some(10),
                offset: 0,
            }),
            predicate: BoundExpr::binary(
                BoundExpr::column(0, "id", DataType::Int64),
                BinaryOp::Gt,
                BoundExpr::Literal(Value::Int64(5)),
            )
            .unwrap(),
        })
        .unwrap();
        assert!(matches!(limited, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn having_stays_above_the_aggregate_rather_than_becoming_a_scan_filter() {
        let plan = opt(
            "SELECT customer_id, count(*) FROM orders GROUP BY customer_id HAVING count(*) > 2",
        );
        assert!(find(&plan, &|p| matches!(p, LogicalPlan::Filter { .. })).is_some());
        for s in scans(&plan) {
            match s {
                LogicalPlan::Scan { pushed_filters, .. } => assert!(pushed_filters.is_empty()),
                _ => unreachable!(),
            }
        }
    }

    // ---- projection pushdown ----

    #[test]
    fn a_scan_reads_only_the_columns_the_query_mentions() {
        let plan = opt("SELECT id FROM orders WHERE amount > 5");
        match scans(&plan)[0] {
            LogicalPlan::Scan {
                projection: Some(p),
                source_schema,
                ..
            } => {
                // `id` and `amount` only — not `customer_id` or `product_id`.
                assert_eq!(p.len(), 2, "read {p:?} of {source_schema}");
                assert!(p.contains(&0) && p.contains(&3));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn each_side_of_a_join_reads_only_its_own_needed_columns() {
        let plan = opt(
            "SELECT c.name FROM orders o JOIN customers c ON o.customer_id = c.id \
             WHERE o.amount > 5",
        );
        for s in scans(&plan) {
            match s {
                LogicalPlan::Scan {
                    table,
                    projection: Some(p),
                    ..
                } => {
                    // orders reads id + amount; customers reads id + name.
                    assert_eq!(p.len(), 2, "{table} read {p:?}");
                }
                other => panic!("unexpected {other:?}"),
            }
        }
    }

    #[test]
    fn a_count_only_query_still_reads_one_column_so_it_has_rows_to_count() {
        let plan = opt("SELECT count(*) FROM orders");
        match scans(&plan)[0] {
            LogicalPlan::Scan {
                projection: Some(p),
                ..
            } => assert_eq!(p.len(), 1),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn pruning_renumbers_the_expressions_above_the_scan() {
        // `amount` is source column 3 but becomes position 1 after pruning;
        // the predicate has to follow it.
        let plan = opt("SELECT id FROM orders WHERE amount > 5");
        match scans(&plan)[0] {
            LogicalPlan::Scan { pushed_filters, .. } => {
                assert_eq!(pushed_filters[0].column_indices(), vec![1]);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn distinct_keeps_every_column_because_pruning_would_change_which_rows_repeat() {
        let plan = opt("SELECT DISTINCT id, customer_id FROM orders");
        match scans(&plan)[0] {
            LogicalPlan::Scan {
                projection: Some(p),
                ..
            } => assert_eq!(p.len(), 2),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_column_used_only_for_sorting_is_still_read() {
        let plan = opt("SELECT id FROM orders ORDER BY amount");
        match scans(&plan)[0] {
            LogicalPlan::Scan {
                projection: Some(p),
                ..
            } => assert_eq!(p.len(), 2),
            other => panic!("unexpected {other:?}"),
        }
    }

    // ---- join reordering ----

    #[test]
    fn a_three_way_join_is_reordered_smallest_first() {
        // Written orders(100k) ⋈ customers(5k) ⋈ products(900); the smallest
        // relation should end up at the bottom of the tree.
        let plan = opt("SELECT o.id FROM orders o \
             JOIN customers c ON o.customer_id = c.id \
             JOIN products p ON o.product_id = p.id");
        let deepest = deepest_scan(&plan);
        assert_eq!(deepest, "products", "plan was:\n{}", plan.explain());
    }

    #[test]
    fn reordering_preserves_the_output_column_order() {
        let sql = "SELECT o.id, c.name, p.title FROM orders o \
                   JOIN customers c ON o.customer_id = c.id \
                   JOIN products p ON o.product_id = p.id";
        let before = raw(sql).schema().unwrap();
        let after = opt(sql).schema().unwrap();
        assert_eq!(before, after);
        assert_eq!(after.field(0).unwrap().name, "id");
        assert_eq!(after.field(1).unwrap().name, "name");
        assert_eq!(after.field(2).unwrap().name, "title");
    }

    #[test]
    fn reordering_keeps_every_join_key() {
        let plan = opt("SELECT o.id FROM orders o \
             JOIN customers c ON o.customer_id = c.id \
             JOIN products p ON o.product_id = p.id");
        let mut keys = 0;
        count_keys(&plan, &mut keys);
        fn count_keys(p: &LogicalPlan, n: &mut usize) {
            if let LogicalPlan::Join { on, .. } = p {
                *n += on.len();
            }
            for c in p.children() {
                count_keys(c, n);
            }
        }
        assert_eq!(keys, 2, "a join key was lost:\n{}", plan.explain());
    }

    #[test]
    fn reordering_never_introduces_a_cross_product_between_related_tables() {
        let plan = opt("SELECT o.id FROM orders o \
             JOIN customers c ON o.customer_id = c.id \
             JOIN products p ON o.product_id = p.id");
        let mut crosses = 0;
        count_crosses(&plan, &mut crosses);
        fn count_crosses(p: &LogicalPlan, n: &mut usize) {
            if let LogicalPlan::Join {
                join_type: JoinType::Cross,
                ..
            } = p
            {
                *n += 1;
            }
            for c in p.children() {
                count_crosses(c, n);
            }
        }
        assert_eq!(crosses, 0, "plan was:\n{}", plan.explain());
    }

    #[test]
    fn a_two_table_join_is_left_alone() {
        let plan = opt("SELECT o.id FROM orders o JOIN customers c ON o.customer_id = c.id");
        // No restoring projection was needed, so the join sits directly under
        // the query's own projection.
        let text = plan.explain();
        assert_eq!(text.matches("Projection").count(), 1, "{text}");
    }

    #[test]
    fn an_outer_join_chain_is_not_reordered() {
        // Reordering outer joins changes which side gets padded.
        let sql = "SELECT o.id FROM orders o \
                   LEFT JOIN customers c ON o.customer_id = c.id \
                   LEFT JOIN products p ON o.product_id = p.id";
        assert_eq!(deepest_scan(&opt(sql)), "orders");
    }

    fn deepest_scan(plan: &LogicalPlan) -> String {
        fn walk(p: &LogicalPlan, depth: usize, best: &mut (usize, String)) {
            if let LogicalPlan::Scan { table, .. } = p {
                if depth > best.0 {
                    *best = (depth, table.clone());
                }
            }
            for c in p.children() {
                walk(c, depth + 1, best);
            }
        }
        let mut best = (0, String::new());
        walk(plan, 0, &mut best);
        best.1
    }

    // ---- the rewriting machinery, exercised directly ----

    fn c(i: usize) -> BoundExpr {
        BoundExpr::column(i, format!("c{i}"), DataType::Int64)
    }

    fn text(i: usize) -> BoundExpr {
        BoundExpr::column(i, format!("s{i}"), DataType::Utf8)
    }

    /// A projection mapping output position i to input position 10 + i.
    fn shift_by_ten() -> Vec<(BoundExpr, String)> {
        (0..4).map(|i| (c(10 + i), format!("out{i}"))).collect()
    }

    #[test]
    fn substitution_reaches_inside_every_expression_shape() {
        let projection = shift_by_ten();
        let shapes: Vec<BoundExpr> = vec![
            BoundExpr::binary(c(0), BinaryOp::Plus, c(1)).unwrap(),
            BoundExpr::unary(UnaryOp::Neg, c(0)).unwrap(),
            BoundExpr::Cast {
                expr: Box::new(c(0)),
                data_type: DataType::Float64,
            },
            BoundExpr::IsNull {
                expr: Box::new(c(0)),
                negated: true,
            },
            BoundExpr::InList {
                expr: Box::new(c(0)),
                list: vec![c(1), BoundExpr::Literal(Value::Int64(3))],
                negated: false,
            },
            BoundExpr::case(
                vec![(BoundExpr::binary(c(0), BinaryOp::Gt, c(1)).unwrap(), c(2))],
                Some(c(3)),
            )
            .unwrap(),
        ];
        for shape in shapes {
            let rewritten = substitute(&shape, &projection).expect("should substitute");
            let expected: Vec<usize> = shape.column_indices().iter().map(|i| i + 10).collect();
            assert_eq!(rewritten.column_indices(), expected, "{shape}");
        }
    }

    #[test]
    fn substitution_reaches_inside_a_like_expression() {
        let projection = vec![(text(10), "a".to_string()), (text(11), "b".to_string())];
        let like = BoundExpr::Like {
            expr: Box::new(text(0)),
            pattern: Box::new(text(1)),
            negated: true,
        };
        let rewritten = substitute(&like, &projection).unwrap();
        assert_eq!(rewritten.column_indices(), vec![10, 11]);
    }

    #[test]
    fn substitution_refuses_a_projection_that_computes_rather_than_renames() {
        // Pushing a predicate through `SELECT a + b AS x` would duplicate the
        // addition rather than save work, so the rule declines.
        let computed = vec![(
            BoundExpr::binary(c(0), BinaryOp::Plus, c(1)).unwrap(),
            "x".to_string(),
        )];
        let predicate =
            BoundExpr::binary(c(0), BinaryOp::Gt, BoundExpr::Literal(Value::Int64(1))).unwrap();
        assert!(substitute(&predicate, &computed).is_none());
    }

    #[test]
    fn substitution_refuses_a_column_the_projection_does_not_produce() {
        let projection = vec![(c(10), "a".to_string())];
        assert!(substitute(&c(5), &projection).is_none());
    }

    #[test]
    fn a_literal_only_expression_substitutes_to_itself() {
        let projection = shift_by_ten();
        let lit = BoundExpr::Literal(Value::Int64(7));
        assert_eq!(substitute(&lit, &projection).unwrap(), lit);
    }

    #[test]
    fn rewriting_a_column_with_no_replacement_is_an_error() {
        assert!(rewrite_columns(&c(0), &|_| None).is_err());
    }

    #[test]
    fn folding_covers_every_comparison_operator() {
        let cases = [
            (BinaryOp::Eq, 1, 1, true),
            (BinaryOp::NotEq, 1, 2, true),
            (BinaryOp::Lt, 1, 2, true),
            (BinaryOp::LtEq, 2, 2, true),
            (BinaryOp::Gt, 3, 2, true),
            (BinaryOp::GtEq, 2, 2, true),
            (BinaryOp::Lt, 3, 2, false),
        ];
        for (op, a, b, want) in cases {
            let e = BoundExpr::binary(
                BoundExpr::Literal(Value::Int64(a)),
                op,
                BoundExpr::Literal(Value::Int64(b)),
            )
            .unwrap();
            assert_eq!(
                fold_expr(&e).unwrap().as_literal(),
                Some(&Value::Boolean(want)),
                "{a} {op} {b}"
            );
        }
    }

    #[test]
    fn folding_covers_every_arithmetic_operator_in_both_int_and_float_form() {
        let cases: [(BinaryOp, i64, i64, Value); 4] = [
            (BinaryOp::Plus, 2, 3, Value::Int64(5)),
            (BinaryOp::Minus, 5, 3, Value::Int64(2)),
            (BinaryOp::Multiply, 4, 3, Value::Int64(12)),
            (BinaryOp::Modulo, 7, 4, Value::Int64(3)),
        ];
        for (op, a, b, want) in cases {
            let e = BoundExpr::binary(
                BoundExpr::Literal(Value::Int64(a)),
                op,
                BoundExpr::Literal(Value::Int64(b)),
            )
            .unwrap();
            assert_eq!(fold_expr(&e).unwrap().as_literal(), Some(&want), "{op}");
        }
        let float = BoundExpr::binary(
            BoundExpr::Literal(Value::Float64(1.5)),
            BinaryOp::Plus,
            BoundExpr::Literal(Value::Int64(1)),
        )
        .unwrap();
        assert_eq!(
            fold_expr(&float).unwrap().as_literal(),
            Some(&Value::Float64(2.5))
        );
    }

    #[test]
    fn comparing_a_literal_with_null_folds_to_null_not_to_false() {
        let e = BoundExpr::binary(
            BoundExpr::Literal(Value::Int64(1)),
            BinaryOp::Eq,
            BoundExpr::Literal(Value::Null),
        )
        .unwrap();
        assert_eq!(fold_expr(&e).unwrap().as_literal(), Some(&Value::Null));
    }

    #[test]
    fn folding_reaches_inside_in_lists_and_like_patterns() {
        let in_list = BoundExpr::InList {
            expr: Box::new(c(0)),
            list: vec![BoundExpr::binary(
                BoundExpr::Literal(Value::Int64(1)),
                BinaryOp::Plus,
                BoundExpr::Literal(Value::Int64(1)),
            )
            .unwrap()],
            negated: false,
        };
        assert_eq!(fold_expr(&in_list).unwrap().to_string(), "c0#0 IN (2)");

        let like = BoundExpr::Like {
            expr: Box::new(text(0)),
            pattern: Box::new(BoundExpr::Cast {
                expr: Box::new(BoundExpr::Literal(Value::Int64(5))),
                data_type: DataType::Utf8,
            }),
            negated: false,
        };
        assert_eq!(fold_expr(&like).unwrap().to_string(), "s0#0 LIKE '5'");
    }

    #[test]
    fn a_case_whose_first_branch_is_unknown_drops_that_branch() {
        let e = BoundExpr::Case {
            branches: vec![
                (BoundExpr::Literal(Value::Null), c(0)),
                (BoundExpr::Literal(Value::Boolean(true)), c(1)),
            ],
            else_result: None,
            data_type: DataType::Int64,
        };
        assert_eq!(fold_expr(&e).unwrap(), c(1));
    }

    #[test]
    fn a_case_with_a_live_branch_before_a_constant_one_keeps_both() {
        let live =
            BoundExpr::binary(c(0), BinaryOp::Gt, BoundExpr::Literal(Value::Int64(1))).unwrap();
        let e = BoundExpr::Case {
            branches: vec![
                (live, c(1)),
                (BoundExpr::Literal(Value::Boolean(true)), c(2)),
                // Unreachable: the branch above always fires.
                (BoundExpr::Literal(Value::Boolean(true)), c(3)),
            ],
            else_result: None,
            data_type: DataType::Int64,
        };
        let folded = fold_expr(&e).unwrap();
        match folded {
            BoundExpr::Case { branches, .. } => assert_eq!(branches.len(), 2),
            other => panic!("unexpected {other}"),
        }
    }

    #[test]
    fn a_scan_of_a_table_with_no_useful_columns_still_reads_one() {
        // `SELECT count(*)` needs rows, not columns.
        let plan = opt("SELECT count(*) FROM orders");
        match scans(&plan)[0] {
            LogicalPlan::Scan {
                projection: Some(p),
                ..
            } => assert_eq!(p.len(), 1),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn projection_pushdown_restores_the_output_order_when_pruning_reorders_it() {
        // Selecting the columns backwards means the pruned scan produces them
        // in source order and something has to put them back.
        let sql = "SELECT amount, id FROM orders";
        let before = raw(sql).schema().unwrap();
        let after = push_down_projections(raw(sql)).unwrap().schema().unwrap();
        assert_eq!(before, after);
        assert_eq!(after.field(0).unwrap().name, "amount");
    }

    #[test]
    fn folding_a_scan_drops_a_pushed_filter_that_became_true() {
        let scan = LogicalPlan::Scan {
            table: "orders".into(),
            source_schema: raw("SELECT * FROM orders").schema().unwrap(),
            projection: None,
            pushed_filters: vec![
                BoundExpr::binary(
                    BoundExpr::Literal(Value::Int64(1)),
                    BinaryOp::Eq,
                    BoundExpr::Literal(Value::Int64(1)),
                )
                .unwrap(),
                BoundExpr::binary(c(0), BinaryOp::Gt, BoundExpr::Literal(Value::Int64(5))).unwrap(),
            ],
            stats: TableStats::default(),
        };
        match fold_constants(scan).unwrap() {
            LogicalPlan::Scan { pushed_filters, .. } => assert_eq!(pushed_filters.len(), 1),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn the_rule_list_is_reported_for_explain() {
        assert_eq!(RULES.len(), 4);
        assert!(RULES.contains(&"predicate pushdown"));
    }
}
