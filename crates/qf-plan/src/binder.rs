//! Name resolution and type checking: AST in, logical plan out.
//!
//! This is the only place in the engine that answers "does this column exist",
//! "which table did it come from" and "is this expression well typed". Once a
//! plan leaves the binder, every column is a position and every node knows its
//! type, so nothing downstream has to carry a catalog around or produce a
//! user-facing name error.

use crate::expr::{AggFunc, BoundAggregate, BoundExpr};
use crate::logical::{LogicalPlan, SortExpr};
use qf_common::{DataType, Error, Field, Result, Schema, Value};
use qf_sql::ast::{
    is_aggregate_name, BinaryOp, Expr, OrderByExpr, Query, SelectItem, TableRef, UnaryOp,
};
use qf_storage::catalog::Catalog;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct ScopeColumn {
    /// The table name or alias a reference may be qualified with.
    qualifier: Option<String>,
    name: String,
    index: usize,
    data_type: DataType,
}

/// The columns visible to an expression, and where each one sits in the input.
#[derive(Debug, Clone, Default)]
struct Scope {
    columns: Vec<ScopeColumn>,
}

impl Scope {
    fn from_schema(schema: &Schema, qualifier: Option<String>, offset: usize) -> Scope {
        Scope {
            columns: schema
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| ScopeColumn {
                    qualifier: qualifier.clone(),
                    name: f.name.clone(),
                    index: i + offset,
                    data_type: f.data_type,
                })
                .collect(),
        }
    }

    fn concat(mut left: Scope, right: Scope) -> Scope {
        left.columns.extend(right.columns);
        left
    }

    fn width(&self) -> usize {
        self.columns.len()
    }

    /// Resolves `[table.]name`.
    ///
    /// An unqualified name matching two tables is an error rather than a
    /// first-match win: silently picking one is how a join query comes back
    /// with the wrong column and no complaint.
    fn resolve(&self, qualifier: Option<&str>, name: &str) -> Result<BoundExpr> {
        let mut found: Option<&ScopeColumn> = None;
        for c in &self.columns {
            let qualifier_matches = match qualifier {
                None => true,
                Some(q) => c
                    .qualifier
                    .as_deref()
                    .is_some_and(|cq| cq.eq_ignore_ascii_case(q)),
            };
            if qualifier_matches && c.name.eq_ignore_ascii_case(name) {
                if found.is_some() {
                    return Err(Error::plan(format!(
                        "column `{name}` is ambiguous — qualify it with a table name"
                    )));
                }
                found = Some(c);
            }
        }
        let c = found.ok_or_else(|| match qualifier {
            Some(q) => Error::plan(format!("no column `{name}` in `{q}`")),
            None => Error::plan(format!("no such column `{name}`")),
        })?;
        Ok(BoundExpr::column(c.index, c.name.clone(), c.data_type))
    }
}

/// Collects the aggregates a query mentions while its expressions are bound.
#[derive(Debug, Default)]
struct AggregateState {
    group_asts: Vec<Expr>,
    group_exprs: Vec<(BoundExpr, String)>,
    aggregates: Vec<BoundAggregate>,
}

impl AggregateState {
    /// Registers an aggregate, reusing an identical one already seen so that
    /// `SELECT count(*) ... HAVING count(*) > 2` computes it once.
    fn add(&mut self, agg: BoundAggregate) -> usize {
        if let Some(i) = self
            .aggregates
            .iter()
            .position(|a| a.func == agg.func && a.arg == agg.arg && a.distinct == agg.distinct)
        {
            return i;
        }
        self.aggregates.push(agg);
        self.aggregates.len() - 1
    }

    fn group_index(&self, ast: &Expr) -> Option<usize> {
        self.group_asts.iter().position(|g| g == ast)
    }
}

pub struct Binder<'a> {
    catalog: &'a Catalog,
}

impl<'a> Binder<'a> {
    pub fn new(catalog: &'a Catalog) -> Binder<'a> {
        Binder { catalog }
    }

    pub fn bind_query(&self, query: &Query) -> Result<LogicalPlan> {
        let (mut plan, scope) = match &query.from {
            Some(from) => self.bind_from(from)?,
            // `SELECT 1 + 1` with no FROM runs over a single empty row.
            None => (
                LogicalPlan::Values {
                    schema: Arc::new(Schema::empty()),
                    rows: vec![vec![]],
                },
                Scope::default(),
            ),
        };

        if let Some(predicate) = &query.selection {
            // WHERE runs before grouping, so an aggregate there has nothing to
            // aggregate over. HAVING is the clause that filters groups.
            if predicate.contains_aggregate() {
                return Err(Error::plan(
                    "WHERE cannot contain an aggregate — use HAVING".to_string(),
                ));
            }
            let bound = self.bind_expr(predicate, &scope)?;
            require_boolean(&bound, "WHERE")?;
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: bound,
            };
        }

        let uses_aggregates = !query.group_by.is_empty()
            || query.projection.iter().any(|item| match item {
                SelectItem::Expr { expr, .. } => expr.contains_aggregate(),
                _ => false,
            })
            || query.having.as_ref().is_some_and(Expr::contains_aggregate)
            || query.order_by.iter().any(|o| o.expr.contains_aggregate());

        if uses_aggregates {
            self.bind_aggregate_query(query, plan, &scope)
        } else {
            self.bind_simple_query(query, plan, &scope)
        }
    }

    // ---- FROM ----

    fn bind_from(&self, from: &TableRef) -> Result<(LogicalPlan, Scope)> {
        match from {
            TableRef::Table { name, alias } => {
                let table = self.catalog.get(name)?;
                let qualifier = Some(alias.clone().unwrap_or_else(|| table.name.clone()));
                let scope = Scope::from_schema(&table.schema, qualifier, 0);
                let plan = LogicalPlan::Scan {
                    table: table.name.clone(),
                    source_schema: Arc::clone(&table.schema),
                    projection: None,
                    pushed_filters: vec![],
                    stats: table.stats.clone(),
                };
                Ok((plan, scope))
            }
            TableRef::Join {
                left,
                right,
                join_type,
                on,
            } => {
                let (left_plan, left_scope) = self.bind_from(left)?;
                let (right_plan, right_scope) = self.bind_from(right)?;
                let offset = left_scope.width();
                let right_scope = Scope {
                    columns: right_scope
                        .columns
                        .into_iter()
                        .map(|mut c| {
                            c.index += offset;
                            c
                        })
                        .collect(),
                };
                let combined = Scope::concat(left_scope, right_scope);

                let (keys, residual) = match on {
                    None => (vec![], None),
                    Some(expr) => {
                        let bound = self.bind_expr(expr, &combined)?;
                        require_boolean(&bound, "ON")?;
                        split_join_condition(bound, offset)?
                    }
                };

                Ok((
                    LogicalPlan::Join {
                        left: Box::new(left_plan),
                        right: Box::new(right_plan),
                        join_type: *join_type,
                        on: keys,
                        filter: residual,
                    },
                    combined,
                ))
            }
        }
    }

    // ---- non-aggregate queries ----

    fn bind_simple_query(
        &self,
        query: &Query,
        input: LogicalPlan,
        scope: &Scope,
    ) -> Result<LogicalPlan> {
        let projection = self.bind_projection(&query.projection, scope, None)?;
        let sorts = self.bind_order_by(&query.order_by, scope, &query.projection, None)?;

        let mut plan = input;
        if !sorts.is_empty() {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                exprs: sorts,
            };
        }
        plan = LogicalPlan::Projection {
            input: Box::new(plan),
            exprs: projection,
        };
        Ok(self.finish(plan, query))
    }

    // ---- aggregate queries ----

    fn bind_aggregate_query(
        &self,
        query: &Query,
        input: LogicalPlan,
        scope: &Scope,
    ) -> Result<LogicalPlan> {
        let mut state = AggregateState::default();
        for g in &query.group_by {
            // `GROUP BY 1` refers to the first output column.
            let ast = resolve_ordinal(g, &query.projection)?;
            if ast.contains_aggregate() {
                return Err(Error::plan(
                    "GROUP BY cannot contain an aggregate".to_string(),
                ));
            }
            let bound = self.bind_expr(&ast, scope)?;
            let name = bound.output_name();
            state.group_asts.push(ast);
            state.group_exprs.push((bound, name));
        }

        let projection = self.bind_projection(&query.projection, scope, Some(&mut state))?;
        let having = match &query.having {
            Some(h) => {
                let bound = self.bind_post_aggregate(h, scope, &mut state)?;
                require_boolean(&bound, "HAVING")?;
                Some(bound)
            }
            None => None,
        };
        let sorts =
            self.bind_order_by(&query.order_by, scope, &query.projection, Some(&mut state))?;

        let mut plan = LogicalPlan::Aggregate {
            input: Box::new(input),
            group_exprs: state.group_exprs,
            aggregates: state.aggregates,
        };
        if let Some(h) = having {
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: h,
            };
        }
        if !sorts.is_empty() {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                exprs: sorts,
            };
        }
        plan = LogicalPlan::Projection {
            input: Box::new(plan),
            exprs: projection,
        };
        Ok(self.finish(plan, query))
    }

    fn finish(&self, mut plan: LogicalPlan, query: &Query) -> LogicalPlan {
        if query.distinct {
            plan = LogicalPlan::Distinct {
                input: Box::new(plan),
            };
        }
        if query.limit.is_some() || query.offset.unwrap_or(0) > 0 {
            plan = LogicalPlan::Limit {
                input: Box::new(plan),
                limit: query.limit,
                offset: query.offset.unwrap_or(0),
            };
        }
        plan
    }

    // ---- projections ----

    fn bind_projection(
        &self,
        items: &[SelectItem],
        scope: &Scope,
        mut state: Option<&mut AggregateState>,
    ) -> Result<Vec<(BoundExpr, String)>> {
        let mut out = Vec::new();
        for item in items {
            match item {
                SelectItem::Wildcard | SelectItem::QualifiedWildcard(_) => {
                    if state.is_some() {
                        return Err(Error::plan(
                            "`*` cannot be mixed with aggregates — list the grouped columns"
                                .to_string(),
                        ));
                    }
                    let qualifier = match item {
                        SelectItem::QualifiedWildcard(q) => Some(q.as_str()),
                        _ => None,
                    };
                    let mut matched = 0;
                    for c in &scope.columns {
                        let ok = qualifier.is_none_or(|q| {
                            c.qualifier
                                .as_deref()
                                .is_some_and(|cq| cq.eq_ignore_ascii_case(q))
                        });
                        if ok {
                            matched += 1;
                            out.push((
                                BoundExpr::column(c.index, c.name.clone(), c.data_type),
                                c.name.clone(),
                            ));
                        }
                    }
                    if matched == 0 {
                        return Err(Error::plan(match qualifier {
                            Some(q) => format!("no table named `{q}` in this query"),
                            None => "there are no columns to select".to_string(),
                        }));
                    }
                }
                SelectItem::Expr { expr, alias } => {
                    let bound = match state.as_deref_mut() {
                        Some(s) => self.bind_post_aggregate(expr, scope, s)?,
                        None => self.bind_expr(expr, scope)?,
                    };
                    let name = alias.clone().unwrap_or_else(|| bound.output_name());
                    out.push((bound, name));
                }
            }
        }
        if out.is_empty() {
            return Err(Error::plan("a query must select something".to_string()));
        }
        Ok(out)
    }

    fn bind_order_by(
        &self,
        items: &[OrderByExpr],
        scope: &Scope,
        projection: &[SelectItem],
        mut state: Option<&mut AggregateState>,
    ) -> Result<Vec<SortExpr>> {
        let mut out = Vec::new();
        for item in items {
            // `ORDER BY 2` and `ORDER BY alias` both mean an output column, so
            // resolve them back to the expression that produced it.
            let ast = resolve_ordinal(&item.expr, projection)?;
            let ast = resolve_alias(&ast, projection);
            let expr = match state.as_deref_mut() {
                Some(s) => self.bind_post_aggregate(&ast, scope, s)?,
                None => self.bind_expr(&ast, scope)?,
            };
            out.push(SortExpr {
                expr,
                ascending: item.ascending,
                nulls_first: item.nulls_first,
            });
        }
        Ok(out)
    }

    // ---- expressions ----

    /// Binds an expression that may reference aggregates, producing positions
    /// in the aggregate node's output rather than in its input.
    fn bind_post_aggregate(
        &self,
        expr: &Expr,
        scope: &Scope,
        state: &mut AggregateState,
    ) -> Result<BoundExpr> {
        // A whole grouped expression resolves to its group column.
        if let Some(i) = state.group_index(expr) {
            let (e, name) = &state.group_exprs[i];
            return Ok(BoundExpr::column(i, name.clone(), e.data_type()));
        }

        if let Expr::Function {
            name,
            args,
            distinct,
        } = expr
        {
            if is_aggregate_name(name) {
                let agg = self.bind_aggregate_call(name, args, *distinct, scope)?;
                let data_type = agg.data_type;
                let output_name = agg.output_name.clone();
                let i = state.add(agg);
                return Ok(BoundExpr::column(
                    state.group_exprs.len() + i,
                    output_name,
                    data_type,
                ));
            }
        }

        match expr {
            // A bare column that is neither grouped nor aggregated has no
            // single value per group. Rejecting it is the difference between
            // an error and a silently arbitrary row.
            Expr::Column { table, name } => {
                let qualified = match table {
                    Some(t) => format!("{t}.{name}"),
                    None => name.clone(),
                };
                // Resolve first so a genuinely missing column reports that
                // rather than the grouping error.
                scope.resolve(table.as_deref(), name)?;
                Err(Error::plan(format!(
                    "`{qualified}` must appear in GROUP BY or inside an aggregate"
                )))
            }
            Expr::Literal(v) => Ok(BoundExpr::Literal(v.clone())),
            Expr::Binary { left, op, right } => BoundExpr::binary(
                self.bind_post_aggregate(left, scope, state)?,
                *op,
                self.bind_post_aggregate(right, scope, state)?,
            ),
            Expr::Unary { op, expr } => {
                BoundExpr::unary(*op, self.bind_post_aggregate(expr, scope, state)?)
            }
            Expr::Cast { expr, data_type } => Ok(BoundExpr::Cast {
                expr: Box::new(self.bind_post_aggregate(expr, scope, state)?),
                data_type: *data_type,
            }),
            Expr::IsNull { expr, negated } => Ok(BoundExpr::IsNull {
                expr: Box::new(self.bind_post_aggregate(expr, scope, state)?),
                negated: *negated,
            }),
            other => {
                // The remaining shapes desugar or bind identically whether or
                // not aggregates are in play, so route them through the same
                // code with a post-aggregate binder for their children.
                self.bind_expr_with(other, scope, &mut |e, sc| {
                    self.bind_post_aggregate(e, sc, state)
                })
            }
        }
    }

    fn bind_aggregate_call(
        &self,
        name: &str,
        args: &[Expr],
        distinct: bool,
        scope: &Scope,
    ) -> Result<BoundAggregate> {
        let func = AggFunc::from_name(name)
            .ok_or_else(|| Error::plan(format!("unknown aggregate `{name}`")))?;

        if args.len() != 1 {
            return Err(Error::plan(format!(
                "{}() takes exactly one argument, found {}",
                func.name(),
                args.len()
            )));
        }

        let (arg, input_type, rendered) = match &args[0] {
            Expr::Wildcard => {
                if !func.accepts_wildcard() {
                    return Err(Error::plan(format!(
                        "{}(*) is not meaningful — name a column",
                        func.name()
                    )));
                }
                (None, None, "*".to_string())
            }
            other => {
                if other.contains_aggregate() {
                    return Err(Error::plan("aggregates cannot be nested".to_string()));
                }
                let bound = self.bind_expr(other, scope)?;
                let t = bound.data_type();
                let rendered = other.to_string();
                (Some(bound), Some(t), rendered)
            }
        };

        let data_type = func.output_type(input_type)?;
        let d = if distinct { "DISTINCT " } else { "" };
        Ok(BoundAggregate {
            func,
            arg,
            distinct,
            output_name: format!("{}({d}{rendered})", func.name()),
            data_type,
        })
    }

    /// Binds an expression against a scope, with no aggregates allowed.
    fn bind_expr(&self, expr: &Expr, scope: &Scope) -> Result<BoundExpr> {
        match expr {
            Expr::Column { table, name } => scope.resolve(table.as_deref(), name),
            Expr::Literal(v) => Ok(BoundExpr::Literal(v.clone())),
            Expr::Function { name, .. } if is_aggregate_name(name) => Err(Error::plan(format!(
                "{name}() is only allowed in SELECT, HAVING or ORDER BY"
            ))),
            Expr::Function { name, .. } => Err(Error::plan(format!("unknown function `{name}`"))),
            Expr::Wildcard => Err(Error::plan(
                "`*` is only meaningful in SELECT or count(*)".to_string(),
            )),
            other => self.bind_expr_with(other, scope, &mut |e, sc| self.bind_expr(e, sc)),
        }
    }

    /// The shapes that bind the same way in both contexts, parameterised by
    /// how their children are bound.
    fn bind_expr_with(
        &self,
        expr: &Expr,
        scope: &Scope,
        bind: &mut dyn FnMut(&Expr, &Scope) -> Result<BoundExpr>,
    ) -> Result<BoundExpr> {
        Ok(match expr {
            Expr::Column { .. } | Expr::Literal(_) | Expr::Function { .. } | Expr::Wildcard => {
                bind(expr, scope)?
            }
            Expr::Binary { left, op, right } => {
                BoundExpr::binary(bind(left, scope)?, *op, bind(right, scope)?)?
            }
            Expr::Unary { op, expr } => BoundExpr::unary(*op, bind(expr, scope)?)?,
            Expr::Cast { expr, data_type } => BoundExpr::Cast {
                expr: Box::new(bind(expr, scope)?),
                data_type: *data_type,
            },
            Expr::IsNull { expr, negated } => BoundExpr::IsNull {
                expr: Box::new(bind(expr, scope)?),
                negated: *negated,
            },
            // BETWEEN desugars rather than becoming its own node: two
            // comparisons are something predicate pushdown already knows how
            // to split, prune with and evaluate.
            Expr::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let e = bind(expr, scope)?;
                let lower = BoundExpr::binary(e.clone(), BinaryOp::GtEq, bind(low, scope)?)?;
                let upper = BoundExpr::binary(e, BinaryOp::LtEq, bind(high, scope)?)?;
                let both = BoundExpr::binary(lower, BinaryOp::And, upper)?;
                if *negated {
                    BoundExpr::unary(UnaryOp::Not, both)?
                } else {
                    both
                }
            }
            Expr::InList {
                expr,
                list,
                negated,
            } => {
                let e = bind(expr, scope)?;
                let items = list
                    .iter()
                    .map(|i| bind(i, scope))
                    .collect::<Result<Vec<_>>>()?;
                for i in &items {
                    if i.is_null_literal() {
                        continue;
                    }
                    DataType::unify(e.data_type(), i.data_type()).map_err(|_| {
                        Error::typ(format!(
                            "IN list holds {} but the column is {}",
                            i.data_type(),
                            e.data_type()
                        ))
                    })?;
                }
                BoundExpr::InList {
                    expr: Box::new(e),
                    list: items,
                    negated: *negated,
                }
            }
            Expr::Like {
                expr,
                pattern,
                negated,
            } => {
                let e = bind(expr, scope)?;
                let p = bind(pattern, scope)?;
                for (side, t) in [("value", e.data_type()), ("pattern", p.data_type())] {
                    if t != DataType::Utf8 {
                        return Err(Error::typ(format!(
                            "LIKE needs text, but its {side} is {t}"
                        )));
                    }
                }
                BoundExpr::Like {
                    expr: Box::new(e),
                    pattern: Box::new(p),
                    negated: *negated,
                }
            }
            Expr::Case {
                operand,
                branches,
                else_result,
            } => {
                let mut bound_branches = Vec::with_capacity(branches.len());
                for (when, then) in branches {
                    // `CASE x WHEN 1 ...` is `CASE WHEN x = 1 ...`.
                    let condition = match operand {
                        Some(o) => {
                            BoundExpr::binary(bind(o, scope)?, BinaryOp::Eq, bind(when, scope)?)?
                        }
                        None => bind(when, scope)?,
                    };
                    bound_branches.push((condition, bind(then, scope)?));
                }
                let else_bound = match else_result {
                    Some(e) => Some(bind(e, scope)?),
                    None => None,
                };
                BoundExpr::case(bound_branches, else_bound)?
            }
        })
    }
}

/// Turns `ORDER BY 2` / `GROUP BY 1` into the expression that output column
/// was built from.
fn resolve_ordinal(expr: &Expr, projection: &[SelectItem]) -> Result<Expr> {
    let Expr::Literal(Value::Int64(n)) = expr else {
        return Ok(expr.clone());
    };
    let n = *n;
    if n < 1 || n as usize > projection.len() {
        return Err(Error::plan(format!(
            "position {n} is not in the select list, which has {} items",
            projection.len()
        )));
    }
    match &projection[(n - 1) as usize] {
        SelectItem::Expr { expr, .. } => Ok(expr.clone()),
        _ => Err(Error::plan(format!(
            "position {n} refers to `*`, which is not a single column"
        ))),
    }
}

/// Turns `ORDER BY alias` into the expression that alias names.
fn resolve_alias(expr: &Expr, projection: &[SelectItem]) -> Expr {
    let Expr::Column { table: None, name } = expr else {
        return expr.clone();
    };
    for item in projection {
        if let SelectItem::Expr {
            expr: inner,
            alias: Some(a),
        } = item
        {
            if a.eq_ignore_ascii_case(name) {
                return inner.clone();
            }
        }
    }
    expr.clone()
}

fn require_boolean(expr: &BoundExpr, clause: &str) -> Result<()> {
    if expr.data_type() != DataType::Boolean {
        return Err(Error::typ(format!(
            "{clause} needs a boolean, found {}",
            expr.data_type()
        )));
    }
    Ok(())
}

/// Splits a bound ON condition into equality key pairs plus whatever is left.
///
/// A conjunct is a join key only if it is an equality with one side reading
/// only left columns and the other only right. Everything else — including an
/// equality between two columns of the same side — stays as a filter, because
/// a hash join cannot build on it.
///
/// Returns the key pairs and whatever could not become a key.
fn split_join_condition(condition: BoundExpr, right_offset: usize) -> Result<SplitCondition> {
    let mut keys = Vec::new();
    let mut residual = Vec::new();

    for conjunct in split_and(condition) {
        if let BoundExpr::Binary {
            left,
            op: BinaryOp::Eq,
            right,
            ..
        } = &conjunct
        {
            let l_side = side_of(left, right_offset);
            let r_side = side_of(right, right_offset);
            match (l_side, r_side) {
                (Side::Left, Side::Right) => {
                    keys.push((left.as_ref().clone(), shift_right(right, right_offset)?));
                    continue;
                }
                (Side::Right, Side::Left) => {
                    keys.push((right.as_ref().clone(), shift_right(left, right_offset)?));
                    continue;
                }
                _ => {}
            }
        }
        residual.push(conjunct);
    }

    let filter = residual.into_iter().reduce(|a, b| BoundExpr::Binary {
        left: Box::new(a),
        op: BinaryOp::And,
        right: Box::new(b),
        data_type: DataType::Boolean,
    });
    Ok((keys, filter))
}

/// Join keys, plus the part of the condition that is not a key.
type SplitCondition = (Vec<(BoundExpr, BoundExpr)>, Option<BoundExpr>);

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Side {
    Left,
    Right,
    Both,
    Neither,
}

fn side_of(expr: &BoundExpr, right_offset: usize) -> Side {
    let indices = expr.column_indices();
    if indices.is_empty() {
        return Side::Neither;
    }
    let left = indices.iter().any(|i| *i < right_offset);
    let right = indices.iter().any(|i| *i >= right_offset);
    match (left, right) {
        (true, false) => Side::Left,
        (false, true) => Side::Right,
        _ => Side::Both,
    }
}

/// Rebases a right-side expression so its column positions are relative to the
/// right input rather than to the joined pair.
fn shift_right(expr: &BoundExpr, offset: usize) -> Result<BoundExpr> {
    expr.remap_columns(&|i| i.checked_sub(offset))
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

/// Binds a `CREATE TABLE` column list into a schema.
pub fn schema_from_columns(columns: &[qf_sql::ast::ColumnDef]) -> Result<Arc<Schema>> {
    let mut seen: Vec<&str> = Vec::new();
    for c in columns {
        if seen.iter().any(|s| s.eq_ignore_ascii_case(&c.name)) {
            return Err(Error::plan(format!("duplicate column `{}`", c.name)));
        }
        seen.push(&c.name);
    }
    Ok(Arc::new(Schema::new(
        columns
            .iter()
            .map(|c| Field::new(&c.name, c.data_type, c.nullable))
            .collect(),
    )))
}

/// Binds a query against a catalog.
pub fn bind(query: &Query, catalog: &Catalog) -> Result<LogicalPlan> {
    Binder::new(catalog).bind_query(query)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qf_storage::array::Array;
    use qf_storage::batch::RecordBatch;

    fn catalog() -> Catalog {
        let mut c = Catalog::new();
        let orders = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("customer_id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Float64, true),
        ]));
        let customers = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        c.register_batches("orders", Arc::clone(&orders), vec![rows(&orders)])
            .unwrap();
        c.register_batches("customers", Arc::clone(&customers), vec![rows(&customers)])
            .unwrap();
        c
    }

    /// One row of type-appropriate values, enough to give the catalog stats.
    fn rows(schema: &Arc<Schema>) -> RecordBatch {
        let columns = schema
            .fields()
            .iter()
            .map(|f| {
                let v = match f.data_type {
                    DataType::Int64 => Value::Int64(1),
                    DataType::Float64 => Value::Float64(1.0),
                    DataType::Utf8 => Value::Utf8("x".into()),
                    DataType::Boolean => Value::Boolean(true),
                };
                Array::from_values(f.data_type, &[v]).unwrap()
            })
            .collect();
        RecordBatch::try_new(Arc::clone(schema), columns).unwrap()
    }

    fn plan(sql: &str) -> Result<LogicalPlan> {
        let stmt = qf_sql::parse(sql)?;
        match stmt {
            qf_sql::ast::Statement::Query(q) => bind(&q, &catalog()),
            other => panic!("expected a query, got {other:?}"),
        }
    }

    fn explain(sql: &str) -> String {
        plan(sql).unwrap().explain()
    }

    fn error(sql: &str) -> String {
        plan(sql).unwrap_err().to_string()
    }

    #[test]
    fn a_projection_sits_above_a_scan() {
        let text = explain("SELECT id FROM orders");
        assert_eq!(text, "Projection: id#0\n  Scan: orders [*]\n");
    }

    #[test]
    fn a_wildcard_expands_to_every_column_in_order() {
        let p = plan("SELECT * FROM orders").unwrap();
        let s = p.schema().unwrap();
        assert_eq!(s.len(), 4);
        assert_eq!(s.field(3).unwrap().name, "amount");
    }

    #[test]
    fn a_qualified_wildcard_expands_to_one_side_of_a_join() {
        let p = plan("SELECT c.* FROM orders o JOIN customers c ON o.customer_id = c.id").unwrap();
        let s = p.schema().unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.field(1).unwrap().name, "name");
        assert!(error("SELECT z.* FROM orders").contains("no table named `z`"));
    }

    #[test]
    fn columns_resolve_to_positions_and_missing_ones_are_reported() {
        let p = plan("SELECT amount FROM orders").unwrap();
        match &p {
            LogicalPlan::Projection { exprs, .. } => {
                assert_eq!(exprs[0].0.column_indices(), vec![3]);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(error("SELECT nope FROM orders").contains("no such column `nope`"));
    }

    #[test]
    fn an_ambiguous_column_across_a_join_is_refused() {
        let err = error("SELECT id FROM orders o JOIN customers c ON o.id = c.id");
        assert!(err.contains("ambiguous"));
        // ...but qualifying it resolves.
        assert!(plan("SELECT o.id FROM orders o JOIN customers c ON o.id = c.id").is_ok());
    }

    #[test]
    fn an_alias_replaces_the_table_name_as_the_qualifier() {
        assert!(plan("SELECT o.id FROM orders o").is_ok());
        assert!(error("SELECT orders.id FROM orders o").contains("no column `id` in `orders`"));
    }

    #[test]
    fn a_where_clause_becomes_a_filter_under_the_projection() {
        let text = explain("SELECT id FROM orders WHERE amount > 10");
        assert!(text.contains("Filter: (amount#3 > 10)"));
        assert!(text.lines().nth(1).unwrap().starts_with("  Filter"));
    }

    #[test]
    fn a_non_boolean_where_clause_is_refused() {
        assert!(error("SELECT id FROM orders WHERE amount").contains("WHERE needs a boolean"));
    }

    #[test]
    fn comparing_a_text_column_with_a_number_is_refused_at_plan_time() {
        assert!(error("SELECT id FROM orders WHERE region > 3").contains("cannot compare"));
    }

    #[test]
    fn an_aggregate_in_where_is_redirected_to_having() {
        assert!(error("SELECT id FROM orders WHERE count(*) > 1").contains("use HAVING"));
    }

    #[test]
    fn between_desugars_into_two_comparisons() {
        let text = explain("SELECT id FROM orders WHERE amount BETWEEN 1 AND 9");
        assert!(text.contains("(amount#3 >= 1)"));
        assert!(text.contains("(amount#3 <= 9)"));
        assert!(text.contains("AND"));
    }

    #[test]
    fn not_between_wraps_the_desugared_form_in_a_negation() {
        let text = explain("SELECT id FROM orders WHERE amount NOT BETWEEN 1 AND 9");
        assert!(text.contains("NOT ("));
    }

    #[test]
    fn a_case_with_an_operand_desugars_into_equality_conditions() {
        let text = explain("SELECT CASE region WHEN 'eu' THEN 1 ELSE 0 END FROM orders");
        assert!(text.contains("WHEN (region#2 = 'eu')"));
    }

    #[test]
    fn in_lists_and_like_patterns_are_type_checked() {
        assert!(plan("SELECT id FROM orders WHERE region IN ('a', 'b')").is_ok());
        assert!(plan("SELECT id FROM orders WHERE region IN ('a', NULL)").is_ok());
        assert!(error("SELECT id FROM orders WHERE region IN (1)").contains("IN list holds"));
        assert!(plan("SELECT id FROM orders WHERE region LIKE 'a%'").is_ok());
        assert!(error("SELECT id FROM orders WHERE amount LIKE 'a%'").contains("LIKE needs text"));
    }

    #[test]
    fn an_equijoin_condition_becomes_join_keys_rebased_to_each_side() {
        let p = plan("SELECT o.id FROM orders o JOIN customers c ON o.customer_id = c.id").unwrap();
        let join = find_join(&p).expect("a join node");
        match join {
            LogicalPlan::Join { on, filter, .. } => {
                assert_eq!(on.len(), 1);
                // Left key is position 1 in orders; right key is position 0 in
                // customers, *not* position 4 of the joined pair.
                assert_eq!(on[0].0.column_indices(), vec![1]);
                assert_eq!(on[0].1.column_indices(), vec![0]);
                assert!(filter.is_none());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_non_equality_in_on_stays_a_filter_rather_than_a_join_key() {
        let p = plan("SELECT o.id FROM orders o JOIN customers c ON o.customer_id > c.id").unwrap();
        match find_join(&p).unwrap() {
            LogicalPlan::Join { on, filter, .. } => {
                assert!(on.is_empty());
                assert!(filter.is_some());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn an_equality_within_one_side_is_not_a_join_key() {
        let p = plan("SELECT o.id FROM orders o JOIN customers c ON o.id = o.customer_id").unwrap();
        match find_join(&p).unwrap() {
            LogicalPlan::Join { on, filter, .. } => {
                assert!(on.is_empty());
                assert!(filter.is_some());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_mixed_on_clause_splits_into_keys_and_a_residual_filter() {
        let p = plan(
            "SELECT o.id FROM orders o JOIN customers c \
             ON o.customer_id = c.id AND o.amount > 5",
        )
        .unwrap();
        match find_join(&p).unwrap() {
            LogicalPlan::Join { on, filter, .. } => {
                assert_eq!(on.len(), 1);
                assert!(filter.is_some());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_reversed_equality_still_becomes_a_join_key_with_the_sides_in_order() {
        let p = plan("SELECT o.id FROM orders o JOIN customers c ON c.id = o.customer_id").unwrap();
        match find_join(&p).unwrap() {
            LogicalPlan::Join { on, .. } => {
                assert_eq!(on[0].0.column_indices(), vec![1], "left key");
                assert_eq!(on[0].1.column_indices(), vec![0], "right key");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_cross_join_has_no_keys_and_needs_no_on_clause() {
        let p = plan("SELECT o.id FROM orders o CROSS JOIN customers c").unwrap();
        match find_join(&p).unwrap() {
            LogicalPlan::Join { on, filter, .. } => {
                assert!(on.is_empty());
                assert!(filter.is_none());
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_non_boolean_on_clause_is_refused() {
        assert!(error("SELECT o.id FROM orders o JOIN customers c ON c.id")
            .contains("ON needs a boolean"));
    }

    #[test]
    fn grouping_produces_an_aggregate_node_with_groups_then_aggregates() {
        let text = explain("SELECT region, count(*) FROM orders GROUP BY region");
        assert!(text.contains("Aggregate: group by [region#2], count(*)"));
        let s = plan("SELECT region, count(*) FROM orders GROUP BY region")
            .unwrap()
            .schema()
            .unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.field(1).unwrap().data_type, DataType::Int64);
    }

    #[test]
    fn a_bare_column_outside_group_by_is_refused() {
        let err = error("SELECT region, amount FROM orders GROUP BY region");
        assert!(err.contains("must appear in GROUP BY"));
        // A genuinely missing column reports that instead.
        assert!(error("SELECT nope FROM orders GROUP BY region").contains("no such column"));
    }

    #[test]
    fn an_aggregate_without_group_by_collapses_to_a_single_row() {
        let text = explain("SELECT count(*), sum(amount) FROM orders");
        assert!(text.contains("Aggregate: count(*), sum(amount#3)"));
        assert!(!text.contains("group by"));
    }

    #[test]
    fn identical_aggregates_are_computed_once() {
        let p = plan("SELECT count(*) FROM orders GROUP BY region HAVING count(*) > 2").unwrap();
        let agg = find_aggregate(&p).unwrap();
        match agg {
            LogicalPlan::Aggregate { aggregates, .. } => assert_eq!(aggregates.len(), 1),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn having_becomes_a_filter_above_the_aggregate() {
        let text = explain("SELECT region FROM orders GROUP BY region HAVING count(*) > 2");
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("Projection"));
        assert!(lines[1].trim_start().starts_with("Filter"));
        assert!(lines[2].trim_start().starts_with("Aggregate"));
    }

    #[test]
    fn aggregate_argument_and_nesting_rules_are_enforced() {
        assert!(error("SELECT sum(*) FROM orders").contains("not meaningful"));
        assert!(error("SELECT sum() FROM orders").contains("exactly one argument"));
        assert!(error("SELECT sum(amount, id) FROM orders").contains("exactly one argument"));
        assert!(error("SELECT sum(count(id)) FROM orders").contains("cannot be nested"));
        assert!(error("SELECT sum(region) FROM orders").contains("cannot sum"));
        assert!(error("SELECT median(amount) FROM orders").contains("unknown"));
        assert!(error("SELECT upper(region) FROM orders").contains("unknown function"));
    }

    #[test]
    fn grouping_by_an_aggregate_or_mixing_a_wildcard_in_is_refused() {
        assert!(error("SELECT count(*) FROM orders GROUP BY count(*)")
            .contains("GROUP BY cannot contain an aggregate"));
        assert!(error("SELECT *, count(*) FROM orders GROUP BY region")
            .contains("cannot be mixed with aggregates"));
    }

    #[test]
    fn avg_widens_to_a_float_even_over_integers() {
        let s = plan("SELECT avg(id) FROM orders")
            .unwrap()
            .schema()
            .unwrap();
        assert_eq!(s.field(0).unwrap().data_type, DataType::Float64);
    }

    #[test]
    fn order_by_sorts_before_the_projection_so_unselected_columns_still_work() {
        let text = explain("SELECT id FROM orders ORDER BY amount DESC");
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("Projection"));
        assert!(lines[1].trim_start().starts_with("Sort: amount#3 DESC"));
    }

    #[test]
    fn order_by_accepts_an_output_position_or_an_alias() {
        assert!(explain("SELECT amount AS a FROM orders ORDER BY a").contains("Sort: amount#3"));
        assert!(explain("SELECT amount FROM orders ORDER BY 1").contains("Sort: amount#3"));
        assert!(error("SELECT amount FROM orders ORDER BY 4").contains("position 4 is not"));
        assert!(error("SELECT * FROM orders ORDER BY 1").contains("refers to `*`"));
    }

    #[test]
    fn group_by_accepts_an_output_position() {
        assert!(explain("SELECT region, count(*) FROM orders GROUP BY 1")
            .contains("group by [region#2]"));
    }

    #[test]
    fn distinct_and_limit_wrap_the_query_outermost() {
        let text = explain("SELECT DISTINCT region FROM orders LIMIT 5 OFFSET 2");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "Limit: 5, offset 2");
        assert_eq!(lines[1].trim(), "Distinct");
        assert!(lines[2].trim().starts_with("Projection"));
    }

    #[test]
    fn a_query_with_no_limit_and_no_offset_gets_neither_node() {
        assert!(!explain("SELECT id FROM orders").contains("Limit"));
        assert!(explain("SELECT id FROM orders OFFSET 3").contains("Offset: 3"));
    }

    #[test]
    fn a_select_without_from_runs_over_one_empty_row() {
        let text = explain("SELECT 1 + 1");
        assert!(text.contains("Values: 1 rows"));
        assert!(text.starts_with("Projection: (1 + 1)"));
    }

    #[test]
    fn a_missing_table_is_reported() {
        assert!(error("SELECT id FROM ghosts").contains("no such table"));
    }

    #[test]
    fn create_table_columns_become_a_schema_and_duplicates_are_refused() {
        use qf_sql::ast::ColumnDef;
        let cols = vec![
            ColumnDef {
                name: "a".into(),
                data_type: DataType::Int64,
                nullable: false,
            },
            ColumnDef {
                name: "b".into(),
                data_type: DataType::Utf8,
                nullable: true,
            },
        ];
        let s = schema_from_columns(&cols).unwrap();
        assert_eq!(s.len(), 2);
        assert!(!s.field(0).unwrap().nullable);

        let dup = vec![cols[0].clone(), cols[0].clone()];
        assert!(schema_from_columns(&dup)
            .unwrap_err()
            .to_string()
            .contains("duplicate column"));
    }

    fn find_join(plan: &LogicalPlan) -> Option<&LogicalPlan> {
        find(plan, &|p| matches!(p, LogicalPlan::Join { .. }))
    }

    fn find_aggregate(plan: &LogicalPlan) -> Option<&LogicalPlan> {
        find(plan, &|p| matches!(p, LogicalPlan::Aggregate { .. }))
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
}
