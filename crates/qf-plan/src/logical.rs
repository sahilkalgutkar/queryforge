//! The logical plan: what the query means, with no commitment yet to how it
//! runs.
//!
//! Every node computes its own output schema, which is what lets the optimiser
//! rewrite the tree and immediately check that the result still type-checks —
//! a rule that drops a column or reorders a join is caught by the schema, not
//! by a wrong answer at the end of a run.

use crate::expr::{BoundAggregate, BoundExpr};
use qf_common::{Error, Field, Result, Schema};
use qf_sql::ast::JoinType;
use qf_storage::stats::TableStats;
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct SortExpr {
    pub expr: BoundExpr,
    pub ascending: bool,
    pub nulls_first: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// A table read.
    ///
    /// `projection` and `pushed_filters` start empty and are filled in by the
    /// optimiser. They are part of the scan rather than separate nodes because
    /// that is exactly what the storage layer can act on: read fewer column
    /// chunks, and skip row groups whose zone map rules the predicate out.
    Scan {
        table: String,
        /// The table's full schema, before projection.
        source_schema: Arc<Schema>,
        projection: Option<Vec<usize>>,
        pushed_filters: Vec<BoundExpr>,
        stats: TableStats,
    },
    Filter {
        input: Box<LogicalPlan>,
        predicate: BoundExpr,
    },
    Projection {
        input: Box<LogicalPlan>,
        exprs: Vec<(BoundExpr, String)>,
    },
    Aggregate {
        input: Box<LogicalPlan>,
        group_exprs: Vec<(BoundExpr, String)>,
        aggregates: Vec<BoundAggregate>,
    },
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        /// Equality pairs, left expression against right expression. These are
        /// what the hash join builds on.
        on: Vec<(BoundExpr, BoundExpr)>,
        /// Anything in the ON clause that is not an equality.
        filter: Option<BoundExpr>,
    },
    Sort {
        input: Box<LogicalPlan>,
        exprs: Vec<SortExpr>,
    },
    Limit {
        input: Box<LogicalPlan>,
        limit: Option<usize>,
        offset: usize,
    },
    Distinct {
        input: Box<LogicalPlan>,
    },
    /// Literal rows, from `INSERT ... VALUES` or a `SELECT` with no `FROM`.
    Values {
        schema: Arc<Schema>,
        rows: Vec<Vec<BoundExpr>>,
    },
}

impl LogicalPlan {
    /// The schema this node produces.
    pub fn schema(&self) -> Result<Arc<Schema>> {
        Ok(match self {
            LogicalPlan::Scan {
                source_schema,
                projection,
                ..
            } => match projection {
                None => Arc::clone(source_schema),
                Some(indices) => Arc::new(source_schema.project(indices)?),
            },
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input } => input.schema()?,
            LogicalPlan::Projection { exprs, .. } => Arc::new(Schema::new(
                exprs
                    .iter()
                    .map(|(e, name)| Field::new(name, e.data_type(), true))
                    .collect(),
            )),
            LogicalPlan::Aggregate {
                group_exprs,
                aggregates,
                ..
            } => {
                let mut fields: Vec<Field> = group_exprs
                    .iter()
                    .map(|(e, name)| Field::new(name, e.data_type(), true))
                    .collect();
                fields.extend(
                    aggregates
                        .iter()
                        .map(|a| Field::new(&a.output_name, a.data_type, true)),
                );
                Arc::new(Schema::new(fields))
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                ..
            } => {
                // An outer join can produce NULLs on the side that did not
                // match, so those columns become nullable whatever the source
                // said.
                let (l, r) = (left.schema()?, right.schema()?);
                let widen_left = matches!(join_type, JoinType::Right | JoinType::Full);
                let widen_right = matches!(join_type, JoinType::Left | JoinType::Full);
                let mut fields: Vec<Field> = l
                    .fields()
                    .iter()
                    .map(|f| Field::new(&f.name, f.data_type, f.nullable || widen_left))
                    .collect();
                fields.extend(
                    r.fields()
                        .iter()
                        .map(|f| Field::new(&f.name, f.data_type, f.nullable || widen_right)),
                );
                Arc::new(Schema::new(fields))
            }
            LogicalPlan::Values { schema, .. } => Arc::clone(schema),
        })
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan { .. } | LogicalPlan::Values { .. } => vec![],
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Projection { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input } => vec![input],
            LogicalPlan::Join { left, right, .. } => vec![left, right],
        }
    }

    /// Replaces this node's children, keeping everything else. The optimiser
    /// rebuilds trees bottom-up through this.
    pub fn with_children(&self, mut new: Vec<LogicalPlan>) -> Result<LogicalPlan> {
        let expected = self.children().len();
        if new.len() != expected {
            return Err(Error::internal(format!(
                "{} expects {expected} children, got {}",
                self.name(),
                new.len()
            )));
        }
        Ok(match self {
            LogicalPlan::Scan { .. } | LogicalPlan::Values { .. } => self.clone(),
            LogicalPlan::Filter { predicate, .. } => LogicalPlan::Filter {
                input: Box::new(new.remove(0)),
                predicate: predicate.clone(),
            },
            LogicalPlan::Projection { exprs, .. } => LogicalPlan::Projection {
                input: Box::new(new.remove(0)),
                exprs: exprs.clone(),
            },
            LogicalPlan::Aggregate {
                group_exprs,
                aggregates,
                ..
            } => LogicalPlan::Aggregate {
                input: Box::new(new.remove(0)),
                group_exprs: group_exprs.clone(),
                aggregates: aggregates.clone(),
            },
            LogicalPlan::Sort { exprs, .. } => LogicalPlan::Sort {
                input: Box::new(new.remove(0)),
                exprs: exprs.clone(),
            },
            LogicalPlan::Limit { limit, offset, .. } => LogicalPlan::Limit {
                input: Box::new(new.remove(0)),
                limit: *limit,
                offset: *offset,
            },
            LogicalPlan::Distinct { .. } => LogicalPlan::Distinct {
                input: Box::new(new.remove(0)),
            },
            LogicalPlan::Join {
                join_type,
                on,
                filter,
                ..
            } => {
                let left = Box::new(new.remove(0));
                let right = Box::new(new.remove(0));
                LogicalPlan::Join {
                    left,
                    right,
                    join_type: *join_type,
                    on: on.clone(),
                    filter: filter.clone(),
                }
            }
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            LogicalPlan::Scan { .. } => "Scan",
            LogicalPlan::Filter { .. } => "Filter",
            LogicalPlan::Projection { .. } => "Projection",
            LogicalPlan::Aggregate { .. } => "Aggregate",
            LogicalPlan::Join { .. } => "Join",
            LogicalPlan::Sort { .. } => "Sort",
            LogicalPlan::Limit { .. } => "Limit",
            LogicalPlan::Distinct { .. } => "Distinct",
            LogicalPlan::Values { .. } => "Values",
        }
    }

    /// One line describing this node, without its children. This is what
    /// `EXPLAIN` prints.
    pub fn describe(&self) -> String {
        match self {
            LogicalPlan::Scan {
                table,
                source_schema,
                projection,
                pushed_filters,
                ..
            } => {
                let cols = match projection {
                    None => "*".to_string(),
                    Some(ix) => ix
                        .iter()
                        .filter_map(|i| source_schema.field(*i).ok())
                        .map(|f| f.name.clone())
                        .collect::<Vec<_>>()
                        .join(", "),
                };
                let mut s = format!("Scan: {table} [{cols}]");
                if !pushed_filters.is_empty() {
                    let fs: Vec<String> = pushed_filters.iter().map(BoundExpr::to_string).collect();
                    s.push_str(&format!(", pushed: {}", fs.join(" AND ")));
                }
                s
            }
            LogicalPlan::Filter { predicate, .. } => format!("Filter: {predicate}"),
            LogicalPlan::Projection { exprs, .. } => {
                let items: Vec<String> = exprs
                    .iter()
                    .map(|(e, n)| {
                        if &e.output_name() == n {
                            e.to_string()
                        } else {
                            format!("{e} AS {n}")
                        }
                    })
                    .collect();
                format!("Projection: {}", items.join(", "))
            }
            LogicalPlan::Aggregate {
                group_exprs,
                aggregates,
                ..
            } => {
                let g: Vec<String> = group_exprs.iter().map(|(e, _)| e.to_string()).collect();
                let a: Vec<String> = aggregates.iter().map(BoundAggregate::to_string).collect();
                if g.is_empty() {
                    format!("Aggregate: {}", a.join(", "))
                } else {
                    format!("Aggregate: group by [{}], {}", g.join(", "), a.join(", "))
                }
            }
            LogicalPlan::Join {
                join_type,
                on,
                filter,
                ..
            } => {
                let keys: Vec<String> = on.iter().map(|(l, r)| format!("{l} = {r}")).collect();
                let mut s = format!("{join_type} Join");
                if !keys.is_empty() {
                    s.push_str(&format!(": on {}", keys.join(" AND ")));
                }
                if let Some(f) = filter {
                    s.push_str(&format!(", filter {f}"));
                }
                s
            }
            LogicalPlan::Sort { exprs, .. } => {
                let items: Vec<String> = exprs
                    .iter()
                    .map(|s| {
                        format!(
                            "{} {}{}",
                            s.expr,
                            if s.ascending { "ASC" } else { "DESC" },
                            if s.nulls_first {
                                " NULLS FIRST"
                            } else {
                                " NULLS LAST"
                            }
                        )
                    })
                    .collect();
                format!("Sort: {}", items.join(", "))
            }
            LogicalPlan::Limit { limit, offset, .. } => match (limit, offset) {
                (Some(l), 0) => format!("Limit: {l}"),
                (Some(l), o) => format!("Limit: {l}, offset {o}"),
                (None, o) => format!("Offset: {o}"),
            },
            LogicalPlan::Distinct { .. } => "Distinct".to_string(),
            LogicalPlan::Values { rows, .. } => format!("Values: {} rows", rows.len()),
        }
    }

    /// The whole tree, indented.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        self.explain_into(&mut out, 0);
        out
    }

    fn explain_into(&self, out: &mut String, depth: usize) {
        for _ in 0..depth {
            out.push_str("  ");
        }
        out.push_str(&self.describe());
        out.push('\n');
        for c in self.children() {
            c.explain_into(out, depth + 1);
        }
    }

    /// Number of nodes in the tree, used by tests to assert a rule removed one.
    pub fn node_count(&self) -> usize {
        1 + self
            .children()
            .iter()
            .map(|c| c.node_count())
            .sum::<usize>()
    }
}

impl fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.explain().trim_end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::AggFunc;
    use qf_common::{DataType, Value};
    use qf_sql::ast::BinaryOp;

    fn table_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Float64, true),
        ]))
    }

    fn scan() -> LogicalPlan {
        LogicalPlan::Scan {
            table: "sales".into(),
            source_schema: table_schema(),
            projection: None,
            pushed_filters: vec![],
            stats: TableStats::default(),
        }
    }

    #[test]
    fn a_scan_reports_its_table_schema_and_a_projected_scan_reports_a_subset() {
        assert_eq!(scan().schema().unwrap().len(), 3);
        let projected = LogicalPlan::Scan {
            table: "sales".into(),
            source_schema: table_schema(),
            projection: Some(vec![2, 0]),
            pushed_filters: vec![],
            stats: TableStats::default(),
        };
        let s = projected.schema().unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.field(0).unwrap().name, "amount");
        assert_eq!(s.field(1).unwrap().name, "id");
    }

    #[test]
    fn filters_sorts_limits_and_distinct_pass_their_input_schema_through() {
        let p = BoundExpr::binary(
            BoundExpr::column(0, "id", DataType::Int64),
            BinaryOp::Gt,
            BoundExpr::Literal(Value::Int64(1)),
        )
        .unwrap();
        for node in [
            LogicalPlan::Filter {
                input: Box::new(scan()),
                predicate: p.clone(),
            },
            LogicalPlan::Sort {
                input: Box::new(scan()),
                exprs: vec![],
            },
            LogicalPlan::Limit {
                input: Box::new(scan()),
                limit: Some(1),
                offset: 0,
            },
            LogicalPlan::Distinct {
                input: Box::new(scan()),
            },
        ] {
            assert_eq!(node.schema().unwrap().len(), 3, "{}", node.name());
        }
    }

    #[test]
    fn a_projection_names_its_output_columns() {
        let p = LogicalPlan::Projection {
            input: Box::new(scan()),
            exprs: vec![(
                BoundExpr::binary(
                    BoundExpr::column(2, "amount", DataType::Float64),
                    BinaryOp::Multiply,
                    BoundExpr::Literal(Value::Int64(2)),
                )
                .unwrap(),
                "doubled".into(),
            )],
        };
        let s = p.schema().unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.field(0).unwrap().name, "doubled");
        assert_eq!(s.field(0).unwrap().data_type, DataType::Float64);
    }

    #[test]
    fn an_aggregate_outputs_its_group_columns_then_its_aggregates() {
        let a = LogicalPlan::Aggregate {
            input: Box::new(scan()),
            group_exprs: vec![(
                BoundExpr::column(1, "region", DataType::Utf8),
                "region".into(),
            )],
            aggregates: vec![BoundAggregate {
                func: AggFunc::Count,
                arg: None,
                distinct: false,
                output_name: "n".into(),
                data_type: DataType::Int64,
            }],
        };
        let s = a.schema().unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.field(0).unwrap().name, "region");
        assert_eq!(s.field(1).unwrap().name, "n");
        assert_eq!(s.field(1).unwrap().data_type, DataType::Int64);
    }

    #[test]
    fn an_outer_join_widens_the_nullability_of_the_side_that_may_not_match() {
        let join = |t| LogicalPlan::Join {
            left: Box::new(scan()),
            right: Box::new(scan()),
            join_type: t,
            on: vec![],
            filter: None,
        };
        // `id` is declared NOT NULL on both sides.
        let inner = join(JoinType::Inner).schema().unwrap();
        assert!(!inner.field(0).unwrap().nullable);
        assert!(!inner.field(3).unwrap().nullable);

        let left = join(JoinType::Left).schema().unwrap();
        assert!(!left.field(0).unwrap().nullable);
        assert!(left.field(3).unwrap().nullable, "right side may be padded");

        let right = join(JoinType::Right).schema().unwrap();
        assert!(right.field(0).unwrap().nullable, "left side may be padded");

        let full = join(JoinType::Full).schema().unwrap();
        assert!(full.field(0).unwrap().nullable);
        assert!(full.field(3).unwrap().nullable);
        assert_eq!(full.len(), 6);
    }

    #[test]
    fn children_can_be_swapped_out_without_losing_the_node_s_own_settings() {
        let filter = LogicalPlan::Filter {
            input: Box::new(scan()),
            predicate: BoundExpr::Literal(Value::Boolean(true)),
        };
        let rebuilt = filter.with_children(vec![scan()]).unwrap();
        assert_eq!(rebuilt, filter);
        assert!(filter.with_children(vec![]).is_err());
        assert!(scan().with_children(vec![]).unwrap() == scan());
    }

    #[test]
    fn with_children_covers_every_node_shape() {
        let nodes = vec![
            LogicalPlan::Projection {
                input: Box::new(scan()),
                exprs: vec![(BoundExpr::column(0, "id", DataType::Int64), "id".into())],
            },
            LogicalPlan::Aggregate {
                input: Box::new(scan()),
                group_exprs: vec![],
                aggregates: vec![],
            },
            LogicalPlan::Sort {
                input: Box::new(scan()),
                exprs: vec![SortExpr {
                    expr: BoundExpr::column(0, "id", DataType::Int64),
                    ascending: true,
                    nulls_first: false,
                }],
            },
            LogicalPlan::Limit {
                input: Box::new(scan()),
                limit: Some(3),
                offset: 1,
            },
            LogicalPlan::Distinct {
                input: Box::new(scan()),
            },
        ];
        for n in nodes {
            let children: Vec<LogicalPlan> = n.children().into_iter().cloned().collect();
            assert_eq!(n.with_children(children).unwrap(), n, "{}", n.name());
        }
        let j = LogicalPlan::Join {
            left: Box::new(scan()),
            right: Box::new(scan()),
            join_type: JoinType::Inner,
            on: vec![],
            filter: None,
        };
        assert_eq!(j.with_children(vec![scan(), scan()]).unwrap(), j);
    }

    #[test]
    fn explain_indents_children_under_their_parent() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan()),
                predicate: BoundExpr::Literal(Value::Boolean(true)),
            }),
            limit: Some(10),
            offset: 0,
        };
        let text = plan.explain();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "Limit: 10");
        assert_eq!(lines[1], "  Filter: true");
        assert_eq!(lines[2], "    Scan: sales [*]");
        assert_eq!(plan.node_count(), 3);
        assert_eq!(plan.to_string(), text.trim_end());
    }

    #[test]
    fn every_node_describes_itself() {
        let scan_with_pushdown = LogicalPlan::Scan {
            table: "sales".into(),
            source_schema: table_schema(),
            projection: Some(vec![0]),
            pushed_filters: vec![BoundExpr::binary(
                BoundExpr::column(0, "id", DataType::Int64),
                BinaryOp::Gt,
                BoundExpr::Literal(Value::Int64(5)),
            )
            .unwrap()],
            stats: TableStats::default(),
        };
        assert!(scan_with_pushdown.describe().contains("pushed:"));
        assert!(scan_with_pushdown.describe().contains("[id]"));

        assert_eq!(
            LogicalPlan::Values {
                schema: table_schema(),
                rows: vec![vec![], vec![]]
            }
            .describe(),
            "Values: 2 rows"
        );
        assert_eq!(
            LogicalPlan::Limit {
                input: Box::new(scan()),
                limit: None,
                offset: 4
            }
            .describe(),
            "Offset: 4"
        );
        assert_eq!(
            LogicalPlan::Limit {
                input: Box::new(scan()),
                limit: Some(2),
                offset: 4
            }
            .describe(),
            "Limit: 2, offset 4"
        );
        assert_eq!(
            LogicalPlan::Distinct {
                input: Box::new(scan())
            }
            .describe(),
            "Distinct"
        );
        assert!(LogicalPlan::Sort {
            input: Box::new(scan()),
            exprs: vec![SortExpr {
                expr: BoundExpr::column(0, "id", DataType::Int64),
                ascending: false,
                nulls_first: true,
            }],
        }
        .describe()
        .contains("DESC NULLS FIRST"));
    }

    #[test]
    fn a_join_describes_its_keys_and_any_leftover_filter() {
        let j = LogicalPlan::Join {
            left: Box::new(scan()),
            right: Box::new(scan()),
            join_type: JoinType::Left,
            on: vec![(
                BoundExpr::column(0, "id", DataType::Int64),
                BoundExpr::column(3, "id", DataType::Int64),
            )],
            filter: Some(BoundExpr::Literal(Value::Boolean(true))),
        };
        let d = j.describe();
        assert!(d.starts_with("LEFT Join: on"));
        assert!(d.contains("filter true"));
        assert_eq!(
            LogicalPlan::Join {
                left: Box::new(scan()),
                right: Box::new(scan()),
                join_type: JoinType::Cross,
                on: vec![],
                filter: None,
            }
            .describe(),
            "CROSS Join"
        );
    }

    #[test]
    fn a_projection_shows_an_alias_only_when_it_differs_from_the_expression() {
        let p = LogicalPlan::Projection {
            input: Box::new(scan()),
            exprs: vec![
                (BoundExpr::column(0, "id", DataType::Int64), "id".into()),
                (BoundExpr::column(1, "region", DataType::Utf8), "r".into()),
            ],
        };
        let d = p.describe();
        assert!(d.contains("id#0,"));
        assert!(d.contains("region#1 AS r"));
    }

    #[test]
    fn an_aggregate_without_grouping_describes_only_its_aggregates() {
        let a = LogicalPlan::Aggregate {
            input: Box::new(scan()),
            group_exprs: vec![],
            aggregates: vec![BoundAggregate {
                func: AggFunc::Count,
                arg: None,
                distinct: false,
                output_name: "n".into(),
                data_type: DataType::Int64,
            }],
        };
        assert_eq!(a.describe(), "Aggregate: count(*)");
    }
}
