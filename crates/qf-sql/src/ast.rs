//! The syntax tree the parser produces and the binder consumes.
//!
//! Nothing here is resolved: a `Column` is a name the user typed, not a
//! position, and a `Function` is a name, not an aggregate. Keeping the AST
//! purely syntactic means the binder is the single place where "does this
//! column exist" is answered, and error messages about missing columns all
//! come from one place.

use qf_common::{DataType, Value};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

impl BinaryOp {
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::LtEq
                | BinaryOp::Gt
                | BinaryOp::GtEq
        )
    }

    pub fn is_arithmetic(self) -> bool {
        matches!(
            self,
            BinaryOp::Plus
                | BinaryOp::Minus
                | BinaryOp::Multiply
                | BinaryOp::Divide
                | BinaryOp::Modulo
        )
    }

    pub fn is_logical(self) -> bool {
        matches!(self, BinaryOp::And | BinaryOp::Or)
    }

    /// Mirrors a comparison so that `5 > x` can be rewritten as `x < 5` — the
    /// form the optimiser needs before it can push a predicate into a scan.
    pub fn swap_operands(self) -> Option<BinaryOp> {
        Some(match self {
            BinaryOp::Eq => BinaryOp::Eq,
            BinaryOp::NotEq => BinaryOp::NotEq,
            BinaryOp::Lt => BinaryOp::Gt,
            BinaryOp::LtEq => BinaryOp::GtEq,
            BinaryOp::Gt => BinaryOp::Lt,
            BinaryOp::GtEq => BinaryOp::LtEq,
            _ => return None,
        })
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinaryOp::Plus => "+",
            BinaryOp::Minus => "-",
            BinaryOp::Multiply => "*",
            BinaryOp::Divide => "/",
            BinaryOp::Modulo => "%",
            BinaryOp::Eq => "=",
            BinaryOp::NotEq => "<>",
            BinaryOp::Lt => "<",
            BinaryOp::LtEq => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::GtEq => ">=",
            BinaryOp::And => "AND",
            BinaryOp::Or => "OR",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => f.write_str("-"),
            UnaryOp::Not => f.write_str("NOT "),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// `col` or `table.col`, as written.
    Column {
        table: Option<String>,
        name: String,
    },
    Literal(Value),
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    /// `f(args)` — resolved to an aggregate or rejected by the binder.
    Function {
        name: String,
        args: Vec<Expr>,
        distinct: bool,
    },
    Cast {
        expr: Box<Expr>,
        data_type: DataType,
    },
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        negated: bool,
    },
    Case {
        operand: Option<Box<Expr>>,
        branches: Vec<(Expr, Expr)>,
        else_result: Option<Box<Expr>>,
    },
    /// The `*` in `count(*)`.
    Wildcard,
}

impl Expr {
    pub fn binary(left: Expr, op: BinaryOp, right: Expr) -> Expr {
        Expr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    pub fn column(name: impl Into<String>) -> Expr {
        Expr::Column {
            table: None,
            name: name.into(),
        }
    }

    pub fn qualified(table: impl Into<String>, name: impl Into<String>) -> Expr {
        Expr::Column {
            table: Some(table.into()),
            name: name.into(),
        }
    }

    /// Splits a chain of `AND`s into its parts. Conjunct-at-a-time is the
    /// unit predicate pushdown works in: `a AND b` may push `a` into one side
    /// of a join and `b` into the other.
    pub fn split_conjuncts(&self) -> Vec<Expr> {
        match self {
            Expr::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => {
                let mut out = left.split_conjuncts();
                out.extend(right.split_conjuncts());
                out
            }
            other => vec![other.clone()],
        }
    }

    /// Rebuilds a conjunction from its parts.
    pub fn join_conjuncts(mut parts: Vec<Expr>) -> Option<Expr> {
        let first = if parts.is_empty() {
            return None;
        } else {
            parts.remove(0)
        };
        Some(
            parts
                .into_iter()
                .fold(first, |acc, p| Expr::binary(acc, BinaryOp::And, p)),
        )
    }

    /// Every column this expression reads, in the order encountered.
    pub fn columns(&self) -> Vec<(Option<String>, String)> {
        let mut out = Vec::new();
        self.walk(&mut |e| {
            if let Expr::Column { table, name } = e {
                out.push((table.clone(), name.clone()));
            }
        });
        out
    }

    pub fn contains_aggregate(&self) -> bool {
        let mut found = false;
        self.walk(&mut |e| {
            if let Expr::Function { name, .. } = e {
                if is_aggregate_name(name) {
                    found = true;
                }
            }
        });
        found
    }

    /// Pre-order walk over every subexpression, including this one.
    pub fn walk(&self, f: &mut impl FnMut(&Expr)) {
        f(self);
        match self {
            Expr::Column { .. } | Expr::Literal(_) | Expr::Wildcard => {}
            Expr::Binary { left, right, .. } => {
                left.walk(f);
                right.walk(f);
            }
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::IsNull { expr, .. } => {
                expr.walk(f)
            }
            Expr::Function { args, .. } => args.iter().for_each(|a| a.walk(f)),
            Expr::Between {
                expr, low, high, ..
            } => {
                expr.walk(f);
                low.walk(f);
                high.walk(f);
            }
            Expr::InList { expr, list, .. } => {
                expr.walk(f);
                list.iter().for_each(|e| e.walk(f));
            }
            Expr::Like { expr, pattern, .. } => {
                expr.walk(f);
                pattern.walk(f);
            }
            Expr::Case {
                operand,
                branches,
                else_result,
            } => {
                if let Some(o) = operand {
                    o.walk(f);
                }
                for (w, t) in branches {
                    w.walk(f);
                    t.walk(f);
                }
                if let Some(e) = else_result {
                    e.walk(f);
                }
            }
        }
    }
}

pub fn is_aggregate_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "count" | "sum" | "min" | "max" | "avg"
    )
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column { table: None, name } => f.write_str(name),
            Expr::Column {
                table: Some(t),
                name,
            } => write!(f, "{t}.{name}"),
            Expr::Literal(Value::Utf8(s)) => write!(f, "'{s}'"),
            Expr::Literal(v) => write!(f, "{v}"),
            Expr::Binary { left, op, right } => write!(f, "({left} {op} {right})"),
            Expr::Unary { op, expr } => write!(f, "{op}{expr}"),
            Expr::Function {
                name,
                args,
                distinct,
            } => {
                let inner: Vec<String> = args.iter().map(Expr::to_string).collect();
                let d = if *distinct { "DISTINCT " } else { "" };
                write!(f, "{name}({d}{})", inner.join(", "))
            }
            Expr::Cast { expr, data_type } => write!(f, "CAST({expr} AS {data_type})"),
            Expr::IsNull { expr, negated } => {
                let not = if *negated { "NOT " } else { "" };
                write!(f, "{expr} IS {not}NULL")
            }
            Expr::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let not = if *negated { "NOT " } else { "" };
                write!(f, "{expr} {not}BETWEEN {low} AND {high}")
            }
            Expr::InList {
                expr,
                list,
                negated,
            } => {
                let not = if *negated { "NOT " } else { "" };
                let items: Vec<String> = list.iter().map(Expr::to_string).collect();
                write!(f, "{expr} {not}IN ({})", items.join(", "))
            }
            Expr::Like {
                expr,
                pattern,
                negated,
            } => {
                let not = if *negated { "NOT " } else { "" };
                write!(f, "{expr} {not}LIKE {pattern}")
            }
            Expr::Case {
                operand,
                branches,
                else_result,
            } => {
                write!(f, "CASE")?;
                if let Some(o) = operand {
                    write!(f, " {o}")?;
                }
                for (w, t) in branches {
                    write!(f, " WHEN {w} THEN {t}")?;
                }
                if let Some(e) = else_result {
                    write!(f, " ELSE {e}")?;
                }
                f.write_str(" END")
            }
            Expr::Wildcard => f.write_str("*"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    /// `*`
    Wildcard,
    /// `t.*`
    QualifiedWildcard(String),
    Expr {
        expr: Expr,
        alias: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            JoinType::Inner => "INNER",
            JoinType::Left => "LEFT",
            JoinType::Right => "RIGHT",
            JoinType::Full => "FULL",
            JoinType::Cross => "CROSS",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TableRef {
    Table {
        name: String,
        alias: Option<String>,
    },
    Join {
        left: Box<TableRef>,
        right: Box<TableRef>,
        join_type: JoinType,
        on: Option<Expr>,
    },
}

impl TableRef {
    /// The names a column can be qualified with in this FROM clause — the
    /// alias when there is one, otherwise the table name.
    pub fn visible_names(&self) -> Vec<String> {
        match self {
            TableRef::Table { name, alias } => vec![alias.clone().unwrap_or_else(|| name.clone())],
            TableRef::Join { left, right, .. } => {
                let mut v = left.visible_names();
                v.extend(right.visible_names());
                v
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderByExpr {
    pub expr: Expr,
    pub ascending: bool,
    /// `NULLS FIRST` / `NULLS LAST`. Defaults to SQL's convention: nulls last
    /// when ascending, first when descending.
    pub nulls_first: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub distinct: bool,
    pub projection: Vec<SelectItem>,
    pub from: Option<TableRef>,
    pub selection: Option<Expr>,
    pub group_by: Vec<Expr>,
    pub having: Option<Expr>,
    pub order_by: Vec<OrderByExpr>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// Boxed because a `Query` is an order of magnitude larger than any other
    /// statement, and every `Statement` would otherwise pay for it.
    Query(Box<Query>),
    CreateTable {
        name: String,
        columns: Vec<ColumnDef>,
    },
    Insert {
        table: String,
        rows: Vec<Vec<Expr>>,
    },
    /// `COPY t FROM 'path.csv'` — bulk load, the only way large data gets in.
    Copy {
        table: String,
        path: String,
    },
    DropTable {
        name: String,
    },
    Explain {
        analyze: bool,
        query: Box<Query>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conjunctions_split_and_rejoin() {
        let e = Expr::binary(
            Expr::binary(Expr::column("a"), BinaryOp::And, Expr::column("b")),
            BinaryOp::And,
            Expr::column("c"),
        );
        let parts = e.split_conjuncts();
        assert_eq!(parts.len(), 3);
        assert_eq!(
            Expr::join_conjuncts(parts).unwrap().to_string(),
            e.to_string()
        );
    }

    #[test]
    fn a_non_conjunction_splits_into_one_part() {
        let e = Expr::binary(Expr::column("a"), BinaryOp::Or, Expr::column("b"));
        assert_eq!(e.split_conjuncts().len(), 1);
        assert!(Expr::join_conjuncts(vec![]).is_none());
    }

    #[test]
    fn comparisons_mirror_when_their_operands_swap() {
        assert_eq!(BinaryOp::Lt.swap_operands(), Some(BinaryOp::Gt));
        assert_eq!(BinaryOp::GtEq.swap_operands(), Some(BinaryOp::LtEq));
        assert_eq!(BinaryOp::Eq.swap_operands(), Some(BinaryOp::Eq));
        assert_eq!(BinaryOp::NotEq.swap_operands(), Some(BinaryOp::NotEq));
        assert_eq!(BinaryOp::Plus.swap_operands(), None);
        assert_eq!(BinaryOp::And.swap_operands(), None);
    }

    #[test]
    fn operators_classify_themselves() {
        assert!(BinaryOp::Lt.is_comparison());
        assert!(BinaryOp::Multiply.is_arithmetic());
        assert!(BinaryOp::Or.is_logical());
        assert!(!BinaryOp::Or.is_comparison());
        assert!(!BinaryOp::Eq.is_arithmetic());
    }

    #[test]
    fn columns_are_collected_from_anywhere_in_an_expression() {
        let e = Expr::Case {
            operand: None,
            branches: vec![(
                Expr::binary(
                    Expr::column("a"),
                    BinaryOp::Gt,
                    Expr::Literal(Value::Int64(1)),
                ),
                Expr::qualified("t", "b"),
            )],
            else_result: Some(Box::new(Expr::column("c"))),
        };
        let cols = e.columns();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[1], (Some("t".to_string()), "b".to_string()));
    }

    #[test]
    fn aggregates_are_found_however_deeply_nested() {
        let e = Expr::binary(
            Expr::Literal(Value::Int64(1)),
            BinaryOp::Plus,
            Expr::Function {
                name: "sum".into(),
                args: vec![Expr::column("x")],
                distinct: false,
            },
        );
        assert!(e.contains_aggregate());
        assert!(!Expr::column("x").contains_aggregate());
        assert!(!Expr::Function {
            name: "upper".into(),
            args: vec![],
            distinct: false
        }
        .contains_aggregate());
    }

    #[test]
    fn the_walk_visits_every_node_shape() {
        let exprs = vec![
            Expr::Wildcard,
            Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(Expr::column("a")),
            },
            Expr::Cast {
                expr: Box::new(Expr::column("a")),
                data_type: DataType::Int64,
            },
            Expr::IsNull {
                expr: Box::new(Expr::column("a")),
                negated: true,
            },
            Expr::Between {
                expr: Box::new(Expr::column("a")),
                low: Box::new(Expr::column("b")),
                high: Box::new(Expr::column("c")),
                negated: false,
            },
            Expr::InList {
                expr: Box::new(Expr::column("a")),
                list: vec![Expr::column("b")],
                negated: false,
            },
            Expr::Like {
                expr: Box::new(Expr::column("a")),
                pattern: Box::new(Expr::Literal(Value::Utf8("x%".into()))),
                negated: false,
            },
            Expr::Case {
                operand: Some(Box::new(Expr::column("a"))),
                branches: vec![(Expr::column("b"), Expr::column("c"))],
                else_result: None,
            },
        ];
        for e in exprs {
            let mut seen = 0;
            e.walk(&mut |_| seen += 1);
            assert!(seen >= 1, "{e} visited nothing");
        }
    }

    #[test]
    fn expressions_render_back_to_readable_sql() {
        assert_eq!(
            Expr::binary(
                Expr::column("a"),
                BinaryOp::Plus,
                Expr::Literal(Value::Int64(1))
            )
            .to_string(),
            "(a + 1)"
        );
        assert_eq!(Expr::qualified("t", "c").to_string(), "t.c");
        assert_eq!(Expr::Literal(Value::Utf8("hi".into())).to_string(), "'hi'");
        assert_eq!(
            Expr::Function {
                name: "count".into(),
                args: vec![Expr::Wildcard],
                distinct: false
            }
            .to_string(),
            "count(*)"
        );
        assert_eq!(
            Expr::Function {
                name: "count".into(),
                args: vec![Expr::column("x")],
                distinct: true
            }
            .to_string(),
            "count(DISTINCT x)"
        );
        assert_eq!(
            Expr::Cast {
                expr: Box::new(Expr::column("a")),
                data_type: DataType::Float64
            }
            .to_string(),
            "CAST(a AS FLOAT64)"
        );
        assert_eq!(
            Expr::IsNull {
                expr: Box::new(Expr::column("a")),
                negated: true
            }
            .to_string(),
            "a IS NOT NULL"
        );
        assert_eq!(
            Expr::Unary {
                op: UnaryOp::Neg,
                expr: Box::new(Expr::column("a"))
            }
            .to_string(),
            "-a"
        );
    }

    #[test]
    fn the_remaining_expression_shapes_render_too() {
        assert_eq!(
            Expr::Between {
                expr: Box::new(Expr::column("a")),
                low: Box::new(Expr::Literal(Value::Int64(1))),
                high: Box::new(Expr::Literal(Value::Int64(9))),
                negated: true
            }
            .to_string(),
            "a NOT BETWEEN 1 AND 9"
        );
        assert_eq!(
            Expr::InList {
                expr: Box::new(Expr::column("a")),
                list: vec![
                    Expr::Literal(Value::Int64(1)),
                    Expr::Literal(Value::Int64(2))
                ],
                negated: false
            }
            .to_string(),
            "a IN (1, 2)"
        );
        assert_eq!(
            Expr::Like {
                expr: Box::new(Expr::column("a")),
                pattern: Box::new(Expr::Literal(Value::Utf8("x%".into()))),
                negated: true
            }
            .to_string(),
            "a NOT LIKE 'x%'"
        );
        assert_eq!(
            Expr::Case {
                operand: Some(Box::new(Expr::column("a"))),
                branches: vec![(
                    Expr::Literal(Value::Int64(1)),
                    Expr::Literal(Value::Int64(2))
                )],
                else_result: Some(Box::new(Expr::Literal(Value::Int64(3))))
            }
            .to_string(),
            "CASE a WHEN 1 THEN 2 ELSE 3 END"
        );
        assert_eq!(Expr::Wildcard.to_string(), "*");
    }

    #[test]
    fn a_from_clause_reports_the_names_columns_may_be_qualified_with() {
        let t = TableRef::Join {
            left: Box::new(TableRef::Table {
                name: "orders".into(),
                alias: Some("o".into()),
            }),
            right: Box::new(TableRef::Table {
                name: "customers".into(),
                alias: None,
            }),
            join_type: JoinType::Inner,
            on: None,
        };
        assert_eq!(t.visible_names(), vec!["o", "customers"]);
    }

    #[test]
    fn join_types_render_for_plan_output() {
        for j in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::Cross,
        ] {
            assert!(!j.to_string().is_empty());
        }
        assert_eq!(JoinType::Left.to_string(), "LEFT");
        assert_eq!(UnaryOp::Not.to_string(), "NOT ");
    }

    #[test]
    fn aggregate_names_are_recognised_case_insensitively() {
        for n in ["count", "SUM", "Min", "max", "avg"] {
            assert!(is_aggregate_name(n));
        }
        assert!(!is_aggregate_name("median"));
    }
}
