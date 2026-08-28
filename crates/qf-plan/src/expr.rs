//! Expressions after binding: every column is a position, every node knows its
//! type, and nothing is left to resolve at execution time.
//!
//! The split from the AST matters. `qf_sql::Expr` is what the user typed;
//! `BoundExpr` is what the engine will run. Once an expression is bound, the
//! evaluator never looks up a name, never asks whether a type is legal, and
//! never has to report a user-facing error about either — all of that
//! happened here, once, at plan time.

use qf_common::{DataType, Error, Result, Value};
use qf_sql::ast::{BinaryOp, UnaryOp};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum BoundExpr {
    /// A position in the input batch, not a name.
    Column {
        index: usize,
        name: String,
        data_type: DataType,
    },
    Literal(Value),
    Binary {
        left: Box<BoundExpr>,
        op: BinaryOp,
        right: Box<BoundExpr>,
        data_type: DataType,
    },
    Unary {
        op: UnaryOp,
        expr: Box<BoundExpr>,
        data_type: DataType,
    },
    Cast {
        expr: Box<BoundExpr>,
        data_type: DataType,
    },
    IsNull {
        expr: Box<BoundExpr>,
        negated: bool,
    },
    InList {
        expr: Box<BoundExpr>,
        list: Vec<BoundExpr>,
        negated: bool,
    },
    Like {
        expr: Box<BoundExpr>,
        pattern: Box<BoundExpr>,
        negated: bool,
    },
    Case {
        branches: Vec<(BoundExpr, BoundExpr)>,
        else_result: Option<Box<BoundExpr>>,
        data_type: DataType,
    },
}

impl BoundExpr {
    pub fn data_type(&self) -> DataType {
        match self {
            BoundExpr::Column { data_type, .. }
            | BoundExpr::Binary { data_type, .. }
            | BoundExpr::Unary { data_type, .. }
            | BoundExpr::Cast { data_type, .. }
            | BoundExpr::Case { data_type, .. } => *data_type,
            BoundExpr::IsNull { .. } | BoundExpr::Like { .. } | BoundExpr::InList { .. } => {
                DataType::Boolean
            }
            // A bare NULL literal has no type of its own; it takes the type of
            // whatever it is compared against, and Int64 is the harmless
            // default for the cases where nothing constrains it.
            BoundExpr::Literal(v) => v.data_type().unwrap_or(DataType::Int64),
        }
    }

    pub fn column(index: usize, name: impl Into<String>, data_type: DataType) -> BoundExpr {
        BoundExpr::Column {
            index,
            name: name.into(),
            data_type,
        }
    }

    /// Builds a binary node, inferring and checking its result type.
    ///
    /// This is where a query like `WHERE name > 3` is rejected — at plan time,
    /// before a single row is read.
    pub fn binary(left: BoundExpr, op: BinaryOp, right: BoundExpr) -> Result<BoundExpr> {
        let (lt, rt) = (left.data_type(), right.data_type());
        let data_type = if op.is_logical() {
            for (side, t) in [("left", lt), ("right", rt)] {
                if t != DataType::Boolean {
                    return Err(Error::typ(format!(
                        "{op} needs a boolean on its {side}, found {t}"
                    )));
                }
            }
            DataType::Boolean
        } else if op.is_comparison() {
            // NULL compares against anything; otherwise the types must reconcile.
            if !left.is_null_literal() && !right.is_null_literal() {
                DataType::unify(lt, rt)
                    .map_err(|_| Error::typ(format!("cannot compare {lt} with {rt} using {op}")))?;
            }
            DataType::Boolean
        } else {
            if !lt.is_numeric() || !rt.is_numeric() {
                return Err(Error::typ(format!(
                    "{op} needs numbers, found {lt} and {rt}"
                )));
            }
            // Division always widens: integer division silently truncating
            // `total / count` is a classic source of quietly wrong reports.
            if op == BinaryOp::Divide {
                DataType::Float64
            } else {
                DataType::unify(lt, rt)?
            }
        };
        Ok(BoundExpr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
            data_type,
        })
    }

    pub fn unary(op: UnaryOp, expr: BoundExpr) -> Result<BoundExpr> {
        let t = expr.data_type();
        let data_type = match op {
            UnaryOp::Neg => {
                if !t.is_numeric() {
                    return Err(Error::typ(format!("cannot negate {t}")));
                }
                t
            }
            UnaryOp::Not => {
                if t != DataType::Boolean {
                    return Err(Error::typ(format!("NOT needs a boolean, found {t}")));
                }
                DataType::Boolean
            }
        };
        Ok(BoundExpr::Unary {
            op,
            expr: Box::new(expr),
            data_type,
        })
    }

    /// All branches of a CASE must agree on a type, so the column it produces
    /// has one.
    pub fn case(
        branches: Vec<(BoundExpr, BoundExpr)>,
        else_result: Option<BoundExpr>,
    ) -> Result<BoundExpr> {
        if branches.is_empty() {
            return Err(Error::plan(
                "CASE needs at least one WHEN branch".to_string(),
            ));
        }
        for (when, _) in &branches {
            if when.data_type() != DataType::Boolean {
                return Err(Error::typ(format!(
                    "a CASE condition must be boolean, found {}",
                    when.data_type()
                )));
            }
        }
        let mut data_type = None;
        for result in branches.iter().map(|(_, t)| t).chain(else_result.iter()) {
            if result.is_null_literal() {
                continue;
            }
            data_type = Some(match data_type {
                None => result.data_type(),
                Some(t) => DataType::unify(t, result.data_type()).map_err(|_| {
                    Error::typ(format!(
                        "CASE branches disagree on type: {t} and {}",
                        result.data_type()
                    ))
                })?,
            });
        }
        Ok(BoundExpr::Case {
            branches,
            else_result: else_result.map(Box::new),
            data_type: data_type.unwrap_or(DataType::Int64),
        })
    }

    pub fn is_null_literal(&self) -> bool {
        matches!(self, BoundExpr::Literal(Value::Null))
    }

    pub fn is_literal(&self) -> bool {
        matches!(self, BoundExpr::Literal(_))
    }

    pub fn as_literal(&self) -> Option<&Value> {
        match self {
            BoundExpr::Literal(v) => Some(v),
            _ => None,
        }
    }

    /// Every column position this expression reads.
    pub fn column_indices(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.walk(&mut |e| {
            if let BoundExpr::Column { index, .. } = e {
                out.push(*index);
            }
        });
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn walk(&self, f: &mut impl FnMut(&BoundExpr)) {
        f(self);
        match self {
            BoundExpr::Column { .. } | BoundExpr::Literal(_) => {}
            BoundExpr::Binary { left, right, .. } => {
                left.walk(f);
                right.walk(f);
            }
            BoundExpr::Unary { expr, .. }
            | BoundExpr::Cast { expr, .. }
            | BoundExpr::IsNull { expr, .. } => expr.walk(f),
            BoundExpr::InList { expr, list, .. } => {
                expr.walk(f);
                list.iter().for_each(|e| e.walk(f));
            }
            BoundExpr::Like { expr, pattern, .. } => {
                expr.walk(f);
                pattern.walk(f);
            }
            BoundExpr::Case {
                branches,
                else_result,
                ..
            } => {
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

    /// Rebuilds the expression with every column index remapped.
    ///
    /// Needed whenever a rule changes the shape of an operator's input — a
    /// scan that stops reading columns it does not need, or a join whose
    /// inputs are reordered. Getting this wrong is silent: the query still
    /// runs and reads the wrong column.
    pub fn remap_columns(&self, map: &dyn Fn(usize) -> Option<usize>) -> Result<BoundExpr> {
        Ok(match self {
            BoundExpr::Column {
                index,
                name,
                data_type,
            } => {
                let new = map(*index).ok_or_else(|| {
                    Error::internal(format!(
                        "column `{name}` (index {index}) has no place in the new input"
                    ))
                })?;
                BoundExpr::Column {
                    index: new,
                    name: name.clone(),
                    data_type: *data_type,
                }
            }
            BoundExpr::Literal(v) => BoundExpr::Literal(v.clone()),
            BoundExpr::Binary {
                left,
                op,
                right,
                data_type,
            } => BoundExpr::Binary {
                left: Box::new(left.remap_columns(map)?),
                op: *op,
                right: Box::new(right.remap_columns(map)?),
                data_type: *data_type,
            },
            BoundExpr::Unary {
                op,
                expr,
                data_type,
            } => BoundExpr::Unary {
                op: *op,
                expr: Box::new(expr.remap_columns(map)?),
                data_type: *data_type,
            },
            BoundExpr::Cast { expr, data_type } => BoundExpr::Cast {
                expr: Box::new(expr.remap_columns(map)?),
                data_type: *data_type,
            },
            BoundExpr::IsNull { expr, negated } => BoundExpr::IsNull {
                expr: Box::new(expr.remap_columns(map)?),
                negated: *negated,
            },
            BoundExpr::InList {
                expr,
                list,
                negated,
            } => BoundExpr::InList {
                expr: Box::new(expr.remap_columns(map)?),
                list: list
                    .iter()
                    .map(|e| e.remap_columns(map))
                    .collect::<Result<Vec<_>>>()?,
                negated: *negated,
            },
            BoundExpr::Like {
                expr,
                pattern,
                negated,
            } => BoundExpr::Like {
                expr: Box::new(expr.remap_columns(map)?),
                pattern: Box::new(pattern.remap_columns(map)?),
                negated: *negated,
            },
            BoundExpr::Case {
                branches,
                else_result,
                data_type,
            } => BoundExpr::Case {
                branches: branches
                    .iter()
                    .map(|(w, t)| Ok((w.remap_columns(map)?, t.remap_columns(map)?)))
                    .collect::<Result<Vec<_>>>()?,
                else_result: match else_result {
                    Some(e) => Some(Box::new(e.remap_columns(map)?)),
                    None => None,
                },
                data_type: *data_type,
            },
        })
    }

    /// A readable name for the column this expression produces, used when the
    /// user did not supply an alias.
    pub fn output_name(&self) -> String {
        match self {
            BoundExpr::Column { name, .. } => name.clone(),
            other => other.to_string(),
        }
    }
}

impl fmt::Display for BoundExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundExpr::Column { name, index, .. } => write!(f, "{name}#{index}"),
            BoundExpr::Literal(Value::Utf8(s)) => write!(f, "'{s}'"),
            BoundExpr::Literal(v) => write!(f, "{v}"),
            BoundExpr::Binary {
                left, op, right, ..
            } => write!(f, "({left} {op} {right})"),
            BoundExpr::Unary { op, expr, .. } => write!(f, "{op}{expr}"),
            BoundExpr::Cast { expr, data_type } => write!(f, "CAST({expr} AS {data_type})"),
            BoundExpr::IsNull { expr, negated } => {
                let not = if *negated { "NOT " } else { "" };
                write!(f, "{expr} IS {not}NULL")
            }
            BoundExpr::InList {
                expr,
                list,
                negated,
            } => {
                let not = if *negated { "NOT " } else { "" };
                let items: Vec<String> = list.iter().map(BoundExpr::to_string).collect();
                write!(f, "{expr} {not}IN ({})", items.join(", "))
            }
            BoundExpr::Like {
                expr,
                pattern,
                negated,
            } => {
                let not = if *negated { "NOT " } else { "" };
                write!(f, "{expr} {not}LIKE {pattern}")
            }
            BoundExpr::Case {
                branches,
                else_result,
                ..
            } => {
                write!(f, "CASE")?;
                for (w, t) in branches {
                    write!(f, " WHEN {w} THEN {t}")?;
                }
                if let Some(e) = else_result {
                    write!(f, " ELSE {e}")?;
                }
                f.write_str(" END")
            }
        }
    }
}

/// The aggregate functions the engine implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

impl AggFunc {
    pub fn from_name(name: &str) -> Option<AggFunc> {
        Some(match name.to_ascii_lowercase().as_str() {
            "count" => AggFunc::Count,
            "sum" => AggFunc::Sum,
            "min" => AggFunc::Min,
            "max" => AggFunc::Max,
            "avg" => AggFunc::Avg,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            AggFunc::Count => "count",
            AggFunc::Sum => "sum",
            AggFunc::Min => "min",
            AggFunc::Max => "max",
            AggFunc::Avg => "avg",
        }
    }

    /// The type this aggregate produces over an input of `input` type.
    pub fn output_type(self, input: Option<DataType>) -> Result<DataType> {
        Ok(match self {
            AggFunc::Count => DataType::Int64,
            // avg of integers is a float — the same reasoning as division.
            AggFunc::Avg => DataType::Float64,
            AggFunc::Sum => match input {
                Some(t) if t.is_numeric() => t,
                Some(t) => return Err(Error::typ(format!("cannot sum {t}"))),
                None => return Err(Error::plan("sum() needs an argument".to_string())),
            },
            AggFunc::Min | AggFunc::Max => {
                input.ok_or_else(|| Error::plan(format!("{}() needs an argument", self.name())))?
            }
        })
    }

    /// Whether `f(*)` is meaningful. Only `count` is.
    pub fn accepts_wildcard(self) -> bool {
        self == AggFunc::Count
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundAggregate {
    pub func: AggFunc,
    /// `None` only for `count(*)`.
    pub arg: Option<BoundExpr>,
    pub distinct: bool,
    pub output_name: String,
    pub data_type: DataType,
}

impl fmt::Display for BoundAggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = if self.distinct { "DISTINCT " } else { "" };
        match &self.arg {
            Some(a) => write!(f, "{}({d}{a})", self.func.name()),
            None => write!(f, "{}(*)", self.func.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(i: usize, t: DataType) -> BoundExpr {
        BoundExpr::column(i, format!("c{i}"), t)
    }

    #[test]
    fn comparisons_produce_a_boolean_whatever_they_compare() {
        let e = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Lt,
            BoundExpr::Literal(Value::Float64(1.5)),
        )
        .unwrap();
        assert_eq!(e.data_type(), DataType::Boolean);
    }

    #[test]
    fn comparing_text_with_a_number_is_rejected_at_plan_time() {
        let err = BoundExpr::binary(
            col(0, DataType::Utf8),
            BinaryOp::Gt,
            BoundExpr::Literal(Value::Int64(3)),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Type(_)));
        assert!(err.to_string().contains("cannot compare"));
    }

    #[test]
    fn null_may_be_compared_with_anything() {
        assert!(BoundExpr::binary(
            col(0, DataType::Utf8),
            BinaryOp::Eq,
            BoundExpr::Literal(Value::Null)
        )
        .is_ok());
    }

    #[test]
    fn arithmetic_widens_int_and_float_and_refuses_text() {
        let e = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Plus,
            col(1, DataType::Float64),
        )
        .unwrap();
        assert_eq!(e.data_type(), DataType::Float64);
        assert!(BoundExpr::binary(
            col(0, DataType::Utf8),
            BinaryOp::Plus,
            col(1, DataType::Int64)
        )
        .is_err());
    }

    #[test]
    fn dividing_two_integers_yields_a_float() {
        // Integer division truncating `total / count` is a quiet way to report
        // the wrong number, so division always widens.
        let e = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Divide,
            col(1, DataType::Int64),
        )
        .unwrap();
        assert_eq!(e.data_type(), DataType::Float64);
    }

    #[test]
    fn logical_operators_demand_booleans_on_both_sides() {
        assert!(BoundExpr::binary(
            col(0, DataType::Boolean),
            BinaryOp::And,
            col(1, DataType::Boolean)
        )
        .is_ok());
        let err = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::And,
            col(1, DataType::Boolean),
        )
        .unwrap_err();
        assert!(err.to_string().contains("on its left"));
        assert!(BoundExpr::binary(
            col(0, DataType::Boolean),
            BinaryOp::Or,
            col(1, DataType::Int64)
        )
        .unwrap_err()
        .to_string()
        .contains("on its right"));
    }

    #[test]
    fn unary_operators_check_their_operand() {
        assert!(BoundExpr::unary(UnaryOp::Neg, col(0, DataType::Int64)).is_ok());
        assert!(BoundExpr::unary(UnaryOp::Neg, col(0, DataType::Utf8)).is_err());
        assert!(BoundExpr::unary(UnaryOp::Not, col(0, DataType::Boolean)).is_ok());
        assert!(BoundExpr::unary(UnaryOp::Not, col(0, DataType::Int64)).is_err());
    }

    #[test]
    fn case_branches_must_agree_on_a_type() {
        let ok = BoundExpr::case(
            vec![(col(0, DataType::Boolean), col(1, DataType::Int64))],
            Some(col(2, DataType::Float64)),
        )
        .unwrap();
        assert_eq!(ok.data_type(), DataType::Float64);

        assert!(BoundExpr::case(
            vec![(col(0, DataType::Boolean), col(1, DataType::Int64))],
            Some(col(2, DataType::Utf8)),
        )
        .unwrap_err()
        .to_string()
        .contains("disagree on type"));
    }

    #[test]
    fn a_case_condition_must_be_boolean_and_a_null_branch_does_not_decide_the_type() {
        assert!(BoundExpr::case(
            vec![(col(0, DataType::Int64), col(1, DataType::Int64))],
            None
        )
        .is_err());
        let e = BoundExpr::case(
            vec![(col(0, DataType::Boolean), BoundExpr::Literal(Value::Null))],
            Some(col(2, DataType::Utf8)),
        )
        .unwrap();
        assert_eq!(e.data_type(), DataType::Utf8);
        assert!(BoundExpr::case(vec![], None).is_err());
    }

    #[test]
    fn column_indices_are_collected_deduplicated_and_sorted() {
        let e = BoundExpr::binary(
            col(3, DataType::Int64),
            BinaryOp::Plus,
            BoundExpr::binary(
                col(1, DataType::Int64),
                BinaryOp::Plus,
                col(3, DataType::Int64),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(e.column_indices(), vec![1, 3]);
    }

    #[test]
    fn remapping_rewrites_every_column_position() {
        let e = BoundExpr::binary(
            col(5, DataType::Int64),
            BinaryOp::Lt,
            col(9, DataType::Int64),
        )
        .unwrap();
        let mapped = e
            .remap_columns(&|i| match i {
                5 => Some(0),
                9 => Some(1),
                _ => None,
            })
            .unwrap();
        assert_eq!(mapped.column_indices(), vec![0, 1]);
    }

    #[test]
    fn remapping_a_column_with_nowhere_to_go_is_an_error_not_a_silent_shift() {
        let e = col(7, DataType::Int64);
        assert!(e.remap_columns(&|_| None).is_err());
    }

    #[test]
    fn remapping_reaches_inside_every_expression_shape() {
        let shapes = vec![
            BoundExpr::Cast {
                expr: Box::new(col(4, DataType::Int64)),
                data_type: DataType::Float64,
            },
            BoundExpr::IsNull {
                expr: Box::new(col(4, DataType::Int64)),
                negated: false,
            },
            BoundExpr::InList {
                expr: Box::new(col(4, DataType::Int64)),
                list: vec![BoundExpr::Literal(Value::Int64(1))],
                negated: false,
            },
            BoundExpr::Like {
                expr: Box::new(col(4, DataType::Utf8)),
                pattern: Box::new(BoundExpr::Literal(Value::Utf8("a%".into()))),
                negated: false,
            },
            BoundExpr::unary(UnaryOp::Neg, col(4, DataType::Int64)).unwrap(),
            BoundExpr::case(
                vec![(
                    BoundExpr::IsNull {
                        expr: Box::new(col(4, DataType::Int64)),
                        negated: false,
                    },
                    col(4, DataType::Int64),
                )],
                Some(col(4, DataType::Int64)),
            )
            .unwrap(),
        ];
        for s in shapes {
            assert_eq!(s.column_indices(), vec![4], "{s}");
            let m = s.remap_columns(&|i| Some(i + 10)).unwrap();
            assert_eq!(m.column_indices(), vec![14], "{s}");
        }
    }

    #[test]
    fn bound_expressions_render_with_their_column_positions() {
        assert_eq!(col(2, DataType::Int64).to_string(), "c2#2");
        assert_eq!(
            BoundExpr::Literal(Value::Utf8("x".into())).to_string(),
            "'x'"
        );
        assert_eq!(
            BoundExpr::binary(
                col(0, DataType::Int64),
                BinaryOp::Plus,
                col(1, DataType::Int64)
            )
            .unwrap()
            .to_string(),
            "(c0#0 + c1#1)"
        );
        assert_eq!(
            BoundExpr::IsNull {
                expr: Box::new(col(0, DataType::Int64)),
                negated: true
            }
            .to_string(),
            "c0#0 IS NOT NULL"
        );
    }

    #[test]
    fn the_remaining_display_shapes_render() {
        assert!(BoundExpr::InList {
            expr: Box::new(col(0, DataType::Int64)),
            list: vec![BoundExpr::Literal(Value::Int64(1))],
            negated: true,
        }
        .to_string()
        .contains("NOT IN"));
        assert!(BoundExpr::Like {
            expr: Box::new(col(0, DataType::Utf8)),
            pattern: Box::new(BoundExpr::Literal(Value::Utf8("a%".into()))),
            negated: false,
        }
        .to_string()
        .contains("LIKE"));
        assert!(BoundExpr::Cast {
            expr: Box::new(col(0, DataType::Int64)),
            data_type: DataType::Utf8,
        }
        .to_string()
        .starts_with("CAST"));
        assert!(BoundExpr::case(
            vec![(col(0, DataType::Boolean), col(1, DataType::Int64))],
            Some(col(2, DataType::Int64))
        )
        .unwrap()
        .to_string()
        .contains("ELSE"));
        assert!(BoundExpr::unary(UnaryOp::Neg, col(0, DataType::Int64))
            .unwrap()
            .to_string()
            .starts_with('-'));
    }

    #[test]
    fn literals_and_output_names_are_reported() {
        let lit = BoundExpr::Literal(Value::Int64(3));
        assert!(lit.is_literal());
        assert_eq!(lit.as_literal(), Some(&Value::Int64(3)));
        assert!(!col(0, DataType::Int64).is_literal());
        assert!(col(0, DataType::Int64).as_literal().is_none());
        assert_eq!(col(0, DataType::Int64).output_name(), "c0");
        assert_eq!(lit.output_name(), "3");
        assert!(BoundExpr::Literal(Value::Null).is_null_literal());
        assert_eq!(BoundExpr::Literal(Value::Null).data_type(), DataType::Int64);
    }

    #[test]
    fn aggregate_output_types_follow_their_input() {
        assert_eq!(AggFunc::Count.output_type(None).unwrap(), DataType::Int64);
        assert_eq!(
            AggFunc::Avg.output_type(Some(DataType::Int64)).unwrap(),
            DataType::Float64
        );
        assert_eq!(
            AggFunc::Sum.output_type(Some(DataType::Int64)).unwrap(),
            DataType::Int64
        );
        assert_eq!(
            AggFunc::Max.output_type(Some(DataType::Utf8)).unwrap(),
            DataType::Utf8
        );
    }

    #[test]
    fn summing_text_or_aggregating_nothing_is_rejected() {
        assert!(AggFunc::Sum.output_type(Some(DataType::Utf8)).is_err());
        assert!(AggFunc::Sum.output_type(None).is_err());
        assert!(AggFunc::Min.output_type(None).is_err());
    }

    #[test]
    fn only_count_accepts_a_wildcard() {
        assert!(AggFunc::Count.accepts_wildcard());
        for f in [AggFunc::Sum, AggFunc::Min, AggFunc::Max, AggFunc::Avg] {
            assert!(!f.accepts_wildcard());
        }
    }

    #[test]
    fn aggregate_names_round_trip() {
        for f in [
            AggFunc::Count,
            AggFunc::Sum,
            AggFunc::Min,
            AggFunc::Max,
            AggFunc::Avg,
        ] {
            assert_eq!(AggFunc::from_name(f.name()), Some(f));
        }
        assert_eq!(AggFunc::from_name("COUNT"), Some(AggFunc::Count));
        assert_eq!(AggFunc::from_name("median"), None);
    }

    #[test]
    fn aggregates_render_for_plan_output() {
        let a = BoundAggregate {
            func: AggFunc::Sum,
            arg: Some(col(1, DataType::Int64)),
            distinct: true,
            output_name: "s".into(),
            data_type: DataType::Int64,
        };
        assert_eq!(a.to_string(), "sum(DISTINCT c1#1)");
        let c = BoundAggregate {
            func: AggFunc::Count,
            arg: None,
            distinct: false,
            output_name: "n".into(),
            data_type: DataType::Int64,
        };
        assert_eq!(c.to_string(), "count(*)");
    }
}
