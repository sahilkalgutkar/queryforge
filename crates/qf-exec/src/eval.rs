//! Vectorised expression evaluation.
//!
//! Every kernel here takes whole columns and returns a whole column. The
//! operator is resolved once, outside the loop, and the loop then runs over a
//! contiguous buffer — which is the entire reason for the columnar layout. The
//! row-at-a-time alternative would walk the expression tree once per row and
//! spend most of its time on dispatch rather than on the comparison.
//!
//! Two cases get their own paths because they dominate real queries:
//! comparing a column against a literal (no constant column is materialised),
//! and comparing two columns of the same type (no boxing through `Value`).

use qf_common::{DataType, Error, Result, Value};
use qf_plan::expr::BoundExpr;
use qf_sql::ast::{BinaryOp, UnaryOp};
use qf_storage::array::{Array, ArrayBuilder, ArrayData};
use qf_storage::batch::RecordBatch;
use qf_storage::bitmap::Bitmap;
use std::cmp::Ordering;

/// Evaluates `expr` over every row of `batch`, producing one column.
pub fn evaluate(expr: &BoundExpr, batch: &RecordBatch) -> Result<Array> {
    match expr {
        BoundExpr::Column { index, .. } => Ok(batch.column(*index)?.clone()),
        BoundExpr::Literal(v) => constant(v, batch.num_rows(), expr.data_type()),
        BoundExpr::Binary {
            left, op, right, ..
        } => eval_binary(left, *op, right, batch),
        BoundExpr::Unary { op, expr, .. } => {
            let input = evaluate(expr, batch)?;
            eval_unary(*op, &input)
        }
        BoundExpr::Cast { expr, data_type } => {
            let input = evaluate(expr, batch)?;
            cast_array(&input, *data_type)
        }
        BoundExpr::IsNull { expr, negated } => {
            let input = evaluate(expr, batch)?;
            let mut bits = Bitmap::with_capacity(input.len());
            for i in 0..input.len() {
                bits.push(input.is_valid(i) == *negated);
            }
            // `IS NULL` is never itself null, so the result carries no mask.
            Array::new(ArrayData::Boolean(bits), None, input.len())
        }
        BoundExpr::InList {
            expr,
            list,
            negated,
        } => eval_in_list(expr, list, *negated, batch),
        BoundExpr::Like {
            expr,
            pattern,
            negated,
        } => eval_like(expr, pattern, *negated, batch),
        BoundExpr::Case {
            branches,
            else_result,
            data_type,
        } => eval_case(branches, else_result.as_deref(), *data_type, batch),
    }
}

/// Evaluates a predicate into a selection mask, treating SQL's unknown as
/// "not selected" — which is what `WHERE` does with NULL.
pub fn evaluate_predicate(expr: &BoundExpr, batch: &RecordBatch) -> Result<Bitmap> {
    let array = evaluate(expr, batch)?;
    let ArrayData::Boolean(bits) = array.data() else {
        return Err(Error::exec(format!(
            "a predicate evaluated to {}, not a boolean",
            array.data_type()
        )));
    };
    let mut mask = Bitmap::with_capacity(array.len());
    for i in 0..array.len() {
        mask.push(array.is_valid(i) && bits.get(i));
    }
    Ok(mask)
}

fn constant(value: &Value, rows: usize, data_type: DataType) -> Result<Array> {
    let mut b = ArrayBuilder::new(data_type);
    b.reserve(rows);
    for _ in 0..rows {
        b.push(value.clone())?;
    }
    b.finish()
}

fn eval_binary(
    left: &BoundExpr,
    op: BinaryOp,
    right: &BoundExpr,
    batch: &RecordBatch,
) -> Result<Array> {
    if op.is_logical() {
        let l = evaluate(left, batch)?;
        let r = evaluate(right, batch)?;
        return eval_logical(op, &l, &r);
    }

    // Column against literal, without building a column of the literal.
    if let (BoundExpr::Column { index, .. }, Some(v)) = (left, right.as_literal()) {
        let column = batch.column(*index)?;
        if op.is_comparison() {
            return compare_scalar(column, op, v, false);
        }
        return arithmetic_scalar(column, op, v, false);
    }
    if let (Some(v), BoundExpr::Column { index, .. }) = (left.as_literal(), right) {
        let column = batch.column(*index)?;
        if op.is_comparison() {
            return compare_scalar(column, op, v, true);
        }
        return arithmetic_scalar(column, op, v, true);
    }

    let l = evaluate(left, batch)?;
    let r = evaluate(right, batch)?;
    if op.is_comparison() {
        compare_arrays(&l, op, &r)
    } else {
        arithmetic_arrays(&l, op, &r)
    }
}

fn ordering_matches(op: BinaryOp, ord: Ordering) -> bool {
    match op {
        BinaryOp::Eq => ord.is_eq(),
        BinaryOp::NotEq => !ord.is_eq(),
        BinaryOp::Lt => ord.is_lt(),
        BinaryOp::LtEq => ord.is_le(),
        BinaryOp::Gt => ord.is_gt(),
        _ => ord.is_ge(),
    }
}

fn boolean_array(bits: Bitmap, validity: Option<Bitmap>, len: usize) -> Result<Array> {
    Array::new(ArrayData::Boolean(bits), validity, len)
}

/// Column against a scalar. `flipped` means the literal was on the left, so
/// the comparison is mirrored rather than the data being copied.
fn compare_scalar(column: &Array, op: BinaryOp, value: &Value, flipped: bool) -> Result<Array> {
    let len = column.len();
    let mut bits = Bitmap::with_capacity(len);
    let mut validity = Bitmap::with_capacity(len);
    let mut any_null = false;

    if value.is_null() {
        // Comparing with NULL is unknown for every row.
        return boolean_array(
            Bitmap::filled(len, false),
            Some(Bitmap::filled(len, false)),
            len,
        );
    }

    // Typed fast paths: the comparison is resolved once, then the loop runs
    // over the raw buffer.
    match (column.data(), value) {
        (ArrayData::Int64(values), Value::Int64(v)) => {
            for (i, x) in values.iter().enumerate() {
                let valid = column.is_valid(i);
                any_null |= !valid;
                validity.push(valid);
                let ord = if flipped { v.cmp(x) } else { x.cmp(v) };
                bits.push(valid && ordering_matches(op, ord));
            }
        }
        (ArrayData::Float64(values), _) if value.as_f64().is_some() => {
            let v = value.as_f64().unwrap();
            for (i, x) in values.iter().enumerate() {
                let valid = column.is_valid(i);
                any_null |= !valid;
                validity.push(valid);
                let ord = if flipped {
                    v.partial_cmp(x)
                } else {
                    x.partial_cmp(&v)
                };
                bits.push(valid && ord.is_some_and(|o| ordering_matches(op, o)));
            }
        }
        (ArrayData::Utf8 { .. }, Value::Utf8(v)) => {
            for i in 0..len {
                let valid = column.is_valid(i);
                any_null |= !valid;
                validity.push(valid);
                let matched = match column.str_value(i) {
                    Some(s) => {
                        let ord = if flipped {
                            v.as_str().cmp(s)
                        } else {
                            s.cmp(v.as_str())
                        };
                        ordering_matches(op, ord)
                    }
                    None => false,
                };
                bits.push(valid && matched);
            }
        }
        _ => {
            // Mixed or unusual types fall back to the scalar comparison, which
            // reports a type error rather than guessing.
            for i in 0..len {
                let x = column.value(i);
                let valid = !x.is_null();
                any_null |= !valid;
                validity.push(valid);
                let ord = if flipped {
                    value.sql_compare(&x)?
                } else {
                    x.sql_compare(value)?
                };
                bits.push(match ord {
                    Some(o) => ordering_matches(op, o),
                    None => false,
                });
            }
        }
    }

    boolean_array(bits, if any_null { Some(validity) } else { None }, len)
}

fn compare_arrays(left: &Array, op: BinaryOp, right: &Array) -> Result<Array> {
    let len = left.len();
    if right.len() != len {
        return Err(Error::internal(format!(
            "comparing a {len}-row column with a {}-row column",
            right.len()
        )));
    }
    let mut bits = Bitmap::with_capacity(len);
    let mut validity = Bitmap::with_capacity(len);
    let mut any_null = false;

    match (left.data(), right.data()) {
        (ArrayData::Int64(a), ArrayData::Int64(b)) => {
            for i in 0..len {
                let valid = left.is_valid(i) && right.is_valid(i);
                any_null |= !valid;
                validity.push(valid);
                bits.push(valid && ordering_matches(op, a[i].cmp(&b[i])));
            }
        }
        (ArrayData::Float64(a), ArrayData::Float64(b)) => {
            for i in 0..len {
                let valid = left.is_valid(i) && right.is_valid(i);
                any_null |= !valid;
                validity.push(valid);
                bits.push(
                    valid
                        && a[i]
                            .partial_cmp(&b[i])
                            .is_some_and(|o| ordering_matches(op, o)),
                );
            }
        }
        (ArrayData::Utf8 { .. }, ArrayData::Utf8 { .. }) => {
            for i in 0..len {
                let valid = left.is_valid(i) && right.is_valid(i);
                any_null |= !valid;
                validity.push(valid);
                let matched = match (left.str_value(i), right.str_value(i)) {
                    (Some(a), Some(b)) => ordering_matches(op, a.cmp(b)),
                    _ => false,
                };
                bits.push(valid && matched);
            }
        }
        _ => {
            for i in 0..len {
                let (a, b) = (left.value(i), right.value(i));
                let valid = !a.is_null() && !b.is_null();
                any_null |= !valid;
                validity.push(valid);
                bits.push(match a.sql_compare(&b)? {
                    Some(o) => ordering_matches(op, o),
                    None => false,
                });
            }
        }
    }
    boolean_array(bits, if any_null { Some(validity) } else { None }, len)
}

fn arith_values(op: BinaryOp, a: &Value, b: &Value) -> Result<Value> {
    if a.is_null() || b.is_null() {
        return Ok(Value::Null);
    }
    Ok(match (a, b) {
        (Value::Int64(x), Value::Int64(y)) => match op {
            BinaryOp::Plus => Value::Int64(x.wrapping_add(*y)),
            BinaryOp::Minus => Value::Int64(x.wrapping_sub(*y)),
            BinaryOp::Multiply => Value::Int64(x.wrapping_mul(*y)),
            // Dividing by zero yields NULL rather than aborting the query.
            // A whole scan should not die because one row held a zero.
            BinaryOp::Divide => {
                if *y == 0 {
                    Value::Null
                } else {
                    Value::Float64(*x as f64 / *y as f64)
                }
            }
            BinaryOp::Modulo => {
                if *y == 0 {
                    Value::Null
                } else {
                    Value::Int64(x % y)
                }
            }
            _ => return Err(Error::exec(format!("{op} is not arithmetic"))),
        },
        _ => {
            let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) else {
                return Err(Error::typ(format!(
                    "cannot apply {op} to {} and {}",
                    a.type_name(),
                    b.type_name()
                )));
            };
            match op {
                BinaryOp::Plus => Value::Float64(x + y),
                BinaryOp::Minus => Value::Float64(x - y),
                BinaryOp::Multiply => Value::Float64(x * y),
                BinaryOp::Divide | BinaryOp::Modulo => {
                    if y == 0.0 {
                        Value::Null
                    } else if op == BinaryOp::Divide {
                        Value::Float64(x / y)
                    } else {
                        Value::Float64(x % y)
                    }
                }
                _ => return Err(Error::exec(format!("{op} is not arithmetic"))),
            }
        }
    })
}

fn result_type(op: BinaryOp, a: DataType, b: DataType) -> Result<DataType> {
    if op == BinaryOp::Divide {
        return Ok(DataType::Float64);
    }
    DataType::unify(a, b)
}

fn arithmetic_scalar(column: &Array, op: BinaryOp, value: &Value, flipped: bool) -> Result<Array> {
    let value_type = value.data_type().unwrap_or(column.data_type());
    let out_type = result_type(op, column.data_type(), value_type)?;
    let mut b = ArrayBuilder::new(out_type);
    b.reserve(column.len());
    for i in 0..column.len() {
        let x = column.value(i);
        let v = if flipped {
            arith_values(op, value, &x)?
        } else {
            arith_values(op, &x, value)?
        };
        b.push(v)?;
    }
    b.finish()
}

fn arithmetic_arrays(left: &Array, op: BinaryOp, right: &Array) -> Result<Array> {
    if left.len() != right.len() {
        return Err(Error::internal(
            "arithmetic over unequal columns".to_string(),
        ));
    }
    let out_type = result_type(op, left.data_type(), right.data_type())?;
    let mut b = ArrayBuilder::new(out_type);
    b.reserve(left.len());
    for i in 0..left.len() {
        b.push(arith_values(op, &left.value(i), &right.value(i))?)?;
    }
    b.finish()
}

/// Three-valued AND/OR.
///
/// `false AND NULL` is `false`, not NULL — the row cannot match whatever the
/// unknown turns out to be. Getting this wrong changes which rows a query
/// returns, so both asymmetric cases have their own test.
fn eval_logical(op: BinaryOp, left: &Array, right: &Array) -> Result<Array> {
    let len = left.len();
    let (ArrayData::Boolean(a), ArrayData::Boolean(b)) = (left.data(), right.data()) else {
        return Err(Error::typ(format!(
            "{op} needs booleans, found {} and {}",
            left.data_type(),
            right.data_type()
        )));
    };
    let mut bits = Bitmap::with_capacity(len);
    let mut validity = Bitmap::with_capacity(len);
    let mut any_null = false;
    for i in 0..len {
        let l = left.is_valid(i).then(|| a.get(i));
        let r = right.is_valid(i).then(|| b.get(i));
        let value = match op {
            BinaryOp::And => match (l, r) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            _ => match (l, r) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
        };
        match value {
            Some(v) => {
                validity.push(true);
                bits.push(v);
            }
            None => {
                any_null = true;
                validity.push(false);
                bits.push(false);
            }
        }
    }
    boolean_array(bits, if any_null { Some(validity) } else { None }, len)
}

fn eval_unary(op: UnaryOp, input: &Array) -> Result<Array> {
    match op {
        UnaryOp::Not => {
            let ArrayData::Boolean(bits) = input.data() else {
                return Err(Error::typ(format!(
                    "NOT needs a boolean, found {}",
                    input.data_type()
                )));
            };
            let mut out = Bitmap::with_capacity(input.len());
            for i in 0..input.len() {
                out.push(input.is_valid(i) && !bits.get(i));
            }
            boolean_array(out, input.validity().cloned(), input.len())
        }
        UnaryOp::Neg => {
            let mut b = ArrayBuilder::new(input.data_type());
            b.reserve(input.len());
            for i in 0..input.len() {
                b.push(match input.value(i) {
                    Value::Int64(v) => Value::Int64(-v),
                    Value::Float64(v) => Value::Float64(-v),
                    Value::Null => Value::Null,
                    other => {
                        return Err(Error::typ(format!("cannot negate {}", other.type_name())))
                    }
                })?;
            }
            b.finish()
        }
    }
}

fn cast_array(input: &Array, target: DataType) -> Result<Array> {
    let mut b = ArrayBuilder::new(target);
    b.reserve(input.len());
    for i in 0..input.len() {
        b.push(input.value(i).cast_to(target)?)?;
    }
    b.finish()
}

fn eval_in_list(
    expr: &BoundExpr,
    list: &[BoundExpr],
    negated: bool,
    batch: &RecordBatch,
) -> Result<Array> {
    let column = evaluate(expr, batch)?;
    // Literal lists — the overwhelmingly common case — become a set once
    // rather than a comparison per row per item.
    let literals: Option<Vec<Value>> = list
        .iter()
        .map(|e| e.as_literal().cloned())
        .collect::<Option<Vec<_>>>();

    let len = column.len();
    let mut bits = Bitmap::with_capacity(len);
    let mut validity = Bitmap::with_capacity(len);
    let mut any_null = false;

    let columns: Vec<Array> = match &literals {
        Some(_) => vec![],
        None => list
            .iter()
            .map(|e| evaluate(e, batch))
            .collect::<Result<Vec<_>>>()?,
    };
    let set: Option<std::collections::HashSet<Value>> = literals
        .as_ref()
        .map(|l| l.iter().filter(|v| !v.is_null()).cloned().collect());
    let list_has_null = literals
        .as_ref()
        .is_some_and(|l| l.iter().any(Value::is_null));

    for i in 0..len {
        let v = column.value(i);
        if v.is_null() {
            any_null = true;
            validity.push(false);
            bits.push(false);
            continue;
        }
        let found = match &set {
            Some(s) => s.contains(&v),
            None => {
                let mut hit = false;
                for c in &columns {
                    if c.value(i) == v {
                        hit = true;
                        break;
                    }
                }
                hit
            }
        };
        // `x IN (1, NULL)` is unknown when x is not 1: the NULL might have
        // been x.
        if !found && list_has_null {
            any_null = true;
            validity.push(false);
            bits.push(false);
            continue;
        }
        validity.push(true);
        bits.push(found != negated);
    }
    boolean_array(bits, if any_null { Some(validity) } else { None }, len)
}

/// SQL `LIKE`: `%` matches any run of characters, `_` matches exactly one.
///
/// Backtracking rather than a regex, and iterative rather than recursive so a
/// pattern of many `%`s cannot blow the stack.
pub fn like_matches(text: &str, pattern: &str) -> bool {
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    let (mut ti, mut pi) = (0usize, 0usize);
    // Where to resume if the current `%` guess turns out to be wrong.
    let (mut star, mut star_ti) = (None, 0usize);

    while ti < t.len() {
        if pi < p.len() && (p[pi] == '_' || p[pi] == t[ti]) {
            ti += 1;
            pi += 1;
        } else if pi < p.len() && p[pi] == '%' {
            star = Some(pi);
            star_ti = ti;
            pi += 1;
        } else if let Some(s) = star {
            // Let the last `%` swallow one more character and try again.
            pi = s + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '%' {
        pi += 1;
    }
    pi == p.len()
}

fn eval_like(
    expr: &BoundExpr,
    pattern: &BoundExpr,
    negated: bool,
    batch: &RecordBatch,
) -> Result<Array> {
    let column = evaluate(expr, batch)?;
    let len = column.len();
    let mut bits = Bitmap::with_capacity(len);
    let mut validity = Bitmap::with_capacity(len);
    let mut any_null = false;

    let constant_pattern = pattern.as_literal().and_then(|v| match v {
        Value::Utf8(s) => Some(s.clone()),
        _ => None,
    });
    let pattern_column = match &constant_pattern {
        Some(_) => None,
        None => Some(evaluate(pattern, batch)?),
    };

    for i in 0..len {
        let text = column.str_value(i);
        let pat = match (&constant_pattern, &pattern_column) {
            (Some(p), _) => Some(p.as_str()),
            (None, Some(c)) => c.str_value(i),
            _ => None,
        };
        match (text, pat) {
            (Some(t), Some(p)) => {
                validity.push(true);
                bits.push(like_matches(t, p) != negated);
            }
            _ => {
                any_null = true;
                validity.push(false);
                bits.push(false);
            }
        }
    }
    boolean_array(bits, if any_null { Some(validity) } else { None }, len)
}

fn eval_case(
    branches: &[(BoundExpr, BoundExpr)],
    else_result: Option<&BoundExpr>,
    data_type: DataType,
    batch: &RecordBatch,
) -> Result<Array> {
    let len = batch.num_rows();
    let conditions = branches
        .iter()
        .map(|(w, _)| evaluate_predicate(w, batch))
        .collect::<Result<Vec<_>>>()?;
    let results = branches
        .iter()
        .map(|(_, t)| evaluate(t, batch))
        .collect::<Result<Vec<_>>>()?;
    let fallback = match else_result {
        Some(e) => Some(evaluate(e, batch)?),
        None => None,
    };

    let mut b = ArrayBuilder::new(data_type);
    b.reserve(len);
    'row: for i in 0..len {
        for (c, r) in conditions.iter().zip(results.iter()) {
            if c.get(i) {
                b.push(r.value(i).cast_to(data_type)?)?;
                continue 'row;
            }
        }
        match &fallback {
            Some(f) => b.push(f.value(i).cast_to(data_type)?)?,
            None => b.push_null(),
        }
    }
    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qf_common::{Field, Schema};
    use std::sync::Arc;

    fn batch(columns: Vec<(&str, DataType, Vec<Value>)>) -> RecordBatch {
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|(n, t, _)| Field::new(*n, *t, true))
                .collect(),
        ));
        let arrays = columns
            .iter()
            .map(|(_, t, v)| Array::from_values(*t, v).unwrap())
            .collect();
        RecordBatch::try_new(schema, arrays).unwrap()
    }

    fn ints(name: &str, values: &[Option<i64>]) -> (String, DataType, Vec<Value>) {
        (
            name.to_string(),
            DataType::Int64,
            values
                .iter()
                .map(|v| v.map_or(Value::Null, Value::Int64))
                .collect(),
        )
    }

    fn sample() -> RecordBatch {
        let (n, t, v) = ints("a", &[Some(1), Some(5), None, Some(10)]);
        batch(vec![
            (Box::leak(n.into_boxed_str()), t, v),
            (
                "s",
                DataType::Utf8,
                vec![
                    Value::Utf8("alpha".into()),
                    Value::Utf8("beta".into()),
                    Value::Null,
                    Value::Utf8("gamma".into()),
                ],
            ),
            (
                "f",
                DataType::Float64,
                vec![
                    Value::Float64(1.5),
                    Value::Float64(2.5),
                    Value::Float64(3.5),
                    Value::Null,
                ],
            ),
        ])
    }

    fn col(i: usize, t: DataType) -> BoundExpr {
        BoundExpr::column(i, format!("c{i}"), t)
    }

    fn lit(v: Value) -> BoundExpr {
        BoundExpr::Literal(v)
    }

    fn mask(expr: &BoundExpr, b: &RecordBatch) -> Vec<bool> {
        evaluate_predicate(expr, b).unwrap().iter().collect()
    }

    #[test]
    fn a_column_evaluates_to_itself_and_a_literal_to_a_constant_column() {
        let b = sample();
        let c = evaluate(&col(0, DataType::Int64), &b).unwrap();
        assert_eq!(c.value(1), Value::Int64(5));
        let k = evaluate(&lit(Value::Int64(7)), &b).unwrap();
        assert_eq!(k.len(), 4);
        assert_eq!(k.value(3), Value::Int64(7));
    }

    #[test]
    fn comparing_a_column_with_a_literal_selects_the_expected_rows() {
        let b = sample();
        let e =
            BoundExpr::binary(col(0, DataType::Int64), BinaryOp::Gt, lit(Value::Int64(4))).unwrap();
        assert_eq!(mask(&e, &b), vec![false, true, false, true]);
    }

    #[test]
    fn a_literal_on_the_left_mirrors_the_comparison_rather_than_copying_the_column() {
        let b = sample();
        // `4 < a` must select the same rows as `a > 4`.
        let e =
            BoundExpr::binary(lit(Value::Int64(4)), BinaryOp::Lt, col(0, DataType::Int64)).unwrap();
        assert_eq!(mask(&e, &b), vec![false, true, false, true]);
    }

    #[test]
    fn a_null_row_never_satisfies_a_comparison() {
        let b = sample();
        for op in [
            BinaryOp::Eq,
            BinaryOp::NotEq,
            BinaryOp::Lt,
            BinaryOp::Gt,
            BinaryOp::LtEq,
            BinaryOp::GtEq,
        ] {
            let e = BoundExpr::binary(col(0, DataType::Int64), op, lit(Value::Int64(5))).unwrap();
            assert!(!mask(&e, &b)[2], "{op} selected the null row");
        }
    }

    #[test]
    fn comparing_against_null_selects_nothing() {
        let b = sample();
        let e = BoundExpr::binary(col(0, DataType::Int64), BinaryOp::Eq, lit(Value::Null)).unwrap();
        assert_eq!(mask(&e, &b), vec![false; 4]);
    }

    #[test]
    fn string_and_float_comparisons_use_their_own_paths() {
        let b = sample();
        let s = BoundExpr::binary(
            col(1, DataType::Utf8),
            BinaryOp::Lt,
            lit(Value::Utf8("delta".into())),
        )
        .unwrap();
        assert_eq!(mask(&s, &b), vec![true, true, false, false]);

        let f = BoundExpr::binary(
            col(2, DataType::Float64),
            BinaryOp::GtEq,
            lit(Value::Float64(2.5)),
        )
        .unwrap();
        assert_eq!(mask(&f, &b), vec![false, true, true, false]);
    }

    #[test]
    fn an_integer_column_compares_against_a_float_literal() {
        let b = sample();
        let e = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Gt,
            lit(Value::Float64(4.5)),
        )
        .unwrap();
        assert_eq!(mask(&e, &b), vec![false, true, false, true]);
    }

    #[test]
    fn two_columns_compare_element_by_element() {
        let b = batch(vec![
            ("a", DataType::Int64, vec![Value::Int64(1), Value::Int64(5)]),
            ("b", DataType::Int64, vec![Value::Int64(3), Value::Int64(2)]),
        ]);
        let e = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Lt,
            col(1, DataType::Int64),
        )
        .unwrap();
        assert_eq!(mask(&e, &b), vec![true, false]);
    }

    #[test]
    fn two_string_columns_and_two_float_columns_compare() {
        let b = batch(vec![
            (
                "a",
                DataType::Utf8,
                vec![Value::Utf8("x".into()), Value::Null],
            ),
            (
                "b",
                DataType::Utf8,
                vec![Value::Utf8("y".into()), Value::Utf8("z".into())],
            ),
            (
                "c",
                DataType::Float64,
                vec![Value::Float64(1.0), Value::Float64(2.0)],
            ),
            (
                "d",
                DataType::Float64,
                vec![Value::Float64(2.0), Value::Float64(1.0)],
            ),
        ]);
        let s = BoundExpr::binary(col(0, DataType::Utf8), BinaryOp::Lt, col(1, DataType::Utf8))
            .unwrap();
        assert_eq!(mask(&s, &b), vec![true, false]);
        let f = BoundExpr::binary(
            col(2, DataType::Float64),
            BinaryOp::Lt,
            col(3, DataType::Float64),
        )
        .unwrap();
        assert_eq!(mask(&f, &b), vec![true, false]);
    }

    #[test]
    fn mixed_type_columns_fall_back_to_the_scalar_path() {
        let b = batch(vec![
            ("a", DataType::Int64, vec![Value::Int64(2)]),
            ("b", DataType::Float64, vec![Value::Float64(2.0)]),
        ]);
        let e = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Eq,
            col(1, DataType::Float64),
        )
        .unwrap();
        assert_eq!(mask(&e, &b), vec![true]);
    }

    #[test]
    fn arithmetic_runs_over_columns_and_scalars_alike() {
        let b = sample();
        let plus = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Plus,
            lit(Value::Int64(1)),
        )
        .unwrap();
        let out = evaluate(&plus, &b).unwrap();
        assert_eq!(out.value(0), Value::Int64(2));
        assert!(out.value(2).is_null());

        let times = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Multiply,
            col(0, DataType::Int64),
        )
        .unwrap();
        assert_eq!(evaluate(&times, &b).unwrap().value(1), Value::Int64(25));
    }

    #[test]
    fn a_literal_on_the_left_of_a_subtraction_keeps_the_operands_in_order() {
        let b = sample();
        let e = BoundExpr::binary(
            lit(Value::Int64(10)),
            BinaryOp::Minus,
            col(0, DataType::Int64),
        )
        .unwrap();
        let out = evaluate(&e, &b).unwrap();
        assert_eq!(out.value(0), Value::Int64(9));
        assert_eq!(out.value(1), Value::Int64(5));
    }

    #[test]
    fn dividing_by_zero_yields_null_rather_than_killing_the_query() {
        // One bad row should not abort a scan of a million.
        let b = batch(vec![
            ("a", DataType::Int64, vec![Value::Int64(6), Value::Int64(6)]),
            ("b", DataType::Int64, vec![Value::Int64(3), Value::Int64(0)]),
        ]);
        let e = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Divide,
            col(1, DataType::Int64),
        )
        .unwrap();
        let out = evaluate(&e, &b).unwrap();
        assert_eq!(out.value(0), Value::Float64(2.0));
        assert!(out.value(1).is_null());

        let m = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Modulo,
            col(1, DataType::Int64),
        )
        .unwrap();
        assert!(evaluate(&m, &b).unwrap().value(1).is_null());
    }

    #[test]
    fn integer_division_produces_a_float_column() {
        let b = batch(vec![
            ("a", DataType::Int64, vec![Value::Int64(7)]),
            ("b", DataType::Int64, vec![Value::Int64(2)]),
        ]);
        let e = BoundExpr::binary(
            col(0, DataType::Int64),
            BinaryOp::Divide,
            col(1, DataType::Int64),
        )
        .unwrap();
        let out = evaluate(&e, &b).unwrap();
        assert_eq!(out.data_type(), DataType::Float64);
        assert_eq!(out.value(0), Value::Float64(3.5));
    }

    #[test]
    fn three_valued_and_treats_false_as_decisive() {
        // `false AND NULL` is false: the row cannot match whatever the unknown
        // turns out to be.
        let b = batch(vec![
            (
                "p",
                DataType::Boolean,
                vec![Value::Boolean(false), Value::Boolean(true), Value::Null],
            ),
            ("q", DataType::Boolean, vec![Value::Null; 3]),
        ]);
        let and = BoundExpr::binary(
            col(0, DataType::Boolean),
            BinaryOp::And,
            col(1, DataType::Boolean),
        )
        .unwrap();
        let out = evaluate(&and, &b).unwrap();
        assert_eq!(out.value(0), Value::Boolean(false));
        assert!(out.value(1).is_null());
        assert!(out.value(2).is_null());
    }

    #[test]
    fn three_valued_or_treats_true_as_decisive() {
        let b = batch(vec![
            (
                "p",
                DataType::Boolean,
                vec![Value::Boolean(true), Value::Boolean(false), Value::Null],
            ),
            ("q", DataType::Boolean, vec![Value::Null; 3]),
        ]);
        let or = BoundExpr::binary(
            col(0, DataType::Boolean),
            BinaryOp::Or,
            col(1, DataType::Boolean),
        )
        .unwrap();
        let out = evaluate(&or, &b).unwrap();
        assert_eq!(out.value(0), Value::Boolean(true));
        assert!(out.value(1).is_null());
        assert!(out.value(2).is_null());
    }

    #[test]
    fn a_predicate_that_is_unknown_does_not_select_its_row() {
        let b = batch(vec![("p", DataType::Boolean, vec![Value::Null])]);
        assert_eq!(mask(&col(0, DataType::Boolean), &b), vec![false]);
    }

    #[test]
    fn not_inverts_and_leaves_nulls_null() {
        let b = batch(vec![(
            "p",
            DataType::Boolean,
            vec![Value::Boolean(true), Value::Boolean(false), Value::Null],
        )]);
        let e = BoundExpr::unary(UnaryOp::Not, col(0, DataType::Boolean)).unwrap();
        let out = evaluate(&e, &b).unwrap();
        assert_eq!(out.value(0), Value::Boolean(false));
        assert_eq!(out.value(1), Value::Boolean(true));
        assert!(out.value(2).is_null());
    }

    #[test]
    fn negation_and_casting_run_over_whole_columns() {
        let b = sample();
        let neg = BoundExpr::unary(UnaryOp::Neg, col(0, DataType::Int64)).unwrap();
        assert_eq!(evaluate(&neg, &b).unwrap().value(1), Value::Int64(-5));

        let cast = BoundExpr::Cast {
            expr: Box::new(col(0, DataType::Int64)),
            data_type: DataType::Utf8,
        };
        let out = evaluate(&cast, &b).unwrap();
        assert_eq!(out.value(0), Value::Utf8("1".into()));
        assert!(out.value(2).is_null());
    }

    #[test]
    fn is_null_and_is_not_null_are_never_themselves_null() {
        let b = sample();
        let is_null = BoundExpr::IsNull {
            expr: Box::new(col(0, DataType::Int64)),
            negated: false,
        };
        assert_eq!(mask(&is_null, &b), vec![false, false, true, false]);
        let not_null = BoundExpr::IsNull {
            expr: Box::new(col(0, DataType::Int64)),
            negated: true,
        };
        assert_eq!(mask(&not_null, &b), vec![true, true, false, true]);
        assert!(evaluate(&is_null, &b).unwrap().validity().is_none());
    }

    #[test]
    fn in_list_matches_against_a_set_built_once() {
        let b = sample();
        let e = BoundExpr::InList {
            expr: Box::new(col(0, DataType::Int64)),
            list: vec![lit(Value::Int64(1)), lit(Value::Int64(10))],
            negated: false,
        };
        assert_eq!(mask(&e, &b), vec![true, false, false, true]);
        let n = BoundExpr::InList {
            expr: Box::new(col(0, DataType::Int64)),
            list: vec![lit(Value::Int64(1))],
            negated: true,
        };
        assert_eq!(mask(&n, &b), vec![false, true, false, true]);
    }

    #[test]
    fn a_null_in_the_list_makes_a_non_match_unknown() {
        // `5 IN (1, NULL)` is unknown, not false — the NULL might have been 5.
        let b = sample();
        let e = BoundExpr::InList {
            expr: Box::new(col(0, DataType::Int64)),
            list: vec![lit(Value::Int64(1)), lit(Value::Null)],
            negated: false,
        };
        let out = evaluate(&e, &b).unwrap();
        assert_eq!(out.value(0), Value::Boolean(true));
        assert!(out.value(1).is_null());
    }

    #[test]
    fn an_in_list_of_expressions_is_evaluated_per_row() {
        let b = batch(vec![
            ("a", DataType::Int64, vec![Value::Int64(3), Value::Int64(4)]),
            ("b", DataType::Int64, vec![Value::Int64(3), Value::Int64(9)]),
        ]);
        let e = BoundExpr::InList {
            expr: Box::new(col(0, DataType::Int64)),
            list: vec![BoundExpr::binary(
                col(1, DataType::Int64),
                BinaryOp::Plus,
                lit(Value::Int64(0)),
            )
            .unwrap()],
            negated: false,
        };
        assert_eq!(mask(&e, &b), vec![true, false]);
    }

    #[test]
    fn like_handles_the_wildcards_and_the_awkward_patterns() {
        assert!(like_matches("alpha", "a%"));
        assert!(like_matches("alpha", "%a"));
        assert!(like_matches("alpha", "%lph%"));
        assert!(like_matches("alpha", "_lpha"));
        assert!(like_matches("alpha", "alpha"));
        assert!(like_matches("alpha", "%"));
        assert!(like_matches("", "%"));
        assert!(like_matches("", ""));
        assert!(!like_matches("alpha", "b%"));
        assert!(!like_matches("alpha", "_lph"));
        assert!(!like_matches("", "_"));
    }

    #[test]
    fn like_backtracks_correctly_on_a_pattern_that_needs_it() {
        // A greedy `%` has to give characters back for this to match.
        assert!(like_matches("aaa", "%a"));
        assert!(like_matches("abcabc", "%abc"));
        assert!(!like_matches("abcabd", "%abc"));
        assert!(like_matches("xaybz", "x%y%z"));
        assert!(!like_matches("xaybz", "x%y%q"));
    }

    #[test]
    fn like_over_a_column_uses_the_constant_pattern_path() {
        let b = sample();
        let e = BoundExpr::Like {
            expr: Box::new(col(1, DataType::Utf8)),
            pattern: Box::new(lit(Value::Utf8("%a".into()))),
            negated: false,
        };
        assert_eq!(mask(&e, &b), vec![true, true, false, true]);
        let n = BoundExpr::Like {
            expr: Box::new(col(1, DataType::Utf8)),
            pattern: Box::new(lit(Value::Utf8("a%".into()))),
            negated: true,
        };
        assert_eq!(mask(&n, &b), vec![false, true, false, true]);
    }

    #[test]
    fn like_with_a_per_row_pattern_column_works_too() {
        let b = batch(vec![
            (
                "s",
                DataType::Utf8,
                vec![Value::Utf8("abc".into()), Value::Utf8("abc".into())],
            ),
            (
                "p",
                DataType::Utf8,
                vec![Value::Utf8("a%".into()), Value::Utf8("z%".into())],
            ),
        ]);
        let e = BoundExpr::Like {
            expr: Box::new(col(0, DataType::Utf8)),
            pattern: Box::new(col(1, DataType::Utf8)),
            negated: false,
        };
        assert_eq!(mask(&e, &b), vec![true, false]);
    }

    #[test]
    fn case_picks_the_first_matching_branch_and_falls_back_to_else() {
        let b = sample();
        let e = BoundExpr::case(
            vec![
                (
                    BoundExpr::binary(col(0, DataType::Int64), BinaryOp::Lt, lit(Value::Int64(3)))
                        .unwrap(),
                    lit(Value::Utf8("small".into())),
                ),
                (
                    BoundExpr::binary(col(0, DataType::Int64), BinaryOp::Lt, lit(Value::Int64(8)))
                        .unwrap(),
                    lit(Value::Utf8("medium".into())),
                ),
            ],
            Some(lit(Value::Utf8("large".into()))),
        )
        .unwrap();
        let out = evaluate(&e, &b).unwrap();
        assert_eq!(out.value(0), Value::Utf8("small".into()));
        assert_eq!(out.value(1), Value::Utf8("medium".into()));
        // The NULL row matches no condition, so it takes the ELSE.
        assert_eq!(out.value(2), Value::Utf8("large".into()));
        assert_eq!(out.value(3), Value::Utf8("large".into()));
    }

    #[test]
    fn a_case_without_else_yields_null_where_nothing_matches() {
        let b = sample();
        let e = BoundExpr::case(
            vec![(
                BoundExpr::binary(
                    col(0, DataType::Int64),
                    BinaryOp::Gt,
                    lit(Value::Int64(100)),
                )
                .unwrap(),
                lit(Value::Int64(1)),
            )],
            None,
        )
        .unwrap();
        let out = evaluate(&e, &b).unwrap();
        assert!(out.iter().all(|v| v.is_null()));
    }

    #[test]
    fn a_non_boolean_predicate_is_an_execution_error() {
        let b = sample();
        let err = evaluate_predicate(&col(0, DataType::Int64), &b).unwrap_err();
        assert!(err.to_string().contains("not a boolean"));
    }

    #[test]
    fn logical_operators_over_non_booleans_are_rejected() {
        let b = sample();
        let e = BoundExpr::Binary {
            left: Box::new(col(0, DataType::Int64)),
            op: BinaryOp::And,
            right: Box::new(col(0, DataType::Int64)),
            data_type: DataType::Boolean,
        };
        assert!(evaluate(&e, &b).is_err());
        let n = BoundExpr::Unary {
            op: UnaryOp::Not,
            expr: Box::new(col(0, DataType::Int64)),
            data_type: DataType::Boolean,
        };
        assert!(evaluate(&n, &b).is_err());
    }

    #[test]
    fn negating_or_adding_text_is_rejected_at_run_time_too() {
        let b = sample();
        let neg = BoundExpr::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(col(1, DataType::Utf8)),
            data_type: DataType::Utf8,
        };
        assert!(evaluate(&neg, &b).is_err());
    }

    #[test]
    fn an_empty_batch_evaluates_to_an_empty_column() {
        let b = batch(vec![("a", DataType::Int64, vec![])]);
        let e =
            BoundExpr::binary(col(0, DataType::Int64), BinaryOp::Gt, lit(Value::Int64(0))).unwrap();
        assert_eq!(evaluate(&e, &b).unwrap().len(), 0);
        assert!(evaluate_predicate(&e, &b).unwrap().is_empty());
    }
}
