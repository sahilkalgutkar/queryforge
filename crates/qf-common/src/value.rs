use crate::error::{Error, Result};
use crate::schema::DataType;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A single scalar. Used for literals, group keys, zone-map bounds and the
/// rows the CLI prints — not for bulk data, which lives in the columnar arrays
/// in `qf-storage`.
#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Boolean(bool),
    Int64(i64),
    Float64(f64),
    Utf8(String),
}

impl Value {
    pub fn data_type(&self) -> Option<DataType> {
        match self {
            Value::Null => None,
            Value::Boolean(_) => Some(DataType::Boolean),
            Value::Int64(_) => Some(DataType::Int64),
            Value::Float64(_) => Some(DataType::Float64),
            Value::Utf8(_) => Some(DataType::Utf8),
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Numeric widening, used everywhere a mixed int/float expression appears.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int64(i) => Some(*i as f64),
            Value::Float64(f) => Some(*f),
            _ => None,
        }
    }

    /// Truthiness under SQL's three-valued logic: NULL is neither true nor
    /// false, so this returns `None` rather than defaulting to false.
    pub fn as_bool(&self) -> Result<Option<bool>> {
        match self {
            Value::Null => Ok(None),
            Value::Boolean(b) => Ok(Some(*b)),
            other => Err(Error::typ(format!(
                "expected a boolean, found {}",
                other.type_name()
            ))),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "NULL",
            Value::Boolean(_) => "BOOLEAN",
            Value::Int64(_) => "INT64",
            Value::Float64(_) => "FLOAT64",
            Value::Utf8(_) => "UTF8",
        }
    }

    /// Explicit cast, as produced by `CAST(x AS t)`.
    ///
    /// Float to int truncates toward zero, which is what every engine I
    /// checked does; a string that doesn't parse is an error rather than NULL,
    /// so a typo in a literal surfaces instead of quietly filtering rows out.
    pub fn cast_to(&self, target: DataType) -> Result<Value> {
        if self.is_null() {
            return Ok(Value::Null);
        }
        let out = match (self, target) {
            (Value::Boolean(_), DataType::Boolean)
            | (Value::Int64(_), DataType::Int64)
            | (Value::Float64(_), DataType::Float64)
            | (Value::Utf8(_), DataType::Utf8) => self.clone(),

            (Value::Int64(i), DataType::Float64) => Value::Float64(*i as f64),
            (Value::Float64(f), DataType::Int64) => Value::Int64(f.trunc() as i64),
            (Value::Boolean(b), DataType::Int64) => Value::Int64(i64::from(*b)),
            (Value::Int64(i), DataType::Boolean) => Value::Boolean(*i != 0),

            (v, DataType::Utf8) => Value::Utf8(v.to_string()),
            (Value::Utf8(s), DataType::Int64) => Value::Int64(
                s.trim()
                    .parse()
                    .map_err(|_| Error::typ(format!("cannot cast `{s}` to INT64")))?,
            ),
            (Value::Utf8(s), DataType::Float64) => Value::Float64(
                s.trim()
                    .parse()
                    .map_err(|_| Error::typ(format!("cannot cast `{s}` to FLOAT64")))?,
            ),
            (Value::Utf8(s), DataType::Boolean) => match s.trim().to_ascii_lowercase().as_str() {
                "true" | "t" | "1" => Value::Boolean(true),
                "false" | "f" | "0" => Value::Boolean(false),
                _ => return Err(Error::typ(format!("cannot cast `{s}` to BOOLEAN"))),
            },
            (v, t) => return Err(Error::typ(format!("cannot cast {} to {t}", v.type_name()))),
        };
        Ok(out)
    }

    /// SQL comparison: NULL compared to anything is unknown, hence `None`.
    /// Ints and floats compare numerically across the type boundary.
    pub fn sql_compare(&self, other: &Value) -> Result<Option<Ordering>> {
        if self.is_null() || other.is_null() {
            return Ok(None);
        }
        let ord = match (self, other) {
            (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
            (Value::Utf8(a), Value::Utf8(b)) => a.cmp(b),
            (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
            (a, b) => match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => x
                    .partial_cmp(&y)
                    .ok_or_else(|| Error::typ("cannot order NaN".to_string()))?,
                _ => {
                    return Err(Error::typ(format!(
                        "cannot compare {} with {}",
                        a.type_name(),
                        b.type_name()
                    )))
                }
            },
        };
        Ok(Some(ord))
    }

    /// A deterministic total order over every value, including NULL and NaN.
    ///
    /// `sql_compare` is the right answer for `WHERE`, but sorting, `min`/`max`
    /// statistics and merge logic all need *some* answer for every pair or they
    /// cannot be implemented at all. NULLs sort first here; the sort operator
    /// flips them to last for ascending order to match Postgres.
    pub fn total_cmp(&self, other: &Value) -> Ordering {
        fn rank(v: &Value) -> u8 {
            match v {
                Value::Null => 0,
                Value::Boolean(_) => 1,
                Value::Int64(_) | Value::Float64(_) => 2,
                Value::Utf8(_) => 3,
            }
        }
        match (self, other) {
            (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
            (Value::Utf8(a), Value::Utf8(b)) => a.cmp(b),
            (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
            (a, b) if rank(a) == 2 && rank(b) == 2 => {
                a.as_f64().unwrap().total_cmp(&b.as_f64().unwrap())
            }
            (a, b) => rank(a).cmp(&rank(b)),
        }
    }

    pub fn min(a: &Value, b: &Value) -> Value {
        if a.is_null() {
            return b.clone();
        }
        if b.is_null() {
            return a.clone();
        }
        if a.total_cmp(b) == Ordering::Greater {
            b.clone()
        } else {
            a.clone()
        }
    }

    pub fn max(a: &Value, b: &Value) -> Value {
        if a.is_null() {
            return b.clone();
        }
        if b.is_null() {
            return a.clone();
        }
        if a.total_cmp(b) == Ordering::Less {
            b.clone()
        } else {
            a.clone()
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => f.write_str("NULL"),
            Value::Boolean(b) => write!(f, "{b}"),
            Value::Int64(i) => write!(f, "{i}"),
            Value::Float64(x) => {
                if x.fract() == 0.0 && x.is_finite() && x.abs() < 1e15 {
                    write!(f, "{x:.1}")
                } else {
                    write!(f, "{x}")
                }
            }
            Value::Utf8(s) => f.write_str(s),
        }
    }
}

// Grouping equality, not SQL equality: two NULLs land in the same group, and
// two identical floats hash alike. `sql_compare` remains the authority for
// `WHERE`.
impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        self.total_cmp(other) == Ordering::Equal
    }
}
impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Null => 0u8.hash(state),
            Value::Boolean(b) => {
                1u8.hash(state);
                b.hash(state);
            }
            // An Int64 and a Float64 holding the same number compare equal
            // under `total_cmp`, so they have to hash alike or a hash join
            // between an int column and a float column silently drops rows.
            Value::Int64(i) => {
                2u8.hash(state);
                (*i as f64).to_bits().hash(state);
            }
            Value::Float64(x) => {
                2u8.hash(state);
                x.to_bits().hash(state);
            }
            Value::Utf8(s) => {
                3u8.hash(state);
                s.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of(v: &Value) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    #[test]
    fn comparing_with_null_is_unknown_not_false() {
        assert_eq!(Value::Int64(1).sql_compare(&Value::Null).unwrap(), None);
        assert_eq!(Value::Null.sql_compare(&Value::Null).unwrap(), None);
    }

    #[test]
    fn ints_and_floats_compare_numerically() {
        assert_eq!(
            Value::Int64(2).sql_compare(&Value::Float64(2.5)).unwrap(),
            Some(Ordering::Less)
        );
        assert_eq!(
            Value::Float64(3.0).sql_compare(&Value::Int64(3)).unwrap(),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn an_int_and_an_equal_float_hash_alike_so_joins_do_not_drop_rows() {
        assert_eq!(Value::Int64(7), Value::Float64(7.0));
        assert_eq!(hash_of(&Value::Int64(7)), hash_of(&Value::Float64(7.0)));
    }

    #[test]
    fn comparing_across_incompatible_types_is_an_error() {
        assert!(Value::Utf8("a".into())
            .sql_compare(&Value::Int64(1))
            .is_err());
        assert!(Value::Boolean(true).sql_compare(&Value::Int64(1)).is_err());
    }

    #[test]
    fn total_order_places_nulls_first_and_is_defined_for_every_pair() {
        let mut vs = [
            Value::Utf8("z".into()),
            Value::Int64(3),
            Value::Null,
            Value::Boolean(true),
        ];
        vs.sort_by(Value::total_cmp);
        assert!(vs[0].is_null());
        assert_eq!(vs[3], Value::Utf8("z".into()));
    }

    #[test]
    fn nan_is_orderable_by_total_cmp_but_not_by_sql_compare() {
        let nan = Value::Float64(f64::NAN);
        assert!(nan.sql_compare(&Value::Float64(1.0)).is_err());
        assert_eq!(nan.total_cmp(&Value::Float64(1.0)), Ordering::Greater);
    }

    #[test]
    fn min_and_max_ignore_nulls_the_way_sql_aggregates_do() {
        assert_eq!(Value::min(&Value::Null, &Value::Int64(4)), Value::Int64(4));
        assert_eq!(Value::max(&Value::Int64(4), &Value::Null), Value::Int64(4));
        assert_eq!(
            Value::min(&Value::Int64(9), &Value::Int64(4)),
            Value::Int64(4)
        );
        assert_eq!(
            Value::max(&Value::Int64(9), &Value::Int64(4)),
            Value::Int64(9)
        );
        assert!(Value::min(&Value::Null, &Value::Null).is_null());
    }

    #[test]
    fn casts_round_trip_through_text_and_truncate_toward_zero() {
        assert_eq!(
            Value::Float64(-2.9).cast_to(DataType::Int64).unwrap(),
            Value::Int64(-2)
        );
        assert_eq!(
            Value::Utf8(" 42 ".into()).cast_to(DataType::Int64).unwrap(),
            Value::Int64(42)
        );
        assert_eq!(
            Value::Int64(1).cast_to(DataType::Boolean).unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            Value::Utf8("TRUE".into())
                .cast_to(DataType::Boolean)
                .unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            Value::Boolean(true).cast_to(DataType::Utf8).unwrap(),
            Value::Utf8("true".into())
        );
    }

    #[test]
    fn a_bad_cast_is_an_error_rather_than_a_silent_null() {
        assert!(Value::Utf8("twelve".into())
            .cast_to(DataType::Int64)
            .is_err());
        assert!(Value::Utf8("x".into()).cast_to(DataType::Float64).is_err());
        assert!(Value::Utf8("maybe".into())
            .cast_to(DataType::Boolean)
            .is_err());
        assert!(Value::Boolean(true).cast_to(DataType::Float64).is_err());
        assert!(Value::Null.cast_to(DataType::Int64).unwrap().is_null());
    }

    #[test]
    fn three_valued_truthiness() {
        assert_eq!(Value::Null.as_bool().unwrap(), None);
        assert_eq!(Value::Boolean(true).as_bool().unwrap(), Some(true));
        assert!(Value::Int64(1).as_bool().is_err());
    }

    #[test]
    fn display_keeps_floats_distinguishable_from_ints() {
        assert_eq!(Value::Float64(3.0).to_string(), "3.0");
        assert_eq!(Value::Float64(3.25).to_string(), "3.25");
        assert_eq!(Value::Int64(3).to_string(), "3");
        assert_eq!(Value::Null.to_string(), "NULL");
        assert_eq!(Value::Utf8("hi".into()).to_string(), "hi");
    }

    #[test]
    fn data_type_of_a_value_is_none_only_for_null() {
        assert_eq!(Value::Null.data_type(), None);
        assert_eq!(Value::Int64(1).data_type(), Some(DataType::Int64));
        assert_eq!(Value::Utf8("a".into()).data_type(), Some(DataType::Utf8));
        assert_eq!(Value::Null.type_name(), "NULL");
    }
}
