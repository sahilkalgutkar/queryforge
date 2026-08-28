use crate::array::Array;
use crate::bytes::{ByteReader, ByteWriter};
use qf_common::{DataType, Result, Value};
use std::cmp::Ordering;
use std::collections::HashSet;

/// The comparison operators a zone map can reason about.
///
/// The planner maps its own binary operators onto these; the storage layer has
/// no opinion about SQL syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

/// Per-column, per-row-group summary — a zone map.
///
/// This is the thing that makes predicate pushdown worth doing. Without it,
/// pushing a filter down only saves the cost of materialising rows; with it,
/// a row group whose `[min, max]` cannot satisfy the predicate is never read
/// off disk at all.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnStats {
    pub min: Value,
    pub max: Value,
    pub null_count: usize,
    pub row_count: usize,
    /// Exact for row groups (they are bounded), summed and capped when merged
    /// across row groups. The planner uses it to estimate join cardinality.
    pub distinct_count: usize,
}

impl ColumnStats {
    pub fn empty() -> Self {
        ColumnStats {
            min: Value::Null,
            max: Value::Null,
            null_count: 0,
            row_count: 0,
            distinct_count: 0,
        }
    }

    pub fn from_array(array: &Array) -> ColumnStats {
        let mut min = Value::Null;
        let mut max = Value::Null;
        let mut distinct: HashSet<Value> = HashSet::new();
        for i in 0..array.len() {
            let v = array.value(i);
            if v.is_null() {
                continue;
            }
            min = Value::min(&min, &v);
            max = Value::max(&max, &v);
            distinct.insert(v);
        }
        ColumnStats {
            min,
            max,
            null_count: array.null_count(),
            row_count: array.len(),
            distinct_count: distinct.len(),
        }
    }

    pub fn all_null(&self) -> bool {
        self.row_count > 0 && self.null_count == self.row_count
    }

    /// Selectivity is estimated as `1 / distinct` for equality, the textbook
    /// uniform-distribution assumption. It is wrong on skewed data, and the
    /// join-ordering rule that depends on it says so.
    pub fn selectivity_eq(&self) -> f64 {
        if self.distinct_count == 0 {
            return 1.0;
        }
        1.0 / self.distinct_count as f64
    }

    /// Combines the stats of two row groups into one summary of both.
    pub fn merge(a: &ColumnStats, b: &ColumnStats) -> ColumnStats {
        ColumnStats {
            min: Value::min(&a.min, &b.min),
            max: Value::max(&a.max, &b.max),
            null_count: a.null_count + b.null_count,
            row_count: a.row_count + b.row_count,
            // Distinct values may overlap between groups, so the true count is
            // somewhere between the larger of the two and their sum. Capping at
            // the row count keeps the estimate from claiming more distinct
            // values than there are rows.
            distinct_count: (a.distinct_count + b.distinct_count)
                .min(a.row_count + b.row_count)
                .max(a.distinct_count.max(b.distinct_count)),
        }
    }

    /// Can any row in this range satisfy `column <op> literal`?
    ///
    /// Answering `true` when the truth is `false` only costs a wasted read;
    /// answering `false` when the truth is `true` drops rows from the result.
    /// Every branch here defaults to `true` when it cannot prove otherwise.
    pub fn may_match(&self, op: PruneOp, literal: &Value) -> bool {
        if self.row_count == 0 {
            return false;
        }
        // A comparison against NULL is never true, whichever side it is on.
        if literal.is_null() || self.all_null() {
            return false;
        }
        if self.min.is_null() || self.max.is_null() {
            return true;
        }
        let (Ok(Some(vs_min)), Ok(Some(vs_max))) = (
            literal.sql_compare(&self.min),
            literal.sql_compare(&self.max),
        ) else {
            // Types the zone map cannot compare — let the row group through and
            // let the execution-time predicate produce the error or the answer.
            return true;
        };

        match op {
            // literal must sit inside [min, max]
            PruneOp::Eq => vs_min != Ordering::Less && vs_max != Ordering::Greater,
            // only prunable when the whole group is one value equal to the literal
            PruneOp::NotEq => {
                !(self.null_count == 0 && vs_min == Ordering::Equal && vs_max == Ordering::Equal)
            }
            // column < literal needs min < literal
            PruneOp::Lt => vs_min == Ordering::Greater,
            PruneOp::LtEq => vs_min != Ordering::Less,
            // column > literal needs max > literal
            PruneOp::Gt => vs_max == Ordering::Less,
            PruneOp::GtEq => vs_max != Ordering::Greater,
        }
    }

    pub(crate) fn encode(&self, w: &mut ByteWriter) {
        w.uvarint(self.null_count as u64);
        w.uvarint(self.row_count as u64);
        w.uvarint(self.distinct_count as u64);
        encode_value(w, &self.min);
        encode_value(w, &self.max);
    }

    pub(crate) fn decode(r: &mut ByteReader<'_>) -> Result<ColumnStats> {
        let null_count = r.uvarint()? as usize;
        let row_count = r.uvarint()? as usize;
        let distinct_count = r.uvarint()? as usize;
        let min = decode_value(r)?;
        let max = decode_value(r)?;
        Ok(ColumnStats {
            min,
            max,
            null_count,
            row_count,
            distinct_count,
        })
    }
}

pub(crate) fn encode_value(w: &mut ByteWriter, v: &Value) {
    match v {
        Value::Null => w.u8(0),
        Value::Boolean(b) => {
            w.u8(1);
            w.u8(u8::from(*b));
        }
        Value::Int64(i) => {
            w.u8(2);
            w.ivarint(*i);
        }
        Value::Float64(f) => {
            w.u8(3);
            w.f64(*f);
        }
        Value::Utf8(s) => {
            w.u8(4);
            w.string(s);
        }
    }
}

pub(crate) fn decode_value(r: &mut ByteReader<'_>) -> Result<Value> {
    Ok(match r.u8()? {
        0 => Value::Null,
        1 => Value::Boolean(r.u8()? != 0),
        2 => Value::Int64(r.ivarint()?),
        3 => Value::Float64(r.f64()?),
        4 => Value::Utf8(r.string()?),
        tag => {
            return Err(qf_common::Error::storage(format!(
                "unknown value tag {tag} in file metadata"
            )))
        }
    })
}

/// Statistics for a whole table, one entry per column.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TableStats {
    pub row_count: usize,
    pub columns: Vec<ColumnStats>,
}

impl TableStats {
    pub fn column(&self, i: usize) -> Option<&ColumnStats> {
        self.columns.get(i)
    }
}

/// Convenience for building stats from a fully materialised column.
pub fn stats_for(data_type: DataType, values: &[Value]) -> Result<ColumnStats> {
    Ok(ColumnStats::from_array(&Array::from_values(
        data_type, values,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(vals: &[Option<i64>]) -> ColumnStats {
        let values: Vec<Value> = vals
            .iter()
            .map(|v| v.map_or(Value::Null, Value::Int64))
            .collect();
        stats_for(DataType::Int64, &values).unwrap()
    }

    #[test]
    fn min_max_and_null_count_come_from_the_data() {
        let s = ints(&[Some(5), None, Some(1), Some(9)]);
        assert_eq!(s.min, Value::Int64(1));
        assert_eq!(s.max, Value::Int64(9));
        assert_eq!(s.null_count, 1);
        assert_eq!(s.row_count, 4);
        assert_eq!(s.distinct_count, 3);
    }

    #[test]
    fn equality_prunes_a_group_whose_range_excludes_the_literal() {
        let s = ints(&[Some(10), Some(20)]);
        assert!(!s.may_match(PruneOp::Eq, &Value::Int64(5)));
        assert!(!s.may_match(PruneOp::Eq, &Value::Int64(25)));
        assert!(s.may_match(PruneOp::Eq, &Value::Int64(15)));
        assert!(s.may_match(PruneOp::Eq, &Value::Int64(10)));
        assert!(s.may_match(PruneOp::Eq, &Value::Int64(20)));
    }

    #[test]
    fn range_predicates_prune_from_the_correct_end() {
        let s = ints(&[Some(10), Some(20)]);
        // column < 10 is impossible when the smallest value is 10
        assert!(!s.may_match(PruneOp::Lt, &Value::Int64(10)));
        assert!(s.may_match(PruneOp::Lt, &Value::Int64(11)));
        assert!(s.may_match(PruneOp::LtEq, &Value::Int64(10)));
        assert!(!s.may_match(PruneOp::LtEq, &Value::Int64(9)));
        // column > 20 is impossible when the largest value is 20
        assert!(!s.may_match(PruneOp::Gt, &Value::Int64(20)));
        assert!(s.may_match(PruneOp::Gt, &Value::Int64(19)));
        assert!(s.may_match(PruneOp::GtEq, &Value::Int64(20)));
        assert!(!s.may_match(PruneOp::GtEq, &Value::Int64(21)));
    }

    #[test]
    fn inequality_only_prunes_a_constant_group() {
        let constant = ints(&[Some(7), Some(7)]);
        assert!(!constant.may_match(PruneOp::NotEq, &Value::Int64(7)));
        assert!(constant.may_match(PruneOp::NotEq, &Value::Int64(8)));

        let varied = ints(&[Some(7), Some(8)]);
        assert!(varied.may_match(PruneOp::NotEq, &Value::Int64(7)));

        // A null alongside the constant still satisfies nothing under `<>`,
        // but the zone map cannot see that, so it must not prune.
        let with_null = ints(&[Some(7), None]);
        assert!(with_null.may_match(PruneOp::NotEq, &Value::Int64(7)));
    }

    #[test]
    fn an_all_null_group_matches_no_predicate() {
        let s = ints(&[None, None]);
        assert!(s.all_null());
        for op in [
            PruneOp::Eq,
            PruneOp::NotEq,
            PruneOp::Lt,
            PruneOp::LtEq,
            PruneOp::Gt,
            PruneOp::GtEq,
        ] {
            assert!(!s.may_match(op, &Value::Int64(1)), "{op:?}");
        }
    }

    #[test]
    fn comparing_against_null_never_matches() {
        let s = ints(&[Some(1), Some(2)]);
        assert!(!s.may_match(PruneOp::Eq, &Value::Null));
    }

    #[test]
    fn an_empty_group_is_always_pruned() {
        assert!(!ColumnStats::empty().may_match(PruneOp::Eq, &Value::Int64(1)));
    }

    #[test]
    fn an_incomparable_literal_leaves_the_group_readable() {
        // Comparing a string against an int column cannot be decided by the
        // zone map, so the group survives and execution reports the type error.
        let s = ints(&[Some(1), Some(2)]);
        assert!(s.may_match(PruneOp::Eq, &Value::Utf8("x".into())));
    }

    #[test]
    fn string_zone_maps_prune_lexicographically() {
        let s = stats_for(
            DataType::Utf8,
            &[Value::Utf8("banana".into()), Value::Utf8("cherry".into())],
        )
        .unwrap();
        assert!(!s.may_match(PruneOp::Eq, &Value::Utf8("apple".into())));
        assert!(s.may_match(PruneOp::Eq, &Value::Utf8("blue".into())));
        assert!(!s.may_match(PruneOp::Gt, &Value::Utf8("cherry".into())));
    }

    #[test]
    fn merging_widens_the_range_and_sums_the_counts() {
        let a = ints(&[Some(1), Some(5)]);
        let b = ints(&[Some(10), None]);
        let m = ColumnStats::merge(&a, &b);
        assert_eq!(m.min, Value::Int64(1));
        assert_eq!(m.max, Value::Int64(10));
        assert_eq!(m.row_count, 4);
        assert_eq!(m.null_count, 1);
        assert!(m.distinct_count <= m.row_count);
        assert!(m.distinct_count >= 2);
    }

    #[test]
    fn selectivity_falls_as_distinct_values_rise() {
        assert!((ints(&[Some(1), Some(2)]).selectivity_eq() - 0.5).abs() < 1e-9);
        assert!((ColumnStats::empty().selectivity_eq() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn stats_survive_a_serialisation_round_trip() {
        for s in [
            ints(&[Some(1), None, Some(3)]),
            stats_for(DataType::Utf8, &[Value::Utf8("a".into())]).unwrap(),
            stats_for(DataType::Float64, &[Value::Float64(1.5)]).unwrap(),
            stats_for(DataType::Boolean, &[Value::Boolean(true)]).unwrap(),
            ColumnStats::empty(),
        ] {
            let mut w = ByteWriter::new();
            s.encode(&mut w);
            let buf = w.into_bytes();
            let mut r = ByteReader::new(&buf);
            assert_eq!(ColumnStats::decode(&mut r).unwrap(), s);
        }
    }

    #[test]
    fn an_unknown_value_tag_is_a_storage_error() {
        let buf = vec![9u8];
        let mut r = ByteReader::new(&buf);
        assert!(decode_value(&mut r).is_err());
    }

    #[test]
    fn table_stats_index_by_column() {
        let t = TableStats {
            row_count: 2,
            columns: vec![ints(&[Some(1)])],
        };
        assert!(t.column(0).is_some());
        assert!(t.column(1).is_none());
    }
}
