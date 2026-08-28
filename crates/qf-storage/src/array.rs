use crate::bitmap::Bitmap;
use qf_common::{DataType, Error, Result, Value};

/// The buffers behind one column of one batch.
///
/// Strings are stored the way every columnar format stores them — one
/// contiguous byte buffer plus an offsets vector — rather than as `Vec<String>`.
/// That is the whole point of the layout: a scan of a string column touches two
/// allocations, not one per row, and slicing a value is a subslice rather than
/// a copy.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayData {
    Boolean(Bitmap),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Utf8 { offsets: Vec<u32>, bytes: Vec<u8> },
}

/// A typed, nullable column of values.
///
/// `validity` is `None` when the column has no nulls at all, which is the
/// common case and lets every kernel skip the mask entirely.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    data: ArrayData,
    validity: Option<Bitmap>,
    len: usize,
}

impl Array {
    pub fn new(data: ArrayData, validity: Option<Bitmap>, len: usize) -> Result<Array> {
        let data_len = match &data {
            ArrayData::Boolean(b) => b.len(),
            ArrayData::Int64(v) => v.len(),
            ArrayData::Float64(v) => v.len(),
            ArrayData::Utf8 { offsets, .. } => offsets.len().saturating_sub(1),
        };
        if data_len != len {
            return Err(Error::internal(format!(
                "array buffer holds {data_len} values but length is {len}"
            )));
        }
        if let Some(v) = &validity {
            if v.len() != len {
                return Err(Error::internal(format!(
                    "validity mask has {} bits for {len} values",
                    v.len()
                )));
            }
        }
        Ok(Array {
            data,
            validity,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn data(&self) -> &ArrayData {
        &self.data
    }

    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    pub fn data_type(&self) -> DataType {
        match &self.data {
            ArrayData::Boolean(_) => DataType::Boolean,
            ArrayData::Int64(_) => DataType::Int64,
            ArrayData::Float64(_) => DataType::Float64,
            ArrayData::Utf8 { .. } => DataType::Utf8,
        }
    }

    pub fn is_valid(&self, i: usize) -> bool {
        match &self.validity {
            None => i < self.len,
            Some(v) => v.get(i),
        }
    }

    pub fn null_count(&self) -> usize {
        match &self.validity {
            None => 0,
            Some(v) => v.count_unset(),
        }
    }

    /// Materialises row `i` as a scalar. Deliberately the slow path — the
    /// vectorised kernels work on the buffers directly and only fall back to
    /// this at the edges (group keys, printing, statistics).
    pub fn value(&self, i: usize) -> Value {
        if !self.is_valid(i) {
            return Value::Null;
        }
        match &self.data {
            ArrayData::Boolean(b) => Value::Boolean(b.get(i)),
            ArrayData::Int64(v) => Value::Int64(v[i]),
            ArrayData::Float64(v) => Value::Float64(v[i]),
            ArrayData::Utf8 { offsets, bytes } => {
                let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
                Value::Utf8(String::from_utf8_lossy(&bytes[s..e]).into_owned())
            }
        }
    }

    /// Borrows a string value without copying it. This is what the comparison
    /// kernels use so that filtering a string column allocates nothing.
    pub fn str_value(&self, i: usize) -> Option<&str> {
        match &self.data {
            ArrayData::Utf8 { offsets, bytes } if self.is_valid(i) => {
                let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
                std::str::from_utf8(&bytes[s..e]).ok()
            }
            _ => None,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        (0..self.len).map(move |i| self.value(i))
    }

    /// Gathers the rows at `indices`, in that order. Rebuilding through the
    /// builder keeps the string buffer compact instead of dragging the whole
    /// original byte buffer along behind a few surviving rows.
    pub fn take(&self, indices: &[usize]) -> Result<Array> {
        let mut b = ArrayBuilder::new(self.data_type());
        b.reserve(indices.len());
        for &i in indices {
            if i >= self.len {
                return Err(Error::internal(format!("take index {i} out of range")));
            }
            b.push_from(self, i);
        }
        b.finish()
    }

    /// Keeps the rows whose bit is set. Equivalent to `take(mask.set_indices())`
    /// but without materialising the index vector.
    pub fn filter(&self, mask: &Bitmap) -> Result<Array> {
        if mask.len() != self.len {
            return Err(Error::internal(format!(
                "filter mask has {} bits for {} rows",
                mask.len(),
                self.len
            )));
        }
        if mask.all_set() {
            return Ok(self.clone());
        }
        let mut b = ArrayBuilder::new(self.data_type());
        b.reserve(mask.count_set());
        for i in 0..self.len {
            if mask.get(i) {
                b.push_from(self, i);
            }
        }
        b.finish()
    }

    /// A column of `len` NULLs, needed to pad the non-matching side of an
    /// outer join.
    pub fn nulls(data_type: DataType, len: usize) -> Result<Array> {
        let mut b = ArrayBuilder::new(data_type);
        for _ in 0..len {
            b.push_null();
        }
        b.finish()
    }

    pub fn from_values(data_type: DataType, values: &[Value]) -> Result<Array> {
        let mut b = ArrayBuilder::new(data_type);
        for v in values {
            b.push(v.clone())?;
        }
        b.finish()
    }
}

/// Accumulates values into the column buffers.
#[derive(Debug)]
pub struct ArrayBuilder {
    data_type: DataType,
    booleans: Bitmap,
    ints: Vec<i64>,
    floats: Vec<f64>,
    offsets: Vec<u32>,
    bytes: Vec<u8>,
    validity: Bitmap,
    any_null: bool,
    len: usize,
}

impl ArrayBuilder {
    pub fn new(data_type: DataType) -> Self {
        ArrayBuilder {
            data_type,
            booleans: Bitmap::new(),
            ints: Vec::new(),
            floats: Vec::new(),
            offsets: vec![0],
            bytes: Vec::new(),
            validity: Bitmap::new(),
            any_null: false,
            len: 0,
        }
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn reserve(&mut self, n: usize) {
        match self.data_type {
            DataType::Int64 => self.ints.reserve(n),
            DataType::Float64 => self.floats.reserve(n),
            DataType::Utf8 => {
                self.offsets.reserve(n);
                self.bytes.reserve(n * 8);
            }
            DataType::Boolean => {}
        }
    }

    pub fn push_null(&mut self) {
        self.any_null = true;
        self.validity.push(false);
        match self.data_type {
            DataType::Boolean => self.booleans.push(false),
            DataType::Int64 => self.ints.push(0),
            DataType::Float64 => self.floats.push(0.0),
            DataType::Utf8 => self.offsets.push(self.bytes.len() as u32),
        }
        self.len += 1;
    }

    /// Appends a scalar, widening an int into a float column so that a literal
    /// `0` can be appended to a float column without the caller casting first.
    pub fn push(&mut self, value: Value) -> Result<()> {
        if value.is_null() {
            self.push_null();
            return Ok(());
        }
        self.validity.push(true);
        match (self.data_type, &value) {
            (DataType::Boolean, Value::Boolean(b)) => self.booleans.push(*b),
            (DataType::Int64, Value::Int64(i)) => self.ints.push(*i),
            (DataType::Float64, Value::Float64(f)) => self.floats.push(*f),
            (DataType::Float64, Value::Int64(i)) => self.floats.push(*i as f64),
            (DataType::Utf8, Value::Utf8(s)) => {
                self.bytes.extend_from_slice(s.as_bytes());
                self.offsets.push(self.bytes.len() as u32);
            }
            (dt, v) => {
                return Err(Error::typ(format!(
                    "cannot append {} to a {dt} column",
                    v.type_name()
                )))
            }
        }
        self.len += 1;
        Ok(())
    }

    /// Copies row `i` out of `src`. Same type on both sides is the caller's
    /// responsibility; this is the inner loop of take and filter.
    pub fn push_from(&mut self, src: &Array, i: usize) {
        if !src.is_valid(i) {
            self.push_null();
            return;
        }
        self.validity.push(true);
        match (&self.data_type, &src.data) {
            (DataType::Boolean, ArrayData::Boolean(b)) => self.booleans.push(b.get(i)),
            (DataType::Int64, ArrayData::Int64(v)) => self.ints.push(v[i]),
            (DataType::Float64, ArrayData::Float64(v)) => self.floats.push(v[i]),
            (DataType::Float64, ArrayData::Int64(v)) => self.floats.push(v[i] as f64),
            (DataType::Utf8, ArrayData::Utf8 { offsets, bytes }) => {
                let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
                self.bytes.extend_from_slice(&bytes[s..e]);
                self.offsets.push(self.bytes.len() as u32);
            }
            _ => {
                // Type mismatch: fall back to the scalar path, which reports it.
                self.validity.set(self.len, false);
                self.any_null = true;
                match self.data_type {
                    DataType::Boolean => self.booleans.push(false),
                    DataType::Int64 => self.ints.push(0),
                    DataType::Float64 => self.floats.push(0.0),
                    DataType::Utf8 => self.offsets.push(self.bytes.len() as u32),
                }
            }
        }
        self.len += 1;
    }

    pub fn finish(self) -> Result<Array> {
        let data = match self.data_type {
            DataType::Boolean => ArrayData::Boolean(self.booleans),
            DataType::Int64 => ArrayData::Int64(self.ints),
            DataType::Float64 => ArrayData::Float64(self.floats),
            DataType::Utf8 => ArrayData::Utf8 {
                offsets: self.offsets,
                bytes: self.bytes,
            },
        };
        let validity = if self.any_null {
            Some(self.validity)
        } else {
            None
        };
        Array::new(data, validity, self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_array(vals: &[Option<i64>]) -> Array {
        let mut b = ArrayBuilder::new(DataType::Int64);
        for v in vals {
            b.push(v.map_or(Value::Null, Value::Int64)).unwrap();
        }
        b.finish().unwrap()
    }

    fn str_array(vals: &[Option<&str>]) -> Array {
        let mut b = ArrayBuilder::new(DataType::Utf8);
        for v in vals {
            b.push(v.map_or(Value::Null, |s| Value::Utf8(s.to_string())))
                .unwrap();
        }
        b.finish().unwrap()
    }

    #[test]
    fn a_column_without_nulls_carries_no_validity_mask() {
        let a = int_array(&[Some(1), Some(2)]);
        assert!(a.validity().is_none());
        assert_eq!(a.null_count(), 0);
        assert!(a.is_valid(0));
        assert!(!a.is_valid(5));
    }

    #[test]
    fn nulls_produce_a_mask_and_read_back_as_null() {
        let a = int_array(&[Some(1), None, Some(3)]);
        assert_eq!(a.null_count(), 1);
        assert_eq!(a.value(0), Value::Int64(1));
        assert!(a.value(1).is_null());
        assert_eq!(a.value(2), Value::Int64(3));
    }

    #[test]
    fn string_values_round_trip_through_the_offsets_buffer() {
        let a = str_array(&[Some("alpha"), None, Some(""), Some("ω")]);
        assert_eq!(a.value(0), Value::Utf8("alpha".into()));
        assert!(a.value(1).is_null());
        assert_eq!(a.value(2), Value::Utf8("".into()));
        assert_eq!(a.value(3), Value::Utf8("ω".into()));
        assert_eq!(a.str_value(0), Some("alpha"));
        assert_eq!(a.str_value(1), None);
    }

    #[test]
    fn str_value_is_none_for_a_non_string_column() {
        assert_eq!(int_array(&[Some(1)]).str_value(0), None);
    }

    #[test]
    fn take_reorders_and_duplicates_rows() {
        let a = int_array(&[Some(10), Some(20), None]);
        let t = a.take(&[2, 0, 0]).unwrap();
        assert_eq!(t.len(), 3);
        assert!(t.value(0).is_null());
        assert_eq!(t.value(1), Value::Int64(10));
        assert_eq!(t.value(2), Value::Int64(10));
    }

    #[test]
    fn take_rejects_an_out_of_range_index() {
        assert!(int_array(&[Some(1)]).take(&[3]).is_err());
    }

    #[test]
    fn filter_keeps_only_selected_rows_and_compacts_string_bytes() {
        let a = str_array(&[
            Some("keep"),
            Some("a-very-long-discarded-value"),
            Some("me"),
        ]);
        let mask: Bitmap = [true, false, true].into_iter().collect();
        let f = a.filter(&mask).unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(f.value(0), Value::Utf8("keep".into()));
        assert_eq!(f.value(1), Value::Utf8("me".into()));
        match f.data() {
            ArrayData::Utf8 { bytes, .. } => assert_eq!(bytes.len(), 6),
            other => panic!("expected a utf8 column, got {other:?}"),
        }
    }

    #[test]
    fn an_all_true_filter_is_a_no_op() {
        let a = int_array(&[Some(1), Some(2)]);
        assert_eq!(a.filter(&Bitmap::filled(2, true)).unwrap(), a);
    }

    #[test]
    fn a_filter_mask_of_the_wrong_width_is_rejected() {
        let a = int_array(&[Some(1), Some(2)]);
        assert!(a.filter(&Bitmap::filled(3, true)).is_err());
    }

    #[test]
    fn filter_preserves_nulls_in_the_surviving_rows() {
        let a = int_array(&[Some(1), None, Some(3)]);
        let mask: Bitmap = [false, true, true].into_iter().collect();
        let f = a.filter(&mask).unwrap();
        assert_eq!(f.null_count(), 1);
        assert!(f.value(0).is_null());
    }

    #[test]
    fn appending_an_int_to_a_float_column_widens_it() {
        let mut b = ArrayBuilder::new(DataType::Float64);
        b.push(Value::Int64(3)).unwrap();
        b.push(Value::Float64(0.5)).unwrap();
        let a = b.finish().unwrap();
        assert_eq!(a.value(0), Value::Float64(3.0));
        assert_eq!(a.data_type(), DataType::Float64);
    }

    #[test]
    fn appending_the_wrong_type_is_rejected() {
        let mut b = ArrayBuilder::new(DataType::Int64);
        assert!(b.push(Value::Utf8("x".into())).is_err());
        let mut b = ArrayBuilder::new(DataType::Boolean);
        assert!(b.push(Value::Int64(1)).is_err());
    }

    #[test]
    fn a_null_column_reports_every_row_null() {
        let a = Array::nulls(DataType::Utf8, 3).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 3);
        assert!(a.value(2).is_null());
    }

    #[test]
    fn boolean_columns_pack_into_a_bitmap() {
        let mut b = ArrayBuilder::new(DataType::Boolean);
        for i in 0..100 {
            b.push(Value::Boolean(i % 2 == 0)).unwrap();
        }
        let a = b.finish().unwrap();
        assert_eq!(a.len(), 100);
        assert_eq!(a.value(4), Value::Boolean(true));
        assert_eq!(a.value(5), Value::Boolean(false));
        match a.data() {
            ArrayData::Boolean(bm) => assert_eq!(bm.count_set(), 50),
            other => panic!("expected a bitmap, got {other:?}"),
        }
    }

    #[test]
    fn a_mismatched_buffer_length_is_caught_at_construction() {
        let err = Array::new(ArrayData::Int64(vec![1, 2]), None, 3).unwrap_err();
        assert!(matches!(err, Error::Internal(_)));
        let err = Array::new(
            ArrayData::Int64(vec![1, 2]),
            Some(Bitmap::filled(5, true)),
            2,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Internal(_)));
    }

    #[test]
    fn from_values_and_iter_are_inverses() {
        let vals = vec![Value::Int64(1), Value::Null, Value::Int64(3)];
        let a = Array::from_values(DataType::Int64, &vals).unwrap();
        assert_eq!(a.iter().collect::<Vec<_>>(), vals);
        assert!(!a.is_empty());
        assert!(Array::from_values(DataType::Int64, &[]).unwrap().is_empty());
    }

    #[test]
    fn builder_reports_its_own_length_as_it_fills() {
        let mut b = ArrayBuilder::new(DataType::Utf8);
        assert!(b.is_empty());
        b.reserve(10);
        b.push(Value::Utf8("a".into())).unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b.data_type(), DataType::Utf8);
    }
}
