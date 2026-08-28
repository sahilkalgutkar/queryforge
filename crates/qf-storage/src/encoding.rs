use crate::array::{Array, ArrayBuilder, ArrayData};
use crate::bitmap::Bitmap;
use crate::bytes::{ByteReader, ByteWriter};
use qf_common::{DataType, Error, Result, Value};
use std::collections::HashMap;

/// How the values of one column chunk are laid out on disk.
///
/// The encoding is chosen per chunk, not per column, because the right answer
/// changes with the data: an `order_status` column can be two distinct values
/// in one row group and forty in the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Values written one after another; varint for ints, bit-packed for bools.
    Plain,
    /// A sorted-order dictionary of distinct values plus one index per row.
    /// Pays for itself as soon as values repeat.
    Dictionary,
    /// (value, run length) pairs. The win on sorted or clustered columns.
    Rle,
}

impl Encoding {
    fn tag(self) -> u8 {
        match self {
            Encoding::Plain => 0,
            Encoding::Dictionary => 1,
            Encoding::Rle => 2,
        }
    }

    fn from_tag(tag: u8) -> Result<Encoding> {
        match tag {
            0 => Ok(Encoding::Plain),
            1 => Ok(Encoding::Dictionary),
            2 => Ok(Encoding::Rle),
            other => Err(Error::storage(format!("unknown encoding tag {other}"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Encoding::Plain => "plain",
            Encoding::Dictionary => "dictionary",
            Encoding::Rle => "rle",
        }
    }
}

/// Picks an encoding by measuring the chunk rather than guessing from the type.
///
/// Two cheap signals decide it: the average run length (long runs → RLE) and
/// the ratio of distinct values to rows (few distinct → dictionary). The
/// thresholds are conservative — both encodings add a decode step, so a chunk
/// that would barely benefit stays plain.
pub fn choose_encoding(array: &Array) -> Encoding {
    let n = array.len();
    if n < 16 {
        return Encoding::Plain;
    }
    let mut runs = 1usize;
    let mut distinct: HashMap<Value, ()> = HashMap::new();
    let mut prev = array.value(0);
    distinct.insert(prev.clone(), ());
    for i in 1..n {
        let v = array.value(i);
        if v != prev {
            runs += 1;
        }
        distinct.entry(v.clone()).or_insert(());
        prev = v;
    }
    let avg_run = n as f64 / runs as f64;
    if avg_run >= 4.0 {
        return Encoding::Rle;
    }
    if (distinct.len() as f64) / (n as f64) <= 0.5 {
        return Encoding::Dictionary;
    }
    Encoding::Plain
}

/// Serialises one column chunk: encoding tag, validity mask, then the values.
///
/// Only non-null values are written. A column that is 90% null costs a bitmap
/// and almost nothing else.
pub fn encode_column(array: &Array, encoding: Encoding) -> Result<Vec<u8>> {
    let mut w = ByteWriter::new();
    w.u8(encoding.tag());
    w.uvarint(array.len() as u64);

    match array.validity() {
        None => w.u8(0),
        Some(v) => {
            w.u8(1);
            write_bitmap(&mut w, v);
        }
    }

    let values: Vec<Value> = (0..array.len())
        .filter(|i| array.is_valid(*i))
        .map(|i| array.value(i))
        .collect();

    match encoding {
        Encoding::Plain => encode_plain(&mut w, array.data_type(), &values)?,
        Encoding::Dictionary => encode_dictionary(&mut w, array.data_type(), &values)?,
        Encoding::Rle => encode_rle(&mut w, array.data_type(), &values)?,
    }
    Ok(w.into_bytes())
}

/// Serialises with whichever encoding `choose_encoding` picks.
pub fn encode_column_auto(array: &Array) -> Result<(Encoding, Vec<u8>)> {
    let enc = choose_encoding(array);
    Ok((enc, encode_column(array, enc)?))
}

pub fn decode_column(data_type: DataType, buf: &[u8]) -> Result<Array> {
    let mut r = ByteReader::new(buf);
    let encoding = Encoding::from_tag(r.u8()?)?;
    let len = r.uvarint()? as usize;
    let validity = if r.u8()? == 1 {
        Some(read_bitmap(&mut r, len)?)
    } else {
        None
    };
    let valid_count = validity.as_ref().map_or(len, Bitmap::count_set);

    let values = match encoding {
        Encoding::Plain => decode_plain(&mut r, data_type, valid_count)?,
        Encoding::Dictionary => decode_dictionary(&mut r, data_type, valid_count)?,
        Encoding::Rle => decode_rle(&mut r, data_type, valid_count)?,
    };
    if values.len() != valid_count {
        return Err(Error::storage(format!(
            "chunk holds {} values but the validity mask expects {valid_count}",
            values.len()
        )));
    }

    let mut b = ArrayBuilder::new(data_type);
    b.reserve(len);
    let mut next = 0usize;
    for i in 0..len {
        let valid = validity.as_ref().is_none_or(|v| v.get(i));
        if valid {
            b.push(values[next].clone())?;
            next += 1;
        } else {
            b.push_null();
        }
    }
    b.finish()
}

fn write_bitmap(w: &mut ByteWriter, bm: &Bitmap) {
    let words = bm.len().div_ceil(64);
    w.uvarint(words as u64);
    for wi in 0..words {
        let mut word = 0u64;
        for b in 0..64 {
            let idx = wi * 64 + b;
            if idx < bm.len() && bm.get(idx) {
                word |= 1u64 << b;
            }
        }
        w.u64(word);
    }
}

fn read_bitmap(r: &mut ByteReader<'_>, len: usize) -> Result<Bitmap> {
    let words = r.uvarint()? as usize;
    if words != len.div_ceil(64) {
        return Err(Error::storage(format!(
            "validity mask has {words} words for {len} rows"
        )));
    }
    let mut bm = Bitmap::with_capacity(len);
    let mut buf = Vec::with_capacity(words);
    for _ in 0..words {
        buf.push(r.u64()?);
    }
    for i in 0..len {
        bm.push(buf[i / 64] & (1u64 << (i % 64)) != 0);
    }
    Ok(bm)
}

fn encode_scalar(w: &mut ByteWriter, data_type: DataType, v: &Value) -> Result<()> {
    match (data_type, v) {
        (DataType::Boolean, Value::Boolean(b)) => w.u8(u8::from(*b)),
        (DataType::Int64, Value::Int64(i)) => w.ivarint(*i),
        (DataType::Float64, Value::Float64(f)) => w.f64(*f),
        (DataType::Float64, Value::Int64(i)) => w.f64(*i as f64),
        (DataType::Utf8, Value::Utf8(s)) => w.string(s),
        (dt, v) => {
            return Err(Error::storage(format!(
                "cannot encode {} into a {dt} chunk",
                v.type_name()
            )))
        }
    }
    Ok(())
}

fn decode_scalar(r: &mut ByteReader<'_>, data_type: DataType) -> Result<Value> {
    Ok(match data_type {
        DataType::Boolean => Value::Boolean(r.u8()? != 0),
        DataType::Int64 => Value::Int64(r.ivarint()?),
        DataType::Float64 => Value::Float64(r.f64()?),
        DataType::Utf8 => Value::Utf8(r.string()?),
    })
}

fn encode_plain(w: &mut ByteWriter, data_type: DataType, values: &[Value]) -> Result<()> {
    if data_type == DataType::Boolean {
        // Bit-pack rather than a byte per value.
        let bm: Bitmap = values
            .iter()
            .map(|v| matches!(v, Value::Boolean(true)))
            .collect();
        write_bitmap(w, &bm);
        return Ok(());
    }
    for v in values {
        encode_scalar(w, data_type, v)?;
    }
    Ok(())
}

fn decode_plain(r: &mut ByteReader<'_>, data_type: DataType, n: usize) -> Result<Vec<Value>> {
    if data_type == DataType::Boolean {
        let bm = read_bitmap(r, n)?;
        return Ok((0..n).map(|i| Value::Boolean(bm.get(i))).collect());
    }
    (0..n).map(|_| decode_scalar(r, data_type)).collect()
}

fn encode_dictionary(w: &mut ByteWriter, data_type: DataType, values: &[Value]) -> Result<()> {
    let mut dict: Vec<Value> = Vec::new();
    let mut index: HashMap<Value, u64> = HashMap::new();
    let mut codes: Vec<u64> = Vec::with_capacity(values.len());
    for v in values {
        let code = match index.get(v) {
            Some(c) => *c,
            None => {
                let c = dict.len() as u64;
                dict.push(v.clone());
                index.insert(v.clone(), c);
                c
            }
        };
        codes.push(code);
    }
    w.uvarint(dict.len() as u64);
    for v in &dict {
        encode_scalar(w, data_type, v)?;
    }
    for c in codes {
        w.uvarint(c);
    }
    Ok(())
}

fn decode_dictionary(r: &mut ByteReader<'_>, data_type: DataType, n: usize) -> Result<Vec<Value>> {
    let dict_len = r.uvarint()? as usize;
    let mut dict = Vec::with_capacity(dict_len);
    for _ in 0..dict_len {
        dict.push(decode_scalar(r, data_type)?);
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let code = r.uvarint()? as usize;
        let v = dict
            .get(code)
            .ok_or_else(|| Error::storage(format!("dictionary code {code} out of range")))?;
        out.push(v.clone());
    }
    Ok(out)
}

fn encode_rle(w: &mut ByteWriter, data_type: DataType, values: &[Value]) -> Result<()> {
    let mut runs: Vec<(&Value, u64)> = Vec::new();
    for v in values {
        match runs.last_mut() {
            Some((prev, count)) if *prev == v => *count += 1,
            _ => runs.push((v, 1)),
        }
    }
    w.uvarint(runs.len() as u64);
    for (v, count) in runs {
        encode_scalar(w, data_type, v)?;
        w.uvarint(count);
    }
    Ok(())
}

fn decode_rle(r: &mut ByteReader<'_>, data_type: DataType, n: usize) -> Result<Vec<Value>> {
    let run_count = r.uvarint()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..run_count {
        let v = decode_scalar(r, data_type)?;
        let count = r.uvarint()? as usize;
        if out.len() + count > n {
            return Err(Error::storage(
                "run-length chunk describes more rows than the header declares".to_string(),
            ));
        }
        for _ in 0..count {
            out.push(v.clone());
        }
    }
    Ok(out)
}

/// Bytes the chunk would occupy under each encoding — used by the CLI's
/// `\encodings` command and by the tests that assert the heuristic picks a
/// genuinely smaller layout.
pub fn measure_encodings(array: &Array) -> Result<Vec<(Encoding, usize)>> {
    let mut out = Vec::new();
    for enc in [Encoding::Plain, Encoding::Dictionary, Encoding::Rle] {
        out.push((enc, encode_column(array, enc)?.len()));
    }
    Ok(out)
}

/// Debug helper: which physical layout is behind an array.
pub fn layout_name(array: &Array) -> &'static str {
    match array.data() {
        ArrayData::Boolean(_) => "bitmap",
        ArrayData::Int64(_) => "i64 buffer",
        ArrayData::Float64(_) => "f64 buffer",
        ArrayData::Utf8 { .. } => "offsets + bytes",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(dt: DataType, values: Vec<Value>) -> Array {
        Array::from_values(dt, &values).unwrap()
    }

    fn round_trip(a: &Array, enc: Encoding) {
        let buf = encode_column(a, enc).unwrap();
        let back = decode_column(a.data_type(), &buf).unwrap();
        assert_eq!(&back, a, "{enc:?} round trip");
    }

    #[test]
    fn every_encoding_round_trips_every_type() {
        let cases = vec![
            arr(
                DataType::Int64,
                (0..40).map(|i| Value::Int64(i % 5)).collect(),
            ),
            arr(
                DataType::Float64,
                (0..40).map(|i| Value::Float64(i as f64 / 3.0)).collect(),
            ),
            arr(
                DataType::Utf8,
                (0..40)
                    .map(|i| Value::Utf8(format!("s{}", i % 4)))
                    .collect(),
            ),
            arr(
                DataType::Boolean,
                (0..40).map(|i| Value::Boolean(i % 3 == 0)).collect(),
            ),
        ];
        for a in cases {
            for enc in [Encoding::Plain, Encoding::Dictionary, Encoding::Rle] {
                round_trip(&a, enc);
            }
        }
    }

    #[test]
    fn nulls_survive_every_encoding() {
        let a = arr(
            DataType::Int64,
            (0..100)
                .map(|i| {
                    if i % 3 == 0 {
                        Value::Null
                    } else {
                        Value::Int64(i)
                    }
                })
                .collect(),
        );
        assert_eq!(a.null_count(), 34);
        for enc in [Encoding::Plain, Encoding::Dictionary, Encoding::Rle] {
            round_trip(&a, enc);
        }
    }

    #[test]
    fn a_fully_null_column_costs_almost_nothing() {
        let a = Array::nulls(DataType::Utf8, 1000).unwrap();
        let buf = encode_column(&a, Encoding::Plain).unwrap();
        // 1000 rows of validity bitmap is 128 bytes; the values cost nothing.
        assert!(buf.len() < 200, "{} bytes", buf.len());
        round_trip(&a, Encoding::Plain);
    }

    #[test]
    fn an_empty_column_round_trips() {
        for dt in [
            DataType::Int64,
            DataType::Utf8,
            DataType::Boolean,
            DataType::Float64,
        ] {
            round_trip(&arr(dt, vec![]), Encoding::Plain);
        }
    }

    #[test]
    fn rle_is_chosen_and_wins_on_a_sorted_column() {
        let a = arr(
            DataType::Int64,
            (0..1000).map(|i| Value::Int64(i / 100)).collect(),
        );
        assert_eq!(choose_encoding(&a), Encoding::Rle);
        let sizes = measure_encodings(&a).unwrap();
        let plain = sizes.iter().find(|(e, _)| *e == Encoding::Plain).unwrap().1;
        let rle = sizes.iter().find(|(e, _)| *e == Encoding::Rle).unwrap().1;
        assert!(rle * 10 < plain, "rle {rle} vs plain {plain}");
    }

    #[test]
    fn dictionary_is_chosen_and_wins_on_a_low_cardinality_shuffled_column() {
        // Low cardinality but no long runs, which is exactly the case RLE
        // cannot help with.
        let words = ["SHIPPED", "PENDING", "CANCELLED", "DELIVERED"];
        let a = arr(
            DataType::Utf8,
            (0..1000)
                .map(|i| Value::Utf8(words[(i * 7 + 3) % 4].to_string()))
                .collect(),
        );
        assert_eq!(choose_encoding(&a), Encoding::Dictionary);
        let sizes = measure_encodings(&a).unwrap();
        let plain = sizes.iter().find(|(e, _)| *e == Encoding::Plain).unwrap().1;
        let dict = sizes
            .iter()
            .find(|(e, _)| *e == Encoding::Dictionary)
            .unwrap()
            .1;
        assert!(dict * 4 < plain, "dict {dict} vs plain {plain}");
    }

    #[test]
    fn plain_is_chosen_when_neither_encoding_would_help() {
        let a = arr(
            DataType::Int64,
            (0..1000).map(|i| Value::Int64(i * 7919)).collect(),
        );
        assert_eq!(choose_encoding(&a), Encoding::Plain);
    }

    #[test]
    fn tiny_chunks_skip_the_analysis_entirely() {
        let a = arr(DataType::Int64, vec![Value::Int64(1); 8]);
        assert_eq!(choose_encoding(&a), Encoding::Plain);
    }

    #[test]
    fn auto_encoding_records_the_encoding_it_used() {
        let a = arr(DataType::Int64, vec![Value::Int64(4); 100]);
        let (enc, buf) = encode_column_auto(&a).unwrap();
        assert_eq!(enc, Encoding::Rle);
        assert_eq!(decode_column(DataType::Int64, &buf).unwrap(), a);
    }

    #[test]
    fn a_corrupt_encoding_tag_is_rejected() {
        let mut buf = encode_column(
            &arr(DataType::Int64, vec![Value::Int64(1)]),
            Encoding::Plain,
        )
        .unwrap();
        buf[0] = 9;
        assert!(decode_column(DataType::Int64, &buf).is_err());
    }

    #[test]
    fn a_dictionary_code_pointing_past_the_dictionary_is_rejected() {
        let mut w = ByteWriter::new();
        w.u8(Encoding::Dictionary.tag());
        w.uvarint(1); // one row
        w.u8(0); // no validity
        w.uvarint(1); // dictionary of one entry
        w.ivarint(5);
        w.uvarint(9); // code 9 does not exist
        let buf = w.into_bytes();
        assert!(decode_column(DataType::Int64, &buf)
            .unwrap_err()
            .to_string()
            .contains("out of range"));
    }

    #[test]
    fn an_rle_run_longer_than_the_declared_row_count_is_rejected() {
        let mut w = ByteWriter::new();
        w.u8(Encoding::Rle.tag());
        w.uvarint(2); // two rows
        w.u8(0);
        w.uvarint(1); // one run...
        w.ivarint(7);
        w.uvarint(50); // ...claiming fifty rows
        let buf = w.into_bytes();
        assert!(decode_column(DataType::Int64, &buf).is_err());
    }

    #[test]
    fn a_short_rle_chunk_is_rejected_rather_than_silently_padded() {
        let mut w = ByteWriter::new();
        w.u8(Encoding::Rle.tag());
        w.uvarint(5);
        w.u8(0);
        w.uvarint(1);
        w.ivarint(7);
        w.uvarint(2); // only two of the five rows described
        let buf = w.into_bytes();
        assert!(decode_column(DataType::Int64, &buf)
            .unwrap_err()
            .to_string()
            .contains("validity mask expects"));
    }

    #[test]
    fn a_malformed_validity_mask_is_rejected() {
        let mut w = ByteWriter::new();
        w.u8(Encoding::Plain.tag());
        w.uvarint(100);
        w.u8(1);
        w.uvarint(1); // one word cannot cover 100 rows
        w.u64(0);
        let buf = w.into_bytes();
        assert!(decode_column(DataType::Int64, &buf).is_err());
    }

    #[test]
    fn encoding_names_and_layout_names_are_reported() {
        assert_eq!(Encoding::Plain.name(), "plain");
        assert_eq!(Encoding::Dictionary.name(), "dictionary");
        assert_eq!(Encoding::Rle.name(), "rle");
        assert_eq!(layout_name(&arr(DataType::Utf8, vec![])), "offsets + bytes");
        assert_eq!(layout_name(&arr(DataType::Int64, vec![])), "i64 buffer");
        assert_eq!(layout_name(&arr(DataType::Float64, vec![])), "f64 buffer");
        assert_eq!(layout_name(&arr(DataType::Boolean, vec![])), "bitmap");
    }
}
