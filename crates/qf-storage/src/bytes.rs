use qf_common::{Error, Result};

/// Little-endian byte writer with LEB128 varints.
///
/// Fixed-width integers everywhere would make the on-disk format simpler, but
/// most of what gets written here is small — run lengths, dictionary indices,
/// string lengths — and a varint spends one byte on those instead of eight.
#[derive(Debug, Default)]
pub struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    pub fn new() -> Self {
        ByteWriter::default()
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn uvarint(&mut self, mut v: u64) {
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            self.buf.push(byte);
            if v == 0 {
                break;
            }
        }
    }

    /// Zigzag then varint, so that small negative numbers stay small.
    pub fn ivarint(&mut self, v: i64) {
        self.uvarint(((v << 1) ^ (v >> 63)) as u64);
    }

    pub fn bytes(&mut self, b: &[u8]) {
        self.uvarint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }

    pub fn string(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }
}

/// The reading half. Every accessor is bounds-checked and returns a
/// `Storage` error rather than panicking: a truncated or corrupt file is an
/// expected failure mode, not a bug.
#[derive(Debug)]
pub struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        ByteReader { buf, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn is_done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(Error::storage(format!(
                "unexpected end of buffer: wanted {n} bytes at offset {}, {} remain",
                self.pos,
                self.remaining()
            )));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    pub fn uvarint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            let byte = self.u8()?;
            if shift >= 64 {
                return Err(Error::storage("varint overflows 64 bits".to_string()));
            }
            result |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Ok(result)
    }

    pub fn ivarint(&mut self) -> Result<i64> {
        let u = self.uvarint()?;
        Ok(((u >> 1) as i64) ^ -((u & 1) as i64))
    }

    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.uvarint()? as usize;
        self.take(n)
    }

    pub fn string(&mut self) -> Result<String> {
        let b = self.bytes()?;
        String::from_utf8(b.to_vec())
            .map_err(|_| Error::storage("column chunk holds invalid UTF-8".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_round_trip() {
        let mut w = ByteWriter::new();
        w.u8(7);
        w.u32(70_000);
        w.u64(u64::MAX);
        w.f64(-1.5);
        w.string("hello");
        assert!(!w.is_empty());
        let buf = w.into_bytes();

        let mut r = ByteReader::new(&buf);
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.u32().unwrap(), 70_000);
        assert_eq!(r.u64().unwrap(), u64::MAX);
        assert_eq!(r.f64().unwrap(), -1.5);
        assert_eq!(r.string().unwrap(), "hello");
        assert!(r.is_done());
    }

    #[test]
    fn varints_round_trip_across_the_whole_range() {
        let cases = [0i64, 1, -1, 63, 64, -64, i64::MAX, i64::MIN, 1 << 40];
        let mut w = ByteWriter::new();
        for c in cases {
            w.ivarint(c);
        }
        let buf = w.into_bytes();
        let mut r = ByteReader::new(&buf);
        for c in cases {
            assert_eq!(r.ivarint().unwrap(), c, "value {c}");
        }
    }

    #[test]
    fn small_negative_numbers_cost_one_byte_thanks_to_zigzag() {
        let mut w = ByteWriter::new();
        w.ivarint(-1);
        assert_eq!(w.len(), 1);
        let mut w = ByteWriter::new();
        w.ivarint(-1_000_000);
        assert!(w.len() < 8);
    }

    #[test]
    fn unsigned_varints_round_trip() {
        let mut w = ByteWriter::new();
        for v in [0u64, 127, 128, 300, u64::MAX] {
            w.uvarint(v);
        }
        let buf = w.into_bytes();
        let mut r = ByteReader::new(&buf);
        for v in [0u64, 127, 128, 300, u64::MAX] {
            assert_eq!(r.uvarint().unwrap(), v);
        }
    }

    #[test]
    fn a_truncated_buffer_is_a_storage_error_not_a_panic() {
        let mut w = ByteWriter::new();
        w.u64(1);
        let buf = w.into_bytes();
        let mut r = ByteReader::new(&buf[..3]);
        let err = r.u64().unwrap_err();
        assert!(matches!(err, Error::Storage(_)));
        assert!(err.to_string().contains("unexpected end of buffer"));
    }

    #[test]
    fn a_varint_that_never_terminates_is_rejected() {
        let buf = vec![0xff; 32];
        let mut r = ByteReader::new(&buf);
        assert!(r.uvarint().is_err());
    }

    #[test]
    fn invalid_utf8_in_a_string_field_is_reported() {
        let mut w = ByteWriter::new();
        w.bytes(&[0xff, 0xfe]);
        let buf = w.into_bytes();
        let mut r = ByteReader::new(&buf);
        assert!(r.string().unwrap_err().to_string().contains("UTF-8"));
    }

    #[test]
    fn the_reader_tracks_its_own_position() {
        let mut w = ByteWriter::new();
        w.u32(1);
        let buf = w.into_bytes();
        let mut r = ByteReader::new(&buf);
        assert_eq!(r.remaining(), 4);
        r.u32().unwrap();
        assert_eq!(r.position(), 4);
        assert_eq!(r.remaining(), 0);
        assert!(ByteWriter::new().as_slice().is_empty());
    }
}
