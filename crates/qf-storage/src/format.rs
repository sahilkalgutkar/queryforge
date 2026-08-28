//! `.qfc` — the columnar file format the engine reads and writes.
//!
//! ```text
//! ┌────────┬──────────────────────────┬─────────┬────────────┬────────┐
//! │ "QFC1" │ row group 0..n           │ footer  │ footer len │ "QFC1" │
//! │  4 B   │ column chunks, back to   │ metadata│   u32 LE   │  4 B   │
//! │        │ back, one per column     │         │            │        │
//! └────────┴──────────────────────────┴─────────┴────────────┴────────┘
//! ```
//!
//! The footer lives at the end, and its length is the last thing before the
//! trailing magic, so a reader seeks to `len - 8`, learns where the metadata
//! starts, and reads it in one go — without touching the data.
//!
//! That layout is what makes projection and predicate pushdown physical rather
//! than cosmetic. Each column chunk records its own byte range and its own
//! zone map, so a query that touches two of forty columns reads two chunks per
//! row group, and a query whose predicate falls outside a row group's `[min,
//! max]` reads none of it at all.

use crate::array::Array;
use crate::batch::RecordBatch;
use crate::bytes::{ByteReader, ByteWriter};
use crate::encoding::{decode_column, encode_column_auto, Encoding};
use crate::stats::ColumnStats;
use qf_common::{DataType, Error, Field, Result, Schema};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"QFC1";

/// Rows per row group. This is the granularity of zone-map pruning: smaller
/// groups prune more precisely but multiply the footer, and past a point the
/// metadata costs more than the reads it saves.
pub const DEFAULT_ROW_GROUP_SIZE: usize = 65_536;

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnChunkMeta {
    pub offset: u64,
    pub length: u64,
    pub encoding: Encoding,
    pub stats: ColumnStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RowGroupMeta {
    pub num_rows: usize,
    pub columns: Vec<ColumnChunkMeta>,
}

impl RowGroupMeta {
    pub fn column(&self, i: usize) -> Result<&ColumnChunkMeta> {
        self.columns
            .get(i)
            .ok_or_else(|| Error::storage(format!("row group has no column {i}")))
    }

    /// Bytes this row group occupies on disk.
    pub fn byte_size(&self) -> u64 {
        self.columns.iter().map(|c| c.length).sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FileMeta {
    pub schema: Arc<Schema>,
    pub row_groups: Vec<RowGroupMeta>,
}

impl FileMeta {
    pub fn num_rows(&self) -> usize {
        self.row_groups.iter().map(|g| g.num_rows).sum()
    }
}

fn type_tag(dt: DataType) -> u8 {
    match dt {
        DataType::Boolean => 0,
        DataType::Int64 => 1,
        DataType::Float64 => 2,
        DataType::Utf8 => 3,
    }
}

fn type_from_tag(tag: u8) -> Result<DataType> {
    Ok(match tag {
        0 => DataType::Boolean,
        1 => DataType::Int64,
        2 => DataType::Float64,
        3 => DataType::Utf8,
        other => return Err(Error::storage(format!("unknown type tag {other}"))),
    })
}

fn encode_footer(meta: &FileMeta) -> Vec<u8> {
    let mut w = ByteWriter::new();
    w.uvarint(meta.schema.len() as u64);
    for f in meta.schema.fields() {
        w.string(&f.name);
        w.u8(type_tag(f.data_type));
        w.u8(u8::from(f.nullable));
    }
    w.uvarint(meta.row_groups.len() as u64);
    for g in &meta.row_groups {
        w.uvarint(g.num_rows as u64);
        w.uvarint(g.columns.len() as u64);
        for c in &g.columns {
            w.u64(c.offset);
            w.u64(c.length);
            w.u8(match c.encoding {
                Encoding::Plain => 0,
                Encoding::Dictionary => 1,
                Encoding::Rle => 2,
            });
            c.stats.encode(&mut w);
        }
    }
    w.into_bytes()
}

fn decode_footer(buf: &[u8]) -> Result<FileMeta> {
    let mut r = ByteReader::new(buf);
    let field_count = r.uvarint()? as usize;
    let mut fields = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        let name = r.string()?;
        let data_type = type_from_tag(r.u8()?)?;
        let nullable = r.u8()? != 0;
        fields.push(Field::new(name, data_type, nullable));
    }
    let group_count = r.uvarint()? as usize;
    let mut row_groups = Vec::with_capacity(group_count);
    for _ in 0..group_count {
        let num_rows = r.uvarint()? as usize;
        let col_count = r.uvarint()? as usize;
        if col_count != field_count {
            return Err(Error::storage(format!(
                "row group describes {col_count} columns but the schema has {field_count}"
            )));
        }
        let mut columns = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            let offset = r.u64()?;
            let length = r.u64()?;
            let encoding = match r.u8()? {
                0 => Encoding::Plain,
                1 => Encoding::Dictionary,
                2 => Encoding::Rle,
                other => return Err(Error::storage(format!("unknown encoding tag {other}"))),
            };
            let stats = ColumnStats::decode(&mut r)?;
            columns.push(ColumnChunkMeta {
                offset,
                length,
                encoding,
                stats,
            });
        }
        row_groups.push(RowGroupMeta { num_rows, columns });
    }
    Ok(FileMeta {
        schema: Arc::new(Schema::new(fields)),
        row_groups,
    })
}

/// Buffers batches into row groups and writes them out.
pub struct QfcWriter {
    file: BufWriter<File>,
    schema: Arc<Schema>,
    row_group_size: usize,
    pending: Vec<RecordBatch>,
    pending_rows: usize,
    row_groups: Vec<RowGroupMeta>,
    offset: u64,
    finished: bool,
}

impl QfcWriter {
    pub fn create(path: impl AsRef<Path>, schema: Arc<Schema>) -> Result<QfcWriter> {
        QfcWriter::create_with_row_group_size(path, schema, DEFAULT_ROW_GROUP_SIZE)
    }

    pub fn create_with_row_group_size(
        path: impl AsRef<Path>,
        schema: Arc<Schema>,
        row_group_size: usize,
    ) -> Result<QfcWriter> {
        if row_group_size == 0 {
            return Err(Error::storage(
                "row group size must be positive".to_string(),
            ));
        }
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(MAGIC)?;
        Ok(QfcWriter {
            file,
            schema,
            row_group_size,
            pending: Vec::new(),
            pending_rows: 0,
            row_groups: Vec::new(),
            offset: MAGIC.len() as u64,
            finished: false,
        })
    }

    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.schema().as_ref() != self.schema.as_ref() {
            return Err(Error::storage(
                "batch schema does not match the file schema".to_string(),
            ));
        }
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.pending_rows += batch.num_rows();
        self.pending.push(batch.clone());
        while self.pending_rows >= self.row_group_size {
            self.flush_row_group(self.row_group_size)?;
        }
        Ok(())
    }

    fn flush_row_group(&mut self, max_rows: usize) -> Result<()> {
        if self.pending_rows == 0 {
            return Ok(());
        }
        let combined = RecordBatch::concat(Arc::clone(&self.schema), &self.pending)?;
        let take = max_rows.min(combined.num_rows());
        let group_rows: Vec<usize> = (0..take).collect();
        let group = combined.take(&group_rows)?;

        let mut columns = Vec::with_capacity(group.num_columns());
        for ci in 0..group.num_columns() {
            let array = group.column(ci)?;
            let (encoding, bytes) = encode_column_auto(array)?;
            self.file.write_all(&bytes)?;
            columns.push(ColumnChunkMeta {
                offset: self.offset,
                length: bytes.len() as u64,
                encoding,
                stats: ColumnStats::from_array(array),
            });
            self.offset += bytes.len() as u64;
        }
        self.row_groups.push(RowGroupMeta {
            num_rows: take,
            columns,
        });

        // Carry the remainder into the next row group.
        let rest: Vec<usize> = (take..combined.num_rows()).collect();
        self.pending_rows = rest.len();
        self.pending = if rest.is_empty() {
            Vec::new()
        } else {
            vec![combined.take(&rest)?]
        };
        Ok(())
    }

    pub fn finish(mut self) -> Result<FileMeta> {
        self.flush_row_group(self.row_group_size)?;
        let meta = FileMeta {
            schema: Arc::clone(&self.schema),
            row_groups: std::mem::take(&mut self.row_groups),
        };
        let footer = encode_footer(&meta);
        self.file.write_all(&footer)?;
        self.file.write_all(&(footer.len() as u32).to_le_bytes())?;
        self.file.write_all(MAGIC)?;
        self.file.flush()?;
        self.finished = true;
        Ok(meta)
    }
}

impl Drop for QfcWriter {
    fn drop(&mut self) {
        if !self.finished {
            // A file without a footer is unreadable, and silently leaving one
            // behind is worse than a loud warning: the next run would report a
            // corrupt file with no hint of why.
            eprintln!(
                "warning: a .qfc writer was dropped without finish(); the file has no footer"
            );
        }
    }
}

/// Reads a `.qfc` file, one column chunk at a time.
#[derive(Debug)]
pub struct QfcReader {
    file: File,
    meta: FileMeta,
}

impl QfcReader {
    pub fn open(path: impl AsRef<Path>) -> Result<QfcReader> {
        let mut file = File::open(path)?;
        let size = file.seek(SeekFrom::End(0))?;
        if size < (MAGIC.len() * 2 + 4) as u64 {
            return Err(Error::storage("file is too short to be a .qfc".to_string()));
        }

        let mut head = [0u8; 4];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut head)?;
        if &head != MAGIC {
            return Err(Error::storage(
                "file does not start with the .qfc magic".to_string(),
            ));
        }

        let mut tail = [0u8; 8];
        file.seek(SeekFrom::End(-8))?;
        file.read_exact(&mut tail)?;
        if &tail[4..] != MAGIC {
            return Err(Error::storage(
                "file does not end with the .qfc magic — it may be truncated".to_string(),
            ));
        }
        let footer_len = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as u64;
        if footer_len + 8 > size {
            return Err(Error::storage(
                "footer length exceeds file size".to_string(),
            ));
        }

        file.seek(SeekFrom::Start(size - 8 - footer_len))?;
        let mut footer = vec![0u8; footer_len as usize];
        file.read_exact(&mut footer)?;
        let meta = decode_footer(&footer)?;

        Ok(QfcReader { file, meta })
    }

    pub fn schema(&self) -> &Arc<Schema> {
        &self.meta.schema
    }

    pub fn meta(&self) -> &FileMeta {
        &self.meta
    }

    pub fn num_row_groups(&self) -> usize {
        self.meta.row_groups.len()
    }

    pub fn num_rows(&self) -> usize {
        self.meta.num_rows()
    }

    pub fn row_group(&self, i: usize) -> Result<&RowGroupMeta> {
        self.meta
            .row_groups
            .get(i)
            .ok_or_else(|| Error::storage(format!("no row group {i}")))
    }

    /// Reads exactly the chunks named by `columns` from one row group.
    ///
    /// The returned batch's schema is the projected one, so a caller asking for
    /// two of forty columns gets a two-column batch and the other thirty-eight
    /// chunks are never touched.
    pub fn read_row_group(&mut self, group: usize, columns: &[usize]) -> Result<RecordBatch> {
        let meta = self.row_group(group)?.clone();
        let schema = Arc::new(self.meta.schema.project(columns)?);
        let mut arrays: Vec<Array> = Vec::with_capacity(columns.len());
        for &ci in columns {
            let chunk = meta.column(ci)?;
            let data_type = self.meta.schema.field(ci)?.data_type;
            let mut buf = vec![0u8; chunk.length as usize];
            self.file.seek(SeekFrom::Start(chunk.offset))?;
            self.file.read_exact(&mut buf)?;
            let array = decode_column(data_type, &buf)?;
            if array.len() != meta.num_rows {
                return Err(Error::storage(format!(
                    "column {ci} of row group {group} decoded {} rows, expected {}",
                    array.len(),
                    meta.num_rows
                )));
            }
            arrays.push(array);
        }
        if arrays.is_empty() {
            // A count-only query projects nothing, but the row count still has
            // to come back — and it comes from the footer, without reading a
            // single byte of column data.
            return RecordBatch::empty(schema)?.take(&vec![0usize; meta.num_rows]);
        }
        RecordBatch::try_new(schema, arrays)
    }

    /// Every row group, every column — the unoptimised path, kept for tests
    /// and for `\dump`.
    pub fn read_all(&mut self) -> Result<Vec<RecordBatch>> {
        let all: Vec<usize> = (0..self.meta.schema.len()).collect();
        let mut out = Vec::with_capacity(self.num_row_groups());
        for g in 0..self.num_row_groups() {
            out.push(self.read_row_group(g, &all)?);
        }
        Ok(out)
    }
}

/// Writes a set of batches to a file in one call.
pub fn write_file(
    path: impl AsRef<Path>,
    schema: Arc<Schema>,
    batches: &[RecordBatch],
    row_group_size: usize,
) -> Result<FileMeta> {
    let mut w = QfcWriter::create_with_row_group_size(path, schema, row_group_size)?;
    for b in batches {
        w.write_batch(b)?;
    }
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use qf_common::Value;
    use std::path::PathBuf;

    struct TempPath(PathBuf);

    impl TempPath {
        fn new(name: &str) -> TempPath {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "qf-{}-{}-{}.qfc",
                name,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            TempPath(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Float64, true),
        ]))
    }

    fn batch(range: std::ops::Range<i64>) -> RecordBatch {
        let ids: Vec<Value> = range.clone().map(Value::Int64).collect();
        let regions: Vec<Value> = range
            .clone()
            .map(|i| Value::Utf8(format!("r{}", i % 4)))
            .collect();
        let amounts: Vec<Value> = range
            .map(|i| {
                if i % 10 == 0 {
                    Value::Null
                } else {
                    Value::Float64(i as f64 * 1.5)
                }
            })
            .collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Array::from_values(DataType::Int64, &ids).unwrap(),
                Array::from_values(DataType::Utf8, &regions).unwrap(),
                Array::from_values(DataType::Float64, &amounts).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn a_file_round_trips_through_write_and_read() {
        let tmp = TempPath::new("roundtrip");
        let original = batch(0..250);
        write_file(tmp.path(), schema(), std::slice::from_ref(&original), 100).unwrap();

        let mut r = QfcReader::open(tmp.path()).unwrap();
        assert_eq!(r.schema().as_ref(), schema().as_ref());
        assert_eq!(r.num_rows(), 250);
        assert_eq!(r.num_row_groups(), 3);
        let read_back = RecordBatch::concat(schema(), &r.read_all().unwrap()).unwrap();
        assert_eq!(read_back, original);
    }

    #[test]
    fn batches_are_repacked_into_even_row_groups() {
        let tmp = TempPath::new("repack");
        // Three uneven batches must still come out as 100/100/40.
        let meta = write_file(
            tmp.path(),
            schema(),
            &[batch(0..30), batch(30..210), batch(210..240)],
            100,
        )
        .unwrap();
        let sizes: Vec<usize> = meta.row_groups.iter().map(|g| g.num_rows).collect();
        assert_eq!(sizes, vec![100, 100, 40]);
        assert_eq!(meta.num_rows(), 240);
    }

    #[test]
    fn reading_a_subset_of_columns_returns_only_those_columns() {
        let tmp = TempPath::new("project");
        write_file(tmp.path(), schema(), &[batch(0..50)], 100).unwrap();
        let mut r = QfcReader::open(tmp.path()).unwrap();
        let b = r.read_row_group(0, &[2, 0]).unwrap();
        assert_eq!(b.num_columns(), 2);
        assert_eq!(b.schema().field(0).unwrap().name, "amount");
        assert_eq!(b.schema().field(1).unwrap().name, "id");
        assert_eq!(b.num_rows(), 50);
        assert_eq!(b.column(1).unwrap().value(3), Value::Int64(3));
    }

    #[test]
    fn each_column_chunk_carries_a_zone_map_the_planner_can_prune_with() {
        let tmp = TempPath::new("zonemap");
        // Row groups of 100 over a sorted id column: group 0 holds 0..99,
        // group 1 holds 100..199.
        let meta = write_file(tmp.path(), schema(), &[batch(0..200)], 100).unwrap();
        let g0 = &meta.row_groups[0].columns[0].stats;
        let g1 = &meta.row_groups[1].columns[0].stats;
        assert_eq!(g0.min, Value::Int64(0));
        assert_eq!(g0.max, Value::Int64(99));
        assert_eq!(g1.min, Value::Int64(100));
        assert_eq!(g1.max, Value::Int64(199));

        use crate::stats::PruneOp;
        assert!(!g0.may_match(PruneOp::Gt, &Value::Int64(150)));
        assert!(g1.may_match(PruneOp::Gt, &Value::Int64(150)));
    }

    #[test]
    fn null_counts_are_recorded_per_row_group() {
        let tmp = TempPath::new("nulls");
        let meta = write_file(tmp.path(), schema(), &[batch(0..100)], 100).unwrap();
        assert_eq!(meta.row_groups[0].columns[2].stats.null_count, 10);
        assert_eq!(meta.row_groups[0].columns[0].stats.null_count, 0);
    }

    #[test]
    fn the_low_cardinality_column_picks_a_compressing_encoding() {
        let tmp = TempPath::new("encodings");
        let meta = write_file(tmp.path(), schema(), &[batch(0..1000)], 1000).unwrap();
        // `region` cycles through four values, so it must not stay plain.
        assert_ne!(meta.row_groups[0].columns[1].encoding, Encoding::Plain);
        assert!(meta.row_groups[0].byte_size() > 0);
    }

    #[test]
    fn an_empty_batch_writes_no_row_group() {
        let tmp = TempPath::new("emptybatch");
        let meta = write_file(
            tmp.path(),
            schema(),
            &[RecordBatch::empty(schema()).unwrap()],
            100,
        )
        .unwrap();
        assert_eq!(meta.row_groups.len(), 0);
        let r = QfcReader::open(tmp.path()).unwrap();
        assert_eq!(r.num_rows(), 0);
        assert_eq!(r.num_row_groups(), 0);
    }

    #[test]
    fn writing_a_batch_with_the_wrong_schema_is_rejected() {
        let tmp = TempPath::new("badschema");
        let other = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let mut w = QfcWriter::create(tmp.path(), schema()).unwrap();
        assert!(w
            .write_batch(&RecordBatch::empty(other).unwrap())
            .is_err_and(|e| e.to_string().contains("does not match")));
        w.finish().unwrap();
    }

    #[test]
    fn a_zero_row_group_size_is_rejected() {
        let tmp = TempPath::new("zerorg");
        assert!(QfcWriter::create_with_row_group_size(tmp.path(), schema(), 0).is_err());
    }

    #[test]
    fn a_file_with_the_wrong_magic_is_refused() {
        let tmp = TempPath::new("badmagic");
        std::fs::write(tmp.path(), b"NOPEnot-a-real-file-at-allQFC1").unwrap();
        assert!(QfcReader::open(tmp.path())
            .unwrap_err()
            .to_string()
            .contains("magic"));
    }

    #[test]
    fn a_truncated_file_is_refused_rather_than_read_as_garbage() {
        let tmp = TempPath::new("truncated");
        write_file(tmp.path(), schema(), &[batch(0..50)], 100).unwrap();
        let bytes = std::fs::read(tmp.path()).unwrap();
        std::fs::write(tmp.path(), &bytes[..bytes.len() - 6]).unwrap();
        assert!(QfcReader::open(tmp.path()).is_err());
    }

    #[test]
    fn a_file_shorter_than_its_own_header_is_refused() {
        let tmp = TempPath::new("tiny");
        std::fs::write(tmp.path(), b"QF").unwrap();
        assert!(QfcReader::open(tmp.path())
            .unwrap_err()
            .to_string()
            .contains("too short"));
    }

    #[test]
    fn a_corrupt_footer_length_is_refused() {
        let tmp = TempPath::new("badfooterlen");
        write_file(tmp.path(), schema(), &[batch(0..10)], 100).unwrap();
        let mut bytes = std::fs::read(tmp.path()).unwrap();
        let n = bytes.len();
        bytes[n - 8..n - 4].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(tmp.path(), &bytes).unwrap();
        assert!(QfcReader::open(tmp.path())
            .unwrap_err()
            .to_string()
            .contains("exceeds file size"));
    }

    #[test]
    fn asking_for_a_row_group_that_does_not_exist_is_an_error() {
        let tmp = TempPath::new("norg");
        write_file(tmp.path(), schema(), &[batch(0..10)], 100).unwrap();
        let mut r = QfcReader::open(tmp.path()).unwrap();
        assert!(r.row_group(4).is_err());
        assert!(r.read_row_group(0, &[9]).is_err());
    }

    #[test]
    fn footer_metadata_survives_its_own_encoding() {
        let meta = FileMeta {
            schema: schema(),
            row_groups: vec![RowGroupMeta {
                num_rows: 5,
                columns: vec![
                    ColumnChunkMeta {
                        offset: 4,
                        length: 20,
                        encoding: Encoding::Rle,
                        stats: ColumnStats::empty(),
                    };
                    3
                ],
            }],
        };
        let buf = encode_footer(&meta);
        assert_eq!(decode_footer(&buf).unwrap(), meta);
    }

    #[test]
    fn a_footer_whose_column_count_contradicts_the_schema_is_refused() {
        let meta = FileMeta {
            schema: schema(),
            row_groups: vec![RowGroupMeta {
                num_rows: 1,
                columns: vec![ColumnChunkMeta {
                    offset: 0,
                    length: 1,
                    encoding: Encoding::Plain,
                    stats: ColumnStats::empty(),
                }],
            }],
        };
        let buf = encode_footer(&meta);
        assert!(decode_footer(&buf)
            .unwrap_err()
            .to_string()
            .contains("but the schema has"));
    }

    #[test]
    fn unknown_tags_in_a_footer_are_refused() {
        assert!(type_from_tag(7).is_err());
        for dt in [
            DataType::Boolean,
            DataType::Int64,
            DataType::Float64,
            DataType::Utf8,
        ] {
            assert_eq!(type_from_tag(type_tag(dt)).unwrap(), dt);
        }
    }
}
