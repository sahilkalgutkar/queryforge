use crate::batch::RecordBatch;
use crate::format::QfcReader;
use crate::stats::{ColumnStats, TableStats};
use qf_common::{Error, Result, Schema};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Where a table's rows actually live.
#[derive(Debug, Clone)]
pub enum TableSource {
    /// Batches held in memory — what `CREATE TABLE ... VALUES` and the test
    /// fixtures produce.
    Memory(Vec<RecordBatch>),
    /// A `.qfc` file, read lazily one row group at a time.
    File(PathBuf),
}

#[derive(Debug, Clone)]
pub struct Table {
    pub name: String,
    pub schema: Arc<Schema>,
    pub source: TableSource,
    pub stats: TableStats,
}

impl Table {
    pub fn row_count(&self) -> usize {
        self.stats.row_count
    }
}

/// Name-to-table map, and the only place the planner learns what exists.
///
/// Lookups are case-insensitive, which is what SQL expects, and the map is
/// ordered so `\dt` and `SHOW TABLES` list tables the same way every run —
/// a hash map's iteration order would make the CLI's output non-deterministic
/// and its tests flaky.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    tables: BTreeMap<String, Table>,
}

impl Catalog {
    pub fn new() -> Catalog {
        Catalog::default()
    }

    fn key(name: &str) -> String {
        name.to_ascii_lowercase()
    }

    /// Registers in-memory batches, deriving the table's statistics from them.
    pub fn register_batches(
        &mut self,
        name: impl Into<String>,
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    ) -> Result<()> {
        let name = name.into();
        for b in &batches {
            if b.schema().as_ref() != schema.as_ref() {
                return Err(Error::plan(format!(
                    "a batch registered for `{name}` does not match the table schema"
                )));
            }
        }
        let stats = stats_from_batches(&schema, &batches)?;
        self.insert(Table {
            name: name.clone(),
            schema,
            source: TableSource::Memory(batches),
            stats,
        })
    }

    /// Registers a `.qfc` file. The schema and statistics come from the
    /// footer, so registering a table reads kilobytes rather than the file.
    pub fn register_file(&mut self, name: impl Into<String>, path: impl AsRef<Path>) -> Result<()> {
        let name = name.into();
        let reader = QfcReader::open(path.as_ref())?;
        let schema = Arc::clone(reader.schema());
        let stats = stats_from_file(&reader);
        self.insert(Table {
            name,
            schema,
            source: TableSource::File(path.as_ref().to_path_buf()),
            stats,
        })
    }

    fn insert(&mut self, table: Table) -> Result<()> {
        let key = Catalog::key(&table.name);
        if self.tables.contains_key(&key) {
            return Err(Error::plan(format!(
                "table `{}` is already registered",
                table.name
            )));
        }
        self.tables.insert(key, table);
        Ok(())
    }

    /// Registers a table, replacing any existing one of the same name.
    pub fn replace(&mut self, table: Table) {
        self.tables.insert(Catalog::key(&table.name), table);
    }

    pub fn get(&self, name: &str) -> Result<&Table> {
        self.tables
            .get(&Catalog::key(name))
            .ok_or_else(|| Error::plan(format!("no such table `{name}`")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tables.contains_key(&Catalog::key(name))
    }

    pub fn drop_table(&mut self, name: &str) -> Result<Table> {
        self.tables
            .remove(&Catalog::key(name))
            .ok_or_else(|| Error::plan(format!("no such table `{name}`")))
    }

    pub fn table_names(&self) -> Vec<&str> {
        self.tables.values().map(|t| t.name.as_str()).collect()
    }

    pub fn tables(&self) -> impl Iterator<Item = &Table> {
        self.tables.values()
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }
}

fn stats_from_batches(schema: &Arc<Schema>, batches: &[RecordBatch]) -> Result<TableStats> {
    let mut columns = vec![ColumnStats::empty(); schema.len()];
    let mut row_count = 0;
    for b in batches {
        row_count += b.num_rows();
        for (i, col) in columns.iter_mut().enumerate() {
            *col = ColumnStats::merge(col, &ColumnStats::from_array(b.column(i)?));
        }
    }
    Ok(TableStats { row_count, columns })
}

/// Folds the per-row-group zone maps in the footer into one table-level
/// summary. No column data is read.
///
/// `min`, `max`, `row_count` and `null_count` come out exact. `distinct_count`
/// does not — row groups can hold overlapping values and the footer cannot say
/// which — so it is an upper bound. A sketch per chunk would fix that; the
/// planner only uses distinct counts to order joins, and an over-estimate there
/// costs a suboptimal plan rather than a wrong answer.
fn stats_from_file(reader: &QfcReader) -> TableStats {
    let meta = reader.meta();
    let mut columns = vec![ColumnStats::empty(); meta.schema.len()];
    for group in &meta.row_groups {
        for (i, chunk) in group.columns.iter().enumerate() {
            columns[i] = ColumnStats::merge(&columns[i], &chunk.stats);
        }
    }
    TableStats {
        row_count: meta.num_rows(),
        columns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::format::write_file;
    use qf_common::{DataType, Field, Value};

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("tag", DataType::Utf8, false),
        ]))
    }

    fn batch(ids: &[i64]) -> RecordBatch {
        let idv: Vec<Value> = ids.iter().map(|i| Value::Int64(*i)).collect();
        let tags: Vec<Value> = ids
            .iter()
            .map(|i| Value::Utf8(format!("t{}", i % 2)))
            .collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Array::from_values(DataType::Int64, &idv).unwrap(),
                Array::from_values(DataType::Utf8, &tags).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn tables_are_found_case_insensitively() {
        let mut c = Catalog::new();
        c.register_batches("Orders", schema(), vec![batch(&[1, 2])])
            .unwrap();
        assert!(c.get("orders").is_ok());
        assert!(c.get("ORDERS").is_ok());
        assert!(c.contains("Orders"));
        assert_eq!(c.get("orders").unwrap().name, "Orders");
    }

    #[test]
    fn statistics_are_derived_when_batches_are_registered() {
        let mut c = Catalog::new();
        c.register_batches("t", schema(), vec![batch(&[3, 1, 2]), batch(&[9])])
            .unwrap();
        let t = c.get("t").unwrap();
        assert_eq!(t.row_count(), 4);
        let id_stats = t.stats.column(0).unwrap();
        assert_eq!(id_stats.min, Value::Int64(1));
        assert_eq!(id_stats.max, Value::Int64(9));
        assert_eq!(id_stats.row_count, 4);
    }

    #[test]
    fn registering_the_same_name_twice_is_refused() {
        let mut c = Catalog::new();
        c.register_batches("t", schema(), vec![]).unwrap();
        let err = c.register_batches("T", schema(), vec![]).unwrap_err();
        assert!(err.to_string().contains("already registered"));
    }

    #[test]
    fn replace_overwrites_an_existing_registration() {
        let mut c = Catalog::new();
        c.register_batches("t", schema(), vec![batch(&[1])])
            .unwrap();
        let replacement = Table {
            name: "t".into(),
            schema: schema(),
            source: TableSource::Memory(vec![batch(&[1, 2, 3])]),
            stats: TableStats::default(),
        };
        c.replace(replacement);
        assert_eq!(c.len(), 1);
        match &c.get("t").unwrap().source {
            TableSource::Memory(b) => assert_eq!(b[0].num_rows(), 3),
            other => panic!("expected memory batches, got {other:?}"),
        }
    }

    #[test]
    fn a_batch_that_contradicts_the_declared_schema_is_refused() {
        let other = Arc::new(Schema::new(vec![Field::new("z", DataType::Int64, false)]));
        let mut c = Catalog::new();
        assert!(c
            .register_batches("t", other, vec![batch(&[1])])
            .unwrap_err()
            .to_string()
            .contains("does not match"));
    }

    #[test]
    fn file_backed_tables_take_their_schema_and_stats_from_the_footer() {
        let mut p = std::env::temp_dir();
        p.push(format!("qf-catalog-{}.qfc", std::process::id()));
        let rows: Vec<i64> = (0..500).collect();
        write_file(&p, schema(), &[batch(&rows)], 100).unwrap();

        let mut c = Catalog::new();
        c.register_file("sales", &p).unwrap();
        let t = c.get("sales").unwrap();
        assert_eq!(t.row_count(), 500);
        assert_eq!(t.schema.as_ref(), schema().as_ref());
        assert_eq!(t.stats.column(0).unwrap().min, Value::Int64(0));
        assert_eq!(t.stats.column(0).unwrap().max, Value::Int64(499));
        // `tag` really holds two distinct values, but the merged estimate is
        // an upper bound: each of the five row groups reports 2 and the merge
        // cannot tell that they are the same two. It stays bounded by the row
        // count, which is the property the join-ordering rule relies on.
        let tag = t.stats.column(1).unwrap();
        assert!(tag.distinct_count >= 2);
        assert!(tag.distinct_count <= tag.row_count);
        assert!(matches!(t.source, TableSource::File(_)));
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn registering_a_file_that_does_not_exist_reports_the_io_error() {
        let mut c = Catalog::new();
        assert!(c.register_file("x", "/nonexistent/path.qfc").is_err());
    }

    #[test]
    fn missing_tables_are_reported_by_name() {
        let c = Catalog::new();
        assert!(c.is_empty());
        assert!(c.get("ghost").unwrap_err().to_string().contains("`ghost`"));
    }

    #[test]
    fn tables_are_listed_in_a_stable_order() {
        let mut c = Catalog::new();
        for n in ["zebra", "apple", "mango"] {
            c.register_batches(n, schema(), vec![]).unwrap();
        }
        assert_eq!(c.table_names(), vec!["apple", "mango", "zebra"]);
        assert_eq!(c.tables().count(), 3);
    }

    #[test]
    fn dropping_removes_a_table_and_reports_an_unknown_one() {
        let mut c = Catalog::new();
        c.register_batches("t", schema(), vec![]).unwrap();
        assert_eq!(c.drop_table("T").unwrap().name, "t");
        assert!(c.is_empty());
        assert!(c.drop_table("t").is_err());
    }
}
