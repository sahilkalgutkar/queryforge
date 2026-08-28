//! CSV ingest — the on-ramp into the columnar format.
//!
//! CSV is how data actually arrives, so the engine has to read it, but nothing
//! else in the engine ever touches a CSV: the loader converts to batches once
//! and everything downstream reads `.qfc`.

use crate::array::ArrayBuilder;
use crate::batch::{RecordBatch, DEFAULT_BATCH_SIZE};
use qf_common::{DataType, Error, Field, Result, Schema, Value};
use std::path::Path;
use std::sync::Arc;

/// Splits one CSV line, honouring double quotes and doubled `""` escapes.
///
/// Written by hand rather than by splitting on commas because a quoted field
/// containing a comma is not an edge case in real data — it is most address
/// columns.
pub fn split_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            }
            '"' => in_quotes = true,
            ',' if !in_quotes => {
                fields.push(std::mem::take(&mut current));
            }
            _ => current.push(c),
        }
    }
    fields.push(current);
    fields
}

/// Infers a column type from the values seen in it.
///
/// The rule is the narrowest type that fits everything: an integer column with
/// one decimal in it is a float column, and anything that fails to parse makes
/// the column text. Empty strings are treated as NULL and never widen the type,
/// which is the difference between a usable `amount` column and a text one.
fn infer_column(samples: &[&str]) -> DataType {
    let mut saw_value = false;
    let mut all_int = true;
    let mut all_float = true;
    let mut all_bool = true;
    for s in samples {
        let t = s.trim();
        if t.is_empty() {
            continue;
        }
        saw_value = true;
        if t.parse::<i64>().is_err() {
            all_int = false;
        }
        if t.parse::<f64>().is_err() {
            all_float = false;
        }
        if !matches!(
            t.to_ascii_lowercase().as_str(),
            "true" | "false" | "t" | "f"
        ) {
            all_bool = false;
        }
    }
    if !saw_value {
        return DataType::Utf8;
    }
    if all_bool {
        DataType::Boolean
    } else if all_int {
        DataType::Int64
    } else if all_float {
        DataType::Float64
    } else {
        DataType::Utf8
    }
}

/// Parses one CSV cell into the column's type. An empty cell is NULL.
fn parse_cell(cell: &str, data_type: DataType) -> Result<Value> {
    let t = cell.trim();
    if t.is_empty() {
        return Ok(Value::Null);
    }
    Value::Utf8(t.to_string()).cast_to(data_type)
}

/// The header row plus the inferred column types.
pub fn infer_schema(text: &str, sample_rows: usize) -> Result<Arc<Schema>> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| Error::storage("CSV input is empty".to_string()))?;
    let names = split_line(header);
    if names.iter().any(|n| n.trim().is_empty()) {
        return Err(Error::storage(
            "CSV header has an empty column name".to_string(),
        ));
    }

    let rows: Vec<Vec<String>> = lines.take(sample_rows).map(split_line).collect();
    let mut fields = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let samples: Vec<&str> = rows
            .iter()
            .filter_map(|r| r.get(i).map(String::as_str))
            .collect();
        let nullable = samples.iter().any(|s| s.trim().is_empty());
        fields.push(Field::new(name.trim(), infer_column(&samples), nullable));
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// Reads CSV text into batches, inferring the schema from the first
/// `sample_rows` data rows.
///
/// Fully blank lines are treated as separators and skipped; a blank *cell*
/// within a row is a NULL.
pub fn read_csv_str(text: &str, sample_rows: usize) -> Result<(Arc<Schema>, Vec<RecordBatch>)> {
    let schema = infer_schema(text, sample_rows)?;
    let batches = read_csv_with_schema(text, Arc::clone(&schema))?;
    Ok((schema, batches))
}

/// Reads CSV text against a schema that is already known.
pub fn read_csv_with_schema(text: &str, schema: Arc<Schema>) -> Result<Vec<RecordBatch>> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    lines.next(); // header

    let mut batches = Vec::new();
    let mut builders: Vec<ArrayBuilder> = schema
        .fields()
        .iter()
        .map(|f| ArrayBuilder::new(f.data_type))
        .collect();
    let mut rows_in_batch = 0usize;

    for (line_no, line) in lines.enumerate() {
        let cells = split_line(line);
        if cells.len() != schema.len() {
            return Err(Error::storage(format!(
                "row {} has {} fields, expected {}",
                line_no + 2,
                cells.len(),
                schema.len()
            )));
        }
        for (i, builder) in builders.iter_mut().enumerate() {
            let value = parse_cell(&cells[i], builder.data_type()).map_err(|e| {
                Error::storage(format!(
                    "row {}, column `{}`: {e}",
                    line_no + 2,
                    schema.field(i).map(|f| f.name.clone()).unwrap_or_default()
                ))
            })?;
            builder.push(value)?;
        }
        rows_in_batch += 1;

        if rows_in_batch == DEFAULT_BATCH_SIZE {
            batches.push(finish_batch(&schema, &mut builders)?);
            rows_in_batch = 0;
        }
    }
    if rows_in_batch > 0 {
        batches.push(finish_batch(&schema, &mut builders)?);
    }
    Ok(batches)
}

fn finish_batch(schema: &Arc<Schema>, builders: &mut Vec<ArrayBuilder>) -> Result<RecordBatch> {
    let fresh: Vec<ArrayBuilder> = schema
        .fields()
        .iter()
        .map(|f| ArrayBuilder::new(f.data_type))
        .collect();
    let done = std::mem::replace(builders, fresh);
    let columns = done
        .into_iter()
        .map(ArrayBuilder::finish)
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(Arc::clone(schema), columns)
}

pub fn read_csv_file(
    path: impl AsRef<Path>,
    sample_rows: usize,
) -> Result<(Arc<Schema>, Vec<RecordBatch>)> {
    let text = std::fs::read_to_string(path)?;
    read_csv_str(&text, sample_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_fields_may_contain_commas_and_escaped_quotes() {
        assert_eq!(
            split_line(r#"1,"Smith, John","say ""hi""",x"#),
            vec!["1", "Smith, John", r#"say "hi""#, "x"]
        );
    }

    #[test]
    fn empty_trailing_fields_are_preserved() {
        assert_eq!(split_line("a,,c,"), vec!["a", "", "c", ""]);
        assert_eq!(split_line(""), vec![""]);
    }

    #[test]
    fn types_are_inferred_as_the_narrowest_that_fits() {
        let csv = "id,price,name,active\n1,1.5,ann,true\n2,2,bob,false\n";
        let s = infer_schema(csv, 100).unwrap();
        assert_eq!(s.field(0).unwrap().data_type, DataType::Int64);
        assert_eq!(s.field(1).unwrap().data_type, DataType::Float64);
        assert_eq!(s.field(2).unwrap().data_type, DataType::Utf8);
        assert_eq!(s.field(3).unwrap().data_type, DataType::Boolean);
    }

    #[test]
    fn one_decimal_widens_an_otherwise_integer_column_to_float() {
        let s = infer_schema("n\n1\n2\n3.5\n", 100).unwrap();
        assert_eq!(s.field(0).unwrap().data_type, DataType::Float64);
    }

    #[test]
    fn one_unparseable_value_makes_the_whole_column_text() {
        let s = infer_schema("n\n1\n2\nn/a\n", 100).unwrap();
        assert_eq!(s.field(0).unwrap().data_type, DataType::Utf8);
    }

    #[test]
    fn a_blank_cell_is_null_and_does_not_widen_the_type() {
        // A blank *cell* is a null. A blank *line* is a separator and is
        // skipped, which is why this case needs a second column to be
        // unambiguous.
        let csv = "label,n\na,1\nb,\nc,3\n";
        let (schema, batches) = read_csv_str(csv, 100).unwrap();
        assert_eq!(schema.field(1).unwrap().data_type, DataType::Int64);
        assert!(schema.field(1).unwrap().nullable);
        assert!(!schema.field(0).unwrap().nullable);
        let b = &batches[0];
        assert_eq!(b.num_rows(), 3);
        assert!(b.column(1).unwrap().value(1).is_null());
        assert_eq!(b.column(1).unwrap().value(2), Value::Int64(3));
    }

    #[test]
    fn an_all_blank_column_falls_back_to_text() {
        let s = infer_schema("a,b\n1,\n2,\n", 100).unwrap();
        assert_eq!(s.field(1).unwrap().data_type, DataType::Utf8);
    }

    #[test]
    fn only_the_sampled_rows_drive_inference() {
        // The float appears after the sample window, so inference says Int64
        // and the value that does not fit is reported rather than guessed at.
        let csv = "n\n1\n2\n3.5\n";
        let s = infer_schema(csv, 2).unwrap();
        assert_eq!(s.field(0).unwrap().data_type, DataType::Int64);
        let err = read_csv_with_schema(csv, s).unwrap_err();
        assert!(err.to_string().contains("row 4"));
        assert!(err.to_string().contains("column `n`"));
    }

    #[test]
    fn a_row_with_the_wrong_field_count_is_reported_with_its_line_number() {
        let err = read_csv_str("a,b\n1,2\n3\n", 10).unwrap_err();
        assert!(err.to_string().contains("row 3"));
        assert!(err.to_string().contains("expected 2"));
    }

    #[test]
    fn values_round_trip_into_typed_columns() {
        let csv = "id,name,score,ok\n1,ann,9.5,true\n2,bob,8,false\n";
        let (_, batches) = read_csv_str(csv, 10).unwrap();
        let b = &batches[0];
        assert_eq!(b.num_rows(), 2);
        assert_eq!(b.column(0).unwrap().value(1), Value::Int64(2));
        assert_eq!(b.column(1).unwrap().value(0), Value::Utf8("ann".into()));
        assert_eq!(b.column(2).unwrap().value(1), Value::Float64(8.0));
        assert_eq!(b.column(3).unwrap().value(0), Value::Boolean(true));
    }

    #[test]
    fn large_inputs_are_split_into_batches() {
        let mut csv = String::from("n\n");
        for i in 0..(DEFAULT_BATCH_SIZE + 5) {
            csv.push_str(&format!("{i}\n"));
        }
        let (_, batches) = read_csv_str(&csv, 10).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), DEFAULT_BATCH_SIZE);
        assert_eq!(batches[1].num_rows(), 5);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let (_, batches) = read_csv_str("n\n1\n\n\n2\n", 10).unwrap();
        assert_eq!(batches[0].num_rows(), 2);
    }

    #[test]
    fn empty_input_and_empty_headers_are_rejected() {
        assert!(infer_schema("", 10).is_err());
        assert!(infer_schema("a,,c\n1,2,3\n", 10).is_err());
    }

    #[test]
    fn a_header_only_file_yields_no_batches() {
        let (schema, batches) = read_csv_str("a,b\n", 10).unwrap();
        assert_eq!(schema.len(), 2);
        assert!(batches.is_empty());
    }

    #[test]
    fn files_are_read_from_disk() {
        let mut p = std::env::temp_dir();
        p.push(format!("qf-csv-{}.csv", std::process::id()));
        std::fs::write(&p, "a,b\n1,x\n2,y\n").unwrap();
        let (schema, batches) = read_csv_file(&p, 10).unwrap();
        assert_eq!(schema.len(), 2);
        assert_eq!(batches[0].num_rows(), 2);
        std::fs::remove_file(&p).unwrap();
    }
}
