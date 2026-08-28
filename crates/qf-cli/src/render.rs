//! Turning result batches into something readable in a terminal.

use qf_common::Schema;
use qf_storage::batch::RecordBatch;

/// Columns wider than this are truncated with an ellipsis, so one long text
/// value cannot push every other column off the screen.
const MAX_COLUMN_WIDTH: usize = 40;

/// Renders batches as a bordered table.
pub fn table(schema: &Schema, batches: &[RecordBatch]) -> String {
    let headers: Vec<String> = schema.fields().iter().map(|f| f.name.clone()).collect();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for b in batches {
        for r in b.rows() {
            rows.push(r.iter().map(|v| truncate(&v.to_string())).collect());
        }
    }

    if headers.is_empty() {
        return format!("({} rows, no columns)\n", rows.len());
    }

    let widths: Vec<usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            rows.iter()
                .filter_map(|r| r.get(i))
                .map(|c| display_width(c))
                .chain(std::iter::once(display_width(h)))
                .max()
                .unwrap_or(0)
        })
        .collect();

    let mut out = String::new();
    out.push_str(&border(&widths, '┌', '┬', '┐'));
    out.push_str(&row(&headers, &widths));
    out.push_str(&border(&widths, '├', '┼', '┤'));
    for r in &rows {
        out.push_str(&row(r, &widths));
    }
    out.push_str(&border(&widths, '└', '┴', '┘'));
    out.push_str(&format!(
        "{} row{}\n",
        rows.len(),
        if rows.len() == 1 { "" } else { "s" }
    ));
    out
}

fn border(widths: &[usize], left: char, mid: char, right: char) -> String {
    let mut s = String::new();
    s.push(left);
    for (i, w) in widths.iter().enumerate() {
        s.push_str(&"─".repeat(w + 2));
        s.push(if i + 1 == widths.len() { right } else { mid });
    }
    s.push('\n');
    s
}

fn row(cells: &[String], widths: &[usize]) -> String {
    let mut s = String::from("│");
    for (i, w) in widths.iter().enumerate() {
        let cell = cells.get(i).cloned().unwrap_or_default();
        let pad = w.saturating_sub(display_width(&cell));
        s.push(' ');
        s.push_str(&cell);
        s.push_str(&" ".repeat(pad));
        s.push_str(" │");
    }
    s.push('\n');
    s
}

/// Character count rather than byte length, so a column holding `naïve` is not
/// padded as though it were six characters wide.
fn display_width(s: &str) -> usize {
    s.chars().count()
}

fn truncate(s: &str) -> String {
    if display_width(s) <= MAX_COLUMN_WIDTH {
        return s.to_string();
    }
    let kept: String = s.chars().take(MAX_COLUMN_WIDTH - 1).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use qf_common::{DataType, Field, Value};
    use qf_storage::array::Array;
    use std::sync::Arc;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn batch(rows: &[(i64, Option<&str>)]) -> RecordBatch {
        let ids: Vec<Value> = rows.iter().map(|r| Value::Int64(r.0)).collect();
        let names: Vec<Value> = rows
            .iter()
            .map(|r| r.1.map_or(Value::Null, |s| Value::Utf8(s.to_string())))
            .collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Array::from_values(DataType::Int64, &ids).unwrap(),
                Array::from_values(DataType::Utf8, &names).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn a_table_has_a_header_a_rule_and_a_row_count() {
        let out = table(&schema(), &[batch(&[(1, Some("ada"))])]);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines[0].starts_with('┌'));
        assert!(lines[1].contains("id") && lines[1].contains("name"));
        assert!(lines[2].starts_with('├'));
        assert!(lines[3].contains("ada"));
        assert!(lines[4].starts_with('└'));
        assert_eq!(lines[5], "1 row");
    }

    #[test]
    fn columns_are_padded_to_their_widest_value() {
        let out = table(
            &schema(),
            &[batch(&[(1, Some("ada")), (2, Some("brendan"))])],
        );
        let widths: Vec<usize> = out
            .lines()
            .filter(|l| l.starts_with('│'))
            .map(|l| l.chars().count())
            .collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "{out}");
    }

    #[test]
    fn nulls_are_printed_as_null() {
        let out = table(&schema(), &[batch(&[(1, None)])]);
        assert!(out.contains("NULL"));
    }

    #[test]
    fn an_empty_result_still_shows_its_columns() {
        let out = table(&schema(), &[]);
        assert!(out.contains("id"));
        assert!(out.ends_with("0 rows\n"));
    }

    #[test]
    fn a_long_value_is_truncated_rather_than_wrapping() {
        let long = "x".repeat(200);
        let out = table(&schema(), &[batch(&[(1, Some(&long))])]);
        assert!(out.contains('…'));
        assert!(out.lines().all(|l| l.chars().count() < 80), "{out}");
    }

    #[test]
    fn multi_byte_characters_are_padded_by_character_not_by_byte() {
        let out = table(
            &schema(),
            &[batch(&[(1, Some("naïve")), (2, Some("plain"))])],
        );
        let widths: Vec<usize> = out
            .lines()
            .filter(|l| l.starts_with('│'))
            .map(|l| l.chars().count())
            .collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "{out}");
    }

    #[test]
    fn a_result_with_no_columns_reports_its_row_count() {
        let empty = Arc::new(Schema::empty());
        let out = table(&empty, &[]);
        assert!(out.contains("no columns"));
    }

    #[test]
    fn several_batches_render_as_one_table() {
        let out = table(
            &schema(),
            &[batch(&[(1, Some("a"))]), batch(&[(2, Some("b"))])],
        );
        assert!(out.contains("2 rows"));
    }
}
