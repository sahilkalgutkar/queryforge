use crate::error::{Error, Result};
use std::fmt;

/// The four physical types the engine stores and computes over.
///
/// I deliberately kept this small. Every extra type multiplies out across the
/// array representations, the encodings, the comparison kernels and the
/// aggregate accumulators, and none of that would teach anything the existing
/// four don't already.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Boolean,
    Int64,
    Float64,
    Utf8,
}

impl DataType {
    pub fn is_numeric(&self) -> bool {
        matches!(self, DataType::Int64 | DataType::Float64)
    }

    /// The type two operands are both widened to before a binary operation.
    ///
    /// Int64 widens to Float64 rather than the reverse so that `1 + 0.5` is
    /// 1.5 and not 1.
    pub fn unify(a: DataType, b: DataType) -> Result<DataType> {
        if a == b {
            return Ok(a);
        }
        match (a, b) {
            (DataType::Int64, DataType::Float64) | (DataType::Float64, DataType::Int64) => {
                Ok(DataType::Float64)
            }
            _ => Err(Error::typ(format!("cannot reconcile types {a} and {b}"))),
        }
    }

    /// Parses the type names accepted in `CREATE TABLE`.
    pub fn from_sql_name(name: &str) -> Result<DataType> {
        match name.to_ascii_uppercase().as_str() {
            "BOOLEAN" | "BOOL" => Ok(DataType::Boolean),
            "INT" | "INTEGER" | "BIGINT" => Ok(DataType::Int64),
            "FLOAT" | "DOUBLE" | "REAL" => Ok(DataType::Float64),
            "TEXT" | "VARCHAR" | "STRING" => Ok(DataType::Utf8),
            other => Err(Error::parse(format!("unknown type `{other}`"))),
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DataType::Boolean => "BOOLEAN",
            DataType::Int64 => "INT64",
            DataType::Float64 => "FLOAT64",
            DataType::Utf8 => "UTF8",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

impl Field {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Field {
            name: name.into(),
            data_type,
            nullable,
        }
    }
}

/// An ordered list of fields. Column order is part of the contract: the
/// storage format, the vectorised batches and the physical plan all address
/// columns positionally, and names are resolved to positions exactly once, in
/// the binder.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        Schema { fields }
    }

    pub fn empty() -> Self {
        Schema { fields: Vec::new() }
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn field(&self, index: usize) -> Result<&Field> {
        self.fields
            .get(index)
            .ok_or_else(|| Error::internal(format!("column index {index} out of range")))
    }

    /// Resolves a column name to its position.
    ///
    /// Matching is case-insensitive, which is what SQL users expect, but an
    /// ambiguous match is an error rather than a silent pick of the first hit —
    /// that is the kind of thing that produces a query returning the wrong
    /// column with no complaint.
    pub fn index_of(&self, name: &str) -> Result<usize> {
        let mut found = None;
        for (i, f) in self.fields.iter().enumerate() {
            if f.name.eq_ignore_ascii_case(name) {
                if found.is_some() {
                    return Err(Error::plan(format!("column `{name}` is ambiguous")));
                }
                found = Some(i);
            }
        }
        found.ok_or_else(|| Error::plan(format!("no such column `{name}`")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.fields
            .iter()
            .any(|f| f.name.eq_ignore_ascii_case(name))
    }

    /// Concatenates two schemas, as a join does to its inputs.
    pub fn join(left: &Schema, right: &Schema) -> Schema {
        let mut fields = left.fields.clone();
        fields.extend(right.fields.iter().cloned());
        Schema::new(fields)
    }

    /// Projects a subset of columns, in the order given.
    pub fn project(&self, indices: &[usize]) -> Result<Schema> {
        let mut fields = Vec::with_capacity(indices.len());
        for &i in indices {
            fields.push(self.field(i)?.clone());
        }
        Ok(Schema::new(fields))
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cols: Vec<String> = self
            .fields
            .iter()
            .map(|c| format!("{} {}", c.name, c.data_type))
            .collect();
        write!(f, "({})", cols.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ])
    }

    #[test]
    fn column_lookup_is_case_insensitive() {
        assert_eq!(schema().index_of("ID").unwrap(), 0);
        assert_eq!(schema().index_of("Name").unwrap(), 1);
    }

    #[test]
    fn duplicate_column_names_are_rejected_rather_than_silently_resolved() {
        let s = Schema::new(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("X", DataType::Utf8, false),
        ]);
        let err = s.index_of("x").unwrap_err();
        assert!(matches!(err, Error::Plan(_)));
        assert!(err.to_string().contains("ambiguous"));
    }

    #[test]
    fn missing_column_is_a_planning_error() {
        assert!(matches!(schema().index_of("nope"), Err(Error::Plan(_))));
    }

    #[test]
    fn int_and_float_unify_to_float() {
        assert_eq!(
            DataType::unify(DataType::Int64, DataType::Float64).unwrap(),
            DataType::Float64
        );
        assert_eq!(
            DataType::unify(DataType::Float64, DataType::Int64).unwrap(),
            DataType::Float64
        );
    }

    #[test]
    fn text_does_not_unify_with_a_number() {
        assert!(DataType::unify(DataType::Utf8, DataType::Int64).is_err());
        assert!(DataType::unify(DataType::Boolean, DataType::Int64).is_err());
        assert_eq!(
            DataType::unify(DataType::Utf8, DataType::Utf8).unwrap(),
            DataType::Utf8
        );
    }

    #[test]
    fn sql_type_names_map_onto_the_four_physical_types() {
        assert_eq!(DataType::from_sql_name("bigint").unwrap(), DataType::Int64);
        assert_eq!(DataType::from_sql_name("VARCHAR").unwrap(), DataType::Utf8);
        assert_eq!(DataType::from_sql_name("Bool").unwrap(), DataType::Boolean);
        assert_eq!(
            DataType::from_sql_name("double").unwrap(),
            DataType::Float64
        );
        assert!(DataType::from_sql_name("blob").is_err());
    }

    #[test]
    fn join_concatenates_and_project_reorders() {
        let joined = Schema::join(&schema(), &schema());
        assert_eq!(joined.len(), 4);
        let projected = joined.project(&[1, 0]).unwrap();
        assert_eq!(projected.field(0).unwrap().name, "name");
        assert_eq!(projected.field(1).unwrap().name, "id");
        assert!(joined.project(&[9]).is_err());
    }

    #[test]
    fn display_renders_name_and_type_pairs() {
        assert_eq!(schema().to_string(), "(id INT64, name UTF8)");
        assert!(Schema::empty().is_empty());
        assert!(schema().contains("ID"));
        assert!(!schema().contains("missing"));
    }

    #[test]
    fn numeric_predicate_covers_only_the_number_types() {
        assert!(DataType::Int64.is_numeric());
        assert!(DataType::Float64.is_numeric());
        assert!(!DataType::Utf8.is_numeric());
        assert!(!DataType::Boolean.is_numeric());
    }
}
