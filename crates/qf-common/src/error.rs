use std::fmt;

/// Every failure in the engine, tagged with the layer that produced it.
///
/// I keep the layers separate rather than collapsing to one string so the CLI
/// can tell a user's typo (`Parse`) apart from a query that parses but cannot
/// be resolved against the catalog (`Plan`) apart from an engine bug (`Internal`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The SQL text is not well-formed.
    Parse(String),
    /// The query is well-formed but cannot be bound or planned.
    Plan(String),
    /// A type rule was violated (comparing a string to an int, summing a bool).
    Type(String),
    /// Something went wrong while executing a physical operator.
    Exec(String),
    /// Storage-layer failure: bad file, short read, corrupt footer.
    Storage(String),
    /// An invariant the engine itself is supposed to uphold was broken.
    Internal(String),
}

impl Error {
    pub fn parse(msg: impl Into<String>) -> Self {
        Error::Parse(msg.into())
    }
    pub fn plan(msg: impl Into<String>) -> Self {
        Error::Plan(msg.into())
    }
    pub fn typ(msg: impl Into<String>) -> Self {
        Error::Type(msg.into())
    }
    pub fn exec(msg: impl Into<String>) -> Self {
        Error::Exec(msg.into())
    }
    pub fn storage(msg: impl Into<String>) -> Self {
        Error::Storage(msg.into())
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        Error::Internal(msg.into())
    }

    /// Short tag used when rendering the error to a terminal.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Parse(_) => "parse error",
            Error::Plan(_) => "planning error",
            Error::Type(_) => "type error",
            Error::Exec(_) => "execution error",
            Error::Storage(_) => "storage error",
            Error::Internal(_) => "internal error",
        }
    }

    fn message(&self) -> &str {
        match self {
            Error::Parse(m)
            | Error::Plan(m)
            | Error::Type(m)
            | Error::Exec(m)
            | Error::Storage(m)
            | Error::Internal(m) => m,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind(), self.message())
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Storage(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_layer_and_message() {
        let e = Error::parse("unexpected token `FROM`");
        assert_eq!(e.to_string(), "parse error: unexpected token `FROM`");
    }

    #[test]
    fn every_variant_reports_a_distinct_kind() {
        let kinds = [
            Error::parse("a").kind(),
            Error::plan("a").kind(),
            Error::typ("a").kind(),
            Error::exec("a").kind(),
            Error::storage("a").kind(),
            Error::internal("a").kind(),
        ];
        let unique: std::collections::HashSet<_> = kinds.iter().collect();
        assert_eq!(unique.len(), kinds.len());
    }

    #[test]
    fn io_errors_become_storage_errors() {
        let io = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read");
        let e: Error = io.into();
        assert!(matches!(e, Error::Storage(_)));
        assert!(e.to_string().contains("short read"));
    }

    #[test]
    fn message_is_preserved_across_variants() {
        for e in [
            Error::plan("m"),
            Error::typ("m"),
            Error::exec("m"),
            Error::internal("m"),
        ] {
            assert!(e.to_string().ends_with(": m"));
        }
    }
}
