//! Types shared by every layer of the engine: the type system, values, schemas
//! and the error type that flows from the parser all the way out to the CLI.

pub mod error;
pub mod schema;
pub mod value;

pub use error::{Error, Result};
pub use schema::{DataType, Field, Schema};
pub use value::Value;
