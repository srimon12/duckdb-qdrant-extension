//! Query Qdrant from DuckDB.
//!
//! Phase 0 registers `qdrant_query`: nearest neighbours, a JSON filter, and a limit.

mod client;
mod query;
mod result;
mod types;

use duckdb_neo::{Result, extension::Extension};

pub const CATEGORY: &str = "qdrant";

pub fn register(extension: &Extension) -> Result<()> {
    query::register(extension)
}
