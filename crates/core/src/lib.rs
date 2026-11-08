//! Query Qdrant from DuckDB.
//!
//! Retrieval stays in Qdrant. DuckDB receives id, score, and payload JSON.

mod args;
mod client;
mod collections;
mod discover;
mod emit;
mod error;
mod hybrid;
mod query;
mod recommend;
mod result;
mod run;
mod scroll;
mod settings;
mod types;
mod upsert;

use duckdb_neo::{Result, extension::Extension};

pub const CATEGORY: &str = "qdrant";

pub fn register(extension: &Extension) -> Result<()> {
    settings::register(extension)?;
    query::register(extension)?;
    hybrid::register(extension)?;
    recommend::register(extension)?;
    discover::register(extension)?;
    scroll::register(extension)?;
    collections::register(extension)?;
    upsert::register(extension)?;
    Ok(())
}

#[cfg(test)]
mod live;
