//! Query Qdrant from DuckDB.
//!
//! Phase 2 is still retrieval: recommend, discover, context, score threshold,
//! selected payload keys, and paged limits. Phase 3 is not search: collection
//! catalog, scroll, and a thin upsert. Qdrant 1.20 adds BM25 text search; 1.20.2
//! adds one batch RPC for many queries and server-side dense + text fusion.
//! Search rows are id, score, and payload JSON; batch rows put query_index first.

mod args;
mod batch;
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
mod text;
mod types;
mod upsert;

use duckdb_neo::{Result, extension::Extension};

/// Category stored on every function in `duckdb_functions()`.
pub const CATEGORY: &str = "qdrant";

/// Register settings and the phase 1, phase 2, and phase 3 SQL functions.
///
/// # Errors
///
/// Returns an error when DuckDB rejects a function or setting name.
pub fn register(extension: &Extension) -> Result<()> {
    settings::register(extension)?;
    query::register(extension)?;
    hybrid::register(extension)?;
    text::register(extension)?;
    batch::register(extension)?;
    recommend::register(extension)?;
    discover::register(extension)?;
    scroll::register(extension)?;
    collections::register(extension)?;
    upsert::register(extension)?;
    Ok(())
}

#[cfg(test)]
mod live;
