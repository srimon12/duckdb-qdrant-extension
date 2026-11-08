//! Execute a universal query, paging with `offset` so a large limit
//! reaches DuckDB in chunks instead of one unbounded response.

use qdrant_client::qdrant::QueryPointsBuilder;

use crate::client::{client, retry};
use crate::error::context;
use crate::result::{QueryRow, scored_rows};
use crate::types::{PAGE_SIZE, Scope};

pub fn run_pages(
    scope: &Scope,
    operation: &str,
    mut page: impl FnMut(u64, u64) -> QueryPointsBuilder,
) -> Result<Vec<QueryRow>, String> {
    let qdrant = client(&scope.endpoint)?;
    let mut rows = Vec::new();
    let mut offset = 0u64;
    while (rows.len() as u64) < scope.limit {
        let size = (scope.limit - rows.len() as u64).min(PAGE_SIZE);
        let request = page(offset, size).build();
        let response = retry(|| {
            let qdrant = qdrant.clone();
            let request = request.clone();
            async move { qdrant.query(request).await }
        })
        .map_err(|err| context(operation, &scope.collection, err))?;
        let count = response.result.len() as u64;
        rows.extend(
            scored_rows(response.result)
                .map_err(|err| context(operation, &scope.collection, err))?,
        );
        if count == 0 || count < size {
            break;
        }
        offset += count;
    }
    Ok(rows)
}
