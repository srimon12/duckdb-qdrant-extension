//! Tokio runtime and a small cache of Qdrant clients.
//!
//! Each distinct url, api key, and timeout keeps its own client. Session
//! settings are read at bind time, so two DuckDB sessions can use two servers.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use qdrant_client::{Qdrant, QdrantError};

use crate::error::{CallError, explain, is_transient};

const CACHE_LIMIT: usize = 8;

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
static CLIENTS: Mutex<Option<HashMap<Endpoint, Qdrant>>> = Mutex::new(None);

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Endpoint {
    pub url: String,
    pub api_key: Option<String>,
    pub timeout_secs: u64,
}

fn runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("qdrant")
            .build()
            .expect("failed to start the qdrant tokio runtime")
    })
}

pub fn block_on<F, T>(future: F) -> T
where
    F: Future<Output = T>,
{
    runtime().block_on(future)
}

pub fn client(endpoint: &Endpoint) -> Result<Qdrant, CallError> {
    // A poisoned lock must not abort the DuckDB process.
    let mut guard = CLIENTS
        .lock()
        .map_err(|_| CallError::other("qdrant client lock poisoned"))?;
    let cache = guard.get_or_insert_with(HashMap::new);
    if let Some(existing) = cache.get(endpoint) {
        return Ok(existing.clone());
    }
    if cache.len() >= CACHE_LIMIT
        && let Some(stale) = cache.keys().next().cloned()
    {
        cache.remove(&stale);
    }
    let mut builder =
        Qdrant::from_url(&endpoint.url).timeout(Duration::from_secs(endpoint.timeout_secs));
    if let Some(key) = &endpoint.api_key {
        builder = builder.api_key(key.clone());
    }
    let client = builder.build().map_err(|err| {
        CallError::other(format!(
            "qdrant connect to {} failed: {}",
            endpoint.url,
            explain(err)
        ))
    })?;
    cache.insert(endpoint.clone(), client.clone());
    Ok(client)
}

/// Run a Qdrant call up to three times when the failure is a transient gRPC status.
pub fn retry<T, F, Fut>(mut operation: F) -> Result<T, CallError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, QdrantError>>,
{
    block_on(async move {
        let mut delay = Duration::from_millis(50);
        let mut attempt = 0;
        loop {
            match operation().await {
                Ok(value) => return Ok(value),
                Err(err) if is_transient(&err) && attempt < 2 => {
                    attempt += 1;
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
                Err(err) => return Err(explain(err)),
            }
        }
    })
}
