//! One Tokio runtime and one Qdrant client for the process.
//!
//! shortcut: process-global client, rebuilt when url, api key, or timeout changes.
//! Move it to connection-scoped state when `SET` options land.

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use qdrant_client::Qdrant;

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
static CLIENT: Mutex<Option<CachedClient>> = Mutex::new(None);

struct CachedClient {
    url: String,
    api_key: Option<String>,
    timeout_secs: u64,
    client: Qdrant,
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
    F: std::future::Future<Output = T>,
{
    runtime().block_on(future)
}

/// Return a client for this endpoint. The api key stays in memory and is not logged.
pub fn client(url: &str, api_key: Option<&str>, timeout_secs: u64) -> Result<Qdrant, String> {
    let mut guard = CLIENT
        .lock()
        .map_err(|_| "qdrant client lock poisoned".to_string())?;
    let reuse = guard.as_ref().is_some_and(|cached| {
        cached.url == url
            && cached.api_key.as_deref() == api_key
            && cached.timeout_secs == timeout_secs
    });
    if !reuse {
        let mut builder = Qdrant::from_url(url).timeout(Duration::from_secs(timeout_secs));
        if let Some(key) = api_key {
            builder = builder.api_key(key.to_string());
        }
        let client = builder
            .build()
            .map_err(|err| format!("qdrant connect to {url} failed: {err}"))?;
        *guard = Some(CachedClient {
            url: url.to_string(),
            api_key: api_key.map(str::to_string),
            timeout_secs,
            client,
        });
    }
    let cached = guard
        .as_ref()
        .ok_or_else(|| "qdrant client missing after connect".to_string())?;
    Ok(cached.client.clone())
}
