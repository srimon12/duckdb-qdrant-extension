//! Qdrant errors as DuckDB messages. API keys are never added here.

use qdrant_client::QdrantError;
use tonic::Code;

pub fn explain(err: QdrantError) -> String {
    match err {
        QdrantError::ResponseError { status } => classify(&status),
        QdrantError::ResourceExhaustedError {
            status,
            retry_after_seconds,
        } => format!(
            "qdrant is busy, retry after {retry_after_seconds}s: {}",
            status.message()
        ),
        QdrantError::InvalidUri(err) => format!("invalid qdrant url: {err}"),
        QdrantError::ConversionError(message) => format!("qdrant conversion failed: {message}"),
        other => other.to_string(),
    }
}

pub fn is_transient(err: &QdrantError) -> bool {
    match err {
        QdrantError::ResourceExhaustedError { .. } => true,
        QdrantError::ResponseError { status } => matches!(
            status.code(),
            Code::Unavailable | Code::DeadlineExceeded | Code::Aborted | Code::Internal
        ),
        _ => false,
    }
}

fn classify(status: &tonic::Status) -> String {
    let message = status.message();
    let lower = message.to_ascii_lowercase();
    match status.code() {
        Code::NotFound => format!("qdrant collection or point was not found: {message}"),
        Code::InvalidArgument if lower.contains("dimension") => {
            format!("qdrant vector dimension mismatch: {message}")
        }
        Code::InvalidArgument => format!("qdrant rejected the request: {message}"),
        Code::Unavailable | Code::DeadlineExceeded => {
            format!("qdrant connection failed: {message}")
        }
        Code::Unauthenticated | Code::PermissionDenied => {
            format!("qdrant authentication failed: {message}")
        }
        code => format!("qdrant error {code}: {message}"),
    }
}

pub fn context(operation: &str, collection: &str, err: String) -> String {
    format!("{operation} collection={collection} failed: {err}")
}
