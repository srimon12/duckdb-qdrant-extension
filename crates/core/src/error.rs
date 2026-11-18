//! Failures from Qdrant, classified so SQL sees a stable message.
//!
//! The API key is never copied into a message. [`CallError`] is the library
//! error. DuckDB still receives [`Display`] text.

use std::fmt;

use qdrant_client::QdrantError;
use tonic::Code;

/// A failure a SQL caller can read. Variants exist so tests and logs can tell
/// a missing collection from a bad vector without parsing prose.
#[derive(Debug)]
pub struct CallError {
    kind: Kind,
    message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Connect,
    NotFound,
    Dimension,
    Rejected,
    Auth,
    Busy,
    Invalid,
    Other,
}

impl CallError {
    /// Build an error that is not a Qdrant status.
    pub fn other(message: impl Into<String>) -> Self {
        Self {
            kind: Kind::Other,
            message: message.into(),
        }
    }

    /// Build an error for a bad argument found before the request is sent.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: Kind::Invalid,
            message: message.into(),
        }
    }

    fn new(kind: Kind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for CallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            Kind::Connect
            | Kind::NotFound
            | Kind::Dimension
            | Kind::Rejected
            | Kind::Auth
            | Kind::Busy
            | Kind::Invalid
            | Kind::Other => formatter.write_str(&self.message),
        }
    }
}

impl std::error::Error for CallError {}

impl From<CallError> for String {
    fn from(err: CallError) -> Self {
        err.message
    }
}

/// Classify a client error. The returned text is the SQL message.
pub fn explain(err: QdrantError) -> CallError {
    match err {
        QdrantError::ResponseError { status } => classify(&status),
        QdrantError::ResourceExhaustedError {
            status,
            retry_after_seconds,
        } => CallError::new(
            Kind::Busy,
            format!(
                "qdrant is busy, retry after {retry_after_seconds}s: {}",
                status.message()
            ),
        ),
        QdrantError::InvalidUri(err) => {
            CallError::new(Kind::Invalid, format!("invalid qdrant url: {err}"))
        }
        QdrantError::ConversionError(message) => CallError::new(
            Kind::Invalid,
            format!("qdrant conversion failed: {message}"),
        ),
        other => CallError::other(other.to_string()),
    }
}

/// True when another attempt may succeed.
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

fn classify(status: &tonic::Status) -> CallError {
    let message = status.message();
    let lower = message.to_ascii_lowercase();
    match status.code() {
        Code::NotFound => CallError::new(
            Kind::NotFound,
            format!("qdrant collection or point was not found: {message}"),
        ),
        Code::InvalidArgument if lower.contains("dimension") => CallError::new(
            Kind::Dimension,
            format!("qdrant vector dimension mismatch: {message}"),
        ),
        Code::InvalidArgument => CallError::new(
            Kind::Rejected,
            format!("qdrant rejected the request: {message}"),
        ),
        Code::Unavailable | Code::DeadlineExceeded => CallError::new(
            Kind::Connect,
            format!("qdrant connection failed: {message}"),
        ),
        Code::Unauthenticated | Code::PermissionDenied => CallError::new(
            Kind::Auth,
            format!("qdrant authentication failed: {message}"),
        ),
        code => CallError::other(format!("qdrant error {code}: {message}")),
    }
}

/// Prefix an operation and collection onto an error. The key is not included.
pub fn context(operation: &str, collection: &str, err: impl fmt::Display) -> String {
    format!("{operation} collection={collection} failed: {err}")
}
