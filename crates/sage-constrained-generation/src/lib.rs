//! Core-independent constrained token generation for Sage's local models.
//!
//! This crate owns JSON and caller-supplied schema prefix constraints around
//! Sage's Qwen decoder. It has no broker, task store, credentials, network,
//! native execution, or authority to approve model output.

#![forbid(unsafe_code)]

pub mod generation;
pub mod preview;
pub mod schema;

pub const MAX_STRUCTURED_JSON_BYTES: usize = 256 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("invalid constrained generation: {0}")]
    Model(String),
    #[error("generation was cancelled")]
    Cancelled,
}

pub type DecodeResult<T> = Result<T, DecodeError>;

impl From<sage_qwen35_runtime::Qwen35Error> for DecodeError {
    fn from(error: sage_qwen35_runtime::Qwen35Error) -> Self {
        match error {
            sage_qwen35_runtime::Qwen35Error::Cancelled => Self::Cancelled,
            other => Self::Model(other.to_string()),
        }
    }
}

impl From<serde_json::Error> for DecodeError {
    fn from(error: serde_json::Error) -> Self {
        Self::Model(error.to_string())
    }
}
