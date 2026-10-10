//! Sage's Core-independent Qwen3.5 text and vision decoder implementation.
//!
//! This crate owns model geometry and numerical execution only. It does not
//! select policy, hold execution authority, or expose host operating-system
//! actions.

#![forbid(unsafe_code)]

pub mod loader;
pub mod qwen35;
pub mod qwen35_vision;
pub mod resource;

/// Errors raised while validating or executing the bounded Qwen candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen35Error {
    /// A model tensor, dimension, numeric result, or backend violated a bound.
    Model(String),
    /// The host cannot admit another model or compute reservation.
    ResourceUnavailable(String),
    /// The caller cancelled an active model operation.
    Cancelled,
}

impl std::fmt::Display for Qwen35Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(message) => formatter.write_str(message),
            Self::ResourceUnavailable(message) => formatter.write_str(message),
            Self::Cancelled => formatter.write_str("model operation was cancelled"),
        }
    }
}

impl std::error::Error for Qwen35Error {}

/// Result type for Qwen model validation and numerical execution.
pub type Qwen35Result<T> = Result<T, Qwen35Error>;

impl From<sage_inference_math::InferenceError> for Qwen35Error {
    fn from(error: sage_inference_math::InferenceError) -> Self {
        Self::Model(error.to_string())
    }
}

impl From<sage_model_package::PackageError> for Qwen35Error {
    fn from(error: sage_model_package::PackageError) -> Self {
        Self::Model(error.to_string())
    }
}

impl From<serde_json::Error> for Qwen35Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Model(error.to_string())
    }
}

impl From<sage_qwen_tokenizer::TokenizerError> for Qwen35Error {
    fn from(error: sage_qwen_tokenizer::TokenizerError) -> Self {
        Self::Model(error.to_string())
    }
}

impl From<std::io::Error> for Qwen35Error {
    fn from(error: std::io::Error) -> Self {
        Self::Model(error.to_string())
    }
}
