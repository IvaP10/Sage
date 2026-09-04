use thiserror::Error;

pub type CoreResult<T> = Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("authentication failed")]
    AuthenticationFailed,
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("invalid action: {0}")]
    InvalidAction(String),
    #[error("policy denied action: {0}")]
    PolicyDenied(String),
    #[error("permission is required: {0}")]
    PermissionRequired(String),
    #[error("capability rejected: {0}")]
    CapabilityRejected(String),
    #[error("approval rejected: {0}")]
    ApprovalRejected(String),
    #[error("executor unavailable: {0}")]
    ExecutorUnavailable(String),
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
    #[error("verification failed: {0}")]
    VerificationFailed(String),
    #[error("model provider failed: {0}")]
    Model(String),
    #[error("storage failed: {0}")]
    Storage(String),
    #[error("secret store failed: {0}")]
    SecretStore(String),
    #[error("I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("task not found: {0}")]
    TaskNotFound(String),
    #[error("task was cancelled")]
    Cancelled,
    #[error("operation timed out: {0}")]
    Timeout(String),
}
