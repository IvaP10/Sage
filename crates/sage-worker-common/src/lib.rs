#![forbid(unsafe_code)]

use sage_core::capability::CapabilityGrant;
use sage_core::compiler::{CompiledAction, ImplementationCandidate};
use sage_core::execution::ExecutionReceipt;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub struct WorkerRequest {
    pub action: CompiledAction,
    pub implementation: ImplementationCandidate,
    pub capability: CapabilityGrant,
}

#[derive(Debug, Serialize)]
pub struct WorkerResponse {
    pub ok: bool,
    pub receipt: Option<ExecutionReceipt>,
    pub error: Option<String>,
}

impl WorkerResponse {
    pub fn success(receipt: ExecutionReceipt) -> Self {
        Self {
            ok: true,
            receipt: Some(receipt),
            error: None,
        }
    }

    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            ok: false,
            receipt: None,
            error: Some(error.into()),
        }
    }
}

pub async fn read_request() -> Result<WorkerRequest, String> {
    let mut stdin = tokio::io::stdin();
    let length = stdin.read_u32().await.map_err(|error| error.to_string())? as usize;
    if length == 0 || length > MAX_REQUEST_BYTES {
        return Err("worker request is outside the accepted size range".into());
    }
    let mut bytes = vec![0_u8; length];
    stdin
        .read_exact(&mut bytes)
        .await
        .map_err(|error| error.to_string())?;
    serde_json::from_slice(&bytes).map_err(|error| error.to_string())
}

pub async fn write_response(response: &WorkerResponse) -> Result<(), String> {
    let bytes = serde_json::to_vec(response).map_err(|error| error.to_string())?;
    if bytes.is_empty() || bytes.len() > MAX_RESPONSE_BYTES {
        return Err("worker response is outside the accepted size range".into());
    }
    let mut stdout = tokio::io::stdout();
    stdout
        .write_u32(bytes.len() as u32)
        .await
        .map_err(|error| error.to_string())?;
    stdout
        .write_all(&bytes)
        .await
        .map_err(|error| error.to_string())?;
    stdout.flush().await.map_err(|error| error.to_string())
}
