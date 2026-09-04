use std::path::PathBuf;
use std::process::Stdio;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use crate::capability::CapabilityGrant;
use crate::compiler::{CompiledAction, ImplementationCandidate};
use crate::domain::ExecutionDomain;
use crate::error::{CoreError, CoreResult};

use super::{ExecutionReceipt, Executor};

const MAX_WORKER_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub executable: PathBuf,
    pub domain: ExecutionDomain,
    pub timeout: Duration,
}

#[derive(Debug)]
pub struct FramedWorkerExecutor {
    config: WorkerConfig,
}

#[derive(Debug, Serialize)]
struct WorkerRequest<'a> {
    action: &'a CompiledAction,
    implementation: &'a ImplementationCandidate,
    capability: &'a CapabilityGrant,
}

#[derive(Debug, Deserialize)]
struct WorkerResponse {
    ok: bool,
    receipt: Option<ExecutionReceipt>,
    error: Option<String>,
}

impl FramedWorkerExecutor {
    pub fn new(config: WorkerConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Executor for FramedWorkerExecutor {
    fn name(&self) -> &'static str {
        match self.config.domain {
            ExecutionDomain::Browser => "browser-worker",
            ExecutionDomain::Sandbox => "sandbox-worker",
            ExecutionDomain::Privileged => "privileged-helper",
            _ => "isolated-worker",
        }
    }

    fn domain(&self) -> ExecutionDomain {
        self.config.domain
    }

    async fn execute(
        &self,
        action: &CompiledAction,
        implementation: &ImplementationCandidate,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        let payload = serde_json::to_vec(&WorkerRequest {
            action,
            implementation,
            capability,
        })?;
        let mut child = Command::new(&self.config.executable)
            .arg("--single-request")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .env_clear()
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| CoreError::ExecutorUnavailable(error.to_string()))?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| CoreError::ExecutionFailed("worker stdin unavailable".into()))?;
        stdin.write_u32(payload.len() as u32).await?;
        stdin.write_all(&payload).await?;
        stdin.shutdown().await?;

        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| CoreError::ExecutionFailed("worker stdout unavailable".into()))?;
        let operation = async {
            let length = stdout.read_u32().await? as usize;
            if length > MAX_WORKER_RESPONSE_BYTES {
                return Err(CoreError::Protocol(
                    "worker response exceeds size limit".into(),
                ));
            }
            let mut response = vec![0_u8; length];
            stdout.read_exact(&mut response).await?;
            let status = child.wait().await?;
            if !status.success() {
                return Err(CoreError::ExecutionFailed(format!(
                    "worker exited with {status}"
                )));
            }
            let response: WorkerResponse = serde_json::from_slice(&response)?;
            if response.ok {
                response.receipt.ok_or_else(|| {
                    CoreError::Protocol("worker omitted the execution receipt".into())
                })
            } else {
                Err(CoreError::ExecutionFailed(
                    response
                        .error
                        .unwrap_or_else(|| "worker rejected action".into()),
                ))
            }
        };
        timeout(self.config.timeout, operation)
            .await
            .map_err(|_| CoreError::Timeout(format!("{} timed out", self.name())))?
    }
}
