use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{Duration, Utc};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::capability::{CapabilityGrant, CapabilityResource};
use crate::compiler::{CompiledAction, ImplementationCandidate};
use crate::domain::{Action, ExecutionDomain};
use crate::error::{CoreError, CoreResult};

use super::{ExecutionReceipt, Executor, RollbackOperation, RollbackPlan};

#[async_trait]
pub trait PlatformController: Send + Sync {
    async fn execute_native(
        &self,
        action: &Action,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt>;
}

#[derive(Debug, Default)]
pub struct UnsupportedPlatformController;

#[async_trait]
impl PlatformController for UnsupportedPlatformController {
    async fn execute_native(
        &self,
        action: &Action,
        _capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        Err(CoreError::ExecutorUnavailable(format!(
            "the platform adapter does not implement {}",
            action.kind()
        )))
    }
}

pub struct NativeExecutor {
    recovery_root: PathBuf,
    platform: Arc<dyn PlatformController>,
    files: Arc<super::files::FileBroker>,
    store: crate::storage::LocalStore,
}

impl std::fmt::Debug for NativeExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeExecutor")
            .field("recovery_root", &self.recovery_root)
            .finish_non_exhaustive()
    }
}

impl NativeExecutor {
    pub fn new(
        recovery_root: PathBuf,
        platform: Arc<dyn PlatformController>,
        files: Arc<super::files::FileBroker>,
        store: crate::storage::LocalStore,
    ) -> Self {
        Self {
            recovery_root,
            platform,
            files,
            store,
        }
    }

    async fn execute_file(
        &self,
        compiled: &CompiledAction,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        let proposal = &compiled.proposal;
        match &proposal.action {
            Action::FetchPublic { url, max_bytes } => {
                if !matches!(&capability.resource,CapabilityResource::NetworkRoute{url:approved} if approved==url)
                {
                    return Err(CoreError::CapabilityRejected(
                        "Public fetch grant targets another URL".into(),
                    ));
                }
                let document = crate::network::fetch_public(url, *max_bytes).await?;
                Ok(ExecutionReceipt {
                    executor: "network-broker".into(),
                    summary: format!("Read {} bytes from {}", document.text.len(), document.url),
                    transient_data: serde_json::to_value(document)?,
                    rollback: None,
                })
            }
            Action::ReadFile { path, max_bytes } => {
                require_exact_file(capability, path)?;
                let bytes = self.files.take(proposal, capability)?.read(*max_bytes)?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: format!("read {} bytes from {}", bytes.len(), path.display()),
                    transient_data: json!({ "bytes_base64": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes) }),
                    rollback: None,
                })
            }
            Action::WriteFile {
                path,
                content,
                overwrite,
            } => {
                require_exact_file(capability, path)?;
                let prepared = self.files.take(proposal, capability)?;
                let expected_sha256 = format!("{:x}", Sha256::digest(content.as_bytes()));
                let operation = if prepared.exists() {
                    if !overwrite {
                        return Err(CoreError::ExecutionFailed(
                            "Overwrite was not authorized".into(),
                        ));
                    }
                    let backup = prepared.read(16 * 1024 * 1024)?;
                    let artifact_id = self.store.save_artifact(proposal.task_id, &backup)?;
                    RollbackOperation::RestoreArtifact {
                        artifact_id,
                        destination: path.to_string_lossy().into_owned(),
                        expected_sha256,
                    }
                } else {
                    RollbackOperation::RemoveCreatedFile {
                        path: path.to_string_lossy().into_owned(),
                        expected_sha256,
                    }
                };
                let rollback = Some(RollbackPlan {
                    action_id: proposal.id,
                    operations: vec![operation],
                    expires_at: Utc::now() + Duration::hours(24),
                });
                self.store
                    .save_rollback(proposal.task_id, rollback.as_ref().unwrap())?;
                prepared.write(content.as_bytes(), *overwrite)?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: format!("wrote {} bytes to {}", content.len(), path.display()),
                    transient_data: json!({ "bytes_written": content.len() }),
                    rollback,
                })
            }
            Action::CreateFolder { path } => {
                require_exact_file(capability, path)?;
                let prepared = self.files.take(proposal, capability)?;
                prepared.create_folder()?;
                let identity = prepared.current_identity()?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: format!("created folder {}", path.display()),
                    transient_data: json!({}),
                    rollback: Some(RollbackPlan {
                        action_id: proposal.id,
                        operations: vec![RollbackOperation::RemoveCreatedFolder {
                            identity,
                            path: path.to_string_lossy().into_owned(),
                        }],
                        expires_at: Utc::now() + Duration::hours(24),
                    }),
                })
            }
            _ => {
                self.platform
                    .execute_native(&proposal.action, capability)
                    .await
            }
        }
    }
}

#[async_trait]
impl Executor for NativeExecutor {
    fn name(&self) -> &'static str {
        "native-os-executor"
    }

    fn domain(&self) -> ExecutionDomain {
        ExecutionDomain::Native
    }

    async fn execute(
        &self,
        action: &CompiledAction,
        _implementation: &ImplementationCandidate,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        self.execute_file(action, capability).await
    }
}

fn require_exact_file(capability: &CapabilityGrant, path: &Path) -> CoreResult<()> {
    match &capability.resource {
        CapabilityResource::File { canonical_path } if Path::new(canonical_path) == path => Ok(()),
        _ => Err(CoreError::CapabilityRejected(
            "filesystem capability does not match the exact action path".into(),
        )),
    }
}
