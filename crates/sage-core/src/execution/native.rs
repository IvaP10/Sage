use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{Duration, Utc};
use serde_json::json;
use tokio::fs;
use uuid::Uuid;

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
    pub fn new(recovery_root: PathBuf, platform: Arc<dyn PlatformController>) -> Self {
        Self {
            recovery_root,
            platform,
        }
    }

    async fn execute_file(
        &self,
        compiled: &CompiledAction,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        let proposal = &compiled.proposal;
        match &proposal.action {
            Action::ReadFile { path, max_bytes } => {
                require_exact_file(capability, path)?;
                let metadata = fs::metadata(path).await?;
                if !metadata.is_file() {
                    return Err(CoreError::ExecutionFailed(
                        "resource is not a regular file".into(),
                    ));
                }
                let limit = (*max_bytes).min(16 * 1024 * 1024) as usize;
                if metadata.len() > limit as u64 {
                    return Err(CoreError::ExecutionFailed(format!(
                        "file exceeds the authorized read limit of {limit} bytes"
                    )));
                }
                let bytes = fs::read(path).await?;
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
                let existed = fs::try_exists(path).await?;
                if existed && !overwrite {
                    return Err(CoreError::ExecutionFailed(
                        "destination already exists and overwrite was not authorized".into(),
                    ));
                }
                let task_recovery = self.task_recovery(proposal.task_id).await?;
                let rollback = if existed {
                    let backup = task_recovery.join(format!("{}-backup", proposal.id));
                    fs::copy(path, &backup).await?;
                    Some(RollbackPlan {
                        action_id: proposal.id,
                        operations: vec![RollbackOperation::RestoreFile {
                            backup: backup.to_string_lossy().into_owned(),
                            destination: path.to_string_lossy().into_owned(),
                        }],
                        expires_at: Utc::now() + Duration::hours(24),
                    })
                } else {
                    Some(RollbackPlan {
                        action_id: proposal.id,
                        operations: vec![RollbackOperation::MoveFile {
                            source: path.to_string_lossy().into_owned(),
                            destination: task_recovery
                                .join(format!("{}-created", proposal.id))
                                .to_string_lossy()
                                .into_owned(),
                        }],
                        expires_at: Utc::now() + Duration::hours(24),
                    })
                };
                let temporary = sibling_temporary_path(path, proposal.id)?;
                fs::write(&temporary, content.as_bytes()).await?;
                if existed {
                    fs::remove_file(path).await?;
                }
                fs::rename(&temporary, path).await?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: format!("wrote {} bytes to {}", content.len(), path.display()),
                    transient_data: json!({ "bytes_written": content.len() }),
                    rollback,
                })
            }
            Action::MoveFile {
                source,
                destination,
            } => {
                require_file_pair(capability, source, destination)?;
                if fs::try_exists(destination).await? {
                    return Err(CoreError::ExecutionFailed(
                        "move destination already exists".into(),
                    ));
                }
                fs::rename(source, destination).await?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: format!("moved {} to {}", source.display(), destination.display()),
                    transient_data: json!({}),
                    rollback: Some(RollbackPlan {
                        action_id: proposal.id,
                        operations: vec![RollbackOperation::MoveFile {
                            source: destination.to_string_lossy().into_owned(),
                            destination: source.to_string_lossy().into_owned(),
                        }],
                        expires_at: Utc::now() + Duration::hours(24),
                    }),
                })
            }
            Action::DeleteFile { path } => {
                require_exact_file(capability, path)?;
                let task_recovery = self.task_recovery(proposal.task_id).await?;
                let quarantined = task_recovery.join(format!("{}-deleted", proposal.id));
                fs::rename(path, &quarantined).await?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: format!("moved {} into SAGE recovery storage", path.display()),
                    transient_data: json!({}),
                    rollback: Some(RollbackPlan {
                        action_id: proposal.id,
                        operations: vec![RollbackOperation::MoveFile {
                            source: quarantined.to_string_lossy().into_owned(),
                            destination: path.to_string_lossy().into_owned(),
                        }],
                        expires_at: Utc::now() + Duration::hours(24),
                    }),
                })
            }
            Action::CreateFolder { path } => {
                require_exact_file(capability, path)?;
                fs::create_dir(path).await?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: format!("created folder {}", path.display()),
                    transient_data: json!({}),
                    rollback: Some(RollbackPlan {
                        action_id: proposal.id,
                        operations: vec![RollbackOperation::RemoveEmptyFolder {
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

    async fn task_recovery(&self, task_id: Uuid) -> CoreResult<PathBuf> {
        let path = self.recovery_root.join(task_id.to_string());
        fs::create_dir_all(&path).await?;
        Ok(path)
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

fn require_file_pair(
    capability: &CapabilityGrant,
    source: &Path,
    destination: &Path,
) -> CoreResult<()> {
    match &capability.resource {
        CapabilityResource::FilePair {
            source: allowed_source,
            destination: allowed_destination,
        } if Path::new(allowed_source) == source
            && Path::new(allowed_destination) == destination =>
        {
            Ok(())
        }
        _ => Err(CoreError::CapabilityRejected(
            "filesystem capability does not match the exact source and destination".into(),
        )),
    }
}

fn sibling_temporary_path(path: &Path, action_id: Uuid) -> CoreResult<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| CoreError::ExecutionFailed("destination has no parent".into()))?;
    Ok(parent.join(format!(".sage-{action_id}.tmp")))
}
