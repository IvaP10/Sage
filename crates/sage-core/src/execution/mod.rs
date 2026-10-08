pub mod bridge;
pub mod directory;
pub mod files;
pub(crate) mod io;
mod native;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::capability::{CapabilityBroker, CapabilityGrant};
use crate::compiler::{CompiledAction, ImplementationCandidate};
use crate::domain::ExecutionDomain;
use crate::error::{CoreError, CoreResult};

use crate::procedure_stream::ProcedureNodeStreams;
pub use native::{NativeExecutor, PlatformController, UnsupportedPlatformController};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RollbackOperation {
    RestoreArtifact {
        artifact_id: Uuid,
        destination: String,
        expected_sha256: String,
    },
    RemoveCreatedFile {
        path: String,
        expected_sha256: String,
    },
    RemoveCreatedFolder {
        path: String,
        identity: String,
    },
    MoveFile {
        source: String,
        destination: String,
    },
    RestoreFile {
        backup: String,
        destination: String,
    },
    RemoveEmptyFolder {
        path: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPlan {
    pub action_id: Uuid,
    pub operations: Vec<RollbackOperation>,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub executor: String,
    pub summary: String,
    pub transient_data: serde_json::Value,
    pub rollback: Option<RollbackPlan>,
}

#[async_trait]
pub trait Executor: Send + Sync {
    fn name(&self) -> &'static str;
    fn domain(&self) -> ExecutionDomain;
    fn supports_procedure_streams(&self) -> bool {
        false
    }
    async fn execute(
        &self,
        action: &CompiledAction,
        implementation: &ImplementationCandidate,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt>;

    /// Execute one broker-authorized action with its declared procedure data
    /// endpoints. The broker retains the bundle and checks every stream's
    /// terminal frame before accepting the executor receipt. Existing
    /// executors remain compatible for actions without streams; consuming
    /// stream data requires an explicit implementation.
    async fn execute_with_streams(
        &self,
        action: &CompiledAction,
        implementation: &ImplementationCandidate,
        capability: &CapabilityGrant,
        streams: Option<&mut ProcedureNodeStreams>,
    ) -> CoreResult<ExecutionReceipt> {
        if streams.as_ref().is_some_and(|streams| !streams.is_empty()) {
            return Err(CoreError::ExecutorUnavailable(format!(
                "{} does not implement procedure stream execution",
                self.name()
            )));
        }
        self.execute(action, implementation, capability).await
    }
}

pub struct ExecutionBroker {
    capabilities: CapabilityBroker,
    executors: HashMap<ExecutionDomain, Arc<dyn Executor>>,
}

impl std::fmt::Debug for ExecutionBroker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionBroker")
            .field("executor_domains", &self.executors.keys())
            .finish()
    }
}

impl ExecutionBroker {
    pub fn new(capabilities: CapabilityBroker) -> Self {
        Self {
            capabilities,
            executors: HashMap::new(),
        }
    }

    pub fn register(&mut self, executor: Arc<dyn Executor>) {
        self.executors.insert(executor.domain(), executor);
    }

    pub fn supports_procedure_streams(&self, domain: ExecutionDomain) -> bool {
        self.executors
            .get(&domain)
            .is_some_and(|executor| executor.supports_procedure_streams())
    }

    pub fn select<'a>(
        &self,
        action: &'a CompiledAction,
    ) -> CoreResult<&'a ImplementationCandidate> {
        action
            .candidates
            .iter()
            .find(|candidate| self.executors.contains_key(&candidate.executor))
            .ok_or_else(|| {
                CoreError::ExecutorUnavailable(format!(
                    "no registered executor can safely implement {}",
                    action.proposal.action.kind()
                ))
            })
    }

    pub async fn execute(
        &self,
        action: &CompiledAction,
        implementation: &ImplementationCandidate,
        grant: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        self.execute_with_streams(action, implementation, grant, None)
            .await
    }

    /// The only broker route for stream-aware execution. It consumes the same
    /// exact, one-use grant as ordinary execution before an executor can see
    /// either the action or its data endpoints.
    pub async fn execute_with_streams(
        &self,
        action: &CompiledAction,
        implementation: &ImplementationCandidate,
        grant: &CapabilityGrant,
        streams: Option<ProcedureNodeStreams>,
    ) -> CoreResult<ExecutionReceipt> {
        let executor = self
            .executors
            .get(&implementation.executor)
            .ok_or_else(|| {
                CoreError::ExecutorUnavailable(format!("{:?}", implementation.executor))
            })?;
        if streams.as_ref().is_some_and(|streams| !streams.is_empty())
            && !executor.supports_procedure_streams()
        {
            return Err(CoreError::ExecutorUnavailable(format!(
                "{} is not qualified for procedure stream execution",
                executor.name()
            )));
        }
        let consumed = self
            .capabilities
            .consume(
                grant.id,
                action.proposal.task_id,
                action.proposal.id,
                implementation.executor,
            )
            .await?;
        if consumed.action_digest != crate::policy::approval_digest(&action.proposal)?
            || consumed.policy_version != crate::contracts::POLICY_VERSION
        {
            return Err(CoreError::CapabilityRejected(
                "Prepared action or policy changed".into(),
            ));
        }
        let mut streams = streams;
        let receipt = executor
            .execute_with_streams(action, implementation, &consumed, streams.as_mut())
            .await?;
        if let Some(streams) = streams {
            streams.validate_terminal_frames()?;
        }
        Ok(receipt)
    }
}
