mod native;
mod worker;

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

pub use native::{NativeExecutor, PlatformController, UnsupportedPlatformController};
pub use worker::{FramedWorkerExecutor, WorkerConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RollbackOperation {
    MoveFile { source: String, destination: String },
    RestoreFile { backup: String, destination: String },
    RemoveEmptyFolder { path: String },
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
    async fn execute(
        &self,
        action: &CompiledAction,
        implementation: &ImplementationCandidate,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt>;
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
        let executor = self
            .executors
            .get(&implementation.executor)
            .ok_or_else(|| {
                CoreError::ExecutorUnavailable(format!("{:?}", implementation.executor))
            })?;
        let consumed = self
            .capabilities
            .consume(
                grant.id,
                action.proposal.task_id,
                action.proposal.id,
                implementation.executor,
            )
            .await?;
        executor.execute(action, implementation, &consumed).await
    }
}
