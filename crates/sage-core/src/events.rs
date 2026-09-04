use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::domain::{Task, TaskStatus};
use crate::policy::RiskLevel;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CoreEventKind {
    TaskStarted,
    PlanGenerated {
        action_count: usize,
    },
    ActionProposed {
        action_id: Uuid,
        summary: String,
    },
    PolicyDenied {
        action_id: Uuid,
        reason: String,
    },
    ApprovalRequested {
        approval_id: Uuid,
        action_id: Uuid,
        digest: String,
        explanation: String,
        resource: String,
        risk: RiskLevel,
        expires_at: DateTime<Utc>,
        reversible: bool,
        requires_native_authentication: bool,
    },
    QuestionRequested {
        question_id: Uuid,
        action_id: Uuid,
        question: String,
        expires_at: DateTime<Utc>,
    },
    ApprovalResolved {
        action_id: Uuid,
        approved: bool,
    },
    ActionStarted {
        action_id: Uuid,
        implementation: String,
    },
    ActionSucceeded {
        action_id: Uuid,
        summary: String,
    },
    ActionFailed {
        action_id: Uuid,
        error: String,
    },
    ObservationReceived {
        action_id: Uuid,
        summary: String,
    },
    VerificationFailed {
        action_id: Uuid,
        reason: String,
    },
    ReplanningStarted {
        attempt: u32,
    },
    PermissionChanged {
        permission: String,
        granted: bool,
    },
    ModelDisconnected {
        provider: String,
    },
    SandboxTerminated {
        reason: String,
    },
    TaskStatusChanged {
        status: TaskStatus,
        summary: String,
    },
    TaskCompleted {
        outcome: String,
    },
    Error {
        code: String,
        message: String,
        recoverable: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreEvent {
    pub id: Uuid,
    pub task_id: Option<Uuid>,
    pub occurred_at: DateTime<Utc>,
    pub kind: CoreEventKind,
}

impl CoreEvent {
    pub fn new(task_id: Option<Uuid>, kind: CoreEventKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            task_id,
            occurred_at: Utc::now(),
            kind,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EventHub {
    sender: Arc<broadcast::Sender<CoreEvent>>,
}

impl Default for EventHub {
    fn default() -> Self {
        let (sender, _) = broadcast::channel(512);
        Self {
            sender: Arc::new(sender),
        }
    }
}

impl EventHub {
    pub fn subscribe(&self) -> broadcast::Receiver<CoreEvent> {
        self.sender.subscribe()
    }

    pub fn publish(&self, event: CoreEvent) {
        let _ = self.sender.send(event);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub tasks: Vec<Task>,
    pub provider_settings: Vec<crate::model::ProviderSettings>,
    pub core_version: String,
    pub protocol_version: u32,
}
