//! Durable user decisions. A response commits before a waiting executor receives
//! it; a restart invalidates outstanding authority and retains the record.
use chrono::{DateTime, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::contracts::ApprovalRecord;
use crate::domain::{Task, TaskStatus};
use crate::events::{CoreEvent, CoreEventKind};
use crate::storage::{LocalStore, write_event, write_task};
use crate::{CoreError, CoreResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionRecord {
    pub question_id: Uuid,
    pub task_id: Uuid,
    pub action_id: Uuid,
    pub question: String,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum DecisionRecord {
    Approval(ApprovalRecord),
    Question(QuestionRecord),
}

impl DecisionRecord {
    pub fn id(&self) -> Uuid {
        match self {
            Self::Approval(p) => p.approval_id,
            Self::Question(q) => q.question_id,
        }
    }
    pub fn task_id(&self) -> Uuid {
        match self {
            Self::Approval(p) => p.task_id,
            Self::Question(q) => q.task_id,
        }
    }
    pub fn action_id(&self) -> Uuid {
        match self {
            Self::Approval(p) => p.action_id,
            Self::Question(q) => q.action_id,
        }
    }
    pub fn expires_at(&self) -> DateTime<Utc> {
        match self {
            Self::Approval(p) => p.expires_at,
            Self::Question(q) => q.expires_at,
        }
    }
    pub fn task_status(&self) -> TaskStatus {
        match self {
            Self::Approval(_) => TaskStatus::WaitingForApproval,
            Self::Question(_) => TaskStatus::WaitingForUser,
        }
    }
    pub fn opened(&self) -> CoreEvent {
        CoreEvent::new(
            Some(self.task_id()),
            match self {
                Self::Approval(p) => CoreEventKind::ApprovalRequested {
                    approval_id: p.approval_id,
                    action_id: p.action_id,
                    digest: p.digest.clone(),
                    explanation: p.explanation.clone(),
                    resource: p.resource.clone(),
                    risk: p.risk,
                    expires_at: p.expires_at,
                    reversible: p.reversible,
                    requires_native_authentication: p.requires_native_authentication,
                },
                Self::Question(q) => CoreEventKind::QuestionRequested {
                    question_id: q.question_id,
                    action_id: q.action_id,
                    question: q.question.clone(),
                    expires_at: q.expires_at,
                },
            },
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "state", content = "answer", rename_all = "snake_case")]
pub(crate) enum DecisionResolution {
    Approved,
    Denied,
    Answered(String),
    Expired,
    Cancelled,
}

impl DecisionResolution {
    pub fn state(&self) -> &'static str {
        match self {
            Self::Approved => "approved",
            Self::Denied => "denied",
            Self::Answered(_) => "answered",
            Self::Expired => "expired",
            Self::Cancelled => "cancelled",
        }
    }
    fn requires_live_decision(&self) -> bool {
        matches!(self, Self::Approved | Self::Denied | Self::Answered(_))
    }
}

impl LocalStore {
    pub(crate) fn open_decision(
        &self,
        task: &mut Task,
        decision: &DecisionRecord,
        event: &CoreEvent,
    ) -> CoreResult<()> {
        if task.id != decision.task_id()
            || task.status != decision.task_status()
            || event.task_id != Some(task.id)
        {
            return Err(CoreError::InvalidAction(
                "Decision does not match its waiting task".into(),
            ));
        }
        let revision = self.with_connection(|db| {
            let transaction = db.transaction()?;
            let revision = write_task(&transaction, task)?;
            transaction.execute(
                "INSERT INTO decisions(id,task_id,action_id,payload_json,state,created_at,expires_at) VALUES(?1,?2,?3,?4,'pending',?5,?6)",
                params![decision.id().to_string(),task.id.to_string(),decision.action_id().to_string(),serde_json::to_string(decision)?,
                    event.occurred_at.to_rfc3339(),decision.expires_at().to_rfc3339()])?;
            write_event(&transaction, event)?;
            transaction.commit()?;
            Ok(revision)
        })?;
        task.revision = revision;
        Ok(())
    }

    /// Returns no event if another response, cancellation or expiry won. The
    /// immutable payload comparison binds the response to the exact prompt.
    pub(crate) fn resolve_decision(
        &self,
        decision: &DecisionRecord,
        resolution: &DecisionResolution,
    ) -> CoreResult<Option<CoreEvent>> {
        let event = CoreEvent::new(
            Some(decision.task_id()),
            CoreEventKind::DecisionResolved {
                decision_id: decision.id(),
                state: resolution.state().into(),
            },
        );
        self.with_connection(|db| {
            let transaction = db.transaction()?;
            let changed = transaction.execute(
                "UPDATE decisions SET state=?2,resolution_json=?3,resolved_at=?4 WHERE id=?1 AND state='pending' AND payload_json=?5 AND (?6=0 OR expires_at>?4)",
                params![decision.id().to_string(),resolution.state(),serde_json::to_string(resolution)?,event.occurred_at.to_rfc3339(),
                    serde_json::to_string(decision)?,resolution.requires_live_decision()])?;
            if changed == 0 { return Ok(None); }
            write_event(&transaction, &event)?;
            transaction.commit()?;
            Ok(Some(event))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_commits_are_atomic_and_restart_cannot_reuse_pending_authority() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.db");
        let key = crate::secrets::SecretBytes::new(vec![19; 32]);
        let store = LocalStore::open_encrypted(&path, &key).unwrap();
        let mut task = Task::new("A pending decision");
        task.status = TaskStatus::Running;
        store.save_task(&mut task).unwrap();
        let record = DecisionRecord::Question(QuestionRecord {
            question_id: Uuid::new_v4(),
            task_id: task.id,
            action_id: Uuid::new_v4(),
            question: "Which document?".into(),
            expires_at: Utc::now() + chrono::Duration::minutes(30),
        });
        let event = record.opened();
        store.save_event(&event).unwrap();
        task.status = TaskStatus::WaitingForUser;
        assert!(store.open_decision(&mut task, &record, &event).is_err());
        assert_eq!(
            store.load_tasks(true).unwrap()[0].status,
            TaskStatus::Running
        );
        assert!(
            store
                .resolve_decision(&record, &DecisionResolution::Answered("orphan".into()))
                .unwrap()
                .is_none()
        );
        store
            .open_decision(&mut task, &record, &record.opened())
            .unwrap();
        drop(store);

        let store = LocalStore::open_encrypted(&path, &key).unwrap();
        assert_eq!(
            store.load_tasks(true).unwrap()[0].status,
            TaskStatus::Interrupted
        );
        assert!(
            store
                .resolve_decision(&record, &DecisionResolution::Answered("stale".into()))
                .unwrap()
                .is_none()
        );
        let expired = DecisionRecord::Question(QuestionRecord {
            question_id: Uuid::new_v4(),
            task_id: task.id,
            action_id: record.action_id(),
            question: "Expired question".into(),
            expires_at: Utc::now() - chrono::Duration::seconds(1),
        });
        assert!(
            store
                .open_decision(&mut task, &expired, &expired.opened())
                .is_err(),
            "The pre-restart task revision is stale"
        );
        task = store.load_tasks(true).unwrap().remove(0);
        task.status = TaskStatus::WaitingForUser;
        store
            .open_decision(&mut task, &expired, &expired.opened())
            .unwrap();
        assert!(
            store
                .resolve_decision(&expired, &DecisionResolution::Answered("late".into()))
                .unwrap()
                .is_none()
        );
        assert!(
            store
                .resolve_decision(&expired, &DecisionResolution::Expired)
                .unwrap()
                .is_some()
        );
        assert!(
            store
                .resolve_decision(&expired, &DecisionResolution::Expired)
                .unwrap()
                .is_none()
        );
    }
}
