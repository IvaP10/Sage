use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::domain::{Task, TaskStatus};
use crate::error::{CoreError, CoreResult};
use crate::events::CoreEvent;

#[derive(Clone)]
pub struct LocalStore {
    connection: Arc<Mutex<Connection>>,
}

impl std::fmt::Debug for LocalStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LocalStore { connection: [REDACTED] }")
    }
}

impl LocalStore {
    pub fn open(path: &Path) -> CoreResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let connection =
            Connection::open(path).map_err(|error| CoreError::Storage(error.to_string()))?;
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(|error| CoreError::Storage(error.to_string()))?;
        connection
            .pragma_update(None, "foreign_keys", "ON")
            .map_err(|error| CoreError::Storage(error.to_string()))?;
        connection
            .pragma_update(None, "busy_timeout", 5_000)
            .map_err(|error| CoreError::Storage(error.to_string()))?;
        let store = Self {
            connection: Arc::new(Mutex::new(connection)),
        };
        store.migrate()?;
        store.mark_incomplete_tasks_interrupted()?;
        Ok(store)
    }

    fn migrate(&self) -> CoreResult<()> {
        self.with_connection(|connection| {
            connection.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    request TEXT NOT NULL,
                    status TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS actions (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    action_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    task_id TEXT,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    status TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                CREATE TABLE IF NOT EXISTS capabilities (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE TABLE IF NOT EXISTS rollback_plans (
                    action_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS permissions (
                    permission TEXT PRIMARY KEY,
                    granted INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    memory_id UNINDEXED,
                    content,
                    tokenize = 'unicode61'
                );
                CREATE TABLE IF NOT EXISTS tool_registry (
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    descriptor_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(name, version)
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    task_id TEXT,
                    action_id TEXT,
                    event_type TEXT NOT NULL,
                    redacted_payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                VALUES (1, CURRENT_TIMESTAMP);
                "#,
            )?;
            Ok(())
        })
    }

    pub fn save_task(&self, task: &Task) -> CoreResult<()> {
        let task_json = serde_json::to_string(task)?;
        let status = status_name(task.status);
        self.with_connection(|connection| {
            let transaction = connection.unchecked_transaction()?;
            transaction.execute(
                r#"INSERT INTO tasks(id, request, status, task_json, created_at, updated_at)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                   ON CONFLICT(id) DO UPDATE SET
                     request=excluded.request,
                     status=excluded.status,
                     task_json=excluded.task_json,
                     updated_at=excluded.updated_at"#,
                params![
                    task.id.to_string(),
                    task.request,
                    status,
                    task_json,
                    task.created_at.to_rfc3339(),
                    task.updated_at.to_rfc3339(),
                ],
            )?;
            for state in task.actions.values() {
                transaction.execute(
                    r#"INSERT INTO actions(id, task_id, kind, status, action_json, updated_at)
                       VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                       ON CONFLICT(id) DO UPDATE SET
                         status=excluded.status,
                         action_json=excluded.action_json,
                         updated_at=excluded.updated_at"#,
                    params![
                        state.proposal.id.to_string(),
                        task.id.to_string(),
                        state.proposal.action.kind(),
                        format!("{:?}", state.status).to_ascii_lowercase(),
                        serde_json::to_string(state)?,
                        task.updated_at.to_rfc3339(),
                    ],
                )?;
            }
            transaction.commit()?;
            Ok(())
        })
    }

    pub fn load_tasks(&self, include_completed: bool) -> CoreResult<Vec<Task>> {
        self.with_connection(|connection| {
            let sql = if include_completed {
                "SELECT task_json FROM tasks ORDER BY created_at DESC"
            } else {
                "SELECT task_json FROM tasks WHERE status NOT IN ('succeeded','failed','cancelled') ORDER BY created_at DESC"
            };
            let mut statement = connection.prepare(sql)?;
            let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
            let mut tasks = Vec::new();
            for row in rows {
                tasks.push(serde_json::from_str(&row?)?);
            }
            Ok(tasks)
        })
    }

    pub fn save_event(&self, event: &CoreEvent) -> CoreResult<()> {
        let payload = serde_json::to_string(event)?;
        self.with_connection(|connection| {
            connection.execute(
                "INSERT INTO events(id, task_id, kind, payload_json, occurred_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    event.id.to_string(),
                    event.task_id.map(|id| id.to_string()),
                    event_kind_name(event),
                    payload,
                    event.occurred_at.to_rfc3339(),
                ],
            )?;
            Ok(())
        })
    }

    pub fn append_audit<T: Serialize>(
        &self,
        task_id: Option<Uuid>,
        action_id: Option<Uuid>,
        event_type: &str,
        redacted_payload: &T,
    ) -> CoreResult<String> {
        let payload = serde_json::to_string(redacted_payload)?;
        let occurred_at = Utc::now().to_rfc3339();
        self.with_connection(|connection| {
            let previous_hash: String = connection
                .query_row(
                    "SELECT record_hash FROM audit_log ORDER BY sequence DESC LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .optional()?
                .unwrap_or_else(|| "GENESIS".into());
            let id = Uuid::new_v4();
            let canonical = format!(
                "{}|{}|{}|{}|{}|{}|{}",
                id,
                task_id.map(|value| value.to_string()).unwrap_or_default(),
                action_id.map(|value| value.to_string()).unwrap_or_default(),
                event_type,
                payload,
                previous_hash,
                occurred_at
            );
            let record_hash = format!("{:x}", Sha256::digest(canonical.as_bytes()));
            connection.execute(
                r#"INSERT INTO audit_log(
                     id, task_id, action_id, event_type, redacted_payload_json,
                     previous_hash, record_hash, occurred_at
                   ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"#,
                params![
                    id.to_string(),
                    task_id.map(|value| value.to_string()),
                    action_id.map(|value| value.to_string()),
                    event_type,
                    payload,
                    previous_hash,
                    record_hash,
                    occurred_at,
                ],
            )?;
            Ok(record_hash)
        })
    }

    pub fn set_permission(&self, permission: &str, granted: bool) -> CoreResult<()> {
        self.with_connection(|connection| {
            connection.execute(
                r#"INSERT INTO permissions(permission, granted, updated_at)
                   VALUES (?1, ?2, ?3)
                   ON CONFLICT(permission) DO UPDATE SET
                     granted=excluded.granted,
                     updated_at=excluded.updated_at"#,
                params![permission, i64::from(granted), Utc::now().to_rfc3339()],
            )?;
            Ok(())
        })
    }

    pub fn save_setting<T: Serialize>(&self, key: &str, value: &T) -> CoreResult<()> {
        let value_json = serde_json::to_string(value)?;
        self.with_connection(|connection| {
            connection.execute(
                r#"INSERT INTO settings(key, value_json, updated_at)
                   VALUES (?1, ?2, ?3)
                   ON CONFLICT(key) DO UPDATE SET
                     value_json=excluded.value_json,
                     updated_at=excluded.updated_at"#,
                params![key, value_json, Utc::now().to_rfc3339()],
            )?;
            Ok(())
        })
    }

    pub fn load_setting<T: DeserializeOwned>(&self, key: &str) -> CoreResult<Option<T>> {
        self.with_connection(|connection| {
            let value: Option<String> = connection
                .query_row(
                    "SELECT value_json FROM settings WHERE key=?1",
                    params![key],
                    |row| row.get(0),
                )
                .optional()?;
            value
                .map(|value| serde_json::from_str(&value))
                .transpose()
                .map_err(Into::into)
        })
    }

    pub fn save_rollback(
        &self,
        task_id: Uuid,
        plan: &crate::execution::RollbackPlan,
    ) -> CoreResult<()> {
        self.with_connection(|connection| {
            connection.execute(
                r#"INSERT INTO rollback_plans(action_id, task_id, plan_json, expires_at, consumed_at)
                   VALUES (?1, ?2, ?3, ?4, NULL)
                   ON CONFLICT(action_id) DO UPDATE SET
                     plan_json=excluded.plan_json,
                     expires_at=excluded.expires_at,
                     consumed_at=NULL"#,
                params![
                    plan.action_id.to_string(),
                    task_id.to_string(),
                    serde_json::to_string(plan)?,
                    plan.expires_at.to_rfc3339(),
                ],
            )?;
            Ok(())
        })
    }

    pub fn latest_rollback(
        &self,
        task_id: Uuid,
    ) -> CoreResult<Option<crate::execution::RollbackPlan>> {
        self.with_connection(|connection| {
            let json: Option<String> = connection
                .query_row(
                    r#"SELECT plan_json FROM rollback_plans
                       WHERE task_id=?1 AND consumed_at IS NULL AND expires_at > ?2
                       ORDER BY rowid DESC LIMIT 1"#,
                    params![task_id.to_string(), Utc::now().to_rfc3339()],
                    |row| row.get(0),
                )
                .optional()?;
            json.map(|value| serde_json::from_str(&value))
                .transpose()
                .map_err(Into::into)
        })
    }

    pub fn consume_rollback(&self, action_id: Uuid) -> CoreResult<()> {
        self.with_connection(|connection| {
            connection.execute(
                "UPDATE rollback_plans SET consumed_at=?2 WHERE action_id=?1 AND consumed_at IS NULL",
                params![action_id.to_string(), Utc::now().to_rfc3339()],
            )?;
            Ok(())
        })
    }

    fn mark_incomplete_tasks_interrupted(&self) -> CoreResult<()> {
        self.with_connection(|connection| {
            let mut statement = connection.prepare(
                "SELECT task_json FROM tasks WHERE status IN ('pending','planning','running','waiting_for_approval','waiting_for_user','paused')",
            )?;
            let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
            let mut tasks = Vec::new();
            for row in rows {
                let mut task: Task = serde_json::from_str(&row?)?;
                task.status = TaskStatus::Interrupted;
                task.final_outcome = Some("SAGE Core restarted before this task completed; it was not resumed automatically.".into());
                task.touch();
                tasks.push(task);
            }
            drop(statement);
            for task in tasks {
                connection.execute(
                    "UPDATE tasks SET status='interrupted', task_json=?2, updated_at=?3 WHERE id=?1",
                    params![
                        task.id.to_string(),
                        serde_json::to_string(&task)?,
                        task.updated_at.to_rfc3339(),
                    ],
                )?;
            }
            Ok(())
        })
    }

    fn with_connection<T>(
        &self,
        operation: impl FnOnce(&mut Connection) -> Result<T, Box<dyn std::error::Error + Send + Sync>>,
    ) -> CoreResult<T> {
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| CoreError::Storage("database lock poisoned".into()))?;
        operation(&mut connection).map_err(|error| CoreError::Storage(error.to_string()))
    }
}

fn status_name(status: TaskStatus) -> &'static str {
    match status {
        TaskStatus::Pending => "pending",
        TaskStatus::Planning => "planning",
        TaskStatus::Running => "running",
        TaskStatus::WaitingForApproval => "waiting_for_approval",
        TaskStatus::WaitingForUser => "waiting_for_user",
        TaskStatus::Paused => "paused",
        TaskStatus::Succeeded => "succeeded",
        TaskStatus::Failed => "failed",
        TaskStatus::Cancelled => "cancelled",
        TaskStatus::Interrupted => "interrupted",
    }
}

fn event_kind_name(event: &CoreEvent) -> &'static str {
    use crate::events::CoreEventKind;
    match &event.kind {
        CoreEventKind::TaskStarted => "task_started",
        CoreEventKind::PlanGenerated { .. } => "plan_generated",
        CoreEventKind::ActionProposed { .. } => "action_proposed",
        CoreEventKind::PolicyDenied { .. } => "policy_denied",
        CoreEventKind::ApprovalRequested { .. } => "approval_requested",
        CoreEventKind::QuestionRequested { .. } => "question_requested",
        CoreEventKind::ApprovalResolved { .. } => "approval_resolved",
        CoreEventKind::ActionStarted { .. } => "action_started",
        CoreEventKind::ActionSucceeded { .. } => "action_succeeded",
        CoreEventKind::ActionFailed { .. } => "action_failed",
        CoreEventKind::ObservationReceived { .. } => "observation_received",
        CoreEventKind::VerificationFailed { .. } => "verification_failed",
        CoreEventKind::ReplanningStarted { .. } => "replanning_started",
        CoreEventKind::PermissionChanged { .. } => "permission_changed",
        CoreEventKind::ModelDisconnected { .. } => "model_disconnected",
        CoreEventKind::SandboxTerminated { .. } => "sandbox_terminated",
        CoreEventKind::TaskStatusChanged { .. } => "task_status_changed",
        CoreEventKind::TaskCompleted { .. } => "task_completed",
        CoreEventKind::Error { .. } => "error",
    }
}
