use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::Serialize;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::domain::{Task, TaskStatus};
use crate::error::{CoreError, CoreResult};
use crate::events::CoreEvent;

const MAX_PROCEDURE_CHECKPOINT_BYTES: usize = 4 * 1024 * 1024;
const MAX_CONTROLLER_RECORD_BYTES: usize = 256 * 1024;
const MAX_CONTROLLERS_PER_SYSTEM: u64 = 128;
type StoredWorldObservationRow = (
    String,
    String,
    String,
    String,
    Option<String>,
    String,
    String,
    String,
    String,
    String,
);

#[derive(Clone)]
pub struct LocalStore {
    connection: Arc<Mutex<Connection>>,
    path: std::path::PathBuf,
    unlocked: Arc<std::sync::atomic::AtomicBool>,
    pub(crate) audit_lock: Arc<Mutex<crate::audit::AuditState>>,
    pub(crate) audit_watch: Arc<crate::audit::AuditWatch>,
    pub(crate) retention_lock: Arc<Mutex<()>>,
}

impl std::fmt::Debug for LocalStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LocalStore { connection: [REDACTED] }")
    }
}

impl LocalStore {
    /// Startup uses an empty, volatile store. OS key access only happens after
    /// an explicit user operation calls unlock; no history is copied here.
    pub fn deferred(path: &Path) -> CoreResult<Self> {
        let store = Self {
            connection: Arc::new(Mutex::new(Connection::open_in_memory()?)),
            path: path.into(),
            unlocked: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            audit_lock: Arc::default(),
            audit_watch: Arc::default(),
            retention_lock: Arc::new(Mutex::new(())),
        };
        store.with_connection(|db| {
            crate::audit::install_watch(db, store.audit_watch.clone());
            Ok(())
        })?;
        store.migrate()?;
        store.migrate_journal()?;
        store.migrate_retention_cleanup()?;
        Ok(store)
    }

    pub fn is_locked(&self) -> bool {
        !self.unlocked.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn open_encrypted(path: &Path, key: &crate::secrets::SecretBytes) -> CoreResult<Self> {
        if key.expose().len() != 32 {
            return Err(CoreError::Storage("Invalid database key length".into()));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        crate::vault::migrate_plaintext(path, key)?;
        let connection = crate::vault::open_encrypted(path, key)?;
        let store = Self {
            connection: Arc::new(Mutex::new(connection)),
            path: path.into(),
            unlocked: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            audit_lock: Arc::default(),
            audit_watch: Arc::default(),
            retention_lock: Arc::new(Mutex::new(())),
        };
        store.with_connection(|db| {
            crate::audit::install_watch(db, store.audit_watch.clone());
            Ok(())
        })?;
        store.migrate()?;
        store.migrate_journal()?;
        store.migrate_retention_cleanup()?;
        store.mark_incomplete_tasks_interrupted()?;
        store.reconcile_stored_worker_receipts()?;
        store.repair_undo_availability()?;
        Ok(store)
    }

    pub fn unlock(&self, secrets: &dyn crate::secrets::SecretStore) -> CoreResult<bool> {
        if !self.is_locked() {
            return Ok(false);
        }
        let key = match secrets.get("database-v2")? {
            Some(key) => key,
            None => {
                if self.path.exists() && !crate::vault::is_plaintext(&self.path)? {
                    return Err(CoreError::Storage("The encrypted database key is unavailable. Restore its OS credential before opening history.".into()));
                }
                let mut bytes = vec![0; 32];
                getrandom::fill(&mut bytes)
                    .map_err(|_| CoreError::SecretStore("Randomness unavailable".into()))?;
                let key = crate::secrets::SecretBytes::new(bytes);
                secrets.set("database-v2", &key)?;
                key
            }
        };
        let replacement = Self::open_encrypted(&self.path, &key)?;
        replacement.migrate_knowledge()?;
        replacement.migrate_workflows()?;
        replacement.migrate_world_model()?;
        replacement.checkpoint_audit(secrets)?;
        replacement.retire_expired_artifacts()?;
        // A verification cursor belongs to one connection and its hooks. Never
        // carry the deferred connection's cursor across the replacement.
        let mut audit = self
            .audit_lock
            .lock()
            .map_err(|_| CoreError::Storage("Audit lock poisoned".into()))?;
        let mut target = self
            .connection
            .lock()
            .map_err(|_| CoreError::Storage("Database lock poisoned".into()))?;
        let mut source = replacement
            .connection
            .lock()
            .map_err(|_| CoreError::Storage("Database lock poisoned".into()))?;
        std::mem::swap(&mut *target, &mut *source);
        crate::audit::install_watch(&target, self.audit_watch.clone());
        *audit = Default::default();
        self.unlocked
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(true)
    }

    fn migrate(&self) -> CoreResult<()> {
        self.with_connection(|connection| {
            connection.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS private_artifacts (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, content BLOB NOT NULL,
                    sha256 TEXT NOT NULL, expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retired_task_data (
                    task_id TEXT PRIMARY KEY, retired_at TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS reject_retired_task_artifacts
                BEFORE INSERT ON private_artifacts
                WHEN EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=NEW.task_id)
                BEGIN SELECT RAISE(ABORT,'Artifact retention was revoked for this task'); END;
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
                CREATE INDEX IF NOT EXISTS tasks_continuation_parent ON tasks(json_extract(task_json,'$.continuation_of'));
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    task_id TEXT,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS command_inbox (
                    principal TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    accepted_at TEXT NOT NULL,
                    PRIMARY KEY(principal, request_id)
                );
                CREATE TABLE IF NOT EXISTS control_scopes (
                    id TEXT PRIMARY KEY, stopped_at TEXT
                );
                CREATE TABLE IF NOT EXISTS run_finalizations (
                    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    execution_attempt INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    task_revision INTEGER NOT NULL,
                    finalized_at TEXT NOT NULL,
                    PRIMARY KEY(task_id,execution_attempt,kind)
                );
                CREATE TABLE IF NOT EXISTS procedure_checkpoints (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL,
                    procedure_sha256 TEXT NOT NULL,
                    procedure_json TEXT NOT NULL,
                    runtime_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS reject_retired_procedure_checkpoints
                BEFORE INSERT ON procedure_checkpoints
                WHEN EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=NEW.task_id)
                BEGIN SELECT RAISE(ABORT,'Procedure checkpoint retention was revoked for this task'); END;
                CREATE TRIGGER IF NOT EXISTS cleanup_retired_task_private_data
                AFTER INSERT ON retired_task_data
                BEGIN
                    DELETE FROM private_artifacts WHERE task_id=NEW.task_id;
                    DELETE FROM procedure_checkpoints WHERE task_id=NEW.task_id;
                END;
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    action_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    resolution_json TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS decisions_pending ON decisions(task_id,state,expires_at);
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
                CREATE TABLE IF NOT EXISTS undo_journal (
                    action_id TEXT PRIMARY KEY REFERENCES rollback_plans(action_id),
                    task_id TEXT NOT NULL REFERENCES tasks(id),
                    intent_json TEXT NOT NULL,
                    phase TEXT NOT NULL CHECK(phase IN ('prepared','dispatched','uncertain','verified')),
                    evidence_json TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS undo_pending ON undo_journal(task_id,phase);
                CREATE INDEX IF NOT EXISTS rollback_task ON rollback_plans(task_id,consumed_at);
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
            let migrated: bool = connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version=5)", [], |row| row.get(0))?;
            if !migrated {
                // Legacy account names did not identify an endpoint. Keep the
                // OS secret intact, but never automatically release it to a
                // destination inferred from old settings.
                let tx = connection.transaction()?;
                tx.execute("UPDATE settings SET value_json=json_set(value_json,'$.has_api_key',json('false')) WHERE key='provider.reasoning'", [])?;
                tx.execute("INSERT INTO schema_migrations VALUES(5,CURRENT_TIMESTAMP)", [])?;
                tx.commit()?;
            }
            let checkpoint_migration: bool = connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version=9)",
                [],
                |row| row.get(0),
            )?;
            if !checkpoint_migration {
                connection.execute(
                    "INSERT INTO schema_migrations VALUES(9, CURRENT_TIMESTAMP)",
                    [],
                )?;
            }
            Ok(())
        })
    }

    /// Run after journal migrations so deleting artifact bodies preserves old
    /// worker-receipt rows through their upgraded `ON DELETE SET NULL` link.
    fn migrate_retention_cleanup(&self) -> CoreResult<()> {
        self.with_connection(|connection| {
            let migrated: bool = connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version=10)",
                [],
                |row| row.get(0),
            )?;
            if !migrated {
                let transaction = connection.transaction()?;
                transaction.execute_batch(
                    "DROP TRIGGER IF EXISTS delete_retired_procedure_checkpoints",
                )?;
                transaction.execute(
                    "DELETE FROM private_artifacts WHERE task_id IN (SELECT task_id FROM retired_task_data)",
                    [],
                )?;
                transaction.execute(
                    "DELETE FROM procedure_checkpoints WHERE task_id IN (SELECT task_id FROM retired_task_data)",
                    [],
                )?;
                transaction.execute(
                    "INSERT INTO schema_migrations VALUES(10, CURRENT_TIMESTAMP)",
                    [],
                )?;
                transaction.commit()?;
            }
            Ok(())
        })
    }

    pub(crate) fn database_path(&self) -> &Path {
        &self.path
    }

    pub fn save_artifact(&self, task: Uuid, bytes: &[u8]) -> CoreResult<Uuid> {
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(CoreError::Storage(
                "Artifact exceeds the 16 MiB limit".into(),
            ));
        }
        let id = Uuid::new_v4();
        let digest = format!("{:x}", Sha256::digest(bytes));
        self.with_connection(|db| {
            db.execute(
                "INSERT INTO private_artifacts VALUES(?1,?2,?3,?4,?5)",
                params![
                    id.to_string(),
                    task.to_string(),
                    bytes,
                    digest,
                    (Utc::now() + chrono::Duration::hours(24)).to_rfc3339()
                ],
            )?;
            Ok(())
        })?;
        Ok(id)
    }

    pub fn read_artifact(&self, id: Uuid) -> CoreResult<Vec<u8>> {
        self.with_connection(|db| {
            let (bytes, digest, expiry): (Vec<u8>, String, String) = db.query_row(
                "SELECT content,sha256,expires_at FROM private_artifacts WHERE id=?1",
                [id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
            if chrono::DateTime::parse_from_rfc3339(&expiry)? <= Utc::now()
                || format!("{:x}", Sha256::digest(&bytes)) != digest
            {
                return Err("Artifact expired or failed integrity verification".into());
            }
            Ok(bytes)
        })
    }

    /// Read an artifact only when it belongs to the requesting task and still
    /// matches the digest bound into that task's verified result. Procedure
    /// inputs must use this path instead of treating an artifact UUID as
    /// authority by itself.
    pub fn read_artifact_for_task(
        &self,
        id: Uuid,
        task_id: Uuid,
        expected_sha256: &str,
    ) -> CoreResult<Vec<u8>> {
        self.read_artifact_for_owner(id, task_id, Some(expected_sha256))
    }

    fn read_artifact_for_owner(
        &self,
        id: Uuid,
        task_id: Uuid,
        expected_sha256: Option<&str>,
    ) -> CoreResult<Vec<u8>> {
        if expected_sha256.is_some_and(|digest| {
            digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        }) {
            return Err(CoreError::InvalidAction(
                "Artifact reference contains an invalid SHA-256 digest".into(),
            ));
        }
        let (bytes, digest, expiry): (Vec<u8>, String, String) = self.with_connection(|db| {
            Ok(db.query_row(
                "SELECT content,sha256,expires_at FROM private_artifacts WHERE id=?1 AND task_id=?2 AND NOT EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=?2)",
                params![id.to_string(), task_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?)
        })?;
        let actual_digest = format!("{:x}", Sha256::digest(&bytes));
        let expires_at = chrono::DateTime::parse_from_rfc3339(&expiry).map_err(|_| {
            CoreError::VerificationFailed("Artifact expiry timestamp is invalid".into())
        })?;
        if bytes.len() > 16 * 1024 * 1024
            || expires_at <= Utc::now()
            || !actual_digest.eq_ignore_ascii_case(&digest)
            || expected_sha256.is_some_and(|expected| !actual_digest.eq_ignore_ascii_case(expected))
        {
            return Err(CoreError::VerificationFailed(
                "Artifact expired or failed owner and integrity verification".into(),
            ));
        }
        Ok(bytes)
    }

    pub(crate) fn read_undo_artifact(&self, id: Uuid, task_id: Uuid) -> CoreResult<Vec<u8>> {
        self.read_artifact_for_owner(id, task_id, None)
    }

    /// Persist an unreviewed controller candidate. The record is descriptive
    /// only; it carries no execution grant and cannot be used without fresh
    /// semantic revalidation at invocation time.
    pub fn save_controller_draft(
        &self,
        controller: &crate::agency::ControllerIr,
    ) -> CoreResult<crate::agency::StoredController> {
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Controller storage is unavailable until Sage storage is unlocked".into(),
            ));
        }
        let controller_id = crate::agency::controller_digest(controller)?;
        let controller_json = serde_json::to_string(controller)?;
        if controller_json.len() > MAX_CONTROLLER_RECORD_BYTES {
            return Err(CoreError::Storage(
                "Controller record exceeds its bounded storage limit".into(),
            ));
        }
        let payload_sha256 = format!("{:x}", Sha256::digest(controller_json.as_bytes()));
        if payload_sha256 != controller_id {
            return Err(CoreError::VerificationFailed(
                "Controller digest changed during serialization".into(),
            ));
        }
        let now = Utc::now();
        self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let current_fingerprint: Option<String> = transaction
                .query_row(
                    "SELECT fingerprint FROM world_systems WHERE id=?1",
                    [controller.system_id.to_string()],
                    |row| row.get(0),
                )
                .optional()?;
            if current_fingerprint.as_deref() != Some(controller.system_fingerprint.as_str()) {
                return Err("Controller target is missing or its identity changed".into());
            }
            controller_evidence_map(&transaction, controller, now)?;
            if let Some(existing) = read_stored_controller_tx(&transaction, &controller_id)? {
                if existing.controller.system_id != controller.system_id {
                    return Err("Controller digest is already bound to another system".into());
                }
                transaction.commit()?;
                return Ok(existing);
            }
            let controller_count: u64 = transaction.query_row(
                "SELECT COUNT(*) FROM world_controllers WHERE system_id=?1",
                [controller.system_id.to_string()],
                |row| row.get(0),
            )?;
            if controller_count >= MAX_CONTROLLERS_PER_SYSTEM {
                return Err("This system has reached its retained-controller limit; forget the system to clear its learning data".into());
            }
            let record = crate::agency::StoredController {
                id: controller_id.clone(),
                controller: controller.clone(),
                status: crate::agency::ControllerStatus::Draft,
                revision: 1,
                reviewed_observation_id: None,
                created_at: now,
                updated_at: now,
            };
            transaction.execute(
                "INSERT INTO world_controllers(controller_id,system_id,system_fingerprint,interface_fingerprint,status,revision,controller_json,payload_sha256,reviewed_observation_id,created_at,updated_at) VALUES(?1,?2,?3,?4,'draft',1,?5,?6,NULL,?7,?7)",
                params![
                    &record.id,
                    record.controller.system_id.to_string(),
                    &record.controller.system_fingerprint,
                    &record.controller.interface_fingerprint,
                    &controller_json,
                    &payload_sha256,
                    now.to_rfc3339(),
                ],
            )?;
            crate::storage::write_audit(
                &transaction,
                None,
                None,
                "controller_draft_saved",
                &serde_json::json!({
                    "controller_id": record.id,
                    "system_id": record.controller.system_id,
                    "step_count": record.controller.steps.len(),
                    "payload_sha256": payload_sha256,
                }),
            )?;
            transaction.commit()?;
            Ok(record)
        })
    }

    /// Compile a completed, durably verified procedure using only current
    /// world-model evidence, then persist the result as an unreviewed draft.
    /// This operation never activates the controller or grants execution
    /// authority.
    pub fn compile_controller_draft(
        &self,
        task_id: Uuid,
    ) -> CoreResult<crate::agency::StoredController> {
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Controller compilation is unavailable until Sage storage is unlocked".into(),
            ));
        }
        let checkpoint = self.load_procedure_checkpoint(task_id)?.ok_or_else(|| {
            CoreError::PermissionRequired(
                "Controller compilation requires a durably saved procedure checkpoint".into(),
            )
        })?;
        let procedure = &checkpoint.procedure;
        let runtime = &checkpoint.runtime;
        let first = procedure.nodes.first().ok_or_else(|| {
            CoreError::InvalidAction("Controller compilation requires a non-empty procedure".into())
        })?;
        if procedure.nodes.len() > 64 {
            return Err(CoreError::InvalidAction(
                "Controller compilation is limited to 64 capability steps".into(),
            ));
        }
        let system_id = match &first.kind {
            crate::agency::ProcedureNodeKind::CapabilityCall { system_id, .. } => *system_id,
            _ => {
                return Err(CoreError::ExecutorUnavailable(
                    "Controller compilation currently supports capability-call procedures only"
                        .into(),
                ));
            }
        };
        let capabilities = self.capability_assessments(system_id)?;
        let mut evidence_ids = runtime.verification_evidence_ids();
        let capabilities_by_id = capabilities
            .iter()
            .map(|assessment| (assessment.descriptor.id.as_str(), &assessment.descriptor))
            .collect::<BTreeMap<_, _>>();
        for node in &procedure.nodes {
            if let crate::agency::ProcedureNodeKind::CapabilityCall { capability_id, .. } =
                &node.kind
                && let Some(capability) = capabilities_by_id.get(capability_id.as_str())
            {
                evidence_ids.extend(capability.evidence_ids.iter().copied());
            }
        }
        if evidence_ids.len() > 64 * 32 + 128 {
            return Err(CoreError::InvalidAction(
                "Controller source evidence exceeds its compilation bound".into(),
            ));
        }
        let now = Utc::now();
        let observations = self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let mut by_id = BTreeMap::<Uuid, crate::world_model::ObservationEnvelope>::new();
            for evidence_id in &evidence_ids {
                if let Some((observation, expires_at)) =
                    checked_world_observation_tx(&transaction, *evidence_id)?
                    && expires_at > now
                    && observation.system_id == system_id
                {
                    by_id.insert(*evidence_id, observation);
                }
            }
            let fresh_after = (now - chrono::Duration::seconds(10)).to_rfc3339();
            let now_text = now.to_rfc3339();
            let mut query = transaction.prepare(
                "SELECT id FROM world_observations WHERE system_id=?1 AND expires_at>?2 AND observed_at>?3 ORDER BY observed_at DESC,id DESC LIMIT 128",
            )?;
            let fresh_ids = query
                .query_map(
                    params![system_id.to_string(), now_text, fresh_after],
                    |row| row.get::<_, String>(0),
                )?
                .collect::<Result<Vec<_>, _>>()?;
            drop(query);
            for fresh_id in fresh_ids {
                let observation_id = Uuid::parse_str(&fresh_id)?;
                if let Some((observation, expires_at)) =
                    checked_world_observation_tx(&transaction, observation_id)?
                    && expires_at > now
                    && observation.system_id == system_id
                {
                    by_id.insert(observation_id, observation);
                }
            }
            validate_procedure_verification_observations(
                &transaction,
                procedure,
                runtime,
                &by_id,
                now,
            )?;
            transaction.commit()?;
            Ok(by_id.into_values().collect::<Vec<_>>())
        })?;
        let controller = crate::agency::compile_controller_from_verified_procedure(
            procedure,
            runtime,
            &capabilities,
            &observations,
            now,
        )?;
        self.save_controller_draft(&controller)
    }

    pub fn load_controller(
        &self,
        controller_id: &str,
    ) -> CoreResult<Option<crate::agency::StoredController>> {
        validate_controller_id(controller_id)?;
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Controller storage is unavailable until Sage storage is unlocked".into(),
            ));
        }
        self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let record = read_stored_controller_tx(&transaction, controller_id)?;
            transaction.commit()?;
            Ok(record)
        })
    }

    pub fn controllers_for_system(
        &self,
        system_id: Uuid,
    ) -> CoreResult<Vec<crate::agency::StoredController>> {
        if system_id.is_nil() {
            return Err(CoreError::InvalidAction(
                "Controller listing requires a system identity".into(),
            ));
        }
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Controller storage is unavailable until Sage storage is unlocked".into(),
            ));
        }
        self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let mut query = transaction.prepare(
                "SELECT controller_id FROM world_controllers WHERE system_id=?1 ORDER BY updated_at DESC,controller_id LIMIT 128",
            )?;
            let ids = query
                .query_map([system_id.to_string()], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            drop(query);
            let mut records = Vec::with_capacity(ids.len());
            for id in ids {
                let record = read_stored_controller_tx(&transaction, &id)?
                    .ok_or_else(|| CoreError::VerificationFailed(
                        "Controller disappeared during a consistent listing".into(),
                    ))?;
                if record.controller.system_id != system_id {
                    return Err("Controller listing returned another system's record".into());
                }
                records.push(record);
            }
            transaction.commit()?;
            Ok(records)
        })
    }

    /// Mark one exact draft reviewed only after a fresh persisted observation
    /// uniquely rebinds every semantic anchor. This does not grant execution
    /// authority; callers must repeat revalidation and the normal broker chain.
    pub fn review_controller(
        &self,
        controller_id: &str,
        expected_revision: u64,
        observation_id: Uuid,
    ) -> CoreResult<crate::agency::StoredController> {
        validate_controller_id(controller_id)?;
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Controller storage is unavailable until Sage storage is unlocked".into(),
            ));
        }
        if expected_revision == 0 || expected_revision > i64::MAX as u64 || observation_id.is_nil()
        {
            return Err(CoreError::InvalidAction(
                "Controller review requires a valid revision and observation".into(),
            ));
        }
        let now = Utc::now();
        self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let mut record = read_stored_controller_tx(&transaction, controller_id)?
                .ok_or_else(|| CoreError::InvalidAction("Controller draft was not found".into()))?;
            if record.revision != expected_revision
                || !matches!(
                    record.status,
                    crate::agency::ControllerStatus::Draft
                        | crate::agency::ControllerStatus::Reviewed
                )
            {
                return Err("Controller changed or is not eligible for review".into());
            }
            let current_fingerprint: Option<String> = transaction
                .query_row(
                    "SELECT fingerprint FROM world_systems WHERE id=?1",
                    [record.controller.system_id.to_string()],
                    |row| row.get(0),
                )
                .optional()?;
            if current_fingerprint.as_deref()
                != Some(record.controller.system_fingerprint.as_str())
            {
                return Err("Controller target identity changed; relearning is required".into());
            }
            let current_evidence =
                controller_evidence_map(&transaction, &record.controller, now)?;
            let (observation, observation_expires_at) = checked_world_observation_tx(&transaction, observation_id)?
                .ok_or_else(|| CoreError::VerificationFailed(
                    "Controller review observation is missing or expired".into(),
                ))?;
            if observation_expires_at <= now {
                return Err("Controller review observation has expired".into());
            }
            let rebinding = crate::agency::revalidate_controller(
                &record.controller,
                &observation,
                &current_evidence,
                now,
            )?;
            let next_revision = expected_revision
                .checked_add(1)
                .filter(|revision| *revision <= i64::MAX as u64)
                .ok_or_else(|| CoreError::Storage("Controller revision is exhausted".into()))?;
            let updated_at = now.to_rfc3339();
            let changed = transaction.execute(
                "UPDATE world_controllers SET status='reviewed',revision=?2,reviewed_observation_id=?3,updated_at=?4 WHERE controller_id=?1 AND revision=?5 AND status IN ('draft','reviewed')",
                params![controller_id, next_revision, observation_id.to_string(), updated_at, expected_revision],
            )?;
            if changed != 1 {
                return Err("Controller revision changed during review".into());
            }
            crate::storage::write_audit(
                &transaction,
                None,
                None,
                "controller_reviewed",
                &serde_json::json!({
                    "controller_id": controller_id,
                    "system_id": record.controller.system_id,
                    "observation_id": observation_id,
                    "interface_changed": rebinding.interface_changed,
                }),
            )?;
            transaction.commit()?;
            record.status = crate::agency::ControllerStatus::Reviewed;
            record.revision = next_revision;
            record.reviewed_observation_id = Some(observation_id);
            record.updated_at = now;
            Ok(record)
        })
    }

    /// Revalidate a reviewed controller against an exact fresh observation.
    /// This is required on every run, including when its stored status is
    /// reviewed, and returns semantic IDs without issuing authority.
    pub fn revalidate_stored_controller(
        &self,
        controller_id: &str,
        observation_id: Uuid,
        now: DateTime<Utc>,
    ) -> CoreResult<crate::agency::RevalidatedStoredController> {
        validate_controller_id(controller_id)?;
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Controller storage is unavailable until Sage storage is unlocked".into(),
            ));
        }
        self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let record = read_stored_controller_tx(&transaction, controller_id)?
                .ok_or_else(|| CoreError::InvalidAction("Controller was not found".into()))?;
            if record.status != crate::agency::ControllerStatus::Reviewed {
                return Err("Controller is not reviewed and enabled".into());
            }
            let current_fingerprint: Option<String> = transaction
                .query_row(
                    "SELECT fingerprint FROM world_systems WHERE id=?1",
                    [record.controller.system_id.to_string()],
                    |row| row.get(0),
                )
                .optional()?;
            if current_fingerprint.as_deref() != Some(record.controller.system_fingerprint.as_str())
            {
                return Err("Controller target identity changed; relearning is required".into());
            }
            let current_evidence = controller_evidence_map(&transaction, &record.controller, now)?;
            let (observation, observation_expires_at) =
                checked_world_observation_tx(&transaction, observation_id)?.ok_or_else(|| {
                    CoreError::VerificationFailed(
                        "Controller revalidation observation is missing or expired".into(),
                    )
                })?;
            if observation_expires_at <= now {
                return Err("Controller revalidation observation has expired".into());
            }
            let rebinding = crate::agency::revalidate_controller(
                &record.controller,
                &observation,
                &current_evidence,
                now,
            )?;
            transaction.commit()?;
            Ok(crate::agency::RevalidatedStoredController { record, rebinding })
        })
    }

    pub fn disable_controller(
        &self,
        controller_id: &str,
        expected_revision: u64,
    ) -> CoreResult<crate::agency::StoredController> {
        validate_controller_id(controller_id)?;
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Controller storage is unavailable until Sage storage is unlocked".into(),
            ));
        }
        if expected_revision == 0 || expected_revision > i64::MAX as u64 {
            return Err(CoreError::InvalidAction(
                "Controller revision is invalid".into(),
            ));
        }
        let now = Utc::now();
        self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let mut record = read_stored_controller_tx(&transaction, controller_id)?
                .ok_or_else(|| CoreError::InvalidAction("Controller was not found".into()))?;
            if record.revision != expected_revision
                || !matches!(
                    record.status,
                    crate::agency::ControllerStatus::Draft
                        | crate::agency::ControllerStatus::Reviewed
                )
            {
                return Err("Controller changed or is already disabled".into());
            }
            let next_revision = expected_revision
                .checked_add(1)
                .filter(|revision| *revision <= i64::MAX as u64)
                .ok_or_else(|| CoreError::Storage("Controller revision is exhausted".into()))?;
            let changed = transaction.execute(
                "UPDATE world_controllers SET status='disabled',revision=?2,updated_at=?3 WHERE controller_id=?1 AND revision=?4 AND status IN ('draft','reviewed')",
                params![controller_id, next_revision, now.to_rfc3339(), expected_revision],
            )?;
            if changed != 1 {
                return Err("Controller revision changed before disable completed".into());
            }
            crate::storage::write_audit(
                &transaction,
                None,
                None,
                "controller_disabled",
                &serde_json::json!({
                    "controller_id": controller_id,
                    "system_id": record.controller.system_id,
                }),
            )?;
            transaction.commit()?;
            record.status = crate::agency::ControllerStatus::Disabled;
            record.revision = next_revision;
            record.updated_at = now;
            Ok(record)
        })
    }

    /// Persist task-owned procedure progress with optimistic revision checks.
    /// The procedure digest and runtime/task bindings are validated before the
    /// checkpoint becomes durable; persistence does not authorize execution.
    pub fn save_procedure_checkpoint(
        &self,
        checkpoint: &mut crate::agency::ProcedureCheckpoint,
    ) -> CoreResult<()> {
        let (next_revision, updated_at) = self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let task_json: Option<String> = transaction
                .query_row(
                    "SELECT task_json FROM tasks WHERE id=?1",
                    [checkpoint.task_id.to_string()],
                    |row| row.get(0),
                )
                .optional()?;
            let Some(task_json) = task_json else {
                return Err("Procedure checkpoint task does not exist".into());
            };
            let task: Task = serde_json::from_str(&task_json)?;
            let (next_revision, updated_at) =
                write_procedure_checkpoint_tx(&transaction, checkpoint, &task)?;
            transaction.commit()?;
            Ok((next_revision, updated_at))
        })?;
        checkpoint.revision = next_revision;
        checkpoint.updated_at = updated_at;
        Ok(())
    }

    /// Load and revalidate a procedure checkpoint. Malformed, oversized,
    /// cross-task, or procedure-mismatched state is never returned as runnable.
    pub fn load_procedure_checkpoint(
        &self,
        task_id: Uuid,
    ) -> CoreResult<Option<crate::agency::ProcedureCheckpoint>> {
        if task_id.is_nil() {
            return Err(CoreError::InvalidAction(
                "Procedure checkpoint requires a task identity".into(),
            ));
        }
        let stored = self.with_connection(|connection| {
            // Keep the size preflight and payload fetch in one SQLite read
            // snapshot. Another process sharing this database must not be
            // able to replace a bounded row between the two queries.
            let transaction = connection.transaction()?;
            let lengths: Option<(i64, i64)> = transaction
                .query_row(
                    "SELECT length(procedure_json),length(runtime_json) FROM procedure_checkpoints WHERE task_id=?1",
                    [task_id.to_string()],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            let Some((procedure_length, runtime_length)) = lengths else {
                transaction.commit()?;
                return Ok(None);
            };
            if procedure_length < 0
                || runtime_length < 0
                || procedure_length as usize > MAX_PROCEDURE_CHECKPOINT_BYTES
                || runtime_length as usize > MAX_PROCEDURE_CHECKPOINT_BYTES
            {
                return Err("Procedure checkpoint exceeds its serialized size limit".into());
            }
            let value = transaction.query_row(
                "SELECT revision,procedure_sha256,procedure_json,runtime_json,updated_at FROM procedure_checkpoints WHERE task_id=?1",
                [task_id.to_string()],
                |row| Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                )),
            )?;
            transaction.commit()?;
            Ok(Some(value))
        })?;
        let Some((revision, stored_digest, procedure_json, runtime_json, updated_at)) = stored
        else {
            return Ok(None);
        };
        let revision = u64::try_from(revision).map_err(|_| {
            CoreError::VerificationFailed("Procedure checkpoint revision is invalid".into())
        })?;
        if revision == 0 {
            return Err(CoreError::VerificationFailed(
                "Procedure checkpoint revision is invalid".into(),
            ));
        }
        let procedure: crate::agency::ProcedureIr = serde_json::from_str(&procedure_json)?;
        let runtime: crate::agency::ProcedureRuntimeState = serde_json::from_str(&runtime_json)?;
        procedure.validate()?;
        if crate::agency::procedure_digest(&procedure)? != stored_digest {
            return Err(CoreError::VerificationFailed(
                "Stored procedure checkpoint digest does not match its content".into(),
            ));
        }
        runtime.validate_checkpoint_for_task(&procedure, task_id)?;
        let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_at)
            .map_err(|_| CoreError::VerificationFailed("Checkpoint timestamp is invalid".into()))?
            .with_timezone(&Utc);
        let checkpoint = crate::agency::ProcedureCheckpoint {
            task_id,
            revision,
            procedure,
            runtime,
            updated_at,
        };
        self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let task_json: String = transaction.query_row(
                "SELECT task_json FROM tasks WHERE id=?1",
                [task_id.to_string()],
                |row| row.get(0),
            )?;
            let task: Task = serde_json::from_str(&task_json)?;
            validate_procedure_checkpoint_receipts(&transaction, &checkpoint, &task)?;
            transaction.commit()?;
            Ok(())
        })?;
        Ok(Some(checkpoint))
    }

    pub(crate) fn task_data_is_retired(&self, task_id: Uuid) -> CoreResult<bool> {
        self.with_connection(|connection| {
            Ok(connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=?1)",
                [task_id.to_string()],
                |row| row.get(0),
            )?)
        })
    }

    pub(crate) fn current_world_evidence_ids(
        &self,
        system_id: Uuid,
        fingerprint: &str,
    ) -> CoreResult<BTreeSet<Uuid>> {
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|connection| {
            let mut statement = connection.prepare(
                "SELECT o.id FROM world_observations o JOIN world_systems s ON s.id=o.system_id AND s.fingerprint=o.fingerprint WHERE o.system_id=?1 AND o.fingerprint=?2 AND o.expires_at>?3 ORDER BY o.observed_at DESC,o.id DESC LIMIT 4096",
            )?;
            let rows = statement.query_map(
                params![system_id.to_string(), fingerprint, Utc::now().to_rfc3339()],
                |row| row.get::<_, String>(0),
            )?;
            rows.map(|row| Ok(Uuid::parse_str(&row?)?)).collect()
        })
    }

    pub fn save_task(&self, task: &mut Task) -> CoreResult<()> {
        let revision = self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let revision = write_task(&transaction, task)?;
            transaction.commit()?;
            Ok(revision)
        })?;
        task.revision = revision;
        Ok(())
    }

    pub(crate) fn accept_task(
        &self,
        task: &mut Task,
        message: &crate::knowledge::Message,
        event: &CoreEvent,
        command: Option<&crate::commands::SubmissionKey>,
    ) -> CoreResult<()> {
        self.accept_run(task, message, event, command, None, None)
            .map(|_| ())
    }

    pub(crate) fn accept_task_with_procedure(
        &self,
        task: &mut Task,
        message: &crate::knowledge::Message,
        event: &CoreEvent,
        command: Option<&crate::commands::SubmissionKey>,
        checkpoint: &crate::agency::ProcedureCheckpoint,
    ) -> CoreResult<()> {
        self.accept_run(task, message, event, command, None, Some(checkpoint))
            .map(|_| ())
    }

    pub(crate) fn accept_continuation(
        &self,
        task: &mut Task,
        message: &crate::knowledge::Message,
        event: &CoreEvent,
        previous: &mut Task,
    ) -> CoreResult<CoreEvent> {
        self.accept_run(task, message, event, None, Some(previous), None)?
            .ok_or_else(|| CoreError::Storage("Continuation event is missing".into()))
    }

    fn accept_run(
        &self,
        task: &mut Task,
        message: &crate::knowledge::Message,
        event: &CoreEvent,
        command: Option<&crate::commands::SubmissionKey>,
        previous: Option<&mut Task>,
        initial_procedure: Option<&crate::agency::ProcedureCheckpoint>,
    ) -> CoreResult<Option<CoreEvent>> {
        if message.task_id != Some(task.id)
            || task.message_id != Some(message.id)
            || task.conversation_id != Some(message.conversation_id)
            || event.task_id != Some(task.id)
        {
            return Err(CoreError::InvalidAction(
                "Acceptance records do not identify the same task".into(),
            ));
        }
        let mut parent = previous.as_deref().cloned();
        let continued_event = if let Some(parent) = &mut parent {
            if parent.status != TaskStatus::Interrupted
                || parent.continued_by.is_some()
                || task.continuation_of != Some(parent.id)
                || task.control_scope() != parent.control_scope()
                || task.conversation_id != parent.conversation_id
            {
                return Err(CoreError::InvalidAction(
                    "Continuation no longer matches its parent task".into(),
                ));
            }
            parent.continued_by = Some(task.id);
            parent.final_outcome = Some("Continued in the next task.".into());
            parent.touch();
            Some(CoreEvent::new(
                Some(parent.id),
                crate::events::CoreEventKind::TaskStatusChanged {
                    status: parent.status,
                    summary: "Continued in the next task.".into(),
                },
            ))
        } else {
            if task.continuation_of.is_some() {
                return Err(CoreError::InvalidAction(
                    "Continuation requires its parent transition".into(),
                ));
            }
            None
        };
        let (revision,parent_revision) = self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            transaction.execute("INSERT OR IGNORE INTO control_scopes(id,stopped_at) VALUES(?1,NULL)", [task.control_scope().to_string()])?;
            let stopped: bool = transaction.query_row("SELECT stopped_at IS NOT NULL FROM control_scopes WHERE id=?1", [task.control_scope().to_string()], |row| row.get(0))?;
            if stopped { return Err(CoreError::Cancelled.into()); }
            let parent_revision = if let Some(parent) = &parent {
                let already_continued: bool = transaction.query_row("SELECT EXISTS(SELECT 1 FROM tasks WHERE json_extract(task_json,'$.continuation_of')=?1)", [parent.id.to_string()], |row| row.get(0))?;
                if already_continued { return Err(CoreError::InvalidAction("This task already has a continuation".into()).into()); }
                let revision = write_task(&transaction, parent)?;
                #[cfg(test)] continuation_checkpoint("parent");
                Some(revision)
            } else { None };
            let revision = write_task(&transaction, task)?;
            if let Some(checkpoint) = initial_procedure {
                if checkpoint.revision != 0
                    || checkpoint.task_id != task.id
                    || checkpoint.procedure.validate().is_err()
                    || checkpoint
                        .runtime
                        .validate_checkpoint_for_task(&checkpoint.procedure, task.id)
                        .is_err()
                {
                    return Err("Initial procedure checkpoint does not match its accepted task".into());
                }
                let procedure_json = serde_json::to_string(&checkpoint.procedure)?;
                let runtime_json = serde_json::to_string(&checkpoint.runtime)?;
                if procedure_json.len() > MAX_PROCEDURE_CHECKPOINT_BYTES
                    || runtime_json.len() > MAX_PROCEDURE_CHECKPOINT_BYTES
                {
                    return Err("Initial procedure checkpoint exceeds its serialized size limit".into());
                }
                validate_procedure_checkpoint_receipts(&transaction, checkpoint, task)?;
                transaction.execute(
                    "INSERT INTO procedure_checkpoints(task_id,revision,procedure_sha256,procedure_json,runtime_json,updated_at) VALUES(?1,1,?2,?3,?4,?5)",
                    params![
                        task.id.to_string(),
                        crate::agency::procedure_digest(&checkpoint.procedure)?,
                        procedure_json,
                        runtime_json,
                        checkpoint.updated_at.to_rfc3339(),
                    ],
                )?;
            }
            #[cfg(test)] if parent.is_some() { continuation_checkpoint("child"); }
            transaction.execute(
                "INSERT INTO messages(id,conversation_id,task_id,role,content,provenance_json,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7)",
                params![message.id.to_string(),message.conversation_id.to_string(),task.id.to_string(),message.role,
                    crate::redaction::redact_for_persistence(&message.content),serde_json::to_string(&message.provenance)?,message.created_at.to_rfc3339()])?;
            transaction.execute("INSERT INTO conversation_tasks(task_id,message_id,conversation_id) VALUES(?1,?2,?3)",
                params![task.id.to_string(),message.id.to_string(),message.conversation_id.to_string()])?;
            transaction.execute("UPDATE conversations SET updated_at=?2 WHERE id=?1",
                params![message.conversation_id.to_string(),task.updated_at.to_rfc3339()])?;
            transaction.execute("INSERT INTO events(id,task_id,kind,payload_json,occurred_at) VALUES(?1,?2,?3,?4,?5)",
                params![event.id.to_string(),task.id.to_string(),event_kind_name(event),serde_json::to_string(event)?,event.occurred_at.to_rfc3339()])?;
            if let Some(command) = command {
                // Keep the opaque receipt even if the user later removes task
                // history: forgetting history must not re-enable an old effect.
                transaction.execute("INSERT INTO command_inbox(principal,request_id,payload_digest,task_id,accepted_at) VALUES('local-user',?1,?2,?3,?4)",
                    params![command.id.to_string(),command.digest,task.id.to_string(),task.created_at.to_rfc3339()])?;
            }
            if let Some(continued) = &continued_event {
                write_event(&transaction, continued)?;
                write_audit(&transaction, task.continuation_of, None, "continuation_accepted", &serde_json::json!({"task_id":task.id,"control_scope_id":task.control_scope(),"event_id":event.id,"parent_event_id":continued.id}))?;
                #[cfg(test)] continuation_checkpoint("records");
            }
            transaction.commit()?;
            #[cfg(test)] if parent.is_some() { continuation_checkpoint("committed"); }
            Ok((revision,parent_revision))
        })?;
        task.revision = revision;
        if let (Some(target), Some(mut parent), Some(revision)) =
            (previous, parent, parent_revision)
        {
            parent.revision = revision;
            *target = parent;
        }
        Ok(continued_event)
    }

    pub(crate) fn stop_control_scope(&self, scope: Uuid) -> CoreResult<()> {
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let changed = tx.execute("INSERT INTO control_scopes(id,stopped_at) VALUES(?1,?2) ON CONFLICT(id) DO UPDATE SET stopped_at=excluded.stopped_at WHERE control_scopes.stopped_at IS NULL", params![scope.to_string(),Utc::now().to_rfc3339()])?;
            if changed == 1 {
                write_audit(&tx, Some(scope), None, "control_scope_stopped", &serde_json::json!({"control_scope_id":scope}))?;
            }
            tx.commit()?;
            Ok(())
        })
    }

    pub(crate) fn control_scope_stopped(&self, scope: Uuid) -> CoreResult<bool> {
        self.with_connection(|db| Ok(db.query_row("SELECT EXISTS(SELECT 1 FROM control_scopes WHERE id=?1 AND stopped_at IS NOT NULL)", [scope.to_string()], |row| row.get(0))?))
    }

    pub fn load_tasks(&self, include_completed: bool) -> CoreResult<Vec<Task>> {
        self.with_connection(|connection| {
            let sql = if include_completed {
                "SELECT task_json FROM tasks ORDER BY created_at DESC"
            } else {
                "SELECT task_json FROM tasks WHERE status NOT IN ('succeeded','answered','partial','failed','cancelled') ORDER BY created_at DESC"
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
        self.with_connection(|connection| {
            write_event(connection, event)?;
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
        self.with_connection(|connection| {
            Ok(write_audit(
                connection,
                task_id,
                action_id,
                event_type,
                redacted_payload,
            )?)
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
            let changed = connection.execute(
                r#"INSERT INTO rollback_plans(action_id, task_id, plan_json, expires_at, consumed_at)
                   VALUES (?1, ?2, ?3, ?4, NULL)
                   ON CONFLICT(action_id) DO UPDATE SET
                     plan_json=excluded.plan_json,
                     expires_at=excluded.expires_at
                   WHERE rollback_plans.task_id=excluded.task_id
                     AND rollback_plans.consumed_at IS NULL
                     AND NOT EXISTS(SELECT 1 FROM undo_journal WHERE action_id=excluded.action_id)"#,
                params![
                    plan.action_id.to_string(),
                    task_id.to_string(),
                    serde_json::to_string(plan)?,
                    plan.expires_at.to_rfc3339(),
                ],
            )?;
            if changed != 1 {
                return Err("Recovery metadata cannot replace an Undo already requested".into());
            }
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
                         AND NOT EXISTS(SELECT 1 FROM undo_journal WHERE action_id=rollback_plans.action_id)
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

    fn mark_incomplete_tasks_interrupted(&self) -> CoreResult<()> {
        self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let mut statement = transaction.prepare(
                "SELECT task_json FROM tasks WHERE status IN ('pending','planning','running','waiting_for_approval','waiting_for_user','paused','interrupted')",
            )?;
            let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
            let mut tasks = Vec::new();
            for row in rows {
                let mut task: Task = serde_json::from_str(&row?)?;
                let stopped: bool = transaction.query_row("SELECT EXISTS(SELECT 1 FROM control_scopes WHERE id=?1 AND stopped_at IS NOT NULL)", [task.control_scope().to_string()], |row| row.get(0))?;
                if stopped {
                    task.status = TaskStatus::Cancelled;
                    task.final_outcome = Some("Stopped at your request before the restart. Existing effects remain recorded.".into());
                } else {
                    if task.status == TaskStatus::Interrupted { continue; }
                    task.status = TaskStatus::Interrupted;
                    task.final_outcome = Some("SAGE Core restarted before this task completed; it was not resumed automatically.".into());
                }
                task.touch();
                tasks.push(task);
            }
            drop(statement);
            for task in tasks { write_task(&transaction, &task)?; }
            transaction.execute("UPDATE decisions SET state='interrupted',resolved_at=?1 WHERE state='pending'", [Utc::now().to_rfc3339()])?;
            transaction.execute("DELETE FROM settings WHERE key LIKE 'pending-approval.%'", [])?;
            transaction.commit()?;
            Ok(())
        })
    }

    pub(crate) fn with_connection<T>(
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
        TaskStatus::Answered => "answered",
        TaskStatus::Partial => "partial",
        TaskStatus::Failed => "failed",
        TaskStatus::Cancelled => "cancelled",
        TaskStatus::Interrupted => "interrupted",
    }
}

fn event_kind_name(event: &CoreEvent) -> &'static str {
    use crate::events::CoreEventKind;
    match &event.kind {
        CoreEventKind::TaskStarted => "task_started",
        CoreEventKind::ModelResponse { .. } => "model_response",
        CoreEventKind::PlanGenerated { .. } => "plan_generated",
        CoreEventKind::ActionProposed { .. } => "action_proposed",
        CoreEventKind::PolicyDenied { .. } => "policy_denied",
        CoreEventKind::ApprovalRequested { .. } => "approval_requested",
        CoreEventKind::QuestionRequested { .. } => "question_requested",
        CoreEventKind::ApprovalResolved { .. } => "approval_resolved",
        CoreEventKind::DecisionResolved { .. } => "decision_resolved",
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
        CoreEventKind::ReferenceContext { .. } => "reference_context",
        CoreEventKind::TaskCompleted { .. } => "task_completed",
        CoreEventKind::UndoChanged { .. } => "undo_changed",
        CoreEventKind::Error { .. } => "error",
    }
}

fn validate_controller_id(controller_id: &str) -> CoreResult<()> {
    if controller_id.len() != 64
        || !controller_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CoreError::InvalidAction(
            "Controller identifier must be a lowercase SHA-256 digest".into(),
        ));
    }
    Ok(())
}

fn read_stored_controller_tx(
    transaction: &Transaction<'_>,
    controller_id: &str,
) -> CoreResult<Option<crate::agency::StoredController>> {
    validate_controller_id(controller_id)?;
    let stored_length: Option<i64> = transaction
        .query_row(
            "SELECT length(controller_json) FROM world_controllers WHERE controller_id=?1",
            [controller_id],
            |row| row.get(0),
        )
        .optional()?;
    let Some(stored_length) = stored_length else {
        return Ok(None);
    };
    if stored_length < 0 || stored_length as usize > MAX_CONTROLLER_RECORD_BYTES {
        return Err(CoreError::VerificationFailed(
            "Stored controller exceeds its bounded size limit".into(),
        ));
    }
    let row: (String, String, String, String, i64, String, String, Option<String>, String, String) =
        transaction.query_row(
            "SELECT system_id,system_fingerprint,interface_fingerprint,status,revision,controller_json,payload_sha256,reviewed_observation_id,created_at,updated_at FROM world_controllers WHERE controller_id=?1",
            [controller_id],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                    row.get(9)?,
                ))
            },
        )?;
    let (
        system_id,
        system_fingerprint,
        interface_fingerprint,
        status,
        revision,
        controller_json,
        payload_sha256,
        reviewed_observation_id,
        created_at,
        updated_at,
    ) = row;
    if format!("{:x}", Sha256::digest(controller_json.as_bytes())) != payload_sha256 {
        return Err(CoreError::VerificationFailed(
            "Stored controller failed its content digest check".into(),
        ));
    }
    let controller: crate::agency::ControllerIr = serde_json::from_str(&controller_json)?;
    let actual_id = crate::agency::controller_digest(&controller)?;
    if actual_id != controller_id
        || controller.system_id.to_string() != system_id
        || controller.system_fingerprint != system_fingerprint
        || controller.interface_fingerprint != interface_fingerprint
        || revision <= 0
    {
        return Err(CoreError::VerificationFailed(
            "Stored controller identity or revision does not match its payload".into(),
        ));
    }
    Ok(Some(crate::agency::StoredController {
        id: controller_id.to_owned(),
        controller,
        status: crate::agency::ControllerStatus::from_storage_value(&status)?,
        revision: u64::try_from(revision).map_err(|_| {
            CoreError::VerificationFailed("Stored controller revision is invalid".into())
        })?,
        reviewed_observation_id: reviewed_observation_id
            .map(|value| {
                Uuid::parse_str(&value).map_err(|_| {
                    CoreError::VerificationFailed(
                        "Stored controller review observation ID is invalid".into(),
                    )
                })
            })
            .transpose()?,
        created_at: DateTime::parse_from_rfc3339(&created_at)
            .map_err(|_| {
                CoreError::VerificationFailed("Stored controller creation time is invalid".into())
            })?
            .to_utc(),
        updated_at: DateTime::parse_from_rfc3339(&updated_at)
            .map_err(|_| {
                CoreError::VerificationFailed("Stored controller update time is invalid".into())
            })?
            .to_utc(),
    }))
}

/// Read one bounded, digest-checked checkpoint inside an existing transaction.
/// Callers use this when a task transition must advance procedure state in the
/// same commit as its action receipt.
pub(crate) fn read_procedure_checkpoint_tx(
    transaction: &Transaction<'_>,
    task_id: Uuid,
) -> CoreResult<Option<crate::agency::ProcedureCheckpoint>> {
    let row: Option<(i64, String, String, String, String)> = transaction
        .query_row(
            "SELECT revision,procedure_sha256,procedure_json,runtime_json,updated_at FROM procedure_checkpoints WHERE task_id=?1 AND length(procedure_json)<=?2 AND length(runtime_json)<=?2",
            params![task_id.to_string(), MAX_PROCEDURE_CHECKPOINT_BYTES as i64],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        )
        .optional()?;
    let Some((revision, digest, procedure_json, runtime_json, updated_at)) = row else {
        let exists: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM procedure_checkpoints WHERE task_id=?1)",
            [task_id.to_string()],
            |row| row.get(0),
        )?;
        if exists {
            return Err(CoreError::VerificationFailed(
                "Procedure checkpoint exceeds its serialized size limit".into(),
            ));
        }
        return Ok(None);
    };
    let revision = u64::try_from(revision)
        .ok()
        .filter(|revision| *revision > 0)
        .ok_or_else(|| {
            CoreError::VerificationFailed("Stored procedure checkpoint revision is invalid".into())
        })?;
    let procedure: crate::agency::ProcedureIr = serde_json::from_str(&procedure_json)?;
    let runtime: crate::agency::ProcedureRuntimeState = serde_json::from_str(&runtime_json)?;
    procedure.validate()?;
    if crate::agency::procedure_digest(&procedure)? != digest {
        return Err(CoreError::VerificationFailed(
            "Stored procedure checkpoint digest does not match its content".into(),
        ));
    }
    runtime.validate_checkpoint_for_task(&procedure, task_id)?;
    let updated_at = DateTime::parse_from_rfc3339(&updated_at)
        .map_err(|_| CoreError::VerificationFailed("Checkpoint timestamp is invalid".into()))?
        .with_timezone(&Utc);
    let checkpoint = crate::agency::ProcedureCheckpoint {
        task_id,
        revision,
        procedure,
        runtime,
        updated_at,
    };
    let task_json: String = transaction.query_row(
        "SELECT task_json FROM tasks WHERE id=?1",
        [task_id.to_string()],
        |row| row.get(0),
    )?;
    let task: Task = serde_json::from_str(&task_json)?;
    validate_procedure_checkpoint_receipts(transaction, &checkpoint, &task)?;
    Ok(Some(checkpoint))
}

/// Persist a mutated checkpoint with optimistic revision checks inside the
/// caller's task/journal transaction. The returned revision and timestamp
/// should be copied into the in-memory checkpoint only after commit succeeds.
pub(crate) fn write_procedure_checkpoint_tx(
    transaction: &Transaction<'_>,
    checkpoint: &crate::agency::ProcedureCheckpoint,
    task: &Task,
) -> CoreResult<(u64, DateTime<Utc>)> {
    if checkpoint.revision >= i64::MAX as u64 {
        return Err(CoreError::Storage(
            "Procedure checkpoint revision is exhausted".into(),
        ));
    }
    checkpoint.procedure.validate()?;
    checkpoint
        .runtime
        .validate_checkpoint_for_task(&checkpoint.procedure, checkpoint.task_id)?;
    if checkpoint.task_id != task.id {
        return Err(CoreError::VerificationFailed(
            "Procedure checkpoint and task identities do not match".into(),
        ));
    }
    let procedure_json = serde_json::to_string(&checkpoint.procedure)?;
    let runtime_json = serde_json::to_string(&checkpoint.runtime)?;
    if procedure_json.len() > MAX_PROCEDURE_CHECKPOINT_BYTES
        || runtime_json.len() > MAX_PROCEDURE_CHECKPOINT_BYTES
    {
        return Err(CoreError::Storage(
            "Procedure checkpoint exceeds its serialized size limit".into(),
        ));
    }
    validate_procedure_checkpoint_receipts(transaction, checkpoint, task)?;
    let retention_revoked: bool = transaction.query_row(
        "SELECT EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=?1)",
        [checkpoint.task_id.to_string()],
        |row| row.get(0),
    )?;
    if retention_revoked {
        return Err(CoreError::Storage(
            "Procedure checkpoint retention was revoked".into(),
        ));
    }
    let expected_revision = checkpoint.revision as i64;
    let current_revision: Option<i64> = transaction
        .query_row(
            "SELECT revision FROM procedure_checkpoints WHERE task_id=?1",
            [checkpoint.task_id.to_string()],
            |row| row.get(0),
        )
        .optional()?;
    if current_revision.unwrap_or(0) != expected_revision {
        return Err(CoreError::Storage(
            "Procedure checkpoint revision changed".into(),
        ));
    }
    let next_revision = checkpoint.revision + 1;
    let updated_at = Utc::now();
    let changed = transaction.execute(
        "INSERT INTO procedure_checkpoints(task_id,revision,procedure_sha256,procedure_json,runtime_json,updated_at) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(task_id) DO UPDATE SET revision=excluded.revision,procedure_sha256=excluded.procedure_sha256,procedure_json=excluded.procedure_json,runtime_json=excluded.runtime_json,updated_at=excluded.updated_at WHERE procedure_checkpoints.revision=?7",
        params![
            checkpoint.task_id.to_string(),
            next_revision as i64,
            crate::agency::procedure_digest(&checkpoint.procedure)?,
            procedure_json,
            runtime_json,
            updated_at.to_rfc3339(),
            expected_revision
        ],
    )?;
    if changed != 1 {
        return Err(CoreError::Storage(
            "Procedure checkpoint revision changed".into(),
        ));
    }
    Ok((next_revision, updated_at))
}

fn read_current_procedure_descriptor_tx(
    transaction: &Transaction<'_>,
    system_id: Uuid,
    capability_id: &str,
    system_fingerprint: &str,
) -> CoreResult<crate::world_model::CapabilityDescriptor> {
    let descriptor_json: Option<String> = transaction
        .query_row(
            "SELECT descriptor_json FROM world_capabilities WHERE system_id=?1 AND capability_id=?2 AND fingerprint=?3",
            params![system_id.to_string(), capability_id, system_fingerprint],
            |row| row.get(0),
        )
        .optional()?;
    let descriptor_json = descriptor_json.ok_or_else(|| {
        CoreError::VerificationFailed(
            "Procedure action capability is missing from the current world model".into(),
        )
    })?;
    if descriptor_json.len() > 128 * 1024 {
        return Err(CoreError::VerificationFailed(
            "Procedure capability descriptor exceeds its size bound".into(),
        ));
    }
    let descriptor: crate::world_model::CapabilityDescriptor =
        serde_json::from_str(&descriptor_json)?;
    descriptor.validate()?;
    if descriptor.id != capability_id
        || descriptor.system_id != system_id
        || descriptor.system_fingerprint != system_fingerprint
    {
        return Err(CoreError::VerificationFailed(
            "Procedure capability does not match its current world identity".into(),
        ));
    }
    Ok(descriptor)
}

fn validate_procedure_checkpoint_receipts(
    transaction: &Transaction<'_>,
    checkpoint: &crate::agency::ProcedureCheckpoint,
    task: &Task,
) -> CoreResult<()> {
    let procedure = &checkpoint.procedure;
    let runtime = &checkpoint.runtime;
    runtime.validate_checkpoint_for_task(procedure, checkpoint.task_id)?;
    if task.id != checkpoint.task_id || runtime.task_id() != task.id {
        return Err(CoreError::VerificationFailed(
            "Procedure checkpoint and task identities do not match".into(),
        ));
    }

    let nodes = procedure
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    let mut descriptors_by_node = BTreeMap::<&str, crate::world_model::CapabilityDescriptor>::new();
    if !procedure.streams.is_empty() {
        let mut descriptors_by_id =
            BTreeMap::<String, crate::world_model::CapabilityDescriptor>::new();
        for node in &procedure.nodes {
            let crate::agency::ProcedureNodeKind::CapabilityCall {
                capability_id,
                system_id,
                system_fingerprint,
                ..
            } = &node.kind
            else {
                continue;
            };
            let descriptor = read_current_procedure_descriptor_tx(
                transaction,
                *system_id,
                capability_id,
                system_fingerprint,
            )?;
            if let Some(previous) = descriptors_by_id.get(&descriptor.id)
                && (previous.system_id != descriptor.system_id
                    || previous.system_fingerprint != descriptor.system_fingerprint)
            {
                return Err(CoreError::VerificationFailed(
                    "Procedure capability identities are ambiguous".into(),
                ));
            }
            descriptors_by_id.insert(descriptor.id.clone(), descriptor.clone());
            descriptors_by_node.insert(node.id.as_str(), descriptor);
        }
        procedure.validate_against(&descriptors_by_id.into_values().collect::<Vec<_>>())?;
    }
    for (node_id, action_id) in runtime.dispatch_action_ids() {
        let node = nodes.get(node_id.as_str()).ok_or_else(|| {
            CoreError::VerificationFailed("Procedure receipt names an unknown node".into())
        })?;
        let crate::agency::ProcedureNodeKind::CapabilityCall {
            capability_id,
            system_id,
            system_fingerprint,
            ..
        } = &node.kind
        else {
            return Err(CoreError::VerificationFailed(
                "Only capability calls can bind broker action receipts".into(),
            ));
        };
        let action = task.actions.get(action_id).ok_or_else(|| {
            CoreError::VerificationFailed(
                "Procedure dispatch receipt does not name a saved task action".into(),
            )
        })?;
        let descriptor = match descriptors_by_node.get(node_id.as_str()) {
            Some(descriptor) => descriptor.clone(),
            None => read_current_procedure_descriptor_tx(
                transaction,
                *system_id,
                capability_id,
                system_fingerprint,
            )?,
        };
        if descriptor.id != *capability_id
            || descriptor.system_id != *system_id
            || descriptor.system_fingerprint != *system_fingerprint
            || descriptor.executor_id.as_deref() != Some(action.proposal.action.kind())
        {
            return Err(CoreError::VerificationFailed(
                "Procedure capability does not resolve to the exact registered task executor"
                    .into(),
            ));
        }
        if matches!(
            &action.proposal.action,
            crate::domain::Action::SetApplicationControl { .. }
        ) && (descriptor.interface_control_id.is_none()
            || descriptor.interface_probe_kind.is_none()
            || descriptor.effects != BTreeSet::from([crate::contracts::Effect::ControlApplication])
            || descriptor.input_ports.len() != 1
            || descriptor.input_ports[0].name != "value"
            || descriptor.output_ports.len() != 1
            || descriptor.output_ports[0].name != "observed_value"
            || descriptor.input_ports[0].value_type != descriptor.output_ports[0].value_type)
        {
            return Err(CoreError::VerificationFailed(
                "Procedure control capability differs from its sealed reversible-control contract"
                    .into(),
            ));
        }
        let journal: Option<(String, String, String, Option<String>)> = transaction
            .query_row(
                "SELECT action_digest,state,prepared_json,verification_json FROM action_journal WHERE run_id=?1 AND action_id=?2",
                params![task.id.to_string(), action_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        let Some((journal_digest, journal_state, prepared_json, verification_json)) = journal
        else {
            return Err(CoreError::VerificationFailed(
                "Procedure dispatch receipt is absent from the durable action journal".into(),
            ));
        };
        if prepared_json.len() > 256 * 1024 {
            return Err(CoreError::VerificationFailed(
                "Procedure action journal entry exceeds its size bound".into(),
            ));
        }
        let prepared: crate::contracts::PreparedAction = serde_json::from_str(&prepared_json)?;
        let proposal = &action.proposal;
        let digest = crate::policy::approval_digest(proposal)?;
        if prepared.intent.proposal != *proposal
            || prepared.intent.proposal.id != *action_id
            || prepared.intent.proposal.task_id != task.id
            || prepared.action_digest != journal_digest
            || prepared.action_digest != digest
        {
            return Err(CoreError::VerificationFailed(
                "Procedure receipt does not match the exact prepared task action".into(),
            ));
        }
        let node_has_stream = procedure
            .streams
            .iter()
            .any(|stream| stream.producer_node == *node_id || stream.consumer_node == *node_id);
        match (
            node_has_stream,
            proposal
                .metadata
                .get("procedure_stream_node")
                .map(String::as_str),
        ) {
            (true, Some("true")) | (false, None) => {}
            _ => {
                return Err(CoreError::VerificationFailed(
                    "Procedure action stream marker differs from its declared channels".into(),
                ));
            }
        }
        let mut action_field_values = serde_json::to_value(&proposal.action)?
            .as_object()
            .cloned()
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Prepared action is not a structured operation".into(),
                )
            })?;
        let action_kind = action_field_values
            .remove("type")
            .and_then(|value| value.as_str().map(str::to_owned));
        let resolved_inputs = crate::agency::resolve_call_inputs(
            node,
            procedure,
            runtime,
            std::slice::from_ref(&descriptor),
        )
        .ok_or_else(|| {
            CoreError::VerificationFailed(
                "Procedure inputs cannot be resolved from typed verified dependencies".into(),
            )
        })?;
        if action_kind.as_deref() != Some(action.proposal.action.kind()) {
            return Err(CoreError::VerificationFailed(
                "Procedure inputs do not cover the exact registered task action".into(),
            ));
        }
        let input_shape_matches = match &action.proposal.action {
            crate::domain::Action::SetApplicationControl {
                application,
                system_id: action_system_id,
                system_fingerprint: action_fingerprint,
                capability_id: action_capability_id,
                control_id,
                ..
            } => {
                let system: Option<(String, String)> = transaction
                    .query_row(
                        "SELECT system_key,fingerprint FROM world_systems WHERE id=?1",
                        [action_system_id.to_string()],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .optional()?;
                let expected_fields = BTreeSet::from([
                    "application",
                    "system_id",
                    "system_fingerprint",
                    "capability_id",
                    "control_id",
                    "value",
                ]);
                let action_fields = action_field_values
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>();
                let input_names = resolved_inputs
                    .iter()
                    .filter_map(|(name, input)| {
                        (!matches!(input, crate::agency::ResolvedProcedureInput::Stream { .. }))
                            .then_some(name.as_str())
                    })
                    .collect::<BTreeSet<_>>();
                let system_matches = system.as_ref().is_some_and(|(key, fingerprint)| {
                    key == application
                        && fingerprint == action_fingerprint
                        && fingerprint == system_fingerprint
                        && action_system_id == system_id
                });
                action_fields == expected_fields
                    && input_names == BTreeSet::from(["value"])
                    && resolved_inputs.len() == 1
                    && system_matches
                    && action_capability_id == capability_id
                    && descriptor.interface_control_id.as_deref() == Some(control_id.as_str())
                    && action_field_values.get("application")
                        == Some(&serde_json::json!(application))
                    && action_field_values.get("system_id")
                        == Some(&serde_json::json!(action_system_id.to_string()))
                    && action_field_values.get("system_fingerprint")
                        == Some(&serde_json::json!(action_fingerprint))
                    && action_field_values.get("capability_id")
                        == Some(&serde_json::json!(action_capability_id))
                    && action_field_values.get("control_id") == Some(&serde_json::json!(control_id))
            }
            _ => {
                let input_names = resolved_inputs
                    .iter()
                    .filter_map(|(name, input)| {
                        (!matches!(input, crate::agency::ResolvedProcedureInput::Stream { .. }))
                            .then_some(name.as_str())
                    })
                    .collect::<BTreeSet<_>>();
                let action_names = action_field_values
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>();
                input_names == action_names
            }
        };
        if !input_shape_matches {
            return Err(CoreError::VerificationFailed(
                "Procedure inputs and sealed capability identity do not cover the exact task action".into(),
            ));
        }
        for (name, input) in resolved_inputs {
            let value = match input {
                crate::agency::ResolvedProcedureInput::Value { value, .. } => match value {
                    crate::agency::ProcedureValue::Text(value)
                    | crate::agency::ProcedureValue::Identifier(value) => {
                        serde_json::Value::String(value)
                    }
                    crate::agency::ProcedureValue::Boolean(value) => serde_json::Value::Bool(value),
                    crate::agency::ProcedureValue::Number(value) => {
                        serde_json::Number::from_f64(value)
                            .map(serde_json::Value::Number)
                            .ok_or_else(|| {
                                CoreError::VerificationFailed(
                                    "Procedure numeric input is not finite".into(),
                                )
                            })?
                    }
                },
                crate::agency::ResolvedProcedureInput::Stream { .. } => continue,
                crate::agency::ResolvedProcedureInput::Artifact { .. } => {
                    return Err(CoreError::VerificationFailed(
                        "Procedure artifact inputs cannot be bound to a task action yet".into(),
                    ));
                }
            };
            if action_field_values.get(&name) != Some(&value) {
                return Err(CoreError::VerificationFailed(
                    "Procedure input value differs from the exact prepared task action".into(),
                ));
            }
        }

        let state = runtime.node_states().get(node_id).ok_or_else(|| {
            CoreError::VerificationFailed("Procedure node state is missing".into())
        })?;
        let state_matches = match state {
            crate::agency::ProcedureNodeState::Running => {
                journal_state == "dispatched"
                    && matches!(
                        action.status,
                        crate::domain::ActionStatus::Running
                            | crate::domain::ActionStatus::Verifying
                    )
            }
            crate::agency::ProcedureNodeState::Succeeded => {
                journal_state == "confirmed"
                    && action.status == crate::domain::ActionStatus::Succeeded
            }
            crate::agency::ProcedureNodeState::Failed => {
                matches!(journal_state.as_str(), "failed" | "cancelled")
                    && action.status == crate::domain::ActionStatus::Failed
            }
            crate::agency::ProcedureNodeState::Uncertain => {
                journal_state == "uncertain"
                    && action.status == crate::domain::ActionStatus::Uncertain
            }
            crate::agency::ProcedureNodeState::Skipped => false,
        };
        if !state_matches {
            return Err(CoreError::VerificationFailed(
                "Procedure runtime state does not match its durable broker receipt".into(),
            ));
        }

        if *state != crate::agency::ProcedureNodeState::Succeeded {
            continue;
        }
        let verification_json = verification_json.ok_or_else(|| {
            CoreError::VerificationFailed(
                "Succeeded procedure action has no independent verification record".into(),
            )
        })?;
        if verification_json.len() > 256 * 1024 {
            return Err(CoreError::VerificationFailed(
                "Procedure verification record exceeds its size bound".into(),
            ));
        }
        let verification: crate::contracts::VerificationRecord =
            serde_json::from_str(&verification_json)?;
        if verification.run_id != task.id
            || verification.action_id != *action_id
            || verification.action_digest != digest
            || verification.target != proposal.target_resource
            || verification.expected != proposal.expected_outcome
            || verification.verdict != crate::contracts::Verdict::Confirmed
            || verification.evidence.is_empty()
        {
            return Err(CoreError::VerificationFailed(
                "Procedure action is not independently verified for its exact prepared target"
                    .into(),
            ));
        }
        let result = task
            .tool_results
            .iter()
            .rev()
            .find(|result| result.action_id == *action_id)
            .filter(|result| result.verdict == crate::contracts::Verdict::Confirmed)
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Procedure action has no confirmed durable task result".into(),
                )
            })?;
        let expected_outputs =
            runtime
                .verified_outputs_by_node()
                .get(node_id)
                .ok_or_else(|| {
                    CoreError::VerificationFailed("Procedure output record is missing".into())
                })?;
        let actual_outputs = result.output.get("procedure_outputs");
        let expected_outputs = serde_json::to_value(expected_outputs)?;
        let outputs_match = actual_outputs == Some(&expected_outputs)
            || (expected_outputs
                .as_object()
                .is_some_and(|outputs| outputs.is_empty())
                && actual_outputs.is_none());
        if !outputs_match {
            return Err(CoreError::VerificationFailed(
                "Procedure outputs do not match the broker-verified task result".into(),
            ));
        }
        for output in runtime.verified_outputs_by_node()[node_id].values() {
            if let crate::agency::ProcedureOutput::Artifact(artifact) = output {
                let artifact_row: Option<(String, String, i64)> = transaction
                    .query_row(
                        "SELECT task_id,sha256,length(content) FROM private_artifacts WHERE id=?1",
                        [artifact.artifact_id.to_string()],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .optional()?;
                if artifact.task_id != task.id
                    || artifact_row.is_none_or(|(owner, digest, size)| {
                        owner != task.id.to_string()
                            || digest != artifact.sha256
                            || u64::try_from(size).ok() != Some(artifact.size_bytes)
                    })
                {
                    return Err(CoreError::VerificationFailed(
                        "Procedure output artifact is not present under its verified task identity"
                            .into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_procedure_verification_observations(
    transaction: &Transaction<'_>,
    procedure: &crate::agency::ProcedureIr,
    runtime: &crate::agency::ProcedureRuntimeState,
    observations: &BTreeMap<Uuid, crate::world_model::ObservationEnvelope>,
    now: DateTime<Utc>,
) -> CoreResult<()> {
    let nodes = procedure
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    for (node_id, observation_id) in runtime.verification_evidence_by_node() {
        let node = nodes.get(node_id.as_str()).ok_or_else(|| {
            CoreError::VerificationFailed(
                "Verification evidence names an unknown procedure node".into(),
            )
        })?;
        let crate::agency::ProcedureNodeKind::CapabilityCall {
            system_id,
            system_fingerprint,
            ..
        } = &node.kind
        else {
            return Err(CoreError::VerificationFailed(
                "Controller verification evidence must belong to a capability call".into(),
            ));
        };
        let action_id = runtime.dispatch_action_ids().get(node_id).ok_or_else(|| {
            CoreError::VerificationFailed(
                "Controller verification is missing its broker action identity".into(),
            )
        })?;
        let (journal_state, verification_json): (String, Option<String>) = transaction.query_row(
            "SELECT state,verification_json FROM action_journal WHERE run_id=?1 AND action_id=?2",
            params![runtime.task_id().to_string(), action_id.to_string()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if journal_state != "confirmed" {
            return Err(CoreError::VerificationFailed(
                "Controller step lacks a confirmed durable broker action".into(),
            ));
        }
        let verification_json = verification_json.ok_or_else(|| {
            CoreError::VerificationFailed(
                "Controller step lacks its durable independent verification".into(),
            )
        })?;
        let verification: crate::contracts::VerificationRecord =
            serde_json::from_str(&verification_json)?;
        let observation = observations.get(observation_id).ok_or_else(|| {
            CoreError::VerificationFailed(
                "Controller verification observation is missing, expired, or unavailable".into(),
            )
        })?;
        if verification.run_id != runtime.task_id()
            || verification.action_id != *action_id
            || verification.verdict != crate::contracts::Verdict::Confirmed
            || verification.evidence.is_empty()
            || observation.id != *observation_id
            || observation.system_id != *system_id
            || observation.system_fingerprint != *system_fingerprint
            || observation.observed_at != verification.observed_at
            || observation.observed_at < now - chrono::Duration::seconds(10)
            || observation.observed_at > now + chrono::Duration::seconds(5)
            || !matches!(
                observation.origin,
                crate::world_model::EvidenceOrigin::OperatingSystem
                    | crate::world_model::EvidenceOrigin::Application
                    | crate::world_model::EvidenceOrigin::Browser
            )
        {
            return Err(CoreError::VerificationFailed(
                "Controller observation does not match the fresh durable verifier record".into(),
            ));
        }
    }
    Ok(())
}

fn checked_world_observation_tx(
    transaction: &Transaction<'_>,
    observation_id: Uuid,
) -> CoreResult<Option<(crate::world_model::ObservationEnvelope, DateTime<Utc>)>> {
    let row: Option<StoredWorldObservationRow> = transaction
        .query_row(
            "SELECT system_id,fingerprint,origin,privacy,session_id,observed_at,expires_at,payload_json,payload_sha256,id FROM world_observations WHERE id=?1",
            [observation_id.to_string()],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                    row.get(9)?,
                ))
            },
        )
        .optional()?;
    let Some((
        system_id,
        fingerprint,
        origin,
        privacy,
        session_id,
        observed_at,
        expires_at,
        payload,
        payload_sha256,
        stored_id,
    )) = row
    else {
        return Ok(None);
    };
    if payload.len() > crate::world_model::MAX_OBSERVATION_BYTES
        || format!("{:x}", Sha256::digest(payload.as_bytes())) != payload_sha256
    {
        return Err(CoreError::VerificationFailed(
            "Stored controller evidence failed its size or integrity check".into(),
        ));
    }
    let observation: crate::world_model::ObservationEnvelope = serde_json::from_str(&payload)?;
    let stored_origin: crate::world_model::EvidenceOrigin = serde_json::from_str(&origin)?;
    let stored_privacy: crate::contracts::Sensitivity = serde_json::from_str(&privacy)?;
    let stored_session = session_id
        .map(|value| {
            Uuid::parse_str(&value).map_err(|_| {
                CoreError::VerificationFailed(
                    "Stored controller evidence session ID is invalid".into(),
                )
            })
        })
        .transpose()?;
    let stored_observed_at = DateTime::parse_from_rfc3339(&observed_at)
        .map_err(|_| {
            CoreError::VerificationFailed("Stored controller evidence timestamp is invalid".into())
        })?
        .to_utc();
    let expires_at = DateTime::parse_from_rfc3339(&expires_at)
        .map_err(|_| {
            CoreError::VerificationFailed("Stored controller evidence expiry is invalid".into())
        })?
        .to_utc();
    observation.validate()?;
    if stored_id != observation_id.to_string()
        || observation.id != observation_id
        || observation.system_id.to_string() != system_id
        || observation.system_fingerprint != fingerprint
        || observation.origin != stored_origin
        || observation.privacy != stored_privacy
        || observation.session_id != stored_session
        || observation.observed_at != stored_observed_at
    {
        return Err(CoreError::VerificationFailed(
            "Stored controller evidence metadata does not match its payload".into(),
        ));
    }
    Ok(Some((observation, expires_at)))
}

fn controller_evidence_map(
    transaction: &Transaction<'_>,
    controller: &crate::agency::ControllerIr,
    now: DateTime<Utc>,
) -> CoreResult<BTreeMap<Uuid, crate::agency::CurrentControllerEvidence>> {
    let ids = controller
        .steps
        .iter()
        .flat_map(|step| step.evidence_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    if ids.is_empty() || ids.len() > 64 * 32 {
        return Err(CoreError::InvalidAction(
            "Controller evidence set is empty or exceeds its bound".into(),
        ));
    }
    let mut evidence = BTreeMap::new();
    for id in ids {
        let (observation, expires_at) =
            checked_world_observation_tx(transaction, id)?.ok_or_else(|| {
                CoreError::VerificationFailed("Controller supporting evidence is missing".into())
            })?;
        if observation.system_id != controller.system_id
            || observation.system_fingerprint != controller.system_fingerprint
            || observation.privacy >= crate::contracts::Sensitivity::Restricted
            || expires_at <= now
            || !matches!(
                observation.origin,
                crate::world_model::EvidenceOrigin::OperatingSystem
                    | crate::world_model::EvidenceOrigin::Application
                    | crate::world_model::EvidenceOrigin::Browser
            )
        {
            return Err(CoreError::VerificationFailed(
                "Controller evidence is stale, private, or belongs to another system".into(),
            ));
        }
        evidence.insert(
            id,
            crate::agency::CurrentControllerEvidence {
                system_id: observation.system_id,
                expires_at,
            },
        );
    }
    Ok(evidence)
}

/// Shared by ordinary updates and the atomic command-acceptance transaction.
pub(crate) fn write_task(connection: &rusqlite::Transaction<'_>, task: &Task) -> CoreResult<u64> {
    let prior: Option<u64> = connection
        .query_row(
            "SELECT COALESCE(json_extract(task_json,'$.revision'),0) FROM tasks WHERE id=?1",
            [task.id.to_string()],
            |row| row.get(0),
        )
        .optional()?;
    if prior.unwrap_or(0) != task.revision || (prior.is_none() && task.revision != 0) {
        return Err(CoreError::InvalidAction(
            "The task changed before this transition; reload its current state".into(),
        ));
    }
    let revision = task
        .revision
        .checked_add(1)
        .filter(|value| *value <= i64::MAX as u64)
        .ok_or_else(|| CoreError::Storage("Task revision exhausted".into()))?;
    let mut persisted = task.clone();
    persisted.revision = revision;
    let task_json = serde_json::to_string(&persisted)?;
    let status = status_name(task.status);
    connection.execute(
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
        connection.execute(
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
    if task.status.is_terminal() || task.status == TaskStatus::Interrupted {
        let state = if task.status == TaskStatus::Interrupted {
            "interrupted"
        } else {
            "cancelled"
        };
        connection.execute(
            "UPDATE decisions SET state=?2,resolved_at=?3 WHERE task_id=?1 AND state='pending'",
            params![task.id.to_string(), state, task.updated_at.to_rfc3339()],
        )?;
    }
    Ok(revision)
}

pub(crate) fn write_event(connection: &Connection, event: &CoreEvent) -> CoreResult<()> {
    connection.execute(
        "INSERT INTO events(id,task_id,kind,payload_json,occurred_at) VALUES(?1,?2,?3,?4,?5)",
        params![
            event.id.to_string(),
            event.task_id.map(|id| id.to_string()),
            event_kind_name(event),
            serde_json::to_string(event)?,
            event.occurred_at.to_rfc3339()
        ],
    )?;
    Ok(())
}

pub(crate) fn write_audit<T: Serialize>(
    connection: &Connection,
    task_id: Option<Uuid>,
    action_id: Option<Uuid>,
    event_type: &str,
    redacted_payload: &T,
) -> CoreResult<String> {
    let payload = serde_json::to_string(redacted_payload)?;
    let occurred_at = Utc::now().to_rfc3339();
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
}

#[cfg(test)]
fn continuation_checkpoint(stage: &str) {
    if std::env::var("SAGE_TEST_CONTINUATION_EXIT_AT")
        .ok()
        .as_deref()
        == Some(stage)
    {
        std::process::exit(88);
    }
}

#[cfg(test)]
mod continuation_tests {
    use super::*;
    use crate::events::CoreEventKind;
    use crate::knowledge::Message;
    use crate::secrets::SecretBytes;

    fn message(task: &Task) -> Message {
        Message {
            id: task.message_id.unwrap(),
            conversation_id: task.conversation_id.unwrap(),
            task_id: Some(task.id),
            role: "user".into(),
            content: task.request.clone(),
            provenance: crate::domain::Provenance::user(),
            created_at: task.created_at,
        }
    }
    fn fixture(path: &Path) -> (LocalStore, Task, Task, Message, CoreEvent) {
        let store = LocalStore::open_encrypted(path, &SecretBytes::new(vec![37; 32])).unwrap();
        store.migrate_knowledge().unwrap();
        let conversation = store
            .ensure_conversation(None, "Continuation acceptance")
            .unwrap();
        let mut parent = Task::new("Continuation acceptance");
        parent.status = TaskStatus::Interrupted;
        parent.budget_exhausted = true;
        parent.conversation_id = Some(conversation.id);
        parent.message_id = Some(Uuid::new_v4());
        let original_message = message(&parent);
        let original_event = CoreEvent::new(Some(parent.id), CoreEventKind::TaskStarted);
        store
            .accept_task(&mut parent, &original_message, &original_event, None)
            .unwrap();
        let mut child = Task::new(parent.request.clone());
        child.continuation_of = Some(parent.id);
        child.control_scope_id = Some(parent.control_scope());
        child.conversation_id = parent.conversation_id;
        child.message_id = Some(Uuid::new_v4());
        let next_message = message(&child);
        let next_event = CoreEvent::new(Some(child.id), CoreEventKind::TaskStarted);
        (store, parent, child, next_message, next_event)
    }
    fn count(store: &LocalStore, table: &str) -> i64 {
        store
            .with_connection(|db| {
                Ok(
                    db.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                        row.get(0)
                    })?,
                )
            })
            .unwrap()
    }

    #[test]
    fn continuation_acceptance_rolls_back_parent_child_and_messages_together() {
        for table in ["messages", "events", "audit_log"] {
            let dir = tempfile::tempdir().unwrap();
            let (store, mut parent, mut child, message, event) =
                fixture(&dir.path().join("continuation.db"));
            let original = parent.clone();
            store.with_connection(|db| { db.execute_batch(&format!("CREATE TEMP TRIGGER fail_continuation BEFORE INSERT ON {table} BEGIN SELECT RAISE(ABORT,'fixture acceptance failure'); END;"))?; Ok(()) }).unwrap();
            assert!(
                store
                    .accept_continuation(&mut child, &message, &event, &mut parent)
                    .is_err()
            );
            assert_eq!(parent, original);
            assert_eq!(child.revision, 0);
            assert_eq!(store.load_tasks(true).unwrap(), vec![original.clone()]);
            assert_eq!(count(&store, "messages"), 1);
            assert_eq!(count(&store, "events"), 1);
            assert_eq!(count(&store, "audit_log"), 0);
            store
                .with_connection(|db| {
                    db.execute_batch("DROP TRIGGER fail_continuation;")?;
                    Ok(())
                })
                .unwrap();
            store
                .accept_continuation(&mut child, &message, &event, &mut parent)
                .unwrap();
            assert_eq!(parent.continued_by, Some(child.id));
            assert_eq!(parent.revision, original.revision + 1);
            assert_eq!(child.revision, 1);
            // Even a stale retry of the original source cannot fork another child.
            let mut retry = child.clone();
            retry.id = Uuid::new_v4();
            retry.message_id = Some(Uuid::new_v4());
            retry.revision = 0;
            let retry_message = super::continuation_tests::message(&retry);
            let retry_event = CoreEvent::new(Some(retry.id), CoreEventKind::TaskStarted);
            assert!(
                store
                    .accept_continuation(
                        &mut retry,
                        &retry_message,
                        &retry_event,
                        &mut original.clone()
                    )
                    .is_err()
            );
            assert_eq!(count(&store, "tasks"), 2);
        }
    }

    #[test]
    fn durable_scope_stop_prevents_acceptance_even_with_an_old_parent_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mut parent, mut child, message, event) =
            fixture(&dir.path().join("continuation.db"));
        store.stop_control_scope(parent.id).unwrap();
        store.stop_control_scope(parent.id).unwrap();
        assert_eq!(
            count(&store, "audit_log"),
            1,
            "Stop is idempotent within its control scope"
        );
        assert!(
            store
                .accept_continuation(&mut child, &message, &event, &mut parent)
                .is_err()
        );
        assert!(parent.continued_by.is_none());
        assert_eq!(count(&store, "tasks"), 1);
        assert_eq!(count(&store, "messages"), 1);
    }

    #[test]
    fn continuation_exit_child() {
        let Ok(path) = std::env::var("SAGE_TEST_CONTINUATION_DB") else {
            return;
        };
        let (store, mut parent, mut child, message, event) = fixture(Path::new(&path));
        store
            .accept_continuation(&mut child, &message, &event, &mut parent)
            .unwrap();
        panic!("Expected abrupt continuation exit");
    }

    #[test]
    fn abrupt_exit_never_leaves_an_unlinked_or_duplicate_continuation() {
        for stage in ["parent", "child", "records", "committed"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("continuation.db");
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "storage::continuation_tests::continuation_exit_child",
                    "--nocapture",
                ])
                .env("SAGE_TEST_CONTINUATION_DB", &path)
                .env("SAGE_TEST_CONTINUATION_EXIT_AT", stage)
                .output()
                .unwrap();
            assert_eq!(
                child.status.code(),
                Some(88),
                "{stage}: {}",
                String::from_utf8_lossy(&child.stderr)
            );
            let store = LocalStore::open_encrypted(&path, &SecretBytes::new(vec![37; 32])).unwrap();
            let tasks = store.load_tasks(true).unwrap();
            let committed = stage == "committed";
            assert_eq!(tasks.len(), if committed { 2 } else { 1 });
            let parent = tasks
                .iter()
                .find(|task| task.continuation_of.is_none())
                .unwrap();
            assert_eq!(parent.continued_by.is_some(), committed);
            assert_eq!(count(&store, "messages"), if committed { 2 } else { 1 });
            assert_eq!(count(&store, "events"), if committed { 3 } else { 1 });
            assert_eq!(count(&store, "audit_log"), i64::from(committed));
            if committed {
                let continuation = tasks
                    .iter()
                    .find(|task| task.continuation_of == Some(parent.id))
                    .unwrap();
                assert_eq!(parent.continued_by, Some(continuation.id));
                assert_eq!(continuation.control_scope(), parent.id);
                assert_eq!(
                    continuation.status,
                    TaskStatus::Interrupted,
                    "Restart never automatically executes the accepted continuation"
                );
            }
        }
    }
}

#[cfg(test)]
mod task_scoped_artifact_tests {
    use super::*;

    fn store() -> LocalStore {
        LocalStore::deferred(Path::new(":memory:")).unwrap()
    }

    #[test]
    fn scoped_artifact_read_checks_task_owner_and_bound_digest() {
        let store = store();
        let owner = Uuid::new_v4();
        let other_task = Uuid::new_v4();
        let bytes = b"verified procedure output";
        let artifact = store.save_artifact(owner, bytes).unwrap();
        let digest = format!("{:x}", Sha256::digest(bytes));

        assert_eq!(
            store
                .read_artifact_for_task(artifact, owner, &digest)
                .unwrap(),
            bytes
        );
        assert!(
            store
                .read_artifact_for_task(artifact, other_task, &digest)
                .is_err()
        );
        assert!(
            store
                .read_artifact_for_task(artifact, owner, &"0".repeat(64))
                .is_err()
        );
        assert!(
            store
                .read_artifact_for_task(artifact, owner, "not-a-digest")
                .is_err()
        );
        store
            .with_connection(|db| {
                db.execute(
                    "INSERT INTO retired_task_data(task_id,retired_at) VALUES(?1,?2)",
                    params![owner.to_string(), Utc::now().to_rfc3339()],
                )?;
                Ok(())
            })
            .unwrap();
        let retained_artifacts: i64 = store
            .with_connection(|db| {
                Ok(db.query_row(
                    "SELECT COUNT(*) FROM private_artifacts WHERE task_id=?1",
                    [owner.to_string()],
                    |row| row.get(0),
                )?)
            })
            .unwrap();
        assert_eq!(retained_artifacts, 0);
        assert!(
            store
                .read_artifact_for_task(artifact, owner, &digest)
                .is_err()
        );
    }

    #[test]
    fn scoped_artifact_read_rejects_expired_or_tampered_content() {
        let store = store();
        let owner = Uuid::new_v4();
        let expired_bytes = b"expired artifact";
        let expired = store.save_artifact(owner, expired_bytes).unwrap();
        let expired_digest = format!("{:x}", Sha256::digest(expired_bytes));
        store
            .with_connection(|db| {
                db.execute(
                    "UPDATE private_artifacts SET expires_at=?1 WHERE id=?2",
                    params![
                        (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339(),
                        expired.to_string()
                    ],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(
            store
                .read_artifact_for_task(expired, owner, &expired_digest)
                .is_err()
        );

        let tampered_bytes = b"before tampering";
        let tampered = store.save_artifact(owner, tampered_bytes).unwrap();
        let original_digest = format!("{:x}", Sha256::digest(tampered_bytes));
        store
            .with_connection(|db| {
                db.execute(
                    "UPDATE private_artifacts SET content=?1 WHERE id=?2",
                    params![b"after tampering".as_slice(), tampered.to_string()],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(
            store
                .read_artifact_for_task(tampered, owner, &original_digest)
                .is_err()
        );
    }
}

#[cfg(test)]
mod procedure_checkpoint_tests {
    use super::*;
    use crate::agency::{
        CompletionCondition, ProcedureCheckpoint, ProcedureIr, ProcedureNode, ProcedureNodeKind,
        ProcedureRuntimeState,
    };
    use crate::domain::Provenance;
    use crate::events::CoreEventKind;
    use crate::knowledge::Message;
    use crate::secrets::SecretBytes;
    use crate::world_model::{
        CapabilityAssessment, CapabilityDescriptor, CapabilityEvidenceState, DataPort, PortType,
        Preconditions,
    };
    use std::collections::{BTreeMap, BTreeSet};

    fn store() -> LocalStore {
        LocalStore::deferred(Path::new(":memory:")).unwrap()
    }

    fn encrypted_store(path: &Path) -> LocalStore {
        LocalStore::open_encrypted(path, &SecretBytes::new(vec![37; 32])).unwrap()
    }

    fn procedure() -> ProcedureIr {
        ProcedureIr {
            schema_version: 1,
            id: "checkpoint-test".into(),
            nodes: vec![ProcedureNode {
                id: "read".into(),
                depends_on: BTreeSet::new(),
                outputs: BTreeMap::from([(
                    "bytes".into(),
                    DataPort {
                        name: "bytes".into(),
                        value_type: PortType::Bytes,
                        max_bytes: 4096,
                        privacy: crate::contracts::Sensitivity::Private,
                    },
                )]),
                kind: ProcedureNodeKind::CapabilityCall {
                    capability_id: "test.read".into(),
                    system_id: Uuid::new_v4(),
                    system_fingerprint: "a".repeat(64),
                    input_bindings: BTreeMap::new(),
                },
            }],
            streams: Vec::new(),
            completion: vec![CompletionCondition::AllNodesSucceeded],
        }
    }

    fn checkpoint(task_id: Uuid, procedure: ProcedureIr) -> ProcedureCheckpoint {
        ProcedureCheckpoint {
            task_id,
            revision: 0,
            runtime: ProcedureRuntimeState::new_for_task(&procedure, task_id).unwrap(),
            procedure,
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn procedure_task_and_initial_checkpoint_are_accepted_atomically() {
        let store = store();
        store.migrate_knowledge().unwrap();
        let conversation = store.ensure_conversation(None, "procedure task").unwrap();
        let mut task = Task::new("Execute a grounded control procedure");
        task.conversation_id = Some(conversation.id);
        task.message_id = Some(Uuid::new_v4());
        let message = Message {
            id: task.message_id.unwrap(),
            conversation_id: conversation.id,
            task_id: Some(task.id),
            role: "user".into(),
            content: task.request.clone(),
            provenance: Provenance::user(),
            created_at: task.created_at,
        };
        let started = CoreEvent::new(Some(task.id), CoreEventKind::TaskStarted);
        let initial = checkpoint(task.id, procedure());
        store
            .accept_task_with_procedure(&mut task, &message, &started, None, &initial)
            .unwrap();
        let loaded = store.load_procedure_checkpoint(task.id).unwrap().unwrap();
        assert_eq!(loaded.revision, 1);
        assert_eq!(loaded.procedure.id, "checkpoint-test");
        assert_eq!(loaded.runtime.task_id(), task.id);
        assert!(loaded.runtime.node_states().is_empty());
    }

    #[test]
    fn failed_initial_procedure_checkpoint_insert_rolls_back_task_acceptance() {
        let store = store();
        store.migrate_knowledge().unwrap();
        let conversation = store
            .ensure_conversation(None, "procedure task rollback")
            .unwrap();
        store
            .with_connection(|db| {
                db.execute_batch("CREATE TRIGGER reject_initial_procedure BEFORE INSERT ON procedure_checkpoints BEGIN SELECT RAISE(ABORT,'injected checkpoint failure'); END;")?;
                Ok(())
            })
            .unwrap();
        let mut task = Task::new("Execute a grounded control procedure");
        task.conversation_id = Some(conversation.id);
        task.message_id = Some(Uuid::new_v4());
        let message = Message {
            id: task.message_id.unwrap(),
            conversation_id: conversation.id,
            task_id: Some(task.id),
            role: "user".into(),
            content: task.request.clone(),
            provenance: Provenance::user(),
            created_at: task.created_at,
        };
        let started = CoreEvent::new(Some(task.id), CoreEventKind::TaskStarted);
        let initial = checkpoint(task.id, procedure());
        assert!(
            store
                .accept_task_with_procedure(&mut task, &message, &started, None, &initial)
                .is_err()
        );
        store
            .with_connection(|db| {
                let task_count: i64 =
                    db.query_row("SELECT COUNT(*) FROM tasks", [], |row| row.get(0))?;
                let message_count: i64 =
                    db.query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;
                assert_eq!(task_count, 0);
                assert_eq!(message_count, 0);
                Ok(())
            })
            .unwrap();
    }

    fn assessment(procedure: &ProcedureIr, evidence_id: Uuid) -> CapabilityAssessment {
        let node = &procedure.nodes[0];
        let ProcedureNodeKind::CapabilityCall {
            capability_id,
            system_id,
            system_fingerprint,
            ..
        } = &node.kind
        else {
            unreachable!("fixture is a capability call")
        };
        let executor_id = crate::features::manifests()
            .into_iter()
            .find(|manifest| manifest.enabled)
            .unwrap()
            .id;
        CapabilityAssessment {
            descriptor: CapabilityDescriptor {
                schema_version: 1,
                id: capability_id.clone(),
                system_id: *system_id,
                system_fingerprint: system_fingerprint.clone(),
                interface_control_id: None,
                interface_probe_kind: None,
                label: "Read test artifact".into(),
                input_ports: Vec::new(),
                output_ports: node.outputs.values().cloned().collect(),
                preconditions: Preconditions {
                    observed_state_fact_ids: vec![evidence_id],
                    description: "Current fixture state".into(),
                },
                effects: BTreeSet::from([crate::contracts::Effect::Read]),
                verification: "Independent artifact digest".into(),
                restoration: None,
                cancellation: "Stop dispatch and settle the read".into(),
                executor_id: Some(executor_id),
                evidence_ids: vec![evidence_id],
                updated_at: Utc::now(),
            },
            evidence_state: CapabilityEvidenceState::ReversiblyExperimented,
        }
    }

    #[test]
    fn procedure_checkpoint_round_trips_with_task_and_revision_fencing() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("procedure-checkpoint.db");
        let store = encrypted_store(&path);
        let mut task = Task::new("persist a reviewed procedure");
        store.save_task(&mut task).unwrap();

        let mut forged = checkpoint(task.id, procedure());
        let evidence_id = Uuid::new_v4();
        let capability = assessment(&forged.procedure, evidence_id);
        let capabilities = [capability];
        let current_evidence = BTreeSet::from([evidence_id]);
        forged
            .runtime
            .record_dispatched(
                &forged.procedure,
                &capabilities,
                &current_evidence,
                "read",
                Uuid::new_v4(),
            )
            .unwrap();
        forged
            .runtime
            .record_verified_success(&forged.procedure, "read", BTreeMap::new(), Uuid::new_v4())
            .unwrap();
        assert!(
            store.save_procedure_checkpoint(&mut forged).is_err(),
            "caller-supplied procedure receipts must resolve to broker journal entries"
        );

        assert!(store.load_procedure_checkpoint(task.id).unwrap().is_none());
        let mut checkpoint = checkpoint(task.id, forged.procedure.clone());
        store.save_procedure_checkpoint(&mut checkpoint).unwrap();
        assert_eq!(checkpoint.revision, 1);
        drop(store);
        let store = encrypted_store(&path);
        let loaded = store.load_procedure_checkpoint(task.id).unwrap().unwrap();
        assert_eq!(loaded.task_id, task.id);
        assert_eq!(loaded.revision, 1);
        assert_eq!(loaded.procedure.id, checkpoint.procedure.id);
        assert_eq!(
            serde_json::to_value(loaded.runtime).unwrap(),
            serde_json::to_value(checkpoint.runtime.clone()).unwrap()
        );
        assert!(store.compile_controller_draft(task.id).is_err());

        let mut stale = checkpoint.clone();
        stale.revision = 0;
        assert!(store.save_procedure_checkpoint(&mut stale).is_err());
        assert_eq!(
            store
                .load_procedure_checkpoint(task.id)
                .unwrap()
                .unwrap()
                .revision,
            1
        );
        store
            .with_connection(|db| {
                db.execute(
                    "INSERT INTO retired_task_data(task_id,retired_at) VALUES(?1,?2)",
                    params![task.id.to_string(), Utc::now().to_rfc3339()],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(store.load_procedure_checkpoint(task.id).unwrap().is_none());
        assert!(store.save_procedure_checkpoint(&mut checkpoint).is_err());
        let retained_artifacts: i64 = store
            .with_connection(|db| {
                Ok(db.query_row(
                    "SELECT COUNT(*) FROM private_artifacts WHERE task_id=?1",
                    [task.id.to_string()],
                    |row| row.get(0),
                )?)
            })
            .unwrap();
        assert_eq!(retained_artifacts, 0);
    }

    #[test]
    fn procedure_checkpoint_accepts_a_success_only_after_broker_verification() {
        use crate::contracts::PreparedAction;
        use crate::domain::{
            Action, ActionProposal, ActionState, ActionStatus, ExpectedOutcome, Provenance,
        };
        use crate::execution::ExecutionReceipt;
        use crate::observation::{Evidence, Observation};

        let directory = tempfile::tempdir().unwrap();
        let store = encrypted_store(&directory.path().join("verified-procedure.db"));
        store.migrate_world_model().unwrap();
        let mut task = Task::new("verify a procedure step");
        task.status = TaskStatus::Running;
        let question = "Confirm the fixture result";
        let proposal = ActionProposal {
            id: Uuid::new_v4(),
            task_id: task.id,
            action: Action::AskUser {
                question: question.into(),
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: "user".into(),
            provenance: Provenance::user(),
            metadata: BTreeMap::new(),
        };
        task.actions.insert(
            proposal.id,
            ActionState {
                proposal: proposal.clone(),
                status: ActionStatus::Pending,
                attempts: 0,
                summary: None,
                error: None,
            },
        );
        store.save_task(&mut task).unwrap();
        let prepared = PreparedAction::new(&proposal, BTreeSet::new()).unwrap();
        let (task, _) = store.commit_prepared_action(task, &prepared).unwrap();
        let (task, _) = store
            .commit_dispatched_action(task, &prepared, None, "fixture")
            .unwrap();
        let receipt = ExecutionReceipt {
            executor: "fixture".into(),
            summary: "User confirmed the fixture result".into(),
            transient_data: serde_json::json!({"user_answered": true}),
            rollback: None,
        };
        let observation = Observation {
            observed_at: Utc::now(),
            provenance: Provenance::external(
                crate::domain::ProvenanceSource::OperatingSystem,
                "fixture",
            ),
            summary: "The user confirmed the fixture result".into(),
            evidence: vec![Evidence::UserAnswer { received: true }],
        };
        let change = crate::transitions::VerifiedAction::from_observation(
            &task,
            &proposal,
            &receipt,
            &observation,
            crate::transitions::VerificationMode::Execution,
        )
        .unwrap();
        let (task, _) = store.commit_verified_action(change).unwrap();

        let mut procedure = procedure();
        procedure.nodes[0].outputs.clear();
        let (system_id, system_fingerprint) = match &procedure.nodes[0].kind {
            ProcedureNodeKind::CapabilityCall {
                system_id,
                system_fingerprint,
                ..
            } => (*system_id, system_fingerprint.clone()),
            _ => unreachable!(),
        };
        let system = store
            .observe_system(crate::world_model::SystemDescriptor {
                id: system_id,
                kind: crate::world_model::SystemKind::Application,
                key: "application:fixture".into(),
                label: "Fixture application".into(),
                fingerprint: system_fingerprint.clone(),
                revision: 0,
                updated_at: Utc::now(),
            })
            .unwrap();
        let evidence_id = Uuid::new_v4();
        let verification_observation = crate::world_model::ObservationEnvelope {
            id: evidence_id,
            system_id: system.id,
            session_id: None,
            worker_session: None,
            system_fingerprint,
            origin: crate::world_model::EvidenceOrigin::Application,
            privacy: crate::contracts::Sensitivity::Private,
            observed_at: observation.observed_at,
            facts: vec![crate::world_model::ObservedFact {
                name: "application.interface_fingerprint".into(),
                subject: None,
                value: crate::world_model::FactValue::Identifier("b".repeat(64)),
            }],
        };
        store
            .record_world_observation(&verification_observation)
            .unwrap();
        let input_port = DataPort {
            name: "question".into(),
            value_type: PortType::Text,
            max_bytes: 512,
            privacy: crate::contracts::Sensitivity::Private,
        };
        if let ProcedureNodeKind::CapabilityCall { input_bindings, .. } =
            &mut procedure.nodes[0].kind
        {
            input_bindings.insert(
                "question".into(),
                crate::agency::ValueBinding::Literal {
                    value: crate::agency::ProcedureValue::Text(question.into()),
                    port: input_port.clone(),
                },
            );
        }
        let mut capability = assessment(&procedure, evidence_id);
        capability.descriptor.input_ports = vec![input_port];
        capability.descriptor.executor_id = Some("ask_user".into());
        store
            .record_capability_candidate(&capability.descriptor)
            .unwrap();
        let mut checkpoint = checkpoint(task.id, procedure);
        checkpoint
            .runtime
            .record_dispatched(
                &checkpoint.procedure,
                &[capability],
                &BTreeSet::from([evidence_id]),
                "read",
                proposal.id,
            )
            .unwrap();
        checkpoint
            .runtime
            .record_verified_success(&checkpoint.procedure, "read", BTreeMap::new(), evidence_id)
            .unwrap();

        store.save_procedure_checkpoint(&mut checkpoint).unwrap();
        let loaded = store.load_procedure_checkpoint(task.id).unwrap().unwrap();
        assert_eq!(loaded.revision, 1);
        assert_eq!(
            loaded.runtime.node_states()["read"],
            crate::agency::ProcedureNodeState::Succeeded
        );
        let mut mismatched_input = loaded.clone();
        if let ProcedureNodeKind::CapabilityCall { input_bindings, .. } =
            &mut mismatched_input.procedure.nodes[0].kind
        {
            input_bindings.insert(
                "question".into(),
                crate::agency::ValueBinding::Literal {
                    value: crate::agency::ProcedureValue::Text("A different question".into()),
                    port: DataPort {
                        name: "question".into(),
                        value_type: PortType::Text,
                        max_bytes: 512,
                        privacy: crate::contracts::Sensitivity::Private,
                    },
                },
            );
        }
        mismatched_input.runtime =
            ProcedureRuntimeState::new_for_task(&mismatched_input.procedure, task.id).unwrap();
        let mut stored_capability = assessment(&mismatched_input.procedure, evidence_id);
        stored_capability.descriptor.input_ports = vec![DataPort {
            name: "question".into(),
            value_type: PortType::Text,
            max_bytes: 512,
            privacy: crate::contracts::Sensitivity::Private,
        }];
        stored_capability.descriptor.executor_id = Some("ask_user".into());
        mismatched_input
            .runtime
            .record_dispatched(
                &mismatched_input.procedure,
                &[stored_capability],
                &BTreeSet::from([evidence_id]),
                "read",
                proposal.id,
            )
            .unwrap();
        mismatched_input
            .runtime
            .record_verified_success(
                &mismatched_input.procedure,
                "read",
                BTreeMap::new(),
                evidence_id,
            )
            .unwrap();
        assert!(
            store
                .save_procedure_checkpoint(&mut mismatched_input)
                .is_err(),
            "procedure node inputs must equal the inputs in the prepared action"
        );
        assert!(
            store.compile_controller_draft(task.id).is_err(),
            "a verified action cannot create a controller without an experimentally bound UI capability"
        );

        store
            .with_connection(|connection| {
                let transaction = connection.transaction()?;
                let (stored_observation, expires_at) =
                    checked_world_observation_tx(&transaction, evidence_id)?.unwrap();
                assert!(expires_at > Utc::now());
                let observations = BTreeMap::from([(evidence_id, stored_observation.clone())]);
                validate_procedure_verification_observations(
                    &transaction,
                    &loaded.procedure,
                    &loaded.runtime,
                    &observations,
                    Utc::now(),
                )?;
                let mut mismatched = stored_observation;
                mismatched.observed_at += chrono::Duration::milliseconds(1);
                assert!(
                    validate_procedure_verification_observations(
                        &transaction,
                        &loaded.procedure,
                        &loaded.runtime,
                        &BTreeMap::from([(evidence_id, mismatched)]),
                        Utc::now(),
                    )
                    .is_err()
                );
                transaction.commit()?;
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn procedure_checkpoint_load_rejects_tampered_procedure_content() {
        let store = store();
        let mut task = Task::new("persist a reviewed procedure");
        store.save_task(&mut task).unwrap();
        let mut checkpoint = checkpoint(task.id, procedure());
        store.save_procedure_checkpoint(&mut checkpoint).unwrap();
        store
            .with_connection(|db| {
                db.execute(
                    "UPDATE procedure_checkpoints SET procedure_json=replace(procedure_json,?1,?2) WHERE task_id=?3",
                    params!["checkpoint-test", "modified-test", task.id.to_string()],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(store.load_procedure_checkpoint(task.id).is_err());
    }
}
