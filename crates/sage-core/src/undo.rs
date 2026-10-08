//! Compensation is a separate effect with its own durable dispatch boundary.
//! After dispatch, recovery observes only. It never repeats the inverse effect.
use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::contracts::FileIdentity;
use crate::domain::{Action, ExecutionFacts, Task, TaskStatus, UndoPhase, UndoProgress};
use crate::events::{CoreEvent, CoreEventKind};
use crate::execution::files::PinnedPath;
use crate::execution::{RollbackOperation, RollbackPlan};
use crate::secrets::SecretStore;
use crate::storage::{LocalStore, write_audit, write_event, write_task};
use crate::{CoreError, CoreResult};

const MAX_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct UndoIntent {
    task_id: Uuid,
    action_id: Uuid,
    plan_digest: String,
    path: String,
    parent_identity: String,
    before: FileIdentity,
    // None means the desired postcondition is absence. This digest is retained
    // separately from private backup bytes so a dispatched Undo can be checked
    // even after the backup expires or is forgotten.
    restored_sha256: Option<String>,
}

#[derive(Debug)]
struct PendingUndo {
    intent: UndoIntent,
    phase: UndoPhase,
}

#[derive(Serialize)]
struct UndoEvidence {
    parent_identity: String,
    identity: Option<FileIdentity>,
    sha256: Option<String>,
    observed_at: DateTime<Utc>,
}

struct PreparedUndo {
    intent: UndoIntent,
    path: PinnedPath,
    operation: RollbackOperation,
    bytes: Option<Vec<u8>>,
}

impl PreparedUndo {
    fn new(store: &LocalStore, task_id: Uuid, plan: &RollbackPlan) -> CoreResult<Self> {
        if plan.expires_at <= Utc::now() {
            return Err(CoreError::InvalidAction("Undo data has expired".into()));
        }
        let [operation] = plan.operations.as_slice() else {
            return Err(CoreError::PermissionRequired(
                "This recovery plan requires manual review".into(),
            ));
        };
        let target = match operation {
            RollbackOperation::RestoreArtifact { destination, .. } => destination,
            RollbackOperation::RemoveCreatedFile { path, .. }
            | RollbackOperation::RemoveCreatedFolder { path, .. } => path,
            _ => {
                return Err(CoreError::PermissionRequired(
                    "Legacy recovery needs manual review".into(),
                ));
            }
        };
        let path = PinnedPath::open(Path::new(target))?;
        let before = path.snapshot_identity()?.ok_or_else(|| {
            CoreError::ApprovalRejected("Undo refused: the target no longer exists".into())
        })?;
        let bytes = match operation {
            RollbackOperation::RestoreArtifact {
                artifact_id,
                expected_sha256,
                ..
            } => {
                check_digest(&path, expected_sha256)?;
                Some(store.read_undo_artifact(*artifact_id, task_id)?)
            }
            RollbackOperation::RemoveCreatedFile {
                expected_sha256, ..
            } => {
                check_digest(&path, expected_sha256)?;
                None
            }
            RollbackOperation::RemoveCreatedFolder { identity, .. } => {
                if !before.directory || before.key != *identity {
                    return Err(CoreError::ApprovalRejected(
                        "Undo refused: folder identity changed".into(),
                    ));
                }
                path.require_empty_directory()?;
                None
            }
            _ => unreachable!("operation was validated above"),
        };
        let intent = UndoIntent {
            task_id,
            action_id: plan.action_id,
            plan_digest: digest(&serde_json::to_vec(plan)?),
            path: target.clone(),
            parent_identity: path.parent_identity()?,
            before,
            restored_sha256: bytes.as_ref().map(|bytes| digest(bytes)),
        };
        Ok(Self {
            intent,
            path,
            operation: operation.clone(),
            bytes,
        })
    }

    fn execute(&self) -> CoreResult<()> {
        match &self.operation {
            RollbackOperation::RestoreArtifact { .. } => self.path.write(
                self.bytes
                    .as_deref()
                    .expect("restore was prepared with bytes"),
                true,
            ),
            RollbackOperation::RemoveCreatedFile {
                expected_sha256, ..
            } => self.path.remove_verified(expected_sha256),
            RollbackOperation::RemoveCreatedFolder { identity, .. } => {
                self.path.remove_empty_verified(identity)
            }
            _ => unreachable!("only supported inverses are prepared"),
        }
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn check_digest(path: &PinnedPath, expected: &str) -> CoreResult<()> {
    if path.digest(MAX_BYTES)? != expected {
        return Err(CoreError::ApprovalRejected(
            "Undo refused: file content changed".into(),
        ));
    }
    Ok(())
}

impl UndoIntent {
    fn observe(&self) -> CoreResult<UndoEvidence> {
        // Open afresh, independently of the mutation handle. A replaced parent
        // cannot be mistaken for the directory in which Undo was dispatched.
        let path = PinnedPath::open(Path::new(&self.path))?;
        let parent_identity = path.parent_identity()?;
        if parent_identity != self.parent_identity {
            return Err(CoreError::VerificationFailed(
                "Undo target directory changed".into(),
            ));
        }
        let identity = path.snapshot_identity()?;
        let sha256 = if let Some(expected) = &self.restored_sha256 {
            check_digest(&path, expected)?;
            Some(expected.clone())
        } else {
            if identity.is_some() {
                return Err(CoreError::VerificationFailed(
                    "Undo target is still present".into(),
                ));
            }
            None
        };
        Ok(UndoEvidence {
            parent_identity,
            identity,
            sha256,
            observed_at: Utc::now(),
        })
    }
}

/// The engine holds admission, mutation, task-cache and retention locks across
/// this synchronous operation. Each committed projection replaces its cached
/// task before publication; there is no await between dispatch and execution.
pub(crate) fn run_requested(
    store: &LocalStore,
    task: &mut Task,
    action_id: Uuid,
    secrets: &dyn SecretStore,
    publish: impl Fn(CoreEvent),
) -> CoreResult<()> {
    let verified = store.with_connection(|db| Ok(db.query_row(
        "SELECT EXISTS(SELECT 1 FROM undo_journal WHERE task_id=?1 AND action_id=?2 AND phase='verified')",
        params![task.id.to_string(), action_id.to_string()], |row| row.get::<_, bool>(0))?))?;
    if verified {
        // Lost acknowledgements and duplicate clicks refer to this inverse,
        // never to whichever action becomes the next available Undo.
        return store.checkpoint_audit(secrets);
    }
    let selected = if let Some(pending) = store.pending_undo(task.id)? {
        Some(pending.intent.action_id)
    } else {
        store.latest_rollback(task.id)?.map(|plan| plan.action_id)
    };
    if selected != Some(action_id) {
        return Err(CoreError::InvalidAction(
            "The available Undo changed. Refresh this task before trying again.".into(),
        ));
    }
    run(store, task, secrets, publish)
}

fn run(
    store: &LocalStore,
    task: &mut Task,
    secrets: &dyn SecretStore,
    publish: impl Fn(CoreEvent),
) -> CoreResult<()> {
    if task.status.is_active() {
        return Err(CoreError::PermissionRequired(
            "Stop the task before Undo".into(),
        ));
    }
    let pending = store.pending_undo(task.id)?;
    if let Some(pending) = &pending
        && pending.phase != UndoPhase::Prepared
    {
        return reconcile(store, task, &pending.intent, secrets, &publish);
    }
    let plan = if let Some(pending) = &pending {
        store.undo_plan(pending.intent.action_id)?
    } else {
        store
            .latest_rollback(task.id)?
            .ok_or_else(|| CoreError::InvalidAction("No reversible action is available".into()))?
    };
    validate_inverse(task, &plan)?;
    let prepared = PreparedUndo::new(store, task.id, &plan)?;
    if let Some(pending) = pending {
        if prepared.intent != pending.intent {
            return Err(CoreError::ApprovalRejected(
                "Undo target or recovery data changed after preparation".into(),
            ));
        }
    } else {
        let (updated, event) = store.prepare_undo(task.clone(), &prepared.intent)?;
        *task = updated;
        publish(event);
    }
    #[cfg(test)]
    crash_checkpoint("prepared");
    let (updated, event) =
        store.transition_undo(task.clone(), &prepared.intent, UndoPhase::Dispatched, None)?;
    *task = updated;
    publish(event);
    #[cfg(test)]
    crash_checkpoint("dispatched");
    // OS-backed audit anchoring is separate from SQLite. If it fails, leave a
    // conservative dispatched record; do not execute or claim verified Undo.
    store.checkpoint_audit(secrets)?;
    let effect = prepared.execute();
    #[cfg(test)]
    crash_checkpoint("effect");
    // Even a successful syscall is only a receipt. Observe the desired state.
    let result = reconcile(store, task, &prepared.intent, secrets, &publish);
    if result.is_ok() {
        return result;
    }
    // Filesystem errors are useful to the caller but never authorize replay.
    effect.and(result)
}

fn validate_inverse(task: &Task, plan: &RollbackPlan) -> CoreResult<()> {
    let proposal = task
        .actions
        .get(&plan.action_id)
        .map(|state| &state.proposal)
        .ok_or_else(|| {
            CoreError::PermissionRequired("Undo does not belong to a recorded action".into())
        })?;
    let before: Option<FileIdentity> =
        serde_json::from_str(proposal.metadata.get("file_precondition").ok_or_else(|| {
            CoreError::PermissionRequired(
                "This older recovery plan lacks a file identity and needs manual review".into(),
            )
        })?)?;
    let valid = proposal.task_id == task.id
        && match (plan.operations.as_slice(), &proposal.action) {
            (
                [
                    RollbackOperation::RestoreArtifact {
                        destination,
                        expected_sha256,
                        ..
                    },
                ],
                Action::WriteFile {
                    path,
                    content,
                    overwrite,
                },
            ) => {
                *overwrite
                    && before.is_some_and(|identity| !identity.directory)
                    && Path::new(destination) == path
                    && *expected_sha256 == digest(content.as_bytes())
            }
            (
                [
                    RollbackOperation::RemoveCreatedFile {
                        path: target,
                        expected_sha256,
                    },
                ],
                Action::WriteFile { path, content, .. },
            ) => {
                before.is_none()
                    && Path::new(target) == path
                    && *expected_sha256 == digest(content.as_bytes())
            }
            (
                [RollbackOperation::RemoveCreatedFolder { path: target, .. }],
                Action::CreateFolder { path },
            ) => before.is_none() && Path::new(target) == path,
            _ => false,
        };
    if !valid {
        return Err(CoreError::PermissionRequired(
            "Undo does not match the recorded file operation".into(),
        ));
    }
    Ok(())
}

fn reconcile(
    store: &LocalStore,
    task: &mut Task,
    intent: &UndoIntent,
    secrets: &dyn SecretStore,
    publish: &impl Fn(CoreEvent),
) -> CoreResult<()> {
    let evidence = intent.observe();
    let phase = if evidence.is_ok() {
        UndoPhase::Verified
    } else {
        UndoPhase::Uncertain
    };
    let (updated, event) =
        store.transition_undo(task.clone(), intent, phase, evidence.as_ref().ok())?;
    *task = updated;
    publish(event);
    store.checkpoint_audit(secrets)?;
    evidence
        .map(|_| ())
        .map_err(|_| CoreError::VerificationFailed(UndoPhase::Uncertain.summary().into()))
}

impl LocalStore {
    pub(crate) fn repair_undo_availability(&self) -> CoreResult<()> {
        // Keyset batches also repair pre-upgrade rows and the crash gap between
        // saving forward recovery metadata and projecting its Undo control.
        let mut after = String::new();
        loop {
            let last = self.with_connection(|db| {
                let tx = db.transaction()?;
                let rows = {
                    let mut statement = tx.prepare("SELECT task_json FROM tasks WHERE id>?1 AND EXISTS(SELECT 1 FROM rollback_plans p WHERE p.task_id=tasks.id) ORDER BY id LIMIT 128")?;
                    statement.query_map([&after], |row| row.get::<_, String>(0))?.collect::<Result<Vec<_>, _>>()?
                };
                let mut last = None;
                for json in rows {
                    let mut task: Task = serde_json::from_str(&json)?;
                    last = Some(task.id.to_string());
                    let selected = available_undo(&tx, task.id)?;
                    if task.rollback_action_id != selected || task.rollback_available != selected.is_some() {
                        task.rollback_action_id = selected;
                        task.rollback_available = selected.is_some();
                        task.touch();
                        write_task(&tx, &task)?;
                    }
                }
                tx.commit()?;
                Ok(last)
            })?;
            let Some(last) = last else {
                break;
            };
            after = last;
        }
        Ok(())
    }

    fn undo_plan(&self, action_id: Uuid) -> CoreResult<RollbackPlan> {
        self.with_connection(|db| {
            let json: String = db.query_row(
                "SELECT plan_json FROM rollback_plans WHERE action_id=?1 AND consumed_at IS NULL",
                [action_id.to_string()],
                |row| row.get(0),
            )?;
            Ok(serde_json::from_str(&json)?)
        })
    }

    fn pending_undo(&self, task_id: Uuid) -> CoreResult<Option<PendingUndo>> {
        self.with_connection(|db| {
            let row: Option<(String, String)> = db.query_row(
                "SELECT intent_json,phase FROM undo_journal WHERE task_id=?1 AND phase!='verified' ORDER BY rowid DESC LIMIT 1",
                [task_id.to_string()], |row| Ok((row.get(0)?, row.get(1)?))).optional()?;
            row.map(|(intent, phase)| Ok(PendingUndo {
                intent: serde_json::from_str(&intent)?,
                phase: serde_json::from_value(json!(phase))?,
            })).transpose()
        })
    }

    pub(crate) fn scope_has_undo(&self, scope_id: Uuid) -> CoreResult<bool> {
        self.with_connection(|db| Ok(db.query_row(
            "SELECT EXISTS(SELECT 1 FROM undo_journal j JOIN tasks t ON t.id=j.task_id WHERE COALESCE(json_extract(t.task_json,'$.control_scope_id'),t.id)=?1)",
            [scope_id.to_string()], |row| row.get(0))?))
    }

    fn prepare_undo(&self, mut task: Task, intent: &UndoIntent) -> CoreResult<(Task, CoreEvent)> {
        let event = undo_event(&mut task, intent.action_id, UndoPhase::Prepared);
        let revision = self.with_connection(|db| {
            let tx = db.transaction()?;
            let (plan, expiry): (String, String) = tx.query_row(
                "SELECT plan_json,expires_at FROM rollback_plans WHERE action_id=?1 AND task_id=?2 AND consumed_at IS NULL",
                params![intent.action_id.to_string(), task.id.to_string()], |row| Ok((row.get(0)?,row.get(1)?)))?;
            if intent.task_id != task.id || digest(plan.as_bytes()) != intent.plan_digest
                || DateTime::parse_from_rfc3339(&expiry)? <= Utc::now() {
                return Err("Recovery data changed before preparing Undo".into());
            }
            let pending: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM undo_journal WHERE task_id=?1 AND phase!='verified')", [task.id.to_string()], |row| row.get(0))?;
            if pending { return Err("Finish checking the previous Undo first".into()); }
            tx.execute("INSERT INTO undo_journal(action_id,task_id,intent_json,phase,updated_at) VALUES(?1,?2,?3,'prepared',?4)", params![intent.action_id.to_string(), task.id.to_string(), serde_json::to_string(intent)?, Utc::now().to_rfc3339()])?;
            let revision = write_undo_projection(&tx, &task, &event, intent, UndoPhase::Prepared)?;
            tx.commit()?;
            Ok(revision)
        })?;
        task.revision = revision;
        Ok((task, event))
    }

    fn transition_undo(
        &self,
        mut task: Task,
        intent: &UndoIntent,
        phase: UndoPhase,
        evidence: Option<&UndoEvidence>,
    ) -> CoreResult<(Task, CoreEvent)> {
        if intent.task_id != task.id || task.status.is_active() || phase == UndoPhase::Prepared {
            return Err(CoreError::InvalidAction("Invalid Undo transition".into()));
        }
        if (phase == UndoPhase::Verified) != evidence.is_some() {
            return Err(CoreError::VerificationFailed(
                "Undo needs fresh postcondition evidence".into(),
            ));
        }
        let event = undo_event(&mut task, intent.action_id, phase);
        let revision = self.with_connection(|db| {
            let tx = db.transaction()?;
            let (prior, saved): (String, String) = tx.query_row("SELECT phase,intent_json FROM undo_journal WHERE action_id=?1 AND task_id=?2", params![intent.action_id.to_string(),task.id.to_string()], |row| Ok((row.get(0)?,row.get(1)?)))?;
            if serde_json::from_str::<UndoIntent>(&saved)? != *intent
                || if phase == UndoPhase::Dispatched { prior != "prepared" } else { !matches!(prior.as_str(), "dispatched" | "uncertain") } {
                return Err("Undo phase or target changed; reload its current state".into());
            }
            tx.execute("UPDATE undo_journal SET phase=?2,evidence_json=?3,updated_at=?4 WHERE action_id=?1", params![intent.action_id.to_string(), phase.as_str(), evidence.map(serde_json::to_string).transpose()?, Utc::now().to_rfc3339()])?;
            #[cfg(test)] if phase == UndoPhase::Verified { crash_checkpoint("verified_journal"); }
            if phase == UndoPhase::Verified {
                let changed = tx.execute("UPDATE rollback_plans SET consumed_at=?2 WHERE action_id=?1 AND consumed_at IS NULL", params![intent.action_id.to_string(),Utc::now().to_rfc3339()])?;
                if changed != 1 { return Err("Undo was already consumed".into()); }
                task.undone_actions.insert(intent.action_id);
                task.rollback_action_id = available_undo(&tx, task.id)?;
                task.rollback_available = task.rollback_action_id.is_some();
            }
            // Completed forward actions remain historical facts. Their current
            // completion and reusable context must reflect the inverse effect.
            if task.status != TaskStatus::Cancelled {
                task.status = if phase == UndoPhase::Verified { TaskStatus::Partial } else { TaskStatus::Interrupted };
            }
            task.final_outcome = Some(format!("{} {}", phase.summary(), ExecutionFacts::for_task(&task).summary()));
            if phase == UndoPhase::Dispatched {
                invalidate_context(&tx, &task)?;
            }
            let revision = write_undo_projection(&tx, &task, &event, intent, phase)?;
            tx.commit()?;
            #[cfg(test)] if phase == UndoPhase::Verified { crash_checkpoint("verified_committed"); }
            Ok(revision)
        })?;
        task.revision = revision;
        Ok((task, event))
    }
}

fn undo_event(task: &mut Task, action_id: Uuid, phase: UndoPhase) -> CoreEvent {
    task.undo = Some(UndoProgress { action_id, phase });
    task.rollback_available = true;
    task.rollback_action_id = Some(action_id);
    task.touch();
    CoreEvent::new(
        Some(task.id),
        CoreEventKind::UndoChanged { action_id, phase },
    )
}

fn write_undo_projection(
    tx: &rusqlite::Transaction<'_>,
    task: &Task,
    event: &CoreEvent,
    intent: &UndoIntent,
    phase: UndoPhase,
) -> CoreResult<u64> {
    let revision = write_task(tx, task)?;
    #[cfg(test)]
    if phase == UndoPhase::Verified {
        crash_checkpoint("verified_task");
    }
    write_event(tx, event)?;
    write_audit(
        tx,
        Some(task.id),
        Some(intent.action_id),
        &format!("undo_{}", phase.as_str()),
        &json!({"plan_digest":intent.plan_digest,"event_id":event.id}),
    )?;
    #[cfg(test)]
    if phase == UndoPhase::Verified {
        crash_checkpoint("verified_records");
    }
    Ok(revision)
}

fn invalidate_context(tx: &rusqlite::Transaction<'_>, task: &Task) -> CoreResult<()> {
    let Some(conversation) = task.conversation_id else {
        return Ok(());
    };
    tx.execute(
        "DELETE FROM working_memory WHERE conversation_id=?1",
        [conversation.to_string()],
    )?;
    tx.execute(
        "UPDATE conversations SET summary='' WHERE id=?1",
        [conversation.to_string()],
    )?;
    tx.execute("INSERT OR IGNORE INTO excluded_context_messages SELECT id FROM messages WHERE task_id=?1 AND role='assistant'", [task.id.to_string()])?;
    tx.execute("WITH RECURSIVE invalid(id) AS (SELECT ?1 UNION SELECT child_id FROM memory_lineage JOIN invalid ON parent_id=invalid.id) UPDATE memory_records SET enabled=0 WHERE id IN (SELECT id FROM invalid)", [task.id.to_string()])?;
    Ok(())
}

fn available_undo(db: &rusqlite::Connection, task_id: Uuid) -> CoreResult<Option<Uuid>> {
    let pending: Option<String> = db.query_row("SELECT action_id FROM undo_journal WHERE task_id=?1 AND phase!='verified' ORDER BY rowid DESC LIMIT 1", [task_id.to_string()], |row| row.get(0)).optional()?;
    let selected = if pending.is_some() {
        pending
    } else {
        db.query_row("SELECT action_id FROM rollback_plans p WHERE task_id=?1 AND consumed_at IS NULL AND expires_at>?2 AND NOT EXISTS(SELECT 1 FROM undo_journal j WHERE j.action_id=p.action_id) ORDER BY rowid DESC LIMIT 1", params![task_id.to_string(),Utc::now().to_rfc3339()], |row| row.get::<_,String>(0)).optional()?
    };
    selected
        .map(|id| {
            Uuid::parse_str(&id)
                .map_err(|_| CoreError::Storage("Invalid Undo action identity".into()))
        })
        .transpose()
}

#[cfg(test)]
fn crash_checkpoint(stage: &str) {
    if std::env::var("SAGE_TEST_UNDO_EXIT_AT").ok().as_deref() == Some(stage) {
        std::process::exit(89);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{DataLabel, ToolResult, Verdict};
    use crate::domain::{
        Action, ActionProposal, ActionState, ActionStatus, ExpectedOutcome, Provenance,
    };
    use crate::secrets::{SecretBytes, testing::MemorySecretStore};

    // Fixture-only credentials survive process exits without using OS secrets.
    struct FixtureSecrets(std::path::PathBuf);
    impl SecretStore for FixtureSecrets {
        fn get(&self, account: &str) -> CoreResult<Option<SecretBytes>> {
            match std::fs::read(self.0.join(digest(account.as_bytes()))) {
                Ok(bytes) => Ok(Some(SecretBytes::new(bytes))),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(error) => Err(error.into()),
            }
        }
        fn set(&self, account: &str, secret: &SecretBytes) -> CoreResult<()> {
            std::fs::create_dir_all(&self.0)?;
            std::fs::write(self.0.join(digest(account.as_bytes())), secret.expose())?;
            Ok(())
        }
        fn delete(&self, account: &str) -> CoreResult<()> {
            std::fs::remove_file(self.0.join(digest(account.as_bytes())))?;
            Ok(())
        }
    }

    fn fixture(root: &Path, kind: &str) -> (LocalStore, Task, RollbackPlan) {
        let root = root.canonicalize().unwrap();
        let store =
            LocalStore::open_encrypted(&root.join("undo.db"), &SecretBytes::new(vec![73; 32]))
                .unwrap();
        let mut task = Task::new("Fixture filesystem change");
        task.status = TaskStatus::Succeeded;
        task.final_outcome = Some("Created the fixture".into());
        let path = root.join("target");
        let operation = match kind {
            "folder" => {
                std::fs::create_dir(&path).unwrap();
                RollbackOperation::RemoveCreatedFolder {
                    path: path.to_string_lossy().into_owned(),
                    identity: PinnedPath::open(&path).unwrap().current_identity().unwrap(),
                }
            }
            "file" => {
                std::fs::write(&path, b"after").unwrap();
                RollbackOperation::RemoveCreatedFile {
                    path: path.to_string_lossy().into_owned(),
                    expected_sha256: digest(b"after"),
                }
            }
            "restore" => {
                std::fs::write(&path, b"after").unwrap();
                RollbackOperation::RestoreArtifact {
                    artifact_id: store.save_artifact(task.id, b"before").unwrap(),
                    destination: path.to_string_lossy().into_owned(),
                    expected_sha256: digest(b"after"),
                }
            }
            _ => panic!("unknown fixture"),
        };
        let action_id = Uuid::new_v4();
        task.actions.insert(
            action_id,
            ActionState {
                proposal: ActionProposal {
                    id: action_id,
                    task_id: task.id,
                    action: if kind == "folder" {
                        Action::CreateFolder { path: path.clone() }
                    } else {
                        Action::WriteFile {
                            path: path.clone(),
                            content: "after".into(),
                            overwrite: kind == "restore",
                        }
                    },
                    expected_outcome: ExpectedOutcome::FileContains {
                        path: path.clone(),
                        sha256: digest(b"after"),
                    },
                    target_resource: path.to_string_lossy().into_owned(),
                    provenance: Provenance::model(vec![]),
                    metadata: std::collections::BTreeMap::from([(
                        "file_precondition".into(),
                        serde_json::to_string(&if kind == "restore" {
                            PinnedPath::open(&path)
                                .unwrap()
                                .snapshot_identity()
                                .unwrap()
                        } else {
                            None
                        })
                        .unwrap(),
                    )]),
                },
                status: ActionStatus::Succeeded,
                attempts: 1,
                summary: Some("Fixture applied".into()),
                error: None,
            },
        );
        task.tool_results.push(ToolResult {
            action_id,
            tool: "fixture".into(),
            verdict: Verdict::Confirmed,
            summary: "Fixture applied".into(),
            output: json!({}),
            label: DataLabel::private(task.id, action_id.to_string()),
            observed_at: Utc::now(),
        });
        task.rollback_available = true;
        task.rollback_action_id = Some(action_id);
        store.save_task(&mut task).unwrap();
        let plan = RollbackPlan {
            action_id,
            operations: vec![operation],
            expires_at: Utc::now() + chrono::Duration::hours(1),
        };
        store.save_rollback(task.id, &plan).unwrap();
        (store, task, plan)
    }

    fn dispatch(store: &LocalStore, task: &mut Task, plan: &RollbackPlan) -> PreparedUndo {
        let prepared = PreparedUndo::new(store, task.id, plan).unwrap();
        *task = store
            .prepare_undo(task.clone(), &prepared.intent)
            .unwrap()
            .0;
        *task = store
            .transition_undo(task.clone(), &prepared.intent, UndoPhase::Dispatched, None)
            .unwrap()
            .0;
        prepared
    }

    fn assert_effect(root: &Path, kind: &str, undone: bool) {
        let path = root.join("target");
        if kind == "restore" {
            assert_eq!(
                std::fs::read(path).unwrap(),
                if undone {
                    b"before".as_slice()
                } else {
                    b"after".as_slice()
                }
            );
        } else {
            assert_eq!(path.exists(), !undone);
        }
    }

    fn assert_accounting(store: &LocalStore, task: &Task, phase: UndoPhase) {
        assert_eq!(task.undo.as_ref().unwrap().phase, phase);
        assert_eq!(store.load_tasks(true).unwrap()[0], *task);
        store.with_connection(|db| {
            let (saved, consumed, proof): (String,bool,bool) = db.query_row(
                "SELECT phase,consumed_at IS NOT NULL,evidence_json IS NOT NULL FROM undo_journal JOIN rollback_plans USING(action_id)", [],
                |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?;
            assert_eq!(saved, phase.as_str());
            assert_eq!(consumed, phase == UndoPhase::Verified);
            assert_eq!(proof, consumed);
            for table in ["events", "audit_log"] {
                let count: i64 = db.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| row.get(0))?;
                assert_eq!(count as u64, task.revision - 1);
            }
            Ok(())
        }).unwrap();
    }

    #[test]
    fn all_supported_inverses_verify_and_consume_only_after_observation() {
        for kind in ["file", "restore", "folder"] {
            let dir = tempfile::tempdir().unwrap();
            let (store, mut task, plan) = fixture(dir.path(), kind);
            let secrets = MemorySecretStore::default();
            run(&store, &mut task, &secrets, |_| {}).unwrap();
            assert_effect(dir.path(), kind, true);
            assert_accounting(&store, &task, UndoPhase::Verified);
            assert!(!task.rollback_available);
            assert_eq!(task.status, TaskStatus::Partial);
            assert_eq!(ExecutionFacts::for_task(&task).undone, 1);
            assert!(!ExecutionFacts::for_task(&task).all_verified());
            assert_eq!(ExecutionFacts::for_task(&task).verified, 0);
            assert!(
                store.save_rollback(task.id, &plan).is_err(),
                "cannot re-arm a consumed Undo"
            );
            assert!(run(&store, &mut task, &secrets, |_| {}).is_err());
            assert_effect(dir.path(), kind, true);
        }
    }

    #[test]
    fn dispatched_undo_observes_without_replaying_even_when_original_state_is_unchanged() {
        for kind in ["file", "restore", "folder"] {
            let dir = tempfile::tempdir().unwrap();
            let (store, mut task, plan) = fixture(dir.path(), kind);
            dispatch(&store, &mut task, &plan);
            for _ in 0..2 {
                assert!(run(&store, &mut task, &MemorySecretStore::default(), |_| {}).is_err());
                assert_effect(dir.path(), kind, false);
                assert_accounting(&store, &task, UndoPhase::Uncertain);
                assert_eq!(ExecutionFacts::for_task(&task).uncertain, 1);
            }
        }
    }

    #[test]
    fn preflight_rejects_changed_targets_wrong_artifact_owners_and_parent_replacement() {
        for kind in ["file", "restore", "folder"] {
            let dir = tempfile::tempdir().unwrap();
            let (store, mut task, plan) = fixture(dir.path(), kind);
            let prepared = PreparedUndo::new(&store, task.id, &plan).unwrap();
            task = store.prepare_undo(task, &prepared.intent).unwrap().0;
            if kind == "folder" {
                std::fs::rename(dir.path().join("target"), dir.path().join("old-target")).unwrap();
                std::fs::create_dir(dir.path().join("target")).unwrap();
            } else {
                std::fs::write(dir.path().join("target"), b"user edits").unwrap();
            }
            assert!(run(&store, &mut task, &MemorySecretStore::default(), |_| {}).is_err());
            assert_accounting(&store, &task, UndoPhase::Prepared);
        }
        let dir = tempfile::tempdir().unwrap();
        let (store, task, mut plan) = fixture(dir.path(), "restore");
        let foreign = store
            .save_artifact(Uuid::new_v4(), b"foreign backup")
            .unwrap();
        if let RollbackOperation::RestoreArtifact { artifact_id, .. } = &mut plan.operations[0] {
            *artifact_id = foreign;
        }
        assert!(PreparedUndo::new(&store, task.id, &plan).is_err());

        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        std::fs::create_dir(&work).unwrap();
        let (store, mut task, plan) = fixture(&work, "file");
        let prepared = dispatch(&store, &mut task, &plan);
        prepared.execute().unwrap();
        std::fs::rename(&work, dir.path().join("old-work")).unwrap();
        std::fs::create_dir(&work).unwrap();
        assert!(
            prepared.intent.observe().is_err(),
            "absence in a different directory is not evidence"
        );
    }

    #[test]
    fn recovery_after_effect_needs_no_backup_bytes_and_preserves_later_edits() {
        for edit in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let (store, mut task, plan) = fixture(dir.path(), "restore");
            let prepared = dispatch(&store, &mut task, &plan);
            prepared.execute().unwrap();
            store
                .with_connection(|db| {
                    db.execute("DELETE FROM private_artifacts", [])?;
                    db.execute(
                        "UPDATE rollback_plans SET expires_at='2000-01-01T00:00:00Z'",
                        [],
                    )?;
                    Ok(())
                })
                .unwrap();
            if edit {
                std::fs::write(dir.path().join("target"), b"new user edits").unwrap();
            }
            let result = run(&store, &mut task, &MemorySecretStore::default(), |_| {});
            assert_eq!(result.is_ok(), !edit);
            assert_accounting(
                &store,
                &task,
                if edit {
                    UndoPhase::Uncertain
                } else {
                    UndoPhase::Verified
                },
            );
            assert_eq!(
                std::fs::read(dir.path().join("target")).unwrap(),
                if edit {
                    b"new user edits".as_slice()
                } else {
                    b"before".as_slice()
                }
            );
        }
    }

    #[test]
    fn failed_undo_transactions_leave_all_projections_and_consumption_unchanged() {
        for boundary in ["prepare", "dispatch", "verify"] {
            for table in [
                "tasks",
                "actions",
                "events",
                "audit_log",
                "undo_journal",
                "rollback_plans",
            ] {
                // This table changes only at verified consumption.
                if table == "rollback_plans" && boundary != "verify" {
                    continue;
                }
                let dir = tempfile::tempdir().unwrap();
                let (store, mut task, plan) = fixture(dir.path(), "restore");
                let prepared = PreparedUndo::new(&store, task.id, &plan).unwrap();
                if boundary != "prepare" {
                    task = store.prepare_undo(task, &prepared.intent).unwrap().0;
                }
                if boundary == "verify" {
                    task = store
                        .transition_undo(task, &prepared.intent, UndoPhase::Dispatched, None)
                        .unwrap()
                        .0;
                    prepared.execute().unwrap();
                }
                let evidence = if boundary == "verify" {
                    Some(prepared.intent.observe().unwrap())
                } else {
                    None
                };
                let before = task.clone();
                let operation = if (table == "undo_journal" && boundary != "prepare")
                    || table == "rollback_plans"
                {
                    "UPDATE"
                } else {
                    "INSERT"
                };
                store.with_connection(|db| { db.execute_batch(&format!("CREATE TRIGGER fail_undo BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'fixture failure'); END;"))?; Ok(()) }).unwrap();
                let result = if boundary == "prepare" {
                    store.prepare_undo(task.clone(), &prepared.intent)
                } else {
                    store.transition_undo(
                        task.clone(),
                        &prepared.intent,
                        if boundary == "dispatch" {
                            UndoPhase::Dispatched
                        } else {
                            UndoPhase::Verified
                        },
                        evidence.as_ref(),
                    )
                };
                assert!(result.is_err(), "{boundary}: {table}");
                assert_eq!(store.load_tasks(true).unwrap()[0], before);
                assert_eq!(task, before);
                if boundary != "prepare" {
                    assert_accounting(
                        &store,
                        &task,
                        if boundary == "dispatch" {
                            UndoPhase::Prepared
                        } else {
                            UndoPhase::Dispatched
                        },
                    );
                } else {
                    assert!(store.pending_undo(task.id).unwrap().is_none());
                    assert!(store.latest_rollback(task.id).unwrap().is_some());
                }
                store
                    .with_connection(|db| {
                        db.execute_batch("DROP TRIGGER fail_undo")?;
                        Ok(())
                    })
                    .unwrap();
                run(&store, &mut task, &MemorySecretStore::default(), |_| {}).unwrap();
                assert_effect(dir.path(), "restore", true);
                assert_accounting(&store, &task, UndoPhase::Verified);
            }
        }
    }

    #[test]
    fn stale_task_cannot_commit_undo_and_cancelled_task_stays_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mut task, plan) = fixture(dir.path(), "file");
        let prepared = dispatch(&store, &mut task, &plan);
        prepared.execute().unwrap();
        let stale = task.clone();
        task.status = TaskStatus::Cancelled;
        store.save_task(&mut task).unwrap();
        assert!(
            store
                .transition_undo(
                    stale,
                    &prepared.intent,
                    UndoPhase::Verified,
                    Some(&prepared.intent.observe().unwrap())
                )
                .is_err()
        );
        run(&store, &mut task, &MemorySecretStore::default(), |_| {}).unwrap();
        assert_eq!(task.status, TaskStatus::Cancelled);
        assert!(task.undone_actions.contains(&plan.action_id));
    }

    #[test]
    fn undo_invalidates_success_context_and_stale_finalization_cannot_recreate_it() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _, _) = fixture(dir.path(), "file");
        store.migrate_knowledge().unwrap();
        let mut task = store.load_tasks(true).unwrap().remove(0);
        store.record_outcome(&task).unwrap();
        assert_eq!(
            store
                .context_messages(task.conversation_id.unwrap(), 10)
                .unwrap()
                .into_iter()
                .filter(|message| message.role == "assistant")
                .collect::<Vec<_>>()
                .len(),
            1
        );
        assert!(
            store
                .working_memory(task.conversation_id.unwrap())
                .unwrap()
                .is_some()
        );
        let stale = task.clone();
        run(&store, &mut task, &MemorySecretStore::default(), |_| {}).unwrap();
        store.record_outcome(&stale).unwrap();
        assert!(
            store
                .context_messages(task.conversation_id.unwrap(), 10)
                .unwrap()
                .iter()
                .all(|message| message.role == "user")
        );
        assert!(
            store
                .working_memory(task.conversation_id.unwrap())
                .unwrap()
                .is_none()
        );
        assert!(store.conversations().unwrap()[0].summary.is_empty());
        assert!(
            store
                .memories(None, true, 10)
                .unwrap()
                .iter()
                .all(|m| m.id != task.id)
        );
    }

    #[test]
    fn abrupt_undo_exit_child() {
        let Some(root) = std::env::var_os("SAGE_TEST_UNDO_ROOT") else {
            return;
        };
        let kind = std::env::var("SAGE_TEST_UNDO_KIND").unwrap();
        let (store, mut task, _) = fixture(Path::new(&root), &kind);
        run(
            &store,
            &mut task,
            &FixtureSecrets(Path::new(&root).join("credentials")),
            |_| {},
        )
        .unwrap();
        panic!("Expected abrupt process exit");
    }

    #[test]
    fn duplicate_undo_identity_never_selects_the_next_older_action() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mut task, older) = fixture(dir.path(), "file");
        let second = dir.path().canonicalize().unwrap().join("second");
        std::fs::write(&second, b"second").unwrap();
        let mut newer = older.clone();
        newer.action_id = Uuid::new_v4();
        newer.operations = vec![RollbackOperation::RemoveCreatedFile {
            path: second.to_string_lossy().into_owned(),
            expected_sha256: digest(b"second"),
        }];
        let mut action = task.actions[&older.action_id].clone();
        action.proposal.id = newer.action_id;
        action.proposal.action = Action::WriteFile {
            path: second.clone(),
            content: "second".into(),
            overwrite: false,
        };
        task.actions.insert(newer.action_id, action);
        store.save_task(&mut task).unwrap();
        store.save_rollback(task.id, &newer).unwrap();
        let secrets = MemorySecretStore::default();
        assert!(run_requested(&store, &mut task, older.action_id, &secrets, |_| {}).is_err());
        run_requested(&store, &mut task, newer.action_id, &secrets, |_| {}).unwrap();
        assert!(!second.exists());
        let revision = task.revision;
        assert_eq!(task.rollback_action_id, Some(older.action_id));
        for _ in 0..3 {
            run_requested(&store, &mut task, newer.action_id, &secrets, |_| {}).unwrap();
            assert_effect(dir.path(), "file", false);
            assert_eq!(task.revision, revision);
        }
        // A separate explicit action identity authorizes the next inverse.
        run_requested(&store, &mut task, older.action_id, &secrets, |_| {}).unwrap();
        assert_effect(dir.path(), "file", true);
    }

    #[test]
    fn restart_repairs_legacy_undo_controls_without_rearming_consumed_plans() {
        for consumed in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let (store, mut task, plan) = fixture(dir.path(), "file");
            task.rollback_action_id = None;
            task.rollback_available = true;
            store.save_task(&mut task).unwrap();
            if consumed {
                store
                    .with_connection(|db| {
                        db.execute(
                            "UPDATE rollback_plans SET consumed_at=?1",
                            [Utc::now().to_rfc3339()],
                        )?;
                        Ok(())
                    })
                    .unwrap();
            }
            drop(store);
            let store = LocalStore::open_encrypted(
                &dir.path().join("undo.db"),
                &SecretBytes::new(vec![73; 32]),
            )
            .unwrap();
            let repaired = store.load_tasks(true).unwrap().remove(0);
            assert_eq!(repaired.rollback_available, !consumed);
            assert_eq!(
                repaired.rollback_action_id,
                if consumed { None } else { Some(plan.action_id) }
            );
            assert!(repaired.revision > task.revision);
            store.repair_undo_availability().unwrap();
            assert_eq!(
                store.load_tasks(true).unwrap()[0].revision,
                repaired.revision
            );
            assert_effect(dir.path(), "file", false);
        }
    }

    #[test]
    fn nonempty_folder_preflight_keeps_undo_retryable_without_a_dispatch() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mut task, plan) = fixture(dir.path(), "folder");
        let child = dir.path().join("target").join("user-file");
        std::fs::write(&child, b"keep").unwrap();
        assert!(
            run_requested(
                &store,
                &mut task,
                plan.action_id,
                &MemorySecretStore::default(),
                |_| {}
            )
            .is_err()
        );
        assert!(store.pending_undo(task.id).unwrap().is_none());
        assert_eq!(std::fs::read(&child).unwrap(), b"keep");
        std::fs::remove_file(child).unwrap();
        run_requested(
            &store,
            &mut task,
            plan.action_id,
            &MemorySecretStore::default(),
            |_| {},
        )
        .unwrap();
        assert_effect(dir.path(), "folder", true);
    }

    #[test]
    fn unavailable_audit_anchor_prevents_effect_and_does_not_authorize_a_retry() {
        struct Unavailable;
        impl SecretStore for Unavailable {
            fn get(&self, _: &str) -> CoreResult<Option<SecretBytes>> {
                Err(CoreError::SecretStore("fixture unavailable".into()))
            }
            fn set(&self, _: &str, _: &SecretBytes) -> CoreResult<()> {
                unreachable!()
            }
            fn delete(&self, _: &str) -> CoreResult<()> {
                unreachable!()
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let (store, mut task, plan) = fixture(dir.path(), "file");
        assert!(run_requested(&store, &mut task, plan.action_id, &Unavailable, |_| {}).is_err());
        assert_accounting(&store, &task, UndoPhase::Dispatched);
        assert_effect(dir.path(), "file", false);
        assert!(
            run_requested(
                &store,
                &mut task,
                plan.action_id,
                &MemorySecretStore::default(),
                |_| {}
            )
            .is_err()
        );
        assert_accounting(&store, &task, UndoPhase::Uncertain);
        assert_effect(dir.path(), "file", false);
    }

    #[test]
    fn recovery_plan_cannot_redirect_or_change_the_recorded_inverse() {
        let dir = tempfile::tempdir().unwrap();
        let (store, task, plan) = fixture(dir.path(), "restore");
        let mut redirected = plan.clone();
        redirected.operations = vec![RollbackOperation::RemoveCreatedFile {
            path: dir
                .path()
                .canonicalize()
                .unwrap()
                .join("target")
                .to_string_lossy()
                .into_owned(),
            expected_sha256: digest(b"after"),
        }];
        assert!(
            validate_inverse(&task, &redirected).is_err(),
            "an overwritten file must not be deleted"
        );
        let other = dir.path().canonicalize().unwrap().join("other");
        std::fs::write(&other, b"after").unwrap();
        redirected = plan.clone();
        if let RollbackOperation::RestoreArtifact { destination, .. } =
            &mut redirected.operations[0]
        {
            *destination = other.to_string_lossy().into_owned();
        }
        assert!(validate_inverse(&task, &redirected).is_err());
        assert!(validate_inverse(&task, &plan).is_ok());
        assert_eq!(std::fs::read(other).unwrap(), b"after");
        assert!(store.pending_undo(task.id).unwrap().is_none());
    }

    #[test]
    fn abrupt_exit_matrix_recovers_without_repeating_an_ambiguous_inverse() {
        for kind in ["file", "restore", "folder"] {
            for stage in [
                "prepared",
                "dispatched",
                "effect",
                "verified_journal",
                "verified_task",
                "verified_records",
                "verified_committed",
            ] {
                let dir = tempfile::tempdir().unwrap();
                let root = dir.path().canonicalize().unwrap();
                let child = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "undo::tests::abrupt_undo_exit_child",
                        "--nocapture",
                    ])
                    .env("SAGE_TEST_UNDO_ROOT", dir.path())
                    .env("SAGE_TEST_UNDO_KIND", kind)
                    .env("SAGE_TEST_UNDO_EXIT_AT", stage)
                    .output()
                    .unwrap();
                assert_eq!(
                    child.status.code(),
                    Some(89),
                    "{kind}/{stage}: {}",
                    String::from_utf8_lossy(&child.stderr)
                );
                let before_effect = matches!(stage, "prepared" | "dispatched");
                assert_effect(dir.path(), kind, !before_effect);
                // Inspect raw committed rows before startup repairs can mask a partial write.
                let key = SecretBytes::new(vec![73; 32]);
                let db = crate::vault::open_encrypted(&dir.path().join("undo.db"), &key).unwrap();
                let committed = stage == "verified_committed";
                let phase = if stage == "prepared" {
                    "prepared"
                } else if committed {
                    "verified"
                } else {
                    "dispatched"
                };
                let (saved, consumed, projection): (String,bool,String) = db.query_row("SELECT j.phase,p.consumed_at IS NOT NULL,json_extract(t.task_json,'$.undo.phase') FROM undo_journal j JOIN rollback_plans p USING(action_id) JOIN tasks t ON t.id=j.task_id", [], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?))).unwrap();
                assert_eq!(
                    (saved, consumed, projection),
                    (phase.into(), committed, phase.into())
                );
                drop(db);
                let store = LocalStore::open_encrypted(&root.join("undo.db"), &key).unwrap();
                let mut task = store.load_tasks(true).unwrap().remove(0);
                // Reopening never touches the filesystem.
                assert_effect(dir.path(), kind, !before_effect);
                if !committed {
                    let result = run(
                        &store,
                        &mut task,
                        &FixtureSecrets(dir.path().join("credentials")),
                        |_| {},
                    );
                    assert_eq!(
                        result.is_ok(),
                        stage != "dispatched",
                        "{kind}/{stage}: {result:?}"
                    );
                    assert_effect(dir.path(), kind, stage != "dispatched");
                    if stage == "dispatched" {
                        assert_accounting(&store, &task, UndoPhase::Uncertain);
                    } else {
                        assert_accounting(&store, &task, UndoPhase::Verified);
                    }
                } else {
                    assert_accounting(&store, &task, UndoPhase::Verified);
                }
            }
        }
    }
}
