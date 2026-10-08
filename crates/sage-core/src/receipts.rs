//! Durable receipt inbox. Transport evidence is projected exactly once into
//! task state, its event and audit link. Reconciliation never executes an effect
//! or treats a worker's success assertion as postcondition verification.
use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use serde_json::json;
use uuid::Uuid;

use crate::contracts::{DataLabel, ToolResult, Verdict};
use crate::domain::{ActionStatus, Task, TaskStatus};
use crate::events::{CoreEvent, CoreEventKind};
use crate::storage::{LocalStore, write_audit, write_event, write_task};
use crate::{CoreError, CoreResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReceiptKind {
    Response,
    CancelAcknowledged,
    NeverSent,
}

impl ReceiptKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Response => "response",
            Self::CancelAcknowledged => "cancel_ack",
            Self::NeverSent => "never_sent",
        }
    }
    fn parse(value: &str) -> CoreResult<Self> {
        match value {
            "response" => Ok(Self::Response),
            "cancel_ack" => Ok(Self::CancelAcknowledged),
            "never_sent" => Ok(Self::NeverSent),
            _ => Err(CoreError::Storage(
                "Unknown saved worker receipt kind".into(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReceiptKey {
    pub request_id: String,
    pub kind: ReceiptKind,
}

impl LocalStore {
    pub(crate) fn pending_worker_receipts(
        &self,
        task_id: Option<Uuid>,
    ) -> CoreResult<Vec<(Uuid, ReceiptKey)>> {
        self.with_connection(|db| {
            let mut query = db.prepare("SELECT r.run_id,r.request_id,r.kind FROM worker_receipts r JOIN tasks t ON t.id=r.run_id LEFT JOIN worker_receipt_projections p USING(request_id,kind) WHERE p.request_id IS NULL AND (?1 IS NULL OR r.run_id=?1) ORDER BY r.received_at,r.request_id,r.kind LIMIT 128")?;
            let rows = query.query_map([task_id.map(|id| id.to_string())], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
            })?;
            let mut pending = Vec::new();
            for row in rows {
                let (task, request_id, kind) = row?;
                pending.push((Uuid::parse_str(&task).map_err(|_| CoreError::Storage("Invalid receipt task identity".into()))?, ReceiptKey { request_id, kind: ReceiptKind::parse(&kind)? }));
            }
            Ok(pending)
        })
    }

    /// Run before loading the task cache on startup. The input is saved receipt
    /// metadata, never the untrusted raw reply body. Every batch is bounded.
    pub(crate) fn reconcile_stored_worker_receipts(&self) -> CoreResult<()> {
        loop {
            let pending = self.pending_worker_receipts(None)?;
            if pending.is_empty() {
                return Ok(());
            }
            for (task_id, key) in pending {
                let task = self.with_connection(|db| {
                    let json: String = db.query_row(
                        "SELECT task_json FROM tasks WHERE id=?1",
                        [task_id.to_string()],
                        |row| row.get(0),
                    )?;
                    Ok(serde_json::from_str::<Task>(&json)?)
                })?;
                self.commit_worker_receipt_projection(task, &key)?;
            }
        }
    }

    pub(crate) fn commit_worker_receipt_projection(
        &self,
        mut task: Task,
        key: &ReceiptKey,
    ) -> CoreResult<Option<(Task, CoreEvent)>> {
        let transition = self.with_connection(|db| {
            let tx = db.transaction()?;
            let projected: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM worker_receipt_projections WHERE request_id=?1 AND kind=?2)",
                params![key.request_id,key.kind.as_str()], |row| row.get(0))?;
            if projected { return Ok(None); }
            let receipt: Option<(String,String)> = tx.query_row("SELECT action_id,action_digest FROM worker_receipts WHERE request_id=?1 AND kind=?2 AND run_id=?3",
                params![key.request_id,key.kind.as_str(),task.id.to_string()], |row| Ok((row.get(0)?,row.get(1)?))).optional()?;
            let (action_id,digest) = receipt.ok_or_else(|| CoreError::Protocol("Saved receipt does not match this task".into()))?;
            let action_id = Uuid::parse_str(&action_id).map_err(|_| CoreError::Storage("Invalid receipt action identity".into()))?;
            let journal: Option<(String,String)> = tx.query_row("SELECT action_digest,state FROM action_journal WHERE run_id=?1 AND action_id=?2",
                params![task.id.to_string(),action_id.to_string()], |row| Ok((row.get(0)?,row.get(1)?))).optional()?;
            let same_action = journal.as_ref().is_some_and(|(current,_)| *current==digest)
                && task.actions.get(&action_id).map(|state| crate::policy::approval_digest(&state.proposal)).transpose()?.is_some_and(|current| current==digest);
            let verified = same_action && task.actions[&action_id].status == ActionStatus::Succeeded
                && task.tool_results.iter().rev().find(|result| result.action_id==action_id).is_some_and(|result| result.verdict==Verdict::Confirmed)
                && journal.as_ref().is_some_and(|(_,state)| state=="confirmed");
            let summary = if !same_action {
                "A saved worker receipt belongs to an earlier action identity. The current action state was preserved."
            } else if verified && key.kind != ReceiptKind::CancelAcknowledged {
                "An additional worker receipt was recorded for an independently verified action."
            } else {
                match key.kind {
                    ReceiptKind::Response => "A worker reply was recorded after its waiter stopped. Its effects require independent verification.",
                    ReceiptKind::CancelAcknowledged => "The worker received Stop. This does not establish whether an effect already occurred.",
                    ReceiptKind::NeverSent => "The broker stopped this operation before sending it to the worker.",
                }
            };
            if same_action && key.kind != ReceiptKind::CancelAcknowledged && !verified {
                let never_sent = key.kind == ReceiptKind::NeverSent;
                if never_sent && journal.as_ref().is_none_or(|(_,state)| state!="cancelled") {
                    return Err(CoreError::VerificationFailed("The journal does not establish that this worker operation was unsent".into()).into());
                }
                let action = task.actions.get_mut(&action_id).expect("matched receipt action");
                action.status = if never_sent { ActionStatus::Failed } else { ActionStatus::Uncertain };
                action.error = Some(summary.into());
                task.tool_results.push(ToolResult {
                    action_id, tool: "worker_receipt".into(), verdict: if never_sent { Verdict::Cancelled } else { Verdict::Uncertain },
                    summary: summary.into(), output: json!({"request_id":key.request_id,"never_sent":never_sent}),
                    label: DataLabel::private(task.id, action_id.to_string()), observed_at: Utc::now(),
                });
                if task.status != TaskStatus::Cancelled { task.status=TaskStatus::Interrupted; }
                task.final_outcome=Some(summary.into());
                if !never_sent {
                    tx.execute("UPDATE action_journal SET state='uncertain',updated_at=?3 WHERE run_id=?1 AND action_id=?2 AND state IN ('prepared','dispatched','failed','cancelled','uncertain')",
                        params![task.id.to_string(),action_id.to_string(),Utc::now().to_rfc3339()])?;
                }
            }
            task.touch();
            let event = CoreEvent::new(Some(task.id), CoreEventKind::ObservationReceived { action_id, summary: summary.into() });
            let revision = write_task(&tx, &task)?;
            let finalized: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM run_finalizations WHERE task_id=?1 AND execution_attempt=?2)", params![task.id.to_string(),task.execution_attempt], |row|row.get(0))?;
            if finalized && same_action && key.kind != ReceiptKind::CancelAcknowledged && !verified {
                crate::knowledge::write_outcome(&tx, &task)?;
            }
            #[cfg(test)] crash_checkpoint("task");
            write_event(&tx, &event)?;
            write_audit(&tx, Some(task.id), Some(action_id), "worker_receipt_projected", &json!({"request_id":key.request_id,"kind":key.kind.as_str(),"event_id":event.id,"same_action":same_action}))?;
            tx.execute("INSERT INTO worker_receipt_projections VALUES(?1,?2,?3,?4)", params![key.request_id,key.kind.as_str(),event.id.to_string(),Utc::now().to_rfc3339()])?;
            #[cfg(test)] crash_checkpoint("marker");
            tx.commit()?;
            #[cfg(test)] crash_checkpoint("committed");
            Ok(Some((revision,event)))
        })?;
        Ok(transition.map(|(revision, event)| {
            task.revision = revision;
            (task, event)
        }))
    }
}

/// Fresh independent verification subsumes earlier transport-only receipts for
/// this exact prepared action. Later receipts are still processed normally.
pub(crate) fn cover_verified_receipts(
    tx: &rusqlite::Transaction<'_>,
    record: &crate::contracts::VerificationRecord,
    event: &CoreEvent,
) -> CoreResult<()> {
    tx.execute("INSERT OR IGNORE INTO worker_receipt_projections SELECT request_id,kind,?4,?5 FROM worker_receipts WHERE run_id=?1 AND action_id=?2 AND action_digest=?3",
        params![record.run_id.to_string(),record.action_id.to_string(),record.action_digest,event.id.to_string(),Utc::now().to_rfc3339()])?;
    Ok(())
}

#[cfg(test)]
fn crash_checkpoint(stage: &str) {
    if std::env::var("SAGE_TEST_RECEIPT_EXIT_AT").ok().as_deref() == Some(stage) {
        std::process::exit(87);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::PreparedAction;
    use crate::domain::{
        Action, ActionProposal, ActionState, ExpectedOutcome, Provenance, ProvenanceSource,
    };
    use crate::execution::{ExecutionReceipt, bridge::EffectBinding};
    use crate::observation::{Evidence, Observation};
    use crate::secrets::{SecretBytes, testing::MemorySecretStore};
    use crate::transitions::{VerificationMode, VerifiedAction};

    fn fixture(path: &std::path::Path) -> (LocalStore, Task, PreparedAction, EffectBinding) {
        let store = LocalStore::open_encrypted(path, &SecretBytes::new(vec![33; 32])).unwrap();
        let mut task = Task::new("Receipt recovery fixture");
        task.status = TaskStatus::Cancelled;
        let proposal = ActionProposal {
            id: Uuid::new_v4(),
            task_id: task.id,
            action: Action::AskUser {
                question: "Choose a result".into(),
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: "user".into(),
            provenance: Provenance::model(vec![]),
            metadata: Default::default(),
        };
        task.actions.insert(
            proposal.id,
            ActionState {
                proposal: proposal.clone(),
                status: ActionStatus::Verifying,
                attempts: 1,
                summary: None,
                error: None,
            },
        );
        store.save_task(&mut task).unwrap();
        let prepared = PreparedAction::new(&proposal, Default::default()).unwrap();
        store.journal_prepared(&prepared).unwrap();
        store.journal_dispatch(&prepared, None).unwrap();
        let binding = EffectBinding {
            task_id: task.id,
            action_id: proposal.id,
            action_digest: prepared.action_digest.clone(),
        };
        (store, task, prepared, binding)
    }

    fn receive(store: &LocalStore, binding: &EffectBinding, kind: ReceiptKind) -> ReceiptKey {
        let key = ReceiptKey {
            request_id: Uuid::new_v4().to_string(),
            kind,
        };
        store
            .record_worker_receipt(
                &key.request_id,
                "fixture-session",
                binding,
                kind.as_str(),
                if kind == ReceiptKind::Response {
                    br#"{"success":true,"text":"private worker body"}"#
                } else {
                    &[]
                },
            )
            .unwrap();
        key
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

    fn verified(task: &Task, proposal: &ActionProposal) -> VerifiedAction {
        VerifiedAction::from_observation(
            task,
            proposal,
            &ExecutionReceipt {
                executor: "fixture".into(),
                summary: "Independent response evidence".into(),
                transient_data: json!({}),
                rollback: None,
            },
            &Observation {
                observed_at: Utc::now(),
                provenance: Provenance::external(ProvenanceSource::OperatingSystem, "fixture"),
                summary: "Response observed".into(),
                evidence: vec![Evidence::UserAnswer { received: true }],
            },
            VerificationMode::Execution,
        )
        .unwrap()
    }

    #[test]
    fn failed_projection_writes_leave_the_receipt_pending_and_retry_is_idempotent() {
        for table in ["tasks", "events", "audit_log", "worker_receipt_projections"] {
            let dir = tempfile::tempdir().unwrap();
            let (store, task, _, binding) = fixture(&dir.path().join("receipt.db"));
            let key = receive(&store, &binding, ReceiptKind::Response);
            store.with_connection(|db| {
                db.execute_batch(&format!("CREATE TEMP TRIGGER fail_projection BEFORE INSERT ON {table} BEGIN SELECT RAISE(ABORT,'injected receipt projection failure'); END;"))?;
                Ok(())
            }).unwrap();
            assert!(
                store
                    .commit_worker_receipt_projection(task.clone(), &key)
                    .is_err(),
                "{table}"
            );
            assert_eq!(store.load_tasks(true).unwrap(), vec![task.clone()]);
            assert_eq!(store.pending_worker_receipts(None).unwrap().len(), 1);
            assert_eq!(count(&store, "worker_receipts"), 1);
            for empty in ["events", "audit_log", "worker_receipt_projections"] {
                assert_eq!(count(&store, empty), 0, "{table}: {empty}");
            }
            store
                .with_connection(|db| {
                    assert_eq!(
                        db.query_row::<String, _, _>(
                            "SELECT state FROM action_journal",
                            [],
                            |row| row.get(0)
                        )?,
                        "dispatched"
                    );
                    db.execute_batch("DROP TRIGGER fail_projection;")?;
                    Ok(())
                })
                .unwrap();
            let (projected, _) = store
                .commit_worker_receipt_projection(task.clone(), &key)
                .unwrap()
                .unwrap();
            assert_eq!(projected.status, TaskStatus::Cancelled);
            assert_eq!(
                projected.actions[&binding.action_id].status,
                ActionStatus::Uncertain
            );
            assert_eq!(projected.tool_results[0].verdict, Verdict::Uncertain);
            assert!(
                !serde_json::to_string(&projected)
                    .unwrap()
                    .contains("private worker body")
            );
            assert!(
                store
                    .commit_worker_receipt_projection(task, &key)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(store.load_tasks(true).unwrap(), vec![projected]);
            for one in ["events", "audit_log", "worker_receipt_projections"] {
                assert_eq!(count(&store, one), 1);
            }
            store
                .checkpoint_audit(&MemorySecretStore::default())
                .unwrap();
        }
    }

    #[test]
    fn startup_reconciles_saved_receipts_once_even_after_their_private_body_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("receipt.db");
        let (store, mut task, _, binding) = fixture(&path);
        task.status = TaskStatus::Running;
        store.save_task(&mut task).unwrap();
        receive(&store, &binding, ReceiptKind::Response);
        store
            .with_connection(|db| {
                db.execute("DELETE FROM private_artifacts", [])?;
                Ok(())
            })
            .unwrap();
        assert_eq!(count(&store, "worker_receipts"), 1);
        drop(store);
        let reopened = LocalStore::open_encrypted(&path, &SecretBytes::new(vec![33; 32])).unwrap();
        let task = reopened.load_tasks(true).unwrap().pop().unwrap();
        assert_eq!(task.status, TaskStatus::Interrupted);
        assert_eq!(
            task.actions[&binding.action_id].status,
            ActionStatus::Uncertain
        );
        assert_eq!(task.tool_results.len(), 1);
        assert!(reopened.pending_worker_receipts(None).unwrap().is_empty());
        assert_eq!(count(&reopened, "private_artifacts"), 0);
        drop(reopened);
        let again = LocalStore::open_encrypted(&path, &SecretBytes::new(vec![33; 32])).unwrap();
        assert_eq!(again.load_tasks(true).unwrap(), vec![task]);
        assert_eq!(count(&again, "worker_receipt_projections"), 1);
        assert_eq!(count(&again, "events"), 1);
    }

    #[test]
    fn acknowledgement_is_not_effect_settlement_and_unsent_evidence_must_be_reconciled_before_retry()
     {
        let dir = tempfile::tempdir().unwrap();
        let (store, task, prepared, binding) = fixture(&dir.path().join("receipt.db"));
        let ack = receive(&store, &binding, ReceiptKind::CancelAcknowledged);
        let (task, _) = store
            .commit_worker_receipt_projection(task, &ack)
            .unwrap()
            .unwrap();
        assert!(task.tool_results.is_empty());
        assert_eq!(
            task.actions[&binding.action_id].status,
            ActionStatus::Verifying
        );
        let unsent = receive(&store, &binding, ReceiptKind::NeverSent);
        assert!(
            store.journal_prepared(&prepared).is_err(),
            "Pending evidence cannot be replaced by a new preparation"
        );
        let (task, _) = store
            .commit_worker_receipt_projection(task, &unsent)
            .unwrap()
            .unwrap();
        assert_eq!(task.tool_results[0].verdict, Verdict::Cancelled);
        assert_eq!(
            task.actions[&binding.action_id].status,
            ActionStatus::Failed
        );
        store.journal_prepared(&prepared).unwrap();
    }

    #[test]
    fn verification_covers_receipts_atomically_and_later_replies_do_not_demote_verified_effects() {
        let dir = tempfile::tempdir().unwrap();
        let (store, task, prepared, binding) = fixture(&dir.path().join("receipt.db"));
        let key = receive(&store, &binding, ReceiptKind::Response);
        store.with_connection(|db| { db.execute_batch("CREATE TEMP TRIGGER fail_projection BEFORE INSERT ON worker_receipt_projections BEGIN SELECT RAISE(ABORT,'fixture marker failure'); END;")?; Ok(()) }).unwrap();
        assert!(
            store
                .commit_verified_action(verified(&task, &prepared.intent.proposal))
                .is_err()
        );
        assert_eq!(store.load_tasks(true).unwrap(), vec![task.clone()]);
        assert_eq!(count(&store, "events"), 0);
        assert_eq!(count(&store, "audit_log"), 0);
        store
            .with_connection(|db| {
                db.execute_batch("DROP TRIGGER fail_projection;")?;
                Ok(())
            })
            .unwrap();
        let (task, _) = store
            .commit_verified_action(verified(&task, &prepared.intent.proposal))
            .unwrap();
        assert!(store.pending_worker_receipts(None).unwrap().is_empty());
        assert!(
            store
                .commit_worker_receipt_projection(task.clone(), &key)
                .unwrap()
                .is_none()
        );
        let late = receive(&store, &binding, ReceiptKind::Response);
        let (task, _) = store
            .commit_worker_receipt_projection(task, &late)
            .unwrap()
            .unwrap();
        assert_eq!(task.status, TaskStatus::Cancelled);
        assert_eq!(task.completed_count(), 1);
        assert_eq!(task.tool_results.len(), 1);
        assert_eq!(task.tool_results[0].verdict, Verdict::Confirmed);
    }

    #[test]
    fn an_earlier_receipt_cannot_change_a_replaced_action_identity() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mut task, _, binding) = fixture(&dir.path().join("receipt.db"));
        let key = receive(&store, &binding, ReceiptKind::Response);
        let action = task.actions.get_mut(&binding.action_id).unwrap();
        action.proposal.target_resource = "replacement identity".into();
        action.status = ActionStatus::Pending;
        store.save_task(&mut task).unwrap();
        let (task, _) = store
            .commit_worker_receipt_projection(task, &key)
            .unwrap()
            .unwrap();
        assert_eq!(
            task.actions[&binding.action_id].status,
            ActionStatus::Pending
        );
        assert!(task.tool_results.is_empty());
        assert!(store.pending_worker_receipts(None).unwrap().is_empty());
    }

    #[test]
    fn legacy_receipt_migration_preserves_metadata_when_artifacts_are_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _, _, binding) = fixture(&dir.path().join("receipt.db"));
        receive(&store, &binding, ReceiptKind::Response);
        store.with_connection(|db| {
            db.execute_batch("DROP TABLE worker_receipt_projections; CREATE TABLE legacy_receipts(request_id TEXT NOT NULL,kind TEXT NOT NULL,session_id TEXT NOT NULL,run_id TEXT NOT NULL,action_id TEXT NOT NULL,action_digest TEXT NOT NULL,artifact_id TEXT REFERENCES private_artifacts(id) ON DELETE CASCADE,received_at TEXT NOT NULL,PRIMARY KEY(request_id,kind)); INSERT INTO legacy_receipts SELECT * FROM worker_receipts; DROP TABLE worker_receipts; ALTER TABLE legacy_receipts RENAME TO worker_receipts;")?;
            Ok(())
        }).unwrap();
        store.migrate_journal().unwrap();
        store
            .with_connection(|db| {
                db.execute("DELETE FROM private_artifacts", [])?;
                Ok(())
            })
            .unwrap();
        assert_eq!(count(&store, "worker_receipts"), 1);
        assert_eq!(store.pending_worker_receipts(None).unwrap().len(), 1);
        store
            .with_connection(|db| {
                assert!(db.query_row::<bool, _, _>(
                    "SELECT artifact_id IS NULL FROM worker_receipts",
                    [],
                    |row| row.get(0)
                )?);
                let violations: i64 =
                    db.query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |row| {
                        row.get(0)
                    })?;
                assert_eq!(violations, 0);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn receipt_projection_exit_child() {
        let Ok(path) = std::env::var("SAGE_TEST_RECEIPT_DB") else {
            return;
        };
        let (store, task, _, binding) = fixture(std::path::Path::new(&path));
        let key = receive(&store, &binding, ReceiptKind::Response);
        crash_checkpoint("received");
        store.commit_worker_receipt_projection(task, &key).unwrap();
        panic!("Expected an abrupt exit at the receipt boundary");
    }

    #[test]
    fn abrupt_exit_recovers_unprojected_receipts_without_duplicate_state_or_events() {
        for stage in ["received", "task", "marker", "committed"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("receipt.db");
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "receipts::tests::receipt_projection_exit_child",
                    "--nocapture",
                ])
                .env("SAGE_TEST_RECEIPT_DB", &path)
                .env("SAGE_TEST_RECEIPT_EXIT_AT", stage)
                .output()
                .unwrap();
            assert_eq!(
                child.status.code(),
                Some(87),
                "{stage}: {}",
                String::from_utf8_lossy(&child.stderr)
            );
            // Inspect before startup repair to prove the transaction boundary.
            let db = crate::vault::open_encrypted(&path, &SecretBytes::new(vec![33; 32])).unwrap();
            for table in ["events", "audit_log", "worker_receipt_projections"] {
                let count: i64 = db
                    .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                        row.get(0)
                    })
                    .unwrap();
                assert_eq!(count, i64::from(stage == "committed"), "{stage}: {table}");
            }
            let raw: String = db
                .query_row("SELECT task_json FROM tasks", [], |row| row.get(0))
                .unwrap();
            assert_eq!(
                serde_json::from_str::<Task>(&raw)
                    .unwrap()
                    .tool_results
                    .len(),
                usize::from(stage == "committed")
            );
            drop(db);
            let reopened =
                LocalStore::open_encrypted(&path, &SecretBytes::new(vec![33; 32])).unwrap();
            let task = reopened.load_tasks(true).unwrap().pop().unwrap();
            assert_eq!(task.status, TaskStatus::Cancelled);
            assert_eq!(task.tool_results.len(), 1);
            assert_eq!(task.tool_results[0].verdict, Verdict::Uncertain);
            for table in ["events", "audit_log", "worker_receipt_projections"] {
                assert_eq!(count(&reopened, table), 1);
            }
            reopened
                .checkpoint_audit(&MemorySecretStore::default())
                .unwrap();
        }
    }
}
