//! One durable run closure: execution facts, cleanup, conversation, memory,
//! final events and audit commit together. External effects remain separate.
use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use serde_json::json;
use uuid::Uuid;

use crate::contracts::{DataLabel, ToolResult, Verdict, VerificationRecord};
use crate::domain::{ActionStatus, ExecutionFacts, Task, TaskStatus};
use crate::events::{CoreEvent, CoreEventKind};
use crate::redaction::redact_for_persistence;
use crate::storage::{LocalStore, write_audit, write_event, write_task};
use crate::{CoreError, CoreResult};

#[derive(Debug, Clone)]
pub(crate) enum Finish {
    Answer(String),
    Failure(String),
    Stop,
    Interrupted {
        summary: String,
        budget_exhausted: bool,
    },
}

impl Finish {
    pub(crate) fn redacted(mut self) -> Self {
        match &mut self {
            Self::Answer(text) | Self::Failure(text) | Self::Interrupted { summary: text, .. } => {
                *text = redact_for_persistence(text);
            }
            Self::Stop => {}
        }
        self
    }
    fn kind(&self) -> &'static str {
        match self {
            Self::Answer(_) => "answer",
            Self::Failure(_) => "failure",
            Self::Stop => "stop",
            Self::Interrupted { .. } => "interrupted",
        }
    }
    fn summary(&self) -> String {
        redact_for_persistence(match self {
            Self::Answer(text) | Self::Failure(text) => text,
            Self::Interrupted { summary, .. } => summary,
            Self::Stop => {
                "Stopped at your request. Completed and uncertain effects remain recorded."
            }
        })
    }
}

impl LocalStore {
    /// Initialization runs this after the knowledge schema is available, before
    /// loading the task cache. It repairs projections only, never tool effects.
    pub(crate) fn reconcile_run_closures(&self) -> CoreResult<()> {
        loop {
            let pending = self.with_connection(|db| {
                let mut query = db.prepare("SELECT task_json FROM tasks t WHERE status IN ('interrupted','cancelled') AND NOT EXISTS(SELECT 1 FROM undo_journal u WHERE u.task_id=t.id) AND NOT EXISTS(SELECT 1 FROM run_finalizations f WHERE f.task_id=t.id AND f.execution_attempt=COALESCE(json_extract(t.task_json,'$.execution_attempt'),0) AND (t.status!='cancelled' OR f.kind='stop')) ORDER BY t.id LIMIT 128")?;
                let rows = query.query_map([], |row|row.get::<_,String>(0))?.collect::<Result<Vec<_>,_>>()?;
                rows.into_iter().map(|json|serde_json::from_str::<Task>(&json).map_err(Into::into)).collect::<Result<Vec<_>,_>>()
            })?;
            if pending.is_empty() {
                return Ok(());
            }
            for task in pending {
                let finish = if task.status == TaskStatus::Cancelled {
                    Finish::Stop
                } else {
                    Finish::Interrupted {
                    summary: "Run interrupted. Review its saved execution evidence before continuing.".into(), budget_exhausted: task.budget_exhausted,
                }
                };
                self.finalize_run(task, finish)?;
            }
        }
    }

    pub(crate) fn finalize_run(
        &self,
        source: Task,
        mut finish: Finish,
    ) -> CoreResult<(Task, Vec<CoreEvent>)> {
        let _retention = self
            .retention_lock
            .lock()
            .map_err(|_| CoreError::Storage("Retention lock poisoned".into()))?;
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let saved: String = tx.query_row("SELECT task_json FROM tasks WHERE id=?1", [source.id.to_string()], |row| row.get(0))?;
            let current: Task = serde_json::from_str(&saved)?;
            if current.execution_attempt != source.execution_attempt {
                return Err("Finalization belongs to a retired execution attempt".into());
            }
            let stopped: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM control_scopes WHERE id=?1 AND stopped_at IS NOT NULL)", [current.control_scope().to_string()], |row| row.get(0))?;
            if stopped || current.status == TaskStatus::Cancelled { finish = Finish::Stop; }
            let closed: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM run_finalizations WHERE task_id=?1 AND execution_attempt=?2 AND (?3=0 OR kind='stop'))", params![source.id.to_string(),source.execution_attempt,matches!(finish,Finish::Stop)], |row| row.get(0))?;
            if closed || (current.status.is_terminal() && current.status != TaskStatus::Cancelled) {
                // A post-commit failure or late completion cannot downgrade a
                // durable result, nor append a second final response.
                return Ok((current, Vec::new()));
            }
            if current.revision != source.revision { return Err("The task changed before finalization; reload its state".into()); }
            let mut task = source;
            let retired = crate::knowledge::outcome_retired(&tx, task.id)?;
            let summary = if retired {
                "The run ended. Returned content was not retained after related data was retired.".to_string()
            } else { finish.summary() };
            let mut events = normalize_effects(&tx, &mut task, &finish, &summary)?;
            #[cfg(test)] crash_checkpoint("effects");
            let facts = ExecutionFacts::for_task(&task);
            task.status = match &finish {
                Finish::Answer(_) => facts.answer_status(),
                Finish::Failure(_) if facts.uncertain > 0 => TaskStatus::Interrupted,
                Finish::Failure(_) if facts.verified > 0 => TaskStatus::Partial,
                Finish::Failure(_) => TaskStatus::Failed,
                Finish::Stop => TaskStatus::Cancelled,
                Finish::Interrupted { budget_exhausted, .. } => {
                    task.budget_exhausted |= *budget_exhausted;
                    TaskStatus::Interrupted
                }
            };
            let outcome = if matches!(finish,Finish::Answer(_)) && !matches!(task.status,TaskStatus::Succeeded|TaskStatus::Answered) {
                facts.summary()
            } else { summary };
            task.final_outcome = Some(outcome.clone());
            task.touch();
            close_authority(&tx, &task, &mut events)?;
            #[cfg(test)] crash_checkpoint("cleanup");
            task.revision = write_task(&tx, &task)?;
            #[cfg(test)] crash_checkpoint("task");
            crate::knowledge::write_outcome(&tx, &task)?;
            events.push(CoreEvent::new(Some(task.id),CoreEventKind::TaskStatusChanged { status: task.status, summary: if matches!(finish,Finish::Answer(_)) { facts.summary() } else { outcome.clone() } }));
            match &finish {
                Finish::Answer(_) => {
                    events.push(CoreEvent::new(Some(task.id),CoreEventKind::ModelResponse { text:outcome.clone(),finished:true }));
                    events.push(CoreEvent::new(Some(task.id),CoreEventKind::TaskCompleted { outcome }));
                }
                Finish::Failure(_) => events.push(CoreEvent::new(Some(task.id),CoreEventKind::Error { code:"task_failed".into(),message:outcome,recoverable:false })),
                _ => {}
            }
            for event in &events { write_event(&tx,event)?; }
            write_audit(&tx,Some(task.id),None,"run_finalized",&json!({"attempt":task.execution_attempt,"kind":finish.kind(),"status":task.status,"facts":facts,"revision":task.revision,"events":events.iter().map(|event|event.id).collect::<Vec<_>>()}))?;
            #[cfg(test)] crash_checkpoint("records");
            tx.execute("INSERT INTO run_finalizations VALUES(?1,?2,?3,?4,?5)", params![task.id.to_string(),task.execution_attempt,finish.kind(),task.revision,Utc::now().to_rfc3339()])?;
            #[cfg(test)] crash_checkpoint("marker");
            tx.commit()?;
            #[cfg(test)] crash_checkpoint("committed");
            Ok((task,events))
        })
    }
}

fn normalize_effects(
    db: &rusqlite::Transaction<'_>,
    task: &mut Task,
    finish: &Finish,
    summary: &str,
) -> CoreResult<Vec<CoreEvent>> {
    let mut events = Vec::new();
    for (id, action) in &mut task.actions {
        if task
            .intent
            .as_ref()
            .is_some_and(|intent| intent.retired.contains(id))
        {
            continue;
        }
        if task.undone_actions.contains(id) {
            continue;
        }
        let journal: Option<(String,String,Option<String>)> = db.query_row("SELECT state,action_digest,verification_json FROM action_journal WHERE run_id=?1 AND action_id=?2", params![task.id.to_string(),id.to_string()], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?))).optional()?;
        let result = task
            .tool_results
            .iter()
            .rev()
            .find(|result| result.action_id == *id);
        let confirmed = action.status == ActionStatus::Succeeded
            && result.is_some_and(|result| result.verdict == Verdict::Confirmed)
            && journal.as_ref().is_some_and(|(state, digest, proof)| {
                state == "confirmed"
                    && crate::policy::approval_digest(&action.proposal)
                        .is_ok_and(|current| current == *digest)
                    && proof
                        .as_ref()
                        .and_then(|proof| serde_json::from_str::<VerificationRecord>(proof).ok())
                        .is_some_and(|record| {
                            record.run_id == task.id
                                && record.action_id == *id
                                && record.action_digest == *digest
                                && record.verdict == Verdict::Confirmed
                                && record.expected == action.proposal.expected_outcome
                                && record.target == action.proposal.target_resource
                        })
            });
        if confirmed {
            continue;
        }
        let may_have_executed = match journal.as_ref().map(|(state, _, _)| state.as_str()) {
            Some("dispatched" | "uncertain" | "confirmed") => true,
            Some("prepared" | "failed" | "cancelled") => false,
            None => matches!(
                action.status,
                ActionStatus::Running
                    | ActionStatus::Verifying
                    | ActionStatus::Uncertain
                    | ActionStatus::Succeeded
            ),
            _ => {
                return Err(CoreError::Storage(
                    "Unknown action journal state at finalization".into(),
                ));
            }
        };
        let next = if may_have_executed {
            ActionStatus::Uncertain
        } else if matches!(finish, Finish::Failure(_) | Finish::Stop)
            && !matches!(action.status, ActionStatus::Failed | ActionStatus::Skipped)
        {
            ActionStatus::Skipped
        } else if matches!(
            action.status,
            ActionStatus::Compiling
                | ActionStatus::WaitingForApproval
                | ActionStatus::Running
                | ActionStatus::Verifying
        ) {
            ActionStatus::Failed
        } else {
            action.status
        };
        if next == action.status
            && (next != ActionStatus::Uncertain
                || result.is_some_and(|result| result.verdict == Verdict::Uncertain))
        {
            continue;
        }
        action.status = next;
        action.error = Some(summary.into());
        let verdict = if may_have_executed {
            Verdict::Uncertain
        } else if matches!(finish, Finish::Stop) {
            Verdict::Cancelled
        } else {
            Verdict::Failed
        };
        task.tool_results.push(ToolResult {
            action_id: *id,
            tool: "run_cleanup".into(),
            verdict,
            summary: if may_have_executed {
                "An action may have executed. Review its current state before continuing."
            } else {
                "This action did not complete before the run ended."
            }
            .into(),
            output: json!({}),
            label: DataLabel::private(task.id, id.to_string()),
            observed_at: Utc::now(),
        });
        db.execute("UPDATE action_journal SET state=?3,updated_at=?4 WHERE run_id=?1 AND action_id=?2 AND state IN ('prepared','dispatched','uncertain')",params![task.id.to_string(),id.to_string(),if may_have_executed {"uncertain"} else {"failed"},Utc::now().to_rfc3339()])?;
        events.push(CoreEvent::new(
            Some(task.id),
            CoreEventKind::ActionFailed {
                action_id: *id,
                error: summary.into(),
            },
        ));
    }
    Ok(events)
}

fn close_authority(
    db: &rusqlite::Transaction<'_>,
    task: &Task,
    events: &mut Vec<CoreEvent>,
) -> CoreResult<()> {
    let mut query = db.prepare("SELECT id FROM decisions WHERE task_id=?1 AND state='pending'")?;
    let ids = query
        .query_map([task.id.to_string()], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    drop(query);
    for id in ids {
        db.execute("UPDATE decisions SET state='cancelled',resolution_json=?2,resolved_at=?3 WHERE id=?1 AND state='pending'",params![id,serde_json::to_string(&crate::decisions::DecisionResolution::Cancelled)?,Utc::now().to_rfc3339()])?;
        events.push(CoreEvent::new(
            Some(task.id),
            CoreEventKind::DecisionResolved {
                decision_id: Uuid::parse_str(&id)
                    .map_err(|_| CoreError::Storage("Invalid decision identity".into()))?,
                state: "cancelled".into(),
            },
        ));
    }
    db.execute(
        "UPDATE capabilities SET revoked_at=?2 WHERE task_id=?1 AND revoked_at IS NULL",
        params![task.id.to_string(), Utc::now().to_rfc3339()],
    )?;
    db.execute("UPDATE approvals SET status='cancelled',resolved_at=?2 WHERE task_id=?1 AND status='pending'",params![task.id.to_string(),Utc::now().to_rfc3339()])?;
    db.execute(
        "DELETE FROM settings WHERE key=?1",
        [format!("pending-approval.{}", task.id)],
    )?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn crash_checkpoint(stage: &str) {
    if std::env::var("SAGE_TEST_FINALIZATION_EXIT_AT")
        .ok()
        .as_deref()
        == Some(stage)
    {
        std::process::exit(90);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::PreparedAction;
    use crate::decisions::{DecisionRecord, QuestionRecord};
    use crate::domain::{
        Action, ActionProposal, ActionState, ExpectedOutcome, Provenance, ProvenanceSource,
    };
    use crate::execution::ExecutionReceipt;
    use crate::knowledge::{MemoryKind, MemoryRecord, Message};
    use crate::observation::{Evidence, Observation};
    use crate::secrets::SecretBytes;
    use crate::transitions::{VerificationMode, VerifiedAction};
    use std::collections::BTreeMap;

    fn fixture(path: &std::path::Path, unsettled: bool) -> (LocalStore, Task) {
        let store = LocalStore::open_encrypted(path, &SecretBytes::new(vec![79; 32])).unwrap();
        store.migrate_knowledge().unwrap();
        let mut task = Task::new("Finalize the synthetic workflow");
        task.status = TaskStatus::Running;
        let conversation = store.ensure_conversation(None, &task.request).unwrap();
        task.conversation_id = Some(conversation.id);
        task.message_id = Some(Uuid::new_v4());
        let proposal = ActionProposal {
            id: Uuid::new_v4(),
            task_id: task.id,
            action: Action::AskUser {
                question: "Fixture response".into(),
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: "user".into(),
            provenance: Provenance::user(),
            metadata: Default::default(),
        };
        task.actions.insert(
            proposal.id,
            ActionState {
                proposal: proposal.clone(),
                status: ActionStatus::Running,
                attempts: 1,
                summary: None,
                error: None,
            },
        );
        store.save_task(&mut task).unwrap();
        store
            .append_message(&Message {
                id: task.message_id.unwrap(),
                conversation_id: conversation.id,
                task_id: Some(task.id),
                role: "user".into(),
                content: task.request.clone(),
                provenance: Provenance::user(),
                created_at: task.created_at,
            })
            .unwrap();
        store.link_task(&task).unwrap();
        let prepared = PreparedAction::new(&proposal, Default::default()).unwrap();
        store.journal_prepared(&prepared).unwrap();
        store.journal_dispatch(&prepared, None).unwrap();
        let receipt = ExecutionReceipt {
            executor: "fixture".into(),
            summary: "Verified fixture response".into(),
            transient_data: json!({}),
            rollback: None,
        };
        let observation = Observation {
            observed_at: Utc::now(),
            provenance: Provenance::external(ProvenanceSource::OperatingSystem, "fixture"),
            summary: "Fixture observed".into(),
            evidence: vec![Evidence::UserAnswer { received: true }],
        };
        task = store
            .commit_verified_action(
                VerifiedAction::from_observation(
                    &task,
                    &proposal,
                    &receipt,
                    &observation,
                    VerificationMode::Execution,
                )
                .unwrap(),
            )
            .unwrap()
            .0;
        if unsettled {
            let mut pending = proposal.clone();
            pending.id = Uuid::new_v4();
            task.actions.insert(
                pending.id,
                ActionState {
                    proposal: pending.clone(),
                    status: ActionStatus::Running,
                    attempts: 1,
                    summary: None,
                    error: None,
                },
            );
            store.save_task(&mut task).unwrap();
            let prepared = PreparedAction::new(&pending, Default::default()).unwrap();
            store.journal_prepared(&prepared).unwrap();
            store.journal_dispatch(&prepared, None).unwrap();
        }
        // Synthetic leftover prompts/authority exercise aggregate cleanup.
        let decision = DecisionRecord::Question(QuestionRecord {
            question_id: Uuid::new_v4(),
            task_id: task.id,
            action_id: proposal.id,
            question: "Leftover fixture prompt".into(),
            expires_at: Utc::now() + chrono::Duration::hours(1),
        });
        task.status = TaskStatus::WaitingForUser;
        store
            .open_decision(&mut task, &decision, &decision.opened())
            .unwrap();
        task.status = TaskStatus::Running;
        store.save_task(&mut task).unwrap();
        let source = store
            .save_memory(MemoryRecord {
                id: Uuid::new_v4(),
                kind: MemoryKind::Preference,
                subject: "Fixture preference".into(),
                content: "Use a compact fixture layout".into(),
                confidence: 1.0,
                provenance: Provenance::user(),
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                metadata: Default::default(),
            })
            .unwrap();
        store.record_context_memories(task.id, &[source]).unwrap();
        store
            .save_setting(&format!("pending-approval.{}", task.id), &true)
            .unwrap();
        store
            .with_connection(|db| {
                db.execute(
                    "INSERT INTO capabilities VALUES(?1,?2,?3,'{}',?4,?4,NULL)",
                    params![
                        Uuid::new_v4().to_string(),
                        task.id.to_string(),
                        proposal.id.to_string(),
                        Utc::now().to_rfc3339()
                    ],
                )?;
                db.execute(
                    "INSERT INTO approvals VALUES(?1,?2,?3,'fixture','pending',?4,NULL)",
                    params![
                        Uuid::new_v4().to_string(),
                        task.id.to_string(),
                        proposal.id.to_string(),
                        Utc::now().to_rfc3339()
                    ],
                )?;
                Ok(())
            })
            .unwrap();
        (store, task)
    }

    fn dump(store: &LocalStore) -> BTreeMap<String, Vec<Vec<String>>> {
        store
            .with_connection(|db| {
                let mut dump = BTreeMap::new();
                for table in [
                    "tasks",
                    "actions",
                    "action_journal",
                    "events",
                    "audit_log",
                    "decisions",
                    "approvals",
                    "capabilities",
                    "settings",
                    "messages",
                    "conversations",
                    "working_memory",
                    "memory",
                    "memory_records",
                    "memory_lineage",
                    "memory_fts",
                    "run_finalizations",
                ] {
                    let mut query = db.prepare(&format!("SELECT * FROM {table} ORDER BY rowid"))?;
                    let count = query.column_count();
                    let rows = query
                        .query_map([], |row| {
                            (0..count)
                                .map(|column| {
                                    row.get::<_, rusqlite::types::Value>(column)
                                        .map(|value| format!("{value:?}"))
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })?
                        .collect::<Result<Vec<_>, _>>()?;
                    dump.insert(table.into(), rows);
                }
                Ok(dump)
            })
            .unwrap()
    }

    fn assert_closed(store: &LocalStore, task: &Task) {
        assert_eq!(store.load_tasks(true).unwrap()[0], *task);
        store
            .with_connection(|db| {
                for sql in [
                    "SELECT COUNT(*) FROM decisions WHERE state='pending'",
                    "SELECT COUNT(*) FROM approvals WHERE status='pending'",
                    "SELECT COUNT(*) FROM capabilities WHERE revoked_at IS NULL",
                    "SELECT COUNT(*) FROM settings WHERE key LIKE 'pending-approval.%'",
                ] {
                    assert_eq!(db.query_row(sql, [], |row| row.get::<_, i64>(0))?, 0);
                }
                Ok(())
            })
            .unwrap();
        let messages = store.messages(task.conversation_id.unwrap(), 100).unwrap();
        let replies = messages
            .iter()
            .filter(|message| message.role == "assistant")
            .collect::<Vec<_>>();
        assert_eq!(replies.len(), 1);
        assert_eq!(&replies[0].content, task.final_outcome.as_ref().unwrap());
        let summary: Vec<serde_json::Value> =
            serde_json::from_str(&store.conversations().unwrap()[0].summary).unwrap();
        assert_eq!(summary.len(), 1);
        assert_eq!(
            summary[0]["status"],
            serde_json::to_value(task.status).unwrap()
        );
    }

    #[test]
    fn completion_is_atomic_idempotent_and_cannot_be_downgraded_by_a_late_error() {
        let dir = tempfile::tempdir().unwrap();
        let (store, task) = fixture(&dir.path().join("state.db"), false);
        let (done, events) = store
            .finalize_run(task.clone(), Finish::Answer("Fixture done".into()))
            .unwrap();
        assert_eq!(done.status, TaskStatus::Succeeded);
        assert_closed(&store, &done);
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, CoreEventKind::TaskCompleted { .. }))
        );
        assert!(
            store
                .memories(None, true, 10)
                .unwrap()
                .iter()
                .any(|memory| memory.id == done.id)
        );
        let before = dump(&store);
        for finish in [
            Finish::Answer("Duplicate body".into()),
            Finish::Failure("Late checkpoint failure".into()),
            Finish::Stop,
        ] {
            let (repeated, events) = store.finalize_run(task.clone(), finish).unwrap();
            assert_eq!(repeated, done);
            assert!(events.is_empty());
            assert_eq!(dump(&store), before);
        }
    }

    #[test]
    fn stop_closes_all_authority_and_preserves_uncertain_effects_and_verified_results() {
        let dir = tempfile::tempdir().unwrap();
        let (store, task) = fixture(&dir.path().join("state.db"), true);
        let (stopped, _) = store.finalize_run(task.clone(), Finish::Stop).unwrap();
        assert_eq!(stopped.status, TaskStatus::Cancelled);
        assert_closed(&store, &stopped);
        let facts = ExecutionFacts::for_task(&stopped);
        assert_eq!((facts.verified, facts.uncertain), (1, 1));
        let (late, events) = store
            .finalize_run(task, Finish::Answer("Unsupported completion".into()))
            .unwrap();
        assert_eq!(late, stopped);
        assert!(events.is_empty());
        assert!(
            store
                .memories(None, true, 10)
                .unwrap()
                .iter()
                .all(|memory| memory.id != stopped.id)
        );
    }

    #[test]
    fn projection_claims_without_matching_verification_cannot_finalize_success() {
        for corruption in ["missing", "changed_digest", "changed_proof"] {
            let dir = tempfile::tempdir().unwrap();
            let (store, task) = fixture(&dir.path().join("state.db"), false);
            store
                .with_connection(|db| {
                    db.execute_batch(match corruption {
                        "missing" => "DELETE FROM action_journal",
                        "changed_digest" => "UPDATE action_journal SET action_digest='changed'",
                        _ => "UPDATE action_journal SET verification_json='{}'",
                    })?;
                    Ok(())
                })
                .unwrap();
            let (done, _) = store
                .finalize_run(
                    task,
                    Finish::Answer("Everything finished successfully".into()),
                )
                .unwrap();
            assert_eq!(done.status, TaskStatus::Interrupted);
            assert_eq!(ExecutionFacts::for_task(&done).verified, 0);
            assert_eq!(ExecutionFacts::for_task(&done).uncertain, 1);
            assert!(
                !done
                    .final_outcome
                    .as_ref()
                    .unwrap()
                    .contains("Everything finished")
            );
            assert_closed(&store, &done);
        }
    }

    #[test]
    fn resuming_replaces_the_old_outcome_and_rejects_an_old_attempt() {
        let dir = tempfile::tempdir().unwrap();
        let (store, task) = fixture(&dir.path().join("state.db"), false);
        let (mut paused, _) = store
            .finalize_run(
                task,
                Finish::Interrupted {
                    summary: "Fixture paused".into(),
                    budget_exhausted: false,
                },
            )
            .unwrap();
        let stale = paused.clone();
        let prior_message = store
            .messages(paused.conversation_id.unwrap(), 10)
            .unwrap()
            .into_iter()
            .find(|m| m.role == "assistant")
            .unwrap()
            .id;
        paused.execution_attempt += 1;
        paused.status = TaskStatus::Running;
        store.save_task(&mut paused).unwrap();
        assert!(
            store
                .finalize_run(stale, Finish::Answer("Old attempt".into()))
                .is_err()
        );
        let (done, _) = store
            .finalize_run(paused, Finish::Answer("Recovered fixture".into()))
            .unwrap();
        assert_closed(&store, &done);
        assert_eq!(done.status, TaskStatus::Succeeded);
        assert_eq!(
            store
                .messages(done.conversation_id.unwrap(), 10)
                .unwrap()
                .into_iter()
                .find(|m| m.role == "assistant")
                .unwrap()
                .id,
            prior_message
        );
        assert_eq!(done.final_outcome.as_deref(), Some("Recovered fixture"));
    }

    #[test]
    fn every_write_failure_rolls_back_the_aggregate_projection() {
        for (table, operation, unsettled) in [
            ("action_journal", "UPDATE", true),
            ("decisions", "UPDATE", false),
            ("approvals", "UPDATE", false),
            ("capabilities", "UPDATE", false),
            ("settings", "DELETE", false),
            ("tasks", "INSERT", false),
            ("actions", "INSERT", false),
            ("messages", "INSERT", false),
            ("working_memory", "INSERT", false),
            ("conversations", "UPDATE", false),
            ("memory", "INSERT", false),
            ("memory_records", "INSERT", false),
            ("memory_lineage", "INSERT", false),
            ("events", "INSERT", false),
            ("audit_log", "INSERT", false),
            ("run_finalizations", "INSERT", false),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let (store, task) = fixture(&dir.path().join("state.db"), unsettled);
            let before = dump(&store);
            store.with_connection(|db|{db.execute_batch(&format!("CREATE TEMP TRIGGER fail_finalization BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'fixture finalization failure'); END;"))?;Ok(())}).unwrap();
            let finish = if unsettled {
                Finish::Stop
            } else {
                Finish::Answer("Fixture completion".into())
            };
            assert!(store.finalize_run(task.clone(), finish).is_err(), "{table}");
            assert_eq!(dump(&store), before, "{table}");
            store
                .with_connection(|db| {
                    db.execute_batch("DROP TRIGGER fail_finalization")?;
                    Ok(())
                })
                .unwrap();
            let (done, _) = store
                .finalize_run(
                    task,
                    if unsettled {
                        Finish::Stop
                    } else {
                        Finish::Answer("Fixture completion".into())
                    },
                )
                .unwrap();
            assert_closed(&store, &done);
        }
    }

    #[test]
    fn forgotten_content_is_not_recreated_by_the_final_response_or_events() {
        let dir = tempfile::tempdir().unwrap();
        let (store, task) = fixture(&dir.path().join("state.db"), false);
        store
            .with_connection(|db| {
                db.execute(
                    "INSERT INTO retired_task_data VALUES(?1,?2)",
                    params![task.id.to_string(), Utc::now().to_rfc3339()],
                )?;
                Ok(())
            })
            .unwrap();
        let (done, events) = store
            .finalize_run(task, Finish::Answer("late private orchid text".into()))
            .unwrap();
        assert!(!serde_json::to_string(&done).unwrap().contains("orchid"));
        assert!(!serde_json::to_string(&events).unwrap().contains("orchid"));
        assert!(
            store
                .messages(done.conversation_id.unwrap(), 10)
                .unwrap()
                .iter()
                .all(|m| m.role == "user")
        );
        assert!(
            store
                .memories(None, true, 10)
                .unwrap()
                .iter()
                .all(|m| m.id != done.id)
        );
    }

    #[test]
    fn a_late_receipt_updates_the_closed_conversation_in_the_same_projection() {
        let dir = tempfile::tempdir().unwrap();
        let (store, task) = fixture(&dir.path().join("state.db"), true);
        let (stopped, _) = store.finalize_run(task, Finish::Stop).unwrap();
        let proposal = &stopped
            .actions
            .values()
            .find(|state| state.status == ActionStatus::Uncertain)
            .unwrap()
            .proposal;
        let binding = crate::execution::bridge::EffectBinding {
            task_id: stopped.id,
            action_id: proposal.id,
            action_digest: crate::policy::approval_digest(proposal).unwrap(),
        };
        let key = crate::receipts::ReceiptKey {
            request_id: Uuid::new_v4().to_string(),
            kind: crate::receipts::ReceiptKind::Response,
        };
        store
            .record_worker_receipt(
                &key.request_id,
                "fixture",
                &binding,
                "response",
                b"untrusted body",
            )
            .unwrap();
        let (changed, _) = store
            .commit_worker_receipt_projection(stopped, &key)
            .unwrap()
            .unwrap();
        assert_closed(&store, &changed);
        assert_eq!(changed.status, TaskStatus::Cancelled);
        assert!(
            !changed
                .final_outcome
                .as_ref()
                .unwrap()
                .contains("untrusted body")
        );
    }

    #[test]
    fn abrupt_finalization_exit_child() {
        let Some(path) = std::env::var_os("SAGE_TEST_FINALIZATION_DB") else {
            return;
        };
        let stop = std::env::var("SAGE_TEST_FINALIZATION_KIND").unwrap() == "stop";
        let (store, task) = fixture(std::path::Path::new(&path), stop);
        if stop {
            store.stop_control_scope(task.control_scope()).unwrap();
        }
        store
            .finalize_run(
                task,
                if stop {
                    Finish::Stop
                } else {
                    Finish::Answer("Fixture durable answer".into())
                },
            )
            .unwrap();
        panic!("Expected abrupt process exit");
    }

    #[test]
    fn abrupt_exit_preserves_all_or_none_then_startup_closes_unfinished_runs() {
        for kind in ["answer", "stop"] {
            for stage in [
                "effects",
                "cleanup",
                "task",
                "message",
                "context",
                "memory",
                "records",
                "marker",
                "committed",
            ] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("state.db");
                let child = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "finalization::tests::abrupt_finalization_exit_child",
                        "--nocapture",
                    ])
                    .env("SAGE_TEST_FINALIZATION_DB", &path)
                    .env("SAGE_TEST_FINALIZATION_KIND", kind)
                    .env("SAGE_TEST_FINALIZATION_EXIT_AT", stage)
                    .output()
                    .unwrap();
                assert_eq!(
                    child.status.code(),
                    Some(90),
                    "{kind}/{stage}: {}",
                    String::from_utf8_lossy(&child.stderr)
                );
                let key = SecretBytes::new(vec![79; 32]);
                let committed = stage == "committed";
                let db = crate::vault::open_encrypted(&path, &key).unwrap();
                let markers: i64 = db
                    .query_row("SELECT COUNT(*) FROM run_finalizations", [], |row| {
                        row.get(0)
                    })
                    .unwrap();
                assert_eq!(markers, i64::from(committed));
                let task: String = db
                    .query_row("SELECT task_json FROM tasks", [], |row| row.get(0))
                    .unwrap();
                let task: Task = serde_json::from_str(&task).unwrap();
                assert_eq!(
                    task.status,
                    if !committed {
                        TaskStatus::Running
                    } else if kind == "stop" {
                        TaskStatus::Cancelled
                    } else {
                        TaskStatus::Succeeded
                    }
                );
                let replies: i64 = db
                    .query_row(
                        "SELECT COUNT(*) FROM messages WHERE role='assistant'",
                        [],
                        |row| row.get(0),
                    )
                    .unwrap();
                assert_eq!(replies, i64::from(committed));
                let live: i64 = db
                    .query_row(
                        "SELECT COUNT(*) FROM capabilities WHERE revoked_at IS NULL",
                        [],
                        |row| row.get(0),
                    )
                    .unwrap();
                assert_eq!(live, i64::from(!committed));
                let decision: String = db
                    .query_row("SELECT state FROM decisions", [], |row| row.get(0))
                    .unwrap();
                assert_eq!(decision, if committed { "cancelled" } else { "pending" });
                drop(db);
                let store = LocalStore::open_encrypted(&path, &key).unwrap();
                store.migrate_knowledge().unwrap();
                let recovered = store.load_tasks(true).unwrap().remove(0);
                assert_closed(&store, &recovered);
                assert_eq!(
                    recovered.status,
                    if kind == "stop" {
                        TaskStatus::Cancelled
                    } else if committed {
                        TaskStatus::Succeeded
                    } else {
                        TaskStatus::Interrupted
                    }
                );
                if kind == "stop" {
                    assert_eq!(ExecutionFacts::for_task(&recovered).uncertain, 1);
                }
                let after = dump(&store);
                store.reconcile_run_closures().unwrap();
                assert_eq!(dump(&store), after);
            }
        }
    }
}
