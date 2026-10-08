//! Local intent revisions retain exact action identities and dependency edges.
//! Retired work remains in the encrypted journal; a new sentence cannot erase
//! an effect, revive a capability, or reset the run's cumulative step budget.
use std::collections::BTreeSet;

use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use serde_json::json;
use uuid::Uuid;

use crate::domain::{Action, ActionState, ActionStatus, Task, TaskStatus};
use crate::events::{CoreEvent, CoreEventKind};
use crate::intent::{CompiledIntent, IntentBinding, IntentState};
use crate::storage::{LocalStore, write_audit, write_event, write_task};
use crate::{CoreError, CoreResult};

pub(crate) struct Revision {
    pub task: Task,
    pub retired: BTreeSet<Uuid>,
    pub kept: usize,
    pub added: usize,
}

pub(crate) fn prepare(
    current: &Task,
    intent: &CompiledIntent,
    request: String,
) -> CoreResult<Revision> {
    let previous = current
        .intent
        .as_ref()
        .ok_or_else(|| CoreError::InvalidAction("This run has no local intent bindings".into()))?;
    let contract = current.contract.as_ref().ok_or(CoreError::Cancelled)?;
    contract.validate(current.id)?;
    if !current.status.is_active() || current.undo.is_some() || previous.revision >= 64 {
        return Err(CoreError::InvalidAction(
            "This run cannot accept another intent revision".into(),
        ));
    }
    let mut task = current.clone();
    let mut steps: Vec<IntentBinding> = Vec::with_capacity(intent.steps.len());
    let mut used = BTreeSet::new();
    let mut kept = 0;
    let graph = intent.graph(task.id, request.clone());
    for (index, step) in intent.steps.iter().enumerate() {
        let dependencies: BTreeSet<_> = step
            .dependencies
            .iter()
            .map(|i| steps[*i].action_id)
            .collect();
        let retained = previous.steps.iter().find(|old| {
            old.action == step.action
                && !used.contains(&old.action_id)
                && current.actions.get(&old.action_id).is_some_and(|state| {
                    (old.dependencies == dependencies || state.status == ActionStatus::Pending)
                        && !matches!(
                            state.status,
                            ActionStatus::Failed | ActionStatus::Skipped | ActionStatus::Uncertain
                        )
                })
        });
        let action_id = if let Some(old) = retained {
            kept += 1;
            // An unstarted D remains the same step when A-B-C-D becomes
            // A-B-E-D. Only its wait edges change; no prepared authority or
            // observed result is carried across a changed dependency.
            task.dependencies
                .insert(old.action_id, dependencies.clone());
            old.action_id
        } else {
            let proposal = graph.nodes[index].proposal.clone();
            let id = proposal.id;
            task.actions.insert(
                id,
                ActionState {
                    proposal,
                    status: ActionStatus::Pending,
                    attempts: 0,
                    summary: None,
                    error: None,
                },
            );
            task.dependencies.insert(id, dependencies.clone());
            id
        };
        used.insert(action_id);
        steps.push(IntentBinding {
            action_id,
            action: step.action.clone(),
            dependencies,
        });
    }
    if task.actions.len() > contract.max_steps as usize {
        return Err(CoreError::PermissionRequired("This correction exceeds the run's cumulative step budget. Start a new request after reviewing its effects.".into()));
    }
    let retired: BTreeSet<_> = previous
        .steps
        .iter()
        .map(|step| step.action_id)
        .filter(|id| !used.contains(id))
        .collect();
    for id in &retired {
        let state = task.actions.get_mut(id).ok_or(CoreError::Cancelled)?;
        let read = matches!(
            state.proposal.action,
            Action::ReadFile { .. } | Action::ListDirectory { .. }
        );
        if !read
            && matches!(
                state.status,
                ActionStatus::Running
                    | ActionStatus::Verifying
                    | ActionStatus::Succeeded
                    | ActionStatus::Uncertain
            )
        {
            return Err(CoreError::PermissionRequired("This correction changes an action that may already have changed the device. Review or undo that effect before replacing it; the run remains paused.".into()));
        }
        // Completed reads remain historical evidence. Unfinished reads have no
        // mutation to settle and their owned futures will be dropped promptly.
        if state.status != ActionStatus::Succeeded {
            state.status = ActionStatus::Skipped;
            state.summary = Some("Replaced by your updated request.".into());
            state.error = None;
        }
    }
    let added = steps.len() - kept;
    let summary = format!(
        "Request updated: kept {kept} steps, added {added}, replaced {}.",
        retired.len()
    );
    task.intent = Some(IntentState {
        revision: previous.revision + 1,
        steps,
        retired: previous.retired.union(&retired).copied().collect(),
        change_summary: summary,
    });
    task.request = request.clone();
    task.goal = Some(request);
    task.final_outcome = None;
    // Preserve a still-valid decision for an unchanged step. The current
    // workflow has serial approval admission, so there is at most one.
    task.status = if task
        .current_actions()
        .any(|(_, state)| state.status == ActionStatus::WaitingForApproval)
    {
        TaskStatus::WaitingForApproval
    } else {
        TaskStatus::Running
    };
    task.touch();
    Ok(Revision {
        task,
        retired,
        kept,
        added,
    })
}

impl LocalStore {
    /// The revised graph, closed decisions, user text, receipt and audit event
    /// commit atomically. Failure leaves the original run held and unchanged.
    pub(crate) fn accept_revision(
        &self,
        prior: &Task,
        revision: &mut Revision,
        key: &crate::commands::SubmissionKey,
    ) -> CoreResult<Vec<CoreEvent>> {
        let task = &mut revision.task;
        let mut events = vec![CoreEvent::new(
            Some(task.id),
            CoreEventKind::TaskStatusChanged {
                status: task.status,
                summary: task
                    .intent
                    .as_ref()
                    .expect("bound revision")
                    .change_summary
                    .clone(),
            },
        )];
        task.revision = self.with_connection(|db| {
            let tx = db.transaction()?;
            let stopped: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM control_scopes WHERE id=?1 AND stopped_at IS NOT NULL)", [task.control_scope().to_string()], |row| row.get(0))?;
            if stopped { return Err(CoreError::Cancelled.into()); }
            for id in &revision.retired {
                // A stale projection cannot turn a dispatched mutation into
                // an unexecuted step. The journal remains authoritative.
                let journal: Option<String> = tx.query_row("SELECT state FROM action_journal WHERE run_id=?1 AND action_id=?2", params![task.id.to_string(),id.to_string()], |row| row.get(0)).optional()?;
                let read = matches!(task.actions[id].proposal.action, Action::ReadFile { .. } | Action::ListDirectory { .. });
                if !read && journal.as_deref().is_some_and(|state| matches!(state, "dispatched" | "uncertain" | "confirmed")) {
                    return Err(CoreError::PermissionRequired("A changed action needs effect settlement before correction".into()).into());
                }
                let mut query = tx.prepare("SELECT id FROM decisions WHERE task_id=?1 AND action_id=?2 AND state='pending'")?;
                let decisions = query.query_map(params![task.id.to_string(),id.to_string()], |row| row.get::<_,String>(0))?.collect::<Result<Vec<_>,_>>()?;
                drop(query);
                for decision in decisions {
                    tx.execute("UPDATE decisions SET state='cancelled',resolution_json=?2,resolved_at=?3 WHERE id=?1 AND state='pending'", params![decision,serde_json::to_string(&crate::decisions::DecisionResolution::Cancelled)?,Utc::now().to_rfc3339()])?;
                    events.push(CoreEvent::new(Some(task.id), CoreEventKind::DecisionResolved { decision_id: decision.parse().map_err(|_| CoreError::Storage("Invalid decision identity".into()))?, state: "cancelled".into() }));
                }
                tx.execute("UPDATE approvals SET status='cancelled',resolved_at=?3 WHERE task_id=?1 AND action_id=?2 AND status='pending'", params![task.id.to_string(),id.to_string(),Utc::now().to_rfc3339()])?;
                tx.execute("UPDATE capabilities SET revoked_at=?3 WHERE task_id=?1 AND action_id=?2 AND revoked_at IS NULL", params![task.id.to_string(),id.to_string(),Utc::now().to_rfc3339()])?;
                tx.execute("UPDATE action_journal SET state='cancelled',updated_at=?3 WHERE run_id=?1 AND action_id=?2 AND state IN ('prepared','dispatched','uncertain')", params![task.id.to_string(),id.to_string(),Utc::now().to_rfc3339()])?;
            }
            let revision_number = write_task(&tx, task)?;
            let changed = tx.execute("UPDATE messages SET content=?2 WHERE task_id=?1 AND role='user'", params![task.id.to_string(),task.request])?;
            if changed != 1 { return Err(CoreError::Storage("The revised request has no user message".into()).into()); }
            tx.execute("UPDATE conversations SET updated_at=?2 WHERE id=?1", params![task.conversation_id.map(|id| id.to_string()),Utc::now().to_rfc3339()])?;
            tx.execute("INSERT INTO command_inbox(principal,request_id,payload_digest,task_id,accepted_at) VALUES('local-user',?1,?2,?3,?4)", params![key.id.to_string(),key.digest,task.id.to_string(),Utc::now().to_rfc3339()])?;
            for event in &events { write_event(&tx, event)?; }
            write_audit(&tx, Some(task.id), None, "intent_revised", &json!({"intent_revision":task.intent.as_ref().unwrap().revision,"kept":revision.kept,"added":revision.added,"retired":revision.retired,"previous_request":prior.request,"request":task.request,"event_id":events[0].id}))?;
            tx.commit()?;
            Ok(revision_number)
        })?;
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn task(text: &str) -> Task {
        let mut task = Task::new(text);
        task.contract = Some(crate::contracts::RunContract::local(task.id));
        let (graph, bindings) = crate::intent::compile(text, &[])
            .unwrap()
            .bind(task.id, text.into());
        task.install_plan(graph).unwrap();
        task.intent = Some(bindings);
        task
    }
    fn revise(task: &Task, text: &str) -> CoreResult<Revision> {
        prepare(
            task,
            &crate::intent::compile(text, &[]).unwrap(),
            text.into(),
        )
    }

    #[test]
    fn changing_c_and_e_preserves_other_steps_and_rewires_unstarted_successors() {
        let old = task(
            "open safari and open notes and open mail and open calendar and open finder then open chrome",
        );
        let previous = &old.intent.as_ref().unwrap().steps;
        let revised = revise(&old, "open safari and open notes and open terminal and open calendar and open calculator then open chrome").unwrap();
        assert_eq!(revised.kept, 4);
        let steps = &revised.task.intent.as_ref().unwrap().steps;
        for i in [0, 1, 3, 5] {
            assert_eq!(steps[i].action_id, previous[i].action_id);
        }
        for i in [2, 4] {
            assert_ne!(steps[i].action_id, previous[i].action_id);
        }
        assert_eq!(
            steps[5].dependencies,
            steps[..5].iter().map(|s| s.action_id).collect()
        );
        assert_eq!(revised.retired.len(), 2);
        // Repeating the same action never aliases two independently requested steps.
        let old = task("open safari and open safari and open notes");
        let revised = revise(&old, "open safari and open notes and open safari").unwrap();
        let ids: BTreeSet<_> = revised
            .task
            .intent
            .unwrap()
            .steps
            .iter()
            .map(|s| s.action_id)
            .collect();
        assert_eq!(revised.kept, 3);
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn correction_cannot_hide_a_mutation_or_refresh_its_step_budget() {
        for status in [
            ActionStatus::Running,
            ActionStatus::Verifying,
            ActionStatus::Succeeded,
            ActionStatus::Uncertain,
        ] {
            let mut old = task("open safari and open notes");
            let id = old.intent.as_ref().unwrap().steps[0].action_id;
            old.actions.get_mut(&id).unwrap().status = status;
            assert!(matches!(
                revise(&old, "open terminal and open notes"),
                Err(CoreError::PermissionRequired(_))
            ));
        }
        let mut current = task("open safari");
        current.contract.as_mut().unwrap().max_steps = 2;
        let revised = revise(&current, "open notes").unwrap().task;
        assert_eq!(revised.actions.len(), 2);
        assert!(matches!(
            revise(&revised, "open terminal"),
            Err(CoreError::PermissionRequired(_))
        ));
        assert_eq!(
            revised.contract, current.contract,
            "A correction does not renew authorization"
        );
    }

    #[test]
    #[ignore = "run in release mode to measure the CPU reconciliation path"]
    fn intent_revision_cpu_latency() {
        let old = task(
            "open safari and open notes and open mail and open calendar and open finder then open chrome",
        );
        let text = "open safari and open notes and open terminal and open calendar and open calculator then open chrome";
        let intent = crate::intent::compile(text, &[]).unwrap();
        let mut samples = Vec::with_capacity(10_000);
        for i in 0..10_500 {
            let start = std::time::Instant::now();
            let revision = prepare(
                std::hint::black_box(&old),
                std::hint::black_box(&intent),
                text.into(),
            )
            .unwrap();
            std::hint::black_box(revision);
            if i >= 500 {
                samples.push(start.elapsed().as_nanos());
            }
        }
        samples.sort_unstable();
        println!(
            "intent_revision_cpu samples={} p50_ns={} p95_ns={} p99_ns={}; excludes persistence, IPC, approvals and effects",
            samples.len(),
            samples[5000],
            samples[9500],
            samples[9900]
        );
    }
}
