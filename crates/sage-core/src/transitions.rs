//! Evidence-driven state transitions. A verified action's journal, result,
//! private payload, task projection, event and audit link commit together.
use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::contracts::{DataLabel, ToolResult, Verdict, VerificationRecord};
use crate::domain::{ActionProposal, ActionStatus, Task, TaskStatus};
use crate::events::{CoreEvent, CoreEventKind};
use crate::execution::ExecutionReceipt;
use crate::observation::Observation;
use crate::redaction::redact_for_persistence;
use crate::storage::{LocalStore, write_audit, write_event, write_task};
use crate::{CoreError, CoreResult};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum VerificationMode {
    Execution,
    Recovery,
}

struct Artifact {
    id: Uuid,
    bytes: Vec<u8>,
}

pub(crate) struct VerifiedAction {
    pub(crate) task: Task,
    record: VerificationRecord,
    event: CoreEvent,
    artifact: Option<Artifact>,
    mode: VerificationMode,
}

impl VerifiedAction {
    pub(crate) fn from_observation(
        current: &Task,
        proposal: &ActionProposal,
        receipt: &ExecutionReceipt,
        observation: &Observation,
        mode: VerificationMode,
    ) -> CoreResult<Self> {
        let mut task = current.clone();
        if !task.action_is_current(proposal.id) {
            return Err(CoreError::Cancelled);
        }
        let state = task
            .actions
            .get_mut(&proposal.id)
            .ok_or_else(|| CoreError::InvalidAction("Verified action is missing".into()))?;
        if proposal.task_id != task.id
            || state.proposal != *proposal
            || !matches!(
                state.status,
                ActionStatus::Running
                    | ActionStatus::Verifying
                    | ActionStatus::Uncertain
                    | ActionStatus::Succeeded
            )
        {
            return Err(CoreError::VerificationFailed(
                "Verification does not match the current prepared action".into(),
            ));
        }
        crate::verification::Verifier.verify(&proposal.expected_outcome, observation)?;
        if let crate::domain::Action::ListDirectory {
            path,
            page_size,
            cursor,
        } = &proposal.action
        {
            let page: crate::execution::directory::DirectoryPage =
                serde_json::from_value(receipt.transient_data.clone())?;
            let digest = page.digest()?;
            if !observation.evidence.iter().any(|evidence| {
                matches!(evidence,
                    crate::observation::Evidence::DirectoryPage {
                        path: observed_path,
                        page_size: observed_size,
                        cursor: observed_cursor,
                        page_sha256,
                        ..
                    } if std::path::Path::new(observed_path) == path
                        && observed_size == page_size
                        && observed_cursor == cursor
                        && page_sha256 == &digest
                )
            }) {
                return Err(CoreError::VerificationFailed(
                    "Directory content does not match the verified page".into(),
                ));
            }
        }
        let (output, artifact) = if mode == VerificationMode::Execution {
            retain_output(&receipt.transient_data)?
        } else {
            (receipt.transient_data.clone(), None)
        };
        let summary = redact_for_persistence(&receipt.summary);
        state.status = ActionStatus::Succeeded;
        state.summary = Some(summary.clone());
        state.error = None;
        task.tool_results.push(ToolResult {
            action_id: proposal.id,
            tool: proposal.action.kind().into(),
            verdict: Verdict::Confirmed,
            summary: summary.clone(),
            output,
            label: DataLabel::private(task.id, proposal.id.to_string()),
            observed_at: observation.observed_at,
        });
        task.touch();
        let record = VerificationRecord {
            run_id: task.id,
            action_id: proposal.id,
            target: proposal.target_resource.clone(),
            action_digest: crate::policy::approval_digest(proposal)?,
            expected: proposal.expected_outcome.clone(),
            observed_at: observation.observed_at,
            verdict: Verdict::Confirmed,
            evidence: observation.evidence.clone(),
        };
        let event = CoreEvent::new(
            Some(task.id),
            CoreEventKind::ActionSucceeded {
                action_id: proposal.id,
                summary,
            },
        );
        Ok(Self {
            task,
            record,
            event,
            artifact,
            mode,
        })
    }

    /// Bind a broker-verified control readback to its task-owned procedure
    /// checkpoint. The value comes from fresh verifier evidence, never from
    /// the proposal or executor's returned payload.
    pub(crate) fn advance_procedure_checkpoint(
        &mut self,
        checkpoint: &mut crate::agency::ProcedureCheckpoint,
    ) -> CoreResult<()> {
        let action = self
            .task
            .actions
            .get(&self.record.action_id)
            .ok_or_else(|| {
                CoreError::InvalidAction("Verified procedure action is missing".into())
            })?;
        let proposal = &action.proposal;
        let procedure_id = proposal
            .metadata
            .get("procedure_id")
            .ok_or_else(|| CoreError::VerificationFailed("Procedure identity is missing".into()))?;
        let node_id = proposal.metadata.get("procedure_node_id").ok_or_else(|| {
            CoreError::VerificationFailed("Procedure node identity is missing".into())
        })?;
        if checkpoint.task_id != self.task.id
            || checkpoint.procedure.id != *procedure_id
            || !checkpoint
                .procedure
                .nodes
                .iter()
                .any(|node| node.id == *node_id)
        {
            return Err(CoreError::VerificationFailed(
                "Verified action does not match its procedure checkpoint".into(),
            ));
        }
        let crate::domain::Action::SetApplicationControl { control_id, .. } = &proposal.action
        else {
            return Err(CoreError::ExecutorUnavailable(
                "This procedure output has no qualified verifier mapping".into(),
            ));
        };
        let observed_value = self
            .record
            .evidence
            .iter()
            .find_map(|evidence| match evidence {
                crate::observation::Evidence::ApplicationControlValue {
                    control_id: observed_control,
                    value,
                    ..
                } if observed_control == control_id => Some(value),
                _ => None,
            })
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Verified procedure step has no matching fresh control readback".into(),
                )
            })?;
        let value = match observed_value {
            crate::domain::ApplicationControlValue::Boolean(value) => {
                crate::agency::ProcedureValue::Boolean(*value)
            }
            crate::domain::ApplicationControlValue::Number(value) => {
                crate::agency::ProcedureValue::Number(*value)
            }
        };
        let outputs = std::collections::BTreeMap::from([(
            "observed_value".to_owned(),
            crate::agency::ProcedureOutput::Value(value),
        )]);
        checkpoint.runtime.record_verified_outputs(
            &checkpoint.procedure,
            node_id,
            outputs.clone(),
            self.record.action_id,
        )?;

        let result = self
            .task
            .tool_results
            .iter_mut()
            .rev()
            .find(|result| result.action_id == self.record.action_id)
            .ok_or_else(|| {
                CoreError::VerificationFailed("Verified procedure result is missing".into())
            })?;
        let outputs = serde_json::to_value(outputs)?;
        match &mut result.output {
            Value::Object(object) => {
                if object.contains_key("procedure_outputs") {
                    return Err(CoreError::VerificationFailed(
                        "Executor output cannot supply procedure verification results".into(),
                    ));
                }
                object.insert("procedure_outputs".into(), outputs);
            }
            original => {
                let prior = std::mem::replace(original, Value::Null);
                *original = json!({"result": prior, "procedure_outputs": outputs});
            }
        }
        Ok(())
    }

    fn discard_returned_content(&mut self) {
        self.artifact = None;
        let summary =
            "Outcome verified; returned content was not retained after forgetting related memory.";
        let result = self
            .task
            .tool_results
            .last_mut()
            .expect("verified action has a result");
        result.output = json!({"retention_revoked":true});
        result.summary = summary.into();
        self.task
            .actions
            .get_mut(&self.record.action_id)
            .expect("verified action exists")
            .summary = Some(summary.into());
        self.event.kind = CoreEventKind::ActionSucceeded {
            action_id: self.record.action_id,
            summary: summary.into(),
        };
    }
}

impl LocalStore {
    pub(crate) fn commit_prepared_action(
        &self,
        mut task: Task,
        prepared: &crate::contracts::PreparedAction,
    ) -> CoreResult<(Task, CoreEvent)> {
        let proposal = &prepared.intent.proposal;
        if !task.status.is_active()
            || proposal.task_id != task.id
            || !task.action_is_current(proposal.id)
        {
            return Err(CoreError::Cancelled);
        }
        let action = task
            .actions
            .get_mut(&proposal.id)
            .ok_or_else(|| CoreError::InvalidAction("Action is missing".into()))?;
        if !matches!(
            action.status,
            ActionStatus::Pending | ActionStatus::Compiling | ActionStatus::Failed
        ) {
            return Err(CoreError::InvalidAction(
                "An executed action cannot be prepared again".into(),
            ));
        }
        action.proposal = proposal.clone();
        action.error = None;
        task.touch();
        let event = CoreEvent::new(
            Some(task.id),
            CoreEventKind::ActionProposed {
                action_id: proposal.id,
                summary: proposal.action.redacted_summary(),
            },
        );
        task.revision = self.with_connection(|db| {
            let transaction = db.transaction()?;
            crate::journal::write_prepared(&transaction, prepared)?;
            let revision = write_task(&transaction, &task)?;
            write_event(&transaction, &event)?;
            write_audit(
                &transaction,
                Some(task.id),
                Some(proposal.id),
                "action_prepared",
                &json!({"action_digest":prepared.action_digest,"event_id":event.id}),
            )?;
            transaction.commit()?;
            Ok(revision)
        })?;
        Ok((task, event))
    }

    #[cfg(test)]
    pub(crate) fn commit_dispatched_action(
        &self,
        task: Task,
        prepared: &crate::contracts::PreparedAction,
        grant: Option<&crate::capability::CapabilityGrant>,
        implementation: &str,
    ) -> CoreResult<(Task, CoreEvent)> {
        self.commit_dispatched_action_with_procedure(task, prepared, grant, implementation, None)
    }

    pub(crate) fn commit_dispatched_action_with_procedure(
        &self,
        mut task: Task,
        prepared: &crate::contracts::PreparedAction,
        grant: Option<&crate::capability::CapabilityGrant>,
        implementation: &str,
        mut checkpoint: Option<crate::agency::ProcedureCheckpoint>,
    ) -> CoreResult<(Task, CoreEvent)> {
        let proposal = &prepared.intent.proposal;
        validate_procedure_action_checkpoint(proposal, task.id, checkpoint.as_ref())?;
        if task.status != TaskStatus::Running
            || proposal.task_id != task.id
            || !task.action_is_current(proposal.id)
        {
            return Err(CoreError::Cancelled);
        }
        let action = task
            .actions
            .get_mut(&proposal.id)
            .ok_or_else(|| CoreError::InvalidAction("Action is missing".into()))?;
        if action.proposal != *proposal
            || !matches!(
                action.status,
                ActionStatus::Compiling | ActionStatus::WaitingForApproval | ActionStatus::Pending
            )
        {
            return Err(CoreError::InvalidAction(
                "Dispatch does not match the prepared action".into(),
            ));
        }
        action.status = ActionStatus::Running;
        task.touch();
        let event = CoreEvent::new(
            Some(task.id),
            CoreEventKind::ActionStarted {
                action_id: proposal.id,
                implementation: implementation.into(),
            },
        );
        let (revision, checkpoint_revision) = self.with_connection(|db| {
            let transaction = db.transaction()?;
            crate::journal::write_dispatch(&transaction, prepared, grant)?;
            let revision = write_task(&transaction, &task)?;
            let checkpoint_revision = checkpoint
                .as_ref()
                .map(|checkpoint| {
                    crate::storage::write_procedure_checkpoint_tx(
                        &transaction,
                        checkpoint,
                        &task,
                    )
                })
                .transpose()?;
            write_event(&transaction, &event)?;
            write_audit(&transaction, Some(task.id), Some(proposal.id), "action_dispatch_intent", &json!({
                "action_digest":prepared.action_digest,"capability_id":grant.map(|grant|grant.id),"implementation":implementation,"event_id":event.id,
            }))?;
            transaction.commit()?;
            Ok((revision, checkpoint_revision))
        })?;
        task.revision = revision;
        if let (Some(checkpoint), Some((revision, updated_at))) =
            (checkpoint.as_mut(), checkpoint_revision)
        {
            checkpoint.revision = revision;
            checkpoint.updated_at = updated_at;
        }
        Ok((task, event))
    }

    /// Commit a complete procedure dispatch wave and all of its single-use
    /// grants as one task/journal/checkpoint transaction. This prevents a
    /// streamed producer from starting unless every connected peer has a
    /// durable dispatch receipt.
    pub(crate) fn commit_dispatched_action_wave(
        &self,
        mut task: Task,
        dispatches: &[(
            &crate::contracts::PreparedAction,
            &crate::capability::CapabilityGrant,
            &str,
        )],
        mut checkpoint: crate::agency::ProcedureCheckpoint,
    ) -> CoreResult<(Task, Vec<CoreEvent>)> {
        if !(2..=16).contains(&dispatches.len())
            || task.status != TaskStatus::Running
            || checkpoint.task_id != task.id
        {
            return Err(CoreError::InvalidAction(
                "Procedure dispatch wave has an invalid size or task state".into(),
            ));
        }

        let mut action_ids = std::collections::BTreeSet::new();
        let mut node_ids = std::collections::BTreeSet::new();
        let mut events = Vec::with_capacity(dispatches.len());
        for (prepared, grant, implementation) in dispatches {
            let proposal = &prepared.intent.proposal;
            let node_id = proposal.metadata.get("procedure_node_id").ok_or_else(|| {
                CoreError::VerificationFailed("Procedure wave action has no node identity".into())
            })?;
            validate_procedure_action_checkpoint(proposal, task.id, Some(&checkpoint))?;
            if proposal.task_id != task.id
                || (!checkpoint.procedure.streams.is_empty()
                    && proposal
                        .metadata
                        .get("procedure_stream_node")
                        .map(String::as_str)
                        != Some("true"))
                || !action_ids.insert(proposal.id)
                || !node_ids.insert(node_id.clone())
                || grant.task_id != task.id
                || grant.action_id != proposal.id
                || grant.action_digest != prepared.action_digest
            {
                return Err(CoreError::VerificationFailed(
                    "Procedure dispatch wave contains a mismatched or duplicate action".into(),
                ));
            }
            if !task.action_is_current(proposal.id) {
                return Err(CoreError::Cancelled);
            }
            let action = task
                .actions
                .get_mut(&proposal.id)
                .ok_or_else(|| CoreError::InvalidAction("Action is missing".into()))?;
            if action.proposal != *proposal
                || !matches!(
                    action.status,
                    ActionStatus::Compiling
                        | ActionStatus::WaitingForApproval
                        | ActionStatus::Pending
                )
            {
                return Err(CoreError::InvalidAction(
                    "Procedure wave dispatch does not match every prepared action".into(),
                ));
            }
            action.status = ActionStatus::Running;
            events.push(CoreEvent::new(
                Some(task.id),
                CoreEventKind::ActionStarted {
                    action_id: proposal.id,
                    implementation: (*implementation).into(),
                },
            ));
        }

        let checkpoint_receipts = checkpoint.runtime.dispatch_action_ids();
        if node_ids.iter().any(|node_id| {
            checkpoint_receipts.get(node_id.as_str()).copied()
                != dispatches.iter().find_map(|(prepared, _, _)| {
                    (prepared.intent.proposal.metadata.get("procedure_node_id") == Some(node_id))
                        .then_some(prepared.intent.proposal.id)
                })
        }) {
            return Err(CoreError::VerificationFailed(
                "Procedure checkpoint receipts do not match the complete dispatch wave".into(),
            ));
        }

        task.touch();
        let (revision, checkpoint_revision) = self.with_connection(|db| {
            let transaction = db.transaction()?;
            for (prepared, grant, _) in dispatches {
                crate::journal::write_dispatch(&transaction, prepared, Some(grant))?;
            }
            let revision = write_task(&transaction, &task)?;
            let checkpoint_revision =
                crate::storage::write_procedure_checkpoint_tx(&transaction, &checkpoint, &task)?;
            for ((prepared, grant, implementation), event) in dispatches.iter().zip(events.iter()) {
                write_event(&transaction, event)?;
                write_audit(
                    &transaction,
                    Some(task.id),
                    Some(prepared.intent.proposal.id),
                    "action_dispatch_intent",
                    &json!({
                        "action_digest":prepared.action_digest,
                        "capability_id":grant.id,
                        "implementation":implementation,
                        "event_id":event.id,
                        "procedure_wave":true,
                    }),
                )?;
            }
            transaction.commit()?;
            Ok((revision, checkpoint_revision))
        })?;

        task.revision = revision;
        checkpoint.revision = checkpoint_revision.0;
        checkpoint.updated_at = checkpoint_revision.1;
        Ok((task, events))
    }

    /// The journal decides whether retry is safe. A task projection is not
    /// allowed to demote a possibly dispatched effect to a retryable failure.
    pub(crate) fn commit_interrupted_action(
        &self,
        mut task: Task,
        action_id: Uuid,
        error: &str,
    ) -> CoreResult<(Task, CoreEvent, bool)> {
        let error = redact_for_persistence(error);
        let event = CoreEvent::new(
            Some(task.id),
            CoreEventKind::ActionFailed {
                action_id,
                error: error.clone(),
            },
        );
        let (revision, may_have_executed) = self.with_connection(|db| {
            let transaction = db.transaction()?;
            let journal: Option<String> = transaction.query_row("SELECT state FROM action_journal WHERE run_id=?1 AND action_id=?2",
                params![task.id.to_string(),action_id.to_string()], |row| row.get(0)).optional()?;
            let proposal = task
                .actions
                .get(&action_id)
                .ok_or_else(|| CoreError::InvalidAction("Action is missing".into()))?
                .proposal
                .clone();
            let binding = procedure_action_binding(&proposal)?;
            let mut checkpoint = if binding.is_some() {
                crate::storage::read_procedure_checkpoint_tx(&transaction, task.id)?
            } else {
                None
            };
            let retention_revoked: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=?1)",
                [task.id.to_string()],
                |row| row.get(0),
            )?;
            if let Some((procedure_id, node_id)) = binding {
                match checkpoint.as_ref() {
                    Some(checkpoint) => {
                        validate_procedure_action_checkpoint(&proposal, task.id, Some(checkpoint))?;
                        if checkpoint.procedure.id != procedure_id {
                            return Err(CoreError::VerificationFailed("Procedure identity changed during interruption".into()).into());
                        }
                        if checkpoint.runtime.node_states().get(node_id).is_none()
                            && journal.as_deref().is_some_and(|state| matches!(state, "dispatched" | "uncertain" | "confirmed"))
                        {
                            return Err(CoreError::VerificationFailed("Dispatched procedure action has no checkpoint receipt".into()).into());
                        }
                    }
                    None if !retention_revoked => {
                        return Err(CoreError::VerificationFailed("Procedure action has no durable checkpoint".into()).into());
                    }
                    None => {}
                }
            }
            let action = task.actions.get_mut(&action_id).ok_or_else(|| CoreError::InvalidAction("Action is missing".into()))?;
            if action.status == ActionStatus::Succeeded && task.tool_results.iter().rev().find(|result| result.action_id==action_id).is_some_and(|result| result.verdict==Verdict::Confirmed) {
                return Err(CoreError::InvalidAction("Committed verification cannot be replaced by interruption".into()).into());
            }
            let may_have_executed = match journal.as_deref() {
                Some("dispatched" | "uncertain" | "confirmed") => true,
                Some("prepared" | "failed" | "cancelled") => false,
                None => matches!(action.status, ActionStatus::Running | ActionStatus::Verifying | ActionStatus::Uncertain | ActionStatus::Succeeded),
                _ => return Err(CoreError::Storage("Unknown action journal state".into()).into()),
            };
            action.status = if may_have_executed { ActionStatus::Uncertain } else { ActionStatus::Failed };
            action.error = Some(error.clone());
            if may_have_executed && task.status != TaskStatus::Cancelled { task.status = TaskStatus::Interrupted; }
            task.tool_results.push(ToolResult {
                action_id, tool: "execution_interrupted".into(), verdict: if may_have_executed { Verdict::Uncertain } else { Verdict::Failed },
                summary: if may_have_executed { "An action may have executed. Review the actual state before continuing." } else { "The action stopped before its external dispatch." }.into(),
                output: json!({"error":error}), label: DataLabel::private(task.id, action_id.to_string()), observed_at: Utc::now(),
            });
            task.touch();
            transaction.execute("UPDATE action_journal SET state=?3,updated_at=?4 WHERE run_id=?1 AND action_id=?2 AND state IN ('prepared','dispatched','uncertain')",
                params![task.id.to_string(),action_id.to_string(),if may_have_executed {"uncertain"} else {"failed"},Utc::now().to_rfc3339()])?;
            let revision = write_task(&transaction, &task)?;
            if let (Some((_, node_id)), Some(checkpoint)) = (binding, checkpoint.as_mut()) {
                match checkpoint.runtime.node_states().get(node_id) {
                    Some(crate::agency::ProcedureNodeState::Running) if may_have_executed => {
                        checkpoint.runtime.record_failure(
                            &checkpoint.procedure,
                            node_id,
                            true,
                        )?;
                        crate::storage::write_procedure_checkpoint_tx(
                            &transaction,
                            checkpoint,
                            &task,
                        )?;
                    }
                    Some(crate::agency::ProcedureNodeState::Uncertain) if may_have_executed => {}
                    None if !may_have_executed => {}
                    _ => {
                        return Err(CoreError::VerificationFailed(
                            "Procedure checkpoint state conflicts with interruption outcome".into(),
                        ).into());
                    }
                }
            }
            write_event(&transaction, &event)?;
            write_audit(&transaction, Some(task.id), Some(action_id), "action_interrupted", &json!({"may_have_executed":may_have_executed,"event_id":event.id}))?;
            transaction.commit()?;
            Ok((revision,may_have_executed))
        })?;
        task.revision = revision;
        Ok((task, event, may_have_executed))
    }

    #[cfg(test)]
    pub(crate) fn commit_verified_action(
        &self,
        change: VerifiedAction,
    ) -> CoreResult<(Task, CoreEvent)> {
        self.commit_verified_action_with_procedure(change, None)
    }

    pub(crate) fn commit_verified_action_with_procedure(
        &self,
        mut change: VerifiedAction,
        mut checkpoint: Option<crate::agency::ProcedureCheckpoint>,
    ) -> CoreResult<(Task, CoreEvent)> {
        let proposal = &change.task.actions[&change.record.action_id].proposal;
        let is_procedure_action = procedure_action_binding(proposal)?.is_some();
        if let Some(checkpoint) = checkpoint.as_ref() {
            validate_procedure_action_checkpoint(proposal, change.task.id, Some(checkpoint))?;
        }
        let (revision, checkpoint_revision) = self.with_connection(|db| {
            let transaction = db.transaction()?;
            let retired: bool = transaction.query_row("SELECT EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=?1)",
                [change.task.id.to_string()], |row| row.get(0))?;
            if is_procedure_action && !retired && checkpoint.is_none() {
                return Err(CoreError::VerificationFailed(
                    "Procedure action has no durable checkpoint".into(),
                ).into());
            }
            if retired { change.discard_returned_content(); }
            let record = &change.record;
            let changed = transaction.execute(
                "UPDATE action_journal SET state='confirmed',verification_json=?4,updated_at=?5 WHERE run_id=?1 AND action_id=?2 AND action_digest=?3 AND (state IN ('dispatched','uncertain') OR (?6 AND state='confirmed'))",
                params![record.run_id.to_string(),record.action_id.to_string(),record.action_digest,serde_json::to_string(record)?,Utc::now().to_rfc3339(),change.mode==VerificationMode::Recovery])?;
            if changed != 1 { return Err(CoreError::VerificationFailed("Verification journal does not match an executed action".into()).into()); }
            #[cfg(test)] crash_checkpoint("journal");
            if let Some(artifact) = &change.artifact {
                transaction.execute("INSERT INTO private_artifacts VALUES(?1,?2,?3,?4,?5)",params![artifact.id.to_string(),change.task.id.to_string(),artifact.bytes,
                    format!("{:x}",Sha256::digest(&artifact.bytes)),(Utc::now()+chrono::Duration::hours(24)).to_rfc3339()])?;
            }
            #[cfg(test)] crash_checkpoint("artifact");
            let revision = write_task(&transaction, &change.task)?;
            #[cfg(test)] crash_checkpoint("task");
            let checkpoint_revision = if retired {
                None
            } else {
                checkpoint
                    .as_ref()
                    .map(|checkpoint| {
                        crate::storage::write_procedure_checkpoint_tx(
                            &transaction,
                            checkpoint,
                            &change.task,
                        )
                    })
                    .transpose()?
            };
            write_event(&transaction, &change.event)?;
            #[cfg(test)] crash_checkpoint("event");
            crate::receipts::cover_verified_receipts(&transaction, record, &change.event)?;
            write_audit(&transaction, Some(record.run_id), Some(record.action_id), "action_verified", &json!({
                "action_digest":record.action_digest,"verdict":record.verdict,"observed_at":record.observed_at,
                "recovered":change.mode==VerificationMode::Recovery,"event_id":change.event.id,
            }))?;
            #[cfg(test)] crash_checkpoint("audit");
            transaction.commit()?;
            #[cfg(test)] crash_checkpoint("committed");
            Ok((revision, checkpoint_revision))
        })?;
        change.task.revision = revision;
        if let (Some(checkpoint), Some((revision, updated_at))) =
            (checkpoint.as_mut(), checkpoint_revision)
        {
            checkpoint.revision = revision;
            checkpoint.updated_at = updated_at;
        }
        Ok((change.task, change.event))
    }
}

fn procedure_action_binding(proposal: &ActionProposal) -> CoreResult<Option<(&str, &str)>> {
    match (
        proposal.metadata.get("procedure_id"),
        proposal.metadata.get("procedure_node_id"),
    ) {
        (None, None) => Ok(None),
        (Some(procedure_id), Some(node_id)) => Ok(Some((procedure_id, node_id))),
        _ => Err(CoreError::VerificationFailed(
            "Procedure action has an incomplete identity binding".into(),
        )),
    }
}

fn validate_procedure_action_checkpoint(
    proposal: &ActionProposal,
    task_id: Uuid,
    checkpoint: Option<&crate::agency::ProcedureCheckpoint>,
) -> CoreResult<()> {
    match (procedure_action_binding(proposal)?, checkpoint) {
        (None, None) => Ok(()),
        (Some((procedure_id, node_id)), Some(checkpoint))
            if checkpoint.task_id == task_id
                && checkpoint.procedure.id == procedure_id
                && checkpoint
                    .procedure
                    .nodes
                    .iter()
                    .any(|node| node.id == node_id) =>
        {
            Ok(())
        }
        (Some(_), None) => Err(CoreError::VerificationFailed(
            "Procedure action requires its durable checkpoint".into(),
        )),
        _ => Err(CoreError::VerificationFailed(
            "Procedure action and checkpoint identities do not match".into(),
        )),
    }
}

#[cfg(test)]
fn crash_checkpoint(stage: &str) {
    if std::env::var("SAGE_TEST_VERIFICATION_EXIT_AT")
        .ok()
        .as_deref()
        == Some(stage)
    {
        // Exit without running Rust/SQLite transaction destructors. This hook
        // and its environment variable do not exist in production builds.
        std::process::exit(86);
    }
}

fn retain_output(input: &Value) -> CoreResult<(Value, Option<Artifact>)> {
    let mut output = input.clone();
    let bytes = if let Some(encoded) = input.get("bytes_base64").and_then(Value::as_str) {
        use base64::Engine;
        if encoded.len() > 24 * 1024 * 1024 {
            return Err(CoreError::Storage(
                "Artifact exceeds the 16 MiB limit".into(),
            ));
        }
        Some(
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|_| CoreError::VerificationFailed("Invalid file read result".into()))?,
        )
    } else {
        input
            .get("text")
            .and_then(Value::as_str)
            .map(|text| text.as_bytes().to_vec())
    };
    let Some(bytes) = bytes else {
        return Ok((output, None));
    };
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(CoreError::Storage(
            "Artifact exceeds the 16 MiB limit".into(),
        ));
    }
    let artifact = Artifact {
        id: Uuid::new_v4(),
        bytes,
    };
    let text = String::from_utf8_lossy(&artifact.bytes);
    if input.get("bytes_base64").is_some() {
        output = json!({"bytes":artifact.bytes.len(),"sha256":format!("{:x}",Sha256::digest(&artifact.bytes))});
    }
    output["text"] = json!(redact_for_persistence(
        &text.chars().take(8000).collect::<String>()
    ));
    output["truncated"] = json!(text.chars().count() > 8000);
    output["artifact_ref"] = json!(artifact.id);
    Ok((output, Some(artifact)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::PreparedAction;
    use crate::domain::{
        Action, ActionState, ExpectedOutcome, Provenance, ProvenanceSource, TaskStatus,
    };
    use crate::observation::Evidence;
    use crate::secrets::{SecretBytes, testing::MemorySecretStore};
    use std::collections::{BTreeMap, BTreeSet};

    fn fixture(
        path: &std::path::Path,
    ) -> (
        LocalStore,
        Task,
        ActionProposal,
        ExecutionReceipt,
        Observation,
    ) {
        let store = LocalStore::open_encrypted(path, &SecretBytes::new(vec![29; 32])).unwrap();
        let mut task = Task::new("Fixture verified response");
        task.status = TaskStatus::Running;
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
        let receipt = ExecutionReceipt {
            executor: "fixture".into(),
            summary: "Received the selected response".into(),
            transient_data: json!({"text":"retained fixture response"}),
            rollback: None,
        };
        let observation = Observation {
            observed_at: Utc::now(),
            provenance: Provenance::external(ProvenanceSource::OperatingSystem, "fixture"),
            summary: "Response was independently observed".into(),
            evidence: vec![Evidence::UserAnswer { received: true }],
        };
        (store, task, proposal, receipt, observation)
    }

    #[test]
    fn procedure_dispatch_wave_commits_all_receipts_or_none() {
        use crate::agency::{
            CompletionCondition, ProcedureCheckpoint, ProcedureIr, ProcedureNode,
            ProcedureNodeKind, ProcedureRuntimeState, ProcedureValue, StreamBackpressure,
            StreamChannel, ValueBinding,
        };
        use crate::capability::{CapabilityGrant, CapabilityOperation, CapabilityResource};
        use crate::contracts::{Effect, POLICY_VERSION};
        use crate::domain::{Action, ActionState, ExpectedOutcome, Provenance, TaskStatus};
        use crate::knowledge::Message;
        use crate::world_model::{
            CapabilityAssessment, CapabilityDescriptor, CapabilityEvidenceState, DataPort,
            EvidenceOrigin, FactValue, ObservationEnvelope, ObservedFact, PortType, Preconditions,
            SystemDescriptor, SystemKind,
        };

        let directory = tempfile::tempdir().unwrap();
        let store = LocalStore::open_encrypted(
            &directory.path().join("stream-dispatch-wave.db"),
            &SecretBytes::new(vec![61; 32]),
        )
        .unwrap();
        store.migrate_knowledge().unwrap();
        store.migrate_world_model().unwrap();

        let now = Utc::now();
        let system_ids = [Uuid::new_v4(), Uuid::new_v4()];
        let fingerprints = ["ab".repeat(32), "cd".repeat(32)];
        let evidence_ids = [Uuid::new_v4(), Uuid::new_v4()];
        let path_port = DataPort {
            name: "path".into(),
            value_type: PortType::Text,
            max_bytes: 512,
            privacy: crate::contracts::Sensitivity::Private,
        };
        let producer_chunk_port = DataPort {
            name: "chunks".into(),
            value_type: PortType::Bytes,
            max_bytes: 1024,
            privacy: crate::contracts::Sensitivity::Private,
        };
        let consumer_chunk_port = DataPort {
            max_bytes: 2048,
            ..producer_chunk_port.clone()
        };
        let producer = CapabilityDescriptor {
            schema_version: 1,
            id: "fixture.media.produce".into(),
            system_id: system_ids[0],
            system_fingerprint: fingerprints[0].clone(),
            interface_control_id: None,
            interface_probe_kind: None,
            label: "Create first folder".into(),
            input_ports: vec![path_port.clone()],
            output_ports: vec![producer_chunk_port.clone()],
            preconditions: Preconditions {
                observed_state_fact_ids: vec![evidence_ids[0]],
                description: "Current first target state".into(),
            },
            effects: BTreeSet::from([Effect::Create]),
            verification: "Fresh directory identity".into(),
            restoration: Some("Remove only the still-empty created directory".into()),
            cancellation: "Settle the first folder operation before retry".into(),
            executor_id: Some("create_folder".into()),
            evidence_ids: vec![evidence_ids[0]],
            updated_at: now,
        };
        let consumer = CapabilityDescriptor {
            schema_version: 1,
            id: "fixture.media.consume".into(),
            system_id: system_ids[1],
            system_fingerprint: fingerprints[1].clone(),
            interface_control_id: None,
            interface_probe_kind: None,
            label: "Create second folder".into(),
            input_ports: vec![path_port.clone(), consumer_chunk_port.clone()],
            output_ports: Vec::new(),
            preconditions: Preconditions {
                observed_state_fact_ids: vec![evidence_ids[1]],
                description: "Current second target state".into(),
            },
            effects: BTreeSet::from([Effect::Create]),
            verification: "Fresh directory identity".into(),
            restoration: Some("Remove only the still-empty created directory".into()),
            cancellation: "Settle the second folder operation before retry".into(),
            executor_id: Some("create_folder".into()),
            evidence_ids: vec![evidence_ids[1]],
            updated_at: now,
        };
        for index in 0..2 {
            store
                .observe_system(SystemDescriptor {
                    id: system_ids[index],
                    kind: SystemKind::Device,
                    key: format!("fixture-folder-system-{index}"),
                    label: format!("Fixture folder system {index}"),
                    fingerprint: fingerprints[index].clone(),
                    revision: 1,
                    updated_at: now,
                })
                .unwrap();
            store
                .record_world_observation(&ObservationEnvelope {
                    id: evidence_ids[index],
                    system_id: system_ids[index],
                    session_id: None,
                    worker_session: None,
                    system_fingerprint: fingerprints[index].clone(),
                    origin: EvidenceOrigin::OperatingSystem,
                    privacy: crate::contracts::Sensitivity::Private,
                    observed_at: now,
                    facts: vec![ObservedFact {
                        name: "system.ready".into(),
                        subject: None,
                        value: FactValue::Boolean(true),
                    }],
                })
                .unwrap();
        }
        store.record_capability_candidate(&producer).unwrap();
        store.record_capability_candidate(&consumer).unwrap();
        let procedure = ProcedureIr {
            schema_version: 2,
            id: "atomic-dispatch-wave".into(),
            nodes: vec![
                ProcedureNode {
                    id: "producer".into(),
                    depends_on: BTreeSet::new(),
                    outputs: BTreeMap::from([("chunks".into(), producer_chunk_port.clone())]),
                    kind: ProcedureNodeKind::CapabilityCall {
                        capability_id: producer.id.clone(),
                        system_id: system_ids[0],
                        system_fingerprint: fingerprints[0].clone(),
                        input_bindings: BTreeMap::from([(
                            "path".into(),
                            ValueBinding::Literal {
                                value: ProcedureValue::Text("/tmp/sage-wave-first".into()),
                                port: path_port.clone(),
                            },
                        )]),
                    },
                },
                ProcedureNode {
                    id: "consumer".into(),
                    depends_on: BTreeSet::new(),
                    outputs: BTreeMap::new(),
                    kind: ProcedureNodeKind::CapabilityCall {
                        capability_id: consumer.id.clone(),
                        system_id: system_ids[1],
                        system_fingerprint: fingerprints[1].clone(),
                        input_bindings: BTreeMap::from([
                            (
                                "chunks".into(),
                                ValueBinding::Stream {
                                    channel_id: "folder-progress".into(),
                                },
                            ),
                            (
                                "path".into(),
                                ValueBinding::Literal {
                                    value: ProcedureValue::Text("/tmp/sage-wave-second".into()),
                                    port: path_port.clone(),
                                },
                            ),
                        ]),
                    },
                },
            ],
            streams: vec![StreamChannel {
                id: "folder-progress".into(),
                producer_node: "producer".into(),
                producer_output: "chunks".into(),
                consumer_node: "consumer".into(),
                consumer_input: "chunks".into(),
                capacity_items: 2,
                maximum_item_bytes: 1024,
                backpressure: StreamBackpressure::BlockProducer,
            }],
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        procedure
            .validate_against(&[producer.clone(), consumer.clone()])
            .unwrap();
        let assessments = [
            CapabilityAssessment {
                descriptor: producer,
                evidence_state: CapabilityEvidenceState::ReversiblyExperimented,
            },
            CapabilityAssessment {
                descriptor: consumer,
                evidence_state: CapabilityEvidenceState::ReversiblyExperimented,
            },
        ];
        let task_id = Uuid::new_v4();
        let conversation = store
            .ensure_conversation(None, "atomic dispatch wave")
            .unwrap();
        let mut task = Task::new("Produce and consume a media stream");
        task.id = task_id;
        task.status = TaskStatus::Running;
        task.workflow_run = true;
        task.goal = Some(task.request.clone());
        task.conversation_id = Some(conversation.id);
        task.message_id = Some(Uuid::new_v4());
        let actions = [
            ("producer", "/tmp/sage-wave-first"),
            ("consumer", "/tmp/sage-wave-second"),
        ]
        .map(|(node_id, path)| {
            let path = std::path::PathBuf::from(path);
            let file_identity = crate::contracts::FileIdentity {
                key: path.to_string_lossy().into_owned(),
                size: 0,
                modified: "fixture".into(),
                directory: false,
            };
            let proposal = ActionProposal {
                id: Uuid::new_v4(),
                task_id,
                action: Action::CreateFolder { path: path.clone() },
                expected_outcome: ExpectedOutcome::Condition {
                    condition: crate::domain::Condition::FolderExists { path: path.clone() },
                },
                target_resource: path.to_string_lossy().into_owned(),
                provenance: Provenance::user(),
                metadata: BTreeMap::from([
                    ("procedure_id".into(), procedure.id.clone()),
                    ("procedure_node_id".into(), node_id.into()),
                    ("procedure_stream_node".into(), "true".into()),
                    (
                        "file_precondition".into(),
                        serde_json::to_string(&file_identity).unwrap(),
                    ),
                ]),
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
            task.dependencies.insert(proposal.id, BTreeSet::new());
            proposal
        });
        let message = Message {
            id: task.message_id.unwrap(),
            conversation_id: conversation.id,
            task_id: Some(task_id),
            role: "user".into(),
            content: task.request.clone(),
            provenance: Provenance::user(),
            created_at: now,
        };
        let initial = ProcedureCheckpoint {
            task_id,
            revision: 0,
            runtime: ProcedureRuntimeState::new_for_task(&procedure, task_id).unwrap(),
            procedure: procedure.clone(),
            updated_at: now,
        };
        store
            .accept_task_with_procedure(
                &mut task,
                &message,
                &CoreEvent::new(Some(task_id), CoreEventKind::TaskStarted),
                None,
                &initial,
            )
            .unwrap();
        let prepared = actions
            .iter()
            .map(|proposal| PreparedAction::new(proposal, BTreeSet::new()).unwrap())
            .collect::<Vec<_>>();
        let mut prepared_task = store.load_tasks(true).unwrap().remove(0);
        for action in &prepared {
            (prepared_task, _) = store.commit_prepared_action(prepared_task, action).unwrap();
        }
        let grants = actions
            .iter()
            .zip(&prepared)
            .map(|(proposal, prepared)| CapabilityGrant {
                id: Uuid::new_v4(),
                task_id,
                action_id: proposal.id,
                action_digest: prepared.action_digest.clone(),
                policy_version: POLICY_VERSION,
                worker_session: None,
                domain: crate::domain::ExecutionDomain::Native,
                resource: CapabilityResource::File {
                    canonical_path: proposal.target_resource.clone(),
                },
                operations: BTreeSet::from([CapabilityOperation::Create]),
                issued_at: now,
                expires_at: now + chrono::Duration::minutes(1),
                remaining_uses: 1,
                revoked: false,
            })
            .collect::<Vec<_>>();
        let mut checkpoint = store.load_procedure_checkpoint(task_id).unwrap().unwrap();
        checkpoint
            .runtime
            .record_dispatched_wave(
                &procedure,
                &assessments,
                &BTreeSet::from(evidence_ids),
                BTreeMap::from([
                    ("producer".into(), actions[0].id),
                    ("consumer".into(), actions[1].id),
                ]),
            )
            .unwrap();
        let dispatches = prepared
            .iter()
            .zip(&grants)
            .map(|(prepared, grant)| (prepared, grant, "fixture-stream"))
            .collect::<Vec<_>>();

        let durable_counts = store
            .with_connection(|db| {
                Ok(db.query_row(
                    "SELECT (SELECT COUNT(*) FROM events), (SELECT COUNT(*) FROM audit_log), (SELECT COUNT(*) FROM action_journal WHERE state='prepared')",
                    [],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
                )?)
            })
            .unwrap();
        store
            .with_connection(|db| {
                db.execute_batch(&format!(
                    "CREATE TRIGGER reject_second_stream_dispatch BEFORE UPDATE OF state ON action_journal WHEN OLD.action_id='{}' AND NEW.state='dispatched' BEGIN SELECT RAISE(ABORT,'injected stream dispatch failure'); END;",
                    actions[1].id
                ))?;
                Ok(())
            })
            .unwrap();
        assert!(
            store
                .commit_dispatched_action_wave(
                    prepared_task.clone(),
                    &dispatches,
                    checkpoint.clone(),
                )
                .is_err()
        );
        assert_eq!(
            store
                .load_procedure_checkpoint(task_id)
                .unwrap()
                .unwrap()
                .revision,
            checkpoint.revision
        );
        assert_eq!(
            store.load_tasks(true).unwrap()[0].actions[&actions[0].id].status,
            ActionStatus::Pending
        );
        assert_eq!(
            store
                .with_connection(|db| {
                    Ok(db.query_row(
                        "SELECT (SELECT COUNT(*) FROM events), (SELECT COUNT(*) FROM audit_log), (SELECT COUNT(*) FROM action_journal WHERE state='prepared')",
                        [],
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
                    )?)
                })
                .unwrap(),
            durable_counts
        );
        store
            .with_connection(|db| {
                db.execute_batch("DROP TRIGGER reject_second_stream_dispatch;")?;
                Ok(())
            })
            .unwrap();

        let (dispatched, events) = store
            .commit_dispatched_action_wave(prepared_task, &dispatches, checkpoint)
            .unwrap();
        assert_eq!(events.len(), 2);
        assert!(
            actions.iter().all(|proposal| {
                dispatched.actions[&proposal.id].status == ActionStatus::Running
            })
        );
        let committed_checkpoint = store.load_procedure_checkpoint(task_id).unwrap().unwrap();
        assert!(["producer", "consumer"].iter().all(|node_id| {
            committed_checkpoint.runtime.node_states()[*node_id]
                == crate::agency::ProcedureNodeState::Running
        }));
        let journal_states = store
            .with_connection(|db| {
                let mut statement = db.prepare(
                    "SELECT state FROM action_journal WHERE run_id=?1 ORDER BY action_id",
                )?;
                let rows = statement
                    .query_map([task_id.to_string()], |row| row.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .unwrap();
        assert_eq!(journal_states, vec!["dispatched", "dispatched"]);
    }

    #[test]
    fn procedure_dispatch_and_verified_output_commit_with_their_broker_receipts() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalStore::open_encrypted(
            &directory.path().join("procedure-transitions.db"),
            &SecretBytes::new(vec![53; 32]),
        )
        .unwrap();
        store.migrate_knowledge().unwrap();
        store.migrate_world_model().unwrap();

        let now = Utc::now();
        let target = crate::application_target::ApplicationTarget {
            platform: "macos".into(),
            bundle_path: "/Applications/Example.app".into(),
            identifier: "com.example.Editor".into(),
            code_digest: crate::application_target::ApplicationTarget::code_set_digest(&[
                "ab".repeat(20)
            ]),
            code_digests: vec!["ab".repeat(20)],
            signer: "TEAM".into(),
        };
        let system = store
            .observe_system(crate::world_model::SystemDescriptor {
                id: Uuid::new_v4(),
                kind: crate::world_model::SystemKind::Application,
                key: target.identifier.clone(),
                label: "Example Editor".into(),
                fingerprint: target.code_digest.clone(),
                revision: 1,
                updated_at: now,
            })
            .unwrap();
        let evidence_id = Uuid::new_v4();
        let control_id = "cd".repeat(32);
        let evidence = crate::world_model::ObservationEnvelope {
            id: evidence_id,
            system_id: system.id,
            session_id: None,
            worker_session: None,
            system_fingerprint: system.fingerprint.clone(),
            origin: crate::world_model::EvidenceOrigin::OperatingSystem,
            privacy: crate::contracts::Sensitivity::Private,
            observed_at: now,
            facts: vec![crate::world_model::ObservedFact {
                name: "control.value".into(),
                subject: Some(control_id.clone()),
                value: crate::world_model::FactValue::Number(0.5),
            }],
        };
        store.record_world_observation(&evidence).unwrap();
        let input_port = crate::world_model::DataPort {
            name: "value".into(),
            value_type: crate::world_model::PortType::Number,
            max_bytes: 8,
            privacy: crate::contracts::Sensitivity::Private,
        };
        let output_port = crate::world_model::DataPort {
            name: "observed_value".into(),
            ..input_port.clone()
        };
        let descriptor = crate::world_model::CapabilityDescriptor {
            schema_version: 1,
            id: "ui.control.slider.fixture".into(),
            system_id: system.id,
            system_fingerprint: system.fingerprint.clone(),
            interface_control_id: Some(control_id.clone()),
            interface_probe_kind: Some(crate::world_model::ProbeKind::RestoreSliderValue),
            label: "Set Volume".into(),
            input_ports: vec![input_port.clone()],
            output_ports: vec![output_port.clone()],
            preconditions: crate::world_model::Preconditions {
                observed_state_fact_ids: vec![evidence_id],
                description: "A current, restored slider observation".into(),
            },
            effects: BTreeSet::from([crate::contracts::Effect::ControlApplication]),
            verification: "Fresh signed-application slider readback".into(),
            restoration: Some("Restore the original slider value".into()),
            cancellation: "Settle the native control receipt".into(),
            executor_id: Some("set_application_control".into()),
            evidence_ids: vec![evidence_id],
            updated_at: now,
        };
        store.record_capability_candidate(&descriptor).unwrap();
        let procedure = crate::agency::ProcedureIr {
            schema_version: 1,
            id: "procedure-transition-test".into(),
            nodes: vec![
                crate::agency::ProcedureNode {
                    id: "set-volume".into(),
                    depends_on: BTreeSet::new(),
                    outputs: BTreeMap::from([(output_port.name.clone(), output_port.clone())]),
                    kind: crate::agency::ProcedureNodeKind::CapabilityCall {
                        capability_id: descriptor.id.clone(),
                        system_id: system.id,
                        system_fingerprint: system.fingerprint.clone(),
                        input_bindings: BTreeMap::from([(
                            "value".into(),
                            crate::agency::ValueBinding::Literal {
                                value: crate::agency::ProcedureValue::Number(0.5),
                                port: input_port.clone(),
                            },
                        )]),
                    },
                },
                crate::agency::ProcedureNode {
                    id: "set-volume-from-result".into(),
                    depends_on: BTreeSet::from(["set-volume".into()]),
                    outputs: BTreeMap::from([(output_port.name.clone(), output_port.clone())]),
                    kind: crate::agency::ProcedureNodeKind::CapabilityCall {
                        capability_id: descriptor.id.clone(),
                        system_id: system.id,
                        system_fingerprint: system.fingerprint.clone(),
                        input_bindings: BTreeMap::from([(
                            "value".into(),
                            crate::agency::ValueBinding::Result {
                                producer: "set-volume".into(),
                                output: "observed_value".into(),
                            },
                        )]),
                    },
                },
            ],
            streams: Vec::new(),
            completion: vec![crate::agency::CompletionCondition::AllNodesSucceeded],
        };
        let task_id = Uuid::new_v4();
        let action_id = Uuid::new_v4();
        let anchor = serde_json::json!({
            "id": control_id,
            "role": "slider",
            "label": "Volume",
            "enabled": true,
            "ancestors": []
        });
        let proposal = ActionProposal {
            id: action_id,
            task_id,
            action: Action::SetApplicationControl {
                application: system.key.clone(),
                system_id: system.id,
                system_fingerprint: system.fingerprint.clone(),
                capability_id: descriptor.id.clone(),
                control_id: control_id.clone(),
                value: crate::domain::ApplicationControlValue::Number(0.5),
            },
            expected_outcome: ExpectedOutcome::ApplicationControlValue {
                target: target.clone(),
                control_id: control_id.clone(),
                value: crate::domain::ApplicationControlValue::Number(0.5),
            },
            target_resource: system.key.clone(),
            provenance: Provenance::user(),
            metadata: BTreeMap::from([
                (
                    "application_target".into(),
                    serde_json::to_string(&target).unwrap(),
                ),
                ("capability_label".into(), "Set Volume".into()),
                ("application_control_anchor".into(), anchor.to_string()),
                ("procedure_id".into(), procedure.id.clone()),
                ("procedure_node_id".into(), "set-volume".into()),
            ]),
        };
        let mut task = Task::new("Set the editor volume");
        task.id = task_id;
        let conversation = store
            .ensure_conversation(None, "procedure transition")
            .unwrap();
        task.conversation_id = Some(conversation.id);
        task.message_id = Some(Uuid::new_v4());
        task.workflow_run = true;
        task.goal = Some(task.request.clone());
        task.status = TaskStatus::Running;
        task.actions.insert(
            action_id,
            ActionState {
                proposal: proposal.clone(),
                status: ActionStatus::Pending,
                attempts: 0,
                summary: None,
                error: None,
            },
        );
        task.dependencies.insert(action_id, BTreeSet::new());
        let message = crate::knowledge::Message {
            id: task.message_id.unwrap(),
            conversation_id: conversation.id,
            task_id: Some(task.id),
            role: "user".into(),
            content: task.request.clone(),
            provenance: Provenance::user(),
            created_at: now,
        };
        let initial = crate::agency::ProcedureCheckpoint {
            task_id,
            revision: 0,
            runtime: crate::agency::ProcedureRuntimeState::new_for_task(&procedure, task_id)
                .unwrap(),
            procedure: procedure.clone(),
            updated_at: now,
        };
        store
            .accept_task_with_procedure(
                &mut task,
                &message,
                &CoreEvent::new(Some(task_id), CoreEventKind::TaskStarted),
                None,
                &initial,
            )
            .unwrap();
        let prepared = PreparedAction::new(&proposal, BTreeSet::new()).unwrap();
        let (prepared_task, _) = store.commit_prepared_action(task, &prepared).unwrap();
        let grant = crate::capability::CapabilityGrant {
            id: Uuid::new_v4(),
            task_id,
            action_id,
            action_digest: prepared.action_digest.clone(),
            policy_version: crate::contracts::POLICY_VERSION,
            worker_session: None,
            domain: crate::domain::ExecutionDomain::Native,
            resource: crate::capability::CapabilityResource::Application {
                identifier: system.key.clone(),
            },
            operations: BTreeSet::from([crate::capability::CapabilityOperation::Control]),
            issued_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::minutes(1),
            remaining_uses: 1,
            revoked: false,
        };
        let assessment = crate::world_model::CapabilityAssessment {
            descriptor,
            evidence_state: crate::world_model::CapabilityEvidenceState::ReversiblyExperimented,
        };

        let mut dispatch_checkpoint = store.load_procedure_checkpoint(task_id).unwrap().unwrap();
        dispatch_checkpoint
            .runtime
            .record_dispatched(
                &procedure,
                std::slice::from_ref(&assessment),
                &BTreeSet::from([evidence_id]),
                "set-volume",
                action_id,
            )
            .unwrap();
        store
            .with_connection(|db| {
                db.execute_batch("CREATE TEMP TRIGGER reject_procedure_update BEFORE UPDATE ON procedure_checkpoints BEGIN SELECT RAISE(ABORT,'injected procedure checkpoint failure'); END;")?;
                Ok(())
            })
            .unwrap();
        assert!(
            store
                .commit_dispatched_action_with_procedure(
                    prepared_task.clone(),
                    &prepared,
                    Some(&grant),
                    "set_application_control",
                    Some(dispatch_checkpoint.clone()),
                )
                .is_err()
        );
        assert_eq!(
            store
                .load_procedure_checkpoint(task_id)
                .unwrap()
                .unwrap()
                .revision,
            1
        );
        assert_eq!(
            store.load_tasks(true).unwrap()[0].actions[&action_id].status,
            ActionStatus::Pending
        );
        store
            .with_connection(|db| {
                db.execute_batch("DROP TRIGGER reject_procedure_update;")?;
                Ok(())
            })
            .unwrap();
        let (dispatched_task, _) = store
            .commit_dispatched_action_with_procedure(
                prepared_task,
                &prepared,
                Some(&grant),
                "set_application_control",
                Some(dispatch_checkpoint),
            )
            .unwrap();
        let dispatched_checkpoint = store.load_procedure_checkpoint(task_id).unwrap().unwrap();
        assert_eq!(dispatched_checkpoint.revision, 2);
        assert_eq!(
            dispatched_checkpoint.runtime.node_states()["set-volume"],
            crate::agency::ProcedureNodeState::Running
        );

        let receipt = ExecutionReceipt {
            executor: "native-control".into(),
            summary: "Set the learned volume control".into(),
            transient_data: json!({"accepted":true}),
            rollback: None,
        };
        let observation = Observation {
            observed_at: Utc::now(),
            provenance: Provenance::external(ProvenanceSource::OperatingSystem, "control readback"),
            summary: "Freshly read the signed editor control".into(),
            evidence: vec![Evidence::ApplicationControlValue {
                target,
                process_id: 77,
                control_id,
                value: crate::domain::ApplicationControlValue::Number(0.5),
            }],
        };
        let mut change = VerifiedAction::from_observation(
            &dispatched_task,
            &proposal,
            &receipt,
            &observation,
            VerificationMode::Execution,
        )
        .unwrap();
        let mut verified_checkpoint = dispatched_checkpoint.clone();
        change
            .advance_procedure_checkpoint(&mut verified_checkpoint)
            .unwrap();
        let mut next_proposal = proposal.clone();
        next_proposal.id = Uuid::new_v4();
        next_proposal
            .metadata
            .insert("procedure_node_id".into(), "set-volume-from-result".into());
        next_proposal.provenance.parent_ids = vec![
            format!("capability:{}", assessment.descriptor.id),
            "node:set-volume-from-result".into(),
        ];
        change
            .task
            .append_plan_with_exact_dependencies(crate::domain::ActionGraph {
                goal: "Set the verified value again".into(),
                nodes: vec![crate::domain::ActionNode {
                    proposal: next_proposal.clone(),
                    depends_on: BTreeSet::from([action_id]),
                }],
            })
            .unwrap();
        store
            .with_connection(|db| {
                db.execute_batch("CREATE TEMP TRIGGER reject_procedure_update BEFORE UPDATE ON procedure_checkpoints BEGIN SELECT RAISE(ABORT,'injected procedure checkpoint failure'); END;")?;
                Ok(())
            })
            .unwrap();
        assert!(
            store
                .commit_verified_action_with_procedure(change, Some(verified_checkpoint))
                .is_err()
        );
        assert_eq!(
            store
                .load_procedure_checkpoint(task_id)
                .unwrap()
                .unwrap()
                .revision,
            2
        );
        assert_eq!(
            store.load_tasks(true).unwrap()[0].actions[&action_id].status,
            ActionStatus::Running
        );
        assert_eq!(store.load_tasks(true).unwrap()[0].actions.len(), 1);
        store
            .with_connection(|db| {
                db.execute_batch("DROP TRIGGER reject_procedure_update;")?;
                Ok(())
            })
            .unwrap();

        let current_task = store.load_tasks(true).unwrap().remove(0);
        let mut change = VerifiedAction::from_observation(
            &current_task,
            &proposal,
            &receipt,
            &observation,
            VerificationMode::Execution,
        )
        .unwrap();
        let mut verified_checkpoint = store.load_procedure_checkpoint(task_id).unwrap().unwrap();
        change
            .advance_procedure_checkpoint(&mut verified_checkpoint)
            .unwrap();
        change
            .task
            .append_plan_with_exact_dependencies(crate::domain::ActionGraph {
                goal: "Set the verified value again".into(),
                nodes: vec![crate::domain::ActionNode {
                    proposal: next_proposal.clone(),
                    depends_on: BTreeSet::from([action_id]),
                }],
            })
            .unwrap();
        store
            .commit_verified_action_with_procedure(change, Some(verified_checkpoint))
            .unwrap();
        let completed_checkpoint = store.load_procedure_checkpoint(task_id).unwrap().unwrap();
        assert_eq!(completed_checkpoint.revision, 3);
        assert_eq!(
            completed_checkpoint.runtime.node_states()["set-volume"],
            crate::agency::ProcedureNodeState::Succeeded
        );
        assert!(
            !completed_checkpoint
                .runtime
                .completion_satisfied(&completed_checkpoint.procedure)
                .unwrap()
        );
        assert_eq!(
            store.load_tasks(true).unwrap()[0].tool_results[0].output["procedure_outputs"]["observed_value"]
                ["output"]["type"],
            "number"
        );
        let completed_task = store.load_tasks(true).unwrap().remove(0);
        assert_eq!(completed_task.actions.len(), 2);
        assert_eq!(
            completed_task.actions[&next_proposal.id].status,
            ActionStatus::Pending
        );
        assert_eq!(
            completed_task.dependencies[&next_proposal.id],
            BTreeSet::from([action_id])
        );

        let uncertain_task_id = Uuid::new_v4();
        let mut uncertain_proposal = proposal.clone();
        uncertain_proposal.id = Uuid::new_v4();
        uncertain_proposal.task_id = uncertain_task_id;
        let mut uncertain_task = Task::new("Set the editor volume and reconcile its outcome");
        uncertain_task.id = uncertain_task_id;
        let uncertain_conversation = store
            .ensure_conversation(None, "uncertain procedure transition")
            .unwrap();
        uncertain_task.conversation_id = Some(uncertain_conversation.id);
        uncertain_task.message_id = Some(Uuid::new_v4());
        uncertain_task.workflow_run = true;
        uncertain_task.goal = Some(uncertain_task.request.clone());
        uncertain_task.status = TaskStatus::Running;
        uncertain_task.actions.insert(
            uncertain_proposal.id,
            ActionState {
                proposal: uncertain_proposal.clone(),
                status: ActionStatus::Pending,
                attempts: 0,
                summary: None,
                error: None,
            },
        );
        uncertain_task
            .dependencies
            .insert(uncertain_proposal.id, BTreeSet::new());
        let uncertain_message = crate::knowledge::Message {
            id: uncertain_task.message_id.unwrap(),
            conversation_id: uncertain_conversation.id,
            task_id: Some(uncertain_task_id),
            role: "user".into(),
            content: uncertain_task.request.clone(),
            provenance: Provenance::user(),
            created_at: Utc::now(),
        };
        let uncertain_initial = crate::agency::ProcedureCheckpoint {
            task_id: uncertain_task_id,
            revision: 0,
            runtime: crate::agency::ProcedureRuntimeState::new_for_task(
                &procedure,
                uncertain_task_id,
            )
            .unwrap(),
            procedure: procedure.clone(),
            updated_at: Utc::now(),
        };
        store
            .accept_task_with_procedure(
                &mut uncertain_task,
                &uncertain_message,
                &CoreEvent::new(Some(uncertain_task_id), CoreEventKind::TaskStarted),
                None,
                &uncertain_initial,
            )
            .unwrap();
        let uncertain_prepared = PreparedAction::new(&uncertain_proposal, BTreeSet::new()).unwrap();
        let (uncertain_prepared_task, _) = store
            .commit_prepared_action(uncertain_task, &uncertain_prepared)
            .unwrap();
        let mut uncertain_checkpoint = store
            .load_procedure_checkpoint(uncertain_task_id)
            .unwrap()
            .unwrap();
        uncertain_checkpoint
            .runtime
            .record_dispatched(
                &procedure,
                std::slice::from_ref(&assessment),
                &BTreeSet::from([evidence_id]),
                "set-volume",
                uncertain_proposal.id,
            )
            .unwrap();
        let mut uncertain_grant = grant.clone();
        uncertain_grant.id = Uuid::new_v4();
        uncertain_grant.task_id = uncertain_task_id;
        uncertain_grant.action_id = uncertain_proposal.id;
        uncertain_grant.action_digest = uncertain_prepared.action_digest.clone();
        uncertain_grant.issued_at = Utc::now();
        uncertain_grant.expires_at = Utc::now() + chrono::Duration::minutes(1);
        let (uncertain_dispatched, _) = store
            .commit_dispatched_action_with_procedure(
                uncertain_prepared_task,
                &uncertain_prepared,
                Some(&uncertain_grant),
                "set_application_control",
                Some(uncertain_checkpoint),
            )
            .unwrap();
        let (interrupted, _, may_have_executed) = store
            .commit_interrupted_action(
                uncertain_dispatched,
                uncertain_proposal.id,
                "native worker reply was lost",
            )
            .unwrap();
        assert!(may_have_executed);
        assert_eq!(interrupted.status, TaskStatus::Interrupted);
        assert_eq!(
            store
                .load_procedure_checkpoint(uncertain_task_id)
                .unwrap()
                .unwrap()
                .runtime
                .node_states()["set-volume"],
            crate::agency::ProcedureNodeState::Uncertain
        );
    }

    #[test]
    fn each_failed_write_rolls_back_the_entire_verified_transition() {
        for table in [
            "private_artifacts",
            "tasks",
            "actions",
            "events",
            "audit_log",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let (store, task, proposal, receipt, observation) =
                fixture(&dir.path().join("test.db"));
            store.with_connection(|db| { db.execute_batch(&format!("CREATE TEMP TRIGGER fail_verified_write BEFORE INSERT ON {table} BEGIN SELECT RAISE(ABORT,'injected persistence failure'); END;"))?; Ok(()) }).unwrap();
            let change = VerifiedAction::from_observation(
                &task,
                &proposal,
                &receipt,
                &observation,
                VerificationMode::Execution,
            )
            .unwrap();
            assert!(store.commit_verified_action(change).is_err(), "{table}");
            assert_eq!(store.load_tasks(true).unwrap(), vec![task], "{table}");
            store
                .with_connection(|db| {
                    let (state, evidence): (String, Option<String>) = db.query_row(
                        "SELECT state,verification_json FROM action_journal",
                        [],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )?;
                    assert_eq!(state, "dispatched", "{table}");
                    assert!(evidence.is_none(), "{table}");
                    for empty in ["private_artifacts", "events", "audit_log"] {
                        let count: i64 =
                            db.query_row(&format!("SELECT COUNT(*) FROM {empty}"), [], |row| {
                                row.get(0)
                            })?;
                        assert_eq!(count, 0, "Failed {table} left rows in {empty}");
                    }
                    Ok(())
                })
                .unwrap();
        }
    }

    #[test]
    fn stale_verification_cannot_overwrite_a_newer_stop_revision() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mut task, proposal, receipt, observation) =
            fixture(&dir.path().join("test.db"));
        let mut stale = task.clone();
        let change = VerifiedAction::from_observation(
            &stale,
            &proposal,
            &receipt,
            &observation,
            VerificationMode::Execution,
        )
        .unwrap();
        task.status = TaskStatus::Cancelled;
        store.save_task(&mut task).unwrap();
        assert!(store.commit_verified_action(change).is_err());
        assert!(store.save_task(&mut stale).is_err());
        assert_eq!(stale.revision + 1, task.revision);
        assert_eq!(store.load_tasks(true).unwrap(), vec![task]);
        store
            .with_connection(|db| {
                let state: String =
                    db.query_row("SELECT state FROM action_journal", [], |row| row.get(0))?;
                assert_eq!(state, "dispatched");
                let artifacts: i64 =
                    db.query_row("SELECT COUNT(*) FROM private_artifacts", [], |row| {
                        row.get(0)
                    })?;
                assert_eq!(artifacts, 0);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn prepare_dispatch_and_interruption_commit_with_their_projection_and_events() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mut task, proposal, _, _) = fixture(&dir.path().join("test.db"));
        store
            .with_connection(|db| {
                db.execute("DELETE FROM action_journal", [])?;
                Ok(())
            })
            .unwrap();
        task.actions.get_mut(&proposal.id).unwrap().status = ActionStatus::Compiling;
        store.save_task(&mut task).unwrap();
        let prepared = PreparedAction::new(&proposal, Default::default()).unwrap();
        let fail = || {
            store.with_connection(|db| { db.execute_batch("CREATE TEMP TRIGGER transition_failure BEFORE INSERT ON audit_log BEGIN SELECT RAISE(ABORT,'fixture commit failure'); END;")?; Ok(()) }).unwrap()
        };
        let repair = || {
            store
                .with_connection(|db| {
                    db.execute_batch("DROP TRIGGER transition_failure;")?;
                    Ok(())
                })
                .unwrap()
        };
        fail();
        assert!(
            store
                .commit_prepared_action(task.clone(), &prepared)
                .is_err()
        );
        assert_eq!(store.load_tasks(true).unwrap(), vec![task.clone()]);
        store
            .with_connection(|db| {
                let count: i64 =
                    db.query_row("SELECT COUNT(*) FROM action_journal", [], |row| row.get(0))?;
                assert_eq!(count, 0);
                Ok(())
            })
            .unwrap();
        repair();
        let (prepared_task, _) = store.commit_prepared_action(task, &prepared).unwrap();
        fail();
        assert!(
            store
                .commit_dispatched_action(prepared_task.clone(), &prepared, None, "fixture")
                .is_err()
        );
        assert_eq!(store.load_tasks(true).unwrap(), vec![prepared_task.clone()]);
        store
            .with_connection(|db| {
                let state: String =
                    db.query_row("SELECT state FROM action_journal", [], |row| row.get(0))?;
                assert_eq!(state, "prepared");
                Ok(())
            })
            .unwrap();
        repair();
        let (mut dispatched, _) = store
            .commit_dispatched_action(prepared_task, &prepared, None, "fixture")
            .unwrap();
        assert_eq!(
            dispatched.actions[&proposal.id].status,
            ActionStatus::Running
        );
        // Simulate a legacy projection error. Durable dispatch still makes the
        // effect uncertain, regardless of this misleading status field.
        dispatched.actions.get_mut(&proposal.id).unwrap().status = ActionStatus::Failed;
        store.save_task(&mut dispatched).unwrap();
        fail();
        assert!(
            store
                .commit_interrupted_action(dispatched.clone(), proposal.id, "worker reply lost")
                .is_err()
        );
        assert_eq!(store.load_tasks(true).unwrap(), vec![dispatched.clone()]);
        repair();
        let (interrupted, _, may_have_executed) = store
            .commit_interrupted_action(dispatched, proposal.id, "worker reply lost")
            .unwrap();
        assert!(may_have_executed);
        assert_eq!(interrupted.status, TaskStatus::Interrupted);
        assert_eq!(
            interrupted.actions[&proposal.id].status,
            ActionStatus::Uncertain
        );
        assert_eq!(
            interrupted.tool_results.last().unwrap().verdict,
            Verdict::Uncertain
        );
        assert!(
            store
                .commit_prepared_action(interrupted.clone(), &prepared)
                .is_err()
        );
        store
            .with_connection(|db| {
                let state: String =
                    db.query_row("SELECT state FROM action_journal", [], |row| row.get(0))?;
                assert_eq!(state, "uncertain");
                let count: i64 =
                    db.query_row("SELECT COUNT(*) FROM events", [], |row| row.get(0))?;
                assert_eq!(count, 3);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn abrupt_transition_exit_child() {
        let Some(path) = std::env::var_os("SAGE_TEST_VERIFICATION_DB") else {
            return;
        };
        let (store, task, proposal, receipt, observation) = fixture(std::path::Path::new(&path));
        let change = VerifiedAction::from_observation(
            &task,
            &proposal,
            &receipt,
            &observation,
            VerificationMode::Execution,
        )
        .unwrap();
        store.commit_verified_action(change).unwrap();
        panic!("Expected an abrupt exit at the configured persistence boundary");
    }

    #[test]
    fn abrupt_exit_at_each_boundary_recovers_all_or_none_of_the_verification() {
        for stage in ["journal", "artifact", "task", "event", "audit", "committed"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("crash.db");
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "transitions::tests::abrupt_transition_exit_child",
                    "--nocapture",
                ])
                .env("SAGE_TEST_VERIFICATION_DB", &path)
                .env("SAGE_TEST_VERIFICATION_EXIT_AT", stage)
                .output()
                .unwrap();
            assert_eq!(
                child.status.code(),
                Some(86),
                "{stage}: {}",
                String::from_utf8_lossy(&child.stderr)
            );
            let store = LocalStore::open_encrypted(&path, &SecretBytes::new(vec![29; 32])).unwrap();
            let tasks = store.load_tasks(true).unwrap();
            assert_eq!(tasks.len(), 1);
            assert_eq!(
                tasks[0].status,
                TaskStatus::Interrupted,
                "Restart never resumes execution"
            );
            let committed = stage == "committed";
            assert_eq!(
                tasks[0].completed_count(),
                u32::from(committed) as usize,
                "{stage}"
            );
            assert_eq!(
                tasks[0].tool_results.len(),
                usize::from(committed),
                "{stage}"
            );
            store
                .with_connection(|db| {
                    let state: String =
                        db.query_row("SELECT state FROM action_journal", [], |row| row.get(0))?;
                    assert_eq!(
                        state,
                        if committed { "confirmed" } else { "dispatched" },
                        "{stage}"
                    );
                    for table in ["private_artifacts", "events", "audit_log"] {
                        let count: i64 =
                            db.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                                row.get(0)
                            })?;
                        assert_eq!(count, i64::from(committed), "{stage}: {table}");
                    }
                    Ok(())
                })
                .unwrap();
            store
                .checkpoint_audit(&MemorySecretStore::default())
                .unwrap();
        }
    }

    #[test]
    fn durable_verification_survives_reopen_without_promoting_a_cancelled_task() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let (store, mut task, proposal, receipt, observation) = fixture(&path);
        task.status = TaskStatus::Cancelled;
        store.save_task(&mut task).unwrap();
        let change = VerifiedAction::from_observation(
            &task,
            &proposal,
            &receipt,
            &observation,
            VerificationMode::Execution,
        )
        .unwrap();
        let (committed, event) = store.commit_verified_action(change).unwrap();
        assert_eq!(committed.status, TaskStatus::Cancelled);
        assert_eq!(committed.completed_count(), 1);
        let artifact: Uuid =
            serde_json::from_value(committed.tool_results[0].output["artifact_ref"].clone())
                .unwrap();
        let secrets = MemorySecretStore::default();
        store.checkpoint_audit(&secrets).unwrap();
        drop(store);
        let reopened = LocalStore::open_encrypted(&path, &SecretBytes::new(vec![29; 32])).unwrap();
        assert_eq!(reopened.load_tasks(true).unwrap(), vec![committed]);
        assert_eq!(
            reopened.read_artifact(artifact).unwrap(),
            b"retained fixture response"
        );
        reopened.checkpoint_audit(&secrets).unwrap();
        reopened
            .with_connection(|db| {
                let event_count: i64 = db.query_row(
                    "SELECT COUNT(*) FROM events WHERE id=?1",
                    [event.id.to_string()],
                    |row| row.get(0),
                )?;
                assert_eq!(event_count, 1);
                let state: String =
                    db.query_row("SELECT state FROM action_journal", [], |row| row.get(0))?;
                assert_eq!(state, "confirmed");
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn recovery_installs_matching_result_evidence_and_execution_replay_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (store, task, proposal, receipt, observation) = fixture(&dir.path().join("test.db"));
        store
            .journal_interrupted(task.id, proposal.id, true)
            .unwrap();
        let recovery = VerifiedAction::from_observation(
            &task,
            &proposal,
            &receipt,
            &observation,
            VerificationMode::Recovery,
        )
        .unwrap();
        let (committed, _) = store.commit_verified_action(recovery).unwrap();
        assert_eq!(committed.completed_count(), 1);
        assert_eq!(
            committed.tool_results.last().unwrap().verdict,
            Verdict::Confirmed
        );
        let replay = VerifiedAction::from_observation(
            &committed,
            &proposal,
            &receipt,
            &observation,
            VerificationMode::Execution,
        )
        .unwrap();
        assert!(store.commit_verified_action(replay).is_err());
        let mut mismatched = proposal.clone();
        mismatched.target_resource = "another target".into();
        assert!(
            VerifiedAction::from_observation(
                &committed,
                &mismatched,
                &receipt,
                &observation,
                VerificationMode::Recovery
            )
            .is_err()
        );
    }

    #[test]
    fn verification_after_forgetting_keeps_only_effect_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let (store, task, proposal, receipt, observation) = fixture(&dir.path().join("test.db"));
        let change = VerifiedAction::from_observation(
            &task,
            &proposal,
            &receipt,
            &observation,
            VerificationMode::Execution,
        )
        .unwrap();
        // Forgetting wins even when the in-memory result was prepared earlier.
        store
            .with_connection(|db| {
                db.execute(
                    "INSERT INTO retired_task_data VALUES(?1,CURRENT_TIMESTAMP)",
                    [task.id.to_string()],
                )?;
                Ok(())
            })
            .unwrap();
        let (committed, _) = store.commit_verified_action(change).unwrap();
        assert_eq!(committed.completed_count(), 1);
        assert_eq!(
            committed.tool_results.last().unwrap().output,
            json!({"retention_revoked":true})
        );
        store
            .with_connection(|db| {
                let count: i64 =
                    db.query_row("SELECT COUNT(*) FROM private_artifacts", [], |row| {
                        row.get(0)
                    })?;
                assert_eq!(count, 0);
                Ok(())
            })
            .unwrap();
    }
}
