use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use chrono::{Duration as ChronoDuration, Utc};
use serde_json::json;
use sha2::Digest;
use tokio::sync::{Mutex, Notify, RwLock, oneshot};
use tokio::time::{Duration, timeout};
use uuid::Uuid;

use crate::capability::CapabilityBroker;
use crate::compiler::{ActionCompiler, ExecutorAvailability};
use crate::config::CoreConfig;
use crate::domain::{
    Action, ActionGraph, ActionNode, ActionProposal, ActionStatus, ExpectedOutcome, Provenance,
    Task, TaskStatus,
};
use crate::error::{CoreError, CoreResult};
use crate::events::{CoreEvent, CoreEventKind, EventHub, StateSnapshot};
use crate::execution::bridge::{AdapterBridge, BrowserExecutor, PlatformObserver};
use crate::execution::{ExecutionBroker, ExecutionReceipt, NativeExecutor};
use crate::knowledge::{KnowledgeSnapshot, Message};
use crate::model::{ModelProvider, ModelTurn, ToolDescriptor, TurnContext};
use crate::observation::Observer;
use crate::policy::{PolicyContext, PolicyDecision, PolicyEngine, RiskLevel};
use crate::redaction::redact_for_persistence;
use crate::resources::ResourceResolver;
#[cfg(test)]
use crate::secrets::SecretBytes;
use crate::secrets::{OsSecretStore, SecretStore};
use crate::storage::LocalStore;
use crate::verification::Verifier;

const APPROVAL_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const QUESTION_TIMEOUT: Duration = Duration::from_secs(30 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalResolution {
    Approved {
        native_authentication_satisfied: bool,
    },
    Denied,
}

struct PendingApproval {
    task_id: Uuid,
    action_id: Uuid,
    digest: String,
    requires_native_authentication: bool,
    expires_at: chrono::DateTime<Utc>,
    record: crate::contracts::ApprovalRecord,
    sender: oneshot::Sender<ApprovalResolution>,
}

struct PendingQuestion {
    task_id: Uuid,
    action_id: Uuid,
    record: crate::decisions::QuestionRecord,
    sender: oneshot::Sender<String>,
}

struct StepFailure {
    error: CoreError,
    recoverable: bool,
    observation: serde_json::Value,
}

/// Once a probe lease is durably dispatched, any early return or cancelled
/// future must leave a durable review obligation for the exact learning
/// session. A successful independent restoration receipt disarms the guard.
struct ProbeSettlementGuard {
    store: LocalStore,
    session_id: Uuid,
    armed: bool,
}

impl ProbeSettlementGuard {
    fn new(store: LocalStore, session_id: Uuid) -> Self {
        Self {
            store,
            session_id,
            armed: true,
        }
    }

    fn settled(&mut self) {
        self.armed = false;
    }
}

impl Drop for ProbeSettlementGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.store.stop_learning_session(
                self.session_id,
                crate::world_model::LearningSessionState::InterruptedNeedsReview,
            );
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AdapterReadback {
    value: serde_json::Value,
    observed_at: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AdapterProbeResponse {
    application_target: crate::application_target::ApplicationTarget,
    observed_process_id: u32,
    control_id: String,
    restoration_verified: bool,
    observations: Vec<AdapterReadback>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BrowserDiscoveryControl {
    role: String,
    kind: String,
    label: String,
    enabled: bool,
    ancestors: Vec<String>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BrowserDiscoveryResponse {
    available: bool,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    browser_target: Option<crate::browser_target::BrowserTarget>,
    #[serde(default)]
    truncated: bool,
    #[serde(default)]
    controls: Vec<BrowserDiscoveryControl>,
}

#[derive(Debug, serde::Serialize)]
struct BrowserSemanticControl {
    id: String,
    role: String,
    kind: String,
    label: String,
    enabled: bool,
    ancestors: Vec<String>,
}

struct PreparedBrowserDiscovery {
    system: crate::world_model::SystemDescriptor,
    observation: crate::world_model::ObservationEnvelope,
    interface_fingerprint: String,
    observed_controls: usize,
    skipped_controls: usize,
    truncated: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplicationDiscoveryControl {
    id: String,
    role: String,
    label: String,
    enabled: bool,
    #[serde(default)]
    ancestors: Vec<String>,
    value: Option<serde_json::Value>,
    step: Option<f64>,
    minimum: Option<f64>,
    maximum: Option<f64>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplicationDiscoveryResponse {
    application_target: crate::application_target::ApplicationTarget,
    observed_process_id: u32,
    application_name: String,
    bundle_identifier: String,
    accessibility_available: bool,
    active_window: String,
    #[serde(default)]
    truncated: bool,
    controls: Vec<ApplicationDiscoveryControl>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ApplicationSemanticControl {
    id: String,
    role: String,
    label: String,
    enabled: bool,
    ancestors: Vec<String>,
}

#[derive(Debug, Clone)]
struct ValidatedApplicationControl {
    semantic: ApplicationSemanticControl,
    value: Option<serde_json::Value>,
    step: Option<f64>,
    minimum: Option<f64>,
    maximum: Option<f64>,
}

struct PreparedApplicationControls {
    controls: Vec<ValidatedApplicationControl>,
    semantic_controls: Vec<ApplicationSemanticControl>,
    facts: Vec<crate::world_model::ObservedFact>,
    interface_fingerprint: String,
    skipped_controls: usize,
    truncated: bool,
}

fn build_learned_application_control_graph(
    request: &str,
    task_id: Uuid,
    discovered: Vec<(
        crate::world_model::SystemDescriptor,
        crate::world_model::CapabilityAssessment,
    )>,
) -> CoreResult<Option<ActionGraph>> {
    let Some((requested_label, value)) = crate::intent::parse_control_assignment(request) else {
        return Ok(None);
    };
    let normalize = |text: &str| {
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let requested_label = normalize(&requested_label);
    let mut matches = discovered.into_iter().filter(|(system, assessment)| {
        let descriptor = &assessment.descriptor;
        system.kind == crate::world_model::SystemKind::Application
            && descriptor.system_id == system.id
            && descriptor.system_fingerprint == system.fingerprint
            && descriptor
                .label
                .strip_prefix("Set ")
                .is_some_and(|label| normalize(label) == requested_label)
    });
    let Some((system, assessment)) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err(CoreError::InvalidAction(
            "That learned control name matches more than one current application; use a more specific control name".into(),
        ));
    }

    let descriptor = assessment.descriptor;
    if assessment.evidence_state
        != crate::world_model::CapabilityEvidenceState::ReversiblyExperimented
        || descriptor.executor_id.as_deref() != Some("set_application_control")
    {
        return Ok(None);
    }
    descriptor.validate()?;
    let Some(control_id) = descriptor.interface_control_id.clone() else {
        return Ok(None);
    };
    let type_matches = matches!(
        (descriptor.interface_probe_kind, &value),
        (
            Some(crate::world_model::ProbeKind::RestoreSliderValue),
            crate::domain::ApplicationControlValue::Number(_)
        ) | (
            Some(crate::world_model::ProbeKind::RestoreToggleState),
            crate::domain::ApplicationControlValue::Boolean(_)
        )
    );
    if !type_matches {
        return Err(CoreError::InvalidAction(
            "The requested value type does not match the learned slider or toggle".into(),
        ));
    }
    let proposal = ActionProposal {
        id: Uuid::new_v4(),
        task_id,
        action: Action::SetApplicationControl {
            application: system.key.clone(),
            system_id: system.id,
            system_fingerprint: system.fingerprint,
            capability_id: descriptor.id,
            control_id,
            value,
        },
        // The standard executor replaces this placeholder from its sealed
        // verifier contract before the action is committed as prepared.
        expected_outcome: ExpectedOutcome::UserAnswered,
        target_resource: system.key,
        provenance: Provenance::user(),
        metadata: BTreeMap::from([("intent_compiler".into(), "learned_control_v1".into())]),
    };
    let graph = ActionGraph {
        goal: request.trim().to_owned(),
        nodes: vec![ActionNode {
            proposal,
            depends_on: BTreeSet::new(),
        }],
    };
    graph.validate(task_id).map_err(CoreError::InvalidAction)?;
    Ok(Some(graph))
}

/// Compile one currently ready wave from the supported, data-only procedure
/// subset into Sage's ordinary task graph. Result bindings are resolved only
/// from values already committed by independent verification. The task runner
/// still performs fresh target resolution, policy, permission, one-use
/// capability, execution, and independent verification for every action.
struct ProcedureCompilationContext<'a> {
    systems: &'a [crate::world_model::SystemDescriptor],
    assessments: &'a [crate::world_model::CapabilityAssessment],
    current_evidence_ids: &'a BTreeSet<Uuid>,
    runtime: &'a crate::agency::ProcedureRuntimeState,
    selected_node_ids: &'a [String],
    existing_action_ids: &'a BTreeMap<String, Uuid>,
}

fn compile_application_control_procedure(
    procedure: &crate::agency::ProcedureIr,
    request: &str,
    task_id: Uuid,
    context: ProcedureCompilationContext<'_>,
) -> CoreResult<ActionGraph> {
    use crate::agency::{
        CompletionCondition, ProcedureNodeKind, ProcedureValue, ResolvedProcedureInput,
    };
    use crate::contracts::Effect;
    use crate::domain::{ProvenanceSource, TrustClass};
    use crate::world_model::{CapabilityEvidenceState, PortType, ProbeKind, SystemKind};

    let ProcedureCompilationContext {
        systems,
        assessments,
        current_evidence_ids,
        runtime,
        selected_node_ids,
        existing_action_ids,
    } = context;

    procedure.validate()?;
    if procedure.nodes.len() > 32
        || !procedure.streams.is_empty()
        || procedure.completion.len() != 1
        || !matches!(
            procedure.completion[0],
            CompletionCondition::AllNodesSucceeded
        )
        || procedure
            .nodes
            .iter()
            .any(|node| !matches!(node.kind, ProcedureNodeKind::CapabilityCall { .. }))
    {
        return Err(CoreError::ExecutorUnavailable(
            "The task runner currently accepts bounded non-streaming procedures that complete after all nodes succeed".into(),
        ));
    }

    let system_by_id = systems
        .iter()
        .map(|system| (system.id, system))
        .collect::<BTreeMap<_, _>>();
    let assessment_by_id = assessments
        .iter()
        .map(|assessment| (assessment.descriptor.id.as_str(), assessment))
        .collect::<BTreeMap<_, _>>();
    if assessment_by_id.len() != assessments.len() {
        return Err(CoreError::VerificationFailed(
            "Current capability identities are ambiguous".into(),
        ));
    }

    let selected_count = selected_node_ids.len();
    let selected_node_ids = selected_node_ids.iter().cloned().collect::<BTreeSet<_>>();
    if selected_node_ids.is_empty()
        || selected_node_ids.len() != selected_count
        || selected_node_ids
            .iter()
            .any(|id| !procedure.nodes.iter().any(|node| node.id == *id))
    {
        return Err(CoreError::InvalidAction(
            "Procedure execution requires a non-empty unique ready-node wave".into(),
        ));
    }
    let mut action_ids = existing_action_ids.clone();
    for node_id in &selected_node_ids {
        action_ids
            .entry(node_id.clone())
            .or_insert_with(Uuid::new_v4);
    }
    let capabilities = assessments
        .iter()
        .map(|assessment| assessment.descriptor.clone())
        .collect::<Vec<_>>();
    let mut nodes = Vec::with_capacity(selected_node_ids.len());
    for node in &procedure.nodes {
        let ProcedureNodeKind::CapabilityCall {
            capability_id,
            system_id,
            system_fingerprint,
            input_bindings,
        } = &node.kind
        else {
            return Err(CoreError::ExecutorUnavailable(
                "Branch, repeat, and nested-skill nodes need their product runtime before execution".into(),
            ));
        };
        let assessment = assessment_by_id
            .get(capability_id.as_str())
            .ok_or_else(|| {
                CoreError::ExecutorUnavailable("Procedure capability is no longer current".into())
            })?;
        let descriptor = &assessment.descriptor;
        let system = system_by_id.get(system_id).ok_or_else(|| {
            CoreError::ExecutorUnavailable("Procedure target system is no longer current".into())
        })?;
        if assessment.evidence_state != CapabilityEvidenceState::ReversiblyExperimented
            || descriptor.executor_id.as_deref() != Some("set_application_control")
            || descriptor.system_id != *system_id
            || descriptor.system_fingerprint != *system_fingerprint
            || system.kind != SystemKind::Application
            || system.fingerprint != *system_fingerprint
            || descriptor.effects != BTreeSet::from([Effect::ControlApplication])
            || descriptor.input_ports.len() != 1
            || descriptor.input_ports[0].name != "value"
            || descriptor.output_ports.len() != 1
            || descriptor.output_ports[0].name != "observed_value"
            || descriptor
                .evidence_ids
                .iter()
                .chain(&descriptor.preconditions.observed_state_fact_ids)
                .any(|evidence| !current_evidence_ids.contains(evidence))
        {
            return Err(CoreError::PermissionRequired(
                "Procedure capability lacks a current restored application-control assessment"
                    .into(),
            ));
        }
        descriptor.validate()?;
        let control_id = descriptor.interface_control_id.clone().ok_or_else(|| {
            CoreError::VerificationFailed(
                "Learned capability has no semantic control identity".into(),
            )
        })?;
        let (expected_probe, port_type) = match descriptor.interface_probe_kind {
            Some(ProbeKind::RestoreSliderValue) => {
                (ProbeKind::RestoreSliderValue, PortType::Number)
            }
            Some(ProbeKind::RestoreToggleState) => {
                (ProbeKind::RestoreToggleState, PortType::Boolean)
            }
            None => {
                return Err(CoreError::PermissionRequired(
                    "Application capability is not bound to a reversible learning probe".into(),
                ));
            }
        };
        if descriptor.input_ports[0].value_type != port_type
            || descriptor.output_ports[0].value_type != port_type
            || node.outputs.get("observed_value") != Some(&descriptor.output_ports[0])
        {
            return Err(CoreError::VerificationFailed(
                "Procedure ports differ from the current learned-control contract".into(),
            ));
        }
        if !selected_node_ids.contains(&node.id) {
            continue;
        }
        if input_bindings.len() != 1 || !input_bindings.contains_key("value") {
            return Err(CoreError::ExecutorUnavailable(
                "Application-control procedures require one typed value input per step".into(),
            ));
        }
        let resolved_inputs =
            crate::agency::resolve_call_inputs(node, procedure, runtime, &capabilities)
                .ok_or_else(|| {
                    CoreError::PermissionRequired(
                        "Procedure input is not backed by a completed, verified dependency".into(),
                    )
                })?;
        let Some(ResolvedProcedureInput::Value {
            value,
            source_port,
            target_port,
        }) = resolved_inputs.get("value")
        else {
            return Err(CoreError::VerificationFailed(
                "Application control accepts only a resolved scalar value".into(),
            ));
        };
        if source_port.value_type != port_type
            || target_port != &descriptor.input_ports[0]
            || target_port.value_type != port_type
        {
            return Err(CoreError::VerificationFailed(
                "Procedure input does not match the exact typed control contract".into(),
            ));
        }
        let value = match (expected_probe, value) {
            (ProbeKind::RestoreSliderValue, ProcedureValue::Number(value)) if value.is_finite() => {
                crate::domain::ApplicationControlValue::Number(*value)
            }
            (ProbeKind::RestoreToggleState, ProcedureValue::Boolean(value)) => {
                crate::domain::ApplicationControlValue::Boolean(*value)
            }
            _ => {
                return Err(CoreError::InvalidAction(
                    "Procedure value type does not match the learned slider or toggle".into(),
                ));
            }
        };
        let action_id = action_ids[&node.id];
        let mut metadata = BTreeMap::from([
            ("procedure_id".into(), procedure.id.clone()),
            ("procedure_node_id".into(), node.id.clone()),
        ]);
        if procedure
            .streams
            .iter()
            .any(|channel| channel.producer_node == node.id || channel.consumer_node == node.id)
        {
            metadata.insert("procedure_stream_node".into(), "true".into());
        }
        let proposal = ActionProposal {
            id: action_id,
            task_id,
            action: Action::SetApplicationControl {
                application: system.key.clone(),
                system_id: *system_id,
                system_fingerprint: system_fingerprint.clone(),
                capability_id: capability_id.clone(),
                control_id,
                value,
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: system.key.clone(),
            provenance: crate::domain::Provenance {
                source: ProvenanceSource::SageCore,
                trust: TrustClass::TrustedComponent,
                source_id: Some(format!("procedure:{}", procedure.id)),
                parent_ids: vec![
                    format!("capability:{capability_id}"),
                    format!("node:{}", node.id),
                ],
            },
            metadata,
        };
        let depends_on = node
            .depends_on
            .iter()
            .map(|dependency| action_ids.get(dependency).copied())
            .collect::<Option<BTreeSet<_>>>()
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "A ready procedure call has no task action for a dependency".into(),
                )
            })?;
        nodes.push(crate::domain::ActionNode {
            proposal,
            depends_on,
        });
    }
    let graph = ActionGraph {
        goal: request.trim().to_owned(),
        nodes,
    };
    let graph_ids = graph
        .nodes
        .iter()
        .map(|node| node.proposal.id)
        .collect::<BTreeSet<_>>();
    let mut internally_validated = graph.clone();
    for node in &mut internally_validated.nodes {
        node.depends_on
            .retain(|dependency| graph_ids.contains(dependency));
    }
    internally_validated
        .validate(task_id)
        .map_err(CoreError::InvalidAction)?;
    Ok(graph)
}

fn validate_procedure_action_binding(
    task: &crate::domain::Task,
    procedure: &crate::agency::ProcedureIr,
) -> CoreResult<()> {
    use crate::agency::ProcedureNodeKind;

    if task.actions.is_empty() || task.actions.len() > procedure.nodes.len() {
        return Err(CoreError::VerificationFailed(
            "Procedure task graph has an invalid node count".into(),
        ));
    }
    let nodes = procedure
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    let mut action_ids = BTreeMap::new();
    for (action_id, state) in &task.actions {
        let node_id = state
            .proposal
            .metadata
            .get("procedure_node_id")
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Procedure task action is missing its node binding".into(),
                )
            })?;
        if state.proposal.metadata.get("procedure_id") != Some(&procedure.id)
            || !nodes.contains_key(node_id.as_str())
            || action_ids.insert(node_id.as_str(), *action_id).is_some()
        {
            return Err(CoreError::VerificationFailed(
                "Procedure task action has a duplicate or mismatched node binding".into(),
            ));
        }
    }
    for (node_id, action_id) in &action_ids {
        let node = nodes.get(node_id).ok_or_else(|| {
            CoreError::VerificationFailed("Procedure task action names an unknown node".into())
        })?;
        let action = &task.actions[action_id];
        let node_has_stream = procedure.streams.iter().any(|channel| {
            channel.producer_node.as_str() == *node_id || channel.consumer_node.as_str() == *node_id
        });
        match (
            node_has_stream,
            action
                .proposal
                .metadata
                .get("procedure_stream_node")
                .map(String::as_str),
        ) {
            (true, Some("true")) | (false, None) => {}
            _ => {
                return Err(CoreError::VerificationFailed(
                    "Procedure action stream binding differs from its declared channels".into(),
                ));
            }
        }
        let ProcedureNodeKind::CapabilityCall { capability_id, .. } = &node.kind else {
            return Err(CoreError::ExecutorUnavailable(
                "Only capability-call procedures can bind to the ordinary task graph".into(),
            ));
        };
        if !matches!(
            &action.proposal.action,
            Action::SetApplicationControl { capability_id: action_capability, .. }
                if action_capability == capability_id
        ) {
            return Err(CoreError::VerificationFailed(
                "Accepted procedure action differs from its capability node".into(),
            ));
        }
        let expected_dependencies = node
            .depends_on
            .iter()
            .map(|dependency| action_ids.get(dependency.as_str()).copied())
            .collect::<Option<BTreeSet<_>>>()
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "A procedure action was accepted before its dependencies had task actions"
                        .into(),
                )
            })?;
        if task.dependencies.get(action_id) != Some(&expected_dependencies) {
            return Err(CoreError::VerificationFailed(
                "Accepted task dependencies differ from the procedure".into(),
            ));
        }
    }
    Ok(())
}

fn prepare_application_controls(
    controls: Vec<ApplicationDiscoveryControl>,
) -> CoreResult<PreparedApplicationControls> {
    const MAX_CONTROLS: usize = 48;
    const ROLES: &[&str] = &[
        "button",
        "checkbox",
        "disclosure_triangle",
        "link",
        "menu_button",
        "menu_item",
        "popup_button",
        "radio_button",
        "slider",
        "switch",
        "tab",
    ];
    if controls.len() > MAX_CONTROLS {
        return Err(CoreError::VerificationFailed(
            "Native discovery exceeded its visible-control limit".into(),
        ));
    }
    let input_count = controls.len();
    let mut accepted = Vec::with_capacity(input_count);
    for control in controls {
        let Some(label) = safe_application_label(&control.label, 256) else {
            continue;
        };
        if control.id.len() != 64
            || !control
                .id
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || !ROLES.contains(&control.role.as_str())
            || control.ancestors.len() > 4
        {
            continue;
        }
        let ancestors = control
            .ancestors
            .iter()
            .map(|ancestor| safe_application_label(ancestor, 64))
            .collect::<Option<Vec<_>>>();
        // An ancestor can contain document-specific text even when the
        // control itself is stable. Keep the control but omit that path.
        let ancestors = ancestors
            .filter(|values| {
                control
                    .ancestors
                    .iter()
                    .zip(values)
                    .all(|(raw, safe)| raw.trim() == safe)
            })
            .unwrap_or_default();
        accepted.push(ValidatedApplicationControl {
            semantic: ApplicationSemanticControl {
                id: control.id,
                role: control.role,
                label,
                enabled: control.enabled,
                ancestors,
            },
            value: control.value,
            step: control.step,
            minimum: control.minimum,
            maximum: control.maximum,
        });
    }
    accepted.sort_by(|left, right| left.semantic.id.cmp(&right.semantic.id));
    let mut counts = BTreeMap::<String, usize>::new();
    for control in &accepted {
        *counts.entry(control.semantic.id.clone()).or_default() += 1;
    }
    accepted.retain(|control| counts.get(&control.semantic.id) == Some(&1));
    let mut skipped_controls = input_count.saturating_sub(accepted.len());
    let mut truncated = false;
    loop {
        let semantic = accepted
            .iter()
            .map(|control| control.semantic.clone())
            .collect::<Vec<_>>();
        let interface_fingerprint =
            format!("{:x}", sha2::Sha256::digest(serde_json::to_vec(&semantic)?));
        let facts = application_control_facts(&accepted)?;
        if serde_json::to_vec(&facts)?.len() <= 30 * 1024 || accepted.is_empty() {
            return Ok(PreparedApplicationControls {
                controls: accepted,
                semantic_controls: semantic,
                facts,
                interface_fingerprint,
                skipped_controls,
                truncated,
            });
        }
        accepted.pop();
        skipped_controls = skipped_controls.saturating_add(1);
        truncated = true;
    }
}

fn application_control_facts(
    controls: &[ValidatedApplicationControl],
) -> CoreResult<Vec<crate::world_model::ObservedFact>> {
    let mut facts = Vec::with_capacity(controls.len() * 8);
    for accepted in controls {
        let control = &accepted.semantic;
        let subject = Some(control.id.clone());
        facts.push(crate::world_model::ObservedFact {
            name: "control.enabled".into(),
            subject: subject.clone(),
            value: crate::world_model::FactValue::Boolean(control.enabled),
        });
        facts.push(crate::world_model::ObservedFact {
            name: "control.label".into(),
            subject: subject.clone(),
            value: crate::world_model::FactValue::Text(control.label.clone()),
        });
        facts.push(crate::world_model::ObservedFact {
            name: "control.role".into(),
            subject: subject.clone(),
            value: crate::world_model::FactValue::Text(control.role.clone()),
        });
        if !control.ancestors.is_empty() {
            facts.push(crate::world_model::ObservedFact {
                name: "application.control.ancestors".into(),
                subject: subject.clone(),
                value: crate::world_model::FactValue::Text(serde_json::to_string(
                    &control.ancestors,
                )?),
            });
        }
        let value = match control.role.as_str() {
            "slider" => accepted
                .value
                .as_ref()
                .and_then(serde_json::Value::as_f64)
                .filter(|value| value.is_finite())
                .map(crate::world_model::FactValue::Number),
            "switch" | "checkbox" => accepted
                .value
                .as_ref()
                .and_then(|value| {
                    value.as_bool().or_else(|| match value.as_i64() {
                        Some(0) => Some(false),
                        Some(1) => Some(true),
                        _ => None,
                    })
                })
                .map(crate::world_model::FactValue::Boolean),
            _ => None,
        };
        if let Some(value) = value {
            facts.push(crate::world_model::ObservedFact {
                name: "control.value".into(),
                subject: subject.clone(),
                value,
            });
        }
        if control.role == "slider" {
            if let Some(step) = accepted
                .step
                .filter(|value| value.is_finite() && *value > 0.0)
            {
                facts.push(crate::world_model::ObservedFact {
                    name: "control.step".into(),
                    subject: subject.clone(),
                    value: crate::world_model::FactValue::Number(step),
                });
            }
            for (name, value) in [
                ("control.minimum", accepted.minimum),
                ("control.maximum", accepted.maximum),
            ] {
                if let Some(value) = value.filter(|value| value.is_finite()) {
                    facts.push(crate::world_model::ObservedFact {
                        name: name.into(),
                        subject: subject.clone(),
                        value: crate::world_model::FactValue::Number(value),
                    });
                }
            }
        }
    }
    facts.sort_by(|left, right| (&left.name, &left.subject).cmp(&(&right.name, &right.subject)));
    Ok(facts)
}

fn safe_application_label(value: &str, maximum_bytes: usize) -> Option<String> {
    let value = value.trim();
    if value.is_empty()
        || value.len() > maximum_bytes
        || value.chars().any(char::is_control)
        || redact_for_persistence(value) != value
    {
        return None;
    }
    Some(value.to_owned())
}

fn safe_browser_label(value: &str, maximum_bytes: usize) -> Option<String> {
    let value = value.trim();
    let digit_count = value.chars().filter(char::is_ascii_digit).count();
    let lower = value.to_ascii_lowercase();
    if value.is_empty()
        || value.len() > maximum_bytes
        || value.chars().any(char::is_control)
        || value.contains('@')
        || value.contains("://")
        || lower.starts_with("www.")
        || [
            "password",
            "passcode",
            "api key",
            "apikey",
            "authorization",
            "bearer",
            "secret",
            "token",
            "credit card",
            "card number",
        ]
        .iter()
        .any(|sensitive| lower.contains(sensitive))
        || digit_count >= 7
        || redact_for_persistence(value) != value
    {
        return None;
    }
    Some(value.to_owned())
}

fn prepare_browser_discovery(
    target: crate::browser_target::BrowserTarget,
    controls: Vec<BrowserDiscoveryControl>,
    truncated: bool,
    observed_at: chrono::DateTime<Utc>,
) -> CoreResult<PreparedBrowserDiscovery> {
    target.validate()?;
    if controls.len() > 28 {
        return Err(CoreError::VerificationFailed(
            "Browser discovery exceeded its control limit".into(),
        ));
    }

    let valid_roles = [
        "button",
        "link",
        "checkbox",
        "switch",
        "slider",
        "combobox",
        "listbox",
        "textbox",
        "searchbox",
        "menuitem",
        "menuitemcheckbox",
        "menuitemradio",
        "radio",
        "tab",
    ];
    let valid_kinds = [
        "button", "link", "checkbox", "switch", "slider", "select", "textbox", "menuitem", "radio",
        "tab",
    ];
    let mut occurrence_counts = BTreeMap::<String, u32>::new();
    let mut semantic_controls = Vec::new();
    let input_control_count = controls.len();
    for control in controls {
        if control.ancestors.len() > 2 {
            return Err(CoreError::VerificationFailed(
                "Browser discovery returned too many semantic ancestors".into(),
            ));
        }
        if !valid_roles.contains(&control.role.as_str())
            || !valid_kinds.contains(&control.kind.as_str())
            || control.role.len() > 48
            || control.kind.len() > 32
        {
            continue;
        }
        let Some(label) = safe_browser_label(&control.label, 160) else {
            continue;
        };
        let ancestors = control
            .ancestors
            .iter()
            .filter_map(|ancestor| safe_browser_label(ancestor, 64))
            .collect::<Vec<_>>();
        let anchor = serde_json::to_string(&(&control.role, &control.kind, &label, &ancestors))?;
        let occurrence = occurrence_counts.entry(anchor.clone()).or_default();
        let control_id = format!(
            "{:x}",
            sha2::Sha256::digest(serde_json::to_vec(&(anchor, *occurrence))?)
        );
        *occurrence = occurrence.saturating_add(1);
        semantic_controls.push(BrowserSemanticControl {
            id: control_id,
            role: control.role,
            kind: control.kind,
            label,
            enabled: control.enabled,
            ancestors,
        });
    }
    semantic_controls.sort_by(|left, right| left.id.cmp(&right.id));
    let interface_fingerprint = format!(
        "{:x}",
        sha2::Sha256::digest(serde_json::to_vec(&semantic_controls)?)
    );
    let target_fingerprint = format!(
        "{:x}",
        sha2::Sha256::digest(serde_json::to_vec(&(&target, &interface_fingerprint))?)
    );
    let document_identity = format!(
        "{:x}",
        sha2::Sha256::digest(serde_json::to_vec(&(
            target.tab_id,
            target.window_id,
            target.frame_id,
            &target.document_id,
            target.navigation_generation,
        ))?)
    );
    let system = crate::world_model::SystemDescriptor {
        id: Uuid::nil(),
        kind: crate::world_model::SystemKind::BrowserOrigin,
        key: target.origin.clone(),
        label: target.origin.clone(),
        fingerprint: target_fingerprint.clone(),
        revision: 0,
        updated_at: observed_at,
    };
    let mut facts = vec![
        crate::world_model::ObservedFact {
            name: "browser.document_identity".into(),
            subject: None,
            value: crate::world_model::FactValue::Identifier(document_identity),
        },
        crate::world_model::ObservedFact {
            name: "browser.interface_fingerprint".into(),
            subject: None,
            value: crate::world_model::FactValue::Identifier(interface_fingerprint.clone()),
        },
        crate::world_model::ObservedFact {
            name: "browser.origin".into(),
            subject: None,
            value: crate::world_model::FactValue::Identifier(target.origin),
        },
    ];
    for control in &semantic_controls {
        let subject = Some(control.id.clone());
        facts.push(crate::world_model::ObservedFact {
            name: "browser.control.enabled".into(),
            subject: subject.clone(),
            value: crate::world_model::FactValue::Boolean(control.enabled),
        });
        facts.push(crate::world_model::ObservedFact {
            name: "browser.control.kind".into(),
            subject: subject.clone(),
            value: crate::world_model::FactValue::Text(control.kind.clone()),
        });
        facts.push(crate::world_model::ObservedFact {
            name: "browser.control.label".into(),
            subject: subject.clone(),
            value: crate::world_model::FactValue::Text(control.label.clone()),
        });
        facts.push(crate::world_model::ObservedFact {
            name: "browser.control.role".into(),
            subject: subject.clone(),
            value: crate::world_model::FactValue::Text(control.role.clone()),
        });
        if !control.ancestors.is_empty() {
            facts.push(crate::world_model::ObservedFact {
                name: "browser.control.ancestors".into(),
                subject,
                value: crate::world_model::FactValue::Text(control.ancestors.join(" > ")),
            });
        }
    }
    facts.sort_by(|left, right| (&left.name, &left.subject).cmp(&(&right.name, &right.subject)));
    let observation = crate::world_model::ObservationEnvelope {
        id: Uuid::new_v4(),
        system_id: Uuid::nil(),
        session_id: None,
        worker_session: None,
        system_fingerprint: target_fingerprint,
        origin: crate::world_model::EvidenceOrigin::Browser,
        privacy: crate::contracts::Sensitivity::Private,
        observed_at,
        facts,
    };
    Ok(PreparedBrowserDiscovery {
        system,
        observation,
        interface_fingerprint,
        observed_controls: semantic_controls.len(),
        skipped_controls: input_control_count.saturating_sub(semantic_controls.len()),
        truncated,
    })
}

fn probe_observations(
    lease: &crate::world_model::ProbeLease,
    kind: crate::world_model::ProbeKind,
    readbacks: Vec<AdapterReadback>,
) -> CoreResult<(
    [crate::world_model::ObservationEnvelope; 3],
    crate::world_model::ProbeEvidence,
)> {
    if readbacks.len() != 3 {
        return Err(CoreError::VerificationFailed(
            "Probe did not return exactly three readbacks".into(),
        ));
    }
    let observations: Vec<_> = readbacks
        .into_iter()
        .map(|readback| {
            let value = match (kind, readback.value) {
                (
                    crate::world_model::ProbeKind::RestoreSliderValue,
                    serde_json::Value::Number(number),
                ) => number
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .map(crate::world_model::FactValue::Number),
                (
                    crate::world_model::ProbeKind::RestoreToggleState,
                    serde_json::Value::Bool(value),
                ) => Some(crate::world_model::FactValue::Boolean(value)),
                _ => None,
            }
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Native readback type differs from the approved probe".into(),
                )
            })?;
            let observed_at = chrono::DateTime::parse_from_rfc3339(&readback.observed_at)
                .map_err(|_| {
                    CoreError::VerificationFailed("Native readback timestamp is invalid".into())
                })?
                .to_utc();
            Ok(crate::world_model::ObservationEnvelope {
                id: Uuid::new_v4(),
                system_id: lease.system_id,
                session_id: Some(lease.session_id),
                worker_session: Some(lease.worker_session.clone()),
                system_fingerprint: lease.system_fingerprint.clone(),
                origin: crate::world_model::EvidenceOrigin::OperatingSystem,
                privacy: crate::contracts::Sensitivity::Private,
                observed_at,
                facts: vec![crate::world_model::ObservedFact {
                    name: "control.value".into(),
                    subject: Some(lease.control_id.clone()),
                    value,
                }],
            })
        })
        .collect::<CoreResult<_>>()?;
    let observations: [crate::world_model::ObservationEnvelope; 3] =
        observations.try_into().map_err(|_| {
            CoreError::VerificationFailed("Probe did not return three readbacks".into())
        })?;
    let evidence = crate::world_model::ProbeEvidence {
        before_observation: observations[0].id,
        changed_observation: observations[1].id,
        restored_observation: observations[2].id,
    };
    Ok((observations, evidence))
}

struct StreamedRun {
    task_id: Uuid,
    prefix_action: Action,
    release: tokio::sync::watch::Sender<bool>,
}

/// Ephemeral, task-owned procedure channels. These are never persisted or
/// transferred; only validated node endpoints can be taken, once, by their
/// matching procedure action.
struct ProcedureStreamRun {
    procedure_id: String,
    endpoints: crate::procedure_stream::ProcedureStreamEndpoints,
    cancellation: crate::procedure_stream::ProcedureStreamCancellation,
}

#[derive(Clone)]
struct ProcedureDispatchSubmission {
    prepared: crate::contracts::PreparedAction,
    grant: crate::capability::CapabilityGrant,
    implementation: String,
}

#[derive(Clone)]
enum ProcedureDispatchFailure {
    Cancelled,
    ApprovalRejected(String),
    PolicyDenied(String),
    PermissionRequired(String),
    CapabilityRejected(String),
    ExecutorUnavailable(String),
    VerificationFailed(String),
    Storage(String),
    SecretStore(String),
    InvalidAction(String),
    Other(String),
}

impl ProcedureDispatchFailure {
    fn from_error(error: &CoreError) -> Self {
        match error {
            CoreError::Cancelled => Self::Cancelled,
            CoreError::ApprovalRejected(message) => Self::ApprovalRejected(message.clone()),
            CoreError::PolicyDenied(message) => Self::PolicyDenied(message.clone()),
            CoreError::PermissionRequired(message) => Self::PermissionRequired(message.clone()),
            CoreError::CapabilityRejected(message) => Self::CapabilityRejected(message.clone()),
            CoreError::ExecutorUnavailable(message) => Self::ExecutorUnavailable(message.clone()),
            CoreError::VerificationFailed(message) => Self::VerificationFailed(message.clone()),
            CoreError::Storage(message) => Self::Storage(message.clone()),
            CoreError::SecretStore(message) => Self::SecretStore(message.clone()),
            CoreError::InvalidAction(message) => Self::InvalidAction(message.clone()),
            other => Self::Other(other.to_string()),
        }
    }

    fn into_error(self) -> CoreError {
        match self {
            Self::Cancelled => CoreError::Cancelled,
            Self::ApprovalRejected(message) => CoreError::ApprovalRejected(message),
            Self::PolicyDenied(message) => CoreError::PolicyDenied(message),
            Self::PermissionRequired(message) => CoreError::PermissionRequired(message),
            Self::CapabilityRejected(message) => CoreError::CapabilityRejected(message),
            Self::ExecutorUnavailable(message) => CoreError::ExecutorUnavailable(message),
            Self::VerificationFailed(message) => CoreError::VerificationFailed(message),
            Self::Storage(message) => CoreError::Storage(message),
            Self::SecretStore(message) => CoreError::SecretStore(message),
            Self::InvalidAction(message) => CoreError::InvalidAction(message),
            Self::Other(message) => CoreError::ExecutionFailed(message),
        }
    }
}

#[derive(Clone)]
struct ProcedureDispatchOutcome {
    failure: Option<ProcedureDispatchFailure>,
}

impl ProcedureDispatchOutcome {
    fn from_result(result: &CoreResult<()>) -> Self {
        Self {
            failure: result
                .as_ref()
                .err()
                .map(ProcedureDispatchFailure::from_error),
        }
    }

    fn result(self) -> CoreResult<()> {
        self.failure
            .map_or(Ok(()), |failure| Err(failure.into_error()))
    }
}

enum ProcedureDispatchWaveState {
    Collecting(BTreeMap<Uuid, ProcedureDispatchSubmission>),
    Committing,
    Finished,
}

struct ProcedureDispatchWave {
    task_id: Uuid,
    members: BTreeSet<Uuid>,
    state: std::sync::Mutex<ProcedureDispatchWaveState>,
    completed: tokio::sync::watch::Sender<Option<ProcedureDispatchOutcome>>,
}

impl ProcedureDispatchWave {
    fn new(task_id: Uuid, members: BTreeSet<Uuid>) -> Self {
        let (completed, _) = tokio::sync::watch::channel(None);
        Self {
            task_id,
            members,
            state: std::sync::Mutex::new(ProcedureDispatchWaveState::Collecting(BTreeMap::new())),
            completed,
        }
    }

    fn abort(&self, error: &CoreError) -> (bool, Vec<Uuid>) {
        let mut state = self.state.lock().expect("procedure dispatch wave poisoned");
        match &*state {
            ProcedureDispatchWaveState::Collecting(submissions) => {
                let submitted = submissions.keys().copied().collect();
                let outcome = ProcedureDispatchOutcome {
                    failure: Some(ProcedureDispatchFailure::from_error(error)),
                };
                *state = ProcedureDispatchWaveState::Finished;
                self.completed.send_replace(Some(outcome));
                (true, submitted)
            }
            ProcedureDispatchWaveState::Committing | ProcedureDispatchWaveState::Finished => {
                (false, Vec::new())
            }
        }
    }

    async fn submit(
        self: &Arc<Self>,
        core: &SageCore,
        action_id: Uuid,
        submission: ProcedureDispatchSubmission,
    ) -> CoreResult<()> {
        if submission.prepared.intent.proposal.task_id != self.task_id
            || submission.prepared.intent.proposal.id != action_id
            || !self.members.contains(&action_id)
        {
            core.capabilities.revoke_grant(submission.grant.id).await;
            return Err(CoreError::VerificationFailed(
                "Procedure dispatch submission does not belong to its registered cohort".into(),
            ));
        }

        let mut duplicate = false;
        let mut duplicate_invalidates_wave = false;
        let mut ready = None;
        let mut revoke = Vec::new();
        {
            let mut state = self.state.lock().expect("procedure dispatch wave poisoned");
            match &mut *state {
                ProcedureDispatchWaveState::Collecting(submissions) => {
                    if let std::collections::btree_map::Entry::Vacant(entry) =
                        submissions.entry(action_id)
                    {
                        entry.insert(submission.clone());
                        if submissions.len() == self.members.len() {
                            ready = Some(submissions.clone());
                            *state = ProcedureDispatchWaveState::Committing;
                        }
                    } else {
                        duplicate = true;
                        duplicate_invalidates_wave = true;
                        revoke.extend(submissions.keys().copied());
                    }
                }
                ProcedureDispatchWaveState::Committing | ProcedureDispatchWaveState::Finished => {
                    duplicate = true;
                }
            }
            if duplicate && duplicate_invalidates_wave {
                let outcome = ProcedureDispatchOutcome {
                    failure: Some(ProcedureDispatchFailure::VerificationFailed(
                        "Procedure action submitted its dispatch receipt twice".into(),
                    )),
                };
                *state = ProcedureDispatchWaveState::Finished;
                self.completed.send_replace(Some(outcome.clone()));
            }
        }

        if duplicate {
            for submitted_id in revoke {
                core.capabilities
                    .revoke_action(self.task_id, submitted_id)
                    .await;
            }
            core.capabilities.revoke_grant(submission.grant.id).await;
            return Err(CoreError::VerificationFailed(
                "Procedure action submitted its dispatch receipt twice".into(),
            ));
        }

        if let Some(submissions) = ready {
            let result = core
                .commit_procedure_dispatch_wave(self.task_id, &submissions)
                .await;
            let outcome = ProcedureDispatchOutcome::from_result(&result);
            if outcome.failure.is_some() {
                for (submitted_id, _) in submissions {
                    core.capabilities
                        .revoke_action(self.task_id, submitted_id)
                        .await;
                }
            }
            {
                let mut state = self.state.lock().expect("procedure dispatch wave poisoned");
                *state = ProcedureDispatchWaveState::Finished;
                self.completed.send_replace(Some(outcome.clone()));
            }
            result?;
        }

        let mut completed = self.completed.subscribe();
        loop {
            let outcome = { completed.borrow_and_update().clone() };
            if let Some(outcome) = outcome {
                if outcome.failure.is_some() {
                    core.capabilities
                        .revoke_action(self.task_id, action_id)
                        .await;
                }
                return outcome.result();
            }
            completed.changed().await.map_err(|_| {
                CoreError::VerificationFailed(
                    "Procedure dispatch wave closed without a durable result".into(),
                )
            })?;
        }
    }

    fn completion_receiver(
        &self,
    ) -> tokio::sync::watch::Receiver<Option<ProcedureDispatchOutcome>> {
        self.completed.subscribe()
    }
}

type ProcedureDispatchWaveRegistry =
    Arc<std::sync::Mutex<HashMap<Uuid, Arc<ProcedureDispatchWave>>>>;

struct ProcedureDispatchWaveLease {
    registry: ProcedureDispatchWaveRegistry,
    wave: Arc<ProcedureDispatchWave>,
}

impl Drop for ProcedureDispatchWaveLease {
    fn drop(&mut self) {
        self.wave.abort(&CoreError::Cancelled);
        let mut registry = self
            .registry
            .lock()
            .expect("procedure dispatch wave registry poisoned");
        registry.retain(|action_id, wave| {
            !self.wave.members.contains(action_id) || !Arc::ptr_eq(wave, &self.wave)
        });
    }
}

enum ApprovalWait {
    Resolution(Result<ApprovalResolution, oneshot::error::RecvError>),
    ProcedureWaveFailed,
}

async fn wait_for_procedure_dispatch_wave(
    receiver: &mut tokio::sync::watch::Receiver<Option<ProcedureDispatchOutcome>>,
) {
    loop {
        let completed = { receiver.borrow_and_update().is_some() };
        if completed || receiver.changed().await.is_err() {
            return;
        }
    }
}

enum RunAuthority {
    Interactive {
        command: Option<crate::commands::SubmissionKey>,
    },
    Scheduled {
        expires_at: chrono::DateTime<Utc>,
    },
    Continued {
        previous: Box<Task>,
        owner: crate::runtime::RunSignal,
    },
}

struct RunSubmission {
    request: String,
    conversation_id: Option<Uuid>,
    task_id: Option<Uuid>,
    graph: Option<crate::domain::ActionGraph>,
    procedure: Option<crate::agency::ProcedureIr>,
    resources: Vec<crate::contracts::ResourceScope>,
    authority: RunAuthority,
    skill_lineages: Vec<(String, String, Vec<String>)>,
    streamed_prefix: Option<(Uuid, Action)>,
}

pub struct SageCore {
    config: CoreConfig,
    model: Arc<dyn ModelProvider>,
    secret_store: Arc<dyn SecretStore>,
    store: LocalStore,
    events: EventHub,
    tasks: RwLock<HashMap<Uuid, Task>>,
    pending_approvals: Mutex<HashMap<Uuid, PendingApproval>>,
    pending_questions: Mutex<HashMap<Uuid, PendingQuestion>>,
    control_changed: Notify,
    runtime: crate::runtime::RunRegistry,
    policy: PolicyEngine,
    capabilities: CapabilityBroker,
    compiler: ActionCompiler,
    availability: ExecutorAvailability,
    broker: ExecutionBroker,
    resolver: ResourceResolver,
    files: Arc<crate::execution::files::FileBroker>,
    observer: Arc<dyn Observer>,
    verifier: Verifier,
    pub adapters: Arc<AdapterBridge>,
    submission_lock: Mutex<()>,
    storage_unlock: Arc<Mutex<()>>,
    storage_hydrated: std::sync::atomic::AtomicBool,
    scheduler_lane: Mutex<()>,
    effect_ownership: crate::effect_ownership::EffectArbiter,
    execution_timings: std::sync::Mutex<crate::scheduling::Timings>,
    pub(crate) application_preparations: std::sync::Mutex<crate::preparation::Applications>,
    streamed_runs: std::sync::Mutex<HashMap<Uuid, StreamedRun>>,
    procedure_streams: std::sync::Mutex<HashMap<Uuid, ProcedureStreamRun>>,
    procedure_dispatch_waves: ProcedureDispatchWaveRegistry,
}

impl std::fmt::Debug for SageCore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SageCore")
            .field("config", &self.config)
            .field("model", &self.model.descriptor().id)
            .finish_non_exhaustive()
    }
}

impl SageCore {
    pub fn new(config: CoreConfig, model: Arc<dyn ModelProvider>) -> CoreResult<Arc<Self>> {
        Self::new_with_secret_store(config, model, Arc::new(OsSecretStore))
    }

    pub fn new_with_secret_store(
        config: CoreConfig,
        model: Arc<dyn ModelProvider>,
        secret_store: Arc<dyn SecretStore>,
    ) -> CoreResult<Arc<Self>> {
        std::fs::create_dir_all(&config.data_dir)?;
        std::fs::create_dir_all(&config.recovery_dir)?;
        #[cfg(not(test))]
        let store = LocalStore::deferred(&config.database_path)?;
        #[cfg(test)]
        let store =
            LocalStore::open_encrypted(&config.database_path, &SecretBytes::new(vec![17; 32]))?;
        store.migrate_knowledge()?;
        store.migrate_workflows()?;
        store.migrate_world_model()?;
        let tasks = store
            .load_tasks(true)?
            .into_iter()
            .map(|task| (task.id, task))
            .collect();
        let runtime = crate::runtime::RunRegistry::default();
        let capabilities = CapabilityBroker::with_runtime(runtime.clone());
        let mut broker = ExecutionBroker::new(capabilities.clone());
        let adapters = Arc::new(AdapterBridge::new(store.clone()));
        let files = Arc::new(crate::execution::files::FileBroker::default());
        broker.register(Arc::new(NativeExecutor::new(
            config.recovery_dir.clone(),
            adapters.clone(),
            files.clone(),
            store.clone(),
        )));
        broker.register(Arc::new(BrowserExecutor(adapters.clone())));
        // Installing a binary cannot enable a domain. VM execution and signed
        // privileged handlers need independently qualified feature manifests.
        let mut resolver = ResourceResolver::platform_default(config.data_dir.clone())?;
        resolver.protect_paths(&config.protected_paths)?;
        let availability = ExecutorAvailability {
            browser_dom: false,
            accessibility: false,
            ..ExecutorAvailability::default()
        };

        let storage_hydrated = std::sync::atomic::AtomicBool::new(!store.is_locked());
        Ok(Arc::new(Self {
            config,
            model,
            secret_store,
            store,
            events: EventHub::default(),
            tasks: RwLock::new(tasks),
            pending_approvals: Mutex::new(HashMap::new()),
            pending_questions: Mutex::new(HashMap::new()),
            control_changed: Notify::new(),
            runtime,
            policy: PolicyEngine,
            capabilities,
            compiler: ActionCompiler,
            availability,
            broker,
            resolver,
            files,
            observer: Arc::new(PlatformObserver(adapters.clone())),
            verifier: Verifier,
            adapters,
            submission_lock: Mutex::new(()),
            storage_unlock: Arc::new(Mutex::new(())),
            storage_hydrated,
            scheduler_lane: Mutex::new(()),
            effect_ownership: crate::effect_ownership::EffectArbiter::default(),
            execution_timings: Default::default(),
            application_preparations: Default::default(),
            streamed_runs: Default::default(),
            procedure_streams: Default::default(),
            procedure_dispatch_waves: Arc::new(Default::default()),
        }))
    }

    pub async fn unlock_storage(self: &Arc<Self>) -> CoreResult<()> {
        let guard = self.storage_unlock.clone().lock_owned().await;
        if self
            .storage_hydrated
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Ok(());
        }
        let core = self.clone();
        // The accepted unlock owns the lane through cache hydration even if
        // its UI disconnects. At most one credential job exists; no blocking
        // keychain call occupies an async executor or the intent/control lane.
        tokio::spawn(async move {
            let _guard = guard;
            let store = core.store.clone();
            let secrets = core.secret_store.clone();
            let tasks = tokio::task::spawn_blocking(move || {
                store.unlock(secrets.as_ref())?;
                store.load_tasks(true)
            })
            .await
            .map_err(|_| CoreError::SecretStore("Protected storage worker exited".into()))??;
            *core.tasks.write().await = tasks.into_iter().map(|task| (task.id, task)).collect();
            core.storage_hydrated
                .store(true, std::sync::atomic::Ordering::Release);
            Ok(())
        })
        .await
        .map_err(|_| CoreError::SecretStore("Protected storage hydration exited".into()))?
    }

    pub fn events(&self) -> &EventHub {
        &self.events
    }

    /// Build an advisory skill-branch proposal only for a typed, explicit
    /// current-reference request. This path uses no planner and stores no
    /// captured reference; choosing a proposal only fills the composer.
    pub(crate) async fn contextual_routine_suggestion(
        &self,
        request: &str,
    ) -> CoreResult<Option<crate::learning::ContextualRoutineSuggestion>> {
        if !crate::context::should_capture_live_reference(request, false, false) {
            return Ok(None);
        }
        let store = self.store.clone();
        let families = tokio::task::spawn_blocking(move || store.routine_families())
            .await
            .map_err(|_| CoreError::Storage("Routine suggestion lookup worker exited".into()))??;
        if families.is_empty() {
            return Ok(None);
        }
        let kind = crate::context::live_reference_kind(request);
        let include_page_text = crate::context::requests_page_text(request);
        let observed = self.adapters.reference(include_page_text).await;
        let reference = crate::context::bind_live_reference_kind(
            crate::context::select_live_reference(observed),
            kind,
        );
        Ok(crate::learning::contextual_branch_suggestion(
            request, &reference, &families,
        ))
    }

    pub(crate) async fn worker_received_cancellation(
        &self,
        request_id: String,
        binding: crate::execution::bridge::EffectBinding,
    ) -> CoreResult<()> {
        self.project_worker_receipt(
            binding.task_id,
            &crate::receipts::ReceiptKey {
                request_id,
                kind: crate::receipts::ReceiptKind::CancelAcknowledged,
            },
        )
        .await
    }

    pub(crate) async fn record_late_adapter_result(
        &self,
        late: crate::execution::bridge::LateAdapterResult,
    ) -> CoreResult<()> {
        let Some(binding) = late.binding else {
            self.settle_late_probe_result(&late)?;
            return Ok(());
        };
        self.project_worker_receipt(
            binding.task_id,
            &crate::receipts::ReceiptKey {
                request_id: late.response.request_id,
                kind: if late.never_sent {
                    crate::receipts::ReceiptKind::NeverSent
                } else {
                    crate::receipts::ReceiptKind::Response
                },
            },
        )
        .await
    }

    fn settle_late_probe_result(
        &self,
        late: &crate::execution::bridge::LateAdapterResult,
    ) -> CoreResult<()> {
        if late.never_sent || !late.response.success {
            return Ok(());
        }
        let Ok(lease_id) = Uuid::parse_str(&late.response.request_id) else {
            return Ok(());
        };
        let Some(lease) = self.store.dispatched_probe_lease(lease_id)? else {
            return Ok(());
        };
        if late.session != lease.worker_session {
            return Err(CoreError::VerificationFailed(
                "Late probe receipt came from a different authenticated worker".into(),
            ));
        }
        let target = crate::application_target::ApplicationTarget::from_wire(
            late.response.application_target.as_ref().ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Late probe receipt omitted its signed application identity".into(),
                )
            })?,
        )?;
        if target.code_digest != lease.system_fingerprint
            || late.response.observed_process_id != lease.expected_process_id
        {
            return Err(CoreError::VerificationFailed(
                "Late probe receipt belongs to a different application identity".into(),
            ));
        }
        let mut value: serde_json::Value = serde_json::from_str(&late.response.json)?;
        let object = value.as_object_mut().ok_or_else(|| {
            CoreError::Protocol("Late native probe receipt has an invalid result schema".into())
        })?;
        object.insert("application_target".into(), serde_json::to_value(target)?);
        object.insert(
            "observed_process_id".into(),
            json!(late.response.observed_process_id),
        );
        let response: AdapterProbeResponse = serde_json::from_value(value)?;
        if response.application_target.code_digest != lease.system_fingerprint
            || response.observed_process_id != late.response.observed_process_id
            || response.control_id != lease.control_id
            || !response.restoration_verified
        {
            return Err(CoreError::VerificationFailed(
                "Late probe receipt did not verify the exact approved control and restoration"
                    .into(),
            ));
        }
        let session = self
            .store
            .learning_session(lease.session_id)?
            .ok_or_else(|| CoreError::PermissionRequired("Probe session was removed".into()))?;
        if session.worker_session != lease.worker_session
            || session.permitted_probes.get(&lease.control_id) != Some(&lease.kind)
        {
            return Err(CoreError::VerificationFailed(
                "Late probe receipt no longer matches its exact approval".into(),
            ));
        }
        let (observations, evidence) =
            probe_observations(&lease, lease.kind, response.observations)?;
        self.store
            .record_probe_observations(lease.id, &observations, Utc::now())?;
        self.store.complete_probe(lease.id, &evidence, Utc::now())
    }

    async fn project_worker_receipt(
        &self,
        task_id: Uuid,
        key: &crate::receipts::ReceiptKey,
    ) -> CoreResult<()> {
        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get(&task_id)
            .cloned()
            .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))?;
        if let Some((task, event)) = self.store.commit_worker_receipt_projection(task, key)? {
            tasks.insert(task_id, task);
            self.events.publish(event);
        }
        Ok(())
    }

    async fn reconcile_task_worker_receipts(&self, task_id: Uuid) -> CoreResult<()> {
        loop {
            let pending = self.store.pending_worker_receipts(Some(task_id))?;
            if pending.is_empty() {
                return Ok(());
            }
            for (_, key) in pending {
                self.project_worker_receipt(task_id, &key).await?;
            }
        }
    }

    pub async fn snapshot(&self, include_completed: bool) -> StateSnapshot {
        let mut tasks: Vec<_> = self
            .tasks
            .read()
            .await
            .values()
            .filter(|task| include_completed || !task.status.is_terminal())
            .cloned()
            .collect();
        for task in &mut tasks {
            self.runtime.project_pending_stop(task);
        }
        tasks.sort_by_key(|task| std::cmp::Reverse(task.updated_at));
        StateSnapshot {
            tasks,
            core_version: env!("CARGO_PKG_VERSION").into(),
            storage_locked: self.store.is_locked(),
            pending_approvals: self
                .pending_approvals
                .lock()
                .await
                .values()
                .filter(|p| p.expires_at > Utc::now() && !self.runtime.is_stopped(p.task_id))
                .map(|p| p.record.clone())
                .collect(),
            pending_questions: self
                .pending_questions
                .lock()
                .await
                .values()
                .filter(|p| p.record.expires_at > Utc::now() && !self.runtime.is_stopped(p.task_id))
                .map(|p| p.record.clone())
                .collect(),
            protocol_version: sage_protocol::PROTOCOL_VERSION,
            knowledge: self.knowledge_snapshot(None).ok(),
        }
    }

    pub async fn submit_task(self: &Arc<Self>, request: impl Into<String>) -> CoreResult<Uuid> {
        self.submit_in_conversation(request.into(), None, false, None)
            .await
    }

    async fn learned_application_control_graph(
        &self,
        request: &str,
        task_id: Uuid,
    ) -> CoreResult<Option<ActionGraph>> {
        if crate::intent::parse_control_assignment(request).is_none() {
            return Ok(None);
        }
        if !self
            .adapters
            .supports_feature("native", "application_control_v1")
            .await
        {
            return Ok(None);
        }

        let store = self.store.clone();
        let discovered = tokio::task::spawn_blocking(move || {
            let mut discovered = Vec::new();
            for system in store.observed_systems()? {
                if system.kind != crate::world_model::SystemKind::Application {
                    continue;
                }
                for assessment in store.capability_assessments(system.id)? {
                    discovered.push((system.clone(), assessment));
                }
            }
            Ok::<_, CoreError>(discovered)
        })
        .await
        .map_err(|_| CoreError::Storage("Learned-control intent lookup worker exited".into()))??;
        build_learned_application_control_graph(request, task_id, discovered)
    }

    pub async fn submit_in_conversation(
        self: &Arc<Self>,
        request: String,
        conversation_id: Option<Uuid>,
        background: bool,
        graph: Option<crate::domain::ActionGraph>,
    ) -> CoreResult<Uuid> {
        self.submit_scoped(request, conversation_id, background, graph, Vec::new())
            .await
    }

    pub async fn submit_scoped(
        self: &Arc<Self>,
        request: String,
        conversation_id: Option<Uuid>,
        background: bool,
        graph: Option<crate::domain::ActionGraph>,
        resources: Vec<crate::contracts::ResourceScope>,
    ) -> CoreResult<Uuid> {
        if background {
            return Err(CoreError::PermissionRequired(
                "Background work must originate from an exact durable schedule authorization"
                    .into(),
            ));
        }
        self.submit_run(RunSubmission {
            request,
            conversation_id,
            task_id: None,
            graph,
            procedure: None,
            resources,
            authority: RunAuthority::Interactive { command: None },
            skill_lineages: Vec::new(),
            streamed_prefix: None,
        })
        .await
    }

    pub(crate) async fn submit_receipted(
        self: &Arc<Self>,
        request: String,
        conversation_id: Option<Uuid>,
        resources: Vec<crate::contracts::ResourceScope>,
        command: crate::commands::SubmissionKey,
    ) -> CoreResult<Uuid> {
        self.submit_run(RunSubmission {
            request,
            conversation_id,
            task_id: None,
            graph: None,
            procedure: None,
            resources,
            authority: RunAuthority::Interactive {
                command: Some(command),
            },
            skill_lineages: Vec::new(),
            streamed_prefix: None,
        })
        .await
    }

    pub(crate) async fn submit_streamed_prefix_receipted(
        self: &Arc<Self>,
        request: String,
        conversation_id: Option<Uuid>,
        resources: Vec<crate::contracts::ResourceScope>,
        command: crate::commands::SubmissionKey,
        stream_id: Uuid,
    ) -> CoreResult<Uuid> {
        let intent = crate::intent::compile(&request, &resources).ok_or_else(|| {
            CoreError::InvalidAction("A streamed prefix must be one complete local action".into())
        })?;
        if intent.steps.len() != 1 {
            return Err(CoreError::PermissionRequired(
                "Only one locally understood step can start during speech".into(),
            ));
        }
        let action = intent.steps[0].action.clone();
        match &action {
            Action::OpenApplication { .. } if !cfg!(target_os = "macos") => {
                return Err(CoreError::ExecutorUnavailable(
                    "Starting app steps during speech is currently available on macOS only".into(),
                ));
            }
            Action::ReadFile { path, .. } => {
                self.prepare_streamed_file_read(path, &resources).await?;
            }
            Action::OpenApplication { .. } => {}
            _ => {
                return Err(CoreError::PermissionRequired(
                    "Only app opening and scoped file reads can start during speech".into(),
                ));
            }
        }
        self.submit_run(RunSubmission {
            request,
            conversation_id,
            task_id: None,
            graph: None,
            procedure: None,
            resources,
            authority: RunAuthority::Interactive {
                command: Some(command),
            },
            skill_lineages: Vec::new(),
            streamed_prefix: Some((stream_id, action)),
        })
        .await
    }

    pub(crate) async fn prepare_streamed_file_read(
        &self,
        path: &std::path::Path,
        scopes: &[crate::contracts::ResourceScope],
    ) -> CoreResult<()> {
        let resolver = self.resolver.clone();
        let path = path.to_path_buf();
        let scopes = scopes.to_vec();
        crate::execution::io::bounded_read(move |_| resolver.prepare_scoped_read(&path, &scopes))
            .await
    }

    pub(crate) async fn finalize_streamed_prefix_receipted(
        &self,
        stream_id: Uuid,
        request: String,
        mut resources: Vec<crate::contracts::ResourceScope>,
        command: crate::commands::SubmissionKey,
    ) -> CoreResult<Uuid> {
        let task_id = self
            .streamed_runs
            .lock()
            .expect("streamed runs poisoned")
            .get(&stream_id)
            .map(|run| run.task_id)
            .ok_or_else(|| {
                CoreError::InvalidAction("The prepared voice run is no longer active".into())
            })?;
        let Some(intent) = crate::intent::compile(&request, &resources) else {
            self.signal_stream_control(stream_id, TaskStatus::Paused);
            return Err(CoreError::InvalidAction(
                "The completed voice request is outside the local streaming grammar. The prepared app step may already have run and is held for review. Stop this run before sending a changed request.".into(),
            ));
        };
        for scope in &mut resources {
            let resolver = self.resolver.clone();
            let root = scope.root.clone();
            scope.root = match crate::execution::io::bounded_read(move |_| {
                resolver.validate_scope_root(&root)
            })
            .await
            {
                Ok(root) => root,
                Err(error) => {
                    self.signal_stream_control(stream_id, TaskStatus::Paused);
                    return Err(error);
                }
            };
            if scope.effects.is_empty()
                || scope.effects.iter().any(|effect| {
                    !matches!(
                        effect,
                        crate::contracts::Effect::Read | crate::contracts::Effect::Create
                    )
                })
            {
                self.signal_stream_control(stream_id, TaskStatus::Paused);
                return Err(CoreError::PermissionRequired(
                    "Unsupported streamed voice scope".into(),
                ));
            }
        }
        if !self
            .try_revise_intent(task_id, &request, &resources, &command)
            .await?
        {
            self.signal_stream_control(stream_id, TaskStatus::Paused);
            return Err(CoreError::PermissionRequired(
                "The prepared step could not be safely reconciled with the completed voice request. It remains held for review.".into(),
            ));
        }
        let still_same_task = self
            .streamed_runs
            .lock()
            .expect("streamed runs poisoned")
            .get(&stream_id)
            .is_some_and(|run| run.task_id == task_id);
        if !still_same_task || intent.steps.is_empty() {
            return Err(CoreError::Cancelled);
        }
        self.release_stream(stream_id, task_id);
        Ok(task_id)
    }

    pub(crate) fn stream_prefix_is_current(&self, stream_id: Uuid, text: &str) -> bool {
        let expected = self
            .streamed_runs
            .lock()
            .expect("streamed runs poisoned")
            .get(&stream_id)
            .map(|run| run.prefix_action.clone());
        let Some(expected) = expected else {
            return true;
        };
        crate::intent::preparation(text, &[])
            .and_then(|intent| intent.steps.first().map(|step| step.action.clone()))
            .is_some_and(|action| action == expected)
    }

    pub(crate) fn has_streamed_run(&self, stream_id: Uuid) -> bool {
        self.streamed_runs
            .lock()
            .expect("streamed runs poisoned")
            .contains_key(&stream_id)
    }

    pub(crate) fn hold_stream_for_revision(&self, stream_id: Uuid) -> bool {
        let runs = self.streamed_runs.lock().expect("streamed runs poisoned");
        let Some(run) = runs.get(&stream_id) else {
            return false;
        };
        self.signal_hold(run.task_id)
    }

    pub(crate) fn signal_stream_control(&self, stream_id: Uuid, status: TaskStatus) -> bool {
        let runs = self.streamed_runs.lock().expect("streamed runs poisoned");
        let Some(run) = runs.get(&stream_id) else {
            return false;
        };
        let task_id = run.task_id;
        match status {
            TaskStatus::Cancelled => {
                self.signal_stop(task_id);
            }
            TaskStatus::Paused => {
                self.signal_hold(task_id);
            }
            _ => return false,
        }
        run.release.send_replace(true);
        true
    }

    pub(crate) async fn control_streamed_task(
        self: &Arc<Self>,
        stream_id: Uuid,
        status: TaskStatus,
    ) -> CoreResult<()> {
        let task_id = self
            .streamed_runs
            .lock()
            .expect("streamed runs poisoned")
            .get(&stream_id)
            .map(|run| run.task_id);
        let Some(task_id) = task_id else {
            // Admission-time Stop can finish cleanup before this queued save
            // reaches its lane; the terminal state has already been persisted.
            return Ok(());
        };
        self.control_task(task_id, status).await
    }

    fn release_stream(&self, stream_id: Uuid, task_id: Uuid) {
        let mut runs = self.streamed_runs.lock().expect("streamed runs poisoned");
        if runs
            .get(&stream_id)
            .is_some_and(|run| run.task_id == task_id)
            && let Some(run) = runs.remove(&stream_id)
        {
            run.release.send_replace(true);
        }
    }

    fn release_streamed_task(&self, task_id: Uuid) {
        let stream_id = self
            .streamed_runs
            .lock()
            .expect("streamed runs poisoned")
            .iter()
            .find_map(|(stream_id, run)| (run.task_id == task_id).then_some(*stream_id));
        if let Some(stream_id) = stream_id {
            self.release_stream(stream_id, task_id);
        }
    }

    pub(crate) async fn supersede_receipted(
        self: &Arc<Self>,
        previous_id: Uuid,
        request: String,
        conversation_id: Option<Uuid>,
        mut resources: Vec<crate::contracts::ResourceScope>,
        command: crate::commands::SubmissionKey,
    ) -> CoreResult<Uuid> {
        // A retry of an accepted correction must not stop the replacement or
        // repeat its effects. The command digest includes the superseded ID.
        if let Some(id) = self.store.accepted_submission(&command)? {
            return Ok(id);
        }
        let previous = self.get_task(previous_id).await?;
        if conversation_id.is_none() || previous.conversation_id != conversation_id {
            return Err(CoreError::InvalidAction(
                "A correction must belong to the same conversation".into(),
            ));
        }
        if request.trim().is_empty() || request.len() > 64 * 1024 || resources.len() > 32 {
            return Err(CoreError::InvalidAction(
                "Invalid replacement request".into(),
            ));
        }
        if resources.is_empty() {
            resources = previous
                .contract
                .as_ref()
                .map(|contract| contract.resources.clone())
                .unwrap_or_default();
        }
        self.signal_hold(previous_id);
        // Freeze further dispatch before potentially slow filesystem checks.
        // An accepted retry returned above without changing the current hold.
        for scope in &mut resources {
            let resolver = self.resolver.clone();
            let root = scope.root.clone();
            scope.root =
                crate::execution::io::bounded_read(move |_| resolver.validate_scope_root(&root))
                    .await?;
            if scope.effects.is_empty()
                || scope.effects.iter().any(|effect| {
                    !matches!(
                        effect,
                        crate::contracts::Effect::Read | crate::contracts::Effect::Create
                    )
                })
            {
                return Err(CoreError::PermissionRequired(
                    "Unsupported correction scope".into(),
                ));
            }
        }
        if self
            .try_revise_intent(previous_id, &request, &resources, &command)
            .await?
        {
            self.release_streamed_task(previous_id);
            return Ok(previous_id);
        }
        self.stop_task(previous_id).await?;
        // This awaits durable Stop admission, not completion of OS effects.
        // Independent new work can start while old adapter cancellation settles.
        self.submit_receipted(request, conversation_id, resources, command)
            .await
    }

    async fn try_revise_intent(
        &self,
        task_id: Uuid,
        request: &str,
        resources: &[crate::contracts::ResourceScope],
        key: &crate::commands::SubmissionKey,
    ) -> CoreResult<bool> {
        let Some(intent) = crate::intent::compile(request, resources) else {
            return Ok(false);
        };
        let _submission = self.submission_lock.lock().await;
        if let Some(id) = self.store.accepted_submission(key)? {
            return Ok(id == task_id);
        }
        let mut tasks = self.tasks.write().await;
        let current = tasks.get(&task_id).ok_or(CoreError::Cancelled)?;
        if current.intent.is_none()
            || !current.status.is_active()
            || !self.runtime.is_active(task_id)
            || current
                .contract
                .as_ref()
                .is_none_or(|contract| contract.resources != resources)
        {
            return Ok(false);
        }
        self.signal_hold(task_id);
        if self.runtime.is_stopped(task_id) {
            return Err(CoreError::Cancelled);
        }
        let mut revision = crate::reconciliation::prepare(
            current,
            &intent,
            redact_for_persistence(request.trim()),
        )?;
        let events = self.store.accept_revision(current, &mut revision, key)?;
        // No await between durable acceptance and synchronous revocation: an
        // already issued grant sees retirement before the run can be resumed.
        self.runtime.retire_actions(task_id, &revision.retired);
        let retired = revision.retired;
        tasks.insert(task_id, revision.task);
        drop(tasks);
        for event in events {
            self.events.publish(event);
        }
        self.pending_approvals
            .lock()
            .await
            .retain(|_, p| p.task_id != task_id || !retired.contains(&p.action_id));
        self.pending_questions
            .lock()
            .await
            .retain(|_, p| p.task_id != task_id || !retired.contains(&p.action_id));
        self.runtime.hold(task_id, false);
        self.control_changed.notify_waiters();
        Ok(true)
    }

    pub(crate) async fn intent_available(&self, intent: &crate::intent::CompiledIntent) -> bool {
        let available = self.available_tools().await;
        intent
            .steps
            .iter()
            .all(|step| available.iter().any(|tool| tool.name == step.action.kind()))
    }

    fn register_procedure_dispatch_wave(
        &self,
        task_id: Uuid,
        action_ids: &[Uuid],
    ) -> CoreResult<ProcedureDispatchWaveLease> {
        let members = action_ids.iter().copied().collect::<BTreeSet<_>>();
        if !(2..=16).contains(&members.len()) || members.len() != action_ids.len() {
            return Err(CoreError::InvalidAction(
                "Procedure stream cohort must contain two to sixteen unique actions".into(),
            ));
        }
        let wave = Arc::new(ProcedureDispatchWave::new(task_id, members.clone()));
        let mut registry = self
            .procedure_dispatch_waves
            .lock()
            .expect("procedure dispatch wave registry poisoned");
        if members
            .iter()
            .any(|action_id| registry.contains_key(action_id))
        {
            return Err(CoreError::VerificationFailed(
                "Procedure action is already registered in a dispatch cohort".into(),
            ));
        }
        for action_id in members {
            registry.insert(action_id, wave.clone());
        }
        Ok(ProcedureDispatchWaveLease {
            registry: self.procedure_dispatch_waves.clone(),
            wave,
        })
    }

    fn procedure_dispatch_wave_for_action(
        &self,
        action_id: Uuid,
    ) -> Option<Arc<ProcedureDispatchWave>> {
        self.procedure_dispatch_waves
            .lock()
            .expect("procedure dispatch wave registry poisoned")
            .get(&action_id)
            .cloned()
    }

    async fn abort_procedure_dispatch_wave(&self, action_id: Uuid, error: &CoreError) {
        let Some(wave) = self.procedure_dispatch_wave_for_action(action_id) else {
            return;
        };
        let (aborted, submitted) = wave.abort(error);
        if aborted {
            for submitted_id in submitted {
                self.capabilities
                    .revoke_action(wave.task_id, submitted_id)
                    .await;
            }
        }
    }

    fn cancel_procedure_stream_run(&self, task_id: Uuid) {
        if let Some(streams) = self
            .procedure_streams
            .lock()
            .expect("procedure streams poisoned")
            .get(&task_id)
        {
            streams.cancellation.cancel();
        }
    }

    fn create_procedure_stream_run(
        &self,
        checkpoint: &crate::agency::ProcedureCheckpoint,
    ) -> CoreResult<Option<ProcedureStreamRun>> {
        if checkpoint.procedure.streams.is_empty() {
            return Ok(None);
        }
        let (_, assessments, _) = self.current_procedure_context(checkpoint)?;
        let capabilities = assessments
            .into_iter()
            .map(|assessment| assessment.descriptor)
            .collect::<Vec<_>>();
        let mut pool = crate::procedure_stream::ProcedureStreamPool::new(
            &checkpoint.procedure,
            &capabilities,
            checkpoint.task_id,
        )?;
        let (cancellation, receiver) = crate::procedure_stream::ProcedureStreamCancellation::new();
        let endpoints = pool.open_all(receiver)?;
        Ok(Some(ProcedureStreamRun {
            procedure_id: checkpoint.procedure.id.clone(),
            endpoints,
            cancellation,
        }))
    }

    fn take_procedure_node_streams(
        &self,
        task_id: Uuid,
        proposal: &ActionProposal,
    ) -> CoreResult<Option<crate::procedure_stream::ProcedureNodeStreams>> {
        const STREAM_NODE_METADATA: &str = "procedure_stream_node";
        let Some(procedure_id) = proposal.metadata.get("procedure_id") else {
            if proposal.metadata.contains_key(STREAM_NODE_METADATA) {
                return Err(CoreError::VerificationFailed(
                    "Stream action is missing its procedure identity".into(),
                ));
            }
            return Ok(None);
        };
        let Some(node_id) = proposal.metadata.get("procedure_node_id") else {
            if proposal.metadata.contains_key(STREAM_NODE_METADATA) {
                return Err(CoreError::VerificationFailed(
                    "Stream action is missing its node identity".into(),
                ));
            }
            return Ok(None);
        };
        match proposal
            .metadata
            .get(STREAM_NODE_METADATA)
            .map(String::as_str)
        {
            None => Ok(None),
            Some("true") => {
                let mut streams = self
                    .procedure_streams
                    .lock()
                    .expect("procedure streams poisoned");
                let run = streams.get_mut(&task_id).ok_or_else(|| {
                    CoreError::PermissionRequired(
                        "Procedure stream endpoints are unavailable for this execution attempt"
                            .into(),
                    )
                })?;
                if run.procedure_id != *procedure_id {
                    return Err(CoreError::VerificationFailed(
                        "Procedure stream belongs to another procedure".into(),
                    ));
                }
                run.endpoints.take_node(node_id).map(Some).ok_or_else(|| {
                    CoreError::VerificationFailed(
                        "Procedure node stream endpoints are missing or already consumed".into(),
                    )
                })
            }
            Some(_) => Err(CoreError::VerificationFailed(
                "Procedure stream marker is malformed".into(),
            )),
        }
    }

    fn restore_procedure_stream_run(&self, task_id: Uuid) -> CoreResult<()> {
        let Some(checkpoint) = self.store.load_procedure_checkpoint(task_id)? else {
            return Ok(());
        };
        if checkpoint.procedure.streams.is_empty() {
            return Ok(());
        }
        let stream_nodes = checkpoint
            .procedure
            .streams
            .iter()
            .flat_map(|channel| {
                [
                    channel.producer_node.as_str(),
                    channel.consumer_node.as_str(),
                ]
            })
            .collect::<BTreeSet<_>>();
        if stream_nodes.iter().any(|node_id| {
            matches!(
                checkpoint.runtime.node_states().get(*node_id),
                Some(
                    crate::agency::ProcedureNodeState::Running
                        | crate::agency::ProcedureNodeState::Uncertain
                )
            )
        }) {
            return Err(CoreError::PermissionRequired(
                "An interrupted procedure stream has unsettled endpoints and cannot be replayed"
                    .into(),
            ));
        }
        let run = self
            .create_procedure_stream_run(&checkpoint)?
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Procedure stream restoration produced no endpoints".into(),
                )
            })?;
        let mut streams = self
            .procedure_streams
            .lock()
            .expect("procedure streams poisoned");
        if streams.insert(task_id, run).is_some() {
            return Err(CoreError::VerificationFailed(
                "Procedure stream runtime already exists for this task".into(),
            ));
        }
        Ok(())
    }

    async fn submit_run(self: &Arc<Self>, submission: RunSubmission) -> CoreResult<Uuid> {
        let RunSubmission {
            request,
            conversation_id,
            task_id: requested_task_id,
            graph,
            procedure,
            mut resources,
            authority,
            skill_lineages,
            streamed_prefix,
        } = submission;
        let (background_expiry, continuation, command) = match authority {
            RunAuthority::Interactive { command } => (None, None, command),
            RunAuthority::Scheduled { expires_at } => (Some(expires_at), None, None),
            RunAuthority::Continued { previous, owner } => (None, Some((*previous, owner)), None),
        };
        self.unlock_storage().await?;
        let _submission = self.submission_lock.lock().await;
        // A receipt lookup precedes scope and conversation checks: the accepted
        // run remains the same even after its resources or status have changed.
        if let Some(ref command) = command
            && let Some(task_id) = self.store.accepted_submission(command)?
        {
            return Ok(task_id);
        }
        if let Some((stream_id, _)) = &streamed_prefix {
            let runs = self.streamed_runs.lock().expect("streamed runs poisoned");
            if runs.contains_key(stream_id) {
                return Err(CoreError::Protocol(
                    "This voice stream already has a prepared run".into(),
                ));
            }
            if runs.len() >= 16 {
                return Err(CoreError::Busy(
                    "The active voice-stream capacity has been reached".into(),
                ));
            }
        }
        for (family_id, family_digest, sources) in &skill_lineages {
            if !self
                .store
                .routine_family_matches(family_id, family_digest, sources)?
            {
                return Err(CoreError::ApprovalRejected(
                    "The source routine branches changed or were forgotten. Review a fresh shared-prefix draft."
                        .into(),
                ));
            }
        }
        if resources.len() > 32 {
            return Err(CoreError::InvalidAction("Too many task folders".into()));
        }
        for resource in &mut resources {
            resource.root = self.resolver.validate_scope_root(&resource.root)?;
            if resource.effects.is_empty()
                || resource.effects.iter().any(|e| {
                    !matches!(
                        e,
                        crate::contracts::Effect::Read | crate::contracts::Effect::Create
                    )
                })
            {
                return Err(CoreError::PermissionRequired(
                    "Only reading and creation can be scoped to a folder".into(),
                ));
            }
        }
        let request = redact_for_persistence(request.trim());
        if request.is_empty() {
            return Err(CoreError::InvalidAction(
                "task request must not be empty".into(),
            ));
        }
        if request.len() > 64 * 1024 {
            return Err(CoreError::InvalidAction(
                "task request exceeds the 64-kilobyte limit".into(),
            ));
        }
        if let Some(id) = conversation_id
            && self
                .tasks
                .read()
                .await
                .values()
                .any(|t| t.conversation_id == Some(id) && t.status.is_active())
        {
            return Err(CoreError::InvalidAction(
                "This conversation already has unfinished work.".into(),
            ));
        }
        let conversation = self.store.ensure_conversation(conversation_id, &request)?;
        let mut task = Task::new(request.clone());
        if let Some(task_id) = requested_task_id {
            if task_id.is_nil() {
                return Err(CoreError::InvalidAction(
                    "Task identity cannot be empty".into(),
                ));
            }
            task.id = task_id;
        }
        task.synthesized_skill_sources = skill_lineages
            .iter()
            .flat_map(|(_, _, sources)| sources.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        task.synthesized_skill_lineages = skill_lineages.clone();
        let mut contract = crate::contracts::RunContract::local(task.id);
        contract.resources = resources;
        if let Some(expiry) = background_expiry {
            contract.expires_at = contract.expires_at.min(expiry);
            contract.validate(task.id)?;
        }
        task.contract = Some(contract);
        task.conversation_id = Some(conversation.id);
        task.message_id = Some(Uuid::new_v4());
        task.background = background_expiry.is_some();
        if let Some((previous, _)) = &continuation {
            task.continuation_of = Some(previous.id);
            task.control_scope_id = Some(previous.control_scope());
            // Copy evidence, never old actions, approvals or capabilities. The
            // explicit native Continue action authorizes this same-scope handoff.
            task.tool_results = previous
                .tool_results
                .iter()
                .rev()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            for result in &mut task.tool_results {
                result
                    .label
                    .source_ids
                    .insert(format!("run:{}", previous.id));
                result.label.scope_id = task.id;
            }
        }
        if let Some(graph) = graph {
            task.install_plan(crate::workflows::instantiate(&graph, task.id)?)
                .map_err(CoreError::InvalidAction)?;
        } else if continuation.is_none()
            && background_expiry.is_none()
            && let Some(intent) = crate::intent::compile(
                &request,
                &task.contract.as_ref().expect("new contract").resources,
            )
        {
            // A local grammar removes the model round trip, not the broker's
            // policy, exact approval, resource identity or verification gates.
            let (graph, binding) = intent.bind(task.id, request.clone());
            task.install_plan(graph).map_err(CoreError::InvalidAction)?;
            task.intent = Some(binding);
        } else if continuation.is_none()
            && background_expiry.is_none()
            && let Some(graph) = self
                .learned_application_control_graph(&request, task.id)
                .await?
        {
            task.install_plan(graph).map_err(CoreError::InvalidAction)?;
        } else if continuation.is_none()
            && background_expiry.is_none()
            && let Some((routine, procedure)) = self.store.compiled_routine(&request)?
        {
            let mut graph = procedure.graph(task.id, request.clone());
            for node in &mut graph.nodes {
                node.proposal.provenance =
                    crate::domain::Provenance::model(vec![format!("reviewed_routine:{routine}")]);
            }
            task.install_plan(graph).map_err(CoreError::InvalidAction)?;
            task.compiled_routine = Some(routine);
        }
        let procedure_checkpoint = if let Some(procedure) = procedure {
            if !task.workflow_run {
                return Err(CoreError::InvalidAction(
                    "An executable procedure requires a compiled task graph".into(),
                ));
            }
            validate_procedure_action_binding(&task, &procedure)?;
            Some(crate::agency::ProcedureCheckpoint {
                task_id: task.id,
                revision: 0,
                runtime: crate::agency::ProcedureRuntimeState::new_for_task(&procedure, task.id)?,
                procedure,
                updated_at: Utc::now(),
            })
        } else {
            None
        };
        let procedure_stream_run = procedure_checkpoint
            .as_ref()
            .map(|checkpoint| self.create_procedure_stream_run(checkpoint))
            .transpose()?
            .flatten();
        let task_id = task.id;
        if procedure_stream_run.is_some()
            && self
                .procedure_streams
                .lock()
                .expect("procedure streams poisoned")
                .contains_key(&task_id)
        {
            return Err(CoreError::VerificationFailed(
                "Procedure stream runtime already exists for this task".into(),
            ));
        }
        let message = Message {
            id: task.message_id.unwrap(),
            conversation_id: conversation.id,
            task_id: Some(task_id),
            role: "user".into(),
            content: request,
            provenance: crate::domain::Provenance::user(),
            created_at: task.created_at,
        };
        let started = CoreEvent::new(Some(task_id), CoreEventKind::TaskStarted);
        let (stream_release, stream_receiver) = streamed_prefix
            .as_ref()
            .map(|_| tokio::sync::watch::channel(false))
            .map_or((None, None), |(sender, receiver)| {
                (Some(sender), Some(receiver))
            });
        let lease = if let Some((previous, owner)) = &continuation {
            self.runtime.continue_run(previous.id, owner, task_id)?
        } else {
            self.runtime.begin(task_id)?
        };
        // Acquire the cache before the durable acceptance: once commit succeeds
        // there is no await at which its execution owner could be dropped.
        let mut tasks = self.tasks.write().await;
        let continued = if let Some((mut previous, _)) = continuation {
            let event =
                self.store
                    .accept_continuation(&mut task, &message, &started, &mut previous)?;
            tasks.insert(previous.id, previous);
            Some(event)
        } else {
            if let Some(checkpoint) = procedure_checkpoint.as_ref() {
                self.store.accept_task_with_procedure(
                    &mut task,
                    &message,
                    &started,
                    command.as_ref(),
                    checkpoint,
                )?;
            } else {
                self.store
                    .accept_task(&mut task, &message, &started, command.as_ref())?;
            }
            None
        };
        lease.mark_accepted();
        tasks.insert(task_id, task);
        drop(tasks);
        if let Some(event) = continued {
            self.events.publish(event);
        }
        self.events.publish(started);

        if let (Some((stream_id, prefix_action)), Some(release)) = (streamed_prefix, stream_release)
        {
            self.streamed_runs
                .lock()
                .expect("streamed runs poisoned")
                .insert(
                    stream_id,
                    StreamedRun {
                        task_id,
                        prefix_action,
                        release,
                    },
                );
        }
        if let Some(stream_run) = procedure_stream_run {
            self.procedure_streams
                .lock()
                .expect("procedure streams poisoned")
                .insert(task_id, stream_run);
        }

        let core = Arc::clone(self);
        tokio::spawn(core.run_owned(task_id, lease, false, stream_receiver));
        Ok(task_id)
    }

    pub async fn control_task(
        self: &Arc<Self>,
        task_id: Uuid,
        status: TaskStatus,
    ) -> CoreResult<()> {
        if status == TaskStatus::Cancelled {
            return self.stop_task(task_id).await;
        }
        if status == TaskStatus::Paused {
            self.signal_hold(task_id);
        }
        if status == TaskStatus::Running
            && self.get_task(task_id).await?.status == TaskStatus::Interrupted
        {
            return self.resume_interrupted(task_id).await;
        }
        self.update_task(task_id, |task| {
            match status {
                TaskStatus::Paused if task.status.is_active() => {
                    task.status = TaskStatus::Paused;
                }
                TaskStatus::Running
                    if task.status == TaskStatus::Paused
                        || (task.status.is_active() && self.runtime.is_held(task_id)) =>
                {
                    task.status = TaskStatus::Running;
                }
                _ => {
                    return Err(CoreError::InvalidAction(format!(
                        "cannot change task from {:?} to {:?}",
                        task.status, status
                    )));
                }
            }
            task.touch();
            Ok(())
        })
        .await?;
        if status == TaskStatus::Running {
            self.runtime.hold(task_id, false);
        }
        self.control_changed.notify_waiters();
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::TaskStatusChanged {
                status,
                summary: match status {
                    TaskStatus::Cancelled => {
                        "Stop requested. Completed or in-flight effects remain in the task record."
                    }
                    TaskStatus::Paused => "Task paused.",
                    _ => "Task resumed.",
                }
                .into(),
            },
        ))
    }

    pub async fn resolve_approval(
        &self,
        approval_id: Uuid,
        task_id: Uuid,
        action_id: Uuid,
        digest: &str,
        resolution: ApprovalResolution,
    ) -> CoreResult<()> {
        if self.runtime.is_stopped(task_id) {
            return Err(CoreError::Cancelled);
        }
        let mut pending = self.pending_approvals.lock().await;
        let matches = pending.get(&approval_id).is_some_and(|approval| {
            approval.expires_at > Utc::now()
                && approval.task_id == task_id
                && approval.action_id == action_id
                && approval.digest == digest
                && (resolution == ApprovalResolution::Denied
                    || !approval.requires_native_authentication
                    || matches!(
                        resolution,
                        ApprovalResolution::Approved {
                            native_authentication_satisfied: true
                        }
                    ))
        });
        if !matches {
            return Err(CoreError::ApprovalRejected(
                "approval is stale, mismatched, or missing required native authentication".into(),
            ));
        }
        let record =
            crate::decisions::DecisionRecord::Approval(pending[&approval_id].record.clone());
        let outcome = match resolution {
            ApprovalResolution::Approved { .. } => crate::decisions::DecisionResolution::Approved,
            ApprovalResolution::Denied => crate::decisions::DecisionResolution::Denied,
        };
        let event = self
            .store
            .resolve_decision(&record, &outcome)?
            .ok_or_else(|| {
                CoreError::ApprovalRejected("This decision has already closed".into())
            })?;
        self.events.publish(event);
        let approval = pending
            .remove(&approval_id)
            .ok_or_else(|| CoreError::ApprovalRejected("approval no longer exists".into()))?;
        approval
            .sender
            .send(resolution)
            .map_err(|_| CoreError::ApprovalRejected("task no longer accepts this approval".into()))
    }

    pub async fn answer_question(
        &self,
        question_id: Uuid,
        task_id: Uuid,
        action_id: Uuid,
        answer: String,
    ) -> CoreResult<()> {
        if self.runtime.is_stopped(task_id) {
            return Err(CoreError::Cancelled);
        }
        if answer.trim().is_empty() || answer.len() > 16 * 1024 {
            return Err(CoreError::InvalidAction(
                "answer must contain between 1 and 16384 characters".into(),
            ));
        }
        let mut pending = self.pending_questions.lock().await;
        let matches = pending.get(&question_id).is_some_and(|question| {
            question.task_id == task_id
                && question.action_id == action_id
                && question.record.expires_at > Utc::now()
        });
        if !matches {
            return Err(CoreError::InvalidAction(
                "question is stale or belongs to another action".into(),
            ));
        }
        let answer = redact_for_persistence(&answer);
        let record =
            crate::decisions::DecisionRecord::Question(pending[&question_id].record.clone());
        let event = self
            .store
            .resolve_decision(
                &record,
                &crate::decisions::DecisionResolution::Answered(answer.clone()),
            )?
            .ok_or_else(|| CoreError::InvalidAction("This question has already closed".into()))?;
        self.events.publish(event);
        let question = pending
            .remove(&question_id)
            .ok_or_else(|| CoreError::InvalidAction("question no longer exists".into()))?;
        question
            .sender
            .send(answer)
            .map_err(|_| CoreError::InvalidAction("task no longer accepts this answer".into()))
    }

    pub async fn undo_last_action(&self, task_id: Uuid, action_id: Uuid) -> CoreResult<()> {
        // Keep Resume/Continue admission outside the entire compensation window.
        let _submission = self.submission_lock.lock().await;
        let _effect_exclusion = self.effect_ownership.acquire_exclusive().await;
        let scope = self.get_task(task_id).await?.control_scope();
        let members: Vec<_> = self
            .tasks
            .read()
            .await
            .values()
            .filter(|task| task.control_scope() == scope)
            .map(|task| (task.id, task.status))
            .collect();
        if members.iter().any(|(id, status)| {
            status.is_active()
                || self.runtime.is_active(*id)
                || self.adapters.has_outstanding_effects(*id)
        }) {
            return Err(CoreError::PermissionRequired(
                "Wait for this task, its continuations and their worker results before Undo".into(),
            ));
        }
        if members.iter().any(|(id, _)| self.runtime.is_stopped(*id)) {
            return Err(CoreError::PermissionRequired(
                "Retry Stop to save its state before Undo".into(),
            ));
        }
        self.reconcile_task_worker_receipts(task_id).await?;
        if self.get_task(task_id).await?.status.is_active() {
            return Err(CoreError::PermissionRequired(
                "Stop the task before undoing its changes".into(),
            ));
        }
        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get_mut(&task_id)
            .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))?;
        let _retention = self
            .store
            .retention_lock
            .lock()
            .map_err(|_| CoreError::Storage("Retention lock poisoned".into()))?;
        crate::undo::run_requested(
            &self.store,
            task,
            action_id,
            self.secret_store.as_ref(),
            |event| {
                self.events.publish(event);
            },
        )
    }

    pub fn update_permission(&self, permission: &str, granted: bool) -> CoreResult<()> {
        let permission = permission.trim();
        if permission.is_empty() || permission.len() > 128 {
            return Err(CoreError::InvalidAction(
                "permission name must contain between 1 and 128 characters".into(),
            ));
        }
        self.store.set_permission(permission, granted)?;
        self.publish(CoreEvent::new(
            None,
            CoreEventKind::PermissionChanged {
                permission: permission.into(),
                granted,
            },
        ))
    }

    pub(crate) fn signal_stop(&self, task_id: Uuid) -> bool {
        // Signal before acquiring a task-cache, database or credential lock.
        let signalled = self.runtime.stop(task_id);
        let mut scope_tasks = self.runtime.scope_tasks(task_id);
        scope_tasks.push(task_id);
        let scope_tasks = scope_tasks.into_iter().collect::<BTreeSet<_>>();
        let streams = self
            .procedure_streams
            .lock()
            .expect("procedure streams poisoned");
        for stream_task in scope_tasks {
            if let Some(streams) = streams.get(&stream_task) {
                streams.cancellation.cancel();
            }
        }
        self.control_changed.notify_waiters();
        signalled
    }

    pub(crate) fn signal_hold(&self, task_id: Uuid) -> bool {
        let held = self.runtime.hold(task_id, true);
        self.control_changed.notify_waiters();
        held
    }

    async fn stop_task(&self, task_id: Uuid) -> CoreResult<()> {
        self.signal_stop(task_id);
        let requested = self.get_task(task_id).await?;
        let scope = requested.control_scope();
        self.signal_stop(scope);
        let mut affected: std::collections::BTreeSet<_> =
            self.runtime.scope_tasks(scope).into_iter().collect();
        affected.insert(task_id);
        affected.extend(
            self.tasks
                .read()
                .await
                .values()
                .filter(|task| {
                    task.control_scope() == scope
                        && !task.status.is_terminal()
                        && task.continued_by.is_none()
                })
                .map(|task| task.id),
        );
        // Revoke every currently accepted member before any storage operation.
        for id in &affected {
            self.capabilities.revoke_task(*id).await;
        }
        self.pending_approvals
            .lock()
            .await
            .retain(|_, pending| !affected.contains(&pending.task_id));
        self.pending_questions
            .lock()
            .await
            .retain(|_, pending| !affected.contains(&pending.task_id));
        let mut first_error = None;
        for id in affected {
            match self.persist_stopped_run(id).await {
                Ok(_) => {}
                Err(error) => {
                    self.events.publish(CoreEvent::new(Some(id), CoreEventKind::Error {
                        code: "stop_state_pending".into(), message: "Stop was signalled, but saving its state is pending. In-flight effects may need review.".into(), recoverable: false,
                    }));
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(()), |error| {
            Err(CoreError::Storage(format!(
                "Stop was signalled; saving its state failed: {error}"
            )))
        })
    }

    async fn persist_stopped_run(&self, task_id: Uuid) -> CoreResult<TaskStatus> {
        // Save scope intent first: startup can recover cancellation even when
        // the aggregate projection transaction cannot commit.
        let scope = self.get_task(task_id).await?.control_scope();
        self.store.stop_control_scope(scope)?;
        self.finalize_task(task_id, crate::finalization::Finish::Stop)
            .await
    }

    async fn finalize_task(
        &self,
        task_id: Uuid,
        mut finish: crate::finalization::Finish,
    ) -> CoreResult<TaskStatus> {
        finish = finish.redacted();
        self.capabilities.revoke_task(task_id).await;
        self.pending_approvals
            .lock()
            .await
            .retain(|_, pending| pending.task_id != task_id);
        self.pending_questions
            .lock()
            .await
            .retain(|_, pending| pending.task_id != task_id);
        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get(&task_id)
            .cloned()
            .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))?;
        if self.runtime.is_stopped(task_id) {
            finish = crate::finalization::Finish::Stop;
        }
        let stopping = matches!(&finish, crate::finalization::Finish::Stop);
        if stopping {
            self.store.stop_control_scope(task.control_scope())?;
        }
        let attempt = task.execution_attempt;
        let (task, events) = match self.store.finalize_run(task, finish.clone()) {
            Ok(result) => result,
            Err(error) => {
                self.runtime.retain_finish(task_id, attempt, finish);
                self.events.publish(CoreEvent::new(Some(task_id), CoreEventKind::Error {
                    code: "finalization_pending".into(), message: "The run ended, but its final record could not be saved. Retry Resume after storage is available.".into(), recoverable: false,
                }));
                return Err(error);
            }
        };
        let status = task.status;
        tasks.insert(task_id, task);
        self.runtime.finish_saved(task_id);
        if stopping {
            self.runtime.persisted_stop(task_id);
        }
        for event in events {
            self.events.publish(event);
        }
        drop(tasks);
        if let Err(error) = self.store.checkpoint_audit(self.secret_store.as_ref()) {
            self.events.publish(CoreEvent::new(
                Some(task_id),
                CoreEventKind::Error {
                    code: "finalization_checkpoint_pending".into(),
                    message: "The result was saved, but its audit checkpoint needs attention."
                        .into(),
                    recoverable: false,
                },
            ));
            return Err(error);
        }
        Ok(status)
    }

    async fn run_owned(
        self: Arc<Self>,
        task_id: Uuid,
        lease: crate::runtime::RunLease,
        recovery: bool,
        mut streamed_gate: Option<tokio::sync::watch::Receiver<bool>>,
    ) {
        let mut recovering = recovery;
        let outcome = {
            let operation = async {
                if recovery {
                    self.prepare_recovery(task_id).await?;
                    self.restore_procedure_stream_run(task_id)?;
                    self.capabilities.reopen_run(task_id).await?;
                }
                recovering = false;
                self.run_task(task_id, &mut streamed_gate).await
            };
            tokio::select! { biased;
                _ = lease.signal.cancelled() => Err(CoreError::Cancelled),
                outcome = operation => outcome,
            }
        }; // The operation future is dropped before cleanup and lease release.
        if let Err(error) = outcome {
            self.capabilities.revoke_task(task_id).await;
            if lease.signal.is_stopped() {
                self.pending_approvals
                    .lock()
                    .await
                    .retain(|_, pending| pending.task_id != task_id);
                self.pending_questions
                    .lock()
                    .await
                    .retain(|_, pending| pending.task_id != task_id);
                match self.persist_stopped_run(task_id).await {
                    Ok(_) => {}
                    Err(_) => self.events.publish(CoreEvent::new(
                        Some(task_id),
                        CoreEventKind::Error {
                            code: "stop_state_pending".into(),
                            message: "The active run stopped, but saving its state is pending."
                                .into(),
                            recoverable: false,
                        },
                    )),
                }
            } else if recovering {
                let _ = self
                    .set_task_status(
                        task_id,
                        TaskStatus::Interrupted,
                        &format!("Recovery requires review: {error}"),
                    )
                    .await;
            } else {
                let _ = self.fail_task(task_id, error.to_string()).await;
            }
        }
        self.remove_streamed_run(task_id);
        self.remove_procedure_stream_run(task_id);
        drop(lease);
    }

    fn remove_procedure_stream_run(&self, task_id: Uuid) {
        if let Some(streams) = self
            .procedure_streams
            .lock()
            .expect("procedure streams poisoned")
            .remove(&task_id)
        {
            streams.cancellation.cancel();
        }
    }

    fn remove_streamed_run(&self, task_id: Uuid) {
        let mut runs = self.streamed_runs.lock().expect("streamed runs poisoned");
        let stream_ids = runs
            .iter()
            .filter_map(|(id, run)| (run.task_id == task_id).then_some(*id))
            .collect::<Vec<_>>();
        for stream_id in stream_ids {
            if let Some(run) = runs.remove(&stream_id) {
                run.release.send_replace(true);
            }
        }
    }

    async fn run_task(
        self: &Arc<Self>,
        task_id: Uuid,
        streamed_gate: &mut Option<tokio::sync::watch::Receiver<bool>>,
    ) -> CoreResult<()> {
        self.wait_until_runnable(task_id).await?;
        let initial = self.get_task(task_id).await?;
        let workflow = initial.workflow_run;
        let reference_requested = crate::context::should_capture_live_reference(
            &initial.request,
            workflow,
            initial.background,
        ) && self.model.descriptor().id != "unconfigured";
        let reference_kind = crate::context::live_reference_kind(&initial.request);
        let include_page_text =
            reference_requested && crate::context::requests_page_text(&initial.request);
        let mut reference_loaded = initial.reference_captured;
        let mut reference_observation: Option<crate::context::ContextObservation> = None;
        if !workflow {
            if initial
                .request
                .trim()
                .to_ascii_lowercase()
                .starts_with("forget ")
            {
                self.cancel_context_readers(Some(task_id)).await?;
            }
            if let Some(answer) = self
                .store
                .memory_command(&initial.request, initial.message_id.unwrap_or(initial.id))?
            {
                return self.finish_answer(task_id, answer).await;
            }
        }
        let mut repairs = 0;
        loop {
            self.wait_until_runnable(task_id).await?;
            let task = self.get_task(task_id).await?;
            let contract = task.contract.as_ref().ok_or_else(|| {
                CoreError::PermissionRequired("This pre-v2 task needs a new authorized run".into())
            })?;
            contract.validate(task_id)?;
            if !task.ready_actions().is_empty() {
                let mut needs_repair = false;
                for (action_id, outcome) in self.execute_ready_actions(task_id).await? {
                    if !self.get_task(task_id).await?.action_is_current(action_id) {
                        continue;
                    }
                    match outcome {
                        Ok(()) => {}
                        Err(failure)
                            if !workflow
                                && failure.recoverable
                                && repairs < contract.max_repairs =>
                        {
                            repairs += 1;
                            needs_repair = true;
                            self.update_task(task_id, |task| {
                                if let Some(state) = task.actions.get_mut(&action_id) {
                                    state.status = ActionStatus::Skipped;
                                }
                                if !task
                                    .tool_results
                                    .iter()
                                    .any(|result| result.action_id == action_id)
                                {
                                    task.tool_results.push(crate::contracts::ToolResult {
                                        action_id,
                                        tool: "planning_error".into(),
                                        verdict: crate::contracts::Verdict::Failed,
                                        summary: failure.error.to_string(),
                                        output: failure.observation,
                                        label: crate::contracts::DataLabel::private(
                                            task_id,
                                            action_id.to_string(),
                                        ),
                                        observed_at: Utc::now(),
                                    });
                                }
                                Ok(())
                            })
                            .await?;
                        }
                        Err(failure) => return Err(failure.error),
                    }
                }
                if !needs_repair {
                    repairs = 0;
                }
                // Completed reads may unlock successors while other reads are
                // still in flight. Planning waits for all admitted work to settle.
                if !needs_repair || !self.get_task(task_id).await?.ready_actions().is_empty() {
                    continue;
                }
            } else {
                self.wait_for_streamed_prefix(task_id, streamed_gate)
                    .await?;
                self.wait_until_runnable(task_id).await?;
                let current = self.get_task(task_id).await?;
                if workflow && current.is_complete() {
                    if !self.procedure_completion_satisfied(&current)? {
                        return Err(CoreError::ExecutorUnavailable(
                            "The goal procedure stopped before every required step had fresh verified completion; review the current system and submit a new goal to continue".into(),
                        ));
                    }
                    if self.succeed_task(task_id).await? {
                        return Ok(());
                    }
                    continue;
                }
                // A streamed-prefix release may have appended ready work
                // while this run waited for speech to finish. Re-enter the
                // scheduler before asking the planner for another turn.
                if workflow && !current.ready_actions().is_empty() {
                    continue;
                }
            }
            self.set_task_status(task_id, TaskStatus::Planning, "Preparing the next response")
                .await?;
            let task = self.get_task(task_id).await?;
            let observations = if reference_requested {
                if !reference_loaded {
                    reference_loaded = true;
                    self.update_task(task_id, |task| {
                        task.reference_captured = true;
                        task.touch();
                        Ok(())
                    })
                    .await?;
                    self.publish(CoreEvent::new(
                        Some(task_id),
                        CoreEventKind::ReferenceContext {
                            summary:
                                "Checking only the current selection or page named in your request."
                                    .into(),
                        },
                    ))?;
                    if task.recovery_attempt {
                        reference_observation = Some(crate::context::bind_live_reference_kind(
                            crate::context::unavailable_reference(true),
                            reference_kind,
                        ));
                    } else {
                        let observed = self.adapters.reference(include_page_text).await;
                        reference_observation = Some(crate::context::bind_live_reference_kind(
                            crate::context::select_live_reference(observed),
                            reference_kind,
                        ));
                    }
                    if let Some(observation) = &reference_observation {
                        self.publish(CoreEvent::new(
                            Some(task_id),
                            CoreEventKind::ReferenceContext {
                                summary: crate::context::reference_summary(observation),
                            },
                        ))?;
                    }
                } else if reference_observation.is_none() {
                    // A resumed process has only the durable consumed marker,
                    // never the selected text. Do not silently bind a new app.
                    reference_observation = Some(crate::context::bind_live_reference_kind(
                        crate::context::unavailable_reference(true),
                        reference_kind,
                    ));
                    self.publish(CoreEvent::new(
                        Some(task_id),
                        CoreEventKind::ReferenceContext {
                            summary: crate::context::reference_summary(
                                reference_observation.as_ref().expect("just initialized"),
                            ),
                        },
                    ))?;
                }
                let observation = reference_observation
                    .as_ref()
                    .expect("explicit reference state is initialized");
                if (Utc::now().timestamp_millis() - observation.observed_at_unix_ms).abs()
                    > crate::context::REFERENCE_CONTEXT_TTL_MS
                    && observation
                        .state
                        .get("expired")
                        .and_then(serde_json::Value::as_bool)
                        != Some(true)
                {
                    reference_observation = Some(crate::context::bind_live_reference_kind(
                        crate::context::unavailable_reference(true),
                        reference_kind,
                    ));
                    self.publish(CoreEvent::new(
                        Some(task_id),
                        CoreEventKind::ReferenceContext {
                            summary: crate::context::reference_summary(
                                reference_observation.as_ref().expect("just initialized"),
                            ),
                        },
                    ))?;
                }
                vec![
                    reference_observation
                        .as_ref()
                        .expect("explicit reference state is initialized")
                        .clone(),
                ]
            } else {
                Vec::new()
            };
            let planning = crate::context::build_context(
                &self.store,
                &task,
                if task.actions.len() >= contract.max_steps as usize {
                    Vec::new()
                } else {
                    self.available_tools().await
                },
                observations,
            )?;
            let destination = self.model.data_destination()?;
            let context = TurnContext {
                planning,
                results: task
                    .tool_results
                    .iter()
                    .rev()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect(),
                destination: destination.clone(),
            };
            if let Some(destination) = &destination {
                self.approve_data_release(&task, destination, &context)
                    .await?;
            }
            self.wait_until_runnable(task_id).await?;
            let (updates, mut prefixes) = tokio::sync::mpsc::channel(8);
            let response = self.model.next_turn_stream(context, updates);
            tokio::pin!(response);
            let turn = loop {
                let changed = self.control_changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if self.get_task(task_id).await?.status == TaskStatus::Cancelled {
                    return Err(CoreError::Cancelled);
                }
                tokio::select! {
                    result=&mut response => break result?, _=&mut changed => {},
                    Some(prefix)=prefixes.recv()=>self.events.publish(CoreEvent::new(Some(task_id),CoreEventKind::ModelResponse{text:redact_for_persistence(&prefix),finished:false})),
                }
            };
            self.wait_until_runnable(task_id).await?;
            match turn {
                ModelTurn::Answer(answer) => return self.finish_answer(task_id, answer).await,
                ModelTurn::Actions(graph) => {
                    let count = graph.nodes.len();
                    if task.actions.len().saturating_add(count) > contract.max_steps as usize {
                        self.finalize_task(task_id, crate::finalization::Finish::Interrupted {
                            summary: "The tool budget is exhausted. Review the completed work; Continue starts a new run with up to 32 steps and the same folders. Expanded access still requires approval.".into(),
                            budget_exhausted: true,
                        }).await?;
                        return Ok(());
                    }
                    self.update_task(task_id, |task| {
                        task.append_plan(graph).map_err(CoreError::InvalidAction)
                    })
                    .await?;
                    self.publish(CoreEvent::new(
                        Some(task_id),
                        CoreEventKind::PlanGenerated {
                            action_count: count,
                        },
                    ))?;
                }
            }
        }
    }

    async fn wait_for_streamed_prefix(
        &self,
        task_id: Uuid,
        streamed_gate: &mut Option<tokio::sync::watch::Receiver<bool>>,
    ) -> CoreResult<()> {
        let Some(receiver) = streamed_gate.as_mut() else {
            return Ok(());
        };
        loop {
            if self.runtime.is_stopped(task_id) {
                return Err(CoreError::Cancelled);
            }
            if *receiver.borrow_and_update() {
                *streamed_gate = None;
                return Ok(());
            }
            receiver.changed().await.map_err(|_| CoreError::Cancelled)?;
        }
    }

    async fn approve_data_release(
        &self,
        task: &Task,
        destination: &str,
        context: &TurnContext,
    ) -> CoreResult<()> {
        use sha2::{Digest, Sha256};
        let payload = serde_json::to_vec(context)?;
        let sources = context
            .planning
            .untrusted_context
            .iter()
            .filter(|c| !c.content.is_empty())
            .map(|c| c.source.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let explanation = format!(
            "Send this request and selected context to {destination}? Sources: {sources}; {} verified tool results; {} bytes. This endpoint may process data outside this device.\n\nExact data selected for release:\n{}",
            context.results.len(),
            payload.len(),
            String::from_utf8_lossy(&payload)
        );
        let proposal = crate::domain::ActionProposal {
            id: Uuid::new_v4(),
            task_id: task.id,
            action: Action::AskUser {
                question: explanation.clone(),
            },
            expected_outcome: crate::domain::ExpectedOutcome::UserAnswered,
            target_resource: destination.into(),
            provenance: crate::domain::Provenance::user(),
            metadata: std::collections::BTreeMap::from([(
                "payload_sha256".into(),
                format!("{:x}", Sha256::digest(&payload)),
            )]),
        };
        self.await_approval(
            &proposal,
            RiskLevel::Consequential,
            explanation,
            crate::policy::approval_digest(&proposal)?,
        )
        .await
    }

    async fn finish_answer(&self, task_id: Uuid, answer: String) -> CoreResult<()> {
        self.wait_until_runnable(task_id).await?;
        self.finalize_task(task_id, crate::finalization::Finish::Answer(answer))
            .await
            .map(|_| ())
    }

    fn ready_procedure_stream_cohorts(
        &self,
        task: &Task,
        ready: &[Uuid],
    ) -> CoreResult<(BTreeSet<Uuid>, Vec<Vec<Uuid>>)> {
        let Some(procedure_id) = task
            .actions
            .values()
            .find_map(|state| state.proposal.metadata.get("procedure_id"))
        else {
            return Ok((BTreeSet::new(), Vec::new()));
        };
        let checkpoint = self
            .store
            .load_procedure_checkpoint(task.id)?
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Procedure stream scheduler has no durable checkpoint".into(),
                )
            })?;
        if checkpoint.procedure.id != *procedure_id {
            return Err(CoreError::VerificationFailed(
                "Procedure stream scheduler loaded another procedure".into(),
            ));
        }
        if checkpoint.procedure.streams.is_empty() {
            return Ok((BTreeSet::new(), Vec::new()));
        }

        let mut action_by_node = BTreeMap::<String, Uuid>::new();
        for (action_id, state) in &task.actions {
            if state.proposal.metadata.get("procedure_id") != Some(procedure_id) {
                return Err(CoreError::VerificationFailed(
                    "Procedure task mixes stream actions from another procedure".into(),
                ));
            }
            let node_id = state
                .proposal
                .metadata
                .get("procedure_node_id")
                .ok_or_else(|| {
                    CoreError::VerificationFailed(
                        "Procedure stream action has no node identity".into(),
                    )
                })?;
            if action_by_node.insert(node_id.clone(), *action_id).is_some() {
                return Err(CoreError::VerificationFailed(
                    "Procedure stream node has multiple task actions".into(),
                ));
            }
        }

        let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
        for channel in &checkpoint.procedure.streams {
            adjacency
                .entry(channel.producer_node.clone())
                .or_default()
                .insert(channel.consumer_node.clone());
            adjacency
                .entry(channel.consumer_node.clone())
                .or_default()
                .insert(channel.producer_node.clone());
        }
        let mut unvisited = adjacency.keys().cloned().collect::<BTreeSet<_>>();
        let ready_ids = ready.iter().copied().collect::<BTreeSet<_>>();
        let mut stream_action_ids = BTreeSet::new();
        let mut ready_cohorts = Vec::new();
        while let Some(first) = unvisited.pop_first() {
            let mut pending = vec![first.clone()];
            let mut nodes = BTreeSet::new();
            while let Some(node) = pending.pop() {
                if !nodes.insert(node.clone()) {
                    continue;
                }
                if let Some(neighbors) = adjacency.get(&node) {
                    for neighbor in neighbors {
                        unvisited.remove(neighbor);
                        if !nodes.contains(neighbor) {
                            pending.push(neighbor.clone());
                        }
                    }
                }
            }
            if nodes.len() > 16 {
                return Err(CoreError::InvalidAction(
                    "Connected procedure stream cohorts are limited to sixteen calls".into(),
                ));
            }
            let cohort_actions = nodes
                .iter()
                .filter_map(|node_id| action_by_node.get(node_id).copied())
                .collect::<BTreeSet<_>>();
            stream_action_ids.extend(cohort_actions.iter().copied());
            let is_complete_ready_cohort = cohort_actions.len() == nodes.len()
                && cohort_actions
                    .iter()
                    .all(|action_id| ready_ids.contains(action_id));
            if is_complete_ready_cohort {
                ready_cohorts.push(cohort_actions.into_iter().collect());
            } else if cohort_actions
                .iter()
                .any(|action_id| ready_ids.contains(action_id))
            {
                return Err(CoreError::PermissionRequired(
                    "A ready procedure stream endpoint has no complete ready peer cohort".into(),
                ));
            }
        }
        Ok((stream_action_ids, ready_cohorts))
    }

    async fn execute_ready_actions(
        &self,
        task_id: Uuid,
    ) -> CoreResult<Vec<(Uuid, Result<(), StepFailure>)>> {
        let mut active = crate::scheduling::InFlight::new();
        let mut outcomes = Vec::new();
        let mut failed = false;
        let mut serial = false;
        let mut _wave_registrations = Vec::new();
        loop {
            let changed = self.control_changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let current = self.get_task(task_id).await?;
            let runnable = current.ready_actions();
            // An admitted but unpolled successor can be rewired by a
            // correction. Drop that future until its new dependencies finish.
            active.retain(|id| {
                current.action_is_current(id)
                    && current.actions.get(&id).is_some_and(|state| {
                        state.status != ActionStatus::Pending || runnable.contains(&id)
                    })
            });
            if serial
                && !active
                    .ids()
                    .any(|id| !crate::scheduling::parallel_read(&current, id))
            {
                serial = false;
            }
            if !failed && !serial {
                let task = current;
                let ready = self
                    .execution_timings
                    .lock()
                    .expect("timings poisoned")
                    .ready(&task);
                let (stream_action_ids, stream_cohorts) =
                    self.ready_procedure_stream_cohorts(&task, &ready)?;
                let mut stream_cohort_started = false;
                if active.len() == 0
                    && let Some(cohort) = stream_cohorts.into_iter().next()
                {
                    _wave_registrations
                        .push(self.register_procedure_dispatch_wave(task_id, &cohort)?);
                    for id in cohort {
                        active.push(id, self.execute_action(task_id, id));
                    }
                    stream_cohort_started = true;
                }
                if !stream_cohort_started {
                    for id in ready {
                        if stream_action_ids.contains(&id) || active.contains(id) {
                            continue;
                        }
                        if active.len() == crate::scheduling::MAX_PARALLEL_READS {
                            break;
                        }
                        let parallel = crate::scheduling::parallel_read(&task, id);
                        if !parallel && active.len() != 0 {
                            continue;
                        }
                        active.push(id, self.execute_action(task_id, id));
                        if !parallel {
                            serial = true;
                            break;
                        }
                    }
                }
            }
            let completed = tokio::select! { biased;
                _ = &mut changed => continue,
                result = active.next() => result,
            };
            let Some((id, result)) = completed else {
                break;
            };
            failed |= result.is_err();
            outcomes.push((id, result));
            // Settle already admitted work on error. Mutations and approval
            // requests are admitted alone, preserving their exact effect gate.
            if active.len() == 0 && (failed || serial) {
                break;
            }
        }
        Ok(outcomes)
    }

    async fn execute_action(&self, task_id: Uuid, action_id: Uuid) -> Result<(), StepFailure> {
        let _prepared = self.files.preparation_guard(action_id);
        let started = std::time::Instant::now();
        let result = self.execute_action_inner(task_id, action_id).await;
        if let Err(error) = &result {
            let is_stream_action = self.procedure_dispatch_wave_for_action(action_id).is_some();
            self.abort_procedure_dispatch_wave(action_id, error).await;
            if is_stream_action {
                self.cancel_procedure_stream_run(task_id);
            }
            self.capabilities.revoke_action(task_id, action_id).await;
        }
        if result.is_ok()
            && let Ok(task) = self.get_task(task_id).await
            && let Some(action) = task.actions.get(&action_id)
        {
            self.execution_timings
                .lock()
                .expect("timings poisoned")
                .record(
                    action.proposal.action.kind(),
                    started.elapsed().as_micros().min(u64::MAX as u128) as u64,
                );
        }
        self.files.discard(action_id);
        let Err(error) = result else {
            return Ok(());
        };
        let mut tasks = self.tasks.write().await;
        let Some(task) = tasks.get(&task_id).cloned() else {
            return Err(StepFailure {
                recoverable: false,
                observation: json!({"error":error.to_string()}),
                error,
            });
        };
        if !task.action_is_current(action_id) {
            return Ok(());
        }
        let confirmed = task
            .actions
            .get(&action_id)
            .is_some_and(|action| action.status == ActionStatus::Succeeded)
            && task
                .tool_results
                .iter()
                .rev()
                .find(|result| result.action_id == action_id)
                .is_some_and(|result| result.verdict == crate::contracts::Verdict::Confirmed);
        if confirmed {
            return Err(StepFailure {
                recoverable: false,
                observation: json!({"verified":true,"error":error.to_string()}),
                error,
            });
        }
        let committed = self
            .store
            .commit_interrupted_action(task, action_id, &error.to_string());
        let may_have_executed = match committed {
            Ok((task, event, may_have_executed)) => {
                tasks.insert(task_id, task);
                drop(tasks);
                self.events.publish(event);
                may_have_executed
            }
            Err(persistence) => {
                // A failed state commit must never turn into a model retry.
                return Err(StepFailure {
                    recoverable: false,
                    observation: json!({"error":error.to_string(),"state_error":persistence.to_string()}),
                    error: persistence,
                });
            }
        };
        Err(StepFailure {
            recoverable: !may_have_executed
                && !matches!(
                    error,
                    CoreError::ApprovalRejected(_)
                        | CoreError::PolicyDenied(_)
                        | CoreError::Cancelled
                        | CoreError::Storage(_)
                        | CoreError::SecretStore(_)
                ),
            observation: json!({"error":error.to_string()}),
            error,
        })
    }

    async fn execute_action_inner(&self, task_id: Uuid, action_id: Uuid) -> CoreResult<()> {
        self.update_action(task_id, action_id, |state| {
            state.status = ActionStatus::Compiling;
            state.attempts += 1;
            Ok(())
        })
        .await?;
        let task = self.get_task(task_id).await?;
        let raw = task
            .actions
            .get(&action_id)
            .ok_or_else(|| CoreError::InvalidAction("action disappeared".into()))?
            .proposal
            .clone();
        self.validate_routine_binding(&task)?;
        let resolver = self.resolver.clone();
        let mut proposal =
            crate::execution::io::bounded_read(move |_| resolver.prepare_proposal(&raw)).await?;
        self.compiler
            .compile(proposal.clone(), &self.current_availability().await)?;
        if proposal.action.domain() == crate::domain::ExecutionDomain::Browser {
            let target: crate::browser_target::BrowserTarget = serde_json::from_value(
                self.adapters
                    .request("browser", "binding", json!({}))
                    .await?,
            )?;
            target.validate()?;
            proposal
                .metadata
                .insert("browser_target".into(), serde_json::to_string(&target)?);
        }
        if let Action::OpenApplication { application } = &proposal.action {
            let target = self.prepared_application(application).await?;
            proposal.action = Action::OpenApplication {
                application: target.identifier.clone(),
            };
            proposal.target_resource = target.identifier.clone();
            // Always replace model-supplied metadata with the native observation.
            proposal
                .metadata
                .insert("application_target".into(), serde_json::to_string(&target)?);
        } else if matches!(proposal.action, Action::SetApplicationControl { .. }) {
            self.prepare_application_control(&mut proposal, None)
                .await?;
        }
        crate::verification::bind_required_outcome(&mut proposal)?;
        self.files.prepare(&mut proposal).await?;
        let prepared = crate::contracts::PreparedAction::new(
            &proposal,
            task.tool_results
                .iter()
                .map(|result| result.action_id)
                .collect(),
        )?;
        self.commit_task_transition(task_id, |task| {
            self.store.commit_prepared_action(task, &prepared)
        })
        .await?;

        let decision = self.policy.evaluate(
            &proposal,
            &PolicyContext {
                task_request: task.request.clone(),
                has_fresh_native_authentication: false,
                is_recovery_attempt: task.recovery_attempt,
            },
        )?;
        let scoped = task
            .contract
            .as_ref()
            .is_some_and(|scope| scope.covers(&proposal.action));
        let mut approved = false;
        match decision {
            PolicyDecision::Deny { reason, .. } => {
                self.publish(CoreEvent::new(
                    Some(task_id),
                    CoreEventKind::PolicyDenied {
                        action_id,
                        reason: reason.clone(),
                    },
                ))?;
                return Err(CoreError::PolicyDenied(reason));
            }
            PolicyDecision::RequireApproval {
                risk,
                explanation,
                digest,
            } => {
                if !scoped || risk >= RiskLevel::Destructive {
                    let explanation = format!("{explanation}\n\n{}", prepared.preview);
                    self.await_approval(&proposal, risk, explanation, digest)
                        .await?;
                    approved = true;
                }
            }
            PolicyDecision::Allow { .. } => {}
        }

        let contract = task
            .contract
            .as_ref()
            .ok_or_else(|| CoreError::PermissionRequired("Missing task scope".into()))?;
        contract.validate(task_id)?;
        crate::authorization::authorize(
            scoped || matches!(proposal.action, Action::AskUser { .. }),
            approved,
            crate::policy::classify(&proposal.action) >= RiskLevel::Destructive,
            false,
            false,
        )?;

        if let Action::AskUser { question } = &proposal.action {
            self.commit_dispatch(task_id, &prepared, None, "native-user-interaction")
                .await?;
            let receipt = self.await_question(&proposal, question.clone()).await?;
            return self
                .observe_and_verify(task_id, action_id, &proposal, &receipt)
                .await;
        }

        // Approval may have waited several minutes. Cancellation and resources
        // must be checked again before issuing a new single-use grant.
        self.wait_until_runnable(task_id).await?;
        let _effect_lease = self.effect_ownership.acquire_action(&prepared).await?;
        self.wait_until_runnable(task_id).await?;
        contract.validate(task_id)?;
        self.validate_routine_binding(&task)?;
        let resolver = self.resolver.clone();
        let proposed = proposal.clone();
        let mut refreshed =
            crate::execution::io::bounded_read(move |_| resolver.prepare_proposal(&proposed))
                .await?;
        if matches!(refreshed.action, Action::SetApplicationControl { .. }) {
            self.prepare_application_control(&mut refreshed, Some(&proposal))
                .await?;
        }
        if refreshed != proposal {
            return Err(CoreError::ApprovalRejected(
                "Resource changed while awaiting approval.".into(),
            ));
        }
        let availability = self.current_availability().await;
        let compiled = self.compiler.compile(proposal.clone(), &availability)?;
        let implementation = self.broker.select(&compiled)?.clone();
        let expects_streams = proposal
            .metadata
            .get("procedure_stream_node")
            .is_some_and(|value| value == "true");
        if expects_streams
            && !self
                .broker
                .supports_procedure_streams(implementation.executor)
        {
            return Err(CoreError::ExecutorUnavailable(
                "No registered executor is qualified for this procedure stream".into(),
            ));
        }
        let stream_endpoints = self.take_procedure_node_streams(task_id, &proposal)?;
        if expects_streams != stream_endpoints.is_some() {
            return Err(CoreError::VerificationFailed(
                "Procedure stream metadata and runtime endpoints disagree".into(),
            ));
        }
        let grant = self
            .capabilities
            .issue(&proposal, implementation.executor)
            .await?;
        let domain = if implementation.executor == crate::domain::ExecutionDomain::Browser {
            "browser"
        } else {
            "native"
        };
        let grant = if let Some(session) = self.adapters.session_id(domain).await {
            self.capabilities.bind_worker(grant.id, session).await?
        } else {
            grant
        };
        self.commit_dispatch(task_id, &prepared, Some(&grant), &implementation.operation)
            .await?;
        self.store
            .update_working_memory(&self.get_task(task_id).await?)?;
        self.store.checkpoint_audit(self.secret_store.as_ref())?;
        let execution =
            self.broker
                .execute_with_streams(&compiled, &implementation, &grant, stream_endpoints);
        tokio::pin!(execution);
        let receipt = loop {
            let changed = self.control_changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.get_task(task_id).await?.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            tokio::select! { result=&mut execution => break result?, _=&mut changed => {} }
        };
        if let Some(rollback) = &receipt.rollback {
            self.store.save_rollback(task_id, rollback)?;
            self.update_task(task_id, |task| {
                task.rollback_available = true;
                task.rollback_action_id = Some(rollback.action_id);
                task.touch();
                Ok(())
            })
            .await?;
        }
        self.observe_and_verify(task_id, action_id, &proposal, &receipt)
            .await
    }

    fn validate_routine_binding(&self, task: &Task) -> CoreResult<()> {
        if let Some(expected) = &task.compiled_routine
            && self
                .store
                .compiled_routine(&task.request)?
                .is_none_or(|(id, _)| id != *expected)
        {
            return Err(CoreError::PermissionRequired("The saved routine was disabled, forgotten or changed. Review a fresh request before continuing.".into()));
        }
        Ok(())
    }

    async fn observe_and_verify(
        &self,
        task_id: Uuid,
        action_id: Uuid,
        proposal: &crate::domain::ActionProposal,
        receipt: &ExecutionReceipt,
    ) -> CoreResult<()> {
        self.update_action(task_id, action_id, |state| {
            state.status = ActionStatus::Verifying;
            Ok(())
        })
        .await?;
        let observation = self.observer.observe(proposal, receipt).await?;
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::ObservationReceived {
                action_id,
                summary: observation.summary.clone(),
            },
        ))?;
        if let Err(error) = self
            .verifier
            .verify(&proposal.expected_outcome, &observation)
        {
            self.publish(CoreEvent::new(
                Some(task_id),
                CoreEventKind::VerificationFailed {
                    action_id,
                    reason: error.to_string(),
                },
            ))?;
            return Err(error);
        }
        self.commit_observation(
            task_id,
            proposal,
            receipt,
            &observation,
            crate::transitions::VerificationMode::Execution,
        )
        .await
    }

    async fn commit_task_transition(
        &self,
        task_id: Uuid,
        transition: impl FnOnce(Task) -> CoreResult<(Task, CoreEvent)>,
    ) -> CoreResult<()> {
        let mut tasks = self.tasks.write().await;
        let current = tasks
            .get(&task_id)
            .cloned()
            .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))?;
        let (task, event) = transition(current)?;
        tasks.insert(task_id, task);
        drop(tasks);
        self.events.publish(event);
        Ok(())
    }

    fn procedure_checkpoint_for_action(
        &self,
        task_id: Uuid,
        proposal: &ActionProposal,
        allow_retired: bool,
    ) -> CoreResult<Option<crate::agency::ProcedureCheckpoint>> {
        let binding = match (
            proposal.metadata.get("procedure_id"),
            proposal.metadata.get("procedure_node_id"),
        ) {
            (None, None) => return Ok(None),
            (Some(procedure_id), Some(node_id)) => (procedure_id, node_id),
            _ => {
                return Err(CoreError::VerificationFailed(
                    "Procedure action has an incomplete identity binding".into(),
                ));
            }
        };
        let Some(checkpoint) = self.store.load_procedure_checkpoint(task_id)? else {
            if allow_retired && self.store.task_data_is_retired(task_id)? {
                return Ok(None);
            }
            return Err(CoreError::VerificationFailed(
                "Procedure action has no durable checkpoint".into(),
            ));
        };
        if checkpoint.task_id != task_id
            || checkpoint.procedure.id != *binding.0
            || !checkpoint
                .procedure
                .nodes
                .iter()
                .any(|node| node.id == *binding.1)
        {
            return Err(CoreError::VerificationFailed(
                "Procedure action does not match its durable checkpoint".into(),
            ));
        }
        Ok(Some(checkpoint))
    }

    fn current_procedure_context(
        &self,
        checkpoint: &crate::agency::ProcedureCheckpoint,
    ) -> CoreResult<(
        Vec<crate::world_model::SystemDescriptor>,
        Vec<crate::world_model::CapabilityAssessment>,
        BTreeSet<Uuid>,
    )> {
        let mut system_fingerprints = BTreeMap::<Uuid, String>::new();
        for node in &checkpoint.procedure.nodes {
            let crate::agency::ProcedureNodeKind::CapabilityCall {
                system_id,
                system_fingerprint,
                ..
            } = &node.kind
            else {
                continue;
            };
            if system_fingerprints
                .insert(*system_id, system_fingerprint.clone())
                .is_some_and(|prior| prior != *system_fingerprint)
            {
                return Err(CoreError::VerificationFailed(
                    "Procedure nodes disagree about the current target fingerprint".into(),
                ));
            }
        }
        if system_fingerprints.is_empty() {
            return Err(CoreError::ExecutorUnavailable(
                "Procedure contains no supported capability calls".into(),
            ));
        }
        let observed = self
            .store
            .observed_systems()?
            .into_iter()
            .map(|system| (system.id, system))
            .collect::<BTreeMap<_, _>>();
        let mut systems = Vec::with_capacity(system_fingerprints.len());
        let mut assessments_by_id = BTreeMap::new();
        let mut evidence_ids = checkpoint.runtime.verification_evidence_ids();
        for (system_id, fingerprint) in system_fingerprints {
            let system = observed.get(&system_id).ok_or_else(|| {
                CoreError::ExecutorUnavailable(
                    "Procedure target system is no longer present in discovery".into(),
                )
            })?;
            if system.fingerprint != fingerprint {
                return Err(CoreError::VerificationFailed(
                    "Procedure target interface changed during execution".into(),
                ));
            }
            evidence_ids.extend(
                self.store
                    .current_world_evidence_ids(system_id, &fingerprint)?,
            );
            systems.push(system.clone());
            for assessment in self.store.capability_assessments(system_id)? {
                if assessment.descriptor.system_fingerprint != fingerprint
                    || assessments_by_id
                        .insert(assessment.descriptor.id.clone(), assessment)
                        .is_some()
                {
                    return Err(CoreError::VerificationFailed(
                        "Current procedure capability identities are stale or ambiguous".into(),
                    ));
                }
            }
        }
        Ok((
            systems,
            assessments_by_id.into_values().collect(),
            evidence_ids,
        ))
    }

    fn compile_next_procedure_wave(
        &self,
        checkpoint: &crate::agency::ProcedureCheckpoint,
        task: &Task,
    ) -> CoreResult<Option<ActionGraph>> {
        if checkpoint.task_id != task.id {
            return Err(CoreError::VerificationFailed(
                "Procedure continuation checkpoint belongs to another task".into(),
            ));
        }
        let (systems, assessments, current_evidence_ids) =
            self.current_procedure_context(checkpoint)?;
        let mut runtime = checkpoint.runtime.clone();
        let advance = crate::agency::advance_procedure(
            &checkpoint.procedure,
            &assessments,
            &current_evidence_ids,
            &mut runtime,
            &BTreeMap::new(),
            4,
        )?;
        let mut existing_action_ids = BTreeMap::new();
        for (action_id, action) in &task.actions {
            let procedure_id = action.proposal.metadata.get("procedure_id");
            let node_id = action.proposal.metadata.get("procedure_node_id");
            if procedure_id != Some(&checkpoint.procedure.id) {
                return Err(CoreError::VerificationFailed(
                    "Procedure task contains an action from another execution route".into(),
                ));
            }
            let node_id = node_id.ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Procedure task action is missing its node identity".into(),
                )
            })?;
            if existing_action_ids
                .insert(node_id.clone(), *action_id)
                .is_some()
            {
                return Err(CoreError::VerificationFailed(
                    "Procedure task contains duplicate node actions".into(),
                ));
            }
        }
        let selected_node_ids = advance
            .proposal_wave
            .iter()
            .map(|proposal| proposal.node_id.clone())
            .filter(|node_id| !existing_action_ids.contains_key(node_id))
            .collect::<Vec<_>>();
        if selected_node_ids.is_empty() {
            return Ok(None);
        }
        let graph = compile_application_control_procedure(
            &checkpoint.procedure,
            &task.request,
            task.id,
            ProcedureCompilationContext {
                systems: &systems,
                assessments: &assessments,
                current_evidence_ids: &current_evidence_ids,
                runtime: &runtime,
                selected_node_ids: &selected_node_ids,
                existing_action_ids: &existing_action_ids,
            },
        )?;
        Ok(Some(graph))
    }

    fn procedure_completion_satisfied(&self, task: &Task) -> CoreResult<bool> {
        let procedure_id = task
            .actions
            .values()
            .find_map(|action| action.proposal.metadata.get("procedure_id"));
        let Some(procedure_id) = procedure_id else {
            return Ok(true);
        };
        if task
            .actions
            .values()
            .any(|action| action.proposal.metadata.get("procedure_id") != Some(procedure_id))
        {
            return Ok(false);
        }
        let Some(checkpoint) = self.store.load_procedure_checkpoint(task.id)? else {
            return Ok(false);
        };
        if checkpoint.procedure.id != *procedure_id {
            return Ok(false);
        }
        checkpoint
            .runtime
            .completion_satisfied(&checkpoint.procedure)
    }

    fn prepare_procedure_dispatch_checkpoint(
        &self,
        task_id: Uuid,
        proposal: &ActionProposal,
    ) -> CoreResult<Option<crate::agency::ProcedureCheckpoint>> {
        let Some(mut checkpoint) =
            self.procedure_checkpoint_for_action(task_id, proposal, false)?
        else {
            return Ok(None);
        };
        let node_id = proposal
            .metadata
            .get("procedure_node_id")
            .expect("procedure binding was checked");
        let (assessments, current_evidence_ids) =
            self.current_procedure_dispatch_evidence(&checkpoint)?;
        checkpoint.runtime.record_dispatched(
            &checkpoint.procedure,
            &assessments,
            &current_evidence_ids,
            node_id,
            proposal.id,
        )?;
        Ok(Some(checkpoint))
    }

    fn prepare_procedure_dispatch_wave_checkpoint(
        &self,
        task_id: Uuid,
        proposals: &[ActionProposal],
    ) -> CoreResult<crate::agency::ProcedureCheckpoint> {
        if !(2..=16).contains(&proposals.len()) {
            return Err(CoreError::InvalidAction(
                "Procedure dispatch wave must contain two to sixteen actions".into(),
            ));
        }
        let first = proposals.first().expect("wave length was validated");
        let mut checkpoint = self
            .procedure_checkpoint_for_action(task_id, first, false)?
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Stream dispatch has no durable procedure checkpoint".into(),
                )
            })?;
        let mut receipts = BTreeMap::new();
        for proposal in proposals {
            if proposal.task_id != task_id
                || proposal.metadata.get("procedure_id") != Some(&checkpoint.procedure.id)
                || proposal
                    .metadata
                    .get("procedure_stream_node")
                    .map(String::as_str)
                    != Some("true")
            {
                return Err(CoreError::VerificationFailed(
                    "Procedure dispatch wave mixes task, procedure, or stream identities".into(),
                ));
            }
            let node_id = proposal.metadata.get("procedure_node_id").ok_or_else(|| {
                CoreError::VerificationFailed("Procedure wave action has no node identity".into())
            })?;
            if receipts.insert(node_id.clone(), proposal.id).is_some() {
                return Err(CoreError::VerificationFailed(
                    "Procedure dispatch wave repeats a node identity".into(),
                ));
            }
        }
        let (assessments, current_evidence_ids) =
            self.current_procedure_dispatch_evidence(&checkpoint)?;
        checkpoint.runtime.record_dispatched_wave(
            &checkpoint.procedure,
            &assessments,
            &current_evidence_ids,
            receipts,
        )?;
        Ok(checkpoint)
    }

    fn current_procedure_dispatch_evidence(
        &self,
        checkpoint: &crate::agency::ProcedureCheckpoint,
    ) -> CoreResult<(
        Vec<crate::world_model::CapabilityAssessment>,
        BTreeSet<Uuid>,
    )> {
        let mut systems = BTreeMap::<Uuid, String>::new();
        for node in &checkpoint.procedure.nodes {
            let crate::agency::ProcedureNodeKind::CapabilityCall {
                system_id,
                system_fingerprint,
                ..
            } = &node.kind
            else {
                continue;
            };
            if systems
                .insert(*system_id, system_fingerprint.clone())
                .is_some_and(|prior| prior != *system_fingerprint)
            {
                return Err(CoreError::VerificationFailed(
                    "Procedure nodes disagree about the current target fingerprint".into(),
                ));
            }
        }
        let mut assessments_by_id = BTreeMap::new();
        let mut current_evidence_ids = checkpoint.runtime.verification_evidence_ids();
        for (system_id, fingerprint) in systems {
            current_evidence_ids.extend(
                self.store
                    .current_world_evidence_ids(system_id, &fingerprint)?,
            );
            for assessment in self.store.capability_assessments(system_id)? {
                if assessment.descriptor.system_fingerprint != fingerprint
                    || assessments_by_id
                        .insert(assessment.descriptor.id.clone(), assessment)
                        .is_some()
                {
                    return Err(CoreError::VerificationFailed(
                        "Current procedure capability identities are stale or ambiguous".into(),
                    ));
                }
            }
        }
        let assessments = assessments_by_id.into_values().collect::<Vec<_>>();
        Ok((assessments, current_evidence_ids))
    }

    async fn commit_dispatch(
        &self,
        task_id: Uuid,
        prepared: &crate::contracts::PreparedAction,
        grant: Option<&crate::capability::CapabilityGrant>,
        implementation: &str,
    ) -> CoreResult<()> {
        if prepared
            .intent
            .proposal
            .metadata
            .get("procedure_stream_node")
            .is_some_and(|value| value == "true")
        {
            let wave = self
                .procedure_dispatch_wave_for_action(prepared.intent.proposal.id)
                .ok_or_else(|| {
                    CoreError::PermissionRequired(
                        "Stream action is not part of a registered dispatch cohort".into(),
                    )
                })?;
            let grant = grant.ok_or_else(|| {
                CoreError::CapabilityRejected(
                    "Stream dispatch requires its own one-use capability".into(),
                )
            })?;
            return wave
                .submit(
                    self,
                    prepared.intent.proposal.id,
                    ProcedureDispatchSubmission {
                        prepared: prepared.clone(),
                        grant: grant.clone(),
                        implementation: implementation.to_owned(),
                    },
                )
                .await;
        }
        loop {
            self.wait_until_runnable(task_id).await?;
            let mut tasks = self.tasks.write().await;
            let current = tasks
                .get(&task_id)
                .cloned()
                .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))?;
            if current.status == TaskStatus::Paused {
                continue;
            }
            let checkpoint =
                self.prepare_procedure_dispatch_checkpoint(task_id, &prepared.intent.proposal)?;
            let (task, event) = self.store.commit_dispatched_action_with_procedure(
                current,
                prepared,
                grant,
                implementation,
                checkpoint,
            )?;
            tasks.insert(task_id, task);
            drop(tasks);
            self.events.publish(event);
            return Ok(());
        }
    }

    async fn commit_procedure_dispatch_wave(
        &self,
        task_id: Uuid,
        submissions: &BTreeMap<Uuid, ProcedureDispatchSubmission>,
    ) -> CoreResult<()> {
        if !(2..=16).contains(&submissions.len()) {
            return Err(CoreError::InvalidAction(
                "Procedure dispatch wave has an invalid number of actions".into(),
            ));
        }
        let proposals = submissions
            .values()
            .map(|submission| submission.prepared.intent.proposal.clone())
            .collect::<Vec<_>>();
        let dispatch_intents = submissions
            .values()
            .map(|submission| {
                (
                    &submission.prepared,
                    &submission.grant,
                    submission.implementation.as_str(),
                )
            })
            .collect::<Vec<_>>();
        loop {
            self.wait_until_runnable(task_id).await?;
            let mut tasks = self.tasks.write().await;
            let current = tasks
                .get(&task_id)
                .cloned()
                .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))?;
            if current.status == TaskStatus::Paused {
                continue;
            }
            if self.runtime.is_stopped(task_id) {
                return Err(CoreError::Cancelled);
            }
            let checkpoint =
                self.prepare_procedure_dispatch_wave_checkpoint(task_id, &proposals)?;
            let (task, events) =
                self.store
                    .commit_dispatched_action_wave(current, &dispatch_intents, checkpoint)?;
            tasks.insert(task_id, task);
            drop(tasks);
            for event in events {
                self.events.publish(event);
            }
            return Ok(());
        }
    }

    async fn commit_observation(
        &self,
        task_id: Uuid,
        proposal: &crate::domain::ActionProposal,
        receipt: &ExecutionReceipt,
        observation: &crate::observation::Observation,
        mode: crate::transitions::VerificationMode,
    ) -> CoreResult<()> {
        let mut tasks = self.tasks.write().await;
        let current = tasks
            .get(&task_id)
            .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))?;
        let mut change = crate::transitions::VerifiedAction::from_observation(
            current,
            proposal,
            receipt,
            observation,
            mode,
        )?;
        let mut checkpoint = self.procedure_checkpoint_for_action(task_id, proposal, true)?;
        if let Some(checkpoint) = checkpoint.as_mut() {
            change.advance_procedure_checkpoint(checkpoint)?;
            if !checkpoint
                .runtime
                .completion_satisfied(&checkpoint.procedure)?
                && let Ok(Some(graph)) = self.compile_next_procedure_wave(checkpoint, &change.task)
            {
                change
                    .task
                    .append_plan_with_exact_dependencies(graph)
                    .map_err(CoreError::InvalidAction)?;
                validate_procedure_action_binding(&change.task, &checkpoint.procedure)?;
            }
        }
        let (task, event) = self
            .store
            .commit_verified_action_with_procedure(change, checkpoint)?;
        tasks.insert(task_id, task);
        drop(tasks);
        // Publish only after both the broker receipt and procedure projection
        // commit, so observers never see a partial success transition.
        self.events.publish(event);
        self.store.checkpoint_audit(self.secret_store.as_ref())
    }

    async fn open_decision(&self, record: &crate::decisions::DecisionRecord) -> CoreResult<()> {
        if self.runtime.is_stopped(record.task_id()) {
            return Err(CoreError::Cancelled);
        }
        let mut tasks = self.tasks.write().await;
        let mut task = tasks
            .get(&record.task_id())
            .cloned()
            .ok_or_else(|| CoreError::TaskNotFound(record.task_id().to_string()))?;
        if !task.action_is_current(record.action_id()) {
            return Err(CoreError::Cancelled);
        }
        if !task.status.is_active() {
            return Err(CoreError::Cancelled);
        }
        task.status = record.task_status();
        if matches!(record, crate::decisions::DecisionRecord::Approval(_))
            && let Some(action) = task.actions.get_mut(&record.action_id())
        {
            action.status = ActionStatus::WaitingForApproval;
        }
        task.touch();
        let event = record.opened();
        self.store.open_decision(&mut task, record, &event)?;
        tasks.insert(task.id, task);
        self.events.publish(event);
        Ok(())
    }

    fn close_decision(
        &self,
        record: &crate::decisions::DecisionRecord,
        resolution: crate::decisions::DecisionResolution,
    ) -> CoreResult<()> {
        if let Some(event) = self.store.resolve_decision(record, &resolution)? {
            self.events.publish(event);
        }
        Ok(())
    }

    async fn await_approval(
        &self,
        proposal: &crate::domain::ActionProposal,
        risk: RiskLevel,
        explanation: String,
        digest: String,
    ) -> CoreResult<()> {
        use crate::decisions::{DecisionRecord, DecisionResolution};
        let approval_id = Uuid::new_v4();
        let expires_at = Utc::now() + ChronoDuration::seconds(APPROVAL_TIMEOUT.as_secs() as i64);
        let requires_native_authentication = risk >= RiskLevel::Privileged;
        let record = crate::contracts::ApprovalRecord {
            approval_id,
            task_id: proposal.task_id,
            action_id: proposal.id,
            digest: digest.clone(),
            explanation,
            resource: proposal.target_resource.clone(),
            risk,
            expires_at,
            reversible: proposal.action.reversible_hint(),
            requires_native_authentication,
        };
        let (sender, receiver) = oneshot::channel();
        self.pending_approvals.lock().await.insert(
            approval_id,
            PendingApproval {
                task_id: proposal.task_id,
                action_id: proposal.id,
                digest,
                requires_native_authentication,
                expires_at,
                record: record.clone(),
                sender,
            },
        );
        let decision = DecisionRecord::Approval(record);
        if let Err(error) = self.open_decision(&decision).await {
            self.pending_approvals.lock().await.remove(&approval_id);
            return Err(error);
        }
        let mut wave_completion = self
            .procedure_dispatch_wave_for_action(proposal.id)
            .map(|wave| wave.completion_receiver());
        let waited = timeout(APPROVAL_TIMEOUT, async {
            if let Some(wave_completion) = wave_completion.as_mut() {
                tokio::select! {
                    resolution = receiver => ApprovalWait::Resolution(resolution),
                    _ = wait_for_procedure_dispatch_wave(wave_completion) => {
                        ApprovalWait::ProcedureWaveFailed
                    }
                }
            } else {
                ApprovalWait::Resolution(receiver.await)
            }
        })
        .await;
        self.pending_approvals.lock().await.remove(&approval_id);
        if self.runtime.action_retired(proposal.task_id, proposal.id) {
            return Err(CoreError::Cancelled);
        }
        let resolution = match waited {
            Ok(ApprovalWait::Resolution(Ok(value))) => value,
            Ok(ApprovalWait::ProcedureWaveFailed) => {
                self.close_decision(&decision, DecisionResolution::Cancelled)?;
                return Err(CoreError::PermissionRequired(
                    "A connected stream action could not be prepared; review and resume the procedure".into(),
                ));
            }
            Ok(ApprovalWait::Resolution(Err(_))) => {
                self.close_decision(&decision, DecisionResolution::Cancelled)?;
                self.capabilities.revoke_task(proposal.task_id).await;
                return Err(CoreError::ApprovalRejected("Approval was cancelled".into()));
            }
            Err(_) => {
                self.close_decision(&decision, DecisionResolution::Expired)?;
                self.capabilities.revoke_task(proposal.task_id).await;
                self.set_task_status(
                    proposal.task_id,
                    TaskStatus::Interrupted,
                    "Approval expired. Resume to review fresh authorization.",
                )
                .await?;
                return Err(CoreError::PermissionRequired(
                    "Resume this task to review a fresh approval".into(),
                ));
            }
        };
        let approved = matches!(resolution, ApprovalResolution::Approved { .. });
        self.publish(CoreEvent::new(
            Some(proposal.task_id),
            CoreEventKind::ApprovalResolved {
                action_id: proposal.id,
                approved,
            },
        ))?;
        if !approved {
            return Err(CoreError::ApprovalRejected("user denied the action".into()));
        }
        self.update_task(proposal.task_id, |task| {
            if task.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            task.status = TaskStatus::Running;
            if let Some(action) = task.actions.get_mut(&proposal.id) {
                action.status = ActionStatus::Compiling;
            }
            task.touch();
            Ok(())
        })
        .await
    }

    async fn await_question(
        &self,
        proposal: &crate::domain::ActionProposal,
        question: String,
    ) -> CoreResult<ExecutionReceipt> {
        use crate::decisions::{DecisionRecord, DecisionResolution, QuestionRecord};
        let question_id = Uuid::new_v4();
        let expires_at = Utc::now() + ChronoDuration::seconds(QUESTION_TIMEOUT.as_secs() as i64);
        let record = QuestionRecord {
            question_id,
            task_id: proposal.task_id,
            action_id: proposal.id,
            question,
            expires_at,
        };
        let (sender, receiver) = oneshot::channel();
        self.pending_questions.lock().await.insert(
            question_id,
            PendingQuestion {
                task_id: proposal.task_id,
                action_id: proposal.id,
                record: record.clone(),
                sender,
            },
        );
        let decision = DecisionRecord::Question(record);
        if let Err(error) = self.open_decision(&decision).await {
            self.pending_questions.lock().await.remove(&question_id);
            return Err(error);
        }
        let waited = timeout(QUESTION_TIMEOUT, receiver).await;
        self.pending_questions.lock().await.remove(&question_id);
        let answer = match waited {
            Ok(Ok(answer)) => answer,
            other => {
                self.close_decision(
                    &decision,
                    if other.is_err() {
                        DecisionResolution::Expired
                    } else {
                        DecisionResolution::Cancelled
                    },
                )?;
                if other.is_err() {
                    self.set_task_status(
                        proposal.task_id,
                        TaskStatus::Interrupted,
                        "The question expired. Resume to answer a fresh question.",
                    )
                    .await?;
                    return Err(CoreError::Timeout("question expired".into()));
                }
                return Err(CoreError::Cancelled);
            }
        };
        self.update_task(proposal.task_id, |task| {
            if task.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            task.status = TaskStatus::Running;
            task.touch();
            Ok(())
        })
        .await?;
        Ok(ExecutionReceipt {
            executor: "native-user-interaction".into(),
            summary: "received user response".into(),
            transient_data: json!({ "user_answered": true, "answer": answer }),
            rollback: None,
        })
    }

    async fn wait_until_runnable(&self, task_id: Uuid) -> CoreResult<()> {
        loop {
            if self.runtime.is_stopped(task_id) {
                return Err(CoreError::Cancelled);
            }
            let changed = self.control_changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.runtime.is_held(task_id) && !self.runtime.is_stopped(task_id) {
                changed.await;
                continue;
            }
            match self.get_task(task_id).await?.status {
                TaskStatus::Cancelled => return Err(CoreError::Cancelled),
                TaskStatus::Interrupted => {
                    return Err(CoreError::PermissionRequired(
                        "This task needs recovery review".into(),
                    ));
                }
                TaskStatus::Paused => changed.await,
                _ => return Ok(()),
            }
        }
    }

    async fn succeed_task(&self, task_id: Uuid) -> CoreResult<bool> {
        self.wait_until_runnable(task_id).await?;
        let _submission = self.submission_lock.lock().await;
        let task = self.get_task(task_id).await?;
        if !task.is_complete() || self.runtime.is_held(task_id) {
            return Ok(false);
        }
        let answer =
            task.tool_results
                .iter()
                .filter(|result| task.action_is_current(result.action_id))
                .map(|r| {
                    if task.actions.get(&r.action_id).is_some_and(|action| {
                        action.proposal.metadata.contains_key("intent_compiler")
                    }) {
                        if let Some(text) = r.output.get("text").and_then(serde_json::Value::as_str)
                        {
                            return format!("{}\n\n{}", r.summary, text);
                        }
                        if r.tool == "list_directory"
                            && let Ok(page) = serde_json::from_value::<
                                crate::execution::directory::DirectoryPage,
                            >(r.output.clone())
                        {
                            return format!(
                                "{}\n\n{}",
                                r.summary,
                                page.entries
                                    .iter()
                                    .map(|entry| entry.name.as_str())
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            );
                        }
                    }
                    r.summary.clone()
                })
                .collect::<Vec<_>>()
                .join("\n");
        self.finalize_task(task_id, crate::finalization::Finish::Answer(answer))
            .await?;
        Ok(true)
    }

    async fn fail_task(&self, task_id: Uuid, error: String) -> CoreResult<()> {
        self.finalize_task(task_id, crate::finalization::Finish::Failure(error))
            .await
            .map(|_| ())
    }

    async fn set_task_status(
        &self,
        task_id: Uuid,
        status: TaskStatus,
        summary: &str,
    ) -> CoreResult<()> {
        if status == TaskStatus::Interrupted {
            return self
                .finalize_task(
                    task_id,
                    crate::finalization::Finish::Interrupted {
                        summary: summary.into(),
                        budget_exhausted: false,
                    },
                )
                .await
                .map(|_| ());
        }
        self.update_task(task_id, |task| {
            if task.status == TaskStatus::Cancelled && status != TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            task.status = status;
            task.touch();
            Ok(())
        })
        .await?;
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::TaskStatusChanged {
                status,
                summary: summary.into(),
            },
        ))
    }

    async fn get_task(&self, task_id: Uuid) -> CoreResult<Task> {
        let mut task = self
            .tasks
            .read()
            .await
            .get(&task_id)
            .cloned()
            .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))?;
        self.runtime.project_pending_stop(&mut task);
        Ok(task)
    }

    async fn update_task(
        &self,
        task_id: Uuid,
        operation: impl FnOnce(&mut Task) -> CoreResult<()>,
    ) -> CoreResult<()> {
        // Persist in the same order as in-memory transitions. A concurrent
        // cancellation must never be overwritten by an older dispatch snapshot.
        let mut tasks = self.tasks.write().await;
        let mut task = tasks
            .get(&task_id)
            .cloned()
            .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))?;
        operation(&mut task)?;
        self.store.save_task(&mut task)?;
        tasks.insert(task_id, task);
        Ok(())
    }

    async fn update_action(
        &self,
        task_id: Uuid,
        action_id: Uuid,
        operation: impl FnOnce(&mut crate::domain::ActionState) -> CoreResult<()>,
    ) -> CoreResult<()> {
        self.update_task(task_id, |task| {
            if !task.action_is_current(action_id) {
                return Err(CoreError::Cancelled);
            }
            let action = task
                .actions
                .get_mut(&action_id)
                .ok_or_else(|| CoreError::InvalidAction("action not found".into()))?;
            operation(action)?;
            task.touch();
            Ok(())
        })
        .await
    }

    pub fn knowledge_snapshot(&self, conversation: Option<Uuid>) -> CoreResult<KnowledgeSnapshot> {
        let conversations = self.store.conversations()?;
        let selected = conversation.or_else(|| conversations.first().map(|c| c.id));
        Ok(KnowledgeSnapshot {
            conversations,
            messages: selected
                .map(|id| self.store.messages(id, 100))
                .transpose()?
                .unwrap_or_default(),
            memories: self.store.memories(None, false, 500)?,
            memory_enabled: self.store.memory_enabled()?,
        })
    }

    async fn cancel_context_readers(self: &Arc<Self>, except: Option<Uuid>) -> CoreResult<()> {
        let active = self
            .tasks
            .read()
            .await
            .values()
            .filter(|t| Some(t.id) != except && t.status.is_active())
            .map(|t| t.id)
            .collect::<Vec<_>>();
        for task_id in &active {
            self.runtime.stop(*task_id);
        }
        self.control_changed.notify_waiters();
        for task_id in active {
            self.persist_stopped_run(task_id).await?;
            self.pending_approvals
                .lock()
                .await
                .retain(|_, p| p.task_id != task_id);
            self.pending_questions
                .lock()
                .await
                .retain(|_, p| p.task_id != task_id);
        }
        self.control_changed.notify_waiters();
        Ok(())
    }

    pub async fn knowledge_command(
        self: &Arc<Self>,
        command: sage_protocol::sage::ipc::v2::KnowledgeCommand,
    ) -> CoreResult<KnowledgeSnapshot> {
        let id = || {
            Uuid::parse_str(&command.id)
                .map_err(|_| CoreError::InvalidAction("Invalid record ID".into()))
        };
        if matches!(command.operation.as_str(), "delete" | "edit" | "disable")
            || (command.operation == "configure" && !command.enabled)
        {
            self.cancel_context_readers(None).await?;
        }
        match command.operation.as_str() {
            "list" => {}
            "remember" => {
                self.store.remember(&command.content, Uuid::new_v4())?;
            }
            "edit" => {
                let mut record = self
                    .store
                    .memories(None, false, 500)?
                    .into_iter()
                    .find(|m| Some(m.id) == id().ok())
                    .ok_or_else(|| CoreError::InvalidAction("Memory not found".into()))?;
                record.content = command.content.clone();
                self.store.save_memory(record)?;
            }
            "delete" => self.store.delete_memory(id()?)?,
            "enable" | "disable" => self
                .store
                .set_memory_record_enabled(id()?, command.operation == "enable")?,
            "configure" => {
                self.store
                    .save_setting("memory.enabled", &command.enabled)?;
            }
            "conversation" => {
                if command.archived
                    && self
                        .tasks
                        .read()
                        .await
                        .values()
                        .any(|t| t.conversation_id == id().ok() && t.status.is_active())
                {
                    return Err(CoreError::InvalidAction(
                        "Stop active work before deleting this conversation.".into(),
                    ));
                }
                self.store.update_conversation(
                    id()?,
                    &command.content,
                    command.pinned,
                    command.archived,
                )?;
            }
            _ => {
                return Err(CoreError::InvalidAction(
                    "Unknown knowledge operation".into(),
                ));
            }
        }
        let conversation = if command.conversation_id.is_empty() {
            None
        } else {
            Some(
                Uuid::parse_str(&command.conversation_id)
                    .map_err(|_| CoreError::InvalidAction("Invalid conversation ID".into()))?,
            )
        };
        self.knowledge_snapshot(conversation)
    }

    pub fn workflow_snapshot(&self) -> CoreResult<serde_json::Value> {
        let routine_families = self.store.routine_families()?;
        let skills = self
            .store
            .skills()?
            .iter()
            .map(|skill| {
                let mut value = serde_json::to_value(skill)?;
                let source_paused = !skill.synthesized_from.is_empty()
                    && !self.store.skill_sources_current(skill)?;
                value["source_paused"] = json!(source_paused);
                value["enabled"] = json!(skill.is_reviewed() && !source_paused);
                value["review_digest_candidate"] = json!(skill.digest()?);
                value["preview"] = json!(skill.preview()?);
                Ok(value)
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let mut workflow_paused = BTreeMap::new();
        let workflows = self
            .store
            .workflows()?
            .iter()
            .map(|workflow| {
                let paused = workflow.skill_ids.iter().try_fold(false, |paused, id| {
                    let Some(skill) = self.store.skill(*id)? else {
                        return Ok::<bool, CoreError>(true);
                    };
                    Ok(paused
                        || (!skill.synthesized_from.is_empty()
                            && !self.store.skill_sources_current(&skill)?))
                })?;
                workflow_paused.insert(workflow.id, paused);
                let mut value = serde_json::to_value(workflow)?;
                value["source_paused"] = json!(paused);
                value["enabled"] = json!(workflow.enabled && !paused);
                Ok(value)
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let schedules = self
            .store
            .schedules()?
            .into_iter()
            .map(|schedule| {
                let paused = schedule
                    .workflow_id
                    .and_then(|id| workflow_paused.get(&id).copied())
                    .unwrap_or(false);
                let mut value = serde_json::to_value(schedule)?;
                value["source_paused"] = json!(paused);
                if paused {
                    value["enabled"] = json!(false);
                }
                Ok(value)
            })
            .collect::<CoreResult<Vec<_>>>()?;
        Ok(
            json!({"skills":skills,"workflows":workflows,"schedules":schedules,"routine_learning_enabled":self.store.routine_learning_enabled()?,"routines":self.store.routines()?,"routine_families":routine_families}),
        )
    }

    /// Resolve a learned application control from the current, signed
    /// foreground interface. The model may select a stored capability id and
    /// supply its typed value, but the stored probe receipt and current OS
    /// observation decide whether this sealed executor is still usable.
    async fn prepare_application_control(
        &self,
        proposal: &mut crate::domain::ActionProposal,
        expected_preparation: Option<&crate::domain::ActionProposal>,
    ) -> CoreResult<()> {
        let Action::SetApplicationControl {
            application,
            system_id,
            system_fingerprint,
            capability_id,
            control_id,
            value,
        } = proposal.action.clone()
        else {
            return Ok(());
        };
        crate::features::validate(&proposal.action)?;

        let store = self.store.clone();
        let (system, assessment) = tokio::task::spawn_blocking(move || {
            let system = store
                .observed_systems()?
                .into_iter()
                .find(|system| system.id == system_id)
                .ok_or_else(|| {
                    CoreError::PermissionRequired(
                        "The learned application is no longer in Sage's current discovery graph"
                            .into(),
                    )
                })?;
            let assessment = store
                .capability_assessments(system_id)?
                .into_iter()
                .find(|assessment| assessment.descriptor.id == capability_id)
                .ok_or_else(|| {
                    CoreError::PermissionRequired(
                        "The selected application capability is not a stored Sage discovery".into(),
                    )
                })?;
            Ok::<_, CoreError>((system, assessment))
        })
        .await
        .map_err(|_| CoreError::Storage("Learned-control lookup worker exited".into()))??;
        if system.kind != crate::world_model::SystemKind::Application
            || system.key != application
            || system.fingerprint != system_fingerprint
        {
            return Err(CoreError::VerificationFailed(
                "The learned control belongs to a different application identity".into(),
            ));
        }
        let mut descriptor = assessment.descriptor;
        if assessment.evidence_state
            != crate::world_model::CapabilityEvidenceState::ReversiblyExperimented
            || descriptor.executor_id.as_deref() != Some("set_application_control")
            || descriptor.system_id != system_id
            || descriptor.system_fingerprint != system_fingerprint
            || descriptor.interface_control_id.as_deref() != Some(control_id.as_str())
            || descriptor.interface_probe_kind.is_none()
            || descriptor.input_ports.len() != 1
            || descriptor.input_ports[0].name != "value"
            || descriptor.output_ports.len() != 1
            || descriptor.output_ports[0].name != "observed_value"
        {
            return Err(CoreError::PermissionRequired(
                "This control has not completed a current, restored Sage learning probe".into(),
            ));
        }
        let (expected_role, expected_port) = match descriptor.interface_probe_kind {
            Some(crate::world_model::ProbeKind::RestoreSliderValue) => {
                ("slider", crate::world_model::PortType::Number)
            }
            Some(crate::world_model::ProbeKind::RestoreToggleState) => {
                ("toggle", crate::world_model::PortType::Boolean)
            }
            None => unreachable!("checked above"),
        };
        if descriptor.input_ports[0].value_type != expected_port
            || descriptor.output_ports[0].value_type != expected_port
        {
            return Err(CoreError::VerificationFailed(
                "The learned control's typed ports no longer match its sealed executor".into(),
            ));
        }

        let session = self.adapters.session_id("native").await.ok_or_else(|| {
            CoreError::ExecutorUnavailable(
                "Connect Sage's Mac adapter before using a learned application control".into(),
            )
        })?;
        let response = self
            .adapters
            .request_in_session("native", &session, "discover_interface", json!({}))
            .await?;
        if self.adapters.session_id("native").await.as_deref() != Some(&session) {
            return Err(CoreError::ExecutorUnavailable(
                "The native adapter changed during learned-control preparation".into(),
            ));
        }
        let response: ApplicationDiscoveryResponse = serde_json::from_value(response)?;
        response.application_target.validate()?;
        if response.observed_process_id == 0
            || response.bundle_identifier != application
            || response.application_target.identifier != application
            || response.application_target.code_digest != system_fingerprint
        {
            return Err(CoreError::VerificationFailed(
                "The current foreground application differs from the learned signed target".into(),
            ));
        }
        let controls = prepare_application_controls(response.controls)?;
        let interface_fingerprint = controls.interface_fingerprint.clone();
        let matching = controls
            .controls
            .iter()
            .filter(|control| control.semantic.id == control_id)
            .collect::<Vec<_>>();
        let [control] = matching.as_slice() else {
            return Err(CoreError::VerificationFailed(
                "The learned control is missing or ambiguous in the current foreground interface"
                    .into(),
            ));
        };
        let original_label = descriptor.label.strip_prefix("Set ").ok_or_else(|| {
            CoreError::VerificationFailed("Stored control label is invalid".into())
        })?;
        if !control.semantic.enabled
            || (control.semantic.role != expected_role
                && !(expected_role == "toggle"
                    && matches!(control.semantic.role.as_str(), "checkbox" | "switch")))
            || control.semantic.label != original_label
        {
            return Err(CoreError::VerificationFailed(
                "The learned control's label, role, or enabled state changed".into(),
            ));
        }
        match (&value, descriptor.interface_probe_kind) {
            (
                crate::domain::ApplicationControlValue::Boolean(_),
                Some(crate::world_model::ProbeKind::RestoreToggleState),
            ) => {}
            (
                crate::domain::ApplicationControlValue::Number(requested),
                Some(crate::world_model::ProbeKind::RestoreSliderValue),
            ) => {
                let (Some(minimum), Some(maximum), Some(step)) =
                    (control.minimum, control.maximum, control.step)
                else {
                    return Err(CoreError::VerificationFailed(
                        "The current slider does not expose a bounded value range".into(),
                    ));
                };
                let grid = (*requested - minimum) / step;
                if !requested.is_finite()
                    || !minimum.is_finite()
                    || !maximum.is_finite()
                    || !step.is_finite()
                    || step <= 0.0
                    || step > 1.0
                    || minimum >= maximum
                    || *requested < minimum
                    || *requested > maximum
                    || !grid.is_finite()
                    || (grid - grid.round()).abs() > 1e-7 * grid.abs().max(1.0)
                {
                    return Err(CoreError::InvalidAction(
                        "The requested slider value is outside its current range or step grid"
                            .into(),
                    ));
                }
            }
            _ => {
                return Err(CoreError::InvalidAction(
                    "The requested value type does not match the learned control".into(),
                ));
            }
        }

        let now = Utc::now();
        let mut facts = vec![
            crate::world_model::ObservedFact {
                name: "application.accessibility_available".into(),
                subject: None,
                value: crate::world_model::FactValue::Boolean(response.accessibility_available),
            },
            crate::world_model::ObservedFact {
                name: "application.bundle_identifier".into(),
                subject: None,
                value: crate::world_model::FactValue::Identifier(application.clone()),
            },
        ];
        facts.extend(controls.facts.iter().cloned());
        facts.push(crate::world_model::ObservedFact {
            name: "application.interface_fingerprint".into(),
            subject: None,
            value: crate::world_model::FactValue::Identifier(interface_fingerprint.clone()),
        });
        facts
            .sort_by(|left, right| (&left.name, &left.subject).cmp(&(&right.name, &right.subject)));
        let observation = crate::world_model::ObservationEnvelope {
            id: Uuid::new_v4(),
            system_id,
            session_id: None,
            worker_session: None,
            system_fingerprint: system_fingerprint.clone(),
            origin: crate::world_model::EvidenceOrigin::OperatingSystem,
            privacy: crate::contracts::Sensitivity::Private,
            observed_at: now,
            facts,
        };
        let candidates = crate::world_model::safe_probe_candidates(&observation);
        if candidates.get(&control_id) != descriptor.interface_probe_kind.as_ref() {
            return Err(CoreError::PermissionRequired(
                "The current control no longer qualifies for its reversible, low-risk evidence class".into(),
            ));
        }
        let anchor = serde_json::to_string(&control.semantic)?;
        let target = response.application_target;
        let expected_metadata = BTreeMap::from([
            ("application_target".into(), serde_json::to_string(&target)?),
            (
                "application_interface_fingerprint".into(),
                interface_fingerprint,
            ),
            ("application_control_anchor".into(), anchor),
            ("capability_label".into(), descriptor.label.clone()),
        ]);
        if let Some(previous) = expected_preparation
            && expected_metadata
                .iter()
                .any(|(key, value)| previous.metadata.get(key) != Some(value))
        {
            return Err(CoreError::ApprovalRejected(
                "The signed app, learned control, or interface changed while approval was pending"
                    .into(),
            ));
        }
        descriptor.preconditions.observed_state_fact_ids = vec![observation.id];
        descriptor.evidence_ids = vec![observation.id];
        descriptor.updated_at = now;
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || {
            store.record_world_observation(&observation)?;
            store.record_capability_candidate(&descriptor)
        })
        .await
        .map_err(|_| CoreError::Storage("Learned-control evidence worker exited".into()))??;
        if expected_preparation.is_none() {
            proposal.metadata.extend(expected_metadata);
        }
        Ok(())
    }

    pub async fn world_model_command(
        self: &Arc<Self>,
        command: sage_protocol::sage::ipc::v2::WorldModelCommand,
    ) -> CoreResult<serde_json::Value> {
        use crate::world_model::LearningSessionState;

        let parse_id = || {
            Uuid::parse_str(&command.id)
                .map_err(|_| CoreError::InvalidAction("Invalid world-model record ID".into()))
        };
        match command.operation.as_str() {
            "discover_current_application" => {
                let session = self.adapters.session_id("native").await.ok_or_else(|| {
                    CoreError::ExecutorUnavailable(
                        "A connected native adapter is required for application discovery".into(),
                    )
                })?;
                let response = self
                    .adapters
                    .request_in_session("native", &session, "discover_interface", json!({}))
                    .await?;
                if self.adapters.session_id("native").await.as_deref() != Some(&session) {
                    return Err(CoreError::ExecutorUnavailable(
                        "The native adapter changed during application discovery".into(),
                    ));
                }
                let response: ApplicationDiscoveryResponse = serde_json::from_value(response)?;
                response.application_target.validate()?;
                if response.observed_process_id == 0
                    || response.bundle_identifier != response.application_target.identifier
                    || response.application_name.len() > 256
                    || response.active_window.len() > 256
                {
                    return Err(CoreError::VerificationFailed(
                        "Native discovery returned an invalid or oversized interface identity"
                            .into(),
                    ));
                }
                let controls = prepare_application_controls(response.controls)?;

                let now = Utc::now();
                let system = self
                    .store
                    .observe_system(crate::world_model::SystemDescriptor {
                        id: Uuid::nil(),
                        kind: crate::world_model::SystemKind::Application,
                        key: response.bundle_identifier.clone(),
                        label: if response.application_name.trim().is_empty() {
                            response.bundle_identifier.clone()
                        } else {
                            redact_for_persistence(&response.application_name)
                        },
                        fingerprint: response.application_target.code_digest.clone(),
                        revision: 0,
                        updated_at: now,
                    })?;

                let mut facts = vec![
                    crate::world_model::ObservedFact {
                        name: "application.accessibility_available".into(),
                        subject: None,
                        value: crate::world_model::FactValue::Boolean(
                            response.accessibility_available,
                        ),
                    },
                    crate::world_model::ObservedFact {
                        name: "application.bundle_identifier".into(),
                        subject: None,
                        value: crate::world_model::FactValue::Identifier(
                            response.bundle_identifier.clone(),
                        ),
                    },
                ];
                facts.extend(controls.facts.iter().cloned());
                facts.sort_by(|left, right| {
                    (&left.name, &left.subject).cmp(&(&right.name, &right.subject))
                });
                let interface_digest = controls.interface_fingerprint.clone();
                facts.push(crate::world_model::ObservedFact {
                    name: "application.interface_fingerprint".into(),
                    subject: None,
                    value: crate::world_model::FactValue::Identifier(interface_digest.clone()),
                });
                facts.sort_by(|left, right| {
                    (&left.name, &left.subject).cmp(&(&right.name, &right.subject))
                });
                let observation = crate::world_model::ObservationEnvelope {
                    id: Uuid::new_v4(),
                    system_id: system.id,
                    session_id: None,
                    worker_session: None,
                    system_fingerprint: system.fingerprint.clone(),
                    origin: crate::world_model::EvidenceOrigin::OperatingSystem,
                    privacy: crate::contracts::Sensitivity::Private,
                    observed_at: now,
                    facts,
                };
                let candidates = crate::world_model::safe_probe_candidates(&observation);
                self.store.record_world_observation(&observation)?;
                for (control_id, probe_kind) in &candidates {
                    let capability = crate::world_model::passive_control_capability(
                        &observation,
                        control_id,
                        *probe_kind,
                    )?;
                    self.store.record_capability_candidate(&capability)?;
                }
                let stored_capabilities = self.store.capability_candidates(system.id)?.len();
                let candidate_details: Vec<_> = controls
                    .semantic_controls
                    .iter()
                    .filter_map(|control| {
                        let kind = candidates.get(&control.id)?;
                        Some(json!({
                            "id": control.id,
                            "kind": match kind {
                                crate::world_model::ProbeKind::RestoreSliderValue => "restore_slider_value",
                                crate::world_model::ProbeKind::RestoreToggleState => "restore_toggle_state",
                            },
                            "role": control.role,
                            "label": control.label,
                        }))
                    })
                    .collect();
                let controller_drafts =
                    controller_draft_summaries(&self.store, system.id, &system.label)?;
                Ok(json!({
                    "system": system,
                    "application_target": response.application_target,
                    "observation_id": observation.id,
                    "interface_fingerprint": interface_digest,
                    "accessibility_available": response.accessibility_available,
                    "observed_controls": controls.semantic_controls.len(),
                    "skipped_controls": controls.skipped_controls,
                    "discovery_truncated": response.truncated || controls.truncated,
                    "safe_learning_candidates": candidates,
                    "safe_learning_candidate_details": candidate_details,
                    "passive_capability_hypotheses": stored_capabilities,
                    "controller_drafts": controller_drafts,
                    "active_window_observed": !response.active_window.is_empty()
                }))
            }
            "discover_paired_browser" => {
                let session = self.adapters.session_id("browser").await.ok_or_else(|| {
                    CoreError::ExecutorUnavailable(
                        "Pair one foreground browser tab before discovering its interface".into(),
                    )
                })?;
                let response = self
                    .adapters
                    .request_in_session("browser", &session, "discover_interface", json!({}))
                    .await?;
                if self.adapters.session_id("browser").await.as_deref() != Some(&session) {
                    return Err(CoreError::ExecutorUnavailable(
                        "The paired browser session changed during interface discovery".into(),
                    ));
                }
                if serde_json::to_vec(&response)?.len() > 64 * 1024 {
                    return Err(CoreError::VerificationFailed(
                        "Browser discovery response exceeded its size limit".into(),
                    ));
                }
                let response: BrowserDiscoveryResponse = serde_json::from_value(response)?;
                if !response.available {
                    let message = match response.reason.as_deref() {
                        Some("paired_tab_not_active") => {
                            "The paired browser tab must be in the foreground to discover controls"
                        }
                        Some("active_tab_changed_during_observation") => {
                            "The active browser tab changed during discovery; no observation was stored"
                        }
                        Some("document_changed_during_observation") => {
                            "The browser document changed during discovery; no observation was stored"
                        }
                        _ => "The paired browser could not provide a stable foreground observation",
                    };
                    return Err(CoreError::VerificationFailed(message.into()));
                }
                let target = response.browser_target.ok_or_else(|| {
                    CoreError::VerificationFailed(
                        "Browser discovery omitted its bound document identity".into(),
                    )
                })?;
                let mut prepared = prepare_browser_discovery(
                    target,
                    response.controls,
                    response.truncated,
                    Utc::now(),
                )?;
                let system = self.store.observe_system(prepared.system)?;
                prepared.observation.system_id = system.id;
                self.store.record_world_observation(&prepared.observation)?;
                Ok(json!({
                    "system": system,
                    "observation_id": prepared.observation.id,
                    "interface_fingerprint": prepared.interface_fingerprint,
                    "observed_controls": prepared.observed_controls,
                    "skipped_controls": prepared.skipped_controls,
                    "truncated": prepared.truncated,
                    "passive_capability_hypotheses": 0,
                    "execution_available": false,
                    "learning_available": false
                }))
            }
            "list" => Ok(json!({ "systems": self.store.observed_systems()? })),
            "record_relation" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct RelationRequest {
                    from_system: Uuid,
                    kind: crate::world_model::WorldRelationKind,
                    to_system: Uuid,
                    evidence_id: Uuid,
                }

                if command.json.len() > 8 * 1024 {
                    return Err(CoreError::InvalidAction(
                        "World relation request exceeds its size bound".into(),
                    ));
                }
                let request: RelationRequest = serde_json::from_str(&command.json)?;
                let relation = crate::world_model::WorldRelation {
                    id: parse_id()?,
                    from_system: request.from_system,
                    kind: request.kind,
                    to_system: request.to_system,
                    evidence_id: request.evidence_id,
                };
                self.store.record_world_relation(&relation)?;
                Ok(json!({
                    "relation": relation,
                    "relations": self.store.world_relations(request.from_system)?
                }))
            }
            "system" => {
                let system_id = parse_id()?;
                let system = self
                    .store
                    .observed_systems()?
                    .into_iter()
                    .find(|system| system.id == system_id)
                    .ok_or_else(|| CoreError::InvalidAction("System was not found".into()))?;
                let controller_drafts =
                    controller_draft_summaries(&self.store, system.id, &system.label)?;
                Ok(json!({
                    "system": system,
                    "observations": self.store.observations_for_system(system_id, None)?,
                    "capabilities": self.store.capability_assessments(system_id)?,
                    "relations": self.store.world_relations(system_id)?,
                    "learning_sessions": self.store.learning_sessions(system_id)?,
                    "controller_drafts": controller_drafts
                }))
            }
            "synthesize_goal" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct GoalSynthesisRequest {
                    system_id: Uuid,
                    goal_capability_id: String,
                    goal_output: String,
                    seeds: Vec<crate::agency::GoalInputSeed>,
                }

                if command.json.len() > 64 * 1024 {
                    return Err(CoreError::InvalidAction(
                        "Goal synthesis request exceeds its size bound".into(),
                    ));
                }
                let request: GoalSynthesisRequest = serde_json::from_str(&command.json)?;
                let systems = self
                    .store
                    .observed_systems()?
                    .into_iter()
                    .filter(|system| system.id == request.system_id)
                    .collect::<Vec<_>>();
                if systems.len() != 1 {
                    return Err(CoreError::ExecutorUnavailable(
                        "Goal target system is no longer present in Sage's discovery graph".into(),
                    ));
                }
                let assessments = self.store.capability_assessments(request.system_id)?;
                let mut current_evidence_ids = self
                    .store
                    .current_world_evidence_ids(request.system_id, &systems[0].fingerprint)?;
                current_evidence_ids.extend(assessments.iter().flat_map(|assessment| {
                    assessment
                        .descriptor
                        .evidence_ids
                        .iter()
                        .chain(&assessment.descriptor.preconditions.observed_state_fact_ids)
                        .copied()
                }));
                let registered_executors = crate::features::manifests()
                    .into_iter()
                    .filter(|manifest| manifest.enabled)
                    .map(|manifest| manifest.id)
                    .collect::<BTreeSet<_>>();
                let proposal = crate::agency::synthesize_goal_procedure(
                    &request.goal_capability_id,
                    &request.goal_output,
                    request.seeds,
                    &assessments,
                    &registered_executors,
                    &current_evidence_ids,
                )?;
                let descriptors = assessments
                    .iter()
                    .map(|assessment| assessment.descriptor.clone())
                    .collect::<Vec<_>>();
                let schedule = crate::agency::schedule_procedure(
                    &proposal,
                    &descriptors,
                    &BTreeMap::new(),
                    4,
                )?;
                let mut runtime_state = crate::agency::ProcedureRuntimeState::new(&proposal)?;
                let initial_runtime_advance = crate::agency::advance_procedure(
                    &proposal,
                    &assessments,
                    &current_evidence_ids,
                    &mut runtime_state,
                    &BTreeMap::new(),
                    4,
                )?;
                Ok(json!({
                    "proposal": proposal,
                    "schedule": schedule,
                    "initial_runtime_advance": initial_runtime_advance,
                    "execution_available": false,
                    "requires_fresh_authority_and_runtime_review": true,
                }))
            }
            "run_goal" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct GoalExecutionRequest {
                    system_id: Uuid,
                    goal_capability_id: String,
                    goal_output: String,
                    seeds: Vec<crate::agency::GoalInputSeed>,
                    request: String,
                }

                if command.json.len() > 64 * 1024 {
                    return Err(CoreError::InvalidAction(
                        "Goal execution request exceeds its size bound".into(),
                    ));
                }
                let command_key = crate::commands::SubmissionKey::for_world_model(
                    &command.id,
                    &command.operation,
                    &command.json,
                )?;
                if let Some(task_id) = self.store.accepted_submission(&command_key)? {
                    return Ok(json!({"task_id":task_id,"deduplicated":true}));
                }
                let request: GoalExecutionRequest = serde_json::from_str(&command.json)?;
                let task_request = request.request.trim();
                if task_request.is_empty() || task_request.len() > 64 * 1024 {
                    return Err(CoreError::InvalidAction(
                        "Goal execution requires a bounded, non-empty user request".into(),
                    ));
                }
                let systems = self
                    .store
                    .observed_systems()?
                    .into_iter()
                    .filter(|system| system.id == request.system_id)
                    .collect::<Vec<_>>();
                if systems.len() != 1 {
                    return Err(CoreError::ExecutorUnavailable(
                        "Goal target system is no longer present in Sage's discovery graph".into(),
                    ));
                }
                let assessments = self.store.capability_assessments(request.system_id)?;
                let mut current_evidence_ids = self
                    .store
                    .current_world_evidence_ids(request.system_id, &systems[0].fingerprint)?;
                current_evidence_ids.extend(assessments.iter().flat_map(|assessment| {
                    assessment
                        .descriptor
                        .evidence_ids
                        .iter()
                        .chain(&assessment.descriptor.preconditions.observed_state_fact_ids)
                        .copied()
                }));
                let registered_executors = crate::features::manifests()
                    .into_iter()
                    .filter(|manifest| manifest.enabled)
                    .map(|manifest| manifest.id)
                    .collect::<BTreeSet<_>>();
                let procedure = crate::agency::synthesize_goal_procedure(
                    &request.goal_capability_id,
                    &request.goal_output,
                    request.seeds,
                    &assessments,
                    &registered_executors,
                    &current_evidence_ids,
                )?;
                let task_id = Uuid::new_v4();
                let mut procedure_runtime =
                    crate::agency::ProcedureRuntimeState::new_for_task(&procedure, task_id)?;
                let first_wave = crate::agency::advance_procedure(
                    &procedure,
                    &assessments,
                    &current_evidence_ids,
                    &mut procedure_runtime,
                    &BTreeMap::new(),
                    4,
                )?;
                if first_wave.proposal_wave.is_empty() {
                    return Err(CoreError::ExecutorUnavailable(
                        "The synthesized procedure has no currently executable first step".into(),
                    ));
                }
                let first_node_ids = first_wave
                    .proposal_wave
                    .iter()
                    .map(|proposal| proposal.node_id.clone())
                    .collect::<Vec<_>>();
                let graph = compile_application_control_procedure(
                    &procedure,
                    task_request,
                    task_id,
                    ProcedureCompilationContext {
                        systems: &systems,
                        assessments: &assessments,
                        current_evidence_ids: &current_evidence_ids,
                        runtime: &procedure_runtime,
                        selected_node_ids: &first_node_ids,
                        existing_action_ids: &BTreeMap::new(),
                    },
                )?;
                let task_id = self
                    .submit_run(RunSubmission {
                        request: task_request.to_owned(),
                        conversation_id: None,
                        task_id: Some(task_id),
                        graph: Some(graph),
                        procedure: Some(procedure.clone()),
                        resources: Vec::new(),
                        authority: RunAuthority::Interactive {
                            command: Some(command_key),
                        },
                        skill_lineages: Vec::new(),
                        streamed_prefix: None,
                    })
                    .await?;
                Ok(json!({
                    "task_id": task_id,
                    "deduplicated": false,
                    "proposal": procedure,
                    "task_submitted": true,
                    "requires_fresh_authority_and_verification": true,
                }))
            }
            "compile_controller_draft" => {
                if !command.json.is_empty() {
                    return Err(CoreError::InvalidAction(
                        "Controller draft compilation accepts only a verified task ID".into(),
                    ));
                }
                let task_id = parse_id()?;
                let record = self.store.compile_controller_draft(task_id)?;
                let system_label = self
                    .store
                    .observed_systems()?
                    .into_iter()
                    .find(|system| system.id == record.controller.system_id)
                    .map(|system| system.label)
                    .unwrap_or_else(|| "Current application".into());
                Ok(json!({
                    "controller_draft": controller_draft_detail_json(
                        &record,
                        &system_label,
                        Some(task_id),
                    ),
                    "requires_fresh_review": true,
                    "grants_execution_authority": false,
                }))
            }
            "get_controller_draft" => {
                if !command.json.is_empty() {
                    return Err(CoreError::InvalidAction(
                        "Controller inspection does not accept extra input".into(),
                    ));
                }
                let record = self.store.load_controller(&command.id)?.ok_or_else(|| {
                    CoreError::InvalidAction("Controller draft was not found".into())
                })?;
                if !matches!(
                    record.status,
                    crate::agency::ControllerStatus::Draft
                        | crate::agency::ControllerStatus::Reviewed
                ) {
                    return Err(CoreError::PermissionRequired(
                        "This controller is disabled or invalidated and cannot be reviewed".into(),
                    ));
                }
                let system_label = self
                    .store
                    .observed_systems()?
                    .into_iter()
                    .find(|system| system.id == record.controller.system_id)
                    .map(|system| system.label)
                    .unwrap_or_else(|| "Current application".into());
                Ok(json!({
                    "controller_draft": controller_draft_detail_json(
                        &record,
                        &system_label,
                        None,
                    ),
                    "grants_execution_authority": false,
                    "execution_available": false,
                }))
            }
            "review_controller_draft" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct ReviewRequest {
                    expected_revision: u64,
                }

                if command.json.len() > 4096 {
                    return Err(CoreError::InvalidAction(
                        "Controller review request exceeds its size bound".into(),
                    ));
                }
                let request: ReviewRequest = serde_json::from_str(&command.json)?;
                let current = self.store.load_controller(&command.id)?.ok_or_else(|| {
                    CoreError::InvalidAction("Controller draft was not found".into())
                })?;
                if current.revision != request.expected_revision
                    || !matches!(
                        current.status,
                        crate::agency::ControllerStatus::Draft
                            | crate::agency::ControllerStatus::Reviewed
                    )
                {
                    return Err(CoreError::VerificationFailed(
                        "Controller draft changed or is not eligible for review".into(),
                    ));
                }
                let native_session = self.adapters.session_id("native").await.ok_or_else(|| {
                    CoreError::ExecutorUnavailable(
                        "A connected native adapter is required to review this controller".into(),
                    )
                })?;
                let discovery = Box::pin(self.world_model_command(
                    sage_protocol::sage::ipc::v2::WorldModelCommand {
                        operation: "discover_current_application".into(),
                        ..Default::default()
                    },
                ))
                .await?;
                if self.adapters.session_id("native").await.as_deref()
                    != Some(native_session.as_str())
                {
                    return Err(CoreError::ExecutorUnavailable(
                        "The native adapter changed during controller review".into(),
                    ));
                }
                let system: crate::world_model::SystemDescriptor =
                    serde_json::from_value(discovery["system"].clone())?;
                let application_target: crate::application_target::ApplicationTarget =
                    serde_json::from_value(discovery["application_target"].clone())?;
                application_target.validate()?;
                if system.id != current.controller.system_id
                    || system.fingerprint != current.controller.system_fingerprint
                    || application_target.identifier != system.key
                    || application_target.code_digest != system.fingerprint
                {
                    return Err(CoreError::PermissionRequired(
                        "The foreground application does not match this controller's exact target"
                            .into(),
                    ));
                }
                let observation_id = discovery["observation_id"]
                    .as_str()
                    .ok_or_else(|| {
                        CoreError::Protocol(
                            "Fresh application discovery omitted its observation identity".into(),
                        )
                    })?
                    .parse::<Uuid>()
                    .map_err(|_| {
                        CoreError::Protocol(
                            "Fresh application discovery returned an invalid observation identity"
                                .into(),
                        )
                    })?;
                let reviewed = self.store.review_controller(
                    &command.id,
                    request.expected_revision,
                    observation_id,
                )?;
                Ok(json!({
                    "controller_draft": controller_draft_detail_json(
                        &reviewed,
                        &system.label,
                        None,
                    ),
                    "controller_reviewed": true,
                    "fresh_rebind_verified": true,
                    "grants_execution_authority": false,
                    "execution_available": false,
                }))
            }
            "forget_system" => {
                Ok(json!({ "forgotten": self.store.forget_world_system(parse_id()?)? }))
            }
            "approve_learning_session" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct ApprovalRequest {
                    system_id: Uuid,
                    system_fingerprint: String,
                    permitted_probes: BTreeMap<String, crate::world_model::ProbeKind>,
                    expires_in_seconds: u32,
                }
                let request: ApprovalRequest = serde_json::from_str(&command.json)?;
                if request.expires_in_seconds == 0 || request.expires_in_seconds > 600 {
                    return Err(CoreError::PermissionRequired(
                        "Learning approvals expire within ten minutes".into(),
                    ));
                }
                let worker_session = self.adapters.session_id("native").await.ok_or_else(|| {
                    CoreError::PermissionRequired(
                        "A live authenticated native application adapter is required for learning"
                            .into(),
                    )
                })?;
                let session = self.store.approve_learning_session(
                    request.system_id,
                    &request.system_fingerprint,
                    &worker_session,
                    request.permitted_probes,
                    Utc::now() + ChronoDuration::seconds(request.expires_in_seconds.into()),
                )?;
                Ok(json!({ "session": session }))
            }
            "run_learning_probe" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct ProbeRequest {
                    control_id: String,
                }

                let session_id = parse_id()?;
                let request: ProbeRequest = serde_json::from_str(&command.json)?;
                let session = self.store.learning_session(session_id)?.ok_or_else(|| {
                    CoreError::PermissionRequired("Learning approval was not found".into())
                })?;
                let now = Utc::now();
                if session.state != crate::world_model::LearningSessionState::Active
                    || session.expires_at <= now
                {
                    return Err(CoreError::PermissionRequired(
                        "Learning approval is expired, stopped, or awaiting restoration review"
                            .into(),
                    ));
                }
                let approved_kind = session
                    .permitted_probes
                    .get(&request.control_id)
                    .copied()
                    .ok_or_else(|| {
                        CoreError::PermissionRequired(
                            "This control was not included in the approved learning session".into(),
                        )
                    })?;
                let worker_session = self.adapters.session_id("native").await.ok_or_else(|| {
                    CoreError::ExecutorUnavailable(
                        "A connected native adapter is required for learning".into(),
                    )
                })?;
                if worker_session != session.worker_session {
                    self.store.stop_learning_session(
                        session_id,
                        crate::world_model::LearningSessionState::InterruptedNeedsReview,
                    )?;
                    return Err(CoreError::PermissionRequired(
                        "The native adapter session changed after learning approval".into(),
                    ));
                }

                // Re-scan the foreground target passively. This refreshes the
                // exact semantic control facts before the single-use lease is
                // consumed and rejects changed apps or controls.
                let scan = sage_protocol::sage::ipc::v2::WorldModelCommand {
                    operation: "discover_current_application".into(),
                    ..Default::default()
                };
                let discovery = Box::pin(self.world_model_command(scan)).await?;
                let system: crate::world_model::SystemDescriptor =
                    serde_json::from_value(discovery.get("system").cloned().ok_or_else(|| {
                        CoreError::Protocol("Fresh scan omitted its target system".into())
                    })?)?;
                let application_target: crate::application_target::ApplicationTarget =
                    serde_json::from_value(
                        discovery
                            .get("application_target")
                            .cloned()
                            .ok_or_else(|| {
                                CoreError::Protocol("Fresh scan omitted its signed target".into())
                            })?,
                    )?;
                application_target.validate()?;
                let process_id = discovery
                    .get("observed_process_id")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok())
                    .filter(|value| *value > 0)
                    .ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Fresh scan omitted its process identity".into(),
                        )
                    })?;
                if system.id != session.system_id
                    || system.fingerprint != session.system_fingerprint
                    || application_target.identifier != system.key
                    || application_target.code_digest != session.system_fingerprint
                {
                    self.store.stop_learning_session(
                        session_id,
                        crate::world_model::LearningSessionState::InterruptedNeedsReview,
                    )?;
                    return Err(CoreError::PermissionRequired(
                        "The foreground application no longer matches the exact approved identity"
                            .into(),
                    ));
                }
                let fresh_candidates = discovery
                    .get("safe_learning_candidates")
                    .and_then(serde_json::Value::as_object)
                    .and_then(|candidates| candidates.get(&request.control_id))
                    .cloned()
                    .map(serde_json::from_value::<crate::world_model::ProbeKind>)
                    .transpose()?
                    .ok_or_else(|| {
                        CoreError::PermissionRequired(
                            "The approved control is no longer a fresh, safe learning candidate"
                                .into(),
                        )
                    })?;
                if fresh_candidates != approved_kind {
                    self.store.stop_learning_session(
                        session_id,
                        crate::world_model::LearningSessionState::InterruptedNeedsReview,
                    )?;
                    return Err(CoreError::PermissionRequired(
                        "The control's type or reversible behavior changed after approval".into(),
                    ));
                }

                let lease = self.store.begin_probe(
                    session_id,
                    &worker_session,
                    &session.system_fingerprint,
                    &request.control_id,
                    process_id,
                    Utc::now(),
                )?;
                let mut settlement = ProbeSettlementGuard::new(self.store.clone(), session_id);
                let response = self.adapters.request_in_session(
                    "native",
                    &worker_session,
                    "probe_control",
                    json!({"application_target":application_target,"expected_process_id":process_id,"probe_lease":lease}),
                ).await?;
                if self.adapters.session_id("native").await.as_deref()
                    != Some(worker_session.as_str())
                {
                    return Err(CoreError::ExecutorUnavailable(
                        "The native adapter changed while the reversible probe was settling".into(),
                    ));
                }
                let response: AdapterProbeResponse = serde_json::from_value(response)?;
                if response.application_target != application_target
                    || response.observed_process_id != process_id
                    || response.observed_process_id != lease.expected_process_id
                    || response.control_id != lease.control_id
                    || !response.restoration_verified
                    || response.observations.len() != 3
                {
                    return Err(CoreError::VerificationFailed(
                        "Native adapter did not verify exact restoration of the approved control"
                            .into(),
                    ));
                }
                let (observations, evidence) =
                    probe_observations(&lease, approved_kind, response.observations)?;
                self.store
                    .record_probe_observations(lease.id, &observations, Utc::now())?;
                self.store.complete_probe(lease.id, &evidence, Utc::now())?;
                settlement.settled();
                Ok(
                    json!({"probe_id":lease.id,"session_id":session_id,"control_id":lease.control_id,"restoration_verified":true,"transition_recorded":true}),
                )
            }
            "stop_learning_session" => {
                self.store
                    .stop_learning_session(parse_id()?, LearningSessionState::Revoked)?;
                Ok(json!({ "stopped": true }))
            }
            _ => Err(CoreError::InvalidAction(
                "Unknown world-model operation".into(),
            )),
        }
    }

    pub(crate) fn native_adapter_disconnected(&self, worker_session: &str) -> CoreResult<()> {
        self.store.stop_learning_sessions_for_worker(worker_session)
    }

    pub async fn workflow_command(
        self: &Arc<Self>,
        command: sage_protocol::sage::ipc::v2::WorkflowCommand,
    ) -> CoreResult<serde_json::Value> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct LearningConfiguration {
            enabled: bool,
        }
        // Validate management input before it can affect an in-flight task.
        // Disabling learning withdraws active learned runs; enabling it does
        // not disturb them, and malformed commands have no side effects.
        let disable_learning = if command.operation == "configure_learning" {
            Some(!serde_json::from_str::<LearningConfiguration>(&command.json)?.enabled)
        } else {
            None
        };
        if command.operation == "forget_routine"
            && !(command.id.len() == 64
                && command
                    .id
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
        {
            return Err(CoreError::InvalidAction("Invalid routine ID".into()));
        }
        let _learning_guard = if matches!(
            command.operation.as_str(),
            "configure_learning"
                | "review_routine"
                | "review_skill"
                | "forget_routine"
                | "synthesize_skill"
                | "save_workflow"
        ) {
            Some(self.submission_lock.lock().await)
        } else {
            None
        };
        if let Some(disable) = disable_learning {
            self.store
                .save_setting("routine_learning.enabled", &!disable)?;
        }
        if matches!(
            command.operation.as_str(),
            "configure_learning" | "forget_routine"
        ) {
            let active: Vec<_> = self
                .tasks
                .read()
                .await
                .values()
                .filter(|task| {
                    let learned_run = task.compiled_routine.as_ref().is_some_and(|id| {
                        disable_learning == Some(true)
                            || (command.operation == "forget_routine" && *id == command.id)
                    });
                    let synthesized_run = !task.synthesized_skill_sources.is_empty()
                        && (disable_learning == Some(true)
                            || (command.operation == "forget_routine"
                                && task.synthesized_skill_sources.contains(&command.id)));
                    task.status.is_active() && (learned_run || synthesized_run)
                })
                .map(|task| task.id)
                .collect();
            for id in active {
                self.stop_task(id).await?;
            }
        }
        let _schedule_guard = if matches!(
            command.operation.as_str(),
            "save_schedule" | "delete_schedule"
        ) {
            Some(self.scheduler_lane.lock().await)
        } else {
            None
        };
        if _schedule_guard.is_some() {
            // Management changes revoke the old firing before changing its
            // authorization. Serialize this with claiming/dispatching a firing.
            let schedule_id = if command.operation == "save_schedule" {
                serde_json::from_str::<crate::workflows::Schedule>(&command.json)?
                    .id
                    .to_string()
            } else {
                command.id.clone()
            };
            if let Some(task_id) = self
                .store
                .schedule(
                    Uuid::parse_str(&schedule_id)
                        .map_err(|_| CoreError::InvalidAction("Invalid schedule ID".into()))?,
                )?
                .and_then(|s| s.last_task_id)
                && let Ok(task) = self.get_task(task_id).await
                && !task.status.is_terminal()
            {
                self.control_task(task_id, TaskStatus::Cancelled).await?;
            }
        }
        let id = || {
            Uuid::parse_str(&command.id).map_err(|_| CoreError::InvalidAction("Invalid ID".into()))
        };
        let conversation = if command.conversation_id.is_empty() {
            None
        } else {
            Some(
                Uuid::parse_str(&command.conversation_id)
                    .map_err(|_| CoreError::InvalidAction("Invalid conversation".into()))?,
            )
        };
        match command.operation.as_str() {
            "list" => {}
            "configure_learning" => {}
            "review_routine" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Review {
                    digest: String,
                }
                let review: Review = serde_json::from_str(&command.json)?;
                self.store.review_routine(&command.id, &review.digest)?;
            }
            "forget_routine" => self.store.forget_routine(&command.id)?,
            "synthesize_skill" => {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Synthesis {
                    digest: String,
                }
                let synthesis: Synthesis = serde_json::from_str(&command.json)?;
                self.store.synthesize_routine_skill(
                    &command.id,
                    &synthesis.digest,
                    &command.name,
                )?;
            }
            "capture_skill" => {
                self.store
                    .capture_skill(&self.get_task(id()?).await?, &command.name)?;
            }
            "run_skill" => {
                let skill = self
                    .store
                    .skill(id()?)?
                    .filter(|s| s.is_reviewed())
                    .ok_or_else(|| CoreError::InvalidAction("Skill unavailable".into()))?;
                let lineages = if skill.synthesized_from.is_empty() {
                    Vec::new()
                } else {
                    vec![(
                        skill.synthesis_family_id.clone().unwrap_or_default(),
                        skill.synthesis_digest.clone().unwrap_or_default(),
                        skill.synthesized_from.clone(),
                    )]
                };
                self.submit_run(RunSubmission {
                    request: format!("Run skill: {}", skill.name),
                    conversation_id: conversation,
                    task_id: None,
                    graph: Some(skill.graph),
                    procedure: None,
                    resources: Vec::new(),
                    authority: RunAuthority::Interactive { command: None },
                    skill_lineages: lineages,
                    streamed_prefix: None,
                })
                .await?;
            }
            "review_skill" => {
                let mut skill = self
                    .store
                    .skill(id()?)?
                    .ok_or_else(|| CoreError::InvalidAction("Skill not found".into()))?;
                if !skill.synthesized_from.is_empty()
                    && !self.store.routine_family_matches(
                        skill.synthesis_family_id.as_deref().unwrap_or_default(),
                        skill.synthesis_digest.as_deref().unwrap_or_default(),
                        &skill.synthesized_from,
                    )?
                {
                    return Err(CoreError::ApprovalRejected(
                        "The source routine branches changed or were forgotten. Review a fresh shared-prefix draft."
                            .into(),
                    ));
                }
                let review: serde_json::Value = serde_json::from_str(&command.json)?;
                let digest = skill.digest()?;
                if review["digest"].as_str() != Some(&digest) {
                    return Err(CoreError::ApprovalRejected(
                        "Skill changed since its review preview".into(),
                    ));
                }
                for node in &skill.graph.nodes {
                    let mut proposal = self.resolver.prepare_proposal(&node.proposal)?;
                    crate::verification::bind_required_outcome(&mut proposal)?;
                    self.compiler
                        .compile(proposal, &self.current_availability().await)?;
                }
                skill.schema_version = 2;
                skill.reviewed_digest = Some(digest);
                skill.enabled = true;
                self.store.save_skill(&skill)?;
            }
            "save_workflow" => self.store.save_workflow(&serde_json::from_str::<
                crate::workflows::Workflow,
            >(&command.json)?)?,
            "run_workflow" => {
                let workflow_id = id()?;
                let graph = self.store.workflow_graph(workflow_id, Uuid::new_v4())?;
                let skill_lineages = self.store.workflow_skill_lineages(workflow_id)?;
                self.submit_run(RunSubmission {
                    request: graph.goal.clone(),
                    conversation_id: conversation,
                    task_id: None,
                    graph: Some(graph),
                    procedure: None,
                    resources: Vec::new(),
                    authority: RunAuthority::Interactive { command: None },
                    skill_lineages,
                    streamed_prefix: None,
                })
                .await?;
            }
            "save_schedule" => {
                let mut schedule: crate::workflows::Schedule = serde_json::from_str(&command.json)?;
                schedule.resources = command
                    .background_resources
                    .iter()
                    .map(|scope| {
                        let effects = scope
                            .effects
                            .iter()
                            .map(|effect| {
                                match sage_protocol::sage::ipc::v2::ResourceEffect::try_from(
                                    *effect,
                                ) {
                                    Ok(sage_protocol::sage::ipc::v2::ResourceEffect::Read) => {
                                        Ok(crate::contracts::Effect::Read)
                                    }
                                    Ok(sage_protocol::sage::ipc::v2::ResourceEffect::Create) => {
                                        Ok(crate::contracts::Effect::Create)
                                    }
                                    _ => Err(CoreError::PermissionRequired(
                                        "Unsupported background effect".into(),
                                    )),
                                }
                            })
                            .collect::<CoreResult<std::collections::BTreeSet<_>>>()?;
                        if effects.is_empty() {
                            return Err(CoreError::PermissionRequired(
                                "Empty background resource grant".into(),
                            ));
                        }
                        Ok(crate::contracts::ResourceScope {
                            root: self
                                .resolver
                                .validate_scope_root(std::path::Path::new(&scope.root))?,
                            effects,
                        })
                    })
                    .collect::<CoreResult<_>>()?;
                schedule.expires_at =
                    chrono::DateTime::from_timestamp_millis(command.background_expires_at_unix_ms);
                schedule.remaining_runs = command.maximum_runs;
                if let crate::workflows::Trigger::FolderChanged { path } = &mut schedule.trigger {
                    *path = self.resolver.validate_scope_root(path)?;
                }
                schedule.request = redact_for_persistence(&schedule.request);
                if let Some(workflow) = schedule.workflow_id {
                    self.store.workflow_graph(workflow, Uuid::new_v4())?;
                }
                self.validate_trigger(&schedule)?;
                self.store
                    .ensure_conversation(Some(schedule.conversation_id), &schedule.name)?;
                if let crate::workflows::Trigger::Once { at } = schedule.trigger {
                    schedule.next_run_at = at;
                }
                self.store.save_schedule(&schedule)?;
            }
            "delete_skill" => self.store.delete_definition("skill", id()?)?,
            "delete_workflow" => self.store.delete_definition("workflow", id()?)?,
            "delete_schedule" => self.store.delete_definition("schedule", id()?)?,
            _ => {
                return Err(CoreError::InvalidAction(
                    "Unknown workflow operation".into(),
                ));
            }
        }
        self.store.append_audit(
            None,
            None,
            "workflow_management",
            &json!({"operation":command.operation,"id":command.id}),
        )?;
        self.workflow_snapshot()
    }

    async fn current_availability(&self) -> ExecutorAvailability {
        let mut availability = self.availability.clone();
        availability.accessibility = self.adapters.available("native").await;
        availability.learned_application_control = self
            .adapters
            .supports_feature("native", "application_control_v1")
            .await;
        availability.browser_dom = self.adapters.available("browser").await;
        availability
    }

    async fn available_tools(&self) -> Vec<ToolDescriptor> {
        let availability = self.current_availability().await;
        default_tools()
            .into_iter()
            .filter(|tool| match tool.executor.as_str() {
                "browser" => availability.browser_dom,
                "native" if tool.name == "open_application" => availability.accessibility,
                "native" if tool.name == "set_application_control" => {
                    availability.accessibility && availability.learned_application_control
                }
                _ => true,
            })
            .collect()
    }

    async fn resume_interrupted(self: &Arc<Self>, task_id: Uuid) -> CoreResult<()> {
        let _submission = self.submission_lock.lock().await;
        if self.runtime.is_active(task_id) {
            return Err(CoreError::Busy(
                "The previous execution is still stopping; wait before resuming".into(),
            ));
        }
        if self.runtime.is_stopped(task_id) {
            return Err(CoreError::Cancelled);
        }
        if let Some((attempt, finish)) = self.runtime.pending_finish(task_id) {
            if self.get_task(task_id).await?.execution_attempt != attempt {
                return Err(CoreError::InvalidAction(
                    "The pending result belongs to an earlier execution attempt".into(),
                ));
            }
            if self.finalize_task(task_id, finish).await? != TaskStatus::Interrupted {
                return Ok(());
            }
        }
        if self.adapters.has_outstanding_effects(task_id) {
            return Err(CoreError::PermissionRequired("A worker has not finished responding to this task. Wait for its result before reconciling or continuing.".into()));
        }
        self.reconcile_task_worker_receipts(task_id).await?;
        let task = self.get_task(task_id).await?;
        if task.status != TaskStatus::Interrupted {
            return Err(CoreError::InvalidAction(
                "This task is no longer waiting for recovery".into(),
            ));
        }
        if task.continued_by.is_some() {
            return Err(CoreError::InvalidAction(
                "This task already continued. Use its latest continuation.".into(),
            ));
        }
        if self.store.control_scope_stopped(task.control_scope())? {
            return Err(CoreError::Cancelled);
        }
        if self.store.scope_has_undo(task.control_scope())? {
            return Err(CoreError::PermissionRequired(
                "This work has an Undo record. Check its result, then start a new request to continue from the current state.".into(),
            ));
        }
        let lease = self.runtime.begin_scoped(task_id, task.control_scope())?;
        lease.mark_accepted();
        if task.budget_exhausted || task.actions.len() >= 32 {
            if task.actions.values().any(|action| {
                !matches!(
                    action.status,
                    ActionStatus::Succeeded | ActionStatus::Skipped
                )
            }) {
                return Err(CoreError::PermissionRequired(
                    "Reconcile unfinished effects before starting a continuation".into(),
                ));
            }
            let contract = task.contract.as_ref().ok_or_else(|| {
                CoreError::PermissionRequired("Pre-v2 work needs a new task scope".into())
            })?;
            let resources = contract.resources.clone();
            let skill_lineages = task.synthesized_skill_lineages.clone();
            drop(_submission);
            let result = self
                .submit_run(RunSubmission {
                    request: task.request.clone(),
                    conversation_id: task.conversation_id,
                    task_id: None,
                    graph: None,
                    procedure: None,
                    resources,
                    authority: RunAuthority::Continued {
                        previous: Box::new(task),
                        owner: lease.signal.clone(),
                    },
                    skill_lineages,
                    streamed_prefix: None,
                })
                .await;
            if lease.signal.is_stopped() {
                let _ = self.persist_stopped_run(task_id).await;
            }
            return result.map(|_| ());
        }
        if self.tasks.read().await.values().any(|other| {
            other.id != task_id
                && other.conversation_id == task.conversation_id
                && other.status.is_active()
        }) {
            return Err(CoreError::InvalidAction(
                "Finish other work in this conversation first.".into(),
            ));
        }
        self.capabilities.revoke_task(task_id).await;
        let core = Arc::clone(self);
        if task.contract.is_none() {
            return Err(CoreError::PermissionRequired(
                "Start a new task to authorize this pre-v2 work".into(),
            ));
        }
        self.update_task(task_id, |task| {
            if task.status != TaskStatus::Interrupted {
                return Err(CoreError::Cancelled);
            }
            task.execution_attempt = task
                .execution_attempt
                .checked_add(1)
                .filter(|attempt| *attempt <= i64::MAX as u64)
                .ok_or_else(|| CoreError::Storage("Execution attempt counter exhausted".into()))?;
            if let Some(contract) = &mut task.contract {
                if !task.background {
                    contract.expires_at = Utc::now() + ChronoDuration::hours(2);
                }
                contract.validate(task_id)?;
            }
            Ok(())
        })
        .await?;
        self.set_task_status(
            task_id,
            TaskStatus::Paused,
            "Re-observing completed actions before recovery",
        )
        .await?;
        // Recovery and execution share one cancellable ownership generation.
        tokio::spawn(core.run_owned(task_id, lease, true, None));
        Ok(())
    }

    async fn prepare_recovery(&self, task_id: Uuid) -> CoreResult<()> {
        let task = self.get_task(task_id).await?;
        let availability = self.current_availability().await;
        for state in task.actions.values() {
            // A running action may already have changed the outside world.
            // Only repeat it if its declared outcome can be independently observed.
            if matches!(
                state.status,
                ActionStatus::Succeeded
                    | ActionStatus::Running
                    | ActionStatus::Verifying
                    | ActionStatus::Uncertain
            ) {
                if let Action::ReadFile { path, .. } = &state.proposal.action {
                    let expected = task.tool_results.iter().rev().find(|result| result.action_id == state.proposal.id && result.verdict == crate::contracts::Verdict::Confirmed)
                        .and_then(|result| result.output["sha256"].as_str())
                        .ok_or_else(|| CoreError::VerificationFailed("The earlier read has no durable content identity; read the source again in a new task".into()))?;
                    let actual = crate::execution::files::PinnedPath::open(path)?
                        .digest(16 * 1024 * 1024)?;
                    if actual != expected {
                        return Err(CoreError::VerificationFailed("A previously read file changed; a fresh task must read it before continuing".into()));
                    }
                }
                if !matches!(
                    state.proposal.expected_outcome,
                    crate::domain::ExpectedOutcome::Condition { .. }
                        | crate::domain::ExpectedOutcome::FileContains { .. }
                        | crate::domain::ExpectedOutcome::DirectoryPage { .. }
                ) {
                    return Err(CoreError::VerificationFailed("An executed action has no independently observable recovery outcome; review it before starting a new task.".into()));
                }
                let receipt = ExecutionReceipt {
                    executor: "recovery".into(),
                    summary: "Verified the current state during recovery.".into(),
                    transient_data: task
                        .tool_results
                        .iter()
                        .rev()
                        .find(|result| {
                            result.action_id == state.proposal.id
                                && result.verdict == crate::contracts::Verdict::Confirmed
                        })
                        .map(|result| result.output.clone())
                        .unwrap_or_else(|| json!({"recovered":true})),
                    rollback: None,
                };
                let observation = self.observer.observe(&state.proposal, &receipt).await?;
                self.verifier
                    .verify(&state.proposal.expected_outcome, &observation)?;
                self.commit_observation(
                    task_id,
                    &state.proposal,
                    &receipt,
                    &observation,
                    crate::transitions::VerificationMode::Recovery,
                )
                .await?;
            } else if state.status != ActionStatus::Skipped {
                let proposal = self.resolver.prepare_proposal(&state.proposal)?;
                if let PolicyDecision::Deny { reason, .. } = self.policy.evaluate(
                    &proposal,
                    &PolicyContext {
                        task_request: task.request.clone(),
                        has_fresh_native_authentication: false,
                        is_recovery_attempt: true,
                    },
                )? {
                    return Err(CoreError::PolicyDenied(reason));
                }
                self.compiler.compile(proposal, &availability)?;
            }
        }
        self.store.append_audit(Some(task_id),None,"recovery_revalidated",&json!({"completed":task.completed_count(),"remaining":task.actions.len()-task.completed_count()}))?;
        self.update_task(task_id, |task| {
            if task.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            for state in task.actions.values_mut() {
                if !matches!(
                    state.status,
                    ActionStatus::Succeeded | ActionStatus::Skipped
                ) {
                    state.status = ActionStatus::Pending;
                    state.error = None;
                }
            }
            task.status = TaskStatus::Running;
            task.recovery_attempt = true;
            task.final_outcome = None;
            task.touch();
            Ok(())
        })
        .await
    }

    fn validate_trigger(&self, schedule: &crate::workflows::Schedule) -> CoreResult<()> {
        use crate::domain::{ActionProposal, Condition, ExpectedOutcome};
        let condition = match &schedule.trigger {
            crate::workflows::Trigger::FolderChanged { path } => {
                Some(Condition::FileExists { path: path.clone() })
            }
            crate::workflows::Trigger::Condition { condition } => Some(condition.clone()),
            _ => None,
        };
        if let Some(condition) = condition {
            let path = match &condition {
                Condition::FileExists { path }
                | Condition::FileAbsent { path }
                | Condition::FolderExists { path } => path,
                _ => {
                    return Err(CoreError::PermissionRequired(
                        "Background application observation is not qualified".into(),
                    ));
                }
            };
            if !schedule.resources.iter().any(|scope| {
                path.starts_with(&scope.root)
                    && scope.effects.contains(&crate::contracts::Effect::Read)
            }) {
                return Err(CoreError::PermissionRequired(
                    "The trigger needs a matching background folder read grant".into(),
                ));
            }
            let proposal = ActionProposal {
                id: Uuid::new_v4(),
                task_id: Uuid::new_v4(),
                action: Action::WaitForCondition {
                    condition: condition.clone(),
                    timeout_ms: 1000,
                },
                expected_outcome: ExpectedOutcome::Condition {
                    condition: condition.clone(),
                },
                target_resource: "condition".into(),
                provenance: crate::domain::Provenance::user(),
                metadata: Default::default(),
            };
            let resolved = self.resolver.prepare_proposal(&proposal)?;
            if let Action::WaitForCondition { condition, .. } = resolved.action {
                let resolved_path = match condition {
                    Condition::FileExists { path }
                    | Condition::FileAbsent { path }
                    | Condition::FolderExists { path } => path,
                    _ => {
                        return Err(CoreError::PermissionRequired(
                            "Unsupported background trigger".into(),
                        ));
                    }
                };
                if resolved_path != *path {
                    return Err(CoreError::PermissionRequired(
                        "Background target identity changed; re-authorize the canonical folder"
                            .into(),
                    ));
                }
                crate::execution::files::PinnedPath::open(path)?;
            }
        }
        Ok(())
    }

    pub fn start_scheduler(self: &Arc<Self>) {
        let core = Arc::downgrade(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(15));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            let (sender, mut changes) = tokio::sync::mpsc::channel(128);
            let mut monitor = crate::workflows::FolderMonitor::new(sender);
            let mut changed_paths = std::collections::HashSet::new();
            loop {
                tokio::select! {
                    _=interval.tick()=>{},
                    Some(())=changes.recv()=>{},
                }
                let Some(core) = core.upgrade() else {
                    break;
                };
                if let Ok(schedules) = core.store.folder_schedules() {
                    let mut valid = Vec::new();
                    for schedule in schedules {
                        if let Err(error) = core.validate_trigger(&schedule) {
                            let _ = core
                                .store
                                .disable_schedule_if_current(&schedule, &error.to_string());
                        } else {
                            valid.push(schedule);
                        }
                    }
                    monitor.sync(&valid);
                    if let Ok((changed, errors)) = monitor.drain() {
                        changed_paths.extend(changed);
                        for schedule in &valid {
                            if let crate::workflows::Trigger::FolderChanged { path } =
                                &schedule.trigger
                                && let Some(error) = errors.get(path)
                            {
                                let _ = core.store.disable_schedule_if_current(schedule, error);
                            }
                        }
                    }
                }
                if let Err(error) = core.scheduler_tick(&mut changed_paths).await {
                    tracing::warn!(error=%error,"scheduler tick failed");
                }
            }
        });
    }

    async fn scheduler_tick(
        self: &Arc<Self>,
        changed_paths: &mut std::collections::HashSet<std::path::PathBuf>,
    ) -> CoreResult<()> {
        let _schedule_guard = self.scheduler_lane.lock().await;
        use crate::workflows::Trigger;
        // Persist a coalesced trigger before checking cooldown or overlap. A
        // busy run must not silently discard a folder event.
        for mut schedule in self.store.folder_schedules()? {
            if let Trigger::FolderChanged { path } = &schedule.trigger
                && changed_paths
                    .iter()
                    .any(|changed| changed.starts_with(path))
            {
                schedule.trigger_pending = true;
                match self.validate_trigger(&schedule) {
                    Ok(()) => self.store.save_schedule(&schedule)?,
                    Err(error) => {
                        self.store
                            .disable_schedule_if_current(&schedule, &error.to_string())?;
                    }
                }
            }
        }
        // Clear only after every matching trigger was saved (or revoked).
        // If storage failed, retry the same dirty roots on the next tick.
        changed_paths.clear();
        let due_at = Utc::now();
        let mut cursor = None;
        loop {
            let page = self.store.due_schedules(due_at, cursor)?;
            if page.is_empty() {
                break;
            }
            cursor = page
                .last()
                .map(|schedule| (schedule.next_run_at, schedule.id));
            for mut schedule in page {
                if let Err(error) = self.validate_trigger(&schedule) {
                    self.store
                        .disable_schedule_if_current(&schedule, &error.to_string())?;
                    continue;
                }
                if schedule
                    .expires_at
                    .is_none_or(|expiry| expiry <= Utc::now())
                    || schedule.remaining_runs == 0
                {
                    schedule.enabled = false;
                    schedule.last_error =
                        Some("Background authorization expired or reached its run budget".into());
                    self.store.save_schedule(&schedule)?;
                    continue;
                }
                if self.tasks.read().await.values().any(|t| {
                    t.conversation_id == Some(schedule.conversation_id) && !t.status.is_terminal()
                }) {
                    continue;
                }
                let ready = match &schedule.trigger {
                    Trigger::Once { at } => *at <= Utc::now(),
                    Trigger::Interval { .. } => true,
                    Trigger::FolderChanged { .. } => schedule.trigger_pending,
                    Trigger::Condition { condition } => {
                        let matches = match condition {
                            crate::domain::Condition::FileExists { path } => {
                                crate::execution::files::PinnedPath::open(path)?.exists()
                            }
                            crate::domain::Condition::FileAbsent { path } => {
                                !crate::execution::files::PinnedPath::open(path)?.exists()
                            }
                            crate::domain::Condition::FolderExists { path } => {
                                crate::execution::files::PinnedPath::open(path)?.is_directory()
                            }
                            _ => self
                                .adapters
                                .observe_condition(condition, None)
                                .await
                                .is_ok_and(|e| {
                                    self.verifier
                                    .verify(
                                        &crate::domain::ExpectedOutcome::Condition {
                                            condition: condition.clone(),
                                        },
                                        &crate::observation::Observation {
                                            observed_at: Utc::now(),
                                            provenance: crate::domain::Provenance::external(
                                                crate::domain::ProvenanceSource::OperatingSystem,
                                                "trigger",
                                            ),
                                            summary: String::new(),
                                            evidence: vec![e],
                                        },
                                    )
                                    .is_ok()
                                }),
                        };
                        let edge = matches && !schedule.last_condition;
                        schedule.last_condition = matches;
                        self.store.save_schedule(&schedule)?;
                        edge
                    }
                };
                if !ready {
                    continue;
                }
                let Some(run) = self.store.claim_schedule(&mut schedule)? else {
                    continue;
                };
                let result = async {
                    let graph = schedule
                        .workflow_id
                        .map(|id| self.store.workflow_graph(id, Uuid::new_v4()))
                        .transpose()?;
                    let skill_lineages = schedule
                        .workflow_id
                        .map(|id| self.store.workflow_skill_lineages(id))
                        .transpose()?
                        .unwrap_or_default();
                    self.submit_run(RunSubmission {
                        request: schedule.request.clone(),
                        conversation_id: Some(schedule.conversation_id),
                        task_id: None,
                        graph,
                        procedure: None,
                        resources: schedule.resources.clone(),
                        authority: RunAuthority::Scheduled {
                            expires_at: schedule.expires_at.ok_or_else(|| {
                                CoreError::PermissionRequired(
                                    "A background run requires an authorization expiry".into(),
                                )
                            })?,
                        },
                        skill_lineages,
                        streamed_prefix: None,
                    })
                    .await
                }
                .await;
                match result {
                    Ok(task) => {
                        schedule.last_task_id = Some(task);
                        schedule.last_error = None;
                        self.store.finish_schedule_claim(run, Some(task))?;
                    }
                    Err(error) => {
                        schedule.last_error = Some(redact_for_persistence(&error.to_string()));
                        schedule.enabled = false;
                        self.store.finish_schedule_claim(run, None)?;
                    }
                }
                self.store.save_schedule(&schedule)?;
            }
        }
        Ok(())
    }

    fn publish(&self, event: CoreEvent) -> CoreResult<()> {
        self.store.save_event(&event)?;
        self.events.publish(event);
        Ok(())
    }
}

fn controller_draft_summaries(
    store: &LocalStore,
    system_id: Uuid,
    system_label: &str,
) -> CoreResult<Vec<serde_json::Value>> {
    Ok(store
        .controllers_for_system(system_id)?
        .into_iter()
        .filter(|record| {
            matches!(
                record.status,
                crate::agency::ControllerStatus::Draft | crate::agency::ControllerStatus::Reviewed
            )
        })
        .take(8)
        .map(|record| controller_draft_summary_json(&record, system_label, None))
        .collect())
}

fn controller_draft_summary_json(
    record: &crate::agency::StoredController,
    system_label: &str,
    task_id: Option<Uuid>,
) -> serde_json::Value {
    json!({
        "id": record.id,
        "system_id": record.controller.system_id,
        "system_label": system_label,
        "revision": record.revision,
        "status": record.status,
        "step_count": record.controller.steps.len(),
        "task_id": task_id,
    })
}

fn controller_draft_detail_json(
    record: &crate::agency::StoredController,
    system_label: &str,
    task_id: Option<Uuid>,
) -> serde_json::Value {
    let mut value = controller_draft_summary_json(record, system_label, task_id);
    let steps = record
        .controller
        .steps
        .iter()
        .map(|step| {
            json!({
                "id": step.id,
                "control_name": step.anchor.accessible_name,
                "role": step.anchor.role,
                "expected_effect": step.expected_effect,
                "verification": step.verification,
                "restoration": step.restoration,
            })
        })
        .collect::<Vec<_>>();
    if let Some(object) = value.as_object_mut() {
        object.insert("steps".into(), json!(steps));
    }
    value
}

fn default_tools() -> Vec<ToolDescriptor> {
    crate::features::descriptors()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;

    use async_trait::async_trait;
    use tempfile::tempdir;

    use crate::domain::{
        Action, ActionGraph, ActionNode, ActionProposal, Condition, ExpectedOutcome, Provenance,
    };
    use crate::events::CoreEventKind;
    use crate::model::{
        PlanningContext, ProviderDescriptor, ReplanContext, UnconfiguredModelProvider,
    };
    use crate::secrets::testing::MemorySecretStore;

    use super::*;

    #[tokio::test]
    async fn failed_procedure_dispatch_wave_unblocks_waiting_peers() {
        let wave = ProcedureDispatchWave::new(
            Uuid::from_u128(1),
            BTreeSet::from([Uuid::from_u128(2), Uuid::from_u128(3)]),
        );
        let mut completion = wave.completion_receiver();
        assert!(completion.borrow().is_none());

        let (aborted, submitted) = wave.abort(&CoreError::PermissionRequired(
            "A peer did not receive approval".into(),
        ));
        assert!(aborted);
        assert!(submitted.is_empty());
        wait_for_procedure_dispatch_wave(&mut completion).await;
        assert!(matches!(
            completion
                .borrow_and_update()
                .as_ref()
                .unwrap()
                .clone()
                .result(),
            Err(CoreError::PermissionRequired(_))
        ));
        assert!(!wave.abort(&CoreError::Cancelled).0);
    }

    fn learned_slider_assessment(
        application: &str,
        label: &str,
    ) -> (
        crate::world_model::SystemDescriptor,
        crate::world_model::CapabilityAssessment,
    ) {
        let now = Utc::now();
        let system = crate::world_model::SystemDescriptor {
            id: Uuid::new_v4(),
            kind: crate::world_model::SystemKind::Application,
            key: application.into(),
            label: "Example Editor".into(),
            fingerprint: "a".repeat(64),
            revision: 1,
            updated_at: now,
        };
        let control_id = "b".repeat(64);
        let observation = crate::world_model::ObservationEnvelope {
            id: Uuid::new_v4(),
            system_id: system.id,
            session_id: None,
            worker_session: None,
            system_fingerprint: system.fingerprint.clone(),
            origin: crate::world_model::EvidenceOrigin::OperatingSystem,
            privacy: crate::contracts::Sensitivity::Private,
            observed_at: now,
            facts: vec![
                crate::world_model::ObservedFact {
                    name: "control.enabled".into(),
                    subject: Some(control_id.clone()),
                    value: crate::world_model::FactValue::Boolean(true),
                },
                crate::world_model::ObservedFact {
                    name: "control.label".into(),
                    subject: Some(control_id.clone()),
                    value: crate::world_model::FactValue::Text(label.into()),
                },
                crate::world_model::ObservedFact {
                    name: "control.maximum".into(),
                    subject: Some(control_id.clone()),
                    value: crate::world_model::FactValue::Number(1.0),
                },
                crate::world_model::ObservedFact {
                    name: "control.minimum".into(),
                    subject: Some(control_id.clone()),
                    value: crate::world_model::FactValue::Number(0.0),
                },
                crate::world_model::ObservedFact {
                    name: "control.role".into(),
                    subject: Some(control_id.clone()),
                    value: crate::world_model::FactValue::Text("slider".into()),
                },
                crate::world_model::ObservedFact {
                    name: "control.step".into(),
                    subject: Some(control_id.clone()),
                    value: crate::world_model::FactValue::Number(0.1),
                },
                crate::world_model::ObservedFact {
                    name: "control.value".into(),
                    subject: Some(control_id.clone()),
                    value: crate::world_model::FactValue::Number(0.5),
                },
            ],
        };
        let mut descriptor = crate::world_model::passive_control_capability(
            &observation,
            &control_id,
            crate::world_model::ProbeKind::RestoreSliderValue,
        )
        .unwrap();
        descriptor.executor_id = Some("set_application_control".into());
        descriptor.preconditions.description =
            "Restored and verified control on the signed foreground application".into();
        descriptor.validate().unwrap();
        (
            system,
            crate::world_model::CapabilityAssessment {
                descriptor,
                evidence_state: crate::world_model::CapabilityEvidenceState::ReversiblyExperimented,
            },
        )
    }

    fn learned_slider_procedure(
        system: &crate::world_model::SystemDescriptor,
        assessment: &crate::world_model::CapabilityAssessment,
        value: f64,
        node_count: usize,
    ) -> crate::agency::ProcedureIr {
        let input = assessment.descriptor.input_ports[0].clone();
        let outputs: BTreeMap<String, crate::world_model::DataPort> = assessment
            .descriptor
            .output_ports
            .iter()
            .cloned()
            .map(|port| (port.name.clone(), port))
            .collect();
        crate::agency::ProcedureIr {
            schema_version: 1,
            id: "learned-control-goal".into(),
            nodes: (0..node_count)
                .map(|index| crate::agency::ProcedureNode {
                    id: format!("control-{}", index + 1),
                    depends_on: if index == 0 {
                        BTreeSet::new()
                    } else {
                        BTreeSet::from([format!("control-{index}")])
                    },
                    outputs: outputs.clone(),
                    kind: crate::agency::ProcedureNodeKind::CapabilityCall {
                        capability_id: assessment.descriptor.id.clone(),
                        system_id: system.id,
                        system_fingerprint: system.fingerprint.clone(),
                        input_bindings: BTreeMap::from([(
                            "value".into(),
                            crate::agency::ValueBinding::Literal {
                                value: crate::agency::ProcedureValue::Number(value),
                                port: input.clone(),
                            },
                        )]),
                    },
                })
                .collect(),
            streams: Vec::new(),
            completion: vec![crate::agency::CompletionCondition::AllNodesSucceeded],
        }
    }

    #[test]
    fn synthesized_control_procedure_compiles_to_normal_task_actions() {
        let task_id = Uuid::new_v4();
        let (system, assessment) = learned_slider_assessment("com.example.Editor", "Volume");
        let mut procedure = learned_slider_procedure(&system, &assessment, 0.7, 2);
        if let crate::agency::ProcedureNodeKind::CapabilityCall { input_bindings, .. } =
            &mut procedure.nodes[1].kind
        {
            input_bindings.insert(
                "value".into(),
                crate::agency::ValueBinding::Result {
                    producer: "control-1".into(),
                    output: "observed_value".into(),
                },
            );
        }
        let evidence_ids = assessment
            .descriptor
            .evidence_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut runtime =
            crate::agency::ProcedureRuntimeState::new_for_task(&procedure, task_id).unwrap();
        let first_advance = crate::agency::advance_procedure(
            &procedure,
            std::slice::from_ref(&assessment),
            &evidence_ids,
            &mut runtime,
            &BTreeMap::new(),
            4,
        )
        .unwrap();
        let first_node_ids = first_advance
            .proposal_wave
            .iter()
            .map(|proposal| proposal.node_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(first_node_ids, vec!["control-1"]);
        let first_graph = compile_application_control_procedure(
            &procedure,
            "Set the verified volume control twice in order",
            task_id,
            ProcedureCompilationContext {
                systems: std::slice::from_ref(&system),
                assessments: std::slice::from_ref(&assessment),
                current_evidence_ids: &evidence_ids,
                runtime: &runtime,
                selected_node_ids: &first_node_ids,
                existing_action_ids: &BTreeMap::new(),
            },
        )
        .unwrap();
        assert_eq!(first_graph.nodes.len(), 1);
        assert!(first_graph.nodes[0].depends_on.is_empty());
        let first_action_id = first_graph.nodes[0].proposal.id;
        for node in &first_graph.nodes {
            assert!(matches!(
                &node.proposal.action,
                Action::SetApplicationControl {
                    application,
                    system_id,
                    capability_id,
                    value: crate::domain::ApplicationControlValue::Number(value),
                    ..
                } if application == &system.key
                    && *system_id == system.id
                    && capability_id == &assessment.descriptor.id
                    && *value == 0.7
            ));
            assert_eq!(
                node.proposal.provenance.source,
                crate::domain::ProvenanceSource::SageCore
            );
            assert_eq!(
                node.proposal.provenance.trust,
                crate::domain::TrustClass::TrustedComponent
            );
            assert_eq!(node.proposal.metadata["procedure_id"], procedure.id);
        }
        let mut task = Task::new("Set the verified volume control twice in order");
        task.id = task_id;
        task.install_plan(first_graph).unwrap();
        validate_procedure_action_binding(&task, &procedure).unwrap();

        runtime
            .record_dispatched(
                &procedure,
                std::slice::from_ref(&assessment),
                &evidence_ids,
                "control-1",
                first_action_id,
            )
            .unwrap();
        runtime
            .record_verified_success(
                &procedure,
                "control-1",
                BTreeMap::from([(
                    "observed_value".into(),
                    crate::agency::ProcedureValue::Number(0.82),
                )]),
                Uuid::new_v4(),
            )
            .unwrap();
        let next_advance = crate::agency::advance_procedure(
            &procedure,
            std::slice::from_ref(&assessment),
            &evidence_ids,
            &mut runtime,
            &BTreeMap::new(),
            4,
        )
        .unwrap();
        let next_node_ids = next_advance
            .proposal_wave
            .iter()
            .map(|proposal| proposal.node_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(next_node_ids, vec!["control-2"]);
        let existing_action_ids = BTreeMap::from([("control-1".to_owned(), first_action_id)]);
        let next_graph = compile_application_control_procedure(
            &procedure,
            "Set the verified volume control twice in order",
            task_id,
            ProcedureCompilationContext {
                systems: std::slice::from_ref(&system),
                assessments: std::slice::from_ref(&assessment),
                current_evidence_ids: &evidence_ids,
                runtime: &runtime,
                selected_node_ids: &next_node_ids,
                existing_action_ids: &existing_action_ids,
            },
        )
        .unwrap();
        assert_eq!(next_graph.nodes.len(), 1);
        assert_eq!(
            next_graph.nodes[0].depends_on,
            BTreeSet::from([first_action_id])
        );
        assert!(matches!(
            &next_graph.nodes[0].proposal.action,
            Action::SetApplicationControl {
                value: crate::domain::ApplicationControlValue::Number(value),
                ..
            } if *value == 0.82
        ));
        task.append_plan_with_exact_dependencies(next_graph)
            .unwrap();
        validate_procedure_action_binding(&task, &procedure).unwrap();
    }

    #[test]
    fn synthesized_control_procedure_rejects_stale_or_unrestored_capabilities() {
        let task_id = Uuid::new_v4();
        let (system, mut assessment) = learned_slider_assessment("com.example.Editor", "Volume");
        let procedure = learned_slider_procedure(&system, &assessment, 0.7, 1);
        let empty = BTreeSet::new();
        assert!(
            compile_application_control_procedure(
                &procedure,
                "Set Volume to 0.7",
                task_id,
                ProcedureCompilationContext {
                    systems: std::slice::from_ref(&system),
                    assessments: std::slice::from_ref(&assessment),
                    current_evidence_ids: &empty,
                    runtime: &crate::agency::ProcedureRuntimeState::new_for_task(
                        &procedure, task_id,
                    )
                    .unwrap(),
                    selected_node_ids: &["control-1".into()],
                    existing_action_ids: &BTreeMap::new(),
                },
            )
            .is_err()
        );

        assessment.evidence_state = crate::world_model::CapabilityEvidenceState::PassivelyObserved;
        let evidence_ids = assessment
            .descriptor
            .evidence_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert!(
            compile_application_control_procedure(
                &procedure,
                "Set Volume to 0.7",
                task_id,
                ProcedureCompilationContext {
                    systems: std::slice::from_ref(&system),
                    assessments: std::slice::from_ref(&assessment),
                    current_evidence_ids: &evidence_ids,
                    runtime: &crate::agency::ProcedureRuntimeState::new_for_task(
                        &procedure, task_id,
                    )
                    .unwrap(),
                    selected_node_ids: &["control-1".into()],
                    existing_action_ids: &BTreeMap::new(),
                },
            )
            .is_err()
        );
    }

    #[test]
    fn exact_learned_control_command_compiles_only_one_verified_typed_target() {
        let task_id = Uuid::new_v4();
        let (system, assessment) = learned_slider_assessment("com.example.Editor", "Volume");
        let graph = build_learned_application_control_graph(
            "Set VOLUME to 0.7",
            task_id,
            vec![(system.clone(), assessment.clone())],
        )
        .unwrap()
        .unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert!(matches!(
            &graph.nodes[0].proposal.action,
            Action::SetApplicationControl {
                application,
                system_id,
                capability_id,
                value: crate::domain::ApplicationControlValue::Number(value),
                ..
            } if application == "com.example.Editor"
                && *system_id == system.id
                && capability_id == &assessment.descriptor.id
                && *value == 0.7
        ));
        assert!(matches!(
            graph.nodes[0].proposal.provenance.source,
            crate::domain::ProvenanceSource::User
        ));

        assert!(
            build_learned_application_control_graph(
                "Set Brightness to 0.7",
                task_id,
                vec![(system.clone(), assessment.clone())],
            )
            .unwrap()
            .is_none()
        );
        assert!(
            build_learned_application_control_graph(
                "Set Volume to on",
                task_id,
                vec![(system.clone(), assessment.clone())],
            )
            .is_err()
        );

        let mut passive = assessment.clone();
        passive.evidence_state = crate::world_model::CapabilityEvidenceState::PassivelyObserved;
        assert!(
            build_learned_application_control_graph(
                "Set Volume to 0.7",
                task_id,
                vec![(system.clone(), passive)],
            )
            .unwrap()
            .is_none()
        );
        let duplicate = learned_slider_assessment("com.other.Editor", "Volume");
        assert!(
            build_learned_application_control_graph(
                "Set Volume to 0.7",
                task_id,
                vec![(system, assessment), duplicate],
            )
            .is_err()
        );
    }

    #[test]
    fn browser_discovery_persists_only_bounded_private_semantics_and_stable_anchors() {
        let target = crate::browser_target::BrowserTarget {
            tab_id: 7,
            window_id: 4,
            frame_id: 0,
            document_id: "document-private-identity".into(),
            navigation_generation: 3,
            origin: "https://example.com".into(),
            url: "https://example.com/private/account?session=do-not-persist".into(),
        };
        let controls = vec![
            BrowserDiscoveryControl {
                role: "button".into(),
                kind: "button".into(),
                label: "Save".into(),
                enabled: true,
                ancestors: vec!["Editor".into()],
            },
            BrowserDiscoveryControl {
                role: "textbox".into(),
                kind: "textbox".into(),
                label: "person@example.com".into(),
                enabled: true,
                ancestors: vec![],
            },
            BrowserDiscoveryControl {
                role: "textbox".into(),
                kind: "textbox".into(),
                label: "Password".into(),
                enabled: true,
                ancestors: vec![],
            },
        ];
        let first =
            prepare_browser_discovery(target.clone(), controls.clone(), false, Utc::now()).unwrap();
        let mut changed = controls;
        changed[0].enabled = false;
        let second = prepare_browser_discovery(target, changed, false, Utc::now()).unwrap();

        assert_eq!(first.observed_controls, 1);
        assert_eq!(first.skipped_controls, 2);
        assert_eq!(
            first.system.kind,
            crate::world_model::SystemKind::BrowserOrigin
        );
        assert_eq!(first.system.key, "https://example.com");
        assert_ne!(first.interface_fingerprint, second.interface_fingerprint);
        assert_eq!(
            first.observation.facts.iter().find_map(|fact| {
                (fact.name == "browser.control.label")
                    .then_some(fact.subject.as_deref())
                    .flatten()
            }),
            second.observation.facts.iter().find_map(|fact| {
                (fact.name == "browser.control.label")
                    .then_some(fact.subject.as_deref())
                    .flatten()
            })
        );
        let serialized = serde_json::to_string(&first.observation).unwrap();
        assert!(serialized.contains("Save"));
        assert!(!serialized.contains("person@example.com"));
        assert!(!serialized.contains("Password"));
        assert!(!serialized.contains("private/account"));
        assert!(!serialized.contains("do-not-persist"));
        assert!(
            first
                .observation
                .facts
                .windows(2)
                .all(|pair| (&pair[0].name, &pair[0].subject) < (&pair[1].name, &pair[1].subject))
        );
    }

    #[test]
    fn application_discovery_keeps_passive_semantics_and_never_promotes_buttons_to_probes() {
        let controls = vec![
            ApplicationDiscoveryControl {
                id: "a".repeat(64),
                role: "button".into(),
                label: "Save".into(),
                enabled: true,
                ancestors: vec!["Document window".into(), "Toolbar".into()],
                value: Some(json!("private text must be ignored")),
                step: None,
                minimum: None,
                maximum: None,
            },
            ApplicationDiscoveryControl {
                id: "b".repeat(64),
                role: "slider".into(),
                label: "Brightness".into(),
                enabled: true,
                ancestors: vec!["Document window".into(), "Display".into()],
                value: Some(json!(0.5)),
                step: Some(0.1),
                minimum: Some(0.0),
                maximum: Some(1.0),
            },
            ApplicationDiscoveryControl {
                id: "c".repeat(64),
                role: "text_field".into(),
                label: "private.person@example.com".into(),
                enabled: true,
                ancestors: vec![],
                value: Some(json!("private.person@example.com")),
                step: None,
                minimum: None,
                maximum: None,
            },
        ];
        let prepared = prepare_application_controls(controls.clone()).unwrap();
        assert_eq!(prepared.semantic_controls.len(), 2);
        assert_eq!(prepared.skipped_controls, 1);
        assert!(!prepared.truncated);
        let button = prepared
            .semantic_controls
            .iter()
            .find(|control| control.role == "button")
            .unwrap();
        let ancestors: Vec<String> = serde_json::from_str(
            prepared
                .facts
                .iter()
                .find(|fact| {
                    fact.name == "application.control.ancestors"
                        && fact.subject.as_deref() == Some(button.id.as_str())
                })
                .and_then(|fact| match &fact.value {
                    crate::world_model::FactValue::Text(value) => Some(value.as_str()),
                    _ => None,
                })
                .unwrap(),
        )
        .unwrap();
        assert_eq!(ancestors, ["Document window", "Toolbar"]);
        assert!(!prepared.facts.iter().any(|fact| {
            fact.name == "control.value" && fact.subject.as_deref() == Some(button.id.as_str())
        }));

        let observation = crate::world_model::ObservationEnvelope {
            id: Uuid::new_v4(),
            system_id: Uuid::new_v4(),
            session_id: None,
            worker_session: None,
            system_fingerprint: "d".repeat(64),
            origin: crate::world_model::EvidenceOrigin::OperatingSystem,
            privacy: crate::contracts::Sensitivity::Private,
            observed_at: Utc::now(),
            facts: prepared.facts.clone(),
        };
        let candidates = crate::world_model::safe_probe_candidates(&observation);
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates.get(&"b".repeat(64)),
            Some(&crate::world_model::ProbeKind::RestoreSliderValue)
        );
        assert!(
            !serde_json::to_string(&prepared.facts)
                .unwrap()
                .contains("private.person@example.com")
        );

        let mut changed = controls;
        changed[0].ancestors[1] = "Navigation toolbar".into();
        let rebound = prepare_application_controls(changed).unwrap();
        assert_ne!(
            prepared.interface_fingerprint,
            rebound.interface_fingerprint
        );
    }

    #[test]
    fn application_discovery_drops_ambiguous_ids_and_stays_within_observation_budget() {
        let ambiguous = vec![
            ApplicationDiscoveryControl {
                id: "a".repeat(64),
                role: "button".into(),
                label: "Save".into(),
                enabled: true,
                ancestors: vec![],
                value: None,
                step: None,
                minimum: None,
                maximum: None,
            },
            ApplicationDiscoveryControl {
                id: "a".repeat(64),
                role: "button".into(),
                label: "Cancel".into(),
                enabled: true,
                ancestors: vec![],
                value: None,
                step: None,
                minimum: None,
                maximum: None,
            },
        ];
        let dropped = prepare_application_controls(ambiguous).unwrap();
        assert!(dropped.semantic_controls.is_empty());
        assert_eq!(dropped.skipped_controls, 2);

        let controls = (0..48)
            .map(|index| ApplicationDiscoveryControl {
                id: format!("{index:064x}"),
                role: "slider".into(),
                label: "B".repeat(256),
                enabled: true,
                ancestors: vec!["A".repeat(64); 4],
                value: Some(json!(0.5)),
                step: Some(0.1),
                minimum: Some(0.0),
                maximum: Some(1.0),
            })
            .collect();
        let prepared = prepare_application_controls(controls).unwrap();
        assert!(prepared.truncated);
        assert!(prepared.skipped_controls > 0);
        let mut facts = prepared.facts;
        facts.push(crate::world_model::ObservedFact {
            name: "application.interface_fingerprint".into(),
            subject: None,
            value: crate::world_model::FactValue::Identifier(prepared.interface_fingerprint),
        });
        facts.push(crate::world_model::ObservedFact {
            name: "application.bundle_identifier".into(),
            subject: None,
            value: crate::world_model::FactValue::Identifier("com.example.Editor".into()),
        });
        facts
            .sort_by(|left, right| (&left.name, &left.subject).cmp(&(&right.name, &right.subject)));
        let observation = crate::world_model::ObservationEnvelope {
            id: Uuid::new_v4(),
            system_id: Uuid::new_v4(),
            session_id: None,
            worker_session: None,
            system_fingerprint: "d".repeat(64),
            origin: crate::world_model::EvidenceOrigin::OperatingSystem,
            privacy: crate::contracts::Sensitivity::Private,
            observed_at: Utc::now(),
            facts,
        };
        assert!(serde_json::to_vec(&observation).unwrap().len() < 32 * 1024);
    }

    #[test]
    fn maximum_browser_discovery_observation_fits_world_model_storage() {
        let target = crate::browser_target::BrowserTarget {
            tab_id: 7,
            window_id: 4,
            frame_id: 0,
            document_id: "document-private-identity".into(),
            navigation_generation: 3,
            origin: "https://example.com".into(),
            url: "https://example.com/editor".into(),
        };
        let controls = (0..28)
            .map(|index| BrowserDiscoveryControl {
                role: "button".into(),
                kind: "button".into(),
                label: format!("{}{}", "x".repeat(158), index % 10),
                enabled: true,
                ancestors: vec!["A".repeat(64), "B".repeat(64)],
            })
            .collect();
        let prepared = prepare_browser_discovery(target, controls, true, Utc::now()).unwrap();
        let encoded = serde_json::to_vec(&prepared.observation).unwrap();
        assert!(encoded.len() < 32 * 1024, "{} bytes", encoded.len());
    }

    #[tokio::test]
    async fn paired_browser_discovery_stores_only_stable_private_facts() {
        use sage_protocol::sage::ipc::v2 as wire;

        let data = tempdir().unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let (requests, mut request_rx) = tokio::sync::mpsc::channel(8);
        let (terminate, _) = tokio::sync::watch::channel(false);
        core.adapters
            .register(
                "browser",
                "browser-worker-session",
                wire::ClientKind::Browser as i32,
                crate::execution::bridge::AdapterEndpoint {
                    requests,
                    cancellations: None,
                    terminate,
                },
                &[],
            )
            .await
            .unwrap();

        let inactive_core = Arc::clone(&core);
        let inactive = tokio::spawn(async move {
            inactive_core
                .world_model_command(wire::WorldModelCommand {
                    operation: "discover_paired_browser".into(),
                    ..Default::default()
                })
                .await
        });
        let request = request_rx.recv().await.unwrap();
        assert_eq!(request.operation, "discover_interface");
        core.adapters
            .complete(
                "browser-worker-session",
                wire::AdapterResult {
                    request_id: request.request_id,
                    success: true,
                    json: json!({
                        "available": false,
                        "reason": "paired_tab_not_active"
                    })
                    .to_string(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(inactive.await.unwrap().is_err());
        assert!(core.store.observed_systems().unwrap().is_empty());

        let active_core = Arc::clone(&core);
        let active = tokio::spawn(async move {
            active_core
                .world_model_command(wire::WorldModelCommand {
                    operation: "discover_paired_browser".into(),
                    ..Default::default()
                })
                .await
        });
        let request = request_rx.recv().await.unwrap();
        assert_eq!(request.operation, "discover_interface");
        let target = crate::browser_target::BrowserTarget {
            tab_id: 7,
            window_id: 4,
            frame_id: 0,
            document_id: "browser-document-id".into(),
            navigation_generation: 1,
            origin: "https://example.com".into(),
            url: "https://example.com/account/private?secret=never-store".into(),
        };
        core.adapters
            .complete(
                "browser-worker-session",
                wire::AdapterResult {
                    request_id: request.request_id,
                    success: true,
                    json: json!({
                        "available": true,
                        "reason": null,
                        "browser_target": target,
                        "truncated": false,
                        "controls": [
                            {"role":"button","kind":"button","label":"Save","enabled":true,"ancestors":["Editor"]},
                            {"role":"textbox","kind":"textbox","label":"private.person@example.com","enabled":true,"ancestors":[]},
                            {"role":"textbox","kind":"textbox","label":"Password","enabled":true,"ancestors":[]}
                        ]
                    })
                    .to_string(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let result = active.await.unwrap().unwrap();
        assert_eq!(result["observed_controls"], 1);
        assert_eq!(result["skipped_controls"], 2);
        assert_eq!(result["passive_capability_hypotheses"], 0);
        assert_eq!(result["execution_available"], false);
        let system: crate::world_model::SystemDescriptor =
            serde_json::from_value(result["system"].clone()).unwrap();
        let observations = core.store.observations_for_system(system.id, None).unwrap();
        assert_eq!(observations.len(), 1);
        assert!(
            core.store
                .capability_assessments(system.id)
                .unwrap()
                .is_empty()
        );
        let serialized = serde_json::to_string(&observations).unwrap();
        assert!(serialized.contains("Save"));
        assert!(!serialized.contains("private.person@example.com"));
        assert!(!serialized.contains("Password"));
        assert!(!serialized.contains("account/private"));
        assert!(!serialized.contains("never-store"));
    }

    #[tokio::test]
    async fn controller_draft_command_requires_a_verified_saved_procedure() {
        use sage_protocol::sage::ipc::v2 as wire;

        let data = tempdir().unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let task_id = Uuid::new_v4();
        let result = core
            .world_model_command(wire::WorldModelCommand {
                operation: "compile_controller_draft".into(),
                id: task_id.to_string(),
                ..Default::default()
            })
            .await;
        let error = result.expect_err("an absent verified procedure cannot be compiled");
        assert!(
            error
                .to_string()
                .contains("durably saved procedure checkpoint")
        );
        assert!(core.store.observed_systems().unwrap().is_empty());
    }

    #[tokio::test]
    async fn controller_review_scans_exact_target_and_records_no_execution_authority() {
        use sage_protocol::sage::ipc::v2 as wire;

        let data = tempdir().unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let code_digest = "ab".repeat(20);
        let target = crate::application_target::ApplicationTarget {
            platform: "macos".into(),
            bundle_path: "/Applications/Example.app".into(),
            identifier: "com.example.Editor".into(),
            code_digest: crate::application_target::ApplicationTarget::code_set_digest(
                std::slice::from_ref(&code_digest),
            ),
            code_digests: vec![code_digest],
            signer: "APPLE".into(),
        };
        let now = Utc::now();
        let system = core
            .store
            .observe_system(crate::world_model::SystemDescriptor {
                id: Uuid::nil(),
                kind: crate::world_model::SystemKind::Application,
                key: target.identifier.clone(),
                label: "Example Editor".into(),
                fingerprint: target.code_digest.clone(),
                revision: 0,
                updated_at: now,
            })
            .unwrap();
        let control_id = "bc".repeat(32);
        let prepared = prepare_application_controls(vec![ApplicationDiscoveryControl {
            id: control_id.clone(),
            role: "slider".into(),
            label: "Volume".into(),
            enabled: true,
            ancestors: vec!["Audio".into()],
            value: Some(json!(0.5)),
            step: Some(0.1),
            minimum: Some(0.0),
            maximum: Some(1.0),
        }])
        .unwrap();
        let evidence_id = Uuid::new_v4();
        let mut evidence_facts = prepared.facts.clone();
        evidence_facts.push(crate::world_model::ObservedFact {
            name: "application.interface_fingerprint".into(),
            subject: None,
            value: crate::world_model::FactValue::Identifier(
                prepared.interface_fingerprint.clone(),
            ),
        });
        evidence_facts
            .sort_by(|left, right| (&left.name, &left.subject).cmp(&(&right.name, &right.subject)));
        core.store
            .record_world_observation(&crate::world_model::ObservationEnvelope {
                id: evidence_id,
                system_id: system.id,
                session_id: None,
                worker_session: None,
                system_fingerprint: system.fingerprint.clone(),
                origin: crate::world_model::EvidenceOrigin::OperatingSystem,
                privacy: crate::contracts::Sensitivity::Private,
                observed_at: now,
                facts: evidence_facts,
            })
            .unwrap();
        let semantic_control = &prepared.semantic_controls[0];
        let controller = crate::agency::ControllerIr {
            schema_version: 1,
            system_id: system.id,
            system_fingerprint: system.fingerprint.clone(),
            interface_fingerprint: prepared.interface_fingerprint,
            steps: vec![crate::agency::ControllerStep {
                id: "set-volume".into(),
                primitive: crate::agency::ControllerPrimitive::ResolveSemanticAnchor,
                anchor: crate::agency::SemanticAnchor {
                    id: semantic_control.id.clone(),
                    role: semantic_control.role.clone(),
                    accessible_name: semantic_control.label.clone(),
                    ancestor_names: semantic_control.ancestors.clone(),
                },
                capability_id: None,
                input_bindings: BTreeMap::new(),
                depends_on: BTreeSet::new(),
                outputs: BTreeMap::new(),
                preconditions: Vec::new(),
                expected_effect: "Set the volume slider to the requested value".into(),
                verification: "Read back the exact volume value".into(),
                restoration: Some("Restore the previous volume value".into()),
                evidence_ids: vec![evidence_id],
            }],
        };
        let draft = core.store.save_controller_draft(&controller).unwrap();

        let (requests, mut request_rx) = tokio::sync::mpsc::channel(8);
        let (terminate, _) = tokio::sync::watch::channel(false);
        core.adapters
            .register(
                "native",
                "native-controller-review-session",
                wire::ClientKind::Macos as i32,
                crate::execution::bridge::AdapterEndpoint {
                    requests,
                    cancellations: None,
                    terminate,
                },
                &[],
            )
            .await
            .unwrap();

        let review_core = Arc::clone(&core);
        let review = tokio::spawn(async move {
            review_core
                .world_model_command(wire::WorldModelCommand {
                    operation: "review_controller_draft".into(),
                    id: draft.id,
                    json: json!({ "expected_revision": 1 }).to_string(),
                })
                .await
        });
        let request = request_rx.recv().await.unwrap();
        assert_eq!(request.operation, "discover_interface");
        core.adapters
            .complete(
                "native-controller-review-session",
                wire::AdapterResult {
                    request_id: request.request_id,
                    success: true,
                    json: json!({
                        "application_target": target.clone(),
                        "observed_process_id": 42,
                        "application_name": "Example Editor",
                        "bundle_identifier": target.identifier.clone(),
                        "accessibility_available": true,
                        "active_window": "Editor",
                        "truncated": false,
                        "controls": [{
                            "id": control_id,
                            "role": "slider",
                            "label": "Volume",
                            "enabled": true,
                            "ancestors": ["Audio"],
                            "value": 0.5,
                            "step": 0.1,
                            "minimum": 0.0,
                            "maximum": 1.0
                        }]
                    })
                    .to_string(),
                    application_target: Some(target.to_wire()),
                    observed_process_id: 42,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let result = review.await.unwrap().unwrap();
        assert_eq!(result["controller_reviewed"], true);
        assert_eq!(result["fresh_rebind_verified"], true);
        assert_eq!(result["grants_execution_authority"], false);
        assert_eq!(result["execution_available"], false);
        assert_eq!(result["controller_draft"]["status"], "reviewed");
        assert_eq!(result["controller_draft"]["revision"], 2);
        assert_eq!(
            result["controller_draft"]["steps"][0]["control_name"],
            "Volume"
        );
        assert!(request_rx.try_recv().is_err());
        let reviewed_id = result["controller_draft"]["id"].as_str().unwrap();
        let stored = core.store.load_controller(reviewed_id).unwrap().unwrap();
        assert_eq!(stored.status, crate::agency::ControllerStatus::Reviewed);
        assert_eq!(stored.revision, 2);
    }

    #[tokio::test]
    async fn late_probe_receipt_settles_only_its_dispatched_approval_lease() {
        use sage_protocol::sage::ipc::v2 as wire;

        let data = tempdir().unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let now = Utc::now();
        let code_digest = "ab".repeat(20);
        let target = crate::application_target::ApplicationTarget {
            platform: "macos".into(),
            bundle_path: "/Applications/Example.app".into(),
            identifier: "com.example.Editor".into(),
            code_digest: crate::application_target::ApplicationTarget::code_set_digest(
                std::slice::from_ref(&code_digest),
            ),
            code_digests: vec![code_digest],
            signer: "APPLE".into(),
        };
        let system = core
            .store
            .observe_system(crate::world_model::SystemDescriptor {
                id: Uuid::nil(),
                kind: crate::world_model::SystemKind::Application,
                key: target.identifier.clone(),
                label: "Example Editor".into(),
                fingerprint: target.code_digest.clone(),
                revision: 0,
                updated_at: now,
            })
            .unwrap();
        let control_id = "c".repeat(64);
        core.store
            .record_world_observation(&crate::world_model::ObservationEnvelope {
                id: Uuid::new_v4(),
                system_id: system.id,
                session_id: None,
                worker_session: None,
                system_fingerprint: target.code_digest.clone(),
                origin: crate::world_model::EvidenceOrigin::OperatingSystem,
                privacy: crate::contracts::Sensitivity::Private,
                observed_at: now,
                facts: vec![
                    crate::world_model::ObservedFact {
                        name: "control.enabled".into(),
                        subject: Some(control_id.clone()),
                        value: crate::world_model::FactValue::Boolean(true),
                    },
                    crate::world_model::ObservedFact {
                        name: "control.label".into(),
                        subject: Some(control_id.clone()),
                        value: crate::world_model::FactValue::Text("Volume".into()),
                    },
                    crate::world_model::ObservedFact {
                        name: "control.maximum".into(),
                        subject: Some(control_id.clone()),
                        value: crate::world_model::FactValue::Number(1.0),
                    },
                    crate::world_model::ObservedFact {
                        name: "control.minimum".into(),
                        subject: Some(control_id.clone()),
                        value: crate::world_model::FactValue::Number(0.0),
                    },
                    crate::world_model::ObservedFact {
                        name: "control.role".into(),
                        subject: Some(control_id.clone()),
                        value: crate::world_model::FactValue::Text("slider".into()),
                    },
                    crate::world_model::ObservedFact {
                        name: "control.step".into(),
                        subject: Some(control_id.clone()),
                        value: crate::world_model::FactValue::Number(0.1),
                    },
                    crate::world_model::ObservedFact {
                        name: "control.value".into(),
                        subject: Some(control_id.clone()),
                        value: crate::world_model::FactValue::Number(0.5),
                    },
                ],
            })
            .unwrap();
        let worker_session = "native-probe-worker";
        let session = core
            .store
            .approve_learning_session(
                system.id,
                &target.code_digest,
                worker_session,
                BTreeMap::from([(
                    control_id.clone(),
                    crate::world_model::ProbeKind::RestoreSliderValue,
                )]),
                now + ChronoDuration::minutes(10),
            )
            .unwrap();
        let lease = core
            .store
            .begin_probe(
                session.id,
                worker_session,
                &target.code_digest,
                &control_id,
                42,
                Utc::now(),
            )
            .unwrap();
        core.native_adapter_disconnected(worker_session).unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        let observations = [0.5, 0.6, 0.5]
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                json!({
                    "value": value,
                    "observed_at": (lease.issued_at + ChronoDuration::milliseconds(index as i64 + 1)).to_rfc3339(),
                })
            })
            .collect::<Vec<_>>();
        let response = wire::AdapterResult {
            request_id: lease.id.to_string(),
            success: true,
            json: json!({
                "control_id": control_id,
                "restoration_verified": true,
                "observations": observations,
            })
            .to_string(),
            application_target: Some(target.to_wire()),
            observed_process_id: 42,
            ..Default::default()
        };
        let mut wrong_process_response = response.clone();
        wrong_process_response.observed_process_id = 43;
        assert!(
            core.record_late_adapter_result(crate::execution::bridge::LateAdapterResult {
                session: worker_session.into(),
                binding: None,
                never_sent: false,
                response: wrong_process_response,
            })
            .await
            .is_err()
        );
        assert_eq!(
            core.store
                .dispatched_probe_lease(lease.id)
                .unwrap()
                .unwrap()
                .expected_process_id,
            42
        );
        core.record_late_adapter_result(crate::execution::bridge::LateAdapterResult {
            session: worker_session.into(),
            binding: None,
            never_sent: false,
            response,
        })
        .await
        .unwrap();

        assert_eq!(
            core.store
                .learning_session(session.id)
                .unwrap()
                .unwrap()
                .state,
            crate::world_model::LearningSessionState::InterruptedNeedsReview
        );
        assert!(
            core.store
                .dispatched_probe_lease(lease.id)
                .unwrap()
                .is_none()
        );
        assert!(
            core.store
                .begin_probe(
                    session.id,
                    worker_session,
                    &target.code_digest,
                    &control_id,
                    42,
                    Utc::now(),
                )
                .is_err()
        );
    }

    struct FolderPlanProvider {
        path: PathBuf,
    }

    struct CaptureContextProvider {
        captured: tokio::sync::mpsc::UnboundedSender<PlanningContext>,
    }

    #[async_trait]
    impl ModelProvider for CaptureContextProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            ProviderDescriptor {
                id: "fixture-local-context".into(),
                display_name: "Fixture planner".into(),
                local: true,
                roles: vec![crate::model::ModelRole::Reasoning],
            }
        }

        async fn create_plan(&self, context: PlanningContext) -> CoreResult<ActionGraph> {
            let _ = self.captured.send(context);
            Err(CoreError::Model(
                "Fixture ends after context capture".into(),
            ))
        }

        async fn replan(&self, _context: ReplanContext) -> CoreResult<ActionGraph> {
            Err(CoreError::Model("Fixture does not replan".into()))
        }
    }

    #[tokio::test]
    async fn explicit_foreground_references_are_fresh_task_local_and_visible() {
        use sage_protocol::sage::ipc::v2 as wire;

        let data = tempdir().unwrap();
        let (captured, mut contexts) = tokio::sync::mpsc::unbounded_channel();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(CaptureContextProvider { captured }),
        )
        .unwrap();
        let mut events = core.events().subscribe();
        let (native_tx, mut native_rx) = tokio::sync::mpsc::channel(8);
        let (browser_tx, mut browser_rx) = tokio::sync::mpsc::channel(8);
        let (native_terminate, _) = tokio::sync::watch::channel(false);
        let (browser_terminate, _) = tokio::sync::watch::channel(false);
        core.adapters
            .register(
                "native",
                "reference-native-fixture",
                wire::ClientKind::Macos as i32,
                crate::execution::bridge::AdapterEndpoint {
                    requests: native_tx,
                    cancellations: None,
                    terminate: native_terminate,
                },
                &[],
            )
            .await
            .unwrap();
        core.adapters
            .register(
                "browser",
                "reference-browser-fixture",
                wire::ClientKind::Browser as i32,
                crate::execution::bridge::AdapterEndpoint {
                    requests: browser_tx,
                    cancellations: None,
                    terminate: browser_terminate,
                },
                &[],
            )
            .await
            .unwrap();
        let native_bridge = core.adapters.clone();
        let native_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let native_count = native_calls.clone();
        let native = tokio::spawn(async move {
            while let Some(request) = native_rx.recv().await {
                native_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                assert_eq!(request.operation, "reference");
                native_bridge
                    .complete(
                        "reference-native-fixture",
                        wire::AdapterResult {
                            request_id: request.request_id,
                            success: true,
                            json: serde_json::json!({
                                "available": true,
                                "active_application": "com.apple.finder",
                                "application_name": "Finder",
                                "active_window": "Downloads",
                                "selected_text": "fixture selection that must not enter the UI summary"
                            })
                            .to_string(),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
            }
        });
        let browser_bridge = core.adapters.clone();
        let browser_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let browser_count = browser_calls.clone();
        let browser = tokio::spawn(async move {
            while let Some(request) = browser_rx.recv().await {
                browser_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                assert_eq!(request.operation, "reference");
                browser_bridge
                    .complete(
                        "reference-browser-fixture",
                        wire::AdapterResult {
                            request_id: request.request_id,
                            success: true,
                            json: serde_json::json!({"available": false}).to_string(),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
            }
        });

        let explicit_id = core.submit_task("Summarize this selection").await.unwrap();
        let explicit = timeout(Duration::from_secs(2), contexts.recv())
            .await
            .unwrap()
            .unwrap();
        let reference = explicit
            .untrusted_context
            .iter()
            .find(|item| item.source == "current_reference_native")
            .expect("the explicitly requested foreground selection is included");
        assert!(reference.content.contains("fixture selection"));
        let stored = core.get_task(explicit_id).await.unwrap();
        assert!(stored.reference_captured);
        assert!(
            !serde_json::to_string(&stored)
                .unwrap()
                .contains("fixture selection")
        );
        let summary = timeout(Duration::from_secs(2), async {
            loop {
                if let Ok(event) = events.recv().await
                    && let CoreEventKind::ReferenceContext { summary } = event.kind
                    && summary.starts_with("Using a current reference")
                {
                    break summary;
                }
            }
        })
        .await
        .unwrap();
        assert!(summary.contains("Finder"));
        assert!(!summary.contains("fixture selection"));

        core.submit_task("What should I do next?").await.unwrap();
        let ordinary = timeout(Duration::from_secs(2), contexts.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(
            ordinary
                .untrusted_context
                .iter()
                .all(|item| !item.source.starts_with("current_reference"))
        );
        tokio::task::yield_now().await;
        assert_eq!(native_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(browser_calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        native.abort();
        browser.abort();
    }

    #[tokio::test]
    async fn unconfigured_planner_does_not_read_a_foreground_reference() {
        use sage_protocol::sage::ipc::v2 as wire;

        let data = tempdir().unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let (native_tx, mut native_rx) = tokio::sync::mpsc::channel(4);
        core.adapters
            .register(
                "native",
                "unconfigured-reference-fixture",
                wire::ClientKind::Macos as i32,
                crate::execution::bridge::AdapterEndpoint {
                    requests: native_tx,
                    cancellations: None,
                    terminate: tokio::sync::watch::channel(false).0,
                },
                &[],
            )
            .await
            .unwrap();
        let id = core.submit_task("Summarize this selection").await.unwrap();
        timeout(Duration::from_secs(2), async {
            loop {
                if core.get_task(id).await.unwrap().status.is_terminal() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert!(native_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn queued_correction_does_not_block_stop_admission_on_a_locked_database() {
        use sage_protocol::sage::ipc::v2 as wire;
        let data = tempdir().unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let store = core.store.clone();
        let (entered, ready) = std::sync::mpsc::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            store
                .with_connection(|_| {
                    entered.send(()).unwrap();
                    blocked.recv().unwrap();
                    Ok(())
                })
                .unwrap()
        });
        ready.recv_timeout(Duration::from_secs(1)).unwrap();
        let (responses, _receiver) = tokio::sync::mpsc::channel(4);
        let dispatcher = crate::ipc::dispatch::CommandDispatcher::new(core, responses);
        let (finished, admitted) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let task_id = Uuid::new_v4().to_string();
            dispatcher
                .enqueue(wire::UiCommand {
                    request_id: Uuid::new_v4().to_string(),
                    command: Some(wire::ui_command::Command::SubmitTask(wire::SubmitTask {
                        text: "open safari".into(),
                        supersedes_task_id: task_id.clone(),
                        ..Default::default()
                    })),
                })
                .unwrap();
            dispatcher
                .enqueue(wire::UiCommand {
                    request_id: Uuid::new_v4().to_string(),
                    command: Some(wire::ui_command::Command::ControlTask(wire::ControlTask {
                        task_id,
                        operation: wire::control_task::Operation::Cancel as i32,
                    })),
                })
                .unwrap();
            finished.send(()).unwrap();
        });
        let result = admitted.recv_timeout(Duration::from_secs(1));
        release.send(()).unwrap();
        holder.join().unwrap();
        reader.join().unwrap();
        assert!(
            result.is_ok(),
            "Admission must finish before the database is released"
        );
    }

    #[tokio::test]
    async fn accepted_unlock_finishes_hydration_after_its_waiter_disconnects() {
        struct DelayedSecretStore {
            memory: MemorySecretStore,
            entered: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
            release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl SecretStore for DelayedSecretStore {
            fn get(&self, account: &str) -> CoreResult<Option<SecretBytes>> {
                if account == "database-v2"
                    && let Some(entered) = self.entered.lock().unwrap().take()
                {
                    let _ = entered.send(());
                    self.release.lock().unwrap().recv().unwrap();
                }
                self.memory.get(account)
            }
            fn set(&self, account: &str, secret: &SecretBytes) -> CoreResult<()> {
                self.memory.set(account, secret)
            }
            fn delete(&self, account: &str) -> CoreResult<()> {
                self.memory.delete(account)
            }
        }
        let data = tempdir().unwrap();
        let (entered, wait) = tokio::sync::oneshot::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        let secrets = Arc::new(DelayedSecretStore {
            memory: MemorySecretStore::default(),
            entered: std::sync::Mutex::new(Some(entered)),
            release: std::sync::Mutex::new(blocked),
        });
        secrets
            .set("database-v2", &SecretBytes::new(vec![17; 32]))
            .unwrap();
        let mut core = SageCore::new_with_secret_store(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
            secrets,
        )
        .unwrap();
        let mut stored = Task::new("Previously saved fixture");
        core.store.save_task(&mut stored).unwrap();
        let mutable = Arc::get_mut(&mut core).unwrap();
        mutable.store = LocalStore::deferred(&mutable.config.database_path).unwrap();
        mutable.store.migrate_knowledge().unwrap();
        mutable.store.migrate_workflows().unwrap();
        mutable
            .storage_hydrated
            .store(false, std::sync::atomic::Ordering::Release);
        let waiting = core.clone();
        let request = tokio::spawn(async move { waiting.unlock_storage().await });
        wait.await.unwrap();
        request.abort();
        // The single-threaded runtime still answers local preparation while
        // credential access is blocked on its dedicated worker.
        use sage_protocol::sage::ipc::v2 as wire;
        let (responses, mut receiver) = tokio::sync::mpsc::channel(4);
        let dispatcher = crate::ipc::dispatch::CommandDispatcher::new(core.clone(), responses);
        dispatcher
            .enqueue(wire::UiCommand {
                request_id: Uuid::new_v4().to_string(),
                command: Some(wire::ui_command::Command::UnlockStorage(
                    wire::UnlockStorage {},
                )),
            })
            .unwrap();
        dispatcher
            .enqueue(wire::UiCommand {
                request_id: Uuid::new_v4().to_string(),
                command: Some(wire::ui_command::Command::UpdateIntent(
                    wire::UpdateIntent {
                        stream_id: Uuid::new_v4().to_string(),
                        revision: 1,
                        text: "open safari".into(),
                        ..Default::default()
                    },
                )),
            })
            .unwrap();
        let preview = timeout(Duration::from_millis(250), receiver.recv()).await;
        release.send(()).unwrap();
        assert!(matches!(
            preview.unwrap(),
            Some(wire::frame::Payload::CoreEvent(wire::CoreEvent {
                event: Some(wire::core_event::Event::IntentPreview(_)),
                ..
            }))
        ));
        timeout(Duration::from_secs(3), core.unlock_storage())
            .await
            .unwrap()
            .unwrap();
        assert!(
            core.storage_hydrated
                .load(std::sync::atomic::Ordering::Acquire)
        );
        assert_eq!(
            core.get_task(stored.id).await.unwrap().request,
            stored.request
        );
    }

    #[tokio::test]
    async fn local_intent_reads_real_files_without_a_model_and_keeps_scope_checks() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let root = workspace.path().canonicalize().unwrap();
        let path = root.join("local.txt");
        std::fs::write(&path, "Verified local command content").unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let request = format!("read \"{}\"", path.display());
        let id = core
            .submit_scoped(
                request.clone(),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root,
                    effects: BTreeSet::from([crate::contracts::Effect::Read]),
                }],
            )
            .await
            .unwrap();
        let task = completed(&core, id).await;
        assert_eq!(
            task.status,
            TaskStatus::Succeeded,
            "{:?}",
            task.final_outcome
        );
        assert!(
            task.final_outcome
                .unwrap()
                .contains("Verified local command content")
        );
        assert!(
            task.actions
                .values()
                .all(|a| a.proposal.metadata.contains_key("intent_compiler"))
        );

        // The same grammar cannot infer access from a pathname in the text.
        let id = core.submit_task(request).await.unwrap();
        let approval = timeout(Duration::from_secs(2), async {
            loop {
                if let Some(approval) = core
                    .snapshot(true)
                    .await
                    .pending_approvals
                    .into_iter()
                    .find(|p| p.task_id == id)
                {
                    break approval;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        core.resolve_approval(
            approval.approval_id,
            id,
            approval.action_id,
            &approval.digest,
            ApprovalResolution::Denied,
        )
        .await
        .unwrap();
        let denied = completed(&core, id).await;
        assert_eq!(denied.completed_count(), 0);
    }

    struct ReadRoutineProvider {
        path: PathBuf,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl ModelProvider for ReadRoutineProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            ProviderDescriptor {
                id: "local-routine-fixture".into(),
                display_name: "Local routine fixture".into(),
                local: true,
                roles: vec![crate::model::ModelRole::Reasoning],
            }
        }

        async fn create_plan(&self, context: PlanningContext) -> CoreResult<ActionGraph> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ActionGraph {
                goal: "Read the authorized notes".into(),
                nodes: vec![ActionNode {
                    proposal: ActionProposal {
                        id: Uuid::new_v4(),
                        task_id: context.task_id,
                        action: Action::ReadFile {
                            path: self.path.clone(),
                            max_bytes: 1024,
                        },
                        expected_outcome: ExpectedOutcome::UserAnswered,
                        target_resource: self.path.to_string_lossy().into_owned(),
                        provenance: Provenance::model(Vec::new()),
                        metadata: BTreeMap::new(),
                    },
                    depends_on: BTreeSet::new(),
                }],
            })
        }

        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!("the one-step fixture needs no replan")
        }
    }

    #[tokio::test]
    async fn reviewed_local_routines_need_verified_examples_and_fresh_run_scope() {
        use sage_protocol::sage::ipc::v2::WorkflowCommand;

        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let root = workspace.path().canonicalize().unwrap();
        let path = root.join("routine-notes.txt");
        std::fs::write(&path, "verified routine fixture").unwrap();
        let provider = Arc::new(ReadRoutineProvider {
            path: path.clone(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let core = SageCore::new(CoreConfig::for_test(data.path()), provider.clone()).unwrap();
        core.workflow_command(WorkflowCommand {
            operation: "configure_learning".into(),
            json: r#"{"enabled":true}"#.into(),
            ..Default::default()
        })
        .await
        .unwrap();

        let request = "review the project notes";
        let scope = crate::contracts::ResourceScope {
            root,
            effects: BTreeSet::from([crate::contracts::Effect::Read]),
        };
        let mut observed_action_ids = BTreeSet::new();
        for sample in 0..3 {
            // "please" is a superficial variant; it remains a single exact
            // locally observed request alias.
            let observed_request = if sample == 2 {
                "please review the project notes"
            } else {
                request
            };
            let task_id = core
                .submit_scoped(
                    observed_request.into(),
                    None,
                    false,
                    None,
                    vec![scope.clone()],
                )
                .await
                .unwrap();
            let task = completed(&core, task_id).await;
            assert_eq!(
                task.status,
                TaskStatus::Succeeded,
                "{:?}",
                task.final_outcome
            );
            assert!(task.actions.values().any(|action| {
                matches!(action.proposal.action, Action::ReadFile { .. })
                    && action.status == ActionStatus::Succeeded
            }));
            observed_action_ids.extend(task.actions.keys().copied());
        }
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 3);

        let snapshot = core.workflow_snapshot().unwrap();
        let routine = snapshot["routines"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["ready"] == true)
            .expect("three verified runs should surface a reviewable routine");
        assert_eq!(routine["verified_runs"], 3);
        assert_eq!(routine["requests"].as_array().unwrap().len(), 1);
        let routine_id = routine["id"].as_str().unwrap().to_owned();
        let review_digest = routine["review_digest"].as_str().unwrap().to_owned();
        core.workflow_command(WorkflowCommand {
            operation: "review_routine".into(),
            id: routine_id.clone(),
            json: serde_json::json!({"digest": review_digest}).to_string(),
            ..Default::default()
        })
        .await
        .unwrap();

        if std::env::var_os("SAGE_BENCH_ROUTINE_LOOKUP").is_some() {
            let mut samples = Vec::with_capacity(10_000);
            for index in 0..10_500 {
                let start = std::time::Instant::now();
                let found = core
                    .store
                    .compiled_routine(std::hint::black_box(request))
                    .unwrap();
                assert_eq!(
                    found.as_ref().map(|(id, _)| id.as_str()),
                    Some(routine_id.as_str())
                );
                std::hint::black_box(found);
                if index >= 500 {
                    samples.push(start.elapsed().as_nanos());
                }
            }
            samples.sort_unstable();
            println!(
                "reviewed_routine_lookup samples={} p50_ns={} p95_ns={} p99_ns={}; excludes UI, IPC, folder resolution, approval, broker, and effects",
                samples.len(),
                samples[5_000],
                samples[9_500],
                samples[9_900]
            );
        }

        // This provider can plan only by incrementing its fixture counter.
        // The unchanged counter proves the approved local pattern bypassed it.
        let reused = core
            .submit_scoped(request.into(), None, false, None, vec![scope])
            .await
            .unwrap();
        let reused_task = completed(&core, reused).await;
        assert_eq!(reused_task.status, TaskStatus::Succeeded);
        assert_eq!(
            reused_task.compiled_routine.as_deref(),
            Some(routine_id.as_str())
        );
        assert!(
            reused_task
                .actions
                .keys()
                .all(|id| !observed_action_ids.contains(id))
        );
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 3);
        assert_eq!(
            core.workflow_snapshot().unwrap()["routines"][0]["verified_runs"],
            3
        );

        // A reviewed action template carries no resource grant. Without a
        // fresh folder scope, the same request pauses at the normal approval
        // gate instead of silently reusing the examples' access.
        let pending = core.submit_task(request).await.unwrap();
        let pending_approval = timeout(Duration::from_secs(2), async {
            loop {
                if let Some(approval) = core
                    .snapshot(true)
                    .await
                    .pending_approvals
                    .into_iter()
                    .find(|approval| approval.task_id == pending)
                {
                    break approval;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("learned routine must ask for new access");
        assert!(!pending_approval.explanation.is_empty());
        assert_eq!(
            core.get_task(pending)
                .await
                .unwrap()
                .compiled_routine
                .as_deref(),
            Some(routine_id.as_str())
        );

        let malformed = core
            .workflow_command(WorkflowCommand {
                operation: "configure_learning".into(),
                json: "{".into(),
                ..Default::default()
            })
            .await;
        assert!(malformed.is_err());
        assert!(core.get_task(pending).await.unwrap().status.is_active());
        core.workflow_command(WorkflowCommand {
            operation: "configure_learning".into(),
            json: r#"{"enabled":true}"#.into(),
            ..Default::default()
        })
        .await
        .unwrap();
        assert!(core.get_task(pending).await.unwrap().status.is_active());
        core.workflow_command(WorkflowCommand {
            operation: "configure_learning".into(),
            json: r#"{"enabled":false}"#.into(),
            ..Default::default()
        })
        .await
        .unwrap();
        assert_eq!(
            completed(&core, pending).await.status,
            TaskStatus::Cancelled
        );
        core.workflow_command(WorkflowCommand {
            operation: "forget_routine".into(),
            id: routine_id,
            ..Default::default()
        })
        .await
        .unwrap();
        assert!(
            core.workflow_snapshot().unwrap()["routines"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    struct BranchingRoutineProvider {
        shared: PathBuf,
        summary: PathBuf,
        brief: PathBuf,
        memo: PathBuf,
        plans: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl ModelProvider for BranchingRoutineProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            ProviderDescriptor {
                id: "local-branch-fixture".into(),
                display_name: "Local branch fixture".into(),
                local: true,
                roles: vec![crate::model::ModelRole::Reasoning],
            }
        }

        async fn create_plan(&self, context: PlanningContext) -> CoreResult<ActionGraph> {
            self.plans.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let shared_id = Uuid::new_v4();
            let branch_id = Uuid::new_v4();
            let branch = if context.user_request.contains("summary") {
                self.summary.clone()
            } else if context.user_request.contains("memo") {
                self.memo.clone()
            } else {
                self.brief.clone()
            };
            let node = |id, path: PathBuf, depends_on| ActionNode {
                proposal: ActionProposal {
                    id,
                    task_id: context.task_id,
                    action: Action::ReadFile {
                        path: path.clone(),
                        max_bytes: 1024,
                    },
                    expected_outcome: ExpectedOutcome::UserAnswered,
                    target_resource: path.to_string_lossy().into_owned(),
                    provenance: Provenance::model(Vec::new()),
                    metadata: BTreeMap::new(),
                },
                depends_on,
            };
            Ok(ActionGraph {
                goal: "Prepare the requested document".into(),
                nodes: vec![
                    node(shared_id, self.shared.clone(), BTreeSet::new()),
                    node(branch_id, branch, BTreeSet::from([shared_id])),
                ],
            })
        }

        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!("the verified branch fixture does not need repair")
        }
    }

    #[tokio::test]
    async fn reviewed_patterns_synthesize_a_multi_branch_prefix_skill_and_forget_its_sources() {
        use sage_protocol::sage::ipc::v2::WorkflowCommand;

        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let root = workspace.path().canonicalize().unwrap();
        let shared = root.join("shared.txt");
        let summary = root.join("summary.txt");
        let brief = root.join("brief.txt");
        let memo = root.join("memo.txt");
        std::fs::write(&shared, "shared verified step").unwrap();
        std::fs::write(&summary, "summary branch").unwrap();
        std::fs::write(&brief, "brief branch").unwrap();
        std::fs::write(&memo, "memo branch").unwrap();
        let provider = Arc::new(BranchingRoutineProvider {
            shared: shared.clone(),
            summary,
            brief,
            memo,
            plans: std::sync::atomic::AtomicUsize::new(0),
        });
        let core = SageCore::new(CoreConfig::for_test(data.path()), provider.clone()).unwrap();
        let scope = crate::contracts::ResourceScope {
            root,
            effects: BTreeSet::from([crate::contracts::Effect::Read]),
        };
        core.workflow_command(WorkflowCommand {
            operation: "configure_learning".into(),
            json: r#"{"enabled":true}"#.into(),
            ..Default::default()
        })
        .await
        .unwrap();

        for (request, expected_suffix) in [
            ("prepare the monthly summary", "summary"),
            ("prepare the customer brief", "brief"),
            ("prepare the quarterly memo", "memo"),
        ] {
            for _ in 0..3 {
                let id = core
                    .submit_scoped(request.into(), None, false, None, vec![scope.clone()])
                    .await
                    .unwrap();
                let task = completed(&core, id).await;
                assert_eq!(
                    task.status,
                    TaskStatus::Succeeded,
                    "{:?}",
                    task.final_outcome
                );
                assert!(
                    task.final_outcome
                        .as_deref()
                        .unwrap_or_default()
                        .contains(expected_suffix)
                );
            }
        }
        assert_eq!(provider.plans.load(std::sync::atomic::Ordering::SeqCst), 9);

        let workflows = core.workflow_snapshot().unwrap();
        let candidates = workflows["routines"].as_array().unwrap();
        for routine in candidates {
            core.workflow_command(WorkflowCommand {
                operation: "review_routine".into(),
                id: routine["id"].as_str().unwrap().into(),
                json: serde_json::json!({"digest": routine["review_digest"]}).to_string(),
                ..Default::default()
            })
            .await
            .unwrap();
        }
        let family = core.workflow_snapshot().unwrap()["routine_families"]
            .as_array()
            .unwrap()
            .iter()
            .find(|family| {
                family["shared_steps"]
                    .as_array()
                    .is_some_and(|steps| steps.len() == 1)
            })
            .cloned()
            .expect("reviewed branches should expose their stable first step");
        assert_eq!(family["verified_runs"], 9);
        assert_eq!(family["branches"].as_array().unwrap().len(), 3);
        assert!(family["branches"].as_array().unwrap().iter().all(|branch| {
            branch["next_steps"]
                .as_array()
                .is_some_and(|steps| steps.len() == 1)
        }));
        core.workflow_command(WorkflowCommand {
            operation: "synthesize_skill".into(),
            id: family["id"].as_str().unwrap().into(),
            name: "Shared document preparation".into(),
            json: serde_json::json!({"digest": family["review_digest"]}).to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
        let draft = core
            .store
            .skills()
            .unwrap()
            .into_iter()
            .find(|skill| !skill.synthesized_from.is_empty())
            .expect("synthesis should save a reviewable skill draft");
        assert!(!draft.enabled);
        assert_eq!(draft.graph.nodes.len(), 1);
        assert_eq!(
            draft.graph.nodes[0].proposal.action,
            Action::ReadFile {
                path: shared.clone(),
                max_bytes: 1024
            }
        );

        let stale_digest = draft.digest().unwrap();
        core.workflow_command(WorkflowCommand {
            operation: "review_skill".into(),
            id: draft.id.to_string(),
            json: serde_json::json!({"digest": stale_digest}).to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
        let reviewed = core.store.skill(draft.id).unwrap().unwrap();
        assert!(reviewed.is_reviewed());
        let workflow = crate::workflows::Workflow {
            id: Uuid::new_v4(),
            name: "Shared preparation".into(),
            skill_ids: vec![draft.id],
            enabled: true,
        };
        core.store.save_workflow(&workflow).unwrap();

        // Run the skill through its normal workflow command. It obtains a
        // fresh approval and is still brokered and verified without planning.
        core.workflow_command(WorkflowCommand {
            operation: "run_workflow".into(),
            id: workflow.id.to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
        let (run, approval) = timeout(Duration::from_secs(3), async {
            loop {
                let snapshot = core.snapshot(true).await;
                if let Some(approval) = snapshot.pending_approvals.first() {
                    let task = snapshot
                        .tasks
                        .iter()
                        .find(|task| task.id == approval.task_id)
                        .expect("approval belongs to a visible task");
                    if task.synthesized_skill_sources == reviewed.synthesized_from {
                        break (task.id, approval.clone());
                    }
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        core.resolve_approval(
            approval.approval_id,
            run,
            approval.action_id,
            &approval.digest,
            ApprovalResolution::Approved {
                native_authentication_satisfied: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(completed(&core, run).await.status, TaskStatus::Succeeded);
        assert_eq!(provider.plans.load(std::sync::atomic::Ordering::SeqCst), 9);

        core.workflow_command(WorkflowCommand {
            operation: "configure_learning".into(),
            json: r#"{"enabled":false}"#.into(),
            ..Default::default()
        })
        .await
        .unwrap();
        let paused = core.workflow_snapshot().unwrap();
        assert_eq!(paused["skills"][0]["source_paused"], true);
        assert_eq!(paused["workflows"][0]["enabled"], false);
        assert!(
            core.workflow_command(WorkflowCommand {
                operation: "run_skill".into(),
                id: draft.id.to_string(),
                ..Default::default()
            })
            .await
            .is_err()
        );
        core.workflow_command(WorkflowCommand {
            operation: "configure_learning".into(),
            json: r#"{"enabled":true}"#.into(),
            ..Default::default()
        })
        .await
        .unwrap();

        let source_routine = reviewed.synthesized_from[0].clone();
        core.workflow_command(WorkflowCommand {
            operation: "run_skill".into(),
            id: draft.id.to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
        let (pending_run, _) = timeout(Duration::from_secs(3), async {
            loop {
                let snapshot = core.snapshot(true).await;
                if let Some(task) = snapshot.tasks.iter().find(|task| {
                    task.status.is_active()
                        && task.synthesized_skill_sources == reviewed.synthesized_from
                }) && snapshot
                    .pending_approvals
                    .iter()
                    .any(|approval| approval.task_id == task.id)
                {
                    break (task.id, ());
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        core.workflow_command(WorkflowCommand {
            operation: "forget_routine".into(),
            id: source_routine.clone(),
            ..Default::default()
        })
        .await
        .unwrap();
        assert_eq!(
            completed(&core, pending_run).await.status,
            TaskStatus::Cancelled
        );
        assert!(core.store.skill(draft.id).unwrap().is_none());
        assert!(!core.store.workflow(workflow.id).unwrap().unwrap().enabled);
        let remaining_families = core.workflow_snapshot().unwrap()["routine_families"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(remaining_families.len(), 1);
        let remaining_branches = remaining_families[0]["branches"].as_array().unwrap();
        assert_eq!(remaining_branches.len(), 2);
        assert!(
            remaining_branches
                .iter()
                .all(|branch| branch["routine_id"] != source_routine)
        );
    }

    struct PipelineObserver {
        inner: crate::observation::DeterministicObserver,
        blocked: PathBuf,
        dependent: PathBuf,
        entered: Notify,
        release: Notify,
        successor: Notify,
    }
    #[async_trait]
    impl Observer for PipelineObserver {
        async fn observe(
            &self,
            proposal: &ActionProposal,
            receipt: &ExecutionReceipt,
        ) -> CoreResult<crate::observation::Observation> {
            if matches!(&proposal.action, Action::ReadFile { path, .. } if path == &self.blocked) {
                self.entered.notify_one();
                self.release.notified().await;
            }
            let observation = self.inner.observe(proposal, receipt).await?;
            if matches!(&proposal.action, Action::ReadFile { path, .. } if path == &self.dependent)
            {
                self.successor.notify_one();
            }
            Ok(observation)
        }
    }

    #[tokio::test]
    async fn pipeline_starts_d_after_a_and_b_while_c_is_still_running() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let root = workspace.path().canonicalize().unwrap();
        let paths: Vec<_> = ["a", "b", "c", "d"].map(|name| root.join(name)).into();
        for path in &paths {
            std::fs::write(path, path.to_string_lossy().as_bytes()).unwrap();
        }
        let observer = Arc::new(PipelineObserver {
            inner: crate::observation::DeterministicObserver,
            blocked: paths[2].clone(),
            dependent: paths[3].clone(),
            entered: Notify::new(),
            release: Notify::new(),
            successor: Notify::new(),
        });
        let mut core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        Arc::get_mut(&mut core).unwrap().observer = observer.clone();
        let request = paths
            .iter()
            .map(|p| format!("read \"{}\"", p.display()))
            .collect::<Vec<_>>()
            .join(" and ");
        let mut graph = crate::intent::compile(&request, &[])
            .unwrap()
            .graph(Uuid::new_v4(), "Pipeline real reads".into());
        graph.nodes[3].depends_on =
            BTreeSet::from([graph.nodes[0].proposal.id, graph.nodes[1].proposal.id]);
        let id = core
            .submit_scoped(
                "Pipeline real reads".into(),
                None,
                false,
                Some(graph),
                vec![crate::contracts::ResourceScope {
                    root,
                    effects: BTreeSet::from([crate::contracts::Effect::Read]),
                }],
            )
            .await
            .unwrap();
        timeout(Duration::from_secs(8), observer.entered.notified())
            .await
            .unwrap();
        timeout(Duration::from_secs(8), observer.successor.notified())
            .await
            .expect("D must not wait for C");
        let snapshot = core.get_task(id).await.unwrap();
        assert!(snapshot.actions.values().any(|state| matches!(&state.proposal.action, Action::ReadFile { path, .. } if path == &paths[2]) && state.status == ActionStatus::Verifying));
        observer.release.notify_one();
        let task = completed(&core, id).await;
        assert_eq!(
            task.status,
            TaskStatus::Succeeded,
            "{:?}",
            task.final_outcome
        );
        assert_eq!(task.completed_count(), 4);
    }

    #[tokio::test]
    async fn correction_replaces_running_intent_without_waiting_for_old_work_and_retries_once() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let root = workspace.path().canonicalize().unwrap();
        let old_path = root.join("old");
        let new_path = root.join("new");
        std::fs::write(&old_path, "old data").unwrap();
        std::fs::write(&new_path, "replacement data").unwrap();
        let observer = Arc::new(PipelineObserver {
            inner: crate::observation::DeterministicObserver,
            blocked: old_path.clone(),
            dependent: new_path.clone(),
            entered: Notify::new(),
            release: Notify::new(),
            successor: Notify::new(),
        });
        let mut core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        Arc::get_mut(&mut core).unwrap().observer = observer.clone();
        let old = core
            .submit_scoped(
                format!("read \"{}\"", old_path.display()),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root,
                    effects: BTreeSet::from([crate::contracts::Effect::Read]),
                }],
            )
            .await
            .unwrap();
        // This bounds cold storage/filesystem startup; the test's behavioral
        // assertion is that replacement completes before the old read is released.
        timeout(Duration::from_secs(8), observer.entered.notified())
            .await
            .unwrap();
        let conversation = core.get_task(old).await.unwrap().conversation_id;
        let request = sage_protocol::sage::ipc::v2::SubmitTask {
            text: format!("read \"{}\"", new_path.display()),
            supersedes_task_id: old.to_string(),
            conversation_id: conversation.unwrap().to_string(),
            ..Default::default()
        };
        let key =
            crate::commands::SubmissionKey::for_request(&Uuid::new_v4().to_string(), &request)
                .unwrap();
        assert!(
            core.supersede_receipted(
                old,
                request.text.clone(),
                Some(Uuid::new_v4()),
                vec![],
                key.clone()
            )
            .await
            .is_err()
        );
        assert!(core.get_task(old).await.unwrap().status.is_active());
        let new = core
            .supersede_receipted(old, request.text.clone(), conversation, vec![], key.clone())
            .await
            .unwrap();
        let finished = completed(&core, new).await;
        assert_eq!(
            finished.status,
            TaskStatus::Succeeded,
            "{:?}",
            finished.final_outcome
        );
        assert!(finished.final_outcome.unwrap().contains("replacement data"));
        assert_eq!(
            new, old,
            "A local correction keeps the run and its evidence"
        );
        let revised = core.get_task(old).await.unwrap();
        assert_eq!(revised.intent.as_ref().unwrap().revision, 2);
        assert_eq!(revised.intent.as_ref().unwrap().retired.len(), 1);
        assert!(
            revised
                .actions
                .values()
                .any(|state| state.status == ActionStatus::Skipped)
        );
        let retried = core
            .supersede_receipted(old, request.text, conversation, vec![], key)
            .await
            .unwrap();
        assert_eq!(retried, new);
        assert_eq!(core.snapshot(true).await.tasks.len(), 1);
    }

    #[tokio::test]
    async fn intent_revision_keeps_completed_creation_and_inflight_read_without_restarting_them() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let root = workspace.path().canonicalize().unwrap();
        let folder = root.join("created-once");
        let b = root.join("b");
        let c = root.join("c");
        let d = root.join("d");
        for (path, text) in [(&b, "retained B"), (&c, "retired C"), (&d, "new D")] {
            std::fs::write(path, text).unwrap();
        }
        let observer = Arc::new(PipelineObserver {
            inner: crate::observation::DeterministicObserver,
            blocked: b.clone(),
            dependent: d.clone(),
            entered: Notify::new(),
            release: Notify::new(),
            successor: Notify::new(),
        });
        let mut core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        Arc::get_mut(&mut core).unwrap().observer = observer.clone();
        let request = |last: &std::path::Path| {
            format!(
                "create folder \"{}\" then read \"{}\" and read \"{}\"",
                folder.display(),
                b.display(),
                last.display()
            )
        };
        let id = core
            .submit_scoped(
                request(&c),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root,
                    effects: BTreeSet::from([
                        crate::contracts::Effect::Read,
                        crate::contracts::Effect::Create,
                    ]),
                }],
            )
            .await
            .unwrap();
        timeout(Duration::from_secs(8), observer.entered.notified())
            .await
            .unwrap();
        timeout(Duration::from_secs(8), async {
            while core.get_task(id).await.unwrap().completed_count() < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let before = core.get_task(id).await.unwrap();
        let original = before.intent.as_ref().unwrap().steps.clone();
        assert!(folder.is_dir());
        let command = sage_protocol::sage::ipc::v2::SubmitTask {
            text: request(&d),
            conversation_id: before.conversation_id.unwrap().to_string(),
            supersedes_task_id: id.to_string(),
            ..Default::default()
        };
        let key =
            crate::commands::SubmissionKey::for_request(&Uuid::new_v4().to_string(), &command)
                .unwrap();
        assert_eq!(
            core.supersede_receipted(id, command.text, before.conversation_id, vec![], key)
                .await
                .unwrap(),
            id
        );
        timeout(Duration::from_secs(8), observer.successor.notified())
            .await
            .expect("D starts while unchanged B is still verifying");
        let during = core.get_task(id).await.unwrap();
        let revised = during.intent.as_ref().unwrap();
        assert_eq!(revised.steps[0].action_id, original[0].action_id);
        assert_eq!(revised.steps[1].action_id, original[1].action_id);
        assert!(revised.retired.contains(&original[2].action_id));
        assert_eq!(
            during.actions[&original[1].action_id].status,
            ActionStatus::Verifying
        );
        assert_eq!(during.actions[&original[0].action_id].attempts, 1);
        observer.release.notify_one();
        let finished = completed(&core, id).await;
        assert_eq!(
            finished.status,
            TaskStatus::Succeeded,
            "{:?}",
            finished.final_outcome
        );
        assert_eq!(finished.actions[&original[1].action_id].attempts, 1);
        assert_eq!(finished.completed_count(), 3);
        assert!(finished.final_outcome.as_ref().unwrap().contains("new D"));
        assert!(
            !finished
                .final_outcome
                .as_ref()
                .unwrap()
                .contains("retired C")
        );
        let skill = core
            .store
            .capture_skill(&finished, "Revised fixture")
            .unwrap();
        assert_eq!(skill.graph.nodes.len(), 3, "Skills exclude retired work");
        assert!(
            skill
                .graph
                .nodes
                .iter()
                .all(|node| node.proposal.id != original[2].action_id)
        );
    }

    #[tokio::test]
    async fn intent_revision_is_atomic_and_closes_only_replaced_approvals() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let a = workspace.path().join("a");
        let b = workspace.path().join("b");
        std::fs::write(&a, "old").unwrap();
        std::fs::write(&b, "new").unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let id = core
            .submit_task(format!("read \"{}\"", a.display()))
            .await
            .unwrap();
        let approval = timeout(Duration::from_secs(3), async {
            loop {
                if let Some(approval) = core.snapshot(true).await.pending_approvals.first() {
                    break approval.clone();
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let before = core.get_task(id).await.unwrap();
        let command = sage_protocol::sage::ipc::v2::SubmitTask {
            text: format!("read \"{}\"", b.display()),
            conversation_id: before.conversation_id.unwrap().to_string(),
            supersedes_task_id: id.to_string(),
            ..Default::default()
        };
        let key =
            crate::commands::SubmissionKey::for_request(&Uuid::new_v4().to_string(), &command)
                .unwrap();
        core.store.with_connection(|db| { db.execute_batch("CREATE TEMP TRIGGER reject_revision BEFORE INSERT ON command_inbox BEGIN SELECT RAISE(ABORT,'fixture receipt failure'); END;")?; Ok(()) }).unwrap();
        assert!(
            core.supersede_receipted(
                id,
                command.text.clone(),
                before.conversation_id,
                vec![],
                key.clone()
            )
            .await
            .is_err()
        );
        let failed = core.get_task(id).await.unwrap();
        assert_eq!(failed.request, before.request);
        assert_eq!(failed.intent, before.intent);
        assert!(core.runtime.is_held(id));
        assert!(!core.runtime.action_retired(id, approval.action_id));
        assert_eq!(core.snapshot(true).await.pending_approvals.len(), 1);
        assert!(core.store.accepted_submission(&key).unwrap().is_none());
        core.store
            .with_connection(|db| {
                db.execute_batch("DROP TRIGGER reject_revision;")?;
                Ok(())
            })
            .unwrap();
        assert_eq!(
            core.supersede_receipted(
                id,
                command.text.clone(),
                before.conversation_id,
                vec![],
                key.clone()
            )
            .await
            .unwrap(),
            id
        );
        assert!(
            core.resolve_approval(
                approval.approval_id,
                id,
                approval.action_id,
                &approval.digest,
                ApprovalResolution::Approved {
                    native_authentication_satisfied: false
                }
            )
            .await
            .is_err()
        );
        let replacement = timeout(Duration::from_secs(3), async {
            loop {
                if let Some(approval) = core.snapshot(true).await.pending_approvals.first() {
                    break approval.clone();
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_ne!(replacement.action_id, approval.action_id);
        assert_eq!(
            core.supersede_receipted(id, command.text, before.conversation_id, vec![], key)
                .await
                .unwrap(),
            id
        );
        assert!(!core.runtime.is_held(id));
        core.resolve_approval(
            replacement.approval_id,
            id,
            replacement.action_id,
            &replacement.digest,
            ApprovalResolution::Approved {
                native_authentication_satisfied: false,
            },
        )
        .await
        .unwrap();
        let finished = completed(&core, id).await;
        assert_eq!(
            finished.status,
            TaskStatus::Succeeded,
            "{:?}",
            finished.final_outcome
        );
        assert_eq!(finished.completed_count(), 1);
    }

    #[async_trait]
    impl ModelProvider for FolderPlanProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            ProviderDescriptor {
                id: "test-folder-plan".into(),
                display_name: "Test folder plan".into(),
                local: true,
                roles: vec![crate::model::ModelRole::Reasoning],
            }
        }

        async fn create_plan(&self, context: PlanningContext) -> CoreResult<ActionGraph> {
            Ok(ActionGraph {
                goal: "Create and verify one folder".into(),
                nodes: vec![ActionNode {
                    proposal: ActionProposal {
                        id: Uuid::new_v4(),
                        task_id: context.task_id,
                        action: Action::CreateFolder {
                            path: self.path.clone(),
                        },
                        expected_outcome: ExpectedOutcome::Condition {
                            condition: Condition::FileExists {
                                path: self.path.clone(),
                            },
                        },
                        target_resource: self.path.to_string_lossy().into_owned(),
                        provenance: Provenance::model(Vec::new()),
                        metadata: BTreeMap::new(),
                    },
                    depends_on: BTreeSet::new(),
                }],
            })
        }

        async fn replan(&self, _context: ReplanContext) -> CoreResult<ActionGraph> {
            Err(CoreError::Model("test does not expect replanning".into()))
        }
    }

    #[tokio::test]
    async fn approval_capability_execution_verification_and_undo_are_end_to_end() {
        let directory = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let destination = workspace.path().join("verified-folder");
        let config = CoreConfig::for_test(directory.path());
        let core = SageCore::new(
            config,
            Arc::new(FolderPlanProvider {
                path: destination.clone(),
            }),
        )
        .unwrap();
        let mut events = core.events().subscribe();
        let task_id = core.submit_task("Create a test folder").await.unwrap();

        let completed = timeout(Duration::from_secs(10), async {
            loop {
                let event = events.recv().await.unwrap();
                match event.kind {
                    CoreEventKind::ApprovalRequested {
                        approval_id,
                        action_id,
                        digest,
                        ..
                    } => {
                        core.resolve_approval(
                            approval_id,
                            task_id,
                            action_id,
                            &digest,
                            ApprovalResolution::Approved {
                                native_authentication_satisfied: false,
                            },
                        )
                        .await
                        .unwrap();
                    }
                    CoreEventKind::TaskCompleted { .. } => break,
                    CoreEventKind::Error { message, .. } => panic!("task failed: {message}"),
                    _ => {}
                }
            }
        })
        .await;
        assert!(completed.is_ok(), "task did not complete before timeout");
        assert!(destination.is_dir());
        let task = core.get_task(task_id).await.unwrap();
        assert_eq!(task.status, TaskStatus::Succeeded);
        assert_eq!(task.completed_count(), 1);

        core.undo_last_action(task_id, task.rollback_action_id.unwrap())
            .await
            .unwrap();
        assert!(!destination.exists());
        let undone = core.get_task(task_id).await.unwrap();
        assert_eq!(
            undone.undo.as_ref().unwrap().phase,
            crate::domain::UndoPhase::Verified
        );
        assert_eq!(crate::domain::ExecutionFacts::for_task(&undone).undone, 1);
        core.undo_last_action(task_id, task.rollback_action_id.unwrap())
            .await
            .unwrap();
        assert_eq!(
            core.get_task(task_id).await.unwrap().revision,
            undone.revision
        );
        // Even an interrupted forward projection cannot reopen a scope whose
        // filesystem state has been changed by compensation.
        core.update_task(task_id, |task| {
            task.status = TaskStatus::Interrupted;
            Ok(())
        })
        .await
        .unwrap();
        assert!(matches!(
            core.control_task(task_id, TaskStatus::Running).await,
            Err(CoreError::PermissionRequired(_))
        ));
    }

    #[derive(Default)]
    struct FailingCheckpointStore {
        inner: MemorySecretStore,
        fail: std::sync::atomic::AtomicBool,
    }
    impl SecretStore for FailingCheckpointStore {
        fn get(&self, account: &str) -> CoreResult<Option<SecretBytes>> {
            self.inner.get(account)
        }
        fn set(&self, account: &str, bytes: &SecretBytes) -> CoreResult<()> {
            if account.starts_with("audit:") && self.fail.load(std::sync::atomic::Ordering::SeqCst)
            {
                return Err(CoreError::SecretStore(
                    "Fixture checkpoint unavailable".into(),
                ));
            }
            self.inner.set(account, bytes)
        }
        fn delete(&self, account: &str) -> CoreResult<()> {
            self.inner.delete(account)
        }
    }
    struct FailCheckpointAfterObservation(Arc<FailingCheckpointStore>);
    #[async_trait]
    impl Observer for FailCheckpointAfterObservation {
        async fn observe(
            &self,
            proposal: &ActionProposal,
            receipt: &ExecutionReceipt,
        ) -> CoreResult<crate::observation::Observation> {
            let observed = crate::observation::DeterministicObserver
                .observe(proposal, receipt)
                .await?;
            self.0.fail.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(observed)
        }
    }

    #[tokio::test]
    async fn checkpoint_failure_after_verification_keeps_the_committed_effect_truth() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let destination = workspace.path().join("verified-before-checkpoint-failure");
        let secrets = Arc::new(FailingCheckpointStore::default());
        let mut core = SageCore::new_with_secret_store(
            CoreConfig::for_test(data.path()),
            Arc::new(FolderPlanProvider {
                path: destination.clone(),
            }),
            secrets.clone(),
        )
        .unwrap();
        Arc::get_mut(&mut core).unwrap().observer =
            Arc::new(FailCheckpointAfterObservation(secrets.clone()));
        let id = core
            .submit_scoped(
                "Create the fixture folder".into(),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root: workspace.path().into(),
                    effects: BTreeSet::from([crate::contracts::Effect::Create]),
                }],
            )
            .await
            .unwrap();
        let task = completed(&core, id).await;
        assert!(destination.is_dir());
        assert_eq!(task.status, TaskStatus::Partial);
        assert_eq!(task.completed_count(), 1);
        assert_eq!(
            task.actions.values().next().unwrap().status,
            ActionStatus::Succeeded
        );
        assert_eq!(
            task.tool_results.last().unwrap().verdict,
            crate::contracts::Verdict::Confirmed
        );
        core.store
            .with_connection(|db| {
                let state: String = db.query_row(
                    "SELECT state FROM action_journal WHERE run_id=?1",
                    [id.to_string()],
                    |row| row.get(0),
                )?;
                assert_eq!(state, "confirmed");
                Ok(())
            })
            .unwrap();
        secrets
            .fail
            .store(false, std::sync::atomic::Ordering::SeqCst);
        core.store.checkpoint_audit(secrets.as_ref()).unwrap();
    }

    #[tokio::test]
    async fn reobserved_crash_recovery_installs_durable_result_evidence() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let destination = workspace.path().join("recover-existing-effect");
        let core = SageCore::new_with_secret_store(
            CoreConfig::for_test(data.path()),
            Arc::new(FolderPlanProvider {
                path: destination.clone(),
            }),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        let mut events = core.events().subscribe();
        let id = core
            .submit_scoped(
                "Create a recoverable folder".into(),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root: workspace.path().into(),
                    effects: BTreeSet::from([crate::contracts::Effect::Create]),
                }],
            )
            .await
            .unwrap();
        timeout(Duration::from_secs(5), async {
            loop {
                match events.recv().await.unwrap().kind {
                    CoreEventKind::TaskCompleted { .. } => break,
                    CoreEventKind::Error { message, .. } => panic!("{message}"),
                    _ => {}
                }
            }
        })
        .await
        .unwrap();
        // Legacy/crash fixture: the filesystem effect exists, but its verified
        // projection was not retained. Recovery must not merely flip a status.
        run_exited(&core, id).await;
        core.update_task(id, |task| {
            task.status = TaskStatus::Interrupted;
            for action in task.actions.values_mut() {
                action.status = ActionStatus::Uncertain;
            }
            task.tool_results.clear();
            Ok(())
        })
        .await
        .unwrap();
        core.store.with_connection(|db| { db.execute("UPDATE action_journal SET state='uncertain',verification_json=NULL WHERE run_id=?1", [id.to_string()])?; Ok(()) }).unwrap();
        let task = core.get_task(id).await.unwrap();
        let proposal = &task.actions.values().next().unwrap().proposal;
        let binding = crate::execution::bridge::EffectBinding {
            task_id: id,
            action_id: proposal.id,
            action_digest: crate::policy::approval_digest(proposal).unwrap(),
        };
        core.store
            .record_worker_receipt(
                &Uuid::new_v4().to_string(),
                "fixture-session",
                &binding,
                "response",
                b"untrusted worker success assertion",
            )
            .unwrap();
        assert_eq!(
            core.store.pending_worker_receipts(Some(id)).unwrap().len(),
            1
        );
        core.control_task(id, TaskStatus::Running).await.unwrap();
        let recovered = completed(&core, id).await;
        assert_eq!(recovered.status, TaskStatus::Succeeded);
        assert_eq!(recovered.completed_count(), 1);
        assert_eq!(recovered.tool_results.len(), 2);
        assert_eq!(
            recovered.tool_results[0].verdict,
            crate::contracts::Verdict::Uncertain
        );
        assert_eq!(
            recovered.tool_results[1].verdict,
            crate::contracts::Verdict::Confirmed
        );
        assert!(
            core.store
                .pending_worker_receipts(Some(id))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            core.store
                .load_tasks(true)
                .unwrap()
                .iter()
                .find(|task| task.id == id)
                .unwrap()
                .completed_count(),
            1
        );
        assert!(destination.is_dir());
    }

    struct BudgetProvider {
        source: PathBuf,
        first: std::sync::Mutex<Option<Uuid>>,
    }
    #[async_trait]
    impl ModelProvider for BudgetProvider {
        fn descriptor(&self) -> crate::model::ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: crate::model::ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, context: TurnContext) -> CoreResult<ModelTurn> {
            let mut first = self.first.lock().unwrap();
            let first = first.get_or_insert(context.planning.task_id);
            if *first != context.planning.task_id {
                assert!(!context.results.is_empty());
                return Ok(ModelTurn::Answer(
                    "Continued using earlier evidence and a fresh scope.".into(),
                ));
            }
            Ok(proposal_turn(
                context.planning.task_id,
                Action::ReadFile {
                    path: self.source.clone(),
                    max_bytes: 64,
                },
            ))
        }
    }
    #[tokio::test]
    async fn exhausted_budget_continues_with_fresh_run_identity_and_no_old_actions() {
        let state = tempdir().unwrap();
        let work = tempdir().unwrap();
        let root = work.path().canonicalize().unwrap();
        let source = root.join("input.txt");
        std::fs::write(&source, b"evidence").unwrap();
        let core = SageCore::new_with_secret_store(
            CoreConfig::for_test(state.path()),
            Arc::new(BudgetProvider {
                source,
                first: Default::default(),
            }),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        let id = core
            .submit_scoped(
                "Read the next input".into(),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root,
                    effects: std::collections::BTreeSet::from([crate::contracts::Effect::Read]),
                }],
            )
            .await
            .unwrap();
        let paused = completed(&core, id).await;
        assert_eq!(paused.status, TaskStatus::Interrupted);
        assert!(paused.budget_exhausted);
        assert_eq!(paused.actions.len(), 32);
        core.control_task(id, TaskStatus::Running).await.unwrap();
        let next = core
            .tasks
            .read()
            .await
            .values()
            .find(|task| task.continuation_of == Some(id))
            .unwrap()
            .id;
        assert_ne!(next, id);
        let done = completed(&core, next).await;
        assert_eq!(done.status, TaskStatus::Answered);
        assert!(done.actions.is_empty());
        assert_eq!(done.contract.unwrap().run_id, next);
        assert_eq!(done.control_scope_id, Some(id));
        assert_eq!(core.get_task(id).await.unwrap().continued_by, Some(next));
        assert!(core.control_task(id, TaskStatus::Running).await.is_err());
        assert_eq!(core.get_task(id).await.unwrap().actions.len(), 32);
    }

    async fn continuation_fixture() -> (
        tempfile::TempDir,
        Arc<SageCore>,
        Uuid,
        Arc<Notify>,
        Arc<Notify>,
    ) {
        let data = tempdir().unwrap();
        let entered = Arc::new(Notify::new());
        let dropped = Arc::new(Notify::new());
        let core = SageCore::new_with_secret_store(
            CoreConfig::for_test(data.path()),
            Arc::new(StalledProvider {
                entered: entered.clone(),
                dropped: dropped.clone(),
            }),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        let conversation = core
            .store
            .ensure_conversation(None, "Continue fixture")
            .unwrap();
        let mut task = Task::new("Continue fixture");
        task.status = TaskStatus::Interrupted;
        task.budget_exhausted = true;
        task.contract = Some(crate::contracts::RunContract::local(task.id));
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
        core.store
            .accept_task(&mut task, &message, &started, None)
            .unwrap();
        let id = task.id;
        core.tasks.write().await.insert(id, task);
        (data, core, id, entered, dropped)
    }

    #[tokio::test]
    async fn stop_before_continuation_acceptance_retires_the_handoff_without_a_child() {
        let (_data, core, id, _, _) = continuation_fixture().await;
        // Resume owns the parent but waits at submit_run's storage-unlock lane.
        let unlock = core.storage_unlock.lock().await;
        let resume_core = core.clone();
        let resume =
            tokio::spawn(async move { resume_core.control_task(id, TaskStatus::Running).await });
        timeout(Duration::from_secs(2), async {
            while !core.runtime.is_active(id) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        core.control_task(id, TaskStatus::Cancelled).await.unwrap();
        drop(unlock);
        assert!(resume.await.unwrap().is_err());
        assert_eq!(core.store.load_tasks(true).unwrap().len(), 1);
        assert_eq!(
            core.get_task(id).await.unwrap().status,
            TaskStatus::Cancelled
        );
        assert!(
            core.runtime.scope_tasks(id).is_empty(),
            "An unaccepted child must not leak a control slot"
        );
        assert!(core.store.control_scope_stopped(id).unwrap());
    }

    #[tokio::test]
    async fn stop_after_handoff_reaches_the_child_and_undo_cannot_overlap_it() {
        let (_data, core, id, entered, dropped) = continuation_fixture().await;
        core.control_task(id, TaskStatus::Running).await.unwrap();
        timeout(Duration::from_secs(2), entered.notified())
            .await
            .unwrap();
        let child = core.get_task(id).await.unwrap().continued_by.unwrap();
        assert!(
            core.runtime.is_active(child),
            "Retiring the parent must leave its child running"
        );
        assert_eq!(core.get_task(child).await.unwrap().control_scope(), id);
        assert!(matches!(
            core.undo_last_action(id, Uuid::new_v4()).await,
            Err(CoreError::PermissionRequired(_))
        ));
        assert!(core.control_task(id, TaskStatus::Running).await.is_err());
        core.control_task(id, TaskStatus::Cancelled).await.unwrap();
        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .unwrap();
        run_exited(&core, child).await;
        assert_eq!(
            core.get_task(child).await.unwrap().status,
            TaskStatus::Cancelled
        );
        assert_eq!(core.store.load_tasks(true).unwrap().len(), 2);
        assert!(core.store.control_scope_stopped(id).unwrap());
    }

    #[tokio::test]
    async fn a_saved_scope_stop_survives_failed_task_projections_and_restart() {
        let (data, core, id, entered, dropped) = continuation_fixture().await;
        core.control_task(id, TaskStatus::Running).await.unwrap();
        timeout(Duration::from_secs(2), entered.notified())
            .await
            .unwrap();
        let child = core.get_task(id).await.unwrap().continued_by.unwrap();
        core.store.with_connection(|db| { db.execute_batch("CREATE TEMP TRIGGER fail_scope_projection BEFORE INSERT ON tasks WHEN NEW.status='cancelled' BEGIN SELECT RAISE(ABORT,'fixture task projection unavailable'); END;")?; Ok(()) }).unwrap();
        assert!(core.control_task(id, TaskStatus::Cancelled).await.is_err());
        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .unwrap();
        run_exited(&core, child).await;
        assert!(core.store.control_scope_stopped(id).unwrap());
        drop(core);
        let reopened = SageCore::new_with_secret_store(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        assert_eq!(
            reopened.get_task(id).await.unwrap().status,
            TaskStatus::Cancelled
        );
        assert_eq!(
            reopened.get_task(child).await.unwrap().status,
            TaskStatus::Cancelled
        );
        assert!(
            reopened
                .control_task(child, TaskStatus::Running)
                .await
                .is_err()
        );
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn background_trigger_cannot_follow_a_replaced_folder_outside_its_scope() {
        let state = tempdir().unwrap();
        let work = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let root = work.path().canonicalize().unwrap();
        let directory = root.join("watched");
        std::fs::create_dir(&directory).unwrap();
        let core = SageCore::new_with_secret_store(
            CoreConfig::for_test(state.path()),
            Arc::new(UnconfiguredModelProvider),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        let mut schedule:crate::workflows::Schedule=serde_json::from_value(json!({"id":Uuid::new_v4(),"name":"watch","request":"Review changes","conversation_id":Uuid::new_v4(),"workflow_id":null,"trigger":{"kind":"folder_changed","path":directory},"enabled":true,"next_run_at":Utc::now(),"last_task_id":null,"last_condition":false,"last_error":null,"resources":[{"root":root,"effects":["read"]}],"expires_at":Utc::now()+ChronoDuration::hours(1),"remaining_runs":2})).unwrap();
        core.validate_trigger(&schedule).unwrap();
        std::fs::remove_dir(&directory).unwrap();
        std::os::unix::fs::symlink(outside.path(), &directory).unwrap();
        assert!(core.validate_trigger(&schedule).is_err());
        schedule.next_run_at = Utc::now() - ChronoDuration::seconds(1);
        core.store.save_schedule(&schedule).unwrap();
        core.scheduler_tick(&mut Default::default()).await.unwrap();
        assert!(!core.store.schedules().unwrap()[0].enabled);
    }
    #[tokio::test]
    async fn folder_trigger_retries_dirty_roots_after_persistence_failure() {
        let state = tempdir().unwrap();
        let work = tempdir().unwrap();
        let root = work.path().canonicalize().unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(state.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let schedule: crate::workflows::Schedule = serde_json::from_value(json!({
            "id":Uuid::new_v4(),"name":"watch","request":"Review changes","conversation_id":Uuid::new_v4(),
            "workflow_id":null,"trigger":{"kind":"folder_changed","path":root},"enabled":true,
            "next_run_at":Utc::now()+ChronoDuration::minutes(5),"last_task_id":null,"last_condition":false,"last_error":null,
            "resources":[{"root":root,"effects":["read"]}],"expires_at":Utc::now()+ChronoDuration::hours(1),"remaining_runs":2
        })).unwrap();
        core.store.save_schedule(&schedule).unwrap();
        core.store.with_connection(|db| { db.execute_batch("CREATE TEMP TRIGGER fail_trigger_save BEFORE UPDATE ON schedules BEGIN SELECT RAISE(ABORT,'fixture storage unavailable'); END;")?; Ok(()) }).unwrap();
        let mut changed = std::collections::HashSet::from([root]);
        assert!(core.scheduler_tick(&mut changed).await.is_err());
        assert_eq!(changed.len(), 1);
        assert!(
            !core
                .store
                .schedule(schedule.id)
                .unwrap()
                .unwrap()
                .trigger_pending
        );
        core.store
            .with_connection(|db| {
                db.execute_batch("DROP TRIGGER fail_trigger_save;")?;
                Ok(())
            })
            .unwrap();
        core.scheduler_tick(&mut changed).await.unwrap();
        assert!(changed.is_empty());
        assert!(
            core.store
                .schedule(schedule.id)
                .unwrap()
                .unwrap()
                .trigger_pending
        );
    }

    struct ResultLoopProvider {
        source: PathBuf,
        destination: PathBuf,
    }
    struct BatchedReadProvider {
        paths: Vec<PathBuf>,
        calls: std::sync::atomic::AtomicUsize,
        missing_second: bool,
    }
    #[async_trait]
    impl ModelProvider for BatchedReadProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, context: TurnContext) -> CoreResult<ModelTurn> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if context.results.is_empty() {
                return crate::model::parse_turn(&json!({"goal":"Compare two known inputs","answer":"","actions":self.paths.iter().map(|path| json!({"kind":"read_file","payload":{"path":path,"max_bytes":128}})).collect::<Vec<_>>()}).to_string(), context.planning.task_id);
            }
            assert_eq!(
                context.results.len(),
                2,
                "The model must receive the whole batch together"
            );
            if self.missing_second {
                assert_eq!(
                    context
                        .results
                        .iter()
                        .filter(|result| result.verdict == crate::contracts::Verdict::Confirmed)
                        .count(),
                    1
                );
                assert_eq!(
                    context
                        .results
                        .iter()
                        .filter(|result| result.verdict == crate::contracts::Verdict::Failed)
                        .count(),
                    1
                );
                return Ok(ModelTurn::Answer(
                    "One input was read; the other was unavailable.".into(),
                ));
            }
            let actual = context
                .results
                .iter()
                .map(|result| {
                    assert_eq!(result.verdict, crate::contracts::Verdict::Confirmed);
                    result.output["text"].as_str().unwrap()
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(
                actual,
                BTreeSet::from(["first real input", "second real input"])
            );
            Ok(ModelTurn::Answer("Compared both verified inputs.".into()))
        }
    }

    #[tokio::test]
    async fn independent_file_batch_uses_one_planning_call_and_verifies_each_read() {
        for missing_second in [false, true] {
            let data = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let root = workspace.path().canonicalize().unwrap();
            let paths = vec![root.join("first.txt"), root.join("second.txt")];
            std::fs::write(&paths[0], "first real input").unwrap();
            if !missing_second {
                std::fs::write(&paths[1], "second real input").unwrap();
            }
            let provider = Arc::new(BatchedReadProvider {
                paths,
                calls: 0.into(),
                missing_second,
            });
            let core = SageCore::new(CoreConfig::for_test(data.path()), provider.clone()).unwrap();
            let id = core
                .submit_scoped(
                    "Compare these two files".into(),
                    None,
                    false,
                    None,
                    vec![crate::contracts::ResourceScope {
                        root,
                        effects: BTreeSet::from([crate::contracts::Effect::Read]),
                    }],
                )
                .await
                .unwrap();
            let task = completed(&core, id).await;
            assert_eq!(
                task.status,
                if missing_second {
                    TaskStatus::Partial
                } else {
                    TaskStatus::Succeeded
                },
                "{:?}",
                task.final_outcome
            );
            assert_eq!(task.completed_count(), if missing_second { 1 } else { 2 });
            assert_eq!(
                provider.calls.load(std::sync::atomic::Ordering::Relaxed),
                2,
                "One planning turn plus one answer"
            );
        }
    }
    struct DirectoryDiscoveryProvider {
        page_size: u32,
        names: std::sync::Mutex<BTreeSet<String>>,
        candidate: std::sync::Mutex<Option<String>>,
    }
    #[async_trait]
    impl ModelProvider for DirectoryDiscoveryProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, context: TurnContext) -> CoreResult<ModelTurn> {
            let scope = context
                .planning
                .untrusted_context
                .iter()
                .find(|item| item.source == "authorized_file_roots")
                .expect("scope must reach the planner");
            let scope: serde_json::Value = serde_json::from_str(&scope.content).unwrap();
            let root = PathBuf::from(scope["roots"][0]["path"].as_str().unwrap());
            assert_eq!(scope["roots"][0]["effects"], json!(["read"]));
            let latest = context.results.last();
            if latest.is_some_and(|result| result.tool == "read_file") {
                let text = latest.unwrap().output["text"].as_str().unwrap();
                return Ok(ModelTurn::Answer(format!("Discovered and read: {text}")));
            }
            let mut cursor = None;
            if let Some(result) = latest {
                assert_eq!(result.tool, "list_directory");
                assert_eq!(result.verdict, crate::contracts::Verdict::Confirmed);
                let page: crate::execution::directory::DirectoryPage =
                    serde_json::from_value(result.output.clone()).unwrap();
                for entry in page.entries {
                    assert_eq!(
                        entry.name_encoding,
                        crate::execution::directory::NameEncoding::Utf8
                    );
                    self.names.lock().unwrap().insert(entry.name.clone());
                    if entry.kind == crate::execution::directory::EntryKind::File
                        && entry.name.ends_with(".txt")
                    {
                        *self.candidate.lock().unwrap() = Some(entry.name);
                    }
                }
                if page.next_cursor.is_none() {
                    let name = self
                        .candidate
                        .lock()
                        .unwrap()
                        .clone()
                        .expect("text file must be discovered before it is read");
                    return Ok(proposal_turn(
                        context.planning.task_id,
                        Action::ReadFile {
                            path: root.join(name),
                            max_bytes: 1024,
                        },
                    ));
                }
                cursor = page.next_cursor;
            }
            Ok(proposal_turn(
                context.planning.task_id,
                Action::ListDirectory {
                    path: root,
                    page_size: self.page_size,
                    cursor,
                },
            ))
        }
    }
    #[tokio::test]
    async fn selected_folder_discovery_drives_reading_and_pages_survive_budget_continuation() {
        for (count, page_size) in [(7, 2), (35, 1)] {
            let state = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let root = workspace.path().canonicalize().unwrap();
            for n in 0..count - 1 {
                std::fs::write(root.join(format!("entry-{n:03}.md")), b"other source").unwrap();
            }
            let filename = format!("z-discovered-{}.txt", Uuid::new_v4());
            std::fs::write(root.join(&filename), b"folder-discovery-body").unwrap();
            let provider = Arc::new(DirectoryDiscoveryProvider {
                page_size,
                names: Default::default(),
                candidate: Default::default(),
            });
            let core = SageCore::new_with_secret_store(
                CoreConfig::for_test(state.path()),
                provider.clone(),
                Arc::new(MemorySecretStore::default()),
            )
            .unwrap();
            let mut events = core.events().subscribe();
            let mut id = core
                .submit_scoped(
                    "Read the text document in my selected folder".into(),
                    None,
                    false,
                    None,
                    vec![crate::contracts::ResourceScope {
                        root,
                        effects: [crate::contracts::Effect::Read].into(),
                    }],
                )
                .await
                .unwrap();
            let mut continuations = 0;
            let result = loop {
                let result = completed(&core, id).await;
                if result.status != TaskStatus::Interrupted {
                    break result;
                }
                assert!(result.budget_exhausted, "{:?}", result.final_outcome);
                timeout(Duration::from_secs(5), async {
                    while core.runtime.is_active(id) {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .unwrap();
                core.resume_interrupted(id).await.unwrap();
                id = core
                    .get_task(id)
                    .await
                    .unwrap()
                    .continued_by
                    .expect("fresh budget must create a continuation");
                continuations += 1;
                assert!(continuations <= 1);
            };
            assert_eq!(
                result.status,
                TaskStatus::Succeeded,
                "{:?}",
                result.final_outcome
            );
            assert_eq!(
                result.final_outcome.as_deref(),
                Some("Discovered and read: folder-discovery-body")
            );
            assert_eq!(provider.names.lock().unwrap().len(), count);
            assert_eq!(
                provider.candidate.lock().unwrap().as_deref(),
                Some(filename.as_str())
            );
            assert_eq!(continuations, usize::from(count > 32));
            while let Ok(event) = events.try_recv() {
                assert!(!matches!(
                    event.kind,
                    CoreEventKind::ApprovalRequested { .. }
                ));
            }
            assert!(result.tool_results.iter().any(
                |r| r.tool == "read_file" && r.verdict == crate::contracts::Verdict::Confirmed
            ));
        }
    }
    fn proposal_turn(task_id: Uuid, action: Action) -> ModelTurn {
        ModelTurn::Actions(ActionGraph {
            goal: "Verified test task".into(),
            nodes: vec![ActionNode {
                proposal: ActionProposal {
                    id: Uuid::new_v4(),
                    task_id,
                    action,
                    expected_outcome: ExpectedOutcome::UserAnswered,
                    target_resource: "untrusted model target".into(),
                    provenance: Provenance::model(Vec::new()),
                    metadata: Default::default(),
                },
                depends_on: Default::default(),
            }],
        })
    }
    #[async_trait]
    impl ModelProvider for ResultLoopProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, context: TurnContext) -> CoreResult<ModelTurn> {
            assert!(
                context
                    .planning
                    .trusted_constraints
                    .iter()
                    .any(|s| s.contains("never approval"))
            );
            Ok(match context.results.len() {
                0 => proposal_turn(
                    context.planning.task_id,
                    Action::ReadFile {
                        path: self.source.clone(),
                        max_bytes: 1024,
                    },
                ),
                1 => {
                    let text = context.results[0].output["text"]
                        .as_str()
                        .expect("actual file result");
                    assert_eq!(text, "orchid-42");
                    proposal_turn(
                        context.planning.task_id,
                        Action::WriteFile {
                            path: self.destination.clone(),
                            content: format!("Read: {text}"),
                            overwrite: false,
                        },
                    )
                }
                2 => ModelTurn::Answer("Read orchid-42 and saved the verified result.".into()),
                _ => panic!("unbounded tool loop"),
            })
        }
    }
    async fn completed(core: &Arc<SageCore>, task: Uuid) -> Task {
        // This is a fixture liveness bound, not an operation latency target.
        // Encrypted-store startup plus concurrent filesystem tests can exceed 5s.
        timeout(Duration::from_secs(15), async {
            loop {
                let value = core.get_task(task).await.unwrap();
                if !value.status.is_active() {
                    break value;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("run did not reach a terminal or review state")
    }
    #[tokio::test]
    async fn actual_file_results_drive_later_actions_without_repeated_scope_approval() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let source = workspace.path().join("source.txt");
        let destination = workspace.path().join("answer.txt");
        std::fs::write(&source, "orchid-42").unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(ResultLoopProvider {
                source,
                destination: destination.clone(),
            }),
        )
        .unwrap();
        let mut events = core.events().subscribe();
        let task = core
            .submit_scoped(
                "Read and summarize the file".into(),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root: workspace.path().into(),
                    effects: BTreeSet::from([
                        crate::contracts::Effect::Read,
                        crate::contracts::Effect::Create,
                    ]),
                }],
            )
            .await
            .unwrap();
        let result = completed(&core, task).await;
        assert_eq!(
            result.status,
            TaskStatus::Succeeded,
            "{:?}",
            result.final_outcome
        );
        assert_eq!(result.completed_count(), 2);
        let read = result
            .actions
            .values()
            .find(|state| matches!(state.proposal.action, Action::ReadFile { .. }))
            .unwrap();
        let write = result
            .actions
            .values()
            .find(|state| matches!(state.proposal.action, Action::WriteFile { .. }))
            .unwrap();
        assert!(
            result.dependencies[&write.proposal.id].contains(&read.proposal.id),
            "Captured workflows must preserve cross-turn causality"
        );
        assert_eq!(
            std::fs::read_to_string(destination).unwrap(),
            "Read: orchid-42"
        );
        while let Ok(event) = events.try_recv() {
            assert!(!matches!(
                event.kind,
                CoreEventKind::ApprovalRequested { .. }
            ));
        }
        assert_eq!(
            result.tool_results[0].label.sensitivity,
            crate::contracts::Sensitivity::Private
        );
    }

    struct AnswerProvider {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        remote: bool,
    }
    #[async_trait]
    impl ModelProvider for AnswerProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        fn data_destination(&self) -> CoreResult<Option<String>> {
            Ok(self
                .remote
                .then(|| "https://provider.example/v1#test".into()))
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, _: TurnContext) -> CoreResult<ModelTurn> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ModelTurn::Answer("A triangle has three sides.".into()))
        }
    }
    #[tokio::test]
    async fn ordinary_question_finishes_without_any_computer_action() {
        let data = tempdir().unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(AnswerProvider {
                calls: calls.clone(),
                remote: false,
            }),
        )
        .unwrap();
        let id = core
            .submit_task("How many sides does a triangle have?")
            .await
            .unwrap();
        let task = completed(&core, id).await;
        assert_eq!(task.status, TaskStatus::Answered);
        assert!(task.actions.is_empty());
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(
            task.final_outcome.as_deref(),
            Some("A triangle has three sides.")
        );
    }

    #[tokio::test]
    async fn concurrent_submission_retries_and_restart_run_the_model_only_once() {
        let data = tempdir().unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let provider = Arc::new(AnswerProvider {
            calls: calls.clone(),
            remote: false,
        });
        let core = SageCore::new(CoreConfig::for_test(data.path()), provider.clone()).unwrap();
        let submit = sage_protocol::sage::ipc::v2::SubmitTask {
            text: "How many sides does a triangle have?".into(),
            conversation_id: Uuid::new_v4().to_string(),
            ..Default::default()
        };
        let key = crate::commands::SubmissionKey::for_request(&Uuid::new_v4().to_string(), &submit)
            .unwrap();
        let conversation = submit.conversation_id.parse().unwrap();
        let mut clients = tokio::task::JoinSet::new();
        for _ in 0..24 {
            let core = core.clone();
            let key = key.clone();
            let text = submit.text.clone();
            clients.spawn(async move {
                core.submit_receipted(text, Some(conversation), Vec::new(), key)
                    .await
                    .unwrap()
            });
        }
        let mut task_ids = BTreeSet::new();
        while let Some(result) = clients.join_next().await {
            task_ids.insert(result.unwrap());
        }
        assert_eq!(task_ids.len(), 1);
        let id = *task_ids.first().unwrap();
        assert_eq!(completed(&core, id).await.status, TaskStatus::Answered);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        drop(core);

        let reopened = SageCore::new(CoreConfig::for_test(data.path()), provider).unwrap();
        let replay = reopened
            .submit_receipted(submit.text, Some(conversation), Vec::new(), key)
            .await
            .unwrap();
        assert_eq!(replay, id);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(reopened.snapshot(true).await.tasks.len(), 1);
    }

    struct MisleadingCompletionProvider {
        actions: Vec<Action>,
        next: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl ModelProvider for MisleadingCompletionProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, context: TurnContext) -> CoreResult<ModelTurn> {
            let index = self.next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(self.actions.get(index).map_or_else(
                || ModelTurn::Answer("Everything succeeded. I read all the files.".into()),
                |action| proposal_turn(context.planning.task_id, action.clone()),
            ))
        }
    }

    #[tokio::test]
    async fn fabricated_model_success_cannot_hide_failed_or_partial_work() {
        for include_successful_read in [false, true] {
            let data = tempdir().unwrap();
            let workspace = tempdir().unwrap();
            let root = workspace.path().canonicalize().unwrap();
            std::fs::write(root.join("present.txt"), "verified fixture content").unwrap();
            let mut actions = Vec::new();
            if include_successful_read {
                actions.push(Action::ReadFile {
                    path: root.join("present.txt"),
                    max_bytes: 1024,
                });
            }
            actions.push(Action::ReadFile {
                path: root.join("missing.txt"),
                max_bytes: 1024,
            });
            let core = SageCore::new_with_secret_store(
                CoreConfig::for_test(data.path()),
                Arc::new(MisleadingCompletionProvider {
                    actions,
                    next: Default::default(),
                }),
                Arc::new(MemorySecretStore::default()),
            )
            .unwrap();
            let id = core
                .submit_scoped(
                    "Read both requested files".into(),
                    None,
                    false,
                    None,
                    vec![crate::contracts::ResourceScope {
                        root,
                        effects: BTreeSet::from([crate::contracts::Effect::Read]),
                    }],
                )
                .await
                .unwrap();
            let task = completed(&core, id).await;
            assert_eq!(
                task.status,
                if include_successful_read {
                    TaskStatus::Partial
                } else {
                    TaskStatus::Failed
                }
            );
            assert_eq!(task.completed_count(), usize::from(include_successful_read));
            assert!(
                task.final_outcome
                    .as_deref()
                    .unwrap()
                    .contains("failed or skipped")
            );
            assert!(
                !task
                    .final_outcome
                    .as_deref()
                    .unwrap()
                    .contains("Everything succeeded")
            );
            let memories = core.store.memories(None, false, 500).unwrap();
            assert!(!memories.iter().any(|memory| memory.id == id));
            let conversation = core
                .store
                .conversations()
                .unwrap()
                .into_iter()
                .find(|conversation| Some(conversation.id) == task.conversation_id)
                .unwrap();
            assert!(!conversation.summary.contains("Everything succeeded"));
            assert!(
                conversation
                    .summary
                    .contains("\"answer_is_execution_evidence\":false")
            );
        }
    }

    #[tokio::test]
    async fn action_status_without_verification_evidence_requires_review() {
        let data = tempdir().unwrap();
        let core = SageCore::new_with_secret_store(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        let mut task = Task::new("Imported task with missing evidence");
        let ModelTurn::Actions(graph) = proposal_turn(
            task.id,
            Action::AskUser {
                question: "Continue?".into(),
            },
        ) else {
            unreachable!()
        };
        task.install_plan(graph).unwrap();
        task.actions.values_mut().next().unwrap().status = ActionStatus::Succeeded;
        core.store.save_task(&mut task).unwrap();
        let id = task.id;
        core.tasks.write().await.insert(id, task);
        core.finish_answer(id, "Verified everything".into())
            .await
            .unwrap();
        let task = core.get_task(id).await.unwrap();
        assert_eq!(task.status, TaskStatus::Interrupted);
        assert_eq!(task.completed_count(), 0);
        assert!(task.final_outcome.unwrap().contains("uncertain effects"));
    }

    #[tokio::test]
    async fn denying_a_native_authentication_decision_never_requires_authentication() {
        let data = tempdir().unwrap();
        let core = SageCore::new_with_secret_store(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        let record = crate::contracts::ApprovalRecord {
            approval_id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            action_id: Uuid::new_v4(),
            digest: "prepared-digest".into(),
            explanation: "fixture".into(),
            resource: "fixture".into(),
            risk: RiskLevel::Privileged,
            expires_at: Utc::now() + ChronoDuration::minutes(1),
            reversible: false,
            requires_native_authentication: true,
        };
        let mut task = Task::new("Native authentication denial fixture");
        task.id = record.task_id;
        task.status = TaskStatus::WaitingForApproval;
        let decision = crate::decisions::DecisionRecord::Approval(record.clone());
        core.store
            .open_decision(&mut task, &decision, &decision.opened())
            .unwrap();
        let (sender, receiver) = oneshot::channel();
        core.pending_approvals.lock().await.insert(
            record.approval_id,
            PendingApproval {
                task_id: record.task_id,
                action_id: record.action_id,
                digest: record.digest.clone(),
                requires_native_authentication: true,
                expires_at: record.expires_at,
                record: record.clone(),
                sender,
            },
        );
        core.resolve_approval(
            record.approval_id,
            record.task_id,
            record.action_id,
            &record.digest,
            ApprovalResolution::Denied,
        )
        .await
        .unwrap();
        assert_eq!(receiver.await.unwrap(), ApprovalResolution::Denied);
        assert!(core.pending_approvals.lock().await.is_empty());
    }

    #[tokio::test]
    async fn questions_survive_reconnect_snapshots_and_cancel_closes_the_waiter() {
        let data = tempdir().unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let mut workers = Vec::new();
        for index in 0..3 {
            let mut task = Task::new(format!("Question fixture {index}"));
            let ModelTurn::Actions(graph) = proposal_turn(
                task.id,
                Action::AskUser {
                    question: format!("Which file for request {index}?"),
                },
            ) else {
                unreachable!()
            };
            let proposal = graph.nodes[0].proposal.clone();
            task.install_plan(graph).unwrap();
            task.status = TaskStatus::Running;
            core.store.save_task(&mut task).unwrap();
            core.tasks.write().await.insert(task.id, task);
            let worker = core.clone();
            workers.push(tokio::spawn(async move {
                worker
                    .await_question(&proposal, format!("Which file for request {index}?"))
                    .await
            }));
        }
        let snapshot = timeout(Duration::from_secs(2), async {
            loop {
                let snapshot = core.snapshot(true).await;
                if snapshot.pending_questions.len() == 3 {
                    break snapshot;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let first = &snapshot.pending_questions[0];
        assert!(
            core.answer_question(
                first.question_id,
                Uuid::new_v4(),
                first.action_id,
                "wrong task".into()
            )
            .await
            .is_err()
        );
        core.answer_question(
            first.question_id,
            first.task_id,
            first.action_id,
            "chosen.txt".into(),
        )
        .await
        .unwrap();
        assert!(
            core.answer_question(
                first.question_id,
                first.task_id,
                first.action_id,
                "replayed.txt".into()
            )
            .await
            .is_err()
        );
        for question in &snapshot.pending_questions[1..] {
            core.control_task(question.task_id, TaskStatus::Cancelled)
                .await
                .unwrap();
            assert!(
                core.answer_question(
                    question.question_id,
                    question.task_id,
                    question.action_id,
                    "too late".into()
                )
                .await
                .is_err()
            );
        }
        let mut answered = 0;
        for worker in workers {
            match timeout(Duration::from_secs(2), worker)
                .await
                .unwrap()
                .unwrap()
            {
                Ok(receipt) => {
                    assert_eq!(receipt.transient_data["answer"], "chosen.txt");
                    answered += 1;
                }
                Err(CoreError::Cancelled) => {}
                other => panic!("unexpected question outcome: {other:?}"),
            }
        }
        assert_eq!(answered, 1);
        assert!(core.snapshot(true).await.pending_questions.is_empty());
        core.store
            .with_connection(|db| {
                let answered: i64 = db.query_row(
                    "SELECT COUNT(*) FROM decisions WHERE state='answered'",
                    [],
                    |row| row.get(0),
                )?;
                let cancelled: i64 = db.query_row(
                    "SELECT COUNT(*) FROM decisions WHERE state='cancelled'",
                    [],
                    |row| row.get(0),
                )?;
                assert_eq!((answered, cancelled), (1, 2));
                Ok(())
            })
            .unwrap();
    }
    #[tokio::test]
    async fn denied_data_release_never_calls_the_provider_and_cannot_replay_approval() {
        let data = tempdir().unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(AnswerProvider {
                calls: calls.clone(),
                remote: true,
            }),
        )
        .unwrap();
        let mut events = core.events().subscribe();
        let id = core
            .submit_task("Use this private project context")
            .await
            .unwrap();
        let (approval_id, action_id, digest) = timeout(Duration::from_secs(5), async {
            loop {
                if let CoreEventKind::ApprovalRequested {
                    approval_id,
                    action_id,
                    digest,
                    ..
                } = events.recv().await.unwrap().kind
                {
                    break (approval_id, action_id, digest);
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(
            core.resolve_approval(
                approval_id,
                id,
                action_id,
                "wrong",
                ApprovalResolution::Denied
            )
            .await
            .is_err()
        );
        core.resolve_approval(
            approval_id,
            id,
            action_id,
            &digest,
            ApprovalResolution::Denied,
        )
        .await
        .unwrap();
        assert!(
            core.resolve_approval(
                approval_id,
                id,
                action_id,
                &digest,
                ApprovalResolution::Approved {
                    native_authentication_satisfied: false
                }
            )
            .await
            .is_err()
        );
        assert_eq!(completed(&core, id).await.status, TaskStatus::Failed);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    struct StalledProvider {
        entered: Arc<Notify>,
        dropped: Arc<Notify>,
    }
    struct OnDrop(Arc<Notify>);
    impl Drop for OnDrop {
        fn drop(&mut self) {
            self.0.notify_one();
        }
    }
    #[async_trait]
    impl ModelProvider for StalledProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, _: TurnContext) -> CoreResult<ModelTurn> {
            let _guard = OnDrop(self.dropped.clone());
            self.entered.notify_one();
            std::future::pending().await
        }
    }
    struct FinalAnswerGate {
        entered: Arc<Notify>,
        release: Arc<Notify>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }
    #[async_trait]
    impl ModelProvider for FinalAnswerGate {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, _: TurnContext) -> CoreResult<ModelTurn> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.entered.notify_one();
            self.release.notified().await;
            Ok(ModelTurn::Answer("Fixture answer saved once".into()))
        }
    }

    #[tokio::test]
    async fn failed_finalization_retries_the_same_answer_without_rerunning_the_model_and_stop_can_replace_it()
     {
        for mode in ["resume", "stop", "checkpoint"] {
            let data = tempdir().unwrap();
            let entered = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let secrets = Arc::new(FailingCheckpointStore::default());
            let core = SageCore::new_with_secret_store(
                CoreConfig::for_test(data.path()),
                Arc::new(FinalAnswerGate {
                    entered: entered.clone(),
                    release: release.clone(),
                    calls: calls.clone(),
                }),
                secrets.clone(),
            )
            .unwrap();
            let id = core.submit_task("Get the fixture answer").await.unwrap();
            timeout(Duration::from_secs(5), entered.notified())
                .await
                .unwrap();
            if mode == "checkpoint" {
                secrets
                    .fail
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            } else {
                core.store.with_connection(|db|{db.execute_batch("CREATE TEMP TRIGGER fail_final_answer BEFORE INSERT ON messages WHEN NEW.role='assistant' BEGIN SELECT RAISE(ABORT,'fixture result storage failure'); END;")?;Ok(())}).unwrap();
            }
            release.notify_one();
            run_exited(&core, id).await;
            if mode != "checkpoint" {
                assert_eq!(
                    core.get_task(id).await.unwrap().status,
                    TaskStatus::Interrupted
                );
                assert!(core.runtime.pending_finish(id).is_some());
                assert!(
                    !core.runtime.is_stopped(id),
                    "Retired authority is separate from user Stop"
                );
                assert!(core.runtime.current(id).is_err());
                assert!(core.runtime.begin(id).is_err());
                assert_eq!(
                    core.store.load_tasks(true).unwrap()[0].status,
                    TaskStatus::Planning
                );
                core.store
                    .with_connection(|db| {
                        db.execute_batch("DROP TRIGGER fail_final_answer")?;
                        Ok(())
                    })
                    .unwrap();
                core.control_task(
                    id,
                    if mode == "stop" {
                        TaskStatus::Cancelled
                    } else {
                        TaskStatus::Running
                    },
                )
                .await
                .unwrap();
            }
            let done = core.get_task(id).await.unwrap();
            assert_eq!(
                done.status,
                if mode == "stop" {
                    TaskStatus::Cancelled
                } else {
                    TaskStatus::Answered
                }
            );
            assert_eq!(
                done.execution_attempt, 0,
                "Retrying persistence does not create a new execution"
            );
            assert!(core.runtime.pending_finish(id).is_none());
            assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
            let messages = core
                .store
                .messages(done.conversation_id.unwrap(), 10)
                .unwrap();
            let replies = messages
                .iter()
                .filter(|message| message.role == "assistant")
                .collect::<Vec<_>>();
            assert_eq!(replies.len(), 1);
            assert_eq!(&replies[0].content, done.final_outcome.as_ref().unwrap());
            if mode != "stop" {
                assert_eq!(
                    done.final_outcome.as_deref(),
                    Some("Fixture answer saved once")
                );
            } else {
                assert!(
                    !done
                        .final_outcome
                        .unwrap()
                        .contains("Fixture answer saved once")
                );
            }
            secrets
                .fail
                .store(false, std::sync::atomic::Ordering::SeqCst);
            core.store.checkpoint_audit(secrets.as_ref()).unwrap();
        }
    }

    async fn stalled_run() -> (tempfile::TempDir, Arc<SageCore>, Uuid, Arc<Notify>) {
        let data = tempdir().unwrap();
        let entered = Arc::new(Notify::new());
        let dropped = Arc::new(Notify::new());
        let core = SageCore::new_with_secret_store(
            CoreConfig::for_test(data.path()),
            Arc::new(StalledProvider {
                entered: entered.clone(),
                dropped: dropped.clone(),
            }),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        let id = core.submit_task("Wait on the fixture model").await.unwrap();
        timeout(Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        (data, core, id, dropped)
    }

    async fn run_exited(core: &SageCore, id: Uuid) {
        timeout(Duration::from_secs(5), async {
            while core.runtime.is_active(id) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("run did not release its execution lease");
    }

    #[tokio::test]
    async fn stop_cancels_despite_storage_failure_and_retains_an_unsaved_stop() {
        let (_data, core, id, dropped) = stalled_run().await;
        let prior = core.get_task(id).await.unwrap();
        core.store.with_connection(|db| {
            db.execute_batch("CREATE TEMP TRIGGER fail_stop BEFORE INSERT ON tasks WHEN NEW.status='cancelled' BEGIN SELECT RAISE(ABORT,'fixture cannot persist Stop'); END;")?;
            Ok(())
        }).unwrap();
        assert!(core.control_task(id, TaskStatus::Cancelled).await.is_err());
        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .unwrap();
        run_exited(&core, id).await;
        assert!(
            core.runtime.is_stopped(id),
            "An unsaved Stop must survive cleanup"
        );
        let snapshot = core.snapshot(false).await;
        let projected = snapshot.tasks.iter().find(|task| task.id == id).unwrap();
        assert_eq!(projected.status, TaskStatus::Cancelled);
        assert!(
            projected
                .final_outcome
                .as_ref()
                .unwrap()
                .contains("Saving this state is still pending")
        );
        assert_eq!(projected.revision, prior.revision);
        let persisted = core
            .store
            .load_tasks(true)
            .unwrap()
            .into_iter()
            .find(|task| task.id == id)
            .unwrap();
        assert_eq!(
            persisted.status, prior.status,
            "The runtime overlay is not fabricated durable evidence"
        );
        assert!(core.control_task(id, TaskStatus::Running).await.is_err());

        core.store
            .with_connection(|db| {
                db.execute_batch("DROP TRIGGER fail_stop;")?;
                Ok(())
            })
            .unwrap();
        core.control_task(id, TaskStatus::Cancelled).await.unwrap();
        assert!(
            !core.runtime.is_stopped(id),
            "Persisted Stop can retire its runtime tombstone"
        );
        let saved = core
            .store
            .load_tasks(true)
            .unwrap()
            .into_iter()
            .find(|task| task.id == id)
            .unwrap();
        assert_eq!(saved.status, TaskStatus::Cancelled);
        assert!(saved.revision > prior.revision);
    }

    #[tokio::test]
    async fn stop_reaches_the_provider_before_the_task_cache_lock_is_released() {
        let (_data, core, id, dropped) = stalled_run().await;
        let cache = core.tasks.write().await;
        let stop_core = core.clone();
        let stop =
            tokio::spawn(async move { stop_core.control_task(id, TaskStatus::Cancelled).await });
        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .unwrap();
        assert!(core.runtime.is_stopped(id));
        assert!(
            !stop.is_finished(),
            "Persistence still needs the locked cache"
        );
        drop(cache);
        stop.await.unwrap().unwrap();
        run_exited(&core, id).await;
        assert_eq!(
            core.get_task(id).await.unwrap().status,
            TaskStatus::Cancelled
        );
    }

    #[tokio::test]
    async fn resume_cannot_replace_a_live_owner_despite_an_interrupted_projection() {
        let (_data, core, id, dropped) = stalled_run().await;
        core.update_task(id, |task| {
            task.status = TaskStatus::Interrupted;
            Ok(())
        })
        .await
        .unwrap();
        assert!(matches!(
            core.control_task(id, TaskStatus::Running).await,
            Err(CoreError::Busy(_))
        ));
        assert!(core.runtime.is_active(id));
        core.control_task(id, TaskStatus::Cancelled).await.unwrap();
        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .unwrap();
        run_exited(&core, id).await;
    }

    struct StalledObserver {
        entered: Arc<Notify>,
        dropped: Arc<Notify>,
    }
    #[async_trait]
    impl Observer for StalledObserver {
        async fn observe(
            &self,
            _: &ActionProposal,
            _: &ExecutionReceipt,
        ) -> CoreResult<crate::observation::Observation> {
            let _guard = OnDrop(self.dropped.clone());
            self.entered.notify_one();
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn stop_drops_a_stalled_observation_but_preserves_the_possible_effect() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let path = workspace.path().join("created-before-stopped-verification");
        let entered = Arc::new(Notify::new());
        let dropped = Arc::new(Notify::new());
        let mut core = SageCore::new_with_secret_store(
            CoreConfig::for_test(data.path()),
            Arc::new(FolderPlanProvider { path: path.clone() }),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        Arc::get_mut(&mut core).unwrap().observer = Arc::new(StalledObserver {
            entered: entered.clone(),
            dropped: dropped.clone(),
        });
        let id = core
            .submit_scoped(
                "Create the fixture folder".into(),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root: workspace.path().into(),
                    effects: BTreeSet::from([crate::contracts::Effect::Create]),
                }],
            )
            .await
            .unwrap();
        timeout(Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        assert!(path.is_dir(), "The effect preceded the observation stall");
        core.control_task(id, TaskStatus::Cancelled).await.unwrap();
        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .unwrap();
        run_exited(&core, id).await;
        let task = core.get_task(id).await.unwrap();
        assert_eq!(task.status, TaskStatus::Cancelled);
        assert_eq!(
            task.actions.values().next().unwrap().status,
            ActionStatus::Uncertain
        );
        assert_eq!(
            task.tool_results.last().unwrap().verdict,
            crate::contracts::Verdict::Uncertain
        );
        assert_eq!(task.completed_count(), 0);
        assert!(
            path.is_dir(),
            "Stop is not compensation for an existing effect"
        );
        core.store
            .with_connection(|db| {
                let state: String = db.query_row(
                    "SELECT state FROM action_journal WHERE run_id=?1",
                    [id.to_string()],
                    |row| row.get(0),
                )?;
                assert_eq!(state, "uncertain");
                Ok(())
            })
            .unwrap();
    }

    #[tokio::test]
    async fn cancellation_drops_an_inflight_model_call_and_persists_cancelled_state() {
        let data = tempdir().unwrap();
        let entered = Arc::new(Notify::new());
        let dropped = Arc::new(Notify::new());
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(StalledProvider {
                entered: entered.clone(),
                dropped: dropped.clone(),
            }),
        )
        .unwrap();
        let id = core.submit_task("Wait on a model").await.unwrap();
        timeout(Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        core.control_task(id, TaskStatus::Cancelled).await.unwrap();
        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .unwrap();
        assert_eq!(
            core.get_task(id).await.unwrap().status,
            TaskStatus::Cancelled
        );
        assert_eq!(
            core.store
                .load_tasks(true)
                .unwrap()
                .into_iter()
                .find(|t| t.id == id)
                .unwrap()
                .status,
            TaskStatus::Cancelled
        );
    }
    #[tokio::test]
    async fn cancellation_prevents_a_queued_mutation_from_starting() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let path = workspace.path().join("never-created");
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(FolderPlanProvider { path: path.clone() }),
        )
        .unwrap();
        let lane = core.effect_ownership.acquire_exclusive().await;
        let mut events = core.events().subscribe();
        let id = core
            .submit_scoped(
                "Create this folder".into(),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root: workspace.path().into(),
                    effects: BTreeSet::from([crate::contracts::Effect::Create]),
                }],
            )
            .await
            .unwrap();
        timeout(Duration::from_secs(5), async {
            loop {
                if matches!(
                    events.recv().await.unwrap().kind,
                    CoreEventKind::ActionProposed { .. }
                ) {
                    break;
                }
            }
        })
        .await
        .unwrap();
        core.control_task(id, TaskStatus::Cancelled).await.unwrap();
        drop(lane);
        timeout(Duration::from_secs(5), async {
            while core.runtime.is_active(id) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(!path.exists());
        assert_eq!(
            core.get_task(id).await.unwrap().status,
            TaskStatus::Cancelled
        );
    }
}
