//! Evidence-backed observations and capability discoveries.
//!
//! Everything in this module is descriptive. A model, application, protocol
//! advertisement, or stored capability can suggest an operation, but none can
//! create broker authority. Execution continues through the normal compiler,
//! policy, approval, capability, and verifier paths.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Duration, Utc};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::contracts::{Effect, Sensitivity};
use crate::error::{CoreError, CoreResult};
use crate::storage::LocalStore;

const MAX_SYSTEM_KEY: usize = 512;
const MAX_LABEL_BYTES: usize = 256;
const MAX_FACTS_PER_OBSERVATION: usize = 512;
pub(crate) const MAX_OBSERVATION_BYTES: usize = 32 * 1024;
const MAX_OBSERVATIONS_PER_SYSTEM: usize = 512;
const MAX_CAPABILITY_PORTS: usize = 16;
const MAX_CAPABILITY_EVIDENCE: usize = 32;
const MAX_LEARNING_TARGETS: usize = 32;
const MAX_SESSION_MINUTES: i64 = 10;
const MAX_SESSION_PROBES: u32 = 20;
const MAX_EXPERIMENT_WINDOW_SECONDS: i64 = 30;
const OBSERVATION_RETENTION_DAYS: i64 = 90;

type LearningSessionStorageRow = (String, String, String, String, String, u32, String, String);
type DispatchedProbeLeaseStorageRow = (
    String,
    String,
    String,
    String,
    String,
    String,
    u32,
    String,
    String,
);
type ProbeCompletionStorageRow = (
    String,
    String,
    String,
    String,
    String,
    String,
    String,
    String,
);

fn bounded_text(value: &str, max_bytes: usize, field: &str) -> CoreResult<()> {
    if value.trim().is_empty()
        || value.len() > max_bytes
        || value.contains('\0')
        || value.chars().any(char::is_control)
    {
        return Err(CoreError::InvalidAction(format!(
            "World-model {field} is empty, oversized, or contains control characters"
        )));
    }
    Ok(())
}

fn digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_key(value: &str) -> CoreResult<()> {
    bounded_text(value, MAX_SYSTEM_KEY, "system identifier")?;
    if value.starts_with('/')
        || value.contains("\\")
        || value.contains('@')
        || value.contains('?')
        || value.contains('#')
    {
        return Err(CoreError::InvalidAction(
            "World-model identifiers cannot contain file paths, credentials, queries, or fragments"
                .into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemKind {
    Application,
    BrowserOrigin,
    Device,
    DataFormat,
}

/// A system's key identifies a bundle, paired peer, browser origin, or media
/// format. It must not be a document path or contain URL credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemDescriptor {
    pub id: Uuid,
    pub kind: SystemKind,
    pub key: String,
    pub label: String,
    /// SHA-256 of the current executable, browser-document, or peer identity.
    pub fingerprint: String,
    pub revision: u64,
    pub updated_at: DateTime<Utc>,
}

impl SystemDescriptor {
    pub fn validate(&self) -> CoreResult<()> {
        validate_key(&self.key)?;
        bounded_text(&self.label, MAX_LABEL_BYTES, "system label")?;
        if !digest(&self.fingerprint) {
            return Err(CoreError::InvalidAction(
                "World-model system fingerprint or revision is invalid".into(),
            ));
        }
        Ok(())
    }
}

/// Origins are evidence labels, never authority. In particular, `Model`
/// records remain hypotheses even when their content looks plausible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOrigin {
    OperatingSystem,
    Application,
    Browser,
    DeviceAdvertisement,
    User,
    Model,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum FactValue {
    Text(String),
    Boolean(bool),
    Number(f64),
    Identifier(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedFact {
    /// A typed, local schema field such as `control.role` or `device.protocol`.
    pub name: String,
    /// Present for interface properties. Causal evidence must bind state to
    /// the exact accessibility identity listed in the approved session.
    pub subject: Option<String>,
    pub value: FactValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationEnvelope {
    pub id: Uuid,
    pub system_id: Uuid,
    pub session_id: Option<Uuid>,
    pub worker_session: Option<String>,
    pub system_fingerprint: String,
    pub origin: EvidenceOrigin,
    pub privacy: Sensitivity,
    pub observed_at: DateTime<Utc>,
    pub facts: Vec<ObservedFact>,
}

impl ObservationEnvelope {
    pub(crate) fn validate(&self) -> CoreResult<usize> {
        if !digest(&self.system_fingerprint)
            || self.facts.is_empty()
            || self.facts.len() > MAX_FACTS_PER_OBSERVATION
        {
            return Err(CoreError::InvalidAction(
                "World-model observation is missing its target identity or bounded facts".into(),
            ));
        }
        if self.privacy >= Sensitivity::Restricted {
            return Err(CoreError::PolicyDenied(
                "Restricted or secret content cannot enter world-model observations".into(),
            ));
        }
        if self.observed_at > Utc::now() + Duration::seconds(30)
            || self.observed_at < Utc::now() - Duration::days(7)
        {
            return Err(CoreError::InvalidAction(
                "World-model observations must be recent".into(),
            ));
        }
        if self.session_id.is_some() != self.worker_session.is_some() {
            return Err(CoreError::PermissionRequired(
                "Experimental observations must identify their approved adapter session".into(),
            ));
        }
        if let Some(worker) = &self.worker_session {
            bounded_text(worker, 128, "adapter session")?;
        }

        let mut last_fact: Option<(&str, Option<&str>)> = None;
        for fact in &self.facts {
            bounded_text(&fact.name, 96, "fact name")?;
            if !fact.name.bytes().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'.' | b'_' | b'-')
            }) {
                return Err(CoreError::InvalidAction(
                    "World-model fact names must use the registered lowercase schema".into(),
                ));
            }
            if let Some(subject) = &fact.subject {
                bounded_text(subject, 256, "fact subject")?;
            }
            let current_key = (fact.name.as_str(), fact.subject.as_deref());
            if last_fact.is_some_and(|prior| prior >= current_key) {
                return Err(CoreError::InvalidAction(
                    "World-model facts must be uniquely sorted by property and subject".into(),
                ));
            }
            last_fact = Some(current_key);
            match &fact.value {
                FactValue::Text(value) | FactValue::Identifier(value) => {
                    let maximum = if matches!(&fact.value, FactValue::Identifier(_)) {
                        MAX_SYSTEM_KEY
                    } else {
                        MAX_LABEL_BYTES
                    };
                    bounded_text(value, maximum, "fact value")?;
                    if crate::redaction::redact_for_persistence(value) != *value {
                        return Err(CoreError::PolicyDenied(
                            "Credential-like content cannot enter world-model observations".into(),
                        ));
                    }
                    if matches!(&fact.value, FactValue::Identifier(_)) {
                        validate_key(value)?;
                    }
                }
                FactValue::Number(value) if !value.is_finite() => {
                    return Err(CoreError::InvalidAction(
                        "World-model numeric facts must be finite".into(),
                    ));
                }
                FactValue::Boolean(_) | FactValue::Number(_) => {}
            }
        }

        let bytes = serde_json::to_vec(self)?.len();
        if bytes > MAX_OBSERVATION_BYTES {
            return Err(CoreError::InvalidAction(
                "World-model observation exceeds the 32 KiB limit".into(),
            ));
        }
        Ok(bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortType {
    Text,
    Boolean,
    Number,
    Bytes,
    Image,
    Video,
    Audio,
    DeviceStream,
    StructuredData,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataPort {
    pub name: String,
    pub value_type: PortType,
    pub max_bytes: u64,
    pub privacy: Sensitivity,
}

impl DataPort {
    pub(crate) fn validate(&self) -> CoreResult<()> {
        bounded_text(&self.name, 96, "port name")?;
        if !self.name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        }) || self.max_bytes == 0
            || self.max_bytes > 64 * 1024 * 1024
        {
            return Err(CoreError::InvalidAction(
                "World-model data port exceeds its schema or size bounds".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preconditions {
    pub observed_state_fact_ids: Vec<Uuid>,
    pub description: String,
}

/// A descriptive claim about an available operation. `executor_id` must
/// resolve to trusted, registered Sage code. A discovered descriptor itself
/// never installs code, enters the executable registry, or issues a grant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityDescriptor {
    pub schema_version: u32,
    pub id: String,
    pub system_id: Uuid,
    pub system_fingerprint: String,
    #[serde(default)]
    pub interface_control_id: Option<String>,
    #[serde(default)]
    pub interface_probe_kind: Option<ProbeKind>,
    pub label: String,
    pub input_ports: Vec<DataPort>,
    pub output_ports: Vec<DataPort>,
    pub preconditions: Preconditions,
    pub effects: BTreeSet<Effect>,
    pub verification: String,
    pub restoration: Option<String>,
    pub cancellation: String,
    pub executor_id: Option<String>,
    pub evidence_ids: Vec<Uuid>,
    pub updated_at: DateTime<Utc>,
}

impl CapabilityDescriptor {
    pub fn validate(&self) -> CoreResult<()> {
        bounded_text(&self.id, 128, "capability identifier")?;
        if !self.id.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b'-' | b':')
        }) || self.schema_version != 1
            || !digest(&self.system_fingerprint)
            || self.input_ports.len() > MAX_CAPABILITY_PORTS
            || self.output_ports.len() > MAX_CAPABILITY_PORTS
            || self.evidence_ids.is_empty()
            || self.evidence_ids.len() > MAX_CAPABILITY_EVIDENCE
            || self.preconditions.observed_state_fact_ids.is_empty()
            || self.preconditions.observed_state_fact_ids.len() > MAX_CAPABILITY_EVIDENCE
            || self
                .preconditions
                .observed_state_fact_ids
                .iter()
                .any(|id| !self.evidence_ids.contains(id))
            || self.evidence_ids.iter().collect::<BTreeSet<_>>().len() != self.evidence_ids.len()
            || self.interface_control_id.is_some() != self.interface_probe_kind.is_some()
            || self
                .interface_control_id
                .as_deref()
                .is_some_and(|control_id| !digest(control_id))
            || self.effects.is_empty()
        {
            return Err(CoreError::InvalidAction(
                "World-model capability descriptor is incomplete or exceeds its bounds".into(),
            ));
        }
        bounded_text(&self.label, MAX_LABEL_BYTES, "capability label")?;
        bounded_text(&self.preconditions.description, 1024, "precondition")?;
        if let Some(control_id) = &self.interface_control_id {
            bounded_text(control_id, 256, "interface control identifier")?;
        }
        bounded_text(&self.verification, 256, "verification contract")?;
        bounded_text(&self.cancellation, 256, "cancellation contract")?;
        if let Some(restoration) = &self.restoration {
            bounded_text(restoration, 256, "restoration contract")?;
        }
        for port in self.input_ports.iter().chain(self.output_ports.iter()) {
            port.validate()?;
        }
        if self
            .input_ports
            .iter()
            .map(|port| port.name.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != self.input_ports.len()
            || self
                .output_ports
                .iter()
                .map(|port| port.name.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                != self.output_ports.len()
        {
            return Err(CoreError::InvalidAction(
                "World-model capability ports must have unique names".into(),
            ));
        }
        if let Some(executor) = &self.executor_id {
            bounded_text(executor, 128, "executor identifier")?;
            if !crate::features::manifests()
                .iter()
                .any(|manifest| manifest.enabled && manifest.id == *executor)
            {
                return Err(CoreError::ExecutorUnavailable(
                    "Discovered capabilities can bind only to trusted registered executors".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldRelationKind {
    Exposes,
    Requires,
    Produces,
    Consumes,
    Controls,
    ConnectsTo,
    Displays,
}

impl WorldRelationKind {
    fn evidence_fact_name(self) -> &'static str {
        match self {
            Self::Exposes => "world.relation.exposes",
            Self::Requires => "world.relation.requires",
            Self::Produces => "world.relation.produces",
            Self::Consumes => "world.relation.consumes",
            Self::Controls => "world.relation.controls",
            Self::ConnectsTo => "world.relation.connects_to",
            Self::Displays => "world.relation.displays",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldRelation {
    pub id: Uuid,
    pub from_system: Uuid,
    pub kind: WorldRelationKind,
    pub to_system: Uuid,
    pub evidence_id: Uuid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEvidenceState {
    HypothesisOnly,
    PassivelyObserved,
    ReversiblyExperimented,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityAssessment {
    pub descriptor: CapabilityDescriptor,
    /// Descriptive evidence strength. This is deliberately not a claim that
    /// the capability is safe or authorized for execution.
    pub evidence_state: CapabilityEvidenceState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeKind {
    RestoreSliderValue,
    RestoreToggleState,
}

impl ProbeKind {
    fn name(self) -> &'static str {
        match self {
            Self::RestoreSliderValue => "restore_slider_value",
            Self::RestoreToggleState => "restore_toggle_state",
        }
    }
}

fn observation_supports_safe_probe(
    observation: &ObservationEnvelope,
    control_id: &str,
    kind: ProbeKind,
) -> bool {
    let get = |name: &str| {
        observation
            .facts
            .iter()
            .find(|fact| fact.name == name && fact.subject.as_deref() == Some(control_id))
            .map(|fact| &fact.value)
    };
    let role = match get("control.role") {
        Some(FactValue::Text(value)) => value.as_str(),
        _ => return false,
    };
    let enabled = matches!(get("control.enabled"), Some(FactValue::Boolean(true)));
    let safe_label = match get("control.label") {
        Some(FactValue::Text(value)) if !value.trim().is_empty() => {
            let normalized = value.to_ascii_lowercase();
            let denied = [
                "send",
                "submit",
                "delete",
                "remove",
                "purchase",
                "buy",
                "pay",
                "publish",
                "security",
                "password",
                "privacy",
                "account",
                "install",
                "uninstall",
                "reset",
                "erase",
                "wipe",
                "logout",
                "sign out",
                "share",
                "connect",
                "disconnect",
                "microphone",
                "camera",
                "location",
                "firewall",
                "encryption",
            ];
            !denied.iter().any(|term| normalized.contains(term))
        }
        _ => false,
    };
    let value_matches = match (kind, role, get("control.value")) {
        (ProbeKind::RestoreSliderValue, "slider", Some(FactValue::Number(value))) => {
            let step = match get("control.step") {
                Some(FactValue::Number(step)) => *step,
                _ => return false,
            };
            let minimum = match get("control.minimum") {
                Some(FactValue::Number(minimum)) => *minimum,
                _ => return false,
            };
            let maximum = match get("control.maximum") {
                Some(FactValue::Number(maximum)) => *maximum,
                _ => return false,
            };
            value.is_finite()
                && step.is_finite()
                && step > 0.0
                && step <= 1.0
                && minimum.is_finite()
                && maximum.is_finite()
                && minimum < maximum
                && *value >= minimum
                && *value <= maximum
        }
        (
            ProbeKind::RestoreToggleState,
            "toggle" | "checkbox" | "switch",
            Some(FactValue::Boolean(_)),
        ) => true,
        _ => false,
    };
    observation.origin == EvidenceOrigin::OperatingSystem
        && observation.session_id.is_none()
        && observation.worker_session.is_none()
        && enabled
        && safe_label
        && value_matches
}

fn passive_control_capability_id(
    system_fingerprint: &str,
    control_id: &str,
    kind: ProbeKind,
) -> String {
    let role = match kind {
        ProbeKind::RestoreSliderValue => "slider",
        ProbeKind::RestoreToggleState => "toggle",
    };
    let mut hasher = Sha256::new();
    hasher.update(system_fingerprint.as_bytes());
    hasher.update(b"\0");
    hasher.update(control_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(kind.name().as_bytes());
    let control_digest = format!("{:x}", hasher.finalize());
    format!("ui.control.{role}.{}", &control_digest[..24])
}

pub(crate) fn safe_probe_candidates(
    observation: &ObservationEnvelope,
) -> BTreeMap<String, ProbeKind> {
    let mut candidates = BTreeMap::new();
    for fact in &observation.facts {
        if fact.name != "control.role" {
            continue;
        }
        let Some(control_id) = &fact.subject else {
            continue;
        };
        for kind in [ProbeKind::RestoreSliderValue, ProbeKind::RestoreToggleState] {
            if observation_supports_safe_probe(observation, control_id, kind) {
                candidates.insert(control_id.clone(), kind);
                break;
            }
        }
    }
    candidates
}

/// Turn a visible, low-risk control into an evidence-linked hypothesis. This
/// record is useful for later review and planning, but deliberately has no
/// executor: passive observation cannot prove the control's causal effect.
pub(crate) fn passive_control_capability(
    observation: &ObservationEnvelope,
    control_id: &str,
    kind: ProbeKind,
) -> CoreResult<CapabilityDescriptor> {
    if !observation_supports_safe_probe(observation, control_id, kind) {
        return Err(CoreError::PermissionRequired(
            "Control no longer qualifies for a passive capability hypothesis".into(),
        ));
    }
    let fact = |name: &str| {
        observation
            .facts
            .iter()
            .find(|fact| fact.name == name && fact.subject.as_deref() == Some(control_id))
            .map(|fact| &fact.value)
    };
    let label = match fact("control.label") {
        Some(FactValue::Text(value)) => value,
        _ => {
            return Err(CoreError::VerificationFailed(
                "Observed control label is missing from its evidence".into(),
            ));
        }
    };
    let mut capability_label = format!("Set {}", label.trim());
    while capability_label.len() > MAX_LABEL_BYTES {
        capability_label.pop();
    }
    let (role, port_type, byte_limit) = match kind {
        ProbeKind::RestoreSliderValue => ("slider", PortType::Number, 8),
        ProbeKind::RestoreToggleState => ("toggle", PortType::Boolean, 8),
    };
    let descriptor = CapabilityDescriptor {
        schema_version: 1,
        id: passive_control_capability_id(&observation.system_fingerprint, control_id, kind),
        system_id: observation.system_id,
        system_fingerprint: observation.system_fingerprint.clone(),
        interface_control_id: Some(control_id.into()),
        interface_probe_kind: Some(kind),
        label: capability_label,
        input_ports: vec![DataPort {
            name: "value".into(),
            value_type: port_type,
            max_bytes: byte_limit,
            privacy: Sensitivity::Private,
        }],
        output_ports: vec![DataPort {
            name: "observed_value".into(),
            value_type: port_type,
            max_bytes: byte_limit,
            privacy: Sensitivity::Private,
        }],
        preconditions: Preconditions {
            observed_state_fact_ids: vec![observation.id],
            description: format!(
                "Current signed application and visible enabled {role} control; causal effect is untested"
            ),
        },
        effects: BTreeSet::from([Effect::ControlApplication]),
        verification: "Fresh independent OS readback is required after any future dispatch".into(),
        restoration: Some(
            "Restore the exact captured initial value and independently verify before release"
                .into(),
        ),
        cancellation: "Stop new dispatch and settle any started effect before releasing the target"
            .into(),
        executor_id: None,
        evidence_ids: vec![observation.id],
        updated_at: observation.observed_at,
    };
    descriptor.validate()?;
    Ok(descriptor)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningSessionState {
    Active,
    Revoked,
    Expired,
    InterruptedNeedsReview,
    RestorationFailed,
}

/// This durable approval is exact to one system fingerprint and an explicit
/// list of reversible control probes. It cannot authorize arbitrary clicks,
/// text entry, network calls, file changes, or OS security settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningSession {
    pub id: Uuid,
    pub system_id: Uuid,
    pub system_fingerprint: String,
    pub worker_session: String,
    pub permitted_probes: BTreeMap<String, ProbeKind>,
    pub state: LearningSessionState,
    pub probes_started: u32,
    pub approved_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeLease {
    pub id: Uuid,
    pub session_id: Uuid,
    pub system_id: Uuid,
    pub system_fingerprint: String,
    pub expected_process_id: u32,
    pub worker_session: String,
    pub control_id: String,
    pub kind: ProbeKind,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeEvidence {
    pub before_observation: Uuid,
    pub changed_observation: Uuid,
    pub restored_observation: Uuid,
}

impl LearningSession {
    fn validate_request(
        system_fingerprint: &str,
        worker_session: &str,
        permitted_probes: &BTreeMap<String, ProbeKind>,
        expires_at: DateTime<Utc>,
    ) -> CoreResult<()> {
        if !digest(system_fingerprint)
            || permitted_probes.is_empty()
            || permitted_probes.len() > MAX_LEARNING_TARGETS
            || expires_at <= Utc::now()
            || expires_at > Utc::now() + Duration::minutes(MAX_SESSION_MINUTES)
        {
            return Err(CoreError::PermissionRequired(
                "A learning approval must name exact controls and expire within ten minutes".into(),
            ));
        }
        bounded_text(worker_session, 128, "adapter session")?;
        for id in permitted_probes.keys() {
            bounded_text(id, 256, "accessibility control identifier")?;
        }
        Ok(())
    }
}

impl LocalStore {
    /// Installs the additive, encrypted world-model schema in a SQLCipher store
    /// or in the volatile, locked startup store.
    pub(crate) fn migrate_world_model(&self) -> CoreResult<()> {
        self.with_connection(|db| {
            db.execute_batch(
                "BEGIN IMMEDIATE;
                 CREATE TABLE IF NOT EXISTS world_systems(
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    system_key TEXT NOT NULL,
                    label TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision>0),
                    updated_at TEXT NOT NULL,
                    UNIQUE(kind,system_key)
                 );
                 CREATE TABLE IF NOT EXISTS world_observations(
                    id TEXT PRIMARY KEY,
                    system_id TEXT NOT NULL REFERENCES world_systems(id) ON DELETE CASCADE,
                    session_id TEXT,
                    fingerprint TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    privacy TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS world_observations_system_time
                    ON world_observations(system_id,observed_at DESC);
                 CREATE TABLE IF NOT EXISTS world_capabilities(
                    system_id TEXT NOT NULL REFERENCES world_systems(id) ON DELETE CASCADE,
                    capability_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    descriptor_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(system_id,capability_id)
                 );
                 CREATE TABLE IF NOT EXISTS world_relations(
                    id TEXT PRIMARY KEY,
                    from_system TEXT NOT NULL REFERENCES world_systems(id) ON DELETE CASCADE,
                    relation_kind TEXT NOT NULL,
                    to_system TEXT NOT NULL REFERENCES world_systems(id) ON DELETE CASCADE,
                    evidence_id TEXT NOT NULL REFERENCES world_observations(id) ON DELETE CASCADE
                 );
                 CREATE TABLE IF NOT EXISTS world_learning_sessions(
                    id TEXT PRIMARY KEY,
                    system_id TEXT NOT NULL REFERENCES world_systems(id) ON DELETE CASCADE,
                    fingerprint TEXT NOT NULL,
                    worker_session TEXT NOT NULL,
                    permitted_probes_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('active','revoked','expired','interrupted_needs_review','restoration_failed')),
                    probes_started INTEGER NOT NULL DEFAULT 0 CHECK(probes_started BETWEEN 0 AND 20),
                    approved_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS world_learning_active
                    ON world_learning_sessions(state,expires_at);
                 CREATE TABLE IF NOT EXISTS world_probe_leases(
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES world_learning_sessions(id) ON DELETE CASCADE,
                    control_id TEXT NOT NULL,
                    probe_kind TEXT NOT NULL,
                    expected_process_id INTEGER NOT NULL DEFAULT 0,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('dispatched','restored','failed'))
                 );
                 CREATE INDEX IF NOT EXISTS world_probe_pending
                    ON world_probe_leases(session_id,state,expires_at);
                 CREATE TABLE IF NOT EXISTS world_transitions(
                    probe_id TEXT PRIMARY KEY,
                    system_id TEXT NOT NULL,
                    control_id TEXT NOT NULL,
                    probe_kind TEXT NOT NULL,
                    before_observation TEXT NOT NULL,
                    changed_observation TEXT NOT NULL,
                    restored_observation TEXT NOT NULL,
                    before_value_sha256 TEXT NOT NULL,
                    changed_value_sha256 TEXT NOT NULL,
                    restored_value_sha256 TEXT NOT NULL,
                    confirmed_at TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS world_controllers(
                    controller_id TEXT PRIMARY KEY CHECK(length(controller_id)=64),
                    system_id TEXT NOT NULL REFERENCES world_systems(id) ON DELETE CASCADE,
                    system_fingerprint TEXT NOT NULL,
                    interface_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('draft','reviewed','disabled','invalidated')),
                    revision INTEGER NOT NULL CHECK(revision>0),
                    controller_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    reviewed_observation_id TEXT REFERENCES world_observations(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                 );
                 CREATE INDEX IF NOT EXISTS world_controllers_system_status
                    ON world_controllers(system_id,status,updated_at DESC);
                 INSERT OR IGNORE INTO schema_migrations(version,applied_at)
                    VALUES(7,CURRENT_TIMESTAMP);
                 COMMIT;",
            )?;
            let has_process_column = db.query_row(
                "SELECT EXISTS(SELECT 1 FROM pragma_table_info('world_probe_leases') WHERE name='expected_process_id')",
                [],
                |row| row.get::<_, bool>(0),
            )?;
            if !has_process_column {
                db.execute_batch(
                    "ALTER TABLE world_probe_leases ADD COLUMN expected_process_id INTEGER NOT NULL DEFAULT 0;",
                )?;
            }
            db.execute(
                "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(8,CURRENT_TIMESTAMP)",
                [],
            )?;
            db.execute(
                "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(9,CURRENT_TIMESTAMP)",
                [],
            )?;
            Ok(())
        })
    }

    pub(crate) fn ensure_world_storage_unlocked(&self) -> CoreResult<()> {
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "World-model and learning data are unavailable until Sage storage is unlocked"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Observe a system identity. A changed identity increments its revision;
    /// old descriptors remain stored for provenance and are stale by digest.
    pub(crate) fn observe_system(
        &self,
        mut system: SystemDescriptor,
    ) -> CoreResult<SystemDescriptor> {
        system.validate()?;
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let existing: Option<(String, String, u64)> = db
                .query_row(
                    "SELECT id,fingerprint,revision FROM world_systems WHERE kind=?1 AND system_key=?2",
                    params![serde_json::to_string(&system.kind)?, &system.key],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .optional()?;
            let tx = db.transaction()?;
            if let Some((id, old_fingerprint, revision)) = existing {
                let existing_id = Uuid::parse_str(&id)?;
                system.id = existing_id;
                let identity_changed = old_fingerprint != system.fingerprint;
                if identity_changed {
                    tx.execute(
                        "DELETE FROM world_relations WHERE from_system=?1 OR to_system=?1",
                        [system.id.to_string()],
                    )?;
                    tx.execute(
                        "UPDATE world_controllers SET status='invalidated',revision=MIN(revision+1,9223372036854775807),updated_at=?2 WHERE system_id=?1 AND status IN ('draft','reviewed')",
                        params![system.id.to_string(), system.updated_at.to_rfc3339()],
                    )?;
                }
                system.revision = revision.saturating_add(u64::from(identity_changed));
                tx.execute(
                    "UPDATE world_systems SET label=?2,fingerprint=?3,revision=?4,updated_at=?5 WHERE id=?1",
                    params![system.id.to_string(), &system.label, &system.fingerprint, system.revision, system.updated_at.to_rfc3339()],
                )?;
            } else {
                if system.id.is_nil() {
                    system.id = Uuid::new_v4();
                }
                system.revision = 1;
                tx.execute(
                    "INSERT INTO world_systems(id,kind,system_key,label,fingerprint,revision,updated_at) VALUES(?1,?2,?3,?4,?5,?6,?7)",
                    params![system.id.to_string(), serde_json::to_string(&system.kind)?, &system.key, &system.label, &system.fingerprint, system.revision, system.updated_at.to_rfc3339()],
                )?;
            }
            tx.execute(
                "UPDATE world_learning_sessions SET state='interrupted_needs_review' WHERE system_id=?1 AND fingerprint<>?2 AND state='active'",
                params![system.id.to_string(), &system.fingerprint],
            )?;
            tx.commit()?;
            Ok(system)
        })
    }

    pub(crate) fn observed_systems(&self) -> CoreResult<Vec<SystemDescriptor>> {
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let mut query = db.prepare(
                "SELECT id,kind,system_key,label,fingerprint,revision,updated_at FROM world_systems ORDER BY updated_at DESC,id LIMIT 2048",
            )?;
            let rows = query.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, u64>(5)?,
                    row.get::<_, String>(6)?,
                ))
            })?;
            let mut systems = Vec::new();
            for row in rows {
                let (id, kind, key, label, fingerprint, revision, updated_at) = row?;
                systems.push(SystemDescriptor {
                    id: Uuid::parse_str(&id)?,
                    kind: serde_json::from_str(&kind)?,
                    key,
                    label,
                    fingerprint,
                    revision,
                    updated_at: DateTime::parse_from_rfc3339(&updated_at)?.to_utc(),
                });
            }
            Ok(systems)
        })
    }

    pub(crate) fn record_world_observation(
        &self,
        observation: &ObservationEnvelope,
    ) -> CoreResult<()> {
        let byte_count = observation.validate()?;
        self.ensure_world_storage_unlocked()?;
        let payload = serde_json::to_string(observation)?;
        let payload_digest = format!("{:x}", Sha256::digest(payload.as_bytes()));
        let privacy = serde_json::to_string(&observation.privacy)?;
        let origin = serde_json::to_string(&observation.origin)?;
        let expires_at =
            (observation.observed_at + Duration::days(OBSERVATION_RETENTION_DAYS)).to_rfc3339();

        self.with_connection(|db| {
            let tx = db.transaction()?;
            let target: Option<String> = tx
                .query_row(
                    "SELECT fingerprint FROM world_systems WHERE id=?1",
                    [observation.system_id.to_string()],
                    |row| row.get(0),
                )
                .optional()?;
            if target.as_deref() != Some(&observation.system_fingerprint) {
                return Err(Box::new(CoreError::VerificationFailed(
                    "Observation belongs to a missing or changed system identity".into(),
                )));
            }

            if let Some(session_id) = observation.session_id {
                let session: Option<(String, String, String)> = tx
                    .query_row(
                        "SELECT system_id,fingerprint,worker_session FROM world_learning_sessions WHERE id=?1 AND state='active' AND expires_at>?2",
                        params![session_id.to_string(), Utc::now().to_rfc3339()],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .optional()?;
                let Some((system_id, fingerprint, worker_session)) = session else {
                    return Err(Box::new(CoreError::PermissionRequired(
                        "Experimental observation has no active, unexpired learning approval".into(),
                    )));
                };
                if system_id != observation.system_id.to_string()
                    || fingerprint != observation.system_fingerprint
                    || observation.worker_session.as_deref() != Some(worker_session.as_str())
                    || observation.origin != EvidenceOrigin::OperatingSystem
                {
                    return Err(Box::new(CoreError::PermissionRequired(
                        "Experimental evidence must come from the approved OS adapter, target, and session".into(),
                    )));
                }
            }

            let existing: Option<String> = tx
                .query_row(
                    "SELECT payload_sha256 FROM world_observations WHERE id=?1",
                    [observation.id.to_string()],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(existing) = existing {
                if existing == payload_digest {
                    tx.commit()?;
                    return Ok(());
                }
                return Err(Box::new(CoreError::Protocol(
                    "Observation identifier was replayed with different evidence".into(),
                )));
            }

            tx.execute(
                "DELETE FROM world_observations WHERE expires_at<=?1",
                [Utc::now().to_rfc3339()],
            )?;
            let count: u64 = tx.query_row(
                "SELECT COUNT(*) FROM world_observations WHERE system_id=?1",
                [observation.system_id.to_string()],
                |row| row.get(0),
            )?;
            let retained_bytes: u64 = tx.query_row(
                "SELECT COALESCE(SUM(length(payload_json)),0) FROM world_observations WHERE system_id=?1",
                [observation.system_id.to_string()],
                |row| row.get(0),
            )?;
            if count >= MAX_OBSERVATIONS_PER_SYSTEM as u64
                || retained_bytes.saturating_add(byte_count as u64) > 16 * 1024 * 1024
            {
                return Err(Box::new(CoreError::PermissionRequired(
                    "This system's bounded discovery history is full; forget old evidence before collecting more".into(),
                )));
            }

            tx.execute(
                "INSERT INTO world_observations(id,system_id,session_id,fingerprint,origin,privacy,observed_at,expires_at,payload_json,payload_sha256) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                params![observation.id.to_string(), observation.system_id.to_string(), observation.session_id.map(|id| id.to_string()), &observation.system_fingerprint, origin, privacy, observation.observed_at.to_rfc3339(), expires_at, payload, payload_digest],
            )?;
            crate::storage::write_audit(
                &tx,
                observation.session_id,
                None,
                "world_observation_recorded",
                &serde_json::json!({"observation_id":observation.id,"system_id":observation.system_id,"origin":observation.origin,"fact_count":observation.facts.len(),"payload_sha256":payload_digest}),
            )?;
            tx.commit()?;
            Ok(())
        })
    }

    /// Persist the exact before/change/restore readbacks for one active lease
    /// in a single transaction. A late receipt may settle an interrupted run,
    /// but only while this lease is still dispatched and unexpired.
    pub(crate) fn record_probe_observations(
        &self,
        lease_id: Uuid,
        observations: &[ObservationEnvelope; 3],
        now: DateTime<Utc>,
    ) -> CoreResult<()> {
        let mut prepared = Vec::with_capacity(3);
        let mut ids = BTreeSet::new();
        for observation in observations {
            let byte_count = observation.validate()?;
            if !ids.insert(observation.id) {
                return Err(CoreError::InvalidAction(
                    "Probe readbacks must have distinct observation identifiers".into(),
                ));
            }
            prepared.push((
                serde_json::to_string(observation)?,
                serde_json::to_string(&observation.privacy)?,
                serde_json::to_string(&observation.origin)?,
                byte_count,
            ));
        }
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let lease: Option<(String, String, String, String, String, String, String)> = tx
                .query_row(
                    "SELECT p.session_id,s.system_id,s.fingerprint,s.worker_session,p.control_id,p.issued_at,p.expires_at FROM world_probe_leases p JOIN world_learning_sessions s ON s.id=p.session_id WHERE p.id=?1 AND p.state='dispatched' AND s.state IN ('active','interrupted_needs_review')",
                    [lease_id.to_string()],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?)),
                )
                .optional()?;
            let Some((session_text, system_text, fingerprint, worker_session, control_id, issued_text, expiry_text)) = lease else {
                return Err(Box::new(CoreError::PermissionRequired(
                    "Probe receipt has no unsettled approval-bound lease".into(),
                )));
            };
            let session_id = Uuid::parse_str(&session_text)?;
            let issued_at = DateTime::parse_from_rfc3339(&issued_text)?.to_utc();
            let expires_at = DateTime::parse_from_rfc3339(&expiry_text)?.to_utc();
            if expires_at <= now
                || observations.iter().any(|observation| {
                    observation.system_id.to_string() != system_text
                        || observation.session_id != Some(session_id)
                        || observation.worker_session.as_deref() != Some(worker_session.as_str())
                        || observation.system_fingerprint != fingerprint
                        || observation.origin != EvidenceOrigin::OperatingSystem
                        || observation.observed_at < issued_at
                        || observation.observed_at > now
                        || observation.observed_at > expires_at
                        || observation.facts.len() != 1
                        || observation.facts[0].name != "control.value"
                        || observation.facts[0].subject.as_deref() != Some(control_id.as_str())
                })
            {
                return Err(Box::new(CoreError::VerificationFailed(
                    "Probe receipt is stale or not bound to its exact lease, system, worker and control".into(),
                )));
            }
            if !(observations[0].observed_at < observations[1].observed_at
                && observations[1].observed_at < observations[2].observed_at)
            {
                return Err(Box::new(CoreError::VerificationFailed(
                    "Probe readbacks must be strictly ordered before, changed, and restored".into(),
                )));
            }
            let target_fingerprint: Option<String> = tx
                .query_row(
                    "SELECT fingerprint FROM world_systems WHERE id=?1",
                    [&system_text],
                    |row| row.get(0),
                )
                .optional()?;
            if target_fingerprint.as_deref() != Some(fingerprint.as_str()) {
                return Err(Box::new(CoreError::VerificationFailed(
                    "Probe receipt belongs to a changed application identity".into(),
                )));
            }
            tx.execute(
                "DELETE FROM world_observations WHERE expires_at<=?1",
                [now.to_rfc3339()],
            )?;
            let mut count: u64 = tx.query_row(
                "SELECT COUNT(*) FROM world_observations WHERE system_id=?1",
                [&system_text],
                |row| row.get(0),
            )?;
            let mut retained_bytes: u64 = tx.query_row(
                "SELECT COALESCE(SUM(length(payload_json)),0) FROM world_observations WHERE system_id=?1",
                [&system_text],
                |row| row.get(0),
            )?;
            for (observation, (payload, privacy, origin, byte_count)) in observations.iter().zip(prepared.iter()) {
                let replayed: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM world_observations WHERE id=?1)",
                    [observation.id.to_string()],
                    |row| row.get(0),
                )?;
                if replayed {
                    return Err(Box::new(CoreError::Protocol(
                        "Probe observation identifiers cannot be replayed".into(),
                    )));
                }
                count = count.saturating_add(1);
                retained_bytes = retained_bytes.saturating_add(*byte_count as u64);
                if count > MAX_OBSERVATIONS_PER_SYSTEM as u64
                    || retained_bytes > 16 * 1024 * 1024
                {
                    return Err(Box::new(CoreError::PermissionRequired(
                        "This system's bounded discovery history is full".into(),
                    )));
                }
                let payload_digest = format!("{:x}", Sha256::digest(payload.as_bytes()));
                let retention_expiry = (observation.observed_at
                    + Duration::days(OBSERVATION_RETENTION_DAYS))
                    .to_rfc3339();
                tx.execute(
                    "INSERT INTO world_observations(id,system_id,session_id,fingerprint,origin,privacy,observed_at,expires_at,payload_json,payload_sha256) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                    params![observation.id.to_string(),system_text,session_text,fingerprint,origin,privacy,observation.observed_at.to_rfc3339(),retention_expiry,payload,payload_digest],
                )?;
                crate::storage::write_audit(
                    &tx,
                    Some(session_id),
                    None,
                    "world_probe_observation_recorded",
                    &serde_json::json!({"observation_id":observation.id,"probe_id":lease_id,"control_id":control_id,"payload_sha256":payload_digest}),
                )?;
            }
            tx.commit()?;
            Ok(())
        })
    }

    pub(crate) fn observations_for_system(
        &self,
        system_id: Uuid,
        after: Option<DateTime<Utc>>,
    ) -> CoreResult<Vec<ObservationEnvelope>> {
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let mut query = db.prepare(
                "SELECT payload_json FROM world_observations WHERE system_id=?1 AND expires_at>?2 AND (?3 IS NULL OR observed_at>?3) ORDER BY observed_at,id LIMIT 128",
            )?;
            query.query_map(
                params![system_id.to_string(), Utc::now().to_rfc3339(), after.map(|value| value.to_rfc3339())],
                |row| row.get::<_, String>(0),
            )?
            .map(|row| Ok(serde_json::from_str(&row?)?))
            .collect()
        })
    }

    /// Store a candidate only when its evidence is current, local, and scoped
    /// to the same system. Confidence remains evidence-derived and descriptive.
    pub(crate) fn record_capability_candidate(
        &self,
        capability: &CapabilityDescriptor,
    ) -> CoreResult<()> {
        capability.validate()?;
        if capability
            .preconditions
            .observed_state_fact_ids
            .iter()
            .any(|id| !capability.evidence_ids.contains(id))
        {
            return Err(CoreError::InvalidAction(
                "Capability preconditions must cite observations included in the descriptor evidence".into(),
            ));
        }
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let fingerprint: Option<String> = tx
                .query_row(
                    "SELECT fingerprint FROM world_systems WHERE id=?1",
                    [capability.system_id.to_string()],
                    |row| row.get(0),
                )
                .optional()?;
            if fingerprint.as_deref() != Some(&capability.system_fingerprint) {
                return Err(Box::new(CoreError::VerificationFailed(
                    "Capability candidate is stale or belongs to a different system".into(),
                )));
            }
            let mut capability_to_store = capability.clone();
            if let (Some(control_id), Some(kind)) = (
                capability_to_store.interface_control_id.as_deref(),
                capability_to_store.interface_probe_kind,
            ) {
                let latest: Option<(String, String, String)> = tx
                    .query_row(
                        "SELECT before_observation,changed_observation,restored_observation FROM world_transitions WHERE system_id=?1 AND control_id=?2 AND probe_kind=?3 ORDER BY confirmed_at DESC,probe_id DESC LIMIT 1",
                        params![capability.system_id.to_string(), control_id, kind.name()],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .optional()?;
                if let Some((before, changed, restored)) = latest {
                    let ids = [before, changed, restored]
                        .into_iter()
                        .map(|id| Uuid::parse_str(&id))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut all_current = true;
                    for evidence_id in &ids {
                        let current: bool = tx.query_row(
                            "SELECT EXISTS(SELECT 1 FROM world_observations WHERE id=?1 AND system_id=?2 AND fingerprint=?3 AND expires_at>?4)",
                            params![evidence_id.to_string(), capability.system_id.to_string(), &capability.system_fingerprint, Utc::now().to_rfc3339()],
                            |row| row.get(0),
                        )?;
                        all_current &= current;
                    }
                    if all_current {
                        let mut evidence = capability_to_store
                            .preconditions
                            .observed_state_fact_ids
                            .iter()
                            .copied()
                            .collect::<BTreeSet<_>>();
                        evidence.extend(ids);
                        capability_to_store.evidence_ids = evidence.into_iter().collect();
                    }
                }
            }
            capability_to_store.validate()?;
            for evidence_id in &capability_to_store.evidence_ids {
                let valid: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM world_observations WHERE id=?1 AND system_id=?2 AND fingerprint=?3 AND expires_at>?4)",
                    params![evidence_id.to_string(), capability_to_store.system_id.to_string(), &capability_to_store.system_fingerprint, Utc::now().to_rfc3339()],
                    |row| row.get(0),
                )?;
                if !valid {
                    return Err(Box::new(CoreError::VerificationFailed(
                        "Capability evidence is missing, expired, or belongs to another target".into(),
                    )));
                }
            }
            let serialized = serde_json::to_string(&capability_to_store)?;
            tx.execute(
                "INSERT INTO world_capabilities(system_id,capability_id,fingerprint,descriptor_json,updated_at) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(system_id,capability_id) DO UPDATE SET fingerprint=excluded.fingerprint,descriptor_json=excluded.descriptor_json,updated_at=excluded.updated_at",
                params![capability_to_store.system_id.to_string(), &capability_to_store.id, &capability_to_store.system_fingerprint, serialized, capability_to_store.updated_at.to_rfc3339()],
            )?;
            crate::storage::write_audit(
                &tx,
                None,
                None,
                "world_capability_candidate_recorded",
                &serde_json::json!({"system_id":capability_to_store.system_id,"capability_id":capability_to_store.id,"executor_id":capability_to_store.executor_id,"evidence_count":capability_to_store.evidence_ids.len()}),
            )?;
            tx.commit()?;
            Ok(())
        })
    }

    pub(crate) fn capability_candidates(
        &self,
        system_id: Uuid,
    ) -> CoreResult<Vec<CapabilityDescriptor>> {
        Ok(self
            .capability_assessments(system_id)?
            .into_iter()
            .map(|assessment| assessment.descriptor)
            .collect())
    }

    pub(crate) fn capability_assessments(
        &self,
        system_id: Uuid,
    ) -> CoreResult<Vec<CapabilityAssessment>> {
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let mut query = db.prepare(
                "SELECT c.descriptor_json,c.fingerprint FROM world_capabilities c JOIN world_systems s ON s.id=c.system_id AND s.fingerprint=c.fingerprint WHERE c.system_id=?1 ORDER BY c.capability_id LIMIT 256",
            )?;
            let rows = query.query_map([system_id.to_string()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut assessments = Vec::new();
            for row in rows {
                let (serialized, fingerprint) = row?;
                let descriptor: CapabilityDescriptor = serde_json::from_str(&serialized)?;
                descriptor.validate()?;
                let mut saw_passive = false;
                let mut saw_probe = false;
                let mut all_evidence_current = true;
                for evidence_id in &descriptor.evidence_ids {
                    let origin: Option<String> = db
                        .query_row(
                            "SELECT origin FROM world_observations WHERE id=?1 AND system_id=?2 AND fingerprint=?3 AND expires_at>?4",
                            params![evidence_id.to_string(), system_id.to_string(), &fingerprint, Utc::now().to_rfc3339()],
                            |row| row.get(0),
                        )
                        .optional()?;
                    if let Some(origin) = origin {
                        let parsed: EvidenceOrigin = serde_json::from_str(&origin)?;
                        saw_passive |= parsed != EvidenceOrigin::Model;
                    } else {
                        all_evidence_current = false;
                    }
                    let supports_probe: bool = db.query_row(
                        "SELECT EXISTS(SELECT 1 FROM world_transitions t WHERE t.system_id=?1 AND (t.before_observation=?2 OR t.changed_observation=?2 OR t.restored_observation=?2))",
                        params![system_id.to_string(), evidence_id.to_string()],
                        |row| row.get(0),
                    )?;
                    saw_probe |= supports_probe;
                }
                if !all_evidence_current {
                    continue;
                }
                let evidence_state = if saw_probe {
                    CapabilityEvidenceState::ReversiblyExperimented
                } else if saw_passive {
                    CapabilityEvidenceState::PassivelyObserved
                } else {
                    CapabilityEvidenceState::HypothesisOnly
                };
                assessments.push(CapabilityAssessment { descriptor, evidence_state });
            }
            Ok(assessments)
        })
    }

    pub(crate) fn record_world_relation(&self, relation: &WorldRelation) -> CoreResult<()> {
        if relation.id.is_nil() || relation.from_system == relation.to_system {
            return Err(CoreError::InvalidAction(
                "World-model relations require an identity and two distinct systems".into(),
            ));
        }
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let endpoint_exists: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM world_systems WHERE id=?1) AND EXISTS(SELECT 1 FROM world_systems WHERE id=?2)",
                params![relation.from_system.to_string(), relation.to_system.to_string()],
                |row| row.get(0),
            )?;
            let model_origin = serde_json::to_string(&EvidenceOrigin::Model)?;
            let evidence_record: Option<(String, String)> = tx.query_row(
                "SELECT system_id,payload_json FROM world_observations WHERE id=?1 AND expires_at>?2 AND origin<>?3 AND fingerprint=(SELECT fingerprint FROM world_systems WHERE id=system_id)",
                params![relation.evidence_id.to_string(), Utc::now().to_rfc3339(), model_origin],
                |row| Ok((row.get(0)?, row.get(1)?)),
            ).optional()?;
            let Some((evidence_system, payload)) = evidence_record else {
                return Err(Box::new(CoreError::VerificationFailed(
                    "World relation requires current, non-model observation evidence".into(),
                )));
            };
            let observation: ObservationEnvelope = serde_json::from_str(&payload)?;
            let target_id = relation.to_system.to_string();
            let names_target = observation.facts.iter().any(|fact| {
                fact.name == relation.kind.evidence_fact_name()
                    && fact.subject.is_none()
                    && matches!(&fact.value, FactValue::Identifier(value) if value == &target_id)
            });
            if !endpoint_exists
                || evidence_system != relation.from_system.to_string()
                || observation.system_id != relation.from_system
                || !names_target
            {
                return Err(Box::new(CoreError::VerificationFailed(
                    "World relation requires current endpoints and source evidence naming its target and relation kind".into(),
                )));
            }
            let existing: Option<(String, String, String, String)> = tx
                .query_row(
                    "SELECT from_system,relation_kind,to_system,evidence_id FROM world_relations WHERE id=?1",
                    [relation.id.to_string()],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional()?;
            if let Some((from, kind, to, evidence)) = existing {
                if from != relation.from_system.to_string()
                    || kind != serde_json::to_string(&relation.kind)?
                    || to != relation.to_system.to_string()
                    || evidence != relation.evidence_id.to_string()
                {
                    return Err(Box::new(CoreError::Protocol(
                        "World relation identity was replayed with different evidence".into(),
                    )));
                }
                tx.commit()?;
                return Ok(());
            }
            tx.execute(
                "INSERT INTO world_relations(id,from_system,relation_kind,to_system,evidence_id) VALUES(?1,?2,?3,?4,?5)",
                params![relation.id.to_string(), relation.from_system.to_string(), serde_json::to_string(&relation.kind)?, relation.to_system.to_string(), relation.evidence_id.to_string()],
            )?;
            crate::storage::write_audit(
                &tx,
                None,
                None,
                "world_relation_recorded",
                &serde_json::json!({"relation_id":relation.id,"from_system":relation.from_system,"kind":relation.kind,"to_system":relation.to_system,"evidence_id":relation.evidence_id}),
            )?;
            tx.commit()?;
            Ok(())
        })
    }

    pub(crate) fn world_relations(&self, system_id: Uuid) -> CoreResult<Vec<WorldRelation>> {
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let mut query = db.prepare(
                "SELECT r.id,r.from_system,r.relation_kind,r.to_system,r.evidence_id FROM world_relations r JOIN world_observations o ON o.id=r.evidence_id AND o.expires_at>?2 JOIN world_systems f ON f.id=r.from_system JOIN world_systems t ON t.id=r.to_system WHERE (r.from_system=?1 OR r.to_system=?1) AND o.fingerprint=(SELECT fingerprint FROM world_systems WHERE id=o.system_id) ORDER BY r.id LIMIT 512",
            )?;
            query.query_map(params![system_id.to_string(), Utc::now().to_rfc3339()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, String>(3)?, row.get::<_, String>(4)?))
            })?
            .map(|row| {
                let (id, from, kind, to, evidence) = row?;
                Ok(WorldRelation {
                    id: Uuid::parse_str(&id)?,
                    from_system: Uuid::parse_str(&from)?,
                    kind: serde_json::from_str(&kind)?,
                    to_system: Uuid::parse_str(&to)?,
                    evidence_id: Uuid::parse_str(&evidence)?,
                })
            })
            .collect()
        })
    }

    /// Forget one system and all of its derived observations, relations,
    /// candidates, sessions, leases, and causal transitions in one commit.
    pub(crate) fn forget_world_system(&self, system_id: Uuid) -> CoreResult<bool> {
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let exists: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM world_systems WHERE id=?1)",
                [system_id.to_string()],
                |row| row.get(0),
            )?;
            if !exists {
                return Ok(false);
            }
            let unsettled: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM world_probe_leases p JOIN world_learning_sessions s ON s.id=p.session_id WHERE s.system_id=?1 AND p.state='dispatched')",
                [system_id.to_string()],
                |row| row.get(0),
            )?;
            if unsettled {
                return Err(Box::new(CoreError::PermissionRequired(
                    "Cannot forget a system while a probe effect is unsettled; settle or review it first".into(),
                )));
            }
            tx.execute("DELETE FROM world_relations WHERE from_system=?1 OR to_system=?1 OR evidence_id IN (SELECT id FROM world_observations WHERE system_id=?1)", [system_id.to_string()])?;
            tx.execute("DELETE FROM world_transitions WHERE system_id=?1", [system_id.to_string()])?;
            tx.execute("DELETE FROM world_probe_leases WHERE session_id IN (SELECT id FROM world_learning_sessions WHERE system_id=?1)", [system_id.to_string()])?;
            tx.execute("DELETE FROM world_learning_sessions WHERE system_id=?1", [system_id.to_string()])?;
            tx.execute("DELETE FROM world_capabilities WHERE system_id=?1", [system_id.to_string()])?;
            tx.execute("DELETE FROM world_controllers WHERE system_id=?1", [system_id.to_string()])?;
            tx.execute("DELETE FROM world_observations WHERE system_id=?1", [system_id.to_string()])?;
            tx.execute("DELETE FROM world_systems WHERE id=?1", [system_id.to_string()])?;
            crate::storage::write_audit(&tx,None,None,"world_system_forgotten",&serde_json::json!({"system_id":system_id}))?;
            tx.commit()?;
            Ok(true)
        })
    }

    pub(crate) fn learning_sessions(&self, system_id: Uuid) -> CoreResult<Vec<LearningSession>> {
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let mut query = db.prepare(
                "SELECT id,fingerprint,worker_session,permitted_probes_json,state,probes_started,approved_at,expires_at FROM world_learning_sessions WHERE system_id=?1 ORDER BY approved_at DESC,id LIMIT 128",
            )?;
            let rows = query.query_map([system_id.to_string()], |row| {
                Ok((row.get::<_, String>(0)?,row.get::<_, String>(1)?,row.get::<_, String>(2)?,row.get::<_, String>(3)?,row.get::<_, String>(4)?,row.get::<_, u32>(5)?,row.get::<_, String>(6)?,row.get::<_, String>(7)?))
            })?;
            let mut sessions = Vec::new();
            for row in rows {
                let (id,fingerprint,worker_session,probes,state,probes_started,approved_at,expires_at) = row?;
                sessions.push(LearningSession {
                    id: Uuid::parse_str(&id)?,
                    system_id,
                    system_fingerprint: fingerprint,
                    worker_session,
                    permitted_probes: serde_json::from_str(&probes)?,
                    state: serde_json::from_str(&format!("\"{state}\""))?,
                    probes_started,
                    approved_at: DateTime::parse_from_rfc3339(&approved_at)?.to_utc(),
                    expires_at: DateTime::parse_from_rfc3339(&expires_at)?.to_utc(),
                });
            }
            Ok(sessions)
        })
    }

    pub(crate) fn learning_session(&self, session_id: Uuid) -> CoreResult<Option<LearningSession>> {
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let row: Option<LearningSessionStorageRow> = db
                .query_row(
                    "SELECT system_id,fingerprint,worker_session,permitted_probes_json,state,probes_started,approved_at,expires_at FROM world_learning_sessions WHERE id=?1",
                    [session_id.to_string()],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?)),
                )
                .optional()?;
            row.map(|(system_id,fingerprint,worker_session,probes,state,probes_started,approved_at,expires_at)| {
                Ok(LearningSession {
                    id: session_id,
                    system_id: Uuid::parse_str(&system_id)?,
                    system_fingerprint: fingerprint,
                    worker_session,
                    permitted_probes: serde_json::from_str(&probes)?,
                    state: serde_json::from_str(&format!("\"{state}\""))?,
                    probes_started,
                    approved_at: DateTime::parse_from_rfc3339(&approved_at)?.to_utc(),
                    expires_at: DateTime::parse_from_rfc3339(&expires_at)?.to_utc(),
                })
            }).transpose()
        })
    }

    pub(crate) fn dispatched_probe_lease(&self, lease_id: Uuid) -> CoreResult<Option<ProbeLease>> {
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let row: Option<DispatchedProbeLeaseStorageRow> = db
                .query_row(
                    "SELECT p.session_id,s.system_id,s.fingerprint,s.worker_session,p.control_id,p.probe_kind,p.expected_process_id,p.issued_at,p.expires_at FROM world_probe_leases p JOIN world_learning_sessions s ON s.id=p.session_id WHERE p.id=?1 AND p.state='dispatched' AND s.state IN ('active','interrupted_needs_review')",
                    [lease_id.to_string()],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?,row.get(7)?,row.get(8)?)),
                )
                .optional()?;
            row.map(|(session_id,system_id,fingerprint,worker_session,control_id,kind,expected_process_id,issued_at,expires_at)| -> Result<ProbeLease, Box<dyn std::error::Error + Send + Sync>> {
                let kind = match kind.as_str() {
                    "restore_slider_value" => ProbeKind::RestoreSliderValue,
                    "restore_toggle_state" => ProbeKind::RestoreToggleState,
                    _ => return Err(Box::new(CoreError::VerificationFailed("Stored probe kind is invalid".into()))),
                };
                Ok(ProbeLease {
                    id: lease_id,
                    session_id: Uuid::parse_str(&session_id)?,
                    system_id: Uuid::parse_str(&system_id)?,
                    system_fingerprint: fingerprint,
                    expected_process_id,
                    worker_session,
                    control_id,
                    kind,
                    issued_at: DateTime::parse_from_rfc3339(&issued_at)?.to_utc(),
                    expires_at: DateTime::parse_from_rfc3339(&expires_at)?.to_utc(),
                })
            }).transpose()
        })
    }

    /// Create a user-approved session for exact native controls. The caller
    /// must first verify the explicit native UI decision and adapter identity.
    pub(crate) fn approve_learning_session(
        &self,
        system_id: Uuid,
        system_fingerprint: &str,
        worker_session: &str,
        permitted_probes: BTreeMap<String, ProbeKind>,
        expires_at: DateTime<Utc>,
    ) -> CoreResult<LearningSession> {
        LearningSession::validate_request(
            system_fingerprint,
            worker_session,
            &permitted_probes,
            expires_at,
        )?;
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let system: Option<(String, String)> = tx
                .query_row(
                    "SELECT kind,fingerprint FROM world_systems WHERE id=?1",
                    [system_id.to_string()],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            let Some((kind, fingerprint)) = system else {
                return Err(Box::new(CoreError::PermissionRequired(
                    "Learning requires a current observed application".into(),
                )));
            };
            if kind != serde_json::to_string(&SystemKind::Application)?
                || fingerprint != system_fingerprint
            {
                return Err(Box::new(CoreError::PermissionRequired(
                    "Learning is limited to the exact current application identity".into(),
                )));
            }
            let now = Utc::now();
            let recent_observations = {
                let mut query = tx.prepare(
                    "SELECT payload_json FROM world_observations WHERE system_id=?1 AND fingerprint=?2 AND origin=?3 AND session_id IS NULL AND observed_at>?4 AND expires_at>?5 ORDER BY observed_at DESC LIMIT 16",
                )?;
                query
                    .query_map(
                        params![
                            system_id.to_string(),
                            system_fingerprint,
                            serde_json::to_string(&EvidenceOrigin::OperatingSystem)?,
                            (now - Duration::seconds(30)).to_rfc3339(),
                            now.to_rfc3339(),
                        ],
                        |row| row.get::<_, String>(0),
                    )?
                    .collect::<Result<Vec<_>, _>>()?
            };
            for (control_id, probe_kind) in &permitted_probes {
                let safe = recent_observations.iter().any(|payload| {
                    serde_json::from_str::<ObservationEnvelope>(payload)
                        .ok()
                        .is_some_and(|observation| {
                            observation.validate().is_ok()
                                && observation.system_id == system_id
                                && observation.system_fingerprint == system_fingerprint
                                && observation_supports_safe_probe(
                                    &observation,
                                    control_id,
                                    *probe_kind,
                                )
                        })
                });
                if !safe {
                    return Err(Box::new(CoreError::PermissionRequired(
                        "Learning can approve only a fresh OS-observed control with a bounded reversible effect classification".into(),
                    )));
                }
            }
            let session = LearningSession {
                id: Uuid::new_v4(),
                system_id,
                system_fingerprint: system_fingerprint.into(),
                worker_session: worker_session.into(),
                permitted_probes,
                state: LearningSessionState::Active,
                probes_started: 0,
                approved_at: Utc::now(),
                expires_at,
            };
            tx.execute(
                "INSERT INTO world_learning_sessions(id,system_id,fingerprint,worker_session,permitted_probes_json,state,probes_started,approved_at,expires_at) VALUES(?1,?2,?3,?4,?5,'active',0,?6,?7)",
                params![session.id.to_string(), system_id.to_string(), system_fingerprint, worker_session, serde_json::to_string(&session.permitted_probes)?, session.approved_at.to_rfc3339(), expires_at.to_rfc3339()],
            )?;
            crate::storage::write_audit(
                &tx,
                Some(session.id),
                None,
                "world_learning_session_approved",
                &serde_json::json!({"session_id":session.id,"system_id":system_id,"fingerprint":system_fingerprint,"worker_session":worker_session,"probe_count":session.permitted_probes.len(),"expires_at":expires_at}),
            )?;
            tx.commit()?;
            Ok(session)
        })
    }

    /// Consume a single short-lived, target-bound probe grant before dispatch.
    pub(crate) fn begin_probe(
        &self,
        session_id: Uuid,
        current_worker_session: &str,
        current_fingerprint: &str,
        control_id: &str,
        expected_process_id: u32,
        now: DateTime<Utc>,
    ) -> CoreResult<ProbeLease> {
        bounded_text(control_id, 256, "accessibility control identifier")?;
        bounded_text(current_worker_session, 128, "adapter session")?;
        if !digest(current_fingerprint) || expected_process_id == 0 {
            return Err(CoreError::PermissionRequired(
                "Learning probe target or process identity is invalid".into(),
            ));
        }
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let row: Option<(String, String, String, String, u32, String)> = tx
                .query_row(
                    "SELECT system_id,fingerprint,worker_session,permitted_probes_json,probes_started,expires_at FROM world_learning_sessions WHERE id=?1 AND state='active'",
                    [session_id.to_string()],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?)),
                )
                .optional()?;
            let Some((system_text,fingerprint,worker_session,permitted_json,started,expiry_text)) = row else {
                return Err(Box::new(CoreError::PermissionRequired(
                    "Learning session is unavailable or revoked".into(),
                )));
            };
            let system_id = Uuid::parse_str(&system_text)?;
            let expires_at = DateTime::parse_from_rfc3339(&expiry_text)?.to_utc();
            let allowed: BTreeMap<String, ProbeKind> = serde_json::from_str(&permitted_json)?;
            let system_is_current: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM world_systems WHERE id=?1 AND fingerprint=?2)",
                params![system_text, &fingerprint],
                |result| result.get(0),
            )?;
            if !system_is_current
                || fingerprint != current_fingerprint
                || worker_session != current_worker_session
                || expires_at <= now
                || started >= MAX_SESSION_PROBES
            {
                tx.execute(
                    "UPDATE world_learning_sessions SET state=?2 WHERE id=?1 AND state='active'",
                    params![session_id.to_string(), if expires_at <= now {"expired"} else {"interrupted_needs_review"}],
                )?;
                tx.commit()?;
                return Err(Box::new(CoreError::PermissionRequired(
                    "Learning approval expired or the target/worker session changed".into(),
                )));
            }
            let kind = allowed.get(control_id).copied().ok_or_else(|| {
                CoreError::PermissionRequired(
                    "This control was not included in the approved learning session".into(),
                )
            })?;
            let unsettled: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM world_probe_leases WHERE session_id=?1 AND state='dispatched')",
                [session_id.to_string()],
                |result| result.get(0),
            )?;
            if unsettled {
                return Err(Box::new(CoreError::Busy(
                    "A learning probe is still settling; this session is serialized".into(),
                )));
            }
            let latest_observation: Option<String> = tx
                .query_row(
                    "SELECT payload_json FROM world_observations WHERE system_id=?1 AND fingerprint=?2 AND origin=?3 AND session_id IS NULL AND observed_at>?4 AND expires_at>?5 ORDER BY observed_at DESC,id DESC LIMIT 1",
                    params![
                        system_id.to_string(),
                        &fingerprint,
                        serde_json::to_string(&EvidenceOrigin::OperatingSystem)?,
                        (now - Duration::seconds(MAX_EXPERIMENT_WINDOW_SECONDS)).to_rfc3339(),
                        now.to_rfc3339(),
                    ],
                    |row| row.get(0),
                )
                .optional()?;
            let target_is_fresh_and_safe = latest_observation
                .as_deref()
                .and_then(|payload| serde_json::from_str::<ObservationEnvelope>(payload).ok())
                .is_some_and(|observation| {
                    observation.observed_at <= now
                        && observation.validate().is_ok()
                        && observation_supports_safe_probe(&observation, control_id, kind)
                });
            if !target_is_fresh_and_safe {
                tx.execute(
                    "UPDATE world_learning_sessions SET state='interrupted_needs_review' WHERE id=?1 AND state='active'",
                    [session_id.to_string()],
                )?;
                tx.commit()?;
                return Err(Box::new(CoreError::PermissionRequired(
                    "The approved control changed or its OS observation is stale; scan and review the target again".into(),
                )));
            }
            let lease = ProbeLease {
                expires_at: std::cmp::min(
                    now + Duration::seconds(MAX_EXPERIMENT_WINDOW_SECONDS),
                    expires_at,
                ),
                id: Uuid::new_v4(),
                session_id,
                system_id,
                system_fingerprint: fingerprint,
                expected_process_id,
                worker_session,
                control_id: control_id.into(),
                kind,
                issued_at: now,
            };
            if lease.expires_at <= now {
                tx.execute(
                    "UPDATE world_learning_sessions SET state='expired' WHERE id=?1 AND state='active'",
                    [session_id.to_string()],
                )?;
                tx.commit()?;
                return Err(Box::new(CoreError::PermissionRequired(
                    "Learning approval expired before a new probe could start".into(),
                )));
            }
            let changed = tx.execute(
                "UPDATE world_learning_sessions SET probes_started=probes_started+1 WHERE id=?1 AND state='active' AND probes_started<20 AND expires_at>?2",
                params![session_id.to_string(), now.to_rfc3339()],
            )?;
            if changed != 1 {
                return Err(Box::new(CoreError::PermissionRequired(
                    "Learning probe could not consume its single-use session budget".into(),
                )));
            }
            tx.execute(
                "INSERT INTO world_probe_leases(id,session_id,control_id,probe_kind,expected_process_id,issued_at,expires_at,state) VALUES(?1,?2,?3,?4,?5,?6,?7,'dispatched')",
                params![lease.id.to_string(),session_id.to_string(),control_id,kind.name(),expected_process_id,lease.issued_at.to_rfc3339(),lease.expires_at.to_rfc3339()],
            )?;
            crate::storage::write_audit(
                &tx,
                Some(session_id),
                None,
                "world_probe_dispatched",
                &serde_json::json!({"probe_id":lease.id,"control_id":control_id,"kind":kind,"expires_at":lease.expires_at}),
            )?;
            tx.commit()?;
            Ok(lease)
        })
    }

    /// Accept an experimental fact only after the exact control changed and
    /// the captured state was independently restored to its original digest.
    pub(crate) fn complete_probe(
        &self,
        lease_id: Uuid,
        evidence: &ProbeEvidence,
        now: DateTime<Utc>,
    ) -> CoreResult<()> {
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let row: Option<ProbeCompletionStorageRow> = tx
                .query_row(
                    "SELECT p.session_id,s.system_id,s.worker_session,p.control_id,p.probe_kind,p.issued_at,p.expires_at,s.fingerprint FROM world_probe_leases p JOIN world_learning_sessions s ON s.id=p.session_id WHERE p.id=?1 AND p.state='dispatched' AND s.state IN ('active','interrupted_needs_review')",
                    [lease_id.to_string()],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?,row.get(7)?)),
                )
                .optional()?;
            let Some((session_text,system_text,worker_session,control_id,kind_text,issued_text,expiry_text,fingerprint)) = row else {
                return Err(Box::new(CoreError::PermissionRequired(
                    "Probe is unavailable, expired, or no longer active".into(),
                )));
            };
            let expiry = DateTime::parse_from_rfc3339(&expiry_text)?.to_utc();
            let issued = DateTime::parse_from_rfc3339(&issued_text)?.to_utc();
            let session_id = Uuid::parse_str(&session_text)?;
            let system_id = Uuid::parse_str(&system_text)?;
            if expiry <= now {
                tx.execute("UPDATE world_probe_leases SET state='failed' WHERE id=?1",[lease_id.to_string()])?;
                tx.execute("UPDATE world_learning_sessions SET state='interrupted_needs_review' WHERE id=?1",[&session_text])?;
                crate::storage::write_audit(&tx,Some(session_id),None,"world_probe_expired_needs_review",&serde_json::json!({"probe_id":lease_id,"control_id":control_id}))?;
                tx.commit()?;
                return Err(Box::new(CoreError::PermissionRequired(
                    "Probe timed out and needs restoration review".into(),
                )));
            }

            let kind: ProbeKind = match kind_text.as_str() {
                "restore_slider_value" => ProbeKind::RestoreSliderValue,
                "restore_toggle_state" => ProbeKind::RestoreToggleState,
                _ => {
                    tx.execute("UPDATE world_probe_leases SET state='failed' WHERE id=?1",[lease_id.to_string()])?;
                    tx.execute("UPDATE world_learning_sessions SET state='interrupted_needs_review' WHERE id=?1",[&session_text])?;
                    tx.commit()?;
                    return Err(Box::new(CoreError::VerificationFailed("Stored probe kind is invalid".into())));
                }
            };

            let mut values = Vec::with_capacity(3);
            let mut timestamps = Vec::with_capacity(3);
            for observation_id in [evidence.before_observation,evidence.changed_observation,evidence.restored_observation] {
                let payload: Option<String> = tx
                    .query_row(
                        "SELECT payload_json FROM world_observations WHERE id=?1 AND system_id=?2 AND session_id=?3 AND fingerprint=?4 AND expires_at>?5",
                        params![observation_id.to_string(), &system_text, &session_text, &fingerprint, now.to_rfc3339()],
                        |row| row.get(0),
                    )
                    .optional()?;
                let Some(payload) = payload else {
                    tx.execute("UPDATE world_probe_leases SET state='failed' WHERE id=?1",[lease_id.to_string()])?;
                    tx.execute("UPDATE world_learning_sessions SET state='interrupted_needs_review' WHERE id=?1",[&session_text])?;
                    crate::storage::write_audit(&tx,Some(session_id),None,"world_probe_evidence_missing_needs_review",&serde_json::json!({"probe_id":lease_id,"observation_id":observation_id}))?;
                    tx.commit()?;
                    return Err(Box::new(CoreError::VerificationFailed("Probe evidence is missing or belongs to another session or target".into())));
                };
                let observation: ObservationEnvelope = serde_json::from_str(&payload)?;
                let valid_identity = observation.id == observation_id
                    && observation.system_id == system_id
                    && observation.session_id == Some(session_id)
                    && observation.worker_session.as_deref() == Some(worker_session.as_str())
                    && observation.system_fingerprint == fingerprint
                    && observation.origin == EvidenceOrigin::OperatingSystem
                    && observation.observed_at >= issued
                    && observation.observed_at <= now
                    && observation.observed_at <= expiry;
                let value = observation.facts.iter().find(|fact| {
                    fact.name == "control.value" && fact.subject.as_deref() == Some(control_id.as_str())
                }).map(|fact| fact.value.clone());
                if !valid_identity || observation.validate().is_err() || value.is_none() {
                    tx.execute("UPDATE world_probe_leases SET state='failed' WHERE id=?1",[lease_id.to_string()])?;
                    tx.execute("UPDATE world_learning_sessions SET state='interrupted_needs_review' WHERE id=?1",[&session_text])?;
                    crate::storage::write_audit(&tx,Some(session_id),None,"world_probe_evidence_invalid_needs_review",&serde_json::json!({"probe_id":lease_id,"observation_id":observation_id}))?;
                    tx.commit()?;
                    return Err(Box::new(CoreError::VerificationFailed("Probe evidence did not come from the approved adapter, exact control, and live target".into())));
                }
                timestamps.push(observation.observed_at);
                values.push(value.expect("checked above"));
            }

            if !(timestamps[0] < timestamps[1] && timestamps[1] < timestamps[2]) {
                tx.execute("UPDATE world_probe_leases SET state='failed' WHERE id=?1",[lease_id.to_string()])?;
                tx.execute("UPDATE world_learning_sessions SET state='interrupted_needs_review' WHERE id=?1",[&session_text])?;
                tx.commit()?;
                return Err(Box::new(CoreError::VerificationFailed("Probe observations are not in strict before/change/restore order".into())));
            }

            let (before_hash, changed_hash, restored_hash) = match (kind, &values[0], &values[1], &values[2]) {
                (ProbeKind::RestoreToggleState, FactValue::Boolean(before), FactValue::Boolean(changed), FactValue::Boolean(restored)) => {
                    if before == changed || before != restored {
                        tx.execute("UPDATE world_probe_leases SET state='failed' WHERE id=?1",[lease_id.to_string()])?;
                        tx.execute("UPDATE world_learning_sessions SET state='restoration_failed' WHERE id=?1",[&session_text])?;
                        crate::storage::write_audit(&tx,Some(session_id),None,"world_probe_restoration_failed",&serde_json::json!({"probe_id":lease_id,"control_id":control_id,"observations":[evidence.before_observation,evidence.changed_observation,evidence.restored_observation]}))?;
                        tx.commit()?;
                        return Err(Box::new(CoreError::VerificationFailed("Toggle state was not restored to its original value".into())));
                    }
                    (value_digest(&values[0])?, value_digest(&values[1])?, value_digest(&values[2])?)
                }
                (ProbeKind::RestoreSliderValue, FactValue::Number(before), FactValue::Number(changed), FactValue::Number(restored)) => {
                    if before == changed || (changed - before).abs() > 1.0 || before != restored {
                        let restoration_failed = before != restored;
                        let terminal = if restoration_failed { "restoration_failed" } else { "interrupted_needs_review" };
                        tx.execute("UPDATE world_probe_leases SET state='failed' WHERE id=?1",[lease_id.to_string()])?;
                        tx.execute("UPDATE world_learning_sessions SET state=?2 WHERE id=?1",params![&session_text,terminal])?;
                        crate::storage::write_audit(&tx,Some(session_id),None,if restoration_failed {"world_probe_restoration_failed"} else {"world_probe_invalid_change"},&serde_json::json!({"probe_id":lease_id,"control_id":control_id,"observations":[evidence.before_observation,evidence.changed_observation,evidence.restored_observation]}))?;
                        tx.commit()?;
                        return Err(Box::new(CoreError::VerificationFailed("Slider probe exceeded its one-unit bound or failed exact restoration".into())));
                    }
                    (value_digest(&values[0])?, value_digest(&values[1])?, value_digest(&values[2])?)
                }
                _ => {
                    tx.execute("UPDATE world_probe_leases SET state='failed' WHERE id=?1",[lease_id.to_string()])?;
                    tx.execute("UPDATE world_learning_sessions SET state='interrupted_needs_review' WHERE id=?1",[&session_text])?;
                    tx.commit()?;
                    return Err(Box::new(CoreError::VerificationFailed("Probe value type does not match its approved probe kind".into())));
                }
            };

            let changed = tx.execute(
                "UPDATE world_probe_leases SET state='restored' WHERE id=?1 AND state='dispatched' AND expires_at>?2",
                params![lease_id.to_string(),now.to_rfc3339()],
            )?;
            if changed != 1 {
                return Err(Box::new(CoreError::PermissionRequired(
                    "Probe completion could not consume the active restoration receipt".into(),
                )));
            }
            tx.execute(
                "INSERT INTO world_transitions(probe_id,system_id,control_id,probe_kind,before_observation,changed_observation,restored_observation,before_value_sha256,changed_value_sha256,restored_value_sha256,confirmed_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                params![lease_id.to_string(),system_text,control_id,kind.name(),evidence.before_observation.to_string(),evidence.changed_observation.to_string(),evidence.restored_observation.to_string(),before_hash,changed_hash,restored_hash,now.to_rfc3339()],
            )?;
            let capability_id = passive_control_capability_id(&fingerprint, &control_id, kind);
            let descriptor_json: Option<String> = tx
                .query_row(
                    "SELECT descriptor_json FROM world_capabilities WHERE system_id=?1 AND capability_id=?2 AND fingerprint=?3",
                    params![&system_text, &capability_id, &fingerprint],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(serialized) = descriptor_json
                && let Ok(mut descriptor) = serde_json::from_str::<CapabilityDescriptor>(&serialized)
                && descriptor.interface_control_id.as_deref() == Some(control_id.as_str())
                && descriptor.interface_probe_kind == Some(kind)
                && descriptor.system_fingerprint == fingerprint
            {
                let mut evidence_ids = descriptor
                    .preconditions
                    .observed_state_fact_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                evidence_ids.extend([
                    evidence.before_observation,
                    evidence.changed_observation,
                    evidence.restored_observation,
                ]);
                descriptor.evidence_ids = evidence_ids.into_iter().collect();
                // Only a durable, independently checked before/change/restore
                // receipt promotes this passive hypothesis to Sage's sealed
                // exact-control executor. Passive discovery never does.
                descriptor.executor_id = Some("set_application_control".into());
                descriptor.preconditions.description = format!(
                    "Current signed application and visible enabled control; a bounded {} change was restored and verified",
                    kind.name()
                );
                descriptor.updated_at = now;
                if descriptor.validate().is_ok()
                    && descriptor.evidence_ids.len() <= MAX_CAPABILITY_EVIDENCE
                {
                    tx.execute(
                        "UPDATE world_capabilities SET descriptor_json=?3,updated_at=?4 WHERE system_id=?1 AND capability_id=?2 AND fingerprint=?5",
                        params![&system_text, &capability_id, serde_json::to_string(&descriptor)?, now.to_rfc3339(), &fingerprint],
                    )?;
                }
            }
            crate::storage::write_audit(
                &tx,
                Some(session_id),
                None,
                "world_probe_restored",
                &serde_json::json!({"probe_id":lease_id,"control_id":control_id,"before_sha256":before_hash,"changed_sha256":changed_hash,"restored_sha256":restored_hash,"observations":[evidence.before_observation,evidence.changed_observation,evidence.restored_observation]}),
            )?;
            tx.commit()?;
            Ok(())
        })
    }

    /// Closing/revoking a session does not erase outcomes. A dispatched but
    /// unsettled probe is marked for human restoration review.
    pub(crate) fn stop_learning_session(
        &self,
        session_id: Uuid,
        state: LearningSessionState,
    ) -> CoreResult<()> {
        if !matches!(
            state,
            LearningSessionState::Revoked
                | LearningSessionState::InterruptedNeedsReview
                | LearningSessionState::RestorationFailed
                | LearningSessionState::Expired
        ) {
            return Err(CoreError::InvalidAction(
                "Only a terminal learning-session state can be recorded".into(),
            ));
        }
        self.ensure_world_storage_unlocked()?;
        self.with_connection(|db| {
            let tx=db.transaction()?;
            let existing: bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM world_learning_sessions WHERE id=?1 AND state='active')",[session_id.to_string()],|row|row.get(0))?;
            if !existing { return Err(Box::new(CoreError::PermissionRequired("Learning session is not active".into()))); }
            let unsettled: bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM world_probe_leases WHERE session_id=?1 AND state='dispatched')",[session_id.to_string()],|row|row.get(0))?;
            let terminal=if unsettled { "interrupted_needs_review" } else { match state { LearningSessionState::Revoked=>"revoked",LearningSessionState::Expired=>"expired",LearningSessionState::RestorationFailed=>"restoration_failed",_=>"interrupted_needs_review" } };
            tx.execute("UPDATE world_learning_sessions SET state=?2 WHERE id=?1 AND state='active'",params![session_id.to_string(),terminal])?;
            crate::storage::write_audit(&tx,Some(session_id),None,"world_learning_session_stopped",&serde_json::json!({"session_id":session_id,"state":terminal,"unsettled_probe":unsettled}))?;
            tx.commit()?;
            Ok(())
        })
    }

    pub(crate) fn stop_learning_sessions_for_worker(&self, worker_session: &str) -> CoreResult<()> {
        bounded_text(worker_session, 128, "adapter session")?;
        if self.is_locked() {
            return Ok(());
        }
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let mut query = tx.prepare(
                "SELECT id FROM world_learning_sessions WHERE worker_session=?1 AND state='active' ORDER BY id LIMIT 128",
            )?;
            let session_ids: Vec<String> = query
                .query_map([worker_session], |row| row.get(0))?
                .collect::<Result<_, _>>()?;
            drop(query);
            for session_text in session_ids {
                let unsettled: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM world_probe_leases WHERE session_id=?1 AND state='dispatched')",
                    [&session_text],
                    |row| row.get(0),
                )?;
                let terminal = if unsettled { "interrupted_needs_review" } else { "revoked" };
                tx.execute(
                    "UPDATE world_learning_sessions SET state=?2 WHERE id=?1 AND state='active'",
                    params![&session_text, terminal],
                )?;
                crate::storage::write_audit(
                    &tx,
                    Some(Uuid::parse_str(&session_text)?),
                    None,
                    "world_learning_session_adapter_disconnected",
                    &serde_json::json!({"worker_session":worker_session,"state":terminal,"unsettled_probe":unsettled}),
                )?;
            }
            tx.commit()?;
            Ok(())
        })
    }
}

fn value_digest(value: &FactValue) -> CoreResult<String> {
    let canonical = serde_json::to_vec(value)?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::testing::MemorySecretStore;
    use tempfile::tempdir;

    #[test]
    fn process_binding_migration_upgrades_existing_probe_lease_tables() {
        let directory = tempdir().unwrap();
        let store = LocalStore::deferred(&directory.path().join("sage.db")).unwrap();
        store.unlock(&MemorySecretStore::default()).unwrap();
        store
            .with_connection(|db| {
                db.execute_batch(
                    "DROP INDEX IF EXISTS world_probe_pending;
                    DROP TABLE world_probe_leases;
                    DROP TABLE world_controllers;
                    DELETE FROM schema_migrations WHERE version=9;
                    CREATE TABLE world_probe_leases(
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        control_id TEXT NOT NULL,
                        probe_kind TEXT NOT NULL,
                        issued_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        state TEXT NOT NULL
                    );",
                )?;
                Ok(())
            })
            .unwrap();
        store.migrate_world_model().unwrap();
        let migrated = store
            .with_connection(|db| {
                let has_process_column = db.query_row(
                    "SELECT EXISTS(SELECT 1 FROM pragma_table_info('world_probe_leases') WHERE name='expected_process_id')",
                    [],
                    |row| row.get::<_, bool>(0),
                )?;
                let migration_recorded = db.query_row(
                    "SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version=8)",
                    [],
                    |row| row.get::<_, bool>(0),
                )?;
                let controller_table = db.query_row(
                    "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='world_controllers')",
                    [],
                    |row| row.get::<_, bool>(0),
                )?;
                let controller_migration = db.query_row(
                    "SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version=9)",
                    [],
                    |row| row.get::<_, bool>(0),
                )?;
                Ok((has_process_column, migration_recorded, controller_table, controller_migration))
            })
            .unwrap();
        assert_eq!(migrated, (true, true, true, true));
    }

    #[test]
    fn relations_require_non_model_current_evidence_and_follow_endpoint_identity() {
        let directory = tempdir().unwrap();
        let store = LocalStore::deferred(&directory.path().join("sage.db")).unwrap();
        store.unlock(&MemorySecretStore::default()).unwrap();
        let now = Utc::now();
        let system = |key: &str, fingerprint: &str| SystemDescriptor {
            id: Uuid::nil(),
            kind: SystemKind::Device,
            key: key.into(),
            label: key.into(),
            fingerprint: fingerprint.into(),
            revision: 0,
            updated_at: Utc::now(),
        };
        let from = store
            .observe_system(system("device:sender", &"a".repeat(64)))
            .unwrap();
        let to = store
            .observe_system(system("device:receiver", &"b".repeat(64)))
            .unwrap();
        let evidence = |origin| ObservationEnvelope {
            id: Uuid::new_v4(),
            system_id: from.id,
            session_id: None,
            worker_session: None,
            system_fingerprint: from.fingerprint.clone(),
            origin,
            privacy: Sensitivity::Private,
            observed_at: now,
            facts: vec![ObservedFact {
                name: "world.relation.connects_to".into(),
                subject: None,
                value: FactValue::Identifier(to.id.to_string()),
            }],
        };
        let direct_evidence = evidence(EvidenceOrigin::OperatingSystem);
        store.record_world_observation(&direct_evidence).unwrap();
        let relation = WorldRelation {
            id: Uuid::new_v4(),
            from_system: from.id,
            kind: WorldRelationKind::ConnectsTo,
            to_system: to.id,
            evidence_id: direct_evidence.id,
        };
        let unrelated = store
            .observe_system(system("device:other", &"d".repeat(64)))
            .unwrap();
        assert!(
            store
                .record_world_relation(&WorldRelation {
                    id: Uuid::new_v4(),
                    to_system: unrelated.id,
                    ..relation.clone()
                })
                .is_err()
        );
        store.record_world_relation(&relation).unwrap();
        assert_eq!(store.world_relations(from.id).unwrap().len(), 1);

        let model_evidence = evidence(EvidenceOrigin::Model);
        store.record_world_observation(&model_evidence).unwrap();
        assert!(
            store
                .record_world_relation(&WorldRelation {
                    id: Uuid::new_v4(),
                    evidence_id: model_evidence.id,
                    ..relation.clone()
                })
                .is_err()
        );

        store
            .observe_system(system("device:receiver", &"c".repeat(64)))
            .unwrap();
        assert!(store.world_relations(from.id).unwrap().is_empty());
    }

    fn slider_fixture() -> (tempfile::TempDir, LocalStore, LearningSession, ProbeLease) {
        let directory = tempdir().unwrap();
        let store = LocalStore::deferred(&directory.path().join("sage.db")).unwrap();
        store.unlock(&MemorySecretStore::default()).unwrap();
        let now = Utc::now();
        let fingerprint = "a".repeat(64);
        let system = store
            .observe_system(SystemDescriptor {
                id: Uuid::nil(),
                kind: SystemKind::Application,
                key: "com.example.Editor".into(),
                label: "Example Editor".into(),
                fingerprint: fingerprint.clone(),
                revision: 0,
                updated_at: now,
            })
            .unwrap();
        let control_id = "c".repeat(64);
        let observation = ObservationEnvelope {
            id: Uuid::new_v4(),
            system_id: system.id,
            session_id: None,
            worker_session: None,
            system_fingerprint: fingerprint.clone(),
            origin: EvidenceOrigin::OperatingSystem,
            privacy: Sensitivity::Private,
            observed_at: now,
            facts: vec![
                ObservedFact {
                    name: "control.enabled".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Boolean(true),
                },
                ObservedFact {
                    name: "control.label".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Text("Volume".into()),
                },
                ObservedFact {
                    name: "control.maximum".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Number(1.0),
                },
                ObservedFact {
                    name: "control.minimum".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Number(0.0),
                },
                ObservedFact {
                    name: "control.role".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Text("slider".into()),
                },
                ObservedFact {
                    name: "control.step".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Number(0.1),
                },
                ObservedFact {
                    name: "control.value".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Number(0.5),
                },
            ],
        };
        store.record_world_observation(&observation).unwrap();
        let candidate =
            passive_control_capability(&observation, &control_id, ProbeKind::RestoreSliderValue)
                .unwrap();
        store.record_capability_candidate(&candidate).unwrap();
        let session = store
            .approve_learning_session(
                system.id,
                &fingerprint,
                "native-session-fixture",
                BTreeMap::from([(control_id.clone(), ProbeKind::RestoreSliderValue)]),
                now + Duration::minutes(10),
            )
            .unwrap();
        let lease = store
            .begin_probe(
                session.id,
                "native-session-fixture",
                &fingerprint,
                &control_id,
                42,
                Utc::now(),
            )
            .unwrap();
        (directory, store, session, lease)
    }

    fn slider_readbacks(
        session: &LearningSession,
        lease: &ProbeLease,
        values: [f64; 3],
    ) -> [ObservationEnvelope; 3] {
        std::array::from_fn(|index| ObservationEnvelope {
            id: Uuid::new_v4(),
            system_id: lease.system_id,
            session_id: Some(session.id),
            worker_session: Some(session.worker_session.clone()),
            system_fingerprint: session.system_fingerprint.clone(),
            origin: EvidenceOrigin::OperatingSystem,
            privacy: Sensitivity::Private,
            observed_at: lease.issued_at + Duration::milliseconds(index as i64 + 1),
            facts: vec![ObservedFact {
                name: "control.value".into(),
                subject: Some(lease.control_id.clone()),
                value: FactValue::Number(values[index]),
            }],
        })
    }

    #[test]
    fn late_verified_restoration_settles_its_lease_without_reopening_stopped_session() {
        let (_directory, store, session, lease) = slider_fixture();
        store
            .stop_learning_sessions_for_worker("native-session-fixture")
            .unwrap();
        let observations = slider_readbacks(&session, &lease, [0.5, 0.6, 0.5]);
        let now = lease.issued_at + Duration::milliseconds(4);
        store
            .record_probe_observations(lease.id, &observations, now)
            .unwrap();
        store
            .complete_probe(
                lease.id,
                &ProbeEvidence {
                    before_observation: observations[0].id,
                    changed_observation: observations[1].id,
                    restored_observation: observations[2].id,
                },
                now,
            )
            .unwrap();
        assert_eq!(
            store.capability_assessments(session.system_id).unwrap()[0].evidence_state,
            CapabilityEvidenceState::ReversiblyExperimented
        );
        assert_eq!(
            store.capability_assessments(session.system_id).unwrap()[0]
                .descriptor
                .executor_id
                .as_deref(),
            Some("set_application_control")
        );
        let mut refreshed_scan = store
            .observations_for_system(session.system_id, None)
            .unwrap()
            .into_iter()
            .find(|observation| observation.session_id.is_none())
            .unwrap();
        refreshed_scan.id = Uuid::new_v4();
        refreshed_scan.observed_at = Utc::now();
        store.record_world_observation(&refreshed_scan).unwrap();
        let refreshed_candidate =
            passive_control_capability(&refreshed_scan, &lease.control_id, lease.kind).unwrap();
        store
            .record_capability_candidate(&refreshed_candidate)
            .unwrap();
        assert_eq!(
            store.capability_assessments(session.system_id).unwrap()[0].evidence_state,
            CapabilityEvidenceState::ReversiblyExperimented
        );
        let settled = store.learning_session(session.id).unwrap().unwrap();
        assert_eq!(settled.state, LearningSessionState::InterruptedNeedsReview);
        assert!(
            store
                .begin_probe(
                    session.id,
                    "native-session-fixture",
                    &session.system_fingerprint,
                    &lease.control_id,
                    lease.expected_process_id,
                    now,
                )
                .is_err()
        );
        assert!(
            store
                .complete_probe(
                    lease.id,
                    &ProbeEvidence {
                        before_observation: observations[0].id,
                        changed_observation: observations[1].id,
                        restored_observation: observations[2].id,
                    },
                    now,
                )
                .is_err()
        );
    }

    #[test]
    fn failed_restoration_records_a_terminal_review_state() {
        let (_directory, store, session, lease) = slider_fixture();
        let observations = slider_readbacks(&session, &lease, [0.5, 0.6, 0.7]);
        let now = lease.issued_at + Duration::milliseconds(4);
        store
            .record_probe_observations(lease.id, &observations, now)
            .unwrap();
        assert!(
            store
                .complete_probe(
                    lease.id,
                    &ProbeEvidence {
                        before_observation: observations[0].id,
                        changed_observation: observations[1].id,
                        restored_observation: observations[2].id,
                    },
                    now,
                )
                .is_err()
        );
        assert_eq!(
            store.learning_session(session.id).unwrap().unwrap().state,
            LearningSessionState::RestorationFailed
        );
        assert_eq!(
            store.capability_assessments(session.system_id).unwrap()[0].evidence_state,
            CapabilityEvidenceState::PassivelyObserved
        );
        assert_eq!(
            store.capability_assessments(session.system_id).unwrap()[0]
                .descriptor
                .executor_id,
            None
        );
    }
}
