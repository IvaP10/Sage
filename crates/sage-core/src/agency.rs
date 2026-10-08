//! Bounded, descriptive contracts for composed procedures, learned controllers,
//! peer leases, and transferable task checkpoints.
//!
//! These records never contain execution grants, credentials, native handles,
//! or arbitrary host code. A runtime must re-resolve targets and pass every
//! effect through Sage's ordinary policy, approval, broker, and verifier path.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::contracts::Effect;
use crate::error::{CoreError, CoreResult};
use crate::world_model::{
    CapabilityAssessment, CapabilityDescriptor, CapabilityEvidenceState, DataPort, FactValue,
    ObservationEnvelope, PortType,
};

const MAX_PROCEDURE_NODES: usize = 128;
const MAX_PROCEDURE_EDGES: usize = 512;
const MAX_STREAMS: usize = 32;
pub const MAX_PROCEDURE_STREAM_ITEM_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_PROCEDURE_STREAM_CHANNEL_BUFFER_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_PROCEDURE_STREAM_TOTAL_BUFFER_BYTES: u64 = 32 * 1024 * 1024;
const MAX_NESTING: usize = 4;
const MAX_REPEAT: u16 = 64;
const MAX_PROCEDURE_TEXT: usize = 1024;
const MAX_CHECKPOINT_ITEMS: usize = 256;
const MAX_PEER_LEASE_HOURS: i64 = 24;

fn bounded(value: &str, limit: usize, name: &str) -> CoreResult<()> {
    if value.trim().is_empty()
        || value.len() > limit
        || value.contains('\0')
        || value.chars().any(char::is_control)
    {
        return Err(CoreError::InvalidAction(format!(
            "{name} is empty, oversized, or contains control characters"
        )));
    }
    Ok(())
}

fn bounded_nonsecret(value: &str, limit: usize, name: &str) -> CoreResult<()> {
    bounded(value, limit, name)?;
    if crate::redaction::redact_for_persistence(value) != value {
        return Err(CoreError::PolicyDenied(format!(
            "Recognized credentials cannot be stored in {name}"
        )));
    }
    Ok(())
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ProcedureValue {
    Text(String),
    Boolean(bool),
    Number(f64),
    Identifier(String),
}

impl ProcedureValue {
    fn validate(&self) -> CoreResult<()> {
        match self {
            Self::Text(value) => bounded_nonsecret(value, 4096, "procedure text value"),
            Self::Identifier(value) => bounded_nonsecret(value, 512, "procedure identifier value"),
            Self::Number(value) if !value.is_finite() => Err(CoreError::InvalidAction(
                "Procedure numeric values must be finite".into(),
            )),
            Self::Number(_) | Self::Boolean(_) => Ok(()),
        }
    }

    fn port_type(&self) -> PortType {
        match self {
            Self::Text(_) => PortType::Text,
            Self::Boolean(_) => PortType::Boolean,
            Self::Number(_) => PortType::Number,
            Self::Identifier(_) => PortType::StructuredData,
        }
    }

    fn payload_bytes(&self) -> u64 {
        match self {
            Self::Text(value) | Self::Identifier(value) => value.len() as u64,
            Self::Boolean(_) => 1,
            Self::Number(_) => std::mem::size_of::<f64>() as u64,
        }
    }
}

/// A binary result is a reference to an immutable, owner-scoped private
/// artifact. The bytes are resolved through `LocalStore::read_artifact_for_task`
/// immediately before a downstream action is prepared.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureArtifactRef {
    pub artifact_id: Uuid,
    pub task_id: Uuid,
    pub sha256: String,
    pub size_bytes: u64,
}

impl ProcedureArtifactRef {
    fn validate_for(&self, task_id: Uuid, port: &DataPort) -> CoreResult<()> {
        if self.artifact_id.is_nil()
            || self.task_id.is_nil()
            || self.task_id != task_id
            || !valid_digest(&self.sha256)
            || self.size_bytes > 16 * 1024 * 1024
            || self.size_bytes > port.max_bytes
            || !matches!(
                port.value_type,
                PortType::Bytes | PortType::Image | PortType::Video | PortType::Audio
            )
        {
            return Err(CoreError::VerificationFailed(
                "Procedure artifact is not bound to this task and binary output port".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "output", rename_all = "snake_case")]
pub enum ProcedureOutput {
    Value(ProcedureValue),
    Artifact(ProcedureArtifactRef),
}

fn validate_procedure_output(
    output: &ProcedureOutput,
    task_id: Uuid,
    port: &DataPort,
) -> CoreResult<()> {
    match output {
        ProcedureOutput::Value(value) => {
            value.validate()?;
            if value.port_type() != port.value_type || value.payload_bytes() > port.max_bytes {
                return Err(CoreError::VerificationFailed(
                    "Verified procedure value violates its typed output port".into(),
                ));
            }
            Ok(())
        }
        ProcedureOutput::Artifact(reference) => reference.validate_for(task_id, port),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum ValueBinding {
    Literal {
        value: ProcedureValue,
        port: DataPort,
    },
    Result {
        producer: String,
        output: String,
    },
    /// A live bounded stream from another capability call. The referenced
    /// channel must bind this exact consumer input and producer output.
    Stream {
        channel_id: String,
    },
    Artifact {
        artifact_id: String,
        sha256: String,
        port: DataPort,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueReference {
    pub node_id: String,
    pub output: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "condition", rename_all = "snake_case")]
pub enum ProcedureCondition {
    BooleanEquals {
        reference: ValueReference,
        value: bool,
    },
    NumberAtLeast {
        reference: ValueReference,
        value: f64,
    },
    TextEquals {
        reference: ValueReference,
        value: String,
    },
    ResultExists {
        reference: ValueReference,
    },
}

impl ProcedureCondition {
    fn references(&self) -> Vec<&ValueReference> {
        match self {
            Self::BooleanEquals { reference, .. }
            | Self::NumberAtLeast { reference, .. }
            | Self::TextEquals { reference, .. }
            | Self::ResultExists { reference } => vec![reference],
        }
    }

    fn validate_value(&self) -> CoreResult<()> {
        match self {
            Self::NumberAtLeast { value, .. } if !value.is_finite() => Err(
                CoreError::InvalidAction("Procedure conditions must use finite numbers".into()),
            ),
            Self::TextEquals { value, .. } => {
                bounded_nonsecret(value, MAX_PROCEDURE_TEXT, "condition text")
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProcedureNodeKind {
    CapabilityCall {
        capability_id: String,
        system_id: Uuid,
        system_fingerprint: String,
        input_bindings: BTreeMap<String, ValueBinding>,
    },
    Branch {
        condition: ProcedureCondition,
        when_true: String,
        when_false: String,
    },
    Repeat {
        body: Box<ProcedureIr>,
        maximum_iterations: u16,
    },
    NestedSkill {
        skill_id: String,
        reviewed_digest: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureNode {
    pub id: String,
    pub depends_on: BTreeSet<String>,
    pub outputs: BTreeMap<String, DataPort>,
    pub kind: ProcedureNodeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamBackpressure {
    BlockProducer,
    CloseOnReceiverFailure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamChannel {
    pub id: String,
    pub producer_node: String,
    pub producer_output: String,
    pub consumer_node: String,
    pub consumer_input: String,
    pub capacity_items: u16,
    pub maximum_item_bytes: u64,
    pub backpressure: StreamBackpressure,
}

impl StreamChannel {
    /// Validate the declared in-memory queue bound without resolving ports.
    pub fn validate_bounds(&self) -> CoreResult<u64> {
        bounded(&self.id, 96, "stream channel identifier")?;
        bounded(&self.producer_node, 96, "stream producer node")?;
        bounded(&self.producer_output, 96, "stream output name")?;
        bounded(&self.consumer_node, 96, "stream consumer node")?;
        bounded(&self.consumer_input, 96, "stream input name")?;
        if self.capacity_items == 0
            || self.capacity_items > 256
            || self.maximum_item_bytes == 0
            || self.maximum_item_bytes > MAX_PROCEDURE_STREAM_ITEM_BYTES
        {
            return Err(CoreError::InvalidAction(
                "Stream channel capacity or item bound is outside the runtime limit".into(),
            ));
        }
        let maximum_buffer_bytes = u64::from(self.capacity_items)
            .checked_mul(self.maximum_item_bytes)
            .ok_or_else(|| CoreError::InvalidAction("Stream buffer size overflowed".into()))?;
        if maximum_buffer_bytes > MAX_PROCEDURE_STREAM_CHANNEL_BUFFER_BYTES {
            return Err(CoreError::InvalidAction(
                "Stream channel exceeds its aggregate buffer limit".into(),
            ));
        }
        Ok(maximum_buffer_bytes)
    }

    /// Bind the declared queue to compatible source and destination data ports.
    pub fn validate_ports(&self, producer: &DataPort, consumer: &DataPort) -> CoreResult<u64> {
        let maximum_buffer_bytes = self.validate_bounds()?;
        producer.validate()?;
        consumer.validate()?;
        if producer.name != self.producer_output
            || consumer.name != self.consumer_input
            || !ports_compatible(producer, consumer)
            || self.maximum_item_bytes > producer.max_bytes
            || self.maximum_item_bytes > consumer.max_bytes
        {
            return Err(CoreError::InvalidAction(
                "Stream channel does not fit its typed, sized, and privacy-scoped ports".into(),
            ));
        }
        Ok(maximum_buffer_bytes)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "condition", rename_all = "snake_case")]
pub enum CompletionCondition {
    AllNodesSucceeded,
    OutputAvailable { reference: ValueReference },
    Predicate { predicate: ProcedureCondition },
}

/// Versioned, closed-shape procedure data. Repetition is represented as a
/// nested procedure with a hard iteration limit; the outer dependency graph
/// must be acyclic. It is not an executable script or an authority grant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureIr {
    pub schema_version: u32,
    pub id: String,
    pub nodes: Vec<ProcedureNode>,
    pub streams: Vec<StreamChannel>,
    pub completion: Vec<CompletionCondition>,
}

impl ProcedureIr {
    pub fn validate(&self) -> CoreResult<Vec<String>> {
        self.validate_at_depth(0)
    }

    fn validate_at_depth(&self, depth: usize) -> CoreResult<Vec<String>> {
        bounded(&self.id, 128, "procedure identifier")?;
        if !matches!(self.schema_version, 1 | 2)
            || self.nodes.is_empty()
            || self.nodes.len() > MAX_PROCEDURE_NODES
            || self.streams.len() > MAX_STREAMS
            || self.completion.is_empty()
            || depth > MAX_NESTING
        {
            return Err(CoreError::InvalidAction(
                "Procedure exceeds its version, node, stream, nesting, or completion bounds".into(),
            ));
        }
        if self.schema_version == 1 && !self.streams.is_empty() {
            return Err(CoreError::InvalidAction(
                "Procedure streams require ProcedureIR schema version 2".into(),
            ));
        }
        let mut index = BTreeMap::new();
        for (position, node) in self.nodes.iter().enumerate() {
            bounded(&node.id, 96, "procedure node identifier")?;
            if index.insert(node.id.as_str(), position).is_some()
                || node.outputs.len() > 32
                || node.depends_on.len() > MAX_PROCEDURE_NODES
            {
                return Err(CoreError::InvalidAction(
                    "Procedure node identifiers or output bounds are invalid".into(),
                ));
            }
            for (name, output) in &node.outputs {
                if name != &output.name {
                    return Err(CoreError::InvalidAction(
                        "Procedure output map keys must match their typed port names".into(),
                    ));
                }
                output.validate()?;
            }
        }

        let mut total_stream_buffer_bytes = 0u64;

        let mut indegree = vec![0_usize; self.nodes.len()];
        let mut successors = vec![Vec::new(); self.nodes.len()];
        let mut edge_count = 0;
        for (position, node) in self.nodes.iter().enumerate() {
            for dependency in &node.depends_on {
                let Some(&source) = index.get(dependency.as_str()) else {
                    return Err(CoreError::InvalidAction(
                        "Procedure dependency references an unknown node".into(),
                    ));
                };
                if source == position {
                    return Err(CoreError::InvalidAction(
                        "Procedure node cannot depend on itself".into(),
                    ));
                }
                indegree[position] += 1;
                successors[source].push(position);
                edge_count += 1;
            }
        }
        if edge_count > MAX_PROCEDURE_EDGES {
            return Err(CoreError::InvalidAction(
                "Procedure dependency graph exceeds its edge bound".into(),
            ));
        }
        let mut ready = VecDeque::new();
        for (position, degree) in indegree.iter().enumerate() {
            if *degree == 0 {
                ready.push_back(position);
            }
        }
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(position) = ready.pop_front() {
            order.push(self.nodes[position].id.clone());
            for successor in &successors[position] {
                indegree[*successor] -= 1;
                if indegree[*successor] == 0 {
                    ready.push_back(*successor);
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(CoreError::InvalidAction(
                "Procedure dependencies must be acyclic; use a bounded repeat node".into(),
            ));
        }

        let ancestors = transitive_ancestors(&self.nodes);
        for node in &self.nodes {
            match &node.kind {
                ProcedureNodeKind::CapabilityCall {
                    capability_id,
                    system_id,
                    system_fingerprint,
                    input_bindings,
                } => {
                    bounded(capability_id, 128, "capability identifier")?;
                    if system_id.is_nil()
                        || !valid_digest(system_fingerprint)
                        || input_bindings.len() > 32
                    {
                        return Err(CoreError::InvalidAction(
                            "Capability call target or input count is invalid".into(),
                        ));
                    }
                    for (name, binding) in input_bindings {
                        bounded(name, 96, "capability input name")?;
                        match binding {
                            ValueBinding::Literal { value, port } => {
                                value.validate()?;
                                validate_seed_port(name, value.port_type(), port)?;
                                if value.payload_bytes() > port.max_bytes {
                                    return Err(CoreError::InvalidAction(
                                        "Procedure literal exceeds its declared data-port bound"
                                            .into(),
                                    ));
                                }
                            }
                            ValueBinding::Result { producer, output } => {
                                validate_reference(
                                    &index,
                                    &self.nodes,
                                    &ancestors,
                                    &node.id,
                                    producer,
                                    output,
                                )?;
                            }
                            ValueBinding::Stream { channel_id } => {
                                bounded(channel_id, 96, "stream channel identifier")?;
                                if self.schema_version < 2 {
                                    return Err(CoreError::InvalidAction(
                                        "Stream bindings require ProcedureIR schema version 2"
                                            .into(),
                                    ));
                                }
                            }
                            ValueBinding::Artifact {
                                artifact_id,
                                sha256,
                                port,
                            } => {
                                bounded(artifact_id, 128, "artifact identifier")?;
                                validate_seed_port(name, PortType::Bytes, port)?;
                                if !valid_digest(sha256) {
                                    return Err(CoreError::InvalidAction(
                                        "Artifact input requires a content digest".into(),
                                    ));
                                }
                            }
                        }
                    }
                }
                ProcedureNodeKind::Branch {
                    condition,
                    when_true,
                    when_false,
                } => {
                    condition.validate_value()?;
                    if !node.outputs.is_empty() {
                        return Err(CoreError::InvalidAction(
                            "Branch nodes cannot manufacture capability outputs".into(),
                        ));
                    }
                    for reference in condition.references() {
                        validate_reference(
                            &index,
                            &self.nodes,
                            &ancestors,
                            &node.id,
                            &reference.node_id,
                            &reference.output,
                        )?;
                    }
                    let reference = condition.references()[0];
                    let source_port =
                        &self.nodes[index[reference.node_id.as_str()]].outputs[&reference.output];
                    let expected_type = match condition {
                        ProcedureCondition::BooleanEquals { .. } => Some(PortType::Boolean),
                        ProcedureCondition::NumberAtLeast { .. } => Some(PortType::Number),
                        ProcedureCondition::TextEquals { .. } => Some(PortType::Text),
                        ProcedureCondition::ResultExists { .. } => None,
                    };
                    if expected_type.is_some_and(|value_type| source_port.value_type != value_type)
                    {
                        return Err(CoreError::InvalidAction(
                            "Procedure branch condition type does not match its referenced output"
                                .into(),
                        ));
                    }
                    for target in [when_true, when_false] {
                        let Some(&target_position) = index.get(target.as_str()) else {
                            return Err(CoreError::InvalidAction(
                                "Procedure branch points to an unknown node".into(),
                            ));
                        };
                        if !self.nodes[target_position].depends_on.contains(&node.id) {
                            return Err(CoreError::InvalidAction(
                                "Branch targets must depend on the branch node".into(),
                            ));
                        }
                    }
                }
                ProcedureNodeKind::Repeat {
                    body,
                    maximum_iterations,
                } => {
                    if *maximum_iterations == 0 || *maximum_iterations > MAX_REPEAT {
                        return Err(CoreError::InvalidAction(
                            "Procedure repetition must have a limit from 1 to 64".into(),
                        ));
                    }
                    body.validate_at_depth(depth + 1)?;
                }
                ProcedureNodeKind::NestedSkill {
                    skill_id,
                    reviewed_digest,
                } => {
                    bounded(skill_id, 128, "skill identifier")?;
                    if !valid_digest(reviewed_digest) {
                        return Err(CoreError::InvalidAction(
                            "Nested skills require a reviewed-content digest".into(),
                        ));
                    }
                }
            }
        }

        for stream in &self.streams {
            total_stream_buffer_bytes = total_stream_buffer_bytes
                .checked_add(stream.validate_bounds()?)
                .ok_or_else(|| CoreError::InvalidAction("Stream buffer size overflowed".into()))?;
            let Some(&producer) = index.get(stream.producer_node.as_str()) else {
                return Err(CoreError::InvalidAction(
                    "Stream producer is unknown".into(),
                ));
            };
            let Some(&consumer) = index.get(stream.consumer_node.as_str()) else {
                return Err(CoreError::InvalidAction(
                    "Stream consumer is unknown".into(),
                ));
            };
            if producer == consumer
                || !self.nodes[producer]
                    .outputs
                    .contains_key(&stream.producer_output)
            {
                return Err(CoreError::InvalidAction(
                    "Stream channel is unbounded or has invalid endpoints".into(),
                ));
            }
            if !matches!(
                &self.nodes[producer].kind,
                ProcedureNodeKind::CapabilityCall { .. }
            ) {
                return Err(CoreError::InvalidAction(
                    "Stream producers must be typed capability calls".into(),
                ));
            }
            let receiving_input_matches = matches!(
                &self.nodes[consumer].kind,
                ProcedureNodeKind::CapabilityCall { input_bindings, .. }
                    if matches!(input_bindings.get(&stream.consumer_input),
                        Some(ValueBinding::Stream { channel_id }) if channel_id == &stream.id)
            );
            if !receiving_input_matches {
                return Err(CoreError::InvalidAction(
                    "Stream consumer input must bind the exact declared channel".into(),
                ));
            }
            if ancestors[&stream.consumer_node].contains(&stream.producer_node)
                || ancestors[&stream.producer_node].contains(&stream.consumer_node)
            {
                return Err(CoreError::InvalidAction(
                    "Stream endpoints cannot depend on one another; both must be ready for concurrent dispatch".into(),
                ));
            }
        }
        if total_stream_buffer_bytes > MAX_PROCEDURE_STREAM_TOTAL_BUFFER_BYTES {
            return Err(CoreError::InvalidAction(
                "Procedure exceeds its aggregate stream memory budget".into(),
            ));
        }
        if self
            .streams
            .iter()
            .map(|stream| stream.id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != self.streams.len()
        {
            return Err(CoreError::InvalidAction(
                "Stream channel identifiers must be unique".into(),
            ));
        }
        let mut stream_producers = BTreeSet::new();
        let mut stream_consumers = BTreeSet::new();
        for stream in &self.streams {
            if !stream_producers.insert((
                stream.producer_node.as_str(),
                stream.producer_output.as_str(),
            )) || !stream_consumers.insert((
                stream.consumer_node.as_str(),
                stream.consumer_input.as_str(),
            )) {
                return Err(CoreError::InvalidAction(
                    "A procedure stream port can participate in only one channel".into(),
                ));
            }
        }
        for node in &self.nodes {
            let ProcedureNodeKind::CapabilityCall { input_bindings, .. } = &node.kind else {
                continue;
            };
            for (input_name, binding) in input_bindings {
                let ValueBinding::Stream { channel_id } = binding else {
                    continue;
                };
                if !self.streams.iter().any(|stream| {
                    stream.id == *channel_id
                        && stream.consumer_node == node.id
                        && stream.consumer_input == *input_name
                }) {
                    return Err(CoreError::InvalidAction(
                        "Stream input binding has no matching declared channel".into(),
                    ));
                }
            }
        }

        for condition in &self.completion {
            match condition {
                CompletionCondition::AllNodesSucceeded => {}
                CompletionCondition::OutputAvailable { reference } => validate_reference(
                    &index,
                    &self.nodes,
                    &ancestors,
                    "",
                    &reference.node_id,
                    &reference.output,
                )?,
                CompletionCondition::Predicate { predicate } => {
                    predicate.validate_value()?;
                    for reference in predicate.references() {
                        if !index.contains_key(reference.node_id.as_str())
                            || !self.nodes[index[reference.node_id.as_str()]]
                                .outputs
                                .contains_key(&reference.output)
                        {
                            return Err(CoreError::InvalidAction(
                                "Completion predicate references an unknown result".into(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(order)
    }

    /// Check port compatibility against currently stored descriptive
    /// capabilities. Success still does not authorize or dispatch an action.
    pub fn validate_against(&self, capabilities: &[CapabilityDescriptor]) -> CoreResult<()> {
        self.validate()?;
        let by_id = capabilities
            .iter()
            .map(|capability| (capability.id.as_str(), capability))
            .collect::<BTreeMap<_, _>>();
        let index = self
            .nodes
            .iter()
            .map(|node| (node.id.as_str(), node))
            .collect::<BTreeMap<_, _>>();
        for node in &self.nodes {
            let ProcedureNodeKind::CapabilityCall {
                capability_id,
                system_id,
                system_fingerprint,
                input_bindings,
            } = &node.kind
            else {
                continue;
            };
            let Some(capability) = by_id.get(capability_id.as_str()) else {
                return Err(CoreError::ExecutorUnavailable(
                    "Procedure references an undiscovered capability".into(),
                ));
            };
            capability.validate()?;
            if capability.system_id != *system_id
                || capability.system_fingerprint != *system_fingerprint
            {
                return Err(CoreError::VerificationFailed(
                    "Procedure target does not match its capability evidence".into(),
                ));
            }
            if node.outputs.len() != capability.output_ports.len()
                || capability.output_ports.iter().any(|port| {
                    node.outputs
                        .get(&port.name)
                        .is_none_or(|declared| declared != port)
                })
            {
                return Err(CoreError::InvalidAction(
                    "Procedure outputs must exactly match the capability's declared output ports"
                        .into(),
                ));
            }
            let ports = capability
                .input_ports
                .iter()
                .map(|port| (port.name.as_str(), port))
                .collect::<BTreeMap<_, _>>();
            if input_bindings.len() != ports.len() {
                return Err(CoreError::InvalidAction(
                    "Procedure inputs do not match the capability signature".into(),
                ));
            }
            for (name, binding) in input_bindings {
                let Some(port) = ports.get(name.as_str()) else {
                    return Err(CoreError::InvalidAction(
                        "Procedure supplies an unknown capability input".into(),
                    ));
                };
                let actual = match binding {
                    ValueBinding::Literal {
                        value,
                        port: supplied,
                    } => {
                        if !ports_compatible(supplied, port)
                            || value.port_type() != supplied.value_type
                        {
                            return Err(CoreError::InvalidAction(
                                "Procedure literal type or privacy exceeds the capability input"
                                    .into(),
                            ));
                        }
                        value.port_type()
                    }
                    ValueBinding::Result { producer, output } => {
                        let supplied = index
                            .get(producer.as_str())
                            .and_then(|producer| producer.outputs.get(output))
                            .ok_or_else(|| {
                                CoreError::InvalidAction(
                                    "Procedure result binding is unknown".into(),
                                )
                            })?;
                        if !ports_compatible(supplied, port) {
                            return Err(CoreError::InvalidAction(
                                "Procedure result exceeds the consumer data-port contract".into(),
                            ));
                        }
                        supplied.value_type
                    }
                    ValueBinding::Stream { channel_id } => {
                        let stream = self
                            .streams
                            .iter()
                            .find(|stream| {
                                stream.id == *channel_id
                                    && stream.consumer_node == node.id
                                    && stream.consumer_input == *name
                            })
                            .ok_or_else(|| {
                                CoreError::InvalidAction(
                                    "Procedure stream binding does not match this consumer input"
                                        .into(),
                                )
                            })?;
                        let source = index
                            .get(stream.producer_node.as_str())
                            .and_then(|producer| producer.outputs.get(&stream.producer_output))
                            .ok_or_else(|| {
                                CoreError::InvalidAction(
                                    "Procedure stream producer output is unknown".into(),
                                )
                            })?;
                        stream.validate_ports(source, port)?;
                        source.value_type
                    }
                    ValueBinding::Artifact { port: supplied, .. } => {
                        if !ports_compatible(supplied, port) {
                            return Err(CoreError::InvalidAction(
                                "Procedure artifact exceeds the capability input contract".into(),
                            ));
                        }
                        supplied.value_type
                    }
                };
                if actual != port.value_type {
                    return Err(CoreError::InvalidAction(
                        "Procedure input type does not match the capability contract".into(),
                    ));
                }
            }
        }
        for stream in &self.streams {
            let producer = index[stream.producer_node.as_str()];
            let consumer = index[stream.consumer_node.as_str()];
            let output = &producer.outputs[&stream.producer_output];
            let ProcedureNodeKind::CapabilityCall { capability_id, .. } = &consumer.kind else {
                return Err(CoreError::InvalidAction(
                    "Stream consumers must resolve to a typed capability input".into(),
                ));
            };
            let capability = by_id.get(capability_id.as_str()).ok_or_else(|| {
                CoreError::ExecutorUnavailable("Stream consumer capability is unavailable".into())
            })?;
            let input = capability
                .input_ports
                .iter()
                .find(|port| port.name == stream.consumer_input)
                .ok_or_else(|| CoreError::InvalidAction("Stream input is not declared".into()))?;
            stream.validate_ports(output, input)?;
        }
        Ok(())
    }
}

/// A deterministic, dependency-aware proposal schedule. Durations are local
/// observations or conservative defaults; they influence ordering only and
/// never establish permission or predict success.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureSchedule {
    pub waves: Vec<Vec<String>>,
    pub estimated_elapsed_micros: u64,
}

/// State for the bounded procedure planner. The runtime records dispatch only
/// after the ordinary broker has durably journaled it, and records successful
/// outputs only after fresh verification. This state is never an authority
/// grant; the broker repeats target, policy, permission, and verifier checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcedureNodeState {
    Running,
    Succeeded,
    Failed,
    Skipped,
    Uncertain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureRuntimeState {
    procedure_sha256: String,
    task_id: Uuid,
    nodes: BTreeMap<String, ProcedureNodeState>,
    branch_outcomes: BTreeMap<String, bool>,
    dispatch_receipts: BTreeMap<String, Uuid>,
    verification_evidence: BTreeMap<String, Uuid>,
    verified_outputs: BTreeMap<String, BTreeMap<String, ProcedureOutput>>,
}

/// Durable procedure progress contains descriptive receipts and verified
/// results only. It never stores a capability grant, credential, or native
/// execution handle.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureCheckpoint {
    pub task_id: Uuid,
    pub revision: u64,
    pub procedure: ProcedureIr,
    pub runtime: ProcedureRuntimeState,
    pub updated_at: DateTime<Utc>,
}

impl ProcedureRuntimeState {
    pub fn new(procedure: &ProcedureIr) -> CoreResult<Self> {
        Self::new_for_task(procedure, Uuid::new_v4())
    }

    /// Construct task-bound runtime state. Rehydrating a procedure for a new
    /// task must use that task's ID so artifact outputs from another task can
    /// never be rebound into this run.
    pub fn new_for_task(procedure: &ProcedureIr, task_id: Uuid) -> CoreResult<Self> {
        procedure.validate()?;
        if task_id.is_nil() {
            return Err(CoreError::InvalidAction(
                "Procedure runtime requires a non-empty task identity".into(),
            ));
        }
        let procedure_sha256 = procedure_digest(procedure)?;
        Ok(Self {
            procedure_sha256,
            task_id,
            nodes: BTreeMap::new(),
            branch_outcomes: BTreeMap::new(),
            dispatch_receipts: BTreeMap::new(),
            verification_evidence: BTreeMap::new(),
            verified_outputs: BTreeMap::new(),
        })
    }

    pub fn validate_checkpoint_for_task(
        &self,
        procedure: &ProcedureIr,
        task_id: Uuid,
    ) -> CoreResult<()> {
        if task_id.is_nil() || self.task_id != task_id {
            return Err(CoreError::VerificationFailed(
                "Procedure checkpoint belongs to a different task".into(),
            ));
        }
        self.validate_for(procedure)
    }

    /// Evidence IDs that must remain current before a completed procedure can
    /// be compiled into a reusable controller draft.
    pub fn verification_evidence_ids(&self) -> BTreeSet<Uuid> {
        self.verification_evidence.values().copied().collect()
    }

    /// Report whether the stored interpreter state satisfies the procedure's
    /// declared completion condition. This is a progress check only; it does
    /// not revalidate current capability authority or permit dispatch.
    pub fn completion_satisfied(&self, procedure: &ProcedureIr) -> CoreResult<bool> {
        self.validate_for(procedure)?;
        Ok(procedure
            .completion
            .iter()
            .all(|condition| match condition {
                CompletionCondition::AllNodesSucceeded => procedure.nodes.iter().all(|node| {
                    matches!(
                        self.nodes.get(&node.id),
                        Some(ProcedureNodeState::Succeeded | ProcedureNodeState::Skipped)
                    )
                }),
                CompletionCondition::OutputAvailable { reference } => {
                    self.nodes.get(&reference.node_id) == Some(&ProcedureNodeState::Succeeded)
                        && self
                            .verified_outputs
                            .get(&reference.node_id)
                            .is_some_and(|outputs| outputs.contains_key(&reference.output))
                }
                CompletionCondition::Predicate { predicate } => {
                    evaluate_procedure_condition(predicate, self) == Some(true)
                }
            }))
    }

    pub(crate) fn task_id(&self) -> Uuid {
        self.task_id
    }

    pub(crate) fn node_states(&self) -> &BTreeMap<String, ProcedureNodeState> {
        &self.nodes
    }

    pub(crate) fn dispatch_action_ids(&self) -> &BTreeMap<String, Uuid> {
        &self.dispatch_receipts
    }

    pub(crate) fn verification_evidence_by_node(&self) -> &BTreeMap<String, Uuid> {
        &self.verification_evidence
    }

    pub(crate) fn verified_outputs_by_node(
        &self,
    ) -> &BTreeMap<String, BTreeMap<String, ProcedureOutput>> {
        &self.verified_outputs
    }

    /// Record a non-stream node as in flight only after its durable dispatch
    /// receipt exists. Stream-connected nodes must use
    /// `record_dispatched_wave`. This does not replace the broker's per-action
    /// capability check.
    pub fn record_dispatched(
        &mut self,
        procedure: &ProcedureIr,
        capabilities: &[CapabilityAssessment],
        current_evidence_ids: &BTreeSet<Uuid>,
        node_id: &str,
        receipt_id: Uuid,
    ) -> CoreResult<()> {
        let dispatches = BTreeMap::from([(node_id.to_owned(), receipt_id)]);
        self.validate_dispatch_batch(procedure, capabilities, current_evidence_ids, &dispatches)?;
        self.nodes
            .insert(node_id.to_owned(), ProcedureNodeState::Running);
        self.dispatch_receipts
            .insert(node_id.to_owned(), receipt_id);
        Ok(())
    }

    /// Atomically bind durable dispatch receipts for one ready proposal wave.
    /// Every live-stream connected component must enter the running state as a
    /// whole so no endpoint is left waiting for a peer that was never started.
    pub fn record_dispatched_wave(
        &mut self,
        procedure: &ProcedureIr,
        capabilities: &[CapabilityAssessment],
        current_evidence_ids: &BTreeSet<Uuid>,
        dispatch_receipts: BTreeMap<String, Uuid>,
    ) -> CoreResult<()> {
        self.validate_dispatch_batch(
            procedure,
            capabilities,
            current_evidence_ids,
            &dispatch_receipts,
        )?;
        for (node_id, receipt_id) in dispatch_receipts {
            self.nodes
                .insert(node_id.clone(), ProcedureNodeState::Running);
            self.dispatch_receipts.insert(node_id, receipt_id);
        }
        Ok(())
    }

    fn validate_dispatch_batch(
        &self,
        procedure: &ProcedureIr,
        capabilities: &[CapabilityAssessment],
        current_evidence_ids: &BTreeSet<Uuid>,
        dispatch_receipts: &BTreeMap<String, Uuid>,
    ) -> CoreResult<()> {
        validate_procedure_assessments(procedure, capabilities, current_evidence_ids)?;
        self.validate_for(procedure)?;
        if dispatch_receipts.is_empty() || dispatch_receipts.len() > 16 {
            return Err(CoreError::InvalidAction(
                "Procedure dispatch wave must contain between 1 and 16 calls".into(),
            ));
        }
        let mut unique_receipts = self
            .dispatch_receipts
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        if dispatch_receipts
            .values()
            .any(|receipt| receipt.is_nil() || !unique_receipts.insert(*receipt))
        {
            return Err(CoreError::InvalidAction(
                "Procedure dispatch receipt is empty or already bound".into(),
            ));
        }
        let selected = dispatch_receipts.keys().cloned().collect::<BTreeSet<_>>();
        for cohort in stream_cohorts(procedure) {
            if cohort.iter().any(|node_id| selected.contains(node_id))
                && !cohort.is_subset(&selected)
            {
                return Err(CoreError::PermissionRequired(
                    "All endpoints in a live-stream component must be dispatched in the same wave"
                        .into(),
                ));
            }
        }
        let descriptors = capabilities
            .iter()
            .map(|assessment| (assessment.descriptor.id.as_str(), &assessment.descriptor))
            .collect::<BTreeMap<_, _>>();
        let descriptor_list = capabilities
            .iter()
            .map(|assessment| assessment.descriptor.clone())
            .collect::<Vec<_>>();
        let nodes = procedure
            .nodes
            .iter()
            .map(|node| (node.id.as_str(), node))
            .collect::<BTreeMap<_, _>>();
        let mut selected_capabilities = Vec::new();
        for node_id in dispatch_receipts.keys() {
            let node = nodes
                .get(node_id.as_str())
                .ok_or_else(|| CoreError::InvalidAction("Unknown procedure node".into()))?;
            let ProcedureNodeKind::CapabilityCall { capability_id, .. } = &node.kind else {
                return Err(CoreError::InvalidAction(
                    "Only capability calls can be dispatched".into(),
                ));
            };
            if self.nodes.contains_key(node_id)
                || !node.depends_on.iter().all(|dependency| {
                    matches!(
                        self.nodes.get(dependency),
                        Some(ProcedureNodeState::Succeeded | ProcedureNodeState::Skipped)
                    )
                })
                || resolve_call_inputs(node, procedure, self, &descriptor_list).is_none()
            {
                return Err(CoreError::PermissionRequired(
                    "Procedure call is not ready from completed dependencies and typed inputs"
                        .into(),
                ));
            }
            selected_capabilities.push(descriptors[capability_id.as_str()]);
        }
        let running_capabilities = procedure
            .nodes
            .iter()
            .filter_map(|node| {
                (self.nodes.get(&node.id) == Some(&ProcedureNodeState::Running))
                    .then(|| match &node.kind {
                        ProcedureNodeKind::CapabilityCall { capability_id, .. } => {
                            descriptors.get(capability_id.as_str()).copied()
                        }
                        _ => None,
                    })
                    .flatten()
            })
            .collect::<Vec<_>>();
        for (position, capability) in selected_capabilities.iter().enumerate() {
            if running_capabilities
                .iter()
                .any(|active| capabilities_conflict(capability, active))
                || selected_capabilities[..position]
                    .iter()
                    .any(|active| capabilities_conflict(capability, active))
            {
                return Err(CoreError::PermissionRequired(
                    "Procedure dispatch wave contains conflicting active effects".into(),
                ));
            }
        }
        Ok(())
    }

    /// Store typed outputs only after the verifier has accepted fresh evidence
    /// for the dispatched call. Missing optional outputs remain absent, which
    /// allows an explicit `ResultExists` branch to handle that verified state.
    pub fn record_verified_success(
        &mut self,
        procedure: &ProcedureIr,
        node_id: &str,
        outputs: BTreeMap<String, ProcedureValue>,
        evidence_id: Uuid,
    ) -> CoreResult<()> {
        self.record_verified_outputs(
            procedure,
            node_id,
            outputs
                .into_iter()
                .map(|(name, value)| (name, ProcedureOutput::Value(value)))
                .collect(),
            evidence_id,
        )
    }

    /// Store scalar and artifact references only after fresh independent
    /// verification. Artifact references remain bound to this runtime's task.
    pub fn record_verified_outputs(
        &mut self,
        procedure: &ProcedureIr,
        node_id: &str,
        outputs: BTreeMap<String, ProcedureOutput>,
        evidence_id: Uuid,
    ) -> CoreResult<()> {
        self.validate_for(procedure)?;
        if evidence_id.is_nil()
            || self
                .verification_evidence
                .values()
                .any(|prior| *prior == evidence_id)
            || !matches!(
                self.nodes.get(node_id),
                Some(ProcedureNodeState::Running | ProcedureNodeState::Uncertain)
            )
        {
            return Err(CoreError::VerificationFailed(
                "Verified procedure result has no unique evidence or unresolved dispatched call"
                    .into(),
            ));
        }
        let node = procedure
            .nodes
            .iter()
            .find(|node| node.id == node_id)
            .ok_or_else(|| CoreError::InvalidAction("Unknown procedure node".into()))?;
        if !matches!(node.kind, ProcedureNodeKind::CapabilityCall { .. })
            || outputs.len() > node.outputs.len()
        {
            return Err(CoreError::InvalidAction(
                "Only declared capability outputs can be recorded".into(),
            ));
        }
        for (name, value) in &outputs {
            let port = node.outputs.get(name).ok_or_else(|| {
                CoreError::InvalidAction("Verified result names an undeclared output".into())
            })?;
            validate_procedure_output(value, self.task_id, port)?;
        }
        self.nodes
            .insert(node_id.to_owned(), ProcedureNodeState::Succeeded);
        self.verification_evidence
            .insert(node_id.to_owned(), evidence_id);
        self.verified_outputs.insert(node_id.to_owned(), outputs);
        Ok(())
    }

    /// Record a settled failure or an unresolved outcome. Uncertain effects
    /// remain blockers for dependent nodes until separately reconciled.
    pub fn record_failure(
        &mut self,
        procedure: &ProcedureIr,
        node_id: &str,
        uncertain: bool,
    ) -> CoreResult<()> {
        self.validate_for(procedure)?;
        if self.nodes.get(node_id) != Some(&ProcedureNodeState::Running) {
            return Err(CoreError::InvalidAction(
                "Only a dispatched procedure call can fail".into(),
            ));
        }
        self.nodes.insert(
            node_id.to_owned(),
            if uncertain {
                ProcedureNodeState::Uncertain
            } else {
                ProcedureNodeState::Failed
            },
        );
        Ok(())
    }

    fn validate_for(&self, procedure: &ProcedureIr) -> CoreResult<()> {
        procedure.validate()?;
        if self.procedure_sha256 != procedure_digest(procedure)? {
            return Err(CoreError::VerificationFailed(
                "Procedure runtime state belongs to different procedure content".into(),
            ));
        }
        let nodes = procedure
            .nodes
            .iter()
            .map(|node| (node.id.as_str(), node))
            .collect::<BTreeMap<_, _>>();
        if self.nodes.keys().any(|id| !nodes.contains_key(id.as_str()))
            || self.branch_outcomes.keys().any(|id| {
                !nodes.get(id.as_str()).is_some_and(|node| {
                    matches!(node.kind, ProcedureNodeKind::Branch { .. })
                        && self.nodes.get(id) == Some(&ProcedureNodeState::Succeeded)
                })
            })
            || self.dispatch_receipts.keys().any(|id| {
                !nodes.get(id.as_str()).is_some_and(|node| {
                    matches!(node.kind, ProcedureNodeKind::CapabilityCall { .. })
                        && self.nodes.contains_key(id)
                        && self.nodes.get(id) != Some(&ProcedureNodeState::Skipped)
                })
            })
            || self.verification_evidence.keys().any(|id| {
                self.nodes.get(id) != Some(&ProcedureNodeState::Succeeded)
                    || !self.dispatch_receipts.contains_key(id)
            })
            || self.verified_outputs.keys().any(|id| {
                self.nodes.get(id) != Some(&ProcedureNodeState::Succeeded)
                    || !self.verification_evidence.contains_key(id)
            })
        {
            return Err(CoreError::VerificationFailed(
                "Procedure runtime state contains an unbound node or receipt".into(),
            ));
        }
        let receipt_ids = self
            .dispatch_receipts
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        let evidence_ids = self
            .verification_evidence
            .values()
            .copied()
            .collect::<BTreeSet<_>>();
        if self.dispatch_receipts.values().any(Uuid::is_nil)
            || receipt_ids.len() != self.dispatch_receipts.len()
            || self.verification_evidence.values().any(Uuid::is_nil)
            || evidence_ids.len() != self.verification_evidence.len()
        {
            return Err(CoreError::VerificationFailed(
                "Procedure runtime receipts and verification evidence must be unique and non-empty"
                    .into(),
            ));
        }
        for (id, status) in &self.nodes {
            let node = nodes[id.as_str()];
            match &node.kind {
                ProcedureNodeKind::CapabilityCall { .. } => {
                    if matches!(
                        status,
                        ProcedureNodeState::Running
                            | ProcedureNodeState::Succeeded
                            | ProcedureNodeState::Failed
                            | ProcedureNodeState::Uncertain
                    ) && !self.dispatch_receipts.contains_key(id)
                        || *status == ProcedureNodeState::Succeeded
                            && !self.verification_evidence.contains_key(id)
                        || self.branch_outcomes.contains_key(id)
                    {
                        return Err(CoreError::VerificationFailed(
                            "Capability call state is missing a dispatch or verification receipt"
                                .into(),
                        ));
                    }
                }
                ProcedureNodeKind::Branch { .. } => {
                    if *status != ProcedureNodeState::Succeeded
                        && *status != ProcedureNodeState::Skipped
                        || *status == ProcedureNodeState::Succeeded
                            && !self.branch_outcomes.contains_key(id)
                        || self.dispatch_receipts.contains_key(id)
                        || self.verification_evidence.contains_key(id)
                        || self.verified_outputs.contains_key(id)
                    {
                        return Err(CoreError::VerificationFailed(
                            "Branch state must be a resolved control result, never a dispatched effect".into(),
                        ));
                    }
                }
                ProcedureNodeKind::Repeat { .. } | ProcedureNodeKind::NestedSkill { .. } => {
                    return Err(CoreError::ExecutorUnavailable(
                        "Procedure runtime currently supports capability calls and branches only"
                            .into(),
                    ));
                }
            }
        }
        for id in self.nodes.keys() {
            if self.nodes[id] == ProcedureNodeState::Succeeded
                && matches!(
                    nodes[id.as_str()].kind,
                    ProcedureNodeKind::CapabilityCall { .. }
                )
                && !self.verified_outputs.contains_key(id)
            {
                return Err(CoreError::VerificationFailed(
                    "Succeeded capability call has no verified output record".into(),
                ));
            }
        }
        for cohort in stream_cohorts(procedure) {
            let dispatched_count = cohort
                .iter()
                .filter(|id| self.dispatch_receipts.contains_key(*id))
                .count();
            if dispatched_count != 0 && dispatched_count != cohort.len() {
                return Err(CoreError::VerificationFailed(
                    "Live-stream endpoints must retain one atomic dispatch-wave receipt set".into(),
                ));
            }
        }
        for (id, outputs) in &self.verified_outputs {
            let node = nodes[id.as_str()];
            if outputs.len() > node.outputs.len() {
                return Err(CoreError::VerificationFailed(
                    "Persisted procedure outputs exceed their declared ports".into(),
                ));
            }
            for (name, value) in outputs {
                let port = node.outputs.get(name).ok_or_else(|| {
                    CoreError::VerificationFailed(
                        "Persisted procedure output is not declared".into(),
                    )
                })?;
                validate_procedure_output(value, self.task_id, port)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedProcedureInput {
    Value {
        value: ProcedureValue,
        source_port: DataPort,
        target_port: DataPort,
    },
    Artifact {
        artifact_id: String,
        #[serde(default)]
        owner_task_id: Option<Uuid>,
        sha256: String,
        #[serde(default)]
        size_bytes: Option<u64>,
        source_port: DataPort,
        target_port: DataPort,
    },
    Stream {
        channel_id: String,
        source_port: DataPort,
        target_port: DataPort,
    },
}

/// One broker-bound proposal from the procedure interpreter. Target identity,
/// input values, and effects are descriptive and must be re-resolved and
/// authorized for every dispatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureCallProposal {
    pub node_id: String,
    pub capability_id: String,
    pub system_id: Uuid,
    pub system_fingerprint: String,
    pub inputs: BTreeMap<String, ResolvedProcedureInput>,
    pub effects: BTreeSet<Effect>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcedureBlockReason {
    DependenciesPending,
    DependencyFailed,
    DependencyUncertain,
    MissingVerifiedInput,
    MissingVerifiedCondition,
    StreamingCohortIncomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureBlock {
    pub node_id: String,
    pub reason: ProcedureBlockReason,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureAdvance {
    pub proposal_wave: Vec<ProcedureCallProposal>,
    pub resolved_branches: BTreeMap<String, bool>,
    pub newly_skipped_nodes: Vec<String>,
    pub blocked_nodes: Vec<ProcedureBlock>,
    pub completion_satisfied: bool,
}

/// Advance the first-party control-flow subset: typed capability calls,
/// result-dependent branches and live-stream input cohorts. It consumes
/// receipts/output supplied by the core runtime, returns one bounded wave of
/// proposals, and never dispatches. Repeats and nested skills remain
/// unavailable until their runtime semantics and persistence are implemented.
pub fn advance_procedure(
    procedure: &ProcedureIr,
    capabilities: &[CapabilityAssessment],
    current_evidence_ids: &BTreeSet<Uuid>,
    state: &mut ProcedureRuntimeState,
    observed_node_micros: &BTreeMap<String, u64>,
    maximum_parallelism: usize,
) -> CoreResult<ProcedureAdvance> {
    if maximum_parallelism == 0
        || maximum_parallelism > 16
        || observed_node_micros.len() > MAX_PROCEDURE_NODES
        || observed_node_micros
            .iter()
            .any(|(id, duration)| id.is_empty() || id.len() > 96 || *duration > 600_000_000)
    {
        return Err(CoreError::InvalidAction(
            "Procedure runtime limits or timing observations are invalid".into(),
        ));
    }
    validate_procedure_assessments(procedure, capabilities, current_evidence_ids)?;
    state.validate_for(procedure)?;
    let descriptors = capabilities
        .iter()
        .map(|assessment| assessment.descriptor.clone())
        .collect::<Vec<_>>();
    let capability_by_id = capabilities
        .iter()
        .map(|capability| (capability.descriptor.id.as_str(), &capability.descriptor))
        .collect::<BTreeMap<_, _>>();
    if capability_by_id.len() != capabilities.len() {
        return Err(CoreError::InvalidAction(
            "Procedure runtime requires unique capability identities".into(),
        ));
    }
    let node_by_id = procedure
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    let topological = procedure.validate()?;
    let successors = procedure_successors(procedure);
    let mut resolved_branches = BTreeMap::new();
    let mut newly_skipped = BTreeSet::new();
    let mut blocked_nodes = Vec::new();

    for node_id in &topological {
        let node = node_by_id[node_id.as_str()];
        let ProcedureNodeKind::Branch {
            condition,
            when_true,
            when_false,
        } = &node.kind
        else {
            continue;
        };
        if state.nodes.contains_key(&node.id) {
            continue;
        }
        if let Some(reason) = dependency_block_reason(node, state) {
            blocked_nodes.push(ProcedureBlock {
                node_id: node.id.clone(),
                reason,
            });
            continue;
        }
        let Some(take_true) = evaluate_procedure_condition(condition, state) else {
            blocked_nodes.push(ProcedureBlock {
                node_id: node.id.clone(),
                reason: ProcedureBlockReason::MissingVerifiedCondition,
            });
            continue;
        };
        state
            .nodes
            .insert(node.id.clone(), ProcedureNodeState::Succeeded);
        state.branch_outcomes.insert(node.id.clone(), take_true);
        resolved_branches.insert(node.id.clone(), take_true);

        let selected = if take_true { when_true } else { when_false };
        let inactive = if take_true { when_false } else { when_true };
        let selected_reachable = reachable_from(selected, &successors);
        let inactive_reachable = reachable_from(inactive, &successors);
        for skipped in inactive_reachable.difference(&selected_reachable) {
            if let Some(existing) = state.nodes.get(skipped) {
                if *existing != ProcedureNodeState::Skipped {
                    return Err(CoreError::VerificationFailed(
                        "An inactive branch already has dispatched or settled work".into(),
                    ));
                }
            } else {
                state
                    .nodes
                    .insert(skipped.clone(), ProcedureNodeState::Skipped);
                newly_skipped.insert(skipped.clone());
            }
        }
    }
    state.validate_for(procedure)?;

    for node in &procedure.nodes {
        if !matches!(node.kind, ProcedureNodeKind::CapabilityCall { .. })
            || state.nodes.contains_key(&node.id)
        {
            continue;
        }
        if let Some(reason) = dependency_block_reason(node, state) {
            blocked_nodes.push(ProcedureBlock {
                node_id: node.id.clone(),
                reason,
            });
        } else if resolve_call_inputs(node, procedure, state, &descriptors).is_none() {
            blocked_nodes.push(ProcedureBlock {
                node_id: node.id.clone(),
                reason: ProcedureBlockReason::MissingVerifiedInput,
            });
        }
    }

    let mut critical_path = BTreeMap::<String, u64>::new();
    for id in topological.iter().rev() {
        if state.nodes.get(id) == Some(&ProcedureNodeState::Skipped) {
            critical_path.insert(id.clone(), 0);
            continue;
        }
        let tail = successors
            .get(id)
            .into_iter()
            .flatten()
            .filter_map(|successor| critical_path.get(successor))
            .copied()
            .max()
            .unwrap_or(0);
        let own = if state.nodes.contains_key(id) {
            0
        } else if matches!(
            node_by_id[id.as_str()].kind,
            ProcedureNodeKind::CapabilityCall { .. }
        ) {
            observed_node_micros
                .get(id)
                .copied()
                .unwrap_or(1_000_000)
                .max(1)
        } else {
            0
        };
        critical_path.insert(id.clone(), own.saturating_add(tail));
    }

    let candidates = procedure
        .nodes
        .iter()
        .filter_map(|node| {
            if state.nodes.contains_key(&node.id)
                || !matches!(node.kind, ProcedureNodeKind::CapabilityCall { .. })
                || dependency_block_reason(node, state).is_some()
            {
                return None;
            }
            let inputs = resolve_call_inputs(node, procedure, state, &descriptors)?;
            let ProcedureNodeKind::CapabilityCall {
                capability_id,
                system_id,
                system_fingerprint,
                ..
            } = &node.kind
            else {
                unreachable!()
            };
            Some((
                critical_path.get(&node.id).copied().unwrap_or(0),
                ProcedureCallProposal {
                    node_id: node.id.clone(),
                    capability_id: capability_id.clone(),
                    system_id: *system_id,
                    system_fingerprint: system_fingerprint.clone(),
                    inputs,
                    effects: capability_by_id[capability_id.as_str()].effects.clone(),
                },
            ))
        })
        .collect::<Vec<_>>();
    let candidate_by_id = candidates
        .into_iter()
        .map(|(cost, proposal)| (proposal.node_id.clone(), (cost, proposal)))
        .collect::<BTreeMap<_, _>>();

    let running_capabilities = procedure
        .nodes
        .iter()
        .filter_map(|node| {
            (state.nodes.get(&node.id) == Some(&ProcedureNodeState::Running))
                .then(|| match &node.kind {
                    ProcedureNodeKind::CapabilityCall { capability_id, .. } => {
                        capability_by_id.get(capability_id.as_str()).copied()
                    }
                    _ => None,
                })
                .flatten()
        })
        .collect::<Vec<_>>();
    let mut selected_capabilities = Vec::<&CapabilityDescriptor>::new();
    let mut proposal_wave = Vec::new();
    let mut cohorts_by_node = BTreeMap::<String, BTreeSet<String>>::new();
    let mut groups = Vec::<(u64, String, Vec<ProcedureCallProposal>)>::new();
    for cohort in stream_cohorts(procedure) {
        for node_id in &cohort {
            cohorts_by_node.insert(node_id.clone(), cohort.clone());
        }
        let ready_members = cohort
            .iter()
            .filter_map(|node_id| candidate_by_id.get(node_id))
            .count();
        if ready_members != cohort.len() {
            for node_id in cohort
                .iter()
                .filter(|node_id| candidate_by_id.contains_key(*node_id))
            {
                blocked_nodes.push(ProcedureBlock {
                    node_id: node_id.clone(),
                    reason: ProcedureBlockReason::StreamingCohortIncomplete,
                });
            }
            continue;
        }
        if cohort.len() > maximum_parallelism {
            return Err(CoreError::ExecutorUnavailable(
                "Procedure parallelism is too small to start every endpoint in a live-stream cohort"
                    .into(),
            ));
        }
        let mut members = Vec::with_capacity(cohort.len());
        for node_id in &cohort {
            members.push(candidate_by_id[node_id].1.clone());
        }
        let cost = cohort
            .iter()
            .filter_map(|node_id| candidate_by_id.get(node_id).map(|(cost, _)| *cost))
            .max()
            .unwrap_or(0);
        let first_id = cohort.iter().next().cloned().unwrap_or_default();
        groups.push((cost, first_id, members));
    }
    for (node_id, (cost, proposal)) in &candidate_by_id {
        if cohorts_by_node.contains_key(node_id) {
            continue;
        }
        groups.push((*cost, node_id.clone(), vec![proposal.clone()]));
    }
    groups.sort_by_key(|(cost, first_id, _)| (std::cmp::Reverse(*cost), first_id.clone()));
    for (_, _, group) in groups {
        if proposal_wave.len() + group.len() > maximum_parallelism {
            continue;
        }
        let group_capabilities = group
            .iter()
            .map(|proposal| capability_by_id[proposal.capability_id.as_str()])
            .collect::<Vec<_>>();
        if group_capabilities
            .iter()
            .enumerate()
            .any(|(position, capability)| {
                group_capabilities[..position]
                    .iter()
                    .any(|active| capabilities_conflict(capability, active))
            })
        {
            if group.len() > 1 {
                return Err(CoreError::ExecutorUnavailable(
                    "Live-stream endpoints have conflicting effects and cannot share a dispatch wave"
                        .into(),
                ));
            }
            continue;
        }
        let can_run = group_capabilities.iter().all(|descriptor| {
            running_capabilities
                .iter()
                .chain(selected_capabilities.iter())
                .all(|active| !capabilities_conflict(descriptor, active))
        });
        if can_run {
            selected_capabilities.extend(group_capabilities);
            proposal_wave.extend(group);
        }
    }

    let completion_satisfied = procedure
        .completion
        .iter()
        .all(|condition| match condition {
            CompletionCondition::AllNodesSucceeded => procedure.nodes.iter().all(|node| {
                matches!(
                    state.nodes.get(&node.id),
                    Some(ProcedureNodeState::Succeeded | ProcedureNodeState::Skipped)
                )
            }),
            CompletionCondition::OutputAvailable { reference } => {
                state.nodes.get(&reference.node_id) == Some(&ProcedureNodeState::Succeeded)
                    && state
                        .verified_outputs
                        .get(&reference.node_id)
                        .is_some_and(|outputs| outputs.contains_key(&reference.output))
            }
            CompletionCondition::Predicate { predicate } => {
                evaluate_procedure_condition(predicate, state) == Some(true)
            }
        });

    Ok(ProcedureAdvance {
        proposal_wave,
        resolved_branches,
        newly_skipped_nodes: newly_skipped.into_iter().collect(),
        blocked_nodes,
        completion_satisfied,
    })
}

fn validate_procedure_assessments(
    procedure: &ProcedureIr,
    assessments: &[CapabilityAssessment],
    current_evidence_ids: &BTreeSet<Uuid>,
) -> CoreResult<()> {
    if assessments.len() > MAX_PROCEDURE_NODES {
        return Err(CoreError::InvalidAction(
            "Procedure capability assessment set exceeds its bound".into(),
        ));
    }
    let descriptors = assessments
        .iter()
        .map(|assessment| assessment.descriptor.clone())
        .collect::<Vec<_>>();
    procedure.validate_against(&descriptors)?;
    let by_id = assessments
        .iter()
        .map(|assessment| (assessment.descriptor.id.as_str(), assessment))
        .collect::<BTreeMap<_, _>>();
    if by_id.len() != assessments.len() {
        return Err(CoreError::InvalidAction(
            "Procedure runtime requires unique capability identities".into(),
        ));
    }
    let executors = crate::features::manifests()
        .into_iter()
        .filter(|manifest| manifest.enabled)
        .map(|manifest| manifest.id)
        .collect::<BTreeSet<_>>();
    for node in &procedure.nodes {
        let ProcedureNodeKind::CapabilityCall { capability_id, .. } = &node.kind else {
            continue;
        };
        let assessment = by_id[capability_id.as_str()];
        if assessment.evidence_state != CapabilityEvidenceState::ReversiblyExperimented
            || assessment
                .descriptor
                .preconditions
                .observed_state_fact_ids
                .iter()
                .any(|id| !current_evidence_ids.contains(id))
            || assessment
                .descriptor
                .executor_id
                .as_ref()
                .is_none_or(|executor| !executors.contains(executor))
        {
            return Err(CoreError::PermissionRequired(
                "Procedure capability lacks current, experimented evidence or a registered Sage executor".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn procedure_digest(procedure: &ProcedureIr) -> CoreResult<String> {
    use sha2::{Digest, Sha256};
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(procedure)?)
    ))
}

fn procedure_successors(procedure: &ProcedureIr) -> BTreeMap<String, BTreeSet<String>> {
    let mut successors = BTreeMap::<String, BTreeSet<String>>::new();
    for node in &procedure.nodes {
        for dependency in &node.depends_on {
            successors
                .entry(dependency.clone())
                .or_default()
                .insert(node.id.clone());
        }
    }
    successors
}

/// Return connected components of the live-stream graph. Every component is
/// one concurrent dispatch cohort; ordinary dependency edges remain separate
/// so a stream does not falsely imply that its consumer needs a completed
/// producer result.
fn stream_cohorts(procedure: &ProcedureIr) -> Vec<BTreeSet<String>> {
    let mut adjacency = BTreeMap::<String, BTreeSet<String>>::new();
    for stream in &procedure.streams {
        adjacency
            .entry(stream.producer_node.clone())
            .or_default()
            .insert(stream.consumer_node.clone());
        adjacency
            .entry(stream.consumer_node.clone())
            .or_default()
            .insert(stream.producer_node.clone());
    }
    let mut visited = BTreeSet::new();
    let mut cohorts = Vec::new();
    for root in adjacency.keys() {
        if visited.contains(root) {
            continue;
        }
        let mut cohort = BTreeSet::new();
        let mut pending = vec![root.clone()];
        while let Some(node) = pending.pop() {
            if !visited.insert(node.clone()) {
                continue;
            }
            cohort.insert(node.clone());
            pending.extend(adjacency.get(&node).into_iter().flatten().cloned());
        }
        cohorts.push(cohort);
    }
    cohorts
}

fn reachable_from(root: &str, successors: &BTreeMap<String, BTreeSet<String>>) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let mut pending = vec![root.to_owned()];
    while let Some(node) = pending.pop() {
        if found.insert(node.clone()) {
            pending.extend(successors.get(&node).into_iter().flatten().cloned());
        }
    }
    found
}

fn dependency_block_reason(
    node: &ProcedureNode,
    state: &ProcedureRuntimeState,
) -> Option<ProcedureBlockReason> {
    for dependency in &node.depends_on {
        match state.nodes.get(dependency) {
            Some(ProcedureNodeState::Failed) => {
                return Some(ProcedureBlockReason::DependencyFailed);
            }
            Some(ProcedureNodeState::Uncertain) => {
                return Some(ProcedureBlockReason::DependencyUncertain);
            }
            Some(ProcedureNodeState::Succeeded | ProcedureNodeState::Skipped) => {}
            Some(ProcedureNodeState::Running) | None => {
                return Some(ProcedureBlockReason::DependenciesPending);
            }
        }
    }
    None
}

fn evaluate_procedure_condition(
    condition: &ProcedureCondition,
    state: &ProcedureRuntimeState,
) -> Option<bool> {
    let reference = match condition {
        ProcedureCondition::BooleanEquals { reference, .. }
        | ProcedureCondition::NumberAtLeast { reference, .. }
        | ProcedureCondition::TextEquals { reference, .. }
        | ProcedureCondition::ResultExists { reference } => reference,
    };
    if state.nodes.get(&reference.node_id) != Some(&ProcedureNodeState::Succeeded) {
        return None;
    }
    let value = state
        .verified_outputs
        .get(&reference.node_id)
        .and_then(|outputs| outputs.get(&reference.output));
    match condition {
        ProcedureCondition::BooleanEquals {
            value: expected, ..
        } => match value {
            Some(ProcedureOutput::Value(ProcedureValue::Boolean(value))) => Some(value == expected),
            _ => None,
        },
        ProcedureCondition::NumberAtLeast { value: minimum, .. } => match value {
            Some(ProcedureOutput::Value(ProcedureValue::Number(value))) => Some(value >= minimum),
            _ => None,
        },
        ProcedureCondition::TextEquals {
            value: expected, ..
        } => match value {
            Some(ProcedureOutput::Value(ProcedureValue::Text(value))) => Some(value == expected),
            _ => None,
        },
        ProcedureCondition::ResultExists { .. } => Some(value.is_some()),
    }
}

pub(crate) fn resolve_call_inputs(
    node: &ProcedureNode,
    procedure: &ProcedureIr,
    state: &ProcedureRuntimeState,
    capabilities: &[CapabilityDescriptor],
) -> Option<BTreeMap<String, ResolvedProcedureInput>> {
    let ProcedureNodeKind::CapabilityCall {
        capability_id,
        input_bindings,
        ..
    } = &node.kind
    else {
        return None;
    };
    let target_capability = capabilities
        .iter()
        .find(|capability| capability.id == *capability_id)?;
    let target_ports = target_capability
        .input_ports
        .iter()
        .map(|port| (port.name.as_str(), port))
        .collect::<BTreeMap<_, _>>();
    let index = procedure
        .nodes
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate))
        .collect::<BTreeMap<_, _>>();
    input_bindings
        .iter()
        .map(|(name, binding)| {
            let target_port = target_ports.get(name.as_str())?.to_owned().clone();
            let resolved = match binding {
                ValueBinding::Literal { value, port } => ResolvedProcedureInput::Value {
                    value: value.clone(),
                    source_port: port.clone(),
                    target_port,
                },
                ValueBinding::Artifact {
                    artifact_id,
                    sha256,
                    port,
                } => ResolvedProcedureInput::Artifact {
                    artifact_id: artifact_id.clone(),
                    owner_task_id: None,
                    sha256: sha256.clone(),
                    size_bytes: None,
                    source_port: port.clone(),
                    target_port,
                },
                ValueBinding::Stream { channel_id } => {
                    let stream = procedure.streams.iter().find(|stream| {
                        stream.id == *channel_id
                            && stream.consumer_node == node.id
                            && stream.consumer_input == *name
                    })?;
                    let source_port = index
                        .get(stream.producer_node.as_str())?
                        .outputs
                        .get(&stream.producer_output)?
                        .clone();
                    ResolvedProcedureInput::Stream {
                        channel_id: channel_id.clone(),
                        source_port,
                        target_port,
                    }
                }
                ValueBinding::Result { producer, output } => {
                    if state.nodes.get(producer) != Some(&ProcedureNodeState::Succeeded) {
                        return None;
                    }
                    let output_value = state
                        .verified_outputs
                        .get(producer)
                        .and_then(|outputs| outputs.get(output))?
                        .clone();
                    let source_port = index.get(producer.as_str())?.outputs.get(output)?.clone();
                    match output_value {
                        ProcedureOutput::Value(value) => ResolvedProcedureInput::Value {
                            value,
                            source_port,
                            target_port,
                        },
                        ProcedureOutput::Artifact(reference) => ResolvedProcedureInput::Artifact {
                            artifact_id: reference.artifact_id.to_string(),
                            owner_task_id: Some(reference.task_id),
                            sha256: reference.sha256,
                            size_bytes: Some(reference.size_bytes),
                            source_port,
                            target_port,
                        },
                    }
                }
            };
            Some((name.clone(), resolved))
        })
        .collect()
}

/// Schedule the straight-line capability subset of `ProcedureIR`. Control
/// flow is rejected rather than flattened; live-stream components are kept as
/// concurrent scheduling cohorts and never split across waves.
pub fn schedule_procedure(
    procedure: &ProcedureIr,
    capabilities: &[CapabilityDescriptor],
    observed_node_micros: &BTreeMap<String, u64>,
    maximum_parallelism: usize,
) -> CoreResult<ProcedureSchedule> {
    if maximum_parallelism == 0 || maximum_parallelism > 16 {
        return Err(CoreError::InvalidAction(
            "Procedure schedule parallelism must be between 1 and 16".into(),
        ));
    }
    if observed_node_micros.len() > MAX_PROCEDURE_NODES
        || observed_node_micros
            .iter()
            .any(|(id, duration)| id.is_empty() || id.len() > 96 || *duration > 600_000_000)
    {
        return Err(CoreError::InvalidAction(
            "Procedure timing observations exceed their bound".into(),
        ));
    }
    procedure.validate_against(capabilities)?;
    let node_ids = procedure
        .nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect::<BTreeSet<_>>();
    if observed_node_micros
        .keys()
        .any(|node_id| !node_ids.contains(node_id.as_str()))
    {
        return Err(CoreError::InvalidAction(
            "Procedure timing observations reference an unknown node".into(),
        ));
    }
    if procedure
        .nodes
        .iter()
        .any(|node| !matches!(node.kind, ProcedureNodeKind::CapabilityCall { .. }))
    {
        return Err(CoreError::ExecutorUnavailable(
            "Static scheduling supports straight-line capability procedures; branches, repeats, and nested skills require the procedure runtime".into(),
        ));
    }

    let by_id = capabilities
        .iter()
        .map(|capability| (capability.id.as_str(), capability))
        .collect::<BTreeMap<_, _>>();
    if by_id.len() != capabilities.len() {
        return Err(CoreError::InvalidAction(
            "Procedure scheduler requires unique capability identities".into(),
        ));
    }
    let mut duration_by_node = BTreeMap::new();
    for node in &procedure.nodes {
        duration_by_node.insert(
            node.id.as_str(),
            observed_node_micros
                .get(&node.id)
                .copied()
                .unwrap_or(1_000_000)
                .max(1),
        );
    }

    let mut successors = BTreeMap::<&str, Vec<&str>>::new();
    for node in &procedure.nodes {
        for dependency in &node.depends_on {
            successors
                .entry(dependency.as_str())
                .or_default()
                .push(node.id.as_str());
        }
    }
    let topological = procedure.validate()?;
    let mut critical_path = BTreeMap::<&str, u64>::new();
    for id in topological.iter().rev().map(String::as_str) {
        let tail = successors
            .get(id)
            .into_iter()
            .flatten()
            .filter_map(|successor| critical_path.get(successor))
            .copied()
            .max()
            .unwrap_or(0);
        critical_path.insert(id, duration_by_node[id].saturating_add(tail));
    }

    let mut completed = BTreeSet::<String>::new();
    let mut waves = Vec::new();
    let mut elapsed = 0_u64;
    while completed.len() < procedure.nodes.len() {
        let mut ready = procedure
            .nodes
            .iter()
            .filter(|node| {
                !completed.contains(&node.id)
                    && node
                        .depends_on
                        .iter()
                        .all(|dependency| completed.contains(dependency))
            })
            .collect::<Vec<_>>();
        ready.sort_by_key(|node| {
            (
                std::cmp::Reverse(critical_path[node.id.as_str()]),
                node.id.as_str(),
            )
        });
        let ready_ids = ready
            .iter()
            .map(|node| node.id.as_str())
            .collect::<BTreeSet<_>>();
        let stream_cohorts = stream_cohorts(procedure);
        let mut cohort_by_node = BTreeMap::<String, BTreeSet<String>>::new();
        let mut groups = Vec::<(u64, String, Vec<&ProcedureNode>)>::new();
        for cohort in stream_cohorts {
            for node_id in &cohort {
                cohort_by_node.insert(node_id.clone(), cohort.clone());
            }
            if !cohort
                .iter()
                .all(|node_id| ready_ids.contains(node_id.as_str()))
            {
                continue;
            }
            if cohort.len() > maximum_parallelism {
                return Err(CoreError::ExecutorUnavailable(
                    "Procedure parallelism is too small to start every endpoint in a live-stream cohort".into(),
                ));
            }
            let members = cohort
                .iter()
                .map(|node_id| {
                    procedure
                        .nodes
                        .iter()
                        .find(|node| node.id == *node_id)
                        .expect("validated cohort node")
                })
                .collect::<Vec<_>>();
            for (position, candidate) in members.iter().enumerate() {
                let ProcedureNodeKind::CapabilityCall { capability_id, .. } = &candidate.kind
                else {
                    unreachable!("validated stream endpoints are capability calls")
                };
                let capability = by_id[capability_id.as_str()];
                if members[..position].iter().any(|scheduled| {
                    let ProcedureNodeKind::CapabilityCall {
                        capability_id: scheduled_id,
                        ..
                    } = &scheduled.kind
                    else {
                        unreachable!("validated stream endpoints are capability calls")
                    };
                    capabilities_conflict(capability, by_id[scheduled_id.as_str()])
                }) {
                    return Err(CoreError::ExecutorUnavailable(
                        "Live-stream endpoints have conflicting effects and cannot share a dispatch wave".into(),
                    ));
                }
            }
            let score = cohort
                .iter()
                .map(|node_id| critical_path[node_id.as_str()])
                .max()
                .unwrap_or(0);
            let first_id = cohort.iter().next().cloned().unwrap_or_default();
            groups.push((score, first_id, members));
        }
        for candidate in ready {
            if cohort_by_node.contains_key(&candidate.id) {
                continue;
            }
            groups.push((
                critical_path[candidate.id.as_str()],
                candidate.id.clone(),
                vec![candidate],
            ));
        }
        groups.sort_by_key(|(score, first_id, _)| (std::cmp::Reverse(*score), first_id.clone()));
        let mut wave = Vec::<&ProcedureNode>::new();
        for (_, _, group) in groups {
            if wave.len() + group.len() > maximum_parallelism {
                continue;
            }
            let can_run = group.iter().all(|candidate| {
                let ProcedureNodeKind::CapabilityCall { capability_id, .. } = &candidate.kind
                else {
                    unreachable!("control-flow nodes were rejected above")
                };
                let candidate_capability = by_id[capability_id.as_str()];
                wave.iter().all(|scheduled| {
                    let ProcedureNodeKind::CapabilityCall {
                        capability_id: scheduled_id,
                        ..
                    } = &scheduled.kind
                    else {
                        unreachable!("control-flow nodes were rejected above")
                    };
                    !capabilities_conflict(candidate_capability, by_id[scheduled_id.as_str()])
                })
            });
            if can_run {
                wave.extend(group);
            }
        }
        if wave.is_empty() {
            return Err(CoreError::InvalidAction(
                "Procedure scheduler found no runnable dependency wave".into(),
            ));
        }
        elapsed = elapsed.saturating_add(
            wave.iter()
                .map(|node| duration_by_node[node.id.as_str()])
                .max()
                .unwrap_or(0),
        );
        for node in &wave {
            completed.insert(node.id.clone());
        }
        waves.push(wave.into_iter().map(|node| node.id.clone()).collect());
    }
    Ok(ProcedureSchedule {
        waves,
        estimated_elapsed_micros: elapsed,
    })
}

fn capabilities_conflict(left: &CapabilityDescriptor, right: &CapabilityDescriptor) -> bool {
    let globally_exclusive = |effects: &BTreeSet<Effect>| {
        effects.iter().any(|effect| {
            matches!(
                effect,
                Effect::ExternalCommitment | Effect::ReleaseData | Effect::ExecuteCode
            )
        })
    };
    if globally_exclusive(&left.effects) || globally_exclusive(&right.effects) {
        return true;
    }
    left.system_id == right.system_id
        && (left.effects.iter().any(|effect| *effect != Effect::Read)
            || right.effects.iter().any(|effect| *effect != Effect::Read))
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalInputSeed {
    pub capability_id: String,
    pub input_name: String,
    /// Only literal values and content-addressed artifacts may enter a new
    /// plan. Result bindings are generated by the backward synthesizer.
    pub binding: ValueBinding,
}

/// Synthesize one deterministic, typed path backward from a requested
/// capability output. Only current, reversibly experimented descriptors
/// whose executor is in the caller's compiled registry are considered. A
/// successful result is a proposal; it grants no authority or permission.
pub fn synthesize_goal_procedure(
    goal_capability_id: &str,
    goal_output: &str,
    seeds: Vec<GoalInputSeed>,
    assessments: &[CapabilityAssessment],
    registered_executors: &BTreeSet<String>,
    current_evidence_ids: &BTreeSet<Uuid>,
) -> CoreResult<ProcedureIr> {
    bounded(goal_capability_id, 128, "goal capability identifier")?;
    bounded(goal_output, 96, "goal output name")?;
    if assessments.len() > 512 || seeds.len() > 512 {
        return Err(CoreError::InvalidAction(
            "Goal synthesis inputs exceed their bounded search size".into(),
        ));
    }

    let mut assessment_ids = BTreeSet::new();
    let mut available = BTreeMap::new();
    for assessment in assessments {
        let descriptor = &assessment.descriptor;
        if !assessment_ids.insert(descriptor.id.clone()) {
            return Err(CoreError::InvalidAction(
                "Goal synthesis requires unique capability identities".into(),
            ));
        }
        let Some(executor_id) = descriptor.executor_id.as_ref() else {
            continue;
        };
        if assessment.evidence_state != CapabilityEvidenceState::ReversiblyExperimented
            || !registered_executors.contains(executor_id)
            || descriptor.preconditions.observed_state_fact_ids.is_empty()
            || descriptor.preconditions.observed_state_fact_ids.len() > 32
            || descriptor
                .evidence_ids
                .iter()
                .any(|evidence| !current_evidence_ids.contains(evidence))
            || descriptor
                .preconditions
                .observed_state_fact_ids
                .iter()
                .any(|evidence| !current_evidence_ids.contains(evidence))
        {
            continue;
        }
        descriptor.validate()?;
        available.insert(descriptor.id.clone(), descriptor.clone());
    }
    let goal = available.get(goal_capability_id).ok_or_else(|| {
        CoreError::ExecutorUnavailable(
            "Goal capability has no current experimented Sage executor".into(),
        )
    })?;
    if !goal
        .output_ports
        .iter()
        .any(|port| port.name == goal_output)
    {
        return Err(CoreError::InvalidAction(
            "Goal output is not declared by the selected capability".into(),
        ));
    }

    let mut seed_map = BTreeMap::new();
    for seed in seeds {
        bounded(&seed.capability_id, 128, "seed capability identifier")?;
        bounded(&seed.input_name, 96, "seed input name")?;
        if seed_map
            .insert((seed.capability_id, seed.input_name), seed.binding)
            .is_some()
        {
            return Err(CoreError::InvalidAction(
                "Goal input seeds must be unique".into(),
            ));
        }
    }

    let mut builder = BackwardProcedureBuilder {
        capabilities: available,
        seeds: seed_map,
        built: BTreeMap::new(),
        active: BTreeSet::new(),
        nodes: Vec::new(),
    };
    let goal_node = builder.build_node(goal_capability_id)?;
    if !builder.seeds.is_empty() {
        return Err(CoreError::InvalidAction(
            "Goal input seed does not belong to a required procedure input".into(),
        ));
    }
    let procedure = ProcedureIr {
        schema_version: 1,
        id: Uuid::new_v4().to_string(),
        nodes: builder.nodes,
        streams: Vec::new(),
        completion: vec![CompletionCondition::OutputAvailable {
            reference: ValueReference {
                node_id: goal_node,
                output: goal_output.into(),
            },
        }],
    };
    let descriptors = builder.capabilities.into_values().collect::<Vec<_>>();
    procedure.validate_against(&descriptors)?;
    Ok(procedure)
}

struct BackwardProcedureBuilder {
    capabilities: BTreeMap<String, CapabilityDescriptor>,
    seeds: BTreeMap<(String, String), ValueBinding>,
    built: BTreeMap<String, String>,
    active: BTreeSet<String>,
    nodes: Vec<ProcedureNode>,
}

impl BackwardProcedureBuilder {
    fn build_node(&mut self, capability_id: &str) -> CoreResult<String> {
        if let Some(existing) = self.built.get(capability_id) {
            return Ok(existing.clone());
        }
        if !self.active.insert(capability_id.into()) {
            return Err(CoreError::InvalidAction(
                "Goal synthesis found a cyclic capability dependency".into(),
            ));
        }
        if self.nodes.len() + self.active.len() > MAX_PROCEDURE_NODES {
            return Err(CoreError::InvalidAction(
                "Goal synthesis exceeded its procedure-node bound".into(),
            ));
        }
        let Some(capability) = self.capabilities.get(capability_id).cloned() else {
            self.active.remove(capability_id);
            return Err(CoreError::ExecutorUnavailable(
                "A required producer has no current experimented Sage executor".into(),
            ));
        };
        let mut input_bindings = BTreeMap::new();
        let mut depends_on = BTreeSet::new();
        for input in &capability.input_ports {
            if let Some(binding) = self
                .seeds
                .remove(&(capability_id.to_string(), input.name.clone()))
            {
                validate_synthesis_seed(&binding, input)?;
                input_bindings.insert(input.name.clone(), binding);
                continue;
            }

            let producers = self
                .capabilities
                .iter()
                .filter(|(producer_id, _)| producer_id.as_str() != capability_id)
                .flat_map(|(producer_id, producer)| {
                    producer
                        .output_ports
                        .iter()
                        .filter(|output| ports_compatible(output, input))
                        .map(|output| (producer_id.clone(), output.name.clone()))
                })
                .collect::<Vec<_>>();
            let [(producer_id, output_name)] = producers.as_slice() else {
                self.active.remove(capability_id);
                return Err(CoreError::ExecutorUnavailable(if producers.is_empty() {
                    format!(
                        "No compatible current capability produces input {}",
                        input.name
                    )
                } else {
                    format!(
                        "Input {} has multiple compatible producers; explicit review is required",
                        input.name
                    )
                }));
            };
            let producer_node = self.build_node(producer_id)?;
            depends_on.insert(producer_node.clone());
            input_bindings.insert(
                input.name.clone(),
                ValueBinding::Result {
                    producer: producer_node,
                    output: output_name.clone(),
                },
            );
        }
        self.active.remove(capability_id);
        let node_id = format!("node-{}", self.nodes.len() + 1);
        let outputs = capability
            .output_ports
            .iter()
            .map(|port| (port.name.clone(), port.clone()))
            .collect();
        self.nodes.push(ProcedureNode {
            id: node_id.clone(),
            depends_on,
            outputs,
            kind: ProcedureNodeKind::CapabilityCall {
                capability_id: capability.id.clone(),
                system_id: capability.system_id,
                system_fingerprint: capability.system_fingerprint.clone(),
                input_bindings,
            },
        });
        self.built.insert(capability_id.into(), node_id.clone());
        Ok(node_id)
    }
}

fn validate_synthesis_seed(binding: &ValueBinding, destination: &DataPort) -> CoreResult<()> {
    let source = match binding {
        ValueBinding::Literal { value, port } => {
            value.validate()?;
            if value.port_type() != port.value_type
                || u64::try_from(serde_json::to_vec(value)?.len()).unwrap_or(u64::MAX)
                    > port.max_bytes
            {
                return Err(CoreError::InvalidAction(
                    "Literal seed does not satisfy its typed size bound".into(),
                ));
            }
            port
        }
        ValueBinding::Artifact {
            artifact_id,
            sha256,
            port,
        } => {
            bounded(artifact_id, 128, "artifact identifier")?;
            if !valid_digest(sha256) || port.value_type != PortType::Bytes {
                return Err(CoreError::InvalidAction(
                    "Artifact seed requires a content digest and bytes port".into(),
                ));
            }
            port
        }
        ValueBinding::Result { .. } => {
            return Err(CoreError::InvalidAction(
                "Caller-supplied result bindings cannot introduce procedure edges".into(),
            ));
        }
        ValueBinding::Stream { .. } => {
            return Err(CoreError::InvalidAction(
                "Caller-supplied stream bindings cannot introduce execution channels".into(),
            ));
        }
    };
    if !ports_compatible(source, destination) {
        return Err(CoreError::InvalidAction(
            "Goal input seed exceeds its declared type, size, or privacy contract".into(),
        ));
    }
    Ok(())
}

fn ports_compatible(source: &DataPort, destination: &DataPort) -> bool {
    source.value_type == destination.value_type
        && source.max_bytes <= destination.max_bytes
        && source.privacy <= destination.privacy
}

fn validate_seed_port(input_name: &str, value_type: PortType, port: &DataPort) -> CoreResult<()> {
    port.validate()?;
    if port.name != input_name || port.value_type != value_type {
        return Err(CoreError::InvalidAction(
            "Procedure input seed does not match its declared data port".into(),
        ));
    }
    Ok(())
}

fn transitive_ancestors(nodes: &[ProcedureNode]) -> BTreeMap<String, BTreeSet<String>> {
    let mut ancestors = BTreeMap::new();
    for node in nodes {
        let mut found = node.depends_on.clone();
        let mut pending = node.depends_on.iter().cloned().collect::<Vec<_>>();
        while let Some(id) = pending.pop() {
            if let Some(parent) = nodes.iter().find(|candidate| candidate.id == id) {
                for ancestor in &parent.depends_on {
                    if found.insert(ancestor.clone()) {
                        pending.push(ancestor.clone());
                    }
                }
            }
        }
        ancestors.insert(node.id.clone(), found);
    }
    ancestors
}

fn validate_reference(
    index: &BTreeMap<&str, usize>,
    nodes: &[ProcedureNode],
    ancestors: &BTreeMap<String, BTreeSet<String>>,
    consumer: &str,
    producer: &str,
    output: &str,
) -> CoreResult<()> {
    let Some(&producer_position) = index.get(producer) else {
        return Err(CoreError::InvalidAction(
            "Procedure result binding references an unknown node".into(),
        ));
    };
    if !nodes[producer_position].outputs.contains_key(output) {
        return Err(CoreError::InvalidAction(
            "Procedure result binding references an unknown output".into(),
        ));
    }
    if !consumer.is_empty()
        && !ancestors
            .get(consumer)
            .is_some_and(|values| values.contains(producer))
    {
        return Err(CoreError::InvalidAction(
            "Procedure results must depend on their producer".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticAnchor {
    pub id: String,
    pub role: String,
    pub accessible_name: String,
    pub ancestor_names: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerPrimitive {
    ResolveSemanticAnchor,
    ReadControlState,
    InvokeRegisteredCapability,
    WaitForObservedState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControllerStep {
    pub id: String,
    pub primitive: ControllerPrimitive,
    pub anchor: SemanticAnchor,
    pub capability_id: Option<String>,
    /// Inputs captured from the verified procedure. Values may be literals or
    /// references to earlier verified outputs; live streams and task-owned
    /// artifacts are not portable controller inputs.
    #[serde(default)]
    pub input_bindings: BTreeMap<String, ValueBinding>,
    pub depends_on: BTreeSet<String>,
    pub outputs: BTreeMap<String, DataPort>,
    pub preconditions: Vec<ProcedureCondition>,
    pub expected_effect: String,
    pub verification: String,
    pub restoration: Option<String>,
    pub evidence_ids: Vec<Uuid>,
}

/// Learned controller data may reference only sealed Sage primitives. It
/// cannot contain scripts, selectors that execute arbitrary JavaScript,
/// coordinates, grants, or OS authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControllerIr {
    pub schema_version: u32,
    pub system_id: Uuid,
    pub system_fingerprint: String,
    pub interface_fingerprint: String,
    pub steps: Vec<ControllerStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerStatus {
    Draft,
    Reviewed,
    Disabled,
    Invalidated,
}

impl ControllerStatus {
    pub(crate) fn from_storage_value(value: &str) -> CoreResult<Self> {
        match value {
            "draft" => Ok(Self::Draft),
            "reviewed" => Ok(Self::Reviewed),
            "disabled" => Ok(Self::Disabled),
            "invalidated" => Ok(Self::Invalidated),
            _ => Err(CoreError::VerificationFailed(
                "Stored controller status is invalid".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredController {
    /// SHA-256 of the canonical serialized ControllerIR payload.
    pub id: String,
    pub controller: ControllerIr,
    pub status: ControllerStatus,
    pub revision: u64,
    pub reviewed_observation_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct RevalidatedStoredController {
    pub record: StoredController,
    pub rebinding: ControllerRebinding,
}

pub fn controller_digest(controller: &ControllerIr) -> CoreResult<String> {
    controller.validate()?;
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(controller)?)
    ))
}

impl ControllerIr {
    pub fn validate(&self) -> CoreResult<()> {
        if !matches!(self.schema_version, 1 | 2)
            || self.system_id.is_nil()
            || !valid_digest(&self.system_fingerprint)
            || !valid_digest(&self.interface_fingerprint)
            || self.steps.is_empty()
            || self.steps.len() > 64
        {
            return Err(CoreError::InvalidAction(
                "Controller identity or step bounds are invalid".into(),
            ));
        }
        let ids = self
            .steps
            .iter()
            .map(|step| step.id.as_str())
            .collect::<BTreeSet<_>>();
        if ids.len() != self.steps.len() {
            return Err(CoreError::InvalidAction(
                "Controller step identifiers must be unique".into(),
            ));
        }
        let mut ancestors = BTreeMap::<String, BTreeSet<String>>::new();
        let mut pending = self
            .steps
            .iter()
            .map(|step| step.id.as_str())
            .collect::<BTreeSet<_>>();
        while !pending.is_empty() {
            let before = pending.len();
            for step in &self.steps {
                if !pending.contains(step.id.as_str())
                    || step
                        .depends_on
                        .iter()
                        .any(|dependency| pending.contains(dependency.as_str()))
                {
                    continue;
                }
                let mut found = step.depends_on.clone();
                for dependency in &step.depends_on {
                    if let Some(parent) = ancestors.get(dependency) {
                        found.extend(parent.iter().cloned());
                    }
                }
                ancestors.insert(step.id.clone(), found);
                pending.remove(step.id.as_str());
            }
            if pending.len() == before {
                return Err(CoreError::InvalidAction(
                    "Controller dependency graph is cyclic or references an unknown step".into(),
                ));
            }
        }
        for step in &self.steps {
            bounded(&step.id, 96, "controller step identifier")?;
            bounded(&step.anchor.id, 256, "semantic anchor identifier")?;
            bounded(&step.anchor.role, 96, "semantic anchor role")?;
            bounded_nonsecret(&step.anchor.accessible_name, 256, "semantic anchor name")?;
            if step.anchor.ancestor_names.len() > 8
                || step.evidence_ids.is_empty()
                || step.evidence_ids.len() > 32
            {
                return Err(CoreError::InvalidAction(
                    "Controller anchor or evidence bounds are invalid".into(),
                ));
            }
            for name in &step.anchor.ancestor_names {
                bounded_nonsecret(name, 256, "semantic ancestor name")?;
            }
            bounded_nonsecret(
                &step.expected_effect,
                MAX_PROCEDURE_TEXT,
                "controller effect",
            )?;
            bounded_nonsecret(
                &step.verification,
                MAX_PROCEDURE_TEXT,
                "controller verification",
            )?;
            if let Some(restoration) = &step.restoration {
                bounded_nonsecret(restoration, MAX_PROCEDURE_TEXT, "controller restoration")?;
            }
            if matches!(
                step.primitive,
                ControllerPrimitive::InvokeRegisteredCapability
            ) != step.capability_id.is_some()
            {
                return Err(CoreError::InvalidAction(
                    "Only the registered-capability primitive may name a capability".into(),
                ));
            }
            if let Some(capability_id) = &step.capability_id {
                bounded(capability_id, 128, "controller capability identifier")?;
            }
            if step.input_bindings.len() > 32
                || (!step.input_bindings.is_empty() && self.schema_version < 2)
            {
                return Err(CoreError::InvalidAction(
                    "Controller input bindings require schema version 2 and at most 32 inputs"
                        .into(),
                ));
            }
            for (name, binding) in &step.input_bindings {
                bounded(name, 96, "controller input name")?;
                match binding {
                    ValueBinding::Literal { value, port } => {
                        value.validate()?;
                        port.validate()?;
                        if port.name != *name
                            || value.port_type() != port.value_type
                            || value.payload_bytes() > port.max_bytes
                        {
                            return Err(CoreError::InvalidAction(
                                "Controller literal does not match its typed input port".into(),
                            ));
                        }
                    }
                    ValueBinding::Result { producer, output } => {
                        bounded(producer, 96, "controller result producer")?;
                        bounded(output, 96, "controller result output")?;
                        let source = self
                            .steps
                            .iter()
                            .find(|candidate| candidate.id == *producer)
                            .and_then(|candidate| candidate.outputs.get(output))
                            .ok_or_else(|| {
                                CoreError::InvalidAction(
                                    "Controller result input references an unknown output".into(),
                                )
                            })?;
                        if !ancestors
                            .get(&step.id)
                            .is_some_and(|parents| parents.contains(producer))
                        {
                            return Err(CoreError::InvalidAction(
                                "Controller result input must depend on its producer".into(),
                            ));
                        }
                        source.validate()?;
                    }
                    ValueBinding::Stream { .. } => {
                        return Err(CoreError::ExecutorUnavailable(
                            "Controller IR cannot yet preserve live stream semantics".into(),
                        ));
                    }
                    ValueBinding::Artifact { .. } => {
                        return Err(CoreError::PolicyDenied(
                            "Task-owned artifacts cannot be retained in a reusable controller"
                                .into(),
                        ));
                    }
                }
            }
            if step
                .depends_on
                .iter()
                .any(|dependency| !ids.contains(dependency.as_str()) || dependency == &step.id)
            {
                return Err(CoreError::InvalidAction(
                    "Controller dependency is invalid".into(),
                ));
            }
            if step.outputs.len() > 32 {
                return Err(CoreError::InvalidAction(
                    "Controller output bound exceeded".into(),
                ));
            }
            for (name, output) in &step.outputs {
                bounded(name, 96, "controller output name")?;
                if name != &output.name {
                    return Err(CoreError::InvalidAction(
                        "Controller output name mismatch".into(),
                    ));
                }
                output.validate()?;
            }
            for condition in &step.preconditions {
                condition.validate_value()?;
                for reference in condition.references() {
                    let known = self.steps.iter().any(|producer| {
                        producer.id == reference.node_id
                            && producer.outputs.contains_key(&reference.output)
                    });
                    let ordered = ancestors
                        .get(&step.id)
                        .is_some_and(|parents| parents.contains(&reference.node_id));
                    if !known || !ordered {
                        return Err(CoreError::InvalidAction(
                            "Controller preconditions must reference an earlier observed result"
                                .into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Check compiled controller calls against the current descriptive
    /// capability signatures. This validates representation only; the normal
    /// broker still resolves targets, policy, permission and verification at
    /// every invocation.
    pub fn validate_against(&self, capabilities: &[CapabilityAssessment]) -> CoreResult<()> {
        self.validate()?;
        let by_id = capabilities
            .iter()
            .map(|assessment| (assessment.descriptor.id.as_str(), assessment))
            .collect::<BTreeMap<_, _>>();
        if by_id.len() != capabilities.len() {
            return Err(CoreError::InvalidAction(
                "Controller validation requires unique capability identities".into(),
            ));
        }
        for step in &self.steps {
            let Some(capability_id) = &step.capability_id else {
                continue;
            };
            let assessment = by_id.get(capability_id.as_str()).ok_or_else(|| {
                CoreError::ExecutorUnavailable(
                    "Controller references an undiscovered capability".into(),
                )
            })?;
            let descriptor = &assessment.descriptor;
            descriptor.validate()?;
            if descriptor.system_id != self.system_id
                || descriptor.system_fingerprint != self.system_fingerprint
                || descriptor.executor_id.is_none()
                || step.input_bindings.len() != descriptor.input_ports.len()
                || step.outputs.len() != descriptor.output_ports.len()
            {
                return Err(CoreError::VerificationFailed(
                    "Controller call no longer matches its target or capability signature".into(),
                ));
            }
            for port in &descriptor.input_ports {
                let Some(binding) = step.input_bindings.get(&port.name) else {
                    return Err(CoreError::InvalidAction(
                        "Controller call is missing a declared capability input".into(),
                    ));
                };
                let supplied = match binding {
                    ValueBinding::Literal {
                        value,
                        port: supplied,
                    } => {
                        if !ports_compatible(supplied, port)
                            || value.port_type() != supplied.value_type
                        {
                            return Err(CoreError::InvalidAction(
                                "Controller literal exceeds the capability input contract".into(),
                            ));
                        }
                        supplied
                    }
                    ValueBinding::Result { producer, output } => {
                        let source = self
                            .steps
                            .iter()
                            .find(|candidate| candidate.id == *producer)
                            .and_then(|candidate| candidate.outputs.get(output))
                            .ok_or_else(|| {
                                CoreError::InvalidAction(
                                    "Controller result binding is unknown".into(),
                                )
                            })?;
                        if !ports_compatible(source, port) {
                            return Err(CoreError::InvalidAction(
                                "Controller result exceeds the capability input contract".into(),
                            ));
                        }
                        source
                    }
                    ValueBinding::Stream { .. } | ValueBinding::Artifact { .. } => {
                        return Err(CoreError::ExecutorUnavailable(
                            "Controller call contains a nonportable input binding".into(),
                        ));
                    }
                };
                if supplied.value_type != port.value_type {
                    return Err(CoreError::InvalidAction(
                        "Controller input type differs from the capability signature".into(),
                    ));
                }
            }
            for port in &descriptor.output_ports {
                if step.outputs.get(&port.name) != Some(port) {
                    return Err(CoreError::InvalidAction(
                        "Controller outputs differ from the capability signature".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Compile one fully settled ProcedureIR run into an unreviewed ControllerIR
/// candidate. Only a single-system DAG of registered, reversibly experimented
/// UI capabilities is representable today. Unsupported control flow fails
/// closed rather than silently changing procedure meaning.
pub(crate) fn compile_controller_from_verified_procedure(
    procedure: &ProcedureIr,
    runtime: &ProcedureRuntimeState,
    capabilities: &[CapabilityAssessment],
    observations: &[ObservationEnvelope],
    now: DateTime<Utc>,
) -> CoreResult<ControllerIr> {
    let topological = procedure.validate()?;
    runtime.validate_for(procedure)?;
    if !procedure.streams.is_empty()
        || procedure.completion.len() != 1
        || !matches!(
            procedure.completion[0],
            CompletionCondition::AllNodesSucceeded
        )
        || procedure
            .nodes
            .iter()
            .any(|node| !matches!(node.kind, ProcedureNodeKind::CapabilityCall { .. }))
        || procedure.nodes.iter().any(|node| {
            runtime.nodes.get(&node.id) != Some(&ProcedureNodeState::Succeeded)
                || !runtime.dispatch_receipts.contains_key(&node.id)
                || !runtime.verification_evidence.contains_key(&node.id)
        })
    {
        return Err(CoreError::ExecutorUnavailable(
            "Controller compilation currently requires a fully succeeded, non-streaming capability DAG with all-nodes completion".into(),
        ));
    }

    let system_id = match &procedure.nodes[0].kind {
        ProcedureNodeKind::CapabilityCall { system_id, .. } => *system_id,
        _ => unreachable!("non-capability nodes were rejected above"),
    };
    let system_fingerprint = match &procedure.nodes[0].kind {
        ProcedureNodeKind::CapabilityCall {
            system_fingerprint, ..
        } => system_fingerprint.clone(),
        _ => unreachable!("non-capability nodes were rejected above"),
    };
    let mut target_observations = Vec::new();
    for observation in observations.iter().filter(|observation| {
        observation.system_id == system_id && observation.system_fingerprint == system_fingerprint
    }) {
        observation.validate()?;
        if observation.observed_at > now + Duration::seconds(5) {
            return Err(CoreError::VerificationFailed(
                "Controller compilation found future-dated target evidence".into(),
            ));
        }
        target_observations.push(observation);
    }
    let current_evidence = target_observations
        .iter()
        .map(|observation| observation.id)
        .collect::<BTreeSet<_>>();
    validate_procedure_assessments(procedure, capabilities, &current_evidence)?;
    let fresh_interface_observations = target_observations
        .iter()
        .copied()
        .filter(|observation| {
            observation.observed_at >= now - Duration::seconds(10)
                && observation.session_id.is_none()
                && observation.worker_session.is_none()
                && matches!(
                    observation.origin,
                    crate::world_model::EvidenceOrigin::OperatingSystem
                        | crate::world_model::EvidenceOrigin::Application
                        | crate::world_model::EvidenceOrigin::Browser
                )
        })
        .collect::<Vec<_>>();

    let capabilities_by_id = capabilities
        .iter()
        .map(|assessment| (assessment.descriptor.id.as_str(), assessment))
        .collect::<BTreeMap<_, _>>();
    for node in &procedure.nodes {
        let ProcedureNodeKind::CapabilityCall { capability_id, .. } = &node.kind else {
            unreachable!("non-capability nodes were rejected above")
        };
        let assessment = capabilities_by_id[capability_id.as_str()];
        for evidence_id in &assessment.descriptor.preconditions.observed_state_fact_ids {
            let Some(observation) = target_observations
                .iter()
                .find(|observation| observation.id == *evidence_id)
            else {
                return Err(CoreError::VerificationFailed(
                    "Controller preconditions require current stored observations".into(),
                ));
            };
            if observation.origin == crate::world_model::EvidenceOrigin::Model {
                return Err(CoreError::PolicyDenied(
                    "Model-generated hypotheses cannot establish controller preconditions".into(),
                ));
            }
        }
    }
    let mut interface_fingerprint: Option<String> = None;
    let mut steps = Vec::with_capacity(procedure.nodes.len());
    for node_id in topological {
        let node = procedure
            .nodes
            .iter()
            .find(|node| node.id == node_id)
            .expect("topological order contains known nodes");
        let ProcedureNodeKind::CapabilityCall {
            capability_id,
            system_id: call_system,
            system_fingerprint: call_fingerprint,
            input_bindings,
        } = &node.kind
        else {
            unreachable!("non-capability nodes were rejected above")
        };
        if *call_system != system_id || call_fingerprint != &system_fingerprint {
            return Err(CoreError::VerificationFailed(
                "Controller compilation cannot combine multiple system identities".into(),
            ));
        }
        let assessment = capabilities_by_id[capability_id.as_str()];
        let descriptor = &assessment.descriptor;
        let control_id = descriptor.interface_control_id.as_deref().ok_or_else(|| {
            CoreError::ExecutorUnavailable(
                "Controller compilation requires a capability bound to an observed UI control"
                    .into(),
            )
        })?;
        let supported_observation = fresh_interface_observations
            .iter()
            .filter(|observation| {
                descriptor.evidence_ids.contains(&observation.id)
                    && observation.system_id == system_id
                    && observation.system_fingerprint == system_fingerprint
                    && observation.session_id.is_none()
                    && observation.worker_session.is_none()
                    && matches!(
                        observation.origin,
                        crate::world_model::EvidenceOrigin::OperatingSystem
                            | crate::world_model::EvidenceOrigin::Application
                            | crate::world_model::EvidenceOrigin::Browser
                    )
            })
            .max_by_key(|observation| observation.observed_at)
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Controller capability has no fresh passive interface evidence".into(),
                )
            })?;
        let (anchor, fingerprint) =
            semantic_anchor_from_observation(supported_observation, control_id)?;
        match &interface_fingerprint {
            Some(existing) if existing != &fingerprint => {
                return Err(CoreError::VerificationFailed(
                    "Controller steps were observed under different interface fingerprints".into(),
                ));
            }
            None => interface_fingerprint = Some(fingerprint),
            _ => {}
        }
        let verification_id = runtime.verification_evidence[&node.id];
        let Some(verification_observation) = target_observations
            .iter()
            .find(|observation| observation.id == verification_id)
        else {
            return Err(CoreError::VerificationFailed(
                "Controller compilation requires stored current verification for every procedure step"
                    .into(),
            ));
        };
        if !matches!(
            verification_observation.origin,
            crate::world_model::EvidenceOrigin::OperatingSystem
                | crate::world_model::EvidenceOrigin::Application
                | crate::world_model::EvidenceOrigin::Browser
        ) {
            return Err(CoreError::VerificationFailed(
                "Procedure verification evidence cannot originate from a model or advertisement"
                    .into(),
            ));
        }
        let mut evidence_ids = descriptor.evidence_ids.clone();
        if !evidence_ids.contains(&verification_id) {
            evidence_ids.push(verification_id);
        }
        evidence_ids.sort_unstable();
        evidence_ids.dedup();
        if evidence_ids.len() > 32 {
            return Err(CoreError::InvalidAction(
                "Controller step exceeds its retained evidence bound".into(),
            ));
        }
        steps.push(ControllerStep {
            id: node.id.clone(),
            primitive: ControllerPrimitive::InvokeRegisteredCapability,
            anchor,
            capability_id: Some(capability_id.clone()),
            input_bindings: input_bindings.clone(),
            depends_on: node.depends_on.clone(),
            outputs: node.outputs.clone(),
            preconditions: Vec::new(),
            expected_effect: format!("Apply the registered capability: {}", descriptor.label),
            verification: descriptor.verification.clone(),
            restoration: descriptor.restoration.clone(),
            evidence_ids,
        });
    }
    let controller = ControllerIr {
        schema_version: 2,
        system_id,
        system_fingerprint,
        interface_fingerprint: interface_fingerprint.ok_or_else(|| {
            CoreError::VerificationFailed("Controller has no interface evidence".into())
        })?,
        steps,
    };
    controller.validate_against(capabilities)?;
    Ok(controller)
}

fn semantic_anchor_from_observation(
    observation: &ObservationEnvelope,
    control_id: &str,
) -> CoreResult<(SemanticAnchor, String)> {
    let mut interface = None;
    let mut role = None;
    let mut label = None;
    let mut ancestors = None;
    let mut enabled = None;
    let mut control_count = BTreeSet::new();
    for fact in &observation.facts {
        if fact.subject.is_none()
            && matches!(
                fact.name.as_str(),
                "application.interface_fingerprint" | "browser.interface_fingerprint"
            )
        {
            let FactValue::Identifier(value) = &fact.value else {
                return Err(CoreError::VerificationFailed(
                    "Controller evidence has an invalid interface fingerprint".into(),
                ));
            };
            if !valid_digest(value) || interface.replace(value.clone()).is_some() {
                return Err(CoreError::VerificationFailed(
                    "Controller evidence has an ambiguous interface fingerprint".into(),
                ));
            }
        }
        let Some(subject) = fact.subject.as_deref() else {
            continue;
        };
        control_count.insert(subject.to_owned());
        if subject != control_id {
            continue;
        }
        let slot = match fact.name.as_str() {
            "control.role" | "application.control.role" | "browser.control.role" => Some(&mut role),
            "control.label" | "application.control.label" | "browser.control.label" => {
                Some(&mut label)
            }
            "application.control.ancestors" | "browser.control.ancestors" => {
                if let FactValue::Text(value) = &fact.value {
                    let values = if value.starts_with('[') {
                        serde_json::from_str::<Vec<String>>(value).map_err(|_| {
                            CoreError::VerificationFailed(
                                "Controller anchor ancestor evidence is malformed".into(),
                            )
                        })?
                    } else {
                        value.split(" > ").map(str::to_owned).collect()
                    };
                    if values.len() > 8
                        || values.iter().any(|value| {
                            bounded_nonsecret(value, 256, "controller ancestor name").is_err()
                        })
                        || ancestors.replace(values).is_some()
                    {
                        return Err(CoreError::VerificationFailed(
                            "Controller anchor ancestor evidence is invalid or duplicated".into(),
                        ));
                    }
                } else {
                    return Err(CoreError::VerificationFailed(
                        "Controller anchor ancestor evidence has an invalid type".into(),
                    ));
                }
                None
            }
            "control.enabled" | "application.control.enabled" | "browser.control.enabled" => {
                if let FactValue::Boolean(value) = fact.value {
                    if enabled.replace(value).is_some() {
                        return Err(CoreError::VerificationFailed(
                            "Controller enabled-state evidence is duplicated".into(),
                        ));
                    }
                } else {
                    return Err(CoreError::VerificationFailed(
                        "Controller enabled-state evidence has an invalid type".into(),
                    ));
                }
                None
            }
            _ => None,
        };
        if let Some(slot) = slot {
            let FactValue::Text(value) = &fact.value else {
                return Err(CoreError::VerificationFailed(
                    "Controller semantic anchor evidence has an invalid type".into(),
                ));
            };
            bounded_nonsecret(value, 256, "controller semantic anchor text")?;
            if slot.replace(value.clone()).is_some() {
                return Err(CoreError::VerificationFailed(
                    "Controller semantic anchor evidence is duplicated".into(),
                ));
            }
        }
    }
    let fingerprint = interface.ok_or_else(|| {
        CoreError::VerificationFailed(
            "Controller evidence omitted the interface fingerprint".into(),
        )
    })?;
    let role = role.ok_or_else(|| {
        CoreError::VerificationFailed("Controller evidence omitted the control role".into())
    })?;
    let label = label.ok_or_else(|| {
        CoreError::VerificationFailed(
            "Controller evidence omitted the accessible control name".into(),
        )
    })?;
    if enabled != Some(true) {
        return Err(CoreError::VerificationFailed(
            "Controller evidence does not identify an enabled control".into(),
        ));
    }
    let ancestors = ancestors.unwrap_or_default();
    let same_semantic_anchor_count = observation
        .facts
        .iter()
        .filter(|fact| {
            fact.name == "control.role"
                || fact.name == "application.control.role"
                || fact.name == "browser.control.role"
        })
        .filter(|fact| {
            fact.subject.as_deref().is_some_and(|subject| {
                observation.facts.iter().any(|candidate| {
                    candidate.subject.as_deref() == Some(subject)
                        && matches!(
                            (candidate.name.as_str(), &candidate.value),
                            (
                                "control.label"
                                    | "application.control.label"
                                    | "browser.control.label",
                                FactValue::Text(value)
                            )
                                if value == &label
                        )
                        && matches!(
                            (fact.name.as_str(), &fact.value),
                            (
                                "control.role"
                                    | "application.control.role"
                                    | "browser.control.role",
                                FactValue::Text(value)
                            )
                                if value == &role
                        )
                })
            })
        })
        .count();
    if same_semantic_anchor_count != 1 || !control_count.contains(control_id) {
        return Err(CoreError::VerificationFailed(
            "Controller anchor is ambiguous in its supporting observation".into(),
        ));
    }
    Ok((
        SemanticAnchor {
            id: control_id.into(),
            role,
            accessible_name: label,
            ancestor_names: ancestors,
        },
        fingerprint,
    ))
}

/// A semantic anchor resolution against one fresh passive observation. The
/// resulting IDs are descriptive bindings only; callers must still resolve
/// the current target, apply policy, obtain permission and a short-lived
/// capability, execute, and independently verify before any effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControllerRebinding {
    pub observed_interface_fingerprint: String,
    pub interface_changed: bool,
    pub control_ids_by_step: BTreeMap<String, String>,
}

/// Storage-backed freshness metadata for one observation used to learn a
/// controller. Callers populate this only from unforgotten world-model
/// observations, not from the controller or a model response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentControllerEvidence {
    pub system_id: Uuid,
    pub expires_at: DateTime<Utc>,
}

#[derive(Default)]
struct ObservedSemanticControl {
    role: Option<String>,
    label: Option<String>,
    ancestors: Option<Vec<String>>,
    enabled: Option<bool>,
}

/// Rebind a reviewed controller to one exact, freshly observed system. An
/// unchanged interface requires the original stable control ID; a changed
/// interface may use a unique semantic match by role, accessible name and
/// available ancestor path. Missing ancestry, stale evidence, disabled
/// controls, duplicate matches, or a target change fail closed.
/// `current_evidence` must be populated from stored observations. The function
/// checks both their exact owning system and expiry against `now`.
pub fn revalidate_controller(
    controller: &ControllerIr,
    observation: &ObservationEnvelope,
    current_evidence: &BTreeMap<Uuid, CurrentControllerEvidence>,
    now: DateTime<Utc>,
) -> CoreResult<ControllerRebinding> {
    controller.validate()?;
    if observation.system_id != controller.system_id
        || observation.system_fingerprint != controller.system_fingerprint
        || observation.observed_at > now + Duration::seconds(5)
        || observation.observed_at < now - Duration::seconds(10)
        || observation.facts.is_empty()
        || observation.facts.len() > 512
        || observation.privacy >= crate::contracts::Sensitivity::Restricted
        || observation.session_id.is_some()
        || observation.worker_session.is_some()
        || !matches!(
            observation.origin,
            crate::world_model::EvidenceOrigin::OperatingSystem
                | crate::world_model::EvidenceOrigin::Application
                | crate::world_model::EvidenceOrigin::Browser
        )
    {
        return Err(CoreError::VerificationFailed(
            "Controller observation is stale, private, or belongs to another system".into(),
        ));
    }
    if controller
        .steps
        .iter()
        .flat_map(|step| &step.evidence_ids)
        .any(|id| {
            current_evidence.get(id).is_none_or(|evidence| {
                evidence.system_id != controller.system_id || evidence.expires_at <= now
            })
        })
    {
        return Err(CoreError::VerificationFailed(
            "Controller evidence expired or was invalidated; relearning is required".into(),
        ));
    }

    let mut observed_fingerprint = None;
    let mut controls = BTreeMap::<String, ObservedSemanticControl>::new();
    for fact in &observation.facts {
        if fact.subject.is_none()
            && (fact.name == "application.interface_fingerprint"
                || fact.name == "browser.interface_fingerprint")
        {
            let FactValue::Identifier(value) = &fact.value else {
                return Err(CoreError::VerificationFailed(
                    "Observed interface fingerprint has an invalid type".into(),
                ));
            };
            if !valid_digest(value) || observed_fingerprint.replace(value.clone()).is_some() {
                return Err(CoreError::VerificationFailed(
                    "Observed interface fingerprint is missing or ambiguous".into(),
                ));
            }
            continue;
        }
        let Some(subject) = fact.subject.as_deref() else {
            continue;
        };
        bounded(subject, 256, "observed semantic control identifier")?;
        let control = controls.entry(subject.to_owned()).or_default();
        let assign_text = |slot: &mut Option<String>| -> CoreResult<()> {
            let FactValue::Text(value) = &fact.value else {
                return Err(CoreError::VerificationFailed(
                    "Observed semantic control fact has an invalid type".into(),
                ));
            };
            bounded_nonsecret(value, 256, "observed semantic control text")?;
            if slot.replace(value.clone()).is_some() {
                return Err(CoreError::VerificationFailed(
                    "Observed semantic control fact is duplicated".into(),
                ));
            }
            Ok(())
        };
        match fact.name.as_str() {
            "control.role" | "application.control.role" | "browser.control.role" => {
                assign_text(&mut control.role)?
            }
            "control.label" | "application.control.label" | "browser.control.label" => {
                assign_text(&mut control.label)?
            }
            "application.control.ancestors" | "browser.control.ancestors" => {
                let FactValue::Text(value) = &fact.value else {
                    return Err(CoreError::VerificationFailed(
                        "Observed semantic ancestor path has an invalid type".into(),
                    ));
                };
                bounded_nonsecret(value, 1024, "observed semantic ancestor path")?;
                let path = if value.starts_with('[') {
                    serde_json::from_str::<Vec<String>>(value).map_err(|_| {
                        CoreError::VerificationFailed(
                            "Observed semantic ancestor list is malformed".into(),
                        )
                    })?
                } else {
                    value.split(" > ").map(str::to_owned).collect()
                };
                if path.len() > 8 {
                    return Err(CoreError::VerificationFailed(
                        "Observed semantic ancestor list exceeds its bound".into(),
                    ));
                }
                for name in &path {
                    bounded_nonsecret(name, 256, "observed semantic ancestor name")?;
                }
                if control.ancestors.replace(path).is_some() {
                    return Err(CoreError::VerificationFailed(
                        "Observed semantic ancestor path is duplicated".into(),
                    ));
                }
            }
            "control.enabled" | "application.control.enabled" | "browser.control.enabled" => {
                let FactValue::Boolean(value) = &fact.value else {
                    return Err(CoreError::VerificationFailed(
                        "Observed control enabled state has an invalid type".into(),
                    ));
                };
                if control.enabled.replace(*value).is_some() {
                    return Err(CoreError::VerificationFailed(
                        "Observed control enabled state is duplicated".into(),
                    ));
                }
            }
            _ => {}
        }
    }
    let observed_fingerprint = observed_fingerprint.ok_or_else(|| {
        CoreError::VerificationFailed("Observation omitted its interface fingerprint".into())
    })?;
    let interface_changed = observed_fingerprint != controller.interface_fingerprint;
    let mut control_ids_by_step = BTreeMap::new();
    for step in &controller.steps {
        let ancestor_matches = |control: &ObservedSemanticControl| {
            step.anchor.ancestor_names.is_empty()
                || control.ancestors.as_deref() == Some(step.anchor.ancestor_names.as_slice())
        };
        let matches = controls
            .iter()
            .filter(|(id, control)| {
                control.enabled == Some(true)
                    && control.role.as_deref() == Some(step.anchor.role.as_str())
                    && control.label.as_deref() == Some(step.anchor.accessible_name.as_str())
                    && ancestor_matches(control)
                    && (interface_changed || id.as_str() == step.anchor.id)
            })
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        let [control_id] = matches.as_slice() else {
            return Err(CoreError::VerificationFailed(if matches.is_empty() {
                "Controller anchor no longer matches a unique enabled control; relearning is required".into()
            } else {
                "Controller anchor matches multiple controls; relearning is required".into()
            }));
        };
        control_ids_by_step.insert(step.id.clone(), control_id.clone());
    }
    Ok(ControllerRebinding {
        observed_interface_fingerprint: observed_fingerprint,
        interface_changed,
        control_ids_by_step,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeerResource {
    CpuMilliseconds,
    MemoryBytes,
    ScratchBytes,
    GpuJobs,
    CameraFrames,
    DisplayFrames,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuotaEnforcement {
    OperatingSystemEnforced,
    CooperativeTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceQuota {
    pub maximum: u64,
    pub enforcement: QuotaEnforcement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeerLease {
    pub id: Uuid,
    pub peer_id: Uuid,
    pub peer_fingerprint: String,
    pub allowed_resources: BTreeMap<PeerResource, ResourceQuota>,
    pub allowed_jobs: BTreeSet<String>,
    pub disclosure_scope: BTreeSet<String>,
    pub expires_at: DateTime<Utc>,
    pub revocation_generation: u64,
}

impl PeerLease {
    pub fn validate(&self, now: DateTime<Utc>) -> CoreResult<()> {
        if self.id.is_nil()
            || self.peer_id.is_nil()
            || !valid_digest(&self.peer_fingerprint)
            || self.allowed_resources.is_empty()
            || self.allowed_resources.len() > 16
            || self.allowed_jobs.is_empty()
            || self.allowed_jobs.len() > 32
            || self.disclosure_scope.len() > 32
            || self.expires_at <= now
            || self.expires_at > now + Duration::hours(MAX_PEER_LEASE_HOURS)
        {
            return Err(CoreError::InvalidAction(
                "Peer lease identity, scope, quota, or expiry is invalid".into(),
            ));
        }
        if self
            .allowed_resources
            .values()
            .any(|quota| quota.maximum == 0)
        {
            return Err(CoreError::InvalidAction(
                "Peer lease resource quotas must be positive".into(),
            ));
        }
        for job in &self.allowed_jobs {
            bounded(job, 96, "peer job identifier")?;
        }
        for scope in &self.disclosure_scope {
            bounded_nonsecret(scope, 128, "peer disclosure scope")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointArtifact {
    pub artifact_id: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub media_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedCheckpointResult {
    pub result_id: String,
    pub action_id: String,
    pub verified_at: DateTime<Utc>,
    pub evidence_ids: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingObligation {
    pub id: String,
    pub kind: String,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointState {
    Paused,
    Settled,
    NeedsReview,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionOwner {
    pub device_id: Uuid,
    pub generation: u64,
}

/// Transfer data for a task. Its type contains evidence and progress only;
/// capabilities, provider secrets and reusable grants have no fields here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCheckpoint {
    pub schema_version: u32,
    pub task_id: Uuid,
    pub intent: String,
    pub procedure: Option<ProcedureIr>,
    pub procedure_state: BTreeMap<String, String>,
    pub artifacts: Vec<CheckpointArtifact>,
    pub verified_results: Vec<VerifiedCheckpointResult>,
    pub receipt_ids: Vec<Uuid>,
    pub pending_obligations: Vec<PendingObligation>,
    pub state: CheckpointState,
    pub execution_owner: ExecutionOwner,
    pub dispatched_effects_settled: bool,
}

impl TaskCheckpoint {
    pub fn validate(&self) -> CoreResult<()> {
        if self.schema_version != 1
            || self.task_id.is_nil()
            || self.execution_owner.device_id.is_nil()
            || self.execution_owner.generation == 0
            || self.intent.len() > 16 * 1024
            || self.procedure_state.len() > MAX_CHECKPOINT_ITEMS
            || self.artifacts.len() > MAX_CHECKPOINT_ITEMS
            || self.verified_results.len() > MAX_CHECKPOINT_ITEMS
            || self.receipt_ids.len() > MAX_CHECKPOINT_ITEMS
            || self.pending_obligations.len() > MAX_CHECKPOINT_ITEMS
            || self.receipt_ids.iter().any(Uuid::is_nil)
            || self.receipt_ids.iter().collect::<BTreeSet<_>>().len() != self.receipt_ids.len()
            || self
                .artifacts
                .iter()
                .map(|artifact| artifact.artifact_id.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                != self.artifacts.len()
            || self
                .verified_results
                .iter()
                .map(|result| result.result_id.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                != self.verified_results.len()
            || self
                .pending_obligations
                .iter()
                .map(|obligation| obligation.id.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                != self.pending_obligations.len()
            || (self.state == CheckpointState::Settled
                && (!self.dispatched_effects_settled || !self.pending_obligations.is_empty()))
        {
            return Err(CoreError::InvalidAction(
                "Task checkpoint exceeds identity or size bounds".into(),
            ));
        }
        bounded_nonsecret(&self.intent, 16 * 1024, "checkpoint intent")?;
        if let Some(procedure) = &self.procedure {
            procedure.validate()?;
        }
        for artifact in &self.artifacts {
            bounded(&artifact.artifact_id, 128, "checkpoint artifact identifier")?;
            bounded(&artifact.media_type, 128, "checkpoint media type")?;
            if !valid_digest(&artifact.sha256) || artifact.size_bytes > 512 * 1024 * 1024 {
                return Err(CoreError::InvalidAction(
                    "Checkpoint artifact digest or size is invalid".into(),
                ));
            }
        }
        for result in &self.verified_results {
            bounded(&result.result_id, 128, "checkpoint result identifier")?;
            bounded(&result.action_id, 128, "checkpoint action identifier")?;
            if result.evidence_ids.is_empty()
                || result.evidence_ids.len() > 32
                || result.evidence_ids.iter().any(Uuid::is_nil)
                || result.evidence_ids.iter().collect::<BTreeSet<_>>().len()
                    != result.evidence_ids.len()
            {
                return Err(CoreError::InvalidAction(
                    "Verified checkpoint results require bounded evidence references".into(),
                ));
            }
        }
        for (key, value) in &self.procedure_state {
            bounded(key, 96, "checkpoint state key")?;
            bounded_nonsecret(value, 1024, "checkpoint state value")?;
        }
        for obligation in &self.pending_obligations {
            bounded(&obligation.id, 128, "checkpoint obligation identifier")?;
            bounded(&obligation.kind, 64, "checkpoint obligation kind")?;
            bounded_nonsecret(&obligation.summary, 512, "checkpoint obligation summary")?;
        }
        Ok(())
    }

    /// Advance the ownership fence only after dispatched effects settle and
    /// pending obligations are resolved. The destination obtains no grant.
    pub fn transfer_ownership(&mut self, destination: Uuid) -> CoreResult<()> {
        if destination.is_nil()
            || destination == self.execution_owner.device_id
            || !self.dispatched_effects_settled
            || !self.pending_obligations.is_empty()
            || self.state != CheckpointState::Settled
        {
            return Err(CoreError::PermissionRequired(
                "Checkpoint ownership can move only from a settled task with no pending obligations".into(),
            ));
        }
        let generation = self
            .execution_owner
            .generation
            .checked_add(1)
            .ok_or_else(|| {
                CoreError::PermissionRequired("Checkpoint ownership generation is exhausted".into())
            })?;
        self.execution_owner = ExecutionOwner {
            device_id: destination,
            generation,
        };
        self.state = CheckpointState::Paused;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::Effect;
    use crate::secrets::testing::MemorySecretStore;
    use tempfile::tempdir;

    fn task_checkpoint_fixture() -> TaskCheckpoint {
        TaskCheckpoint {
            schema_version: 1,
            task_id: Uuid::new_v4(),
            intent: "continue the verified task".into(),
            procedure: None,
            procedure_state: BTreeMap::new(),
            artifacts: Vec::new(),
            verified_results: Vec::new(),
            receipt_ids: Vec::new(),
            pending_obligations: Vec::new(),
            state: CheckpointState::Settled,
            execution_owner: ExecutionOwner {
                device_id: Uuid::new_v4(),
                generation: 1,
            },
            dispatched_effects_settled: true,
        }
    }

    #[test]
    fn task_checkpoint_rejects_replayed_receipts_and_unsettled_settled_state() {
        let mut checkpoint = task_checkpoint_fixture();
        let receipt = Uuid::new_v4();
        checkpoint.receipt_ids = vec![receipt, receipt];
        assert!(checkpoint.validate().is_err());

        checkpoint.receipt_ids = vec![Uuid::nil()];
        assert!(checkpoint.validate().is_err());

        checkpoint.receipt_ids.clear();
        checkpoint.dispatched_effects_settled = false;
        assert!(checkpoint.validate().is_err());
    }

    #[test]
    fn task_checkpoint_requires_unique_artifacts_obligations_and_result_evidence() {
        let mut checkpoint = task_checkpoint_fixture();
        let artifact = CheckpointArtifact {
            artifact_id: "artifact-1".into(),
            sha256: "a".repeat(64),
            size_bytes: 128,
            media_type: "application/octet-stream".into(),
        };
        checkpoint.artifacts = vec![artifact.clone(), artifact];
        assert!(checkpoint.validate().is_err());

        checkpoint.artifacts.clear();
        let evidence = Uuid::new_v4();
        checkpoint.verified_results = vec![VerifiedCheckpointResult {
            result_id: "result-1".into(),
            action_id: "action-1".into(),
            verified_at: Utc::now(),
            evidence_ids: vec![evidence, evidence],
        }];
        assert!(checkpoint.validate().is_err());

        checkpoint.verified_results.clear();
        checkpoint.pending_obligations = vec![
            PendingObligation {
                id: "obligation-1".into(),
                kind: "review".into(),
                summary: "Review the saved state".into(),
            },
            PendingObligation {
                id: "obligation-1".into(),
                kind: "review".into(),
                summary: "Review the saved state".into(),
            },
        ];
        assert!(checkpoint.validate().is_err());
    }

    #[test]
    fn task_checkpoint_ownership_transfer_uses_a_checked_generation_fence() {
        let mut checkpoint = task_checkpoint_fixture();
        let prior_owner = checkpoint.execution_owner.clone();
        let destination = Uuid::new_v4();
        checkpoint.transfer_ownership(destination).unwrap();
        assert_eq!(checkpoint.execution_owner.device_id, destination);
        assert_eq!(
            checkpoint.execution_owner.generation,
            prior_owner.generation + 1
        );
        assert_eq!(checkpoint.state, CheckpointState::Paused);

        checkpoint.state = CheckpointState::Settled;
        checkpoint.execution_owner.generation = u64::MAX;
        let before = checkpoint.clone();
        assert!(checkpoint.transfer_ownership(Uuid::new_v4()).is_err());
        assert_eq!(checkpoint.execution_owner, before.execution_owner);
        assert_eq!(checkpoint.state, before.state);
    }

    fn port(
        name: &str,
        value_type: PortType,
        max_bytes: u64,
        privacy: crate::contracts::Sensitivity,
    ) -> DataPort {
        DataPort {
            name: name.into(),
            value_type,
            max_bytes,
            privacy,
        }
    }

    fn assessment(
        id: &str,
        inputs: Vec<DataPort>,
        outputs: Vec<DataPort>,
        executor_id: &str,
        evidence_id: Uuid,
    ) -> CapabilityAssessment {
        CapabilityAssessment {
            descriptor: CapabilityDescriptor {
                schema_version: 1,
                id: id.into(),
                system_id: Uuid::new_v4(),
                system_fingerprint: "a".repeat(64),
                interface_control_id: None,
                interface_probe_kind: None,
                label: id.into(),
                input_ports: inputs,
                output_ports: outputs,
                preconditions: crate::world_model::Preconditions {
                    observed_state_fact_ids: vec![evidence_id],
                    description: "Fresh matching system state".into(),
                },
                effects: BTreeSet::from([Effect::Read]),
                verification: "Independent result verification".into(),
                restoration: None,
                cancellation: "Stop future dispatch and settle current work".into(),
                executor_id: Some(executor_id.into()),
                evidence_ids: vec![evidence_id],
                updated_at: Utc::now(),
            },
            evidence_state: CapabilityEvidenceState::ReversiblyExperimented,
        }
    }

    fn registered_executor() -> String {
        crate::features::manifests()
            .into_iter()
            .find(|manifest| manifest.enabled)
            .expect("test build includes a registered executor")
            .id
    }

    fn call_node(
        id: &str,
        capability: &CapabilityAssessment,
        depends_on: BTreeSet<String>,
        input_bindings: BTreeMap<String, ValueBinding>,
    ) -> ProcedureNode {
        ProcedureNode {
            id: id.into(),
            depends_on,
            outputs: capability
                .descriptor
                .output_ports
                .iter()
                .cloned()
                .map(|port| (port.name.clone(), port))
                .collect(),
            kind: ProcedureNodeKind::CapabilityCall {
                capability_id: capability.descriptor.id.clone(),
                system_id: capability.descriptor.system_id,
                system_fingerprint: capability.descriptor.system_fingerprint.clone(),
                input_bindings,
            },
        }
    }

    fn streaming_procedure_fixture() -> (ProcedureIr, Vec<CapabilityAssessment>, BTreeSet<Uuid>) {
        let executor = registered_executor();
        let producer_evidence = Uuid::new_v4();
        let consumer_evidence = Uuid::new_v4();
        let producer = assessment(
            "media.produce",
            Vec::new(),
            vec![port(
                "chunks",
                PortType::Bytes,
                1024,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            producer_evidence,
        );
        let consumer = assessment(
            "media.consume",
            vec![port(
                "chunks",
                PortType::Bytes,
                2048,
                crate::contracts::Sensitivity::Private,
            )],
            Vec::new(),
            &executor,
            consumer_evidence,
        );
        let channel = StreamChannel {
            id: "media-chunks".into(),
            producer_node: "producer".into(),
            producer_output: "chunks".into(),
            consumer_node: "consumer".into(),
            consumer_input: "chunks".into(),
            capacity_items: 2,
            maximum_item_bytes: 1024,
            backpressure: StreamBackpressure::BlockProducer,
        };
        let procedure = ProcedureIr {
            schema_version: 2,
            id: "streaming-media-pipeline".into(),
            nodes: vec![
                call_node("producer", &producer, BTreeSet::new(), BTreeMap::new()),
                call_node(
                    "consumer",
                    &consumer,
                    BTreeSet::new(),
                    BTreeMap::from([(
                        "chunks".into(),
                        ValueBinding::Stream {
                            channel_id: channel.id.clone(),
                        },
                    )]),
                ),
            ],
            streams: vec![channel],
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        (
            procedure,
            vec![producer, consumer],
            BTreeSet::from([producer_evidence, consumer_evidence]),
        )
    }

    #[test]
    fn procedure_streams_share_one_aggregate_memory_budget() {
        let executor = registered_executor();
        let large_bytes = 12 * 1024 * 1024;
        let output_ports = (0..3)
            .map(|index| {
                port(
                    &format!("out_{index}"),
                    PortType::Bytes,
                    large_bytes,
                    crate::contracts::Sensitivity::Private,
                )
            })
            .collect::<Vec<_>>();
        let input_ports = (0..3)
            .map(|index| {
                port(
                    &format!("in_{index}"),
                    PortType::Bytes,
                    large_bytes,
                    crate::contracts::Sensitivity::Private,
                )
            })
            .collect::<Vec<_>>();
        let source = assessment(
            "source",
            Vec::new(),
            output_ports.clone(),
            &executor,
            Uuid::new_v4(),
        );
        let sink = assessment("sink", input_ports, Vec::new(), &executor, Uuid::new_v4());
        let bindings = (0..3)
            .map(|index| {
                (
                    format!("in_{index}"),
                    ValueBinding::Stream {
                        channel_id: format!("stream-{index}"),
                    },
                )
            })
            .collect();
        let nodes = vec![
            call_node("producer", &source, BTreeSet::new(), BTreeMap::new()),
            call_node("consumer", &sink, BTreeSet::new(), bindings),
        ];
        let streams = (0..3)
            .map(|index| StreamChannel {
                id: format!("stream-{index}"),
                producer_node: "producer".into(),
                producer_output: format!("out_{index}"),
                consumer_node: "consumer".into(),
                consumer_input: format!("in_{index}"),
                capacity_items: 1,
                maximum_item_bytes: large_bytes,
                backpressure: StreamBackpressure::BlockProducer,
            })
            .collect();
        let procedure = ProcedureIr {
            schema_version: 2,
            id: "aggregate-stream-bound".into(),
            nodes,
            streams,
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };

        assert!(procedure.validate().is_err());
    }

    #[test]
    fn live_stream_endpoints_are_proposed_scheduled_and_recorded_as_one_wave() {
        let (procedure, assessments, current_evidence_ids) = streaming_procedure_fixture();
        let descriptors = assessments
            .iter()
            .map(|assessment| assessment.descriptor.clone())
            .collect::<Vec<_>>();
        procedure.validate_against(&descriptors).unwrap();
        let schedule = schedule_procedure(&procedure, &descriptors, &BTreeMap::new(), 2).unwrap();
        assert_eq!(schedule.waves.len(), 1);
        assert_eq!(schedule.waves[0].len(), 2);

        let mut state = ProcedureRuntimeState::new(&procedure).unwrap();
        let advance = advance_procedure(
            &procedure,
            &assessments,
            &current_evidence_ids,
            &mut state,
            &BTreeMap::new(),
            2,
        )
        .unwrap();
        assert_eq!(advance.proposal_wave.len(), 2);
        let consumer = advance
            .proposal_wave
            .iter()
            .find(|proposal| proposal.node_id == "consumer")
            .unwrap();
        assert!(matches!(
            consumer.inputs.get("chunks"),
            Some(ResolvedProcedureInput::Stream {
                channel_id,
                source_port,
                target_port,
            }) if channel_id == "media-chunks"
                && source_port.value_type == PortType::Bytes
                && target_port.value_type == PortType::Bytes
        ));

        assert!(
            state
                .record_dispatched(
                    &procedure,
                    &assessments,
                    &current_evidence_ids,
                    "producer",
                    Uuid::new_v4(),
                )
                .is_err()
        );
        assert!(state.nodes.is_empty());
        state
            .record_dispatched_wave(
                &procedure,
                &assessments,
                &current_evidence_ids,
                BTreeMap::from([
                    ("producer".into(), Uuid::new_v4()),
                    ("consumer".into(), Uuid::new_v4()),
                ]),
            )
            .unwrap();
        assert_eq!(state.nodes.len(), 2);
        assert!(
            state
                .nodes
                .values()
                .all(|status| *status == ProcedureNodeState::Running)
        );
    }

    #[test]
    fn live_streams_reject_legacy_versions_wrong_bindings_and_serial_dependencies() {
        let (mut procedure, _, _) = streaming_procedure_fixture();
        procedure.schema_version = 1;
        assert!(procedure.validate().is_err());

        let (mut procedure, _, _) = streaming_procedure_fixture();
        if let ProcedureNodeKind::CapabilityCall { input_bindings, .. } =
            &mut procedure.nodes[1].kind
        {
            input_bindings.insert(
                "chunks".into(),
                ValueBinding::Stream {
                    channel_id: "different-channel".into(),
                },
            );
        }
        assert!(procedure.validate().is_err());

        let (mut procedure, assessments, _) = streaming_procedure_fixture();
        procedure.nodes[1].depends_on.insert("producer".into());
        assert!(procedure.validate().is_err());

        let (mut procedure, _, _) = streaming_procedure_fixture();
        procedure.nodes[0].kind = ProcedureNodeKind::NestedSkill {
            skill_id: "unreviewed-producer".into(),
            reviewed_digest: "f".repeat(64),
        };
        assert!(procedure.validate().is_err());

        let descriptors = assessments
            .iter()
            .map(|assessment| assessment.descriptor.clone())
            .collect::<Vec<_>>();
        let (procedure, _, _) = streaming_procedure_fixture();
        assert!(schedule_procedure(&procedure, &descriptors, &BTreeMap::new(), 1).is_err());
        let mut state = ProcedureRuntimeState::new(&procedure).unwrap();
        assert!(
            advance_procedure(
                &procedure,
                &assessments,
                &BTreeSet::from([
                    assessments[0].descriptor.evidence_ids[0],
                    assessments[1].descriptor.evidence_ids[0],
                ]),
                &mut state,
                &BTreeMap::new(),
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn backward_goal_synthesis_builds_only_a_unique_typed_evidence_bound_path() {
        let executor = registered_executor();
        let producer_evidence = Uuid::new_v4();
        let goal_evidence = Uuid::new_v4();
        let input = port(
            "text",
            PortType::Text,
            4096,
            crate::contracts::Sensitivity::Private,
        );
        let mut producer = assessment(
            "source.read",
            vec![port(
                "seed",
                PortType::Text,
                4096,
                crate::contracts::Sensitivity::Private,
            )],
            vec![input.clone()],
            &executor,
            producer_evidence,
        );
        let goal = assessment(
            "document.export",
            vec![port(
                "text",
                PortType::Text,
                8192,
                crate::contracts::Sensitivity::Restricted,
            )],
            vec![port(
                "video",
                PortType::Bytes,
                4 * 1024 * 1024,
                crate::contracts::Sensitivity::Restricted,
            )],
            &executor,
            goal_evidence,
        );
        let procedure = synthesize_goal_procedure(
            "document.export",
            "video",
            vec![GoalInputSeed {
                capability_id: "source.read".into(),
                input_name: "seed".into(),
                binding: ValueBinding::Literal {
                    value: ProcedureValue::Text("selected source".into()),
                    port: port(
                        "seed",
                        PortType::Text,
                        4096,
                        crate::contracts::Sensitivity::Private,
                    ),
                },
            }],
            &[producer.clone(), goal.clone()],
            &BTreeSet::from([executor]),
            &BTreeSet::from([producer_evidence, goal_evidence]),
        )
        .unwrap();
        assert_eq!(procedure.validate().unwrap().len(), 2);
        assert_eq!(procedure.nodes.len(), 2);
        assert!(matches!(
            &procedure.nodes[1].kind,
            ProcedureNodeKind::CapabilityCall { capability_id, input_bindings, .. }
                if capability_id == "document.export"
                    && matches!(input_bindings.get("text"), Some(ValueBinding::Result { producer, output })
                        if producer == "node-1" && output == "text")
        ));
        let schedule = schedule_procedure(
            &procedure,
            &[producer.descriptor.clone(), goal.descriptor.clone()],
            &BTreeMap::new(),
            4,
        )
        .unwrap();
        assert_eq!(
            schedule.waves,
            vec![vec!["node-1".to_string()], vec!["node-2".to_string()]]
        );
        producer.evidence_state = CapabilityEvidenceState::PassivelyObserved;
        assert!(
            synthesize_goal_procedure(
                "document.export",
                "video",
                Vec::new(),
                &[producer, goal],
                &BTreeSet::new(),
                &BTreeSet::from([producer_evidence, goal_evidence]),
            )
            .is_err()
        );
    }

    #[test]
    fn backward_goal_synthesis_rejects_privacy_downgrades_and_ambiguous_producers() {
        let executor = registered_executor();
        let evidence = [Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
        let restricted_source = assessment(
            "source.private",
            Vec::new(),
            vec![port(
                "text",
                PortType::Text,
                4096,
                crate::contracts::Sensitivity::Restricted,
            )],
            &executor,
            evidence[0],
        );
        let goal = assessment(
            "document.export",
            vec![port(
                "text",
                PortType::Text,
                8192,
                crate::contracts::Sensitivity::Private,
            )],
            vec![port(
                "video",
                PortType::Bytes,
                4096,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            evidence[1],
        );
        let mut second_source = restricted_source.clone();
        second_source.descriptor.id = "source.private_alt".into();
        second_source.descriptor.system_id = Uuid::new_v4();
        second_source
            .descriptor
            .preconditions
            .observed_state_fact_ids = vec![evidence[2]];
        second_source.descriptor.evidence_ids = vec![evidence[2]];
        assert!(
            synthesize_goal_procedure(
                "document.export",
                "video",
                Vec::new(),
                &[restricted_source.clone(), goal.clone()],
                &BTreeSet::from([executor.clone()]),
                &BTreeSet::from([evidence[0], evidence[1]]),
            )
            .is_err()
        );

        let permissive_source = assessment(
            "source.public",
            Vec::new(),
            vec![port(
                "text",
                PortType::Text,
                1024,
                crate::contracts::Sensitivity::Public,
            )],
            &executor,
            evidence[0],
        );
        let mut second_public = permissive_source.clone();
        second_public.descriptor.id = "source.public_alt".into();
        second_public.descriptor.system_id = Uuid::new_v4();
        second_public
            .descriptor
            .preconditions
            .observed_state_fact_ids = vec![evidence[2]];
        second_public.descriptor.evidence_ids = vec![evidence[2]];
        let ambiguous_goal = assessment(
            "document.export",
            vec![port(
                "text",
                PortType::Text,
                8192,
                crate::contracts::Sensitivity::Private,
            )],
            vec![port(
                "video",
                PortType::Bytes,
                4096,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            evidence[1],
        );
        assert!(
            synthesize_goal_procedure(
                "document.export",
                "video",
                Vec::new(),
                &[permissive_source, second_public, ambiguous_goal],
                &BTreeSet::from([executor]),
                &BTreeSet::from([evidence[0], evidence[1], evidence[2]]),
            )
            .is_err()
        );
    }

    #[test]
    fn procedure_scheduler_groups_independent_reads_and_orders_by_critical_path_cost() {
        let executor = registered_executor();
        let evidence_a = Uuid::new_v4();
        let evidence_b = Uuid::new_v4();
        let a = assessment(
            "read.a",
            Vec::new(),
            vec![port(
                "result",
                PortType::Text,
                1024,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            evidence_a,
        );
        let b = assessment(
            "read.b",
            Vec::new(),
            vec![port(
                "result",
                PortType::Text,
                1024,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            evidence_b,
        );
        let node = |id: &str, capability: &CapabilityAssessment| ProcedureNode {
            id: id.into(),
            depends_on: BTreeSet::new(),
            outputs: capability
                .descriptor
                .output_ports
                .iter()
                .cloned()
                .map(|port| (port.name.clone(), port))
                .collect(),
            kind: ProcedureNodeKind::CapabilityCall {
                capability_id: capability.descriptor.id.clone(),
                system_id: capability.descriptor.system_id,
                system_fingerprint: capability.descriptor.system_fingerprint.clone(),
                input_bindings: BTreeMap::new(),
            },
        };
        let procedure = ProcedureIr {
            schema_version: 1,
            id: "independent-reads".into(),
            nodes: vec![node("fast", &a), node("slow", &b)],
            streams: Vec::new(),
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        let schedule = schedule_procedure(
            &procedure,
            &[a.descriptor, b.descriptor],
            &BTreeMap::from([("fast".into(), 1_000_000), ("slow".into(), 9_000_000)]),
            4,
        )
        .unwrap();
        assert_eq!(
            schedule.waves,
            vec![vec!["slow".to_string(), "fast".to_string()]]
        );
        assert_eq!(schedule.estimated_elapsed_micros, 9_000_000);
    }

    #[test]
    fn procedure_scheduler_serializes_nonread_effects_on_the_same_system() {
        let executor = registered_executor();
        let mut a = assessment(
            "control.a",
            Vec::new(),
            vec![port(
                "result",
                PortType::Boolean,
                8,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            Uuid::new_v4(),
        );
        let mut b = assessment(
            "control.b",
            Vec::new(),
            vec![port(
                "result",
                PortType::Boolean,
                8,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            Uuid::new_v4(),
        );
        b.descriptor.system_id = a.descriptor.system_id;
        a.descriptor.effects = BTreeSet::from([Effect::ControlApplication]);
        b.descriptor.effects = BTreeSet::from([Effect::ControlApplication]);
        let node = |id: &str, capability: &CapabilityAssessment| ProcedureNode {
            id: id.into(),
            depends_on: BTreeSet::new(),
            outputs: capability
                .descriptor
                .output_ports
                .iter()
                .cloned()
                .map(|port| (port.name.clone(), port))
                .collect(),
            kind: ProcedureNodeKind::CapabilityCall {
                capability_id: capability.descriptor.id.clone(),
                system_id: capability.descriptor.system_id,
                system_fingerprint: capability.descriptor.system_fingerprint.clone(),
                input_bindings: BTreeMap::new(),
            },
        };
        let procedure = ProcedureIr {
            schema_version: 1,
            id: "same-system-controls".into(),
            nodes: vec![node("first", &a), node("second", &b)],
            streams: Vec::new(),
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        let schedule = schedule_procedure(
            &procedure,
            &[a.descriptor, b.descriptor],
            &BTreeMap::new(),
            4,
        )
        .unwrap();
        assert_eq!(schedule.waves.len(), 2);
        assert_ne!(schedule.waves[0], schedule.waves[1]);
    }

    #[test]
    fn procedure_runtime_waits_for_verified_results_and_selects_one_branch() {
        let executor = registered_executor();
        let probe = assessment(
            "state.probe",
            Vec::new(),
            vec![port(
                "ready",
                PortType::Boolean,
                8,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            Uuid::new_v4(),
        );
        let yes = assessment(
            "operation.yes",
            Vec::new(),
            Vec::new(),
            &executor,
            Uuid::new_v4(),
        );
        let no = assessment(
            "operation.no",
            Vec::new(),
            Vec::new(),
            &executor,
            Uuid::new_v4(),
        );
        let branch = ProcedureNode {
            id: "choose".into(),
            depends_on: BTreeSet::from(["probe".into()]),
            outputs: BTreeMap::new(),
            kind: ProcedureNodeKind::Branch {
                condition: ProcedureCondition::BooleanEquals {
                    reference: ValueReference {
                        node_id: "probe".into(),
                        output: "ready".into(),
                    },
                    value: true,
                },
                when_true: "yes".into(),
                when_false: "no".into(),
            },
        };
        let procedure = ProcedureIr {
            schema_version: 1,
            id: "verified-branch".into(),
            nodes: vec![
                call_node("probe", &probe, BTreeSet::new(), BTreeMap::new()),
                branch,
                call_node(
                    "yes",
                    &yes,
                    BTreeSet::from(["choose".into()]),
                    BTreeMap::new(),
                ),
                call_node(
                    "no",
                    &no,
                    BTreeSet::from(["choose".into()]),
                    BTreeMap::new(),
                ),
            ],
            streams: Vec::new(),
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        let capabilities = vec![probe.clone(), yes.clone(), no.clone()];
        let current_evidence_ids = capabilities
            .iter()
            .flat_map(|assessment| {
                assessment
                    .descriptor
                    .preconditions
                    .observed_state_fact_ids
                    .iter()
                    .copied()
            })
            .collect::<BTreeSet<_>>();
        let mut state = ProcedureRuntimeState::new(&procedure).unwrap();

        assert!(
            advance_procedure(
                &procedure,
                &capabilities,
                &BTreeSet::new(),
                &mut state,
                &BTreeMap::new(),
                4,
            )
            .is_err()
        );
        let mut untried = capabilities.clone();
        untried[0].evidence_state = CapabilityEvidenceState::PassivelyObserved;
        assert!(
            advance_procedure(
                &procedure,
                &untried,
                &current_evidence_ids,
                &mut state,
                &BTreeMap::new(),
                4,
            )
            .is_err()
        );

        let first = advance_procedure(
            &procedure,
            &capabilities,
            &current_evidence_ids,
            &mut state,
            &BTreeMap::new(),
            4,
        )
        .unwrap();
        assert_eq!(
            first
                .proposal_wave
                .iter()
                .map(|proposal| proposal.node_id.as_str())
                .collect::<Vec<_>>(),
            vec!["probe"]
        );
        assert!(first.blocked_nodes.iter().any(|blocked| {
            blocked.node_id == "choose"
                && blocked.reason == ProcedureBlockReason::DependenciesPending
        }));

        state
            .record_dispatched(
                &procedure,
                &capabilities,
                &current_evidence_ids,
                "probe",
                Uuid::new_v4(),
            )
            .unwrap();
        state
            .record_verified_success(
                &procedure,
                "probe",
                BTreeMap::from([("ready".into(), ProcedureValue::Boolean(true))]),
                Uuid::new_v4(),
            )
            .unwrap();
        let selected = advance_procedure(
            &procedure,
            &capabilities,
            &current_evidence_ids,
            &mut state,
            &BTreeMap::new(),
            4,
        )
        .unwrap();
        assert_eq!(
            selected.resolved_branches,
            BTreeMap::from([("choose".into(), true)])
        );
        assert_eq!(selected.newly_skipped_nodes, vec!["no"]);
        assert_eq!(
            selected
                .proposal_wave
                .iter()
                .map(|proposal| proposal.node_id.as_str())
                .collect::<Vec<_>>(),
            vec!["yes"]
        );
        assert!(!selected.completion_satisfied);

        state
            .record_dispatched(
                &procedure,
                &capabilities,
                &current_evidence_ids,
                "yes",
                Uuid::new_v4(),
            )
            .unwrap();
        state
            .record_verified_success(&procedure, "yes", BTreeMap::new(), Uuid::new_v4())
            .unwrap();
        let complete = advance_procedure(
            &procedure,
            &capabilities,
            &current_evidence_ids,
            &mut state,
            &BTreeMap::new(),
            4,
        )
        .unwrap();
        assert!(complete.completion_satisfied);
        assert!(complete.proposal_wave.is_empty());
    }

    #[test]
    fn procedure_runtime_refuses_result_consumption_before_verified_completion() {
        let executor = registered_executor();
        let producer = assessment(
            "source.read",
            Vec::new(),
            vec![port(
                "text",
                PortType::Text,
                128,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            Uuid::new_v4(),
        );
        let consumer = assessment(
            "document.process",
            vec![port(
                "text",
                PortType::Text,
                256,
                crate::contracts::Sensitivity::Private,
            )],
            Vec::new(),
            &executor,
            Uuid::new_v4(),
        );
        let procedure = ProcedureIr {
            schema_version: 1,
            id: "verified-binding".into(),
            nodes: vec![
                call_node("read", &producer, BTreeSet::new(), BTreeMap::new()),
                call_node(
                    "process",
                    &consumer,
                    BTreeSet::from(["read".into()]),
                    BTreeMap::from([(
                        "text".into(),
                        ValueBinding::Result {
                            producer: "read".into(),
                            output: "text".into(),
                        },
                    )]),
                ),
            ],
            streams: Vec::new(),
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        let capabilities = vec![producer, consumer];
        let current_evidence_ids = capabilities
            .iter()
            .flat_map(|assessment| {
                assessment
                    .descriptor
                    .preconditions
                    .observed_state_fact_ids
                    .iter()
                    .copied()
            })
            .collect::<BTreeSet<_>>();
        let mut state = ProcedureRuntimeState::new(&procedure).unwrap();
        assert!(
            state
                .record_dispatched(
                    &procedure,
                    &capabilities,
                    &current_evidence_ids,
                    "process",
                    Uuid::new_v4(),
                )
                .is_err()
        );
        state
            .record_dispatched(
                &procedure,
                &capabilities,
                &current_evidence_ids,
                "read",
                Uuid::new_v4(),
            )
            .unwrap();
        state
            .record_verified_success(
                &procedure,
                "read",
                BTreeMap::from([("text".into(), ProcedureValue::Text("verified".into()))]),
                Uuid::new_v4(),
            )
            .unwrap();
        let next = advance_procedure(
            &procedure,
            &capabilities,
            &current_evidence_ids,
            &mut state,
            &BTreeMap::new(),
            2,
        )
        .unwrap();
        assert_eq!(next.proposal_wave.len(), 1);
        assert!(matches!(
            next.proposal_wave[0].inputs.get("text"),
            Some(ResolvedProcedureInput::Value {
                value: ProcedureValue::Text(value),
                source_port,
                target_port,
            }) if value == "verified"
                && source_port.privacy == crate::contracts::Sensitivity::Private
                && target_port.max_bytes == 256
        ));
    }

    #[test]
    fn procedure_artifact_results_stay_task_bound_through_typed_inputs() {
        let executor = registered_executor();
        let producer = assessment(
            "export.render",
            Vec::new(),
            vec![port(
                "output",
                PortType::Bytes,
                4096,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            Uuid::new_v4(),
        );
        let consumer = assessment(
            "media.inspect",
            vec![port(
                "input",
                PortType::Bytes,
                8192,
                crate::contracts::Sensitivity::Private,
            )],
            Vec::new(),
            &executor,
            Uuid::new_v4(),
        );
        let procedure = ProcedureIr {
            schema_version: 1,
            id: "artifact-binding".into(),
            nodes: vec![
                call_node("render", &producer, BTreeSet::new(), BTreeMap::new()),
                call_node(
                    "inspect",
                    &consumer,
                    BTreeSet::from(["render".into()]),
                    BTreeMap::from([(
                        "input".into(),
                        ValueBinding::Result {
                            producer: "render".into(),
                            output: "output".into(),
                        },
                    )]),
                ),
            ],
            streams: Vec::new(),
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        let capabilities = vec![producer, consumer];
        let current_evidence_ids = capabilities
            .iter()
            .flat_map(|assessment| {
                assessment
                    .descriptor
                    .preconditions
                    .observed_state_fact_ids
                    .iter()
                    .copied()
            })
            .collect::<BTreeSet<_>>();
        let task_id = Uuid::new_v4();
        let artifact_id = Uuid::new_v4();
        let mut state = ProcedureRuntimeState::new_for_task(&procedure, task_id).unwrap();
        state
            .record_dispatched(
                &procedure,
                &capabilities,
                &current_evidence_ids,
                "render",
                Uuid::new_v4(),
            )
            .unwrap();

        let foreign_artifact = ProcedureOutput::Artifact(ProcedureArtifactRef {
            artifact_id,
            task_id: Uuid::new_v4(),
            sha256: "a".repeat(64),
            size_bytes: 256,
        });
        assert!(
            state
                .record_verified_outputs(
                    &procedure,
                    "render",
                    BTreeMap::from([("output".into(), foreign_artifact)]),
                    Uuid::new_v4(),
                )
                .is_err()
        );
        let oversized_artifact = ProcedureOutput::Artifact(ProcedureArtifactRef {
            artifact_id,
            task_id,
            sha256: "a".repeat(64),
            size_bytes: 4097,
        });
        assert!(
            state
                .record_verified_outputs(
                    &procedure,
                    "render",
                    BTreeMap::from([("output".into(), oversized_artifact)]),
                    Uuid::new_v4(),
                )
                .is_err()
        );

        state
            .record_verified_outputs(
                &procedure,
                "render",
                BTreeMap::from([(
                    "output".into(),
                    ProcedureOutput::Artifact(ProcedureArtifactRef {
                        artifact_id,
                        task_id,
                        sha256: "a".repeat(64),
                        size_bytes: 256,
                    }),
                )]),
                Uuid::new_v4(),
            )
            .unwrap();
        let next = advance_procedure(
            &procedure,
            &capabilities,
            &current_evidence_ids,
            &mut state,
            &BTreeMap::new(),
            2,
        )
        .unwrap();
        assert_eq!(next.proposal_wave.len(), 1);
        assert!(matches!(
            next.proposal_wave[0].inputs.get("input"),
            Some(ResolvedProcedureInput::Artifact {
                artifact_id: resolved_id,
                owner_task_id: Some(owner_task_id),
                sha256,
                size_bytes: Some(256),
                source_port,
                target_port,
            }) if resolved_id == &artifact_id.to_string()
                && *owner_task_id == task_id
                && sha256 == &"a".repeat(64)
                && source_port.value_type == PortType::Bytes
                && target_port.value_type == PortType::Bytes
        ));
    }

    #[test]
    fn procedure_branch_conditions_must_match_the_referenced_output_type() {
        let executor = registered_executor();
        let source = assessment(
            "source.read",
            Vec::new(),
            vec![port(
                "value",
                PortType::Text,
                64,
                crate::contracts::Sensitivity::Private,
            )],
            &executor,
            Uuid::new_v4(),
        );
        let sink = assessment(
            "sink.write",
            Vec::new(),
            Vec::new(),
            &executor,
            Uuid::new_v4(),
        );
        let procedure = ProcedureIr {
            schema_version: 1,
            id: "wrong-branch-type".into(),
            nodes: vec![
                call_node("source", &source, BTreeSet::new(), BTreeMap::new()),
                ProcedureNode {
                    id: "choose".into(),
                    depends_on: BTreeSet::from(["source".into()]),
                    outputs: BTreeMap::new(),
                    kind: ProcedureNodeKind::Branch {
                        condition: ProcedureCondition::BooleanEquals {
                            reference: ValueReference {
                                node_id: "source".into(),
                                output: "value".into(),
                            },
                            value: true,
                        },
                        when_true: "yes".into(),
                        when_false: "no".into(),
                    },
                },
                call_node(
                    "yes",
                    &sink,
                    BTreeSet::from(["choose".into()]),
                    BTreeMap::new(),
                ),
                call_node(
                    "no",
                    &sink,
                    BTreeSet::from(["choose".into()]),
                    BTreeMap::new(),
                ),
            ],
            streams: Vec::new(),
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        assert!(procedure.validate().is_err());
    }

    fn controller_fixture(evidence: Uuid, interface_fingerprint: &str) -> ControllerIr {
        ControllerIr {
            schema_version: 1,
            system_id: Uuid::new_v4(),
            system_fingerprint: "a".repeat(64),
            interface_fingerprint: interface_fingerprint.into(),
            steps: vec![ControllerStep {
                id: "set-brightness".into(),
                primitive: ControllerPrimitive::ResolveSemanticAnchor,
                anchor: SemanticAnchor {
                    id: "old-control-id".into(),
                    role: "slider".into(),
                    accessible_name: "Brightness".into(),
                    ancestor_names: vec!["Display".into(), "Video".into()],
                },
                capability_id: None,
                input_bindings: BTreeMap::new(),
                depends_on: BTreeSet::new(),
                outputs: BTreeMap::new(),
                preconditions: Vec::new(),
                expected_effect: "Brightness value changes".into(),
                verification: "Read the current value".into(),
                restoration: Some("Restore the prior value".into()),
                evidence_ids: vec![evidence],
            }],
        }
    }

    #[test]
    fn verified_linear_ui_procedure_compiles_to_an_unreviewed_controller_candidate() {
        let now = Utc::now();
        let system_id = Uuid::new_v4();
        let system_fingerprint = "a".repeat(64);
        let interface_fingerprint = "b".repeat(64);
        let control_id = "c".repeat(64);
        let anchor_evidence = Uuid::new_v4();
        let verification_evidence = Uuid::new_v4();
        let second_verification_evidence = Uuid::new_v4();
        let number_port = DataPort {
            name: "observed_value".into(),
            value_type: PortType::Number,
            max_bytes: 8,
            privacy: crate::contracts::Sensitivity::Private,
        };
        let input_port = DataPort {
            name: "value".into(),
            ..number_port.clone()
        };
        let mut interface_observation = ObservationEnvelope {
            id: anchor_evidence,
            system_id,
            session_id: None,
            worker_session: None,
            system_fingerprint: system_fingerprint.clone(),
            origin: crate::world_model::EvidenceOrigin::Application,
            privacy: crate::contracts::Sensitivity::Private,
            observed_at: now,
            facts: vec![
                crate::world_model::ObservedFact {
                    name: "application.control.ancestors".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Text("Display > Picture".into()),
                },
                crate::world_model::ObservedFact {
                    name: "application.control.enabled".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Boolean(true),
                },
                crate::world_model::ObservedFact {
                    name: "application.control.label".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Text("Brightness".into()),
                },
                crate::world_model::ObservedFact {
                    name: "application.control.role".into(),
                    subject: Some(control_id.clone()),
                    value: FactValue::Text("slider".into()),
                },
                crate::world_model::ObservedFact {
                    name: "application.interface_fingerprint".into(),
                    subject: None,
                    value: FactValue::Identifier(interface_fingerprint.clone()),
                },
            ],
        };
        interface_observation
            .facts
            .sort_by(|left, right| (&left.name, &left.subject).cmp(&(&right.name, &right.subject)));
        let verification_observation = ObservationEnvelope {
            id: verification_evidence,
            observed_at: now,
            ..interface_observation.clone()
        };
        let second_verification_observation = ObservationEnvelope {
            id: second_verification_evidence,
            observed_at: now,
            ..interface_observation.clone()
        };
        let capability = CapabilityAssessment {
            descriptor: CapabilityDescriptor {
                schema_version: 1,
                id: "ui.control.slider.brightness".into(),
                system_id,
                system_fingerprint: system_fingerprint.clone(),
                interface_control_id: Some(control_id),
                interface_probe_kind: Some(crate::world_model::ProbeKind::RestoreSliderValue),
                label: "Set Brightness".into(),
                input_ports: vec![input_port.clone()],
                output_ports: vec![number_port.clone()],
                preconditions: crate::world_model::Preconditions {
                    observed_state_fact_ids: vec![anchor_evidence],
                    description: "The current brightness control is available".into(),
                },
                effects: BTreeSet::from([Effect::ControlApplication]),
                verification: "Read back the resulting brightness".into(),
                restoration: Some("Restore the captured brightness value".into()),
                cancellation: "Settle an already dispatched control effect".into(),
                executor_id: Some("read_file".into()),
                evidence_ids: vec![anchor_evidence],
                updated_at: now,
            },
            evidence_state: CapabilityEvidenceState::ReversiblyExperimented,
        };
        let procedure = ProcedureIr {
            schema_version: 1,
            id: "set-brightness".into(),
            nodes: vec![
                ProcedureNode {
                    id: "set-brightness".into(),
                    depends_on: BTreeSet::new(),
                    outputs: BTreeMap::from([("observed_value".into(), number_port.clone())]),
                    kind: ProcedureNodeKind::CapabilityCall {
                        capability_id: capability.descriptor.id.clone(),
                        system_id,
                        system_fingerprint: system_fingerprint.clone(),
                        input_bindings: BTreeMap::from([(
                            "value".into(),
                            ValueBinding::Literal {
                                value: ProcedureValue::Number(0.7),
                                port: input_port,
                            },
                        )]),
                    },
                },
                ProcedureNode {
                    id: "use-verified-result".into(),
                    depends_on: BTreeSet::from(["set-brightness".into()]),
                    outputs: BTreeMap::from([("observed_value".into(), number_port)]),
                    kind: ProcedureNodeKind::CapabilityCall {
                        capability_id: capability.descriptor.id.clone(),
                        system_id,
                        system_fingerprint,
                        input_bindings: BTreeMap::from([(
                            "value".into(),
                            ValueBinding::Result {
                                producer: "set-brightness".into(),
                                output: "observed_value".into(),
                            },
                        )]),
                    },
                },
            ],
            streams: Vec::new(),
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        let mut runtime = ProcedureRuntimeState::new(&procedure).unwrap();
        runtime
            .record_dispatched(
                &procedure,
                std::slice::from_ref(&capability),
                &BTreeSet::from([anchor_evidence]),
                "set-brightness",
                Uuid::new_v4(),
            )
            .unwrap();
        runtime
            .record_verified_success(
                &procedure,
                "set-brightness",
                BTreeMap::from([("observed_value".into(), ProcedureValue::Number(0.7))]),
                verification_evidence,
            )
            .unwrap();
        runtime
            .record_dispatched(
                &procedure,
                std::slice::from_ref(&capability),
                &BTreeSet::from([anchor_evidence]),
                "use-verified-result",
                Uuid::new_v4(),
            )
            .unwrap();
        runtime
            .record_verified_success(
                &procedure,
                "use-verified-result",
                BTreeMap::from([("observed_value".into(), ProcedureValue::Number(0.7))]),
                second_verification_evidence,
            )
            .unwrap();

        let controller = compile_controller_from_verified_procedure(
            &procedure,
            &runtime,
            &[capability],
            &[
                interface_observation,
                verification_observation,
                second_verification_observation,
            ],
            now,
        )
        .unwrap();

        assert_eq!(controller.schema_version, 2);
        assert_eq!(controller.steps.len(), 2);
        assert_eq!(controller.steps[0].anchor.accessible_name, "Brightness");
        assert!(matches!(
            controller.steps[0].input_bindings["value"],
            ValueBinding::Literal {
                value: ProcedureValue::Number(0.7),
                ..
            }
        ));
        assert!(
            controller.steps[0]
                .evidence_ids
                .contains(&verification_evidence)
        );
        assert!(matches!(
            &controller.steps[1].input_bindings["value"],
            ValueBinding::Result { producer, output }
                if producer == "set-brightness" && output == "observed_value"
        ));
    }

    #[test]
    fn controller_compiler_refuses_incomplete_procedure_runs() {
        let system_id = Uuid::new_v4();
        let procedure = ProcedureIr {
            schema_version: 1,
            id: "not-finished".into(),
            nodes: vec![ProcedureNode {
                id: "read".into(),
                depends_on: BTreeSet::new(),
                outputs: BTreeMap::new(),
                kind: ProcedureNodeKind::CapabilityCall {
                    capability_id: "read".into(),
                    system_id,
                    system_fingerprint: "a".repeat(64),
                    input_bindings: BTreeMap::new(),
                },
            }],
            streams: Vec::new(),
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        let runtime = ProcedureRuntimeState::new(&procedure).unwrap();
        let result =
            compile_controller_from_verified_procedure(&procedure, &runtime, &[], &[], Utc::now());
        assert!(matches!(result, Err(CoreError::ExecutorUnavailable(_))));
    }

    fn current_controller_evidence(
        evidence_id: Uuid,
        system_id: Uuid,
        expires_at: DateTime<Utc>,
    ) -> BTreeMap<Uuid, CurrentControllerEvidence> {
        BTreeMap::from([(
            evidence_id,
            CurrentControllerEvidence {
                system_id,
                expires_at,
            },
        )])
    }

    fn controller_observation(
        system_id: Uuid,
        system_fingerprint: &str,
        interface_fingerprint: &str,
        controls: &[(&str, &str, &str, bool, Option<&str>)],
        observed_at: DateTime<Utc>,
    ) -> ObservationEnvelope {
        let mut facts = vec![crate::world_model::ObservedFact {
            name: "browser.interface_fingerprint".into(),
            subject: None,
            value: FactValue::Identifier(interface_fingerprint.into()),
        }];
        for (id, role, label, enabled, ancestors) in controls {
            let subject = Some((*id).into());
            facts.push(crate::world_model::ObservedFact {
                name: "browser.control.enabled".into(),
                subject: subject.clone(),
                value: FactValue::Boolean(*enabled),
            });
            facts.push(crate::world_model::ObservedFact {
                name: "browser.control.label".into(),
                subject: subject.clone(),
                value: FactValue::Text((*label).into()),
            });
            facts.push(crate::world_model::ObservedFact {
                name: "browser.control.role".into(),
                subject: subject.clone(),
                value: FactValue::Text((*role).into()),
            });
            if let Some(ancestors) = ancestors {
                facts.push(crate::world_model::ObservedFact {
                    name: "browser.control.ancestors".into(),
                    subject,
                    value: FactValue::Text((*ancestors).into()),
                });
            }
        }
        facts
            .sort_by(|left, right| (&left.name, &left.subject).cmp(&(&right.name, &right.subject)));
        ObservationEnvelope {
            id: Uuid::new_v4(),
            system_id,
            session_id: None,
            worker_session: None,
            system_fingerprint: system_fingerprint.into(),
            origin: crate::world_model::EvidenceOrigin::Browser,
            privacy: crate::contracts::Sensitivity::Private,
            observed_at,
            facts,
        }
    }

    #[test]
    fn controller_revalidation_semantically_rebinds_only_unique_fresh_anchors() {
        let now = Utc::now();
        let evidence = Uuid::new_v4();
        let controller = controller_fixture(evidence, &"b".repeat(64));
        let observation = controller_observation(
            controller.system_id,
            &controller.system_fingerprint,
            &"c".repeat(64),
            &[(
                "new-control-id",
                "slider",
                "Brightness",
                true,
                Some("Display > Video"),
            )],
            now,
        );
        let rebound = revalidate_controller(
            &controller,
            &observation,
            &current_controller_evidence(evidence, controller.system_id, now + Duration::hours(1)),
            now,
        )
        .expect("unique semantic anchor after interface update");
        assert!(rebound.interface_changed);
        assert_eq!(
            rebound.control_ids_by_step.get("set-brightness"),
            Some(&"new-control-id".to_owned())
        );

        let duplicate = controller_observation(
            controller.system_id,
            &controller.system_fingerprint,
            &"c".repeat(64),
            &[
                (
                    "new-control-id",
                    "slider",
                    "Brightness",
                    true,
                    Some("Display > Video"),
                ),
                (
                    "other-control-id",
                    "slider",
                    "Brightness",
                    true,
                    Some("Display > Video"),
                ),
            ],
            now,
        );
        assert!(
            revalidate_controller(
                &controller,
                &duplicate,
                &current_controller_evidence(
                    evidence,
                    controller.system_id,
                    now + Duration::hours(1)
                ),
                now,
            )
            .is_err()
        );

        assert!(revalidate_controller(&controller, &observation, &BTreeMap::new(), now,).is_err());
        assert!(
            revalidate_controller(
                &controller,
                &observation,
                &current_controller_evidence(evidence, Uuid::new_v4(), now + Duration::hours(1)),
                now,
            )
            .is_err()
        );
        assert!(
            revalidate_controller(
                &controller,
                &observation,
                &current_controller_evidence(evidence, controller.system_id, now),
                now,
            )
            .is_err()
        );
        let stale = controller_observation(
            controller.system_id,
            &controller.system_fingerprint,
            &"c".repeat(64),
            &[(
                "new-control-id",
                "slider",
                "Brightness",
                true,
                Some("Display > Video"),
            )],
            now - Duration::seconds(11),
        );
        assert!(
            revalidate_controller(
                &controller,
                &stale,
                &current_controller_evidence(
                    evidence,
                    controller.system_id,
                    now + Duration::hours(1)
                ),
                now,
            )
            .is_err()
        );
    }

    #[test]
    fn unchanged_controller_interface_requires_the_original_control_identity() {
        let now = Utc::now();
        let evidence = Uuid::new_v4();
        let fingerprint = "b".repeat(64);
        let controller = controller_fixture(evidence, &fingerprint);
        let changed_id = controller_observation(
            controller.system_id,
            &controller.system_fingerprint,
            &fingerprint,
            &[(
                "replaced-id",
                "slider",
                "Brightness",
                true,
                Some("Display > Video"),
            )],
            now,
        );
        assert!(
            revalidate_controller(
                &controller,
                &changed_id,
                &current_controller_evidence(
                    evidence,
                    controller.system_id,
                    now + Duration::hours(1)
                ),
                now,
            )
            .is_err()
        );

        let same_id = controller_observation(
            controller.system_id,
            &controller.system_fingerprint,
            &fingerprint,
            &[(
                "old-control-id",
                "slider",
                "Brightness",
                true,
                Some("Display > Video"),
            )],
            now,
        );
        let rebound = revalidate_controller(
            &controller,
            &same_id,
            &current_controller_evidence(evidence, controller.system_id, now + Duration::hours(1)),
            now,
        )
        .expect("stable control identity on unchanged interface");
        assert!(!rebound.interface_changed);
        assert_eq!(
            rebound.control_ids_by_step.get("set-brightness"),
            Some(&"old-control-id".to_owned())
        );
    }

    #[test]
    fn controller_rebinding_uses_passive_application_ancestor_evidence() {
        let now = Utc::now();
        let evidence = Uuid::new_v4();
        let controller = controller_fixture(evidence, &"b".repeat(64));
        let mut observation = controller_observation(
            controller.system_id,
            &controller.system_fingerprint,
            &"c".repeat(64),
            &[("new-control-id", "slider", "Brightness", true, None)],
            now,
        );
        observation.origin = crate::world_model::EvidenceOrigin::OperatingSystem;
        for fact in &mut observation.facts {
            if fact.subject.as_deref() == Some("new-control-id") {
                fact.name = fact
                    .name
                    .replace("browser.control.", "application.control.");
            }
        }
        observation.facts.push(crate::world_model::ObservedFact {
            name: "application.control.ancestors".into(),
            subject: Some("new-control-id".into()),
            value: FactValue::Text(serde_json::to_string(&["Display", "Video"]).unwrap()),
        });
        observation
            .facts
            .sort_by(|left, right| (&left.name, &left.subject).cmp(&(&right.name, &right.subject)));

        let rebound = revalidate_controller(
            &controller,
            &observation,
            &current_controller_evidence(evidence, controller.system_id, now + Duration::hours(1)),
            now,
        )
        .expect("a changed interface can use a unique, evidence-backed app ancestry path");
        assert!(rebound.interface_changed);
        assert_eq!(
            rebound.control_ids_by_step.get("set-brightness"),
            Some(&"new-control-id".to_owned())
        );
    }

    #[test]
    fn controller_records_require_review_revalidate_freshly_and_follow_system_forgetting() {
        let directory = tempdir().unwrap();
        let store =
            crate::storage::LocalStore::deferred(&directory.path().join("sage.db")).unwrap();
        store.unlock(&MemorySecretStore::default()).unwrap();
        let system = store
            .observe_system(crate::world_model::SystemDescriptor {
                id: Uuid::nil(),
                kind: crate::world_model::SystemKind::Application,
                key: "com.example.editor".into(),
                label: "Example Editor".into(),
                fingerprint: "a".repeat(64),
                revision: 0,
                updated_at: Utc::now(),
            })
            .unwrap();

        let observation = controller_observation(
            system.id,
            &system.fingerprint,
            &"b".repeat(64),
            &[(
                "old-control-id",
                "slider",
                "Brightness",
                true,
                Some("Display > Video"),
            )],
            Utc::now(),
        );
        store.record_world_observation(&observation).unwrap();
        let mut controller = controller_fixture(observation.id, &"b".repeat(64));
        controller.system_id = system.id;
        let draft = store.save_controller_draft(&controller).unwrap();
        assert_eq!(draft.status, ControllerStatus::Draft);
        assert_eq!(draft.revision, 1);
        assert_eq!(
            store
                .controllers_for_system(system.id)
                .unwrap()
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            [draft.id.as_str()]
        );

        let reviewed = store
            .review_controller(&draft.id, draft.revision, observation.id)
            .unwrap();
        assert_eq!(reviewed.status, ControllerStatus::Reviewed);
        assert_eq!(reviewed.revision, 2);
        assert_eq!(reviewed.reviewed_observation_id, Some(observation.id));
        let idempotent = store.save_controller_draft(&controller).unwrap();
        assert_eq!(idempotent.status, ControllerStatus::Reviewed);
        assert_eq!(idempotent.revision, reviewed.revision);

        let fresh_observation = controller_observation(
            system.id,
            &system.fingerprint,
            &"c".repeat(64),
            &[(
                "new-control-id",
                "slider",
                "Brightness",
                true,
                Some("Display > Video"),
            )],
            Utc::now(),
        );
        store.record_world_observation(&fresh_observation).unwrap();
        let rebound = store
            .revalidate_stored_controller(&draft.id, fresh_observation.id, Utc::now())
            .unwrap();
        assert!(rebound.rebinding.interface_changed);
        assert_eq!(
            rebound.rebinding.control_ids_by_step.get("set-brightness"),
            Some(&"new-control-id".to_owned())
        );

        let changed_system = store
            .observe_system(crate::world_model::SystemDescriptor {
                id: Uuid::nil(),
                kind: crate::world_model::SystemKind::Application,
                key: "com.example.editor".into(),
                label: "Example Editor".into(),
                fingerprint: "d".repeat(64),
                revision: 0,
                updated_at: Utc::now(),
            })
            .unwrap();
        assert_eq!(changed_system.id, system.id);
        let invalidated = store.load_controller(&draft.id).unwrap().unwrap();
        assert_eq!(invalidated.status, ControllerStatus::Invalidated);
        assert!(
            store
                .revalidate_stored_controller(&draft.id, fresh_observation.id, Utc::now())
                .is_err()
        );

        assert!(store.forget_world_system(system.id).unwrap());
        assert!(store.load_controller(&draft.id).unwrap().is_none());
    }

    #[test]
    fn controller_review_rejects_ambiguous_semantic_anchors_without_activating_draft() {
        let directory = tempdir().unwrap();
        let store =
            crate::storage::LocalStore::deferred(&directory.path().join("sage.db")).unwrap();
        store.unlock(&MemorySecretStore::default()).unwrap();
        let system = store
            .observe_system(crate::world_model::SystemDescriptor {
                id: Uuid::nil(),
                kind: crate::world_model::SystemKind::Application,
                key: "com.example.editor".into(),
                label: "Example Editor".into(),
                fingerprint: "a".repeat(64),
                revision: 0,
                updated_at: Utc::now(),
            })
            .unwrap();
        let evidence = controller_observation(
            system.id,
            &system.fingerprint,
            &"b".repeat(64),
            &[(
                "old-control-id",
                "slider",
                "Brightness",
                true,
                Some("Display > Video"),
            )],
            Utc::now(),
        );
        store.record_world_observation(&evidence).unwrap();
        let mut controller = controller_fixture(evidence.id, &"b".repeat(64));
        controller.system_id = system.id;
        let draft = store.save_controller_draft(&controller).unwrap();
        let ambiguous = controller_observation(
            system.id,
            &system.fingerprint,
            &"c".repeat(64),
            &[
                (
                    "control-one",
                    "slider",
                    "Brightness",
                    true,
                    Some("Display > Video"),
                ),
                (
                    "control-two",
                    "slider",
                    "Brightness",
                    true,
                    Some("Display > Video"),
                ),
            ],
            Utc::now(),
        );
        store.record_world_observation(&ambiguous).unwrap();

        assert!(
            store
                .review_controller(&draft.id, draft.revision, ambiguous.id)
                .is_err()
        );
        let still_draft = store.load_controller(&draft.id).unwrap().unwrap();
        assert_eq!(still_draft.status, ControllerStatus::Draft);
        assert_eq!(still_draft.revision, draft.revision);
        assert_eq!(still_draft.reviewed_observation_id, None);
    }
}
