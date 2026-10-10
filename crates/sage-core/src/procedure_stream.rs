//! Task-local bounded streams and validated endpoint bundles for ProcedureIR.
//!
//! A stream carries data under a validated producer/consumer port contract; it
//! never carries a capability grant or execution authority. Payloads are
//! ephemeral and zeroized when dropped. A consuming runner must still resolve
//! current targets, apply policy and permissions, obtain an expiring capability,
//! execute, and verify each effect through Sage's ordinary authority path.

use tokio::sync::{mpsc, watch};
use uuid::Uuid;
use zeroize::Zeroizing;

use crate::agency::{
    MAX_PROCEDURE_STREAM_ITEM_BYTES, ProcedureIr, ProcedureNodeKind, StreamChannel,
};
use crate::error::{CoreError, CoreResult};
use crate::world_model::{DataPort, PortType};

/// A non-cloneable owner for all queues declared by one procedure. Creating
/// the pool validates current capability port contracts and the aggregate
/// budget; each channel can be opened once, preventing accidental duplicate
/// queues or caller-supplied port substitutions from bypassing those checks.
pub struct ProcedureStreamPool {
    task_id: Uuid,
    channels: std::collections::BTreeMap<String, DeclaredStreamBinding>,
    opened: std::collections::BTreeSet<String>,
}

/// Endpoints owned by one capability node. They transport only the stream
/// payloads declared by that node's validated input and output ports.
#[derive(Default)]
pub struct ProcedureNodeStreams {
    inputs: std::collections::BTreeMap<String, ProcedureStreamReceiver>,
    outputs: std::collections::BTreeMap<String, ProcedureStreamSender>,
    expected_inputs: std::collections::BTreeMap<String, String>,
    expected_outputs: std::collections::BTreeMap<String, String>,
}

impl ProcedureNodeStreams {
    /// True when this capability call has no declared stream endpoints. An
    /// executor's compatibility path may ignore an empty endpoint bundle, but
    /// must explicitly opt in before consuming any stream data.
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.outputs.is_empty()
    }

    pub fn inputs_mut(
        &mut self,
    ) -> &mut std::collections::BTreeMap<String, ProcedureStreamReceiver> {
        &mut self.inputs
    }

    pub fn outputs_mut(
        &mut self,
    ) -> &mut std::collections::BTreeMap<String, ProcedureStreamSender> {
        &mut self.outputs
    }

    pub(crate) fn validate_terminal_frames(&self) -> CoreResult<()> {
        if self.inputs.len() != self.expected_inputs.len()
            || self.outputs.len() != self.expected_outputs.len()
        {
            return Err(CoreError::VerificationFailed(
                "Stream executor did not retain every declared endpoint".into(),
            ));
        }
        for (port, channel_id) in &self.expected_inputs {
            let receiver = self.inputs.get(port).ok_or_else(|| {
                CoreError::VerificationFailed("Declared stream input disappeared".into())
            })?;
            if receiver.binding.channel.id != *channel_id || !receiver.finished {
                return Err(CoreError::VerificationFailed(
                    "Stream executor returned before consuming a valid terminal frame".into(),
                ));
            }
        }
        for (port, channel_id) in &self.expected_outputs {
            let sender = self.outputs.get(port).ok_or_else(|| {
                CoreError::VerificationFailed("Declared stream output disappeared".into())
            })?;
            if sender.binding.channel.id != *channel_id || !sender.finished {
                return Err(CoreError::VerificationFailed(
                    "Stream executor returned before sending a valid terminal frame".into(),
                ));
            }
        }
        Ok(())
    }
}

/// One-shot endpoint set opened for all channels of one task-bound procedure.
/// Callers take the bundle for a node before starting its registered executor.
pub struct ProcedureStreamEndpoints {
    nodes: std::collections::BTreeMap<String, ProcedureNodeStreams>,
}

impl ProcedureStreamEndpoints {
    pub fn take_node(&mut self, node_id: &str) -> Option<ProcedureNodeStreams> {
        self.nodes.remove(node_id)
    }

    pub fn remaining_node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// One-way cancellation signal shared by all channels in a procedure run.
pub struct ProcedureStreamCancellation {
    sender: watch::Sender<bool>,
}

/// Receiver side of a procedure cancellation signal. Cloning it observes the
/// same terminal cancellation event without exposing a way to clear that event.
#[derive(Clone)]
pub struct ProcedureStreamCancellationReceiver {
    receiver: watch::Receiver<bool>,
}

impl ProcedureStreamCancellation {
    pub fn new() -> (Self, ProcedureStreamCancellationReceiver) {
        let (sender, receiver) = watch::channel(false);
        (
            Self { sender },
            ProcedureStreamCancellationReceiver { receiver },
        )
    }

    pub fn cancel(&self) {
        self.sender.send_replace(true);
    }
}

impl ProcedureStreamPool {
    pub fn new(
        procedure: &ProcedureIr,
        capabilities: &[crate::world_model::CapabilityDescriptor],
        task_id: Uuid,
    ) -> CoreResult<Self> {
        if task_id.is_nil() {
            return Err(CoreError::InvalidAction(
                "Procedure streams require a non-nil owning task".into(),
            ));
        }
        procedure.validate_against(capabilities)?;
        let capabilities_by_id = capabilities
            .iter()
            .map(|capability| (capability.id.as_str(), capability))
            .collect::<std::collections::BTreeMap<_, _>>();
        if capabilities_by_id.len() != capabilities.len() {
            return Err(CoreError::InvalidAction(
                "Procedure stream capabilities are ambiguous".into(),
            ));
        }
        let nodes = procedure
            .nodes
            .iter()
            .map(|node| (node.id.as_str(), node))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut declared_channels = std::collections::BTreeMap::new();
        for channel in &procedure.streams {
            let producer = nodes[channel.producer_node.as_str()];
            let consumer = nodes[channel.consumer_node.as_str()];
            let ProcedureNodeKind::CapabilityCall {
                capability_id: producer_capability_id,
                ..
            } = &producer.kind
            else {
                return Err(CoreError::InvalidAction(
                    "Procedure stream producer must be a capability call".into(),
                ));
            };
            let ProcedureNodeKind::CapabilityCall {
                capability_id: consumer_capability_id,
                ..
            } = &consumer.kind
            else {
                return Err(CoreError::InvalidAction(
                    "Procedure stream consumer must be a capability call".into(),
                ));
            };
            let producer_capability = capabilities_by_id
                .get(producer_capability_id.as_str())
                .ok_or_else(|| {
                    CoreError::ExecutorUnavailable(
                        "Stream producer capability is unavailable".into(),
                    )
                })?;
            let producer_port = producer_capability
                .output_ports
                .iter()
                .find(|port| port.name == channel.producer_output)
                .ok_or_else(|| {
                    CoreError::InvalidAction("Stream producer output is not declared".into())
                })?
                .clone();
            let consumer_capability = capabilities_by_id
                .get(consumer_capability_id.as_str())
                .ok_or_else(|| {
                    CoreError::ExecutorUnavailable(
                        "Stream consumer capability is unavailable".into(),
                    )
                })?;
            let consumer_port = consumer_capability
                .input_ports
                .iter()
                .find(|port| port.name == channel.consumer_input)
                .ok_or_else(|| {
                    CoreError::InvalidAction("Stream consumer input is not declared".into())
                })?
                .clone();
            channel.validate_ports(&producer_port, &consumer_port)?;
            declared_channels.insert(
                channel.id.clone(),
                DeclaredStreamBinding {
                    channel: channel.clone(),
                    producer_port,
                    consumer_port,
                },
            );
        }
        Ok(Self {
            task_id,
            channels: declared_channels,
            opened: std::collections::BTreeSet::new(),
        })
    }

    /// Open every declared channel exactly once and return endpoints grouped
    /// by the validated producer and consumer node/port identities. A caller
    /// receives no partial endpoint set if cancellation or an invariant error
    /// occurs while opening the cohort; locally opened queues are dropped.
    pub fn open_all(
        &mut self,
        mut cancelled: ProcedureStreamCancellationReceiver,
    ) -> CoreResult<ProcedureStreamEndpoints> {
        if self.channels.is_empty() {
            return Err(CoreError::InvalidAction(
                "Procedure does not declare any live stream channels".into(),
            ));
        }
        if !self.opened.is_empty() {
            return Err(CoreError::InvalidAction(
                "Procedure stream cohort may be opened only once".into(),
            ));
        }
        if *cancelled.receiver.borrow_and_update() || cancelled.receiver.has_changed().is_err() {
            return Err(CoreError::Cancelled);
        }
        let cancellation = cancelled.receiver;
        let channel_ids = self.channels.keys().cloned().collect::<Vec<_>>();
        let mut nodes = std::collections::BTreeMap::<String, ProcedureNodeStreams>::new();
        for channel_id in &channel_ids {
            let declared = self.channels.get(channel_id).ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Validated procedure stream disappeared while opening its cohort".into(),
                )
            })?;
            let channel = declared.channel.clone();
            let (sender, receiver) = mpsc::channel(usize::from(channel.capacity_items));
            let binding = StreamBinding {
                channel: channel.clone(),
                task_id: self.task_id,
                producer_port: declared.producer_port.clone(),
                consumer_port: declared.consumer_port.clone(),
                maximum_total_bytes: declared
                    .producer_port
                    .max_bytes
                    .min(declared.consumer_port.max_bytes),
            };
            let sender = ProcedureStreamSender {
                sender,
                cancelled: cancellation.clone(),
                binding: binding.clone(),
                next_sequence: 0,
                total_bytes: 0,
                finished: false,
                failed: false,
            };
            let receiver = ProcedureStreamReceiver {
                receiver,
                cancelled: cancellation.clone(),
                binding,
                next_sequence: 0,
                total_bytes: 0,
                finished: false,
            };
            {
                let producer = nodes.entry(channel.producer_node).or_default();
                if producer
                    .outputs
                    .insert(channel.producer_output.clone(), sender)
                    .is_some()
                {
                    return Err(CoreError::InvalidAction(
                        "Procedure stream output is bound more than once".into(),
                    ));
                }
                producer
                    .expected_outputs
                    .insert(channel.producer_output, declared.channel.id.clone());
            }
            {
                let consumer = nodes.entry(channel.consumer_node).or_default();
                if consumer
                    .inputs
                    .insert(channel.consumer_input.clone(), receiver)
                    .is_some()
                {
                    return Err(CoreError::InvalidAction(
                        "Procedure stream input is bound more than once".into(),
                    ));
                }
                consumer
                    .expected_inputs
                    .insert(channel.consumer_input, declared.channel.id.clone());
            }
        }
        // Mark the pool consumed only after every local endpoint has been
        // constructed and grouped successfully. On failure, the local bundles
        // drop and a caller may retry with a fresh cancellation signal.
        self.opened.extend(channel_ids);
        Ok(ProcedureStreamEndpoints { nodes })
    }
}

#[derive(Clone)]
struct DeclaredStreamBinding {
    channel: StreamChannel,
    producer_port: DataPort,
    consumer_port: DataPort,
}

#[derive(Clone)]
struct StreamBinding {
    channel: StreamChannel,
    task_id: Uuid,
    producer_port: DataPort,
    consumer_port: DataPort,
    maximum_total_bytes: u64,
}

/// The unique producer endpoint. It is deliberately not `Clone`, so the
/// runtime has one sequence owner and at most one item waiting outside the
/// declared queue due to backpressure.
pub struct ProcedureStreamSender {
    sender: mpsc::Sender<ProcedureStreamFrame>,
    cancelled: watch::Receiver<bool>,
    binding: StreamBinding,
    next_sequence: u64,
    total_bytes: u64,
    finished: bool,
    failed: bool,
}

impl ProcedureStreamSender {
    pub fn channel_id(&self) -> &str {
        &self.binding.channel.id
    }

    pub fn producer_node(&self) -> &str {
        &self.binding.channel.producer_node
    }

    pub fn consumer_node(&self) -> &str {
        &self.binding.channel.consumer_node
    }

    pub fn output_port(&self) -> &DataPort {
        &self.binding.producer_port
    }

    pub fn maximum_total_bytes(&self) -> u64 {
        self.binding.maximum_total_bytes
    }

    pub fn maximum_item_bytes(&self) -> u64 {
        self.binding.channel.maximum_item_bytes
    }

    /// Send one typed item. A full queue blocks the producer; cancellation or
    /// receiver failure settles the send without dispatching later items.
    pub async fn send(&mut self, bytes: Vec<u8>) -> CoreResult<()> {
        if self.finished || self.failed {
            return Err(CoreError::InvalidAction(
                "Procedure stream is already terminal".into(),
            ));
        }
        let payload = Zeroizing::new(bytes);
        validate_payload(
            &payload,
            &self.binding.producer_port,
            self.binding.channel.maximum_item_bytes,
        )?;
        if *self.cancelled.borrow() {
            return Err(CoreError::Cancelled);
        }
        if self.next_sequence == u64::MAX {
            return Err(CoreError::InvalidAction(
                "Procedure stream sequence limit was reached".into(),
            ));
        }
        let Some(next_total_bytes) = self
            .total_bytes
            .checked_add(payload.len() as u64)
            .filter(|total| *total <= self.binding.maximum_total_bytes)
        else {
            // A rejected over-limit item makes the stream incomplete. The
            // producer cannot later publish a successful end marker for a
            // truncated result.
            self.failed = true;
            return Err(CoreError::InvalidAction(
                "Procedure stream exceeded its cumulative data-port limit".into(),
            ));
        };
        let item = ProcedureStreamItem {
            channel_id: self.binding.channel.id.clone(),
            task_id: self.binding.task_id,
            producer_node: self.binding.channel.producer_node.clone(),
            producer_output: self.binding.channel.producer_output.clone(),
            port: self.binding.producer_port.clone(),
            sequence: self.next_sequence,
            payload,
        };
        let frame = ProcedureStreamFrame::Item(item);
        let result = tokio::select! {
            biased;
            changed = self.cancelled.changed() => {
                let _ = changed;
                return Err(CoreError::Cancelled);
            }
            result = self.sender.send(frame) => result,
        };
        result.map_err(|_| {
            CoreError::ExecutorUnavailable("Procedure stream receiver has closed".into())
        })?;
        self.next_sequence += 1;
        self.total_bytes = next_total_bytes;
        Ok(())
    }

    /// End the stream successfully after every item has been sent. Dropping a
    /// sender without this explicit terminal frame is treated as incomplete by
    /// the receiver.
    pub async fn finish(&mut self) -> CoreResult<()> {
        if self.finished || self.failed {
            return Err(CoreError::InvalidAction(
                "Procedure stream cannot be completed after a terminal error".into(),
            ));
        }
        if *self.cancelled.borrow() {
            return Err(CoreError::Cancelled);
        }
        if self.next_sequence == u64::MAX {
            return Err(CoreError::InvalidAction(
                "Procedure stream sequence limit was reached".into(),
            ));
        }
        let frame = ProcedureStreamFrame::End {
            channel_id: self.binding.channel.id.clone(),
            task_id: self.binding.task_id,
            producer_node: self.binding.channel.producer_node.clone(),
            producer_output: self.binding.channel.producer_output.clone(),
            sequence: self.next_sequence,
        };
        let result = tokio::select! {
            biased;
            changed = self.cancelled.changed() => {
                let _ = changed;
                return Err(CoreError::Cancelled);
            }
            result = self.sender.send(frame) => result,
        };
        result.map_err(|_| {
            CoreError::ExecutorUnavailable("Procedure stream receiver has closed".into())
        })?;
        self.finished = true;
        Ok(())
    }
}

/// One ephemeral stream item. Its bytes are accessible only as a borrowed
/// slice unless the consumer explicitly takes ownership.
pub struct ProcedureStreamItem {
    channel_id: String,
    task_id: Uuid,
    producer_node: String,
    producer_output: String,
    port: DataPort,
    sequence: u64,
    payload: Zeroizing<Vec<u8>>,
}

/// Explicit terminal framing prevents an executor crash or dropped producer
/// from being mistaken for a successfully completed data stream.
enum ProcedureStreamFrame {
    Item(ProcedureStreamItem),
    End {
        channel_id: String,
        task_id: Uuid,
        producer_node: String,
        producer_output: String,
        sequence: u64,
    },
}

impl ProcedureStreamItem {
    pub fn channel_id(&self) -> &str {
        &self.channel_id
    }

    pub fn task_id(&self) -> Uuid {
        self.task_id
    }

    pub fn producer_node(&self) -> &str {
        &self.producer_node
    }

    pub fn producer_output(&self) -> &str {
        &self.producer_output
    }

    pub fn port(&self) -> &DataPort {
        &self.port
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.payload.as_slice()
    }

    /// Transfer this payload while retaining zeroization on drop. Consumers
    /// that copy the bytes must separately manage the copied allocation.
    pub fn into_bytes(self) -> Zeroizing<Vec<u8>> {
        self.payload
    }
}

/// The unique consumer endpoint. Every received item is rechecked against the
/// bound task, channel, source port, size, type, and monotone sequence.
pub struct ProcedureStreamReceiver {
    receiver: mpsc::Receiver<ProcedureStreamFrame>,
    cancelled: watch::Receiver<bool>,
    binding: StreamBinding,
    next_sequence: u64,
    total_bytes: u64,
    finished: bool,
}

impl ProcedureStreamReceiver {
    pub fn channel_id(&self) -> &str {
        &self.binding.channel.id
    }

    pub fn producer_node(&self) -> &str {
        &self.binding.channel.producer_node
    }

    pub fn input_port(&self) -> &DataPort {
        &self.binding.consumer_port
    }

    pub fn maximum_total_bytes(&self) -> u64 {
        self.binding.maximum_total_bytes
    }

    pub async fn recv(&mut self) -> CoreResult<Option<ProcedureStreamItem>> {
        if self.finished {
            return Ok(None);
        }
        if *self.cancelled.borrow() {
            self.cancel_and_clear();
            return Err(CoreError::Cancelled);
        }
        let frame = tokio::select! {
            biased;
            changed = self.cancelled.changed() => {
                let _ = changed;
                self.cancel_and_clear();
                return Err(CoreError::Cancelled);
            }
            item = self.receiver.recv() => item,
        };
        let Some(frame) = frame else {
            self.cancel_and_clear();
            return Err(CoreError::VerificationFailed(
                "Procedure stream producer ended without a terminal frame".into(),
            ));
        };
        let item = match frame {
            ProcedureStreamFrame::Item(item) => item,
            ProcedureStreamFrame::End {
                channel_id,
                task_id,
                producer_node,
                producer_output,
                sequence,
            } => {
                if channel_id != self.binding.channel.id
                    || task_id != self.binding.task_id
                    || producer_node != self.binding.channel.producer_node
                    || producer_output != self.binding.channel.producer_output
                    || sequence != self.next_sequence
                {
                    self.cancel_and_clear();
                    return Err(CoreError::VerificationFailed(
                        "Procedure stream terminal frame violated its bound channel contract"
                            .into(),
                    ));
                }
                self.next_sequence = self.next_sequence.checked_add(1).ok_or_else(|| {
                    CoreError::VerificationFailed("Procedure stream sequence overflowed".into())
                })?;
                self.finished = true;
                return Ok(None);
            }
        };
        if item.channel_id != self.binding.channel.id
            || item.task_id != self.binding.task_id
            || item.producer_node != self.binding.channel.producer_node
            || item.producer_output != self.binding.channel.producer_output
            || item.port != self.binding.producer_port
            || self.binding.consumer_port.name != self.binding.channel.consumer_input
            || item.sequence != self.next_sequence
            || validate_payload(
                &item.payload,
                &self.binding.producer_port,
                self.binding.channel.maximum_item_bytes,
            )
            .is_err()
        {
            self.cancel_and_clear();
            return Err(CoreError::VerificationFailed(
                "Procedure stream item violated its bound channel contract".into(),
            ));
        }
        let Some(next_total_bytes) = self
            .total_bytes
            .checked_add(item.payload.len() as u64)
            .filter(|total| *total <= self.binding.maximum_total_bytes)
        else {
            self.cancel_and_clear();
            return Err(CoreError::VerificationFailed(
                "Procedure stream exceeded its cumulative data-port limit".into(),
            ));
        };
        self.next_sequence = self.next_sequence.checked_add(1).ok_or_else(|| {
            CoreError::VerificationFailed("Procedure stream sequence overflowed".into())
        })?;
        self.total_bytes = next_total_bytes;
        Ok(Some(item))
    }

    fn cancel_and_clear(&mut self) {
        self.receiver.close();
        while let Ok(item) = self.receiver.try_recv() {
            drop(item);
        }
    }
}

impl Drop for ProcedureStreamReceiver {
    fn drop(&mut self) {
        self.cancel_and_clear();
    }
}

fn validate_payload(bytes: &[u8], port: &DataPort, maximum_item_bytes: u64) -> CoreResult<()> {
    if bytes.len() as u64 > maximum_item_bytes
        || bytes.len() as u64 > port.max_bytes
        || bytes.len() as u64 > MAX_PROCEDURE_STREAM_ITEM_BYTES
    {
        return Err(CoreError::InvalidAction(
            "Procedure stream item exceeds its declared byte bound".into(),
        ));
    }
    match port.value_type {
        PortType::Text => {
            let value = std::str::from_utf8(bytes).map_err(|_| {
                CoreError::InvalidAction("Procedure text stream item is not UTF-8".into())
            })?;
            if value.contains('\0') || crate::redaction::redact_for_persistence(value) != value {
                return Err(CoreError::PolicyDenied(
                    "Procedure text stream item contains a credential-like value".into(),
                ));
            }
        }
        PortType::Boolean if bytes.len() != 1 || bytes[0] > 1 => {
            return Err(CoreError::InvalidAction(
                "Boolean stream items must be one byte containing zero or one".into(),
            ));
        }
        PortType::Number => {
            let value = <[u8; 8]>::try_from(bytes)
                .ok()
                .map(f64::from_le_bytes)
                .filter(|value| value.is_finite());
            if value.is_none() {
                return Err(CoreError::InvalidAction(
                    "Numeric stream items must be finite little-endian f64 values".into(),
                ));
            }
        }
        PortType::StructuredData => {
            serde_json::from_slice::<serde_json::Value>(bytes).map_err(|_| {
                CoreError::InvalidAction("Structured stream item must contain valid JSON".into())
            })?;
        }
        PortType::Boolean
        | PortType::Bytes
        | PortType::Image
        | PortType::Video
        | PortType::Audio
        | PortType::DeviceStream => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::{ProcedureNodeStreams, ProcedureStreamPool};
    use crate::agency::{
        CompletionCondition, ProcedureIr, ProcedureNode, ProcedureNodeKind, StreamBackpressure,
        StreamChannel, ValueBinding,
    };
    use crate::capability::{CapabilityBroker, CapabilityGrant};
    use crate::compiler::{CompiledAction, ImplementationCandidate, InteractionTier};
    use crate::contracts::{Effect, Sensitivity};
    use crate::domain::{Action, ActionProposal, ExecutionDomain, ExpectedOutcome, Provenance};
    use crate::error::CoreError;
    use crate::execution::{ExecutionBroker, ExecutionReceipt, Executor};
    use crate::world_model::{CapabilityDescriptor, DataPort, PortType, Preconditions};
    use uuid::Uuid;

    fn port(name: &str, value_type: PortType, max_bytes: u64) -> DataPort {
        DataPort {
            name: name.into(),
            value_type,
            max_bytes,
            privacy: Sensitivity::Private,
        }
    }

    fn channel(capacity_items: u16, maximum_item_bytes: u64) -> StreamChannel {
        StreamChannel {
            id: "text-stream".into(),
            producer_node: "producer".into(),
            producer_output: "chunk".into(),
            consumer_node: "consumer".into(),
            consumer_input: "chunk".into(),
            capacity_items,
            maximum_item_bytes,
            backpressure: StreamBackpressure::BlockProducer,
        }
    }

    fn capability(
        id: &str,
        system_id: Uuid,
        system_fingerprint: &str,
        input_ports: Vec<DataPort>,
        output_ports: Vec<DataPort>,
    ) -> CapabilityDescriptor {
        let evidence_id = Uuid::new_v4();
        CapabilityDescriptor {
            schema_version: 1,
            id: id.into(),
            system_id,
            system_fingerprint: system_fingerprint.into(),
            interface_control_id: None,
            interface_probe_kind: None,
            label: id.into(),
            input_ports,
            output_ports,
            preconditions: Preconditions {
                observed_state_fact_ids: vec![evidence_id],
                description: "Current state".into(),
            },
            effects: std::collections::BTreeSet::from([Effect::Read]),
            verification: "Check output".into(),
            restoration: None,
            cancellation: "Stop and settle".into(),
            executor_id: None,
            evidence_ids: vec![evidence_id],
            updated_at: chrono::Utc::now(),
        }
    }

    fn procedure(
        channel: StreamChannel,
        producer_port: DataPort,
        consumer_port: DataPort,
    ) -> (ProcedureIr, Vec<CapabilityDescriptor>) {
        let producer_system_id = Uuid::new_v4();
        let consumer_system_id = Uuid::new_v4();
        let producer_fingerprint = "a".repeat(64);
        let consumer_fingerprint = "b".repeat(64);
        let producer_capability = capability(
            "test.producer",
            producer_system_id,
            &producer_fingerprint,
            Vec::new(),
            vec![producer_port.clone()],
        );
        let consumer_capability = capability(
            "test.consumer",
            consumer_system_id,
            &consumer_fingerprint,
            vec![consumer_port],
            Vec::new(),
        );
        let producer = ProcedureNode {
            controller_binding: None,
            id: "producer".into(),
            depends_on: Default::default(),
            outputs: std::collections::BTreeMap::from([(
                producer_port.name.clone(),
                producer_port,
            )]),
            kind: ProcedureNodeKind::CapabilityCall {
                capability_id: "test.producer".into(),
                system_id: producer_system_id,
                system_fingerprint: producer_fingerprint,
                input_bindings: Default::default(),
            },
        };
        let consumer = ProcedureNode {
            controller_binding: None,
            id: "consumer".into(),
            depends_on: Default::default(),
            outputs: Default::default(),
            kind: ProcedureNodeKind::CapabilityCall {
                capability_id: "test.consumer".into(),
                system_id: consumer_system_id,
                system_fingerprint: consumer_fingerprint,
                input_bindings: std::collections::BTreeMap::from([(
                    "chunk".into(),
                    ValueBinding::Stream {
                        channel_id: channel.id.clone(),
                    },
                )]),
            },
        };
        (
            ProcedureIr {
                schema_version: 2,
                id: "test-stream-procedure".into(),
                nodes: vec![producer, consumer],
                streams: vec![channel],
                completion: vec![CompletionCondition::AllNodesSucceeded],
            },
            vec![producer_capability, consumer_capability],
        )
    }

    fn open(
        port_type: PortType,
        capacity_items: u16,
        maximum_item_bytes: u64,
    ) -> (
        super::ProcedureStreamSender,
        super::ProcedureStreamReceiver,
        super::ProcedureStreamCancellation,
    ) {
        open_with_total_limit(
            port_type,
            capacity_items,
            maximum_item_bytes,
            maximum_item_bytes,
        )
    }

    fn open_with_total_limit(
        port_type: PortType,
        capacity_items: u16,
        maximum_item_bytes: u64,
        maximum_total_bytes: u64,
    ) -> (
        super::ProcedureStreamSender,
        super::ProcedureStreamReceiver,
        super::ProcedureStreamCancellation,
    ) {
        let (cancel, receiver) = super::ProcedureStreamCancellation::new();
        let producer = port("chunk", port_type, maximum_total_bytes);
        let consumer = producer.clone();
        let stream = channel(capacity_items, maximum_item_bytes);
        let (procedure, capabilities) = procedure(stream.clone(), producer, consumer);
        let mut pool = ProcedureStreamPool::new(&procedure, &capabilities, Uuid::new_v4())
            .expect("valid procedure stream pool");
        let mut endpoints = pool.open_all(receiver).expect("valid stream cohort");
        let sender = endpoints
            .take_node("producer")
            .expect("producer node")
            .outputs_mut()
            .remove("chunk")
            .expect("producer output");
        let receiver = endpoints
            .take_node("consumer")
            .expect("consumer node")
            .inputs_mut()
            .remove("chunk")
            .expect("consumer input");
        (sender, receiver, cancel)
    }

    fn expect_item(
        result: crate::CoreResult<Option<super::ProcedureStreamItem>>,
    ) -> super::ProcedureStreamItem {
        match result {
            Ok(Some(item)) => item,
            Ok(None) => panic!("stream ended before the expected item"),
            Err(_) => panic!("stream receive failed"),
        }
    }

    #[tokio::test]
    async fn queue_capacity_applies_backpressure_until_an_item_is_consumed() {
        let (mut sender, mut receiver, _cancel) = open(PortType::Text, 1, 16);
        sender.send(b"first".to_vec()).await.expect("first item");
        let mut second = Box::pin(sender.send(b"second".to_vec()));
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut second)
                .await
                .is_err()
        );
        let first = expect_item(receiver.recv().await);
        assert_eq!(first.sequence(), 0);
        assert_eq!(first.as_bytes(), b"first");
        second.await.expect("second item");
        let second = expect_item(receiver.recv().await);
        assert_eq!(second.sequence(), 1);
        assert_eq!(second.as_bytes(), b"second");
        sender.finish().await.expect("explicit stream completion");
        assert!(receiver.recv().await.expect("terminal frame").is_none());
        assert!(sender.send(b"after end".to_vec()).await.is_err());
    }

    #[tokio::test]
    async fn cumulative_port_limit_rejects_more_data_and_prevents_successful_truncation() {
        let (mut sender, mut receiver, _cancel) = open_with_total_limit(PortType::Bytes, 4, 2, 3);
        sender.send(vec![1, 2]).await.expect("first bounded chunk");
        sender
            .send(vec![3])
            .await
            .expect("final byte within the port");
        assert!(matches!(
            sender.send(vec![4]).await,
            Err(CoreError::InvalidAction(message)) if message.contains("cumulative")
        ));
        assert!(sender.finish().await.is_err());

        assert_eq!(expect_item(receiver.recv().await).as_bytes(), [1, 2]);
        assert_eq!(expect_item(receiver.recv().await).as_bytes(), [3]);
        drop(sender);
        assert!(matches!(
            receiver.recv().await,
            Err(CoreError::VerificationFailed(_))
        ));
    }

    #[tokio::test]
    async fn cancellation_wakes_a_blocked_producer_and_clears_queued_items() {
        let (mut sender, mut receiver, cancel) = open(PortType::Text, 1, 16);
        sender.send(b"queued".to_vec()).await.expect("queue item");
        let mut blocked = Box::pin(sender.send(b"pending".to_vec()));
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut blocked)
                .await
                .is_err()
        );
        cancel.cancel();
        assert!(matches!(blocked.await, Err(CoreError::Cancelled)));
        assert!(matches!(receiver.recv().await, Err(CoreError::Cancelled)));
        assert!(matches!(
            sender.send(b"later".to_vec()).await,
            Err(CoreError::Cancelled)
        ));
    }

    #[tokio::test]
    async fn receiver_disconnect_settles_future_sends() {
        let (mut sender, receiver, _cancel) = open(PortType::Text, 1, 16);
        drop(receiver);
        assert!(matches!(
            sender.send(b"item".to_vec()).await,
            Err(CoreError::ExecutorUnavailable(_))
        ));
    }

    #[tokio::test]
    async fn a_dropped_producer_cannot_look_like_successful_end_of_stream() {
        let (mut sender, mut receiver, _cancel) = open(PortType::Bytes, 2, 64);
        sender.send(b"partial".to_vec()).await.expect("first item");
        let item = expect_item(receiver.recv().await);
        assert_eq!(item.as_bytes(), b"partial");
        drop(sender);
        assert!(matches!(
            receiver.recv().await,
            Err(CoreError::VerificationFailed(_))
        ));
    }

    #[tokio::test]
    async fn open_all_groups_validated_endpoints_by_node_and_port() {
        let stream = channel(1, 16);
        let data_port = port("chunk", PortType::Text, 16);
        let (procedure, capabilities) = procedure(stream.clone(), data_port.clone(), data_port);
        let mut pool = ProcedureStreamPool::new(&procedure, &capabilities, Uuid::new_v4())
            .expect("validated procedure stream pool");
        let (_cancel, cancelled) = super::ProcedureStreamCancellation::new();
        let mut endpoints = pool.open_all(cancelled).expect("complete stream cohort");
        assert_eq!(endpoints.remaining_node_count(), 2);

        let mut producer = endpoints.take_node("producer").expect("producer endpoints");
        let mut consumer = endpoints.take_node("consumer").expect("consumer endpoints");
        producer
            .outputs_mut()
            .get_mut("chunk")
            .expect("declared output")
            .send(b"bounded payload".to_vec())
            .await
            .expect("send through declared channel");
        let received = expect_item(
            consumer
                .inputs_mut()
                .get_mut("chunk")
                .expect("declared input")
                .recv()
                .await,
        );
        assert_eq!(received.task_id(), pool.task_id);
        assert_eq!(received.as_bytes(), b"bounded payload");

        producer
            .outputs_mut()
            .get_mut("chunk")
            .expect("declared output")
            .finish()
            .await
            .expect("explicit producer completion");
        assert!(
            consumer
                .inputs_mut()
                .get_mut("chunk")
                .expect("declared input")
                .recv()
                .await
                .expect("terminal frame is a clean stream end")
                .is_none()
        );
        assert_eq!(endpoints.remaining_node_count(), 0);

        let (_new_cancel, new_cancelled) = super::ProcedureStreamCancellation::new();
        assert!(pool.open_all(new_cancelled).is_err());
    }

    #[tokio::test]
    async fn node_settlement_requires_every_input_and_output_terminal_frame() {
        let stream = channel(2, 16);
        let data_port = port("chunk", PortType::Bytes, 16);
        let (procedure, capabilities) = procedure(stream, data_port.clone(), data_port);
        let mut pool = ProcedureStreamPool::new(&procedure, &capabilities, Uuid::new_v4())
            .expect("validated procedure stream pool");
        let (_cancel, cancelled) = super::ProcedureStreamCancellation::new();
        let mut endpoints = pool.open_all(cancelled).expect("all stream endpoints");
        let mut producer = endpoints.take_node("producer").expect("producer endpoints");
        let mut consumer = endpoints.take_node("consumer").expect("consumer endpoints");

        assert!(producer.validate_terminal_frames().is_err());
        assert!(consumer.validate_terminal_frames().is_err());
        let sender = producer.outputs_mut().get_mut("chunk").unwrap();
        sender.send(b"payload".to_vec()).await.unwrap();
        sender.finish().await.unwrap();
        assert!(producer.validate_terminal_frames().is_ok());

        {
            let receiver = consumer.inputs_mut().get_mut("chunk").unwrap();
            assert!(receiver.recv().await.unwrap().is_some());
        }
        assert!(consumer.validate_terminal_frames().is_err());
        assert!(
            consumer
                .inputs_mut()
                .get_mut("chunk")
                .unwrap()
                .recv()
                .await
                .unwrap()
                .is_none()
        );
        assert!(consumer.validate_terminal_frames().is_ok());
    }

    struct StreamCollectingExecutor;

    #[async_trait::async_trait]
    impl Executor for StreamCollectingExecutor {
        fn name(&self) -> &'static str {
            "stream-collecting-test"
        }

        fn domain(&self) -> ExecutionDomain {
            ExecutionDomain::Native
        }

        fn supports_procedure_streams(&self) -> bool {
            true
        }

        async fn execute(
            &self,
            _action: &CompiledAction,
            _implementation: &ImplementationCandidate,
            _capability: &CapabilityGrant,
        ) -> crate::CoreResult<ExecutionReceipt> {
            Err(CoreError::ExecutorUnavailable(
                "the test executor requires its declared stream input".into(),
            ))
        }

        async fn execute_with_streams(
            &self,
            _action: &CompiledAction,
            _implementation: &ImplementationCandidate,
            _capability: &CapabilityGrant,
            streams: Option<&mut ProcedureNodeStreams>,
        ) -> crate::CoreResult<ExecutionReceipt> {
            let mut bytes = Vec::new();
            let streams = streams.ok_or_else(|| {
                CoreError::ExecutorUnavailable("stream endpoints are missing".into())
            })?;
            let receiver = streams
                .inputs_mut()
                .get_mut("chunk")
                .ok_or_else(|| CoreError::InvalidAction("stream input is missing".into()))?;
            while let Some(item) = receiver.recv().await? {
                bytes.extend_from_slice(item.as_bytes());
            }
            Ok(ExecutionReceipt {
                executor: self.name().into(),
                summary: "collected verified-contract stream data".into(),
                transient_data: serde_json::json!({
                    "text": String::from_utf8(bytes)
                        .expect("the test stream carries UTF-8")
                }),
                rollback: None,
            })
        }
    }

    struct StreamIgnoringExecutor;

    #[async_trait::async_trait]
    impl Executor for StreamIgnoringExecutor {
        fn name(&self) -> &'static str {
            "stream-ignoring-test"
        }

        fn domain(&self) -> ExecutionDomain {
            ExecutionDomain::Native
        }

        fn supports_procedure_streams(&self) -> bool {
            true
        }

        async fn execute(
            &self,
            _action: &CompiledAction,
            _implementation: &ImplementationCandidate,
            _capability: &CapabilityGrant,
        ) -> crate::CoreResult<ExecutionReceipt> {
            Err(CoreError::ExecutorUnavailable(
                "the test executor requires its declared stream input".into(),
            ))
        }

        async fn execute_with_streams(
            &self,
            _action: &CompiledAction,
            _implementation: &ImplementationCandidate,
            _capability: &CapabilityGrant,
            _streams: Option<&mut ProcedureNodeStreams>,
        ) -> crate::CoreResult<ExecutionReceipt> {
            Ok(ExecutionReceipt {
                executor: self.name().into(),
                summary: "fixture deliberately skipped its stream input".into(),
                transient_data: serde_json::Value::Null,
                rollback: None,
            })
        }
    }

    #[tokio::test]
    async fn stream_aware_broker_route_consumes_the_same_one_use_grant() {
        let stream = channel(2, 64);
        let data_port = port("chunk", PortType::Bytes, 64);
        let (procedure, capabilities) = procedure(stream, data_port.clone(), data_port.clone());
        let task_id = Uuid::new_v4();
        let mut pool = ProcedureStreamPool::new(&procedure, &capabilities, task_id)
            .expect("validated procedure stream pool");
        let (cancel, cancelled) = super::ProcedureStreamCancellation::new();
        let mut endpoints = pool.open_all(cancelled).expect("all stream endpoints");
        let mut producer = endpoints.take_node("producer").expect("producer endpoints");
        let consumer = endpoints.take_node("consumer").expect("consumer endpoints");
        producer
            .outputs_mut()
            .get_mut("chunk")
            .expect("producer stream output")
            .send(b"Sage stream payload".to_vec())
            .await
            .expect("stream payload");
        producer
            .outputs_mut()
            .get_mut("chunk")
            .expect("producer stream output")
            .finish()
            .await
            .expect("producer terminal frame");
        drop(producer);

        let path = std::path::PathBuf::from("/tmp/sage-stream-test.txt");
        let proposal = ActionProposal {
            id: Uuid::new_v4(),
            task_id,
            action: Action::ReadFile {
                path: path.clone(),
                max_bytes: 1024,
            },
            expected_outcome: ExpectedOutcome::FileContains {
                path: path.clone(),
                sha256: "a".repeat(64),
            },
            target_resource: path.to_string_lossy().into_owned(),
            provenance: Provenance::user(),
            metadata: Default::default(),
        };
        let implementation = ImplementationCandidate {
            tier: InteractionTier::StructuredIntegration,
            executor: ExecutionDomain::Native,
            operation: "test stream consumer".into(),
        };
        let compiled = CompiledAction {
            proposal: proposal.clone(),
            candidates: vec![implementation.clone()],
        };
        let capabilities_broker = CapabilityBroker::default();
        let grant = capabilities_broker
            .issue_unprepared_for_test(&proposal, ExecutionDomain::Native)
            .await
            .expect("one-use action capability");
        let mut broker = ExecutionBroker::new(capabilities_broker);
        broker.register(Arc::new(StreamCollectingExecutor));

        let receipt = broker
            .execute_with_streams(&compiled, &implementation, &grant, Some(consumer))
            .await
            .expect("authorized stream execution");
        assert_eq!(receipt.transient_data["text"], "Sage stream payload");
        assert!(
            broker
                .execute_with_streams(&compiled, &implementation, &grant, None)
                .await
                .is_err()
        );
        cancel.cancel();
    }

    #[tokio::test]
    async fn stream_broker_rejects_success_before_consumer_reads_the_terminal_frame() {
        let stream = channel(2, 64);
        let data_port = port("chunk", PortType::Bytes, 64);
        let (procedure, capabilities) = procedure(stream, data_port.clone(), data_port);
        let task_id = Uuid::new_v4();
        let mut pool = ProcedureStreamPool::new(&procedure, &capabilities, task_id)
            .expect("validated procedure stream pool");
        let (_cancel, cancelled) = super::ProcedureStreamCancellation::new();
        let mut endpoints = pool.open_all(cancelled).expect("all stream endpoints");
        let producer = endpoints.take_node("producer").expect("producer endpoints");
        let consumer = endpoints.take_node("consumer").expect("consumer endpoints");
        drop(producer);

        let path = std::path::PathBuf::from("/tmp/sage-stream-ignored.txt");
        let proposal = ActionProposal {
            id: Uuid::new_v4(),
            task_id,
            action: Action::ReadFile {
                path: path.clone(),
                max_bytes: 1024,
            },
            expected_outcome: ExpectedOutcome::FileContains {
                path: path.clone(),
                sha256: "b".repeat(64),
            },
            target_resource: path.to_string_lossy().into_owned(),
            provenance: Provenance::user(),
            metadata: Default::default(),
        };
        let implementation = ImplementationCandidate {
            tier: InteractionTier::StructuredIntegration,
            executor: ExecutionDomain::Native,
            operation: "test stream consumer".into(),
        };
        let compiled = CompiledAction {
            proposal: proposal.clone(),
            candidates: vec![implementation.clone()],
        };
        let capabilities = CapabilityBroker::default();
        let grant = capabilities
            .issue_unprepared_for_test(&proposal, ExecutionDomain::Native)
            .await
            .expect("one-use action capability");
        let mut broker = ExecutionBroker::new(capabilities);
        broker.register(Arc::new(StreamIgnoringExecutor));

        assert!(matches!(
            broker
                .execute_with_streams(&compiled, &implementation, &grant, Some(consumer))
                .await,
            Err(CoreError::VerificationFailed(_))
        ));
    }

    #[tokio::test]
    async fn validates_typed_payloads_and_rejects_oversized_or_secret_text() {
        let (mut sender, _, _cancel) = open(PortType::Text, 1, 8);
        assert!(sender.send(b"too long".to_vec()).await.is_err());
        assert!(sender.send(b"api_key=private".to_vec()).await.is_err());

        let (mut boolean_sender, mut boolean_receiver, _boolean_cancel) =
            open(PortType::Boolean, 1, 1);
        assert!(boolean_sender.send(vec![2]).await.is_err());
        boolean_sender.send(vec![1]).await.expect("typed boolean");
        assert_eq!(expect_item(boolean_receiver.recv().await).as_bytes(), &[1]);

        let (mut number_sender, _, _number_cancel) = open(PortType::Number, 1, 8);
        assert!(
            number_sender
                .send(f64::NAN.to_le_bytes().to_vec())
                .await
                .is_err()
        );
    }

    #[test]
    fn rejects_channels_whose_declared_buffer_exceeds_runtime_limits() {
        let producer = port("chunk", PortType::Bytes, 64 * 1024 * 1024);
        let consumer = producer.clone();
        let (oversized_procedure, oversized_capabilities) =
            procedure(channel(2, 16 * 1024 * 1024), producer.clone(), consumer);
        assert!(
            ProcedureStreamPool::new(
                &oversized_procedure,
                &oversized_capabilities,
                Uuid::new_v4(),
            )
            .is_err()
        );

        let mut restricted = port("chunk", PortType::Bytes, 64 * 1024 * 1024);
        restricted.privacy = Sensitivity::Restricted;
        let private_consumer = port("chunk", PortType::Bytes, 64 * 1024 * 1024);
        let (procedure, capabilities) = procedure(channel(1, 1024), restricted, private_consumer);
        assert!(ProcedureStreamPool::new(&procedure, &capabilities, Uuid::new_v4()).is_err());
    }

    #[test]
    fn a_procedure_stream_cohort_cannot_be_opened_twice() {
        let stream = channel(1, 1024);
        let data_port = port("chunk", PortType::Bytes, 1024);
        let (procedure, capabilities) =
            procedure(stream.clone(), data_port.clone(), data_port.clone());
        let mut pool =
            ProcedureStreamPool::new(&procedure, &capabilities, Uuid::new_v4()).expect("pool");
        let (_first_cancel, first_cancel_rx) = super::ProcedureStreamCancellation::new();
        let _endpoints = pool.open_all(first_cancel_rx).expect("first open");
        let (_second_cancel, second_cancel_rx) = super::ProcedureStreamCancellation::new();
        assert!(pool.open_all(second_cancel_rx).is_err());
    }

    #[test]
    fn cancelled_cohort_open_does_not_leave_partial_open_state() {
        let stream = channel(1, 1024);
        let data_port = port("chunk", PortType::Bytes, 1024);
        let (procedure, capabilities) = procedure(stream, data_port.clone(), data_port);
        let mut pool =
            ProcedureStreamPool::new(&procedure, &capabilities, Uuid::new_v4()).expect("pool");
        let (cancel, cancelled) = super::ProcedureStreamCancellation::new();
        cancel.cancel();
        assert!(matches!(
            pool.open_all(cancelled),
            Err(CoreError::Cancelled)
        ));

        let (_fresh_cancel, fresh_cancellation) = super::ProcedureStreamCancellation::new();
        assert!(pool.open_all(fresh_cancellation).is_ok());
    }
}
