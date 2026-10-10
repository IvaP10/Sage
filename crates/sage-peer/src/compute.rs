//! Bounded first-party contracts for partitioning independent peer work.
//!
//! The contracts bind every input and output to a deterministic unit index,
//! exact job plan, resource, and result verifier. They provide integrity and
//! structural completeness; the caller remains responsible for task-specific
//! correctness checks and encrypted durable checkpoint storage. Typed payloads
//! are capped at 256 MiB and travel as ordered, digest-bound chunks over the
//! secure channel's 1 MiB message limit.

use std::fmt;

use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use super::{
    DelegatedJobKind, DisclosureScope, LeaseId, LeaseRequest, PeerDispatchJournal, PeerError,
    PeerId, PeerLeaseGrant, PeerMessageKind, PeerResult, PermittedPeerJob, PublicPeerIdentity,
};

const MAX_COMPUTE_UNITS: usize = 65_536;
const MAX_COMPUTE_PARTITIONS: usize = 1_024;
const MAX_COMPUTE_BYTES: u64 = super::MAX_PEER_COMPUTE_BYTES;
const MAX_COMPUTE_PAYLOAD_BYTES: usize = MAX_COMPUTE_BYTES as usize;
const PLAN_DOMAIN: &[u8] = b"sage:peer:compute-plan:v1\0";
const RESULT_DOMAIN: &[u8] = b"sage:peer:compute-result:v1\0";
const COMPUTE_WIRE_MAGIC: &[u8; 4] = b"SGC1";
const COMPUTE_WIRE_VERSION: u16 = 1;
const COMPUTE_REQUEST_KIND: u8 = 1;
const COMPUTE_RESULT_KIND: u8 = 2;
const COMPUTE_REQUEST_HEADER_BYTES: usize = 150;
const COMPUTE_RESULT_HEADER_BYTES: usize = 61;
const COMPUTE_CHUNK_MAGIC: &[u8; 4] = b"SGT1";
const COMPUTE_CHUNK_VERSION: u16 = 1;
const COMPUTE_CHUNK_HEADER_BYTES: usize = 105;
const COMPUTE_CHUNK_DATA_BYTES: usize = super::MAX_PEER_MESSAGE_BYTES - COMPUTE_CHUNK_HEADER_BYTES;

/// Typed binding repeated in every encrypted chunk of one request or result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeTransferKind {
    PartitionRequest,
    PartitionResult,
}

impl ComputeTransferKind {
    fn wire_tag(self) -> u8 {
        match self {
            Self::PartitionRequest => COMPUTE_REQUEST_KIND,
            Self::PartitionResult => COMPUTE_RESULT_KIND,
        }
    }

    fn message_kind(self) -> PeerMessageKind {
        match self {
            Self::PartitionRequest => PeerMessageKind::JobOffer,
            Self::PartitionResult => PeerMessageKind::JobReceipt,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TransferMetadata {
    kind: ComputeTransferKind,
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    partition_index: u16,
    total_bytes: u32,
    chunk_count: u32,
    digest: [u8; 32],
}

/// The exact transfer the receiver is prepared to assemble. Its maximum is
/// derived from the registered plan before allocating reassembly RAM; the live
/// lease gate still runs before input bytes become visible to a worker.
#[derive(Debug, Clone, Copy)]
pub struct ComputeTransferExpectation {
    kind: ComputeTransferKind,
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    partition_index: u16,
    maximum_payload_bytes: usize,
}

impl ComputeTransferExpectation {
    fn new(
        kind: ComputeTransferKind,
        job_id: [u8; 16],
        plan_digest: [u8; 32],
        partition_index: u16,
        maximum_payload_bytes: usize,
    ) -> PeerResult<Self> {
        if maximum_payload_bytes == 0 || maximum_payload_bytes > MAX_COMPUTE_PAYLOAD_BYTES {
            return Err(PeerError::InvalidComputeJob);
        }
        Ok(Self {
            kind,
            job_id,
            plan_digest,
            partition_index,
            maximum_payload_bytes,
        })
    }
}

/// Lazy frame encoder: it holds one bounded zeroizing payload and allocates
/// only the current <=1 MiB frame, rather than a second full-sized frame list.
pub struct ComputeTransferEncoder {
    metadata: TransferMetadata,
    payload: Zeroizing<Vec<u8>>,
    next_chunk: u32,
}

impl ComputeTransferEncoder {
    fn new(
        kind: ComputeTransferKind,
        job_id: [u8; 16],
        plan_digest: [u8; 32],
        partition_index: u16,
        payload: Zeroizing<Vec<u8>>,
    ) -> PeerResult<Self> {
        if payload.is_empty() || payload.len() > MAX_COMPUTE_PAYLOAD_BYTES {
            return Err(PeerError::InvalidFrame);
        }
        let chunk_count = payload
            .len()
            .checked_add(COMPUTE_CHUNK_DATA_BYTES - 1)
            .ok_or(PeerError::InvalidFrame)?
            / COMPUTE_CHUNK_DATA_BYTES;
        let metadata = TransferMetadata {
            kind,
            job_id,
            plan_digest,
            partition_index,
            total_bytes: u32::try_from(payload.len()).map_err(|_| PeerError::InvalidFrame)?,
            chunk_count: u32::try_from(chunk_count).map_err(|_| PeerError::InvalidFrame)?,
            digest: Sha256::digest(&payload).into(),
        };
        Ok(Self {
            metadata,
            payload,
            next_chunk: 0,
        })
    }

    pub fn message_kind(&self) -> PeerMessageKind {
        self.metadata.kind.message_kind()
    }

    pub fn chunk_count(&self) -> u32 {
        self.metadata.chunk_count
    }

    pub fn next_frame(&mut self) -> PeerResult<Option<Zeroizing<Vec<u8>>>> {
        if self.next_chunk == self.metadata.chunk_count {
            return Ok(None);
        }
        let start = usize::try_from(self.next_chunk)
            .ok()
            .and_then(|index| index.checked_mul(COMPUTE_CHUNK_DATA_BYTES))
            .ok_or(PeerError::InvalidFrame)?;
        let end = start
            .saturating_add(COMPUTE_CHUNK_DATA_BYTES)
            .min(self.payload.len());
        let data = self
            .payload
            .get(start..end)
            .ok_or(PeerError::InvalidFrame)?;
        let frame_len = COMPUTE_CHUNK_HEADER_BYTES
            .checked_add(data.len())
            .filter(|length| *length <= super::MAX_PEER_MESSAGE_BYTES)
            .ok_or(PeerError::InvalidFrame)?;
        let mut frame = Zeroizing::new(Vec::new());
        frame
            .try_reserve_exact(frame_len)
            .map_err(|_| PeerError::InvalidFrame)?;
        frame.extend_from_slice(COMPUTE_CHUNK_MAGIC);
        frame.extend_from_slice(&COMPUTE_CHUNK_VERSION.to_be_bytes());
        frame.push(self.metadata.kind.wire_tag());
        frame.extend_from_slice(&self.metadata.job_id);
        frame.extend_from_slice(&self.metadata.plan_digest);
        frame.extend_from_slice(&self.metadata.partition_index.to_be_bytes());
        frame.extend_from_slice(&self.next_chunk.to_be_bytes());
        frame.extend_from_slice(&self.metadata.chunk_count.to_be_bytes());
        frame.extend_from_slice(&self.metadata.total_bytes.to_be_bytes());
        frame.extend_from_slice(
            &u32::try_from(data.len())
                .map_err(|_| PeerError::InvalidFrame)?
                .to_be_bytes(),
        );
        frame.extend_from_slice(&self.metadata.digest);
        frame.extend_from_slice(data);
        self.next_chunk += 1;
        Ok(Some(frame))
    }
}

/// Strict in-order reassembler. It allocates only after the first authenticated
/// chunk matches the registered job binding and declared size limit.
pub struct ComputeTransferAssembler {
    expected: ComputeTransferExpectation,
    metadata: Option<TransferMetadata>,
    next_chunk: u32,
    payload: Zeroizing<Vec<u8>>,
    complete_or_failed: bool,
}

impl ComputeTransferAssembler {
    pub fn new(expected: ComputeTransferExpectation) -> Self {
        Self {
            expected,
            metadata: None,
            next_chunk: 0,
            payload: Zeroizing::new(Vec::new()),
            complete_or_failed: false,
        }
    }

    pub fn push(
        &mut self,
        authenticated_message_kind: PeerMessageKind,
        frame: &[u8],
    ) -> PeerResult<Option<Zeroizing<Vec<u8>>>> {
        let result = self.push_inner(authenticated_message_kind, frame);
        if result.is_err() {
            self.payload.zeroize();
            self.payload.clear();
            self.complete_or_failed = true;
        }
        result
    }

    fn push_inner(
        &mut self,
        authenticated_message_kind: PeerMessageKind,
        frame: &[u8],
    ) -> PeerResult<Option<Zeroizing<Vec<u8>>>> {
        if self.complete_or_failed
            || frame.len() < COMPUTE_CHUNK_HEADER_BYTES
            || frame.len() > super::MAX_PEER_MESSAGE_BYTES
            || !frame.starts_with(COMPUTE_CHUNK_MAGIC)
        {
            return Err(PeerError::InvalidFrame);
        }
        let mut offset = COMPUTE_CHUNK_MAGIC.len();
        if wire_read_u16(frame, &mut offset)? != COMPUTE_CHUNK_VERSION {
            return Err(PeerError::InvalidFrame);
        }
        let kind = match wire_read_u8(frame, &mut offset)? {
            COMPUTE_REQUEST_KIND => ComputeTransferKind::PartitionRequest,
            COMPUTE_RESULT_KIND => ComputeTransferKind::PartitionResult,
            _ => return Err(PeerError::InvalidFrame),
        };
        let job_id = wire_read_array(frame, &mut offset)?;
        let plan_digest = wire_read_array(frame, &mut offset)?;
        let partition_index = wire_read_u16(frame, &mut offset)?;
        let chunk_index = wire_read_u32(frame, &mut offset)?;
        let chunk_count = wire_read_u32(frame, &mut offset)?;
        let total_bytes = wire_read_u32(frame, &mut offset)?;
        let data_bytes = wire_read_u32(frame, &mut offset)? as usize;
        let digest = wire_read_array(frame, &mut offset)?;
        let metadata = TransferMetadata {
            kind,
            job_id,
            plan_digest,
            partition_index,
            total_bytes,
            chunk_count,
            digest,
        };
        let expected_count = usize::try_from(total_bytes)
            .ok()
            .and_then(|bytes| bytes.checked_add(COMPUTE_CHUNK_DATA_BYTES - 1))
            .map(|bytes| bytes / COMPUTE_CHUNK_DATA_BYTES)
            .and_then(|count| u32::try_from(count).ok());
        let start = usize::try_from(chunk_index)
            .ok()
            .and_then(|index| index.checked_mul(COMPUTE_CHUNK_DATA_BYTES))
            .ok_or(PeerError::InvalidFrame)?;
        let expected_data_bytes = usize::try_from(total_bytes)
            .ok()
            .and_then(|total| total.checked_sub(start))
            .map(|remaining| remaining.min(COMPUTE_CHUNK_DATA_BYTES));
        if metadata.kind != self.expected.kind
            || authenticated_message_kind != self.expected.kind.message_kind()
            || metadata.job_id != self.expected.job_id
            || metadata.plan_digest != self.expected.plan_digest
            || metadata.partition_index != self.expected.partition_index
            || total_bytes == 0
            || total_bytes as usize > self.expected.maximum_payload_bytes
            || total_bytes as usize > MAX_COMPUTE_PAYLOAD_BYTES
            || chunk_count == 0
            || Some(chunk_count) != expected_count
            || chunk_index >= chunk_count
            || Some(data_bytes) != expected_data_bytes
            || frame.len() != offset.saturating_add(data_bytes)
        {
            return Err(PeerError::InvalidFrame);
        }
        if let Some(previous) = self.metadata {
            if previous != metadata || chunk_index != self.next_chunk {
                return Err(PeerError::ReplayOrReordering);
            }
        } else {
            if chunk_index != 0 {
                return Err(PeerError::ReplayOrReordering);
            }
            self.payload
                .try_reserve_exact(total_bytes as usize)
                .map_err(|_| PeerError::InvalidFrame)?;
            self.metadata = Some(metadata);
        }
        let end = offset
            .checked_add(data_bytes)
            .ok_or(PeerError::InvalidFrame)?;
        self.payload
            .extend_from_slice(frame.get(offset..end).ok_or(PeerError::InvalidFrame)?);
        self.next_chunk += 1;
        if self.next_chunk < chunk_count {
            return Ok(None);
        }
        if self.payload.len() != total_bytes as usize
            || Sha256::digest(&self.payload).as_slice() != metadata.digest
        {
            return Err(PeerError::AuthenticationFailed);
        }
        self.complete_or_failed = true;
        Ok(Some(std::mem::take(&mut self.payload)))
    }
}

fn maximum_result_data_bytes(output_count: usize) -> Option<u64> {
    let record_bytes = output_count.checked_mul(40)?;
    MAX_COMPUTE_PAYLOAD_BYTES
        .checked_sub(COMPUTE_RESULT_HEADER_BYTES.checked_add(record_bytes)?)
        .map(|bytes| bytes as u64)
}

fn wire_read_array<const N: usize>(payload: &[u8], offset: &mut usize) -> PeerResult<[u8; N]> {
    let end = offset.checked_add(N).ok_or(PeerError::InvalidFrame)?;
    let bytes = payload.get(*offset..end).ok_or(PeerError::InvalidFrame)?;
    let mut value = [0_u8; N];
    value.copy_from_slice(bytes);
    *offset = end;
    Ok(value)
}

fn wire_read_u8(payload: &[u8], offset: &mut usize) -> PeerResult<u8> {
    Ok(wire_read_array::<1>(payload, offset)?[0])
}

fn wire_read_u16(payload: &[u8], offset: &mut usize) -> PeerResult<u16> {
    Ok(u16::from_be_bytes(wire_read_array(payload, offset)?))
}

fn wire_read_u32(payload: &[u8], offset: &mut usize) -> PeerResult<u32> {
    Ok(u32::from_be_bytes(wire_read_array(payload, offset)?))
}

fn wire_read_u64(payload: &[u8], offset: &mut usize) -> PeerResult<u64> {
    Ok(u64::from_be_bytes(wire_read_array(payload, offset)?))
}

/// Binds one explicit input to its expected byte length and output ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeUnitSpec {
    index: u32,
    input_digest: [u8; 32],
    input_bytes: u32,
    maximum_output_bytes: u32,
}

impl ComputeUnitSpec {
    pub fn new(
        index: u32,
        input_digest: [u8; 32],
        input_bytes: u32,
        maximum_output_bytes: u32,
    ) -> PeerResult<Self> {
        if input_bytes == 0 || maximum_output_bytes == 0 {
            return Err(PeerError::InvalidComputeJob);
        }
        Ok(Self {
            index,
            input_digest,
            input_bytes,
            maximum_output_bytes,
        })
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn input_digest(&self) -> &[u8; 32] {
        &self.input_digest
    }

    pub fn input_bytes(&self) -> u32 {
        self.input_bytes
    }

    pub fn maximum_output_bytes(&self) -> u32 {
        self.maximum_output_bytes
    }
}

/// A secret-bearing work unit. Debug output never includes input bytes.
pub struct UnitInput {
    index: u32,
    bytes: Zeroizing<Vec<u8>>,
}

impl UnitInput {
    pub fn new(index: u32, bytes: Vec<u8>) -> PeerResult<Self> {
        if bytes.is_empty() || bytes.len() as u64 > MAX_COMPUTE_BYTES {
            return Err(PeerError::InvalidComputeJob);
        }
        Ok(Self {
            index,
            bytes: Zeroizing::new(bytes),
        })
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl fmt::Debug for UnitInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnitInput")
            .field("index", &self.index)
            .field("bytes", &"[redacted]")
            .field("byte_count", &self.bytes.len())
            .finish()
    }
}

/// A bounded, secret-bearing result for one exact unit.
pub struct UnitOutput {
    index: u32,
    input_digest: [u8; 32],
    bytes: Zeroizing<Vec<u8>>,
}

impl UnitOutput {
    pub fn new(index: u32, input_digest: [u8; 32], bytes: Vec<u8>) -> PeerResult<Self> {
        Self::from_zeroizing(index, input_digest, Zeroizing::new(bytes))
    }

    /// Construct an output while retaining ownership of a zeroizing buffer,
    /// for bounded storage readers that must scrub partially loaded data on
    /// every error path.
    pub fn from_zeroizing(
        index: u32,
        input_digest: [u8; 32],
        bytes: Zeroizing<Vec<u8>>,
    ) -> PeerResult<Self> {
        if bytes.len() as u64 > MAX_COMPUTE_BYTES {
            return Err(PeerError::InvalidComputeResult);
        }
        Ok(Self {
            index,
            input_digest,
            bytes,
        })
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn input_digest(&self) -> &[u8; 32] {
        &self.input_digest
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl fmt::Debug for UnitOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnitOutput")
            .field("index", &self.index)
            .field("input_digest", &"[redacted]")
            .field("bytes", &"[redacted]")
            .field("byte_count", &self.bytes.len())
            .finish()
    }
}

/// Worker response bound to a job plan. The peer is taken from the
/// authenticated channel, never from an untrusted response payload.
pub struct PartitionResult {
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    partition_index: u16,
    outputs: Vec<UnitOutput>,
}

impl PartitionResult {
    pub fn new(
        job_id: [u8; 16],
        plan_digest: [u8; 32],
        partition_index: u16,
        outputs: Vec<UnitOutput>,
    ) -> PeerResult<Self> {
        let output_bytes = outputs.iter().try_fold(0_u64, |total, output| {
            total.checked_add(output.bytes.len() as u64)
        });
        if outputs.is_empty()
            || outputs.len() > MAX_COMPUTE_UNITS
            || output_bytes.is_none_or(|total| total > MAX_COMPUTE_BYTES)
        {
            return Err(PeerError::InvalidComputeResult);
        }
        Ok(Self {
            job_id,
            plan_digest,
            partition_index,
            outputs,
        })
    }

    /// Total result bytes, excluding wire framing and unit metadata.
    pub fn output_bytes(&self) -> u64 {
        self.outputs
            .iter()
            .map(|output| output.bytes.len() as u64)
            .sum()
    }

    /// Encode one bounded result payload. Use `into_transfer` to split payloads
    /// across authenticated `JobReceipt` frames without exceeding frame limits.
    pub fn encode(&self) -> PeerResult<Zeroizing<Vec<u8>>> {
        if self.outputs.is_empty() || self.outputs.len() > MAX_COMPUTE_UNITS {
            return Err(PeerError::InvalidComputeResult);
        }
        let encoded_len = self
            .outputs
            .iter()
            .try_fold(COMPUTE_RESULT_HEADER_BYTES, |total, output| {
                total.checked_add(40)?.checked_add(output.bytes.len())
            });
        let Some(encoded_len) = encoded_len.filter(|length| *length <= MAX_COMPUTE_PAYLOAD_BYTES)
        else {
            return Err(PeerError::InvalidFrame);
        };
        let mut payload = Zeroizing::new(Vec::new());
        payload
            .try_reserve_exact(encoded_len)
            .map_err(|_| PeerError::InvalidFrame)?;
        payload.extend_from_slice(COMPUTE_WIRE_MAGIC);
        payload.extend_from_slice(&COMPUTE_WIRE_VERSION.to_be_bytes());
        payload.push(COMPUTE_RESULT_KIND);
        payload.extend_from_slice(&self.job_id);
        payload.extend_from_slice(&self.plan_digest);
        payload.extend_from_slice(&self.partition_index.to_be_bytes());
        payload.extend_from_slice(
            &u32::try_from(self.outputs.len())
                .map_err(|_| PeerError::InvalidComputeResult)?
                .to_be_bytes(),
        );
        let mut previous_index = None;
        for output in &self.outputs {
            if previous_index.is_some_and(|previous| output.index <= previous)
                || output.bytes.len() > u32::MAX as usize
            {
                return Err(PeerError::InvalidComputeResult);
            }
            previous_index = Some(output.index);
            payload.extend_from_slice(&output.index.to_be_bytes());
            payload.extend_from_slice(&output.input_digest);
            payload.extend_from_slice(&(output.bytes.len() as u32).to_be_bytes());
            payload.extend_from_slice(&output.bytes);
        }
        debug_assert_eq!(payload.len(), encoded_len);
        Ok(payload)
    }

    pub fn into_transfer(self) -> PeerResult<ComputeTransferEncoder> {
        let payload = self.encode()?;
        ComputeTransferEncoder::new(
            ComputeTransferKind::PartitionResult,
            self.job_id,
            self.plan_digest,
            self.partition_index,
            payload,
        )
    }

    /// Decode a bounded result after all authenticated transfer chunks settle.
    pub fn decode(payload: &[u8]) -> PeerResult<Self> {
        if payload.len() < COMPUTE_RESULT_HEADER_BYTES
            || payload.len() > MAX_COMPUTE_PAYLOAD_BYTES
            || !payload.starts_with(COMPUTE_WIRE_MAGIC)
        {
            return Err(PeerError::InvalidFrame);
        }
        let mut offset = COMPUTE_WIRE_MAGIC.len();
        if wire_read_u16(payload, &mut offset)? != COMPUTE_WIRE_VERSION
            || wire_read_u8(payload, &mut offset)? != COMPUTE_RESULT_KIND
        {
            return Err(PeerError::InvalidFrame);
        }
        let job_id = wire_read_array::<16>(payload, &mut offset)?;
        let plan_digest = wire_read_array::<32>(payload, &mut offset)?;
        let partition_index = wire_read_u16(payload, &mut offset)?;
        let output_count = usize::try_from(wire_read_u32(payload, &mut offset)?)
            .map_err(|_| PeerError::InvalidComputeResult)?;
        let remaining = payload.len().saturating_sub(offset);
        if output_count == 0 || output_count > MAX_COMPUTE_UNITS || output_count > remaining / 40 {
            return Err(PeerError::InvalidComputeResult);
        }
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(output_count)
            .map_err(|_| PeerError::InvalidFrame)?;
        let mut previous_index = None;
        for _ in 0..output_count {
            let index = wire_read_u32(payload, &mut offset)?;
            if previous_index.is_some_and(|previous| index <= previous) {
                return Err(PeerError::InvalidComputeResult);
            }
            previous_index = Some(index);
            let input_digest = wire_read_array::<32>(payload, &mut offset)?;
            let length = usize::try_from(wire_read_u32(payload, &mut offset)?)
                .map_err(|_| PeerError::InvalidComputeResult)?;
            let end = offset.checked_add(length).ok_or(PeerError::InvalidFrame)?;
            let bytes = payload.get(offset..end).ok_or(PeerError::InvalidFrame)?;
            outputs.push(UnitOutput::new(index, input_digest, bytes.to_vec())?);
            offset = end;
        }
        if offset != payload.len() {
            return Err(PeerError::InvalidFrame);
        }
        Self::new(job_id, plan_digest, partition_index, outputs)
    }
}

impl fmt::Debug for PartitionResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PartitionResult")
            .field("partition_index", &self.partition_index)
            .field("outputs", &self.outputs.len())
            .field("payloads", &"[redacted]")
            .finish_non_exhaustive()
    }
}

/// Inputs serialized only after the donor-signed grant and exact unit binding
/// have been checked. The donor must consume its own live one-use lease gate
/// before exposing decoded inputs to a worker.
pub struct PartitionRequest {
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    partition_index: u16,
    lease_id: LeaseId,
    peer: PeerId,
    job: DelegatedJobKind,
    resource_id: [u8; 32],
    maximum_output_bytes: u64,
    inputs: Vec<UnitInput>,
}

/// Untrusted request data decoded from the authenticated channel. Its inputs
/// remain inaccessible until a registered plan and the donor's live lease
/// authority both approve it.
pub struct ReceivedPartitionRequest {
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    partition_index: u16,
    lease_id: LeaseId,
    authenticated_peer: PeerId,
    job: DelegatedJobKind,
    resource_id: [u8; 32],
    maximum_output_bytes: u64,
    inputs: Vec<UnitInput>,
}

impl ReceivedPartitionRequest {
    /// Decode a bounded request and bind its claimed sender to the identity
    /// authenticated by the enclosing encrypted channel.
    pub fn decode(payload: &[u8], authenticated_peer: PeerId) -> PeerResult<Self> {
        if payload.len() < COMPUTE_REQUEST_HEADER_BYTES
            || payload.len() > MAX_COMPUTE_PAYLOAD_BYTES
            || !payload.starts_with(COMPUTE_WIRE_MAGIC)
        {
            return Err(PeerError::InvalidFrame);
        }
        let mut offset = COMPUTE_WIRE_MAGIC.len();
        if wire_read_u16(payload, &mut offset)? != COMPUTE_WIRE_VERSION
            || wire_read_u8(payload, &mut offset)? != COMPUTE_REQUEST_KIND
        {
            return Err(PeerError::InvalidFrame);
        }
        let job_id = wire_read_array::<16>(payload, &mut offset)?;
        let plan_digest = wire_read_array::<32>(payload, &mut offset)?;
        let partition_index = wire_read_u16(payload, &mut offset)?;
        let lease_id = LeaseId::from_bytes(wire_read_array::<16>(payload, &mut offset)?);
        let claimed_peer = wire_read_array::<32>(payload, &mut offset)?;
        if claimed_peer != *authenticated_peer.as_bytes() {
            return Err(PeerError::AuthenticationFailed);
        }
        let job = DelegatedJobKind::try_from(wire_read_u8(payload, &mut offset)?)
            .map_err(|_| PeerError::InvalidComputeJob)?;
        let resource_id = wire_read_array::<32>(payload, &mut offset)?;
        let maximum_output_bytes = wire_read_u64(payload, &mut offset)?;
        let input_count = usize::try_from(wire_read_u32(payload, &mut offset)?)
            .map_err(|_| PeerError::InvalidComputeJob)?;
        let remaining = payload.len().saturating_sub(offset);
        if maximum_output_bytes == 0
            || maximum_output_bytes > MAX_COMPUTE_BYTES
            || input_count == 0
            || input_count > MAX_COMPUTE_UNITS
            || input_count > remaining / 9
            || maximum_result_data_bytes(input_count)
                .is_none_or(|maximum| maximum_output_bytes > maximum)
        {
            return Err(PeerError::InvalidComputeJob);
        }
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(input_count)
            .map_err(|_| PeerError::InvalidFrame)?;
        let mut previous_index = None;
        let mut total_input_bytes = 0_u64;
        for _ in 0..input_count {
            let index = wire_read_u32(payload, &mut offset)?;
            if previous_index.is_some_and(|previous| index <= previous) {
                return Err(PeerError::InvalidComputeJob);
            }
            previous_index = Some(index);
            let length = usize::try_from(wire_read_u32(payload, &mut offset)?)
                .map_err(|_| PeerError::InvalidComputeJob)?;
            if length == 0 || length as u64 > MAX_COMPUTE_BYTES {
                return Err(PeerError::InvalidComputeJob);
            }
            let end = offset.checked_add(length).ok_or(PeerError::InvalidFrame)?;
            let bytes = payload.get(offset..end).ok_or(PeerError::InvalidFrame)?;
            total_input_bytes = total_input_bytes
                .checked_add(length as u64)
                .filter(|total| *total <= MAX_COMPUTE_BYTES)
                .ok_or(PeerError::InvalidComputeJob)?;
            inputs.push(UnitInput::new(index, bytes.to_vec())?);
            offset = end;
        }
        if offset != payload.len() {
            return Err(PeerError::InvalidFrame);
        }
        Ok(Self {
            job_id,
            plan_digest,
            partition_index,
            lease_id,
            authenticated_peer,
            job,
            resource_id,
            maximum_output_bytes,
            inputs,
        })
    }

    pub fn job_id(&self) -> &[u8; 16] {
        &self.job_id
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }

    pub fn partition_index(&self) -> u16 {
        self.partition_index
    }

    pub fn lease_id(&self) -> LeaseId {
        self.lease_id
    }

    pub fn authenticated_peer(&self) -> PeerId {
        self.authenticated_peer
    }

    pub fn job(&self) -> DelegatedJobKind {
        self.job
    }

    pub fn resource_id(&self) -> &[u8; 32] {
        &self.resource_id
    }

    pub fn input_bytes(&self) -> u64 {
        self.inputs
            .iter()
            .map(|input| input.bytes.len() as u64)
            .sum()
    }

    pub fn maximum_output_bytes(&self) -> u64 {
        self.maximum_output_bytes
    }
}

impl fmt::Debug for ReceivedPartitionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReceivedPartitionRequest")
            .field("job_id", &"[redacted]")
            .field("partition_index", &self.partition_index)
            .field("lease_id", &"[redacted]")
            .field("authenticated_peer", &self.authenticated_peer)
            .field("job", &self.job)
            .field("resource_id", &"[redacted]")
            .field("input_count", &self.inputs.len())
            .field("input_payloads", &"[redacted]")
            .finish()
    }
}

/// Plan-bound request ready to reserve against the local owner's live lease.
pub struct PreparedPartitionRequest {
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    partition_index: u16,
    lease_id: LeaseId,
    authenticated_peer: PeerId,
    job: DelegatedJobKind,
    resource_id: [u8; 32],
    input_bytes: u64,
    maximum_output_bytes: u64,
    maximum_output_bytes_per_unit: Vec<u32>,
    inputs: Vec<UnitInput>,
}

impl PreparedPartitionRequest {
    pub fn lease_id(&self) -> LeaseId {
        self.lease_id
    }

    pub fn authenticated_peer(&self) -> PeerId {
        self.authenticated_peer
    }

    /// Exact reservation request derived from the registered plan, not from
    /// an independently trusted worker payload.
    pub fn lease_request(&self) -> LeaseRequest {
        LeaseRequest {
            job_id: self.job_id,
            partition_index: self.partition_index,
            job: self.job,
            resource_id: self.resource_id,
            disclosure: DisclosureScope::ExplicitJobInputAndResult,
            input_bytes: self.input_bytes,
            maximum_output_bytes: self.maximum_output_bytes,
        }
    }

    /// Consume the owner's one-use admission proof, validate exact bindings,
    /// and durably fence this job partition before the request can reach a
    /// worker. If journal persistence fails, the zeroizing request is dropped
    /// and the caller must treat the already-consumed lease permit as uncertain.
    pub fn authorize_durable(
        self,
        permitted: PermittedPeerJob,
        journal: &mut PeerDispatchJournal,
    ) -> PeerResult<AuthorizedPartitionRequest> {
        let authorized = self.authorize(permitted)?;
        journal.record_dispatch(&authorized)?;
        Ok(authorized)
    }

    /// Shared binding validator. Keep this crate-internal so callers cannot
    /// bypass the durable fence before worker dispatch.
    pub(crate) fn authorize(
        self,
        permitted: PermittedPeerJob,
    ) -> PeerResult<AuthorizedPartitionRequest> {
        let per_unit_output_total = self
            .maximum_output_bytes_per_unit
            .iter()
            .try_fold(0_u64, |total, limit| total.checked_add(u64::from(*limit)));
        if permitted.lease_id() != self.lease_id
            || permitted.peer() != self.authenticated_peer
            || permitted.job_id() != &self.job_id
            || permitted.partition_index() != self.partition_index
            || permitted.job() != self.job
            || permitted.resource_id() != &self.resource_id
            || permitted.disclosure() != DisclosureScope::ExplicitJobInputAndResult
            || permitted.maximum_input_bytes() != self.input_bytes
            || permitted.maximum_output_bytes() != self.maximum_output_bytes
            || self.maximum_output_bytes_per_unit.len() != self.inputs.len()
            || per_unit_output_total != Some(self.maximum_output_bytes)
        {
            return Err(PeerError::LeaseDenied);
        }
        Ok(AuthorizedPartitionRequest {
            job_id: self.job_id,
            plan_digest: self.plan_digest,
            partition_index: self.partition_index,
            lease_id: self.lease_id,
            peer: self.authenticated_peer,
            job: self.job,
            resource_id: self.resource_id,
            maximum_output_bytes: self.maximum_output_bytes,
            maximum_output_bytes_per_unit: self.maximum_output_bytes_per_unit,
            inputs: self.inputs,
        })
    }
}

/// Non-cloneable request that passed both registered-plan validation and the
/// donor's current peer-lease dispatch gate.
pub struct AuthorizedPartitionRequest {
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    partition_index: u16,
    lease_id: LeaseId,
    peer: PeerId,
    job: DelegatedJobKind,
    resource_id: [u8; 32],
    maximum_output_bytes: u64,
    maximum_output_bytes_per_unit: Vec<u32>,
    inputs: Vec<UnitInput>,
}

impl AuthorizedPartitionRequest {
    pub fn job_id(&self) -> &[u8; 16] {
        &self.job_id
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }

    pub fn partition_index(&self) -> u16 {
        self.partition_index
    }

    pub fn lease_id(&self) -> LeaseId {
        self.lease_id
    }

    pub fn peer(&self) -> PeerId {
        self.peer
    }

    pub fn job(&self) -> DelegatedJobKind {
        self.job
    }

    pub fn resource_id(&self) -> &[u8; 32] {
        &self.resource_id
    }

    pub fn maximum_output_bytes(&self) -> u64 {
        self.maximum_output_bytes
    }

    pub fn maximum_output_bytes_per_unit(&self) -> &[u32] {
        &self.maximum_output_bytes_per_unit
    }

    pub fn inputs(&self) -> &[UnitInput] {
        &self.inputs
    }
}

impl fmt::Debug for AuthorizedPartitionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthorizedPartitionRequest")
            .field("job_id", &"[redacted]")
            .field("partition_index", &self.partition_index)
            .field("lease_id", &"[redacted]")
            .field("peer", &self.peer)
            .field("job", &self.job)
            .field("resource_id", &"[redacted]")
            .field("input_count", &self.inputs.len())
            .field("input_payloads", &"[redacted]")
            .finish()
    }
}

impl PartitionRequest {
    pub fn job_id(&self) -> &[u8; 16] {
        &self.job_id
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }

    pub fn partition_index(&self) -> u16 {
        self.partition_index
    }

    pub fn lease_id(&self) -> LeaseId {
        self.lease_id
    }

    pub fn peer(&self) -> PeerId {
        self.peer
    }

    pub fn job(&self) -> DelegatedJobKind {
        self.job
    }

    pub fn resource_id(&self) -> &[u8; 32] {
        &self.resource_id
    }

    pub fn maximum_output_bytes(&self) -> u64 {
        self.maximum_output_bytes
    }

    pub fn inputs(&self) -> &[UnitInput] {
        &self.inputs
    }

    /// Encode one bounded request payload. Use `into_transfer` to split large
    /// payloads across authenticated `JobOffer` frames.
    pub fn encode(&self) -> PeerResult<Zeroizing<Vec<u8>>> {
        if self.inputs.is_empty()
            || self.inputs.len() > MAX_COMPUTE_UNITS
            || self.maximum_output_bytes == 0
            || self.maximum_output_bytes > MAX_COMPUTE_BYTES
            || maximum_result_data_bytes(self.inputs.len())
                .is_none_or(|maximum| self.maximum_output_bytes > maximum)
        {
            return Err(PeerError::InvalidComputeJob);
        }
        let encoded_len = self
            .inputs
            .iter()
            .try_fold(COMPUTE_REQUEST_HEADER_BYTES, |total, input| {
                total.checked_add(8)?.checked_add(input.bytes.len())
            });
        let Some(encoded_len) = encoded_len.filter(|length| *length <= MAX_COMPUTE_PAYLOAD_BYTES)
        else {
            return Err(PeerError::InvalidFrame);
        };
        let mut payload = Zeroizing::new(Vec::new());
        payload
            .try_reserve_exact(encoded_len)
            .map_err(|_| PeerError::InvalidFrame)?;
        payload.extend_from_slice(COMPUTE_WIRE_MAGIC);
        payload.extend_from_slice(&COMPUTE_WIRE_VERSION.to_be_bytes());
        payload.push(COMPUTE_REQUEST_KIND);
        payload.extend_from_slice(&self.job_id);
        payload.extend_from_slice(&self.plan_digest);
        payload.extend_from_slice(&self.partition_index.to_be_bytes());
        payload.extend_from_slice(self.lease_id.as_bytes());
        payload.extend_from_slice(self.peer.as_bytes());
        payload.push(self.job as u8);
        payload.extend_from_slice(&self.resource_id);
        payload.extend_from_slice(&self.maximum_output_bytes.to_be_bytes());
        payload.extend_from_slice(
            &u32::try_from(self.inputs.len())
                .map_err(|_| PeerError::InvalidComputeJob)?
                .to_be_bytes(),
        );
        let mut previous_index = None;
        for input in &self.inputs {
            if input.bytes.is_empty()
                || input.bytes.len() > u32::MAX as usize
                || previous_index.is_some_and(|previous| input.index <= previous)
            {
                return Err(PeerError::InvalidComputeJob);
            }
            previous_index = Some(input.index);
            payload.extend_from_slice(&input.index.to_be_bytes());
            payload.extend_from_slice(&(input.bytes.len() as u32).to_be_bytes());
            payload.extend_from_slice(&input.bytes);
        }
        debug_assert_eq!(payload.len(), encoded_len);
        Ok(payload)
    }

    pub fn into_transfer(self) -> PeerResult<ComputeTransferEncoder> {
        let payload = self.encode()?;
        ComputeTransferEncoder::new(
            ComputeTransferKind::PartitionRequest,
            self.job_id,
            self.plan_digest,
            self.partition_index,
            payload,
        )
    }
}

impl fmt::Debug for PartitionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PartitionRequest")
            .field("job_id", &"[redacted]")
            .field("partition_index", &self.partition_index)
            .field("lease_id", &"[redacted]")
            .field("peer", &self.peer)
            .field("input_count", &self.inputs.len())
            .field("inputs", &"[redacted]")
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
struct UnitRange {
    start: usize,
    end: usize,
}

/// Immutable deterministic partition plan for a bounded independent batch.
pub struct ComputePlan {
    job_id: [u8; 16],
    job: DelegatedJobKind,
    resource_id: [u8; 32],
    units: Vec<ComputeUnitSpec>,
    ranges: Vec<UnitRange>,
    digest: [u8; 32],
    total_input_bytes: u64,
    total_output_bytes: u64,
}

impl ComputePlan {
    pub fn new(
        job: DelegatedJobKind,
        resource_id: [u8; 32],
        units: Vec<ComputeUnitSpec>,
        partition_count: u16,
    ) -> PeerResult<Self> {
        let partition_count = usize::from(partition_count);
        if units.is_empty()
            || units.len() > MAX_COMPUTE_UNITS
            || partition_count == 0
            || partition_count > MAX_COMPUTE_PARTITIONS
            || partition_count > units.len()
        {
            return Err(PeerError::InvalidComputeJob);
        }
        for (expected, unit) in units.iter().enumerate() {
            if usize::try_from(unit.index).ok() != Some(expected)
                || unit.input_bytes == 0
                || unit.maximum_output_bytes == 0
            {
                return Err(PeerError::InvalidComputeJob);
            }
        }
        let total_input_bytes = units.iter().try_fold(0_u64, |sum, unit| {
            sum.checked_add(u64::from(unit.input_bytes))
        });
        let total_output_bytes = units.iter().try_fold(0_u64, |sum, unit| {
            sum.checked_add(u64::from(unit.maximum_output_bytes))
        });
        let (Some(total_input_bytes), Some(total_output_bytes)) =
            (total_input_bytes, total_output_bytes)
        else {
            return Err(PeerError::InvalidComputeJob);
        };
        if total_input_bytes > MAX_COMPUTE_BYTES || total_output_bytes > MAX_COMPUTE_BYTES {
            return Err(PeerError::InvalidComputeJob);
        }

        let mut job_id = [0_u8; 16];
        getrandom::fill(&mut job_id).map_err(|_| PeerError::RandomnessUnavailable)?;
        let base = units.len() / partition_count;
        let remainder = units.len() % partition_count;
        let mut cursor = 0;
        let ranges = (0..partition_count)
            .map(|index| {
                let count = base + usize::from(index < remainder);
                let range = UnitRange {
                    start: cursor,
                    end: cursor + count,
                };
                cursor += count;
                range
            })
            .collect::<Vec<_>>();
        let digest = plan_digest(job, resource_id, &units, &ranges);
        Ok(Self {
            job_id,
            job,
            resource_id,
            units,
            ranges,
            digest,
            total_input_bytes,
            total_output_bytes,
        })
    }

    pub fn job_id(&self) -> &[u8; 16] {
        &self.job_id
    }

    pub fn job(&self) -> DelegatedJobKind {
        self.job
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    pub fn partition_count(&self) -> usize {
        self.ranges.len()
    }

    pub fn partition_unit_range(&self, partition_index: u16) -> PeerResult<std::ops::Range<u32>> {
        let range = self
            .ranges
            .get(usize::from(partition_index))
            .ok_or(PeerError::InvalidComputeJob)?;
        Ok((range.start as u32)..(range.end as u32))
    }

    /// Create a receiver bound from the exact registered partition. This
    /// limits reassembly allocation to the maximum encoded request or result
    /// for this plan before any remote bytes are accumulated.
    pub fn transfer_expectation(
        &self,
        kind: ComputeTransferKind,
        partition_index: u16,
    ) -> PeerResult<ComputeTransferExpectation> {
        let range = *self
            .ranges
            .get(usize::from(partition_index))
            .ok_or(PeerError::InvalidComputeJob)?;
        let mut maximum_payload_bytes = match kind {
            ComputeTransferKind::PartitionRequest => COMPUTE_REQUEST_HEADER_BYTES,
            ComputeTransferKind::PartitionResult => COMPUTE_RESULT_HEADER_BYTES,
        };
        for unit in &self.units[range.start..range.end] {
            let (record_bytes, data_bytes) = match kind {
                ComputeTransferKind::PartitionRequest => {
                    (8_usize, usize::try_from(unit.input_bytes).ok())
                }
                ComputeTransferKind::PartitionResult => {
                    (40_usize, usize::try_from(unit.maximum_output_bytes).ok())
                }
            };
            maximum_payload_bytes = maximum_payload_bytes
                .checked_add(record_bytes)
                .and_then(|total| total.checked_add(data_bytes?))
                .ok_or(PeerError::InvalidComputeJob)?;
        }
        if maximum_payload_bytes > MAX_COMPUTE_PAYLOAD_BYTES {
            return Err(PeerError::InvalidComputeJob);
        }
        ComputeTransferExpectation::new(
            kind,
            self.job_id,
            self.digest,
            partition_index,
            maximum_payload_bytes,
        )
    }

    /// Bind exact unit payloads to a donor-signed grant. The remote donor must
    /// independently enforce its live lease book before execution; the local
    /// copy cannot observe revocation or consume remote quotas.
    pub fn make_partition_request(
        &self,
        grant: &PeerLeaseGrant,
        expected_owner: &PublicPeerIdentity,
        recipient: PeerId,
        partition_index: u16,
        inputs: Vec<UnitInput>,
        now_unix_seconds: u64,
    ) -> PeerResult<PartitionRequest> {
        grant.verify(expected_owner, recipient, now_unix_seconds)?;
        let partition = self
            .ranges
            .get(usize::from(partition_index))
            .ok_or(PeerError::InvalidComputeJob)?;
        if !grant.scopes().any(|scope| {
            scope.job == self.job
                && scope.resource_id == self.resource_id
                && scope.disclosure == DisclosureScope::ExplicitJobInputAndResult
        }) || inputs.len() != partition.end - partition.start
        {
            return Err(PeerError::LeaseDenied);
        }
        let mut input_total = 0_u64;
        let mut output_total = 0_u64;
        for (offset, input) in inputs.iter().enumerate() {
            let spec = &self.units[partition.start + offset];
            let actual_bytes = input.bytes.len() as u64;
            if input.index != spec.index
                || actual_bytes != u64::from(spec.input_bytes)
                || Sha256::digest(&input.bytes).as_slice() != spec.input_digest
            {
                return Err(PeerError::InvalidComputeJob);
            }
            input_total = input_total
                .checked_add(actual_bytes)
                .ok_or(PeerError::InvalidComputeJob)?;
            output_total = output_total
                .checked_add(u64::from(spec.maximum_output_bytes))
                .ok_or(PeerError::InvalidComputeJob)?;
        }
        if input_total == 0
            || input_total > grant.quotas().maximum_input_bytes_per_job
            || output_total == 0
            || output_total > grant.quotas().maximum_output_bytes_per_job
        {
            return Err(PeerError::QuotaExceeded);
        }
        Ok(PartitionRequest {
            job_id: self.job_id,
            plan_digest: self.digest,
            partition_index,
            lease_id: grant.id(),
            peer: recipient,
            job: self.job,
            resource_id: self.resource_id,
            maximum_output_bytes: output_total,
            inputs,
        })
    }

    /// Bind an untrusted wire request to the exact locally registered plan.
    /// The returned prepared value must still pass the owner's live lease book
    /// before its payload becomes visible to the worker.
    pub fn prepare_partition_request(
        &self,
        request: ReceivedPartitionRequest,
    ) -> PeerResult<PreparedPartitionRequest> {
        let range = *self
            .ranges
            .get(usize::from(request.partition_index))
            .ok_or(PeerError::InvalidComputeJob)?;
        if request.job_id != self.job_id
            || request.plan_digest != self.digest
            || request.job != self.job
            || request.resource_id != self.resource_id
            || request.inputs.len() != range.end - range.start
        {
            return Err(PeerError::InvalidComputeJob);
        }
        let mut input_bytes = 0_u64;
        let mut maximum_output_bytes = 0_u64;
        let mut maximum_output_bytes_per_unit = Vec::new();
        maximum_output_bytes_per_unit
            .try_reserve_exact(request.inputs.len())
            .map_err(|_| PeerError::InvalidComputeJob)?;
        for (offset, input) in request.inputs.iter().enumerate() {
            let spec = &self.units[range.start + offset];
            let length = input.bytes.len() as u64;
            if input.index != spec.index
                || length != u64::from(spec.input_bytes)
                || Sha256::digest(&input.bytes).as_slice() != spec.input_digest
            {
                return Err(PeerError::InvalidComputeJob);
            }
            input_bytes = input_bytes
                .checked_add(length)
                .ok_or(PeerError::InvalidComputeJob)?;
            maximum_output_bytes = maximum_output_bytes
                .checked_add(u64::from(spec.maximum_output_bytes))
                .ok_or(PeerError::InvalidComputeJob)?;
            maximum_output_bytes_per_unit.push(spec.maximum_output_bytes);
        }
        if input_bytes == 0
            || input_bytes > MAX_COMPUTE_BYTES
            || maximum_output_bytes == 0
            || maximum_output_bytes > MAX_COMPUTE_BYTES
            || request.maximum_output_bytes != maximum_output_bytes
            || maximum_result_data_bytes(range.end - range.start)
                .is_none_or(|maximum| maximum_output_bytes > maximum)
        {
            return Err(PeerError::InvalidComputeJob);
        }
        Ok(PreparedPartitionRequest {
            job_id: request.job_id,
            plan_digest: request.plan_digest,
            partition_index: request.partition_index,
            lease_id: request.lease_id,
            authenticated_peer: request.authenticated_peer,
            job: request.job,
            resource_id: request.resource_id,
            input_bytes,
            maximum_output_bytes,
            maximum_output_bytes_per_unit,
            inputs: request.inputs,
        })
    }

    /// Verify exact partition membership, input binding, output bounds, and a
    /// caller-provided semantic result check before producing a checkpoint.
    pub fn verify_partition_result(
        &self,
        partition_index: u16,
        authenticated_peer: PeerId,
        result: PartitionResult,
        mut verify_unit: impl FnMut(&ComputeUnitSpec, &[u8]) -> bool,
    ) -> PeerResult<VerifiedPartition> {
        let range = *self
            .ranges
            .get(usize::from(partition_index))
            .ok_or(PeerError::InvalidComputeResult)?;
        if result.job_id != self.job_id
            || result.plan_digest != self.digest
            || result.partition_index != partition_index
            || result.outputs.len() != range.end - range.start
        {
            return Err(PeerError::InvalidComputeResult);
        }
        let mut total_output_bytes = 0_u64;
        for (offset, output) in result.outputs.iter().enumerate() {
            let spec = &self.units[range.start + offset];
            let output_bytes = output.bytes.len() as u64;
            if output.index != spec.index
                || output.input_digest != spec.input_digest
                || output_bytes > u64::from(spec.maximum_output_bytes)
            {
                return Err(PeerError::InvalidComputeResult);
            }
            if !verify_unit(spec, &output.bytes) {
                return Err(PeerError::ComputeVerificationFailed);
            }
            total_output_bytes = total_output_bytes
                .checked_add(output_bytes)
                .ok_or(PeerError::InvalidComputeResult)?;
        }
        if total_output_bytes > MAX_COMPUTE_BYTES || total_output_bytes > self.total_output_bytes {
            return Err(PeerError::InvalidComputeResult);
        }
        let result_digest =
            partition_result_digest(self.job_id, self.digest, partition_index, &result.outputs);
        Ok(VerifiedPartition {
            job_id: self.job_id,
            plan_digest: self.digest,
            partition_index,
            range,
            peer: authenticated_peer,
            result_digest,
            outputs: result.outputs,
        })
    }

    /// Revalidate a restored checkpoint before treating it as completed work.
    /// Persistence and its integrity protection belong to the owning broker.
    pub fn restore_checkpoint(
        &self,
        checkpoint: PartitionCheckpoint,
        verify_unit: impl FnMut(&ComputeUnitSpec, &[u8]) -> bool,
    ) -> PeerResult<VerifiedPartition> {
        if checkpoint.job_id != self.job_id || checkpoint.plan_digest != self.digest {
            return Err(PeerError::InvalidComputeResult);
        }
        let result = PartitionResult::new(
            checkpoint.job_id,
            checkpoint.plan_digest,
            checkpoint.partition_index,
            checkpoint.outputs,
        )?;
        let restored = self.verify_partition_result(
            checkpoint.partition_index,
            checkpoint.peer,
            result,
            verify_unit,
        )?;
        if restored.result_digest != checkpoint.result_digest {
            return Err(PeerError::InvalidComputeResult);
        }
        Ok(restored)
    }

    /// Assemble each exact partition once, in stable unit-index order.
    pub fn assemble(&self, mut partitions: Vec<VerifiedPartition>) -> PeerResult<AssembledJob> {
        if partitions.len() != self.ranges.len() {
            return Err(PeerError::InvalidComputeResult);
        }
        partitions.sort_by_key(|partition| partition.partition_index);
        let mut outputs = Vec::with_capacity(self.units.len());
        let mut result_digest = Sha256::new();
        result_digest.update(RESULT_DOMAIN);
        result_digest.update(self.job_id);
        result_digest.update(self.digest);
        for (expected_index, partition) in partitions.into_iter().enumerate() {
            let expected_range = self.ranges[expected_index];
            if partition.job_id != self.job_id
                || partition.plan_digest != self.digest
                || usize::from(partition.partition_index) != expected_index
                || partition.range.start != expected_range.start
                || partition.range.end != expected_range.end
                || partition.outputs.len() != expected_range.end - expected_range.start
                || partition_result_digest(
                    partition.job_id,
                    partition.plan_digest,
                    partition.partition_index,
                    &partition.outputs,
                ) != partition.result_digest
            {
                return Err(PeerError::InvalidComputeResult);
            }
            result_digest.update(partition.result_digest);
            outputs.extend(partition.outputs);
        }
        let total_output_bytes = outputs
            .iter()
            .map(|output| output.bytes.len() as u64)
            .sum::<u64>();
        if outputs.len() != self.units.len()
            || total_output_bytes > self.total_output_bytes
            || total_output_bytes > MAX_COMPUTE_BYTES
        {
            return Err(PeerError::InvalidComputeResult);
        }
        Ok(AssembledJob {
            job_id: self.job_id,
            plan_digest: self.digest,
            result_digest: result_digest.finalize().into(),
            outputs,
        })
    }
}

impl fmt::Debug for ComputePlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ComputePlan")
            .field("job", &self.job)
            .field("resource_id", &"[redacted]")
            .field("unit_count", &self.units.len())
            .field("partition_count", &self.ranges.len())
            .field("total_input_bytes", &self.total_input_bytes)
            .field("total_output_bytes", &self.total_output_bytes)
            .field("digest", &"[redacted]")
            .finish()
    }
}

/// Verified output kept in zeroizing memory until assembled or persisted by
/// the encrypted broker. It carries no reusable lease authority.
pub struct VerifiedPartition {
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    partition_index: u16,
    range: UnitRange,
    peer: PeerId,
    result_digest: [u8; 32],
    outputs: Vec<UnitOutput>,
}

impl VerifiedPartition {
    pub fn checkpoint(self) -> PartitionCheckpoint {
        PartitionCheckpoint {
            job_id: self.job_id,
            plan_digest: self.plan_digest,
            partition_index: self.partition_index,
            peer: self.peer,
            result_digest: self.result_digest,
            outputs: self.outputs,
        }
    }

    pub fn partition_index(&self) -> u16 {
        self.partition_index
    }

    pub fn result_digest(&self) -> &[u8; 32] {
        &self.result_digest
    }
}

impl fmt::Debug for VerifiedPartition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedPartition")
            .field("partition_index", &self.partition_index)
            .field("peer", &self.peer)
            .field("result_digest", &"[redacted]")
            .field("unit_count", &self.outputs.len())
            .field("outputs", &"[redacted]")
            .finish()
    }
}

/// A checkpoint has verified output but deliberately contains no credential,
/// lease permit, or authority to dispatch again.
pub struct PartitionCheckpoint {
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    partition_index: u16,
    peer: PeerId,
    result_digest: [u8; 32],
    outputs: Vec<UnitOutput>,
}

impl PartitionCheckpoint {
    pub fn job_id(&self) -> &[u8; 16] {
        &self.job_id
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }

    pub fn partition_index(&self) -> u16 {
        self.partition_index
    }

    pub fn peer(&self) -> PeerId {
        self.peer
    }

    pub fn result_digest(&self) -> &[u8; 32] {
        &self.result_digest
    }

    pub fn outputs(&self) -> &[UnitOutput] {
        &self.outputs
    }
}

impl fmt::Debug for PartitionCheckpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PartitionCheckpoint")
            .field("partition_index", &self.partition_index)
            .field("peer", &self.peer)
            .field("result_digest", &"[redacted]")
            .field("unit_count", &self.outputs.len())
            .field("outputs", &"[redacted]")
            .finish()
    }
}

/// Complete, deterministic result assembled only from every verified range.
pub struct AssembledJob {
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    result_digest: [u8; 32],
    outputs: Vec<UnitOutput>,
}

impl AssembledJob {
    pub fn job_id(&self) -> &[u8; 16] {
        &self.job_id
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }

    pub fn result_digest(&self) -> &[u8; 32] {
        &self.result_digest
    }

    pub fn outputs(&self) -> &[UnitOutput] {
        &self.outputs
    }
}

impl fmt::Debug for AssembledJob {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AssembledJob")
            .field("unit_count", &self.outputs.len())
            .field("result_digest", &"[redacted]")
            .field("outputs", &"[redacted]")
            .finish()
    }
}

fn plan_digest(
    job: DelegatedJobKind,
    resource_id: [u8; 32],
    units: &[ComputeUnitSpec],
    ranges: &[UnitRange],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(PLAN_DOMAIN);
    digest.update([job as u8]);
    digest.update(resource_id);
    digest.update((units.len() as u32).to_be_bytes());
    digest.update((ranges.len() as u16).to_be_bytes());
    for unit in units {
        digest.update(unit.index.to_be_bytes());
        digest.update(unit.input_digest);
        digest.update(unit.input_bytes.to_be_bytes());
        digest.update(unit.maximum_output_bytes.to_be_bytes());
    }
    for range in ranges {
        digest.update((range.start as u32).to_be_bytes());
        digest.update((range.end as u32).to_be_bytes());
    }
    digest.finalize().into()
}

fn partition_result_digest(
    job_id: [u8; 16],
    plan_digest: [u8; 32],
    partition_index: u16,
    outputs: &[UnitOutput],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(RESULT_DOMAIN);
    digest.update(job_id);
    digest.update(plan_digest);
    digest.update(partition_index.to_be_bytes());
    digest.update((outputs.len() as u32).to_be_bytes());
    for output in outputs {
        digest.update(output.index.to_be_bytes());
        digest.update(output.input_digest);
        digest.update((output.bytes.len() as u64).to_be_bytes());
        digest.update(&output.bytes);
    }
    digest.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceIdentity, PeerLease, PeerLeaseGrant, PeerLeaseQuotas};

    fn plan(unit_count: usize, partition_count: u16) -> (ComputePlan, Vec<Vec<u8>>) {
        let inputs = (0..unit_count)
            .map(|index| vec![index as u8 + 1; index % 5 + 1])
            .collect::<Vec<_>>();
        let specs = inputs
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                ComputeUnitSpec::new(
                    index as u32,
                    Sha256::digest(bytes).into(),
                    bytes.len() as u32,
                    16,
                )
                .unwrap()
            })
            .collect();
        (
            ComputePlan::new(
                DelegatedJobKind::BatchAnalysis,
                [44; 32],
                specs,
                partition_count,
            )
            .unwrap(),
            inputs,
        )
    }

    fn make_grant(
        plan: &ComputePlan,
        input_bytes: u64,
        output_bytes: u64,
    ) -> (PeerLeaseGrant, DeviceIdentity, DeviceIdentity) {
        let owner = DeviceIdentity::from_seed([11; 32]);
        let peer = DeviceIdentity::from_seed([22; 32]);
        let lease = PeerLease::issue(
            &owner,
            peer.peer_id(),
            100,
            120,
            [crate::LeaseScope {
                job: plan.job,
                resource_id: plan.resource_id,
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 2,
                maximum_jobs: 4,
                maximum_input_bytes_per_job: input_bytes,
                maximum_output_bytes_per_job: output_bytes,
                maximum_total_input_bytes: input_bytes,
                maximum_total_output_bytes: output_bytes,
            },
        )
        .unwrap();
        (lease.grant(), owner, peer)
    }

    fn outputs(plan: &ComputePlan, partition_index: u16, inputs: &[Vec<u8>]) -> Vec<UnitOutput> {
        let range = plan.ranges[usize::from(partition_index)];
        inputs[range.start..range.end]
            .iter()
            .enumerate()
            .map(|(offset, bytes)| {
                let index = (range.start + offset) as u32;
                UnitOutput::new(index, Sha256::digest(bytes).into(), bytes.clone()).unwrap()
            })
            .collect()
    }

    #[test]
    fn splits_units_into_balanced_contiguous_deterministic_ranges() {
        let (plan, _) = plan(10, 3);
        assert_eq!(plan.partition_unit_range(0).unwrap(), 0..4);
        assert_eq!(plan.partition_unit_range(1).unwrap(), 4..7);
        assert_eq!(plan.partition_unit_range(2).unwrap(), 7..10);
        assert_eq!(
            plan.partition_unit_range(3),
            Err(PeerError::InvalidComputeJob)
        );
    }

    #[test]
    fn dispatch_binds_exact_scope_and_input_bytes_to_lease_limits() {
        let (plan, inputs) = plan(4, 2);
        let (grant, owner, recipient) = make_grant(&plan, 64, 64);
        let request = plan
            .make_partition_request(
                &grant,
                &owner.public_identity(),
                recipient.peer_id(),
                0,
                inputs[..2]
                    .iter()
                    .enumerate()
                    .map(|(index, bytes)| UnitInput::new(index as u32, bytes.clone()).unwrap())
                    .collect(),
                101,
            )
            .unwrap();
        assert_eq!(request.inputs().len(), 2);
        assert_eq!(request.peer(), recipient.peer_id());
        assert_eq!(request.plan_digest(), plan.plan_digest());

        let mut wrong = inputs[..2]
            .iter()
            .enumerate()
            .map(|(index, bytes)| UnitInput::new(index as u32, bytes.clone()).unwrap())
            .collect::<Vec<_>>();
        wrong.swap(0, 1);
        assert_eq!(
            plan.make_partition_request(
                &make_grant(&plan, 64, 64).0,
                &owner.public_identity(),
                recipient.peer_id(),
                0,
                wrong,
                101,
            )
            .err(),
            Some(PeerError::InvalidComputeJob)
        );

        let (small_grant, owner, recipient) = make_grant(&plan, 1, 64);
        let exact_inputs = inputs[..2]
            .iter()
            .enumerate()
            .map(|(index, bytes)| UnitInput::new(index as u32, bytes.clone()).unwrap())
            .collect();
        assert_eq!(
            plan.make_partition_request(
                &small_grant,
                &owner.public_identity(),
                recipient.peer_id(),
                0,
                exact_inputs,
                101,
            )
            .err(),
            Some(PeerError::QuotaExceeded)
        );
    }

    #[test]
    fn request_codec_holds_inputs_until_plan_and_live_lease_admission() {
        let (plan, input_bytes) = plan(4, 2);
        let owner = DeviceIdentity::from_seed([31; 32]);
        let requester = DeviceIdentity::from_seed([47; 32]);
        let lease = PeerLease::issue(
            &owner,
            requester.peer_id(),
            100,
            120,
            [crate::LeaseScope {
                job: plan.job,
                resource_id: plan.resource_id,
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 1,
                maximum_jobs: 2,
                maximum_input_bytes_per_job: 8,
                maximum_output_bytes_per_job: 32,
                maximum_total_input_bytes: 8,
                maximum_total_output_bytes: 32,
            },
        )
        .unwrap();
        let grant = lease.grant();
        let mut lease_book = crate::PeerLeaseBook::new(owner.peer_id());
        let lease_id = lease_book.insert(lease).unwrap();
        assert_eq!(lease_id, grant.id());

        let inputs = input_bytes[..2]
            .iter()
            .enumerate()
            .map(|(index, bytes)| UnitInput::new(index as u32, bytes.clone()).unwrap())
            .collect();
        let request = plan
            .make_partition_request(
                &grant,
                &owner.public_identity(),
                requester.peer_id(),
                0,
                inputs,
                101,
            )
            .unwrap();
        let payload = request.encode().unwrap();
        assert_eq!(
            ReceivedPartitionRequest::decode(&payload, owner.peer_id()).err(),
            Some(PeerError::AuthenticationFailed)
        );
        let mut wrong_resource = payload.to_vec();
        wrong_resource[106] ^= 1;
        let wrong_resource =
            ReceivedPartitionRequest::decode(&wrong_resource, requester.peer_id()).unwrap();
        assert_eq!(
            plan.prepare_partition_request(wrong_resource).err(),
            Some(PeerError::InvalidComputeJob)
        );
        let received = ReceivedPartitionRequest::decode(&payload, requester.peer_id()).unwrap();
        assert_eq!(received.job_id(), plan.job_id());
        assert_eq!(received.plan_digest(), plan.plan_digest());
        assert_eq!(received.input_bytes(), 3);

        let mismatched = plan
            .prepare_partition_request(
                ReceivedPartitionRequest::decode(&payload, requester.peer_id()).unwrap(),
            )
            .unwrap();
        let mut wrong_identity_request = mismatched.lease_request();
        wrong_identity_request.job_id = [0xee; 16];
        let wrong_permit = lease_book
            .reserve(lease_id, requester.peer_id(), wrong_identity_request, 101)
            .unwrap();
        let wrong_proof = lease_book.authorize_dispatch(&wrong_permit, 101).unwrap();
        assert_eq!(
            mismatched.authorize(wrong_proof).err(),
            Some(PeerError::LeaseDenied)
        );
        assert!(lease_book.settle(wrong_permit, 0).unwrap().dispatched);

        let prepared = plan.prepare_partition_request(received).unwrap();
        assert_eq!(prepared.lease_id(), lease_id);
        let permit = lease_book
            .reserve(lease_id, requester.peer_id(), prepared.lease_request(), 101)
            .unwrap();
        let permitted = lease_book.authorize_dispatch(&permit, 101).unwrap();
        let authorized = prepared.authorize(permitted).unwrap();
        assert_eq!(authorized.peer(), requester.peer_id());
        assert_eq!(authorized.inputs().len(), 2);
        assert_eq!(authorized.inputs()[1].bytes(), input_bytes[1]);
        assert!(lease_book.settle(permit, 0).unwrap().dispatched);
    }

    #[test]
    fn request_codec_rejects_output_envelopes_that_exceed_transfer_limit() {
        let input = b"x";
        let output_limit = maximum_result_data_bytes(1).unwrap();
        let spec = ComputeUnitSpec::new(
            0,
            Sha256::digest(input).into(),
            input.len() as u32,
            u32::try_from(output_limit + 1).unwrap(),
        )
        .unwrap();
        let plan =
            ComputePlan::new(DelegatedJobKind::BatchAnalysis, [44; 32], vec![spec], 1).unwrap();
        let owner = DeviceIdentity::from_seed([61; 32]);
        let requester = DeviceIdentity::from_seed([73; 32]);
        let output_bytes = output_limit + 1;
        let lease = PeerLease::issue(
            &owner,
            requester.peer_id(),
            100,
            120,
            [crate::LeaseScope {
                job: plan.job,
                resource_id: plan.resource_id,
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 1,
                maximum_jobs: 1,
                maximum_input_bytes_per_job: 1,
                maximum_output_bytes_per_job: output_bytes,
                maximum_total_input_bytes: 1,
                maximum_total_output_bytes: output_bytes,
            },
        )
        .unwrap();
        let request = plan
            .make_partition_request(
                &lease.grant(),
                &owner.public_identity(),
                requester.peer_id(),
                0,
                vec![UnitInput::new(0, input.to_vec()).unwrap()],
                101,
            )
            .unwrap();
        assert_eq!(request.encode().err(), Some(PeerError::InvalidComputeJob));
    }

    #[test]
    fn compute_transfers_are_bounded_ordered_and_digest_checked() {
        let payload_len = COMPUTE_CHUNK_DATA_BYTES * 2 + 37;
        let original = Zeroizing::new(vec![0x5a; payload_len]);
        let mut encoder = ComputeTransferEncoder::new(
            ComputeTransferKind::PartitionRequest,
            [7; 16],
            [9; 32],
            3,
            original.clone(),
        )
        .unwrap();
        assert_eq!(encoder.message_kind(), PeerMessageKind::JobOffer);
        assert_eq!(encoder.chunk_count(), 3);
        let mut frames = Vec::new();
        while let Some(frame) = encoder.next_frame().unwrap() {
            assert!(frame.len() <= super::super::MAX_PEER_MESSAGE_BYTES);
            frames.push(frame);
        }
        assert!(encoder.next_frame().unwrap().is_none());

        let expectation = ComputeTransferExpectation::new(
            ComputeTransferKind::PartitionRequest,
            [7; 16],
            [9; 32],
            3,
            payload_len,
        )
        .unwrap();
        let mut assembler = ComputeTransferAssembler::new(expectation);
        for (index, frame) in frames.iter().enumerate() {
            let result = assembler.push(PeerMessageKind::JobOffer, frame).unwrap();
            if index + 1 == frames.len() {
                assert_eq!(result.unwrap().as_slice(), original.as_slice());
            } else {
                assert!(result.is_none());
            }
        }

        let mut out_of_order = ComputeTransferAssembler::new(expectation);
        assert_eq!(
            out_of_order
                .push(PeerMessageKind::JobOffer, &frames[1])
                .err(),
            Some(PeerError::ReplayOrReordering)
        );
        assert_eq!(
            out_of_order
                .push(PeerMessageKind::JobOffer, &frames[0])
                .err(),
            Some(PeerError::InvalidFrame)
        );

        let bounded = ComputeTransferExpectation::new(
            ComputeTransferKind::PartitionRequest,
            [7; 16],
            [9; 32],
            3,
            payload_len - 1,
        )
        .unwrap();
        let mut over_limit = ComputeTransferAssembler::new(bounded);
        assert_eq!(
            over_limit.push(PeerMessageKind::JobOffer, &frames[0]).err(),
            Some(PeerError::InvalidFrame)
        );

        let mut corrupted_frames = frames;
        corrupted_frames[0][COMPUTE_CHUNK_HEADER_BYTES] ^= 1;
        let mut corrupted = ComputeTransferAssembler::new(expectation);
        assert!(
            corrupted
                .push(PeerMessageKind::JobOffer, &corrupted_frames[0])
                .unwrap()
                .is_none()
        );
        assert!(
            corrupted
                .push(PeerMessageKind::JobOffer, &corrupted_frames[1])
                .unwrap()
                .is_none()
        );
        assert_eq!(
            corrupted
                .push(PeerMessageKind::JobOffer, &corrupted_frames[2])
                .err(),
            Some(PeerError::AuthenticationFailed)
        );

        let mut wrong_message_kind = ComputeTransferAssembler::new(expectation);
        assert_eq!(
            wrong_message_kind
                .push(PeerMessageKind::JobReceipt, &frames_for_kind().unwrap())
                .err(),
            Some(PeerError::InvalidFrame)
        );
    }

    fn frames_for_kind() -> PeerResult<Zeroizing<Vec<u8>>> {
        ComputeTransferEncoder::new(
            ComputeTransferKind::PartitionRequest,
            [7; 16],
            [9; 32],
            3,
            Zeroizing::new(vec![0x5a]),
        )?
        .next_frame()?
        .ok_or(PeerError::InvalidFrame)
    }

    #[test]
    fn result_codec_round_trips_bound_outputs_and_rejects_malformed_or_oversized_data() {
        let (plan, input_bytes) = plan(4, 2);
        let result = PartitionResult::new(
            *plan.job_id(),
            *plan.plan_digest(),
            0,
            outputs(&plan, 0, &input_bytes),
        )
        .unwrap();
        let payload = result.encode().unwrap();
        let decoded = PartitionResult::decode(&payload).unwrap();
        let verified = plan
            .verify_partition_result(
                0,
                DeviceIdentity::from_seed([51; 32]).peer_id(),
                decoded,
                |spec, bytes| Sha256::digest(bytes).as_slice() == spec.input_digest,
            )
            .unwrap();
        assert_eq!(verified.outputs.len(), 2);

        assert_eq!(
            PartitionResult::decode(&payload[..payload.len() - 1]).err(),
            Some(PeerError::InvalidFrame)
        );
        let mut unknown_kind = payload.to_vec();
        unknown_kind[6] = 99;
        assert_eq!(
            PartitionResult::decode(&unknown_kind).err(),
            Some(PeerError::InvalidFrame)
        );

        let oversized = PartitionResult::new(
            *plan.job_id(),
            *plan.plan_digest(),
            0,
            vec![
                UnitOutput::new(0, [9; 32], vec![7; super::super::MAX_PEER_MESSAGE_BYTES]).unwrap(),
            ],
        )
        .unwrap();
        let mut transfer = oversized.into_transfer().unwrap();
        assert_eq!(transfer.chunk_count(), 2);
        let mut assembler = ComputeTransferAssembler::new(
            ComputeTransferExpectation::new(
                ComputeTransferKind::PartitionResult,
                *plan.job_id(),
                *plan.plan_digest(),
                0,
                MAX_COMPUTE_PAYLOAD_BYTES,
            )
            .unwrap(),
        );
        let mut assembled = None;
        while let Some(frame) = transfer.next_frame().unwrap() {
            if let Some(payload) = assembler.push(PeerMessageKind::JobReceipt, &frame).unwrap() {
                assert!(assembled.replace(payload).is_none());
            }
        }
        let decoded = PartitionResult::decode(&assembled.unwrap()).unwrap();
        assert_eq!(
            decoded.outputs[0].bytes().len(),
            super::super::MAX_PEER_MESSAGE_BYTES
        );
    }

    #[test]
    fn assembly_sorts_partitions_and_requires_semantic_verification() {
        let (plan, inputs) = plan(5, 2);
        let peer = DeviceIdentity::from_seed([33; 32]).peer_id();
        let mut verified = Vec::new();
        for partition_index in (0..2).rev() {
            let result = PartitionResult::new(
                *plan.job_id(),
                *plan.plan_digest(),
                partition_index,
                outputs(&plan, partition_index, &inputs),
            )
            .unwrap();
            verified.push(
                plan.verify_partition_result(partition_index, peer, result, |spec, bytes| {
                    Sha256::digest(bytes).as_slice() == spec.input_digest
                })
                .unwrap(),
            );
        }
        let assembled = plan.assemble(verified).unwrap();
        assert_eq!(assembled.outputs().len(), inputs.len());
        assert!(
            assembled
                .outputs()
                .iter()
                .enumerate()
                .all(|(index, output)| output.index() == index as u32
                    && output.bytes() == inputs[index])
        );
    }

    #[test]
    fn rejects_missing_duplicate_misbound_and_task_invalid_results() {
        let (plan, inputs) = plan(4, 2);
        let peer = DeviceIdentity::from_seed([33; 32]).peer_id();
        let result = || {
            PartitionResult::new(
                *plan.job_id(),
                *plan.plan_digest(),
                0,
                outputs(&plan, 0, &inputs),
            )
            .unwrap()
        };
        assert_eq!(
            plan.verify_partition_result(0, peer, result(), |_, _| false)
                .err(),
            Some(PeerError::ComputeVerificationFailed)
        );
        assert_eq!(
            plan.verify_partition_result(1, peer, result(), |_, _| true)
                .err(),
            Some(PeerError::InvalidComputeResult)
        );

        let mut wrong_job_result = result();
        wrong_job_result.job_id[0] ^= 1;
        assert_eq!(
            plan.verify_partition_result(0, peer, wrong_job_result, |_, _| true)
                .err(),
            Some(PeerError::InvalidComputeResult)
        );

        let mut bad_outputs = outputs(&plan, 0, &inputs);
        bad_outputs.swap(0, 1);
        let reordered =
            PartitionResult::new(*plan.job_id(), *plan.plan_digest(), 0, bad_outputs).unwrap();
        assert_eq!(
            plan.verify_partition_result(0, peer, reordered, |_, _| true)
                .err(),
            Some(PeerError::InvalidComputeResult)
        );
    }

    #[test]
    fn checkpoint_restore_reverifies_outputs_and_contains_no_lease_permit() {
        let (plan, inputs) = plan(4, 2);
        let peer = DeviceIdentity::from_seed([33; 32]).peer_id();
        let result = PartitionResult::new(
            *plan.job_id(),
            *plan.plan_digest(),
            0,
            outputs(&plan, 0, &inputs),
        )
        .unwrap();
        let checkpoint = plan
            .verify_partition_result(0, peer, result, |spec, bytes| {
                Sha256::digest(bytes).as_slice() == spec.input_digest
            })
            .unwrap()
            .checkpoint();
        let restored = plan
            .restore_checkpoint(checkpoint, |spec, bytes| {
                Sha256::digest(bytes).as_slice() == spec.input_digest
            })
            .unwrap();
        assert_eq!(restored.partition_index(), 0);
        assert_eq!(restored.peer, peer);
    }
}
