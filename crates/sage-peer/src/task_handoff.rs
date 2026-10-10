//! Bounded transport framing for authority-free task checkpoints.
//!
//! Frames travel only as authenticated `Checkpoint` messages over an already
//! paired `PeerSecureChannel`. This module verifies framing and peer binding;
//! the receiving Sage Core must still decode and validate the checkpoint,
//! fence ownership, resolve resources, and obtain fresh authority before work.

use std::fmt;

use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use super::{
    AuthenticatedPeerMessage, MAX_PEER_MESSAGE_BYTES, MAX_TASK_HANDOFF_ARTIFACT_BYTES,
    MAX_TASK_HANDOFF_BYTES, PeerError, PeerId, PeerMessageKind, PeerResult,
};

const HANDOFF_MAGIC: &[u8; 4] = b"SGH1";
const HANDOFF_VERSION: u16 = 1;
const RECEIPT_MAGIC: &[u8; 4] = b"SGR1";
const RECEIPT_VERSION: u16 = 1;
const RECEIPT_BYTES: usize = 4 + 2 + 16 + 16 + 32 + 32 + 32;
const HANDOFF_CHUNK_HEADER_BYTES: usize = 150;
const HANDOFF_CHUNK_DATA_BYTES: usize = MAX_PEER_MESSAGE_BYTES - HANDOFF_CHUNK_HEADER_BYTES;
const ARTIFACT_MAGIC: &[u8; 4] = b"SGA1";
const ARTIFACT_VERSION: u16 = 1;
const ARTIFACT_CHUNK_HEADER_BYTES: usize = 4 + 2 + 16 + 16 + 16 + 32 + 32 + 4 + 4 + 4 + 4 + 32;
const ARTIFACT_CHUNK_DATA_BYTES: usize = MAX_PEER_MESSAGE_BYTES - ARTIFACT_CHUNK_HEADER_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HandoffMetadata {
    transfer_id: [u8; 16],
    task_id: [u8; 16],
    source_peer: PeerId,
    destination_peer: PeerId,
    total_bytes: u32,
    chunk_count: u32,
    digest: [u8; 32],
}

/// Receiver-side expectation. The peer IDs must come from the authenticated
/// pairing record and the current local device identity.
#[derive(Debug, Clone, Copy)]
pub struct TaskHandoffExpectation {
    transfer_id: [u8; 16],
    task_id: [u8; 16],
    source_peer: PeerId,
    destination_peer: PeerId,
    maximum_payload_bytes: usize,
}

impl TaskHandoffExpectation {
    pub fn new(
        transfer_id: [u8; 16],
        task_id: [u8; 16],
        source_peer: PeerId,
        destination_peer: PeerId,
        maximum_payload_bytes: usize,
    ) -> PeerResult<Self> {
        if !valid_transfer_id(&transfer_id)
            || !valid_transfer_id(&task_id)
            || source_peer == destination_peer
            || maximum_payload_bytes == 0
            || maximum_payload_bytes > MAX_TASK_HANDOFF_BYTES
        {
            return Err(PeerError::InvalidInput);
        }
        Ok(Self {
            transfer_id,
            task_id,
            source_peer,
            destination_peer,
            maximum_payload_bytes,
        })
    }
}

/// Lazy, zeroizing encoder that retains one payload and allocates only the
/// current secure-message-sized frame.
pub struct TaskHandoffEncoder {
    metadata: HandoffMetadata,
    payload: Zeroizing<Vec<u8>>,
    next_chunk: u32,
}

impl TaskHandoffEncoder {
    pub fn new(
        transfer_id: [u8; 16],
        task_id: [u8; 16],
        source_peer: PeerId,
        destination_peer: PeerId,
        payload: Zeroizing<Vec<u8>>,
    ) -> PeerResult<Self> {
        if !valid_transfer_id(&transfer_id)
            || !valid_transfer_id(&task_id)
            || source_peer == destination_peer
            || payload.is_empty()
            || payload.len() > MAX_TASK_HANDOFF_BYTES
        {
            return Err(PeerError::InvalidFrame);
        }
        let chunk_count = payload
            .len()
            .checked_add(HANDOFF_CHUNK_DATA_BYTES - 1)
            .ok_or(PeerError::InvalidFrame)?
            / HANDOFF_CHUNK_DATA_BYTES;
        let metadata = HandoffMetadata {
            transfer_id,
            task_id,
            source_peer,
            destination_peer,
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
        PeerMessageKind::Checkpoint
    }

    pub fn chunk_count(&self) -> u32 {
        self.metadata.chunk_count
    }

    /// Identify the exact payload a receiver must durably stage before it
    /// sends a receipt. This contains no execution authority.
    pub fn receipt(&self) -> TaskHandoffReceipt {
        TaskHandoffReceipt {
            transfer_id: self.metadata.transfer_id,
            task_id: self.metadata.task_id,
            source_peer: self.metadata.source_peer,
            destination_peer: self.metadata.destination_peer,
            payload_digest: self.metadata.digest,
        }
    }

    pub fn next_frame(&mut self) -> PeerResult<Option<Zeroizing<Vec<u8>>>> {
        if self.next_chunk == self.metadata.chunk_count {
            return Ok(None);
        }
        let start = usize::try_from(self.next_chunk)
            .ok()
            .and_then(|index| index.checked_mul(HANDOFF_CHUNK_DATA_BYTES))
            .ok_or(PeerError::InvalidFrame)?;
        let end = start
            .saturating_add(HANDOFF_CHUNK_DATA_BYTES)
            .min(self.payload.len());
        let data = self
            .payload
            .get(start..end)
            .ok_or(PeerError::InvalidFrame)?;
        let frame_len = HANDOFF_CHUNK_HEADER_BYTES
            .checked_add(data.len())
            .filter(|length| *length <= MAX_PEER_MESSAGE_BYTES)
            .ok_or(PeerError::InvalidFrame)?;
        let mut frame = Zeroizing::new(Vec::new());
        frame
            .try_reserve_exact(frame_len)
            .map_err(|_| PeerError::InvalidFrame)?;
        frame.extend_from_slice(HANDOFF_MAGIC);
        frame.extend_from_slice(&HANDOFF_VERSION.to_be_bytes());
        frame.extend_from_slice(&self.metadata.transfer_id);
        frame.extend_from_slice(&self.metadata.task_id);
        frame.extend_from_slice(self.metadata.source_peer.as_bytes());
        frame.extend_from_slice(self.metadata.destination_peer.as_bytes());
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

/// Completed, zeroizing payload returned only after identity, order, size and
/// whole-checkpoint digest checks pass.
pub struct TaskHandoffPayload {
    transfer_id: [u8; 16],
    task_id: [u8; 16],
    source_peer: PeerId,
    destination_peer: PeerId,
    payload_digest: [u8; 32],
    bytes: Zeroizing<Vec<u8>>,
}

/// Exact, bounded expectation for one task-owned artifact. The descriptor
/// fields are supplied from the already validated checkpoint manifest.
#[derive(Debug, Clone, Copy)]
pub struct ArtifactHandoffExpectation {
    transfer_id: [u8; 16],
    task_id: [u8; 16],
    artifact_id: [u8; 16],
    source_peer: PeerId,
    destination_peer: PeerId,
    size_bytes: usize,
    digest: [u8; 32],
}

impl ArtifactHandoffExpectation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        transfer_id: [u8; 16],
        task_id: [u8; 16],
        artifact_id: [u8; 16],
        source_peer: PeerId,
        destination_peer: PeerId,
        size_bytes: usize,
        digest: [u8; 32],
    ) -> PeerResult<Self> {
        if !valid_transfer_id(&transfer_id)
            || !valid_transfer_id(&task_id)
            || !valid_transfer_id(&artifact_id)
            || source_peer == destination_peer
            || size_bytes > MAX_TASK_HANDOFF_ARTIFACT_BYTES
        {
            return Err(PeerError::InvalidInput);
        }
        Ok(Self {
            transfer_id,
            task_id,
            artifact_id,
            source_peer,
            destination_peer,
            size_bytes,
            digest,
        })
    }
}

/// Lazy, zeroizing encoder for a single content-addressed task artifact.
/// Each emitted frame fits one authenticated peer message.
pub struct ArtifactHandoffEncoder {
    transfer_id: [u8; 16],
    task_id: [u8; 16],
    artifact_id: [u8; 16],
    source_peer: PeerId,
    destination_peer: PeerId,
    digest: [u8; 32],
    payload: Zeroizing<Vec<u8>>,
    chunk_count: u32,
    next_chunk: u32,
}

impl ArtifactHandoffEncoder {
    pub fn new(
        transfer_id: [u8; 16],
        task_id: [u8; 16],
        artifact_id: [u8; 16],
        source_peer: PeerId,
        destination_peer: PeerId,
        payload: Zeroizing<Vec<u8>>,
    ) -> PeerResult<Self> {
        if !valid_transfer_id(&transfer_id)
            || !valid_transfer_id(&task_id)
            || !valid_transfer_id(&artifact_id)
            || source_peer == destination_peer
            || payload.len() > MAX_TASK_HANDOFF_ARTIFACT_BYTES
        {
            return Err(PeerError::InvalidFrame);
        }
        let chunks = payload
            .len()
            .checked_add(ARTIFACT_CHUNK_DATA_BYTES - 1)
            .ok_or(PeerError::InvalidFrame)?
            / ARTIFACT_CHUNK_DATA_BYTES;
        let chunk_count = u32::try_from(chunks.max(1)).map_err(|_| PeerError::InvalidFrame)?;
        let digest = Sha256::digest(&payload).into();
        Ok(Self {
            transfer_id,
            task_id,
            artifact_id,
            source_peer,
            destination_peer,
            digest,
            payload,
            chunk_count,
            next_chunk: 0,
        })
    }

    pub fn message_kind(&self) -> PeerMessageKind {
        PeerMessageKind::Artifact
    }

    pub fn artifact_id(&self) -> &[u8; 16] {
        &self.artifact_id
    }

    pub fn chunk_count(&self) -> u32 {
        self.chunk_count
    }

    pub fn next_frame(&mut self) -> PeerResult<Option<Zeroizing<Vec<u8>>>> {
        if self.next_chunk == self.chunk_count {
            return Ok(None);
        }
        let start = usize::try_from(self.next_chunk)
            .ok()
            .and_then(|index| index.checked_mul(ARTIFACT_CHUNK_DATA_BYTES))
            .ok_or(PeerError::InvalidFrame)?;
        let end = start
            .saturating_add(ARTIFACT_CHUNK_DATA_BYTES)
            .min(self.payload.len());
        let data = self
            .payload
            .get(start..end)
            .ok_or(PeerError::InvalidFrame)?;
        let frame_len = ARTIFACT_CHUNK_HEADER_BYTES
            .checked_add(data.len())
            .filter(|length| *length <= MAX_PEER_MESSAGE_BYTES)
            .ok_or(PeerError::InvalidFrame)?;
        let mut frame = Zeroizing::new(Vec::new());
        frame
            .try_reserve_exact(frame_len)
            .map_err(|_| PeerError::InvalidFrame)?;
        frame.extend_from_slice(ARTIFACT_MAGIC);
        frame.extend_from_slice(&ARTIFACT_VERSION.to_be_bytes());
        frame.extend_from_slice(&self.transfer_id);
        frame.extend_from_slice(&self.task_id);
        frame.extend_from_slice(&self.artifact_id);
        frame.extend_from_slice(self.source_peer.as_bytes());
        frame.extend_from_slice(self.destination_peer.as_bytes());
        frame.extend_from_slice(&self.next_chunk.to_be_bytes());
        frame.extend_from_slice(&self.chunk_count.to_be_bytes());
        frame.extend_from_slice(
            &u32::try_from(self.payload.len())
                .map_err(|_| PeerError::InvalidFrame)?
                .to_be_bytes(),
        );
        frame.extend_from_slice(
            &u32::try_from(data.len())
                .map_err(|_| PeerError::InvalidFrame)?
                .to_be_bytes(),
        );
        frame.extend_from_slice(&self.digest);
        frame.extend_from_slice(data);
        self.next_chunk += 1;
        Ok(Some(frame))
    }
}

/// Completed artifact content returned only after frame order, authenticated
/// peer identity, expected manifest fields and the whole-body digest pass.
pub struct ArtifactHandoffPayload {
    transfer_id: [u8; 16],
    task_id: [u8; 16],
    artifact_id: [u8; 16],
    source_peer: PeerId,
    destination_peer: PeerId,
    digest: [u8; 32],
    bytes: Zeroizing<Vec<u8>>,
}

impl ArtifactHandoffPayload {
    pub fn transfer_id(&self) -> &[u8; 16] {
        &self.transfer_id
    }

    pub fn task_id(&self) -> &[u8; 16] {
        &self.task_id
    }

    pub fn artifact_id(&self) -> &[u8; 16] {
        &self.artifact_id
    }

    pub fn source_peer(&self) -> PeerId {
        self.source_peer
    }

    pub fn destination_peer(&self) -> PeerId {
        self.destination_peer
    }

    pub fn sha256(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Ordered reassembler for one artifact. Any invalid input clears content and
/// permanently poisons this assembler instance.
pub struct ArtifactHandoffAssembler {
    expected: ArtifactHandoffExpectation,
    metadata: Option<(u32, u32, [u8; 32])>,
    next_chunk: u32,
    bytes: Zeroizing<Vec<u8>>,
    complete_or_failed: bool,
}

impl ArtifactHandoffAssembler {
    pub fn new(expected: ArtifactHandoffExpectation) -> Self {
        Self {
            expected,
            metadata: None,
            next_chunk: 0,
            bytes: Zeroizing::new(Vec::new()),
            complete_or_failed: false,
        }
    }

    pub fn push(
        &mut self,
        message: &AuthenticatedPeerMessage,
    ) -> PeerResult<Option<ArtifactHandoffPayload>> {
        let result = self.push_inner(message);
        if result.is_err() {
            self.bytes.zeroize();
            self.bytes.clear();
            self.complete_or_failed = true;
        }
        result
    }

    fn push_inner(
        &mut self,
        message: &AuthenticatedPeerMessage,
    ) -> PeerResult<Option<ArtifactHandoffPayload>> {
        let frame = message.payload();
        if self.complete_or_failed
            || message.kind() != PeerMessageKind::Artifact
            || message.source_peer() != self.expected.source_peer
            || message.destination_peer() != self.expected.destination_peer
            || frame.len() < ARTIFACT_CHUNK_HEADER_BYTES
            || frame.len() > MAX_PEER_MESSAGE_BYTES
            || !frame.starts_with(ARTIFACT_MAGIC)
        {
            return Err(PeerError::InvalidFrame);
        }
        let mut offset = ARTIFACT_MAGIC.len();
        if read_u16(frame, &mut offset)? != ARTIFACT_VERSION {
            return Err(PeerError::InvalidFrame);
        }
        let transfer_id = read_array::<16>(frame, &mut offset)?;
        let task_id = read_array::<16>(frame, &mut offset)?;
        let artifact_id = read_array::<16>(frame, &mut offset)?;
        let source_peer = PeerId::from_bytes(read_array::<32>(frame, &mut offset)?);
        let destination_peer = PeerId::from_bytes(read_array::<32>(frame, &mut offset)?);
        let chunk_index = read_u32(frame, &mut offset)?;
        let chunk_count = read_u32(frame, &mut offset)?;
        let total_bytes = read_u32(frame, &mut offset)?;
        let data_bytes = read_u32(frame, &mut offset)? as usize;
        let digest = read_array::<32>(frame, &mut offset)?;
        let expected_count = usize::try_from(total_bytes)
            .ok()
            .and_then(|bytes| bytes.checked_add(ARTIFACT_CHUNK_DATA_BYTES - 1))
            .and_then(|bytes| u32::try_from((bytes / ARTIFACT_CHUNK_DATA_BYTES).max(1)).ok());
        let start = usize::try_from(chunk_index)
            .ok()
            .and_then(|index| index.checked_mul(ARTIFACT_CHUNK_DATA_BYTES))
            .ok_or(PeerError::InvalidFrame)?;
        let expected_data_bytes = usize::try_from(total_bytes)
            .ok()
            .and_then(|total| total.checked_sub(start))
            .map(|remaining| remaining.min(ARTIFACT_CHUNK_DATA_BYTES));
        if transfer_id != self.expected.transfer_id
            || task_id != self.expected.task_id
            || artifact_id != self.expected.artifact_id
            || source_peer != self.expected.source_peer
            || destination_peer != self.expected.destination_peer
            || total_bytes as usize != self.expected.size_bytes
            || total_bytes as usize > MAX_TASK_HANDOFF_ARTIFACT_BYTES
            || digest != self.expected.digest
            || chunk_count == 0
            || Some(chunk_count) != expected_count
            || chunk_index >= chunk_count
            || Some(data_bytes) != expected_data_bytes
            || frame.len() != offset.saturating_add(data_bytes)
        {
            return Err(PeerError::InvalidFrame);
        }
        if let Some((previous_count, previous_total, previous_digest)) = self.metadata {
            if previous_count != chunk_count
                || previous_total != total_bytes
                || previous_digest != digest
                || chunk_index != self.next_chunk
            {
                return Err(PeerError::ReplayOrReordering);
            }
        } else {
            if chunk_index != 0 {
                return Err(PeerError::ReplayOrReordering);
            }
            self.bytes
                .try_reserve_exact(total_bytes as usize)
                .map_err(|_| PeerError::InvalidFrame)?;
            self.metadata = Some((chunk_count, total_bytes, digest));
        }
        let end = offset
            .checked_add(data_bytes)
            .ok_or(PeerError::InvalidFrame)?;
        self.bytes
            .extend_from_slice(frame.get(offset..end).ok_or(PeerError::InvalidFrame)?);
        self.next_chunk += 1;
        if self.next_chunk < chunk_count {
            return Ok(None);
        }
        if self.bytes.len() != total_bytes as usize
            || Sha256::digest(&self.bytes).as_slice() != digest
        {
            return Err(PeerError::AuthenticationFailed);
        }
        self.complete_or_failed = true;
        Ok(Some(ArtifactHandoffPayload {
            transfer_id,
            task_id,
            artifact_id,
            source_peer,
            destination_peer,
            digest,
            bytes: std::mem::take(&mut self.bytes),
        }))
    }
}

impl TaskHandoffPayload {
    pub fn transfer_id(&self) -> &[u8; 16] {
        &self.transfer_id
    }

    pub fn task_id(&self) -> &[u8; 16] {
        &self.task_id
    }

    pub fn source_peer(&self) -> PeerId {
        self.source_peer
    }

    pub fn destination_peer(&self) -> PeerId {
        self.destination_peer
    }

    pub fn receipt(&self) -> TaskHandoffReceipt {
        TaskHandoffReceipt {
            transfer_id: self.transfer_id,
            task_id: self.task_id,
            source_peer: self.source_peer,
            destination_peer: self.destination_peer,
            payload_digest: self.payload_digest,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Proof-shaped acknowledgement of durable staging, bound to both peer
/// identities and the exact whole-checkpoint digest. Its transport must be
/// authenticated as `destination_peer` before the source accepts it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TaskHandoffReceipt {
    transfer_id: [u8; 16],
    task_id: [u8; 16],
    source_peer: PeerId,
    destination_peer: PeerId,
    payload_digest: [u8; 32],
}

impl TaskHandoffReceipt {
    /// Construct the acknowledgement for the exact durable checkpoint. The
    /// caller must only expose it after validating/staging the payload and all
    /// artifact bodies named by that checkpoint.
    pub fn new(
        transfer_id: [u8; 16],
        task_id: [u8; 16],
        source_peer: PeerId,
        destination_peer: PeerId,
        payload_digest: [u8; 32],
    ) -> PeerResult<Self> {
        if !valid_transfer_id(&transfer_id)
            || !valid_transfer_id(&task_id)
            || source_peer == destination_peer
        {
            return Err(PeerError::InvalidInput);
        }
        Ok(Self {
            transfer_id,
            task_id,
            source_peer,
            destination_peer,
            payload_digest,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(RECEIPT_BYTES);
        bytes.extend_from_slice(RECEIPT_MAGIC);
        bytes.extend_from_slice(&RECEIPT_VERSION.to_be_bytes());
        bytes.extend_from_slice(&self.transfer_id);
        bytes.extend_from_slice(&self.task_id);
        bytes.extend_from_slice(self.source_peer.as_bytes());
        bytes.extend_from_slice(self.destination_peer.as_bytes());
        bytes.extend_from_slice(&self.payload_digest);
        bytes
    }

    pub fn decode(bytes: &[u8]) -> PeerResult<Self> {
        if bytes.len() != RECEIPT_BYTES || !bytes.starts_with(RECEIPT_MAGIC) {
            return Err(PeerError::InvalidFrame);
        }
        let mut offset = RECEIPT_MAGIC.len();
        if read_u16(bytes, &mut offset)? != RECEIPT_VERSION {
            return Err(PeerError::InvalidFrame);
        }
        let receipt = Self {
            transfer_id: read_array(bytes, &mut offset)?,
            task_id: read_array(bytes, &mut offset)?,
            source_peer: PeerId::from_bytes(read_array(bytes, &mut offset)?),
            destination_peer: PeerId::from_bytes(read_array(bytes, &mut offset)?),
            payload_digest: read_array(bytes, &mut offset)?,
        };
        if offset != bytes.len()
            || !valid_transfer_id(&receipt.transfer_id)
            || !valid_transfer_id(&receipt.task_id)
            || receipt.source_peer == receipt.destination_peer
        {
            return Err(PeerError::InvalidFrame);
        }
        Ok(receipt)
    }

    pub fn transfer_id(&self) -> &[u8; 16] {
        &self.transfer_id
    }

    pub fn task_id(&self) -> &[u8; 16] {
        &self.task_id
    }

    pub fn source_peer(&self) -> PeerId {
        self.source_peer
    }

    pub fn destination_peer(&self) -> PeerId {
        self.destination_peer
    }

    pub fn payload_digest(&self) -> &[u8; 32] {
        &self.payload_digest
    }

    pub fn is_bound_to_channel(&self, authenticated_peer: PeerId, local_peer: PeerId) -> bool {
        self.destination_peer == authenticated_peer && self.source_peer == local_peer
    }
}

impl fmt::Debug for TaskHandoffReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TaskHandoffReceipt")
            .field("transfer_id", &"[redacted]")
            .field("task_id", &self.task_id)
            .field("source_peer", &self.source_peer)
            .field("destination_peer", &self.destination_peer)
            .field("payload_digest", &"[redacted]")
            .finish()
    }
}

impl fmt::Debug for TaskHandoffPayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TaskHandoffPayload")
            .field("transfer_id", &"[redacted]")
            .field("task_id", &self.task_id)
            .field("source_peer", &self.source_peer)
            .field("destination_peer", &self.destination_peer)
            .field("payload_bytes", &self.bytes.len())
            .field("payload", &"[redacted]")
            .finish()
    }
}

/// Strict ordered reassembler. A malformed frame poisons the instance and
/// clears accumulated checkpoint bytes.
pub struct TaskHandoffAssembler {
    expected: TaskHandoffExpectation,
    metadata: Option<HandoffMetadata>,
    next_chunk: u32,
    payload: Zeroizing<Vec<u8>>,
    complete_or_failed: bool,
}

impl TaskHandoffAssembler {
    pub fn new(expected: TaskHandoffExpectation) -> Self {
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
        message: &AuthenticatedPeerMessage,
    ) -> PeerResult<Option<TaskHandoffPayload>> {
        let result = self.push_inner(message);
        if result.is_err() {
            self.payload.zeroize();
            self.payload.clear();
            self.complete_or_failed = true;
        }
        result
    }

    fn push_inner(
        &mut self,
        message: &AuthenticatedPeerMessage,
    ) -> PeerResult<Option<TaskHandoffPayload>> {
        let frame = message.payload();
        if self.complete_or_failed
            || message.kind() != PeerMessageKind::Checkpoint
            || message.source_peer() != self.expected.source_peer
            || message.destination_peer() != self.expected.destination_peer
            || frame.len() < HANDOFF_CHUNK_HEADER_BYTES
            || frame.len() > MAX_PEER_MESSAGE_BYTES
            || !frame.starts_with(HANDOFF_MAGIC)
        {
            return Err(PeerError::InvalidFrame);
        }
        let mut offset = HANDOFF_MAGIC.len();
        if read_u16(frame, &mut offset)? != HANDOFF_VERSION {
            return Err(PeerError::InvalidFrame);
        }
        let transfer_id = read_array::<16>(frame, &mut offset)?;
        let task_id = read_array::<16>(frame, &mut offset)?;
        let source_peer = PeerId::from_bytes(read_array::<32>(frame, &mut offset)?);
        let destination_peer = PeerId::from_bytes(read_array::<32>(frame, &mut offset)?);
        let chunk_index = read_u32(frame, &mut offset)?;
        let chunk_count = read_u32(frame, &mut offset)?;
        let total_bytes = read_u32(frame, &mut offset)?;
        let data_bytes = read_u32(frame, &mut offset)? as usize;
        let digest = read_array::<32>(frame, &mut offset)?;
        let metadata = HandoffMetadata {
            transfer_id,
            task_id,
            source_peer,
            destination_peer,
            total_bytes,
            chunk_count,
            digest,
        };
        let expected_count = usize::try_from(total_bytes)
            .ok()
            .and_then(|bytes| bytes.checked_add(HANDOFF_CHUNK_DATA_BYTES - 1))
            .map(|bytes| bytes / HANDOFF_CHUNK_DATA_BYTES)
            .and_then(|count| u32::try_from(count).ok());
        let start = usize::try_from(chunk_index)
            .ok()
            .and_then(|index| index.checked_mul(HANDOFF_CHUNK_DATA_BYTES))
            .ok_or(PeerError::InvalidFrame)?;
        let expected_data_bytes = usize::try_from(total_bytes)
            .ok()
            .and_then(|total| total.checked_sub(start))
            .map(|remaining| remaining.min(HANDOFF_CHUNK_DATA_BYTES));
        if metadata.transfer_id != self.expected.transfer_id
            || metadata.task_id != self.expected.task_id
            || metadata.source_peer != self.expected.source_peer
            || metadata.destination_peer != self.expected.destination_peer
            || total_bytes == 0
            || total_bytes as usize > self.expected.maximum_payload_bytes
            || total_bytes as usize > MAX_TASK_HANDOFF_BYTES
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
        Ok(Some(TaskHandoffPayload {
            transfer_id: metadata.transfer_id,
            task_id: metadata.task_id,
            source_peer: metadata.source_peer,
            destination_peer: metadata.destination_peer,
            payload_digest: metadata.digest,
            bytes: std::mem::take(&mut self.payload),
        }))
    }
}

fn valid_transfer_id(value: &[u8; 16]) -> bool {
    value.iter().any(|byte| *byte != 0)
}

fn read_array<const N: usize>(bytes: &[u8], offset: &mut usize) -> PeerResult<[u8; N]> {
    let end = offset.checked_add(N).ok_or(PeerError::InvalidFrame)?;
    let slice = bytes.get(*offset..end).ok_or(PeerError::InvalidFrame)?;
    *offset = end;
    slice.try_into().map_err(|_| PeerError::InvalidFrame)
}

fn read_u16(bytes: &[u8], offset: &mut usize) -> PeerResult<u16> {
    Ok(u16::from_be_bytes(read_array(bytes, offset)?))
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> PeerResult<u32> {
    Ok(u32::from_be_bytes(read_array(bytes, offset)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DeviceIdentity, EncryptedPeerFrame, PairingConfirmation, PairingQr, PairingResponse,
        create_pairing_invitation,
    };
    use std::time::Duration;

    fn paired() -> (crate::PairingOutcome, crate::PairingOutcome) {
        let inviter = DeviceIdentity::from_seed([31; 32]);
        let responder = DeviceIdentity::from_seed([47; 32]);
        let (qr, pending_inviter) =
            create_pairing_invitation(&inviter, 1_800_000_000, Duration::from_secs(120)).unwrap();
        let decoded_qr = PairingQr::from_payload(&qr.to_payload()).unwrap();
        let (response, pending_responder) = decoded_qr
            .respond_after_local_confirmation(&responder, 1_800_000_010)
            .unwrap();
        let response = PairingResponse::decode(&response.encode()).unwrap();
        let code = pending_inviter.authentication_code(&response).unwrap();
        let (confirmation, inviter_outcome) = pending_inviter
            .confirm_after_local_confirmation(&inviter, &response, &code, 1_800_000_011)
            .unwrap();
        let confirmation = PairingConfirmation::decode(&confirmation.encode()).unwrap();
        let responder_outcome = pending_responder
            .finish_after_local_confirmation(&confirmation, &code, 1_800_000_012)
            .unwrap();
        (inviter_outcome, responder_outcome)
    }

    fn transfer_ids() -> ([u8; 16], [u8; 16]) {
        ([1; 16], [2; 16])
    }

    fn authenticated_message(
        source: &mut crate::PairingOutcome,
        destination: &mut crate::PairingOutcome,
        kind: PeerMessageKind,
        payload: &[u8],
    ) -> AuthenticatedPeerMessage {
        let encrypted = source.channel.seal(kind, payload).unwrap();
        let encrypted = EncryptedPeerFrame::decode(&encrypted.encode().unwrap()).unwrap();
        destination.channel.open_authenticated(&encrypted).unwrap()
    }

    #[test]
    fn artifact_payload_moves_in_authenticated_manifest_bound_chunks() {
        let (mut source, mut destination) = paired();
        let (transfer_id, task_id) = transfer_ids();
        let artifact_id = [7; 16];
        let payload = Zeroizing::new(vec![0x6d; ARTIFACT_CHUNK_DATA_BYTES + 211]);
        let mut encoder = ArtifactHandoffEncoder::new(
            transfer_id,
            task_id,
            artifact_id,
            source.local_peer_id,
            destination.local_peer_id,
            payload.clone(),
        )
        .unwrap();
        let expectation = ArtifactHandoffExpectation::new(
            transfer_id,
            task_id,
            artifact_id,
            source.local_peer_id,
            destination.local_peer_id,
            payload.len(),
            Sha256::digest(&payload).into(),
        )
        .unwrap();
        let mut assembler = ArtifactHandoffAssembler::new(expectation);
        assert_eq!(encoder.message_kind(), PeerMessageKind::Artifact);
        assert_eq!(encoder.chunk_count(), 2);
        let mut assembled = None;
        while let Some(frame) = encoder.next_frame().unwrap() {
            let message = authenticated_message(
                &mut source,
                &mut destination,
                PeerMessageKind::Artifact,
                &frame,
            );
            let next = assembler.push(&message).unwrap();
            if next.is_some() {
                assembled = next;
            }
        }
        let assembled = assembled.expect("all artifact chunks assemble");
        assert_eq!(assembled.transfer_id(), &transfer_id);
        assert_eq!(assembled.task_id(), &task_id);
        assert_eq!(assembled.artifact_id(), &artifact_id);
        assert_eq!(assembled.source_peer(), source.local_peer_id);
        assert_eq!(assembled.destination_peer(), destination.local_peer_id);
        assert_eq!(assembled.as_bytes(), payload.as_slice());
        let expected_digest: [u8; 32] = Sha256::digest(&payload).into();
        assert_eq!(assembled.sha256(), &expected_digest);
        assert!(encoder.next_frame().unwrap().is_none());
    }

    #[test]
    fn empty_artifact_has_one_bounded_authenticated_frame() {
        let (mut source, mut destination) = paired();
        let (transfer_id, task_id) = transfer_ids();
        let artifact_id = [8; 16];
        let payload = Zeroizing::new(Vec::new());
        let mut encoder = ArtifactHandoffEncoder::new(
            transfer_id,
            task_id,
            artifact_id,
            source.local_peer_id,
            destination.local_peer_id,
            payload,
        )
        .unwrap();
        let expectation = ArtifactHandoffExpectation::new(
            transfer_id,
            task_id,
            artifact_id,
            source.local_peer_id,
            destination.local_peer_id,
            0,
            Sha256::digest([]).into(),
        )
        .unwrap();
        let mut assembler = ArtifactHandoffAssembler::new(expectation);
        let frame = encoder.next_frame().unwrap().unwrap();
        let message = authenticated_message(
            &mut source,
            &mut destination,
            PeerMessageKind::Artifact,
            &frame,
        );
        let assembled = assembler.push(&message).unwrap().unwrap();
        assert!(assembled.as_bytes().is_empty());
        assert!(encoder.next_frame().unwrap().is_none());
    }

    #[test]
    fn checkpoint_payload_moves_in_bounded_chunks_over_the_paired_channel() {
        let (mut source, mut destination) = paired();
        let (transfer_id, task_id) = transfer_ids();
        let payload = Zeroizing::new(vec![0x5a; HANDOFF_CHUNK_DATA_BYTES + 137]);
        let mut encoder = TaskHandoffEncoder::new(
            transfer_id,
            task_id,
            source.local_peer_id,
            destination.local_peer_id,
            payload.clone(),
        )
        .unwrap();
        let expectation = TaskHandoffExpectation::new(
            transfer_id,
            task_id,
            destination.trusted_peer.peer_id(),
            destination.local_peer_id,
            payload.len(),
        )
        .unwrap();
        let mut assembler = TaskHandoffAssembler::new(expectation);
        assert_eq!(encoder.message_kind(), PeerMessageKind::Checkpoint);
        assert_eq!(encoder.chunk_count(), 2);
        let mut assembled = None;
        while let Some(chunk) = encoder.next_frame().unwrap() {
            let message = authenticated_message(
                &mut source,
                &mut destination,
                PeerMessageKind::Checkpoint,
                &chunk,
            );
            let next_assembled = assembler.push(&message).unwrap();
            if next_assembled.is_some() {
                assembled = next_assembled;
            }
        }
        let assembled = assembled.expect("all checkpoint chunks assemble");
        assert_eq!(assembled.transfer_id(), &transfer_id);
        assert_eq!(assembled.task_id(), &task_id);
        assert_eq!(assembled.source_peer(), source.local_peer_id);
        assert_eq!(assembled.destination_peer(), destination.local_peer_id);
        assert_eq!(assembled.as_bytes(), payload.as_slice());
        assert!(encoder.next_frame().unwrap().is_none());
    }

    #[test]
    fn receipt_round_trips_over_the_reverse_authenticated_peer_channel() {
        let (mut source, mut destination) = paired();
        let (transfer_id, task_id) = transfer_ids();
        let encoder = TaskHandoffEncoder::new(
            transfer_id,
            task_id,
            source.local_peer_id,
            destination.local_peer_id,
            Zeroizing::new(b"durably staged checkpoint".to_vec()),
        )
        .unwrap();
        let receipt = encoder.receipt();
        let encoded = receipt.encode();
        assert_eq!(TaskHandoffReceipt::decode(&encoded).unwrap(), receipt);

        let encrypted = destination
            .channel
            .seal(PeerMessageKind::CheckpointReceipt, &encoded)
            .unwrap();
        let encrypted = EncryptedPeerFrame::decode(&encrypted.encode().unwrap()).unwrap();
        assert_eq!(encrypted.kind, PeerMessageKind::CheckpointReceipt);
        let plaintext = source.channel.open(&encrypted).unwrap();
        let recovered = TaskHandoffReceipt::decode(&plaintext).unwrap();
        assert_eq!(recovered, receipt);
        assert!(
            recovered.is_bound_to_channel(source.trusted_peer.peer_id(), source.local_peer_id,)
        );
        assert!(
            !recovered.is_bound_to_channel(PeerId::from_bytes([88; 32]), source.local_peer_id,)
        );

        let mut malformed = encoded;
        malformed[5] = 2;
        assert_eq!(
            TaskHandoffReceipt::decode(&malformed),
            Err(PeerError::InvalidFrame)
        );
    }

    #[test]
    fn checkpoint_frames_reject_wrong_peer_destination_kind_and_transfer_identity() {
        let (mut source, mut destination) = paired();
        let (transfer_id, task_id) = transfer_ids();
        let payload = Zeroizing::new(vec![7_u8; 64]);
        let mut encoder = TaskHandoffEncoder::new(
            transfer_id,
            task_id,
            source.local_peer_id,
            destination.local_peer_id,
            payload.clone(),
        )
        .unwrap();
        let frame = encoder.next_frame().unwrap().unwrap();
        let expectation = TaskHandoffExpectation::new(
            transfer_id,
            task_id,
            source.local_peer_id,
            destination.local_peer_id,
            payload.len(),
        )
        .unwrap();

        let message = authenticated_message(
            &mut source,
            &mut destination,
            PeerMessageKind::Checkpoint,
            &frame,
        );

        let wrong_peer_expectation = TaskHandoffExpectation::new(
            transfer_id,
            task_id,
            PeerId::from_bytes([99; 32]),
            destination.local_peer_id,
            payload.len(),
        )
        .unwrap();
        let mut wrong_peer = TaskHandoffAssembler::new(wrong_peer_expectation);
        assert!(matches!(
            wrong_peer.push(&message),
            Err(PeerError::InvalidFrame)
        ));

        let wrong_destination_expectation = TaskHandoffExpectation::new(
            transfer_id,
            task_id,
            source.local_peer_id,
            PeerId::from_bytes([88; 32]),
            payload.len(),
        )
        .unwrap();
        let mut wrong_destination = TaskHandoffAssembler::new(wrong_destination_expectation);
        assert!(matches!(
            wrong_destination.push(&message),
            Err(PeerError::InvalidFrame)
        ));

        let (mut wrong_kind_source, mut wrong_kind_destination) = paired();
        let wrong_kind_message = authenticated_message(
            &mut wrong_kind_source,
            &mut wrong_kind_destination,
            PeerMessageKind::JobOffer,
            &frame,
        );
        let mut wrong_kind = TaskHandoffAssembler::new(expectation);
        assert!(matches!(
            wrong_kind.push(&wrong_kind_message),
            Err(PeerError::InvalidFrame)
        ));

        let wrong_task = TaskHandoffExpectation::new(
            transfer_id,
            [3; 16],
            source.local_peer_id,
            destination.local_peer_id,
            payload.len(),
        )
        .unwrap();
        let mut wrong_task = TaskHandoffAssembler::new(wrong_task);
        assert!(matches!(
            wrong_task.push(&message),
            Err(PeerError::InvalidFrame)
        ));
    }

    #[test]
    fn checkpoint_assembly_is_ordered_digest_checked_and_single_use() {
        let (mut source, mut destination) = paired();
        let (transfer_id, task_id) = transfer_ids();
        let payload = Zeroizing::new(vec![0x36; HANDOFF_CHUNK_DATA_BYTES + 1]);
        let mut encoder = TaskHandoffEncoder::new(
            transfer_id,
            task_id,
            source.local_peer_id,
            destination.local_peer_id,
            payload,
        )
        .unwrap();
        let first = encoder.next_frame().unwrap().unwrap();
        let second = encoder.next_frame().unwrap().unwrap();
        let expectation = TaskHandoffExpectation::new(
            transfer_id,
            task_id,
            source.local_peer_id,
            destination.local_peer_id,
            HANDOFF_CHUNK_DATA_BYTES + 1,
        )
        .unwrap();
        let mut reordered = TaskHandoffAssembler::new(expectation);
        let second_message = authenticated_message(
            &mut source,
            &mut destination,
            PeerMessageKind::Checkpoint,
            &second,
        );
        assert!(matches!(
            reordered.push(&second_message),
            Err(PeerError::ReplayOrReordering)
        ));
        let first_message = authenticated_message(
            &mut source,
            &mut destination,
            PeerMessageKind::Checkpoint,
            &first,
        );
        assert!(matches!(
            reordered.push(&first_message),
            Err(PeerError::InvalidFrame)
        ));

        let expectation = TaskHandoffExpectation::new(
            transfer_id,
            task_id,
            source.local_peer_id,
            destination.local_peer_id,
            HANDOFF_CHUNK_DATA_BYTES + 1,
        )
        .unwrap();
        let mut corrupted = first.to_vec();
        corrupted[HANDOFF_CHUNK_HEADER_BYTES] ^= 0x80;
        let mut assembler = TaskHandoffAssembler::new(expectation);
        let corrupted_message = authenticated_message(
            &mut source,
            &mut destination,
            PeerMessageKind::Checkpoint,
            &corrupted,
        );
        assert!(assembler.push(&corrupted_message).unwrap().is_none());
        let second_message = authenticated_message(
            &mut source,
            &mut destination,
            PeerMessageKind::Checkpoint,
            &second,
        );
        assert!(matches!(
            assembler.push(&second_message),
            Err(PeerError::AuthenticationFailed)
        ));
    }

    #[test]
    fn checkpoint_codec_rejects_oversized_payloads_and_invalid_expectations() {
        let (source, destination) = paired();
        let (transfer_id, task_id) = transfer_ids();
        assert!(
            TaskHandoffEncoder::new(
                transfer_id,
                task_id,
                source.local_peer_id,
                destination.local_peer_id,
                Zeroizing::new(vec![0; MAX_TASK_HANDOFF_BYTES + 1]),
            )
            .is_err()
        );
        assert!(
            TaskHandoffExpectation::new(
                transfer_id,
                task_id,
                source.local_peer_id,
                destination.local_peer_id,
                MAX_TASK_HANDOFF_BYTES + 1,
            )
            .is_err()
        );
        assert!(
            TaskHandoffExpectation::new(
                [0; 16],
                task_id,
                source.local_peer_id,
                destination.local_peer_id,
                1024,
            )
            .is_err()
        );
    }
}
