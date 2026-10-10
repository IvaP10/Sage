//! Core bindings between durable task checkpoints and the peer framing codec.
//!
//! Encoding fences the source owner in encrypted storage before returning any
//! bytes to send. Decoding returns inert descriptive state only; callers must
//! durably claim the transfer, install task state, transfer artifacts, resolve
//! destination resources, and obtain fresh authority before resuming work.

use zeroize::Zeroizing;

use crate::{
    agency::{CheckpointArtifact, CheckpointState, TaskCheckpoint},
    error::{CoreError, CoreResult},
    storage::LocalStore,
};
use rusqlite::OptionalExtension;
use sage_peer::{
    ArtifactHandoffEncoder, ArtifactHandoffPayload, MAX_TASK_HANDOFF_ARTIFACT_BYTES,
    MAX_TASK_HANDOFF_BYTES, PeerId, PeerSecureChannel, TaskHandoffEncoder, TaskHandoffPayload,
    TaskHandoffReceipt,
};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const MAX_STAGED_INCOMING_HANDOFFS: i64 = 128;
const MAX_STAGED_INCOMING_HANDOFF_BYTES: u64 = 64 * 1024 * 1024;
const MAX_STAGED_OUTGOING_HANDOFFS: i64 = 128;
const MAX_STAGED_OUTGOING_HANDOFF_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TASK_HANDOFF_ARTIFACTS_BYTES: u64 = 64 * 1024 * 1024;

fn digest_hex(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn parse_digest_hex(value: &str) -> CoreResult<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CoreError::AuthenticationFailed);
    }
    let mut decoded = [0_u8; 32];
    for (index, byte) in decoded.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| CoreError::AuthenticationFailed)?;
    }
    Ok(decoded)
}

#[derive(Debug, Clone, Copy)]
pub struct TaskCheckpointHandoffBinding {
    pub transfer_id: [u8; 16],
    pub task_id: Uuid,
    pub source_peer: PeerId,
    pub destination_peer: PeerId,
    pub source_device_id: Uuid,
    pub destination_device_id: Uuid,
    pub source_owner_generation: u64,
}

/// Map an authenticated peer identity to the stable UUID used by execution
/// ownership checkpoints. The peer key is Sage's device identity; domain
/// separation keeps this identifier independent from transfer and task IDs.
pub fn device_id_for_peer(peer: PeerId) -> Uuid {
    let mut digest = Sha256::new();
    digest.update(b"sage:device-id:v1\0");
    digest.update(peer.as_bytes());
    let digest = digest.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

#[derive(Debug, Clone)]
pub struct StagedTaskCheckpointHandoff {
    pub binding: TaskCheckpointHandoffBinding,
    pub checkpoint: TaskCheckpoint,
}

pub struct StagedOutgoingTaskCheckpointHandoff {
    pub binding: TaskCheckpointHandoffBinding,
    pub encoder: TaskHandoffEncoder,
    pub artifacts: Vec<StagedTaskHandoffArtifact>,
}

pub struct StagedTaskHandoffArtifact {
    pub descriptor: CheckpointArtifact,
    pub encoder: ArtifactHandoffEncoder,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceivedTaskHandoffStage {
    pub newly_staged: bool,
    /// Present only after the checkpoint and every referenced artifact body
    /// have been durably staged and independently revalidated.
    pub receipt: Option<TaskHandoffReceipt>,
}

/// Durable destination ownership metadata. This identifies a claim record;
/// it contains no credentials or reusable execution authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestinationTaskExecutionClaim {
    pub claim_id: Uuid,
    pub transfer_id: [u8; 16],
    pub task_id: Uuid,
    pub destination_peer: PeerId,
    pub destination_device_id: Uuid,
    pub owner_generation: u64,
    pub checkpoint_sha256: String,
}

impl TaskCheckpointHandoffBinding {
    pub fn new(
        transfer_id: [u8; 16],
        task_id: Uuid,
        source_peer: PeerId,
        destination_peer: PeerId,
        source_device_id: Uuid,
        destination_device_id: Uuid,
        source_owner_generation: u64,
    ) -> CoreResult<Self> {
        if transfer_id.iter().all(|byte| *byte == 0)
            || task_id.is_nil()
            || source_peer == destination_peer
            || source_device_id.is_nil()
            || destination_device_id.is_nil()
            || source_device_id == destination_device_id
            || source_owner_generation == 0
            || source_owner_generation >= i64::MAX as u64
        {
            return Err(CoreError::InvalidAction(
                "Task checkpoint handoff binding is invalid".into(),
            ));
        }
        Ok(Self {
            transfer_id,
            task_id,
            source_peer,
            destination_peer,
            source_device_id,
            destination_device_id,
            source_owner_generation,
        })
    }
}

/// Prepare an outbound handoff from the exact durable revision. The source's
/// settled owner-generation fence and retryable payload are committed in one
/// encrypted transaction before the encoder is returned.
pub fn begin_task_checkpoint_handoff(
    store: &LocalStore,
    expected_revision: u64,
    binding: TaskCheckpointHandoffBinding,
) -> CoreResult<(u64, TaskHandoffEncoder)> {
    if store.is_locked() {
        return Err(CoreError::PermissionRequired(
            "Task handoff requires unlocked encrypted storage".into(),
        ));
    }
    let (mut checkpoint, revision) = store
        .load_task_checkpoint(binding.task_id)?
        .ok_or_else(|| CoreError::TaskNotFound(binding.task_id.to_string()))?;
    if revision != expected_revision
        || checkpoint.task_id != binding.task_id
        || checkpoint.execution_owner.device_id != binding.source_device_id
        || checkpoint.execution_owner.generation != binding.source_owner_generation
        || checkpoint.state != CheckpointState::Settled
        || !checkpoint.dispatched_effects_settled
        || !checkpoint.pending_obligations.is_empty()
    {
        return Err(CoreError::PermissionRequired(
            "Task checkpoint is stale, unsettled, or owned by another device".into(),
        ));
    }

    checkpoint.transfer_ownership(binding.destination_device_id)?;
    store.stage_outgoing_task_handoff(&checkpoint, revision, binding)
}

/// Validate a fully assembled peer payload and return a checkpoint snapshot.
/// The result is not an import or execution claim and carries no authority.
pub fn decode_received_task_checkpoint(
    payload: TaskHandoffPayload,
    binding: TaskCheckpointHandoffBinding,
) -> CoreResult<TaskCheckpoint> {
    if payload.transfer_id() != &binding.transfer_id
        || payload.task_id() != binding.task_id.as_bytes()
        || payload.source_peer() != binding.source_peer
        || payload.destination_peer() != binding.destination_peer
    {
        return Err(CoreError::AuthenticationFailed);
    }
    let checkpoint: TaskCheckpoint =
        serde_json::from_slice(payload.as_bytes()).map_err(|_| CoreError::AuthenticationFailed)?;
    validate_checkpoint_binding(&checkpoint, binding)?;
    Ok(checkpoint)
}

fn validate_checkpoint_binding(
    checkpoint: &TaskCheckpoint,
    binding: TaskCheckpointHandoffBinding,
) -> CoreResult<()> {
    checkpoint.validate()?;
    let next_generation = binding
        .source_owner_generation
        .checked_add(1)
        .ok_or(CoreError::AuthenticationFailed)?;
    if binding.source_device_id != device_id_for_peer(binding.source_peer)
        || binding.destination_device_id != device_id_for_peer(binding.destination_peer)
        || checkpoint.task_id != binding.task_id
        || checkpoint.execution_owner.device_id != binding.destination_device_id
        || checkpoint.execution_owner.generation != next_generation
        || checkpoint.state != CheckpointState::Paused
        || !checkpoint.dispatched_effects_settled
        || !checkpoint.pending_obligations.is_empty()
    {
        return Err(CoreError::AuthenticationFailed);
    }
    Ok(())
}

struct IncomingHandoffArtifact {
    artifact_id: String,
    sha256: String,
    content: Zeroizing<Vec<u8>>,
}

/// Return None while expected bodies are still arriving; any present body
/// that does not match the exact checkpoint manifest fails closed.
fn validate_staged_incoming_artifacts(
    transaction: &rusqlite::Transaction<'_>,
    transfer_id: [u8; 16],
    task_id: Uuid,
    checkpoint: &TaskCheckpoint,
) -> CoreResult<Option<Vec<IncomingHandoffArtifact>>> {
    type Stored = (String, String, String, i64, Vec<u8>);
    let mut statement = transaction.prepare(
        "SELECT task_id,artifact_id,sha256,size_bytes,content FROM incoming_task_handoff_artifacts WHERE transfer_id=?1 ORDER BY artifact_id",
    )?;
    let rows = statement.query_map([transfer_id.as_slice()], |row| {
        Ok((
            row.get(0)?,
            row.get(1)?,
            row.get(2)?,
            row.get(3)?,
            row.get(4)?,
        ))
    })?;
    let mut stored = rows.collect::<Result<Vec<Stored>, _>>()?;
    if stored.len() > checkpoint.artifacts.len()
        || stored.iter().any(|row| {
            !checkpoint
                .artifacts
                .iter()
                .any(|item| item.artifact_id == row.1)
        })
    {
        return Err(CoreError::VerificationFailed(
            "Staged task handoff contains an unlisted artifact body".into(),
        ));
    }
    if stored.len() < checkpoint.artifacts.len() {
        return Ok(None);
    }

    let mut total_bytes = 0_u64;
    let mut artifacts = Vec::with_capacity(checkpoint.artifacts.len());
    for descriptor in &checkpoint.artifacts {
        let index = stored
            .iter()
            .position(|row| row.1 == descriptor.artifact_id)
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Staged task handoff is missing an artifact body".into(),
                )
            })?;
        let (stored_task, artifact_id, stored_digest, size_bytes, content) =
            stored.swap_remove(index);
        let actual_digest = format!("{:x}", Sha256::digest(&content));
        if stored_task != task_id.to_string()
            || artifact_id != descriptor.artifact_id
            || Uuid::parse_str(&artifact_id).is_err()
            || !stored_digest.eq_ignore_ascii_case(&actual_digest)
            || !actual_digest.eq_ignore_ascii_case(&descriptor.sha256)
            || size_bytes < 0
            || size_bytes as u64 != descriptor.size_bytes
            || content.len() as u64 != descriptor.size_bytes
            || content.len() > MAX_TASK_HANDOFF_ARTIFACT_BYTES
        {
            return Err(CoreError::VerificationFailed(
                "Staged task artifact failed owner, size, or digest validation".into(),
            ));
        }
        total_bytes = total_bytes
            .checked_add(content.len() as u64)
            .filter(|total| *total <= MAX_TASK_HANDOFF_ARTIFACTS_BYTES)
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Staged task artifacts exceed the aggregate transfer limit".into(),
                )
            })?;
        artifacts.push(IncomingHandoffArtifact {
            artifact_id,
            sha256: actual_digest,
            content: Zeroizing::new(content),
        });
    }
    Ok(Some(artifacts))
}

impl LocalStore {
    /// Atomically fence source ownership and persist the exact retry payload.
    /// The encoder is returned only after both records commit successfully.
    pub fn stage_outgoing_task_handoff(
        &self,
        checkpoint: &TaskCheckpoint,
        expected_revision: u64,
        binding: TaskCheckpointHandoffBinding,
    ) -> CoreResult<(u64, TaskHandoffEncoder)> {
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Task handoff requires unlocked encrypted storage".into(),
            ));
        }
        validate_checkpoint_binding(checkpoint, binding)?;
        let mut outgoing_artifacts = Vec::with_capacity(checkpoint.artifacts.len());
        let mut artifact_bytes_total = 0_u64;
        for descriptor in &checkpoint.artifacts {
            let artifact_id = Uuid::parse_str(&descriptor.artifact_id).map_err(|_| {
                CoreError::InvalidAction(
                    "Task handoff artifact identifier is not a local artifact UUID".into(),
                )
            })?;
            if descriptor.size_bytes > MAX_TASK_HANDOFF_ARTIFACT_BYTES as u64 {
                return Err(CoreError::InvalidAction(
                    "Task handoff artifact exceeds its per-artifact transfer limit".into(),
                ));
            }
            let bytes = Zeroizing::new(self.read_artifact_for_task(
                artifact_id,
                checkpoint.task_id,
                &descriptor.sha256,
            )?);
            if bytes.len() as u64 != descriptor.size_bytes {
                return Err(CoreError::VerificationFailed(
                    "Task artifact size does not match its checkpoint descriptor".into(),
                ));
            }
            artifact_bytes_total = artifact_bytes_total
                .checked_add(bytes.len() as u64)
                .filter(|total| *total <= MAX_TASK_HANDOFF_ARTIFACTS_BYTES)
                .ok_or_else(|| {
                    CoreError::InvalidAction(
                        "Task handoff artifacts exceed the aggregate transfer limit".into(),
                    )
                })?;
            outgoing_artifacts.push((descriptor.clone(), artifact_id, bytes));
        }
        let payload = Zeroizing::new(serde_json::to_vec(checkpoint)?);
        if payload.is_empty() || payload.len() > MAX_TASK_HANDOFF_BYTES {
            return Err(CoreError::InvalidAction(
                "Task checkpoint exceeds the peer handoff limit".into(),
            ));
        }
        let checkpoint_json = std::str::from_utf8(&payload)
            .map_err(|_| CoreError::InvalidAction("Serialized checkpoint is not UTF-8".into()))?;
        let checkpoint_sha256 = format!("{:x}", Sha256::digest(checkpoint_json.as_bytes()));
        let encoder = TaskHandoffEncoder::new(
            binding.transfer_id,
            binding.task_id.as_bytes().to_owned(),
            binding.source_peer,
            binding.destination_peer,
            Zeroizing::new(payload.to_vec()),
        )
        .map_err(|error| CoreError::Protocol(error.to_string()))?;

        let task_id = binding.task_id.to_string();
        let source_device_id = binding.source_device_id.to_string();
        let destination_device_id = binding.destination_device_id.to_string();
        let source_owner_generation = binding.source_owner_generation as i64;
        let next_revision = self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let source: Option<(i64, String)> = transaction
                .query_row(
                    "SELECT revision,checkpoint_json FROM task_checkpoints WHERE task_id=?1",
                    [&task_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            let Some((stored_revision, stored_json)) = source else {
                return Err(Box::new(CoreError::TaskNotFound(task_id.clone())));
            };
            if stored_revision < 1 || stored_revision as u64 != expected_revision {
                return Err(Box::new(CoreError::Storage(
                    "Task checkpoint revision changed".into(),
                )));
            }
            let previous: TaskCheckpoint = serde_json::from_str(&stored_json)?;
            previous.validate()?;
            if previous.execution_owner.device_id != binding.source_device_id
                || previous.execution_owner.generation != binding.source_owner_generation
                || previous.state != CheckpointState::Settled
                || !previous.dispatched_effects_settled
                || !previous.pending_obligations.is_empty()
            {
                return Err(Box::new(CoreError::PermissionRequired(
                    "Task checkpoint is stale, unsettled, or owned by another device".into(),
                )));
            }

            let unsettled_dispatch: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM action_journal WHERE run_id=?1 AND state IN ('prepared','dispatched','uncertain'))",
                [&task_id],
                |row| row.get(0),
            )?;
            let unprojected_receipt: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM worker_receipts r LEFT JOIN worker_receipt_projections p USING(request_id,kind) WHERE r.run_id=?1 AND p.request_id IS NULL)",
                [&task_id],
                |row| row.get(0),
            )?;
            if unsettled_dispatch || unprojected_receipt {
                return Err(Box::new(CoreError::PermissionRequired(
                    "Task handoff is blocked while durable action dispatches or worker receipts remain unsettled".into(),
                )));
            }

            let existing: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM outgoing_task_handoffs WHERE transfer_id=?1)",
                [binding.transfer_id.as_slice()],
                |row| row.get(0),
            )?;
            if existing {
                return Err(Box::new(CoreError::AuthenticationFailed));
            }
            let conflict: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM outgoing_task_handoffs WHERE task_id=?1 AND destination_device_id=?2)",
                rusqlite::params![task_id, destination_device_id],
                |row| row.get(0),
            )?;
            if conflict {
                return Err(Box::new(CoreError::AuthenticationFailed));
            }
            let (staged_count, staged_bytes): (i64, i64) = transaction.query_row(
                "SELECT COUNT(*),COALESCE(SUM(length(checkpoint_json)),0) + (SELECT COALESCE(SUM(length(content)),0) FROM outgoing_task_handoff_artifacts) FROM outgoing_task_handoffs",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            let projected_bytes = u64::try_from(staged_bytes)
                .ok()
                .and_then(|bytes| bytes.checked_add(checkpoint_json.len() as u64))
                .and_then(|bytes| bytes.checked_add(artifact_bytes_total));
            if staged_count >= MAX_STAGED_OUTGOING_HANDOFFS
                || projected_bytes.is_none_or(|bytes| {
                    bytes > MAX_STAGED_OUTGOING_HANDOFF_BYTES + MAX_TASK_HANDOFF_ARTIFACTS_BYTES
                })
            {
                return Err(Box::new(CoreError::Busy(
                    "Outgoing task handoff queue reached its bounded capacity".into(),
                )));
            }

            let next_revision = LocalStore::save_task_checkpoint_in_transaction(
                &transaction,
                checkpoint,
                expected_revision,
            )?;
            transaction.execute(
                "INSERT INTO outgoing_task_handoffs(transfer_id,task_id,source_peer,destination_peer,source_device_id,destination_device_id,source_owner_generation,checkpoint_sha256,checkpoint_json,staged_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                rusqlite::params![
                    binding.transfer_id.as_slice(),
                    task_id,
                    binding.source_peer.as_bytes().as_slice(),
                    binding.destination_peer.as_bytes().as_slice(),
                    source_device_id,
                    destination_device_id,
                    source_owner_generation,
                    checkpoint_sha256,
                    checkpoint_json,
                    chrono::Utc::now().to_rfc3339(),
                ],
            )?;
            for (descriptor, artifact_id, bytes) in &outgoing_artifacts {
                transaction.execute(
                    "INSERT INTO outgoing_task_handoff_artifacts(transfer_id,task_id,artifact_id,sha256,size_bytes,content) VALUES(?1,?2,?3,?4,?5,?6)",
                    rusqlite::params![
                        binding.transfer_id.as_slice(),
                        task_id,
                        artifact_id.to_string(),
                        descriptor.sha256.to_ascii_lowercase(),
                        bytes.len() as i64,
                        bytes.as_slice(),
                    ],
                )?;
            }
            transaction.commit()?;
            Ok(next_revision)
        })?;
        Ok((next_revision, encoder))
    }

    /// Recover a retry encoder after restart. The stored binding, size, digest,
    /// owner generation and checkpoint structure are revalidated before use.
    pub fn load_staged_outgoing_task_handoff(
        &self,
        transfer_id: [u8; 16],
    ) -> CoreResult<Option<StagedOutgoingTaskCheckpointHandoff>> {
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Task handoff requires unlocked encrypted storage".into(),
            ));
        }
        if transfer_id.iter().all(|byte| *byte == 0) {
            return Err(CoreError::InvalidAction(
                "Task handoff transfer ID is invalid".into(),
            ));
        }
        type Stored = (
            String,
            Vec<u8>,
            Vec<u8>,
            String,
            String,
            i64,
            String,
            String,
            i64,
        );
        let stored = self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let row: Option<Stored> = transaction
                .query_row(
                    "SELECT task_id,source_peer,destination_peer,source_device_id,destination_device_id,source_owner_generation,checkpoint_sha256,checkpoint_json,length(checkpoint_json) FROM outgoing_task_handoffs WHERE transfer_id=?1",
                    [transfer_id.as_slice()],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?,row.get(7)?,row.get(8)?)),
                )
                .optional()?;
            transaction.commit()?;
            Ok(row)
        })?;
        let Some((
            task_id,
            source_peer,
            destination_peer,
            source_device_id,
            destination_device_id,
            source_owner_generation,
            checkpoint_sha256,
            checkpoint_json,
            encoded_length,
        )) = stored
        else {
            return Ok(None);
        };
        if encoded_length < 0
            || encoded_length as usize > MAX_TASK_HANDOFF_BYTES
            || checkpoint_json.len() > MAX_TASK_HANDOFF_BYTES
            || format!("{:x}", Sha256::digest(checkpoint_json.as_bytes())) != checkpoint_sha256
        {
            return Err(CoreError::VerificationFailed(
                "Staged outgoing handoff failed its size or digest check".into(),
            ));
        }
        let binding = TaskCheckpointHandoffBinding::new(
            transfer_id,
            Uuid::parse_str(&task_id).map_err(|_| CoreError::AuthenticationFailed)?,
            PeerId::from_bytes(
                source_peer
                    .try_into()
                    .map_err(|_| CoreError::AuthenticationFailed)?,
            ),
            PeerId::from_bytes(
                destination_peer
                    .try_into()
                    .map_err(|_| CoreError::AuthenticationFailed)?,
            ),
            Uuid::parse_str(&source_device_id).map_err(|_| CoreError::AuthenticationFailed)?,
            Uuid::parse_str(&destination_device_id).map_err(|_| CoreError::AuthenticationFailed)?,
            u64::try_from(source_owner_generation).map_err(|_| CoreError::AuthenticationFailed)?,
        )?;
        let checkpoint: TaskCheckpoint =
            serde_json::from_str(&checkpoint_json).map_err(|_| CoreError::AuthenticationFailed)?;
        validate_checkpoint_binding(&checkpoint, binding)?;
        let encoder = TaskHandoffEncoder::new(
            binding.transfer_id,
            binding.task_id.as_bytes().to_owned(),
            binding.source_peer,
            binding.destination_peer,
            Zeroizing::new(checkpoint_json.into_bytes()),
        )
        .map_err(|_| CoreError::AuthenticationFailed)?;
        type StoredArtifact = (String, String, String, i64, Vec<u8>);
        let mut stored_artifacts = self.with_connection(|connection| {
            let mut statement = connection.prepare(
                "SELECT task_id,artifact_id,sha256,size_bytes,content FROM outgoing_task_handoff_artifacts WHERE transfer_id=?1 ORDER BY artifact_id",
            )?;
            let rows = statement.query_map([transfer_id.as_slice()], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))
            })?;
            rows.collect::<Result<Vec<StoredArtifact>, _>>()
                .map_err(Into::into)
        })?;
        if stored_artifacts.len() != checkpoint.artifacts.len() {
            return Err(CoreError::VerificationFailed(
                "Staged task handoff is missing artifact bodies".into(),
            ));
        }
        let mut artifacts = Vec::with_capacity(checkpoint.artifacts.len());
        let mut artifact_bytes_total = 0_u64;
        for descriptor in &checkpoint.artifacts {
            let index = stored_artifacts
                .iter()
                .position(|row| row.1 == descriptor.artifact_id)
                .ok_or_else(|| {
                    CoreError::VerificationFailed(
                        "Staged task handoff is missing an artifact body".into(),
                    )
                })?;
            let (stored_task, stored_id, stored_digest, stored_size, bytes) =
                stored_artifacts.swap_remove(index);
            let artifact_id = Uuid::parse_str(&descriptor.artifact_id)
                .map_err(|_| CoreError::AuthenticationFailed)?;
            let actual_digest = format!("{:x}", Sha256::digest(&bytes));
            if stored_task != binding.task_id.to_string()
                || stored_id != descriptor.artifact_id
                || stored_digest != actual_digest
                || !actual_digest.eq_ignore_ascii_case(&descriptor.sha256)
                || stored_size < 0
                || stored_size as u64 != descriptor.size_bytes
                || bytes.len() as u64 != descriptor.size_bytes
            {
                return Err(CoreError::VerificationFailed(
                    "Staged task artifact failed owner, size, or digest validation".into(),
                ));
            }
            artifact_bytes_total = artifact_bytes_total
                .checked_add(bytes.len() as u64)
                .filter(|total| *total <= MAX_TASK_HANDOFF_ARTIFACTS_BYTES)
                .ok_or_else(|| {
                    CoreError::VerificationFailed(
                        "Staged task artifacts exceed the aggregate transfer limit".into(),
                    )
                })?;
            let encoder = ArtifactHandoffEncoder::new(
                binding.transfer_id,
                binding.task_id.as_bytes().to_owned(),
                *artifact_id.as_bytes(),
                binding.source_peer,
                binding.destination_peer,
                Zeroizing::new(bytes),
            )
            .map_err(|_| CoreError::AuthenticationFailed)?;
            artifacts.push(StagedTaskHandoffArtifact {
                descriptor: descriptor.clone(),
                encoder,
            });
        }
        Ok(Some(StagedOutgoingTaskCheckpointHandoff {
            binding,
            encoder,
            artifacts,
        }))
    }

    /// Persist an authenticated incoming envelope in the encrypted inbox.
    /// Exact retries are idempotent; conflicting transfers for the same task
    /// and destination are rejected. Staging does not create or resume a task.
    pub fn stage_received_task_handoff(
        &self,
        payload: TaskHandoffPayload,
        binding: TaskCheckpointHandoffBinding,
    ) -> CoreResult<ReceivedTaskHandoffStage> {
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Task handoff requires unlocked encrypted storage".into(),
            ));
        }
        let receipt = payload.receipt();
        let checkpoint = decode_received_task_checkpoint(payload, binding)?;
        let checkpoint_bytes = Zeroizing::new(serde_json::to_vec(&checkpoint)?);
        if checkpoint_bytes.len() > MAX_TASK_HANDOFF_BYTES {
            return Err(CoreError::InvalidAction(
                "Task checkpoint exceeds the peer handoff limit".into(),
            ));
        }
        let checkpoint_json = std::str::from_utf8(&checkpoint_bytes)
            .map_err(|_| CoreError::InvalidAction("Serialized checkpoint is not UTF-8".into()))?;
        let checkpoint_sha256 = format!("{:x}", Sha256::digest(checkpoint_json.as_bytes()));
        let task_id = binding.task_id.to_string();
        let source_device_id = binding.source_device_id.to_string();
        let destination_device_id = binding.destination_device_id.to_string();
        let source_owner_generation = binding.source_owner_generation as i64;

        let (newly_staged, all_content_staged) = self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let retired: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=?1)",
                [&task_id],
                |row| row.get(0),
            )?;
            if retired {
                return Err(Box::new(CoreError::Storage(
                    "Task retention was revoked before checkpoint handoff".into(),
                )));
            }
            type Existing = (String, Vec<u8>, Vec<u8>, String, String, i64, String, String);
            let existing: Option<Existing> = transaction
                .query_row(
                    "SELECT task_id,source_peer,destination_peer,source_device_id,destination_device_id,source_owner_generation,checkpoint_sha256,checkpoint_json FROM incoming_task_handoffs WHERE transfer_id=?1",
                    [binding.transfer_id.as_slice()],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?,row.get(7)?)),
                )
                .optional()?;
            if let Some(existing) = existing {
                let stored_digest = format!("{:x}", Sha256::digest(existing.7.as_bytes()));
                let exact = existing.0 == task_id
                    && existing.1.as_slice() == binding.source_peer.as_bytes()
                    && existing.2.as_slice() == binding.destination_peer.as_bytes()
                    && existing.3 == source_device_id
                    && existing.4 == destination_device_id
                    && existing.5 == source_owner_generation
                    && existing.6 == stored_digest
                    && existing.6 == checkpoint_sha256
                    && existing.7 == checkpoint_json;
                if !exact {
                    return Err(Box::new(CoreError::AuthenticationFailed));
                }
                let all_content_staged = validate_staged_incoming_artifacts(
                    &transaction,
                    binding.transfer_id,
                    binding.task_id,
                    &checkpoint,
                )?
                .is_some();
                transaction.commit()?;
                return Ok((false, all_content_staged));
            }

            let conflict: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM incoming_task_handoffs WHERE task_id=?1 AND destination_device_id=?2)",
                rusqlite::params![task_id, destination_device_id],
                |row| row.get(0),
            )?;
            if conflict {
                return Err(Box::new(CoreError::AuthenticationFailed));
            }
            let (staged_count, staged_bytes): (i64, i64) = transaction.query_row(
                "SELECT COUNT(*),COALESCE(SUM(length(checkpoint_json)),0) FROM incoming_task_handoffs",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            let projected_bytes = u64::try_from(staged_bytes)
                .ok()
                .and_then(|bytes| bytes.checked_add(checkpoint_json.len() as u64));
            if staged_count >= MAX_STAGED_INCOMING_HANDOFFS
                || projected_bytes.is_none_or(|bytes| bytes > MAX_STAGED_INCOMING_HANDOFF_BYTES)
            {
                return Err(Box::new(CoreError::Busy(
                    "Incoming task handoff inbox reached its bounded capacity".into(),
                )));
            }
            transaction.execute(
                "INSERT INTO incoming_task_handoffs(transfer_id,task_id,source_peer,destination_peer,source_device_id,destination_device_id,source_owner_generation,checkpoint_sha256,checkpoint_json,received_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                rusqlite::params![
                    binding.transfer_id.as_slice(),
                    task_id,
                    binding.source_peer.as_bytes().as_slice(),
                    binding.destination_peer.as_bytes().as_slice(),
                    source_device_id,
                    destination_device_id,
                    source_owner_generation,
                    checkpoint_sha256,
                    checkpoint_json,
                    chrono::Utc::now().to_rfc3339(),
                ],
            )?;
            let all_content_staged = validate_staged_incoming_artifacts(
                &transaction,
                binding.transfer_id,
                binding.task_id,
                &checkpoint,
            )?
            .is_some();
            transaction.commit()?;
            Ok((true, all_content_staged))
        })?;
        Ok(ReceivedTaskHandoffStage {
            newly_staged,
            receipt: all_content_staged.then_some(receipt),
        })
    }

    /// Stage one completed artifact only when its encrypted peer assembler
    /// verified the exact transfer, task, artifact, source, destination and
    /// whole-body digest. A checkpoint receipt becomes available only after
    /// every listed artifact has been persisted and revalidated.
    pub fn stage_received_task_handoff_artifact(
        &self,
        payload: ArtifactHandoffPayload,
        binding: TaskCheckpointHandoffBinding,
    ) -> CoreResult<Option<TaskHandoffReceipt>> {
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Task handoff requires unlocked encrypted storage".into(),
            ));
        }
        if payload.transfer_id() != &binding.transfer_id
            || payload.task_id() != binding.task_id.as_bytes()
            || payload.source_peer() != binding.source_peer
            || payload.destination_peer() != binding.destination_peer
            || payload.as_bytes().len() > MAX_TASK_HANDOFF_ARTIFACT_BYTES
            || Sha256::digest(payload.as_bytes()).as_slice() != payload.sha256()
        {
            return Err(CoreError::AuthenticationFailed);
        }
        let artifact_id = Uuid::from_bytes(*payload.artifact_id()).to_string();
        let artifact_sha256 = digest_hex(payload.sha256());
        let task_id = binding.task_id.to_string();
        let checkpoint_digest = self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            type StoredIncoming = (String, Vec<u8>, Vec<u8>, String, String, i64, String, String);
            let staged: Option<StoredIncoming> = transaction
                .query_row(
                    "SELECT task_id,source_peer,destination_peer,source_device_id,destination_device_id,source_owner_generation,checkpoint_sha256,checkpoint_json FROM incoming_task_handoffs WHERE transfer_id=?1",
                    [binding.transfer_id.as_slice()],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?,row.get(7)?)),
                )
                .optional()?;
            let Some((staged_task, source_peer, destination_peer, source_device, destination_device, generation, checkpoint_sha, checkpoint_json)) = staged else {
                return Err(Box::new(CoreError::AuthenticationFailed));
            };
            if staged_task != task_id
                || source_peer.as_slice() != binding.source_peer.as_bytes()
                || destination_peer.as_slice() != binding.destination_peer.as_bytes()
                || source_device != binding.source_device_id.to_string()
                || destination_device != binding.destination_device_id.to_string()
                || generation != binding.source_owner_generation as i64
                || format!("{:x}", Sha256::digest(checkpoint_json.as_bytes())) != checkpoint_sha
            {
                return Err(Box::new(CoreError::AuthenticationFailed));
            }
            let checkpoint: TaskCheckpoint = serde_json::from_str(&checkpoint_json)
                .map_err(|_| CoreError::AuthenticationFailed)?;
            validate_checkpoint_binding(&checkpoint, binding)?;
            let descriptor = checkpoint
                .artifacts
                .iter()
                .find(|item| item.artifact_id == artifact_id)
                .ok_or(CoreError::AuthenticationFailed)?;
            if descriptor.size_bytes != payload.as_bytes().len() as u64
                || !descriptor.sha256.eq_ignore_ascii_case(&artifact_sha256)
            {
                return Err(Box::new(CoreError::AuthenticationFailed));
            }
            let retired: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=?1)",
                [&task_id],
                |row| row.get(0),
            )?;
            if retired {
                return Err(Box::new(CoreError::Storage(
                    "Task retention was revoked before artifact handoff".into(),
                )));
            }
            let existing: Option<(String, i64, Vec<u8>)> = transaction
                .query_row(
                    "SELECT sha256,size_bytes,content FROM incoming_task_handoff_artifacts WHERE transfer_id=?1 AND artifact_id=?2",
                    rusqlite::params![binding.transfer_id.as_slice(), artifact_id],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)),
                )
                .optional()?;
            if let Some((sha, size, content)) = existing {
                let exact = sha == artifact_sha256
                    && size == payload.as_bytes().len() as i64
                    && content.as_slice() == payload.as_bytes();
                if !exact {
                    return Err(Box::new(CoreError::AuthenticationFailed));
                }
                let all_content_staged = validate_staged_incoming_artifacts(
                    &transaction,
                    binding.transfer_id,
                    binding.task_id,
                    &checkpoint,
                )?
                .is_some();
                transaction.commit()?;
                return Ok(all_content_staged.then_some(checkpoint_sha));
            }
            let staged_bytes: i64 = transaction.query_row(
                "SELECT COALESCE(SUM(length(content)),0) FROM incoming_task_handoff_artifacts",
                [],
                |row| row.get(0),
            )?;
            let projected_bytes = u64::try_from(staged_bytes)
                .ok()
                .and_then(|bytes| bytes.checked_add(payload.as_bytes().len() as u64));
            if projected_bytes.is_none_or(|bytes| bytes > MAX_TASK_HANDOFF_ARTIFACTS_BYTES) {
                return Err(Box::new(CoreError::Busy(
                    "Incoming task artifact inbox reached its bounded capacity".into(),
                )));
            }
            transaction.execute(
                "INSERT INTO incoming_task_handoff_artifacts(transfer_id,task_id,artifact_id,sha256,size_bytes,content) VALUES(?1,?2,?3,?4,?5,?6)",
                rusqlite::params![
                    binding.transfer_id.as_slice(),
                    task_id,
                    artifact_id,
                    artifact_sha256,
                    payload.as_bytes().len() as i64,
                    payload.as_bytes(),
                ],
            )?;
            let all_content_staged = validate_staged_incoming_artifacts(
                &transaction,
                binding.transfer_id,
                binding.task_id,
                &checkpoint,
            )?
            .is_some();
            transaction.commit()?;
            Ok(all_content_staged.then_some(checkpoint_sha))
        })?;
        let Some(checkpoint_digest) = checkpoint_digest else {
            return Ok(None);
        };
        let bytes = parse_digest_hex(&checkpoint_digest)?;
        Ok(Some(
            TaskHandoffReceipt::new(
                binding.transfer_id,
                *binding.task_id.as_bytes(),
                binding.source_peer,
                binding.destination_peer,
                bytes,
            )
            .map_err(|error| CoreError::Protocol(error.to_string()))?,
        ))
    }

    /// Claim one staged transfer on the paired destination and atomically
    /// install its checkpoint as a paused local task. The supplied channel is
    /// identity-bearing and cannot be assembled by callers; a successful
    /// claim still grants no action capability and leaves the task fenced
    /// until destination resources are re-resolved and authorized.
    pub fn claim_staged_task_handoff(
        &self,
        transfer_id: [u8; 16],
        authenticated_channel: &PeerSecureChannel,
    ) -> CoreResult<DestinationTaskExecutionClaim> {
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Claiming a task handoff requires unlocked encrypted storage".into(),
            ));
        }
        if transfer_id.iter().all(|byte| *byte == 0) {
            return Err(CoreError::InvalidAction(
                "Task handoff transfer ID is invalid".into(),
            ));
        }

        self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            type StoredIncoming = (
                String,
                Vec<u8>,
                Vec<u8>,
                String,
                String,
                i64,
                String,
                String,
                i64,
            );
            let staged: Option<StoredIncoming> = transaction
                .query_row(
                    "SELECT task_id,source_peer,destination_peer,source_device_id,destination_device_id,source_owner_generation,checkpoint_sha256,checkpoint_json,length(checkpoint_json) FROM incoming_task_handoffs WHERE transfer_id=?1",
                    [transfer_id.as_slice()],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?,row.get(7)?,row.get(8)?)),
                )
                .optional()?;
            let Some((
                task_id,
                source_peer,
                destination_peer,
                source_device_id,
                destination_device_id,
                source_owner_generation,
                checkpoint_sha256,
                checkpoint_json,
                encoded_length,
            )) = staged
            else {
                return Err(Box::new(CoreError::AuthenticationFailed));
            };
            if encoded_length < 0
                || encoded_length as usize > MAX_TASK_HANDOFF_BYTES
                || checkpoint_json.len() > MAX_TASK_HANDOFF_BYTES
                || format!("{:x}", Sha256::digest(checkpoint_json.as_bytes()))
                    != checkpoint_sha256
            {
                return Err(Box::new(CoreError::VerificationFailed(
                    "Staged task handoff failed its size or digest check".into(),
                )));
            }

            let binding = TaskCheckpointHandoffBinding::new(
                transfer_id,
                Uuid::parse_str(&task_id).map_err(|_| CoreError::AuthenticationFailed)?,
                PeerId::from_bytes(
                    source_peer
                        .try_into()
                        .map_err(|_| CoreError::AuthenticationFailed)?,
                ),
                PeerId::from_bytes(
                    destination_peer
                        .try_into()
                        .map_err(|_| CoreError::AuthenticationFailed)?,
                ),
                Uuid::parse_str(&source_device_id)
                    .map_err(|_| CoreError::AuthenticationFailed)?,
                Uuid::parse_str(&destination_device_id)
                    .map_err(|_| CoreError::AuthenticationFailed)?,
                u64::try_from(source_owner_generation)
                    .map_err(|_| CoreError::AuthenticationFailed)?,
            )?;
            if authenticated_channel.remote_peer_id() != binding.source_peer
                || authenticated_channel.local_peer_id() != binding.destination_peer
                || binding.destination_device_id
                    != device_id_for_peer(authenticated_channel.local_peer_id())
            {
                return Err(Box::new(CoreError::AuthenticationFailed));
            }
            let checkpoint: TaskCheckpoint = serde_json::from_str(&checkpoint_json)
                .map_err(|_| CoreError::AuthenticationFailed)?;
            validate_checkpoint_binding(&checkpoint, binding)?;
            let retired: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=?1)",
                [&task_id],
                |row| row.get(0),
            )?;
            if retired {
                return Err(Box::new(CoreError::Storage(
                    "Task retention was revoked before ownership claim".into(),
                )));
            }

            type StoredClaim = (String, Vec<u8>, String, i64, String);
            let existing: Option<StoredClaim> = transaction
                .query_row(
                    "SELECT claim_id,destination_peer,destination_device_id,owner_generation,checkpoint_sha256 FROM task_handoff_execution_claims WHERE transfer_id=?1",
                    [transfer_id.as_slice()],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?)),
                )
                .optional()?;
            if let Some((claim_id, claimed_peer, claimed_device, generation, digest)) = existing {
                if claimed_peer.as_slice() != binding.destination_peer.as_bytes()
                    || claimed_device != binding.destination_device_id.to_string()
                    || generation != checkpoint.execution_owner.generation as i64
                    || digest != checkpoint_sha256
                {
                    return Err(Box::new(CoreError::AuthenticationFailed));
                }
                let claim = DestinationTaskExecutionClaim {
                    claim_id: Uuid::parse_str(&claim_id)
                        .map_err(|_| CoreError::AuthenticationFailed)?,
                    transfer_id,
                    task_id: binding.task_id,
                    destination_peer: binding.destination_peer,
                    destination_device_id: binding.destination_device_id,
                    owner_generation: checkpoint.execution_owner.generation,
                    checkpoint_sha256,
                };
                transaction.commit()?;
                return Ok(claim);
            }

            let task_exists: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM tasks WHERE id=?1)",
                [&task_id],
                |row| row.get(0),
            )?;
            let checkpoint_exists: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM task_checkpoints WHERE task_id=?1)",
                [&task_id],
                |row| row.get(0),
            )?;
            if task_exists || checkpoint_exists {
                return Err(Box::new(CoreError::AuthenticationFailed));
            }

            let incoming_artifacts = validate_staged_incoming_artifacts(
                &transaction,
                transfer_id,
                binding.task_id,
                &checkpoint,
            )?
            .ok_or_else(|| {
                CoreError::PermissionRequired(
                    "Task handoff is incomplete until every checkpoint artifact is received".into(),
                )
            })?;

            let mut task = crate::domain::Task::new(checkpoint.intent.clone());
            task.id = checkpoint.task_id;
            task.status = crate::domain::TaskStatus::Paused;
            task.goal = Some(checkpoint.intent.clone());
            crate::storage::write_task(&transaction, &task)?;

            let claim_id = Uuid::new_v4();
            transaction.execute(
                "INSERT INTO task_handoff_execution_claims(claim_id,transfer_id,task_id,destination_peer,destination_device_id,owner_generation,checkpoint_sha256,claimed_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
                rusqlite::params![
                    claim_id.to_string(),
                    transfer_id.as_slice(),
                    binding.task_id.to_string(),
                    binding.destination_peer.as_bytes().as_slice(),
                    binding.destination_device_id.to_string(),
                    checkpoint.execution_owner.generation as i64,
                    checkpoint_sha256,
                    chrono::Utc::now().to_rfc3339(),
                ],
            )?;
            crate::storage::LocalStore::save_task_checkpoint_in_transaction(
                &transaction,
                &checkpoint,
                0,
            )?;
            let artifact_expiry =
                (chrono::Utc::now() + chrono::Duration::hours(24)).to_rfc3339();
            for artifact in &incoming_artifacts {
                transaction.execute(
                    "INSERT INTO private_artifacts(id,task_id,content,sha256,expires_at) VALUES(?1,?2,?3,?4,?5)",
                    rusqlite::params![
                        artifact.artifact_id,
                        binding.task_id.to_string(),
                        artifact.content.as_slice(),
                        artifact.sha256,
                        artifact_expiry,
                    ],
                )?;
            }
            transaction.execute(
                "DELETE FROM incoming_task_handoff_artifacts WHERE transfer_id=?1",
                [transfer_id.as_slice()],
            )?;
            crate::storage::write_audit(
                &transaction,
                Some(binding.task_id),
                None,
                "task_handoff_destination_claimed",
                &serde_json::json!({
                    "claim_id": claim_id,
                    "transfer_id": transfer_id,
                    "destination_peer": binding.destination_peer.as_bytes(),
                    "destination_device_id": binding.destination_device_id,
                    "owner_generation": checkpoint.execution_owner.generation,
                    "checkpoint_sha256": checkpoint_sha256,
                    "task_status": "paused",
                }),
            )?;
            transaction.commit()?;
            Ok(DestinationTaskExecutionClaim {
                claim_id,
                transfer_id,
                task_id: binding.task_id,
                destination_peer: binding.destination_peer,
                destination_device_id: binding.destination_device_id,
                owner_generation: checkpoint.execution_owner.generation,
                checkpoint_sha256,
            })
        })
    }

    /// Clear a sender retry row only after a receipt has been authenticated by
    /// the paired transport. Callers must pass peer identities obtained from
    /// that authenticated channel, never from message fields alone.
    pub fn acknowledge_outgoing_task_handoff(
        &self,
        receipt: &TaskHandoffReceipt,
        authenticated_peer: PeerId,
        local_peer: PeerId,
    ) -> CoreResult<bool> {
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Task handoff requires unlocked encrypted storage".into(),
            ));
        }
        if !receipt.is_bound_to_channel(authenticated_peer, local_peer) {
            return Err(CoreError::AuthenticationFailed);
        }
        type Stored = (String, Vec<u8>, Vec<u8>, String);
        self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let stored: Option<Stored> = transaction
                .query_row(
                    "SELECT task_id,source_peer,destination_peer,checkpoint_sha256 FROM outgoing_task_handoffs WHERE transfer_id=?1",
                    [receipt.transfer_id().as_slice()],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?)),
                )
                .optional()?;
            let Some((task_id, source_peer, destination_peer, checkpoint_sha256)) = stored else {
                transaction.commit()?;
                return Ok(false);
            };
            let expected_task_id = Uuid::from_bytes(*receipt.task_id()).to_string();
            let expected_digest = digest_hex(receipt.payload_digest());
            if task_id != expected_task_id
                || source_peer.as_slice() != receipt.source_peer().as_bytes()
                || destination_peer.as_slice() != receipt.destination_peer().as_bytes()
                || checkpoint_sha256 != expected_digest
            {
                return Err(Box::new(CoreError::AuthenticationFailed));
            }
            let deleted = transaction.execute(
                "DELETE FROM outgoing_task_handoffs WHERE transfer_id=?1 AND checkpoint_sha256=?2",
                rusqlite::params![receipt.transfer_id().as_slice(), expected_digest],
            )?;
            if deleted != 1 {
                return Err(Box::new(CoreError::Storage(
                    "Outgoing handoff changed while acknowledging receipt".into(),
                )));
            }
            crate::storage::write_audit(
                &transaction,
                Some(Uuid::from_bytes(*receipt.task_id())),
                None,
                "task_checkpoint_handoff_acknowledged",
                &serde_json::json!({
                    "transfer_id": Uuid::from_bytes(*receipt.transfer_id()),
                    "payload_digest": expected_digest,
                }),
            )?;
            transaction.commit()?;
            Ok(true)
        })
    }

    /// Recover one staged checkpoint after restart, rechecking row size,
    /// content digest, peer bindings and owner generation before returning it.
    pub fn load_staged_task_handoff(
        &self,
        transfer_id: [u8; 16],
    ) -> CoreResult<Option<StagedTaskCheckpointHandoff>> {
        if self.is_locked() {
            return Err(CoreError::PermissionRequired(
                "Task handoff requires unlocked encrypted storage".into(),
            ));
        }
        if transfer_id.iter().all(|byte| *byte == 0) {
            return Err(CoreError::InvalidAction(
                "Task handoff transfer ID is invalid".into(),
            ));
        }
        type Stored = (
            String,
            Vec<u8>,
            Vec<u8>,
            String,
            String,
            i64,
            String,
            String,
            i64,
        );
        let stored = self.with_connection(|connection| {
            let transaction = connection.transaction()?;
            let row: Option<Stored> = transaction
                .query_row(
                    "SELECT task_id,source_peer,destination_peer,source_device_id,destination_device_id,source_owner_generation,checkpoint_sha256,checkpoint_json,length(checkpoint_json) FROM incoming_task_handoffs WHERE transfer_id=?1",
                    [transfer_id.as_slice()],
                    |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?,row.get(7)?,row.get(8)?)),
                )
                .optional()?;
            transaction.commit()?;
            Ok(row)
        })?;
        let Some((
            task_id,
            source_peer,
            destination_peer,
            source_device_id,
            destination_device_id,
            source_owner_generation,
            checkpoint_sha256,
            checkpoint_json,
            encoded_length,
        )) = stored
        else {
            return Ok(None);
        };
        if encoded_length < 0
            || encoded_length as usize > MAX_TASK_HANDOFF_BYTES
            || checkpoint_json.len() > MAX_TASK_HANDOFF_BYTES
            || format!("{:x}", Sha256::digest(checkpoint_json.as_bytes())) != checkpoint_sha256
        {
            return Err(CoreError::VerificationFailed(
                "Staged task handoff failed its size or digest check".into(),
            ));
        }
        let binding = TaskCheckpointHandoffBinding::new(
            transfer_id,
            Uuid::parse_str(&task_id).map_err(|_| CoreError::AuthenticationFailed)?,
            PeerId::from_bytes(
                source_peer
                    .try_into()
                    .map_err(|_| CoreError::AuthenticationFailed)?,
            ),
            PeerId::from_bytes(
                destination_peer
                    .try_into()
                    .map_err(|_| CoreError::AuthenticationFailed)?,
            ),
            Uuid::parse_str(&source_device_id).map_err(|_| CoreError::AuthenticationFailed)?,
            Uuid::parse_str(&destination_device_id).map_err(|_| CoreError::AuthenticationFailed)?,
            u64::try_from(source_owner_generation).map_err(|_| CoreError::AuthenticationFailed)?,
        )?;
        let checkpoint: TaskCheckpoint =
            serde_json::from_str(&checkpoint_json).map_err(|_| CoreError::AuthenticationFailed)?;
        validate_checkpoint_binding(&checkpoint, binding)?;
        Ok(Some(StagedTaskCheckpointHandoff {
            binding,
            checkpoint,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agency::{CheckpointArtifact, ExecutionOwner, PendingObligation},
        domain::Task,
        secrets::SecretBytes,
    };
    use sage_peer::{
        ArtifactHandoffAssembler, ArtifactHandoffEncoder, ArtifactHandoffExpectation,
        DeviceIdentity, EncryptedPeerFrame, PairingConfirmation, PairingOutcome, PairingQr,
        PairingResponse, PeerMessageKind, TaskHandoffAssembler, TaskHandoffEncoder,
        TaskHandoffExpectation, create_pairing_invitation,
    };
    use std::path::Path;
    use std::time::Duration;

    fn test_store() -> (tempfile::TempDir, LocalStore) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("task-handoff.db");
        let store = LocalStore::open_encrypted(&path, &SecretBytes::new(vec![77; 32])).unwrap();
        (directory, store)
    }

    fn checkpoint(task_id: Uuid, source_device_id: Uuid) -> TaskCheckpoint {
        TaskCheckpoint {
            schema_version: 1,
            task_id,
            intent: "continue a settled task".into(),
            procedure: None,
            goal_coordination: None,
            procedure_state: Default::default(),
            artifacts: Vec::new(),
            verified_results: Vec::new(),
            receipt_ids: Vec::new(),
            pending_obligations: Vec::new(),
            state: CheckpointState::Settled,
            execution_owner: ExecutionOwner {
                device_id: source_device_id,
                generation: 1,
            },
            dispatched_effects_settled: true,
        }
    }

    fn binding(
        task_id: Uuid,
        source_device_id: Uuid,
        destination_device_id: Uuid,
    ) -> TaskCheckpointHandoffBinding {
        let (source, destination) = paired_peers();
        TaskCheckpointHandoffBinding::new(
            [11; 16],
            task_id,
            source.local_peer_id,
            destination.local_peer_id,
            source_device_id,
            destination_device_id,
            1,
        )
        .unwrap()
    }

    fn paired_peers() -> (PairingOutcome, PairingOutcome) {
        paired_peers_with_seeds([31; 32], [47; 32])
    }

    fn paired_peers_with_seeds(
        inviter_seed: [u8; 32],
        responder_seed: [u8; 32],
    ) -> (PairingOutcome, PairingOutcome) {
        let inviter = DeviceIdentity::from_seed(inviter_seed);
        let responder = DeviceIdentity::from_seed(responder_seed);
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

    fn paired_device_ids() -> (Uuid, Uuid) {
        let (source, destination) = paired_peers();
        (
            device_id_for_peer(source.local_peer_id),
            device_id_for_peer(destination.local_peer_id),
        )
    }

    #[test]
    fn peer_device_uuid_is_stable_domain_separated_and_distinct() {
        let (source, destination) = paired_peers();
        let source_id = device_id_for_peer(source.local_peer_id);
        let destination_id = device_id_for_peer(destination.local_peer_id);
        assert_eq!(source_id, device_id_for_peer(source.local_peer_id));
        assert_ne!(source_id, destination_id);
        assert_ne!(source_id, Uuid::nil());
        assert_ne!(destination_id, Uuid::nil());
    }

    #[test]
    fn checkpoint_artifacts_are_staged_before_receipt_and_imported_with_paused_claim() {
        let (source_dir, source_store) = test_store();
        let (_destination_dir, destination_store) = test_store();
        let mut task = Task::new("transfer a task and its verified artifacts");
        source_store.save_task(&mut task).unwrap();
        let (source_device_id, destination_device_id) = paired_device_ids();
        let binding = binding(task.id, source_device_id, destination_device_id);
        let first_content: &[u8] = b"verified first result";
        let second_content: &[u8] = b"verified second result";
        let first_id = source_store.save_artifact(task.id, first_content).unwrap();
        let second_id = source_store.save_artifact(task.id, second_content).unwrap();
        let mut initial = checkpoint(task.id, source_device_id);
        initial.artifacts = vec![
            CheckpointArtifact {
                artifact_id: first_id.to_string(),
                sha256: format!("{:x}", Sha256::digest(first_content)),
                size_bytes: first_content.len() as u64,
                media_type: "text/plain".into(),
            },
            CheckpointArtifact {
                artifact_id: second_id.to_string(),
                sha256: format!("{:x}", Sha256::digest(second_content)),
                size_bytes: second_content.len() as u64,
                media_type: "text/plain".into(),
            },
        ];
        source_store.save_task_checkpoint(&initial, 0).unwrap();
        let (_, _checkpoint_encoder) =
            begin_task_checkpoint_handoff(&source_store, 1, binding).unwrap();
        drop(source_store);
        let source_store = LocalStore::open_encrypted(
            &source_dir.path().join("task-handoff.db"),
            &SecretBytes::new(vec![77; 32]),
        )
        .unwrap();
        let mut outgoing = source_store
            .load_staged_outgoing_task_handoff(binding.transfer_id)
            .unwrap()
            .unwrap();
        assert_eq!(outgoing.artifacts.len(), 2);

        let (transferred, _) = source_store.load_task_checkpoint(task.id).unwrap().unwrap();
        let staged_checkpoint = destination_store
            .stage_received_task_handoff(assembled_payload(&transferred, binding), binding)
            .unwrap();
        assert!(staged_checkpoint.receipt.is_none());
        let (mut source, mut destination) = paired_peers();
        assert!(
            destination_store
                .claim_staged_task_handoff(binding.transfer_id, &destination.channel)
                .is_err()
        );

        let mut completed_receipt = None;
        for artifact in outgoing.artifacts.drain(..) {
            let artifact_id = Uuid::parse_str(&artifact.descriptor.artifact_id).unwrap();
            let content = if artifact_id == first_id {
                first_content
            } else {
                second_content
            };
            let staged_receipt = destination_store
                .stage_received_task_handoff_artifact(
                    assembled_artifact_payload(binding, artifact_id, content),
                    binding,
                )
                .unwrap();
            if artifact_id == first_id {
                assert!(staged_receipt.is_none());
            } else {
                completed_receipt = staged_receipt;
            }
        }
        let completed_receipt = completed_receipt.expect("all task artifacts were staged");
        let encrypted_receipt = destination
            .channel
            .seal(
                PeerMessageKind::CheckpointReceipt,
                &completed_receipt.encode(),
            )
            .unwrap()
            .encode()
            .unwrap();
        let authenticated_receipt = source
            .channel
            .open_authenticated(&EncryptedPeerFrame::decode(&encrypted_receipt).unwrap())
            .unwrap();
        assert_eq!(
            authenticated_receipt.kind(),
            PeerMessageKind::CheckpointReceipt
        );
        let peer_receipt = TaskHandoffReceipt::decode(authenticated_receipt.payload()).unwrap();
        assert_eq!(peer_receipt, completed_receipt);
        assert!(
            source_store
                .acknowledge_outgoing_task_handoff(
                    &peer_receipt,
                    authenticated_receipt.source_peer(),
                    authenticated_receipt.destination_peer(),
                )
                .unwrap()
        );
        let remaining_outgoing_artifacts: i64 = source_store
            .with_connection(|connection| {
                Ok(connection.query_row(
                    "SELECT count(*) FROM outgoing_task_handoff_artifacts WHERE transfer_id=?1",
                    [binding.transfer_id.as_slice()],
                    |row| row.get(0),
                )?)
            })
            .unwrap();
        assert_eq!(remaining_outgoing_artifacts, 0);
        let claim = destination_store
            .claim_staged_task_handoff(binding.transfer_id, &destination.channel)
            .unwrap();
        assert_eq!(claim.task_id, task.id);
        assert_eq!(
            destination_store
                .read_artifact_for_task(
                    first_id,
                    task.id,
                    &format!("{:x}", Sha256::digest(first_content)),
                )
                .unwrap(),
            first_content
        );
        assert_eq!(
            destination_store
                .read_artifact_for_task(
                    second_id,
                    task.id,
                    &format!("{:x}", Sha256::digest(second_content)),
                )
                .unwrap(),
            second_content
        );
        let (installed, _) = destination_store
            .load_task_checkpoint(task.id)
            .unwrap()
            .unwrap();
        assert_eq!(installed.state, CheckpointState::Paused);
    }

    fn assembled_payload(
        checkpoint: &TaskCheckpoint,
        binding: TaskCheckpointHandoffBinding,
    ) -> TaskHandoffPayload {
        let (mut source, mut destination) = paired_peers();
        let mut encoder = TaskHandoffEncoder::new(
            binding.transfer_id,
            *binding.task_id.as_bytes(),
            binding.source_peer,
            binding.destination_peer,
            Zeroizing::new(serde_json::to_vec(checkpoint).unwrap()),
        )
        .unwrap();
        let expectation = TaskHandoffExpectation::new(
            binding.transfer_id,
            *binding.task_id.as_bytes(),
            binding.source_peer,
            binding.destination_peer,
            MAX_TASK_HANDOFF_BYTES,
        )
        .unwrap();
        let mut assembler = TaskHandoffAssembler::new(expectation);
        let mut completed = None;
        while let Some(frame) = encoder.next_frame().unwrap() {
            let encrypted = source
                .channel
                .seal(PeerMessageKind::Checkpoint, &frame)
                .unwrap();
            let encrypted = EncryptedPeerFrame::decode(&encrypted.encode().unwrap()).unwrap();
            let message = destination.channel.open_authenticated(&encrypted).unwrap();
            let received = assembler.push(&message).unwrap();
            if received.is_some() {
                completed = received;
            }
        }
        completed.unwrap()
    }

    fn assembled_artifact_payload(
        binding: TaskCheckpointHandoffBinding,
        artifact_id: Uuid,
        content: &[u8],
    ) -> ArtifactHandoffPayload {
        let (mut source, mut destination) = paired_peers();
        let mut encoder = ArtifactHandoffEncoder::new(
            binding.transfer_id,
            *binding.task_id.as_bytes(),
            *artifact_id.as_bytes(),
            source.local_peer_id,
            destination.local_peer_id,
            Zeroizing::new(content.to_vec()),
        )
        .unwrap();
        let digest = Sha256::digest(content).into();
        let expectation = ArtifactHandoffExpectation::new(
            binding.transfer_id,
            *binding.task_id.as_bytes(),
            *artifact_id.as_bytes(),
            source.local_peer_id,
            destination.local_peer_id,
            content.len(),
            digest,
        )
        .unwrap();
        let mut assembler = ArtifactHandoffAssembler::new(expectation);
        let mut completed = None;
        while let Some(frame) = encoder.next_frame().unwrap() {
            let encrypted = source
                .channel
                .seal(PeerMessageKind::Artifact, &frame)
                .unwrap();
            let encrypted = EncryptedPeerFrame::decode(&encrypted.encode().unwrap()).unwrap();
            let message = destination.channel.open_authenticated(&encrypted).unwrap();
            if let Some(payload) = assembler.push(&message).unwrap() {
                assert!(completed.replace(payload).is_none());
            }
        }
        completed.unwrap()
    }

    #[test]
    fn outbound_handoff_fences_the_source_before_emitting_a_validated_payload() {
        let (directory, store) = test_store();
        let mut task = Task::new("transfer a settled checkpoint");
        store.save_task(&mut task).unwrap();
        let (source_device_id, destination_device_id) = paired_device_ids();
        let initial = checkpoint(task.id, source_device_id);
        store.save_task_checkpoint(&initial, 0).unwrap();

        let binding = binding(task.id, source_device_id, destination_device_id);
        let (revision, mut encoder) = begin_task_checkpoint_handoff(&store, 1, binding).unwrap();
        assert_eq!(revision, 2);
        let (persisted, persisted_revision) = store.load_task_checkpoint(task.id).unwrap().unwrap();
        assert_eq!(persisted_revision, revision);
        assert_eq!(persisted.execution_owner.device_id, destination_device_id);
        assert_eq!(persisted.execution_owner.generation, 2);
        assert_eq!(persisted.state, CheckpointState::Paused);

        let expectation = TaskHandoffExpectation::new(
            binding.transfer_id,
            *task.id.as_bytes(),
            binding.source_peer,
            binding.destination_peer,
            MAX_TASK_HANDOFF_BYTES,
        )
        .unwrap();
        let mut assembler = TaskHandoffAssembler::new(expectation);
        let (mut source, mut destination) = paired_peers();
        let mut received = None;
        let mut original_frames = Vec::new();
        while let Some(frame) = encoder.next_frame().unwrap() {
            original_frames.push(frame.clone());
            let encrypted = source
                .channel
                .seal(PeerMessageKind::Checkpoint, &frame)
                .unwrap();
            let encrypted = EncryptedPeerFrame::decode(&encrypted.encode().unwrap()).unwrap();
            let message = destination.channel.open_authenticated(&encrypted).unwrap();
            let payload = assembler.push(&message).unwrap();
            if payload.is_some() {
                received = payload;
            }
        }
        let decoded = decode_received_task_checkpoint(received.unwrap(), binding).unwrap();
        assert_eq!(decoded.task_id, persisted.task_id);
        assert_eq!(decoded.execution_owner, persisted.execution_owner);
        assert_eq!(decoded.intent, initial.intent);

        drop(store);
        let reopened = LocalStore::open_encrypted(
            &directory.path().join("task-handoff.db"),
            &SecretBytes::new(vec![77; 32]),
        )
        .unwrap();
        let staged = reopened
            .load_staged_outgoing_task_handoff(binding.transfer_id)
            .unwrap()
            .unwrap();
        assert_eq!(staged.binding.task_id, task.id);
        let mut retried_frames = Vec::new();
        let mut retry_encoder = staged.encoder;
        while let Some(frame) = retry_encoder.next_frame().unwrap() {
            retried_frames.push(frame);
        }
        assert_eq!(retried_frames, original_frames);
    }

    #[test]
    fn failed_outbox_insert_rolls_back_the_source_ownership_fence() {
        let (_directory, store) = test_store();
        let mut task = Task::new("keep source ownership when handoff staging fails");
        store.save_task(&mut task).unwrap();
        let (source_device_id, destination_device_id) = paired_device_ids();
        let initial = checkpoint(task.id, source_device_id);
        store.save_task_checkpoint(&initial, 0).unwrap();
        let binding = binding(task.id, source_device_id, destination_device_id);
        store
            .with_connection(|connection| {
                connection.execute_batch(
                    "CREATE TRIGGER reject_outgoing_handoff
                     BEFORE INSERT ON outgoing_task_handoffs
                     BEGIN SELECT RAISE(ABORT,'injected staging failure'); END;",
                )?;
                Ok(())
            })
            .unwrap();

        assert!(begin_task_checkpoint_handoff(&store, 1, binding).is_err());
        let (persisted, revision) = store.load_task_checkpoint(task.id).unwrap().unwrap();
        assert_eq!(revision, 1);
        assert_eq!(persisted.execution_owner.device_id, source_device_id);
        let staged = store
            .load_staged_outgoing_task_handoff(binding.transfer_id)
            .unwrap();
        assert!(staged.is_none());
    }

    #[test]
    fn handoff_export_rejects_stale_revision_and_unsettled_effects() {
        let (_directory, store) = test_store();
        let mut task = Task::new("reject an unsafe transfer");
        store.save_task(&mut task).unwrap();
        let (source_device_id, destination_device_id) = paired_device_ids();
        let mut initial = checkpoint(task.id, source_device_id);
        initial.pending_obligations.push(PendingObligation {
            id: "settlement-required".into(),
            kind: "effect_settlement".into(),
            summary: "Wait for the dispatched effect to settle".into(),
        });
        initial.state = CheckpointState::NeedsReview;
        initial.dispatched_effects_settled = false;
        store.save_task_checkpoint(&initial, 0).unwrap();
        let binding = binding(task.id, source_device_id, destination_device_id);

        assert!(begin_task_checkpoint_handoff(&store, 0, binding).is_err());
        assert!(begin_task_checkpoint_handoff(&store, 1, binding).is_err());
        let (persisted, revision) = store.load_task_checkpoint(task.id).unwrap().unwrap();
        assert_eq!(revision, 1);
        assert_eq!(persisted.execution_owner.device_id, source_device_id);
    }

    #[test]
    fn handoff_export_rejects_a_settled_snapshot_with_an_unsettled_journal_action() {
        let (_directory, store) = test_store();
        let mut task = Task::new("do not transfer while a dispatch is in flight");
        store.save_task(&mut task).unwrap();
        let (source_device_id, destination_device_id) = paired_device_ids();
        let initial = checkpoint(task.id, source_device_id);
        store.save_task_checkpoint(&initial, 0).unwrap();

        let proposal = crate::domain::ActionProposal {
            id: Uuid::new_v4(),
            task_id: task.id,
            action: crate::domain::Action::AskUser {
                question: "Wait for the active result".into(),
            },
            expected_outcome: crate::domain::ExpectedOutcome::UserAnswered,
            target_resource: "user".into(),
            provenance: crate::domain::Provenance::model(vec![]),
            metadata: Default::default(),
        };
        let prepared =
            crate::contracts::PreparedAction::new(&proposal, Default::default()).unwrap();
        store.journal_prepared(&prepared).unwrap();
        store.journal_dispatch(&prepared, None).unwrap();

        let binding = binding(task.id, source_device_id, destination_device_id);
        assert!(begin_task_checkpoint_handoff(&store, 1, binding).is_err());
        let (persisted, revision) = store.load_task_checkpoint(task.id).unwrap().unwrap();
        assert_eq!(revision, 1);
        assert_eq!(persisted.execution_owner.device_id, source_device_id);
        assert_eq!(persisted.execution_owner.generation, 1);
        assert!(
            store
                .load_staged_outgoing_task_handoff(binding.transfer_id)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn incoming_handoff_is_durably_staged_idempotently_and_revalidated_after_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("incoming-handoff.db");
        let key = SecretBytes::new(vec![77; 32]);
        let (source_device_id, destination_device_id) = paired_device_ids();
        let task_id = Uuid::new_v4();
        let binding = binding(task_id, source_device_id, destination_device_id);
        let mut checkpoint = checkpoint(task_id, source_device_id);
        checkpoint
            .transfer_ownership(destination_device_id)
            .unwrap();
        let store = LocalStore::open_encrypted(&path, &key).unwrap();

        let payload = assembled_payload(&checkpoint, binding);
        let staged = store.stage_received_task_handoff(payload, binding).unwrap();
        assert!(staged.newly_staged);
        let retry = assembled_payload(&checkpoint, binding);
        let retried = store.stage_received_task_handoff(retry, binding).unwrap();
        assert!(!retried.newly_staged);
        assert_eq!(retried.receipt, staged.receipt);
        drop(store);

        let reopened = LocalStore::open_encrypted(&path, &key).unwrap();
        let staged = reopened
            .load_staged_task_handoff(binding.transfer_id)
            .unwrap()
            .unwrap();
        assert_eq!(staged.binding.task_id, task_id);
        assert_eq!(
            serde_json::to_value(&staged.checkpoint).unwrap(),
            serde_json::to_value(&checkpoint).unwrap()
        );

        let conflicting_binding = TaskCheckpointHandoffBinding::new(
            [12; 16],
            task_id,
            binding.source_peer,
            binding.destination_peer,
            source_device_id,
            destination_device_id,
            1,
        )
        .unwrap();
        assert!(
            reopened
                .stage_received_task_handoff(
                    assembled_payload(&checkpoint, conflicting_binding),
                    conflicting_binding,
                )
                .is_err()
        );

        let mut changed_checkpoint = checkpoint.clone();
        changed_checkpoint
            .intent
            .push_str(" with conflicting content");
        assert!(
            reopened
                .stage_received_task_handoff(
                    assembled_payload(&changed_checkpoint, binding),
                    binding,
                )
                .is_err()
        );
    }

    #[test]
    fn staged_handoff_recovery_detects_payload_tampering_and_retention_deletes_it() {
        let (_directory, store) = test_store();
        let (source_device_id, destination_device_id) = paired_device_ids();
        let task_id = Uuid::new_v4();
        let binding = binding(task_id, source_device_id, destination_device_id);
        let mut checkpoint = checkpoint(task_id, source_device_id);
        checkpoint
            .transfer_ownership(destination_device_id)
            .unwrap();
        assert!(
            store
                .stage_received_task_handoff(assembled_payload(&checkpoint, binding), binding,)
                .unwrap()
                .newly_staged
        );

        store
            .with_connection(|connection| {
                connection.execute(
                    "UPDATE incoming_task_handoffs SET checkpoint_json='{}' WHERE transfer_id=?1",
                    [binding.transfer_id.as_slice()],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(store.load_staged_task_handoff(binding.transfer_id).is_err());

        store
            .with_connection(|connection| {
                connection.execute(
                    "INSERT INTO retired_task_data(task_id,retired_at) VALUES(?1,?2)",
                    rusqlite::params![task_id.to_string(), chrono::Utc::now().to_rfc3339()],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(
            store
                .load_staged_task_handoff(binding.transfer_id)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn outgoing_retry_record_is_digest_checked_and_removed_by_retention() {
        let (_directory, store) = test_store();
        let mut task = Task::new("retain a retryable outgoing handoff");
        store.save_task(&mut task).unwrap();
        let (source_device_id, destination_device_id) = paired_device_ids();
        store
            .save_task_checkpoint(&checkpoint(task.id, source_device_id), 0)
            .unwrap();
        let binding = binding(task.id, source_device_id, destination_device_id);
        begin_task_checkpoint_handoff(&store, 1, binding).unwrap();
        store
            .with_connection(|connection| {
                connection.execute(
                    "UPDATE outgoing_task_handoffs SET checkpoint_json='{}' WHERE transfer_id=?1",
                    [binding.transfer_id.as_slice()],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(
            store
                .load_staged_outgoing_task_handoff(binding.transfer_id)
                .is_err()
        );

        store
            .with_connection(|connection| {
                connection.execute(
                    "INSERT INTO retired_task_data(task_id,retired_at) VALUES(?1,?2)",
                    rusqlite::params![task.id.to_string(), chrono::Utc::now().to_rfc3339()],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(
            store
                .load_staged_outgoing_task_handoff(binding.transfer_id)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn authenticated_matching_receipt_releases_only_its_outgoing_retry_record() {
        let (_source_directory, source_store) = test_store();
        let (_destination_directory, destination_store) = test_store();
        let mut task = Task::new("acknowledge a durable task handoff");
        source_store.save_task(&mut task).unwrap();
        let (source_device_id, destination_device_id) = paired_device_ids();
        let initial = checkpoint(task.id, source_device_id);
        source_store.save_task_checkpoint(&initial, 0).unwrap();
        let binding = binding(task.id, source_device_id, destination_device_id);
        let (_, encoder) = begin_task_checkpoint_handoff(&source_store, 1, binding).unwrap();
        let expected_receipt = encoder.receipt();
        assert!(
            source_store
                .load_staged_outgoing_task_handoff(binding.transfer_id)
                .unwrap()
                .is_some()
        );

        let mut transferred = initial;
        transferred
            .transfer_ownership(destination_device_id)
            .unwrap();
        let received = destination_store
            .stage_received_task_handoff(assembled_payload(&transferred, binding), binding)
            .unwrap();
        assert!(received.newly_staged);
        let received_receipt = received.receipt.expect("artifact-free checkpoint receipt");
        assert_eq!(received_receipt, expected_receipt);

        let wrong_peer = PeerId::from_bytes([99; 32]);
        assert!(
            source_store
                .acknowledge_outgoing_task_handoff(
                    &received_receipt,
                    wrong_peer,
                    binding.source_peer,
                )
                .is_err()
        );
        let mut wrong_digest_bytes = received_receipt.encode();
        *wrong_digest_bytes.last_mut().unwrap() ^= 1;
        let wrong_digest = TaskHandoffReceipt::decode(&wrong_digest_bytes).unwrap();
        assert!(
            source_store
                .acknowledge_outgoing_task_handoff(
                    &wrong_digest,
                    binding.destination_peer,
                    binding.source_peer,
                )
                .is_err()
        );
        assert!(
            source_store
                .load_staged_outgoing_task_handoff(binding.transfer_id)
                .unwrap()
                .is_some()
        );

        assert!(
            source_store
                .acknowledge_outgoing_task_handoff(
                    &received_receipt,
                    binding.destination_peer,
                    binding.source_peer,
                )
                .unwrap()
        );
        assert!(
            !source_store
                .acknowledge_outgoing_task_handoff(
                    &received_receipt,
                    binding.destination_peer,
                    binding.source_peer,
                )
                .unwrap()
        );
        assert!(
            source_store
                .load_staged_outgoing_task_handoff(binding.transfer_id)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn settled_checkpoint_delivery_and_receipt_commit_across_paired_stores() {
        let (_source_directory, source_store) = test_store();
        let (_destination_directory, destination_store) = test_store();
        let mut task = Task::new("transfer a checkpoint over an authenticated channel");
        source_store.save_task(&mut task).unwrap();
        let (source_device_id, destination_device_id) = paired_device_ids();
        let initial = checkpoint(task.id, source_device_id);
        source_store.save_task_checkpoint(&initial, 0).unwrap();
        let (mut source, mut destination) = paired_peers();
        let binding = TaskCheckpointHandoffBinding::new(
            [91; 16],
            task.id,
            source.local_peer_id,
            destination.local_peer_id,
            source_device_id,
            destination_device_id,
            1,
        )
        .unwrap();
        let (_, mut encoder) = begin_task_checkpoint_handoff(&source_store, 1, binding).unwrap();
        let expectation = TaskHandoffExpectation::new(
            binding.transfer_id,
            *task.id.as_bytes(),
            binding.source_peer,
            binding.destination_peer,
            MAX_TASK_HANDOFF_BYTES,
        )
        .unwrap();
        let mut assembler = TaskHandoffAssembler::new(expectation);
        let mut payload = None;
        while let Some(chunk) = encoder.next_frame().unwrap() {
            let encrypted = source
                .channel
                .seal(PeerMessageKind::Checkpoint, &chunk)
                .unwrap()
                .encode()
                .unwrap();
            let encrypted = EncryptedPeerFrame::decode(&encrypted).unwrap();
            let message = destination.channel.open_authenticated(&encrypted).unwrap();
            if let Some(assembled) = assembler.push(&message).unwrap() {
                assert!(payload.replace(assembled).is_none());
            }
        }
        let staged = destination_store
            .stage_received_task_handoff(payload.expect("checkpoint chunks complete"), binding)
            .unwrap();
        assert!(staged.newly_staged);
        let staged_receipt = staged.receipt.expect("artifact-free checkpoint receipt");
        let encrypted_receipt = destination
            .channel
            .seal(PeerMessageKind::CheckpointReceipt, &staged_receipt.encode())
            .unwrap()
            .encode()
            .unwrap();
        let encrypted_receipt = EncryptedPeerFrame::decode(&encrypted_receipt).unwrap();
        let receipt_plaintext = source.channel.open(&encrypted_receipt).unwrap();
        let receipt = TaskHandoffReceipt::decode(&receipt_plaintext).unwrap();
        assert!(
            source_store
                .acknowledge_outgoing_task_handoff(
                    &receipt,
                    source.trusted_peer.peer_id(),
                    source.local_peer_id,
                )
                .unwrap()
        );
        assert!(
            source_store
                .load_staged_outgoing_task_handoff(binding.transfer_id)
                .unwrap()
                .is_none()
        );
        let destination_checkpoint = destination_store
            .load_staged_task_handoff(binding.transfer_id)
            .unwrap()
            .unwrap();
        assert_eq!(destination_checkpoint.checkpoint.task_id, task.id);
        assert_eq!(
            destination_checkpoint.checkpoint.execution_owner.device_id,
            destination_device_id
        );

        let (_, wrong_destination) = paired_peers_with_seeds([32; 32], [48; 32]);
        assert!(
            destination_store
                .claim_staged_task_handoff(binding.transfer_id, &wrong_destination.channel)
                .is_err()
        );

        let claim = destination_store
            .claim_staged_task_handoff(binding.transfer_id, &destination.channel)
            .unwrap();
        assert_eq!(claim.transfer_id, binding.transfer_id);
        assert_eq!(claim.task_id, task.id);
        assert_eq!(claim.destination_peer, binding.destination_peer);
        assert_eq!(claim.destination_device_id, destination_device_id);
        assert_eq!(claim.owner_generation, 2);
        let expected_checkpoint_digest = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&destination_checkpoint.checkpoint).unwrap())
        );
        assert_eq!(claim.checkpoint_sha256, expected_checkpoint_digest);
        assert_eq!(
            destination_store
                .claim_staged_task_handoff(binding.transfer_id, &destination.channel)
                .unwrap(),
            claim,
            "reclaiming the same transfer returns its durable claim"
        );

        let installed_task = destination_store
            .load_tasks(true)
            .unwrap()
            .into_iter()
            .find(|candidate| candidate.id == task.id)
            .expect("destination claim installs a local task record");
        assert_eq!(installed_task.request, initial.intent);
        assert_eq!(installed_task.status, crate::domain::TaskStatus::Paused);
        let (installed_checkpoint, revision) = destination_store
            .load_task_checkpoint(task.id)
            .unwrap()
            .expect("destination claim installs the paused checkpoint");
        assert_eq!(revision, 1);
        assert_eq!(installed_checkpoint.state, CheckpointState::Paused);
        assert_eq!(
            installed_checkpoint.execution_owner.device_id,
            destination_device_id
        );
        assert_eq!(installed_checkpoint.execution_owner.generation, 2);
        assert!(
            destination_store
                .with_connection(|connection| {
                    LocalStore::validate_task_checkpoint_dispatch_owner(connection, task.id)?;
                    Ok(())
                })
                .is_err(),
            "claim metadata does not bypass the paused dispatch fence"
        );

        let destination_path = _destination_directory.path().join("task-handoff.db");
        drop(destination_store);
        let destination_store =
            LocalStore::open_encrypted(&destination_path, &SecretBytes::new(vec![77; 32])).unwrap();
        assert_eq!(
            destination_store
                .claim_staged_task_handoff(binding.transfer_id, &destination.channel)
                .unwrap(),
            claim,
            "the durable claim and paused task survive store reopen"
        );

        destination_store
            .with_connection(|connection| {
                connection.execute(
                    "INSERT INTO retired_task_data(task_id,retired_at) VALUES(?1,?2)",
                    rusqlite::params![task.id.to_string(), chrono::Utc::now().to_rfc3339()],
                )?;
                let claims: i64 = connection.query_row(
                    "SELECT count(*) FROM task_handoff_execution_claims WHERE task_id=?1",
                    [task.id.to_string()],
                    |row| row.get(0),
                )?;
                assert_eq!(claims, 0);
                Ok(())
            })
            .unwrap();
        assert!(
            destination_store
                .load_task_checkpoint(task.id)
                .unwrap()
                .is_none()
        );
        assert!(
            destination_store
                .load_staged_task_handoff(binding.transfer_id)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn locked_storage_cannot_stage_export_or_recover_handoff_bytes() {
        let store = LocalStore::deferred(Path::new(":memory:")).unwrap();
        let mut task = Task::new("reject handoff while storage is locked");
        store.save_task(&mut task).unwrap();
        let (source_device_id, destination_device_id) = paired_device_ids();
        let checkpoint = checkpoint(task.id, source_device_id);
        store.save_task_checkpoint(&checkpoint, 0).unwrap();
        let binding = binding(task.id, source_device_id, destination_device_id);

        assert!(begin_task_checkpoint_handoff(&store, 1, binding).is_err());
        let mut incoming = checkpoint;
        incoming.transfer_ownership(destination_device_id).unwrap();
        assert!(
            store
                .stage_received_task_handoff(assembled_payload(&incoming, binding), binding)
                .is_err()
        );
        assert!(store.load_staged_task_handoff(binding.transfer_id).is_err());
    }
}
