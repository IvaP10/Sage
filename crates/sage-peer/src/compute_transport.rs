//! Authenticated async transport helpers for bounded partition requests and results.
//!
//! These functions join the compute codecs to Sage's paired peer stream. They
//! only move plan-bound payloads; lease admission, durable dispatch fencing,
//! execution, settlement, and result verification remain explicit caller steps.

use std::time::Duration;

use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};
use zeroize::Zeroizing;

use crate::{
    AuthenticatedPeerMessage, ComputePlan, ComputeTransferAssembler, ComputeTransferEncoder,
    ComputeTransferKind, MAX_PEER_STREAM_TIMEOUT, PartitionRequest, PartitionResult, PeerError,
    PeerId, PeerSecureReader, PeerSecureWriter, PeerTransportError, PreparedPartitionRequest,
    ReceivedPartitionRequest, VerifiedPartition,
};

#[derive(Debug, Error)]
pub enum ComputeTransportError {
    #[error(transparent)]
    Peer(#[from] PeerError),
    #[error(transparent)]
    Transport(#[from] PeerTransportError),
    #[error("authenticated peer stream closed before the partition transfer completed")]
    UnexpectedEof,
    #[error("partition transfer came from an unexpected paired device")]
    UnexpectedPeer,
}

/// Send one bounded request as ordered authenticated `JobOffer` chunks.
/// `deadline` bounds the whole transfer as well as each underlying frame I/O.
pub async fn send_partition_request<W: AsyncWrite + Unpin>(
    writer: &mut PeerSecureWriter<W>,
    request: PartitionRequest,
    deadline: Duration,
) -> Result<(), ComputeTransportError> {
    send_transfer(writer, request.into_transfer()?, deadline).await
}

/// Receive a request for the exact registered partition and bind its sender
/// to the peer authenticated by the paired stream. `deadline` bounds the
/// complete transfer. The returned request still requires the owner's live
/// lease and durable-dispatch checks.
pub async fn receive_partition_request<R: AsyncRead + Unpin>(
    reader: &mut PeerSecureReader<R>,
    plan: &ComputePlan,
    partition_index: u16,
    expected_source: PeerId,
    local_peer: PeerId,
    deadline: Duration,
) -> Result<PreparedPartitionRequest, ComputeTransportError> {
    let expected =
        plan.transfer_expectation(ComputeTransferKind::PartitionRequest, partition_index)?;
    let payload = receive_transfer(reader, expected, expected_source, local_peer, deadline).await?;
    let request = ReceivedPartitionRequest::decode(&payload, expected_source)?;
    Ok(plan.prepare_partition_request(request)?)
}

/// Send a bounded result as ordered authenticated `JobReceipt` chunks.
pub async fn send_partition_result<W: AsyncWrite + Unpin>(
    writer: &mut PeerSecureWriter<W>,
    result: PartitionResult,
    deadline: Duration,
) -> Result<(), ComputeTransportError> {
    send_transfer(writer, result.into_transfer()?, deadline).await
}

/// Receive, decode, and semantically verify a result from the exact paired
/// peer and registered plan before exposing a checkpointable partition.
/// `deadline` bounds the complete transfer.
pub async fn receive_verified_partition_result<R, F>(
    reader: &mut PeerSecureReader<R>,
    plan: &ComputePlan,
    partition_index: u16,
    expected_source: PeerId,
    local_peer: PeerId,
    deadline: Duration,
    verify_unit: F,
) -> Result<VerifiedPartition, ComputeTransportError>
where
    R: AsyncRead + Unpin,
    F: FnMut(&crate::ComputeUnitSpec, &[u8]) -> bool,
{
    let expected =
        plan.transfer_expectation(ComputeTransferKind::PartitionResult, partition_index)?;
    let payload = receive_transfer(reader, expected, expected_source, local_peer, deadline).await?;
    let result = PartitionResult::decode(&payload)?;
    Ok(plan.verify_partition_result(partition_index, expected_source, result, verify_unit)?)
}

async fn send_transfer<W: AsyncWrite + Unpin>(
    writer: &mut PeerSecureWriter<W>,
    mut transfer: ComputeTransferEncoder,
    deadline: Duration,
) -> Result<(), ComputeTransportError> {
    validate_transfer_deadline(deadline)?;
    tokio::time::timeout(deadline, async {
        let kind = transfer.message_kind();
        while let Some(frame) = transfer.next_frame()? {
            writer.send(kind, &frame, deadline).await?;
        }
        Ok::<(), ComputeTransportError>(())
    })
    .await
    .map_err(|_| PeerTransportError::TimedOut)??;
    Ok(())
}

fn validate_transfer_deadline(deadline: Duration) -> Result<(), ComputeTransportError> {
    if deadline < Duration::from_millis(1) || deadline > MAX_PEER_STREAM_TIMEOUT {
        return Err(PeerTransportError::InvalidTimeout.into());
    }
    Ok(())
}

async fn receive_transfer<R: AsyncRead + Unpin>(
    reader: &mut PeerSecureReader<R>,
    expected: crate::ComputeTransferExpectation,
    expected_source: PeerId,
    local_peer: PeerId,
    deadline: Duration,
) -> Result<Zeroizing<Vec<u8>>, ComputeTransportError> {
    validate_transfer_deadline(deadline)?;
    tokio::time::timeout(deadline, async {
        let mut assembler = ComputeTransferAssembler::new(expected);
        loop {
            let Some(message) = reader.receive(deadline).await? else {
                return Err(ComputeTransportError::UnexpectedEof);
            };
            if !authenticated_for(&message, expected_source, local_peer) {
                return Err(ComputeTransportError::UnexpectedPeer);
            }
            if let Some(payload) = assembler.push(message.kind(), message.payload())? {
                return Ok(payload);
            }
        }
    })
    .await
    .map_err(|_| ComputeTransportError::Transport(PeerTransportError::TimedOut))?
}

fn authenticated_for(
    message: &AuthenticatedPeerMessage,
    expected_source: PeerId,
    local_peer: PeerId,
) -> bool {
    message.source_peer() == expected_source && message.destination_peer() == local_peer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ComputeUnitSpec, DelegatedJobKind, DeviceIdentity, DisclosureScope, LeaseScope,
        PeerDispatchJournal, PeerLease, PeerLeaseBook, PeerLeaseQuotas, PeerMessageKind, UnitInput,
        UnitOutput, byte_histogram_v1_resource_id, create_pairing_invitation,
        execute_authorized_partition, split_secure_stream, verify_builtin_output,
    };
    use sha2::{Digest, Sha256};
    use std::{sync::Arc, time::Duration};
    use zeroize::Zeroizing;

    const NOW: u64 = 1_800_000_014;
    const DEADLINE: Duration = Duration::from_secs(10);

    fn paired_channels() -> (crate::PairingOutcome, crate::PairingOutcome) {
        let inviter = DeviceIdentity::from_seed([7; 32]);
        let responder = DeviceIdentity::from_seed([19; 32]);
        let (qr, pending_inviter) =
            create_pairing_invitation(&inviter, NOW - 10, Duration::from_secs(120)).unwrap();
        let (response, pending_responder) = PairingQr::from_payload(&qr.to_payload())
            .unwrap()
            .respond_after_local_confirmation(&responder, NOW - 5)
            .unwrap();
        let code = pending_inviter.authentication_code(&response).unwrap();
        assert_eq!(code, pending_responder.authentication_code());
        let (confirmation, inviter_outcome) = pending_inviter
            .confirm_after_local_confirmation(&inviter, &response, &code, NOW - 4)
            .unwrap();
        let responder_outcome = pending_responder
            .finish_after_local_confirmation(&confirmation, &code, NOW - 3)
            .unwrap();
        (inviter_outcome, responder_outcome)
    }

    #[tokio::test]
    async fn chunked_partition_request_runs_through_lease_fence_and_verified_result() {
        let (requester, donor) = paired_channels();
        let input = vec![0x5a; crate::MAX_PEER_MESSAGE_BYTES + 4_096];
        let input_digest: [u8; 32] = Sha256::digest(&input).into();
        let resource_id = byte_histogram_v1_resource_id();
        let maximum_output_bytes = (12 + 256 * 8) as u32;
        let plan = Arc::new(
            ComputePlan::new(
                DelegatedJobKind::BatchAnalysis,
                resource_id,
                vec![
                    ComputeUnitSpec::new(
                        0,
                        input_digest,
                        u32::try_from(input.len()).unwrap(),
                        maximum_output_bytes,
                    )
                    .unwrap(),
                ],
                1,
            )
            .unwrap(),
        );
        let lease = PeerLease::issue(
            &DeviceIdentity::from_seed([19; 32]),
            requester.local_peer_id,
            NOW - 1,
            120,
            [LeaseScope {
                job: plan.job(),
                resource_id,
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 1,
                maximum_jobs: 1,
                maximum_input_bytes_per_job: input.len() as u64,
                maximum_output_bytes_per_job: maximum_output_bytes as u64,
                maximum_total_input_bytes: input.len() as u64,
                maximum_total_output_bytes: maximum_output_bytes as u64,
            },
        )
        .unwrap();
        let grant = lease.grant();
        let request = plan
            .make_partition_request(
                &grant,
                &requester.trusted_peer,
                requester.local_peer_id,
                0,
                vec![UnitInput::new(0, input.clone()).unwrap()],
                NOW,
            )
            .unwrap();
        let lease_id = lease.id();
        let requester_id = requester.local_peer_id;
        let donor_id = donor.local_peer_id;
        let expected_output_bytes = u64::from(maximum_output_bytes);
        let (request_stream, worker_stream) = tokio::io::duplex(512 * 1024);
        let (mut requester_writer, mut requester_reader) =
            split_secure_stream(request_stream, requester.channel);
        let (mut donor_writer, mut donor_reader) =
            split_secure_stream(worker_stream, donor.channel);

        let worker_plan = Arc::clone(&plan);
        let worker = tokio::spawn(async move {
            let prepared = receive_partition_request(
                &mut donor_reader,
                &worker_plan,
                0,
                requester_id,
                donor_id,
                DEADLINE,
            )
            .await
            .unwrap();

            let mut leases = PeerLeaseBook::new(donor_id);
            assert_eq!(leases.insert(lease).unwrap(), lease_id);
            let permit = leases
                .reserve(lease_id, requester_id, prepared.lease_request(), NOW)
                .unwrap();
            let permitted = leases.authorize_dispatch(&permit, NOW).unwrap();
            let mut journal_nonce = [0; 16];
            getrandom::fill(&mut journal_nonce).unwrap();
            let journal_path = std::env::temp_dir().join(format!(
                "sage-peer-transport-journal-{}-{:032x}.bin",
                std::process::id(),
                u128::from_be_bytes(journal_nonce)
            ));
            let mut journal =
                PeerDispatchJournal::open(&journal_path, Zeroizing::new([0x7c; 32])).unwrap();
            let authorized = prepared.authorize_durable(permitted, &mut journal).unwrap();
            assert!(journal.contains(worker_plan.job_id(), 0));
            let result = execute_authorized_partition(authorized).unwrap();
            send_partition_result(&mut donor_writer, result, DEADLINE)
                .await
                .unwrap();
            let settlement = leases.settle(permit, expected_output_bytes).unwrap();
            drop(journal);
            std::fs::remove_file(journal_path).unwrap();
            settlement
        });

        send_partition_request(&mut requester_writer, request, DEADLINE)
            .await
            .unwrap();
        let verified = receive_verified_partition_result(
            &mut requester_reader,
            &plan,
            0,
            donor_id,
            requester_id,
            DEADLINE,
            |spec, output| {
                spec.index() == 0 && verify_builtin_output(plan.job(), &resource_id, &input, output)
            },
        )
        .await
        .unwrap();
        assert_eq!(verified.checkpoint().outputs().len(), 1);
        assert_eq!(worker.await.unwrap().output_bytes, expected_output_bytes);
    }

    #[tokio::test]
    async fn partition_receive_rejects_a_different_authenticated_peer() {
        let (requester, donor) = paired_channels();
        let input = b"bounded input".to_vec();
        let digest: [u8; 32] = Sha256::digest(&input).into();
        let resource_id = byte_histogram_v1_resource_id();
        let plan = ComputePlan::new(
            DelegatedJobKind::BatchAnalysis,
            resource_id,
            vec![
                ComputeUnitSpec::new(0, digest, input.len() as u32, (12 + 256 * 8) as u32).unwrap(),
            ],
            1,
        )
        .unwrap();
        let lease = PeerLease::issue(
            &DeviceIdentity::from_seed([19; 32]),
            requester.local_peer_id,
            NOW - 1,
            120,
            [LeaseScope {
                job: plan.job(),
                resource_id,
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 1,
                maximum_jobs: 1,
                maximum_input_bytes_per_job: input.len() as u64,
                maximum_output_bytes_per_job: (12 + 256 * 8) as u64,
                maximum_total_input_bytes: input.len() as u64,
                maximum_total_output_bytes: (12 + 256 * 8) as u64,
            },
        )
        .unwrap();
        let request = plan
            .make_partition_request(
                &lease.grant(),
                &requester.trusted_peer,
                requester.local_peer_id,
                0,
                vec![UnitInput::new(0, input).unwrap()],
                NOW,
            )
            .unwrap();
        let (request_stream, worker_stream) = tokio::io::duplex(64 * 1024);
        let (mut writer, _reader) = split_secure_stream(request_stream, requester.channel);
        let (_worker_writer, mut worker_reader) = split_secure_stream(worker_stream, donor.channel);
        send_partition_request(&mut writer, request, DEADLINE)
            .await
            .unwrap();

        let wrong_peer = DeviceIdentity::from_seed([31; 32]).peer_id();
        let result = receive_partition_request(
            &mut worker_reader,
            &plan,
            0,
            wrong_peer,
            donor.local_peer_id,
            DEADLINE,
        )
        .await;
        assert!(matches!(result, Err(ComputeTransportError::UnexpectedPeer)));
    }

    #[tokio::test]
    async fn total_transfer_deadline_cancels_and_poisons_a_partial_send() {
        let (sender, _) = paired_channels();
        let (stream, _unread_peer) = tokio::io::duplex(1_024);
        let (mut writer, _reader) = split_secure_stream(stream, sender.channel);
        let output = vec![0xa5; crate::MAX_PEER_MESSAGE_BYTES + 4_096];
        let result = PartitionResult::new(
            [0x31; 16],
            [0x42; 32],
            0,
            vec![UnitOutput::new(0, [0x53; 32], output).unwrap()],
        )
        .unwrap();

        let error = send_partition_result(&mut writer, result, Duration::from_millis(50))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            ComputeTransportError::Transport(PeerTransportError::TimedOut)
        ));
        assert!(matches!(
            writer
                .send(
                    PeerMessageKind::JobOffer,
                    b"no reuse after cancellation",
                    DEADLINE
                )
                .await,
            Err(PeerTransportError::Poisoned)
        ));
    }

    use crate::PairingQr;
}
