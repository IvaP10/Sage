//! Core-owned bridge from paired peer compute to verified task storage.
//!
//! The requester accepts a result only from the peer authenticated by the
//! secure stream, verifies it against the immutable plan and caller's semantic
//! verifier, then persists it under the owning task. The worker consumes a
//! live lease reservation and writes the durable dispatch fence before its
//! bounded CPU executor receives any input bytes.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use sage_peer::{
    ComputePlan, ComputeTransportError, ComputeUnitSpec, LeaseSettlement, PartitionCheckpoint,
    PeerDispatchJournal, PeerError, PeerId, PeerLeaseBook, PeerLeaseGrant, PeerSecureReader,
    PeerSecureWriter, PublicPeerIdentity, UnitInput, execute_authorized_partition_with_control,
    receive_partition_request, receive_verified_partition_result, send_partition_request,
    send_partition_result,
};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    sync::watch,
};
use uuid::Uuid;

use crate::{CoreError, storage::LocalStore};

struct PeerComputeCancellationInner {
    cancelled: AtomicBool,
    state: watch::Sender<bool>,
    run_signal: Option<crate::runtime::RunSignal>,
}

/// Shared cancellation state for one peer transfer and its bounded worker.
/// Clones observe the same cancellation, which cannot be reversed.
#[derive(Clone)]
pub struct PeerComputeCancellation {
    inner: Arc<PeerComputeCancellationInner>,
}

impl PeerComputeCancellation {
    pub fn new() -> Self {
        let (state, _) = watch::channel(false);
        Self {
            inner: Arc::new(PeerComputeCancellationInner {
                cancelled: AtomicBool::new(false),
                state,
                run_signal: None,
            }),
        }
    }

    pub fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::AcqRel) {
            self.inner.state.send_replace(true);
        }
    }

    pub(crate) fn from_run_signal(signal: crate::runtime::RunSignal) -> Self {
        let initially_cancelled = signal.is_stopped();
        let (state, _) = watch::channel(initially_cancelled);
        Self {
            inner: Arc::new(PeerComputeCancellationInner {
                cancelled: AtomicBool::new(initially_cancelled),
                state,
                run_signal: Some(signal),
            }),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
            || self
                .inner
                .run_signal
                .as_ref()
                .is_some_and(crate::runtime::RunSignal::is_stopped)
    }

    pub async fn cancelled(&self) {
        let mut state = self.inner.state.subscribe();
        if let Some(signal) = &self.inner.run_signal {
            tokio::select! {
                biased;
                _ = signal.cancelled() => {},
                _ = wait_for_manual_cancel(&mut state) => {},
            }
        } else {
            wait_for_manual_cancel(&mut state).await;
        }
    }
}

async fn wait_for_manual_cancel(state: &mut watch::Receiver<bool>) {
    loop {
        if *state.borrow_and_update() {
            return;
        }
        if state.changed().await.is_err() {
            return;
        }
    }
}

impl Default for PeerComputeCancellation {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Error)]
pub enum PeerComputeError {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error(transparent)]
    Peer(#[from] PeerError),
    #[error(transparent)]
    Transport(#[from] ComputeTransportError),
    #[error("peer compute worker task failed")]
    WorkerTask(#[from] tokio::task::JoinError),
    #[error("system clock is before the Unix epoch")]
    Clock,
}

pub type PeerComputeResult<T> = Result<T, PeerComputeError>;

/// Settlement evidence for one bounded worker request. It says only that the
/// local lease quota was consumed and the result frame was sent; it does not
/// claim that the requester received or accepted the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerWorkerReceipt {
    pub job_id: [u8; 16],
    pub partition_index: u16,
    pub settlement: LeaseSettlement,
}

/// Authority and immutable work description for one remote partition.
pub struct PeerComputeSubmission<'a> {
    pub task_id: Uuid,
    pub plan: &'a ComputePlan,
    pub grant: &'a PeerLeaseGrant,
    pub expected_owner: &'a PublicPeerIdentity,
    pub local_peer: PeerId,
    pub partition_index: u16,
    pub inputs: Vec<UnitInput>,
    pub cancellation: PeerComputeCancellation,
    pub deadline: std::time::Duration,
}

/// Dependencies owned by the local peer broker for one received partition.
pub struct PeerComputeWorker<'a> {
    pub plan: &'a ComputePlan,
    pub partition_index: u16,
    pub expected_source: PeerId,
    pub local_peer: PeerId,
    pub lease_book: &'a mut PeerLeaseBook,
    pub dispatch_journal: Arc<Mutex<PeerDispatchJournal>>,
    pub cancellation: PeerComputeCancellation,
    pub deadline: std::time::Duration,
}

/// Send one plan-bound partition to the exact paired owner and durably save
/// its result only after independent plan and semantic verification.
///
/// `grant` must have arrived through an authenticated peer session. Its
/// signature, owner, recipient, scope, quota envelope and expiry are checked
/// again before any input is serialized. The verifier must be Sage-owned and
/// appropriate for this registered job; returning `false` prevents storage.
pub async fn submit_partition_and_persist<W, R, F>(
    writer: &mut PeerSecureWriter<W>,
    reader: &mut PeerSecureReader<R>,
    store: Arc<LocalStore>,
    submission: PeerComputeSubmission<'_>,
    verify_unit: F,
) -> PeerComputeResult<PartitionCheckpoint>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
    F: FnMut(&ComputeUnitSpec, &[u8]) -> bool,
{
    let cancellation = submission.cancellation.clone();
    if cancellation.is_cancelled() {
        return Err(CoreError::Cancelled.into());
    }
    let preflight_store = Arc::clone(&store);
    tokio::task::spawn_blocking(move || {
        preflight_store.validate_peer_compute_task_dispatch(submission.task_id)
    })
    .await??;
    if cancellation.is_cancelled() {
        return Err(CoreError::Cancelled.into());
    }

    let PeerComputeSubmission {
        task_id,
        plan,
        grant,
        expected_owner,
        local_peer,
        partition_index,
        inputs,
        cancellation,
        deadline,
    } = submission;
    let request = plan.make_partition_request(
        grant,
        expected_owner,
        local_peer,
        partition_index,
        inputs,
        unix_time_seconds()?,
    )?;
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(CoreError::Cancelled.into()),
        sent = send_partition_request(writer, request, deadline) => sent?,
    }

    let verified = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(CoreError::Cancelled.into()),
        verified = receive_verified_partition_result(
            reader,
            plan,
            partition_index,
            expected_owner.peer_id(),
            local_peer,
            deadline,
            verify_unit,
        ) => verified?,
    };
    let checkpoint = verified.checkpoint();
    let persisted = tokio::task::spawn_blocking(move || {
        store.save_peer_partition_checkpoint(task_id, &checkpoint)?;
        Ok::<_, CoreError>(checkpoint)
    })
    .await??;
    Ok(persisted)
}

/// Receive and execute exactly one authenticated partition request.
///
/// The remote stream identity is checked before lease lookup. The request is
/// matched to the registered plan, reserved and authorized against the live
/// lease book, and durably journaled before the closed executor sees input.
/// CPU work and journal I/O run on Tokio's blocking pool. The owning broker
/// must cancel the shared token when Sage Stop arrives; cancellation zeroizes
/// and discards partial output.
pub async fn serve_one_partition<R, W>(
    reader: &mut PeerSecureReader<R>,
    writer: &mut PeerSecureWriter<W>,
    worker: PeerComputeWorker<'_>,
) -> PeerComputeResult<PeerWorkerReceipt>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let PeerComputeWorker {
        plan,
        partition_index,
        expected_source,
        local_peer,
        lease_book,
        dispatch_journal,
        cancellation,
        deadline,
    } = worker;
    if cancellation.is_cancelled() {
        return Err(CoreError::Cancelled.into());
    }
    let prepared = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(CoreError::Cancelled.into()),
        prepared = receive_partition_request(
            reader,
            plan,
            partition_index,
            expected_source,
            local_peer,
            deadline,
        ) => prepared?,
    };
    if cancellation.is_cancelled() {
        return Err(CoreError::Cancelled.into());
    }

    let lease_id = prepared.lease_id();
    let now = unix_time_seconds()?;
    let permit = lease_book.reserve(
        lease_id,
        prepared.authenticated_peer(),
        prepared.lease_request(),
        now,
    )?;
    let permitted = match lease_book.authorize_dispatch(&permit, now) {
        Ok(permitted) => permitted,
        Err(error) => {
            let _ = lease_book.cancel(permit);
            return Err(error.into());
        }
    };

    let worker_cancel = cancellation.clone();
    let work = tokio::task::spawn_blocking(move || -> Result<_, PeerError> {
        let authorized = {
            let mut journal = dispatch_journal
                .lock()
                .map_err(|_| PeerError::DispatchJournalUnavailable)?;
            prepared.authorize_durable(permitted, &mut journal)?
        };
        let result =
            execute_authorized_partition_with_control(authorized, || worker_cancel.is_cancelled())?;
        let output_bytes = result.output_bytes();
        Ok((result, output_bytes))
    })
    .await;

    let (result, output_bytes) = match work {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            // Authorization has consumed the permit even if the durable
            // journal or bounded executor failed. Settle it with no output.
            let _ = lease_book.settle(permit, 0);
            return Err(error.into());
        }
        Err(error) => {
            let _ = lease_book.settle(permit, 0);
            return Err(error.into());
        }
    };

    let settlement = lease_book.settle(permit, output_bytes)?;
    send_partition_result(writer, result, deadline).await?;
    Ok(PeerWorkerReceipt {
        job_id: *plan.job_id(),
        partition_index,
        settlement,
    })
}

fn unix_time_seconds() -> PeerComputeResult<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PeerComputeError::Clock)?
        .as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{domain::Task, runtime::RunRegistry, secrets::SecretBytes};
    use sage_peer::{
        ComputeUnitSpec, DelegatedJobKind, DeviceIdentity, DisclosureScope, LeaseScope, PairingQr,
        PeerLease, PeerLeaseQuotas, byte_histogram_v1_resource_id, create_pairing_invitation,
        split_secure_stream, verify_builtin_output,
    };
    use sha2::{Digest, Sha256};
    use std::time::Duration;

    fn paired_channels() -> (sage_peer::PairingOutcome, sage_peer::PairingOutcome, u64) {
        let now = unix_time_seconds().unwrap();
        let owner_identity = DeviceIdentity::from_seed([0x27; 32]);
        let requester_identity = DeviceIdentity::from_seed([0x39; 32]);
        let (qr, pending_owner) = create_pairing_invitation(
            &owner_identity,
            now.saturating_sub(1),
            Duration::from_secs(120),
        )
        .unwrap();
        let (response, pending_requester) = PairingQr::from_payload(&qr.to_payload())
            .unwrap()
            .respond_after_local_confirmation(&requester_identity, now.saturating_sub(1))
            .unwrap();
        let code = pending_owner.authentication_code(&response).unwrap();
        assert_eq!(code, pending_requester.authentication_code());
        let (confirmation, owner) = pending_owner
            .confirm_after_local_confirmation(&owner_identity, &response, &code, now)
            .unwrap();
        let requester = pending_requester
            .finish_after_local_confirmation(&confirmation, &code, now)
            .unwrap();
        (requester, owner, now)
    }

    fn histogram_plan(input: &[u8]) -> ComputePlan {
        let resource_id = byte_histogram_v1_resource_id();
        ComputePlan::new(
            DelegatedJobKind::BatchAnalysis,
            resource_id,
            vec![
                ComputeUnitSpec::new(
                    0,
                    Sha256::digest(input).into(),
                    u32::try_from(input.len()).unwrap(),
                    12 + 256 * 8,
                )
                .unwrap(),
            ],
            1,
        )
        .unwrap()
    }

    async fn run_partition(
        verifier_accepts: bool,
        cancel_before_dispatch: bool,
    ) -> PeerComputeResult<(Uuid, PartitionCheckpoint)> {
        let (requester, owner, now) = paired_channels();
        let input = vec![0x5a; 1_048_576 + 4_096];
        let plan = Arc::new(histogram_plan(&input));
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(
            LocalStore::open_encrypted(
                &directory.path().join("peer-results.db"),
                &SecretBytes::new(vec![0x73; 32]),
            )
            .unwrap(),
        );
        let mut task = Task::new("compute and verify a peer partition");
        task.status = crate::domain::TaskStatus::Running;
        store.save_task(&mut task)?;
        let task_id = task.id;
        let run_registry = RunRegistry::default();
        let run_lease = run_registry.begin(task_id)?;
        if cancel_before_dispatch {
            assert!(run_registry.stop(task_id));
        }
        let cancellation = run_lease.signal.peer_compute_cancellation();

        let maximum_output_bytes = u64::from(12_u32 + 256 * 8);
        let lease = PeerLease::issue(
            &DeviceIdentity::from_seed([0x27; 32]),
            requester.local_peer_id,
            now.saturating_sub(1),
            120,
            [LeaseScope {
                job: plan.job(),
                resource_id: byte_histogram_v1_resource_id(),
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 1,
                maximum_jobs: 1,
                maximum_input_bytes_per_job: input.len() as u64,
                maximum_output_bytes_per_job: maximum_output_bytes,
                maximum_total_input_bytes: input.len() as u64,
                maximum_total_output_bytes: maximum_output_bytes,
            },
        )
        .unwrap();
        let grant = lease.grant();
        let lease_id = lease.id();
        let mut lease_book = PeerLeaseBook::new(owner.local_peer_id);
        lease_book.insert(lease).unwrap();

        let journal_path = directory.path().join("peer-dispatch.journal");
        let journal =
            PeerDispatchJournal::open(journal_path, zeroize::Zeroizing::new([0x6b; 32])).unwrap();
        let journal = Arc::new(Mutex::new(journal));
        let (request_stream, owner_stream) = tokio::io::duplex(256 * 1024);
        let (mut requester_writer, mut requester_reader) =
            split_secure_stream(request_stream, requester.channel);
        let (mut owner_writer, mut owner_reader) = split_secure_stream(owner_stream, owner.channel);
        let worker_plan = Arc::clone(&plan);
        let worker_journal = Arc::clone(&journal);
        let cancellation_for_worker = cancellation.clone();
        let worker = tokio::spawn(async move {
            serve_one_partition(
                &mut owner_reader,
                &mut owner_writer,
                PeerComputeWorker {
                    plan: &worker_plan,
                    partition_index: 0,
                    expected_source: requester.local_peer_id,
                    local_peer: owner.local_peer_id,
                    lease_book: &mut lease_book,
                    dispatch_journal: worker_journal,
                    cancellation: cancellation_for_worker,
                    deadline: Duration::from_secs(10),
                },
            )
            .await
        });

        let expected_input = input.clone();
        let input_for_initial_verification = expected_input.clone();
        let result = submit_partition_and_persist(
            &mut requester_writer,
            &mut requester_reader,
            Arc::clone(&store),
            PeerComputeSubmission {
                task_id,
                plan: &plan,
                grant: &grant,
                expected_owner: &requester.trusted_peer,
                local_peer: requester.local_peer_id,
                partition_index: 0,
                inputs: vec![UnitInput::new(0, input)?],
                cancellation: cancellation.clone(),
                deadline: Duration::from_secs(10),
            },
            move |spec, output| {
                verifier_accepts
                    && Sha256::digest(&input_for_initial_verification).as_slice()
                        == spec.input_digest()
                    && verify_builtin_output(
                        DelegatedJobKind::BatchAnalysis,
                        &byte_histogram_v1_resource_id(),
                        &input_for_initial_verification,
                        output,
                    )
            },
        )
        .await;
        if cancel_before_dispatch {
            assert!(matches!(
                &result,
                Err(PeerComputeError::Core(CoreError::Cancelled))
            ));
            assert!(matches!(
                worker.await.unwrap(),
                Err(PeerComputeError::Core(CoreError::Cancelled))
            ));
            assert!(journal.lock().unwrap().is_empty());
            assert!(
                store
                    .load_peer_partition_checkpoint(task_id, &plan, 0, |_, _| true)?
                    .is_none()
            );
            return Err(CoreError::Cancelled.into());
        }
        if verifier_accepts {
            let checkpoint = result?;
            let worker_receipt = worker.await??;
            assert_eq!(worker_receipt.job_id, *plan.job_id());
            assert_eq!(
                worker_receipt.settlement.output_bytes,
                checkpoint.outputs()[0].bytes().len() as u64
            );
            assert!(journal.lock().unwrap().contains(plan.job_id(), 0));
            assert_eq!(lease_id, grant.id());
            let restored = store
                .load_peer_partition_checkpoint(task_id, &plan, 0, |spec, output| {
                    Sha256::digest(&expected_input).as_slice() == spec.input_digest()
                        && verify_builtin_output(
                            DelegatedJobKind::BatchAnalysis,
                            &byte_histogram_v1_resource_id(),
                            &expected_input,
                            output,
                        )
                })?
                .expect("verified result was persisted under its task id");
            assert_eq!(restored.result_digest(), checkpoint.result_digest());
            Ok((task_id, checkpoint))
        } else {
            assert!(matches!(
                &result,
                Err(PeerComputeError::Transport(ComputeTransportError::Peer(
                    PeerError::ComputeVerificationFailed
                )))
            ));
            worker.await??;
            assert!(
                store
                    .load_peer_partition_checkpoint(task_id, &plan, 0, |_, _| true)?
                    .is_none(),
                "rejected semantic result must not reach encrypted checkpoint storage"
            );
            result.map(|checkpoint| (task_id, checkpoint))
        }
    }

    #[tokio::test]
    async fn paired_partition_is_lease_fenced_verified_and_saved_to_its_task() {
        let (task_id, checkpoint) = run_partition(true, false).await.unwrap();
        assert_eq!(checkpoint.partition_index(), 0);
        assert_eq!(checkpoint.outputs().len(), 1);
        assert_ne!(task_id, Uuid::nil());
    }

    #[tokio::test]
    async fn rejected_semantic_result_never_returns_a_checkpoint() {
        assert!(run_partition(false, false).await.is_err());
    }

    #[tokio::test]
    async fn stop_before_dispatch_leaves_no_journal_fence_or_saved_result() {
        assert!(matches!(
            run_partition(true, true).await,
            Err(PeerComputeError::Core(CoreError::Cancelled))
        ));
    }
}
