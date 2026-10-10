//! Small owned-future scheduler. Dropping a run drops every child future in the
//! same stack; no detached action can survive the run's cancellation owner.
use std::collections::{BTreeMap, BTreeSet};
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::task::Poll;
use uuid::Uuid;

use crate::domain::{Action, ActionStatus, Task};

pub(crate) const MAX_PARALLEL_READS: usize = 4;

type ActionFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub(crate) struct InFlight<'a, T> {
    entries: Vec<(Uuid, ActionFuture<'a, T>)>,
}

impl<'a, T> InFlight<'a, T> {
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(MAX_PARALLEL_READS),
        }
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn contains(&self, id: Uuid) -> bool {
        self.entries.iter().any(|(key, _)| *key == id)
    }
    pub fn retain(&mut self, current: impl Fn(Uuid) -> bool) {
        self.entries.retain(|(id, _)| current(*id));
    }

    pub fn ids(&self) -> impl Iterator<Item = Uuid> + '_ {
        self.entries.iter().map(|(id, _)| *id)
    }
    pub fn push(&mut self, id: Uuid, future: impl Future<Output = T> + Send + 'a) {
        assert!(self.entries.len() < MAX_PARALLEL_READS);
        assert!(!self.contains(id));
        self.entries.push((id, Box::pin(future)));
    }
    pub async fn next(&mut self) -> Option<(Uuid, T)> {
        poll_fn(|cx| {
            if self.entries.is_empty() {
                return Poll::Ready(None);
            }
            for i in 0..self.entries.len() {
                if let Poll::Ready(value) = self.entries[i].1.as_mut().poll(cx) {
                    let (id, _) = self.entries.swap_remove(i);
                    return Poll::Ready(Some((id, value)));
                }
            }
            Poll::Pending
        })
        .await
    }
}

/// Only explicit scoped local reads share a wave. Requests needing review,
/// external egress and mutations retain serial admission and exact approvals.
pub(crate) fn parallel_read(task: &Task, id: Uuid) -> bool {
    task.actions.get(&id).is_some_and(|state| {
        matches!(
            state.proposal.action,
            Action::ReadFile { .. } | Action::ListDirectory { .. }
        ) && task
            .contract
            .as_ref()
            .is_some_and(|contract| contract.covers(&state.proposal.action))
    })
}

/// Cost is an estimate of local observed execution time, never an authority or
/// success prediction. Fixed operation keys bound statistics independently of
/// filenames, transcript count or conversation history.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PersistedTiming {
    pub operation: String,
    pub ewma_micros: u64,
    pub attempts: u64,
    pub successes: u64,
    pub failures: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct TimingMeasurement {
    pub operation: String,
    pub elapsed_micros: u64,
    pub succeeded: bool,
}

enum TimingWriteMessage {
    Sample(TimingMeasurement),
    #[cfg(test)]
    Flush(std::sync::mpsc::SyncSender<()>),
}

const MAX_TIMING_OPERATIONS: usize = 64;
const MAX_TIMING_SAMPLES: u64 = 1_000_000;

#[derive(Debug, Default)]
pub(crate) struct Timings {
    operations: BTreeMap<String, PersistedTiming>,
}

impl Timings {
    pub fn restore(&mut self, records: impl IntoIterator<Item = PersistedTiming>) {
        self.operations.clear();
        for record in records.into_iter().take(MAX_TIMING_OPERATIONS) {
            if valid_operation(&record.operation)
                && (1..=60_000_000).contains(&record.ewma_micros)
                && record.attempts > 0
                && record.attempts <= MAX_TIMING_SAMPLES
                && record.successes.saturating_add(record.failures) == record.attempts
            {
                self.operations.insert(record.operation.clone(), record);
            }
        }
    }

    pub fn record(&mut self, operation: &'static str, elapsed: u64, succeeded: bool) {
        let sample = elapsed.clamp(1, 60_000_000);
        self.operations
            .entry(operation.to_owned())
            .and_modify(|prior| {
                prior.ewma_micros = (prior.ewma_micros * 7 + sample) / 8;
                if prior.attempts < MAX_TIMING_SAMPLES {
                    prior.attempts += 1;
                    if succeeded {
                        prior.successes += 1;
                    } else {
                        prior.failures += 1;
                    }
                } else {
                    prior.attempts = (prior.attempts / 2) + 1;
                    prior.successes = (prior.successes / 2) + u64::from(succeeded);
                    prior.failures = (prior.failures / 2) + u64::from(!succeeded);
                }
            })
            .or_insert_with(|| PersistedTiming {
                operation: operation.to_owned(),
                ewma_micros: sample,
                attempts: 1,
                successes: u64::from(succeeded),
                failures: u64::from(!succeeded),
            });
    }

    fn estimate(&self, action: &Action) -> u64 {
        self.operations
            .get(action.kind())
            .map(|record| record.ewma_micros)
            .unwrap_or(match action {
                Action::ReadFile { .. } => 2_000,
                Action::ListDirectory { .. } => 8_000,
                Action::OpenApplication { .. } => 250_000,
                _ => 20_000,
            })
    }
    pub fn ready(&self, task: &Task) -> Vec<Uuid> {
        fn cost(
            id: Uuid,
            task: &Task,
            timings: &Timings,
            costs: &mut BTreeMap<Uuid, u64>,
            visiting: &mut BTreeSet<Uuid>,
        ) -> u64 {
            if let Some(cost) = costs.get(&id) {
                return *cost;
            }
            if !visiting.insert(id) {
                return 0;
            } // Stored plans still fail validation independently.
            let successor = task
                .dependencies
                .iter()
                .filter(|(child, parents)| {
                    parents.contains(&id)
                        && task
                            .actions
                            .get(child)
                            .is_some_and(|a| a.status == ActionStatus::Pending)
                })
                .map(|(child, _)| cost(*child, task, timings, costs, visiting))
                .max()
                .unwrap_or(0);
            visiting.remove(&id);
            let own = task
                .actions
                .get(&id)
                .map_or(0, |s| timings.estimate(&s.proposal.action));
            let value = own.saturating_add(successor);
            costs.insert(id, value);
            value
        }
        let mut ready = task.ready_actions();
        let mut costs = BTreeMap::new();
        for id in &ready {
            cost(*id, task, self, &mut costs, &mut BTreeSet::new());
        }
        ready.sort_by_key(|id| (std::cmp::Reverse(costs[id]), *id));
        ready
    }
}

fn valid_operation(operation: &str) -> bool {
    !operation.is_empty()
        && operation.len() <= 64
        && operation
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

/// Persists low-cardinality executor timing samples away from the control
/// future. Saturation only drops optimization data; it cannot delay execution
/// or change the authority decision.
pub(crate) struct TimingPersistence {
    sender: Option<SyncSender<TimingWriteMessage>>,
}

impl std::fmt::Debug for TimingPersistence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TimingPersistence")
            .field("available", &self.sender.is_some())
            .finish()
    }
}

impl TimingPersistence {
    pub fn new(store: crate::storage::LocalStore) -> Self {
        let (sender, receiver) = sync_channel::<TimingWriteMessage>(128);
        let worker = std::thread::Builder::new()
            .name("sage-execution-timing-store".into())
            .spawn(move || {
                while let Ok(message) = receiver.recv() {
                    match message {
                        TimingWriteMessage::Sample(sample) => {
                            let _ = store.record_execution_timing(
                                &sample.operation,
                                sample.elapsed_micros,
                                sample.succeeded,
                            );
                        }
                        #[cfg(test)]
                        TimingWriteMessage::Flush(acknowledgement) => {
                            let _ = acknowledgement.send(());
                        }
                    }
                }
            });
        Self {
            sender: worker.ok().map(|_| sender),
        }
    }

    pub fn submit(&self, sample: TimingMeasurement) {
        if let Some(sender) = &self.sender {
            let _ = sender.try_send(TimingWriteMessage::Sample(sample));
        }
    }

    #[cfg(test)]
    fn flush_for_test(&self) {
        let sender = self.sender.as_ref().expect("timing writer started");
        let (acknowledgement, received) = std::sync::mpsc::sync_channel(0);
        sender
            .send(TimingWriteMessage::Flush(acknowledgement))
            .expect("timing writer accepts a flush barrier");
        received
            .recv()
            .expect("timing writer crosses the flush barrier");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_background_timing_writer_flushes_queued_samples() {
        let directory = tempfile::tempdir().unwrap();
        let store = crate::storage::LocalStore::open_encrypted(
            &directory.path().join("timing-writer.db"),
            &crate::secrets::SecretBytes::new(vec![73; 32]),
        )
        .unwrap();
        let writer = TimingPersistence::new(store.clone());
        writer.submit(TimingMeasurement {
            operation: "read_file".into(),
            elapsed_micros: 3_200,
            succeeded: true,
        });
        writer.flush_for_test();

        let records = store.load_execution_timings().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].operation, "read_file");
        assert_eq!(records[0].ewma_micros, 3_200);
        assert_eq!((records[0].attempts, records[0].successes), (1, 1));
    }

    #[test]
    fn restored_executor_timings_drive_the_existing_ready_cost_estimate() {
        let action = Action::ReadFile {
            path: std::path::PathBuf::from("/tmp/evidence.txt"),
            max_bytes: 1024,
        };
        let mut timings = Timings::default();
        timings.restore([
            PersistedTiming {
                operation: "read_file".into(),
                ewma_micros: 12_000,
                attempts: 10,
                successes: 9,
                failures: 1,
            },
            PersistedTiming {
                operation: "../tmp".into(),
                ewma_micros: u64::MAX,
                attempts: 1,
                successes: 1,
                failures: 0,
            },
        ]);
        assert_eq!(timings.estimate(&action), 12_000);
        timings.record("read_file", 4_000, false);
        let sample = timings.operations.get("read_file").unwrap();
        assert_eq!(sample.ewma_micros, 11_000);
        assert_eq!(
            (sample.attempts, sample.successes, sample.failures),
            (11, 9, 2)
        );
        assert_eq!(timings.operations.len(), 1);
    }

    #[tokio::test]
    async fn dropping_run_synchronously_drops_all_children() {
        struct Guard(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut pending = InFlight::new();
        for _ in 0..4 {
            let guard = Guard(dropped.clone());
            pending.push(Uuid::new_v4(), async move {
                let _guard = guard;
                std::future::pending::<()>().await
            });
        }
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(1), pending.next())
                .await
                .is_err()
        );
        drop(pending);
        assert_eq!(dropped.load(std::sync::atomic::Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn returns_completed_child_without_waiting_for_first() {
        let mut pending = InFlight::new();
        pending.push(Uuid::new_v4(), std::future::pending::<u32>());
        let fast = Uuid::new_v4();
        pending.push(fast, async { 42 });
        assert_eq!(pending.next().await, Some((fast, 42)));
        assert_eq!(pending.len(), 1);
    }
}
