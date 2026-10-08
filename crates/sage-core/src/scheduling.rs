//! Small owned-future scheduler. Dropping a run drops every child future in the
//! same stack; no detached action can survive the run's cancellation owner.
use std::collections::{BTreeMap, BTreeSet};
use std::future::{Future, poll_fn};
use std::pin::Pin;
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
#[derive(Debug, Default)]
pub(crate) struct Timings {
    micros: BTreeMap<&'static str, u64>,
}

impl Timings {
    pub fn record(&mut self, operation: &'static str, elapsed: u64) {
        let sample = elapsed.clamp(1, 60_000_000);
        self.micros
            .entry(operation)
            .and_modify(|prior| *prior = (*prior * 7 + sample) / 8)
            .or_insert(sample);
    }
    fn estimate(&self, action: &Action) -> u64 {
        self.micros
            .get(action.kind())
            .copied()
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

#[cfg(test)]
mod tests {
    use super::*;
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
