//! Runtime cancellation is independent of the durable task projection. A lease
//! owns one execution generation, including cleanup; it cannot be replaced by
//! Resume while its prior future is still unwinding.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::watch;
use uuid::Uuid;

use crate::{CoreError, CoreResult};

#[derive(Clone, Debug)]
pub(crate) struct RunSignal {
    generation: Uuid,
    stopped: watch::Sender<bool>,
    scope_id: Uuid,
    scope_stopped: watch::Sender<bool>,
    held: watch::Sender<bool>,
    retired_actions: Arc<std::sync::RwLock<std::collections::BTreeSet<Uuid>>>,
}

impl RunSignal {
    pub(crate) fn peer_compute_cancellation(&self) -> crate::peer_compute::PeerComputeCancellation {
        crate::peer_compute::PeerComputeCancellation::from_run_signal(self.clone())
    }

    pub(crate) fn action_retired(&self, id: Uuid) -> bool {
        self.retired_actions
            .read()
            .expect("action retirement poisoned")
            .contains(&id)
    }
    pub(crate) fn is_held(&self) -> bool {
        *self.held.borrow()
    }
    pub(crate) async fn resumed(&self) -> CoreResult<()> {
        let mut held = self.held.subscribe();
        loop {
            if self.is_stopped() {
                return Err(CoreError::Cancelled);
            }
            if !*held.borrow_and_update() {
                return Ok(());
            }
            tokio::select! {
                _ = self.cancelled() => return Err(CoreError::Cancelled),
                changed = held.changed() => if changed.is_err() { return Err(CoreError::Cancelled); },
            }
        }
    }
    pub(crate) fn is_stopped(&self) -> bool {
        *self.stopped.borrow() || *self.scope_stopped.borrow()
    }
    pub(crate) async fn cancelled(&self) {
        tokio::select! {
            _ = Self::wait_for_stop(self.stopped.subscribe()) => {},
            _ = Self::wait_for_stop(self.scope_stopped.subscribe()) => {},
        }
    }
    async fn wait_for_stop(mut state: watch::Receiver<bool>) {
        loop {
            if *state.borrow_and_update() {
                return;
            }
            if state.changed().await.is_err() {
                return;
            }
        }
    }
}

struct Entry {
    signal: RunSignal,
    active: bool,
    accepted: bool,
    stop_persisted: bool,
    pending_finish: Option<(u64, crate::finalization::Finish)>,
}

#[derive(Clone, Default)]
pub(crate) struct RunRegistry {
    entries: Arc<Mutex<HashMap<Uuid, Entry>>>,
}

impl std::fmt::Debug for RunRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RunRegistry")
    }
}

pub(crate) struct RunLease {
    registry: RunRegistry,
    task_id: Uuid,
    pub signal: RunSignal,
}

impl RunRegistry {
    pub(crate) fn begin(&self, task_id: Uuid) -> CoreResult<RunLease> {
        self.begin_scoped(task_id, task_id)
    }

    pub(crate) fn begin_scoped(&self, task_id: Uuid, scope_id: Uuid) -> CoreResult<RunLease> {
        self.begin_owned(task_id, scope_id, None)
    }

    pub(crate) fn continue_run(
        &self,
        prior_id: Uuid,
        prior: &RunSignal,
        task_id: Uuid,
    ) -> CoreResult<RunLease> {
        self.begin_owned(task_id, prior.scope_id, Some((prior_id, prior)))
    }

    fn begin_owned(
        &self,
        task_id: Uuid,
        scope_id: Uuid,
        prior: Option<(Uuid, &RunSignal)>,
    ) -> CoreResult<RunLease> {
        let mut entries = self.entries.lock().expect("run registry poisoned");
        if let Some(entry) = entries.get(&task_id) {
            return Err(if entry.active {
                CoreError::Busy("The previous execution is still running or stopping".into())
            } else {
                CoreError::PermissionRequired(
                    "The prior run has unsaved state. Retry its control before continuing this task."
                        .into(),
                )
            });
        }
        if entries.len() >= 64 {
            return Err(CoreError::Busy(
                "The active-run control capacity has been reached".into(),
            ));
        }
        let scope_stopped = if let Some((prior_id, prior)) = prior {
            let entry = entries
                .get(&prior_id)
                .filter(|entry| {
                    entry.active
                        && entry.signal.generation == prior.generation
                        && !entry.signal.is_stopped()
                })
                .ok_or(CoreError::Cancelled)?;
            entry.signal.scope_stopped.clone()
        } else {
            if entries
                .values()
                .any(|entry| entry.signal.scope_id == scope_id)
            {
                return Err(CoreError::Busy(
                    "The control scope still has an active execution or unsaved state".into(),
                ));
            }
            watch::channel(false).0
        };
        let signal = RunSignal {
            generation: Uuid::new_v4(),
            stopped: watch::channel(false).0,
            scope_id,
            scope_stopped,
            held: prior
                .map(|(_, signal)| signal.held.clone())
                .unwrap_or_else(|| watch::channel(false).0),
            retired_actions: Default::default(),
        };
        entries.insert(
            task_id,
            Entry {
                signal: signal.clone(),
                active: true,
                accepted: false,
                stop_persisted: false,
                pending_finish: None,
            },
        );
        Ok(RunLease {
            registry: self.clone(),
            task_id,
            signal,
        })
    }

    /// This path takes no task-cache, database or credential-store lock.
    pub(crate) fn stop(&self, task_id: Uuid) -> bool {
        let entries = self.entries.lock().expect("run registry poisoned");
        let entry = entries.get(&task_id).or_else(|| {
            entries
                .values()
                .find(|entry| entry.signal.scope_id == task_id)
        });
        let Some(entry) = entry else {
            return false;
        };
        entry.signal.scope_stopped.send_replace(true);
        true
    }

    pub(crate) fn is_active(&self, task_id: Uuid) -> bool {
        self.entries
            .lock()
            .expect("run registry poisoned")
            .get(&task_id)
            .is_some_and(|entry| entry.active)
    }

    pub(crate) fn scope_tasks(&self, id: Uuid) -> Vec<Uuid> {
        let entries = self.entries.lock().expect("run registry poisoned");
        let scope = entries.get(&id).map_or(id, |entry| entry.signal.scope_id);
        entries
            .iter()
            .filter(|(_, entry)| entry.accepted && entry.signal.scope_id == scope)
            .map(|(task_id, _)| *task_id)
            .collect()
    }

    pub(crate) fn is_stopped(&self, task_id: Uuid) -> bool {
        self.entries
            .lock()
            .expect("run registry poisoned")
            .get(&task_id)
            .is_some_and(|entry| *entry.signal.scope_stopped.borrow())
    }

    pub(crate) fn hold(&self, task_id: Uuid, held: bool) -> bool {
        let entries = self.entries.lock().expect("run registry poisoned");
        let entry = entries.get(&task_id).or_else(|| {
            entries
                .values()
                .find(|entry| entry.signal.scope_id == task_id)
        });
        if let Some(entry) = entry.filter(|entry| entry.active) {
            entry.signal.held.send_replace(held);
            true
        } else {
            false
        }
    }

    pub(crate) fn is_held(&self, task_id: Uuid) -> bool {
        self.entries
            .lock()
            .expect("run registry poisoned")
            .get(&task_id)
            .is_some_and(|entry| entry.signal.is_held())
    }

    pub(crate) fn retire_actions(&self, task_id: Uuid, actions: &std::collections::BTreeSet<Uuid>) {
        if let Some(entry) = self
            .entries
            .lock()
            .expect("run registry poisoned")
            .get(&task_id)
        {
            entry
                .signal
                .retired_actions
                .write()
                .expect("action retirement poisoned")
                .extend(actions);
        }
    }

    pub(crate) fn action_retired(&self, task_id: Uuid, action_id: Uuid) -> bool {
        self.entries
            .lock()
            .expect("run registry poisoned")
            .get(&task_id)
            .is_some_and(|entry| entry.signal.action_retired(action_id))
    }

    pub(crate) fn current(&self, task_id: Uuid) -> CoreResult<RunSignal> {
        self.entries
            .lock()
            .expect("run registry poisoned")
            .get(&task_id)
            .filter(|entry| entry.active && !entry.signal.is_stopped())
            .map(|entry| entry.signal.clone())
            .ok_or(CoreError::Cancelled)
    }

    pub(crate) fn persisted_stop(&self, task_id: Uuid) {
        let mut entries = self.entries.lock().expect("run registry poisoned");
        if let Some(entry) = entries.get_mut(&task_id) {
            entry.stop_persisted = true;
            entry.pending_finish = None;
            if !entry.active {
                entries.remove(&task_id);
            }
        }
    }

    pub(crate) fn project_pending_stop(&self, task: &mut crate::domain::Task) {
        let entries = self.entries.lock().expect("run registry poisoned");
        if entries
            .get(&task.id)
            .is_some_and(|entry| *entry.signal.scope_stopped.borrow() && !entry.stop_persisted)
        {
            task.status = crate::domain::TaskStatus::Cancelled;
            task.final_outcome = Some("Stop requested. Saving this state is still pending; in-flight effects may need review.".into());
        } else if entries
            .get(&task.id)
            .is_some_and(|entry| entry.active && entry.signal.is_held())
            && task.status.is_active()
        {
            task.status = crate::domain::TaskStatus::Paused;
            task.final_outcome = Some(
                "Paused before further actions. Already dispatched effects may still finish."
                    .into(),
            );
        } else if entries
            .get(&task.id)
            .is_some_and(|entry| entry.pending_finish.is_some())
        {
            task.status = crate::domain::TaskStatus::Interrupted;
            task.final_outcome = Some("The run ended, but its final record could not be saved. Retry Resume after storage is available.".into());
        }
    }

    pub(crate) fn retain_finish(
        &self,
        task_id: Uuid,
        attempt: u64,
        finish: crate::finalization::Finish,
    ) {
        if let Some(entry) = self
            .entries
            .lock()
            .expect("run registry poisoned")
            .get_mut(&task_id)
            && (entry.pending_finish.is_none()
                || matches!(finish, crate::finalization::Finish::Stop))
        {
            entry.pending_finish = Some((attempt, finish));
        }
    }

    pub(crate) fn pending_finish(
        &self,
        task_id: Uuid,
    ) -> Option<(u64, crate::finalization::Finish)> {
        self.entries
            .lock()
            .expect("run registry poisoned")
            .get(&task_id)
            .and_then(|entry| entry.pending_finish.clone())
    }

    pub(crate) fn finish_saved(&self, task_id: Uuid) {
        let mut entries = self.entries.lock().expect("run registry poisoned");
        if let Some(entry) = entries.get_mut(&task_id) {
            entry.pending_finish = None;
            if !entry.active && (!*entry.signal.scope_stopped.borrow() || entry.stop_persisted) {
                entries.remove(&task_id);
            }
        }
    }
}

impl RunLease {
    pub(crate) fn mark_accepted(&self) {
        let mut entries = self.registry.entries.lock().expect("run registry poisoned");
        if let Some(entry) = entries.get_mut(&self.task_id)
            && entry.signal.generation == self.signal.generation
        {
            entry.accepted = true;
        }
    }
}

impl Drop for RunLease {
    fn drop(&mut self) {
        let mut entries = self.registry.entries.lock().expect("run registry poisoned");
        if let Some(entry) = entries.get_mut(&self.task_id)
            && entry.signal.generation == self.signal.generation
        {
            let was_stopped = entry.signal.is_stopped();
            entry.active = false;
            // Outstanding grants retain this signal after the registry entry
            // disappears. Completing normally must retire their authority too.
            entry.signal.stopped.send_replace(true);
            if !entry.accepted
                || entry.stop_persisted
                || (!was_stopped && entry.pending_finish.is_none())
            {
                entries.remove(&self.task_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn handoff_shares_stop_but_retires_execution_grants_independently() {
        let registry = RunRegistry::default();
        let root = Uuid::new_v4();
        let parent = registry.begin(root).unwrap();
        parent.mark_accepted();
        let old_signal = parent.signal.clone();
        let child_id = Uuid::new_v4();
        let child = registry
            .continue_run(root, &parent.signal, child_id)
            .unwrap();
        child.mark_accepted();
        assert_ne!(old_signal.generation, child.signal.generation);
        drop(parent);
        assert!(old_signal.is_stopped());
        assert!(!child.signal.is_stopped());
        let mut owner = child;
        let mut owner_id = child_id;
        // Long continuation chains retain a stable control address without
        // accumulating one runtime alias/entry for every historical task.
        for _ in 0..128 {
            let next_id = Uuid::new_v4();
            let next = registry
                .continue_run(owner_id, &owner.signal, next_id)
                .unwrap();
            next.mark_accepted();
            drop(owner);
            owner = next;
            owner_id = next_id;
            assert_eq!(registry.scope_tasks(root), vec![owner_id]);
            assert!(!owner.signal.is_stopped());
        }
        assert!(
            registry.stop(root),
            "The original control identity still reaches the child"
        );
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            owner.signal.cancelled(),
        )
        .await
        .unwrap();
        assert!(registry.current(owner_id).is_err());
        assert!(
            registry
                .continue_run(owner_id, &owner.signal, Uuid::new_v4())
                .is_err()
        );
        registry.persisted_stop(owner_id);
        drop(owner);
        assert!(registry.scope_tasks(root).is_empty());
    }

    #[test]
    fn cancelled_unaccepted_reservations_do_not_leave_unsaveable_tombstones() {
        let registry = RunRegistry::default();
        let root = Uuid::new_v4();
        let parent = registry.begin(root).unwrap();
        parent.mark_accepted();
        let child_id = Uuid::new_v4();
        let child = registry
            .continue_run(root, &parent.signal, child_id)
            .unwrap();
        registry.stop(root);
        drop(child);
        assert!(!registry.is_stopped(child_id));
        assert_eq!(registry.scope_tasks(root), vec![root]);
        assert!(
            registry
                .continue_run(root, &parent.signal, Uuid::new_v4())
                .is_err()
        );
    }
    #[test]
    fn unsaved_finishes_retain_bounded_control_slots_and_retire_all_authority() {
        let registry = RunRegistry::default();
        let mut tasks = Vec::new();
        for _ in 0..64 {
            let id = Uuid::new_v4();
            let lease = registry.begin(id).unwrap();
            lease.mark_accepted();
            let signal = lease.signal.clone();
            registry.retain_finish(id, 0, crate::finalization::Finish::Answer("fixture".into()));
            drop(lease);
            assert!(signal.is_stopped());
            assert!(!registry.is_stopped(id));
            assert!(!registry.is_active(id));
            assert!(registry.current(id).is_err());
            assert!(registry.begin(id).is_err());
            tasks.push(id);
        }
        assert!(registry.begin(Uuid::new_v4()).is_err());
        for id in tasks {
            registry.finish_saved(id);
        }
        assert!(registry.entries.lock().unwrap().is_empty());
        assert!(registry.begin(Uuid::new_v4()).is_ok());
    }

    #[tokio::test]
    async fn stop_survives_missing_waiter_and_unsaved_cleanup_and_generations_do_not_reopen() {
        let registry = RunRegistry::default();
        let id = Uuid::new_v4();
        let lease = registry.begin(id).unwrap();
        lease.mark_accepted();
        let old = lease.signal.clone();
        assert!(registry.begin(id).is_err());
        assert!(registry.stop(id));
        tokio::time::timeout(std::time::Duration::from_millis(100), old.cancelled())
            .await
            .unwrap();
        assert!(registry.current(id).is_err());
        drop(lease);
        assert!(!registry.is_active(id));
        assert!(
            registry.begin(id).is_err(),
            "Unsaved Stop is retained after the future exits"
        );
        registry.persisted_stop(id);
        let next = registry.begin(id).unwrap();
        assert_ne!(next.signal.generation, old.generation);
        assert!(old.is_stopped());
        assert!(!next.signal.is_stopped());
    }
}
