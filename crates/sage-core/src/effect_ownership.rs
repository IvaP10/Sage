//! Bounded execution admission and resource-level effect ownership.
//!
//! Disjoint prepared targets can run concurrently. Reads share ownership of an
//! exact resource, writes exclude reads and writes of that resource, directory
//! listings exclude child creation, and Undo fences every Sage dispatch. The
//! broker still owns policy and authority.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, Weak};

use tokio::sync::{
    OwnedRwLockReadGuard, OwnedRwLockWriteGuard, OwnedSemaphorePermit, RwLock, Semaphore,
};

use crate::contracts::{PreparedAction, PreparedTarget};
use crate::domain::Action;
use crate::error::{CoreError, CoreResult};

const MAX_ACTIVE_DISPATCHES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    Shared,
    Exclusive,
}

impl Access {
    fn merge(&mut self, other: Self) {
        if other == Self::Exclusive {
            *self = Self::Exclusive;
        }
    }
}

struct Footprint {
    resources: BTreeMap<String, Access>,
}

impl Footprint {
    fn from_prepared(prepared: &PreparedAction) -> CoreResult<Self> {
        let mut footprint = Self {
            resources: BTreeMap::new(),
        };
        let mut add = |key: String, access: Access| {
            footprint
                .resources
                .entry(key)
                .and_modify(|current| current.merge(access))
                .or_insert(access);
        };
        match (&prepared.intent.proposal.action, &prepared.target) {
            (
                Action::ReadFile { path, .. },
                PreparedTarget::File {
                    path: prepared_path,
                    ..
                },
            ) if path == prepared_path => {
                add(file_key(path), Access::Shared);
            }
            (
                Action::ListDirectory { path, .. },
                PreparedTarget::File {
                    path: prepared_path,
                    ..
                },
            ) if path == prepared_path => {
                add(file_key(path), Access::Shared);
                add(namespace_key(path), Access::Exclusive);
            }
            (
                Action::WriteFile { path, .. },
                PreparedTarget::File {
                    path: prepared_path,
                    before,
                },
            ) if path == prepared_path => {
                add(file_key(path), Access::Exclusive);
                if before.is_none() {
                    add(add_parent_namespace(path)?, Access::Shared);
                }
            }
            (
                Action::CreateFolder { path },
                PreparedTarget::File {
                    path: prepared_path,
                    ..
                },
            ) if path == prepared_path => {
                add(file_key(path), Access::Exclusive);
                add(add_parent_namespace(path)?, Access::Shared);
            }
            (
                Action::FetchPublic {
                    url: requested_url, ..
                },
                PreparedTarget::Network { url },
            ) if requested_url == url => {
                let parsed = url::Url::parse(url).map_err(|_| {
                    CoreError::InvalidAction("Prepared fetch URL is invalid".into())
                })?;
                if !parsed.username().is_empty() || parsed.password().is_some() {
                    return Err(CoreError::PolicyDenied(
                        "Prepared fetch targets cannot contain credentials".into(),
                    ));
                }
                add(
                    format!("network:{}", parsed.origin().ascii_serialization()),
                    Access::Shared,
                );
            }
            (Action::OpenApplication { application }, PreparedTarget::Application { identity })
                if application == &identity.identifier =>
            {
                add(
                    format!("application:{}", identity.identifier),
                    Access::Exclusive,
                );
            }
            (Action::NavigateUrl { .. }, PreparedTarget::Browser { document }) => {
                add(
                    format!("browser:{}:{}", document.window_id, document.tab_id),
                    Access::Exclusive,
                );
            }
            (Action::AskUser { .. }, PreparedTarget::User) => {}
            _ => {
                return Err(CoreError::CapabilityRejected(
                    "Effect ownership target does not match the prepared action".into(),
                ));
            }
        }
        Ok(footprint)
    }
}

fn file_key(path: &std::path::Path) -> String {
    format!("file:{}", path.to_string_lossy())
}

fn namespace_key(path: &std::path::Path) -> String {
    format!("directory-namespace:{}", path.to_string_lossy())
}

fn add_parent_namespace(path: &std::path::Path) -> CoreResult<String> {
    path.parent()
        .map(namespace_key)
        .ok_or_else(|| CoreError::InvalidAction("Prepared file has no parent directory".into()))
}

enum LockGuard {
    Shared { _guard: OwnedRwLockReadGuard<()> },
    Exclusive { _guard: OwnedRwLockWriteGuard<()> },
}

/// RAII lease held from final target revalidation through fresh verification.
pub(crate) struct EffectLease {
    _resources: Vec<LockGuard>,
    _dispatch_slot: Option<OwnedSemaphorePermit>,
    // Keep the global fence last so it is released after every resource lock.
    _global: Option<LockGuard>,
}

/// Bounded scheduler for Sage action effects. Weak entries are pruned during
/// admission so finished resource keys do not accumulate indefinitely.
pub(crate) struct EffectArbiter {
    global: Arc<RwLock<()>>,
    dispatch_slots: Arc<Semaphore>,
    resources: Mutex<HashMap<String, Weak<RwLock<()>>>>,
}

impl Default for EffectArbiter {
    fn default() -> Self {
        Self {
            global: Arc::new(RwLock::new(())),
            dispatch_slots: Arc::new(Semaphore::new(MAX_ACTIVE_DISPATCHES)),
            resources: Mutex::new(HashMap::new()),
        }
    }
}

impl EffectArbiter {
    pub(crate) async fn acquire_action(
        &self,
        prepared: &PreparedAction,
    ) -> CoreResult<EffectLease> {
        let footprint = Footprint::from_prepared(prepared)?;
        // A question waits on the person rather than owning a system effect.
        // It must not consume dispatch capacity or delay global compensation.
        if footprint.resources.is_empty() {
            return Ok(EffectLease {
                _resources: Vec::new(),
                _dispatch_slot: None,
                _global: None,
            });
        }
        let dispatch_slot = self
            .dispatch_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| CoreError::Cancelled)?;
        let global = acquire_lock(self.global.clone(), Access::Shared).await;
        let resource_locks = {
            let mut resources = self.resources.lock().map_err(|_| {
                CoreError::ExecutionFailed("Effect resource table is unavailable".into())
            })?;
            resources.retain(|_, lock| lock.strong_count() > 0);
            let mut locks = Vec::with_capacity(footprint.resources.len());
            for key in footprint.resources.keys() {
                let lock = resources
                    .get(key)
                    .and_then(Weak::upgrade)
                    .unwrap_or_else(|| {
                        let owned = Arc::new(RwLock::new(()));
                        resources.insert(key.clone(), Arc::downgrade(&owned));
                        owned
                    });
                locks.push(lock);
            }
            locks
        };
        let mut guards = Vec::with_capacity(resource_locks.len());
        for ((_, access), lock) in footprint.resources.iter().zip(resource_locks) {
            guards.push(acquire_lock(lock, *access).await);
        }
        Ok(EffectLease {
            _resources: guards,
            _dispatch_slot: Some(dispatch_slot),
            _global: Some(global),
        })
    }

    /// Reserve exclusive ownership of the whole execution surface for guarded
    /// compensation. Existing effects settle before Undo; new dispatch waits.
    pub(crate) async fn acquire_exclusive(&self) -> EffectLease {
        EffectLease {
            _resources: Vec::new(),
            _dispatch_slot: None,
            _global: Some(LockGuard::Exclusive {
                _guard: self.global.clone().write_owned().await,
            }),
        }
    }
}

async fn acquire_lock(lock: Arc<RwLock<()>>, access: Access) -> LockGuard {
    match access {
        Access::Shared => LockGuard::Shared {
            _guard: lock.read_owned().await,
        },
        Access::Exclusive => LockGuard::Exclusive {
            _guard: lock.write_owned().await,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use super::*;
    use crate::contracts::{ActionIntent, Effect, FileIdentity, POLICY_VERSION};
    use crate::domain::{ExpectedOutcome, Provenance};
    use uuid::Uuid;

    fn prepared(action: Action, target: PreparedTarget, effects: &[Effect]) -> PreparedAction {
        PreparedAction {
            intent: ActionIntent {
                schema_version: 2,
                tool_version: 2,
                proposal: crate::domain::ActionProposal {
                    id: Uuid::new_v4(),
                    task_id: Uuid::new_v4(),
                    target_resource: "fixture".into(),
                    expected_outcome: ExpectedOutcome::UserAnswered,
                    action,
                    provenance: Provenance::user(),
                    metadata: BTreeMap::new(),
                },
                input_refs: Default::default(),
            },
            action_digest: "a".repeat(64),
            policy_version: POLICY_VERSION,
            target,
            effects: effects.iter().copied().collect(),
            preview: "fixture".into(),
            prepared_at: chrono::Utc::now(),
        }
    }

    fn write(path: &Path, exists: bool) -> PreparedAction {
        let path = path.to_path_buf();
        prepared(
            Action::WriteFile {
                path: path.clone(),
                content: "body".into(),
                overwrite: exists,
            },
            PreparedTarget::File {
                path,
                before: exists.then(|| FileIdentity {
                    key: "file-id".into(),
                    size: 4,
                    modified: "fixture".into(),
                    directory: false,
                }),
            },
            &[if exists {
                Effect::Modify
            } else {
                Effect::Create
            }],
        )
    }

    fn listing(path: &Path) -> PreparedAction {
        let path = path.to_path_buf();
        prepared(
            Action::ListDirectory {
                path: path.clone(),
                page_size: 16,
                cursor: None,
            },
            PreparedTarget::File {
                path,
                before: Some(FileIdentity {
                    key: "directory-id".into(),
                    size: 0,
                    modified: "fixture".into(),
                    directory: true,
                }),
            },
            &[Effect::Read],
        )
    }

    #[tokio::test]
    async fn disjoint_file_creations_overlap_but_directory_listing_waits() {
        let directory = PathBuf::from(if cfg!(windows) { r"C:\work" } else { "/work" });
        let first = write(&directory.join("a.txt"), false);
        let second = write(&directory.join("b.txt"), false);
        let list = listing(&directory);
        let arbiter = EffectArbiter::default();

        let first_lease = arbiter.acquire_action(&first).await.unwrap();
        let second_lease =
            tokio::time::timeout(Duration::from_millis(100), arbiter.acquire_action(&second))
                .await
                .expect("disjoint file creation should not serialize")
                .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), arbiter.acquire_action(&list),)
                .await
                .is_err()
        );
        drop((first_lease, second_lease));
        tokio::time::timeout(Duration::from_millis(100), arbiter.acquire_action(&list))
            .await
            .expect("listing should proceed after child creation settles")
            .unwrap();
    }

    #[tokio::test]
    async fn same_file_read_waits_for_write_and_disjoint_read_does_not() {
        let directory = PathBuf::from(if cfg!(windows) { r"C:\work" } else { "/work" });
        let existing = directory.join("shared.txt");
        let writer = write(&existing, true);
        let reader = prepared(
            Action::ReadFile {
                path: existing.clone(),
                max_bytes: 128,
            },
            PreparedTarget::File {
                path: existing,
                before: Some(FileIdentity {
                    key: "file-id".into(),
                    size: 4,
                    modified: "fixture".into(),
                    directory: false,
                }),
            },
            &[Effect::Read],
        );
        let other_reader = prepared(
            Action::ReadFile {
                path: directory.join("other.txt"),
                max_bytes: 128,
            },
            PreparedTarget::File {
                path: directory.join("other.txt"),
                before: None,
            },
            &[Effect::Read],
        );
        let arbiter = EffectArbiter::default();
        let read_one = arbiter.acquire_action(&reader).await.unwrap();
        let read_two =
            tokio::time::timeout(Duration::from_millis(100), arbiter.acquire_action(&reader))
                .await
                .expect("same-file reads should share ownership")
                .unwrap();
        drop((read_one, read_two));

        let writer_lease = arbiter.acquire_action(&writer).await.unwrap();
        let other_lease = tokio::time::timeout(
            Duration::from_millis(100),
            arbiter.acquire_action(&other_reader),
        )
        .await
        .expect("disjoint resource read should overlap")
        .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), arbiter.acquire_action(&reader),)
                .await
                .is_err()
        );
        drop((writer_lease, other_lease));
        arbiter.acquire_action(&reader).await.unwrap();
    }

    #[tokio::test]
    async fn exclusive_compensation_fences_all_action_dispatch() {
        let path = PathBuf::from(if cfg!(windows) {
            r"C:\work\a.txt"
        } else {
            "/work/a.txt"
        });
        let action = write(&path, true);
        let arbiter = EffectArbiter::default();
        let exclusive = arbiter.acquire_exclusive().await;
        assert!(
            tokio::time::timeout(Duration::from_millis(20), arbiter.acquire_action(&action),)
                .await
                .is_err()
        );
        drop(exclusive);
        tokio::time::timeout(Duration::from_millis(100), arbiter.acquire_action(&action))
            .await
            .expect("dispatch should resume after exclusive compensation")
            .unwrap();
    }

    #[tokio::test]
    async fn user_question_does_not_hold_dispatch_or_compensation_fences() {
        let arbiter = EffectArbiter::default();
        let question = prepared(
            Action::AskUser {
                question: "Which document?".into(),
            },
            PreparedTarget::User,
            &[],
        );
        let _question_lease = arbiter.acquire_action(&question).await.unwrap();

        tokio::time::timeout(Duration::from_millis(50), arbiter.acquire_exclusive())
            .await
            .expect("a user response wait must not block compensation");

        let mut effect_leases = Vec::new();
        for index in 0..MAX_ACTIVE_DISPATCHES {
            let path = PathBuf::from(format!("/tmp/sage-question-fence-{index}.txt"));
            let read = prepared(
                Action::ReadFile {
                    path: path.clone(),
                    max_bytes: 16,
                },
                PreparedTarget::File { path, before: None },
                &[Effect::Read],
            );
            effect_leases.push(arbiter.acquire_action(&read).await.unwrap());
        }
        assert_eq!(effect_leases.len(), MAX_ACTIVE_DISPATCHES);
    }
}
