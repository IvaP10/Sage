//! Shared first-party compute admission for inference and isolated jobs.
//!
//! Reservations are an admission estimate, not an OS-enforced memory or CPU
//! quota. Callers must keep their own bounded allocations and report measured
//! usage separately from this cooperative budget.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Qwen35Error, Qwen35Result};

/// Per-job ceiling for the initial 16 GB desktop tier, leaving system headroom.
pub const LOCAL_COMPUTE_BUDGET: u64 = 8 * 1024 * 1024 * 1024;
const OS_HEADROOM: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryEnvelope {
    pub weights: u64,
    pub state: u64,
    pub activations: u64,
    pub runtime: u64,
}

impl MemoryEnvelope {
    pub fn total(&self) -> Qwen35Result<u64> {
        [self.weights, self.state, self.activations, self.runtime]
            .into_iter()
            .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
            .filter(|total| *total > 0)
            .ok_or_else(|| Qwen35Error::Model("Invalid memory envelope".into()))
    }
}

#[derive(Default)]
struct Reservations {
    bytes: HashMap<Uuid, u64>,
    heavy: Option<Uuid>,
}

/// Cooperative process-wide memory admission. It may reject work early, but
/// does not promise that the OS will enforce a reservation after admission.
#[derive(Default, Clone)]
pub struct ResourceGovernor {
    state: Arc<Mutex<Reservations>>,
}

pub struct Reservation {
    id: Uuid,
    governor: ResourceGovernor,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if let Ok(mut state) = self.governor.state.lock() {
            state.bytes.remove(&self.id);
            if state.heavy == Some(self.id) {
                state.heavy = None;
            }
        }
    }
}

impl ResourceGovernor {
    /// Admit a measured estimate if the shared budget has room. At most one
    /// reservation may claim the heavy inference lane at a time.
    pub fn reserve(&self, bytes: u64, heavy: bool, available: u64) -> Qwen35Result<Reservation> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Qwen35Error::Model("Resource manager unavailable".into()))?;
        let used = state
            .bytes
            .values()
            .try_fold(0u64, |sum, reserved| sum.checked_add(*reserved))
            .ok_or_else(|| Qwen35Error::Model("Resource reservation total overflow".into()))?;
        let limit =
            LOCAL_COMPUTE_BUDGET.min(available.saturating_sub(OS_HEADROOM).saturating_add(used));
        if bytes == 0 || bytes > limit.saturating_sub(used) || (heavy && state.heavy.is_some()) {
            return Err(Qwen35Error::ResourceUnavailable(
                "Local resources are busy or insufficient; retry after other work finishes. No cloud fallback was used.".into(),
            ));
        }
        let id = Uuid::new_v4();
        state.bytes.insert(id, bytes);
        if heavy {
            state.heavy = Some(id);
        }
        Ok(Reservation {
            id,
            governor: self.clone(),
        })
    }

    pub fn reserve_current(&self, bytes: u64, heavy: bool) -> Qwen35Result<Reservation> {
        let available = sage_host_memory::available_bytes().map_err(|error| {
            Qwen35Error::ResourceUnavailable(format!(
                "Cannot inspect available host memory: {error}"
            ))
        })?;
        self.reserve(bytes, heavy, available)
    }
}

#[cfg(test)]
mod tests {
    use super::{LOCAL_COMPUTE_BUDGET, MemoryEnvelope, ResourceGovernor};

    #[test]
    fn concurrent_compute_roles_share_one_admission_budget() {
        let governor = ResourceGovernor::default();
        let gib = 1024 * 1024 * 1024;
        let model = governor.reserve(6 * gib, true, 16 * gib).unwrap();
        assert!(governor.reserve(4 * gib, false, 16 * gib).is_err());
        assert!(governor.reserve(gib, true, 16 * gib).is_err());
        assert!(governor.reserve(gib, false, gib).is_err());
        drop(model);
        assert!(governor.reserve(4 * gib, true, 16 * gib).is_ok());
        assert_eq!(LOCAL_COMPUTE_BUDGET, 8 * gib);
    }

    #[test]
    fn memory_envelopes_reject_zero_and_overflow() {
        assert!(
            MemoryEnvelope {
                weights: 0,
                state: 0,
                activations: 0,
                runtime: 0,
            }
            .total()
            .is_err()
        );
        assert!(
            MemoryEnvelope {
                weights: u64::MAX,
                state: 1,
                activations: 0,
                runtime: 0,
            }
            .total()
            .is_err()
        );
    }
}
