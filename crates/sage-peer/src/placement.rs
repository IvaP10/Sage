//! Conservative placement for bounded, explicitly delegated peer work.
//!
//! This module estimates completion from caller-supplied lower-bound capacity
//! and link measurements. It creates no lease, permit, or dispatch authority;
//! the live lease book must still reserve and authorize every remote send.

use super::{
    DelegatedJobKind, DisclosureScope, LeaseScope, PeerError, PeerId, PeerLease, PeerResult,
};
use sage_host_memory::{CpuTimeSnapshot, available_bytes, cpu_time_snapshot};
use std::{
    io,
    time::{Duration, Instant},
};

pub const MAX_PLACEMENT_PAYLOAD_BYTES: u64 = 256 * 1024 * 1024;
const MILLION: u128 = 1_000_000;
const PERMILLE: u128 = 1_000;
const MAX_WORK_UNITS: u64 = 1_000_000_000_000;
const MAX_CANDIDATES: usize = 64;
pub const CAPACITY_SAMPLE_WINDOW: usize = 32;

/// Fresh local pressure evidence for one placement decision. CPU load excludes
/// this process only; other Sage processes remain in the residual and therefore
/// conservatively count as background work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalHostPressure {
    pub available_memory_bytes: u64,
    pub external_cpu_load_permille: u16,
    /// Monotonic sample completion time. It must be checked before use so an
    /// old pressure reading cannot silently steer a later placement decision.
    pub sampled_at: Instant,
}

/// Pairs native cumulative counters to produce local pressure observations.
/// Construct and sample this on a blocking resource lane, not the control lane.
#[derive(Debug, Default)]
pub struct LocalHostPressureSampler {
    previous_cpu: Option<CpuTimeSnapshot>,
}

impl LocalHostPressureSampler {
    /// The first call establishes a baseline and returns `None`. Later calls
    /// return a sample only when the native counters form a valid interval.
    /// Invalid intervals replace the baseline and fail closed for one sample.
    pub fn sample(&mut self) -> io::Result<Option<LocalHostPressure>> {
        let current_cpu = cpu_time_snapshot()?;
        let previous_cpu = self.previous_cpu.replace(current_cpu);
        let Some(previous_cpu) = previous_cpu else {
            return Ok(None);
        };
        let Some(external_cpu_load_permille) =
            current_cpu.busy_permille_excluding_process_since(previous_cpu)
        else {
            return Ok(None);
        };
        let available_memory_bytes = available_bytes()?;
        Ok(Some(LocalHostPressure {
            available_memory_bytes,
            external_cpu_load_permille,
            sampled_at: Instant::now(),
        }))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CapacitySample {
    lower_work_units_per_second: u64,
    observed_at_millis: u64,
    energy_microwatt_hours_per_unit_upper_bound: Option<u64>,
}

/// Fixed-memory rolling measurements for one trusted local job/resource pair.
/// Callers should record only completed work whose result passed its verifier.
#[derive(Debug, Clone)]
pub struct CapacitySampler {
    job: DelegatedJobKind,
    resource_id: [u8; 32],
    samples: [CapacitySample; CAPACITY_SAMPLE_WINDOW],
    next_slot: usize,
    sample_count: usize,
    last_observed_at_millis: Option<u64>,
}

impl CapacitySampler {
    pub fn new(job: DelegatedJobKind, resource_id: [u8; 32]) -> Self {
        Self {
            job,
            resource_id,
            samples: [CapacitySample::default(); CAPACITY_SAMPLE_WINDOW],
            next_slot: 0,
            sample_count: 0,
            last_observed_at_millis: None,
        }
    }

    /// Record one completed measurement using monotonic elapsed time. The
    /// sampler retains no job payload, only rate, time and optional energy.
    pub fn record_completion_measurement(
        &mut self,
        work_units: u64,
        elapsed_micros: u64,
        energy_microwatt_hours_per_unit_upper_bound: Option<u64>,
        observed_at_millis: u64,
    ) -> PeerResult<()> {
        if work_units == 0
            || work_units > MAX_WORK_UNITS
            || elapsed_micros == 0
            || self
                .last_observed_at_millis
                .is_some_and(|last| observed_at_millis < last)
        {
            return Err(PeerError::InvalidComputeJob);
        }
        let rate = u64::try_from(
            u128::from(work_units)
                .checked_mul(MILLION)
                .ok_or(PeerError::InvalidComputeJob)?
                / u128::from(elapsed_micros),
        )
        .map_err(|_| PeerError::InvalidComputeJob)?;
        if rate == 0 {
            return Err(PeerError::InvalidComputeJob);
        }

        self.samples[self.next_slot] = CapacitySample {
            lower_work_units_per_second: rate,
            observed_at_millis,
            energy_microwatt_hours_per_unit_upper_bound,
        };
        self.next_slot = (self.next_slot + 1) % CAPACITY_SAMPLE_WINDOW;
        self.sample_count = (self.sample_count + 1).min(CAPACITY_SAMPLE_WINDOW);
        self.last_observed_at_millis = Some(observed_at_millis);
        Ok(())
    }

    /// Return a conservative estimate using the lowest observed throughput,
    /// oldest sample age and highest complete energy estimate in the window.
    pub fn observation(&self, now_millis: u64) -> Option<CapacityObservation> {
        if self.sample_count == 0 {
            return None;
        }
        let samples = &self.samples[..self.sample_count];
        let oldest = samples
            .iter()
            .map(|sample| sample.observed_at_millis)
            .min()?;
        let latest = self.last_observed_at_millis?;
        if now_millis < latest {
            return None;
        }
        let mut energy = Some(0_u64);
        for sample in samples {
            energy = match (energy, sample.energy_microwatt_hours_per_unit_upper_bound) {
                (Some(current), Some(observed)) => Some(current.max(observed)),
                _ => None,
            };
        }
        Some(CapacityObservation {
            job: self.job,
            resource_id: self.resource_id,
            lower_work_units_per_second: samples
                .iter()
                .map(|sample| sample.lower_work_units_per_second)
                .min()?,
            energy_microwatt_hours_per_unit_upper_bound: energy,
            sample_count: u32::try_from(self.sample_count).ok()?,
            age_millis: now_millis.checked_sub(oldest)?,
        })
    }
}

/// A task-specific capacity sample. Rates are conservative lower bounds from
/// measured completed work, not advertised peak hardware specifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityObservation {
    pub job: DelegatedJobKind,
    pub resource_id: [u8; 32],
    pub lower_work_units_per_second: u64,
    /// Upper-bound measured battery cost per work unit in micro-watt-hours.
    /// Battery execution requires a value; external-power execution may omit it.
    pub energy_microwatt_hours_per_unit_upper_bound: Option<u64>,
    pub sample_count: u32,
    /// Age measured by the scheduling owner when it received this sample.
    pub age_millis: u64,
}

/// Conservative link measurements from the same paired peer session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkObservation {
    pub round_trip_p95_micros: u64,
    pub upload_bytes_per_second_lower: u64,
    pub download_bytes_per_second_lower: u64,
    pub sample_count: u32,
    pub age_millis: u64,
}

/// Resource snapshot for one local or peer execution route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeResourceSnapshot {
    pub capacity: CapacityObservation,
    pub available_memory_bytes: u64,
    /// Background load outside Sage's own queued work, in permille.
    pub external_load_permille: u16,
    /// Observed throttle pressure, in permille; 1000 means no effective rate.
    pub thermal_throttle_permille: u16,
    pub battery_permille: u16,
    pub remaining_battery_microwatt_hours: u64,
    pub reserved_battery_microwatt_hours: u64,
    pub on_external_power: bool,
    pub active_jobs: u16,
    pub maximum_concurrent_jobs: u16,
    /// Sage-owned work already ahead of this candidate on the same resource.
    pub queued_work_units: u64,
    pub link: Option<LinkObservation>,
}

impl ComputeResourceSnapshot {
    /// Apply a newly sampled local OS observation only while it remains fresh.
    /// Existing memory caps are preserved; the measured residual CPU load
    /// replaces older load estimates. A stale or future-dated sample leaves the
    /// snapshot unchanged and returns `false`.
    pub fn apply_local_host_pressure(
        &mut self,
        pressure: LocalHostPressure,
        now: Instant,
        maximum_age: Duration,
    ) -> bool {
        let Some(age) = now.checked_duration_since(pressure.sampled_at) else {
            return false;
        };
        if age > maximum_age {
            return false;
        }
        self.available_memory_bytes = self
            .available_memory_bytes
            .min(pressure.available_memory_bytes);
        self.external_load_permille = pressure.external_cpu_load_permille;
        true
    }
}

/// Exact work contract used for estimating both local and leased peer routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementWorkload {
    pub job: DelegatedJobKind,
    pub resource_id: [u8; 32],
    pub disclosure: DisclosureScope,
    pub work_units: u64,
    pub input_bytes: u64,
    pub maximum_result_bytes: u64,
    pub required_memory_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementPolicy {
    pub maximum_capacity_age_millis: u64,
    pub maximum_link_age_millis: u64,
    pub minimum_capacity_samples: u32,
    pub minimum_link_samples: u32,
    /// Required improvement before sending data away; 100 means 10%.
    pub minimum_remote_improvement_permille: u16,
    pub minimum_remote_battery_permille: u16,
    pub maximum_round_trip_micros: u64,
}

impl Default for PlacementPolicy {
    fn default() -> Self {
        Self {
            maximum_capacity_age_millis: 30_000,
            maximum_link_age_millis: 15_000,
            minimum_capacity_samples: 3,
            minimum_link_samples: 5,
            minimum_remote_improvement_permille: 100,
            minimum_remote_battery_permille: 200,
            maximum_round_trip_micros: 2_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementReason {
    LocalFastest,
    RemoteConservativelyFaster,
    LocalCapacityInsufficient,
}

/// A scheduling suggestion only. Remote variants do not authorize dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementDecision {
    Local {
        estimated_completion_micros: Option<u64>,
        reason: PlacementReason,
    },
    Peer {
        peer: PeerId,
        estimated_completion_micros: u64,
        improvement_permille: Option<u16>,
        reason: PlacementReason,
    },
    NoFeasibleRoute,
}

#[derive(Clone, Copy)]
pub struct LeasedPeerRoute<'a> {
    pub peer: PeerId,
    pub lease: &'a PeerLease,
    pub resources: ComputeResourceSnapshot,
}

/// Select the fastest route whose conservative estimate is supported by fresh
/// measurements and whose peer lease exactly covers this bounded workload.
/// Candidate iteration is capped and deterministic; ties stay local, then use
/// the peer's stable identifier ordering.
pub fn choose_compute_placement(
    workload: PlacementWorkload,
    local_owner: PeerId,
    local: ComputeResourceSnapshot,
    peers: &[LeasedPeerRoute<'_>],
    now_unix_seconds: u64,
    policy: PlacementPolicy,
) -> PeerResult<PlacementDecision> {
    validate_workload(workload)?;
    if peers.len() > MAX_CANDIDATES || !valid_policy(policy) {
        return Err(PeerError::InvalidComputeJob);
    }

    let local_estimate = estimate_completion(workload, local, policy, false);
    let local_fits_memory = local.available_memory_bytes >= workload.required_memory_bytes;
    let local_is_feasible = local_fits_memory
        && local.active_jobs < local.maximum_concurrent_jobs
        && local_estimate.is_some();

    let mut best_peer: Option<(PeerId, u64, u16)> = None;
    for candidate in peers {
        if !lease_covers(
            workload,
            local_owner,
            candidate.peer,
            candidate.lease,
            now_unix_seconds,
        ) || candidate.resources.available_memory_bytes < workload.required_memory_bytes
            || candidate.resources.active_jobs >= candidate.resources.maximum_concurrent_jobs
            || candidate.resources.active_jobs >= candidate.lease.quotas().maximum_concurrent_jobs
            || (!candidate.resources.on_external_power
                && candidate.resources.battery_permille < policy.minimum_remote_battery_permille)
        {
            continue;
        }
        let Some(estimated) = estimate_completion(workload, candidate.resources, policy, true)
        else {
            continue;
        };

        let improvement = match local_estimate {
            Some(local_micros) if local_is_feasible => {
                let remote_scaled = u128::from(estimated)
                    .checked_mul(PERMILLE + u128::from(policy.minimum_remote_improvement_permille))
                    .ok_or(PeerError::InvalidComputeJob)?;
                let local_scaled = u128::from(local_micros)
                    .checked_mul(PERMILLE)
                    .ok_or(PeerError::InvalidComputeJob)?;
                if remote_scaled > local_scaled {
                    continue;
                }
                Some(
                    u16::try_from(
                        u128::from(local_micros.saturating_sub(estimated)) * PERMILLE
                            / u128::from(local_micros.max(1)),
                    )
                    .unwrap_or(u16::MAX),
                )
            }
            _ if !local_is_feasible => None,
            _ => continue,
        };

        if best_peer.is_none_or(|(best_id, best_time, _)| {
            estimated < best_time || (estimated == best_time && candidate.peer < best_id)
        }) {
            best_peer = Some((candidate.peer, estimated, improvement.unwrap_or(0)));
        }
    }

    if let Some((peer, estimated, improvement)) = best_peer {
        let local_is_feasible = local_fits_memory
            && local.active_jobs < local.maximum_concurrent_jobs
            && local_estimate.is_some();
        return Ok(PlacementDecision::Peer {
            peer,
            estimated_completion_micros: estimated,
            improvement_permille: local_is_feasible.then_some(improvement),
            reason: if local_is_feasible {
                PlacementReason::RemoteConservativelyFaster
            } else {
                PlacementReason::LocalCapacityInsufficient
            },
        });
    }

    if local_is_feasible {
        Ok(PlacementDecision::Local {
            estimated_completion_micros: local_estimate,
            reason: PlacementReason::LocalFastest,
        })
    } else {
        Ok(PlacementDecision::NoFeasibleRoute)
    }
}

fn validate_workload(workload: PlacementWorkload) -> PeerResult<()> {
    if workload.work_units == 0
        || workload.work_units > MAX_WORK_UNITS
        || workload.input_bytes == 0
        || workload.input_bytes > MAX_PLACEMENT_PAYLOAD_BYTES
        || workload.maximum_result_bytes == 0
        || workload.maximum_result_bytes > MAX_PLACEMENT_PAYLOAD_BYTES
        || workload.required_memory_bytes == 0
    {
        return Err(PeerError::InvalidComputeJob);
    }
    Ok(())
}

fn valid_policy(policy: PlacementPolicy) -> bool {
    policy.maximum_capacity_age_millis > 0
        && policy.maximum_link_age_millis > 0
        && policy.minimum_capacity_samples > 0
        && policy.minimum_link_samples > 0
        && policy.minimum_remote_improvement_permille < 1_000
        && policy.minimum_remote_battery_permille <= 1_000
        && policy.maximum_round_trip_micros > 0
}

fn lease_covers(
    workload: PlacementWorkload,
    expected_owner: PeerId,
    expected_peer: PeerId,
    lease: &PeerLease,
    now_unix_seconds: u64,
) -> bool {
    if lease.owner() != expected_owner
        || lease.peer() != expected_peer
        || lease.issued_at_unix_seconds() > now_unix_seconds
        || lease.expires_at_unix_seconds() <= now_unix_seconds
        || lease.quotas().maximum_concurrent_jobs == 0
        || workload.input_bytes > lease.quotas().maximum_input_bytes_per_job
        || workload.maximum_result_bytes > lease.quotas().maximum_output_bytes_per_job
    {
        return false;
    }
    lease.scopes().any(|scope: &LeaseScope| {
        scope.job == workload.job
            && scope.resource_id == workload.resource_id
            && scope.disclosure == workload.disclosure
    })
}

fn estimate_completion(
    workload: PlacementWorkload,
    resources: ComputeResourceSnapshot,
    policy: PlacementPolicy,
    remote: bool,
) -> Option<u64> {
    let capacity = resources.capacity;
    if capacity.job != workload.job
        || capacity.resource_id != workload.resource_id
        || capacity.lower_work_units_per_second == 0
        || capacity.sample_count < policy.minimum_capacity_samples
        || capacity.age_millis > policy.maximum_capacity_age_millis
        || resources.external_load_permille > 1_000
        || resources.thermal_throttle_permille > 1_000
        || resources.battery_permille > 1_000
        || resources.maximum_concurrent_jobs == 0
    {
        return None;
    }
    if !resources.on_external_power {
        let energy_per_unit = capacity.energy_microwatt_hours_per_unit_upper_bound?;
        let required_energy =
            u128::from(energy_per_unit).checked_mul(u128::from(workload.work_units))?;
        let available_energy = resources
            .remaining_battery_microwatt_hours
            .saturating_sub(resources.reserved_battery_microwatt_hours);
        if required_energy > u128::from(available_energy) {
            return None;
        }
    }

    let usable_rate = u128::from(capacity.lower_work_units_per_second)
        .checked_mul(PERMILLE - u128::from(resources.external_load_permille))?
        .checked_mul(PERMILLE - u128::from(resources.thermal_throttle_permille))?
        / (PERMILLE * PERMILLE);
    if usable_rate == 0 {
        return None;
    }
    let queued_and_requested =
        u128::from(resources.queued_work_units).checked_add(u128::from(workload.work_units))?;
    let compute_micros = ceil_div(queued_and_requested.checked_mul(MILLION)?, usable_rate)?;

    let transfer_micros = if remote {
        let link = resources.link?;
        if link.sample_count < policy.minimum_link_samples
            || link.age_millis > policy.maximum_link_age_millis
            || link.round_trip_p95_micros == 0
            || link.round_trip_p95_micros > policy.maximum_round_trip_micros
            || link.upload_bytes_per_second_lower == 0
            || link.download_bytes_per_second_lower == 0
        {
            return None;
        }
        let upload = ceil_div(
            u128::from(workload.input_bytes).checked_mul(MILLION)?,
            u128::from(link.upload_bytes_per_second_lower),
        )?;
        let download = ceil_div(
            u128::from(workload.maximum_result_bytes).checked_mul(MILLION)?,
            u128::from(link.download_bytes_per_second_lower),
        )?;
        u128::from(link.round_trip_p95_micros)
            .checked_add(upload)?
            .checked_add(download)?
    } else {
        0
    };
    u64::try_from(compute_micros.checked_add(transfer_micros)?).ok()
}

fn ceil_div(numerator: u128, denominator: u128) -> Option<u128> {
    if denominator == 0 {
        return None;
    }
    numerator
        .checked_add(denominator - 1)
        .map(|rounded| rounded / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeviceIdentity, PeerLeaseQuotas, byte_histogram_v1_resource_id};

    const NOW: u64 = 1_800_000_000;

    fn workload() -> PlacementWorkload {
        PlacementWorkload {
            job: DelegatedJobKind::BatchAnalysis,
            resource_id: byte_histogram_v1_resource_id(),
            disclosure: DisclosureScope::ExplicitJobInputAndResult,
            work_units: 10_000,
            input_bytes: 1_000_000,
            maximum_result_bytes: 4_096,
            required_memory_bytes: 1_000_000,
        }
    }

    fn resources(rate: u64, memory: u64) -> ComputeResourceSnapshot {
        ComputeResourceSnapshot {
            capacity: CapacityObservation {
                job: DelegatedJobKind::BatchAnalysis,
                resource_id: byte_histogram_v1_resource_id(),
                lower_work_units_per_second: rate,
                energy_microwatt_hours_per_unit_upper_bound: Some(1),
                sample_count: 9,
                age_millis: 1_000,
            },
            available_memory_bytes: memory,
            external_load_permille: 0,
            thermal_throttle_permille: 0,
            battery_permille: 1_000,
            remaining_battery_microwatt_hours: 1_000_000,
            reserved_battery_microwatt_hours: 0,
            on_external_power: true,
            active_jobs: 0,
            maximum_concurrent_jobs: 4,
            queued_work_units: 0,
            link: None,
        }
    }

    fn lease_for(
        owner: &DeviceIdentity,
        peer: PeerId,
        workload: PlacementWorkload,
        now: u64,
        lifetime: u64,
    ) -> PeerLease {
        PeerLease::issue(
            owner,
            peer,
            now,
            lifetime,
            [LeaseScope {
                job: workload.job,
                resource_id: workload.resource_id,
                disclosure: workload.disclosure,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 2,
                maximum_jobs: 10,
                maximum_input_bytes_per_job: MAX_PLACEMENT_PAYLOAD_BYTES,
                maximum_output_bytes_per_job: MAX_PLACEMENT_PAYLOAD_BYTES,
                maximum_total_input_bytes: MAX_PLACEMENT_PAYLOAD_BYTES,
                maximum_total_output_bytes: MAX_PLACEMENT_PAYLOAD_BYTES,
            },
        )
        .unwrap()
    }

    fn peer_resources(rate: u64) -> ComputeResourceSnapshot {
        ComputeResourceSnapshot {
            link: Some(LinkObservation {
                round_trip_p95_micros: 1_000,
                upload_bytes_per_second_lower: 10_000_000_000,
                download_bytes_per_second_lower: 10_000_000_000,
                sample_count: 8,
                age_millis: 500,
            }),
            ..resources(rate, 4_000_000)
        }
    }

    #[test]
    fn fresh_local_host_pressure_constrains_memory_and_changes_route_estimate() {
        let work = workload();
        let owner = DeviceIdentity::from_seed([7; 32]);
        let peer_identity = DeviceIdentity::from_seed([19; 32]);
        let peer = peer_identity.peer_id();
        let lease = lease_for(&owner, peer, work, NOW, 60);
        let candidate = LeasedPeerRoute {
            peer,
            lease: &lease,
            resources: peer_resources(110_000),
        };
        let mut local = resources(100_000, 4_000_000);

        assert!(matches!(
            choose_compute_placement(
                work,
                owner.peer_id(),
                local,
                &[candidate],
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap(),
            PlacementDecision::Local { .. }
        ));

        let pressure_now = Instant::now();
        assert!(local.apply_local_host_pressure(
            LocalHostPressure {
                available_memory_bytes: 3_000_000,
                external_cpu_load_permille: 200,
                sampled_at: pressure_now,
            },
            pressure_now,
            Duration::from_secs(1),
        ));
        assert_eq!(local.available_memory_bytes, 3_000_000);
        assert_eq!(local.external_load_permille, 200);
        assert!(matches!(
            choose_compute_placement(
                work,
                owner.peer_id(),
                local,
                &[candidate],
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap(),
            PlacementDecision::Peer {
                peer: selected,
                reason: PlacementReason::RemoteConservativelyFaster,
                ..
            } if selected == peer
        ));
    }

    #[test]
    fn stale_or_future_local_pressure_cannot_change_a_route_snapshot() {
        let mut local = resources(100_000, 4_000_000);
        let original = local;
        let now = Instant::now();

        assert!(!local.apply_local_host_pressure(
            LocalHostPressure {
                available_memory_bytes: 1,
                external_cpu_load_permille: 1_000,
                sampled_at: now - Duration::from_secs(2),
            },
            now,
            Duration::from_secs(1),
        ));
        assert_eq!(local, original);

        assert!(!local.apply_local_host_pressure(
            LocalHostPressure {
                available_memory_bytes: 1,
                external_cpu_load_permille: 1_000,
                sampled_at: now + Duration::from_secs(1),
            },
            now,
            Duration::from_secs(1),
        ));
        assert_eq!(local, original);
    }

    #[test]
    fn capacity_sampler_uses_worst_recent_rate_and_complete_energy_upper_bound() {
        let mut sampler = CapacitySampler::new(
            DelegatedJobKind::BatchAnalysis,
            byte_histogram_v1_resource_id(),
        );
        sampler
            .record_completion_measurement(100, 1_000_000, Some(2), 100)
            .unwrap();
        sampler
            .record_completion_measurement(100, 500_000, Some(5), 200)
            .unwrap();
        sampler
            .record_completion_measurement(50, 1_000_000, Some(4), 300)
            .unwrap();

        let observation = sampler.observation(800).unwrap();
        assert_eq!(observation.lower_work_units_per_second, 50);
        assert_eq!(observation.sample_count, 3);
        assert_eq!(
            observation.energy_microwatt_hours_per_unit_upper_bound,
            Some(5)
        );
        assert_eq!(observation.age_millis, 700);
        assert_eq!(observation.job, DelegatedJobKind::BatchAnalysis);
    }

    #[test]
    fn capacity_sampler_bounds_history_and_rejects_invalid_or_unordered_samples() {
        let mut sampler = CapacitySampler::new(
            DelegatedJobKind::BatchAnalysis,
            byte_histogram_v1_resource_id(),
        );
        assert_eq!(sampler.observation(100), None);
        assert_eq!(
            sampler.record_completion_measurement(0, 1_000_000, Some(1), 100),
            Err(PeerError::InvalidComputeJob)
        );
        assert_eq!(
            sampler.record_completion_measurement(1, 0, Some(1), 100),
            Err(PeerError::InvalidComputeJob)
        );
        assert_eq!(
            sampler.record_completion_measurement(1, 2_000_000, Some(1), 100),
            Err(PeerError::InvalidComputeJob)
        );
        assert_eq!(
            sampler.record_completion_measurement(1, 1_000_000, Some(1), 100),
            Ok(())
        );
        assert_eq!(
            sampler.record_completion_measurement(1, 1_000_000, Some(1), 99),
            Err(PeerError::InvalidComputeJob)
        );

        for index in 0..CAPACITY_SAMPLE_WINDOW {
            sampler
                .record_completion_measurement(100, 1_000_000, Some(2), 101 + index as u64)
                .unwrap();
        }
        let observation = sampler.observation(200).unwrap();
        assert_eq!(observation.sample_count as usize, CAPACITY_SAMPLE_WINDOW);
        assert_eq!(observation.lower_work_units_per_second, 100);
        assert_eq!(observation.age_millis, 99);
    }

    #[test]
    fn offloads_only_when_fresh_conservative_peer_estimate_beats_local_margin() {
        let work = workload();
        let owner = DeviceIdentity::from_seed([7; 32]);
        let peer_identity = DeviceIdentity::from_seed([19; 32]);
        let peer = peer_identity.peer_id();
        let lease = lease_for(&owner, peer, work, NOW, 60);
        let candidate = LeasedPeerRoute {
            peer,
            lease: &lease,
            resources: peer_resources(100_000),
        };
        assert!(matches!(
            choose_compute_placement(
                work,
                owner.peer_id(),
                resources(10_000, 4_000_000),
                &[candidate],
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap(),
            PlacementDecision::Peer {
                peer: observed,
                reason: PlacementReason::RemoteConservativelyFaster,
                improvement_permille: Some(_),
                ..
            } if observed == peer
        ));

        let slow_candidate = LeasedPeerRoute {
            resources: peer_resources(11_000),
            ..candidate
        };
        assert!(matches!(
            choose_compute_placement(
                work,
                owner.peer_id(),
                resources(10_000, 4_000_000),
                &[slow_candidate],
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap(),
            PlacementDecision::Local {
                reason: PlacementReason::LocalFastest,
                ..
            }
        ));
    }

    #[test]
    fn stale_busy_low_battery_or_underscoped_peers_are_never_selected() {
        let work = workload();
        let owner = DeviceIdentity::from_seed([7; 32]);
        let peer_identity = DeviceIdentity::from_seed([19; 32]);
        let peer = peer_identity.peer_id();
        let lease = lease_for(&owner, peer, work, NOW, 60);

        let mut stale = peer_resources(1_000_000);
        stale.capacity.age_millis = 31_000;
        let mut busy = peer_resources(1_000_000);
        busy.active_jobs = busy.maximum_concurrent_jobs;
        let mut low_battery = peer_resources(1_000_000);
        low_battery.on_external_power = false;
        low_battery.battery_permille = 100;
        let peer_routes = [stale, busy, low_battery].map(|resources| LeasedPeerRoute {
            peer,
            lease: &lease,
            resources,
        });
        assert!(matches!(
            choose_compute_placement(
                work,
                owner.peer_id(),
                resources(10_000, 4_000_000),
                &peer_routes,
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap(),
            PlacementDecision::Local { .. }
        ));

        let expired = lease_for(&owner, peer, work, NOW - 60, 30);
        assert_eq!(
            choose_compute_placement(
                work,
                owner.peer_id(),
                resources(10_000, 4_000_000),
                &[LeasedPeerRoute {
                    peer,
                    lease: &expired,
                    resources: peer_resources(1_000_000),
                }],
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap(),
            PlacementDecision::Local {
                estimated_completion_micros: Some(1_000_000),
                reason: PlacementReason::LocalFastest,
            }
        );

        let foreign_owner = DeviceIdentity::from_seed([23; 32]);
        let foreign_lease = lease_for(&foreign_owner, peer, work, NOW, 60);
        assert!(matches!(
            choose_compute_placement(
                work,
                owner.peer_id(),
                resources(10_000, 4_000_000),
                &[LeasedPeerRoute {
                    peer,
                    lease: &foreign_lease,
                    resources: peer_resources(1_000_000),
                }],
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap(),
            PlacementDecision::Local { .. }
        ));
    }

    #[test]
    fn battery_energy_budget_is_enforced_for_local_and_peer_routes() {
        let work = workload();
        let owner = DeviceIdentity::from_seed([7; 32]);
        let peer_identity = DeviceIdentity::from_seed([19; 32]);
        let peer = peer_identity.peer_id();
        let lease = lease_for(&owner, peer, work, NOW, 60);

        let mut local = resources(10_000, 4_000_000);
        local.on_external_power = false;
        local.remaining_battery_microwatt_hours = work.work_units - 1;
        local.capacity.energy_microwatt_hours_per_unit_upper_bound = Some(1);
        let mut peer_on_battery = peer_resources(1_000_000);
        peer_on_battery.on_external_power = false;
        peer_on_battery.remaining_battery_microwatt_hours = work.work_units - 1;
        peer_on_battery
            .capacity
            .energy_microwatt_hours_per_unit_upper_bound = Some(1);

        assert_eq!(
            choose_compute_placement(
                work,
                owner.peer_id(),
                local,
                &[LeasedPeerRoute {
                    peer,
                    lease: &lease,
                    resources: peer_on_battery,
                }],
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap(),
            PlacementDecision::NoFeasibleRoute
        );

        let mut peer_on_power = peer_resources(1_000_000);
        peer_on_power.remaining_battery_microwatt_hours = 0;
        assert!(matches!(
            choose_compute_placement(
                work,
                owner.peer_id(),
                local,
                &[LeasedPeerRoute {
                    peer,
                    lease: &lease,
                    resources: peer_on_power,
                }],
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap(),
            PlacementDecision::Peer {
                reason: PlacementReason::LocalCapacityInsufficient,
                ..
            }
        ));
    }

    #[test]
    fn memory_pressure_and_thermal_load_are_applied_before_route_comparison() {
        let work = workload();
        let owner = DeviceIdentity::from_seed([7; 32]);
        let peer_identity = DeviceIdentity::from_seed([19; 32]);
        let peer = peer_identity.peer_id();
        let lease = lease_for(&owner, peer, work, NOW, 60);
        let local = ComputeResourceSnapshot {
            thermal_throttle_permille: 500,
            external_load_permille: 300,
            ..resources(100_000, work.required_memory_bytes - 1)
        };
        assert!(matches!(
            choose_compute_placement(
                work,
                owner.peer_id(),
                local,
                &[LeasedPeerRoute {
                    peer,
                    lease: &lease,
                    resources: peer_resources(100_000),
                }],
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap(),
            PlacementDecision::Peer {
                reason: PlacementReason::LocalCapacityInsufficient,
                improvement_permille: None,
                ..
            }
        ));
    }

    #[test]
    fn invalid_payloads_and_unbounded_candidate_lists_fail_closed() {
        let mut work = workload();
        work.input_bytes = MAX_PLACEMENT_PAYLOAD_BYTES + 1;
        assert_eq!(
            choose_compute_placement(
                work,
                DeviceIdentity::from_seed([7; 32]).peer_id(),
                resources(10_000, 4_000_000),
                &[],
                NOW,
                PlacementPolicy::default(),
            )
            .unwrap_err(),
            PeerError::InvalidComputeJob
        );

        let policy = PlacementPolicy {
            minimum_remote_improvement_permille: 1_000,
            ..PlacementPolicy::default()
        };
        assert_eq!(
            choose_compute_placement(
                workload(),
                DeviceIdentity::from_seed([7; 32]).peer_id(),
                resources(10_000, 4_000_000),
                &[],
                NOW,
                policy,
            )
            .unwrap_err(),
            PeerError::InvalidComputeJob
        );
    }
}
