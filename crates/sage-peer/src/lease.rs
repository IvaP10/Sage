use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use ed25519_dalek::{Signature, Signer, VerifyingKey};

use super::{
    DeviceIdentity, MAX_PEER_LEASE_SECONDS, PeerError, PeerId, PeerResult, PublicPeerIdentity,
};

const MAX_LEASE_SCOPES: usize = 32;
const MAX_CONCURRENT_JOBS: u16 = 32;
const MAX_JOBS_PER_LEASE: u32 = 10_000;
const LEASE_GRANT_MAGIC: &[u8; 4] = b"SGL1";
const LEASE_GRANT_VERSION: u16 = 1;
const LEASE_GRANT_DOMAIN: &[u8] = b"sage:peer:lease-grant:v1\0";
const LEASE_GRANT_FIXED_BYTES: usize = 4 + 2 + 16 + 32 + 32 + 8 + 8 + 1 + 2 + 4 + 4 * 8 + 64;

/// The initial worker is planned to accept only these partitionable, headless
/// job families. They describe no shell, filesystem, account, camera, or
/// microphone RPC and do not imply that a worker is connected yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum DelegatedJobKind {
    FrameRender = 1,
    BatchAnalysis = 2,
}

impl TryFrom<u8> for DelegatedJobKind {
    type Error = PeerError;

    fn try_from(value: u8) -> PeerResult<Self> {
        match value {
            1 => Ok(Self::FrameRender),
            2 => Ok(Self::BatchAnalysis),
            _ => Err(PeerError::InvalidInput),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LeaseId([u8; 16]);

impl LeaseId {
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    pub(super) fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
}

/// Disclosure is limited to explicit, bounded job inputs and that job's
/// output. Other data sources require a separately designed scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DisclosureScope {
    ExplicitJobInputAndResult,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LeaseScope {
    pub job: DelegatedJobKind,
    pub resource_id: [u8; 32],
    pub disclosure: DisclosureScope,
}

impl fmt::Debug for LeaseScope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LeaseScope")
            .field("job", &self.job)
            .field("resource_id", &"[redacted]")
            .field("disclosure", &self.disclosure)
            .finish()
    }
}

/// Quotas enforced by the Sage lease gate: dispatch count, concurrency, and
/// byte budgets. CPU/GPU percentages and hard memory limits are intentionally
/// absent until peer workers can enforce them with native OS controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerLeaseQuotas {
    pub maximum_concurrent_jobs: u16,
    pub maximum_jobs: u32,
    pub maximum_input_bytes_per_job: u64,
    pub maximum_output_bytes_per_job: u64,
    pub maximum_total_input_bytes: u64,
    pub maximum_total_output_bytes: u64,
}

/// Owner-signed, bounded authority statement that a paired recipient can
/// verify before considering work. It carries no credential and grants only
/// the explicitly listed operation/resource/disclosure combinations.
#[derive(Clone, PartialEq, Eq)]
pub struct PeerLeaseGrant {
    id: LeaseId,
    owner: PeerId,
    peer: PeerId,
    issued_at_unix_seconds: u64,
    expires_at_unix_seconds: u64,
    scopes: Vec<LeaseScope>,
    quotas: PeerLeaseQuotas,
    signature: [u8; 64],
}

impl PeerLeaseGrant {
    pub fn id(&self) -> LeaseId {
        self.id
    }

    pub fn owner(&self) -> PeerId {
        self.owner
    }

    pub fn peer(&self) -> PeerId {
        self.peer
    }

    pub fn issued_at_unix_seconds(&self) -> u64 {
        self.issued_at_unix_seconds
    }

    pub fn expires_at_unix_seconds(&self) -> u64 {
        self.expires_at_unix_seconds
    }

    pub fn scopes(&self) -> impl Iterator<Item = &LeaseScope> {
        self.scopes.iter()
    }

    pub fn quotas(&self) -> PeerLeaseQuotas {
        self.quotas
    }

    /// Verify the owner's signature, expected paired identity, exact recipient,
    /// bounded lease lifetime and current expiry before accepting the grant.
    pub fn verify(
        &self,
        expected_owner: &PublicPeerIdentity,
        recipient: PeerId,
        now_unix_seconds: u64,
    ) -> PeerResult<()> {
        validate_grant_fields(
            self.owner,
            self.peer,
            self.issued_at_unix_seconds,
            self.expires_at_unix_seconds,
            &self.scopes,
            self.quotas,
        )?;
        if expected_owner.peer_id() != self.owner || self.peer != recipient {
            return Err(PeerError::LeaseDenied);
        }
        if now_unix_seconds < self.issued_at_unix_seconds
            || now_unix_seconds >= self.expires_at_unix_seconds
        {
            return Err(PeerError::InvalidLease);
        }
        let key = VerifyingKey::from_bytes(expected_owner.verifying_key_bytes())
            .map_err(|_| PeerError::AuthenticationFailed)?;
        let signature = Signature::from_bytes(&self.signature);
        key.verify_strict(&self.signing_bytes(), &signature)
            .map_err(|_| PeerError::AuthenticationFailed)
    }

    /// Encode the canonical bounded binary form for the authenticated peer
    /// channel. The channel protects confidentiality and replay ordering;
    /// this signature independently binds the grant to its issuing identity.
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = self.body();
        bytes.extend_from_slice(&self.signature);
        bytes
    }

    pub fn decode(bytes: &[u8]) -> PeerResult<Self> {
        if bytes.len() < LEASE_GRANT_FIXED_BYTES + 34
            || bytes.len() > LEASE_GRANT_FIXED_BYTES + MAX_LEASE_SCOPES * 34
            || !bytes.starts_with(LEASE_GRANT_MAGIC)
        {
            return Err(PeerError::InvalidLease);
        }
        let mut offset = LEASE_GRANT_MAGIC.len();
        if lease_read_u16(bytes, &mut offset)? != LEASE_GRANT_VERSION {
            return Err(PeerError::InvalidLease);
        }
        let id = LeaseId(lease_read_array::<16>(bytes, &mut offset)?);
        let owner = PeerId(lease_read_array::<32>(bytes, &mut offset)?);
        let peer = PeerId(lease_read_array::<32>(bytes, &mut offset)?);
        let issued_at_unix_seconds = lease_read_u64(bytes, &mut offset)?;
        let expires_at_unix_seconds = lease_read_u64(bytes, &mut offset)?;
        let scope_count = usize::from(lease_read_u8(bytes, &mut offset)?);
        if scope_count == 0 || scope_count > MAX_LEASE_SCOPES {
            return Err(PeerError::InvalidLease);
        }
        let mut scopes = Vec::with_capacity(scope_count);
        for _ in 0..scope_count {
            let job = DelegatedJobKind::try_from(lease_read_u8(bytes, &mut offset)?)?;
            let resource_id = lease_read_array::<32>(bytes, &mut offset)?;
            let disclosure = match lease_read_u8(bytes, &mut offset)? {
                1 => DisclosureScope::ExplicitJobInputAndResult,
                _ => return Err(PeerError::InvalidLease),
            };
            scopes.push(LeaseScope {
                job,
                resource_id,
                disclosure,
            });
        }
        if scopes.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PeerError::InvalidLease);
        }
        let quotas = PeerLeaseQuotas {
            maximum_concurrent_jobs: lease_read_u16(bytes, &mut offset)?,
            maximum_jobs: lease_read_u32(bytes, &mut offset)?,
            maximum_input_bytes_per_job: lease_read_u64(bytes, &mut offset)?,
            maximum_output_bytes_per_job: lease_read_u64(bytes, &mut offset)?,
            maximum_total_input_bytes: lease_read_u64(bytes, &mut offset)?,
            maximum_total_output_bytes: lease_read_u64(bytes, &mut offset)?,
        };
        let signature = lease_read_array::<64>(bytes, &mut offset)?;
        if offset != bytes.len() {
            return Err(PeerError::InvalidLease);
        }
        validate_grant_fields(
            owner,
            peer,
            issued_at_unix_seconds,
            expires_at_unix_seconds,
            &scopes,
            quotas,
        )?;
        Ok(Self {
            id,
            owner,
            peer,
            issued_at_unix_seconds,
            expires_at_unix_seconds,
            scopes,
            quotas,
            signature,
        })
    }

    fn body(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(LEASE_GRANT_FIXED_BYTES - 64 + self.scopes.len() * 34);
        body.extend_from_slice(LEASE_GRANT_MAGIC);
        body.extend_from_slice(&LEASE_GRANT_VERSION.to_be_bytes());
        body.extend_from_slice(self.id.as_bytes());
        body.extend_from_slice(self.owner.as_bytes());
        body.extend_from_slice(self.peer.as_bytes());
        body.extend_from_slice(&self.issued_at_unix_seconds.to_be_bytes());
        body.extend_from_slice(&self.expires_at_unix_seconds.to_be_bytes());
        body.push(self.scopes.len() as u8);
        for scope in &self.scopes {
            body.push(scope.job as u8);
            body.extend_from_slice(&scope.resource_id);
            body.push(match scope.disclosure {
                DisclosureScope::ExplicitJobInputAndResult => 1,
            });
        }
        body.extend_from_slice(&self.quotas.maximum_concurrent_jobs.to_be_bytes());
        body.extend_from_slice(&self.quotas.maximum_jobs.to_be_bytes());
        body.extend_from_slice(&self.quotas.maximum_input_bytes_per_job.to_be_bytes());
        body.extend_from_slice(&self.quotas.maximum_output_bytes_per_job.to_be_bytes());
        body.extend_from_slice(&self.quotas.maximum_total_input_bytes.to_be_bytes());
        body.extend_from_slice(&self.quotas.maximum_total_output_bytes.to_be_bytes());
        body
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let body = self.body();
        let mut bytes = Vec::with_capacity(LEASE_GRANT_DOMAIN.len() + body.len());
        bytes.extend_from_slice(LEASE_GRANT_DOMAIN);
        bytes.extend_from_slice(&body);
        bytes
    }
}

impl fmt::Debug for PeerLeaseGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PeerLeaseGrant")
            .field("id", &"[redacted]")
            .field("owner", &self.owner)
            .field("peer", &self.peer)
            .field("issued_at_unix_seconds", &self.issued_at_unix_seconds)
            .field("expires_at_unix_seconds", &self.expires_at_unix_seconds)
            .field("scopes", &self.scopes)
            .field("quotas", &self.quotas)
            .field("signature", &"[redacted]")
            .finish()
    }
}

#[derive(Debug)]
pub struct PeerLease {
    id: LeaseId,
    owner: PeerId,
    peer: PeerId,
    issued_at_unix_seconds: u64,
    expires_at_unix_seconds: u64,
    scopes: BTreeSet<LeaseScope>,
    quotas: PeerLeaseQuotas,
    grant: PeerLeaseGrant,
}

impl PeerLease {
    pub fn issue(
        owner_identity: &DeviceIdentity,
        peer: PeerId,
        now_unix_seconds: u64,
        lifetime_seconds: u64,
        scopes: impl IntoIterator<Item = LeaseScope>,
        quotas: PeerLeaseQuotas,
    ) -> PeerResult<Self> {
        let owner = owner_identity.peer_id();
        if !(1..=MAX_PEER_LEASE_SECONDS).contains(&lifetime_seconds) {
            return Err(PeerError::InvalidLease);
        }
        let expires_at_unix_seconds = now_unix_seconds
            .checked_add(lifetime_seconds)
            .ok_or(PeerError::InvalidLease)?;
        let scopes = scopes.into_iter().collect::<BTreeSet<_>>();
        if scopes.is_empty() || scopes.len() > MAX_LEASE_SCOPES {
            return Err(PeerError::InvalidLease);
        }
        let grant_scopes = scopes.iter().copied().collect::<Vec<_>>();
        validate_grant_fields(
            owner,
            peer,
            now_unix_seconds,
            expires_at_unix_seconds,
            &grant_scopes,
            quotas,
        )?;
        let mut id = [0_u8; 16];
        getrandom::fill(&mut id).map_err(|_| PeerError::RandomnessUnavailable)?;
        let mut grant = PeerLeaseGrant {
            id: LeaseId(id),
            owner,
            peer,
            issued_at_unix_seconds: now_unix_seconds,
            expires_at_unix_seconds,
            scopes: grant_scopes,
            quotas,
            signature: [0; 64],
        };
        grant.signature = owner_identity
            .signing_key
            .sign(&grant.signing_bytes())
            .to_bytes();
        Ok(Self {
            id: grant.id,
            owner,
            peer,
            issued_at_unix_seconds: now_unix_seconds,
            expires_at_unix_seconds,
            scopes,
            quotas,
            grant,
        })
    }

    pub fn id(&self) -> LeaseId {
        self.id
    }

    pub fn owner(&self) -> PeerId {
        self.owner
    }

    pub fn peer(&self) -> PeerId {
        self.peer
    }

    pub fn issued_at_unix_seconds(&self) -> u64 {
        self.issued_at_unix_seconds
    }

    pub fn expires_at_unix_seconds(&self) -> u64 {
        self.expires_at_unix_seconds
    }

    pub fn scopes(&self) -> impl Iterator<Item = &LeaseScope> {
        self.scopes.iter()
    }

    pub fn quotas(&self) -> PeerLeaseQuotas {
        self.quotas
    }

    pub fn grant(&self) -> PeerLeaseGrant {
        self.grant.clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct LeaseRequest {
    pub job_id: [u8; 16],
    pub partition_index: u16,
    pub job: DelegatedJobKind,
    pub resource_id: [u8; 32],
    pub disclosure: DisclosureScope,
    pub input_bytes: u64,
    pub maximum_output_bytes: u64,
}

impl fmt::Debug for LeaseRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LeaseRequest")
            .field("job_id", &"[redacted]")
            .field("partition_index", &self.partition_index)
            .field("job", &self.job)
            .field("resource_id", &"[redacted]")
            .field("disclosure", &self.disclosure)
            .field("input_bytes", &self.input_bytes)
            .field("maximum_output_bytes", &self.maximum_output_bytes)
            .finish()
    }
}

#[derive(Debug)]
pub struct LeasePermit {
    lease_id: LeaseId,
    reservation_id: u64,
    peer: PeerId,
}

impl LeasePermit {
    pub fn lease_id(&self) -> LeaseId {
        self.lease_id
    }
}

#[derive(PartialEq, Eq)]
pub struct PermittedPeerJob {
    pub(super) lease_id: LeaseId,
    pub(super) peer: PeerId,
    pub(super) job_id: [u8; 16],
    pub(super) partition_index: u16,
    pub(super) job: DelegatedJobKind,
    pub(super) resource_id: [u8; 32],
    pub(super) disclosure: DisclosureScope,
    pub(super) maximum_input_bytes: u64,
    pub(super) maximum_output_bytes: u64,
}

impl PermittedPeerJob {
    pub fn lease_id(&self) -> LeaseId {
        self.lease_id
    }

    pub fn peer(&self) -> PeerId {
        self.peer
    }

    pub fn job_id(&self) -> &[u8; 16] {
        &self.job_id
    }

    pub fn partition_index(&self) -> u16 {
        self.partition_index
    }

    pub fn job(&self) -> DelegatedJobKind {
        self.job
    }

    pub fn resource_id(&self) -> &[u8; 32] {
        &self.resource_id
    }

    pub fn disclosure(&self) -> DisclosureScope {
        self.disclosure
    }

    pub fn maximum_input_bytes(&self) -> u64 {
        self.maximum_input_bytes
    }

    pub fn maximum_output_bytes(&self) -> u64 {
        self.maximum_output_bytes
    }
}

impl fmt::Debug for PermittedPeerJob {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PermittedPeerJob")
            .field("lease_id", &self.lease_id)
            .field("peer", &self.peer)
            .field("job_id", &"[redacted]")
            .field("partition_index", &self.partition_index)
            .field("job", &self.job)
            .field("resource_id", &"[redacted]")
            .field("disclosure", &self.disclosure)
            .field("maximum_input_bytes", &self.maximum_input_bytes)
            .field("maximum_output_bytes", &self.maximum_output_bytes)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseSettlement {
    pub dispatched: bool,
    pub output_bytes: u64,
}

#[derive(Debug)]
struct PendingReservation {
    request: LeaseRequest,
    dispatched: bool,
}

#[derive(Debug)]
struct ActiveLease {
    lease: PeerLease,
    revoked: bool,
    next_reservation_id: u64,
    jobs_started: u32,
    input_bytes_reserved: u64,
    input_bytes_used: u64,
    output_bytes_reserved: u64,
    output_bytes_used: u64,
    pending: BTreeMap<u64, PendingReservation>,
    job_identities: BTreeSet<([u8; 16], u16)>,
}

/// Single-writer lease authority owned by the paired-device broker. Every job
/// must reserve here before serialization, then pass `authorize_dispatch`
/// immediately before sending. In-flight jobs remain settleable after expiry
/// or revocation, while new dispatches fail closed.
#[derive(Debug)]
pub struct PeerLeaseBook {
    local_owner: PeerId,
    leases: BTreeMap<LeaseId, ActiveLease>,
}

impl PeerLeaseBook {
    pub fn new(local_owner: PeerId) -> Self {
        Self {
            local_owner,
            leases: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, lease: PeerLease) -> PeerResult<LeaseId> {
        if lease.owner != self.local_owner {
            return Err(PeerError::LeaseDenied);
        }
        let id = lease.id;
        if self.leases.contains_key(&id) {
            return Err(PeerError::InvalidLease);
        }
        self.leases.insert(
            id,
            ActiveLease {
                lease,
                revoked: false,
                next_reservation_id: 1,
                jobs_started: 0,
                input_bytes_reserved: 0,
                input_bytes_used: 0,
                output_bytes_reserved: 0,
                output_bytes_used: 0,
                pending: BTreeMap::new(),
                job_identities: BTreeSet::new(),
            },
        );
        Ok(id)
    }

    pub fn reserve(
        &mut self,
        lease_id: LeaseId,
        authenticated_peer: PeerId,
        request: LeaseRequest,
        now_unix_seconds: u64,
    ) -> PeerResult<LeasePermit> {
        let active = self
            .leases
            .get_mut(&lease_id)
            .ok_or(PeerError::InvalidLease)?;
        if active.revoked {
            return Err(PeerError::LeaseRevoked);
        }
        if active.lease.peer != authenticated_peer {
            return Err(PeerError::LeaseDenied);
        }
        let job_identity = (request.job_id, request.partition_index);
        if active.job_identities.contains(&job_identity) {
            return Err(PeerError::DuplicateJob);
        }
        if now_unix_seconds < active.lease.issued_at_unix_seconds
            || now_unix_seconds >= active.lease.expires_at_unix_seconds
        {
            return Err(PeerError::InvalidLease);
        }
        let scope = LeaseScope {
            job: request.job,
            resource_id: request.resource_id,
            disclosure: request.disclosure,
        };
        let quotas = active.lease.quotas;
        if !active.lease.scopes.contains(&scope) {
            return Err(PeerError::LeaseDenied);
        }
        if request.input_bytes == 0
            || request.input_bytes > quotas.maximum_input_bytes_per_job
            || request.maximum_output_bytes == 0
            || request.maximum_output_bytes > quotas.maximum_output_bytes_per_job
        {
            return Err(PeerError::QuotaExceeded);
        }
        if active.pending.len() >= usize::from(quotas.maximum_concurrent_jobs)
            || active.jobs_started.saturating_add(
                active
                    .pending
                    .values()
                    .filter(|reservation| !reservation.dispatched)
                    .count() as u32,
            ) >= quotas.maximum_jobs
            || active
                .input_bytes_used
                .saturating_add(active.input_bytes_reserved)
                .saturating_add(request.input_bytes)
                > quotas.maximum_total_input_bytes
            || active
                .output_bytes_used
                .saturating_add(active.output_bytes_reserved)
                .saturating_add(request.maximum_output_bytes)
                > quotas.maximum_total_output_bytes
        {
            return Err(PeerError::QuotaExceeded);
        }
        let reservation_id = active.next_reservation_id;
        active.next_reservation_id = reservation_id
            .checked_add(1)
            .ok_or(PeerError::InvalidPermit)?;
        active.input_bytes_reserved += request.input_bytes;
        active.output_bytes_reserved += request.maximum_output_bytes;
        active.job_identities.insert(job_identity);
        active.pending.insert(
            reservation_id,
            PendingReservation {
                request,
                dispatched: false,
            },
        );
        Ok(LeasePermit {
            lease_id,
            reservation_id,
            peer: authenticated_peer,
        })
    }

    /// Consume the reservation's one dispatch opportunity after rechecking
    /// peer, expiry, revocation, scope and quota state.
    pub fn authorize_dispatch(
        &mut self,
        permit: &LeasePermit,
        now_unix_seconds: u64,
    ) -> PeerResult<PermittedPeerJob> {
        let active = self
            .leases
            .get_mut(&permit.lease_id)
            .ok_or(PeerError::InvalidLease)?;
        if active.revoked {
            return Err(PeerError::LeaseRevoked);
        }
        if permit.peer != active.lease.peer
            || now_unix_seconds < active.lease.issued_at_unix_seconds
            || now_unix_seconds >= active.lease.expires_at_unix_seconds
        {
            return Err(PeerError::InvalidLease);
        }
        let reservation = active
            .pending
            .get_mut(&permit.reservation_id)
            .ok_or(PeerError::InvalidPermit)?;
        if reservation.dispatched {
            return Err(PeerError::InvalidPermit);
        }
        reservation.dispatched = true;
        active.input_bytes_reserved = active
            .input_bytes_reserved
            .saturating_sub(reservation.request.input_bytes);
        active.input_bytes_used = active
            .input_bytes_used
            .saturating_add(reservation.request.input_bytes);
        active.jobs_started = active.jobs_started.saturating_add(1);
        Ok(PermittedPeerJob {
            lease_id: permit.lease_id,
            peer: permit.peer,
            job_id: reservation.request.job_id,
            partition_index: reservation.request.partition_index,
            job: reservation.request.job,
            resource_id: reservation.request.resource_id,
            disclosure: reservation.request.disclosure,
            maximum_input_bytes: reservation.request.input_bytes,
            maximum_output_bytes: reservation.request.maximum_output_bytes,
        })
    }

    /// Settle a dispatched job even if its lease expired or was revoked while
    /// it ran. An output overrun consumes the reservation, revokes the lease,
    /// and returns an error so the oversized result is discarded by the peer.
    pub fn settle(
        &mut self,
        permit: LeasePermit,
        output_bytes: u64,
    ) -> PeerResult<LeaseSettlement> {
        let active = self
            .leases
            .get_mut(&permit.lease_id)
            .ok_or(PeerError::InvalidLease)?;
        let reservation = active
            .pending
            .remove(&permit.reservation_id)
            .ok_or(PeerError::InvalidPermit)?;
        active.output_bytes_reserved = active
            .output_bytes_reserved
            .saturating_sub(reservation.request.maximum_output_bytes);
        if !reservation.dispatched {
            active.input_bytes_reserved = active
                .input_bytes_reserved
                .saturating_sub(reservation.request.input_bytes);
            active.job_identities.remove(&(
                reservation.request.job_id,
                reservation.request.partition_index,
            ));
            return Err(PeerError::InvalidPermit);
        }
        if output_bytes > reservation.request.maximum_output_bytes {
            active.revoked = true;
            return Err(PeerError::QuotaExceeded);
        }
        active.output_bytes_used = active.output_bytes_used.saturating_add(output_bytes);
        Ok(LeaseSettlement {
            dispatched: true,
            output_bytes,
        })
    }

    /// Release a reservation that was never dispatched. Revocation or expiry
    /// does not prevent cleanup of already-reserved protocol state.
    pub fn cancel(&mut self, permit: LeasePermit) -> PeerResult<()> {
        let active = self
            .leases
            .get_mut(&permit.lease_id)
            .ok_or(PeerError::InvalidLease)?;
        let reservation = active
            .pending
            .remove(&permit.reservation_id)
            .ok_or(PeerError::InvalidPermit)?;
        if reservation.dispatched {
            active.pending.insert(permit.reservation_id, reservation);
            return Err(PeerError::InvalidPermit);
        }
        active.input_bytes_reserved = active
            .input_bytes_reserved
            .saturating_sub(reservation.request.input_bytes);
        active.output_bytes_reserved = active
            .output_bytes_reserved
            .saturating_sub(reservation.request.maximum_output_bytes);
        active.job_identities.remove(&(
            reservation.request.job_id,
            reservation.request.partition_index,
        ));
        Ok(())
    }

    /// Fence future dispatches immediately. Existing dispatched jobs retain a
    /// settlement path so their final effects and receipts remain observable.
    pub fn revoke(&mut self, lease_id: LeaseId) -> PeerResult<()> {
        let active = self
            .leases
            .get_mut(&lease_id)
            .ok_or(PeerError::InvalidLease)?;
        active.revoked = true;
        Ok(())
    }

    pub fn pending_jobs(&self, lease_id: LeaseId) -> PeerResult<usize> {
        self.leases
            .get(&lease_id)
            .map(|active| active.pending.len())
            .ok_or(PeerError::InvalidLease)
    }
}

fn validate_grant_fields(
    owner: PeerId,
    peer: PeerId,
    issued_at_unix_seconds: u64,
    expires_at_unix_seconds: u64,
    scopes: &[LeaseScope],
    quotas: PeerLeaseQuotas,
) -> PeerResult<()> {
    let lifetime = expires_at_unix_seconds
        .checked_sub(issued_at_unix_seconds)
        .ok_or(PeerError::InvalidLease)?;
    if owner == peer
        || !(1..=MAX_PEER_LEASE_SECONDS).contains(&lifetime)
        || scopes.is_empty()
        || scopes.len() > MAX_LEASE_SCOPES
        || scopes.windows(2).any(|pair| pair[0] >= pair[1])
        || quotas.maximum_concurrent_jobs == 0
        || quotas.maximum_concurrent_jobs > MAX_CONCURRENT_JOBS
        || quotas.maximum_jobs == 0
        || quotas.maximum_jobs > MAX_JOBS_PER_LEASE
        || quotas.maximum_input_bytes_per_job == 0
        || quotas.maximum_input_bytes_per_job > super::MAX_PEER_COMPUTE_BYTES
        || quotas.maximum_output_bytes_per_job == 0
        || quotas.maximum_output_bytes_per_job > super::MAX_PEER_COMPUTE_BYTES
        || quotas.maximum_total_input_bytes < quotas.maximum_input_bytes_per_job
        || quotas.maximum_total_input_bytes > super::MAX_PEER_COMPUTE_BYTES.saturating_mul(64)
        || quotas.maximum_total_output_bytes < quotas.maximum_output_bytes_per_job
        || quotas.maximum_total_output_bytes > super::MAX_PEER_COMPUTE_BYTES.saturating_mul(64)
    {
        return Err(PeerError::InvalidLease);
    }
    Ok(())
}

fn lease_read_array<const N: usize>(bytes: &[u8], offset: &mut usize) -> PeerResult<[u8; N]> {
    let end = offset.checked_add(N).ok_or(PeerError::InvalidLease)?;
    let slice = bytes.get(*offset..end).ok_or(PeerError::InvalidLease)?;
    let mut value = [0_u8; N];
    value.copy_from_slice(slice);
    *offset = end;
    Ok(value)
}

fn lease_read_u8(bytes: &[u8], offset: &mut usize) -> PeerResult<u8> {
    Ok(lease_read_array::<1>(bytes, offset)?[0])
}

fn lease_read_u16(bytes: &[u8], offset: &mut usize) -> PeerResult<u16> {
    Ok(u16::from_be_bytes(lease_read_array(bytes, offset)?))
}

fn lease_read_u32(bytes: &[u8], offset: &mut usize) -> PeerResult<u32> {
    Ok(u32::from_be_bytes(lease_read_array(bytes, offset)?))
}

fn lease_read_u64(bytes: &[u8], offset: &mut usize) -> PeerResult<u64> {
    Ok(u64::from_be_bytes(lease_read_array(bytes, offset)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceIdentity;

    fn lease() -> (PeerLeaseBook, LeaseId, PeerId, PeerId, [u8; 32]) {
        let owner_identity = DeviceIdentity::from_seed([4; 32]);
        let owner = owner_identity.peer_id();
        let peer = DeviceIdentity::from_seed([5; 32]).peer_id();
        let resource_id = [0x91; 32];
        let scope = LeaseScope {
            job: DelegatedJobKind::FrameRender,
            resource_id,
            disclosure: DisclosureScope::ExplicitJobInputAndResult,
        };
        let lease = PeerLease::issue(
            &owner_identity,
            peer,
            2_000,
            60,
            [scope],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 2,
                maximum_jobs: 3,
                maximum_input_bytes_per_job: 4_096,
                maximum_output_bytes_per_job: 8_192,
                maximum_total_input_bytes: 8_192,
                maximum_total_output_bytes: 16_384,
            },
        )
        .unwrap();
        let mut book = PeerLeaseBook::new(owner);
        let id = book.insert(lease).unwrap();
        (book, id, owner, peer, resource_id)
    }

    fn request(resource_id: [u8; 32]) -> LeaseRequest {
        use std::sync::atomic::{AtomicU32, Ordering};

        static NEXT_TEST_JOB_ID: AtomicU32 = AtomicU32::new(1);
        let mut job_id = [0_u8; 16];
        job_id[..4].copy_from_slice(
            &NEXT_TEST_JOB_ID
                .fetch_add(1, Ordering::Relaxed)
                .to_be_bytes(),
        );
        LeaseRequest {
            job_id,
            partition_index: 0,
            job: DelegatedJobKind::FrameRender,
            resource_id,
            disclosure: DisclosureScope::ExplicitJobInputAndResult,
            input_bytes: 1_024,
            maximum_output_bytes: 2_048,
        }
    }

    #[test]
    fn signed_lease_grants_round_trip_and_bind_owner_peer_scope_quota_and_expiry() {
        let owner = DeviceIdentity::from_seed([31; 32]);
        let peer = DeviceIdentity::from_seed([32; 32]);
        let resource_id = [0x6a; 32];
        let lease = PeerLease::issue(
            &owner,
            peer.peer_id(),
            10_000,
            300,
            [LeaseScope {
                job: DelegatedJobKind::BatchAnalysis,
                resource_id,
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 3,
                maximum_jobs: 100,
                maximum_input_bytes_per_job: 16_384,
                maximum_output_bytes_per_job: 8_192,
                maximum_total_input_bytes: 64 * 1024,
                maximum_total_output_bytes: 32 * 1024,
            },
        )
        .unwrap();
        let grant = lease.grant();
        let bytes = grant.encode();
        assert!(bytes.len() <= 2_048);
        let decoded = PeerLeaseGrant::decode(&bytes).unwrap();
        assert_eq!(decoded, grant);
        decoded
            .verify(&owner.public_identity(), peer.peer_id(), 10_001)
            .unwrap();
        assert_eq!(decoded.owner(), owner.peer_id());
        assert_eq!(decoded.peer(), peer.peer_id());
        assert_eq!(decoded.scopes().next().unwrap().resource_id, resource_id);

        assert_eq!(
            decoded.verify(&owner.public_identity(), owner.peer_id(), 10_001),
            Err(PeerError::LeaseDenied)
        );
        let impostor = DeviceIdentity::from_seed([33; 32]);
        assert_eq!(
            decoded.verify(&impostor.public_identity(), peer.peer_id(), 10_001),
            Err(PeerError::LeaseDenied)
        );
        assert_eq!(
            decoded.verify(&owner.public_identity(), peer.peer_id(), 10_300),
            Err(PeerError::InvalidLease)
        );

        let mut tampered = bytes.clone();
        tampered[104] ^= 1;
        let tampered = PeerLeaseGrant::decode(&tampered).unwrap();
        assert_eq!(
            tampered.verify(&owner.public_identity(), peer.peer_id(), 10_001),
            Err(PeerError::AuthenticationFailed)
        );
        assert_eq!(
            PeerLeaseGrant::decode(&bytes[..bytes.len() - 1]).err(),
            Some(PeerError::InvalidLease)
        );
    }

    #[test]
    fn lease_rechecks_peer_scope_and_expiry_at_dispatch() {
        let (mut book, id, _owner, peer, resource_id) = lease();
        let permit = book.reserve(id, peer, request(resource_id), 2_010).unwrap();
        assert_eq!(
            book.authorize_dispatch(&permit, 2_060),
            Err(PeerError::InvalidLease)
        );
        book.cancel(permit).unwrap();

        let mut wrong_resource = request(resource_id);
        wrong_resource.resource_id[0] ^= 1;
        assert_eq!(
            book.reserve(id, peer, wrong_resource, 2_010).err(),
            Some(PeerError::LeaseDenied)
        );
        assert_eq!(
            book.reserve(id, _owner, request(resource_id), 2_010).err(),
            Some(PeerError::LeaseDenied)
        );
    }

    #[test]
    fn lease_enforces_concurrency_job_and_byte_quotas() {
        let (mut book, id, _, peer, resource_id) = lease();
        let first = book.reserve(id, peer, request(resource_id), 2_001).unwrap();
        let second = book.reserve(id, peer, request(resource_id), 2_001).unwrap();
        assert_eq!(
            book.reserve(id, peer, request(resource_id), 2_001).err(),
            Some(PeerError::QuotaExceeded)
        );
        let first_job = book.authorize_dispatch(&first, 2_002).unwrap();
        assert_eq!(first_job.maximum_input_bytes, 1_024);
        assert_eq!(
            book.authorize_dispatch(&first, 2_002),
            Err(PeerError::InvalidPermit)
        );
        book.settle(first, 1_000).unwrap();
        book.cancel(second).unwrap();

        let mut too_large = request(resource_id);
        too_large.maximum_output_bytes = 8_193;
        assert_eq!(
            book.reserve(id, peer, too_large, 2_003).err(),
            Some(PeerError::QuotaExceeded)
        );
    }

    #[test]
    fn duplicate_job_partition_is_fenced_after_dispatch_but_cancel_releases_identity() {
        let (mut book, id, _, peer, resource_id) = lease();
        let retryable = request(resource_id);
        let reserved = book.reserve(id, peer, retryable, 2_001).unwrap();
        assert_eq!(
            book.reserve(id, peer, retryable, 2_001).err(),
            Some(PeerError::DuplicateJob)
        );
        book.cancel(reserved).unwrap();

        let retry = book.reserve(id, peer, retryable, 2_002).unwrap();
        book.authorize_dispatch(&retry, 2_003).unwrap();
        book.settle(retry, 1_024).unwrap();
        assert_eq!(
            book.reserve(id, peer, retryable, 2_004).err(),
            Some(PeerError::DuplicateJob)
        );
    }

    #[test]
    fn revocation_blocks_dispatch_but_dispatched_work_can_settle() {
        let (mut book, id, _, peer, resource_id) = lease();
        let before_dispatch = book.reserve(id, peer, request(resource_id), 2_001).unwrap();
        book.revoke(id).unwrap();
        assert_eq!(
            book.authorize_dispatch(&before_dispatch, 2_002),
            Err(PeerError::LeaseRevoked)
        );
        book.cancel(before_dispatch).unwrap();
        assert_eq!(
            book.reserve(id, peer, request(resource_id), 2_003).err(),
            Some(PeerError::LeaseRevoked)
        );

        let (mut book, id, _, peer, resource_id) = lease();
        let dispatched = book.reserve(id, peer, request(resource_id), 2_001).unwrap();
        book.authorize_dispatch(&dispatched, 2_002).unwrap();
        book.revoke(id).unwrap();
        assert_eq!(book.settle(dispatched, 1_024).unwrap().output_bytes, 1_024);
    }

    #[test]
    fn output_overrun_revokes_lease_and_releases_pending_slot() {
        let (mut book, id, _, peer, resource_id) = lease();
        let permit = book.reserve(id, peer, request(resource_id), 2_001).unwrap();
        book.authorize_dispatch(&permit, 2_002).unwrap();
        assert_eq!(book.settle(permit, 2_049), Err(PeerError::QuotaExceeded));
        assert_eq!(book.pending_jobs(id), Ok(0));
        assert_eq!(
            book.reserve(id, peer, request(resource_id), 2_003).err(),
            Some(PeerError::LeaseRevoked)
        );
    }

    #[test]
    fn debug_output_redacts_resource_identifiers() {
        let scope = LeaseScope {
            job: DelegatedJobKind::FrameRender,
            resource_id: [0x91; 32],
            disclosure: DisclosureScope::ExplicitJobInputAndResult,
        };
        let debug = format!("{scope:?}");
        assert!(debug.contains("[redacted]"));
        assert!(!debug.contains("145"));
        assert!(!format!("{:?}", request([0x91; 32])).contains("145"));
    }

    #[test]
    fn lease_book_is_bound_to_its_issuing_device_identity() {
        let owner_identity = DeviceIdentity::from_seed([4; 32]);
        let other_owner = DeviceIdentity::from_seed([6; 32]).peer_id();
        let peer = DeviceIdentity::from_seed([5; 32]).peer_id();
        let lease = PeerLease::issue(
            &owner_identity,
            peer,
            2_000,
            60,
            [LeaseScope {
                job: DelegatedJobKind::BatchAnalysis,
                resource_id: [0x42; 32],
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 1,
                maximum_jobs: 1,
                maximum_input_bytes_per_job: 128,
                maximum_output_bytes_per_job: 128,
                maximum_total_input_bytes: 128,
                maximum_total_output_bytes: 128,
            },
        )
        .unwrap();
        assert_eq!(
            PeerLeaseBook::new(other_owner).insert(lease),
            Err(PeerError::LeaseDenied)
        );
    }
}
