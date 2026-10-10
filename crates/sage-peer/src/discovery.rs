//! Bounded signed LAN announcements for finding pairing candidates.
//!
//! Announcements reveal only a stable peer identity, a short-lived pairing
//! port and a random nonce. A valid signature proves possession of the
//! advertised signing key; it does not pair the device, create a lease, or
//! authorize any request. Native callers choose the interface and own the
//! socket lifecycle so platform-specific network policy stays explicit.

use std::{
    collections::BTreeMap,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use ed25519_dalek::{Signature, Signer, VerifyingKey};
use tokio::net::UdpSocket;

use crate::{DeviceIdentity, PeerError, PeerId, PeerResult, peer_id_for};

const DISCOVERY_MAGIC: &[u8; 4] = b"SGD1";
const DISCOVERY_VERSION: u16 = 1;
const DISCOVERY_DOMAIN: &[u8] = b"sage:peer:discovery:v1\0";
const SIGNING_KEY_BYTES: usize = 32;
const NONCE_BYTES: usize = 16;
const SIGNATURE_BYTES: usize = 64;
const BODY_BYTES: usize = 4 + 2 + SIGNING_KEY_BYTES + 8 + 8 + 2 + NONCE_BYTES;
pub const PEER_DISCOVERY_DATAGRAM_BYTES: usize = BODY_BYTES + SIGNATURE_BYTES;
pub const PEER_DISCOVERY_MAX_TTL_SECONDS: u64 = 30;
const MAX_CANDIDATES: usize = 64;
const MAX_SEEN_NONCES: usize = 256;
const MAX_DISCOVERY_DATAGRAMS_PER_SECOND: usize = 128;

/// One signed, short-lived announcement. It contains no invitation secret.
#[derive(Debug, Clone)]
pub struct PeerDiscoveryAdvertisement {
    bytes: Vec<u8>,
    expires_at_unix_seconds: u64,
}

impl PeerDiscoveryAdvertisement {
    /// Create a signature-bound announcement for a currently open pairing
    /// listener. Its expiry is at most thirty seconds from creation.
    pub fn create(
        identity: &DeviceIdentity,
        pairing_port: u16,
        lifetime_seconds: u64,
    ) -> PeerResult<Self> {
        let issued_at = unix_time_now()?;
        if pairing_port == 0
            || lifetime_seconds == 0
            || lifetime_seconds > PEER_DISCOVERY_MAX_TTL_SECONDS
        {
            return Err(PeerError::InvalidInput);
        }
        let expires_at = issued_at
            .checked_add(lifetime_seconds)
            .ok_or(PeerError::InvalidInput)?;
        let mut nonce = [0_u8; NONCE_BYTES];
        getrandom::fill(&mut nonce).map_err(|_| PeerError::RandomnessUnavailable)?;
        let verifying_key = identity.signing_key.verifying_key().to_bytes();
        let mut body = Vec::with_capacity(BODY_BYTES);
        body.extend_from_slice(DISCOVERY_MAGIC);
        body.extend_from_slice(&DISCOVERY_VERSION.to_be_bytes());
        body.extend_from_slice(&verifying_key);
        body.extend_from_slice(&issued_at.to_be_bytes());
        body.extend_from_slice(&expires_at.to_be_bytes());
        body.extend_from_slice(&pairing_port.to_be_bytes());
        body.extend_from_slice(&nonce);

        let mut signing_bytes = Vec::with_capacity(DISCOVERY_DOMAIN.len() + BODY_BYTES);
        signing_bytes.extend_from_slice(DISCOVERY_DOMAIN);
        signing_bytes.extend_from_slice(&body);
        let signature = identity.signing_key.sign(&signing_bytes);
        body.extend_from_slice(&signature.to_bytes());
        debug_assert_eq!(body.len(), PEER_DISCOVERY_DATAGRAM_BYTES);

        Ok(Self {
            bytes: body,
            expires_at_unix_seconds: expires_at,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn expires_at_unix_seconds(&self) -> u64 {
        self.expires_at_unix_seconds
    }
}

/// Cryptographically checked announcement data. It is still only a candidate
/// and must pass the ordinary user-confirmed pairing exchange before trust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPeerAnnouncement {
    peer_id: PeerId,
    issued_at_unix_seconds: u64,
    expires_at_unix_seconds: u64,
    pairing_port: u16,
    nonce: [u8; NONCE_BYTES],
}

impl VerifiedPeerAnnouncement {
    pub fn peer_id(&self) -> PeerId {
        self.peer_id
    }

    pub fn pairing_port(&self) -> u16 {
        self.pairing_port
    }

    pub fn issued_at_unix_seconds(&self) -> u64 {
        self.issued_at_unix_seconds
    }

    pub fn expires_at_unix_seconds(&self) -> u64 {
        self.expires_at_unix_seconds
    }
}

/// Verify an exact-size discovery datagram against the current wall clock.
/// Small clock skew is tolerated only for a just-created announcement.
pub fn verify_peer_announcement(
    packet: &[u8],
    now_unix_seconds: u64,
) -> PeerResult<VerifiedPeerAnnouncement> {
    if packet.len() != PEER_DISCOVERY_DATAGRAM_BYTES
        || &packet[..4] != DISCOVERY_MAGIC
        || u16::from_be_bytes([packet[4], packet[5]]) != DISCOVERY_VERSION
    {
        return Err(PeerError::InvalidInput);
    }

    let key_start = 6;
    let issued_start = key_start + SIGNING_KEY_BYTES;
    let expires_start = issued_start + 8;
    let port_start = expires_start + 8;
    let nonce_start = port_start + 2;
    let signature_start = BODY_BYTES;
    let verifying_key_bytes: [u8; SIGNING_KEY_BYTES] = packet[key_start..issued_start]
        .try_into()
        .map_err(|_| PeerError::InvalidInput)?;
    let issued_at = u64::from_be_bytes(
        packet[issued_start..expires_start]
            .try_into()
            .map_err(|_| PeerError::InvalidInput)?,
    );
    let expires_at = u64::from_be_bytes(
        packet[expires_start..port_start]
            .try_into()
            .map_err(|_| PeerError::InvalidInput)?,
    );
    let pairing_port = u16::from_be_bytes(
        packet[port_start..nonce_start]
            .try_into()
            .map_err(|_| PeerError::InvalidInput)?,
    );
    let nonce: [u8; NONCE_BYTES] = packet[nonce_start..BODY_BYTES]
        .try_into()
        .map_err(|_| PeerError::InvalidInput)?;
    let lifetime = expires_at
        .checked_sub(issued_at)
        .ok_or(PeerError::InvalidInput)?;
    if pairing_port == 0
        || nonce.iter().all(|byte| *byte == 0)
        || lifetime == 0
        || lifetime > PEER_DISCOVERY_MAX_TTL_SECONDS
        || issued_at > now_unix_seconds.saturating_add(2)
        || expires_at <= now_unix_seconds
        || expires_at > now_unix_seconds.saturating_add(PEER_DISCOVERY_MAX_TTL_SECONDS + 2)
    {
        return Err(PeerError::Expired);
    }

    let key = VerifyingKey::from_bytes(&verifying_key_bytes)
        .map_err(|_| PeerError::AuthenticationFailed)?;
    let signature_bytes: [u8; SIGNATURE_BYTES] = packet[signature_start..]
        .try_into()
        .map_err(|_| PeerError::InvalidInput)?;
    let signature = Signature::from_bytes(&signature_bytes);
    let mut signing_bytes = Vec::with_capacity(DISCOVERY_DOMAIN.len() + BODY_BYTES);
    signing_bytes.extend_from_slice(DISCOVERY_DOMAIN);
    signing_bytes.extend_from_slice(&packet[..BODY_BYTES]);
    key.verify_strict(&signing_bytes, &signature)
        .map_err(|_| PeerError::AuthenticationFailed)?;

    Ok(VerifiedPeerAnnouncement {
        peer_id: peer_id_for(&verifying_key_bytes),
        issued_at_unix_seconds: issued_at,
        expires_at_unix_seconds: expires_at,
        pairing_port,
        nonce,
    })
}

/// A volatile UI candidate. Its network address is observed metadata, not a
/// signed identity claim or a grant to connect with privileged operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerDiscoveryCandidate {
    peer_id: PeerId,
    pairing_address: SocketAddr,
    expires_at_unix_seconds: u64,
    valid_until: Instant,
}

impl PeerDiscoveryCandidate {
    pub fn peer_id(&self) -> PeerId {
        self.peer_id
    }

    pub fn pairing_address(&self) -> SocketAddr {
        self.pairing_address
    }

    pub fn expires_at_unix_seconds(&self) -> u64 {
        self.expires_at_unix_seconds
    }
}

/// Bounded, volatile replay and candidate state for one local Sage identity.
#[derive(Debug)]
pub struct PeerDiscoveryCache {
    local_peer_id: PeerId,
    candidates: BTreeMap<PeerId, PeerDiscoveryCandidate>,
    seen_nonces: BTreeMap<(PeerId, [u8; NONCE_BYTES]), Instant>,
}

impl PeerDiscoveryCache {
    pub fn new(local_peer_id: PeerId) -> Self {
        Self {
            local_peer_id,
            candidates: BTreeMap::new(),
            seen_nonces: BTreeMap::new(),
        }
    }

    /// Add a verified announcement as an expiring candidate. Self adverts and
    /// replayed nonces are ignored. The cache never persists or pairs peers.
    pub fn observe(
        &mut self,
        announcement: VerifiedPeerAnnouncement,
        source: SocketAddr,
        now_unix_seconds: u64,
        observed_at: Instant,
    ) -> Option<PeerDiscoveryCandidate> {
        self.expire(observed_at);
        if announcement.peer_id == self.local_peer_id
            || announcement.expires_at_unix_seconds <= now_unix_seconds
        {
            return None;
        }
        let replay_key = (announcement.peer_id, announcement.nonce);
        if self.seen_nonces.contains_key(&replay_key) {
            return None;
        }
        if !self.candidates.contains_key(&announcement.peer_id)
            && self.candidates.len() >= MAX_CANDIDATES
        {
            return None;
        }
        if self.seen_nonces.len() >= MAX_SEEN_NONCES
            && let Some(oldest) = self
                .seen_nonces
                .iter()
                .min_by_key(|(_, expires)| **expires)
                .map(|(key, _)| *key)
        {
            self.seen_nonces.remove(&oldest);
        }

        let lifetime = announcement
            .expires_at_unix_seconds
            .saturating_sub(now_unix_seconds)
            .min(PEER_DISCOVERY_MAX_TTL_SECONDS);
        if lifetime == 0 {
            return None;
        }
        let valid_until = observed_at + Duration::from_secs(lifetime);
        let candidate = PeerDiscoveryCandidate {
            peer_id: announcement.peer_id,
            pairing_address: SocketAddr::new(source.ip(), announcement.pairing_port),
            expires_at_unix_seconds: announcement.expires_at_unix_seconds,
            valid_until,
        };
        self.seen_nonces.insert(replay_key, valid_until);
        self.candidates
            .insert(announcement.peer_id, candidate.clone());
        Some(candidate)
    }

    /// Expire candidates and replay markers using monotonic time.
    pub fn expire(&mut self, now: Instant) -> usize {
        let before = self.candidates.len();
        self.candidates
            .retain(|_, candidate| candidate.valid_until > now);
        self.seen_nonces.retain(|_, until| *until > now);
        before - self.candidates.len()
    }

    pub fn candidates(&mut self, now: Instant) -> Vec<PeerDiscoveryCandidate> {
        self.expire(now);
        self.candidates.values().cloned().collect()
    }
}

/// IPv4 multicast discovery socket. The native caller supplies the selected
/// interface address; the kernel limits multicast propagation to one hop.
pub struct LanDiscoveryService {
    socket: UdpSocket,
    destination: SocketAddr,
    rate_limit: DiscoveryRateLimit,
}

#[derive(Debug)]
struct DiscoveryRateLimit {
    window_started: Instant,
    admitted: usize,
}

impl DiscoveryRateLimit {
    fn new(now: Instant) -> Self {
        Self {
            window_started: now,
            admitted: 0,
        }
    }

    fn admit(&mut self, now: Instant) -> bool {
        if now.saturating_duration_since(self.window_started) >= Duration::from_secs(1) {
            self.window_started = now;
            self.admitted = 0;
        }
        if self.admitted >= MAX_DISCOVERY_DATAGRAMS_PER_SECOND {
            return false;
        }
        self.admitted += 1;
        true
    }
}

impl LanDiscoveryService {
    pub async fn bind_multicast(
        multicast_group: Ipv4Addr,
        multicast_port: u16,
        local_port: u16,
        interface: Ipv4Addr,
    ) -> PeerResult<Self> {
        if !multicast_group.is_multicast() || multicast_port == 0 {
            return Err(PeerError::InvalidInput);
        }
        let socket = UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, local_port))
            .await
            .map_err(|_| PeerError::DiscoveryIo)?;
        socket
            .join_multicast_v4(multicast_group, interface)
            .map_err(|_| PeerError::DiscoveryIo)?;
        socket
            .set_multicast_ttl_v4(1)
            .map_err(|_| PeerError::DiscoveryIo)?;
        socket
            .set_multicast_loop_v4(true)
            .map_err(|_| PeerError::DiscoveryIo)?;
        Ok(Self {
            socket,
            destination: SocketAddr::V4(SocketAddrV4::new(multicast_group, multicast_port)),
            rate_limit: DiscoveryRateLimit::new(Instant::now()),
        })
    }

    /// Wrap a socket configured by a native interface manager. Useful where
    /// platform code must set reuse or interface-specific socket options.
    pub fn from_socket(socket: UdpSocket, destination: SocketAddr) -> Self {
        Self {
            socket,
            destination,
            rate_limit: DiscoveryRateLimit::new(Instant::now()),
        }
    }

    pub fn local_addr(&self) -> PeerResult<SocketAddr> {
        self.socket.local_addr().map_err(|_| PeerError::DiscoveryIo)
    }

    pub async fn announce(&self, announcement: &PeerDiscoveryAdvertisement) -> PeerResult<()> {
        let sent = self
            .socket
            .send_to(announcement.as_bytes(), self.destination)
            .await
            .map_err(|_| PeerError::DiscoveryIo)?;
        if sent != PEER_DISCOVERY_DATAGRAM_BYTES {
            return Err(PeerError::DiscoveryIo);
        }
        Ok(())
    }

    /// Receive one datagram. Malformed or unauthenticated packets are dropped
    /// as ordinary network noise; socket failures are returned to the owner.
    pub async fn receive_candidate(
        &mut self,
        cache: &mut PeerDiscoveryCache,
    ) -> PeerResult<Option<PeerDiscoveryCandidate>> {
        let mut packet = [0_u8; PEER_DISCOVERY_DATAGRAM_BYTES + 1];
        let (length, source) = self
            .socket
            .recv_from(&mut packet)
            .await
            .map_err(|_| PeerError::DiscoveryIo)?;
        if !self.rate_limit.admit(Instant::now()) {
            return Ok(None);
        }
        if length != PEER_DISCOVERY_DATAGRAM_BYTES {
            return Ok(None);
        }
        let now_unix_seconds = unix_time_now()?;
        let Ok(announcement) = verify_peer_announcement(&packet[..length], now_unix_seconds) else {
            return Ok(None);
        };
        Ok(cache.observe(announcement, source, now_unix_seconds, Instant::now()))
    }
}

fn unix_time_now() -> PeerResult<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PeerError::DiscoveryClock)
        .map(|duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(seed: u8) -> DeviceIdentity {
        DeviceIdentity::from_seed([seed; 32])
    }

    #[test]
    fn signed_announcement_binds_identity_listener_and_expiry() {
        let owner = identity(7);
        let advertisement = PeerDiscoveryAdvertisement::create(&owner, 43127, 30).unwrap();
        let verified =
            verify_peer_announcement(advertisement.as_bytes(), unix_time_now().unwrap()).unwrap();
        assert_eq!(verified.peer_id(), owner.peer_id());
        assert_eq!(verified.pairing_port(), 43127);
        assert!(verified.expires_at_unix_seconds() - verified.issued_at_unix_seconds() <= 30);

        let mut changed_port = advertisement.as_bytes().to_vec();
        let port_offset = 6 + SIGNING_KEY_BYTES + 16;
        changed_port[port_offset + 1] ^= 1;
        assert_eq!(
            verify_peer_announcement(&changed_port, unix_time_now().unwrap()),
            Err(PeerError::AuthenticationFailed)
        );
        assert_eq!(
            verify_peer_announcement(
                advertisement.as_bytes(),
                advertisement.expires_at_unix_seconds() + 1
            ),
            Err(PeerError::Expired)
        );
    }

    #[test]
    fn candidate_cache_drops_self_replay_and_expired_announcements() {
        let local = identity(1);
        let remote = identity(2);
        let local_id = local.peer_id();
        let now_unix = unix_time_now().unwrap();
        let now = Instant::now();
        let mut cache = PeerDiscoveryCache::new(local_id);

        let self_ad = PeerDiscoveryAdvertisement::create(&local, 9000, 30).unwrap();
        let self_verified = verify_peer_announcement(self_ad.as_bytes(), now_unix).unwrap();
        assert!(
            cache
                .observe(
                    self_verified,
                    "127.0.0.1:12000".parse().unwrap(),
                    now_unix,
                    now,
                )
                .is_none()
        );

        let remote_ad = PeerDiscoveryAdvertisement::create(&remote, 43127, 30).unwrap();
        let packet = remote_ad.as_bytes().to_vec();
        let verified = verify_peer_announcement(&packet, now_unix).unwrap();
        let source = "192.0.2.12:54700".parse().unwrap();
        let candidate = cache
            .observe(verified, source, now_unix, now)
            .expect("verified peer becomes an expiring candidate");
        assert_eq!(candidate.peer_id(), remote.peer_id());
        assert_eq!(
            candidate.pairing_address(),
            "192.0.2.12:43127".parse().unwrap()
        );

        let replay = verify_peer_announcement(&packet, now_unix).unwrap();
        assert!(cache.observe(replay, source, now_unix, now).is_none());
        assert_eq!(cache.candidates(now).len(), 1);
        assert_eq!(cache.expire(now + Duration::from_secs(31)), 1);
        assert!(cache.candidates(now + Duration::from_secs(31)).is_empty());
    }

    #[test]
    fn discovery_rate_limit_bounds_signature_verification_work() {
        let start = Instant::now();
        let mut limiter = DiscoveryRateLimit::new(start);
        for _ in 0..MAX_DISCOVERY_DATAGRAMS_PER_SECOND {
            assert!(limiter.admit(start));
        }
        assert!(!limiter.admit(start));
        assert!(limiter.admit(start + Duration::from_secs(1)));
    }

    #[tokio::test]
    async fn loopback_discovery_socket_returns_only_a_verified_candidate() {
        let receiver = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let receiver_address = receiver.local_addr().unwrap();
        let mut service = LanDiscoveryService::from_socket(receiver, receiver_address);
        let sender = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let remote = identity(9);
        let local = identity(10);
        let announcement = PeerDiscoveryAdvertisement::create(&remote, 46001, 30).unwrap();
        let mut cache = PeerDiscoveryCache::new(local.peer_id());
        let sender = LanDiscoveryService::from_socket(sender, receiver_address);

        sender
            .socket
            .send_to(b"not a Sage discovery packet", receiver_address)
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(
                Duration::from_secs(1),
                service.receive_candidate(&mut cache),
            )
            .await
            .unwrap()
            .unwrap()
            .is_none()
        );

        sender.announce(&announcement).await.unwrap();
        let candidate = tokio::time::timeout(
            Duration::from_secs(1),
            service.receive_candidate(&mut cache),
        )
        .await
        .unwrap()
        .unwrap()
        .unwrap();
        assert_eq!(candidate.peer_id(), remote.peer_id());
        assert_eq!(candidate.pairing_address().port(), 46001);
    }
}
