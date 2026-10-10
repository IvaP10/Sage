//! First-party trusted-peer pairing and encrypted message primitives.
//!
//! This crate owns Sage's pairing transcript, identity binding, short
//! authentication string, replay sequence, frame encoding, and message
//! policy. Cryptographic primitives are narrow dependencies: Ed25519 for
//! signatures, X25519 for ephemeral agreement, HMAC-SHA-256 for transcript
//! key derivation, and XChaCha20-Poly1305 for authenticated encryption.
//! Network discovery advertisements produce candidates only; pairing, not
//! discovery, establishes trust. OS key-store integration and OS-isolated
//! worker processes remain outside this crate. Bounded partition/result codecs feed
//! a plan-and-live-lease admission gate; closed CPU executors consume only
//! explicitly authorized input bytes, and remote output still requires a
//! caller verifier.

#![forbid(unsafe_code)]

use std::{fmt, time::Duration};

use base64::Engine;
use chacha20poly1305::{
    Key, Tag, XChaCha20Poly1305, XNonce,
    aead::{AeadInOut, KeyInit, inout::InOutBuf},
};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use zeroize::{Zeroize, Zeroizing};

type HmacSha256 = Hmac<Sha256>;

const PAIRING_VERSION: u16 = 1;
const QR_MAGIC: &[u8; 4] = b"SGP1";
const PAIRING_ACK_MAGIC: &[u8; 4] = b"SGA1";
const PAIRING_ACK_VERSION: u16 = 1;
const OFFER_DOMAIN: &[u8] = b"sage:peer:pair-offer:v1\0";
const RESPONSE_DOMAIN: &[u8] = b"sage:peer:pair-response:v1\0";
const CONFIRM_DOMAIN: &[u8] = b"sage:peer:pair-confirm:v1\0";
const ACK_DOMAIN: &[u8] = b"sage:peer:pair-ack:v1\0";
const TRANSCRIPT_DOMAIN: &[u8] = b"sage:peer:pair-transcript:v1\0";
const KDF_SALT_DOMAIN: &[u8] = b"sage:peer:kdf-salt:v1\0";
const KDF_ROOT_INFO: &[u8] = b"sage:peer:trust-root:v1\0";
const KDF_INITIATOR_TO_RESPONDER: &[u8] = b"sage:peer:initiator-to-responder:v1\0";
const KDF_RESPONDER_TO_INITIATOR: &[u8] = b"sage:peer:responder-to-initiator:v1\0";
const MIN_INVITATION_TTL_SECONDS: u64 = 30;
const MAX_INVITATION_TTL_SECONDS: u64 = 300;
const MAX_PEER_LEASE_SECONDS: u64 = 3_600;
const MAX_PEER_MESSAGE_BYTES: usize = 1024 * 1024;
pub const MAX_TASK_HANDOFF_BYTES: usize = 4 * 1024 * 1024;
/// Maximum size of one task-owned artifact body accepted by the handoff codec.
pub const MAX_TASK_HANDOFF_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
const MAX_PEER_COMPUTE_BYTES: u64 = 256 * 1024 * 1024;
const PEER_FRAME_VERSION: u16 = 1;
const PEER_FRAME_HEADER_BYTES: usize = 15;
const AEAD_TAG_BYTES: usize = 16;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum PeerError {
    #[error("secure randomness was unavailable")]
    RandomnessUnavailable,
    #[error("pairing input is malformed or outside its bound")]
    InvalidInput,
    #[error("peer discovery socket operation failed")]
    DiscoveryIo,
    #[error("system clock is unavailable for peer discovery")]
    DiscoveryClock,
    #[error("pairing invitation has expired")]
    Expired,
    #[error("pairing identity or transcript authentication failed")]
    AuthenticationFailed,
    #[error("the two devices displayed different authentication codes")]
    AuthenticationCodeMismatch,
    #[error("the encrypted frame is malformed or exceeds its bound")]
    InvalidFrame,
    #[error("the encrypted frame is replayed or out of order")]
    ReplayOrReordering,
    #[error("the secure channel sequence is exhausted")]
    SequenceExhausted,
    #[error("peer lease is unknown, expired, or malformed")]
    InvalidLease,
    #[error("peer lease is revoked")]
    LeaseRevoked,
    #[error("peer lease does not cover the requested job or resource")]
    LeaseDenied,
    #[error("peer lease quota is exhausted")]
    QuotaExceeded,
    #[error("peer lease reservation is stale or in the wrong state")]
    InvalidPermit,
    #[error("compute job or partition is malformed or outside its bound")]
    InvalidComputeJob,
    #[error("compute partition result does not match its assigned work")]
    InvalidComputeResult,
    #[error("this job partition was already reserved or dispatched under the lease")]
    DuplicateJob,
    #[error("compute output failed its task-specific verification")]
    ComputeVerificationFailed,
    #[error("peer compute partition was cooperatively cancelled")]
    ComputeCancelled,
    #[error("peer dispatch journal is unavailable")]
    DispatchJournalUnavailable,
    #[error("peer dispatch journal is already open by another broker")]
    DispatchJournalInUse,
    #[error("peer dispatch journal integrity key is invalid")]
    DispatchJournalKeyInvalid,
    #[error("peer dispatch journal is corrupt or was truncated")]
    DispatchJournalCorrupt,
    #[error("peer dispatch journal reached its configured capacity")]
    DispatchJournalFull,
}

pub type PeerResult<T> = Result<T, PeerError>;

/// A stable device identifier derived from its Ed25519 signing key.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PeerId([u8; 32]);

impl PeerId {
    /// Restore an already-stored 32-byte peer identifier. This checks only
    /// its wire width; it does not authenticate a device or establish trust.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for PeerId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PeerId([redacted])")
    }
}

/// Device signing identity. Persist its 32-byte seed only through the native
/// OS key store; this type deliberately provides no serialization or Clone.
pub struct DeviceIdentity {
    signing_key: SigningKey,
}

impl DeviceIdentity {
    pub fn generate() -> PeerResult<Self> {
        let mut seed = Zeroizing::new([0_u8; 32]);
        getrandom::fill(seed.as_mut()).map_err(|_| PeerError::RandomnessUnavailable)?;
        Ok(Self::from_seed(*seed))
    }

    pub fn from_seed(seed: [u8; 32]) -> Self {
        let seed = Zeroizing::new(seed);
        Self {
            signing_key: SigningKey::from_bytes(&seed),
        }
    }

    pub fn peer_id(&self) -> PeerId {
        peer_id_for(&self.signing_key.verifying_key().to_bytes())
    }

    pub fn public_identity(&self) -> PublicPeerIdentity {
        let verifying_key = self.signing_key.verifying_key().to_bytes();
        PublicPeerIdentity {
            peer_id: peer_id_for(&verifying_key),
            verifying_key,
        }
    }
}

impl fmt::Debug for DeviceIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceIdentity")
            .field("peer_id", &self.peer_id())
            .field("signing_key", &"[redacted]")
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PublicPeerIdentity {
    peer_id: PeerId,
    verifying_key: [u8; 32],
}

impl PublicPeerIdentity {
    pub fn peer_id(&self) -> PeerId {
        self.peer_id
    }

    pub fn verifying_key_bytes(&self) -> &[u8; 32] {
        &self.verifying_key
    }
}

impl fmt::Debug for PublicPeerIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicPeerIdentity")
            .field("peer_id", &self.peer_id)
            .finish_non_exhaustive()
    }
}

/// A compact QR payload. Its secret is never included in Debug output.
pub struct PairingQr {
    offer: PairingOffer,
    invitation_secret: Zeroizing<[u8; 32]>,
}

impl PairingQr {
    /// Encode a bounded binary invitation using URL-safe unpadded Base64 for
    /// native QR renderers. The QR is a pairing secret and should be shown only
    /// for the invitation's short lifetime.
    pub fn to_payload(&self) -> String {
        let mut bytes = Zeroizing::new(self.offer.body());
        bytes.extend_from_slice(&self.offer.signature);
        bytes.extend_from_slice(self.invitation_secret.as_ref());
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    }

    pub fn from_payload(payload: &str) -> PeerResult<Self> {
        if payload.len() > 512 {
            return Err(PeerError::InvalidInput);
        }
        let bytes = Zeroizing::new(
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(payload)
                .map_err(|_| PeerError::InvalidInput)?,
        );
        let expected_bytes = offer_body_len() + 64 + 32;
        if bytes.len() != expected_bytes || !bytes.starts_with(QR_MAGIC) {
            return Err(PeerError::InvalidInput);
        }
        let mut offset = QR_MAGIC.len();
        let version = read_u16(&bytes, &mut offset)?;
        if version != PAIRING_VERSION {
            return Err(PeerError::InvalidInput);
        }
        let pairing_id = read_array::<16>(&bytes, &mut offset)?;
        let expires_at_unix_seconds = read_u64(&bytes, &mut offset)?;
        let inviter_verifying_key = read_array::<32>(&bytes, &mut offset)?;
        let inviter_ephemeral_key = read_array::<32>(&bytes, &mut offset)?;
        let inviter_nonce = read_array::<32>(&bytes, &mut offset)?;
        let invitation_commitment = read_array::<32>(&bytes, &mut offset)?;
        let signature = read_array::<64>(&bytes, &mut offset)?;
        let invitation_secret = read_array::<32>(&bytes, &mut offset)?;
        if offset != bytes.len() {
            return Err(PeerError::InvalidInput);
        }
        Ok(Self {
            offer: PairingOffer {
                pairing_id,
                expires_at_unix_seconds,
                inviter_verifying_key,
                inviter_ephemeral_key,
                inviter_nonce,
                invitation_commitment,
                signature,
            },
            invitation_secret: Zeroizing::new(invitation_secret),
        })
    }

    /// Validate the signed, expiring invitation before showing its identity to
    /// the receiving user. This creates no trust and sends no response.
    pub fn verified_inviter(&self, now_unix_seconds: u64) -> PeerResult<PublicPeerIdentity> {
        validate_offer(&self.offer, &self.invitation_secret, now_unix_seconds)?;
        Ok(PublicPeerIdentity {
            peer_id: peer_id_for(&self.offer.inviter_verifying_key),
            verifying_key: self.offer.inviter_verifying_key,
        })
    }

    /// Called by the receiving device only after its UI has shown the inviter
    /// identity and the local user has explicitly accepted the invitation.
    pub fn respond_after_local_confirmation(
        &self,
        identity: &DeviceIdentity,
        now_unix_seconds: u64,
    ) -> PeerResult<(PairingResponse, PendingResponder)> {
        validate_offer(&self.offer, &self.invitation_secret, now_unix_seconds)?;
        let local_identity = identity.public_identity();
        if local_identity.peer_id == peer_id_for(&self.offer.inviter_verifying_key) {
            return Err(PeerError::InvalidInput);
        }
        let (ephemeral_secret, ephemeral_public) = random_x25519_keypair()?;
        let mut responder_nonce = [0_u8; 32];
        getrandom::fill(&mut responder_nonce).map_err(|_| PeerError::RandomnessUnavailable)?;
        let offer_digest = self.offer.digest();
        let mut response = PairingResponse {
            pairing_id: self.offer.pairing_id,
            responder_verifying_key: local_identity.verifying_key,
            responder_ephemeral_key: ephemeral_public,
            responder_nonce,
            offer_digest,
            signature: [0; 64],
        };
        response.signature = identity
            .signing_key
            .sign(&response.signing_bytes())
            .to_bytes();
        let pending = PendingResponder {
            offer: self.offer.clone(),
            invitation_secret: Zeroizing::new(*self.invitation_secret),
            response: response.clone(),
            ephemeral_secret,
        };
        Ok((response, pending))
    }
}

impl fmt::Debug for PairingQr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingQr")
            .field("pairing_id", &self.offer.pairing_id)
            .field(
                "expires_at_unix_seconds",
                &self.offer.expires_at_unix_seconds,
            )
            .field("invitation_secret", &"[redacted]")
            .finish()
    }
}

#[derive(Clone)]
struct PairingOffer {
    pairing_id: [u8; 16],
    expires_at_unix_seconds: u64,
    inviter_verifying_key: [u8; 32],
    inviter_ephemeral_key: [u8; 32],
    inviter_nonce: [u8; 32],
    invitation_commitment: [u8; 32],
    signature: [u8; 64],
}

impl PairingOffer {
    fn body(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(offer_body_len());
        body.extend_from_slice(QR_MAGIC);
        body.extend_from_slice(&PAIRING_VERSION.to_be_bytes());
        body.extend_from_slice(&self.pairing_id);
        body.extend_from_slice(&self.expires_at_unix_seconds.to_be_bytes());
        body.extend_from_slice(&self.inviter_verifying_key);
        body.extend_from_slice(&self.inviter_ephemeral_key);
        body.extend_from_slice(&self.inviter_nonce);
        body.extend_from_slice(&self.invitation_commitment);
        body
    }

    fn signed_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(OFFER_DOMAIN.len() + offer_body_len());
        bytes.extend_from_slice(OFFER_DOMAIN);
        bytes.extend_from_slice(&self.body());
        bytes
    }

    fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(self.body());
        hash.update(self.signature);
        hash.finalize().into()
    }
}

/// Signed proof that the responder scanned the exact invitation and presented
/// its identity and ephemeral key.
#[derive(Clone)]
pub struct PairingResponse {
    pairing_id: [u8; 16],
    responder_verifying_key: [u8; 32],
    responder_ephemeral_key: [u8; 32],
    responder_nonce: [u8; 32],
    offer_digest: [u8; 32],
    signature: [u8; 64],
}

impl PairingResponse {
    fn body(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(150);
        body.extend_from_slice(QR_MAGIC);
        body.extend_from_slice(&PAIRING_VERSION.to_be_bytes());
        body.extend_from_slice(&self.pairing_id);
        body.extend_from_slice(&self.responder_verifying_key);
        body.extend_from_slice(&self.responder_ephemeral_key);
        body.extend_from_slice(&self.responder_nonce);
        body.extend_from_slice(&self.offer_digest);
        body
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(RESPONSE_DOMAIN.len() + 150);
        bytes.extend_from_slice(RESPONSE_DOMAIN);
        bytes.extend_from_slice(&self.body());
        bytes
    }

    fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(self.body());
        hash.update(self.signature);
        hash.finalize().into()
    }

    pub fn responder(&self) -> PublicPeerIdentity {
        PublicPeerIdentity {
            peer_id: peer_id_for(&self.responder_verifying_key),
            verifying_key: self.responder_verifying_key,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = self.body();
        bytes.extend_from_slice(&self.signature);
        bytes
    }

    pub fn decode(bytes: &[u8]) -> PeerResult<Self> {
        if bytes.len() != 150 + 64 || !bytes.starts_with(QR_MAGIC) {
            return Err(PeerError::InvalidInput);
        }
        let mut offset = QR_MAGIC.len();
        if read_u16(bytes, &mut offset)? != PAIRING_VERSION {
            return Err(PeerError::InvalidInput);
        }
        let pairing_id = read_array::<16>(bytes, &mut offset)?;
        let responder_verifying_key = read_array::<32>(bytes, &mut offset)?;
        let responder_ephemeral_key = read_array::<32>(bytes, &mut offset)?;
        let responder_nonce = read_array::<32>(bytes, &mut offset)?;
        let offer_digest = read_array::<32>(bytes, &mut offset)?;
        let signature = read_array::<64>(bytes, &mut offset)?;
        if offset != bytes.len() {
            return Err(PeerError::InvalidInput);
        }
        Ok(Self {
            pairing_id,
            responder_verifying_key,
            responder_ephemeral_key,
            responder_nonce,
            offer_digest,
            signature,
        })
    }
}

impl fmt::Debug for PairingResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingResponse")
            .field("pairing_id", &self.pairing_id)
            .field("responder", &self.responder())
            .field("signature", &"[redacted]")
            .finish()
    }
}

/// Signed final approval from the invitation owner. It binds approval to the
/// exact responder identity and ephemeral transcript.
#[derive(Clone)]
pub struct PairingConfirmation {
    pairing_id: [u8; 16],
    response_digest: [u8; 32],
    inviter_signature: [u8; 64],
}

impl PairingConfirmation {
    fn body(pairing_id: [u8; 16], response_digest: [u8; 32]) -> Vec<u8> {
        let mut body = Vec::with_capacity(54);
        body.extend_from_slice(QR_MAGIC);
        body.extend_from_slice(&PAIRING_VERSION.to_be_bytes());
        body.extend_from_slice(&pairing_id);
        body.extend_from_slice(&response_digest);
        body
    }

    fn signing_bytes(pairing_id: [u8; 16], response_digest: [u8; 32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(CONFIRM_DOMAIN.len() + 50);
        bytes.extend_from_slice(CONFIRM_DOMAIN);
        bytes.extend_from_slice(&Self::body(pairing_id, response_digest));
        bytes
    }

    fn digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(Self::body(self.pairing_id, self.response_digest));
        hash.update(self.inviter_signature);
        hash.finalize().into()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Self::body(self.pairing_id, self.response_digest);
        bytes.extend_from_slice(&self.inviter_signature);
        bytes
    }

    pub fn decode(bytes: &[u8]) -> PeerResult<Self> {
        if bytes.len() != 54 + 64 || !bytes.starts_with(QR_MAGIC) {
            return Err(PeerError::InvalidInput);
        }
        let mut offset = QR_MAGIC.len();
        if read_u16(bytes, &mut offset)? != PAIRING_VERSION {
            return Err(PeerError::InvalidInput);
        }
        let pairing_id = read_array::<16>(bytes, &mut offset)?;
        let response_digest = read_array::<32>(bytes, &mut offset)?;
        let inviter_signature = read_array::<64>(bytes, &mut offset)?;
        if offset != bytes.len() {
            return Err(PeerError::InvalidInput);
        }
        Ok(Self {
            pairing_id,
            response_digest,
            inviter_signature,
        })
    }
}

impl fmt::Debug for PairingConfirmation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingConfirmation")
            .field("pairing_id", &self.pairing_id)
            .field("signature", &"[redacted]")
            .finish()
    }
}

/// Signed proof that the responder received and accepted the inviter's final
/// confirmation after its own user approved the matching authentication code.
pub(crate) struct PairingAcknowledgement {
    pairing_id: [u8; 16],
    response_digest: [u8; 32],
    confirmation_digest: [u8; 32],
    responder_signature: [u8; 64],
}

impl PairingAcknowledgement {
    const BODY_BYTES: usize = 4 + 2 + 16 + 32 + 32;

    fn body(
        pairing_id: [u8; 16],
        response_digest: [u8; 32],
        confirmation_digest: [u8; 32],
    ) -> Vec<u8> {
        let mut body = Vec::with_capacity(Self::BODY_BYTES);
        body.extend_from_slice(PAIRING_ACK_MAGIC);
        body.extend_from_slice(&PAIRING_ACK_VERSION.to_be_bytes());
        body.extend_from_slice(&pairing_id);
        body.extend_from_slice(&response_digest);
        body.extend_from_slice(&confirmation_digest);
        body
    }

    fn signing_bytes(
        pairing_id: [u8; 16],
        response_digest: [u8; 32],
        confirmation_digest: [u8; 32],
    ) -> Vec<u8> {
        let body = Self::body(pairing_id, response_digest, confirmation_digest);
        let mut bytes = Vec::with_capacity(ACK_DOMAIN.len() + body.len());
        bytes.extend_from_slice(ACK_DOMAIN);
        bytes.extend_from_slice(&body);
        bytes
    }

    /// Create this only after the responder validated the inviter's signature
    /// and its local user accepted the authentication code.
    pub(crate) fn sign_after_local_confirmation(
        identity: &DeviceIdentity,
        response: &PairingResponse,
        confirmation: &PairingConfirmation,
    ) -> PeerResult<Self> {
        if identity.peer_id() != response.responder().peer_id()
            || confirmation.pairing_id != response.pairing_id
            || confirmation.response_digest != response.digest()
        {
            return Err(PeerError::AuthenticationFailed);
        }
        let response_digest = response.digest();
        let confirmation_digest = confirmation.digest();
        let signing_bytes =
            Self::signing_bytes(response.pairing_id, response_digest, confirmation_digest);
        let responder_signature = identity.signing_key.sign(&signing_bytes).to_bytes();
        Ok(Self {
            pairing_id: response.pairing_id,
            response_digest,
            confirmation_digest,
            responder_signature,
        })
    }

    /// Check that the acknowledgement came from the exact responder whose
    /// signed response and the inviter's exact confirmation form this pairing.
    pub(crate) fn verify(
        &self,
        response: &PairingResponse,
        confirmation: &PairingConfirmation,
    ) -> PeerResult<()> {
        if self.pairing_id != response.pairing_id
            || self.response_digest != response.digest()
            || confirmation.pairing_id != response.pairing_id
            || confirmation.response_digest != response.digest()
            || self.confirmation_digest != confirmation.digest()
        {
            return Err(PeerError::AuthenticationFailed);
        }
        let key = VerifyingKey::from_bytes(response.responder().verifying_key_bytes())
            .map_err(|_| PeerError::AuthenticationFailed)?;
        key.verify_strict(
            &Self::signing_bytes(
                self.pairing_id,
                self.response_digest,
                self.confirmation_digest,
            ),
            &Signature::from_bytes(&self.responder_signature),
        )
        .map_err(|_| PeerError::AuthenticationFailed)
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut bytes = Self::body(
            self.pairing_id,
            self.response_digest,
            self.confirmation_digest,
        );
        bytes.extend_from_slice(&self.responder_signature);
        bytes
    }

    pub(crate) fn decode(bytes: &[u8]) -> PeerResult<Self> {
        if bytes.len() != Self::BODY_BYTES + 64 || &bytes[..4] != PAIRING_ACK_MAGIC {
            return Err(PeerError::InvalidInput);
        }
        let mut offset = PAIRING_ACK_MAGIC.len();
        if read_u16(bytes, &mut offset)? != PAIRING_ACK_VERSION {
            return Err(PeerError::InvalidInput);
        }
        let pairing_id = read_array::<16>(bytes, &mut offset)?;
        let response_digest = read_array::<32>(bytes, &mut offset)?;
        let confirmation_digest = read_array::<32>(bytes, &mut offset)?;
        let responder_signature = read_array::<64>(bytes, &mut offset)?;
        if offset != bytes.len() {
            return Err(PeerError::InvalidInput);
        }
        Ok(Self {
            pairing_id,
            response_digest,
            confirmation_digest,
            responder_signature,
        })
    }
}

impl fmt::Debug for PairingAcknowledgement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingAcknowledgement")
            .field("pairing_id", &self.pairing_id)
            .field("signature", &"[redacted]")
            .finish()
    }
}

/// The invitation owner retains this only while the QR pairing is pending.
pub struct PendingInitiator {
    offer: PairingOffer,
    invitation_secret: Zeroizing<[u8; 32]>,
    ephemeral_secret: StaticSecret,
}

impl PendingInitiator {
    /// Returns the short authentication string that both native UIs must show
    /// to their users and compare before confirming the peer.
    pub fn authentication_code(&self, response: &PairingResponse) -> PeerResult<String> {
        validate_response(&self.offer, response)?;
        Ok(authentication_code(
            &self.offer,
            response,
            &self.invitation_secret,
        ))
    }

    /// Called only after the invitation owner has confirmed the responder and
    /// compared the same short code shown on both devices.
    pub fn confirm_after_local_confirmation(
        self,
        identity: &DeviceIdentity,
        response: &PairingResponse,
        displayed_code: &str,
        now_unix_seconds: u64,
    ) -> PeerResult<(PairingConfirmation, PairingOutcome)> {
        validate_offer(&self.offer, &self.invitation_secret, now_unix_seconds)?;
        if identity.peer_id() != peer_id_for(&self.offer.inviter_verifying_key) {
            return Err(PeerError::AuthenticationFailed);
        }
        validate_response(&self.offer, response)?;
        let expected_code = authentication_code(&self.offer, response, &self.invitation_secret);
        if !constant_time_string_eq(&expected_code, displayed_code) {
            return Err(PeerError::AuthenticationCodeMismatch);
        }
        let response_digest = response.digest();
        let inviter_signature = identity
            .signing_key
            .sign(&PairingConfirmation::signing_bytes(
                self.offer.pairing_id,
                response_digest,
            ))
            .to_bytes();
        let confirmation = PairingConfirmation {
            pairing_id: self.offer.pairing_id,
            response_digest,
            inviter_signature,
        };
        let outcome = derive_pairing_outcome(
            self.ephemeral_secret,
            response.responder_ephemeral_key,
            &self.invitation_secret,
            &self.offer,
            response,
            &confirmation,
            true,
        )?;
        Ok((confirmation, outcome))
    }
}

/// The responding device retains this until it receives the invitation
/// owner's signed confirmation and the user compares the displayed code.
pub struct PendingResponder {
    offer: PairingOffer,
    invitation_secret: Zeroizing<[u8; 32]>,
    response: PairingResponse,
    ephemeral_secret: StaticSecret,
}

impl PendingResponder {
    pub fn authentication_code(&self) -> String {
        authentication_code(&self.offer, &self.response, &self.invitation_secret)
    }

    /// Called only after the responder has verified the invitation owner's
    /// confirmation, displayed the matching code, and received local approval.
    pub fn finish_after_local_confirmation(
        self,
        confirmation: &PairingConfirmation,
        displayed_code: &str,
        now_unix_seconds: u64,
    ) -> PeerResult<PairingOutcome> {
        validate_offer(&self.offer, &self.invitation_secret, now_unix_seconds)?;
        validate_confirmation(&self.offer, &self.response, confirmation)?;
        let expected_code =
            authentication_code(&self.offer, &self.response, &self.invitation_secret);
        if !constant_time_string_eq(&expected_code, displayed_code) {
            return Err(PeerError::AuthenticationCodeMismatch);
        }
        derive_pairing_outcome(
            self.ephemeral_secret,
            self.offer.inviter_ephemeral_key,
            &self.invitation_secret,
            &self.offer,
            &self.response,
            confirmation,
            false,
        )
    }
}

/// Pairwise root material must be stored by a native OS key store. It is never
/// serializable and its Debug representation is redacted.
pub struct PeerTrustMaterial {
    secret: Zeroizing<[u8; 32]>,
}

impl PeerTrustMaterial {
    /// Copy the key into the caller's protected native-key-store operation.
    /// The caller should immediately drop or zeroize its temporary copy.
    pub fn copy_for_key_store(&self) -> Zeroizing<[u8; 32]> {
        Zeroizing::new(*self.secret)
    }
}

impl fmt::Debug for PeerTrustMaterial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PeerTrustMaterial([redacted])")
    }
}

pub struct PairingOutcome {
    pub local_peer_id: PeerId,
    pub trusted_peer: PublicPeerIdentity,
    pub trust_material: PeerTrustMaterial,
    pub channel: PeerSecureChannel,
}

impl fmt::Debug for PairingOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingOutcome")
            .field("local_peer_id", &self.local_peer_id)
            .field("trusted_peer", &self.trusted_peer)
            .field("trust_material", &"[redacted]")
            .field("channel", &self.channel)
            .finish()
    }
}

/// Message kinds are fixed and versioned. A peer cannot use the encrypted
/// channel as an arbitrary RPC or to request general access to host resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PeerMessageKind {
    LeaseOffer = 1,
    LeaseRevoke = 2,
    JobOffer = 3,
    JobReceipt = 4,
    Checkpoint = 5,
    CheckpointReceipt = 6,
    Artifact = 7,
}

impl TryFrom<u8> for PeerMessageKind {
    type Error = PeerError;

    fn try_from(value: u8) -> PeerResult<Self> {
        match value {
            1 => Ok(Self::LeaseOffer),
            2 => Ok(Self::LeaseRevoke),
            3 => Ok(Self::JobOffer),
            4 => Ok(Self::JobReceipt),
            5 => Ok(Self::Checkpoint),
            6 => Ok(Self::CheckpointReceipt),
            7 => Ok(Self::Artifact),
            _ => Err(PeerError::InvalidFrame),
        }
    }
}

/// A peer message whose sender and recipient come from the paired channel
/// that successfully authenticated and opened the encrypted frame.
/// Callers cannot construct this value from frame metadata or caller-supplied
/// peer IDs.
pub struct AuthenticatedPeerMessage {
    kind: PeerMessageKind,
    source_peer: PeerId,
    destination_peer: PeerId,
    payload: Zeroizing<Vec<u8>>,
}

impl AuthenticatedPeerMessage {
    pub fn kind(&self) -> PeerMessageKind {
        self.kind
    }

    pub fn source_peer(&self) -> PeerId {
        self.source_peer
    }

    pub fn destination_peer(&self) -> PeerId {
        self.destination_peer
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

impl fmt::Debug for AuthenticatedPeerMessage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthenticatedPeerMessage")
            .field("kind", &self.kind)
            .field("source_peer", &self.source_peer)
            .field("destination_peer", &self.destination_peer)
            .field("payload_bytes", &self.payload.len())
            .field("payload", &"[redacted]")
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct EncryptedPeerFrame {
    pub sequence: u64,
    pub kind: PeerMessageKind,
    ciphertext: Vec<u8>,
    tag: [u8; AEAD_TAG_BYTES],
}

impl EncryptedPeerFrame {
    pub fn encode(&self) -> PeerResult<Vec<u8>> {
        if self.ciphertext.len() > MAX_PEER_MESSAGE_BYTES {
            return Err(PeerError::InvalidFrame);
        }
        let mut frame =
            Vec::with_capacity(PEER_FRAME_HEADER_BYTES + self.ciphertext.len() + AEAD_TAG_BYTES);
        frame.extend_from_slice(&PEER_FRAME_VERSION.to_be_bytes());
        frame.extend_from_slice(&self.sequence.to_be_bytes());
        frame.push(self.kind as u8);
        frame.extend_from_slice(
            &u32::try_from(self.ciphertext.len())
                .map_err(|_| PeerError::InvalidFrame)?
                .to_be_bytes(),
        );
        frame.extend_from_slice(&self.ciphertext);
        frame.extend_from_slice(&self.tag);
        Ok(frame)
    }

    pub fn decode(frame: &[u8]) -> PeerResult<Self> {
        if frame.len() < PEER_FRAME_HEADER_BYTES + AEAD_TAG_BYTES
            || frame.len() > PEER_FRAME_HEADER_BYTES + MAX_PEER_MESSAGE_BYTES + AEAD_TAG_BYTES
        {
            return Err(PeerError::InvalidFrame);
        }
        let version = u16::from_be_bytes([frame[0], frame[1]]);
        let sequence = u64::from_be_bytes(frame[2..10].try_into().expect("fixed frame slice"));
        let kind = PeerMessageKind::try_from(frame[10])?;
        let payload_len =
            u32::from_be_bytes(frame[11..15].try_into().expect("fixed frame slice")) as usize;
        if version != PEER_FRAME_VERSION
            || payload_len > MAX_PEER_MESSAGE_BYTES
            || frame.len() != PEER_FRAME_HEADER_BYTES + payload_len + AEAD_TAG_BYTES
        {
            return Err(PeerError::InvalidFrame);
        }
        let payload_end = PEER_FRAME_HEADER_BYTES + payload_len;
        let mut tag = [0_u8; AEAD_TAG_BYTES];
        tag.copy_from_slice(&frame[payload_end..]);
        Ok(Self {
            sequence,
            kind,
            ciphertext: frame[PEER_FRAME_HEADER_BYTES..payload_end].to_vec(),
            tag,
        })
    }
}

impl fmt::Debug for EncryptedPeerFrame {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EncryptedPeerFrame")
            .field("sequence", &self.sequence)
            .field("kind", &self.kind)
            .field("ciphertext_bytes", &self.ciphertext.len())
            .field("ciphertext", &"[redacted]")
            .finish()
    }
}

/// One directional pair of authenticated, replay-ordered message streams.
/// Each side receives opposite send/receive keys derived from the same pairing
/// transcript. Dropping the channel zeroizes both traffic keys.
pub struct PeerSecureChannel {
    send_key: Zeroizing<[u8; 32]>,
    receive_key: Zeroizing<[u8; 32]>,
    local_peer_id: PeerId,
    remote_peer_id: PeerId,
    next_send_sequence: u64,
    next_receive_sequence: u64,
}

impl PeerSecureChannel {
    pub fn local_peer_id(&self) -> PeerId {
        self.local_peer_id
    }

    pub fn remote_peer_id(&self) -> PeerId {
        self.remote_peer_id
    }

    pub fn seal(
        &mut self,
        kind: PeerMessageKind,
        plaintext: &[u8],
    ) -> PeerResult<EncryptedPeerFrame> {
        if plaintext.len() > MAX_PEER_MESSAGE_BYTES {
            return Err(PeerError::InvalidFrame);
        }
        if self.next_send_sequence == u64::MAX {
            return Err(PeerError::SequenceExhausted);
        }
        let sequence = self.next_send_sequence;
        let mut ciphertext = plaintext.to_vec();
        let aad = frame_aad(sequence, kind);
        let nonce = sequence_nonce(sequence);
        let key_bytes: &[u8; 32] = &self.send_key;
        let key: &Key = key_bytes.into();
        let nonce: &XNonce = (&nonce).into();
        let cipher = XChaCha20Poly1305::new(key);
        let tag = cipher
            .encrypt_inout_detached(
                nonce,
                &aad,
                InOutBuf::new(plaintext, &mut ciphertext).map_err(|_| PeerError::InvalidFrame)?,
            )
            .map_err(|_| PeerError::AuthenticationFailed)?;
        let mut tag_bytes = [0_u8; AEAD_TAG_BYTES];
        tag_bytes.copy_from_slice(tag.as_slice());
        self.next_send_sequence += 1;
        Ok(EncryptedPeerFrame {
            sequence,
            kind,
            ciphertext,
            tag: tag_bytes,
        })
    }

    pub fn open(&mut self, frame: &EncryptedPeerFrame) -> PeerResult<Zeroizing<Vec<u8>>> {
        if frame.ciphertext.len() > MAX_PEER_MESSAGE_BYTES {
            return Err(PeerError::InvalidFrame);
        }
        if self.next_receive_sequence == u64::MAX {
            return Err(PeerError::SequenceExhausted);
        }
        if frame.sequence != self.next_receive_sequence {
            return Err(PeerError::ReplayOrReordering);
        }
        let mut plaintext = Zeroizing::new(vec![0; frame.ciphertext.len()]);
        let aad = frame_aad(frame.sequence, frame.kind);
        let nonce = sequence_nonce(frame.sequence);
        let key_bytes: &[u8; 32] = &self.receive_key;
        let key: &Key = key_bytes.into();
        let nonce: &XNonce = (&nonce).into();
        let tag: &Tag = (&frame.tag).into();
        let cipher = XChaCha20Poly1305::new(key);
        let inout = InOutBuf::new(&frame.ciphertext, &mut plaintext)
            .map_err(|_| PeerError::InvalidFrame)?;
        if cipher
            .decrypt_inout_detached(nonce, &aad, inout, tag)
            .is_err()
        {
            plaintext.zeroize();
            return Err(PeerError::AuthenticationFailed);
        }
        self.next_receive_sequence += 1;
        Ok(plaintext)
    }

    /// Open a frame and bind its payload to the peer identities established by
    /// pairing. Use this for routing; it prevents callers from asserting which
    /// peer sent an otherwise valid encrypted frame.
    pub fn open_authenticated(
        &mut self,
        frame: &EncryptedPeerFrame,
    ) -> PeerResult<AuthenticatedPeerMessage> {
        let payload = self.open(frame)?;
        Ok(AuthenticatedPeerMessage {
            kind: frame.kind,
            source_peer: self.remote_peer_id,
            destination_peer: self.local_peer_id,
            payload,
        })
    }
}

impl fmt::Debug for PeerSecureChannel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PeerSecureChannel")
            .field("next_send_sequence", &self.next_send_sequence)
            .field("next_receive_sequence", &self.next_receive_sequence)
            .field("local_peer_id", &self.local_peer_id)
            .field("remote_peer_id", &self.remote_peer_id)
            .field("keys", &"[redacted]")
            .finish()
    }
}

fn offer_body_len() -> usize {
    QR_MAGIC.len() + 2 + 16 + 8 + 32 + 32 + 32 + 32
}

fn random_x25519_keypair() -> PeerResult<(StaticSecret, [u8; 32])> {
    let mut secret_bytes = Zeroizing::new([0_u8; 32]);
    getrandom::fill(secret_bytes.as_mut()).map_err(|_| PeerError::RandomnessUnavailable)?;
    let secret = StaticSecret::from(*secret_bytes);
    let public = X25519PublicKey::from(&secret).to_bytes();
    Ok((secret, public))
}

/// Create a short-lived QR invitation after the inviter's identity has been
/// loaded from its native key store.
pub fn create_pairing_invitation(
    identity: &DeviceIdentity,
    now_unix_seconds: u64,
    lifetime: Duration,
) -> PeerResult<(PairingQr, PendingInitiator)> {
    let lifetime_seconds = lifetime.as_secs();
    if !(MIN_INVITATION_TTL_SECONDS..=MAX_INVITATION_TTL_SECONDS).contains(&lifetime_seconds) {
        return Err(PeerError::InvalidInput);
    }
    let expires_at_unix_seconds = now_unix_seconds
        .checked_add(lifetime_seconds)
        .ok_or(PeerError::InvalidInput)?;
    let mut pairing_id = [0_u8; 16];
    let mut invitation_secret = Zeroizing::new([0_u8; 32]);
    let mut inviter_nonce = [0_u8; 32];
    getrandom::fill(&mut pairing_id).map_err(|_| PeerError::RandomnessUnavailable)?;
    getrandom::fill(invitation_secret.as_mut()).map_err(|_| PeerError::RandomnessUnavailable)?;
    getrandom::fill(&mut inviter_nonce).map_err(|_| PeerError::RandomnessUnavailable)?;
    let (ephemeral_secret, inviter_ephemeral_key) = random_x25519_keypair()?;
    let mut offer = PairingOffer {
        pairing_id,
        expires_at_unix_seconds,
        inviter_verifying_key: identity.signing_key.verifying_key().to_bytes(),
        inviter_ephemeral_key,
        inviter_nonce,
        invitation_commitment: Sha256::digest(invitation_secret.as_ref()).into(),
        signature: [0; 64],
    };
    offer.signature = identity.signing_key.sign(&offer.signed_bytes()).to_bytes();
    let qr = PairingQr {
        offer: offer.clone(),
        invitation_secret: Zeroizing::new(*invitation_secret),
    };
    let pending = PendingInitiator {
        offer,
        invitation_secret,
        ephemeral_secret,
    };
    Ok((qr, pending))
}

fn validate_offer(
    offer: &PairingOffer,
    invitation_secret: &[u8; 32],
    now_unix_seconds: u64,
) -> PeerResult<()> {
    if now_unix_seconds >= offer.expires_at_unix_seconds {
        return Err(PeerError::Expired);
    }
    let observed_commitment: [u8; 32] = Sha256::digest(invitation_secret).into();
    if !bool::from(observed_commitment.ct_eq(&offer.invitation_commitment)) {
        return Err(PeerError::AuthenticationFailed);
    }
    let key = VerifyingKey::from_bytes(&offer.inviter_verifying_key)
        .map_err(|_| PeerError::AuthenticationFailed)?;
    key.verify_strict(
        &offer.signed_bytes(),
        &Signature::from_bytes(&offer.signature),
    )
    .map_err(|_| PeerError::AuthenticationFailed)
}

fn validate_response(offer: &PairingOffer, response: &PairingResponse) -> PeerResult<()> {
    if response.pairing_id != offer.pairing_id || response.offer_digest != offer.digest() {
        return Err(PeerError::AuthenticationFailed);
    }
    let key = VerifyingKey::from_bytes(&response.responder_verifying_key)
        .map_err(|_| PeerError::AuthenticationFailed)?;
    if response.responder_verifying_key == offer.inviter_verifying_key {
        return Err(PeerError::AuthenticationFailed);
    }
    key.verify_strict(
        &response.signing_bytes(),
        &Signature::from_bytes(&response.signature),
    )
    .map_err(|_| PeerError::AuthenticationFailed)
}

fn validate_confirmation(
    offer: &PairingOffer,
    response: &PairingResponse,
    confirmation: &PairingConfirmation,
) -> PeerResult<()> {
    if confirmation.pairing_id != offer.pairing_id
        || confirmation.response_digest != response.digest()
    {
        return Err(PeerError::AuthenticationFailed);
    }
    let key = VerifyingKey::from_bytes(&offer.inviter_verifying_key)
        .map_err(|_| PeerError::AuthenticationFailed)?;
    key.verify_strict(
        &PairingConfirmation::signing_bytes(confirmation.pairing_id, confirmation.response_digest),
        &Signature::from_bytes(&confirmation.inviter_signature),
    )
    .map_err(|_| PeerError::AuthenticationFailed)
}

fn authentication_code(
    offer: &PairingOffer,
    response: &PairingResponse,
    invitation_secret: &[u8; 32],
) -> String {
    let mut hash = Sha256::new();
    hash.update(TRANSCRIPT_DOMAIN);
    hash.update(offer.body());
    hash.update(offer.signature);
    hash.update(response.body());
    hash.update(response.signature);
    hash.update(invitation_secret);
    let digest = hash.finalize();
    let mut code = String::with_capacity(12);
    for byte in &digest[..6] {
        use fmt::Write as _;
        write!(&mut code, "{byte:02x}").expect("writing to a String cannot fail");
    }
    code
}

fn derive_pairing_outcome(
    local_ephemeral_secret: StaticSecret,
    remote_ephemeral_public: [u8; 32],
    invitation_secret: &[u8; 32],
    offer: &PairingOffer,
    response: &PairingResponse,
    confirmation: &PairingConfirmation,
    local_is_initiator: bool,
) -> PeerResult<PairingOutcome> {
    let remote = X25519PublicKey::from(remote_ephemeral_public);
    let shared = local_ephemeral_secret.diffie_hellman(&remote);
    if shared.as_bytes().iter().all(|byte| *byte == 0) {
        return Err(PeerError::AuthenticationFailed);
    }
    let mut transcript = Zeroizing::new(Vec::with_capacity(
        offer_body_len() + 64 + response.body().len() + 64 + 54 + 64 + 32,
    ));
    transcript.extend_from_slice(TRANSCRIPT_DOMAIN);
    transcript.extend_from_slice(&offer.body());
    transcript.extend_from_slice(&offer.signature);
    transcript.extend_from_slice(&response.body());
    transcript.extend_from_slice(&response.signature);
    transcript.extend_from_slice(&PairingConfirmation::body(
        confirmation.pairing_id,
        confirmation.response_digest,
    ));
    transcript.extend_from_slice(&confirmation.inviter_signature);
    transcript.extend_from_slice(invitation_secret);
    let transcript_digest: [u8; 32] = Sha256::digest(&transcript).into();
    let mut salt_hash = Sha256::new();
    salt_hash.update(KDF_SALT_DOMAIN);
    salt_hash.update(transcript_digest);
    let salt: [u8; 32] = salt_hash.finalize().into();
    let mut input_key_material = Zeroizing::new([0_u8; 64]);
    input_key_material[..32].copy_from_slice(shared.as_bytes());
    input_key_material[32..].copy_from_slice(invitation_secret);
    let pseudorandom_key = Zeroizing::new(hmac_sha256(&salt, input_key_material.as_ref()));
    let root_info = concat_bytes(&[KDF_ROOT_INFO, &transcript_digest, &[1]]);
    let root = Zeroizing::new(hmac_sha256(pseudorandom_key.as_ref(), &root_info));
    let initiator_to_responder = Zeroizing::new(hmac_sha256(
        root.as_ref(),
        &concat_bytes(&[KDF_INITIATOR_TO_RESPONDER, &transcript_digest, &[1]]),
    ));
    let responder_to_initiator = Zeroizing::new(hmac_sha256(
        root.as_ref(),
        &concat_bytes(&[KDF_RESPONDER_TO_INITIATOR, &transcript_digest, &[1]]),
    ));

    let (send_key, receive_key) = if local_is_initiator {
        (*initiator_to_responder, *responder_to_initiator)
    } else {
        (*responder_to_initiator, *initiator_to_responder)
    };
    let local_peer_id = if local_is_initiator {
        peer_id_for(&offer.inviter_verifying_key)
    } else {
        peer_id_for(&response.responder_verifying_key)
    };
    let peer_key = if local_is_initiator {
        response.responder_verifying_key
    } else {
        offer.inviter_verifying_key
    };
    let remote_peer_id = peer_id_for(&peer_key);
    Ok(PairingOutcome {
        local_peer_id,
        trusted_peer: PublicPeerIdentity {
            peer_id: peer_id_for(&peer_key),
            verifying_key: peer_key,
        },
        trust_material: PeerTrustMaterial {
            secret: Zeroizing::new(*root),
        },
        channel: PeerSecureChannel {
            send_key: Zeroizing::new(send_key),
            receive_key: Zeroizing::new(receive_key),
            local_peer_id,
            remote_peer_id,
            next_send_sequence: 0,
            next_receive_sequence: 0,
        },
    })
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts keys of every length");
    mac.update(message);
    let result = mac.finalize().into_bytes();
    let mut output = [0_u8; 32];
    output.copy_from_slice(&result);
    output
}

fn concat_bytes(parts: &[&[u8]]) -> Vec<u8> {
    let capacity = parts.iter().map(|part| part.len()).sum();
    let mut bytes = Vec::with_capacity(capacity);
    for part in parts {
        bytes.extend_from_slice(part);
    }
    bytes
}

fn frame_aad(sequence: u64, kind: PeerMessageKind) -> [u8; 11] {
    let mut aad = [0_u8; 11];
    aad[..2].copy_from_slice(&PEER_FRAME_VERSION.to_be_bytes());
    aad[2..10].copy_from_slice(&sequence.to_be_bytes());
    aad[10] = kind as u8;
    aad
}

fn sequence_nonce(sequence: u64) -> [u8; 24] {
    let mut nonce = [0_u8; 24];
    nonce[16..].copy_from_slice(&sequence.to_be_bytes());
    nonce
}

fn peer_id_for(verifying_key: &[u8; 32]) -> PeerId {
    let mut hash = Sha256::new();
    hash.update(b"sage:peer:id:v1\0");
    hash.update(verifying_key);
    PeerId(hash.finalize().into())
}

fn constant_time_string_eq(expected: &str, observed: &str) -> bool {
    expected.len() == observed.len() && bool::from(expected.as_bytes().ct_eq(observed.as_bytes()))
}

fn read_array<const N: usize>(bytes: &[u8], offset: &mut usize) -> PeerResult<[u8; N]> {
    let end = offset.checked_add(N).ok_or(PeerError::InvalidInput)?;
    let slice = bytes.get(*offset..end).ok_or(PeerError::InvalidInput)?;
    let mut value = [0_u8; N];
    value.copy_from_slice(slice);
    *offset = end;
    Ok(value)
}

fn read_u16(bytes: &[u8], offset: &mut usize) -> PeerResult<u16> {
    Ok(u16::from_be_bytes(read_array(bytes, offset)?))
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> PeerResult<u64> {
    Ok(u64::from_be_bytes(read_array(bytes, offset)?))
}

mod compute;
mod compute_transport;
mod discovery;
mod dispatch_journal;
mod lease;
mod pairing_transport;
mod placement;
mod task_handoff;
mod transport;
mod worker;
pub use compute::{
    AssembledJob, AuthorizedPartitionRequest, ComputePlan, ComputeTransferAssembler,
    ComputeTransferEncoder, ComputeTransferExpectation, ComputeTransferKind, ComputeUnitSpec,
    PartitionCheckpoint, PartitionRequest, PartitionResult, PreparedPartitionRequest,
    ReceivedPartitionRequest, UnitInput, UnitOutput, VerifiedPartition,
};
pub use compute_transport::{
    ComputeTransportError, receive_partition_request, receive_verified_partition_result,
    send_partition_request, send_partition_result,
};
pub use discovery::*;
pub use dispatch_journal::PeerDispatchJournal;
pub use lease::{
    DelegatedJobKind, DisclosureScope, LeaseId, LeasePermit, LeaseRequest, LeaseScope,
    LeaseSettlement, PeerLease, PeerLeaseBook, PeerLeaseGrant, PeerLeaseQuotas, PermittedPeerJob,
};
pub use pairing_transport::{
    PairedPeerStream, PairingTransportError, PairingTransportResult, pair_as_initiator,
    pair_as_responder,
};
pub use placement::{
    CAPACITY_SAMPLE_WINDOW, CapacityObservation, CapacitySampler, ComputeResourceSnapshot,
    LeasedPeerRoute, LinkObservation, LocalHostPressure, LocalHostPressureSampler,
    MAX_PLACEMENT_PAYLOAD_BYTES, PlacementDecision, PlacementPolicy, PlacementReason,
    PlacementWorkload, choose_compute_placement,
};
pub use task_handoff::{
    ArtifactHandoffAssembler, ArtifactHandoffEncoder, ArtifactHandoffExpectation,
    ArtifactHandoffPayload, TaskHandoffAssembler, TaskHandoffEncoder, TaskHandoffExpectation,
    TaskHandoffPayload, TaskHandoffReceipt,
};
pub use transport::{
    MAX_PEER_STREAM_TIMEOUT, PeerSecureReader, PeerSecureWriter, PeerTransportError,
    split_secure_stream,
};
pub use worker::{
    MandelbrotTileSpec, byte_histogram_v1_resource_id, execute_authorized_partition,
    execute_authorized_partition_with_control, mandelbrot_rgb_tile_v1_resource_id,
    verify_builtin_output, verify_builtin_output_with_control,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn identities() -> (DeviceIdentity, DeviceIdentity) {
        (
            DeviceIdentity::from_seed([7; 32]),
            DeviceIdentity::from_seed([19; 32]),
        )
    }

    fn paired() -> (PairingOutcome, PairingOutcome) {
        let (inviter, responder) = identities();
        let (qr, pending_inviter) =
            create_pairing_invitation(&inviter, 1_800_000_000, Duration::from_secs(120))
                .expect("create pairing invite");
        let decoded_qr = PairingQr::from_payload(&qr.to_payload()).expect("parse QR payload");
        let (response, pending_responder) = decoded_qr
            .respond_after_local_confirmation(&responder, 1_800_000_010)
            .expect("locally approved response");
        let response = PairingResponse::decode(&response.encode()).expect("decode response");
        let inviter_code = pending_inviter
            .authentication_code(&response)
            .expect("compute SAS");
        assert_eq!(inviter_code, pending_responder.authentication_code());
        let (confirmation, inviter_outcome) = pending_inviter
            .confirm_after_local_confirmation(&inviter, &response, &inviter_code, 1_800_000_011)
            .expect("inviter confirmation");
        let confirmation =
            PairingConfirmation::decode(&confirmation.encode()).expect("decode confirmation");
        let responder_outcome = pending_responder
            .finish_after_local_confirmation(&confirmation, &inviter_code, 1_800_000_012)
            .expect("responder confirmation");
        (inviter_outcome, responder_outcome)
    }

    #[test]
    fn pairing_binds_both_identities_and_yields_matching_trust_root() {
        let (initiator, responder) = paired();
        assert_eq!(initiator.trusted_peer.peer_id(), responder.local_peer_id);
        assert_eq!(responder.trusted_peer.peer_id(), initiator.local_peer_id);
        assert_eq!(
            initiator.trust_material.copy_for_key_store(),
            responder.trust_material.copy_for_key_store()
        );
        assert_eq!(
            initiator.local_peer_id,
            DeviceIdentity::from_seed([7; 32]).peer_id()
        );
    }

    #[test]
    fn encrypted_peer_frames_authenticate_kind_and_reject_replay() {
        let (mut initiator, mut responder) = paired();
        let frame = initiator
            .channel
            .seal(PeerMessageKind::JobOffer, b"bounded render input")
            .expect("seal bounded request");
        let wire = frame.encode().expect("encode encrypted frame");
        let decoded = EncryptedPeerFrame::decode(&wire).expect("decode encrypted frame");
        assert_eq!(
            responder
                .channel
                .open(&decoded)
                .expect("open request")
                .as_slice(),
            b"bounded render input"
        );
        assert_eq!(
            responder.channel.open(&decoded),
            Err(PeerError::ReplayOrReordering)
        );

        let response = responder
            .channel
            .seal(PeerMessageKind::JobReceipt, b"verified result")
            .expect("seal result");
        assert_eq!(
            initiator
                .channel
                .open(&response)
                .expect("open result")
                .as_slice(),
            b"verified result"
        );
    }

    #[test]
    fn typed_compute_job_and_result_round_trip_through_live_lease_and_secure_channel() {
        let (mut requester, mut donor) = paired();
        let donor_identity = DeviceIdentity::from_seed([19; 32]);
        let input = vec![0x5a; MAX_PEER_MESSAGE_BYTES + 4096];
        let input_digest: [u8; 32] = Sha256::digest(&input).into();
        let plan = ComputePlan::new(
            DelegatedJobKind::BatchAnalysis,
            [0x42; 32],
            vec![
                ComputeUnitSpec::new(0, input_digest, input.len() as u32, input.len() as u32)
                    .unwrap(),
            ],
            1,
        )
        .unwrap();
        let lease = PeerLease::issue(
            &donor_identity,
            requester.local_peer_id,
            1_800_000_013,
            120,
            [LeaseScope {
                job: plan.job(),
                resource_id: [0x42; 32],
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 1,
                maximum_jobs: 1,
                maximum_input_bytes_per_job: input.len() as u64,
                maximum_output_bytes_per_job: input.len() as u64,
                maximum_total_input_bytes: input.len() as u64,
                maximum_total_output_bytes: input.len() as u64,
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
                1_800_000_014,
            )
            .unwrap();
        let mut request_transfer = request.into_transfer().unwrap();
        assert!(request_transfer.chunk_count() > 1);
        let mut request_assembler = ComputeTransferAssembler::new(
            plan.transfer_expectation(ComputeTransferKind::PartitionRequest, 0)
                .unwrap(),
        );
        let mut request_payload = None;
        while let Some(chunk) = request_transfer.next_frame().unwrap() {
            let encoded_frame = requester
                .channel
                .seal(request_transfer.message_kind(), &chunk)
                .unwrap()
                .encode()
                .unwrap();
            let received_frame = EncryptedPeerFrame::decode(&encoded_frame).unwrap();
            let received_payload = donor.channel.open(&received_frame).unwrap();
            if let Some(assembled) = request_assembler
                .push(received_frame.kind, &received_payload)
                .unwrap()
            {
                assert!(request_payload.replace(assembled).is_none());
            }
        }
        let request_payload = request_payload.expect("all request chunks assemble");
        let received =
            ReceivedPartitionRequest::decode(&request_payload, requester.local_peer_id).unwrap();
        let prepared = plan.prepare_partition_request(received).unwrap();

        let mut lease_book = PeerLeaseBook::new(donor.local_peer_id);
        let lease_id = lease_book.insert(lease).unwrap();
        let permit = lease_book
            .reserve(
                lease_id,
                requester.local_peer_id,
                prepared.lease_request(),
                1_800_000_014,
            )
            .unwrap();
        let permitted = lease_book
            .authorize_dispatch(&permit, 1_800_000_014)
            .unwrap();
        let mut journal_nonce = [0; 16];
        getrandom::fill(&mut journal_nonce).unwrap();
        let journal_path = std::env::temp_dir().join(format!(
            "sage-peer-e2e-journal-{}-{:032x}.bin",
            std::process::id(),
            u128::from_be_bytes(journal_nonce)
        ));
        let mut dispatch_journal =
            PeerDispatchJournal::open(&journal_path, Zeroizing::new([0x7c; 32])).unwrap();
        let authorized = prepared
            .authorize_durable(permitted, &mut dispatch_journal)
            .unwrap();
        assert!(dispatch_journal.contains(plan.job_id(), 0));
        assert_eq!(authorized.inputs()[0].bytes(), input.as_slice());

        let result = PartitionResult::new(
            *plan.job_id(),
            *plan.plan_digest(),
            0,
            vec![UnitOutput::new(0, input_digest, input.clone()).unwrap()],
        )
        .unwrap();
        let mut result_transfer = result.into_transfer().unwrap();
        assert!(result_transfer.chunk_count() > 1);
        let mut result_assembler = ComputeTransferAssembler::new(
            plan.transfer_expectation(ComputeTransferKind::PartitionResult, 0)
                .unwrap(),
        );
        let mut result_payload = None;
        while let Some(chunk) = result_transfer.next_frame().unwrap() {
            let encoded_frame = donor
                .channel
                .seal(result_transfer.message_kind(), &chunk)
                .unwrap()
                .encode()
                .unwrap();
            let received_frame = EncryptedPeerFrame::decode(&encoded_frame).unwrap();
            let received_payload = requester.channel.open(&received_frame).unwrap();
            if let Some(assembled) = result_assembler
                .push(received_frame.kind, &received_payload)
                .unwrap()
            {
                assert!(result_payload.replace(assembled).is_none());
            }
        }
        let result_payload = result_payload.expect("all result chunks assemble");
        let result = PartitionResult::decode(&result_payload).unwrap();
        let verified = plan
            .verify_partition_result(0, donor.local_peer_id, result, |spec, output| {
                Sha256::digest(output).as_slice() == spec.input_digest()
            })
            .unwrap();
        let checkpoint = verified.checkpoint();
        assert_eq!(checkpoint.outputs().len(), 1);
        assert_eq!(checkpoint.outputs()[0].bytes(), input.as_slice());
        let settled = lease_book.settle(permit, input.len() as u64).unwrap();
        assert!(settled.dispatched);
        assert_eq!(settled.output_bytes, input.len() as u64);
        drop(dispatch_journal);
        std::fs::remove_file(journal_path).unwrap();
    }

    #[test]
    fn signed_lease_offer_is_authenticated_by_paired_identity_over_secure_channel() {
        let (mut donor, mut requester) = paired();
        let donor_identity = DeviceIdentity::from_seed([7; 32]);
        let grant = PeerLease::issue(
            &donor_identity,
            requester.local_peer_id,
            1_800_000_013,
            120,
            [LeaseScope {
                job: DelegatedJobKind::FrameRender,
                resource_id: [0x73; 32],
                disclosure: DisclosureScope::ExplicitJobInputAndResult,
            }],
            PeerLeaseQuotas {
                maximum_concurrent_jobs: 2,
                maximum_jobs: 20,
                maximum_input_bytes_per_job: 1_024 * 1_024,
                maximum_output_bytes_per_job: 2_048 * 1_024,
                maximum_total_input_bytes: 8 * 1_024 * 1_024,
                maximum_total_output_bytes: 16 * 1_024 * 1_024,
            },
        )
        .unwrap()
        .grant();
        let frame = donor
            .channel
            .seal(PeerMessageKind::LeaseOffer, &grant.encode())
            .unwrap();
        let wire = frame.encode().unwrap();
        let decoded_frame = EncryptedPeerFrame::decode(&wire).unwrap();
        let payload = requester.channel.open(&decoded_frame).unwrap();
        let decoded_grant = PeerLeaseGrant::decode(&payload).unwrap();
        decoded_grant
            .verify(
                &requester.trusted_peer,
                requester.local_peer_id,
                1_800_000_014,
            )
            .unwrap();
        assert_eq!(decoded_grant.owner(), donor.local_peer_id);
        assert_eq!(decoded_grant.peer(), requester.local_peer_id);
    }

    #[test]
    fn failed_authentication_does_not_consume_receive_sequence() {
        let (mut initiator, mut responder) = paired();
        let valid = initiator
            .channel
            .seal(PeerMessageKind::Checkpoint, b"checkpoint")
            .expect("seal checkpoint");
        let mut tampered = valid.clone();
        tampered.kind = PeerMessageKind::JobOffer;
        assert_eq!(
            responder.channel.open(&tampered),
            Err(PeerError::AuthenticationFailed)
        );
        assert_eq!(
            responder
                .channel
                .open(&valid)
                .expect("valid retry")
                .as_slice(),
            b"checkpoint"
        );
    }

    #[test]
    fn pairing_rejects_mismatched_short_code_and_expired_invitation() {
        let (inviter, responder) = identities();
        let (qr, pending) = create_pairing_invitation(&inviter, 100, Duration::from_secs(30))
            .expect("create shortest valid invitation");
        let (response, _) = qr
            .respond_after_local_confirmation(&responder, 101)
            .expect("respond before expiry");
        assert_eq!(
            pending
                .confirm_after_local_confirmation(&inviter, &response, "000000000000", 102)
                .unwrap_err(),
            PeerError::AuthenticationCodeMismatch
        );
        assert_eq!(
            qr.respond_after_local_confirmation(&responder, 130).err(),
            Some(PeerError::Expired)
        );
        assert!(create_pairing_invitation(&inviter, 100, Duration::from_secs(29)).is_err());
        assert!(create_pairing_invitation(&inviter, 100, Duration::from_secs(301)).is_err());
    }

    #[test]
    fn qr_and_frame_decoders_reject_malformed_or_oversized_inputs() {
        assert!(PairingQr::from_payload("not a QR payload").is_err());
        assert!(PairingQr::from_payload(&"A".repeat(513)).is_err());
        assert!(EncryptedPeerFrame::decode(&[0; 30]).is_err());

        let (mut initiator, _) = paired();
        let frame = initiator
            .channel
            .seal(PeerMessageKind::LeaseOffer, b"x")
            .expect("seal lease");
        let mut wire = frame.encode().expect("encode frame");
        wire[0] = 0xff;
        assert_eq!(
            EncryptedPeerFrame::decode(&wire),
            Err(PeerError::InvalidFrame)
        );
    }

    #[test]
    fn pairing_wire_messages_are_bounded_and_keep_signature_validation() {
        let (inviter, responder) = identities();
        let (qr, pending_inviter) =
            create_pairing_invitation(&inviter, 1_800_000_000, Duration::from_secs(120)).unwrap();
        let (response, pending_responder) = qr
            .respond_after_local_confirmation(&responder, 1_800_000_001)
            .unwrap();
        let encoded_response = response.encode();
        let decoded_response = PairingResponse::decode(&encoded_response).unwrap();
        assert_eq!(decoded_response.responder(), responder.public_identity());
        let code = pending_inviter
            .authentication_code(&decoded_response)
            .unwrap();
        let (confirmation, _) = pending_inviter
            .confirm_after_local_confirmation(&inviter, &decoded_response, &code, 1_800_000_002)
            .unwrap();
        let encoded_confirmation = confirmation.encode();
        assert!(PairingConfirmation::decode(&encoded_confirmation).is_ok());
        assert!(PairingConfirmation::decode(&encoded_confirmation[..117]).is_err());

        let ack = PairingAcknowledgement::sign_after_local_confirmation(
            &responder,
            &decoded_response,
            &confirmation,
        )
        .unwrap();
        let encoded_ack = ack.encode();
        let decoded_ack = PairingAcknowledgement::decode(&encoded_ack).unwrap();
        decoded_ack
            .verify(&decoded_response, &confirmation)
            .unwrap();
        assert_eq!(
            PairingAcknowledgement::sign_after_local_confirmation(
                &inviter,
                &decoded_response,
                &confirmation,
            )
            .unwrap_err(),
            PeerError::AuthenticationFailed
        );
        let mut tampered_ack = encoded_ack;
        tampered_ack[60] ^= 1;
        PairingAcknowledgement::decode(&tampered_ack)
            .unwrap()
            .verify(&decoded_response, &confirmation)
            .expect_err("changed transcript bytes invalidate the response signature");
        assert_eq!(pending_responder.authentication_code(), code);

        let mut wrong_version = encoded_response;
        wrong_version[5] = 0xff;
        assert_eq!(
            PairingResponse::decode(&wrong_version).unwrap_err(),
            PeerError::InvalidInput
        );
    }
}
