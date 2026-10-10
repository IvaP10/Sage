//! Bounded full-duplex transport for an already paired secure peer channel.
//!
//! This module frames encrypted messages over any Tokio byte stream. It does
//! not discover devices, pair identities, grant leases, or dispatch jobs; the
//! supplied channel must come from a completed local-confirmation pairing.

use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadHalf, WriteHalf},
    sync::Mutex,
    time::timeout,
};
use zeroize::Zeroize;

use super::{
    AuthenticatedPeerMessage, EncryptedPeerFrame, MAX_PEER_MESSAGE_BYTES, PeerError,
    PeerMessageKind, PeerSecureChannel,
};

const OUTER_LENGTH_BYTES: usize = std::mem::size_of::<u32>();
const PEER_FRAME_HEADER_BYTES: usize = 15;
const PEER_FRAME_TAG_BYTES: usize = 16;
const MIN_ENCRYPTED_FRAME_BYTES: usize = PEER_FRAME_HEADER_BYTES + PEER_FRAME_TAG_BYTES;

/// Maximum bytes carried after the outer TCP-style length prefix.
pub const MAX_ENCRYPTED_PEER_FRAME_BYTES: usize =
    PEER_FRAME_HEADER_BYTES + MAX_PEER_MESSAGE_BYTES + PEER_FRAME_TAG_BYTES;

/// Peer I/O deadlines must be positive and no longer than this bound.
pub const MAX_PEER_STREAM_TIMEOUT: Duration = Duration::from_secs(60);
const MIN_PEER_STREAM_TIMEOUT: Duration = Duration::from_millis(1);

#[derive(Debug, Error)]
pub enum PeerTransportError {
    #[error(transparent)]
    Peer(#[from] PeerError),
    #[error("peer stream I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("peer stream operation exceeded its deadline")]
    TimedOut,
    #[error("peer stream timeout is outside its supported bound")]
    InvalidTimeout,
    #[error("peer stream was interrupted and cannot be reused")]
    Poisoned,
}

/// A serialized writer half. Callers should close the connection after any
/// error; a cancelled or partial frame poisons this half and consumes its
/// secure-channel sequence number.
pub struct PeerSecureWriter<W> {
    writer: W,
    channel: Arc<Mutex<PeerSecureChannel>>,
    poisoned: Arc<AtomicBool>,
}

impl<W: AsyncWrite + Unpin> PeerSecureWriter<W> {
    /// Seal and send one authenticated message with a bounded deadline.
    pub async fn send(
        &mut self,
        kind: PeerMessageKind,
        plaintext: &[u8],
        deadline: Duration,
    ) -> Result<(), PeerTransportError> {
        validate_deadline(deadline)?;
        if plaintext.len() > MAX_PEER_MESSAGE_BYTES {
            return Err(PeerError::InvalidFrame.into());
        }
        self.ensure_usable()?;
        let mut guard = PoisonOnDrop::new(Arc::clone(&self.poisoned));
        let writer = &mut self.writer;
        let channel = Arc::clone(&self.channel);
        let poisoned = Arc::clone(&self.poisoned);
        timeout(deadline, async move {
            if poisoned.load(Ordering::Acquire) {
                return Err(PeerTransportError::Poisoned);
            }
            let frame = channel.lock().await.seal(kind, plaintext)?;
            let encoded = frame.encode()?;
            let encoded_length =
                u32::try_from(encoded.len()).map_err(|_| PeerError::InvalidFrame)?;
            let mut length = encoded_length.to_be_bytes();
            writer.write_all(&length).await?;
            length.zeroize();
            writer.write_all(&encoded).await?;
            writer.flush().await?;
            Ok::<(), PeerTransportError>(())
        })
        .await
        .map_err(|_| PeerTransportError::TimedOut)??;

        guard.disarm();
        Ok(())
    }

    fn ensure_usable(&self) -> Result<(), PeerTransportError> {
        if self.poisoned.load(Ordering::Acquire) {
            Err(PeerTransportError::Poisoned)
        } else {
            Ok(())
        }
    }
}

/// A serialized reader half. A timeout, cancellation, malformed frame, or
/// authentication failure poisons this half because the next byte boundary
/// or secure sequence can no longer be trusted.
pub struct PeerSecureReader<R> {
    reader: R,
    channel: Arc<Mutex<PeerSecureChannel>>,
    poisoned: Arc<AtomicBool>,
}

impl<R: AsyncRead + Unpin> PeerSecureReader<R> {
    /// Receive one authenticated message. `None` means the peer cleanly
    /// closed the stream between messages.
    pub async fn receive(
        &mut self,
        deadline: Duration,
    ) -> Result<Option<AuthenticatedPeerMessage>, PeerTransportError> {
        validate_deadline(deadline)?;
        self.ensure_usable()?;
        let mut guard = PoisonOnDrop::new(Arc::clone(&self.poisoned));
        let reader = &mut self.reader;
        let channel = Arc::clone(&self.channel);
        let poisoned = Arc::clone(&self.poisoned);
        let message = timeout(deadline, async move {
            let frame = read_encrypted_frame(reader).await?;
            let Some(frame) = frame else {
                return Ok::<_, PeerTransportError>(None);
            };
            if poisoned.load(Ordering::Acquire) {
                return Err(PeerTransportError::Poisoned);
            }
            let message = channel.lock().await.open_authenticated(&frame)?;
            Ok(Some(message))
        })
        .await
        .map_err(|_| PeerTransportError::TimedOut)??;
        let Some(message) = message else {
            // A clean EOF is terminal for both halves of this stream.
            return Ok(None);
        };
        guard.disarm();
        Ok(Some(message))
    }

    fn ensure_usable(&self) -> Result<(), PeerTransportError> {
        if self.poisoned.load(Ordering::Acquire) {
            Err(PeerTransportError::Poisoned)
        } else {
            Ok(())
        }
    }
}

/// Split a byte stream around a channel created only after both devices
/// confirmed the same pairing transcript. The halves share one channel owner;
/// channel locks are held only while sealing/opening, never during socket I/O.
pub fn split_secure_stream<S>(
    stream: S,
    channel: PeerSecureChannel,
) -> (
    PeerSecureWriter<WriteHalf<S>>,
    PeerSecureReader<ReadHalf<S>>,
)
where
    S: AsyncRead + AsyncWrite,
{
    let (reader, writer) = tokio::io::split(stream);
    let channel = Arc::new(Mutex::new(channel));
    let poisoned = Arc::new(AtomicBool::new(false));
    (
        PeerSecureWriter {
            writer,
            channel: Arc::clone(&channel),
            poisoned: Arc::clone(&poisoned),
        },
        PeerSecureReader {
            reader,
            channel,
            poisoned,
        },
    )
}

async fn read_encrypted_frame<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<Option<EncryptedPeerFrame>, PeerTransportError> {
    let mut length = [0_u8; OUTER_LENGTH_BYTES];
    if reader.read(&mut length[..1]).await? == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut length[1..]).await?;
    let frame_bytes = u32::from_be_bytes(length) as usize;
    length.zeroize();
    if !(MIN_ENCRYPTED_FRAME_BYTES..=MAX_ENCRYPTED_PEER_FRAME_BYTES).contains(&frame_bytes) {
        return Err(PeerError::InvalidFrame.into());
    }

    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(frame_bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::OutOfMemory, error))?;
    encoded.resize(frame_bytes, 0);
    reader.read_exact(&mut encoded).await?;
    EncryptedPeerFrame::decode(&encoded)
        .map(Some)
        .map_err(PeerTransportError::from)
}

fn validate_deadline(deadline: Duration) -> Result<(), PeerTransportError> {
    if !(MIN_PEER_STREAM_TIMEOUT..=MAX_PEER_STREAM_TIMEOUT).contains(&deadline) {
        return Err(PeerTransportError::InvalidTimeout);
    }
    Ok(())
}

struct PoisonOnDrop {
    poisoned: Arc<AtomicBool>,
    armed: bool,
}

impl PoisonOnDrop {
    fn new(poisoned: Arc<AtomicBool>) -> Self {
        Self {
            poisoned,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PoisonOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.poisoned.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceIdentity;
    use std::time::Duration;
    use tokio::net::{TcpListener, TcpStream};

    fn paired_channels() -> (PeerSecureChannel, PeerSecureChannel) {
        let inviter = DeviceIdentity::from_seed([7; 32]);
        let responder = DeviceIdentity::from_seed([19; 32]);
        let (qr, pending_inviter) = super::super::create_pairing_invitation(
            &inviter,
            1_800_000_000,
            Duration::from_secs(120),
        )
        .expect("create pairing invite");
        let decoded_qr =
            super::super::PairingQr::from_payload(&qr.to_payload()).expect("parse QR payload");
        let (response, pending_responder) = decoded_qr
            .respond_after_local_confirmation(&responder, 1_800_000_010)
            .expect("locally approved response");
        let response = super::super::PairingResponse::decode(&response.encode())
            .expect("decode pairing response");
        let code = pending_inviter
            .authentication_code(&response)
            .expect("compute authentication code");
        assert_eq!(code, pending_responder.authentication_code());
        let (confirmation, inviter_outcome) = pending_inviter
            .confirm_after_local_confirmation(&inviter, &response, &code, 1_800_000_011)
            .expect("confirm inviter");
        let confirmation = super::super::PairingConfirmation::decode(&confirmation.encode())
            .expect("decode pairing confirmation");
        let responder_outcome = pending_responder
            .finish_after_local_confirmation(&confirmation, &code, 1_800_000_012)
            .expect("confirm responder");
        (inviter_outcome.channel, responder_outcome.channel)
    }

    #[tokio::test]
    async fn paired_tcp_stream_carries_full_duplex_encrypted_messages() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback test listener");
        let address = listener.local_addr().expect("read listener address");
        let connect = TcpStream::connect(address);
        let accept = listener.accept();
        let (client, accepted) = tokio::join!(connect, accept);
        let client = client.expect("connect loopback client");
        let (server, _) = accepted.expect("accept loopback client");
        let (client_channel, server_channel) = paired_channels();
        let server_local_peer = server_channel.local_peer_id();
        let server_remote_peer = server_channel.remote_peer_id();
        let (mut client_writer, mut client_reader) = split_secure_stream(client, client_channel);
        let (mut server_writer, mut server_reader) = split_secure_stream(server, server_channel);
        let deadline = Duration::from_secs(2);

        let (client_sent, server_sent, at_server, at_client) = tokio::join!(
            client_writer.send(PeerMessageKind::JobOffer, b"bounded request", deadline),
            server_writer.send(PeerMessageKind::JobReceipt, b"verified result", deadline),
            server_reader.receive(deadline),
            client_reader.receive(deadline),
        );
        client_sent.expect("send encrypted client request");
        server_sent.expect("send encrypted server receipt");
        let server_message = at_server
            .expect("receive client message")
            .expect("client stream remains open");
        let client_message = at_client
            .expect("receive server message")
            .expect("server stream remains open");
        assert_eq!(server_message.kind(), PeerMessageKind::JobOffer);
        assert_eq!(server_message.source_peer(), server_remote_peer);
        assert_eq!(server_message.destination_peer(), server_local_peer);
        assert_eq!(server_message.payload(), b"bounded request");
        assert_eq!(client_message.kind(), PeerMessageKind::JobReceipt);
        assert_eq!(client_message.payload(), b"verified result");
    }

    #[tokio::test]
    async fn receive_timeout_poisons_the_partial_stream_and_enforces_bounds() {
        let (stream, _peer) = tokio::io::duplex(128);
        let (channel, _) = paired_channels();
        let (mut writer, mut reader) = split_secure_stream(stream, channel);
        let error = reader
            .receive(Duration::from_millis(5))
            .await
            .expect_err("idle reader hits its deadline");
        assert!(matches!(error, PeerTransportError::TimedOut));
        assert!(matches!(
            reader.receive(Duration::from_secs(1)).await,
            Err(PeerTransportError::Poisoned)
        ));
        assert!(matches!(
            writer
                .send(
                    PeerMessageKind::JobOffer,
                    b"request",
                    Duration::from_secs(1)
                )
                .await,
            Err(PeerTransportError::Poisoned)
        ));
        assert!(matches!(
            reader.receive(Duration::from_millis(0)).await,
            Err(PeerTransportError::InvalidTimeout)
        ));
    }

    #[tokio::test]
    async fn partial_write_timeout_poisons_both_stream_halves() {
        let (stream, _peer) = tokio::io::duplex(1);
        let (channel, _) = paired_channels();
        let (mut writer, mut reader) = split_secure_stream(stream, channel);
        let error = writer
            .send(
                PeerMessageKind::JobOffer,
                b"request that will not fit without a reader",
                Duration::from_millis(10),
            )
            .await
            .expect_err("backpressured writer hits its deadline");
        assert!(matches!(error, PeerTransportError::TimedOut));
        assert!(matches!(
            reader.receive(Duration::from_secs(1)).await,
            Err(PeerTransportError::Poisoned)
        ));
        assert!(matches!(
            writer
                .send(PeerMessageKind::JobOffer, b"again", Duration::from_secs(1))
                .await,
            Err(PeerTransportError::Poisoned)
        ));
    }

    #[tokio::test]
    async fn oversized_outer_frame_is_rejected_before_payload_allocation() {
        let (stream, mut peer) = tokio::io::duplex(128);
        let (channel, _) = paired_channels();
        let (_, mut reader) = split_secure_stream(stream, channel);
        peer.write_all(&u32::MAX.to_be_bytes())
            .await
            .expect("write oversized length");
        assert!(matches!(
            reader.receive(Duration::from_secs(1)).await,
            Err(PeerTransportError::Peer(PeerError::InvalidFrame))
        ));
        assert!(matches!(
            reader.receive(Duration::from_secs(1)).await,
            Err(PeerTransportError::Poisoned)
        ));
    }
}
