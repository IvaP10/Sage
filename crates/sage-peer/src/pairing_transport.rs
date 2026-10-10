//! Bounded network exchange for the two-sided QR pairing protocol.
//!
//! The caller owns socket discovery and UI. Both devices independently
//! approve signed identities and compare the short authentication code;
//! denial drops the stream before a trusted channel is returned.

use std::{
    future::Future,
    io,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    time::timeout,
};

use crate::{
    DeviceIdentity, PairingAcknowledgement, PairingConfirmation, PairingOutcome, PairingQr,
    PairingResponse, PeerError, PeerResult, PendingInitiator, PublicPeerIdentity,
};

const LENGTH_BYTES: usize = 2;
const MAX_PAIRING_FRAME_BYTES: usize = 256;
const MIN_PAIRING_IO_TIMEOUT: Duration = Duration::from_millis(1);
const MAX_PAIRING_IO_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Error)]
pub enum PairingTransportError {
    #[error(transparent)]
    Peer(#[from] PeerError),
    #[error("pairing stream I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("pairing stream operation exceeded its deadline")]
    TimedOut,
    #[error("pairing I/O timeout is outside its supported bound")]
    InvalidTimeout,
    #[error("local user declined peer pairing")]
    Rejected,
    #[error("pairing frame exceeds its fixed bound")]
    OversizedFrame,
}

pub type PairingTransportResult<T> = Result<T, PairingTransportError>;

/// A successfully paired connection. `outcome.channel` can be passed to
/// `split_secure_stream`; `outcome.trust_material` must be persisted through
/// the caller's native key store.
pub struct PairedPeerStream<S> {
    pub outcome: PairingOutcome,
    pub stream: S,
}

impl<S> PairedPeerStream<S> {
    pub fn into_parts(self) -> (PairingOutcome, S) {
        (self.outcome, self.stream)
    }
}

/// Complete the inviter side on an already connected stream. The callback
/// must show the responder identity and authentication code, and return true
/// only after this device's user explicitly compares and accepts them.
pub async fn pair_as_initiator<S, F, Fut>(
    mut stream: S,
    pending: PendingInitiator,
    identity: &DeviceIdentity,
    io_timeout: Duration,
    confirm_peer: F,
) -> PairingTransportResult<PairedPeerStream<S>>
where
    S: AsyncRead + AsyncWrite + Unpin,
    F: FnOnce(PublicPeerIdentity, String) -> Fut,
    Fut: Future<Output = bool>,
{
    validate_timeout(io_timeout)?;
    let response = PairingResponse::decode(&receive_frame(&mut stream, io_timeout).await?)?;
    let peer = response.responder();
    let authentication_code = pending.authentication_code(&response)?;
    if !confirm_peer(peer, authentication_code.clone()).await {
        return Err(PairingTransportError::Rejected);
    }

    let (confirmation, outcome) = pending.confirm_after_local_confirmation(
        identity,
        &response,
        &authentication_code,
        unix_time_seconds()?,
    )?;
    send_frame(&mut stream, &confirmation.encode(), io_timeout).await?;
    let acknowledgement =
        PairingAcknowledgement::decode(&receive_frame(&mut stream, io_timeout).await?)?;
    acknowledgement.verify(&response, &confirmation)?;
    Ok(PairedPeerStream { outcome, stream })
}

/// Complete the scanned-device side. It validates the QR signature and
/// expiry, asks for local approval of the inviter, then sends the signed
/// response. A second callback confirms the matching code before accepting
/// the inviter's signed final confirmation.
pub async fn pair_as_responder<S, F, Fut, G, FutG>(
    mut stream: S,
    qr: PairingQr,
    identity: &DeviceIdentity,
    io_timeout: Duration,
    confirm_inviter: F,
    confirm_code: G,
) -> PairingTransportResult<PairedPeerStream<S>>
where
    S: AsyncRead + AsyncWrite + Unpin,
    F: FnOnce(PublicPeerIdentity) -> Fut,
    Fut: Future<Output = bool>,
    G: FnOnce(PublicPeerIdentity, String) -> FutG,
    FutG: Future<Output = bool>,
{
    validate_timeout(io_timeout)?;
    let inviter = qr.verified_inviter(unix_time_seconds()?)?;
    if !confirm_inviter(inviter).await {
        return Err(PairingTransportError::Rejected);
    }

    let (response, pending) =
        qr.respond_after_local_confirmation(identity, unix_time_seconds()?)?;
    send_frame(&mut stream, &response.encode(), io_timeout).await?;
    let authentication_code = pending.authentication_code();
    if !confirm_code(inviter, authentication_code.clone()).await {
        return Err(PairingTransportError::Rejected);
    }

    let confirmation = PairingConfirmation::decode(&receive_frame(&mut stream, io_timeout).await?)?;
    let outcome = pending.finish_after_local_confirmation(
        &confirmation,
        &authentication_code,
        unix_time_seconds()?,
    )?;
    let acknowledgement =
        PairingAcknowledgement::sign_after_local_confirmation(identity, &response, &confirmation)?;
    send_frame(&mut stream, &acknowledgement.encode(), io_timeout).await?;
    Ok(PairedPeerStream { outcome, stream })
}

fn validate_timeout(duration: Duration) -> PairingTransportResult<()> {
    if !(MIN_PAIRING_IO_TIMEOUT..=MAX_PAIRING_IO_TIMEOUT).contains(&duration) {
        return Err(PairingTransportError::InvalidTimeout);
    }
    Ok(())
}

async fn send_frame<S: AsyncWrite + Unpin>(
    stream: &mut S,
    payload: &[u8],
    io_timeout: Duration,
) -> PairingTransportResult<()> {
    if payload.len() > MAX_PAIRING_FRAME_BYTES {
        return Err(PairingTransportError::OversizedFrame);
    }
    let length = u16::try_from(payload.len()).map_err(|_| PairingTransportError::OversizedFrame)?;
    timeout(io_timeout, async {
        stream.write_all(&length.to_be_bytes()).await?;
        stream.write_all(payload).await?;
        stream.flush().await
    })
    .await
    .map_err(|_| PairingTransportError::TimedOut)??;
    Ok(())
}

async fn receive_frame<S: AsyncRead + Unpin>(
    stream: &mut S,
    io_timeout: Duration,
) -> PairingTransportResult<Vec<u8>> {
    let mut frame = Vec::new();
    timeout(io_timeout, async {
        let mut length_bytes = [0_u8; LENGTH_BYTES];
        stream.read_exact(&mut length_bytes).await?;
        let length = usize::from(u16::from_be_bytes(length_bytes));
        if length > MAX_PAIRING_FRAME_BYTES {
            return Err(PairingTransportError::OversizedFrame);
        }
        frame
            .try_reserve_exact(length)
            .map_err(|error| io::Error::new(io::ErrorKind::OutOfMemory, error))?;
        frame.resize(length, 0);
        stream.read_exact(&mut frame).await?;
        Ok::<(), PairingTransportError>(())
    })
    .await
    .map_err(|_| PairingTransportError::TimedOut)??;
    Ok(frame)
}

fn unix_time_seconds() -> PeerResult<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PeerError::DiscoveryClock)?
        .as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DeviceIdentity, PairingQr, PeerMessageKind, create_pairing_invitation, split_secure_stream,
    };
    use tokio::net::{TcpListener, TcpStream};

    fn unix_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    async fn tcp_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let connect = TcpStream::connect(address);
        let accept = listener.accept();
        let (client, accepted) = tokio::join!(connect, accept);
        (client.unwrap(), accepted.unwrap().0)
    }

    fn invitation() -> (DeviceIdentity, DeviceIdentity, PairingQr, PendingInitiator) {
        let inviter = DeviceIdentity::from_seed([0x81; 32]);
        let responder = DeviceIdentity::from_seed([0x92; 32]);
        let (qr, pending) =
            create_pairing_invitation(&inviter, unix_now(), Duration::from_secs(120)).unwrap();
        let qr = PairingQr::from_payload(&qr.to_payload()).unwrap();
        (inviter, responder, qr, pending)
    }

    #[tokio::test]
    async fn both_users_confirm_over_tcp_before_secure_channel_is_returned() {
        let (inviter, responder, qr, pending) = invitation();
        let (initiator_stream, responder_stream) = tcp_pair().await;
        let expected_initiator_peer = responder.peer_id();
        let expected_responder_peer = inviter.peer_id();
        let (initiator_result, responder_result) = tokio::join!(
            pair_as_initiator(
                initiator_stream,
                pending,
                &inviter,
                Duration::from_secs(2),
                move |peer, code| async move {
                    peer.peer_id() == expected_initiator_peer && code.len() == 12
                },
            ),
            pair_as_responder(
                responder_stream,
                qr,
                &responder,
                Duration::from_secs(2),
                move |peer| async move { peer.peer_id() == expected_responder_peer },
                |_peer, code| async move { code.len() == 12 },
            ),
        );
        let initiator = initiator_result.unwrap();
        let responder = responder_result.unwrap();
        assert_eq!(
            initiator.outcome.trusted_peer.peer_id(),
            responder.outcome.local_peer_id
        );
        assert_eq!(
            responder.outcome.trusted_peer.peer_id(),
            initiator.outcome.local_peer_id
        );

        let (mut initiator_writer, mut initiator_reader) =
            split_secure_stream(initiator.stream, initiator.outcome.channel);
        let (mut responder_writer, mut responder_reader) =
            split_secure_stream(responder.stream, responder.outcome.channel);
        let (sent, received) = tokio::join!(
            initiator_writer.send(
                PeerMessageKind::Artifact,
                b"paired channel proof",
                Duration::from_secs(2),
            ),
            responder_reader.receive(Duration::from_secs(2)),
        );
        sent.unwrap();
        let received = received.unwrap().unwrap();
        assert_eq!(received.source_peer(), initiator.outcome.local_peer_id);
        assert_eq!(received.payload(), b"paired channel proof");

        let (sent, received) = tokio::join!(
            responder_writer.send(
                PeerMessageKind::Artifact,
                b"return proof",
                Duration::from_secs(2),
            ),
            initiator_reader.receive(Duration::from_secs(2)),
        );
        sent.unwrap();
        assert_eq!(received.unwrap().unwrap().payload(), b"return proof");
    }

    #[tokio::test]
    async fn inviter_denial_never_creates_or_returns_a_trusted_channel() {
        let (inviter, responder, qr, pending) = invitation();
        let (initiator_stream, responder_stream) = tcp_pair().await;
        let initiator = pair_as_initiator(
            initiator_stream,
            pending,
            &inviter,
            Duration::from_secs(2),
            |_peer, _code| async { false },
        );
        let responder = pair_as_responder(
            responder_stream,
            qr,
            &responder,
            Duration::from_secs(2),
            |_peer| async { true },
            |_peer, _code| async { true },
        );
        let (initiator, responder) = tokio::join!(initiator, responder);
        assert!(matches!(initiator, Err(PairingTransportError::Rejected)));
        assert!(responder.is_err());
    }

    #[tokio::test]
    async fn oversized_pairing_frame_is_rejected_before_payload_allocation() {
        let (mut sender, mut receiver) = tokio::io::duplex(16);
        sender.write_all(&u16::MAX.to_be_bytes()).await.unwrap();
        assert!(matches!(
            receive_frame(&mut receiver, Duration::from_secs(1)).await,
            Err(PairingTransportError::OversizedFrame)
        ));
    }

    #[tokio::test]
    async fn stalled_pairing_io_is_bounded() {
        let (mut sender, mut receiver) = tokio::io::duplex(1);
        let error = send_frame(
            &mut sender,
            &[0x55; MAX_PAIRING_FRAME_BYTES],
            Duration::from_millis(10),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, PairingTransportError::TimedOut));
        assert!(matches!(
            receive_frame(&mut receiver, Duration::from_millis(10)).await,
            Err(PairingTransportError::TimedOut)
        ));
    }
}
