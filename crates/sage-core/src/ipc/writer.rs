//! One transport owner keeps frame order while inbound control remains available.
//! Budgets include the frame currently being written, not only channel entries.
use std::sync::Arc;

use prost::Message;
use sage_protocol::{MAX_FRAME_BYTES, PROTOCOL_VERSION, sage::ipc::v2 as wire};
use tokio::io::AsyncWrite;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant, timeout_at};

use crate::{CoreError, CoreResult, SageCore};

const REGULAR_FRAMES: usize = 64;
const CONTROL_FRAMES: usize = 16;
const REGULAR_BYTES: usize = 2 * (MAX_FRAME_BYTES + 4);
const CONTROL_BYTES: usize = MAX_FRAME_BYTES + 4;
pub(super) const WRITE_DEADLINE: Duration = Duration::from_secs(5);

#[derive(Clone, Copy)]
pub(super) enum Lane {
    Regular,
    Control,
}

struct Budget {
    frames: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
}

impl Budget {
    fn new(frames: usize, bytes: usize) -> Self {
        Self {
            frames: Arc::new(Semaphore::new(frames)),
            bytes: Arc::new(Semaphore::new(bytes)),
        }
    }
}

struct QueuedFrame {
    frame: wire::Frame,
    deadline: Instant,
    _frame_permit: OwnedSemaphorePermit,
    _byte_permit: OwnedSemaphorePermit,
}

pub(super) struct OutboundWriter {
    sender: mpsc::Sender<QueuedFrame>,
    regular: Budget,
    control: Budget,
    sequence: u64,
    deadline: Duration,
    task: JoinHandle<CoreResult<()>>,
}

impl OutboundWriter {
    pub fn new<W: AsyncWrite + Unpin + Send + 'static>(
        writer: W,
        sequence: u64,
        core: Arc<SageCore>,
        session_id: String,
    ) -> Self {
        Self::with_deadline(writer, sequence, core, session_id, WRITE_DEADLINE)
    }

    fn with_deadline<W: AsyncWrite + Unpin + Send + 'static>(
        mut writer: W,
        sequence: u64,
        core: Arc<SageCore>,
        session_id: String,
        deadline: Duration,
    ) -> Self {
        let (sender, mut receiver) = mpsc::channel::<QueuedFrame>(REGULAR_FRAMES + CONTROL_FRAMES);
        let task = tokio::spawn(async move {
            while let Some(queued) = receiver.recv().await {
                // The operation can be cancelled while waiting for the peer to
                // consume earlier frames. Never send an already retired request.
                if let Some(wire::frame::Payload::AdapterRequest(request)) = &queued.frame.payload
                    && !core.adapters.is_pending(&session_id, &request.request_id)
                {
                    if let Some(late) = core
                        .adapters
                        .cancelled_before_send(&session_id, &request.request_id)?
                    {
                        core.record_late_adapter_result(late).await?;
                    }
                    continue;
                }
                timeout_at(
                    queued.deadline,
                    super::codec::write_frame(&mut writer, &queued.frame),
                )
                .await
                .map_err(|_| {
                    CoreError::Timeout("IPC peer stopped consuming outbound frames".into())
                })??;
            }
            Ok(())
        });
        Self {
            sender,
            regular: Budget::new(REGULAR_FRAMES, REGULAR_BYTES),
            control: Budget::new(CONTROL_FRAMES, CONTROL_BYTES),
            sequence,
            deadline,
            task,
        }
    }

    /// Never wait for socket capacity on the authenticated admission loop.
    /// Control has reserved capacity; FIFO ordering is preserved across lanes.
    pub fn enqueue(&mut self, payload: wire::frame::Payload, lane: Lane) -> CoreResult<()> {
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| CoreError::Protocol("IPC sequence exhausted".into()))?;
        let frame = wire::Frame {
            protocol_version: PROTOCOL_VERSION,
            sequence,
            payload: Some(payload),
        };
        let length = frame.encoded_len();
        if length == 0 || length > MAX_FRAME_BYTES {
            return Err(CoreError::Protocol(
                "encoded frame is outside the accepted size range".into(),
            ));
        }
        let budget = match lane {
            Lane::Regular => &self.regular,
            Lane::Control => &self.control,
        };
        let full = || {
            CoreError::Busy("IPC outbound capacity exhausted; reconnect to refresh state".into())
        };
        let frame_permit = budget
            .frames
            .clone()
            .try_acquire_owned()
            .map_err(|_| full())?;
        let byte_permit = budget
            .bytes
            .clone()
            .try_acquire_many_owned((length + 4) as u32)
            .map_err(|_| full())?;
        self.sender
            .try_send(QueuedFrame {
                frame,
                deadline: Instant::now() + self.deadline,
                _frame_permit: frame_permit,
                _byte_permit: byte_permit,
            })
            .map_err(|_| full())?;
        self.sequence = sequence;
        Ok(())
    }

    pub async fn finished(&mut self) -> CoreResult<()> {
        (&mut self.task)
            .await
            .map_err(|error| CoreError::Protocol(format!("IPC writer failed: {error}")))?
    }
}

impl Drop for OutboundWriter {
    fn drop(&mut self) {
        // A partial frame cannot be resumed by another writer. Closing the
        // transport also releases all queued byte and count reservations.
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::CoreConfig, model::UnconfiguredModelProvider};
    use tokio::time::timeout;

    fn fixture() -> (tempfile::TempDir, Arc<SageCore>) {
        let data = tempfile::tempdir().unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        (data, core)
    }

    fn pong(value: u64) -> wire::frame::Payload {
        wire::frame::Payload::Pong(wire::Pong { value })
    }

    #[tokio::test]
    async fn frame_count_is_bounded_and_control_capacity_preserves_order() {
        let (_data, core) = fixture();
        let (mut client, server) = tokio::io::duplex(1);
        let mut writer = OutboundWriter::new(server, 2, core, "test".into());
        for index in 0..REGULAR_FRAMES {
            writer.enqueue(pong(index as u64), Lane::Regular).unwrap();
        }
        assert!(writer.enqueue(pong(999), Lane::Regular).is_err());
        writer.enqueue(pong(999), Lane::Control).unwrap();
        timeout(Duration::from_secs(2), async {
            for index in 0..=REGULAR_FRAMES {
                let frame = super::super::codec::read_frame(&mut client).await.unwrap();
                assert_eq!(frame.sequence, index as u64 + 3);
                let Some(wire::frame::Payload::Pong(pong)) = frame.payload else {
                    panic!("Expected pong")
                };
                assert_eq!(
                    pong.value,
                    if index == REGULAR_FRAMES {
                        999
                    } else {
                        index as u64
                    }
                );
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn byte_budget_includes_in_flight_frame_and_timeout_releases_capacity() {
        let (_data, core) = fixture();
        let (_client, server) = tokio::io::duplex(1);
        let mut writer = OutboundWriter::with_deadline(
            server,
            2,
            core,
            "test".into(),
            Duration::from_millis(50),
        );
        let large = || {
            wire::frame::Payload::CoreEvent(wire::CoreEvent {
                event: Some(wire::core_event::Event::Error(wire::ErrorEvent {
                    message: "x".repeat(MAX_FRAME_BYTES - 128),
                    ..Default::default()
                })),
                ..Default::default()
            })
        };
        writer.enqueue(large(), Lane::Regular).unwrap();
        tokio::task::yield_now().await;
        writer.enqueue(large(), Lane::Regular).unwrap();
        assert!(writer.enqueue(large(), Lane::Regular).is_err());
        writer.enqueue(pong(1), Lane::Control).unwrap();
        assert!(matches!(
            timeout(Duration::from_secs(1), writer.finished())
                .await
                .unwrap(),
            Err(CoreError::Timeout(_))
        ));
        assert_eq!(writer.regular.bytes.available_permits(), REGULAR_BYTES);
        assert_eq!(writer.regular.frames.available_permits(), REGULAR_FRAMES);
        assert_eq!(writer.control.frames.available_permits(), CONTROL_FRAMES);
    }
}
