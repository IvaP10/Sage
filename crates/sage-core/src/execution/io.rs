//! A bounded lane for filesystem reads/preparation. Kernel calls can wait for
//! network volumes or OS privacy consent; they must not occupy control workers.
use crate::{CoreError, CoreResult};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

pub(super) const MAX_READERS: usize = 4;
const MAX_WRITERS: usize = 2;
const READ_DEADLINE: Duration = Duration::from_secs(3);
const STREAM_READ_DEADLINE: Duration = Duration::from_secs(90);
const WRITE_DEADLINE: Duration = Duration::from_secs(90);
const READ_TIMEOUT_MESSAGE: &str = "File access did not respond within three seconds. Select the folder again or check Sage's system file permissions and connected drives.";
const STREAM_READ_TIMEOUT_MESSAGE: &str = "File stream did not settle within 90 seconds. Check the source, destination, or connected drive.";
const WRITE_TIMEOUT_MESSAGE: &str =
    "File write did not settle within 90 seconds. Check the destination or connected drive.";

struct CancelOperation(Arc<AtomicBool>);
impl Drop for CancelOperation {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

pub(crate) async fn bounded_read<T: Send + 'static>(
    read: impl FnOnce(&AtomicBool) -> CoreResult<T> + Send + 'static,
) -> CoreResult<T> {
    bounded_read_with_deadline(READ_DEADLINE, read).await
}

/// A stream can be deliberately backpressured by its consumer, so it shares
/// the bounded read lane but uses the same 90-second ceiling as writes.
pub(crate) async fn bounded_stream_read<T: Send + 'static>(
    read: impl FnOnce(&AtomicBool) -> CoreResult<T> + Send + 'static,
) -> CoreResult<T> {
    bounded_read_with_deadline(STREAM_READ_DEADLINE, read).await
}

async fn bounded_read_with_deadline<T: Send + 'static>(
    deadline: Duration,
    read: impl FnOnce(&AtomicBool) -> CoreResult<T> + Send + 'static,
) -> CoreResult<T> {
    static READERS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    let pool = READERS
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(MAX_READERS)))
        .clone();
    let timeout_message = if deadline == STREAM_READ_DEADLINE {
        STREAM_READ_TIMEOUT_MESSAGE
    } else {
        READ_TIMEOUT_MESSAGE
    };
    run(pool, deadline, timeout_message, read).await
}

pub(crate) async fn bounded_write<T: Send + 'static>(
    write: impl FnOnce(&AtomicBool) -> CoreResult<T> + Send + 'static,
) -> CoreResult<T> {
    static WRITERS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    let pool = WRITERS
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(MAX_WRITERS)))
        .clone();
    run(pool, WRITE_DEADLINE, WRITE_TIMEOUT_MESSAGE, write).await
}

async fn run<T: Send + 'static>(
    pool: Arc<tokio::sync::Semaphore>,
    deadline: Duration,
    timeout_message: &'static str,
    operation: impl FnOnce(&AtomicBool) -> CoreResult<T> + Send + 'static,
) -> CoreResult<T> {
    tokio::time::timeout(deadline, async {
        let permit = pool
            .acquire_owned()
            .await
            .map_err(|_| CoreError::Cancelled)?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let guard = CancelOperation(cancelled.clone());
        let result = tokio::task::spawn_blocking(move || {
            // The kernel operation keeps its slot even after its owner stops.
            // Repeated cancellations cannot create unbounded blocking threads.
            let _permit = permit;
            if cancelled.load(Ordering::Acquire) {
                return Err(CoreError::Cancelled);
            }
            let result = operation(&cancelled);
            if cancelled.load(Ordering::Acquire) {
                return Err(CoreError::Cancelled);
            }
            result
        })
        .await
        .map_err(|_| CoreError::ExecutionFailed("File worker exited unexpectedly".into()))?;
        drop(guard);
        result
    })
    .await
    .map_err(|_| CoreError::Timeout(timeout_message.into()))?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn stalled_kernel_read_times_out_without_releasing_its_worker_budget() {
        let pool = Arc::new(tokio::sync::Semaphore::new(1));
        let (release, blocked) = std::sync::mpsc::channel();
        let (started, entered) = tokio::sync::oneshot::channel();
        let read_pool = pool.clone();
        let operation = tokio::spawn(async move {
            run(
                read_pool,
                Duration::from_millis(30),
                "test read timed out",
                move |_| {
                    let _ = started.send(());
                    blocked.recv().unwrap();
                    Ok(1)
                },
            )
            .await
        });
        entered.await.unwrap();
        assert!(matches!(
            operation.await.unwrap(),
            Err(CoreError::Timeout(_))
        ));
        assert_eq!(pool.available_permits(), 0);
        assert!(matches!(
            run(
                pool.clone(),
                Duration::from_millis(5),
                "test read timed out",
                |_| Ok(2),
            )
            .await,
            Err(CoreError::Timeout(_))
        ));
        release.send(()).unwrap();
        let _permit = tokio::time::timeout(Duration::from_secs(1), pool.acquire())
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn backpressured_stream_read_uses_longer_deadline_than_single_read() {
        let ordinary_read = bounded_read(|_| {
            std::thread::sleep(Duration::from_millis(3_100));
            Ok(7)
        })
        .await;
        assert!(matches!(ordinary_read, Err(CoreError::Timeout(_))));

        let streamed_read = bounded_stream_read(|_| {
            std::thread::sleep(Duration::from_millis(3_100));
            Ok(7)
        })
        .await;
        assert_eq!(streamed_read.unwrap(), 7);
    }
}
