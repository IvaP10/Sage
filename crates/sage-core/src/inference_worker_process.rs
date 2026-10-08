//! Owns the restricted first-party inference helper process.
//!
//! This is only a lifecycle and protocol boundary. The current helper reports
//! `model_not_admitted`; it cannot generate tokens and must not be selected as
//! a model provider until a signed package and generation protocol are ready.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, bail};
use serde::Deserialize;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::time::timeout;

const PROTOCOL_VERSION: u16 = 1;
const MAX_FRAME_BYTES: usize = 4 * 1024;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(2);

/// A Core-owned worker whose identity and readiness were checked at startup.
/// Dropping it terminates the child through Tokio's `kill_on_drop` behavior.
pub(super) struct OwnedInferenceWorker {
    _child: Child,
    _stdin: tokio::process::ChildStdin,
    _stdout: tokio::process::ChildStdout,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerResponse {
    protocol_version: u16,
    kind: String,
    challenge: String,
    process_id: u32,
    readiness: String,
}

/// Locates and starts only the worker shipped beside the running Sage Core.
/// Source builds may opt into an explicit executable path for local testing.
pub(super) async fn start_bundled() -> anyhow::Result<Option<OwnedInferenceWorker>> {
    let Some(path) = bundled_path()? else {
        return Ok(None);
    };
    launch(&path).await.map(Some)
}

fn bundled_path() -> anyhow::Result<Option<PathBuf>> {
    #[cfg(all(target_os = "macos", debug_assertions))]
    if let Some(path) = std::env::var_os("SAGE_INFERENCE_WORKER_EXECUTABLE") {
        return canonical_executable(PathBuf::from(path)).map(Some);
    }

    #[cfg(target_os = "macos")]
    {
        let core = std::env::current_exe().context("locate running Sage Core executable")?;
        let helpers = core
            .parent()
            .context("Sage Core executable has no containing directory")?;
        let candidate =
            helpers.join("SageInferenceWorker.app/Contents/MacOS/sage-inference-worker");
        if !candidate.is_file() {
            return Ok(None);
        }
        canonical_executable(candidate).map(Some)
    }

    #[cfg(not(target_os = "macos"))]
    {
        Ok(None)
    }
}

fn canonical_executable(path: PathBuf) -> anyhow::Result<PathBuf> {
    if !path.is_absolute() {
        bail!("inference worker path must be absolute");
    }
    let path = path
        .canonicalize()
        .context("resolve first-party inference worker executable")?;
    if !path.is_file() {
        bail!("first-party inference worker is not a regular file");
    }
    Ok(path)
}

async fn launch(path: &Path) -> anyhow::Result<OwnedInferenceWorker> {
    let challenge = random_challenge()?;
    let mut child = Command::new(path)
        .env_clear()
        .current_dir("/")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("start bundled inference worker at {}", path.display()))?;
    let pid = child
        .id()
        .context("inference worker has no process identity")?;
    let Some(mut stdin) = child.stdin.take() else {
        terminate(&mut child).await;
        bail!("inference worker stdin pipe was unavailable");
    };
    let Some(mut stdout) = child.stdout.take() else {
        terminate(&mut child).await;
        bail!("inference worker stdout pipe was unavailable");
    };

    let result = timeout(
        HANDSHAKE_TIMEOUT,
        exchange_hello(&mut stdin, &mut stdout, &challenge, pid),
    )
    .await;
    match result {
        Ok(Ok(())) => Ok(OwnedInferenceWorker {
            _child: child,
            _stdin: stdin,
            _stdout: stdout,
        }),
        Ok(Err(error)) => {
            terminate(&mut child).await;
            Err(error).context("inference worker handshake failed")
        }
        Err(_) => {
            terminate(&mut child).await;
            bail!("inference worker handshake exceeded its 2 second deadline")
        }
    }
}

async fn terminate(child: &mut Child) {
    let _ = child.kill().await;
    let _ = timeout(Duration::from_secs(1), child.wait()).await;
}

async fn exchange_hello<W, R>(
    writer: &mut W,
    reader: &mut R,
    challenge: &str,
    expected_pid: u32,
) -> anyhow::Result<()>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let request = serde_json::json!({
        "protocol_version": PROTOCOL_VERSION,
        "kind": "hello",
        "challenge": challenge,
    });
    write_frame(writer, &serde_json::to_vec(&request)?).await?;
    let response = serde_json::from_slice::<WorkerResponse>(&read_frame(reader).await?)?;
    validate_response(response, challenge, expected_pid)
}

fn validate_response(
    response: WorkerResponse,
    challenge: &str,
    expected_pid: u32,
) -> anyhow::Result<()> {
    if response.protocol_version != PROTOCOL_VERSION
        || response.kind != "hello"
        || response.challenge != challenge
        || response.process_id != expected_pid
        || response.readiness != "model_not_admitted"
    {
        bail!("inference worker returned an unexpected identity or readiness state");
    }
    Ok(())
}

async fn write_frame<W: AsyncWrite + Unpin>(writer: &mut W, bytes: &[u8]) -> anyhow::Result<()> {
    if bytes.is_empty() || bytes.len() > MAX_FRAME_BYTES {
        bail!("inference worker request exceeded the frame bound");
    }
    let length = u32::try_from(bytes.len()).context("inference worker frame length overflow")?;
    writer.write_all(&length.to_be_bytes()).await?;
    writer.write_all(bytes).await?;
    writer.flush().await?;
    Ok(())
}

async fn read_frame<R: AsyncRead + Unpin>(reader: &mut R) -> anyhow::Result<Vec<u8>> {
    let mut length = [0_u8; 4];
    reader.read_exact(&mut length).await?;
    let length = u32::from_be_bytes(length) as usize;
    if length == 0 || length > MAX_FRAME_BYTES {
        bail!("inference worker response exceeded the frame bound");
    }
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes).await?;
    Ok(bytes)
}

fn random_challenge() -> anyhow::Result<String> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes).map_err(|error| {
        anyhow::anyhow!("generate inference worker handshake challenge: {error}")
    })?;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    #[test]
    fn random_challenge_is_32_bytes_encoded_as_lowercase_hex() {
        let challenge = random_challenge().unwrap();
        assert_eq!(challenge.len(), 64);
        assert!(
            challenge
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
    }

    #[test]
    fn worker_response_must_match_challenge_pid_and_closed_readiness() {
        let valid = WorkerResponse {
            protocol_version: PROTOCOL_VERSION,
            kind: "hello".into(),
            challenge: "ab".repeat(32),
            process_id: 42,
            readiness: "model_not_admitted".into(),
        };
        assert!(validate_response(valid, &"ab".repeat(32), 42).is_ok());

        let wrong_pid = WorkerResponse {
            protocol_version: PROTOCOL_VERSION,
            kind: "hello".into(),
            challenge: "ab".repeat(32),
            process_id: 43,
            readiness: "model_not_admitted".into(),
        };
        assert!(validate_response(wrong_pid, &"ab".repeat(32), 42).is_err());
    }

    #[tokio::test]
    async fn handshake_rejects_oversized_response_before_allocating_payload() {
        let (mut writer, mut reader) = duplex(8);
        writer
            .write_all(&((MAX_FRAME_BYTES + 1) as u32).to_be_bytes())
            .await
            .unwrap();
        assert!(read_frame(&mut reader).await.is_err());
    }

    #[tokio::test]
    #[ignore = "requires SAGE_TEST_INFERENCE_WORKER_EXECUTABLE pointing to a built worker"]
    async fn launches_the_configured_worker_and_completes_the_bounded_handshake() {
        let path = std::env::var_os("SAGE_TEST_INFERENCE_WORKER_EXECUTABLE")
            .expect("test worker executable path");
        let OwnedInferenceWorker {
            mut _child,
            _stdin,
            _stdout,
        } = launch(Path::new(&path)).await.unwrap();
        drop(_stdout);
        drop(_stdin);
        let status = timeout(Duration::from_secs(2), _child.wait())
            .await
            .expect("worker exits after Core closes its request pipe")
            .unwrap();
        assert!(status.success());
    }
}
