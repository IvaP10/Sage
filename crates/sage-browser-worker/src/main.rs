#![forbid(unsafe_code)]
//! Native messaging host. Browser output is restricted to pending adapter
//! replies. No browser payload can become a trusted UI command.
use sage_core::config::{CoreConfig, IpcEndpoint};
use sage_core::ipc::{authentication_proof, read_frame, write_frame};
use sage_core::secrets::{OsSecretStore, load_browser_ipc_secret};
use sage_protocol::{PROTOCOL_VERSION, sage::ipc::v2 as wire};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("Sage browser connection failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let origin = std::env::args()
        .nth(1)
        .ok_or("Native messaging origin missing")?;
    let extension = origin
        .strip_prefix("chrome-extension://")
        .and_then(|s| s.strip_suffix('/'))
        .ok_or("Invalid browser extension origin")?;
    if extension.len() != 32 || !extension.bytes().all(|b| (b'a'..=b'p').contains(&b)) {
        return Err("Invalid browser extension identifier".into());
    }
    let config = CoreConfig::platform_default()?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(config.data_dir.join("browser-host.json"))?)?;
    if !manifest["allowed_origins"]
        .as_array()
        .is_some_and(|origins| {
            origins
                .iter()
                .any(|allowed| allowed.as_str() == Some(&origin))
        })
    {
        return Err("Extension is not paired with Sage".into());
    }
    match config.ipc_endpoint {
        #[cfg(unix)]
        IpcEndpoint::UnixSocket(path) => relay(tokio::net::UnixStream::connect(path).await?).await,
        #[cfg(windows)]
        IpcEndpoint::NamedPipe(name) => {
            relay(tokio::net::windows::named_pipe::ClientOptions::new().open(name)?).await
        }
    }
}

async fn relay<S: AsyncRead + AsyncWrite + Unpin>(
    mut stream: S,
) -> Result<(), Box<dyn std::error::Error>> {
    let challenge = read_frame(&mut stream).await?;
    let Some(wire::frame::Payload::ServerChallenge(challenge)) = challenge.payload else {
        return Err("Missing core challenge".into());
    };
    let secret =
        load_browser_ipc_secret(&CoreConfig::platform_default()?.data_dir, &OsSecretStore)?;
    let mut nonce = vec![0_u8; 32];
    getrandom::fill(&mut nonce).map_err(|_| "Randomness unavailable")?;
    let version = env!("CARGO_PKG_VERSION");
    let proof = authentication_proof(
        secret.expose(),
        &challenge.nonce,
        &nonce,
        PROTOCOL_VERSION,
        wire::ClientKind::Browser as i32,
        version,
    )?;
    write_frame(
        &mut stream,
        &frame(wire::frame::Payload::ClientAuthenticate(
            wire::ClientAuthenticate {
                client_kind: wire::ClientKind::Browser as i32,
                client_version: version.into(),
                client_nonce: nonce,
                proof: proof.to_vec(),
            },
        )),
    )
    .await?;
    let response = read_frame(&mut stream).await?;
    let Some(wire::frame::Payload::AuthenticationResult(result)) = response.payload else {
        return Err("Core authentication rejected".into());
    };
    let expected =
        sage_core::ipc::server_authentication_proof(secret.expose(), &proof, &result.session_id)?;
    if !result.accepted || !constant_equal(&expected, &result.server_proof) {
        return Err("Core identity proof rejected".into());
    }
    write_frame(
        &mut stream,
        &frame(wire::frame::Payload::AdapterHello(wire::AdapterHello {
            domain: "browser".into(),
        })),
    )
    .await?;
    let mut input = tokio::io::stdin();
    let mut output = tokio::io::stdout();
    loop {
        let incoming = read_frame(&mut stream).await?;
        let Some(wire::frame::Payload::AdapterRequest(request)) = incoming.payload else {
            continue;
        };
        let mut body = serde_json::from_str::<serde_json::Value>(&request.json)?;
        if let Some(target) = &request.browser_target {
            body.as_object_mut().ok_or("Invalid action body")?.insert(
                "browser_target".into(),
                serde_json::to_value(sage_core::browser_target::BrowserTarget::from_wire(target)?)?,
            );
        }
        if request.operation == "execute" {
            let grant = request
                .grant
                .as_ref()
                .ok_or("Missing typed execution grant")?;
            if grant.worker_session != result.session_id || grant.domain != "browser" {
                return Err("Grant belongs to another worker".into());
            }
            body.as_object_mut().ok_or("Invalid action body")?.insert(
                "capability".into(),
                sage_core::capability::wire_grant_json(grant)?,
            );
        }
        let payload = serde_json::to_vec(
            &serde_json::json!({"request_id":request.request_id,"operation":request.operation,"payload":body,"expires_at_unix_ms":request.expires_at_unix_ms}),
        )?;
        if payload.len() > 1024 * 1024 {
            return Err("Browser payload exceeds one MiB".into());
        }
        output.write_u32_le(payload.len() as u32).await?;
        output.write_all(&payload).await?;
        output.flush().await?;
        let length = tokio::time::timeout(std::time::Duration::from_secs(30), input.read_u32_le())
            .await?? as usize;
        if length == 0 || length > 256 * 1024 {
            return Err("Browser response exceeds bounds".into());
        }
        let mut bytes = vec![0; length];
        input.read_exact(&mut bytes).await?;
        let result: serde_json::Value = serde_json::from_slice(&bytes)?;
        if result["request_id"].as_str() != Some(&request.request_id) {
            return Err("Browser response does not match pending request".into());
        }
        write_frame(
            &mut stream,
            &frame(wire::frame::Payload::AdapterResult(wire::AdapterResult {
                request_id: request.request_id,
                success: result["success"].as_bool().unwrap_or(false),
                json: serde_json::to_string(&result["data"])?,
                error: result["error"]
                    .as_str()
                    .unwrap_or("Browser action failed")
                    .chars()
                    .take(4096)
                    .collect(),
            })),
        )
        .await?;
    }
}

fn frame(payload: wire::frame::Payload) -> wire::Frame {
    wire::Frame {
        protocol_version: PROTOCOL_VERSION,
        sequence: NEXT_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        payload: Some(payload),
    }
}

static NEXT_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
fn constant_equal(a: &[u8], b: &[u8]) -> bool {
    use subtle::ConstantTimeEq;
    bool::from(a.ct_eq(b))
}
