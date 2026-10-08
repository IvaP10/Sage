#![forbid(unsafe_code)]
//! Native messaging host. Browser output is restricted to pending adapter
//! replies. No browser payload can become a trusted UI command.
use sage_core::config::{CoreConfig, IpcEndpoint};
use sage_core::ipc::{authentication_proof, read_frame, write_frame};
use sage_core::secrets::{OsSecretStore, load_browser_ipc_secret};
use sage_protocol::{PROTOCOL_VERSION, sage::ipc::v2 as wire};
use tokio::io::{AsyncRead, AsyncWrite};

mod relay;
type HostResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("native host runtime");
    let result = runtime.block_on(run());
    // stdin uses a blocking system read. A disconnected core must not leave
    // its native host waiting indefinitely for another extension message.
    runtime.shutdown_timeout(std::time::Duration::from_millis(200));
    if let Err(error) = result {
        eprintln!("Sage browser connection failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> HostResult<()> {
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

async fn relay<S: AsyncRead + AsyncWrite + Unpin + Send + 'static>(
    mut stream: S,
) -> HostResult<()> {
    let challenge = read_frame(&mut stream).await?;
    if challenge.protocol_version != PROTOCOL_VERSION || challenge.sequence != 1 {
        return Err("Invalid core challenge frame".into());
    }
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
        &frame(
            1,
            wire::frame::Payload::ClientAuthenticate(wire::ClientAuthenticate {
                client_kind: wire::ClientKind::Browser as i32,
                client_version: version.into(),
                client_nonce: nonce,
                proof: proof.to_vec(),
                supported_features: Vec::new(),
            }),
        ),
    )
    .await?;
    let response = read_frame(&mut stream).await?;
    if response.protocol_version != PROTOCOL_VERSION || response.sequence != 2 {
        return Err("Invalid core authentication frame".into());
    }
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
        &frame(
            2,
            wire::frame::Payload::AdapterHello(wire::AdapterHello {
                cancellation_protocol: 1,
                domain: "browser".into(),
            }),
        ),
    )
    .await?;
    relay::session(
        stream,
        tokio::io::stdin(),
        tokio::io::stdout(),
        result.session_id,
        2,
        2,
    )
    .await
}

fn frame(sequence: u64, payload: wire::frame::Payload) -> wire::Frame {
    wire::Frame {
        protocol_version: PROTOCOL_VERSION,
        sequence,
        payload: Some(payload),
    }
}

fn constant_equal(a: &[u8], b: &[u8]) -> bool {
    use subtle::ConstantTimeEq;
    bool::from(a.ct_eq(b))
}
