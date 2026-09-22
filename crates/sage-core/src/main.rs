use std::path::PathBuf;
use std::sync::Arc;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use sage_core::config::{CoreConfig, IpcEndpoint};
use sage_core::ipc::{IpcAuthenticator, serve};
use sage_core::model::{ModelProvider, OpenAICompatibleProvider};
use sage_core::secrets::{OsSecretStore, SecretBytes, SecretStore, load_or_create_ipc_secret};
use sage_core::{CoreError, CoreResult, SageCore};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("sage_core=info")),
        )
        .with_target(false)
        .init();

    let mut config = CoreConfig::platform_default()?;
    let launch = apply_arguments(&mut config)?;
    let secret_store: Arc<dyn SecretStore> = Arc::new(OsSecretStore);
    let secret = if launch.bootstrap_stdin {
        read_bootstrap_secret()?
    } else {
        load_or_create_ipc_secret(Arc::clone(&secret_store))?
    };
    sage_core::secrets::install_browser_ipc_secret(
        &config.data_dir,
        &secret,
        secret_store.as_ref(),
    )?;
    let authenticator = Arc::new(IpcAuthenticator::new(secret));
    let model: Arc<dyn ModelProvider> = match (launch.model_profile, launch.model_trust_key) {
        (Some(profile), Some(key)) => {
            config
                .protected_paths
                .extend([profile.canonicalize()?, key.canonicalize()?]);
            let bytes =
                sage_core::execution::files::PinnedPath::open(&key.canonicalize()?)?.read(32)?;
            let key: [u8; 32] = bytes.try_into().map_err(|_| {
                CoreError::Model("Catalog public key must contain exactly 32 bytes".into())
            })?;
            let profile = sage_core::inference::ModelProfile::load_signed(&profile, &key)?;
            config.protected_paths.extend(profile.protected_paths());
            Arc::new(sage_core::inference::ManagedLocalProvider::new(
                profile,
                sage_core::inference::ResourceGovernor::default(),
            )?)
        }
        (None, None) => Arc::new(OpenAICompatibleProvider::default()),
        _ => {
            return Err(CoreError::InvalidAction(
                "Use --model-profile with --model-trust-key".into(),
            )
            .into());
        }
    };
    let core = SageCore::new_with_secret_store(config.clone(), model, secret_store)?;
    core.start_scheduler();
    serve(core, config.ipc_endpoint, authenticator).await?;
    Ok(())
}

#[derive(Debug, Default)]
struct LaunchOptions {
    bootstrap_stdin: bool,
    model_profile: Option<PathBuf>,
    model_trust_key: Option<PathBuf>,
}

fn apply_arguments(config: &mut CoreConfig) -> CoreResult<LaunchOptions> {
    let mut launch = LaunchOptions::default();
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--model-profile") => {
                launch.model_profile = Some(
                    arguments
                        .next()
                        .ok_or_else(|| CoreError::InvalidAction("Missing model profile".into()))?
                        .into(),
                );
            }
            Some("--model-trust-key") => {
                launch.model_trust_key = Some(
                    arguments
                        .next()
                        .ok_or_else(|| CoreError::InvalidAction("Missing catalog key".into()))?
                        .into(),
                );
            }
            Some("--data-dir") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| CoreError::InvalidAction("--data-dir requires a path".into()))?;
                let path = PathBuf::from(value);
                if !path.is_absolute() {
                    return Err(CoreError::InvalidAction(
                        "--data-dir must be an absolute path".into(),
                    ));
                }
                config.data_dir = path.clone();
                config.database_path = path.join("sage.db");
                config.recovery_dir = path.join("recovery");
                #[cfg(unix)]
                {
                    config.ipc_endpoint = IpcEndpoint::UnixSocket(path.join("sage-core.sock"));
                }
            }
            #[cfg(unix)]
            Some("--socket") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| CoreError::InvalidAction("--socket requires a path".into()))?;
                let path = PathBuf::from(value);
                if !path.is_absolute() {
                    return Err(CoreError::InvalidAction(
                        "--socket must be an absolute path".into(),
                    ));
                }
                config.ipc_endpoint = IpcEndpoint::UnixSocket(path);
            }
            #[cfg(windows)]
            Some("--pipe") => {
                let value = arguments
                    .next()
                    .ok_or_else(|| CoreError::InvalidAction("--pipe requires a name".into()))?;
                config.ipc_endpoint = IpcEndpoint::NamedPipe(value.to_string_lossy().into_owned());
            }
            Some("--version") => {
                println!("sage-core {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            Some("--bootstrap-stdin") => {
                launch.bootstrap_stdin = true;
            }
            _ => {
                return Err(CoreError::InvalidAction(format!(
                    "unknown argument: {}",
                    argument.to_string_lossy()
                )));
            }
        }
    }
    Ok(launch)
}

fn read_bootstrap_secret() -> CoreResult<SecretBytes> {
    use std::io::{BufRead, Read};

    let mut encoded = String::new();
    std::io::stdin()
        .lock()
        .take(256)
        .read_line(&mut encoded)
        .map_err(CoreError::Io)?;
    let bytes = STANDARD
        .decode(encoded.trim())
        .map_err(|_| CoreError::AuthenticationFailed)?;
    if bytes.len() != 32 {
        return Err(CoreError::AuthenticationFailed);
    }
    Ok(SecretBytes::new(bytes))
}
