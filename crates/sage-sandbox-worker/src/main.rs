#![forbid(unsafe_code)]

use std::path::Path;
use std::process::Stdio;

use sage_core::capability::CapabilityResource;
use sage_core::domain::{Action, ExecutionDomain};
use sage_core::execution::ExecutionReceipt;
use sage_worker_common::{WorkerRequest, WorkerResponse, read_request, write_response};
use serde_json::json;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::{Duration, timeout};

const OUTPUT_LIMIT: usize = 1024 * 1024;

#[tokio::main]
async fn main() {
    let response = match run().await {
        Ok(receipt) => WorkerResponse::success(receipt),
        Err(error) => WorkerResponse::failure(error),
    };
    if write_response(&response).await.is_err() {
        std::process::exit(2);
    }
}

async fn run() -> Result<ExecutionReceipt, String> {
    if std::env::args().nth(1).as_deref() != Some("--single-request") {
        return Err("sandbox worker accepts only --single-request".into());
    }
    let request = read_request().await?;
    validate(&request)?;
    let Action::RunCommand {
        program,
        args,
        working_directory,
        network,
        timeout_seconds,
    } = &request.action.proposal.action
    else {
        return Err("sandbox worker accepts only RunCommand".into());
    };

    #[cfg(target_os = "macos")]
    let mut command = {
        let sandbox = Path::new("/usr/bin/sandbox-exec");
        if !sandbox.exists() {
            return Err(
                "macOS sandbox backend is unavailable; refusing unsandboxed execution".into(),
            );
        }
        let profile = macos_profile(program, working_directory.as_deref(), *network);
        let mut command = Command::new(sandbox);
        command.arg("-p").arg(profile).arg(program).args(args);
        command
    };

    #[cfg(target_os = "windows")]
    return Err(
        "Windows AppContainer sandbox backend is not installed; refusing unsandboxed execution"
            .into(),
    );

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    return Err("this platform has no production sandbox backend".into());

    #[cfg(target_os = "macos")]
    {
        if let Some(directory) = working_directory {
            command.current_dir(directory);
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .kill_on_drop(true);
        let mut child = command.spawn().map_err(|error| error.to_string())?;
        let stdout = child.stdout.take().ok_or("sandbox stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("sandbox stderr unavailable")?;
        let operation = async {
            let mut stdout_bytes = Vec::new();
            let mut stderr_bytes = Vec::new();
            stdout
                .take((OUTPUT_LIMIT + 1) as u64)
                .read_to_end(&mut stdout_bytes)
                .await
                .map_err(|error| error.to_string())?;
            stderr
                .take((OUTPUT_LIMIT + 1) as u64)
                .read_to_end(&mut stderr_bytes)
                .await
                .map_err(|error| error.to_string())?;
            if stdout_bytes.len() > OUTPUT_LIMIT || stderr_bytes.len() > OUTPUT_LIMIT {
                return Err("sandbox output exceeded the one-megabyte limit".into());
            }
            let status = child.wait().await.map_err(|error| error.to_string())?;
            Ok::<_, String>((status.code().unwrap_or(-1), stdout_bytes, stderr_bytes))
        };
        let (exit_code, stdout, stderr) =
            timeout(Duration::from_secs(*timeout_seconds as u64), operation)
                .await
                .map_err(|_| "sandbox command timed out".to_string())??;
        Ok(ExecutionReceipt {
            executor: "sandbox-worker".into(),
            summary: format!("sandboxed process exited with code {exit_code}"),
            transient_data: json!({
                "exit_code": exit_code,
                "stdout": String::from_utf8_lossy(&stdout),
                "stderr": String::from_utf8_lossy(&stderr),
            }),
            rollback: None,
        })
    }
}

fn validate(request: &WorkerRequest) -> Result<(), String> {
    if request.implementation.executor != ExecutionDomain::Sandbox
        || request.capability.domain != ExecutionDomain::Sandbox
        || request.capability.task_id != request.action.proposal.task_id
        || request.capability.action_id != request.action.proposal.id
    {
        return Err("sandbox request capability binding is invalid".into());
    }
    let Action::RunCommand {
        program,
        working_directory,
        network,
        ..
    } = &request.action.proposal.action
    else {
        return Err("sandbox request is not a structured command".into());
    };
    match &request.capability.resource {
        CapabilityResource::Command {
            executable,
            working_directory: allowed_directory,
            network: allowed_network,
        } if executable == program
            && allowed_directory.as_deref()
                == working_directory
                    .as_ref()
                    .map(|path| path.to_string_lossy())
                    .as_deref()
            && allowed_network == network => {}
        _ => return Err("sandbox executable is outside the capability".into()),
    }
    if !Path::new(program).is_absolute() {
        return Err("sandbox executable must be an absolute path".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn macos_profile(program: &str, working_directory: Option<&Path>, network: bool) -> String {
    fn literal(path: &str) -> String {
        path.replace('\\', "\\\\").replace('"', "\\\"")
    }
    let mut profile = format!(
        r#"(version 1)
(deny default)
(allow process-exec (literal "{}"))
(allow process-fork)
(allow sysctl-read)
(allow file-read* (subpath "/System") (subpath "/usr/lib") (subpath "/private/var/db/dyld") (literal "{}"))
(allow file-write* (subpath "/private/tmp"))
"#,
        literal(program),
        literal(program),
    );
    if let Some(directory) = working_directory {
        profile.push_str(&format!(
            "(allow file-read* file-write* (subpath \"{}\"))\n",
            literal(&directory.to_string_lossy())
        ));
    }
    if network {
        profile.push_str("(allow network-outbound)\n");
    }
    profile
}
