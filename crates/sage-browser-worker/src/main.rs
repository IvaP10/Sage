#![forbid(unsafe_code)]

use sage_core::capability::CapabilityResource;
use sage_core::domain::{Action, ExecutionDomain};
use sage_worker_common::{WorkerResponse, read_request, write_response};

#[tokio::main]
async fn main() {
    let response = match validate().await {
        Ok(()) => WorkerResponse::failure(
            "no authenticated browser session is paired; refusing visual or unscoped fallback",
        ),
        Err(error) => WorkerResponse::failure(error),
    };
    if write_response(&response).await.is_err() {
        std::process::exit(2);
    }
}

async fn validate() -> Result<(), String> {
    if std::env::args().nth(1).as_deref() != Some("--single-request") {
        return Err("browser worker accepts only --single-request".into());
    }
    let request = read_request().await?;
    if request.implementation.executor != ExecutionDomain::Browser
        || request.capability.domain != ExecutionDomain::Browser
        || request.capability.task_id != request.action.proposal.task_id
        || request.capability.action_id != request.action.proposal.id
    {
        return Err("browser request capability binding is invalid".into());
    }
    let expected_origin = match &request.action.proposal.action {
        Action::NavigateUrl { url, .. } | Action::DownloadFile { url, .. } => origin(url)?,
        Action::UploadFile {
            destination_origin, ..
        } => destination_origin.clone(),
        Action::SubmitForm { origin, .. } => origin.clone(),
        _ => return Err("browser worker received a non-browser action".into()),
    };
    match &request.capability.resource {
        CapabilityResource::BrowserOrigin { origin } if origin == &expected_origin => Ok(()),
        _ => Err("browser origin is outside the single-use capability".into()),
    }
}

fn origin(url: &str) -> Result<String, String> {
    let (scheme, rest) = url.split_once("://").ok_or("URL has no scheme")?;
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .filter(|value| !value.is_empty())
        .ok_or("URL has no authority")?;
    Ok(format!("{scheme}://{authority}"))
}
