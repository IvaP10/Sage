#![forbid(unsafe_code)]

use sage_core::capability::CapabilityResource;
use sage_core::domain::{Action, ExecutionDomain};
use sage_worker_common::{WorkerResponse, read_request, write_response};

#[tokio::main]
async fn main() {
    let response = match validate().await {
        Ok(operation) => WorkerResponse::failure(format!(
            "privileged operation '{operation}' has no installed signed helper implementation"
        )),
        Err(error) => WorkerResponse::failure(error),
    };
    if write_response(&response).await.is_err() {
        std::process::exit(2);
    }
}

async fn validate() -> Result<&'static str, String> {
    if std::env::args().nth(1).as_deref() != Some("--single-request") {
        return Err("privileged helper accepts only --single-request".into());
    }
    let request = read_request().await?;
    if request.implementation.executor != ExecutionDomain::Privileged
        || request.capability.domain != ExecutionDomain::Privileged
        || request.capability.task_id != request.action.proposal.task_id
        || request.capability.action_id != request.action.proposal.id
    {
        return Err("privileged capability binding is invalid".into());
    }
    match (
        &request.action.proposal.action,
        &request.capability.resource,
    ) {
        (Action::InstallApplication { source }, CapabilityResource::Application { identifier })
            if source == identifier =>
        {
            Ok("install_verified_application")
        }
        _ => Err("operation is not in the privileged helper allowlist".into()),
    }
}
