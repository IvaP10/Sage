#![forbid(unsafe_code)]

// Kept as a protocol-compatible refusal for older packaging scripts. Arbitrary
// code requires an attested VM backend; the former host sandbox is retired.
use sage_worker_common::{WorkerResponse, write_response};

#[tokio::main]
async fn main() {
    let response = WorkerResponse::failure(
        "Code execution is unavailable until a signed VM image and platform isolation checks are qualified. Host command execution is disabled.",
    );
    if write_response(&response).await.is_err() {
        std::process::exit(2);
    }
}
