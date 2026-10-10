use std::path::Path;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::domain::{ActionProposal, Condition, ExpectedOutcome, Provenance, ProvenanceSource};
use crate::error::{CoreError, CoreResult};
use crate::execution::ExecutionReceipt;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Evidence {
    DirectoryPage {
        path: String,
        page_size: u32,
        cursor: Option<String>,
        page_sha256: String,
        snapshot_sha256: String,
        total_entries: u32,
    },
    SignedApplication {
        target: crate::application_target::ApplicationTarget,
        process_id: u32,
    },
    ApplicationControlValue {
        target: crate::application_target::ApplicationTarget,
        process_id: u32,
        control_id: String,
        value: crate::domain::ApplicationControlValue,
    },
    FetchedResource {
        url: String,
        status: u16,
        sha256: String,
    },
    FileState {
        path: String,
        exists: bool,
        is_file: bool,
        is_directory: bool,
        size: u64,
    },
    FileHash {
        path: String,
        sha256: String,
    },
    FileStreamHash {
        path: String,
        channel_id: String,
        producer_node: String,
        file_sha256: String,
        stream_sha256: String,
        bytes: u64,
    },
    FileReadStreamHash {
        path: String,
        channel_id: String,
        producer_node: String,
        output_port: String,
        consumer_node: String,
        file_sha256: String,
        stream_sha256: String,
        bytes: u64,
    },
    ApplicationState {
        application: String,
        running: bool,
    },
    BrowserState {
        url: String,
    },
    ElementState {
        description: String,
        present: bool,
    },
    CommandState {
        exit_code: i32,
    },
    ExternalSuccess {
        marker: String,
        observed: bool,
    },
    UserAnswer {
        received: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub observed_at: DateTime<Utc>,
    pub provenance: Provenance,
    pub summary: String,
    pub evidence: Vec<Evidence>,
}

#[async_trait]
pub trait Observer: Send + Sync {
    async fn observe(
        &self,
        proposal: &ActionProposal,
        receipt: &ExecutionReceipt,
    ) -> CoreResult<Observation>;
}

#[derive(Debug, Default)]
pub struct DeterministicObserver;

#[async_trait]
impl Observer for DeterministicObserver {
    async fn observe(
        &self,
        proposal: &ActionProposal,
        receipt: &ExecutionReceipt,
    ) -> CoreResult<Observation> {
        let evidence = match &proposal.expected_outcome {
            ExpectedOutcome::DirectoryPage {
                path,
                page_size,
                cursor,
            } => {
                let fresh = crate::execution::directory::inspect(
                    None,
                    path.clone(),
                    *page_size,
                    cursor.clone(),
                    None,
                )
                .await?;
                let returned: crate::execution::directory::DirectoryPage =
                    serde_json::from_value(receipt.transient_data.clone())?;
                let before: Option<crate::contracts::FileIdentity> = serde_json::from_str(
                    proposal.metadata.get("file_precondition").ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Prepared directory identity is missing".into(),
                        )
                    })?,
                )?;
                if fresh != returned || before.as_ref() != Some(&fresh.directory_identity) {
                    return Err(CoreError::VerificationFailed(
                        "Directory result differs from the fresh independent listing".into(),
                    ));
                }
                vec![Evidence::DirectoryPage {
                    path: fresh.path.clone(),
                    page_size: *page_size,
                    cursor: cursor.clone(),
                    page_sha256: fresh.digest()?,
                    snapshot_sha256: fresh.snapshot_sha256,
                    total_entries: fresh.total_entries,
                }]
            }
            ExpectedOutcome::SignedApplication { .. } => {
                return Err(CoreError::VerificationFailed(
                    "A fresh signed application observation is required".into(),
                ));
            }
            ExpectedOutcome::ApplicationControlValue { .. } => {
                return Err(CoreError::VerificationFailed(
                    "A fresh signed accessibility-control observation is required".into(),
                ));
            }
            ExpectedOutcome::PublicResource { url } => {
                let document: crate::network::FetchedResource =
                    serde_json::from_value(receipt.transient_data.clone())?;
                document.validate()?;
                if receipt.executor != "network-broker" || document.url != *url {
                    return Err(CoreError::VerificationFailed(
                        "HTTP evidence belongs to another resource".into(),
                    ));
                }
                vec![Evidence::FetchedResource {
                    url: document.url,
                    status: document.status,
                    sha256: document.sha256,
                }]
            }
            ExpectedOutcome::Condition { condition } => vec![observe_condition(condition).await?],
            ExpectedOutcome::FileContains { path, .. } => vec![Evidence::FileHash {
                path: path.to_string_lossy().into_owned(),
                sha256: hash_file(path).await?,
            }],
            ExpectedOutcome::FileMatchesStream {
                path,
                channel_id,
                producer_node,
                maximum_bytes,
            } => {
                let stream_channel_id = receipt
                    .transient_data
                    .get("stream_channel_id")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Streamed write receipt has no channel identity".into(),
                        )
                    })?;
                let stream_producer_node = receipt
                    .transient_data
                    .get("stream_producer_node")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Streamed write receipt has no producer identity".into(),
                        )
                    })?;
                let stream_sha256 = receipt
                    .transient_data
                    .get("stream_sha256")
                    .and_then(serde_json::Value::as_str)
                    .filter(|digest| valid_sha256(digest))
                    .ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Streamed write receipt has no valid content digest".into(),
                        )
                    })?;
                let bytes = receipt
                    .transient_data
                    .get("bytes_written")
                    .and_then(serde_json::Value::as_u64)
                    .filter(|bytes| {
                        *maximum_bytes > 0
                            && *maximum_bytes <= crate::execution::files::MAX_BYTES
                            && *bytes <= *maximum_bytes
                    })
                    .ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Streamed write receipt has no valid byte count".into(),
                        )
                    })?;
                let (file_sha256, file_bytes) = hash_file_with_size(path).await?;
                if receipt.executor != "native-os-executor"
                    || stream_channel_id != channel_id
                    || stream_producer_node != producer_node
                    || file_sha256 != stream_sha256
                    || file_bytes != bytes
                {
                    return Err(CoreError::VerificationFailed(
                        "Fresh file contents do not match the completed input stream".into(),
                    ));
                }
                vec![Evidence::FileStreamHash {
                    path: path.to_string_lossy().into_owned(),
                    channel_id: channel_id.clone(),
                    producer_node: producer_node.clone(),
                    file_sha256,
                    stream_sha256: stream_sha256.to_owned(),
                    bytes: file_bytes,
                }]
            }
            ExpectedOutcome::FileReadMatchesStream {
                path,
                channel_id,
                producer_node,
                output_port,
                consumer_node,
                maximum_bytes,
            } => {
                let receipt_text = |field: &str| {
                    receipt
                        .transient_data
                        .get(field)
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| {
                            CoreError::VerificationFailed(format!(
                                "Streamed read receipt has no {field}"
                            ))
                        })
                };
                let stream_channel_id = receipt_text("stream_channel_id")?;
                let stream_producer_node = receipt_text("stream_producer_node")?;
                let stream_output_port = receipt_text("stream_output_port")?;
                let stream_consumer_node = receipt_text("stream_consumer_node")?;
                let stream_sha256 = receipt_text("stream_sha256")?;
                let bytes = receipt
                    .transient_data
                    .get("bytes_read")
                    .and_then(serde_json::Value::as_u64)
                    .filter(|bytes| *maximum_bytes > 0 && *bytes <= *maximum_bytes)
                    .ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Streamed read receipt has no valid byte count".into(),
                        )
                    })?;
                let (file_sha256, file_bytes) = hash_file_with_size(path).await?;
                if receipt.executor != "native-os-executor"
                    || stream_channel_id != channel_id
                    || stream_producer_node != producer_node
                    || stream_output_port != output_port
                    || stream_consumer_node != consumer_node
                    || !valid_sha256(stream_sha256)
                    || file_sha256 != stream_sha256
                    || file_bytes != bytes
                {
                    return Err(CoreError::VerificationFailed(
                        "Fresh source file differs from the completed output stream".into(),
                    ));
                }
                vec![Evidence::FileReadStreamHash {
                    path: path.to_string_lossy().into_owned(),
                    channel_id: channel_id.clone(),
                    producer_node: producer_node.clone(),
                    output_port: output_port.clone(),
                    consumer_node: consumer_node.clone(),
                    file_sha256,
                    stream_sha256: stream_sha256.to_owned(),
                    bytes: file_bytes,
                }]
            }
            ExpectedOutcome::CommandExit { .. } => {
                let code = receipt
                    .transient_data
                    .get("exit_code")
                    .and_then(serde_json::Value::as_i64)
                    .ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "sandbox worker did not return a structured exit code".into(),
                        )
                    })?;
                vec![Evidence::CommandState {
                    exit_code: code as i32,
                }]
            }
            ExpectedOutcome::ExternalSuccess { marker } => vec![Evidence::ExternalSuccess {
                marker: marker.clone(),
                observed: receipt
                    .transient_data
                    .get("external_success")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            }],
            ExpectedOutcome::UserAnswered => vec![Evidence::UserAnswer {
                received: receipt
                    .transient_data
                    .get("user_answered")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            }],
        };
        Ok(Observation {
            observed_at: Utc::now(),
            provenance: Provenance {
                source: ProvenanceSource::SageCore,
                trust: crate::domain::TrustClass::Observation,
                source_id: Some(proposal.id.to_string()),
                parent_ids: Vec::new(),
            },
            summary: summarize(&evidence),
            evidence,
        })
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

async fn observe_condition(condition: &Condition) -> CoreResult<Evidence> {
    match condition {
        Condition::FileExists { path }
        | Condition::FolderExists { path }
        | Condition::FileAbsent { path } => {
            let exists = tokio::fs::try_exists(path).await?;
            let metadata = if exists {
                Some(tokio::fs::metadata(path).await?)
            } else {
                None
            };
            Ok(Evidence::FileState {
                path: path.to_string_lossy().into_owned(),
                exists,
                is_file: metadata.as_ref().is_some_and(|value| value.is_file()),
                is_directory: metadata.as_ref().is_some_and(|value| value.is_dir()),
                size: metadata.as_ref().map_or(0, std::fs::Metadata::len),
            })
        }
        _ => Err(CoreError::ExecutorUnavailable(
            "A fresh platform/browser observation is required.".into(),
        )),
    }
}

async fn hash_file(path: &Path) -> CoreResult<String> {
    let path = path.to_path_buf();
    crate::execution::io::bounded_read(move |_| crate::execution::files::hash_file(&path)).await
}

async fn hash_file_with_size(path: &Path) -> CoreResult<(String, u64)> {
    let path = path.to_path_buf();
    crate::execution::io::bounded_read(move |_| crate::execution::files::hash_file_with_size(&path))
        .await
}

fn summarize(evidence: &[Evidence]) -> String {
    evidence
        .iter()
        .map(|item| match item {
            Evidence::DirectoryPage {
                path,
                total_entries,
                ..
            } => format!("Verified directory page from {total_entries} entries in {path}"),
            Evidence::SignedApplication { target, process_id } => format!(
                "Signed application {} at {}, process {}",
                target.identifier, target.bundle_path, process_id
            ),
            Evidence::ApplicationControlValue {
                target,
                control_id,
                value,
                ..
            } => format!(
                "Application control {} on {} now reads {}",
                control_id,
                target.identifier,
                serde_json::to_string(value).unwrap_or_else(|_| "[invalid]".into())
            ),
            Evidence::FetchedResource { url, status, .. } => {
                format!("Fetched {url}; HTTP {status}")
            }
            Evidence::FileState {
                path, exists, size, ..
            } => {
                format!("file state: {path}, exists={exists}, size={size}")
            }
            Evidence::FileHash { path, sha256 } => {
                format!("file hash: {path}, sha256={sha256}")
            }
            Evidence::FileStreamHash {
                path,
                channel_id,
                producer_node,
                bytes,
                ..
            } => format!(
                "streamed file {path} received {bytes} bytes from {producer_node} via {channel_id}"
            ),
            Evidence::FileReadStreamHash {
                path,
                channel_id,
                producer_node,
                consumer_node,
                bytes,
                ..
            } => format!(
                "streamed {bytes} bytes from {path} at {producer_node} to {consumer_node} via {channel_id}"
            ),
            Evidence::ApplicationState {
                application,
                running,
            } => {
                format!("application state: {application}, running={running}")
            }
            Evidence::BrowserState { url } => format!("browser URL: {url}"),
            Evidence::ElementState {
                description,
                present,
            } => {
                format!("element: {description}, present={present}")
            }
            Evidence::CommandState { exit_code } => format!("command exit code: {exit_code}"),
            Evidence::ExternalSuccess { marker, observed } => {
                format!("external marker {marker}, observed={observed}")
            }
            Evidence::UserAnswer { received } => format!("user answered={received}"),
        })
        .collect::<Vec<_>>()
        .join("; ")
}
