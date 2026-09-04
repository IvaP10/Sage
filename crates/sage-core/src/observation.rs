use std::path::Path;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::domain::{ActionProposal, Condition, ExpectedOutcome, Provenance, ProvenanceSource};
use crate::error::{CoreError, CoreResult};
use crate::execution::ExecutionReceipt;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Evidence {
    FileState {
        path: String,
        exists: bool,
        is_file: bool,
        size: u64,
    },
    FileHash {
        path: String,
        sha256: String,
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
            ExpectedOutcome::Condition { condition } => vec![observe_condition(condition).await?],
            ExpectedOutcome::FileContains { path, .. } => vec![Evidence::FileHash {
                path: path.to_string_lossy().into_owned(),
                sha256: hash_file(path).await?,
            }],
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

async fn observe_condition(condition: &Condition) -> CoreResult<Evidence> {
    match condition {
        Condition::FileExists { path } | Condition::FileAbsent { path } => {
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
                size: metadata.as_ref().map_or(0, std::fs::Metadata::len),
            })
        }
        Condition::ApplicationRunning { application } => Ok(Evidence::ApplicationState {
            application: application.clone(),
            running: false,
        }),
        Condition::UrlEquals { url } => Ok(Evidence::BrowserState { url: url.clone() }),
        Condition::ElementPresent { selector } => Ok(Evidence::ElementState {
            description: format!("{selector:?}"),
            present: false,
        }),
    }
}

async fn hash_file(path: &Path) -> CoreResult<String> {
    let bytes = tokio::fs::read(path).await?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn summarize(evidence: &[Evidence]) -> String {
    evidence
        .iter()
        .map(|item| match item {
            Evidence::FileState {
                path, exists, size, ..
            } => {
                format!("file state: {path}, exists={exists}, size={size}")
            }
            Evidence::FileHash { path, sha256 } => {
                format!("file hash: {path}, sha256={sha256}")
            }
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
