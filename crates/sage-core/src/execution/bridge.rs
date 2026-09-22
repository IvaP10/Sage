//! Session-bound adapter RPC. Only the broker may request execution; adapters
//! cannot submit tasks, approve actions, or inject trusted planning constraints.
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use sage_protocol::sage::ipc::v2 as wire;
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::time::{Duration, timeout};
use uuid::Uuid;

use super::{ExecutionReceipt, Executor, PlatformController};
use crate::capability::CapabilityGrant;
use crate::compiler::{CompiledAction, ImplementationCandidate};
use crate::context::ContextObservation;
use crate::domain::{
    Action, ActionProposal, Condition, ExecutionDomain, ExpectedOutcome, Provenance,
    ProvenanceSource, TrustClass,
};
use crate::error::{CoreError, CoreResult};
use crate::observation::{DeterministicObserver, Evidence, Observation, Observer};

struct Session {
    id: String,
    sender: mpsc::Sender<wire::AdapterRequest>,
}
struct Pending {
    session: String,
    sender: oneshot::Sender<wire::AdapterResult>,
}

#[derive(Default)]
pub struct AdapterBridge {
    sessions: Mutex<HashMap<String, Session>>,
    pending: Mutex<HashMap<String, Pending>>,
}

impl AdapterBridge {
    pub async fn register(
        &self,
        domain: &str,
        session: &str,
        kind: i32,
        sender: mpsc::Sender<wire::AdapterRequest>,
    ) -> CoreResult<()> {
        let valid = match domain {
            "native" => matches!(
                wire::ClientKind::try_from(kind),
                Ok(wire::ClientKind::Macos | wire::ClientKind::Windows)
            ),
            "browser" => kind == wire::ClientKind::Browser as i32,
            _ => false,
        };
        if !valid {
            return Err(CoreError::Protocol(
                "Adapter domain does not match authenticated client.".into(),
            ));
        }
        let mut sessions = self.sessions.lock().await;
        if sessions
            .get(domain)
            .is_some_and(|s| s.id != session && !s.sender.is_closed())
        {
            return Err(CoreError::Protocol(
                "An adapter is already connected for this domain.".into(),
            ));
        }
        sessions.insert(
            domain.into(),
            Session {
                id: session.into(),
                sender,
            },
        );
        Ok(())
    }

    pub async fn disconnect(&self, session: &str) {
        self.sessions
            .lock()
            .await
            .retain(|_, value| value.id != session);
        self.pending
            .lock()
            .await
            .retain(|_, value| value.session != session);
    }

    pub async fn complete(&self, session: &str, response: wire::AdapterResult) -> CoreResult<()> {
        let mut pending = self.pending.lock().await;
        if !pending
            .get(&response.request_id)
            .is_some_and(|p| p.session == session)
        {
            return Err(CoreError::Protocol(
                "Stale or mismatched adapter response.".into(),
            ));
        }
        if response.json.len() > 256 * 1024 || response.error.len() > 4096 {
            return Err(CoreError::Protocol(
                "Adapter response exceeds bounds.".into(),
            ));
        }
        if let Some(waiter) = pending.remove(&response.request_id) {
            let _ = waiter.sender.send(response);
        }
        Ok(())
    }

    pub async fn session_id(&self, domain: &str) -> Option<String> {
        self.sessions
            .lock()
            .await
            .get(domain)
            .filter(|s| !s.sender.is_closed())
            .map(|s| s.id.clone())
    }

    pub async fn available(&self, domain: &str) -> bool {
        self.sessions
            .lock()
            .await
            .get(domain)
            .is_some_and(|s| !s.sender.is_closed())
    }

    pub async fn request(
        &self,
        domain: &str,
        operation: &str,
        mut payload: Value,
    ) -> CoreResult<Value> {
        let (session, sender) = {
            let sessions = self.sessions.lock().await;
            let session = sessions.get(domain).ok_or_else(|| {
                CoreError::ExecutorUnavailable(format!("{domain} adapter is not connected"))
            })?;
            (session.id.clone(), session.sender.clone())
        };
        let id = Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();
        let duration = if operation == "context" {
            Duration::from_secs(3)
        } else {
            Duration::from_secs(30)
        };
        let grant = if operation == "execute" {
            let value = payload
                .as_object_mut()
                .and_then(|p| p.remove("capability"))
                .ok_or_else(|| CoreError::CapabilityRejected("Missing execution grant".into()))?;
            let capability: CapabilityGrant = serde_json::from_value(value)?;
            if capability.worker_session.as_deref() != Some(&session) {
                return Err(CoreError::CapabilityRejected(
                    "Worker session changed before dispatch".into(),
                ));
            }
            Some(crate::capability::grant_to_wire(&capability)?)
        } else {
            None
        };
        let browser_target = payload
            .as_object_mut()
            .and_then(|p| p.remove("browser_target"))
            .map(serde_json::from_value::<crate::browser_target::BrowserTarget>)
            .transpose()?;
        if let Some(target) = &browser_target {
            target.validate()?;
        }
        if domain == "browser" && operation == "execute" && browser_target.is_none() {
            return Err(CoreError::CapabilityRejected(
                "Missing prepared browser target".into(),
            ));
        }
        let request = wire::AdapterRequest {
            browser_target: browser_target.map(|t| t.to_wire()),
            grant,
            request_id: id.clone(),
            operation: operation.into(),
            json: serde_json::to_string(&payload)?,
            expires_at_unix_ms: Utc::now().timestamp_millis() + duration.as_millis() as i64,
        };
        if request.json.len() > 1024 * 1024 {
            return Err(CoreError::InvalidAction(
                "Adapter payload exceeds one MiB.".into(),
            ));
        }
        self.pending.lock().await.insert(
            id.clone(),
            Pending {
                session,
                sender: tx,
            },
        );
        let result = timeout(duration, async {
            sender
                .send(request)
                .await
                .map_err(|_| CoreError::ExecutorUnavailable("Adapter disconnected".into()))?;
            let response = rx
                .await
                .map_err(|_| CoreError::ExecutorUnavailable("Adapter disconnected".into()))?;
            if !response.success {
                return Err(CoreError::ExecutionFailed(
                    crate::redaction::redact_for_persistence(&response.error),
                ));
            }
            serde_json::from_str(&response.json).map_err(Into::into)
        })
        .await
        .map_err(|_| CoreError::Timeout("Adapter request expired; no automatic retry.".into()));
        self.pending.lock().await.remove(&id);
        result?
    }

    pub async fn context(&self) -> Vec<ContextObservation> {
        let (native, browser) = tokio::join!(
            self.request("native", "context", json!({})),
            self.request("browser", "context", json!({}))
        );
        [("native", native), ("browser", browser)]
            .into_iter()
            .map(|(source, result)| ContextObservation {
                source: source.into(),
                observed_at_unix_ms: Utc::now().timestamp_millis(),
                state: result.unwrap_or_else(|_| json!({"available":false})),
            })
            .collect()
    }

    pub async fn observe_condition(
        &self,
        condition: &Condition,
        application: Option<&str>,
    ) -> CoreResult<Evidence> {
        let domain = match condition {
            Condition::UrlEquals { .. } => "browser",
            Condition::ElementPresent { selector } if selector.browser_selector.is_some() => {
                "browser"
            }
            _ => "native",
        };
        let value = self
            .request(
                domain,
                "observe",
                json!({"condition":condition,"application":application}),
            )
            .await?;
        match condition {
            Condition::ApplicationRunning { application } => Ok(Evidence::ApplicationState {
                application: application.clone(),
                running: required_bool(&value, "running")?,
            }),
            Condition::UrlEquals { .. } => Ok(Evidence::BrowserState {
                url: value["url"]
                    .as_str()
                    .ok_or_else(|| {
                        CoreError::VerificationFailed("Browser omitted observed URL".into())
                    })?
                    .into(),
            }),
            Condition::ElementPresent { selector } => Ok(Evidence::ElementState {
                description: format!("{selector:?}"),
                present: required_bool(&value, "present")?,
            }),
            _ => Err(CoreError::VerificationFailed(
                "Unsupported adapter observation".into(),
            )),
        }
    }
}

fn required_bool(value: &Value, key: &str) -> CoreResult<bool> {
    value[key]
        .as_bool()
        .ok_or_else(|| CoreError::VerificationFailed(format!("Adapter omitted {key}")))
}

#[async_trait]
impl PlatformController for AdapterBridge {
    async fn execute_native(
        &self,
        action: &Action,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        if capability.domain != ExecutionDomain::Native
            || capability.expires_at <= Utc::now()
            || capability.revoked
            || capability.remaining_uses != 0
        {
            return Err(CoreError::CapabilityRejected(
                "Invalid consumed native grant".into(),
            ));
        }
        if let Action::WaitForCondition {
            condition,
            timeout_ms,
        } = action
        {
            let deadline =
                tokio::time::Instant::now() + Duration::from_millis((*timeout_ms).min(300_000));
            loop {
                let satisfied = match condition {
                    Condition::FolderExists { path } => {
                        tokio::fs::metadata(path).await.is_ok_and(|m| m.is_dir())
                    }
                    Condition::FileExists { path } => tokio::fs::try_exists(path).await?,
                    Condition::FileAbsent { path } => !tokio::fs::try_exists(path).await?,
                    _ => {
                        let evidence = self.observe_condition(condition, None).await?;
                        crate::verification::Verifier
                            .verify(
                                &ExpectedOutcome::Condition {
                                    condition: condition.clone(),
                                },
                                &Observation {
                                    observed_at: Utc::now(),
                                    provenance: Provenance::external(
                                        ProvenanceSource::OperatingSystem,
                                        "wait",
                                    ),
                                    summary: String::new(),
                                    evidence: vec![evidence],
                                },
                            )
                            .is_ok()
                    }
                };
                if satisfied {
                    return Ok(ExecutionReceipt {
                        executor: "condition-observer".into(),
                        summary: "Condition observed".into(),
                        transient_data: json!({}),
                        rollback: None,
                    });
                }
                if tokio::time::Instant::now() >= deadline {
                    return Err(CoreError::Timeout(
                        "Condition was not observed before timeout".into(),
                    ));
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
        if capability.worker_session != self.session_id("native").await {
            return Err(CoreError::CapabilityRejected(
                "Native worker session changed".into(),
            ));
        }
        let data = self
            .request(
                "native",
                "execute",
                json!({"action":action,"capability":capability}),
            )
            .await?;
        Ok(ExecutionReceipt {
            executor: "native-platform-adapter".into(),
            summary: action.redacted_summary(),
            transient_data: data,
            rollback: None,
        })
    }
}

pub struct BrowserExecutor(pub Arc<AdapterBridge>);
#[async_trait]
impl Executor for BrowserExecutor {
    fn name(&self) -> &'static str {
        "paired-browser-session"
    }
    fn domain(&self) -> ExecutionDomain {
        ExecutionDomain::Browser
    }
    async fn execute(
        &self,
        action: &CompiledAction,
        _implementation: &ImplementationCandidate,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        if capability.domain != ExecutionDomain::Browser
            || capability.task_id != action.proposal.task_id
            || capability.action_id != action.proposal.id
            || capability.expires_at <= Utc::now()
            || capability.revoked
            || capability.remaining_uses != 0
        {
            return Err(CoreError::CapabilityRejected(
                "Invalid consumed browser grant".into(),
            ));
        }
        if capability.worker_session != self.0.session_id("browser").await {
            return Err(CoreError::CapabilityRejected(
                "Browser worker session changed".into(),
            ));
        }
        let target: crate::browser_target::BrowserTarget =
            serde_json::from_str(action.proposal.metadata.get("browser_target").ok_or_else(
                || CoreError::CapabilityRejected("Browser target was not prepared".into()),
            )?)?;
        target.validate()?;
        let data=self.0.request("browser","execute",json!({"action":action.proposal.action,"capability":capability,"browser_target":target})).await?;
        Ok(ExecutionReceipt {
            executor: self.name().into(),
            summary: action.proposal.action.redacted_summary(),
            transient_data: data,
            rollback: None,
        })
    }
}

pub struct PlatformObserver(pub Arc<AdapterBridge>);
#[async_trait]
impl Observer for PlatformObserver {
    async fn observe(
        &self,
        proposal: &ActionProposal,
        receipt: &ExecutionReceipt,
    ) -> CoreResult<Observation> {
        if let Action::NavigateUrl { url, .. } = &proposal.action {
            let target: crate::browser_target::BrowserTarget = serde_json::from_value(
                receipt
                    .transient_data
                    .get("browser_target")
                    .cloned()
                    .ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Navigation omitted its resulting target".into(),
                        )
                    })?,
            )?;
            target.validate()?;
            let observed: crate::browser_target::BrowserTarget =
                serde_json::from_value(self.0.request("browser", "binding", json!({})).await?)?;
            observed.validate()?;
            if target != observed || observed.url != *url {
                return Err(CoreError::VerificationFailed(
                    "Navigation target changed before verification".into(),
                ));
            }
            return Ok(Observation {
                observed_at: Utc::now(),
                provenance: Provenance::external(
                    ProvenanceSource::OperatingSystem,
                    "paired_browser",
                ),
                summary: "Observed the approved URL in the resulting paired document".into(),
                evidence: vec![Evidence::BrowserState { url: observed.url }],
            });
        }
        if let ExpectedOutcome::Condition { condition } = &proposal.expected_outcome
            && !matches!(
                condition,
                Condition::FileExists { .. }
                    | Condition::FileAbsent { .. }
                    | Condition::FolderExists { .. }
            )
        {
            let app = match &proposal.action {
                Action::ClickElement { application, .. }
                | Action::TypeText { application, .. }
                | Action::PressShortcut { application, .. } => Some(application.as_str()),
                _ => None,
            };
            let evidence = self.0.observe_condition(condition, app).await?;
            return Ok(Observation {
                observed_at: Utc::now(),
                provenance: Provenance {
                    source: ProvenanceSource::OperatingSystem,
                    trust: TrustClass::Observation,
                    source_id: Some(proposal.id.to_string()),
                    parent_ids: Vec::new(),
                },
                summary: crate::redaction::redact_for_persistence(&serde_json::to_string(
                    &evidence,
                )?),
                evidence: vec![evidence],
            });
        }
        DeterministicObserver.observe(proposal, receipt).await
    }
}
