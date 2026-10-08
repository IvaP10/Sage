//! Session-bound adapter RPC. Only the broker may request execution; adapters
//! cannot submit tasks, approve actions, or inject trusted planning constraints.
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex as SyncMutex};

use async_trait::async_trait;
use chrono::Utc;
use prost::Message;
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

#[derive(Clone)]
pub struct AdapterEndpoint {
    pub requests: mpsc::Sender<wire::AdapterRequest>,
    pub cancellations: Option<mpsc::Sender<wire::AdapterCancel>>,
    pub terminate: tokio::sync::watch::Sender<bool>,
}

struct Session {
    id: String,
    endpoint: AdapterEndpoint,
    features: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub struct EffectBinding {
    pub task_id: Uuid,
    pub action_id: Uuid,
    pub action_digest: String,
}

#[derive(Clone)]
struct RequestIdentity {
    session: String,
    binding: Option<EffectBinding>,
    expires_at_unix_ms: i64,
}

struct Pending {
    identity: RequestIdentity,
    endpoint: AdapterEndpoint,
    sender: oneshot::Sender<wire::AdapterResult>,
}

struct Retired {
    identity: RequestIdentity,
    expects_ack: bool,
    acknowledged: bool,
    replied: bool,
}

#[derive(Default)]
struct Requests {
    pending: HashMap<String, Pending>,
    retired: HashMap<String, Retired>,
}

/// This is transport evidence, not verification of the requested postcondition.
#[derive(Debug)]
pub struct LateAdapterResult {
    pub session: String,
    pub binding: Option<EffectBinding>,
    pub response: wire::AdapterResult,
    pub never_sent: bool,
}

struct PendingGuard<'a> {
    requests: &'a SyncMutex<Requests>,
    id: String,
}
impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut requests) = self.requests.lock()
            && let Some(pending) = requests.pending.remove(&self.id)
        {
            let cancel = wire::AdapterCancel {
                request_id: self.id.clone(),
                expires_at_unix_ms: pending.identity.expires_at_unix_ms,
            };
            let expects_ack = pending.endpoint.cancellations.is_some();
            requests.retired.insert(
                self.id.clone(),
                Retired {
                    identity: pending.identity,
                    expects_ack,
                    acknowledged: false,
                    replied: false,
                },
            );
            if let Some(control) = pending.endpoint.cancellations
                && control.try_send(cancel).is_err()
            {
                // A saturated control lane cannot silently discard revocation.
                // Disconnect makes the native peer cancel its session's workers.
                let _ = pending.endpoint.terminate.send(true);
            }
        }
    }
}

pub struct AdapterBridge {
    sessions: Mutex<HashMap<String, Session>>,
    requests: SyncMutex<Requests>,
    store: crate::storage::LocalStore,
}

impl AdapterBridge {
    pub fn new(store: crate::storage::LocalStore) -> Self {
        Self {
            sessions: Default::default(),
            requests: Default::default(),
            store,
        }
    }
    pub async fn register(
        &self,
        domain: &str,
        session: &str,
        kind: i32,
        endpoint: AdapterEndpoint,
        negotiated_features: &[String],
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
            .is_some_and(|s| s.id != session && !s.endpoint.requests.is_closed())
        {
            return Err(CoreError::Protocol(
                "An adapter is already connected for this domain.".into(),
            ));
        }
        sessions.insert(
            domain.into(),
            Session {
                id: session.into(),
                endpoint,
                features: negotiated_features.iter().cloned().collect(),
            },
        );
        Ok(())
    }

    pub async fn disconnect(&self, session: &str) {
        self.sessions
            .lock()
            .await
            .retain(|_, value| value.id != session);
        let mut requests = self.requests.lock().expect("adapter request lock poisoned");
        requests
            .pending
            .retain(|_, value| value.identity.session != session);
        requests
            .retired
            .retain(|_, value| value.identity.session != session);
    }

    pub fn is_pending(&self, session: &str, id: &str) -> bool {
        self.requests
            .lock()
            .expect("adapter request lock poisoned")
            .pending
            .get(id)
            .is_some_and(|pending| pending.identity.session == session)
    }

    pub fn has_outstanding_effects(&self, task_id: Uuid) -> bool {
        let requests = self.requests.lock().expect("adapter request lock poisoned");
        requests.pending.values().any(|p| {
            p.identity
                .binding
                .as_ref()
                .is_some_and(|b| b.task_id == task_id)
        }) || requests.retired.values().any(|p| {
            p.identity
                .binding
                .as_ref()
                .is_some_and(|b| b.task_id == task_id)
        })
    }

    pub fn cancelled_before_send(
        &self,
        session: &str,
        id: &str,
    ) -> CoreResult<Option<LateAdapterResult>> {
        let late = {
            let mut requests = self.requests.lock().expect("adapter request lock poisoned");
            let Some(retired) = requests.retired.get_mut(id) else {
                return Ok(None);
            };
            if retired.identity.session != session || retired.replied {
                return Ok(None);
            }
            retired.replied = true;
            let late = LateAdapterResult {
                session: session.into(),
                binding: retired.identity.binding.clone(),
                never_sent: true,
                response: wire::AdapterResult {
                    request_id: id.into(),
                    error: "Cancelled before IPC dispatch".into(),
                    ..Default::default()
                },
            };
            if !retired.expects_ack || retired.acknowledged {
                requests.retired.remove(id);
            }
            late
        };
        if let Some(binding) = &late.binding {
            self.store
                .record_worker_receipt(id, session, binding, "never_sent", &[])?;
        }
        Ok(Some(late))
    }

    pub fn acknowledge_cancel(&self, session: &str, id: &str) -> CoreResult<Option<EffectBinding>> {
        let binding = {
            let mut requests = self.requests.lock().expect("adapter request lock poisoned");
            let retired = requests
                .retired
                .get_mut(id)
                .filter(|retired| {
                    retired.identity.session == session
                        && retired.expects_ack
                        && !retired.acknowledged
                })
                .ok_or_else(|| {
                    CoreError::Protocol("Uncorrelated cancellation acknowledgement".into())
                })?;
            retired.acknowledged = true;
            let binding = retired.identity.binding.clone();
            if retired.replied {
                requests.retired.remove(id);
            }
            binding
        };
        if let Some(binding) = &binding {
            self.store
                .record_worker_receipt(id, session, binding, "cancel_ack", &[])?;
        }
        Ok(binding)
    }

    pub async fn complete(
        &self,
        session: &str,
        response: wire::AdapterResult,
    ) -> CoreResult<Option<LateAdapterResult>> {
        if response.encoded_len() > 512 * 1024
            || response.json.len() > 256 * 1024
            || response.error.len() > 4096
        {
            return Err(CoreError::Protocol(
                "Adapter response exceeds bounds.".into(),
            ));
        }
        let (binding, sender) = {
            let mut requests = self.requests.lock().expect("adapter request lock poisoned");
            if requests
                .pending
                .get(&response.request_id)
                .is_some_and(|pending| pending.identity.session == session)
            {
                let pending = requests
                    .pending
                    .remove(&response.request_id)
                    .expect("checked pending request");
                (pending.identity.binding, Some(pending.sender))
            } else if let Some(retired) = requests.retired.get_mut(&response.request_id)
                && retired.identity.session == session
                && !retired.replied
            {
                retired.replied = true;
                let binding = retired.identity.binding.clone();
                if !retired.expects_ack || retired.acknowledged {
                    requests.retired.remove(&response.request_id);
                }
                (binding, None)
            } else {
                return Err(CoreError::Protocol(
                    "Uncorrelated or mismatched adapter response.".into(),
                ));
            }
        };
        // Persist even normal effect replies before waking their consumer. The
        // consumer can be cancelled immediately after send() succeeds.
        if let Some(binding) = &binding {
            self.store.record_worker_receipt(
                &response.request_id,
                session,
                binding,
                "response",
                &response.encode_to_vec(),
            )?;
        }
        let response = if let Some(sender) = sender {
            match sender.send(response) {
                Ok(()) => return Ok(None),
                Err(response) => response,
            }
        } else {
            response
        };
        Ok(Some(LateAdapterResult {
            session: session.into(),
            binding,
            response,
            never_sent: false,
        }))
    }

    pub async fn session_id(&self, domain: &str) -> Option<String> {
        self.sessions
            .lock()
            .await
            .get(domain)
            .filter(|s| !s.endpoint.requests.is_closed())
            .map(|s| s.id.clone())
    }

    pub async fn available(&self, domain: &str) -> bool {
        self.sessions
            .lock()
            .await
            .get(domain)
            .is_some_and(|s| !s.endpoint.requests.is_closed())
    }

    pub async fn supports_feature(&self, domain: &str, feature: &str) -> bool {
        self.sessions
            .lock()
            .await
            .get(domain)
            .is_some_and(|session| {
                !session.endpoint.requests.is_closed() && session.features.contains(feature)
            })
    }

    pub async fn request(
        &self,
        domain: &str,
        operation: &str,
        payload: Value,
    ) -> CoreResult<Value> {
        self.request_bound(domain, None, operation, payload).await
    }

    pub(crate) async fn request_in_session(
        &self,
        domain: &str,
        session: &str,
        operation: &str,
        payload: Value,
    ) -> CoreResult<Value> {
        self.request_bound(domain, Some(session), operation, payload)
            .await
    }

    async fn request_bound(
        &self,
        domain: &str,
        expected_session: Option<&str>,
        operation: &str,
        mut payload: Value,
    ) -> CoreResult<Value> {
        let (session, endpoint, supports_learned_control) = {
            let sessions = self.sessions.lock().await;
            let session = sessions.get(domain).ok_or_else(|| {
                CoreError::ExecutorUnavailable(format!("{domain} adapter is not connected"))
            })?;
            if expected_session.is_some_and(|expected| expected != session.id) {
                return Err(CoreError::ExecutorUnavailable(
                    "Adapter session changed during preparation".into(),
                ));
            }
            (
                session.id.clone(),
                session.endpoint.clone(),
                session.features.contains("application_control_v1"),
            )
        };
        if domain == "native"
            && operation == "execute"
            && payload["action"]["type"] == "set_application_control"
            && !supports_learned_control
        {
            return Err(CoreError::ExecutorUnavailable(
                "The connected native client did not negotiate application_control_v1".into(),
            ));
        }
        let id = if operation == "probe_control" {
            payload
                .get("probe_lease")
                .and_then(|lease| lease.get("id"))
                .and_then(Value::as_str)
                .and_then(|value| Uuid::parse_str(value).ok())
                .map(|value| value.to_string())
                .ok_or_else(|| {
                    CoreError::CapabilityRejected(
                        "Probe request has no valid dispatched lease identity".into(),
                    )
                })?
        } else {
            Uuid::new_v4().to_string()
        };
        let (tx, rx) = oneshot::channel();
        let duration = match operation {
            "discover_interface" => Duration::from_secs(3),
            "reference" => Duration::from_secs(2),
            _ => Duration::from_secs(30),
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
        let application_target = payload
            .as_object_mut()
            .and_then(|p| p.remove("application_target"))
            .map(serde_json::from_value::<crate::application_target::ApplicationTarget>)
            .transpose()?;
        if let Some(target) = &application_target {
            target.validate()?;
        }
        if domain == "native"
            && matches!(
                operation,
                "execute" | "observe_application" | "observe_application_control" | "probe_control"
            )
            && application_target.is_none()
        {
            return Err(CoreError::CapabilityRejected(
                "Missing signed application target".into(),
            ));
        }
        let binding = grant
            .as_ref()
            .map(|grant| {
                Ok::<_, CoreError>(EffectBinding {
                    task_id: grant.run_id.parse().map_err(|_| {
                        CoreError::CapabilityRejected("Invalid run identity".into())
                    })?,
                    action_id: grant.action_id.parse().map_err(|_| {
                        CoreError::CapabilityRejected("Invalid action identity".into())
                    })?,
                    action_digest: grant.action_digest.clone(),
                })
            })
            .transpose()?;
        let request = wire::AdapterRequest {
            application_target: application_target.map(|t| t.to_wire()),
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
        // Admission and enqueue contain no await after the pending record is
        // installed. Otherwise cancellation while waiting for channel space
        // leaves a retired request that the IPC writer can never reconcile.
        let permit = endpoint
            .requests
            .try_reserve()
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => {
                    CoreError::Busy("Adapter dispatch queue is full".into())
                }
                mpsc::error::TrySendError::Closed(_) => {
                    CoreError::ExecutorUnavailable("Adapter disconnected".into())
                }
            })?;
        {
            let mut requests = self.requests.lock().expect("adapter request lock poisoned");
            if requests.pending.len() >= 128
                || requests.pending.len() + requests.retired.len() >= 4096
            {
                return Err(CoreError::Busy(
                    "Adapter request capacity reached; finish or reconcile outstanding requests"
                        .into(),
                ));
            }
            requests.pending.insert(
                id.clone(),
                Pending {
                    identity: RequestIdentity {
                        session,
                        binding,
                        expires_at_unix_ms: request.expires_at_unix_ms,
                    },
                    endpoint: endpoint.clone(),
                    sender: tx,
                },
            );
        }
        let _pending_guard = PendingGuard {
            requests: &self.requests,
            id: id.clone(),
        };
        permit.send(request);
        let result = timeout(duration, async {
            let response = rx
                .await
                .map_err(|_| CoreError::ExecutorUnavailable("Adapter disconnected".into()))?;
            if !response.success {
                return Err(CoreError::ExecutionFailed(
                    crate::redaction::redact_for_persistence(&response.error),
                ));
            }
            let mut value: Value = serde_json::from_str(&response.json)?;
            if domain == "native"
                && matches!(
                    operation,
                    "application_identity"
                        | "observe_application"
                        | "observe_application_control"
                        | "discover_interface"
                        | "probe_control"
                        | "execute"
                )
            {
                let target = crate::application_target::ApplicationTarget::from_wire(
                    response.application_target.as_ref().ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Adapter omitted typed signed identity".into(),
                        )
                    })?,
                )?;
                let object = value
                    .as_object_mut()
                    .ok_or_else(|| CoreError::Protocol("Invalid native result schema".into()))?;
                // The typed response owns identity; unchecked JSON cannot override it.
                object.insert("application_target".into(), serde_json::to_value(target)?);
                object.insert(
                    "observed_process_id".into(),
                    json!(response.observed_process_id),
                );
            }
            Ok::<Value, CoreError>(value)
        })
        .await
        .map_err(|_| CoreError::Timeout("Adapter request expired; no automatic retry.".into()));
        result?
    }

    /// Capture a minimal foreground reference only after the request was
    /// classified as explicitly referential. Adapter results are one-shot,
    /// untrusted planning data and are never cached as authority.
    pub async fn reference(&self, include_page_text: bool) -> Vec<ContextObservation> {
        let native = async {
            let result = self
                .request(
                    "native",
                    "reference",
                    json!({"include_page_text":include_page_text}),
                )
                .await;
            ContextObservation {
                source: "native".into(),
                observed_at_unix_ms: Utc::now().timestamp_millis(),
                state: result.unwrap_or_else(|_| json!({"available":false})),
            }
        };
        let browser = async {
            let result = self
                .request(
                    "browser",
                    "reference",
                    json!({"include_page_text":include_page_text}),
                )
                .await;
            ContextObservation {
                source: "browser".into(),
                observed_at_unix_ms: Utc::now().timestamp_millis(),
                state: result.unwrap_or_else(|_| json!({"available":false})),
            }
        };
        let (native, browser) = tokio::join!(native, browser);
        vec![native, browser]
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
        proposal: &ActionProposal,
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
        if capability.task_id != proposal.task_id || capability.action_id != proposal.id {
            return Err(CoreError::CapabilityRejected(
                "Native grant belongs to another action".into(),
            ));
        }
        let target = crate::application_target::ApplicationTarget::from_proposal(proposal)?;
        if capability.worker_session != self.session_id("native").await {
            return Err(CoreError::CapabilityRejected(
                "Native worker session changed".into(),
            ));
        }
        let mut payload = json!({
            "action":proposal.action,
            "capability":capability,
            "application_target":target,
        });
        if matches!(
            &proposal.action,
            crate::domain::Action::SetApplicationControl { .. }
        ) {
            let metadata = payload.as_object_mut().expect("constructed as an object");
            metadata.insert(
                "application_control_anchor".into(),
                serde_json::from_str(
                    proposal
                        .metadata
                        .get("application_control_anchor")
                        .ok_or_else(|| {
                            CoreError::CapabilityRejected(
                                "Prepared semantic control anchor is missing".into(),
                            )
                        })?,
                )?,
            );
            let fingerprint = proposal
                .metadata
                .get("application_interface_fingerprint")
                .ok_or_else(|| {
                    CoreError::CapabilityRejected(
                        "Prepared application interface fingerprint is missing".into(),
                    )
                })?;
            metadata.insert(
                "application_interface_fingerprint".into(),
                json!(fingerprint),
            );
        }
        let data = self.request("native", "execute", payload).await?;
        Ok(ExecutionReceipt {
            executor: "native-platform-adapter".into(),
            summary: proposal.action.redacted_summary(),
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
        if matches!(proposal.action, Action::OpenApplication { .. }) {
            let target = crate::application_target::ApplicationTarget::from_proposal(proposal)?;
            let value = self
                .0
                .request(
                    "native",
                    "observe_application",
                    json!({"application_target":target}),
                )
                .await?;
            let observed: crate::application_target::ApplicationTarget =
                serde_json::from_value(value["application_target"].clone())?;
            let process_id = value["observed_process_id"]
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .filter(|v| *v > 0)
                .ok_or_else(|| {
                    CoreError::VerificationFailed("Signed application is not running".into())
                })?;
            if observed != target {
                return Err(CoreError::VerificationFailed(
                    "Application changed after approval".into(),
                ));
            }
            return Ok(Observation {
                observed_at: Utc::now(),
                provenance: Provenance::external(
                    ProvenanceSource::OperatingSystem,
                    "signed_application",
                ),
                summary: "Observed the approved signed bundle and running process".into(),
                evidence: vec![Evidence::SignedApplication {
                    target: observed,
                    process_id,
                }],
            });
        }
        if let ExpectedOutcome::ApplicationControlValue {
            target,
            control_id,
            value: _,
        } = &proposal.expected_outcome
        {
            let expected_target =
                crate::application_target::ApplicationTarget::from_proposal(proposal)?;
            if target != &expected_target {
                return Err(CoreError::VerificationFailed(
                    "Prepared control verification target changed".into(),
                ));
            }
            let observed = self
                .0
                .request(
                    "native",
                    "observe_application_control",
                    json!({
                        "application_target": target,
                        "control_id": control_id,
                    }),
                )
                .await?;
            let observed_target: crate::application_target::ApplicationTarget =
                serde_json::from_value(observed["application_target"].clone())?;
            let process_id = observed["observed_process_id"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    CoreError::VerificationFailed(
                        "Control observation omitted its live process identity".into(),
                    )
                })?;
            let observed_control = observed["control_id"]
                .as_str()
                .ok_or_else(|| CoreError::VerificationFailed("Control ID is missing".into()))?;
            let role = observed["role"]
                .as_str()
                .ok_or_else(|| CoreError::VerificationFailed("Control role is missing".into()))?;
            let anchor: Value = serde_json::from_str(
                proposal
                    .metadata
                    .get("application_control_anchor")
                    .ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Prepared semantic control anchor is missing".into(),
                        )
                    })?,
            )?;
            if observed_target != *target
                || observed_control != control_id
                || !matches!(role, "slider" | "checkbox" | "switch")
                || observed["label"] != anchor["label"]
                || observed["ancestors"] != anchor["ancestors"]
                || anchor["role"] != role
            {
                return Err(CoreError::VerificationFailed(
                    "The signed application or learned control changed before verification".into(),
                ));
            }
            let value: crate::domain::ApplicationControlValue =
                serde_json::from_value(observed["value"].clone())?;
            value.validate().map_err(CoreError::VerificationFailed)?;
            let observed_at = observed["observed_at"]
                .as_str()
                .and_then(|value| chrono::DateTime::parse_from_rfc3339(value).ok())
                .map(|value| value.to_utc())
                .ok_or_else(|| {
                    CoreError::VerificationFailed(
                        "Control observation omitted its readback time".into(),
                    )
                })?;
            return Ok(Observation {
                observed_at,
                provenance: Provenance::external(
                    ProvenanceSource::OperatingSystem,
                    "application_accessibility_control",
                ),
                summary: "Observed the exact learned control on the signed foreground application"
                    .into(),
                evidence: vec![Evidence::ApplicationControlValue {
                    target: observed_target,
                    process_id,
                    control_id: observed_control.to_owned(),
                    value,
                }],
            });
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn full_dispatch_queue_does_not_create_unsettleable_cancellation_records() {
        let bridge = Arc::new(AdapterBridge::new(
            crate::storage::LocalStore::deferred(std::path::Path::new(":memory:")).unwrap(),
        ));
        let (tx, _rx) = mpsc::channel(1);
        tx.try_send(wire::AdapterRequest::default()).unwrap();
        let (control, mut cancellations) = mpsc::channel(1);
        bridge
            .register(
                "native",
                "session",
                wire::ClientKind::Macos as i32,
                AdapterEndpoint {
                    requests: tx,
                    cancellations: Some(control),
                    terminate: tokio::sync::watch::channel(false).0,
                },
                &[],
            )
            .await
            .unwrap();
        assert!(matches!(
            bridge.request("native", "reference", json!({})).await,
            Err(CoreError::Busy(_))
        ));
        let requests = bridge.requests.lock().unwrap();
        assert!(requests.pending.is_empty());
        assert!(requests.retired.is_empty());
        assert!(cancellations.try_recv().is_err());
    }

    #[tokio::test]
    async fn native_peers_without_negotiated_control_support_never_receive_control_actions() {
        let bridge = AdapterBridge::new(
            crate::storage::LocalStore::deferred(std::path::Path::new(":memory:")).unwrap(),
        );
        let (tx, mut rx) = mpsc::channel(1);
        bridge
            .register(
                "native",
                "older-mac-session",
                wire::ClientKind::Macos as i32,
                AdapterEndpoint {
                    requests: tx,
                    cancellations: None,
                    terminate: tokio::sync::watch::channel(false).0,
                },
                &[],
            )
            .await
            .unwrap();
        assert!(
            !bridge
                .supports_feature("native", "application_control_v1")
                .await
        );
        assert!(
            bridge
                .request(
                    "native",
                    "execute",
                    json!({"action":{"type":"set_application_control"}}),
                )
                .await
                .is_err()
        );
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn cancellation_ack_and_late_reply_preserve_the_session_and_reject_other_peers() {
        let bridge = Arc::new(AdapterBridge::new(
            crate::storage::LocalStore::deferred(std::path::Path::new(":memory:")).unwrap(),
        ));
        let (tx, mut rx) = mpsc::channel(2);
        let (control, mut cancellations) = mpsc::channel(2);
        let (terminate, _terminated) = tokio::sync::watch::channel(false);
        bridge
            .register(
                "native",
                "session",
                wire::ClientKind::Macos as i32,
                AdapterEndpoint {
                    requests: tx,
                    cancellations: Some(control),
                    terminate,
                },
                &[],
            )
            .await
            .unwrap();
        let caller = bridge.clone();
        let task =
            tokio::spawn(async move { caller.request("native", "reference", json!({})).await });
        let request = rx.recv().await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let cancellation = cancellations.recv().await.unwrap();
        assert_eq!(cancellation.request_id, request.request_id);
        let response = wire::AdapterResult {
            request_id: request.request_id.clone(),
            success: true,
            json: "{}".into(),
            ..Default::default()
        };
        assert!(
            bridge
                .complete("another-session", response.clone())
                .await
                .is_err()
        );
        assert!(
            bridge
                .acknowledge_cancel("another-session", &request.request_id)
                .is_err()
        );
        bridge
            .acknowledge_cancel("session", &request.request_id)
            .unwrap();
        assert!(
            bridge
                .complete("session", response.clone())
                .await
                .unwrap()
                .is_some()
        );
        assert!(bridge.complete("session", response).await.is_err());
        assert!(bridge.requests.lock().unwrap().retired.is_empty());

        let caller = bridge.clone();
        let next =
            tokio::spawn(async move { caller.request("native", "reference", json!({})).await });
        let request = rx.recv().await.unwrap();
        bridge
            .complete(
                "session",
                wire::AdapterResult {
                    request_id: request.request_id,
                    success: true,
                    json: "{\"live\":true}".into(),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(next.await.unwrap().unwrap()["live"], true);
    }

    #[tokio::test]
    async fn a_cancelled_queued_request_is_suppressed_before_ipc_dispatch() {
        let bridge = Arc::new(AdapterBridge::new(
            crate::storage::LocalStore::deferred(std::path::Path::new(":memory:")).unwrap(),
        ));
        let (tx, mut rx) = mpsc::channel(1);
        bridge
            .register(
                "native",
                "session",
                wire::ClientKind::Macos as i32,
                AdapterEndpoint {
                    requests: tx,
                    cancellations: None,
                    terminate: tokio::sync::watch::channel(false).0,
                },
                &[],
            )
            .await
            .unwrap();
        let caller = bridge.clone();
        let task =
            tokio::spawn(async move { caller.request("native", "reference", json!({})).await });
        let queued = rx.recv().await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(!bridge.is_pending("session", &queued.request_id));
        let suppressed = bridge
            .cancelled_before_send("session", &queued.request_id)
            .unwrap()
            .unwrap();
        assert!(suppressed.never_sent);
        assert!(bridge.requests.lock().unwrap().retired.is_empty());
    }
    #[tokio::test]
    async fn cancelled_requests_accept_correlated_late_replies_without_reviving_waiters() {
        let bridge = Arc::new(AdapterBridge::new(
            crate::storage::LocalStore::deferred(std::path::Path::new(":memory:")).unwrap(),
        ));
        let (tx, mut rx) = mpsc::channel(1);
        bridge
            .register(
                "native",
                "session",
                wire::ClientKind::Macos as i32,
                AdapterEndpoint {
                    requests: tx,
                    cancellations: None,
                    terminate: tokio::sync::watch::channel(false).0,
                },
                &[],
            )
            .await
            .unwrap();
        let pending_bridge = bridge.clone();
        let task = tokio::spawn(async move {
            pending_bridge
                .request(
                    "native",
                    "application_identity",
                    json!({"application":"com.example.App"}),
                )
                .await
        });
        let request = rx.recv().await.unwrap();
        assert_eq!(bridge.requests.lock().unwrap().pending.len(), 1);
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(bridge.requests.lock().unwrap().pending.is_empty());
        assert!(
            bridge
                .complete(
                    "session",
                    wire::AdapterResult {
                        request_id: request.request_id,
                        success: true,
                        json: "{}".into(),
                        ..Default::default()
                    }
                )
                .await
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn application_identity_cannot_be_supplied_in_unchecked_json() {
        let bridge = Arc::new(AdapterBridge::new(
            crate::storage::LocalStore::deferred(std::path::Path::new(":memory:")).unwrap(),
        ));
        let (tx, mut rx) = mpsc::channel(1);
        bridge
            .register(
                "native",
                "session",
                wire::ClientKind::Macos as i32,
                AdapterEndpoint {
                    requests: tx,
                    cancellations: None,
                    terminate: tokio::sync::watch::channel(false).0,
                },
                &[],
            )
            .await
            .unwrap();
        let pending_bridge = bridge.clone();
        let task = tokio::spawn(async move {
            pending_bridge
                .request(
                    "native",
                    "application_identity",
                    json!({"application":"com.example.App"}),
                )
                .await
        });
        let request = rx.recv().await.unwrap();
        bridge.complete("session", wire::AdapterResult { request_id:request.request_id, success:true,
            json:json!({"application_target":{"identifier":"com.example.App"},"observed_process_id":1}).to_string(), ..Default::default() }).await.unwrap();
        assert!(
            task.await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("typed signed identity")
        );
    }

    #[tokio::test]
    #[ignore = "release-mode adapter bridge latency measurement"]
    async fn explicit_reference_bridge_latency() {
        use std::time::Instant;

        const SAMPLES: usize = 1000;
        let store = crate::storage::LocalStore::deferred(std::path::Path::new(":memory:")).unwrap();
        let bridge = Arc::new(AdapterBridge::new(store));
        let (native_tx, mut native_rx) = mpsc::channel(8);
        let (browser_tx, mut browser_rx) = mpsc::channel(8);
        bridge
            .register(
                "native",
                "reference-benchmark-native",
                wire::ClientKind::Macos as i32,
                AdapterEndpoint {
                    requests: native_tx,
                    cancellations: None,
                    terminate: tokio::sync::watch::channel(false).0,
                },
                &[],
            )
            .await
            .unwrap();
        bridge
            .register(
                "browser",
                "reference-benchmark-browser",
                wire::ClientKind::Browser as i32,
                AdapterEndpoint {
                    requests: browser_tx,
                    cancellations: None,
                    terminate: tokio::sync::watch::channel(false).0,
                },
                &[],
            )
            .await
            .unwrap();
        let native_bridge = bridge.clone();
        let native = tokio::spawn(async move {
            while let Some(request) = native_rx.recv().await {
                native_bridge
                    .complete(
                        "reference-benchmark-native",
                        wire::AdapterResult {
                            request_id: request.request_id,
                            success: true,
                            json: json!({"available":true,"active_application":"com.apple.finder","application_name":"Finder","active_window":"Downloads","selected_text":"fixture"}).to_string(),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
            }
        });
        let browser_bridge = bridge.clone();
        let browser = tokio::spawn(async move {
            while let Some(request) = browser_rx.recv().await {
                browser_bridge
                    .complete(
                        "reference-benchmark-browser",
                        wire::AdapterResult {
                            request_id: request.request_id,
                            success: true,
                            json: json!({"available":false}).to_string(),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
            }
        });

        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let started = Instant::now();
            let observations = bridge.reference(false).await;
            let selected = crate::context::select_live_reference(observations);
            assert_eq!(selected.source, "current_reference_native");
            samples.push(started.elapsed().as_nanos());
        }
        samples.sort_unstable();
        let percentile = |percent: usize| {
            let index = (samples.len() * percent).div_ceil(100).saturating_sub(1);
            samples[index]
        };
        eprintln!(
            "explicit reference bridge samples={SAMPLES} p50_ns={} p95_ns={} p99_ns={}",
            percentile(50),
            percentile(95),
            percentile(99)
        );
        native.abort();
        browser.abort();
    }
}
