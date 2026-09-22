use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::domain::{Action, ActionProposal, ExecutionDomain};
use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityOperation {
    Read,
    Write,
    Create,
    Delete,
    Execute,
    Observe,
    Control,
    Network,
    Capture,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CapabilityResource {
    NetworkRoute {
        url: String,
    },
    File {
        canonical_path: String,
    },
    FilePair {
        source: String,
        destination: String,
    },
    Application {
        identifier: String,
    },
    BrowserOrigin {
        origin: String,
    },
    Command {
        executable: String,
        working_directory: Option<String>,
        network: bool,
    },
    Setting {
        namespace: String,
        key: String,
    },
    UserInteraction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityGrant {
    pub id: Uuid,
    pub task_id: Uuid,
    pub action_id: Uuid,
    pub action_digest: String,
    pub policy_version: u32,
    pub worker_session: Option<String>,
    pub domain: ExecutionDomain,
    pub resource: CapabilityResource,
    pub operations: BTreeSet<CapabilityOperation>,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub remaining_uses: u32,
    pub revoked: bool,
}

#[derive(Debug, Default, Clone)]
pub struct CapabilityBroker {
    grants: Arc<RwLock<HashMap<Uuid, CapabilityGrant>>>,
    cancelled: Arc<tokio::sync::Mutex<BTreeSet<Uuid>>>,
}

impl CapabilityBroker {
    pub async fn issue(
        &self,
        proposal: &ActionProposal,
        domain: ExecutionDomain,
    ) -> CoreResult<CapabilityGrant> {
        let cancelled = self.cancelled.lock().await;
        if cancelled.contains(&proposal.task_id) {
            return Err(CoreError::Cancelled);
        }
        let (resource, operations) = requirements(&proposal.action)?;
        let grant = CapabilityGrant {
            id: Uuid::new_v4(),
            task_id: proposal.task_id,
            action_id: proposal.id,
            action_digest: crate::policy::approval_digest(proposal)?,
            policy_version: crate::contracts::POLICY_VERSION,
            worker_session: None,
            domain,
            resource,
            operations,
            issued_at: Utc::now(),
            expires_at: Utc::now() + Duration::minutes(5),
            remaining_uses: 1,
            revoked: false,
        };
        self.grants.write().await.insert(grant.id, grant.clone());
        Ok(grant)
    }

    pub async fn bind_worker(&self, id: Uuid, session: String) -> CoreResult<CapabilityGrant> {
        let mut grants = self.grants.write().await;
        let grant = grants
            .get_mut(&id)
            .ok_or_else(|| CoreError::CapabilityRejected("Unknown grant".into()))?;
        if grant.remaining_uses != 1 || grant.worker_session.is_some() {
            return Err(CoreError::CapabilityRejected(
                "Grant cannot be rebound".into(),
            ));
        }
        grant.worker_session = Some(session);
        Ok(grant.clone())
    }

    pub async fn consume(
        &self,
        grant_id: Uuid,
        task_id: Uuid,
        action_id: Uuid,
        expected_domain: ExecutionDomain,
    ) -> CoreResult<CapabilityGrant> {
        let mut grants = self.grants.write().await;
        let grant = grants
            .get_mut(&grant_id)
            .ok_or_else(|| CoreError::CapabilityRejected("unknown capability".into()))?;
        if grant.revoked {
            return Err(CoreError::CapabilityRejected(
                "capability was revoked".into(),
            ));
        }
        if grant.expires_at <= Utc::now() {
            return Err(CoreError::CapabilityRejected("capability expired".into()));
        }
        if grant.task_id != task_id || grant.action_id != action_id {
            return Err(CoreError::CapabilityRejected(
                "capability is bound to another task or action".into(),
            ));
        }
        if grant.domain != expected_domain {
            return Err(CoreError::CapabilityRejected(
                "capability is bound to another executor domain".into(),
            ));
        }
        if grant.remaining_uses == 0 {
            return Err(CoreError::CapabilityRejected(
                "capability was already used".into(),
            ));
        }
        grant.remaining_uses -= 1;
        Ok(grant.clone())
    }

    pub async fn reopen_run(&self, task_id: Uuid) {
        self.cancelled.lock().await.remove(&task_id);
    }

    pub async fn revoke_task(&self, task_id: Uuid) {
        let mut cancelled = self.cancelled.lock().await;
        cancelled.insert(task_id);
        for grant in self.grants.write().await.values_mut() {
            if grant.task_id == task_id {
                grant.revoked = true;
            }
        }
    }
}

fn requirements(
    action: &Action,
) -> CoreResult<(CapabilityResource, BTreeSet<CapabilityOperation>)> {
    use CapabilityOperation as Op;

    let (resource, operations) = match action {
        Action::FetchPublic { url, .. } => (
            CapabilityResource::NetworkRoute { url: url.clone() },
            BTreeSet::from([Op::Read, Op::Network]),
        ),
        Action::ReadFile { path, .. } => (
            CapabilityResource::File {
                canonical_path: path.to_string_lossy().into_owned(),
            },
            BTreeSet::from([Op::Read]),
        ),
        Action::WriteFile { path, .. } => (
            CapabilityResource::File {
                canonical_path: path.to_string_lossy().into_owned(),
            },
            BTreeSet::from([Op::Write]),
        ),
        Action::MoveFile {
            source,
            destination,
        } => (
            CapabilityResource::FilePair {
                source: source.to_string_lossy().into_owned(),
                destination: destination.to_string_lossy().into_owned(),
            },
            BTreeSet::from([Op::Read, Op::Write, Op::Delete]),
        ),
        Action::DeleteFile { path } => (
            CapabilityResource::File {
                canonical_path: path.to_string_lossy().into_owned(),
            },
            BTreeSet::from([Op::Delete]),
        ),
        Action::CreateFolder { path } => (
            CapabilityResource::File {
                canonical_path: path.to_string_lossy().into_owned(),
            },
            BTreeSet::from([Op::Create]),
        ),
        Action::ClickElement { application, .. } | Action::TypeText { application, .. }
            if application.starts_with("browser:") =>
        {
            (
                CapabilityResource::BrowserOrigin {
                    origin: origin_from_url(&application[8..])?,
                },
                BTreeSet::from([Op::Observe, Op::Control]),
            )
        }
        Action::OpenApplication { application }
        | Action::CloseApplication { application }
        | Action::ClickElement { application, .. }
        | Action::TypeText { application, .. }
        | Action::PressShortcut { application, .. }
        | Action::SendMessage { application, .. } => (
            CapabilityResource::Application {
                identifier: application.clone(),
            },
            BTreeSet::from([Op::Observe, Op::Control]),
        ),
        Action::NavigateUrl { url, .. } | Action::DownloadFile { url, .. } => (
            CapabilityResource::BrowserOrigin {
                origin: origin_from_url(url)?,
            },
            BTreeSet::from([Op::Network, Op::Control]),
        ),
        Action::UploadFile {
            destination_origin, ..
        }
        | Action::SubmitForm {
            origin: destination_origin,
            ..
        } => (
            CapabilityResource::BrowserOrigin {
                origin: destination_origin.clone(),
            },
            BTreeSet::from([Op::Network, Op::Control]),
        ),
        Action::RunCommand {
            program,
            working_directory,
            network,
            ..
        } => {
            let mut operations = BTreeSet::from([Op::Execute]);
            if *network {
                operations.insert(Op::Network);
            }
            (
                CapabilityResource::Command {
                    executable: program.clone(),
                    working_directory: working_directory
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                    network: *network,
                },
                operations,
            )
        }
        Action::InstallApplication { source } => (
            CapabilityResource::Application {
                identifier: source.clone(),
            },
            BTreeSet::from([Op::Execute, Op::Write]),
        ),
        Action::ChangeSetting { namespace, key, .. } => (
            CapabilityResource::Setting {
                namespace: namespace.clone(),
                key: key.clone(),
            },
            BTreeSet::from([Op::Write]),
        ),
        Action::WaitForCondition { .. } => (
            CapabilityResource::UserInteraction,
            BTreeSet::from([Op::Observe]),
        ),
        Action::AskUser { .. } => (
            CapabilityResource::UserInteraction,
            BTreeSet::from([Op::Control]),
        ),
    };
    Ok((resource, operations))
}

fn origin_from_url(url: &str) -> CoreResult<String> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| CoreError::InvalidAction("URL must include a scheme".into()))?;
    if !matches!(scheme, "https" | "http") {
        return Err(CoreError::InvalidAction(
            "browser URL must use HTTP or HTTPS".into(),
        ));
    }
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| CoreError::InvalidAction("URL must include a host".into()))?;
    if authority.contains('@') {
        return Err(CoreError::InvalidAction(
            "credential-bearing URLs are prohibited".into(),
        ));
    }
    Ok(format!("{scheme}://{authority}"))
}

pub fn grant_to_wire(
    grant: &CapabilityGrant,
) -> CoreResult<sage_protocol::sage::ipc::v2::ExecutionGrant> {
    use sage_protocol::sage::ipc::v2::{ExecutionGrant, FilePair, execution_grant::Resource};
    let resource = match &grant.resource {
        CapabilityResource::File { canonical_path } => Resource::File(canonical_path.clone()),
        CapabilityResource::FilePair {
            source,
            destination,
        } => Resource::FilePair(FilePair {
            source: source.clone(),
            destination: destination.clone(),
        }),
        CapabilityResource::Application { identifier } => Resource::Application(identifier.clone()),
        CapabilityResource::BrowserOrigin { origin } => Resource::BrowserOrigin(origin.clone()),
        CapabilityResource::UserInteraction => Resource::UserInteraction(true),
        _ => {
            return Err(CoreError::CapabilityRejected(
                "Unqualified worker resource".into(),
            ));
        }
    };
    Ok(ExecutionGrant {
        grant_id: grant.id.to_string(),
        run_id: grant.task_id.to_string(),
        action_id: grant.action_id.to_string(),
        action_digest: grant.action_digest.clone(),
        policy_version: grant.policy_version,
        worker_session: grant.worker_session.clone().unwrap_or_default(),
        expires_at_unix_ms: grant.expires_at.timestamp_millis(),
        domain: serde_json::to_value(grant.domain)?
            .as_str()
            .unwrap_or_default()
            .into(),
        operations: grant
            .operations
            .iter()
            .map(|operation| {
                serde_json::to_value(operation).map(|v| v.as_str().unwrap_or_default().to_string())
            })
            .collect::<Result<_, _>>()?,
        resource: Some(resource),
    })
}

/// Compatibility representation inside an adapter only. Authority on the wire
/// is typed and mandatory; no JSON field supplied by a tool can replace it.
pub fn wire_grant_json(
    grant: &sage_protocol::sage::ipc::v2::ExecutionGrant,
) -> CoreResult<serde_json::Value> {
    use sage_protocol::sage::ipc::v2::execution_grant::Resource;
    use serde_json::json;
    if grant.policy_version != crate::contracts::POLICY_VERSION
        || grant.expires_at_unix_ms <= Utc::now().timestamp_millis()
        || grant.worker_session.is_empty()
        || grant.action_digest.len() != 64
    {
        return Err(CoreError::CapabilityRejected("Invalid typed grant".into()));
    }
    let resource = match grant.resource.as_ref() {
        Some(Resource::File(path)) => json!({"kind":"file","canonical_path":path}),
        Some(Resource::FilePair(pair)) => {
            json!({"kind":"file_pair","source":pair.source,"destination":pair.destination})
        }
        Some(Resource::Application(identifier)) => {
            json!({"kind":"application","identifier":identifier})
        }
        Some(Resource::BrowserOrigin(origin)) => json!({"kind":"browser_origin","origin":origin}),
        Some(Resource::UserInteraction(true)) => json!({"kind":"user_interaction"}),
        _ => {
            return Err(CoreError::CapabilityRejected(
                "Missing typed resource".into(),
            ));
        }
    };
    Ok(
        json!({"id":grant.grant_id,"task_id":grant.run_id,"action_id":grant.action_id,"action_digest":grant.action_digest,
        "policy_version":grant.policy_version,"worker_session":grant.worker_session,"domain":grant.domain,
        "expires_at":chrono::DateTime::from_timestamp_millis(grant.expires_at_unix_ms).ok_or_else(||CoreError::CapabilityRejected("Invalid expiry".into()))?.to_rfc3339(),
        "operations":grant.operations,"resource":resource,"remaining_uses":0,"revoked":false}),
    )
}

#[cfg(test)]
mod v2_tests {
    use super::*;
    fn proposal() -> ActionProposal {
        ActionProposal {
            id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            action: Action::ReadFile {
                path: "/work/a".into(),
                max_bytes: 128,
            },
            expected_outcome: crate::domain::ExpectedOutcome::UserAnswered,
            target_resource: "/work/a".into(),
            provenance: crate::domain::Provenance::model(vec![]),
            metadata: Default::default(),
        }
    }
    #[tokio::test]
    async fn capabilities_are_single_use_worker_bound_and_revoked_with_the_run() {
        let broker = CapabilityBroker::default();
        let p = proposal();
        let grant = broker.issue(&p, ExecutionDomain::Native).await.unwrap();
        broker.bind_worker(grant.id, "one".into()).await.unwrap();
        assert!(broker.bind_worker(grant.id, "two".into()).await.is_err());
        assert!(
            broker
                .consume(grant.id, p.task_id, Uuid::new_v4(), ExecutionDomain::Native)
                .await
                .is_err()
        );
        assert!(
            broker
                .consume(grant.id, p.task_id, p.id, ExecutionDomain::Browser)
                .await
                .is_err()
        );
        let used = broker
            .consume(grant.id, p.task_id, p.id, ExecutionDomain::Native)
            .await
            .unwrap();
        assert_eq!(used.remaining_uses, 0);
        assert_eq!(used.worker_session.as_deref(), Some("one"));
        assert!(
            broker
                .consume(grant.id, p.task_id, p.id, ExecutionDomain::Native)
                .await
                .is_err()
        );
        broker.revoke_task(p.task_id).await;
        assert!(broker.issue(&p, ExecutionDomain::Native).await.is_err());
    }
    #[tokio::test]
    async fn every_cancel_issue_interleaving_leaves_no_live_authority() {
        for _ in 0..128 {
            let broker = CapabilityBroker::default();
            let p = proposal();
            let (grant, _) = tokio::join!(
                broker.issue(&p, ExecutionDomain::Native),
                broker.revoke_task(p.task_id)
            );
            if let Ok(grant) = grant {
                assert!(
                    broker
                        .consume(grant.id, p.task_id, p.id, ExecutionDomain::Native)
                        .await
                        .is_err()
                );
            }
            assert!(broker.issue(&p, ExecutionDomain::Native).await.is_err());
        }
    }
    #[test]
    fn approval_digest_changes_with_payload_target_and_browser_identity() {
        let p = proposal();
        let original = crate::policy::approval_digest(&p).unwrap();
        let mut changed = p.clone();
        changed.action = Action::ReadFile {
            path: "/work/b".into(),
            max_bytes: 128,
        };
        assert_ne!(original, crate::policy::approval_digest(&changed).unwrap());
        let mut changed = p;
        changed
            .metadata
            .insert("browser_target".into(), "new-document".into());
        assert_ne!(original, crate::policy::approval_digest(&changed).unwrap());
    }
}
