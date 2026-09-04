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
}

impl CapabilityBroker {
    pub async fn issue(
        &self,
        proposal: &ActionProposal,
        domain: ExecutionDomain,
    ) -> CoreResult<CapabilityGrant> {
        let (resource, operations) = requirements(&proposal.action)?;
        let grant = CapabilityGrant {
            id: Uuid::new_v4(),
            task_id: proposal.task_id,
            action_id: proposal.id,
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

    pub async fn revoke_task(&self, task_id: Uuid) {
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
