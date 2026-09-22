//! Broker-owned v2 contracts. Models may propose actions, never these grants.
use std::{collections::BTreeSet, path::PathBuf};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::{Action, Provenance};
use crate::{CoreError, CoreResult};

pub const POLICY_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    Read,
    Create,
    Modify,
    Delete,
    ExternalCommitment,
    ControlApplication,
    ExecuteCode,
    ReleaseData,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceScope {
    pub root: PathBuf,
    pub effects: BTreeSet<Effect>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunContract {
    pub run_id: Uuid,
    pub principal: String,
    pub resources: Vec<ResourceScope>,
    /// Exact provider destinations explicitly approved by the user for this run.
    pub data_destinations: BTreeSet<String>,
    pub expires_at: DateTime<Utc>,
    pub max_steps: u32,
    pub max_repairs: u32,
    pub policy_version: u32,
}

impl RunContract {
    pub fn local(run_id: Uuid) -> Self {
        Self {
            run_id,
            principal: "local-user".into(),
            resources: Vec::new(),
            data_destinations: BTreeSet::new(),
            expires_at: Utc::now() + Duration::hours(2),
            max_steps: 32,
            max_repairs: 2,
            policy_version: POLICY_VERSION,
        }
    }

    pub fn validate(&self, run_id: Uuid) -> CoreResult<()> {
        if self.run_id != run_id
            || self.expires_at <= Utc::now()
            || self.policy_version != POLICY_VERSION
            || self.max_steps == 0
            || self.max_steps > 32
            || self.max_repairs > 2
            || self.resources.len() > 32
            || self.resources.iter().any(|scope| {
                crate::resources::validate_file_path(&scope.root).is_err()
                    || scope.effects.is_empty()
                    || scope
                        .effects
                        .iter()
                        .any(|effect| !matches!(effect, Effect::Read | Effect::Create))
            })
        {
            return Err(CoreError::PermissionRequired(
                "Task scope expired or is invalid; authorize a new scope.".into(),
            ));
        }
        Ok(())
    }

    /// Only reads and non-overwriting creation can inherit a folder grant.
    /// Destruction and external effects always require a prepared-action approval.
    pub fn covers(&self, action: &Action) -> bool {
        if self.validate(self.run_id).is_err() {
            return false;
        }
        let (path, effect) = match action {
            Action::ReadFile { path, .. } => (path, Effect::Read),
            Action::WriteFile {
                path,
                overwrite: false,
                ..
            }
            | Action::CreateFolder { path } => (path, Effect::Create),
            _ => return false,
        };
        if crate::resources::validate_file_path(path).is_err() {
            return false;
        }
        self.resources
            .iter()
            .any(|scope| path.starts_with(&scope.root) && scope.effects.contains(&effect))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sensitivity {
    Public,
    Private,
    Restricted,
    Secret,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataLabel {
    pub sensitivity: Sensitivity,
    pub scope_id: Uuid,
    pub source_ids: BTreeSet<String>,
}

impl DataLabel {
    pub fn private(scope_id: Uuid, source: String) -> Self {
        Self {
            sensitivity: Sensitivity::Private,
            scope_id,
            source_ids: BTreeSet::from([source]),
        }
    }
    pub fn derive(&self, other: &Self) -> CoreResult<Self> {
        if self.scope_id != other.scope_id {
            return Err(CoreError::PolicyDenied(
                "Cross-scope context combination requires explicit permission.".into(),
            ));
        }
        Ok(Self {
            sensitivity: self.sensitivity.max(other.sensitivity),
            scope_id: self.scope_id,
            source_ids: self.source_ids.union(&other.source_ids).cloned().collect(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    pub id: Uuid,
    pub label: DataLabel,
    pub provenance: Provenance,
    pub observed_at: DateTime<Utc>,
    pub content_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileIdentity {
    pub key: String,
    pub size: u64,
    pub modified: String,
    pub directory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PreparedTarget {
    File {
        path: PathBuf,
        before: Option<FileIdentity>,
    },
    Browser {
        document: crate::browser_target::BrowserTarget,
    },
    Application {
        identifier: String,
    },
    Network {
        url: String,
    },
    User,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionIntent {
    pub schema_version: u32,
    pub tool_version: u32,
    pub proposal: crate::domain::ActionProposal,
    pub input_refs: BTreeSet<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedAction {
    pub intent: ActionIntent,
    pub action_digest: String,
    pub policy_version: u32,
    pub target: PreparedTarget,
    pub effects: BTreeSet<Effect>,
    pub preview: String,
    pub prepared_at: DateTime<Utc>,
}

impl PreparedAction {
    pub fn new(
        proposal: &crate::domain::ActionProposal,
        input_refs: BTreeSet<Uuid>,
    ) -> CoreResult<Self> {
        crate::features::validate(&proposal.action)?;
        let file = |path: &PathBuf| -> CoreResult<PreparedTarget> {
            let before = proposal.metadata.get("file_precondition").ok_or_else(|| {
                CoreError::CapabilityRejected("File preparation identity is missing".into())
            })?;
            Ok(PreparedTarget::File {
                path: path.clone(),
                before: serde_json::from_str(before)?,
            })
        };
        let (target, effects) = match &proposal.action {
            Action::ReadFile { path, .. } => (file(path)?, BTreeSet::from([Effect::Read])),
            Action::WriteFile {
                path, overwrite, ..
            } => (
                file(path)?,
                BTreeSet::from([if *overwrite {
                    Effect::Modify
                } else {
                    Effect::Create
                }]),
            ),
            Action::CreateFolder { path } => (file(path)?, BTreeSet::from([Effect::Create])),
            Action::FetchPublic { url, .. } => (
                PreparedTarget::Network { url: url.clone() },
                BTreeSet::from([Effect::Read, Effect::ReleaseData]),
            ),
            Action::OpenApplication { application } => (
                PreparedTarget::Application {
                    identifier: application.clone(),
                },
                BTreeSet::from([Effect::ControlApplication]),
            ),
            Action::NavigateUrl { .. } => {
                let document: crate::browser_target::BrowserTarget = serde_json::from_str(
                    proposal.metadata.get("browser_target").ok_or_else(|| {
                        CoreError::CapabilityRejected(
                            "Browser preparation identity is missing".into(),
                        )
                    })?,
                )?;
                document.validate()?;
                (
                    PreparedTarget::Browser { document },
                    BTreeSet::from([Effect::ExternalCommitment, Effect::ReleaseData]),
                )
            }
            Action::AskUser { .. } => (PreparedTarget::User, BTreeSet::new()),
            _ => {
                return Err(CoreError::ExecutorUnavailable(
                    "No prepared feature contract is installed".into(),
                ));
            }
        };
        Ok(Self {
            intent: ActionIntent {
                schema_version: 2,
                tool_version: 2,
                proposal: proposal.clone(),
                input_refs,
            },
            action_digest: crate::policy::approval_digest(proposal)?,
            policy_version: POLICY_VERSION,
            target,
            effects,
            preview: crate::policy::action_preview(&proposal.action)?,
            prepared_at: Utc::now(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Confirmed,
    Failed,
    Cancelled,
    Uncertain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub action_id: Uuid,
    pub tool: String,
    pub verdict: Verdict,
    pub summary: String,
    pub output: serde_json::Value,
    pub label: DataLabel,
    pub observed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRecord {
    pub run_id: Uuid,
    pub action_id: Uuid,
    pub target: String,
    pub action_digest: String,
    pub expected: crate::domain::ExpectedOutcome,
    pub observed_at: DateTime<Utc>,
    pub verdict: Verdict,
    pub evidence: Vec<crate::observation::Evidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRecord {
    pub approval_id: Uuid,
    pub task_id: Uuid,
    pub action_id: Uuid,
    pub digest: String,
    pub explanation: String,
    pub resource: String,
    pub risk: crate::policy::RiskLevel,
    pub expires_at: DateTime<Utc>,
    pub reversible: bool,
    pub requires_native_authentication: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scope_never_grants_overwrite_or_external_effects() {
        let mut contract = RunContract::local(Uuid::new_v4());
        contract.resources.push(ResourceScope {
            root: "/work".into(),
            effects: BTreeSet::from([Effect::Read, Effect::Create]),
        });
        assert!(contract.covers(&Action::ReadFile {
            path: "/work/a".into(),
            max_bytes: 64
        }));
        assert!(!contract.covers(&Action::ReadFile {
            path: "/work-other/a".into(),
            max_bytes: 64
        }));
        assert!(!contract.covers(&Action::ReadFile {
            path: "/work/../private/key".into(),
            max_bytes: 64
        }));
        assert!(!contract.covers(&Action::WriteFile {
            path: "/work/a".into(),
            content: "x".into(),
            overwrite: true
        }));
        contract.expires_at = Utc::now() - Duration::seconds(1);
        assert!(!contract.covers(&Action::ReadFile {
            path: "/work/a".into(),
            max_bytes: 64
        }));
    }
    #[test]
    fn derived_data_retains_restrictions() {
        let id = Uuid::new_v4();
        let a = DataLabel::private(id, "file".into());
        let mut b = DataLabel::private(id, "history".into());
        b.sensitivity = Sensitivity::Restricted;
        let derived = a.derive(&b).unwrap();
        assert_eq!(derived.sensitivity, Sensitivity::Restricted);
        assert_eq!(derived.source_ids.len(), 2);
        assert!(
            a.derive(&DataLabel::private(Uuid::new_v4(), "other".into()))
                .is_err()
        );
    }
}
