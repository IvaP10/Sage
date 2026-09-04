use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::domain::{Action, ActionProposal, TrustClass};
use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Safe,
    Sensitive,
    Consequential,
    Destructive,
    Privileged,
    Prohibited,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    Allow {
        risk: RiskLevel,
    },
    RequireApproval {
        risk: RiskLevel,
        explanation: String,
        digest: String,
    },
    Deny {
        risk: RiskLevel,
        reason: String,
    },
}

impl PolicyDecision {
    pub fn risk(&self) -> RiskLevel {
        match self {
            Self::Allow { risk } | Self::RequireApproval { risk, .. } | Self::Deny { risk, .. } => {
                *risk
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PolicyContext {
    pub task_request: String,
    pub has_fresh_native_authentication: bool,
    pub is_recovery_attempt: bool,
}

#[derive(Debug, Default)]
pub struct PolicyEngine;

impl PolicyEngine {
    pub fn evaluate(
        &self,
        proposal: &ActionProposal,
        context: &PolicyContext,
    ) -> CoreResult<PolicyDecision> {
        self.validate_schema_constraints(proposal)?;

        if proposal.provenance.trust == TrustClass::UntrustedExternalContent {
            return Ok(PolicyDecision::Deny {
                risk: RiskLevel::Prohibited,
                reason: "external content cannot directly propose executable actions".into(),
            });
        }

        if let Some(reason) = prohibited_reason(&proposal.action) {
            return Ok(PolicyDecision::Deny {
                risk: RiskLevel::Prohibited,
                reason,
            });
        }

        let risk = classify(&proposal.action);
        match risk {
            RiskLevel::Safe => Ok(PolicyDecision::Allow { risk }),
            RiskLevel::Sensitive if context.has_fresh_native_authentication => {
                Ok(PolicyDecision::Allow { risk })
            }
            RiskLevel::Sensitive => Ok(PolicyDecision::RequireApproval {
                risk,
                explanation: "This action accesses private local data and needs a fresh, scoped authorization.".into(),
                digest: approval_digest(proposal)?,
            }),
            RiskLevel::Consequential => Ok(PolicyDecision::RequireApproval {
                risk,
                explanation: "This action can affect another person, service, or external system.".into(),
                digest: approval_digest(proposal)?,
            }),
            RiskLevel::Destructive => Ok(PolicyDecision::RequireApproval {
                risk,
                explanation: "This action can remove or overwrite user data. Approval applies once to the exact resource shown.".into(),
                digest: approval_digest(proposal)?,
            }),
            RiskLevel::Privileged => Ok(PolicyDecision::RequireApproval {
                risk,
                explanation: "This action requires a narrowly scoped privileged operation and native device authentication.".into(),
                digest: approval_digest(proposal)?,
            }),
            RiskLevel::Prohibited => Ok(PolicyDecision::Deny {
                risk,
                reason: "operation is prohibited by policy".into(),
            }),
        }
    }

    fn validate_schema_constraints(&self, proposal: &ActionProposal) -> CoreResult<()> {
        if proposal.target_resource.trim().is_empty() {
            return Err(CoreError::InvalidAction(
                "target_resource must not be empty".into(),
            ));
        }

        match &proposal.action {
            Action::TypeText { text, .. } if text.len() > 1_000_000 => Err(
                CoreError::InvalidAction("typed text exceeds the one-megabyte limit".into()),
            ),
            Action::WriteFile { content, .. } if content.len() > 16 * 1024 * 1024 => Err(
                CoreError::InvalidAction("file write exceeds the 16-megabyte action limit".into()),
            ),
            Action::RunCommand {
                timeout_seconds, ..
            } if *timeout_seconds == 0 || *timeout_seconds > 300 => Err(CoreError::InvalidAction(
                "command timeout must be between 1 and 300 seconds".into(),
            )),
            _ => Ok(()),
        }
    }
}

pub fn classify(action: &Action) -> RiskLevel {
    match action {
        Action::WaitForCondition { .. }
        | Action::OpenApplication { .. }
        | Action::CloseApplication { .. }
        | Action::PressShortcut { .. }
        | Action::AskUser { .. } => RiskLevel::Safe,
        Action::ReadFile { .. }
        | Action::ClickElement { .. }
        | Action::TypeText {
            sensitive: false, ..
        }
        | Action::NavigateUrl { .. }
        | Action::RunCommand { network: false, .. } => RiskLevel::Sensitive,
        Action::TypeText {
            sensitive: true, ..
        }
        | Action::WriteFile {
            overwrite: false, ..
        }
        | Action::MoveFile { .. }
        | Action::CreateFolder { .. }
        | Action::DownloadFile { .. }
        | Action::UploadFile { .. }
        | Action::SendMessage { .. }
        | Action::SubmitForm { .. }
        | Action::RunCommand { network: true, .. } => RiskLevel::Consequential,
        Action::WriteFile {
            overwrite: true, ..
        }
        | Action::DeleteFile { .. } => RiskLevel::Destructive,
        Action::InstallApplication { .. } | Action::ChangeSetting { .. } => RiskLevel::Privileged,
    }
}

fn prohibited_reason(action: &Action) -> Option<String> {
    match action {
        Action::RunCommand { program, args, .. } => {
            let executable = Path::new(program)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(program)
                .to_ascii_lowercase();
            let is_shell = matches!(
                executable.as_str(),
                "sh" | "bash"
                    | "zsh"
                    | "fish"
                    | "cmd"
                    | "cmd.exe"
                    | "powershell"
                    | "powershell.exe"
                    | "pwsh"
                    | "pwsh.exe"
            );
            if is_shell {
                return Some(
                    "general-purpose shell interpreters are not valid structured command actions"
                        .into(),
                );
            }
            if args.iter().any(|arg| arg.contains('\0')) {
                return Some("command arguments may not contain NUL bytes".into());
            }
            None
        }
        Action::ReadFile { path, .. }
        | Action::WriteFile { path, .. }
        | Action::DeleteFile { path }
        | Action::CreateFolder { path }
        | Action::UploadFile { path, .. } => protected_path_reason(path),
        Action::MoveFile {
            source,
            destination,
        } => protected_path_reason(source).or_else(|| protected_path_reason(destination)),
        _ => None,
    }
}

fn protected_path_reason(path: &Path) -> Option<String> {
    let protected = [
        ".ssh",
        ".gnupg",
        ".aws",
        ".azure",
        ".config/gcloud",
        "Library/Keychains",
        "AppData/Roaming/Microsoft/Credentials",
    ];
    let normalized = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/");
    protected
        .iter()
        .find(|candidate| normalized.contains(**candidate))
        .map(|candidate| {
            format!("direct access to protected credential path '{candidate}' is prohibited")
        })
}

pub fn approval_digest(proposal: &ActionProposal) -> CoreResult<String> {
    let canonical = serde_json::to_vec(proposal)?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use uuid::Uuid;

    use crate::domain::{Action, ActionProposal, ExpectedOutcome, Provenance};

    use super::*;

    fn proposal(action: Action) -> ActionProposal {
        ActionProposal {
            id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            action,
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: "test".into(),
            provenance: Provenance::model(Vec::new()),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn shell_commands_are_prohibited_below_the_model() {
        let proposal = proposal(Action::RunCommand {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "echo unsafe".into()],
            working_directory: None,
            network: false,
            timeout_seconds: 10,
        });
        let decision = PolicyEngine
            .evaluate(
                &proposal,
                &PolicyContext {
                    task_request: "run a command".into(),
                    has_fresh_native_authentication: false,
                    is_recovery_attempt: false,
                },
            )
            .unwrap();
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
    }
}
