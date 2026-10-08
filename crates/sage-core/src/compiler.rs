use serde::{Deserialize, Serialize};

use crate::domain::{Action, ActionProposal, ExecutionDomain};
use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionTier {
    StructuredIntegration,
    Accessibility,
    BrowserDom,
    UserInteraction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationCandidate {
    pub tier: InteractionTier,
    pub executor: ExecutionDomain,
    pub operation: String,
    pub requires_fresh_observation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledAction {
    pub proposal: ActionProposal,
    pub candidates: Vec<ImplementationCandidate>,
}

#[derive(Debug, Clone)]
pub struct ExecutorAvailability {
    pub accessibility: bool,
    pub learned_application_control: bool,
    pub browser_dom: bool,
}

impl Default for ExecutorAvailability {
    fn default() -> Self {
        Self {
            accessibility: true,
            learned_application_control: false,
            browser_dom: true,
        }
    }
}

#[derive(Debug, Default)]
pub struct ActionCompiler;

impl ActionCompiler {
    pub fn compile(
        &self,
        proposal: ActionProposal,
        availability: &ExecutorAvailability,
    ) -> CoreResult<CompiledAction> {
        crate::features::validate(&proposal.action)?;
        if !crate::features::manifests()
            .iter()
            .any(|manifest| manifest.enabled && manifest.id == proposal.action.kind())
        {
            return Err(CoreError::ExecutorUnavailable(
                "This operation has no enabled feature contract".into(),
            ));
        }
        if matches!(
            &proposal.action,
            crate::domain::Action::SetApplicationControl { .. }
        ) && !availability.learned_application_control
        {
            return Err(CoreError::ExecutorUnavailable(
                "The connected native client has not negotiated learned application-control support".into(),
            ));
        }
        let mut candidates = Vec::new();
        match &proposal.action {
            Action::NavigateUrl { .. } => {
                if availability.browser_dom {
                    candidates.push(candidate(
                        InteractionTier::StructuredIntegration,
                        ExecutionDomain::Browser,
                        "browser integration",
                    ));
                }
                if availability.browser_dom {
                    candidates.push(candidate(
                        InteractionTier::BrowserDom,
                        ExecutionDomain::Browser,
                        "DOM operation",
                    ));
                }
            }
            Action::AskUser { .. } => candidates.push(candidate(
                InteractionTier::UserInteraction,
                ExecutionDomain::UserInteraction,
                "native question",
            )),
            Action::ReadFile { .. }
            | Action::ListDirectory { .. }
            | Action::FetchPublic { .. }
            | Action::WriteFile { .. }
            | Action::MoveFile { .. }
            | Action::DeleteFile { .. }
            | Action::CreateFolder { .. }
            | Action::WaitForCondition { .. } => {
                candidates.push(candidate(
                    InteractionTier::StructuredIntegration,
                    ExecutionDomain::Native,
                    "scoped filesystem operation",
                ));
            }
            Action::OpenApplication { .. } if availability.accessibility => {
                candidates.push(candidate(
                    InteractionTier::Accessibility,
                    ExecutionDomain::Native,
                    "identified application",
                ));
            }
            Action::SetApplicationControl { .. }
                if availability.accessibility && availability.learned_application_control =>
            {
                candidates.push(candidate(
                    InteractionTier::Accessibility,
                    ExecutionDomain::Native,
                    "exact experimentally verified accessibility control",
                ));
            }
            // No generic fallback: installed worker binaries are not proof of
            // supported, isolated, independently verifiable operations.
            _ => {}
        }

        if candidates.is_empty() {
            return Err(CoreError::ExecutorUnavailable(format!(
                "no safe implementation is available for {}",
                proposal.action.kind()
            )));
        }

        candidates.sort_by_key(|candidate| candidate.tier);
        Ok(CompiledAction {
            proposal,
            candidates,
        })
    }
}

fn candidate(
    tier: InteractionTier,
    executor: ExecutionDomain,
    operation: impl Into<String>,
) -> ImplementationCandidate {
    ImplementationCandidate {
        tier,
        executor,
        operation: operation.into(),
        requires_fresh_observation: true,
    }
}
