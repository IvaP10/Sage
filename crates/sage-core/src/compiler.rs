use serde::{Deserialize, Serialize};

use crate::domain::{Action, ActionProposal, ExecutionDomain};
use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionTier {
    StructuredIntegration,
    Accessibility,
    BrowserDom,
    KeyboardShortcut,
    Vision,
    Coordinate,
    SandboxedProcess,
    PrivilegedOperation,
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
    pub structured_integrations: bool,
    pub accessibility: bool,
    pub browser_dom: bool,
    pub keyboard: bool,
    pub vision: bool,
    pub coordinates: bool,
    pub sandbox: bool,
    pub privileged_helper: bool,
}

impl Default for ExecutorAvailability {
    fn default() -> Self {
        Self {
            structured_integrations: true,
            accessibility: true,
            browser_dom: true,
            keyboard: true,
            vision: false,
            coordinates: false,
            sandbox: true,
            privileged_helper: false,
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
        let mut candidates = Vec::new();
        match &proposal.action {
            Action::ClickElement { .. } | Action::TypeText { .. } => {
                if availability.structured_integrations {
                    candidates.push(candidate(
                        InteractionTier::StructuredIntegration,
                        ExecutionDomain::Native,
                        "application integration",
                    ));
                }
                if availability.accessibility {
                    candidates.push(candidate(
                        InteractionTier::Accessibility,
                        ExecutionDomain::Native,
                        "semantic accessibility element",
                    ));
                }
                if availability.keyboard {
                    candidates.push(candidate(
                        InteractionTier::KeyboardShortcut,
                        ExecutionDomain::Native,
                        "validated keyboard interaction",
                    ));
                }
                if availability.vision {
                    candidates.push(candidate(
                        InteractionTier::Vision,
                        ExecutionDomain::Native,
                        "fresh visual localization",
                    ));
                }
                if availability.coordinates {
                    candidates.push(candidate(
                        InteractionTier::Coordinate,
                        ExecutionDomain::Native,
                        "fresh coordinates with stale-state guard",
                    ));
                }
            }
            Action::NavigateUrl { .. }
            | Action::DownloadFile { .. }
            | Action::UploadFile { .. }
            | Action::SubmitForm { .. } => {
                if availability.structured_integrations {
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
                if availability.accessibility {
                    candidates.push(candidate(
                        InteractionTier::Accessibility,
                        ExecutionDomain::Browser,
                        "browser accessibility fallback",
                    ));
                }
                if availability.vision {
                    candidates.push(candidate(
                        InteractionTier::Vision,
                        ExecutionDomain::Browser,
                        "fresh browser visual fallback",
                    ));
                }
            }
            Action::RunCommand { .. } if availability.sandbox => candidates.push(candidate(
                InteractionTier::SandboxedProcess,
                ExecutionDomain::Sandbox,
                "isolated command worker",
            )),
            Action::InstallApplication { .. } if availability.privileged_helper => {
                candidates.push(candidate(
                    InteractionTier::PrivilegedOperation,
                    ExecutionDomain::Privileged,
                    "allowlisted privileged helper operation",
                ));
            }
            Action::AskUser { .. } => candidates.push(candidate(
                InteractionTier::UserInteraction,
                ExecutionDomain::UserInteraction,
                "native approval or question surface",
            )),
            _ => candidates.push(candidate(
                InteractionTier::StructuredIntegration,
                proposal.action.domain(),
                "native structured operation",
            )),
        }

        if candidates.is_empty() {
            return Err(CoreError::ExecutorUnavailable(format!(
                "no safe implementation is available for {}",
                proposal.action.kind()
            )));
        }

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
