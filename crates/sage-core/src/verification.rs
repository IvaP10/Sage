use crate::domain::{Action, ActionProposal, Condition, ExpectedOutcome};
use crate::error::{CoreError, CoreResult};
use crate::observation::{Evidence, Observation};

#[derive(Debug, Default)]
pub struct Verifier;

/// Verification is selected by the installed tool, never by a model's claim.
pub fn bind_required_outcome(proposal: &mut ActionProposal) -> CoreResult<()> {
    use sha2::{Digest, Sha256};
    proposal.expected_outcome = match &proposal.action {
        Action::FetchPublic { url, .. } => ExpectedOutcome::PublicResource { url: url.clone() },
        Action::ReadFile { path, .. } => ExpectedOutcome::Condition {
            condition: Condition::FileExists { path: path.clone() },
        },
        Action::WriteFile { path, content, .. } => ExpectedOutcome::FileContains {
            path: path.clone(),
            sha256: format!("{:x}", Sha256::digest(content.as_bytes())),
        },
        Action::CreateFolder { path } => ExpectedOutcome::Condition {
            condition: Condition::FolderExists { path: path.clone() },
        },
        Action::DeleteFile { path } => ExpectedOutcome::Condition {
            condition: Condition::FileAbsent { path: path.clone() },
        },
        Action::OpenApplication { application } => ExpectedOutcome::Condition {
            condition: Condition::ApplicationRunning {
                application: application.clone(),
            },
        },
        Action::NavigateUrl {
            url,
            new_tab: false,
        } => ExpectedOutcome::Condition {
            condition: Condition::UrlEquals { url: url.clone() },
        },
        Action::WaitForCondition { condition, .. } => ExpectedOutcome::Condition {
            condition: condition.clone(),
        },
        Action::AskUser { .. } => ExpectedOutcome::UserAnswered,
        _ => {
            return Err(CoreError::ExecutorUnavailable(
                "This operation has no qualified independent verifier".into(),
            ));
        }
    };
    Ok(())
}

impl Verifier {
    pub fn verify(&self, expected: &ExpectedOutcome, observation: &Observation) -> CoreResult<()> {
        if (chrono::Utc::now() - observation.observed_at)
            .num_seconds()
            .abs()
            > 30
        {
            return Err(CoreError::VerificationFailed("Observation expired".into()));
        }
        let verified = match expected {
            ExpectedOutcome::PublicResource {url}=>observation.evidence.iter().any(|evidence|matches!(evidence,Evidence::FetchedResource{url:observed,status,sha256} if observed==url && (200..300).contains(status) && sha256.len()==64)),
            ExpectedOutcome::Condition { condition } => match condition {
                Condition::FolderExists { path } => observation.evidence.iter().any(|evidence| {
                    matches!(evidence, Evidence::FileState { path: observed, exists: true, is_directory: true, .. } if observed == &path.to_string_lossy())
                }),
                Condition::FileExists { path } => observation.evidence.iter().any(|evidence| {
                    matches!(evidence, Evidence::FileState { path: observed, exists: true, .. } if observed == &path.to_string_lossy())
                }),
                Condition::FileAbsent { path } => observation.evidence.iter().any(|evidence| {
                    matches!(evidence, Evidence::FileState { path: observed, exists: false, .. } if observed == &path.to_string_lossy())
                }),
                Condition::ApplicationRunning { application } => observation.evidence.iter().any(
                    |evidence| matches!(evidence, Evidence::ApplicationState { application: observed, running: true } if observed == application),
                ),
                Condition::UrlEquals { url } => observation.evidence.iter().any(
                    |evidence| matches!(evidence, Evidence::BrowserState { url: observed } if observed == url),
                ),
                Condition::ElementPresent { selector } => observation
                    .evidence
                    .iter()
                    .any(|evidence| matches!(evidence, Evidence::ElementState { description, present: true } if description == &format!("{selector:?}"))),
            },
            ExpectedOutcome::FileContains { path, sha256 } => observation.evidence.iter().any(
                |evidence| matches!(evidence, Evidence::FileHash { path: observed_path, sha256: observed_hash } if observed_path == &path.to_string_lossy() && observed_hash.eq_ignore_ascii_case(sha256)),
            ),
            ExpectedOutcome::CommandExit { code } => observation.evidence.iter().any(
                |evidence| matches!(evidence, Evidence::CommandState { exit_code } if exit_code == code),
            ),
            ExpectedOutcome::ExternalSuccess { marker } => observation.evidence.iter().any(
                |evidence| matches!(evidence, Evidence::ExternalSuccess { marker: observed, observed: true } if observed == marker),
            ),
            ExpectedOutcome::UserAnswered => observation
                .evidence
                .iter()
                .any(|evidence| matches!(evidence, Evidence::UserAnswer { received: true })),
        };
        if verified {
            Ok(())
        } else {
            Err(CoreError::VerificationFailed(observation.summary.clone()))
        }
    }
}
