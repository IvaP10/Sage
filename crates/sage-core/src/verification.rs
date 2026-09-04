use crate::domain::{Condition, ExpectedOutcome};
use crate::error::{CoreError, CoreResult};
use crate::observation::{Evidence, Observation};

#[derive(Debug, Default)]
pub struct Verifier;

impl Verifier {
    pub fn verify(&self, expected: &ExpectedOutcome, observation: &Observation) -> CoreResult<()> {
        let verified = match expected {
            ExpectedOutcome::Condition { condition } => match condition {
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
                Condition::ElementPresent { .. } => observation
                    .evidence
                    .iter()
                    .any(|evidence| matches!(evidence, Evidence::ElementState { present: true, .. })),
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
