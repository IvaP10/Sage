//! An installed application's signed identity is prepared before approval.
use crate::{CoreError, CoreResult};
use sage_protocol::sage::ipc::v2 as wire;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationTarget {
    pub platform: String,
    pub bundle_path: String,
    pub identifier: String,
    pub code_digest: String,
    pub code_digests: Vec<String>,
    pub signer: String,
}

impl ApplicationTarget {
    pub fn code_set_digest(digests: &[String]) -> String {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(digests.join("\n").as_bytes()))
    }
    pub fn validate(&self) -> CoreResult<()> {
        let bounded = |s: &str, max: usize| {
            !s.is_empty() && s.len() <= max && !s.chars().any(char::is_control)
        };
        // This version only has a macOS signature verifier. Windows must not
        // reinterpret a path or process name as a verified application.
        if self.platform != "macos"
            || !bounded(&self.bundle_path, 4096)
            || !self.bundle_path.starts_with('/')
            || !self.bundle_path.ends_with(".app")
            || self
                .bundle_path
                .split('/')
                .any(|part| matches!(part, "." | ".."))
            || !bounded(&self.identifier, 256)
            || !self
                .identifier
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b".-_".contains(&c))
            || !bounded(&self.signer, 128)
            || self.code_digests.is_empty()
            || self.code_digests.len() > 32
            || self.code_digests.windows(2).any(|pair| pair[0] >= pair[1])
            || self.code_digests.iter().any(|digest| {
                !(40..=128).contains(&digest.len())
                    || !digest.len().is_multiple_of(2)
                    || !digest
                        .bytes()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            })
            || self.code_digest != Self::code_set_digest(&self.code_digests)
        {
            return Err(CoreError::CapabilityRejected(
                "Incomplete signed application identity".into(),
            ));
        }
        Ok(())
    }

    pub fn from_proposal(proposal: &crate::domain::ActionProposal) -> CoreResult<Self> {
        let target: Self =
            serde_json::from_str(proposal.metadata.get("application_target").ok_or_else(
                || CoreError::CapabilityRejected("Application identity was not prepared".into()),
            )?)?;
        target.validate()?;
        let action_matches = match &proposal.action {
            crate::domain::Action::OpenApplication { application }
            | crate::domain::Action::SetApplicationControl { application, .. } => {
                application == &target.identifier
            }
            _ => false,
        };
        if !action_matches {
            return Err(CoreError::CapabilityRejected(
                "Prepared application differs from the action".into(),
            ));
        }
        Ok(target)
    }

    pub fn to_wire(&self) -> wire::ApplicationTarget {
        wire::ApplicationTarget {
            platform: self.platform.clone(),
            bundle_path: self.bundle_path.clone(),
            identifier: self.identifier.clone(),
            code_digest: self.code_digest.clone(),
            code_digests: self.code_digests.clone(),
            signer: self.signer.clone(),
        }
    }
    pub fn from_wire(value: &wire::ApplicationTarget) -> CoreResult<Self> {
        let target = Self {
            platform: value.platform.clone(),
            bundle_path: value.bundle_path.clone(),
            identifier: value.identifier.clone(),
            code_digest: value.code_digest.clone(),
            code_digests: value.code_digests.clone(),
            signer: value.signer.clone(),
        };
        target.validate()?;
        Ok(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_targets_reject_ambiguous_paths_and_unqualified_platforms() {
        let target = ApplicationTarget {
            platform: "macos".into(),
            bundle_path: "/Applications/Example.app".into(),
            identifier: "com.example.App".into(),
            code_digest: ApplicationTarget::code_set_digest(&["ab".repeat(20)]),
            code_digests: vec!["ab".repeat(20)],
            signer: "APPLE".into(),
        };
        assert_eq!(
            ApplicationTarget::from_wire(&target.to_wire()).unwrap(),
            target
        );
        for path in [
            "Example.app",
            "/Applications/../tmp/Example.app",
            "/Applications/Example",
            "/tmp/Example\0.app",
        ] {
            assert!(
                ApplicationTarget {
                    bundle_path: path.into(),
                    ..target.clone()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            ApplicationTarget {
                platform: "windows".into(),
                ..target.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            ApplicationTarget {
                code_digest: "unverified".into(),
                ..target
            }
            .validate()
            .is_err()
        );
    }
}
