use crate::domain::{Action, ActionProposal, Condition, ExpectedOutcome};
use crate::error::{CoreError, CoreResult};
use crate::observation::{Evidence, Observation};

#[derive(Debug, Default)]
pub struct Verifier;

/// Verification is selected by the installed tool, never by a model's claim.
pub fn bind_required_outcome(proposal: &mut ActionProposal) -> CoreResult<()> {
    use sha2::{Digest, Sha256};
    proposal.expected_outcome = match &proposal.action {
        Action::ListDirectory {
            path,
            page_size,
            cursor,
        } => ExpectedOutcome::DirectoryPage {
            path: path.clone(),
            page_size: *page_size,
            cursor: cursor.clone(),
        },
        Action::FetchPublic { url, .. } => ExpectedOutcome::PublicResource { url: url.clone() },
        Action::ReadFile { path, max_bytes }
            if proposal.metadata.contains_key("procedure_stream_output") =>
        {
            let valid_identifier = |name: &str| {
                !name.is_empty()
                    && name.len() <= 96
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
            };
            let metadata_value = |key: &str, description: &str| {
                proposal
                    .metadata
                    .get(key)
                    .filter(|value| valid_identifier(value))
                    .ok_or_else(|| CoreError::InvalidAction(description.into()))
            };
            let channel_id = metadata_value(
                "procedure_stream_channel_id",
                "Streamed file read requires one exact bounded channel identity",
            )?;
            let producer_node = metadata_value(
                "procedure_node_id",
                "Streamed file read requires an exact producer node",
            )?;
            let output_port = metadata_value(
                "procedure_stream_output",
                "Streamed file read requires one exact output port",
            )?;
            let consumer_node = metadata_value(
                "procedure_stream_consumer_node",
                "Streamed file read requires an exact consumer node",
            )?;
            let maximum_bytes = proposal
                .metadata
                .get("procedure_stream_max_bytes")
                .and_then(|bytes| bytes.parse::<u64>().ok())
                .filter(|bytes| {
                    (1..=crate::execution::files::MAX_BYTES).contains(bytes) && bytes <= max_bytes
                })
                .ok_or_else(|| {
                    CoreError::InvalidAction(
                        "Streamed file read requires a bounded byte count within its read grant"
                            .into(),
                    )
                })?;
            if proposal
                .metadata
                .get("procedure_stream_node")
                .map(String::as_str)
                != Some("true")
                || !proposal.metadata.contains_key("procedure_id")
            {
                return Err(CoreError::InvalidAction(
                    "Streamed file read metadata is missing its procedure identity".into(),
                ));
            }
            ExpectedOutcome::FileReadMatchesStream {
                path: path.clone(),
                channel_id: channel_id.clone(),
                producer_node: producer_node.clone(),
                output_port: output_port.clone(),
                consumer_node: consumer_node.clone(),
                maximum_bytes,
            }
        }
        Action::ReadFile { path, .. } => ExpectedOutcome::Condition {
            condition: Condition::FileExists { path: path.clone() },
        },
        Action::WriteFile { path, content, .. }
            if proposal.metadata.contains_key("procedure_stream_input") =>
        {
            let channel_id = proposal
                .metadata
                .get("procedure_stream_channel_id")
                .filter(|channel| {
                    !channel.is_empty()
                        && channel.len() <= 96
                        && channel
                            .bytes()
                            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
                })
                .ok_or_else(|| {
                    CoreError::InvalidAction(
                        "Streamed file write requires one exact bounded channel identity".into(),
                    )
                })?;
            let source_node = proposal
                .metadata
                .get("procedure_stream_producer_node")
                .filter(|node| {
                    !node.is_empty()
                        && node.len() <= 96
                        && node.bytes().all(|byte| {
                            byte.is_ascii_lowercase()
                                || byte.is_ascii_digit()
                                || b"._-".contains(&byte)
                        })
                })
                .ok_or_else(|| {
                    CoreError::InvalidAction(
                        "Streamed file write requires an exact producer node".into(),
                    )
                })?;
            let maximum_bytes = proposal
                .metadata
                .get("procedure_stream_max_bytes")
                .and_then(|bytes| bytes.parse::<u64>().ok())
                .filter(|bytes| (1..=crate::execution::files::MAX_BYTES).contains(bytes))
                .ok_or_else(|| {
                    CoreError::InvalidAction(
                        "Streamed file write requires a bounded byte count for approval".into(),
                    )
                })?;
            if !content.is_empty()
                || proposal
                    .metadata
                    .get("procedure_stream_node")
                    .map(String::as_str)
                    != Some("true")
                || !proposal.metadata.contains_key("procedure_id")
                || !proposal.metadata.contains_key("procedure_node_id")
                || proposal
                    .metadata
                    .get("procedure_stream_input")
                    .map(String::as_str)
                    != Some("content")
            {
                return Err(CoreError::InvalidAction(
                    "Streamed file write metadata does not match the file-content input".into(),
                ));
            }
            ExpectedOutcome::FileMatchesStream {
                path: path.clone(),
                channel_id: channel_id.clone(),
                producer_node: source_node.clone(),
                maximum_bytes,
            }
        }
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
        Action::OpenApplication { .. } => ExpectedOutcome::SignedApplication {
            target: crate::application_target::ApplicationTarget::from_proposal(proposal)?,
        },
        Action::SetApplicationControl {
            control_id, value, ..
        } => ExpectedOutcome::ApplicationControlValue {
            target: crate::application_target::ApplicationTarget::from_proposal(proposal)?,
            control_id: control_id.clone(),
            value: value.clone(),
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
            ExpectedOutcome::DirectoryPage { path, page_size, cursor } => observation.evidence.iter().any(|evidence| matches!(evidence,
                Evidence::DirectoryPage { path: observed, page_size: observed_size, cursor: observed_cursor, page_sha256, snapshot_sha256, total_entries }
                if observed == &path.to_string_lossy() && observed_size == page_size && observed_cursor == cursor
                    && (1..=crate::execution::directory::MAX_PAGE_ENTRIES).contains(page_size)
                    && *total_entries as usize <= crate::execution::directory::MAX_DIRECTORY_ENTRIES
                    && [page_sha256,snapshot_sha256].iter().all(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())))),
            ExpectedOutcome::SignedApplication { target } => target.validate().is_ok() && observation.evidence.iter().any(
                |evidence| matches!(evidence, Evidence::SignedApplication { target: observed, process_id } if observed == target && *process_id > 0)),
            ExpectedOutcome::ApplicationControlValue { target, control_id, value } => {
                target.validate().is_ok()
                    && observation.evidence.iter().any(|evidence| matches!(
                        evidence,
                        Evidence::ApplicationControlValue {
                            target: observed_target,
                            process_id,
                            control_id: observed_control,
                            value: observed_value,
                        } if observed_target == target
                            && *process_id > 0
                            && observed_control == control_id
                            && observed_value == value
                    ))
            }
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
            ExpectedOutcome::FileMatchesStream {
                path,
                channel_id,
                producer_node,
                maximum_bytes,
            } => observation
                .evidence
                .iter()
                .any(|evidence| matches!(
                    evidence,
                    Evidence::FileStreamHash {
                        path: observed_path,
                        channel_id: observed_channel,
                        producer_node: observed_producer,
                        file_sha256,
                        stream_sha256,
                        bytes,
                    } if observed_path == &path.to_string_lossy()
                        && observed_channel == channel_id
                        && observed_producer == producer_node
                        && file_sha256.len() == 64
                        && file_sha256.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                        && file_sha256 == stream_sha256
                        && *maximum_bytes > 0
                        && *maximum_bytes <= crate::execution::files::MAX_BYTES
                        && *bytes <= *maximum_bytes
                )),
            ExpectedOutcome::FileReadMatchesStream {
                path,
                channel_id,
                producer_node,
                output_port,
                consumer_node,
                maximum_bytes,
            } => observation.evidence.iter().any(|evidence| matches!(
                evidence,
                Evidence::FileReadStreamHash {
                    path: observed_path,
                    channel_id: observed_channel,
                    producer_node: observed_producer,
                    output_port: observed_port,
                    consumer_node: observed_consumer,
                    file_sha256,
                    stream_sha256,
                    bytes,
                } if observed_path == &path.to_string_lossy()
                    && observed_channel == channel_id
                    && observed_producer == producer_node
                    && observed_port == output_port
                    && observed_consumer == consumer_node
                    && file_sha256.len() == 64
                    && file_sha256.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                    && file_sha256 == stream_sha256
                    && *maximum_bytes > 0
                    && *maximum_bytes <= crate::execution::files::MAX_BYTES
                    && *bytes <= *maximum_bytes
            )),
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn app_success_requires_the_prepared_signature_and_a_running_process() {
        let target = crate::application_target::ApplicationTarget {
            platform: "macos".into(),
            bundle_path: "/Applications/Example.app".into(),
            identifier: "com.example.App".into(),
            code_digest: crate::application_target::ApplicationTarget::code_set_digest(&[
                "ab".repeat(20)
            ]),
            code_digests: vec!["ab".repeat(20)],
            signer: "TEAM".into(),
        };
        let expected = ExpectedOutcome::SignedApplication {
            target: target.clone(),
        };
        let observation = |evidence| Observation {
            observed_at: chrono::Utc::now(),
            provenance: crate::domain::Provenance::external(
                crate::domain::ProvenanceSource::OperatingSystem,
                "test",
            ),
            summary: "signed application observation".into(),
            evidence: vec![evidence],
        };
        assert!(
            Verifier
                .verify(
                    &expected,
                    &observation(Evidence::ApplicationState {
                        application: target.identifier.clone(),
                        running: true
                    })
                )
                .is_err()
        );
        assert!(
            Verifier
                .verify(
                    &expected,
                    &observation(Evidence::SignedApplication {
                        target: target.clone(),
                        process_id: 0
                    })
                )
                .is_err()
        );
        for altered in [
            crate::application_target::ApplicationTarget {
                bundle_path: "/tmp/Example.app".into(),
                ..target.clone()
            },
            crate::application_target::ApplicationTarget {
                code_digest: "cd".repeat(20),
                ..target.clone()
            },
            crate::application_target::ApplicationTarget {
                signer: "OTHER".into(),
                ..target.clone()
            },
        ] {
            assert!(
                Verifier
                    .verify(
                        &expected,
                        &observation(Evidence::SignedApplication {
                            target: altered,
                            process_id: 10
                        })
                    )
                    .is_err()
            );
        }
        assert!(
            Verifier
                .verify(
                    &expected,
                    &observation(Evidence::SignedApplication {
                        target,
                        process_id: 10
                    })
                )
                .is_ok()
        );
    }

    #[test]
    fn application_control_verification_requires_fresh_exact_typed_readback() {
        let target = crate::application_target::ApplicationTarget {
            platform: "macos".into(),
            bundle_path: "/Applications/Example.app".into(),
            identifier: "com.example.App".into(),
            code_digest: crate::application_target::ApplicationTarget::code_set_digest(&[
                "ab".repeat(20)
            ]),
            code_digests: vec!["ab".repeat(20)],
            signer: "TEAM".into(),
        };
        let control_id = "control-42".to_string();
        let value = crate::domain::ApplicationControlValue::Number(0.75);
        let expected = ExpectedOutcome::ApplicationControlValue {
            target: target.clone(),
            control_id: control_id.clone(),
            value: value.clone(),
        };
        let observation = |evidence| Observation {
            observed_at: chrono::Utc::now(),
            provenance: crate::domain::Provenance::external(
                crate::domain::ProvenanceSource::OperatingSystem,
                "test",
            ),
            summary: "application control readback".into(),
            evidence: vec![evidence],
        };
        let evidence = |target, process_id, control_id, value| Evidence::ApplicationControlValue {
            target,
            process_id,
            control_id,
            value,
        };

        assert!(
            Verifier
                .verify(
                    &expected,
                    &observation(evidence(
                        target.clone(),
                        42,
                        control_id.clone(),
                        value.clone()
                    ))
                )
                .is_ok()
        );
        assert!(
            Verifier
                .verify(
                    &expected,
                    &observation(evidence(
                        target.clone(),
                        0,
                        control_id.clone(),
                        value.clone()
                    ))
                )
                .is_err()
        );
        assert!(
            Verifier
                .verify(
                    &expected,
                    &observation(evidence(
                        target.clone(),
                        42,
                        "different-control".into(),
                        value.clone()
                    ))
                )
                .is_err()
        );
        assert!(
            Verifier
                .verify(
                    &expected,
                    &observation(evidence(
                        target.clone(),
                        42,
                        control_id.clone(),
                        crate::domain::ApplicationControlValue::Number(0.5),
                    ))
                )
                .is_err()
        );

        let stale = Observation {
            observed_at: chrono::Utc::now() - chrono::Duration::seconds(31),
            provenance: crate::domain::Provenance::external(
                crate::domain::ProvenanceSource::OperatingSystem,
                "test",
            ),
            summary: "stale application control readback".into(),
            evidence: vec![evidence(target, 42, control_id, value)],
        };
        assert!(Verifier.verify(&expected, &stale).is_err());
    }
}
