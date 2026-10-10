//! Closed, versioned messages shared by Sage Core and the restricted local inference worker.
//!
//! Protocol data carries no paths, credentials, raw model bytes, or execution authority.
//! The worker may verify a pinned package manifest against its compiled trust set and exact
//! inherited artifact handles, then optionally load and generate from a candidate under separate
//! negotiated features. Model admission remains unavailable through this protocol.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PROTOCOL_VERSION: u16 = 2;
pub const MAX_FRAME_BYTES: usize = 4 * 1024;
pub const MAX_REQUESTED_FEATURES: usize = 8;
pub const HEARTBEAT_FEATURE_NAME: &str = "heartbeat-v1";
pub const ARTIFACT_READ_FEATURE_NAME: &str = "artifact-read-v1";
pub const QWEN35_PACKAGE_VERIFY_FEATURE_NAME: &str = "qwen35-package-verify-v1";
pub const QWEN35_CANDIDATE_LOAD_FEATURE_NAME: &str = "qwen35-candidate-load-v1";
pub const QWEN35_GENERATE_FEATURE_NAME: &str = "qwen35-generate-v1";
pub const QWEN35_GENERATION_PREVIEW_FEATURE_NAME: &str = "qwen35-generation-preview-v1";
pub const MAX_ARTIFACT_SLOTS: usize = 8;
pub const MAX_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024 * 1024;
pub const QWEN35_PACKAGE_ARTIFACT_COUNT: usize = 6;
pub const QWEN35_ADMITTED_CONTEXT_TOKENS: u32 = 8_192;
pub const MAX_QWEN35_MODEL_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;
pub const MAX_QWEN35_SYSTEM_MESSAGE_BYTES: usize = 64 * 1024;
pub const MAX_QWEN35_USER_MESSAGE_BYTES: usize = 256 * 1024;
pub const MAX_QWEN35_SCHEMA_BYTES: usize = 16 * 1024;
pub const MAX_QWEN35_PROMPT_CHUNK_BYTES: usize = 512;
pub const MAX_QWEN35_GENERATION_PREVIEW_CHUNK_BYTES: usize = 512;
pub const MAX_QWEN35_GENERATION_OUTPUT_BYTES: usize = 256 * 1024;
pub const MAX_QWEN35_GENERATION_TOKENS: u16 = 2_048;
pub const QWEN35_INDEPENDENT_READ_TOOL_KINDS: &[&str] =
    &["read_file", "list_directory", "fetch_public"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerReadiness {
    ModelNotAdmitted,
    PackageVerifiedModelNotAdmitted,
    CandidateLoadedModelNotAdmitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerFeature {
    #[serde(rename = "heartbeat-v1")]
    HeartbeatV1,
    #[serde(rename = "artifact-read-v1")]
    ArtifactReadV1,
    #[serde(rename = "qwen35-package-verify-v1")]
    Qwen35PackageVerifyV1,
    #[serde(rename = "qwen35-candidate-load-v1")]
    Qwen35CandidateLoadV1,
    #[serde(rename = "qwen35-generate-v1")]
    Qwen35GenerateV1,
    #[serde(rename = "qwen35-generation-preview-v1")]
    Qwen35GenerationPreviewV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35PromptField {
    Schema,
    SystemMessage,
    UserMessage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedPackageArtifactReceipt {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerRequest {
    Hello {
        protocol_version: u16,
        challenge: String,
        requested_features: Vec<String>,
    },
    Ping {
        protocol_version: u16,
        challenge: String,
        sequence: u64,
    },
    VerifyArtifact {
        protocol_version: u16,
        challenge: String,
        sequence: u64,
        artifact_slot: u8,
        expected_bytes: u64,
        expected_sha256: String,
    },
    VerifyQwen35Package {
        protocol_version: u16,
        challenge: String,
        sequence: u64,
        manifest_json: String,
        artifact_slots: [u8; QWEN35_PACKAGE_ARTIFACT_COUNT],
    },
    LoadQwen35Candidate {
        protocol_version: u16,
        challenge: String,
        sequence: u64,
    },
    BeginQwen35Generation {
        protocol_version: u16,
        challenge: String,
        sequence: u64,
        generation_id: String,
        maximum_new_tokens: u16,
        schema_bytes: u32,
        system_message_bytes: u32,
        user_message_bytes: u32,
        payload_sha256: String,
    },
    AppendQwen35PromptChunk {
        protocol_version: u16,
        challenge: String,
        sequence: u64,
        generation_id: String,
        field: Qwen35PromptField,
        offset: u32,
        chunk: String,
    },
    GenerateQwen35 {
        protocol_version: u16,
        challenge: String,
        sequence: u64,
        generation_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerResponse {
    Hello {
        protocol_version: u16,
        challenge: String,
        process_id: u32,
        features: Vec<WorkerFeature>,
        readiness: WorkerReadiness,
    },
    Pong {
        protocol_version: u16,
        challenge: String,
        process_id: u32,
        sequence: u64,
        readiness: WorkerReadiness,
    },
    ArtifactVerified {
        protocol_version: u16,
        challenge: String,
        process_id: u32,
        sequence: u64,
        artifact_slot: u8,
        bytes: u64,
        sha256: String,
    },
    Qwen35PackageVerified {
        protocol_version: u16,
        challenge: String,
        process_id: u32,
        sequence: u64,
        manifest_sha256: String,
        key_id: String,
        checkpoint_revision: String,
        artifacts: Vec<VerifiedPackageArtifactReceipt>,
    },
    Qwen35CandidateLoaded {
        protocol_version: u16,
        challenge: String,
        process_id: u32,
        sequence: u64,
        manifest_sha256: String,
        checkpoint_revision: String,
        maximum_context_tokens: u32,
        estimated_memory_bytes: u64,
    },
    Qwen35GenerationStarted {
        protocol_version: u16,
        challenge: String,
        process_id: u32,
        sequence: u64,
        generation_id: String,
    },
    Qwen35GenerationOutputChunk {
        protocol_version: u16,
        challenge: String,
        process_id: u32,
        sequence: u64,
        generation_id: String,
        chunk_index: u32,
        chunk: String,
    },
    /// An append-only fragment of the answer field, for display while the
    /// worker is still decoding. It is not a verified result or authority.
    Qwen35GenerationPreviewChunk {
        protocol_version: u16,
        challenge: String,
        process_id: u32,
        sequence: u64,
        generation_id: String,
        preview_offset: u32,
        chunk: String,
    },
    Qwen35GenerationFinished {
        protocol_version: u16,
        challenge: String,
        process_id: u32,
        sequence: u64,
        generation_id: String,
        output_chunks: u32,
        output_bytes: u32,
        output_sha256: String,
    },
}

/// Returns only the worker features requested by this client and implemented here.
/// Unknown but well-formed feature names are ignored so v2 extensions can negotiate safely.
pub fn negotiate_features(
    requested: &[String],
    artifact_read_available: bool,
    qwen35_package_verify_available: bool,
    qwen35_candidate_load_available: bool,
    qwen35_generate_available: bool,
) -> Option<Vec<WorkerFeature>> {
    negotiate_features_with_previews(
        requested,
        artifact_read_available,
        qwen35_package_verify_available,
        qwen35_candidate_load_available,
        qwen35_generate_available,
        false,
    )
}

/// Negotiate the append-only preview extension separately from generation so
/// older protocol-v2 peers keep their original generation response shape.
pub fn negotiate_features_with_previews(
    requested: &[String],
    artifact_read_available: bool,
    qwen35_package_verify_available: bool,
    qwen35_candidate_load_available: bool,
    qwen35_generate_available: bool,
    qwen35_generation_previews_available: bool,
) -> Option<Vec<WorkerFeature>> {
    if requested.len() > MAX_REQUESTED_FEATURES
        || requested.iter().any(|feature| !valid_feature_name(feature))
    {
        return None;
    }

    let mut features = Vec::with_capacity(6);
    if requested
        .iter()
        .any(|feature| feature.as_str() == HEARTBEAT_FEATURE_NAME)
    {
        features.push(WorkerFeature::HeartbeatV1);
    }
    if artifact_read_available
        && requested
            .iter()
            .any(|feature| feature.as_str() == ARTIFACT_READ_FEATURE_NAME)
    {
        features.push(WorkerFeature::ArtifactReadV1);
    }
    if qwen35_package_verify_available
        && requested
            .iter()
            .any(|feature| feature.as_str() == QWEN35_PACKAGE_VERIFY_FEATURE_NAME)
    {
        features.push(WorkerFeature::Qwen35PackageVerifyV1);
    }
    if qwen35_package_verify_available
        && qwen35_candidate_load_available
        && requested
            .iter()
            .any(|feature| feature.as_str() == QWEN35_CANDIDATE_LOAD_FEATURE_NAME)
    {
        features.push(WorkerFeature::Qwen35CandidateLoadV1);
    }
    if qwen35_package_verify_available
        && qwen35_candidate_load_available
        && qwen35_generate_available
        && requested
            .iter()
            .any(|feature| feature.as_str() == QWEN35_GENERATE_FEATURE_NAME)
    {
        features.push(WorkerFeature::Qwen35GenerateV1);
    }
    if qwen35_package_verify_available
        && qwen35_candidate_load_available
        && qwen35_generate_available
        && qwen35_generation_previews_available
        && requested
            .iter()
            .any(|feature| feature.as_str() == QWEN35_GENERATION_PREVIEW_FEATURE_NAME)
    {
        features.push(WorkerFeature::Qwen35GenerationPreviewV1);
    }
    Some(features)
}

/// Hashes the exact typed prompt payload with domain separation and explicit
/// field lengths, so chunk order or field-boundary changes cannot alias.
pub fn qwen35_generation_payload_sha256(schema: &[u8], system: &[u8], user: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"sage-qwen35-planner-prompt-v1\0");
    for field in [schema, system, user] {
        digest.update((field.len() as u64).to_be_bytes());
        digest.update(field);
    }
    let digest = digest.finalize();
    encode_sha256(&digest)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    encode_sha256(&digest)
}

fn encode_sha256(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

pub fn valid_challenge(challenge: &str) -> bool {
    challenge.len() == 64
        && challenge
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn valid_sha256_hex(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_feature_name(feature: &str) -> bool {
    !feature.is_empty()
        && feature.len() <= 64
        && feature
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_schema_is_versioned_closed_and_feature_negotiated() {
        let challenge = "a5".repeat(32);
        let request = WorkerRequest::Hello {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            requested_features: vec![
                HEARTBEAT_FEATURE_NAME.into(),
                ARTIFACT_READ_FEATURE_NAME.into(),
                QWEN35_PACKAGE_VERIFY_FEATURE_NAME.into(),
                QWEN35_CANDIDATE_LOAD_FEATURE_NAME.into(),
                "future-feature-v1".into(),
            ],
        };
        let bytes = serde_json::to_vec(&request).unwrap();
        let parsed: WorkerRequest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed, request);
        assert_eq!(
            negotiate_features(
                &[
                    HEARTBEAT_FEATURE_NAME.into(),
                    ARTIFACT_READ_FEATURE_NAME.into(),
                    QWEN35_PACKAGE_VERIFY_FEATURE_NAME.into(),
                    QWEN35_CANDIDATE_LOAD_FEATURE_NAME.into(),
                    QWEN35_GENERATE_FEATURE_NAME.into(),
                    "future-feature-v1".into(),
                ],
                true,
                true,
                true,
                true,
            ),
            Some(vec![
                WorkerFeature::HeartbeatV1,
                WorkerFeature::ArtifactReadV1,
                WorkerFeature::Qwen35PackageVerifyV1,
                WorkerFeature::Qwen35CandidateLoadV1,
                WorkerFeature::Qwen35GenerateV1,
            ])
        );
        assert_eq!(
            negotiate_features(
                &[
                    HEARTBEAT_FEATURE_NAME.into(),
                    ARTIFACT_READ_FEATURE_NAME.into()
                ],
                false,
                false,
                false,
                false,
            ),
            Some(vec![WorkerFeature::HeartbeatV1])
        );
        assert_eq!(
            negotiate_features(
                &[QWEN35_PACKAGE_VERIFY_FEATURE_NAME.into()],
                false,
                true,
                false,
                false,
            ),
            Some(vec![WorkerFeature::Qwen35PackageVerifyV1])
        );
        assert_eq!(
            negotiate_features(
                &[QWEN35_PACKAGE_VERIFY_FEATURE_NAME.into()],
                false,
                false,
                true,
                false,
            ),
            Some(Vec::new())
        );
        assert_eq!(
            negotiate_features(
                &[QWEN35_CANDIDATE_LOAD_FEATURE_NAME.into()],
                false,
                true,
                true,
                true,
            ),
            Some(vec![WorkerFeature::Qwen35CandidateLoadV1])
        );
        assert_eq!(
            negotiate_features(
                &[QWEN35_GENERATE_FEATURE_NAME.into()],
                false,
                true,
                true,
                false,
            ),
            Some(Vec::new())
        );
        assert_eq!(
            negotiate_features(
                &[QWEN35_GENERATE_FEATURE_NAME.into()],
                false,
                true,
                true,
                true,
            ),
            Some(vec![WorkerFeature::Qwen35GenerateV1])
        );
        assert_eq!(
            negotiate_features_with_previews(
                &[
                    QWEN35_GENERATE_FEATURE_NAME.into(),
                    QWEN35_GENERATION_PREVIEW_FEATURE_NAME.into(),
                ],
                false,
                true,
                true,
                true,
                true,
            ),
            Some(vec![
                WorkerFeature::Qwen35GenerateV1,
                WorkerFeature::Qwen35GenerationPreviewV1,
            ])
        );
        assert_eq!(
            negotiate_features_with_previews(
                &[QWEN35_GENERATE_FEATURE_NAME.into()],
                false,
                true,
                true,
                true,
                true,
            ),
            Some(vec![WorkerFeature::Qwen35GenerateV1])
        );
        assert_eq!(
            negotiate_features_with_previews(
                &[
                    QWEN35_GENERATE_FEATURE_NAME.into(),
                    QWEN35_GENERATION_PREVIEW_FEATURE_NAME.into(),
                ],
                false,
                true,
                true,
                true,
                false,
            ),
            Some(vec![WorkerFeature::Qwen35GenerateV1])
        );

        let package_request = WorkerRequest::VerifyQwen35Package {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            sequence: 4,
            manifest_json: "{\"schema_version\":2}".into(),
            artifact_slots: [0, 1, 2, 3, 4, 5],
        };
        let package_frame = serde_json::to_vec(&package_request).unwrap();
        assert!(package_frame.len() <= MAX_FRAME_BYTES);
        assert_eq!(
            serde_json::from_slice::<WorkerRequest>(&package_frame).unwrap(),
            package_request
        );
        let load_request = WorkerRequest::LoadQwen35Candidate {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            sequence: 5,
        };
        assert_eq!(
            serde_json::from_slice::<WorkerRequest>(&serde_json::to_vec(&load_request).unwrap())
                .unwrap(),
            load_request
        );
        let generation_request = WorkerRequest::BeginQwen35Generation {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            sequence: 6,
            generation_id: "ef".repeat(32),
            maximum_new_tokens: MAX_QWEN35_GENERATION_TOKENS,
            schema_bytes: 1024,
            system_message_bytes: 2048,
            user_message_bytes: 4096,
            payload_sha256: "12".repeat(32),
        };
        let generation_frame = serde_json::to_vec(&generation_request).unwrap();
        assert!(generation_frame.len() <= MAX_FRAME_BYTES);
        assert_eq!(
            serde_json::from_slice::<WorkerRequest>(&generation_frame).unwrap(),
            generation_request
        );
        let generation_chunk = WorkerRequest::AppendQwen35PromptChunk {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            sequence: 7,
            generation_id: "ef".repeat(32),
            field: Qwen35PromptField::UserMessage,
            offset: 0,
            chunk: "\0".repeat(MAX_QWEN35_PROMPT_CHUNK_BYTES),
        };
        let chunk_frame = serde_json::to_vec(&generation_chunk).unwrap();
        assert!(chunk_frame.len() <= MAX_FRAME_BYTES);
        assert_eq!(
            serde_json::from_slice::<WorkerRequest>(&chunk_frame).unwrap(),
            generation_chunk
        );
        let package_receipt = WorkerResponse::Qwen35PackageVerified {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 7,
            sequence: 4,
            manifest_sha256: "ab".repeat(32),
            key_id: "sage-test".into(),
            checkpoint_revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a".into(),
            artifacts: vec![VerifiedPackageArtifactReceipt {
                name: "config.json".into(),
                bytes: 128,
                sha256: "cd".repeat(32),
            }],
        };
        let receipt_frame = serde_json::to_vec(&package_receipt).unwrap();
        assert!(receipt_frame.len() <= MAX_FRAME_BYTES);
        assert_eq!(
            serde_json::from_slice::<WorkerResponse>(&receipt_frame).unwrap(),
            package_receipt
        );
        let candidate_loaded = WorkerResponse::Qwen35CandidateLoaded {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 7,
            sequence: 5,
            manifest_sha256: "ab".repeat(32),
            checkpoint_revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a".into(),
            maximum_context_tokens: QWEN35_ADMITTED_CONTEXT_TOKENS,
            estimated_memory_bytes: 7 * 1024 * 1024 * 1024,
        };
        let loaded_frame = serde_json::to_vec(&candidate_loaded).unwrap();
        assert!(loaded_frame.len() <= MAX_FRAME_BYTES);
        assert_eq!(
            serde_json::from_slice::<WorkerResponse>(&loaded_frame).unwrap(),
            candidate_loaded
        );
        let generation_started = WorkerResponse::Qwen35GenerationStarted {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 7,
            sequence: 6,
            generation_id: "ef".repeat(32),
        };
        let started_frame = serde_json::to_vec(&generation_started).unwrap();
        assert!(started_frame.len() <= MAX_FRAME_BYTES);
        assert_eq!(
            serde_json::from_slice::<WorkerResponse>(&started_frame).unwrap(),
            generation_started
        );
        let output_chunk = WorkerResponse::Qwen35GenerationOutputChunk {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 7,
            sequence: 8,
            generation_id: "ef".repeat(32),
            chunk_index: 0,
            chunk: "x".repeat(MAX_QWEN35_PROMPT_CHUNK_BYTES),
        };
        let output_chunk_frame = serde_json::to_vec(&output_chunk).unwrap();
        assert!(output_chunk_frame.len() <= MAX_FRAME_BYTES);
        assert_eq!(
            serde_json::from_slice::<WorkerResponse>(&output_chunk_frame).unwrap(),
            output_chunk
        );
        let finished = WorkerResponse::Qwen35GenerationFinished {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 7,
            sequence: 8,
            generation_id: "ef".repeat(32),
            output_chunks: 1,
            output_bytes: 1,
            output_sha256: sha256_hex(b"x"),
        };
        let finished_frame = serde_json::to_vec(&finished).unwrap();
        assert!(finished_frame.len() <= MAX_FRAME_BYTES);
        assert_eq!(
            serde_json::from_slice::<WorkerResponse>(&finished_frame).unwrap(),
            finished
        );

        let unknown_field = format!(
            "{{\"kind\":\"ping\",\"protocol_version\":2,\"challenge\":\"{challenge}\",\"sequence\":1,\"grant\":\"ambient\"}}"
        );
        assert!(serde_json::from_str::<WorkerRequest>(&unknown_field).is_err());
        assert!(
            serde_json::from_str::<WorkerRequest>(&format!(
                "{{\"kind\":\"execute\",\"protocol_version\":2,\"challenge\":\"{challenge}\"}}"
            ))
            .is_err()
        );
    }

    #[test]
    fn challenge_and_feature_inputs_are_bounded_and_canonical() {
        assert!(valid_challenge(&"ab".repeat(32)));
        assert!(!valid_challenge(&"AB".repeat(32)));
        assert!(!valid_challenge(&"ab".repeat(31)));
        assert!(
            negotiate_features(
                &vec!["future-v1".into(); MAX_REQUESTED_FEATURES],
                false,
                false,
                false,
                false
            )
            .is_some()
        );
        assert!(
            negotiate_features(
                &vec!["future-v1".into(); MAX_REQUESTED_FEATURES + 1],
                false,
                false,
                false,
                false
            )
            .is_none()
        );
        assert!(
            negotiate_features(&["invalid feature".into()], false, false, false, false).is_none()
        );
        assert!(negotiate_features(&["x".repeat(65)], false, false, false, false).is_none());
        assert!(valid_sha256_hex(&"ab".repeat(32)));
        assert!(!valid_sha256_hex(&"AB".repeat(32)));
        assert!(!valid_sha256_hex(&"ab".repeat(31)));
    }

    #[test]
    fn generation_payload_digest_binds_field_boundaries_and_content() {
        let base = qwen35_generation_payload_sha256(b"a", b"bc", b"d");
        assert_eq!(base.len(), 64);
        assert!(valid_sha256_hex(&base));
        assert_ne!(base, qwen35_generation_payload_sha256(b"ab", b"c", b"d"));
        assert_ne!(base, qwen35_generation_payload_sha256(b"a", b"bc", b"e"));
    }
}
