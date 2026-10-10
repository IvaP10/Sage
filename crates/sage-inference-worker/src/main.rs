#![forbid(unsafe_code)]

use std::io::{self, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::path::PathBuf;
#[cfg(feature = "qwen35-model-generate")]
use std::time::Duration;
use std::{collections::BTreeMap, fs::File};

use ed25519_dalek::VerifyingKey;
#[cfg(feature = "qwen35-model-load")]
use sage_inference_protocol::QWEN35_ADMITTED_CONTEXT_TOKENS;
use sage_inference_protocol::{
    MAX_ARTIFACT_BYTES, MAX_ARTIFACT_SLOTS, MAX_FRAME_BYTES, PROTOCOL_VERSION,
    QWEN35_PACKAGE_ARTIFACT_COUNT, VerifiedPackageArtifactReceipt, WorkerFeature, WorkerReadiness,
    WorkerRequest, WorkerResponse, negotiate_features_with_previews, valid_challenge,
    valid_sha256_hex,
};
#[cfg(feature = "qwen35-model-generate")]
use sage_inference_protocol::{
    MAX_QWEN35_GENERATION_OUTPUT_BYTES, MAX_QWEN35_GENERATION_PREVIEW_CHUNK_BYTES,
    MAX_QWEN35_GENERATION_TOKENS, MAX_QWEN35_PROMPT_CHUNK_BYTES, MAX_QWEN35_SCHEMA_BYTES,
    MAX_QWEN35_SYSTEM_MESSAGE_BYTES, MAX_QWEN35_USER_MESSAGE_BYTES,
    QWEN35_INDEPENDENT_READ_TOOL_KINDS, Qwen35PromptField, qwen35_generation_payload_sha256,
    sha256_hex,
};
use sage_model_package::{
    QWEN35_PACKAGE_ARTIFACTS, Qwen35PackageManifest, VerifiedPackageArtifact,
    VerifiedQwen35Package, application_trusted_qwen35_package_keys,
};
#[cfg(feature = "qwen35-model-load")]
use sage_qwen35_runtime::{
    loader::{
        Qwen35CandidateLoadOptions, Qwen35PackageIndexExt, VerifiedQwen35Config,
        VerifiedQwen35ImageProcessor, VerifiedQwen35TextWeights, VerifiedQwen35Tokenizer,
    },
    qwen35::Qwen35WeightIndex,
    resource::{LOCAL_COMPUTE_BUDGET, ResourceGovernor},
};
use sha2::{Digest, Sha256};

#[cfg(feature = "qwen35-model-generate")]
use sage_qwen_tokenizer::{ChatMessage, ChatRole};
#[cfg(feature = "qwen35-model-generate")]
use serde_json::Value;

const MAX_MESSAGES_PER_PROCESS: usize = 65_536;
#[cfg(feature = "qwen35-model-generate")]
const QWEN35_ANSWER_PREVIEW_MAX_DELAY: Duration = Duration::from_millis(50);
#[cfg(feature = "qwen35-model-generate")]
const QWEN35_ANSWER_PREVIEW_BATCH_BYTES: usize = 256;

struct WorkerSession {
    challenge: String,
    next_sequence: u64,
    features: Vec<WorkerFeature>,
    artifacts: Vec<Option<File>>,
    verified_qwen35: Option<WorkerVerifiedQwen35Package>,
    #[cfg(feature = "qwen35-model-load")]
    loaded_qwen35: Option<WorkerLoadedQwen35Candidate>,
    #[cfg(feature = "qwen35-model-generate")]
    pending_generation: Option<PendingQwen35Generation>,
    #[cfg(feature = "qwen35-model-generate")]
    completed_generation: Option<CompletedQwen35Generation>,
}

#[derive(Debug)]
struct WorkerVerifiedQwen35Package {
    package: VerifiedQwen35Package,
    artifacts: Vec<VerifiedPackageArtifact<File>>,
}

#[cfg(feature = "qwen35-model-load")]
struct WorkerLoadedQwen35Candidate {
    _model: sage_qwen35_runtime::loader::Qwen35CandidateModel,
    manifest_sha256: String,
    checkpoint_revision: String,
    maximum_context_tokens: u32,
    estimated_memory_bytes: u64,
}

#[cfg(feature = "qwen35-model-generate")]
struct PendingQwen35Generation {
    generation_id: String,
    maximum_new_tokens: u16,
    expected_schema_bytes: usize,
    expected_system_bytes: usize,
    expected_user_bytes: usize,
    expected_payload_sha256: String,
    schema: Vec<u8>,
    system_message: Vec<u8>,
    user_message: Vec<u8>,
}

#[cfg(feature = "qwen35-model-generate")]
struct CompletedQwen35Generation {
    generation_id: String,
    sequence: u64,
    output: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sage-inference-worker: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    match arguments.next().as_deref() {
        Some(argument) if argument == "--version" && arguments.next().is_none() => {
            println!(
                "sage-inference-worker {} protocol {PROTOCOL_VERSION}",
                env!("CARGO_PKG_VERSION")
            );
            Ok(())
        }
        #[cfg(all(feature = "sandbox-probe", target_os = "macos"))]
        Some(argument) if argument == "--sandbox-probe" => sandbox_probe(arguments.collect()),
        None => run_stdio_protocol(Vec::new()),
        Some(argument) if argument == "--artifact-fd" => {
            let descriptors =
                parse_artifact_descriptors(std::iter::once(argument.into()).chain(arguments))?;
            run_stdio_protocol(open_inherited_artifacts(&descriptors)?)
        }
        _ => Err("unsupported command line".into()),
    }
}

fn parse_artifact_descriptors<I>(arguments: I) -> Result<Vec<i32>, Box<dyn std::error::Error>>
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let mut values = arguments.into_iter();
    let mut descriptors = Vec::new();
    while let Some(option) = values.next() {
        if option != "--artifact-fd" {
            return Err("unsupported worker argument".into());
        }
        let value = values.next().ok_or("missing artifact descriptor")?;
        let descriptor = value
            .to_str()
            .ok_or("artifact descriptor is not UTF-8")?
            .parse::<i32>()?;
        if !(3..=4096).contains(&descriptor)
            || descriptors.contains(&descriptor)
            || descriptors.len() >= MAX_ARTIFACT_SLOTS
        {
            return Err("artifact descriptor is duplicate or outside its bound".into());
        }
        descriptors.push(descriptor);
    }
    if descriptors.is_empty() {
        return Err("artifact descriptor list is empty".into());
    }
    Ok(descriptors)
}

fn open_inherited_artifacts(descriptors: &[i32]) -> Result<Vec<File>, Box<dyn std::error::Error>> {
    #[cfg(unix)]
    {
        descriptors
            .iter()
            .map(|descriptor| {
                let path = PathBuf::from(format!("/dev/fd/{descriptor}"));
                let file = File::open(path)?;
                if !file.metadata()?.is_file() || file.metadata()?.len() == 0 {
                    return Err("inherited artifact is not a nonempty regular file".into());
                }
                Ok(file)
            })
            .collect()
    }
    #[cfg(not(unix))]
    {
        let _ = descriptors;
        Err("inherited artifact descriptors are not implemented on this platform".into())
    }
}

fn run_stdio_protocol(mut artifacts: Vec<File>) -> Result<(), Box<dyn std::error::Error>> {
    let trusted_keys = application_trusted_qwen35_package_keys()?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();

    let mut session = None;
    for _ in 0..MAX_MESSAGES_PER_PROCESS {
        let Some(request_bytes) = read_frame(&mut input)? else {
            return Ok(());
        };
        let request: WorkerRequest = serde_json::from_slice(&request_bytes)?;
        let response = handle_request_inner(
            request,
            &mut session,
            &mut artifacts,
            &trusted_keys,
            &mut output,
        )?;
        #[cfg(feature = "qwen35-model-generate")]
        if let Some(completed) = session
            .as_mut()
            .and_then(|session| session.completed_generation.take())
        {
            let challenge = session
                .as_ref()
                .map(|session| session.challenge.as_str())
                .ok_or("Qwen generation session disappeared")?;
            write_qwen35_generation_output(&mut output, challenge, &completed)?;
        }
        if let Some(response) = response {
            write_frame(&mut output, &serde_json::to_vec(&response)?)?;
        }
    }

    Err("worker message limit reached".into())
}

#[cfg(test)]
fn handle_request(
    request: WorkerRequest,
    session: &mut Option<WorkerSession>,
    available_artifacts: &mut Vec<File>,
    trusted_keys: &BTreeMap<String, VerifyingKey>,
) -> Result<WorkerResponse, Box<dyn std::error::Error>> {
    let mut sink = io::sink();
    handle_request_inner(
        request,
        session,
        available_artifacts,
        trusted_keys,
        &mut sink,
    )?
    .ok_or_else(|| "one-way worker request has no response frame".into())
}

fn handle_request_inner<W: Write>(
    request: WorkerRequest,
    session: &mut Option<WorkerSession>,
    available_artifacts: &mut Vec<File>,
    trusted_keys: &BTreeMap<String, VerifyingKey>,
    output: &mut W,
) -> Result<Option<WorkerResponse>, Box<dyn std::error::Error>> {
    #[cfg(not(feature = "qwen35-model-generate"))]
    let _ = output;

    match request {
        WorkerRequest::Hello {
            protocol_version,
            challenge,
            requested_features,
        } => {
            if session.is_some() {
                return Err("worker hello may only occur once per process".into());
            }
            if protocol_version != PROTOCOL_VERSION {
                return Err("unsupported worker protocol version".into());
            }
            if !valid_challenge(&challenge) {
                return Err("worker challenge must be 32 lowercase-hex bytes".into());
            }
            let package_verify_available = !trusted_keys.is_empty()
                && available_artifacts.len() == QWEN35_PACKAGE_ARTIFACT_COUNT;
            let candidate_load_available =
                cfg!(feature = "qwen35-model-load") && package_verify_available;
            let qwen35_generate_available =
                cfg!(feature = "qwen35-model-generate") && candidate_load_available;
            let features = negotiate_features_with_previews(
                &requested_features,
                !available_artifacts.is_empty(),
                package_verify_available,
                candidate_load_available,
                qwen35_generate_available,
                qwen35_generate_available,
            )
            .ok_or("worker feature list is malformed or exceeds its bound")?;
            *session = Some(WorkerSession {
                challenge: challenge.clone(),
                next_sequence: 1,
                features: features.clone(),
                artifacts: std::mem::take(available_artifacts)
                    .into_iter()
                    .map(Some)
                    .collect(),
                verified_qwen35: None,
                #[cfg(feature = "qwen35-model-load")]
                loaded_qwen35: None,
                #[cfg(feature = "qwen35-model-generate")]
                pending_generation: None,
                #[cfg(feature = "qwen35-model-generate")]
                completed_generation: None,
            });
            Ok(Some(WorkerResponse::Hello {
                protocol_version: PROTOCOL_VERSION,
                challenge,
                process_id: std::process::id(),
                features,
                readiness: WorkerReadiness::ModelNotAdmitted,
            }))
        }
        WorkerRequest::Ping {
            protocol_version,
            challenge,
            sequence,
        } => {
            if protocol_version != PROTOCOL_VERSION {
                return Err("unsupported worker protocol version".into());
            }
            let Some(session) = session.as_mut() else {
                return Err("worker ping requires a completed hello".into());
            };
            if challenge != session.challenge || sequence != session.next_sequence {
                return Err("worker ping challenge or sequence is stale".into());
            }
            session.next_sequence = session
                .next_sequence
                .checked_add(1)
                .ok_or("worker ping sequence is exhausted")?;
            Ok(Some(WorkerResponse::Pong {
                protocol_version: PROTOCOL_VERSION,
                challenge,
                process_id: std::process::id(),
                sequence,
                readiness: session.readiness(),
            }))
        }
        WorkerRequest::VerifyArtifact {
            protocol_version,
            challenge,
            sequence,
            artifact_slot,
            expected_bytes,
            expected_sha256,
        } => {
            if protocol_version != PROTOCOL_VERSION {
                return Err("unsupported worker protocol version".into());
            }
            let Some(session) = session.as_mut() else {
                return Err("artifact verification requires a completed hello".into());
            };
            if challenge != session.challenge || sequence != session.next_sequence {
                return Err("artifact request challenge or sequence is stale".into());
            }
            if !session.features.contains(&WorkerFeature::ArtifactReadV1) {
                return Err("artifact verification was not negotiated".into());
            }
            if !valid_sha256_hex(&expected_sha256)
                || expected_bytes == 0
                || expected_bytes > MAX_ARTIFACT_BYTES
            {
                return Err(
                    "artifact verification contract is malformed or exceeds its bound".into(),
                );
            }
            let slot = usize::from(artifact_slot);
            let artifact = session
                .artifacts
                .get_mut(slot)
                .and_then(Option::as_mut)
                .ok_or("artifact slot is outside the available inherited handle set")?;
            verify_artifact(artifact, expected_bytes, &expected_sha256)?;
            session.next_sequence = sequence
                .checked_add(1)
                .ok_or("worker protocol sequence is exhausted")?;
            Ok(Some(WorkerResponse::ArtifactVerified {
                protocol_version: PROTOCOL_VERSION,
                challenge,
                process_id: std::process::id(),
                sequence,
                artifact_slot,
                bytes: expected_bytes,
                sha256: expected_sha256,
            }))
        }
        WorkerRequest::VerifyQwen35Package {
            protocol_version,
            challenge,
            sequence,
            manifest_json,
            artifact_slots,
        } => {
            if protocol_version != PROTOCOL_VERSION {
                return Err("unsupported worker protocol version".into());
            }
            let Some(session) = session.as_mut() else {
                return Err("Qwen package verification requires a completed hello".into());
            };
            if challenge != session.challenge || sequence != session.next_sequence {
                return Err("Qwen package request challenge or sequence is stale".into());
            }
            if !session
                .features
                .contains(&WorkerFeature::Qwen35PackageVerifyV1)
                || trusted_keys.is_empty()
                || session.artifacts.len() != QWEN35_PACKAGE_ARTIFACT_COUNT
                || artifact_slots != [0, 1, 2, 3, 4, 5]
                || manifest_json.is_empty()
                || manifest_json.len() > MAX_FRAME_BYTES
                || session.verified_qwen35.is_some()
            {
                return Err(
                    "Qwen package verification is unnegotiated or outside its bounds".into(),
                );
            }
            let manifest = serde_json::from_str::<Qwen35PackageManifest>(&manifest_json)?;
            let package = manifest.verify_signature(trusted_keys)?;
            let mut verified_artifacts = Vec::with_capacity(QWEN35_PACKAGE_ARTIFACT_COUNT);
            let mut receipts = Vec::with_capacity(QWEN35_PACKAGE_ARTIFACT_COUNT);
            for (name, slot) in QWEN35_PACKAGE_ARTIFACTS.into_iter().zip(artifact_slots) {
                let artifact = session.artifacts[usize::from(slot)]
                    .take()
                    .ok_or("Qwen package artifact slot was already consumed")?;
                let verified = package.verify_owned_artifact(name, artifact)?;
                receipts.push(VerifiedPackageArtifactReceipt {
                    name: verified.name().to_owned(),
                    bytes: verified.bytes(),
                    sha256: verified.sha256().to_owned(),
                });
                verified_artifacts.push(verified);
            }
            let manifest_sha256 = package.manifest_sha256().to_owned();
            let key_id = package.key_id().to_owned();
            let checkpoint_revision = package.checkpoint_revision().to_owned();
            session.verified_qwen35 = Some(WorkerVerifiedQwen35Package {
                package,
                artifacts: verified_artifacts,
            });
            session.next_sequence = sequence
                .checked_add(1)
                .ok_or("worker protocol sequence is exhausted")?;
            Ok(Some(WorkerResponse::Qwen35PackageVerified {
                protocol_version: PROTOCOL_VERSION,
                challenge,
                process_id: std::process::id(),
                sequence,
                manifest_sha256,
                key_id,
                checkpoint_revision,
                artifacts: receipts,
            }))
        }
        WorkerRequest::LoadQwen35Candidate {
            protocol_version,
            challenge,
            sequence,
        } => {
            if protocol_version != PROTOCOL_VERSION {
                return Err("unsupported worker protocol version".into());
            }
            let Some(session) = session.as_mut() else {
                return Err("Qwen candidate load requires a completed hello".into());
            };
            if challenge != session.challenge || sequence != session.next_sequence {
                return Err("Qwen candidate load challenge or sequence is stale".into());
            }
            if !session
                .features
                .contains(&WorkerFeature::Qwen35CandidateLoadV1)
                || session.candidate_is_loaded()
            {
                return Err(
                    "Qwen candidate loading was not negotiated or is already complete".into(),
                );
            }
            #[cfg(feature = "qwen35-model-load")]
            {
                let verified = session
                    .verified_qwen35
                    .take()
                    .ok_or("Qwen candidate load requires a verified signed package")?;
                let loaded = load_qwen35_candidate(verified)?;
                let response = WorkerResponse::Qwen35CandidateLoaded {
                    protocol_version: PROTOCOL_VERSION,
                    challenge,
                    process_id: std::process::id(),
                    sequence,
                    manifest_sha256: loaded.manifest_sha256.clone(),
                    checkpoint_revision: loaded.checkpoint_revision.clone(),
                    maximum_context_tokens: loaded.maximum_context_tokens,
                    estimated_memory_bytes: loaded.estimated_memory_bytes,
                };
                session.loaded_qwen35 = Some(loaded);
                session.next_sequence = sequence
                    .checked_add(1)
                    .ok_or("worker protocol sequence is exhausted")?;
                Ok(Some(response))
            }
            #[cfg(not(feature = "qwen35-model-load"))]
            {
                let _ = session;
                Err("Qwen candidate loading is not compiled into this worker".into())
            }
        }
        WorkerRequest::BeginQwen35Generation {
            protocol_version,
            challenge,
            sequence,
            generation_id,
            maximum_new_tokens,
            schema_bytes,
            system_message_bytes,
            user_message_bytes,
            payload_sha256,
        } => {
            if protocol_version != PROTOCOL_VERSION {
                return Err("unsupported worker protocol version".into());
            }
            let Some(session) = session.as_mut() else {
                return Err("Qwen generation requires a completed hello".into());
            };
            if challenge != session.challenge || sequence != session.next_sequence {
                return Err("Qwen generation challenge or sequence is stale".into());
            }
            #[cfg(feature = "qwen35-model-generate")]
            {
                let schema_bytes = usize::try_from(schema_bytes)?;
                let system_message_bytes = usize::try_from(system_message_bytes)?;
                let user_message_bytes = usize::try_from(user_message_bytes)?;
                if !session.features.contains(&WorkerFeature::Qwen35GenerateV1)
                    || !session.candidate_is_loaded()
                    || session.pending_generation.is_some()
                    || session.completed_generation.is_some()
                    || !valid_challenge(&generation_id)
                    || !valid_sha256_hex(&payload_sha256)
                    || maximum_new_tokens == 0
                    || maximum_new_tokens > MAX_QWEN35_GENERATION_TOKENS
                    || schema_bytes == 0
                    || schema_bytes > MAX_QWEN35_SCHEMA_BYTES
                    || system_message_bytes == 0
                    || system_message_bytes > MAX_QWEN35_SYSTEM_MESSAGE_BYTES
                    || user_message_bytes == 0
                    || user_message_bytes > MAX_QWEN35_USER_MESSAGE_BYTES
                {
                    return Err(
                        "Qwen generation request is unnegotiated or outside its bounds".into(),
                    );
                }
                session.pending_generation = Some(PendingQwen35Generation {
                    generation_id: generation_id.clone(),
                    maximum_new_tokens,
                    expected_schema_bytes: schema_bytes,
                    expected_system_bytes: system_message_bytes,
                    expected_user_bytes: user_message_bytes,
                    expected_payload_sha256: payload_sha256,
                    schema: Vec::with_capacity(schema_bytes),
                    system_message: Vec::with_capacity(system_message_bytes),
                    user_message: Vec::with_capacity(user_message_bytes),
                });
                session.next_sequence = sequence
                    .checked_add(1)
                    .ok_or("worker protocol sequence is exhausted")?;
                Ok(Some(WorkerResponse::Qwen35GenerationStarted {
                    protocol_version: PROTOCOL_VERSION,
                    challenge,
                    process_id: std::process::id(),
                    sequence,
                    generation_id,
                }))
            }
            #[cfg(not(feature = "qwen35-model-generate"))]
            {
                let _ = (
                    session,
                    generation_id,
                    maximum_new_tokens,
                    schema_bytes,
                    system_message_bytes,
                    user_message_bytes,
                    payload_sha256,
                );
                Err("Qwen generation is not compiled into this worker".into())
            }
        }
        WorkerRequest::AppendQwen35PromptChunk {
            protocol_version,
            challenge,
            sequence,
            generation_id,
            field,
            offset,
            chunk,
        } => {
            if protocol_version != PROTOCOL_VERSION {
                return Err("unsupported worker protocol version".into());
            }
            let Some(session) = session.as_mut() else {
                return Err("Qwen prompt chunks require a completed hello".into());
            };
            if challenge != session.challenge || sequence != session.next_sequence {
                return Err("Qwen prompt chunk challenge or sequence is stale".into());
            }
            #[cfg(feature = "qwen35-model-generate")]
            {
                append_qwen35_prompt_chunk(
                    session,
                    &generation_id,
                    field,
                    offset,
                    chunk.as_bytes(),
                )?;
                session.next_sequence = sequence
                    .checked_add(1)
                    .ok_or("worker protocol sequence is exhausted")?;
                Ok(None)
            }
            #[cfg(not(feature = "qwen35-model-generate"))]
            {
                let _ = (session, generation_id, field, offset, chunk);
                Err("Qwen generation is not compiled into this worker".into())
            }
        }
        WorkerRequest::GenerateQwen35 {
            protocol_version,
            challenge,
            sequence,
            generation_id,
        } => {
            if protocol_version != PROTOCOL_VERSION {
                return Err("unsupported worker protocol version".into());
            }
            let Some(session) = session.as_mut() else {
                return Err("Qwen generation requires a completed hello".into());
            };
            if challenge != session.challenge || sequence != session.next_sequence {
                return Err("Qwen generation challenge or sequence is stale".into());
            }
            #[cfg(feature = "qwen35-model-generate")]
            {
                if !session.features.contains(&WorkerFeature::Qwen35GenerateV1)
                    || session.completed_generation.is_some()
                {
                    return Err("Qwen generation is unnegotiated or already active".into());
                }
                let pending = session
                    .pending_generation
                    .take()
                    .ok_or("Qwen generation has no staged planner prompt")?;
                validate_pending_qwen35_generation(&pending, &generation_id)?;
                let previews_enabled = session
                    .features
                    .contains(&WorkerFeature::Qwen35GenerationPreviewV1);
                let loaded = session
                    .loaded_qwen35
                    .as_mut()
                    .ok_or("Qwen generation requires a loaded candidate")?;
                let output = generate_qwen35_candidate(
                    loaded,
                    pending,
                    output,
                    &challenge,
                    sequence,
                    &generation_id,
                    previews_enabled,
                )?;
                if output.is_empty() || output.len() > MAX_QWEN35_GENERATION_OUTPUT_BYTES {
                    return Err("Qwen output exceeded its byte limit".into());
                }
                let output_chunks = qwen35_output_chunk_count(&output)?;
                let output_bytes = u32::try_from(output.len())?;
                let output_sha256 = sha256_hex(output.as_bytes());
                session.completed_generation = Some(CompletedQwen35Generation {
                    generation_id: generation_id.clone(),
                    sequence,
                    output,
                });
                session.next_sequence = sequence
                    .checked_add(1)
                    .ok_or("worker protocol sequence is exhausted")?;
                Ok(Some(WorkerResponse::Qwen35GenerationFinished {
                    protocol_version: PROTOCOL_VERSION,
                    challenge,
                    process_id: std::process::id(),
                    sequence,
                    generation_id,
                    output_chunks,
                    output_bytes,
                    output_sha256,
                }))
            }
            #[cfg(not(feature = "qwen35-model-generate"))]
            {
                let _ = (session, generation_id);
                Err("Qwen generation is not compiled into this worker".into())
            }
        }
    }
}

impl WorkerSession {
    fn candidate_is_loaded(&self) -> bool {
        #[cfg(feature = "qwen35-model-load")]
        {
            self.loaded_qwen35.is_some()
        }
        #[cfg(not(feature = "qwen35-model-load"))]
        {
            false
        }
    }

    fn readiness(&self) -> WorkerReadiness {
        #[cfg(feature = "qwen35-model-load")]
        if self.loaded_qwen35.as_ref().is_some_and(|loaded| {
            loaded.checkpoint_revision == sage_model_package::QWEN35_CHECKPOINT_REVISION
                && loaded.maximum_context_tokens == QWEN35_ADMITTED_CONTEXT_TOKENS
                && loaded.estimated_memory_bytes > 0
                && loaded.estimated_memory_bytes <= LOCAL_COMPUTE_BUDGET
        }) {
            return WorkerReadiness::CandidateLoadedModelNotAdmitted;
        }
        match self.verified_qwen35.as_ref() {
            Some(verified)
                if verified.artifacts.len() == QWEN35_PACKAGE_ARTIFACT_COUNT
                    && verified.package.checkpoint_revision()
                        == sage_model_package::QWEN35_CHECKPOINT_REVISION =>
            {
                WorkerReadiness::PackageVerifiedModelNotAdmitted
            }
            _ => WorkerReadiness::ModelNotAdmitted,
        }
    }
}

#[cfg(feature = "qwen35-model-generate")]
fn append_qwen35_prompt_chunk(
    session: &mut WorkerSession,
    generation_id: &str,
    field: Qwen35PromptField,
    offset: u32,
    chunk: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    if !session.features.contains(&WorkerFeature::Qwen35GenerateV1)
        || chunk.is_empty()
        || chunk.len() > MAX_QWEN35_PROMPT_CHUNK_BYTES
    {
        return Err("Qwen prompt chunk is unnegotiated or outside its bound".into());
    }
    let pending = session
        .pending_generation
        .as_mut()
        .ok_or("Qwen prompt chunk has no active generation")?;
    if pending.generation_id != generation_id {
        return Err("Qwen prompt chunk targets a different generation".into());
    }
    let offset = usize::try_from(offset)?;
    let (destination, expected_bytes) = match field {
        Qwen35PromptField::Schema => (&mut pending.schema, pending.expected_schema_bytes),
        Qwen35PromptField::SystemMessage => {
            (&mut pending.system_message, pending.expected_system_bytes)
        }
        Qwen35PromptField::UserMessage => (&mut pending.user_message, pending.expected_user_bytes),
    };
    let next_len = destination
        .len()
        .checked_add(chunk.len())
        .ok_or("Qwen prompt length overflowed")?;
    if offset != destination.len() || next_len > expected_bytes {
        return Err("Qwen prompt chunk is out of order or exceeds its field".into());
    }
    destination.extend_from_slice(chunk);
    Ok(())
}

#[cfg(feature = "qwen35-model-generate")]
fn validate_pending_qwen35_generation(
    pending: &PendingQwen35Generation,
    generation_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if pending.generation_id != generation_id
        || pending.schema.len() != pending.expected_schema_bytes
        || pending.system_message.len() != pending.expected_system_bytes
        || pending.user_message.len() != pending.expected_user_bytes
        || qwen35_generation_payload_sha256(
            &pending.schema,
            &pending.system_message,
            &pending.user_message,
        ) != pending.expected_payload_sha256
    {
        return Err("Qwen prompt payload is incomplete or failed its digest".into());
    }
    Ok(())
}

#[cfg(feature = "qwen35-model-generate")]
fn generate_qwen35_candidate<W: Write>(
    loaded: &mut WorkerLoadedQwen35Candidate,
    pending: PendingQwen35Generation,
    output: &mut W,
    challenge: &str,
    sequence: u64,
    generation_id: &str,
    previews_enabled: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let schema_json = String::from_utf8(pending.schema)?;
    let system_message = String::from_utf8(pending.system_message)?;
    let user_message = String::from_utf8(pending.user_message)?;
    let schema = serde_json::from_str::<Value>(&schema_json)?;
    if !schema.is_object() {
        return Err("Qwen output schema must be a JSON object".into());
    }
    let embedded_schema = format!("\n\nClosed output schema:\n{schema_json}");
    if !system_message.ends_with(&embedded_schema) {
        return Err("Qwen system prompt does not carry the transmitted output schema".into());
    }
    let maximum_context = usize::try_from(loaded.maximum_context_tokens)?;
    let messages = [
        ChatMessage {
            role: ChatRole::System,
            content: &system_message,
        },
        ChatMessage {
            role: ChatRole::User,
            content: &user_message,
        },
    ];
    let token_ids = loaded._model.tokenizer().encode_chat(&messages, true)?;
    let reserved = token_ids
        .len()
        .checked_add(usize::from(pending.maximum_new_tokens))
        .ok_or("Qwen planner token reservation overflowed")?;
    if reserved > maximum_context {
        return Err("Qwen planner prompt and output reservation exceed the model context".into());
    }
    let tokenizer = loaded._model.tokenizer();
    let end_of_turn_token_id = tokenizer
        .token_id("<|im_end|>")
        .ok_or("Qwen end-of-turn token is missing")?;
    let (decoder, tokenizer) = loaded._model.decoder_and_tokenizer_mut();
    let mut pending_preview = String::new();
    let mut preview_offset = 0usize;
    let (generated, preview_write_error) = {
        let preview_write_failed = std::cell::Cell::new(false);
        let mut preview_write_error = None;
        let mut previous_preview = String::new();
        let mut last_preview_flush = std::time::Instant::now();
        let mut answer_update = |answer: &str| {
            if !previews_enabled {
                return;
            }
            let Some(delta) = answer.strip_prefix(&previous_preview) else {
                preview_write_failed.set(true);
                preview_write_error = Some(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Qwen answer preview was not append-only",
                ));
                return;
            };
            pending_preview.push_str(delta);
            previous_preview.push_str(delta);
            if pending_preview.len() >= QWEN35_ANSWER_PREVIEW_BATCH_BYTES
                || last_preview_flush.elapsed() >= QWEN35_ANSWER_PREVIEW_MAX_DELAY
            {
                if let Err(error) = write_qwen35_generation_preview(
                    output,
                    challenge,
                    sequence,
                    generation_id,
                    &mut preview_offset,
                    &pending_preview,
                ) {
                    preview_write_failed.set(true);
                    preview_write_error = Some(error);
                    return;
                }
                pending_preview.clear();
                last_preview_flush = std::time::Instant::now();
            }
        };
        let mut cancelled = || preview_write_failed.get();
        let generated = sage_constrained_generation::generation::generate_schema_greedy(
            decoder,
            tokenizer,
            &token_ids,
            sage_constrained_generation::generation::SchemaGenerationOptions {
                schema: &schema,
                independent_read_kinds: QWEN35_INDEPENDENT_READ_TOOL_KINDS,
                end_of_turn_token_id,
                maximum_new_tokens: usize::from(pending.maximum_new_tokens),
                validate_complete: valid_json_document,
                answer_update: &mut answer_update,
                cancelled: &mut cancelled,
            },
        );
        (generated, preview_write_error)
    };
    if let Some(error) = preview_write_error {
        return Err(error.into());
    }
    if !pending_preview.is_empty() {
        write_qwen35_generation_preview(
            output,
            challenge,
            sequence,
            generation_id,
            &mut preview_offset,
            &pending_preview,
        )?;
    }
    generated.map_err(|error| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()).into()
    })
}

#[cfg(feature = "qwen35-model-generate")]
fn valid_json_document(text: &str) -> bool {
    serde_json::from_str::<Value>(text).is_ok()
}

#[cfg(feature = "qwen35-model-generate")]
fn write_qwen35_generation_preview<W: Write>(
    writer: &mut W,
    challenge: &str,
    sequence: u64,
    generation_id: &str,
    preview_offset: &mut usize,
    delta: &str,
) -> io::Result<()> {
    let total = preview_offset
        .checked_add(delta.len())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Qwen preview size overflow"))?;
    if total > MAX_QWEN35_GENERATION_OUTPUT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Qwen answer preview exceeded its output bound",
        ));
    }
    let mut start = 0usize;
    while start < delta.len() {
        let mut end = start
            .saturating_add(MAX_QWEN35_GENERATION_PREVIEW_CHUNK_BYTES)
            .min(delta.len());
        while !delta.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Qwen preview chunk could not advance at a UTF-8 boundary",
            ));
        }
        let chunk = &delta[start..end];
        let response = WorkerResponse::Qwen35GenerationPreviewChunk {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.to_owned(),
            process_id: std::process::id(),
            sequence,
            generation_id: generation_id.to_owned(),
            preview_offset: u32::try_from(*preview_offset).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "Qwen preview offset overflow")
            })?,
            chunk: chunk.to_owned(),
        };
        let frame = serde_json::to_vec(&response)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        write_frame(writer, &frame)
            .map_err(|error| io::Error::new(io::ErrorKind::BrokenPipe, error.to_string()))?;
        *preview_offset = preview_offset.checked_add(chunk.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Qwen preview offset overflow")
        })?;
        start = end;
    }
    Ok(())
}

#[cfg(feature = "qwen35-model-generate")]
fn qwen35_output_chunk_count(output: &str) -> Result<u32, Box<dyn std::error::Error>> {
    let mut count = 0u32;
    let mut start = 0usize;
    while start < output.len() {
        let mut end = start
            .saturating_add(MAX_QWEN35_PROMPT_CHUNK_BYTES)
            .min(output.len());
        while !output.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            return Err("Qwen output chunk could not advance at a UTF-8 boundary".into());
        }
        count = count
            .checked_add(1)
            .ok_or("Qwen output chunk count overflowed")?;
        start = end;
    }
    Ok(count)
}

#[cfg(feature = "qwen35-model-generate")]
fn write_qwen35_generation_output<W: Write>(
    writer: &mut W,
    challenge: &str,
    completed: &CompletedQwen35Generation,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut start = 0usize;
    let mut chunk_index = 0u32;
    while start < completed.output.len() {
        let mut end = start
            .saturating_add(MAX_QWEN35_PROMPT_CHUNK_BYTES)
            .min(completed.output.len());
        while !completed.output.is_char_boundary(end) {
            end -= 1;
        }
        let chunk = completed
            .output
            .get(start..end)
            .ok_or("Qwen output chunk did not align to UTF-8")?;
        let response = WorkerResponse::Qwen35GenerationOutputChunk {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.to_owned(),
            process_id: std::process::id(),
            sequence: completed.sequence,
            generation_id: completed.generation_id.clone(),
            chunk_index,
            chunk: chunk.to_owned(),
        };
        write_frame(writer, &serde_json::to_vec(&response)?)?;
        chunk_index = chunk_index
            .checked_add(1)
            .ok_or("Qwen output chunk sequence exhausted")?;
        start = end;
    }
    Ok(())
}

#[cfg(feature = "qwen35-model-load")]
fn load_qwen35_candidate(
    verified: WorkerVerifiedQwen35Package,
) -> Result<WorkerLoadedQwen35Candidate, Box<dyn std::error::Error>> {
    let WorkerVerifiedQwen35Package { package, artifacts } = verified;
    if artifacts.len() != QWEN35_PACKAGE_ARTIFACT_COUNT
        || package.checkpoint_revision() != sage_model_package::QWEN35_CHECKPOINT_REVISION
    {
        return Err("verified Qwen package is incomplete or has an unsupported revision".into());
    }
    let mut artifacts = artifacts.into_iter();

    let mut config_source = artifacts
        .next()
        .ok_or("verified Qwen config handle is missing")?
        .into_verified_reader();
    let config = VerifiedQwen35Config::parse(
        &package,
        package.read_verified_small_artifact(QWEN35_PACKAGE_ARTIFACTS[0], &mut config_source)?,
    )?;

    let first_shard = artifacts
        .next()
        .ok_or("verified first Qwen shard handle is missing")?;
    let second_shard = artifacts
        .next()
        .ok_or("verified second Qwen shard handle is missing")?;

    let mut index_source = artifacts
        .next()
        .ok_or("verified Qwen index handle is missing")?
        .into_verified_reader();
    let index = Qwen35WeightIndex::parse_verified(
        &package,
        package.read_verified_small_artifact(QWEN35_PACKAGE_ARTIFACTS[3], &mut index_source)?,
    )?;
    let estimated_memory_bytes = index
        .memory_envelope(128, QWEN35_ADMITTED_CONTEXT_TOKENS as usize)?
        .total()?;
    if estimated_memory_bytes > LOCAL_COMPUTE_BUDGET {
        return Err("Qwen candidate exceeds Sage's cooperative model memory budget".into());
    }

    let mut image_processor_source = artifacts
        .next()
        .ok_or("verified Qwen image processor handle is missing")?
        .into_verified_reader();
    let image_processor = VerifiedQwen35ImageProcessor::parse(
        &package,
        package.read_verified_small_artifact(
            QWEN35_PACKAGE_ARTIFACTS[4],
            &mut image_processor_source,
        )?,
    )?;
    let mut tokenizer_source = artifacts
        .next()
        .ok_or("verified Qwen tokenizer handle is missing")?
        .into_verified_reader();
    let tokenizer = VerifiedQwen35Tokenizer::parse(
        &package,
        package.read_verified_small_artifact(QWEN35_PACKAGE_ARTIFACTS[5], &mut tokenizer_source)?,
    )?;
    if artifacts.next().is_some() {
        return Err("verified Qwen package contains unexpected artifact handles".into());
    }

    let candidate = VerifiedQwen35TextWeights::open(&package, index, first_shard, second_shard)?
        .load_candidate_model(
            &package,
            config,
            tokenizer,
            image_processor,
            &ResourceGovernor::default(),
            Qwen35CandidateLoadOptions {
                q4_group_size: 128,
                maximum_context: QWEN35_ADMITTED_CONTEXT_TOKENS as usize,
            },
        )?;

    Ok(WorkerLoadedQwen35Candidate {
        _model: candidate,
        manifest_sha256: package.manifest_sha256().to_owned(),
        checkpoint_revision: package.checkpoint_revision().to_owned(),
        maximum_context_tokens: QWEN35_ADMITTED_CONTEXT_TOKENS,
        estimated_memory_bytes,
    })
}

fn verify_artifact(
    artifact: &mut File,
    expected_bytes: u64,
    expected_sha256: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    const HASH_CHUNK_BYTES: usize = 1024 * 1024;

    if artifact.metadata()?.len() != expected_bytes {
        return Err("inherited artifact size does not match the request".into());
    }
    artifact.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_CHUNK_BYTES];
    let mut total = 0_u64;
    loop {
        let count = artifact.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or("inherited artifact size overflow")?;
        if total > expected_bytes {
            return Err("inherited artifact grew while it was verified".into());
        }
        hasher.update(&buffer[..count]);
    }
    artifact.seek(SeekFrom::Start(0))?;
    let digest = format!("{:x}", hasher.finalize());
    if total != expected_bytes || digest != expected_sha256 {
        return Err("inherited artifact digest does not match the request".into());
    }
    Ok(())
}

fn read_frame<R: Read>(reader: &mut R) -> io::Result<Option<Vec<u8>>> {
    let mut length = [0_u8; 4];
    let first = reader.read(&mut length[..1])?;
    if first == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut length[1..])?;
    let size = u32::from_be_bytes(length) as usize;
    if size == 0 || size > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "worker frame exceeds its size bound",
        ));
    }
    let mut frame = vec![0; size];
    reader.read_exact(&mut frame)?;
    Ok(Some(frame))
}

fn write_frame<W: Write>(writer: &mut W, frame: &[u8]) -> io::Result<()> {
    if frame.is_empty() || frame.len() > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "worker response exceeds its size bound",
        ));
    }
    let length = u32::try_from(frame.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "worker frame is too large"))?;
    writer.write_all(&length.to_be_bytes())?;
    writer.write_all(frame)?;
    writer.flush()
}

#[cfg(all(feature = "sandbox-probe", target_os = "macos"))]
fn sandbox_probe(arguments: Vec<std::ffi::OsString>) -> Result<(), Box<dyn std::error::Error>> {
    use std::{
        fs,
        net::{SocketAddr, TcpStream},
        os::fd::RawFd,
        path::PathBuf,
        time::Duration,
    };

    let mut read_path = None;
    let mut write_path = None;
    let mut connect_address = None;
    let mut inherited_fd = None;
    let mut values = arguments.into_iter();
    while let Some(option) = values.next() {
        let value = values.next().ok_or("missing sandbox probe option value")?;
        match option.to_str() {
            Some("--read-path") => read_path = Some(PathBuf::from(value)),
            Some("--write-path") => write_path = Some(PathBuf::from(value)),
            Some("--connect") => {
                connect_address = Some(value.to_string_lossy().parse::<SocketAddr>()?)
            }
            Some("--read-fd") => inherited_fd = Some(value.to_string_lossy().parse::<RawFd>()?),
            _ => return Err("unsupported sandbox probe option".into()),
        }
    }
    let read_path = read_path.ok_or("missing --read-path")?;
    let write_path = write_path.ok_or("missing --write-path")?;
    let connect_address = connect_address.ok_or("missing --connect")?;
    let inherited_fd = inherited_fd.ok_or("missing --read-fd")?;
    if !(3..=4096).contains(&inherited_fd) {
        return Err("inherited descriptor is outside the accepted range".into());
    }

    require_denied(fs::read(read_path), "filesystem read")?;
    require_denied(
        fs::write(write_path, b"sandbox violation"),
        "filesystem write",
    )?;
    match TcpStream::connect_timeout(&connect_address, Duration::from_millis(500)) {
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {}
        Err(error) => return Err(format!("network denial was not established: {error}").into()),
        Ok(_) => return Err("sandbox permitted a network connection".into()),
    }

    let descriptor_path = PathBuf::from(format!("/dev/fd/{inherited_fd}"));
    let descriptor_bytes = fs::read(descriptor_path)?;
    if descriptor_bytes != b"sage-inherited-read-only-descriptor\n" {
        return Err("inherited descriptor contents did not match".into());
    }

    println!(
        "{{\"read_path\":\"denied\",\"write_path\":\"denied\",\"network\":\"denied\",\"inherited_fd\":\"readable\"}}"
    );
    Ok(())
}

#[cfg(all(feature = "sandbox-probe", target_os = "macos"))]
fn require_denied<T>(
    result: io::Result<T>,
    operation: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match result {
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => Ok(()),
        Err(error) => Err(format!("{operation} denial was not established: {error}").into()),
        Ok(_) => Err(format!("sandbox permitted {operation}").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use sage_inference_protocol::{
        ARTIFACT_READ_FEATURE_NAME, HEARTBEAT_FEATURE_NAME, QWEN35_PACKAGE_VERIFY_FEATURE_NAME,
        WorkerFeature,
    };
    use sage_model_package::{QWEN35_CHECKPOINT_REVISION, QWEN35_PACKAGE_ARTIFACTS};
    use std::io::{Cursor, Write};

    #[test]
    fn frames_round_trip_and_reject_zero_or_oversized_lengths() {
        let payload = br#"{"kind":"hello","protocol_version":2,"challenge":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","requested_features":["heartbeat-v1"]}"#;
        let mut encoded = Vec::new();
        write_frame(&mut encoded, payload).expect("write bounded frame");
        let mut input = Cursor::new(encoded);
        assert_eq!(
            read_frame(&mut input).expect("read frame"),
            Some(payload.to_vec())
        );
        assert_eq!(read_frame(&mut input).expect("clean eof"), None);

        for size in [0_u32, (MAX_FRAME_BYTES + 1) as u32] {
            let mut input = Cursor::new(size.to_be_bytes().to_vec());
            assert_eq!(
                read_frame(&mut input).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
    }

    #[cfg(feature = "qwen35-model-generate")]
    #[test]
    fn prompt_chunks_require_exact_offsets_and_never_exceed_field_bounds() {
        let generation_id = "cd".repeat(32);
        let mut session = WorkerSession {
            challenge: "ab".repeat(32),
            next_sequence: 1,
            features: vec![WorkerFeature::Qwen35GenerateV1],
            artifacts: Vec::new(),
            verified_qwen35: None,
            loaded_qwen35: None,
            pending_generation: Some(PendingQwen35Generation {
                generation_id: generation_id.clone(),
                maximum_new_tokens: 64,
                expected_schema_bytes: 2,
                expected_system_bytes: 4,
                expected_user_bytes: 3,
                expected_payload_sha256: String::new(),
                schema: Vec::with_capacity(2),
                system_message: Vec::with_capacity(4),
                user_message: Vec::with_capacity(3),
            }),
            completed_generation: None,
        };
        assert!(
            append_qwen35_prompt_chunk(
                &mut session,
                &generation_id,
                Qwen35PromptField::SystemMessage,
                1,
                b"x",
            )
            .is_err()
        );
        assert!(
            append_qwen35_prompt_chunk(
                &mut session,
                &generation_id,
                Qwen35PromptField::Schema,
                0,
                &vec![b'x'; MAX_QWEN35_PROMPT_CHUNK_BYTES + 1],
            )
            .is_err()
        );
        append_qwen35_prompt_chunk(
            &mut session,
            &generation_id,
            Qwen35PromptField::Schema,
            0,
            b"{}",
        )
        .unwrap();
        append_qwen35_prompt_chunk(
            &mut session,
            &generation_id,
            Qwen35PromptField::SystemMessage,
            0,
            b"safe",
        )
        .unwrap();
        append_qwen35_prompt_chunk(
            &mut session,
            &generation_id,
            Qwen35PromptField::UserMessage,
            0,
            b"ask",
        )
        .unwrap();

        let pending = session.pending_generation.as_ref().unwrap();
        let schema = pending.schema.clone();
        let system_message = pending.system_message.clone();
        let user_message = pending.user_message.clone();
        let mut valid = PendingQwen35Generation {
            generation_id: pending.generation_id.clone(),
            maximum_new_tokens: pending.maximum_new_tokens,
            expected_schema_bytes: pending.expected_schema_bytes,
            expected_system_bytes: pending.expected_system_bytes,
            expected_user_bytes: pending.expected_user_bytes,
            expected_payload_sha256: qwen35_generation_payload_sha256(
                &schema,
                &system_message,
                &user_message,
            ),
            schema,
            system_message,
            user_message,
        };
        validate_pending_qwen35_generation(&valid, &generation_id).unwrap();
        valid.user_message[0] = b'x';
        assert!(validate_pending_qwen35_generation(&valid, &generation_id).is_err());
    }

    #[cfg(feature = "qwen35-model-generate")]
    #[test]
    fn generated_output_chunks_stay_bounded_and_preserve_utf8() {
        let output = format!("{}🙂{}", "a".repeat(510), "b".repeat(600));
        let count = qwen35_output_chunk_count(&output).unwrap();
        let completed = CompletedQwen35Generation {
            generation_id: "cd".repeat(32),
            sequence: 19,
            output: output.clone(),
        };
        let mut frames = Vec::new();
        write_qwen35_generation_output(&mut frames, &"ab".repeat(32), &completed).unwrap();
        let mut reader = Cursor::new(frames);
        let mut observed = String::new();
        let mut index = 0;
        while let Some(frame) = read_frame(&mut reader).unwrap() {
            assert!(frame.len() <= MAX_FRAME_BYTES);
            let response = serde_json::from_slice::<WorkerResponse>(&frame).unwrap();
            let WorkerResponse::Qwen35GenerationOutputChunk {
                chunk_index,
                chunk,
                sequence,
                ..
            } = response
            else {
                panic!("expected generated output chunk");
            };
            assert_eq!(chunk_index, index);
            assert_eq!(sequence, 19);
            assert!(chunk.len() <= MAX_QWEN35_PROMPT_CHUNK_BYTES);
            observed.push_str(&chunk);
            index += 1;
        }
        assert_eq!(index, count);
        assert_eq!(observed, output);
    }

    #[cfg(feature = "qwen35-model-generate")]
    #[test]
    fn generated_answer_preview_chunks_are_bounded_ordered_and_utf8_safe() {
        let challenge = "ab".repeat(32);
        let generation_id = "cd".repeat(32);
        let preview = "€🙂é".repeat(180);
        let mut frames = Vec::new();
        let mut offset = 0;
        write_qwen35_generation_preview(
            &mut frames,
            &challenge,
            19,
            &generation_id,
            &mut offset,
            &preview,
        )
        .unwrap();
        assert_eq!(offset, preview.len());

        let mut reader = Cursor::new(frames);
        let mut observed = String::new();
        let mut expected_offset = 0;
        let mut chunks = 0;
        while let Some(frame) = read_frame(&mut reader).unwrap() {
            assert!(frame.len() <= MAX_FRAME_BYTES);
            let response = serde_json::from_slice::<WorkerResponse>(&frame).unwrap();
            let WorkerResponse::Qwen35GenerationPreviewChunk {
                challenge: observed_challenge,
                process_id,
                sequence,
                generation_id: observed_generation_id,
                preview_offset,
                chunk,
                ..
            } = response
            else {
                panic!("expected generated answer preview chunk");
            };
            assert_eq!(observed_challenge, challenge);
            assert_eq!(process_id, std::process::id());
            assert_eq!(sequence, 19);
            assert_eq!(observed_generation_id, generation_id);
            assert_eq!(usize::try_from(preview_offset).unwrap(), expected_offset);
            assert!(chunk.len() <= MAX_QWEN35_GENERATION_PREVIEW_CHUNK_BYTES);
            assert!(chunk.is_char_boundary(chunk.len()));
            expected_offset += chunk.len();
            observed.push_str(&chunk);
            chunks += 1;
        }
        assert!(chunks > 1);
        assert_eq!(expected_offset, preview.len());
        assert_eq!(observed, preview);
    }

    #[test]
    fn artifact_descriptor_arguments_are_closed_unique_and_bounded() {
        let valid = ["--artifact-fd", "3", "--artifact-fd", "7"]
            .into_iter()
            .map(std::ffi::OsString::from);
        assert_eq!(parse_artifact_descriptors(valid).unwrap(), [3, 7]);

        for invalid in [
            vec!["--artifact-fd"],
            vec!["--artifact-fd", "2"],
            vec!["--artifact-fd", "3", "--artifact-fd", "3"],
            vec!["--artifact-fd", "3", "--unexpected", "4"],
            vec!["--artifact-fd", "not-a-number"],
        ] {
            let arguments = invalid.into_iter().map(std::ffi::OsString::from);
            assert!(parse_artifact_descriptors(arguments).is_err());
        }

        let too_many = (0..=MAX_ARTIFACT_SLOTS)
            .flat_map(|offset| ["--artifact-fd".to_owned(), (10 + offset).to_string()])
            .map(std::ffi::OsString::from);
        assert!(parse_artifact_descriptors(too_many).is_err());
    }

    #[test]
    fn hello_negotiates_only_supported_features_and_ping_is_sequenced() {
        let valid = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let mut session = None;
        let hello = WorkerRequest::Hello {
            protocol_version: PROTOCOL_VERSION,
            challenge: valid.into(),
            requested_features: vec![HEARTBEAT_FEATURE_NAME.into(), "future-feature-v1".into()],
        };
        let mut artifacts = Vec::new();
        let response = handle_request(
            hello.clone(),
            &mut session,
            &mut artifacts,
            &BTreeMap::new(),
        )
        .expect("valid hello");
        assert!(matches!(
            response,
            WorkerResponse::Hello {
                features,
                readiness: WorkerReadiness::ModelNotAdmitted,
                ..
            } if features == vec![WorkerFeature::HeartbeatV1]
        ));

        let ping = WorkerRequest::Ping {
            protocol_version: PROTOCOL_VERSION,
            challenge: valid.into(),
            sequence: 1,
        };
        assert!(matches!(
            handle_request(ping.clone(), &mut session, &mut artifacts, &BTreeMap::new(),)
                .expect("first ping"),
            WorkerResponse::Pong { sequence: 1, .. }
        ));
        assert!(handle_request(ping, &mut session, &mut artifacts, &BTreeMap::new()).is_err());
        assert!(handle_request(hello, &mut session, &mut artifacts, &BTreeMap::new()).is_err());
    }

    #[test]
    fn hello_and_ping_reject_wrong_version_unknown_fields_and_missing_session() {
        let valid = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let unknown = serde_json::from_str::<WorkerRequest>(&format!(
            "{{\"kind\":\"hello\",\"protocol_version\":2,\"challenge\":\"{valid}\",\"requested_features\":[],\"authority\":true}}"
        ));
        assert!(unknown.is_err());

        let mut session = None;
        let wrong_version = WorkerRequest::Hello {
            protocol_version: PROTOCOL_VERSION + 1,
            challenge: valid.into(),
            requested_features: Vec::new(),
        };
        let mut artifacts = Vec::new();
        assert!(
            handle_request(
                wrong_version,
                &mut session,
                &mut artifacts,
                &BTreeMap::new()
            )
            .is_err()
        );

        let malformed_challenge = WorkerRequest::Hello {
            protocol_version: PROTOCOL_VERSION,
            challenge: "short".into(),
            requested_features: Vec::new(),
        };
        assert!(
            handle_request(
                malformed_challenge,
                &mut session,
                &mut artifacts,
                &BTreeMap::new()
            )
            .is_err()
        );

        let ping_without_hello = WorkerRequest::Ping {
            protocol_version: PROTOCOL_VERSION,
            challenge: valid.into(),
            sequence: 1,
        };
        assert!(
            handle_request(
                ping_without_hello,
                &mut session,
                &mut artifacts,
                &BTreeMap::new()
            )
            .is_err()
        );
    }

    #[test]
    fn worker_verifies_only_the_exact_inherited_artifact_contract() {
        let challenge = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let payload = b"read-only artifact payload";
        let mut fixture = tempfile::NamedTempFile::new().unwrap();
        fixture.write_all(payload).unwrap();
        let artifact = File::open(fixture.path()).unwrap();
        let digest = format!("{:x}", Sha256::digest(payload));
        let mut artifacts = vec![artifact];
        let mut session = None;
        let hello = WorkerRequest::Hello {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.into(),
            requested_features: vec![
                sage_inference_protocol::HEARTBEAT_FEATURE_NAME.into(),
                sage_inference_protocol::ARTIFACT_READ_FEATURE_NAME.into(),
            ],
        };
        assert!(matches!(
            handle_request(hello, &mut session, &mut artifacts, &BTreeMap::new()).unwrap(),
            WorkerResponse::Hello { features, .. }
                if features == vec![WorkerFeature::HeartbeatV1, WorkerFeature::ArtifactReadV1]
        ));

        let request = WorkerRequest::VerifyArtifact {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.into(),
            sequence: 1,
            artifact_slot: 0,
            expected_bytes: payload.len() as u64,
            expected_sha256: digest.clone(),
        };
        assert!(matches!(
            handle_request(request, &mut session, &mut artifacts, &BTreeMap::new()).unwrap(),
            WorkerResponse::ArtifactVerified {
                sequence: 1,
                artifact_slot: 0,
                bytes,
                sha256,
                ..
            } if bytes == payload.len() as u64 && sha256 == digest
        ));
    }

    #[test]
    fn worker_verifies_signed_manifest_and_retains_the_same_artifact_handles() {
        let challenge = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let signer = SigningKey::from_bytes(&[41; 32]);
        let key_id = "sage-worker-test";
        let temporary_directory = tempfile::tempdir().unwrap();
        let mut artifacts = Vec::new();
        let mut expected_digests = BTreeMap::new();

        for name in QWEN35_PACKAGE_ARTIFACTS {
            let bytes = format!("test artifact: {name}").into_bytes();
            let path = temporary_directory.path().join(name);
            let mut file = File::create(&path).unwrap();
            file.write_all(&bytes).unwrap();
            file.sync_all().unwrap();
            drop(file);
            artifacts.push(File::open(path).unwrap());
            expected_digests.insert(name.to_owned(), format!("{:x}", Sha256::digest(bytes)));
        }

        let mut manifest = Qwen35PackageManifest {
            schema_version: 2,
            checkpoint_revision: QWEN35_CHECKPOINT_REVISION.into(),
            key_id: key_id.into(),
            artifacts: expected_digests.clone(),
            signature_hex: String::new(),
        };
        manifest.signature_hex = signer
            .sign(&manifest.signing_bytes().unwrap())
            .to_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let trusted_keys = BTreeMap::from([(key_id.to_owned(), signer.verifying_key())]);
        let mut session = None;
        let hello = WorkerRequest::Hello {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.into(),
            requested_features: vec![
                HEARTBEAT_FEATURE_NAME.into(),
                ARTIFACT_READ_FEATURE_NAME.into(),
                QWEN35_PACKAGE_VERIFY_FEATURE_NAME.into(),
            ],
        };
        assert!(matches!(
            handle_request(hello, &mut session, &mut artifacts, &trusted_keys).unwrap(),
            WorkerResponse::Hello { features, .. }
                if features == vec![
                    WorkerFeature::HeartbeatV1,
                    WorkerFeature::ArtifactReadV1,
                    WorkerFeature::Qwen35PackageVerifyV1,
                ]
        ));

        let request = WorkerRequest::VerifyQwen35Package {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.into(),
            sequence: 1,
            manifest_json: serde_json::to_string(&manifest).unwrap(),
            artifact_slots: [0, 1, 2, 3, 4, 5],
        };
        let response =
            handle_request(request, &mut session, &mut artifacts, &trusted_keys).unwrap();
        assert!(serde_json::to_vec(&response).unwrap().len() <= MAX_FRAME_BYTES);
        assert!(matches!(
            response,
            WorkerResponse::Qwen35PackageVerified {
                sequence: 1,
                key_id: returned_key,
                checkpoint_revision,
                artifacts: receipts,
                ..
            } if returned_key == key_id
                && checkpoint_revision == QWEN35_CHECKPOINT_REVISION
                && receipts.len() == QWEN35_PACKAGE_ARTIFACT_COUNT
                && receipts.iter().all(|receipt| {
                    expected_digests.get(&receipt.name) == Some(&receipt.sha256)
                        && receipt.bytes > 0
                })
        ));

        assert!(matches!(
            handle_request(
                WorkerRequest::Ping {
                    protocol_version: PROTOCOL_VERSION,
                    challenge: challenge.into(),
                    sequence: 2,
                },
                &mut session,
                &mut artifacts,
                &trusted_keys,
            )
            .unwrap(),
            WorkerResponse::Pong {
                readiness: WorkerReadiness::PackageVerifiedModelNotAdmitted,
                ..
            }
        ));
    }

    #[cfg(feature = "qwen35-model-load")]
    #[test]
    fn candidate_load_requires_a_previously_verified_signed_package() {
        let challenge = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let mut session = Some(WorkerSession {
            challenge: challenge.into(),
            next_sequence: 1,
            features: vec![WorkerFeature::Qwen35CandidateLoadV1],
            artifacts: Vec::new(),
            verified_qwen35: None,
            loaded_qwen35: None,
            #[cfg(feature = "qwen35-model-generate")]
            pending_generation: None,
            #[cfg(feature = "qwen35-model-generate")]
            completed_generation: None,
        });
        let error = handle_request(
            WorkerRequest::LoadQwen35Candidate {
                protocol_version: PROTOCOL_VERSION,
                challenge: challenge.into(),
                sequence: 1,
            },
            &mut session,
            &mut Vec::new(),
            &BTreeMap::new(),
        )
        .expect_err("candidate load cannot precede signed package verification");
        assert!(error.to_string().contains("verified signed package"));
    }

    #[test]
    fn worker_does_not_verify_untrusted_or_misbound_packages() {
        let challenge = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let signer = SigningKey::from_bytes(&[43; 32]);
        let wrong_signer = SigningKey::from_bytes(&[47; 32]);
        let temporary_directory = tempfile::tempdir().unwrap();
        let mut artifacts = Vec::new();
        let mut digests = BTreeMap::new();
        for name in QWEN35_PACKAGE_ARTIFACTS {
            let bytes = format!("test artifact: {name}").into_bytes();
            let path = temporary_directory.path().join(name);
            std::fs::write(&path, &bytes).unwrap();
            artifacts.push(File::open(path).unwrap());
            digests.insert(name.to_owned(), format!("{:x}", Sha256::digest(bytes)));
        }
        let mut manifest = Qwen35PackageManifest {
            schema_version: 2,
            checkpoint_revision: QWEN35_CHECKPOINT_REVISION.into(),
            key_id: "trusted-test-key".into(),
            artifacts: digests,
            signature_hex: String::new(),
        };
        manifest.signature_hex = signer
            .sign(&manifest.signing_bytes().unwrap())
            .to_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let trusted_keys =
            BTreeMap::from([("trusted-test-key".to_owned(), wrong_signer.verifying_key())]);
        let mut session = None;
        handle_request(
            WorkerRequest::Hello {
                protocol_version: PROTOCOL_VERSION,
                challenge: challenge.into(),
                requested_features: vec![QWEN35_PACKAGE_VERIFY_FEATURE_NAME.into()],
            },
            &mut session,
            &mut artifacts,
            &trusted_keys,
        )
        .unwrap();
        assert!(
            handle_request(
                WorkerRequest::VerifyQwen35Package {
                    protocol_version: PROTOCOL_VERSION,
                    challenge: challenge.into(),
                    sequence: 1,
                    manifest_json: serde_json::to_string(&manifest).unwrap(),
                    artifact_slots: [0, 1, 2, 3, 4, 5],
                },
                &mut session,
                &mut artifacts,
                &trusted_keys,
            )
            .is_err()
        );

        let trusted_signer =
            BTreeMap::from([("trusted-test-key".to_owned(), signer.verifying_key())]);
        assert!(
            handle_request(
                WorkerRequest::VerifyQwen35Package {
                    protocol_version: PROTOCOL_VERSION,
                    challenge: challenge.into(),
                    sequence: 1,
                    manifest_json: serde_json::to_string(&manifest).unwrap(),
                    artifact_slots: [5, 4, 3, 2, 1, 0],
                },
                &mut session,
                &mut artifacts,
                &trusted_signer,
            )
            .is_err()
        );
    }
}
