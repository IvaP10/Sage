//! Owns the restricted first-party inference helper process.
//!
//! The isolated helper owns candidate package verification, optional loading,
//! and feature-gated constrained generation. A loaded candidate remains
//! unadmitted, and normal Core startup does not select this route as a provider.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use anyhow::{Context, bail};
#[cfg(test)]
use sage_inference_protocol::valid_sha256_hex;
use sage_inference_protocol::{
    ARTIFACT_READ_FEATURE_NAME, HEARTBEAT_FEATURE_NAME, MAX_FRAME_BYTES, PROTOCOL_VERSION,
    QWEN35_PACKAGE_ARTIFACT_COUNT, QWEN35_PACKAGE_VERIFY_FEATURE_NAME, WorkerFeature,
    WorkerReadiness, WorkerRequest, WorkerResponse,
};
#[cfg(feature = "qwen35-worker-verification")]
use sage_inference_protocol::{MAX_ARTIFACT_BYTES, VerifiedPackageArtifactReceipt};
#[cfg(feature = "qwen35-worker-generate")]
use sage_inference_protocol::{
    MAX_QWEN35_GENERATION_OUTPUT_BYTES, MAX_QWEN35_GENERATION_PREVIEW_CHUNK_BYTES,
    MAX_QWEN35_GENERATION_TOKENS, MAX_QWEN35_PROMPT_CHUNK_BYTES, MAX_QWEN35_SCHEMA_BYTES,
    MAX_QWEN35_SYSTEM_MESSAGE_BYTES, MAX_QWEN35_USER_MESSAGE_BYTES, QWEN35_GENERATE_FEATURE_NAME,
    QWEN35_GENERATION_PREVIEW_FEATURE_NAME, Qwen35PromptField, qwen35_generation_payload_sha256,
    sha256_hex,
};
#[cfg(feature = "qwen35-worker-load")]
use sage_inference_protocol::{
    MAX_QWEN35_MODEL_MEMORY_BYTES, QWEN35_ADMITTED_CONTEXT_TOKENS,
    QWEN35_CANDIDATE_LOAD_FEATURE_NAME,
};
#[cfg(feature = "qwen35-worker-verification")]
use sage_model_package::{
    QWEN35_PACKAGE_ARTIFACTS, Qwen35PackageManifest, VerifiedQwen35Package,
    application_trusted_qwen35_package_keys,
};
use sage_worker_fd::InheritedReadOnlyFiles;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(2);
#[cfg(feature = "qwen35-worker-load")]
const QWEN35_CANDIDATE_LOAD_TIMEOUT: Duration = Duration::from_secs(30 * 60);
#[cfg(feature = "qwen35-worker-generate")]
const QWEN35_GENERATION_FRAME_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const HEALTH_INTERVAL: Duration = Duration::from_secs(10);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_WORKER_RESTARTS: u8 = 3;

/// A Core-owned worker whose identity and readiness were checked at startup.
/// Dropping it terminates the child through Tokio's `kill_on_drop` behavior.
struct OwnedInferenceWorker {
    child: Child,
    stdin: tokio::process::ChildStdin,
    stdout: tokio::process::ChildStdout,
    challenge: String,
    process_id: u32,
    next_sequence: u64,
    #[cfg(feature = "qwen35-worker-generate")]
    supports_qwen35_generation_previews: bool,
    #[cfg(test)]
    artifact_slots: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct WorkerHelloFeatures {
    has_artifacts: bool,
    qwen35_package_verification: bool,
    qwen35_candidate_load: bool,
    qwen35_generation: bool,
    qwen35_generation_previews: bool,
}

/// Starts bounded lifecycle supervision for only the worker shipped beside
/// Sage Core. Source builds may opt into an explicit executable for testing.
pub fn supervise_bundled() -> anyhow::Result<Option<JoinHandle<()>>> {
    let Some(path) = bundled_path()? else {
        return Ok(None);
    };
    Ok(Some(tokio::spawn(supervise(path))))
}

async fn supervise(path: PathBuf) {
    let mut retry = 0;
    loop {
        match launch(&path).await {
            Ok(mut worker) => {
                tracing::info!(
                    "first-party inference worker negotiated its bounded health protocol"
                );
                match worker.wait_until_exit_or_failed_health().await {
                    Ok(status) => tracing::warn!(
                        %status,
                        "first-party inference worker exited; model generation remains unavailable"
                    ),
                    Err(error) => tracing::warn!(
                        %error,
                        "first-party inference worker health check failed"
                    ),
                }
            }
            Err(error) => tracing::warn!(
                %error,
                "first-party inference worker failed to start; model generation remains unavailable"
            ),
        }

        let Some(delay) = restart_delay(retry) else {
            tracing::error!(
                restarts = retry,
                "first-party inference worker restart limit reached; model generation remains unavailable"
            );
            return;
        };
        retry += 1;
        sleep(delay).await;
    }
}

impl OwnedInferenceWorker {
    async fn wait_until_exit_or_failed_health(&mut self) -> anyhow::Result<ExitStatus> {
        loop {
            if let Ok(status) = timeout(HEALTH_INTERVAL, self.child.wait()).await {
                return status.context("observe inference worker process exit");
            }

            match timeout(HEALTH_TIMEOUT, self.heartbeat()).await {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => return Err(error).context("inference worker heartbeat failed"),
                Err(_) => bail!("inference worker heartbeat exceeded its 2 second deadline"),
            }
        }
    }

    async fn heartbeat(&mut self) -> anyhow::Result<WorkerReadiness> {
        let sequence = self.next_sequence;
        let request = WorkerRequest::Ping {
            protocol_version: PROTOCOL_VERSION,
            challenge: self.challenge.clone(),
            sequence,
        };
        write_frame(&mut self.stdin, &serde_json::to_vec(&request)?).await?;
        let response =
            serde_json::from_slice::<WorkerResponse>(&read_frame(&mut self.stdout).await?)?;
        let readiness = validate_pong(response, &self.challenge, self.process_id, sequence)?;
        self.next_sequence = sequence
            .checked_add(1)
            .context("inference worker health sequence exhausted")?;
        Ok(readiness)
    }

    #[cfg(test)]
    async fn verify_inherited_artifact(
        &mut self,
        artifact_slot: u8,
        expected_bytes: u64,
        expected_sha256: &str,
    ) -> anyhow::Result<()> {
        if usize::from(artifact_slot) >= self.artifact_slots
            || expected_bytes == 0
            || !valid_sha256_hex(expected_sha256)
        {
            bail!("inference worker artifact request is outside its admitted bounds");
        }
        let sequence = self.next_sequence;
        let request = WorkerRequest::VerifyArtifact {
            protocol_version: PROTOCOL_VERSION,
            challenge: self.challenge.clone(),
            sequence,
            artifact_slot,
            expected_bytes,
            expected_sha256: expected_sha256.to_owned(),
        };
        write_frame(&mut self.stdin, &serde_json::to_vec(&request)?).await?;
        let response =
            serde_json::from_slice::<WorkerResponse>(&read_frame(&mut self.stdout).await?)?;
        validate_artifact_verified(
            response,
            &self.challenge,
            self.process_id,
            sequence,
            artifact_slot,
            expected_bytes,
            expected_sha256,
        )?;
        self.next_sequence = sequence
            .checked_add(1)
            .context("inference worker artifact sequence exhausted")?;
        Ok(())
    }
}

fn restart_delay(retry: u8) -> Option<Duration> {
    if retry >= MAX_WORKER_RESTARTS {
        return None;
    }
    Some(Duration::from_secs(1_u64 << retry))
}

fn bundled_path() -> anyhow::Result<Option<PathBuf>> {
    #[cfg(all(target_os = "macos", debug_assertions))]
    if let Some(path) = std::env::var_os("SAGE_INFERENCE_WORKER_EXECUTABLE") {
        return canonical_executable(PathBuf::from(path)).map(Some);
    }

    #[cfg(target_os = "macos")]
    {
        let core = std::env::current_exe().context("locate running Sage Core executable")?;
        let helpers = core
            .parent()
            .context("Sage Core executable has no containing directory")?;
        let candidate =
            helpers.join("SageInferenceWorker.app/Contents/MacOS/sage-inference-worker");
        if !candidate.is_file() {
            return Ok(None);
        }
        canonical_executable(candidate).map(Some)
    }

    #[cfg(not(target_os = "macos"))]
    {
        Ok(None)
    }
}

fn canonical_executable(path: PathBuf) -> anyhow::Result<PathBuf> {
    if !path.is_absolute() {
        bail!("inference worker path must be absolute");
    }
    let path = path
        .canonicalize()
        .context("resolve first-party inference worker executable")?;
    if !path.is_file() {
        bail!("first-party inference worker is not a regular file");
    }
    Ok(path)
}

async fn launch(path: &Path) -> anyhow::Result<OwnedInferenceWorker> {
    launch_with_features(path, &[], false, false, false, false).await
}

#[cfg(test)]
async fn launch_with_readonly_files(
    path: &Path,
    files: &[&File],
) -> anyhow::Result<OwnedInferenceWorker> {
    launch_with_features(path, files, false, false, false, false).await
}

async fn launch_with_features(
    path: &Path,
    files: &[&File],
    request_qwen35_package_verification: bool,
    request_qwen35_candidate_load: bool,
    request_qwen35_generation: bool,
    request_qwen35_generation_previews: bool,
) -> anyhow::Result<OwnedInferenceWorker> {
    if request_qwen35_candidate_load && !request_qwen35_package_verification {
        bail!("Qwen candidate loading requires signed package verification");
    }
    if request_qwen35_generation && !request_qwen35_candidate_load {
        bail!("Qwen generation requires candidate loading");
    }
    if request_qwen35_generation_previews && !request_qwen35_generation {
        bail!("Qwen answer previews require generation");
    }
    if request_qwen35_package_verification && files.len() != QWEN35_PACKAGE_ARTIFACT_COUNT {
        bail!("Qwen package verification requires exactly six inherited artifact handles");
    }
    let challenge = random_challenge()?;
    let inherited = InheritedReadOnlyFiles::duplicate(files)
        .context("duplicate verified read-only model artifact handles")?;
    let descriptors = inherited.raw_descriptors().collect::<Vec<_>>();
    let mut command = Command::new(path);
    command
        .env_clear()
        .current_dir("/")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    for descriptor in &descriptors {
        command.arg("--artifact-fd").arg(descriptor.to_string());
    }
    inherited
        .configure_child(command.as_std_mut())
        .context("configure read-only model artifact handles for the worker")?;
    let mut child = command
        .spawn()
        .with_context(|| format!("start bundled inference worker at {}", path.display()))?;
    let pid = child
        .id()
        .context("inference worker has no process identity")?;
    let Some(mut stdin) = child.stdin.take() else {
        terminate(&mut child).await;
        bail!("inference worker stdin pipe was unavailable");
    };
    let Some(mut stdout) = child.stdout.take() else {
        terminate(&mut child).await;
        bail!("inference worker stdout pipe was unavailable");
    };

    let result = timeout(
        HANDSHAKE_TIMEOUT,
        exchange_hello(
            &mut stdin,
            &mut stdout,
            &challenge,
            pid,
            WorkerHelloFeatures {
                has_artifacts: !descriptors.is_empty(),
                qwen35_package_verification: request_qwen35_package_verification,
                qwen35_candidate_load: request_qwen35_candidate_load,
                qwen35_generation: request_qwen35_generation,
                qwen35_generation_previews: request_qwen35_generation_previews,
            },
        ),
    )
    .await;
    match result {
        Ok(Ok(supports_qwen35_generation_previews)) => {
            #[cfg(not(feature = "qwen35-worker-generate"))]
            let _ = supports_qwen35_generation_previews;
            Ok(OwnedInferenceWorker {
                child,
                stdin,
                stdout,
                challenge,
                process_id: pid,
                next_sequence: 1,
                #[cfg(feature = "qwen35-worker-generate")]
                supports_qwen35_generation_previews,
                #[cfg(test)]
                artifact_slots: descriptors.len(),
            })
        }
        Ok(Err(error)) => {
            terminate(&mut child).await;
            Err(error).context("inference worker handshake failed")
        }
        Err(_) => {
            terminate(&mut child).await;
            bail!("inference worker handshake exceeded its 2 second deadline")
        }
    }
}

async fn terminate(child: &mut Child) {
    let _ = child.kill().await;
    let _ = timeout(Duration::from_secs(1), child.wait()).await;
}

async fn exchange_hello<W, R>(
    writer: &mut W,
    reader: &mut R,
    challenge: &str,
    expected_pid: u32,
    requested: WorkerHelloFeatures,
) -> anyhow::Result<bool>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let mut requested_features = vec![HEARTBEAT_FEATURE_NAME.to_owned()];
    if requested.has_artifacts {
        requested_features.push(ARTIFACT_READ_FEATURE_NAME.to_owned());
    }
    if requested.qwen35_package_verification {
        requested_features.push(QWEN35_PACKAGE_VERIFY_FEATURE_NAME.to_owned());
    }
    #[cfg(feature = "qwen35-worker-load")]
    if requested.qwen35_candidate_load {
        requested_features.push(QWEN35_CANDIDATE_LOAD_FEATURE_NAME.to_owned());
    }
    #[cfg(not(feature = "qwen35-worker-load"))]
    if requested.qwen35_candidate_load {
        bail!("Qwen candidate loading is not enabled in this Core build");
    }
    #[cfg(feature = "qwen35-worker-generate")]
    if requested.qwen35_generation {
        requested_features.push(QWEN35_GENERATE_FEATURE_NAME.to_owned());
    }
    #[cfg(not(feature = "qwen35-worker-generate"))]
    if requested.qwen35_generation {
        bail!("Qwen generation is not enabled in this Core build");
    }
    #[cfg(feature = "qwen35-worker-generate")]
    if requested.qwen35_generation_previews {
        requested_features.push(QWEN35_GENERATION_PREVIEW_FEATURE_NAME.to_owned());
    }
    #[cfg(not(feature = "qwen35-worker-generate"))]
    if requested.qwen35_generation_previews {
        bail!("Qwen generation previews are not enabled in this Core build");
    }
    let request = WorkerRequest::Hello {
        protocol_version: PROTOCOL_VERSION,
        challenge: challenge.to_owned(),
        requested_features,
    };
    write_frame(writer, &serde_json::to_vec(&request)?).await?;
    let response = serde_json::from_slice::<WorkerResponse>(&read_frame(reader).await?)?;
    validate_hello(response, challenge, expected_pid, requested)
}

fn validate_hello(
    response: WorkerResponse,
    challenge: &str,
    expected_pid: u32,
    requested: WorkerHelloFeatures,
) -> anyhow::Result<bool> {
    match response {
        WorkerResponse::Hello {
            protocol_version,
            challenge: observed_challenge,
            process_id,
            features,
            readiness,
        } if protocol_version == PROTOCOL_VERSION
            && observed_challenge == challenge
            && process_id == expected_pid
            && readiness == WorkerReadiness::ModelNotAdmitted =>
        {
            let requested_previews = requested.qwen35_generation_previews;
            let mut legacy_request = requested;
            legacy_request.qwen35_generation_previews = false;
            let legacy_features = expected_features(legacy_request);
            if features == legacy_features {
                return Ok(false);
            }
            if requested_previews {
                let mut preview_features = legacy_features;
                preview_features.push(WorkerFeature::Qwen35GenerationPreviewV1);
                if features == preview_features {
                    return Ok(true);
                }
            }
            bail!("inference worker returned an unexpected feature set")
        }
        _ => bail!("inference worker returned an unexpected identity, feature, or readiness state"),
    }
}

fn expected_features(requested: WorkerHelloFeatures) -> Vec<WorkerFeature> {
    let mut expected = vec![WorkerFeature::HeartbeatV1];
    if requested.has_artifacts {
        expected.push(WorkerFeature::ArtifactReadV1);
    }
    if requested.qwen35_package_verification {
        expected.push(WorkerFeature::Qwen35PackageVerifyV1);
    }
    #[cfg(feature = "qwen35-worker-load")]
    if requested.qwen35_candidate_load {
        expected.push(WorkerFeature::Qwen35CandidateLoadV1);
    }
    #[cfg(feature = "qwen35-worker-generate")]
    if requested.qwen35_generation {
        expected.push(WorkerFeature::Qwen35GenerateV1);
    }
    expected
}

#[cfg(feature = "qwen35-worker-generate")]
#[test]
fn preview_handshake_accepts_legacy_workers_and_detects_the_extension() {
    let challenge = "ef".repeat(32);
    let requested = WorkerHelloFeatures {
        has_artifacts: true,
        qwen35_package_verification: true,
        qwen35_candidate_load: true,
        qwen35_generation: true,
        qwen35_generation_previews: true,
    };
    let mut legacy_request = requested;
    legacy_request.qwen35_generation_previews = false;
    let legacy_features = expected_features(legacy_request);
    let legacy_response = WorkerResponse::Hello {
        protocol_version: PROTOCOL_VERSION,
        challenge: challenge.clone(),
        process_id: 42,
        features: legacy_features.clone(),
        readiness: WorkerReadiness::ModelNotAdmitted,
    };
    assert!(!validate_hello(legacy_response, &challenge, 42, requested).unwrap());

    let mut preview_features = legacy_features;
    preview_features.push(WorkerFeature::Qwen35GenerationPreviewV1);
    let preview_response = WorkerResponse::Hello {
        protocol_version: PROTOCOL_VERSION,
        challenge: challenge.clone(),
        process_id: 42,
        features: preview_features.clone(),
        readiness: WorkerReadiness::ModelNotAdmitted,
    };
    assert!(validate_hello(preview_response, &challenge, 42, requested).unwrap());

    let unrequested_preview = WorkerResponse::Hello {
        protocol_version: PROTOCOL_VERSION,
        challenge: challenge.clone(),
        process_id: 42,
        features: preview_features,
        readiness: WorkerReadiness::ModelNotAdmitted,
    };
    let mut without_preview_request = requested;
    without_preview_request.qwen35_generation_previews = false;
    assert!(validate_hello(unrequested_preview, &challenge, 42, without_preview_request).is_err());
}

#[cfg(test)]
fn validate_artifact_verified(
    response: WorkerResponse,
    challenge: &str,
    expected_pid: u32,
    expected_sequence: u64,
    expected_slot: u8,
    expected_bytes: u64,
    expected_sha256: &str,
) -> anyhow::Result<()> {
    match response {
        WorkerResponse::ArtifactVerified {
            protocol_version,
            challenge: observed_challenge,
            process_id,
            sequence,
            artifact_slot,
            bytes,
            sha256,
        } if protocol_version == PROTOCOL_VERSION
            && observed_challenge == challenge
            && process_id == expected_pid
            && sequence == expected_sequence
            && artifact_slot == expected_slot
            && bytes == expected_bytes
            && sha256 == expected_sha256 =>
        {
            Ok(())
        }
        _ => bail!("inference worker artifact receipt failed identity or digest validation"),
    }
}

fn validate_pong(
    response: WorkerResponse,
    challenge: &str,
    expected_pid: u32,
    expected_sequence: u64,
) -> anyhow::Result<WorkerReadiness> {
    match response {
        WorkerResponse::Pong {
            protocol_version,
            challenge: observed_challenge,
            process_id,
            sequence,
            readiness,
        } if protocol_version == PROTOCOL_VERSION
            && observed_challenge == challenge
            && process_id == expected_pid
            && sequence == expected_sequence
            && matches!(
                readiness,
                WorkerReadiness::ModelNotAdmitted
                    | WorkerReadiness::PackageVerifiedModelNotAdmitted
                    | WorkerReadiness::CandidateLoadedModelNotAdmitted
            ) =>
        {
            Ok(readiness)
        }
        _ => bail!("inference worker heartbeat response failed identity or sequence validation"),
    }
}

#[cfg(feature = "qwen35-worker-verification")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35WorkerPackageReceipt {
    pub manifest_sha256: String,
    pub key_id: String,
    pub checkpoint_revision: String,
    pub artifacts: Vec<VerifiedPackageArtifactReceipt>,
}

/// A live worker that has verified a signed Qwen package and retains the exact
/// inherited read-only artifact handles. Candidate loading is separately gated.
#[cfg(feature = "qwen35-worker-verification")]
#[must_use = "dropping the worker closes its verified artifact handles"]
pub struct VerifiedQwen35PackageWorker {
    worker: OwnedInferenceWorker,
    receipt: Qwen35WorkerPackageReceipt,
}

#[cfg(feature = "qwen35-worker-verification")]
impl VerifiedQwen35PackageWorker {
    pub fn receipt(&self) -> &Qwen35WorkerPackageReceipt {
        &self.receipt
    }

    pub async fn health_check(&mut self) -> anyhow::Result<()> {
        let readiness = self.worker.heartbeat().await?;
        if readiness != WorkerReadiness::PackageVerifiedModelNotAdmitted {
            bail!("inference worker did not retain the verified Qwen package");
        }
        Ok(())
    }

    /// Load the verified checkpoint inside its isolated worker. Consuming this
    /// handle means a cancelled request drops and terminates the worker rather
    /// than leaving a half-loaded model running without an owner.
    #[cfg(feature = "qwen35-worker-load")]
    pub async fn load_candidate(self) -> anyhow::Result<LoadedQwen35CandidateWorker> {
        let Self {
            mut worker,
            receipt: package_receipt,
        } = self;
        let sequence = worker.next_sequence;
        let request = WorkerRequest::LoadQwen35Candidate {
            protocol_version: PROTOCOL_VERSION,
            challenge: worker.challenge.clone(),
            sequence,
        };
        write_frame(&mut worker.stdin, &serde_json::to_vec(&request)?).await?;
        let response = timeout(
            QWEN35_CANDIDATE_LOAD_TIMEOUT,
            read_frame(&mut worker.stdout),
        )
        .await
        .context("Qwen candidate loading exceeded its 30 minute deadline")??;
        let response = serde_json::from_slice::<WorkerResponse>(&response)?;
        let load_receipt = validate_qwen35_candidate_load_receipt(
            response,
            &worker.challenge,
            worker.process_id,
            sequence,
            &package_receipt,
        )?;
        worker.next_sequence = sequence
            .checked_add(1)
            .context("Qwen candidate-load sequence exhausted")?;
        Ok(LoadedQwen35CandidateWorker {
            worker,
            package_receipt,
            load_receipt,
        })
    }
}

#[cfg(feature = "qwen35-worker-load")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35WorkerLoadReceipt {
    pub manifest_sha256: String,
    pub checkpoint_revision: String,
    pub maximum_context_tokens: u32,
    pub estimated_memory_bytes: u64,
}

#[cfg(feature = "qwen35-worker-load")]
#[must_use = "dropping the worker releases the resident candidate model"]
pub struct LoadedQwen35CandidateWorker {
    worker: OwnedInferenceWorker,
    package_receipt: Qwen35WorkerPackageReceipt,
    load_receipt: Qwen35WorkerLoadReceipt,
}

#[cfg(feature = "qwen35-worker-load")]
impl LoadedQwen35CandidateWorker {
    pub fn package_receipt(&self) -> &Qwen35WorkerPackageReceipt {
        &self.package_receipt
    }

    pub fn load_receipt(&self) -> &Qwen35WorkerLoadReceipt {
        &self.load_receipt
    }

    pub async fn health_check(&mut self) -> anyhow::Result<()> {
        let readiness = self.worker.heartbeat().await?;
        if readiness != WorkerReadiness::CandidateLoadedModelNotAdmitted {
            bail!("inference worker no longer retains the loaded Qwen candidate");
        }
        Ok(())
    }

    /// Generate one planner JSON response in the isolated, package-bound
    /// worker. This consumes the handle so cancellation by dropping the future
    /// also drops and terminates the child instead of leaving orphaned decode
    /// work or a desynchronized pipe behind.
    #[cfg(feature = "qwen35-worker-generate")]
    pub async fn generate_planner_json(
        self,
        system_message: &str,
        user_message: &str,
        output_schema: &serde_json::Value,
        maximum_new_tokens: u16,
    ) -> anyhow::Result<(Self, String)> {
        self.generate_planner_json_with_previews(
            system_message,
            user_message,
            output_schema,
            maximum_new_tokens,
            |_| {},
        )
        .await
    }

    /// Generate a planner turn while forwarding append-only root-answer
    /// previews as they arrive. Preview text is untrusted and incomplete: a
    /// caller may display it as progress but must wait for the final JSON and
    /// completed-turn validation before using any result or proposed action.
    #[cfg(feature = "qwen35-worker-generate")]
    pub async fn generate_planner_json_with_previews<F>(
        self,
        system_message: &str,
        user_message: &str,
        output_schema: &serde_json::Value,
        maximum_new_tokens: u16,
        mut on_answer_preview: F,
    ) -> anyhow::Result<(Self, String)>
    where
        F: FnMut(&str) + Send,
    {
        let Self {
            mut worker,
            package_receipt,
            load_receipt,
        } = self;
        if !output_schema.is_object()
            || system_message.is_empty()
            || system_message.len() > MAX_QWEN35_SYSTEM_MESSAGE_BYTES
            || user_message.is_empty()
            || user_message.len() > MAX_QWEN35_USER_MESSAGE_BYTES
            || maximum_new_tokens == 0
            || maximum_new_tokens > MAX_QWEN35_GENERATION_TOKENS
        {
            bail!("Qwen planner prompt is outside its bounded input contract");
        }
        let schema_text = checked_qwen35_output_schema(system_message, output_schema)?;
        let generation_id = random_challenge()?;
        let payload_sha256 = qwen35_generation_payload_sha256(
            schema_text.as_bytes(),
            system_message.as_bytes(),
            user_message.as_bytes(),
        );
        let begin_sequence = worker.next_sequence;
        let begin = WorkerRequest::BeginQwen35Generation {
            protocol_version: PROTOCOL_VERSION,
            challenge: worker.challenge.clone(),
            sequence: begin_sequence,
            generation_id: generation_id.clone(),
            maximum_new_tokens,
            schema_bytes: u32::try_from(schema_text.len())?,
            system_message_bytes: u32::try_from(system_message.len())?,
            user_message_bytes: u32::try_from(user_message.len())?,
            payload_sha256,
        };
        write_frame(&mut worker.stdin, &serde_json::to_vec(&begin)?).await?;
        let started = timeout(HEALTH_TIMEOUT, read_frame(&mut worker.stdout))
            .await
            .context("Qwen generation start acknowledgment timed out")??;
        validate_qwen35_generation_started(
            serde_json::from_slice::<WorkerResponse>(&started)?,
            &worker.challenge,
            worker.process_id,
            begin_sequence,
            &generation_id,
        )?;
        worker.next_sequence = begin_sequence
            .checked_add(1)
            .context("Qwen generation sequence exhausted")?;

        send_qwen35_prompt_chunks(
            &mut worker,
            &generation_id,
            Qwen35PromptField::Schema,
            &schema_text,
        )
        .await?;
        send_qwen35_prompt_chunks(
            &mut worker,
            &generation_id,
            Qwen35PromptField::SystemMessage,
            system_message,
        )
        .await?;
        send_qwen35_prompt_chunks(
            &mut worker,
            &generation_id,
            Qwen35PromptField::UserMessage,
            user_message,
        )
        .await?;

        let generation_sequence = worker.next_sequence;
        let generate = WorkerRequest::GenerateQwen35 {
            protocol_version: PROTOCOL_VERSION,
            challenge: worker.challenge.clone(),
            sequence: generation_sequence,
            generation_id: generation_id.clone(),
        };
        write_frame(&mut worker.stdin, &serde_json::to_vec(&generate)?).await?;
        worker.next_sequence = generation_sequence
            .checked_add(1)
            .context("Qwen generation sequence exhausted")?;

        let mut collector = Qwen35GenerationOutputCollector::new(
            worker.challenge.clone(),
            worker.process_id,
            generation_sequence,
            generation_id,
            worker.supports_qwen35_generation_previews,
        );
        let output = loop {
            let frame = timeout(
                QWEN35_GENERATION_FRAME_TIMEOUT,
                read_frame(&mut worker.stdout),
            )
            .await
            .context("Qwen generation exceeded its response deadline")??;
            let response = serde_json::from_slice::<WorkerResponse>(&frame)?;
            if let Some(output) = collector.accept(response, &mut on_answer_preview)? {
                break output;
            }
        };
        crate::model::validate_turn_json(&output)
            .context("isolated Qwen worker returned an invalid planner turn")?;
        Ok((
            Self {
                worker,
                package_receipt,
                load_receipt,
            },
            output,
        ))
    }
}

#[cfg(feature = "qwen35-worker-generate")]
fn checked_qwen35_output_schema(
    system_message: &str,
    output_schema: &serde_json::Value,
) -> anyhow::Result<String> {
    if !output_schema.is_object() {
        bail!("Qwen planner schema must be a JSON object");
    }
    let schema_text = serde_json::to_string(output_schema)?;
    if schema_text.is_empty() || schema_text.len() > MAX_QWEN35_SCHEMA_BYTES {
        bail!("Qwen planner schema is outside its byte bound");
    }
    if !system_message.ends_with(&format!("\n\nClosed output schema:\n{schema_text}")) {
        bail!("Qwen system prompt and constrained output schema do not match");
    }
    Ok(schema_text)
}

#[cfg(feature = "qwen35-worker-generate")]
async fn send_qwen35_prompt_chunks(
    worker: &mut OwnedInferenceWorker,
    generation_id: &str,
    field: Qwen35PromptField,
    text: &str,
) -> anyhow::Result<()> {
    let mut offset = 0usize;
    while offset < text.len() {
        let mut end = offset
            .saturating_add(MAX_QWEN35_PROMPT_CHUNK_BYTES)
            .min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == offset {
            bail!("Qwen prompt chunk did not advance at a UTF-8 boundary");
        }
        let sequence = worker.next_sequence;
        let request = WorkerRequest::AppendQwen35PromptChunk {
            protocol_version: PROTOCOL_VERSION,
            challenge: worker.challenge.clone(),
            sequence,
            generation_id: generation_id.to_owned(),
            field,
            offset: u32::try_from(offset)?,
            chunk: text[offset..end].to_owned(),
        };
        write_frame(&mut worker.stdin, &serde_json::to_vec(&request)?).await?;
        worker.next_sequence = sequence
            .checked_add(1)
            .context("Qwen prompt chunk sequence exhausted")?;
        offset = end;
    }
    Ok(())
}

#[cfg(feature = "qwen35-worker-generate")]
fn validate_qwen35_generation_started(
    response: WorkerResponse,
    challenge: &str,
    process_id: u32,
    sequence: u64,
    generation_id: &str,
) -> anyhow::Result<()> {
    match response {
        WorkerResponse::Qwen35GenerationStarted {
            protocol_version,
            challenge: observed_challenge,
            process_id: observed_process_id,
            sequence: observed_sequence,
            generation_id: observed_generation_id,
        } if protocol_version == PROTOCOL_VERSION
            && observed_challenge == challenge
            && observed_process_id == process_id
            && observed_sequence == sequence
            && observed_generation_id == generation_id =>
        {
            Ok(())
        }
        _ => bail!("Qwen worker returned an unexpected generation-start receipt"),
    }
}

#[cfg(feature = "qwen35-worker-generate")]
struct Qwen35GenerationOutputCollector {
    challenge: String,
    process_id: u32,
    sequence: u64,
    generation_id: String,
    previews_negotiated: bool,
    next_chunk: u32,
    preview: String,
    output_started: bool,
    output: Vec<u8>,
}

#[cfg(feature = "qwen35-worker-generate")]
impl Qwen35GenerationOutputCollector {
    fn new(
        challenge: String,
        process_id: u32,
        sequence: u64,
        generation_id: String,
        previews_negotiated: bool,
    ) -> Self {
        Self {
            challenge,
            process_id,
            sequence,
            generation_id,
            previews_negotiated,
            next_chunk: 0,
            preview: String::new(),
            output_started: false,
            output: Vec::new(),
        }
    }

    fn accept<F>(
        &mut self,
        response: WorkerResponse,
        on_answer_preview: &mut F,
    ) -> anyhow::Result<Option<String>>
    where
        F: FnMut(&str),
    {
        match response {
            WorkerResponse::Qwen35GenerationPreviewChunk {
                protocol_version,
                challenge,
                process_id,
                sequence,
                generation_id,
                preview_offset,
                chunk,
            } => {
                let next_len = self
                    .preview
                    .len()
                    .checked_add(chunk.len())
                    .context("Qwen preview byte count overflowed")?;
                if !self.previews_negotiated
                    || protocol_version != PROTOCOL_VERSION
                    || challenge != self.challenge
                    || process_id != self.process_id
                    || sequence != self.sequence
                    || generation_id != self.generation_id
                    || usize::try_from(preview_offset)? != self.preview.len()
                    || chunk.is_empty()
                    || chunk.len() > MAX_QWEN35_GENERATION_PREVIEW_CHUNK_BYTES
                    || next_len > MAX_QWEN35_GENERATION_OUTPUT_BYTES
                    || self.output_started
                {
                    bail!("Qwen answer preview failed identity, ordering or bound checks");
                }
                self.preview.push_str(&chunk);
                on_answer_preview(&self.preview);
                Ok(None)
            }
            WorkerResponse::Qwen35GenerationOutputChunk {
                protocol_version,
                challenge,
                process_id,
                sequence,
                generation_id,
                chunk_index,
                chunk,
            } => {
                let next_len = self
                    .output
                    .len()
                    .checked_add(chunk.len())
                    .context("Qwen output byte count overflowed")?;
                if protocol_version != PROTOCOL_VERSION
                    || challenge != self.challenge
                    || process_id != self.process_id
                    || sequence != self.sequence
                    || generation_id != self.generation_id
                    || chunk_index != self.next_chunk
                    || chunk.is_empty()
                    || chunk.len() > MAX_QWEN35_PROMPT_CHUNK_BYTES
                    || next_len > MAX_QWEN35_GENERATION_OUTPUT_BYTES
                {
                    bail!("Qwen generation output chunk failed identity or bound checks");
                }
                self.output_started = true;
                self.output.extend_from_slice(chunk.as_bytes());
                self.next_chunk = self
                    .next_chunk
                    .checked_add(1)
                    .context("Qwen output chunk count overflowed")?;
                Ok(None)
            }
            WorkerResponse::Qwen35GenerationFinished {
                protocol_version,
                challenge,
                process_id,
                sequence,
                generation_id,
                output_chunks,
                output_bytes,
                output_sha256,
            } => {
                if protocol_version != PROTOCOL_VERSION
                    || challenge != self.challenge
                    || process_id != self.process_id
                    || sequence != self.sequence
                    || generation_id != self.generation_id
                    || output_chunks != self.next_chunk
                    || output_chunks == 0
                    || usize::try_from(output_bytes)? != self.output.len()
                    || self.output.is_empty()
                    || output_sha256 != sha256_hex(&self.output)
                {
                    bail!("Qwen generation terminal receipt failed identity or digest checks");
                }
                let output = String::from_utf8(std::mem::take(&mut self.output))?;
                let final_turn = serde_json::from_str::<serde_json::Value>(&output)?;
                let final_answer = final_turn
                    .get("answer")
                    .and_then(serde_json::Value::as_str)
                    .context("Qwen planner output omitted its root answer field")?;
                if !final_answer.starts_with(&self.preview) {
                    bail!("Qwen final answer does not extend its streamed preview");
                }
                Ok(Some(output))
            }
            _ => bail!("Qwen worker returned an unexpected generation response"),
        }
    }
}

/// Launch the bundled worker with six caller-opened read-only package files.
/// Sage validates the signed manifest locally and in the isolated worker. On
/// success, the returned object keeps the child and its verified file handles
/// alive for a later first-party loader protocol. Files must be supplied in
/// `QWEN35_PACKAGE_ARTIFACTS` order. This verifies their bytes at that time; it
/// does not make a mutable backing file immutable for later tensor import.
#[cfg(feature = "qwen35-worker-verification")]
pub async fn launch_verified_qwen35_package(
    manifest_json: &str,
    artifacts: &[&File],
) -> anyhow::Result<VerifiedQwen35PackageWorker> {
    let executable = bundled_path()?.context("bundled inference worker is unavailable")?;
    launch_verified_qwen35_package_at(&executable, manifest_json, artifacts).await
}

#[cfg(feature = "qwen35-worker-verification")]
async fn launch_verified_qwen35_package_at(
    executable: &Path,
    manifest_json: &str,
    artifacts: &[&File],
) -> anyhow::Result<VerifiedQwen35PackageWorker> {
    if artifacts.len() != QWEN35_PACKAGE_ARTIFACT_COUNT
        || manifest_json.is_empty()
        || manifest_json.len() > MAX_FRAME_BYTES
    {
        bail!("Qwen package request is outside its bounded input shape");
    }
    let manifest = serde_json::from_str::<Qwen35PackageManifest>(manifest_json)
        .context("parse the closed Qwen package manifest")?;
    let trusted_keys = application_trusted_qwen35_package_keys()
        .context("load the build-embedded Qwen package trust roots")?;
    let package = manifest
        .verify_signature(&trusted_keys)
        .context("verify the Qwen package against Sage's build-embedded trust roots")?;

    let mut worker = launch_with_features(
        executable,
        artifacts,
        true,
        cfg!(feature = "qwen35-worker-load"),
        cfg!(feature = "qwen35-worker-generate"),
        cfg!(feature = "qwen35-worker-generate"),
    )
    .await?;
    let sequence = worker.next_sequence;
    let request = WorkerRequest::VerifyQwen35Package {
        protocol_version: PROTOCOL_VERSION,
        challenge: worker.challenge.clone(),
        sequence,
        manifest_json: manifest_json.to_owned(),
        artifact_slots: [0, 1, 2, 3, 4, 5],
    };
    write_frame(&mut worker.stdin, &serde_json::to_vec(&request)?).await?;
    let response =
        serde_json::from_slice::<WorkerResponse>(&read_frame(&mut worker.stdout).await?)?;
    let receipt = validate_qwen35_package_receipt(
        response,
        &worker.challenge,
        worker.process_id,
        sequence,
        artifacts,
        &manifest,
        &package,
    )?;
    worker.next_sequence = sequence
        .checked_add(1)
        .context("inference worker package sequence exhausted")?;
    let verified = VerifiedQwen35PackageWorker { worker, receipt };
    let mut verified = verified;
    verified.health_check().await?;
    Ok(verified)
}

#[cfg(feature = "qwen35-worker-verification")]
fn validate_qwen35_package_receipt(
    response: WorkerResponse,
    challenge: &str,
    expected_pid: u32,
    expected_sequence: u64,
    artifact_files: &[&File],
    manifest: &Qwen35PackageManifest,
    package: &VerifiedQwen35Package,
) -> anyhow::Result<Qwen35WorkerPackageReceipt> {
    let WorkerResponse::Qwen35PackageVerified {
        protocol_version,
        challenge: observed_challenge,
        process_id,
        sequence,
        manifest_sha256,
        key_id,
        checkpoint_revision,
        artifacts,
    } = response
    else {
        bail!("inference worker returned no signed Qwen package receipt");
    };
    if protocol_version != PROTOCOL_VERSION
        || observed_challenge != challenge
        || process_id != expected_pid
        || sequence != expected_sequence
        || manifest_sha256 != package.manifest_sha256()
        || key_id != package.key_id()
        || checkpoint_revision != package.checkpoint_revision()
        || artifacts.len() != QWEN35_PACKAGE_ARTIFACT_COUNT
        || artifact_files.len() != QWEN35_PACKAGE_ARTIFACT_COUNT
    {
        bail!("inference worker Qwen package receipt failed identity validation");
    }
    for ((expected_name, file), receipt) in QWEN35_PACKAGE_ARTIFACTS
        .into_iter()
        .zip(artifact_files)
        .zip(&artifacts)
    {
        let expected_digest = manifest
            .artifacts
            .get(expected_name)
            .context("signed Qwen manifest omitted an expected artifact")?;
        let expected_bytes = file.metadata()?.len();
        if receipt.name != expected_name
            || &receipt.sha256 != expected_digest
            || receipt.bytes != expected_bytes
            || receipt.bytes == 0
            || receipt.bytes > MAX_ARTIFACT_BYTES
        {
            bail!(
                "inference worker Qwen package artifact receipt failed digest or size validation"
            );
        }
    }
    Ok(Qwen35WorkerPackageReceipt {
        manifest_sha256,
        key_id,
        checkpoint_revision,
        artifacts,
    })
}

#[cfg(feature = "qwen35-worker-load")]
fn validate_qwen35_candidate_load_receipt(
    response: WorkerResponse,
    challenge: &str,
    expected_pid: u32,
    expected_sequence: u64,
    package_receipt: &Qwen35WorkerPackageReceipt,
) -> anyhow::Result<Qwen35WorkerLoadReceipt> {
    let WorkerResponse::Qwen35CandidateLoaded {
        protocol_version,
        challenge: observed_challenge,
        process_id,
        sequence,
        manifest_sha256,
        checkpoint_revision,
        maximum_context_tokens,
        estimated_memory_bytes,
    } = response
    else {
        bail!("inference worker returned no Qwen candidate-load receipt");
    };
    if protocol_version != PROTOCOL_VERSION
        || observed_challenge != challenge
        || process_id != expected_pid
        || sequence != expected_sequence
        || manifest_sha256 != package_receipt.manifest_sha256
        || checkpoint_revision != package_receipt.checkpoint_revision
        || maximum_context_tokens != QWEN35_ADMITTED_CONTEXT_TOKENS
        || estimated_memory_bytes == 0
        || estimated_memory_bytes > MAX_QWEN35_MODEL_MEMORY_BYTES
    {
        bail!(
            "inference worker Qwen candidate-load receipt failed identity or resource validation"
        );
    }
    Ok(Qwen35WorkerLoadReceipt {
        manifest_sha256,
        checkpoint_revision,
        maximum_context_tokens,
        estimated_memory_bytes,
    })
}

#[cfg(all(test, feature = "qwen35-worker-load"))]
#[test]
fn candidate_load_receipt_is_bound_to_the_package_and_resource_contract() {
    let challenge = "ef".repeat(32);
    let package_receipt = Qwen35WorkerPackageReceipt {
        manifest_sha256: "ab".repeat(32),
        key_id: "test-key".into(),
        checkpoint_revision: "candidate-revision".into(),
        artifacts: Vec::new(),
    };
    let valid = WorkerResponse::Qwen35CandidateLoaded {
        protocol_version: PROTOCOL_VERSION,
        challenge: challenge.clone(),
        process_id: 73,
        sequence: 9,
        manifest_sha256: package_receipt.manifest_sha256.clone(),
        checkpoint_revision: package_receipt.checkpoint_revision.clone(),
        maximum_context_tokens: QWEN35_ADMITTED_CONTEXT_TOKENS,
        estimated_memory_bytes: 4 * 1024 * 1024 * 1024,
    };
    let load_receipt =
        validate_qwen35_candidate_load_receipt(valid.clone(), &challenge, 73, 9, &package_receipt)
            .expect("valid candidate receipt");
    assert_eq!(load_receipt.estimated_memory_bytes, 4 * 1024 * 1024 * 1024);

    let oversized = WorkerResponse::Qwen35CandidateLoaded {
        protocol_version: PROTOCOL_VERSION,
        challenge: challenge.clone(),
        process_id: 73,
        sequence: 9,
        manifest_sha256: package_receipt.manifest_sha256.clone(),
        checkpoint_revision: package_receipt.checkpoint_revision.clone(),
        maximum_context_tokens: QWEN35_ADMITTED_CONTEXT_TOKENS,
        estimated_memory_bytes: MAX_QWEN35_MODEL_MEMORY_BYTES + 1,
    };
    assert!(
        validate_qwen35_candidate_load_receipt(oversized, &challenge, 73, 9, &package_receipt,)
            .is_err()
    );
}

async fn write_frame<W: AsyncWrite + Unpin>(writer: &mut W, bytes: &[u8]) -> anyhow::Result<()> {
    if bytes.is_empty() || bytes.len() > MAX_FRAME_BYTES {
        bail!("inference worker request exceeded the frame bound");
    }
    let length = u32::try_from(bytes.len()).context("inference worker frame length overflow")?;
    writer.write_all(&length.to_be_bytes()).await?;
    writer.write_all(bytes).await?;
    writer.flush().await?;
    Ok(())
}

async fn read_frame<R: AsyncRead + Unpin>(reader: &mut R) -> anyhow::Result<Vec<u8>> {
    let mut length = [0_u8; 4];
    reader.read_exact(&mut length).await?;
    let length = u32::from_be_bytes(length) as usize;
    if length == 0 || length > MAX_FRAME_BYTES {
        bail!("inference worker response exceeded the frame bound");
    }
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes).await?;
    Ok(bytes)
}

fn random_challenge() -> anyhow::Result<String> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes).map_err(|error| {
        anyhow::anyhow!("generate inference worker handshake challenge: {error}")
    })?;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    #[test]
    fn random_challenge_is_32_bytes_encoded_as_lowercase_hex() {
        let challenge = random_challenge().unwrap();
        assert_eq!(challenge.len(), 64);
        assert!(
            challenge
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
    }

    #[test]
    fn worker_restarts_use_bounded_exponential_backoff() {
        assert_eq!(restart_delay(0), Some(Duration::from_secs(1)));
        assert_eq!(restart_delay(1), Some(Duration::from_secs(2)));
        assert_eq!(restart_delay(2), Some(Duration::from_secs(4)));
        assert_eq!(restart_delay(MAX_WORKER_RESTARTS), None);
        assert_eq!(restart_delay(u8::MAX), None);
    }

    #[test]
    fn worker_hello_and_health_responses_are_bound_to_identity_and_sequence() {
        let challenge = "ab".repeat(32);
        let valid_hello = WorkerResponse::Hello {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 42,
            features: vec![WorkerFeature::HeartbeatV1],
            readiness: WorkerReadiness::ModelNotAdmitted,
        };
        assert!(
            validate_hello(valid_hello, &challenge, 42, WorkerHelloFeatures::default()).is_ok()
        );

        let wrong_pid = WorkerResponse::Hello {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 43,
            features: vec![WorkerFeature::HeartbeatV1],
            readiness: WorkerReadiness::ModelNotAdmitted,
        };
        assert!(validate_hello(wrong_pid, &challenge, 42, WorkerHelloFeatures::default()).is_err());

        let valid_pong = WorkerResponse::Pong {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 42,
            sequence: 7,
            readiness: WorkerReadiness::ModelNotAdmitted,
        };
        assert!(validate_pong(valid_pong, &challenge, 42, 7).is_ok());

        let package_verified_pong = WorkerResponse::Pong {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 42,
            sequence: 7,
            readiness: WorkerReadiness::PackageVerifiedModelNotAdmitted,
        };
        assert!(validate_pong(package_verified_pong, &challenge, 42, 7).is_ok());

        let replayed_pong = WorkerResponse::Pong {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 42,
            sequence: 6,
            readiness: WorkerReadiness::ModelNotAdmitted,
        };
        assert!(validate_pong(replayed_pong, &challenge, 42, 7).is_err());
    }

    #[cfg(feature = "qwen35-worker-generate")]
    #[test]
    fn generated_output_collector_checks_order_identity_digest_and_utf8_turn() {
        fn ignore_preview(_: &str) {}

        let challenge = "ab".repeat(32);
        let generation_id = "cd".repeat(32);
        let output = r#"{"goal":"","answer":"ready","actions":[]}"#;
        let mut collector = Qwen35GenerationOutputCollector::new(
            challenge.clone(),
            42,
            9,
            generation_id.clone(),
            false,
        );
        let invalid_order = WorkerResponse::Qwen35GenerationOutputChunk {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 42,
            sequence: 9,
            generation_id: generation_id.clone(),
            chunk_index: 1,
            chunk: output.to_owned(),
        };
        assert!(
            collector
                .accept(invalid_order, &mut ignore_preview)
                .is_err()
        );

        let chunk = WorkerResponse::Qwen35GenerationOutputChunk {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 42,
            sequence: 9,
            generation_id: generation_id.clone(),
            chunk_index: 0,
            chunk: output.to_owned(),
        };
        assert_eq!(collector.accept(chunk, &mut ignore_preview).unwrap(), None);
        let wrong_digest = WorkerResponse::Qwen35GenerationFinished {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 42,
            sequence: 9,
            generation_id: generation_id.clone(),
            output_chunks: 1,
            output_bytes: u32::try_from(output.len()).unwrap(),
            output_sha256: "00".repeat(32),
        };
        assert!(collector.accept(wrong_digest, &mut ignore_preview).is_err());

        let mut valid = Qwen35GenerationOutputCollector::new(
            challenge.clone(),
            42,
            9,
            generation_id.clone(),
            true,
        );
        let (observed, previews) = {
            let mut previews = Vec::new();
            let mut receive_preview = |preview: &str| previews.push(preview.to_owned());
            valid
                .accept(
                    WorkerResponse::Qwen35GenerationPreviewChunk {
                        protocol_version: PROTOCOL_VERSION,
                        challenge: challenge.clone(),
                        process_id: 42,
                        sequence: 9,
                        generation_id: generation_id.clone(),
                        preview_offset: 0,
                        chunk: "rea".into(),
                    },
                    &mut receive_preview,
                )
                .unwrap();
            valid
                .accept(
                    WorkerResponse::Qwen35GenerationOutputChunk {
                        protocol_version: PROTOCOL_VERSION,
                        challenge: challenge.clone(),
                        process_id: 42,
                        sequence: 9,
                        generation_id: generation_id.clone(),
                        chunk_index: 0,
                        chunk: output.to_owned(),
                    },
                    &mut receive_preview,
                )
                .unwrap();
            assert!(
                valid
                    .accept(
                        WorkerResponse::Qwen35GenerationPreviewChunk {
                            protocol_version: PROTOCOL_VERSION,
                            challenge: challenge.clone(),
                            process_id: 42,
                            sequence: 9,
                            generation_id: generation_id.clone(),
                            preview_offset: 3,
                            chunk: "dy".into(),
                        },
                        &mut receive_preview,
                    )
                    .is_err()
            );
            let observed = valid
                .accept(
                    WorkerResponse::Qwen35GenerationFinished {
                        protocol_version: PROTOCOL_VERSION,
                        challenge,
                        process_id: 42,
                        sequence: 9,
                        generation_id,
                        output_chunks: 1,
                        output_bytes: u32::try_from(output.len()).unwrap(),
                        output_sha256: sha256_hex(output.as_bytes()),
                    },
                    &mut receive_preview,
                )
                .unwrap()
                .unwrap();
            (observed, previews)
        };
        assert_eq!(observed, output);
        assert_eq!(previews, ["rea"]);
        crate::model::validate_turn_json(&observed).unwrap();
    }

    #[cfg(feature = "qwen35-worker-generate")]
    #[test]
    fn generated_output_must_extend_every_previously_streamed_answer_preview() {
        let challenge = "ab".repeat(32);
        let generation_id = "cd".repeat(32);
        let mut collector = Qwen35GenerationOutputCollector::new(
            challenge.clone(),
            42,
            9,
            generation_id.clone(),
            true,
        );
        fn ignore_preview(_: &str) {}
        collector
            .accept(
                WorkerResponse::Qwen35GenerationPreviewChunk {
                    protocol_version: PROTOCOL_VERSION,
                    challenge: challenge.clone(),
                    process_id: 42,
                    sequence: 9,
                    generation_id: generation_id.clone(),
                    preview_offset: 0,
                    chunk: "ready".into(),
                },
                &mut ignore_preview,
            )
            .unwrap();
        let output = r#"{"goal":"","answer":"no","actions":[]}"#;
        collector
            .accept(
                WorkerResponse::Qwen35GenerationOutputChunk {
                    protocol_version: PROTOCOL_VERSION,
                    challenge: challenge.clone(),
                    process_id: 42,
                    sequence: 9,
                    generation_id: generation_id.clone(),
                    chunk_index: 0,
                    chunk: output.to_owned(),
                },
                &mut ignore_preview,
            )
            .unwrap();
        assert!(
            collector
                .accept(
                    WorkerResponse::Qwen35GenerationFinished {
                        protocol_version: PROTOCOL_VERSION,
                        challenge,
                        process_id: 42,
                        sequence: 9,
                        generation_id,
                        output_chunks: 1,
                        output_bytes: u32::try_from(output.len()).unwrap(),
                        output_sha256: sha256_hex(output.as_bytes()),
                    },
                    &mut ignore_preview,
                )
                .is_err()
        );
    }

    #[cfg(feature = "qwen35-worker-generate")]
    #[test]
    fn worker_prompt_requires_the_same_bounded_schema_embedded_in_policy() {
        let schema = crate::model::draft_schema();
        let schema_text = serde_json::to_string(&schema).unwrap();
        let system_message = format!("Sage policy\n\nClosed output schema:\n{schema_text}");
        assert!(schema_text.len() <= MAX_QWEN35_SCHEMA_BYTES);
        assert_eq!(
            checked_qwen35_output_schema(&system_message, &schema).unwrap(),
            schema_text
        );
        let different_schema = serde_json::json!({"type":"object"});
        assert!(checked_qwen35_output_schema(&system_message, &different_schema).is_err());
        assert!(checked_qwen35_output_schema("policy", &schema).is_err());
    }

    #[test]
    fn package_verification_handshake_requires_the_exact_negotiated_feature_set() {
        let challenge = "cd".repeat(32);
        let mut expected_features = vec![
            WorkerFeature::HeartbeatV1,
            WorkerFeature::ArtifactReadV1,
            WorkerFeature::Qwen35PackageVerifyV1,
        ];
        if cfg!(feature = "qwen35-worker-load") {
            expected_features.push(WorkerFeature::Qwen35CandidateLoadV1);
        }
        if cfg!(feature = "qwen35-worker-generate") {
            expected_features.push(WorkerFeature::Qwen35GenerateV1);
        }
        let response = WorkerResponse::Hello {
            protocol_version: PROTOCOL_VERSION,
            challenge: challenge.clone(),
            process_id: 91,
            features: expected_features,
            readiness: WorkerReadiness::ModelNotAdmitted,
        };
        assert!(
            validate_hello(
                response.clone(),
                &challenge,
                91,
                WorkerHelloFeatures {
                    has_artifacts: true,
                    qwen35_package_verification: true,
                    qwen35_candidate_load: cfg!(feature = "qwen35-worker-load"),
                    qwen35_generation: cfg!(feature = "qwen35-worker-generate"),
                    qwen35_generation_previews: cfg!(feature = "qwen35-worker-generate"),
                },
            )
            .is_ok()
        );
        assert!(
            validate_hello(
                response,
                &challenge,
                91,
                WorkerHelloFeatures {
                    has_artifacts: true,
                    ..WorkerHelloFeatures::default()
                },
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn handshake_rejects_oversized_response_before_allocating_payload() {
        let (mut writer, mut reader) = duplex(8);
        writer
            .write_all(&((MAX_FRAME_BYTES + 1) as u32).to_be_bytes())
            .await
            .unwrap();
        assert!(read_frame(&mut reader).await.is_err());
    }

    #[tokio::test]
    #[ignore = "requires SAGE_TEST_INFERENCE_WORKER_EXECUTABLE pointing to a built worker"]
    async fn launches_worker_and_verifies_inherited_readonly_artifact() {
        use sha2::{Digest, Sha256};
        use std::io::Write;

        let path = std::env::var_os("SAGE_TEST_INFERENCE_WORKER_EXECUTABLE")
            .expect("test worker executable path");
        let payload = b"checkpoint artifact bytes passed through one read-only descriptor";
        let mut fixture = tempfile::NamedTempFile::new().unwrap();
        fixture.write_all(payload).unwrap();
        let artifact = File::open(fixture.path()).unwrap();
        let digest = format!("{:x}", Sha256::digest(payload));

        let mut worker = launch_with_readonly_files(Path::new(&path), &[&artifact])
            .await
            .unwrap();
        worker
            .verify_inherited_artifact(0, payload.len() as u64, &digest)
            .await
            .expect("worker verifies the exact inherited bytes");
        worker.heartbeat().await.expect("negotiated heartbeat");
        drop(worker.stdout);
        drop(worker.stdin);
        let status = timeout(Duration::from_secs(2), worker.child.wait())
            .await
            .expect("worker exits after Core closes its request pipe")
            .unwrap();
        assert!(status.success());
    }

    #[cfg(feature = "qwen35-worker-verification")]
    #[test]
    fn package_receipts_are_bound_to_the_signed_manifest_and_exact_file_handles() {
        use sage_inference_protocol::VerifiedPackageArtifactReceipt;

        let (fixtures, files, manifest, package) = signed_package_fixture();
        let receipts = QWEN35_PACKAGE_ARTIFACTS
            .into_iter()
            .zip(&files)
            .map(|(name, file)| VerifiedPackageArtifactReceipt {
                name: name.into(),
                bytes: file.metadata().unwrap().len(),
                sha256: manifest.artifacts[name].clone(),
            })
            .collect::<Vec<_>>();
        let response = |receipts| WorkerResponse::Qwen35PackageVerified {
            protocol_version: PROTOCOL_VERSION,
            challenge: "ef".repeat(32),
            process_id: 73,
            sequence: 8,
            manifest_sha256: package.manifest_sha256().into(),
            key_id: package.key_id().into(),
            checkpoint_revision: package.checkpoint_revision().into(),
            artifacts: receipts,
        };
        let validated = validate_qwen35_package_receipt(
            response(receipts.clone()),
            &"ef".repeat(32),
            73,
            8,
            &files.iter().collect::<Vec<_>>(),
            &manifest,
            &package,
        )
        .unwrap();
        assert_eq!(validated.artifacts, receipts);

        let mut wrong_digest = receipts.clone();
        wrong_digest[2].sha256 = "00".repeat(32);
        assert!(
            validate_qwen35_package_receipt(
                response(wrong_digest),
                &"ef".repeat(32),
                73,
                8,
                &files.iter().collect::<Vec<_>>(),
                &manifest,
                &package,
            )
            .is_err()
        );

        let mut wrong_order = receipts.clone();
        wrong_order.swap(0, 1);
        assert!(
            validate_qwen35_package_receipt(
                response(wrong_order),
                &"ef".repeat(32),
                73,
                8,
                &files.iter().collect::<Vec<_>>(),
                &manifest,
                &package,
            )
            .is_err()
        );
        drop(fixtures);
    }

    #[cfg(feature = "qwen35-worker-verification")]
    #[tokio::test]
    #[ignore = "requires a worker and Core binary built with the sage-test-2026 trust key"]
    async fn core_verifies_a_signed_package_in_the_worker_and_retains_its_handles() {
        use ed25519_dalek::SigningKey;

        assert!(std::env::var_os("SAGE_INFERENCE_WORKER_EXECUTABLE").is_some());
        const TEST_SIGNING_SEED: [u8; 32] = [
            0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec,
            0x2c, 0xcc, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03,
            0x1c, 0xae, 0x7f, 0x60,
        ];
        let expected_key = SigningKey::from_bytes(&TEST_SIGNING_SEED).verifying_key();
        let trusted_keys = application_trusted_qwen35_package_keys().unwrap();
        assert_eq!(
            trusted_keys
                .get("sage-test-2026")
                .expect("build the test root into both Core and worker")
                .as_bytes(),
            expected_key.as_bytes()
        );
        let (fixtures, files, manifest, package) = signed_package_fixture();
        let manifest_json = serde_json::to_string(&manifest).unwrap();
        let mut worker =
            launch_verified_qwen35_package(&manifest_json, &files.iter().collect::<Vec<_>>())
                .await
                .unwrap();
        assert_eq!(worker.receipt().key_id, "sage-test-2026");
        assert_eq!(worker.receipt().manifest_sha256, package.manifest_sha256());
        assert_eq!(
            worker.receipt().artifacts.len(),
            QWEN35_PACKAGE_ARTIFACT_COUNT
        );
        worker.health_check().await.unwrap();

        let VerifiedQwen35PackageWorker { mut worker, .. } = worker;
        drop(worker.stdout);
        drop(worker.stdin);
        let status = timeout(Duration::from_secs(2), worker.child.wait())
            .await
            .expect("verified worker exits after Core closes its request pipe")
            .unwrap();
        assert!(status.success());
        drop(fixtures);
    }

    #[cfg(feature = "qwen35-worker-verification")]
    fn signed_package_fixture() -> (
        Vec<tempfile::NamedTempFile>,
        Vec<File>,
        Qwen35PackageManifest,
        VerifiedQwen35Package,
    ) {
        use ed25519_dalek::{Signer, SigningKey};
        use sha2::{Digest, Sha256};
        use std::io::Write;

        const TEST_SIGNING_SEED: [u8; 32] = [
            0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec,
            0x2c, 0xcc, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03,
            0x1c, 0xae, 0x7f, 0x60,
        ];
        let signing_key = SigningKey::from_bytes(&TEST_SIGNING_SEED);
        let mut fixtures = Vec::with_capacity(QWEN35_PACKAGE_ARTIFACT_COUNT);
        let mut files = Vec::with_capacity(QWEN35_PACKAGE_ARTIFACT_COUNT);
        let mut digests = std::collections::BTreeMap::new();
        for (index, name) in QWEN35_PACKAGE_ARTIFACTS.into_iter().enumerate() {
            let payload = format!("Sage package fixture {index}: {name}");
            let mut fixture = tempfile::NamedTempFile::new().unwrap();
            fixture.write_all(payload.as_bytes()).unwrap();
            digests.insert(
                name.to_owned(),
                format!("{:x}", Sha256::digest(payload.as_bytes())),
            );
            files.push(File::open(fixture.path()).unwrap());
            fixtures.push(fixture);
        }
        let mut manifest = Qwen35PackageManifest {
            schema_version: 2,
            checkpoint_revision: sage_model_package::QWEN35_CHECKPOINT_REVISION.into(),
            key_id: "sage-test-2026".into(),
            artifacts: digests,
            signature_hex: String::new(),
        };
        let signature = signing_key.sign(&manifest.signing_bytes().unwrap());
        manifest.signature_hex = signature
            .to_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let trust = std::collections::BTreeMap::from([(
            manifest.key_id.clone(),
            signing_key.verifying_key(),
        )]);
        let package = manifest.verify_signature(&trust).unwrap();
        (fixtures, files, manifest, package)
    }
}
