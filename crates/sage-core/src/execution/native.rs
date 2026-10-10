use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{Duration, Utc};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::capability::{CapabilityGrant, CapabilityResource};
use crate::compiler::{CompiledAction, ImplementationCandidate};
use crate::domain::{Action, ActionProposal, ExecutionDomain};
use crate::error::{CoreError, CoreResult};

use super::{ExecutionReceipt, Executor, RollbackOperation, RollbackPlan};

#[async_trait]
pub trait PlatformController: Send + Sync {
    async fn execute_native(
        &self,
        proposal: &ActionProposal,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt>;
}

#[derive(Debug, Default)]
pub struct UnsupportedPlatformController;

#[async_trait]
impl PlatformController for UnsupportedPlatformController {
    async fn execute_native(
        &self,
        proposal: &ActionProposal,
        _capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        Err(CoreError::ExecutorUnavailable(format!(
            "the platform adapter does not implement {}",
            proposal.action.kind()
        )))
    }
}

pub struct NativeExecutor {
    recovery_root: PathBuf,
    platform: Arc<dyn PlatformController>,
    files: Arc<super::files::FileBroker>,
    store: crate::storage::LocalStore,
}

impl std::fmt::Debug for NativeExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeExecutor")
            .field("recovery_root", &self.recovery_root)
            .finish_non_exhaustive()
    }
}

impl NativeExecutor {
    pub fn new(
        recovery_root: PathBuf,
        platform: Arc<dyn PlatformController>,
        files: Arc<super::files::FileBroker>,
        store: crate::storage::LocalStore,
    ) -> Self {
        Self {
            recovery_root,
            platform,
            files,
            store,
        }
    }

    async fn execute_file(
        &self,
        compiled: &CompiledAction,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        let proposal = &compiled.proposal;
        match &proposal.action {
            Action::FetchPublic { url, max_bytes } => {
                if !matches!(&capability.resource,CapabilityResource::NetworkRoute{url:approved} if approved==url)
                {
                    return Err(CoreError::CapabilityRejected(
                        "Public fetch grant targets another URL".into(),
                    ));
                }
                let document = crate::network::fetch_public(url, *max_bytes).await?;
                Ok(ExecutionReceipt {
                    executor: "network-broker".into(),
                    summary: format!("Read {} bytes from {}", document.text.len(), document.url),
                    transient_data: serde_json::to_value(document)?,
                    rollback: None,
                })
            }
            Action::ReadFile { path, max_bytes } => {
                require_exact_file(capability, path)?;
                let pinned = self.files.take(proposal, capability)?;
                let maximum = *max_bytes;
                let expiry = capability.expires_at;
                let bytes = super::io::bounded_read(move |cancelled| {
                    if expiry <= Utc::now() {
                        return Err(CoreError::CapabilityRejected("Read grant expired".into()));
                    }
                    let bytes = pinned.read(maximum)?;
                    if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                        return Err(CoreError::Cancelled);
                    }
                    Ok(bytes)
                })
                .await?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: format!("read {} bytes from {}", bytes.len(), path.display()),
                    transient_data: json!({ "bytes_base64": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes) }),
                    rollback: None,
                })
            }
            Action::ListDirectory {
                path,
                page_size,
                cursor,
            } => {
                require_exact_file(capability, path)?;
                let pinned = self.files.take(proposal, capability)?;
                let page = super::directory::inspect(
                    Some(pinned),
                    path.clone(),
                    *page_size,
                    cursor.clone(),
                    Some(capability.expires_at),
                )
                .await?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: page.summary(),
                    transient_data: serde_json::to_value(page)?,
                    rollback: None,
                })
            }
            Action::WriteFile {
                path,
                content,
                overwrite,
            } => {
                require_exact_file(capability, path)?;
                let prepared = self.files.take(proposal, capability)?;
                let expected_sha256 = format!("{:x}", Sha256::digest(content.as_bytes()));
                let operation = if prepared.exists() {
                    if !overwrite {
                        return Err(CoreError::ExecutionFailed(
                            "Overwrite was not authorized".into(),
                        ));
                    }
                    let backup = prepared.read(16 * 1024 * 1024)?;
                    let artifact_id = self.store.save_artifact(proposal.task_id, &backup)?;
                    RollbackOperation::RestoreArtifact {
                        artifact_id,
                        destination: path.to_string_lossy().into_owned(),
                        expected_sha256,
                    }
                } else {
                    RollbackOperation::RemoveCreatedFile {
                        path: path.to_string_lossy().into_owned(),
                        expected_sha256,
                    }
                };
                let rollback = Some(RollbackPlan {
                    action_id: proposal.id,
                    operations: vec![operation],
                    expires_at: Utc::now() + Duration::hours(24),
                });
                self.store
                    .save_rollback(proposal.task_id, rollback.as_ref().unwrap())?;
                prepared.write(content.as_bytes(), *overwrite)?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: format!("wrote {} bytes to {}", content.len(), path.display()),
                    transient_data: json!({ "bytes_written": content.len() }),
                    rollback,
                })
            }
            Action::CreateFolder { path } => {
                require_exact_file(capability, path)?;
                let prepared = self.files.take(proposal, capability)?;
                prepared.create_folder()?;
                let identity = prepared.current_identity()?;
                Ok(ExecutionReceipt {
                    executor: self.name().into(),
                    summary: format!("created folder {}", path.display()),
                    transient_data: json!({}),
                    rollback: Some(RollbackPlan {
                        action_id: proposal.id,
                        operations: vec![RollbackOperation::RemoveCreatedFolder {
                            identity,
                            path: path.to_string_lossy().into_owned(),
                        }],
                        expires_at: Utc::now() + Duration::hours(24),
                    }),
                })
            }
            _ => self.platform.execute_native(proposal, capability).await,
        }
    }

    async fn execute_streamed_file(
        &self,
        compiled: &CompiledAction,
        capability: &CapabilityGrant,
        streams: &mut crate::procedure_stream::ProcedureNodeStreams,
    ) -> CoreResult<ExecutionReceipt> {
        let proposal = &compiled.proposal;
        let Action::WriteFile {
            path,
            content,
            overwrite,
        } = &proposal.action
        else {
            return Err(CoreError::ExecutorUnavailable(
                "The native stream executor currently accepts only file-content writes".into(),
            ));
        };
        if !content.is_empty()
            || proposal
                .metadata
                .get("procedure_stream_node")
                .map(String::as_str)
                != Some("true")
            || proposal
                .metadata
                .get("procedure_stream_input")
                .map(String::as_str)
                != Some("content")
        {
            return Err(CoreError::InvalidAction(
                "Streamed writes require an empty scalar body and the declared content input"
                    .into(),
            ));
        }
        let channel_id = proposal
            .metadata
            .get("procedure_stream_channel_id")
            .ok_or_else(|| {
                CoreError::VerificationFailed("Streamed write has no bound channel identity".into())
            })?;
        let producer_node = proposal
            .metadata
            .get("procedure_stream_producer_node")
            .ok_or_else(|| {
                CoreError::VerificationFailed(
                    "Streamed write has no bound producer identity".into(),
                )
            })?;
        let approved_maximum_bytes = proposal
            .metadata
            .get("procedure_stream_max_bytes")
            .and_then(|bytes| bytes.parse::<u64>().ok())
            .filter(|bytes| (1..=super::files::MAX_BYTES).contains(bytes))
            .ok_or_else(|| {
                CoreError::VerificationFailed("Streamed write has no approved byte limit".into())
            })?;
        match &proposal.expected_outcome {
            crate::domain::ExpectedOutcome::FileMatchesStream {
                path: expected_path,
                channel_id: expected_channel,
                producer_node: expected_producer,
                maximum_bytes: expected_maximum,
            } if expected_path == path
                && expected_channel == channel_id
                && expected_producer == producer_node
                && *expected_maximum == approved_maximum_bytes => {}
            _ => {
                return Err(CoreError::VerificationFailed(
                    "Streamed write outcome does not bind the exact file and channel".into(),
                ));
            }
        }
        require_exact_file(capability, path)?;
        if !streams.outputs_mut().is_empty() || streams.inputs_mut().len() != 1 {
            return Err(CoreError::InvalidAction(
                "A streamed file write requires exactly one input and no output streams".into(),
            ));
        }
        let input = streams
            .inputs_mut()
            .get_mut("content")
            .ok_or_else(|| CoreError::InvalidAction("File content stream is missing".into()))?;
        if input.channel_id() != channel_id
            || input.producer_node() != producer_node
            || input.input_port().name != "content"
            || input.input_port().value_type != crate::world_model::PortType::Bytes
        {
            return Err(CoreError::VerificationFailed(
                "File content stream differs from the exact declared channel and port".into(),
            ));
        }
        let maximum_bytes = input.maximum_total_bytes().min(super::files::MAX_BYTES);
        if maximum_bytes != approved_maximum_bytes {
            return Err(CoreError::VerificationFailed(
                "Stream byte limit changed after approval".into(),
            ));
        }
        let prepared = self.files.take(proposal, capability)?;
        if prepared.exists() && !overwrite {
            return Err(CoreError::ExecutionFailed(
                "Destination already exists".into(),
            ));
        }

        let (writer_sender, writer_receiver) = tokio::sync::mpsc::channel(1);
        let store = self.store.clone();
        let task_id = proposal.task_id;
        let action_id = proposal.id;
        let destination = path.to_string_lossy().into_owned();
        let overwrite = *overwrite;
        let write = super::io::bounded_write(move |cancelled| {
            let artifact_id = if prepared.exists() {
                let previous = prepared.read(super::files::MAX_BYTES)?;
                Some(store.save_artifact(task_id, &previous)?)
            } else {
                None
            };
            prepared.write_stream(
                writer_receiver,
                maximum_bytes,
                overwrite,
                cancelled,
                |stream_sha256, _bytes_written| {
                    let operation = match artifact_id {
                        Some(artifact_id) => RollbackOperation::RestoreArtifact {
                            artifact_id,
                            destination: destination.clone(),
                            expected_sha256: stream_sha256.to_owned(),
                        },
                        None => RollbackOperation::RemoveCreatedFile {
                            path: destination.clone(),
                            expected_sha256: stream_sha256.to_owned(),
                        },
                    };
                    let rollback = RollbackPlan {
                        action_id,
                        operations: vec![operation],
                        expires_at: Utc::now() + Duration::hours(24),
                    };
                    store.save_rollback(task_id, &rollback)?;
                    Ok(rollback)
                },
            )
        });
        tokio::pin!(write);
        loop {
            tokio::select! {
                biased;
                result = &mut write => {
                    result?;
                    return Err(CoreError::VerificationFailed(
                        "Streamed file writer completed before the input stream ended".into(),
                    ));
                }
                item = input.recv() => match item {
                    Ok(Some(item)) => {
                        writer_sender
                            .send(super::files::StreamWriteInput::Chunk(item.into_bytes()))
                            .await
                            .map_err(|_| CoreError::ExecutionFailed(
                                "Streamed file writer closed before receiving all input".into(),
                            ))?;
                    }
                    Ok(None) => {
                        writer_sender
                            .send(super::files::StreamWriteInput::Finish)
                            .await
                            .map_err(|_| CoreError::ExecutionFailed(
                                "Streamed file writer closed before finalization".into(),
                            ))?;
                        break;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        drop(writer_sender);
        let (stream_sha256, bytes_written, rollback) = write.await?;
        Ok(ExecutionReceipt {
            executor: self.name().into(),
            summary: format!("wrote {bytes_written} streamed bytes to {}", path.display()),
            transient_data: json!({
                "stream_channel_id": channel_id,
                "stream_producer_node": producer_node,
                "stream_sha256": stream_sha256,
                "bytes_written": bytes_written,
            }),
            rollback: Some(rollback),
        })
    }

    async fn execute_streamed_read(
        &self,
        compiled: &CompiledAction,
        capability: &CapabilityGrant,
        streams: &mut crate::procedure_stream::ProcedureNodeStreams,
    ) -> CoreResult<ExecutionReceipt> {
        let proposal = &compiled.proposal;
        let Action::ReadFile { path, max_bytes } = &proposal.action else {
            return Err(CoreError::ExecutorUnavailable(
                "The native stream producer currently accepts only file reads".into(),
            ));
        };
        if proposal
            .metadata
            .get("procedure_stream_node")
            .map(String::as_str)
            != Some("true")
        {
            return Err(CoreError::InvalidAction(
                "Streamed file reads require a procedure-bound stream endpoint".into(),
            ));
        }
        let crate::domain::ExpectedOutcome::FileReadMatchesStream {
            path: expected_path,
            channel_id,
            producer_node,
            output_port,
            consumer_node,
            maximum_bytes,
        } = &proposal.expected_outcome
        else {
            return Err(CoreError::VerificationFailed(
                "Streamed file read has no exact, independently verified output contract".into(),
            ));
        };
        if expected_path != path
            || producer_node
                != proposal
                    .metadata
                    .get("procedure_node_id")
                    .map(String::as_str)
                    .unwrap_or_default()
            || *maximum_bytes > *max_bytes
            || *maximum_bytes > super::files::MAX_BYTES
        {
            return Err(CoreError::VerificationFailed(
                "Streamed file read differs from its approved path or byte bound".into(),
            ));
        }
        require_exact_file(capability, path)?;
        if !streams.inputs_mut().is_empty() || streams.outputs_mut().len() != 1 {
            return Err(CoreError::InvalidAction(
                "A streamed file read requires exactly one output and no input streams".into(),
            ));
        }
        let output = streams
            .outputs_mut()
            .get_mut(output_port)
            .ok_or_else(|| CoreError::InvalidAction("File output stream is missing".into()))?;
        if output.channel_id() != channel_id
            || output.producer_node() != producer_node
            || output.consumer_node() != consumer_node
            || output.output_port().name != *output_port
            || output.output_port().value_type != crate::world_model::PortType::Bytes
            || output.maximum_total_bytes() != *maximum_bytes
        {
            return Err(CoreError::VerificationFailed(
                "File output stream differs from its exact declared channel and port".into(),
            ));
        }

        let prepared = self.files.take(proposal, capability)?;
        let stream_limit = *maximum_bytes;
        let item_limit = output.maximum_item_bytes();
        let (read_sender, mut read_receiver) = tokio::sync::mpsc::channel(1);
        let read = super::io::bounded_stream_read(move |cancelled| {
            prepared.read_stream(stream_limit, item_limit, read_sender, cancelled)
        });
        tokio::pin!(read);
        let mut read_result = None;
        loop {
            tokio::select! {
                result = &mut read, if read_result.is_none() => match result {
                    Ok(result) => read_result = Some(result),
                    Err(error) => return Err(error),
                },
                item = read_receiver.recv() => match item {
                    Some(super::files::StreamReadOutput::Chunk(mut bytes)) => {
                        output.send(std::mem::take(&mut *bytes)).await?;
                    }
                    Some(super::files::StreamReadOutput::Finish) => {
                        if read_result.is_none() {
                            read_result = Some(read.await?);
                        }
                        output.finish().await?;
                        break;
                    }
                    None => {
                        if read_result.is_none() {
                            let _ = read.await?;
                        }
                        return Err(CoreError::VerificationFailed(
                            "File reader closed without its explicit stream terminator".into(),
                        ));
                    }
                }
            }
        }
        let (stream_sha256, bytes_read) = read_result.ok_or_else(|| {
            CoreError::VerificationFailed("Streamed file read has no settled receipt".into())
        })?;
        Ok(ExecutionReceipt {
            executor: self.name().into(),
            summary: format!("streamed {bytes_read} bytes from {}", path.display()),
            transient_data: json!({
                "stream_channel_id": channel_id,
                "stream_producer_node": producer_node,
                "stream_output_port": output_port,
                "stream_consumer_node": consumer_node,
                "stream_sha256": stream_sha256,
                "bytes_read": bytes_read,
            }),
            rollback: None,
        })
    }
}

#[async_trait]
impl Executor for NativeExecutor {
    fn name(&self) -> &'static str {
        "native-os-executor"
    }

    fn domain(&self) -> ExecutionDomain {
        ExecutionDomain::Native
    }

    fn supports_procedure_streams(&self) -> bool {
        true
    }

    async fn execute(
        &self,
        action: &CompiledAction,
        _implementation: &ImplementationCandidate,
        capability: &CapabilityGrant,
    ) -> CoreResult<ExecutionReceipt> {
        self.execute_file(action, capability).await
    }

    async fn execute_with_streams(
        &self,
        action: &CompiledAction,
        _implementation: &ImplementationCandidate,
        capability: &CapabilityGrant,
        streams: Option<&mut crate::procedure_stream::ProcedureNodeStreams>,
    ) -> CoreResult<ExecutionReceipt> {
        let Some(streams) = streams.filter(|streams| !streams.is_empty()) else {
            return self.execute_file(action, capability).await;
        };
        match &action.proposal.action {
            Action::ReadFile { .. } => {
                self.execute_streamed_read(action, capability, streams)
                    .await
            }
            Action::WriteFile { .. } => {
                self.execute_streamed_file(action, capability, streams)
                    .await
            }
            _ => Err(CoreError::ExecutorUnavailable(
                "The native stream executor currently accepts only bounded file reads and writes"
                    .into(),
            )),
        }
    }
}

fn require_exact_file(capability: &CapabilityGrant, path: &Path) -> CoreResult<()> {
    match &capability.resource {
        CapabilityResource::File { canonical_path } if Path::new(canonical_path) == path => Ok(()),
        _ => Err(CoreError::CapabilityRejected(
            "filesystem capability does not match the exact action path".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        sync::Arc,
    };

    use super::*;
    use crate::{
        agency::{
            CompletionCondition, ProcedureIr, ProcedureNode, ProcedureNodeKind, StreamBackpressure,
            StreamChannel, ValueBinding,
        },
        capability::CapabilityBroker,
        contracts::{Effect, Sensitivity},
        domain::{ActionProposal, ExpectedOutcome, Provenance},
        execution::ExecutionBroker,
        observation::{DeterministicObserver, Observer},
        verification::Verifier,
        world_model::{CapabilityDescriptor, DataPort, PortType, Preconditions},
    };
    use uuid::Uuid;

    fn port(name: &str, max_bytes: u64) -> DataPort {
        DataPort {
            name: name.into(),
            value_type: PortType::Bytes,
            max_bytes,
            privacy: Sensitivity::Private,
        }
    }

    fn capability(
        id: &str,
        system_id: Uuid,
        fingerprint: &str,
        inputs: Vec<DataPort>,
        outputs: Vec<DataPort>,
    ) -> CapabilityDescriptor {
        let evidence_id = Uuid::new_v4();
        CapabilityDescriptor {
            schema_version: 1,
            id: id.into(),
            system_id,
            system_fingerprint: fingerprint.into(),
            interface_control_id: None,
            interface_probe_kind: None,
            label: id.into(),
            input_ports: inputs,
            output_ports: outputs,
            preconditions: Preconditions {
                observed_state_fact_ids: vec![evidence_id],
                description: "Exact bounded stream operation".into(),
            },
            effects: BTreeSet::from([Effect::Read]),
            verification: "Verify output".into(),
            restoration: None,
            cancellation: "Stop and settle".into(),
            executor_id: None,
            evidence_ids: vec![evidence_id],
            updated_at: Utc::now(),
        }
    }

    fn procedure() -> (ProcedureIr, Vec<CapabilityDescriptor>) {
        let producer_system = Uuid::new_v4();
        let consumer_system = Uuid::new_v4();
        let producer_fingerprint = "a".repeat(64);
        let consumer_fingerprint = "b".repeat(64);
        let output = port("bytes", 64);
        let input = port("content", 64);
        let producer = capability(
            "fixture.stream_source",
            producer_system,
            &producer_fingerprint,
            Vec::new(),
            vec![output.clone()],
        );
        let consumer = capability(
            "native.write_file",
            consumer_system,
            &consumer_fingerprint,
            vec![input],
            Vec::new(),
        );
        let channel = StreamChannel {
            id: "file-content".into(),
            producer_node: "source".into(),
            producer_output: "bytes".into(),
            consumer_node: "sink".into(),
            consumer_input: "content".into(),
            capacity_items: 3,
            maximum_item_bytes: 8,
            backpressure: StreamBackpressure::BlockProducer,
        };
        let procedure = ProcedureIr {
            schema_version: 2,
            id: "native-stream-file-write".into(),
            nodes: vec![
                ProcedureNode {
                    controller_binding: None,
                    id: "source".into(),
                    depends_on: BTreeSet::new(),
                    outputs: BTreeMap::from([("bytes".into(), output)]),
                    kind: ProcedureNodeKind::CapabilityCall {
                        capability_id: producer.id.clone(),
                        system_id: producer_system,
                        system_fingerprint: producer_fingerprint,
                        input_bindings: BTreeMap::new(),
                    },
                },
                ProcedureNode {
                    controller_binding: None,
                    id: "sink".into(),
                    depends_on: BTreeSet::new(),
                    outputs: BTreeMap::new(),
                    kind: ProcedureNodeKind::CapabilityCall {
                        capability_id: consumer.id.clone(),
                        system_id: consumer_system,
                        system_fingerprint: consumer_fingerprint,
                        input_bindings: BTreeMap::from([(
                            "content".into(),
                            ValueBinding::Stream {
                                channel_id: channel.id.clone(),
                            },
                        )]),
                    },
                },
            ],
            streams: vec![channel],
            completion: vec![CompletionCondition::AllNodesSucceeded],
        };
        (procedure, vec![producer, consumer])
    }

    #[tokio::test]
    async fn native_stream_write_uses_one_grant_and_freshly_verifies_published_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let path = root.join("streamed.bin");
        let task_id = Uuid::new_v4();
        let channel_id = "file-content";
        let mut proposal = ActionProposal {
            id: Uuid::new_v4(),
            task_id,
            action: Action::WriteFile {
                path: path.clone(),
                content: String::new(),
                overwrite: false,
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: path.to_string_lossy().into_owned(),
            provenance: Provenance::user(),
            metadata: BTreeMap::from([
                ("procedure_id".into(), "native-stream-file-write".into()),
                ("procedure_node_id".into(), "sink".into()),
                ("procedure_stream_node".into(), "true".into()),
                ("procedure_stream_input".into(), "content".into()),
                ("procedure_stream_channel_id".into(), channel_id.into()),
                ("procedure_stream_producer_node".into(), "source".into()),
                ("procedure_stream_max_bytes".into(), "64".into()),
            ]),
        };
        crate::verification::bind_required_outcome(&mut proposal).unwrap();
        let preview = crate::policy::prepared_preview(&proposal).unwrap();
        assert!(preview.contains("procedure node source"));
        assert!(preview.contains("channel file-content"));
        assert!(preview.contains("Maximum stream size: 64 bytes"));
        assert!(
            matches!(proposal.expected_outcome, ExpectedOutcome::FileMatchesStream { ref channel_id, ref producer_node, maximum_bytes, .. } if channel_id == "file-content" && producer_node == "source" && maximum_bytes == 64)
        );

        let files = Arc::new(crate::execution::files::FileBroker::default());
        files.prepare(&mut proposal).await.unwrap();
        let store = crate::storage::LocalStore::deferred(&root.join("deferred.db")).unwrap();
        let capabilities = CapabilityBroker::default();
        let grant = capabilities
            .issue_unprepared_for_test(&proposal, ExecutionDomain::Native)
            .await
            .unwrap();
        let implementation = ImplementationCandidate {
            tier: crate::compiler::InteractionTier::StructuredIntegration,
            executor: ExecutionDomain::Native,
            operation: "write verified byte stream to exact file".into(),
        };
        let compiled = CompiledAction {
            proposal: proposal.clone(),
            candidates: vec![implementation.clone()],
        };
        let native = NativeExecutor::new(
            root.clone(),
            Arc::new(UnsupportedPlatformController),
            files,
            store,
        );
        let mut broker = ExecutionBroker::new(capabilities);
        broker.register(Arc::new(native));

        let (procedure, descriptors) = procedure();
        let mut pool =
            crate::procedure_stream::ProcedureStreamPool::new(&procedure, &descriptors, task_id)
                .unwrap();
        let (_cancel, cancellation) = crate::procedure_stream::ProcedureStreamCancellation::new();
        let mut endpoints = pool.open_all(cancellation).unwrap();
        let mut producer = endpoints.take_node("source").unwrap();
        let consumer = endpoints.take_node("sink").unwrap();
        let sender = producer.outputs_mut().get_mut("bytes").unwrap();
        sender.send(b"Sage".to_vec()).await.unwrap();
        sender.send(b" data".to_vec()).await.unwrap();
        sender.finish().await.unwrap();

        let receipt = broker
            .execute_with_streams(&compiled, &implementation, &grant, Some(consumer))
            .await
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"Sage data");
        assert_eq!(receipt.transient_data["bytes_written"], 9);
        assert_eq!(receipt.transient_data["stream_channel_id"], channel_id);
        assert!(
            broker
                .execute_with_streams(&compiled, &implementation, &grant, None)
                .await
                .is_err()
        );

        let observation = DeterministicObserver
            .observe(&proposal, &receipt)
            .await
            .unwrap();
        Verifier
            .verify(&proposal.expected_outcome, &observation)
            .unwrap();
        std::fs::write(&path, b"changed after the stream").unwrap();
        assert!(
            DeterministicObserver
                .observe(&proposal, &receipt)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn native_stream_read_forwards_bounded_bytes_and_freshly_verifies_source() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let path = root.join("stream-source.bin");
        std::fs::write(&path, b"bounded source bytes").unwrap();
        let task_id = Uuid::new_v4();
        let channel_id = "file-content";
        let mut proposal = ActionProposal {
            id: Uuid::new_v4(),
            task_id,
            action: Action::ReadFile {
                path: path.clone(),
                max_bytes: 64,
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: path.to_string_lossy().into_owned(),
            provenance: Provenance::user(),
            metadata: BTreeMap::from([
                ("procedure_id".into(), "native-stream-file-write".into()),
                ("procedure_node_id".into(), "source".into()),
                ("procedure_stream_node".into(), "true".into()),
                ("procedure_stream_output".into(), "bytes".into()),
                ("procedure_stream_channel_id".into(), channel_id.into()),
                ("procedure_stream_consumer_node".into(), "sink".into()),
                ("procedure_stream_max_bytes".into(), "64".into()),
            ]),
        };
        crate::verification::bind_required_outcome(&mut proposal).unwrap();
        let preview = crate::policy::prepared_preview(&proposal).unwrap();
        assert!(preview.contains("stream its bytes to procedure node sink"));
        assert!(preview.contains("Output port: bytes"));
        assert!(matches!(
            proposal.expected_outcome,
            ExpectedOutcome::FileReadMatchesStream {
                ref channel_id,
                ref producer_node,
                ref output_port,
                ref consumer_node,
                maximum_bytes,
                ..
            } if channel_id == "file-content"
                && producer_node == "source"
                && output_port == "bytes"
                && consumer_node == "sink"
                && maximum_bytes == 64
        ));

        let files = Arc::new(crate::execution::files::FileBroker::default());
        files.prepare(&mut proposal).await.unwrap();
        let store = crate::storage::LocalStore::deferred(&root.join("deferred.db")).unwrap();
        let capabilities = CapabilityBroker::default();
        let grant = capabilities
            .issue_unprepared_for_test(&proposal, ExecutionDomain::Native)
            .await
            .unwrap();
        let implementation = ImplementationCandidate {
            tier: crate::compiler::InteractionTier::StructuredIntegration,
            executor: ExecutionDomain::Native,
            operation: "stream bounded bytes from the exact approved file".into(),
        };
        let compiled = CompiledAction {
            proposal: proposal.clone(),
            candidates: vec![implementation.clone()],
        };
        let native = NativeExecutor::new(
            root.clone(),
            Arc::new(UnsupportedPlatformController),
            files,
            store,
        );
        let mut broker = ExecutionBroker::new(capabilities);
        broker.register(Arc::new(native));

        let (procedure, descriptors) = procedure();
        let mut pool =
            crate::procedure_stream::ProcedureStreamPool::new(&procedure, &descriptors, task_id)
                .unwrap();
        let (_cancel, cancellation) = crate::procedure_stream::ProcedureStreamCancellation::new();
        let mut endpoints = pool.open_all(cancellation).unwrap();
        let source = endpoints.take_node("source").unwrap();
        let mut consumer = endpoints.take_node("sink").unwrap();
        let mut receiver = consumer.inputs_mut().remove("content").unwrap();
        let receive = tokio::spawn(async move {
            let mut bytes = Vec::new();
            while let Some(item) = receiver.recv().await? {
                bytes.extend_from_slice(item.as_bytes());
            }
            CoreResult::Ok(bytes)
        });

        let receipt = broker
            .execute_with_streams(&compiled, &implementation, &grant, Some(source))
            .await
            .unwrap();
        let streamed_bytes = receive.await.unwrap().unwrap();
        assert_eq!(streamed_bytes, b"bounded source bytes");
        assert_eq!(receipt.transient_data["bytes_read"], streamed_bytes.len());
        assert_eq!(receipt.transient_data["stream_channel_id"], channel_id);
        assert!(receipt.transient_data.get("bytes_base64").is_none());

        let observation = DeterministicObserver
            .observe(&proposal, &receipt)
            .await
            .unwrap();
        Verifier
            .verify(&proposal.expected_outcome, &observation)
            .unwrap();
        std::fs::write(&path, b"changed after the stream").unwrap();
        assert!(
            DeterministicObserver
                .observe(&proposal, &receipt)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn native_stream_read_to_write_verifies_the_complete_file_pipeline() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let source_path = root.join("pipeline-source.bin");
        let destination_path = root.join("pipeline-destination.bin");
        let source_bytes = b"bounded first-party pipeline bytes";
        std::fs::write(&source_path, source_bytes).unwrap();
        let task_id = Uuid::new_v4();
        let channel_id = "file-content";
        let mut read_proposal = ActionProposal {
            id: Uuid::new_v4(),
            task_id,
            action: Action::ReadFile {
                path: source_path.clone(),
                max_bytes: 64,
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: source_path.to_string_lossy().into_owned(),
            provenance: Provenance::user(),
            metadata: BTreeMap::from([
                ("procedure_id".into(), "native-stream-file-write".into()),
                ("procedure_node_id".into(), "source".into()),
                ("procedure_stream_node".into(), "true".into()),
                ("procedure_stream_output".into(), "bytes".into()),
                ("procedure_stream_channel_id".into(), channel_id.into()),
                ("procedure_stream_consumer_node".into(), "sink".into()),
                ("procedure_stream_max_bytes".into(), "64".into()),
            ]),
        };
        let mut write_proposal = ActionProposal {
            id: Uuid::new_v4(),
            task_id,
            action: Action::WriteFile {
                path: destination_path.clone(),
                content: String::new(),
                overwrite: false,
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: destination_path.to_string_lossy().into_owned(),
            provenance: Provenance::user(),
            metadata: BTreeMap::from([
                ("procedure_id".into(), "native-stream-file-write".into()),
                ("procedure_node_id".into(), "sink".into()),
                ("procedure_stream_node".into(), "true".into()),
                ("procedure_stream_input".into(), "content".into()),
                ("procedure_stream_channel_id".into(), channel_id.into()),
                ("procedure_stream_producer_node".into(), "source".into()),
                ("procedure_stream_max_bytes".into(), "64".into()),
            ]),
        };
        crate::verification::bind_required_outcome(&mut read_proposal).unwrap();
        crate::verification::bind_required_outcome(&mut write_proposal).unwrap();

        let files = Arc::new(crate::execution::files::FileBroker::default());
        files.prepare(&mut read_proposal).await.unwrap();
        files.prepare(&mut write_proposal).await.unwrap();
        let store = crate::storage::LocalStore::deferred(&root.join("deferred.db")).unwrap();
        let capabilities = CapabilityBroker::default();
        let read_grant = capabilities
            .issue_unprepared_for_test(&read_proposal, ExecutionDomain::Native)
            .await
            .unwrap();
        let write_grant = capabilities
            .issue_unprepared_for_test(&write_proposal, ExecutionDomain::Native)
            .await
            .unwrap();
        let implementation = ImplementationCandidate {
            tier: crate::compiler::InteractionTier::StructuredIntegration,
            executor: ExecutionDomain::Native,
            operation: "stream approved file bytes to an approved destination".into(),
        };
        let read_action = CompiledAction {
            proposal: read_proposal.clone(),
            candidates: vec![implementation.clone()],
        };
        let write_action = CompiledAction {
            proposal: write_proposal.clone(),
            candidates: vec![implementation.clone()],
        };
        let native = NativeExecutor::new(
            root.clone(),
            Arc::new(UnsupportedPlatformController),
            files,
            store,
        );
        let mut broker = ExecutionBroker::new(capabilities);
        broker.register(Arc::new(native));

        let (procedure, descriptors) = procedure();
        let mut pool =
            crate::procedure_stream::ProcedureStreamPool::new(&procedure, &descriptors, task_id)
                .unwrap();
        let (_cancel, cancellation) = crate::procedure_stream::ProcedureStreamCancellation::new();
        let mut endpoints = pool.open_all(cancellation).unwrap();
        let source_stream = endpoints.take_node("source").unwrap();
        let sink_stream = endpoints.take_node("sink").unwrap();

        let (read_receipt, write_receipt) = tokio::join!(
            broker.execute_with_streams(
                &read_action,
                &implementation,
                &read_grant,
                Some(source_stream),
            ),
            broker.execute_with_streams(
                &write_action,
                &implementation,
                &write_grant,
                Some(sink_stream),
            ),
        );
        let read_receipt = read_receipt.unwrap();
        let write_receipt = write_receipt.unwrap();

        assert_eq!(std::fs::read(&destination_path).unwrap(), source_bytes);
        assert_eq!(
            read_receipt.transient_data["stream_sha256"],
            write_receipt.transient_data["stream_sha256"]
        );
        let read_observation = DeterministicObserver
            .observe(&read_proposal, &read_receipt)
            .await
            .unwrap();
        Verifier
            .verify(&read_proposal.expected_outcome, &read_observation)
            .unwrap();
        let write_observation = DeterministicObserver
            .observe(&write_proposal, &write_receipt)
            .await
            .unwrap();
        Verifier
            .verify(&write_proposal.expected_outcome, &write_observation)
            .unwrap();
    }
}
