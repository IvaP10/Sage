use std::sync::Arc;

use chrono::Utc;
use sage_protocol::PROTOCOL_VERSION;
use sage_protocol::sage::ipc::v2 as wire;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::time::{Duration, timeout};
use uuid::Uuid;
use zeroize::Zeroize;

use crate::config::IpcEndpoint;
use crate::domain::{Task, TaskStatus};
use crate::engine::{ApprovalResolution, SageCore};
use crate::error::{CoreError, CoreResult};
use crate::events::{CoreEvent, CoreEventKind};
use crate::ipc::auth::{FeatureAuthentication, IpcAuthenticator};
use crate::ipc::codec::{read_frame, write_frame};
use crate::policy::RiskLevel;

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const WORLD_MODEL_FEATURE: &str = "world_model_v1";
const APPLICATION_CONTROL_FEATURE: &str = "application_control_v1";
const PROCEDURE_EXECUTION_FEATURE: &str = "procedure_execution_v1";
const NATIVE_FILE_STREAM_FEATURE: &str = "native_file_stream_v1";
const LAN_DEVICE_DISCOVERY_FEATURE: &str = "lan_device_discovery_v1";
const LAN_RENDERER_OBSERVATION_FEATURE: &str = "lan_renderer_observation_v1";

fn requires_lan_device_discovery_feature(operation: &str) -> bool {
    operation == "discover_upnp_media_renderers"
}

fn requires_lan_renderer_observation_feature(operation: &str) -> bool {
    matches!(
        operation,
        "observe_upnp_transport" | "observe_upnp_protocol_info"
    )
}

fn negotiated_lan_renderer_observation(features: &[String]) -> bool {
    features
        .iter()
        .any(|feature| feature == LAN_RENDERER_OBSERVATION_FEATURE)
        && features
            .iter()
            .any(|feature| feature == LAN_DEVICE_DISCOVERY_FEATURE)
}

fn requires_procedure_execution_feature(operation: &str) -> bool {
    matches!(
        operation,
        "run_goal" | "run_controller" | "run_stream_procedure" | "run_file_stream_copy"
    )
}

fn requires_native_file_stream_feature(operation: &str) -> bool {
    matches!(operation, "run_stream_procedure" | "run_file_stream_copy")
}

fn negotiate_features(client_kind: i32, client_features: &[String]) -> Vec<String> {
    let mut negotiated = Vec::new();
    if client_features
        .iter()
        .any(|feature| feature == WORLD_MODEL_FEATURE)
    {
        negotiated.push(WORLD_MODEL_FEATURE.to_owned());
    }
    if client_features
        .iter()
        .any(|feature| feature == LAN_DEVICE_DISCOVERY_FEATURE)
    {
        negotiated.push(LAN_DEVICE_DISCOVERY_FEATURE.to_owned());
    }
    if client_kind == wire::ClientKind::Macos as i32
        && cfg!(target_os = "macos")
        && client_features
            .iter()
            .any(|feature| feature == LAN_DEVICE_DISCOVERY_FEATURE)
        && client_features
            .iter()
            .any(|feature| feature == LAN_RENDERER_OBSERVATION_FEATURE)
    {
        negotiated.push(LAN_RENDERER_OBSERVATION_FEATURE.to_owned());
    }
    if client_kind == wire::ClientKind::Macos as i32
        && cfg!(target_os = "macos")
        && client_features
            .iter()
            .any(|feature| feature == APPLICATION_CONTROL_FEATURE)
    {
        negotiated.push(APPLICATION_CONTROL_FEATURE.to_owned());
    }
    if client_kind == wire::ClientKind::Macos as i32
        && cfg!(target_os = "macos")
        && client_features
            .iter()
            .any(|feature| feature == PROCEDURE_EXECUTION_FEATURE)
        && client_features
            .iter()
            .any(|feature| feature == APPLICATION_CONTROL_FEATURE)
        && client_features
            .iter()
            .any(|feature| feature == WORLD_MODEL_FEATURE)
    {
        negotiated.push(PROCEDURE_EXECUTION_FEATURE.to_owned());
    }
    if client_kind == wire::ClientKind::Macos as i32
        && cfg!(target_os = "macos")
        && client_features
            .iter()
            .any(|feature| feature == NATIVE_FILE_STREAM_FEATURE)
        && negotiated
            .iter()
            .any(|feature| feature == PROCEDURE_EXECUTION_FEATURE)
    {
        negotiated.push(NATIVE_FILE_STREAM_FEATURE.to_owned());
    }
    negotiated.sort();
    negotiated
}

pub async fn serve(
    core: Arc<SageCore>,
    endpoint: IpcEndpoint,
    authenticator: Arc<IpcAuthenticator>,
) -> CoreResult<()> {
    match endpoint {
        #[cfg(unix)]
        IpcEndpoint::UnixSocket(path) => {
            use std::os::unix::fs::{FileTypeExt, PermissionsExt};

            if tokio::fs::try_exists(&path).await? {
                let metadata = tokio::fs::symlink_metadata(&path).await?;
                if !metadata.file_type().is_socket() {
                    return Err(CoreError::Protocol(format!(
                        "refusing to replace non-socket IPC path {}",
                        path.display()
                    )));
                }
                match tokio::net::UnixStream::connect(&path).await {
                    Ok(_) => {
                        return Err(CoreError::Protocol(
                            "another SAGE Core instance is already serving this socket".into(),
                        ));
                    }
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
                        ) =>
                    {
                        tokio::fs::remove_file(&path).await?;
                    }
                    Err(error) => return Err(CoreError::Io(error)),
                }
            }
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            let listener = tokio::net::UnixListener::bind(&path)?;
            tokio::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).await?;
            tracing::info!(path = %path.display(), "SAGE Core IPC ready");
            loop {
                let (stream, _) = listener.accept().await?;
                // The socket is owner-only; additionally bind acceptance to the
                // owning OS user rather than trusting a role string alone.
                let owner = std::os::unix::fs::MetadataExt::uid(&std::fs::metadata(&path)?);
                if stream.peer_cred()?.uid() != owner {
                    continue;
                }
                let core = Arc::clone(&core);
                let authenticator = Arc::clone(&authenticator);
                tokio::spawn(async move {
                    if let Err(error) = handle_connection(stream, core, authenticator).await {
                        tracing::warn!(error = %error, "local IPC connection closed");
                    }
                });
            }
        }
        #[cfg(windows)]
        IpcEndpoint::NamedPipe(name) => {
            use tokio::net::windows::named_pipe::ServerOptions;

            let mut first = true;
            loop {
                let server = ServerOptions::new()
                    .first_pipe_instance(first)
                    .create(&name)?;
                first = false;
                server.connect().await?;
                let core = Arc::clone(&core);
                let authenticator = Arc::clone(&authenticator);
                tokio::spawn(async move {
                    if let Err(error) = handle_connection(server, core, authenticator).await {
                        tracing::warn!(error = %error, "local IPC connection closed");
                    }
                });
            }
        }
    }
}

async fn handle_connection<S>(
    stream: S,
    core: Arc<SageCore>,
    authenticator: Arc<IpcAuthenticator>,
) -> CoreResult<()>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (mut reader, mut writer) = tokio::io::split(stream);
    let mut server_nonce = vec![0_u8; 32];
    getrandom::fill(&mut server_nonce)
        .map_err(|error| CoreError::Protocol(format!("secure randomness failed: {error}")))?;
    let instance_id = Uuid::new_v4().to_string();
    let mut sequence = 1_u64;
    timeout(
        HANDSHAKE_TIMEOUT,
        write_frame(
            &mut writer,
            &frame(
                sequence,
                wire::frame::Payload::ServerChallenge(wire::ServerChallenge {
                    nonce: server_nonce.clone(),
                    minimum_protocol_version: PROTOCOL_VERSION,
                    maximum_protocol_version: PROTOCOL_VERSION,
                    core_instance_id: instance_id,
                    supported_features: vec![
                        APPLICATION_CONTROL_FEATURE.into(),
                        NATIVE_FILE_STREAM_FEATURE.into(),
                        PROCEDURE_EXECUTION_FEATURE.into(),
                        WORLD_MODEL_FEATURE.into(),
                        LAN_DEVICE_DISCOVERY_FEATURE.into(),
                        LAN_RENDERER_OBSERVATION_FEATURE.into(),
                    ],
                }),
            ),
        ),
    )
    .await
    .map_err(|_| CoreError::AuthenticationFailed)??;

    let authentication = timeout(HANDSHAKE_TIMEOUT, read_frame(&mut reader))
        .await
        .map_err(|_| CoreError::AuthenticationFailed)??;
    if authentication.protocol_version != PROTOCOL_VERSION {
        return Err(CoreError::AuthenticationFailed);
    }
    let wire::frame::Payload::ClientAuthenticate(client) = authentication
        .payload
        .ok_or(CoreError::AuthenticationFailed)?
    else {
        return Err(CoreError::AuthenticationFailed);
    };
    authenticator.verify_with_features(
        &server_nonce,
        &client.client_nonce,
        FeatureAuthentication {
            protocol_version: authentication.protocol_version,
            client_kind: client.client_kind,
            client_version: &client.client_version,
            features: &client.supported_features,
        },
        &client.proof,
    )?;
    let negotiated_features = negotiate_features(client.client_kind, &client.supported_features);
    sequence += 1;
    let session_id = Uuid::new_v4().to_string();
    timeout(
        HANDSHAKE_TIMEOUT,
        write_frame(
            &mut writer,
            &frame(
                sequence,
                wire::frame::Payload::AuthenticationResult(wire::AuthenticationResult {
                    accepted: true,
                    session_id: session_id.clone(),
                    error_code: String::new(),
                    message: "authenticated".into(),
                    server_proof: authenticator
                        .server_proof_with_features(
                            client.client_kind,
                            &client.proof,
                            &session_id,
                            &negotiated_features,
                        )?
                        .to_vec(),
                    negotiated_features: negotiated_features.clone(),
                }),
            ),
        ),
    )
    .await
    .map_err(|_| CoreError::AuthenticationFailed)??;

    let mut outbound =
        super::writer::OutboundWriter::new(writer, sequence, core.clone(), session_id.clone());
    use super::writer::Lane;
    let mut event_receiver = core.events().subscribe();
    let (adapter_sender, mut adapter_receiver) = tokio::sync::mpsc::channel(16);
    let (cancel_sender, mut cancel_receiver) = tokio::sync::mpsc::channel(128);
    let (terminate_sender, mut terminate_receiver) = tokio::sync::watch::channel(false);
    let is_browser = client.client_kind == wire::ClientKind::Browser as i32;
    let mut received_sequence = authentication.sequence;
    let mut request_ids = std::collections::HashSet::new();
    let (command_responses, mut response_receiver) = tokio::sync::mpsc::channel(16);
    let dispatcher = super::dispatch::CommandDispatcher::new(core.clone(), command_responses);
    // read_exact is not cancellation-safe. A dedicated reader retains partial
    // frames while adapter responses or UI events are being written.
    let (frames_sender, mut frames_receiver) = tokio::sync::mpsc::channel(4);
    let reader_task = tokio::spawn(async move {
        loop {
            let incoming = read_frame(&mut reader).await;
            let failed = incoming.is_err();
            if frames_sender.send(incoming).await.is_err() || failed {
                break;
            }
        }
    });
    let result=async {
    loop {
        tokio::select! {
            result = outbound.finished() => { return result; }
            incoming = frames_receiver.recv() => {
                let incoming = incoming.ok_or_else(||CoreError::Protocol("IPC reader disconnected".into()))??;
                if incoming.sequence <= received_sequence { return Err(CoreError::Protocol("Replayed IPC frame".into())); }
                received_sequence=incoming.sequence;
                if incoming.protocol_version != PROTOCOL_VERSION {
                    return Err(CoreError::Protocol("protocol version changed within a session".into()));
                }
                if let Some(wire::frame::Payload::UiCommand(ref command)) = incoming.payload {
                    let identity = parse_uuid(&command.request_id, "request_id")?;
                    let fresh = request_ids.insert(identity);
                    let submission = matches!(command.command, Some(wire::ui_command::Command::SubmitTask(_)));
                    if (!fresh && !submission) || request_ids.len() > 65_536 {
                        return Err(CoreError::Protocol("Replayed or excessive UI requests".into()));
                    }
                }
                match incoming.payload.clone() {
                    Some(wire::frame::Payload::AdapterHello(hello))=>{
                        if hello.cancellation_protocol > 1 { return Err(CoreError::Protocol("Unsupported adapter cancellation protocol".into())); }
                        core.adapters.register(&hello.domain,&session_id,client.client_kind,crate::execution::bridge::AdapterEndpoint {
                            requests: adapter_sender.clone(), cancellations: (hello.cancellation_protocol == 1).then(||cancel_sender.clone()), terminate: terminate_sender.clone(),
                        },&negotiated_features).await?;
                        continue;
                    },
                    Some(wire::frame::Payload::AdapterResult(response))=>{
                        if let Some(late) = core.adapters.complete(&session_id,response).await? { core.record_late_adapter_result(late).await?; }
                        continue;
                    },
                    Some(wire::frame::Payload::AdapterCancelAcknowledged(ack))=>{
                        if let Some(binding) = core.adapters.acknowledge_cancel(&session_id,&ack.request_id)? { core.worker_received_cancellation(ack.request_id,binding).await?; }
                        continue;
                    },
                    Some(wire::frame::Payload::Ping(_))=>{},
                    _ if is_browser=>return Err(CoreError::Protocol("Browser sessions cannot issue UI commands.".into())),
                    _=>{},
                }
                if let Some(wire::frame::Payload::UiCommand(command)) = incoming.payload {
                    let request_id = command.request_id.clone();
                    if matches!(
                        command.command,
                        Some(wire::ui_command::Command::WorldModelCommand(_))
                    ) && !negotiated_features
                        .iter()
                        .any(|feature| feature == WORLD_MODEL_FEATURE)
                    {
                        outbound.enqueue(
                            wire::frame::Payload::CoreEvent(error_event(
                                request_id,
                                CoreError::Protocol(
                                    "Client did not negotiate world_model_v1".into(),
                                ),
                            )),
                            super::writer::Lane::Control,
                        )?;
                        continue;
                    }
                    let requests_procedure_execution = matches!(
                        command.command.as_ref(),
                        Some(wire::ui_command::Command::WorldModelCommand(world_model))
                            if requires_procedure_execution_feature(&world_model.operation)
                    );
                    if requests_procedure_execution
                        && !negotiated_features
                            .iter()
                            .any(|feature| feature == PROCEDURE_EXECUTION_FEATURE)
                    {
                        outbound.enqueue(
                            wire::frame::Payload::CoreEvent(error_event(
                                request_id,
                                CoreError::Protocol(
                                    "Client did not negotiate procedure_execution_v1".into(),
                                ),
                            )),
                            super::writer::Lane::Control,
                        )?;
                        continue;
                    }
                    let requests_native_file_stream = matches!(
                        command.command.as_ref(),
                        Some(wire::ui_command::Command::WorldModelCommand(world_model))
                            if requires_native_file_stream_feature(&world_model.operation)
                    );
                    if requests_native_file_stream
                        && !negotiated_features
                            .iter()
                            .any(|feature| feature == NATIVE_FILE_STREAM_FEATURE)
                    {
                        outbound.enqueue(
                            wire::frame::Payload::CoreEvent(error_event(
                                request_id,
                                CoreError::Protocol(
                                    "Client did not negotiate native_file_stream_v1".into(),
                                ),
                            )),
                            super::writer::Lane::Control,
                        )?;
                        continue;
                    }
                    let requests_lan_device_discovery = matches!(
                        command.command.as_ref(),
                        Some(wire::ui_command::Command::WorldModelCommand(world_model))
                            if requires_lan_device_discovery_feature(&world_model.operation)
                    );
                    if requests_lan_device_discovery
                        && !negotiated_features
                            .iter()
                            .any(|feature| feature == LAN_DEVICE_DISCOVERY_FEATURE)
                    {
                        outbound.enqueue(
                            wire::frame::Payload::CoreEvent(error_event(
                                request_id,
                                CoreError::Protocol(
                                    "Client did not negotiate lan_device_discovery_v1".into(),
                                ),
                            )),
                            super::writer::Lane::Control,
                        )?;
                        continue;
                    }
                    let requests_lan_renderer_observation = matches!(
                        command.command.as_ref(),
                        Some(wire::ui_command::Command::WorldModelCommand(world_model))
                            if requires_lan_renderer_observation_feature(&world_model.operation)
                    );
                    if requests_lan_renderer_observation
                        && !negotiated_lan_renderer_observation(&negotiated_features)
                    {
                        outbound.enqueue(
                            wire::frame::Payload::CoreEvent(error_event(
                                request_id,
                                CoreError::Protocol(
                                    "Client did not negotiate lan_renderer_observation_v1 with lan_device_discovery_v1".into(),
                                ),
                            )),
                            super::writer::Lane::Control,
                        )?;
                        continue;
                    }
                    if let Err(error) = dispatcher.enqueue(command) {
                        outbound.enqueue(wire::frame::Payload::CoreEvent(error_event(request_id, error)), Lane::Control)?;
                    }
                    continue;
                }
                if let Some(response) = handle_frame(&core, incoming).await {
                    outbound.enqueue(response, Lane::Control)?;
                }
            }
            response = response_receiver.recv() => {
                if let Some(response) = response {
                    outbound.enqueue(response, Lane::Control)?;
                }
            }
            request = adapter_receiver.recv() => {
                if let Some(request)=request {
                    if core.adapters.is_pending(&session_id,&request.request_id) {
                        outbound.enqueue(wire::frame::Payload::AdapterRequest(Box::new(request)), Lane::Regular)?;
                    } else if let Some(late) = core.adapters.cancelled_before_send(&session_id,&request.request_id)? {
                        core.record_late_adapter_result(late).await?;
                    }
                }
            }
            cancel = cancel_receiver.recv() => {
                if let Some(cancel) = cancel { outbound.enqueue(wire::frame::Payload::AdapterCancel(cancel), Lane::Control)?; }
            }
            _ = terminate_receiver.changed() => {
                if *terminate_receiver.borrow() { return Err(CoreError::Protocol("Adapter cancellation capacity exhausted; reconnect to reconcile unfinished effects".into())); }
            }
            event = event_receiver.recv(), if !is_browser => {
                match event {
                    Ok(event) => {
                        outbound.enqueue(wire::frame::Payload::CoreEvent(event_to_wire(event)), Lane::Regular)?;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        let snapshot = core.snapshot(false).await;
                        outbound.enqueue(wire::frame::Payload::CoreEvent(snapshot_event(snapshot)), Lane::Regular)?;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(()),
                }
            }
        }
    }
    }.await;
    reader_task.abort();
    drop(outbound);
    drop(dispatcher);
    core.adapters.disconnect(&session_id).await;
    let _ = core.native_adapter_disconnected(&session_id);
    result
}

async fn handle_frame(core: &Arc<SageCore>, incoming: wire::Frame) -> Option<wire::frame::Payload> {
    match incoming.payload {
        Some(wire::frame::Payload::Ping(ping)) => {
            Some(wire::frame::Payload::Pong(wire::Pong { value: ping.value }))
        }
        Some(wire::frame::Payload::UiCommand(command)) => {
            let request_id = command.request_id.clone();
            let result = handle_command(core, command).await;
            match result {
                Ok(Some(event)) => Some(wire::frame::Payload::CoreEvent(event)),
                Ok(None) => None,
                Err(error) => Some(wire::frame::Payload::CoreEvent(error_event(
                    request_id, error,
                ))),
            }
        }
        _ => Some(wire::frame::Payload::CoreEvent(error_event(
            String::new(),
            CoreError::Protocol("unexpected frame after authentication".into()),
        ))),
    }
}

/// Enrich an already delivered grammar preview. The dispatcher may drop this
/// future at any point; this path must never perform a durable transition.
pub(super) async fn prepare_preview(
    core: &Arc<SageCore>,
    input: &wire::UpdateIntent,
    mut event: wire::CoreEvent,
) -> Option<wire::CoreEvent> {
    let Some(wire::core_event::Event::IntentPreview(preview)) = &mut event.event else {
        return None;
    };
    let can_suggest = !input.voice_input
        && preview.status != "unavailable"
        && crate::context::should_capture_live_reference(&input.text, false, false);
    if !matches!(preview.status.as_str(), "ready" | "preparing") && !can_suggest {
        return None;
    }
    let scopes = input
        .folder_roots
        .iter()
        .map(|root| crate::contracts::ResourceScope {
            root: root.into(),
            effects: std::collections::BTreeSet::from([crate::contracts::Effect::Read]),
        })
        .collect::<Vec<_>>();
    let intent = crate::intent::preparation(&input.text, &scopes);
    let prediction = async {
        if !can_suggest {
            return None;
        }
        match timeout(
            Duration::from_secs(2),
            core.contextual_routine_suggestion(&input.text),
        )
        .await
        {
            Ok(Ok(suggestion)) => suggestion,
            Err(_) | Ok(Err(_)) => None,
        }
    };
    #[cfg(target_os = "macos")]
    if input.voice_input
        && let Some((streamed_intent, prefix)) =
            crate::intent::streamed_action_prefix(&input.text, &scopes)
        && let Some(crate::domain::Action::ReadFile { path, .. }) =
            streamed_intent.steps.first().map(|step| &step.action)
    {
        let path = path.clone();
        match timeout(
            Duration::from_secs(2),
            core.prepare_streamed_file_read(&path, &scopes),
        )
        .await
        {
            Ok(Ok(())) => {
                preview.status = "prepared".into();
                preview.streamed_prefix = prefix;
                preview.detail = "This file is in a selected read folder. Sage will read it while you finish speaking, then wait before starting another step.".into();
            }
            Err(_) | Ok(Err(CoreError::Busy(_) | CoreError::Timeout(_))) => {
                preview.status = "preparing".into();
                preview.detail = "The file check is taking longer than expected. Send to retry; nothing has been read yet.".into();
            }
            Ok(Err(_)) => {
                preview.status = "unavailable".into();
                preview.detail = "Sage can start this read while you speak only when the file is inside a selected read folder. Nothing has been read yet.".into();
            }
        }
        event.event_id = Uuid::new_v4().to_string();
        event.occurred_at_unix_ms = Utc::now().timestamp_millis();
        return Some(event);
    }
    let Some(intent) = intent else {
        let suggestion = prediction.await?;
        preview.routine_suggestions = suggestion.requests;
        preview.routine_suggestion_detail = suggestion.detail;
        if preview.steps.is_empty() {
            preview.status = "suggested".into();
            preview.detail =
                "Choose a request to place it in the composer. Nothing has been submitted.".into();
        }
        event.event_id = Uuid::new_v4().to_string();
        event.occurred_at_unix_ms = Utc::now().timestamp_millis();
        return Some(event);
    };
    let (applications, suggestion) = tokio::join!(
        timeout(
            Duration::from_secs(2),
            core.prepare_intent_applications(&intent),
        ),
        prediction,
    );
    if let Some(suggestion) = suggestion {
        preview.routine_suggestions = suggestion.requests;
        preview.routine_suggestion_detail = suggestion.detail;
    }
    match applications {
        Ok(Ok(0)) if preview.routine_suggestions.is_empty() => return None,
        Ok(Ok(0)) => {}
        Ok(Ok(count)) => {
            preview.status = "prepared".into();
            let located = if count == 1 {
                "The app is located.".into()
            } else {
                format!("{count} apps are located.")
            };
            preview.detail = if preview.compiled_locally {
                format!(
                    "{located} Send when ready. Access and app identity are checked before opening."
                )
            } else {
                format!("{located} Finish your request and send when ready. Nothing has opened.")
            };
            #[cfg(target_os = "macos")]
            if input.voice_input
                && count == 1
                && let Some((prefix_intent, prefix)) =
                    crate::intent::streamed_action_prefix(&input.text, &scopes)
                && matches!(
                    prefix_intent.steps.first().map(|step| &step.action),
                    Some(crate::domain::Action::OpenApplication { .. })
                )
                && prefix_intent.steps.first().map(|step| &step.action)
                    == intent.steps.first().map(|step| &step.action)
            {
                preview.streamed_prefix = prefix.clone();
                preview.detail = format!(
                    "{located} Sage can request approval for “{prefix}” while you finish speaking. The app will open only after you approve."
                );
            }
        }
        Err(_) | Ok(Err(CoreError::Busy(_) | CoreError::Timeout(_))) => {
            preview.status = "preparing".into();
            preview.detail =
                "App preparation did not finish. You can send your request to retry.".into();
        }
        Ok(Err(_)) => {
            preview.status = "unavailable".into();
            preview.detail = "An app could not be located or verified. Check that it is installed and connected, then send to retry.".into();
        }
    }
    event.event_id = Uuid::new_v4().to_string();
    event.occurred_at_unix_ms = Utc::now().timestamp_millis();
    Some(event)
}

pub(super) async fn handle_command(
    core: &Arc<SageCore>,
    command: wire::UiCommand,
) -> CoreResult<Option<wire::CoreEvent>> {
    use wire::ui_command::Command;
    let request_id = command.request_id.clone();
    let needs_storage = match command.command.as_ref() {
        Some(
            Command::GetState(_)
            | Command::UpdateIntent(_)
            | Command::SaveProviderSettings(_)
            | Command::TestProviderConnection(_),
        ) => false,
        Some(Command::ControlTask(control))
            if control.operation == wire::control_task::Operation::Cancel as i32 =>
        {
            false
        }
        Some(Command::KnowledgeCommand(c)) if c.operation == "list" => false,
        Some(Command::WorkflowCommand(c)) if c.operation == "list" => false,
        _ => true,
    };
    if needs_storage {
        core.unlock_storage().await?;
    }
    match command
        .command
        .ok_or_else(|| CoreError::Protocol("UI command has no payload".into()))?
    {
        Command::UnlockStorage(_) => Ok(Some(snapshot_event(core.snapshot(true).await))),
        Command::UpdateIntent(input) => {
            if input.text.len() > crate::intent::MAX_INTENT_BYTES || input.folder_roots.len() > 32 {
                return Err(CoreError::Protocol(
                    "Intent input exceeds its bounds".into(),
                ));
            }
            let started = std::time::Instant::now();
            let scopes = input
                .folder_roots
                .iter()
                .map(|root| crate::contracts::ResourceScope {
                    root: root.into(),
                    effects: std::collections::BTreeSet::from([crate::contracts::Effect::Read]),
                })
                .collect::<Vec<_>>();
            let intent = crate::intent::compile(&input.text, &scopes);
            let prepared_intent = crate::intent::preparation(&input.text, &scopes);
            let steps = prepared_intent
                .as_ref()
                .map(|value| value.summaries())
                .unwrap_or_default();
            let compile_micros = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
            let mut status = if intent.is_some() {
                "ready"
            } else if steps.is_empty() {
                "listening"
            } else {
                "preparing"
            };
            let mut detail = if intent.is_some() {
                "Ready to run locally. Access is checked when you send.".to_string()
            } else if !steps.is_empty() {
                "This part is understood. Waiting for the rest of your request.".to_string()
            } else {
                String::new()
            };
            if prepared_intent.as_ref().is_some_and(|intent| {
                intent.steps.iter().any(|step| {
                    matches!(step.action, crate::domain::Action::OpenApplication { .. })
                })
            }) {
                status = "preparing";
                detail = "Locating your apps. You can send while preparation continues.".into();
            }
            if let Some(compiled) = &prepared_intent
                && !core.intent_available(compiled).await
            {
                status = "unavailable";
                detail = "A required action is unavailable on this device or its native connection is offline.".into();
            }
            if input.allow_interrupt
                && let Some(reflex) = crate::intent::reflex(&input.text)
            {
                let stream_id = parse_uuid(&input.stream_id, "stream_id")?;
                if core.has_streamed_run(stream_id) || !input.active_task_id.is_empty() {
                    let operation = if reflex == crate::intent::Reflex::Stop {
                        TaskStatus::Cancelled
                    } else {
                        TaskStatus::Paused
                    };
                    if core.has_streamed_run(stream_id) {
                        core.control_streamed_task(stream_id, operation).await?;
                    } else {
                        let task_id = parse_uuid(&input.active_task_id, "active_task_id")?;
                        core.control_task(task_id, operation).await?;
                    }
                    status = if operation == TaskStatus::Cancelled {
                        "stopping"
                    } else {
                        "paused"
                    };
                    detail =
                        "Further actions are held. Already dispatched effects may still finish."
                            .into();
                }
            }
            Ok(Some(wire::CoreEvent {
                event_id: Uuid::new_v4().to_string(),
                occurred_at_unix_ms: Utc::now().timestamp_millis(),
                event: Some(wire::core_event::Event::IntentPreview(
                    wire::IntentPreview {
                        stream_id: input.stream_id,
                        revision: input.revision,
                        steps,
                        compiled_locally: intent.is_some(),
                        status: status.into(),
                        detail,
                        compile_micros,
                        streamed_prefix: String::new(),
                        routine_suggestions: Vec::new(),
                        routine_suggestion_detail: String::new(),
                    },
                )),
            }))
        }
        Command::SubmitTask(submit) => {
            let key = crate::commands::SubmissionKey::for_request(&request_id, &submit)?;
            let is_streamed_submit = submit.streamed_prefix || submit.finalize_stream;
            if (is_streamed_submit
                && (submit.source != wire::InputSource::Voice as i32
                    || submit.voice_stream_id.is_empty()
                    || submit.streamed_prefix == submit.finalize_stream
                    || !submit.supersedes_task_id.is_empty()))
                || (!is_streamed_submit && !submit.voice_stream_id.is_empty())
            {
                return Err(CoreError::Protocol(
                    "Invalid streamed voice submission fields".into(),
                ));
            }
            let voice_stream_id = if is_streamed_submit {
                Some(parse_uuid(&submit.voice_stream_id, "voice_stream_id")?)
            } else {
                None
            };
            let conversation = if submit.conversation_id.is_empty() {
                None
            } else {
                Some(parse_uuid(&submit.conversation_id, "conversation_id")?)
            };
            let resources = submit
                .resources
                .into_iter()
                .map(|scope| {
                    let effects = scope
                        .effects
                        .into_iter()
                        .map(|effect| match wire::ResourceEffect::try_from(effect) {
                            Ok(wire::ResourceEffect::Read) => Ok(crate::contracts::Effect::Read),
                            Ok(wire::ResourceEffect::Create) => {
                                Ok(crate::contracts::Effect::Create)
                            }
                            _ => Err(CoreError::PermissionRequired(
                                "Unsupported scope effect".into(),
                            )),
                        })
                        .collect::<CoreResult<std::collections::BTreeSet<_>>>()?;
                    Ok(crate::contracts::ResourceScope {
                        root: scope.root.into(),
                        effects,
                    })
                })
                .collect::<CoreResult<Vec<_>>>()?;
            let task_id = if submit.streamed_prefix {
                core.submit_streamed_prefix_receipted(
                    submit.text,
                    conversation,
                    resources,
                    key,
                    voice_stream_id.expect("validated streamed prefix"),
                )
                .await?
            } else if submit.finalize_stream {
                core.finalize_streamed_prefix_receipted(
                    voice_stream_id.expect("validated stream finalization"),
                    submit.text,
                    resources,
                    key,
                )
                .await?
            } else if submit.supersedes_task_id.is_empty() {
                core.submit_receipted(submit.text, conversation, resources, key)
                    .await?
            } else {
                core.supersede_receipted(
                    parse_uuid(&submit.supersedes_task_id, "supersedes_task_id")?,
                    submit.text,
                    conversation,
                    resources,
                    key,
                )
                .await?
            };
            Ok(Some(wire::CoreEvent {
                event_id: Uuid::new_v4().to_string(),
                occurred_at_unix_ms: Utc::now().timestamp_millis(),
                event: Some(wire::core_event::Event::TaskAccepted(wire::TaskAccepted {
                    request_id,
                    task_id: task_id.to_string(),
                })),
            }))
        }
        Command::KnowledgeCommand(command) => {
            let snapshot = core.knowledge_command(command).await?;
            Ok(Some(wire::CoreEvent {
                event_id: Uuid::new_v4().to_string(),
                occurred_at_unix_ms: Utc::now().timestamp_millis(),
                event: Some(wire::core_event::Event::KnowledgeState(
                    wire::KnowledgeState {
                        json: serde_json::to_string(&snapshot)?,
                    },
                )),
            }))
        }
        Command::WorkflowCommand(command) => {
            let snapshot = core.workflow_command(command).await?;
            Ok(Some(wire::CoreEvent {
                event_id: Uuid::new_v4().to_string(),
                occurred_at_unix_ms: Utc::now().timestamp_millis(),
                event: Some(wire::core_event::Event::WorkflowState(
                    wire::WorkflowState {
                        json: serde_json::to_string(&snapshot)?,
                    },
                )),
            }))
        }
        Command::WorldModelCommand(command) => {
            let snapshot = core.world_model_command(command).await?;
            Ok(Some(wire::CoreEvent {
                event_id: Uuid::new_v4().to_string(),
                occurred_at_unix_ms: Utc::now().timestamp_millis(),
                event: Some(wire::core_event::Event::WorldModelState(
                    wire::WorldModelState {
                        json: serde_json::to_string(&snapshot)?,
                    },
                )),
            }))
        }
        Command::ControlTask(control) => {
            let task_id = parse_uuid(&control.task_id, "task_id")?;
            let status = match wire::control_task::Operation::try_from(control.operation) {
                Ok(wire::control_task::Operation::Pause) => TaskStatus::Paused,
                Ok(wire::control_task::Operation::Resume) => TaskStatus::Running,
                Ok(wire::control_task::Operation::Cancel) => TaskStatus::Cancelled,
                _ => return Err(CoreError::Protocol("invalid task operation".into())),
            };
            core.control_task(task_id, status).await?;
            Ok(None)
        }
        Command::ApprovalResponse(response) => {
            let resolution = match wire::ApprovalDecision::try_from(response.decision) {
                Ok(wire::ApprovalDecision::ApproveOnce) => ApprovalResolution::Approved {
                    native_authentication_satisfied: response.native_authentication_satisfied,
                },
                Ok(wire::ApprovalDecision::Deny) => ApprovalResolution::Denied,
                _ => return Err(CoreError::Protocol("invalid approval decision".into())),
            };
            core.resolve_approval(
                parse_uuid(&response.approval_id, "approval_id")?,
                parse_uuid(&response.task_id, "task_id")?,
                parse_uuid(&response.action_id, "action_id")?,
                &response.approval_digest,
                resolution,
            )
            .await?;
            Ok(None)
        }
        Command::GetState(request) => Ok(Some(snapshot_event(
            core.snapshot(request.include_completed_tasks).await,
        ))),
        Command::UpdatePermission(permission) => {
            core.update_permission(&permission.permission, permission.granted)?;
            Ok(None)
        }
        Command::UndoLastAction(undo) => {
            if undo.action_id.is_empty() {
                return Err(CoreError::InvalidAction(
                    "Refresh or update Sage before Undo; the action identity is missing.".into(),
                ));
            }
            core.undo_last_action(
                parse_uuid(&undo.task_id, "task_id")?,
                parse_uuid(&undo.action_id, "action_id")?,
            )
            .await?;
            Ok(None)
        }
        Command::UserAnswer(answer) => {
            core.answer_question(
                parse_uuid(&answer.question_id, "question_id")?,
                parse_uuid(&answer.task_id, "task_id")?,
                parse_uuid(&answer.action_id, "action_id")?,
                answer.answer,
            )
            .await?;
            Ok(None)
        }
        Command::SaveProviderSettings(mut settings) => {
            settings.api_key.zeroize();
            settings.endpoint.zeroize();
            Err(CoreError::PolicyDenied(
                "Provider configuration is disabled; Sage uses only first-party local inference"
                    .into(),
            ))
        }
        Command::TestProviderConnection(mut settings) => {
            settings.api_key.zeroize();
            settings.endpoint.zeroize();
            Ok(Some(provider_connection_result_event(
                request_id,
                false,
                String::new(),
                String::new(),
                "Provider connection tests are disabled in Sage".into(),
            )))
        }
    }
}

fn frame(sequence: u64, payload: wire::frame::Payload) -> wire::Frame {
    wire::Frame {
        protocol_version: PROTOCOL_VERSION,
        sequence,
        payload: Some(payload),
    }
}

fn snapshot_event(snapshot: crate::events::StateSnapshot) -> wire::CoreEvent {
    wire::CoreEvent {
        event_id: Uuid::new_v4().to_string(),
        occurred_at_unix_ms: Utc::now().timestamp_millis(),
        event: Some(wire::core_event::Event::StateSnapshot(
            wire::StateSnapshot {
                tasks: snapshot.tasks.into_iter().map(task_to_wire).collect(),
                pending_approvals: snapshot
                    .pending_approvals
                    .into_iter()
                    .map(|p| wire::ApprovalRequest {
                        approval_id: p.approval_id.to_string(),
                        approval_digest: p.digest,
                        task_id: p.task_id.to_string(),
                        action_id: p.action_id.to_string(),
                        title: "Review prepared action".into(),
                        explanation: p.explanation,
                        resource: p.resource,
                        risk: risk_to_wire(p.risk) as i32,
                        expires_at_unix_ms: p.expires_at.timestamp_millis(),
                        reversible: p.reversible,
                        requires_native_authentication: p.requires_native_authentication,
                    })
                    .collect(),
                pending_questions: snapshot
                    .pending_questions
                    .into_iter()
                    .map(|q| wire::QuestionRequest {
                        question_id: q.question_id.to_string(),
                        task_id: q.task_id.to_string(),
                        action_id: q.action_id.to_string(),
                        question: q.question,
                        expires_at_unix_ms: q.expires_at.timestamp_millis(),
                    })
                    .collect(),
                storage_locked: snapshot.storage_locked,
                core_version: snapshot.core_version,
                protocol_version: snapshot.protocol_version,
                knowledge: snapshot
                    .knowledge
                    .and_then(|knowledge| serde_json::to_string(&knowledge).ok())
                    .map(|json| wire::KnowledgeState { json }),
                provider_settings: Vec::new(),
            },
        )),
    }
}

fn provider_connection_result_event(
    request_id: String,
    success: bool,
    provider: String,
    model: String,
    message: String,
) -> wire::CoreEvent {
    wire::CoreEvent {
        event_id: Uuid::new_v4().to_string(),
        occurred_at_unix_ms: Utc::now().timestamp_millis(),
        event: Some(wire::core_event::Event::ProviderConnectionResult(
            wire::ProviderConnectionResult {
                request_id,
                success,
                provider,
                model,
                message,
            },
        )),
    }
}

fn event_to_wire(event: CoreEvent) -> wire::CoreEvent {
    let event_id = event.id.to_string();
    let occurred_at_unix_ms = event.occurred_at.timestamp_millis();
    let task_id = event.task_id.map(|id| id.to_string()).unwrap_or_default();
    let payload = match event.kind {
        CoreEventKind::DecisionResolved { decision_id, state } => {
            wire::core_event::Event::DecisionResolved(wire::DecisionResolved {
                decision_id: decision_id.to_string(),
                task_id,
                state,
            })
        }
        CoreEventKind::ModelResponse { text, finished } => {
            wire::core_event::Event::ModelResponseDelta(wire::ModelResponseDelta {
                task_id,
                text,
                finished,
            })
        }
        CoreEventKind::ApprovalRequested {
            approval_id,
            action_id,
            digest,
            explanation,
            resource,
            risk,
            expires_at,
            reversible,
            requires_native_authentication,
        } => wire::core_event::Event::ApprovalRequest(wire::ApprovalRequest {
            approval_id: approval_id.to_string(),
            approval_digest: digest,
            task_id,
            action_id: action_id.to_string(),
            title: "SAGE needs approval".into(),
            explanation,
            resource,
            risk: risk_to_wire(risk) as i32,
            expires_at_unix_ms: expires_at.timestamp_millis(),
            reversible,
            requires_native_authentication,
        }),
        CoreEventKind::QuestionRequested {
            question_id,
            action_id,
            question,
            expires_at,
        } => wire::core_event::Event::QuestionRequest(wire::QuestionRequest {
            question_id: question_id.to_string(),
            task_id,
            action_id: action_id.to_string(),
            question,
            expires_at_unix_ms: expires_at.timestamp_millis(),
        }),
        CoreEventKind::Error {
            code,
            message,
            recoverable,
        } => wire::core_event::Event::Error(wire::ErrorEvent {
            request_id: String::new(),
            task_id,
            code,
            message,
            recoverable,
        }),
        CoreEventKind::TaskCompleted { outcome } => {
            wire::core_event::Event::Notification(wire::NotificationEvent {
                title: "Task completed".into(),
                body: outcome,
                task_id,
            })
        }
        kind => wire::core_event::Event::AgentEvent(agent_event(task_id, kind)),
    };
    wire::CoreEvent {
        event_id,
        occurred_at_unix_ms,
        event: Some(payload),
    }
}

fn agent_event(task_id: String, kind: CoreEventKind) -> wire::AgentEvent {
    let (action_id, name, title, detail, risk) = match kind {
        CoreEventKind::ModelResponse { text, .. } => (
            String::new(),
            "model_response",
            "Response".into(),
            text,
            None,
        ),
        CoreEventKind::TaskStarted => (
            String::new(),
            "task_started",
            "Task started".into(),
            String::new(),
            None,
        ),
        CoreEventKind::PlanGenerated { action_count } => (
            String::new(),
            "plan_generated",
            "Plan ready".into(),
            format!("{action_count} structured actions"),
            None,
        ),
        CoreEventKind::ActionProposed { action_id, summary } => (
            action_id.to_string(),
            "action_proposed",
            "Action proposed".into(),
            summary,
            None,
        ),
        CoreEventKind::PolicyDenied { action_id, reason } => (
            action_id.to_string(),
            "policy_denied",
            "Policy denied action".into(),
            reason,
            Some(RiskLevel::Prohibited),
        ),
        CoreEventKind::ApprovalResolved {
            action_id,
            approved,
        } => (
            action_id.to_string(),
            "approval_resolved",
            "Approval resolved".into(),
            format!("approved={approved}"),
            None,
        ),
        CoreEventKind::ActionStarted {
            action_id,
            implementation,
        } => (
            action_id.to_string(),
            "action_started",
            "Action started".into(),
            implementation,
            None,
        ),
        CoreEventKind::ActionSucceeded { action_id, summary } => (
            action_id.to_string(),
            "action_succeeded",
            "Action verified".into(),
            summary,
            None,
        ),
        CoreEventKind::ActionFailed { action_id, error } => (
            action_id.to_string(),
            "action_failed",
            "Action failed".into(),
            error,
            None,
        ),
        CoreEventKind::ObservationReceived { action_id, summary } => (
            action_id.to_string(),
            "observation_received",
            "State observed".into(),
            summary,
            None,
        ),
        CoreEventKind::VerificationFailed { action_id, reason } => (
            action_id.to_string(),
            "verification_failed",
            "Verification failed".into(),
            reason,
            None,
        ),
        CoreEventKind::ReplanningStarted { attempt } => (
            String::new(),
            "replanning_started",
            "Replanning".into(),
            format!("attempt {attempt}"),
            None,
        ),
        CoreEventKind::ReferenceContext { summary } => (
            String::new(),
            "reference_context",
            "Current context".into(),
            summary,
            None,
        ),
        CoreEventKind::PermissionChanged {
            permission,
            granted,
        } => (
            String::new(),
            "permission_changed",
            "Permission changed".into(),
            format!("{permission}: {granted}"),
            None,
        ),
        CoreEventKind::ModelDisconnected { provider } => (
            String::new(),
            "model_disconnected",
            "Model disconnected".into(),
            provider,
            None,
        ),
        CoreEventKind::SandboxTerminated { reason } => (
            String::new(),
            "sandbox_terminated",
            "Sandbox terminated".into(),
            reason,
            None,
        ),
        CoreEventKind::TaskStatusChanged { status, summary } => (
            String::new(),
            "task_status_changed",
            format!("Task {status:?}"),
            summary,
            None,
        ),
        CoreEventKind::UndoChanged { action_id, phase } => (
            action_id.to_string(),
            "undo_changed",
            if phase == crate::domain::UndoPhase::Verified {
                "Undo verified"
            } else {
                "Undo"
            }
            .into(),
            phase.summary().into(),
            None,
        ),
        CoreEventKind::ApprovalRequested { .. }
        | CoreEventKind::DecisionResolved { .. }
        | CoreEventKind::QuestionRequested { .. }
        | CoreEventKind::TaskCompleted { .. }
        | CoreEventKind::Error { .. } => (
            String::new(),
            "event",
            "SAGE event".into(),
            String::new(),
            None,
        ),
    };
    wire::AgentEvent {
        task_id,
        action_id,
        kind: name.into(),
        title,
        detail,
        risk: risk.map_or(wire::RiskLevel::Unspecified as i32, |value| {
            risk_to_wire(value) as i32
        }),
    }
}

fn task_to_wire(task: Task) -> wire::TaskUpdate {
    let facts = crate::domain::ExecutionFacts::for_task(&task);
    let completed_actions = facts.verified;
    let total_actions = task.current_actions().count() as u32;
    let (intent_revision, intent_change_summary) = task
        .intent
        .as_ref()
        .map(|intent| (intent.revision, intent.change_summary.clone()))
        .unwrap_or_default();
    let current_action = task
        .current_actions()
        .map(|(_, action)| action)
        .find(|action| {
            matches!(
                action.status,
                crate::domain::ActionStatus::Running
                    | crate::domain::ActionStatus::WaitingForApproval
                    | crate::domain::ActionStatus::Verifying
            )
        })
        .map(|action| crate::intent::summary(&action.proposal.action))
        .unwrap_or_default();
    wire::TaskUpdate {
        intent_revision,
        intent_change_summary,
        routine_summary: if task.compiled_routine.is_some() {
            "Using your reviewed routine".into()
        } else {
            String::new()
        },
        task_id: task.id.to_string(),
        control_scope_id: task.control_scope().to_string(),
        continued_task_id: task
            .continued_by
            .map(|id| id.to_string())
            .unwrap_or_default(),
        request: task.request,
        status: task_status_to_wire(task.status) as i32,
        summary: task.goal.unwrap_or_default(),
        completed_actions,
        total_actions,
        current_action,
        final_outcome: task.final_outcome.unwrap_or_default(),
        undo_available: task.rollback_available,
        undo_action_id: task
            .rollback_action_id
            .map(|id| id.to_string())
            .unwrap_or_default(),
        undo_state: task
            .undo
            .as_ref()
            .map(|undo| undo.phase.as_str())
            .unwrap_or_default()
            .into(),
        undo_summary: task
            .undo
            .as_ref()
            .map(|undo| undo.phase.summary())
            .unwrap_or_default()
            .into(),
        execution_facts: Some(wire::ExecutionFacts {
            verified: facts.verified,
            failed: facts.failed,
            pending: facts.pending,
            uncertain: facts.uncertain,
            undone: facts.undone,
            summary: facts.summary(),
        }),
        conversation_id: task
            .conversation_id
            .map(|id| id.to_string())
            .unwrap_or_default(),
        message_id: task.message_id.map(|id| id.to_string()).unwrap_or_default(),
        actions: task
            .actions
            .values()
            .map(|state| wire::ActionProgress {
                action_id: state.proposal.id.to_string(),
                summary: crate::intent::summary(&state.proposal.action),
                status: if task
                    .intent
                    .as_ref()
                    .is_some_and(|intent| intent.retired.contains(&state.proposal.id))
                {
                    "replaced".into()
                } else if task.undone_actions.contains(&state.proposal.id) {
                    "undone".into()
                } else if task.undo.as_ref().is_some_and(|undo| {
                    undo.action_id == state.proposal.id
                        && matches!(
                            undo.phase,
                            crate::domain::UndoPhase::Dispatched
                                | crate::domain::UndoPhase::Uncertain
                        )
                }) {
                    "undo_uncertain".into()
                } else {
                    format!("{:?}", state.status).to_ascii_lowercase()
                },
            })
            .collect(),
    }
}

fn task_status_to_wire(status: TaskStatus) -> wire::TaskStatus {
    match status {
        TaskStatus::Pending => wire::TaskStatus::Pending,
        TaskStatus::Planning => wire::TaskStatus::Planning,
        TaskStatus::Running => wire::TaskStatus::Running,
        TaskStatus::WaitingForApproval => wire::TaskStatus::WaitingForApproval,
        TaskStatus::WaitingForUser => wire::TaskStatus::WaitingForUser,
        TaskStatus::Paused => wire::TaskStatus::Paused,
        TaskStatus::Succeeded => wire::TaskStatus::Succeeded,
        TaskStatus::Answered => wire::TaskStatus::Answered,
        TaskStatus::Partial => wire::TaskStatus::Partial,
        TaskStatus::Failed => wire::TaskStatus::Failed,
        TaskStatus::Cancelled => wire::TaskStatus::Cancelled,
        TaskStatus::Interrupted => wire::TaskStatus::Interrupted,
    }
}

fn risk_to_wire(risk: RiskLevel) -> wire::RiskLevel {
    match risk {
        RiskLevel::Safe => wire::RiskLevel::Safe,
        RiskLevel::Sensitive => wire::RiskLevel::Sensitive,
        RiskLevel::Consequential => wire::RiskLevel::Consequential,
        RiskLevel::Destructive => wire::RiskLevel::Destructive,
        RiskLevel::Privileged => wire::RiskLevel::Privileged,
        RiskLevel::Prohibited => wire::RiskLevel::Prohibited,
    }
}

pub(super) fn error_event(request_id: String, error: CoreError) -> wire::CoreEvent {
    // A transient failure can occur before a retry reaches the durable inbox.
    // The client must retain that retry's identity: an earlier attempt may have
    // committed even though this session could not confirm it.
    let retryable = matches!(
        error,
        CoreError::Busy(_)
            | CoreError::Storage(_)
            | CoreError::SecretStore(_)
            | CoreError::Io(_)
            | CoreError::Timeout(_)
            | CoreError::ExecutorUnavailable(_)
    );
    wire::CoreEvent {
        event_id: Uuid::new_v4().to_string(),
        occurred_at_unix_ms: Utc::now().timestamp_millis(),
        event: Some(wire::core_event::Event::Error(wire::ErrorEvent {
            request_id,
            task_id: String::new(),
            code: if retryable {
                "ipc_command_retryable"
            } else {
                "ipc_command_failed"
            }
            .into(),
            message: error.to_string(),
            recoverable: true,
        })),
    }
}

fn parse_uuid(value: &str, field: &str) -> CoreResult<Uuid> {
    Uuid::parse_str(value).map_err(|_| CoreError::Protocol(format!("{field} is not a valid UUID")))
}

#[cfg(test)]
mod protocol_v2_tests {
    use super::*;
    use crate::{model::UnconfiguredModelProvider, secrets::SecretBytes};

    #[test]
    fn learned_control_feature_requires_explicit_mac_client_support() {
        let features = vec![
            APPLICATION_CONTROL_FEATURE.to_owned(),
            WORLD_MODEL_FEATURE.to_owned(),
        ];
        assert_eq!(
            negotiate_features(wire::ClientKind::Windows as i32, &features),
            vec![WORLD_MODEL_FEATURE.to_owned()]
        );
        assert_eq!(
            negotiate_features(
                wire::ClientKind::Macos as i32,
                &[WORLD_MODEL_FEATURE.into()]
            ),
            vec![WORLD_MODEL_FEATURE.to_owned()]
        );
        assert_eq!(
            negotiate_features(wire::ClientKind::Macos as i32, &features),
            if cfg!(target_os = "macos") {
                vec![
                    APPLICATION_CONTROL_FEATURE.to_owned(),
                    WORLD_MODEL_FEATURE.to_owned(),
                ]
            } else {
                vec![WORLD_MODEL_FEATURE.to_owned()]
            }
        );
        let procedure_features = vec![
            APPLICATION_CONTROL_FEATURE.to_owned(),
            PROCEDURE_EXECUTION_FEATURE.to_owned(),
            WORLD_MODEL_FEATURE.to_owned(),
        ];
        assert_eq!(
            negotiate_features(wire::ClientKind::Macos as i32, &procedure_features),
            if cfg!(target_os = "macos") {
                vec![
                    APPLICATION_CONTROL_FEATURE.to_owned(),
                    PROCEDURE_EXECUTION_FEATURE.to_owned(),
                    WORLD_MODEL_FEATURE.to_owned(),
                ]
            } else {
                vec![WORLD_MODEL_FEATURE.to_owned()]
            }
        );
        assert_eq!(
            negotiate_features(
                wire::ClientKind::Macos as i32,
                &[
                    APPLICATION_CONTROL_FEATURE.into(),
                    PROCEDURE_EXECUTION_FEATURE.into()
                ],
            ),
            if cfg!(target_os = "macos") {
                vec![APPLICATION_CONTROL_FEATURE.to_owned()]
            } else {
                Vec::new()
            }
        );
        assert!(requires_procedure_execution_feature("run_stream_procedure"));
        assert!(requires_procedure_execution_feature("run_file_stream_copy"));
        assert!(requires_procedure_execution_feature("run_goal"));
        assert!(!requires_procedure_execution_feature("synthesize_goal"));
        assert!(requires_native_file_stream_feature("run_stream_procedure"));
        assert!(requires_native_file_stream_feature("run_file_stream_copy"));
        assert!(!requires_native_file_stream_feature("run_goal"));
        assert!(requires_lan_device_discovery_feature(
            "discover_upnp_media_renderers"
        ));
        assert!(!requires_lan_device_discovery_feature(
            "discover_current_application"
        ));
        assert!(requires_lan_renderer_observation_feature(
            "observe_upnp_transport"
        ));
        assert!(requires_lan_renderer_observation_feature(
            "observe_upnp_protocol_info"
        ));
        assert!(!requires_lan_renderer_observation_feature(
            "discover_upnp_media_renderers"
        ));
        let lan_observation_features = vec![
            LAN_DEVICE_DISCOVERY_FEATURE.to_owned(),
            LAN_RENDERER_OBSERVATION_FEATURE.to_owned(),
        ];
        assert!(negotiated_lan_renderer_observation(
            &lan_observation_features
        ));
        assert!(!negotiated_lan_renderer_observation(&[
            LAN_DEVICE_DISCOVERY_FEATURE.to_owned()
        ]));
        assert!(!negotiated_lan_renderer_observation(&[
            LAN_RENDERER_OBSERVATION_FEATURE.to_owned()
        ]));
        assert_eq!(
            negotiate_features(wire::ClientKind::Windows as i32, &lan_observation_features),
            vec![LAN_DEVICE_DISCOVERY_FEATURE.to_owned()]
        );
        assert_eq!(
            negotiate_features(wire::ClientKind::Macos as i32, &lan_observation_features),
            if cfg!(target_os = "macos") {
                vec![
                    LAN_DEVICE_DISCOVERY_FEATURE.to_owned(),
                    LAN_RENDERER_OBSERVATION_FEATURE.to_owned(),
                ]
            } else {
                vec![LAN_DEVICE_DISCOVERY_FEATURE.to_owned()]
            }
        );
        let file_stream_features = vec![
            APPLICATION_CONTROL_FEATURE.to_owned(),
            PROCEDURE_EXECUTION_FEATURE.to_owned(),
            WORLD_MODEL_FEATURE.to_owned(),
            NATIVE_FILE_STREAM_FEATURE.to_owned(),
        ];
        assert_eq!(
            negotiate_features(wire::ClientKind::Macos as i32, &file_stream_features),
            if cfg!(target_os = "macos") {
                vec![
                    APPLICATION_CONTROL_FEATURE.to_owned(),
                    NATIVE_FILE_STREAM_FEATURE.to_owned(),
                    PROCEDURE_EXECUTION_FEATURE.to_owned(),
                    WORLD_MODEL_FEATURE.to_owned(),
                ]
            } else {
                vec![WORLD_MODEL_FEATURE.to_owned()]
            }
        );
    }

    #[tokio::test]
    async fn legacy_provider_commands_cannot_persist_or_echo_credentials() {
        let data = tempfile::tempdir().unwrap();
        let core = SageCore::new(
            crate::config::CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();

        let save = wire::UiCommand {
            request_id: "legacy-provider-save".into(),
            command: Some(wire::ui_command::Command::SaveProviderSettings(
                wire::SaveProviderSettings {
                    role: "reasoning".into(),
                    provider: "external".into(),
                    model: "remote-model".into(),
                    endpoint: "https://provider.invalid".into(),
                    api_key: "sk-test-secret".into(),
                    remove_saved_key: false,
                    native_authentication_satisfied: false,
                },
            )),
        };
        let error = handle_command(&core, save).await.unwrap_err();
        assert!(matches!(error, CoreError::PolicyDenied(_)));
        assert!(!error.to_string().contains("sk-test-secret"));
        assert!(!error.to_string().contains("provider.invalid"));

        let probe = wire::UiCommand {
            request_id: "legacy-provider-probe".into(),
            command: Some(wire::ui_command::Command::TestProviderConnection(
                wire::TestProviderConnection {
                    role: "reasoning".into(),
                    provider: "external".into(),
                    model: "remote-model".into(),
                    endpoint: "https://provider.invalid".into(),
                    api_key: "sk-test-secret".into(),
                },
            )),
        };
        let event = handle_command(&core, probe)
            .await
            .unwrap()
            .expect("legacy clients receive a terminal compatibility event");
        let Some(wire::core_event::Event::ProviderConnectionResult(result)) = event.event else {
            panic!("provider probe compatibility event is missing");
        };
        assert!(!result.success);
        assert_eq!(result.request_id, "legacy-provider-probe");
        assert!(result.provider.is_empty());
        assert!(result.model.is_empty());
        assert!(!result.message.contains("sk-test-secret"));
        assert!(!result.message.contains("provider.invalid"));
    }

    struct StalledConnectionTest;

    #[async_trait::async_trait]
    impl crate::model::ModelProvider for StalledConnectionTest {
        fn descriptor(&self) -> crate::model::ProviderDescriptor {
            crate::model::ModelProvider::descriptor(&UnconfiguredModelProvider)
        }
        async fn create_plan(
            &self,
            _: crate::model::PlanningContext,
        ) -> CoreResult<crate::domain::ActionGraph> {
            std::future::pending().await
        }
        async fn replan(
            &self,
            _: crate::model::ReplanContext,
        ) -> CoreResult<crate::domain::ActionGraph> {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn stop_commits_while_authenticated_peer_does_not_read_outbound_frames() {
        let data = tempfile::tempdir().unwrap();
        let core = SageCore::new(
            crate::config::CoreConfig::for_test(data.path()),
            Arc::new(StalledConnectionTest),
        )
        .unwrap();
        let task_id = core
            .submit_task("Wait for the fixture model")
            .await
            .unwrap();
        let auth = Arc::new(IpcAuthenticator::new(SecretBytes::new(vec![7; 32])));
        let (mut client, server) = tokio::io::duplex(1024);
        let session = tokio::spawn(handle_connection(server, core.clone(), auth));
        let kind = if cfg!(windows) {
            wire::ClientKind::Windows
        } else {
            wire::ClientKind::Macos
        };
        handshake(&mut client, kind, &[7; 32]).await;
        // This frame cannot fit in the client's unread buffer. The old server
        // awaited write_all here and never admitted the following Stop.
        core.events().publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::ModelResponse {
                text: "x".repeat(128 * 1024),
                finished: false,
            },
        ));
        tokio::task::yield_now().await;
        write_frame(
            &mut client,
            &frame(
                2,
                wire::frame::Payload::UiCommand(wire::UiCommand {
                    request_id: Uuid::new_v4().to_string(),
                    command: Some(wire::ui_command::Command::ControlTask(wire::ControlTask {
                        task_id: task_id.to_string(),
                        operation: wire::control_task::Operation::Cancel as i32,
                    })),
                }),
            ),
        )
        .await
        .unwrap();
        timeout(Duration::from_millis(500), async {
            loop {
                if core
                    .snapshot(true)
                    .await
                    .tasks
                    .iter()
                    .any(|task| task.id == task_id && task.status == TaskStatus::Cancelled)
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("Stop admission waited for the outbound reader");
        assert!(
            !session.is_finished(),
            "The session should still be waiting for the peer to read"
        );
        drop(client);
        assert!(
            timeout(Duration::from_secs(2), session)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
    }

    #[tokio::test]
    async fn stop_and_ping_bypass_a_stalled_and_saturated_ordinary_lane() {
        let data = tempfile::tempdir().unwrap();
        let provider = Arc::new(StalledConnectionTest);
        let core = SageCore::new(
            crate::config::CoreConfig::for_test(data.path()),
            provider.clone(),
        )
        .unwrap();
        let task_id = core
            .submit_task("Wait for the fixture model")
            .await
            .unwrap();
        let auth = Arc::new(IpcAuthenticator::new(SecretBytes::new(vec![7; 32])));
        let (mut client, server) = tokio::io::duplex(8192);
        let session = tokio::spawn(handle_connection(server, core.clone(), auth));
        let kind = if cfg!(windows) {
            wire::ClientKind::Windows
        } else {
            wire::ClientKind::Macos
        };
        handshake(&mut client, kind, &[7; 32]).await;
        write_frame(
            &mut client,
            &frame(
                2,
                wire::frame::Payload::AdapterHello(wire::AdapterHello {
                    domain: "native".into(),
                    cancellation_protocol: 1,
                }),
            ),
        )
        .await
        .unwrap();
        write_frame(
            &mut client,
            &frame(
                3,
                wire::frame::Payload::UiCommand(wire::UiCommand {
                    request_id: Uuid::new_v4().to_string(),
                    command: Some(wire::ui_command::Command::WorldModelCommand(
                        wire::WorldModelCommand {
                            operation: "discover_current_application".into(),
                            ..Default::default()
                        },
                    )),
                }),
            ),
        )
        .await
        .unwrap();
        timeout(Duration::from_secs(2), async {
            loop {
                if let Some(wire::frame::Payload::AdapterRequest(request)) =
                    read_frame(&mut client).await.unwrap().payload
                {
                    assert_eq!(request.operation, "discover_interface");
                    break;
                }
            }
        })
        .await
        .expect("The world-model command did not reach the native adapter");
        for sequence in 4..=20 {
            write_frame(
                &mut client,
                &frame(
                    sequence,
                    wire::frame::Payload::UiCommand(wire::UiCommand {
                        request_id: Uuid::new_v4().to_string(),
                        command: Some(wire::ui_command::Command::GetState(wire::GetState {
                            include_completed_tasks: true,
                        })),
                    }),
                ),
            )
            .await
            .unwrap();
        }
        let start = std::time::Instant::now();
        write_frame(
            &mut client,
            &frame(
                21,
                wire::frame::Payload::UiCommand(wire::UiCommand {
                    request_id: Uuid::new_v4().to_string(),
                    command: Some(wire::ui_command::Command::ControlTask(wire::ControlTask {
                        task_id: task_id.to_string(),
                        operation: wire::control_task::Operation::Cancel as i32,
                    })),
                }),
            ),
        )
        .await
        .unwrap();
        write_frame(
            &mut client,
            &frame(22, wire::frame::Payload::Ping(wire::Ping { value: 42 })),
        )
        .await
        .unwrap();
        timeout(Duration::from_millis(500), async {
            let mut overloaded = false;
            loop {
                match read_frame(&mut client).await.unwrap().payload {
                    Some(wire::frame::Payload::CoreEvent(event)) => {
                        if let Some(wire::core_event::Event::Error(error)) = event.event {
                            assert_eq!(error.code, "ipc_command_retryable");
                            overloaded = true;
                        }
                    }
                    Some(wire::frame::Payload::Pong(pong)) => {
                        assert_eq!(pong.value, 42);
                        assert!(overloaded);
                        break;
                    }
                    _ => {}
                }
            }
            loop {
                if core
                    .snapshot(true)
                    .await
                    .tasks
                    .iter()
                    .any(|task| task.id == task_id && task.status == TaskStatus::Cancelled)
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("Control waited behind the stalled provider command");
        eprintln!(
            "Fixture Stop committed and Ping replied in {:?} while the ordinary lane remained stalled",
            start.elapsed()
        );
        drop(client);
        assert!(
            timeout(Duration::from_secs(2), session)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
    }

    async fn handshake(
        client: &mut tokio::io::DuplexStream,
        kind: wire::ClientKind,
        key: &[u8],
    ) -> wire::AuthenticationResult {
        let challenge = read_frame(client).await.unwrap();
        let Some(wire::frame::Payload::ServerChallenge(challenge)) = challenge.payload else {
            panic!("challenge expected")
        };
        let supported_features = vec![WORLD_MODEL_FEATURE.to_owned()];
        let proof = crate::ipc::authentication_proof_with_features(
            key,
            &challenge.nonce,
            &[2; 32],
            PROTOCOL_VERSION,
            kind as i32,
            "test",
            &supported_features,
        )
        .unwrap();
        write_frame(
            client,
            &frame(
                1,
                wire::frame::Payload::ClientAuthenticate(wire::ClientAuthenticate {
                    client_kind: kind as i32,
                    client_version: "test".into(),
                    client_nonce: vec![2; 32],
                    proof: proof.to_vec(),
                    supported_features,
                }),
            ),
        )
        .await
        .unwrap();
        let Some(wire::frame::Payload::AuthenticationResult(result)) =
            read_frame(client).await.unwrap().payload
        else {
            panic!("authentication expected")
        };
        assert_eq!(
            result.server_proof,
            crate::ipc::server_authentication_proof_with_features(
                key,
                &proof,
                &result.session_id,
                &result.negotiated_features,
            )
            .unwrap()
        );
        result
    }
    #[tokio::test]
    async fn worker_cancel_ack_and_late_reply_leave_the_authenticated_connection_usable() {
        let data = tempfile::tempdir().unwrap();
        let core = SageCore::new(
            crate::config::CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let auth = Arc::new(IpcAuthenticator::new(SecretBytes::new(vec![7; 32])));
        let (mut client, server) = tokio::io::duplex(8192);
        let session = tokio::spawn(handle_connection(server, core.clone(), auth));
        let kind = if cfg!(windows) {
            wire::ClientKind::Windows
        } else {
            wire::ClientKind::Macos
        };
        timeout(Duration::from_secs(2), async {
            handshake(&mut client, kind, &[7; 32]).await;
            write_frame(
                &mut client,
                &frame(
                    2,
                    wire::frame::Payload::AdapterHello(wire::AdapterHello {
                        domain: "native".into(),
                        cancellation_protocol: 1,
                    }),
                ),
            )
            .await
            .unwrap();
            while !core.adapters.available("native").await {
                tokio::task::yield_now().await;
            }
            let caller = core.clone();
            let operation = tokio::spawn(async move {
                caller
                    .adapters
                    .request("native", "reference", serde_json::json!({}))
                    .await
            });
            let Some(wire::frame::Payload::AdapterRequest(request)) =
                read_frame(&mut client).await.unwrap().payload
            else {
                panic!("Expected adapter request");
            };
            operation.abort();
            assert!(operation.await.unwrap_err().is_cancelled());
            let Some(wire::frame::Payload::AdapterCancel(cancel)) =
                read_frame(&mut client).await.unwrap().payload
            else {
                panic!("Expected priority cancellation");
            };
            assert_eq!(cancel.request_id, request.request_id);
            write_frame(
                &mut client,
                &frame(
                    3,
                    wire::frame::Payload::AdapterCancelAcknowledged(
                        wire::AdapterCancelAcknowledged {
                            request_id: request.request_id.clone(),
                        },
                    ),
                ),
            )
            .await
            .unwrap();
            write_frame(
                &mut client,
                &frame(
                    4,
                    wire::frame::Payload::AdapterResult(wire::AdapterResult {
                        request_id: request.request_id,
                        success: true,
                        json: "{}".into(),
                        ..Default::default()
                    }),
                ),
            )
            .await
            .unwrap();
            write_frame(
                &mut client,
                &frame(5, wire::frame::Payload::Ping(wire::Ping { value: 91 })),
            )
            .await
            .unwrap();
            assert!(matches!(
                read_frame(&mut client).await.unwrap().payload,
                Some(wire::frame::Payload::Pong(wire::Pong { value: 91 }))
            ));
        })
        .await
        .expect("Worker control or late result stalled the session");
        drop(client);
        assert!(
            timeout(Duration::from_secs(2), session)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
    }
    #[tokio::test]
    async fn authenticated_browser_cannot_send_ui_commands() {
        let data = tempfile::tempdir().unwrap();
        let core = SageCore::new(
            crate::config::CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let root = SecretBytes::new(vec![7; 32]);
        let key = super::super::auth::derive_browser_secret(&root);
        let auth = Arc::new(IpcAuthenticator::new(root));
        let (mut client, server) = tokio::io::duplex(8192);
        let task = tokio::spawn(handle_connection(server, core.clone(), auth));
        handshake(&mut client, wire::ClientKind::Browser, key.expose()).await;
        write_frame(
            &mut client,
            &frame(
                2,
                wire::frame::Payload::UiCommand(wire::UiCommand {
                    request_id: Uuid::new_v4().to_string(),
                    command: Some(wire::ui_command::Command::SubmitTask(wire::SubmitTask {
                        text: "unauthorized".into(),
                        ..Default::default()
                    })),
                }),
            ),
        )
        .await
        .unwrap();
        assert!(
            timeout(Duration::from_secs(2), task)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
        assert!(core.snapshot(true).await.tasks.is_empty());
    }
    #[tokio::test]
    async fn lost_acceptance_receipt_can_be_retried_in_a_new_session() {
        let data = tempfile::tempdir().unwrap();
        let core = SageCore::new(
            crate::config::CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let auth = Arc::new(IpcAuthenticator::new(SecretBytes::new(vec![7; 32])));
        let kind = if cfg!(windows) {
            wire::ClientKind::Windows
        } else {
            wire::ClientKind::Macos
        };
        let command = wire::UiCommand {
            request_id: Uuid::new_v4().to_string(),
            command: Some(wire::ui_command::Command::SubmitTask(wire::SubmitTask {
                text: "One accepted task".into(),
                ..Default::default()
            })),
        };
        let (mut first, server) = tokio::io::duplex(8192);
        let first_session = tokio::spawn(handle_connection(server, core.clone(), auth.clone()));
        handshake(&mut first, kind, &[7; 32]).await;
        write_frame(
            &mut first,
            &frame(2, wire::frame::Payload::UiCommand(command.clone())),
        )
        .await
        .unwrap();
        let original = timeout(Duration::from_secs(2), async {
            loop {
                if let Some(task) = core.snapshot(true).await.tasks.first() {
                    break task.id;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        // Acceptance reached storage, but the client never consumed the reply.
        drop(first);
        assert!(
            timeout(Duration::from_secs(2), first_session)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );

        let (mut second, server) = tokio::io::duplex(8192);
        let second_session = tokio::spawn(handle_connection(server, core.clone(), auth));
        handshake(&mut second, kind, &[7; 32]).await;
        for sequence in [2, 3] {
            write_frame(
                &mut second,
                &frame(sequence, wire::frame::Payload::UiCommand(command.clone())),
            )
            .await
            .unwrap();
            let receipt = timeout(Duration::from_secs(2), async {
                loop {
                    if let Some(wire::frame::Payload::CoreEvent(event)) =
                        read_frame(&mut second).await.unwrap().payload
                        && let Some(wire::core_event::Event::TaskAccepted(receipt)) = event.event
                    {
                        break receipt;
                    }
                }
            })
            .await
            .unwrap();
            assert_eq!(receipt.request_id, command.request_id);
            assert_eq!(receipt.task_id, original.to_string());
        }
        assert_eq!(core.snapshot(true).await.tasks.len(), 1);
        let mut conflicting = command;
        if let Some(wire::ui_command::Command::SubmitTask(ref mut request)) = conflicting.command {
            request.text = "Different effect".into();
        }
        write_frame(
            &mut second,
            &frame(4, wire::frame::Payload::UiCommand(conflicting)),
        )
        .await
        .unwrap();
        timeout(Duration::from_secs(2), async {
            loop {
                if let Some(wire::frame::Payload::CoreEvent(event)) =
                    read_frame(&mut second).await.unwrap().payload
                    && let Some(wire::core_event::Event::Error(error)) = event.event
                {
                    assert!(error.message.contains("different content"));
                    break;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(core.snapshot(true).await.tasks.len(), 1);
        drop(second);
        assert!(
            timeout(Duration::from_secs(2), second_session)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
    }

    #[tokio::test]
    async fn replayed_frame_is_rejected_after_a_valid_native_handshake() {
        let data = tempfile::tempdir().unwrap();
        let core = SageCore::new(
            crate::config::CoreConfig::for_test(data.path()),
            Arc::new(UnconfiguredModelProvider),
        )
        .unwrap();
        let auth = Arc::new(IpcAuthenticator::new(SecretBytes::new(vec![7; 32])));
        let (mut client, server) = tokio::io::duplex(8192);
        let task = tokio::spawn(handle_connection(server, core, auth));
        let kind = if cfg!(windows) {
            wire::ClientKind::Windows
        } else {
            wire::ClientKind::Macos
        };
        handshake(&mut client, kind, &[7; 32]).await;
        let ping = frame(2, wire::frame::Payload::Ping(wire::Ping { value: 19 }));
        write_frame(&mut client, &ping).await.unwrap();
        assert!(matches!(
            read_frame(&mut client).await.unwrap().payload,
            Some(wire::frame::Payload::Pong(_))
        ));
        write_frame(&mut client, &ping).await.unwrap();
        assert!(
            timeout(Duration::from_secs(2), task)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
    }
}
