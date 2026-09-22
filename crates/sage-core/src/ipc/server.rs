use std::sync::Arc;

use chrono::Utc;
use sage_protocol::PROTOCOL_VERSION;
use sage_protocol::sage::ipc::v2 as wire;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::time::{Duration, timeout};
use uuid::Uuid;

use crate::config::IpcEndpoint;
use crate::domain::{Task, TaskStatus};
use crate::engine::{ApprovalResolution, SageCore};
use crate::error::{CoreError, CoreResult};
use crate::events::{CoreEvent, CoreEventKind};
use crate::ipc::auth::IpcAuthenticator;
use crate::ipc::codec::{read_frame, write_frame};
use crate::policy::RiskLevel;

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

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
    write_frame(
        &mut writer,
        &frame(
            sequence,
            wire::frame::Payload::ServerChallenge(wire::ServerChallenge {
                nonce: server_nonce.clone(),
                minimum_protocol_version: PROTOCOL_VERSION,
                maximum_protocol_version: PROTOCOL_VERSION,
                core_instance_id: instance_id,
            }),
        ),
    )
    .await?;

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
    authenticator.verify(
        &server_nonce,
        &client.client_nonce,
        authentication.protocol_version,
        client.client_kind,
        &client.client_version,
        &client.proof,
    )?;
    sequence += 1;
    let session_id = Uuid::new_v4().to_string();
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
                    .server_proof(client.client_kind, &client.proof, &session_id)?
                    .to_vec(),
            }),
        ),
    )
    .await?;

    let mut event_receiver = core.events().subscribe();
    let (adapter_sender, mut adapter_receiver) = tokio::sync::mpsc::channel(16);
    let is_browser = client.client_kind == wire::ClientKind::Browser as i32;
    let mut received_sequence = authentication.sequence;
    let mut request_ids = std::collections::HashSet::new();
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
            incoming = frames_receiver.recv() => {
                let incoming = incoming.ok_or_else(||CoreError::Protocol("IPC reader disconnected".into()))??;
                if incoming.sequence <= received_sequence { return Err(CoreError::Protocol("Replayed IPC frame".into())); }
                received_sequence=incoming.sequence;
                if incoming.protocol_version != PROTOCOL_VERSION {
                    return Err(CoreError::Protocol("protocol version changed within a session".into()));
                }
                match incoming.payload.clone() {
                    Some(wire::frame::Payload::UiCommand(ref command)) if command.request_id.is_empty() || !request_ids.insert(command.request_id.clone()) || request_ids.len()>65_536 => return Err(CoreError::Protocol("Replayed or excessive UI requests".into())),
                    Some(wire::frame::Payload::AdapterHello(hello))=>{core.adapters.register(&hello.domain,&session_id,client.client_kind,adapter_sender.clone()).await?;continue;},
                    Some(wire::frame::Payload::AdapterResult(response))=>{core.adapters.complete(&session_id,response).await?;continue;},
                    Some(wire::frame::Payload::Ping(_))=>{},
                    _ if is_browser=>return Err(CoreError::Protocol("Browser sessions cannot issue UI commands.".into())),
                    _=>{},
                }
                if let Some(response) = handle_frame(&core, incoming).await {
                    sequence += 1;
                    write_frame(&mut writer, &frame(sequence, response)).await?;
                }
            }
            request = adapter_receiver.recv() => {
                if let Some(request)=request {sequence+=1;write_frame(&mut writer,&frame(sequence,wire::frame::Payload::AdapterRequest(request))).await?;}
            }
            event = event_receiver.recv(), if !is_browser => {
                match event {
                    Ok(event) => {
                        sequence += 1;
                        write_frame(
                            &mut writer,
                            &frame(sequence, wire::frame::Payload::CoreEvent(event_to_wire(event))),
                        ).await?;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        sequence += 1;
                        let snapshot = core.snapshot(false).await;
                        write_frame(
                            &mut writer,
                            &frame(sequence, wire::frame::Payload::CoreEvent(snapshot_event(snapshot))),
                        ).await?;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(()),
                }
            }
        }
    }
    }.await;
    reader_task.abort();
    core.adapters.disconnect(&session_id).await;
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

async fn handle_command(
    core: &Arc<SageCore>,
    command: wire::UiCommand,
) -> CoreResult<Option<wire::CoreEvent>> {
    use wire::ui_command::Command;
    let request_id = command.request_id.clone();
    let needs_storage = match command.command.as_ref() {
        Some(Command::GetState(_)) => false,
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
        Command::SubmitTask(submit) => {
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
            core.submit_scoped(submit.text, conversation, false, None, resources)
                .await?;
            Ok(None)
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
            core.undo_last_action(parse_uuid(&undo.task_id, "task_id")?)
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
        Command::SaveProviderSettings(settings) => {
            core.save_provider_settings(
                settings.role,
                settings.provider,
                settings.model,
                settings.endpoint,
                settings.api_key,
                settings.remove_saved_key,
                settings.native_authentication_satisfied,
            )
            .await?;
            Ok(Some(snapshot_event(core.snapshot(true).await)))
        }
        Command::TestProviderConnection(settings) => {
            let provider = settings.provider.clone();
            let model = settings.model.clone();
            let result = core
                .test_provider_connection(
                    settings.role,
                    settings.provider,
                    settings.model,
                    settings.endpoint,
                    settings.api_key,
                )
                .await;
            let (success, message) = match result {
                Ok(message) => (true, message),
                Err(error) => (false, error.to_string()),
            };
            Ok(Some(provider_connection_result_event(
                request_id, success, provider, model, message,
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
                storage_locked: snapshot.storage_locked,
                core_version: snapshot.core_version,
                protocol_version: snapshot.protocol_version,
                knowledge: snapshot
                    .knowledge
                    .and_then(|knowledge| serde_json::to_string(&knowledge).ok())
                    .map(|json| wire::KnowledgeState { json }),
                provider_settings: snapshot
                    .provider_settings
                    .into_iter()
                    .map(|settings| wire::ProviderSettings {
                        role: settings.role,
                        provider: settings.provider,
                        model: settings.model,
                        endpoint: settings.endpoint,
                        has_api_key: settings.has_api_key,
                    })
                    .collect(),
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
        CoreEventKind::ApprovalRequested { .. }
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
    let completed_actions = task.completed_count() as u32;
    let total_actions = task.actions.len() as u32;
    let current_action = task
        .actions
        .values()
        .find(|action| {
            matches!(
                action.status,
                crate::domain::ActionStatus::Running
                    | crate::domain::ActionStatus::WaitingForApproval
                    | crate::domain::ActionStatus::Verifying
            )
        })
        .map(|action| action.proposal.action.redacted_summary())
        .unwrap_or_default();
    wire::TaskUpdate {
        task_id: task.id.to_string(),
        request: task.request,
        status: task_status_to_wire(task.status) as i32,
        summary: task.goal.unwrap_or_default(),
        completed_actions,
        total_actions,
        current_action,
        final_outcome: task.final_outcome.unwrap_or_default(),
        undo_available: task.rollback_available,
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
                summary: state.proposal.action.redacted_summary(),
                status: format!("{:?}", state.status).to_ascii_lowercase(),
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

fn error_event(request_id: String, error: CoreError) -> wire::CoreEvent {
    wire::CoreEvent {
        event_id: Uuid::new_v4().to_string(),
        occurred_at_unix_ms: Utc::now().timestamp_millis(),
        event: Some(wire::core_event::Event::Error(wire::ErrorEvent {
            request_id,
            task_id: String::new(),
            code: "ipc_command_failed".into(),
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
    async fn handshake(
        client: &mut tokio::io::DuplexStream,
        kind: wire::ClientKind,
        key: &[u8],
    ) -> wire::AuthenticationResult {
        let challenge = read_frame(client).await.unwrap();
        let Some(wire::frame::Payload::ServerChallenge(challenge)) = challenge.payload else {
            panic!("challenge expected")
        };
        let proof = crate::ipc::authentication_proof(
            key,
            &challenge.nonce,
            &[2; 32],
            PROTOCOL_VERSION,
            kind as i32,
            "test",
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
            crate::ipc::server_authentication_proof(key, &proof, &result.session_id).unwrap()
        );
        result
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
                    request_id: "attacker".into(),
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
