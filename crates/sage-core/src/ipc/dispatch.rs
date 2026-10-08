//! Bounded command lanes keep the authenticated reader available while ordinary
//! commands await models or services. Control does not queue behind that work.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use sage_protocol::sage::ipc::v2 as wire;
use tokio::sync::{mpsc, watch};

use crate::{CoreError, CoreResult, SageCore};

pub(crate) struct CommandDispatcher {
    core: Arc<SageCore>,
    regular: mpsc::Sender<wire::UiCommand>,
    control: mpsc::Sender<wire::UiCommand>,
    preparation: watch::Sender<Option<wire::UiCommand>>,
    connected: Arc<AtomicBool>,
    intent_revisions: std::sync::Mutex<std::collections::HashMap<uuid::Uuid, u64>>,
    streamed_prefixes: Arc<std::sync::Mutex<std::collections::HashSet<uuid::Uuid>>>,
}

impl CommandDispatcher {
    pub fn new(core: Arc<SageCore>, responses: mpsc::Sender<wire::frame::Payload>) -> Self {
        let (regular, regular_rx) = mpsc::channel(16);
        let (control, control_rx) = mpsc::channel(8);
        let (preparation, preparation_rx) = watch::channel(None);
        let connected = Arc::new(AtomicBool::new(true));
        let streamed_prefixes = Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        tokio::spawn(run_lane(
            core.clone(),
            connected.clone(),
            regular_rx,
            responses.clone(),
            streamed_prefixes.clone(),
        ));
        tokio::spawn(run_lane(
            core.clone(),
            connected.clone(),
            control_rx,
            responses.clone(),
            streamed_prefixes.clone(),
        ));
        tokio::spawn(run_preparation(
            core.clone(),
            connected.clone(),
            preparation_rx,
            responses,
        ));
        Self {
            core,
            regular,
            control,
            preparation,
            connected,
            intent_revisions: Default::default(),
            streamed_prefixes,
        }
    }

    pub fn enqueue(&self, command: wire::UiCommand) -> CoreResult<()> {
        let mut interrupted = false;
        let mut intent_control = false;
        // Typed corrections use the priority lane. Receipt lookup and Hold
        // happen there, never under the authenticated frame reader: a database
        // lock must not delay admission of a subsequent Stop. Voice reflexes
        // below still interrupt synchronously before the sentence is complete.
        let correction = matches!(command.command.as_ref(), Some(wire::ui_command::Command::SubmitTask(input)) if !input.supersedes_task_id.is_empty());
        let streamed_submit = matches!(
            command.command.as_ref(),
            Some(wire::ui_command::Command::SubmitTask(input))
                if input.streamed_prefix || input.finalize_stream
        );
        if let Some(wire::ui_command::Command::UpdateIntent(input)) = command.command.as_ref() {
            let stream = uuid::Uuid::parse_str(&input.stream_id)
                .map_err(|_| CoreError::Protocol("Invalid intent stream".into()))?;
            if input.revision == 0
                || input.text.len() > crate::intent::MAX_INTENT_BYTES
                || input.folder_roots.len() > 32
                || input.folder_roots.iter().any(|p| p.len() > 4096)
            {
                return Err(CoreError::Protocol(
                    "Intent input exceeds its bounds".into(),
                ));
            }
            let mut revisions = self
                .intent_revisions
                .lock()
                .expect("intent revisions poisoned");
            if revisions
                .get(&stream)
                .is_some_and(|revision| input.revision <= *revision)
            {
                return Err(CoreError::Protocol("Stale intent revision".into()));
            }
            if !revisions.contains_key(&stream) && revisions.len() >= 16 {
                return Err(CoreError::Busy(
                    "This connection has reached its intent stream limit".into(),
                ));
            }
            revisions.insert(stream, input.revision);
            if !self.core.stream_prefix_is_current(stream, &input.text) {
                self.core.hold_stream_for_revision(stream);
            }
            if input.allow_interrupt
                && let Some(reflex) = crate::intent::reflex(&input.text)
            {
                let status = if reflex == crate::intent::Reflex::Stop {
                    crate::domain::TaskStatus::Cancelled
                } else {
                    crate::domain::TaskStatus::Paused
                };
                intent_control = true;
                interrupted = match reflex {
                    crate::intent::Reflex::Stop => self.core.signal_stream_control(stream, status),
                    crate::intent::Reflex::Hold => self.core.signal_stream_control(stream, status),
                    crate::intent::Reflex::Correct => self.core.hold_stream_for_revision(stream),
                };
                if !interrupted && !input.active_task_id.is_empty() {
                    let task = uuid::Uuid::parse_str(&input.active_task_id)
                        .map_err(|_| CoreError::Protocol("Invalid active task".into()))?;
                    interrupted = match reflex {
                        crate::intent::Reflex::Stop => self.core.signal_stop(task),
                        _ => self.core.signal_hold(task),
                    };
                }
            }
            if !intent_control {
                if self.preparation.is_closed() {
                    return Err(CoreError::Busy("Intent preparation disconnected".into()));
                }
                // Only the latest draft matters. Slow identity work and stale
                // queued drafts cannot delay a new transcript or typed edit.
                self.preparation.send_replace(Some(command));
                return Ok(());
            }
            // A reflex also retires speculative work, but its durable control
            // command must never be dropped by a subsequent draft revision.
            self.preparation.send_replace(None);
        }
        // Only the authenticated native-UI session can reach this dispatcher.
        // Signal Stop at admission, even when a control command is awaiting
        // storage or its response queue is full. Saving Stop still uses the lane.
        let stop_signalled = match command.command.as_ref() {
            Some(wire::ui_command::Command::ControlTask(control))
                if control.operation == wire::control_task::Operation::Cancel as i32 =>
            {
                uuid::Uuid::parse_str(&control.task_id).is_ok_and(|id| self.core.signal_stop(id))
            }
            Some(wire::ui_command::Command::ControlTask(control))
                if control.operation == wire::control_task::Operation::Pause as i32 =>
            {
                uuid::Uuid::parse_str(&control.task_id).is_ok_and(|id| self.core.signal_hold(id))
            }
            _ => false,
        };
        let sender = if intent_control
            || correction
            || streamed_submit
            || matches!(
                command.command,
                Some(
                    wire::ui_command::Command::ControlTask(_)
                        | wire::ui_command::Command::ApprovalResponse(_)
                        | wire::ui_command::Command::UserAnswer(_)
                )
            ) {
            &self.control
        } else {
            &self.regular
        };
        let streamed_prefix = match command.command.as_ref() {
            Some(wire::ui_command::Command::SubmitTask(input))
                if input.streamed_prefix || input.finalize_stream =>
            {
                Some(
                    uuid::Uuid::parse_str(&input.voice_stream_id)
                        .map_err(|_| CoreError::Protocol("Invalid voice stream identity".into()))?,
                )
            }
            _ => None,
        };
        if let Some(stream_id) = streamed_prefix {
            self.streamed_prefixes
                .lock()
                .expect("streamed prefixes poisoned")
                .insert(stream_id);
        }
        if sender.try_send(command).is_err() {
            if let Some(stream_id) = streamed_prefix {
                self.streamed_prefixes
                    .lock()
                    .expect("streamed prefixes poisoned")
                    .remove(&stream_id);
            }
            return Err(CoreError::Busy(
                if stop_signalled || interrupted {
                    "Interruption was signalled, but its save request could not be queued. Retry the control to confirm its saved state."
                } else {
                    "The command queue is full or disconnected. Retry after pending requests finish."
                }.into(),
            ));
        }
        Ok(())
    }
}

async fn run_preparation(
    core: Arc<SageCore>,
    connected: Arc<AtomicBool>,
    mut commands: watch::Receiver<Option<wire::UiCommand>>,
    responses: mpsc::Sender<wire::frame::Payload>,
) {
    let mut pending = None;
    loop {
        if pending.is_none() {
            if commands.changed().await.is_err() {
                return;
            }
            pending = commands.borrow_and_update().clone();
        }
        if !connected.load(Ordering::Acquire) {
            return;
        }
        let Some(command) = pending.take() else {
            continue;
        };
        let work = async {
            let Some(wire::ui_command::Command::UpdateIntent(input)) = command.command.clone()
            else {
                return;
            };
            let event = match super::server::handle_command(&core, command.clone()).await {
                Ok(Some(event)) => event,
                Ok(None) => return,
                Err(error) => super::server::error_event(command.request_id, error),
            };
            // Previews are best effort. Backpressure must not hold a worker or
            // delay a control response; the UI discards older stream revisions.
            let _ = responses.try_send(wire::frame::Payload::CoreEvent(event.clone()));
            if let Some(event) = super::server::prepare_preview(&core, &input, event).await {
                let _ = responses.try_send(wire::frame::Payload::CoreEvent(event));
            }
        };
        tokio::select! {
            biased;
            changed = commands.changed() => {
                if changed.is_err() { return; }
                pending = commands.borrow_and_update().clone();
            }
            _ = work => {}
        }
        // Replacing or closing the watch drops the request future. The adapter
        // bridge then sends cancellation and retires any late native reply.
    }
}

impl Drop for CommandDispatcher {
    fn drop(&mut self) {
        self.connected.store(false, Ordering::Release);
        let streams = self
            .streamed_prefixes
            .lock()
            .expect("streamed prefixes poisoned")
            .drain()
            .collect::<Vec<_>>();
        for stream_id in streams {
            self.core
                .signal_stream_control(stream_id, crate::domain::TaskStatus::Cancelled);
        }
    }
}

async fn run_lane(
    core: Arc<SageCore>,
    connected: Arc<AtomicBool>,
    mut commands: mpsc::Receiver<wire::UiCommand>,
    responses: mpsc::Sender<wire::frame::Payload>,
    streamed_prefixes: Arc<std::sync::Mutex<std::collections::HashSet<uuid::Uuid>>>,
) {
    while let Some(command) = commands.recv().await {
        if !connected.load(Ordering::Acquire) {
            break;
        }
        let request_id = command.request_id.clone();
        let finalized_stream = match command.command.as_ref() {
            Some(wire::ui_command::Command::SubmitTask(input)) if input.finalize_stream => {
                uuid::Uuid::parse_str(&input.voice_stream_id).ok()
            }
            _ => None,
        };
        let completed_stream = command.command.as_ref().and_then(|command| match command {
            wire::ui_command::Command::UpdateIntent(input)
                if input.allow_interrupt
                    && matches!(
                        crate::intent::reflex(&input.text),
                        Some(crate::intent::Reflex::Stop | crate::intent::Reflex::Hold)
                    ) =>
            {
                uuid::Uuid::parse_str(&input.stream_id).ok()
            }
            _ => None,
        });
        // Once started, let the command finish its durable transition even if
        // the connection closes. Unstarted queued commands are discarded.
        let event = match super::server::handle_command(&core, command).await {
            Ok(event) => event,
            Err(error) => Some(super::server::error_event(request_id, error)),
        };
        if let Some(stream_id) = finalized_stream
            && !matches!(
                event.as_ref().and_then(|event| event.event.as_ref()),
                Some(wire::core_event::Event::Error(_))
            )
        {
            streamed_prefixes
                .lock()
                .expect("streamed prefixes poisoned")
                .remove(&stream_id);
        }
        if let Some(stream_id) = completed_stream {
            streamed_prefixes
                .lock()
                .expect("streamed prefixes poisoned")
                .remove(&stream_id);
        }
        if let Some(event) = event
            && responses
                .send(wire::frame::Payload::CoreEvent(event))
                .await
                .is_err()
        {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{ActionGraph, TaskStatus};
    use crate::model::{
        ModelProvider, ModelTurn, PlanningContext, ProviderDescriptor, ReplanContext, TurnContext,
        UnconfiguredModelProvider,
    };
    use tokio::sync::Notify;
    use tokio::time::{Duration, timeout};
    use uuid::Uuid;

    struct StalledModel {
        entered: Notify,
        dropped: Arc<Notify>,
    }
    struct Dropped(Arc<Notify>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.notify_one();
        }
    }
    #[async_trait::async_trait]
    impl ModelProvider for StalledModel {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, _: TurnContext) -> CoreResult<ModelTurn> {
            let _guard = Dropped(self.dropped.clone());
            self.entered.notify_one();
            std::future::pending().await
        }
    }
    fn stop(task_id: String) -> wire::UiCommand {
        wire::UiCommand {
            request_id: Uuid::new_v4().to_string(),
            command: Some(wire::ui_command::Command::ControlTask(wire::ControlTask {
                task_id,
                operation: wire::control_task::Operation::Cancel as i32,
            })),
        }
    }
    #[tokio::test]
    async fn stop_signals_even_when_control_commands_and_responses_are_saturated() {
        let data = tempfile::tempdir().unwrap();
        let dropped = Arc::new(Notify::new());
        let provider = Arc::new(StalledModel {
            entered: Notify::new(),
            dropped: dropped.clone(),
        });
        let core = SageCore::new_with_secret_store(
            crate::config::CoreConfig::for_test(data.path()),
            provider.clone(),
            Arc::new(crate::secrets::testing::MemorySecretStore::default()),
        )
        .unwrap();
        let id = core
            .submit_task("Wait for the fixture provider")
            .await
            .unwrap();
        timeout(Duration::from_secs(5), provider.entered.notified())
            .await
            .unwrap();
        let (responses, receiver) = mpsc::channel(1);
        let dispatcher = CommandDispatcher::new(core.clone(), responses);
        // One error fills the response queue; the next blocks its consumer lane.
        dispatcher.enqueue(stop("invalid-task".into())).unwrap();
        timeout(Duration::from_secs(1), async {
            while receiver.len() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        dispatcher.enqueue(stop("invalid-task".into())).unwrap();
        timeout(Duration::from_secs(1), async {
            while dispatcher.control.capacity() != 8 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        for _ in 0..8 {
            dispatcher.enqueue(stop("invalid-task".into())).unwrap();
        }
        let error = dispatcher.enqueue(stop(id.to_string())).unwrap_err();
        assert!(matches!(error, CoreError::Busy(_)));
        assert!(error.to_string().contains("Interruption was signalled"));
        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .unwrap();
        assert_eq!(
            core.snapshot(true)
                .await
                .tasks
                .iter()
                .find(|task| task.id == id)
                .unwrap()
                .status,
            TaskStatus::Cancelled
        );
        drop(dispatcher);
        drop(receiver);
    }

    #[tokio::test]
    async fn voice_reflex_holds_at_admission_and_rejects_stale_transcripts() {
        let data = tempfile::tempdir().unwrap();
        let provider = Arc::new(StalledModel {
            entered: Notify::new(),
            dropped: Arc::new(Notify::new()),
        });
        let core = SageCore::new(
            crate::config::CoreConfig::for_test(data.path()),
            provider.clone(),
        )
        .unwrap();
        let id = core.submit_task("Fixture reasoning").await.unwrap();
        timeout(Duration::from_secs(3), provider.entered.notified())
            .await
            .unwrap();
        let (responses, mut receiver) = mpsc::channel(16);
        let dispatcher = CommandDispatcher::new(core.clone(), responses);
        let stream_id = Uuid::new_v4().to_string();
        let input = |revision, text: &str| wire::UiCommand {
            request_id: Uuid::new_v4().to_string(),
            command: Some(wire::ui_command::Command::UpdateIntent(
                wire::UpdateIntent {
                    stream_id: stream_id.clone(),
                    revision,
                    text: text.into(),
                    active_task_id: id.to_string(),
                    allow_interrupt: true,
                    folder_roots: vec![],
                    voice_input: false,
                },
            )),
        };
        // Snapshot projects the admission signal even before its durable save.
        dispatcher.enqueue(input(2, "No, open Firefox")).unwrap();
        dispatcher
            .enqueue(input(3, "open Firefox and then"))
            .unwrap();
        assert_eq!(
            core.snapshot(true)
                .await
                .tasks
                .iter()
                .find(|t| t.id == id)
                .unwrap()
                .status,
            TaskStatus::Paused
        );
        assert!(dispatcher.enqueue(input(1, "Stop")).is_err());
        assert_ne!(
            core.snapshot(true)
                .await
                .tasks
                .iter()
                .find(|t| t.id == id)
                .unwrap()
                .status,
            TaskStatus::Cancelled
        );
        // A newer ordinary draft cannot replace the durable Hold response.
        timeout(Duration::from_secs(2), async {
            loop {
                if let Some(wire::frame::Payload::CoreEvent(event)) = receiver.recv().await
                    && let Some(wire::core_event::Event::IntentPreview(preview)) = event.event
                    && preview.revision == 2
                {
                    assert_eq!(preview.status, "paused");
                    break;
                }
            }
        })
        .await
        .unwrap();
        dispatcher.enqueue(input(4, "Stop")).unwrap();
        dispatcher.enqueue(input(5, "open Notes")).unwrap();
        assert_eq!(
            core.snapshot(true)
                .await
                .tasks
                .iter()
                .find(|t| t.id == id)
                .unwrap()
                .status,
            TaskStatus::Cancelled
        );
        timeout(Duration::from_secs(2), async {
            loop {
                if let Some(wire::frame::Payload::CoreEvent(event)) = receiver.recv().await
                    && let Some(wire::core_event::Event::IntentPreview(preview)) = event.event
                    && preview.revision == 4
                {
                    assert_eq!(preview.status, "stopping");
                    break;
                }
            }
        })
        .await
        .unwrap();
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn changing_and_closing_drafts_cancel_native_preparation_without_launching() {
        use crate::preparation::tests::{core, native, reply, target};
        let (_data, core) = core();
        let (mut requests, mut cancelled) = native(&core, "preview-session").await;
        let (responses, mut receiver) = mpsc::channel(4);
        let dispatcher = CommandDispatcher::new(core.clone(), responses);
        let stream = Uuid::new_v4().to_string();
        let input = |revision, text: &str| wire::UiCommand {
            request_id: Uuid::new_v4().to_string(),
            command: Some(wire::ui_command::Command::UpdateIntent(
                wire::UpdateIntent {
                    stream_id: stream.clone(),
                    revision,
                    text: text.into(),
                    ..Default::default()
                },
            )),
        };
        dispatcher
            .enqueue(input(1, "open Notes and then read"))
            .unwrap();
        let first = timeout(Duration::from_secs(2), requests.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first.operation, "application_identity");
        assert!(first.grant.is_none());
        // Recognition arrives before the slow native observation.
        let wire::frame::Payload::CoreEvent(event) = receiver.recv().await.unwrap() else {
            panic!("expected event")
        };
        let Some(wire::core_event::Event::IntentPreview(preview)) = event.event else {
            panic!("expected preview")
        };
        assert_eq!(preview.status, "preparing");
        assert_eq!(preview.steps, ["Open Notes"]);
        assert!(!preview.compiled_locally);

        dispatcher.enqueue(input(2, "open Firefox")).unwrap();
        assert_eq!(
            timeout(Duration::from_secs(2), cancelled.recv())
                .await
                .unwrap()
                .unwrap()
                .request_id,
            first.request_id
        );
        let second = timeout(Duration::from_secs(2), requests.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second.operation, "application_identity");
        assert!(second.grant.is_none());
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&second.json).unwrap()["application"],
            "Firefox"
        );
        let mut firefox = target();
        firefox.identifier = "org.mozilla.firefox".into();
        firefox.bundle_path = "/Applications/Firefox.app".into();
        reply(&core, "preview-session", &second, firefox, 0).await;
        timeout(Duration::from_secs(2), async {
            loop {
                let wire::frame::Payload::CoreEvent(event) = receiver.recv().await.unwrap() else {
                    panic!("expected event")
                };
                let Some(wire::core_event::Event::IntentPreview(preview)) = event.event else {
                    panic!("expected preview")
                };
                assert_eq!(preview.revision, 2);
                if preview.status == "prepared" {
                    break;
                }
            }
        })
        .await
        .unwrap();
        // A late response from the cancelled request never populates the cache.
        reply(&core, "preview-session", &first, target(), 0).await;
        dispatcher.enqueue(input(3, "open Notes")).unwrap();
        let third = timeout(Duration::from_secs(2), requests.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(third.operation, "application_identity");
        assert!(third.grant.is_none());
        dispatcher.enqueue(input(4, "")).unwrap();
        assert_eq!(
            timeout(Duration::from_secs(2), cancelled.recv())
                .await
                .unwrap()
                .unwrap()
                .request_id,
            third.request_id
        );
        dispatcher.enqueue(input(5, "open Notes")).unwrap();
        let fourth = timeout(Duration::from_secs(2), requests.recv())
            .await
            .unwrap()
            .unwrap();
        drop(dispatcher);
        assert_eq!(
            timeout(Duration::from_secs(2), cancelled.recv())
                .await
                .unwrap()
                .unwrap()
                .request_id,
            fourth.request_id
        );
        assert!(core.snapshot(true).await.tasks.is_empty());
    }

    #[tokio::test]
    async fn draft_bursts_keep_only_latest_revision() {
        let (_data, core) = crate::preparation::tests::core();
        let (responses, mut receiver) = mpsc::channel(1);
        let dispatcher = CommandDispatcher::new(core.clone(), responses);
        let stream = Uuid::new_v4().to_string();
        let input = |revision| wire::UiCommand {
            request_id: Uuid::new_v4().to_string(),
            command: Some(wire::ui_command::Command::UpdateIntent(
                wire::UpdateIntent {
                    stream_id: stream.clone(),
                    revision,
                    text: format!("read \"/tmp/draft-{revision}\""),
                    ..Default::default()
                },
            )),
        };
        for revision in 1..=512 {
            dispatcher.enqueue(input(revision)).unwrap();
        }
        let event = timeout(Duration::from_secs(2), receiver.recv())
            .await
            .unwrap()
            .unwrap();
        let wire::frame::Payload::CoreEvent(event) = event else {
            panic!("expected event")
        };
        let Some(wire::core_event::Event::IntentPreview(preview)) = event.event else {
            panic!("expected preview")
        };
        assert_eq!(preview.revision, 512);
        assert!(core.snapshot(true).await.tasks.is_empty());
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn prepared_single_app_prefix_is_explicitly_marked_for_streaming() {
        use crate::preparation::tests::{core, native, reply, target};
        let (_data, core) = core();
        let (mut requests, _cancelled) = native(&core, "stream-preview-session").await;
        let (responses, mut receiver) = mpsc::channel(4);
        let dispatcher = CommandDispatcher::new(core.clone(), responses);
        let stream_id = Uuid::new_v4().to_string();
        for (revision, voice_input, expected_prefix) in [(1, false, ""), (2, true, "Open Notes")] {
            dispatcher
                .enqueue(wire::UiCommand {
                    request_id: Uuid::new_v4().to_string(),
                    command: Some(wire::ui_command::Command::UpdateIntent(
                        wire::UpdateIntent {
                            stream_id: stream_id.clone(),
                            revision,
                            text: "Open Notes and then".into(),
                            voice_input,
                            ..Default::default()
                        },
                    )),
                })
                .unwrap();
            if revision == 1 {
                let request = timeout(Duration::from_secs(2), requests.recv())
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(request.operation, "application_identity");
                assert!(request.grant.is_none());
                reply(&core, "stream-preview-session", &request, target(), 0).await;
            }
            let preview = timeout(Duration::from_secs(2), async {
                loop {
                    let wire::frame::Payload::CoreEvent(event) = receiver.recv().await.unwrap()
                    else {
                        panic!("expected preview")
                    };
                    if let Some(wire::core_event::Event::IntentPreview(preview)) = event.event
                        && preview.revision == revision
                        && preview.status == "prepared"
                    {
                        break preview;
                    }
                }
            })
            .await
            .unwrap();
            assert_eq!(preview.streamed_prefix, expected_prefix);
            assert_eq!(preview.steps, ["Open Notes"]);
        }
        assert!(core.snapshot(true).await.tasks.is_empty());
        assert!(matches!(
            requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn prepared_file_prefix_requires_a_selected_read_folder_and_never_reads_early() {
        use crate::preparation::tests::core;

        let (_data, core) = core();
        let workspace = tempfile::tempdir().unwrap();
        let root = workspace.path().canonicalize().unwrap();
        let file = root.join("notes.txt");
        std::fs::write(&file, "fixture content").unwrap();
        let (responses, mut receiver) = mpsc::channel(8);
        let dispatcher = CommandDispatcher::new(core.clone(), responses);
        let stream_id = Uuid::new_v4().to_string();
        dispatcher
            .enqueue(wire::UiCommand {
                request_id: Uuid::new_v4().to_string(),
                command: Some(wire::ui_command::Command::UpdateIntent(
                    wire::UpdateIntent {
                        stream_id: stream_id.clone(),
                        revision: 1,
                        text: format!("Read \"{}\" and then", file.display()),
                        folder_roots: vec![root.display().to_string()],
                        voice_input: true,
                        ..Default::default()
                    },
                )),
            })
            .unwrap();
        let prepared = timeout(Duration::from_secs(2), async {
            loop {
                let wire::frame::Payload::CoreEvent(event) = receiver.recv().await.unwrap() else {
                    panic!("expected prepared read preview")
                };
                if let Some(wire::core_event::Event::IntentPreview(preview)) = event.event
                    && preview.revision == 1
                    && preview.status == "prepared"
                {
                    break preview;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(prepared.steps, [format!("Read {}", file.display())]);
        assert_eq!(
            prepared.streamed_prefix,
            format!("Read \"{}\"", file.display())
        );
        assert!(prepared.detail.contains("selected read folder"));
        assert!(!prepared.detail.contains("fixture content"));
        assert!(core.snapshot(true).await.tasks.is_empty());

        dispatcher
            .enqueue(wire::UiCommand {
                request_id: Uuid::new_v4().to_string(),
                command: Some(wire::ui_command::Command::UpdateIntent(
                    wire::UpdateIntent {
                        stream_id,
                        revision: 2,
                        text: format!("Read \"{}\" and then", file.display()),
                        voice_input: true,
                        ..Default::default()
                    },
                )),
            })
            .unwrap();
        let denied = timeout(Duration::from_secs(2), async {
            loop {
                let wire::frame::Payload::CoreEvent(event) = receiver.recv().await.unwrap() else {
                    panic!("expected unscoped read preview")
                };
                if let Some(wire::core_event::Event::IntentPreview(preview)) = event.event
                    && preview.revision == 2
                    && preview.status == "unavailable"
                {
                    break preview;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(denied.status, "unavailable");
        assert!(denied.streamed_prefix.is_empty());
        assert!(denied.detail.contains("selected read folder"));
        assert!(core.snapshot(true).await.tasks.is_empty());
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn disconnect_stops_a_streamed_prefix_before_an_unapproved_launch() {
        use crate::events::CoreEventKind;
        use crate::preparation::tests::{core, lookup, native, target};
        let (_data, core) = core();
        let mut events = core.events().subscribe();
        let session = Uuid::new_v4().to_string();
        let (mut requests, _cancellations) = native(&core, &session).await;
        lookup(&core, &mut requests, &session, target())
            .await
            .unwrap();
        let (responses, mut receiver) = mpsc::channel(16);
        let dispatcher = CommandDispatcher::new(core.clone(), responses);
        let stream_id = Uuid::new_v4();
        let conversation_id = Uuid::new_v4();
        dispatcher
            .enqueue(wire::UiCommand {
                request_id: Uuid::new_v4().to_string(),
                command: Some(wire::ui_command::Command::SubmitTask(wire::SubmitTask {
                    text: "Open Notes".into(),
                    source: wire::InputSource::Voice as i32,
                    conversation_id: conversation_id.to_string(),
                    voice_stream_id: stream_id.to_string(),
                    streamed_prefix: true,
                    ..Default::default()
                })),
            })
            .unwrap();
        let task_id = timeout(Duration::from_secs(3), async {
            loop {
                if let Some(wire::frame::Payload::CoreEvent(event)) = receiver.recv().await
                    && let Some(wire::core_event::Event::TaskAccepted(receipt)) = event.event
                {
                    break Uuid::parse_str(&receipt.task_id).unwrap();
                }
            }
        })
        .await
        .unwrap();
        timeout(Duration::from_secs(3), async {
            loop {
                if matches!(
                    events.recv().await.unwrap().kind,
                    CoreEventKind::ApprovalRequested { .. }
                ) {
                    break;
                }
            }
        })
        .await
        .unwrap();
        drop(dispatcher);
        timeout(Duration::from_secs(3), async {
            loop {
                let task = core
                    .snapshot(true)
                    .await
                    .tasks
                    .into_iter()
                    .find(|task| task.id == task_id)
                    .unwrap();
                if task.status == TaskStatus::Cancelled {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(matches!(
            requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    #[ignore = "release-mode latency measurement"]
    async fn warm_application_preview_latency() {
        use crate::preparation::tests::{core, native, reply, target};
        let (_data, core) = core();
        let (mut requests, _cancelled) = native(&core, "latency-session").await;
        let lookup = core.clone();
        let preparing = tokio::spawn(async move { lookup.prepared_application("Notes").await });
        let request = requests.recv().await.unwrap();
        reply(&core, "latency-session", &request, target(), 0).await;
        preparing.await.unwrap().unwrap();
        let (responses, mut receiver) = mpsc::channel(4);
        let dispatcher = CommandDispatcher::new(core, responses);
        let stream = Uuid::new_v4().to_string();
        let mut samples = Vec::with_capacity(1000);
        for revision in 1..=1000 {
            let started = std::time::Instant::now();
            dispatcher
                .enqueue(wire::UiCommand {
                    request_id: Uuid::new_v4().to_string(),
                    command: Some(wire::ui_command::Command::UpdateIntent(
                        wire::UpdateIntent {
                            stream_id: stream.clone(),
                            revision,
                            text: "open Notes and then read".into(),
                            ..Default::default()
                        },
                    )),
                })
                .unwrap();
            loop {
                let wire::frame::Payload::CoreEvent(event) = receiver.recv().await.unwrap() else {
                    panic!("expected event")
                };
                let Some(wire::core_event::Event::IntentPreview(preview)) = event.event else {
                    panic!("expected preview")
                };
                assert_eq!(preview.revision, revision);
                if preview.status == "prepared" {
                    break;
                }
            }
            samples.push(started.elapsed().as_nanos());
        }
        assert!(
            requests.is_empty(),
            "warm preparation must not query the native worker again"
        );
        samples.sort_unstable();
        println!(
            "warm_application_preview samples={} p50_ns={} p95_ns={} p99_ns={}",
            samples.len(),
            samples[500],
            samples[950],
            samples[990]
        );
    }
}
