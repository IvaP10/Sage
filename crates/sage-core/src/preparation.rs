//! Ephemeral, bounded observations. This cache contains no grants, approvals,
//! handles or task state. Native execution revalidates the signed bundle.
use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

use crate::application_target::ApplicationTarget;
use crate::{CoreError, CoreResult, SageCore};

const APPLICATION_TTL: Duration = Duration::from_secs(5);
const MAX_APPLICATIONS: usize = 16;

struct Entry {
    target: ApplicationTarget,
    expires: Instant,
}

#[derive(Default)]
pub(crate) struct Applications {
    session: String,
    entries: HashMap<String, Entry>,
}

impl Applications {
    fn prune(&mut self, session: &str, now: Instant) {
        if self.session != session {
            self.entries.clear();
            self.session = session.into();
        }
        self.entries.retain(|_, entry| entry.expires > now);
    }

    fn get(&mut self, session: &str, application: &str, now: Instant) -> Option<ApplicationTarget> {
        self.prune(session, now);
        self.entries
            .get(application)
            .map(|entry| entry.target.clone())
    }

    fn insert(
        &mut self,
        session: &str,
        application: &str,
        target: ApplicationTarget,
        now: Instant,
    ) {
        self.prune(session, now);
        if self.entries.len() >= MAX_APPLICATIONS
            && let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.expires)
                .map(|(key, _)| key.clone())
        {
            self.entries.remove(&oldest);
        }
        self.entries.insert(
            application.into(),
            Entry {
                target,
                expires: now + APPLICATION_TTL,
            },
        );
    }
}

impl SageCore {
    pub(crate) async fn prepared_application(
        &self,
        application: &str,
    ) -> CoreResult<ApplicationTarget> {
        let session =
            self.adapters.session_id("native").await.ok_or_else(|| {
                CoreError::ExecutorUnavailable("Native connection is offline".into())
            })?;
        if let Some(target) = self
            .application_preparations
            .lock()
            .expect("application preparations poisoned")
            .get(&session, application, Instant::now())
        {
            return Ok(target);
        }
        let response = self
            .adapters
            .request_in_session(
                "native",
                &session,
                "application_identity",
                serde_json::json!({"application": application}),
            )
            .await?;
        let target: ApplicationTarget =
            serde_json::from_value(response["application_target"].clone())?;
        target.validate()?;
        if self.adapters.session_id("native").await.as_deref() != Some(&session) {
            return Err(CoreError::ExecutorUnavailable(
                "Native connection changed during preparation".into(),
            ));
        }
        self.application_preparations
            .lock()
            .expect("application preparations poisoned")
            .insert(&session, application, target.clone(), Instant::now());
        Ok(target)
    }

    pub(crate) async fn prepare_intent_applications(
        &self,
        intent: &crate::intent::CompiledIntent,
    ) -> CoreResult<usize> {
        let applications = intent
            .steps
            .iter()
            .filter_map(|step| match &step.action {
                crate::domain::Action::OpenApplication { application } => {
                    Some(application.as_str())
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        // At most eight clauses. Sequential lookup keeps speculation from
        // occupying every native worker; execution uses its normal broker lane.
        for application in &applications {
            self.prepared_application(application).await?;
        }
        Ok(applications.len())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Arc;

    use sage_protocol::sage::ipc::v2 as wire;
    use tokio::sync::mpsc;
    use tokio::time::timeout;

    pub(crate) fn target() -> ApplicationTarget {
        ApplicationTarget {
            platform: "macos".into(),
            bundle_path: "/System/Applications/Notes.app".into(),
            identifier: "com.apple.Notes".into(),
            code_digest: ApplicationTarget::code_set_digest(&["ab".repeat(20)]),
            code_digests: vec!["ab".repeat(20)],
            signer: "APPLE".into(),
        }
    }

    pub(crate) fn core() -> (tempfile::TempDir, Arc<SageCore>) {
        let data = tempfile::tempdir().unwrap();
        let core = SageCore::new_with_secret_store(
            crate::config::CoreConfig::for_test(data.path()),
            Arc::new(crate::model::UnconfiguredModelProvider),
            Arc::new(crate::secrets::testing::MemorySecretStore::default()),
        )
        .unwrap();
        (data, core)
    }

    pub(crate) async fn native(
        core: &SageCore,
        session: &str,
    ) -> (
        mpsc::Receiver<wire::AdapterRequest>,
        mpsc::Receiver<wire::AdapterCancel>,
    ) {
        let (requests, receiver) = mpsc::channel(8);
        let (cancellations, cancelled) = mpsc::channel(8);
        core.adapters
            .register(
                "native",
                session,
                wire::ClientKind::Macos as i32,
                crate::execution::bridge::AdapterEndpoint {
                    requests,
                    cancellations: Some(cancellations),
                    terminate: tokio::sync::watch::channel(false).0,
                },
                &[],
            )
            .await
            .unwrap();
        (receiver, cancelled)
    }

    pub(crate) async fn reply(
        core: &SageCore,
        session: &str,
        request: &wire::AdapterRequest,
        target: ApplicationTarget,
        process: u32,
    ) {
        core.adapters
            .complete(
                session,
                wire::AdapterResult {
                    request_id: request.request_id.clone(),
                    success: true,
                    json: "{}".into(),
                    application_target: Some(target.to_wire()),
                    observed_process_id: process,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    pub(crate) async fn lookup(
        core: &Arc<SageCore>,
        receiver: &mut mpsc::Receiver<wire::AdapterRequest>,
        session: &str,
        target: ApplicationTarget,
    ) -> CoreResult<ApplicationTarget> {
        let running = core.clone();
        let pending = tokio::spawn(async move { running.prepared_application("Notes").await });
        let request = timeout(Duration::from_secs(2), receiver.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(request.operation, "application_identity");
        assert!(request.grant.is_none());
        assert!(request.application_target.is_none());
        reply(core, session, &request, target, 0).await;
        pending.await.unwrap()
    }

    #[test]
    fn cache_is_exact_bounded_expiring_and_session_scoped() {
        let now = Instant::now();
        let mut cache = Applications::default();
        cache.insert("first", "Notes", target(), now);
        assert!(cache.get("first", "notes", now).is_none());
        assert!(
            cache
                .get(
                    "first",
                    "Notes",
                    now + APPLICATION_TTL - Duration::from_nanos(1)
                )
                .is_some()
        );
        assert!(cache.get("first", "Notes", now + APPLICATION_TTL).is_none());
        for index in 0..MAX_APPLICATIONS + 1 {
            cache.insert(
                "first",
                &format!("app-{index}"),
                target(),
                now + Duration::from_millis(index as u64),
            );
        }
        assert_eq!(cache.entries.len(), MAX_APPLICATIONS);
        assert!(cache.get("first", "app-0", now).is_none());
        assert!(cache.get("second", "app-1", now).is_none());
        assert!(cache.entries.is_empty());
    }

    #[tokio::test]
    async fn prepared_identity_reuses_only_valid_current_session_observations() {
        let (_data, core) = core();
        let (mut requests, _cancellations) = native(&core, "first").await;
        assert_eq!(
            lookup(&core, &mut requests, "first", target())
                .await
                .unwrap(),
            target()
        );
        assert_eq!(core.prepared_application("Notes").await.unwrap(), target());
        assert!(matches!(
            requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(core.snapshot(true).await.tasks.is_empty());

        core.application_preparations
            .lock()
            .unwrap()
            .entries
            .get_mut("Notes")
            .unwrap()
            .expires = Instant::now();
        lookup(&core, &mut requests, "first", target())
            .await
            .unwrap();
        core.adapters.disconnect("first").await;
        let (mut requests, _cancellations) = native(&core, "second").await;
        assert!(
            core.adapters
                .request_in_session(
                    "native",
                    "first",
                    "application_identity",
                    serde_json::json!({"application":"Notes"})
                )
                .await
                .is_err()
        );
        let mut invalid = target();
        invalid.code_digest = "unverified".into();
        assert!(
            lookup(&core, &mut requests, "second", invalid)
                .await
                .is_err()
        );
        assert!(
            core.application_preparations
                .lock()
                .unwrap()
                .entries
                .is_empty()
        );
        lookup(&core, &mut requests, "second", target())
            .await
            .unwrap();
        core.adapters.disconnect("second").await;
        assert!(core.prepared_application("Notes").await.is_err());
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn cached_identity_still_requires_fresh_approval_grant_and_observation() {
        use crate::domain::TaskStatus;
        use crate::events::CoreEventKind;
        let (_data, core) = core();
        let session = uuid::Uuid::new_v4().to_string();
        let (mut requests, _cancellations) = native(&core, &session).await;
        lookup(&core, &mut requests, &session, target())
            .await
            .unwrap();
        let mut events = core.events().subscribe();
        let mut grants = BTreeSet::new();
        // The second run returns a mismatched observation. A cached lookup
        // cannot turn that failed postcondition into a successful task.
        for observation_matches in [true, false] {
            let task_id = core.submit_task("open Notes").await.unwrap();
            timeout(Duration::from_secs(5), async {
                loop {
                    let event = events.recv().await.unwrap();
                    if let CoreEventKind::ApprovalRequested { approval_id, action_id, digest, .. } = event.kind {
                        assert!(matches!(requests.try_recv(), Err(mpsc::error::TryRecvError::Empty)), "Nothing may launch before approval, nor should cached identity be queried again");
                        core.resolve_approval(approval_id, task_id, action_id, &digest,
                            crate::engine::ApprovalResolution::Approved { native_authentication_satisfied: false }).await.unwrap();
                        break;
                    }
                }
            }).await.unwrap();
            let execute = timeout(Duration::from_secs(5), requests.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(execute.operation, "execute");
            assert_eq!(execute.application_target, Some(target().to_wire()));
            let grant = execute.grant.as_ref().unwrap();
            assert_eq!(grant.worker_session, session);
            assert_eq!(grant.run_id, task_id.to_string());
            assert!(grants.insert(grant.grant_id.clone()));
            reply(&core, &session, &execute, target(), 123).await;
            let observation = timeout(Duration::from_secs(5), requests.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(observation.operation, "observe_application");
            assert!(observation.grant.is_none());
            let mut observed = target();
            if !observation_matches {
                observed.code_digests = vec!["cd".repeat(20)];
                observed.code_digest = ApplicationTarget::code_set_digest(&observed.code_digests);
            }
            reply(&core, &session, &observation, observed, 123).await;
            let finished = timeout(Duration::from_secs(5), async {
                loop {
                    let snapshot = core.snapshot(true).await;
                    let task = snapshot
                        .tasks
                        .iter()
                        .find(|task| task.id == task_id)
                        .unwrap();
                    if !task.status.is_active() {
                        break task.clone();
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(
                finished.status == TaskStatus::Succeeded,
                observation_matches,
                "{finished:?}"
            );
            assert_eq!(finished.completed_count(), usize::from(observation_matches));
        }
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn streamed_app_prefix_waits_for_revision_and_retains_the_verified_step() {
        use crate::domain::{Action, ActionStatus, TaskStatus};
        use crate::events::CoreEventKind;
        use crate::model::{
            ModelRole, ModelTurn, PlanningContext, ProviderDescriptor, ReplanContext, TurnContext,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Debug, Default)]
        struct CountingModel(AtomicUsize);
        #[async_trait::async_trait]
        impl crate::model::ModelProvider for CountingModel {
            fn descriptor(&self) -> ProviderDescriptor {
                ProviderDescriptor {
                    id: "streaming-test".into(),
                    display_name: "Streaming fixture".into(),
                    local: true,
                    roles: vec![ModelRole::Reasoning],
                }
            }

            async fn create_plan(
                &self,
                _context: PlanningContext,
            ) -> CoreResult<crate::domain::ActionGraph> {
                Err(CoreError::Model("unexpected plan request".into()))
            }

            async fn replan(
                &self,
                _context: ReplanContext,
            ) -> CoreResult<crate::domain::ActionGraph> {
                Err(CoreError::Model("unexpected replan request".into()))
            }

            async fn next_turn(&self, _context: TurnContext) -> CoreResult<ModelTurn> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Err(CoreError::Model("fixture planner reached".into()))
            }
        }

        let data = tempfile::tempdir().unwrap();
        let model = Arc::new(CountingModel::default());
        let core = SageCore::new_with_secret_store(
            crate::config::CoreConfig::for_test(data.path()),
            model.clone(),
            Arc::new(crate::secrets::testing::MemorySecretStore::default()),
        )
        .unwrap();
        let session = uuid::Uuid::new_v4().to_string();
        let (mut requests, _cancellations) = native(&core, &session).await;
        lookup(&core, &mut requests, &session, target())
            .await
            .unwrap();

        let workspace = tempfile::tempdir().unwrap();
        let root = workspace.path().canonicalize().unwrap();
        let source = root.join("result.txt");
        std::fs::write(&source, "Read after the app opens").unwrap();
        let scope = crate::contracts::ResourceScope {
            root: root.clone(),
            effects: BTreeSet::from([crate::contracts::Effect::Read]),
        };
        let wire_scope = wire::ResourceScope {
            root: root.display().to_string(),
            effects: vec![wire::ResourceEffect::Read as i32],
        };
        let conversation_id = uuid::Uuid::new_v4();
        let stream_id = uuid::Uuid::new_v4();
        let prefix = wire::SubmitTask {
            text: "Open Notes".into(),
            source: wire::InputSource::Voice as i32,
            conversation_id: conversation_id.to_string(),
            resources: vec![wire_scope.clone()],
            voice_stream_id: stream_id.to_string(),
            streamed_prefix: true,
            ..Default::default()
        };
        let prefix_key =
            crate::commands::SubmissionKey::for_request(&uuid::Uuid::new_v4().to_string(), &prefix)
                .unwrap();
        let mut events = core.events().subscribe();
        let task_id = core
            .submit_streamed_prefix_receipted(
                prefix.text.clone(),
                Some(conversation_id),
                vec![scope.clone()],
                prefix_key,
                stream_id,
            )
            .await
            .unwrap();

        let (approval_id, action_id, digest) = timeout(Duration::from_secs(5), async {
            loop {
                if let CoreEventKind::ApprovalRequested {
                    approval_id,
                    action_id,
                    digest,
                    ..
                } = events.recv().await.unwrap().kind
                {
                    break (approval_id, action_id, digest);
                }
            }
        })
        .await
        .unwrap();
        assert!(matches!(
            requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        core.resolve_approval(
            approval_id,
            task_id,
            action_id,
            &digest,
            crate::engine::ApprovalResolution::Approved {
                native_authentication_satisfied: false,
            },
        )
        .await
        .unwrap();
        let execute = timeout(Duration::from_secs(5), requests.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(execute.operation, "execute");
        assert!(execute.grant.is_some());
        reply(&core, &session, &execute, target(), 123).await;
        let observation = timeout(Duration::from_secs(5), requests.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(observation.operation, "observe_application");
        reply(&core, &session, &observation, target(), 123).await;

        timeout(Duration::from_secs(5), async {
            loop {
                let task = core
                    .snapshot(true)
                    .await
                    .tasks
                    .into_iter()
                    .find(|task| task.id == task_id)
                    .unwrap();
                if task.completed_count() == 1 {
                    break;
                }
                assert_ne!(task.status, TaskStatus::Failed, "{task:?}");
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(model.0.load(Ordering::Relaxed), 0);
        let waiting = core
            .snapshot(true)
            .await
            .tasks
            .into_iter()
            .find(|task| task.id == task_id)
            .unwrap();
        assert!(waiting.status.is_active(), "{waiting:?}");
        assert!(core.signal_stream_control(stream_id, TaskStatus::Paused));
        core.control_task(task_id, TaskStatus::Paused)
            .await
            .unwrap();
        assert_eq!(
            core.snapshot(true)
                .await
                .tasks
                .into_iter()
                .find(|task| task.id == task_id)
                .unwrap()
                .status,
            TaskStatus::Paused
        );

        let request = format!("Open Notes and then read \"{}\"", source.display());
        let final_submit = wire::SubmitTask {
            text: request.clone(),
            source: wire::InputSource::Voice as i32,
            conversation_id: conversation_id.to_string(),
            resources: vec![wire_scope],
            voice_stream_id: stream_id.to_string(),
            finalize_stream: true,
            ..Default::default()
        };
        let final_key = crate::commands::SubmissionKey::for_request(
            &uuid::Uuid::new_v4().to_string(),
            &final_submit,
        )
        .unwrap();
        let accepted = core
            .finalize_streamed_prefix_receipted(stream_id, request, vec![scope], final_key)
            .await
            .unwrap();
        assert_eq!(accepted, task_id);

        let finished = timeout(Duration::from_secs(5), async {
            loop {
                let task = core
                    .snapshot(true)
                    .await
                    .tasks
                    .into_iter()
                    .find(|task| task.id == task_id)
                    .unwrap();
                if !task.status.is_active() {
                    break task;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(finished.status, TaskStatus::Succeeded);
        assert_eq!(model.0.load(Ordering::Relaxed), 0);
        assert_eq!(finished.actions.len(), 2);
        assert_eq!(
            finished
                .actions
                .values()
                .filter(|state| matches!(state.proposal.action, Action::OpenApplication { .. }))
                .count(),
            1
        );
        assert!(
            finished
                .actions
                .values()
                .all(|state| state.status == ActionStatus::Succeeded)
        );
        assert!(matches!(
            requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn streamed_scoped_read_runs_during_speech_and_extends_the_same_task() {
        use crate::domain::{Action, ActionStatus, TaskStatus};

        let (_data, core) = core();
        let workspace = tempfile::tempdir().unwrap();
        let root = workspace.path().canonicalize().unwrap();
        let first = root.join("first.txt");
        let second = root.join("second.txt");
        std::fs::write(&first, "first private result").unwrap();
        std::fs::write(&second, "second private result").unwrap();
        let scope = crate::contracts::ResourceScope {
            root: root.clone(),
            effects: BTreeSet::from([crate::contracts::Effect::Read]),
        };
        let conversation_id = uuid::Uuid::new_v4();
        let blocked_stream = uuid::Uuid::new_v4();
        let blocked_submit = wire::SubmitTask {
            text: format!("Read \"{}\"", first.display()),
            source: wire::InputSource::Voice as i32,
            conversation_id: conversation_id.to_string(),
            voice_stream_id: blocked_stream.to_string(),
            streamed_prefix: true,
            ..Default::default()
        };
        let blocked_key = crate::commands::SubmissionKey::for_request(
            &uuid::Uuid::new_v4().to_string(),
            &blocked_submit,
        )
        .unwrap();
        assert!(
            core.submit_streamed_prefix_receipted(
                blocked_submit.text,
                Some(conversation_id),
                Vec::new(),
                blocked_key,
                blocked_stream,
            )
            .await
            .is_err()
        );
        assert!(core.snapshot(true).await.tasks.is_empty());

        let stream_id = uuid::Uuid::new_v4();
        let prefix = format!("Read \"{}\"", first.display());
        let submit = wire::SubmitTask {
            text: prefix.clone(),
            source: wire::InputSource::Voice as i32,
            conversation_id: conversation_id.to_string(),
            resources: vec![wire::ResourceScope {
                root: root.display().to_string(),
                effects: vec![wire::ResourceEffect::Read as i32],
            }],
            voice_stream_id: stream_id.to_string(),
            streamed_prefix: true,
            ..Default::default()
        };
        let key =
            crate::commands::SubmissionKey::for_request(&uuid::Uuid::new_v4().to_string(), &submit)
                .unwrap();
        let task_id = core
            .submit_streamed_prefix_receipted(
                prefix,
                Some(conversation_id),
                vec![scope.clone()],
                key,
                stream_id,
            )
            .await
            .unwrap();
        let first_result = timeout(Duration::from_secs(5), async {
            loop {
                let task = core
                    .snapshot(true)
                    .await
                    .tasks
                    .into_iter()
                    .find(|task| task.id == task_id)
                    .unwrap();
                if task.completed_count() == 1 {
                    break task;
                }
                assert_ne!(task.status, TaskStatus::Failed, "{task:?}");
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(first_result.status.is_active(), "{first_result:?}");
        assert_eq!(first_result.actions.len(), 1);
        assert!(first_result.actions.values().all(|state| {
            matches!(state.proposal.action, Action::ReadFile { .. })
                && state.status == ActionStatus::Succeeded
        }));
        assert!(first_result.tool_results.iter().any(|result| {
            result.output["text"]
                .as_str()
                .is_some_and(|text| text.contains("first private result"))
        }));

        let request = format!(
            "Read \"{}\" and then read \"{}\"",
            first.display(),
            second.display()
        );
        let final_submit = wire::SubmitTask {
            text: request.clone(),
            source: wire::InputSource::Voice as i32,
            conversation_id: conversation_id.to_string(),
            resources: vec![wire::ResourceScope {
                root: root.display().to_string(),
                effects: vec![wire::ResourceEffect::Read as i32],
            }],
            voice_stream_id: stream_id.to_string(),
            finalize_stream: true,
            ..Default::default()
        };
        let final_key = crate::commands::SubmissionKey::for_request(
            &uuid::Uuid::new_v4().to_string(),
            &final_submit,
        )
        .unwrap();
        let accepted = core
            .finalize_streamed_prefix_receipted(stream_id, request, vec![scope], final_key)
            .await
            .unwrap();
        assert_eq!(accepted, task_id);

        let finished = timeout(Duration::from_secs(5), async {
            loop {
                let task = core
                    .snapshot(true)
                    .await
                    .tasks
                    .into_iter()
                    .find(|task| task.id == task_id)
                    .unwrap();
                if !task.status.is_active() {
                    break task;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(finished.status, TaskStatus::Succeeded, "{finished:?}");
        assert_eq!(finished.actions.len(), 2);
        assert_eq!(finished.completed_count(), 2);
        assert_eq!(
            finished
                .actions
                .values()
                .filter(|state| matches!(state.proposal.action, Action::ReadFile { .. }))
                .count(),
            2
        );
        assert!(finished.tool_results.iter().any(|result| {
            result.output["text"]
                .as_str()
                .is_some_and(|text| text.contains("second private result"))
        }));
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    #[ignore = "release-mode latency measurement"]
    async fn streamed_prefix_approval_latency() {
        use crate::domain::TaskStatus;
        use crate::events::CoreEventKind;

        let (_data, core) = core();
        let session = uuid::Uuid::new_v4().to_string();
        let (mut requests, _cancellations) = native(&core, &session).await;
        lookup(&core, &mut requests, &session, target())
            .await
            .unwrap();
        let mut events = core.events().subscribe();
        let mut samples = Vec::with_capacity(100);
        for _ in 0..100 {
            let conversation_id = uuid::Uuid::new_v4();
            let stream_id = uuid::Uuid::new_v4();
            let submit = wire::SubmitTask {
                text: "Open Notes".into(),
                source: wire::InputSource::Voice as i32,
                conversation_id: conversation_id.to_string(),
                voice_stream_id: stream_id.to_string(),
                streamed_prefix: true,
                ..Default::default()
            };
            let key = crate::commands::SubmissionKey::for_request(
                &uuid::Uuid::new_v4().to_string(),
                &submit,
            )
            .unwrap();
            let started = Instant::now();
            let task_id = core
                .submit_streamed_prefix_receipted(
                    submit.text,
                    Some(conversation_id),
                    Vec::new(),
                    key,
                    stream_id,
                )
                .await
                .unwrap();
            timeout(Duration::from_secs(3), async {
                loop {
                    let event = events.recv().await.unwrap();
                    if event.task_id == Some(task_id)
                        && matches!(event.kind, CoreEventKind::ApprovalRequested { .. })
                    {
                        break;
                    }
                }
            })
            .await
            .unwrap();
            samples.push(started.elapsed().as_nanos());
            core.control_streamed_task(stream_id, TaskStatus::Cancelled)
                .await
                .unwrap();
            timeout(Duration::from_secs(3), async {
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
            .unwrap();
        }
        samples.sort_unstable();
        println!(
            "streamed_prefix_approval samples={} p50_ns={} p95_ns={} p99_ns={}",
            samples.len(),
            samples[50],
            samples[95],
            samples[99]
        );
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    #[ignore = "release-mode latency measurement"]
    async fn streamed_file_read_latency() {
        use crate::domain::TaskStatus;
        use crate::events::CoreEventKind;

        let (_data, core) = core();
        let workspace = tempfile::tempdir().unwrap();
        let root = workspace.path().canonicalize().unwrap();
        let path = root.join("latency.txt");
        std::fs::write(&path, "bounded local read for streamed voice").unwrap();
        let scope = crate::contracts::ResourceScope {
            root: root.clone(),
            effects: BTreeSet::from([crate::contracts::Effect::Read]),
        };
        let mut events = core.events().subscribe();
        let mut samples = Vec::with_capacity(100);
        for _ in 0..100 {
            let conversation_id = uuid::Uuid::new_v4();
            let stream_id = uuid::Uuid::new_v4();
            let submit = wire::SubmitTask {
                text: format!("Read \"{}\"", path.display()),
                source: wire::InputSource::Voice as i32,
                conversation_id: conversation_id.to_string(),
                resources: vec![wire::ResourceScope {
                    root: root.display().to_string(),
                    effects: vec![wire::ResourceEffect::Read as i32],
                }],
                voice_stream_id: stream_id.to_string(),
                streamed_prefix: true,
                ..Default::default()
            };
            let key = crate::commands::SubmissionKey::for_request(
                &uuid::Uuid::new_v4().to_string(),
                &submit,
            )
            .unwrap();
            let started = Instant::now();
            let task_id = core
                .submit_streamed_prefix_receipted(
                    submit.text,
                    Some(conversation_id),
                    vec![scope.clone()],
                    key,
                    stream_id,
                )
                .await
                .unwrap();
            timeout(Duration::from_secs(3), async {
                loop {
                    let event = events.recv().await.unwrap();
                    if event.task_id == Some(task_id)
                        && matches!(event.kind, CoreEventKind::ActionSucceeded { .. })
                    {
                        break;
                    }
                }
            })
            .await
            .unwrap();
            samples.push(started.elapsed().as_nanos());
            core.control_streamed_task(stream_id, TaskStatus::Cancelled)
                .await
                .unwrap();
            timeout(Duration::from_secs(3), async {
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
            .unwrap();
        }
        samples.sort_unstable();
        println!(
            "streamed_file_read samples={} p50_ns={} p95_ns={} p99_ns={}",
            samples.len(),
            samples[50],
            samples[95],
            samples[99]
        );
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn streamed_stop_cancels_the_pending_approval_before_launch() {
        use crate::domain::TaskStatus;
        use crate::events::CoreEventKind;
        let (_data, core) = core();
        let session = uuid::Uuid::new_v4().to_string();
        let (mut requests, _cancellations) = native(&core, &session).await;
        lookup(&core, &mut requests, &session, target())
            .await
            .unwrap();
        let conversation_id = uuid::Uuid::new_v4();
        let stream_id = uuid::Uuid::new_v4();
        let submit = wire::SubmitTask {
            text: "Open Notes".into(),
            source: wire::InputSource::Voice as i32,
            conversation_id: conversation_id.to_string(),
            voice_stream_id: stream_id.to_string(),
            streamed_prefix: true,
            ..Default::default()
        };
        let key =
            crate::commands::SubmissionKey::for_request(&uuid::Uuid::new_v4().to_string(), &submit)
                .unwrap();
        let mut events = core.events().subscribe();
        let task_id = core
            .submit_streamed_prefix_receipted(
                submit.text,
                Some(conversation_id),
                Vec::new(),
                key,
                stream_id,
            )
            .await
            .unwrap();
        timeout(Duration::from_secs(5), async {
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
        assert!(matches!(
            requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        let unsupported = wire::SubmitTask {
            text: "Open Notes and then search the web for today's news".into(),
            source: wire::InputSource::Voice as i32,
            conversation_id: conversation_id.to_string(),
            voice_stream_id: stream_id.to_string(),
            finalize_stream: true,
            ..Default::default()
        };
        let unsupported_key = crate::commands::SubmissionKey::for_request(
            &uuid::Uuid::new_v4().to_string(),
            &unsupported,
        )
        .unwrap();
        assert!(
            core.finalize_streamed_prefix_receipted(
                stream_id,
                unsupported.text,
                Vec::new(),
                unsupported_key,
            )
            .await
            .is_err()
        );
        assert!(core.signal_stream_control(stream_id, TaskStatus::Cancelled));
        core.control_streamed_task(stream_id, TaskStatus::Cancelled)
            .await
            .unwrap();
        let stopped = timeout(Duration::from_secs(5), async {
            loop {
                let snapshot = core.snapshot(true).await;
                let task = snapshot
                    .tasks
                    .iter()
                    .find(|task| task.id == task_id)
                    .unwrap()
                    .clone();
                if task.status == TaskStatus::Cancelled {
                    break (task, snapshot);
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(stopped.0.completed_count(), 0);
        assert!(stopped.1.pending_approvals.is_empty());
        assert!(matches!(
            requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }
}
