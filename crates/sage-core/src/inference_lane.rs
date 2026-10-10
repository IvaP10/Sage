//! One-owner, bounded generation lane for Sage's unadmitted Qwen candidate.
//!
//! A persistent thread keeps the large decoder resident and prevents logical
//! callers from racing its recurrent/KV state. This is an offline evaluation
//! seam only: an in-process thread is not an OS isolation boundary and is not
//! selected by product startup.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{SyncSender, TrySendError, sync_channel},
    },
    thread,
};

use tokio::sync::{mpsc, oneshot};

use sage_qwen35_runtime::qwen35::SAGE_OUTPUT_LIMIT;

use crate::model::{ModelProvider, ModelRole, PlanningContext, ProviderDescriptor, ReplanContext};
use crate::{
    CoreError, CoreResult,
    model::{ModelTurn, TurnContext, UntrustedContext},
    qwen35_loader::Qwen35CandidateModel,
};

const PENDING_REQUEST_CAPACITY: usize = 1;
const MAX_REPLAN_OBSERVATION_BYTES: usize = 16 * 1024;

struct GenerationRequest {
    context: TurnContext,
    maximum_new_tokens: usize,
    cancelled: Arc<AtomicBool>,
    updates: Option<mpsc::Sender<String>>,
    response: oneshot::Sender<CoreResult<ModelTurn>>,
}

/// A cloneable handle to one persistent scalar candidate owner. There may be
/// one active generation and at most one queued generation. Additional work
/// receives `Busy` immediately rather than accumulating stale private prompts.
#[derive(Clone)]
pub struct CandidateGenerationLane {
    sender: SyncSender<GenerationRequest>,
}

impl CandidateGenerationLane {
    /// Start a dedicated, long-lived thread around an already verified but
    /// unadmitted candidate. The thread owns the decoder for its whole life.
    pub fn start(mut candidate: Qwen35CandidateModel) -> CoreResult<Self> {
        let (sender, receiver) = sync_channel::<GenerationRequest>(PENDING_REQUEST_CAPACITY);
        let _worker = thread::Builder::new()
            .name("sage-qwen35-candidate".into())
            .spawn(move || {
                run_worker_with_updates(
                    receiver,
                    |context, maximum_new_tokens, cancelled, updates| {
                        candidate.generate_turn_with_answer_updates(
                            context,
                            maximum_new_tokens,
                            |prefix| {
                                if let Some(updates) = updates {
                                    let _ = updates.try_send(prefix.to_owned());
                                }
                            },
                            || cancelled.load(Ordering::Acquire),
                        )
                    },
                )
            })
            .map_err(CoreError::Io)?;
        Ok(Self { sender })
    }

    /// Submit a bounded request. Dropping this future signals cooperative
    /// cancellation, including when the request is waiting in the one-item
    /// queue. Model output is still only an untrusted proposal.
    pub async fn generate(
        &self,
        context: TurnContext,
        maximum_new_tokens: usize,
    ) -> CoreResult<ModelTurn> {
        self.generate_inner(context, maximum_new_tokens, None).await
    }

    /// Generate one turn and send bounded cumulative answer previews. A full
    /// preview channel drops intermediate updates instead of blocking decoding.
    pub async fn generate_with_updates(
        &self,
        context: TurnContext,
        maximum_new_tokens: usize,
        updates: mpsc::Sender<String>,
    ) -> CoreResult<ModelTurn> {
        self.generate_inner(context, maximum_new_tokens, Some(updates))
            .await
    }

    async fn generate_inner(
        &self,
        context: TurnContext,
        maximum_new_tokens: usize,
        updates: Option<mpsc::Sender<String>>,
    ) -> CoreResult<ModelTurn> {
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancellation_guard = CancellationGuard::new(Arc::clone(&cancelled));
        let receive = self.submit(context, maximum_new_tokens, cancelled, updates)?;
        let result = receive.await.map_err(|_| {
            CoreError::Model("The local candidate generation lane stopped unexpectedly".into())
        })?;
        cancellation_guard.complete();
        result
    }

    fn submit(
        &self,
        context: TurnContext,
        maximum_new_tokens: usize,
        cancelled: Arc<AtomicBool>,
        updates: Option<mpsc::Sender<String>>,
    ) -> CoreResult<oneshot::Receiver<CoreResult<ModelTurn>>> {
        if maximum_new_tokens == 0 || maximum_new_tokens > SAGE_OUTPUT_LIMIT as usize {
            return Err(CoreError::Model(
                "Candidate output reservation is outside Sage's limit".into(),
            ));
        }
        let (response, receive) = oneshot::channel();
        let request = GenerationRequest {
            context,
            maximum_new_tokens,
            cancelled,
            updates,
            response,
        };
        self.sender.try_send(request).map_err(|error| match error {
            TrySendError::Full(_) => {
                CoreError::Busy("The local candidate generation lane is full".into())
            }
            TrySendError::Disconnected(_) => {
                CoreError::Model("The local candidate generation lane has stopped".into())
            }
        })?;
        Ok(receive)
    }
}

/// Model-provider adapter for offline evaluation of one loaded scalar
/// candidate. It keeps the decoder on its dedicated owner thread, but that
/// thread is in-process and is not a product security boundary.
#[derive(Clone)]
pub struct CandidateModelProvider {
    lane: CandidateGenerationLane,
}

impl CandidateModelProvider {
    pub fn start(candidate: Qwen35CandidateModel) -> CoreResult<Self> {
        Ok(Self {
            lane: CandidateGenerationLane::start(candidate)?,
        })
    }

    async fn generate(&self, context: TurnContext) -> CoreResult<ModelTurn> {
        self.lane
            .generate(context, SAGE_OUTPUT_LIMIT as usize)
            .await
    }

    async fn generate_actions(
        &self,
        context: TurnContext,
    ) -> CoreResult<crate::domain::ActionGraph> {
        match self.generate(context).await? {
            ModelTurn::Actions(graph) => Ok(graph),
            ModelTurn::Answer(_) => Err(CoreError::Model(
                "The local candidate returned an answer where an action graph was required".into(),
            )),
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for CandidateModelProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        ProviderDescriptor {
            id: "sage-qwen35-scalar-candidate".into(),
            display_name: "Qwen3.5-4B scalar candidate (evaluation only)".into(),
            local: true,
            roles: vec![ModelRole::Reasoning],
        }
    }

    async fn create_plan(
        &self,
        context: PlanningContext,
    ) -> CoreResult<crate::domain::ActionGraph> {
        self.generate_actions(TurnContext {
            planning: context,
            results: Vec::new(),
            destination: None,
        })
        .await
    }

    async fn replan(&self, context: ReplanContext) -> CoreResult<crate::domain::ActionGraph> {
        let mut planning = context.planning.ok_or_else(|| {
            CoreError::Model("Replanning requires a fresh broker-built planning context".into())
        })?;
        if planning.task_id != context.task.id || context.failed_action_id.is_nil() {
            return Err(CoreError::Model(
                "Replanning identity does not match its task and failed action".into(),
            ));
        }
        let failure = serde_json::to_string(&context.observation)
            .map_err(|_| CoreError::Model("Failed observation could not be encoded".into()))?;
        if failure.len() > MAX_REPLAN_OBSERVATION_BYTES {
            return Err(CoreError::Model(
                "Failed observation exceeds the bounded replan context".into(),
            ));
        }
        planning.untrusted_context.push(UntrustedContext {
            source: format!("failed_action_observation:{}", context.failed_action_id),
            content: failure,
        });
        let results = context
            .task
            .tool_results
            .iter()
            .rev()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        self.generate_actions(TurnContext {
            planning,
            results,
            destination: None,
        })
        .await
    }

    fn data_destination(&self) -> CoreResult<Option<String>> {
        Ok(None)
    }

    async fn next_turn(&self, context: TurnContext) -> CoreResult<ModelTurn> {
        self.generate(context).await
    }

    async fn next_turn_stream(
        &self,
        context: TurnContext,
        updates: mpsc::Sender<String>,
    ) -> CoreResult<ModelTurn> {
        self.lane
            .generate_with_updates(context, SAGE_OUTPUT_LIMIT as usize, updates)
            .await
    }
}

#[cfg(test)]
fn run_worker<F>(receiver: std::sync::mpsc::Receiver<GenerationRequest>, mut generate: F)
where
    F: FnMut(&TurnContext, usize, &AtomicBool) -> CoreResult<ModelTurn>,
{
    run_worker_with_updates(receiver, |context, maximum_new_tokens, cancelled, _| {
        generate(context, maximum_new_tokens, cancelled)
    });
}

fn run_worker_with_updates<F>(
    receiver: std::sync::mpsc::Receiver<GenerationRequest>,
    mut generate: F,
) where
    F: FnMut(
        &TurnContext,
        usize,
        &AtomicBool,
        Option<&mpsc::Sender<String>>,
    ) -> CoreResult<ModelTurn>,
{
    while let Ok(request) = receiver.recv() {
        let result = if request.cancelled.load(Ordering::Acquire) {
            Err(CoreError::Cancelled)
        } else {
            generate(
                &request.context,
                request.maximum_new_tokens,
                &request.cancelled,
                request.updates.as_ref(),
            )
        };
        let _ = request.response.send(result);
    }
}

struct CancellationGuard {
    flag: Arc<AtomicBool>,
    armed: bool,
}

impl CancellationGuard {
    fn new(flag: Arc<AtomicBool>) -> Self {
        Self { flag, armed: true }
    }

    fn complete(self) {
        let mut guard = self;
        guard.armed = false;
    }
}

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        if self.armed {
            self.flag.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CancellationGuard, CandidateGenerationLane, CandidateModelProvider, GenerationRequest,
        run_worker, run_worker_with_updates,
    };
    use crate::{
        CoreError,
        domain::Task,
        model::{ModelProvider, ModelTurn, PlanningContext, ReplanContext, TurnContext},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::sync_channel,
    };
    use tokio::sync::{mpsc, oneshot};
    use uuid::Uuid;

    fn context() -> TurnContext {
        TurnContext {
            planning: PlanningContext {
                task_id: Uuid::from_u128(1),
                user_request: "test".into(),
                current_state: serde_json::Value::Null,
                available_tools: vec![],
                trusted_constraints: vec![],
                untrusted_context: vec![],
            },
            results: vec![],
            destination: None,
        }
    }

    fn test_lane<F>(execute: F) -> CandidateGenerationLane
    where
        F: FnMut(&TurnContext, usize, &AtomicBool) -> crate::CoreResult<ModelTurn> + Send + 'static,
    {
        let (sender, receiver) = sync_channel(1);
        std::thread::spawn(move || run_worker(receiver, execute));
        CandidateGenerationLane { sender }
    }

    fn test_lane_with_updates<F>(execute: F) -> CandidateGenerationLane
    where
        F: FnMut(
                &TurnContext,
                usize,
                &AtomicBool,
                Option<&mpsc::Sender<String>>,
            ) -> crate::CoreResult<ModelTurn>
            + Send
            + 'static,
    {
        let (sender, receiver) = sync_channel(1);
        std::thread::spawn(move || run_worker_with_updates(receiver, execute));
        CandidateGenerationLane { sender }
    }

    fn request(
        response: oneshot::Sender<crate::CoreResult<ModelTurn>>,
        cancelled: Arc<AtomicBool>,
    ) -> GenerationRequest {
        GenerationRequest {
            context: context(),
            maximum_new_tokens: 16,
            cancelled,
            updates: None,
            response,
        }
    }

    #[test]
    fn abandoned_generation_signals_cooperative_cancellation() {
        let cancelled = Arc::new(AtomicBool::new(false));
        drop(CancellationGuard::new(Arc::clone(&cancelled)));
        assert!(cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn completed_generation_does_not_retain_a_cancellation_guard() {
        let cancelled = Arc::new(AtomicBool::new(false));
        CancellationGuard::new(Arc::clone(&cancelled)).complete();
        assert!(!cancelled.load(Ordering::Acquire));
        assert_eq!(Arc::strong_count(&cancelled), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lane_runs_one_request_and_bounds_the_pending_queue_to_one() {
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mut first = true;
        let lane = test_lane(move |_, _, _| {
            if first {
                first = false;
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            }
            Ok(ModelTurn::Answer("ok".into()))
        });

        let (first_response, first_receive) = oneshot::channel();
        lane.sender
            .try_send(request(first_response, Arc::new(AtomicBool::new(false))))
            .unwrap();
        tokio::task::spawn_blocking(move || started_rx.recv().unwrap())
            .await
            .unwrap();

        let (second_response, second_receive) = oneshot::channel();
        lane.sender
            .try_send(request(second_response, Arc::new(AtomicBool::new(false))))
            .unwrap();
        let (overflow_response, _) = oneshot::channel();
        let overflow = lane
            .sender
            .try_send(request(overflow_response, Arc::new(AtomicBool::new(false))));
        assert!(matches!(
            overflow,
            Err(std::sync::mpsc::TrySendError::Full(_))
        ));

        release_tx.send(()).unwrap();
        assert!(matches!(
            first_receive.await.unwrap(),
            Ok(ModelTurn::Answer(_))
        ));
        assert!(matches!(
            second_receive.await.unwrap(),
            Ok(ModelTurn::Answer(_))
        ));
    }

    #[tokio::test]
    async fn candidate_provider_routes_planning_to_local_generation_without_data_release() {
        let invoked = Arc::new(AtomicBool::new(false));
        let invoked_by_worker = Arc::clone(&invoked);
        let lane = test_lane(move |context, maximum_new_tokens, cancelled| {
            assert_eq!(context.planning.user_request, "test");
            assert_eq!(maximum_new_tokens, super::SAGE_OUTPUT_LIMIT as usize);
            assert!(!cancelled.load(Ordering::Acquire));
            invoked_by_worker.store(true, Ordering::Release);
            Ok(ModelTurn::Answer("generated locally".into()))
        });
        let provider = CandidateModelProvider { lane };

        let turn = provider.next_turn(context()).await.unwrap();
        assert!(matches!(turn, ModelTurn::Answer(answer) if answer == "generated locally"));
        assert!(invoked.load(Ordering::Acquire));
        assert!(provider.descriptor().local);
        assert!(provider.data_destination().unwrap().is_none());
    }

    #[tokio::test]
    async fn streaming_provider_forwards_bounded_cumulative_answer_previews() {
        let lane = test_lane_with_updates(|_, _, _, updates| {
            let updates = updates.expect("streaming request supplies an update channel");
            assert!(updates.try_send("Working".into()).is_ok());
            assert!(updates.try_send("Working on it".into()).is_err());
            Ok(ModelTurn::Answer("Working on it".into()))
        });
        let provider = CandidateModelProvider { lane };
        let (sender, mut receiver) = mpsc::channel(1);

        let turn = provider.next_turn_stream(context(), sender).await.unwrap();

        assert!(matches!(turn, ModelTurn::Answer(answer) if answer == "Working on it"));
        assert_eq!(receiver.recv().await.as_deref(), Some("Working"));
        assert!(receiver.recv().await.is_none());
    }

    #[tokio::test]
    async fn replanning_binds_failure_evidence_to_the_untrusted_context() {
        let (context_tx, context_rx) = std::sync::mpsc::channel();
        let lane = test_lane(move |context, _, _| {
            context_tx
                .send(context.planning.untrusted_context.clone())
                .unwrap();
            Ok(ModelTurn::Answer("answer cannot replace a replan".into()))
        });
        let provider = CandidateModelProvider { lane };
        let planning = context().planning;
        let task_id = planning.task_id;
        let failed_action_id = Uuid::from_u128(9);
        let mut task = Task::new("test");
        task.id = task_id;

        let result = provider
            .replan(ReplanContext {
                task,
                failed_action_id,
                observation: serde_json::json!({"state":"changed"}),
                attempt: 1,
                planning: Some(planning),
            })
            .await;
        assert!(result.is_err());

        let captured = tokio::task::spawn_blocking(move || context_rx.recv().unwrap())
            .await
            .unwrap();
        let failure = captured.last().unwrap();
        assert_eq!(
            failure.source,
            "failed_action_observation:00000000-0000-0000-0000-000000000009"
        );
        assert_eq!(failure.content, r#"{"state":"changed"}"#);
    }

    #[tokio::test]
    async fn replan_rejects_cross_task_context_before_generation() {
        let invoked = Arc::new(AtomicBool::new(false));
        let invoked_by_worker = Arc::clone(&invoked);
        let lane = test_lane(move |_, _, _| {
            invoked_by_worker.store(true, Ordering::Release);
            Ok(ModelTurn::Answer("unexpected".into()))
        });
        let provider = CandidateModelProvider { lane };
        let mut planning = context().planning;
        planning.task_id = Uuid::from_u128(2);
        let task = Task::new("test");

        let result = provider
            .replan(ReplanContext {
                task,
                failed_action_id: Uuid::from_u128(9),
                observation: serde_json::Value::Null,
                attempt: 1,
                planning: Some(planning),
            })
            .await;

        assert!(result.is_err());
        assert!(!invoked.load(Ordering::Acquire));
    }

    #[test]
    fn worker_skips_a_cancelled_request_before_running_the_model() {
        let (sender, receiver) = sync_channel(1);
        let invoked = Arc::new(AtomicBool::new(false));
        let invoked_by_worker = Arc::clone(&invoked);
        let worker = std::thread::spawn(move || {
            run_worker(receiver, move |_, _, _| {
                invoked_by_worker.store(true, Ordering::Release);
                Ok(ModelTurn::Answer("unexpected".into()))
            })
        });
        let cancelled = Arc::new(AtomicBool::new(true));
        let (response, receive) = oneshot::channel();
        sender.send(request(response, cancelled)).unwrap();
        assert!(matches!(
            receive.blocking_recv().unwrap(),
            Err(CoreError::Cancelled)
        ));
        assert!(!invoked.load(Ordering::Acquire));
        drop(sender);
        worker.join().unwrap();
    }
}
