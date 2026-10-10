//! Model-provider adapter for the isolated, signed-package Qwen candidate.
//!
//! This adapter is available only in an explicit generation build and accepts
//! an already verified and loaded worker. It does not discover packages,
//! configure trust roots, or make model admission decisions. Core still treats
//! every generated turn as an untrusted proposal.

use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

use async_trait::async_trait;
use sage_inference_protocol::MAX_QWEN35_GENERATION_TOKENS;
use tokio::sync::mpsc;

use crate::{
    CoreError, CoreResult,
    inference_worker_process::LoadedQwen35CandidateWorker,
    model::{
        ModelProvider, ModelRole, ModelTurn, PlanningContext, ProviderDescriptor, ReplanContext,
        TurnContext, UntrustedContext,
    },
    qwen_prompt::format_turn_prompt,
};

const MAX_REPLAN_OBSERVATION_BYTES: usize = 16 * 1024;

/// A one-generation-at-a-time local planner backed by Sage's restricted worker.
///
/// Dropping an in-flight generation drops the owned worker process. The
/// provider then fails closed until its caller verifies and loads a fresh
/// worker; it never leaves detached model work running after cancellation.
pub struct IsolatedQwen35WorkerProvider {
    worker: Mutex<Option<LoadedQwen35CandidateWorker>>,
    generation_active: AtomicBool,
}

impl IsolatedQwen35WorkerProvider {
    /// Construct the adapter from a worker whose package and candidate-load
    /// receipts have already been validated by Core.
    pub fn new(worker: LoadedQwen35CandidateWorker) -> Self {
        Self {
            worker: Mutex::new(Some(worker)),
            generation_active: AtomicBool::new(false),
        }
    }

    async fn generate_turn(
        &self,
        context: TurnContext,
        updates: Option<mpsc::Sender<String>>,
    ) -> CoreResult<ModelTurn> {
        let prompt = format_turn_prompt(&context)?;
        if self
            .generation_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(CoreError::Busy(
                "The isolated Qwen generation lane is already occupied".into(),
            ));
        }
        let _active = ActiveGenerationLease(&self.generation_active);
        let worker = self
            .worker
            .lock()
            .map_err(|_| CoreError::Busy("The isolated Qwen worker state is unavailable".into()))?
            .take()
            .ok_or_else(|| {
                CoreError::Model(
                    "The isolated Qwen worker was retired; verify and load the package again"
                        .into(),
                )
            })?;

        let preview_sender = updates;
        let generated = worker
            .generate_planner_json_with_previews(
                &prompt.system_message,
                &prompt.user_message,
                &prompt.output_schema,
                MAX_QWEN35_GENERATION_TOKENS,
                move |preview| {
                    if let Some(sender) = &preview_sender {
                        let _ = sender.try_send(preview.to_owned());
                    }
                },
            )
            .await;

        let (worker, output) = generated.map_err(|error| {
            tracing::warn!(%error, "isolated Qwen generation failed; retiring its worker");
            CoreError::Model(
                "Isolated Qwen generation failed; verify and load the package again".into(),
            )
        })?;
        *self.worker.lock().map_err(|_| {
            CoreError::Busy("The isolated Qwen worker state is unavailable".into())
        })? = Some(worker);

        crate::model::parse_turn(&output, context.planning.task_id)
    }
}

struct ActiveGenerationLease<'a>(&'a AtomicBool);

impl Drop for ActiveGenerationLease<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

#[async_trait]
impl ModelProvider for IsolatedQwen35WorkerProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        ProviderDescriptor {
            id: "sage-qwen35-isolated-candidate".into(),
            display_name: "Sage isolated Qwen3.5 candidate (not admitted)".into(),
            local: true,
            roles: vec![ModelRole::Reasoning],
        }
    }

    async fn create_plan(
        &self,
        context: PlanningContext,
    ) -> CoreResult<crate::domain::ActionGraph> {
        match self
            .generate_turn(
                TurnContext {
                    planning: context,
                    results: Vec::new(),
                    destination: None,
                },
                None,
            )
            .await?
        {
            ModelTurn::Actions(graph) => Ok(graph),
            ModelTurn::Answer(_) => Err(CoreError::Model(
                "The local Qwen planner returned an answer where a plan was required".into(),
            )),
        }
    }

    async fn replan(&self, context: ReplanContext) -> CoreResult<crate::domain::ActionGraph> {
        let planning = replan_planning_context(&context)?;
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
        match self
            .generate_turn(
                TurnContext {
                    planning,
                    results,
                    destination: None,
                },
                None,
            )
            .await?
        {
            ModelTurn::Actions(graph) => Ok(graph),
            ModelTurn::Answer(_) => Err(CoreError::Model(
                "The local Qwen planner returned an answer where a replacement plan was required"
                    .into(),
            )),
        }
    }

    async fn next_turn(&self, context: TurnContext) -> CoreResult<ModelTurn> {
        self.generate_turn(context, None).await
    }

    async fn next_turn_stream(
        &self,
        context: TurnContext,
        updates: mpsc::Sender<String>,
    ) -> CoreResult<ModelTurn> {
        self.generate_turn(context, Some(updates)).await
    }

    fn data_destination(&self) -> CoreResult<Option<String>> {
        Ok(None)
    }
}

fn replan_planning_context(context: &ReplanContext) -> CoreResult<PlanningContext> {
    let mut planning = context.planning.clone().ok_or_else(|| {
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
    Ok(planning)
}

#[cfg(test)]
mod tests {
    use super::replan_planning_context;
    use crate::model::{PlanningContext, ReplanContext};
    use uuid::Uuid;

    #[test]
    fn replan_binds_fresh_task_identity_and_marks_failure_observation_untrusted() {
        let task_id = Uuid::from_u128(7);
        let failed_action_id = Uuid::from_u128(8);
        let mut task = crate::domain::Task::new("Inspect a file");
        task.id = task_id;
        let context = ReplanContext {
            task,
            failed_action_id,
            observation: serde_json::json!({"status":"changed"}),
            attempt: 1,
            planning: Some(PlanningContext {
                task_id,
                user_request: "Inspect a file".into(),
                current_state: serde_json::Value::Null,
                available_tools: Vec::new(),
                trusted_constraints: Vec::new(),
                untrusted_context: Vec::new(),
            }),
        };

        let planning = replan_planning_context(&context).unwrap();

        assert_eq!(planning.task_id, task_id);
        assert_eq!(planning.untrusted_context.len(), 1);
        assert_eq!(
            planning.untrusted_context[0].source,
            format!("failed_action_observation:{failed_action_id}")
        );
        assert_eq!(
            planning.untrusted_context[0].content,
            r#"{"status":"changed"}"#
        );
    }

    #[test]
    fn replan_rejects_missing_or_mismatched_current_planning_context() {
        let mut task = crate::domain::Task::new("Inspect");
        task.id = Uuid::from_u128(7);
        let mut context = ReplanContext {
            task,
            failed_action_id: Uuid::from_u128(8),
            observation: serde_json::Value::Null,
            attempt: 1,
            planning: None,
        };
        assert!(replan_planning_context(&context).is_err());

        context.planning = Some(PlanningContext {
            task_id: Uuid::from_u128(9),
            user_request: "Inspect".into(),
            current_state: serde_json::Value::Null,
            available_tools: Vec::new(),
            trusted_constraints: Vec::new(),
            untrusted_context: Vec::new(),
        });
        assert!(replan_planning_context(&context).is_err());
    }
}
