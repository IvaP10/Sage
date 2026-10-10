use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

use crate::domain::{ActionGraph, Task};
use crate::error::{CoreError, CoreResult};

pub(crate) const MAX_TURN_BYTES: usize = 256 * 1024;
const MAX_ANSWER_BYTES: usize = 64 * 1024;
pub(crate) const MAX_READ_BATCH: usize = 8;
pub(crate) const INDEPENDENT_READ_TOOLS: &[&str] =
    sage_inference_protocol::QWEN35_INDEPENDENT_READ_TOOL_KINDS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Reasoning,
    Vision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderDescriptor {
    pub id: String,
    pub display_name: String,
    pub local: bool,
    pub roles: Vec<ModelRole>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningContext {
    pub task_id: uuid::Uuid,
    pub user_request: String,
    pub current_state: serde_json::Value,
    pub available_tools: Vec<ToolDescriptor>,
    pub trusted_constraints: Vec<String>,
    pub untrusted_context: Vec<UntrustedContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplanContext {
    pub task: Task,
    pub failed_action_id: uuid::Uuid,
    pub observation: serde_json::Value,
    pub attempt: u32,
    #[serde(default)]
    pub planning: Option<PlanningContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UntrustedContext {
    pub source: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub name: String,
    pub version: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
    pub risk: String,
    pub required_capabilities: Vec<String>,
    pub supported_platforms: Vec<String>,
    pub requires_confirmation: bool,
    pub executor: String,
    pub timeout_ms: u64,
    pub verification_strategy: String,
}

#[derive(Debug, Clone)]
pub enum ModelTurn {
    Answer(String),
    Actions(ActionGraph),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnContext {
    pub planning: PlanningContext,
    pub results: Vec<crate::contracts::ToolResult>,
    pub destination: Option<String>,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn descriptor(&self) -> ProviderDescriptor;
    async fn create_plan(&self, context: PlanningContext) -> CoreResult<ActionGraph>;
    async fn replan(&self, context: ReplanContext) -> CoreResult<ActionGraph>;

    /// A nonempty destination is data-release scoped by the broker.
    fn data_destination(&self) -> CoreResult<Option<String>> {
        Ok(None)
    }

    async fn next_turn(&self, context: TurnContext) -> CoreResult<ModelTurn> {
        if context.results.is_empty() {
            self.create_plan(context.planning)
                .await
                .map(ModelTurn::Actions)
        } else {
            Ok(ModelTurn::Answer(
                context
                    .results
                    .iter()
                    .map(|r| r.summary.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            ))
        }
    }

    async fn next_turn_stream(
        &self,
        context: TurnContext,
        _updates: tokio::sync::mpsc::Sender<String>,
    ) -> CoreResult<ModelTurn> {
        self.next_turn(context).await
    }
}

#[derive(Debug, Default)]
pub struct UnconfiguredModelProvider;

#[async_trait]
impl ModelProvider for UnconfiguredModelProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        ProviderDescriptor {
            id: "unconfigured".into(),
            display_name: "Sage first-party inference is not yet available".into(),
            local: true,
            roles: vec![ModelRole::Reasoning],
        }
    }

    async fn create_plan(&self, _context: PlanningContext) -> CoreResult<ActionGraph> {
        Err(CoreError::Model(
            "Sage's first-party local inference worker is not yet available; no external model service or runtime is enabled".into(),
        ))
    }

    async fn replan(&self, _context: ReplanContext) -> CoreResult<ActionGraph> {
        Err(CoreError::Model(
            "Sage's first-party local inference worker is not yet available for replanning".into(),
        ))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DraftPlan {
    goal: String,
    answer: String,
    actions: Vec<DraftAction>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DraftAction {
    kind: String,
    payload: serde_json::Value,
}

/// Closed structured-output schema for Sage's untrusted local planner.
pub fn draft_schema() -> serde_json::Value {
    let variants=crate::features::manifests().into_iter().filter(|feature|feature.enabled).map(|feature|serde_json::json!({
        "type":"object","additionalProperties":false,"required":["kind","payload"],
        "properties":{"kind":{"type":"string","enum":[feature.id]},"payload":feature.input_schema}
    })).collect::<Vec<_>>();
    serde_json::json!({"type":"object","additionalProperties":false,"required":["goal","answer","actions"],
        "properties":{"goal":{"type":"string","maxLength":4096},"answer":{"type":"string","maxLength":MAX_ANSWER_BYTES},
            "actions":{"type":"array","maxItems":MAX_READ_BATCH,"items":{"anyOf":variants}}}})
}

/// Decode one bounded planner turn and assign Sage-owned identities and provenance.
pub fn parse_turn(content: &str, task_id: Uuid) -> CoreResult<ModelTurn> {
    if task_id.is_nil() {
        return Err(CoreError::Model(
            "Structured model task identity is invalid".into(),
        ));
    }
    draft_to_turn(parse_draft(content)?, task_id)
}

/// Validate a complete planner turn against the same closed schema and
/// action-graph rules as `parse_turn`, without attaching it to a live task.
/// Constrained decoding uses this before admitting the model end token.
pub fn validate_turn_json(content: &str) -> CoreResult<()> {
    let draft = parse_draft(content)?;
    validate_draft_shape(&draft)?;
    if draft.actions.is_empty() {
        return Ok(());
    }
    let proposal_ids = (0..draft.actions.len())
        .map(|index| Uuid::from_u128(index as u128 + 2))
        .collect();
    draft_to_graph_with_ids(draft, Uuid::from_u128(1), proposal_ids).map(|_| ())
}

fn parse_draft(content: &str) -> CoreResult<DraftPlan> {
    if content.len() > MAX_TURN_BYTES {
        return Err(CoreError::Model(
            "Structured model response exceeds its input bound".into(),
        ));
    }
    serde_json::from_str(content)
        .map_err(|_| CoreError::Model("Malformed structured model response".into()))
}

fn draft_to_turn(draft: DraftPlan, task_id: Uuid) -> CoreResult<ModelTurn> {
    validate_draft_shape(&draft)?;
    if draft.actions.is_empty() {
        return Ok(ModelTurn::Answer(draft.answer));
    }
    draft_to_graph(draft, task_id).map(ModelTurn::Actions)
}

fn validate_draft_shape(draft: &DraftPlan) -> CoreResult<()> {
    if draft.actions.is_empty()
        && !draft.answer.trim().is_empty()
        && draft.answer.len() <= MAX_ANSWER_BYTES
    {
        return Ok(());
    }
    if draft.actions.is_empty() || draft.actions.len() > MAX_READ_BATCH || !draft.answer.is_empty()
    {
        return Err(CoreError::Model(
            "Expected one answer, one action, or up to eight independent reads".into(),
        ));
    }
    if draft.actions.len() > 1
        && draft
            .actions
            .iter()
            .any(|action| !INDEPENDENT_READ_TOOLS.contains(&action.kind.as_str()))
    {
        return Err(CoreError::Model(
            "Only independent read actions can share a turn".into(),
        ));
    }
    Ok(())
}

fn draft_to_graph(draft: DraftPlan, task_id: Uuid) -> CoreResult<ActionGraph> {
    let ids = (0..draft.actions.len()).map(|_| Uuid::new_v4()).collect();
    draft_to_graph_with_ids(draft, task_id, ids)
}

fn draft_to_graph_with_ids(
    draft: DraftPlan,
    task_id: Uuid,
    ids: Vec<Uuid>,
) -> CoreResult<ActionGraph> {
    if draft.goal.trim().is_empty() || draft.goal.len() > 4096 {
        return Err(CoreError::Model(
            "model returned an invalid plan goal".into(),
        ));
    }
    if draft.actions.is_empty()
        || draft.actions.len() > MAX_READ_BATCH
        || ids.len() != draft.actions.len()
        || ids.iter().any(Uuid::is_nil)
        || ids.iter().collect::<BTreeSet<_>>().len() != ids.len()
    {
        return Err(CoreError::Model(
            "Planner action count or proposal identities are invalid".into(),
        ));
    }
    let mut nodes = Vec::with_capacity(draft.actions.len());
    for (index, draft_action) in draft.actions.into_iter().enumerate() {
        let action = action_from_draft(draft_action.kind, draft_action.payload)?;
        // These placeholders are replaced by Core's action-specific target
        // resolver and verifier before policy, approval, or dispatch.
        let expected_outcome = crate::domain::ExpectedOutcome::ExternalSuccess {
            marker: "awaiting-broker-binding".into(),
        };
        let proposal = crate::domain::ActionProposal {
            id: ids[index],
            task_id,
            action,
            expected_outcome,
            target_resource: "broker-resolved".into(),
            provenance: crate::domain::Provenance::model(vec![task_id.to_string()]),
            metadata: BTreeMap::new(),
        };
        nodes.push(crate::domain::ActionNode {
            proposal,
            depends_on: BTreeSet::new(),
        });
    }
    let graph = ActionGraph {
        goal: draft.goal,
        nodes,
    };
    graph
        .validate(task_id)
        .map_err(|error| CoreError::Model(format!("model plan failed Sage validation: {error}")))?;
    Ok(graph)
}

fn action_from_draft(
    kind: String,
    payload: serde_json::Value,
) -> CoreResult<crate::domain::Action> {
    let mut object = payload
        .as_object()
        .cloned()
        .ok_or_else(|| CoreError::Model("model action payload must be an object".into()))?;
    object.insert("type".into(), serde_json::Value::String(kind));
    serde_json::from_value(serde_json::Value::Object(object)).map_err(|_| {
        CoreError::Model("model returned an action payload that Sage does not recognize".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_batches_accept_only_bounded_independent_reads() {
        let read = serde_json::json!({"kind":"read_file","payload":{"path":"/tmp/input.txt","max_bytes":64}});
        let turn = |actions: Vec<serde_json::Value>| {
            serde_json::json!({"goal":"Inspect inputs","answer":"","actions":actions}).to_string()
        };
        let task_id = Uuid::new_v4();
        assert!(validate_turn_json(&turn(vec![read.clone(); 8])).is_ok());
        assert!(validate_turn_json(&turn(vec![read.clone(); 9])).is_err());
        let ModelTurn::Actions(graph) = parse_turn(&turn(vec![read.clone(); 8]), task_id).unwrap()
        else {
            panic!("Expected reads")
        };
        assert_eq!(graph.nodes.len(), 8);
        assert!(
            graph
                .nodes
                .iter()
                .all(|node| node.proposal.task_id == task_id && node.depends_on.is_empty())
        );
        assert!(parse_turn(&turn(vec![read.clone(); 9]), task_id).is_err());
        for action in [
            serde_json::json!({"kind":"write_file","payload":{"path":"/tmp/out","content":"data","overwrite":false}}),
            serde_json::json!({"kind":"ask_user","payload":{"question":"Proceed?"}}),
            serde_json::json!({"kind":"read_file","payload":{"path":"/tmp/derived","max_bytes":64},"depends_on":[0]}),
            serde_json::json!({"kind":"read_file","payload":{"path":"/tmp/derived","max_bytes":64},"expected_outcome":{"kind":"external_success","marker":"claimed"}}),
        ] {
            assert!(parse_turn(&turn(vec![read.clone(), action]), task_id).is_err());
        }
        assert!(matches!(
            parse_turn(
                r#"{"goal":"Answer","answer":"Ready","actions":[]}"#,
                task_id
            )
            .unwrap(),
            ModelTurn::Answer(_)
        ));
        assert!(validate_turn_json(r#"{"goal":"Answer","answer":"Ready","actions":[]}"#).is_ok());
        assert!(
            validate_turn_json(
                r#"{"goal":"Answer","answer":"Ready","actions":[],"target":"/tmp"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn decoder_rejects_oversized_output_and_missing_task_identity() {
        let oversized = " ".repeat(MAX_TURN_BYTES + 1);
        assert!(parse_turn(&oversized, Uuid::new_v4()).is_err());
        assert!(
            parse_turn(
                r#"{"goal":"Answer","answer":"Ready","actions":[]}"#,
                Uuid::nil()
            )
            .is_err()
        );
    }

    #[test]
    fn graph_validation_preserves_the_sage_assigned_proposal_ids() {
        let task_id = Uuid::from_u128(1);
        let proposal_id = Uuid::from_u128(2);
        let draft = DraftPlan {
            goal: "Inspect the file".into(),
            answer: String::new(),
            actions: vec![DraftAction {
                kind: "read_file".into(),
                payload: serde_json::json!({"path":"/tmp/input.txt","max_bytes":64}),
            }],
        };

        let graph = draft_to_graph_with_ids(draft, task_id, vec![proposal_id]).unwrap();

        assert_eq!(graph.nodes[0].proposal.id, proposal_id);
        assert_eq!(graph.nodes[0].proposal.task_id, task_id);
    }
}
