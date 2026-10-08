use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{ActionGraph, ActionProposal};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Planning,
    Running,
    WaitingForApproval,
    WaitingForUser,
    Paused,
    Succeeded,
    Answered,
    Partial,
    Failed,
    Cancelled,
    Interrupted,
}

impl TaskStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Answered | Self::Partial | Self::Failed | Self::Cancelled
        )
    }

    pub fn is_active(self) -> bool {
        !self.is_terminal() && self != Self::Interrupted
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionStatus {
    Pending,
    Compiling,
    WaitingForApproval,
    Running,
    Verifying,
    Uncertain,
    Succeeded,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionState {
    pub proposal: ActionProposal,
    pub status: ActionStatus,
    pub attempts: u32,
    pub summary: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub execution_attempt: u64,
    #[serde(default)]
    pub continuation_of: Option<Uuid>,
    #[serde(default)]
    pub continued_by: Option<Uuid>,
    #[serde(default)]
    pub control_scope_id: Option<Uuid>,
    #[serde(default)]
    pub budget_exhausted: bool,
    #[serde(default)]
    pub conversation_id: Option<Uuid>,
    #[serde(default)]
    pub message_id: Option<Uuid>,
    #[serde(default)]
    pub recovery_attempt: bool,
    #[serde(default)]
    pub workflow_run: bool,
    #[serde(default)]
    pub intent: Option<crate::intent::IntentState>,
    #[serde(default)]
    pub compiled_routine: Option<String>,
    #[serde(default)]
    pub synthesized_skill_sources: Vec<String>,
    #[serde(default)]
    pub synthesized_skill_lineages: Vec<(String, String, Vec<String>)>,
    /// Marks an explicit foreground-reference lookup as consumed. Raw
    /// selection text stays in the active run only and is never serialized.
    #[serde(default)]
    pub reference_captured: bool,
    #[serde(default)]
    pub background: bool,
    #[serde(default)]
    pub contract: Option<crate::contracts::RunContract>,
    #[serde(default)]
    pub tool_results: Vec<crate::contracts::ToolResult>,
    pub request: String,
    pub status: TaskStatus,
    pub goal: Option<String>,
    pub actions: BTreeMap<Uuid, ActionState>,
    pub dependencies: BTreeMap<Uuid, BTreeSet<Uuid>>,
    pub created_resources: BTreeSet<String>,
    pub final_outcome: Option<String>,
    #[serde(default)]
    pub rollback_available: bool,
    #[serde(default)]
    pub rollback_action_id: Option<Uuid>,
    #[serde(default)]
    pub undo: Option<UndoProgress>,
    #[serde(default)]
    pub undone_actions: BTreeSet<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Task {
    pub fn new(request: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            revision: 0,
            execution_attempt: 0,
            continuation_of: None,
            continued_by: None,
            control_scope_id: None,
            budget_exhausted: false,
            conversation_id: None,
            message_id: None,
            recovery_attempt: false,
            workflow_run: false,
            intent: None,
            compiled_routine: None,
            synthesized_skill_sources: Vec::new(),
            synthesized_skill_lineages: Vec::new(),
            reference_captured: false,
            background: false,
            contract: None,
            tool_results: Vec::new(),
            request: request.into(),
            status: TaskStatus::Pending,
            goal: None,
            actions: BTreeMap::new(),
            dependencies: BTreeMap::new(),
            created_resources: BTreeSet::new(),
            final_outcome: None,
            rollback_available: false,
            rollback_action_id: None,
            undo: None,
            undone_actions: BTreeSet::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn control_scope(&self) -> Uuid {
        self.control_scope_id.unwrap_or(self.id)
    }

    pub fn action_is_current(&self, id: Uuid) -> bool {
        self.intent
            .as_ref()
            .is_none_or(|intent| !intent.retired.contains(&id))
    }

    pub fn current_actions(&self) -> impl Iterator<Item = (&Uuid, &ActionState)> {
        self.actions
            .iter()
            .filter(|(id, _)| self.action_is_current(**id))
    }

    pub fn install_plan(&mut self, graph: ActionGraph) -> Result<(), String> {
        graph.validate(self.id)?;
        if graph.nodes.len() > 32 {
            return Err("Workflow exceeds the 32-step run budget".into());
        }
        self.workflow_run = true;
        self.goal = Some(graph.goal);
        self.actions = graph
            .nodes
            .iter()
            .map(|node| {
                (
                    node.proposal.id,
                    ActionState {
                        proposal: node.proposal.clone(),
                        status: ActionStatus::Pending,
                        attempts: 0,
                        summary: None,
                        error: None,
                    },
                )
            })
            .collect();
        self.dependencies = graph
            .nodes
            .into_iter()
            .map(|node| (node.proposal.id, node.depends_on))
            .collect();
        self.status = TaskStatus::Running;
        self.touch();
        Ok(())
    }

    pub fn append_plan(&mut self, graph: ActionGraph) -> Result<(), String> {
        graph.validate(self.id)?;
        if graph
            .nodes
            .iter()
            .any(|n| self.actions.contains_key(&n.proposal.id))
        {
            return Err("Every turn must use fresh action identities".into());
        }
        if self.actions.len() + graph.nodes.len() > 32 {
            return Err("Task reached its 32-step budget".into());
        }
        // A later reasoning turn consumed earlier verified effects. Preserve
        // that causal boundary when the task is saved as a reusable skill;
        // UUID sort order is not execution order. Failed attempts are excluded
        // so a recovery turn can still run, and cannot be captured as a skill.
        let prior_verified = self
            .actions
            .iter()
            .filter(|(_, state)| state.status == ActionStatus::Succeeded)
            .map(|(id, _)| *id)
            .collect::<BTreeSet<_>>();
        self.goal = Some(graph.goal);
        for mut node in graph.nodes {
            if node.depends_on.is_empty() {
                node.depends_on.extend(prior_verified.iter().copied());
            }
            self.dependencies.insert(node.proposal.id, node.depends_on);
            self.actions.insert(
                node.proposal.id,
                ActionState {
                    proposal: node.proposal,
                    status: ActionStatus::Pending,
                    attempts: 0,
                    summary: None,
                    error: None,
                },
            );
        }
        self.status = TaskStatus::Running;
        self.touch();
        Ok(())
    }

    /// Append interpreter-selected actions while preserving their exact
    /// dependency edges. Unlike `append_plan`, this does not implicitly make
    /// every prior successful action a dependency of a new root node.
    pub(crate) fn append_plan_with_exact_dependencies(
        &mut self,
        graph: ActionGraph,
    ) -> Result<(), String> {
        let new_ids = graph
            .nodes
            .iter()
            .map(|node| node.proposal.id)
            .collect::<BTreeSet<_>>();
        if new_ids.len() != graph.nodes.len()
            || graph
                .nodes
                .iter()
                .any(|node| self.actions.contains_key(&node.proposal.id))
        {
            return Err("Every appended action must use a fresh identity".into());
        }
        if self.actions.len() + graph.nodes.len() > 32 {
            return Err("Task reached its 32-step budget".into());
        }
        let known = self.actions.keys().copied().collect::<BTreeSet<_>>();
        if graph.nodes.iter().any(|node| {
            node.depends_on
                .iter()
                .any(|dependency| !known.contains(dependency) && !new_ids.contains(dependency))
        }) {
            return Err("An appended action depends on an action outside the task".into());
        }
        let mut internal_graph = graph.clone();
        for node in &mut internal_graph.nodes {
            node.depends_on
                .retain(|dependency| new_ids.contains(dependency));
        }
        internal_graph.validate(self.id)?;
        self.goal = Some(graph.goal);
        self.workflow_run = true;
        for node in graph.nodes {
            self.dependencies.insert(node.proposal.id, node.depends_on);
            self.actions.insert(
                node.proposal.id,
                ActionState {
                    proposal: node.proposal,
                    status: ActionStatus::Pending,
                    attempts: 0,
                    summary: None,
                    error: None,
                },
            );
        }
        self.status = TaskStatus::Running;
        self.touch();
        Ok(())
    }

    pub fn ready_actions(&self) -> Vec<Uuid> {
        self.current_actions()
            .filter(|(_, state)| state.status == ActionStatus::Pending)
            .filter(|(id, _)| {
                self.dependencies.get(id).is_none_or(|dependencies| {
                    dependencies.iter().all(|dependency| {
                        self.actions
                            .get(dependency)
                            .is_some_and(|state| state.status == ActionStatus::Succeeded)
                    })
                })
            })
            .map(|(id, _)| *id)
            .collect()
    }

    pub fn completed_count(&self) -> usize {
        super::ExecutionFacts::for_task(self).verified as usize
    }

    pub fn is_complete(&self) -> bool {
        super::ExecutionFacts::for_task(self).all_verified()
    }

    pub fn install_replan(
        &mut self,
        failed_action_id: Uuid,
        graph: ActionGraph,
    ) -> Result<(), String> {
        graph.validate(self.id)?;
        if graph
            .nodes
            .iter()
            .any(|node| self.actions.contains_key(&node.proposal.id))
        {
            return Err("replan must use fresh action ids".into());
        }
        if let Some(failed) = self.actions.get_mut(&failed_action_id) {
            failed.status = ActionStatus::Skipped;
        }
        self.goal = Some(graph.goal);
        for node in graph.nodes {
            self.dependencies
                .insert(node.proposal.id, node.depends_on.clone());
            self.actions.insert(
                node.proposal.id,
                ActionState {
                    proposal: node.proposal,
                    status: ActionStatus::Pending,
                    attempts: 0,
                    summary: None,
                    error: None,
                },
            );
        }
        self.status = TaskStatus::Running;
        self.touch();
        Ok(())
    }

    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndoPhase {
    Prepared,
    Dispatched,
    Uncertain,
    Verified,
}

impl UndoPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prepared => "prepared",
            Self::Dispatched => "dispatched",
            Self::Uncertain => "uncertain",
            Self::Verified => "verified",
        }
    }

    pub fn summary(self) -> &'static str {
        match self {
            Self::Prepared => "Undo is ready but has not started. Retry Undo to continue.",
            Self::Dispatched => "Undo may have run. Check its result before continuing.",
            Self::Uncertain => {
                "Undo could not be verified. Check the file or folder; another check will not repeat the change."
            }
            Self::Verified => "Undo verified against the current file or folder.",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UndoProgress {
    pub action_id: Uuid,
    pub phase: UndoPhase,
}
