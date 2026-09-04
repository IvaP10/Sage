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
    Failed,
    Cancelled,
    Interrupted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionStatus {
    Pending,
    Compiling,
    WaitingForApproval,
    Running,
    Verifying,
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
    pub request: String,
    pub status: TaskStatus,
    pub goal: Option<String>,
    pub actions: BTreeMap<Uuid, ActionState>,
    pub dependencies: BTreeMap<Uuid, BTreeSet<Uuid>>,
    pub created_resources: BTreeSet<String>,
    pub final_outcome: Option<String>,
    #[serde(default)]
    pub rollback_available: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Task {
    pub fn new(request: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            request: request.into(),
            status: TaskStatus::Pending,
            goal: None,
            actions: BTreeMap::new(),
            dependencies: BTreeMap::new(),
            created_resources: BTreeSet::new(),
            final_outcome: None,
            rollback_available: false,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn install_plan(&mut self, graph: ActionGraph) -> Result<(), String> {
        graph.validate(self.id)?;
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

    pub fn ready_actions(&self) -> Vec<Uuid> {
        self.actions
            .iter()
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
        self.actions
            .values()
            .filter(|state| state.status == ActionStatus::Succeeded)
            .count()
    }

    pub fn is_complete(&self) -> bool {
        !self.actions.is_empty()
            && self.actions.values().all(|state| {
                matches!(
                    state.status,
                    ActionStatus::Succeeded | ActionStatus::Skipped
                )
            })
            && self
                .actions
                .values()
                .any(|state| state.status == ActionStatus::Succeeded)
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
