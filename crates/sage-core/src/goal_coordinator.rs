//! Bounded, authority-free coordination for delegated Sage subgoals.
//!
//! A delegation scope constrains proposals. It is not a permission, grant, or
//! execution token; every eventual action must still pass through the normal
//! target, policy, approval, capability, execution, and verification chain.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::contracts::Effect;
use crate::{CoreError, CoreResult};

const MAX_GOAL_ID_BYTES: usize = 128;
const MAX_BRANCHES: usize = 3;
const MAX_SUBGOALS_PER_BRANCH: usize = 16;
const MAX_SCOPE_CAPABILITIES: usize = 64;
const MAX_SCOPE_SYSTEMS: usize = 32;
const MAX_SCOPE_DISCLOSURES: usize = 32;
const MAX_SUBGOAL_TEXT_BYTES: usize = 1024;
const MAX_SUBGOAL_ATTEMPTS: u8 = 3;
const MAX_EVIDENCE_IDS: usize = 16;
const MAX_OUTPUT_BYTES: u64 = 16 * 1024 * 1024;

fn bounded_text(value: &str, limit: usize, label: &str) -> CoreResult<()> {
    if value.trim().is_empty()
        || value.len() > limit
        || value.contains('\0')
        || value.chars().any(char::is_control)
        || crate::redaction::redact_for_persistence(value) != value
    {
        return Err(CoreError::InvalidAction(format!(
            "{label} is empty, oversized, contains control characters, or resembles a credential"
        )));
    }
    Ok(())
}

fn valid_capability_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b'-' | b':')
        })
}

/// Limits what one logical specialist may propose. These limits confer no
/// execution authority and cannot be converted into a broker grant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelegationScope {
    pub capability_ids: BTreeSet<String>,
    pub system_ids: BTreeSet<Uuid>,
    pub effects: BTreeSet<Effect>,
    pub disclosure_scopes: BTreeSet<String>,
    pub maximum_calls: u16,
    pub maximum_output_bytes: u64,
}

impl DelegationScope {
    pub fn validate(&self) -> CoreResult<()> {
        if self.capability_ids.is_empty()
            || self.capability_ids.len() > MAX_SCOPE_CAPABILITIES
            || self.system_ids.is_empty()
            || self.system_ids.len() > MAX_SCOPE_SYSTEMS
            || self.effects.is_empty()
            || self.disclosure_scopes.len() > MAX_SCOPE_DISCLOSURES
            || self.maximum_calls == 0
            || self.maximum_calls > 32
            || self.maximum_output_bytes == 0
            || self.maximum_output_bytes > MAX_OUTPUT_BYTES
            || self.system_ids.iter().any(Uuid::is_nil)
            || self
                .capability_ids
                .iter()
                .any(|capability| !valid_capability_id(capability))
        {
            return Err(CoreError::InvalidAction(
                "Delegation proposal scope is empty or outside the coordinator bounds".into(),
            ));
        }
        for disclosure in &self.disclosure_scopes {
            bounded_text(disclosure, 128, "delegation disclosure scope")?;
        }
        Ok(())
    }

    /// Check strict attenuation against the parent scope. Empty disclosure
    /// scope is valid and means no data disclosure is proposed.
    pub fn is_attenuated_by(&self, parent: &Self) -> bool {
        self.capability_ids.is_subset(&parent.capability_ids)
            && self.system_ids.is_subset(&parent.system_ids)
            && self.effects.is_subset(&parent.effects)
            && self.disclosure_scopes.is_subset(&parent.disclosure_scopes)
            && self.maximum_calls <= parent.maximum_calls
            && self.maximum_output_bytes <= parent.maximum_output_bytes
    }

    fn conflicts_with(&self, other: &Self) -> bool {
        self.system_ids
            .iter()
            .any(|system| other.system_ids.contains(system))
            && (self.effects.iter().any(|effect| *effect != Effect::Read)
                || other.effects.iter().any(|effect| *effect != Effect::Read))
    }
}

/// Initial first-party specialists are deliberately logical roles. They do
/// not imply separate processes or trusted model identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalSpecialist {
    Planner,
    Verifier,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelegatedSubgoal {
    pub id: String,
    pub specialist: LogicalSpecialist,
    pub instruction: String,
    pub success_condition: String,
    pub dependencies: BTreeSet<String>,
    pub scope: DelegationScope,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateBranch {
    pub id: String,
    pub rationale: String,
    pub subgoals: Vec<DelegatedSubgoal>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinationPlan {
    pub schema_version: u32,
    pub goal_id: String,
    /// Candidate alternatives are descriptive proposals and must be selected
    /// before any subgoal is dispatched.
    pub candidates: Vec<CandidateBranch>,
}

impl CoordinationPlan {
    pub fn validate(&self, parent_scope: &DelegationScope) -> CoreResult<()> {
        parent_scope.validate()?;
        bounded_text(&self.goal_id, MAX_GOAL_ID_BYTES, "delegated goal identity")?;
        if self.schema_version != 1
            || self.candidates.is_empty()
            || self.candidates.len() > MAX_BRANCHES
        {
            return Err(CoreError::InvalidAction(
                "Coordination plan version or candidate count is invalid".into(),
            ));
        }

        let mut branch_ids = BTreeSet::new();
        for branch in &self.candidates {
            bounded_text(&branch.id, 96, "candidate branch identity")?;
            bounded_text(
                &branch.rationale,
                MAX_SUBGOAL_TEXT_BYTES,
                "candidate rationale",
            )?;
            if !branch_ids.insert(&branch.id)
                || branch.subgoals.is_empty()
                || branch.subgoals.len() > MAX_SUBGOALS_PER_BRANCH
            {
                return Err(CoreError::InvalidAction(
                    "Candidate branch identity or subgoal count is invalid".into(),
                ));
            }

            let mut subgoal_ids = BTreeSet::new();
            let mut subgoal_by_id = BTreeMap::new();
            let mut specialists = BTreeSet::new();
            let mut branch_maximum_calls = 0u32;
            let mut branch_maximum_output_bytes = 0u64;
            for (index, subgoal) in branch.subgoals.iter().enumerate() {
                bounded_text(&subgoal.id, 96, "delegated subgoal identity")?;
                bounded_text(
                    &subgoal.instruction,
                    MAX_SUBGOAL_TEXT_BYTES,
                    "delegated subgoal instruction",
                )?;
                bounded_text(
                    &subgoal.success_condition,
                    MAX_SUBGOAL_TEXT_BYTES,
                    "delegated subgoal success condition",
                )?;
                subgoal.scope.validate()?;
                if !subgoal.scope.is_attenuated_by(parent_scope)
                    || !subgoal_ids.insert(&subgoal.id)
                    || subgoal.dependencies.len() > MAX_SUBGOALS_PER_BRANCH
                    || subgoal.dependencies.contains(&subgoal.id)
                {
                    return Err(CoreError::PolicyDenied(
                        "Delegated subgoal widens scope or has duplicate/invalid identity".into(),
                    ));
                }
                specialists.insert(subgoal.specialist);
                branch_maximum_calls = branch_maximum_calls
                    .checked_add(u32::from(subgoal.scope.maximum_calls))
                    .ok_or_else(|| {
                        CoreError::InvalidAction("Delegated call budget overflowed".into())
                    })?;
                branch_maximum_output_bytes = branch_maximum_output_bytes
                    .checked_add(subgoal.scope.maximum_output_bytes)
                    .ok_or_else(|| {
                        CoreError::InvalidAction("Delegated output budget overflowed".into())
                    })?;
                subgoal_by_id.insert(subgoal.id.as_str(), index);
            }
            if specialists.len() > 2
                || branch_maximum_calls > u32::from(parent_scope.maximum_calls)
                || branch_maximum_output_bytes > parent_scope.maximum_output_bytes
            {
                return Err(CoreError::InvalidAction(
                    "Candidate branch exceeds the specialist count or aggregate scope budget"
                        .into(),
                ));
            }
            validate_dependencies(branch, &subgoal_ids, &subgoal_by_id)?;
        }
        Ok(())
    }
}

fn validate_dependencies(
    branch: &CandidateBranch,
    subgoal_ids: &BTreeSet<&String>,
    subgoal_by_id: &BTreeMap<&str, usize>,
) -> CoreResult<()> {
    let mut indegree = vec![0usize; branch.subgoals.len()];
    let mut successors = vec![Vec::<usize>::new(); branch.subgoals.len()];
    for (target, subgoal) in branch.subgoals.iter().enumerate() {
        for dependency in &subgoal.dependencies {
            if !subgoal_ids.contains(dependency) {
                return Err(CoreError::InvalidAction(
                    "Delegated subgoal names an unknown dependency".into(),
                ));
            }
            let source = subgoal_by_id[dependency.as_str()];
            indegree[target] += 1;
            successors[source].push(target);
        }
    }

    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect::<Vec<_>>();
    let mut visited = 0usize;
    while let Some(source) = ready.pop() {
        visited += 1;
        for target in &successors[source] {
            indegree[*target] -= 1;
            if indegree[*target] == 0 {
                ready.push(*target);
            }
        }
    }
    if visited != branch.subgoals.len() {
        return Err(CoreError::InvalidAction(
            "Delegated subgoal dependencies contain a cycle".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "details", rename_all = "snake_case")]
enum SubgoalState {
    Planned,
    Running,
    RepairRequired {
        failure: String,
    },
    NeedsDecision {
        question: String,
        unsettled_effect: bool,
    },
    Succeeded {
        evidence_ids: BTreeSet<Uuid>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SubgoalProgress {
    instruction: String,
    scope: DelegationScope,
    attempts: u8,
    state: SubgoalState,
}

/// Descriptive serialized coordination progress. It contains proposal scopes,
/// statuses and evidence references only; it cannot resume a dispatched effect
/// or confer authority by itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalCoordinationCheckpoint {
    schema_version: u32,
    plan: CoordinationPlan,
    parent_scope: DelegationScope,
    selected_branch: Option<String>,
    progress: BTreeMap<String, SubgoalProgress>,
    parallelism: u8,
    repairs_remaining: u8,
}

impl GoalCoordinationCheckpoint {
    pub fn validate(&self) -> CoreResult<()> {
        GoalCoordinator::restore(self.clone()).map(|_| ())
    }

    pub fn has_unsettled_effects(&self) -> bool {
        self.progress.values().any(|progress| {
            matches!(
                progress.state,
                SubgoalState::Running
                    | SubgoalState::NeedsDecision {
                        unsettled_effect: true,
                        ..
                    }
            )
        })
    }

    pub fn has_pending_obligations(&self) -> bool {
        self.selected_branch.is_some()
            && self
                .progress
                .values()
                .any(|progress| !matches!(progress.state, SubgoalState::Succeeded { .. }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadySubgoal {
    pub id: String,
    pub specialist: LogicalSpecialist,
    pub instruction: String,
    pub success_condition: String,
    pub scope: DelegationScope,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureDisposition {
    RepairRequired { repairs_remaining: u8 },
    DecisionRequired { question: String },
}

/// Bounded orchestration state for proposals and evidence. Callers must only
/// mark a node dispatched after the broker durably admits it, and may record
/// success only after the ordinary independent verifier accepts fresh evidence.
#[derive(Debug, Clone)]
pub struct GoalCoordinator {
    plan: CoordinationPlan,
    parent_scope: DelegationScope,
    selected_branch: Option<String>,
    progress: BTreeMap<String, SubgoalProgress>,
    parallelism: u8,
    repairs_remaining: u8,
}

impl GoalCoordinator {
    pub fn new(
        plan: CoordinationPlan,
        parent_scope: DelegationScope,
        parallelism: u8,
        repair_budget: u8,
    ) -> CoreResult<Self> {
        plan.validate(&parent_scope)?;
        if parallelism == 0 || parallelism > 2 || repair_budget > 2 {
            return Err(CoreError::InvalidAction(
                "Goal coordinator resource or repair budget is outside its limit".into(),
            ));
        }
        Ok(Self {
            plan,
            parent_scope,
            selected_branch: None,
            progress: BTreeMap::new(),
            parallelism,
            repairs_remaining: repair_budget,
        })
    }

    pub fn checkpoint(&self) -> GoalCoordinationCheckpoint {
        GoalCoordinationCheckpoint {
            schema_version: 1,
            plan: self.plan.clone(),
            parent_scope: self.parent_scope.clone(),
            selected_branch: self.selected_branch.clone(),
            progress: self.progress.clone(),
            parallelism: self.parallelism,
            repairs_remaining: self.repairs_remaining,
        }
    }

    pub fn restore(checkpoint: GoalCoordinationCheckpoint) -> CoreResult<Self> {
        checkpoint.parent_scope.validate()?;
        checkpoint.plan.validate(&checkpoint.parent_scope)?;
        if checkpoint.schema_version != 1
            || checkpoint.parallelism == 0
            || checkpoint.parallelism > 2
            || checkpoint.repairs_remaining > 2
        {
            return Err(CoreError::InvalidAction(
                "Goal coordination checkpoint version or runtime bounds are invalid".into(),
            ));
        }
        let selected = match checkpoint.selected_branch.as_deref() {
            Some(branch_id) => Some(
                checkpoint
                    .plan
                    .candidates
                    .iter()
                    .find(|branch| branch.id == branch_id)
                    .ok_or_else(|| {
                        CoreError::VerificationFailed(
                            "Goal coordination checkpoint selected an unknown branch".into(),
                        )
                    })?,
            ),
            None => None,
        };
        if let Some(branch) = selected {
            if checkpoint.progress.len() != branch.subgoals.len() {
                return Err(CoreError::VerificationFailed(
                    "Goal coordination checkpoint does not cover its selected branch".into(),
                ));
            }
            for subgoal in &branch.subgoals {
                let progress = checkpoint.progress.get(&subgoal.id).ok_or_else(|| {
                    CoreError::VerificationFailed(
                        "Goal coordination checkpoint omits a selected subgoal".into(),
                    )
                })?;
                bounded_text(
                    &progress.instruction,
                    MAX_SUBGOAL_TEXT_BYTES,
                    "checkpoint delegated instruction",
                )?;
                progress.scope.validate()?;
                if !progress.scope.is_attenuated_by(&subgoal.scope)
                    || !progress.scope.is_attenuated_by(&checkpoint.parent_scope)
                    || progress.attempts > MAX_SUBGOAL_ATTEMPTS
                {
                    return Err(CoreError::PolicyDenied(
                        "Goal coordination checkpoint widens scope or exceeds attempt limits"
                            .into(),
                    ));
                }
                let requires_attempt = matches!(
                    progress.state,
                    SubgoalState::Running
                        | SubgoalState::RepairRequired { .. }
                        | SubgoalState::Succeeded { .. }
                        | SubgoalState::NeedsDecision {
                            unsettled_effect: true,
                            ..
                        }
                );
                let has_dispatched = requires_attempt || progress.attempts > 0;
                if has_dispatched && progress.attempts == 0 {
                    return Err(CoreError::VerificationFailed(
                        "Dispatched checkpoint subgoal has no attempt receipt".into(),
                    ));
                }
                match &progress.state {
                    SubgoalState::NeedsDecision { question, .. } => bounded_text(
                        question,
                        MAX_SUBGOAL_TEXT_BYTES,
                        "checkpoint decision question",
                    )?,
                    SubgoalState::RepairRequired { failure } => bounded_text(
                        failure,
                        MAX_SUBGOAL_TEXT_BYTES,
                        "checkpoint failure evidence",
                    )?,
                    SubgoalState::Succeeded { evidence_ids }
                        if evidence_ids.is_empty()
                            || evidence_ids.len() > MAX_EVIDENCE_IDS
                            || evidence_ids.iter().any(Uuid::is_nil) =>
                    {
                        return Err(CoreError::VerificationFailed(
                            "Completed checkpoint subgoal has invalid verification evidence".into(),
                        ));
                    }
                    _ => {}
                }
                if matches!(
                    progress.state,
                    SubgoalState::Running
                        | SubgoalState::RepairRequired { .. }
                        | SubgoalState::Succeeded { .. }
                ) && !subgoal.dependencies.iter().all(|dependency| {
                    checkpoint.progress.get(dependency).is_some_and(|progress| {
                        matches!(progress.state, SubgoalState::Succeeded { .. })
                    })
                }) {
                    return Err(CoreError::VerificationFailed(
                        "Checkpoint advanced a subgoal before its dependencies succeeded".into(),
                    ));
                }
            }
        } else if !checkpoint.progress.is_empty() {
            return Err(CoreError::VerificationFailed(
                "Unselected coordination checkpoint contains subgoal progress".into(),
            ));
        }
        Ok(Self {
            plan: checkpoint.plan,
            parent_scope: checkpoint.parent_scope,
            selected_branch: checkpoint.selected_branch,
            progress: checkpoint.progress,
            parallelism: checkpoint.parallelism,
            repairs_remaining: checkpoint.repairs_remaining,
        })
    }

    pub fn candidate_branch_ids(&self) -> impl Iterator<Item = &str> {
        self.plan.candidates.iter().map(|branch| branch.id.as_str())
    }

    /// Select a single branch before dispatch. Candidate branches remain
    /// suggestions until an explicit caller decision chooses one.
    pub fn select_branch(&mut self, branch_id: &str) -> CoreResult<()> {
        if self.selected_branch.is_some() {
            return Err(CoreError::InvalidAction(
                "A coordination branch was already selected".into(),
            ));
        }
        let branch = self
            .plan
            .candidates
            .iter()
            .find(|branch| branch.id == branch_id)
            .ok_or_else(|| CoreError::InvalidAction("Unknown candidate branch".into()))?;
        self.progress = branch
            .subgoals
            .iter()
            .map(|subgoal| {
                (
                    subgoal.id.clone(),
                    SubgoalProgress {
                        instruction: subgoal.instruction.clone(),
                        scope: subgoal.scope.clone(),
                        attempts: 0,
                        state: SubgoalState::Planned,
                    },
                )
            })
            .collect();
        self.selected_branch = Some(branch_id.to_owned());
        Ok(())
    }

    /// Return dependency-ready proposals bounded by currently admitted
    /// concurrency. This function does not authorize or dispatch them.
    pub fn ready_subgoals(&self) -> Vec<ReadySubgoal> {
        let Some(branch_id) = self.selected_branch.as_deref() else {
            return Vec::new();
        };
        let branch = self
            .plan
            .candidates
            .iter()
            .find(|branch| branch.id == branch_id)
            .expect("selected from validated candidate list");
        let running = self
            .progress
            .values()
            .filter(|progress| progress.state == SubgoalState::Running)
            .count();
        let capacity = usize::from(self.parallelism).saturating_sub(running);
        if capacity == 0 {
            return Vec::new();
        }
        let mut reserved_scopes = branch
            .subgoals
            .iter()
            .filter_map(|subgoal| {
                let progress = self.progress.get(&subgoal.id)?;
                (progress.state == SubgoalState::Running).then_some(progress.scope.clone())
            })
            .collect::<Vec<_>>();
        let mut proposals = Vec::new();
        for subgoal in &branch.subgoals {
            let Some(progress) = self.progress.get(&subgoal.id) else {
                continue;
            };
            if progress.state != SubgoalState::Planned
                || !subgoal.dependencies.iter().all(|dependency| {
                    self.progress.get(dependency).is_some_and(|progress| {
                        matches!(progress.state, SubgoalState::Succeeded { .. })
                    })
                })
                || reserved_scopes
                    .iter()
                    .any(|scope| progress.scope.conflicts_with(scope))
            {
                continue;
            }
            reserved_scopes.push(progress.scope.clone());
            proposals.push(ReadySubgoal {
                id: subgoal.id.clone(),
                specialist: subgoal.specialist,
                instruction: progress.instruction.clone(),
                success_condition: subgoal.success_condition.clone(),
                scope: progress.scope.clone(),
            });
            if proposals.len() == capacity {
                break;
            }
        }
        proposals
    }

    /// Record broker-confirmed dispatch. It is an accounting transition, not
    /// a mechanism for obtaining permission or dispatching an action.
    pub fn mark_dispatched(&mut self, subgoal_id: &str) -> CoreResult<()> {
        if !self
            .ready_subgoals()
            .iter()
            .any(|ready| ready.id == subgoal_id)
        {
            return Err(CoreError::PermissionRequired(
                "Subgoal is not currently ready for broker dispatch".into(),
            ));
        }
        let progress = self
            .progress
            .get_mut(subgoal_id)
            .expect("ready subgoals have progress entries");
        if progress.attempts >= MAX_SUBGOAL_ATTEMPTS {
            return Err(CoreError::Busy(
                "Subgoal reached its bounded attempt limit".into(),
            ));
        }
        progress.attempts += 1;
        progress.state = SubgoalState::Running;
        Ok(())
    }

    /// Record only evidence IDs already accepted by Sage's independent
    /// verifier. An empty or repeated evidence set cannot complete a subgoal.
    pub fn record_verified(
        &mut self,
        subgoal_id: &str,
        evidence_ids: BTreeSet<Uuid>,
    ) -> CoreResult<()> {
        if evidence_ids.is_empty()
            || evidence_ids.len() > MAX_EVIDENCE_IDS
            || evidence_ids.iter().any(Uuid::is_nil)
        {
            return Err(CoreError::VerificationFailed(
                "Delegated subgoal requires bounded, non-empty verification evidence".into(),
            ));
        }
        let progress = self
            .progress
            .get_mut(subgoal_id)
            .ok_or_else(|| CoreError::InvalidAction("Unknown delegated subgoal".into()))?;
        if progress.state != SubgoalState::Running {
            return Err(CoreError::VerificationFailed(
                "Only a dispatched subgoal can accept verification evidence".into(),
            ));
        }
        progress.state = SubgoalState::Succeeded { evidence_ids };
        Ok(())
    }

    /// Uncertain dispatched effects always pause for a decision, since a retry
    /// could duplicate a mutation. Settled failures may request a bounded
    /// repair proposal under the existing or narrower scope.
    pub fn record_failure(
        &mut self,
        subgoal_id: &str,
        failure: &str,
        effect_settled: bool,
    ) -> CoreResult<FailureDisposition> {
        bounded_text(
            failure,
            MAX_SUBGOAL_TEXT_BYTES,
            "delegated failure evidence",
        )?;
        let progress = self
            .progress
            .get_mut(subgoal_id)
            .ok_or_else(|| CoreError::InvalidAction("Unknown delegated subgoal".into()))?;
        if progress.state != SubgoalState::Running {
            return Err(CoreError::InvalidAction(
                "Only a dispatched subgoal can record a failure".into(),
            ));
        }
        if !effect_settled
            || self.repairs_remaining == 0
            || progress.attempts >= MAX_SUBGOAL_ATTEMPTS
        {
            let question = if effect_settled {
                format!("The subgoal failed: {failure}. Choose whether to stop or revise the plan.")
            } else {
                format!(
                    "The subgoal outcome is uncertain: {failure}. Reconcile its effect before continuing."
                )
            };
            progress.state = SubgoalState::NeedsDecision {
                question: question.clone(),
                unsettled_effect: !effect_settled,
            };
            return Ok(FailureDisposition::DecisionRequired { question });
        }
        self.repairs_remaining -= 1;
        progress.state = SubgoalState::RepairRequired {
            failure: failure.to_owned(),
        };
        Ok(FailureDisposition::RepairRequired {
            repairs_remaining: self.repairs_remaining,
        })
    }

    /// Accept a revised subgoal only when it preserves the assigned specialist
    /// and dependencies and narrows (or preserves) all proposal constraints.
    pub fn accept_repair(
        &mut self,
        subgoal_id: &str,
        instruction: &str,
        scope: DelegationScope,
    ) -> CoreResult<()> {
        bounded_text(
            instruction,
            MAX_SUBGOAL_TEXT_BYTES,
            "revised delegated instruction",
        )?;
        scope.validate()?;
        let original_scope = self
            .selected_branch()
            .and_then(|branch| {
                branch
                    .subgoals
                    .iter()
                    .find(|subgoal| subgoal.id == subgoal_id)
            })
            .map(|subgoal| subgoal.scope.clone())
            .ok_or_else(|| CoreError::InvalidAction("Unknown delegated subgoal".into()))?;
        let progress = self
            .progress
            .get_mut(subgoal_id)
            .ok_or_else(|| CoreError::InvalidAction("Unknown delegated subgoal".into()))?;
        if !matches!(progress.state, SubgoalState::RepairRequired { .. })
            || !scope.is_attenuated_by(&progress.scope)
            || !scope.is_attenuated_by(&self.parent_scope)
            || !progress.scope.is_attenuated_by(&original_scope)
        {
            return Err(CoreError::PolicyDenied(
                "Repair is not pending or its proposal scope widens prior limits".into(),
            ));
        }
        progress.scope = scope;
        progress.instruction = instruction.to_owned();
        progress.state = SubgoalState::Planned;
        Ok(())
    }

    /// Ask for a user decision on a currently blocked subgoal, for example
    /// when required context or an acceptable route is missing.
    pub fn require_decision(&mut self, subgoal_id: &str, question: &str) -> CoreResult<()> {
        bounded_text(question, MAX_SUBGOAL_TEXT_BYTES, "delegated user decision")?;
        let progress = self
            .progress
            .get_mut(subgoal_id)
            .ok_or_else(|| CoreError::InvalidAction("Unknown delegated subgoal".into()))?;
        if !matches!(
            progress.state,
            SubgoalState::Planned | SubgoalState::RepairRequired { .. }
        ) {
            return Err(CoreError::InvalidAction(
                "A running or settled subgoal cannot be replaced by a pending decision".into(),
            ));
        }
        progress.state = SubgoalState::NeedsDecision {
            question: question.to_owned(),
            unsettled_effect: false,
        };
        Ok(())
    }

    pub fn pending_decisions(&self) -> Vec<(String, String)> {
        self.progress
            .iter()
            .filter_map(|(id, progress)| match &progress.state {
                SubgoalState::NeedsDecision { question, .. } => {
                    Some((id.clone(), question.clone()))
                }
                _ => None,
            })
            .collect()
    }

    pub fn pending_repairs(&self) -> Vec<(String, String)> {
        self.progress
            .iter()
            .filter_map(|(id, progress)| match &progress.state {
                SubgoalState::RepairRequired { failure } => Some((id.clone(), failure.clone())),
                _ => None,
            })
            .collect()
    }

    pub fn verified_evidence(&self, subgoal_id: &str) -> Option<&BTreeSet<Uuid>> {
        match &self.progress.get(subgoal_id)?.state {
            SubgoalState::Succeeded { evidence_ids } => Some(evidence_ids),
            _ => None,
        }
    }

    pub fn is_complete(&self) -> bool {
        !self.progress.is_empty()
            && self
                .progress
                .values()
                .all(|progress| matches!(progress.state, SubgoalState::Succeeded { .. }))
    }

    fn selected_branch(&self) -> Option<&CandidateBranch> {
        let selected = self.selected_branch.as_deref()?;
        self.plan
            .candidates
            .iter()
            .find(|branch| branch.id == selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> DelegationScope {
        DelegationScope {
            capability_ids: BTreeSet::from(["sage.app.inspect".into(), "sage.app.toggle".into()]),
            system_ids: BTreeSet::from([Uuid::from_u128(1)]),
            effects: BTreeSet::from([Effect::Read, Effect::ControlApplication]),
            disclosure_scopes: BTreeSet::from(["local-observation".into()]),
            maximum_calls: 8,
            maximum_output_bytes: 4096,
        }
    }

    fn subgoal(id: &str, specialist: LogicalSpecialist, dependencies: &[&str]) -> DelegatedSubgoal {
        let mut delegated_scope = scope();
        delegated_scope.maximum_calls = 4;
        delegated_scope.maximum_output_bytes = 2048;
        DelegatedSubgoal {
            id: id.into(),
            specialist,
            instruction: format!("Complete {id}"),
            success_condition: format!("Verify {id}"),
            dependencies: dependencies.iter().map(|value| (*value).into()).collect(),
            scope: delegated_scope,
        }
    }

    fn plan() -> CoordinationPlan {
        CoordinationPlan {
            schema_version: 1,
            goal_id: "goal-1".into(),
            candidates: vec![CandidateBranch {
                id: "branch-a".into(),
                rationale: "Use the current local app controls".into(),
                subgoals: vec![
                    subgoal("inspect", LogicalSpecialist::Planner, &[]),
                    subgoal("verify", LogicalSpecialist::Verifier, &["inspect"]),
                ],
            }],
        }
    }

    #[test]
    fn coordinator_releases_dependency_waves_and_requires_verified_evidence() {
        let mut coordinator = GoalCoordinator::new(plan(), scope(), 2, 1).unwrap();
        assert_eq!(
            coordinator.candidate_branch_ids().collect::<Vec<_>>(),
            ["branch-a"]
        );
        assert!(coordinator.ready_subgoals().is_empty());
        coordinator.select_branch("branch-a").unwrap();
        assert_eq!(
            coordinator
                .ready_subgoals()
                .iter()
                .map(|subgoal| subgoal.id.as_str())
                .collect::<Vec<_>>(),
            ["inspect"]
        );
        coordinator.mark_dispatched("inspect").unwrap();
        assert!(coordinator.ready_subgoals().is_empty());
        assert!(
            coordinator
                .record_verified("inspect", BTreeSet::new())
                .is_err()
        );
        coordinator
            .record_verified("inspect", BTreeSet::from([Uuid::from_u128(21)]))
            .unwrap();
        assert!(coordinator.verified_evidence("inspect").is_some());
        assert_eq!(coordinator.ready_subgoals()[0].id, "verify");
        coordinator.mark_dispatched("verify").unwrap();
        coordinator
            .record_verified("verify", BTreeSet::from([Uuid::from_u128(22)]))
            .unwrap();
        assert!(coordinator.is_complete());
    }

    #[test]
    fn plan_rejects_any_scope_escalation_and_cyclic_dependencies() {
        let parent = scope();
        let mut widened = plan();
        widened.candidates[0].subgoals[0]
            .scope
            .effects
            .insert(Effect::Delete);
        assert!(widened.validate(&parent).is_err());

        let mut cyclic = plan();
        cyclic.candidates[0].subgoals[0]
            .dependencies
            .insert("verify".into());
        assert!(cyclic.validate(&parent).is_err());

        let mut excessive_budget = plan();
        excessive_budget.candidates[0].subgoals.push(subgoal(
            "extra",
            LogicalSpecialist::Planner,
            &[],
        ));
        assert!(excessive_budget.validate(&parent).is_err());
    }

    #[test]
    fn uncertain_effect_pauses_and_settled_failure_allows_only_narrower_repair() {
        let mut coordinator = GoalCoordinator::new(plan(), scope(), 2, 1).unwrap();
        coordinator.select_branch("branch-a").unwrap();
        coordinator.mark_dispatched("inspect").unwrap();
        assert!(matches!(
            coordinator
                .record_failure("inspect", "Receipt timed out", false)
                .unwrap(),
            FailureDisposition::DecisionRequired { .. }
        ));
        assert_eq!(coordinator.pending_decisions().len(), 1);

        let mut coordinator = GoalCoordinator::new(plan(), scope(), 2, 1).unwrap();
        coordinator.select_branch("branch-a").unwrap();
        coordinator.mark_dispatched("inspect").unwrap();
        assert!(matches!(
            coordinator
                .record_failure("inspect", "Readback failed", true)
                .unwrap(),
            FailureDisposition::RepairRequired {
                repairs_remaining: 0
            }
        ));
        let mut wider = scope();
        wider.maximum_calls += 1;
        assert!(
            coordinator
                .accept_repair("inspect", "Retry with same target", wider)
                .is_err()
        );
        let mut narrower = scope();
        narrower.maximum_calls = 1;
        narrower.maximum_output_bytes = 1024;
        coordinator
            .accept_repair(
                "inspect",
                "Use the same target with one bounded read",
                narrower,
            )
            .unwrap();
        assert_eq!(coordinator.ready_subgoals()[0].scope.maximum_calls, 1);
    }

    #[test]
    fn conflicting_effects_serialize_on_the_same_system_but_independent_reads_overlap() {
        let mut read_scope = scope();
        read_scope.effects = BTreeSet::from([Effect::Read]);
        read_scope.maximum_calls = 4;
        read_scope.maximum_output_bytes = 2048;
        let mut write_scope = scope();
        write_scope.effects = BTreeSet::from([Effect::ControlApplication]);
        write_scope.maximum_calls = 4;
        write_scope.maximum_output_bytes = 2048;
        let mut conflicting = plan();
        conflicting.candidates[0].subgoals = vec![
            subgoal("read", LogicalSpecialist::Planner, &[]),
            subgoal("mutate", LogicalSpecialist::Verifier, &[]),
        ];
        conflicting.candidates[0].subgoals[0].scope = read_scope.clone();
        conflicting.candidates[0].subgoals[1].scope = write_scope;
        let mut coordinator = GoalCoordinator::new(conflicting, scope(), 2, 0).unwrap();
        coordinator.select_branch("branch-a").unwrap();
        assert_eq!(coordinator.ready_subgoals().len(), 1);
        coordinator.mark_dispatched("read").unwrap();
        assert!(coordinator.ready_subgoals().is_empty());
        coordinator
            .record_verified("read", BTreeSet::from([Uuid::from_u128(23)]))
            .unwrap();
        assert_eq!(coordinator.ready_subgoals()[0].id, "mutate");

        let mut independent_reads = plan();
        independent_reads.candidates[0].subgoals = vec![
            subgoal("read-a", LogicalSpecialist::Planner, &[]),
            subgoal("read-b", LogicalSpecialist::Verifier, &[]),
        ];
        independent_reads.candidates[0].subgoals[0].scope = read_scope.clone();
        independent_reads.candidates[0].subgoals[1].scope = read_scope;
        let mut coordinator = GoalCoordinator::new(independent_reads, scope(), 2, 0).unwrap();
        coordinator.select_branch("branch-a").unwrap();
        assert_eq!(coordinator.ready_subgoals().len(), 2);
    }

    #[test]
    fn plan_and_runtime_limits_are_hard_bounds() {
        let mut many = plan();
        many.candidates.extend([
            CandidateBranch {
                id: "branch-b".into(),
                rationale: "Alternate local path".into(),
                subgoals: vec![subgoal("only-b", LogicalSpecialist::Planner, &[])],
            },
            CandidateBranch {
                id: "branch-c".into(),
                rationale: "Second alternate local path".into(),
                subgoals: vec![subgoal("only-c", LogicalSpecialist::Planner, &[])],
            },
        ]);
        assert!(GoalCoordinator::new(many.clone(), scope(), 2, 0).is_ok());
        many.candidates.push(CandidateBranch {
            id: "branch-d".into(),
            rationale: "Over-bound path".into(),
            subgoals: vec![subgoal("only-d", LogicalSpecialist::Planner, &[])],
        });
        assert!(GoalCoordinator::new(many, scope(), 2, 0).is_err());
        assert!(GoalCoordinator::new(plan(), scope(), 3, 0).is_err());
    }

    #[test]
    fn serialized_checkpoint_restores_dispatched_work_as_fenced_until_reconciliation() {
        let mut coordinator = GoalCoordinator::new(plan(), scope(), 2, 1).unwrap();
        coordinator.select_branch("branch-a").unwrap();
        coordinator.mark_dispatched("inspect").unwrap();
        let bytes = serde_json::to_vec(&coordinator.checkpoint()).unwrap();
        let checkpoint: GoalCoordinationCheckpoint = serde_json::from_slice(&bytes).unwrap();
        checkpoint.validate().unwrap();
        let mut restored = GoalCoordinator::restore(checkpoint).unwrap();
        assert!(restored.ready_subgoals().is_empty());
        assert!(matches!(
            restored
                .record_failure("inspect", "Receipt has not settled", false)
                .unwrap(),
            FailureDisposition::DecisionRequired { .. }
        ));
        let paused = restored.checkpoint();
        assert!(paused.has_unsettled_effects());
        assert!(paused.has_pending_obligations());
    }

    #[test]
    fn task_checkpoint_keeps_coordination_and_rejects_forged_settlement() {
        let mut coordinator = GoalCoordinator::new(plan(), scope(), 2, 1).unwrap();
        coordinator.select_branch("branch-a").unwrap();
        coordinator.mark_dispatched("inspect").unwrap();
        let mut task = crate::agency::TaskCheckpoint {
            schema_version: 1,
            task_id: Uuid::from_u128(99),
            intent: "complete a bounded local goal".into(),
            procedure: None,
            goal_coordination: Some(coordinator.checkpoint()),
            procedure_state: BTreeMap::new(),
            artifacts: Vec::new(),
            verified_results: Vec::new(),
            receipt_ids: Vec::new(),
            pending_obligations: vec![crate::agency::PendingObligation {
                id: "goal-inspect-settlement".into(),
                kind: "effect_settlement".into(),
                summary: "Reconcile the dispatched inspect result".into(),
            }],
            state: crate::agency::CheckpointState::Paused,
            execution_owner: crate::agency::ExecutionOwner {
                device_id: Uuid::from_u128(98),
                generation: 1,
            },
            dispatched_effects_settled: false,
        };
        task.validate().unwrap();
        let encoded = serde_json::to_value(&task).unwrap();
        let restored: crate::agency::TaskCheckpoint = serde_json::from_value(encoded).unwrap();
        restored.validate().unwrap();

        task.dispatched_effects_settled = true;
        assert!(task.validate().is_err());
        task.dispatched_effects_settled = false;
        task.pending_obligations.clear();
        assert!(task.validate().is_err());
    }
}
