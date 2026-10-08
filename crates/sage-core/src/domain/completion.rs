//! Execution truth is derived from broker evidence, never from model prose.
use serde::{Deserialize, Serialize};

use super::{ActionStatus, Task, TaskStatus};
use crate::contracts::Verdict;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionFacts {
    pub verified: u32,
    pub failed: u32,
    pub pending: u32,
    pub uncertain: u32,
    #[serde(default)]
    pub undone: u32,
}

impl ExecutionFacts {
    pub fn for_task(task: &Task) -> Self {
        let mut facts = Self::default();
        for (id, action) in task.current_actions() {
            if task.undone_actions.contains(id) {
                facts.undone += 1;
                continue;
            }
            if task.undo.as_ref().is_some_and(|undo| {
                undo.action_id == *id
                    && matches!(
                        undo.phase,
                        super::UndoPhase::Dispatched | super::UndoPhase::Uncertain
                    )
            }) {
                facts.uncertain += 1;
                continue;
            }
            // Continuation context can contain another run's results. Only
            // evidence for this run's actual actions contributes to its status.
            let result = task
                .tool_results
                .iter()
                .rev()
                .find(|result| result.action_id == *id);
            match action.status {
                ActionStatus::Uncertain => facts.uncertain += 1,
                ActionStatus::Succeeded => match result.map(|result| result.verdict) {
                    Some(Verdict::Confirmed) => facts.verified += 1,
                    Some(Verdict::Failed) => facts.failed += 1,
                    _ => facts.uncertain += 1,
                },
                ActionStatus::Failed | ActionStatus::Skipped => facts.failed += 1,
                _ => facts.pending += 1,
            }
        }
        facts
    }

    pub fn all_verified(self) -> bool {
        self.verified > 0
            && self.failed == 0
            && self.pending == 0
            && self.uncertain == 0
            && self.undone == 0
    }

    pub fn answer_status(self) -> TaskStatus {
        if self.uncertain > 0 || self.pending > 0 {
            TaskStatus::Interrupted
        } else if self.undone > 0 || (self.failed > 0 && self.verified > 0) {
            TaskStatus::Partial
        } else if self.failed > 0 {
            TaskStatus::Failed
        } else if self.verified > 0 {
            TaskStatus::Succeeded
        } else {
            TaskStatus::Answered
        }
    }

    pub fn summary(self) -> String {
        let mut parts = vec![if self.undone > 0 {
            format!(
                "{} {}",
                self.verified,
                if self.verified == 1 {
                    "action remains verified."
                } else {
                    "actions remain verified."
                }
            )
        } else if self.verified == 0 {
            "No actions were verified.".to_string()
        } else {
            format!(
                "{} {} verified.",
                self.verified,
                if self.verified == 1 {
                    "action"
                } else {
                    "actions"
                }
            )
        }];
        if self.failed > 0 {
            parts.push(format!("{} failed or skipped.", self.failed));
        }
        if self.undone > 0 {
            parts.push(format!("{} undone.", self.undone));
        }
        if self.pending > 0 {
            parts.push(format!("{} unfinished.", self.pending));
        }
        if self.uncertain > 0 {
            parts.push(format!(
                "{} with uncertain effects; review the actual state before retrying.",
                self.uncertain
            ));
        }
        parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_combination_of_failed_or_unresolved_work_is_success() {
        for verified in 0..=2 {
            for failed in 0..=2 {
                for pending in 0..=2 {
                    for uncertain in 0..=2 {
                        let facts = ExecutionFacts {
                            verified,
                            failed,
                            pending,
                            uncertain,
                            undone: 0,
                        };
                        assert_eq!(
                            facts.answer_status() == TaskStatus::Succeeded,
                            verified > 0 && failed == 0 && pending == 0 && uncertain == 0
                        );
                        assert_eq!(
                            facts.answer_status() == TaskStatus::Answered,
                            verified + failed + pending + uncertain == 0
                        );
                    }
                }
            }
        }
    }
}
