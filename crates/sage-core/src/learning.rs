//! Bounded procedural learning from verified user runs. Observations contain
//! typed action templates, never model weights, captured screens or authority.
use std::collections::{BTreeMap, BTreeSet};

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::domain::{Action, ExecutionFacts, Task, TaskStatus};
use crate::intent::{CompiledIntent, IntentStep};
use crate::storage::LocalStore;
use crate::workflows::Skill;
use crate::{CoreError, CoreResult};
use uuid::Uuid;

const MIN_RUNS: u32 = 3;
const MAX_SAMPLES: u32 = 1024;
const MAX_REQUEST_BYTES: usize = 1024;
const MAX_PROCEDURE_BYTES: usize = 64 * 1024;
const MAX_VISIBLE_ROUTINES: usize = 16;
const MAX_ALIASES: usize = 16;
const ROUTINE_LOOKUP_SQL: &str = "SELECT matches.graph_digest,(SELECT p.graph_json FROM procedure_samples p WHERE p.graph_digest=matches.graph_digest ORDER BY p.observed_at DESC,p.task_id DESC LIMIT 1),(SELECT COUNT(*) FROM procedure_samples p WHERE p.graph_digest=matches.graph_digest),r.aliases_json,COALESCE((SELECT value_json FROM settings WHERE key='routine_learning.enabled'),'false')='true',COALESCE((SELECT value_json FROM settings WHERE key='memory.enabled'),'true')='true' FROM (SELECT DISTINCT graph_digest FROM procedure_samples WHERE request_key=?1 LIMIT 2) AS matches LEFT JOIN procedure_reviews AS r ON r.graph_digest=matches.graph_digest";

#[derive(Debug, Serialize)]
pub(crate) struct Routine {
    pub id: String,
    pub requests: Vec<String>,
    pub steps: Vec<String>,
    pub verified_runs: u32,
    pub ready: bool,
    pub enabled: bool,
    pub review_digest: String,
    pub evolution: Option<RoutineEvolution>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RoutineEvolution {
    pub source_request: String,
    pub source_verified_runs: u32,
    pub unchanged_effect_classes: Vec<String>,
    pub added_effect_classes: Vec<String>,
    pub removed_effect_classes: Vec<String>,
}

#[derive(Debug)]
struct ReviewedPredecessor {
    digest: String,
    review_digest: String,
    evolution: RoutineEvolution,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RoutineBranch {
    pub routine_id: String,
    pub requests: Vec<String>,
    pub next_steps: Vec<String>,
    pub verified_runs: u32,
}

#[cfg(test)]
mod tests {
    use super::{
        MIN_RUNS, RoutineBranch, RoutineFamily, contextual_branch_suggestion, digest, families,
        family_review_digest, migrate,
    };
    use crate::{
        domain::Action,
        intent::{CompiledIntent, IntentStep},
    };
    use chrono::Utc;
    use rusqlite::Connection;
    use serde_json::json;
    use std::{collections::BTreeSet, time::Instant};
    use uuid::Uuid;

    fn sample_family() -> RoutineFamily {
        RoutineFamily {
            id: "course-notes-family".into(),
            review_digest: "reviewed-digest".into(),
            shared_steps: vec!["Read the course outline".into()],
            branches: vec![
                RoutineBranch {
                    routine_id: "chemistry".into(),
                    requests: vec!["prepare lab notes for chemistry".into()],
                    next_steps: vec!["Create chemistry-summary.md".into()],
                    verified_runs: 5,
                },
                RoutineBranch {
                    routine_id: "biology".into(),
                    requests: vec!["prepare lab notes for biology".into()],
                    next_steps: vec!["Create biology-summary.md".into()],
                    verified_runs: 4,
                },
            ],
            verified_runs: 9,
            prefix: CompiledIntent { steps: Vec::new() },
            source_task_id: Uuid::nil(),
            source_routines: vec!["biology".into(), "chemistry".into()],
        }
    }

    fn reference(
        kind: &str,
        text: &str,
        observed_at_unix_ms: i64,
    ) -> crate::context::ContextObservation {
        let value = match kind {
            "selection" => json!({"available":true,"reference_kind":kind,"selected_text":text}),
            "page" => {
                json!({"available":true,"reference_kind":kind,"title":"Course notes","url":"https://example.test/notes","page_text":text})
            }
            _ => json!({"available":false,"reference_kind":kind}),
        };
        crate::context::ContextObservation {
            source: if kind == "page" {
                "current_reference_browser".into()
            } else {
                "current_reference_native".into()
            },
            observed_at_unix_ms,
            state: value,
        }
    }

    #[test]
    fn contextual_branch_proposal_uses_fresh_reference_and_returns_only_reviewed_requests() {
        let now = Utc::now().timestamp_millis();
        let family = sample_family();
        let selected = reference("selection", "Chemistry titration report", now);
        let prediction = contextual_branch_suggestion(
            "Prepare lab notes from this selection",
            &selected,
            std::slice::from_ref(&family),
        )
        .expect("fresh selection identifies a branch-specific cue");
        assert_eq!(prediction.requests, ["prepare lab notes for chemistry"]);
        assert!(prediction.detail.contains("fill the composer"));
        assert!(!prediction.detail.contains("Chemistry titration report"));

        let page = reference("page", "The chemistry syllabus", now);
        let prediction = contextual_branch_suggestion(
            "Prepare notes from this page",
            &page,
            std::slice::from_ref(&family),
        )
        .expect("page context can distinguish a branch for a generic request");
        assert_eq!(prediction.requests, ["prepare lab notes for chemistry"]);

        let ambiguous = reference("page", "Chemistry and biology course notes", now);
        let prediction =
            contextual_branch_suggestion("Prepare notes from this page", &ambiguous, &[family])
                .expect("an ambiguous page offers a choice instead of selecting a path");
        assert_eq!(prediction.requests.len(), 2);
        assert!(prediction.detail.contains("Several reviewed paths"));
    }

    #[test]
    fn contextual_branch_proposal_fails_closed_without_an_explicit_fresh_reference() {
        let family = sample_family();
        let now = Utc::now().timestamp_millis();
        let selection = reference("selection", "Chemistry titration report", now);
        assert!(
            contextual_branch_suggestion(
                "Prepare lab notes",
                &selection,
                std::slice::from_ref(&family),
            )
            .is_none()
        );

        let stale = reference(
            "selection",
            "Chemistry titration report",
            now - crate::context::REFERENCE_CONTEXT_TTL_MS - 1,
        );
        assert!(
            contextual_branch_suggestion(
                "Prepare lab notes from this selection",
                &stale,
                std::slice::from_ref(&family),
            )
            .is_none()
        );

        let mut page_from_native = reference("page", "Chemistry syllabus", now);
        page_from_native.source = "current_reference_native".into();
        assert!(
            contextual_branch_suggestion(
                "Prepare notes from this page",
                &page_from_native,
                std::slice::from_ref(&family),
            )
            .is_none()
        );
    }

    #[test]
    fn shared_prefix_review_digest_ignores_observation_counts() {
        let shared = vec!["Read the shared document".to_owned()];
        let branch = RoutineBranch {
            routine_id: "source-a".into(),
            requests: vec!["prepare summary".into()],
            next_steps: vec!["Write the summary".into()],
            verified_runs: 3,
        };
        let before =
            family_review_digest("family", &shared, std::slice::from_ref(&branch)).unwrap();
        let mut more_observations = branch.clone();
        more_observations.verified_runs = 100;
        let after = family_review_digest("family", &shared, &[more_observations]).unwrap();
        assert_eq!(before, after);

        let mut changed_branch = branch;
        changed_branch.next_steps = vec!["Send the summary".into()];
        let changed = family_review_digest("family", &shared, &[changed_branch]).unwrap();
        assert_ne!(before, changed);
    }

    #[test]
    #[ignore = "release-only benchmark for the bounded multi-branch synthesis pass"]
    fn multi_branch_family_aggregation_latency() {
        const ROUTINES: usize = 16;
        const SAMPLES: usize = 1_000;

        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(
            "CREATE TABLE settings(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
             INSERT INTO settings(key,value_json) VALUES('routine_learning.enabled','true'),('memory.enabled','true');
             CREATE TABLE tasks(id TEXT PRIMARY KEY);
             CREATE TABLE retired_task_data(task_id TEXT);",
        )
        .unwrap();
        migrate(&db).unwrap();

        for index in 0..ROUTINES {
            let procedure = CompiledIntent {
                steps: vec![
                    IntentStep {
                        action: Action::ReadFile {
                            path: "/tmp/sage-shared-prefix.txt".into(),
                            max_bytes: 128,
                        },
                        dependencies: BTreeSet::new(),
                    },
                    IntentStep {
                        action: Action::ReadFile {
                            path: format!("/tmp/sage-branch-{index}.txt").into(),
                            max_bytes: 128,
                        },
                        dependencies: BTreeSet::from([0]),
                    },
                ],
            };
            let graph_digest = digest(&procedure).unwrap();
            let graph_json = serde_json::to_string(&procedure).unwrap();
            let request = format!("prepare branch {index}");
            for run in 0..MIN_RUNS {
                let task_id = Uuid::new_v4().to_string();
                db.execute("INSERT INTO tasks(id) VALUES(?1)", [&task_id])
                    .unwrap();
                db.execute(
                    "INSERT INTO procedure_samples(task_id,request_key,graph_digest,graph_json,observed_at) VALUES(?1,?2,?3,?4,?5)",
                    rusqlite::params![task_id, request, graph_digest, graph_json, format!("{run:03}")],
                )
                .unwrap();
            }
            db.execute(
                "INSERT INTO procedure_reviews(graph_digest,aliases_json,reviewed_at) VALUES(?1,?2,'reviewed')",
                rusqlite::params![graph_digest, serde_json::to_string(&vec![request]).unwrap()],
            )
            .unwrap();
        }

        let initial = families(&db).unwrap();
        assert_eq!(initial.len(), 1);
        assert_eq!(initial[0].branches.len(), ROUTINES);

        let mut samples_ns = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let started = Instant::now();
            let result = families(&db).unwrap();
            samples_ns.push(started.elapsed().as_nanos());
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].branches.len(), ROUTINES);
        }
        samples_ns.sort_unstable();
        let percentile = |percent: usize| {
            let index = (samples_ns.len() * percent).div_ceil(100).saturating_sub(1);
            samples_ns[index]
        };
        eprintln!(
            "16-routine in-memory family aggregation ({SAMPLES} samples): p50={} ns p95={} ns p99={} ns",
            percentile(50),
            percentile(95),
            percentile(99)
        );
    }

    #[test]
    #[ignore = "release-only benchmark for explicit-context branch prediction"]
    fn contextual_branch_prediction_latency() {
        const SAMPLES: usize = 1_000;
        const MAX_FAMILIES_FROM_ROUTINE_LIMIT: usize =
            super::MAX_VISIBLE_ROUTINES * (super::MAX_VISIBLE_ROUTINES - 1) / 2;
        let labels = (0..MAX_FAMILIES_FROM_ROUTINE_LIMIT)
            .map(|index| format!("family{index:03}"))
            .collect::<Vec<_>>();
        let families = labels
            .iter()
            .enumerate()
            .map(|(index, label)| RoutineFamily {
                id: format!("family-{index}"),
                review_digest: format!("digest-{index}"),
                shared_steps: vec!["Read the course outline".into()],
                branches: vec![
                    RoutineBranch {
                        routine_id: format!("routine-{index}-a"),
                        requests: vec![format!("prepare {label} chemistry notes")],
                        next_steps: vec![format!("Create {label}-chemistry-summary.md")],
                        verified_runs: 4,
                    },
                    RoutineBranch {
                        routine_id: format!("routine-{index}-b"),
                        requests: vec![format!("review {label} biology notes")],
                        next_steps: vec![format!("Create {label}-biology-review.md")],
                        verified_runs: 3,
                    },
                ],
                verified_runs: 7,
                prefix: CompiledIntent { steps: Vec::new() },
                source_task_id: Uuid::nil(),
                source_routines: vec![format!("routine-{index}-a"), format!("routine-{index}-b")],
            })
            .collect::<Vec<_>>();
        let observation = reference(
            "page",
            &"family000 chemistry ".repeat(375),
            Utc::now().timestamp_millis(),
        );
        let mut samples_ns = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let started = Instant::now();
            let prediction = contextual_branch_suggestion(
                "Prepare notes from this page",
                &observation,
                &families,
            );
            samples_ns.push(started.elapsed().as_nanos());
            assert!(prediction.is_some());
        }
        samples_ns.sort_unstable();
        let percentile = |percent: usize| {
            let index = (samples_ns.len() * percent).div_ceil(100).saturating_sub(1);
            samples_ns[index]
        };
        eprintln!(
            "{}-family contextual branch prediction ({SAMPLES} samples, 7.5 KiB page): p50={} ns p95={} ns p99={} ns",
            MAX_FAMILIES_FROM_ROUTINE_LIMIT,
            percentile(50),
            percentile(95),
            percentile(99)
        );
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct RoutineFamily {
    pub id: String,
    pub review_digest: String,
    pub shared_steps: Vec<String>,
    pub branches: Vec<RoutineBranch>,
    pub verified_runs: u32,
    #[serde(skip)]
    prefix: CompiledIntent,
    #[serde(skip)]
    source_task_id: Uuid,
    #[serde(skip)]
    source_routines: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContextualRoutineSuggestion {
    pub requests: Vec<String>,
    pub detail: String,
}

/// Rank reviewed branch aliases locally against the user's explicit request
/// and its fresh one-shot reference. A result can fill the composer; it cannot
/// start a task or grant any resource access.
pub(crate) fn contextual_branch_suggestion(
    request: &str,
    observation: &crate::context::ContextObservation,
    families: &[RoutineFamily],
) -> Option<ContextualRoutineSuggestion> {
    if families.is_empty() || !crate::context::requests_live_reference(request) {
        return None;
    }
    if !observation.source.starts_with("current_reference_") {
        return None;
    }
    let age = Utc::now()
        .timestamp_millis()
        .saturating_sub(observation.observed_at_unix_ms);
    if !(0..=crate::context::REFERENCE_CONTEXT_TTL_MS).contains(&age) {
        return None;
    }
    let state = observation.state.as_object()?;
    if state.get("available").and_then(serde_json::Value::as_bool) != Some(true) {
        return None;
    }
    let kind = state
        .get("reference_kind")
        .and_then(serde_json::Value::as_str)?;
    let source_matches_kind = matches!(
        (kind, observation.source.as_str()),
        ("page", "current_reference_browser")
            | (
                "selection",
                "current_reference_browser" | "current_reference_native"
            )
            | (
                "file" | "window" | "application",
                "current_reference_native"
            )
    );
    if !source_matches_kind {
        return None;
    }
    let request_terms = prediction_terms(request);
    let reference_text = prediction_reference_text(state, kind)?;
    let reference_terms = prediction_terms(&reference_text);
    if reference_terms.is_empty() {
        return None;
    }

    struct FamilyRank {
        branches: Vec<(usize, String)>,
    }
    let mut ranked_families = Vec::<(usize, String, FamilyRank)>::new();
    for family in families.iter().filter(|family| family.branches.len() >= 2) {
        let branch_terms = family
            .branches
            .iter()
            .map(|branch| {
                prediction_terms(
                    &branch
                        .requests
                        .iter()
                        .chain(&branch.next_steps)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" "),
                )
            })
            .collect::<Vec<_>>();
        let Some(first) = branch_terms.first() else {
            continue;
        };
        let common = branch_terms
            .iter()
            .skip(1)
            .fold(first.clone(), |common, terms| {
                common.intersection(terms).cloned().collect()
            });
        let mut family_terms = prediction_terms(&family.shared_steps.join(" "));
        for terms in &branch_terms {
            family_terms.extend(terms.iter().cloned());
        }
        let family_fit = 3 * overlap_count(&request_terms, &family_terms)
            + 2 * overlap_count(&reference_terms, &family_terms);
        if family_fit == 0 {
            continue;
        }

        let mut branches = Vec::new();
        for (branch, terms) in family.branches.iter().zip(&branch_terms) {
            let distinctive = terms.difference(&common).cloned().collect::<BTreeSet<_>>();
            let score = 3 * overlap_count(&request_terms, &distinctive)
                + 2 * overlap_count(&reference_terms, &distinctive);
            if score == 0 {
                continue;
            }
            if let Some(alias) = branch.requests.first() {
                branches.push((score, alias.clone()));
            }
        }
        if branches.is_empty() {
            continue;
        }
        branches.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
        let score = family_fit + branches[0].0;
        ranked_families.push((score, family.id.clone(), FamilyRank { branches }));
    }
    ranked_families.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    if ranked_families.len() > 1 && ranked_families[0].0 <= ranked_families[1].0 + 1 {
        return None;
    }
    let (_, _, best) = ranked_families.into_iter().next()?;
    let mut requests = vec![best.branches[0].1.clone()];
    if best
        .branches
        .get(1)
        .is_some_and(|second| best.branches[0].0 <= second.0 + 2)
    {
        requests.push(best.branches[1].1.clone());
    }
    let reference_name = match kind {
        "selection" => "selection",
        "page" => "page",
        "file" => "file",
        "window" => "window",
        "application" => "app",
        _ => return None,
    };
    let detail = if requests.len() > 1 {
        format!(
            "Several reviewed paths may fit this {reference_name}. Choose one to fill the composer; nothing is submitted."
        )
    } else {
        format!(
            "A reviewed path may fit this {reference_name}. Choose it to fill the composer; nothing is submitted."
        )
    };
    Some(ContextualRoutineSuggestion { requests, detail })
}

fn prediction_reference_text(
    state: &serde_json::Map<String, serde_json::Value>,
    kind: &str,
) -> Option<String> {
    let fields: &[&str] = match kind {
        "selection" => &[
            "selected_text",
            "active_window",
            "current_resource",
            "application_name",
        ],
        "page" => &["page_text", "selected_text", "title", "url"],
        "file" => &["current_resource", "active_window", "application_name"],
        "window" => &["active_window", "application_name", "current_resource"],
        "application" => &["active_application", "application_name"],
        _ => return None,
    };
    let mut text = String::new();
    for field in fields {
        if let Some(value) = state.get(*field).and_then(serde_json::Value::as_str) {
            text.push(' ');
            text.push_str(&crate::redaction::redact_for_persistence(value));
        }
    }
    (!text.trim().is_empty()).then_some(text)
}

fn prediction_terms(value: &str) -> BTreeSet<String> {
    const STOP_WORDS: &[&str] = &[
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "that",
        "this",
        "use",
        "your",
        "current",
        "active",
        "selection",
        "selected",
        "highlighted",
        "text",
        "page",
        "document",
        "window",
        "application",
        "app",
        "please",
        "based",
        "using",
        "could",
        "would",
        "should",
        "what",
        "when",
        "where",
        "read",
        "write",
        "create",
        "open",
        "copy",
        "move",
        "review",
        "prepare",
        "summarize",
        "summary",
        "notes",
        "note",
        "routine",
        "skill",
        "then",
        "from",
        "into",
        "with",
    ];
    let mut terms = BTreeSet::new();
    for term in value
        .chars()
        .take(16 * 1024)
        .collect::<String>()
        .to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
    {
        if term.chars().count() >= 3 && !STOP_WORDS.contains(&term) {
            terms.insert(term.to_owned());
        }
        if terms.len() >= 512 {
            break;
        }
    }
    terms
}

fn overlap_count(left: &BTreeSet<String>, right: &BTreeSet<String>) -> usize {
    left.intersection(right).count()
}

pub(crate) fn migrate(db: &Connection) -> CoreResult<()> {
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS procedure_samples(
            task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
            request_key TEXT NOT NULL,
            graph_digest TEXT NOT NULL,
            graph_json TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            intent_revision INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS procedure_samples_request
            ON procedure_samples(request_key,graph_digest);
        CREATE INDEX IF NOT EXISTS procedure_samples_graph
            ON procedure_samples(graph_digest,observed_at);
        CREATE TABLE IF NOT EXISTS procedure_reviews(
            graph_digest TEXT PRIMARY KEY,
            aliases_json TEXT NOT NULL,
            reviewed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS procedure_evolution_samples(
            task_id TEXT PRIMARY KEY REFERENCES procedure_samples(task_id) ON DELETE CASCADE,
            candidate_digest TEXT NOT NULL,
            source_digest TEXT NOT NULL,
            request_key TEXT NOT NULL,
            source_review_digest TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS procedure_evolution_source
            ON procedure_evolution_samples(source_digest,candidate_digest);
        CREATE INDEX IF NOT EXISTS procedure_evolution_candidate
            ON procedure_evolution_samples(candidate_digest,request_key);
        CREATE TRIGGER IF NOT EXISTS retire_procedure_sample
            AFTER INSERT ON retired_task_data BEGIN
                DELETE FROM procedure_samples WHERE task_id=NEW.task_id;
            END;
        CREATE TRIGGER IF NOT EXISTS invalidate_procedure_review
            AFTER DELETE ON procedure_samples BEGIN
                DELETE FROM procedure_reviews WHERE graph_digest=OLD.graph_digest;
            END;
        CREATE TRIGGER IF NOT EXISTS invalidate_evolved_samples
            AFTER DELETE ON procedure_reviews BEGIN
                DELETE FROM procedure_samples
                WHERE task_id IN (
                    SELECT task_id FROM procedure_evolution_samples
                    WHERE source_digest=OLD.graph_digest
                );
            END;",
    )?;
    let has_revision: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM pragma_table_info('procedure_samples') WHERE name='intent_revision')",
        [],
        |row| row.get(0),
    )?;
    if !has_revision {
        db.execute_batch(
            "ALTER TABLE procedure_samples ADD COLUMN intent_revision INTEGER NOT NULL DEFAULT 1",
        )?;
    }
    Ok(())
}

/// Only superficial wording variations are folded. Quoted paths retain case
/// and whitespace; negation and additional clauses remain distinct requests.
fn request_key(text: &str) -> Option<String> {
    if text.len() > MAX_REQUEST_BYTES
        || text.chars().any(char::is_control)
        || crate::knowledge::contains_sensitive(text)
    {
        return None;
    }
    let mut result = String::with_capacity(text.len());
    let mut quoted = false;
    let mut space = false;
    for ch in text.trim().chars() {
        if ch == '"' {
            quoted = !quoted;
        }
        if !quoted && ch.is_whitespace() {
            space = !result.is_empty();
            continue;
        }
        if space {
            result.push(' ');
            space = false;
        }
        if quoted {
            result.push(ch);
        } else {
            result.extend(ch.to_lowercase());
        }
    }
    if quoted {
        return None;
    }
    let key = result.strip_prefix("please ").unwrap_or(&result).trim();
    (!key.is_empty()).then(|| key.to_owned())
}

fn supported(action: &Action) -> bool {
    matches!(
        action,
        Action::ReadFile { .. }
            | Action::ListDirectory { cursor: None, .. }
            | Action::CreateFolder { .. }
            | Action::OpenApplication { .. }
    )
}

fn validate(procedure: &CompiledIntent) -> CoreResult<()> {
    if procedure.steps.is_empty()
        || procedure.steps.len() > 32
        || procedure.steps.iter().enumerate().any(|(i, step)| {
            !supported(&step.action) || step.dependencies.iter().any(|parent| *parent >= i)
        })
    {
        return Err(CoreError::InvalidAction("Invalid learned procedure".into()));
    }
    let bytes = serde_json::to_vec(procedure)?;
    if bytes.len() > MAX_PROCEDURE_BYTES
        || crate::knowledge::contains_sensitive(&String::from_utf8_lossy(&bytes))
    {
        return Err(CoreError::InvalidAction(
            "The procedure contains sensitive or oversized data".into(),
        ));
    }
    Ok(())
}

fn digest(procedure: &CompiledIntent) -> CoreResult<String> {
    validate(procedure)?;
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(procedure)?)
    ))
}

/// Canonical topological order ignores run/action UUIDs, prepared metadata and
/// old capabilities. Ambiguous isomorphic DAGs may remain separate patterns;
/// they can never collapse different action parameters into the same routine.
fn procedure(task: &Task) -> CoreResult<CompiledIntent> {
    let mut pending: BTreeSet<_> = task.current_actions().map(|(id, _)| *id).collect();
    let mut resolved = BTreeMap::new();
    let mut steps = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let mut ready = Vec::new();
        for id in &pending {
            let parents = task.dependencies.get(id).cloned().unwrap_or_default();
            if parents.iter().all(|id| resolved.contains_key(id)) {
                ready.push((
                    serde_json::to_string(&task.actions[id].proposal.action)?,
                    *id,
                    parents,
                ));
            }
        }
        ready.sort_by(|left, right| (&left.0, left.1).cmp(&(&right.0, right.1)));
        let Some((_, id, parents)) = ready.into_iter().next() else {
            return Err(CoreError::InvalidAction(
                "Incomplete procedure dependencies".into(),
            ));
        };
        steps.push(IntentStep {
            action: task.actions[&id].proposal.action.clone(),
            dependencies: parents.iter().map(|id| resolved[id]).collect(),
        });
        resolved.insert(id, steps.len() - 1);
        pending.remove(&id);
    }
    let procedure = CompiledIntent { steps };
    validate(&procedure)?;
    Ok(procedure)
}

fn enabled(db: &Connection) -> CoreResult<bool> {
    let setting = |key: &str| -> CoreResult<bool> {
        let value: Option<String> = db
            .query_row(
                "SELECT value_json FROM settings WHERE key=?1",
                [key],
                |row| row.get(0),
            )
            .optional()?;
        Ok(value
            .as_deref()
            .map(|v| serde_json::from_str::<bool>(v).unwrap_or(false))
            .unwrap_or(key == "memory.enabled"))
    };
    Ok(setting("routine_learning.enabled")? && setting("memory.enabled")?)
}

fn effect_classes(procedure: &CompiledIntent) -> BTreeSet<&'static str> {
    procedure
        .steps
        .iter()
        .map(|step| match &step.action {
            Action::ReadFile { .. } => "file.read",
            Action::ListDirectory { .. } => "directory.list",
            Action::CreateFolder { .. } => "folder.create",
            Action::OpenApplication { .. } => "application.open",
            _ => "unsupported",
        })
        .collect()
}

/// Return one currently reviewed procedure that uses this exact request alias.
/// Multiple, stale, oversized, or corrupted sources fail closed.
fn reviewed_predecessor(
    db: &Connection,
    request: &str,
    candidate_digest: &str,
    candidate: &CompiledIntent,
) -> CoreResult<Option<ReviewedPredecessor>> {
    let mut query = db.prepare(&format!(
        "SELECT DISTINCT r.graph_digest,r.aliases_json FROM procedure_reviews r JOIN procedure_samples p ON p.graph_digest=r.graph_digest WHERE p.request_key=?1 AND r.graph_digest<>?2 ORDER BY r.graph_digest LIMIT {}",
        MAX_VISIBLE_ROUTINES + 1
    ))?;
    let sources = query
        .query_map(params![request, candidate_digest], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    drop(query);
    if sources.len() > MAX_VISIBLE_ROUTINES {
        return Ok(None);
    }

    let mut predecessor = None;
    for (source_digest, stored_aliases_json) in sources {
        let stored_aliases: Vec<String> = serde_json::from_str(&stored_aliases_json)
            .map_err(|_| CoreError::Storage("Invalid reviewed routine aliases".into()))?;
        let current_aliases = aliases(db, &source_digest)?;
        let alias_count: u32 = db.query_row(
            "SELECT COUNT(DISTINCT request_key) FROM procedure_samples WHERE graph_digest=?1",
            [&source_digest],
            |row| row.get(0),
        )?;
        if alias_count as usize > MAX_ALIASES
            || current_aliases != stored_aliases
            || !current_aliases.iter().any(|alias| alias == request)
        {
            continue;
        }
        let (source_runs, source_json): (u32, String) = db.query_row(
            "SELECT COUNT(*),(SELECT graph_json FROM procedure_samples WHERE graph_digest=?1 ORDER BY observed_at DESC,task_id DESC LIMIT 1) FROM procedure_samples WHERE graph_digest=?1",
            [&source_digest],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if source_runs < MIN_RUNS {
            continue;
        }
        let source: CompiledIntent = serde_json::from_str(&source_json)?;
        if digest(&source)? != source_digest {
            return Err(CoreError::Storage(
                "Reviewed routine procedure digest mismatch".into(),
            ));
        }
        let source_effects = effect_classes(&source);
        let candidate_effects = effect_classes(candidate);
        let same = source_effects
            .intersection(&candidate_effects)
            .map(|effect| (*effect).to_owned())
            .collect();
        let added = candidate_effects
            .difference(&source_effects)
            .map(|effect| (*effect).to_owned())
            .collect();
        let removed = source_effects
            .difference(&candidate_effects)
            .map(|effect| (*effect).to_owned())
            .collect();
        let current = ReviewedPredecessor {
            digest: source_digest.clone(),
            review_digest: review_digest(&source_digest, &current_aliases)?,
            evolution: RoutineEvolution {
                source_request: request.to_owned(),
                source_verified_runs: source_runs,
                unchanged_effect_classes: same,
                added_effect_classes: added,
                removed_effect_classes: removed,
            },
        };
        if predecessor.replace(current).is_some() {
            return Ok(None);
        }
    }
    Ok(predecessor)
}

fn candidate_evolution(
    db: &Connection,
    candidate_digest: &str,
    candidate: &CompiledIntent,
    current_requests: &[String],
) -> CoreResult<Option<RoutineEvolution>> {
    let mut query = db.prepare(&format!(
        "SELECT DISTINCT request_key,source_digest,source_review_digest FROM procedure_evolution_samples WHERE candidate_digest=?1 ORDER BY request_key,source_digest LIMIT {}",
        MAX_VISIBLE_ROUTINES + 1
    ))?;
    let sources = query
        .query_map([candidate_digest], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    drop(query);
    if sources.is_empty() || sources.len() > MAX_VISIBLE_ROUTINES {
        return Ok(None);
    }

    let mut selected: Option<(String, String, String, RoutineEvolution)> = None;
    for (request, source_digest, source_review_digest) in sources {
        if !current_requests.iter().any(|alias| alias == &request) {
            continue;
        }
        let Some(predecessor) = reviewed_predecessor(db, &request, candidate_digest, candidate)?
        else {
            continue;
        };
        if predecessor.digest != source_digest || predecessor.review_digest != source_review_digest
        {
            continue;
        }
        let relation = (
            request,
            source_digest,
            source_review_digest,
            predecessor.evolution,
        );
        if let Some((selected_request, selected_source, selected_review, _)) = &selected {
            if selected_request != &relation.0
                || selected_source != &relation.1
                || selected_review != &relation.2
            {
                return Ok(None);
            }
        } else {
            selected = Some(relation);
        }
    }
    Ok(selected.map(|(_, _, _, evolution)| evolution))
}

/// Runs inside outcome finalization. A failed/undone run withdraws its sample;
/// retries of the same finalization cannot count as additional observations.
pub(crate) fn record_outcome(db: &Transaction<'_>, task: &Task) -> CoreResult<()> {
    let intent_revision = task.intent.as_ref().map_or(1, |intent| intent.revision);
    let corrected = intent_revision > 1;
    if task.status != TaskStatus::Succeeded
        || !ExecutionFacts::for_task(task).all_verified()
        || task.undo.is_some()
        || crate::knowledge::outcome_retired(db, task.id)?
    {
        db.execute(
            "DELETE FROM procedure_samples WHERE task_id=?1",
            [task.id.to_string()],
        )?;
        if let Some(routine) = &task.compiled_routine {
            db.execute(
                "DELETE FROM procedure_reviews WHERE graph_digest=?1",
                [routine],
            )?;
        }
        return Ok(());
    }
    // A routine's own executions and scheduled work cannot reinforce its
    // confidence. Only independently requested and verified examples count.
    if !enabled(db)? || task.background || task.compiled_routine.is_some() {
        if corrected {
            db.execute(
                "DELETE FROM procedure_samples WHERE task_id=?1",
                [task.id.to_string()],
            )?;
        }
        return Ok(());
    }
    let Some(key) = request_key(&task.request) else {
        if corrected {
            db.execute(
                "DELETE FROM procedure_samples WHERE task_id=?1",
                [task.id.to_string()],
            )?;
        }
        return Ok(());
    };
    let Ok(procedure) = procedure(task) else {
        if corrected {
            db.execute(
                "DELETE FROM procedure_samples WHERE task_id=?1",
                [task.id.to_string()],
            )?;
        }
        return Ok(());
    };
    for (_, action) in task.current_actions() {
        let confirmed: bool = db.query_row("SELECT EXISTS(SELECT 1 FROM action_journal WHERE run_id=?1 AND action_id=?2 AND action_digest=?3 AND state='confirmed' AND verification_json IS NOT NULL)", params![task.id.to_string(),action.proposal.id.to_string(),crate::policy::approval_digest(&action.proposal)?], |row| row.get(0))?;
        if !confirmed {
            if corrected {
                db.execute(
                    "DELETE FROM procedure_samples WHERE task_id=?1",
                    [task.id.to_string()],
                )?;
            }
            return Ok(());
        }
    }
    let candidate_digest = digest(&procedure)?;
    let revision = i64::try_from(intent_revision)
        .map_err(|_| CoreError::Storage("Routine intent revision is out of range".into()))?;
    let previous: Option<(String, String, i64)> = db
        .query_row(
            "SELECT request_key,graph_digest,intent_revision FROM procedure_samples WHERE task_id=?1",
            [task.id.to_string()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    if let Some((previous_key, previous_digest, previous_revision)) = previous {
        if previous_key == key
            && previous_digest == candidate_digest
            && previous_revision == revision
        {
            return Ok(());
        }
        if corrected {
            db.execute(
                "DELETE FROM procedure_samples WHERE task_id=?1",
                [task.id.to_string()],
            )?;
        }
    }
    let predecessor = if corrected {
        let Some(predecessor) = reviewed_predecessor(db, &key, &candidate_digest, &procedure)?
        else {
            return Ok(());
        };
        Some(predecessor)
    } else {
        None
    };
    db.execute(
        "INSERT OR IGNORE INTO procedure_samples(task_id,request_key,graph_digest,graph_json,observed_at,intent_revision) VALUES(?1,?2,?3,?4,?5,?6)",
        params![
            task.id.to_string(),
            key,
            candidate_digest,
            serde_json::to_string(&procedure)?,
            Utc::now().to_rfc3339(),
            revision
        ],
    )?;
    if let Some(predecessor) = predecessor {
        db.execute(
            "INSERT OR IGNORE INTO procedure_evolution_samples(task_id,candidate_digest,source_digest,request_key,source_review_digest,created_at) VALUES(?1,?2,?3,?4,?5,?6)",
            params![
                task.id.to_string(),
                candidate_digest,
                predecessor.digest,
                key,
                predecessor.review_digest,
                Utc::now().to_rfc3339(),
            ],
        )?;
    }
    db.execute("DELETE FROM procedure_samples WHERE task_id IN (SELECT task_id FROM procedure_samples ORDER BY observed_at DESC,task_id DESC LIMIT -1 OFFSET ?1)", [MAX_SAMPLES])?;
    Ok(())
}

fn aliases(db: &Connection, id: &str) -> CoreResult<Vec<String>> {
    let mut query = db.prepare(&format!("SELECT DISTINCT request_key FROM procedure_samples WHERE graph_digest=?1 ORDER BY request_key LIMIT {MAX_ALIASES}"))?;
    Ok(query
        .query_map([id], |row| row.get(0))?
        .collect::<Result<_, _>>()?)
}

fn review_digest(id: &str, requests: &[String]) -> CoreResult<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(id, requests))?)
    ))
}

fn candidates(db: &Connection) -> CoreResult<Vec<Routine>> {
    let learning_enabled = enabled(db)?;
    let mut query = db.prepare(&format!("SELECT graph_digest,COUNT(*),MAX(observed_at) FROM procedure_samples GROUP BY graph_digest ORDER BY COUNT(*) DESC,MAX(observed_at) DESC LIMIT {MAX_VISIBLE_ROUTINES}"))?;
    let rows = query
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    drop(query);
    let mut routines = Vec::new();
    for (id, count) in rows {
        let json: String = db.query_row("SELECT graph_json FROM procedure_samples WHERE graph_digest=?1 ORDER BY observed_at DESC,task_id DESC LIMIT 1", [&id], |row| row.get(0))?;
        let procedure: CompiledIntent = serde_json::from_str(&json)?;
        if digest(&procedure)? != id {
            return Err(CoreError::Storage(
                "Learned procedure digest mismatch".into(),
            ));
        }
        let requests = aliases(db, &id)?;
        let alias_count: u32 = db.query_row(
            "SELECT COUNT(DISTINCT request_key) FROM procedure_samples WHERE graph_digest=?1",
            [&id],
            |row| row.get(0),
        )?;
        let review: Option<String> = db
            .query_row(
                "SELECT aliases_json FROM procedure_reviews WHERE graph_digest=?1",
                [&id],
                |row| row.get(0),
            )
            .optional()?;
        let reviewed = (alias_count as usize <= MAX_ALIASES)
            && review
                .as_deref()
                .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                .is_some_and(|approved| approved == requests);
        let ready = count >= MIN_RUNS;
        let evolution = candidate_evolution(db, &id, &procedure, &requests)?;
        routines.push(Routine {
            review_digest: review_digest(&id, &requests)?,
            id,
            requests,
            steps: procedure.summaries(),
            verified_runs: count,
            ready,
            enabled: learning_enabled && ready && reviewed,
            evolution,
        });
    }
    Ok(routines)
}

fn family_digest(value: &impl Serialize) -> CoreResult<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn family_review_digest(
    id: &str,
    shared_steps: &[String],
    branches: &[RoutineBranch],
) -> CoreResult<String> {
    let review_branches = branches
        .iter()
        .map(|branch| (&branch.routine_id, &branch.requests, &branch.next_steps))
        .collect::<Vec<_>>();
    family_digest(&(id, shared_steps, review_branches))
}

/// A family is only synthesized from separately reviewed, unambiguous exact
/// request branches. Shared actions and dependency edges must match exactly.
fn families(db: &Connection) -> CoreResult<Vec<RoutineFamily>> {
    let mut patterns = Vec::new();
    for routine in candidates(db)?
        .into_iter()
        .filter(|routine| routine.enabled)
    {
        let (task, body): (String, String) = db.query_row(
            "SELECT task_id,graph_json FROM procedure_samples WHERE graph_digest=?1 ORDER BY observed_at DESC,task_id DESC LIMIT 1",
            [&routine.id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let procedure: CompiledIntent = serde_json::from_str(&body)?;
        if digest(&procedure)? != routine.id {
            return Err(CoreError::Storage(
                "Learned procedure digest mismatch".into(),
            ));
        }
        patterns.push((
            routine,
            procedure,
            Uuid::parse_str(&task)
                .map_err(|_| CoreError::Storage("Invalid routine source task identity".into()))?,
        ));
    }
    patterns.sort_by(|left, right| left.0.id.cmp(&right.0.id));

    let mut result = Vec::new();
    let mut seen_families = BTreeSet::new();
    for left in 0..patterns.len() {
        for right in (left + 1)..patterns.len() {
            let (left_routine, left_procedure, _) = &patterns[left];
            let (right_routine, right_procedure, _) = &patterns[right];
            if requests_overlap(&left_routine.requests, &right_routine.requests) {
                continue;
            }
            let shared = left_procedure
                .steps
                .iter()
                .zip(&right_procedure.steps)
                .take_while(|(left, right)| {
                    left.action == right.action && left.dependencies == right.dependencies
                })
                .map(|(step, _)| step.clone())
                .collect::<Vec<_>>();
            if shared.is_empty() {
                continue;
            }
            let prefix = CompiledIntent { steps: shared };
            validate(&prefix)?;

            // Start from a distinct, unambiguous pair, then add every other
            // reviewed routine with this exact prefix whose request aliases
            // remain disjoint from all branches already in the family.
            let mut members = vec![left, right];
            for candidate in 0..patterns.len() {
                if members.contains(&candidate)
                    || !has_prefix(&patterns[candidate].1, &prefix)
                    || members.iter().any(|member| {
                        requests_overlap(
                            &patterns[*member].0.requests,
                            &patterns[candidate].0.requests,
                        )
                    })
                {
                    continue;
                }
                members.push(candidate);
            }
            members.sort_by(|left, right| patterns[*left].0.id.cmp(&patterns[*right].0.id));
            let source_routines = members
                .iter()
                .map(|index| patterns[*index].0.id.clone())
                .collect::<Vec<_>>();
            let id = family_digest(&("routine-prefix-v1", &prefix, &source_routines))?;
            if !seen_families.insert(id.clone()) {
                continue;
            }
            let mut branches = members
                .iter()
                .map(|index| {
                    let (routine, procedure, _) = &patterns[*index];
                    RoutineBranch {
                        routine_id: routine.id.clone(),
                        requests: routine.requests.clone(),
                        next_steps: CompiledIntent {
                            steps: procedure.steps[prefix.steps.len()..].to_vec(),
                        }
                        .summaries(),
                        verified_runs: routine.verified_runs,
                    }
                })
                .collect::<Vec<_>>();
            branches.sort_by(|left, right| left.requests.cmp(&right.requests));
            let verified_runs = branches.iter().map(|branch| branch.verified_runs).sum();
            let shared_steps = prefix.summaries();
            let review_digest = family_review_digest(&id, &shared_steps, &branches)?;
            let source_task_id = patterns[members[0]].2;
            result.push(RoutineFamily {
                id,
                review_digest,
                shared_steps,
                branches,
                verified_runs,
                prefix,
                source_task_id,
                source_routines,
            });
        }
    }
    result.sort_by(|left, right| {
        right
            .verified_runs
            .cmp(&left.verified_runs)
            .then_with(|| left.id.cmp(&right.id))
    });
    result.truncate(MAX_VISIBLE_ROUTINES);
    Ok(result)
}

fn requests_overlap(left: &[String], right: &[String]) -> bool {
    left.iter().any(|request| right.contains(request))
}

fn has_prefix(procedure: &CompiledIntent, prefix: &CompiledIntent) -> bool {
    procedure.steps.len() >= prefix.steps.len()
        && procedure
            .steps
            .iter()
            .zip(&prefix.steps)
            .all(|(step, expected)| {
                step.action == expected.action && step.dependencies == expected.dependencies
            })
}

impl LocalStore {
    pub(crate) fn routine_learning_enabled(&self) -> CoreResult<bool> {
        self.with_connection(|db| Ok(enabled(db)?))
    }

    pub(crate) fn routines(&self) -> CoreResult<Vec<Routine>> {
        self.with_connection(|db| Ok(candidates(db)?))
    }

    pub(crate) fn routine_families(&self) -> CoreResult<Vec<RoutineFamily>> {
        self.with_connection(|db| Ok(families(db)?))
    }

    pub(crate) fn routine_family_matches(
        &self,
        id: &str,
        digest: &str,
        sources: &[String],
    ) -> CoreResult<bool> {
        self.with_connection(|db| {
            Ok(families(db)?.into_iter().any(|family| {
                family.id == id
                    && family.review_digest == digest
                    && family.source_routines == sources
            }))
        })
    }

    pub(crate) fn skill_sources_current(&self, skill: &Skill) -> CoreResult<bool> {
        if skill.synthesized_from.is_empty() {
            return Ok(true);
        }
        let (Some(family_id), Some(family_digest)) =
            (&skill.synthesis_family_id, &skill.synthesis_digest)
        else {
            return Ok(false);
        };
        self.routine_family_matches(family_id, family_digest, &skill.synthesized_from)
    }

    pub(crate) fn synthesize_routine_skill(
        &self,
        family_id: &str,
        expected_digest: &str,
        requested_name: &str,
    ) -> CoreResult<Skill> {
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let family = families(&tx)?
                .into_iter()
                .find(|family| family.id == family_id && family.review_digest == expected_digest)
                .ok_or_else(|| {
                    CoreError::ApprovalRejected(
                        "The reviewed routine branches changed before this skill was drafted".into(),
                    )
                })?;
            let name = if requested_name.trim().is_empty() {
                "Shared routine prefix"
            } else {
                requested_name
            };
            let name = crate::knowledge::safe_memory_text(name)?;
            let description = crate::knowledge::safe_memory_text(&format!(
                "Shared steps observed across {} verified runs and {} reviewed request branches.",
                family.verified_runs,
                family.branches.len()
            ))?;
            let graph = family.prefix.graph(
                family.source_task_id,
                format!("Shared routine prefix: {name}"),
            );
            graph
                .validate(family.source_task_id)
                .map_err(CoreError::InvalidAction)?;
            let serialized = serde_json::to_string(&graph)?;
            if serialized.len() > MAX_PROCEDURE_BYTES
                || crate::knowledge::contains_sensitive(&serialized)
            {
                return Err(CoreError::InvalidAction(
                    "The synthesized skill is sensitive or oversized".into(),
                )
                .into());
            }
            let skill = Skill {
                schema_version: 2,
                reviewed_digest: None,
                synthesized_from: family.source_routines.clone(),
                synthesis_family_id: Some(family.id.clone()),
                synthesis_digest: Some(family.review_digest.clone()),
                id: Uuid::new_v4(),
                name,
                description,
                graph,
                source_task_id: family.source_task_id,
                enabled: false,
                created_at: Utc::now(),
            };
            tx.execute(
                "INSERT INTO skills(id,content_json) VALUES(?1,?2)",
                params![skill.id.to_string(), serde_json::to_string(&skill)?],
            )?;
            crate::storage::write_audit(
                &tx,
                None,
                None,
                "routine_prefix_skill_drafted",
                &serde_json::json!({"skill_id":skill.id,"family_id":family.id,"source_routines":family.source_routines.len(),"steps":family.prefix.steps.len()}),
            )?;
            tx.commit()?;
            Ok(skill)
        })
    }

    pub(crate) fn review_routine(&self, id: &str, expected: &str) -> CoreResult<()> {
        self.with_connection(|db| {
            let tx = db.transaction()?;
            if !enabled(&tx)? { return Err(CoreError::PermissionRequired("Enable routine learning before reviewing a routine".into()).into()); }
            let routine = candidates(&tx)?.into_iter().find(|r| r.id == id && r.ready && r.review_digest == expected).ok_or_else(|| CoreError::ApprovalRejected("The observed routine changed or has fewer than three verified examples".into()))?;
            tx.execute("INSERT INTO procedure_reviews VALUES(?1,?2,?3) ON CONFLICT(graph_digest) DO UPDATE SET aliases_json=excluded.aliases_json,reviewed_at=excluded.reviewed_at", params![id,serde_json::to_string(&routine.requests)?,Utc::now().to_rfc3339()])?;
            crate::storage::write_audit(&tx, None, None, "routine_reviewed", &serde_json::json!({"graph_digest":id,"request_count":routine.requests.len(),"verified_runs":routine.verified_runs}))?;
            tx.commit()?;
            Ok(())
        })
    }

    pub(crate) fn forget_routine(&self, id: &str) -> CoreResult<()> {
        self.with_connection(|db| {
            let tx = db.transaction()?;
            let derived_skills: Vec<String> = {
                let mut statement = tx.prepare(
                    "SELECT id FROM skills WHERE EXISTS(SELECT 1 FROM json_each(skills.content_json,'$.synthesized_from') WHERE value=?1)",
                )?;
                statement
                    .query_map([id], |row| row.get(0))?
                    .collect::<Result<_, _>>()?
            };
            let mut affected_workflows = BTreeSet::new();
            if !derived_skills.is_empty() {
                let mut statement = tx.prepare("SELECT id,content_json FROM workflows")?;
                let rows = statement
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                drop(statement);
                for (workflow_id, body) in rows {
                    let mut workflow: crate::workflows::Workflow = serde_json::from_str(&body)?;
                    if workflow
                        .skill_ids
                        .iter()
                        .any(|skill| derived_skills.contains(&skill.to_string()))
                    {
                        workflow.enabled = false;
                        tx.execute(
                            "UPDATE workflows SET content_json=?2 WHERE id=?1",
                            params![workflow_id, serde_json::to_string(&workflow)?],
                        )?;
                        affected_workflows.insert(workflow.id);
                    }
                }
                let mut statement = tx.prepare("SELECT id,content_json FROM schedules")?;
                let rows = statement
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                drop(statement);
                for (schedule_id, body) in rows {
                    let mut schedule: crate::workflows::Schedule = serde_json::from_str(&body)?;
                    if schedule
                        .workflow_id
                        .is_some_and(|workflow| affected_workflows.contains(&workflow))
                    {
                        schedule.enabled = false;
                        schedule.last_error = Some(
                            "A synthesized skill source was forgotten. Review this schedule before re-enabling it."
                                .into(),
                        );
                        tx.execute(
                            "UPDATE schedules SET enabled=0,content_json=?2 WHERE id=?1",
                            params![schedule_id, serde_json::to_string(&schedule)?],
                        )?;
                    }
                }
            }
            for skill_id in &derived_skills {
                tx.execute("DELETE FROM skills WHERE id=?1", [skill_id])?;
            }
            tx.execute("DELETE FROM procedure_samples WHERE graph_digest=?1", [id])?;
            tx.execute("DELETE FROM procedure_reviews WHERE graph_digest=?1", [id])?;
            crate::storage::write_audit(
                &tx,
                None,
                None,
                "routine_forgotten",
                &serde_json::json!({"graph_digest":id,"synthesized_skills_removed":derived_skills.len(),"workflows_disabled":affected_workflows.len()}),
            )?;
            tx.commit()?;
            Ok(())
        })
    }

    pub(crate) fn compiled_routine(
        &self,
        request: &str,
    ) -> CoreResult<Option<(String, CompiledIntent)>> {
        let Some(key) = request_key(request) else {
            return Ok(None);
        };
        self.with_connection(|db| {
            let mut query = db.prepare_cached(ROUTINE_LOOKUP_SQL)?;
            let rows = query
                .query_map([&key], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, u32>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, bool>(4)?,
                        row.get::<_, bool>(5)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            let [(id, body, count, Some(review), true, true)] = rows.as_slice() else {
                // Ambiguous, unreviewed, disabled or incomplete bindings never
                // enter the no-model fast path.
                return Ok(None);
            };
            if *count < MIN_RUNS {
                return Ok(None);
            }
            let Some(approved) = serde_json::from_str::<Vec<String>>(review).ok() else {
                return Ok(None);
            };
            let current_aliases = aliases(db, id)?;
            let alias_count: u32 = db.query_row(
                "SELECT COUNT(DISTINCT request_key) FROM procedure_samples WHERE graph_digest=?1",
                [id],
                |row| row.get(0),
            )?;
            if alias_count as usize > MAX_ALIASES
                || approved != current_aliases
                || !approved.contains(&key)
            {
                return Ok(None);
            }
            let procedure: CompiledIntent = serde_json::from_str(body)?;
            if digest(&procedure)? != *id {
                return Err(CoreError::Storage("Learned procedure digest mismatch".into()).into());
            }
            Ok(Some((id.clone(), procedure)))
        })
    }
}

#[cfg(test)]
mod evolution_tests {
    use super::{
        MIN_RUNS, candidates, digest, migrate, record_outcome, request_key, reviewed_predecessor,
    };
    use crate::{
        contracts::{DataLabel, ToolResult, Verdict},
        domain::{
            Action, ActionProposal, ActionState, ActionStatus, Condition, ExpectedOutcome,
            Provenance, Task, TaskStatus,
        },
        intent::{CompiledIntent, IntentBinding, IntentState, IntentStep},
    };
    use chrono::Utc;
    use rusqlite::{Connection, params};
    use std::collections::{BTreeMap, BTreeSet};
    use uuid::Uuid;

    fn database() -> Connection {
        let db = Connection::open_in_memory().expect("in-memory learning database");
        db.execute_batch(
            "PRAGMA foreign_keys=ON;
             CREATE TABLE settings(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);
             INSERT INTO settings(key,value_json) VALUES
                 ('routine_learning.enabled','true'),('memory.enabled','true');
             CREATE TABLE tasks(id TEXT PRIMARY KEY);
             CREATE TABLE retired_task_data(task_id TEXT PRIMARY KEY);
             CREATE TABLE undo_journal(task_id TEXT NOT NULL,phase TEXT NOT NULL);
             CREATE TABLE action_journal(
                 run_id TEXT NOT NULL,action_id TEXT NOT NULL,action_digest TEXT NOT NULL,
                 state TEXT NOT NULL,verification_json TEXT
             );",
        )
        .expect("learning schema fixture");
        migrate(&db).expect("learning migration");
        db
    }

    fn procedure(action: Action) -> CompiledIntent {
        CompiledIntent {
            steps: vec![IntentStep {
                action,
                dependencies: BTreeSet::new(),
            }],
        }
    }

    fn insert_reviewed(
        db: &Connection,
        request: &str,
        procedure: &CompiledIntent,
        run_count: u32,
    ) -> String {
        let key = request_key(request).expect("valid request alias");
        let graph_digest = digest(procedure).expect("valid procedure digest");
        let graph_json = serde_json::to_string(procedure).expect("serialize procedure");
        for index in 0..run_count {
            let task_id = Uuid::new_v4().to_string();
            db.execute("INSERT INTO tasks(id) VALUES(?1)", [&task_id])
                .expect("source task");
            db.execute(
                "INSERT INTO procedure_samples(task_id,request_key,graph_digest,graph_json,observed_at,intent_revision) VALUES(?1,?2,?3,?4,?5,1)",
                params![task_id, key, graph_digest, graph_json, format!("{index:04}")],
            )
            .expect("source sample");
        }
        let current_aliases = serde_json::to_string(&vec![key]).expect("serialize alias");
        db.execute(
            "INSERT INTO procedure_reviews(graph_digest,aliases_json,reviewed_at) VALUES(?1,?2,'reviewed')",
            params![graph_digest, current_aliases],
        )
        .expect("review source routine");
        graph_digest
    }

    fn verified_task(request: &str, revision: u64, action: Action) -> Task {
        let mut task = Task::new(request);
        let action_id = Uuid::new_v4();
        let expected_outcome = match &action {
            Action::ReadFile { path, .. } => ExpectedOutcome::FileContains {
                path: path.clone(),
                sha256: "a".repeat(64),
            },
            Action::ListDirectory {
                path,
                page_size,
                cursor,
            } => ExpectedOutcome::DirectoryPage {
                path: path.clone(),
                page_size: *page_size,
                cursor: cursor.clone(),
            },
            Action::CreateFolder { path } => ExpectedOutcome::Condition {
                condition: Condition::FolderExists { path: path.clone() },
            },
            Action::OpenApplication { application } => ExpectedOutcome::Condition {
                condition: Condition::ApplicationRunning {
                    application: application.clone(),
                },
            },
            _ => panic!("unsupported learning test action"),
        };
        let proposal = ActionProposal {
            id: action_id,
            task_id: task.id,
            action: action.clone(),
            expected_outcome,
            target_resource: format!("test:{action_id}"),
            provenance: Provenance::user(),
            metadata: BTreeMap::new(),
        };
        task.status = TaskStatus::Succeeded;
        task.actions.insert(
            action_id,
            ActionState {
                proposal: proposal.clone(),
                status: ActionStatus::Succeeded,
                attempts: 1,
                summary: Some("verified test action".into()),
                error: None,
            },
        );
        task.dependencies.insert(action_id, BTreeSet::new());
        task.intent = Some(IntentState {
            revision,
            steps: vec![IntentBinding {
                action_id,
                action,
                dependencies: BTreeSet::new(),
            }],
            retired: BTreeSet::new(),
            change_summary: if revision > 1 {
                "corrected procedure".into()
            } else {
                String::new()
            },
        });
        task.tool_results.push(ToolResult {
            action_id,
            tool: "test".into(),
            verdict: Verdict::Confirmed,
            summary: "fresh verified result".into(),
            output: serde_json::json!({"verified": true}),
            label: DataLabel::private(task.id, "evolution-test".into()),
            observed_at: Utc::now(),
        });
        task
    }

    fn record(db: &mut Connection, task: &Task) {
        db.execute(
            "INSERT OR IGNORE INTO tasks(id) VALUES(?1)",
            [task.id.to_string()],
        )
        .expect("task row");
        for (action_id, action) in &task.actions {
            db.execute(
                "INSERT OR REPLACE INTO action_journal(run_id,action_id,action_digest,state,verification_json) VALUES(?1,?2,?3,'confirmed','{}')",
                params![
                    task.id.to_string(),
                    action_id.to_string(),
                    crate::policy::approval_digest(&action.proposal).expect("action digest"),
                ],
            )
            .expect("verified journal row");
        }
        let tx = db.transaction().expect("learning outcome transaction");
        record_outcome(&tx, task).expect("record outcome");
        tx.commit().expect("commit learning outcome");
    }

    fn source_action() -> Action {
        Action::ReadFile {
            path: "/tmp/sage-source.txt".into(),
            max_bytes: 512,
        }
    }

    fn evolved_action() -> Action {
        Action::CreateFolder {
            path: "/tmp/sage-evolved".into(),
        }
    }

    #[test]
    fn corrected_run_seeds_a_separate_reviewable_evolution_and_source_forgetting_cascades() {
        let mut db = database();
        let request = "Summarize this document";
        let source_digest = insert_reviewed(&db, request, &procedure(source_action()), MIN_RUNS);
        let correction = verified_task(request, 2, evolved_action());
        record(&mut db, &correction);

        let candidate_digest = digest(&procedure(evolved_action())).unwrap();
        assert_ne!(candidate_digest, source_digest);
        let candidate_samples: u32 = db
            .query_row(
                "SELECT COUNT(*) FROM procedure_samples WHERE graph_digest=?1",
                [&candidate_digest],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(candidate_samples, 1);
        let lineage: (String, String, String) = db
            .query_row(
                "SELECT source_digest,request_key,source_review_digest FROM procedure_evolution_samples WHERE task_id=?1",
                [correction.id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(lineage.0, source_digest);
        assert_eq!(lineage.1, "summarize this document");
        assert!(!lineage.2.is_empty());

        for _ in 0..2 {
            record(&mut db, &verified_task(request, 1, evolved_action()));
        }
        let candidate = candidates(&db)
            .unwrap()
            .into_iter()
            .find(|candidate| candidate.id == candidate_digest)
            .expect("evolution candidate");
        assert_eq!(candidate.verified_runs, 3);
        assert!(candidate.ready);
        assert!(
            !candidate.enabled,
            "a corrected variation still needs review"
        );
        let evolution = candidate.evolution.expect("reviewed source comparison");
        assert_eq!(evolution.source_request, "summarize this document");
        assert_eq!(evolution.source_verified_runs, MIN_RUNS);
        assert_eq!(evolution.unchanged_effect_classes, Vec::<String>::new());
        assert_eq!(evolution.added_effect_classes, ["folder.create"]);
        assert_eq!(evolution.removed_effect_classes, ["file.read"]);

        let source_task: String = db
            .query_row(
                "SELECT task_id FROM procedure_samples WHERE graph_digest=?1 ORDER BY task_id LIMIT 1",
                [&source_digest],
                |row| row.get(0),
            )
            .unwrap();
        db.execute(
            "DELETE FROM procedure_samples WHERE task_id=?1",
            [source_task],
        )
        .unwrap();
        let remaining_candidate_samples: u32 = db
            .query_row(
                "SELECT COUNT(*) FROM procedure_samples WHERE graph_digest=?1",
                [&candidate_digest],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining_candidate_samples, 2);
        let remaining_lineage: u32 = db
            .query_row(
                "SELECT COUNT(*) FROM procedure_evolution_samples WHERE candidate_digest=?1",
                [&candidate_digest],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining_lineage, 0);
    }

    #[test]
    fn corrected_run_requires_one_currently_reviewed_exact_request_predecessor() {
        let request = "summarize this document";
        let candidate = procedure(evolved_action());
        let candidate_digest = digest(&candidate).unwrap();

        let ambiguous = database();
        insert_reviewed(&ambiguous, request, &procedure(source_action()), MIN_RUNS);
        insert_reviewed(
            &ambiguous,
            request,
            &procedure(Action::ListDirectory {
                path: "/tmp/sage-other-source".into(),
                page_size: 16,
                cursor: None,
            }),
            MIN_RUNS,
        );
        assert!(
            reviewed_predecessor(&ambiguous, request, &candidate_digest, &candidate)
                .unwrap()
                .is_none()
        );

        let stale = database();
        let source_digest = insert_reviewed(&stale, request, &procedure(source_action()), MIN_RUNS);
        let extra_task = Uuid::new_v4().to_string();
        stale
            .execute("INSERT INTO tasks(id) VALUES(?1)", [&extra_task])
            .unwrap();
        stale
            .execute(
                "INSERT INTO procedure_samples(task_id,request_key,graph_digest,graph_json,observed_at,intent_revision) VALUES(?1,'different alias',?2,?3,'extra',1)",
                params![
                    extra_task,
                    source_digest,
                    serde_json::to_string(&procedure(source_action())).unwrap(),
                ],
            )
            .unwrap();
        assert!(
            reviewed_predecessor(&stale, request, &candidate_digest, &candidate)
                .unwrap()
                .is_none(),
            "a review no longer matching the source aliases is stale"
        );
    }

    #[test]
    fn corrected_same_graph_does_not_reinforce_a_routine_and_retries_are_idempotent() {
        let mut db = database();
        let request = "summarize this document";
        let source_digest = insert_reviewed(&db, request, &procedure(source_action()), MIN_RUNS);
        let correction = verified_task(request, 2, source_action());
        record(&mut db, &correction);
        let source_count: u32 = db
            .query_row(
                "SELECT COUNT(*) FROM procedure_samples WHERE graph_digest=?1",
                [&source_digest],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(source_count, MIN_RUNS);
        assert_eq!(
            db.query_row(
                "SELECT COUNT(*) FROM procedure_evolution_samples",
                [],
                |row| row.get::<_, u32>(0)
            )
            .unwrap(),
            0
        );

        let variant = verified_task(request, 2, evolved_action());
        record(&mut db, &variant);
        record(&mut db, &variant);
        let candidate_digest = digest(&procedure(evolved_action())).unwrap();
        assert_eq!(
            db.query_row(
                "SELECT COUNT(*) FROM procedure_samples WHERE graph_digest=?1",
                [&candidate_digest],
                |row| row.get::<_, u32>(0),
            )
            .unwrap(),
            1
        );
    }

    #[test]
    fn learning_migration_adds_revision_metadata_to_existing_samples() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(
            "CREATE TABLE tasks(id TEXT PRIMARY KEY);
             CREATE TABLE retired_task_data(task_id TEXT PRIMARY KEY);
             CREATE TABLE procedure_samples(task_id TEXT PRIMARY KEY,request_key TEXT NOT NULL,graph_digest TEXT NOT NULL,graph_json TEXT NOT NULL,observed_at TEXT NOT NULL);
             INSERT INTO tasks(id) VALUES('old-task');
             INSERT INTO procedure_samples VALUES('old-task','old request','digest','{}','observed');",
        )
        .unwrap();
        migrate(&db).unwrap();
        let revision: i64 = db
            .query_row(
                "SELECT intent_revision FROM procedure_samples WHERE task_id='old-task'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(revision, 1);
        assert_eq!(
            db.query_row(
                "SELECT COUNT(*) FROM procedure_samples WHERE task_id='old-task'",
                [],
                |row| row.get::<_, u32>(0)
            )
            .unwrap(),
            1
        );
    }
}
