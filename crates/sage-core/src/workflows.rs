//! Reusable graphs and durable triggers. Definitions contain no execution grants.
use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Duration, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::{
    ActionGraph, ActionNode, ActionStatus, Condition, Provenance, Task, TaskStatus,
};
use crate::error::{CoreError, CoreResult};
use crate::knowledge::{clipped, safe_memory_text};
use crate::storage::LocalStore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    #[serde(default)]
    pub schema_version: u32,
    #[serde(default)]
    pub reviewed_digest: Option<String>,
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub graph: ActionGraph,
    pub source_task_id: Uuid,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
}

impl Skill {
    pub fn digest(&self) -> CoreResult<String> {
        use sha2::{Digest, Sha256};
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(
                self.schema_version,
                &self.name,
                &self.description,
                &self.graph,
                self.source_task_id
            ))?)
        ))
    }
    pub fn is_reviewed(&self) -> bool {
        self.schema_version == 2
            && self.enabled
            && self.digest().ok().as_ref() == self.reviewed_digest.as_ref()
            && self.reviewed_digest.is_some()
    }
    pub fn preview(&self) -> CoreResult<String> {
        self.graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                Ok(format!(
                    "{}. {}",
                    i + 1,
                    crate::policy::action_preview(&node.proposal.action)?
                ))
            })
            .collect::<CoreResult<Vec<_>>>()
            .map(|steps| steps.join("\n\n"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: Uuid,
    pub name: String,
    pub skill_ids: Vec<Uuid>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Trigger {
    Once { at: DateTime<Utc> },
    Interval { seconds: u64 },
    FolderChanged { path: std::path::PathBuf },
    Condition { condition: Condition },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    #[serde(default)]
    pub trigger_pending: bool,
    #[serde(default)]
    pub resources: Vec<crate::contracts::ResourceScope>,
    #[serde(default)]
    pub expires_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub remaining_runs: u32,
    pub id: Uuid,
    pub name: String,
    pub request: String,
    pub conversation_id: Uuid,
    pub workflow_id: Option<Uuid>,
    pub trigger: Trigger,
    pub enabled: bool,
    pub next_run_at: DateTime<Utc>,
    pub last_task_id: Option<Uuid>,
    pub last_condition: bool,
    pub last_error: Option<String>,
}

impl LocalStore {
    pub fn migrate_workflows(&self) -> CoreResult<()> {
        self.with_connection(|db| {
            db.execute_batch("CREATE TABLE IF NOT EXISTS skills(id TEXT PRIMARY KEY, content_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS workflows(id TEXT PRIMARY KEY, content_json TEXT NOT NULL); CREATE TABLE IF NOT EXISTS schedules(id TEXT PRIMARY KEY, content_json TEXT NOT NULL, enabled INTEGER NOT NULL, next_run_at TEXT NOT NULL); CREATE TABLE IF NOT EXISTS schedule_runs(id TEXT PRIMARY KEY,schedule_id TEXT NOT NULL,task_id TEXT,claimed_at TEXT NOT NULL,status TEXT NOT NULL); CREATE INDEX IF NOT EXISTS schedules_due ON schedules(enabled,next_run_at); UPDATE schedules SET enabled=0,content_json=json_set(content_json,'$.enabled',json('false'),'$.last_error','A schedule dispatch was interrupted; review it before re-enabling') WHERE id IN (SELECT schedule_id FROM schedule_runs WHERE status='claimed'); UPDATE schedule_runs SET status='interrupted' WHERE status='claimed'; INSERT OR IGNORE INTO schema_migrations VALUES(3,CURRENT_TIMESTAMP);")?;
            Ok(())
        })
        ?;
        self.with_connection(|db| {
            // Import earlier procedures as drafts. Stored traces never carry
            // authority forward across a protocol/security migration.
            let migrated:bool=db.query_row("SELECT EXISTS(SELECT 1 FROM schema_migrations WHERE version=4)",[],|r|r.get(0))?;
            if !migrated {
                let tx=db.transaction()?;
                tx.execute("UPDATE skills SET content_json=json_set(content_json,'$.enabled',json('false'),'$.schema_version',2,'$.reviewed_digest',NULL)",[])?;
                tx.execute("UPDATE workflows SET content_json=json_set(content_json,'$.enabled',json('false'))",[])?;
                tx.execute("UPDATE schedules SET enabled=0,content_json=json_set(content_json,'$.enabled',json('false'),'$.last_error','Re-authorize this schedule after the v2 migration')",[])?;
                tx.execute("INSERT INTO schema_migrations VALUES(4,CURRENT_TIMESTAMP)",[])?;
                tx.commit()?;
            }
            Ok(())
        })
    }

    pub fn skills(&self) -> CoreResult<Vec<Skill>> {
        self.load_definitions("skills")
    }
    pub fn workflows(&self) -> CoreResult<Vec<Workflow>> {
        self.load_definitions("workflows")
    }
    pub fn schedules(&self) -> CoreResult<Vec<Schedule>> {
        self.load_definitions("schedules")
    }

    fn load_definitions<T: serde::de::DeserializeOwned>(&self, table: &str) -> CoreResult<Vec<T>> {
        self.with_connection(|db| {
            // table names are private compile-time callers, never IPC input.
            let mut statement = db.prepare(&format!(
                "SELECT content_json FROM {table} ORDER BY rowid DESC LIMIT 200"
            ))?;
            let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
            rows.map(|row| Ok(serde_json::from_str(&row?)?)).collect()
        })
    }

    pub fn capture_skill(&self, task: &Task, name: &str) -> CoreResult<Skill> {
        if task.status != TaskStatus::Succeeded
            || task.actions.is_empty()
            || task
                .actions
                .values()
                .any(|a| a.status != ActionStatus::Succeeded)
        {
            return Err(CoreError::InvalidAction(
                "Only fully verified tasks can become skills.".into(),
            ));
        }
        let name = safe_memory_text(name)?;
        let graph = ActionGraph {
            goal: task.goal.clone().unwrap_or_else(|| name.clone()),
            nodes: task
                .actions
                .iter()
                .map(|(id, state)| ActionNode {
                    proposal: state.proposal.clone(),
                    depends_on: task.dependencies.get(id).cloned().unwrap_or_default(),
                })
                .collect(),
        };
        // Reject secrets in action payloads too; a skill is persistent reusable data.
        let serialized = serde_json::to_string(&graph)?;
        if serialized.len() > 1024 * 1024 || crate::knowledge::contains_sensitive(&serialized) {
            return Err(CoreError::InvalidAction(
                "Skill contains sensitive data or exceeds one MiB.".into(),
            ));
        }
        let skill = Skill {
            schema_version: 2,
            reviewed_digest: None,
            id: Uuid::new_v4(),
            name,
            description: clipped(&task.request, 300),
            graph,
            source_task_id: task.id,
            enabled: false,
            created_at: Utc::now(),
        };
        self.save_skill(&skill)?;
        Ok(skill)
    }

    pub fn save_skill(&self, skill: &Skill) -> CoreResult<()> {
        skill
            .graph
            .validate(skill.source_task_id)
            .map_err(CoreError::InvalidAction)?;
        if skill.graph.nodes.len() > 32 || (skill.enabled && !skill.is_reviewed()) {
            return Err(CoreError::PermissionRequired(
                "Review this exact skill before enabling it".into(),
            ));
        }
        self.with_connection(|db| {db.execute("INSERT INTO skills VALUES(?1,?2) ON CONFLICT(id) DO UPDATE SET content_json=excluded.content_json",params![skill.id.to_string(),serde_json::to_string(skill)?])?;Ok(())})
    }

    pub fn delete_definition(&self, kind: &str, id: Uuid) -> CoreResult<()> {
        let table = match kind {
            "skill" => "skills",
            "workflow" => "workflows",
            "schedule" => "schedules",
            _ => return Err(CoreError::InvalidAction("Unknown definition kind".into())),
        };
        self.with_connection(|db| {
            db.execute(
                &format!("DELETE FROM {table} WHERE id=?1"),
                params![id.to_string()],
            )?;
            Ok(())
        })
    }

    pub fn save_workflow(&self, workflow: &Workflow) -> CoreResult<()> {
        if workflow.name.trim().is_empty()
            || workflow.skill_ids.is_empty()
            || workflow.skill_ids.len() > 16
        {
            return Err(CoreError::InvalidAction(
                "A workflow needs a name and 1–16 skills.".into(),
            ));
        }
        let skills = self.skills()?;
        if workflow
            .skill_ids
            .iter()
            .any(|id| !skills.iter().any(|s| s.id == *id && s.is_reviewed()))
        {
            return Err(CoreError::InvalidAction(
                "A workflow references an unavailable skill.".into(),
            ));
        }
        self.with_connection(|db| {db.execute("INSERT INTO workflows VALUES(?1,?2) ON CONFLICT(id) DO UPDATE SET content_json=excluded.content_json",params![workflow.id.to_string(),serde_json::to_string(workflow)?])?;Ok(())})
    }

    pub fn workflow_graph(&self, workflow_id: Uuid, task_id: Uuid) -> CoreResult<ActionGraph> {
        let workflow = self
            .workflows()?
            .into_iter()
            .find(|w| w.id == workflow_id && w.enabled)
            .ok_or_else(|| CoreError::InvalidAction("Workflow is disabled or missing.".into()))?;
        let skills = self.skills()?;
        let mut graph = ActionGraph {
            goal: workflow.name,
            nodes: Vec::new(),
        };
        let mut previous = BTreeSet::new();
        for id in workflow.skill_ids {
            let skill = skills
                .iter()
                .find(|s| s.id == id && s.is_reviewed())
                .ok_or_else(|| CoreError::InvalidAction("Skill is disabled or missing.".into()))?;
            let mut expanded = instantiate(&skill.graph, task_id)?;
            for node in &mut expanded.nodes {
                if node.depends_on.is_empty() {
                    node.depends_on.extend(&previous);
                }
            }
            previous = expanded.nodes.iter().map(|node| node.proposal.id).collect();
            graph.nodes.extend(expanded.nodes);
        }
        if graph.nodes.len() > 32 {
            return Err(CoreError::InvalidAction(
                "Workflow exceeds 32 actions.".into(),
            ));
        }
        graph.validate(task_id).map_err(CoreError::InvalidAction)?;
        Ok(graph)
    }

    pub fn save_schedule(&self, schedule: &Schedule) -> CoreResult<()> {
        if schedule.enabled
            && (schedule.expires_at.is_none_or(|expiry| {
                expiry <= Utc::now() || expiry > Utc::now() + Duration::days(90)
            }) || schedule.remaining_runs == 0
                || schedule.remaining_runs > 10_000
                || schedule.resources.len() > 32)
        {
            return Err(CoreError::PermissionRequired(
                "A background task needs a current scope, expiry and run budget".into(),
            ));
        }
        if schedule.name.trim().is_empty()
            || schedule.request.trim().is_empty()
            || schedule.request.len() > 8192
        {
            return Err(CoreError::InvalidAction(
                "A schedule needs a name and bounded request.".into(),
            ));
        }
        if matches!(schedule.trigger,Trigger::Interval{seconds} if !(60..=31_536_000).contains(&seconds))
        {
            return Err(CoreError::InvalidAction(
                "Intervals must be between one minute and one year.".into(),
            ));
        }
        self.with_connection(|db| {db.execute("INSERT INTO schedules VALUES(?1,?2,?3,?4) ON CONFLICT(id) DO UPDATE SET content_json=excluded.content_json,enabled=excluded.enabled,next_run_at=excluded.next_run_at",params![schedule.id.to_string(),serde_json::to_string(schedule)?,schedule.enabled,schedule.next_run_at.to_rfc3339()])?;Ok(())})
    }

    /// Claim and advance in one transaction. An ambiguous crash never retries a firing.
    pub fn claim_schedule(&self, schedule: &mut Schedule) -> CoreResult<Option<Uuid>> {
        let now = Utc::now();
        if !schedule.enabled
            || schedule.next_run_at > now
            || schedule.remaining_runs == 0
            || schedule.expires_at.is_none_or(|expiry| expiry <= now)
        {
            return Ok(None);
        }
        let previous = serde_json::to_string(schedule)?;
        let mut candidate = schedule.clone();
        candidate.trigger_pending = false;
        candidate.next_run_at = now
            + Duration::seconds(match candidate.trigger {
                Trigger::Interval { seconds } => seconds as i64,
                _ => 60,
            });
        if matches!(candidate.trigger, Trigger::Once { .. }) {
            candidate.enabled = false;
        }
        candidate.remaining_runs -= 1;
        if candidate.remaining_runs == 0 {
            candidate.enabled = false;
        }
        let run = Uuid::new_v4();
        let claimed = self.with_connection(|db| {
            let tx=db.transaction()?;
            if tx.execute("UPDATE schedules SET content_json=?2,enabled=?3,next_run_at=?4 WHERE id=?1 AND enabled=1 AND content_json=?5",params![candidate.id.to_string(),serde_json::to_string(&candidate)?,candidate.enabled,candidate.next_run_at.to_rfc3339(),previous])?!=1 {return Ok(None);}
            tx.execute("INSERT INTO schedule_runs VALUES(?1,?2,NULL,?3,'claimed')",params![run.to_string(),candidate.id.to_string(),now.to_rfc3339()])?;
            tx.commit()?;Ok(Some(run))
        })?;
        if claimed.is_some() {
            *schedule = candidate;
        }
        Ok(claimed)
    }

    pub fn disable_schedule_if_current(&self, schedule: &Schedule, reason: &str) -> CoreResult<()> {
        let mut disabled = schedule.clone();
        disabled.enabled = false;
        disabled.last_error = Some(crate::redaction::redact_for_persistence(reason));
        self.with_connection(|db| {
            db.execute(
                "UPDATE schedules SET enabled=0,content_json=?2 WHERE id=?1 AND content_json=?3",
                params![
                    schedule.id.to_string(),
                    serde_json::to_string(&disabled)?,
                    serde_json::to_string(schedule)?
                ],
            )?;
            Ok(())
        })
    }

    pub fn finish_schedule_claim(&self, run: Uuid, task: Option<Uuid>) -> CoreResult<()> {
        self.with_connection(|db| {
            db.execute(
                "UPDATE schedule_runs SET task_id=?2,status=?3 WHERE id=?1",
                params![
                    run.to_string(),
                    task.map(|id| id.to_string()),
                    if task.is_some() {
                        "dispatched"
                    } else {
                        "failed"
                    }
                ],
            )?;
            Ok(())
        })
    }
}

pub fn instantiate(template: &ActionGraph, task: Uuid) -> CoreResult<ActionGraph> {
    let ids: BTreeMap<_, _> = template
        .nodes
        .iter()
        .map(|node| (node.proposal.id, Uuid::new_v4()))
        .collect();
    let mut graph = template.clone();
    for node in &mut graph.nodes {
        node.proposal.id = ids[&node.proposal.id];
        node.proposal.task_id = task;
        node.proposal.provenance = Provenance::model(vec!["reusable_skill".into()]);
        node.proposal.metadata.clear();
        node.depends_on = node
            .depends_on
            .iter()
            .map(|id| {
                ids.get(id)
                    .copied()
                    .ok_or_else(|| CoreError::InvalidAction("Invalid skill dependency".into()))
            })
            .collect::<CoreResult<_>>()?;
    }
    graph.validate(task).map_err(CoreError::InvalidAction)?;
    Ok(graph)
}

/// OS change events are coalesced per authorized root. Channel pressure never
/// discards the dirty state; the periodic scheduler drains the same state.
#[derive(Default)]
struct WatchState {
    changed: BTreeSet<std::path::PathBuf>,
    errors: BTreeMap<std::path::PathBuf, String>,
}
pub struct FolderMonitor {
    sender: tokio::sync::mpsc::Sender<()>,
    state: std::sync::Arc<std::sync::Mutex<WatchState>>,
    watchers: BTreeMap<std::path::PathBuf, notify::RecommendedWatcher>,
}
impl FolderMonitor {
    pub fn new(sender: tokio::sync::mpsc::Sender<()>) -> Self {
        Self {
            sender,
            state: Default::default(),
            watchers: BTreeMap::new(),
        }
    }
    pub fn drain(
        &self,
    ) -> CoreResult<(
        BTreeSet<std::path::PathBuf>,
        BTreeMap<std::path::PathBuf, String>,
    )> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CoreError::ExecutionFailed("Folder monitor state unavailable".into()))?;
        Ok((
            std::mem::take(&mut state.changed),
            std::mem::take(&mut state.errors),
        ))
    }
    pub fn sync(&mut self, schedules: &[Schedule]) {
        use notify::Watcher;
        let paths: BTreeSet<_> = schedules
            .iter()
            .filter(|s| {
                s.enabled
                    && s.remaining_runs > 0
                    && s.expires_at.is_some_and(|expiry| expiry > Utc::now())
            })
            .filter_map(|s| match &s.trigger {
                Trigger::FolderChanged { path } => Some(path.clone()),
                _ => None,
            })
            .collect();
        self.watchers.retain(|path, _| paths.contains(path));
        for path in paths {
            if self.watchers.contains_key(&path) {
                continue;
            }
            let sender = self.sender.clone();
            let state = self.state.clone();
            let root = path.clone();
            let watcher =
                notify::recommended_watcher(move |event: Result<notify::Event, notify::Error>| {
                    if let Ok(mut state) = state.lock() {
                        match event {
                            Ok(event)
                                if matches!(
                                    event.kind,
                                    notify::EventKind::Create(_)
                                        | notify::EventKind::Modify(_)
                                        | notify::EventKind::Remove(_)
                                ) =>
                            {
                                state.changed.insert(root.clone());
                            }
                            Err(_) => {
                                state.errors.insert(
                                    root.clone(),
                                    "Folder watcher lost access; review the background scope"
                                        .into(),
                                );
                            }
                            _ => return,
                        }
                        let _ = sender.try_send(());
                    }
                })
                .and_then(|mut watcher| {
                    watcher.watch(&path, notify::RecursiveMode::NonRecursive)?;
                    Ok(watcher)
                });
            match watcher {
                Ok(watcher) => {
                    self.watchers.insert(path, watcher);
                }
                Err(_) => {
                    if let Ok(mut state) = self.state.lock() {
                        state.errors.insert(
                            path,
                            "Folder watcher could not start; review the background scope".into(),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod v2_tests {
    use super::*;
    fn schedule() -> Schedule {
        Schedule {
            trigger_pending: false,
            resources: vec![],
            expires_at: Some(Utc::now() + Duration::hours(1)),
            remaining_runs: 2,
            id: Uuid::new_v4(),
            name: "Review".into(),
            request: "Review a document".into(),
            conversation_id: Uuid::new_v4(),
            workflow_id: None,
            trigger: Trigger::Interval { seconds: 60 },
            enabled: true,
            next_run_at: Utc::now() - Duration::seconds(1),
            last_task_id: None,
            last_condition: false,
            last_error: None,
        }
    }
    #[test]
    fn claim_is_single_use_and_cannot_reuse_changed_authorization() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::open_encrypted(
            &dir.path().join("test.db"),
            &crate::secrets::SecretBytes::new(vec![12; 32]),
        )
        .unwrap();
        store.migrate_workflows().unwrap();
        let mut first = schedule();
        store.save_schedule(&first).unwrap();
        let mut stale = first.clone();
        assert!(store.claim_schedule(&mut first).unwrap().is_some());
        assert_eq!(first.remaining_runs, 1);
        assert!(store.claim_schedule(&mut stale).unwrap().is_none());
        assert_eq!(
            stale.remaining_runs, 2,
            "A rejected claim must not mutate caller state"
        );
        first.next_run_at = Utc::now() - Duration::seconds(1);
        store.save_schedule(&first).unwrap();
        let mut old = first.clone();
        first.request = "A materially different task".into();
        store.save_schedule(&first).unwrap();
        assert!(store.claim_schedule(&mut old).unwrap().is_none());
        store.migrate_workflows().unwrap();
        let recovered = store.schedules().unwrap().remove(0);
        assert!(!recovered.enabled);
        assert!(recovered.last_error.unwrap().contains("interrupted"));
    }
    #[test]
    fn schedules_need_unexpired_authority_and_stop_at_their_budget() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStore::open_encrypted(
            &dir.path().join("test.db"),
            &crate::secrets::SecretBytes::new(vec![12; 32]),
        )
        .unwrap();
        store.migrate_workflows().unwrap();
        let mut value = schedule();
        value.expires_at = Some(Utc::now() - Duration::seconds(1));
        assert!(store.save_schedule(&value).is_err());
        assert!(store.claim_schedule(&mut value).unwrap().is_none());
        value.expires_at = Some(Utc::now() + Duration::minutes(10));
        value.remaining_runs = 1;
        value.trigger_pending = true;
        store.save_schedule(&value).unwrap();
        assert!(store.claim_schedule(&mut value).unwrap().is_some());
        assert!(!value.enabled);
        assert!(!value.trigger_pending);
        assert_eq!(value.remaining_runs, 0);
        assert!(store.claim_schedule(&mut value).unwrap().is_none());
    }
}
