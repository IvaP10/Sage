//! Local conversational state and selective, provenance-bearing memory.
//! These records are planning hints. None can grant capabilities or approval.
use std::collections::BTreeMap;

use chrono::{DateTime, Duration, Utc};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::{Provenance, ProvenanceSource, Task, TaskStatus};
use crate::error::{CoreError, CoreResult};
use crate::redaction::redact_for_persistence;
use crate::storage::LocalStore;

pub const RECENT_MESSAGES: usize = 12;
pub const SUMMARY_CHARS: usize = 3_000;
pub const MEMORY_LIMIT: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: Uuid,
    pub title: String,
    pub summary: String,
    pub pinned: bool,
    pub archived: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub task_id: Option<Uuid>,
    pub role: String,
    pub content: String,
    pub provenance: Provenance,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Preference,
    Episodic,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub kind: MemoryKind,
    pub subject: String,
    pub content: String,
    pub confidence: f64,
    pub provenance: Provenance,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    pub conversation_id: Uuid,
    pub task_id: Uuid,
    pub goal: String,
    pub resources: Vec<String>,
    pub applications: Vec<String>,
    pub entities: Vec<String>,
    pub recent_actions: Vec<String>,
    #[serde(default)]
    pub selections: Vec<String>,
    #[serde(default)]
    pub selection_expires_at: Option<DateTime<Utc>>,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSnapshot {
    pub conversations: Vec<Conversation>,
    pub messages: Vec<Message>,
    pub memories: Vec<MemoryRecord>,
    pub memory_enabled: bool,
}

pub fn clipped(text: &str, characters: usize) -> String {
    text.chars().take(characters).collect()
}

pub fn contains_sensitive(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    redact_for_persistence(text).contains("[REDACTED]")
        || [
            "password",
            "api key",
            "api_key",
            "access token",
            "private key",
            "secret",
            "bearer ",
            "credit card",
            "social security",
            "seed phrase",
        ]
        .iter()
        .any(|label| lower.contains(label))
}

pub fn safe_memory_text(text: &str) -> CoreResult<String> {
    let text = text.trim();
    let redacted = redact_for_persistence(text);
    // Do not convert credentials into searchable personal facts. Also catch
    // conversational labels that the line-oriented persistence redactor cannot.
    if contains_sensitive(text) {
        return Err(CoreError::InvalidAction(
            "Sensitive information cannot be stored in memory.".into(),
        ));
    }
    if text.is_empty() || text.len() > 4_096 {
        return Err(CoreError::InvalidAction(
            "Memory must contain 1–4096 bytes.".into(),
        ));
    }
    Ok(redacted)
}

/// Deliberately finite preference vocabulary; no executable code or policy keys.
pub fn preference(text: &str) -> Option<(String, String)> {
    let lower = text.to_lowercase();
    for key in [
        "browser",
        "editor",
        "terminal",
        "folder",
        "interaction style",
        "voice",
        "workflow",
    ] {
        for prefix in [
            format!("my preferred {key} is "),
            format!("preferred {key} is "),
            format!("my {key} is "),
            format!("{key}="),
        ] {
            if lower.starts_with(&prefix) {
                let value = text[prefix.len()..].trim().trim_end_matches('.');
                if !value.is_empty() {
                    return Some((key.replace(' ', "_"), value.into()));
                }
            }
        }
    }
    if lower.starts_with("i prefer ") {
        return Some((
            "workflow".into(),
            text[9..].trim().trim_end_matches('.').into(),
        ));
    }
    None
}

impl LocalStore {
    pub fn migrate_knowledge(&self) -> CoreResult<()> {
        self.with_connection(|db| {
            db.execute_batch(r#"
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
                    pinned INTEGER NOT NULL DEFAULT 0, archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    task_id TEXT REFERENCES tasks(id), role TEXT NOT NULL, content TEXT NOT NULL,
                    provenance_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS context_memory_sources (
                    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    memory_id TEXT NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
                    PRIMARY KEY(task_id,memory_id)
                );
                CREATE TABLE IF NOT EXISTS memory_lineage (
                    parent_id TEXT NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
                    child_id TEXT NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
                    PRIMARY KEY(parent_id,child_id)
                );
                CREATE TABLE IF NOT EXISTS excluded_context_messages (
                    message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS messages_task_role ON messages(task_id, role) WHERE task_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS conversation_tasks (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
                    message_id TEXT NOT NULL REFERENCES messages(id),
                    conversation_id TEXT NOT NULL REFERENCES conversations(id)
                );
                CREATE TABLE IF NOT EXISTS working_memory (
                    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                    content_json TEXT NOT NULL, expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY REFERENCES memory(id) ON DELETE CASCADE,
                    subject TEXT NOT NULL, confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    provenance_json TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1
                );
                CREATE TRIGGER IF NOT EXISTS memory_index_insert AFTER INSERT ON memory BEGIN
                    INSERT INTO memory_fts(memory_id,content) VALUES(new.id,new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS memory_index_update AFTER UPDATE ON memory BEGIN
                    DELETE FROM memory_fts WHERE memory_id=old.id;
                    INSERT INTO memory_fts(memory_id,content) VALUES(new.id,new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS memory_index_delete AFTER DELETE ON memory BEGIN
                    DELETE FROM memory_fts WHERE memory_id=old.id;
                END;
                CREATE TRIGGER IF NOT EXISTS memory_subject_insert AFTER INSERT ON memory_records BEGIN
                    UPDATE memory_fts SET content=new.subject || ' ' || (SELECT content FROM memory WHERE id=new.id) WHERE memory_id=new.id;
                END;
                CREATE TRIGGER IF NOT EXISTS memory_subject_update AFTER UPDATE ON memory_records BEGIN
                    UPDATE memory_fts SET content=new.subject || ' ' || (SELECT content FROM memory WHERE id=new.id) WHERE memory_id=new.id;
                END;
                INSERT INTO memory_fts(memory_id,content)
                    SELECT id,content FROM memory WHERE id NOT IN (SELECT memory_id FROM memory_fts);
                INSERT OR IGNORE INTO memory_records(id,subject,confidence,provenance_json,enabled)
                    SELECT id,'Imported memory',0.5,'{"source":"sage_core","trust":"untrusted_external_content","source_id":null,"parent_ids":[]}',1 FROM memory;
                INSERT OR IGNORE INTO schema_migrations VALUES(2,CURRENT_TIMESTAMP);
            "#)?;
            Ok(())
        })?;
        // Migration is idempotent and preserves the old task IDs and audit chain.
        for mut task in self.load_tasks(true)? {
            if task.conversation_id.is_none() {
                let conversation = self.ensure_conversation(Some(task.id), &task.request)?;
                task.conversation_id = Some(conversation.id);
                task.message_id = Some(task.id);
                self.save_task(&task)?;
                self.append_message(&Message {
                    id: task.id,
                    conversation_id: conversation.id,
                    task_id: Some(task.id),
                    role: "user".into(),
                    content: task.request.clone(),
                    provenance: Provenance::user(),
                    created_at: task.created_at,
                })?;
                self.link_task(&task)?;
                if task.final_outcome.is_some() {
                    self.record_outcome(&task)?;
                }
            }
        }
        Ok(())
    }

    pub fn ensure_conversation(&self, id: Option<Uuid>, title: &str) -> CoreResult<Conversation> {
        let id = id.unwrap_or_else(Uuid::new_v4);
        self.with_connection(|db| {
            db.execute("INSERT OR IGNORE INTO conversations(id,title,created_at,updated_at) VALUES(?1,?2,?3,?3)", params![id.to_string(), clipped(&redact_for_persistence(title), 100), Utc::now().to_rfc3339()])?;
            db.query_row("SELECT id,title,summary,pinned,archived,created_at,updated_at FROM conversations WHERE id=?1 AND archived=0", params![id.to_string()], conversation_row).map_err(Into::into)
        })
    }

    pub fn conversations(&self) -> CoreResult<Vec<Conversation>> {
        self.with_connection(|db| {
            let mut statement = db.prepare("SELECT id,title,summary,pinned,archived,created_at,updated_at FROM conversations WHERE archived=0 ORDER BY pinned DESC,updated_at DESC LIMIT 200")?;
            Ok(statement.query_map([], conversation_row)?.collect::<Result<Vec<_>,_>>()?)
        })
    }

    pub fn update_conversation(
        &self,
        id: Uuid,
        title: &str,
        pinned: bool,
        archived: bool,
    ) -> CoreResult<()> {
        if title.trim().is_empty() || title.len() > 512 {
            return Err(CoreError::InvalidAction(
                "Conversation title must contain 1–512 bytes.".into(),
            ));
        }
        self.with_connection(|db| {
            if db.execute(
                "UPDATE conversations SET title=?2,pinned=?3,archived=?4,updated_at=?5 WHERE id=?1",
                params![
                    id.to_string(),
                    redact_for_persistence(title),
                    pinned,
                    archived,
                    Utc::now().to_rfc3339()
                ],
            )? != 1
            {
                return Err("conversation not found".into());
            }
            Ok(())
        })
    }

    pub fn append_message(&self, message: &Message) -> CoreResult<()> {
        self.with_connection(|db| {
            let transaction = db.transaction()?;
            transaction.execute("INSERT OR IGNORE INTO messages(id,conversation_id,task_id,role,content,provenance_json,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7)", params![message.id.to_string(),message.conversation_id.to_string(),message.task_id.map(|id| id.to_string()),message.role,redact_for_persistence(&message.content),serde_json::to_string(&message.provenance)?,message.created_at.to_rfc3339()])?;
            transaction.execute("UPDATE conversations SET updated_at=?2 WHERE id=?1", params![message.conversation_id.to_string(),Utc::now().to_rfc3339()])?;
            transaction.commit()?;
            Ok(())
        })
    }

    pub fn link_task(&self, task: &Task) -> CoreResult<()> {
        self.with_connection(|db| {
            if let (Some(conversation), Some(message)) = (task.conversation_id, task.message_id) {
                db.execute(
                    "INSERT OR IGNORE INTO conversation_tasks VALUES(?1,?2,?3)",
                    params![
                        task.id.to_string(),
                        message.to_string(),
                        conversation.to_string()
                    ],
                )?;
            }
            Ok(())
        })
    }

    pub fn messages(&self, conversation: Uuid, limit: usize) -> CoreResult<Vec<Message>> {
        self.with_connection(|db| {
            let mut statement = db.prepare("SELECT id,conversation_id,task_id,role,content,provenance_json,created_at FROM messages WHERE conversation_id=?1 ORDER BY created_at DESC,rowid DESC LIMIT ?2")?;
            let rows = statement.query_map(params![conversation.to_string(),limit.min(200)], |row| {
                Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?,row.get::<_,Option<String>>(2)?,row.get::<_,String>(3)?,row.get::<_,String>(4)?,row.get::<_,String>(5)?,row.get::<_,String>(6)?))
            })?;
            let mut messages = Vec::new();
            for row in rows { let (id,conversation,task,role,content,provenance,created) = row?; messages.push(Message { id: id.parse()?, conversation_id: conversation.parse()?, task_id: task.map(|id| id.parse()).transpose()?,role,content,provenance: serde_json::from_str(&provenance)?,created_at: created.parse()? }); }
            messages.reverse();
            Ok(messages)
        })
    }

    pub fn memory_enabled(&self) -> CoreResult<bool> {
        Ok(self.load_setting("memory.enabled")?.unwrap_or(true))
    }

    pub fn save_memory(&self, mut record: MemoryRecord) -> CoreResult<MemoryRecord> {
        if !self.memory_enabled()? {
            return Err(CoreError::InvalidAction(
                "Memory is disabled in Settings.".into(),
            ));
        }
        record.content = safe_memory_text(&record.content)?;
        record.subject = safe_memory_text(&record.subject)?;
        if !record.confidence.is_finite() || !(0.0..=1.0).contains(&record.confidence) {
            return Err(CoreError::InvalidAction(
                "Invalid memory confidence.".into(),
            ));
        }
        record.updated_at = Utc::now();
        self.with_connection(|db| {
            let tx = db.transaction()?;
            tx.execute("INSERT INTO memory(id,kind,content,metadata_json,created_at,updated_at) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(id) DO UPDATE SET kind=excluded.kind,content=excluded.content,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at", params![record.id.to_string(),serde_json::to_value(record.kind)?.as_str(),record.content,serde_json::to_string(&record.metadata)?,record.created_at.to_rfc3339(),record.updated_at.to_rfc3339()])?;
            tx.execute("INSERT INTO memory_records(id,subject,confidence,provenance_json,enabled) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(id) DO UPDATE SET subject=excluded.subject,confidence=excluded.confidence,provenance_json=excluded.provenance_json,enabled=excluded.enabled", params![record.id.to_string(),record.subject,record.confidence,serde_json::to_string(&record.provenance)?,record.enabled])?;
            tx.commit()?;
            Ok(())
        })?;
        self.append_audit(
            None,
            None,
            "memory_saved",
            &serde_json::json!({"id":record.id,"kind":record.kind}),
        )?;
        Ok(record)
    }

    pub fn memories(
        &self,
        query: Option<&str>,
        enabled_only: bool,
        limit: usize,
    ) -> CoreResult<Vec<MemoryRecord>> {
        self.memories_scoped(query, enabled_only, limit, None)
    }

    pub fn memories_scoped(
        &self,
        query: Option<&str>,
        enabled_only: bool,
        limit: usize,
        scope: Option<Uuid>,
    ) -> CoreResult<Vec<MemoryRecord>> {
        if enabled_only && !self.memory_enabled()? {
            return Ok(Vec::new());
        }
        let search = query.map(fts_query).filter(|s| !s.is_empty());
        if query.is_some() && search.is_none() {
            return Ok(Vec::new());
        }
        self.with_connection(|db| {
            let sql = if search.is_some() {
                "SELECT m.id,m.kind,m.content,m.metadata_json,m.created_at,m.updated_at,r.subject,r.confidence,r.provenance_json,r.enabled FROM memory m JOIN memory_records r ON r.id=m.id JOIN memory_fts f ON f.memory_id=m.id WHERE memory_fts MATCH ?1 AND (?2=0 OR r.enabled=1) AND (?4 IS NULL OR json_extract(m.metadata_json,'$.scope_id')=?4 OR (m.kind!='episodic' AND json_extract(m.metadata_json,'$.scope_id') IS NULL)) ORDER BY rank,m.updated_at DESC LIMIT ?3"
            } else {
                "SELECT m.id,m.kind,m.content,m.metadata_json,m.created_at,m.updated_at,r.subject,r.confidence,r.provenance_json,r.enabled FROM memory m JOIN memory_records r ON r.id=m.id WHERE (?1 IS NULL) AND (?2=0 OR r.enabled=1) AND (?4 IS NULL OR json_extract(m.metadata_json,'$.scope_id')=?4 OR (m.kind!='episodic' AND json_extract(m.metadata_json,'$.scope_id') IS NULL)) ORDER BY m.updated_at DESC LIMIT ?3"
            };
            let mut statement = db.prepare(sql)?;
            let mut rows = statement.query(params![search,enabled_only,limit.min(500),scope.map(|id|id.to_string())])?;
            let mut records = Vec::new();
            while let Some(row) = rows.next()? {
                let kind: String = row.get(1)?;
                records.push(MemoryRecord {
                    id: row.get::<_,String>(0)?.parse()?, kind: serde_json::from_value(serde_json::Value::String(kind)).unwrap_or(MemoryKind::Semantic),
                    content: row.get(2)?, metadata: serde_json::from_str(&row.get::<_,String>(3)?)?,
                    created_at: row.get::<_,String>(4)?.parse()?, updated_at: row.get::<_,String>(5)?.parse()?,
                    subject: row.get(6)?,confidence:row.get(7)?, provenance:serde_json::from_str(&row.get::<_,String>(8)?)?,enabled:row.get(9)?,
                });
            }
            Ok(records)
        })
    }

    pub fn delete_memory(&self, id: Uuid) -> CoreResult<()> {
        self.with_connection(|db| {
            let tx=db.transaction()?;
            tx.execute_batch("CREATE TEMP TABLE IF NOT EXISTS forgetting_ids(id TEXT PRIMARY KEY); DELETE FROM forgetting_ids; CREATE TEMP TABLE IF NOT EXISTS forgetting_tasks(id TEXT PRIMARY KEY); DELETE FROM forgetting_tasks;")?;
            tx.execute("INSERT INTO forgetting_ids WITH RECURSIVE children(id) AS (SELECT ?1 UNION SELECT l.child_id FROM memory_lineage l JOIN children c ON c.id=l.parent_id) SELECT id FROM children",params![id.to_string()])?;
            tx.execute_batch("INSERT OR IGNORE INTO forgetting_tasks SELECT task_id FROM context_memory_sources WHERE memory_id IN (SELECT id FROM forgetting_ids);")?;
            // Preserve explicit history for inspection, while preventing its
            // retained source or derived answers from re-entering model context.
            tx.execute_batch("INSERT OR IGNORE INTO excluded_context_messages SELECT messages.id FROM messages WHERE task_id IN (SELECT id FROM forgetting_tasks) OR id IN (SELECT json_extract(provenance_json,'$.source_id') FROM memory_records WHERE id IN (SELECT id FROM forgetting_ids)); DELETE FROM working_memory WHERE conversation_id IN (SELECT conversation_id FROM conversation_tasks WHERE task_id IN (SELECT id FROM forgetting_tasks)); UPDATE conversations SET summary='' WHERE id IN (SELECT conversation_id FROM conversation_tasks WHERE task_id IN (SELECT id FROM forgetting_tasks)); DELETE FROM private_artifacts WHERE task_id IN (SELECT id FROM forgetting_tasks); DELETE FROM memory WHERE id IN (SELECT id FROM forgetting_ids); DELETE FROM forgetting_ids; DELETE FROM forgetting_tasks;")?;
            tx.execute_batch("DELETE FROM working_memory WHERE conversation_id IN (SELECT conversation_id FROM messages JOIN excluded_context_messages ON messages.id=message_id);")?;
            tx.commit()?;Ok(())
        })?;
        // Content is intentionally absent from audit so forgetting does not leave a second copy.
        self.append_audit(None, None, "memory_deleted", &serde_json::json!({"id":id}))?;
        Ok(())
    }

    pub fn set_memory_record_enabled(&self, id: Uuid, enabled: bool) -> CoreResult<()> {
        self.with_connection(|db| {
            db.execute(
                "UPDATE memory_records SET enabled=?2 WHERE id=?1",
                params![id.to_string(), enabled],
            )?;
            Ok(())
        })
    }

    pub fn record_context_memories(&self, task: Uuid, memories: &[MemoryRecord]) -> CoreResult<()> {
        self.with_connection(|db| {
            let tx = db.transaction()?;
            for memory in memories {
                tx.execute(
                    "INSERT OR IGNORE INTO context_memory_sources VALUES(?1,?2)",
                    params![task.to_string(), memory.id.to_string()],
                )?;
            }
            tx.commit()?;
            Ok(())
        })
    }
    pub fn context_messages(&self, conversation: Uuid, limit: usize) -> CoreResult<Vec<Message>> {
        let excluded=self.with_connection(|db| {
            let mut statement=db.prepare("SELECT message_id FROM excluded_context_messages JOIN messages ON messages.id=message_id WHERE conversation_id=?1")?;
            Ok(statement.query_map(params![conversation.to_string()],|r|r.get::<_,String>(0))?.collect::<Result<std::collections::BTreeSet<_>,_>>()?)
        })?;
        Ok(self
            .messages(conversation, limit)?
            .into_iter()
            .filter(|m| !excluded.contains(&m.id.to_string()))
            .collect())
    }

    pub fn record_history_sources(&self, task: Uuid, messages: &[Message]) -> CoreResult<()> {
        self.with_connection(|db| {
            let tx=db.transaction()?;
            for message in messages {if let Some(previous)=message.task_id {
                tx.execute("INSERT OR IGNORE INTO context_memory_sources SELECT ?1,memory_id FROM context_memory_sources WHERE task_id=?2",params![task.to_string(),previous.to_string()])?;
            }}
            tx.commit()?;Ok(())
        })
    }

    pub fn remember(&self, text: &str, source: Uuid) -> CoreResult<MemoryRecord> {
        let text = safe_memory_text(text)?;
        let pref = preference(&text);
        let (kind, subject, content) = match pref {
            Some((subject, value)) => (MemoryKind::Preference, subject, value),
            None => (MemoryKind::Semantic, clipped(&text, 80), text),
        };
        let previous = self.memories(None, false, 500)?.into_iter().find(|memory| {
            memory.kind == kind
                && (if kind == MemoryKind::Preference {
                    memory.subject == subject
                } else {
                    memory.content.eq_ignore_ascii_case(&content)
                })
        });
        self.save_memory(MemoryRecord {
            id: previous.as_ref().map_or_else(Uuid::new_v4, |m| m.id),
            kind,
            subject,
            content,
            confidence: 1.0,
            provenance: Provenance::external(ProvenanceSource::Message, source.to_string()),
            enabled: true,
            created_at: previous.map_or_else(Utc::now, |m| m.created_at),
            updated_at: Utc::now(),
            metadata: BTreeMap::new(),
        })
    }

    /// Only explicit user text is inspected for preferences. External context,
    /// summaries and model proposals can never invoke these commands.
    pub fn memory_command(&self, text: &str, source: Uuid) -> CoreResult<Option<String>> {
        let text = text.trim();
        let lower = text.to_ascii_lowercase();
        if let Some(prefix) = ["remember that ", "remember "]
            .iter()
            .find(|prefix| lower.starts_with(**prefix))
        {
            let memory = self.remember(&text[prefix.len()..], source)?;
            return Ok(Some(format!("Remembered: {}", memory.content)));
        }
        if lower.trim_end_matches(['?', '.']) == "what do you remember about me" {
            let records = self.memories(None, true, 100)?;
            return Ok(Some(if records.is_empty() {
                "There are no enabled memories.".into()
            } else {
                records
                    .iter()
                    .map(|m| format!("{}: {}", m.subject, m.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            }));
        }
        if lower.starts_with("forget ") {
            let subject = text[7..].trim().trim_end_matches('.');
            let records = self.memories(None, false, 500)?;
            let exact: Vec<_> = records
                .into_iter()
                .filter(|m| {
                    m.id.to_string() == subject
                        || m.subject.eq_ignore_ascii_case(subject)
                        || m.content.eq_ignore_ascii_case(subject)
                })
                .collect();
            if exact.len() != 1 {
                return Ok(Some("Choose the exact memory in Settings → Memory, or use “Forget” followed by its subject or ID.".into()));
            }
            self.delete_memory(exact[0].id)?;
            return Ok(Some("Forgot that memory.".into()));
        }
        if self.memory_enabled()? && preference(text).is_some() {
            let _ = self.remember(text, source)?;
        }
        Ok(None)
    }

    pub fn working_memory(&self, conversation: Uuid) -> CoreResult<Option<WorkingMemory>> {
        self.with_connection(|db| {
            db.execute(
                "DELETE FROM working_memory WHERE expires_at<=?1",
                params![Utc::now().to_rfc3339()],
            )?;
            let json: Option<String> = db
                .query_row(
                    "SELECT content_json FROM working_memory WHERE conversation_id=?1",
                    params![conversation.to_string()],
                    |row| row.get(0),
                )
                .optional()?;
            let mut memory: Option<WorkingMemory> =
                json.map(|json| serde_json::from_str(&json)).transpose()?;
            if let Some(memory) = memory.as_mut()
                && memory
                    .selection_expires_at
                    .is_none_or(|expires| expires <= Utc::now())
            {
                memory.selections.clear();
            }
            Ok(memory)
        })
    }

    pub fn update_working_memory(&self, task: &Task) -> CoreResult<()> {
        let Some(conversation_id) = task.conversation_id else {
            return Ok(());
        };
        let mut memory = self
            .working_memory(conversation_id)?
            .unwrap_or(WorkingMemory {
                conversation_id,
                task_id: task.id,
                goal: String::new(),
                resources: Vec::new(),
                applications: Vec::new(),
                entities: Vec::new(),
                recent_actions: Vec::new(),
                selections: Vec::new(),
                selection_expires_at: None,
                expires_at: Utc::now(),
            });
        memory.task_id = task.id;
        memory.goal = clipped(task.goal.as_deref().unwrap_or(&task.request), 500);
        for state in task
            .actions
            .values()
            .filter(|state| state.status == crate::domain::ActionStatus::Succeeded)
        {
            let resource = redact_for_persistence(&state.proposal.target_resource);
            if !memory.resources.contains(&resource) {
                memory.resources.push(resource);
            }
            let summary = clipped(
                &redact_for_persistence(state.summary.as_deref().unwrap_or("Verified action")),
                300,
            );
            if !memory.recent_actions.contains(&summary) {
                memory.recent_actions.push(summary);
            }
            let application = match &state.proposal.action {
                crate::domain::Action::OpenApplication { application }
                | crate::domain::Action::ClickElement { application, .. }
                | crate::domain::Action::TypeText { application, .. } => Some(application),
                _ => None,
            };
            if let Some(app) = application
                && !memory.applications.contains(app)
            {
                memory.applications.push(app.clone());
            }
        }
        for values in [
            &mut memory.resources,
            &mut memory.recent_actions,
            &mut memory.applications,
        ] {
            if values.len() > 16 {
                values.drain(..values.len() - 16);
            }
        }
        memory.expires_at = Utc::now() + Duration::hours(2);
        self.with_connection(|db| { db.execute("INSERT INTO working_memory VALUES(?1,?2,?3) ON CONFLICT(conversation_id) DO UPDATE SET content_json=excluded.content_json,expires_at=excluded.expires_at",params![conversation_id.to_string(),serde_json::to_string(&memory)?,memory.expires_at.to_rfc3339()])?; Ok(()) })
    }

    pub fn record_outcome(&self, task: &Task) -> CoreResult<()> {
        let Some(conversation) = task.conversation_id else {
            return Ok(());
        };
        self.append_message(&Message {
            id: Uuid::new_v4(),
            conversation_id: conversation,
            task_id: Some(task.id),
            role: "assistant".into(),
            content: task.final_outcome.clone().unwrap_or_default(),
            provenance: Provenance::external(ProvenanceSource::SageCore, task.id.to_string()),
            created_at: Utc::now(),
        })?;
        if !is_memory_command(&task.request) {
            self.update_working_memory(task)?;
        }
        // Rolling extractive summary: bounded factual outcomes, no additional model call.
        self.with_connection(|db| {
            let previous:String=db.query_row("SELECT summary FROM conversations WHERE id=?1",params![conversation.to_string()],|row|row.get(0))?;
            let mut facts:Vec<serde_json::Value>=serde_json::from_str(&previous).unwrap_or_default();
            if !facts.iter().any(|fact|fact["task_id"].as_str()==Some(&task.id.to_string())) && !is_memory_command(&task.request) {
                facts.push(serde_json::json!({"task_id":task.id,"goal":clipped(task.goal.as_deref().unwrap_or(&task.request),180),"outcome":clipped(task.final_outcome.as_deref().unwrap_or(""),220),"resources":task.actions.values().filter(|s|s.status==crate::domain::ActionStatus::Succeeded).take(4).map(|s|clipped(&s.proposal.target_resource,180)).collect::<Vec<_>>() }));
            }
            while facts.len()>1 && serde_json::to_string(&facts)?.len()>SUMMARY_CHARS {facts.remove(0);}
            db.execute("UPDATE conversations SET summary=?2 WHERE id=?1",params![conversation.to_string(),serde_json::to_string(&facts)?])?;Ok(())
        })?;
        if task.status == TaskStatus::Succeeded
            && task.completed_count() > 0
            && self.memory_enabled()?
        {
            let content = format!(
                "{} — {}",
                task.goal.as_deref().unwrap_or(&task.request),
                task.final_outcome
                    .as_deref()
                    .unwrap_or("Verified completion")
            );
            if safe_memory_text(&content).is_ok() {
                self.save_memory(MemoryRecord {
                    id: task.id,
                    kind: MemoryKind::Episodic,
                    subject: clipped(task.goal.as_deref().unwrap_or(&task.request), 100),
                    content: clipped(&content, 1500),
                    confidence: 0.9,
                    provenance: Provenance::external(
                        ProvenanceSource::SageCore,
                        task.id.to_string(),
                    ),
                    enabled: true,
                    created_at: task.created_at,
                    updated_at: Utc::now(),
                    metadata: BTreeMap::from([
                        ("task_id".into(), task.id.to_string()),
                        ("scope_id".into(), conversation.to_string()),
                    ]),
                })?;
                self.with_connection(|db| {db.execute("INSERT OR IGNORE INTO memory_lineage SELECT memory_id,task_id FROM context_memory_sources WHERE task_id=?1 AND memory_id!=task_id",params![task.id.to_string()])?;Ok(())})?;
            }
        }
        Ok(())
    }
}

pub fn is_memory_command(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    lower.starts_with("remember ")
        || lower.starts_with("forget ")
        || lower.starts_with("what do you remember about me")
}

fn conversation_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Conversation> {
    use rusqlite::types::Type;
    let parse_error = |error| rusqlite::Error::FromSqlConversionFailure(0, Type::Text, error);
    Ok(Conversation {
        id: row
            .get::<_, String>(0)?
            .parse()
            .map_err(|e| parse_error(Box::new(e)))?,
        title: row.get(1)?,
        summary: row.get(2)?,
        pinned: row.get(3)?,
        archived: row.get(4)?,
        created_at: row
            .get::<_, String>(5)?
            .parse()
            .map_err(|e| parse_error(Box::new(e)))?,
        updated_at: row
            .get::<_, String>(6)?
            .parse()
            .map_err(|e| parse_error(Box::new(e)))?,
    })
}

fn fts_query(query: &str) -> String {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.chars().count() > 1)
        .take(16)
        .map(|word| format!("\"{}\"", clipped(word, 80)))
        .collect::<Vec<_>>()
        .join(" OR ")
}

#[cfg(test)]
mod v2_tests {
    use super::*;
    fn store(path: &std::path::Path) -> LocalStore {
        let store =
            LocalStore::open_encrypted(path, &crate::secrets::SecretBytes::new(vec![31; 32]))
                .unwrap();
        store.migrate_knowledge().unwrap();
        store
    }
    #[test]
    fn permission_filter_runs_before_rank_and_limit() {
        let directory = tempfile::tempdir().unwrap();
        let store = store(&directory.path().join("vault.db"));
        let alpha = Uuid::new_v4();
        let beta = Uuid::new_v4();
        for (scope, content) in [
            (alpha, "orchid allowed"),
            (beta, "orchid forbidden orchid orchid"),
        ] {
            store
                .save_memory(MemoryRecord {
                    id: Uuid::new_v4(),
                    kind: MemoryKind::Episodic,
                    subject: "orchid".into(),
                    content: content.into(),
                    confidence: 1.0,
                    provenance: Provenance::user(),
                    enabled: true,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    metadata: BTreeMap::from([("scope_id".into(), scope.to_string())]),
                })
                .unwrap();
        }
        let found = store
            .memories_scoped(Some("orchid"), true, 1, Some(alpha))
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].content, "orchid allowed");
    }
    #[test]
    fn forgetting_removes_derived_memory_and_excludes_its_context_lineage() {
        let directory = tempfile::tempdir().unwrap();
        let store = store(&directory.path().join("vault.db"));
        let conversation = store.ensure_conversation(None, "test").unwrap();
        let mut source = Task::new("remember orchid");
        source.conversation_id = Some(conversation.id);
        store.save_task(&source).unwrap();
        store.link_task(&source).unwrap();
        let message = Message {
            id: Uuid::new_v4(),
            conversation_id: conversation.id,
            task_id: Some(source.id),
            role: "user".into(),
            content: "remember orchid".into(),
            provenance: Provenance::user(),
            created_at: Utc::now(),
        };
        store.append_message(&message).unwrap();
        let memory = store.remember("orchid", message.id).unwrap();
        let mut derived = Task::new("use the memory");
        derived.conversation_id = Some(conversation.id);
        store.save_task(&derived).unwrap();
        store.link_task(&derived).unwrap();
        store
            .record_context_memories(derived.id, std::slice::from_ref(&memory))
            .unwrap();
        let answer = Message {
            id: Uuid::new_v4(),
            conversation_id: conversation.id,
            task_id: Some(derived.id),
            role: "assistant".into(),
            content: "orchid is the remembered value".into(),
            provenance: Provenance::user(),
            created_at: Utc::now(),
        };
        store.append_message(&answer).unwrap();
        let child = store.remember("derived orchid value", answer.id).unwrap();
        store
            .with_connection(|db| {
                db.execute(
                    "INSERT INTO memory_lineage VALUES(?1,?2)",
                    params![memory.id.to_string(), child.id.to_string()],
                )?;
                Ok(())
            })
            .unwrap();
        store.delete_memory(memory.id).unwrap();
        assert!(store.memories(None, false, 500).unwrap().is_empty());
        assert!(
            store
                .context_messages(conversation.id, 100)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            store.messages(conversation.id, 100).unwrap().len(),
            2,
            "inspectable source history is retained"
        );
    }
}
