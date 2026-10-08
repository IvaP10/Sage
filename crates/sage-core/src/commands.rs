//! Durable identities for accepted interactive commands. Transport replay
//! protection and effect idempotency are separate from command deduplication.
use prost::Message;
use rusqlite::{OptionalExtension, params};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::storage::LocalStore;
use crate::{CoreError, CoreResult};

#[derive(Debug, Clone)]
pub(crate) struct SubmissionKey {
    pub id: Uuid,
    pub digest: String,
}

impl SubmissionKey {
    pub fn for_request(
        id: &str,
        request: &sage_protocol::sage::ipc::v2::SubmitTask,
    ) -> CoreResult<Self> {
        let id = Uuid::parse_str(id).map_err(|_| {
            CoreError::Protocol("Submission requires a UUID request identity".into())
        })?;
        let mut digest = Sha256::new();
        digest.update(b"sage:interactive-submit:v1\0");
        digest.update(request.encode_to_vec());
        Ok(Self {
            id,
            digest: format!("{:x}", digest.finalize()),
        })
    }

    /// Bind a world-model execution command to its inner request identity.
    /// The request envelope already rejects replayed UI ids; this durable key
    /// makes retries after reconnect return the accepted task instead of
    /// dispatching a second copy.
    pub fn for_world_model(id: &str, operation: &str, json: &str) -> CoreResult<Self> {
        let id = Uuid::parse_str(id).map_err(|_| {
            CoreError::Protocol("World-model execution requires a UUID request identity".into())
        })?;
        if operation.len() > 64 || json.len() > 64 * 1024 {
            return Err(CoreError::Protocol(
                "World-model execution request exceeds its bound".into(),
            ));
        }
        let mut digest = Sha256::new();
        digest.update(b"sage:world-model-execution:v1\0");
        digest.update(operation.as_bytes());
        digest.update([0]);
        digest.update(json.as_bytes());
        Ok(Self {
            id,
            digest: format!("{:x}", digest.finalize()),
        })
    }
}

impl LocalStore {
    pub(crate) fn accepted_submission(&self, key: &SubmissionKey) -> CoreResult<Option<Uuid>> {
        let prior: Option<(String, String)> = self.with_connection(|db| {
            Ok(db.query_row(
                "SELECT payload_digest,task_id FROM command_inbox WHERE principal='local-user' AND request_id=?1",
                params![key.id.to_string()], |row| Ok((row.get(0)?,row.get(1)?))).optional()?)
        })?;
        match prior {
            Some((digest, task)) if digest == key.digest => {
                Ok(Some(task.parse().map_err(|_| {
                    CoreError::Storage("Invalid accepted task identity".into())
                })?))
            }
            Some(_) => Err(CoreError::Protocol(
                "The request identity was already used for different content".into(),
            )),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Provenance, Task};
    use crate::events::{CoreEvent, CoreEventKind};
    use crate::knowledge::Message;
    use crate::secrets::SecretBytes;

    fn records(conversation: Uuid) -> (Task, Message, CoreEvent) {
        let mut task = Task::new("Accept one durable request");
        task.conversation_id = Some(conversation);
        task.message_id = Some(Uuid::new_v4());
        let message = Message {
            id: task.message_id.unwrap(),
            conversation_id: conversation,
            task_id: Some(task.id),
            role: "user".into(),
            content: task.request.clone(),
            provenance: Provenance::user(),
            created_at: task.created_at,
        };
        let event = CoreEvent::new(Some(task.id), CoreEventKind::TaskStarted);
        (task, message, event)
    }

    #[test]
    fn acceptance_is_atomic_and_receipts_survive_reopening() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.db");
        let secret = SecretBytes::new(vec![17; 32]);
        let store = LocalStore::open_encrypted(&path, &secret).unwrap();
        store.migrate_knowledge().unwrap();
        let conversation = store.ensure_conversation(None, "fixture").unwrap();
        let key = SubmissionKey::for_request(
            &Uuid::new_v4().to_string(),
            &sage_protocol::sage::ipc::v2::SubmitTask {
                text: "fixture".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let (mut first, message, event) = records(conversation.id);
        store
            .accept_task(&mut first, &message, &event, Some(&key))
            .unwrap();

        // Force a failure at the final INSERT, after the task, user message,
        // conversation link and event have all been written in the transaction.
        let (mut second, message, event) = records(conversation.id);
        assert!(
            store
                .accept_task(&mut second, &message, &event, Some(&key))
                .is_err()
        );
        store
            .with_connection(|db| {
                for table in [
                    "tasks",
                    "messages",
                    "conversation_tasks",
                    "events",
                    "command_inbox",
                ] {
                    let count: i64 =
                        db.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                            row.get(0)
                        })?;
                    assert_eq!(count, 1, "partial acceptance leaked into {table}");
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(store.accepted_submission(&key).unwrap(), Some(first.id));
        let changed = SubmissionKey {
            id: key.id,
            digest: "different payload".into(),
        };
        assert!(matches!(
            store.accepted_submission(&changed),
            Err(CoreError::Protocol(_))
        ));
        drop(store);

        let reopened = LocalStore::open_encrypted(&path, &secret).unwrap();
        assert_eq!(reopened.accepted_submission(&key).unwrap(), Some(first.id));
    }
}
