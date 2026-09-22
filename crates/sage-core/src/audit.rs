//! Hash-chain verification anchored outside the database in the OS secret store.
//! A compromised OS or a process allowed to rewrite both stores is outside this
//! boundary. Verification never describes the journal as immutable.
use crate::secrets::{SecretBytes, SecretStore};
use crate::storage::LocalStore;
use crate::{CoreError, CoreResult};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    version: u32,
    database_id: Uuid,
    sequence: u64,
    hash: String,
}

fn integrity_error() -> CoreError {
    CoreError::Storage("Audit integrity could not be established. Execution is paused; restore the database and its matching OS checkpoint.".into())
}

impl LocalStore {
    fn audit_account(&self) -> String {
        format!(
            "audit:v2:{:x}",
            Sha256::digest(self.database_path().as_os_str().as_encoded_bytes())
        )
    }

    /// Verify the full chain and the prior protected anchor before advancing it.
    /// Serialized with other checkpoint writers; concurrent appends may leave a
    /// valid unanchored tail, which the next checkpoint or unlock verifies.
    pub fn checkpoint_audit(&self, secrets: &dyn SecretStore) -> CoreResult<()> {
        let _guard = self.audit_lock.lock().map_err(|_| integrity_error())?;
        let account = self.audit_account();
        let prior = secrets
            .get(&account)?
            .map(|bytes| serde_json::from_slice::<Checkpoint>(bytes.expose()))
            .transpose()
            .map_err(|_| integrity_error())?;
        let expected: bool = self
            .load_setting("audit.checkpoint_expected")?
            .unwrap_or(false);
        if expected && prior.is_none() {
            return Err(integrity_error());
        }
        let database_id: Uuid = match self.load_setting("audit.database_id")? {
            Some(id) => id,
            None if prior.is_none() && !expected => {
                let id = Uuid::new_v4();
                self.save_setting("audit.database_id", &id)?;
                id
            }
            None => return Err(integrity_error()),
        };
        if prior
            .as_ref()
            .is_some_and(|p| p.version != 2 || p.database_id != database_id)
        {
            return Err(integrity_error());
        }
        let (sequence, hash) = self.with_connection(|db| {
            let mut statement = db.prepare("SELECT sequence,id,task_id,action_id,event_type,redacted_payload_json,previous_hash,record_hash,occurred_at FROM audit_log ORDER BY sequence")?;
            let mut rows = statement.query([])?;
            let mut sequence = 0;
            let mut previous = "GENESIS".to_string();
            let mut anchored = prior.as_ref().is_none_or(|p| p.sequence == 0 && p.hash == previous);
            while let Some(row) = rows.next()? {
                let next: u64 = row.get(0)?;
                let id: String = row.get(1)?;
                let task: Option<String> = row.get(2)?;
                let action: Option<String> = row.get(3)?;
                let event: String = row.get(4)?;
                let payload: String = row.get(5)?;
                let link: String = row.get(6)?;
                let hash: String = row.get(7)?;
                let at: String = row.get(8)?;
                let canonical = format!("{id}|{}|{}|{event}|{payload}|{link}|{at}", task.unwrap_or_default(), action.unwrap_or_default());
                if next != sequence + 1 || link != previous || format!("{:x}", Sha256::digest(canonical.as_bytes())) != hash { return Err(integrity_error().into()); }
                if prior.as_ref().is_some_and(|p| p.sequence == next && p.hash == hash) { anchored = true; }
                sequence = next;
                previous = hash;
            }
            if !anchored { return Err(integrity_error().into()); }
            Ok((sequence, previous))
        })?;
        let checkpoint = Checkpoint {
            version: 2,
            database_id,
            sequence,
            hash,
        };
        secrets.set(
            &account,
            &SecretBytes::new(serde_json::to_vec(&checkpoint)?),
        )?;
        self.save_setting("audit.checkpoint_expected", &true)
    }

    pub fn retire_expired_artifacts(&self) -> CoreResult<usize> {
        self.with_connection(|db| {
            Ok(db.execute(
                "DELETE FROM private_artifacts WHERE expires_at<=?1",
                params![chrono::Utc::now().to_rfc3339()],
            )?)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::testing::MemorySecretStore;
    #[test]
    fn protected_checkpoint_rejects_truncation_tampering_and_missing_anchor() {
        for attack in ["truncate", "tamper", "missing"] {
            let directory = tempfile::tempdir().unwrap();
            let store = LocalStore::open_encrypted(
                &directory.path().join("history.db"),
                &SecretBytes::new(vec![19; 32]),
            )
            .unwrap();
            let secrets = MemorySecretStore::default();
            store
                .append_audit(
                    None,
                    None,
                    "dispatch",
                    &serde_json::json!({"target":"document"}),
                )
                .unwrap();
            store.checkpoint_audit(&secrets).unwrap();
            store.append_audit(None, None, "verified", &true).unwrap();
            store.checkpoint_audit(&secrets).unwrap();
            match attack {
                "truncate" => store
                    .with_connection(|db| {
                        db.execute("DELETE FROM audit_log WHERE sequence=2", [])?;
                        Ok(())
                    })
                    .unwrap(),
                "tamper" => store
                    .with_connection(|db| {
                        db.execute(
                            "UPDATE audit_log SET redacted_payload_json='false' WHERE sequence=1",
                            [],
                        )?;
                        Ok(())
                    })
                    .unwrap(),
                _ => secrets.delete(&store.audit_account()).unwrap(),
            }
            assert!(store.checkpoint_audit(&secrets).is_err(), "{attack}");
        }
    }
}
