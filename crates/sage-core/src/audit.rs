//! Hash-chain verification anchored outside the database in the OS secret store.
//! A connection-local cursor avoids rereading an unchanged, verified prefix.
//! Row hooks and SQLite change counters invalidate it; restart always scans all
//! records. The cursor is never persisted or accepted as a protected anchor.
//! Each incremental checkpoint also scrubs a bounded part of the old prefix.
//! A compromised OS or a process allowed to rewrite both stores is outside this
//! boundary. Verification never describes the journal as immutable.
use crate::secrets::{SecretBytes, SecretStore};
use crate::storage::LocalStore;
use crate::{CoreError, CoreResult};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, hooks::PreUpdateCase, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicI64, AtomicU64, Ordering},
};
use uuid::Uuid;

const SCRUB_ROWS: u64 = 128;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    version: u32,
    database_id: Uuid,
    sequence: u64,
    hash: String,
}

#[derive(Default)]
pub(crate) struct AuditState {
    verified: Option<VerifiedHead>,
    #[cfg(test)]
    last_verified_rows: u64,
    #[cfg(test)]
    last_full_scan: bool,
    #[cfg(test)]
    last_scrubbed_rows: u64,
}

#[derive(Default)]
pub(crate) struct AuditWatch {
    revision: AtomicU64,
    verified_sequence: AtomicI64,
}

impl AuditWatch {
    fn invalidate(&self) {
        // Saturation disables cursor reuse permanently rather than wrapping to
        // a previously accepted revision. The hook does no SQL or locking.
        let _ = self
            .revision
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                Some(old.saturating_add(1))
            });
    }
}

pub(crate) fn install_watch(db: &Connection, watch: Arc<AuditWatch>) {
    watch.invalidate();
    watch.verified_sequence.store(0, Ordering::Release);
    db.preupdate_hook(Some(
        move |_, database: &str, table: &str, case: &PreUpdateCase| {
            if database == "main" && table.eq_ignore_ascii_case("audit_log") {
                match case {
                    PreUpdateCase::Insert(row)
                        if row.get_new_row_id()
                            > watch.verified_sequence.load(Ordering::Acquire) => {}
                    _ => watch.invalidate(),
                }
            }
        },
    ));
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Stamp {
    external_version: i64,
    schema_version: i64,
    mutation_revision: u64,
}

impl Stamp {
    fn read(db: &Connection, watch: &AuditWatch) -> CoreResult<Self> {
        Ok(Self {
            external_version: db.query_row("PRAGMA main.data_version", [], |row| row.get(0))?,
            schema_version: db.query_row("PRAGMA main.schema_version", [], |row| row.get(0))?,
            mutation_revision: watch.revision.load(Ordering::Acquire),
        })
    }
}

struct VerifiedHead {
    checkpoint: Checkpoint,
    stamp: Stamp,
    scrub: Scrub,
}

#[derive(Clone)]
struct Scrub {
    sequence: u64,
    hash: String,
    end_sequence: u64,
    end_hash: String,
}

impl Scrub {
    fn new(head: &Checkpoint) -> Self {
        Self {
            sequence: 0,
            hash: "GENESIS".into(),
            end_sequence: head.sequence,
            end_hash: head.hash.clone(),
        }
    }

    fn advance(&mut self, db: &Connection) -> CoreResult<u64> {
        let mut statement = db.prepare("SELECT sequence,id,task_id,action_id,event_type,redacted_payload_json,previous_hash,record_hash,occurred_at FROM main.audit_log WHERE sequence>?1 AND sequence<=?2 ORDER BY sequence LIMIT ?3")?;
        let mut rows = statement.query(params![self.sequence, self.end_sequence, SCRUB_ROWS])?;
        let mut count = 0;
        while let Some(row) = rows.next()? {
            let (sequence, hash) = verify_record(row, self.sequence, &self.hash)?;
            self.sequence = sequence;
            self.hash = hash;
            count += 1;
        }
        if (self.sequence == self.end_sequence && self.hash != self.end_hash)
            || (count < SCRUB_ROWS && self.sequence != self.end_sequence)
        {
            return Err(integrity_error());
        }
        Ok(count)
    }
}

struct Verification {
    head: VerifiedHead,
    expected: bool,
    #[cfg(test)]
    rows: u64,
    #[cfg(test)]
    full_scan: bool,
    #[cfg(test)]
    scrubbed_rows: u64,
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

    /// Verify new records against this connection's unchanged verified prefix,
    /// or the entire chain after any loss of that proof. The OS anchor is read
    /// every time and advanced only after verification. Concurrent appends may
    /// leave an unanchored tail for the next checkpoint; no-op calls do not
    /// rewrite an identical OS credential.
    pub fn checkpoint_audit(&self, secrets: &dyn SecretStore) -> CoreResult<()> {
        let mut state = self.audit_lock.lock().map_err(|_| integrity_error())?;
        // Any failure retires the cursor, including ambiguous credential writes.
        let cached = state.verified.take();
        let account = self.audit_account();
        let prior = secrets
            .get(&account)?
            .map(|bytes| serde_json::from_slice::<Checkpoint>(bytes.expose()))
            .transpose()
            .map_err(|_| integrity_error())?;
        let verification = self.with_connection(|db| {
            Ok(verify(
                db,
                &self.audit_watch,
                cached.as_ref(),
                prior.as_ref(),
            )?)
        })?;
        if prior.as_ref() != Some(&verification.head.checkpoint) {
            secrets.set(
                &account,
                &SecretBytes::new(serde_json::to_vec(&verification.head.checkpoint)?),
            )?;
        }
        if !verification.expected {
            self.save_setting("audit.checkpoint_expected", &true)?;
        }
        // Credential calls run without the database mutex. A prefix/schema or
        // external change during that interval must not be accepted as current.
        self.with_connection(|db| {
            if Stamp::read(db, &self.audit_watch)? != verification.head.stamp
                || setting::<Uuid>(db, "audit.database_id")?
                    != Some(verification.head.checkpoint.database_id)
                || setting::<bool>(db, "audit.checkpoint_expected")? != Some(true)
            {
                return Err(integrity_error().into());
            }
            Ok(())
        })?;
        #[cfg(test)]
        {
            state.last_verified_rows = verification.rows;
            state.last_full_scan = verification.full_scan;
            state.last_scrubbed_rows = verification.scrubbed_rows;
        }
        state.verified = Some(verification.head);
        Ok(())
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

fn setting<T: serde::de::DeserializeOwned>(db: &Connection, key: &str) -> CoreResult<Option<T>> {
    db.query_row(
        "SELECT value_json FROM main.settings WHERE key=?1",
        [key],
        |row| row.get::<_, String>(0),
    )
    .optional()?
    .map(|value| serde_json::from_str(&value).map_err(CoreError::from))
    .transpose()
}

fn verify(
    db: &mut Connection,
    watch: &AuditWatch,
    cached: Option<&VerifiedHead>,
    prior: Option<&Checkpoint>,
) -> CoreResult<Verification> {
    // Reserve the writer while reading the counters, metadata and chain. This
    // closes the race between checking data_version and acquiring a snapshot.
    let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let stamp = Stamp::read(&tx, watch)?;
    let expected = setting::<bool>(&tx, "audit.checkpoint_expected")?.unwrap_or(false);
    if expected && prior.is_none() {
        return Err(integrity_error());
    }
    let database_id = match setting::<Uuid>(&tx, "audit.database_id")? {
        Some(id) => id,
        None if prior.is_none() && !expected => {
            let id = Uuid::new_v4();
            tx.execute("INSERT INTO main.settings(key,value_json,updated_at) VALUES('audit.database_id',?1,?2)", params![serde_json::to_string(&id)?,chrono::Utc::now().to_rfc3339()])?;
            id
        }
        None => return Err(integrity_error()),
    };
    if prior.is_some_and(|p| {
        p.version != 2 || p.database_id != database_id || p.sequence > i64::MAX as u64
    }) {
        return Err(integrity_error());
    }
    let cached = cached.filter(|head| {
        prior == Some(&head.checkpoint)
            && head.checkpoint.database_id == database_id
            && head.stamp == stamp
            && stamp.mutation_revision != u64::MAX
    });
    let mut sequence = cached.map_or(0, |head| head.checkpoint.sequence);
    let mut previous = cached.map_or_else(|| "GENESIS".into(), |head| head.checkpoint.hash.clone());
    if sequence != 0 {
        let hash: Option<String> = tx
            .query_row(
                "SELECT record_hash FROM main.audit_log WHERE sequence=?1",
                [sequence],
                |row| row.get(0),
            )
            .optional()?;
        if hash.as_deref() != Some(&previous) {
            return Err(integrity_error());
        }
    }
    let mut anchored =
        cached.is_some() || prior.is_none_or(|p| p.sequence == 0 && p.hash == previous);
    #[cfg(test)]
    let mut count = 0;
    {
        let columns = "SELECT sequence,id,task_id,action_id,event_type,redacted_payload_json,previous_hash,record_hash,occurred_at FROM main.audit_log";
        // Separate statements preserve an indexed range scan for the tail.
        // A full scan also sees invalid zero/negative sequences.
        let sql = if cached.is_some() {
            format!("{columns} WHERE sequence>?1 ORDER BY sequence")
        } else {
            format!("{columns} ORDER BY sequence")
        };
        let mut statement = tx.prepare(&sql)?;
        let mut rows = if cached.is_some() {
            statement.query([sequence])?
        } else {
            statement.query([])?
        };
        while let Some(row) = rows.next()? {
            let (next, hash) = verify_record(row, sequence, &previous)?;
            if prior.is_some_and(|p| p.sequence == next && p.hash == hash) {
                anchored = true;
            }
            sequence = next;
            previous = hash;
            #[cfg(test)]
            {
                count += 1;
            }
        }
    }
    if !anchored {
        return Err(integrity_error());
    }
    let checkpoint = Checkpoint {
        version: 2,
        database_id,
        sequence,
        hash: previous,
    };
    let mut scrub = cached.map_or_else(|| Scrub::new(&checkpoint), |head| head.scrub.clone());
    #[cfg(test)]
    let mut scrubbed_rows = 0;
    if cached.is_some() {
        let _count = scrub.advance(&tx)?;
        #[cfg(test)]
        {
            scrubbed_rows = _count;
        }
        if scrub.sequence == scrub.end_sequence {
            scrub = Scrub::new(&checkpoint);
        }
    }
    tx.commit()?;
    watch.verified_sequence.store(
        sequence.try_into().map_err(|_| integrity_error())?,
        Ordering::Release,
    );
    Ok(Verification {
        head: VerifiedHead {
            checkpoint,
            stamp,
            scrub,
        },
        expected,
        #[cfg(test)]
        rows: count,
        #[cfg(test)]
        full_scan: cached.is_none(),
        #[cfg(test)]
        scrubbed_rows,
    })
}

fn verify_record(
    row: &rusqlite::Row<'_>,
    sequence: u64,
    previous: &str,
) -> CoreResult<(u64, String)> {
    let next: u64 = row.get(0)?;
    let id: String = row.get(1)?;
    let task: Option<String> = row.get(2)?;
    let action: Option<String> = row.get(3)?;
    let event: String = row.get(4)?;
    let payload: String = row.get(5)?;
    let link: String = row.get(6)?;
    let hash: String = row.get(7)?;
    let at: String = row.get(8)?;
    let canonical = format!(
        "{id}|{}|{}|{event}|{payload}|{link}|{at}",
        task.unwrap_or_default(),
        action.unwrap_or_default()
    );
    if sequence.checked_add(1) != Some(next)
        || link != previous
        || format!("{:x}", Sha256::digest(canonical.as_bytes())) != hash
    {
        return Err(integrity_error());
    }
    Ok((next, hash))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::testing::MemorySecretStore;
    use std::sync::{Mutex, atomic::AtomicU8};

    type DuringSet = Box<dyn FnOnce() + Send>;

    #[derive(Default)]
    struct CountingSecrets {
        inner: MemorySecretStore,
        gets: AtomicU64,
        sets: AtomicU64,
        fail_set: AtomicU8,
        during_set: Mutex<Option<DuringSet>>,
    }

    impl SecretStore for CountingSecrets {
        fn get(&self, account: &str) -> CoreResult<Option<SecretBytes>> {
            self.gets.fetch_add(1, Ordering::Relaxed);
            self.inner.get(account)
        }
        fn set(&self, account: &str, secret: &SecretBytes) -> CoreResult<()> {
            self.sets.fetch_add(1, Ordering::Relaxed);
            let failure = self.fail_set.swap(0, Ordering::Relaxed);
            if failure == 1 {
                return Err(CoreError::SecretStore("fixture before write".into()));
            }
            self.inner.set(account, secret)?;
            if let Some(callback) = self.during_set.lock().unwrap().take() {
                callback();
            }
            if failure == 2 {
                return Err(CoreError::SecretStore("fixture ambiguous write".into()));
            }
            Ok(())
        }
        fn delete(&self, account: &str) -> CoreResult<()> {
            self.inner.delete(account)
        }
    }

    fn fixture() -> (tempfile::TempDir, LocalStore, Arc<CountingSecrets>) {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalStore::open_encrypted(
            &directory.path().join("audit.db"),
            &SecretBytes::new(vec![19; 32]),
        )
        .unwrap();
        (directory, store, Arc::default())
    }

    fn seed(store: &LocalStore, count: u64) {
        store
            .with_connection(|db| {
                let tx = db.transaction()?;
                for index in 0..count {
                    crate::storage::write_audit(
                        &tx,
                        None,
                        None,
                        "fixture",
                        &serde_json::json!({"index":index}),
                    )?;
                }
                tx.commit()?;
                Ok(())
            })
            .unwrap();
    }

    fn stats(store: &LocalStore) -> (bool, u64, u64) {
        let state = store.audit_lock.lock().unwrap();
        (
            state.last_full_scan,
            state.last_verified_rows,
            state.last_scrubbed_rows,
        )
    }

    fn anchor(store: &LocalStore, secrets: &dyn SecretStore) -> Checkpoint {
        serde_json::from_slice(
            secrets
                .get(&store.audit_account())
                .unwrap()
                .unwrap()
                .expose(),
        )
        .unwrap()
    }

    #[test]
    fn checkpoints_bound_historical_work_skip_identical_writes_and_rescan_after_restart() {
        for count in [100, 1_000, 10_000] {
            let (directory, store, secrets) = fixture();
            seed(&store, count);
            let cold = std::time::Instant::now();
            store.checkpoint_audit(secrets.as_ref()).unwrap();
            let cold_us = cold.elapsed().as_micros();
            assert_eq!(stats(&store), (true, count, 0));
            let mut unchanged = Vec::new();
            let mut appended = Vec::new();
            for index in 0..5 {
                let writes = secrets.sets.load(Ordering::Relaxed);
                let reads = secrets.gets.load(Ordering::Relaxed);
                let started = std::time::Instant::now();
                store.checkpoint_audit(secrets.as_ref()).unwrap();
                unchanged.push(started.elapsed().as_micros());
                assert_eq!(secrets.sets.load(Ordering::Relaxed), writes);
                assert_eq!(secrets.gets.load(Ordering::Relaxed), reads + 1);
                let (full, new, scrubbed) = stats(&store);
                assert!(!full);
                assert_eq!(new, 0);
                assert!(scrubbed > 0 && scrubbed <= SCRUB_ROWS);
                store
                    .append_audit(None, None, "one_new_record", &index)
                    .unwrap();
                let started = std::time::Instant::now();
                store.checkpoint_audit(secrets.as_ref()).unwrap();
                appended.push(started.elapsed().as_micros());
                let (full, new, scrubbed) = stats(&store);
                assert!(!full);
                assert_eq!(new, 1);
                assert!(scrubbed <= SCRUB_ROWS);
            }
            println!(
                "AUDIT_PROBE {}",
                serde_json::json!({"history_rows":count,"initial_full_us":cold_us,"unchanged_us":unchanged,"one_new_record_us":appended,"historical_rows_per_incremental_checkpoint":SCRUB_ROWS,"profile":"debug, SQLCipher temporary database, in-memory secret store"})
            );
            drop(store);
            let reopened = LocalStore::open_encrypted(
                &directory.path().join("audit.db"),
                &SecretBytes::new(vec![19; 32]),
            )
            .unwrap();
            reopened.checkpoint_audit(secrets.as_ref()).unwrap();
            assert_eq!(stats(&reopened), (true, count + 5, 0));
            assert_eq!(secrets.sets.load(Ordering::Relaxed), 6);
        }
    }

    #[test]
    fn local_prefix_edits_deletes_replacements_and_invalid_sequences_retire_the_cursor() {
        for attack in [
            "edit",
            "delete_all",
            "cached_delete",
            "replace_unique_id",
            "negative_sequence",
        ] {
            let (_directory, store, secrets) = fixture();
            seed(&store, 3);
            store.checkpoint_audit(secrets.as_ref()).unwrap();
            if attack == "cached_delete" {
                store
                    .with_connection(|db| {
                        let tx = db.transaction()?;
                        tx.prepare_cached("DELETE FROM audit_log")?.execute([])?;
                        tx.rollback()?;
                        Ok(())
                    })
                    .unwrap();
                store.checkpoint_audit(secrets.as_ref()).unwrap();
                assert_eq!(stats(&store), (true, 3, 0));
            }
            let before = anchor(&store, secrets.as_ref());
            let revision = store.audit_watch.revision.load(Ordering::Acquire);
            store.with_connection(|db| {
                match attack {
                    "edit" => { db.execute("UPDATE audit_log SET redacted_payload_json='false' WHERE sequence=1", [])?; }
                    "delete_all" => { db.execute("DELETE FROM audit_log", [])?; }
                    "cached_delete" => { db.prepare_cached("DELETE FROM audit_log")?.execute([])?; }
                    "replace_unique_id" => {
                        let id: String = db.query_row("SELECT id FROM audit_log WHERE sequence=1", [], |r| r.get(0))?;
                        let previous: String = db.query_row("SELECT record_hash FROM audit_log WHERE sequence=3", [], |r| r.get(0))?;
                        let at = chrono::Utc::now().to_rfc3339();
                        let canonical = format!("{id}|||fixture|true|{previous}|{at}");
                        let hash = format!("{:x}",Sha256::digest(canonical.as_bytes()));
                        // The new tail is independently well-formed. The old
                        // row removed by REPLACE is what makes it invalid.
                        db.execute("INSERT OR REPLACE INTO audit_log(sequence,id,event_type,redacted_payload_json,previous_hash,record_hash,occurred_at) VALUES(4,?1,'fixture','true',?2,?3,?4)",params![id,previous,hash,at])?;
                    }
                    _ => { db.execute("INSERT INTO audit_log(sequence,id,event_type,redacted_payload_json,previous_hash,record_hash,occurred_at) VALUES(-1,'negative','fixture','true','GENESIS','invalid','now')", [])?; }
                }
                Ok(())
            }).unwrap();
            assert!(
                store.checkpoint_audit(secrets.as_ref()).is_err(),
                "{attack}"
            );
            assert!(
                store.audit_watch.revision.load(Ordering::Acquire) > revision,
                "{attack}"
            );
            assert!(store.audit_lock.lock().unwrap().verified.is_none());
            assert_eq!(anchor(&store, secrets.as_ref()), before);
        }
    }

    #[test]
    fn external_commits_and_schema_changes_require_full_verification() {
        let (directory, store, secrets) = fixture();
        seed(&store, 3);
        store.checkpoint_audit(secrets.as_ref()).unwrap();
        let other = crate::vault::open_encrypted(
            &directory.path().join("audit.db"),
            &SecretBytes::new(vec![19; 32]),
        )
        .unwrap();
        other
            .execute("INSERT INTO settings VALUES('outside','true','now')", [])
            .unwrap();
        store.checkpoint_audit(secrets.as_ref()).unwrap();
        assert_eq!(stats(&store), (true, 3, 0));
        crate::storage::write_audit(&other, None, None, "external_append", &true).unwrap();
        store.checkpoint_audit(secrets.as_ref()).unwrap();
        assert_eq!(stats(&store), (true, 4, 0));
        store
            .with_connection(|db| {
                db.execute("CREATE TABLE audit_schema_fixture(id INTEGER)", [])?;
                Ok(())
            })
            .unwrap();
        store.checkpoint_audit(secrets.as_ref()).unwrap();
        assert_eq!(stats(&store), (true, 4, 0));
        other
            .execute(
                "UPDATE audit_log SET redacted_payload_json='false' WHERE sequence=1",
                [],
            )
            .unwrap();
        assert!(store.checkpoint_audit(secrets.as_ref()).is_err());
    }

    #[test]
    fn changes_during_anchor_writes_are_checked_and_new_local_appends_remain_a_tail() {
        for change in [
            "local_prefix",
            "external_prefix",
            "local_append",
            "external_append",
            "database_identity",
        ] {
            let (directory, store, secrets) = fixture();
            seed(&store, 3);
            store.checkpoint_audit(secrets.as_ref()).unwrap();
            store.append_audit(None, None, "next", &true).unwrap();
            let concurrent = store.clone();
            let path = directory.path().join("audit.db");
            *secrets.during_set.lock().unwrap() = Some(Box::new(move || {
                if change == "local_append" {
                    concurrent
                        .append_audit(None, None, "during_anchor", &true)
                        .unwrap();
                } else if change == "database_identity" {
                    concurrent
                        .save_setting("audit.database_id", &Uuid::new_v4())
                        .unwrap();
                } else if change.starts_with("external") {
                    let other =
                        crate::vault::open_encrypted(&path, &SecretBytes::new(vec![19; 32]))
                            .unwrap();
                    if change == "external_append" {
                        crate::storage::write_audit(&other, None, None, "during_anchor", &true)
                            .unwrap();
                    } else {
                        other.execute("UPDATE audit_log SET redacted_payload_json='false' WHERE sequence=1", []).unwrap();
                    }
                } else {
                    concurrent.with_connection(|db| { db.execute("UPDATE audit_log SET redacted_payload_json='false' WHERE sequence=1", [])?; Ok(()) }).unwrap();
                }
            }));
            let result = store.checkpoint_audit(secrets.as_ref());
            assert_eq!(result.is_ok(), change == "local_append", "{change}");
            if change.ends_with("append") {
                store.checkpoint_audit(secrets.as_ref()).unwrap();
                assert_eq!(anchor(&store, secrets.as_ref()).sequence, 5);
                assert_eq!(stats(&store).0, change == "external_append");
                assert_eq!(
                    stats(&store).1,
                    if change == "external_append" { 5 } else { 1 }
                );
            } else {
                assert!(
                    store.checkpoint_audit(secrets.as_ref()).is_err(),
                    "{change}"
                );
            }
        }
    }

    #[test]
    fn failed_or_ambiguous_anchor_writes_and_marker_failure_do_not_cache_success() {
        for failure in [1, 2] {
            let (_directory, store, secrets) = fixture();
            seed(&store, 3);
            store.checkpoint_audit(secrets.as_ref()).unwrap();
            store.append_audit(None, None, "next", &true).unwrap();
            secrets.fail_set.store(failure, Ordering::Relaxed);
            assert!(store.checkpoint_audit(secrets.as_ref()).is_err());
            assert!(store.audit_lock.lock().unwrap().verified.is_none());
            assert_eq!(
                anchor(&store, secrets.as_ref()).sequence,
                if failure == 1 { 3 } else { 4 }
            );
            store.checkpoint_audit(secrets.as_ref()).unwrap();
            assert_eq!(stats(&store), (true, 4, 0));
        }
        let (_directory, store, secrets) = fixture();
        seed(&store, 3);
        store.with_connection(|db| { db.execute_batch("CREATE TEMP TRIGGER reject_expected BEFORE INSERT ON settings WHEN NEW.key='audit.checkpoint_expected' BEGIN SELECT RAISE(ABORT,'fixture marker failure'); END;")?; Ok(()) }).unwrap();
        assert!(store.checkpoint_audit(secrets.as_ref()).is_err());
        assert!(store.audit_lock.lock().unwrap().verified.is_none());
        assert_eq!(anchor(&store, secrets.as_ref()).sequence, 3);
        store
            .with_connection(|db| {
                db.execute_batch("DROP TRIGGER reject_expected")?;
                Ok(())
            })
            .unwrap();
        store.checkpoint_audit(secrets.as_ref()).unwrap();
        assert_eq!(stats(&store), (true, 3, 0));
        assert_eq!(secrets.sets.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn historical_scrub_is_bounded_and_detects_a_change_hidden_from_notifications() {
        let (_directory, store, secrets) = fixture();
        seed(&store, 512);
        store.checkpoint_audit(secrets.as_ref()).unwrap();
        store
            .with_connection(|db| {
                // Test-only fault injection: bypass the installed notification to
                // exercise the independent scrub rather than its invalidation path.
                db.preupdate_hook(None::<fn(rusqlite::hooks::Action, &str, &str, &PreUpdateCase)>);
                db.execute(
                    "UPDATE audit_log SET redacted_payload_json='false' WHERE sequence=500",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        for _ in 0..3 {
            store.checkpoint_audit(secrets.as_ref()).unwrap();
            assert_eq!(stats(&store), (false, 0, 128));
        }
        assert!(store.checkpoint_audit(secrets.as_ref()).is_err());
        assert!(store.audit_lock.lock().unwrap().verified.is_none());
    }

    #[test]
    fn cloned_stores_serialize_checkpoint_writers_and_preserve_concurrent_appends() {
        let (_directory, store, secrets) = fixture();
        store.checkpoint_audit(secrets.as_ref()).unwrap();
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let store = store.clone();
                let secrets = secrets.clone();
                scope.spawn(move || {
                    for index in 0..25 {
                        store
                            .append_audit(None, None, "concurrent", &(worker, index))
                            .unwrap();
                        store.checkpoint_audit(secrets.as_ref()).unwrap();
                    }
                });
            }
        });
        store.checkpoint_audit(secrets.as_ref()).unwrap();
        assert_eq!(anchor(&store, secrets.as_ref()).sequence, 100);
        assert!(!stats(&store).0);
    }

    #[test]
    fn unlock_rebinds_the_observer_and_revision_saturation_never_reuses_a_cursor() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalStore::deferred(&directory.path().join("unlock.db")).unwrap();
        let secrets = CountingSecrets::default();
        secrets
            .inner
            .set("database-v2", &SecretBytes::new(vec![19; 32]))
            .unwrap();
        assert!(store.unlock(&secrets).unwrap());
        seed(&store, 512);
        store.checkpoint_audit(&secrets).unwrap();
        assert!(stats(&store).0);
        store.checkpoint_audit(&secrets).unwrap();
        assert!(!stats(&store).0);
        let revision = store.audit_watch.revision.load(Ordering::Acquire);
        store
            .with_connection(|db| {
                db.execute(
                    "UPDATE audit_log SET redacted_payload_json='false' WHERE sequence=500",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(store.audit_watch.revision.load(Ordering::Acquire) > revision);
        assert!(store.checkpoint_audit(&secrets).is_err());
        store.with_connection(|db| {
            db.execute("UPDATE audit_log SET redacted_payload_json='{\"index\":499}' WHERE sequence=500", [])?;
            Ok(())
        }).unwrap();
        store
            .audit_watch
            .revision
            .store(u64::MAX, Ordering::Release);
        for _ in 0..2 {
            store.checkpoint_audit(&secrets).unwrap();
            assert!(stats(&store).0);
        }
        store
            .with_connection(|db| {
                db.execute(
                    "UPDATE audit_log SET redacted_payload_json='false' WHERE sequence=1",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(store.checkpoint_audit(&secrets).is_err());
    }

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
