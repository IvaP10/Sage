//! Prepared intent is durable before authority is consumed. Journal state never
//! permits replay of an ambiguous dispatch, including after a process crash.
use crate::capability::CapabilityGrant;
use crate::contracts::{POLICY_VERSION, PreparedAction};
#[cfg(test)]
use crate::contracts::{Verdict, VerificationRecord};
use crate::storage::LocalStore;
use crate::{CoreError, CoreResult};
use rusqlite::params;
use sha2::{Digest, Sha256};
use uuid::Uuid;

fn refused() -> CoreError {
    CoreError::CapabilityRejected(
        "Action journal identity or transition is invalid; reconcile the run before retrying"
            .into(),
    )
}

impl LocalStore {
    pub(crate) fn migrate_journal(&self) -> CoreResult<()> {
        self.with_connection(|db| {
            db.execute_batch("CREATE TABLE IF NOT EXISTS action_journal(action_id TEXT PRIMARY KEY,run_id TEXT NOT NULL,action_digest TEXT NOT NULL,prepared_json TEXT NOT NULL,state TEXT NOT NULL,grant_json TEXT,verification_json TEXT,updated_at TEXT NOT NULL);")?;
            crate::learning::migrate(db)?;
            db.execute_batch("CREATE TABLE IF NOT EXISTS worker_receipts(request_id TEXT NOT NULL,kind TEXT NOT NULL,session_id TEXT NOT NULL,run_id TEXT NOT NULL,action_id TEXT NOT NULL,action_digest TEXT NOT NULL,artifact_id TEXT REFERENCES private_artifacts(id) ON DELETE SET NULL,received_at TEXT NOT NULL,PRIMARY KEY(request_id,kind));")?;
            let old_foreign_key: bool = db.query_row("SELECT EXISTS(SELECT 1 FROM pragma_foreign_key_list('worker_receipts') WHERE on_delete='CASCADE')", [], |row| row.get(0))?;
            if old_foreign_key {
                // Removing a private body must not remove opaque effect history
                // or an obligation to reconcile its task after a crash.
                let tx = db.transaction()?;
                tx.execute_batch("CREATE TABLE worker_receipts_retained(request_id TEXT NOT NULL,kind TEXT NOT NULL,session_id TEXT NOT NULL,run_id TEXT NOT NULL,action_id TEXT NOT NULL,action_digest TEXT NOT NULL,artifact_id TEXT REFERENCES private_artifacts(id) ON DELETE SET NULL,received_at TEXT NOT NULL,PRIMARY KEY(request_id,kind)); INSERT INTO worker_receipts_retained SELECT * FROM worker_receipts; DROP TABLE worker_receipts; ALTER TABLE worker_receipts_retained RENAME TO worker_receipts;")?;
                tx.commit()?;
            }
            db.execute_batch("CREATE INDEX IF NOT EXISTS worker_receipts_action ON worker_receipts(run_id,action_id); CREATE TABLE IF NOT EXISTS worker_receipt_projections(request_id TEXT NOT NULL,kind TEXT NOT NULL,event_id TEXT NOT NULL,projected_at TEXT NOT NULL,PRIMARY KEY(request_id,kind),FOREIGN KEY(request_id,kind) REFERENCES worker_receipts(request_id,kind) ON DELETE CASCADE);")?;
            Ok(())
        })
    }
    #[cfg(test)]
    pub(crate) fn journal_prepared(&self, prepared: &PreparedAction) -> CoreResult<()> {
        self.with_connection(|db| Ok(write_prepared(db, prepared)?))
    }
    #[cfg(test)]
    pub(crate) fn journal_dispatch(
        &self,
        prepared: &PreparedAction,
        grant: Option<&CapabilityGrant>,
    ) -> CoreResult<()> {
        self.with_connection(|db| Ok(write_dispatch(db, prepared, grant)?))
    }
    #[cfg(test)]
    pub(crate) fn journal_verification(&self, record: &VerificationRecord) -> CoreResult<()> {
        let state = match record.verdict {
            Verdict::Confirmed => "confirmed",
            Verdict::Failed => "failed",
            Verdict::Cancelled => "cancelled",
            Verdict::Uncertain => "uncertain",
        };
        self.with_connection(|db| {
            let changes = db.execute("UPDATE action_journal SET state=?4,verification_json=?5,updated_at=?6 WHERE action_id=?1 AND run_id=?2 AND action_digest=?3 AND state IN ('dispatched','uncertain')",params![record.action_id.to_string(),record.run_id.to_string(),record.action_digest,state,serde_json::to_string(record)?,chrono::Utc::now().to_rfc3339()])?;
            if changes != 1 { return Err(refused().into()); }
            Ok(())
        })
    }
    #[cfg(test)]
    pub(crate) fn journal_interrupted(
        &self,
        run: Uuid,
        action: Uuid,
        dispatched: bool,
    ) -> CoreResult<()> {
        self.with_connection(|db| {
            db.execute("UPDATE action_journal SET state=?3,updated_at=?4 WHERE run_id=?1 AND action_id=?2 AND state IN ('prepared','dispatched')",params![run.to_string(),action.to_string(),if dispatched {"uncertain"} else {"failed"},chrono::Utc::now().to_rfc3339()])?;
            Ok(())
        })
    }

    /// Preserve the worker's bounded reply before its waiting future can be
    /// released or cancelled. Receipt existence is not postcondition success.
    pub(crate) fn record_worker_receipt(
        &self,
        request_id: &str,
        session: &str,
        binding: &crate::execution::bridge::EffectBinding,
        kind: &str,
        bytes: &[u8],
    ) -> CoreResult<()> {
        if bytes.len() > 512 * 1024 || !matches!(kind, "response" | "cancel_ack" | "never_sent") {
            return Err(CoreError::Protocol("Invalid worker receipt".into()));
        }
        self.with_connection(|db| {
            let transaction = db.transaction()?;
            let matches: bool = transaction.query_row("SELECT EXISTS(SELECT 1 FROM action_journal WHERE run_id=?1 AND action_id=?2 AND action_digest=?3)",
                params![binding.task_id.to_string(),binding.action_id.to_string(),binding.action_digest], |row| row.get(0))?;
            if !matches { return Err(refused().into()); }
            let retired: bool = transaction.query_row("SELECT EXISTS(SELECT 1 FROM retired_task_data WHERE task_id=?1)",
                [binding.task_id.to_string()], |row| row.get(0))?;
            // Keep opaque effect accounting after forgetting, never its body.
            let artifact_id = if bytes.is_empty() || retired { None } else { Some(Uuid::new_v4()) };
            if let Some(id) = artifact_id {
                transaction.execute("INSERT INTO private_artifacts VALUES(?1,?2,?3,?4,?5)", params![id.to_string(),binding.task_id.to_string(),bytes,
                    format!("{:x}", Sha256::digest(bytes)),(chrono::Utc::now()+chrono::Duration::hours(24)).to_rfc3339()])?;
            }
            transaction.execute("INSERT INTO worker_receipts VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
                params![request_id,kind,session,binding.task_id.to_string(),binding.action_id.to_string(),binding.action_digest,
                    artifact_id.map(|id|id.to_string()),chrono::Utc::now().to_rfc3339()])?;
            if kind == "never_sent" {
                transaction.execute("UPDATE action_journal SET state='cancelled',updated_at=?3 WHERE run_id=?1 AND action_id=?2 AND state IN ('prepared','dispatched','uncertain')",
                    params![binding.task_id.to_string(),binding.action_id.to_string(),chrono::Utc::now().to_rfc3339()])?;
            }
            transaction.commit()?;
            Ok(())
        })
    }
}

pub(crate) fn write_prepared(
    db: &rusqlite::Connection,
    prepared: &PreparedAction,
) -> CoreResult<()> {
    let proposal = &prepared.intent.proposal;
    if prepared.policy_version != POLICY_VERSION
        || prepared.action_digest != crate::policy::approval_digest(proposal)?
    {
        return Err(refused());
    }
    let unsettled: bool = db.query_row("SELECT EXISTS(SELECT 1 FROM worker_receipts r LEFT JOIN worker_receipt_projections p USING(request_id,kind) WHERE r.run_id=?1 AND r.action_id=?2 AND p.request_id IS NULL)",
        params![proposal.task_id.to_string(),proposal.id.to_string()], |row| row.get(0))?;
    if unsettled {
        return Err(CoreError::PermissionRequired(
            "Reconcile the saved worker receipts before preparing this action again".into(),
        ));
    }
    let changed = db.execute("INSERT INTO action_journal VALUES(?1,?2,?3,?4,'prepared',NULL,NULL,?5) ON CONFLICT(action_id) DO UPDATE SET action_digest=excluded.action_digest,prepared_json=excluded.prepared_json,state='prepared',grant_json=NULL,verification_json=NULL,updated_at=excluded.updated_at WHERE action_journal.run_id=excluded.run_id AND action_journal.state IN ('prepared','failed','cancelled')",
        params![proposal.id.to_string(),proposal.task_id.to_string(),prepared.action_digest,serde_json::to_string(prepared)?,chrono::Utc::now().to_rfc3339()])?;
    if changed != 1 {
        return Err(refused());
    }
    Ok(())
}

pub(crate) fn write_dispatch(
    db: &rusqlite::Connection,
    prepared: &PreparedAction,
    grant: Option<&CapabilityGrant>,
) -> CoreResult<()> {
    let proposal = &prepared.intent.proposal;
    if prepared.policy_version != POLICY_VERSION
        || prepared.action_digest != crate::policy::approval_digest(proposal)?
    {
        return Err(refused());
    }
    if let Some(grant) = grant {
        if grant.task_id != proposal.task_id
            || grant.action_id != proposal.id
            || grant.action_digest != prepared.action_digest
            || grant.policy_version != prepared.policy_version
            || grant.revoked
            || grant.expires_at <= chrono::Utc::now()
        {
            return Err(refused());
        }
    } else if !matches!(proposal.action, crate::domain::Action::AskUser { .. }) {
        return Err(refused());
    }
    let changed = db.execute("UPDATE action_journal SET state='dispatched',grant_json=?4,updated_at=?5 WHERE action_id=?1 AND run_id=?2 AND action_digest=?3 AND state='prepared'",
        params![proposal.id.to_string(),proposal.task_id.to_string(),prepared.action_digest,grant.map(serde_json::to_string).transpose()?,chrono::Utc::now().to_rfc3339()])?;
    if changed != 1 {
        return Err(refused());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Action, ActionProposal, ExpectedOutcome, Provenance};
    #[test]
    fn journal_rejects_replayed_dispatch_and_repreparation_after_uncertain_effects() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalStore::open_encrypted(
            &directory.path().join("test.db"),
            &crate::secrets::SecretBytes::new(vec![14; 32]),
        )
        .unwrap();
        let proposal = ActionProposal {
            id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            action: Action::AskUser {
                question: "Choose the prepared result".into(),
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: "user".into(),
            provenance: Provenance::model(vec![]),
            metadata: Default::default(),
        };
        let prepared = PreparedAction::new(&proposal, Default::default()).unwrap();
        store.journal_prepared(&prepared).unwrap();
        let mut tampered = prepared.clone();
        tampered.intent.proposal.action = Action::AskUser {
            question: "A different unprepared question".into(),
        };
        assert!(store.journal_dispatch(&tampered, None).is_err());
        store.journal_dispatch(&prepared, None).unwrap();
        let binding = crate::execution::bridge::EffectBinding {
            task_id: proposal.task_id,
            action_id: proposal.id,
            action_digest: prepared.action_digest.clone(),
        };
        let request_id = Uuid::new_v4().to_string();
        store
            .record_worker_receipt(
                &request_id,
                "fixture-session",
                &binding,
                "response",
                b"retained worker reply",
            )
            .unwrap();
        let mut forged = binding.clone();
        forged.action_digest = "another prepared effect".into();
        assert!(
            store
                .record_worker_receipt(
                    &Uuid::new_v4().to_string(),
                    "fixture-session",
                    &forged,
                    "response",
                    b"unrelated reply"
                )
                .is_err()
        );
        store.with_connection(|db| {
            let body: Vec<u8> = db.query_row("SELECT content FROM private_artifacts JOIN worker_receipts ON private_artifacts.id=worker_receipts.artifact_id WHERE request_id=?1", [&request_id], |row| row.get(0))?;
            assert_eq!(body, b"retained worker reply");
            let count: i64 = db.query_row("SELECT COUNT(*) FROM private_artifacts", [], |row| row.get(0))?;
            assert_eq!(count, 1, "A rejected binding must not leave an orphan artifact");
            Ok(())
        }).unwrap();
        store
            .with_connection(|db| {
                db.execute(
                    "INSERT INTO retired_task_data VALUES(?1,CURRENT_TIMESTAMP)",
                    [proposal.task_id.to_string()],
                )?;
                Ok(())
            })
            .unwrap();
        let late_id = Uuid::new_v4().to_string();
        store
            .record_worker_receipt(
                &late_id,
                "fixture-session",
                &binding,
                "response",
                b"forgotten late reply",
            )
            .unwrap();
        store.with_connection(|db| {
            let count: i64 = db.query_row("SELECT COUNT(*) FROM private_artifacts", [], |row| row.get(0))?;
            assert_eq!(count, 0, "A late reply must not restore forgotten content");
            let receipt: bool = db.query_row("SELECT EXISTS(SELECT 1 FROM worker_receipts WHERE request_id=?1 AND artifact_id IS NULL)", [&late_id], |row| row.get(0))?;
            assert!(receipt, "Opaque effect accounting remains available");
            Ok(())
        }).unwrap();
        assert!(store.journal_dispatch(&prepared, None).is_err());
        store
            .journal_interrupted(proposal.task_id, proposal.id, true)
            .unwrap();
        assert!(store.journal_prepared(&prepared).is_err());
        let mut changed = prepared.clone();
        changed.intent.proposal.task_id = Uuid::new_v4();
        assert!(store.journal_prepared(&changed).is_err());
        let record = VerificationRecord {
            run_id: proposal.task_id,
            action_id: proposal.id,
            target: "user".into(),
            action_digest: prepared.action_digest.clone(),
            expected: ExpectedOutcome::UserAnswered,
            observed_at: chrono::Utc::now(),
            verdict: Verdict::Confirmed,
            evidence: vec![crate::observation::Evidence::UserAnswer { received: true }],
        };
        store.journal_verification(&record).unwrap();
        assert!(store.journal_dispatch(&prepared, None).is_err());
    }
}
