//! Prepared intent is durable before authority is consumed. Journal state never
//! permits replay of an ambiguous dispatch, including after a process crash.
use crate::capability::CapabilityGrant;
use crate::contracts::{POLICY_VERSION, PreparedAction, Verdict, VerificationRecord};
use crate::storage::LocalStore;
use crate::{CoreError, CoreResult};
use rusqlite::params;
use uuid::Uuid;

fn refused() -> CoreError {
    CoreError::CapabilityRejected(
        "Action journal identity or transition is invalid; reconcile the run before retrying"
            .into(),
    )
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
        store.journal_dispatch(&prepared, None).unwrap();
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

impl LocalStore {
    pub(crate) fn migrate_journal(&self) -> CoreResult<()> {
        self.with_connection(|db| {
            db.execute_batch("CREATE TABLE IF NOT EXISTS action_journal(action_id TEXT PRIMARY KEY,run_id TEXT NOT NULL,action_digest TEXT NOT NULL,prepared_json TEXT NOT NULL,state TEXT NOT NULL,grant_json TEXT,verification_json TEXT,updated_at TEXT NOT NULL);")?;
            Ok(())
        })
    }
    pub fn journal_prepared(&self, prepared: &PreparedAction) -> CoreResult<()> {
        let proposal = &prepared.intent.proposal;
        if prepared.policy_version != POLICY_VERSION
            || prepared.action_digest != crate::policy::approval_digest(proposal)?
        {
            return Err(refused());
        }
        self.with_connection(|db| {
            let changes = db.execute("INSERT INTO action_journal VALUES(?1,?2,?3,?4,'prepared',NULL,NULL,?5) ON CONFLICT(action_id) DO UPDATE SET action_digest=excluded.action_digest,prepared_json=excluded.prepared_json,state='prepared',grant_json=NULL,verification_json=NULL,updated_at=excluded.updated_at WHERE action_journal.run_id=excluded.run_id AND action_journal.state IN ('prepared','failed','cancelled')",
                params![proposal.id.to_string(),proposal.task_id.to_string(),prepared.action_digest,serde_json::to_string(prepared)?,chrono::Utc::now().to_rfc3339()])?;
            if changes != 1 { return Err(refused().into()); }
            Ok(())
        })
    }
    pub fn journal_dispatch(
        &self,
        prepared: &PreparedAction,
        grant: Option<&CapabilityGrant>,
    ) -> CoreResult<()> {
        let proposal = &prepared.intent.proposal;
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
        self.with_connection(|db| {
            let changes = db.execute("UPDATE action_journal SET state='dispatched',grant_json=?4,updated_at=?5 WHERE action_id=?1 AND run_id=?2 AND action_digest=?3 AND state='prepared'",params![proposal.id.to_string(),proposal.task_id.to_string(),prepared.action_digest,grant.map(serde_json::to_string).transpose()?,chrono::Utc::now().to_rfc3339()])?;
            if changes != 1 { return Err(refused().into()); }
            Ok(())
        })
    }
    pub fn journal_verification(&self, record: &VerificationRecord) -> CoreResult<()> {
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
    pub fn journal_interrupted(&self, run: Uuid, action: Uuid, dispatched: bool) -> CoreResult<()> {
        self.with_connection(|db| {
            db.execute("UPDATE action_journal SET state=?3,updated_at=?4 WHERE run_id=?1 AND action_id=?2 AND state IN ('prepared','dispatched')",params![run.to_string(),action.to_string(),if dispatched {"uncertain"} else {"failed"},chrono::Utc::now().to_rfc3339()])?;
            Ok(())
        })
    }
}
