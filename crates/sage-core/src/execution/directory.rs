//! Bounded, nonrecursive directory pages from pinned native handles. Cursors
//! select a range in a verified listing; they never grant filesystem access.
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{files::PinnedPath, io::bounded_read};
use crate::{CoreError, CoreResult, contracts::FileIdentity};

pub const MAX_PAGE_ENTRIES: u32 = 64;
pub const MAX_DIRECTORY_ENTRIES: usize = 16_384;
pub const MAX_PAGE_BYTES: usize = 12 * 1024;
const MAX_NAME_BYTES: usize = 4 * 1024 * 1024;
const INSPECTION_TIME: Duration = Duration::from_secs(3);
pub(crate) async fn inspect(
    pinned: Option<PinnedPath>,
    path: std::path::PathBuf,
    page_size: u32,
    cursor: Option<String>,
    expires_at: Option<chrono::DateTime<chrono::Utc>>,
) -> CoreResult<DirectoryPage> {
    bounded_read(move |cancelled| {
        if expires_at.is_some_and(|expiry| expiry <= chrono::Utc::now()) {
            return Err(CoreError::CapabilityRejected(
                "Directory read grant expired while waiting".into(),
            ));
        }
        let pinned = match pinned {
            Some(pinned) => pinned,
            None => PinnedPath::open_directory(&path)?,
        };
        read_page(&pinned, &path, page_size, cursor.as_deref(), cancelled)
    })
    .await
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NameEncoding {
    Utf8,
    NativeBase64,
    Redacted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    File,
    Directory,
    Symlink,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectoryEntry {
    pub name: String,
    pub name_encoding: NameEncoding,
    pub kind: EntryKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectoryPage {
    pub version: u32,
    pub path: String,
    pub directory_identity: FileIdentity,
    pub snapshot_sha256: String,
    pub total_entries: u32,
    pub offset: u32,
    pub entries: Vec<DirectoryEntry>,
    pub next_cursor: Option<String>,
    /// Only true when this page contains the entire listing.
    pub complete: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cursor {
    version: u32,
    target_sha256: String,
    snapshot_sha256: String,
    offset: u32,
}

impl DirectoryPage {
    pub fn digest(&self) -> CoreResult<String> {
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }

    pub fn summary(&self) -> String {
        if self.total_entries == 0 {
            format!("Folder {} is empty", self.path)
        } else {
            format!(
                "Listed entries {}–{} of {} in {}{}",
                self.offset + 1,
                self.offset as usize + self.entries.len(),
                self.total_entries,
                self.path,
                if self.next_cursor.is_some() {
                    "; more entries are available"
                } else {
                    ""
                }
            )
        }
    }
}

fn read_page(
    pinned: &PinnedPath,
    path: &Path,
    page_size: u32,
    cursor: Option<&str>,
    cancelled: &AtomicBool,
) -> CoreResult<DirectoryPage> {
    if !(1..=MAX_PAGE_ENTRIES).contains(&page_size) {
        return Err(CoreError::InvalidAction(
            "Directory page size must be between 1 and 64".into(),
        ));
    }
    let path = path
        .to_str()
        .filter(|p| p.len() <= 4096)
        .ok_or_else(|| CoreError::InvalidAction("Directory path must be bounded Unicode".into()))?;
    let target_sha256 = format!("{:x}", Sha256::digest(path.as_bytes()));
    let cursor = cursor
        .map(|token| {
            if token.is_empty() || token.len() > 4096 {
                return Err(stale_cursor());
            }
            let bytes = URL_SAFE_NO_PAD.decode(token).map_err(|_| stale_cursor())?;
            let cursor: Cursor = serde_json::from_slice(&bytes).map_err(|_| stale_cursor())?;
            if cursor.version != 1 || cursor.target_sha256 != target_sha256 || cursor.offset == 0 {
                return Err(stale_cursor());
            }
            Ok(cursor)
        })
        .transpose()?;
    let started = Instant::now();
    let check = || {
        if cancelled.load(Ordering::Acquire) {
            return Err(CoreError::Cancelled);
        }
        if started.elapsed() > INSPECTION_TIME {
            return Err(CoreError::Timeout(
                "Directory inspection exceeded its time budget".into(),
            ));
        }
        Ok(())
    };
    check()?;
    let directory = pinned.directory_handle()?;
    let directory_identity = pinned.validate_directory_handle(&directory)?;
    let mut entries = Vec::new();
    let mut name_bytes = 0_usize;
    let mut iterator = directory.entries()?;
    loop {
        check()?;
        let Some(entry) = iterator.next() else {
            break;
        };
        if entries.len() >= MAX_DIRECTORY_ENTRIES {
            return Err(CoreError::ExecutionFailed("Directory exceeds this inspection's entry or time budget. Select a smaller folder; no complete listing was returned".into()));
        }
        let entry = entry?;
        let name = entry.file_name();
        let raw = name.as_encoded_bytes().to_vec();
        name_bytes = name_bytes.saturating_add(raw.len());
        if name_bytes > MAX_NAME_BYTES {
            return Err(CoreError::ExecutionFailed(
                "Directory names exceed the inspection's memory budget".into(),
            ));
        }
        let kind = entry.file_type()?;
        let kind = if kind.is_symlink() {
            EntryKind::Symlink
        } else if kind.is_dir() {
            EntryKind::Directory
        } else if kind.is_file() {
            EntryKind::File
        } else {
            EntryKind::Other
        };
        let (name, name_encoding) = display_name(&name);
        entries.push((
            raw,
            DirectoryEntry {
                name,
                name_encoding,
                kind,
            },
        ));
    }
    pinned.validate_directory_handle(&directory)?;
    check()?;
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(CoreError::VerificationFailed(
            "Directory enumeration changed while reading".into(),
        ));
    }
    let mut hash = Sha256::new();
    hash.update(b"sage-directory-v1\0");
    hash.update(serde_json::to_vec(&directory_identity)?);
    for (raw, item) in &entries {
        hash.update((raw.len() as u64).to_le_bytes());
        hash.update(raw);
        hash.update(serde_json::to_vec(&item.kind)?);
    }
    let snapshot_sha256 = format!("{:x}", hash.finalize());
    check()?;
    let offset = if let Some(cursor) = cursor {
        if cursor.snapshot_sha256 != snapshot_sha256 || cursor.offset as usize >= entries.len() {
            return Err(stale_cursor());
        }
        cursor.offset
    } else {
        0
    };
    let total_entries = entries.len() as u32;
    let mut page = DirectoryPage {
        version: 1,
        path: path.into(),
        directory_identity,
        snapshot_sha256,
        total_entries,
        offset,
        entries: entries
            .into_iter()
            .skip(offset as usize)
            .take(page_size as usize)
            .map(|(_, entry)| entry)
            .collect(),
        next_cursor: None,
        complete: false,
    };
    loop {
        let end = offset + page.entries.len() as u32;
        page.complete = offset == 0 && end == total_entries;
        page.next_cursor = if end < total_entries {
            Some(URL_SAFE_NO_PAD.encode(serde_json::to_vec(&Cursor {
                version: 1,
                target_sha256: target_sha256.clone(),
                snapshot_sha256: page.snapshot_sha256.clone(),
                offset: end,
            })?))
        } else {
            None
        };
        if serde_json::to_vec(&page)?.len() <= MAX_PAGE_BYTES {
            break;
        }
        if page.entries.len() <= 1 {
            return Err(CoreError::ExecutionFailed(
                "A directory entry cannot fit the bounded result page".into(),
            ));
        }
        page.entries.pop();
    }
    check()?;
    Ok(page)
}

fn display_name(name: &std::ffi::OsStr) -> (String, NameEncoding) {
    // Check the text-like spans before encoding native bytes. Otherwise a
    // single invalid Unicode byte could carry a recognized secret through
    // persistence disguised as a base64 name.
    let readable = name.to_string_lossy();
    if crate::redaction::redact_for_persistence(&readable) != readable {
        ("[redacted filename]".into(), NameEncoding::Redacted)
    } else {
        match name.to_str() {
            Some(text) => (text.to_owned(), NameEncoding::Utf8),
            None => (
                URL_SAFE_NO_PAD.encode(name.as_encoded_bytes()),
                NameEncoding::NativeBase64,
            ),
        }
    }
}

fn stale_cursor() -> CoreError {
    CoreError::VerificationFailed("Directory cursor is invalid or its entries changed. Restart listing this folder with cursor=null".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::io::MAX_READERS;
    use crate::{
        domain::{
            Action, ActionProposal, ActionState, ActionStatus, ExpectedOutcome, Provenance, Task,
        },
        execution::{ExecutionReceipt, files::FileBroker},
        observation::{DeterministicObserver, Observer},
    };
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    fn page(path: &Path, size: u32, cursor: Option<&str>) -> CoreResult<DirectoryPage> {
        read_page(
            &PinnedPath::open_directory(path)?,
            path,
            size,
            cursor,
            &AtomicBool::new(false),
        )
    }

    #[test]
    fn pages_are_complete_ordered_bounded_and_cursor_changes_cannot_skip_silently() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let empty = page(&root, 64, None).unwrap();
        assert!(empty.complete);
        assert!(empty.entries.is_empty() && empty.next_cursor.is_none());
        let names = (0..137)
            .map(|n| format!("entry-{n:03}-{}", "x".repeat(220)))
            .collect::<BTreeSet<_>>();
        for name in &names {
            std::fs::write(root.join(name), b"private file body").unwrap();
        }
        let first = page(&root, 64, None).unwrap();
        assert!(!first.complete);
        assert!(
            first.entries.len() < 64,
            "byte budget must reduce the requested count"
        );
        let mut seen = BTreeSet::new();
        let mut cursor = None;
        let mut offset = 0;
        loop {
            let result = page(&root, 64, cursor.as_deref()).unwrap();
            assert_eq!(result.offset, offset);
            assert_eq!(result.total_entries, 137);
            assert_eq!(result.snapshot_sha256, first.snapshot_sha256);
            assert!(serde_json::to_vec(&result).unwrap().len() <= MAX_PAGE_BYTES);
            for entry in &result.entries {
                assert_eq!(entry.kind, EntryKind::File);
                assert!(seen.insert(entry.name.clone()), "duplicate page entry");
            }
            offset += result.entries.len() as u32;
            cursor = result.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(seen, names);
        assert_eq!(offset, 137);
        let cursor = first.next_cursor.unwrap();
        let other = tempfile::tempdir().unwrap();
        assert!(page(&other.path().canonicalize().unwrap(), 64, Some(&cursor)).is_err());
        for invalid in ["", "not a cursor", &URL_SAFE_NO_PAD.encode(b"{}")] {
            assert!(page(&root, 64, Some(invalid)).is_err());
        }
        assert!(page(&root, 0, None).is_err());
        assert!(page(&root, 65, None).is_err());
        std::fs::write(root.join("new-entry"), b"new").unwrap();
        assert!(
            page(&root, 64, Some(&cursor)).is_err(),
            "an old page cursor must not silently continue a changed directory"
        );
        assert_eq!(page(&root, 64, None).unwrap().total_entries, 138);
    }

    #[cfg(unix)]
    #[test]
    fn links_are_named_without_traversal_and_a_replaced_directory_is_rejected() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        std::fs::write(outside.path().join("hidden.txt"), b"outside-content-canary").unwrap();
        std::fs::create_dir(root.join("folder")).unwrap();
        symlink(outside.path(), root.join("outside-link")).unwrap();
        symlink(outside.path().join("missing"), root.join("broken-link")).unwrap();
        let result = page(&root, 64, None).unwrap();
        assert_eq!(result.total_entries, 3);
        assert_eq!(
            result
                .entries
                .iter()
                .filter(|e| e.kind == EntryKind::Symlink)
                .count(),
            2
        );
        assert!(
            !serde_json::to_string(&result)
                .unwrap()
                .contains("outside-content-canary")
        );
        assert!(PinnedPath::open_directory(&root.join("outside-link")).is_err());
        let pinned = PinnedPath::open_directory(&root.join("folder")).unwrap();
        std::fs::rename(root.join("folder"), root.join("previous")).unwrap();
        std::fs::create_dir(root.join("folder")).unwrap();
        assert!(
            read_page(
                &pinned,
                &root.join("folder"),
                64,
                None,
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }

    #[test]
    fn filenames_preserve_unicode_and_redact_recognized_secrets() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let unicode = "résumé 🦀.txt";
        let secret = format!("{}{}.txt", "sk-", "123456789012345");
        for name in [unicode, &secret] {
            std::fs::File::create(root.join(name)).unwrap();
        }
        let result = page(&root, 64, None).unwrap();
        assert!(result.complete);
        assert_eq!(result.entries.len(), 2);
        assert!(
            result.entries.iter().any(|entry| {
                entry.name == unicode && entry.name_encoding == NameEncoding::Utf8
            })
        );
        assert_eq!(
            result
                .entries
                .iter()
                .filter(|entry| entry.name_encoding == NameEncoding::Redacted)
                .count(),
            1
        );
        let serialized = serde_json::to_string(&result).unwrap();
        assert!(!serialized.contains(&secret));
    }

    #[cfg(unix)]
    #[test]
    fn native_name_encoding_preserves_bytes_but_cannot_hide_recognized_secrets() {
        use std::os::unix::ffi::OsStrExt;
        let native = b"notes_\xff.txt";
        let secret = format!("{}{}.txt", "sk-", "123456789012345");
        let mut native_secret = secret.as_bytes().to_vec();
        native_secret.push(0xff);
        let (encoded, encoding) = display_name(std::ffi::OsStr::from_bytes(native));
        assert_eq!(encoding, NameEncoding::NativeBase64);
        assert_eq!(URL_SAFE_NO_PAD.decode(&encoded).unwrap(), native);
        let (redacted, encoding) = display_name(std::ffi::OsStr::from_bytes(&native_secret));
        assert_eq!(encoding, NameEncoding::Redacted);
        assert_eq!(redacted, "[redacted filename]");
        // APFS rejects these names. Linux can also exercise the actual native
        // enumeration path, in addition to the codec boundary above.
        #[cfg(target_os = "linux")]
        {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().canonicalize().unwrap();
            for name in [native.as_slice(), native_secret.as_slice()] {
                std::fs::File::create(root.join(std::ffi::OsStr::from_bytes(name))).unwrap();
            }
            let result = page(&root, 64, None).unwrap();
            assert_eq!(result.total_entries, 2);
            assert!(result.entries.iter().any(|entry| entry.name == encoded));
            assert!(result.entries.iter().any(|entry| entry.name == redacted));
        }
    }

    #[test]
    fn entry_admission_refuses_oversize_directories_instead_of_claiming_a_partial_list_is_complete()
    {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        for n in 0..=MAX_DIRECTORY_ENTRIES {
            std::fs::File::create(root.join(format!("entry-{n}"))).unwrap();
        }
        let result = page(&root, 64, None);
        assert!(matches!(
            result,
            Err(CoreError::ExecutionFailed(_)) | Err(CoreError::Timeout(_))
        ));
    }

    #[tokio::test]
    async fn independent_observation_and_committed_content_reject_false_pages() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        std::fs::write(root.join("report.txt"), b"source").unwrap();
        let mut task = Task::new("Inspect selected folder");
        let mut proposal = ActionProposal {
            id: uuid::Uuid::new_v4(),
            task_id: task.id,
            action: Action::ListDirectory {
                path: root.clone(),
                page_size: 64,
                cursor: None,
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: root.to_string_lossy().into_owned(),
            provenance: Provenance::user(),
            metadata: BTreeMap::new(),
        };
        crate::verification::bind_required_outcome(&mut proposal).unwrap();
        FileBroker::default().prepare(&mut proposal).await.unwrap();
        let page = page(&root, 64, None).unwrap();
        let receipt = ExecutionReceipt {
            executor: "native-os-executor".into(),
            summary: page.summary(),
            transient_data: serde_json::to_value(&page).unwrap(),
            rollback: None,
        };
        let observation = DeterministicObserver
            .observe(&proposal, &receipt)
            .await
            .unwrap();
        crate::verification::Verifier
            .verify(&proposal.expected_outcome, &observation)
            .unwrap();
        task.actions.insert(
            proposal.id,
            ActionState {
                proposal: proposal.clone(),
                status: ActionStatus::Verifying,
                attempts: 1,
                summary: None,
                error: None,
            },
        );
        let mut false_receipt = receipt.clone();
        false_receipt.transient_data["entries"][0]["name"] = serde_json::json!("invented.txt");
        assert!(
            DeterministicObserver
                .observe(&proposal, &false_receipt)
                .await
                .is_err()
        );
        assert!(
            crate::transitions::VerifiedAction::from_observation(
                &task,
                &proposal,
                &false_receipt,
                &observation,
                crate::transitions::VerificationMode::Execution
            )
            .is_err()
        );
        let mut mixed = observation.clone();
        let false_page: DirectoryPage =
            serde_json::from_value(false_receipt.transient_data.clone()).unwrap();
        mixed
            .evidence
            .push(crate::observation::Evidence::DirectoryPage {
                path: root.join("another-folder").to_str().unwrap().into(),
                page_size: 64,
                cursor: None,
                page_sha256: false_page.digest().unwrap(),
                snapshot_sha256: false_page.snapshot_sha256,
                total_entries: false_page.total_entries,
            });
        assert!(
            crate::transitions::VerifiedAction::from_observation(
                &task,
                &proposal,
                &false_receipt,
                &mixed,
                crate::transitions::VerificationMode::Execution
            )
            .is_err(),
            "evidence for another target cannot supply this page's content digest"
        );
        assert!(
            crate::transitions::VerifiedAction::from_observation(
                &task,
                &proposal,
                &receipt,
                &observation,
                crate::transitions::VerificationMode::Execution
            )
            .is_ok()
        );
        std::fs::write(root.join("later.txt"), b"later").unwrap();
        assert!(
            DeterministicObserver
                .observe(&proposal, &receipt)
                .await
                .is_err()
        );
        assert!(
            inspect(
                None,
                root,
                64,
                None,
                Some(chrono::Utc::now() - chrono::Duration::seconds(1))
            )
            .await
            .is_err()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_readers_keep_their_slots_until_the_blocking_work_retires() {
        let mut workers = Vec::new();
        let mut releases = Vec::new();
        let cancelled_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let all_cancelled = Arc::new(tokio::sync::Notify::new());
        for _ in 0..MAX_READERS {
            let (started_tx, started_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
            let observed = cancelled_count.clone();
            let retired = all_cancelled.clone();
            workers.push(tokio::spawn(bounded_read(move |cancelled| {
                let _ = started_tx.send(());
                let _ = release_rx.recv();
                if cancelled.load(Ordering::Acquire)
                    && observed.fetch_add(1, Ordering::AcqRel) + 1 == MAX_READERS
                {
                    retired.notify_one();
                }
                Ok(())
            })));
            releases.push(release_tx);
            tokio::time::timeout(Duration::from_secs(10), started_rx)
                .await
                .unwrap()
                .unwrap();
        }
        for worker in workers {
            worker.abort();
            assert!(worker.await.unwrap_err().is_cancelled());
        }
        let entered = Arc::new(AtomicBool::new(false));
        let observed = entered.clone();
        let next = tokio::spawn(bounded_read(move |_| {
            observed.store(true, Ordering::Release);
            Ok(())
        }));
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!entered.load(Ordering::Acquire));
        let all_retired = all_cancelled.notified();
        drop(releases);
        tokio::time::timeout(Duration::from_secs(10), all_retired)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(10), next)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(cancelled_count.load(Ordering::Acquire), MAX_READERS);
    }
}
