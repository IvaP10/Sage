//! Directory-handle anchored I/O. No final-component symlinks, no parent path
//! re-resolution after preparation, exclusive staging, bounded reads.
use crate::capability::CapabilityGrant;
use crate::domain::{Action, ActionProposal};
use crate::{CoreError, CoreResult};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    ffi::OsString,
    io::{Read, Write},
    path::{Component, Path},
    sync::Mutex,
};
use uuid::Uuid;
use zeroize::Zeroizing;

pub(crate) const MAX_BYTES: u64 = 16 * 1024 * 1024;

use crate::contracts::FileIdentity as Identity;

fn identity(metadata: cap_std::fs::Metadata) -> CoreResult<Identity> {
    #[cfg(windows)]
    use cap_primitives::fs::_WindowsByHandle;
    #[cfg(unix)]
    use cap_std::fs::MetadataExt;
    #[cfg(unix)]
    let key = {
        if metadata.is_file() && metadata.nlink() != 1 {
            return Err(CoreError::PolicyDenied(
                "Multiply linked files are unavailable to agent tools".into(),
            ));
        }
        format!("{}:{}", metadata.dev(), metadata.ino())
    };
    #[cfg(windows)]
    let key = {
        if metadata.file_attributes() & 0x400 != 0 {
            return Err(CoreError::PolicyDenied(
                "Windows reparse points are unavailable to file tools".into(),
            ));
        }
        if metadata.number_of_links().is_none_or(|n| n != 1) && metadata.is_file() {
            return Err(CoreError::PolicyDenied(
                "File link identity is unavailable".into(),
            ));
        }
        let volume = metadata
            .volume_serial_number()
            .ok_or_else(|| CoreError::PolicyDenied("File volume identity is unavailable".into()))?;
        let index = metadata
            .file_index()
            .ok_or_else(|| CoreError::PolicyDenied("File identity is unavailable".into()))?;
        format!("{volume}:{index}")
    };
    #[cfg(not(any(unix, windows)))]
    let key = return Err(CoreError::ExecutorUnavailable(
        "File identity is unsupported".into(),
    ));
    Ok(Identity {
        key,
        size: metadata.len(),
        modified: format!("{:?}", metadata.modified()?),
        directory: metadata.is_dir(),
    })
}

pub struct PinnedPath {
    parent: Dir,
    leaf: OsString,
    before: Option<Identity>,
}

pub(crate) enum StreamWriteInput {
    Chunk(Zeroizing<Vec<u8>>),
    Finish,
}

pub(crate) enum StreamReadOutput {
    Chunk(Zeroizing<Vec<u8>>),
    Finish,
}

impl PinnedPath {
    pub(crate) fn open_directory(path: &Path) -> CoreResult<Self> {
        let pinned = if path.is_absolute() && path.parent().is_none() {
            crate::resources::validate_file_path(path)?;
            let parent = Dir::open_ambient_dir(path, cap_std::ambient_authority())?;
            let before = Some(identity(parent.dir_metadata()?)?);
            Self {
                parent,
                leaf: OsString::from("."),
                before,
            }
        } else {
            Self::open(path)?
        };
        if !pinned.is_directory() {
            return Err(CoreError::InvalidAction(
                "Select a directory to list".into(),
            ));
        }
        Ok(pinned)
    }

    pub(crate) fn directory_handle(&self) -> CoreResult<Dir> {
        self.revalidate()?;
        let directory = self.parent.open_dir_nofollow(&self.leaf)?;
        self.validate_directory_handle(&directory)?;
        Ok(directory)
    }

    pub(crate) fn validate_directory_handle(&self, directory: &Dir) -> CoreResult<Identity> {
        self.revalidate()?;
        let observed = identity(directory.dir_metadata()?)?;
        if !observed.directory || Some(&observed) != self.before.as_ref() {
            return Err(CoreError::VerificationFailed(
                "Directory changed during inspection; start its listing again".into(),
            ));
        }
        Ok(observed)
    }

    pub fn open(path: &Path) -> CoreResult<Self> {
        crate::resources::validate_file_path(path)?;
        let parent_path = path
            .parent()
            .ok_or_else(|| CoreError::InvalidAction("Path has no parent".into()))?;
        let root = parent_path
            .ancestors()
            .last()
            .ok_or_else(|| CoreError::InvalidAction("Path has no root".into()))?;
        if !path.is_absolute() {
            return Err(CoreError::InvalidAction("Expected an absolute path".into()));
        }
        let mut parent = Dir::open_ambient_dir(root, cap_std::ambient_authority())?;
        for component in parent_path.components() {
            match component {
                Component::Normal(name) => {
                    parent = parent.open_dir_nofollow(name)?;
                }
                Component::RootDir | Component::Prefix(_) => {}
                _ => {
                    return Err(CoreError::PolicyDenied(
                        "Relative components are prohibited".into(),
                    ));
                }
            }
        }
        let leaf = path
            .file_name()
            .ok_or_else(|| CoreError::InvalidAction("Missing filename".into()))?
            .to_owned();
        let mut pinned = Self {
            parent,
            leaf,
            before: None,
        };
        pinned.before = pinned.current()?;
        Ok(pinned)
    }

    fn current(&self) -> CoreResult<Option<Identity>> {
        match self.parent.symlink_metadata(&self.leaf) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !(metadata.is_file() || metadata.is_dir()) {
                    return Err(CoreError::PolicyDenied(
                        "Symlinks and special files are unavailable".into(),
                    ));
                }
                identity(metadata).map(Some)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    pub fn exists(&self) -> bool {
        self.before.is_some()
    }
    pub fn is_directory(&self) -> bool {
        self.before
            .as_ref()
            .is_some_and(|identity| identity.directory)
    }
    pub fn current_identity(&self) -> CoreResult<String> {
        self.current()?
            .map(|identity| identity.key)
            .ok_or_else(|| CoreError::VerificationFailed("File identity is missing".into()))
    }
    pub(crate) fn snapshot_identity(&self) -> CoreResult<Option<Identity>> {
        self.revalidate()?;
        Ok(self.before.clone())
    }
    pub(crate) fn parent_identity(&self) -> CoreResult<String> {
        Ok(identity(self.parent.dir_metadata()?)?.key)
    }
    pub(crate) fn require_empty_directory(&self) -> CoreResult<()> {
        self.revalidate()?;
        let directory = self.parent.open_dir_nofollow(&self.leaf)?;
        if directory.entries()?.next().transpose()?.is_some() {
            return Err(CoreError::ApprovalRejected(
                "Undo refused: folder contains files".into(),
            ));
        }
        self.revalidate()
    }
    pub fn remove_verified(&self, expected: &str) -> CoreResult<()> {
        if format!("{:x}", Sha256::digest(self.read(MAX_BYTES)?)) != expected {
            return Err(CoreError::ApprovalRejected(
                "Undo refused: file content changed".into(),
            ));
        }
        self.revalidate()?;
        self.parent.remove_file(&self.leaf)?;
        self.parent.try_clone()?.into_std_file().sync_all()?;
        Ok(())
    }
    pub fn remove_empty_verified(&self, expected: &str) -> CoreResult<()> {
        if self.current_identity()? != expected {
            return Err(CoreError::ApprovalRejected(
                "Undo refused: folder identity changed".into(),
            ));
        }
        self.revalidate()?;
        self.parent.remove_dir(&self.leaf)?;
        self.parent.try_clone()?.into_std_file().sync_all()?;
        Ok(())
    }

    fn revalidate(&self) -> CoreResult<()> {
        if self.current()? != self.before {
            return Err(CoreError::ApprovalRejected(
                "File target changed after preparation".into(),
            ));
        }
        Ok(())
    }

    pub fn read(&self, limit: u64) -> CoreResult<Vec<u8>> {
        self.revalidate()?;
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let file = self.parent.open_with(&self.leaf, &options)?;
        if !file.metadata()?.is_file() {
            return Err(CoreError::PolicyDenied(
                "Only regular files can be read".into(),
            ));
        }
        if Some(identity(file.metadata()?)?) != self.before {
            return Err(CoreError::ApprovalRejected("Read target changed".into()));
        }
        let mut bytes = Vec::new();
        file.take(limit.min(MAX_BYTES) + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > limit.min(MAX_BYTES) {
            return Err(CoreError::ExecutionFailed(
                "File exceeds its authorized read limit".into(),
            ));
        }
        self.revalidate()?;
        Ok(bytes)
    }

    /// Read a pinned regular file into a one-item-bounded async channel. The
    /// caller must run this blocking operation on Sage's bounded file lane.
    pub(crate) fn read_stream(
        &self,
        limit: u64,
        maximum_item_bytes: u64,
        sender: tokio::sync::mpsc::Sender<StreamReadOutput>,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> CoreResult<(String, u64)> {
        let limit = limit.min(MAX_BYTES);
        if limit == 0 || maximum_item_bytes == 0 {
            return Err(CoreError::InvalidAction(
                "Streamed file read requires positive total and per-item byte limits".into(),
            ));
        }
        self.revalidate()?;
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let mut file = self.parent.open_with(&self.leaf, &options)?;
        let before = identity(file.metadata()?)?;
        if before.directory || Some(&before) != self.before.as_ref() {
            return Err(CoreError::ApprovalRejected(
                "Read target changed before streaming".into(),
            ));
        }
        if before.size > limit {
            return Err(CoreError::ExecutionFailed(
                "File exceeds its authorized stream read limit".into(),
            ));
        }

        let mut buffer = Zeroizing::new([0u8; 64 * 1024]);
        let chunk_limit = maximum_item_bytes.min(buffer.len() as u64) as usize;
        let mut digest = Sha256::new();
        let mut total = 0u64;
        loop {
            if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                return Err(CoreError::Cancelled);
            }
            let count = file.read(&mut buffer[..chunk_limit])?;
            if count == 0 {
                break;
            }
            let next_total = total
                .checked_add(count as u64)
                .filter(|value| *value <= limit)
                .ok_or_else(|| {
                    CoreError::ExecutionFailed(
                        "File grew beyond its authorized stream read limit".into(),
                    )
                })?;
            digest.update(&buffer[..count]);
            sender
                .blocking_send(StreamReadOutput::Chunk(Zeroizing::new(
                    buffer[..count].to_vec(),
                )))
                .map_err(|_| CoreError::Cancelled)?;
            total = next_total;
        }
        if identity(file.metadata()?)? != before {
            return Err(CoreError::ApprovalRejected(
                "Read target changed while streaming".into(),
            ));
        }
        self.revalidate()?;
        sender
            .blocking_send(StreamReadOutput::Finish)
            .map_err(|_| CoreError::Cancelled)?;
        Ok((format!("{:x}", digest.finalize()), total))
    }

    pub fn digest(&self, limit: u64) -> CoreResult<String> {
        self.digest_with_size(limit).map(|(digest, _)| digest)
    }

    pub fn digest_with_size(&self, limit: u64) -> CoreResult<(String, u64)> {
        self.revalidate()?;
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let mut file = self.parent.open_with(&self.leaf, &options)?;
        if !file.metadata()?.is_file()
            || Some(identity(file.metadata()?)?) != self.before
            || file.metadata()?.len() > limit
        {
            return Err(CoreError::PolicyDenied(
                "Asset identity or size is invalid".into(),
            ));
        }
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        let mut total = 0u64;
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            total = total.saturating_add(count as u64);
            if total > limit {
                return Err(CoreError::PolicyDenied(
                    "Asset grew beyond its limit".into(),
                ));
            }
            hash.update(&buffer[..count]);
        }
        if Some(identity(file.metadata()?)?) != self.before {
            return Err(CoreError::ApprovalRejected(
                "Asset changed while hashing".into(),
            ));
        }
        self.revalidate()?;
        Ok((format!("{:x}", hash.finalize()), total))
    }

    pub fn write(&self, content: &[u8], overwrite: bool) -> CoreResult<()> {
        self.revalidate()?;
        if content.len() as u64 > MAX_BYTES {
            return Err(CoreError::InvalidAction("File exceeds write budget".into()));
        }
        if self.before.is_some() && !overwrite {
            return Err(CoreError::ExecutionFailed(
                "Destination already exists".into(),
            ));
        }
        if self.before.as_ref().is_some_and(|i| i.directory) {
            return Err(CoreError::PolicyDenied("Cannot replace a directory".into()));
        }
        let temporary = OsString::from(format!(".sage-{}.tmp", Uuid::new_v4()));
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let result = (|| {
            let mut file = self.parent.open_with(&temporary, &options)?;
            file.write_all(content)?;
            file.sync_all()?;
            self.revalidate()?;
            self.publish_temporary(&temporary, overwrite)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = self.parent.remove_file(&temporary);
        }
        result
    }

    pub(crate) fn write_stream<R>(
        &self,
        mut receiver: tokio::sync::mpsc::Receiver<StreamWriteInput>,
        limit: u64,
        overwrite: bool,
        cancelled: &std::sync::atomic::AtomicBool,
        before_publish: impl FnOnce(&str, u64) -> CoreResult<R>,
    ) -> CoreResult<(String, u64, R)> {
        self.revalidate()?;
        if limit == 0 || limit > MAX_BYTES {
            return Err(CoreError::InvalidAction(
                "Streamed file write exceeds its authorized byte bound".into(),
            ));
        }
        if self.before.is_some() && !overwrite {
            return Err(CoreError::ExecutionFailed(
                "Destination already exists".into(),
            ));
        }
        if self
            .before
            .as_ref()
            .is_some_and(|identity| identity.directory)
        {
            return Err(CoreError::PolicyDenied("Cannot replace a directory".into()));
        }

        let temporary = OsString::from(format!(".sage-{}.tmp", Uuid::new_v4()));
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt;
            options.mode(0o600);
        }

        let result = (|| {
            let mut file = self.parent.open_with(&temporary, &options)?;
            let mut buffer = Zeroizing::new([0u8; 64 * 1024]);
            let mut digest = Sha256::new();
            let mut total_bytes = 0u64;
            let mut current = Zeroizing::new(Vec::new());
            let mut current_offset = 0usize;
            let mut finished = false;
            loop {
                if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                    return Err(CoreError::Cancelled);
                }
                if current_offset == current.len() && !finished {
                    match receiver.blocking_recv() {
                        Some(StreamWriteInput::Chunk(chunk)) => {
                            current = chunk;
                            current_offset = 0;
                        }
                        Some(StreamWriteInput::Finish) => finished = true,
                        None => {
                            return Err(CoreError::VerificationFailed(
                                "Streamed file input closed before its explicit end".into(),
                            ));
                        }
                    }
                }
                if finished && current_offset == current.len() {
                    break;
                }
                let count = if current_offset < current.len() {
                    let count = (current.len() - current_offset).min(buffer.len());
                    buffer[..count]
                        .copy_from_slice(&current[current_offset..current_offset + count]);
                    current_offset += count;
                    count
                } else {
                    0
                };
                let next_total = total_bytes.checked_add(count as u64).ok_or_else(|| {
                    CoreError::InvalidAction("Streamed file size overflowed".into())
                })?;
                if next_total > limit {
                    return Err(CoreError::InvalidAction(
                        "Streamed file exceeds its authorized byte bound".into(),
                    ));
                }
                file.write_all(&buffer[..count])?;
                digest.update(&buffer[..count]);
                total_bytes = next_total;
            }
            file.sync_all()?;
            self.revalidate()?;
            let stream_sha256 = format!("{:x}", digest.finalize());
            let commit_value = before_publish(&stream_sha256, total_bytes)?;
            self.revalidate()?;
            self.publish_temporary(&temporary, overwrite)?;
            Ok((stream_sha256, total_bytes, commit_value))
        })();
        if result.is_err() {
            let _ = self.parent.remove_file(&temporary);
        }
        result
    }

    fn publish_temporary(&self, temporary: &OsString, overwrite: bool) -> CoreResult<()> {
        self.revalidate()?;
        if overwrite {
            // Atomic same-directory replacement. Never remove the old
            // destination first. External concurrent writers remain a
            // documented platform limitation until CAS support is qualified.
            self.parent.rename(temporary, &self.parent, &self.leaf)?;
        } else {
            // Atomic no-clobber publication, even if another process creates
            // the destination between revalidation and commit.
            self.parent.hard_link(temporary, &self.parent, &self.leaf)?;
            self.parent.remove_file(temporary)?;
        }
        self.parent.try_clone()?.into_std_file().sync_all()?;
        Ok(())
    }

    pub fn create_folder(&self) -> CoreResult<()> {
        self.revalidate()?;
        if self.before.is_some() {
            return Err(CoreError::ExecutionFailed("Folder already exists".into()));
        }
        self.parent.create_dir(&self.leaf)?;
        Ok(())
    }
}

struct PreparedFile {
    path: PinnedPath,
    digest: String,
}
#[derive(Default)]
pub struct FileBroker {
    prepared: Mutex<HashMap<Uuid, PreparedFile>>,
}

impl FileBroker {
    pub(crate) fn preparation_guard(&self, id: Uuid) -> PreparationGuard<'_> {
        PreparationGuard { files: self, id }
    }

    pub async fn prepare(&self, proposal: &mut ActionProposal) -> CoreResult<()> {
        let path = match &proposal.action {
            Action::ReadFile { path, .. }
            | Action::ListDirectory { path, .. }
            | Action::WriteFile { path, .. }
            | Action::CreateFolder { path } => path,
            _ => return Ok(()),
        };
        let path = path.clone();
        let directory = matches!(proposal.action, Action::ListDirectory { .. });
        let pinned = super::io::bounded_read(move |_| {
            if directory {
                PinnedPath::open_directory(&path)
            } else {
                PinnedPath::open(&path)
            }
        })
        .await?;
        proposal.metadata.insert(
            "file_precondition".into(),
            serde_json::to_string(&pinned.before)?,
        );
        let prepared = PreparedFile {
            path: pinned,
            digest: crate::policy::approval_digest(proposal)?,
        };
        self.prepared
            .lock()
            .map_err(|_| CoreError::ExecutionFailed("File broker unavailable".into()))?
            .insert(proposal.id, prepared);
        Ok(())
    }
    pub fn discard(&self, id: Uuid) {
        if let Ok(mut files) = self.prepared.lock() {
            files.remove(&id);
        }
    }
    pub fn take(
        &self,
        proposal: &ActionProposal,
        grant: &CapabilityGrant,
    ) -> CoreResult<PinnedPath> {
        let prepared = self
            .prepared
            .lock()
            .map_err(|_| CoreError::ExecutionFailed("File broker unavailable".into()))?
            .remove(&proposal.id)
            .ok_or_else(|| CoreError::CapabilityRejected("No prepared file handle".into()))?;
        if prepared.digest != grant.action_digest
            || prepared.digest != crate::policy::approval_digest(proposal)?
        {
            return Err(CoreError::CapabilityRejected(
                "Prepared file does not match grant".into(),
            ));
        }
        prepared.path.revalidate()?;
        Ok(prepared.path)
    }
}

pub(crate) struct PreparationGuard<'a> {
    files: &'a FileBroker,
    id: Uuid,
}
impl Drop for PreparationGuard<'_> {
    fn drop(&mut self) {
        self.files.discard(self.id);
    }
}

pub fn hash_file(path: &Path) -> CoreResult<String> {
    hash_file_with_size(path).map(|(digest, _)| digest)
}

pub fn hash_file_with_size(path: &Path) -> CoreResult<(String, u64)> {
    PinnedPath::open(path)?.digest_with_size(MAX_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exclusive_write_and_changed_target_are_enforced() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("note");
        let prepared = PinnedPath::open(&path).unwrap();
        std::fs::write(&path, "external").unwrap();
        assert!(prepared.write(b"agent", false).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "external");
        let prepared = PinnedPath::open(&path).unwrap();
        prepared.write(b"replacement", true).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "replacement");
    }
    #[cfg(unix)]
    #[test]
    fn symlink_replacement_cannot_redirect_a_prepared_write() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        std::fs::create_dir(root.join("parent")).unwrap();
        std::fs::create_dir(root.join("outside")).unwrap();
        let prepared = PinnedPath::open(&root.join("parent/file")).unwrap();
        std::fs::rename(root.join("parent"), root.join("original")).unwrap();
        symlink(root.join("outside"), root.join("parent")).unwrap();
        prepared.write(b"approved", false).unwrap();
        assert!(!root.join("outside/file").exists());
        assert_eq!(
            std::fs::read(root.join("original/file")).unwrap(),
            b"approved"
        );
        assert!(PinnedPath::open(&root.join("parent/file")).is_err());
    }
    #[test]
    fn read_limit_is_applied_to_the_open_handle() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().canonicalize().unwrap().join("large");
        std::fs::write(&path, [1; 100]).unwrap();
        assert!(PinnedPath::open(&path).unwrap().read(10).is_err());
    }
}
