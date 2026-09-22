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

const MAX_BYTES: u64 = 16 * 1024 * 1024;

use crate::contracts::FileIdentity as Identity;

fn identity(metadata: cap_std::fs::Metadata) -> CoreResult<Identity> {
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
        if metadata.number_of_links().is_none_or(|n| n != 1) && metadata.is_file() {
            return Err(CoreError::PolicyDenied(
                "File link identity is unavailable".into(),
            ));
        }
        format!(
            "{:?}:{:?}",
            metadata.volume_serial_number(),
            metadata.file_index()
        )
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

impl PinnedPath {
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
    pub fn remove_verified(&self, expected: &str) -> CoreResult<()> {
        if format!("{:x}", Sha256::digest(self.read(MAX_BYTES)?)) != expected {
            return Err(CoreError::ApprovalRejected(
                "Undo refused: file content changed".into(),
            ));
        }
        self.revalidate()?;
        self.parent.remove_file(&self.leaf)?;
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

    pub fn digest(&self, limit: u64) -> CoreResult<String> {
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
        Ok(format!("{:x}", hash.finalize()))
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
            if overwrite {
                // Atomic same-directory replacement. Never remove the old
                // destination first. External concurrent writers remain a
                // documented platform limitation until CAS support is qualified.
                self.parent.rename(&temporary, &self.parent, &self.leaf)?;
            } else {
                // Atomic no-clobber publication, even if another process creates
                // the destination between revalidation and commit.
                self.parent
                    .hard_link(&temporary, &self.parent, &self.leaf)?;
                self.parent.remove_file(&temporary)?;
            }
            self.parent.try_clone()?.into_std_file().sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = self.parent.remove_file(&temporary);
        }
        result
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
    pub fn prepare(&self, proposal: &mut ActionProposal) -> CoreResult<()> {
        let path = match &proposal.action {
            Action::ReadFile { path, .. }
            | Action::WriteFile { path, .. }
            | Action::CreateFolder { path } => path,
            _ => return Ok(()),
        };
        let pinned = PinnedPath::open(path)?;
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

pub fn hash_file(path: &Path) -> CoreResult<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(PinnedPath::open(path)?.read(MAX_BYTES)?)
    ))
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
