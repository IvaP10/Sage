use std::path::{Component, Path, PathBuf};

use crate::domain::{Action, ActionProposal, Condition, ExpectedOutcome};
use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone)]
pub struct ResourceResolver {
    allowed_roots: Vec<PathBuf>,
    protected_roots: Vec<PathBuf>,
    preview_only: bool,
}

impl ResourceResolver {
    pub fn protect_paths(&mut self, paths: &[PathBuf]) -> CoreResult<()> {
        for path in paths {
            self.protected_roots.push(path.canonicalize()?);
        }
        self.protected_roots.sort();
        self.protected_roots.dedup();
        Ok(())
    }
    pub fn new(allowed_roots: Vec<PathBuf>) -> CoreResult<Self> {
        let mut resolved = Vec::new();
        for root in allowed_roots {
            if root.exists() {
                resolved.push(root.canonicalize()?);
            }
        }
        resolved.sort();
        resolved.dedup();
        Ok(Self {
            allowed_roots: resolved,
            protected_roots: Vec::new(),
            preview_only: false,
        })
    }

    pub fn platform_default(internal_root: PathBuf) -> CoreResult<Self> {
        let mut resolver = Self::new(Vec::new())?;
        resolver.protected_roots.push(internal_root.canonicalize()?);
        if let Ok(executable) = std::env::current_exe()
            && let Some(parent) = executable.parent()
        {
            resolver.protected_roots.push(parent.canonicalize()?);
        }
        Ok(resolver)
    }

    /// Resolve identity for an exact-action preview. This does not grant access;
    /// the broker must authorize the result before execution.
    pub fn prepare_proposal(&self, proposal: &ActionProposal) -> CoreResult<ActionProposal> {
        let mut preparation = self.clone();
        preparation.preview_only = true;
        preparation.resolve_proposal(proposal)
    }

    pub fn validate_scope_root(&self, root: &Path) -> CoreResult<PathBuf> {
        validate_file_path(root)?;
        if !root.is_absolute() || !root.is_dir() {
            return Err(CoreError::PermissionRequired(
                "Select an existing absolute folder".into(),
            ));
        }
        let root = root.canonicalize()?;
        if self
            .protected_roots
            .iter()
            .any(|internal| root.starts_with(internal))
        {
            return Err(CoreError::PolicyDenied(
                "Sage internal folders cannot be granted to tools".into(),
            ));
        }
        Ok(root)
    }

    pub fn resolve_proposal(&self, proposal: &ActionProposal) -> CoreResult<ActionProposal> {
        let mut resolved = proposal.clone();
        resolved.action = self.resolve_action(&proposal.action)?;
        resolved.expected_outcome = self.resolve_expected(&proposal.expected_outcome)?;
        resolved.target_resource = target_resource(&resolved.action);
        Ok(resolved)
    }

    fn resolve_action(&self, action: &Action) -> CoreResult<Action> {
        Ok(match action {
            Action::FetchPublic { url, max_bytes } => Action::FetchPublic {
                url: crate::network::public_url(url)?.to_string(),
                max_bytes: *max_bytes,
            },
            Action::ReadFile { path, max_bytes } => Action::ReadFile {
                path: self.resolve(path, false)?,
                max_bytes: *max_bytes,
            },
            Action::WriteFile {
                path,
                content,
                overwrite,
            } => Action::WriteFile {
                path: self.resolve(path, true)?,
                content: content.clone(),
                overwrite: *overwrite,
            },
            Action::MoveFile {
                source,
                destination,
            } => Action::MoveFile {
                source: self.resolve(source, false)?,
                destination: self.resolve(destination, true)?,
            },
            Action::DeleteFile { path } => Action::DeleteFile {
                path: self.resolve(path, false)?,
            },
            Action::CreateFolder { path } => Action::CreateFolder {
                path: self.resolve(path, true)?,
            },
            Action::DownloadFile { url, destination } => Action::DownloadFile {
                url: url.clone(),
                destination: self.resolve(destination, true)?,
            },
            Action::UploadFile {
                path,
                destination_origin,
            } => Action::UploadFile {
                path: self.resolve(path, false)?,
                destination_origin: destination_origin.clone(),
            },
            Action::RunCommand {
                program,
                args,
                working_directory,
                network,
                timeout_seconds,
            } => Action::RunCommand {
                program: program.clone(),
                args: args.clone(),
                working_directory: working_directory
                    .as_ref()
                    .map(|path| self.resolve(path, false))
                    .transpose()?,
                network: *network,
                timeout_seconds: *timeout_seconds,
            },
            Action::WaitForCondition {
                condition,
                timeout_ms,
            } => Action::WaitForCondition {
                condition: self.resolve_condition(condition)?,
                timeout_ms: *timeout_ms,
            },
            other => other.clone(),
        })
    }

    fn resolve_expected(&self, expected: &ExpectedOutcome) -> CoreResult<ExpectedOutcome> {
        Ok(match expected {
            ExpectedOutcome::Condition { condition } => ExpectedOutcome::Condition {
                condition: self.resolve_condition(condition)?,
            },
            ExpectedOutcome::FileContains { path, sha256 } => ExpectedOutcome::FileContains {
                path: self.resolve(path, true)?,
                sha256: sha256.clone(),
            },
            other => other.clone(),
        })
    }

    fn resolve_condition(&self, condition: &Condition) -> CoreResult<Condition> {
        Ok(match condition {
            Condition::FolderExists { path } => Condition::FolderExists {
                path: self.resolve(path, true)?,
            },
            Condition::FileExists { path } => Condition::FileExists {
                path: self.resolve(path, true)?,
            },
            Condition::FileAbsent { path } => Condition::FileAbsent {
                path: self.resolve(path, true)?,
            },
            other => other.clone(),
        })
    }

    fn resolve(&self, path: &Path, may_not_exist: bool) -> CoreResult<PathBuf> {
        validate_file_path(path)?;

        let canonical = if path.exists() {
            path.canonicalize()?
        } else if may_not_exist {
            let parent = path
                .parent()
                .ok_or_else(|| CoreError::InvalidAction("path has no parent directory".into()))?;
            let canonical_parent = parent.canonicalize()?;
            let file_name = path
                .file_name()
                .ok_or_else(|| CoreError::InvalidAction("path has no final component".into()))?;
            canonical_parent.join(file_name)
        } else {
            return Err(CoreError::InvalidAction(format!(
                "resource does not exist: {}",
                path.display()
            )));
        };

        if self
            .protected_roots
            .iter()
            .any(|root| canonical.starts_with(root) || root.starts_with(&canonical))
        {
            return Err(CoreError::PolicyDenied(
                "Sage internal state and installed components are unavailable to agent file tools"
                    .into(),
            ));
        }
        if !self.preview_only
            && !self
                .allowed_roots
                .iter()
                .any(|root| canonical.starts_with(root))
        {
            return Err(CoreError::PermissionRequired(format!(
                "{} is outside the task's authorized filesystem roots",
                canonical.display()
            )));
        }
        Ok(canonical)
    }
}

/// Reject ambiguous lexical forms before scope matching or filesystem access.
pub(crate) fn validate_file_path(path: &Path) -> CoreResult<()> {
    if !path.is_absolute()
        || path.components().any(|part| part == Component::ParentDir)
        || path.as_os_str().as_encoded_bytes().contains(&0)
    {
        return Err(CoreError::InvalidAction(
            "Filesystem paths must be absolute, without '..' or NUL".into(),
        ));
    }
    #[cfg(windows)]
    for part in path.components() {
        match part {
            Component::Prefix(prefix)
                if !matches!(
                    prefix.kind(),
                    std::path::Prefix::Disk(_) | std::path::Prefix::VerbatimDisk(_)
                ) =>
            {
                return Err(CoreError::InvalidAction(
                    "Network shares and Windows device paths are unavailable to file tools".into(),
                ));
            }
            Component::Normal(name) => {
                let name = name.to_str().ok_or_else(|| {
                    CoreError::InvalidAction("File name is not valid Unicode".into())
                })?;
                let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
                let device = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
                    || ["COM", "LPT"].iter().any(|prefix| {
                        stem.strip_prefix(prefix).is_some_and(|n| {
                            matches!(n, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
                        })
                    });
                if name.contains(':') || name.ends_with(['.', ' ']) || device {
                    return Err(CoreError::InvalidAction("Alternate streams, device names and ambiguous Windows file names are unavailable".into()));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn target_resource(action: &Action) -> String {
    match action {
        Action::FetchPublic { url, .. } => url.clone(),
        Action::ReadFile { path, .. }
        | Action::WriteFile { path, .. }
        | Action::DeleteFile { path }
        | Action::CreateFolder { path }
        | Action::UploadFile { path, .. } => path.to_string_lossy().into_owned(),
        Action::MoveFile {
            source,
            destination,
        } => {
            format!("{} -> {}", source.display(), destination.display())
        }
        Action::DownloadFile { destination, .. } => destination.to_string_lossy().into_owned(),
        Action::OpenApplication { application }
        | Action::CloseApplication { application }
        | Action::ClickElement { application, .. }
        | Action::TypeText { application, .. }
        | Action::PressShortcut { application, .. }
        | Action::SendMessage { application, .. } => application.clone(),
        Action::NavigateUrl { url, .. } => url.clone(),
        Action::SubmitForm { origin, form_id } => format!("{origin}#{form_id}"),
        Action::RunCommand { program, .. } => program.clone(),
        Action::InstallApplication { source } => source.clone(),
        Action::ChangeSetting { namespace, key, .. } => format!("{namespace}.{key}"),
        Action::WaitForCondition { .. } => "condition".into(),
        Action::AskUser { .. } => "user".into(),
    }
}

#[cfg(test)]
mod v2_tests {
    use super::*;
    #[test]
    fn model_assets_and_internal_state_cannot_be_prepared_even_under_a_parent_grant() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let internal = root.join("state");
        std::fs::create_dir(&internal).unwrap();
        let model = root.join("model.gguf");
        std::fs::write(&model, b"pinned model").unwrap();
        let ordinary = root.join("notes.txt");
        std::fs::write(&ordinary, b"ordinary").unwrap();
        let mut resolver = ResourceResolver::platform_default(internal.clone()).unwrap();
        resolver.protect_paths(&[model.clone()]).unwrap();
        resolver.validate_scope_root(&root).unwrap();
        assert!(resolver.resolve(&model, false).is_err());
        assert!(resolver.resolve(&internal, false).is_err());
        let mut preview = resolver.clone();
        preview.preview_only = true;
        assert!(preview.resolve(&model, false).is_err());
        assert!(preview.resolve(&internal.join("new.db"), true).is_err());
        assert_eq!(preview.resolve(&ordinary, false).unwrap(), ordinary);
    }
    #[cfg(windows)]
    #[test]
    fn windows_device_network_and_alternate_stream_paths_are_rejected() {
        for path in [
            r"\\server\share\file",
            r"\\.\C:\file",
            r"C:\work\notes.txt:secret",
            r"C:\work\NUL.txt",
            r"C:\work\name.",
        ] {
            assert!(validate_file_path(Path::new(path)).is_err(), "{path}");
        }
    }
}
