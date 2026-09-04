use std::path::{Component, Path, PathBuf};

use crate::domain::{Action, ActionProposal, Condition, ExpectedOutcome};
use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone)]
pub struct ResourceResolver {
    allowed_roots: Vec<PathBuf>,
}

impl ResourceResolver {
    pub fn new(allowed_roots: Vec<PathBuf>) -> CoreResult<Self> {
        let mut resolved = Vec::new();
        for root in allowed_roots {
            if root.exists() {
                resolved.push(root.canonicalize()?);
            }
        }
        if resolved.is_empty() {
            return Err(CoreError::PermissionRequired(
                "no filesystem roots have been authorized".into(),
            ));
        }
        resolved.sort();
        resolved.dedup();
        Ok(Self {
            allowed_roots: resolved,
        })
    }

    pub fn platform_default(extra_root: PathBuf) -> CoreResult<Self> {
        let user = directories::UserDirs::new().ok_or_else(|| {
            CoreError::PermissionRequired("user folders could not be resolved".into())
        })?;
        let mut roots = vec![extra_root];
        if let Some(path) = user.desktop_dir() {
            roots.push(path.to_path_buf());
        }
        if let Some(path) = user.document_dir() {
            roots.push(path.to_path_buf());
        }
        if let Some(path) = user.download_dir() {
            roots.push(path.to_path_buf());
        }
        Self::new(roots)
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
                path: self.resolve(path, false)?,
                sha256: sha256.clone(),
            },
            other => other.clone(),
        })
    }

    fn resolve_condition(&self, condition: &Condition) -> CoreResult<Condition> {
        Ok(match condition {
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
        if path.as_os_str().is_empty() || path.components().any(|part| part == Component::ParentDir)
        {
            return Err(CoreError::InvalidAction(
                "filesystem paths must be absolute and may not contain '..'".into(),
            ));
        }
        if !path.is_absolute() {
            return Err(CoreError::InvalidAction(
                "filesystem paths must be absolute after native resolution".into(),
            ));
        }

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

        if !self
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

fn target_resource(action: &Action) -> String {
    match action {
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
