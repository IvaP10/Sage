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

    /// Check a single file target against an explicit read scope without
    /// consuming its contents. Call from the bounded filesystem lane.
    pub fn prepare_scoped_read(
        &self,
        path: &Path,
        scopes: &[crate::contracts::ResourceScope],
    ) -> CoreResult<()> {
        validate_file_path(path)?;
        let target = path.canonicalize()?;
        if !target.is_file() {
            return Err(CoreError::InvalidAction(
                "A streamed read requires an existing regular file".into(),
            ));
        }
        if self
            .protected_roots
            .iter()
            .any(|root| target.starts_with(root))
        {
            return Err(CoreError::PolicyDenied(
                "Sage internal state and installed components are unavailable to agent file tools"
                    .into(),
            ));
        }
        let scoped = scopes.iter().any(|scope| {
            scope.effects.contains(&crate::contracts::Effect::Read)
                && self
                    .validate_scope_root(&scope.root)
                    .is_ok_and(|root| target.starts_with(root))
        });
        if !scoped {
            return Err(CoreError::PermissionRequired(
                "The file is outside the selected read folders".into(),
            ));
        }
        // Opening checks current OS access without exposing or caching content.
        drop(std::fs::File::open(target)?);
        Ok(())
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
            Action::ListDirectory {
                path,
                page_size,
                cursor,
            } => Action::ListDirectory {
                path: self.resolve_path(path, false, true)?,
                page_size: *page_size,
                cursor: cursor.clone(),
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
            ExpectedOutcome::FileMatchesStream {
                path,
                channel_id,
                producer_node,
                maximum_bytes,
            } => ExpectedOutcome::FileMatchesStream {
                path: self.resolve(path, true)?,
                channel_id: channel_id.clone(),
                producer_node: producer_node.clone(),
                maximum_bytes: *maximum_bytes,
            },
            ExpectedOutcome::FileReadMatchesStream {
                path,
                channel_id,
                producer_node,
                output_port,
                consumer_node,
                maximum_bytes,
            } => ExpectedOutcome::FileReadMatchesStream {
                path: self.resolve(path, true)?,
                channel_id: channel_id.clone(),
                producer_node: producer_node.clone(),
                output_port: output_port.clone(),
                consumer_node: consumer_node.clone(),
                maximum_bytes: *maximum_bytes,
            },
            ExpectedOutcome::DirectoryPage {
                path,
                page_size,
                cursor,
            } => ExpectedOutcome::DirectoryPage {
                path: self.resolve_path(path, false, true)?,
                page_size: *page_size,
                cursor: cursor.clone(),
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
        self.resolve_path(path, may_not_exist, false)
    }

    fn resolve_path(&self, path: &Path, may_not_exist: bool, listing: bool) -> CoreResult<PathBuf> {
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
            .any(|root| canonical.starts_with(root) || (!listing && root.starts_with(&canonical)))
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
                if !valid_windows_component(name) {
                    return Err(CoreError::InvalidAction("Alternate streams, device names and ambiguous Windows file names are unavailable".into()));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

// Windows reserves these DOS device names even with extensions; superscript
// 1/2/3 also name COM/LPT devices. Keep the lexical check testable on Unix CI.
#[cfg(any(windows, test))]
fn valid_windows_component(name: &str) -> bool {
    let stem = name
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    let device = matches!(
        stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || ["COM", "LPT"].iter().any(|prefix| {
        stem.strip_prefix(prefix).is_some_and(|n| {
            matches!(
                n,
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
    });
    !name.is_empty()
        && !name.ends_with(['.', ' '])
        && !device
        && !name.chars().any(|c| c < ' ' || "<>:\"/\\|?*".contains(c))
}

fn target_resource(action: &Action) -> String {
    match action {
        Action::FetchPublic { url, .. } => url.clone(),
        Action::ReadFile { path, .. }
        | Action::ListDirectory { path, .. }
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
        | Action::SetApplicationControl { application, .. }
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
    #[test]
    fn listing_an_authorized_parent_never_grants_access_to_protected_children() {
        use crate::contracts::{Effect, ResourceScope, RunContract};
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().canonicalize().unwrap();
        let internal = parent.join("private-state");
        std::fs::create_dir(&internal).unwrap();
        let mut resolver = ResourceResolver::new(vec![parent.clone()]).unwrap();
        resolver
            .protect_paths(std::slice::from_ref(&internal))
            .unwrap();
        let id = uuid::Uuid::new_v4();
        let proposal = |path| ActionProposal {
            id: uuid::Uuid::new_v4(),
            task_id: id,
            action: Action::ListDirectory {
                path,
                page_size: 64,
                cursor: None,
            },
            expected_outcome: ExpectedOutcome::UserAnswered,
            target_resource: "model supplied".into(),
            provenance: crate::domain::Provenance::user(),
            metadata: Default::default(),
        };
        let listing = resolver
            .prepare_proposal(&proposal(parent.clone()))
            .unwrap();
        let mut scope = RunContract::local(id);
        scope.resources.push(ResourceScope {
            root: parent,
            effects: [Effect::Read].into(),
        });
        assert!(scope.covers(&listing.action));
        assert!(resolver.prepare_proposal(&proposal(internal)).is_err());
        let other = tempfile::tempdir().unwrap();
        let outside = resolver
            .prepare_proposal(&proposal(other.path().canonicalize().unwrap()))
            .unwrap();
        assert!(!scope.covers(&outside.action));
        scope.resources[0].effects = [Effect::Create].into();
        assert!(!scope.covers(&listing.action));
        let credentials = proposal(scope.resources[0].root.join(".ssh"));
        assert!(matches!(
            crate::policy::PolicyEngine
                .evaluate(
                    &credentials,
                    &crate::policy::PolicyContext {
                        task_request: "list".into(),
                        has_fresh_native_authentication: false,
                        is_recovery_attempt: false
                    }
                )
                .unwrap(),
            crate::policy::PolicyDecision::Deny { .. }
        ));
    }
    #[test]
    fn windows_device_stream_and_ambiguous_names_are_never_ordinary_files() {
        for name in [
            "CON",
            "nul.txt",
            "PRN.log",
            "AUX",
            "COM1.csv",
            "LPT9",
            "COM¹",
            "LPT².txt",
            "COM³.log",
            "CONIN$",
            "CONOUT$",
            "NUL .txt",
            "data:secret",
            "trailing.",
            "trailing ",
            "wild*",
            "wild?",
            "line\nfeed",
            "pipe|target",
            "relative\\child",
        ] {
            assert!(!super::valid_windows_component(name), "{name}");
        }
        for name in [
            "report.txt",
            "COM10.txt",
            "company",
            "こんにちは.txt",
            ".gitignore",
            "project notes",
        ] {
            assert!(super::valid_windows_component(name), "{name}");
        }
    }
    use super::*;

    #[test]
    fn speculative_file_read_requires_a_selected_scope_and_rejects_symlink_escape() {
        use crate::contracts::{Effect, ResourceScope};

        let selected = tempfile::tempdir().unwrap();
        let root = selected.path().canonicalize().unwrap();
        let inside = root.join("notes.txt");
        std::fs::write(&inside, "private note").unwrap();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside = outside_dir.path().join("outside.txt");
        std::fs::write(&outside, "outside note").unwrap();
        let resolver = ResourceResolver::new(Vec::new()).unwrap();
        let read_scope = ResourceScope {
            root: root.clone(),
            effects: [Effect::Read].into(),
        };

        assert!(
            resolver
                .prepare_scoped_read(&inside, std::slice::from_ref(&read_scope))
                .is_ok()
        );
        assert!(
            resolver
                .prepare_scoped_read(&outside, std::slice::from_ref(&read_scope))
                .is_err()
        );
        let create_only = ResourceScope {
            root: root.clone(),
            effects: [Effect::Create].into(),
        };
        assert!(
            resolver
                .prepare_scoped_read(&inside, &[create_only])
                .is_err()
        );

        #[cfg(unix)]
        {
            let link = root.join("linked.txt");
            std::os::unix::fs::symlink(&outside, &link).unwrap();
            assert!(resolver.prepare_scoped_read(&link, &[read_scope]).is_err());
        }
    }

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
        resolver
            .protect_paths(std::slice::from_ref(&model))
            .unwrap();
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
