use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::Provenance;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDomain {
    Native,
    Browser,
    Sandbox,
    Privileged,
    UserInteraction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementSelector {
    pub role: Option<String>,
    pub label: Option<String>,
    pub automation_id: Option<String>,
    pub browser_selector: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Condition {
    FileExists { path: PathBuf },
    FolderExists { path: PathBuf },
    FileAbsent { path: PathBuf },
    ApplicationRunning { application: String },
    UrlEquals { url: String },
    ElementPresent { selector: ElementSelector },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExpectedOutcome {
    DirectoryPage {
        path: PathBuf,
        page_size: u32,
        cursor: Option<String>,
    },
    SignedApplication {
        target: crate::application_target::ApplicationTarget,
    },
    ApplicationControlValue {
        target: crate::application_target::ApplicationTarget,
        control_id: String,
        value: ApplicationControlValue,
    },
    PublicResource {
        url: String,
    },
    Condition {
        condition: Condition,
    },
    FileContains {
        path: PathBuf,
        sha256: String,
    },
    FileMatchesStream {
        path: PathBuf,
        channel_id: String,
        producer_node: String,
        maximum_bytes: u64,
    },
    FileReadMatchesStream {
        path: PathBuf,
        channel_id: String,
        producer_node: String,
        output_port: String,
        consumer_node: String,
        maximum_bytes: u64,
    },
    CommandExit {
        code: i32,
    },
    ExternalSuccess {
        marker: String,
    },
    UserAnswered,
}

/// The only value types exposed by Sage's first-party learned application
/// control primitive. These values remain data; the native adapter never
/// interprets them as code or selectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ApplicationControlValue {
    Boolean(bool),
    Number(f64),
}

impl ApplicationControlValue {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Boolean(_) => Ok(()),
            Self::Number(value) if value.is_finite() => Ok(()),
            Self::Number(_) => Err("Application control value must be finite".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    FetchPublic {
        url: String,
        max_bytes: u64,
    },
    OpenApplication {
        application: String,
    },
    SetApplicationControl {
        application: String,
        system_id: Uuid,
        system_fingerprint: String,
        capability_id: String,
        control_id: String,
        value: ApplicationControlValue,
    },
    CloseApplication {
        application: String,
    },
    ReadFile {
        path: PathBuf,
        max_bytes: u64,
    },
    ListDirectory {
        path: PathBuf,
        page_size: u32,
        cursor: Option<String>,
    },
    WriteFile {
        path: PathBuf,
        content: String,
        overwrite: bool,
    },
    MoveFile {
        source: PathBuf,
        destination: PathBuf,
    },
    DeleteFile {
        path: PathBuf,
    },
    CreateFolder {
        path: PathBuf,
    },
    ClickElement {
        application: String,
        selector: ElementSelector,
    },
    TypeText {
        application: String,
        selector: Option<ElementSelector>,
        text: String,
        sensitive: bool,
    },
    PressShortcut {
        application: String,
        keys: Vec<String>,
    },
    NavigateUrl {
        url: String,
        new_tab: bool,
    },
    DownloadFile {
        url: String,
        destination: PathBuf,
    },
    UploadFile {
        path: PathBuf,
        destination_origin: String,
    },
    SendMessage {
        application: String,
        recipient: String,
        content: String,
    },
    SubmitForm {
        origin: String,
        form_id: String,
    },
    RunCommand {
        program: String,
        args: Vec<String>,
        working_directory: Option<PathBuf>,
        network: bool,
        timeout_seconds: u32,
    },
    InstallApplication {
        source: String,
    },
    ChangeSetting {
        namespace: String,
        key: String,
        value: serde_json::Value,
    },
    WaitForCondition {
        condition: Condition,
        timeout_ms: u64,
    },
    AskUser {
        question: String,
    },
}

impl Action {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::FetchPublic { .. } => "fetch_public",
            Self::OpenApplication { .. } => "open_application",
            Self::SetApplicationControl { .. } => "set_application_control",
            Self::CloseApplication { .. } => "close_application",
            Self::ReadFile { .. } => "read_file",
            Self::ListDirectory { .. } => "list_directory",
            Self::WriteFile { .. } => "write_file",
            Self::MoveFile { .. } => "move_file",
            Self::DeleteFile { .. } => "delete_file",
            Self::CreateFolder { .. } => "create_folder",
            Self::ClickElement { .. } => "click_element",
            Self::TypeText { .. } => "type_text",
            Self::PressShortcut { .. } => "press_shortcut",
            Self::NavigateUrl { .. } => "navigate_url",
            Self::DownloadFile { .. } => "download_file",
            Self::UploadFile { .. } => "upload_file",
            Self::SendMessage { .. } => "send_message",
            Self::SubmitForm { .. } => "submit_form",
            Self::RunCommand { .. } => "run_command",
            Self::InstallApplication { .. } => "install_application",
            Self::ChangeSetting { .. } => "change_setting",
            Self::WaitForCondition { .. } => "wait_for_condition",
            Self::AskUser { .. } => "ask_user",
        }
    }

    pub fn domain(&self) -> ExecutionDomain {
        match self {
            Self::ClickElement { application, .. } | Self::TypeText { application, .. }
                if application.starts_with("browser:") =>
            {
                ExecutionDomain::Browser
            }
            Self::NavigateUrl { .. }
            | Self::DownloadFile { .. }
            | Self::UploadFile { .. }
            | Self::SubmitForm { .. } => ExecutionDomain::Browser,
            Self::RunCommand { .. } => ExecutionDomain::Sandbox,
            Self::InstallApplication { .. } => ExecutionDomain::Privileged,
            Self::SetApplicationControl { .. } => ExecutionDomain::Native,
            Self::AskUser { .. } => ExecutionDomain::UserInteraction,
            _ => ExecutionDomain::Native,
        }
    }

    pub fn reversible_hint(&self) -> bool {
        matches!(
            self,
            Self::WriteFile { .. }
                | Self::MoveFile { .. }
                | Self::DeleteFile { .. }
                | Self::CreateFolder { .. }
                | Self::ChangeSetting { .. }
        )
    }

    pub fn redacted_summary(&self) -> String {
        match self {
            Self::TypeText {
                application,
                sensitive,
                ..
            } => {
                format!("type text in {application} (sensitive={sensitive})")
            }
            Self::SendMessage {
                application,
                recipient,
                ..
            } => {
                format!("send message with {application} to {recipient}")
            }
            Self::WriteFile {
                path,
                content,
                overwrite,
            } => format!(
                "write {} bytes to {} (overwrite={overwrite})",
                content.len(),
                path.display()
            ),
            Self::RunCommand { program, args, .. } => {
                format!(
                    "run sandboxed program {program} with {} arguments",
                    args.len()
                )
            }
            Self::SetApplicationControl {
                application,
                control_id,
                value,
                ..
            } => format!(
                "set learned control {} in {} to {}",
                control_id.chars().take(12).collect::<String>(),
                application,
                serde_json::to_string(value).unwrap_or_else(|_| "[invalid]".into())
            ),
            _ => self.kind().replace('_', " "),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionProposal {
    pub id: Uuid,
    pub task_id: Uuid,
    pub action: Action,
    pub expected_outcome: ExpectedOutcome,
    pub target_resource: String,
    pub provenance: Provenance,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionNode {
    pub proposal: ActionProposal,
    #[serde(default)]
    pub depends_on: BTreeSet<Uuid>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionGraph {
    pub goal: String,
    pub nodes: Vec<ActionNode>,
}

impl ActionGraph {
    pub fn validate(&self, task_id: Uuid) -> Result<(), String> {
        if self.goal.trim().is_empty() {
            return Err("plan goal must not be empty".into());
        }
        if self.nodes.is_empty() {
            return Err("plan must contain at least one action".into());
        }
        if self.nodes.len() > 128 {
            return Err("plan exceeds 128 actions".into());
        }

        let ids: BTreeSet<_> = self.nodes.iter().map(|node| node.proposal.id).collect();
        if ids.len() != self.nodes.len() {
            return Err("plan contains duplicate action ids".into());
        }

        for node in &self.nodes {
            if node.proposal.task_id != task_id {
                return Err("action belongs to a different task".into());
            }
            if node.depends_on.contains(&node.proposal.id) {
                return Err("action cannot depend on itself".into());
            }
            if !node.depends_on.is_subset(&ids) {
                return Err("action references an unknown dependency".into());
            }
        }

        let mut resolved = BTreeSet::new();
        loop {
            let before = resolved.len();
            for node in &self.nodes {
                if node.depends_on.is_subset(&resolved) {
                    resolved.insert(node.proposal.id);
                }
            }
            if resolved.len() == self.nodes.len() {
                return Ok(());
            }
            if resolved.len() == before {
                return Err("plan contains a dependency cycle".into());
            }
        }
    }
}
