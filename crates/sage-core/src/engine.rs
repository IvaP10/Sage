use std::collections::HashMap;
use std::sync::Arc;

use chrono::{Duration as ChronoDuration, Utc};
use serde_json::json;
use tokio::sync::{Mutex, Notify, RwLock, oneshot};
use tokio::time::{Duration, timeout};
use uuid::Uuid;

use crate::capability::CapabilityBroker;
use crate::compiler::{ActionCompiler, ExecutorAvailability};
use crate::config::CoreConfig;
use crate::domain::{Action, ActionStatus, Task, TaskStatus};
use crate::error::{CoreError, CoreResult};
use crate::events::{CoreEvent, CoreEventKind, EventHub, StateSnapshot};
use crate::execution::bridge::{AdapterBridge, BrowserExecutor, PlatformObserver};
use crate::execution::{ExecutionBroker, ExecutionReceipt, NativeExecutor, RollbackOperation};
use crate::knowledge::{KnowledgeSnapshot, Message};
use crate::model::{
    ModelProvider, ModelTurn, ProviderSettings, ToolDescriptor, TurnContext,
    validate_provider_endpoint,
};
use crate::observation::Observer;
use crate::policy::{PolicyContext, PolicyDecision, PolicyEngine, RiskLevel};
use crate::redaction::redact_for_persistence;
use crate::resources::ResourceResolver;
use crate::secrets::{OsSecretStore, SecretBytes, SecretStore};
use crate::storage::LocalStore;
use crate::verification::Verifier;

const APPROVAL_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const QUESTION_TIMEOUT: Duration = Duration::from_secs(30 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalResolution {
    Approved {
        native_authentication_satisfied: bool,
    },
    Denied,
}

struct PendingApproval {
    task_id: Uuid,
    action_id: Uuid,
    digest: String,
    requires_native_authentication: bool,
    expires_at: chrono::DateTime<Utc>,
    record: crate::contracts::ApprovalRecord,
    sender: oneshot::Sender<ApprovalResolution>,
}

struct PendingQuestion {
    task_id: Uuid,
    action_id: Uuid,
    sender: oneshot::Sender<String>,
}

struct StepFailure {
    error: CoreError,
    recoverable: bool,
    observation: serde_json::Value,
}

pub struct SageCore {
    config: CoreConfig,
    model: Arc<dyn ModelProvider>,
    secret_store: Arc<dyn SecretStore>,
    store: LocalStore,
    events: EventHub,
    tasks: RwLock<HashMap<Uuid, Task>>,
    provider_settings: RwLock<HashMap<String, ProviderSettings>>,
    pending_approvals: Mutex<HashMap<Uuid, PendingApproval>>,
    pending_questions: Mutex<HashMap<Uuid, PendingQuestion>>,
    control_changed: Notify,
    policy: PolicyEngine,
    capabilities: CapabilityBroker,
    compiler: ActionCompiler,
    availability: ExecutorAvailability,
    broker: ExecutionBroker,
    resolver: ResourceResolver,
    files: Arc<crate::execution::files::FileBroker>,
    observer: Arc<dyn Observer>,
    verifier: Verifier,
    pub adapters: Arc<AdapterBridge>,
    submission_lock: Mutex<()>,
    storage_unlock: Mutex<()>,
    mutation_lane: Mutex<()>,
    scheduler_lane: Mutex<()>,
    read_lanes: tokio::sync::Semaphore,
}

impl std::fmt::Debug for SageCore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SageCore")
            .field("config", &self.config)
            .field("model", &self.model.descriptor().id)
            .finish_non_exhaustive()
    }
}

impl SageCore {
    pub fn new(config: CoreConfig, model: Arc<dyn ModelProvider>) -> CoreResult<Arc<Self>> {
        Self::new_with_secret_store(config, model, Arc::new(OsSecretStore))
    }

    pub fn new_with_secret_store(
        config: CoreConfig,
        model: Arc<dyn ModelProvider>,
        secret_store: Arc<dyn SecretStore>,
    ) -> CoreResult<Arc<Self>> {
        std::fs::create_dir_all(&config.data_dir)?;
        std::fs::create_dir_all(&config.recovery_dir)?;
        #[cfg(not(test))]
        let store = LocalStore::deferred(&config.database_path)?;
        #[cfg(test)]
        let store =
            LocalStore::open_encrypted(&config.database_path, &SecretBytes::new(vec![17; 32]))?;
        store.migrate_knowledge()?;
        store.migrate_workflows()?;
        let tasks = store
            .load_tasks(true)?
            .into_iter()
            .map(|task| (task.id, task))
            .collect();
        let mut provider_settings = HashMap::new();
        if let Some(settings) = store.load_setting::<ProviderSettings>("provider.reasoning")? {
            provider_settings.insert(settings.role.clone(), settings);
        }
        model.configure(
            provider_settings.get("reasoning").cloned(),
            Arc::clone(&secret_store),
        )?;
        let capabilities = CapabilityBroker::default();
        let mut broker = ExecutionBroker::new(capabilities.clone());
        let adapters = Arc::new(AdapterBridge::default());
        let files = Arc::new(crate::execution::files::FileBroker::default());
        broker.register(Arc::new(NativeExecutor::new(
            config.recovery_dir.clone(),
            adapters.clone(),
            files.clone(),
            store.clone(),
        )));
        broker.register(Arc::new(BrowserExecutor(adapters.clone())));
        // Installing a binary cannot enable a domain. VM execution and signed
        // privileged handlers need independently qualified feature manifests.
        let mut resolver = ResourceResolver::platform_default(config.data_dir.clone())?;
        resolver.protect_paths(&config.protected_paths)?;
        let availability = ExecutorAvailability {
            browser_dom: false,
            accessibility: false,
            keyboard: false,
            sandbox: false,
            privileged_helper: false,
            ..ExecutorAvailability::default()
        };

        Ok(Arc::new(Self {
            config,
            model,
            secret_store,
            store,
            events: EventHub::default(),
            tasks: RwLock::new(tasks),
            provider_settings: RwLock::new(provider_settings),
            pending_approvals: Mutex::new(HashMap::new()),
            pending_questions: Mutex::new(HashMap::new()),
            control_changed: Notify::new(),
            policy: PolicyEngine,
            capabilities,
            compiler: ActionCompiler,
            availability,
            broker,
            resolver,
            files,
            observer: Arc::new(PlatformObserver(adapters.clone())),
            verifier: Verifier,
            adapters,
            submission_lock: Mutex::new(()),
            storage_unlock: Mutex::new(()),
            mutation_lane: Mutex::new(()),
            scheduler_lane: Mutex::new(()),
            read_lanes: tokio::sync::Semaphore::new(2),
        }))
    }

    pub async fn unlock_storage(&self) -> CoreResult<()> {
        let _guard = self.storage_unlock.lock().await;
        if !self.store.unlock(self.secret_store.as_ref())? {
            return Ok(());
        }
        *self.tasks.write().await = self
            .store
            .load_tasks(true)?
            .into_iter()
            .map(|task| (task.id, task))
            .collect();
        if let Some(settings) = self
            .store
            .load_setting::<ProviderSettings>("provider.reasoning")?
        {
            self.model
                .configure(Some(settings.clone()), self.secret_store.clone())?;
            self.provider_settings
                .write()
                .await
                .insert(settings.role.clone(), settings);
        }
        Ok(())
    }

    pub fn events(&self) -> &EventHub {
        &self.events
    }

    pub async fn snapshot(&self, include_completed: bool) -> StateSnapshot {
        let mut tasks: Vec<_> = self
            .tasks
            .read()
            .await
            .values()
            .filter(|task| {
                include_completed
                    || !matches!(
                        task.status,
                        TaskStatus::Succeeded | TaskStatus::Failed | TaskStatus::Cancelled
                    )
            })
            .cloned()
            .collect();
        tasks.sort_by_key(|task| std::cmp::Reverse(task.updated_at));
        let provider_settings = self
            .provider_settings
            .read()
            .await
            .values()
            .cloned()
            .collect();
        StateSnapshot {
            tasks,
            provider_settings,
            core_version: env!("CARGO_PKG_VERSION").into(),
            storage_locked: self.store.is_locked(),
            pending_approvals: self
                .pending_approvals
                .lock()
                .await
                .values()
                .map(|p| p.record.clone())
                .collect(),
            protocol_version: sage_protocol::PROTOCOL_VERSION,
            knowledge: self.knowledge_snapshot(None).ok(),
        }
    }

    pub async fn submit_task(self: &Arc<Self>, request: impl Into<String>) -> CoreResult<Uuid> {
        self.submit_in_conversation(request.into(), None, false, None)
            .await
    }

    pub async fn submit_in_conversation(
        self: &Arc<Self>,
        request: String,
        conversation_id: Option<Uuid>,
        background: bool,
        graph: Option<crate::domain::ActionGraph>,
    ) -> CoreResult<Uuid> {
        self.submit_scoped(request, conversation_id, background, graph, Vec::new())
            .await
    }

    pub async fn submit_scoped(
        self: &Arc<Self>,
        request: String,
        conversation_id: Option<Uuid>,
        background: bool,
        graph: Option<crate::domain::ActionGraph>,
        resources: Vec<crate::contracts::ResourceScope>,
    ) -> CoreResult<Uuid> {
        if background {
            return Err(CoreError::PermissionRequired(
                "Background work must originate from an exact durable schedule authorization"
                    .into(),
            ));
        }
        self.submit_run(request, conversation_id, graph, resources, None, None)
            .await
    }

    async fn submit_run(
        self: &Arc<Self>,
        request: String,
        conversation_id: Option<Uuid>,
        graph: Option<crate::domain::ActionGraph>,
        mut resources: Vec<crate::contracts::ResourceScope>,
        background_expiry: Option<chrono::DateTime<Utc>>,
        continuation: Option<Task>,
    ) -> CoreResult<Uuid> {
        self.unlock_storage().await?;
        if resources.len() > 32 {
            return Err(CoreError::InvalidAction("Too many task folders".into()));
        }
        for resource in &mut resources {
            resource.root = self.resolver.validate_scope_root(&resource.root)?;
            if resource.effects.is_empty()
                || resource.effects.iter().any(|e| {
                    !matches!(
                        e,
                        crate::contracts::Effect::Read | crate::contracts::Effect::Create
                    )
                })
            {
                return Err(CoreError::PermissionRequired(
                    "Only reading and creation can be scoped to a folder".into(),
                ));
            }
        }
        let _submission = self.submission_lock.lock().await;
        let request = redact_for_persistence(request.trim());
        if request.is_empty() {
            return Err(CoreError::InvalidAction(
                "task request must not be empty".into(),
            ));
        }
        if request.len() > 64 * 1024 {
            return Err(CoreError::InvalidAction(
                "task request exceeds the 64-kilobyte limit".into(),
            ));
        }
        if let Some(id) = conversation_id
            && self.tasks.read().await.values().any(|t| {
                t.conversation_id == Some(id)
                    && !matches!(
                        t.status,
                        TaskStatus::Succeeded
                            | TaskStatus::Failed
                            | TaskStatus::Cancelled
                            | TaskStatus::Interrupted
                    )
            })
        {
            return Err(CoreError::InvalidAction(
                "This conversation already has unfinished work.".into(),
            ));
        }
        let conversation = self.store.ensure_conversation(conversation_id, &request)?;
        let mut task = Task::new(request.clone());
        let mut contract = crate::contracts::RunContract::local(task.id);
        contract.resources = resources;
        if let Some(expiry) = background_expiry {
            contract.expires_at = contract.expires_at.min(expiry);
            contract.validate(task.id)?;
        }
        task.contract = Some(contract);
        task.conversation_id = Some(conversation.id);
        task.message_id = Some(Uuid::new_v4());
        task.background = background_expiry.is_some();
        if let Some(previous) = continuation {
            task.continuation_of = Some(previous.id);
            // Copy evidence, never old actions, approvals or capabilities. The
            // explicit native Continue action authorizes this same-scope handoff.
            task.tool_results = previous
                .tool_results
                .into_iter()
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            for result in &mut task.tool_results {
                result
                    .label
                    .source_ids
                    .insert(format!("run:{}", previous.id));
                result.label.scope_id = task.id;
            }
        }
        if let Some(graph) = graph {
            task.install_plan(crate::workflows::instantiate(&graph, task.id)?)
                .map_err(CoreError::InvalidAction)?;
        }
        let task_id = task.id;
        self.store.save_task(&task)?;
        self.store.append_message(&Message {
            id: task.message_id.unwrap(),
            conversation_id: conversation.id,
            task_id: Some(task_id),
            role: "user".into(),
            content: request,
            provenance: crate::domain::Provenance::user(),
            created_at: task.created_at,
        })?;
        self.store.link_task(&task)?;
        self.tasks.write().await.insert(task_id, task);
        self.publish(CoreEvent::new(Some(task_id), CoreEventKind::TaskStarted))?;

        let core = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = core.run_task(task_id).await {
                let _ = core.fail_task(task_id, error.to_string()).await;
            }
        });
        Ok(task_id)
    }

    pub async fn control_task(
        self: &Arc<Self>,
        task_id: Uuid,
        status: TaskStatus,
    ) -> CoreResult<()> {
        if status == TaskStatus::Running
            && self.get_task(task_id).await?.status == TaskStatus::Interrupted
        {
            return self.resume_interrupted(task_id).await;
        }
        self.update_task(task_id, |task| {
            match status {
                TaskStatus::Paused if task.status == TaskStatus::Running => {
                    task.status = TaskStatus::Paused;
                }
                TaskStatus::Running if task.status == TaskStatus::Paused => {
                    task.status = TaskStatus::Running;
                }
                TaskStatus::Cancelled
                    if !matches!(task.status, TaskStatus::Succeeded | TaskStatus::Failed) =>
                {
                    task.status = TaskStatus::Cancelled;
                }
                _ => {
                    return Err(CoreError::InvalidAction(format!(
                        "cannot change task from {:?} to {:?}",
                        task.status, status
                    )));
                }
            }
            task.touch();
            Ok(())
        })
        .await?;
        if status == TaskStatus::Cancelled {
            self.capabilities.revoke_task(task_id).await;
            self.pending_approvals
                .lock()
                .await
                .retain(|_, v| v.task_id != task_id);
            self.pending_questions
                .lock()
                .await
                .retain(|_, v| v.task_id != task_id);
        }
        self.control_changed.notify_waiters();
        Ok(())
    }

    pub async fn resolve_approval(
        &self,
        approval_id: Uuid,
        task_id: Uuid,
        action_id: Uuid,
        digest: &str,
        resolution: ApprovalResolution,
    ) -> CoreResult<()> {
        let mut pending = self.pending_approvals.lock().await;
        let matches = pending.get(&approval_id).is_some_and(|approval| {
            approval.expires_at > Utc::now()
                && approval.task_id == task_id
                && approval.action_id == action_id
                && approval.digest == digest
                && (!approval.requires_native_authentication
                    || matches!(
                        resolution,
                        ApprovalResolution::Approved {
                            native_authentication_satisfied: true
                        }
                    ))
        });
        if !matches {
            return Err(CoreError::ApprovalRejected(
                "approval is stale, mismatched, or missing required native authentication".into(),
            ));
        }
        let approval = pending
            .remove(&approval_id)
            .ok_or_else(|| CoreError::ApprovalRejected("approval no longer exists".into()))?;
        approval
            .sender
            .send(resolution)
            .map_err(|_| CoreError::ApprovalRejected("task no longer accepts this approval".into()))
    }

    pub async fn answer_question(
        &self,
        question_id: Uuid,
        task_id: Uuid,
        action_id: Uuid,
        answer: String,
    ) -> CoreResult<()> {
        if answer.trim().is_empty() || answer.len() > 16 * 1024 {
            return Err(CoreError::InvalidAction(
                "answer must contain between 1 and 16384 characters".into(),
            ));
        }
        let mut pending = self.pending_questions.lock().await;
        let matches = pending
            .get(&question_id)
            .is_some_and(|question| question.task_id == task_id && question.action_id == action_id);
        if !matches {
            return Err(CoreError::InvalidAction(
                "question is stale or belongs to another action".into(),
            ));
        }
        let question = pending
            .remove(&question_id)
            .ok_or_else(|| CoreError::InvalidAction("question no longer exists".into()))?;
        question
            .sender
            .send(redact_for_persistence(&answer))
            .map_err(|_| CoreError::InvalidAction("task no longer accepts this answer".into()))
    }

    pub async fn undo_last_action(&self, task_id: Uuid) -> CoreResult<()> {
        let _mutation = self.mutation_lane.lock().await;
        if !matches!(
            self.get_task(task_id).await?.status,
            TaskStatus::Succeeded
                | TaskStatus::Failed
                | TaskStatus::Cancelled
                | TaskStatus::Interrupted
        ) {
            return Err(CoreError::PermissionRequired(
                "Stop the task before undoing its changes".into(),
            ));
        }
        let plan = self
            .store
            .latest_rollback(task_id)?
            .ok_or_else(|| CoreError::InvalidAction("no reversible action is available".into()))?;
        if plan.expires_at <= Utc::now() {
            return Err(CoreError::InvalidAction("rollback metadata expired".into()));
        }
        if plan.operations.len() != 1 {
            return Err(CoreError::PermissionRequired(
                "This recovery plan requires manual review".into(),
            ));
        }
        // Journal Undo dispatch before its effects; a crash during Undo must not
        // cause the same destructive recovery operation to run a second time.
        self.store.append_audit(
            Some(task_id),
            Some(plan.action_id),
            "undo_dispatch_intent",
            &json!({"operation_count":plan.operations.len()}),
        )?;
        self.store.checkpoint_audit(self.secret_store.as_ref())?;
        for operation in &plan.operations {
            match operation {
                RollbackOperation::RestoreArtifact {
                    artifact_id,
                    destination,
                    expected_sha256,
                } => {
                    let path = std::path::Path::new(destination);
                    use sha2::{Digest, Sha256};
                    let pinned = crate::execution::files::PinnedPath::open(path)?;
                    if format!("{:x}", Sha256::digest(pinned.read(16 * 1024 * 1024)?))
                        != *expected_sha256
                    {
                        return Err(CoreError::ApprovalRejected(
                            "Undo refused: file changed since the action".into(),
                        ));
                    }
                    let bytes = self.store.read_artifact(*artifact_id)?;
                    self.store.consume_rollback(plan.action_id)?;
                    pinned.write(&bytes, true)?;
                }
                RollbackOperation::RemoveCreatedFile {
                    path,
                    expected_sha256,
                } => {
                    use sha2::{Digest, Sha256};
                    let pinned =
                        crate::execution::files::PinnedPath::open(std::path::Path::new(path))?;
                    if format!("{:x}", Sha256::digest(pinned.read(16 * 1024 * 1024)?))
                        != *expected_sha256
                    {
                        return Err(CoreError::ApprovalRejected(
                            "Undo refused: file changed".into(),
                        ));
                    }
                    self.store.consume_rollback(plan.action_id)?;
                    pinned.remove_verified(expected_sha256)?;
                }
                RollbackOperation::RemoveCreatedFolder { path, identity } => {
                    let pinned =
                        crate::execution::files::PinnedPath::open(std::path::Path::new(path))?;
                    if pinned.current_identity()? != *identity {
                        return Err(CoreError::ApprovalRejected(
                            "Undo refused: folder changed".into(),
                        ));
                    }
                    self.store.consume_rollback(plan.action_id)?;
                    pinned.remove_empty_verified(identity)?;
                }
                _ => return Err(CoreError::PermissionRequired(
                    "Pre-v2 recovery needs manual review; no legacy file mutation was performed"
                        .into(),
                )),
            }
        }
        let rollback_available = self.store.latest_rollback(task_id)?.is_some();
        self.update_task(task_id, |task| {
            task.rollback_available = rollback_available;
            task.touch();
            Ok(())
        })
        .await?;
        self.store.append_audit(
            Some(task_id),
            Some(plan.action_id),
            "rollback_completed",
            &json!({ "operation_count": plan.operations.len() }),
        )?;
        Ok(())
    }

    pub fn update_permission(&self, permission: &str, granted: bool) -> CoreResult<()> {
        let permission = permission.trim();
        if permission.is_empty() || permission.len() > 128 {
            return Err(CoreError::InvalidAction(
                "permission name must contain between 1 and 128 characters".into(),
            ));
        }
        self.store.set_permission(permission, granted)?;
        self.publish(CoreEvent::new(
            None,
            CoreEventKind::PermissionChanged {
                permission: permission.into(),
                granted,
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn save_provider_settings(
        &self,
        role: String,
        provider: String,
        model: String,
        endpoint: String,
        api_key: String,
        remove_saved_key: bool,
        native_authentication_satisfied: bool,
    ) -> CoreResult<()> {
        let role = role.trim().to_ascii_lowercase();
        let provider = provider.trim().to_ascii_lowercase();
        let model = model.trim().to_string();
        let endpoint = endpoint.trim().to_string();
        let api_key = api_key.trim().to_string();

        if role != "reasoning" {
            return Err(CoreError::InvalidAction(
                "only the reasoning provider role is currently configurable".into(),
            ));
        }
        if !matches!(provider.as_str(), "openai" | "openai-compatible") {
            return Err(CoreError::InvalidAction("unsupported provider".into()));
        }
        if model.is_empty() || model.len() > 256 {
            return Err(CoreError::InvalidAction(
                "model name must contain between 1 and 256 characters".into(),
            ));
        }
        if endpoint.len() > 2_048 {
            return Err(CoreError::InvalidAction(
                "provider endpoint exceeds 2,048 characters".into(),
            ));
        }
        if provider == "openai-compatible" && endpoint.is_empty() {
            return Err(CoreError::InvalidAction(
                "an OpenAI-compatible endpoint is required".into(),
            ));
        }
        if !endpoint.is_empty() && validate_provider_endpoint(&endpoint).is_err() {
            return Err(CoreError::InvalidAction(
                "provider endpoints must use HTTPS, or HTTP only for localhost, 127.0.0.1, or [::1]".into(),
            ));
        }
        if api_key.len() > 8_192 {
            return Err(CoreError::InvalidAction(
                "provider credential exceeds 8,192 characters".into(),
            ));
        }
        if remove_saved_key && !api_key.is_empty() {
            return Err(CoreError::InvalidAction(
                "a credential cannot be saved and removed in the same update".into(),
            ));
        }

        let mut settings = ProviderSettings {
            role: role.clone(),
            provider,
            model,
            endpoint,
            has_api_key: false,
        };
        let previous = self.provider_settings.read().await.get(&role).cloned();
        let mutates_keychain = remove_saved_key || !api_key.is_empty();
        if mutates_keychain && !native_authentication_satisfied {
            return Err(CoreError::ApprovalRejected(
                "native authentication is required before changing a Keychain credential".into(),
            ));
        }

        if remove_saved_key {
            let account = previous
                .as_ref()
                .map(ProviderSettings::credential_account)
                .unwrap_or_else(|| settings.credential_account());
            self.secret_store.delete(&account)?;
        } else if !api_key.is_empty() {
            self.secret_store.set(
                &settings.credential_account(),
                &SecretBytes::new(api_key.into_bytes()),
            )?;
            settings.has_api_key = true;
        } else if let Some(previous) = &previous {
            settings.has_api_key = previous.credential_account() == settings.credential_account()
                && previous.has_api_key;
        }

        self.store
            .save_setting(&format!("provider.{role}"), &settings)?;
        self.store.append_audit(
            None,
            None,
            "provider_settings_saved",
            &json!({
                "role": settings.role,
                "provider": settings.provider,
                "model": settings.model,
                "endpoint_configured": !settings.endpoint.is_empty(),
                "credential_changed": mutates_keychain,
                "has_api_key": settings.has_api_key,
            }),
        )?;
        self.provider_settings.write().await.insert(role, settings);
        self.model.configure(
            self.provider_settings
                .read()
                .await
                .get("reasoning")
                .cloned(),
            Arc::clone(&self.secret_store),
        )?;
        Ok(())
    }

    pub async fn test_provider_connection(
        &self,
        role: String,
        provider: String,
        model: String,
        endpoint: String,
        api_key: String,
    ) -> CoreResult<String> {
        let role = role.trim().to_ascii_lowercase();
        let provider = provider.trim().to_ascii_lowercase();
        let model = model.trim().to_string();
        let endpoint = endpoint.trim().to_string();
        let api_key = api_key.trim().to_string();
        let previous = self.provider_settings.read().await.get(&role).cloned();
        let mut settings = ProviderSettings {
            role,
            provider,
            model,
            endpoint,
            has_api_key: !api_key.is_empty(),
        };
        settings.has_api_key |= previous.as_ref().is_some_and(|prior| {
            prior.has_api_key && prior.credential_account() == settings.credential_account()
        });
        if settings.provider == "openai-compatible" && settings.endpoint.is_empty() {
            return Err(CoreError::InvalidAction(
                "an OpenAI-compatible endpoint is required".into(),
            ));
        }
        if !settings.endpoint.is_empty() {
            validate_provider_endpoint(&settings.endpoint)?;
        }
        let key = if api_key.is_empty() {
            None
        } else {
            Some(SecretBytes::new(api_key.into_bytes()))
        };
        self.model.test_connection(settings, key).await
    }

    async fn run_task(self: &Arc<Self>, task_id: Uuid) -> CoreResult<()> {
        let initial = self.get_task(task_id).await?;
        let workflow = initial.workflow_run;
        if !workflow {
            if initial
                .request
                .trim()
                .to_ascii_lowercase()
                .starts_with("forget ")
            {
                self.cancel_context_readers(Some(task_id)).await?;
            }
            if let Some(answer) = self
                .store
                .memory_command(&initial.request, initial.message_id.unwrap_or(initial.id))?
            {
                return self.finish_answer(task_id, answer).await;
            }
        }
        let mut repairs = 0;
        loop {
            self.wait_until_runnable(task_id).await?;
            let task = self.get_task(task_id).await?;
            let contract = task.contract.as_ref().ok_or_else(|| {
                CoreError::PermissionRequired("This pre-v2 task needs a new authorized run".into())
            })?;
            contract.validate(task_id)?;
            if let Some(action_id) = task.ready_actions().into_iter().next() {
                match self.execute_action(task_id, action_id).await {
                    Ok(()) => {
                        repairs = 0;
                        continue;
                    }
                    Err(failure) if failure.recoverable && repairs < contract.max_repairs => {
                        repairs += 1;
                        self.update_task(task_id, |task| {
                            if let Some(state) = task.actions.get_mut(&action_id) {
                                state.status = ActionStatus::Skipped;
                            }
                            task.tool_results.push(crate::contracts::ToolResult {
                                action_id,
                                tool: "planning_error".into(),
                                verdict: crate::contracts::Verdict::Failed,
                                summary: failure.error.to_string(),
                                output: failure.observation,
                                label: crate::contracts::DataLabel::private(
                                    task_id,
                                    action_id.to_string(),
                                ),
                                observed_at: Utc::now(),
                            });
                            Ok(())
                        })
                        .await?;
                    }
                    Err(failure) => return Err(failure.error),
                }
            } else if workflow && task.is_complete() {
                return self.succeed_task(task_id).await;
            }
            self.set_task_status(task_id, TaskStatus::Planning, "Preparing the next response")
                .await?;
            let task = self.get_task(task_id).await?;
            // Observing another app is a separate action, never implicit context.
            let planning = crate::context::build_context(
                &self.store,
                &task,
                if task.actions.len() >= contract.max_steps as usize {
                    Vec::new()
                } else {
                    self.available_tools().await
                },
                Vec::new(),
            )?;
            let destination = self.model.data_destination()?;
            let context = TurnContext {
                planning,
                results: task
                    .tool_results
                    .iter()
                    .rev()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect(),
                destination: destination.clone(),
            };
            if let Some(destination) = &destination {
                self.approve_data_release(&task, destination, &context)
                    .await?;
            }
            self.wait_until_runnable(task_id).await?;
            let (updates, mut prefixes) = tokio::sync::mpsc::channel(8);
            let response = self.model.next_turn_stream(context, updates);
            tokio::pin!(response);
            let turn = loop {
                let changed = self.control_changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if self.get_task(task_id).await?.status == TaskStatus::Cancelled {
                    return Err(CoreError::Cancelled);
                }
                tokio::select! {
                    result=&mut response => break result?, _=&mut changed => {},
                    Some(prefix)=prefixes.recv()=>self.events.publish(CoreEvent::new(Some(task_id),CoreEventKind::ModelResponse{text:redact_for_persistence(&prefix),finished:false})),
                }
            };
            self.wait_until_runnable(task_id).await?;
            match turn {
                ModelTurn::Answer(answer) => return self.finish_answer(task_id, answer).await,
                ModelTurn::Actions(graph) => {
                    let count = graph.nodes.len();
                    if task.actions.len().saturating_add(count) > contract.max_steps as usize {
                        self.update_task(task_id, |task| {
                            task.budget_exhausted = true;
                            Ok(())
                        })
                        .await?;
                        self.set_task_status(task_id, TaskStatus::Interrupted,
                            "The tool budget is exhausted. Review the completed work; Continue starts a new run with up to 32 steps and the same folders. Expanded access still requires approval.").await?;
                        self.capabilities.revoke_task(task_id).await;
                        return Ok(());
                    }
                    self.update_task(task_id, |task| {
                        task.append_plan(graph).map_err(CoreError::InvalidAction)
                    })
                    .await?;
                    self.publish(CoreEvent::new(
                        Some(task_id),
                        CoreEventKind::PlanGenerated {
                            action_count: count,
                        },
                    ))?;
                }
            }
        }
    }

    async fn approve_data_release(
        &self,
        task: &Task,
        destination: &str,
        context: &TurnContext,
    ) -> CoreResult<()> {
        use sha2::{Digest, Sha256};
        let payload = serde_json::to_vec(context)?;
        let sources = context
            .planning
            .untrusted_context
            .iter()
            .filter(|c| !c.content.is_empty())
            .map(|c| c.source.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let explanation = format!(
            "Send this request and selected context to {destination}? Sources: {sources}; {} verified tool results; {} bytes. This endpoint may process data outside this device.\n\nExact data selected for release:\n{}",
            context.results.len(),
            payload.len(),
            String::from_utf8_lossy(&payload)
        );
        let proposal = crate::domain::ActionProposal {
            id: Uuid::new_v4(),
            task_id: task.id,
            action: Action::AskUser {
                question: explanation.clone(),
            },
            expected_outcome: crate::domain::ExpectedOutcome::UserAnswered,
            target_resource: destination.into(),
            provenance: crate::domain::Provenance::user(),
            metadata: std::collections::BTreeMap::from([(
                "payload_sha256".into(),
                format!("{:x}", Sha256::digest(&payload)),
            )]),
        };
        self.await_approval(
            &proposal,
            RiskLevel::Consequential,
            explanation,
            crate::policy::approval_digest(&proposal)?,
        )
        .await
    }

    async fn finish_answer(&self, task_id: Uuid, answer: String) -> CoreResult<()> {
        let answer = redact_for_persistence(&answer);
        self.wait_until_runnable(task_id).await?;
        self.update_task(task_id, |task| {
            if task.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            task.status = TaskStatus::Succeeded;
            task.final_outcome = Some(answer.clone());
            task.touch();
            Ok(())
        })
        .await?;
        self.capabilities.revoke_task(task_id).await;
        self.store.record_outcome(&self.get_task(task_id).await?)?;
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::ModelResponse {
                text: answer.clone(),
                finished: true,
            },
        ))?;
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::TaskCompleted { outcome: answer },
        ))
    }

    async fn execute_action(&self, task_id: Uuid, action_id: Uuid) -> Result<(), StepFailure> {
        let result = self.execute_action_inner(task_id, action_id).await;
        self.files.discard(action_id);
        let dispatched = self
            .get_task(task_id)
            .await
            .ok()
            .and_then(|t| {
                t.actions.get(&action_id).map(|s| {
                    matches!(
                        s.status,
                        ActionStatus::Running | ActionStatus::Verifying | ActionStatus::Succeeded
                    )
                })
            })
            .unwrap_or(true);
        if let Err(error) = &result {
            let _ = self
                .store
                .journal_interrupted(task_id, action_id, dispatched);
            if dispatched {
                // Dispatch might have changed the world even if the response was
                // lost. Never turn this state into an automatically retryable failure.
                let _ = self.update_task(task_id, |task| {
                    if task.status != TaskStatus::Cancelled { task.status = TaskStatus::Interrupted; }
                    if let Some(state) = task.actions.get_mut(&action_id) {
                        state.status = ActionStatus::Uncertain;
                        state.error = Some(error.to_string());
                    }
                    task.tool_results.push(crate::contracts::ToolResult {
                        action_id, tool: "interrupted_dispatch".into(), verdict: crate::contracts::Verdict::Uncertain,
                        summary: "An action was dispatched but its result could not be confirmed. Review the actual state before continuing.".into(),
                        output: json!({"error":error.to_string()}),
                        label: crate::contracts::DataLabel::private(task_id,action_id.to_string()), observed_at: Utc::now(),
                    });
                    Ok(())
                }).await;
            } else {
                let _ = self
                    .mark_action_failed(task_id, action_id, error.to_string())
                    .await;
            }
            let _ = self.publish(CoreEvent::new(
                Some(task_id),
                CoreEventKind::ActionFailed {
                    action_id,
                    error: error.to_string(),
                },
            ));
        }
        result.map_err(|error| StepFailure {
            recoverable: !dispatched
                && !matches!(
                    error,
                    CoreError::ApprovalRejected(_)
                        | CoreError::PolicyDenied(_)
                        | CoreError::Cancelled
                ),
            observation: json!({ "error": error.to_string() }),
            error,
        })
    }

    async fn execute_action_inner(&self, task_id: Uuid, action_id: Uuid) -> CoreResult<()> {
        self.update_action(task_id, action_id, |state| {
            state.status = ActionStatus::Compiling;
            state.attempts += 1;
            Ok(())
        })
        .await?;
        let task = self.get_task(task_id).await?;
        let raw = task
            .actions
            .get(&action_id)
            .ok_or_else(|| CoreError::InvalidAction("action disappeared".into()))?
            .proposal
            .clone();
        let mut proposal = self.resolver.prepare_proposal(&raw)?;
        crate::verification::bind_required_outcome(&mut proposal)?;
        self.compiler
            .compile(proposal.clone(), &self.current_availability().await)?;
        if proposal.action.domain() == crate::domain::ExecutionDomain::Browser {
            let target: crate::browser_target::BrowserTarget = serde_json::from_value(
                self.adapters
                    .request("browser", "binding", json!({}))
                    .await?,
            )?;
            target.validate()?;
            proposal
                .metadata
                .insert("browser_target".into(), serde_json::to_string(&target)?);
        }
        self.files.prepare(&mut proposal)?;
        let prepared = crate::contracts::PreparedAction::new(
            &proposal,
            task.tool_results
                .iter()
                .map(|result| result.action_id)
                .collect(),
        )?;
        self.store.journal_prepared(&prepared)?;
        self.update_action(task_id, action_id, |state| {
            state.proposal = proposal.clone();
            Ok(())
        })
        .await?;
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::ActionProposed {
                action_id,
                summary: proposal.action.redacted_summary(),
            },
        ))?;

        let decision = self.policy.evaluate(
            &proposal,
            &PolicyContext {
                task_request: task.request.clone(),
                has_fresh_native_authentication: false,
                is_recovery_attempt: task.recovery_attempt,
            },
        )?;
        let scoped = task
            .contract
            .as_ref()
            .is_some_and(|scope| scope.covers(&proposal.action));
        let mut approved = false;
        match decision {
            PolicyDecision::Deny { reason, .. } => {
                self.publish(CoreEvent::new(
                    Some(task_id),
                    CoreEventKind::PolicyDenied {
                        action_id,
                        reason: reason.clone(),
                    },
                ))?;
                return Err(CoreError::PolicyDenied(reason));
            }
            PolicyDecision::RequireApproval {
                risk,
                explanation,
                digest,
            } => {
                if !scoped || risk >= RiskLevel::Destructive {
                    let explanation = format!(
                        "{explanation}\n\n{}",
                        crate::policy::action_preview(&proposal.action)?
                    );
                    self.await_approval(&proposal, risk, explanation, digest)
                        .await?;
                    approved = true;
                }
            }
            PolicyDecision::Allow { .. } => {}
        }

        let contract = task
            .contract
            .as_ref()
            .ok_or_else(|| CoreError::PermissionRequired("Missing task scope".into()))?;
        contract.validate(task_id)?;
        crate::authorization::authorize(
            scoped || matches!(proposal.action, Action::AskUser { .. }),
            approved,
            crate::policy::classify(&proposal.action) >= RiskLevel::Destructive,
            false,
            false,
        )?;

        if let Action::AskUser { question } = &proposal.action {
            self.store.journal_dispatch(&prepared, None)?;
            let receipt = self.await_question(&proposal, question.clone()).await?;
            return self
                .observe_and_verify(task_id, action_id, &proposal, &receipt)
                .await;
        }

        // Approval may have waited several minutes. Cancellation and resources
        // must be checked again before issuing a new single-use grant.
        self.wait_until_runnable(task_id).await?;
        let _mutation_guard = if !matches!(
            proposal.action,
            Action::ReadFile { .. } | Action::WaitForCondition { .. } | Action::FetchPublic { .. }
        ) {
            Some(self.mutation_lane.lock().await)
        } else {
            None
        };
        let _read_guard = if _mutation_guard.is_none() {
            Some(
                self.read_lanes
                    .acquire()
                    .await
                    .map_err(|_| CoreError::Cancelled)?,
            )
        } else {
            None
        };
        self.wait_until_runnable(task_id).await?;
        contract.validate(task_id)?;
        let refreshed = self.resolver.prepare_proposal(&proposal)?;
        if refreshed != proposal {
            return Err(CoreError::ApprovalRejected(
                "Resource changed while awaiting approval.".into(),
            ));
        }
        let availability = self.current_availability().await;
        let compiled = self.compiler.compile(proposal.clone(), &availability)?;
        let implementation = self.broker.select(&compiled)?.clone();
        let grant = self
            .capabilities
            .issue(&proposal, implementation.executor)
            .await?;
        let domain = if implementation.executor == crate::domain::ExecutionDomain::Browser {
            "browser"
        } else {
            "native"
        };
        let grant = if let Some(session) = self.adapters.session_id(domain).await {
            self.capabilities.bind_worker(grant.id, session).await?
        } else {
            grant
        };
        self.store.append_audit(
            Some(task_id),
            Some(action_id),
            "capability_issued",
            &json!({
                "capability_id": grant.id,
                "domain": format!("{:?}", grant.domain),
                "expires_at": grant.expires_at,
                "remaining_uses": grant.remaining_uses,
            }),
        )?;
        self.update_action(task_id, action_id, |state| {
            state.status = ActionStatus::Running;
            Ok(())
        })
        .await?;
        self.store
            .update_working_memory(&self.get_task(task_id).await?)?;
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::ActionStarted {
                action_id,
                implementation: implementation.operation.clone(),
            },
        ))?;
        // The Running state is persisted before dispatch. A crash in this
        // interval is reconciled as potentially executed, never blindly retried.
        self.store.journal_dispatch(&prepared, Some(&grant))?;
        self.store.checkpoint_audit(self.secret_store.as_ref())?;
        let execution = self.broker.execute(&compiled, &implementation, &grant);
        tokio::pin!(execution);
        let receipt = loop {
            let changed = self.control_changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.get_task(task_id).await?.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            tokio::select! { result=&mut execution => break result?, _=&mut changed => {} }
        };
        if let Some(rollback) = &receipt.rollback {
            self.store.save_rollback(task_id, rollback)?;
            self.update_task(task_id, |task| {
                task.rollback_available = true;
                task.touch();
                Ok(())
            })
            .await?;
        }
        self.observe_and_verify(task_id, action_id, &proposal, &receipt)
            .await
    }

    async fn observe_and_verify(
        &self,
        task_id: Uuid,
        action_id: Uuid,
        proposal: &crate::domain::ActionProposal,
        receipt: &ExecutionReceipt,
    ) -> CoreResult<()> {
        self.update_action(task_id, action_id, |state| {
            state.status = ActionStatus::Verifying;
            Ok(())
        })
        .await?;
        let observation = self.observer.observe(proposal, receipt).await?;
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::ObservationReceived {
                action_id,
                summary: observation.summary.clone(),
            },
        ))?;
        if let Err(error) = self
            .verifier
            .verify(&proposal.expected_outcome, &observation)
        {
            self.publish(CoreEvent::new(
                Some(task_id),
                CoreEventKind::VerificationFailed {
                    action_id,
                    reason: error.to_string(),
                },
            ))?;
            return Err(error);
        }
        self.store
            .journal_verification(&crate::contracts::VerificationRecord {
                run_id: task_id,
                action_id,
                target: proposal.target_resource.clone(),
                action_digest: crate::policy::approval_digest(proposal)?,
                expected: proposal.expected_outcome.clone(),
                observed_at: observation.observed_at,
                verdict: crate::contracts::Verdict::Confirmed,
                evidence: observation.evidence.clone(),
            })?;
        let mut output = receipt.transient_data.clone();
        if let Some(text) = output.get("text").and_then(serde_json::Value::as_str) {
            let full = text.to_string();
            let artifact = self.store.save_artifact(task_id, full.as_bytes())?;
            output["text"] = json!(redact_for_persistence(
                &full.chars().take(8000).collect::<String>()
            ));
            output["truncated"] = json!(full.chars().count() > 8000);
            output["artifact_ref"] = json!(artifact);
        }
        if let Some(encoded) = output
            .get("bytes_base64")
            .and_then(serde_json::Value::as_str)
        {
            use base64::Engine;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|_| CoreError::VerificationFailed("Invalid file read result".into()))?;
            use sha2::{Digest, Sha256};
            let artifact = self.store.save_artifact(task_id, &bytes)?;
            output = json!({"text":redact_for_persistence(&String::from_utf8_lossy(&bytes).chars().take(8000).collect::<String>()),"bytes":bytes.len(),"truncated":bytes.len()>8000,"sha256":format!("{:x}",Sha256::digest(&bytes)),"artifact_ref":artifact});
        }
        self.update_task(task_id, |task| {
            task.tool_results.push(crate::contracts::ToolResult {
                action_id,
                tool: proposal.action.kind().into(),
                verdict: crate::contracts::Verdict::Confirmed,
                summary: receipt.summary.clone(),
                output,
                label: crate::contracts::DataLabel::private(task_id, action_id.to_string()),
                observed_at: observation.observed_at,
            });
            Ok(())
        })
        .await?;
        self.update_action(task_id, action_id, |state| {
            state.status = ActionStatus::Succeeded;
            state.summary = Some(receipt.summary.clone());
            Ok(())
        })
        .await?;
        self.store.append_audit(
            Some(task_id),
            Some(action_id),
            "action_verified",
            &json!({
                "action": proposal.action.kind(),
                "resource": proposal.target_resource,
                "summary": receipt.summary,
                "observation": observation.summary,
            }),
        )?;
        self.store.checkpoint_audit(self.secret_store.as_ref())?;
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::ActionSucceeded {
                action_id,
                summary: receipt.summary.clone(),
            },
        ))?;
        Ok(())
    }

    async fn await_approval(
        &self,
        proposal: &crate::domain::ActionProposal,
        risk: RiskLevel,
        explanation: String,
        digest: String,
    ) -> CoreResult<()> {
        let approval_id = Uuid::new_v4();
        let expires_at = Utc::now() + ChronoDuration::seconds(APPROVAL_TIMEOUT.as_secs() as i64);
        let requires_native_authentication = risk >= RiskLevel::Privileged;
        let (sender, receiver) = oneshot::channel();
        self.pending_approvals.lock().await.insert(
            approval_id,
            PendingApproval {
                task_id: proposal.task_id,
                action_id: proposal.id,
                digest: digest.clone(),
                requires_native_authentication,
                expires_at,
                record: crate::contracts::ApprovalRecord {
                    approval_id,
                    task_id: proposal.task_id,
                    action_id: proposal.id,
                    digest: digest.clone(),
                    explanation: explanation.clone(),
                    resource: proposal.target_resource.clone(),
                    risk,
                    expires_at,
                    reversible: proposal.action.reversible_hint(),
                    requires_native_authentication,
                },
                sender,
            },
        );
        self.update_task(proposal.task_id, |task| {
            if task.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            task.status = TaskStatus::WaitingForApproval;
            if let Some(action) = task.actions.get_mut(&proposal.id) {
                action.status = ActionStatus::WaitingForApproval;
            }
            task.touch();
            Ok(())
        })
        .await?;
        self.publish(CoreEvent::new(
            Some(proposal.task_id),
            CoreEventKind::ApprovalRequested {
                approval_id,
                action_id: proposal.id,
                digest,
                explanation,
                resource: proposal.target_resource.clone(),
                risk,
                expires_at,
                reversible: proposal.action.reversible_hint(),
                requires_native_authentication,
            },
        ))?;
        self.store.save_setting(
            &format!("pending-approval.{approval_id}"),
            &self
                .pending_approvals
                .lock()
                .await
                .get(&approval_id)
                .map(|p| p.record.clone()),
        )?;
        let resolution = match timeout(APPROVAL_TIMEOUT, receiver).await {
            Ok(Ok(value)) => value,
            other => {
                self.pending_approvals.lock().await.remove(&approval_id);
                self.capabilities.revoke_task(proposal.task_id).await;
                if other.is_err() {
                    self.set_task_status(proposal.task_id,TaskStatus::Interrupted,"Approval expired. The task is retained for review and fresh authorization.").await?;
                    return Err(CoreError::PermissionRequired(
                        "Resume this task to review a fresh approval".into(),
                    ));
                }
                return Err(CoreError::ApprovalRejected("Approval was cancelled".into()));
            }
        };
        self.store.save_setting(
            &format!("pending-approval.{approval_id}"),
            &Option::<crate::contracts::ApprovalRecord>::None,
        )?;
        self.pending_approvals.lock().await.remove(&approval_id);
        let approved = matches!(resolution, ApprovalResolution::Approved { .. });
        self.publish(CoreEvent::new(
            Some(proposal.task_id),
            CoreEventKind::ApprovalResolved {
                action_id: proposal.id,
                approved,
            },
        ))?;
        if !approved {
            return Err(CoreError::ApprovalRejected("user denied the action".into()));
        }
        self.update_task(proposal.task_id, |task| {
            if task.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            task.status = TaskStatus::Running;
            if let Some(action) = task.actions.get_mut(&proposal.id) {
                action.status = ActionStatus::Compiling;
            }
            task.touch();
            Ok(())
        })
        .await
    }

    async fn await_question(
        &self,
        proposal: &crate::domain::ActionProposal,
        question: String,
    ) -> CoreResult<ExecutionReceipt> {
        let question_id = Uuid::new_v4();
        let expires_at = Utc::now() + ChronoDuration::seconds(QUESTION_TIMEOUT.as_secs() as i64);
        let (sender, receiver) = oneshot::channel();
        self.pending_questions.lock().await.insert(
            question_id,
            PendingQuestion {
                task_id: proposal.task_id,
                action_id: proposal.id,
                sender,
            },
        );
        self.update_task(proposal.task_id, |task| {
            if task.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            task.status = TaskStatus::WaitingForUser;
            task.touch();
            Ok(())
        })
        .await?;
        self.publish(CoreEvent::new(
            Some(proposal.task_id),
            CoreEventKind::QuestionRequested {
                question_id,
                action_id: proposal.id,
                question,
                expires_at,
            },
        ))?;
        let answer = timeout(QUESTION_TIMEOUT, receiver)
            .await
            .map_err(|_| CoreError::Timeout("question expired".into()))?
            .map_err(|_| CoreError::InvalidAction("question channel closed".into()))?;
        self.pending_questions.lock().await.remove(&question_id);
        self.update_task(proposal.task_id, |task| {
            if task.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            task.status = TaskStatus::Running;
            task.touch();
            Ok(())
        })
        .await?;
        Ok(ExecutionReceipt {
            executor: "native-user-interaction".into(),
            summary: "received user response".into(),
            transient_data: json!({ "user_answered": true, "answer": answer }),
            rollback: None,
        })
    }

    async fn wait_until_runnable(&self, task_id: Uuid) -> CoreResult<()> {
        loop {
            let changed = self.control_changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            match self.get_task(task_id).await?.status {
                TaskStatus::Cancelled => return Err(CoreError::Cancelled),
                TaskStatus::Interrupted => {
                    return Err(CoreError::PermissionRequired(
                        "This task needs recovery review".into(),
                    ));
                }
                TaskStatus::Paused => changed.await,
                _ => return Ok(()),
            }
        }
    }

    async fn succeed_task(&self, task_id: Uuid) -> CoreResult<()> {
        let task = self.get_task(task_id).await?;
        let answer = task
            .tool_results
            .iter()
            .map(|r| r.summary.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        self.finish_answer(task_id, answer).await
    }

    async fn fail_task(&self, task_id: Uuid, error: String) -> CoreResult<()> {
        let error = redact_for_persistence(&error);
        self.update_task(task_id, |task| {
            if !matches!(task.status, TaskStatus::Cancelled | TaskStatus::Interrupted) {
                task.status = TaskStatus::Failed;
            }
            task.final_outcome = Some(error.clone());
            task.touch();
            Ok(())
        })
        .await?;
        self.capabilities.revoke_task(task_id).await;
        self.store.record_outcome(&self.get_task(task_id).await?)?;
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::Error {
                code: "task_failed".into(),
                message: error,
                recoverable: false,
            },
        ))
    }

    async fn mark_action_failed(
        &self,
        task_id: Uuid,
        action_id: Uuid,
        error: String,
    ) -> CoreResult<()> {
        self.update_action(task_id, action_id, |state| {
            state.status = ActionStatus::Failed;
            state.error = Some(error);
            Ok(())
        })
        .await
    }

    async fn set_task_status(
        &self,
        task_id: Uuid,
        status: TaskStatus,
        summary: &str,
    ) -> CoreResult<()> {
        self.update_task(task_id, |task| {
            if task.status == TaskStatus::Cancelled && status != TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            task.status = status;
            task.touch();
            Ok(())
        })
        .await?;
        self.publish(CoreEvent::new(
            Some(task_id),
            CoreEventKind::TaskStatusChanged {
                status,
                summary: summary.into(),
            },
        ))
    }

    async fn get_task(&self, task_id: Uuid) -> CoreResult<Task> {
        self.tasks
            .read()
            .await
            .get(&task_id)
            .cloned()
            .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))
    }

    async fn update_task(
        &self,
        task_id: Uuid,
        operation: impl FnOnce(&mut Task) -> CoreResult<()>,
    ) -> CoreResult<()> {
        // Persist in the same order as in-memory transitions. A concurrent
        // cancellation must never be overwritten by an older dispatch snapshot.
        let mut tasks = self.tasks.write().await;
        let mut task = tasks
            .get(&task_id)
            .cloned()
            .ok_or_else(|| CoreError::TaskNotFound(task_id.to_string()))?;
        operation(&mut task)?;
        self.store.save_task(&task)?;
        tasks.insert(task_id, task);
        Ok(())
    }

    async fn update_action(
        &self,
        task_id: Uuid,
        action_id: Uuid,
        operation: impl FnOnce(&mut crate::domain::ActionState) -> CoreResult<()>,
    ) -> CoreResult<()> {
        self.update_task(task_id, |task| {
            let action = task
                .actions
                .get_mut(&action_id)
                .ok_or_else(|| CoreError::InvalidAction("action not found".into()))?;
            operation(action)?;
            task.touch();
            Ok(())
        })
        .await
    }

    pub fn knowledge_snapshot(&self, conversation: Option<Uuid>) -> CoreResult<KnowledgeSnapshot> {
        let conversations = self.store.conversations()?;
        let selected = conversation.or_else(|| conversations.first().map(|c| c.id));
        Ok(KnowledgeSnapshot {
            conversations,
            messages: selected
                .map(|id| self.store.messages(id, 100))
                .transpose()?
                .unwrap_or_default(),
            memories: self.store.memories(None, false, 500)?,
            memory_enabled: self.store.memory_enabled()?,
        })
    }

    async fn cancel_context_readers(self: &Arc<Self>, except: Option<Uuid>) -> CoreResult<()> {
        let active = self
            .tasks
            .read()
            .await
            .values()
            .filter(|t| {
                Some(t.id) != except
                    && !matches!(
                        t.status,
                        TaskStatus::Succeeded
                            | TaskStatus::Failed
                            | TaskStatus::Cancelled
                            | TaskStatus::Interrupted
                    )
            })
            .map(|t| t.id)
            .collect::<Vec<_>>();
        for task_id in active {
            self.update_task(task_id, |task| {
                if !matches!(
                    task.status,
                    TaskStatus::Succeeded | TaskStatus::Failed | TaskStatus::Cancelled
                ) {
                    task.status = TaskStatus::Cancelled;
                    task.final_outcome = Some(
                        "Memory permissions changed. Start a new task with fresh context.".into(),
                    );
                }
                Ok(())
            })
            .await?;
            self.capabilities.revoke_task(task_id).await;
            self.pending_approvals
                .lock()
                .await
                .retain(|_, p| p.task_id != task_id);
            self.pending_questions
                .lock()
                .await
                .retain(|_, p| p.task_id != task_id);
        }
        self.control_changed.notify_waiters();
        Ok(())
    }

    pub async fn knowledge_command(
        self: &Arc<Self>,
        command: sage_protocol::sage::ipc::v2::KnowledgeCommand,
    ) -> CoreResult<KnowledgeSnapshot> {
        let id = || {
            Uuid::parse_str(&command.id)
                .map_err(|_| CoreError::InvalidAction("Invalid record ID".into()))
        };
        if matches!(command.operation.as_str(), "delete" | "edit" | "disable")
            || (command.operation == "configure" && !command.enabled)
        {
            self.cancel_context_readers(None).await?;
        }
        match command.operation.as_str() {
            "list" => {}
            "remember" => {
                self.store.remember(&command.content, Uuid::new_v4())?;
            }
            "edit" => {
                let mut record = self
                    .store
                    .memories(None, false, 500)?
                    .into_iter()
                    .find(|m| Some(m.id) == id().ok())
                    .ok_or_else(|| CoreError::InvalidAction("Memory not found".into()))?;
                record.content = command.content.clone();
                self.store.save_memory(record)?;
            }
            "delete" => self.store.delete_memory(id()?)?,
            "enable" | "disable" => self
                .store
                .set_memory_record_enabled(id()?, command.operation == "enable")?,
            "configure" => {
                self.store
                    .save_setting("memory.enabled", &command.enabled)?;
            }
            "conversation" => {
                if command.archived
                    && self.tasks.read().await.values().any(|t| {
                        t.conversation_id == id().ok()
                            && !matches!(
                                t.status,
                                TaskStatus::Succeeded
                                    | TaskStatus::Failed
                                    | TaskStatus::Cancelled
                                    | TaskStatus::Interrupted
                            )
                    })
                {
                    return Err(CoreError::InvalidAction(
                        "Stop active work before deleting this conversation.".into(),
                    ));
                }
                self.store.update_conversation(
                    id()?,
                    &command.content,
                    command.pinned,
                    command.archived,
                )?;
            }
            _ => {
                return Err(CoreError::InvalidAction(
                    "Unknown knowledge operation".into(),
                ));
            }
        }
        let conversation = if command.conversation_id.is_empty() {
            None
        } else {
            Some(
                Uuid::parse_str(&command.conversation_id)
                    .map_err(|_| CoreError::InvalidAction("Invalid conversation ID".into()))?,
            )
        };
        self.knowledge_snapshot(conversation)
    }

    pub fn workflow_snapshot(&self) -> CoreResult<serde_json::Value> {
        let skills = self
            .store
            .skills()?
            .iter()
            .map(|skill| {
                let mut value = serde_json::to_value(skill)?;
                value["enabled"] = json!(skill.is_reviewed());
                value["review_digest_candidate"] = json!(skill.digest()?);
                value["preview"] = json!(skill.preview()?);
                Ok(value)
            })
            .collect::<CoreResult<Vec<_>>>()?;
        Ok(
            json!({"skills":skills,"workflows":self.store.workflows()?,"schedules":self.store.schedules()?}),
        )
    }

    pub async fn workflow_command(
        self: &Arc<Self>,
        command: sage_protocol::sage::ipc::v2::WorkflowCommand,
    ) -> CoreResult<serde_json::Value> {
        let _schedule_guard = if matches!(
            command.operation.as_str(),
            "save_schedule" | "delete_schedule"
        ) {
            Some(self.scheduler_lane.lock().await)
        } else {
            None
        };
        if _schedule_guard.is_some() {
            // Management changes revoke the old firing before changing its
            // authorization. Serialize this with claiming/dispatching a firing.
            let schedule_id = if command.operation == "save_schedule" {
                serde_json::from_str::<crate::workflows::Schedule>(&command.json)?
                    .id
                    .to_string()
            } else {
                command.id.clone()
            };
            if let Some(task_id) = self
                .store
                .schedules()?
                .iter()
                .find(|s| s.id.to_string() == schedule_id)
                .and_then(|s| s.last_task_id)
                && let Ok(task) = self.get_task(task_id).await
                && !matches!(
                    task.status,
                    TaskStatus::Succeeded | TaskStatus::Failed | TaskStatus::Cancelled
                )
            {
                self.control_task(task_id, TaskStatus::Cancelled).await?;
            }
        }
        let id = || {
            Uuid::parse_str(&command.id).map_err(|_| CoreError::InvalidAction("Invalid ID".into()))
        };
        let conversation = if command.conversation_id.is_empty() {
            None
        } else {
            Some(
                Uuid::parse_str(&command.conversation_id)
                    .map_err(|_| CoreError::InvalidAction("Invalid conversation".into()))?,
            )
        };
        match command.operation.as_str() {
            "list" => {}
            "capture_skill" => {
                self.store
                    .capture_skill(&self.get_task(id()?).await?, &command.name)?;
            }
            "run_skill" => {
                let skill = self
                    .store
                    .skills()?
                    .into_iter()
                    .find(|s| Some(s.id) == id().ok() && s.is_reviewed())
                    .ok_or_else(|| CoreError::InvalidAction("Skill unavailable".into()))?;
                self.submit_in_conversation(
                    format!("Run skill: {}", skill.name),
                    conversation,
                    false,
                    Some(skill.graph),
                )
                .await?;
            }
            "review_skill" => {
                let mut skill = self
                    .store
                    .skills()?
                    .into_iter()
                    .find(|s| Some(s.id) == id().ok())
                    .ok_or_else(|| CoreError::InvalidAction("Skill not found".into()))?;
                let review: serde_json::Value = serde_json::from_str(&command.json)?;
                let digest = skill.digest()?;
                if review["digest"].as_str() != Some(&digest) {
                    return Err(CoreError::ApprovalRejected(
                        "Skill changed since its review preview".into(),
                    ));
                }
                for node in &skill.graph.nodes {
                    let mut proposal = self.resolver.prepare_proposal(&node.proposal)?;
                    crate::verification::bind_required_outcome(&mut proposal)?;
                    self.compiler
                        .compile(proposal, &self.current_availability().await)?;
                }
                skill.schema_version = 2;
                skill.reviewed_digest = Some(digest);
                skill.enabled = true;
                self.store.save_skill(&skill)?;
            }
            "save_workflow" => self.store.save_workflow(&serde_json::from_str::<
                crate::workflows::Workflow,
            >(&command.json)?)?,
            "run_workflow" => {
                let graph = self.store.workflow_graph(id()?, Uuid::new_v4())?;
                self.submit_in_conversation(graph.goal.clone(), conversation, false, Some(graph))
                    .await?;
            }
            "save_schedule" => {
                let mut schedule: crate::workflows::Schedule = serde_json::from_str(&command.json)?;
                schedule.resources = command
                    .background_resources
                    .iter()
                    .map(|scope| {
                        let effects = scope
                            .effects
                            .iter()
                            .map(|effect| {
                                match sage_protocol::sage::ipc::v2::ResourceEffect::try_from(
                                    *effect,
                                ) {
                                    Ok(sage_protocol::sage::ipc::v2::ResourceEffect::Read) => {
                                        Ok(crate::contracts::Effect::Read)
                                    }
                                    Ok(sage_protocol::sage::ipc::v2::ResourceEffect::Create) => {
                                        Ok(crate::contracts::Effect::Create)
                                    }
                                    _ => Err(CoreError::PermissionRequired(
                                        "Unsupported background effect".into(),
                                    )),
                                }
                            })
                            .collect::<CoreResult<std::collections::BTreeSet<_>>>()?;
                        if effects.is_empty() {
                            return Err(CoreError::PermissionRequired(
                                "Empty background resource grant".into(),
                            ));
                        }
                        Ok(crate::contracts::ResourceScope {
                            root: self
                                .resolver
                                .validate_scope_root(std::path::Path::new(&scope.root))?,
                            effects,
                        })
                    })
                    .collect::<CoreResult<_>>()?;
                schedule.expires_at =
                    chrono::DateTime::from_timestamp_millis(command.background_expires_at_unix_ms);
                schedule.remaining_runs = command.maximum_runs;
                if let crate::workflows::Trigger::FolderChanged { path } = &mut schedule.trigger {
                    *path = self.resolver.validate_scope_root(path)?;
                }
                schedule.request = redact_for_persistence(&schedule.request);
                if let Some(workflow) = schedule.workflow_id {
                    self.store.workflow_graph(workflow, Uuid::new_v4())?;
                }
                self.validate_trigger(&schedule)?;
                self.store
                    .ensure_conversation(Some(schedule.conversation_id), &schedule.name)?;
                if let crate::workflows::Trigger::Once { at } = schedule.trigger {
                    schedule.next_run_at = at;
                }
                self.store.save_schedule(&schedule)?;
            }
            "delete_skill" => self.store.delete_definition("skill", id()?)?,
            "delete_workflow" => self.store.delete_definition("workflow", id()?)?,
            "delete_schedule" => self.store.delete_definition("schedule", id()?)?,
            _ => {
                return Err(CoreError::InvalidAction(
                    "Unknown workflow operation".into(),
                ));
            }
        }
        self.store.append_audit(
            None,
            None,
            "workflow_management",
            &json!({"operation":command.operation,"id":command.id}),
        )?;
        self.workflow_snapshot()
    }

    async fn current_availability(&self) -> ExecutorAvailability {
        let mut availability = self.availability.clone();
        availability.accessibility = self.adapters.available("native").await;
        availability.keyboard = availability.accessibility;
        availability.browser_dom = self.adapters.available("browser").await;
        availability
    }

    async fn available_tools(&self) -> Vec<ToolDescriptor> {
        let availability = self.current_availability().await;
        default_tools()
            .into_iter()
            .filter(|tool| match tool.executor.as_str() {
                "browser" => availability.browser_dom,
                "sandbox" => availability.sandbox,
                "privileged" => availability.privileged_helper,
                "native" if tool.name == "open_application" => availability.accessibility,
                _ => true,
            })
            .collect()
    }

    async fn resume_interrupted(self: &Arc<Self>, task_id: Uuid) -> CoreResult<()> {
        let _submission = self.submission_lock.lock().await;
        let task = self.get_task(task_id).await?;
        if task.budget_exhausted || task.actions.len() >= 32 {
            if task.actions.values().any(|action| {
                !matches!(
                    action.status,
                    ActionStatus::Succeeded | ActionStatus::Skipped
                )
            }) {
                return Err(CoreError::PermissionRequired(
                    "Reconcile unfinished effects before starting a continuation".into(),
                ));
            }
            let contract = task.contract.as_ref().ok_or_else(|| {
                CoreError::PermissionRequired("Pre-v2 work needs a new task scope".into())
            })?;
            let resources = contract.resources.clone();
            drop(_submission);
            self.submit_run(
                task.request.clone(),
                task.conversation_id,
                None,
                resources,
                None,
                Some(task),
            )
            .await?;
            return Ok(());
        }
        if self.tasks.read().await.values().any(|other| {
            other.id != task_id
                && other.conversation_id == task.conversation_id
                && !matches!(
                    other.status,
                    TaskStatus::Succeeded
                        | TaskStatus::Failed
                        | TaskStatus::Cancelled
                        | TaskStatus::Interrupted
                )
        }) {
            return Err(CoreError::InvalidAction(
                "Finish other work in this conversation first.".into(),
            ));
        }
        self.capabilities.revoke_task(task_id).await;
        let core = Arc::clone(self);
        if task.contract.is_none() {
            return Err(CoreError::PermissionRequired(
                "Start a new task to authorize this pre-v2 work".into(),
            ));
        }
        self.update_task(task_id, |task| {
            if let Some(contract) = &mut task.contract {
                if !task.background {
                    contract.expires_at = Utc::now() + ChronoDuration::hours(2);
                }
                contract.validate(task_id)?;
            }
            Ok(())
        })
        .await?;
        self.set_task_status(
            task_id,
            TaskStatus::Paused,
            "Re-observing completed actions before recovery",
        )
        .await?;
        // Run observation away from the IPC reader, which must receive adapter replies.
        tokio::spawn(async move {
            let result = core.prepare_recovery(task_id).await;
            match result {
                Ok(()) => {
                    core.capabilities.reopen_run(task_id).await;
                    if let Err(error) = core.run_task(task_id).await {
                        let _ = core.fail_task(task_id, error.to_string()).await;
                    }
                }
                Err(error) => {
                    let _ = core
                        .set_task_status(
                            task_id,
                            TaskStatus::Interrupted,
                            &format!("Recovery requires review: {error}"),
                        )
                        .await;
                }
            }
        });
        Ok(())
    }

    async fn prepare_recovery(&self, task_id: Uuid) -> CoreResult<()> {
        let task = self.get_task(task_id).await?;
        let availability = self.current_availability().await;
        for state in task.actions.values() {
            // A running action may already have changed the outside world.
            // Only repeat it if its declared outcome can be independently observed.
            if matches!(
                state.status,
                ActionStatus::Succeeded
                    | ActionStatus::Running
                    | ActionStatus::Verifying
                    | ActionStatus::Uncertain
            ) {
                if let Action::ReadFile { path, .. } = &state.proposal.action {
                    let expected = task.tool_results.iter().find(|result| result.action_id == state.proposal.id)
                        .and_then(|result| result.output["sha256"].as_str())
                        .ok_or_else(|| CoreError::VerificationFailed("The earlier read has no durable content identity; read the source again in a new task".into()))?;
                    let actual = crate::execution::files::PinnedPath::open(path)?
                        .digest(16 * 1024 * 1024)?;
                    if actual != expected {
                        return Err(CoreError::VerificationFailed("A previously read file changed; a fresh task must read it before continuing".into()));
                    }
                }
                if !matches!(
                    state.proposal.expected_outcome,
                    crate::domain::ExpectedOutcome::Condition { .. }
                        | crate::domain::ExpectedOutcome::FileContains { .. }
                ) {
                    return Err(CoreError::VerificationFailed("An executed action has no independently observable recovery outcome; review it before starting a new task.".into()));
                }
                let receipt = ExecutionReceipt {
                    executor: "recovery".into(),
                    summary: String::new(),
                    transient_data: json!({}),
                    rollback: None,
                };
                let observation = self.observer.observe(&state.proposal, &receipt).await?;
                self.verifier
                    .verify(&state.proposal.expected_outcome, &observation)?;
                self.update_action(task_id, state.proposal.id, |s| {
                    s.status = ActionStatus::Succeeded;
                    Ok(())
                })
                .await?;
            } else if state.status != ActionStatus::Skipped {
                let proposal = self.resolver.prepare_proposal(&state.proposal)?;
                if let PolicyDecision::Deny { reason, .. } = self.policy.evaluate(
                    &proposal,
                    &PolicyContext {
                        task_request: task.request.clone(),
                        has_fresh_native_authentication: false,
                        is_recovery_attempt: true,
                    },
                )? {
                    return Err(CoreError::PolicyDenied(reason));
                }
                self.compiler.compile(proposal, &availability)?;
            }
        }
        self.store.append_audit(Some(task_id),None,"recovery_revalidated",&json!({"completed":task.completed_count(),"remaining":task.actions.len()-task.completed_count()}))?;
        self.update_task(task_id, |task| {
            if task.status == TaskStatus::Cancelled {
                return Err(CoreError::Cancelled);
            }
            for state in task.actions.values_mut() {
                if !matches!(
                    state.status,
                    ActionStatus::Succeeded | ActionStatus::Skipped
                ) {
                    state.status = ActionStatus::Pending;
                    state.error = None;
                }
            }
            task.status = TaskStatus::Running;
            task.recovery_attempt = true;
            task.final_outcome = None;
            task.touch();
            Ok(())
        })
        .await
    }

    fn validate_trigger(&self, schedule: &crate::workflows::Schedule) -> CoreResult<()> {
        use crate::domain::{ActionProposal, Condition, ExpectedOutcome};
        let condition = match &schedule.trigger {
            crate::workflows::Trigger::FolderChanged { path } => {
                Some(Condition::FileExists { path: path.clone() })
            }
            crate::workflows::Trigger::Condition { condition } => Some(condition.clone()),
            _ => None,
        };
        if let Some(condition) = condition {
            let path = match &condition {
                Condition::FileExists { path }
                | Condition::FileAbsent { path }
                | Condition::FolderExists { path } => path,
                _ => {
                    return Err(CoreError::PermissionRequired(
                        "Background application observation is not qualified".into(),
                    ));
                }
            };
            if !schedule.resources.iter().any(|scope| {
                path.starts_with(&scope.root)
                    && scope.effects.contains(&crate::contracts::Effect::Read)
            }) {
                return Err(CoreError::PermissionRequired(
                    "The trigger needs a matching background folder read grant".into(),
                ));
            }
            let proposal = ActionProposal {
                id: Uuid::new_v4(),
                task_id: Uuid::new_v4(),
                action: Action::WaitForCondition {
                    condition: condition.clone(),
                    timeout_ms: 1000,
                },
                expected_outcome: ExpectedOutcome::Condition {
                    condition: condition.clone(),
                },
                target_resource: "condition".into(),
                provenance: crate::domain::Provenance::user(),
                metadata: Default::default(),
            };
            let resolved = self.resolver.prepare_proposal(&proposal)?;
            if let Action::WaitForCondition { condition, .. } = resolved.action {
                let resolved_path = match condition {
                    Condition::FileExists { path }
                    | Condition::FileAbsent { path }
                    | Condition::FolderExists { path } => path,
                    _ => {
                        return Err(CoreError::PermissionRequired(
                            "Unsupported background trigger".into(),
                        ));
                    }
                };
                if resolved_path != *path {
                    return Err(CoreError::PermissionRequired(
                        "Background target identity changed; re-authorize the canonical folder"
                            .into(),
                    ));
                }
                crate::execution::files::PinnedPath::open(path)?;
            }
        }
        Ok(())
    }

    pub fn start_scheduler(self: &Arc<Self>) {
        let core = Arc::downgrade(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(15));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            let (sender, mut changes) = tokio::sync::mpsc::channel(128);
            let mut monitor = crate::workflows::FolderMonitor::new(sender);
            let mut changed_paths = std::collections::HashSet::new();
            loop {
                tokio::select! {
                    _=interval.tick()=>{},
                    Some(())=changes.recv()=>{},
                }
                let Some(core) = core.upgrade() else {
                    break;
                };
                if let Ok(schedules) = core.store.schedules() {
                    let mut valid = Vec::new();
                    for schedule in schedules {
                        if let Err(error) = core.validate_trigger(&schedule) {
                            let _ = core
                                .store
                                .disable_schedule_if_current(&schedule, &error.to_string());
                        } else {
                            valid.push(schedule);
                        }
                    }
                    monitor.sync(&valid);
                    if let Ok((changed, errors)) = monitor.drain() {
                        changed_paths.extend(changed);
                        for schedule in &valid {
                            if let crate::workflows::Trigger::FolderChanged { path } =
                                &schedule.trigger
                                && let Some(error) = errors.get(path)
                            {
                                let _ = core.store.disable_schedule_if_current(schedule, error);
                            }
                        }
                    }
                }
                if let Err(error) = core.scheduler_tick(&mut changed_paths).await {
                    tracing::warn!(error=%error,"scheduler tick failed");
                }
                changed_paths.clear();
            }
        });
    }

    async fn scheduler_tick(
        self: &Arc<Self>,
        changed_paths: &mut std::collections::HashSet<std::path::PathBuf>,
    ) -> CoreResult<()> {
        let _schedule_guard = self.scheduler_lane.lock().await;
        use crate::workflows::Trigger;
        // Persist a coalesced trigger before checking cooldown or overlap. A
        // busy run must not silently discard a folder event.
        for mut schedule in self.store.schedules()?.into_iter().filter(|s| s.enabled) {
            if let Trigger::FolderChanged { path } = &schedule.trigger
                && changed_paths
                    .iter()
                    .any(|changed| changed.starts_with(path))
            {
                schedule.trigger_pending = true;
                if self.validate_trigger(&schedule).is_ok() {
                    self.store.save_schedule(&schedule)?;
                }
            }
        }
        for mut schedule in self
            .store
            .schedules()?
            .into_iter()
            .filter(|s| s.enabled && s.next_run_at <= Utc::now())
        {
            if let Err(error) = self.validate_trigger(&schedule) {
                self.store
                    .disable_schedule_if_current(&schedule, &error.to_string())?;
                continue;
            }
            if schedule
                .expires_at
                .is_none_or(|expiry| expiry <= Utc::now())
                || schedule.remaining_runs == 0
            {
                schedule.enabled = false;
                schedule.last_error =
                    Some("Background authorization expired or reached its run budget".into());
                self.store.save_schedule(&schedule)?;
                continue;
            }
            if self.tasks.read().await.values().any(|t| {
                t.conversation_id == Some(schedule.conversation_id)
                    && !matches!(
                        t.status,
                        TaskStatus::Succeeded | TaskStatus::Failed | TaskStatus::Cancelled
                    )
            }) {
                continue;
            }
            let ready = match &schedule.trigger {
                Trigger::Once { at } => *at <= Utc::now(),
                Trigger::Interval { .. } => true,
                Trigger::FolderChanged { .. } => schedule.trigger_pending,
                Trigger::Condition { condition } => {
                    let matches = match condition {
                        crate::domain::Condition::FileExists { path } => {
                            crate::execution::files::PinnedPath::open(path)?.exists()
                        }
                        crate::domain::Condition::FileAbsent { path } => {
                            !crate::execution::files::PinnedPath::open(path)?.exists()
                        }
                        crate::domain::Condition::FolderExists { path } => {
                            crate::execution::files::PinnedPath::open(path)?.is_directory()
                        }
                        _ => self
                            .adapters
                            .observe_condition(condition, None)
                            .await
                            .is_ok_and(|e| {
                                self.verifier
                                    .verify(
                                        &crate::domain::ExpectedOutcome::Condition {
                                            condition: condition.clone(),
                                        },
                                        &crate::observation::Observation {
                                            observed_at: Utc::now(),
                                            provenance: crate::domain::Provenance::external(
                                                crate::domain::ProvenanceSource::OperatingSystem,
                                                "trigger",
                                            ),
                                            summary: String::new(),
                                            evidence: vec![e],
                                        },
                                    )
                                    .is_ok()
                            }),
                    };
                    let edge = matches && !schedule.last_condition;
                    schedule.last_condition = matches;
                    self.store.save_schedule(&schedule)?;
                    edge
                }
            };
            if !ready {
                continue;
            }
            let Some(run) = self.store.claim_schedule(&mut schedule)? else {
                continue;
            };
            let result = async {
                let graph = schedule
                    .workflow_id
                    .map(|id| self.store.workflow_graph(id, Uuid::new_v4()))
                    .transpose()?;
                self.submit_run(
                    schedule.request.clone(),
                    Some(schedule.conversation_id),
                    graph,
                    schedule.resources.clone(),
                    schedule.expires_at,
                    None,
                )
                .await
            }
            .await;
            match result {
                Ok(task) => {
                    schedule.last_task_id = Some(task);
                    schedule.last_error = None;
                    self.store.finish_schedule_claim(run, Some(task))?;
                }
                Err(error) => {
                    schedule.last_error = Some(redact_for_persistence(&error.to_string()));
                    schedule.enabled = false;
                    self.store.finish_schedule_claim(run, None)?;
                }
            }
            self.store.save_schedule(&schedule)?;
        }
        Ok(())
    }

    fn publish(&self, event: CoreEvent) -> CoreResult<()> {
        self.store.save_event(&event)?;
        self.events.publish(event);
        Ok(())
    }
}

fn default_tools() -> Vec<ToolDescriptor> {
    crate::features::descriptors()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;

    use async_trait::async_trait;
    use tempfile::tempdir;

    use crate::domain::{
        Action, ActionGraph, ActionNode, ActionProposal, Condition, ExpectedOutcome, Provenance,
    };
    use crate::events::CoreEventKind;
    use crate::model::{
        PlanningContext, ProviderDescriptor, ReplanContext, UnconfiguredModelProvider,
    };
    use crate::secrets::testing::MemorySecretStore;

    use super::*;

    struct FolderPlanProvider {
        path: PathBuf,
    }

    #[async_trait]
    impl ModelProvider for FolderPlanProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            ProviderDescriptor {
                id: "test-folder-plan".into(),
                display_name: "Test folder plan".into(),
                local: true,
                roles: vec![crate::model::ModelRole::Reasoning],
            }
        }

        async fn create_plan(&self, context: PlanningContext) -> CoreResult<ActionGraph> {
            Ok(ActionGraph {
                goal: "Create and verify one folder".into(),
                nodes: vec![ActionNode {
                    proposal: ActionProposal {
                        id: Uuid::new_v4(),
                        task_id: context.task_id,
                        action: Action::CreateFolder {
                            path: self.path.clone(),
                        },
                        expected_outcome: ExpectedOutcome::Condition {
                            condition: Condition::FileExists {
                                path: self.path.clone(),
                            },
                        },
                        target_resource: self.path.to_string_lossy().into_owned(),
                        provenance: Provenance::model(Vec::new()),
                        metadata: BTreeMap::new(),
                    },
                    depends_on: BTreeSet::new(),
                }],
            })
        }

        async fn replan(&self, _context: ReplanContext) -> CoreResult<ActionGraph> {
            Err(CoreError::Model("test does not expect replanning".into()))
        }
    }

    #[tokio::test]
    async fn approval_capability_execution_verification_and_undo_are_end_to_end() {
        let directory = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let destination = workspace.path().join("verified-folder");
        let config = CoreConfig::for_test(directory.path());
        let core = SageCore::new(
            config,
            Arc::new(FolderPlanProvider {
                path: destination.clone(),
            }),
        )
        .unwrap();
        let mut events = core.events().subscribe();
        let task_id = core.submit_task("Create a test folder").await.unwrap();

        let completed = timeout(Duration::from_secs(10), async {
            loop {
                let event = events.recv().await.unwrap();
                match event.kind {
                    CoreEventKind::ApprovalRequested {
                        approval_id,
                        action_id,
                        digest,
                        ..
                    } => {
                        core.resolve_approval(
                            approval_id,
                            task_id,
                            action_id,
                            &digest,
                            ApprovalResolution::Approved {
                                native_authentication_satisfied: false,
                            },
                        )
                        .await
                        .unwrap();
                    }
                    CoreEventKind::TaskCompleted { .. } => break,
                    CoreEventKind::Error { message, .. } => panic!("task failed: {message}"),
                    _ => {}
                }
            }
        })
        .await;
        assert!(completed.is_ok(), "task did not complete before timeout");
        assert!(destination.is_dir());
        let task = core.get_task(task_id).await.unwrap();
        assert_eq!(task.status, TaskStatus::Succeeded);
        assert_eq!(task.completed_count(), 1);

        core.undo_last_action(task_id).await.unwrap();
        assert!(!destination.exists());
    }

    #[tokio::test]
    async fn provider_credential_is_saved_only_after_native_authentication() {
        let directory = tempdir().unwrap();
        let secret_store = Arc::new(MemorySecretStore::default());
        let core = SageCore::new_with_secret_store(
            CoreConfig::for_test(directory.path()),
            Arc::new(UnconfiguredModelProvider),
            secret_store.clone(),
        )
        .unwrap();
        let account = ProviderSettings {
            role: "reasoning".into(),
            provider: "openai".into(),
            model: "test".into(),
            endpoint: String::new(),
            has_api_key: false,
        }
        .credential_account();

        let rejected = core
            .save_provider_settings(
                "reasoning".into(),
                "openai".into(),
                "gpt-5.4".into(),
                String::new(),
                "sk-test-secret".into(),
                false,
                false,
            )
            .await;
        assert!(matches!(rejected, Err(CoreError::ApprovalRejected(_))));
        assert!(secret_store.get(&account).unwrap().is_none());

        core.save_provider_settings(
            "reasoning".into(),
            "openai".into(),
            "gpt-5.4".into(),
            String::new(),
            "sk-test-secret".into(),
            false,
            true,
        )
        .await
        .unwrap();

        assert!(secret_store.get(&account).unwrap().is_some());
        let snapshot = core.snapshot(true).await;
        assert_eq!(snapshot.provider_settings.len(), 1);
        assert!(snapshot.provider_settings[0].has_api_key);

        core.save_provider_settings(
            "reasoning".into(),
            "openai".into(),
            "test".into(),
            "https://different.example/v1".into(),
            String::new(),
            false,
            false,
        )
        .await
        .unwrap();
        let changed = core.snapshot(true).await;
        assert!(!changed.provider_settings[0].has_api_key);
        assert!(
            secret_store
                .get(&changed.provider_settings[0].credential_account())
                .unwrap()
                .is_none()
        );
        assert!(
            secret_store.get(&account).unwrap().is_some(),
            "Endpoint changes must preserve the old key without forwarding it"
        );
    }

    struct BudgetProvider {
        source: PathBuf,
        first: std::sync::Mutex<Option<Uuid>>,
    }
    #[async_trait]
    impl ModelProvider for BudgetProvider {
        fn descriptor(&self) -> crate::model::ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: crate::model::ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, context: TurnContext) -> CoreResult<ModelTurn> {
            let mut first = self.first.lock().unwrap();
            let first = first.get_or_insert(context.planning.task_id);
            if *first != context.planning.task_id {
                assert!(!context.results.is_empty());
                return Ok(ModelTurn::Answer(
                    "Continued using earlier evidence and a fresh scope.".into(),
                ));
            }
            Ok(proposal_turn(
                context.planning.task_id,
                Action::ReadFile {
                    path: self.source.clone(),
                    max_bytes: 64,
                },
            ))
        }
    }
    #[tokio::test]
    async fn exhausted_budget_continues_with_fresh_run_identity_and_no_old_actions() {
        let state = tempdir().unwrap();
        let work = tempdir().unwrap();
        let root = work.path().canonicalize().unwrap();
        let source = root.join("input.txt");
        std::fs::write(&source, b"evidence").unwrap();
        let core = SageCore::new_with_secret_store(
            CoreConfig::for_test(state.path()),
            Arc::new(BudgetProvider {
                source,
                first: Default::default(),
            }),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        let id = core
            .submit_scoped(
                "Read the next input".into(),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root,
                    effects: std::collections::BTreeSet::from([crate::contracts::Effect::Read]),
                }],
            )
            .await
            .unwrap();
        let paused = completed(&core, id).await;
        assert_eq!(paused.status, TaskStatus::Interrupted);
        assert!(paused.budget_exhausted);
        assert_eq!(paused.actions.len(), 32);
        core.control_task(id, TaskStatus::Running).await.unwrap();
        let next = core
            .tasks
            .read()
            .await
            .values()
            .find(|task| task.continuation_of == Some(id))
            .unwrap()
            .id;
        assert_ne!(next, id);
        let done = completed(&core, next).await;
        assert_eq!(done.status, TaskStatus::Succeeded);
        assert!(done.actions.is_empty());
        assert_eq!(done.contract.unwrap().run_id, next);
        assert_eq!(core.get_task(id).await.unwrap().actions.len(), 32);
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn background_trigger_cannot_follow_a_replaced_folder_outside_its_scope() {
        let state = tempdir().unwrap();
        let work = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let root = work.path().canonicalize().unwrap();
        let directory = root.join("watched");
        std::fs::create_dir(&directory).unwrap();
        let core = SageCore::new_with_secret_store(
            CoreConfig::for_test(state.path()),
            Arc::new(UnconfiguredModelProvider),
            Arc::new(MemorySecretStore::default()),
        )
        .unwrap();
        let mut schedule:crate::workflows::Schedule=serde_json::from_value(json!({"id":Uuid::new_v4(),"name":"watch","request":"Review changes","conversation_id":Uuid::new_v4(),"workflow_id":null,"trigger":{"kind":"folder_changed","path":directory},"enabled":true,"next_run_at":Utc::now(),"last_task_id":null,"last_condition":false,"last_error":null,"resources":[{"root":root,"effects":["read"]}],"expires_at":Utc::now()+ChronoDuration::hours(1),"remaining_runs":2})).unwrap();
        core.validate_trigger(&schedule).unwrap();
        std::fs::remove_dir(&directory).unwrap();
        std::os::unix::fs::symlink(outside.path(), &directory).unwrap();
        assert!(core.validate_trigger(&schedule).is_err());
        schedule.next_run_at = Utc::now() - ChronoDuration::seconds(1);
        core.store.save_schedule(&schedule).unwrap();
        core.scheduler_tick(&mut Default::default()).await.unwrap();
        assert!(!core.store.schedules().unwrap()[0].enabled);
    }
    struct ResultLoopProvider {
        source: PathBuf,
        destination: PathBuf,
    }
    fn proposal_turn(task_id: Uuid, action: Action) -> ModelTurn {
        ModelTurn::Actions(ActionGraph {
            goal: "Verified test task".into(),
            nodes: vec![ActionNode {
                proposal: ActionProposal {
                    id: Uuid::new_v4(),
                    task_id,
                    action,
                    expected_outcome: ExpectedOutcome::UserAnswered,
                    target_resource: "untrusted model target".into(),
                    provenance: Provenance::model(Vec::new()),
                    metadata: Default::default(),
                },
                depends_on: Default::default(),
            }],
        })
    }
    #[async_trait]
    impl ModelProvider for ResultLoopProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, context: TurnContext) -> CoreResult<ModelTurn> {
            assert!(
                context
                    .planning
                    .trusted_constraints
                    .iter()
                    .any(|s| s.contains("never approval"))
            );
            Ok(match context.results.len() {
                0 => proposal_turn(
                    context.planning.task_id,
                    Action::ReadFile {
                        path: self.source.clone(),
                        max_bytes: 1024,
                    },
                ),
                1 => {
                    let text = context.results[0].output["text"]
                        .as_str()
                        .expect("actual file result");
                    assert_eq!(text, "orchid-42");
                    proposal_turn(
                        context.planning.task_id,
                        Action::WriteFile {
                            path: self.destination.clone(),
                            content: format!("Read: {text}"),
                            overwrite: false,
                        },
                    )
                }
                2 => ModelTurn::Answer("Read orchid-42 and saved the verified result.".into()),
                _ => panic!("unbounded tool loop"),
            })
        }
    }
    async fn completed(core: &Arc<SageCore>, task: Uuid) -> Task {
        timeout(Duration::from_secs(5), async {
            loop {
                let value = core.get_task(task).await.unwrap();
                if matches!(
                    value.status,
                    TaskStatus::Succeeded
                        | TaskStatus::Failed
                        | TaskStatus::Cancelled
                        | TaskStatus::Interrupted
                ) {
                    break value;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("run did not reach a terminal or review state")
    }
    #[tokio::test]
    async fn actual_file_results_drive_later_actions_without_repeated_scope_approval() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let source = workspace.path().join("source.txt");
        let destination = workspace.path().join("answer.txt");
        std::fs::write(&source, "orchid-42").unwrap();
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(ResultLoopProvider {
                source,
                destination: destination.clone(),
            }),
        )
        .unwrap();
        let mut events = core.events().subscribe();
        let task = core
            .submit_scoped(
                "Read and summarize the file".into(),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root: workspace.path().into(),
                    effects: BTreeSet::from([
                        crate::contracts::Effect::Read,
                        crate::contracts::Effect::Create,
                    ]),
                }],
            )
            .await
            .unwrap();
        let result = completed(&core, task).await;
        assert_eq!(
            result.status,
            TaskStatus::Succeeded,
            "{:?}",
            result.final_outcome
        );
        assert_eq!(result.completed_count(), 2);
        assert_eq!(
            std::fs::read_to_string(destination).unwrap(),
            "Read: orchid-42"
        );
        while let Ok(event) = events.try_recv() {
            assert!(!matches!(
                event.kind,
                CoreEventKind::ApprovalRequested { .. }
            ));
        }
        assert_eq!(
            result.tool_results[0].label.sensitivity,
            crate::contracts::Sensitivity::Private
        );
    }

    struct AnswerProvider {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        remote: bool,
    }
    #[async_trait]
    impl ModelProvider for AnswerProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        fn data_destination(&self) -> CoreResult<Option<String>> {
            Ok(self
                .remote
                .then(|| "https://provider.example/v1#test".into()))
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, _: TurnContext) -> CoreResult<ModelTurn> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ModelTurn::Answer("A triangle has three sides.".into()))
        }
    }
    #[tokio::test]
    async fn ordinary_question_finishes_without_any_computer_action() {
        let data = tempdir().unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(AnswerProvider {
                calls: calls.clone(),
                remote: false,
            }),
        )
        .unwrap();
        let id = core
            .submit_task("How many sides does a triangle have?")
            .await
            .unwrap();
        let task = completed(&core, id).await;
        assert_eq!(task.status, TaskStatus::Succeeded);
        assert!(task.actions.is_empty());
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(
            task.final_outcome.as_deref(),
            Some("A triangle has three sides.")
        );
    }
    #[tokio::test]
    async fn denied_data_release_never_calls_the_provider_and_cannot_replay_approval() {
        let data = tempdir().unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(AnswerProvider {
                calls: calls.clone(),
                remote: true,
            }),
        )
        .unwrap();
        let mut events = core.events().subscribe();
        let id = core
            .submit_task("Use this private project context")
            .await
            .unwrap();
        let (approval_id, action_id, digest) = timeout(Duration::from_secs(5), async {
            loop {
                if let CoreEventKind::ApprovalRequested {
                    approval_id,
                    action_id,
                    digest,
                    ..
                } = events.recv().await.unwrap().kind
                {
                    break (approval_id, action_id, digest);
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(
            core.resolve_approval(
                approval_id,
                id,
                action_id,
                "wrong",
                ApprovalResolution::Denied
            )
            .await
            .is_err()
        );
        core.resolve_approval(
            approval_id,
            id,
            action_id,
            &digest,
            ApprovalResolution::Denied,
        )
        .await
        .unwrap();
        assert!(
            core.resolve_approval(
                approval_id,
                id,
                action_id,
                &digest,
                ApprovalResolution::Approved {
                    native_authentication_satisfied: false
                }
            )
            .await
            .is_err()
        );
        assert_eq!(completed(&core, id).await.status, TaskStatus::Failed);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    struct StalledProvider {
        entered: Arc<Notify>,
        dropped: Arc<Notify>,
    }
    struct OnDrop(Arc<Notify>);
    impl Drop for OnDrop {
        fn drop(&mut self) {
            self.0.notify_one();
        }
    }
    #[async_trait]
    impl ModelProvider for StalledProvider {
        fn descriptor(&self) -> ProviderDescriptor {
            UnconfiguredModelProvider.descriptor()
        }
        async fn create_plan(&self, _: PlanningContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn replan(&self, _: ReplanContext) -> CoreResult<ActionGraph> {
            unreachable!()
        }
        async fn next_turn(&self, _: TurnContext) -> CoreResult<ModelTurn> {
            let _guard = OnDrop(self.dropped.clone());
            self.entered.notify_one();
            std::future::pending().await
        }
    }
    #[tokio::test]
    async fn cancellation_drops_an_inflight_model_call_and_persists_cancelled_state() {
        let data = tempdir().unwrap();
        let entered = Arc::new(Notify::new());
        let dropped = Arc::new(Notify::new());
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(StalledProvider {
                entered: entered.clone(),
                dropped: dropped.clone(),
            }),
        )
        .unwrap();
        let id = core.submit_task("Wait on a model").await.unwrap();
        timeout(Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        core.control_task(id, TaskStatus::Cancelled).await.unwrap();
        timeout(Duration::from_secs(1), dropped.notified())
            .await
            .unwrap();
        assert_eq!(
            core.get_task(id).await.unwrap().status,
            TaskStatus::Cancelled
        );
        assert_eq!(
            core.store
                .load_tasks(true)
                .unwrap()
                .into_iter()
                .find(|t| t.id == id)
                .unwrap()
                .status,
            TaskStatus::Cancelled
        );
    }
    #[tokio::test]
    async fn cancellation_prevents_a_queued_mutation_from_starting() {
        let data = tempdir().unwrap();
        let workspace = tempdir().unwrap();
        let path = workspace.path().join("never-created");
        let core = SageCore::new(
            CoreConfig::for_test(data.path()),
            Arc::new(FolderPlanProvider { path: path.clone() }),
        )
        .unwrap();
        let lane = core.mutation_lane.lock().await;
        let mut events = core.events().subscribe();
        let id = core
            .submit_scoped(
                "Create this folder".into(),
                None,
                false,
                None,
                vec![crate::contracts::ResourceScope {
                    root: workspace.path().into(),
                    effects: BTreeSet::from([crate::contracts::Effect::Create]),
                }],
            )
            .await
            .unwrap();
        timeout(Duration::from_secs(5), async {
            loop {
                if matches!(
                    events.recv().await.unwrap().kind,
                    CoreEventKind::ActionProposed { .. }
                ) {
                    break;
                }
            }
        })
        .await
        .unwrap();
        core.control_task(id, TaskStatus::Cancelled).await.unwrap();
        drop(lane);
        timeout(Duration::from_secs(5), async {
            loop {
                if matches!(
                    events.recv().await.unwrap().kind,
                    CoreEventKind::Error { .. }
                ) {
                    break;
                }
            }
        })
        .await
        .unwrap();
        assert!(!path.exists());
        assert_eq!(
            core.get_task(id).await.unwrap().status,
            TaskStatus::Cancelled
        );
    }
}
