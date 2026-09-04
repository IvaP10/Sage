use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

use crate::domain::{ActionGraph, Task};
use crate::error::{CoreError, CoreResult};
use crate::secrets::{SecretBytes, SecretStore};

const DEFAULT_OPENAI_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
const REQUEST_TIMEOUT_SECONDS: u64 = 90;
const MAX_PROVIDER_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_PLAN_ACTIONS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Reasoning,
    Vision,
    SpeechRecognition,
    SpeechSynthesis,
    Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderDescriptor {
    pub id: String,
    pub display_name: String,
    pub local: bool,
    pub roles: Vec<ModelRole>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderSettings {
    pub role: String,
    pub provider: String,
    pub model: String,
    pub endpoint: String,
    pub has_api_key: bool,
}

impl ProviderSettings {
    pub fn credential_account(&self) -> String {
        format!("provider:{}:{}", self.role, self.provider)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningContext {
    pub task_id: uuid::Uuid,
    pub user_request: String,
    pub current_state: serde_json::Value,
    pub available_tools: Vec<ToolDescriptor>,
    pub trusted_constraints: Vec<String>,
    pub untrusted_context: Vec<UntrustedContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplanContext {
    pub task: Task,
    pub failed_action_id: uuid::Uuid,
    pub observation: serde_json::Value,
    pub attempt: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UntrustedContext {
    pub source: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub name: String,
    pub version: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
    pub risk: String,
    pub required_capabilities: Vec<String>,
    pub supported_platforms: Vec<String>,
    pub requires_confirmation: bool,
    pub executor: String,
    pub timeout_ms: u64,
    pub verification_strategy: String,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn descriptor(&self) -> ProviderDescriptor;
    async fn create_plan(&self, context: PlanningContext) -> CoreResult<ActionGraph>;
    async fn replan(&self, context: ReplanContext) -> CoreResult<ActionGraph>;

    /// Apply settings without making a network request. Implementations that
    /// do not use a remote provider may leave this as a no-op.
    fn configure(
        &self,
        _settings: Option<ProviderSettings>,
        _secret_store: Arc<dyn SecretStore>,
    ) -> CoreResult<()> {
        Ok(())
    }

    async fn test_connection(
        &self,
        _settings: ProviderSettings,
        _api_key: Option<SecretBytes>,
    ) -> CoreResult<String> {
        Err(CoreError::Model(
            "provider does not support connection tests".into(),
        ))
    }
}

#[async_trait]
pub trait SpeechRecognitionProvider: Send + Sync {
    async fn transcribe(&self, audio: &[u8]) -> CoreResult<String>;
}

#[async_trait]
pub trait SpeechSynthesisProvider: Send + Sync {
    async fn synthesize(&self, text: &str) -> CoreResult<Vec<u8>>;
}

#[async_trait]
pub trait VisionProvider: Send + Sync {
    async fn inspect(&self, image: &[u8], question: &str) -> CoreResult<serde_json::Value>;
}

#[derive(Debug, Default)]
pub struct UnconfiguredModelProvider;

#[async_trait]
impl ModelProvider for UnconfiguredModelProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        ProviderDescriptor {
            id: "unconfigured".into(),
            display_name: "No reasoning model configured".into(),
            local: true,
            roles: vec![ModelRole::Reasoning],
        }
    }

    async fn create_plan(&self, _context: PlanningContext) -> CoreResult<ActionGraph> {
        Err(CoreError::Model(
            "configure a local or cloud reasoning provider before starting an agent task".into(),
        ))
    }

    async fn replan(&self, _context: ReplanContext) -> CoreResult<ActionGraph> {
        Err(CoreError::Model(
            "reasoning provider is unavailable for replanning".into(),
        ))
    }
}

/// OpenAI Chat Completions-compatible provider used by both native clients.
///
/// The model is only allowed to return a draft plan. IDs, dependencies that
/// refer to IDs, and provenance are assigned by Sage in `draft_to_graph`.
pub struct OpenAICompatibleProvider {
    settings: RwLock<Option<ProviderSettings>>,
    secret_store: RwLock<Option<Arc<dyn SecretStore>>>,
}

impl std::fmt::Debug for OpenAICompatibleProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAICompatibleProvider")
            .field(
                "configured",
                &self
                    .settings
                    .read()
                    .map(|value| value.is_some())
                    .unwrap_or(false),
            )
            .finish()
    }
}

impl Default for OpenAICompatibleProvider {
    fn default() -> Self {
        Self {
            settings: RwLock::new(None),
            secret_store: RwLock::new(None),
        }
    }
}

impl OpenAICompatibleProvider {
    fn current_settings(&self) -> CoreResult<ProviderSettings> {
        self.settings
            .read()
            .map_err(|_| CoreError::Model("provider settings lock poisoned".into()))?
            .clone()
            .ok_or_else(|| CoreError::Model("configure an OpenAI-compatible reasoning provider before starting an agent task".into()))
    }

    fn current_secret_store(&self) -> CoreResult<Arc<dyn SecretStore>> {
        self.secret_store
            .read()
            .map_err(|_| CoreError::Model("provider secret-store lock poisoned".into()))?
            .clone()
            .ok_or_else(|| CoreError::Model("provider secret store is unavailable".into()))
    }

    fn load_key(&self, settings: &ProviderSettings) -> CoreResult<Option<SecretBytes>> {
        if !settings.has_api_key {
            return Ok(None);
        }
        self.current_secret_store()?
            .get(&settings.credential_account())
    }

    fn endpoint_for(settings: &ProviderSettings) -> CoreResult<String> {
        let endpoint = if settings.provider == "openai" && settings.endpoint.trim().is_empty() {
            DEFAULT_OPENAI_ENDPOINT.to_string()
        } else {
            settings.endpoint.trim().trim_end_matches('/').to_string()
        };
        if endpoint.is_empty() {
            return Err(CoreError::InvalidAction(
                "an OpenAI-compatible endpoint is required".into(),
            ));
        }
        validate_provider_endpoint(&endpoint)?;
        if endpoint.ends_with("/chat/completions") {
            Ok(endpoint)
        } else {
            Ok(format!("{endpoint}/chat/completions"))
        }
    }

    async fn request_plan(
        &self,
        task_id: Uuid,
        settings: &ProviderSettings,
        api_key: Option<&SecretBytes>,
        system_prompt: &str,
        user_prompt: &str,
    ) -> CoreResult<ActionGraph> {
        let endpoint = Self::endpoint_for(settings)?;
        let client = reqwest::Client::builder()
            // Provider credentials must never be sent through ambient proxy
            // configuration that Sage cannot validate or disclose to users.
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECONDS))
            .build()
            .map_err(|error| CoreError::Model(format!("provider client unavailable: {error}")))?;

        let request = serde_json::json!({
            "model": settings.model,
            "temperature": 0,
            "max_tokens": 4096,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "sage_action_graph",
                    "strict": true,
                    "schema": draft_schema(),
                }
            }
        });
        let mut builder = client.post(endpoint).json(&request);
        if let Some(api_key) = api_key {
            let value =
                reqwest::header::HeaderValue::from_bytes(api_key.expose()).map_err(|_| {
                    CoreError::Model("provider credential contains invalid bytes".into())
                })?;
            let mut value = value;
            value.set_sensitive(true);
            builder = builder.header(reqwest::header::AUTHORIZATION, value);
        }
        let response = builder
            .send()
            .await
            .map_err(|error| CoreError::Model(provider_network_error(error)))?;
        if !response.status().is_success() {
            return Err(CoreError::Model(format!(
                "provider returned HTTP {} (structured JSON output may be unsupported)",
                response.status().as_u16()
            )));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_PROVIDER_RESPONSE_BYTES as u64)
        {
            return Err(CoreError::Model(
                "provider response exceeded the 1 MiB limit".into(),
            ));
        }
        let mut bytes = Vec::new();
        let mut response = response;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| CoreError::Model(provider_network_error(error)))?
        {
            if bytes.len().saturating_add(chunk.len()) > MAX_PROVIDER_RESPONSE_BYTES {
                return Err(CoreError::Model(
                    "provider response exceeded the 1 MiB limit".into(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        let envelope: ChatCompletion = serde_json::from_slice(&bytes)
            .map_err(|_| CoreError::Model("provider response was not valid JSON".into()))?;
        let content = envelope
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                CoreError::Model("provider did not return structured message content".into())
            })?;
        let draft: DraftPlan = serde_json::from_str(content).map_err(|_| {
            CoreError::Model("provider returned malformed Sage structured output".into())
        })?;
        draft_to_graph(draft, task_id)
    }
}

#[async_trait]
impl ModelProvider for OpenAICompatibleProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        let settings = self.settings.read().ok().and_then(|value| value.clone());
        match settings {
            Some(settings) => ProviderDescriptor {
                id: settings.provider,
                display_name: "OpenAI-compatible reasoning provider".into(),
                local: settings.endpoint.starts_with("http://localhost")
                    || settings.endpoint.starts_with("http://127.0.0.1")
                    || settings.endpoint.starts_with("http://[::1]"),
                roles: vec![ModelRole::Reasoning],
            },
            None => ProviderDescriptor {
                id: "unconfigured".into(),
                display_name: "No reasoning model configured".into(),
                local: true,
                roles: vec![ModelRole::Reasoning],
            },
        }
    }

    fn configure(
        &self,
        settings: Option<ProviderSettings>,
        secret_store: Arc<dyn SecretStore>,
    ) -> CoreResult<()> {
        *self
            .secret_store
            .write()
            .map_err(|_| CoreError::Model("provider secret-store lock poisoned".into()))? =
            Some(secret_store);
        let settings = settings
            .filter(|value| matches!(value.provider.as_str(), "openai" | "openai-compatible"));
        *self
            .settings
            .write()
            .map_err(|_| CoreError::Model("provider settings lock poisoned".into()))? = settings;
        Ok(())
    }

    async fn create_plan(&self, context: PlanningContext) -> CoreResult<ActionGraph> {
        let settings = self.current_settings()?;
        let api_key = self.load_key(&settings)?;
        if settings.provider == "openai" && api_key.is_none() {
            return Err(CoreError::Model(
                "an OpenAI API key is required before starting an agent task".into(),
            ));
        }
        let system_prompt = "You are Sage's planning model. Return only the requested JSON object. Your output is an untrusted draft: never include action IDs, credentials, authorization, or executable shell text outside action payloads. Use only the action kinds in the schema. Keep the plan minimal and safe; actions still pass through Sage policy, capability, approval, observation, and verification checks.";
        let user_prompt = serde_json::to_string(&serde_json::json!({
            "request": context.user_request,
            "current_state": context.current_state,
            "available_tools": context.available_tools,
            "trusted_constraints": context.trusted_constraints,
            "untrusted_context": context.untrusted_context,
        }))?;
        self.request_plan(
            context.task_id,
            &settings,
            api_key.as_ref(),
            system_prompt,
            &user_prompt,
        )
        .await
    }

    async fn replan(&self, context: ReplanContext) -> CoreResult<ActionGraph> {
        let settings = self.current_settings()?;
        let api_key = self.load_key(&settings)?;
        if settings.provider == "openai" && api_key.is_none() {
            return Err(CoreError::Model(
                "an OpenAI API key is required before replanning".into(),
            ));
        }
        let system_prompt = "You are Sage's recovery planning model. Return only a new JSON draft plan. Use fresh action positions, do not include IDs or provenance, and never bypass approval or policy. Only return structured JSON matching the supplied schema.";
        let user_prompt = serde_json::to_string(&serde_json::json!({
            "task": context.task,
            "failed_action_id": context.failed_action_id,
            "observation": context.observation,
            "attempt": context.attempt,
        }))?;
        self.request_plan(
            context.task.id,
            &settings,
            api_key.as_ref(),
            system_prompt,
            &user_prompt,
        )
        .await
    }

    async fn test_connection(
        &self,
        settings: ProviderSettings,
        api_key: Option<SecretBytes>,
    ) -> CoreResult<String> {
        validate_settings(&settings)?;
        if settings.provider == "openai" && api_key.is_none() && !settings.has_api_key {
            return Err(CoreError::Model(
                "an OpenAI API key is required to test OpenAI".into(),
            ));
        }
        let key = match api_key {
            Some(key) => Some(key),
            None => self.load_key(&settings)?,
        };
        let graph = self
            .request_plan(
                Uuid::new_v4(),
                &settings,
                key.as_ref(),
                "Return a minimal valid Sage plan with one ask_user action. This is a connectivity and structured-output test; do not perform any external action.",
                "Return the minimal connectivity test plan.",
            )
            .await?;
        if graph.nodes.is_empty() {
            return Err(CoreError::Model(
                "provider returned an empty structured plan".into(),
            ));
        }
        Ok(format!(
            "Connected to {} ({})",
            settings.provider, settings.model
        ))
    }
}

#[derive(Debug, Deserialize)]
struct ChatCompletion {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct DraftPlan {
    goal: String,
    actions: Vec<DraftAction>,
}

#[derive(Debug, Deserialize)]
struct DraftAction {
    kind: String,
    payload: serde_json::Value,
    target_resource: String,
    expected_outcome: serde_json::Value,
    depends_on: Vec<usize>,
}

fn draft_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["goal", "actions"],
        "properties": {
            "goal": {"type": "string", "minLength": 1, "maxLength": 4096},
            "actions": {
                "type": "array", "minItems": 1, "maxItems": MAX_PLAN_ACTIONS,
                "items": {
                    "type": "object", "additionalProperties": false,
                    "required": ["kind", "payload", "target_resource", "expected_outcome", "depends_on"],
                    "properties": {
                        "kind": {"type": "string", "enum": [
                            "open_application", "close_application", "read_file", "write_file", "move_file", "delete_file", "create_folder", "click_element", "type_text", "press_shortcut", "navigate_url", "download_file", "upload_file", "send_message", "submit_form", "run_command", "install_application", "change_setting", "wait_for_condition", "ask_user"
                        ]},
                        "payload": {"type": "object", "additionalProperties": true},
                        "target_resource": {"type": "string", "maxLength": 4096},
                        "expected_outcome": {"type": "object", "additionalProperties": true},
                        "depends_on": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": MAX_PLAN_ACTIONS}, "maxItems": MAX_PLAN_ACTIONS}
                    }
                }
            }
        }
    })
}

fn draft_to_graph(draft: DraftPlan, task_id: Uuid) -> CoreResult<ActionGraph> {
    if draft.goal.trim().is_empty() || draft.goal.len() > 4096 {
        return Err(CoreError::Model(
            "provider returned an invalid plan goal".into(),
        ));
    }
    if draft.actions.is_empty() || draft.actions.len() > MAX_PLAN_ACTIONS {
        return Err(CoreError::Model(
            "provider returned an invalid action count".into(),
        ));
    }
    let ids: Vec<Uuid> = (0..draft.actions.len()).map(|_| Uuid::new_v4()).collect();
    let mut nodes = Vec::with_capacity(draft.actions.len());
    for (index, draft_action) in draft.actions.into_iter().enumerate() {
        let action = action_from_draft(draft_action.kind, draft_action.payload)?;
        let expected_outcome = if draft_action.expected_outcome.is_null() {
            crate::domain::ExpectedOutcome::ExternalSuccess {
                marker: "provider-confirmed".into(),
            }
        } else {
            serde_json::from_value(draft_action.expected_outcome).map_err(|_| {
                CoreError::Model("provider returned an invalid expected outcome".into())
            })?
        };
        let mut dependencies = BTreeSet::new();
        for dependency in draft_action.depends_on {
            if dependency >= ids.len() || dependency == index {
                return Err(CoreError::Model(
                    "provider returned an invalid action dependency".into(),
                ));
            }
            dependencies.insert(ids[dependency]);
        }
        let proposal = crate::domain::ActionProposal {
            id: ids[index],
            task_id,
            action,
            expected_outcome,
            target_resource: draft_action.target_resource,
            provenance: crate::domain::Provenance::model(vec![task_id.to_string()]),
            metadata: BTreeMap::new(),
        };
        nodes.push(crate::domain::ActionNode {
            proposal,
            depends_on: dependencies,
        });
    }
    let graph = ActionGraph {
        goal: draft.goal,
        nodes,
    };
    graph.validate(task_id).map_err(|error| {
        CoreError::Model(format!("provider plan failed Sage validation: {error}"))
    })?;
    Ok(graph)
}

fn action_from_draft(
    kind: String,
    payload: serde_json::Value,
) -> CoreResult<crate::domain::Action> {
    let mut object = payload
        .as_object()
        .cloned()
        .ok_or_else(|| CoreError::Model("provider action payload must be an object".into()))?;
    object.insert("type".into(), serde_json::Value::String(kind));
    serde_json::from_value(serde_json::Value::Object(object)).map_err(|_| {
        CoreError::Model("provider returned an action payload that Sage does not recognize".into())
    })
}

fn validate_settings(settings: &ProviderSettings) -> CoreResult<()> {
    if settings.role != "reasoning" {
        return Err(CoreError::InvalidAction(
            "only the reasoning provider role is supported".into(),
        ));
    }
    if !matches!(settings.provider.as_str(), "openai" | "openai-compatible") {
        return Err(CoreError::InvalidAction("unsupported provider".into()));
    }
    if settings.model.trim().is_empty() || settings.model.len() > 256 {
        return Err(CoreError::InvalidAction(
            "model name must contain between 1 and 256 characters".into(),
        ));
    }
    if settings.endpoint.len() > 2_048 {
        return Err(CoreError::InvalidAction(
            "provider endpoint exceeds 2,048 characters".into(),
        ));
    }
    if settings.provider == "openai-compatible" && settings.endpoint.trim().is_empty() {
        return Err(CoreError::InvalidAction(
            "an OpenAI-compatible endpoint is required".into(),
        ));
    }
    if !settings.endpoint.trim().is_empty() {
        validate_provider_endpoint(&settings.endpoint)?;
    }
    Ok(())
}

pub fn validate_provider_endpoint(endpoint: &str) -> CoreResult<()> {
    let parsed = url::Url::parse(endpoint.trim())
        .map_err(|_| CoreError::InvalidAction("provider endpoint is not a valid URL".into()))?;
    if parsed.username() != ""
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(CoreError::InvalidAction(
            "provider endpoint must not contain credentials, query parameters, or fragments".into(),
        ));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| CoreError::InvalidAction("provider endpoint must include a host".into()))?
        .to_ascii_lowercase();
    match parsed.scheme() {
        "https" => Ok(()),
        "http" if matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1" | "[::1]") => Ok(()),
        _ => Err(CoreError::InvalidAction(
            "provider endpoints must use HTTPS, or HTTP only for localhost, 127.0.0.1, or [::1]"
                .into(),
        )),
    }
}

fn provider_network_error(error: reqwest::Error) -> String {
    if error.is_timeout() {
        "provider request timed out".into()
    } else if error.is_redirect() {
        "provider redirects are not allowed".into()
    } else {
        "provider request failed".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::testing::MemorySecretStore;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    #[test]
    fn endpoint_validation_allows_tls_and_exact_loopback_only() {
        assert!(validate_provider_endpoint("https://api.example.test/v1").is_ok());
        assert!(validate_provider_endpoint("http://localhost:11434/v1").is_ok());
        assert!(validate_provider_endpoint("http://127.0.0.1:11434/v1").is_ok());
        assert!(validate_provider_endpoint("http://[::1]:11434/v1").is_ok());
        assert!(validate_provider_endpoint("http://127.0.0.1.evil.test/v1").is_err());
        assert!(validate_provider_endpoint("http://example.test/v1").is_err());
        assert!(validate_provider_endpoint("https://user:secret@example.test/v1").is_err());
        assert!(validate_provider_endpoint("https://example.test/v1?api_key=secret").is_err());
    }

    #[tokio::test]
    async fn mock_openai_response_is_structured_and_sage_assigns_identity() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let response_body = serde_json::json!({
            "choices": [{
                "message": {"content": serde_json::json!({
                    "goal": "Ask for confirmation",
                    "actions": [{
                        "kind": "ask_user",
                        "payload": {"question": "Continue?"},
                        "target_resource": "user",
                        "expected_outcome": {"kind": "user_answered"},
                        "depends_on": []
                    }]
                }).to_string()}
            }]
        })
        .to_string();
        let response_body_for_server = response_body.clone();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 8 * 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut request).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body_for_server.len(),
                response_body_for_server
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        let provider = OpenAICompatibleProvider::default();
        provider
            .configure(
                Some(ProviderSettings {
                    role: "reasoning".into(),
                    provider: "openai-compatible".into(),
                    model: "mock-model".into(),
                    endpoint: format!("http://{address}/v1"),
                    has_api_key: false,
                }),
                Arc::new(MemorySecretStore::default()),
            )
            .unwrap();
        let task_id = Uuid::new_v4();
        let graph = provider
            .create_plan(PlanningContext {
                task_id,
                user_request: "confirm".into(),
                current_state: serde_json::json!({}),
                available_tools: Vec::new(),
                trusted_constraints: Vec::new(),
                untrusted_context: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].proposal.task_id, task_id);
        assert_ne!(graph.nodes[0].proposal.id, Uuid::nil());
        assert_eq!(
            graph.nodes[0].proposal.provenance.source,
            crate::domain::ProvenanceSource::Model
        );
        assert!(graph.validate(task_id).is_ok());
    }

    #[tokio::test]
    async fn malformed_structured_content_is_rejected_without_secret_echo() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 8 * 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut request).await;
            let body = r#"{"choices":[{"message":{"content":"not-json"}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let provider = OpenAICompatibleProvider::default();
        provider
            .configure(
                Some(ProviderSettings {
                    role: "reasoning".into(),
                    provider: "openai-compatible".into(),
                    model: "mock-model".into(),
                    endpoint: format!("http://{address}/v1"),
                    has_api_key: true,
                }),
                {
                    let store = Arc::new(MemorySecretStore::default());
                    store
                        .set(
                            "provider:reasoning:openai-compatible",
                            &SecretBytes::new(b"do-not-echo".to_vec()),
                        )
                        .unwrap();
                    store
                },
            )
            .unwrap();
        let error = provider
            .create_plan(PlanningContext {
                task_id: Uuid::new_v4(),
                user_request: "confirm".into(),
                current_state: serde_json::json!({}),
                available_tools: Vec::new(),
                trusted_constraints: Vec::new(),
                untrusted_context: Vec::new(),
            })
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("malformed Sage structured output"));
        assert!(!error.contains("do-not-echo"));
    }

    #[tokio::test]
    async fn provider_redirects_are_rejected_without_following_location() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 8 * 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut request).await;
            let response = "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:9/redirect-target\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        let provider = OpenAICompatibleProvider::default();
        provider
            .configure(
                Some(ProviderSettings {
                    role: "reasoning".into(),
                    provider: "openai-compatible".into(),
                    model: "mock-model".into(),
                    endpoint: format!("http://{address}/v1"),
                    has_api_key: false,
                }),
                Arc::new(MemorySecretStore::default()),
            )
            .unwrap();
        let error = provider
            .create_plan(PlanningContext {
                task_id: Uuid::new_v4(),
                user_request: "redirect test".into(),
                current_state: serde_json::json!({}),
                available_tools: Vec::new(),
                trusted_constraints: Vec::new(),
                untrusted_context: Vec::new(),
            })
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("HTTP 302"));
    }
}
