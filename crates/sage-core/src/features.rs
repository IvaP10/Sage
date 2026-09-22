//! The feature registry is shared by discovery and validation. Adding a tool
//! requires a concrete schema, an execution boundary and a trusted verifier.
use crate::model::ToolDescriptor;
use crate::{CoreError, CoreResult};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Independent validation for the closed built-in schema set. The model's
/// grammar and a provider's "strict" setting are never an enforcement point.
pub fn validate(action: &crate::domain::Action) -> CoreResult<()> {
    use crate::domain::Action;
    let bounded = |s: &str| max_text(s, 4096);
    let path = |p: &std::path::Path| p.to_str().is_some_and(bounded);
    let valid = match action {
        Action::FetchPublic { url, max_bytes } => {
            bounded(url) && (1..=1_048_576).contains(max_bytes)
        }
        Action::ReadFile { path: p, max_bytes } => path(p) && (1..=16_777_216).contains(max_bytes),
        Action::WriteFile {
            path: p, content, ..
        } => path(p) && content.len() <= 1_048_576,
        Action::CreateFolder { path: p } => path(p),
        Action::OpenApplication { application } => bounded(application),
        Action::NavigateUrl { url, new_tab } => bounded(url) && !new_tab,
        Action::AskUser { question } => bounded(question),
        _ => false,
    };
    if !valid {
        return Err(CoreError::InvalidAction(
            "Action does not satisfy an enabled feature schema".into(),
        ));
    }
    Ok(())
}
fn max_text(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max && !value.contains('\0')
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureManifest {
    pub id: String,
    pub version: u32,
    pub input_schema: Value,
    pub executor: String,
    pub effects: Vec<String>,
    pub verifier: String,
    pub max_duration_ms: u64,
    pub enabled: bool,
}

pub fn manifests() -> Vec<FeatureManifest> {
    let string = || json!({"type":"string","minLength":1,"maxLength":4096});
    let schema = |properties: Value, required: Vec<&str>| json!({"type":"object","additionalProperties":false,"properties":properties,"required":required});
    [
        ("fetch_public","native",vec!["read","network"],"TLS response URL, status and content digest",schema(json!({"url":string(),"max_bytes":{"type":"integer","minimum":1,"maximum":1048576}}),vec!["url","max_bytes"])),
        ("read_file","native",vec!["read"],"bounded regular-file read",schema(json!({"path":string(),"max_bytes":{"type":"integer","minimum":1,"maximum":16777216}}),vec!["path","max_bytes"])),
        ("write_file","native",vec!["create","modify"],"exact target content hash",schema(json!({"path":string(),"content":{"type":"string","maxLength":1048576},"overwrite":{"type":"boolean"}}),vec!["path","content","overwrite"])),
        ("create_folder","native",vec!["create"],"directory identity",schema(json!({"path":string()}),vec!["path"])),
        ("open_application","native",vec!["control_application"],"application identity and running state",schema(json!({"application":string()}),vec!["application"])),
        ("navigate_url","browser",vec!["external_commitment"],"paired document and exact URL",schema(json!({"url":string(),"new_tab":{"type":"boolean","enum":[false]}}),vec!["url","new_tab"])),
        ("ask_user","user_interaction",vec![],"authenticated user reply",schema(json!({"question":string()}),vec!["question"])),
    ].into_iter().map(|(id,executor,effects,verifier,input_schema)|FeatureManifest {
        id:id.into(),version:2,input_schema,executor:executor.into(),effects:effects.into_iter().map(String::from).collect(),
        verifier:verifier.into(),max_duration_ms:30_000,enabled:true,
    }).collect()
}

pub fn descriptors() -> Vec<ToolDescriptor> {
    manifests()
        .into_iter()
        .filter(|m| m.enabled)
        .map(|m| ToolDescriptor {
            name: m.id,
            version: m.version.to_string(),
            input_schema: m.input_schema,
            output_schema: json!({"type":"object","required":["verdict","summary","output"]}),
            risk: "broker-classified".into(),
            required_capabilities: m.effects,
            supported_platforms: vec!["macos".into(), "windows".into()],
            requires_confirmation: true,
            executor: m.executor,
            timeout_ms: m.max_duration_ms,
            verification_strategy: m.verifier,
        })
        .collect()
}
