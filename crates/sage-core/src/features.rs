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
        Action::ListDirectory {
            path: p,
            page_size,
            cursor,
        } => {
            path(p)
                && (1..=crate::execution::directory::MAX_PAGE_ENTRIES).contains(page_size)
                && cursor.as_ref().is_none_or(|cursor| bounded(cursor))
        }
        Action::WriteFile {
            path: p, content, ..
        } => path(p) && content.len() <= 1_048_576,
        Action::CreateFolder { path: p } => path(p),
        Action::OpenApplication { application } => bounded(application),
        Action::SetApplicationControl {
            application,
            system_id,
            system_fingerprint,
            capability_id,
            control_id,
            value,
        } => {
            bounded(application)
                && !system_id.is_nil()
                && valid_lower_hex_digest(system_fingerprint)
                && !capability_id.is_empty()
                && capability_id.len() <= 128
                && capability_id.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'.' | b'_' | b'-' | b':')
                })
                && valid_lower_hex_digest(control_id)
                && value.validate().is_ok()
        }
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
    pub supported_platforms: Vec<String>,
}

pub fn manifests() -> Vec<FeatureManifest> {
    let string = || json!({"type":"string","minLength":1,"maxLength":4096});
    let schema = |properties: Value, required: Vec<&str>| json!({"type":"object","additionalProperties":false,"properties":properties,"required":required});
    [
        ("fetch_public","native",vec!["read","network"],"TLS response URL, status and content digest",schema(json!({"url":string(),"max_bytes":{"type":"integer","minimum":1,"maximum":1048576}}),vec!["url","max_bytes"])),
        ("read_file","native",vec!["read"],"bounded regular-file read",schema(json!({"path":string(),"max_bytes":{"type":"integer","minimum":1,"maximum":16777216}}),vec!["path","max_bytes"])),
        ("list_directory","native",vec!["read"],"fresh directory identity and exact bounded page; continue with next_cursor until null",schema(json!({"path":string(),"page_size":{"type":"integer","minimum":1,"maximum":64},"cursor":{"type":["string","null"],"maxLength":4096}}),vec!["path","page_size","cursor"])),
        ("write_file","native",vec!["create","modify"],"exact target content hash",schema(json!({"path":string(),"content":{"type":"string","maxLength":1048576},"overwrite":{"type":"boolean"}}),vec!["path","content","overwrite"])),
        ("create_folder","native",vec!["create"],"directory identity",schema(json!({"path":string()}),vec!["path"])),
        ("open_application","native",vec!["control_application"],"application identity and running state",schema(json!({"application":string()}),vec!["application"])),
        ("set_application_control","native",vec!["control_application"],"fresh independent accessibility value readback for the exact signed foreground application and learned control",schema(json!({"application":string(),"system_id":{"type":"string","format":"uuid"},"system_fingerprint":{"type":"string","pattern":"^[0-9a-f]{64}$"},"capability_id":{"type":"string","minLength":1,"maxLength":128},"control_id":{"type":"string","pattern":"^[0-9a-f]{64}$"},"value":{"type":["boolean","number"]}}),vec!["application","system_id","system_fingerprint","capability_id","control_id","value"])),
        ("navigate_url","browser",vec!["external_commitment"],"paired document and exact URL",schema(json!({"url":string(),"new_tab":{"type":"boolean","enum":[false]}}),vec!["url","new_tab"])),
        ("ask_user","user_interaction",vec![],"authenticated user reply",schema(json!({"question":string()}),vec!["question"])),
    ].into_iter().map(|(id,executor,effects,verifier,input_schema)|FeatureManifest {
        id:id.into(),version:2,input_schema,executor:executor.into(),effects:effects.into_iter().map(String::from).collect(),
        verifier:verifier.into(),max_duration_ms:30_000,
        enabled:!matches!(id, "open_application" | "set_application_control") || cfg!(target_os = "macos"),
        supported_platforms:if matches!(id, "open_application" | "set_application_control") { vec!["macos".into()] } else { vec!["macos".into(), "windows".into()] },
    }).collect()
}
fn valid_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
            supported_platforms: m.supported_platforms,
            requires_confirmation: true,
            executor: m.executor,
            timeout_ms: m.max_duration_ms,
            verification_strategy: m.verifier,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Action, ApplicationControlValue};
    use uuid::Uuid;

    fn action(value: ApplicationControlValue) -> Action {
        Action::SetApplicationControl {
            application: "com.example.Editor".into(),
            system_id: Uuid::new_v4(),
            system_fingerprint: "a".repeat(64),
            capability_id: "ui.control.slider.0123456789abcdef01234567".into(),
            control_id: "b".repeat(64),
            value,
        }
    }

    #[test]
    fn learned_controls_accept_only_bounded_identity_and_finite_typed_values() {
        assert!(validate(&action(ApplicationControlValue::Boolean(true))).is_ok());
        assert!(validate(&action(ApplicationControlValue::Number(0.25))).is_ok());
        assert!(validate(&action(ApplicationControlValue::Number(f64::INFINITY))).is_err());
        assert!(
            validate(&Action::SetApplicationControl {
                application: "com.example.Editor".into(),
                system_id: Uuid::nil(),
                system_fingerprint: "a".repeat(64),
                capability_id: "ui.control.slider.0123456789abcdef01234567".into(),
                control_id: "b".repeat(64),
                value: ApplicationControlValue::Boolean(false),
            })
            .is_err()
        );
    }

    #[test]
    fn learned_control_is_advertised_only_on_its_qualified_native_platform() {
        let manifest = manifests()
            .into_iter()
            .find(|manifest| manifest.id == "set_application_control")
            .unwrap();
        assert_eq!(manifest.enabled, cfg!(target_os = "macos"));
        assert_eq!(manifest.supported_platforms, ["macos"]);
        assert_eq!(
            manifest.input_schema["properties"]["value"]["type"],
            json!(["boolean", "number"])
        );
    }
}
