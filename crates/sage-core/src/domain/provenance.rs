use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceSource {
    User,
    SageCore,
    Model,
    OperatingSystem,
    Application,
    Browser,
    Document,
    Message,
    Terminal,
    Plugin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustClass {
    UserAuthority,
    TrustedComponent,
    Observation,
    UntrustedExternalContent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub source: ProvenanceSource,
    pub trust: TrustClass,
    pub source_id: Option<String>,
    pub parent_ids: Vec<String>,
}

impl Provenance {
    pub fn user() -> Self {
        Self {
            source: ProvenanceSource::User,
            trust: TrustClass::UserAuthority,
            source_id: None,
            parent_ids: Vec::new(),
        }
    }

    pub fn model(parent_ids: Vec<String>) -> Self {
        Self {
            source: ProvenanceSource::Model,
            trust: TrustClass::TrustedComponent,
            source_id: None,
            parent_ids,
        }
    }

    pub fn external(source: ProvenanceSource, source_id: impl Into<String>) -> Self {
        Self {
            source,
            trust: TrustClass::UntrustedExternalContent,
            source_id: Some(source_id.into()),
            parent_ids: Vec::new(),
        }
    }

    pub fn carries_user_authority(&self) -> bool {
        self.trust == TrustClass::UserAuthority
    }
}
