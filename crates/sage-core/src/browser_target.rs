//! A paired top-level document is a concrete target, not a browser-wide grant.
use crate::{CoreError, CoreResult};
use sage_protocol::sage::ipc::v2 as wire;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BrowserTarget {
    pub tab_id: i32,
    pub window_id: i32,
    pub frame_id: i32,
    pub document_id: String,
    pub navigation_generation: u64,
    pub origin: String,
    pub url: String,
}
impl BrowserTarget {
    pub fn validate(&self) -> CoreResult<()> {
        let url = url::Url::parse(&self.url)
            .map_err(|_| CoreError::InvalidAction("Invalid browser target URL".into()))?;
        if self.tab_id < 0
            || self.window_id < 0
            || self.frame_id != 0
            || self.document_id.is_empty()
            || self.document_id.len() > 128
            || self.navigation_generation == 0
            || self.navigation_generation > 9_007_199_254_740_991
            || !matches!(url.scheme(), "http" | "https")
            || url.origin().ascii_serialization() != self.origin
            || !url.username().is_empty()
            || url.password().is_some()
            || self.url.len() > 8192
        {
            return Err(CoreError::InvalidAction(
                "Browser target identity is incomplete or invalid".into(),
            ));
        }
        Ok(())
    }
    pub fn to_wire(&self) -> wire::BrowserTarget {
        wire::BrowserTarget {
            tab_id: self.tab_id,
            window_id: self.window_id,
            frame_id: self.frame_id,
            document_id: self.document_id.clone(),
            navigation_generation: self.navigation_generation,
            origin: self.origin.clone(),
            url: self.url.clone(),
        }
    }
    pub fn from_wire(value: &wire::BrowserTarget) -> CoreResult<Self> {
        let target = Self {
            tab_id: value.tab_id,
            window_id: value.window_id,
            frame_id: value.frame_id,
            document_id: value.document_id.clone(),
            navigation_generation: value.navigation_generation,
            origin: value.origin.clone(),
            url: value.url.clone(),
        };
        target.validate()?;
        Ok(target)
    }
}
