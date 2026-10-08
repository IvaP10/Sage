#![forbid(unsafe_code)]

pub mod agency;
pub mod application_target;
pub mod audit;
pub mod authorization;
pub mod browser_target;
pub mod capability;
mod commands;
pub mod compiler;
pub mod config;
pub mod context;
pub mod contracts;
pub mod decisions;
pub mod domain;
pub(crate) mod effect_ownership;
pub mod engine;
pub mod error;
pub mod events;
pub mod execution;
pub mod features;
mod finalization;
#[cfg(feature = "qwen35-evaluation")]
pub mod inference_cpu;
#[cfg(feature = "qwen35-evaluation")]
pub mod inference_lane;
#[cfg(feature = "qwen35-evaluation")]
pub mod inference_resources;
pub mod intent;
pub mod ipc;
pub mod journal;
pub mod knowledge;
mod learning;
pub mod model;
#[cfg(feature = "qwen35-evaluation")]
pub mod model_package;
pub mod network;
pub mod observation;
#[cfg(feature = "qwen35-evaluation")]
mod planner_schema;
pub mod policy;
mod preparation;
pub mod procedure_stream;
#[cfg(feature = "qwen35-evaluation")]
pub mod qwen35;
#[cfg(feature = "qwen35-evaluation")]
pub mod qwen35_vision;
#[cfg(feature = "qwen35-evaluation")]
pub mod qwen_prompt;
#[cfg(feature = "qwen35-evaluation")]
pub mod qwen_tokenizer;
mod receipts;
mod reconciliation;
pub mod redaction;
pub mod resources;
mod runtime;
#[cfg(feature = "qwen35-evaluation")]
pub mod safetensors;
mod scheduling;
pub mod secrets;
pub mod storage;
#[cfg(feature = "qwen35-evaluation")]
pub mod streaming;
#[cfg(feature = "qwen35-evaluation")]
pub mod structured_decode;
mod transitions;
mod undo;
pub mod vault;
pub mod verification;
pub mod workflows;
pub mod world_model;

pub use engine::SageCore;
pub use error::{CoreError, CoreResult};
