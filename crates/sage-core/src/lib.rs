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
pub mod goal_coordinator;
#[cfg(feature = "qwen35-evaluation")]
pub mod inference_lane;
pub mod inference_worker_process;
#[cfg(feature = "qwen35-worker-generate")]
pub mod inference_worker_provider;
pub mod intent;
pub mod ipc;
pub mod journal;
pub mod knowledge;
mod learning;
pub mod model;
pub mod network;
pub mod observation;
pub mod peer_compute;
pub mod policy;
mod preparation;
pub mod procedure_stream;
#[cfg(feature = "qwen35-evaluation")]
pub mod qwen35_loader;
#[cfg(any(feature = "qwen35-evaluation", feature = "qwen35-worker-generate"))]
pub mod qwen_prompt;
mod receipts;
mod reconciliation;
pub mod redaction;
pub mod resources;
mod runtime;
mod scheduling;
pub mod secrets;
pub mod storage;
pub mod task_handoff;
mod transitions;
mod undo;
pub mod vault;
pub mod verification;
pub mod workflows;
pub mod world_model;

pub use engine::SageCore;
pub use error::{CoreError, CoreResult};
