#![forbid(unsafe_code)]

pub mod audit;
pub mod authorization;
pub mod browser_target;
pub mod capability;
pub mod compiler;
pub mod config;
pub mod context;
pub mod contracts;
pub mod domain;
pub mod engine;
pub mod error;
pub mod events;
pub mod execution;
pub mod features;
pub mod inference;
pub mod ipc;
pub mod journal;
pub mod knowledge;
pub mod model;
pub mod network;
pub mod observation;
pub mod policy;
pub mod redaction;
pub mod resources;
pub mod secrets;
pub mod storage;
pub mod streaming;
pub mod vault;
pub mod verification;
pub mod workflows;

pub use engine::SageCore;
pub use error::{CoreError, CoreResult};
