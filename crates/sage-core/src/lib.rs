#![forbid(unsafe_code)]

pub mod capability;
pub mod compiler;
pub mod config;
pub mod domain;
pub mod engine;
pub mod error;
pub mod events;
pub mod execution;
pub mod ipc;
pub mod model;
pub mod observation;
pub mod policy;
pub mod redaction;
pub mod resources;
pub mod secrets;
pub mod storage;
pub mod verification;

pub use engine::SageCore;
pub use error::{CoreError, CoreResult};
