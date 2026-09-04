mod auth;
mod codec;
mod server;

pub use auth::{IpcAuthenticator, authentication_proof};
pub use server::serve;
