mod auth;
mod codec;
pub(crate) mod dispatch;
mod server;
mod writer;

pub use auth::{
    IpcAuthenticator, authentication_proof, authentication_proof_with_features,
    derive_browser_secret, server_authentication_proof, server_authentication_proof_with_features,
};
pub use codec::{read_frame, write_frame};
pub use server::serve;
