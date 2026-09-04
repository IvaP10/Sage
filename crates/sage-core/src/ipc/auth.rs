use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::error::{CoreError, CoreResult};
use crate::secrets::SecretBytes;

type HmacSha256 = Hmac<Sha256>;
const DOMAIN: &[u8] = b"SAGE-LOCAL-IPC-AUTH-V1\0";

#[derive(Debug)]
pub struct IpcAuthenticator {
    secret: SecretBytes,
}

impl IpcAuthenticator {
    pub fn new(secret: SecretBytes) -> Self {
        Self { secret }
    }

    pub fn verify(
        &self,
        server_nonce: &[u8],
        client_nonce: &[u8],
        protocol_version: u32,
        client_kind: i32,
        client_version: &str,
        proof: &[u8],
    ) -> CoreResult<()> {
        if server_nonce.len() != 32 || client_nonce.len() != 32 || proof.len() != 32 {
            return Err(CoreError::AuthenticationFailed);
        }
        let expected = authentication_proof(
            self.secret.expose(),
            server_nonce,
            client_nonce,
            protocol_version,
            client_kind,
            client_version,
        )?;
        if expected.ct_eq(proof).into() {
            Ok(())
        } else {
            Err(CoreError::AuthenticationFailed)
        }
    }
}

pub fn authentication_proof(
    secret: &[u8],
    server_nonce: &[u8],
    client_nonce: &[u8],
    protocol_version: u32,
    client_kind: i32,
    client_version: &str,
) -> CoreResult<[u8; 32]> {
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|_| CoreError::Protocol("invalid IPC authentication key".into()))?;
    mac.update(DOMAIN);
    mac.update(server_nonce);
    mac.update(client_nonce);
    mac.update(&protocol_version.to_be_bytes());
    mac.update(&client_kind.to_be_bytes());
    mac.update(&(client_version.len() as u32).to_be_bytes());
    mac.update(client_version.as_bytes());
    Ok(mac.finalize().into_bytes().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_is_bound_to_client_and_protocol() {
        let proof = authentication_proof(&[7; 32], &[1; 32], &[2; 32], 1, 1, "1.0.0").unwrap();
        let authenticator = IpcAuthenticator::new(SecretBytes::new(vec![7; 32]));
        authenticator
            .verify(&[1; 32], &[2; 32], 1, 1, "1.0.0", &proof)
            .unwrap();
        assert!(
            authenticator
                .verify(&[1; 32], &[2; 32], 2, 1, "1.0.0", &proof)
                .is_err()
        );
    }
}
