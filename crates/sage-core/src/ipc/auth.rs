use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::error::{CoreError, CoreResult};
use crate::secrets::SecretBytes;

type HmacSha256 = Hmac<Sha256>;
const DOMAIN: &[u8] = b"SAGE-LOCAL-IPC-AUTH-V2\0";

pub struct FeatureAuthentication<'a> {
    pub protocol_version: u32,
    pub client_kind: i32,
    pub client_version: &'a str,
    pub features: &'a [String],
}

#[derive(Debug)]
pub struct IpcAuthenticator {
    secret: SecretBytes,
    browser_secret: SecretBytes,
}

impl IpcAuthenticator {
    pub fn new(secret: SecretBytes) -> Self {
        let browser_secret = derive_browser_secret(&secret);
        Self {
            secret,
            browser_secret,
        }
    }

    fn role_secret(&self, kind: i32) -> CoreResult<&SecretBytes> {
        match sage_protocol::sage::ipc::v2::ClientKind::try_from(kind) {
            Ok(sage_protocol::sage::ipc::v2::ClientKind::Browser) => Ok(&self.browser_secret),
            Ok(sage_protocol::sage::ipc::v2::ClientKind::Macos) if cfg!(target_os = "macos") => {
                Ok(&self.secret)
            }
            Ok(sage_protocol::sage::ipc::v2::ClientKind::Windows) if cfg!(windows) => {
                Ok(&self.secret)
            }
            #[cfg(test)]
            Ok(sage_protocol::sage::ipc::v2::ClientKind::Macos) => Ok(&self.secret),
            _ => Err(CoreError::AuthenticationFailed),
        }
    }

    pub fn server_proof(
        &self,
        kind: i32,
        client_proof: &[u8],
        session: &str,
    ) -> CoreResult<[u8; 32]> {
        server_authentication_proof(self.role_secret(kind)?.expose(), client_proof, session)
    }

    pub fn server_proof_with_features(
        &self,
        kind: i32,
        client_proof: &[u8],
        session: &str,
        features: &[String],
    ) -> CoreResult<[u8; 32]> {
        server_authentication_proof_with_features(
            self.role_secret(kind)?.expose(),
            client_proof,
            session,
            features,
        )
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
            self.role_secret(client_kind)?.expose(),
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

    pub fn verify_with_features(
        &self,
        server_nonce: &[u8],
        client_nonce: &[u8],
        authentication: FeatureAuthentication<'_>,
        proof: &[u8],
    ) -> CoreResult<()> {
        if server_nonce.len() != 32 || client_nonce.len() != 32 || proof.len() != 32 {
            return Err(CoreError::AuthenticationFailed);
        }
        validate_features(authentication.features)?;
        let expected = authentication_proof_with_features(
            self.role_secret(authentication.client_kind)?.expose(),
            server_nonce,
            client_nonce,
            authentication.protocol_version,
            authentication.client_kind,
            authentication.client_version,
            authentication.features,
        )?;
        if expected.ct_eq(proof).into() {
            Ok(())
        } else {
            Err(CoreError::AuthenticationFailed)
        }
    }
}

pub fn derive_browser_secret(secret: &SecretBytes) -> SecretBytes {
    let mut mac = HmacSha256::new_from_slice(secret.expose()).expect("HMAC accepts any key length");
    mac.update(b"SAGE-BROWSER-ROLE-V2\0");
    SecretBytes::new(mac.finalize().into_bytes().to_vec())
}

pub fn server_authentication_proof(
    secret: &[u8],
    client_proof: &[u8],
    session: &str,
) -> CoreResult<[u8; 32]> {
    server_authentication_proof_with_features(secret, client_proof, session, &[])
}

pub fn server_authentication_proof_with_features(
    secret: &[u8],
    client_proof: &[u8],
    session: &str,
    features: &[String],
) -> CoreResult<[u8; 32]> {
    validate_features(features)?;
    let mut mac =
        HmacSha256::new_from_slice(secret).map_err(|_| CoreError::AuthenticationFailed)?;
    mac.update(b"SAGE-CORE-PROOF-V2\0");
    mac.update(client_proof);
    mac.update(&(session.len() as u32).to_be_bytes());
    mac.update(session.as_bytes());
    update_feature_transcript(&mut mac, features);
    Ok(mac.finalize().into_bytes().into())
}

pub fn authentication_proof(
    secret: &[u8],
    server_nonce: &[u8],
    client_nonce: &[u8],
    protocol_version: u32,
    client_kind: i32,
    client_version: &str,
) -> CoreResult<[u8; 32]> {
    authentication_proof_with_features(
        secret,
        server_nonce,
        client_nonce,
        protocol_version,
        client_kind,
        client_version,
        &[],
    )
}

pub fn authentication_proof_with_features(
    secret: &[u8],
    server_nonce: &[u8],
    client_nonce: &[u8],
    protocol_version: u32,
    client_kind: i32,
    client_version: &str,
    features: &[String],
) -> CoreResult<[u8; 32]> {
    validate_features(features)?;
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|_| CoreError::Protocol("invalid IPC authentication key".into()))?;
    mac.update(DOMAIN);
    mac.update(server_nonce);
    mac.update(client_nonce);
    mac.update(&protocol_version.to_be_bytes());
    mac.update(&client_kind.to_be_bytes());
    mac.update(&(client_version.len() as u32).to_be_bytes());
    mac.update(client_version.as_bytes());
    update_feature_transcript(&mut mac, features);
    Ok(mac.finalize().into_bytes().into())
}

fn validate_features(features: &[String]) -> CoreResult<()> {
    if features.len() > 32
        || features.iter().any(|feature| {
            feature.is_empty()
                || feature.len() > 96
                || !feature.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || byte == b'_'
                        || byte == b'.'
                })
        })
        || features.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(CoreError::AuthenticationFailed);
    }
    Ok(())
}

fn update_feature_transcript(mac: &mut HmacSha256, features: &[String]) {
    if features.is_empty() {
        // Empty capability lists retain the v2 transcript for older clients.
        return;
    }
    mac.update(b"SAGE-IPC-FEATURES-V1\0");
    mac.update(&(features.len() as u32).to_be_bytes());
    for feature in features {
        mac.update(&(feature.len() as u32).to_be_bytes());
        mac.update(feature.as_bytes());
    }
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

    #[test]
    fn browser_key_cannot_authenticate_as_native_ui() {
        let root = SecretBytes::new(vec![7; 32]);
        let browser = derive_browser_secret(&root);
        let authenticator = IpcAuthenticator::new(root);
        let proof = authentication_proof(browser.expose(), &[1; 32], &[2; 32], 2, 1, "2").unwrap();
        assert!(
            authenticator
                .verify(&[1; 32], &[2; 32], 2, 1, "2", &proof)
                .is_err()
        );
        let proof = authentication_proof(browser.expose(), &[1; 32], &[2; 32], 2, 4, "2").unwrap();
        authenticator
            .verify(&[1; 32], &[2; 32], 2, 4, "2", &proof)
            .unwrap();
        assert_ne!(
            server_authentication_proof(browser.expose(), &proof, "a").unwrap(),
            server_authentication_proof(browser.expose(), &proof, "b").unwrap()
        );
    }
}
