use std::sync::Arc;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{CoreError, CoreResult};

const SERVICE: &str = "com.ivanpadeliya.sage";
const IPC_SECRET_ACCOUNT: &str = "local-ipc-v2";

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SecretBytes(Vec<u8>);

impl SecretBytes {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub fn expose(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Debug for SecretBytes {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SecretBytes([REDACTED])")
    }
}

pub trait SecretStore: Send + Sync {
    fn get(&self, account: &str) -> CoreResult<Option<SecretBytes>>;
    fn set(&self, account: &str, secret: &SecretBytes) -> CoreResult<()>;
    fn delete(&self, account: &str) -> CoreResult<()>;
}

#[derive(Debug, Default)]
pub struct OsSecretStore;

impl SecretStore for OsSecretStore {
    fn get(&self, account: &str) -> CoreResult<Option<SecretBytes>> {
        let entry = keyring::Entry::new(SERVICE, account)
            .map_err(|error| CoreError::SecretStore(error.to_string()))?;
        match entry.get_password() {
            Ok(encoded) => STANDARD
                .decode(encoded)
                .map(SecretBytes::new)
                .map(Some)
                .map_err(|_| CoreError::SecretStore("secure-store value was malformed".into())),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(error) => Err(CoreError::SecretStore(error.to_string())),
        }
    }

    fn set(&self, account: &str, secret: &SecretBytes) -> CoreResult<()> {
        let entry = keyring::Entry::new(SERVICE, account)
            .map_err(|error| CoreError::SecretStore(error.to_string()))?;
        entry
            .set_password(&STANDARD.encode(secret.expose()))
            .map_err(|error| CoreError::SecretStore(error.to_string()))
    }

    fn delete(&self, account: &str) -> CoreResult<()> {
        let entry = keyring::Entry::new(SERVICE, account)
            .map_err(|error| CoreError::SecretStore(error.to_string()))?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(CoreError::SecretStore(error.to_string())),
        }
    }
}

pub fn load_or_create_ipc_secret(store: Arc<dyn SecretStore>) -> CoreResult<SecretBytes> {
    if let Some(secret) = store.get(IPC_SECRET_ACCOUNT)? {
        if secret.expose().len() != 32 {
            return Err(CoreError::SecretStore(
                "local IPC key has an invalid length".into(),
            ));
        }
        return Ok(secret);
    }

    let mut bytes = vec![0_u8; 32];
    getrandom::fill(&mut bytes)
        .map_err(|error| CoreError::SecretStore(format!("secure randomness failed: {error}")))?;
    let secret = SecretBytes::new(bytes);
    store.set(IPC_SECRET_ACCOUNT, &secret)?;
    Ok(secret)
}

/// Browser hosts receive only their role key, never the native UI credential.
pub fn install_browser_ipc_secret(
    directory: &std::path::Path,
    root: &SecretBytes,
    store: &dyn SecretStore,
) -> CoreResult<()> {
    let secret = crate::ipc::derive_browser_secret(root);
    #[cfg(windows)]
    {
        let _ = directory;
        store.set("browser-ipc-v2", &secret)?;
    }
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let _ = store;
        std::fs::create_dir_all(directory)?;
        std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
        let staged = directory.join(format!(".browser-key-{}", uuid::Uuid::new_v4()));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&staged)?;
        file.write_all(secret.expose())?;
        file.sync_all()?;
        std::fs::rename(staged, directory.join("browser-ipc-v2.key"))?;
    }
    Ok(())
}

pub fn load_browser_ipc_secret(
    directory: &std::path::Path,
    store: &dyn SecretStore,
) -> CoreResult<SecretBytes> {
    #[cfg(windows)]
    {
        let _ = directory;
        return store
            .get("browser-ipc-v2")?
            .ok_or(CoreError::AuthenticationFailed);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let _ = store;
        let path = directory.join("browser-ipc-v2.key");
        let metadata = std::fs::symlink_metadata(&path)?;
        if !metadata.is_file()
            || metadata.mode() & 0o077 != 0
            || metadata.nlink() != 1
            || metadata.uid() != std::fs::metadata(directory)?.uid()
        {
            return Err(CoreError::AuthenticationFailed);
        }
        let bytes = crate::execution::files::PinnedPath::open(&path.canonicalize()?)?.read(32)?;
        if bytes.len() != 32 {
            return Err(CoreError::AuthenticationFailed);
        }
        Ok(SecretBytes::new(bytes))
    }
}

#[cfg(test)]
pub mod testing {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    pub struct MemorySecretStore {
        values: Mutex<HashMap<String, Vec<u8>>>,
    }

    impl SecretStore for MemorySecretStore {
        fn get(&self, account: &str) -> CoreResult<Option<SecretBytes>> {
            Ok(self
                .values
                .lock()
                .map_err(|_| CoreError::SecretStore("secret-store lock poisoned".into()))?
                .get(account)
                .cloned()
                .map(SecretBytes::new))
        }

        fn set(&self, account: &str, secret: &SecretBytes) -> CoreResult<()> {
            self.values
                .lock()
                .map_err(|_| CoreError::SecretStore("secret-store lock poisoned".into()))?
                .insert(account.into(), secret.expose().to_vec());
            Ok(())
        }

        fn delete(&self, account: &str) -> CoreResult<()> {
            self.values
                .lock()
                .map_err(|_| CoreError::SecretStore("secret-store lock poisoned".into()))?
                .remove(account);
            Ok(())
        }
    }
}
