use std::path::PathBuf;

use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone)]
pub enum IpcEndpoint {
    #[cfg(unix)]
    UnixSocket(PathBuf),
    #[cfg(windows)]
    NamedPipe(String),
}

#[derive(Debug, Clone)]
pub struct CoreConfig {
    pub data_dir: PathBuf,
    pub database_path: PathBuf,
    pub recovery_dir: PathBuf,
    pub ipc_endpoint: IpcEndpoint,
    pub maximum_replans: u32,
}

impl CoreConfig {
    pub fn platform_default() -> CoreResult<Self> {
        let base = directories::BaseDirs::new().ok_or_else(|| {
            CoreError::Storage("operating-system data directory is unavailable".into())
        })?;
        #[cfg(target_os = "macos")]
        let data_dir = base.home_dir().join("Library/Application Support/Sage");
        #[cfg(target_os = "windows")]
        let data_dir = base.data_local_dir().join("Sage");
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let data_dir = base.data_local_dir().join("sage");
        let database_path = data_dir.join("sage.db");
        let recovery_dir = data_dir.join("recovery");

        #[cfg(unix)]
        let ipc_endpoint = IpcEndpoint::UnixSocket(data_dir.join("sage-core.sock"));
        #[cfg(windows)]
        let ipc_endpoint = IpcEndpoint::NamedPipe(r"\\.\pipe\sage-core-v1".into());

        Ok(Self {
            data_dir,
            database_path,
            recovery_dir,
            ipc_endpoint,
            maximum_replans: 2,
        })
    }

    #[cfg(test)]
    pub fn for_test(root: &std::path::Path) -> Self {
        Self {
            data_dir: root.to_path_buf(),
            database_path: root.join("sage.db"),
            recovery_dir: root.join("recovery"),
            #[cfg(unix)]
            ipc_endpoint: IpcEndpoint::UnixSocket(root.join("sage-core.sock")),
            #[cfg(windows)]
            ipc_endpoint: IpcEndpoint::NamedPipe(format!(
                r"\\.\pipe\sage-core-test-{}",
                uuid::Uuid::new_v4()
            )),
            maximum_replans: 1,
        }
    }
}
