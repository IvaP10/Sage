//! Small first-party Metal compute boundary for Sage's own inference kernels.
//!
//! The crate intentionally exposes only bounded buffer and Q4 projection
//! operations. Model parsing, quantization, scheduling, and CPU reference math
//! stay in Sage; this wrapper owns the narrow Objective-C/Metal calls.

#[cfg(target_os = "macos")]
mod native;

#[cfg(target_os = "macos")]
pub use native::{MetalBuffer, MetalContext, MetalQ4Workspace};

#[cfg(not(target_os = "macos"))]
mod unavailable {
    use std::sync::Arc;

    #[derive(Debug)]
    pub struct MetalContext;

    #[derive(Debug)]
    pub struct MetalBuffer;

    #[derive(Debug)]
    pub struct MetalQ4Workspace;

    impl MetalContext {
        pub fn shared() -> Result<Arc<Self>, String> {
            Err("Sage Metal inference is available only on macOS".into())
        }

        pub fn buffer(&self, _bytes: &[u8]) -> Result<MetalBuffer, String> {
            Err("Sage Metal inference is available only on macOS".into())
        }

        pub fn buffer_private(&self, _bytes: &[u8]) -> Result<MetalBuffer, String> {
            Err("Sage Metal inference is available only on macOS".into())
        }

        pub fn q4_workspace(
            self: &Arc<Self>,
            _rows: usize,
            _columns: usize,
        ) -> Result<MetalQ4Workspace, String> {
            Err("Sage Metal inference is available only on macOS".into())
        }

        pub fn q4_batch_workspace(
            self: &Arc<Self>,
            _rows: usize,
            _columns: usize,
            _batch_size: usize,
        ) -> Result<MetalQ4Workspace, String> {
            Err("Sage Metal inference is available only on macOS".into())
        }

        pub fn q4_batch_workspace_with_tile(
            self: &Arc<Self>,
            _rows: usize,
            _columns: usize,
            _batch_size: usize,
            _batch_tile_size: usize,
        ) -> Result<MetalQ4Workspace, String> {
            Err("Sage Metal inference is available only on macOS".into())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn project_q4_into(
            &self,
            _weights: &MetalBuffer,
            _scales: &MetalBuffer,
            _rows: usize,
            _columns: usize,
            _group_size: usize,
            _input: &[f32],
            _output: &mut [f32],
            _workspace: &mut MetalQ4Workspace,
        ) -> Result<(), String> {
            Err("Sage Metal inference is available only on macOS".into())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn project_q4_profiled(
            &self,
            _weights: &MetalBuffer,
            _scales: &MetalBuffer,
            _rows: usize,
            _columns: usize,
            _group_size: usize,
            _input: &[f32],
            _output: &mut [f32],
            _workspace: &mut MetalQ4Workspace,
        ) -> Result<u64, String> {
            Err("Sage Metal inference is available only on macOS".into())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn project_q4_rows4_profiled(
            &self,
            _weights: &MetalBuffer,
            _scales: &MetalBuffer,
            _rows: usize,
            _columns: usize,
            _group_size: usize,
            _input: &[f32],
            _output: &mut [f32],
            _workspace: &mut MetalQ4Workspace,
        ) -> Result<u64, String> {
            Err("Sage Metal inference is available only on macOS".into())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn project_q4_batch_into(
            &self,
            _weights: &MetalBuffer,
            _scales: &MetalBuffer,
            _rows: usize,
            _columns: usize,
            _group_size: usize,
            _batch_size: usize,
            _input: &[f32],
            _output: &mut [f32],
            _workspace: &mut MetalQ4Workspace,
        ) -> Result<(), String> {
            Err("Sage Metal inference is available only on macOS".into())
        }
    }

    impl MetalBuffer {
        pub fn byte_len(&self) -> usize {
            0
        }

        pub fn byte(&self, _index: usize) -> Option<u8> {
            None
        }

        pub fn as_bytes(&self) -> Option<&[u8]> {
            None
        }

        pub fn f32(&self, _index: usize) -> Option<f32> {
            None
        }

        pub fn copy_range(&self, _offset: usize, _length: usize) -> Option<Vec<u8>> {
            None
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use unavailable::{MetalBuffer, MetalContext, MetalQ4Workspace};

/// Report whether the native Metal device and Sage's Q4 kernel can be created.
pub fn is_available() -> bool {
    MetalContext::shared().is_ok()
}
