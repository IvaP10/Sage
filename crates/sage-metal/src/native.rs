use std::ffi::{CStr, c_char, c_void};
use std::ptr::NonNull;
use std::slice;
use std::sync::{Arc, OnceLock};
use zeroize::Zeroize;

unsafe extern "C" {
    fn sage_metal_context_create(error: *mut c_char, capacity: usize) -> *mut c_void;
    fn sage_metal_context_release(context: *mut c_void);
    fn sage_metal_buffer_create(
        context: *mut c_void,
        bytes: *const u8,
        length: usize,
        error: *mut c_char,
        capacity: usize,
    ) -> *mut c_void;
    fn sage_metal_buffer_allocate(
        context: *mut c_void,
        length: usize,
        error: *mut c_char,
        capacity: usize,
    ) -> *mut c_void;
    fn sage_metal_buffer_release(buffer: *mut c_void);
    fn sage_metal_buffer_contents(buffer: *mut c_void) -> *const u8;
    fn sage_metal_buffer_contents_mut(buffer: *mut c_void) -> *mut u8;
    fn sage_metal_buffer_length(buffer: *mut c_void) -> usize;
    fn sage_metal_q4_project(
        context: *mut c_void,
        weights: *mut c_void,
        scales: *mut c_void,
        input: *mut c_void,
        output: *mut c_void,
        rows: u32,
        columns: u32,
        group_size: u32,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
}

#[derive(Debug)]
pub struct MetalContext {
    raw: NonNull<c_void>,
}

// Apple documents MTLCommandQueue as thread-safe. The device and pipeline are
// immutable after creation, and each dispatch has private command and I/O
// buffers; no command encoder or command buffer is shared across calls.
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

#[derive(Debug)]
pub struct MetalBuffer {
    raw: NonNull<c_void>,
    byte_len: usize,
    context: Arc<MetalContext>,
}

/// Per-matrix activation storage reused by every Q4 projection. Metal buffers
/// use shared storage on Apple Silicon; the host copies only the current
/// activation into the retained input buffer and reads the result from the
/// retained output buffer after the command completes.
#[derive(Debug)]
pub struct MetalQ4Workspace {
    input: MetalBuffer,
    output: MetalBuffer,
    rows: usize,
    columns: usize,
}

impl MetalQ4Workspace {
    fn clear(&mut self) {
        self.input.clear();
        self.output.clear();
    }
}

// Sage creates shared-storage buffers once from validated Q4 bytes. After
// upload, kernels bind weights/scales as `device const`; Sage only reads their
// contents and never mutates them. The owning context outlives each buffer.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

static SHARED_CONTEXT: OnceLock<Result<Arc<MetalContext>, String>> = OnceLock::new();

impl MetalContext {
    pub fn shared() -> Result<Arc<Self>, String> {
        SHARED_CONTEXT.get_or_init(Self::create).clone()
    }

    fn create() -> Result<Arc<Self>, String> {
        let mut error = [0_i8; 512];
        // The native function returns one retained Objective-C reference.
        let raw = unsafe { sage_metal_context_create(error.as_mut_ptr(), error.len()) };
        NonNull::new(raw)
            .map(|raw| Arc::new(Self { raw }))
            .ok_or_else(|| error_message(&error, "Metal device or Q4 pipeline is unavailable"))
    }

    pub fn buffer(self: &Arc<Self>, bytes: &[u8]) -> Result<MetalBuffer, String> {
        if bytes.is_empty() {
            return Err("Sage Metal buffers cannot be empty".into());
        }
        let mut error = [0_i8; 512];
        let raw = unsafe {
            sage_metal_buffer_create(
                self.raw.as_ptr(),
                bytes.as_ptr(),
                bytes.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let raw = NonNull::new(raw)
            .ok_or_else(|| error_message(&error, "Metal buffer allocation failed"))?;
        let byte_len = unsafe { sage_metal_buffer_length(raw.as_ptr()) };
        if byte_len != bytes.len() {
            unsafe { sage_metal_buffer_release(raw.as_ptr()) };
            return Err("Metal returned a buffer with an unexpected length".into());
        }
        Ok(MetalBuffer {
            raw,
            byte_len,
            context: Arc::clone(self),
        })
    }

    fn buffer_with_length(self: &Arc<Self>, length: usize) -> Result<MetalBuffer, String> {
        if length == 0 {
            return Err("Sage Metal buffers cannot be empty".into());
        }
        let mut error = [0_i8; 512];
        let raw = unsafe {
            sage_metal_buffer_allocate(self.raw.as_ptr(), length, error.as_mut_ptr(), error.len())
        };
        let raw = NonNull::new(raw)
            .ok_or_else(|| error_message(&error, "Metal buffer allocation failed"))?;
        let byte_len = unsafe { sage_metal_buffer_length(raw.as_ptr()) };
        if byte_len != length {
            unsafe { sage_metal_buffer_release(raw.as_ptr()) };
            return Err("Metal returned a buffer with an unexpected length".into());
        }
        Ok(MetalBuffer {
            raw,
            byte_len,
            context: Arc::clone(self),
        })
    }

    /// Allocate a bounded, reusable activation workspace for one fixed Q4
    /// matrix. The geometry is sealed into the workspace so it cannot be
    /// accidentally reused with another projection shape.
    pub fn q4_workspace(
        self: &Arc<Self>,
        rows: usize,
        columns: usize,
    ) -> Result<MetalQ4Workspace, String> {
        let input_bytes = columns
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Metal Q4 input workspace size overflow".to_owned())?;
        let output_bytes = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Metal Q4 output workspace size overflow".to_owned())?;
        if rows == 0 || columns == 0 || rows > u32::MAX as usize || columns > u32::MAX as usize {
            return Err("Metal Q4 workspace geometry is invalid".into());
        }
        let input = self.buffer_with_length(input_bytes)?;
        let output = self.buffer_with_length(output_bytes)?;
        Ok(MetalQ4Workspace {
            input,
            output,
            rows,
            columns,
        })
    }

    /// Run a Q4 projection with caller-owned output and retained I/O buffers.
    /// Input and output activation bytes are scrubbed before returning on both
    /// success and failure.
    #[allow(clippy::too_many_arguments)]
    pub fn project_q4_into(
        &self,
        weights: &MetalBuffer,
        scales: &MetalBuffer,
        rows: usize,
        columns: usize,
        group_size: usize,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut MetalQ4Workspace,
    ) -> Result<(), String> {
        let result = (|| {
            if rows == 0
                || columns == 0
                || group_size == 0
                || input.len() != columns
                || output.len() != rows
                || input.iter().any(|value| !value.is_finite())
            {
                return Err("Metal Q4 projection geometry or input is invalid".into());
            }
            let elements = rows
                .checked_mul(columns)
                .ok_or_else(|| "Metal Q4 projection dimensions overflow".to_owned())?;
            if elements > 700_000_000 {
                return Err("Metal Q4 projection exceeds Sage's element bound".into());
            }
            let required_weights = elements.div_ceil(2);
            let required_scales = elements
                .div_ceil(group_size)
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "Metal Q4 scale length overflow".to_owned())?;
            if !std::ptr::eq(Arc::as_ptr(&weights.context), self)
                || !Arc::ptr_eq(&scales.context, &weights.context)
                || !Arc::ptr_eq(&workspace.input.context, &weights.context)
                || !Arc::ptr_eq(&workspace.output.context, &weights.context)
                || weights.byte_len != required_weights
                || scales.byte_len != required_scales
                || workspace.rows != rows
                || workspace.columns != columns
                || rows > u32::MAX as usize
                || columns > u32::MAX as usize
                || group_size > u32::MAX as usize
            {
                return Err("Metal Q4 projection geometry or storage is invalid".into());
            }

            workspace.input.write_f32s(input)?;
            let mut error = [0_i8; 512];
            let status = unsafe {
                // The workspace owns all four retained Metal objects, has
                // exclusive access to its activation buffers under the
                // caller's mutex, and Metal work is complete before return.
                sage_metal_q4_project(
                    self.raw.as_ptr(),
                    weights.raw.as_ptr(),
                    scales.raw.as_ptr(),
                    workspace.input.raw.as_ptr(),
                    workspace.output.raw.as_ptr(),
                    rows as u32,
                    columns as u32,
                    group_size as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(error_message(&error, "Metal Q4 projection failed"));
            }
            workspace.output.copy_f32s(output)?;
            if output.iter().any(|value| !value.is_finite()) {
                return Err("Metal Q4 projection produced a non-finite output".into());
            }
            Ok(())
        })();
        workspace.clear();
        if result.is_err() {
            output.fill(0.0);
        }
        result
    }
}

impl MetalBuffer {
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Borrow the immutable shared-storage bytes without copying them. Sage
    /// uploads these buffers once and binds them read-only to Metal kernels.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        let contents = unsafe { sage_metal_buffer_contents(self.raw.as_ptr()) };
        if contents.is_null() {
            return None;
        }
        // The MTLBuffer is owned by `self`, uses shared storage, and is never
        // mutated after upload, so this borrowed slice remains valid and
        // immutable for the lifetime of the buffer borrow.
        Some(unsafe { slice::from_raw_parts(contents, self.byte_len) })
    }

    pub fn byte(&self, index: usize) -> Option<u8> {
        if index >= self.byte_len {
            return None;
        }
        let contents = unsafe { sage_metal_buffer_contents(self.raw.as_ptr()) };
        if contents.is_null() {
            return None;
        }
        Some(unsafe { *contents.add(index) })
    }

    pub fn f32(&self, index: usize) -> Option<f32> {
        let start = index.checked_mul(4)?;
        let bytes = [
            self.byte(start)?,
            self.byte(start + 1)?,
            self.byte(start + 2)?,
            self.byte(start + 3)?,
        ];
        Some(f32::from_ne_bytes(bytes))
    }

    pub fn copy_range(&self, offset: usize, length: usize) -> Option<Vec<u8>> {
        let end = offset.checked_add(length)?;
        if end > self.byte_len {
            return None;
        }
        if length == 0 {
            return Some(Vec::new());
        }
        let contents = unsafe { sage_metal_buffer_contents(self.raw.as_ptr()) };
        if contents.is_null() {
            return None;
        }
        Some(unsafe { slice::from_raw_parts(contents.add(offset), length) }.to_vec())
    }

    fn write_f32s(&mut self, values: &[f32]) -> Result<(), String> {
        let byte_length = values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Metal activation byte length overflow".to_owned())?;
        if byte_length != self.byte_len {
            return Err("Metal activation buffer has an unexpected length".into());
        }
        let contents = unsafe { sage_metal_buffer_contents_mut(self.raw.as_ptr()) };
        if contents.is_null() {
            return Err("Metal activation buffer is not CPU accessible".into());
        }
        // The workspace owns this shared buffer exclusively for the call and
        // the checked byte length is exactly the source slice size.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), contents, byte_length);
        }
        Ok(())
    }

    fn copy_f32s(&self, output: &mut [f32]) -> Result<(), String> {
        let byte_length = output
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Metal output byte length overflow".to_owned())?;
        if byte_length != self.byte_len {
            return Err("Metal output buffer has an unexpected length".into());
        }
        let contents = unsafe { sage_metal_buffer_contents(self.raw.as_ptr()) };
        if contents.is_null() {
            return Err("Metal output buffer is not CPU accessible".into());
        }
        // Metal shared allocations are page-aligned, the exact byte length was
        // checked above, and the command buffer completed before this read.
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents.cast::<f32>(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok(())
    }

    fn clear(&mut self) {
        let contents = unsafe { sage_metal_buffer_contents_mut(self.raw.as_ptr()) };
        if contents.is_null() {
            return;
        }
        // The allocation was zero-initialized at creation and is exclusively
        // borrowed while no command buffer is in flight.
        let bytes = unsafe { slice::from_raw_parts_mut(contents, self.byte_len) };
        bytes.zeroize();
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        unsafe { sage_metal_context_release(self.raw.as_ptr()) };
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        unsafe { sage_metal_buffer_release(self.raw.as_ptr()) };
    }
}

fn error_message(buffer: &[i8], fallback: &str) -> String {
    let pointer = buffer.as_ptr();
    if pointer.is_null() {
        return fallback.to_owned();
    }
    // The C bridge always writes a NUL-terminated message into this fixed
    // buffer before returning an error.
    let message = unsafe { CStr::from_ptr(pointer) }.to_string_lossy();
    if message.is_empty() {
        fallback.to_owned()
    } else {
        message.into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::MetalContext;

    #[test]
    #[ignore = "requires a Metal-capable Apple GPU"]
    fn q4_workspace_is_reused_and_activation_bytes_are_scrubbed() {
        let context = MetalContext::shared().expect("Metal device");
        let weights = context
            .buffer(&[0xa9, 0xcb, 0x87, 0xa9])
            .expect("packed Q4 weights");
        let scales = context.buffer(&1.0_f32.to_ne_bytes()).expect("Q4 scales");
        let mut workspace = context.q4_workspace(2, 4).expect("Q4 workspace");
        let input_address = workspace.input.as_bytes().unwrap().as_ptr();
        let output_address = workspace.output.as_bytes().unwrap().as_ptr();
        let mut output = [0.0_f32; 2];

        context
            .project_q4_into(
                &weights,
                &scales,
                2,
                4,
                16,
                &[1.0; 4],
                &mut output,
                &mut workspace,
            )
            .expect("first Q4 projection");
        assert_eq!(output, [10.0, 2.0]);
        assert_eq!(workspace.input.as_bytes().unwrap().as_ptr(), input_address);
        assert_eq!(
            workspace.output.as_bytes().unwrap().as_ptr(),
            output_address
        );
        assert!(
            workspace
                .input
                .as_bytes()
                .unwrap()
                .iter()
                .all(|byte| *byte == 0)
        );
        assert!(
            workspace
                .output
                .as_bytes()
                .unwrap()
                .iter()
                .all(|byte| *byte == 0)
        );

        context
            .project_q4_into(
                &weights,
                &scales,
                2,
                4,
                16,
                &[1.0, 2.0, 3.0, 4.0],
                &mut output,
                &mut workspace,
            )
            .expect("reused Q4 projection");
        assert_eq!(output, [30.0, 10.0]);
        assert!(
            workspace
                .input
                .as_bytes()
                .unwrap()
                .iter()
                .all(|byte| *byte == 0)
        );
        assert!(
            workspace
                .output
                .as_bytes()
                .unwrap()
                .iter()
                .all(|byte| *byte == 0)
        );

        output.fill(9.0);
        assert!(
            context
                .project_q4_into(
                    &weights,
                    &scales,
                    2,
                    4,
                    16,
                    &[1.0, f32::INFINITY, 3.0, 4.0],
                    &mut output,
                    &mut workspace,
                )
                .is_err()
        );
        assert_eq!(output, [0.0; 2]);
        assert!(
            workspace
                .input
                .as_bytes()
                .unwrap()
                .iter()
                .all(|byte| *byte == 0)
        );
        assert!(
            workspace
                .output
                .as_bytes()
                .unwrap()
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    #[ignore = "requires a Metal-capable Apple GPU"]
    fn q4_packed_pairs_preserve_odd_quantization_group_boundaries() {
        let context = MetalContext::shared().expect("Metal device");
        let rows = 2;
        let columns = 34;
        let group_size = 17;
        let codes = (0..rows * columns)
            .map(|index| ((index * 7 + 3) % 16) as u8)
            .collect::<Vec<_>>();
        let packed = codes
            .chunks(2)
            .map(|pair| pair[0] | (pair[1] << 4))
            .collect::<Vec<_>>();
        let group_scales = [0.25_f32, 0.5, 0.75, 1.0];
        let scale_bytes = group_scales
            .iter()
            .flat_map(|scale| scale.to_ne_bytes())
            .collect::<Vec<_>>();
        let weights = context.buffer(&packed).expect("packed Q4 weights");
        let scales = context.buffer(&scale_bytes).expect("Q4 scales");
        let input = (0..columns)
            .map(|column| (column as f32 - 17.0) * 0.03)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0_f32; rows];
        for (row, result) in expected.iter_mut().enumerate() {
            for (column, activation) in input.iter().copied().enumerate() {
                let index = row * columns + column;
                let quantized = f32::from(codes[index]) - 8.0;
                *result += quantized * group_scales[index / group_size] * activation;
            }
        }

        let mut workspace = context.q4_workspace(rows, columns).expect("Q4 workspace");
        let mut actual = [0.0_f32; 2];
        context
            .project_q4_into(
                &weights,
                &scales,
                rows,
                columns,
                group_size,
                &input,
                &mut actual,
                &mut workspace,
            )
            .expect("Q4 projection across packed scale boundaries");
        for (observed, reference) in actual.into_iter().zip(expected) {
            assert!((observed - reference).abs() <= 1.0e-4 + reference.abs() * 1.0e-4);
        }
    }
}
