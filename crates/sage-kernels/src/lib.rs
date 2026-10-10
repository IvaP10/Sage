//! First-party architecture-specific inference kernels.
//!
//! The safe API validates dimensions and lengths before entering a narrow SIMD
//! implementation, then rejects non-finite results. The scalar path remains
//! the portable reference.

#![deny(unsafe_op_in_unsafe_fn)]

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, OnceLock,
        mpsc::{self, SyncSender},
    },
    thread::{self, Builder},
};

const MAX_Q4_ELEMENTS: usize = 1 << 30;
pub const Q4_BATCH_TILE_SIZE: usize = 32;
pub const Q4_BATCH_MAX_SIZE: usize = 256;
pub const Q4_BATCH_MAX_SCRATCH_ELEMENTS: usize = 1 << 20;
pub const Q4_BATCH_MAX_IO_ELEMENTS: usize = 1 << 26;
const MAX_RMS_NORM_ELEMENTS: usize = 250_000_000;
const MAX_INFERENCE_CPU_WORKERS: usize = 8;
const MIN_PARALLEL_Q4_ROWS: usize = 256;
const MIN_PARALLEL_Q4_ELEMENTS: usize = 2_000_000;
#[cfg(any(target_arch = "aarch64", test))]
const MAX_ATTENTION_VALUE_DIMENSION: usize = 512;
#[cfg(target_arch = "aarch64")]
const MAX_ATTENTION_VECTOR_BLOCKS: usize = MAX_ATTENTION_VALUE_DIMENSION / 4;
const MAX_GROUPED_QUERY_HEADS: usize = 5;
#[cfg(target_arch = "aarch64")]
const MAX_GROUPED_ATTENTION_VALUE_DIMENSION: usize = 256;
#[cfg(target_arch = "aarch64")]
const MAX_GROUPED_ATTENTION_VECTOR_BLOCKS: usize = MAX_GROUPED_ATTENTION_VALUE_DIMENSION / 4;
const MAX_DELTA_HEAD_ELEMENTS: usize = 1_000_000;
const MAX_DELTA_KEY_DIMENSION: usize = 512;
const MAX_DELTA_VALUE_DIMENSION: usize = 512;
#[cfg(target_arch = "aarch64")]
const MAX_DELTA_VECTOR_BLOCKS: usize = MAX_DELTA_VALUE_DIMENSION / 4;

/// Decode one IEEE-754 binary16 value using first-party bit operations.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut normalized = fraction;
            let mut unbiased = -14i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                unbiased -= 1;
            }
            normalized &= 0x03ff;
            sign | (u32::try_from(unbiased + 127).unwrap_or(0) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (u32::from(exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

/// Encode f32 as IEEE-754 binary16, rounding ties to even. Overflow becomes
/// infinity; callers storing activations must validate their own range.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let f32_exponent = ((bits >> 23) & 0xff) as i32;
    if f32_exponent == 0xff {
        let fraction = bits & 0x007f_ffff;
        return if fraction == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7c00 | ((fraction >> 13) as u16).max(1)
        };
    }
    if f32_exponent == 0 {
        return sign;
    }
    let half_exponent = f32_exponent - 127 + 15;
    let fraction = bits & 0x007f_ffff;
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let shift = u32::try_from(14 - half_exponent).unwrap_or(24);
        let rounded = round_shift_ties_even(0x0080_0000 | fraction, shift);
        return sign | rounded as u16;
    }

    let mut half_exponent = half_exponent;
    let mut half_fraction = round_shift_ties_even(fraction, 13);
    if half_fraction == 0x0400 {
        half_exponent += 1;
        half_fraction = 0;
    }
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    sign | ((half_exponent as u16) << 10) | half_fraction as u16
}

/// Convert a slice of f32 values into IEEE binary16 bits in caller-owned
/// storage. AArch64 systems with native FP16 conversion use a four-lane
/// `FCVTN` path for ordinary finite values; boundary, subnormal, and special
/// values use Sage's exact scalar reference conversion.
pub fn f32_to_f16_bits_into(values: &[f32], output: &mut [u16]) -> Result<(), &'static str> {
    if values.len() != output.len() {
        output.fill(0);
        return Err("Binary16 conversion input and output lengths do not match");
    }

    #[cfg(target_arch = "aarch64")]
    if values.len() >= 16 && native_aarch64_fp16_conversion_available() {
        // SAFETY: runtime detection confirms native FP16 support; the target
        // function validates each four-lane group before using FCVTN.
        unsafe { f32_to_f16_bits_neon(values, output) };
        return Ok(());
    }

    for (value, destination) in values.iter().zip(output) {
        *destination = f32_to_f16_bits(*value);
    }
    Ok(())
}

/// Validate and convert values that will be stored in Sage's binary16 KV
/// cache. Validation is fused with conversion: ordinary finite AArch64 lanes
/// are range-checked in NEON registers and converted from the same loaded
/// vector. Subnormals retain the exact scalar reference behavior.
pub fn f32_to_f16_bits_checked_into(
    values: &[f32],
    output: &mut [u16],
) -> Result<(), &'static str> {
    if values.len() != output.len() {
        output.fill(0);
        return Err("Binary16 conversion input and output lengths do not match");
    }

    #[cfg(target_arch = "aarch64")]
    if values.len() >= 4 && native_aarch64_fp16_conversion_available() {
        // SAFETY: runtime detection confirms native FP16 support; the target
        // function checks every lane before conversion and stays in bounds.
        let result = unsafe { f32_to_f16_bits_checked_neon(values, output.as_mut_ptr()) };
        if result.is_err() {
            output.fill(0);
        }
        return result;
    }

    for index in 0..values.len() {
        let value = values[index];
        if !value.is_finite() || value.abs() > 65_504.0 {
            output.fill(0);
            return Err("KV cache value is outside the finite binary16 range");
        }
        output[index] = f32_to_f16_bits(value);
    }
    Ok(())
}

/// Validate and append binary16 values directly into a vector's spare
/// capacity. Unlike `resize` followed by conversion, this writes each new
/// element once. If validation fails after a vector prefix was converted, the
/// complete spare region is scrubbed while the vector's initialized length
/// remains unchanged.
pub fn f32_to_f16_bits_checked_extend(
    values: &[f32],
    output: &mut Vec<u16>,
) -> Result<(), &'static str> {
    let initial_length = output.len();
    let final_length = initial_length
        .checked_add(values.len())
        .ok_or("Binary16 append length overflowed")?;
    output
        .try_reserve(values.len())
        .map_err(|_| "Binary16 append allocation was denied")?;

    let converted = {
        let spare = &mut output.spare_capacity_mut()[..values.len()];
        #[cfg(target_arch = "aarch64")]
        let result = if values.len() >= 4 && native_aarch64_fp16_conversion_available() {
            // SAFETY: `try_reserve` guarantees this spare range is writable;
            // the converter writes at most `values.len()` initialized u16s.
            unsafe { f32_to_f16_bits_checked_neon(values, spare.as_mut_ptr().cast::<u16>()) }
        } else {
            write_checked_f16_scalar(values, spare)
        };
        #[cfg(not(target_arch = "aarch64"))]
        let result = write_checked_f16_scalar(values, spare);

        if result.is_err() {
            for slot in spare {
                slot.write(0);
            }
        }
        result
    };
    converted?;

    // SAFETY: every slot in the appended range was initialized by the scalar
    // or SIMD converter before this length change.
    unsafe { output.set_len(final_length) };
    Ok(())
}

/// Quantize one finite group into Sage's symmetric signed four-bit values.
/// The returned scale and rounded values match the scalar Q4 reference.
pub fn quantize_symmetric_q4_group(
    values: &[f32],
    quantized: &mut [i8],
) -> Result<f32, &'static str> {
    if values.is_empty() || values.len() > 4096 || values.len() != quantized.len() {
        quantized.fill(0);
        return Err("Q4 quantization group dimensions are invalid");
    }
    if values.iter().any(|value| !value.is_finite()) {
        quantized.fill(0);
        return Err("Q4 quantization group contains a non-finite value");
    }

    quantize_symmetric_q4_group_finite(values, quantized)
}

/// Quantize a group already proven finite by its streaming owner.
///
/// Callers must validate the complete source chunk before passing any slices
/// here. This avoids rescanning every model weight once per quantization group.
pub fn quantize_symmetric_q4_group_finite(
    values: &[f32],
    quantized: &mut [i8],
) -> Result<f32, &'static str> {
    if values.is_empty() || values.len() > 4096 || values.len() != quantized.len() {
        quantized.fill(0);
        return Err("Q4 quantization group dimensions are invalid");
    }
    #[cfg(target_arch = "aarch64")]
    if values.len() >= 8 {
        // SAFETY: lengths are bounded above; the streaming owner validated
        // every source chunk before constructing these group slices.
        return Ok(unsafe { quantize_symmetric_q4_group_neon(values, quantized) });
    }

    quantize_symmetric_q4_group_scalar(values, quantized)
}

fn quantize_symmetric_q4_group_scalar(
    values: &[f32],
    quantized: &mut [i8],
) -> Result<f32, &'static str> {
    let maximum = values
        .iter()
        .fold(0.0f32, |prior, value| prior.max(value.abs()));
    let scale = if maximum == 0.0 { 1.0 } else { maximum / 7.0 };
    for (value, output) in values.iter().zip(quantized) {
        *output = ((*value / scale).round().clamp(-7.0, 7.0)) as i8;
    }
    Ok(scale)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_symmetric_q4_group_neon(values: &[f32], quantized: &mut [i8]) -> f32 {
    use std::arch::aarch64::{
        vabsq_f32, vcombine_s16, vcvtaq_s32_f32, vdivq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32,
        vmaxq_s32, vmaxvq_f32, vminq_s32, vmovn_s16, vmovn_s32, vst1_s8,
    };

    let mut maximum_lanes = vdupq_n_f32(0.0);
    let mut index = 0usize;
    while index + 4 <= values.len() {
        // SAFETY: the loop bound guarantees four readable f32 values.
        let lanes = unsafe { vld1q_f32(values.as_ptr().add(index)) };
        maximum_lanes = vmaxq_f32(maximum_lanes, vabsq_f32(lanes));
        index += 4;
    }
    let mut maximum = vmaxvq_f32(maximum_lanes);
    for value in values.iter().skip(index) {
        maximum = maximum.max(value.abs());
    }
    let scale = if maximum == 0.0 { 1.0 } else { maximum / 7.0 };
    if scale == 0.0 {
        for (value, output) in values.iter().zip(quantized) {
            *output = ((*value / scale).round().clamp(-7.0, 7.0)) as i8;
        }
        return scale;
    }
    let scale_lanes = vdupq_n_f32(scale);
    let lower_bound = std::arch::aarch64::vdupq_n_s32(-7);
    let upper_bound = std::arch::aarch64::vdupq_n_s32(7);
    index = 0;
    while index + 8 <= values.len() {
        // SAFETY: each iteration reads and writes eight validated elements.
        let first = unsafe { vld1q_f32(values.as_ptr().add(index)) };
        // SAFETY: the eight-element loop bound includes this second four-lane load.
        let second = unsafe { vld1q_f32(values.as_ptr().add(index + 4)) };
        let first = vcvtaq_s32_f32(vdivq_f32(first, scale_lanes));
        let second = vcvtaq_s32_f32(vdivq_f32(second, scale_lanes));
        let first = vmaxq_s32(vminq_s32(first, upper_bound), lower_bound);
        let second = vmaxq_s32(vminq_s32(second, upper_bound), lower_bound);
        let narrowed = vcombine_s16(vmovn_s32(first), vmovn_s32(second));
        let packed = vmovn_s16(narrowed);
        // SAFETY: the loop bound reserves all eight output bytes at this offset.
        unsafe { vst1_s8(quantized.as_mut_ptr().add(index), packed) };
        index += 8;
    }
    while index < values.len() {
        quantized[index] = ((values[index] / scale).round().clamp(-7.0, 7.0)) as i8;
        index += 1;
    }
    scale
}

fn write_checked_f16_scalar(
    values: &[f32],
    output: &mut [std::mem::MaybeUninit<u16>],
) -> Result<(), &'static str> {
    for (value, destination) in values.iter().zip(output) {
        if !value.is_finite() || value.abs() > 65_504.0 {
            return Err("KV cache value is outside the finite binary16 range");
        }
        destination.write(f32_to_f16_bits(*value));
    }
    Ok(())
}

/// Qwen3.5 zero-centered RMS normalization into caller-owned storage.
///
/// AArch64 widens four input values at a time and performs the reduction and
/// scaling in f64 NEON lanes. Other targets use the portable f64 reference.
/// The output is cleared on invalid input or a non-finite result.
pub fn rms_norm_zero_centered_into(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), &'static str> {
    if input.is_empty()
        || input.len() > MAX_RMS_NORM_ELEMENTS
        || input.len() != weight.len()
        || input.len() != output.len()
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || input.iter().chain(weight).any(|value| !value.is_finite())
    {
        output.fill(0.0);
        return Err("Zero-centered RMS normalization dimensions or values are invalid");
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: lengths and finite input values are validated above; the kernel
    // uses bounded four-value loads and scalar handling for the final tail.
    unsafe {
        rms_norm_zero_centered_neon(input, weight, epsilon, output);
    }
    #[cfg(not(target_arch = "aarch64"))]
    rms_norm_zero_centered_scalar(input, weight, epsilon, output);

    if output.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        output.fill(0.0);
        Err("Zero-centered RMS normalization result is non-finite")
    }
}

/// Qwen gated RMS normalization: scale the RMS-normalized input, then apply
/// the SiLU gate elementwise. AArch64 uses NEON for the sum-of-squares
/// reduction; the SiLU output remains scalar to preserve the f32 reference
/// behavior exactly.
pub fn rms_norm_silu_gated_into(
    input: &[f32],
    gate: &[f32],
    scale: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), &'static str> {
    if input.is_empty()
        || input.len() > MAX_RMS_NORM_ELEMENTS
        || input.len() != gate.len()
        || input.len() != scale.len()
        || input.len() != output.len()
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        output.fill(0.0);
        return Err("Gated RMS normalization dimensions are invalid");
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: lengths are checked above; the reduction uses bounded four-value
    // loads and explicitly handles the scalar tail. The kernel checks values.
    let result = unsafe { rms_norm_silu_gated_neon(input, gate, scale, epsilon, output) };
    #[cfg(not(target_arch = "aarch64"))]
    let result = rms_norm_silu_gated_scalar(input, gate, scale, epsilon, output);
    result
}

#[cfg(not(target_arch = "aarch64"))]
fn rms_norm_zero_centered_scalar(input: &[f32], weight: &[f32], epsilon: f32, output: &mut [f32]) {
    let mean_square = input.iter().fold(0.0f64, |sum, value| {
        sum + f64::from(*value) * f64::from(*value)
    }) / input.len() as f64;
    let inverse = (mean_square + f64::from(epsilon)).sqrt().recip();
    for ((output, value), offset) in output.iter_mut().zip(input).zip(weight) {
        *output = (f64::from(*value) * inverse * (1.0 + f64::from(*offset))) as f32;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn rms_norm_silu_gated_scalar(
    input: &[f32],
    gate: &[f32],
    scale: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), &'static str> {
    let mut sum_squares = 0.0f64;
    for index in 0..input.len() {
        if !input[index].is_finite() || !gate[index].is_finite() || !scale[index].is_finite() {
            output.fill(0.0);
            return Err("Gated RMS normalization input is non-finite");
        }
        sum_squares += f64::from(input[index]) * f64::from(input[index]);
    }
    let mean_square = sum_squares / input.len() as f64;
    let inverse = (mean_square + f64::from(epsilon)).sqrt().recip();
    for index in 0..input.len() {
        let normalized = (f64::from(input[index]) * inverse * f64::from(scale[index])) as f32;
        let gate_activation = gate[index] / (1.0 + (-gate[index]).exp());
        output[index] = (f64::from(normalized) * f64::from(gate_activation)) as f32;
        if !output[index].is_finite() {
            output.fill(0.0);
            return Err("Gated RMS normalization result is non-finite");
        }
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rms_norm_zero_centered_neon(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) {
    use std::arch::aarch64::{
        float64x2_t, vaddq_f64, vaddvq_f64, vcombine_f32, vcvt_f32_f64, vcvt_f64_f32, vdupq_n_f64,
        vfmaq_f64, vget_high_f32, vget_low_f32, vld1q_f32, vmulq_f64, vst1q_f32,
    };

    let zero = vdupq_n_f64(0.0);
    let mut low_sums: [float64x2_t; 4] = [zero; 4];
    let mut high_sums: [float64x2_t; 4] = [zero; 4];
    let vector_length = input.len() / 4 * 4;
    for start in (0..vector_length).step_by(4) {
        let values = unsafe { vld1q_f32(input.as_ptr().add(start)) };
        let low = vcvt_f64_f32(vget_low_f32(values));
        let high = vcvt_f64_f32(vget_high_f32(values));
        let accumulator = (start / 4) % 4;
        low_sums[accumulator] = vfmaq_f64(low_sums[accumulator], low, low);
        high_sums[accumulator] = vfmaq_f64(high_sums[accumulator], high, high);
    }
    let low_total = vaddq_f64(
        vaddq_f64(low_sums[0], low_sums[1]),
        vaddq_f64(low_sums[2], low_sums[3]),
    );
    let high_total = vaddq_f64(
        vaddq_f64(high_sums[0], high_sums[1]),
        vaddq_f64(high_sums[2], high_sums[3]),
    );
    let mut sum_squares = vaddvq_f64(vaddq_f64(low_total, high_total));
    for value in &input[vector_length..] {
        sum_squares += f64::from(*value) * f64::from(*value);
    }
    let inverse_scalar = (sum_squares / input.len() as f64 + f64::from(epsilon))
        .sqrt()
        .recip();

    let ones = vdupq_n_f64(1.0);
    let inverse = vdupq_n_f64(inverse_scalar);
    for start in (0..vector_length).step_by(4) {
        let values = unsafe { vld1q_f32(input.as_ptr().add(start)) };
        let weights = unsafe { vld1q_f32(weight.as_ptr().add(start)) };
        let low = vmulq_f64(
            vmulq_f64(vcvt_f64_f32(vget_low_f32(values)), inverse),
            vaddq_f64(vcvt_f64_f32(vget_low_f32(weights)), ones),
        );
        let high = vmulq_f64(
            vmulq_f64(vcvt_f64_f32(vget_high_f32(values)), inverse),
            vaddq_f64(vcvt_f64_f32(vget_high_f32(weights)), ones),
        );
        let normalized = vcombine_f32(vcvt_f32_f64(low), vcvt_f32_f64(high));
        unsafe { vst1q_f32(output.as_mut_ptr().add(start), normalized) };
    }
    for index in vector_length..input.len() {
        output[index] =
            (f64::from(input[index]) * inverse_scalar * (1.0 + f64::from(weight[index]))) as f32;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rms_norm_silu_gated_neon(
    input: &[f32],
    gate: &[f32],
    scale: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), &'static str> {
    use std::arch::aarch64::{
        float64x2_t, vaddq_f64, vaddvq_f64, vcvt_f64_f32, vdupq_n_f64, vfmaq_f64, vget_high_f32,
        vget_low_f32, vld1q_f32,
    };

    let zero = vdupq_n_f64(0.0);
    let mut low_sums: [float64x2_t; 4] = [zero; 4];
    let mut high_sums: [float64x2_t; 4] = [zero; 4];
    let vector_length = input.len() / 4 * 4;
    for start in (0..vector_length).step_by(4) {
        for index in start..start + 4 {
            if !input[index].is_finite() || !gate[index].is_finite() || !scale[index].is_finite() {
                output.fill(0.0);
                return Err("Gated RMS normalization input is non-finite");
            }
        }
        let values = unsafe { vld1q_f32(input.as_ptr().add(start)) };
        let low = vcvt_f64_f32(vget_low_f32(values));
        let high = vcvt_f64_f32(vget_high_f32(values));
        let accumulator = (start / 4) % 4;
        low_sums[accumulator] = vfmaq_f64(low_sums[accumulator], low, low);
        high_sums[accumulator] = vfmaq_f64(high_sums[accumulator], high, high);
    }
    let low_total = vaddq_f64(
        vaddq_f64(low_sums[0], low_sums[1]),
        vaddq_f64(low_sums[2], low_sums[3]),
    );
    let high_total = vaddq_f64(
        vaddq_f64(high_sums[0], high_sums[1]),
        vaddq_f64(high_sums[2], high_sums[3]),
    );
    let mut sum_squares = vaddvq_f64(vaddq_f64(low_total, high_total));
    for index in vector_length..input.len() {
        if !input[index].is_finite() || !gate[index].is_finite() || !scale[index].is_finite() {
            output.fill(0.0);
            return Err("Gated RMS normalization input is non-finite");
        }
        sum_squares += f64::from(input[index]) * f64::from(input[index]);
    }
    let inverse = (sum_squares / input.len() as f64 + f64::from(epsilon))
        .sqrt()
        .recip();
    for index in 0..input.len() {
        let normalized = (f64::from(input[index]) * inverse * f64::from(scale[index])) as f32;
        let gate_activation = gate[index] / (1.0 + (-gate[index]).exp());
        output[index] = (f64::from(normalized) * f64::from(gate_activation)) as f32;
        if !output[index].is_finite() {
            output.fill(0.0);
            return Err("Gated RMS normalization result is non-finite");
        }
    }
    Ok(())
}

fn round_shift_ties_even(value: u32, shift: u32) -> u32 {
    let truncated = value >> shift;
    let remainder_mask = (1_u32 << shift) - 1;
    let remainder = value & remainder_mask;
    let halfway = 1_u32 << (shift - 1);
    truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

/// Project one input vector through Sage's signed grouped-Q4 matrix.
///
/// The matrix is row-major. Each quantized nibble stores a value in `[-8, 7]`
/// and `scales` contains one scale for every `group_size` flattened elements.
/// AArch64 uses NEON, x86-64 uses runtime-gated AVX2, and other architectures
/// use the scalar reference implementation.
pub fn project_q4(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
) -> Result<Vec<f32>, &'static str> {
    validate_q4_matrix(packed, scales, input, rows, columns, group_size)?;
    let mut output = vec![0.0; rows];
    project_q4_validated(
        packed,
        scales,
        input,
        rows,
        columns,
        group_size,
        &mut output,
    )?;
    Ok(output)
}

/// Project a grouped-Q4 matrix into caller-owned storage so token-generation
/// layers can reuse output memory. On a numerical failure the destination is
/// zeroed before returning.
pub fn project_q4_into(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    if let Err(error) = validate_q4_matrix(packed, scales, input, rows, columns, group_size) {
        output.fill(0.0);
        return Err(error);
    }
    if output.len() != rows {
        output.fill(0.0);
        return Err("Q4 projection output length does not match the matrix rows");
    }
    if let Err(error) =
        project_q4_validated(packed, scales, input, rows, columns, group_size, output)
    {
        output.fill(0.0);
        return Err(error);
    }
    Ok(())
}

/// Project a long-lived grouped-Q4 matrix through Sage's bounded persistent
/// CPU workers. Workers write only their assigned rows into `output`; the
/// function waits for every dispatched range before returning, so the caller's
/// input and output slices remain borrowed for the complete operation.
pub fn project_q4_into_pooled(
    packed: Arc<Vec<u8>>,
    scales: Arc<Vec<f32>>,
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    if let Err(error) = validate_q4_matrix(&packed, &scales, input, rows, columns, group_size) {
        output.fill(0.0);
        return Err(error);
    }
    if output.len() != rows {
        output.fill(0.0);
        return Err("Q4 projection output length does not match the matrix rows");
    }
    if input.iter().any(|value| !value.is_finite()) {
        output.fill(0.0);
        return Err("Q4 projection input contains a non-finite value");
    }

    let enough_work = rows >= MIN_PARALLEL_Q4_ROWS
        && rows
            .checked_mul(columns)
            .is_some_and(|elements| elements >= MIN_PARALLEL_Q4_ELEMENTS);
    if !enough_work {
        return project_q4_into(&packed, &scales, input, rows, columns, group_size, output);
    }

    let pool = q4_cpu_worker_pool();
    let worker_count = pool.worker_count.min(rows);
    if worker_count <= 1 {
        return project_q4_into(&packed, &scales, input, rows, columns, group_size, output);
    }

    #[cfg(target_arch = "aarch64")]
    let use_scalar = {
        let maximum_input = input
            .iter()
            .map(|value| f64::from(value.abs()))
            .fold(0.0f64, f64::max);
        let maximum_group_terms = (group_size.min(columns) as f64) * 8.0;
        maximum_input * maximum_group_terms > f64::from(f32::MAX)
    };
    #[cfg(target_arch = "x86_64")]
    let use_scalar = !native_x86_avx2_available();
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let use_scalar = true;

    if let Some(result) = project_q4_into_pooled_parallel(
        pool,
        Arc::clone(&packed),
        Arc::clone(&scales),
        input.as_ptr(),
        output.as_mut_ptr(),
        rows,
        columns,
        group_size,
        worker_count,
        use_scalar,
    ) {
        if result.is_err() {
            output.fill(0.0);
        }
        return result;
    }

    // A closed or saturated pool never weakens result semantics. All submitted
    // ranges have settled before the helper returns `None`, so scoped workers
    // can safely recompute the complete destination.
    project_q4_into(&packed, &scales, input, rows, columns, group_size, output)
}

/// Project a grouped-Q4 matrix and return the index and value of its largest
/// output without materializing the row-sized output vector. Equal values keep
/// the lower row index, matching a dense left-to-right greedy scan. Large
/// projections use the same bounded worker ceiling as `project_q4_into`.
pub fn project_q4_argmax(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
) -> Result<(usize, f32), &'static str> {
    validate_q4_matrix(packed, scales, input, rows, columns, group_size)?;
    if input.iter().any(|value| !value.is_finite()) {
        return Err("Q4 projection input contains a non-finite value");
    }

    let use_scalar = q4_argmax_uses_scalar(input, group_size, columns);

    let worker_count = inference_cpu_worker_limit().min(rows);
    let enough_work = rows >= MIN_PARALLEL_Q4_ROWS
        && rows
            .checked_mul(columns)
            .is_some_and(|elements| elements >= MIN_PARALLEL_Q4_ELEMENTS);
    if enough_work
        && worker_count > 1
        && let Some(result) = project_q4_argmax_parallel(
            packed,
            scales,
            input,
            rows,
            columns,
            group_size,
            worker_count,
            use_scalar,
        )
    {
        return result;
    }

    project_q4_argmax_range(
        packed, scales, input, columns, group_size, 0, rows, use_scalar,
    )
}

/// Project a row-major batch through one grouped-Q4 matrix. Activations are
/// transposed in bounded tiles so each packed weight is reused across up to
/// 32 inputs and AArch64 can process four inputs per NEON vector. `scratch`
/// must hold `columns * min(batch_size, Q4_BATCH_TILE_SIZE)` values; it is
/// cleared on both success and failure. The output remains batch-major.
#[allow(clippy::too_many_arguments)]
pub fn project_q4_batch_into(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    batch_size: usize,
    scratch: &mut [f32],
    output: &mut [f32],
) -> Result<(), &'static str> {
    let validation = validate_q4_batch_matrix(
        packed, scales, input, rows, columns, group_size, batch_size, scratch, output,
    );
    if let Err(error) = validation {
        scratch.fill(0.0);
        output.fill(0.0);
        return Err(error);
    }

    #[cfg(target_arch = "aarch64")]
    let use_scalar = {
        let maximum_input = input
            .iter()
            .map(|value| f64::from(value.abs()))
            .fold(0.0f64, f64::max);
        let maximum_group_terms = (group_size.min(columns) as f64) * 8.0;
        maximum_input * maximum_group_terms > f64::from(f32::MAX)
    };
    #[cfg(not(target_arch = "aarch64"))]
    let use_scalar = true;

    if batch_size == 1 {
        scratch.fill(0.0);
        return project_q4_into(packed, scales, input, rows, columns, group_size, output);
    }
    let tile_size = q4_batch_tile_size(batch_size);
    let tile_count = batch_size.div_ceil(tile_size);
    let worker_count = inference_cpu_worker_limit().min(tile_count);
    let enough_work = rows
        .checked_mul(columns)
        .and_then(|elements| elements.checked_mul(batch_size))
        .is_some_and(|elements| elements >= MIN_PARALLEL_Q4_ELEMENTS);
    let row_worker_count = inference_cpu_worker_limit().min(rows);
    // A single four-input tile shares each decoded Q4 weight across NEON
    // lanes, but scheduling by input would leave the other cores idle. Split
    // its independent output rows while sharing one transposed activation
    // tile across workers.
    if enough_work && worker_count == 1 && batch_size > 1 && row_worker_count > 1 {
        transpose_q4_batch_tile(
            input,
            columns,
            0,
            batch_size,
            &mut scratch[..columns * batch_size],
        );
        if let Some(result) = project_q4_batch_rows_parallel(
            packed,
            scales,
            &scratch[..columns * batch_size],
            rows,
            columns,
            group_size,
            batch_size,
            output,
            row_worker_count,
            use_scalar,
        ) {
            scratch.fill(0.0);
            if result.is_err() {
                output.fill(0.0);
            }
            return result;
        }
        // If thread creation fails, the helper waits for launched workers and
        // returns only after their row ranges have settled. Recompute the full
        // output through the established bounded path below.
    }
    if enough_work
        && worker_count > 1
        && let Some(result) = project_q4_batch_parallel(
            packed,
            scales,
            input,
            rows,
            columns,
            group_size,
            batch_size,
            tile_size,
            output,
            worker_count,
            use_scalar,
        )
    {
        scratch.fill(0.0);
        if result.is_err() {
            output.fill(0.0);
        }
        return result;
    }

    let result = project_q4_batch_range(
        packed, scales, input, rows, columns, group_size, 0, batch_size, tile_size, scratch,
        output, use_scalar,
    );
    scratch.fill(0.0);
    if result.is_err() {
        output.fill(0.0);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn project_q4_batch_rows_parallel(
    packed: &[u8],
    scales: &[f32],
    transposed: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    batch_size: usize,
    output: &mut [f32],
    worker_count: usize,
    use_scalar: bool,
) -> Option<Result<(), &'static str>> {
    let rows_per_worker = rows.div_ceil(worker_count);
    let mut output_by_batch = output.chunks_mut(rows).collect::<Vec<_>>();
    let mut row_partitions = Vec::with_capacity(worker_count);
    for worker_index in 0..worker_count {
        let first_row = worker_index * rows_per_worker;
        if first_row >= rows {
            break;
        }
        let row_count = (rows - first_row).min(rows_per_worker);
        let mut output_segments = Vec::with_capacity(batch_size);
        let mut remaining_batches = Vec::with_capacity(batch_size);
        for remaining_rows in output_by_batch.drain(..) {
            let (segment, remaining) = remaining_rows.split_at_mut(row_count);
            output_segments.push(segment);
            remaining_batches.push(remaining);
        }
        row_partitions.push((first_row, row_count, output_segments));
        output_by_batch = remaining_batches;
    }

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(row_partitions.len());
        let mut spawn_failed = false;
        for (first_row, row_count, mut output_segments) in row_partitions {
            let result = Builder::new()
                .name("sage-q4-batch-row-worker".into())
                .spawn_scoped(scope, move || {
                    project_q4_batch_row_range(
                        packed,
                        scales,
                        transposed,
                        first_row,
                        row_count,
                        columns,
                        group_size,
                        use_scalar,
                        &mut output_segments,
                    )
                });
            match result {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    spawn_failed = true;
                    break;
                }
            }
        }

        let mut worker_error = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    worker_error.get_or_insert(error);
                }
                Err(_) => {
                    worker_error.get_or_insert("Q4 batch row worker panicked");
                }
            }
        }
        if spawn_failed {
            None
        } else {
            Some(worker_error.map_or(Ok(()), Err))
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn project_q4_batch_row_range(
    packed: &[u8],
    scales: &[f32],
    transposed: &[f32],
    first_row: usize,
    row_count: usize,
    columns: usize,
    group_size: usize,
    use_scalar: bool,
    output_segments: &mut [&mut [f32]],
) -> Result<(), &'static str> {
    #[cfg(not(target_arch = "aarch64"))]
    let _ = use_scalar;

    #[cfg(target_arch = "aarch64")]
    if !use_scalar {
        let batch_size = output_segments.len();
        // SAFETY: public batch validation bounds each row and token segment;
        // worker slices are disjoint and the shared transpose is read-only.
        return unsafe {
            project_q4_batch_neon_rows(
                packed,
                scales,
                transposed,
                first_row,
                row_count,
                columns,
                group_size,
                batch_size,
                |batch_index, local_row, value| {
                    output_segments[batch_index][local_row] = value;
                },
            )
        };
    }
    project_q4_batch_scalar_segments(
        packed,
        scales,
        transposed,
        first_row,
        row_count,
        columns,
        group_size,
        output_segments,
    )
}

fn q4_batch_tile_size(batch_size: usize) -> usize {
    let workers = inference_cpu_worker_limit()
        .min(batch_size.div_ceil(4))
        .max(1);
    let target = batch_size.div_ceil(workers);
    target
        .div_ceil(4)
        .saturating_mul(4)
        .min(Q4_BATCH_TILE_SIZE)
        .min(batch_size)
}

#[allow(clippy::too_many_arguments)]
fn project_q4_batch_parallel(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    batch_size: usize,
    tile_size: usize,
    output: &mut [f32],
    worker_count: usize,
    use_scalar: bool,
) -> Option<Result<(), &'static str>> {
    let tile_count = batch_size.div_ceil(tile_size);
    let tiles_per_worker = tile_count.div_ceil(worker_count);
    let batches_per_worker = tiles_per_worker * tile_size;
    let scratch_elements = columns * tile_size.min(batch_size);

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let mut output_tail = output;
        let mut spawn_failed = false;
        for worker_index in 0..worker_count {
            let first_batch = worker_index * batches_per_worker;
            if first_batch >= batch_size {
                break;
            }
            let end_batch = (first_batch + batches_per_worker).min(batch_size);
            let output_elements = (end_batch - first_batch) * rows;
            let (output_chunk, remaining) = output_tail.split_at_mut(output_elements);
            output_tail = remaining;
            let result = Builder::new()
                .name("sage-q4-batch-worker".into())
                .spawn_scoped(scope, move || {
                    let mut scratch = Q4Scratch::new(scratch_elements)?;
                    project_q4_batch_range(
                        packed,
                        scales,
                        input,
                        rows,
                        columns,
                        group_size,
                        first_batch,
                        end_batch,
                        tile_size,
                        &mut scratch.0,
                        output_chunk,
                        use_scalar,
                    )
                });
            match result {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    spawn_failed = true;
                    break;
                }
            }
        }

        let mut worker_error = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    worker_error.get_or_insert(error);
                }
                Err(_) => {
                    worker_error.get_or_insert("Q4 batch worker panicked");
                }
            }
        }
        if spawn_failed {
            None
        } else {
            Some(worker_error.map_or(Ok(()), Err))
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn project_q4_batch_range(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    first_batch: usize,
    end_batch: usize,
    tile_size: usize,
    scratch: &mut [f32],
    output: &mut [f32],
    use_scalar: bool,
) -> Result<(), &'static str> {
    #[cfg(not(target_arch = "aarch64"))]
    let _ = use_scalar;

    for batch_start in (first_batch..end_batch).step_by(tile_size) {
        let tile_count = (end_batch - batch_start).min(tile_size);
        let tile_elements = columns * tile_count;
        let tile_scratch = &mut scratch[..tile_elements];
        transpose_q4_batch_tile(input, columns, batch_start, tile_count, tile_scratch);
        let output_start = (batch_start - first_batch) * rows;
        let output_end = output_start + tile_count * rows;
        let tile_output = &mut output[output_start..output_end];

        #[cfg(target_arch = "aarch64")]
        let tile_result = if use_scalar {
            project_q4_batch_scalar_tile(
                packed,
                scales,
                tile_scratch,
                rows,
                columns,
                group_size,
                tile_output,
            )
        } else {
            // SAFETY: the public wrapper validates matrix, tile, scratch,
            // and output geometry before entering the NEON implementation.
            unsafe {
                project_q4_batch_neon_tile(
                    packed,
                    scales,
                    tile_scratch,
                    rows,
                    columns,
                    group_size,
                    tile_output,
                )
            }
        };
        #[cfg(not(target_arch = "aarch64"))]
        let tile_result = project_q4_batch_scalar_tile(
            packed,
            scales,
            tile_scratch,
            rows,
            columns,
            group_size,
            tile_output,
        );

        tile_scratch.fill(0.0);
        tile_result?;
    }
    Ok(())
}

struct Q4Scratch(Vec<f32>);

impl Q4Scratch {
    fn new(elements: usize) -> Result<Self, &'static str> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| "Q4 batch worker scratch allocation was denied")?;
        values.resize(elements, 0.0);
        Ok(Self(values))
    }
}

impl Drop for Q4Scratch {
    fn drop(&mut self) {
        self.0.fill(0.0);
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_q4_batch_matrix(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    batch_size: usize,
    scratch: &[f32],
    output: &[f32],
) -> Result<(), &'static str> {
    let matrix_elements = rows
        .checked_mul(columns)
        .filter(|elements| *elements > 0 && *elements <= MAX_Q4_ELEMENTS)
        .ok_or("Q4 batch matrix shape is empty, oversized, or overflows")?;
    let input_elements = batch_size
        .checked_mul(columns)
        .filter(|elements| *elements <= Q4_BATCH_MAX_IO_ELEMENTS)
        .ok_or("Q4 batch input size is outside the bounded kernel limit")?;
    let output_elements = batch_size
        .checked_mul(rows)
        .filter(|elements| *elements <= Q4_BATCH_MAX_IO_ELEMENTS)
        .ok_or("Q4 batch output size is outside the bounded kernel limit")?;
    let scratch_elements = columns
        .checked_mul(batch_size.min(Q4_BATCH_TILE_SIZE))
        .filter(|elements| *elements <= Q4_BATCH_MAX_SCRATCH_ELEMENTS)
        .ok_or("Q4 batch transpose scratch exceeds the kernel limit")?;
    if batch_size == 0
        || batch_size > Q4_BATCH_MAX_SIZE
        || rows > 1_048_576
        || columns == 0
        || group_size == 0
        || group_size > MAX_Q4_ELEMENTS
        || input.len() != input_elements
        || output.len() != output_elements
        || scratch.len() < scratch_elements
        || packed.len() != matrix_elements.div_ceil(2)
        || scales.len() != matrix_elements.div_ceil(group_size)
        || input.iter().any(|value| !value.is_finite())
    {
        return Err("Q4 batch projection buffers or dimensions are invalid");
    }
    Ok(())
}

fn transpose_q4_batch_tile(
    input: &[f32],
    columns: usize,
    batch_start: usize,
    tile_count: usize,
    scratch: &mut [f32],
) {
    for (column, output_column) in scratch.chunks_exact_mut(tile_count).enumerate() {
        for (tile_index, value) in output_column.iter_mut().enumerate() {
            *value = input[(batch_start + tile_index) * columns + column];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn project_q4_batch_scalar_tile(
    packed: &[u8],
    scales: &[f32],
    transposed: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    let batch_size = output.len() / rows;
    project_q4_batch_scalar_rows(
        packed,
        scales,
        transposed,
        0,
        rows,
        columns,
        group_size,
        batch_size,
        |batch_index, row, value| output[batch_index * rows + row] = value,
    )
}

#[allow(clippy::too_many_arguments)]
fn project_q4_batch_scalar_segments(
    packed: &[u8],
    scales: &[f32],
    transposed: &[f32],
    first_row: usize,
    row_count: usize,
    columns: usize,
    group_size: usize,
    output_segments: &mut [&mut [f32]],
) -> Result<(), &'static str> {
    let batch_size = output_segments.len();
    project_q4_batch_scalar_rows(
        packed,
        scales,
        transposed,
        first_row,
        row_count,
        columns,
        group_size,
        batch_size,
        |batch_index, row, value| output_segments[batch_index][row] = value,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn project_q4_batch_scalar_rows<F>(
    packed: &[u8],
    scales: &[f32],
    transposed: &[f32],
    first_row: usize,
    row_count: usize,
    columns: usize,
    group_size: usize,
    batch_size: usize,
    mut write_output: F,
) -> Result<(), &'static str>
where
    F: FnMut(usize, usize, f32),
{
    let mut sums = [0.0f64; Q4_BATCH_TILE_SIZE];
    for local_row in 0..row_count {
        sums.fill(0.0);
        let row_start = (first_row + local_row) * columns;
        for column in 0..columns {
            let index = row_start + column;
            let weight = f64::from(signed_q4_at(packed, index) as f32 * scales[index / group_size]);
            let activations = &transposed[column * batch_size..(column + 1) * batch_size];
            for batch_index in 0..batch_size {
                sums[batch_index] += weight * f64::from(activations[batch_index]);
            }
        }
        for (batch_index, sum) in sums.iter().take(batch_size).enumerate() {
            let projected = *sum as f32;
            if !projected.is_finite() {
                return Err("Q4 batch projection result is non-finite");
            }
            write_output(batch_index, local_row, projected);
        }
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn project_q4_batch_neon_tile(
    packed: &[u8],
    scales: &[f32],
    transposed: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    let tile_count = output.len() / rows;
    // SAFETY: callers validate the matrix, transposed tile, and output rows.
    unsafe {
        project_q4_batch_neon_rows(
            packed,
            scales,
            transposed,
            0,
            rows,
            columns,
            group_size,
            tile_count,
            |batch_index, row, value| output[batch_index * rows + row] = value,
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn project_q4_batch_neon_rows<F>(
    packed: &[u8],
    scales: &[f32],
    transposed: &[f32],
    first_row: usize,
    row_count: usize,
    columns: usize,
    group_size: usize,
    tile_count: usize,
    mut write_output: F,
) -> Result<(), &'static str>
where
    F: FnMut(usize, usize, f32),
{
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32,
    };

    let vector_count = tile_count / 4 * 4;
    let vector_groups = vector_count / 4;
    let mut sums = [0.0f64; Q4_BATCH_TILE_SIZE];
    for local_row in 0..row_count {
        sums.fill(0.0);
        let row = first_row + local_row;
        let row_start = row * columns;
        let zero = vdupq_n_f32(0.0);
        let mut column = 0usize;
        while column < columns {
            let first_index = row_start + column;
            let group = first_index / group_size;
            let segment_end = (column + group_size - first_index % group_size).min(columns);
            let scale = scales[group];
            let mut accumulators: [[float32x4_t; 2]; Q4_BATCH_TILE_SIZE / 4] =
                [[zero; 2]; Q4_BATCH_TILE_SIZE / 4];
            let mut element = 0usize;
            while column < segment_end {
                let signed = signed_q4_at(packed, row_start + column) as f32;
                let weight = vdupq_n_f32(signed);
                for (vector_group, vector_accumulators) in
                    accumulators.iter_mut().enumerate().take(vector_groups)
                {
                    let activation_offset = column * tile_count + vector_group * 4;
                    // SAFETY: validation bounds the transposed tile, and each
                    // vector group starts at a complete four-value segment.
                    let activation =
                        unsafe { vld1q_f32(transposed.as_ptr().add(activation_offset)) };
                    let accumulator = &mut vector_accumulators[element & 1];
                    *accumulator = vfmaq_f32(*accumulator, weight, activation);
                }
                element += 1;
                column += 1;
            }

            for (vector_group, vector_accumulators) in
                accumulators.iter().enumerate().take(vector_groups)
            {
                let combined = vaddq_f32(vector_accumulators[0], vector_accumulators[1]);
                let mut group_sums = [0.0f32; 4];
                // SAFETY: `group_sums` contains four writable f32 lanes.
                unsafe { vst1q_f32(group_sums.as_mut_ptr(), combined) };
                for lane in 0..4 {
                    sums[vector_group * 4 + lane] += f64::from(group_sums[lane]) * f64::from(scale);
                }
            }
        }

        // A final partial SIMD group uses the scalar f64 reference arithmetic.
        for tile_index in vector_count..tile_count {
            let mut total = 0.0f64;
            for column in 0..columns {
                let index = row_start + column;
                let weight =
                    f64::from(signed_q4_at(packed, index) as f32 * scales[index / group_size]);
                total += weight * f64::from(transposed[column * tile_count + tile_index]);
            }
            sums[tile_index] = total;
        }

        for (tile_index, sum) in sums.iter().take(tile_count).enumerate() {
            let projected = *sum as f32;
            if !projected.is_finite() {
                return Err("Q4 batch projection result is non-finite");
            }
            write_output(tile_index, local_row, projected);
        }
    }
    Ok(())
}

/// Project only selected rows of a grouped-Q4 matrix into caller-owned
/// storage. This is useful for constrained vocabulary projection, where a
/// grammar permits far fewer tokens than the full vocabulary.
// Keep matrix geometry explicit at this low-level kernel boundary.
#[allow(clippy::too_many_arguments)]
pub fn project_q4_selected_into(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    selected_rows: &[usize],
    output: &mut [f32],
) -> Result<(), &'static str> {
    if let Err(error) = validate_q4_matrix(packed, scales, input, rows, columns, group_size) {
        output.fill(0.0);
        return Err(error);
    }
    if output.len() != selected_rows.len()
        || selected_rows.len() > rows
        || input.iter().any(|value| !value.is_finite())
        || selected_rows.iter().any(|row| *row >= rows)
    {
        output.fill(0.0);
        return Err("Selected Q4 projection rows or output buffer are invalid");
    }
    #[cfg(target_arch = "aarch64")]
    let use_scalar = {
        let maximum_input = input
            .iter()
            .map(|value| f64::from(value.abs()))
            .fold(0.0f64, f64::max);
        let maximum_group_terms = (group_size.min(columns) as f64) * 8.0;
        maximum_input * maximum_group_terms > f64::from(f32::MAX)
    };
    #[cfg(target_arch = "x86_64")]
    let use_scalar = !native_x86_avx2_available();
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let use_scalar = true;

    let worker_count = inference_cpu_worker_limit().min(selected_rows.len());
    let enough_work = selected_rows.len() >= MIN_PARALLEL_Q4_ROWS
        && selected_rows
            .len()
            .checked_mul(columns)
            .is_some_and(|elements| elements >= MIN_PARALLEL_Q4_ELEMENTS);
    if enough_work
        && worker_count > 1
        && let Some(result) = project_q4_selected_parallel(
            packed,
            scales,
            input,
            columns,
            group_size,
            selected_rows,
            output,
            worker_count,
            use_scalar,
        )
    {
        if result.is_err() {
            output.fill(0.0);
        }
        return result;
    }

    let result = project_q4_selected_range(
        packed,
        scales,
        input,
        columns,
        group_size,
        selected_rows,
        output,
        use_scalar,
    );
    if result.is_err() {
        output.fill(0.0);
    }
    result
}

fn validate_q4_matrix(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
) -> Result<(), &'static str> {
    let elements = rows
        .checked_mul(columns)
        .filter(|elements| *elements > 0 && *elements <= MAX_Q4_ELEMENTS)
        .ok_or("Q4 matrix shape is empty, oversized, or overflows")?;
    if rows > 1_048_576
        || columns == 0
        || group_size == 0
        || group_size > MAX_Q4_ELEMENTS
        || input.len() != columns
        || packed.len() != elements.div_ceil(2)
        || scales.len() != elements.div_ceil(group_size)
    {
        return Err("Q4 projection buffers or dimensions are invalid");
    }
    Ok(())
}

fn project_q4_validated(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    // A projection is independent across output rows. Keep small matrices
    // single-threaded, but fan large GEMVs across a bounded number of CPU
    // workers. Sage currently admits one active generation, so this caps the
    // inference lane's CPU use while avoiding per-matrix oversubscription.
    let worker_count = inference_cpu_worker_limit().min(rows);
    let enough_work = rows >= MIN_PARALLEL_Q4_ROWS
        && rows
            .checked_mul(columns)
            .is_some_and(|elements| elements >= MIN_PARALLEL_Q4_ELEMENTS);

    #[cfg(target_arch = "aarch64")]
    let use_scalar = {
        // Advanced SIMD is mandatory in AArch64's architectural baseline.
        // Extreme finite activations use the f64 scalar reference so the
        // unscaled SIMD accumulators cannot overflow before their group scale.
        let maximum_input = input
            .iter()
            .map(|value| f64::from(value.abs()))
            .fold(0.0f64, f64::max);
        let maximum_group_terms = (group_size.min(columns) as f64) * 8.0;
        maximum_input * maximum_group_terms > f64::from(f32::MAX)
    };
    #[cfg(target_arch = "x86_64")]
    let use_scalar = !native_x86_avx2_available();
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let use_scalar = true;

    if enough_work
        && worker_count > 1
        && let Some(result) = project_q4_parallel(
            packed,
            scales,
            input,
            columns,
            group_size,
            output,
            worker_count,
            use_scalar,
        )
    {
        return result;
    }
    // If the OS cannot create a temporary worker, finish safely on the caller
    // instead of making inference depend on thread admission.

    project_q4_range(
        packed, scales, input, columns, group_size, 0, output, use_scalar,
    )
}

/// Return Sage's bounded CPU worker ceiling for independent local inference
/// work. The value is based on available parallelism and never exceeds eight.
pub fn bounded_inference_cpu_worker_count() -> usize {
    inference_cpu_worker_limit()
}

fn inference_cpu_worker_limit() -> usize {
    static WORKER_LIMIT: OnceLock<usize> = OnceLock::new();
    *WORKER_LIMIT.get_or_init(|| {
        thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(MAX_INFERENCE_CPU_WORKERS)
    })
}

#[cfg(target_arch = "x86_64")]
fn native_x86_avx2_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

#[allow(clippy::too_many_arguments)]
fn project_q4_parallel(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    output: &mut [f32],
    worker_count: usize,
    use_scalar: bool,
) -> Option<Result<(), &'static str>> {
    run_q4_workers(output, worker_count, |first_row, output_chunk| {
        project_q4_range(
            packed,
            scales,
            input,
            columns,
            group_size,
            first_row,
            output_chunk,
            use_scalar,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn project_q4_argmax_parallel(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
    worker_count: usize,
    use_scalar: bool,
) -> Option<Result<(usize, f32), &'static str>> {
    let rows_per_worker = rows.div_ceil(worker_count);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let mut spawn_failed = false;
        for worker_index in 0..worker_count {
            let first_row = worker_index * rows_per_worker;
            if first_row >= rows {
                break;
            }
            let end_row = (first_row + rows_per_worker).min(rows);
            let result =
                Builder::new()
                    .name("sage-q4-argmax".into())
                    .spawn_scoped(scope, move || {
                        project_q4_argmax_range(
                            packed, scales, input, columns, group_size, first_row, end_row,
                            use_scalar,
                        )
                    });
            match result {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    spawn_failed = true;
                    break;
                }
            }
        }

        let mut best = None;
        let mut worker_error = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(candidate)) => keep_q4_argmax(&mut best, candidate),
                Ok(Err(error)) => {
                    worker_error.get_or_insert(error);
                }
                Err(_) => {
                    worker_error.get_or_insert("Q4 argmax worker panicked");
                }
            }
        }
        if spawn_failed {
            None
        } else if let Some(error) = worker_error {
            Some(Err(error))
        } else {
            Some(best.ok_or("Q4 argmax did not project any rows"))
        }
    })
}

struct Q4CpuWorkerPool {
    sender: SyncSender<Q4ProjectionWork>,
    worker_count: usize,
}

struct Q4ProjectionWork {
    packed: Arc<Vec<u8>>,
    scales: Arc<Vec<f32>>,
    input: *const f32,
    output: *mut f32,
    columns: usize,
    group_size: usize,
    first_row: usize,
    end_row: usize,
    use_scalar: bool,
    completion: SyncSender<Result<(), &'static str>>,
}

// SAFETY: projection work is created only by `project_q4_into_pooled`, which
// holds the borrowed input/output slices until every submitted range completes.
// The input is shared read-only, and the submitter assigns disjoint output rows.
unsafe impl Send for Q4ProjectionWork {}

fn q4_cpu_worker_pool() -> &'static Q4CpuWorkerPool {
    static POOL: OnceLock<Q4CpuWorkerPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let requested_workers = inference_cpu_worker_limit();
        let (sender, receiver) =
            mpsc::sync_channel::<Q4ProjectionWork>(requested_workers.max(1) * 2);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut worker_count = 0;
        for worker_index in 0..requested_workers {
            let receiver = Arc::clone(&receiver);
            let result = Builder::new()
                .name(format!("sage-q4-pool-{worker_index}"))
                .spawn(move || q4_cpu_worker(receiver));
            if result.is_ok() {
                worker_count += 1;
            } else {
                break;
            }
        }
        Q4CpuWorkerPool {
            sender,
            worker_count,
        }
    })
}

fn q4_cpu_worker(receiver: Arc<Mutex<mpsc::Receiver<Q4ProjectionWork>>>) {
    loop {
        let work = match receiver.lock() {
            Ok(receiver) => receiver.recv(),
            Err(_) => return,
        };
        let Ok(work) = work else {
            return;
        };
        let Q4ProjectionWork {
            packed,
            scales,
            input: input_pointer,
            output: output_pointer,
            columns,
            group_size,
            first_row,
            end_row,
            use_scalar,
            completion,
        } = work;
        let result = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: the submitting call keeps its input and output slices
            // alive until it receives this job's completion. Row ranges are
            // disjoint; each worker creates only its own mutable range.
            let input = unsafe { std::slice::from_raw_parts(input_pointer, columns) };
            let output_length = end_row - first_row;
            let output = unsafe {
                std::slice::from_raw_parts_mut(output_pointer.add(first_row), output_length)
            };
            project_q4_range(
                &packed, &scales, input, columns, group_size, first_row, output, use_scalar,
            )
        }))
        .unwrap_or(Err("Q4 persistent projection worker panicked"));
        drop(packed);
        drop(scales);
        let _ = completion.send(result);
    }
}

#[allow(clippy::too_many_arguments)]
fn project_q4_into_pooled_parallel(
    pool: &Q4CpuWorkerPool,
    packed: Arc<Vec<u8>>,
    scales: Arc<Vec<f32>>,
    input: *const f32,
    output: *mut f32,
    rows: usize,
    columns: usize,
    group_size: usize,
    worker_count: usize,
    use_scalar: bool,
) -> Option<Result<(), &'static str>> {
    let rows_per_worker = rows.div_ceil(worker_count);
    let job_count = rows.div_ceil(rows_per_worker);
    let (completion, completed) = mpsc::sync_channel(job_count);
    let mut sent_count = 0;
    for worker_index in 0..job_count {
        let first_row = worker_index * rows_per_worker;
        let work = Q4ProjectionWork {
            packed: Arc::clone(&packed),
            scales: Arc::clone(&scales),
            input,
            output,
            columns,
            group_size,
            first_row,
            end_row: (first_row + rows_per_worker).min(rows),
            use_scalar,
            completion: completion.clone(),
        };
        if pool.sender.send(work).is_err() {
            break;
        }
        sent_count += 1;
    }
    drop(completion);

    let send_succeeded = sent_count == job_count;
    let mut worker_error = None;
    let mut completed_count = 0;
    while completed_count < sent_count {
        match completed.recv() {
            Ok(Ok(())) => completed_count += 1,
            Ok(Err(error)) => {
                worker_error.get_or_insert(error);
                completed_count += 1;
            }
            Err(_) => break,
        }
    }

    if !send_succeeded || completed_count != job_count {
        None
    } else {
        Some(worker_error.map_or(Ok(()), Err))
    }
}

#[cfg(target_arch = "aarch64")]
fn q4_argmax_uses_scalar(input: &[f32], group_size: usize, columns: usize) -> bool {
    // Advanced SIMD is mandatory in AArch64's architectural baseline.
    // Extreme finite activations use the f64 scalar reference so the
    // unscaled SIMD accumulators cannot overflow before their group scale.
    let maximum_input = input
        .iter()
        .map(|value| f64::from(value.abs()))
        .fold(0.0f64, f64::max);
    let maximum_group_terms = (group_size.min(columns) as f64) * 8.0;
    maximum_input * maximum_group_terms > f64::from(f32::MAX)
}

#[cfg(target_arch = "x86_64")]
fn q4_argmax_uses_scalar(_input: &[f32], _group_size: usize, _columns: usize) -> bool {
    !native_x86_avx2_available()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn q4_argmax_uses_scalar(_input: &[f32], _group_size: usize, _columns: usize) -> bool {
    true
}

#[allow(clippy::too_many_arguments)]
fn project_q4_argmax_range(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    first_row: usize,
    end_row: usize,
    use_scalar: bool,
) -> Result<(usize, f32), &'static str> {
    let mut best = None;
    for row in first_row..end_row {
        let projected = if use_scalar {
            project_q4_scalar_row(packed, scales, input, row, columns, group_size)?
        } else {
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: the public entry point validated matrix geometry and
                // finite input; this row is within the validated row range.
                unsafe { project_q4_neon_row(packed, scales, input, row, columns, group_size)? }
            }
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: public dispatch selects this function only after
                // runtime detection confirms AVX2 support and row geometry is validated.
                unsafe { project_q4_avx2_row(packed, scales, input, row, columns, group_size)? }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                project_q4_scalar_row(packed, scales, input, row, columns, group_size)?
            }
        };
        keep_q4_argmax(&mut best, (row, projected));
    }
    best.ok_or("Q4 argmax did not project any rows")
}

fn keep_q4_argmax(best: &mut Option<(usize, f32)>, candidate: (usize, f32)) {
    if best.is_none_or(|(best_row, best_value)| {
        candidate.1 > best_value || (candidate.1 == best_value && candidate.0 < best_row)
    }) {
        *best = Some(candidate);
    }
}

#[allow(clippy::too_many_arguments)]
fn project_q4_selected_parallel(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    selected_rows: &[usize],
    output: &mut [f32],
    worker_count: usize,
    use_scalar: bool,
) -> Option<Result<(), &'static str>> {
    run_q4_workers(output, worker_count, |first_index, output_chunk| {
        let rows = &selected_rows[first_index..first_index + output_chunk.len()];
        project_q4_selected_range(
            packed,
            scales,
            input,
            columns,
            group_size,
            rows,
            output_chunk,
            use_scalar,
        )
    })
}

fn run_q4_workers<F>(
    output: &mut [f32],
    worker_count: usize,
    work: F,
) -> Option<Result<(), &'static str>>
where
    F: Fn(usize, &mut [f32]) -> Result<(), &'static str> + Sync,
{
    let rows_per_worker = output.len().div_ceil(worker_count);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let mut spawn_failed = false;
        for (worker_index, output_chunk) in output.chunks_mut(rows_per_worker).enumerate() {
            let first_index = worker_index * rows_per_worker;
            let work = &work;
            let result = Builder::new()
                .name("sage-q4-worker".into())
                .spawn_scoped(scope, move || work(first_index, output_chunk));
            match result {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    spawn_failed = true;
                    break;
                }
            }
        }

        let mut worker_error = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    worker_error.get_or_insert(error);
                }
                Err(_) => {
                    worker_error.get_or_insert("Q4 projection worker panicked");
                }
            }
        }
        if spawn_failed {
            None
        } else {
            Some(worker_error.map_or(Ok(()), Err))
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn project_q4_range(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    first_row: usize,
    output: &mut [f32],
    use_scalar: bool,
) -> Result<(), &'static str> {
    if use_scalar {
        return project_q4_scalar_range(
            packed, scales, input, columns, group_size, first_row, output,
        );
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the public entry point validated the full Q4 geometry and
        // activation range before splitting the disjoint output row ranges.
        unsafe {
            project_q4_neon_range(
                packed, scales, input, columns, group_size, first_row, output,
            )
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: public dispatch selects this function only after runtime
        // detection confirms AVX2 support; matrix and input bounds are validated.
        unsafe {
            project_q4_avx2_range(
                packed, scales, input, columns, group_size, first_row, output,
            )
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        project_q4_scalar_range(
            packed, scales, input, columns, group_size, first_row, output,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn project_q4_selected_range(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    selected_rows: &[usize],
    output: &mut [f32],
    use_scalar: bool,
) -> Result<(), &'static str> {
    if selected_rows.len() != output.len() {
        return Err("Selected Q4 row chunk and output lengths do not match");
    }
    if use_scalar {
        for (selected_index, row) in selected_rows.iter().copied().enumerate() {
            output[selected_index] =
                project_q4_scalar_row(packed, scales, input, row, columns, group_size)?;
        }
        return Ok(());
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the public entry point validated every selected row and
        // splits output into disjoint chunks before dispatching workers.
        unsafe {
            project_q4_neon_selected(
                packed,
                scales,
                input,
                columns,
                group_size,
                selected_rows,
                output,
            )
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        for (selected_index, row) in selected_rows.iter().copied().enumerate() {
            // SAFETY: public dispatch selects this function only after runtime
            // detection confirms AVX2 support; each selected row was validated.
            output[selected_index] =
                unsafe { project_q4_avx2_row(packed, scales, input, row, columns, group_size)? };
        }
        Ok(())
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for (selected_index, row) in selected_rows.iter().copied().enumerate() {
            output[selected_index] =
                project_q4_scalar_row(packed, scales, input, row, columns, group_size)?;
        }
        Ok(())
    }
}

/// Dot one query head against position-major key rows without allocating a
/// temporary score vector. `output` is filled with one dot product per row.
pub fn dot_rows(
    query: &[f32],
    keys: &[f32],
    row_stride: usize,
    column_offset: usize,
    output: &mut [f64],
) -> Result<(), &'static str> {
    let value_count = output
        .len()
        .checked_mul(row_stride)
        .filter(|count| *count > 0 && *count <= MAX_Q4_ELEMENTS)
        .ok_or("Attention key shape is empty, oversized, or overflows")?;
    if output.is_empty()
        || query.is_empty()
        || query.len() > 1024
        || keys.len() != value_count
        || row_stride == 0
        || column_offset
            .checked_add(query.len())
            .is_none_or(|end| end > row_stride)
    {
        return Err("Attention dot-product dimensions are invalid");
    }
    for (position, result) in output.iter_mut().enumerate() {
        let start = position * row_stride + column_offset;
        let key = &keys[start..start + query.len()];
        #[cfg(target_arch = "aarch64")]
        let dot = unsafe { dot_f32_neon(query, key) };
        #[cfg(not(target_arch = "aarch64"))]
        let dot = dot_f32_scalar(query, key);
        if !dot.is_finite() {
            return Err("Attention dot product is non-finite");
        }
        *result = f64::from(dot);
    }
    Ok(())
}

/// Dot one query head against position-major f32 key rows with f64
/// accumulation. This is the accuracy-preserving attention path used by the
/// Qwen vision reference: AArch64 widens four inputs into two f64 NEON lanes,
/// while other targets use the scalar f64 implementation.
pub fn dot_rows_f64_into(
    query: &[f32],
    keys: &[f32],
    row_stride: usize,
    column_offset: usize,
    output: &mut [f64],
) -> Result<(), &'static str> {
    let value_count = output
        .len()
        .checked_mul(row_stride)
        .filter(|count| *count > 0 && *count <= MAX_Q4_ELEMENTS);
    if output.is_empty()
        || query.is_empty()
        || query.len() > 1024
        || value_count != Some(keys.len())
        || row_stride == 0
        || column_offset
            .checked_add(query.len())
            .is_none_or(|end| end > row_stride)
        || query.iter().any(|value| !value.is_finite())
    {
        output.fill(0.0);
        return Err("Wide attention dot-product dimensions or query are invalid");
    }

    for (position, result) in output.iter_mut().enumerate() {
        let start = position * row_stride + column_offset;
        let key = &keys[start..start + query.len()];
        #[cfg(target_arch = "aarch64")]
        let dot = unsafe { dot_f32_f64_neon(query, key) };
        #[cfg(not(target_arch = "aarch64"))]
        let dot = query.iter().zip(key).fold(0.0f64, |sum, (left, right)| {
            sum + f64::from(*left) * f64::from(*right)
        });
        if !dot.is_finite() {
            output.fill(0.0);
            return Err("Wide attention dot product is non-finite");
        }
        *result = dot;
    }
    Ok(())
}

/// Dot one query against position-major binary16 key rows. The public
/// validation matches `dot_rows`; accumulation widens each cached key to f32
/// before the architecture-specific or scalar dot kernel.
pub fn dot_rows_f16(
    query: &[f32],
    keys: &[u16],
    row_stride: usize,
    column_offset: usize,
    output: &mut [f64],
) -> Result<(), &'static str> {
    let value_count = output
        .len()
        .checked_mul(row_stride)
        .filter(|count| *count > 0 && *count <= MAX_Q4_ELEMENTS)
        .ok_or("Attention key shape is empty, oversized, or overflows")?;
    if output.is_empty()
        || query.is_empty()
        || query.len() > 1024
        || keys.len() != value_count
        || row_stride == 0
        || column_offset
            .checked_add(query.len())
            .is_none_or(|end| end > row_stride)
        || query.iter().any(|value| !value.is_finite())
    {
        return Err("Binary16 attention dot-product dimensions are invalid");
    }
    for (position, result) in output.iter_mut().enumerate() {
        let start = position * row_stride + column_offset;
        let row = &keys[start..start + query.len()];
        #[cfg(target_arch = "aarch64")]
        let dot = unsafe { dot_f16_neon(query, row) };
        #[cfg(not(target_arch = "aarch64"))]
        let dot = dot_f16_scalar(query, row);
        if !dot.is_finite() {
            return Err("Binary16 attention dot product is non-finite");
        }
        *result = f64::from(dot);
    }
    Ok(())
}

/// Dot up to five query heads against one shared position-major KV-head slice.
/// Query rows are `[query_head, dimension]`; scores are `[query_head, position]`.
/// On AArch64 the cached key vector is widened once and reused across all query
/// heads in the group, matching grouped-query attention's shared KV layout.
pub fn dot_rows_f16_grouped(
    queries: &[f32],
    query_heads: usize,
    keys: &[u16],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f64],
) -> Result<(), &'static str> {
    let query_dimension = queries.len().checked_div(query_heads.max(1));
    let score_count = query_heads.checked_mul(rows);
    let value_count = rows.checked_mul(row_stride);
    if query_heads == 0
        || query_heads > MAX_GROUPED_QUERY_HEADS
        || rows == 0
        || row_stride == 0
        || query_dimension.is_none_or(|dimension| dimension == 0 || dimension > 1024)
        || query_dimension.and_then(|dimension| dimension.checked_mul(query_heads))
            != Some(queries.len())
        || score_count != Some(output.len())
        || value_count.is_none_or(|count| count == 0 || count > MAX_Q4_ELEMENTS)
        || value_count != Some(keys.len())
        || query_dimension.is_some_and(|dimension| {
            column_offset
                .checked_add(dimension)
                .is_none_or(|end| end > row_stride)
        })
        || queries.iter().any(|value| !value.is_finite())
    {
        output.fill(0.0);
        return Err("Grouped binary16 attention dot-product dimensions are invalid");
    }
    let dimension = query_dimension.expect("validated query dimension");
    #[cfg(target_arch = "aarch64")]
    if query_heads > 1 {
        // SAFETY: the public wrapper validates every query, score, row, and
        // selected-head range before the grouped NEON implementation runs.
        unsafe {
            dot_rows_f16_grouped_neon(
                queries,
                query_heads,
                dimension,
                keys,
                rows,
                row_stride,
                column_offset,
                output,
            );
        }
    } else if let Err(error) = dot_rows_f16(queries, keys, row_stride, column_offset, output) {
        output.fill(0.0);
        return Err(error);
    }
    #[cfg(not(target_arch = "aarch64"))]
    for query_head in 0..query_heads {
        let query_start = query_head * dimension;
        let score_start = query_head * rows;
        if let Err(error) = dot_rows_f16(
            &queries[query_start..query_start + dimension],
            keys,
            row_stride,
            column_offset,
            &mut output[score_start..score_start + rows],
        ) {
            output.fill(0.0);
            return Err(error);
        }
    }
    if output.iter().all(|score| score.is_finite()) {
        Ok(())
    } else {
        output.fill(0.0);
        Err("Grouped binary16 attention dot product is non-finite")
    }
}

/// Compute grouped binary16 query-key scores, apply a finite scale, and return
/// each query row's maximum in the same pass. Attention uses that maximum to
/// start its stable blockwise softmax, avoiding a second scan of each score
/// block. AArch64 folds the scale and maximum into the NEON dot-product loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupedAttentionScoreConfig {
    pub query_heads: usize,
    pub rows: usize,
    pub row_stride: usize,
    pub column_offset: usize,
    pub scale: f64,
}

pub fn dot_rows_f16_grouped_scaled_max(
    queries: &[f32],
    keys: &[u16],
    config: GroupedAttentionScoreConfig,
    output: &mut [f64],
    maxima: &mut [f64],
) -> Result<(), &'static str> {
    let GroupedAttentionScoreConfig {
        query_heads,
        rows,
        row_stride,
        column_offset,
        scale,
    } = config;
    let query_dimension = queries.len().checked_div(query_heads.max(1));
    let score_count = query_heads.checked_mul(rows);
    let value_count = rows.checked_mul(row_stride);
    if query_heads == 0
        || query_heads > MAX_GROUPED_QUERY_HEADS
        || rows == 0
        || row_stride == 0
        || query_dimension.is_none_or(|dimension| dimension == 0 || dimension > 1024)
        || query_dimension.and_then(|dimension| dimension.checked_mul(query_heads))
            != Some(queries.len())
        || score_count != Some(output.len())
        || maxima.len() != query_heads
        || value_count.is_none_or(|count| count == 0 || count > MAX_Q4_ELEMENTS)
        || value_count != Some(keys.len())
        || query_dimension.is_some_and(|dimension| {
            column_offset
                .checked_add(dimension)
                .is_none_or(|end| end > row_stride)
        })
        || !scale.is_finite()
        || queries.iter().any(|value| !value.is_finite())
    {
        output.fill(0.0);
        maxima.fill(0.0);
        return Err("Grouped binary16 attention dot-product dimensions are invalid");
    }
    let dimension = query_dimension.expect("validated query dimension");
    maxima.fill(f64::NEG_INFINITY);
    #[cfg(target_arch = "aarch64")]
    if query_heads > 1 {
        // SAFETY: the public wrapper validates every query, score, row, and
        // selected-head range before the grouped NEON implementation runs.
        unsafe {
            dot_rows_f16_grouped_scaled_max_neon(
                queries,
                query_heads,
                dimension,
                keys,
                rows,
                row_stride,
                column_offset,
                scale,
                output,
                maxima,
            );
        }
    } else {
        if let Err(error) = dot_rows_f16(queries, keys, row_stride, column_offset, output) {
            output.fill(0.0);
            maxima.fill(0.0);
            return Err(error);
        }
        let maximum = &mut maxima[0];
        for score in output.iter_mut() {
            *score *= scale;
            *maximum = maximum.max(*score);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for query_head in 0..query_heads {
        let query_start = query_head * dimension;
        let score_start = query_head * rows;
        if let Err(error) = dot_rows_f16(
            &queries[query_start..query_start + dimension],
            keys,
            row_stride,
            column_offset,
            &mut output[score_start..score_start + rows],
        ) {
            output.fill(0.0);
            maxima.fill(0.0);
            return Err(error);
        }
        let maximum = &mut maxima[query_head];
        for score in &mut output[score_start..score_start + rows] {
            *score *= scale;
            *maximum = maximum.max(*score);
        }
    }
    if output.iter().all(|score| score.is_finite()) {
        Ok(())
    } else {
        output.fill(0.0);
        maxima.fill(0.0);
        Err("Grouped binary16 attention dot product is non-finite")
    }
}

/// Weighted sum of contiguous head slices stored in position-major rows.
/// `row_stride` is the complete key/value row width and `column_offset` points
/// at the selected value head within each row. The weights are already
/// normalized by the caller's stable softmax.
pub fn weighted_sum_rows(
    values: &[f32],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    columns: usize,
) -> Result<Vec<f32>, &'static str> {
    validate_weighted_sum_inputs(values, weights, rows, row_stride, column_offset, columns)?;
    let mut output = vec![0.0f32; columns];
    weighted_sum_rows_into_validated(
        values,
        weights,
        rows,
        row_stride,
        column_offset,
        &mut output,
    )?;
    Ok(output)
}

/// Write a weighted sum directly into caller-owned output storage.
///
/// This lets attention reuse its final output vector instead of allocating an
/// intermediate result for every query head.
pub fn weighted_sum_rows_into(
    values: &[f32],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    let columns = output.len();
    validate_weighted_sum_inputs(values, weights, rows, row_stride, column_offset, columns)?;
    weighted_sum_rows_into_validated(values, weights, rows, row_stride, column_offset, output)
}

/// Write a weighted sum to caller-owned f64 output with f64 accumulation.
/// AArch64 vectorizes across adjacent value dimensions and uses fused
/// multiply-adds; the scalar path is the portable numerical reference.
pub fn weighted_sum_rows_f64_into(
    values: &[f32],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f64],
) -> Result<(), &'static str> {
    let columns = output.len();
    if let Err(error) =
        validate_weighted_sum_inputs(values, weights, rows, row_stride, column_offset, columns)
    {
        output.fill(0.0);
        return Err(error);
    }
    #[cfg(target_arch = "aarch64")]
    if columns <= MAX_ATTENTION_VALUE_DIMENSION {
        // SAFETY: validation bounds every row, selected value slice, weight,
        // and output dimension consumed by the NEON implementation.
        unsafe {
            weighted_sum_rows_f64_neon(values, weights, rows, row_stride, column_offset, output);
        }
    } else {
        weighted_sum_rows_f64_scalar(values, weights, rows, row_stride, column_offset, output);
    }
    #[cfg(not(target_arch = "aarch64"))]
    weighted_sum_rows_f64_scalar(values, weights, rows, row_stride, column_offset, output);

    if output.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        output.fill(0.0);
        Err("Wide attention weighted sum is non-finite")
    }
}

/// Weighted sum of binary16 value rows into caller-owned output. Supported
/// AArch64 systems widen four half values at a time with NEON integer
/// operations; other targets use the scalar reference conversion.
pub fn weighted_sum_rows_f16_into(
    values: &[u16],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    validate_weighted_sum_shape(
        values.len(),
        weights,
        rows,
        row_stride,
        column_offset,
        output.len(),
    )?;
    #[cfg(target_arch = "aarch64")]
    if output.len() <= MAX_ATTENTION_VALUE_DIMENSION {
        // SAFETY: shape, row, head, and output dimensions were validated
        // above, and the conversion helper uses only validated four-lane rows.
        unsafe {
            weighted_sum_rows_f16_neon(values, weights, rows, row_stride, column_offset, output);
        }
    } else {
        weighted_sum_rows_f16_scalar(values, weights, rows, row_stride, column_offset, output);
    }
    #[cfg(not(target_arch = "aarch64"))]
    weighted_sum_rows_f16_scalar(values, weights, rows, row_stride, column_offset, output);

    if output.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err("Binary16 attention weighted sum is non-finite")
    }
}

/// Compute up to five query heads' weighted value sums while loading each
/// binary16 value row once. `weights` and `output` are head-major.
pub fn weighted_sum_rows_f16_grouped_into(
    values: &[u16],
    weights: &[f64],
    query_heads: usize,
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    let columns = output.len().checked_div(query_heads.max(1));
    let expected_weights = query_heads.checked_mul(rows);
    let expected_output = query_heads.checked_mul(columns.unwrap_or(0));
    let value_count = rows.checked_mul(row_stride);
    if query_heads == 0
        || query_heads > MAX_GROUPED_QUERY_HEADS
        || rows == 0
        || row_stride == 0
        || columns.is_none_or(|count| count == 0)
        || columns.is_some_and(|count| !output.len().is_multiple_of(count))
        || expected_weights != Some(weights.len())
        || expected_output != Some(output.len())
        || value_count.is_none_or(|count| count == 0 || count > MAX_Q4_ELEMENTS)
        || value_count != Some(values.len())
        || columns.is_some_and(|count| {
            column_offset
                .checked_add(count)
                .is_none_or(|end| end > row_stride)
        })
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err("Grouped binary16 attention weighted-sum dimensions are invalid");
    }
    let columns = columns.expect("validated grouped output width");
    #[cfg(target_arch = "aarch64")]
    if query_heads > 1 && columns <= MAX_GROUPED_ATTENTION_VALUE_DIMENSION {
        // SAFETY: all row, head, output, and selected-slice bounds were
        // validated above; the native kernel uses fixed bounded accumulators.
        unsafe {
            weighted_sum_rows_f16_grouped_neon(
                values,
                weights,
                query_heads,
                rows,
                row_stride,
                column_offset,
                columns,
                output,
            );
        }
    } else {
        for query_head in 0..query_heads {
            weighted_sum_rows_f16_into(
                values,
                &weights[query_head * rows..(query_head + 1) * rows],
                rows,
                row_stride,
                column_offset,
                &mut output[query_head * columns..(query_head + 1) * columns],
            )?;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for query_head in 0..query_heads {
        weighted_sum_rows_f16_into(
            values,
            &weights[query_head * rows..(query_head + 1) * rows],
            rows,
            row_stride,
            column_offset,
            &mut output[query_head * columns..(query_head + 1) * columns],
        )?;
    }
    if output.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err("Grouped binary16 attention weighted sum is non-finite")
    }
}

fn validate_weighted_sum_inputs(
    values: &[f32],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    columns: usize,
) -> Result<(), &'static str> {
    validate_weighted_sum_shape(
        values.len(),
        weights,
        rows,
        row_stride,
        column_offset,
        columns,
    )
}

fn validate_weighted_sum_shape(
    values_len: usize,
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    columns: usize,
) -> Result<(), &'static str> {
    let value_count = rows
        .checked_mul(row_stride)
        .filter(|count| *count > 0 && *count <= MAX_Q4_ELEMENTS)
        .ok_or("Attention value shape is empty, oversized, or overflows")?;
    if weights.is_empty()
        || rows != weights.len()
        || values_len != value_count
        || row_stride == 0
        || columns == 0
        || column_offset
            .checked_add(columns)
            .is_none_or(|end| end > row_stride)
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err("Attention weighted-sum dimensions or weights are invalid");
    }
    Ok(())
}

fn weighted_sum_rows_into_validated(
    values: &[f32],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    #[cfg(target_arch = "aarch64")]
    let columns = output.len();
    #[cfg(target_arch = "aarch64")]
    if columns <= MAX_ATTENTION_VALUE_DIMENSION {
        // SAFETY: the function validates the full matrix, head slice, and
        // output dimensions before entering the bounded NEON implementation.
        unsafe {
            weighted_sum_rows_neon(values, weights, rows, row_stride, column_offset, output);
        }
    } else {
        weighted_sum_rows_scalar_into(values, weights, rows, row_stride, column_offset, output);
    }
    #[cfg(not(target_arch = "aarch64"))]
    weighted_sum_rows_scalar_into(values, weights, rows, row_stride, column_offset, output);

    if output.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err("Attention weighted sum is non-finite")
    }
}

/// Update one row-major gated-delta state head and read its post-update value.
/// `key_direction` is the L2-normalized key; `query_direction` is the normalized
/// query already multiplied by `1/sqrt(key_dimension)`.
/// Callers must discard or clear the state if this function returns an error,
/// because a numerical failure can occur after an in-place partial update.
pub fn gated_delta_step(
    state: &mut [f32],
    key_direction: &[f32],
    query_direction: &[f32],
    value: &[f32],
    decay: f32,
    beta: f32,
    output: &mut [f32],
) -> Result<(), &'static str> {
    let elements = key_direction
        .len()
        .checked_mul(value.len())
        .filter(|elements| *elements > 0 && *elements <= MAX_DELTA_HEAD_ELEMENTS)
        .ok_or("Gated-delta head shape is empty, oversized, or overflows")?;
    if key_direction.is_empty()
        || key_direction.len() > MAX_DELTA_KEY_DIMENSION
        || key_direction.len() != query_direction.len()
        || value.is_empty()
        || value.len() > MAX_DELTA_VALUE_DIMENSION
        || state.len() != elements
        || output.len() != value.len()
        || !decay.is_finite()
        || !(0.0..=1.0).contains(&decay)
        || !beta.is_finite()
        || !(0.0..=1.0).contains(&beta)
        || key_direction
            .iter()
            .chain(query_direction)
            .chain(value)
            .any(|number| !number.is_finite())
    {
        return Err("Gated-delta head inputs or gates are invalid");
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: validated dimensions bound every state, vector, and output access.
    unsafe {
        gated_delta_step_neon(
            state,
            key_direction,
            query_direction,
            value,
            decay,
            beta,
            output,
        )?;
    }
    #[cfg(not(target_arch = "aarch64"))]
    gated_delta_step_scalar(
        state,
        key_direction,
        query_direction,
        value,
        decay,
        beta,
        output,
    )?;

    if output.iter().all(|number| number.is_finite()) {
        Ok(())
    } else {
        Err("Gated-delta result is non-finite")
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn gated_delta_step_scalar(
    state: &mut [f32],
    key_direction: &[f32],
    query_direction: &[f32],
    value: &[f32],
    decay: f32,
    beta: f32,
    output: &mut [f32],
) -> Result<(), &'static str> {
    let value_dimension = value.len();
    for value_index in 0..value_dimension {
        let mut prior_read = 0.0f64;
        for (key_index, key) in key_direction.iter().enumerate() {
            prior_read += f64::from(state[key_index * value_dimension + value_index])
                * f64::from(decay)
                * f64::from(*key);
        }
        let correction = f64::from(beta) * (f64::from(value[value_index]) - prior_read);
        let mut read = 0.0f64;
        for key_index in 0..key_direction.len() {
            let state_index = key_index * value_dimension + value_index;
            let updated = (f64::from(decay) * f64::from(state[state_index])
                + correction * f64::from(key_direction[key_index]))
                as f32;
            if !updated.is_finite() {
                return Err("Gated-delta state update is non-finite");
            }
            state[state_index] = updated;
            read += f64::from(updated) * f64::from(query_direction[key_index]);
        }
        output[value_index] = read as f32;
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gated_delta_step_neon(
    state: &mut [f32],
    key_direction: &[f32],
    query_direction: &[f32],
    value: &[f32],
    decay: f32,
    beta: f32,
    output: &mut [f32],
) -> Result<(), &'static str> {
    use std::arch::aarch64::{
        vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_n_f32, vst1q_f32, vsubq_f32,
    };

    let key_dimension = key_direction.len();
    let value_dimension = value.len();
    let vector_values = value_dimension / 4;
    let zero = vdupq_n_f32(0.0);
    // Qwen's admitted head width is bounded at 512. Keep this per-head scratch
    // on the stack so a decode token does not allocate two heap vectors for
    // every value head.
    let mut accumulators = [[zero; 4]; MAX_DELTA_VECTOR_BLOCKS];
    for (key_index, key_weight) in key_direction.iter().enumerate() {
        let row_start = key_index * value_dimension;
        let decay_key = vdupq_n_f32(decay * key_weight);
        let accumulator = key_index % 4;
        for (block, sums) in accumulators[..vector_values].iter_mut().enumerate() {
            let column = block * 4;
            // SAFETY: validated state shape contains the complete row and the
            // vector block fits inside `value_dimension`.
            let prior = unsafe { vld1q_f32(state.as_ptr().add(row_start + column)) };
            sums[accumulator] = vfmaq_f32(sums[accumulator], prior, decay_key);
        }
    }

    let mut correction = [0.0f32; MAX_DELTA_VALUE_DIMENSION];
    for (block, sums) in accumulators[..vector_values].iter().enumerate() {
        let prior = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
        let column = block * 4;
        // SAFETY: the vector block is within the validated value slice.
        let current_value = unsafe { vld1q_f32(value.as_ptr().add(column)) };
        let next_correction = vmulq_n_f32(vsubq_f32(current_value, prior), beta);
        // SAFETY: the vector block fits inside the bounded correction array.
        unsafe { vst1q_f32(correction.as_mut_ptr().add(column), next_correction) };
    }
    for column in vector_values * 4..value_dimension {
        let mut prior_read = 0.0f64;
        for (key_index, key_weight) in key_direction.iter().enumerate() {
            let matrix_index = key_index * value_dimension + column;
            prior_read +=
                f64::from(state[matrix_index]) * f64::from(decay) * f64::from(*key_weight);
        }
        correction[column] = (f64::from(beta) * (f64::from(value[column]) - prior_read)) as f32;
    }

    accumulators[..vector_values].fill([zero; 4]);
    for key_index in 0..key_dimension {
        let row_start = key_index * value_dimension;
        let key_weight = vdupq_n_f32(key_direction[key_index]);
        let query_weight = vdupq_n_f32(query_direction[key_index]);
        let accumulator = key_index % 4;
        for (block, sums) in accumulators[..vector_values].iter_mut().enumerate() {
            let column = block * 4;
            // SAFETY: validated state and correction lengths bound these loads
            // and the corresponding state store.
            let current = unsafe { vld1q_f32(state.as_ptr().add(row_start + column)) };
            let update = unsafe { vld1q_f32(correction.as_ptr().add(column)) };
            let updated = vfmaq_f32(vmulq_n_f32(current, decay), update, key_weight);
            // SAFETY: the state row has a validated complete four-value block.
            unsafe { vst1q_f32(state.as_mut_ptr().add(row_start + column), updated) };
            if state[row_start + column..row_start + column + 4]
                .iter()
                .any(|number| !number.is_finite())
            {
                return Err("Gated-delta state update is non-finite");
            }
            sums[accumulator] = vfmaq_f32(sums[accumulator], updated, query_weight);
        }
    }

    for (block, sums) in accumulators[..vector_values].iter().enumerate() {
        let read = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
        let column = block * 4;
        // SAFETY: the output contains a validated complete four-value block.
        unsafe { vst1q_f32(output.as_mut_ptr().add(column), read) };
        if output[column..column + 4]
            .iter()
            .any(|number| !number.is_finite())
        {
            return Err("Gated-delta output is non-finite");
        }
    }

    for column in vector_values * 4..value_dimension {
        let mut read = 0.0f64;
        for key_index in 0..key_dimension {
            let matrix_index = key_index * value_dimension + column;
            let updated = (f64::from(decay) * f64::from(state[matrix_index])
                + f64::from(correction[column]) * f64::from(key_direction[key_index]))
                as f32;
            if !updated.is_finite() {
                return Err("Gated-delta state update is non-finite");
            }
            state[matrix_index] = updated;
            read += f64::from(updated) * f64::from(query_direction[key_index]);
        }
        output[column] = read as f32;
    }

    Ok(())
}

fn weighted_sum_rows_scalar_into(
    values: &[f32],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f32],
) {
    for (column, result) in output.iter_mut().enumerate() {
        let mut total = 0.0f64;
        for (position, weight) in weights.iter().enumerate().take(rows) {
            let index = position * row_stride + column_offset + column;
            total += *weight * f64::from(values[index]);
        }
        *result = total as f32;
    }
}

fn weighted_sum_rows_f64_scalar(
    values: &[f32],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f64],
) {
    for (column, result) in output.iter_mut().enumerate() {
        let mut total = 0.0f64;
        for (position, weight) in weights.iter().enumerate().take(rows) {
            let index = position * row_stride + column_offset + column;
            total += *weight * f64::from(values[index]);
        }
        *result = total;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_f32_scalar(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).fold(0.0f64, |sum, (left, right)| {
        sum + f64::from(*left) * f64::from(*right)
    }) as f32
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_f16_scalar(left: &[f32], right: &[u16]) -> f32 {
    left.iter().zip(right).fold(0.0f64, |sum, (left, right)| {
        sum + f64::from(*left) * f64::from(f16_bits_to_f32(*right))
    }) as f32
}

fn weighted_sum_rows_f16_scalar(
    values: &[u16],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f32],
) {
    for (column, result) in output.iter_mut().enumerate() {
        let mut total = 0.0f64;
        for (position, weight) in weights.iter().enumerate().take(rows) {
            let index = position * row_stride + column_offset + column;
            total += *weight * f64::from(f16_bits_to_f32(values[index]));
        }
        *result = total as f32;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let mut index = 0usize;
    while index + 16 <= left.len() {
        // SAFETY: the loop bound guarantees all four 4-float windows fit both
        // validated slices.
        let left0 = unsafe { vld1q_f32(left.as_ptr().add(index)) };
        let left1 = unsafe { vld1q_f32(left.as_ptr().add(index + 4)) };
        let left2 = unsafe { vld1q_f32(left.as_ptr().add(index + 8)) };
        let left3 = unsafe { vld1q_f32(left.as_ptr().add(index + 12)) };
        let right0 = unsafe { vld1q_f32(right.as_ptr().add(index)) };
        let right1 = unsafe { vld1q_f32(right.as_ptr().add(index + 4)) };
        let right2 = unsafe { vld1q_f32(right.as_ptr().add(index + 8)) };
        let right3 = unsafe { vld1q_f32(right.as_ptr().add(index + 12)) };
        sum0 = vfmaq_f32(sum0, left0, right0);
        sum1 = vfmaq_f32(sum1, left1, right1);
        sum2 = vfmaq_f32(sum2, left2, right2);
        sum3 = vfmaq_f32(sum3, left3, right3);
        index += 16;
    }
    let combined = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    let mut total = f64::from(vaddvq_f32(combined));
    while index + 4 <= left.len() {
        // SAFETY: the loop bound guarantees both 4-float windows fit.
        let left_vector = unsafe { vld1q_f32(left.as_ptr().add(index)) };
        let right_vector = unsafe { vld1q_f32(right.as_ptr().add(index)) };
        total += f64::from(vaddvq_f32(vfmaq_f32(
            vdupq_n_f32(0.0),
            left_vector,
            right_vector,
        )));
        index += 4;
    }
    while index < left.len() {
        total += f64::from(left[index]) * f64::from(right[index]);
        index += 1;
    }
    total as f32
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_f64_neon(left: &[f32], right: &[f32]) -> f64 {
    use std::arch::aarch64::{
        float64x2_t, vaddq_f64, vaddvq_f64, vcvt_f64_f32, vdupq_n_f64, vfmaq_f64, vget_high_f32,
        vget_low_f32, vld1q_f32,
    };

    let zero = vdupq_n_f64(0.0);
    let mut low_sums: [float64x2_t; 4] = [zero; 4];
    let mut high_sums: [float64x2_t; 4] = [zero; 4];
    let vector_length = left.len() / 4 * 4;
    for start in (0..vector_length).step_by(4) {
        // SAFETY: the loop bound guarantees each four-lane load is within both
        // validated query and key slices.
        let left_values = unsafe { vld1q_f32(left.as_ptr().add(start)) };
        let right_values = unsafe { vld1q_f32(right.as_ptr().add(start)) };
        let accumulator = (start / 4) % 4;
        low_sums[accumulator] = vfmaq_f64(
            low_sums[accumulator],
            vcvt_f64_f32(vget_low_f32(left_values)),
            vcvt_f64_f32(vget_low_f32(right_values)),
        );
        high_sums[accumulator] = vfmaq_f64(
            high_sums[accumulator],
            vcvt_f64_f32(vget_high_f32(left_values)),
            vcvt_f64_f32(vget_high_f32(right_values)),
        );
    }
    let low_total = vaddq_f64(
        vaddq_f64(low_sums[0], low_sums[1]),
        vaddq_f64(low_sums[2], low_sums[3]),
    );
    let high_total = vaddq_f64(
        vaddq_f64(high_sums[0], high_sums[1]),
        vaddq_f64(high_sums[2], high_sums[3]),
    );
    let mut total = vaddvq_f64(vaddq_f64(low_total, high_total));
    for index in vector_length..left.len() {
        total += f64::from(left[index]) * f64::from(right[index]);
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f16_neon(left: &[f32], right: &[u16]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1_u16, vld1q_f32};

    let native_fp16 = native_aarch64_fp16_conversion_available();
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let mut index = 0usize;
    while index + 16 <= left.len() {
        // SAFETY: validated row lengths and loop bounds make every load a
        // complete four-element block.
        let left0 = unsafe { vld1q_f32(left.as_ptr().add(index)) };
        let left1 = unsafe { vld1q_f32(left.as_ptr().add(index + 4)) };
        let left2 = unsafe { vld1q_f32(left.as_ptr().add(index + 8)) };
        let left3 = unsafe { vld1q_f32(left.as_ptr().add(index + 12)) };
        let right0 = unsafe { widen_f16x4_neon(vld1_u16(right.as_ptr().add(index)), native_fp16) };
        let right1 =
            unsafe { widen_f16x4_neon(vld1_u16(right.as_ptr().add(index + 4)), native_fp16) };
        let right2 =
            unsafe { widen_f16x4_neon(vld1_u16(right.as_ptr().add(index + 8)), native_fp16) };
        let right3 =
            unsafe { widen_f16x4_neon(vld1_u16(right.as_ptr().add(index + 12)), native_fp16) };
        sum0 = vfmaq_f32(sum0, left0, right0);
        sum1 = vfmaq_f32(sum1, left1, right1);
        sum2 = vfmaq_f32(sum2, left2, right2);
        sum3 = vfmaq_f32(sum3, left3, right3);
        index += 16;
    }
    let combined = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    let mut total = f64::from(vaddvq_f32(combined));
    while index + 4 <= left.len() {
        // SAFETY: the loop bound guarantees both four-element loads fit.
        let query = unsafe { vld1q_f32(left.as_ptr().add(index)) };
        let keys = unsafe { widen_f16x4_neon(vld1_u16(right.as_ptr().add(index)), native_fp16) };
        total += f64::from(vaddvq_f32(vfmaq_f32(vdupq_n_f32(0.0), query, keys)));
        index += 4;
    }
    while index < left.len() {
        total += f64::from(left[index]) * f64::from(f16_bits_to_f32(right[index]));
        index += 1;
    }
    total as f32
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// Keep the validated geometry explicit at this narrow unsafe SIMD boundary.
#[allow(clippy::too_many_arguments)]
unsafe fn dot_rows_f16_grouped_neon(
    queries: &[f32],
    query_heads: usize,
    dimension: usize,
    keys: &[u16],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f64],
) {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1_u16, vld1q_f32};

    let native_fp16 = native_aarch64_fp16_conversion_available();
    let zero = vdupq_n_f32(0.0);
    for position in 0..rows {
        let row_start = position * row_stride + column_offset;
        let row = &keys[row_start..row_start + dimension];
        let mut accumulators = [[zero; 4]; MAX_GROUPED_QUERY_HEADS];
        let mut index = 0usize;
        while index + 16 <= dimension {
            // SAFETY: the 16-element loop bound proves all four packed key
            // loads fit in the selected row.
            let key0 = unsafe { widen_f16x4_neon(vld1_u16(row.as_ptr().add(index)), native_fp16) };
            let key1 =
                unsafe { widen_f16x4_neon(vld1_u16(row.as_ptr().add(index + 4)), native_fp16) };
            let key2 =
                unsafe { widen_f16x4_neon(vld1_u16(row.as_ptr().add(index + 8)), native_fp16) };
            let key3 =
                unsafe { widen_f16x4_neon(vld1_u16(row.as_ptr().add(index + 12)), native_fp16) };
            for head in 0..query_heads {
                let query = &queries[head * dimension..(head + 1) * dimension];
                // SAFETY: the same 16-element loop bound fits four query loads.
                let query0 = unsafe { vld1q_f32(query.as_ptr().add(index)) };
                let query1 = unsafe { vld1q_f32(query.as_ptr().add(index + 4)) };
                let query2 = unsafe { vld1q_f32(query.as_ptr().add(index + 8)) };
                let query3 = unsafe { vld1q_f32(query.as_ptr().add(index + 12)) };
                accumulators[head][0] = vfmaq_f32(accumulators[head][0], query0, key0);
                accumulators[head][1] = vfmaq_f32(accumulators[head][1], query1, key1);
                accumulators[head][2] = vfmaq_f32(accumulators[head][2], query2, key2);
                accumulators[head][3] = vfmaq_f32(accumulators[head][3], query3, key3);
            }
            index += 16;
        }
        let tail_start = index;
        for head in 0..query_heads {
            let sums = accumulators[head];
            let combined = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
            let mut total = f64::from(vaddvq_f32(combined));
            let query = &queries[head * dimension..(head + 1) * dimension];
            for tail in tail_start..dimension {
                total += f64::from(query[tail]) * f64::from(f16_bits_to_f32(row[tail]));
            }
            output[head * rows + position] = total;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// Keep the validated geometry explicit at this narrow unsafe SIMD boundary.
#[allow(clippy::too_many_arguments)]
unsafe fn dot_rows_f16_grouped_scaled_max_neon(
    queries: &[f32],
    query_heads: usize,
    dimension: usize,
    keys: &[u16],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    scale: f64,
    output: &mut [f64],
    maxima: &mut [f64],
) {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1_u16, vld1q_f32};

    let native_fp16 = native_aarch64_fp16_conversion_available();
    let zero = vdupq_n_f32(0.0);
    for position in 0..rows {
        let row_start = position * row_stride + column_offset;
        let row = &keys[row_start..row_start + dimension];
        let mut accumulators = [[zero; 4]; MAX_GROUPED_QUERY_HEADS];
        let mut index = 0usize;
        while index + 16 <= dimension {
            // SAFETY: the 16-element loop bound proves all four packed key
            // loads fit in the selected row.
            let key0 = unsafe { widen_f16x4_neon(vld1_u16(row.as_ptr().add(index)), native_fp16) };
            let key1 =
                unsafe { widen_f16x4_neon(vld1_u16(row.as_ptr().add(index + 4)), native_fp16) };
            let key2 =
                unsafe { widen_f16x4_neon(vld1_u16(row.as_ptr().add(index + 8)), native_fp16) };
            let key3 =
                unsafe { widen_f16x4_neon(vld1_u16(row.as_ptr().add(index + 12)), native_fp16) };
            for head in 0..query_heads {
                let query = &queries[head * dimension..(head + 1) * dimension];
                // SAFETY: the same 16-element loop bound fits four query loads.
                let query0 = unsafe { vld1q_f32(query.as_ptr().add(index)) };
                let query1 = unsafe { vld1q_f32(query.as_ptr().add(index + 4)) };
                let query2 = unsafe { vld1q_f32(query.as_ptr().add(index + 8)) };
                let query3 = unsafe { vld1q_f32(query.as_ptr().add(index + 12)) };
                accumulators[head][0] = vfmaq_f32(accumulators[head][0], query0, key0);
                accumulators[head][1] = vfmaq_f32(accumulators[head][1], query1, key1);
                accumulators[head][2] = vfmaq_f32(accumulators[head][2], query2, key2);
                accumulators[head][3] = vfmaq_f32(accumulators[head][3], query3, key3);
            }
            index += 16;
        }
        let tail_start = index;
        for head in 0..query_heads {
            let sums = accumulators[head];
            let combined = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
            let mut total = f64::from(vaddvq_f32(combined));
            let query = &queries[head * dimension..(head + 1) * dimension];
            for tail in tail_start..dimension {
                total += f64::from(query[tail]) * f64::from(f16_bits_to_f32(row[tail]));
            }
            let scaled = total * scale;
            output[head * rows + position] = scaled;
            maxima[head] = maxima[head].max(scaled);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn widen_f16x4_neon(
    values: std::arch::aarch64::uint16x4_t,
    native_fp16: bool,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        vaddq_u32, vandq_u32, vbslq_u32, vceqq_u32, vdupq_n_u32, vld1q_f32, vmovl_u16, vorrq_u32,
        vreinterpretq_f32_u32, vshlq_n_u32, vshrq_n_u32, vst1_u16,
    };

    if native_fp16 {
        let widened;
        // Apple Silicon and other FP16-capable AArch64 CPUs can convert four
        // half lanes in one instruction. Runtime detection keeps this path
        // safe for older AArch64 machines using the same binary.
        unsafe {
            core::arch::asm!(
                "fcvtl {output:v}.4s, {input:v}.4h",
                output = lateout(vreg) widened,
                input = in(vreg) values,
                options(pure, nomem, nostack),
            );
        }
        return widened;
    }

    let mut half_bits = [0_u16; 4];
    // SAFETY: the NEON value always contains exactly four u16 lanes and the
    // local array is a writable four-element destination.
    unsafe { vst1_u16(half_bits.as_mut_ptr(), values) };
    if half_bits
        .iter()
        .any(|bits| bits & 0x7c00 == 0 && bits & 0x03ff != 0)
    {
        let widened = half_bits.map(f16_bits_to_f32);
        // SAFETY: `widened` contains four initialized f32 lanes.
        return unsafe { vld1q_f32(widened.as_ptr()) };
    }

    let expanded = vmovl_u16(values);
    let zero = vdupq_n_u32(0);
    let exponent = vshrq_n_u32(vandq_u32(expanded, vdupq_n_u32(0x7c00)), 10);
    let fraction = vandq_u32(expanded, vdupq_n_u32(0x03ff));
    let sign = vshlq_n_u32(vandq_u32(expanded, vdupq_n_u32(0x8000)), 16);
    let mut exponent_bits = vshlq_n_u32(vaddq_u32(exponent, vdupq_n_u32(112)), 23);
    exponent_bits = vbslq_u32(
        vceqq_u32(exponent, vdupq_n_u32(0x1f)),
        vdupq_n_u32(0x7f80_0000),
        exponent_bits,
    );
    exponent_bits = vbslq_u32(vceqq_u32(exponent, zero), zero, exponent_bits);
    let fraction_bits = vshlq_n_u32(fraction, 13);
    vreinterpretq_f32_u32(vorrq_u32(vorrq_u32(sign, exponent_bits), fraction_bits))
}

#[cfg(target_arch = "aarch64")]
fn native_aarch64_fp16_conversion_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| std::arch::is_aarch64_feature_detected!("fp16"))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn f32_to_f16_bits_neon(values: &[f32], output: &mut [u16]) {
    use std::arch::aarch64::{vld1q_f32, vst1_u16};

    const MIN_NORMAL_F16_AS_F32: f32 = f32::from_bits(0x3880_0000);
    let (value_groups, value_tail) = values.as_chunks::<4>();
    let (output_groups, output_tail) = output.as_chunks_mut::<4>();
    for (source, destination) in value_groups.iter().zip(output_groups) {
        if source.iter().all(|value| {
            value.is_finite() && value.abs() >= MIN_NORMAL_F16_AS_F32 && value.abs() <= 65_504.0
        }) {
            // SAFETY: each iterator item has four readable f32 values and four
            // writable u16 destinations. FP16 support was checked by the
            // caller before entering this target-feature function.
            let input = unsafe { vld1q_f32(source.as_ptr()) };
            let converted;
            // FCVTN rounds to nearest, ties to even, matching Sage's scalar
            // reference for normal finite values. Special and subnormal cases
            // stay on that scalar path to preserve its exact bit behavior.
            unsafe {
                core::arch::asm!(
                    "fcvtn {output:v}.4h, {input:v}.4s",
                    output = lateout(vreg) converted,
                    input = in(vreg) input,
                    options(pure, nomem, nostack),
                );
            }
            // SAFETY: `converted` contains the four half lanes represented by
            // this destination slice.
            unsafe { vst1_u16(destination.as_mut_ptr(), converted) };
        } else {
            for (value, output) in source.iter().zip(destination) {
                *output = f32_to_f16_bits(*value);
            }
        }
    }

    for (value, output) in value_tail.iter().zip(output_tail) {
        *output = f32_to_f16_bits(*value);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn f32_to_f16_bits_checked_neon(
    values: &[f32],
    output: *mut u16,
) -> Result<(), &'static str> {
    use std::arch::aarch64::{
        vabsq_f32, vandq_u32, vcgeq_f32, vcleq_f32, vdupq_n_f32, vld1q_f32, vminvq_u32, vst1_u16,
    };

    const MIN_NORMAL_F16_AS_F32: f32 = f32::from_bits(0x3880_0000);
    const MAX_FINITE_F16_AS_F32: f32 = 65_504.0;
    let (value_groups, value_tail) = values.as_chunks::<4>();
    for (group_index, source) in value_groups.iter().enumerate() {
        // SAFETY: the caller provides `values.len()` output slots and each
        // group addresses four consecutive values within that range.
        let destination = unsafe { output.add(group_index * 4) };
        // SAFETY: each source group contains four readable f32 values. FP16
        // support was checked by the safe caller.
        let input = unsafe { vld1q_f32(source.as_ptr()) };
        let magnitude = vabsq_f32(input);
        let in_range = vandq_u32(
            vcgeq_f32(magnitude, vdupq_n_f32(MIN_NORMAL_F16_AS_F32)),
            vcleq_f32(magnitude, vdupq_n_f32(MAX_FINITE_F16_AS_F32)),
        );
        if vminvq_u32(in_range) == u32::MAX {
            let converted;
            // FCVTN rounds to nearest, ties to even. The range mask excludes
            // subnormals and special values, so this matches the scalar path.
            unsafe {
                core::arch::asm!(
                    "fcvtn {output:v}.4h, {input:v}.4s",
                    output = lateout(vreg) converted,
                    input = in(vreg) input,
                    options(pure, nomem, nostack),
                );
            }
            // SAFETY: the converted register and destination each have four
            // binary16 lanes.
            unsafe { vst1_u16(destination, converted) };
        } else {
            for (lane, value) in source.iter().enumerate() {
                if !value.is_finite() || value.abs() > MAX_FINITE_F16_AS_F32 {
                    return Err("KV cache value is outside the finite binary16 range");
                }
                // SAFETY: `destination` points to this group's four writable
                // slots, and `lane` is in 0..4.
                unsafe { destination.add(lane).write(f32_to_f16_bits(*value)) };
            }
        }
    }

    let tail_start = value_groups.len() * 4;
    for (tail_index, value) in value_tail.iter().enumerate() {
        if !value.is_finite() || value.abs() > MAX_FINITE_F16_AS_F32 {
            return Err("KV cache value is outside the finite binary16 range");
        }
        // SAFETY: the tail starts immediately after the complete groups and
        // contains fewer than four values within the caller's output range.
        unsafe {
            output
                .add(tail_start + tail_index)
                .write(f32_to_f16_bits(*value))
        };
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn weighted_sum_rows_f64_neon(
    values: &[f32],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f64],
) {
    use std::arch::aarch64::{
        float64x2_t, vcvt_f64_f32, vdupq_n_f64, vfmaq_f64, vget_high_f32, vget_low_f32, vld1q_f32,
        vst1q_f64,
    };

    let zero = vdupq_n_f64(0.0);
    let vector_columns = output.len() / 4;
    let mut low_sums: [float64x2_t; MAX_ATTENTION_VECTOR_BLOCKS] =
        [zero; MAX_ATTENTION_VECTOR_BLOCKS];
    let mut high_sums: [float64x2_t; MAX_ATTENTION_VECTOR_BLOCKS] =
        [zero; MAX_ATTENTION_VECTOR_BLOCKS];
    for (position, weight) in weights.iter().enumerate().take(rows) {
        let row = position * row_stride + column_offset;
        let wide_weight = vdupq_n_f64(*weight);
        for block in 0..vector_columns {
            // SAFETY: the validated output width and selected row span ensure
            // this four-value load stays within the requested head slice.
            let packed = unsafe { vld1q_f32(values.as_ptr().add(row + block * 4)) };
            low_sums[block] = vfmaq_f64(
                low_sums[block],
                vcvt_f64_f32(vget_low_f32(packed)),
                wide_weight,
            );
            high_sums[block] = vfmaq_f64(
                high_sums[block],
                vcvt_f64_f32(vget_high_f32(packed)),
                wide_weight,
            );
        }
    }
    for block in 0..vector_columns {
        // SAFETY: each accumulator is exactly two f64 lanes and the output
        // slice has room for four values for every complete block.
        unsafe {
            vst1q_f64(output.as_mut_ptr().add(block * 4), low_sums[block]);
            vst1q_f64(output.as_mut_ptr().add(block * 4 + 2), high_sums[block]);
        }
    }
    for (column, result) in output.iter_mut().enumerate().skip(vector_columns * 4) {
        let mut total = 0.0f64;
        for (position, weight) in weights.iter().enumerate().take(rows) {
            let index = position * row_stride + column_offset + column;
            total += *weight * f64::from(values[index]);
        }
        *result = total;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn weighted_sum_rows_neon(
    values: &[f32],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f32],
) {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32,
    };

    let vector_columns = output.len() / 4;
    let zero = vdupq_n_f32(0.0);
    // Supported Qwen value heads are at most 512 elements. The public wrapper
    // routes wider inputs through the scalar implementation.
    let mut accumulators = [[zero; 4]; MAX_ATTENTION_VECTOR_BLOCKS];
    let complete_rows = rows / 4 * 4;
    for position in (0..complete_rows).step_by(4) {
        let weight0 = vdupq_n_f32(weights[position] as f32);
        let weight1 = vdupq_n_f32(weights[position + 1] as f32);
        let weight2 = vdupq_n_f32(weights[position + 2] as f32);
        let weight3 = vdupq_n_f32(weights[position + 3] as f32);
        for (block, sums) in accumulators[..vector_columns].iter_mut().enumerate() {
            let column = block * 4;
            let row0 = position * row_stride + column_offset + column;
            let row1 = (position + 1) * row_stride + column_offset + column;
            let row2 = (position + 2) * row_stride + column_offset + column;
            let row3 = (position + 3) * row_stride + column_offset + column;
            // SAFETY: validated row count/stride/offset bounds every four-lane
            // load to a complete value-head slice in the input buffer.
            let value0 = unsafe { vld1q_f32(values.as_ptr().add(row0)) };
            let value1 = unsafe { vld1q_f32(values.as_ptr().add(row1)) };
            let value2 = unsafe { vld1q_f32(values.as_ptr().add(row2)) };
            let value3 = unsafe { vld1q_f32(values.as_ptr().add(row3)) };
            sums[0] = vfmaq_f32(sums[0], value0, weight0);
            sums[1] = vfmaq_f32(sums[1], value1, weight1);
            sums[2] = vfmaq_f32(sums[2], value2, weight2);
            sums[3] = vfmaq_f32(sums[3], value3, weight3);
        }
    }

    for (block, sums) in accumulators[..vector_columns].iter().enumerate() {
        let combined: float32x4_t =
            vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
        let mut lanes = [0.0f32; 4];
        // SAFETY: `lanes` is a writable four-float array.
        unsafe { vst1q_f32(lanes.as_mut_ptr(), combined) };
        for (lane, value) in lanes.into_iter().enumerate() {
            output[block * 4 + lane] = value;
        }
    }
    for (column, output_value) in output.iter_mut().enumerate().take(vector_columns * 4) {
        let mut total = f64::from(*output_value);
        for (position, weight) in weights.iter().enumerate().take(rows).skip(complete_rows) {
            let value_index = position * row_stride + column_offset + column;
            total += *weight * f64::from(values[value_index]);
        }
        *output_value = total as f32;
    }
    for (column, output_value) in output.iter_mut().enumerate().skip(vector_columns * 4) {
        let mut total = 0.0f64;
        for (position, weight) in weights.iter().enumerate() {
            let value_index = position * row_stride + column_offset + column;
            total += *weight * f64::from(values[value_index]);
        }
        *output_value = total as f32;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn weighted_sum_rows_f16_neon(
    values: &[u16],
    weights: &[f64],
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    output: &mut [f32],
) {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1_u16, vst1q_f32};

    let native_fp16 = native_aarch64_fp16_conversion_available();
    let vector_columns = output.len() / 4;
    let zero = vdupq_n_f32(0.0);
    let mut accumulators = [[zero; 4]; MAX_ATTENTION_VECTOR_BLOCKS];
    for (position, weight) in weights.iter().enumerate().take(rows) {
        let row_group = position % 4;
        let vector_weight = vdupq_n_f32(*weight as f32);
        for (block, sums) in accumulators[..vector_columns].iter_mut().enumerate() {
            let column = block * 4;
            let value_index = position * row_stride + column_offset + column;
            // SAFETY: validated row count, stride, offset and vector bound make
            // the four u16 values a complete contiguous head fragment.
            let packed = unsafe { vld1_u16(values.as_ptr().add(value_index)) };
            let widened = unsafe { widen_f16x4_neon(packed, native_fp16) };
            sums[row_group] = vfmaq_f32(sums[row_group], widened, vector_weight);
        }
    }

    for (block, sums) in accumulators[..vector_columns].iter().enumerate() {
        let combined = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
        // SAFETY: the output has the validated four-lane block at this index.
        unsafe { vst1q_f32(output.as_mut_ptr().add(block * 4), combined) };
    }
    for (column, result) in output.iter_mut().enumerate().skip(vector_columns * 4) {
        let mut total = 0.0f64;
        for (position, weight) in weights.iter().enumerate().take(rows) {
            let index = position * row_stride + column_offset + column;
            total += *weight * f64::from(f16_bits_to_f32(values[index]));
        }
        *result = total as f32;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// Keep the validated geometry explicit at this narrow unsafe SIMD boundary.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn weighted_sum_rows_f16_grouped_neon(
    values: &[u16],
    weights: &[f64],
    query_heads: usize,
    rows: usize,
    row_stride: usize,
    column_offset: usize,
    columns: usize,
    output: &mut [f32],
) {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1_u16, vst1q_f32};

    let native_fp16 = native_aarch64_fp16_conversion_available();
    let vector_columns = columns / 4;
    let zero = vdupq_n_f32(0.0);
    let mut accumulators =
        [[[zero; MAX_GROUPED_ATTENTION_VECTOR_BLOCKS]; 4]; MAX_GROUPED_QUERY_HEADS];
    for position in 0..rows {
        let row_group = position % 4;
        for block in 0..vector_columns {
            let column = block * 4;
            let value_index = position * row_stride + column_offset + column;
            // SAFETY: validation guarantees a complete four-value read inside
            // this row and a corresponding vector block in every output head.
            let packed = unsafe { vld1_u16(values.as_ptr().add(value_index)) };
            let value = unsafe { widen_f16x4_neon(packed, native_fp16) };
            for head in 0..query_heads {
                let weight = vdupq_n_f32(weights[head * rows + position] as f32);
                accumulators[head][row_group][block] =
                    vfmaq_f32(accumulators[head][row_group][block], value, weight);
            }
        }
    }
    for head in 0..query_heads {
        let output_head = &mut output[head * columns..(head + 1) * columns];
        let (output_vectors, _) = output_head[..vector_columns * 4].as_chunks_mut::<4>();
        for (block, output_values) in output_vectors.iter_mut().enumerate() {
            let combined = vaddq_f32(
                vaddq_f32(accumulators[head][0][block], accumulators[head][1][block]),
                vaddq_f32(accumulators[head][2][block], accumulators[head][3][block]),
            );
            // SAFETY: each vector block maps to four validated output lanes.
            unsafe { vst1q_f32(output_values.as_mut_ptr(), combined) };
        }
        for (column, result) in output_head.iter_mut().enumerate().skip(vector_columns * 4) {
            let mut total = 0.0f64;
            for position in 0..rows {
                let index = position * row_stride + column_offset + column;
                total +=
                    weights[head * rows + position] * f64::from(f16_bits_to_f32(values[index]));
            }
            *result = total as f32;
        }
    }
}

fn project_q4_scalar_range(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    first_row: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    for (offset, output_value) in output.iter_mut().enumerate() {
        *output_value = project_q4_scalar_row(
            packed,
            scales,
            input,
            first_row + offset,
            columns,
            group_size,
        )?;
    }
    Ok(())
}

fn project_q4_scalar_row(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    row: usize,
    columns: usize,
    group_size: usize,
) -> Result<f32, &'static str> {
    let row_start = row * columns;
    let mut sum = 0.0f64;
    for (column, input_value) in input.iter().enumerate() {
        let index = row_start + column;
        let signed = signed_q4_at(packed, index);
        let weight = f64::from(signed as f32 * scales[index / group_size]);
        sum += weight * f64::from(*input_value);
    }
    let projected = sum as f32;
    if !projected.is_finite() {
        return Err("Q4 projection result is non-finite");
    }
    Ok(projected)
}

#[inline]
fn signed_q4_at(packed: &[u8], index: usize) -> i32 {
    let byte = packed[index / 2];
    let nibble = if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    i32::from(nibble) - 8
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn project_q4_avx2_range(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    first_row: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    for (offset, output_value) in output.iter_mut().enumerate() {
        // SAFETY: the public wrapper validates the Q4 geometry and finite
        // activation range before splitting disjoint output rows.
        *output_value = unsafe {
            project_q4_avx2_row(
                packed,
                scales,
                input,
                first_row + offset,
                columns,
                group_size,
            )?
        };
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn project_q4_avx2_row(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    row: usize,
    columns: usize,
    group_size: usize,
) -> Result<f32, &'static str> {
    use std::arch::x86_64::*;

    let row_start = row * columns;
    let mut column = 0usize;
    let mut total = 0.0f64;

    while column < columns {
        let first_index = row_start + column;
        let scale = scales[first_index / group_size];
        let segment_end = (column + group_size - first_index % group_size).min(columns);
        let scale_vector = _mm256_set1_ps(scale);
        let nibble_offset = _mm_set1_epi8(8);
        let mut sum0 = _mm256_setzero_pd();
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();
        let mut scalar_sum = 0.0f64;

        // Align the first packed block to a byte boundary. This also handles
        // odd row widths, whose next row begins in a high nibble.
        if first_index & 1 != 0 {
            let quantized = signed_q4_at(packed, first_index) as f32 * scale;
            scalar_sum += f64::from(quantized) * f64::from(input[column]);
            column += 1;
        }

        while column + 16 <= segment_end {
            let packed_offset = (row_start + column) / 2;
            // SAFETY: the global Q4 index is even and the 16-element bound
            // guarantees eight readable bytes inside the validated matrix.
            let bytes =
                unsafe { _mm_loadl_epi64(packed.as_ptr().add(packed_offset).cast::<__m128i>()) };
            let low = _mm_and_si128(bytes, _mm_set1_epi8(0x0f));
            let high = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0f));
            let first_eight = _mm_sub_epi8(_mm_unpacklo_epi8(low, high), nibble_offset);
            let later_bytes = _mm_srli_si128(bytes, 4);
            let later_low = _mm_and_si128(later_bytes, _mm_set1_epi8(0x0f));
            let later_high = _mm_and_si128(_mm_srli_epi16(later_bytes, 4), _mm_set1_epi8(0x0f));
            let second_eight =
                _mm_sub_epi8(_mm_unpacklo_epi8(later_low, later_high), nibble_offset);
            let quantized0 = _mm256_cvtepi8_epi32(first_eight);
            let quantized1 = _mm256_cvtepi8_epi32(second_eight);
            let weights0 = _mm256_mul_ps(_mm256_cvtepi32_ps(quantized0), scale_vector);
            let weights1 = _mm256_mul_ps(_mm256_cvtepi32_ps(quantized1), scale_vector);
            // SAFETY: all activation lanes lie inside the validated input
            // slice because the vector block ends at `segment_end`.
            let activations0 = unsafe { _mm256_loadu_ps(input.as_ptr().add(column)) };
            // SAFETY: the second eight-value block shares the same validated
            // 16-element bound as the first.
            let activations1 = unsafe { _mm256_loadu_ps(input.as_ptr().add(column + 8)) };

            sum0 = _mm256_add_pd(
                sum0,
                _mm256_mul_pd(
                    _mm256_cvtps_pd(_mm256_castps256_ps128(weights0)),
                    _mm256_cvtps_pd(_mm256_castps256_ps128(activations0)),
                ),
            );
            sum1 = _mm256_add_pd(
                sum1,
                _mm256_mul_pd(
                    _mm256_cvtps_pd(_mm256_extractf128_ps(weights0, 1)),
                    _mm256_cvtps_pd(_mm256_extractf128_ps(activations0, 1)),
                ),
            );
            sum2 = _mm256_add_pd(
                sum2,
                _mm256_mul_pd(
                    _mm256_cvtps_pd(_mm256_castps256_ps128(weights1)),
                    _mm256_cvtps_pd(_mm256_castps256_ps128(activations1)),
                ),
            );
            sum3 = _mm256_add_pd(
                sum3,
                _mm256_mul_pd(
                    _mm256_cvtps_pd(_mm256_extractf128_ps(weights1, 1)),
                    _mm256_cvtps_pd(_mm256_extractf128_ps(activations1, 1)),
                ),
            );
            column += 16;
        }

        let combined = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
        let mut lanes = [0.0f64; 4];
        // SAFETY: `lanes` has exactly four writable f64 values.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), combined) };
        let mut group_sum = scalar_sum + lanes.into_iter().sum::<f64>();
        while column < segment_end {
            let index = row_start + column;
            let quantized = signed_q4_at(packed, index) as f32 * scale;
            group_sum += f64::from(quantized) * f64::from(input[column]);
            column += 1;
        }
        total += group_sum;
    }

    debug_assert_eq!(column, columns);
    let projected = total as f32;
    if projected.is_finite() {
        Ok(projected)
    } else {
        Err("Q4 projection result is non-finite")
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn project_q4_neon_range(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    first_row: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    if group_size == 0
        || group_size < 16
        || !group_size.is_multiple_of(16)
        || !columns.is_multiple_of(group_size)
        || !columns.is_multiple_of(16)
    {
        // SAFETY: the caller established the same validated matrix and input
        // bounds required by the row kernel.
        return unsafe {
            project_q4_neon_range_reference(
                packed, scales, input, columns, group_size, first_row, output,
            )
        };
    }

    let mut first_output = 0;
    let mut row = first_row;
    if row & 1 != 0 && !output.is_empty() {
        // SAFETY: this row belongs to the validated output range.
        output[0] =
            unsafe { project_q4_neon_row(packed, scales, input, row, columns, group_size)? };
        first_output = 1;
        row += 1;
    }

    let paired_length = (output.len() - first_output) / 2 * 2;
    if paired_length > 0 {
        // SAFETY: even row starts, group-aligned rows, and complete paired
        // output ranges are required and checked above.
        unsafe {
            project_q4_neon_paired_range(
                packed,
                scales,
                input,
                columns,
                group_size,
                row,
                &mut output[first_output..first_output + paired_length],
            )?
        };
        row += paired_length;
    }

    if first_output + paired_length < output.len() {
        // SAFETY: the final unpaired row is inside the validated row range.
        output[first_output + paired_length] =
            unsafe { project_q4_neon_row(packed, scales, input, row, columns, group_size)? };
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn project_q4_neon_range_reference(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    first_row: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    for (offset, output_value) in output.iter_mut().enumerate() {
        // SAFETY: the public wrapper validated the matrix and this disjoint
        // output chunk's row range.
        *output_value = unsafe {
            project_q4_neon_row(
                packed,
                scales,
                input,
                first_row + offset,
                columns,
                group_size,
            )?
        };
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn project_q4_neon_paired_range(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    first_row: usize,
    output: &mut [f32],
) -> Result<(), &'static str> {
    use std::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vand_u8, vcombine_u8, vcvtq_f32_s32, vdup_n_u8, vdupq_n_f32,
        vdupq_n_u8, vfmaq_f32, vget_high_s8, vget_high_s16, vget_low_s8, vget_low_s16, vld1_u8,
        vld1q_f32, vmovl_s8, vmovl_s16, vreinterpretq_s8_u8, vshr_n_u8, vsubq_u8, vzip_u8,
    };

    debug_assert_eq!(output.len() % 2, 0);
    debug_assert_eq!(first_row % 2, 0);
    let zero = vdupq_n_f32(0.0);
    let (output_pairs, _) = output.as_chunks_mut::<2>();
    for (pair_index, pair_output) in output_pairs.iter_mut().enumerate() {
        let row0 = first_row + pair_index * 2;
        let row1 = row0 + 1;
        let row0_start = row0 * columns;
        let row1_start = row1 * columns;
        let mut total0 = 0.0f64;
        let mut total1 = 0.0f64;

        for group_start in (0..columns).step_by(group_size) {
            let group_end = group_start + group_size;
            let scale0 = scales[(row0_start + group_start) / group_size];
            let scale1 = scales[(row1_start + group_start) / group_size];
            let mut sum0 = [zero; 4];
            let mut sum1 = [zero; 4];
            let mut column = group_start;

            while column + 16 <= group_end {
                let packed0 = (row0_start + column) / 2;
                let packed1 = (row1_start + column) / 2;
                // SAFETY: rows and groups are even and the 16-element tile
                // contains exactly eight packed bytes in each validated row.
                let bytes0 = unsafe { vld1_u8(packed.as_ptr().add(packed0)) };
                let bytes1 = unsafe { vld1_u8(packed.as_ptr().add(packed1)) };
                let low0 = vand_u8(bytes0, vdup_n_u8(0x0f));
                let high0 = vshr_n_u8::<4>(bytes0);
                let low1 = vand_u8(bytes1, vdup_n_u8(0x0f));
                let high1 = vshr_n_u8::<4>(bytes1);
                let interleaved0 = vzip_u8(low0, high0);
                let interleaved1 = vzip_u8(low1, high1);
                let values0 = vreinterpretq_s8_u8(vsubq_u8(
                    vcombine_u8(interleaved0.0, interleaved0.1),
                    vdupq_n_u8(8),
                ));
                let values1 = vreinterpretq_s8_u8(vsubq_u8(
                    vcombine_u8(interleaved1.0, interleaved1.1),
                    vdupq_n_u8(8),
                ));
                let lower0 = vmovl_s8(vget_low_s8(values0));
                let upper0 = vmovl_s8(vget_high_s8(values0));
                let lower1 = vmovl_s8(vget_low_s8(values1));
                let upper1 = vmovl_s8(vget_high_s8(values1));
                let quantized0 = [
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(lower0))),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(lower0))),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(upper0))),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(upper0))),
                ];
                let quantized1 = [
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(lower1))),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(lower1))),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(upper1))),
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(upper1))),
                ];
                // Load each activation vector once and reuse it for both
                // adjacent output rows in this tile.
                // SAFETY: each four-value window lies within the validated
                // input slice because the tile ends no later than group_end.
                let activations = [
                    unsafe { vld1q_f32(input.as_ptr().add(column)) },
                    unsafe { vld1q_f32(input.as_ptr().add(column + 4)) },
                    unsafe { vld1q_f32(input.as_ptr().add(column + 8)) },
                    unsafe { vld1q_f32(input.as_ptr().add(column + 12)) },
                ];
                for vector in 0..4 {
                    sum0[vector] = vfmaq_f32(sum0[vector], quantized0[vector], activations[vector]);
                    sum1[vector] = vfmaq_f32(sum1[vector], quantized1[vector], activations[vector]);
                }
                column += 16;
            }

            let combined0 = vaddq_f32(vaddq_f32(sum0[0], sum0[1]), vaddq_f32(sum0[2], sum0[3]));
            let combined1 = vaddq_f32(vaddq_f32(sum1[0], sum1[1]), vaddq_f32(sum1[2], sum1[3]));
            let mut group0 = f64::from(vaddvq_f32(combined0));
            let mut group1 = f64::from(vaddvq_f32(combined1));
            while column < group_end {
                group0 +=
                    f64::from(signed_q4_at(packed, row0_start + column)) * f64::from(input[column]);
                group1 +=
                    f64::from(signed_q4_at(packed, row1_start + column)) * f64::from(input[column]);
                column += 1;
            }
            total0 += group0 * f64::from(scale0);
            total1 += group1 * f64::from(scale1);
        }

        let projected0 = total0 as f32;
        let projected1 = total1 as f32;
        if !projected0.is_finite() || !projected1.is_finite() {
            return Err("Q4 projection result is non-finite");
        }
        pair_output[0] = projected0;
        pair_output[1] = projected1;
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn project_q4_neon_selected(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    columns: usize,
    group_size: usize,
    selected_rows: &[usize],
    output: &mut [f32],
) -> Result<(), &'static str> {
    for (output_index, row) in selected_rows.iter().copied().enumerate() {
        // SAFETY: the public wrapper validated every selected row.
        output[output_index] =
            unsafe { project_q4_neon_row(packed, scales, input, row, columns, group_size)? };
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn project_q4_neon_row(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    row: usize,
    columns: usize,
    group_size: usize,
) -> Result<f32, &'static str> {
    use std::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vand_u8, vcombine_u8, vcvtq_f32_s32, vdup_n_u8, vdupq_n_f32,
        vdupq_n_u8, vfmaq_f32, vget_high_s8, vget_high_s16, vget_low_s8, vget_low_s16, vld1_u8,
        vld1q_f32, vld1q_s32, vmovl_s8, vmovl_s16, vreinterpretq_s8_u8, vshr_n_u8, vsubq_u8,
        vzip_u8,
    };

    let row_start = row * columns;
    let row_end = row_start + columns;
    let mut column = 0usize;
    let mut scalar_sum = 0.0f64;
    while column < columns {
        let global_start = row_start + column;
        let group = global_start / group_size;
        let group_end = (global_start + group_size - global_start % group_size).min(row_end);
        let segment_end = group_end - row_start;
        let scale = scales[group];
        let mut group_scalar_sum = 0.0f64;
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        // A row or group can begin on the high half-byte. Consume one
        // scalar element to align the vector unpacker to a packed byte.
        if (row_start + column) & 1 != 0 {
            let index = row_start + column;
            group_scalar_sum += f64::from(signed_q4_at(packed, index)) * f64::from(input[column]);
            column += 1;
        }

        while column + 16 <= segment_end {
            // Four independent accumulators reduce dependency stalls while
            // preserving the quantization group's common scale. One load
            // expands eight packed bytes into sixteen signed Q4 values.
            let packed_offset = (row_start + column) / 2;
            // SAFETY: the even global index and 16-element bound above
            // guarantee eight readable bytes in the validated buffer.
            let bytes = unsafe { vld1_u8(packed.as_ptr().add(packed_offset)) };
            let low = vand_u8(bytes, vdup_n_u8(0x0f));
            let high = vshr_n_u8::<4>(bytes);
            let interleaved = vzip_u8(low, high);
            let nibbles = vcombine_u8(interleaved.0, interleaved.1);
            let signed = vreinterpretq_s8_u8(vsubq_u8(nibbles, vdupq_n_u8(8)));
            let lower = vmovl_s8(vget_low_s8(signed));
            let upper = vmovl_s8(vget_high_s8(signed));
            let q0 = vmovl_s16(vget_low_s16(lower));
            let q1 = vmovl_s16(vget_high_s16(lower));
            let q2 = vmovl_s16(vget_low_s16(upper));
            let q3 = vmovl_s16(vget_high_s16(upper));
            // SAFETY: each activation window lies within the validated input
            // slice because the block is bounded by segment_end.
            let input0 = unsafe { vld1q_f32(input.as_ptr().add(column)) };
            let input1 = unsafe { vld1q_f32(input.as_ptr().add(column + 4)) };
            let input2 = unsafe { vld1q_f32(input.as_ptr().add(column + 8)) };
            let input3 = unsafe { vld1q_f32(input.as_ptr().add(column + 12)) };
            let weights0 = vcvtq_f32_s32(q0);
            let weights1 = vcvtq_f32_s32(q1);
            let weights2 = vcvtq_f32_s32(q2);
            let weights3 = vcvtq_f32_s32(q3);
            sum0 = vfmaq_f32(sum0, weights0, input0);
            sum1 = vfmaq_f32(sum1, weights1, input1);
            sum2 = vfmaq_f32(sum2, weights2, input2);
            sum3 = vfmaq_f32(sum3, weights3, input3);
            column += 16;
        }
        let combined = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        let mut vector_sum = f64::from(vaddvq_f32(combined));
        while column + 4 <= segment_end {
            let q = [
                signed_q4_at(packed, row_start + column),
                signed_q4_at(packed, row_start + column + 1),
                signed_q4_at(packed, row_start + column + 2),
                signed_q4_at(packed, row_start + column + 3),
            ];
            // SAFETY: the four-element windows are within validated slices.
            let input_vector = unsafe { vld1q_f32(input.as_ptr().add(column)) };
            let weights = unsafe { vcvtq_f32_s32(vld1q_s32(q.as_ptr())) };
            vector_sum += f64::from(vaddvq_f32(vfmaq_f32(
                vdupq_n_f32(0.0),
                weights,
                input_vector,
            )));
            column += 4;
        }
        while column < segment_end {
            let index = row_start + column;
            group_scalar_sum += f64::from(signed_q4_at(packed, index)) * f64::from(input[column]);
            column += 1;
        }
        // Every value in this quantization group shares one scale. Keep the
        // vector accumulators in the small integer domain and apply the scale
        // once after reduction instead of once per SIMD lane.
        scalar_sum += (group_scalar_sum + vector_sum) * f64::from(scale);
    }
    let projected = scalar_sum as f32;
    if !projected.is_finite() {
        return Err("Q4 projection result is non-finite");
    }
    Ok(projected)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        GroupedAttentionScoreConfig, MAX_ATTENTION_VALUE_DIMENSION, dot_rows, dot_rows_f16,
        dot_rows_f16_grouped, dot_rows_f16_grouped_scaled_max, dot_rows_f64_into, f16_bits_to_f32,
        f32_to_f16_bits, gated_delta_step, project_q4, project_q4_argmax, project_q4_batch_into,
        project_q4_into, project_q4_into_pooled, project_q4_selected_into,
        rms_norm_silu_gated_into, rms_norm_zero_centered_into, run_q4_workers, weighted_sum_rows,
        weighted_sum_rows_f16_grouped_into, weighted_sum_rows_f16_into, weighted_sum_rows_f64_into,
        weighted_sum_rows_into,
    };

    #[test]
    fn binary16_conversion_round_trips_all_finite_values_and_rounds_ties_even() {
        for bits in 0_u16..=u16::MAX {
            if bits & 0x7c00 != 0x7c00 {
                let value = f16_bits_to_f32(bits);
                assert_eq!(f32_to_f16_bits(value), bits, "binary16 bits {bits:#06x}");
            }
        }
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-1.0), 0xbc00);
        assert_eq!(f32_to_f16_bits(65_504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(f32::from_bits(0x3380_0000)), 0x0001);
        assert_eq!(f32_to_f16_bits(1.0 + 2.0f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16_bits(65_520.0), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NAN) & 0x7c00, 0x7c00);
    }

    #[test]
    fn qwen_zero_centered_rms_kernel_matches_wide_reference_and_clears_errors() {
        for length in [1, 2, 3, 4, 5, 255, 256, 257] {
            let input = (0..length)
                .map(|index| ((index as f32 * 0.037) - 3.0).sin() * 1.7)
                .collect::<Vec<_>>();
            let weight = (0..length)
                .map(|index| ((index as f32 * 0.019) - 2.0).cos() * 0.12)
                .collect::<Vec<_>>();
            let mean_square = input.iter().fold(0.0f64, |sum, value| {
                sum + f64::from(*value) * f64::from(*value)
            }) / length as f64;
            let inverse = (mean_square + 1e-6f64).sqrt().recip();
            let expected = input
                .iter()
                .zip(&weight)
                .map(|(value, offset)| {
                    (f64::from(*value) * inverse * (1.0 + f64::from(*offset))) as f32
                })
                .collect::<Vec<_>>();
            let mut actual = vec![f32::NAN; length];
            rms_norm_zero_centered_into(&input, &weight, 1e-6, &mut actual)
                .expect("valid zero-centered RMS input");
            for (actual, expected) in actual.iter().zip(expected) {
                let tolerance = 2e-7 + expected.abs() * 2e-7;
                assert!((actual - expected).abs() <= tolerance);
            }
        }

        let mut cleared = [99.0; 2];
        assert!(rms_norm_zero_centered_into(&[1.0], &[0.0], 1e-6, &mut cleared).is_err());
        assert_eq!(cleared, [0.0; 2]);
        cleared.fill(99.0);
        assert!(
            rms_norm_zero_centered_into(&[f32::NAN, 1.0], &[0.0, 0.0], 1e-6, &mut cleared).is_err()
        );
        assert_eq!(cleared, [0.0; 2]);
        cleared.fill(99.0);
        assert!(rms_norm_zero_centered_into(&[1.0, 1.0], &[0.0, 0.0], 0.0, &mut cleared).is_err());
        assert_eq!(cleared, [0.0; 2]);
        cleared.fill(99.0);
        assert!(
            rms_norm_zero_centered_into(
                &[1e-20, 0.0],
                &[f32::MAX, 0.0],
                f32::from_bits(1),
                &mut cleared,
            )
            .is_err()
        );
        assert_eq!(cleared, [0.0; 2]);
    }

    #[test]
    fn qwen_silu_gated_rms_kernel_matches_wide_reference_and_clears_errors() {
        for length in [1, 2, 3, 4, 5, 127, 128, 129] {
            let input = (0..length)
                .map(|index| ((index as f32 * 0.031) - 2.0).sin() * 1.3)
                .collect::<Vec<_>>();
            let gate = (0..length)
                .map(|index| ((index as f32 * 0.043) - 2.5).cos() * 4.0)
                .collect::<Vec<_>>();
            let scale = (0..length)
                .map(|index| ((index as f32 * 0.017) - 1.0).sin() * 0.2 + 1.0)
                .collect::<Vec<_>>();
            let mean_square = input.iter().fold(0.0f64, |sum, value| {
                sum + f64::from(*value) * f64::from(*value)
            }) / length as f64;
            let inverse = (mean_square + 1e-6f64).sqrt().recip();
            let expected = input
                .iter()
                .zip(&gate)
                .zip(&scale)
                .map(|((value, gate), scale)| {
                    let normalized = (f64::from(*value) * inverse * f64::from(*scale)) as f32;
                    let silu = *gate / (1.0 + (-*gate).exp());
                    (f64::from(normalized) * f64::from(silu)) as f32
                })
                .collect::<Vec<_>>();
            let mut actual = vec![f32::NAN; length];
            rms_norm_silu_gated_into(&input, &gate, &scale, 1e-6, &mut actual)
                .expect("valid gated RMS input");
            for (actual, expected) in actual.iter().zip(expected) {
                let tolerance = 2e-7 + expected.abs() * 2e-7;
                assert!((actual - expected).abs() <= tolerance);
            }
        }

        let mut cleared = [99.0; 2];
        assert!(rms_norm_silu_gated_into(&[1.0], &[1.0], &[1.0], 1e-6, &mut cleared).is_err());
        assert_eq!(cleared, [0.0; 2]);
        cleared.fill(99.0);
        assert!(
            rms_norm_silu_gated_into(
                &[1.0, 2.0],
                &[f32::NAN, 0.0],
                &[1.0, 1.0],
                1e-6,
                &mut cleared
            )
            .is_err()
        );
        assert_eq!(cleared, [0.0; 2]);
        cleared.fill(99.0);
        assert!(
            rms_norm_silu_gated_into(&[1.0], &[100.0], &[f32::MAX], 1e-6, &mut cleared).is_err()
        );
        assert_eq!(cleared, [0.0; 2]);
    }

    #[test]
    #[ignore = "release-only allocating scalar versus reusable Qwen gated RMS measurement"]
    fn qwen_silu_gated_rms_latency_measurement() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        fn allocating_scalar(input: &[f32], gate: &[f32], scale: &[f32]) -> Vec<f32> {
            let mean_square = input.iter().fold(0.0f64, |sum, value| {
                sum + f64::from(*value) * f64::from(*value)
            }) / input.len() as f64;
            let inverse = (mean_square + 1e-6f64).sqrt().recip();
            let mut output = Vec::with_capacity(input.len());
            for ((value, gate), scale) in input.iter().zip(gate).zip(scale) {
                let normalized = (f64::from(*value) * inverse * f64::from(*scale)) as f32;
                let silu = *gate / (1.0 + (-*gate).exp());
                output.push((f64::from(normalized) * f64::from(silu)) as f32);
            }
            output
        }

        let heads = 32;
        let dimensions = 128;
        let input = (0..heads * dimensions)
            .map(|index| ((index as f32 * 0.013) - 17.0).sin() * 0.8)
            .collect::<Vec<_>>();
        let gate = (0..heads * dimensions)
            .map(|index| ((index as f32 * 0.019) - 11.0).cos() * 2.5)
            .collect::<Vec<_>>();
        let scale = (0..dimensions)
            .map(|index| ((index as f32 * 0.031) - 2.0).sin() * 0.1 + 1.0)
            .collect::<Vec<_>>();
        let mut reused = vec![0.0; heads * dimensions];
        let mut allocating_times = Vec::<Duration>::with_capacity(1_001);
        let mut reused_times = Vec::<Duration>::with_capacity(1_001);

        for sample in 0..1_001 {
            let (first, second) = if sample % 2 == 0 {
                let start = Instant::now();
                for head in 0..heads {
                    let start = head * dimensions;
                    let output = allocating_scalar(
                        &input[start..start + dimensions],
                        &gate[start..start + dimensions],
                        &scale,
                    );
                    black_box(output);
                }
                let allocating = start.elapsed();

                let start = Instant::now();
                for head in 0..heads {
                    let start = head * dimensions;
                    rms_norm_silu_gated_into(
                        &input[start..start + dimensions],
                        &gate[start..start + dimensions],
                        &scale,
                        1e-6,
                        &mut reused[start..start + dimensions],
                    )
                    .expect("reusable gated RMS sample");
                }
                (allocating, start.elapsed())
            } else {
                let start = Instant::now();
                for head in 0..heads {
                    let start = head * dimensions;
                    rms_norm_silu_gated_into(
                        &input[start..start + dimensions],
                        &gate[start..start + dimensions],
                        &scale,
                        1e-6,
                        &mut reused[start..start + dimensions],
                    )
                    .expect("reusable gated RMS sample");
                }
                let reused_duration = start.elapsed();

                let start = Instant::now();
                for head in 0..heads {
                    let start = head * dimensions;
                    let output = allocating_scalar(
                        &input[start..start + dimensions],
                        &gate[start..start + dimensions],
                        &scale,
                    );
                    black_box(output);
                }
                (start.elapsed(), reused_duration)
            };
            allocating_times.push(first);
            reused_times.push(second);
        }
        allocating_times.sort_unstable();
        reused_times.sort_unstable();
        eprintln!(
            "qwen-gated-rms heads={heads} dimensions={dimensions} samples=1001 allocating_p50_us={} allocating_p95_us={} reused_p50_us={} reused_p95_us={}",
            allocating_times[500].as_nanos() / 1_000,
            allocating_times[950].as_nanos() / 1_000,
            reused_times[500].as_nanos() / 1_000,
            reused_times[950].as_nanos() / 1_000,
        );
        black_box(&mut reused);
    }

    #[test]
    #[ignore = "release-only allocating scalar versus reusable NEON Qwen RMS measurement"]
    fn qwen_zero_centered_rms_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        fn allocating_scalar(input: &[f32], weight: &[f32]) -> std::time::Duration {
            let started = Instant::now();
            for _ in 0..20 {
                assert!(input.iter().chain(weight).all(|value| value.is_finite()));
                let mean_square = input.iter().fold(0.0f64, |sum, value| {
                    sum + f64::from(*value) * f64::from(*value)
                }) / input.len() as f64;
                let inverse = (mean_square + 1e-6f64).sqrt().recip();
                let mut output = input
                    .iter()
                    .zip(weight)
                    .map(|(value, offset)| {
                        (f64::from(*value) * inverse * (1.0 + f64::from(*offset))) as f32
                    })
                    .collect::<Vec<_>>();
                assert!(output.iter().all(|value| value.is_finite()));
                black_box(&output);
                output.fill(0.0);
            }
            started.elapsed()
        }

        fn reusable(input: &[f32], weight: &[f32], output: &mut [f32]) -> std::time::Duration {
            let started = Instant::now();
            for _ in 0..20 {
                rms_norm_zero_centered_into(input, weight, 1e-6, output)
                    .expect("valid zero-centered RMS input");
                black_box(&*output);
                output.fill(0.0);
            }
            started.elapsed()
        }

        let input = (0..256)
            .map(|index| ((index as f32 * 0.013) - 2.0).sin())
            .collect::<Vec<_>>();
        let weight = (0..256)
            .map(|index| ((index as f32 * 0.021) - 1.0).cos() * 0.1)
            .collect::<Vec<_>>();
        let mut output = vec![0.0; 256];
        for _ in 0..10 {
            black_box(allocating_scalar(&input, &weight));
            black_box(reusable(&input, &weight, &mut output));
        }
        let mut allocating_samples = Vec::with_capacity(101);
        let mut reused_samples = Vec::with_capacity(101);
        for sample in 0_usize..101 {
            if sample.is_multiple_of(2) {
                allocating_samples.push(allocating_scalar(&input, &weight));
                reused_samples.push(reusable(&input, &weight, &mut output));
            } else {
                reused_samples.push(reusable(&input, &weight, &mut output));
                allocating_samples.push(allocating_scalar(&input, &weight));
            }
        }
        allocating_samples.sort_unstable();
        reused_samples.sort_unstable();
        eprintln!(
            "qwen-rms-heads=20 dimension=256 samples=101 allocating_scalar_p50_ns={} allocating_scalar_p95_ns={} reused_simd_p50_ns={} reused_simd_p95_ns={}",
            allocating_samples[50].as_nanos(),
            allocating_samples[95].as_nanos(),
            reused_samples[50].as_nanos(),
            reused_samples[95].as_nanos(),
        );
    }

    #[test]
    fn binary16_attention_kernels_match_dequantized_reference_for_tails() {
        let rows = 7;
        let row_stride = 13;
        let column_offset = 2;
        let columns = 7;
        let mut keys = (0..rows * row_stride)
            .map(|index| ((index as f32 - 27.0) * 0.019).sin())
            .collect::<Vec<_>>();
        let mut values = (0..rows * row_stride)
            .map(|index| ((index as f32 - 15.0) * 0.027).cos())
            .collect::<Vec<_>>();
        keys[column_offset] = f16_bits_to_f32(0x0001);
        values[column_offset] = f16_bits_to_f32(0x0001);
        let keys = keys
            .iter()
            .map(|value| f32_to_f16_bits(*value))
            .collect::<Vec<_>>();
        let values = values
            .iter()
            .map(|value| f32_to_f16_bits(*value))
            .collect::<Vec<_>>();
        let query = (0..columns)
            .map(|index| ((index as f32 - 3.0) * 0.23).cos())
            .collect::<Vec<_>>();
        let mut scores = vec![0.0f64; rows];
        dot_rows_f16(&query, &keys, row_stride, column_offset, &mut scores)
            .expect("binary16 attention QK");
        for (position, actual) in scores.iter().enumerate() {
            let start = position * row_stride + column_offset;
            let expected = query
                .iter()
                .zip(&keys[start..start + columns])
                .fold(0.0f64, |sum, (left, right)| {
                    sum + f64::from(*left) * f64::from(f16_bits_to_f32(*right))
                });
            assert!((actual - expected).abs() <= 2.0e-5 + expected.abs() * 2.0e-5);
        }

        let weights = [0.04, 0.08, 0.12, 0.16, 0.20, 0.18, 0.22];
        let mut output = vec![f32::NAN; columns];
        weighted_sum_rows_f16_into(
            &values,
            &weights,
            rows,
            row_stride,
            column_offset,
            &mut output,
        )
        .expect("binary16 attention WV");
        for (column, actual) in output.iter().enumerate() {
            let expected = (0..rows).fold(0.0f64, |sum, position| {
                sum + weights[position]
                    * f64::from(f16_bits_to_f32(
                        values[position * row_stride + column_offset + column],
                    ))
            });
            assert!((f64::from(*actual) - expected).abs() <= 2.0e-5 + expected.abs() * 2.0e-5);
        }

        let mut malformed = values.clone();
        malformed[column_offset] = 0x7c00;
        assert!(
            weighted_sum_rows_f16_into(
                &malformed,
                &weights,
                rows,
                row_stride,
                column_offset,
                &mut output,
            )
            .is_err()
        );
    }

    #[test]
    fn grouped_binary16_attention_reuses_kv_rows_and_matches_independent_heads() {
        let query_heads = 4;
        let dimension = 19;
        let rows = 13;
        let row_stride = 80;
        let column_offset = 21;
        let queries = (0..query_heads * dimension)
            .map(|index| ((index * 7 + 5) as f32 * 0.013).sin())
            .collect::<Vec<_>>();
        let keys = (0..rows * row_stride)
            .map(|index| f32_to_f16_bits(((index * 11 + 3) as f32 * 0.017).cos()))
            .collect::<Vec<_>>();
        let values = (0..rows * row_stride)
            .map(|index| f32_to_f16_bits(((index * 13 + 9) as f32 * 0.019).sin()))
            .collect::<Vec<_>>();
        let weights = (0..query_heads * rows)
            .map(|index| f64::from((index % 17 + 1) as f32) / 170.0)
            .collect::<Vec<_>>();

        let mut grouped_scores = vec![0.0; query_heads * rows];
        dot_rows_f16_grouped(
            &queries,
            query_heads,
            &keys,
            rows,
            row_stride,
            column_offset,
            &mut grouped_scores,
        )
        .expect("grouped query-key scores");
        for head in 0..query_heads {
            let mut expected = vec![0.0; rows];
            dot_rows_f16(
                &queries[head * dimension..(head + 1) * dimension],
                &keys,
                row_stride,
                column_offset,
                &mut expected,
            )
            .expect("single-head query-key reference");
            for (actual, expected) in grouped_scores[head * rows..(head + 1) * rows]
                .iter()
                .zip(expected)
            {
                assert!((actual - expected).abs() < 2.0e-5);
            }
        }

        let mut grouped_output = vec![0.0; query_heads * dimension];
        weighted_sum_rows_f16_grouped_into(
            &values,
            &weights,
            query_heads,
            rows,
            row_stride,
            column_offset,
            &mut grouped_output,
        )
        .expect("grouped weighted value sums");
        for head in 0..query_heads {
            let mut expected = vec![0.0; dimension];
            weighted_sum_rows_f16_into(
                &values,
                &weights[head * rows..(head + 1) * rows],
                rows,
                row_stride,
                column_offset,
                &mut expected,
            )
            .expect("single-head weighted value reference");
            for (actual, expected) in grouped_output[head * dimension..(head + 1) * dimension]
                .iter()
                .zip(expected)
            {
                assert!((actual - expected).abs() < 2.0e-5);
            }
        }
    }

    #[test]
    fn grouped_binary16_scaled_scores_and_maxima_match_unscaled_reference() {
        let query_heads = 4;
        let dimension = 19;
        let rows = 13;
        let row_stride = 80;
        let column_offset = 21;
        let scale = 0.125;
        let queries = (0..query_heads * dimension)
            .map(|index| ((index * 7 + 5) as f32 * 0.013).sin())
            .collect::<Vec<_>>();
        let keys = (0..rows * row_stride)
            .map(|index| f32_to_f16_bits(((index * 11 + 3) as f32 * 0.017).cos()))
            .collect::<Vec<_>>();
        let mut reference = vec![0.0; query_heads * rows];
        dot_rows_f16_grouped(
            &queries,
            query_heads,
            &keys,
            rows,
            row_stride,
            column_offset,
            &mut reference,
        )
        .expect("unscaled reference scores");

        let mut actual = vec![0.0; query_heads * rows];
        let mut maxima = vec![0.0; query_heads];
        dot_rows_f16_grouped_scaled_max(
            &queries,
            &keys,
            GroupedAttentionScoreConfig {
                query_heads,
                rows,
                row_stride,
                column_offset,
                scale,
            },
            &mut actual,
            &mut maxima,
        )
        .expect("scaled scores and row maxima");
        for head in 0..query_heads {
            let expected = &reference[head * rows..(head + 1) * rows];
            let observed = &actual[head * rows..(head + 1) * rows];
            let expected_maximum = expected
                .iter()
                .map(|score| score * scale)
                .fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(maxima[head], expected_maximum);
            assert_eq!(
                observed,
                expected
                    .iter()
                    .map(|score| score * scale)
                    .collect::<Vec<_>>()
            );
        }

        actual.fill(1.0);
        maxima.fill(1.0);
        assert!(
            dot_rows_f16_grouped_scaled_max(
                &queries,
                &keys,
                GroupedAttentionScoreConfig {
                    query_heads,
                    rows,
                    row_stride,
                    column_offset,
                    scale: f64::INFINITY,
                },
                &mut actual,
                &mut maxima,
            )
            .is_err()
        );
        assert!(actual.iter().all(|score| *score == 0.0));
        assert!(maxima.iter().all(|maximum| *maximum == 0.0));

        let one_query = [0.25, -0.75];
        let one_head_keys = [
            f32_to_f16_bits(1.0),
            f32_to_f16_bits(0.5),
            f32_to_f16_bits(-0.25),
            f32_to_f16_bits(0.75),
            f32_to_f16_bits(0.5),
            f32_to_f16_bits(-0.5),
        ];
        let mut one_head_reference = [0.0; 3];
        dot_rows_f16(&one_query, &one_head_keys, 2, 0, &mut one_head_reference)
            .expect("one-head unscaled reference");
        let mut one_head_actual = [0.0; 3];
        let mut one_head_maximum = [0.0; 1];
        dot_rows_f16_grouped_scaled_max(
            &one_query,
            &one_head_keys,
            GroupedAttentionScoreConfig {
                query_heads: 1,
                rows: 3,
                row_stride: 2,
                column_offset: 0,
                scale,
            },
            &mut one_head_actual,
            &mut one_head_maximum,
        )
        .expect("one-head scaled scores");
        let expected_one_head = one_head_reference.map(|score| score * scale);
        assert_eq!(one_head_actual, expected_one_head);
        assert_eq!(
            one_head_maximum[0],
            expected_one_head
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max)
        );
    }

    #[test]
    #[ignore = "release-only grouped QK scale-and-maximum fusion measurement"]
    fn qwen_grouped_qk_scale_max_fusion_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        const QUERY_HEADS: usize = 16;
        const KV_HEADS: usize = 4;
        const HEAD_DIMENSION: usize = 256;
        const CONTEXT: usize = 8_192;
        const BLOCK_SIZE: usize = 256;
        const GROUP_SIZE: usize = QUERY_HEADS / KV_HEADS;
        const BLOCKS: usize = CONTEXT / BLOCK_SIZE;
        let scale = (HEAD_DIMENSION as f64).sqrt().recip();
        let queries = (0..QUERY_HEADS * HEAD_DIMENSION)
            .map(|index| (index as f32 * 0.013).sin())
            .collect::<Vec<_>>();
        let keys = (0..KV_HEADS)
            .map(|head| {
                (0..CONTEXT * HEAD_DIMENSION)
                    .map(|index| f32_to_f16_bits(((index * 13 + head * 7) as f32 * 0.0013).sin()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut reference_scores = vec![0.0; QUERY_HEADS * CONTEXT];
        let mut scalar_fused_scores = vec![0.0; QUERY_HEADS * CONTEXT];
        let mut fused_scores = vec![0.0; QUERY_HEADS * CONTEXT];
        let mut reference_maxima = vec![0.0; QUERY_HEADS * BLOCKS];
        let mut scalar_fused_maxima = vec![0.0; QUERY_HEADS * BLOCKS];
        let mut fused_maxima = vec![0.0; QUERY_HEADS * BLOCKS];
        let mut reference_block = vec![0.0; GROUP_SIZE * BLOCK_SIZE];
        let mut scalar_fused_block = vec![0.0; GROUP_SIZE * BLOCK_SIZE];
        let mut fused_block = vec![0.0; GROUP_SIZE * BLOCK_SIZE];
        let mut fused_block_maxima = vec![0.0; GROUP_SIZE];

        let mut run_reference = || {
            let started = Instant::now();
            for block_index in 0..BLOCKS {
                let block_start = block_index * BLOCK_SIZE;
                let block_end = block_start + BLOCK_SIZE;
                for (kv_head, key_bank) in keys.iter().enumerate() {
                    let query_start = kv_head * GROUP_SIZE * HEAD_DIMENSION;
                    let key_start = block_start * HEAD_DIMENSION;
                    let key_end = block_end * HEAD_DIMENSION;
                    dot_rows_f16_grouped(
                        &queries[query_start..query_start + GROUP_SIZE * HEAD_DIMENSION],
                        GROUP_SIZE,
                        &key_bank[key_start..key_end],
                        BLOCK_SIZE,
                        HEAD_DIMENSION,
                        0,
                        &mut reference_block,
                    )
                    .expect("reference grouped QK");
                    for head in 0..GROUP_SIZE {
                        let scores =
                            &mut reference_block[head * BLOCK_SIZE..(head + 1) * BLOCK_SIZE];
                        for score in scores.iter_mut() {
                            *score *= scale;
                        }
                        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let query_head = kv_head * GROUP_SIZE + head;
                        reference_maxima[query_head * BLOCKS + block_index] = maximum;
                        reference_scores
                            [query_head * CONTEXT + block_start..query_head * CONTEXT + block_end]
                            .copy_from_slice(scores);
                        black_box(maximum);
                    }
                }
            }
            black_box(&reference_scores);
            started.elapsed()
        };
        let mut run_scalar_fused = || {
            let started = Instant::now();
            for block_index in 0..BLOCKS {
                let block_start = block_index * BLOCK_SIZE;
                let block_end = block_start + BLOCK_SIZE;
                for (kv_head, key_bank) in keys.iter().enumerate() {
                    let query_start = kv_head * GROUP_SIZE * HEAD_DIMENSION;
                    let key_start = block_start * HEAD_DIMENSION;
                    let key_end = block_end * HEAD_DIMENSION;
                    dot_rows_f16_grouped(
                        &queries[query_start..query_start + GROUP_SIZE * HEAD_DIMENSION],
                        GROUP_SIZE,
                        &key_bank[key_start..key_end],
                        BLOCK_SIZE,
                        HEAD_DIMENSION,
                        0,
                        &mut scalar_fused_block,
                    )
                    .expect("scalar-fused grouped QK");
                    for head in 0..GROUP_SIZE {
                        let scores =
                            &mut scalar_fused_block[head * BLOCK_SIZE..(head + 1) * BLOCK_SIZE];
                        let maximum =
                            scores.iter_mut().fold(f64::NEG_INFINITY, |maximum, score| {
                                *score *= scale;
                                maximum.max(*score)
                            });
                        let query_head = kv_head * GROUP_SIZE + head;
                        scalar_fused_maxima[query_head * BLOCKS + block_index] = maximum;
                        scalar_fused_scores
                            [query_head * CONTEXT + block_start..query_head * CONTEXT + block_end]
                            .copy_from_slice(scores);
                        black_box(maximum);
                    }
                }
            }
            black_box((&scalar_fused_scores, &scalar_fused_maxima));
            started.elapsed()
        };
        let mut run_fused = || {
            let started = Instant::now();
            for block_index in 0..BLOCKS {
                let block_start = block_index * BLOCK_SIZE;
                let block_end = block_start + BLOCK_SIZE;
                for (kv_head, key_bank) in keys.iter().enumerate() {
                    let query_start = kv_head * GROUP_SIZE * HEAD_DIMENSION;
                    let key_start = block_start * HEAD_DIMENSION;
                    let key_end = block_end * HEAD_DIMENSION;
                    dot_rows_f16_grouped_scaled_max(
                        &queries[query_start..query_start + GROUP_SIZE * HEAD_DIMENSION],
                        &key_bank[key_start..key_end],
                        GroupedAttentionScoreConfig {
                            query_heads: GROUP_SIZE,
                            rows: BLOCK_SIZE,
                            row_stride: HEAD_DIMENSION,
                            column_offset: 0,
                            scale,
                        },
                        &mut fused_block,
                        &mut fused_block_maxima,
                    )
                    .expect("fused grouped QK");
                    for head in 0..GROUP_SIZE {
                        let query_head = kv_head * GROUP_SIZE + head;
                        fused_maxima[query_head * BLOCKS + block_index] = fused_block_maxima[head];
                        fused_scores
                            [query_head * CONTEXT + block_start..query_head * CONTEXT + block_end]
                            .copy_from_slice(
                                &fused_block[head * BLOCK_SIZE..(head + 1) * BLOCK_SIZE],
                            );
                    }
                }
            }
            black_box((&fused_scores, &fused_maxima));
            started.elapsed()
        };

        black_box(run_reference());
        black_box(run_scalar_fused());
        black_box(run_fused());
        let mut reference_times = Vec::with_capacity(51);
        let mut scalar_fused_times = Vec::with_capacity(51);
        let mut fused_times = Vec::with_capacity(51);
        for sample in 0_usize..51 {
            match sample % 3 {
                0 => {
                    reference_times.push(run_reference());
                    scalar_fused_times.push(run_scalar_fused());
                    fused_times.push(run_fused());
                }
                1 => {
                    scalar_fused_times.push(run_scalar_fused());
                    fused_times.push(run_fused());
                    reference_times.push(run_reference());
                }
                _ => {
                    fused_times.push(run_fused());
                    reference_times.push(run_reference());
                    scalar_fused_times.push(run_scalar_fused());
                }
            }
        }
        reference_times.sort_unstable();
        scalar_fused_times.sort_unstable();
        fused_times.sort_unstable();
        for index in 0..QUERY_HEADS * CONTEXT {
            assert_eq!(fused_scores[index], reference_scores[index]);
            assert_eq!(scalar_fused_scores[index], reference_scores[index]);
        }
        for head in 0..QUERY_HEADS {
            for block_index in 0..BLOCKS {
                assert_eq!(
                    scalar_fused_maxima[head * BLOCKS + block_index],
                    reference_maxima[head * BLOCKS + block_index]
                );
                assert_eq!(
                    fused_maxima[head * BLOCKS + block_index],
                    reference_maxima[head * BLOCKS + block_index]
                );
            }
        }
        eprintln!(
            "qwen-grouped-qk-scale-max context={CONTEXT} block={BLOCK_SIZE} query_heads={QUERY_HEADS} kv_heads={KV_HEADS} head_dim={HEAD_DIMENSION} samples=51 reference_p50_us={} reference_p95_us={} scalar_fused_p50_us={} scalar_fused_p95_us={} kernel_fused_p50_us={} kernel_fused_p95_us={}",
            reference_times[25].as_nanos() / 1_000,
            reference_times[48].as_nanos() / 1_000,
            scalar_fused_times[25].as_nanos() / 1_000,
            scalar_fused_times[48].as_nanos() / 1_000,
            fused_times[25].as_nanos() / 1_000,
            fused_times[48].as_nanos() / 1_000,
        );
    }

    #[test]
    #[ignore = "release-only binary16 attention-kernel latency measurement"]
    fn binary16_attention_kernel_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        struct AttentionFixture<'a, Value> {
            query: &'a [f32],
            keys: &'a [Value],
            values: &'a [Value],
            weights: &'a [f64],
            rows: usize,
            row_stride: usize,
            column_offset: usize,
        }

        fn run_f32_kernels(
            fixture: &AttentionFixture<'_, f32>,
            scores: &mut [f64],
            output: &mut [f32],
        ) -> std::time::Duration {
            let started = Instant::now();
            dot_rows(
                fixture.query,
                fixture.keys,
                fixture.row_stride,
                fixture.column_offset,
                scores,
            )
            .expect("f32 QK kernel");
            weighted_sum_rows_into(
                fixture.values,
                fixture.weights,
                fixture.rows,
                fixture.row_stride,
                fixture.column_offset,
                output,
            )
            .expect("f32 WV kernel");
            black_box((&scores, &output));
            started.elapsed()
        }

        fn run_f16_kernels(
            fixture: &AttentionFixture<'_, u16>,
            scores: &mut [f64],
            output: &mut [f32],
        ) -> std::time::Duration {
            let started = Instant::now();
            dot_rows_f16(
                fixture.query,
                fixture.keys,
                fixture.row_stride,
                fixture.column_offset,
                scores,
            )
            .expect("binary16 QK kernel");
            weighted_sum_rows_f16_into(
                fixture.values,
                fixture.weights,
                fixture.rows,
                fixture.row_stride,
                fixture.column_offset,
                output,
            )
            .expect("binary16 WV kernel");
            black_box((&scores, &output));
            started.elapsed()
        }

        let rows = 2048;
        let row_stride = 256;
        let column_offset = 128;
        let columns = 64;
        let keys_f16 = (0..rows * row_stride)
            .map(|index| f32_to_f16_bits(((index as f32 - 73.0) * 0.0017).sin()))
            .collect::<Vec<_>>();
        let values_f16 = (0..rows * row_stride)
            .map(|index| f32_to_f16_bits(((index as f32 - 19.0) * 0.0021).cos()))
            .collect::<Vec<_>>();
        let keys_f32 = keys_f16
            .iter()
            .map(|bits| f16_bits_to_f32(*bits))
            .collect::<Vec<_>>();
        let values_f32 = values_f16
            .iter()
            .map(|bits| f16_bits_to_f32(*bits))
            .collect::<Vec<_>>();
        let query = (0..columns)
            .map(|index| ((index as f32 - 17.0) * 0.13).sin())
            .collect::<Vec<_>>();
        let mut weights = (0..rows)
            .map(|index| (((index as f64 - 601.0) * 0.0007).sin() * 0.7).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f64>();
        weights.iter_mut().for_each(|weight| *weight /= denominator);
        let mut scores_f32 = vec![0.0f64; rows];
        let mut scores_f16 = vec![0.0f64; rows];
        let mut output_f32 = vec![0.0f32; columns];
        let mut output_f16 = vec![0.0f32; columns];
        let f32_fixture = AttentionFixture {
            query: &query,
            keys: &keys_f32,
            values: &values_f32,
            weights: &weights,
            rows,
            row_stride,
            column_offset,
        };
        let f16_fixture = AttentionFixture {
            query: &query,
            keys: &keys_f16,
            values: &values_f16,
            weights: &weights,
            rows,
            row_stride,
            column_offset,
        };

        let run_f32 =
            |scores: &mut [f64], output: &mut [f32]| run_f32_kernels(&f32_fixture, scores, output);
        let run_f16 =
            |scores: &mut [f64], output: &mut [f32]| run_f16_kernels(&f16_fixture, scores, output);

        black_box(run_f32(&mut scores_f32, &mut output_f32));
        black_box(run_f16(&mut scores_f16, &mut output_f16));
        let maximum_score_error = scores_f32
            .iter()
            .zip(&scores_f16)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f64, f64::max);
        let maximum_value_error = output_f32
            .iter()
            .zip(&output_f16)
            .map(|(left, right)| f64::from((*left - *right).abs()))
            .fold(0.0f64, f64::max);
        assert!(maximum_score_error < 2.0e-5);
        assert!(maximum_value_error < 2.0e-5);

        for _ in 0..10 {
            black_box(run_f32(&mut scores_f32, &mut output_f32));
            black_box(run_f16(&mut scores_f16, &mut output_f16));
        }
        let mut f32_times = Vec::with_capacity(101);
        let mut f16_times = Vec::with_capacity(101);
        for sample in 0_usize..101 {
            if sample.is_multiple_of(2) {
                f32_times.push(run_f32(&mut scores_f32, &mut output_f32));
                f16_times.push(run_f16(&mut scores_f16, &mut output_f16));
            } else {
                f16_times.push(run_f16(&mut scores_f16, &mut output_f16));
                f32_times.push(run_f32(&mut scores_f32, &mut output_f32));
            }
        }
        f32_times.sort_unstable();
        f16_times.sort_unstable();
        eprintln!(
            "qwen-attention-kernels context={rows} query_dim={columns} row_stride={row_stride} samples=101 f32_p50_us={} f32_p95_us={} f16_p50_us={} f16_p95_us={} max_score_error={maximum_score_error:.8} max_value_error={maximum_value_error:.8}",
            f32_times[50].as_nanos() / 1_000,
            f32_times[95].as_nanos() / 1_000,
            f16_times[50].as_nanos() / 1_000,
            f16_times[95].as_nanos() / 1_000,
        );
    }

    fn scalar_reference(
        packed: &[u8],
        scales: &[f32],
        input: &[f32],
        rows: usize,
        columns: usize,
        group_size: usize,
    ) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                let sum = (0..columns).fold(0.0f64, |sum, column| {
                    let index = row * columns + column;
                    let byte = packed[index / 2];
                    let nibble = if index & 1 == 0 { byte & 15 } else { byte >> 4 };
                    let weight =
                        f64::from((i32::from(nibble) - 8) as f32 * scales[index / group_size]);
                    sum + weight * f64::from(input[column])
                });
                sum as f32
            })
            .collect()
    }

    #[test]
    fn q4_projection_matches_scalar_for_odd_rows_and_crossing_groups() {
        for (rows, columns, group_size) in [(5, 7, 9), (3, 257, 16), (4, 65, 31), (2, 1024, 128)] {
            let elements: usize = rows * columns;
            let quantized = (0..elements)
                .map(|index| ((index * 17 + 5) % 15) as u8)
                .collect::<Vec<_>>();
            let packed = quantized
                .chunks(2)
                .map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(8) << 4))
                .collect::<Vec<_>>();
            let scales = (0..elements.div_ceil(group_size))
                .map(|group| 0.001 + group as f32 * 0.0003)
                .collect::<Vec<_>>();
            let input = (0..columns)
                .map(|column| ((column as f32 - 23.0) * 0.071).sin())
                .collect::<Vec<_>>();
            let actual = project_q4(&packed, &scales, &input, rows, columns, group_size)
                .expect("validated Q4 projection");
            let expected = scalar_reference(&packed, &scales, &input, rows, columns, group_size);
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                let tolerance = 2.0e-5 + expected.abs() * 2.0e-5;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{actual} != {expected}"
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_q4_projection_matches_reference_across_nibble_and_scale_boundaries() {
        if !super::native_x86_avx2_available() {
            return;
        }

        for (rows, columns, group_size) in [(5, 7, 9), (3, 257, 16), (4, 65, 31), (2, 1024, 128)] {
            let elements: usize = rows * columns;
            let quantized = (0..elements)
                .map(|index| ((index * 29 + 11) % 16) as u8)
                .collect::<Vec<_>>();
            let packed = quantized
                .chunks(2)
                .map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(8) << 4))
                .collect::<Vec<_>>();
            let scales = (0..elements.div_ceil(group_size))
                .map(|group| 0.001 + (group % 23) as f32 * 0.00017)
                .collect::<Vec<_>>();
            let input = (0..columns)
                .map(|column| ((column as f32 - 37.0) * 0.043).cos())
                .collect::<Vec<_>>();
            let actual = project_q4(&packed, &scales, &input, rows, columns, group_size)
                .expect("AVX2 Q4 projection");
            let expected = scalar_reference(&packed, &scales, &input, rows, columns, group_size);
            for (actual, expected) in actual.iter().zip(expected) {
                let tolerance = 2.0e-5 + expected.abs() * 2.0e-5;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "AVX2 {actual} != scalar {expected}"
                );
            }
        }

        let rows = 3;
        let columns = 33;
        let group_size = 13;
        let elements: usize = rows * columns;
        let quantized = (0..elements)
            .map(|index| ((index * 7 + 2) % 16) as u8)
            .collect::<Vec<_>>();
        let packed = quantized
            .chunks(2)
            .map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(8) << 4))
            .collect::<Vec<_>>();
        let scales = vec![1.0e-38; elements.div_ceil(group_size)];
        let input = vec![f32::MAX; columns];
        let actual = project_q4(&packed, &scales, &input, rows, columns, group_size)
            .expect("AVX2 f64 accumulation handles extreme finite activations");
        let expected = scalar_reference(&packed, &scales, &input, rows, columns, group_size);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= expected.abs() * 2.0e-5 + 1.0e-3);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "release-only AVX2 versus scalar grouped-Q4 projection benchmark"]
    fn avx2_q4_projection_latency_measurement() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        if !super::native_x86_avx2_available() {
            eprintln!("AVX2 unavailable; no x86 projection measurement was collected");
            return;
        }

        let rows = 1024usize;
        let columns = 4096usize;
        let group_size = 64usize;
        let elements = rows * columns;
        let packed = (0..elements.div_ceil(2))
            .map(|index| (index.wrapping_mul(37).wrapping_add(9) & 0xff) as u8)
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|index| 0.004 + (index % 19) as f32 * 0.0003)
            .collect::<Vec<_>>();
        let input = (0..columns)
            .map(|index| ((index as f32 * 0.013) - 9.0).sin())
            .collect::<Vec<_>>();
        let mut scalar_output = vec![0.0f32; rows];
        let mut avx2_output = vec![0.0f32; rows];

        fn measure_scalar(
            packed: &[u8],
            scales: &[f32],
            input: &[f32],
            columns: usize,
            group_size: usize,
            output: &mut [f32],
        ) -> Duration {
            let started = Instant::now();
            super::project_q4_scalar_range(packed, scales, input, columns, group_size, 0, output)
                .expect("scalar Q4 projection");
            black_box(output);
            started.elapsed()
        }

        fn measure_avx2(
            packed: &[u8],
            scales: &[f32],
            input: &[f32],
            columns: usize,
            group_size: usize,
            output: &mut [f32],
        ) -> Duration {
            let started = Instant::now();
            // SAFETY: the test checks runtime AVX2 support and uses bounded,
            // finite inputs with geometry matched to the allocated buffers.
            unsafe {
                super::project_q4_avx2_range(packed, scales, input, columns, group_size, 0, output)
            }
            .expect("AVX2 Q4 projection");
            black_box(output);
            started.elapsed()
        }

        for _ in 0..4 {
            measure_scalar(
                &packed,
                &scales,
                &input,
                columns,
                group_size,
                &mut scalar_output,
            );
            measure_avx2(
                &packed,
                &scales,
                &input,
                columns,
                group_size,
                &mut avx2_output,
            );
        }

        let mut scalar_samples = Vec::with_capacity(51);
        let mut avx2_samples = Vec::with_capacity(51);
        for sample in 0..51 {
            if sample % 2 == 0 {
                scalar_samples.push(measure_scalar(
                    &packed,
                    &scales,
                    &input,
                    columns,
                    group_size,
                    &mut scalar_output,
                ));
                avx2_samples.push(measure_avx2(
                    &packed,
                    &scales,
                    &input,
                    columns,
                    group_size,
                    &mut avx2_output,
                ));
            } else {
                avx2_samples.push(measure_avx2(
                    &packed,
                    &scales,
                    &input,
                    columns,
                    group_size,
                    &mut avx2_output,
                ));
                scalar_samples.push(measure_scalar(
                    &packed,
                    &scales,
                    &input,
                    columns,
                    group_size,
                    &mut scalar_output,
                ));
            }
        }
        scalar_samples.sort_unstable();
        avx2_samples.sort_unstable();
        let scalar_p50 = scalar_samples[25].as_nanos();
        let scalar_p95 = scalar_samples[48].as_nanos();
        let avx2_p50 = avx2_samples[25].as_nanos();
        let avx2_p95 = avx2_samples[48].as_nanos();
        let maximum_absolute_error = scalar_output
            .iter()
            .zip(&avx2_output)
            .map(|(scalar, avx2)| f64::from((*scalar - *avx2).abs()))
            .fold(0.0f64, f64::max);
        println!(
            "grouped-q4-avx2 rows={rows} columns={columns} group_size={group_size} samples=51 scalar_p50_ns={scalar_p50} scalar_p95_ns={scalar_p95} avx2_p50_ns={avx2_p50} avx2_p95_ns={avx2_p95} p50_speedup={:.3} max_abs_error={maximum_absolute_error}",
            scalar_p50 as f64 / avx2_p50 as f64
        );
        assert!(maximum_absolute_error <= 2.0e-5);
    }

    #[test]
    fn large_q4_projection_matches_scalar_reference_across_parallel_row_chunks() {
        let rows = super::MIN_PARALLEL_Q4_ROWS;
        let columns = super::MIN_PARALLEL_Q4_ELEMENTS.div_ceil(rows);
        let group_size = 127;
        let elements = rows * columns;
        let quantized = (0..elements)
            .map(|index| ((index * 13 + 9) % 16) as u8)
            .collect::<Vec<_>>();
        let packed = quantized
            .chunks(2)
            .map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(8) << 4))
            .collect::<Vec<_>>();
        let scales = (0usize..elements.div_ceil(group_size))
            .map(|group| 0.015 + (group % 17) as f32 * 0.001)
            .collect::<Vec<_>>();
        let input = (0..columns)
            .map(|column| ((column as f32 - 61.0) * 0.0031).sin())
            .collect::<Vec<_>>();

        let actual = project_q4(&packed, &scales, &input, rows, columns, group_size)
            .expect("large Q4 row-parallel projection");
        let fused = project_q4_argmax(&packed, &scales, &input, rows, columns, group_size)
            .expect("large Q4 row-parallel fused argmax");
        let expected = scalar_reference(&packed, &scales, &input, rows, columns, group_size);
        let expected_argmax = actual
            .iter()
            .copied()
            .enumerate()
            .fold(None, |best: Option<(usize, f32)>, candidate| {
                if best.is_none_or(|prior| candidate.1 > prior.1) {
                    Some(candidate)
                } else {
                    best
                }
            })
            .expect("projected rows");
        assert_eq!(fused, expected_argmax);
        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 3.0e-5 + expected.abs() * 3.0e-5;
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn large_selected_q4_projection_preserves_order_across_parallel_chunks() {
        let rows = 512_usize;
        let columns = 4096_usize;
        let group_size = 127_usize;
        let elements = rows * columns;
        let quantized = (0..elements)
            .map(|index| ((index * 19 + 3) % 16) as u8)
            .collect::<Vec<_>>();
        let packed = quantized
            .chunks(2)
            .map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(8) << 4))
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|group| 0.02 + (group % 13) as f32 * 0.0015)
            .collect::<Vec<_>>();
        let input = (0..columns)
            .map(|column| ((column as f32 - 29.0) * 0.0027).cos())
            .collect::<Vec<_>>();
        let selected_rows = (0..rows)
            .map(|index| (index * 197) % rows)
            .collect::<Vec<_>>();
        let expected = scalar_reference(&packed, &scales, &input, rows, columns, group_size);
        let mut actual = vec![f32::NAN; selected_rows.len()];

        project_q4_selected_into(
            &packed,
            &scales,
            &input,
            rows,
            columns,
            group_size,
            &selected_rows,
            &mut actual,
        )
        .expect("large selected Q4 row-parallel projection");
        for (observed, row) in actual.iter().zip(selected_rows) {
            let expected = expected[row];
            let tolerance = 3.0e-5 + expected.abs() * 3.0e-5;
            assert!(
                (observed - expected).abs() <= tolerance,
                "{observed} != {expected} for row {row}"
            );
        }
    }

    #[test]
    fn q4_projection_reuses_output_and_clears_it_after_an_invalid_call() {
        let packed = [0x97, 0x18, 0xef];
        let scales = [0.25, 0.5];
        let mut output = [99.0; 2];

        project_q4_into(&packed, &scales, &[1.0, -2.0, 0.5], 2, 3, 3, &mut output)
            .expect("first reused projection");
        let expected_first =
            project_q4(&packed, &scales, &[1.0, -2.0, 0.5], 2, 3, 3).expect("allocating reference");
        assert_eq!(output.as_slice(), expected_first.as_slice());

        project_q4_into(&packed, &scales, &[-0.25, 1.5, 3.0], 2, 3, 3, &mut output)
            .expect("second reused projection");
        let expected_second = project_q4(&packed, &scales, &[-0.25, 1.5, 3.0], 2, 3, 3)
            .expect("allocating reference");
        assert_eq!(output.as_slice(), expected_second.as_slice());

        assert!(
            project_q4_into(
                &packed,
                &scales,
                &[f32::INFINITY, 0.0, 0.0],
                2,
                3,
                3,
                &mut output,
            )
            .is_err()
        );
        assert_eq!(output, [0.0; 2]);
    }

    #[test]
    fn q4_batch_projection_matches_individual_rows_across_odd_groups_and_tiles() {
        let rows = 7usize;
        let columns = 19usize;
        let group_size = 11usize;
        let batch_size = 37usize;
        let elements = rows * columns;
        let quantized = (0..elements)
            .map(|index| ((index * 13 + 3) % 15) as u8)
            .collect::<Vec<_>>();
        let packed = quantized
            .chunks(2)
            .map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(8) << 4))
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|group| 0.03 + group as f32 * 0.007)
            .collect::<Vec<_>>();
        let input = (0..batch_size * columns)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.013)
            .collect::<Vec<_>>();
        let mut output = vec![99.0; batch_size * rows];
        let scratch_len = columns * batch_size.min(super::Q4_BATCH_TILE_SIZE);
        let mut scratch = vec![99.0; scratch_len];

        project_q4_batch_into(
            &packed,
            &scales,
            &input,
            rows,
            columns,
            group_size,
            batch_size,
            &mut scratch,
            &mut output,
        )
        .expect("bounded Q4 batch projection");

        for batch in 0..batch_size {
            let expected = project_q4(
                &packed,
                &scales,
                &input[batch * columns..(batch + 1) * columns],
                rows,
                columns,
                group_size,
            )
            .expect("individual reference projection");
            for row in 0..rows {
                let observed = output[batch * rows + row];
                let tolerance = 2.0e-5 + expected[row].abs() * 2.0e-5;
                assert!(
                    (observed - expected[row]).abs() <= tolerance,
                    "batch {batch}, row {row}: {observed} != {}",
                    expected[row]
                );
            }
        }
        assert!(scratch.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn q4_batch_projection_parallel_tiles_match_individual_projections() {
        let rows = 256usize;
        let columns = 128usize;
        let group_size = 97usize;
        let batch_size = 64usize;
        let elements = rows * columns;
        let packed = (0..elements.div_ceil(2))
            .map(|index| (index.wrapping_mul(31) ^ 0xa5) as u8)
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|index| 0.02 + (index % 11) as f32 * 0.003)
            .collect::<Vec<_>>();
        let input = (0..batch_size * columns)
            .map(|index| ((index.wrapping_mul(19) % 127) as f32 - 63.0) * 0.004)
            .collect::<Vec<_>>();
        let mut output = vec![0.0; batch_size * rows];
        let mut scratch = vec![0.0; columns * super::Q4_BATCH_TILE_SIZE];
        project_q4_batch_into(
            &packed,
            &scales,
            &input,
            rows,
            columns,
            group_size,
            batch_size,
            &mut scratch,
            &mut output,
        )
        .expect("parallel Q4 batch projection");

        for batch in 0..batch_size {
            let expected = project_q4(
                &packed,
                &scales,
                &input[batch * columns..(batch + 1) * columns],
                rows,
                columns,
                group_size,
            )
            .expect("individual Q4 projection");
            for (observed, reference) in output[batch * rows..(batch + 1) * rows]
                .iter()
                .zip(expected)
            {
                let tolerance = 5.0e-5 + reference.abs() * 5.0e-5;
                assert!((observed - reference).abs() <= tolerance);
            }
        }
        assert!(scratch.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn q4_batch_projection_parallel_rows_match_individual_projections() {
        let rows = 256usize;
        let columns = 512usize;
        let group_size = 127usize;
        let batch_size = 4usize;
        let elements = rows * columns;
        let packed = (0..elements.div_ceil(2))
            .map(|index| (index.wrapping_mul(31) ^ 0x6d) as u8)
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|index| 0.009 + (index % 13) as f32 * 0.0007)
            .collect::<Vec<_>>();
        let input = (0..batch_size * columns)
            .map(|index| ((index.wrapping_mul(23) % 193) as f32 - 96.0) * 0.002)
            .collect::<Vec<_>>();
        let mut output = vec![f32::NAN; batch_size * rows];
        let mut scratch = vec![f32::NAN; columns * batch_size];

        project_q4_batch_into(
            &packed,
            &scales,
            &input,
            rows,
            columns,
            group_size,
            batch_size,
            &mut scratch,
            &mut output,
        )
        .expect("row-parallel Q4 batch projection");

        for batch in 0..batch_size {
            let expected = project_q4(
                &packed,
                &scales,
                &input[batch * columns..(batch + 1) * columns],
                rows,
                columns,
                group_size,
            )
            .expect("individual Q4 projection");
            for (observed, reference) in output[batch * rows..(batch + 1) * rows]
                .iter()
                .zip(expected)
            {
                let tolerance = 5.0e-5 + reference.abs() * 5.0e-5;
                assert!((observed - reference).abs() <= tolerance);
            }
        }
        assert!(scratch.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn q4_batch_projection_clears_outputs_and_scratch_after_invalid_input() {
        let mut output = [7.0; 2];
        let mut scratch = [9.0; 3];
        assert!(
            project_q4_batch_into(
                &[0x88],
                &[1.0],
                &[f32::INFINITY, 0.0, 0.0],
                1,
                2,
                2,
                1,
                &mut scratch,
                &mut output,
            )
            .is_err()
        );
        assert_eq!(output, [0.0; 2]);
        assert_eq!(scratch, [0.0; 3]);
    }

    #[test]
    #[ignore = "release-only Q4 single-input versus batched vision projection benchmark"]
    fn q4_batch_vision_projection_latency_measurement() {
        use std::time::Instant;

        let rows = 1024usize;
        let columns = 1536usize;
        let group_size = 128usize;
        let batch_size = 256usize;
        let elements = rows * columns;
        let packed = (0..elements.div_ceil(2))
            .map(|index| (index.wrapping_mul(37) ^ 0x97) as u8)
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|index| 0.0125 + (index % 17) as f32 * 0.0001)
            .collect::<Vec<_>>();
        let input = (0..batch_size * columns)
            .map(|index| ((index.wrapping_mul(29) % 251) as f32 - 125.0) * 0.001)
            .collect::<Vec<_>>();
        let mut baseline_output = vec![0.0; batch_size * rows];
        let mut batch_output = vec![0.0; batch_size * rows];
        let mut scratch = vec![0.0; columns * super::Q4_BATCH_TILE_SIZE];
        let mut baseline_samples = Vec::new();
        let mut batch_samples = Vec::new();

        for iteration in 0..101 {
            let start = Instant::now();
            if iteration % 2 == 0 {
                for batch in 0..batch_size {
                    project_q4_into(
                        &packed,
                        &scales,
                        &input[batch * columns..(batch + 1) * columns],
                        rows,
                        columns,
                        group_size,
                        &mut baseline_output[batch * rows..(batch + 1) * rows],
                    )
                    .expect("single-input Q4 baseline");
                }
                baseline_samples.push(start.elapsed().as_nanos());
            } else {
                project_q4_batch_into(
                    &packed,
                    &scales,
                    &input,
                    rows,
                    columns,
                    group_size,
                    batch_size,
                    &mut scratch,
                    &mut batch_output,
                )
                .expect("batched Q4 projection");
                batch_samples.push(start.elapsed().as_nanos());
            }
        }

        baseline_samples.sort_unstable();
        batch_samples.sort_unstable();
        let baseline_median = baseline_samples[baseline_samples.len() / 2];
        let batch_median = batch_samples[batch_samples.len() / 2];
        let baseline_p95 = baseline_samples[(baseline_samples.len() * 95).div_ceil(100) - 1];
        let batch_p95 = batch_samples[(batch_samples.len() * 95).div_ceil(100) - 1];
        let max_difference = baseline_output
            .iter()
            .zip(&batch_output)
            .map(|(left, right)| f64::from((*left - *right).abs()))
            .fold(0.0f64, f64::max);
        println!(
            "Q4 vision projection ns baseline_p50={baseline_median} baseline_p95={baseline_p95} batch_p50={batch_median} batch_p95={batch_p95} speedup={:.3} max_abs_difference={max_difference}",
            baseline_median as f64 / batch_median as f64
        );
        assert!(max_difference < 0.01);
    }

    #[test]
    fn q4_selected_rows_match_the_full_projection_and_clear_invalid_results() {
        let rows = 7;
        let columns = 19;
        let group_size = 11;
        let elements: usize = rows * columns;
        let quantized = (0..elements)
            .map(|index| ((index * 13 + 3) % 15) as u8)
            .collect::<Vec<_>>();
        let packed = quantized
            .chunks(2)
            .map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(8) << 4))
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|group| 0.03 + group as f32 * 0.007)
            .collect::<Vec<_>>();
        let input = (0..columns)
            .map(|column| ((column as f32 - 4.0) * 0.09).cos())
            .collect::<Vec<_>>();
        let full = project_q4(&packed, &scales, &input, rows, columns, group_size)
            .expect("complete reference projection");
        let selected_rows = [6, 0, 3, 3];
        let mut selected = [f32::NAN; 4];
        project_q4_selected_into(
            &packed,
            &scales,
            &input,
            rows,
            columns,
            group_size,
            &selected_rows,
            &mut selected,
        )
        .expect("selected row projections");
        for (observed, row) in selected.iter().zip(selected_rows) {
            assert!((*observed - full[row]).abs() < 2.0e-5);
        }

        assert!(
            project_q4_selected_into(
                &packed,
                &scales,
                &input,
                rows,
                columns,
                group_size,
                &[rows],
                &mut selected[..1],
            )
            .is_err()
        );
        assert_eq!(selected[0], 0.0);
    }

    #[test]
    fn q4_fused_argmax_matches_projection_and_keeps_lowest_row_on_ties() {
        for (rows, columns, group_size) in [(5, 7, 9), (3, 257, 16), (4, 65, 31)] {
            let elements: usize = rows * columns;
            let quantized = (0..elements)
                .map(|index| ((index * 17 + 5) % 15) as u8)
                .collect::<Vec<_>>();
            let packed = quantized
                .chunks(2)
                .map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(8) << 4))
                .collect::<Vec<_>>();
            let scales = (0..elements.div_ceil(group_size))
                .map(|group| 0.001 + group as f32 * 0.0003)
                .collect::<Vec<_>>();
            let input = (0..columns)
                .map(|column| ((column as f32 - 23.0) * 0.071).sin())
                .collect::<Vec<_>>();
            let projected = project_q4(&packed, &scales, &input, rows, columns, group_size)
                .expect("dense Q4 reference");
            let expected = projected
                .iter()
                .copied()
                .enumerate()
                .fold(None, |best: Option<(usize, f32)>, candidate| {
                    if best.is_none_or(|prior| candidate.1 > prior.1) {
                        Some(candidate)
                    } else {
                        best
                    }
                })
                .expect("projected rows");
            assert_eq!(
                project_q4_argmax(&packed, &scales, &input, rows, columns, group_size)
                    .expect("fused Q4 argmax"),
                expected
            );
        }

        let tied =
            project_q4_argmax(&[0x88, 0x88], &[0.25; 2], &[1.0], 3, 1, 2).expect("tied Q4 rows");
        assert_eq!(tied, (0, 0.0));
        assert!(project_q4_argmax(&[0x88], &[1.0], &[f32::NAN], 1, 1, 1).is_err());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4_neon_two_row_tiles_match_the_independent_row_kernel() {
        let rows = 9usize;
        let columns = 256usize;
        let group_size = 128usize;
        let elements = rows * columns;
        let packed = (0..elements.div_ceil(2))
            .map(|index| (index.wrapping_mul(37) ^ 0x93) as u8)
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|index| 0.001 + (index % 19) as f32 * 0.0007)
            .collect::<Vec<_>>();
        let input = (0..columns)
            .map(|index| ((index.wrapping_mul(23) % 251) as f32 - 125.0) * 0.001)
            .collect::<Vec<_>>();

        for (first_row, row_count) in [(0, 9), (1, 7), (2, 5), (8, 1)] {
            let mut expected = vec![f32::NAN; row_count];
            let mut actual = vec![f32::NAN; row_count];
            // SAFETY: this fixture has validated matrix geometry, finite
            // activations, and in-range output slices for both kernels.
            unsafe {
                super::project_q4_neon_range_reference(
                    &packed,
                    &scales,
                    &input,
                    columns,
                    group_size,
                    first_row,
                    &mut expected,
                )
                .expect("reference NEON rows");
                super::project_q4_neon_range(
                    &packed,
                    &scales,
                    &input,
                    columns,
                    group_size,
                    first_row,
                    &mut actual,
                )
                .expect("paired NEON rows");
            }
            assert_eq!(
                actual,
                expected,
                "rows {first_row}..{}",
                first_row + row_count
            );
        }

        let rows = 7usize;
        let columns = 19usize;
        let group_size = 11usize;
        let elements = rows * columns;
        let packed = (0..elements.div_ceil(2))
            .map(|index| (index.wrapping_mul(29) ^ 0x5b) as u8)
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|index| 0.02 + index as f32 * 0.001)
            .collect::<Vec<_>>();
        let input = (0..columns)
            .map(|index| ((index as f32 - 7.0) * 0.031).sin())
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; rows];
        let mut fallback = vec![0.0; rows];
        // SAFETY: both calls use the same validated odd-row reference fixture;
        // the optimized dispatcher intentionally falls back for this shape.
        unsafe {
            super::project_q4_neon_range_reference(
                &packed,
                &scales,
                &input,
                columns,
                group_size,
                0,
                &mut expected,
            )
            .expect("odd-shape reference NEON rows");
            super::project_q4_neon_range(
                &packed,
                &scales,
                &input,
                columns,
                group_size,
                0,
                &mut fallback,
            )
            .expect("odd-shape row fallback");
        }
        assert_eq!(fallback, expected);
    }

    #[test]
    fn persistent_q4_projection_pool_matches_scoped_workers_under_concurrent_calls() {
        let rows = 512usize;
        let columns = 4096usize;
        let group_size = 128usize;
        let elements = rows * columns;
        let packed = Arc::new(
            (0..elements.div_ceil(2))
                .map(|index| {
                    let low = ((index * 2 * 13 + 5) % 15) as u8;
                    let high = ((index * 2 * 13 + 18) % 15) as u8;
                    low | (high << 4)
                })
                .collect::<Vec<_>>(),
        );
        let scales = Arc::new(
            (0..elements.div_ceil(group_size))
                .map(|index| 0.005 + (index % 31) as f32 * 0.0001)
                .collect::<Vec<_>>(),
        );
        let input = (0..columns)
            .map(|index| ((index * 17 % 251) as f32 - 125.0) * 0.001)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; rows];
        project_q4_into(
            &packed,
            &scales,
            &input,
            rows,
            columns,
            group_size,
            &mut expected,
        )
        .expect("scoped reference projection");

        std::thread::scope(|scope| {
            let mut calls = Vec::new();
            for _ in 0..4 {
                let packed = Arc::clone(&packed);
                let scales = Arc::clone(&scales);
                let input = input.clone();
                calls.push(scope.spawn(move || {
                    let mut actual = vec![f32::NAN; rows];
                    project_q4_into_pooled(
                        packed,
                        scales,
                        &input,
                        rows,
                        columns,
                        group_size,
                        &mut actual,
                    )
                    .expect("persistent pooled projection");
                    actual
                }));
            }
            for call in calls {
                assert_eq!(
                    call.join().expect("projection caller remains healthy"),
                    expected
                );
            }
        });

        let mut rejected = vec![1.0; rows];
        assert!(
            project_q4_into_pooled(
                Arc::clone(&packed),
                Arc::clone(&scales),
                &[f32::NAN; 4096],
                rows,
                columns,
                group_size,
                &mut rejected,
            )
            .is_err()
        );
        assert!(rejected.iter().all(|value| *value == 0.0));
    }

    #[test]
    #[ignore = "release-only representative Qwen MLP scoped versus persistent Q4 projection measurement"]
    fn qwen_mlp_q4_projection_worker_pool_latency_measurement() {
        use std::time::Instant;

        let rows = 11_008usize;
        let columns = 2_560usize;
        let group_size = 128usize;
        let elements = rows * columns;
        let packed = Arc::new(
            (0..elements.div_ceil(2))
                .map(|index| {
                    let low = ((index * 2 * 13 + 5) % 15) as u8;
                    let high = ((index * 2 * 13 + 18) % 15) as u8;
                    low | (high << 4)
                })
                .collect::<Vec<_>>(),
        );
        let scales = Arc::new(
            (0..elements.div_ceil(group_size))
                .map(|index| 0.005 + (index % 31) as f32 * 0.0001)
                .collect::<Vec<_>>(),
        );
        let input = (0..columns)
            .map(|index| ((index * 17 % 251) as f32 - 125.0) * 0.001)
            .collect::<Vec<_>>();
        let mut scoped_output = vec![0.0; rows];
        let mut pooled_output = vec![0.0; rows];
        project_q4_into(
            &packed,
            &scales,
            &input,
            rows,
            columns,
            group_size,
            &mut scoped_output,
        )
        .expect("scoped Qwen MLP warmup");
        let startup_started = Instant::now();
        project_q4_into_pooled(
            Arc::clone(&packed),
            Arc::clone(&scales),
            &input,
            rows,
            columns,
            group_size,
            &mut pooled_output,
        )
        .expect("pooled Qwen MLP warmup");
        let pool_warmup_ns = startup_started.elapsed().as_nanos();
        assert_eq!(scoped_output, pooled_output);

        let mut scoped_samples = Vec::with_capacity(60);
        let mut pooled_samples = Vec::with_capacity(60);
        for iteration in 0..120 {
            let started = Instant::now();
            if iteration % 2 == 0 {
                project_q4_into(
                    &packed,
                    &scales,
                    &input,
                    rows,
                    columns,
                    group_size,
                    &mut scoped_output,
                )
                .expect("scoped Qwen MLP projection");
                scoped_samples.push(started.elapsed().as_nanos());
            } else {
                project_q4_into_pooled(
                    Arc::clone(&packed),
                    Arc::clone(&scales),
                    &input,
                    rows,
                    columns,
                    group_size,
                    &mut pooled_output,
                )
                .expect("pooled Qwen MLP projection");
                pooled_samples.push(started.elapsed().as_nanos());
            }
        }
        assert_eq!(scoped_output, pooled_output);
        scoped_samples.sort_unstable();
        pooled_samples.sort_unstable();
        let scoped_p50 = scoped_samples[scoped_samples.len() / 2];
        let pooled_p50 = pooled_samples[pooled_samples.len() / 2];
        let scoped_p95 = scoped_samples[(scoped_samples.len() * 95).div_ceil(100) - 1];
        let pooled_p95 = pooled_samples[(pooled_samples.len() * 95).div_ceil(100) - 1];
        println!(
            "Qwen MLP Q4 projection rows={rows} columns={columns} samples={} scoped_p50_us={} scoped_p95_us={} pooled_p50_us={} pooled_p95_us={} pool_warmup_ms={:.3} pooled_speedup_p50={:.3}",
            scoped_samples.len(),
            scoped_p50 / 1_000,
            scoped_p95 / 1_000,
            pooled_p50 / 1_000,
            pooled_p95 / 1_000,
            pool_warmup_ns as f64 / 1_000_000.0,
            scoped_p50 as f64 / pooled_p50 as f64,
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "release-only two-row Q4 activation reuse measurement"]
    fn qwen_mlp_q4_two_row_activation_reuse_latency_measurement() {
        use std::time::Instant;

        let rows = 11_008usize;
        let columns = 2_560usize;
        let group_size = 128usize;
        let elements = rows * columns;
        let packed = (0..elements.div_ceil(2))
            .map(|index| (index.wrapping_mul(37) ^ 0x93) as u8)
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|index| 0.001 + (index % 19) as f32 * 0.0007)
            .collect::<Vec<_>>();
        let input = (0..columns)
            .map(|index| ((index.wrapping_mul(23) % 251) as f32 - 125.0) * 0.001)
            .collect::<Vec<_>>();
        let workers = super::inference_cpu_worker_limit().min(rows);
        let mut row_output = vec![0.0; rows];
        let mut paired_output = vec![0.0; rows];

        let project = |paired: bool, output: &mut [f32]| {
            run_q4_workers(output, workers, |first_row, output_chunk| {
                // SAFETY: the release fixture validates every kernel bound;
                // each worker owns a distinct output-row segment.
                unsafe {
                    if paired {
                        super::project_q4_neon_range(
                            &packed,
                            &scales,
                            &input,
                            columns,
                            group_size,
                            first_row,
                            output_chunk,
                        )
                    } else {
                        super::project_q4_neon_range_reference(
                            &packed,
                            &scales,
                            &input,
                            columns,
                            group_size,
                            first_row,
                            output_chunk,
                        )
                    }
                }
            })
            .expect("bounded workers start")
            .expect("valid Qwen MLP projection");
        };

        project(false, &mut row_output);
        project(true, &mut paired_output);
        assert_eq!(paired_output, row_output);

        let mut row_samples = Vec::with_capacity(61);
        let mut paired_samples = Vec::with_capacity(61);
        for iteration in 0..122 {
            let started = Instant::now();
            if iteration % 2 == 0 {
                project(false, &mut row_output);
                row_samples.push(started.elapsed().as_nanos());
                assert_eq!(paired_output, row_output);
            } else {
                project(true, &mut paired_output);
                paired_samples.push(started.elapsed().as_nanos());
                assert_eq!(paired_output, row_output);
            }
        }
        row_samples.sort_unstable();
        paired_samples.sort_unstable();
        let percentile = |samples: &[u128], numerator: usize| {
            samples[(samples.len() * numerator).div_ceil(100) - 1]
        };
        let row_p50 = row_samples[row_samples.len() / 2];
        let paired_p50 = paired_samples[paired_samples.len() / 2];
        println!(
            "Qwen MLP Q4 paired activation reuse rows={rows} columns={columns} workers={workers} samples={} row_p50_us={} row_p95_us={} paired_p50_us={} paired_p95_us={} p50_speedup={:.3}",
            row_samples.len(),
            row_p50 / 1_000,
            percentile(&row_samples, 95) / 1_000,
            paired_p50 / 1_000,
            percentile(&paired_samples, 95) / 1_000,
            row_p50 as f64 / paired_p50 as f64,
        );
    }

    #[test]
    #[ignore = "release-only full-vocabulary Q4 dense versus fused output-head benchmark"]
    fn qwen_output_head_fused_argmax_latency_measurement() {
        use std::time::Instant;

        let rows = 151_936usize;
        let columns = 2_560usize;
        let group_size = 128usize;
        let elements = rows * columns;
        let packed = (0..elements.div_ceil(2))
            .map(|index| (index.wrapping_mul(37) ^ 0x97) as u8)
            .collect::<Vec<_>>();
        let scales = (0..elements.div_ceil(group_size))
            .map(|index| 0.0125 + (index % 17) as f32 * 0.0001)
            .collect::<Vec<_>>();
        let input = (0..columns)
            .map(|index| ((index.wrapping_mul(29) % 251) as f32 - 125.0) * 0.001)
            .collect::<Vec<_>>();
        let mut logits = vec![0.0; rows];
        project_q4_into(
            &packed,
            &scales,
            &input,
            rows,
            columns,
            group_size,
            &mut logits,
        )
        .expect("dense output-head warmup");
        let dense_winner = logits
            .iter()
            .copied()
            .enumerate()
            .fold(None, |best: Option<(usize, f32)>, candidate| {
                if best.is_none_or(|prior| candidate.1 > prior.1) {
                    Some(candidate)
                } else {
                    best
                }
            })
            .expect("dense output rows");
        let fused_winner = project_q4_argmax(&packed, &scales, &input, rows, columns, group_size)
            .expect("fused output-head warmup");
        assert_eq!(dense_winner, fused_winner);

        let mut dense_samples = Vec::with_capacity(51);
        let mut fused_samples = Vec::with_capacity(51);
        for iteration in 0..102 {
            let started = Instant::now();
            if iteration % 2 == 0 {
                project_q4_into(
                    &packed,
                    &scales,
                    &input,
                    rows,
                    columns,
                    group_size,
                    &mut logits,
                )
                .expect("dense output-head projection");
                let winner = logits
                    .iter()
                    .copied()
                    .enumerate()
                    .fold(None, |best: Option<(usize, f32)>, candidate| {
                        if best.is_none_or(|prior| candidate.1 > prior.1) {
                            Some(candidate)
                        } else {
                            best
                        }
                    })
                    .expect("dense output rows");
                dense_samples.push(started.elapsed().as_nanos());
                assert_eq!(winner, fused_winner);
            } else {
                let winner = project_q4_argmax(&packed, &scales, &input, rows, columns, group_size)
                    .expect("fused output-head argmax");
                fused_samples.push(started.elapsed().as_nanos());
                assert_eq!(winner, dense_winner);
            }
        }
        dense_samples.sort_unstable();
        fused_samples.sort_unstable();
        let percentile = |samples: &[u128], numerator: usize| {
            samples[(samples.len() * numerator).div_ceil(100) - 1]
        };
        let dense_p50 = dense_samples[dense_samples.len() / 2];
        let fused_p50 = fused_samples[fused_samples.len() / 2];
        println!(
            "Qwen output head rows={rows} columns={columns} samples={} dense_p50_us={} dense_p95_us={} fused_p50_us={} fused_p95_us={} p50_speedup={:.3}",
            dense_samples.len(),
            dense_p50 / 1_000,
            percentile(&dense_samples, 95) / 1_000,
            fused_p50 / 1_000,
            percentile(&fused_samples, 95) / 1_000,
            dense_p50 as f64 / fused_p50 as f64,
        );
    }

    #[test]
    fn q4_projection_uses_wide_accumulation_for_extreme_finite_activations() {
        // AArch64's fast kernel accumulates unscaled weights in f32. This
        // fixture would overflow that intermediate even though the scaled
        // result is finite, so it must use the wide scalar fallback.
        let output = project_q4(&[0xff], &[1e-30], &[1e38, 1e38], 1, 2, 2)
            .expect("finite scaled projection");
        assert!(output[0].is_finite());
        assert!((output[0] - 1.4e9).abs() < 256.0);
    }

    #[test]
    fn q4_projection_rejects_malformed_or_nonfinite_inputs() {
        assert!(project_q4(&[], &[], &[], 1, 1, 1).is_err());
        assert!(project_q4(&[0x88], &[f32::NAN], &[1.0], 1, 2, 2).is_err());
        assert!(project_q4(&[0x88], &[1.0], &[f32::INFINITY, 1.0], 1, 2, 2).is_err());
    }

    #[test]
    fn attention_kernels_match_f64_for_row_tails_and_head_offsets() {
        let rows = 7;
        let stride = 17;
        let offset = 3;
        let columns = 7;
        let keys = (0..rows * stride)
            .map(|index| ((index as f32 - 37.0) * 0.031).cos())
            .collect::<Vec<_>>();
        let values = (0..rows * stride)
            .map(|index| ((index as f32 - 11.0) * 0.047).sin())
            .collect::<Vec<_>>();
        let query = (0..columns)
            .map(|index| ((index as f32 - 2.0) * 0.13).sin())
            .collect::<Vec<_>>();
        let mut scores = vec![0.0f64; rows];
        dot_rows(&query, &keys, stride, offset, &mut scores).expect("attention QK kernel");
        let mut wide_scores = vec![0.0f64; rows];
        dot_rows_f64_into(&query, &keys, stride, offset, &mut wide_scores)
            .expect("wide attention QK kernel");
        for (position, actual) in scores.iter().enumerate() {
            let start = position * stride + offset;
            let expected = query
                .iter()
                .zip(&keys[start..start + columns])
                .fold(0.0f64, |sum, (left, right)| {
                    sum + f64::from(*left) * f64::from(*right)
                });
            assert!((actual - expected).abs() <= 2.0e-5 + expected.abs() * 2.0e-5);
            assert!((wide_scores[position] - expected).abs() <= 1.0e-12);
        }

        let weights = (0..rows)
            .map(|position| (position + 1) as f64 / 28.0)
            .collect::<Vec<_>>();
        let actual = weighted_sum_rows(&values, &weights, rows, stride, offset, columns)
            .expect("attention WV kernel");
        let mut reused_output = vec![f32::NAN; columns];
        weighted_sum_rows_into(&values, &weights, rows, stride, offset, &mut reused_output)
            .expect("caller-owned attention output");
        assert_eq!(actual, reused_output);
        let mut wide_output = vec![f64::NAN; columns];
        weighted_sum_rows_f64_into(&values, &weights, rows, stride, offset, &mut wide_output)
            .expect("wide caller-owned attention output");
        for (column, actual) in actual.iter().enumerate() {
            let expected = (0..rows).fold(0.0f64, |sum, position| {
                sum + weights[position] * f64::from(values[position * stride + offset + column])
            });
            let tolerance = 2.0e-5 + expected.abs() * 2.0e-5;
            assert!(
                (f64::from(*actual) - expected).abs() <= tolerance,
                "column {column}: actual={actual} expected={expected} tolerance={tolerance}"
            );
            assert!((wide_output[column] - expected).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn wide_attention_kernels_clear_outputs_after_invalid_results() {
        let mut scores = [9.0f64; 2];
        assert!(dot_rows_f64_into(&[f32::NAN], &[1.0, 2.0], 1, 0, &mut scores).is_err());
        assert_eq!(scores, [0.0; 2]);

        let values = [f32::INFINITY, 1.0];
        let mut output = [9.0f64];
        assert!(weighted_sum_rows_f64_into(&values, &[1.0, 0.0], 2, 1, 0, &mut output).is_err());
        assert_eq!(output, [0.0]);
    }

    #[test]
    fn attention_weighted_sum_routes_wide_heads_through_bounded_fallback() {
        let rows = 3;
        let columns = MAX_ATTENTION_VALUE_DIMENSION + 1;
        let stride = columns + 2;
        let offset = 1;
        let values = (0..rows * stride)
            .map(|index| ((index as f32 - 11.0) * 0.013).cos())
            .collect::<Vec<_>>();
        let weights = [0.2, 0.3, 0.5];
        let mut output = vec![0.0; columns];
        weighted_sum_rows_into(&values, &weights, rows, stride, offset, &mut output)
            .expect("wide scalar fallback");
        for (column, actual) in output.iter().enumerate() {
            let expected = (0..rows).fold(0.0f64, |sum, row| {
                sum + weights[row] * f64::from(values[row * stride + offset + column])
            }) as f32;
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn gated_delta_kernel_matches_f64_state_across_tokens_and_tails() {
        let key_dimension = 7;
        let value_dimension = 9;
        let mut scalar_state = (0..key_dimension * value_dimension)
            .map(|index| ((index as f32 - 23.0) * 0.017).sin() * 0.2)
            .collect::<Vec<_>>();
        let mut fast_state = scalar_state.clone();
        for token in 0..5 {
            let query = (0..key_dimension)
                .map(|index| ((index as f32 + token as f32 * 0.37 - 4.0) * 0.21).sin())
                .collect::<Vec<_>>();
            let key = (0..key_dimension)
                .map(|index| ((index as f32 - token as f32 * 0.19 + 1.0) * 0.17).cos())
                .collect::<Vec<_>>();
            let value = (0..value_dimension)
                .map(|index| ((index as f32 + token as f32 * 0.11 - 3.0) * 0.13).sin())
                .collect::<Vec<_>>();
            let key_norm = (key
                .iter()
                .fold(0.0f64, |sum, item| sum + f64::from(*item).powi(2))
                + 1e-6)
                .sqrt();
            let query_norm = (query
                .iter()
                .fold(0.0f64, |sum, item| sum + f64::from(*item).powi(2))
                + 1e-6)
                .sqrt();
            let key_direction = key
                .iter()
                .map(|item| (f64::from(*item) / key_norm) as f32)
                .collect::<Vec<_>>();
            let query_direction = query
                .iter()
                .map(|item| (f64::from(*item) / query_norm / (key_dimension as f64).sqrt()) as f32)
                .collect::<Vec<_>>();
            let decay = 0.82 + token as f32 * 0.025;
            let beta = 0.21 + token as f32 * 0.03;
            let mut expected_output = vec![0.0f32; value_dimension];
            for column in 0..value_dimension {
                let mut prior = 0.0f64;
                for row in 0..key_dimension {
                    prior += f64::from(scalar_state[row * value_dimension + column])
                        * f64::from(decay)
                        * f64::from(key_direction[row]);
                }
                let correction = f64::from(beta) * (f64::from(value[column]) - prior);
                let mut read = 0.0f64;
                for row in 0..key_dimension {
                    let index = row * value_dimension + column;
                    scalar_state[index] = (f64::from(decay) * f64::from(scalar_state[index])
                        + correction * f64::from(key_direction[row]))
                        as f32;
                    read += f64::from(scalar_state[index]) * f64::from(query_direction[row]);
                }
                expected_output[column] = read as f32;
            }
            let mut actual_output = vec![0.0f32; value_dimension];
            gated_delta_step(
                &mut fast_state,
                &key_direction,
                &query_direction,
                &value,
                decay,
                beta,
                &mut actual_output,
            )
            .expect("bounded gated-delta update");
            for (actual, expected) in actual_output.iter().zip(expected_output) {
                let tolerance = 3.0e-5 + expected.abs() * 3.0e-5;
                assert!((actual - expected).abs() <= tolerance);
            }
            for (actual, expected) in fast_state.iter().zip(&scalar_state) {
                let tolerance = 3.0e-5 + expected.abs() * 3.0e-5;
                assert!((actual - expected).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn gated_delta_kernel_rejects_invalid_shapes_before_mutating_state() {
        let mut state = vec![0.25; 12];
        let original = state.clone();
        let mut output = [0.0; 4];
        assert!(
            gated_delta_step(
                &mut state,
                &[1.0, 2.0],
                &[1.0],
                &[1.0; 4],
                0.9,
                0.2,
                &mut output
            )
            .is_err()
        );
        assert!(
            gated_delta_step(
                &mut state,
                &[1.0; 4],
                &[1.0; 4],
                &[1.0; 4],
                0.9,
                0.2,
                &mut output
            )
            .is_err()
        );
        assert!(
            gated_delta_step(
                &mut state,
                &[1.0, 2.0, 3.0],
                &[1.0; 3],
                &[1.0; 4],
                1.1,
                0.2,
                &mut output
            )
            .is_err()
        );
        assert_eq!(state, original);

        let mut oversized_state = vec![0.25; 513];
        let oversized_original = oversized_state.clone();
        let oversized_direction = vec![1.0; 513];
        let mut scalar_output = [0.0];
        assert!(
            gated_delta_step(
                &mut oversized_state,
                &oversized_direction,
                &oversized_direction,
                &[1.0],
                0.9,
                0.2,
                &mut scalar_output
            )
            .is_err()
        );
        assert_eq!(oversized_state, oversized_original);
    }

    #[test]
    #[ignore = "release-only gated-delta layer latency measurement"]
    fn gated_delta_layer_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        const HEADS: usize = 32;
        const KEY_DIMENSION: usize = 128;
        const VALUE_DIMENSION: usize = 128;
        const HEAD_STATE: usize = KEY_DIMENSION * VALUE_DIMENSION;

        let initial_state = (0..HEADS * HEAD_STATE)
            .map(|index| ((index % 997) as f32 - 498.0) * 0.0002)
            .collect::<Vec<_>>();
        let mut keys = (0..HEADS * KEY_DIMENSION)
            .map(|index| ((index % 127) as f32 - 63.0) * 0.002)
            .collect::<Vec<_>>();
        let mut queries = (0..HEADS * KEY_DIMENSION)
            .map(|index| ((index % 151) as f32 - 75.0) * 0.0015)
            .collect::<Vec<_>>();
        let values = (0..HEADS * VALUE_DIMENSION)
            .map(|index| ((index % 89) as f32 - 44.0) * 0.003)
            .collect::<Vec<_>>();
        let query_scale = (KEY_DIMENSION as f64).sqrt().recip();
        for head in 0..HEADS {
            let start = head * KEY_DIMENSION;
            let key_norm = (keys[start..start + KEY_DIMENSION]
                .iter()
                .fold(0.0f64, |sum, item| sum + f64::from(*item).powi(2))
                + 1e-6)
                .sqrt();
            let query_norm = (queries[start..start + KEY_DIMENSION]
                .iter()
                .fold(0.0f64, |sum, item| sum + f64::from(*item).powi(2))
                + 1e-6)
                .sqrt();
            for index in start..start + KEY_DIMENSION {
                keys[index] = (f64::from(keys[index]) / key_norm) as f32;
                queries[index] = (f64::from(queries[index]) / query_norm * query_scale) as f32;
            }
        }
        let mut scalar_state = initial_state.clone();
        let mut fast_state = initial_state;
        let mut scalar_output = vec![0.0f32; HEADS * VALUE_DIMENSION];
        let mut fast_output = vec![0.0f32; HEADS * VALUE_DIMENSION];

        let scalar_layer = |state: &mut [f32], output: &mut [f32]| {
            for head in 0..HEADS {
                scalar_gated_delta_step(
                    &mut state[head * HEAD_STATE..(head + 1) * HEAD_STATE],
                    &keys[head * KEY_DIMENSION..(head + 1) * KEY_DIMENSION],
                    &queries[head * KEY_DIMENSION..(head + 1) * KEY_DIMENSION],
                    &values[head * VALUE_DIMENSION..(head + 1) * VALUE_DIMENSION],
                    0.985,
                    0.27,
                    &mut output[head * VALUE_DIMENSION..(head + 1) * VALUE_DIMENSION],
                );
            }
        };
        let fast_layer = |state: &mut [f32], output: &mut [f32]| {
            for head in 0..HEADS {
                gated_delta_step(
                    &mut state[head * HEAD_STATE..(head + 1) * HEAD_STATE],
                    &keys[head * KEY_DIMENSION..(head + 1) * KEY_DIMENSION],
                    &queries[head * KEY_DIMENSION..(head + 1) * KEY_DIMENSION],
                    &values[head * VALUE_DIMENSION..(head + 1) * VALUE_DIMENSION],
                    0.985,
                    0.27,
                    &mut output[head * VALUE_DIMENSION..(head + 1) * VALUE_DIMENSION],
                )
                .expect("NEON gated-delta head");
            }
        };

        scalar_layer(&mut scalar_state, &mut scalar_output);
        fast_layer(&mut fast_state, &mut fast_output);
        for (actual, expected) in fast_state.iter().zip(&scalar_state) {
            let tolerance = 2.0e-4 + expected.abs() * 2.0e-4;
            assert!((actual - expected).abs() <= tolerance);
        }
        for (actual, expected) in fast_output.iter().zip(&scalar_output) {
            let tolerance = 2.0e-4 + expected.abs() * 2.0e-4;
            assert!((actual - expected).abs() <= tolerance);
        }

        for _ in 0..2 {
            scalar_layer(&mut scalar_state, &mut scalar_output);
            black_box((&scalar_state, &scalar_output));
            fast_layer(&mut fast_state, &mut fast_output);
            black_box((&fast_state, &fast_output));
        }
        let mut scalar_times = Vec::with_capacity(21);
        let mut fast_times = Vec::with_capacity(21);
        for _ in 0..21 {
            let start = Instant::now();
            scalar_layer(&mut scalar_state, &mut scalar_output);
            black_box((&scalar_state, &scalar_output));
            scalar_times.push(start.elapsed());
            let start = Instant::now();
            fast_layer(&mut fast_state, &mut fast_output);
            black_box((&fast_state, &fast_output));
            fast_times.push(start.elapsed());
        }
        for (actual, expected) in fast_state.iter().zip(&scalar_state) {
            let tolerance = 1.0e-3 + expected.abs() * 1.0e-3;
            assert!((actual - expected).abs() <= tolerance);
        }
        scalar_times.sort_unstable();
        fast_times.sort_unstable();
        eprintln!(
            "gated-delta heads=32 key_dim=128 value_dim=128 samples=21 scalar_p50_us={} scalar_p95_us={} simd_p50_us={} simd_p95_us={}",
            duration_micros(scalar_times[10]),
            duration_micros(scalar_times[19]),
            duration_micros(fast_times[10]),
            duration_micros(fast_times[19]),
        );
    }

    fn scalar_gated_delta_step(
        state: &mut [f32],
        key_direction: &[f32],
        query_direction: &[f32],
        value: &[f32],
        decay: f32,
        beta: f32,
        output: &mut [f32],
    ) {
        for column in 0..value.len() {
            let prior = (0..key_direction.len()).fold(0.0f64, |sum, row| {
                sum + f64::from(state[row * value.len() + column])
                    * f64::from(decay)
                    * f64::from(key_direction[row])
            });
            let correction = f64::from(beta) * (f64::from(value[column]) - prior);
            let mut read = 0.0f64;
            for row in 0..key_direction.len() {
                let index = row * value.len() + column;
                state[index] = (f64::from(decay) * f64::from(state[index])
                    + correction * f64::from(key_direction[row]))
                    as f32;
                read += f64::from(state[index]) * f64::from(query_direction[row]);
            }
            output[column] = read as f32;
        }
    }

    #[test]
    #[ignore = "release-only attention value kernel latency measurement"]
    fn attention_value_kernel_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        let rows = 4_096;
        let row_stride = 512;
        let column_offset = 192;
        let columns = 128;
        let values = (0..rows * row_stride)
            .map(|index| ((index % 4093) as f32 - 2046.0) / 2048.0)
            .collect::<Vec<_>>();
        let weights = vec![1.0 / rows as f64; rows];
        let scalar = || {
            let mut output = vec![0.0f64; columns];
            for (position, weight) in weights.iter().enumerate() {
                let start = position * row_stride + column_offset;
                for (column, total) in output.iter_mut().enumerate() {
                    *total += *weight * f64::from(values[start + column]);
                }
            }
            output
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>()
        };
        let fast = || {
            weighted_sum_rows(&values, &weights, rows, row_stride, column_offset, columns)
                .expect("NEON attention weighted sum")
        };
        let mut reused_output = vec![0.0f32; columns];
        let expected = scalar();
        let actual = fast();
        weighted_sum_rows_into(
            &values,
            &weights,
            rows,
            row_stride,
            column_offset,
            &mut reused_output,
        )
        .expect("NEON attention weighted sum into caller storage");
        for (actual, expected) in actual.iter().zip(&expected) {
            let tolerance = 1.0e-5 + expected.abs() * 1.0e-5;
            assert!((actual - expected).abs() <= tolerance);
        }
        for (actual, expected) in reused_output.iter().zip(&expected) {
            let tolerance = 1.0e-5 + expected.abs() * 1.0e-5;
            assert!((actual - expected).abs() <= tolerance);
        }
        for _ in 0..2 {
            black_box(scalar());
            black_box(fast());
            weighted_sum_rows_into(
                &values,
                &weights,
                rows,
                row_stride,
                column_offset,
                &mut reused_output,
            )
            .expect("NEON attention weighted sum into caller storage");
            black_box(&reused_output);
        }
        let mut scalar_times = Vec::with_capacity(21);
        let mut fast_times = Vec::with_capacity(21);
        let mut reused_times = Vec::with_capacity(21);
        for _ in 0..21 {
            let start = Instant::now();
            black_box(scalar());
            scalar_times.push(start.elapsed());
            let start = Instant::now();
            black_box(fast());
            fast_times.push(start.elapsed());
            let start = Instant::now();
            weighted_sum_rows_into(
                &values,
                &weights,
                rows,
                row_stride,
                column_offset,
                &mut reused_output,
            )
            .expect("NEON attention weighted sum into caller storage");
            black_box(&reused_output);
            reused_times.push(start.elapsed());
        }
        scalar_times.sort_unstable();
        fast_times.sort_unstable();
        reused_times.sort_unstable();
        eprintln!(
            "attention-wv rows=4096 stride=512 columns=128 samples=21 scalar_p50_us={} scalar_p95_us={} simd_allocating_p50_us={} simd_allocating_p95_us={} simd_reused_p50_us={} simd_reused_p95_us={}",
            duration_micros(scalar_times[10]),
            duration_micros(scalar_times[19]),
            duration_micros(fast_times[10]),
            duration_micros(fast_times[19]),
            duration_micros(reused_times[10]),
            duration_micros(reused_times[19]),
        );
    }

    #[test]
    #[ignore = "release-only attention query/key kernel latency measurement"]
    fn attention_query_key_kernel_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        let rows = 4_096;
        let row_stride = 512;
        let column_offset = 128;
        let columns = 256;
        let keys = (0..rows * row_stride)
            .map(|index| ((index % 2017) as f32 - 1008.0) / 1024.0)
            .collect::<Vec<_>>();
        let query = (0..columns)
            .map(|index| ((index % 503) as f32 - 251.0) / 256.0)
            .collect::<Vec<_>>();
        let scalar = || {
            (0..rows)
                .map(|position| {
                    let start = position * row_stride + column_offset;
                    query
                        .iter()
                        .zip(&keys[start..start + columns])
                        .fold(0.0f64, |sum, (query, key)| {
                            sum + f64::from(*query) * f64::from(*key)
                        })
                })
                .collect::<Vec<_>>()
        };
        let fast = || {
            let mut output = vec![0.0f64; rows];
            dot_rows(&query, &keys, row_stride, column_offset, &mut output)
                .expect("NEON attention query/key projection");
            output
        };
        let expected = scalar();
        let actual = fast();
        for (actual, expected) in actual.iter().zip(&expected) {
            let tolerance = 2.0e-5 + expected.abs() * 2.0e-5;
            assert!((actual - expected).abs() <= tolerance);
        }
        for _ in 0..2 {
            black_box(scalar());
            black_box(fast());
        }
        let mut scalar_times = Vec::with_capacity(21);
        let mut fast_times = Vec::with_capacity(21);
        for _ in 0..21 {
            let start = Instant::now();
            black_box(scalar());
            scalar_times.push(start.elapsed());
            let start = Instant::now();
            black_box(fast());
            fast_times.push(start.elapsed());
        }
        scalar_times.sort_unstable();
        fast_times.sort_unstable();
        eprintln!(
            "attention-qk rows=4096 stride=512 columns=256 samples=21 scalar_p50_us={} scalar_p95_us={} simd_p50_us={} simd_p95_us={}",
            duration_micros(scalar_times[10]),
            duration_micros(scalar_times[19]),
            duration_micros(fast_times[10]),
            duration_micros(fast_times[19]),
        );
    }

    fn duration_micros(duration: std::time::Duration) -> u128 {
        duration.as_nanos() / 1_000
    }
}
