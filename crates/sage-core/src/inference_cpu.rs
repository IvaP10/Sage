//! First-party scalar CPU reference math and bounded inference state.
//!
//! Scalar operations favor explicit bounds and repeatable arithmetic. Sage's
//! architecture-specific fast paths live in `sage-kernels` and are compared
//! against these references without a model-runtime framework dependency.

use crate::{CoreError, CoreResult};
use sage_metal::{MetalBuffer, MetalContext, MetalQ4Workspace};
use std::sync::{Arc, Mutex};
use zeroize::{Zeroize, Zeroizing};

const MAX_RANK: usize = 8;
const MAX_REFERENCE_ELEMENTS: usize = 250_000_000;
const MAX_KV_CACHE_ELEMENTS: usize = 32_000_000;
const MAX_ATTENTION_MULTIPLIES: usize = 256_000_000;
pub(crate) const ATTENTION_SCORE_BLOCK_SIZE: usize = 2_048;
const MAX_DELTA_STATE_ELEMENTS: usize = 1_000_000;
const MAX_CONVOLUTION_STATE_ELEMENTS: usize = 1_000_000;
const MAX_STREAMED_Q4_ELEMENTS: usize = 700_000_000;
const MAX_STREAMED_Q4_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    dimensions: Vec<usize>,
    elements: usize,
}

impl TensorShape {
    pub fn new(dimensions: Vec<usize>) -> CoreResult<Self> {
        if dimensions.is_empty() || dimensions.len() > MAX_RANK || dimensions.contains(&0) {
            return Err(CoreError::Model(
                "Tensor rank or dimension is outside Sage reference-kernel bounds".into(),
            ));
        }
        let elements = dimensions
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .filter(|total| *total <= MAX_REFERENCE_ELEMENTS)
            .ok_or_else(|| {
                CoreError::Model("Tensor size exceeds Sage reference-kernel bounds".into())
            })?;
        Ok(Self {
            dimensions,
            elements,
        })
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn elements(&self) -> usize {
        self.elements
    }
}

#[derive(Debug, Clone)]
pub struct CpuTensor {
    shape: TensorShape,
    values: Vec<f32>,
}

impl CpuTensor {
    pub fn new(dimensions: Vec<usize>, values: Vec<f32>) -> CoreResult<Self> {
        let shape = TensorShape::new(dimensions)?;
        if shape.elements() != values.len() || values.iter().any(|value| !value.is_finite()) {
            return Err(CoreError::Model(
                "Tensor values do not match the finite, declared tensor shape".into(),
            ));
        }
        Ok(Self { shape, values })
    }

    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

/// Dense row-major matrix used by the deterministic CPU reference path.
#[derive(Debug, Clone)]
pub struct CpuMatrix {
    rows: usize,
    columns: usize,
    values: Vec<f32>,
}

impl CpuMatrix {
    pub fn new(rows: usize, columns: usize, values: Vec<f32>) -> CoreResult<Self> {
        let shape = TensorShape::new(vec![rows, columns])?;
        if shape.elements() != values.len() || values.iter().any(|value| !value.is_finite()) {
            return Err(CoreError::Model(
                "Matrix values do not match the finite, declared shape".into(),
            ));
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    pub fn project(&self, input: &[f32], bias: Option<&[f32]>) -> CoreResult<Vec<f32>> {
        if input.len() != self.columns
            || input.iter().any(|value| !value.is_finite())
            || bias.is_some_and(|values| {
                values.len() != self.rows || values.iter().any(|value| !value.is_finite())
            })
        {
            return Err(CoreError::Model(
                "Projection input, bias, or weights do not match".into(),
            ));
        }
        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let start = row * self.columns;
            let sum = self.values[start..start + self.columns]
                .iter()
                .zip(input)
                .fold(0.0f64, |accumulator, (weight, activation)| {
                    accumulator + f64::from(*weight) * f64::from(*activation)
                });
            let value = (sum + f64::from(bias.map_or(0.0, |values| values[row]))) as f32;
            if !value.is_finite() {
                return Err(CoreError::Model(
                    "Projection produced a non-finite result".into(),
                ));
            }
            output.push(value);
        }
        Ok(output)
    }

    /// Project into caller-owned storage so repeated token steps can reuse
    /// activation buffers instead of allocating one result vector per matrix.
    pub fn project_into(
        &self,
        input: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
    ) -> CoreResult<()> {
        if input.len() != self.columns
            || output.len() != self.rows
            || input.iter().any(|value| !value.is_finite())
            || bias.is_some_and(|values| {
                values.len() != self.rows || values.iter().any(|value| !value.is_finite())
            })
        {
            return Err(CoreError::Model(
                "Projection input, output, bias, or weights do not match".into(),
            ));
        }
        for row in 0..self.rows {
            let start = row * self.columns;
            let sum = self.values[start..start + self.columns]
                .iter()
                .zip(input)
                .fold(0.0f64, |accumulator, (weight, activation)| {
                    accumulator + f64::from(*weight) * f64::from(*activation)
                });
            let value = (sum + f64::from(bias.map_or(0.0, |values| values[row]))) as f32;
            if !value.is_finite() {
                output.fill(0.0);
                return Err(CoreError::Model(
                    "Projection produced a non-finite result".into(),
                ));
            }
            output[row] = value;
        }
        Ok(())
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Last-dimension RMS normalization with f64 accumulation for the reference.
pub fn rms_norm(input: &[f32], scale: &[f32], epsilon: f32) -> CoreResult<Vec<f32>> {
    if input.is_empty()
        || input.len() != scale.len()
        || input.len() > MAX_REFERENCE_ELEMENTS
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || input.iter().chain(scale).any(|value| !value.is_finite())
    {
        return Err(CoreError::Model("Invalid RMS normalization input".into()));
    }
    let mean_square = input.iter().fold(0.0f64, |sum, value| {
        sum + f64::from(*value) * f64::from(*value)
    }) / input.len() as f64;
    let inverse = (mean_square + f64::from(epsilon)).sqrt().recip();
    let normalized: Vec<f32> = input
        .iter()
        .zip(scale)
        .map(|(value, weight)| (f64::from(*value) * inverse * f64::from(*weight)) as f32)
        .collect();
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(CoreError::Model(
            "RMS normalization produced a non-finite result".into(),
        ));
    }
    Ok(normalized)
}

/// Write Qwen3.5 zero-centered RMS normalization into caller-owned storage.
/// Attention layers reuse this for each head instead of allocating a vector
/// for every query and key on every token. Its learned tensor is zero-centered,
/// so the effective scale is `1 + weight` rather than `weight`.
pub fn rms_norm_zero_centered_into(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> CoreResult<()> {
    if input.len() > MAX_REFERENCE_ELEMENTS {
        output.fill(0.0);
        return Err(CoreError::Model(
            "Invalid zero-centered RMS normalization input".into(),
        ));
    }
    sage_kernels::rms_norm_zero_centered_into(input, weight, epsilon, output)
        .map_err(|error| CoreError::Model(error.into()))
}

/// Qwen's gated RMS normalization over one value head. It applies RMS
/// normalization and the learned scale to the recurrent read, then multiplies
/// by SiLU of the projected gate vector. The gate activation scratch is
/// zeroized before return.
pub fn rms_norm_gated(
    input: &[f32],
    gate: &[f32],
    scale: &[f32],
    epsilon: f32,
) -> CoreResult<Vec<f32>> {
    let mut output = vec![0.0; input.len()];
    rms_norm_gated_into(input, gate, scale, epsilon, &mut output)?;
    Ok(output)
}

/// Write SiLU-gated RMS normalization into caller-owned storage. The
/// first-party kernel uses a NEON sum-of-squares reduction on AArch64 and the
/// same scalar f64 reference elsewhere.
pub fn rms_norm_gated_into(
    input: &[f32],
    gate: &[f32],
    scale: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> CoreResult<()> {
    if input.len() > MAX_REFERENCE_ELEMENTS {
        output.fill(0.0);
        return Err(CoreError::Model(
            "Invalid gated RMS normalization input".into(),
        ));
    }
    sage_kernels::rms_norm_silu_gated_into(input, gate, scale, epsilon, output)
        .map_err(|error| CoreError::Model(error.into()))
}

pub fn silu(input: &mut [f32]) -> CoreResult<()> {
    if input.iter().any(|value| !value.is_finite()) {
        return Err(CoreError::Model("SiLU input is non-finite".into()));
    }
    for value in input {
        *value /= 1.0 + (-*value).exp();
    }
    Ok(())
}

/// Numerically stable softmax for a single attention or vocabulary row.
/// Negative infinity is accepted for masked logits; NaN and positive infinity
/// are rejected so malformed masks cannot silently change the distribution.
pub fn softmax(logits: &[f32]) -> CoreResult<Vec<f32>> {
    if logits.is_empty()
        || logits.len() > MAX_REFERENCE_ELEMENTS
        || logits
            .iter()
            .any(|value| value.is_nan() || *value == f32::INFINITY)
    {
        return Err(CoreError::Model("Invalid softmax logits".into()));
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        return Err(CoreError::Model("Softmax row is fully masked".into()));
    }
    let exponentials: Vec<f64> = logits
        .iter()
        .map(|value| f64::from(*value - maximum).exp())
        .collect();
    let denominator = exponentials.iter().sum::<f64>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(CoreError::Model("Softmax normalization failed".into()));
    }
    Ok(exponentials
        .into_iter()
        .map(|value| (value / denominator) as f32)
        .collect())
}

/// Interleaved rotary embedding for one head. Multimodal axis selection is a
/// separate model-layer concern and must provide the position for each pair.
pub fn rotary_interleaved(
    vector: &mut [f32],
    position: u64,
    rotary_dimensions: usize,
    theta: f32,
) -> CoreResult<()> {
    if vector.is_empty()
        || vector.len() > MAX_REFERENCE_ELEMENTS
        || rotary_dimensions == 0
        || rotary_dimensions > vector.len()
        || !rotary_dimensions.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 1.0
        || vector.iter().any(|value| !value.is_finite())
    {
        return Err(CoreError::Model("Invalid rotary embedding input".into()));
    }
    for pair in 0..rotary_dimensions / 2 {
        let exponent = (2 * pair) as f32 / rotary_dimensions as f32;
        let angle = (position as f64 / f64::from(theta).powf(f64::from(exponent))) as f32;
        let (sin, cos) = angle.sin_cos();
        let left = vector[pair * 2];
        let right = vector[pair * 2 + 1];
        vector[pair * 2] = left * cos - right * sin;
        vector[pair * 2 + 1] = left * sin + right * cos;
    }
    if vector[..rotary_dimensions]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(CoreError::Model(
            "Rotary embedding produced a non-finite result".into(),
        ));
    }
    Ok(())
}

/// Qwen3.5 partial RoPE for one head, using the model's split-half layout.
/// Dimensions after `rotary_dimensions` are passed through unchanged. The
/// scalar position form is for text tokens whose three MRoPE axes agree.
pub fn rotary_qwen35_partial(
    vector: &mut [f32],
    position: u64,
    rotary_dimensions: usize,
    theta: f32,
) -> CoreResult<()> {
    if vector.is_empty()
        || vector.len() > MAX_REFERENCE_ELEMENTS
        || rotary_dimensions == 0
        || rotary_dimensions > vector.len()
        || !rotary_dimensions.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 1.0
        || vector.iter().any(|value| !value.is_finite())
    {
        return Err(CoreError::Model("Invalid Qwen3.5 rotary input".into()));
    }
    let half = rotary_dimensions / 2;
    for pair in 0..half {
        let exponent = (2 * pair) as f32 / rotary_dimensions as f32;
        let angle = (position as f64 / f64::from(theta).powf(f64::from(exponent))) as f32;
        let (sin, cos) = angle.sin_cos();
        let left = vector[pair];
        let right = vector[pair + half];
        vector[pair] = left * cos - right * sin;
        vector[pair + half] = right * cos + left * sin;
    }
    if vector[..rotary_dimensions]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(CoreError::Model(
            "Qwen3.5 rotary embedding produced a non-finite result".into(),
        ));
    }
    Ok(())
}

/// Qwen3.5 multimodal RoPE for one head. Its configured frequency pairs cycle
/// through temporal, height, and width axes; the section counts must match
/// that layout exactly. Dimensions outside the partial rotary prefix pass
/// through unchanged.
pub fn rotary_qwen35_mrope_partial(
    vector: &mut [f32],
    positions: [u64; 3],
    rotary_dimensions: usize,
    sections: [usize; 3],
    theta: f32,
) -> CoreResult<()> {
    if vector.is_empty()
        || vector.len() > MAX_REFERENCE_ELEMENTS
        || rotary_dimensions == 0
        || rotary_dimensions > vector.len()
        || !rotary_dimensions.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 1.0
        || vector.iter().any(|value| !value.is_finite())
    {
        return Err(CoreError::Model("Invalid Qwen3.5 MRoPE input".into()));
    }
    let half = rotary_dimensions / 2;
    let mut observed_sections = [0_usize; 3];
    for pair in 0..half {
        observed_sections[pair % 3] += 1;
    }
    if sections != observed_sections {
        return Err(CoreError::Model(
            "Qwen3.5 MRoPE sections do not match the interleaved axis layout".into(),
        ));
    }

    let half = rotary_dimensions / 2;
    if half <= 512 {
        let mut denominators = [0.0f64; 512];
        fill_qwen35_mrope_denominators(
            &mut denominators[..half],
            rotary_dimensions,
            theta,
            sections,
        )?;
        let mut angles = [(0.0f32, 0.0f32); 512];
        prepare_qwen35_mrope_angles(&mut angles[..half], &denominators[..half], positions)?;
        rotary_qwen35_mrope_with_angles(vector, rotary_dimensions, &angles[..half])
    } else {
        let denominators = qwen35_mrope_denominators(rotary_dimensions, theta, sections)?;
        let mut angles = vec![(0.0f32, 0.0f32); half];
        prepare_qwen35_mrope_angles(&mut angles, &denominators, positions)?;
        rotary_qwen35_mrope_with_angles(vector, rotary_dimensions, &angles)
    }
}

/// Precompute the position-independent Qwen MRoPE frequency denominators for
/// one attention layer. Keeping these with the layer removes repeated `powf`
/// work from every decoded token.
pub(crate) fn qwen35_mrope_denominators(
    rotary_dimensions: usize,
    theta: f32,
    sections: [usize; 3],
) -> CoreResult<Vec<f64>> {
    if rotary_dimensions == 0
        || rotary_dimensions > MAX_REFERENCE_ELEMENTS
        || !rotary_dimensions.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 1.0
    {
        return Err(CoreError::Model("Invalid Qwen3.5 MRoPE geometry".into()));
    }
    let half = rotary_dimensions / 2;
    let mut observed_sections = [0_usize; 3];
    for pair in 0..half {
        observed_sections[pair % 3] += 1;
    }
    if sections != observed_sections {
        return Err(CoreError::Model(
            "Qwen3.5 MRoPE sections do not match the interleaved axis layout".into(),
        ));
    }
    let mut denominators = vec![0.0f64; half];
    fill_qwen35_mrope_denominators(&mut denominators, rotary_dimensions, theta, sections)?;
    Ok(denominators)
}

fn fill_qwen35_mrope_denominators(
    denominators: &mut [f64],
    rotary_dimensions: usize,
    theta: f32,
    sections: [usize; 3],
) -> CoreResult<()> {
    if rotary_dimensions == 0
        || rotary_dimensions > MAX_REFERENCE_ELEMENTS
        || !rotary_dimensions.is_multiple_of(2)
        || denominators.len() != rotary_dimensions / 2
        || !theta.is_finite()
        || theta <= 1.0
    {
        return Err(CoreError::Model("Invalid Qwen3.5 MRoPE geometry".into()));
    }
    let half = rotary_dimensions / 2;
    let mut observed_sections = [0_usize; 3];
    for pair in 0..half {
        observed_sections[pair % 3] += 1;
    }
    if sections != observed_sections {
        return Err(CoreError::Model(
            "Qwen3.5 MRoPE sections do not match the interleaved axis layout".into(),
        ));
    }
    for (pair, denominator) in denominators.iter_mut().enumerate() {
        let exponent = (2 * pair) as f32 / rotary_dimensions as f32;
        *denominator = f64::from(theta).powf(f64::from(exponent));
    }
    Ok(())
}

/// Fill caller-owned sine/cosine storage once per token position. A full
/// attention layer then reuses these angles across every query and KV head.
pub(crate) fn prepare_qwen35_mrope_angles(
    angles: &mut [(f32, f32)],
    denominators: &[f64],
    positions: [u64; 3],
) -> CoreResult<()> {
    if angles.is_empty()
        || angles.len() != denominators.len()
        || denominators
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(CoreError::Model(
            "Invalid prepared Qwen3.5 MRoPE table".into(),
        ));
    }
    for (pair, (sine, cosine)) in angles.iter_mut().enumerate() {
        let angle = (positions[pair % 3] as f64 / denominators[pair]) as f32;
        (*sine, *cosine) = angle.sin_cos();
        if !sine.is_finite() || !cosine.is_finite() {
            return Err(CoreError::Model("Qwen3.5 MRoPE angle is non-finite".into()));
        }
    }
    Ok(())
}

/// Apply already prepared MRoPE angles to one split-half head without doing
/// transcendental work. Query and key heads in the same layer share the table.
pub(crate) fn rotary_qwen35_mrope_with_angles(
    vector: &mut [f32],
    rotary_dimensions: usize,
    angles: &[(f32, f32)],
) -> CoreResult<()> {
    if vector.is_empty()
        || vector.len() > MAX_REFERENCE_ELEMENTS
        || rotary_dimensions == 0
        || rotary_dimensions > vector.len()
        || !rotary_dimensions.is_multiple_of(2)
        || angles.len() != rotary_dimensions / 2
        || vector.iter().any(|value| !value.is_finite())
        || angles
            .iter()
            .any(|(sine, cosine)| !sine.is_finite() || !cosine.is_finite())
    {
        return Err(CoreError::Model(
            "Invalid prepared Qwen3.5 MRoPE input".into(),
        ));
    }

    let half = rotary_dimensions / 2;
    for (pair, (sin, cos)) in angles.iter().copied().enumerate() {
        let left = vector[pair];
        let right = vector[pair + half];
        vector[pair] = left * cos - right * sin;
        vector[pair + half] = right * cos + left * sin;
    }
    if vector[..rotary_dimensions]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(CoreError::Model(
            "Qwen3.5 MRoPE produced a non-finite result".into(),
        ));
    }
    Ok(())
}

/// Qwen vision's axial 2D RoPE for one complete attention head. The first
/// half of split-half frequency pairs use the height coordinate; the second
/// half use width. Unlike text MRoPE, vision RoPE rotates the entire head.
pub fn rotary_qwen35_vision_axial(
    vector: &mut [f32],
    position: [u64; 2],
    theta: f32,
) -> CoreResult<()> {
    if vector.len() < 4
        || vector.len() > MAX_REFERENCE_ELEMENTS
        || !vector.len().is_multiple_of(4)
        || !theta.is_finite()
        || theta <= 1.0
        || vector.iter().any(|value| !value.is_finite())
    {
        return Err(CoreError::Model(
            "Invalid Qwen vision axial rotary input".into(),
        ));
    }
    let half = vector.len() / 2;
    let frequencies_per_axis = half / 2;
    for pair in 0..half {
        let axis = usize::from(pair >= frequencies_per_axis);
        let frequency = pair % frequencies_per_axis;
        let exponent = (2 * frequency) as f64 / half as f64;
        let angle = position[axis] as f64 / f64::from(theta).powf(exponent);
        let (sin, cos) = angle.sin_cos();
        let left = f64::from(vector[pair]);
        let right = f64::from(vector[pair + half]);
        let rotated_left = (left * cos - right * sin) as f32;
        let rotated_right = (right * cos + left * sin) as f32;
        if !rotated_left.is_finite() || !rotated_right.is_finite() {
            return Err(CoreError::Model(
                "Qwen vision axial RoPE produced a non-finite result".into(),
            ));
        }
        vector[pair] = rotated_left;
        vector[pair + half] = rotated_right;
    }
    Ok(())
}

/// Session-local KV state for one full-attention layer. Each key/value head is
/// stored as a contiguous `[position, dimension]` bank in IEEE binary16. This
/// halves storage versus f32 and lets long-context attention stream each head
/// without jumping over the other KV heads. Capacity grows geometrically; the
/// cache is deliberately non-cloneable and zeroizes activations on clear/drop.
pub struct KvCache {
    key_value_heads: usize,
    head_dimension: usize,
    maximum_context: usize,
    keys: Vec<Vec<u16>>,
    values: Vec<Vec<u16>>,
}

impl KvCache {
    pub fn new(
        key_value_heads: usize,
        head_dimension: usize,
        maximum_context: usize,
    ) -> CoreResult<Self> {
        if key_value_heads == 0
            || key_value_heads > 32
            || head_dimension == 0
            || head_dimension > 1024
            || maximum_context == 0
            || maximum_context > crate::qwen35::SAGE_CONTEXT_LIMIT as usize
        {
            return Err(CoreError::Model(
                "KV cache dimensions exceed Sage's configured bounds".into(),
            ));
        }
        let row_elements = key_value_heads
            .checked_mul(head_dimension)
            .ok_or_else(|| CoreError::Model("KV cache shape overflow".into()))?;
        let _maximum_elements = row_elements
            .checked_mul(maximum_context)
            .filter(|elements| {
                elements
                    .checked_mul(2)
                    .is_some_and(|total| total <= MAX_KV_CACHE_ELEMENTS)
            })
            .ok_or_else(|| {
                CoreError::Model("KV cache exceeds Sage's per-layer memory bound".into())
            })?;
        Ok(Self {
            key_value_heads,
            head_dimension,
            maximum_context,
            keys: (0..key_value_heads).map(|_| Vec::new()).collect(),
            values: (0..key_value_heads).map(|_| Vec::new()).collect(),
        })
    }

    /// Reserve exactly the validated prompt prefix before prefill starts.
    /// Incremental decode uses geometric growth separately.
    pub fn reserve_positions(&mut self, positions: usize) -> CoreResult<()> {
        if positions > self.maximum_context {
            return Err(CoreError::Model(
                "KV cache reservation exceeds its admitted context".into(),
            ));
        }
        self.grow_to_positions(positions)
    }

    fn reserve_geometrically(&mut self, positions: usize) -> CoreResult<()> {
        if positions > self.maximum_context {
            return Err(CoreError::Model(
                "KV cache reservation exceeds its admitted context".into(),
            ));
        }
        let current_positions = self.capacity_positions();
        if positions <= current_positions {
            return Ok(());
        }

        let geometric_positions = if current_positions == 0 {
            16
        } else {
            current_positions.saturating_mul(2)
        };
        let target_positions = positions.max(geometric_positions).min(self.maximum_context);
        self.grow_to_positions(target_positions)
    }

    fn grow_to_positions(&mut self, target_positions: usize) -> CoreResult<()> {
        let target_elements = target_positions
            .checked_mul(self.head_dimension)
            .ok_or_else(|| CoreError::Model("KV cache reservation shape overflow".into()))?;
        for bank in self.keys.iter_mut().chain(&mut self.values) {
            grow_zeroizing_u16(bank, target_elements)?;
        }
        Ok(())
    }

    /// Append exactly one decoded position. Position order is implicit and
    /// monotonic, so a cache cannot be populated with gaps or future keys.
    pub fn append(&mut self, keys: &[f32], values: &[f32]) -> CoreResult<()> {
        if self.context_length() >= self.maximum_context
            || keys.len() != self.key_value_heads * self.head_dimension
            || values.len() != self.key_value_heads * self.head_dimension
            || keys
                .iter()
                .chain(values)
                .any(|value| !value.is_finite() || value.abs() > 65_504.0)
        {
            return Err(CoreError::Model(
                "KV cache append is malformed, non-finite, or beyond context".into(),
            ));
        }
        self.reserve_geometrically(self.context_length() + 1)?;
        for ((key_bank, value_bank), (key_head, value_head)) in
            self.keys.iter_mut().zip(&mut self.values).zip(
                keys.chunks_exact(self.head_dimension)
                    .zip(values.chunks_exact(self.head_dimension)),
            )
        {
            key_bank.extend(
                key_head
                    .iter()
                    .map(|value| sage_kernels::f32_to_f16_bits(*value)),
            );
            value_bank.extend(
                value_head
                    .iter()
                    .map(|value| sage_kernels::f32_to_f16_bits(*value)),
            );
        }
        Ok(())
    }

    pub fn context_length(&self) -> usize {
        self.keys
            .first()
            .map_or(0, |bank| bank.len() / self.head_dimension)
    }

    pub fn key_value_heads(&self) -> usize {
        self.key_value_heads
    }

    pub fn head_dimension(&self) -> usize {
        self.head_dimension
    }

    pub fn keys_for_head(&self, head: usize) -> Option<&[u16]> {
        self.keys.get(head).map(Vec::as_slice)
    }

    pub fn values_for_head(&self, head: usize) -> Option<&[u16]> {
        self.values.get(head).map(Vec::as_slice)
    }

    /// Number of positions backed by both key and value allocations. Exposed
    /// for bounded-memory diagnostics; this can exceed the live prefix due to
    /// geometric growth but never exceeds the configured context limit.
    pub fn capacity_positions(&self) -> usize {
        self.keys
            .iter()
            .chain(&self.values)
            .map(|bank| bank.capacity() / self.head_dimension)
            .min()
            .unwrap_or(0)
            .min(self.maximum_context)
    }

    pub fn clear(&mut self) {
        for bank in self.keys.iter_mut().chain(&mut self.values) {
            bank.as_mut_slice().zeroize();
            bank.clear();
        }
    }
}

fn grow_zeroizing_u16(values: &mut Vec<u16>, target_elements: usize) -> CoreResult<()> {
    if target_elements <= values.capacity() {
        return Ok(());
    }
    let mut replacement = Vec::new();
    replacement
        .try_reserve_exact(target_elements)
        .map_err(|_| CoreError::Model("KV cache allocation was denied".into()))?;
    replacement.extend_from_slice(values);
    // Do not let a capacity change leave old key/value bytes in a freed
    // allocation. Zero before replacing the allocation, including on growth.
    values.as_mut_slice().zeroize();
    *values = replacement;
    Ok(())
}

impl Drop for KvCache {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Reusable bounded workspace for blockwise grouped-query attention. Score
/// storage is fixed-size regardless of context; one head-sized temporary
/// combines each block into the running output.
pub(crate) struct GroupedQueryAttentionScratch {
    output: Vec<f32>,
    block_output: Vec<f32>,
    scores: Vec<f64>,
    query_tile_heads: usize,
    scores_live: bool,
}

impl GroupedQueryAttentionScratch {
    pub(crate) fn new(
        query_heads: usize,
        key_value_heads: usize,
        head_dimension: usize,
        maximum_context: usize,
    ) -> CoreResult<Self> {
        let output_elements = query_heads.checked_mul(head_dimension);
        if query_heads == 0
            || query_heads > 64
            || key_value_heads == 0
            || key_value_heads > 32
            || !query_heads.is_multiple_of(key_value_heads)
            || output_elements.is_none_or(|elements| elements > 64 * 1024)
            || head_dimension == 0
            || head_dimension > 512
            || maximum_context == 0
            || maximum_context > crate::qwen35::SAGE_CONTEXT_LIMIT as usize
        {
            return Err(CoreError::Model(
                "Attention scratch dimensions exceed Sage's configured bounds".into(),
            ));
        }
        let output_elements = output_elements.expect("validated attention output size");
        let query_tile_heads = (query_heads / key_value_heads).min(4);
        let block_output_elements = query_tile_heads
            .checked_mul(head_dimension)
            .ok_or_else(|| CoreError::Model("Attention block output shape overflow".into()))?;
        let score_elements = maximum_context
            .min(ATTENTION_SCORE_BLOCK_SIZE)
            .checked_mul(query_tile_heads)
            .ok_or_else(|| CoreError::Model("Attention score shape overflow".into()))?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_elements)
            .map_err(|_| CoreError::Model("Attention output allocation was denied".into()))?;
        output.resize(output_elements, 0.0);
        let mut block_output = Vec::new();
        block_output
            .try_reserve_exact(block_output_elements)
            .map_err(|_| CoreError::Model("Attention block output allocation was denied".into()))?;
        block_output.resize(block_output_elements, 0.0);
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(score_elements)
            .map_err(|_| CoreError::Model("Attention score allocation was denied".into()))?;
        scores.resize(score_elements, 0.0);
        Ok(Self {
            output,
            block_output,
            scores,
            query_tile_heads,
            scores_live: false,
        })
    }

    pub(crate) fn output_mut(&mut self) -> &mut [f32] {
        &mut self.output
    }

    /// Zero activation contents while retaining the admitted allocations for
    /// the next token.
    pub(crate) fn clear(&mut self) {
        self.clear_output();
        self.block_output.as_mut_slice().zeroize();
        if self.scores_live {
            self.scores.as_mut_slice().zeroize();
            self.scores_live = false;
        }
    }

    /// Clear the projected attention vector after use. `grouped_query_attention_into`
    /// has already zeroized the live score prefix before returning.
    pub(crate) fn clear_output(&mut self) {
        self.output.as_mut_slice().zeroize();
    }

    #[cfg(test)]
    pub(crate) fn is_cleared(&self) -> bool {
        self.output.iter().all(|value| *value == 0.0)
            && self.block_output.iter().all(|value| *value == 0.0)
            && self.scores.iter().all(|value| *value == 0.0)
            && !self.scores_live
    }
}

impl Drop for GroupedQueryAttentionScratch {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Fill caller-owned output using caller-owned scratch. Queries are laid out
/// `[query_head, dimension]`; cache rows are `[position, key_value_head, dimension]`.
/// A stable blockwise softmax combines each score block into a running output,
/// so scratch remains bounded as context grows. Scores and temporary values are
/// zeroized before returning; on failure partial activations are cleared.
pub(crate) fn grouped_query_attention_into(
    query: &[f32],
    query_heads: usize,
    cache: &KvCache,
    scratch: &mut GroupedQueryAttentionScratch,
) -> CoreResult<()> {
    let key_value_heads = cache.key_value_heads;
    let head_dimension = cache.head_dimension;
    let context = cache.context_length();
    let expected_query = query_heads.checked_mul(head_dimension);
    let attention_multiplies = query_heads
        .checked_mul(head_dimension)
        .and_then(|elements| elements.checked_mul(context))
        .and_then(|elements| elements.checked_mul(2));
    let query_group_size = query_heads.checked_div(key_value_heads.max(1));
    let query_tile_heads = query_group_size.map(|count| count.min(4));
    if query_heads == 0
        || query_heads > 64
        || key_value_heads == 0
        || key_value_heads > 32
        || !query_heads.is_multiple_of(key_value_heads)
        || expected_query != Some(query.len())
        || scratch.output.len() != query.len()
        || query_tile_heads != Some(scratch.query_tile_heads)
        || scratch.scores.len() < context.min(ATTENTION_SCORE_BLOCK_SIZE) * scratch.query_tile_heads
        || scratch.block_output.len() != scratch.query_tile_heads * head_dimension
        || context == 0
        || query.len() > MAX_REFERENCE_ELEMENTS
        || attention_multiplies.is_none_or(|count| count > MAX_ATTENTION_MULTIPLIES)
        || query.iter().any(|value| !value.is_finite())
    {
        scratch.clear();
        return Err(CoreError::Model(
            "Grouped-query attention dimensions or inputs are invalid".into(),
        ));
    }

    let group_size = query_group_size.expect("validated grouped-query dimensions");
    let query_tile_heads = scratch.query_tile_heads;
    let scale = (head_dimension as f64).sqrt().recip();
    scratch.scores_live = true;
    let result = (|| {
        let scores = &mut scratch.scores;
        let block_output = &mut scratch.block_output;
        let output = &mut scratch.output;
        for kv_head in 0..key_value_heads {
            for group_offset in (0..group_size).step_by(query_tile_heads) {
                let tile_heads = (group_size - group_offset).min(query_tile_heads);
                let first_query_head = kv_head * group_size + group_offset;
                let query_start = first_query_head * head_dimension;
                let tile_query = &query[query_start..query_start + tile_heads * head_dimension];
                let mut maximum = [f64::NEG_INFINITY; 4];
                let mut denominator = [0.0f64; 4];
                let mut previous_scales = [0.0f64; 4];
                for query_head in 0..tile_heads {
                    let output_start = query_start + query_head * head_dimension;
                    output[output_start..output_start + head_dimension].fill(0.0);
                }
                for block_start in (0..context).step_by(ATTENTION_SCORE_BLOCK_SIZE) {
                    let block_length = (context - block_start).min(ATTENTION_SCORE_BLOCK_SIZE);
                    let score_count = tile_heads * block_length;
                    let block_scores = &mut scores[..score_count];
                    let row_start = block_start * head_dimension;
                    let row_end = row_start + block_length * head_dimension;
                    sage_kernels::dot_rows_f16_grouped(
                        tile_query,
                        tile_heads,
                        &cache.keys[kv_head][row_start..row_end],
                        block_length,
                        head_dimension,
                        0,
                        block_scores,
                    )
                    .map_err(|error| {
                        CoreError::Model(format!(
                            "Sage grouped attention QK kernel failed: {error}"
                        ))
                    })?;
                    for query_head in 0..tile_heads {
                        let head_scores = &mut block_scores
                            [query_head * block_length..(query_head + 1) * block_length];
                        for score in head_scores.iter_mut() {
                            *score *= scale;
                        }
                        let block_maximum = head_scores
                            .iter()
                            .copied()
                            .fold(f64::NEG_INFINITY, f64::max);
                        let next_maximum = maximum[query_head].max(block_maximum);
                        let previous_scale = if maximum[query_head].is_finite() {
                            (maximum[query_head] - next_maximum).exp()
                        } else {
                            0.0
                        };
                        previous_scales[query_head] = previous_scale;
                        let mut block_denominator = 0.0f64;
                        for score in head_scores.iter_mut() {
                            *score = (*score - next_maximum).exp();
                            block_denominator += *score;
                        }
                        denominator[query_head] =
                            denominator[query_head] * previous_scale + block_denominator;
                        if !denominator[query_head].is_finite() || denominator[query_head] <= 0.0 {
                            return Err(CoreError::Model(
                                "Grouped-query attention softmax failed".into(),
                            ));
                        }
                        maximum[query_head] = next_maximum;
                    }

                    sage_kernels::weighted_sum_rows_f16_grouped_into(
                        &cache.values[kv_head][row_start..row_end],
                        block_scores,
                        tile_heads,
                        block_length,
                        head_dimension,
                        0,
                        &mut block_output[..tile_heads * head_dimension],
                    )
                    .map_err(|error| {
                        CoreError::Model(format!(
                            "Sage grouped attention WV kernel failed: {error}"
                        ))
                    })?;
                    for query_head in 0..tile_heads {
                        let output_start = query_start + query_head * head_dimension;
                        let output_head = &mut output[output_start..output_start + head_dimension];
                        let block_head = &block_output
                            [query_head * head_dimension..(query_head + 1) * head_dimension];
                        for (accumulator, block_value) in
                            output_head.iter_mut().zip(block_head.iter())
                        {
                            *accumulator = (f64::from(*accumulator) * previous_scales[query_head]
                                + f64::from(*block_value))
                                as f32;
                        }
                    }
                }
                for (query_head, head_denominator) in
                    denominator.iter().enumerate().take(tile_heads)
                {
                    let output_start = query_start + query_head * head_dimension;
                    let output_head = &mut output[output_start..output_start + head_dimension];
                    for value in output_head.iter_mut() {
                        *value = (f64::from(*value) / head_denominator) as f32;
                    }
                    if output_head.iter().any(|value| !value.is_finite()) {
                        return Err(CoreError::Model(
                            "Grouped-query attention output is non-finite".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    })();
    if result.is_err() {
        scratch.scores.as_mut_slice().zeroize();
        scratch.output.as_mut_slice().zeroize();
        scratch.block_output.as_mut_slice().zeroize();
    } else {
        scratch.scores.as_mut_slice().zeroize();
        scratch.block_output.as_mut_slice().zeroize();
    }
    scratch.scores_live = false;
    result
}

/// Per-head recurrent matrix state for the gated delta rule. Layout is
/// `[value_head, key_dimension, value_dimension]`, matching the mathematical
/// state `S ∈ R^(d_key × d_value)` used by Qwen3.5's linear-attention layers.
/// A state belongs to one inference session and is zeroized when cleared or
/// dropped. AArch64 uses Sage's measured NEON update; other targets use the
/// portable f64 scalar reference.
pub struct GatedDeltaState {
    value_heads: usize,
    key_dimension: usize,
    value_dimension: usize,
    values: Vec<f32>,
}

impl GatedDeltaState {
    pub fn new(
        value_heads: usize,
        key_dimension: usize,
        value_dimension: usize,
    ) -> CoreResult<Self> {
        if value_heads == 0
            || value_heads > 64
            || key_dimension == 0
            || key_dimension > 512
            || value_dimension == 0
            || value_dimension > 512
        {
            return Err(CoreError::Model(
                "Gated delta state dimensions exceed Sage's bounds".into(),
            ));
        }
        let elements = value_heads
            .checked_mul(key_dimension)
            .and_then(|count| count.checked_mul(value_dimension))
            .filter(|count| *count <= MAX_DELTA_STATE_ELEMENTS)
            .ok_or_else(|| {
                CoreError::Model("Gated delta state exceeds Sage's per-layer memory bound".into())
            })?;
        Ok(Self {
            value_heads,
            key_dimension,
            value_dimension,
            values: vec![0.0; elements],
        })
    }

    /// Apply one token to every value head and return its post-update read.
    /// Query and key vectors are L2-normalized with epsilon 1e-6; query is
    /// additionally scaled by `1/sqrt(key_dimension)`. `log_decay` is the
    /// non-positive log of alpha, and beta is the bounded delta write rate.
    ///
    /// The update is `S' = exp(g) * S + beta * (v - exp(g) * S*k) * k^T` and
    /// the output is `S' * q`. This is algebraically the gated delta update.
    pub fn step(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        log_decay: &[f32],
        beta: &[f32],
    ) -> CoreResult<Vec<f32>> {
        let output_len = self
            .value_heads
            .checked_mul(self.value_dimension)
            .ok_or_else(|| CoreError::Model("Gated delta output width overflow".into()))?;
        let mut output = vec![0.0f32; output_len];
        self.step_into(query, key, value, log_decay, beta, &mut output)?;
        Ok(output)
    }

    /// Apply one token into caller-owned output storage. This keeps the
    /// per-head update allocation-free for a retained inference session.
    pub fn step_into(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        log_decay: &[f32],
        beta: &[f32],
        output: &mut [f32],
    ) -> CoreResult<()> {
        let query_len = self.value_heads.checked_mul(self.key_dimension);
        let value_len = self.value_heads.checked_mul(self.value_dimension);
        if query_len != Some(query.len())
            || query.len() != key.len()
            || value_len != Some(value.len())
            || output.len() != value.len()
            || log_decay.len() != self.value_heads
            || beta.len() != self.value_heads
            || query
                .iter()
                .chain(key)
                .chain(value)
                .chain(log_decay)
                .chain(beta)
                .any(|number| !number.is_finite())
            || log_decay.iter().any(|number| *number > 0.0)
            || beta.iter().any(|number| !(0.0..=1.0).contains(number))
        {
            output.zeroize();
            return Err(CoreError::Model(
                "Gated delta step has malformed tensors or invalid gates".into(),
            ));
        }
        output.fill(0.0);

        #[cfg(target_arch = "aarch64")]
        {
            let matrix_elements = self.key_dimension * self.value_dimension;
            let inverse_query_scale = (self.key_dimension as f64).sqrt().recip();
            let epsilon = 1e-6f64;
            let mut key_direction = Zeroizing::new([0.0f32; 512]);
            let mut query_direction = Zeroizing::new([0.0f32; 512]);

            for head in 0..self.value_heads {
                let query_start = head * self.key_dimension;
                let value_start = head * self.value_dimension;
                let state_start = head * matrix_elements;
                let query_norm = query[query_start..query_start + self.key_dimension]
                    .iter()
                    .fold(0.0f64, |sum, item| sum + f64::from(*item).powi(2))
                    + epsilon;
                let query_norm = query_norm.sqrt();
                let key_norm = key[query_start..query_start + self.key_dimension]
                    .iter()
                    .fold(0.0f64, |sum, item| sum + f64::from(*item).powi(2))
                    + epsilon;
                let key_norm = key_norm.sqrt();
                for index in 0..self.key_dimension {
                    key_direction[index] = (f64::from(key[query_start + index]) / key_norm) as f32;
                    query_direction[index] = (f64::from(query[query_start + index]) / query_norm
                        * inverse_query_scale) as f32;
                }
                let state_end = state_start + matrix_elements;
                let output_end = value_start + self.value_dimension;
                if let Err(error) = sage_kernels::gated_delta_step(
                    &mut self.values[state_start..state_end],
                    &key_direction[..self.key_dimension],
                    &query_direction[..self.key_dimension],
                    &value[value_start..output_end],
                    f64::from(log_decay[head]).exp() as f32,
                    beta[head],
                    &mut output[value_start..output_end],
                ) {
                    output.zeroize();
                    self.clear();
                    return Err(CoreError::Model(format!(
                        "Gated delta state update failed and was cleared: {error}"
                    )));
                }
            }
            Ok(())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let matrix_elements = self.key_dimension * self.value_dimension;
            let inverse_query_scale = (self.key_dimension as f64).sqrt().recip();
            let epsilon = 1e-6f64;

            for head in 0..self.value_heads {
                let query_start = head * self.key_dimension;
                let key_start = query_start;
                let value_start = head * self.value_dimension;
                let state_start = head * matrix_elements;
                let query_norm = query[query_start..query_start + self.key_dimension]
                    .iter()
                    .fold(0.0f64, |sum, item| sum + f64::from(*item).powi(2))
                    + epsilon;
                let query_norm = query_norm.sqrt();
                let key_norm = key[key_start..key_start + self.key_dimension]
                    .iter()
                    .fold(0.0f64, |sum, item| sum + f64::from(*item).powi(2))
                    + epsilon;
                let key_norm = key_norm.sqrt();
                let decay = f64::from(log_decay[head]).exp();
                let write_rate = f64::from(beta[head]);

                for value_index in 0..self.value_dimension {
                    let mut prior_read = 0.0f64;
                    for key_index in 0..self.key_dimension {
                        let matrix_index =
                            state_start + key_index * self.value_dimension + value_index;
                        prior_read += f64::from(self.values[matrix_index])
                            * decay
                            * (f64::from(key[key_start + key_index]) / key_norm);
                    }
                    let correction =
                        write_rate * (f64::from(value[value_start + value_index]) - prior_read);
                    for key_index in 0..self.key_dimension {
                        let matrix_index =
                            state_start + key_index * self.value_dimension + value_index;
                        let normalized_key = f64::from(key[key_start + key_index]) / key_norm;
                        let updated = (decay * f64::from(self.values[matrix_index])
                            + correction * normalized_key)
                            as f32;
                        if !updated.is_finite() {
                            output.zeroize();
                            self.clear();
                            return Err(CoreError::Model(
                                "Gated delta state update became non-finite and was cleared".into(),
                            ));
                        }
                        self.values[matrix_index] = updated;
                    }
                }

                for value_index in 0..self.value_dimension {
                    let mut read = 0.0f64;
                    for key_index in 0..self.key_dimension {
                        let matrix_index =
                            state_start + key_index * self.value_dimension + value_index;
                        let normalized_query =
                            f64::from(query[query_start + key_index]) / query_norm;
                        read += f64::from(self.values[matrix_index])
                            * normalized_query
                            * inverse_query_scale;
                    }
                    let read = read as f32;
                    if !read.is_finite() {
                        output.zeroize();
                        self.clear();
                        return Err(CoreError::Model(
                            "Gated delta output became non-finite and state was cleared".into(),
                        ));
                    }
                    output[value_start + value_index] = read;
                }
            }
            Ok(())
        }
    }

    pub fn value_heads(&self) -> usize {
        self.value_heads
    }

    pub fn key_dimension(&self) -> usize {
        self.key_dimension
    }

    pub fn value_dimension(&self) -> usize {
        self.value_dimension
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn clear(&mut self) {
        self.values.zeroize();
    }
}

impl Drop for GatedDeltaState {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Per-session input history for a causal, depthwise QKV convolution. Weights
/// are laid out `[channel, tap]`, oldest tap first, like a flattened
/// `[channel, 1, kernel]` model tensor. The convolution output uses SiLU, as
/// required by the pinned Qwen3.5 linear-attention block.
pub struct CausalDepthwiseConvState {
    channels: usize,
    kernel_size: usize,
    history: Vec<f32>,
}

impl CausalDepthwiseConvState {
    pub fn new(channels: usize, kernel_size: usize) -> CoreResult<Self> {
        if channels == 0 || channels > 16_384 || kernel_size == 0 || kernel_size > 64 {
            return Err(CoreError::Model(
                "Causal convolution dimensions exceed Sage's bounds".into(),
            ));
        }
        let history_elements = channels
            .checked_mul(kernel_size - 1)
            .filter(|count| *count <= MAX_CONVOLUTION_STATE_ELEMENTS)
            .ok_or_else(|| {
                CoreError::Model("Causal convolution state exceeds Sage's memory bound".into())
            })?;
        Ok(Self {
            channels,
            kernel_size,
            history: vec![0.0; history_elements],
        })
    }

    /// Process one projected QKV token. Validation and convolution finish
    /// before history advances, so malformed weights cannot partially consume
    /// a token from the session stream.
    pub fn step(
        &mut self,
        input: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
    ) -> CoreResult<Vec<f32>> {
        let mut output = vec![0.0f32; self.channels];
        self.step_into(input, weights, bias, &mut output)?;
        Ok(output)
    }

    /// Convolve one token into caller-owned output. History advances only
    /// after every channel succeeds, preserving the previous prefix on error.
    pub fn step_into(
        &mut self,
        input: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        output: &mut [f32],
    ) -> CoreResult<()> {
        let expected_weights = self.channels.checked_mul(self.kernel_size);
        if input.len() != self.channels
            || output.len() != self.channels
            || expected_weights != Some(weights.len())
            || bias.is_some_and(|values| values.len() != self.channels)
            || input
                .iter()
                .chain(weights)
                .chain(bias.into_iter().flatten())
                .any(|number| !number.is_finite())
        {
            output.zeroize();
            return Err(CoreError::Model(
                "Causal depthwise convolution tensors do not match".into(),
            ));
        }

        output.fill(0.0);
        let history_per_channel = self.kernel_size - 1;
        for channel in 0..self.channels {
            let history_start = channel * history_per_channel;
            let mut sum = f64::from(bias.map_or(0.0, |values| values[channel]));
            for tap in 0..self.kernel_size {
                let sample = if tap < history_per_channel {
                    self.history[history_start + tap]
                } else {
                    input[channel]
                };
                sum += f64::from(sample) * f64::from(weights[channel * self.kernel_size + tap]);
            }
            let convolved = sum as f32;
            if !convolved.is_finite() {
                output.zeroize();
                return Err(CoreError::Model(
                    "Causal depthwise convolution became non-finite".into(),
                ));
            }
            let activated = (f64::from(convolved) / (1.0 + (-f64::from(convolved)).exp())) as f32;
            if !activated.is_finite() {
                output.zeroize();
                return Err(CoreError::Model(
                    "Causal depthwise SiLU became non-finite".into(),
                ));
            }
            output[channel] = activated;
        }

        if history_per_channel > 0 {
            for (channel, value) in input.iter().enumerate() {
                let start = channel * history_per_channel;
                if history_per_channel > 1 {
                    self.history
                        .copy_within(start + 1..start + history_per_channel, start);
                }
                self.history[start + history_per_channel - 1] = *value;
            }
        }
        Ok(())
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn clear(&mut self) {
        self.history.zeroize();
    }
}

impl Drop for CausalDepthwiseConvState {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Convert Qwen3.5's `in_proj_a`, `in_proj_b`, `A_log`, and `dt_bias` outputs
/// into the log-decay and write-rate vectors consumed by `GatedDeltaState`.
pub fn qwen35_gated_delta_coefficients(
    projected_decay: &[f32],
    projected_beta: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
) -> CoreResult<(Vec<f32>, Vec<f32>)> {
    let heads = projected_decay.len();
    if heads == 0 || heads > 64 {
        return Err(CoreError::Model(
            "Qwen gated-delta coefficient tensors do not match".into(),
        ));
    }
    let mut log_decay = vec![0.0; heads];
    let mut beta = vec![0.0; heads];
    qwen35_gated_delta_coefficients_into(
        projected_decay,
        projected_beta,
        a_log,
        dt_bias,
        &mut log_decay,
        &mut beta,
    )?;
    Ok((log_decay, beta))
}

/// Convert Qwen's decay and beta projections into caller-owned coefficient
/// buffers without allocating per-token vectors.
pub fn qwen35_gated_delta_coefficients_into(
    projected_decay: &[f32],
    projected_beta: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    log_decay: &mut [f32],
    beta: &mut [f32],
) -> CoreResult<()> {
    let heads = projected_decay.len();
    if heads == 0
        || heads > 64
        || log_decay.len() != heads
        || beta.len() != heads
        || projected_beta.len() != heads
        || a_log.len() != heads
        || dt_bias.len() != heads
        || projected_decay
            .iter()
            .chain(projected_beta)
            .chain(a_log)
            .chain(dt_bias)
            .any(|number| !number.is_finite())
    {
        log_decay.zeroize();
        beta.zeroize();
        return Err(CoreError::Model(
            "Qwen gated-delta coefficient tensors do not match".into(),
        ));
    }
    for head in 0..heads {
        let preactivation = f64::from(projected_decay[head]) + f64::from(dt_bias[head]);
        let softplus = if preactivation > 20.0 {
            preactivation
        } else if preactivation < -20.0 {
            preactivation.exp()
        } else {
            preactivation.exp().ln_1p()
        };
        let decay = -f64::from(a_log[head]).exp() * softplus;
        let beta_logit = f64::from(projected_beta[head]);
        let write_rate = if beta_logit >= 0.0 {
            1.0 / (1.0 + (-beta_logit).exp())
        } else {
            let exponent = beta_logit.exp();
            exponent / (1.0 + exponent)
        };
        let decay = decay as f32;
        let write_rate = write_rate as f32;
        if !decay.is_finite() || !write_rate.is_finite() {
            log_decay.zeroize();
            beta.zeroize();
            return Err(CoreError::Model(
                "Qwen gated-delta coefficient computation overflowed".into(),
            ));
        }
        log_decay[head] = decay;
        beta[head] = write_rate;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct QuantizedQ4Matrix {
    rows: usize,
    columns: usize,
    group_size: usize,
    scales: Vec<f32>,
    packed_signed_values: Vec<u8>,
    metal: Option<MetalQ4Storage>,
}

#[derive(Debug, Clone)]
struct MetalQ4Storage {
    context: Arc<MetalContext>,
    scales: Arc<MetalBuffer>,
    packed_signed_values: Arc<MetalBuffer>,
    workspace: Arc<Mutex<MetalQ4Workspace>>,
}

/// Incremental Sage Q4 weight importer. It retains only the compressed matrix,
/// one quantization group, and a possible half-byte between groups. This lets
/// the pinned Qwen embedding table be quantized from bounded safetensors
/// chunks without holding a second full f32 copy in memory.
pub struct QuantizedQ4Builder {
    rows: usize,
    columns: usize,
    group_size: usize,
    expected_elements: usize,
    seen_elements: usize,
    group: Vec<f32>,
    scales: Vec<f32>,
    packed_signed_values: Vec<u8>,
    pending_nibble: Option<u8>,
}

impl QuantizedQ4Builder {
    /// `maximum_elements` is an explicit caller budget. The builder also caps
    /// the packed weights and scales at 1 GiB, regardless of that budget.
    pub fn new(
        rows: usize,
        columns: usize,
        group_size: usize,
        maximum_elements: usize,
    ) -> CoreResult<Self> {
        let expected_elements = rows
            .checked_mul(columns)
            .filter(|elements| *elements > 0)
            .ok_or_else(|| CoreError::Model("Q4 streamed matrix shape is invalid".into()))?;
        if !(16..=4096).contains(&group_size)
            || expected_elements > maximum_elements.min(MAX_STREAMED_Q4_ELEMENTS)
        {
            return Err(CoreError::Model(
                "Q4 streamed matrix exceeds its group or element budget".into(),
            ));
        }
        let packed_bytes = expected_elements.div_ceil(2);
        let scale_count = expected_elements.div_ceil(group_size);
        let estimated_bytes = u64::try_from(packed_bytes)
            .ok()
            .and_then(|bytes| {
                u64::try_from(scale_count)
                    .ok()?
                    .checked_mul(std::mem::size_of::<f32>() as u64)?
                    .checked_add(bytes)
            })
            .ok_or_else(|| CoreError::Model("Q4 streamed matrix size overflow".into()))?;
        if estimated_bytes > MAX_STREAMED_Q4_BYTES {
            return Err(CoreError::Model(
                "Q4 streamed matrix exceeds Sage's 1 GiB compressed-weight budget".into(),
            ));
        }
        Ok(Self {
            rows,
            columns,
            group_size,
            expected_elements,
            seen_elements: 0,
            group: Vec::with_capacity(group_size),
            scales: Vec::with_capacity(scale_count),
            packed_signed_values: Vec::with_capacity(packed_bytes),
            pending_nibble: None,
        })
    }

    /// Add the next finite source chunk. Chunks may split quantization groups
    /// or end on an odd element; output remains byte-for-byte equivalent to
    /// Sage's batch `QuantizedQ4Matrix::encode` layout.
    pub fn push(&mut self, values: &[f32]) -> CoreResult<()> {
        let next_seen = self
            .seen_elements
            .checked_add(values.len())
            .filter(|count| *count <= self.expected_elements)
            .ok_or_else(|| {
                CoreError::Model("Q4 source exceeds the declared matrix shape".into())
            })?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CoreError::Model(
                "Q4 source contains non-finite weights".into(),
            ));
        }
        for value in values {
            self.group.push(*value);
            if self.group.len() == self.group_size {
                self.flush_group();
            }
        }
        self.seen_elements = next_seen;
        Ok(())
    }

    /// Finish only after exactly the declared number of source elements has
    /// arrived. A partial or truncated model tensor cannot become a matrix.
    pub fn finish(mut self) -> CoreResult<QuantizedQ4Matrix> {
        if self.seen_elements != self.expected_elements {
            return Err(CoreError::Model(
                "Q4 source ended before the declared matrix shape was complete".into(),
            ));
        }
        if !self.group.is_empty() {
            self.flush_group();
        }
        if let Some(low) = self.pending_nibble.take() {
            self.packed_signed_values.push(low | 0x80);
        }
        if self.packed_signed_values.len() != self.expected_elements.div_ceil(2)
            || self.scales.len() != self.expected_elements.div_ceil(self.group_size)
        {
            return Err(CoreError::Model(
                "Q4 streamed matrix packing did not match its declared shape".into(),
            ));
        }
        Ok(QuantizedQ4Matrix {
            rows: self.rows,
            columns: self.columns,
            group_size: self.group_size,
            scales: self.scales,
            packed_signed_values: self.packed_signed_values,
            metal: None,
        })
    }

    fn flush_group(&mut self) {
        debug_assert!(!self.group.is_empty() && self.group.len() <= self.group_size);
        let maximum = self
            .group
            .iter()
            .fold(0.0f32, |prior, value| prior.max(value.abs()));
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 7.0 };
        self.scales.push(scale);
        for value in self.group.drain(..) {
            let nibble = ((value / scale).round().clamp(-7.0, 7.0) as i8 + 8) as u8;
            if let Some(low) = self.pending_nibble.take() {
                self.packed_signed_values.push(low | (nibble << 4));
            } else {
                self.pending_nibble = Some(nibble);
            }
        }
    }
}

impl QuantizedQ4Matrix {
    /// Sage's symmetric signed Q4 reference format uses values in [-7, 7],
    /// stores two offset nibbles per byte, and records one scale per group.
    pub fn encode(
        rows: usize,
        columns: usize,
        group_size: usize,
        values: &[f32],
    ) -> CoreResult<Self> {
        let shape = TensorShape::new(vec![rows, columns])?;
        if shape.elements() != values.len()
            || group_size == 0
            || group_size > 4096
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(CoreError::Model("Invalid Q4 matrix input".into()));
        }
        let groups = values.len().div_ceil(group_size);
        let mut scales = Vec::with_capacity(groups);
        let mut quantized = Vec::with_capacity(values.len());
        for group in values.chunks(group_size) {
            let maximum = group
                .iter()
                .fold(0.0f32, |prior, value| prior.max(value.abs()));
            let scale = if maximum == 0.0 { 1.0 } else { maximum / 7.0 };
            scales.push(scale);
            quantized.extend(
                group
                    .iter()
                    .map(|value| ((*value / scale).round().clamp(-7.0, 7.0) as i8 + 8) as u8),
            );
        }
        let mut packed_signed_values = Vec::with_capacity(quantized.len().div_ceil(2));
        for pair in quantized.chunks(2) {
            let low = pair[0];
            let high = pair.get(1).copied().unwrap_or(8);
            packed_signed_values.push(low | (high << 4));
        }
        Ok(Self {
            rows,
            columns,
            group_size,
            scales,
            packed_signed_values,
            metal: None,
        })
    }

    pub fn project(&self, input: &[f32]) -> CoreResult<Vec<f32>> {
        let elements = self
            .rows
            .checked_mul(self.columns)
            .ok_or_else(|| CoreError::Model("Q4 matrix shape overflow".into()))?;
        if self.group_size == 0
            || input.len() != self.columns
            || input.iter().any(|value| !value.is_finite())
            || self.packed_len() != elements.div_ceil(2)
            || self.scale_len() != elements.div_ceil(self.group_size)
        {
            return Err(CoreError::Model(
                "Corrupt Q4 matrix or projection buffers".into(),
            ));
        }
        if self.metal.is_some() {
            let mut output = Vec::new();
            output.try_reserve_exact(self.rows).map_err(|_| {
                CoreError::Model("Q4 projection output allocation was denied".into())
            })?;
            output.resize(self.rows, 0.0);
            self.project_into(input, &mut output)?;
            return Ok(output);
        }
        sage_kernels::project_q4(
            &self.packed_signed_values,
            &self.scales,
            input,
            self.rows,
            self.columns,
            self.group_size,
        )
        .map_err(|error| CoreError::Model(format!("Sage Q4 CPU projection failed: {error}")))
    }

    /// Project the trained Q4 matrix into caller-owned storage. CPU execution
    /// writes through Sage's NEON kernel; the evaluation-only Metal backend
    /// reuses a per-matrix I/O workspace and zeroes activation buffers on exit.
    pub fn project_into(&self, input: &[f32], output: &mut [f32]) -> CoreResult<()> {
        let elements = self
            .rows
            .checked_mul(self.columns)
            .ok_or_else(|| CoreError::Model("Q4 matrix shape overflow".into()))?;
        if self.group_size == 0
            || output.len() != self.rows
            || input.len() != self.columns
            || input.iter().any(|value| !value.is_finite())
            || self.packed_len() != elements.div_ceil(2)
            || self.scale_len() != elements.div_ceil(self.group_size)
        {
            output.fill(0.0);
            return Err(CoreError::Model(
                "Corrupt Q4 matrix or projection buffers".into(),
            ));
        }
        if let Some(metal) = &self.metal {
            let mut workspace = metal
                .workspace
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            return metal
                .context
                .project_q4_into(
                    &metal.packed_signed_values,
                    &metal.scales,
                    self.rows,
                    self.columns,
                    self.group_size,
                    input,
                    output,
                    &mut workspace,
                )
                .map_err(|error| {
                    output.fill(0.0);
                    CoreError::Model(format!("Sage Metal Q4 projection failed: {error}"))
                });
        }
        sage_kernels::project_q4_into(
            &self.packed_signed_values,
            &self.scales,
            input,
            self.rows,
            self.columns,
            self.group_size,
            output,
        )
        .map_err(|error| CoreError::Model(format!("Sage Q4 CPU projection failed: {error}")))
    }

    /// Project a bounded row-major activation batch while reusing each Q4
    /// weight across a tile of inputs. `scratch` is temporary transposed
    /// activation storage and is cleared before returning.
    pub fn project_batch_into(
        &self,
        input: &[f32],
        batch_size: usize,
        scratch: &mut [f32],
        output: &mut [f32],
    ) -> CoreResult<()> {
        let elements = self.rows.checked_mul(self.columns);
        let input_elements = batch_size.checked_mul(self.columns);
        let output_elements = batch_size.checked_mul(self.rows);
        let scratch_elements = self
            .columns
            .checked_mul(batch_size.min(sage_kernels::Q4_BATCH_TILE_SIZE));
        let valid = elements.is_some_and(|count| {
            self.group_size > 0
                && self.packed_len() == count.div_ceil(2)
                && self.scale_len() == count.div_ceil(self.group_size)
        }) && input_elements == Some(input.len())
            && input_elements.is_some_and(|count| count <= sage_kernels::Q4_BATCH_MAX_IO_ELEMENTS)
            && output_elements == Some(output.len())
            && output_elements.is_some_and(|count| count <= sage_kernels::Q4_BATCH_MAX_IO_ELEMENTS)
            && scratch_elements.is_some_and(|count| {
                count <= sage_kernels::Q4_BATCH_MAX_SCRATCH_ELEMENTS && count <= scratch.len()
            })
            && batch_size > 0
            && batch_size <= sage_kernels::Q4_BATCH_MAX_SIZE
            && input.iter().all(|value| value.is_finite());
        if !valid {
            scratch.fill(0.0);
            output.fill(0.0);
            return Err(CoreError::Model(
                "Corrupt Q4 batch matrix or projection buffers".into(),
            ));
        }

        if self.metal.is_some() {
            scratch.fill(0.0);
            for batch in 0..batch_size {
                let input_start = batch * self.columns;
                let output_start = batch * self.rows;
                if let Err(error) = self.project_into(
                    &input[input_start..input_start + self.columns],
                    &mut output[output_start..output_start + self.rows],
                ) {
                    output.fill(0.0);
                    scratch.fill(0.0);
                    return Err(error);
                }
            }
            return Ok(());
        }

        sage_kernels::project_q4_batch_into(
            &self.packed_signed_values,
            &self.scales,
            input,
            self.rows,
            self.columns,
            self.group_size,
            batch_size,
            scratch,
            output,
        )
        .map_err(|error| CoreError::Model(format!("Sage Q4 batch projection failed: {error}")))
    }

    /// Project only selected matrix rows into reusable storage. The CPU path
    /// uses Sage's bounded scalar Q4 kernel; shared Metal weights remain
    /// readable for this sparse output-head route without projecting the full
    /// vocabulary.
    pub fn project_selected_rows_into(
        &self,
        input: &[f32],
        selected_rows: &[usize],
        output: &mut [f32],
    ) -> CoreResult<()> {
        let elements = self
            .rows
            .checked_mul(self.columns)
            .ok_or_else(|| CoreError::Model("Q4 matrix shape overflow".into()))?;
        if self.group_size == 0
            || input.len() != self.columns
            || output.len() != selected_rows.len()
            || selected_rows.len() > self.rows
            || input.iter().any(|value| !value.is_finite())
            || selected_rows.iter().any(|row| *row >= self.rows)
            || self.packed_len() != elements.div_ceil(2)
            || self.scale_len() != elements.div_ceil(self.group_size)
        {
            output.fill(0.0);
            return Err(CoreError::Model(
                "Corrupt Q4 matrix or selected-row projection buffers".into(),
            ));
        }
        if self.metal.is_none() {
            return sage_kernels::project_q4_selected_into(
                &self.packed_signed_values,
                &self.scales,
                input,
                self.rows,
                self.columns,
                self.group_size,
                selected_rows,
                output,
            )
            .map_err(|error| {
                CoreError::Model(format!("Sage selected Q4 CPU projection failed: {error}"))
            });
        }

        let result =
            (|| {
                let metal = self
                    .metal
                    .as_ref()
                    .ok_or_else(|| CoreError::Model("Q4 Metal storage is unavailable".into()))?;
                let packed = metal.packed_signed_values.as_bytes().ok_or_else(|| {
                    CoreError::Model("Q4 Metal weight bytes are unavailable".into())
                })?;
                let scales = metal.scales.as_bytes().ok_or_else(|| {
                    CoreError::Model("Q4 Metal scale bytes are unavailable".into())
                })?;
                for (output_index, row) in selected_rows.iter().copied().enumerate() {
                    let row_start = row.checked_mul(self.columns).ok_or_else(|| {
                        CoreError::Model("Q4 selected-row offset overflow".into())
                    })?;
                    let mut sum = 0.0f64;
                    let mut previous_scale_index = usize::MAX;
                    let mut scale = 1.0f32;
                    for (column, activation) in input.iter().copied().enumerate() {
                        let index = row_start + column;
                        let scale_index = index / self.group_size;
                        if scale_index != previous_scale_index {
                            let scale_offset = scale_index
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    CoreError::Model("Q4 selected-row scale offset overflow".into())
                                })?;
                            let scale_bytes = scales
                                .get(scale_offset..scale_offset + std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    CoreError::Model("Q4 selected-row scale is unavailable".into())
                                })?;
                            scale = f32::from_ne_bytes(scale_bytes.try_into().map_err(|_| {
                                CoreError::Model("Q4 selected-row scale is truncated".into())
                            })?);
                            previous_scale_index = scale_index;
                        }
                        let byte = *packed.get(index / 2).ok_or_else(|| {
                            CoreError::Model("Q4 selected-row weight is unavailable".into())
                        })?;
                        let nibble = if index.is_multiple_of(2) {
                            byte & 0x0f
                        } else {
                            byte >> 4
                        };
                        let weight = f64::from((i32::from(nibble) - 8) as f32 * scale);
                        sum += weight * f64::from(activation);
                    }
                    let projected = sum as f32;
                    if !projected.is_finite() {
                        return Err(CoreError::Model(
                            "Selected Q4 projection result is non-finite".into(),
                        ));
                    }
                    output[output_index] = projected;
                }
                Ok(())
            })();
        if result.is_err() {
            output.fill(0.0);
        }
        result
    }

    /// Double-precision scalar reference used to qualify architecture-specific
    /// Q4 paths. Product/evaluation generation uses `project` above.
    #[cfg(test)]
    fn project_scalar(&self, input: &[f32]) -> CoreResult<Vec<f32>> {
        let elements = self
            .rows
            .checked_mul(self.columns)
            .ok_or_else(|| CoreError::Model("Q4 matrix shape overflow".into()))?;
        if self.group_size == 0
            || input.len() != self.columns
            || input.iter().any(|value| !value.is_finite())
            || self.packed_len() != elements.div_ceil(2)
            || self.scale_len() != elements.div_ceil(self.group_size)
        {
            return Err(CoreError::Model(
                "Corrupt Q4 matrix or projection input".into(),
            ));
        }
        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let mut sum = 0.0f64;
            for (column, input_value) in input.iter().enumerate() {
                let index = row * self.columns + column;
                let byte = self.packed_signed_values[index / 2];
                let nibble = if index.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let signed = i16::from(nibble) - 8;
                let scale = self.scales[index / self.group_size];
                let weight = f64::from(signed as f32 * scale);
                sum += weight * f64::from(*input_value);
            }
            let projected = sum as f32;
            if !projected.is_finite() {
                return Err(CoreError::Model(
                    "Q4 projection result is non-finite".into(),
                ));
            }
            output.push(projected);
        }
        Ok(output)
    }

    /// Read one dequantized row, used for token embedding lookup. Constructors
    /// validate finite positive scales and exact packed lengths once; reads
    /// check the selected row without rescanning every model scale.
    pub fn row(&self, row: usize) -> CoreResult<Vec<f32>> {
        let elements = self
            .rows
            .checked_mul(self.columns)
            .ok_or_else(|| CoreError::Model("Q4 matrix shape overflow".into()))?;
        if row >= self.rows
            || self.group_size == 0
            || self.packed_len() != elements.div_ceil(2)
            || self.scale_len() != elements.div_ceil(self.group_size)
        {
            return Err(CoreError::Model(
                "Corrupt Q4 matrix or embedding row index".into(),
            ));
        }
        let start = row
            .checked_mul(self.columns)
            .ok_or_else(|| CoreError::Model("Q4 row offset overflow".into()))?;
        let end = start
            .checked_add(self.columns)
            .ok_or_else(|| CoreError::Model("Q4 row end overflow".into()))?;
        let packed_offset = start / 2;
        let scale_offset = start / self.group_size;
        let metal_windows = self
            .metal
            .as_ref()
            .map(|metal| {
                let packed_end = end.div_ceil(2);
                let scale_end = end.div_ceil(self.group_size);
                let packed = metal
                    .packed_signed_values
                    .copy_range(packed_offset, packed_end - packed_offset)
                    .ok_or_else(|| {
                        CoreError::Model("Metal packed row read is out of bounds".into())
                    })?;
                let scale_byte_offset = scale_offset
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| CoreError::Model("Metal scale offset overflow".into()))?;
                let scale_byte_length = (scale_end - scale_offset)
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| CoreError::Model("Metal scale length overflow".into()))?;
                let scales = metal
                    .scales
                    .copy_range(scale_byte_offset, scale_byte_length)
                    .ok_or_else(|| {
                        CoreError::Model("Metal scale row read is out of bounds".into())
                    })?;
                Ok::<_, CoreError>((packed, scales))
            })
            .transpose()?;
        let mut output = Vec::with_capacity(self.columns);
        for column in 0..self.columns {
            let index = start + column;
            let byte = if let Some((packed, _)) = &metal_windows {
                *packed
                    .get(index / 2 - packed_offset)
                    .ok_or_else(|| CoreError::Model("Q4 packed row is incomplete".into()))?
            } else {
                self.packed_at(index)
                    .ok_or_else(|| CoreError::Model("Q4 packed storage is incomplete".into()))?
            };
            let nibble = if index.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            };
            let signed = i16::from(nibble) - 8;
            let scale = if let Some((_, scales)) = &metal_windows {
                let offset = (index / self.group_size - scale_offset)
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| CoreError::Model("Q4 row scale offset overflow".into()))?;
                let raw = scales
                    .get(offset..offset + std::mem::size_of::<f32>())
                    .ok_or_else(|| CoreError::Model("Q4 row scale is incomplete".into()))?;
                f32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]])
            } else {
                self.scale_at(index / self.group_size)
                    .ok_or_else(|| CoreError::Model("Q4 scale storage is incomplete".into()))?
            };
            let value = signed as f32 * scale;
            if !value.is_finite() {
                return Err(CoreError::Model(
                    "Q4 embedding row contains a non-finite value".into(),
                ));
            }
            output.push(value);
        }
        Ok(output)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Move immutable Q4 weights and scales into macOS shared Metal storage.
    /// The Metal buffers remain CPU-readable for the reference path, so this
    /// does not keep a second copy of checkpoint weights resident.
    pub fn move_to_metal(&mut self) -> CoreResult<()> {
        if self.metal.is_some() {
            return Ok(());
        }
        let context = MetalContext::shared().map_err(|error| {
            CoreError::Model(format!("Sage Metal backend is unavailable: {error}"))
        })?;
        if self.packed_signed_values.is_empty() || self.scales.is_empty() {
            return Err(CoreError::Model(
                "Cannot move an empty Q4 matrix to Metal".into(),
            ));
        }
        let packed_signed_values = Arc::new(context.buffer(&self.packed_signed_values).map_err(
            |error| CoreError::Model(format!("Metal weight allocation failed: {error}")),
        )?);
        let scale_bytes_len = self
            .scales
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CoreError::Model("Q4 scale byte length overflow".into()))?;
        let mut scale_bytes = Zeroizing::new(Vec::with_capacity(scale_bytes_len));
        for scale in &self.scales {
            scale_bytes.extend_from_slice(&scale.to_ne_bytes());
        }
        let scales = Arc::new(context.buffer(&scale_bytes).map_err(|error| {
            CoreError::Model(format!("Metal scale allocation failed: {error}"))
        })?);
        scale_bytes.zeroize();
        let workspace = context
            .q4_workspace(self.rows, self.columns)
            .map_err(|error| {
                CoreError::Model(format!(
                    "Metal activation workspace allocation failed: {error}"
                ))
            })?;

        self.packed_signed_values.zeroize();
        self.packed_signed_values = Vec::new();
        self.scales.zeroize();
        self.scales = Vec::new();
        self.metal = Some(MetalQ4Storage {
            context,
            scales,
            packed_signed_values,
            workspace: Arc::new(Mutex::new(workspace)),
        });
        Ok(())
    }

    fn packed_len(&self) -> usize {
        self.metal
            .as_ref()
            .map_or(self.packed_signed_values.len(), |metal| {
                metal.packed_signed_values.byte_len()
            })
    }

    fn scale_len(&self) -> usize {
        self.metal.as_ref().map_or(self.scales.len(), |metal| {
            metal.scales.byte_len() / std::mem::size_of::<f32>()
        })
    }

    fn packed_at(&self, index: usize) -> Option<u8> {
        let byte_index = index / 2;
        if let Some(metal) = &self.metal {
            metal.packed_signed_values.byte(byte_index)
        } else {
            self.packed_signed_values.get(byte_index).copied()
        }
    }

    fn scale_at(&self, index: usize) -> Option<f32> {
        if let Some(metal) = &self.metal {
            metal.scales.f32(index)
        } else {
            self.scales.get(index).copied()
        }
    }
}

pub use sage_model_package::{bf16_to_f32, f16_to_f32};

#[cfg(test)]
mod tests {
    use super::{
        ATTENTION_SCORE_BLOCK_SIZE, CausalDepthwiseConvState, GatedDeltaState,
        GroupedQueryAttentionScratch, KvCache, QuantizedQ4Builder, QuantizedQ4Matrix,
        grouped_query_attention_into, prepare_qwen35_mrope_angles, qwen35_gated_delta_coefficients,
        qwen35_mrope_denominators, rms_norm_gated, rms_norm_zero_centered_into,
        rotary_qwen35_mrope_partial, rotary_qwen35_mrope_with_angles, rotary_qwen35_partial,
        rotary_qwen35_vision_axial,
    };

    #[cfg(target_os = "macos")]
    fn project_q4_with_fresh_workspace(
        matrix: &QuantizedQ4Matrix,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let metal = matrix
            .metal
            .as_ref()
            .ok_or_else(|| "Q4 matrix has no Metal storage".to_owned())?;
        let mut workspace = metal.context.q4_workspace(matrix.rows, matrix.columns)?;
        metal.context.project_q4_into(
            &metal.packed_signed_values,
            &metal.scales,
            matrix.rows,
            matrix.columns,
            matrix.group_size,
            input,
            output,
            &mut workspace,
        )
    }

    #[test]
    fn grouped_query_attention_reuses_bounded_workspace_and_matches_reference() {
        let mut cache = KvCache::new(2, 3, 8).expect("bounded KV cache");
        let mut reference_keys = Vec::new();
        let mut reference_values = Vec::new();
        for position in 0..5 {
            let key = (0..6)
                .map(|index| ((position * 7 + index * 3) as f32 * 0.13).sin())
                .collect::<Vec<_>>();
            let value = (0..6)
                .map(|index| ((position * 5 + index * 2) as f32 * 0.17).cos())
                .collect::<Vec<_>>();
            reference_keys.extend_from_slice(&key);
            reference_values.extend_from_slice(&value);
            cache.append(&key, &value).expect("cache row");
        }
        let query = [
            0.2, -0.4, 0.7, 0.9, -0.1, 0.3, -0.8, 0.5, 0.6, 0.1, 0.4, -0.2,
        ];
        let mut scratch =
            GroupedQueryAttentionScratch::new(4, 2, 3, 8).expect("bounded attention scratch");
        let output_address = scratch.output.as_ptr();
        let score_address = scratch.scores.as_ptr();

        grouped_query_attention_into(&query, 4, &cache, &mut scratch)
            .expect("first attention pass");
        let first = scratch.output.clone();

        let query_heads = 4;
        let head_dimension = 3;
        let key_value_heads = 2;
        let group_size = query_heads / key_value_heads;
        let scale = (head_dimension as f64).sqrt().recip();
        let mut expected = vec![0.0f64; query_heads * head_dimension];
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let query_start = query_head * head_dimension;
            let mut scores = Vec::new();
            for position in 0..cache.context_length() {
                let row_start = position * head_dimension;
                let score = query[query_start..query_start + head_dimension]
                    .iter()
                    .zip(&cache.keys[kv_head][row_start..row_start + head_dimension])
                    .map(|(left, right)| {
                        f64::from(*left) * f64::from(sage_kernels::f16_bits_to_f32(*right))
                    })
                    .sum::<f64>()
                    * scale;
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denominator = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .sum::<f64>();
            for (position, score) in scores.iter().enumerate() {
                let weight = (*score - maximum).exp() / denominator;
                let row_start = position * head_dimension;
                for dimension in 0..head_dimension {
                    expected[query_start + dimension] += weight
                        * f64::from(sage_kernels::f16_bits_to_f32(
                            cache.values[kv_head][row_start + dimension],
                        ));
                }
            }
        }
        for (observed, expected) in first.iter().zip(expected) {
            assert!((f64::from(*observed) - expected).abs() < 5.0e-5);
        }

        let mut full_precision_output = vec![0.0f64; query.len()];
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let query_start = query_head * head_dimension;
            let mut scores = Vec::with_capacity(cache.context_length());
            for position in 0..cache.context_length() {
                let row_start =
                    position * key_value_heads * head_dimension + kv_head * head_dimension;
                scores.push(
                    query[query_start..query_start + head_dimension]
                        .iter()
                        .zip(&reference_keys[row_start..row_start + head_dimension])
                        .map(|(left, right)| f64::from(*left) * f64::from(*right))
                        .sum::<f64>()
                        * scale,
                );
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denominator = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .sum::<f64>();
            for (position, score) in scores.iter().enumerate() {
                let weight = (*score - maximum).exp() / denominator;
                let row_start =
                    position * key_value_heads * head_dimension + kv_head * head_dimension;
                for dimension in 0..head_dimension {
                    full_precision_output[query_start + dimension] +=
                        weight * f64::from(reference_values[row_start + dimension]);
                }
            }
        }
        let maximum_binary16_error = first
            .iter()
            .zip(full_precision_output)
            .map(|(observed, expected)| (f64::from(*observed) - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(maximum_binary16_error < 5.0e-4);

        let changed_query = query.map(|value| -value);
        grouped_query_attention_into(&changed_query, 4, &cache, &mut scratch)
            .expect("reused attention pass");
        assert_eq!(scratch.output.as_ptr(), output_address);
        assert_eq!(scratch.scores.as_ptr(), score_address);
        assert_ne!(scratch.output, first);
        assert!(scratch.scores.iter().all(|score| *score == 0.0));

        let invalid_query = [f32::NAN; 12];
        assert!(grouped_query_attention_into(&invalid_query, 4, &cache, &mut scratch).is_err());
        assert!(scratch.output.iter().all(|value| *value == 0.0));
        assert!(scratch.scores.iter().all(|score| *score == 0.0));
    }

    #[test]
    #[ignore = "release-only grouped-query attention workspace measurement"]
    fn grouped_query_attention_workspace_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        let query_heads = 16;
        let key_value_heads = 4;
        let head_dimension = 64;
        let context = 2048;
        let mut cache = KvCache::new(key_value_heads, head_dimension, context)
            .expect("bounded benchmark KV cache");
        cache
            .reserve_positions(context)
            .expect("reserve benchmark KV cache");
        for position in 0..context {
            let key = (0..key_value_heads * head_dimension)
                .map(|index| ((position * 13 + index * 7) as f32 * 0.0013).sin())
                .collect::<Vec<_>>();
            let value = (0..key_value_heads * head_dimension)
                .map(|index| ((position * 5 + index * 11) as f32 * 0.0017).cos())
                .collect::<Vec<_>>();
            cache.append(&key, &value).expect("benchmark KV row");
        }
        let query = (0..query_heads * head_dimension)
            .map(|index| (index as f32 * 0.013).sin())
            .collect::<Vec<_>>();
        let mut reused = GroupedQueryAttentionScratch::new(
            query_heads,
            key_value_heads,
            head_dimension,
            context,
        )
        .expect("reusable benchmark workspace");

        let run_fresh_workspace = || {
            let started = Instant::now();
            let mut fresh = GroupedQueryAttentionScratch::new(
                query_heads,
                key_value_heads,
                head_dimension,
                context,
            )
            .expect("fresh benchmark workspace");
            grouped_query_attention_into(&query, query_heads, &cache, &mut fresh)
                .expect("fresh benchmark attention");
            black_box(fresh.output_mut());
            drop(fresh);
            started.elapsed()
        };
        let mut run_reused_workspace = || {
            let started = Instant::now();
            grouped_query_attention_into(&query, query_heads, &cache, &mut reused)
                .expect("reused benchmark attention");
            black_box(reused.output_mut());
            reused.clear_output();
            started.elapsed()
        };
        for _ in 0..10 {
            black_box(run_fresh_workspace());
            black_box(run_reused_workspace());
        }

        let mut fresh_times = Vec::with_capacity(101);
        let mut reused_times = Vec::with_capacity(101);
        for sample in 0_usize..101 {
            if sample.is_multiple_of(2) {
                fresh_times.push(run_fresh_workspace());
                reused_times.push(run_reused_workspace());
            } else {
                reused_times.push(run_reused_workspace());
                fresh_times.push(run_fresh_workspace());
            }
        }
        fresh_times.sort_unstable();
        reused_times.sort_unstable();
        eprintln!(
            "qwen-grouped-query-workspace context={context} query_heads={query_heads} head_dim={head_dimension} samples=101 fresh_p50_us={} fresh_p95_us={} reused_p50_us={} reused_p95_us={}",
            fresh_times[50].as_nanos() / 1_000,
            fresh_times[95].as_nanos() / 1_000,
            reused_times[50].as_nanos() / 1_000,
            reused_times[95].as_nanos() / 1_000,
        );
    }

    #[test]
    #[ignore = "release-only full-score versus blockwise attention measurement"]
    fn grouped_query_attention_blockwise_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        fn run_full_score(
            query: &[f32],
            query_heads: usize,
            cache: &KvCache,
            scores: &mut [f64],
            output: &mut [f32],
        ) -> std::time::Duration {
            let started = Instant::now();
            let context = cache.context_length();
            let group_size = query_heads / cache.key_value_heads;
            let scale = (cache.head_dimension as f64).sqrt().recip();
            for query_head in 0..query_heads {
                let key_value_head = query_head / group_size;
                let query_start = query_head * cache.head_dimension;
                let head_scores = &mut scores[..context];
                sage_kernels::dot_rows_f16(
                    &query[query_start..query_start + cache.head_dimension],
                    &cache.keys[key_value_head],
                    cache.head_dimension,
                    0,
                    head_scores,
                )
                .expect("full-score QK kernel");
                let mut maximum = f64::NEG_INFINITY;
                for score in head_scores.iter_mut() {
                    *score *= scale;
                    maximum = maximum.max(*score);
                }
                let mut denominator = 0.0f64;
                for score in head_scores.iter_mut() {
                    *score = (*score - maximum).exp();
                    denominator += *score;
                }
                for score in head_scores.iter_mut() {
                    *score /= denominator;
                }
                let output_start = query_head * cache.head_dimension;
                sage_kernels::weighted_sum_rows_f16_into(
                    &cache.values[key_value_head],
                    head_scores,
                    context,
                    cache.head_dimension,
                    0,
                    &mut output[output_start..output_start + cache.head_dimension],
                )
                .expect("full-score WV kernel");
            }
            black_box(output);
            started.elapsed()
        }

        fn run_blockwise(
            query: &[f32],
            query_heads: usize,
            cache: &KvCache,
            scratch: &mut GroupedQueryAttentionScratch,
        ) -> std::time::Duration {
            let started = Instant::now();
            grouped_query_attention_into(query, query_heads, cache, scratch)
                .expect("blockwise grouped-query attention");
            black_box(scratch.output_mut());
            started.elapsed()
        }

        let query_heads = 16;
        let key_value_heads = 4;
        let head_dimension = 256;
        for context in [2_048, 8_192] {
            let mut cache = KvCache::new(key_value_heads, head_dimension, context)
                .expect("bounded benchmark KV cache");
            cache
                .reserve_positions(context)
                .expect("reserve benchmark KV cache");
            for position in 0..context {
                let key = (0..key_value_heads * head_dimension)
                    .map(|index| ((position * 13 + index * 7) as f32 * 0.0013).sin())
                    .collect::<Vec<_>>();
                let value = (0..key_value_heads * head_dimension)
                    .map(|index| ((position * 5 + index * 11) as f32 * 0.0017).cos())
                    .collect::<Vec<_>>();
                cache.append(&key, &value).expect("benchmark KV row");
            }
            let query = (0..query_heads * head_dimension)
                .map(|index| (index as f32 * 0.013).sin())
                .collect::<Vec<_>>();
            let mut full_scores = vec![0.0f64; context];
            let mut full_output = vec![0.0f32; query.len()];
            let mut blockwise = GroupedQueryAttentionScratch::new(
                query_heads,
                key_value_heads,
                head_dimension,
                context,
            )
            .expect("bounded blockwise benchmark workspace");
            black_box(run_full_score(
                &query,
                query_heads,
                &cache,
                &mut full_scores,
                &mut full_output,
            ));
            black_box(run_blockwise(&query, query_heads, &cache, &mut blockwise));
            let maximum_output_error = full_output
                .iter()
                .zip(blockwise.output.iter())
                .map(|(left, right)| f64::from((*left - *right).abs()))
                .fold(0.0f64, f64::max);
            assert!(maximum_output_error < 5.0e-4);

            for _ in 0..10 {
                black_box(run_full_score(
                    &query,
                    query_heads,
                    &cache,
                    &mut full_scores,
                    &mut full_output,
                ));
                black_box(run_blockwise(&query, query_heads, &cache, &mut blockwise));
            }
            let mut full_times = Vec::with_capacity(101);
            let mut blockwise_times = Vec::with_capacity(101);
            for sample in 0_usize..101 {
                if sample.is_multiple_of(2) {
                    full_times.push(run_full_score(
                        &query,
                        query_heads,
                        &cache,
                        &mut full_scores,
                        &mut full_output,
                    ));
                    blockwise_times.push(run_blockwise(
                        &query,
                        query_heads,
                        &cache,
                        &mut blockwise,
                    ));
                } else {
                    blockwise_times.push(run_blockwise(
                        &query,
                        query_heads,
                        &cache,
                        &mut blockwise,
                    ));
                    full_times.push(run_full_score(
                        &query,
                        query_heads,
                        &cache,
                        &mut full_scores,
                        &mut full_output,
                    ));
                }
            }
            full_times.sort_unstable();
            blockwise_times.sort_unstable();
            eprintln!(
                "qwen-grouped-kv-attention context={context} query_heads={query_heads} kv_heads={key_value_heads} head_dim={head_dimension} score_block={} samples=101 separate_p50_us={} separate_p95_us={} grouped_p50_us={} grouped_p95_us={} max_output_error={maximum_output_error:.8}",
                ATTENTION_SCORE_BLOCK_SIZE,
                full_times[50].as_nanos() / 1_000,
                full_times[95].as_nanos() / 1_000,
                blockwise_times[50].as_nanos() / 1_000,
                blockwise_times[95].as_nanos() / 1_000,
            );
        }
    }

    #[test]
    #[ignore = "release-only interleaved versus head-major KV bandwidth measurement"]
    fn grouped_query_kv_layout_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        struct KernelFixture<'a> {
            query: &'a [f32],
            query_heads: usize,
            key_value_heads: usize,
            head_dimension: usize,
            context: usize,
            keys: &'a [u16],
            values: &'a [u16],
            row_stride: usize,
            head_stride: usize,
            head_plane_stride: usize,
            column_stride: usize,
            weights: &'a [f64],
        }

        fn run_kernels(
            fixture: &KernelFixture<'_>,
            scores: &mut [f64],
            output: &mut [f32],
        ) -> std::time::Duration {
            let started = Instant::now();
            let group_size = fixture.query_heads / fixture.key_value_heads;
            for kv_head in 0..fixture.key_value_heads {
                let query_start = kv_head * group_size * fixture.head_dimension;
                let query_end = query_start + group_size * fixture.head_dimension;
                let scores_per_group = group_size * fixture.context;
                let score_start = kv_head * scores_per_group;
                let score_end = score_start + scores_per_group;
                sage_kernels::dot_rows_f16_grouped(
                    &fixture.query[query_start..query_end],
                    group_size,
                    &fixture.keys[kv_head * fixture.head_plane_stride
                        ..kv_head * fixture.head_plane_stride + fixture.head_stride],
                    fixture.context,
                    fixture.row_stride,
                    kv_head * fixture.column_stride,
                    &mut scores[score_start..score_end],
                )
                .expect("layout benchmark QK kernel");
                let output_start = kv_head * group_size * fixture.head_dimension;
                let output_end = output_start + group_size * fixture.head_dimension;
                sage_kernels::weighted_sum_rows_f16_grouped_into(
                    &fixture.values[kv_head * fixture.head_plane_stride
                        ..kv_head * fixture.head_plane_stride + fixture.head_stride],
                    &fixture.weights[..scores_per_group],
                    group_size,
                    fixture.context,
                    fixture.row_stride,
                    kv_head * fixture.column_stride,
                    &mut output[output_start..output_end],
                )
                .expect("layout benchmark WV kernel");
            }
            black_box(output);
            started.elapsed()
        }

        let query_heads = 16;
        let key_value_heads = 4;
        let head_dimension = 256;
        let group_size = query_heads / key_value_heads;
        for context in [2_048, 8_192] {
            let mut cache = KvCache::new(key_value_heads, head_dimension, context)
                .expect("bounded benchmark KV cache");
            cache
                .reserve_positions(context)
                .expect("reserve benchmark KV cache");
            for position in 0..context {
                let key = (0..key_value_heads * head_dimension)
                    .map(|index| ((position * 13 + index * 7) as f32 * 0.0013).sin())
                    .collect::<Vec<_>>();
                let value = (0..key_value_heads * head_dimension)
                    .map(|index| ((position * 5 + index * 11) as f32 * 0.0017).cos())
                    .collect::<Vec<_>>();
                cache.append(&key, &value).expect("benchmark KV row");
            }

            let row_elements = key_value_heads * head_dimension;
            let head_elements = context * head_dimension;
            let mut head_major_keys = Vec::with_capacity(key_value_heads * head_elements);
            let mut head_major_values = Vec::with_capacity(key_value_heads * head_elements);
            for kv_head in 0..key_value_heads {
                head_major_keys.extend_from_slice(
                    cache
                        .keys_for_head(kv_head)
                        .expect("known benchmark KV head"),
                );
                head_major_values.extend_from_slice(
                    cache
                        .values_for_head(kv_head)
                        .expect("known benchmark KV head"),
                );
            }
            let mut interleaved_keys = Vec::with_capacity(key_value_heads * head_elements);
            let mut interleaved_values = Vec::with_capacity(key_value_heads * head_elements);
            for position in 0..context {
                for kv_head in 0..key_value_heads {
                    let bank_start = position * head_dimension;
                    interleaved_keys.extend_from_slice(
                        &cache
                            .keys_for_head(kv_head)
                            .expect("known benchmark KV head")
                            [bank_start..bank_start + head_dimension],
                    );
                    interleaved_values.extend_from_slice(
                        &cache
                            .values_for_head(kv_head)
                            .expect("known benchmark KV head")
                            [bank_start..bank_start + head_dimension],
                    );
                }
            }

            let query = (0..query_heads * head_dimension)
                .map(|index| (index as f32 * 0.013).sin())
                .collect::<Vec<_>>();
            let mut weights = vec![0.0f64; group_size * context];
            for head in 0..group_size {
                let start = head * context;
                let denominator = (0..context)
                    .map(|position| ((position + head * 17) as f64 * 0.0003).exp())
                    .sum::<f64>();
                for position in 0..context {
                    weights[start + position] =
                        ((position + head * 17) as f64 * 0.0003).exp() / denominator;
                }
            }
            let interleaved = KernelFixture {
                query: &query,
                query_heads,
                key_value_heads,
                head_dimension,
                context,
                keys: &interleaved_keys,
                values: &interleaved_values,
                row_stride: row_elements,
                head_stride: context * row_elements,
                head_plane_stride: 0,
                column_stride: head_dimension,
                weights: &weights,
            };
            let head_major = KernelFixture {
                query: &query,
                query_heads,
                key_value_heads,
                head_dimension,
                context,
                keys: &head_major_keys,
                values: &head_major_values,
                row_stride: head_dimension,
                head_stride: head_elements,
                head_plane_stride: head_elements,
                column_stride: 0,
                weights: &weights,
            };
            let mut interleaved_scores = vec![0.0f64; query_heads * context];
            let mut head_major_scores = vec![0.0f64; query_heads * context];
            let mut interleaved_output = vec![0.0f32; query.len()];
            let mut head_major_output = vec![0.0f32; query.len()];

            black_box(run_kernels(
                &interleaved,
                &mut interleaved_scores,
                &mut interleaved_output,
            ));
            black_box(run_kernels(
                &head_major,
                &mut head_major_scores,
                &mut head_major_output,
            ));
            let maximum_output_error = interleaved_output
                .iter()
                .zip(&head_major_output)
                .map(|(left, right)| f64::from((*left - *right).abs()))
                .fold(0.0f64, f64::max);
            let maximum_score_error = interleaved_scores
                .iter()
                .zip(&head_major_scores)
                .map(|(left, right)| (*left - *right).abs())
                .fold(0.0f64, f64::max);
            assert!(maximum_output_error < 1.0e-6);
            assert!(maximum_score_error < 1.0e-6);

            let mut run_interleaved = || {
                run_kernels(
                    &interleaved,
                    &mut interleaved_scores,
                    &mut interleaved_output,
                )
            };
            let mut run_head_major =
                || run_kernels(&head_major, &mut head_major_scores, &mut head_major_output);
            for _ in 0..10 {
                black_box(run_interleaved());
                black_box(run_head_major());
            }
            let mut interleaved_times = Vec::with_capacity(101);
            let mut head_major_times = Vec::with_capacity(101);
            for sample in 0_usize..101 {
                if sample.is_multiple_of(2) {
                    interleaved_times.push(run_interleaved());
                    head_major_times.push(run_head_major());
                } else {
                    head_major_times.push(run_head_major());
                    interleaved_times.push(run_interleaved());
                }
            }
            interleaved_times.sort_unstable();
            head_major_times.sort_unstable();
            eprintln!(
                "qwen-kv-layout context={context} query_heads={query_heads} kv_heads={key_value_heads} head_dim={head_dimension} samples=101 interleaved_p50_us={} interleaved_p95_us={} head_major_p50_us={} head_major_p95_us={} max_score_error={maximum_score_error:.8} max_output_error={maximum_output_error:.8}",
                interleaved_times[50].as_nanos() / 1_000,
                interleaved_times[95].as_nanos() / 1_000,
                head_major_times[50].as_nanos() / 1_000,
                head_major_times[95].as_nanos() / 1_000,
            );
        }
    }

    #[test]
    fn kv_cache_reserves_only_observed_context_and_grows_geometrically() {
        let mut cache = KvCache::new(2, 3, 100).expect("bounded cache");
        assert_eq!(cache.capacity_positions(), 0);
        assert!(cache.reserve_positions(101).is_err());
        assert_eq!(cache.capacity_positions(), 0);

        let key = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let value = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
        for _ in 0..16 {
            cache.append(&key, &value).expect("valid cache row");
        }
        assert_eq!(cache.context_length(), 16);
        assert_eq!(cache.capacity_positions(), 16);
        cache
            .append(&key, &value)
            .expect("cache grows after its prefix");
        assert_eq!(cache.context_length(), 17);
        assert_eq!(cache.capacity_positions(), 32);
        assert_eq!(cache.keys_for_head(0).unwrap().len(), 17 * 3);
        assert_eq!(cache.values_for_head(1).unwrap().len(), 17 * 3);
        assert_eq!(
            &cache.keys_for_head(0).unwrap()[..3],
            &[1.0_f32, 2.0, 3.0].map(sage_kernels::f32_to_f16_bits)
        );
        assert_eq!(
            &cache.keys_for_head(1).unwrap()[..3],
            &[4.0_f32, 5.0, 6.0].map(sage_kernels::f32_to_f16_bits)
        );
        assert_eq!(
            &cache.values_for_head(0).unwrap()[..3],
            &[-1.0_f32, -2.0, -3.0].map(sage_kernels::f32_to_f16_bits)
        );
        assert_eq!(
            &cache.values_for_head(1).unwrap()[..3],
            &[-4.0_f32, -5.0, -6.0].map(sage_kernels::f32_to_f16_bits)
        );

        let before = cache.context_length();
        let out_of_range = [f32::MAX, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(cache.append(&out_of_range, &value).is_err());
        assert_eq!(cache.context_length(), before);

        cache.clear();
        assert_eq!(cache.context_length(), 0);
        assert!(cache.keys_for_head(0).unwrap().is_empty());
        assert!(cache.keys_for_head(1).unwrap().is_empty());
        assert!(cache.values_for_head(0).unwrap().is_empty());
        assert!(cache.values_for_head(1).unwrap().is_empty());
        assert_eq!(cache.capacity_positions(), 32);
    }

    #[test]
    fn streamed_q4_is_byte_identical_across_arbitrary_chunk_and_group_boundaries() {
        // 33 values force both an odd final nibble and a 17-element group
        // boundary that crosses packed-byte alignment.
        let values = (0..33)
            .map(|index| ((index as f32 - 16.0) * 0.17).sin())
            .collect::<Vec<_>>();
        let batch = QuantizedQ4Matrix::encode(3, 11, 17, &values).expect("batch Q4 fixture");
        let mut streamed = QuantizedQ4Builder::new(3, 11, 17, 33).expect("bounded Q4 builder");
        for chunk in [
            &values[..1],
            &values[1..16],
            &values[16..18],
            &values[18..32],
            &values[32..],
        ] {
            streamed.push(chunk).expect("valid streamed Q4 chunk");
        }
        let streamed = streamed.finish().expect("complete Q4 import");
        assert_eq!(streamed.scales, batch.scales);
        assert_eq!(streamed.packed_signed_values, batch.packed_signed_values);
        assert_eq!((streamed.rows(), streamed.columns()), (3, 11));
        let input = [0.5, -0.25, 1.0, 0.75, -0.5, 0.0, 0.2, -1.0, 0.1, 0.4, -0.3];
        assert_eq!(
            streamed.project(&input).unwrap(),
            batch.project(&input).unwrap()
        );
        assert_eq!(streamed.row(2).unwrap(), batch.row(2).unwrap());
    }

    #[test]
    fn q4_projection_matches_independent_reference_when_groups_cross_rows() {
        let rows = 5;
        let columns = 7;
        let group_size = 9;
        let values = (0..rows * columns)
            .map(|index| ((index as f32 - 13.0) * 0.29).sin())
            .collect::<Vec<_>>();
        let input = [0.5, -0.25, 1.0, 0.75, -0.5, 0.0, 0.2];
        let matrix = QuantizedQ4Matrix::encode(rows, columns, group_size, &values)
            .expect("valid cross-row grouped-Q4 matrix");

        let mut expected = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut sum = 0.0f64;
            for (column, activation) in input.iter().enumerate() {
                let index = row * columns + column;
                let packed = matrix.packed_signed_values[index / 2];
                let nibble = if index.is_multiple_of(2) {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                let signed = i16::from(nibble) - 8;
                let scale = matrix.scales[index / group_size];
                sum += f64::from(signed as f32 * scale) * f64::from(*activation);
            }
            expected.push(sum as f32);
        }

        let actual = matrix.project(&input).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            // NEON accumulates in f32 while this independent reference uses
            // f64, so require close agreement rather than identical bits.
            let tolerance = 2.0e-5 + expected.abs() * 2.0e-5;
            assert!((actual - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn streamed_q4_rejects_short_long_nonfinite_and_over_budget_weights() {
        let mut short = QuantizedQ4Builder::new(2, 16, 16, 32).expect("bounded builder");
        short.push(&[0.0; 31]).expect("partial source chunk");
        assert!(short.finish().is_err());

        let mut long = QuantizedQ4Builder::new(1, 16, 16, 16).expect("bounded builder");
        assert!(long.push(&[0.0; 17]).is_err());

        let mut nonfinite = QuantizedQ4Builder::new(1, 16, 16, 16).expect("bounded builder");
        assert!(nonfinite.push(&[f32::NAN]).is_err());

        assert!(QuantizedQ4Builder::new(2, 16, 16, 31).is_err());
        assert!(QuantizedQ4Builder::new(2, 16, 8, 32).is_err());
        assert!(QuantizedQ4Builder::new(100_000, 100_000, 16, usize::MAX).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal-capable Apple GPU"]
    fn metal_q4_projection_matches_the_scalar_reference_and_reuses_shared_weights() {
        let rows = 7;
        let columns = 31;
        let values = (0..rows * columns)
            .map(|index| ((index as f32 - 83.0) * 0.071).sin() * 0.9)
            .collect::<Vec<_>>();
        let mut matrix =
            QuantizedQ4Matrix::encode(rows, columns, 16, &values).expect("valid grouped-Q4 matrix");
        let input = (0..columns)
            .map(|index| ((index as f32 - 13.0) * 0.19).cos())
            .collect::<Vec<_>>();
        let expected = matrix.project(&input).expect("scalar reference projection");

        matrix
            .move_to_metal()
            .expect("move Q4 weights to shared Metal storage");
        assert!(matrix.packed_signed_values.is_empty());
        assert!(matrix.scales.is_empty());
        let actual = matrix.project(&input).expect("Metal Q4 projection");
        assert_eq!(actual.len(), expected.len());
        for (observed, reference) in actual.iter().zip(&expected) {
            let tolerance = 2.0e-4 + reference.abs() * 2.0e-4;
            assert!((observed - reference).abs() <= tolerance);
        }

        let selected_rows = [6, 0, 3];
        let mut selected = [0.0; 3];
        matrix
            .project_selected_rows_into(&input, &selected_rows, &mut selected)
            .expect("sparse projection from shared Metal storage");
        for (observed, row) in selected.iter().zip(selected_rows) {
            let reference = expected[row];
            let tolerance = 2.0e-4 + reference.abs() * 2.0e-4;
            assert!((observed - reference).abs() <= tolerance);
        }

        let expected_embedding = QuantizedQ4Matrix::encode(rows, columns, 16, &values)
            .expect("valid embedding matrix")
            .row(5)
            .expect("scalar embedding row");
        assert_eq!(
            matrix.row(5).expect("shared-storage embedding row"),
            expected_embedding
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a Metal-capable Apple GPU"]
    fn concurrent_metal_q4_projections_match_independent_scalar_results() {
        let rows = 9;
        let columns = 67;
        let values = (0..rows * columns)
            .map(|index| ((index as f32 - 117.0) * 0.037).sin() * 0.8)
            .collect::<Vec<_>>();
        let mut matrix =
            QuantizedQ4Matrix::encode(rows, columns, 32, &values).expect("valid grouped-Q4 matrix");
        let inputs = (0..4)
            .map(|request| {
                (0..columns)
                    .map(|column| ((column as f32 * 0.17) + request as f32 * 0.61).cos())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected = inputs
            .iter()
            .map(|input| matrix.project(input).expect("scalar Q4 reference"))
            .collect::<Vec<_>>();
        matrix.move_to_metal().expect("move weights to Metal");

        std::thread::scope(|scope| {
            let matrix = &matrix;
            let workers = inputs
                .iter()
                .map(|input| scope.spawn(move || matrix.project(input).expect("Metal projection")))
                .collect::<Vec<_>>();
            for (worker, reference) in workers.into_iter().zip(expected) {
                let actual = worker.join().expect("Metal projection worker");
                for (observed, expected) in actual.iter().zip(reference) {
                    let tolerance = 2.0e-4 + expected.abs() * 2.0e-4;
                    assert!((observed - expected).abs() <= tolerance);
                }
            }
        });
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "release-only CPU versus Metal Qwen hidden projection measurement"]
    fn qwen_hidden_q4_projection_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        let rows = 2_560;
        let columns = 2_560;
        let elements = rows * columns;
        let values = (0..elements)
            .map(|index| ((index % 251) as f32 - 125.0) / 251.0)
            .collect::<Vec<_>>();
        let cpu = QuantizedQ4Matrix::encode(rows, columns, 128, &values)
            .expect("encode Qwen-sized grouped-Q4 fixture");
        let mut metal = cpu.clone();
        metal
            .move_to_metal()
            .expect("move Qwen-sized fixture to Metal");
        let mut reused_output = vec![0.0; rows];
        let mut metal_reused_output = vec![0.0; rows];
        let input = (0..columns)
            .map(|index| ((index % 67) as f32 - 33.0) / 67.0)
            .collect::<Vec<_>>();
        let mut fresh_metal_output = vec![0.0; rows];

        let scalar_result = cpu.project_scalar(&input).expect("scalar Q4 reference");
        let cpu_result = cpu.project(&input).expect("SIMD CPU Q4 projection");
        cpu.project_into(&input, &mut reused_output)
            .expect("reused CPU Q4 projection");
        let metal_result = metal.project(&input).expect("Metal Q4 projection");
        for (observed, reference) in cpu_result.iter().zip(&scalar_result) {
            let tolerance = 1.0e-3 + reference.abs() * 1.0e-3;
            assert!((observed - reference).abs() <= tolerance);
        }
        for (observed, reference) in metal_result.iter().zip(&scalar_result) {
            let tolerance = 1.0e-3 + reference.abs() * 1.0e-3;
            assert!((observed - reference).abs() <= tolerance);
        }
        for (observed, reference) in reused_output.iter().zip(&scalar_result) {
            let tolerance = 1.0e-3 + reference.abs() * 1.0e-3;
            assert!((observed - reference).abs() <= tolerance);
        }

        for _ in 0..2 {
            black_box(
                cpu.project_scalar(black_box(&input))
                    .expect("warm scalar reference projection"),
            );
            black_box(
                cpu.project(black_box(&input))
                    .expect("warm SIMD CPU projection"),
            );
            cpu.project_into(black_box(&input), black_box(&mut reused_output))
                .expect("warm reused CPU Q4 projection");
            black_box(
                metal
                    .project(black_box(&input))
                    .expect("warm Metal projection"),
            );
            metal
                .project_into(black_box(&input), black_box(&mut metal_reused_output))
                .expect("warm reused Metal projection");
            project_q4_with_fresh_workspace(&metal, &input, &mut fresh_metal_output)
                .expect("warm fresh-workspace Metal projection");
        }
        let mut scalar_times = Vec::with_capacity(101);
        let mut cpu_times = Vec::with_capacity(101);
        let mut cpu_reused_times = Vec::with_capacity(101);
        let mut metal_fresh_workspace_times = Vec::with_capacity(101);
        let mut metal_reused_allocating_times = Vec::with_capacity(101);
        let mut metal_reused_times = Vec::with_capacity(101);
        for sample in 0_usize..101 {
            let start = Instant::now();
            black_box(
                cpu.project_scalar(black_box(&input))
                    .expect("scalar reference projection"),
            );
            scalar_times.push(start.elapsed());

            if sample.is_multiple_of(2) {
                let start = Instant::now();
                black_box(cpu.project(black_box(&input)).expect("SIMD CPU projection"));
                cpu_times.push(start.elapsed());

                let start = Instant::now();
                cpu.project_into(black_box(&input), black_box(&mut reused_output))
                    .expect("reused CPU Q4 projection");
                black_box(&reused_output);
                cpu_reused_times.push(start.elapsed());
            } else {
                let start = Instant::now();
                cpu.project_into(black_box(&input), black_box(&mut reused_output))
                    .expect("reused CPU Q4 projection");
                black_box(&reused_output);
                cpu_reused_times.push(start.elapsed());

                let start = Instant::now();
                black_box(cpu.project(black_box(&input)).expect("SIMD CPU projection"));
                cpu_times.push(start.elapsed());
            }

            if sample.is_multiple_of(2) {
                let start = Instant::now();
                project_q4_with_fresh_workspace(
                    &metal,
                    black_box(&input),
                    black_box(&mut fresh_metal_output),
                )
                .expect("fresh-workspace Metal projection");
                black_box(&fresh_metal_output);
                metal_fresh_workspace_times.push(start.elapsed());

                let start = Instant::now();
                black_box(
                    metal
                        .project(black_box(&input))
                        .expect("reused-workspace projection"),
                );
                metal_reused_allocating_times.push(start.elapsed());
            } else {
                let start = Instant::now();
                black_box(
                    metal
                        .project(black_box(&input))
                        .expect("reused-workspace projection"),
                );
                metal_reused_allocating_times.push(start.elapsed());

                let start = Instant::now();
                project_q4_with_fresh_workspace(
                    &metal,
                    black_box(&input),
                    black_box(&mut fresh_metal_output),
                )
                .expect("fresh-workspace Metal projection");
                black_box(&fresh_metal_output);
                metal_fresh_workspace_times.push(start.elapsed());
            }

            let start = Instant::now();
            metal
                .project_into(black_box(&input), black_box(&mut metal_reused_output))
                .expect("reused Metal projection");
            black_box(&metal_reused_output);
            metal_reused_times.push(start.elapsed());
        }
        scalar_times.sort_unstable();
        cpu_times.sort_unstable();
        cpu_reused_times.sort_unstable();
        metal_fresh_workspace_times.sort_unstable();
        metal_reused_allocating_times.sort_unstable();
        metal_reused_times.sort_unstable();
        eprintln!(
            "qwen-hidden-q4-projection release samples=101 scalar_p50_us={} scalar_p95_us={} cpu_allocating_p50_us={} cpu_allocating_p95_us={} cpu_reused_p50_us={} cpu_reused_p95_us={} metal_fresh_workspace_p50_us={} metal_fresh_workspace_p95_us={} metal_reused_workspace_allocating_p50_us={} metal_reused_workspace_allocating_p95_us={} metal_reused_workspace_into_p50_us={} metal_reused_workspace_into_p95_us={}",
            duration_micros(scalar_times[50]),
            duration_micros(scalar_times[95]),
            duration_micros(cpu_times[50]),
            duration_micros(cpu_times[95]),
            duration_micros(cpu_reused_times[50]),
            duration_micros(cpu_reused_times[95]),
            duration_micros(metal_fresh_workspace_times[50]),
            duration_micros(metal_fresh_workspace_times[95]),
            duration_micros(metal_reused_allocating_times[50]),
            duration_micros(metal_reused_allocating_times[95]),
            duration_micros(metal_reused_times[50]),
            duration_micros(metal_reused_times[95]),
        );
    }

    #[cfg(target_os = "macos")]
    fn qwen_sparse_output_head_fixture() -> (QuantizedQ4Matrix, Vec<f32>) {
        let rows = 248_320_usize;
        let columns = 2_560_usize;
        let group_size = 128_usize;
        let elements = rows * columns;
        let packed_signed_values = (0..elements.div_ceil(2))
            .map(|index| {
                let low = ((index * 7 + 3) % 15 + 1) as u8;
                let high = ((index * 11 + 5) % 15 + 1) as u8;
                low | (high << 4)
            })
            .collect();
        let scales = (0..elements.div_ceil(group_size))
            .map(|index| 0.01 + (index % 13) as f32 * 0.001)
            .collect();
        let matrix = QuantizedQ4Matrix {
            rows,
            columns,
            group_size,
            scales,
            packed_signed_values,
            metal: None,
        };
        let input = (0..columns)
            .map(|index| ((index % 67) as f32 - 33.0) / 67.0)
            .collect();
        (matrix, input)
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "release-only sparse versus dense Qwen tied-output projection measurement"]
    fn qwen_sparse_output_head_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        // Build directly in packed form; a batch f32 tensor would exceed 2 GiB.
        let (matrix, input) = qwen_sparse_output_head_fixture();
        let rows = matrix.rows();
        let columns = matrix.columns();
        let group_size = matrix.group_size;
        let selected_rows = (0..64).map(|index| index * 251).collect::<Vec<_>>();
        let mut dense_output = vec![0.0; rows];
        let mut sparse_output = vec![0.0; selected_rows.len()];

        matrix
            .project_into(&input, &mut dense_output)
            .expect("dense Q4 output-head projection");
        matrix
            .project_selected_rows_into(&input, &selected_rows, &mut sparse_output)
            .expect("sparse Q4 output-head projection");
        for (observed, row) in sparse_output.iter().zip(&selected_rows) {
            let reference = dense_output[*row];
            let tolerance = 1.0e-3 + reference.abs() * 1.0e-3;
            assert!((observed - reference).abs() <= tolerance);
        }

        for _ in 0..2 {
            matrix
                .project_into(black_box(&input), black_box(&mut dense_output))
                .expect("warm dense output-head projection");
            matrix
                .project_selected_rows_into(
                    black_box(&input),
                    black_box(&selected_rows),
                    black_box(&mut sparse_output),
                )
                .expect("warm sparse output-head projection");
        }
        let mut dense_times = Vec::with_capacity(101);
        let mut sparse_times = Vec::with_capacity(101);
        for sample in 0_usize..101 {
            if sample.is_multiple_of(2) {
                let start = Instant::now();
                matrix
                    .project_into(black_box(&input), black_box(&mut dense_output))
                    .expect("dense output-head projection");
                black_box(&dense_output);
                dense_times.push(start.elapsed());

                let start = Instant::now();
                matrix
                    .project_selected_rows_into(
                        black_box(&input),
                        black_box(&selected_rows),
                        black_box(&mut sparse_output),
                    )
                    .expect("sparse output-head projection");
                black_box(&sparse_output);
                sparse_times.push(start.elapsed());
            } else {
                let start = Instant::now();
                matrix
                    .project_selected_rows_into(
                        black_box(&input),
                        black_box(&selected_rows),
                        black_box(&mut sparse_output),
                    )
                    .expect("sparse output-head projection");
                black_box(&sparse_output);
                sparse_times.push(start.elapsed());

                let start = Instant::now();
                matrix
                    .project_into(black_box(&input), black_box(&mut dense_output))
                    .expect("dense output-head projection");
                black_box(&dense_output);
                dense_times.push(start.elapsed());
            }
        }
        dense_times.sort_unstable();
        sparse_times.sort_unstable();
        eprintln!(
            "qwen-sparse-output-head release rows={rows} columns={columns} group_size={group_size} allowed=64 samples=101 dense_p50_us={} dense_p95_us={} sparse_p50_us={} sparse_p95_us={}",
            duration_micros(dense_times[50]),
            duration_micros(dense_times[95]),
            duration_micros(sparse_times[50]),
            duration_micros(sparse_times[95]),
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "release-only selected-row versus dense output-head crossover measurement"]
    fn qwen_sparse_output_head_crossover_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        let (matrix, input) = qwen_sparse_output_head_fixture();
        let rows = matrix.rows();
        let mut dense_output = vec![0.0; rows];
        matrix
            .project_into(&input, &mut dense_output)
            .expect("dense Q4 output-head projection for crossover reference");

        for allowed_count in [4_096, 16_384, 32_768, 65_536, 131_072] {
            let selected_rows = (0..allowed_count)
                .map(|index| index * rows / allowed_count)
                .collect::<Vec<_>>();
            let mut sparse_output = vec![0.0; allowed_count];
            matrix
                .project_selected_rows_into(&input, &selected_rows, &mut sparse_output)
                .expect("selected Q4 rows for crossover reference");
            for (observed, row) in sparse_output.iter().zip(&selected_rows) {
                let reference = dense_output[*row];
                let tolerance = 1.0e-3 + reference.abs() * 1.0e-3;
                assert!((observed - reference).abs() <= tolerance);
            }

            let mut dense_times = Vec::with_capacity(31);
            let mut sparse_times = Vec::with_capacity(31);
            for sample in 0_usize..31 {
                if sample.is_multiple_of(2) {
                    let start = Instant::now();
                    matrix
                        .project_into(black_box(&input), black_box(&mut dense_output))
                        .expect("dense crossover sample");
                    black_box(&dense_output);
                    dense_times.push(start.elapsed());

                    let start = Instant::now();
                    matrix
                        .project_selected_rows_into(
                            black_box(&input),
                            black_box(&selected_rows),
                            black_box(&mut sparse_output),
                        )
                        .expect("sparse crossover sample");
                    black_box(&sparse_output);
                    sparse_times.push(start.elapsed());
                } else {
                    let start = Instant::now();
                    matrix
                        .project_selected_rows_into(
                            black_box(&input),
                            black_box(&selected_rows),
                            black_box(&mut sparse_output),
                        )
                        .expect("sparse crossover sample");
                    black_box(&sparse_output);
                    sparse_times.push(start.elapsed());

                    let start = Instant::now();
                    matrix
                        .project_into(black_box(&input), black_box(&mut dense_output))
                        .expect("dense crossover sample");
                    black_box(&dense_output);
                    dense_times.push(start.elapsed());
                }
            }
            dense_times.sort_unstable();
            sparse_times.sort_unstable();
            eprintln!(
                "qwen-sparse-crossover rows={} allowed={allowed_count} samples=31 dense_p50_us={} dense_p95_us={} sparse_p50_us={} sparse_p95_us={}",
                matrix.rows(),
                duration_micros(dense_times[15]),
                duration_micros(dense_times[29]),
                duration_micros(sparse_times[15]),
                duration_micros(sparse_times[29]),
            );
        }
    }

    #[cfg(target_os = "macos")]
    fn duration_micros(duration: std::time::Duration) -> u128 {
        duration.as_nanos() / 1_000
    }

    #[test]
    fn gated_delta_step_matches_scalar_recurrence_across_tokens() {
        let mut state = GatedDeltaState::new(1, 1, 1).expect("bounded state");
        let query = [1.0];
        let key = [1.0];
        let value = [4.0];
        let query_key_scale = (1.0f64 + 1.0e-6).sqrt().recip();

        let first = state
            .step(&query, &key, &value, &[0.0], &[0.25])
            .expect("first recurrent step");
        let first_state = 0.25 * 4.0 * query_key_scale;
        assert!((f64::from(first[0]) - first_state * query_key_scale).abs() < 1e-6);

        let second = state
            .step(&query, &key, &value, &[0.5f32.ln()], &[0.25])
            .expect("second recurrent step");
        let decayed_state = 0.5 * first_state;
        let expected_state =
            decayed_state + 0.25 * (4.0 - decayed_state * query_key_scale) * query_key_scale;
        assert!((f64::from(second[0]) - expected_state * query_key_scale).abs() < 1e-6);
    }

    #[test]
    fn gated_delta_simd_path_matches_scalar_multihead_state_across_tokens() {
        let heads = 3;
        let key_dimension = 7;
        let value_dimension = 9;
        let mut actual_state =
            GatedDeltaState::new(heads, key_dimension, value_dimension).expect("bounded state");
        let mut expected_state = vec![0.0f32; heads * key_dimension * value_dimension];

        for token in 0..4 {
            let query = (0..heads * key_dimension)
                .map(|index| ((index as f32 + token as f32 * 0.17 - 9.0) * 0.11).sin())
                .collect::<Vec<_>>();
            let key = (0..heads * key_dimension)
                .map(|index| ((index as f32 - token as f32 * 0.13 + 3.0) * 0.09).cos())
                .collect::<Vec<_>>();
            let value = (0..heads * value_dimension)
                .map(|index| ((index as f32 + token as f32 * 0.07 - 5.0) * 0.08).sin())
                .collect::<Vec<_>>();
            let log_decay = [-0.02, -0.1, -0.24];
            let beta = [0.2, 0.43, 0.71];
            let mut actual = vec![0.0f32; value.len()];
            actual_state
                .step_into(&query, &key, &value, &log_decay, &beta, &mut actual)
                .expect("caller-owned NEON multi-head recurrent step");

            let mut expected = vec![0.0f32; value.len()];
            let inverse_query_scale = (key_dimension as f64).sqrt().recip();
            for head in 0..heads {
                let query_start = head * key_dimension;
                let value_start = head * value_dimension;
                let state_start = head * key_dimension * value_dimension;
                let query_norm = (query[query_start..query_start + key_dimension]
                    .iter()
                    .fold(0.0f64, |sum, item| sum + f64::from(*item).powi(2))
                    + 1e-6)
                    .sqrt();
                let key_norm = (key[query_start..query_start + key_dimension]
                    .iter()
                    .fold(0.0f64, |sum, item| sum + f64::from(*item).powi(2))
                    + 1e-6)
                    .sqrt();
                let decay = f64::from(log_decay[head]).exp();
                for column in 0..value_dimension {
                    let mut prior = 0.0f64;
                    for row in 0..key_dimension {
                        let index = state_start + row * value_dimension + column;
                        prior += f64::from(expected_state[index])
                            * decay
                            * (f64::from(key[query_start + row]) / key_norm);
                    }
                    let correction =
                        f64::from(beta[head]) * (f64::from(value[value_start + column]) - prior);
                    let mut read = 0.0f64;
                    for row in 0..key_dimension {
                        let index = state_start + row * value_dimension + column;
                        let normalized_key = f64::from(key[query_start + row]) / key_norm;
                        expected_state[index] = (decay * f64::from(expected_state[index])
                            + correction * normalized_key)
                            as f32;
                        let normalized_query = f64::from(query[query_start + row]) / query_norm;
                        read += f64::from(expected_state[index])
                            * normalized_query
                            * inverse_query_scale;
                    }
                    expected[value_start + column] = read as f32;
                }
            }

            for (actual, expected) in actual.iter().zip(expected) {
                let tolerance = 4.0e-5 + expected.abs() * 4.0e-5;
                assert!((actual - expected).abs() <= tolerance);
            }
            for (actual, expected) in actual_state.values().iter().zip(&expected_state) {
                let tolerance = 4.0e-5 + expected.abs() * 4.0e-5;
                assert!((actual - expected).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn gated_delta_rejects_invalid_coefficients_without_changing_state() {
        let mut state = GatedDeltaState::new(1, 1, 1).expect("bounded state");
        assert!(state.step(&[1.0], &[1.0], &[2.0], &[0.0], &[1.1]).is_err());
        assert_eq!(state.values(), &[0.0]);
        assert!(state.step(&[1.0], &[1.0], &[2.0], &[0.1], &[0.5]).is_err());
        assert_eq!(state.values(), &[0.0]);
    }

    #[test]
    fn qwen_gated_coefficients_match_softplus_and_sigmoid() {
        let (log_decay, beta) = qwen35_gated_delta_coefficients(&[0.0], &[0.0], &[0.0], &[0.0])
            .expect("valid Qwen gate projections");
        assert!((log_decay[0] + 2.0f32.ln()).abs() < 1e-6);
        assert!((beta[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn qwen_gated_rms_norm_applies_silu_gate_after_scaled_normalization() {
        let output = rms_norm_gated(&[3.0, 4.0], &[1.0, -1.0], &[2.0, 0.5], 1e-6)
            .expect("valid gated RMS normalization");
        let inverse = 1.0f64 / (12.5f64 + 1.0e-6).sqrt();
        let silu = |value: f64| value / (1.0 + (-value).exp());
        let expected = [6.0 * inverse * silu(1.0), 2.0 * inverse * silu(-1.0)];
        assert!((f64::from(output[0]) - expected[0]).abs() < 1e-6);
        assert!((f64::from(output[1]) - expected[1]).abs() < 1e-6);
        assert!(rms_norm_gated(&[1.0], &[1.0, 2.0], &[1.0], 1e-6).is_err());
    }

    #[test]
    fn qwen_attention_rms_norm_uses_zero_centered_learned_weights() {
        let mut reused = [0.0; 2];
        rms_norm_zero_centered_into(&[3.0, 4.0], &[1.0, -1.0], 1e-6, &mut reused)
            .expect("caller-owned Qwen head normalization");
        let inverse = 1.0f64 / (12.5f64 + 1e-6).sqrt();
        assert!((f64::from(reused[0]) - 6.0 * inverse).abs() < 1e-6);
        assert!(reused[1].abs() < 1e-6);
        let mut invalid = [42.0];
        assert!(rms_norm_zero_centered_into(&[1.0], &[f32::NAN], 1e-6, &mut invalid).is_err());
        assert_eq!(invalid, [0.0]);
        assert!(rms_norm_zero_centered_into(&[1.0, 2.0], &[0.0, 0.0], 1e-6, &mut invalid).is_err());
        assert_eq!(invalid, [0.0]);
    }

    #[test]
    fn qwen_partial_rope_uses_split_half_pairs_and_preserves_unrotated_dimensions() {
        let mut vector = [1.0, 0.0, 5.0];
        rotary_qwen35_partial(&mut vector, 1, 2, 10_000.0).expect("valid Qwen partial RoPE");
        assert!((vector[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((vector[1] - 1.0f32.sin()).abs() < 1e-6);
        assert_eq!(vector[2], 5.0);
        assert!(rotary_qwen35_partial(&mut vector, 0, 4, 10_000.0).is_err());
    }

    #[test]
    fn qwen_mrope_assigns_interleaved_frequency_pairs_to_three_axes() {
        let mut vector = vec![1.0f32; 68];
        rotary_qwen35_mrope_partial(&mut vector, [1, 2, 3], 64, [11, 11, 10], 10_000.0)
            .expect("valid Qwen three-axis partial RoPE");
        for pair in 0..3 {
            let position = (pair + 1) as f64;
            let exponent = (2 * pair) as f64 / 64.0;
            let angle = position / 10_000.0f64.powf(exponent);
            assert!((f64::from(vector[pair]) - (angle.cos() - angle.sin())).abs() < 1e-6);
            assert!((f64::from(vector[pair + 32]) - (angle.cos() + angle.sin())).abs() < 1e-6);
        }
        assert_eq!(&vector[64..], &[1.0, 1.0, 1.0, 1.0]);
        assert!(
            rotary_qwen35_mrope_partial(&mut vector, [1, 2, 3], 64, [10, 11, 11], 10_000.0)
                .is_err()
        );
    }

    #[test]
    fn qwen_mrope_reference_preserves_large_bounded_head_dimensions() {
        let mut vector = vec![0.25f32; 2048];
        rotary_qwen35_mrope_partial(&mut vector, [7, 13, 29], 2048, [342, 341, 341], 10_000.0)
            .expect("bounded generic MRoPE dimension");
        assert!(vector.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn prepared_qwen_mrope_angles_match_direct_reference_for_every_head() {
        let dimensions = 64;
        let sections = [11, 11, 10];
        let denominators = qwen35_mrope_denominators(dimensions, 10_000.0, sections)
            .expect("validated MRoPE denominators");
        let mut angles = vec![(0.0, 0.0); dimensions / 2];
        let positions = [7, 13, 29];
        prepare_qwen35_mrope_angles(&mut angles, &denominators, positions)
            .expect("prepared MRoPE angles");

        for head in 0..20 {
            let mut actual = (0..80)
                .map(|index| ((head * 80 + index) as f32 * 0.017).sin())
                .collect::<Vec<_>>();
            let mut expected = actual.clone();
            for pair in 0..dimensions / 2 {
                let exponent = (2 * pair) as f32 / dimensions as f32;
                let angle =
                    (positions[pair % 3] as f64 / 10_000.0f64.powf(f64::from(exponent))) as f32;
                let (sine, cosine) = angle.sin_cos();
                let left = expected[pair];
                let right = expected[pair + dimensions / 2];
                expected[pair] = left * cosine - right * sine;
                expected[pair + dimensions / 2] = right * cosine + left * sine;
            }
            rotary_qwen35_mrope_with_angles(&mut actual, dimensions, &angles)
                .expect("apply prepared MRoPE angles");
            assert_eq!(actual, expected, "head {head}");
        }
    }

    #[test]
    #[ignore = "release-only Qwen MRoPE repeated-head latency measurement"]
    fn qwen_mrope_prepared_angle_latency_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        fn rotate_direct(head: &mut [f32], positions: [u64; 3], dimensions: usize) {
            let half = dimensions / 2;
            for pair in 0..half {
                let exponent = (2 * pair) as f32 / dimensions as f32;
                let angle =
                    (positions[pair % 3] as f64 / 10_000_000.0f64.powf(f64::from(exponent))) as f32;
                let (sine, cosine) = angle.sin_cos();
                let left = head[pair];
                let right = head[pair + half];
                head[pair] = left * cosine - right * sine;
                head[pair + half] = right * cosine + left * sine;
            }
        }

        let dimensions = 64;
        let sections = [11, 11, 10];
        let positions = [123, 47, 89];
        let denominators = qwen35_mrope_denominators(dimensions, 10_000_000.0, sections)
            .expect("validated MRoPE denominators");
        let source = (0..20)
            .map(|head| {
                (0..256)
                    .map(|index| ((head * 256 + index) as f32 * 0.011).sin())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut repeated_heads = source.clone();
        let mut prepared_heads = source.clone();
        let mut angle_scratch = vec![(0.0, 0.0); dimensions / 2];
        let mut run_repeated = || {
            for (head, original) in repeated_heads.iter_mut().zip(&source) {
                head.copy_from_slice(original);
                rotate_direct(head, positions, dimensions);
            }
            black_box(&repeated_heads);
        };
        let mut run_prepared = || {
            for (head, original) in prepared_heads.iter_mut().zip(&source) {
                head.copy_from_slice(original);
            }
            prepare_qwen35_mrope_angles(&mut angle_scratch, &denominators, positions)
                .expect("prepare reusable MRoPE angles");
            for head in &mut prepared_heads {
                rotary_qwen35_mrope_with_angles(head, dimensions, &angle_scratch)
                    .expect("apply reusable MRoPE angles");
            }
            black_box(&prepared_heads);
        };

        run_repeated();
        run_prepared();
        let mut repeated = Vec::with_capacity(101);
        let mut prepared = Vec::with_capacity(101);
        for sample in 0_usize..101 {
            let start = Instant::now();
            if sample.is_multiple_of(2) {
                run_repeated();
                repeated.push(start.elapsed());
                let start = Instant::now();
                run_prepared();
                prepared.push(start.elapsed());
            } else {
                run_prepared();
                prepared.push(start.elapsed());
                let start = Instant::now();
                run_repeated();
                repeated.push(start.elapsed());
            }
        }
        repeated.sort_unstable();
        prepared.sort_unstable();
        eprintln!(
            "qwen-mrope heads=20 rotary_dimensions={dimensions} samples=101 repeated_p50_us={} repeated_p95_us={} prepared_p50_us={} prepared_p95_us={} p50_reduction_percent={:.1} p95_reduction_percent={:.1}",
            repeated[50].as_nanos() / 1_000,
            repeated[95].as_nanos() / 1_000,
            prepared[50].as_nanos() / 1_000,
            prepared[95].as_nanos() / 1_000,
            (1.0 - prepared[50].as_secs_f64() / repeated[50].as_secs_f64()) * 100.0,
            (1.0 - prepared[95].as_secs_f64() / repeated[95].as_secs_f64()) * 100.0,
        );
    }

    #[test]
    fn qwen_vision_axial_rope_uses_height_then_width_frequency_halves() {
        let mut vector = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        rotary_qwen35_vision_axial(&mut vector, [1, 2], 10_000.0).expect("valid vision axial RoPE");
        let expected_angles = [1.0f64, 1.0 / 100.0, 2.0, 2.0 / 100.0];
        for (pair, angle) in expected_angles.into_iter().enumerate() {
            assert!((f64::from(vector[pair]) - angle.cos()).abs() < 1e-6);
            assert!((f64::from(vector[pair + 4]) - angle.sin()).abs() < 1e-6);
        }
        assert!(rotary_qwen35_vision_axial(&mut [1.0, 0.0, 0.0], [0, 0], 10_000.0).is_err());
        assert!(rotary_qwen35_vision_axial(&mut [f32::NAN; 4], [0, 0], 10_000.0).is_err());
    }

    #[test]
    fn causal_depthwise_convolution_keeps_ordered_history_and_silu() {
        let mut state = CausalDepthwiseConvState::new(1, 3).expect("bounded convolution state");
        let mut into_state = CausalDepthwiseConvState::new(1, 3).expect("caller-owned state");
        let weights = [0.01, 0.1, 1.0];
        for (input, expected_convolution) in [(1.0f32, 1.0f32), (2.0f32, 2.1f32), (3.0f32, 3.21f32)]
        {
            let output = state
                .step(&[input], &weights, None)
                .expect("causal convolution step");
            let mut reused = [99.0];
            into_state
                .step_into(&[input], &weights, None, &mut reused)
                .expect("caller-owned causal convolution step");
            assert_eq!(reused, output.as_slice());
            let expected = expected_convolution / (1.0 + (-expected_convolution).exp());
            assert!((output[0] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn causal_convolution_does_not_advance_after_invalid_weights() {
        let mut state = CausalDepthwiseConvState::new(1, 2).expect("bounded convolution state");
        assert!(state.step(&[2.0], &[1.0], None).is_err());
        let output = state
            .step(&[3.0], &[0.0, 1.0], None)
            .expect("valid convolution step");
        let expected = 3.0 / (1.0 + (-3.0f32).exp());
        assert!((output[0] - expected).abs() < 1e-6);
    }
}
