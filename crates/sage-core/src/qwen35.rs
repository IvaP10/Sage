//! Checked architecture metadata for Sage's initial Qwen3.5-4B target.
//!
//! This module defines the supported tensor geometry; it does not dispatch to
//! Transformers, llama.cpp, ONNX Runtime, or a remote inference service.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{Read, Seek};

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use zeroize::{Zeroize, Zeroizing};

use crate::inference_cpu::{
    CausalDepthwiseConvState, CpuMatrix, GatedDeltaState, GroupedQueryAttentionScratch, KvCache,
    QuantizedQ4Builder, QuantizedQ4Matrix, grouped_query_attention_into,
    prepare_qwen35_mrope_angles, qwen35_gated_delta_coefficients_into, qwen35_mrope_denominators,
    rms_norm_gated_into, rms_norm_zero_centered_into, rotary_qwen35_mrope_with_angles, silu,
};
use crate::{CoreError, CoreResult};

#[path = "qwen35_loader.rs"]
mod weight_loader;
pub use weight_loader::{
    Qwen35CandidateLoadOptions, Qwen35CandidateModel, Qwen35DecoderLayerTensorNames,
    Qwen35LayerMixerTensorNames, VerifiedQwen35Config, VerifiedQwen35ImageProcessor,
    VerifiedQwen35TextWeights, VerifiedQwen35Tokenizer, VerifiedQwen35WeightIndex,
};

pub const SAGE_CONTEXT_LIMIT: u32 = 8_192;
pub const SAGE_OUTPUT_LIMIT: u32 = 2_048;
const MAX_CONFIG_BYTES: usize = 1024 * 1024;
const MAX_INDEX_BYTES: usize = 1024 * 1024;
const EXPECTED_WEIGHT_BYTES: u64 = 9_319_737_856;
const EXPECTED_TENSOR_COUNT: usize = 738;
const HIDDEN_SIZE: usize = 2_560;
const INTERMEDIATE_SIZE: usize = 9_216;
const VOCABULARY_SIZE: usize = 248_320;
const QUERY_PROJECTION_SIZE: usize = 8_192;
const KEY_VALUE_PROJECTION_SIZE: usize = 1_024;
const ATTENTION_OUTPUT_SIZE: usize = 4_096;
const DELTA_QKV_SIZE: usize = 8_192;
const DELTA_VALUE_SIZE: usize = 4_096;
const QWEN35_4B_KEY_HEADS: usize = 16;
const QWEN35_4B_VALUE_HEADS: usize = 32;
const QWEN35_4B_HEAD_DIMENSION: usize = 128;
const QWEN35_4B_CONV_KERNEL_SIZE: usize = 4;
const QWEN35_4B_MAX_POSITION_EMBEDDINGS: u64 = 262_144;
const MAX_WEIGHT_IMPORT_CHUNK_ELEMENTS: usize = 1_048_576;
const MAX_QWEN_CPU_MATRIX_IMPORT_ELEMENTS: usize = 32_000_000;

/// Session-local state for one Qwen3.5 linear-attention block. It composes the
/// causal depthwise QKV history with the recurrent gated-delta matrix. Model
/// projections and trained weights are supplied by the caller; this type does
/// not load a model or create an inference worker.
pub struct Qwen35LinearAttentionState {
    key_heads: usize,
    value_heads: usize,
    key_dimension: usize,
    value_dimension: usize,
    convolution: CausalDepthwiseConvState,
    recurrence: GatedDeltaState,
    convolution_scratch: Zeroizing<Vec<f32>>,
    query_scratch: Zeroizing<Vec<f32>>,
    key_scratch: Zeroizing<Vec<f32>>,
    log_decay_scratch: Zeroizing<Vec<f32>>,
    beta_scratch: Zeroizing<Vec<f32>>,
}

impl Qwen35LinearAttentionState {
    pub fn new(
        key_heads: usize,
        value_heads: usize,
        key_dimension: usize,
        value_dimension: usize,
        convolution_kernel_size: usize,
    ) -> CoreResult<Self> {
        if key_heads == 0
            || value_heads == 0
            || !value_heads.is_multiple_of(key_heads)
            || key_heads > 64
            || value_heads > 64
        {
            return Err(CoreError::Model(
                "Qwen linear-attention head mapping is invalid".into(),
            ));
        }
        let key_width = key_heads
            .checked_mul(key_dimension)
            .ok_or_else(|| CoreError::Model("Qwen key width overflow".into()))?;
        let value_width = value_heads
            .checked_mul(value_dimension)
            .ok_or_else(|| CoreError::Model("Qwen value width overflow".into()))?;
        let convolution_channels = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or_else(|| CoreError::Model("Qwen QKV width overflow".into()))?;
        let repeated_width = value_heads
            .checked_mul(key_dimension)
            .ok_or_else(|| CoreError::Model("Qwen repeated key width overflow".into()))?;
        Ok(Self {
            key_heads,
            value_heads,
            key_dimension,
            value_dimension,
            convolution: CausalDepthwiseConvState::new(
                convolution_channels,
                convolution_kernel_size,
            )?,
            recurrence: GatedDeltaState::new(value_heads, key_dimension, value_dimension)?,
            convolution_scratch: Zeroizing::new(vec![0.0; convolution_channels]),
            query_scratch: Zeroizing::new(vec![0.0; repeated_width]),
            key_scratch: Zeroizing::new(vec![0.0; repeated_width]),
            log_decay_scratch: Zeroizing::new(vec![0.0; value_heads]),
            beta_scratch: Zeroizing::new(vec![0.0; value_heads]),
        })
    }

    /// Construct one of the 24 linear-attention layer states in the pinned
    /// Qwen3.5-4B profile: 16 key heads, 32 value heads, 128-wide heads and a
    /// four-tap causal convolution.
    pub fn for_qwen35_4b() -> CoreResult<Self> {
        Self::new(
            QWEN35_4B_KEY_HEADS,
            QWEN35_4B_VALUE_HEADS,
            QWEN35_4B_HEAD_DIMENSION,
            QWEN35_4B_HEAD_DIMENSION,
            QWEN35_4B_CONV_KERNEL_SIZE,
        )
    }

    pub fn projected_qkv_width(&self) -> usize {
        self.convolution.channels()
    }

    /// Apply Qwen's SiLU-gated RMS normalization independently to each value
    /// head's recurrent read, returning the flattened value width expected by
    /// `out_proj`.
    pub fn normalize_gated_output(
        &self,
        recurrent_output: &[f32],
        projected_gate: &[f32],
        norm_weight: &[f32],
        epsilon: f32,
    ) -> CoreResult<Vec<f32>> {
        let output_width = self
            .value_heads
            .checked_mul(self.value_dimension)
            .ok_or_else(|| CoreError::Model("Qwen gated output width overflow".into()))?;
        let mut output = vec![0.0; output_width];
        self.normalize_gated_output_into(
            recurrent_output,
            projected_gate,
            norm_weight,
            epsilon,
            &mut output,
        )?;
        Ok(output)
    }

    pub fn normalize_gated_output_into(
        &self,
        recurrent_output: &[f32],
        projected_gate: &[f32],
        norm_weight: &[f32],
        epsilon: f32,
        output: &mut [f32],
    ) -> CoreResult<()> {
        let output_width = self
            .value_heads
            .checked_mul(self.value_dimension)
            .ok_or_else(|| CoreError::Model("Qwen gated output width overflow".into()))?;
        if recurrent_output.len() != output_width
            || projected_gate.len() != output_width
            || norm_weight.len() != self.value_dimension
            || output.len() != output_width
        {
            output.fill(0.0);
            return Err(CoreError::Model(
                "Qwen gated output normalization shapes do not match".into(),
            ));
        }
        let result = (|| {
            for head in 0..self.value_heads {
                let start = head * self.value_dimension;
                rms_norm_gated_into(
                    &recurrent_output[start..start + self.value_dimension],
                    &projected_gate[start..start + self.value_dimension],
                    norm_weight,
                    epsilon,
                    &mut output[start..start + self.value_dimension],
                )?;
            }
            Ok(())
        })();
        if result.is_err() {
            output.zeroize();
        }
        result
    }

    /// Consume one token's projected QKV and scalar gate values. The Qwen
    /// convolution has no bias. Key/query heads are repeated contiguously to
    /// match value heads before the recurrent update.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        projected_qkv: &[f32],
        convolution_weights: &[f32],
        projected_decay: &[f32],
        projected_beta: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
    ) -> CoreResult<Vec<f32>> {
        let value_width = self.value_heads * self.value_dimension;
        let mut output = vec![0.0; value_width];
        self.step_into(
            projected_qkv,
            convolution_weights,
            projected_decay,
            projected_beta,
            a_log,
            dt_bias,
            &mut output,
        )?;
        Ok(output)
    }

    /// Consume one token into caller-owned recurrent output storage. All
    /// convolution, head-expansion and gate workspaces are retained by this
    /// session state, so the hot path performs no intermediate allocations.
    #[allow(clippy::too_many_arguments)]
    pub fn step_into(
        &mut self,
        projected_qkv: &[f32],
        convolution_weights: &[f32],
        projected_decay: &[f32],
        projected_beta: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        output: &mut [f32],
    ) -> CoreResult<()> {
        output.zeroize();
        let value_width = self
            .value_heads
            .checked_mul(self.value_dimension)
            .ok_or_else(|| CoreError::Model("Qwen recurrent output width overflow".into()))?;
        if output.len() != value_width {
            self.clear_scratch();
            return Err(CoreError::Model(
                "Qwen recurrent output buffer has the wrong shape".into(),
            ));
        }
        let mut convolution_advanced = false;
        let result = (|| {
            qwen35_gated_delta_coefficients_into(
                projected_decay,
                projected_beta,
                a_log,
                dt_bias,
                &mut self.log_decay_scratch,
                &mut self.beta_scratch,
            )?;
            self.convolution.step_into(
                projected_qkv,
                convolution_weights,
                None,
                &mut self.convolution_scratch,
            )?;
            convolution_advanced = true;
            let key_width = self.key_heads * self.key_dimension;
            let repeated_width = self.value_heads * self.key_dimension;
            let repeated_heads = self.value_heads / self.key_heads;
            for value_head in 0..self.value_heads {
                let key_head = value_head / repeated_heads;
                let start = key_head * self.key_dimension;
                let repeated_start = value_head * self.key_dimension;
                self.query_scratch[repeated_start..repeated_start + self.key_dimension]
                    .copy_from_slice(&self.convolution_scratch[start..start + self.key_dimension]);
                self.key_scratch[repeated_start..repeated_start + self.key_dimension]
                    .copy_from_slice(
                        &self.convolution_scratch
                            [key_width + start..key_width + start + self.key_dimension],
                    );
            }
            let value_start = key_width * 2;
            self.recurrence.step_into(
                &self.query_scratch[..repeated_width],
                &self.key_scratch[..repeated_width],
                &self.convolution_scratch[value_start..value_start + value_width],
                &self.log_decay_scratch,
                &self.beta_scratch,
                output,
            )
        })();
        if result.is_err() && convolution_advanced {
            // Convolution advanced but recurrence did not finish. Discard both
            // caches rather than retain a split-brain token prefix.
            self.clear();
        }
        self.clear_scratch();
        result
    }

    pub fn clear(&mut self) {
        self.convolution.clear();
        self.recurrence.clear();
        self.clear_scratch();
    }

    fn clear_scratch(&mut self) {
        self.convolution_scratch.as_mut_slice().zeroize();
        self.query_scratch.as_mut_slice().zeroize();
        self.key_scratch.as_mut_slice().zeroize();
        self.log_decay_scratch.as_mut_slice().zeroize();
        self.beta_scratch.as_mut_slice().zeroize();
    }
}

/// Matrix representation for a trained Qwen projection. Q4 is Sage's scalar
/// reference format and remains subject to checkpoint-level quality tests.
#[derive(Debug, Clone)]
pub enum Qwen35ProjectionMatrix {
    F32(CpuMatrix),
    GroupedQ4(QuantizedQ4Matrix),
}

impl Qwen35ProjectionMatrix {
    pub fn quantize_q4(self, group_size: usize) -> CoreResult<Self> {
        match self {
            Self::F32(matrix) => Ok(Self::GroupedQ4(QuantizedQ4Matrix::encode(
                matrix.rows(),
                matrix.columns(),
                group_size,
                matrix.values(),
            )?)),
            quantized @ Self::GroupedQ4(_) => Ok(quantized),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::F32(matrix) => matrix.rows(),
            Self::GroupedQ4(matrix) => matrix.rows(),
        }
    }

    pub fn columns(&self) -> usize {
        match self {
            Self::F32(matrix) => matrix.columns(),
            Self::GroupedQ4(matrix) => matrix.columns(),
        }
    }

    fn project(&self, input: &[f32]) -> CoreResult<Vec<f32>> {
        match self {
            Self::F32(matrix) => matrix.project(input, None),
            Self::GroupedQ4(matrix) => matrix.project(input),
        }
    }

    fn project_into(&self, input: &[f32], output: &mut [f32]) -> CoreResult<()> {
        match self {
            Self::F32(matrix) => matrix.project_into(input, None, output),
            Self::GroupedQ4(matrix) => matrix.project_into(input, output),
        }
    }

    fn project_selected_rows_into(
        &self,
        input: &[f32],
        selected_rows: &[usize],
        output: &mut [f32],
    ) -> CoreResult<()> {
        if input.len() != self.columns()
            || output.len() != selected_rows.len()
            || selected_rows.len() > self.rows()
            || input.iter().any(|value| !value.is_finite())
            || selected_rows.iter().any(|row| *row >= self.rows())
        {
            output.fill(0.0);
            return Err(CoreError::Model(
                "Qwen selected projection rows or buffers are invalid".into(),
            ));
        }
        match self {
            Self::F32(matrix) => {
                for (output_index, row) in selected_rows.iter().copied().enumerate() {
                    let start = row
                        .checked_mul(matrix.columns())
                        .ok_or_else(|| CoreError::Model("Qwen row offset overflow".into()))?;
                    let end = start + matrix.columns();
                    let sum = matrix.values()[start..end].iter().zip(input).fold(
                        0.0f64,
                        |sum, (weight, activation)| {
                            sum + f64::from(*weight) * f64::from(*activation)
                        },
                    );
                    let projected = sum as f32;
                    if !projected.is_finite() {
                        output.fill(0.0);
                        return Err(CoreError::Model(
                            "Qwen selected projection produced a non-finite result".into(),
                        ));
                    }
                    output[output_index] = projected;
                }
                Ok(())
            }
            Self::GroupedQ4(matrix) => {
                matrix.project_selected_rows_into(input, selected_rows, output)
            }
        }
    }

    pub(crate) fn project_batch_with_bias_into(
        &self,
        input: &[f32],
        batch_size: usize,
        bias: &[f32],
        output: &mut [f32],
    ) -> CoreResult<()> {
        let input_elements = batch_size.checked_mul(self.columns());
        let output_elements = batch_size.checked_mul(self.rows());
        if batch_size == 0
            || batch_size > sage_kernels::Q4_BATCH_MAX_SIZE
            || input_elements != Some(input.len())
            || output_elements != Some(output.len())
            || input.iter().any(|value| !value.is_finite())
            || bias.len() != self.rows()
            || bias.iter().any(|value| !value.is_finite())
        {
            output.fill(0.0);
            return Err(CoreError::Model(
                "Qwen batched projection input, output, or bias is invalid".into(),
            ));
        }

        match self {
            Self::F32(matrix) => {
                for batch in 0..batch_size {
                    let input_start = batch * matrix.columns();
                    let output_start = batch * matrix.rows();
                    if let Err(error) = matrix.project_into(
                        &input[input_start..input_start + matrix.columns()],
                        Some(bias),
                        &mut output[output_start..output_start + matrix.rows()],
                    ) {
                        output.fill(0.0);
                        return Err(error);
                    }
                }
            }
            Self::GroupedQ4(matrix) => {
                if batch_size < sage_kernels::Q4_BATCH_TILE_SIZE * 2 {
                    for batch in 0..batch_size {
                        let input_start = batch * matrix.columns();
                        let output_start = batch * matrix.rows();
                        let batch_output = &mut output[output_start..output_start + matrix.rows()];
                        if let Err(error) = matrix.project_into(
                            &input[input_start..input_start + matrix.columns()],
                            batch_output,
                        ) {
                            output.fill(0.0);
                            return Err(error);
                        }
                        let mut non_finite = false;
                        for (value, offset) in batch_output.iter_mut().zip(bias) {
                            *value += *offset;
                            non_finite |= !value.is_finite();
                        }
                        if non_finite {
                            output.fill(0.0);
                            return Err(CoreError::Model(
                                "Qwen batched projection produced a non-finite result".into(),
                            ));
                        }
                    }
                    return Ok(());
                }
                let Some(scratch_len) = matrix
                    .columns()
                    .checked_mul(batch_size.min(sage_kernels::Q4_BATCH_TILE_SIZE))
                    .filter(|count| *count <= sage_kernels::Q4_BATCH_MAX_SCRATCH_ELEMENTS)
                else {
                    output.fill(0.0);
                    return Err(CoreError::Model(
                        "Qwen batch transpose scratch exceeds its bounded limit".into(),
                    ));
                };
                let mut scratch = Zeroizing::new(Vec::new());
                if scratch.try_reserve_exact(scratch_len).is_err() {
                    output.fill(0.0);
                    return Err(CoreError::Model(
                        "Qwen batch transpose scratch allocation was denied".into(),
                    ));
                }
                scratch.resize(scratch_len, 0.0);
                if let Err(error) =
                    matrix.project_batch_into(input, batch_size, &mut scratch, output)
                {
                    output.fill(0.0);
                    return Err(error);
                }
                let mut non_finite = false;
                for batch_output in output.chunks_exact_mut(matrix.rows()) {
                    for (value, offset) in batch_output.iter_mut().zip(bias) {
                        *value += *offset;
                        if !value.is_finite() {
                            non_finite = true;
                        }
                    }
                }
                if non_finite {
                    output.fill(0.0);
                    return Err(CoreError::Model(
                        "Qwen batched projection produced a non-finite result".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn project_with_bias(&self, input: &[f32], bias: &[f32]) -> CoreResult<Vec<f32>> {
        if bias.len() != self.rows() || bias.iter().any(|value| !value.is_finite()) {
            return Err(CoreError::Model(
                "Qwen projection bias does not match the output geometry".into(),
            ));
        }
        match self {
            Self::F32(matrix) => matrix.project(input, Some(bias)),
            Self::GroupedQ4(matrix) => {
                let mut output = matrix.project(input)?;
                for (value, offset) in output.iter_mut().zip(bias) {
                    *value += *offset;
                    if !value.is_finite() {
                        output.zeroize();
                        return Err(CoreError::Model(
                            "Qwen biased projection produced a non-finite result".into(),
                        ));
                    }
                }
                Ok(output)
            }
        }
    }

    fn row(&self, row: usize) -> CoreResult<Vec<f32>> {
        match self {
            Self::F32(matrix) => {
                if row >= matrix.rows() {
                    return Err(CoreError::Model(
                        "Qwen embedding token ID is outside the vocabulary".into(),
                    ));
                }
                let start = row
                    .checked_mul(matrix.columns())
                    .ok_or_else(|| CoreError::Model("Qwen embedding row overflow".into()))?;
                Ok(matrix.values()[start..start + matrix.columns()].to_vec())
            }
            Self::GroupedQ4(matrix) => matrix.row(row),
        }
    }
}

impl From<CpuMatrix> for Qwen35ProjectionMatrix {
    fn from(value: CpuMatrix) -> Self {
        Self::F32(value)
    }
}

impl From<QuantizedQ4Matrix> for Qwen35ProjectionMatrix {
    fn from(value: QuantizedQ4Matrix) -> Self {
        Self::GroupedQ4(value)
    }
}

/// Trained projection and normalization tensors for one Qwen3.5 linear-
/// attention sublayer. Matrix rows follow the model's `[out, in]` layout.
/// Construction of a block validates every dimension before these weights can
/// be used with session state.
#[derive(Debug, Clone)]
pub struct Qwen35LinearAttentionWeights {
    pub qkv_projection: Qwen35ProjectionMatrix,
    pub gate_projection: Qwen35ProjectionMatrix,
    pub decay_projection: Qwen35ProjectionMatrix,
    pub beta_projection: Qwen35ProjectionMatrix,
    pub convolution: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub output_projection: Qwen35ProjectionMatrix,
}

/// One-token Qwen3.5 linear-attention sublayer with caller-owned recurrent
/// state. It expects input already normalized by the decoder layer and does
/// not implement the residual, MLP, full-attention blocks, or token loop.
pub struct Qwen35LinearAttentionBlock {
    hidden_size: usize,
    norm_epsilon: f32,
    weights: Qwen35LinearAttentionWeights,
    state: Qwen35LinearAttentionState,
    qkv_projection_scratch: Zeroizing<Vec<f32>>,
    gate_projection_scratch: Zeroizing<Vec<f32>>,
    decay_projection_scratch: Zeroizing<Vec<f32>>,
    beta_projection_scratch: Zeroizing<Vec<f32>>,
    recurrent_scratch: Zeroizing<Vec<f32>>,
    normalization_scratch: Zeroizing<Vec<f32>>,
}

impl Qwen35LinearAttentionBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        key_heads: usize,
        value_heads: usize,
        key_dimension: usize,
        value_dimension: usize,
        convolution_kernel_size: usize,
        norm_epsilon: f32,
        weights: Qwen35LinearAttentionWeights,
    ) -> CoreResult<Self> {
        let state = Qwen35LinearAttentionState::new(
            key_heads,
            value_heads,
            key_dimension,
            value_dimension,
            convolution_kernel_size,
        )?;
        let qkv_width = state.projected_qkv_width();
        let value_width = value_heads
            .checked_mul(value_dimension)
            .ok_or_else(|| CoreError::Model("Qwen value width overflow".into()))?;
        let convolution_weight_count = qkv_width
            .checked_mul(convolution_kernel_size)
            .ok_or_else(|| CoreError::Model("Qwen convolution weight count overflow".into()))?;
        if hidden_size == 0
            || !norm_epsilon.is_finite()
            || norm_epsilon <= 0.0
            || weights.qkv_projection.rows() != qkv_width
            || weights.qkv_projection.columns() != hidden_size
            || weights.gate_projection.rows() != value_width
            || weights.gate_projection.columns() != hidden_size
            || weights.decay_projection.rows() != value_heads
            || weights.decay_projection.columns() != hidden_size
            || weights.beta_projection.rows() != value_heads
            || weights.beta_projection.columns() != hidden_size
            || weights.convolution.len() != convolution_weight_count
            || weights.a_log.len() != value_heads
            || weights.dt_bias.len() != value_heads
            || weights.norm_weight.len() != value_dimension
            || weights.output_projection.rows() != hidden_size
            || weights.output_projection.columns() != value_width
            || weights
                .convolution
                .iter()
                .chain(&weights.a_log)
                .chain(&weights.dt_bias)
                .chain(&weights.norm_weight)
                .any(|value| !value.is_finite())
        {
            return Err(CoreError::Model(
                "Qwen linear-attention weights do not match the declared layer geometry".into(),
            ));
        }
        Ok(Self {
            hidden_size,
            norm_epsilon,
            qkv_projection_scratch: Zeroizing::new(vec![0.0; qkv_width]),
            gate_projection_scratch: Zeroizing::new(vec![0.0; value_width]),
            decay_projection_scratch: Zeroizing::new(vec![0.0; value_heads]),
            beta_projection_scratch: Zeroizing::new(vec![0.0; value_heads]),
            recurrent_scratch: Zeroizing::new(vec![0.0; value_width]),
            weights,
            state,
            normalization_scratch: Zeroizing::new(vec![0.0; value_width]),
        })
    }

    /// Construct the pinned 4B linear-attention geometry (hidden width 2560,
    /// 16 key heads, 32 value heads, 128 dimensions, four convolution taps).
    pub fn for_qwen35_4b(
        norm_epsilon: f32,
        weights: Qwen35LinearAttentionWeights,
    ) -> CoreResult<Self> {
        Self::new(
            HIDDEN_SIZE,
            QWEN35_4B_KEY_HEADS,
            QWEN35_4B_VALUE_HEADS,
            QWEN35_4B_HEAD_DIMENSION,
            QWEN35_4B_HEAD_DIMENSION,
            QWEN35_4B_CONV_KERNEL_SIZE,
            norm_epsilon,
            weights,
        )
    }

    /// Apply one normalized hidden vector and return the projected layer
    /// output. If an error occurs after convolution or recurrence has advanced,
    /// clear both caches so later calls cannot continue from a partial token.
    pub fn step(&mut self, normalized_hidden: &[f32]) -> CoreResult<Vec<f32>> {
        let mut output = vec![0.0; self.hidden_size];
        self.step_into(normalized_hidden, &mut output)?;
        Ok(output)
    }

    fn step_into(&mut self, normalized_hidden: &[f32], output: &mut [f32]) -> CoreResult<()> {
        output.zeroize();
        if normalized_hidden.len() != self.hidden_size
            || output.len() != self.hidden_size
            || normalized_hidden.iter().any(|value| !value.is_finite())
        {
            self.clear_scratch();
            return Err(CoreError::Model(
                "Qwen linear-attention hidden state has the wrong shape or non-finite values"
                    .into(),
            ));
        }
        let mut state_advanced = false;
        let result = (|| {
            self.weights
                .qkv_projection
                .project_into(normalized_hidden, &mut self.qkv_projection_scratch)?;
            self.weights
                .gate_projection
                .project_into(normalized_hidden, &mut self.gate_projection_scratch)?;
            self.weights
                .decay_projection
                .project_into(normalized_hidden, &mut self.decay_projection_scratch)?;
            self.weights
                .beta_projection
                .project_into(normalized_hidden, &mut self.beta_projection_scratch)?;
            self.state.step_into(
                &self.qkv_projection_scratch,
                &self.weights.convolution,
                &self.decay_projection_scratch,
                &self.beta_projection_scratch,
                &self.weights.a_log,
                &self.weights.dt_bias,
                &mut self.recurrent_scratch,
            )?;
            state_advanced = true;
            self.state.normalize_gated_output_into(
                &self.recurrent_scratch,
                &self.gate_projection_scratch,
                &self.weights.norm_weight,
                self.norm_epsilon,
                &mut self.normalization_scratch,
            )?;
            self.weights
                .output_projection
                .project_into(&self.normalization_scratch, output)
        })();
        if result.is_err() && state_advanced {
            self.state.clear();
        }
        self.clear_scratch();
        if result.is_err() {
            output.zeroize();
        }
        result
    }

    pub fn clear(&mut self) {
        self.state.clear();
        self.clear_scratch();
    }

    fn clear_scratch(&mut self) {
        self.qkv_projection_scratch.as_mut_slice().zeroize();
        self.gate_projection_scratch.as_mut_slice().zeroize();
        self.decay_projection_scratch.as_mut_slice().zeroize();
        self.beta_projection_scratch.as_mut_slice().zeroize();
        self.recurrent_scratch.as_mut_slice().zeroize();
        self.normalization_scratch.as_mut_slice().zeroize();
    }
}

/// Trained projections and head-normalization weights for one full-attention
/// Qwen3.5 layer. Projection rows retain the model's serialized order:
/// `[head, query-or-gate, dimension]` for q_proj and `[head, dimension]` for
/// k_proj/v_proj.
#[derive(Debug, Clone)]
pub struct Qwen35FullAttentionWeights {
    pub query_gate_projection: Qwen35ProjectionMatrix,
    pub key_projection: Qwen35ProjectionMatrix,
    pub value_projection: Qwen35ProjectionMatrix,
    pub output_projection: Qwen35ProjectionMatrix,
    pub query_norm_weight: Vec<f32>,
    pub key_norm_weight: Vec<f32>,
}

/// Stateful single-token Qwen3.5 full-attention sublayer. Input is already
/// normalized by its decoder layer; residual and MLP behavior belong to that
/// layer. Vision MRoPE positions are not accepted here yet: this path is for
/// text tokens with a shared position across the three Qwen rotary axes.
pub struct Qwen35FullAttentionBlock {
    hidden_size: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dimension: usize,
    rotary_dimensions: usize,
    mrope_denominators: Vec<f64>,
    mrope_angles: Vec<(f32, f32)>,
    norm_epsilon: f32,
    weights: Qwen35FullAttentionWeights,
    cache: KvCache,
    attention_scratch: GroupedQueryAttentionScratch,
    query_gate_scratch: Zeroizing<Vec<f32>>,
    key_projection_scratch: Zeroizing<Vec<f32>>,
    value_projection_scratch: Zeroizing<Vec<f32>>,
    query_scratch: Zeroizing<Vec<f32>>,
    gate_scratch: Zeroizing<Vec<f32>>,
    key_scratch: Zeroizing<Vec<f32>>,
}

impl Qwen35FullAttentionBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        query_heads: usize,
        key_value_heads: usize,
        head_dimension: usize,
        rotary_dimensions: usize,
        rotary_theta: f32,
        norm_epsilon: f32,
        maximum_context: usize,
        weights: Qwen35FullAttentionWeights,
    ) -> CoreResult<Self> {
        let query_width = query_heads.checked_mul(head_dimension);
        let query_gate_width = query_width.and_then(|width| width.checked_mul(2));
        let key_value_width = key_value_heads.checked_mul(head_dimension);
        if hidden_size == 0
            || query_heads == 0
            || query_heads > 64
            || key_value_heads == 0
            || key_value_heads > 32
            || !query_heads.is_multiple_of(key_value_heads)
            || head_dimension == 0
            || head_dimension > 1024
            || rotary_dimensions == 0
            || rotary_dimensions > head_dimension
            || !rotary_dimensions.is_multiple_of(2)
            || !rotary_theta.is_finite()
            || rotary_theta <= 1.0
            || !norm_epsilon.is_finite()
            || norm_epsilon <= 0.0
            || query_gate_width != Some(weights.query_gate_projection.rows())
            || weights.query_gate_projection.columns() != hidden_size
            || key_value_width != Some(weights.key_projection.rows())
            || weights.key_projection.columns() != hidden_size
            || key_value_width != Some(weights.value_projection.rows())
            || weights.value_projection.columns() != hidden_size
            || query_width != Some(weights.output_projection.columns())
            || weights.output_projection.rows() != hidden_size
            || weights.query_norm_weight.len() != head_dimension
            || weights.key_norm_weight.len() != head_dimension
            || weights
                .query_norm_weight
                .iter()
                .chain(&weights.key_norm_weight)
                .any(|value| !value.is_finite())
        {
            return Err(CoreError::Model(
                "Qwen full-attention weights do not match the declared layer geometry".into(),
            ));
        }
        let cache = KvCache::new(key_value_heads, head_dimension, maximum_context)?;
        let attention_scratch = GroupedQueryAttentionScratch::new(
            query_heads,
            key_value_heads,
            head_dimension,
            maximum_context,
        )?;
        let mut mrope_sections = [0_usize; 3];
        for pair in 0..rotary_dimensions / 2 {
            mrope_sections[pair % 3] += 1;
        }
        let mrope_denominators =
            qwen35_mrope_denominators(rotary_dimensions, rotary_theta, mrope_sections)?;
        let mrope_angles = vec![(0.0, 0.0); rotary_dimensions / 2];
        let query_width = query_width.expect("validated Qwen query width");
        let key_value_width = key_value_width.expect("validated Qwen key/value width");
        let query_gate_width = query_gate_width.expect("validated Qwen query/gate width");
        Ok(Self {
            hidden_size,
            query_heads,
            key_value_heads,
            head_dimension,
            rotary_dimensions,
            mrope_denominators,
            mrope_angles,
            norm_epsilon,
            weights,
            cache,
            attention_scratch,
            query_gate_scratch: Zeroizing::new(vec![0.0; query_gate_width]),
            key_projection_scratch: Zeroizing::new(vec![0.0; key_value_width]),
            value_projection_scratch: Zeroizing::new(vec![0.0; key_value_width]),
            query_scratch: Zeroizing::new(vec![0.0; query_width]),
            gate_scratch: Zeroizing::new(vec![0.0; query_width]),
            key_scratch: Zeroizing::new(vec![0.0; key_value_width]),
        })
    }

    /// Construct the pinned 4B full-attention geometry. The Qwen config uses
    /// partial rotary on 64 of each 256-dimensional head and theta 10,000,000.
    pub fn for_qwen35_4b(
        norm_epsilon: f32,
        maximum_context: usize,
        weights: Qwen35FullAttentionWeights,
    ) -> CoreResult<Self> {
        Self::new(
            HIDDEN_SIZE,
            16,
            4,
            256,
            64,
            10_000_000.0,
            norm_epsilon,
            maximum_context,
            weights,
        )
    }

    fn reserve_cache_positions(&mut self, positions: usize) -> CoreResult<()> {
        self.cache.reserve_positions(positions)
    }

    /// Apply one normalized hidden vector and return its attention projection.
    /// A failure after appending the new key/value invalidates the entire cache
    /// so later tokens cannot continue from a partially processed position.
    pub fn step(&mut self, normalized_hidden: &[f32]) -> CoreResult<Vec<f32>> {
        let position = self.cache.context_length() as u64;
        self.step_with_mrope_positions(normalized_hidden, [position; 3])
    }

    /// Apply one hidden vector using the pinned temporal/height/width position
    /// triplet. Text-only callers use `step`, where all three axes are equal.
    pub fn step_with_mrope_positions(
        &mut self,
        normalized_hidden: &[f32],
        positions: [u64; 3],
    ) -> CoreResult<Vec<f32>> {
        let mut output = vec![0.0; self.hidden_size];
        self.step_with_mrope_positions_into(normalized_hidden, positions, &mut output)?;
        Ok(output)
    }

    fn step_with_mrope_positions_into(
        &mut self,
        normalized_hidden: &[f32],
        positions: [u64; 3],
        output: &mut [f32],
    ) -> CoreResult<()> {
        let result = self.step_inner(normalized_hidden, positions, output);
        self.clear_step_scratch();
        if result.is_err() {
            output.zeroize();
        }
        result
    }

    fn step_inner(
        &mut self,
        normalized_hidden: &[f32],
        positions: [u64; 3],
        output: &mut [f32],
    ) -> CoreResult<()> {
        if normalized_hidden.len() != self.hidden_size
            || output.len() != self.hidden_size
            || normalized_hidden.iter().any(|value| !value.is_finite())
        {
            return Err(CoreError::Model(
                "Qwen full-attention hidden state has the wrong shape or non-finite values".into(),
            ));
        }
        prepare_qwen35_mrope_angles(&mut self.mrope_angles, &self.mrope_denominators, positions)?;
        self.weights
            .query_gate_projection
            .project_into(normalized_hidden, &mut self.query_gate_scratch)?;
        self.weights
            .key_projection
            .project_into(normalized_hidden, &mut self.key_projection_scratch)?;
        self.weights
            .value_projection
            .project_into(normalized_hidden, &mut self.value_projection_scratch)?;
        for head in 0..self.query_heads {
            let start = head * self.head_dimension * 2;
            let raw_query = &self.query_gate_scratch[start..start + self.head_dimension];
            let raw_gate = &self.query_gate_scratch
                [start + self.head_dimension..start + self.head_dimension * 2];
            let query_range = head * self.head_dimension..(head + 1) * self.head_dimension;
            rms_norm_zero_centered_into(
                raw_query,
                &self.weights.query_norm_weight,
                self.norm_epsilon,
                &mut self.query_scratch[query_range.clone()],
            )?;
            rotary_qwen35_mrope_with_angles(
                &mut self.query_scratch[query_range],
                self.rotary_dimensions,
                &self.mrope_angles,
            )?;
            self.gate_scratch[head * self.head_dimension..(head + 1) * self.head_dimension]
                .copy_from_slice(raw_gate);
        }

        for head in 0..self.key_value_heads {
            let key_range = head * self.head_dimension..(head + 1) * self.head_dimension;
            rms_norm_zero_centered_into(
                &self.key_projection_scratch[key_range.clone()],
                &self.weights.key_norm_weight,
                self.norm_epsilon,
                &mut self.key_scratch[key_range.clone()],
            )?;
            rotary_qwen35_mrope_with_angles(
                &mut self.key_scratch[key_range],
                self.rotary_dimensions,
                &self.mrope_angles,
            )?;
        }

        self.cache
            .append(&self.key_scratch, &self.value_projection_scratch)?;
        if let Err(error) = grouped_query_attention_into(
            &self.query_scratch,
            self.query_heads,
            &self.cache,
            &mut self.attention_scratch,
        ) {
            self.cache.clear();
            return Err(error);
        }
        let projected = {
            let gated = self.attention_scratch.output_mut();
            for (head, gate_head) in self
                .gate_scratch
                .chunks_exact(self.head_dimension)
                .enumerate()
            {
                let start = head * self.head_dimension;
                for dimension in 0..self.head_dimension {
                    gated[start + dimension] *= stable_sigmoid(gate_head[dimension]);
                }
            }
            self.weights.output_projection.project_into(gated, output)
        };
        self.attention_scratch.clear_output();
        match projected {
            Ok(()) => Ok(()),
            Err(error) => {
                self.cache.clear();
                Err(error)
            }
        }
    }

    pub fn context_length(&self) -> usize {
        self.cache.context_length()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.clear_step_scratch();
    }

    fn clear_step_scratch(&mut self) {
        self.attention_scratch.clear();
        self.query_gate_scratch.as_mut_slice().zeroize();
        self.key_projection_scratch.as_mut_slice().zeroize();
        self.value_projection_scratch.as_mut_slice().zeroize();
        self.query_scratch.as_mut_slice().zeroize();
        self.gate_scratch.as_mut_slice().zeroize();
        self.key_scratch.as_mut_slice().zeroize();
    }
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// Qwen gated MLP with caller-owned projection outputs and zeroizing scratch.
pub struct Qwen35Mlp {
    hidden_size: usize,
    gate_projection: Qwen35ProjectionMatrix,
    up_projection: Qwen35ProjectionMatrix,
    down_projection: Qwen35ProjectionMatrix,
    gate_scratch: Zeroizing<Vec<f32>>,
    up_scratch: Zeroizing<Vec<f32>>,
}

impl Qwen35Mlp {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        gate_projection: Qwen35ProjectionMatrix,
        up_projection: Qwen35ProjectionMatrix,
        down_projection: Qwen35ProjectionMatrix,
    ) -> CoreResult<Self> {
        if hidden_size == 0
            || intermediate_size == 0
            || gate_projection.rows() != intermediate_size
            || gate_projection.columns() != hidden_size
            || up_projection.rows() != intermediate_size
            || up_projection.columns() != hidden_size
            || down_projection.rows() != hidden_size
            || down_projection.columns() != intermediate_size
        {
            return Err(CoreError::Model(
                "Qwen MLP projections do not match the declared layer geometry".into(),
            ));
        }
        Ok(Self {
            hidden_size,
            gate_projection,
            up_projection,
            down_projection,
            gate_scratch: Zeroizing::new(vec![0.0; intermediate_size]),
            up_scratch: Zeroizing::new(vec![0.0; intermediate_size]),
        })
    }

    fn step_into(&mut self, hidden: &[f32], output: &mut [f32]) -> CoreResult<()> {
        if hidden.len() != self.hidden_size
            || output.len() != self.hidden_size
            || hidden.iter().any(|value| !value.is_finite())
        {
            output.fill(0.0);
            self.clear_scratch();
            return Err(CoreError::Model(
                "Qwen MLP hidden or output buffer has the wrong shape or non-finite values".into(),
            ));
        }
        let result = (|| {
            self.gate_projection
                .project_into(hidden, &mut self.gate_scratch)?;
            self.up_projection
                .project_into(hidden, &mut self.up_scratch)?;
            silu(&mut self.gate_scratch)?;
            for (activation, value) in self.gate_scratch.iter_mut().zip(self.up_scratch.iter()) {
                *activation *= *value;
                if !activation.is_finite() {
                    return Err(CoreError::Model(
                        "Qwen MLP activation produced a non-finite value".into(),
                    ));
                }
            }
            self.down_projection
                .project_into(&self.gate_scratch, output)
        })();
        self.clear_scratch();
        if result.is_err() {
            output.fill(0.0);
        }
        result
    }

    fn clear_scratch(&mut self) {
        self.gate_scratch.fill(0.0);
        self.up_scratch.fill(0.0);
    }
}

/// One complete stateful Qwen3.5 decoder layer. Attention state advances only
/// with the token that is returned; if normalization or the MLP fails after
/// attention advanced, the layer drops its attention prefix rather than leave
/// a cache that no longer represents a complete hidden-state sequence.
pub struct Qwen35DecoderLayer {
    hidden_size: usize,
    norm_epsilon: f32,
    input_norm_weight: Vec<f32>,
    post_attention_norm_weight: Vec<f32>,
    token_mixer: Qwen35TokenMixer,
    mlp: Qwen35Mlp,
    normalization_scratch: Zeroizing<Vec<f32>>,
    feed_forward_scratch: Zeroizing<Vec<f32>>,
}

pub enum Qwen35TokenMixer {
    Linear(Qwen35LinearAttentionBlock),
    Full(Qwen35FullAttentionBlock),
}

impl Qwen35TokenMixer {
    fn hidden_size(&self) -> usize {
        match self {
            Self::Linear(block) => block.hidden_size,
            Self::Full(block) => block.hidden_size,
        }
    }

    fn step_into(
        &mut self,
        hidden: &[f32],
        positions: Option<[u64; 3]>,
        output: &mut [f32],
    ) -> CoreResult<()> {
        let hidden_size = self.hidden_size();
        if output.len() != hidden_size {
            output.zeroize();
            return Err(CoreError::Model(
                "Qwen token-mixer output buffer has the wrong width".into(),
            ));
        }
        let result = match self {
            Self::Linear(block) => block.step_into(hidden, output),
            Self::Full(block) => {
                let positions = positions.unwrap_or_else(|| {
                    let position = block.context_length() as u64;
                    [position; 3]
                });
                block.step_with_mrope_positions_into(hidden, positions, output)
            }
        };
        if result.is_err() {
            output.zeroize();
        }
        result
    }

    fn clear(&mut self) {
        match self {
            Self::Linear(block) => block.clear(),
            Self::Full(block) => block.clear(),
        }
    }

    fn reserve_cache_positions(&mut self, positions: usize) -> CoreResult<()> {
        match self {
            Self::Linear(_) => Ok(()),
            Self::Full(block) => block.reserve_cache_positions(positions),
        }
    }

    fn is_full_attention(&self) -> bool {
        matches!(self, Self::Full(_))
    }
}

impl Qwen35DecoderLayer {
    pub fn new(
        hidden_size: usize,
        norm_epsilon: f32,
        input_norm_weight: Vec<f32>,
        post_attention_norm_weight: Vec<f32>,
        token_mixer: Qwen35TokenMixer,
        mlp: Qwen35Mlp,
    ) -> CoreResult<Self> {
        if hidden_size == 0
            || !norm_epsilon.is_finite()
            || norm_epsilon <= 0.0
            || input_norm_weight.len() != hidden_size
            || post_attention_norm_weight.len() != hidden_size
            || input_norm_weight
                .iter()
                .chain(&post_attention_norm_weight)
                .any(|value| !value.is_finite())
            || token_mixer.hidden_size() != hidden_size
            || mlp.hidden_size != hidden_size
        {
            return Err(CoreError::Model(
                "Qwen decoder layer components do not match its hidden width".into(),
            ));
        }
        Ok(Self {
            hidden_size,
            norm_epsilon,
            input_norm_weight,
            post_attention_norm_weight,
            token_mixer,
            mlp,
            normalization_scratch: Zeroizing::new(vec![0.0; hidden_size]),
            feed_forward_scratch: Zeroizing::new(vec![0.0; hidden_size]),
        })
    }

    pub fn step(&mut self, hidden: &[f32]) -> CoreResult<Vec<f32>> {
        let mut output = vec![0.0; self.hidden_size];
        self.step_internal(hidden, None, &mut output)?;
        Ok(output)
    }

    /// Apply one text or multimodal hidden vector with its Qwen MRoPE axes.
    /// Linear-attention layers do not consume rotary positions.
    pub fn step_with_mrope_positions(
        &mut self,
        hidden: &[f32],
        positions: [u64; 3],
    ) -> CoreResult<Vec<f32>> {
        let mut output = vec![0.0; self.hidden_size];
        self.step_internal(hidden, Some(positions), &mut output)?;
        Ok(output)
    }

    fn step_internal(
        &mut self,
        hidden: &[f32],
        positions: Option<[u64; 3]>,
        output: &mut [f32],
    ) -> CoreResult<()> {
        if hidden.len() != self.hidden_size
            || output.len() != self.hidden_size
            || hidden.iter().any(|value| !value.is_finite())
        {
            output.zeroize();
            self.clear_scratch();
            return Err(CoreError::Model(
                "Qwen decoder hidden state has the wrong shape or non-finite values".into(),
            ));
        }
        let result = self.step_complete(hidden, positions, output);
        if result.is_err() {
            self.token_mixer.clear();
            output.zeroize();
        }
        result
    }

    fn step_complete(
        &mut self,
        hidden: &[f32],
        positions: Option<[u64; 3]>,
        output: &mut [f32],
    ) -> CoreResult<()> {
        let result = (|| {
            rms_norm_zero_centered_into(
                hidden,
                &self.input_norm_weight,
                self.norm_epsilon,
                &mut self.normalization_scratch,
            )?;
            self.token_mixer
                .step_into(&self.normalization_scratch, positions, output)?;
            add_residual(output, hidden)?;
            rms_norm_zero_centered_into(
                output,
                &self.post_attention_norm_weight,
                self.norm_epsilon,
                &mut self.normalization_scratch,
            )?;
            self.mlp
                .step_into(&self.normalization_scratch, &mut self.feed_forward_scratch)?;
            add_residual(output, &self.feed_forward_scratch)
        })();
        self.clear_scratch();
        if result.is_err() {
            output.zeroize();
        }
        result
    }

    pub fn clear(&mut self) {
        self.token_mixer.clear();
        self.clear_scratch();
    }

    fn clear_scratch(&mut self) {
        self.normalization_scratch.as_mut_slice().zeroize();
        self.feed_forward_scratch.as_mut_slice().zeroize();
    }

    fn reserve_cache_positions(&mut self, positions: usize) -> CoreResult<()> {
        self.token_mixer.reserve_cache_positions(positions)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn uses_full_attention(&self) -> bool {
        self.token_mixer.is_full_attention()
    }
}

/// Stateful Sage text decoder using tied token embeddings, portable reference
/// math, and qualified architecture-specific fast kernels. Each successful
/// call consumes one input token, advances every layer cache exactly once, and
/// returns next-token logits. A mid-layer failure clears every layer so caches
/// cannot retain different token prefixes. This remains a reference path; it
/// does not load or admit model packages.
pub struct Qwen35TextDecoder {
    token_embeddings: Qwen35ProjectionMatrix,
    final_norm_weight: Vec<f32>,
    layers: Vec<Qwen35DecoderLayer>,
    hidden_size: usize,
    vocabulary_size: usize,
    norm_epsilon: f32,
    maximum_context: usize,
    context_length: usize,
    next_mrope_position: u64,
}

/// One contiguous image-pad run whose token embeddings are replaced by the
/// corresponding package-bound vision rows during multimodal prefill.
pub struct Qwen35EmbeddedSpan<'a> {
    pub token_start: usize,
    pub token_id: u32,
    pub embeddings: &'a [f32],
    pub positions: &'a [[u64; 3]],
}

/// Explicit position IDs and image embedding replacements for one prompt.
/// The prompt token count still includes one slot for every image embedding.
pub struct Qwen35EmbeddedPrompt<'a> {
    pub token_ids: &'a [u32],
    pub positions: &'a [[u64; 3]],
    pub spans: &'a [Qwen35EmbeddedSpan<'a>],
}

impl Qwen35TextDecoder {
    pub fn new(
        token_embeddings: Qwen35ProjectionMatrix,
        final_norm_weight: Vec<f32>,
        layers: Vec<Qwen35DecoderLayer>,
        norm_epsilon: f32,
        maximum_context: usize,
    ) -> CoreResult<Self> {
        let hidden_size = token_embeddings.columns();
        let vocabulary_size = token_embeddings.rows();
        if hidden_size == 0
            || vocabulary_size == 0
            || vocabulary_size > VOCABULARY_SIZE
            || final_norm_weight.len() != hidden_size
            || final_norm_weight.iter().any(|value| !value.is_finite())
            || layers.is_empty()
            || layers.len() > 32
            || layers
                .iter()
                .any(|layer| layer.hidden_size() != hidden_size)
            || !norm_epsilon.is_finite()
            || norm_epsilon <= 0.0
            || maximum_context == 0
            || maximum_context > SAGE_CONTEXT_LIMIT as usize
        {
            return Err(CoreError::Model(
                "Qwen text decoder components exceed the admitted profile".into(),
            ));
        }
        Ok(Self {
            token_embeddings,
            final_norm_weight,
            layers,
            hidden_size,
            vocabulary_size,
            norm_epsilon,
            maximum_context,
            context_length: 0,
            next_mrope_position: 0,
        })
    }

    /// Validate the complete pinned text stack before a weight loader can
    /// advertise Qwen3.5-4B generation as available.
    pub fn for_qwen35_4b(
        token_embeddings: Qwen35ProjectionMatrix,
        final_norm_weight: Vec<f32>,
        layers: Vec<Qwen35DecoderLayer>,
        norm_epsilon: f32,
        maximum_context: usize,
    ) -> CoreResult<Self> {
        if token_embeddings.rows() != VOCABULARY_SIZE
            || token_embeddings.columns() != HIDDEN_SIZE
            || final_norm_weight.len() != HIDDEN_SIZE
            || layers.len() != 32
            || layers.iter().enumerate().any(|(index, layer)| {
                layer.hidden_size() != HIDDEN_SIZE
                    || layer.uses_full_attention() != (index % 4 == 3)
            })
        {
            return Err(CoreError::Model(
                "Qwen3.5-4B text decoder requires all 32 pinned layers and tied embeddings".into(),
            ));
        }
        Self::new(
            token_embeddings,
            final_norm_weight,
            layers,
            norm_epsilon,
            maximum_context,
        )
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn context_length(&self) -> usize {
        self.context_length
    }

    pub fn next_mrope_position(&self) -> u64 {
        self.next_mrope_position
    }

    pub fn maximum_context(&self) -> usize {
        self.maximum_context
    }

    fn reserve_cache_positions<C>(&mut self, positions: usize, cancelled: &mut C) -> CoreResult<()>
    where
        C: FnMut() -> bool,
    {
        for layer in &mut self.layers {
            if cancelled() {
                return Err(CoreError::Cancelled);
            }
            layer.reserve_cache_positions(positions)?;
        }
        Ok(())
    }

    /// Consume one token and return the tied-embedding next-token logits. A
    /// rejected out-of-range token or full context leaves the valid prefix
    /// untouched; numerical or layer failures clear the full decoder state.
    pub fn step_token(&mut self, token_id: u32) -> CoreResult<Vec<f32>> {
        self.step_token_controlled(token_id, || false)
    }

    /// Cooperative cancellation is checked at each layer boundary and before
    /// the final vocabulary projection. A cancelled partial token invalidates
    /// every layer cache because their sequence prefixes would otherwise
    /// disagree.
    pub fn step_token_controlled<C>(&mut self, token_id: u32, cancelled: C) -> CoreResult<Vec<f32>>
    where
        C: FnMut() -> bool,
    {
        let position = self.next_mrope_position;
        self.step_token_with_mrope_positions_controlled(token_id, [position; 3], cancelled)
    }

    pub fn step_token_with_mrope_positions(
        &mut self,
        token_id: u32,
        positions: [u64; 3],
    ) -> CoreResult<Vec<f32>> {
        self.step_token_with_mrope_positions_controlled(token_id, positions, || false)
    }

    pub fn step_token_with_mrope_positions_controlled<C>(
        &mut self,
        token_id: u32,
        positions: [u64; 3],
        cancelled: C,
    ) -> CoreResult<Vec<f32>>
    where
        C: FnMut() -> bool,
    {
        let hidden =
            self.step_token_hidden_with_mrope_positions_controlled(token_id, positions, cancelled)?;
        self.project_step_output(&hidden)
    }

    fn step_token_hidden_controlled<C>(
        &mut self,
        token_id: u32,
        cancelled: C,
    ) -> CoreResult<Zeroizing<Vec<f32>>>
    where
        C: FnMut() -> bool,
    {
        let position = self.next_mrope_position;
        self.step_token_hidden_with_mrope_positions_controlled(token_id, [position; 3], cancelled)
    }

    fn step_token_hidden_with_mrope_positions_controlled<C>(
        &mut self,
        token_id: u32,
        positions: [u64; 3],
        cancelled: C,
    ) -> CoreResult<Zeroizing<Vec<f32>>>
    where
        C: FnMut() -> bool,
    {
        let token_index = usize::try_from(token_id)
            .ok()
            .filter(|token| *token < self.vocabulary_size)
            .ok_or_else(|| CoreError::Model("Qwen token ID is outside the vocabulary".into()))?;
        if self.context_length >= self.maximum_context {
            return Err(CoreError::Model(
                "Qwen text context exceeded Sage's admitted limit".into(),
            ));
        }
        let hidden = match self.token_embeddings.row(token_index) {
            Ok(hidden) => Zeroizing::new(hidden),
            Err(error) => {
                self.clear();
                return Err(error);
            }
        };
        self.step_embedding_hidden_with_mrope_positions_controlled(&hidden, positions, cancelled)
    }

    /// Consume one already projected text-width embedding with its multimodal
    /// temporal/height/width positions. The caller owns image-feature creation;
    /// this method still advances the same bounded causal decoder state.
    pub fn step_embedded_with_mrope_positions(
        &mut self,
        embedding: &[f32],
        positions: [u64; 3],
    ) -> CoreResult<Vec<f32>> {
        self.step_embedding_with_mrope_positions_controlled(embedding, positions, || false)
    }

    pub fn step_embedding_with_mrope_positions_controlled<C>(
        &mut self,
        embedding: &[f32],
        positions: [u64; 3],
        cancelled: C,
    ) -> CoreResult<Vec<f32>>
    where
        C: FnMut() -> bool,
    {
        let hidden = self.step_embedding_hidden_with_mrope_positions_controlled(
            embedding, positions, cancelled,
        )?;
        self.project_step_output(&hidden)
    }

    fn step_embedding_hidden_with_mrope_positions_controlled<C>(
        &mut self,
        embedding: &[f32],
        positions: [u64; 3],
        mut cancelled: C,
    ) -> CoreResult<Zeroizing<Vec<f32>>>
    where
        C: FnMut() -> bool,
    {
        if embedding.len() != self.hidden_size || embedding.iter().any(|value| !value.is_finite()) {
            return Err(CoreError::Model(
                "Qwen embedded input has the wrong width or non-finite values".into(),
            ));
        }
        if positions
            .iter()
            .any(|position| *position >= QWEN35_4B_MAX_POSITION_EMBEDDINGS)
        {
            return Err(CoreError::Model(
                "Qwen multimodal position exceeds the checkpoint limit".into(),
            ));
        }
        if self.context_length >= self.maximum_context {
            return Err(CoreError::Model(
                "Qwen text context exceeded Sage's admitted limit".into(),
            ));
        }
        let result = (|| {
            if cancelled() {
                return Err(CoreError::Cancelled);
            }
            let mut hidden = Zeroizing::new(embedding.to_vec());
            let mut next_hidden = Zeroizing::new(vec![0.0; self.hidden_size]);
            for layer in &mut self.layers {
                if cancelled() {
                    return Err(CoreError::Cancelled);
                }
                layer.step_internal(&hidden, Some(positions), &mut next_hidden)?;
                std::mem::swap(&mut hidden, &mut next_hidden);
            }
            if cancelled() {
                return Err(CoreError::Cancelled);
            }
            rms_norm_zero_centered_into(
                &hidden,
                &self.final_norm_weight,
                self.norm_epsilon,
                &mut next_hidden,
            )?;
            Ok(next_hidden)
        })();
        match result {
            Ok(normalized_hidden) => {
                self.context_length += 1;
                self.next_mrope_position = positions.into_iter().max().unwrap_or(0) + 1;
                Ok(normalized_hidden)
            }
            Err(error) => {
                self.clear();
                Err(error)
            }
        }
    }

    fn project_step_output(&mut self, normalized_hidden: &[f32]) -> CoreResult<Vec<f32>> {
        let result = self
            .token_embeddings
            .project(normalized_hidden)
            .and_then(|logits| {
                if logits.len() != self.vocabulary_size {
                    return Err(CoreError::Model(
                        "Qwen tied output projection changed its vocabulary size".into(),
                    ));
                }
                Ok(logits)
            });
        if result.is_err() {
            self.clear();
        }
        result
    }

    fn project_generation_logits(
        &self,
        normalized_hidden: &[f32],
        allowed_token_ids: Option<&[u32]>,
        selected_rows: &mut Vec<usize>,
        output: &mut [f32],
    ) -> CoreResult<bool> {
        if output.len() != self.vocabulary_size {
            output.fill(0.0);
            return Err(CoreError::Model(
                "Qwen generation output buffer does not match the vocabulary".into(),
            ));
        }
        if let Some(allowed) = allowed_token_ids {
            if allowed.is_empty() {
                output.fill(0.0);
                return Err(CoreError::Model(
                    "Qwen constrained vocabulary is empty or out of range".into(),
                ));
            }
            selected_rows.clear();
            let project_sparse = allowed.len() <= self.vocabulary_size / 2;
            let mut previous = None;
            for &token_id in allowed {
                if token_id as usize >= self.vocabulary_size
                    || previous.is_some_and(|value| token_id <= value)
                {
                    selected_rows.clear();
                    output.fill(0.0);
                    return Err(CoreError::Model(
                        "Qwen constrained vocabulary is unordered or out of range".into(),
                    ));
                }
                if project_sparse {
                    selected_rows.push(token_id as usize);
                }
                previous = Some(token_id);
            }
            if project_sparse {
                self.token_embeddings.project_selected_rows_into(
                    normalized_hidden,
                    selected_rows,
                    &mut output[..selected_rows.len()],
                )?;
                return Ok(true);
            }
        }
        selected_rows.clear();
        self.token_embeddings
            .project_into(normalized_hidden, output)?;
        Ok(false)
    }

    /// Prefill a complete prompt while retaining only the final logits. Input
    /// validation occurs before cache mutation, and generation remains capped
    /// by the same 8K context bound.
    pub fn prefill(&mut self, token_ids: &[u32]) -> CoreResult<Vec<f32>> {
        self.prefill_controlled(token_ids, || false)
    }

    pub fn prefill_controlled<C>(&mut self, token_ids: &[u32], cancelled: C) -> CoreResult<Vec<f32>>
    where
        C: FnMut() -> bool,
    {
        let hidden = self.prefill_hidden_controlled(token_ids, cancelled)?;
        self.project_step_output(&hidden)
    }

    fn prefill_hidden_controlled<C>(
        &mut self,
        token_ids: &[u32],
        mut cancelled: C,
    ) -> CoreResult<Zeroizing<Vec<f32>>>
    where
        C: FnMut() -> bool,
    {
        let resulting_length = self
            .context_length
            .checked_add(token_ids.len())
            .filter(|length| !token_ids.is_empty() && *length <= self.maximum_context)
            .ok_or_else(|| CoreError::Model("Qwen prompt exceeds the admitted context".into()))?;
        if token_ids
            .iter()
            .any(|token| usize::try_from(*token).map_or(true, |id| id >= self.vocabulary_size))
        {
            return Err(CoreError::Model(
                "Qwen prompt contains a token outside the vocabulary".into(),
            ));
        }
        self.reserve_cache_positions(resulting_length, &mut cancelled)?;
        let mut hidden = None;
        for token in token_ids {
            hidden = Some(self.step_token_hidden_controlled(*token, &mut cancelled)?);
        }
        debug_assert_eq!(self.context_length, resulting_length);
        hidden.ok_or_else(|| CoreError::Model("Qwen prompt is empty".into()))
    }

    /// Prefill text and image embeddings in one causal sequence. Image rows
    /// replace only the exact validated token runs; every token receives its
    /// caller-computed temporal/height/width RoPE position. All spans, shapes,
    /// IDs, positions and bounds are checked before the decoder cache advances.
    pub fn prefill_embedded_prompt_controlled<C>(
        &mut self,
        prompt: &Qwen35EmbeddedPrompt<'_>,
        cancelled: C,
    ) -> CoreResult<Vec<f32>>
    where
        C: FnMut() -> bool,
    {
        let hidden = self.prefill_embedded_prompt_hidden_controlled(prompt, cancelled)?;
        self.project_step_output(&hidden)
    }

    fn prefill_embedded_prompt_hidden_controlled<C>(
        &mut self,
        prompt: &Qwen35EmbeddedPrompt<'_>,
        mut cancelled: C,
    ) -> CoreResult<Zeroizing<Vec<f32>>>
    where
        C: FnMut() -> bool,
    {
        let resulting_length = self
            .context_length
            .checked_add(prompt.token_ids.len())
            .filter(|length| !prompt.token_ids.is_empty() && *length <= self.maximum_context)
            .ok_or_else(|| CoreError::Model("Qwen multimodal prompt exceeds context".into()))?;
        if prompt.positions.len() != prompt.token_ids.len()
            || prompt
                .token_ids
                .iter()
                .any(|token| *token as usize >= self.vocabulary_size)
            || prompt
                .positions
                .iter()
                .flatten()
                .any(|position| *position >= QWEN35_4B_MAX_POSITION_EMBEDDINGS)
            || prompt.spans.len() > 8
        {
            return Err(CoreError::Model(
                "Qwen multimodal prompt has invalid token IDs, positions or span count".into(),
            ));
        }

        let mut previous_end = 0usize;
        for span in prompt.spans {
            let embedded_count = span.positions.len();
            let end = span
                .token_start
                .checked_add(embedded_count)
                .filter(|end| {
                    embedded_count > 0
                        && span.token_start >= previous_end
                        && *end <= prompt.token_ids.len()
                })
                .ok_or_else(|| CoreError::Model("Qwen image embedding span is invalid".into()))?;
            let expected_values = embedded_count
                .checked_mul(self.hidden_size)
                .ok_or_else(|| CoreError::Model("Qwen image embedding size overflow".into()))?;
            if span.embeddings.len() != expected_values
                || span.embeddings.iter().any(|value| !value.is_finite())
                || prompt.token_ids[span.token_start..end]
                    .iter()
                    .any(|token_id| *token_id != span.token_id)
                || span
                    .positions
                    .iter()
                    .flatten()
                    .any(|position| *position >= QWEN35_4B_MAX_POSITION_EMBEDDINGS)
            {
                return Err(CoreError::Model(
                    "Qwen image embedding span does not match its token run".into(),
                ));
            }
            previous_end = end;
        }

        self.reserve_cache_positions(resulting_length, &mut cancelled)?;
        let mut hidden = None;
        let mut span_index = 0usize;
        let mut token_index = 0usize;
        while token_index < prompt.token_ids.len() {
            if let Some(span) = prompt.spans.get(span_index)
                && span.token_start == token_index
            {
                let count = span.positions.len();
                for embedded_index in 0..count {
                    let start = embedded_index * self.hidden_size;
                    let end = start + self.hidden_size;
                    hidden = Some(self.step_embedding_hidden_with_mrope_positions_controlled(
                        &span.embeddings[start..end],
                        span.positions[embedded_index],
                        &mut cancelled,
                    )?);
                }
                token_index += count;
                span_index += 1;
            } else {
                hidden = Some(self.step_token_hidden_with_mrope_positions_controlled(
                    prompt.token_ids[token_index],
                    prompt.positions[token_index],
                    &mut cancelled,
                )?);
                token_index += 1;
            }
        }
        debug_assert_eq!(self.context_length, resulting_length);
        hidden.ok_or_else(|| CoreError::Model("Qwen multimodal prompt is empty".into()))
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.context_length = 0;
        self.next_mrope_position = 0;
    }
}

/// Deterministic greedy selection. A constrained caller supplies strictly
/// increasing token IDs, so selection visits only eligible logits and keeps
/// the lower-token-ID tie break without a per-vocabulary membership lookup.
pub fn greedy_token_from_logits(
    logits: &[f32],
    allowed_token_ids: Option<&[u32]>,
) -> CoreResult<u32> {
    if logits.is_empty() || logits.len() > VOCABULARY_SIZE {
        return Err(CoreError::Model(
            "Token logits or constrained vocabulary are invalid".into(),
        ));
    }
    let mut best: Option<(u32, f32)> = None;
    if let Some(allowed) = allowed_token_ids {
        if allowed.is_empty() {
            return Err(CoreError::Model(
                "Token logits or constrained vocabulary are invalid".into(),
            ));
        }
        let mut previous = None;
        for &token_id in allowed {
            if token_id as usize >= logits.len() || previous.is_some_and(|value| token_id <= value)
            {
                return Err(CoreError::Model(
                    "Token logits or constrained vocabulary are invalid".into(),
                ));
            }
            let logit = logits[token_id as usize];
            if !logit.is_finite() {
                return Err(CoreError::Model(
                    "Token logits contain a non-finite value".into(),
                ));
            }
            if best.is_none_or(|(_, best_logit)| logit > best_logit) {
                best = Some((token_id, logit));
            }
            previous = Some(token_id);
        }
    } else {
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(CoreError::Model(
                "Token logits contain a non-finite value".into(),
            ));
        }
        for (token_id, logit) in logits.iter().copied().enumerate() {
            if best.is_none_or(|(_, best_logit)| logit > best_logit) {
                best = Some((token_id as u32, logit));
            }
        }
    }
    best.map(|(token_id, _)| token_id)
        .ok_or_else(|| CoreError::Model("No token is permitted by the active constraint".into()))
}

/// Greedy selection from logits projected only for the sorted allowed token
/// IDs. This preserves the dense reference path's lower-ID tie break.
pub fn greedy_token_from_allowed_logits(
    logits: &[f32],
    allowed_token_ids: &[u32],
    vocabulary_size: usize,
) -> CoreResult<u32> {
    if logits.is_empty() || logits.len() != allowed_token_ids.len() {
        return Err(CoreError::Model(
            "Sparse token logits do not match the permitted vocabulary".into(),
        ));
    }
    let mut best: Option<(u32, f32)> = None;
    let mut previous = None;
    for (token_id, logit) in allowed_token_ids
        .iter()
        .copied()
        .zip(logits.iter().copied())
    {
        if token_id as usize >= vocabulary_size
            || previous.is_some_and(|value| token_id <= value)
            || !logit.is_finite()
        {
            return Err(CoreError::Model(
                "Sparse token logits do not match the permitted vocabulary".into(),
            ));
        }
        if best.is_none_or(|(_, best_logit)| logit > best_logit) {
            best = Some((token_id, logit));
        }
        previous = Some(token_id);
    }
    best.map(|(token_id, _)| token_id)
        .ok_or_else(|| CoreError::Model("No token is permitted by the active constraint".into()))
}

/// Bounded first-party greedy decoding over Sage's local text path. The caller
/// supplies terminal tokens and a per-prefix callback that fills a reusable
/// sorted allow-list buffer. Callback constraints never bypass the output bound.
/// Decoder caches are cleared on both success and failure so this helper does
/// not retain prompt activations after the response completes.
pub fn generate_greedy<F, C>(
    decoder: &mut Qwen35TextDecoder,
    prompt_token_ids: &[u32],
    stop_token_ids: &BTreeSet<u32>,
    maximum_new_tokens: usize,
    allowed_tokens: F,
    cancelled: C,
) -> CoreResult<Vec<u32>>
where
    F: FnMut(&[u32], &mut Vec<u32>) -> CoreResult<bool>,
    C: FnMut() -> bool,
{
    generate_greedy_with_prefill(
        decoder,
        prompt_token_ids,
        stop_token_ids,
        maximum_new_tokens,
        |decoder, cancelled| decoder.prefill_hidden_controlled(prompt_token_ids, cancelled),
        allowed_tokens,
        cancelled,
    )
}

/// Greedy generation with an interleaved text+embedding prompt. The same
/// bounded JSON/grammar caller may use this after replacing image-pad spans.
pub fn generate_greedy_with_embedded_prompt<F, C>(
    decoder: &mut Qwen35TextDecoder,
    prompt: &Qwen35EmbeddedPrompt<'_>,
    stop_token_ids: &BTreeSet<u32>,
    maximum_new_tokens: usize,
    allowed_tokens: F,
    cancelled: C,
) -> CoreResult<Vec<u32>>
where
    F: FnMut(&[u32], &mut Vec<u32>) -> CoreResult<bool>,
    C: FnMut() -> bool,
{
    generate_greedy_with_prefill(
        decoder,
        prompt.token_ids,
        stop_token_ids,
        maximum_new_tokens,
        |decoder, cancelled| decoder.prefill_embedded_prompt_hidden_controlled(prompt, cancelled),
        allowed_tokens,
        cancelled,
    )
}

fn generate_greedy_with_prefill<F, C, P>(
    decoder: &mut Qwen35TextDecoder,
    prompt_token_ids: &[u32],
    stop_token_ids: &BTreeSet<u32>,
    maximum_new_tokens: usize,
    prefill: P,
    mut allowed_tokens: F,
    mut cancelled: C,
) -> CoreResult<Vec<u32>>
where
    F: FnMut(&[u32], &mut Vec<u32>) -> CoreResult<bool>,
    C: FnMut() -> bool,
    P: FnOnce(&mut Qwen35TextDecoder, &mut C) -> CoreResult<Zeroizing<Vec<f32>>>,
{
    decoder.clear();
    if prompt_token_ids.is_empty()
        || maximum_new_tokens == 0
        || maximum_new_tokens > SAGE_OUTPUT_LIMIT as usize
        || stop_token_ids.is_empty()
        || stop_token_ids
            .iter()
            .any(|token| *token as usize >= decoder.vocabulary_size())
    {
        return Err(CoreError::Model(
            "Greedy generation prompt, stop tokens, or output bound is invalid".into(),
        ));
    }
    let result = (|| {
        let mut final_hidden = prefill(decoder, &mut cancelled)?;
        let mut logits = Zeroizing::new(vec![0.0; decoder.vocabulary_size()]);
        let mut selected_rows = Vec::new();
        let mut allowed_token_ids = Vec::new();
        let mut generated = Vec::with_capacity(maximum_new_tokens);
        for index in 0..maximum_new_tokens {
            if cancelled() {
                return Err(CoreError::Cancelled);
            }
            allowed_token_ids.clear();
            let has_constraints = allowed_tokens(&generated, &mut allowed_token_ids)?;
            let allowed = has_constraints.then_some(allowed_token_ids.as_slice());
            let selected = decoder.project_generation_logits(
                &final_hidden,
                allowed,
                &mut selected_rows,
                &mut logits,
            )?;
            let token = match (selected, allowed) {
                (true, Some(allowed_ids)) => {
                    let sparse_logits = logits.get(..allowed_ids.len()).ok_or_else(|| {
                        CoreError::Model("Sparse token projection is incomplete".into())
                    })?;
                    greedy_token_from_allowed_logits(
                        sparse_logits,
                        allowed_ids,
                        decoder.vocabulary_size(),
                    )?
                }
                (true, None) => {
                    return Err(CoreError::Model(
                        "Sparse token projection has no allow-list".into(),
                    ));
                }
                (false, allowed_ids) => greedy_token_from_logits(&logits, allowed_ids)?,
            };
            if stop_token_ids.contains(&token) {
                return Ok(generated);
            }
            generated.push(token);
            if index + 1 < maximum_new_tokens {
                final_hidden = decoder.step_token_hidden_controlled(token, &mut cancelled)?;
            }
        }
        Ok(generated)
    })();
    decoder.clear();
    result
}

fn add_residual(residual: &mut [f32], update: &[f32]) -> CoreResult<()> {
    if residual.len() != update.len() {
        return Err(CoreError::Model(
            "Qwen residual dimensions do not match".into(),
        ));
    }
    for (value, update) in residual.iter_mut().zip(update) {
        *value += *update;
        if !value.is_finite() {
            return Err(CoreError::Model(
                "Qwen residual addition produced a non-finite value".into(),
            ));
        }
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensorIndex {
    metadata: RawIndexMetadata,
    weight_map: RawWeightMap,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndexMetadata {
    total_size: u64,
}

struct RawWeightMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for RawWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct WeightMapVisitor;

        impl<'de> Visitor<'de> for WeightMapVisitor {
            type Value = RawWeightMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a unique map of Qwen tensor names to shard filenames")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut weights = BTreeMap::new();
                while let Some((name, shard)) = map.next_entry::<String, String>()? {
                    if name.is_empty()
                        || name.len() > 512
                        || name.chars().any(char::is_control)
                        || !matches!(
                            shard.as_str(),
                            "model.safetensors-00001-of-00002.safetensors"
                                | "model.safetensors-00002-of-00002.safetensors"
                        )
                        || weights.insert(name, shard).is_some()
                        || weights.len() > EXPECTED_TENSOR_COUNT
                    {
                        return Err(serde::de::Error::custom(
                            "invalid or duplicate Qwen tensor index entry",
                        ));
                    }
                }
                Ok(RawWeightMap(weights))
            }
        }

        deserializer.deserialize_map(WeightMapVisitor)
    }
}

/// A checked name-to-shard map for the immutable Qwen3.5-4B candidate. This
/// validates the index contract only; a signed package and each shard header
/// must still be verified before weights are loaded.
#[derive(Debug, Clone)]
pub struct Qwen35WeightIndex {
    total_size: u64,
    weight_map: BTreeMap<String, String>,
    tensor_specs: BTreeMap<String, Qwen35TensorSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TensorSpec {
    dtype: crate::safetensors::TensorDType,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35LinearAttentionTensorNames {
    pub a_log: String,
    pub convolution: String,
    pub dt_bias: String,
    pub decay_projection: String,
    pub beta_projection: String,
    pub qkv_projection: String,
    pub gate_projection: String,
    pub norm_weight: String,
    pub output_projection: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35FullAttentionTensorNames {
    pub query_gate_projection: String,
    pub key_projection: String,
    pub value_projection: String,
    pub output_projection: String,
    pub query_norm_weight: String,
    pub key_norm_weight: String,
}

impl Qwen35TensorSpec {
    pub fn dtype(&self) -> crate::safetensors::TensorDType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Qwen35WeightIndex {
    pub fn parse(bytes: &[u8]) -> CoreResult<Self> {
        if bytes.is_empty() || bytes.len() > MAX_INDEX_BYTES {
            return Err(CoreError::Model(
                "Qwen tensor index is empty or exceeds Sage's 1 MiB bound".into(),
            ));
        }
        let index: RawTensorIndex = serde_json::from_slice(bytes)
            .map_err(|_| CoreError::Model("Qwen tensor index is malformed".into()))?;
        let tensor_specs = expected_tensor_specs();
        let expected_weight_bytes = tensor_specs.values().try_fold(0u64, |total, spec| {
            total.checked_add(tensor_byte_size(spec)?)
        });
        let weight_map = index.weight_map.0;
        if index.metadata.total_size != EXPECTED_WEIGHT_BYTES
            || expected_weight_bytes != Some(index.metadata.total_size)
            || weight_map.len() != EXPECTED_TENSOR_COUNT
            || weight_map.keys().cloned().collect::<BTreeSet<_>>()
                != tensor_specs.keys().cloned().collect::<BTreeSet<_>>()
            || weight_map.values().cloned().collect::<BTreeSet<_>>()
                != BTreeSet::from([
                    "model.safetensors-00001-of-00002.safetensors".to_owned(),
                    "model.safetensors-00002-of-00002.safetensors".to_owned(),
                ])
        {
            return Err(CoreError::Model(
                "Qwen tensor index does not match Sage's pinned 4B tensor profile".into(),
            ));
        }
        Ok(Self {
            total_size: index.metadata.total_size,
            weight_map,
            tensor_specs,
        })
    }

    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    pub fn shard_for(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    pub fn shard_files(&self) -> BTreeSet<&str> {
        self.weight_map.values().map(String::as_str).collect()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    pub fn tensor_spec(&self, tensor_name: &str) -> Option<&Qwen35TensorSpec> {
        self.tensor_specs.get(tensor_name)
    }

    /// Resolve the nine pinned tensors for a linear-attention decoder layer.
    /// Full-attention layers and names absent from the validated index fail
    /// closed. The returned names can be used by a later authenticated loader;
    /// they do not themselves admit or load model bytes.
    pub fn linear_attention_tensor_names(
        &self,
        layer_index: usize,
    ) -> CoreResult<Qwen35LinearAttentionTensorNames> {
        if layer_index >= 32 || layer_index % 4 == 3 {
            return Err(CoreError::Model(
                "Qwen decoder layer is outside the pinned linear-attention positions".into(),
            ));
        }
        let prefix = format!("model.language_model.layers.{layer_index}.linear_attn.");
        let names = Qwen35LinearAttentionTensorNames {
            a_log: format!("{prefix}A_log"),
            convolution: format!("{prefix}conv1d.weight"),
            dt_bias: format!("{prefix}dt_bias"),
            decay_projection: format!("{prefix}in_proj_a.weight"),
            beta_projection: format!("{prefix}in_proj_b.weight"),
            qkv_projection: format!("{prefix}in_proj_qkv.weight"),
            gate_projection: format!("{prefix}in_proj_z.weight"),
            norm_weight: format!("{prefix}norm.weight"),
            output_projection: format!("{prefix}out_proj.weight"),
        };
        use crate::safetensors::TensorDType::{BF16, F32};
        let expected = [
            (&names.a_log, F32, vec![32]),
            (&names.convolution, BF16, vec![DELTA_QKV_SIZE, 1, 4]),
            (&names.dt_bias, BF16, vec![32]),
            (&names.decay_projection, BF16, vec![32, HIDDEN_SIZE]),
            (&names.beta_projection, BF16, vec![32, HIDDEN_SIZE]),
            (
                &names.qkv_projection,
                BF16,
                vec![DELTA_QKV_SIZE, HIDDEN_SIZE],
            ),
            (
                &names.gate_projection,
                BF16,
                vec![DELTA_VALUE_SIZE, HIDDEN_SIZE],
            ),
            (&names.norm_weight, F32, vec![QWEN35_4B_HEAD_DIMENSION]),
            (
                &names.output_projection,
                BF16,
                vec![HIDDEN_SIZE, DELTA_VALUE_SIZE],
            ),
        ];
        for (name, dtype, shape) in expected {
            if self.shard_for(name).is_none()
                || self
                    .tensor_spec(name)
                    .is_none_or(|spec| spec.dtype() != dtype || spec.shape() != shape)
            {
                return Err(CoreError::Model(
                    "Qwen linear-attention tensors do not match the pinned index".into(),
                ));
            }
        }
        Ok(names)
    }

    /// Resolve the six pinned tensors for a full-attention decoder layer.
    /// Linear-attention layers and any missing or mismatched tensor fail
    /// closed; this only identifies validated index entries and does not load
    /// or authorize model bytes.
    pub fn full_attention_tensor_names(
        &self,
        layer_index: usize,
    ) -> CoreResult<Qwen35FullAttentionTensorNames> {
        if layer_index >= 32 || layer_index % 4 != 3 {
            return Err(CoreError::Model(
                "Qwen decoder layer is outside the pinned full-attention positions".into(),
            ));
        }
        let prefix = format!("model.language_model.layers.{layer_index}.self_attn.");
        let names = Qwen35FullAttentionTensorNames {
            query_gate_projection: format!("{prefix}q_proj.weight"),
            key_projection: format!("{prefix}k_proj.weight"),
            value_projection: format!("{prefix}v_proj.weight"),
            output_projection: format!("{prefix}o_proj.weight"),
            query_norm_weight: format!("{prefix}q_norm.weight"),
            key_norm_weight: format!("{prefix}k_norm.weight"),
        };
        use crate::safetensors::TensorDType::BF16;
        let expected = [
            (
                &names.query_gate_projection,
                vec![QUERY_PROJECTION_SIZE, HIDDEN_SIZE],
            ),
            (
                &names.key_projection,
                vec![KEY_VALUE_PROJECTION_SIZE, HIDDEN_SIZE],
            ),
            (
                &names.value_projection,
                vec![KEY_VALUE_PROJECTION_SIZE, HIDDEN_SIZE],
            ),
            (
                &names.output_projection,
                vec![HIDDEN_SIZE, ATTENTION_OUTPUT_SIZE],
            ),
            (&names.query_norm_weight, vec![256]),
            (&names.key_norm_weight, vec![256]),
        ];
        for (name, shape) in expected {
            if self.shard_for(name).is_none()
                || self
                    .tensor_spec(name)
                    .is_none_or(|spec| spec.dtype() != BF16 || spec.shape() != shape)
            {
                return Err(CoreError::Model(
                    "Qwen full-attention tensors do not match the pinned index".into(),
                ));
            }
        }
        Ok(names)
    }

    /// Require a shard header to contain exactly the tensor entries assigned
    /// to it by the checked index; neither missing nor hidden tensors pass.
    pub fn validate_shard_names(
        &self,
        shard: &str,
        tensor_names: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> CoreResult<()> {
        if !self.shard_files().contains(shard) {
            return Err(CoreError::Model(
                "Safetensors shard is not present in the pinned index".into(),
            ));
        }
        let expected = self
            .weight_map
            .iter()
            .filter(|(_, assigned_shard)| assigned_shard.as_str() == shard)
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>();
        let mut observed = BTreeSet::new();
        for name in tensor_names {
            if !observed.insert(name.as_ref().to_owned()) {
                return Err(CoreError::Model(
                    "Safetensors shard repeats a tensor name".into(),
                ));
            }
        }
        if expected.len() != observed.len()
            || !expected
                .iter()
                .copied()
                .eq(observed.iter().map(String::as_str))
        {
            return Err(CoreError::Model(
                "Safetensors shard names do not match the pinned tensor index".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_shard<R: Read + Seek>(
        &self,
        shard: &str,
        reader: &crate::safetensors::SafeTensorReader<R>,
    ) -> CoreResult<()> {
        self.validate_shard_names(shard, reader.tensors().keys())?;
        let expected_bytes = self
            .weight_map
            .iter()
            .filter(|(_, assigned_shard)| assigned_shard.as_str() == shard)
            .try_fold(0u64, |total, (name, _)| {
                total.checked_add(tensor_byte_size(self.tensor_specs.get(name)?)?)
            });
        if expected_bytes != Some(reader.data_bytes()) {
            return Err(CoreError::Model(
                "Safetensors shard byte count differs from the pinned tensor profile".into(),
            ));
        }
        for (name, metadata) in reader.tensors() {
            let Some(expected) = self.tensor_specs.get(name) else {
                return Err(CoreError::Model(
                    "Safetensors shard contains an unknown tensor".into(),
                ));
            };
            if metadata.dtype != expected.dtype || metadata.shape != expected.shape {
                return Err(CoreError::Model(format!(
                    "Tensor shape or dtype differs from Sage's pinned profile: {name}"
                )));
            }
        }
        Ok(())
    }

    /// Validate the complete shard header against the pinned name, shape, and
    /// dtype map, then expose bounded tensor reads. This validates layout; it
    /// does not authenticate or admit the containing model package.
    pub fn open_shard<'a, R: Read + Seek>(
        &'a self,
        shard: &str,
        reader: &'a mut crate::safetensors::SafeTensorReader<R>,
    ) -> CoreResult<ValidatedQwen35Shard<'a, R>> {
        self.validate_shard(shard, reader)?;
        Ok(ValidatedQwen35Shard {
            index: self,
            shard: shard.to_owned(),
            reader,
        })
    }
}

/// A schema-checked view of one shard. It borrows the already-open handle so
/// callers can keep model data lazy and load large embeddings in bounded
/// ranges instead of materializing a full 9.3 GB checkpoint in memory.
pub struct ValidatedQwen35Shard<'a, R: Read + Seek> {
    index: &'a Qwen35WeightIndex,
    shard: String,
    reader: &'a mut crate::safetensors::SafeTensorReader<R>,
}

impl<R: Read + Seek> ValidatedQwen35Shard<'_, R> {
    pub fn tensor(&mut self, name: &str) -> CoreResult<Qwen35TensorStream<'_, R>> {
        if self.index.shard_for(name) != Some(self.shard.as_str()) {
            return Err(CoreError::Model(
                "Tensor is not assigned to this validated Qwen shard".into(),
            ));
        }
        let spec = self.index.tensor_spec(name).ok_or_else(|| {
            CoreError::Model("Tensor is absent from the pinned Qwen index".into())
        })?;
        let metadata =
            self.reader.tensors().get(name).ok_or_else(|| {
                CoreError::Model("Tensor is absent from the validated shard".into())
            })?;
        if metadata.shape != spec.shape || metadata.dtype != spec.dtype {
            return Err(CoreError::Model(
                "Tensor metadata changed after shard validation".into(),
            ));
        }
        let total_elements = spec
            .shape
            .iter()
            .try_fold(1_u64, |count, dimension| {
                count.checked_mul(u64::try_from(*dimension).ok()?)
            })
            .ok_or_else(|| CoreError::Model("Tensor element count overflow".into()))?;
        Ok(Qwen35TensorStream {
            reader: self.reader,
            name: name.to_owned(),
            shape: spec.shape.clone(),
            total_elements,
            next_element: 0,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35TensorChunk {
    pub start_element: u64,
    pub values: Vec<f32>,
}

/// Bounded f32 decoding cursor over a tensor whose metadata was validated
/// against Sage's pinned Qwen3.5-4B index.
pub struct Qwen35TensorStream<'a, R: Read + Seek> {
    reader: &'a mut crate::safetensors::SafeTensorReader<R>,
    name: String,
    shape: Vec<usize>,
    total_elements: u64,
    next_element: u64,
}

impl<R: Read + Seek> Qwen35TensorStream<'_, R> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn total_elements(&self) -> u64 {
        self.total_elements
    }

    pub fn remaining_elements(&self) -> u64 {
        self.total_elements.saturating_sub(self.next_element)
    }

    pub fn next_chunk(&mut self, maximum_elements: usize) -> CoreResult<Option<Qwen35TensorChunk>> {
        if maximum_elements == 0 || maximum_elements > MAX_WEIGHT_IMPORT_CHUNK_ELEMENTS {
            return Err(CoreError::Model(
                "Qwen tensor chunk size must be between 1 and Sage's import bound".into(),
            ));
        }
        if self.next_element == self.total_elements {
            return Ok(None);
        }
        let count = usize::try_from(
            self.remaining_elements()
                .min(u64::try_from(maximum_elements).unwrap_or(u64::MAX)),
        )
        .map_err(|_| CoreError::Model("Tensor chunk exceeds platform bounds".into()))?;
        let start_element = self.next_element;
        let values = self
            .reader
            .read_f32_range(&self.name, start_element, count)?;
        self.next_element =
            self.next_element
                .checked_add(u64::try_from(values.len()).map_err(|_| {
                    CoreError::Model("Tensor cursor exceeds platform bounds".into())
                })?)
                .filter(|next| *next <= self.total_elements)
                .ok_or_else(|| CoreError::Model("Tensor cursor overflow".into()))?;
        Ok(Some(Qwen35TensorChunk {
            start_element,
            values,
        }))
    }

    /// Import one indexed rank-2 model tensor into Sage's scalar CPU matrix
    /// representation. The explicit element budget prevents a caller from
    /// accidentally materializing embeddings or other oversized tensors.
    pub fn into_cpu_matrix(mut self, maximum_elements: usize) -> CoreResult<CpuMatrix> {
        if self.shape.len() != 2 || self.shape.contains(&0) || maximum_elements == 0 {
            return Err(CoreError::Model(
                "Qwen CPU matrix import requires a non-empty rank-2 tensor and a positive bound"
                    .into(),
            ));
        }
        let rows = self.shape[0];
        let columns = self.shape[1];
        let element_count = rows
            .checked_mul(columns)
            .filter(|count| *count <= maximum_elements.min(MAX_QWEN_CPU_MATRIX_IMPORT_ELEMENTS))
            .ok_or_else(|| {
                CoreError::Model(
                    "Qwen tensor exceeds the requested or 32-million-element CPU matrix bound"
                        .into(),
                )
            })?;
        let mut values = Zeroizing::new(Vec::with_capacity(element_count));
        while let Some(chunk) = self.next_chunk(MAX_WEIGHT_IMPORT_CHUNK_ELEMENTS)? {
            values.extend_from_slice(&chunk.values);
        }
        if values.len() != element_count {
            return Err(CoreError::Model(
                "Qwen tensor stream ended before the indexed matrix was complete".into(),
            ));
        }
        let values = std::mem::take(&mut *values);
        CpuMatrix::new(rows, columns, values)
    }

    /// Flatten a small validated tensor into f32 for normalization or
    /// recurrent parameters. Rank and allocation remain explicitly bounded.
    pub fn into_f32_vector(mut self, maximum_elements: usize) -> CoreResult<Vec<f32>> {
        if self.shape.is_empty() || self.shape.contains(&0) || maximum_elements == 0 {
            return Err(CoreError::Model(
                "Qwen vector import requires a non-empty tensor and positive bound".into(),
            ));
        }
        let element_count = usize::try_from(self.total_elements)
            .ok()
            .filter(|count| *count <= maximum_elements)
            .ok_or_else(|| {
                CoreError::Model("Qwen vector exceeds the requested element bound".into())
            })?;
        let mut values = Zeroizing::new(Vec::with_capacity(element_count));
        while let Some(chunk) = self.next_chunk(MAX_WEIGHT_IMPORT_CHUNK_ELEMENTS)? {
            values.extend_from_slice(&chunk.values);
        }
        if values.len() != element_count {
            return Err(CoreError::Model(
                "Qwen vector stream ended before the indexed tensor was complete".into(),
            ));
        }
        Ok(std::mem::take(&mut *values))
    }

    /// Stream an indexed rank-2 model tensor directly into Sage's bounded Q4
    /// representation. Unlike `into_cpu_matrix`, this path can admit the
    /// 248,320-by-2,560 tied token embedding matrix without allocating a full
    /// f32 copy. Its caller must still authenticate the model package and
    /// reserve the compressed weight memory before invoking this importer.
    pub fn into_q4_matrix(
        mut self,
        group_size: usize,
        maximum_elements: usize,
    ) -> CoreResult<QuantizedQ4Matrix> {
        if self.shape.len() != 2 || self.shape.contains(&0) || maximum_elements == 0 {
            return Err(CoreError::Model(
                "Qwen Q4 import requires a non-empty rank-2 tensor and an explicit element budget"
                    .into(),
            ));
        }
        let rows = self.shape[0];
        let columns = self.shape[1];
        let element_count = rows
            .checked_mul(columns)
            .filter(|count| *count <= maximum_elements)
            .ok_or_else(|| {
                CoreError::Model("Qwen tensor exceeds the requested Q4 element bound".into())
            })?;
        let mut builder = QuantizedQ4Builder::new(rows, columns, group_size, maximum_elements)?;
        let mut imported = 0usize;
        while let Some(chunk) = self.next_chunk(MAX_WEIGHT_IMPORT_CHUNK_ELEMENTS)? {
            builder.push(&chunk.values)?;
            imported = imported
                .checked_add(chunk.values.len())
                .ok_or_else(|| CoreError::Model("Qwen Q4 import count overflow".into()))?;
        }
        if imported != element_count {
            return Err(CoreError::Model(
                "Qwen tensor stream ended before the Q4 matrix was complete".into(),
            ));
        }
        builder.finish()
    }
}

fn tensor_byte_size(spec: &Qwen35TensorSpec) -> Option<u64> {
    let elements = spec.shape.iter().try_fold(1u64, |total, dimension| {
        total.checked_mul(u64::try_from(*dimension).ok()?)
    })?;
    let bytes_per_element = match spec.dtype {
        crate::safetensors::TensorDType::F32 => 4,
        crate::safetensors::TensorDType::F16 | crate::safetensors::TensorDType::BF16 => 2,
    };
    elements.checked_mul(bytes_per_element)
}

fn expected_tensor_specs() -> BTreeMap<String, Qwen35TensorSpec> {
    use crate::safetensors::TensorDType::{BF16, F32};

    fn add(
        tensors: &mut BTreeMap<String, Qwen35TensorSpec>,
        name: impl Into<String>,
        dtype: crate::safetensors::TensorDType,
        shape: &[usize],
    ) {
        tensors.insert(
            name.into(),
            Qwen35TensorSpec {
                dtype,
                shape: shape.to_vec(),
            },
        );
    }

    let mut tensors = BTreeMap::new();
    add(
        &mut tensors,
        "model.language_model.embed_tokens.weight",
        BF16,
        &[VOCABULARY_SIZE, HIDDEN_SIZE],
    );
    add(
        &mut tensors,
        "model.language_model.norm.weight",
        BF16,
        &[HIDDEN_SIZE],
    );
    for layer in 0..32 {
        let prefix = format!("model.language_model.layers.{layer}.");
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            add(
                &mut tensors,
                format!("{prefix}{name}"),
                BF16,
                &[HIDDEN_SIZE],
            );
        }
        add(
            &mut tensors,
            format!("{prefix}mlp.down_proj.weight"),
            BF16,
            &[HIDDEN_SIZE, INTERMEDIATE_SIZE],
        );
        for name in ["gate_proj", "up_proj"] {
            add(
                &mut tensors,
                format!("{prefix}mlp.{name}.weight"),
                BF16,
                &[INTERMEDIATE_SIZE, HIDDEN_SIZE],
            );
        }
        if layer % 4 == 3 {
            add(
                &mut tensors,
                format!("{prefix}self_attn.q_proj.weight"),
                BF16,
                &[QUERY_PROJECTION_SIZE, HIDDEN_SIZE],
            );
            for name in ["k_proj", "v_proj"] {
                add(
                    &mut tensors,
                    format!("{prefix}self_attn.{name}.weight"),
                    BF16,
                    &[KEY_VALUE_PROJECTION_SIZE, HIDDEN_SIZE],
                );
            }
            add(
                &mut tensors,
                format!("{prefix}self_attn.o_proj.weight"),
                BF16,
                &[HIDDEN_SIZE, ATTENTION_OUTPUT_SIZE],
            );
            for name in ["q_norm.weight", "k_norm.weight"] {
                add(
                    &mut tensors,
                    format!("{prefix}self_attn.{name}"),
                    BF16,
                    &[256],
                );
            }
        } else {
            add(
                &mut tensors,
                format!("{prefix}linear_attn.A_log"),
                F32,
                &[32],
            );
            add(
                &mut tensors,
                format!("{prefix}linear_attn.conv1d.weight"),
                BF16,
                &[DELTA_QKV_SIZE, 1, 4],
            );
            add(
                &mut tensors,
                format!("{prefix}linear_attn.dt_bias"),
                BF16,
                &[32],
            );
            for name in ["in_proj_a", "in_proj_b"] {
                add(
                    &mut tensors,
                    format!("{prefix}linear_attn.{name}.weight"),
                    BF16,
                    &[32, HIDDEN_SIZE],
                );
            }
            add(
                &mut tensors,
                format!("{prefix}linear_attn.in_proj_qkv.weight"),
                BF16,
                &[DELTA_QKV_SIZE, HIDDEN_SIZE],
            );
            add(
                &mut tensors,
                format!("{prefix}linear_attn.in_proj_z.weight"),
                BF16,
                &[DELTA_VALUE_SIZE, HIDDEN_SIZE],
            );
            add(
                &mut tensors,
                format!("{prefix}linear_attn.norm.weight"),
                F32,
                &[128],
            );
            add(
                &mut tensors,
                format!("{prefix}linear_attn.out_proj.weight"),
                BF16,
                &[HIDDEN_SIZE, DELTA_VALUE_SIZE],
            );
        }
    }

    for block in 0..24 {
        let prefix = format!("model.visual.blocks.{block}.");
        for (name, shape) in [
            ("attn.proj.bias", vec![1024]),
            ("attn.proj.weight", vec![1024, 1024]),
            ("attn.qkv.bias", vec![3072]),
            ("attn.qkv.weight", vec![3072, 1024]),
            ("mlp.linear_fc1.bias", vec![4096]),
            ("mlp.linear_fc1.weight", vec![4096, 1024]),
            ("mlp.linear_fc2.bias", vec![1024]),
            ("mlp.linear_fc2.weight", vec![1024, 4096]),
            ("norm1.bias", vec![1024]),
            ("norm1.weight", vec![1024]),
            ("norm2.bias", vec![1024]),
            ("norm2.weight", vec![1024]),
        ] {
            add(&mut tensors, format!("{prefix}{name}"), BF16, &shape);
        }
    }
    for (name, shape) in [
        ("model.visual.merger.linear_fc1.bias", vec![4096]),
        ("model.visual.merger.linear_fc1.weight", vec![4096, 4096]),
        ("model.visual.merger.linear_fc2.bias", vec![HIDDEN_SIZE]),
        (
            "model.visual.merger.linear_fc2.weight",
            vec![HIDDEN_SIZE, 4096],
        ),
        ("model.visual.merger.norm.bias", vec![1024]),
        ("model.visual.merger.norm.weight", vec![1024]),
        ("model.visual.patch_embed.proj.bias", vec![1024]),
        (
            "model.visual.patch_embed.proj.weight",
            vec![1024, 3, 2, 16, 16],
        ),
        ("model.visual.pos_embed.weight", vec![2304, 1024]),
    ] {
        add(&mut tensors, name, BF16, &shape);
    }

    add(&mut tensors, "mtp.fc.weight", BF16, &[HIDDEN_SIZE, 5120]);
    for name in [
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ] {
        add(&mut tensors, name, BF16, &[HIDDEN_SIZE]);
    }
    let prefix = "mtp.layers.0.";
    for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
        add(
            &mut tensors,
            format!("{prefix}{name}"),
            BF16,
            &[HIDDEN_SIZE],
        );
    }
    add(
        &mut tensors,
        format!("{prefix}mlp.down_proj.weight"),
        BF16,
        &[HIDDEN_SIZE, INTERMEDIATE_SIZE],
    );
    for name in ["gate_proj", "up_proj"] {
        add(
            &mut tensors,
            format!("{prefix}mlp.{name}.weight"),
            BF16,
            &[INTERMEDIATE_SIZE, HIDDEN_SIZE],
        );
    }
    add(
        &mut tensors,
        format!("{prefix}self_attn.q_proj.weight"),
        BF16,
        &[QUERY_PROJECTION_SIZE, HIDDEN_SIZE],
    );
    for name in ["k_proj", "v_proj"] {
        add(
            &mut tensors,
            format!("{prefix}self_attn.{name}.weight"),
            BF16,
            &[KEY_VALUE_PROJECTION_SIZE, HIDDEN_SIZE],
        );
    }
    add(
        &mut tensors,
        format!("{prefix}self_attn.o_proj.weight"),
        BF16,
        &[HIDDEN_SIZE, ATTENTION_OUTPUT_SIZE],
    );
    for name in ["q_norm.weight", "k_norm.weight"] {
        add(
            &mut tensors,
            format!("{prefix}self_attn.{name}"),
            BF16,
            &[256],
        );
    }
    tensors
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Qwen35Config {
    model_type: String,
    text_config: Qwen35TextConfig,
    vision_config: Qwen35VisionConfig,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Qwen35TextConfig {
    model_type: String,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    max_position_embeddings: u32,
    full_attention_interval: u32,
    layer_types: Vec<String>,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    rms_norm_eps: f64,
    tie_word_embeddings: bool,
    rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
struct RopeParameters {
    mrope_interleaved: bool,
    mrope_section: Vec<u32>,
    rope_type: String,
    rope_theta: f64,
    partial_rotary_factor: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Qwen35VisionConfig {
    depth: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_heads: u32,
    patch_size: u32,
    temporal_patch_size: u32,
    spatial_merge_size: u32,
    out_hidden_size: u32,
}

impl Qwen35Config {
    pub fn parse(bytes: &[u8]) -> CoreResult<Self> {
        if bytes.is_empty() || bytes.len() > MAX_CONFIG_BYTES {
            return Err(CoreError::Model(
                "Qwen model configuration is empty or exceeds Sage's 1 MiB limit".into(),
            ));
        }
        let config: Self = serde_json::from_slice(bytes)
            .map_err(|_| CoreError::Model("Qwen model configuration is malformed".into()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> CoreResult<()> {
        let text = &self.text_config;
        let rope = &text.rope_parameters;
        let expected_layers: Vec<&str> = (0..32)
            .map(|layer| {
                if layer % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let observed_layers: Vec<&str> = text.layer_types.iter().map(String::as_str).collect();
        let rope_dimension = f64::from(text.head_dim) * rope.partial_rotary_factor;
        let vision = &self.vision_config;
        let valid = self.model_type == "qwen3_5"
            && text.model_type == "qwen3_5_text"
            && text.hidden_size == 2560
            && text.intermediate_size == 9216
            && text.num_hidden_layers == 32
            && text.num_attention_heads == 16
            && text.num_key_value_heads == 4
            && text.head_dim == 256
            && text.vocab_size == 248_320
            && text.max_position_embeddings >= SAGE_CONTEXT_LIMIT
            && text.full_attention_interval == 4
            && observed_layers == expected_layers
            && text.linear_num_key_heads == 16
            && text.linear_num_value_heads == 32
            && text.linear_key_head_dim == 128
            && text.linear_value_head_dim == 128
            && text.linear_conv_kernel_dim == 4
            && text.rms_norm_eps.is_finite()
            && (text.rms_norm_eps - 0.000001).abs() < 1e-12
            && text.tie_word_embeddings
            && rope.mrope_interleaved
            && rope.mrope_section == [11, 11, 10]
            && rope.rope_type == "default"
            && (rope.rope_theta - 10_000_000.0).abs() < 1.0
            && (rope_dimension - 64.0).abs() < 1e-9
            && vision.depth == 24
            && vision.hidden_size == 1024
            && vision.intermediate_size == 4096
            && vision.num_heads == 16
            && vision.patch_size == 16
            && vision.temporal_patch_size == 2
            && vision.spatial_merge_size == 2
            && vision.out_hidden_size == text.hidden_size
            && [
                self.image_token_id,
                self.video_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
            ]
            .into_iter()
            .all(|token| token < text.vocab_size);
        if !valid {
            return Err(CoreError::Model(
                "Model configuration does not match Sage's pinned Qwen3.5-4B architecture profile"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn text(&self) -> &Qwen35TextConfig {
        &self.text_config
    }

    pub fn vision(&self) -> &Qwen35VisionConfig {
        &self.vision_config
    }

    pub fn multimodal_token_ids(&self) -> (u32, u32, u32) {
        (
            self.vision_start_token_id,
            self.image_token_id,
            self.vision_end_token_id,
        )
    }

    /// Sage intentionally admits only the approved short context at first,
    /// even when the checkpoint supports a much longer native context.
    pub fn admitted_context(&self) -> u32 {
        self.text_config
            .max_position_embeddings
            .min(SAGE_CONTEXT_LIMIT)
    }

    pub fn admitted_output(&self) -> u32 {
        SAGE_OUTPUT_LIMIT
    }

    pub fn rms_norm_epsilon(&self) -> f32 {
        self.text_config.rms_norm_eps as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DELTA_QKV_SIZE, Qwen35Config, Qwen35DecoderLayer, Qwen35EmbeddedPrompt, Qwen35EmbeddedSpan,
        Qwen35FullAttentionBlock, Qwen35FullAttentionWeights, Qwen35LinearAttentionBlock,
        Qwen35LinearAttentionState, Qwen35LinearAttentionWeights, Qwen35Mlp,
        Qwen35ProjectionMatrix, Qwen35TensorSpec, Qwen35TextDecoder, Qwen35TokenMixer,
        Qwen35WeightIndex, VOCABULARY_SIZE, generate_greedy, greedy_token_from_allowed_logits,
        greedy_token_from_logits,
    };
    use crate::CoreError;
    use crate::inference_cpu::{CpuMatrix, QuantizedQ4Matrix};
    use crate::safetensors::{SafeTensorReader, TensorDType};
    use std::{collections::BTreeMap, io::Cursor};

    const TEST_SHARD: &str = "model.safetensors-00001-of-00002.safetensors";

    #[test]
    #[ignore = "requires config.json and model.safetensors.index.json from the pinned checkpoint"]
    fn pinned_checkpoint_metadata_matches_closed_qwen_profile() {
        use sha2::{Digest, Sha256};

        fn read_pinned_artifact(variable: &str, expected_digest: &str) -> Vec<u8> {
            let path = std::env::var_os(variable)
                .unwrap_or_else(|| panic!("set {variable} to the pinned Qwen artifact"));
            let bytes = std::fs::read(path).expect("read pinned Qwen metadata artifact");
            let digest = Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            assert_eq!(
                digest, expected_digest,
                "wrong pinned artifact for {variable}"
            );
            bytes
        }

        let config_bytes = read_pinned_artifact(
            "SAGE_QWEN35_CONFIG_JSON",
            "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
        );
        let index_bytes = read_pinned_artifact(
            "SAGE_QWEN35_INDEX_JSON",
            "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
        );

        let config = Qwen35Config::parse(&config_bytes).expect("validate pinned model config");
        assert_eq!(config.admitted_context(), 8_192);
        assert_eq!(config.vision().depth, 24);
        assert_eq!(config.vision().hidden_size, 1_024);
        assert_eq!(config.vision().out_hidden_size, 2_560);
        assert_eq!(config.multimodal_token_ids(), (248_053, 248_056, 248_054));

        let index = Qwen35WeightIndex::parse(&index_bytes).expect("validate pinned weight index");
        assert_eq!(index.total_size(), 9_319_737_856);
        assert_eq!(index.tensor_names().count(), 738);
        assert_eq!(index.shard_files().len(), 2);
        assert_eq!(
            index.shard_for("model.language_model.embed_tokens.weight"),
            Some("model.safetensors-00001-of-00002.safetensors")
        );
        assert_eq!(
            index.shard_for("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
            Some("model.safetensors-00002-of-00002.safetensors")
        );
    }

    #[test]
    fn grouped_q4_projection_adds_only_finite_geometry_matched_bias() {
        let quantized = QuantizedQ4Matrix::encode(1, 4, 4, &[1.0, 2.0, 3.0, 4.0])
            .expect("valid small grouped-Q4 projection");
        let projection = Qwen35ProjectionMatrix::GroupedQ4(quantized);
        let result = projection
            .project_with_bias(&[1.0, 0.0, 0.0, 0.0], &[0.25])
            .expect("valid biased grouped-Q4 projection");
        assert!((result[0] - (8.0 / 7.0 + 0.25)).abs() < 1e-6);
        assert!(
            projection
                .project_with_bias(&[1.0, 0.0, 0.0, 0.0], &[])
                .is_err()
        );
        assert!(
            projection
                .project_with_bias(&[1.0, 0.0, 0.0, 0.0], &[f32::INFINITY])
                .is_err()
        );
    }

    #[test]
    fn grouped_q4_selected_rows_match_dense_tied_output_projection() {
        let values = (0..5 * 7)
            .map(|index| ((index as f32 - 9.0) * 0.13).sin())
            .collect::<Vec<_>>();
        let projection = Qwen35ProjectionMatrix::GroupedQ4(
            QuantizedQ4Matrix::encode(5, 7, 9, &values).expect("small Q4 output matrix"),
        );
        let input = [0.2, -0.5, 0.1, 0.7, -0.3, 0.4, 0.9];
        let dense = projection.project(&input).expect("dense output logits");
        let rows = [4, 1, 3];
        let mut sparse = [0.0; 3];
        projection
            .project_selected_rows_into(&input, &rows, &mut sparse)
            .expect("sparse output logits");
        for (observed, row) in sparse.iter().zip(rows) {
            assert!((*observed - dense[row]).abs() < 2.0e-5);
        }

        assert!(
            projection
                .project_selected_rows_into(&input, &[5], &mut sparse[..1])
                .is_err()
        );
        assert_eq!(sparse[0], 0.0);
    }

    fn tiny_shard() -> SafeTensorReader<Cursor<Vec<u8>>> {
        let mut header =
            br#"{"block.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&header);
        for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        SafeTensorReader::new(Cursor::new(bytes)).expect("valid tiny shard")
    }

    fn tiny_index() -> Qwen35WeightIndex {
        Qwen35WeightIndex {
            total_size: 24,
            weight_map: BTreeMap::from([("block.weight".into(), TEST_SHARD.into())]),
            tensor_specs: BTreeMap::from([(
                "block.weight".into(),
                Qwen35TensorSpec {
                    dtype: TensorDType::F32,
                    shape: vec![2, 3],
                },
            )]),
        }
    }

    #[test]
    fn indexed_weight_import_reads_validated_tensors_in_exact_bounded_ranges() {
        let index = tiny_index();
        let mut reader = tiny_shard();
        let mut shard = index
            .open_shard(TEST_SHARD, &mut reader)
            .expect("shard matches the pinned weight index");
        let mut tensor = shard
            .tensor("block.weight")
            .expect("tensor belongs to the validated shard");
        assert_eq!(tensor.name(), "block.weight");
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.total_elements(), 6);
        assert_eq!(tensor.remaining_elements(), 6);
        assert!(tensor.next_chunk(0).is_err());
        let first = tensor.next_chunk(2).unwrap().unwrap();
        assert_eq!(first.start_element, 0);
        assert_eq!(first.values, vec![1.0, 2.0]);
        let second = tensor.next_chunk(3).unwrap().unwrap();
        assert_eq!(second.start_element, 2);
        assert_eq!(second.values, vec![3.0, 4.0, 5.0]);
        let last = tensor.next_chunk(4).unwrap().unwrap();
        assert_eq!(last.start_element, 5);
        assert_eq!(last.values, vec![6.0]);
        assert_eq!(tensor.remaining_elements(), 0);
        assert!(tensor.next_chunk(1).unwrap().is_none());
    }

    #[test]
    fn indexed_weight_import_builds_a_bounded_cpu_matrix() {
        let index = tiny_index();
        let mut reader = tiny_shard();
        let mut shard = index
            .open_shard(TEST_SHARD, &mut reader)
            .expect("validated tiny shard");
        let matrix = shard
            .tensor("block.weight")
            .expect("indexed matrix")
            .into_cpu_matrix(6)
            .expect("bounded CPU matrix");
        assert_eq!((matrix.rows(), matrix.columns()), (2, 3));
        assert_eq!(matrix.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut reader = tiny_shard();
        let mut shard = index
            .open_shard(TEST_SHARD, &mut reader)
            .expect("validated tiny shard");
        assert!(
            shard
                .tensor("block.weight")
                .expect("indexed matrix")
                .into_cpu_matrix(5)
                .is_err()
        );
    }

    #[test]
    fn indexed_weight_import_streams_directly_to_q4_without_an_f32_matrix_copy() {
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let index = tiny_index();
        let mut reader = tiny_shard();
        let mut shard = index
            .open_shard(TEST_SHARD, &mut reader)
            .expect("validated tiny shard");
        let quantized = shard
            .tensor("block.weight")
            .expect("indexed matrix")
            .into_q4_matrix(16, 6)
            .expect("bounded streaming Q4 matrix");
        let expected = crate::inference_cpu::QuantizedQ4Matrix::encode(2, 3, 16, &values)
            .expect("batch Q4 reference");
        assert_eq!((quantized.rows(), quantized.columns()), (2, 3));
        assert_eq!(
            quantized.project(&[0.5, -1.0, 0.25]).unwrap(),
            expected.project(&[0.5, -1.0, 0.25]).unwrap()
        );

        let mut reader = tiny_shard();
        let mut shard = index
            .open_shard(TEST_SHARD, &mut reader)
            .expect("validated tiny shard");
        assert!(
            shard
                .tensor("block.weight")
                .expect("indexed matrix")
                .into_q4_matrix(16, 5)
                .is_err()
        );
    }

    #[test]
    fn indexed_weight_import_rejects_wrong_shards_and_tensor_shapes() {
        let index = tiny_index();
        let mut reader = tiny_shard();
        assert!(
            index
                .open_shard("wrong-shard.safetensors", &mut reader)
                .is_err()
        );
        assert!(
            index
                .open_shard(TEST_SHARD, &mut reader)
                .unwrap()
                .tensor("not.indexed")
                .is_err()
        );

        let mut mismatched = tiny_index();
        mismatched
            .tensor_specs
            .get_mut("block.weight")
            .unwrap()
            .shape = vec![3, 2];
        let mut reader = tiny_shard();
        assert!(mismatched.open_shard(TEST_SHARD, &mut reader).is_err());
    }

    #[test]
    fn pinned_index_resolves_linear_and_full_attention_layer_tensors() {
        let tensor_specs = super::expected_tensor_specs();
        let weight_map = tensor_specs
            .keys()
            .enumerate()
            .map(|(index, name)| {
                (
                    name.clone(),
                    if index % 2 == 0 {
                        "model.safetensors-00001-of-00002.safetensors".into()
                    } else {
                        "model.safetensors-00002-of-00002.safetensors".into()
                    },
                )
            })
            .collect();
        let index = Qwen35WeightIndex {
            total_size: super::EXPECTED_WEIGHT_BYTES,
            weight_map,
            tensor_specs,
        };
        for layer in 0..32 {
            if layer % 4 == 3 {
                assert!(index.linear_attention_tensor_names(layer).is_err());
                let names = index
                    .full_attention_tensor_names(layer)
                    .expect("full-attention tensor mapping");
                assert_eq!(
                    names.query_gate_projection,
                    format!("model.language_model.layers.{layer}.self_attn.q_proj.weight")
                );
                assert_eq!(
                    names.query_norm_weight,
                    format!("model.language_model.layers.{layer}.self_attn.q_norm.weight")
                );
            } else {
                let names = index
                    .linear_attention_tensor_names(layer)
                    .expect("linear-attention tensor mapping");
                assert_eq!(
                    names.qkv_projection,
                    format!("model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight")
                );
                assert!(index.full_attention_tensor_names(layer).is_err());
            }
        }
        assert!(index.linear_attention_tensor_names(32).is_err());
        assert!(index.full_attention_tensor_names(32).is_err());
    }

    fn tiny_full_attention_weights() -> Qwen35FullAttentionWeights {
        Qwen35FullAttentionWeights {
            query_gate_projection: CpuMatrix::new(
                4,
                2,
                vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            )
            .expect("query and gate projection")
            .into(),
            key_projection: CpuMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0])
                .expect("key projection")
                .into(),
            value_projection: CpuMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0])
                .expect("value projection")
                .into(),
            output_projection: CpuMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0])
                .expect("output projection")
                .into(),
            query_norm_weight: vec![0.0; 2],
            key_norm_weight: vec![0.0; 2],
        }
    }

    fn tiny_full_attention_block(
        maximum_context: usize,
        weights: Qwen35FullAttentionWeights,
    ) -> Qwen35FullAttentionBlock {
        Qwen35FullAttentionBlock::new(2, 1, 1, 2, 2, 10_000.0, 1e-6, maximum_context, weights)
            .expect("small full-attention geometry")
    }

    fn tiny_decoder_layer(output_weight: f32) -> Qwen35DecoderLayer {
        tiny_decoder_layer_with_context(output_weight, 4)
    }

    fn tiny_decoder_layer_with_context(
        output_weight: f32,
        maximum_context: usize,
    ) -> Qwen35DecoderLayer {
        let mlp = Qwen35Mlp::new(
            2,
            1,
            CpuMatrix::new(1, 2, vec![1.0, 0.0])
                .expect("MLP gate projection")
                .into(),
            CpuMatrix::new(1, 2, vec![1.0, 0.0])
                .expect("MLP up projection")
                .into(),
            CpuMatrix::new(2, 1, vec![output_weight, 0.0])
                .expect("MLP output projection")
                .into(),
        )
        .expect("small Qwen MLP");
        Qwen35DecoderLayer::new(
            2,
            1e-6,
            vec![0.0; 2],
            vec![0.0; 2],
            Qwen35TokenMixer::Full(tiny_full_attention_block(
                maximum_context,
                tiny_full_attention_weights(),
            )),
            mlp,
        )
        .expect("complete small Qwen decoder layer")
    }

    #[test]
    fn qwen_full_attention_projects_normalizes_rotates_gates_and_caches_tokens() {
        let mut block = tiny_full_attention_block(4, tiny_full_attention_weights());
        let scratch_addresses = [
            block.query_gate_scratch.as_ptr(),
            block.key_projection_scratch.as_ptr(),
            block.value_projection_scratch.as_ptr(),
            block.query_scratch.as_ptr(),
            block.gate_scratch.as_ptr(),
            block.key_scratch.as_ptr(),
        ];
        let first = block.step(&[1.0, 0.0]).expect("first token");
        assert!((first[0] - 0.5).abs() < 1e-6);
        assert!(first[1].abs() < 1e-6);
        assert_eq!(block.context_length(), 1);
        assert!(block.attention_scratch.is_cleared());
        assert_eq!(
            [
                block.query_gate_scratch.as_ptr(),
                block.key_projection_scratch.as_ptr(),
                block.value_projection_scratch.as_ptr(),
                block.query_scratch.as_ptr(),
                block.gate_scratch.as_ptr(),
                block.key_scratch.as_ptr(),
            ],
            scratch_addresses,
            "token steps reuse the same full-attention allocations"
        );
        assert!(block.query_gate_scratch.iter().all(|value| *value == 0.0));
        assert!(
            block
                .key_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            block
                .value_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(block.query_scratch.iter().all(|value| *value == 0.0));
        assert!(block.gate_scratch.iter().all(|value| *value == 0.0));
        assert!(block.key_scratch.iter().all(|value| *value == 0.0));

        let second = block.step(&[0.0, 1.0]).expect("second token");
        let probability_for_current =
            1.0 / (1.0 + (-((2.0 + 2.0 * 1.0f64.sin()) / 2.0f64.sqrt())).exp());
        let expected = [
            0.5 * (1.0 - probability_for_current),
            0.5 * probability_for_current,
        ];
        assert!((f64::from(second[0]) - expected[0]).abs() < 1e-5);
        assert!((f64::from(second[1]) - expected[1]).abs() < 1e-5);
        assert_eq!(block.context_length(), 2);
        assert!(block.attention_scratch.is_cleared());
        assert_eq!(
            [
                block.query_gate_scratch.as_ptr(),
                block.key_projection_scratch.as_ptr(),
                block.value_projection_scratch.as_ptr(),
                block.query_scratch.as_ptr(),
                block.gate_scratch.as_ptr(),
                block.key_scratch.as_ptr(),
            ],
            scratch_addresses,
            "full-attention scratch remains allocated across token steps"
        );
        assert!(block.query_gate_scratch.iter().all(|value| *value == 0.0));
        assert!(
            block
                .key_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            block
                .value_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(block.query_scratch.iter().all(|value| *value == 0.0));
        assert!(block.gate_scratch.iter().all(|value| *value == 0.0));
        assert!(block.key_scratch.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn qwen_full_attention_keeps_a_valid_prefix_when_context_is_exhausted() {
        let mut block = tiny_full_attention_block(1, tiny_full_attention_weights());
        block.step(&[1.0, 0.0]).expect("first token fits");
        assert!(block.step(&[0.0, 1.0]).is_err());
        assert_eq!(block.context_length(), 1);
    }

    #[test]
    fn qwen_full_attention_discards_partial_state_after_output_failure() {
        let mut weights = tiny_full_attention_weights();
        weights.output_projection = CpuMatrix::new(2, 2, vec![3.4e38, 3.4e38, 0.0, 0.0])
            .expect("finite but overflowing output projection")
            .into();
        let mut block = tiny_full_attention_block(2, weights);
        assert!(block.step(&[2.0, 2.0]).is_err());
        assert_eq!(block.context_length(), 0);
        assert!(block.attention_scratch.is_cleared());
        assert!(block.query_gate_scratch.iter().all(|value| *value == 0.0));
        assert!(
            block
                .key_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            block
                .value_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(block.query_scratch.iter().all(|value| *value == 0.0));
        assert!(block.gate_scratch.iter().all(|value| *value == 0.0));
        assert!(block.key_scratch.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn qwen_decoder_layer_composes_zero_centered_norm_attention_residual_and_gated_mlp() {
        let mut layer = tiny_decoder_layer(1.0);
        let normalization_scratch_address = layer.normalization_scratch.as_ptr();
        let feed_forward_scratch_address = layer.feed_forward_scratch.as_ptr();
        let output = layer.step(&[1.0, 0.0]).expect("complete decoder step");
        let normalized_value = 2.0f64.sqrt();
        let mlp_value = 2.0 / (1.0 + (-normalized_value).exp());
        let expected = 1.0 + 1.0 / 2.0f64.sqrt() + mlp_value;
        // Full-attention K/V values are stored in binary16 before the MLP
        // residual path, so this f32 analytical reference allows that one
        // quantization step while remaining much tighter than model tolerances.
        assert!((f64::from(output[0]) - expected).abs() < 1e-4);
        assert!(output[1].abs() < 1e-6);
        let Qwen35TokenMixer::Full(attention) = &layer.token_mixer else {
            panic!("full-attention fixture");
        };
        assert_eq!(attention.context_length(), 1);
        assert_eq!(
            layer.normalization_scratch.as_ptr(),
            normalization_scratch_address
        );
        assert_eq!(
            layer.feed_forward_scratch.as_ptr(),
            feed_forward_scratch_address
        );
        assert!(
            layer
                .normalization_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(layer.feed_forward_scratch.iter().all(|value| *value == 0.0));
        layer
            .step(&[1.0, 0.0])
            .expect("second step reuses feed-forward storage");
        assert_eq!(
            layer.normalization_scratch.as_ptr(),
            normalization_scratch_address
        );
        assert_eq!(
            layer.feed_forward_scratch.as_ptr(),
            feed_forward_scratch_address
        );
        assert!(
            layer
                .normalization_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(layer.feed_forward_scratch.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn qwen_mlp_reuses_and_zeroes_gate_and_up_projection_scratch() {
        let mut mlp = Qwen35Mlp::new(
            2,
            1,
            CpuMatrix::new(1, 2, vec![1.0, 0.0])
                .expect("gate projection")
                .into(),
            CpuMatrix::new(1, 2, vec![1.0, 0.0])
                .expect("up projection")
                .into(),
            CpuMatrix::new(2, 1, vec![1.0, 0.0])
                .expect("down projection")
                .into(),
        )
        .expect("valid Qwen MLP");

        let gate_scratch_address = mlp.gate_scratch.as_ptr();
        let up_scratch_address = mlp.up_scratch.as_ptr();
        let mut output = [0.0; 2];
        mlp.step_into(&[1.0, 0.0], &mut output)
            .expect("first token MLP");
        let first = output;
        mlp.step_into(&[1.0, 0.0], &mut output)
            .expect("reused token MLP");
        assert_eq!(first, output);
        assert_eq!(mlp.gate_scratch.as_ptr(), gate_scratch_address);
        assert_eq!(mlp.up_scratch.as_ptr(), up_scratch_address);
        assert!(mlp.gate_scratch.iter().all(|value| *value == 0.0));
        assert!(mlp.up_scratch.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn qwen_mlp_zeroes_reused_scratch_after_activation_overflow() {
        let mut mlp = Qwen35Mlp::new(
            1,
            1,
            CpuMatrix::new(1, 1, vec![f32::MAX])
                .expect("finite gate projection")
                .into(),
            CpuMatrix::new(1, 1, vec![f32::MAX])
                .expect("finite up projection")
                .into(),
            CpuMatrix::new(1, 1, vec![1.0])
                .expect("down projection")
                .into(),
        )
        .expect("valid Qwen MLP");

        let mut output = [42.0];
        assert!(mlp.step_into(&[1.0], &mut output).is_err());
        assert_eq!(output, [0.0]);
        assert!(mlp.gate_scratch.iter().all(|value| *value == 0.0));
        assert!(mlp.up_scratch.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn qwen_decoder_layer_clears_attention_state_if_the_mlp_fails() {
        let mut layer = tiny_decoder_layer(3.4e38);
        assert!(layer.step(&[1.0, 0.0]).is_err());
        assert!(
            layer
                .normalization_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(layer.feed_forward_scratch.iter().all(|value| *value == 0.0));
        let Qwen35TokenMixer::Full(attention) = &layer.token_mixer else {
            panic!("full-attention fixture");
        };
        assert_eq!(attention.context_length(), 0);
    }

    #[test]
    fn complete_text_decoder_prefills_returns_tied_logits_and_preserves_valid_context_bounds() {
        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("small tied embedding table")
                .into();
        let mut decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![1.0; 2],
            vec![tiny_decoder_layer(1.0)],
            1e-6,
            2,
        )
        .expect("complete small text decoder");
        let logits = decoder.prefill(&[0, 1]).expect("bounded prompt prefill");
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|logit| logit.is_finite()));
        assert_eq!(decoder.vocabulary_size(), 3);
        assert_eq!(decoder.context_length(), 2);
        assert!(decoder.step_token(2).is_err());
        assert_eq!(decoder.context_length(), 2);
        assert!(decoder.step_token(3).is_err());
        assert_eq!(decoder.context_length(), 2);

        decoder.clear();
        assert_eq!(decoder.context_length(), 0);
        assert!(decoder.prefill(&[]).is_err());
        assert!(decoder.prefill(&[0, 3]).is_err());
        assert_eq!(decoder.context_length(), 0);
    }

    #[test]
    fn text_decoder_ping_pong_matches_sequential_two_layer_reference() {
        let embedding = [1.0, 0.0];
        let mut reference_first = tiny_decoder_layer(1.0);
        let mut reference_second = tiny_decoder_layer(0.5);
        let first_hidden = reference_first
            .step_with_mrope_positions(&embedding, [0; 3])
            .expect("first reference layer");
        let second_hidden = reference_second
            .step_with_mrope_positions(&first_hidden, [0; 3])
            .expect("second reference layer");
        let mut normalized = [0.0; 2];
        crate::inference_cpu::rms_norm_zero_centered_into(
            &second_hidden,
            &[0.0; 2],
            1e-6,
            &mut normalized,
        )
        .expect("reference final norm");
        let expected = [normalized[0], normalized[1], normalized[0] + normalized[1]];

        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("small tied embedding table")
                .into();
        let mut decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![0.0; 2],
            vec![tiny_decoder_layer(1.0), tiny_decoder_layer(0.5)],
            1e-6,
            2,
        )
        .expect("two-layer ping-pong decoder");
        let actual = decoder
            .step_embedded_with_mrope_positions(&embedding, [0; 3])
            .expect("two-layer ping-pong step");
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert_eq!(decoder.context_length(), 1);
    }

    #[test]
    fn text_decoder_advances_precomputed_multimodal_embeddings_with_mrope_positions() {
        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("small tied embedding table")
                .into();
        let mut decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![1.0; 2],
            vec![tiny_decoder_layer(1.0)],
            1e-6,
            2,
        )
        .expect("complete small text decoder");

        let logits = decoder
            .step_embedded_with_mrope_positions(&[0.25, -0.5], [7, 9, 11])
            .expect("bounded embedded image token");
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|value| value.is_finite()));
        assert_eq!(decoder.context_length(), 1);
        assert_eq!(decoder.next_mrope_position(), 12);

        assert!(
            decoder
                .step_embedded_with_mrope_positions(&[0.25, -0.5], [262_144, 0, 0])
                .is_err()
        );
        assert_eq!(decoder.context_length(), 1);
    }

    #[test]
    fn text_decoder_prefills_interleaved_image_embeddings_and_text_positions() {
        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("small tied embedding table")
                .into();
        let mut decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![1.0; 2],
            vec![tiny_decoder_layer_with_context(1.0, 5)],
            1e-6,
            5,
        )
        .expect("complete small text decoder");
        let token_ids = [0, 2, 2, 1];
        let positions = [[0, 0, 0], [1, 1, 1], [1, 1, 2], [2, 2, 2]];
        let image_values = [0.25, -0.5, 0.75, 1.0];
        let image_positions = [[1, 1, 1], [1, 1, 2]];
        let spans = [Qwen35EmbeddedSpan {
            token_start: 1,
            token_id: 2,
            embeddings: &image_values,
            positions: &image_positions,
        }];
        let prompt = Qwen35EmbeddedPrompt {
            token_ids: &token_ids,
            positions: &positions,
            spans: &spans,
        };

        let logits = decoder
            .prefill_embedded_prompt_controlled(&prompt, || false)
            .expect("interleaved multimodal prompt");
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|value| value.is_finite()));
        assert_eq!(decoder.context_length(), 4);
        assert_eq!(decoder.next_mrope_position(), 3);
        decoder
            .step_token(1)
            .expect("generation continues from the multimodal text position");
        assert_eq!(decoder.context_length(), 5);
        assert_eq!(decoder.next_mrope_position(), 4);
    }

    #[test]
    fn text_decoder_rejects_misaligned_image_embeddings_before_cache_mutation() {
        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("small tied embedding table")
                .into();
        let mut decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![1.0; 2],
            vec![tiny_decoder_layer(1.0)],
            1e-6,
            4,
        )
        .expect("complete small text decoder");
        let token_ids = [0, 2, 1];
        let positions = [[0, 0, 0], [1, 1, 1], [2, 2, 2]];
        let image_values = [0.25, -0.5];
        let image_positions = [[1, 1, 1]];
        let spans = [Qwen35EmbeddedSpan {
            token_start: 1,
            token_id: 0,
            embeddings: &image_values,
            positions: &image_positions,
        }];
        let prompt = Qwen35EmbeddedPrompt {
            token_ids: &token_ids,
            positions: &positions,
            spans: &spans,
        };

        assert!(
            decoder
                .prefill_embedded_prompt_controlled(&prompt, || false)
                .is_err()
        );
        assert_eq!(decoder.context_length(), 0);
    }

    #[test]
    fn text_decoder_q4_embeddings_support_lookup_and_reset_all_layers_after_failure() {
        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("small tied embedding table")
                .into();
        let embeddings = embeddings
            .quantize_q4(16)
            .expect("small Q4 tied embedding table");
        let mut decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![1.0; 2],
            vec![tiny_decoder_layer(1.0)],
            1e-6,
            2,
        )
        .expect("Q4 text decoder");
        assert_eq!(decoder.step_token(0).unwrap().len(), 3);
        decoder.clear();

        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("small tied embedding table")
                .into();
        let mut decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![1.0; 2],
            vec![tiny_decoder_layer(1.0), tiny_decoder_layer(3.4e38)],
            1e-6,
            2,
        )
        .expect("two-layer failure fixture");
        assert!(decoder.step_token(0).is_err());
        assert_eq!(decoder.context_length(), 0);
        let Qwen35TokenMixer::Full(attention) = &decoder.layers[0].token_mixer else {
            panic!("full-attention fixture");
        };
        assert_eq!(attention.context_length(), 0);
    }

    #[test]
    fn greedy_token_selection_is_deterministic_and_honors_closed_token_sets() {
        let logits = [0.5, 2.0, 2.0, 1.0];
        assert_eq!(greedy_token_from_logits(&logits, None).unwrap(), 1);
        assert_eq!(greedy_token_from_logits(&logits, Some(&[0, 3])).unwrap(), 3);
        assert!(greedy_token_from_logits(&logits, Some(&[])).is_err());
        assert!(greedy_token_from_logits(&logits, Some(&[9])).is_err());
        let allowed = [0, 3];
        assert_eq!(
            greedy_token_from_allowed_logits(&[0.5, 1.0], &allowed, 4).unwrap(),
            3
        );
        assert_eq!(
            greedy_token_from_allowed_logits(&[2.0, 2.0], &allowed, 4).unwrap(),
            0
        );
        assert!(greedy_token_from_allowed_logits(&[1.0], &allowed, 4).is_err());
        assert!(greedy_token_from_allowed_logits(&[1.0, 2.0], &allowed, 3).is_err());
        assert!(greedy_token_from_logits(&logits, Some(&[3, 0])).is_err());
        let disallowed_nan = [0.5, 2.0, f32::NAN, 1.0];
        assert_eq!(
            greedy_token_from_logits(&disallowed_nan, Some(&[0, 1, 3])).unwrap(),
            1
        );
        assert!(greedy_token_from_logits(&disallowed_nan, Some(&[0, 2, 3])).is_err());
        assert!(greedy_token_from_logits(&[f32::NAN], None).is_err());
    }

    #[test]
    #[ignore = "release-only dense constrained-token selection comparison"]
    fn constrained_token_selection_buffer_measurement() {
        use std::collections::BTreeSet;
        use std::hint::black_box;
        use std::time::Instant;

        fn previous_tree_selection(logits: &[f32], allowed: &BTreeSet<u32>) -> u32 {
            assert!(logits.iter().all(|logit| logit.is_finite()));
            let mut best: Option<(u32, f32)> = None;
            for (token_id, logit) in logits.iter().copied().enumerate() {
                if !allowed.contains(&(token_id as u32)) {
                    continue;
                }
                if best.is_none_or(|(_, best_logit)| logit > best_logit) {
                    best = Some((token_id as u32, logit));
                }
            }
            best.expect("nonempty mask").0
        }

        let vocabulary_size = VOCABULARY_SIZE;
        let logits = (0..vocabulary_size)
            .map(|token| ((token * 7919 % 100_003) as f32) / 100_003.0)
            .collect::<Vec<_>>();
        for allowed_count in [131_072, 180_000, 223_488] {
            let allowed = (0..allowed_count)
                .map(|index| (index * vocabulary_size / allowed_count) as u32)
                .collect::<Vec<_>>();
            let tree = allowed.iter().copied().collect::<BTreeSet<_>>();
            let expected = previous_tree_selection(&logits, &tree);
            assert_eq!(
                greedy_token_from_logits(&logits, Some(&allowed)).unwrap(),
                expected
            );

            let mut tree_times = Vec::with_capacity(31);
            let mut slice_times = Vec::with_capacity(31);
            for sample in 0_usize..31 {
                if sample.is_multiple_of(2) {
                    let start = Instant::now();
                    black_box(previous_tree_selection(
                        black_box(&logits),
                        black_box(&tree),
                    ));
                    tree_times.push(start.elapsed());

                    let start = Instant::now();
                    black_box(greedy_token_from_logits(
                        black_box(&logits),
                        Some(black_box(&allowed)),
                    ))
                    .expect("sorted-slice constrained selection");
                    slice_times.push(start.elapsed());
                } else {
                    let start = Instant::now();
                    black_box(greedy_token_from_logits(
                        black_box(&logits),
                        Some(black_box(&allowed)),
                    ))
                    .expect("sorted-slice constrained selection");
                    slice_times.push(start.elapsed());

                    let start = Instant::now();
                    black_box(previous_tree_selection(
                        black_box(&logits),
                        black_box(&tree),
                    ));
                    tree_times.push(start.elapsed());
                }
            }
            tree_times.sort_unstable();
            slice_times.sort_unstable();
            eprintln!(
                "qwen-constrained-selection vocab={vocabulary_size} allowed={allowed_count} samples=31 tree_p50_us={} tree_p95_us={} sorted_slice_p50_us={} sorted_slice_p95_us={}",
                tree_times[15].as_nanos() / 1_000,
                tree_times[29].as_nanos() / 1_000,
                slice_times[15].as_nanos() / 1_000,
                slice_times[29].as_nanos() / 1_000,
            );
        }
    }

    #[test]
    fn constrained_generation_uses_sparse_projection_through_half_the_vocabulary() {
        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(4, 2, vec![0.5, 0.25, -0.5, 1.0, 0.75, -0.25, 1.0, 0.5])
                .expect("small tied embedding table")
                .into();
        let decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![1.0; 2],
            vec![tiny_decoder_layer(1.0)],
            1e-6,
            4,
        )
        .expect("complete small text decoder");
        let normalized_hidden = [0.25, -0.75];
        let mut selected_rows = Vec::new();
        let mut logits = [0.0; 4];
        let half_allow_list = [0, 2];
        assert!(
            decoder
                .project_generation_logits(
                    &normalized_hidden,
                    Some(&half_allow_list),
                    &mut selected_rows,
                    &mut logits,
                )
                .expect("sparse projection through the measured cutoff")
        );
        assert_eq!(selected_rows, [0, 2]);

        let dense = decoder
            .token_embeddings
            .project(&normalized_hidden)
            .expect("dense output-head reference");
        assert_eq!(&logits[..half_allow_list.len()], &[dense[0], dense[2]]);

        let above_half_allow_list = [0, 1, 2];
        assert!(
            !decoder
                .project_generation_logits(
                    &normalized_hidden,
                    Some(&above_half_allow_list),
                    &mut selected_rows,
                    &mut logits,
                )
                .expect("dense projection above the measured cutoff")
        );
        assert!(selected_rows.is_empty());
        assert_eq!(logits.as_slice(), dense.as_slice());
    }

    #[test]
    fn greedy_generation_uses_per_prefix_constraints_stops_and_clears_its_cache() {
        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("small tied embedding table")
                .into();
        let mut decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![1.0; 2],
            vec![tiny_decoder_layer(1.0)],
            1e-6,
            4,
        )
        .expect("small text decoder");
        let mut constraint_step = 0;
        let generated = generate_greedy(
            &mut decoder,
            &[0],
            &std::collections::BTreeSet::from([2]),
            4,
            |_, allowed| {
                constraint_step += 1;
                allowed.clear();
                allowed.push(if constraint_step == 1 { 1 } else { 2 });
                Ok(true)
            },
            || false,
        )
        .expect("bounded constrained greedy decode");
        assert_eq!(generated, vec![1]);
        assert_eq!(decoder.context_length(), 0);
    }

    #[test]
    fn cancellation_between_decoder_layers_invalidates_the_partial_prefix() {
        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("small tied embedding table")
                .into();
        let mut decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![1.0; 2],
            vec![tiny_decoder_layer(1.0), tiny_decoder_layer(1.0)],
            1e-6,
            4,
        )
        .expect("two-layer text decoder");
        let mut checks = 0;
        assert!(
            decoder
                .step_token_controlled(0, || {
                    checks += 1;
                    checks >= 3
                })
                .is_err()
        );
        assert_eq!(decoder.context_length(), 0);
        let Qwen35TokenMixer::Full(attention) = &decoder.layers[0].token_mixer else {
            panic!("full-attention fixture");
        };
        assert_eq!(attention.context_length(), 0);
    }

    #[test]
    fn prefill_reserves_prompt_prefix_and_checks_cancellation_before_allocation() {
        let embeddings: Qwen35ProjectionMatrix =
            CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("small tied embedding table")
                .into();
        let mut decoder = Qwen35TextDecoder::new(
            embeddings,
            vec![1.0; 2],
            vec![tiny_decoder_layer_with_context(1.0, 4)],
            1e-6,
            4,
        )
        .expect("small text decoder");
        let Qwen35TokenMixer::Full(attention) = &decoder.layers[0].token_mixer else {
            panic!("full-attention fixture");
        };
        assert_eq!(attention.cache.capacity_positions(), 0);

        assert!(matches!(
            decoder.prefill_controlled(&[0, 1], || true),
            Err(CoreError::Cancelled)
        ));
        let Qwen35TokenMixer::Full(attention) = &decoder.layers[0].token_mixer else {
            panic!("full-attention fixture");
        };
        assert_eq!(attention.cache.capacity_positions(), 0);
        assert_eq!(decoder.context_length(), 0);

        decoder
            .prefill_controlled(&[0, 1], || false)
            .expect("valid prompt prefill");
        let Qwen35TokenMixer::Full(attention) = &decoder.layers[0].token_mixer else {
            panic!("full-attention fixture");
        };
        assert_eq!(attention.cache.context_length(), 2);
        assert_eq!(attention.cache.capacity_positions(), 2);
    }

    #[test]
    fn qwen35_linear_layer_composes_convolution_head_repetition_and_recurrence() {
        let mut state = Qwen35LinearAttentionState::new(1, 2, 1, 1, 1)
            .expect("valid small linear-attention geometry");
        let scratch_addresses = (
            state.convolution_scratch.as_ptr(),
            state.query_scratch.as_ptr(),
            state.key_scratch.as_ptr(),
            state.log_decay_scratch.as_ptr(),
            state.beta_scratch.as_ptr(),
        );
        assert_eq!(state.projected_qkv_width(), 4);
        let projected_qkv = [1.0, 2.0, 3.0, 4.0];
        let convolution_weights = [1.0; 4];

        assert!(
            state
                .step(
                    &projected_qkv,
                    &convolution_weights,
                    &[0.0],
                    &[0.0],
                    &[0.0],
                    &[0.0],
                )
                .is_err()
        );

        let output = state
            .step(
                &projected_qkv,
                &convolution_weights,
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[0.0, 0.0],
            )
            .expect("composed Qwen layer step");
        let silu = |value: f64| value / (1.0 + (-value).exp());
        let query = silu(1.0);
        let key = silu(2.0);
        let query_norm = query / (query.powi(2) + 1.0e-6).sqrt();
        let key_norm = key / (key.powi(2) + 1.0e-6).sqrt();
        for (actual, projected_value) in output.iter().zip([3.0, 4.0]) {
            let expected = 0.5 * silu(projected_value) * query_norm * key_norm;
            assert!((f64::from(*actual) - expected).abs() < 1e-6);
        }
        assert_eq!(
            scratch_addresses,
            (
                state.convolution_scratch.as_ptr(),
                state.query_scratch.as_ptr(),
                state.key_scratch.as_ptr(),
                state.log_decay_scratch.as_ptr(),
                state.beta_scratch.as_ptr(),
            )
        );
        assert!(state.convolution_scratch.iter().all(|value| *value == 0.0));
        assert!(state.query_scratch.iter().all(|value| *value == 0.0));
        assert!(state.key_scratch.iter().all(|value| *value == 0.0));
        assert!(state.log_decay_scratch.iter().all(|value| *value == 0.0));
        assert!(state.beta_scratch.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn pinned_qwen35_layer_state_uses_the_expected_qkv_width() {
        let state =
            Qwen35LinearAttentionState::for_qwen35_4b().expect("pinned Qwen3.5-4B layer state");
        assert_eq!(state.projected_qkv_width(), DELTA_QKV_SIZE);
    }

    #[test]
    fn qwen35_linear_attention_block_projects_one_token_through_the_full_sublayer() {
        let weights = Qwen35LinearAttentionWeights {
            qkv_projection: CpuMatrix::new(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("qkv projection")
                .into(),
            gate_projection: CpuMatrix::new(1, 2, vec![1.0, 1.0])
                .expect("gate projection")
                .into(),
            decay_projection: CpuMatrix::new(1, 2, vec![0.0, 0.0])
                .expect("decay projection")
                .into(),
            beta_projection: CpuMatrix::new(1, 2, vec![0.0, 0.0])
                .expect("beta projection")
                .into(),
            convolution: vec![1.0; 3],
            a_log: vec![0.0],
            dt_bias: vec![0.0],
            norm_weight: vec![1.0],
            output_projection: CpuMatrix::new(2, 1, vec![2.0, -2.0])
                .expect("output projection")
                .into(),
        };
        let mut block = Qwen35LinearAttentionBlock::new(2, 1, 1, 1, 1, 1, 1e-6, weights.clone())
            .expect("complete small Qwen linear-attention sublayer");
        let scratch_addresses = (
            block.qkv_projection_scratch.as_ptr(),
            block.gate_projection_scratch.as_ptr(),
            block.decay_projection_scratch.as_ptr(),
            block.beta_projection_scratch.as_ptr(),
            block.recurrent_scratch.as_ptr(),
            block.normalization_scratch.as_ptr(),
        );
        let state_scratch_addresses = (
            block.state.convolution_scratch.as_ptr(),
            block.state.query_scratch.as_ptr(),
            block.state.key_scratch.as_ptr(),
            block.state.log_decay_scratch.as_ptr(),
            block.state.beta_scratch.as_ptr(),
        );

        let output = block.step(&[1.0, 2.0]).expect("one-token sublayer output");
        let silu = |value: f64| value / (1.0 + (-value).exp());
        let query = silu(1.0);
        let key = silu(2.0);
        let value = silu(3.0);
        let recurrent = 0.5
            * value
            * (query / (query * query + 1e-6).sqrt())
            * (key / (key * key + 1e-6).sqrt());
        let gated = recurrent / (recurrent * recurrent + 1e-6).sqrt() * silu(3.0);
        assert_eq!(output.len(), 2);
        assert!((f64::from(output[0]) - gated * 2.0).abs() < 1e-6);
        assert!((f64::from(output[1]) + gated * 2.0).abs() < 1e-6);
        assert_eq!(
            scratch_addresses,
            (
                block.qkv_projection_scratch.as_ptr(),
                block.gate_projection_scratch.as_ptr(),
                block.decay_projection_scratch.as_ptr(),
                block.beta_projection_scratch.as_ptr(),
                block.recurrent_scratch.as_ptr(),
                block.normalization_scratch.as_ptr(),
            )
        );
        assert_eq!(
            state_scratch_addresses,
            (
                block.state.convolution_scratch.as_ptr(),
                block.state.query_scratch.as_ptr(),
                block.state.key_scratch.as_ptr(),
                block.state.log_decay_scratch.as_ptr(),
                block.state.beta_scratch.as_ptr(),
            )
        );
        assert!(
            block
                .qkv_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            block
                .gate_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            block
                .decay_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            block
                .beta_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(block.recurrent_scratch.iter().all(|value| *value == 0.0));
        assert!(
            block
                .normalization_scratch
                .iter()
                .all(|value| *value == 0.0)
        );

        let mut into_block =
            Qwen35LinearAttentionBlock::new(2, 1, 1, 1, 1, 1, 1e-6, weights.clone())
                .expect("caller-owned Qwen linear-attention sublayer");
        let mut into_output = [0.0; 2];
        into_block
            .step_into(&[1.0, 2.0], &mut into_output)
            .expect("caller-owned one-token output");
        assert_eq!(into_output.as_slice(), output.as_slice());
        assert!(
            into_block
                .normalization_scratch
                .iter()
                .all(|value| *value == 0.0)
        );

        block.weights.output_projection = CpuMatrix::new(2, 1, vec![3.4e38, 3.4e38])
            .expect("finite overflowing output projection")
            .into();
        assert!(block.step(&[1.0, 2.0]).is_err());
        assert!(
            block
                .state
                .recurrence
                .values()
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            block
                .qkv_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            block
                .gate_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            block
                .decay_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            block
                .beta_projection_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(block.recurrent_scratch.iter().all(|value| *value == 0.0));
        assert!(
            block
                .normalization_scratch
                .iter()
                .all(|value| *value == 0.0)
        );
        assert_eq!(
            scratch_addresses,
            (
                block.qkv_projection_scratch.as_ptr(),
                block.gate_projection_scratch.as_ptr(),
                block.decay_projection_scratch.as_ptr(),
                block.beta_projection_scratch.as_ptr(),
                block.recurrent_scratch.as_ptr(),
                block.normalization_scratch.as_ptr(),
            )
        );

        let quantize = |matrix: Qwen35ProjectionMatrix| {
            matrix.quantize_q4(2).expect("exact Q4 fixture conversion")
        };
        let quantized_weights = Qwen35LinearAttentionWeights {
            qkv_projection: quantize(weights.qkv_projection),
            gate_projection: quantize(weights.gate_projection),
            decay_projection: quantize(weights.decay_projection),
            beta_projection: quantize(weights.beta_projection),
            convolution: weights.convolution,
            a_log: weights.a_log,
            dt_bias: weights.dt_bias,
            norm_weight: weights.norm_weight,
            output_projection: quantize(weights.output_projection),
        };
        let mut quantized =
            Qwen35LinearAttentionBlock::new(2, 1, 1, 1, 1, 1, 1e-6, quantized_weights)
                .expect("Q4 Qwen linear-attention sublayer");
        let quantized_output = quantized
            .step(&[1.0, 2.0])
            .expect("quantized one-token sublayer output");
        for (quantized, reference) in quantized_output.iter().zip(output) {
            // Grouped Q4 NEON now applies its shared scale after accumulation,
            // so the mathematically exact fixture may differ by a few f32 ulps.
            assert!((quantized - reference).abs() <= 2.0e-6);
        }
    }

    #[test]
    fn qwen35_linear_attention_block_rejects_weight_geometry_mismatch() {
        let weights = Qwen35LinearAttentionWeights {
            qkv_projection: CpuMatrix::new(2, 2, vec![0.0; 4])
                .expect("qkv projection")
                .into(),
            gate_projection: CpuMatrix::new(1, 2, vec![0.0; 2])
                .expect("gate projection")
                .into(),
            decay_projection: CpuMatrix::new(1, 2, vec![0.0; 2])
                .expect("decay projection")
                .into(),
            beta_projection: CpuMatrix::new(1, 2, vec![0.0; 2])
                .expect("beta projection")
                .into(),
            convolution: vec![0.0; 3],
            a_log: vec![0.0],
            dt_bias: vec![0.0],
            norm_weight: vec![1.0],
            output_projection: CpuMatrix::new(2, 1, vec![0.0; 2])
                .expect("output projection")
                .into(),
        };
        assert!(Qwen35LinearAttentionBlock::new(2, 1, 1, 1, 1, 1, 1e-6, weights).is_err());
    }

    #[test]
    fn qwen35_gated_output_normalization_is_per_value_head() {
        let state = Qwen35LinearAttentionState::new(1, 2, 2, 2, 1)
            .expect("valid small linear-attention geometry");
        let output = state
            .normalize_gated_output(
                &[3.0, 4.0, 0.0, 5.0],
                &[1.0, -1.0, 0.0, 2.0],
                &[2.0, 0.5],
                1e-6,
            )
            .expect("per-head gated output normalization");
        let mut reused = [0.0; 4];
        state
            .normalize_gated_output_into(
                &[3.0, 4.0, 0.0, 5.0],
                &[1.0, -1.0, 0.0, 2.0],
                &[2.0, 0.5],
                1e-6,
                &mut reused,
            )
            .expect("caller-owned per-head gated normalization");
        assert_eq!(reused.as_slice(), output.as_slice());
        let inverse = 1.0f64 / (12.5f64 + 1.0e-6).sqrt();
        let silu = |value: f64| value / (1.0 + (-value).exp());
        let expected = [
            6.0 * inverse * silu(1.0),
            2.0 * inverse * silu(-1.0),
            0.0,
            2.5 * inverse * silu(2.0),
        ];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((f64::from(*actual) - expected).abs() < 1e-6);
        }
        reused.fill(9.0);
        assert!(
            state
                .normalize_gated_output_into(
                    &[1.0; 4],
                    &[0.0; 4],
                    &[1.0; 2],
                    1e-6,
                    &mut reused[..3],
                )
                .is_err()
        );
        assert_eq!(reused[..3], [0.0; 3]);
    }
}
