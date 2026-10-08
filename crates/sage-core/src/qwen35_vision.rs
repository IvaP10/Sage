//! Sage-owned preprocessing for the pinned Qwen3.5 image path.
//!
//! This bounded implementation resizes RGB pixels, applies the signed model
//! normalization, produces temporal patch vectors, and exposes scalar vision
//! block references. It does not decode image files or assemble the full tower.

use serde::Deserialize;
use zeroize::Zeroizing;

use crate::inference_cpu::{CpuMatrix, rotary_qwen35_vision_axial};
use crate::qwen35::Qwen35ProjectionMatrix;
use crate::{CoreError, CoreResult, model_package::VerifiedQwen35Package};

pub const PREPROCESSOR_CONFIG_NAME: &str = "preprocessor_config.json";

const MAX_CONFIG_BYTES: usize = 64 * 1024;
const MAX_RAW_IMAGE_BYTES: usize = 64 * 1024 * 1024;
const MIN_PIXELS: u64 = 65_536;
const MAX_PIXELS: u64 = 16_777_216;
const PATCH: usize = 16;
const TEMPORAL: usize = 2;
const MERGE: usize = 2;
const CHANNELS: usize = 3;
const PATCH_FEATURES: usize = CHANNELS * TEMPORAL * PATCH * PATCH;
const MAX_IMAGE_TOKENS: usize = 4_096;
const MAX_ASPECT_RATIO: usize = 200;
const POSITION_GRID_SIDE: usize = 48;
const VISION_HIDDEN_SIZE: usize = 1024;
const TEXT_HIDDEN_SIZE: usize = 2560;
const VISION_ROPE_THETA: f32 = 10_000.0;
const VISION_LAYER_NORM_EPSILON: f32 = 1e-6;
const VISION_BLOCK_COUNT: usize = 24;
const MAX_VISION_BATCH_TOKENS: usize = sage_kernels::Q4_BATCH_MAX_SIZE;
const MAX_SCRATCH_VALUES: usize = MAX_IMAGE_TOKENS * MERGE * MERGE * PATCH_FEATURES;
const MAX_VISION_ATTENTION_MULTIPLIES: usize = 256_000_000;
const MAX_VISION_STACK_ATTENTION_MULTIPLIES: usize = 2_000_000_000;
const MIN_PARALLEL_VISION_ATTENTION_MULTIPLIES: usize = 1_000_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProcessorConfig {
    size: RawImageSize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    image_mean: [f32; CHANNELS],
    image_std: [f32; CHANNELS],
    processor_class: String,
    image_processor_type: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawImageSize {
    longest_edge: u64,
    shortest_edge: u64,
}

/// Parsed only from the preprocessor file authenticated by the same signed
/// manifest as the model weights.
#[derive(Debug, Clone)]
pub struct Qwen35VisionProcessor {
    manifest_sha256: String,
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35RgbImage<'a> {
    pub width: usize,
    pub height: usize,
    /// Interleaved RGB bytes in row-major order.
    pub pixels: &'a [u8],
}

/// Patch rows are `[channel, temporal frame, row, column]` flattened.
pub struct PreparedQwen35Image {
    width: usize,
    height: usize,
    patches_high: usize,
    patches_wide: usize,
    merged_tokens: usize,
    patch_values: Zeroizing<Vec<f32>>,
}

/// Sage's scalar reference for the pinned Qwen patch embedding convolution.
/// It accepts weights already validated against the signed package index and
/// projects one patch at a time to bound temporary activation memory.
pub struct Qwen35PatchEmbed {
    projection: Qwen35ProjectionMatrix,
    bias: Vec<f32>,
}

/// Learned Qwen vision positions on the checkpoint's fixed 48-by-48 grid.
/// The table is sampled with bilinear interpolation and aligned grid corners.
pub struct Qwen35VisionPositionEmbedding {
    table: CpuMatrix,
}

/// Trained tensors for one Qwen vision transformer block. Matrices use
/// `[output, input]` row-major order, matching the checkpoint linear layers.
pub struct Qwen35VisionBlockWeights {
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub qkv_projection: Qwen35ProjectionMatrix,
    pub qkv_bias: Vec<f32>,
    pub attention_projection: Qwen35ProjectionMatrix,
    pub attention_bias: Vec<f32>,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub mlp_input_projection: Qwen35ProjectionMatrix,
    pub mlp_input_bias: Vec<f32>,
    pub mlp_output_projection: Qwen35ProjectionMatrix,
    pub mlp_output_bias: Vec<f32>,
}

/// Sage's bounded scalar reference for one Qwen vision transformer block.
/// It performs non-causal, per-image multi-head attention, axial 2D RoPE,
/// pre-norm residuals and the tanh-approximate GELU MLP.
pub struct Qwen35VisionBlock {
    weights: Qwen35VisionBlockWeights,
    hidden_size: usize,
    heads: usize,
    head_size: usize,
}

/// Trained tensors for the pinned pre-shuffle-normalized Qwen patch merger.
pub struct Qwen35VisionPatchMergerWeights {
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub input_projection: Qwen35ProjectionMatrix,
    pub input_bias: Vec<f32>,
    pub output_projection: Qwen35ProjectionMatrix,
    pub output_bias: Vec<f32>,
}

/// Sage's scalar reference for merging each spatial 2×2 patch block into one
/// text-width visual embedding.
pub struct Qwen35VisionPatchMerger {
    weights: Qwen35VisionPatchMergerWeights,
    hidden_size: usize,
    output_size: usize,
    merged_input_size: usize,
}

/// Ordered execution for the pinned 24-layer Qwen vision transformer.
pub struct Qwen35VisionBlockStack {
    blocks: Vec<Qwen35VisionBlock>,
    hidden_size: usize,
}

/// Image-conditioned text-width embeddings produced by Sage's vision path.
pub struct Qwen35VisionEncoding {
    token_count: usize,
    hidden_size: usize,
    values: Zeroizing<Vec<f32>>,
}

impl Qwen35VisionEncoding {
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Complete Qwen3.5 image path assembled from caller-supplied package-verified
/// tensors: patch embedding, positions, 24 blocks and spatial merger.
pub struct Qwen35VisionEncoder {
    patch_embed: Qwen35PatchEmbed,
    position_embedding: Qwen35VisionPositionEmbedding,
    blocks: Qwen35VisionBlockStack,
    merger: Qwen35VisionPatchMerger,
}

impl Qwen35PatchEmbed {
    /// Construct from weights whose package authenticity was established by
    /// the caller. This operation checks tensor geometry and finite values;
    /// it does not authenticate or admit model packages.
    pub fn new(projection: Qwen35ProjectionMatrix, bias: Vec<f32>) -> CoreResult<Self> {
        if projection.rows() != VISION_HIDDEN_SIZE
            || projection.columns() != PATCH_FEATURES
            || bias.len() != VISION_HIDDEN_SIZE
            || bias.iter().any(|value| !value.is_finite())
        {
            return Err(vision_error(
                "Qwen patch embedding weights do not match the pinned [1024,3,2,16,16] geometry",
            ));
        }
        Ok(Self { projection, bias })
    }

    /// Project one row-major patch. The flattened feature order is exactly the
    /// checkpoint convolution order: channel, temporal frame, row, column.
    pub fn project_patch(
        &self,
        image: &PreparedQwen35Image,
        patch_index: usize,
    ) -> CoreResult<Zeroizing<Vec<f32>>> {
        let patch_count = image
            .patches_high
            .checked_mul(image.patches_wide)
            .filter(|count| *count <= MAX_IMAGE_TOKENS * MERGE * MERGE)
            .ok_or_else(|| vision_error("Image patch count is outside the encoder bound"))?;
        if image.patch_values.len() != patch_count * PATCH_FEATURES || patch_index >= patch_count {
            return Err(vision_error(
                "Patch index or prepared patch buffer does not match its grid",
            ));
        }
        let mut output = reserved_vision_buffer(VISION_HIDDEN_SIZE)?;
        output.resize(VISION_HIDDEN_SIZE, 0.0);
        self.project_patches_into(image, patch_index, 1, &mut output)?;
        Ok(output)
    }

    /// Project a bounded consecutive patch batch into caller-owned
    /// `[patch, hidden]` storage, reusing matrix weights across the batch.
    pub fn project_patches_into(
        &self,
        image: &PreparedQwen35Image,
        first_patch: usize,
        batch_size: usize,
        output: &mut [f32],
    ) -> CoreResult<()> {
        let Some(patch_count) = image
            .patches_high
            .checked_mul(image.patches_wide)
            .filter(|count| *count <= MAX_IMAGE_TOKENS * MERGE * MERGE)
        else {
            output.fill(0.0);
            return Err(vision_error(
                "Image patch count is outside the encoder bound",
            ));
        };
        let input_elements = batch_size.checked_mul(PATCH_FEATURES);
        let output_elements = batch_size.checked_mul(VISION_HIDDEN_SIZE);
        let input_start = first_patch.checked_mul(PATCH_FEATURES);
        let input_end = input_start
            .zip(input_elements)
            .and_then(|(start, count)| start.checked_add(count));
        let valid = batch_size > 0
            && batch_size <= sage_kernels::Q4_BATCH_MAX_SIZE
            && first_patch
                .checked_add(batch_size)
                .is_some_and(|end| end <= patch_count)
            && image.patch_values.len() == patch_count * PATCH_FEATURES
            && output_elements == Some(output.len())
            && input_start.is_some()
            && input_end.is_some_and(|end| end <= image.patch_values.len());
        if !valid {
            output.fill(0.0);
            return Err(vision_error(
                "Patch batch buffers do not match the image grid",
            ));
        }
        let start = input_start.unwrap_or(0);
        let end = input_end.unwrap_or(0);
        self.projection.project_batch_with_bias_into(
            &image.patch_values[start..end],
            batch_size,
            &self.bias,
            output,
        )
    }
}

impl Qwen35VisionPositionEmbedding {
    pub fn new(table: CpuMatrix) -> CoreResult<Self> {
        if table.rows() != POSITION_GRID_SIDE * POSITION_GRID_SIDE
            || table.columns() != VISION_HIDDEN_SIZE
        {
            return Err(vision_error(
                "Qwen vision position table does not match the pinned [2304,1024] geometry",
            ));
        }
        Ok(Self { table })
    }

    /// Add the interpolated learned position vector to one patch activation.
    /// The activation is updated only after the complete result is finite.
    pub fn add_to_patch(
        &self,
        image: &PreparedQwen35Image,
        patch_index: usize,
        activation: &mut [f32],
    ) -> CoreResult<()> {
        if activation.len() != VISION_HIDDEN_SIZE
            || activation.iter().any(|value| !value.is_finite())
        {
            return Err(vision_error(
                "Patch activation does not match the Qwen vision hidden width",
            ));
        }
        let [row, column] =
            patch_grid_position(patch_index, image.patches_high, image.patches_wide)?;
        let (row0, row1, row_weight) = aligned_axis_sample(row, image.patches_high)?;
        let (column0, column1, column_weight) = aligned_axis_sample(column, image.patches_wide)?;
        let mut adjusted = Zeroizing::new(Vec::new());
        adjusted
            .try_reserve_exact(VISION_HIDDEN_SIZE)
            .map_err(|_| vision_error("Position embedding allocation was denied"))?;
        for (dimension, activation_value) in activation.iter().copied().enumerate() {
            let at = |source_row: usize, source_column: usize| {
                self.table.values()[(source_row * POSITION_GRID_SIDE + source_column)
                    * VISION_HIDDEN_SIZE
                    + dimension]
            };
            let top = f64::from(at(row0, column0)) * (1.0 - column_weight)
                + f64::from(at(row0, column1)) * column_weight;
            let bottom = f64::from(at(row1, column0)) * (1.0 - column_weight)
                + f64::from(at(row1, column1)) * column_weight;
            let positional = top * (1.0 - row_weight) + bottom * row_weight;
            let value = (f64::from(activation_value) + positional) as f32;
            if !value.is_finite() {
                return Err(vision_error(
                    "Adding the learned vision position produced a non-finite value",
                ));
            }
            adjusted.push(value);
        }
        activation.copy_from_slice(&adjusted);
        Ok(())
    }
}

impl Qwen35VisionBlock {
    pub fn new(heads: usize, weights: Qwen35VisionBlockWeights) -> CoreResult<Self> {
        let hidden_size = weights.norm1_weight.len();
        let intermediate_size = weights.mlp_input_projection.rows();
        let shape_is_valid = heads > 0
            && hidden_size > 0
            && hidden_size.is_multiple_of(heads)
            && (hidden_size / heads).is_multiple_of(4)
            && weights.norm1_bias.len() == hidden_size
            && weights.norm2_weight.len() == hidden_size
            && weights.norm2_bias.len() == hidden_size
            && weights.qkv_projection.rows() == hidden_size.saturating_mul(3)
            && weights.qkv_projection.columns() == hidden_size
            && weights.qkv_bias.len() == hidden_size.saturating_mul(3)
            && weights.attention_projection.rows() == hidden_size
            && weights.attention_projection.columns() == hidden_size
            && weights.attention_bias.len() == hidden_size
            && intermediate_size > 0
            && weights.mlp_input_projection.columns() == hidden_size
            && weights.mlp_input_bias.len() == intermediate_size
            && weights.mlp_output_projection.rows() == hidden_size
            && weights.mlp_output_projection.columns() == intermediate_size
            && weights.mlp_output_bias.len() == hidden_size;
        if !shape_is_valid
            || weights
                .norm1_weight
                .iter()
                .chain(&weights.norm1_bias)
                .chain(&weights.norm2_weight)
                .chain(&weights.norm2_bias)
                .chain(&weights.qkv_bias)
                .chain(&weights.attention_bias)
                .chain(&weights.mlp_input_bias)
                .chain(&weights.mlp_output_bias)
                .any(|value| !value.is_finite())
        {
            return Err(vision_error(
                "Qwen vision block tensors do not match finite, compatible projection geometry",
            ));
        }
        Ok(Self {
            weights,
            hidden_size,
            heads,
            head_size: hidden_size / heads,
        })
    }

    /// Run one image's flattened `[patch, hidden]` activations through the
    /// block. This scalar reference is deliberately bounded by both tensor
    /// size and attention work; larger images require an admitted optimized
    /// kernel and cannot silently consume unbounded CPU time.
    pub fn forward_image(
        &self,
        hidden_states: &[f32],
        axial_positions: &[[u64; 2]],
    ) -> CoreResult<Zeroizing<Vec<f32>>> {
        let token_count = axial_positions.len();
        let elements = token_count
            .checked_mul(self.hidden_size)
            .filter(|count| *count <= MAX_SCRATCH_VALUES)
            .ok_or_else(|| vision_error("Vision block input exceeds its tensor bound"))?;
        let _attention_work = token_count
            .checked_mul(token_count)
            .and_then(|count| count.checked_mul(self.hidden_size))
            .filter(|work| *work <= MAX_VISION_ATTENTION_MULTIPLIES)
            .ok_or_else(|| {
                vision_error("Vision attention exceeds the scalar reference work bound")
            })?;
        if token_count == 0
            || hidden_states.len() != elements
            || hidden_states.iter().any(|value| !value.is_finite())
        {
            return Err(vision_error(
                "Vision block inputs do not match its bounded token geometry",
            ));
        }

        let mut normalized = zeroed_vision_buffer(elements)?;
        for token in 0..token_count {
            let range = token * self.hidden_size..(token + 1) * self.hidden_size;
            vision_layer_norm_into(
                &hidden_states[range.clone()],
                &self.weights.norm1_weight,
                &self.weights.norm1_bias,
                &mut normalized[range],
            )?;
        }

        let qkv_width = self
            .hidden_size
            .checked_mul(3)
            .ok_or_else(|| vision_error("Vision QKV width overflow"))?;
        let qkv_elements = elements
            .checked_mul(3)
            .filter(|count| *count <= MAX_SCRATCH_VALUES)
            .ok_or_else(|| vision_error("Vision QKV scratch exceeds its tensor bound"))?;
        let mut qkv = zeroed_vision_buffer(qkv_elements)?;
        for first_token in (0..token_count).step_by(MAX_VISION_BATCH_TOKENS) {
            let batch_size = (token_count - first_token).min(MAX_VISION_BATCH_TOKENS);
            let input_start = first_token * self.hidden_size;
            let input_end = input_start + batch_size * self.hidden_size;
            let output_start = first_token * qkv_width;
            let output_end = output_start + batch_size * qkv_width;
            self.weights.qkv_projection.project_batch_with_bias_into(
                &normalized[input_start..input_end],
                batch_size,
                &self.weights.qkv_bias,
                &mut qkv[output_start..output_end],
            )?;
        }

        for (token, position) in axial_positions.iter().enumerate() {
            let token_offset = token * self.hidden_size * 3;
            for head in 0..self.heads {
                let head_offset = head * self.head_size;
                let query = token_offset + head_offset;
                let key = token_offset + self.hidden_size + head_offset;
                rotary_qwen35_vision_axial(
                    &mut qkv[query..query + self.head_size],
                    *position,
                    VISION_ROPE_THETA,
                )?;
                rotary_qwen35_vision_axial(
                    &mut qkv[key..key + self.head_size],
                    *position,
                    VISION_ROPE_THETA,
                )?;
            }
        }

        let batch_capacity = token_count.min(MAX_VISION_BATCH_TOKENS);
        let batch_hidden = batch_capacity
            .checked_mul(self.hidden_size)
            .filter(|count| *count <= MAX_SCRATCH_VALUES)
            .ok_or_else(|| vision_error("Vision batch scratch exceeds its tensor bound"))?;
        let mut context_batch = zeroed_vision_buffer(batch_hidden)?;
        let mut attended = zeroed_vision_buffer(elements)?;
        for first_token in (0..token_count).step_by(MAX_VISION_BATCH_TOKENS) {
            let batch_size = (token_count - first_token).min(MAX_VISION_BATCH_TOKENS);
            let context_count = batch_size * self.hidden_size;
            vision_attention_batch_into(
                &qkv,
                token_count,
                self.hidden_size,
                self.heads,
                self.head_size,
                first_token,
                batch_size,
                &mut context_batch[..context_count],
            )?;

            let input_start = 0;
            let input_end = context_count;
            let output_start = first_token * self.hidden_size;
            let output_end = output_start + input_end;
            self.weights
                .attention_projection
                .project_batch_with_bias_into(
                    &context_batch[input_start..input_end],
                    batch_size,
                    &self.weights.attention_bias,
                    &mut attended[output_start..output_end],
                )?;
            for (index, value) in attended[output_start..output_end].iter_mut().enumerate() {
                *value += hidden_states[output_start + index];
                if !value.is_finite() {
                    return Err(vision_error("Vision attention residual is non-finite"));
                }
            }
        }

        let intermediate_width = self.weights.mlp_input_projection.rows();
        let intermediate_capacity = batch_capacity
            .checked_mul(intermediate_width)
            .filter(|count| *count <= MAX_SCRATCH_VALUES)
            .ok_or_else(|| vision_error("Vision MLP batch scratch exceeds its tensor bound"))?;
        let mut intermediate = zeroed_vision_buffer(intermediate_capacity)?;
        let mut mlp_output = zeroed_vision_buffer(batch_hidden)?;
        let mut output = zeroed_vision_buffer(elements)?;
        for first_token in (0..token_count).step_by(MAX_VISION_BATCH_TOKENS) {
            let batch_size = (token_count - first_token).min(MAX_VISION_BATCH_TOKENS);
            let hidden_start = first_token * self.hidden_size;
            let hidden_count = batch_size * self.hidden_size;
            let hidden_end = hidden_start + hidden_count;
            for local_token in 0..batch_size {
                let local_start = local_token * self.hidden_size;
                let global_start = hidden_start + local_start;
                let range = global_start..global_start + self.hidden_size;
                vision_layer_norm_into(
                    &attended[range.clone()],
                    &self.weights.norm2_weight,
                    &self.weights.norm2_bias,
                    &mut normalized[range],
                )?;
            }

            let intermediate_count = batch_size * intermediate_width;
            self.weights
                .mlp_input_projection
                .project_batch_with_bias_into(
                    &normalized[hidden_start..hidden_end],
                    batch_size,
                    &self.weights.mlp_input_bias,
                    &mut intermediate[..intermediate_count],
                )?;
            gelu_tanh(&mut intermediate[..intermediate_count])?;
            self.weights
                .mlp_output_projection
                .project_batch_with_bias_into(
                    &intermediate[..intermediate_count],
                    batch_size,
                    &self.weights.mlp_output_bias,
                    &mut mlp_output[..hidden_count],
                )?;

            for (index, (destination, update)) in output[hidden_start..hidden_end]
                .iter_mut()
                .zip(mlp_output[..hidden_count].iter())
                .enumerate()
            {
                let value = attended[hidden_start + index] + update;
                if !value.is_finite() {
                    return Err(vision_error("Vision MLP residual is non-finite"));
                }
                *destination = value;
            }
        }
        Ok(output)
    }
}

impl Qwen35VisionPatchMerger {
    pub fn new(
        hidden_size: usize,
        output_size: usize,
        weights: Qwen35VisionPatchMergerWeights,
    ) -> CoreResult<Self> {
        let merged_input_size = hidden_size
            .checked_mul(MERGE * MERGE)
            .ok_or_else(|| vision_error("Patch merger feature width overflow"))?;
        let intermediate_size = weights.input_projection.rows();
        let valid_shape = hidden_size > 0
            && output_size > 0
            && weights.input_projection.columns() == merged_input_size
            && weights.output_projection.columns() == intermediate_size
            && weights.norm_weight.len() == hidden_size
            && weights.norm_bias.len() == hidden_size
            && intermediate_size > 0
            && weights.input_bias.len() == intermediate_size
            && weights.output_projection.rows() == output_size
            && weights.output_bias.len() == output_size;
        if !valid_shape
            || weights
                .norm_weight
                .iter()
                .chain(&weights.norm_bias)
                .chain(&weights.input_bias)
                .chain(&weights.output_bias)
                .any(|value| !value.is_finite())
        {
            return Err(vision_error(
                "Qwen patch merger tensors do not match finite, compatible projection geometry",
            ));
        }
        Ok(Self {
            weights,
            hidden_size,
            output_size,
            merged_input_size,
        })
    }

    /// Normalize each patch before concatenating adjacent 2×2 block-ordered
    /// features, then apply the checkpoint's two affine layers and tanh GELU.
    pub fn forward_image(
        &self,
        patch_states: &[f32],
        patches_high: usize,
        patches_wide: usize,
    ) -> CoreResult<Zeroizing<Vec<f32>>> {
        let patch_count = patches_high
            .checked_mul(patches_wide)
            .filter(|count| {
                *count > 0
                    && *count <= MAX_IMAGE_TOKENS * MERGE * MERGE
                    && patches_high.is_multiple_of(MERGE)
                    && patches_wide.is_multiple_of(MERGE)
            })
            .ok_or_else(|| vision_error("Patch merger grid is outside its merge-aligned bound"))?;
        let patch_elements = patch_count
            .checked_mul(self.hidden_size)
            .filter(|count| *count <= MAX_SCRATCH_VALUES)
            .ok_or_else(|| vision_error("Patch merger input exceeds its tensor bound"))?;
        if patch_states.len() != patch_elements
            || patch_states.iter().any(|value| !value.is_finite())
        {
            return Err(vision_error(
                "Patch merger inputs do not match its grid geometry",
            ));
        }
        let merged_count = patch_count / (MERGE * MERGE);
        let output_elements = merged_count
            .checked_mul(self.output_size)
            .filter(|count| *count <= MAX_SCRATCH_VALUES)
            .ok_or_else(|| vision_error("Patch merger output exceeds its tensor bound"))?;
        let mut normalized = zeroed_vision_buffer(patch_elements)?;
        for patch in 0..patch_count {
            let range = patch * self.hidden_size..(patch + 1) * self.hidden_size;
            let row = vision_layer_norm(
                &patch_states[range.clone()],
                &self.weights.norm_weight,
                &self.weights.norm_bias,
            )?;
            normalized[range].copy_from_slice(&row);
        }

        let mut output = zeroed_vision_buffer(output_elements)?;
        for merged in 0..merged_count {
            let first_patch = merged * MERGE * MERGE;
            let mut group = reserved_vision_buffer(self.merged_input_size)?;
            for within_group in 0..MERGE * MERGE {
                let start = (first_patch + within_group) * self.hidden_size;
                group.extend_from_slice(&normalized[start..start + self.hidden_size]);
            }
            let mut intermediate = Zeroizing::new(
                self.weights
                    .input_projection
                    .project_with_bias(&group, &self.weights.input_bias)?,
            );
            gelu_tanh(&mut intermediate)?;
            let merged_features = Zeroizing::new(
                self.weights
                    .output_projection
                    .project_with_bias(&intermediate, &self.weights.output_bias)?,
            );
            let start = merged * self.output_size;
            output[start..start + self.output_size].copy_from_slice(&merged_features);
        }
        Ok(output)
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }
}

impl Qwen35VisionBlockStack {
    pub fn new(blocks: Vec<Qwen35VisionBlock>) -> CoreResult<Self> {
        let hidden_size = blocks
            .first()
            .map(|block| block.hidden_size)
            .ok_or_else(|| vision_error("Qwen vision block stack cannot be empty"))?;
        let heads = blocks[0].heads;
        if blocks.len() != VISION_BLOCK_COUNT
            || blocks
                .iter()
                .any(|block| block.hidden_size != hidden_size || block.heads != heads)
        {
            return Err(vision_error(
                "Qwen vision block stack must contain 24 compatible layers",
            ));
        }
        Ok(Self {
            blocks,
            hidden_size,
        })
    }

    /// Apply all 24 blocks to one image, carrying only the current hidden
    /// state forward. Each replaced activation buffer is zeroized on drop.
    pub fn forward_image(
        &self,
        hidden_states: &[f32],
        axial_positions: &[[u64; 2]],
    ) -> CoreResult<Zeroizing<Vec<f32>>> {
        let expected_elements = axial_positions
            .len()
            .checked_mul(self.hidden_size)
            .filter(|count| *count <= MAX_SCRATCH_VALUES)
            .ok_or_else(|| vision_error("Vision stack input exceeds its tensor bound"))?;
        let _attention_work = axial_positions
            .len()
            .checked_mul(axial_positions.len())
            .and_then(|count| count.checked_mul(self.hidden_size))
            .and_then(|count| count.checked_mul(self.blocks.len()))
            .filter(|work| *work <= MAX_VISION_STACK_ATTENTION_MULTIPLIES)
            .ok_or_else(|| vision_error("Vision stack exceeds the scalar reference work bound"))?;
        if hidden_states.len() != expected_elements
            || hidden_states.iter().any(|value| !value.is_finite())
        {
            return Err(vision_error(
                "Vision stack input shape or values are invalid",
            ));
        }
        let mut state = reserved_vision_buffer(expected_elements)?;
        state.extend_from_slice(hidden_states);
        for block in &self.blocks {
            state = block.forward_image(&state, axial_positions)?;
        }
        Ok(state)
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Qwen35VisionEncoder {
    pub fn new(
        patch_embed: Qwen35PatchEmbed,
        position_embedding: Qwen35VisionPositionEmbedding,
        blocks: Qwen35VisionBlockStack,
        merger: Qwen35VisionPatchMerger,
    ) -> CoreResult<Self> {
        if blocks.hidden_size() != VISION_HIDDEN_SIZE
            || merger.hidden_size() != VISION_HIDDEN_SIZE
            || merger.output_size() != TEXT_HIDDEN_SIZE
        {
            return Err(vision_error(
                "Qwen vision encoder components do not match the pinned 4B text interface",
            ));
        }
        Ok(Self {
            patch_embed,
            position_embedding,
            blocks,
            merger,
        })
    }

    /// Encode one already-preprocessed image. The returned text-width feature
    /// matrix and all intermediate activations use zeroizing storage.
    pub fn encode_prepared_image(
        &self,
        image: &PreparedQwen35Image,
    ) -> CoreResult<Qwen35VisionEncoding> {
        let positions = image.axial_positions()?;
        let activation_count = positions
            .len()
            .checked_mul(VISION_HIDDEN_SIZE)
            .filter(|count| *count <= MAX_SCRATCH_VALUES)
            .ok_or_else(|| vision_error("Vision patch activations exceed their tensor bound"))?;
        let mut activations = reserved_vision_buffer(activation_count)?;
        activations.resize(activation_count, 0.0);
        for first_patch in (0..positions.len()).step_by(sage_kernels::Q4_BATCH_MAX_SIZE) {
            let batch_size = (positions.len() - first_patch).min(sage_kernels::Q4_BATCH_MAX_SIZE);
            let output_start = first_patch * VISION_HIDDEN_SIZE;
            let output_end = output_start + batch_size * VISION_HIDDEN_SIZE;
            self.patch_embed.project_patches_into(
                image,
                first_patch,
                batch_size,
                &mut activations[output_start..output_end],
            )?;
            for patch_index in first_patch..first_patch + batch_size {
                let start = patch_index * VISION_HIDDEN_SIZE;
                let end = start + VISION_HIDDEN_SIZE;
                self.position_embedding.add_to_patch(
                    image,
                    patch_index,
                    &mut activations[start..end],
                )?;
            }
        }

        let transformed = self.blocks.forward_image(&activations, &positions)?;
        let (patches_high, patches_wide) = image.patch_grid();
        let merged = self
            .merger
            .forward_image(&transformed, patches_high, patches_wide)?;
        let token_count = image.merged_tokens();
        let expected_values = token_count
            .checked_mul(TEXT_HIDDEN_SIZE)
            .ok_or_else(|| vision_error("Merged vision output size overflow"))?;
        if merged.len() != expected_values {
            return Err(vision_error(
                "Qwen vision merger output does not match its declared token count",
            ));
        }
        Ok(Qwen35VisionEncoding {
            token_count,
            hidden_size: TEXT_HIDDEN_SIZE,
            values: merged,
        })
    }
}

fn zeroed_vision_buffer(elements: usize) -> CoreResult<Zeroizing<Vec<f32>>> {
    let mut values = reserved_vision_buffer(elements)?;
    values.resize(elements, 0.0);
    Ok(values)
}

fn reserved_vision_buffer(capacity: usize) -> CoreResult<Zeroizing<Vec<f32>>> {
    let mut values = Zeroizing::new(Vec::new());
    values
        .try_reserve_exact(capacity)
        .map_err(|_| vision_error("Vision activation allocation was denied"))?;
    Ok(values)
}

fn zeroed_vision_f64_buffer(elements: usize) -> CoreResult<Zeroizing<Vec<f64>>> {
    let mut values = Zeroizing::new(Vec::new());
    values
        .try_reserve_exact(elements)
        .map_err(|_| vision_error("Vision attention context allocation was denied"))?;
    values.resize(elements, 0.0);
    Ok(values)
}

fn vision_layer_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
) -> CoreResult<Zeroizing<Vec<f32>>> {
    let mut normalized = zeroed_vision_buffer(input.len())?;
    vision_layer_norm_into(input, weight, bias, &mut normalized)?;
    Ok(normalized)
}

fn vision_layer_norm_into(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
) -> CoreResult<()> {
    if input.is_empty()
        || input.len() != weight.len()
        || input.len() != bias.len()
        || input.len() != output.len()
        || input
            .iter()
            .chain(weight)
            .chain(bias)
            .any(|value| !value.is_finite())
    {
        output.fill(0.0);
        return Err(vision_error(
            "Vision LayerNorm input shape or values are invalid",
        ));
    }
    let mean = input.iter().map(|value| f64::from(*value)).sum::<f64>() / input.len() as f64;
    let variance = input
        .iter()
        .map(|value| {
            let centered = f64::from(*value) - mean;
            centered * centered
        })
        .sum::<f64>()
        / input.len() as f64;
    let inverse_stddev = (variance + f64::from(VISION_LAYER_NORM_EPSILON))
        .sqrt()
        .recip();
    for index in 0..input.len() {
        let value = (((f64::from(input[index]) - mean) * inverse_stddev * f64::from(weight[index]))
            + f64::from(bias[index])) as f32;
        if !value.is_finite() {
            output.fill(0.0);
            return Err(vision_error("Vision LayerNorm output is non-finite"));
        }
        output[index] = value;
    }
    Ok(())
}

fn vision_softmax_into(logits: &[f64], weights: &mut [f64]) -> CoreResult<()> {
    if logits.is_empty()
        || logits.len() != weights.len()
        || logits.iter().any(|value| !value.is_finite())
    {
        weights.fill(0.0);
        return Err(vision_error(
            "Vision softmax input shape or values are invalid",
        ));
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        weights.fill(0.0);
        return Err(vision_error("Vision softmax row is fully masked"));
    }
    for (weight, logit) in weights.iter_mut().zip(logits) {
        *weight = (*logit - maximum).exp();
    }
    let denominator = weights.iter().sum::<f64>();
    if !denominator.is_finite() || denominator <= 0.0 {
        weights.fill(0.0);
        return Err(vision_error("Vision softmax normalization failed"));
    }
    for index in 0..weights.len() {
        weights[index] /= denominator;
        if !weights[index].is_finite() || weights[index] < 0.0 {
            weights.fill(0.0);
            return Err(vision_error("Vision softmax output is non-finite"));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn vision_attention_batch_into(
    qkv: &[f32],
    token_count: usize,
    hidden_size: usize,
    heads: usize,
    head_size: usize,
    first_query: usize,
    query_count: usize,
    output: &mut [f32],
) -> CoreResult<()> {
    let qkv_width = hidden_size.checked_mul(3);
    let qkv_elements = qkv_width.and_then(|width| token_count.checked_mul(width));
    let output_elements = query_count.checked_mul(hidden_size);
    let query_end = first_query.checked_add(query_count);
    let work = query_count
        .checked_mul(token_count)
        .and_then(|count| count.checked_mul(hidden_size));
    if token_count == 0
        || hidden_size == 0
        || heads == 0
        || head_size == 0
        || heads.checked_mul(head_size) != Some(hidden_size)
        || qkv_elements != Some(qkv.len())
        || output_elements != Some(output.len())
        || output_elements.is_none_or(|count| count > MAX_SCRATCH_VALUES)
        || query_count == 0
        || query_end.is_none_or(|end| end > token_count)
        || work.is_none_or(|count| count > MAX_VISION_ATTENTION_MULTIPLIES)
    {
        output.fill(0.0);
        return Err(vision_error(
            "Vision attention batch geometry exceeds its checked bounds",
        ));
    }

    let worker_count = sage_kernels::bounded_inference_cpu_worker_count().min(query_count);
    if work.is_some_and(|count| count >= MIN_PARALLEL_VISION_ATTENTION_MULTIPLIES)
        && worker_count > 1
        && let Some(result) = vision_attention_batch_parallel(
            qkv,
            token_count,
            hidden_size,
            heads,
            head_size,
            first_query,
            query_count,
            output,
            worker_count,
        )
    {
        if result.is_err() {
            output.fill(0.0);
        }
        return result;
    }

    let result = vision_attention_query_range(
        qkv,
        token_count,
        hidden_size,
        heads,
        head_size,
        first_query,
        query_count,
        output,
    );
    if result.is_err() {
        output.fill(0.0);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn vision_attention_batch_parallel(
    qkv: &[f32],
    token_count: usize,
    hidden_size: usize,
    heads: usize,
    head_size: usize,
    first_query: usize,
    query_count: usize,
    output: &mut [f32],
    worker_count: usize,
) -> Option<CoreResult<()>> {
    let queries_per_worker = query_count.div_ceil(worker_count);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let mut output_tail = output;
        let mut spawn_failed = false;
        for worker_index in 0..worker_count {
            let local_first = worker_index * queries_per_worker;
            if local_first >= query_count {
                break;
            }
            let local_count = (query_count - local_first).min(queries_per_worker);
            let output_count = local_count * hidden_size;
            let (worker_output, remaining) = output_tail.split_at_mut(output_count);
            output_tail = remaining;
            let absolute_first = first_query + local_first;
            let result = std::thread::Builder::new()
                .name("sage-vision-attention".into())
                .spawn_scoped(scope, move || {
                    vision_attention_query_range(
                        qkv,
                        token_count,
                        hidden_size,
                        heads,
                        head_size,
                        absolute_first,
                        local_count,
                        worker_output,
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
                    worker_error
                        .get_or_insert_with(|| vision_error("Vision attention worker panicked"));
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
fn vision_attention_query_range(
    qkv: &[f32],
    token_count: usize,
    hidden_size: usize,
    heads: usize,
    head_size: usize,
    first_query: usize,
    query_count: usize,
    output: &mut [f32],
) -> CoreResult<()> {
    let qkv_width = hidden_size * 3;
    let scaling = (head_size as f64).sqrt().recip();
    let mut context = zeroed_vision_f64_buffer(hidden_size)?;
    let mut logits = zeroed_vision_f64_buffer(token_count)?;
    let mut attention_weights = zeroed_vision_f64_buffer(token_count)?;
    for local_query in 0..query_count {
        context.fill(0.0);
        let query_offset = (first_query + local_query) * qkv_width;
        for head in 0..heads {
            let head_offset = head * head_size;
            let query_start = query_offset + head_offset;
            sage_kernels::dot_rows_f64_into(
                &qkv[query_start..query_start + head_size],
                qkv,
                qkv_width,
                hidden_size + head_offset,
                &mut logits,
            )
            .map_err(vision_error)?;
            for logit in logits.iter_mut() {
                *logit *= scaling;
                if !logit.is_finite() {
                    return Err(vision_error("Vision attention score is non-finite"));
                }
            }
            vision_softmax_into(&logits, &mut attention_weights)?;
            sage_kernels::weighted_sum_rows_f64_into(
                qkv,
                &attention_weights,
                token_count,
                qkv_width,
                hidden_size * 2 + head_offset,
                &mut context[head_offset..head_offset + head_size],
            )
            .map_err(vision_error)?;
            logits.fill(0.0);
            attention_weights.fill(0.0);
        }

        let output_start = local_query * hidden_size;
        for (destination, value) in output[output_start..output_start + hidden_size]
            .iter_mut()
            .zip(context.iter())
        {
            *destination = *value as f32;
            if !destination.is_finite() {
                return Err(vision_error("Vision attention context is non-finite"));
            }
        }
    }
    Ok(())
}

fn gelu_tanh(values: &mut [f32]) -> CoreResult<()> {
    const SQRT_TWO_OVER_PI: f64 = 0.797_884_560_802_865_4;
    for value in values {
        if !value.is_finite() {
            return Err(vision_error("Vision GELU input is non-finite"));
        }
        let input = f64::from(*value);
        let cubic = input * input * input;
        let activated =
            0.5 * input * (1.0 + (SQRT_TWO_OVER_PI * (input + 0.044715 * cubic)).tanh());
        *value = activated as f32;
        if !value.is_finite() {
            return Err(vision_error("Vision GELU output is non-finite"));
        }
    }
    Ok(())
}

fn aligned_axis_sample(index: usize, target_size: usize) -> CoreResult<(usize, usize, f64)> {
    if target_size == 0 || index >= target_size {
        return Err(vision_error("Position interpolation coordinate is invalid"));
    }
    let source = if target_size == 1 {
        0.0
    } else {
        index as f64 * (POSITION_GRID_SIDE - 1) as f64 / (target_size - 1) as f64
    };
    let lower = source.floor() as usize;
    let upper = (lower + 1).min(POSITION_GRID_SIDE - 1);
    Ok((lower, upper, source - lower as f64))
}

fn patch_grid_position(
    patch_index: usize,
    patches_high: usize,
    patches_wide: usize,
) -> CoreResult<[usize; 2]> {
    let count = patches_high
        .checked_mul(patches_wide)
        .filter(|count| {
            *count <= MAX_IMAGE_TOKENS * MERGE * MERGE
                && patches_high.is_multiple_of(MERGE)
                && patches_wide.is_multiple_of(MERGE)
        })
        .ok_or_else(|| vision_error("Vision patch grid is outside the merge-aligned bound"))?;
    if patch_index >= count {
        return Err(vision_error("Vision patch index is outside its grid"));
    }
    let patches_per_block = MERGE * MERGE;
    let blocks_wide = patches_wide / MERGE;
    let block = patch_index / patches_per_block;
    let within = patch_index % patches_per_block;
    Ok([
        (block / blocks_wide) * MERGE + within / MERGE,
        (block % blocks_wide) * MERGE + within % MERGE,
    ])
}

impl Qwen35VisionProcessor {
    pub(crate) fn parse_verified(
        package: &VerifiedQwen35Package,
        artifact: crate::model_package::VerifiedPackageArtifact<std::io::Cursor<Vec<u8>>>,
    ) -> CoreResult<Self> {
        package.require_owned_receipt(PREPROCESSOR_CONFIG_NAME, &artifact)?;
        let manifest_sha256 = package.manifest_sha256().to_owned();
        let bytes = artifact.into_verified_bytes();
        if bytes.is_empty() || bytes.len() > MAX_CONFIG_BYTES {
            return Err(vision_error(
                "Image preprocessor config exceeds its byte limit",
            ));
        }
        let raw: RawProcessorConfig = serde_json::from_slice(&bytes)
            .map_err(|_| vision_error("Image preprocessor config is malformed"))?;
        if raw.size.longest_edge != MAX_PIXELS
            || raw.size.shortest_edge != MIN_PIXELS
            || raw.patch_size != PATCH
            || raw.temporal_patch_size != TEMPORAL
            || raw.merge_size != MERGE
            || raw.image_mean != [0.5; CHANNELS]
            || raw.image_std != [0.5; CHANNELS]
            || raw.processor_class != "Qwen3VLProcessor"
            || raw.image_processor_type != "Qwen2VLImageProcessorFast"
        {
            return Err(vision_error(
                "Preprocessor config does not match Sage's pinned Qwen3.5 profile",
            ));
        }
        Ok(Self { manifest_sha256 })
    }

    pub(crate) fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// Prepare one still RGB image within the remaining visual-token budget.
    /// It is repeated into both temporal slots required by the patch kernel.
    pub fn prepare_rgb(
        &self,
        image: Qwen35RgbImage<'_>,
        maximum_vision_tokens: usize,
    ) -> CoreResult<PreparedQwen35Image> {
        let source_length = image
            .width
            .checked_mul(image.height)
            .and_then(|pixels| pixels.checked_mul(CHANNELS));
        if image.width == 0
            || image.height == 0
            || source_length != Some(image.pixels.len())
            || image.pixels.len() > MAX_RAW_IMAGE_BYTES
            || image.width.max(image.height)
                > image
                    .width
                    .min(image.height)
                    .saturating_mul(MAX_ASPECT_RATIO)
            || maximum_vision_tokens == 0
            || maximum_vision_tokens > MAX_IMAGE_TOKENS
        {
            return Err(vision_error(
                "RGB dimensions, byte count, aspect ratio, or vision-token budget is invalid",
            ));
        }

        let pixels_per_merged_token = (PATCH * MERGE).pow(2) as u64;
        let token_pixel_limit = u64::try_from(maximum_vision_tokens)
            .ok()
            .and_then(|tokens| tokens.checked_mul(pixels_per_merged_token))
            .ok_or_else(|| vision_error("Vision-token budget overflow"))?;
        let max_pixels = MAX_PIXELS.min(token_pixel_limit);
        let min_pixels = MIN_PIXELS.min(max_pixels);
        let (width, height) = resized_dimensions(
            image.width,
            image.height,
            min_pixels,
            max_pixels,
            PATCH * MERGE,
        )?;
        let resized = Zeroizing::new(resize_bicubic(image, width, height)?);
        let patches_high = height / PATCH;
        let patches_wide = width / PATCH;
        let patch_count = patches_high
            .checked_mul(patches_wide)
            .ok_or_else(|| vision_error("Patch count overflow"))?;
        let merged_tokens = patch_count / MERGE.pow(2);
        if merged_tokens > maximum_vision_tokens {
            return Err(vision_error("Prepared patch grid exceeds its token budget"));
        }
        let value_count = patch_count
            .checked_mul(PATCH_FEATURES)
            .filter(|count| *count <= MAX_SCRATCH_VALUES)
            .ok_or_else(|| vision_error("Prepared patches exceed Sage's memory bound"))?;
        let mut patch_values = Zeroizing::new(Vec::new());
        patch_values
            .try_reserve_exact(value_count)
            .map_err(|_| vision_error("Prepared image allocation was denied"))?;
        for patch_index in 0..patch_count {
            let [patch_y, patch_x] = patch_grid_position(patch_index, patches_high, patches_wide)?;
            for channel in 0..CHANNELS {
                for _time in 0..TEMPORAL {
                    for y in 0..PATCH {
                        for x in 0..PATCH {
                            let offset = ((patch_y * PATCH + y) * width + patch_x * PATCH + x)
                                * CHANNELS
                                + channel;
                            patch_values.push(f32::from(resized[offset]) / 127.5 - 1.0);
                        }
                    }
                }
            }
        }
        debug_assert_eq!(patch_values.len(), value_count);
        Ok(PreparedQwen35Image {
            width,
            height,
            patches_high,
            patches_wide,
            merged_tokens,
            patch_values,
        })
    }
}

impl PreparedQwen35Image {
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn patch_grid(&self) -> (usize, usize) {
        (self.patches_high, self.patches_wide)
    }
    pub fn merged_tokens(&self) -> usize {
        self.merged_tokens
    }

    /// Return unmerged patch coordinates in the checkpoint's spatial-merge
    /// block order, matching patch values and learned position embeddings.
    pub fn axial_positions(&self) -> CoreResult<Vec<[u64; 2]>> {
        let count = self
            .patches_high
            .checked_mul(self.patches_wide)
            .filter(|count| *count <= MAX_IMAGE_TOKENS * MERGE * MERGE)
            .ok_or_else(|| vision_error("Image patch positions exceed their grid bound"))?;
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(count)
            .map_err(|_| vision_error("Image axial-position allocation was denied"))?;
        for patch_index in 0..count {
            let [row, column] =
                patch_grid_position(patch_index, self.patches_high, self.patches_wide)?;
            positions.push([row as u64, column as u64]);
        }
        Ok(positions)
    }

    /// Return temporal/height/width text-RoPE positions for the merged image
    /// tokens in row-major order. `position_offset` is the current text offset
    /// chosen by the multimodal prompt assembler.
    pub fn text_mrope_positions(&self, position_offset: u64) -> CoreResult<Vec<[u64; 3]>> {
        if !self.patches_high.is_multiple_of(MERGE) || !self.patches_wide.is_multiple_of(MERGE) {
            return Err(vision_error("Image patch grid is not merge-aligned"));
        }
        let rows = self.patches_high / MERGE;
        let columns = self.patches_wide / MERGE;
        let count = rows
            .checked_mul(columns)
            .filter(|count| *count == self.merged_tokens && *count <= MAX_IMAGE_TOKENS)
            .ok_or_else(|| vision_error("Merged image positions exceed their grid bound"))?;
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(count)
            .map_err(|_| vision_error("Image position allocation was denied"))?;
        for row in 0..rows {
            let height = position_offset
                .checked_add(row as u64)
                .ok_or_else(|| vision_error("Image height position overflow"))?;
            for column in 0..columns {
                let width = position_offset
                    .checked_add(column as u64)
                    .ok_or_else(|| vision_error("Image width position overflow"))?;
                positions.push([position_offset, height, width]);
            }
        }
        Ok(positions)
    }

    /// Return the next shared text-axis position after this image's 2D grid.
    pub fn next_text_position(&self, position_offset: u64) -> CoreResult<u64> {
        if !self.patches_high.is_multiple_of(MERGE) || !self.patches_wide.is_multiple_of(MERGE) {
            return Err(vision_error("Image patch grid is not merge-aligned"));
        }
        let rows = self.patches_high / MERGE;
        let columns = self.patches_wide / MERGE;
        position_offset
            .checked_add(rows.max(columns) as u64)
            .ok_or_else(|| vision_error("Next text RoPE position overflow"))
    }

    pub fn patch_values(&self) -> &[f32] {
        &self.patch_values
    }
}

fn resized_dimensions(
    source_width: usize,
    source_height: usize,
    minimum_pixels: u64,
    maximum_pixels: u64,
    factor: usize,
) -> CoreResult<(usize, usize)> {
    let source_pixels = source_width
        .checked_mul(source_height)
        .ok_or_else(|| vision_error("Image area overflow"))? as f64;
    let mut width = round_multiple(source_width as f64, factor)?;
    let mut height = round_multiple(source_height as f64, factor)?;
    let max = usize::try_from(maximum_pixels)
        .map_err(|_| vision_error("Image maximum exceeds platform bounds"))?;
    let min = usize::try_from(minimum_pixels)
        .map_err(|_| vision_error("Image minimum exceeds platform bounds"))?;

    let initial_area = width
        .checked_mul(height)
        .ok_or_else(|| vision_error("Rounded image area overflow"))?;
    if initial_area > max {
        let scale = (maximum_pixels as f64 / source_pixels).sqrt();
        width = floor_scaled_multiple(source_width, scale, factor)?;
        height = floor_scaled_multiple(source_height, scale, factor)?;
    } else if initial_area < min {
        let scale = (minimum_pixels as f64 / source_pixels).sqrt();
        width = ceil_scaled_multiple(source_width, scale, factor)?;
        height = ceil_scaled_multiple(source_height, scale, factor)?;
        if width
            .checked_mul(height)
            .is_none_or(|rounded_area| rounded_area > max)
        {
            // At very small per-call token budgets, preserving the checkpoint's
            // usual minimum pixel area can conflict with its 32-pixel grid. The
            // caller's explicit token ceiling wins; use the largest simple
            // aspect-preserving grid that fits under that ceiling.
            let scale = (maximum_pixels as f64 / source_pixels).sqrt();
            width = floor_scaled_multiple(source_width, scale, factor)?;
            height = floor_scaled_multiple(source_height, scale, factor)?;
        }
    }

    let final_area = width
        .checked_mul(height)
        .ok_or_else(|| vision_error("Resized image area overflow"))?;
    if final_area > max || !width.is_multiple_of(factor) || !height.is_multiple_of(factor) {
        return Err(vision_error(
            "Resized image cannot satisfy the admitted pixel grid",
        ));
    }
    Ok((width, height))
}

fn round_multiple(value: f64, factor: usize) -> CoreResult<usize> {
    if !value.is_finite() || value <= 0.0 || factor == 0 {
        return Err(vision_error("Resized image dimension is invalid"));
    }
    Ok(((value / factor as f64).round().max(1.0) * factor as f64) as usize)
}

fn floor_scaled_multiple(source: usize, scale: f64, factor: usize) -> CoreResult<usize> {
    scaled_multiple(source, scale, factor, f64::floor)
}

fn ceil_scaled_multiple(source: usize, scale: f64, factor: usize) -> CoreResult<usize> {
    scaled_multiple(source, scale, factor, f64::ceil)
}

fn scaled_multiple(
    source: usize,
    scale: f64,
    factor: usize,
    rounding: fn(f64) -> f64,
) -> CoreResult<usize> {
    if !scale.is_finite() || scale <= 0.0 || factor == 0 {
        return Err(vision_error("Image resize scale is invalid"));
    }
    let units = rounding(source as f64 * scale / factor as f64).max(1.0);
    let resized = units * factor as f64;
    if !resized.is_finite() || resized > usize::MAX as f64 {
        return Err(vision_error(
            "Resized image dimension exceeds platform bounds",
        ));
    }
    Ok(resized as usize)
}

/// Four-tap separable bicubic resampling with half-pixel centers, clamped
/// borders, and cubic parameter -0.5. This code owns pixel behavior directly.
fn resize_bicubic(image: Qwen35RgbImage<'_>, width: usize, height: usize) -> CoreResult<Vec<u8>> {
    let intermediate_len = image
        .height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(CHANNELS))
        .filter(|len| *len <= MAX_SCRATCH_VALUES)
        .ok_or_else(|| vision_error("Image resize scratch exceeds Sage's memory bound"))?;
    let mut intermediate = Vec::new();
    intermediate
        .try_reserve_exact(intermediate_len)
        .map_err(|_| vision_error("Image resize allocation was denied"))?;
    intermediate.resize(intermediate_len, 0.0f32);
    let horizontal_scale = image.width as f64 / width as f64;
    let horizontal_filter_scale = horizontal_scale.max(1.0);
    for y in 0..image.height {
        for x in 0..width {
            let source_x = (x as f64 + 0.5) * horizontal_scale - 0.5;
            let support = 2.0 * horizontal_filter_scale;
            let first_source_x = (source_x - support).floor() as isize;
            let last_source_x = (source_x + support).ceil() as isize;
            for channel in 0..CHANNELS {
                let mut value = 0.0;
                let mut weights = 0.0;
                for source_index in first_source_x..=last_source_x {
                    let index =
                        source_index.clamp(0, image.width.saturating_sub(1) as isize) as usize;
                    let weight =
                        cubic_weight((source_x - source_index as f64) / horizontal_filter_scale)
                            / horizontal_filter_scale;
                    value +=
                        f64::from(image.pixels[(y * image.width + index) * CHANNELS + channel])
                            * weight;
                    weights += weight;
                }
                intermediate[(y * width + x) * CHANNELS + channel] = (value / weights) as f32;
            }
        }
    }

    let output_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(CHANNELS))
        .ok_or_else(|| vision_error("Resized image buffer size overflow"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| vision_error("Resized image allocation was denied"))?;
    output.resize(output_len, 0u8);
    let vertical_scale = image.height as f64 / height as f64;
    let vertical_filter_scale = vertical_scale.max(1.0);
    for y in 0..height {
        let source_y = (y as f64 + 0.5) * vertical_scale - 0.5;
        let support = 2.0 * vertical_filter_scale;
        let first_source_y = (source_y - support).floor() as isize;
        let last_source_y = (source_y + support).ceil() as isize;
        for x in 0..width {
            for channel in 0..CHANNELS {
                let mut value = 0.0;
                let mut weights = 0.0;
                for source_index in first_source_y..=last_source_y {
                    let index =
                        source_index.clamp(0, image.height.saturating_sub(1) as isize) as usize;
                    let weight =
                        cubic_weight((source_y - source_index as f64) / vertical_filter_scale)
                            / vertical_filter_scale;
                    value +=
                        f64::from(intermediate[(index * width + x) * CHANNELS + channel]) * weight;
                    weights += weight;
                }
                output[(y * width + x) * CHANNELS + channel] =
                    (value / weights).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(output)
}

fn cubic_weight(distance: f64) -> f64 {
    let x = distance.abs();
    if x < 1.0 {
        1.5 * x.powi(3) - 2.5 * x.powi(2) + 1.0
    } else if x < 2.0 {
        -0.5 * x.powi(3) + 2.5 * x.powi(2) - 4.0 * x + 2.0
    } else {
        0.0
    }
}

fn vision_error(message: &str) -> CoreError {
    CoreError::Model(message.into())
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_PIXELS, MIN_PIXELS, PATCH, Qwen35PatchEmbed, Qwen35RgbImage, Qwen35VisionBlock,
        Qwen35VisionBlockStack, Qwen35VisionBlockWeights, Qwen35VisionPatchMerger,
        Qwen35VisionPatchMergerWeights, Qwen35VisionProcessor, resized_dimensions,
    };
    use crate::{inference_cpu::CpuMatrix, qwen35::Qwen35ProjectionMatrix};

    fn processor() -> Qwen35VisionProcessor {
        Qwen35VisionProcessor {
            manifest_sha256: "a".repeat(64),
        }
    }

    fn reference_vision_block() -> Qwen35VisionBlock {
        let hidden = 4;
        let intermediate = 6;
        let mut qkv = vec![0.0; hidden * 3 * hidden];
        for dimension in 0..hidden {
            qkv[(hidden * 2 + dimension) * hidden + dimension] = 1.0;
        }
        let mut attention = vec![0.0; hidden * hidden];
        for dimension in 0..hidden {
            attention[dimension * hidden + dimension] = 1.0;
        }
        Qwen35VisionBlock::new(
            1,
            Qwen35VisionBlockWeights {
                norm1_weight: vec![1.0; hidden],
                norm1_bias: vec![0.0; hidden],
                qkv_projection: Qwen35ProjectionMatrix::from(
                    CpuMatrix::new(hidden * 3, hidden, qkv).unwrap(),
                ),
                qkv_bias: vec![0.0; hidden * 3],
                attention_projection: Qwen35ProjectionMatrix::from(
                    CpuMatrix::new(hidden, hidden, attention).unwrap(),
                ),
                attention_bias: vec![0.0; hidden],
                norm2_weight: vec![1.0; hidden],
                norm2_bias: vec![0.0; hidden],
                mlp_input_projection: Qwen35ProjectionMatrix::from(
                    CpuMatrix::new(intermediate, hidden, vec![0.0; intermediate * hidden]).unwrap(),
                ),
                mlp_input_bias: vec![0.0; intermediate],
                mlp_output_projection: Qwen35ProjectionMatrix::from(
                    CpuMatrix::new(hidden, intermediate, vec![0.0; hidden * intermediate]).unwrap(),
                ),
                mlp_output_bias: vec![0.0; hidden],
            },
        )
        .unwrap()
    }

    fn zero_q4_projection(rows: usize, columns: usize) -> Qwen35ProjectionMatrix {
        Qwen35ProjectionMatrix::from(
            CpuMatrix::new(rows, columns, vec![0.0; rows * columns]).unwrap(),
        )
        .quantize_q4(16)
        .unwrap()
    }

    fn reference_patch_merger() -> Qwen35VisionPatchMerger {
        let hidden = 4;
        let merged_hidden = hidden * 4;
        let mut first_projection = vec![0.0; 4 * merged_hidden];
        for patch in 0..4 {
            first_projection[patch * merged_hidden + patch * hidden] = 1.0;
        }
        let mut second_projection = vec![0.0; 4 * 4];
        for dimension in 0..4 {
            second_projection[dimension * 4 + dimension] = 1.0;
        }
        Qwen35VisionPatchMerger::new(
            hidden,
            4,
            Qwen35VisionPatchMergerWeights {
                norm_weight: vec![1.0; hidden],
                norm_bias: vec![0.0; hidden],
                input_projection: Qwen35ProjectionMatrix::from(
                    CpuMatrix::new(4, merged_hidden, first_projection).unwrap(),
                ),
                input_bias: vec![0.0; 4],
                output_projection: Qwen35ProjectionMatrix::from(
                    CpuMatrix::new(4, 4, second_projection).unwrap(),
                ),
                output_bias: vec![0.0; 4],
            },
        )
        .unwrap()
    }

    #[test]
    fn resize_rounds_to_patch_merge_grid_and_obeys_pixel_budget() {
        let (width, height) = resized_dimensions(1920, 1080, MIN_PIXELS, 1_048_576, 32).unwrap();
        assert_eq!(width % 32, 0);
        assert_eq!(height % 32, 0);
        assert!((MIN_PIXELS as usize..=1_048_576).contains(&(width * height)));
        assert!(width > height);
        assert_eq!(MAX_PIXELS, 16_777_216);
    }

    #[test]
    fn small_token_budget_wins_when_grid_rounding_conflicts_with_model_minimum() {
        let pixels = vec![127u8; 120 * 80 * 3];
        let result = processor()
            .prepare_rgb(
                Qwen35RgbImage {
                    width: 120,
                    height: 80,
                    pixels: &pixels,
                },
                16,
            )
            .unwrap();
        assert!(result.merged_tokens() <= 16);
        assert!(result.width() * result.height() <= 16 * 32 * 32);
        let narrow_budget = processor()
            .prepare_rgb(
                Qwen35RgbImage {
                    width: 120,
                    height: 80,
                    pixels: &pixels,
                },
                65,
            )
            .unwrap();
        assert!(narrow_budget.merged_tokens() <= 65);
    }

    #[test]
    fn still_rgb_image_yields_bounded_temporal_normalized_patch_rows() {
        let pixels = vec![255u8; 32 * 32 * 3];
        let result = processor()
            .prepare_rgb(
                Qwen35RgbImage {
                    width: 32,
                    height: 32,
                    pixels: &pixels,
                },
                16,
            )
            .unwrap();
        assert_eq!(result.patch_grid(), (8, 8));
        assert_eq!(result.merged_tokens(), 16);
        assert_eq!(result.patch_values().len(), 64 * 1536);
        let axial_positions = result.axial_positions().unwrap();
        assert_eq!(axial_positions.len(), 64);
        assert_eq!(axial_positions[0], [0, 0]);
        assert_eq!(axial_positions[1], [0, 1]);
        assert_eq!(axial_positions[2], [1, 0]);
        assert_eq!(axial_positions[4], [0, 2]);
        assert_eq!(axial_positions[16], [2, 0]);
        let positions = result.text_mrope_positions(7).unwrap();
        assert_eq!(positions.len(), 16);
        assert_eq!(positions[0], [7, 7, 7]);
        assert_eq!(positions[1], [7, 7, 8]);
        assert_eq!(positions[4], [7, 8, 7]);
        assert_eq!(result.next_text_position(7).unwrap(), 11);
        assert!(result.patch_values().iter().all(|value| *value == 1.0));
    }

    #[test]
    fn prepared_patch_rows_follow_spatial_merge_block_order() {
        let mut pixels = Vec::with_capacity(128 * 128 * 3);
        for y in 0..128 {
            for x in 0..128 {
                let patch_y = y / PATCH;
                let patch_x = x / PATCH;
                let value = ((patch_y * 8 + patch_x) * 4) as u8;
                pixels.extend_from_slice(&[value, value, value]);
            }
        }
        let result = processor()
            .prepare_rgb(
                Qwen35RgbImage {
                    width: 128,
                    height: 128,
                    pixels: &pixels,
                },
                16,
            )
            .unwrap();
        assert_eq!(result.patch_grid(), (8, 8));
        let positions = result.axial_positions().unwrap();
        for (patch_index, [row, column]) in positions.into_iter().enumerate() {
            let source = ((row as usize * 8 + column as usize) * 4) as f32;
            let expected = source / 127.5 - 1.0;
            assert!((result.patch_values()[patch_index * 1536] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn learned_vision_positions_interpolate_with_aligned_grid_corners() {
        let mut position_values = vec![0.0; 2304 * 1024];
        for row in 0..48 {
            for column in 0..48 {
                position_values[(row * 48 + column) * 1024] = (row * 100 + column) as f32;
            }
        }
        let position_embedding = super::Qwen35VisionPositionEmbedding::new(
            CpuMatrix::new(2304, 1024, position_values).unwrap(),
        )
        .unwrap();
        let pixels = vec![127; 32 * 32 * 3];
        let image = processor()
            .prepare_rgb(
                Qwen35RgbImage {
                    width: 32,
                    height: 32,
                    pixels: &pixels,
                },
                16,
            )
            .unwrap();
        assert_eq!(image.patch_grid(), (8, 8));

        let mut first = vec![0.0; 1024];
        position_embedding
            .add_to_patch(&image, 0, &mut first)
            .unwrap();
        assert_eq!(first[0], 0.0);

        let mut right = vec![0.0; 1024];
        position_embedding
            .add_to_patch(&image, 1, &mut right)
            .unwrap();
        assert!((right[0] - (47.0 / 7.0)).abs() < 1e-5);

        let mut lower = vec![0.0; 1024];
        position_embedding
            .add_to_patch(&image, 2, &mut lower)
            .unwrap();
        assert!((lower[0] - (47.0 / 7.0 * 100.0)).abs() < 1e-4);
        assert!(
            position_embedding
                .add_to_patch(&image, 64, &mut lower)
                .is_err()
        );
    }

    #[test]
    fn vision_block_matches_uniform_noncausal_attention_and_residuals() {
        let block = reference_vision_block();
        let hidden = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let coordinates = [[0, 0], [0, 1]];
        let output = block
            .forward_image(&hidden, &coordinates)
            .expect("bounded vision block reference");
        let normalized_scale = (0.1875f64 + 1e-6).sqrt().recip();
        let expected_attention = [
            0.25 * normalized_scale,
            0.25 * normalized_scale,
            -0.25 * normalized_scale,
            -0.25 * normalized_scale,
        ];
        let expected = [
            1.0 + expected_attention[0] as f32,
            expected_attention[1] as f32,
            expected_attention[2] as f32,
            expected_attention[3] as f32,
            expected_attention[0] as f32,
            1.0 + expected_attention[1] as f32,
            expected_attention[2] as f32,
            expected_attention[3] as f32,
        ];
        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert!(block.forward_image(&[f32::NAN; 8], &coordinates).is_err());
    }

    #[test]
    fn vision_block_batches_q4_projections_and_preserves_zero_weight_residual() {
        let hidden = 4;
        let intermediate = 6;
        let block = Qwen35VisionBlock::new(
            1,
            Qwen35VisionBlockWeights {
                norm1_weight: vec![1.0; hidden],
                norm1_bias: vec![0.0; hidden],
                qkv_projection: zero_q4_projection(hidden * 3, hidden),
                qkv_bias: vec![0.0; hidden * 3],
                attention_projection: zero_q4_projection(hidden, hidden),
                attention_bias: vec![0.0; hidden],
                norm2_weight: vec![1.0; hidden],
                norm2_bias: vec![0.0; hidden],
                mlp_input_projection: zero_q4_projection(intermediate, hidden),
                mlp_input_bias: vec![0.0; intermediate],
                mlp_output_projection: zero_q4_projection(hidden, intermediate),
                mlp_output_bias: vec![0.0; hidden],
            },
        )
        .unwrap();
        let token_count = 64;
        let hidden_states = (0..token_count * hidden)
            .map(|index| (index as f32 - 91.0) * 0.03125)
            .collect::<Vec<_>>();
        let positions = (0..token_count)
            .map(|index| [index as u64 / 8, index as u64 % 8])
            .collect::<Vec<_>>();

        let output = block
            .forward_image(&hidden_states, &positions)
            .expect("Q4-projected vision batch");
        assert_eq!(output.as_slice(), hidden_states.as_slice());
    }

    #[test]
    fn vision_attention_parallel_worker_ranges_match_single_range() {
        const TOKENS: usize = 64;
        const HIDDEN: usize = 256;
        const HEADS: usize = 8;
        const HEAD_SIZE: usize = HIDDEN / HEADS;
        const QKV_WIDTH: usize = HIDDEN * 3;
        let qkv = (0..TOKENS * QKV_WIDTH)
            .map(|index| ((index.wrapping_mul(17) % 127) as f32 - 63.0) * 0.003)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; TOKENS * HIDDEN];
        super::vision_attention_query_range(
            &qkv,
            TOKENS,
            HIDDEN,
            HEADS,
            HEAD_SIZE,
            0,
            TOKENS,
            &mut expected,
        )
        .expect("single-range attention reference");

        let mut actual = vec![0.0; TOKENS * HIDDEN];
        let result = super::vision_attention_batch_parallel(
            &qkv,
            TOKENS,
            HIDDEN,
            HEADS,
            HEAD_SIZE,
            0,
            TOKENS,
            &mut actual,
            2,
        )
        .expect("two bounded workers can be started");
        result.expect("parallel attention workers complete");
        for (expected, actual) in expected.iter().zip(&actual) {
            assert!((expected - actual).abs() <= 1.0e-7);
        }
    }

    #[test]
    #[ignore = "release-only synthetic comparison of first-party vision attention kernels"]
    fn vision_attention_kernel_latency_measurement() {
        use std::time::Instant;

        const TOKENS: usize = 128;
        const HIDDEN: usize = 1024;
        const HEADS: usize = 16;
        const HEAD_SIZE: usize = HIDDEN / HEADS;
        const QKV_WIDTH: usize = HIDDEN * 3;
        let qkv = (0..TOKENS * QKV_WIDTH)
            .map(|index| ((index.wrapping_mul(29) % 251) as f32 - 125.0) * 0.0008)
            .collect::<Vec<_>>();
        let mut reference = vec![0.0f32; TOKENS * HIDDEN];
        let mut optimized = vec![0.0f32; TOKENS * HIDDEN];
        let mut reference_scores = vec![0.0f64; TOKENS];
        let mut reference_logits = vec![0.0f32; TOKENS];
        let mut reference_exponentials = vec![0.0f64; TOKENS];
        let mut reference_weights = vec![0.0f32; TOKENS];

        scalar_vision_attention_reference_into(
            &qkv,
            TOKENS,
            HIDDEN,
            HEADS,
            HEAD_SIZE,
            &mut reference,
            &mut reference_scores,
            &mut reference_logits,
            &mut reference_exponentials,
            &mut reference_weights,
        );
        first_party_vision_attention_into(&qkv, TOKENS, HIDDEN, HEADS, HEAD_SIZE, &mut optimized);
        let max_difference = reference
            .iter()
            .zip(&optimized)
            .map(|(expected, actual)| (f64::from(*expected) - f64::from(*actual)).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_difference <= 1.0e-7,
            "maximum difference {max_difference}"
        );

        let mut reference_samples = Vec::with_capacity(51);
        let mut optimized_samples = Vec::with_capacity(50);
        for iteration in 0..101 {
            let start = Instant::now();
            if iteration % 2 == 0 {
                scalar_vision_attention_reference_into(
                    &qkv,
                    TOKENS,
                    HIDDEN,
                    HEADS,
                    HEAD_SIZE,
                    &mut reference,
                    &mut reference_scores,
                    &mut reference_logits,
                    &mut reference_exponentials,
                    &mut reference_weights,
                );
                reference_samples.push(start.elapsed().as_nanos());
            } else {
                first_party_vision_attention_into(
                    &qkv,
                    TOKENS,
                    HIDDEN,
                    HEADS,
                    HEAD_SIZE,
                    &mut optimized,
                );
                optimized_samples.push(start.elapsed().as_nanos());
            }
        }
        reference_samples.sort_unstable();
        optimized_samples.sort_unstable();
        let reference_p50 = reference_samples[reference_samples.len() / 2];
        let reference_p95 = reference_samples[(reference_samples.len() * 95).div_ceil(100) - 1];
        let optimized_p50 = optimized_samples[optimized_samples.len() / 2];
        let optimized_p95 = optimized_samples[(optimized_samples.len() * 95).div_ceil(100) - 1];
        println!(
            "vision attention ns scalar_p50={reference_p50} scalar_p95={reference_p95} kernel_p50={optimized_p50} kernel_p95={optimized_p95} speedup={:.3} max_abs_difference={max_difference}",
            reference_p50 as f64 / optimized_p50 as f64
        );
    }

    #[test]
    fn vision_reference_rejects_attention_work_above_its_cpu_bound() {
        let block = reference_vision_block();
        let token_count = 8_192;
        let hidden = vec![0.0; token_count * 4];
        let coordinates = vec![[0, 0]; token_count];
        assert!(block.forward_image(&hidden, &coordinates).is_err());
    }

    #[test]
    fn patch_merger_normalizes_then_preserves_spatial_block_groups() {
        let merger = reference_patch_merger();
        let values = [1.0f32, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let mut patches = Vec::new();
        for value in values {
            patches.extend_from_slice(&[value, 0.0, 0.0, 0.0]);
        }
        let merged = merger
            .forward_image(&patches, 2, 4)
            .expect("merge-aligned image");
        assert_eq!(merged.len(), 8);
        for group in 0..2 {
            for within_group in 0..4 {
                let value = values[group * 4 + within_group];
                let mean = f64::from(value) / 4.0;
                let variance = 3.0 * f64::from(value).powi(2) / 16.0;
                let normalized = (f64::from(value) - mean) / (variance + 1e-6).sqrt();
                let expected = (0.5
                    * normalized
                    * (1.0
                        + (0.797_884_560_802_865_4 * (normalized + 0.044715 * normalized.powi(3)))
                            .tanh())) as f32;
                assert!((merged[group * 4 + within_group] - expected).abs() < 1e-5);
            }
        }
        assert!(merger.forward_image(&patches, 1, 8).is_err());
    }

    #[test]
    fn vision_stack_requires_and_executes_all_24_compatible_blocks() {
        let blocks = (0..24)
            .map(|_| reference_vision_block())
            .collect::<Vec<_>>();
        let stack = Qwen35VisionBlockStack::new(blocks).expect("24 compatible layers");
        let output = stack
            .forward_image(&[0.0; 8], &[[0, 0], [0, 1]])
            .expect("bounded full stack");
        assert_eq!(output.as_slice(), &[0.0; 8]);

        let incomplete = (0..23)
            .map(|_| reference_vision_block())
            .collect::<Vec<_>>();
        assert!(Qwen35VisionBlockStack::new(incomplete).is_err());
    }

    #[test]
    fn rgb_channels_are_normalized_in_model_order() {
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for _ in 0..32 * 32 {
            pixels.extend_from_slice(&[255, 0, 127]);
        }
        let result = processor()
            .prepare_rgb(
                Qwen35RgbImage {
                    width: 32,
                    height: 32,
                    pixels: &pixels,
                },
                16,
            )
            .unwrap();
        let values_per_channel = 2 * PATCH * PATCH;
        let first_patch = result.patch_values();
        assert!(
            first_patch[..values_per_channel]
                .iter()
                .all(|value| *value == 1.0)
        );
        assert!(
            first_patch[values_per_channel..2 * values_per_channel]
                .iter()
                .all(|value| *value == -1.0)
        );
        assert!(
            first_patch[2 * values_per_channel..3 * values_per_channel]
                .iter()
                .all(|value| (*value - (127.0 / 127.5 - 1.0)).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn patch_embedding_uses_checkpoint_channel_order_and_returns_zeroizing_output() {
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for _ in 0..32 * 32 {
            pixels.extend_from_slice(&[255, 0, 127]);
        }
        let image = processor()
            .prepare_rgb(
                Qwen35RgbImage {
                    width: 32,
                    height: 32,
                    pixels: &pixels,
                },
                16,
            )
            .unwrap();
        let mut weights = vec![0.0; 1024 * 1536];
        weights[0] = 2.0;
        weights[512] = 3.0;
        let projection = Qwen35ProjectionMatrix::from(CpuMatrix::new(1024, 1536, weights).unwrap());
        let mut bias = vec![0.0; 1024];
        bias[0] = 0.5;
        bias[1] = -2.0;
        let patch_embed = Qwen35PatchEmbed::new(projection, bias).unwrap();
        let embedding = patch_embed.project_patch(&image, 0).unwrap();
        assert_eq!(embedding.len(), 1024);
        assert_eq!(embedding[0], -0.5);
        assert_eq!(embedding[1], -2.0);
        assert!(patch_embed.project_patch(&image, 64).is_err());

        let quantized = Qwen35ProjectionMatrix::from(
            CpuMatrix::new(
                1024,
                1536,
                (0..1024 * 1536)
                    .map(|index| match index {
                        0 => 2.0,
                        512 => 3.0,
                        _ => 0.0,
                    })
                    .collect(),
            )
            .unwrap(),
        )
        .quantize_q4(128)
        .unwrap();
        let mut quantized_bias = vec![0.0; 1024];
        quantized_bias[0] = 0.5;
        quantized_bias[1] = -2.0;
        let quantized_embed = Qwen35PatchEmbed::new(quantized, quantized_bias).unwrap();
        let (patches_high, patches_wide) = image.patch_grid();
        let patch_count = patches_high * patches_wide;
        let mut batched = vec![0.0; patch_count * 1024];
        for first_patch in (0..patch_count).step_by(sage_kernels::Q4_BATCH_MAX_SIZE) {
            let batch_size = (patch_count - first_patch).min(sage_kernels::Q4_BATCH_MAX_SIZE);
            let start = first_patch * 1024;
            let end = start + batch_size * 1024;
            quantized_embed
                .project_patches_into(&image, first_patch, batch_size, &mut batched[start..end])
                .unwrap();
        }
        for patch_index in 0..patch_count {
            let expected = patch_embed.project_patch(&image, patch_index).unwrap();
            let actual = &batched[patch_index * 1024..(patch_index + 1) * 1024];
            for (observed, reference) in actual.iter().zip(expected.iter()) {
                assert!((observed - reference).abs() <= 1.0e-6);
            }
        }
    }

    #[test]
    fn bicubic_reduction_suppresses_a_high_frequency_checkerboard() {
        let mut pixels = Vec::with_capacity(64 * 64 * 3);
        for y in 0usize..64 {
            for x in 0usize..64 {
                let value = if (x + y).is_multiple_of(2) { 0 } else { 255 };
                pixels.extend_from_slice(&[value, value, value]);
            }
        }
        let result = processor()
            .prepare_rgb(
                Qwen35RgbImage {
                    width: 64,
                    height: 64,
                    pixels: &pixels,
                },
                1,
            )
            .unwrap();
        assert_eq!(result.width(), 32);
        assert_eq!(result.height(), 32);
        let first_patch_channel = &result.patch_values()[..PATCH * PATCH];
        for row in 4..12 {
            assert!(
                first_patch_channel[row * PATCH + 4..row * PATCH + 12]
                    .iter()
                    .all(|value| value.abs() < 0.02)
            );
        }
    }

    #[test]
    fn image_source_and_budget_bounds_fail_closed() {
        let image = Qwen35RgbImage {
            width: 2,
            height: 2,
            pixels: &[0, 0, 0],
        };
        assert!(processor().prepare_rgb(image, 16).is_err());
        let pixels = [0u8; 3];
        assert!(
            processor()
                .prepare_rgb(
                    Qwen35RgbImage {
                        width: 1,
                        height: 1,
                        pixels: &pixels,
                    },
                    4_097,
                )
                .is_err()
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_vision_attention_reference_into(
        qkv: &[f32],
        tokens: usize,
        hidden: usize,
        heads: usize,
        head_size: usize,
        output: &mut [f32],
        scores: &mut [f64],
        logits: &mut [f32],
        exponentials: &mut [f64],
        weights: &mut [f32],
    ) {
        output.fill(0.0);
        let qkv_width = hidden * 3;
        let scaling = (head_size as f64).sqrt().recip();
        for query_token in 0..tokens {
            let query_offset = query_token * qkv_width;
            let context = &mut output[query_token * hidden..(query_token + 1) * hidden];
            for head in 0..heads {
                let head_offset = head * head_size;
                let query =
                    &qkv[query_offset + head_offset..query_offset + head_offset + head_size];
                for (key_token, score) in scores.iter_mut().enumerate() {
                    let key_offset = key_token * qkv_width + hidden + head_offset;
                    *score = query
                        .iter()
                        .zip(&qkv[key_offset..key_offset + head_size])
                        .fold(0.0f64, |sum, (query, key)| {
                            sum + f64::from(*query) * f64::from(*key)
                        });
                    logits[key_token] = (*score * scaling) as f32;
                }
                let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denominator = 0.0f64;
                for (exponential, logit) in exponentials.iter_mut().zip(logits.iter()) {
                    *exponential = f64::from(*logit - maximum).exp();
                    denominator += *exponential;
                }
                for (weight, exponential) in weights.iter_mut().zip(&exponentials[..tokens]) {
                    *weight = (*exponential / denominator) as f32;
                }
                for dimension in 0..head_size {
                    let mut sum = 0.0f64;
                    for (key_token, weight) in weights.iter().enumerate() {
                        let value_offset = key_token * qkv_width + hidden * 2 + head_offset;
                        sum += f64::from(*weight) * f64::from(qkv[value_offset + dimension]);
                    }
                    context[head_offset + dimension] = sum as f32;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn first_party_vision_attention_into(
        qkv: &[f32],
        tokens: usize,
        hidden: usize,
        heads: usize,
        head_size: usize,
        output: &mut [f32],
    ) {
        super::vision_attention_batch_into(
            qkv, tokens, hidden, heads, head_size, 0, tokens, output,
        )
        .expect("bounded first-party vision attention");
    }
}
