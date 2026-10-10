//! Bounded assembly of the pinned Qwen text stack from a signed package.
//!
//! The loader keeps the package manifest identity attached to parsed metadata,
//! consumes already-verified open shard handles, quantizes rank-2 tensors as
//! they stream, and withholds the model until both shard digests are rechecked.

use std::{
    collections::BTreeSet,
    io::{Cursor, Read, Seek},
};

use crate::{
    Qwen35Error, Qwen35Result,
    qwen35_vision::{
        PreparedQwen35Image, Qwen35RgbImage, Qwen35VisionBlock, Qwen35VisionBlockStack,
        Qwen35VisionBlockWeights, Qwen35VisionEncoder, Qwen35VisionEncoding,
        Qwen35VisionPatchMerger, Qwen35VisionPatchMergerWeights, Qwen35VisionPositionEmbedding,
        Qwen35VisionProcessor,
    },
    resource::{MemoryEnvelope, Reservation, ResourceGovernor},
};
use sage_inference_math::CpuMatrix;
use sage_model_package::{
    VerifiedPackageArtifact, VerifiedQwen35Package, safetensors::SafeTensorReader,
};
use sage_qwen_tokenizer::{EncodedImageSpan, Qwen35Tokenizer};

use crate::qwen35::{
    ATTENTION_OUTPUT_SIZE, DELTA_QKV_SIZE, DELTA_VALUE_SIZE, HIDDEN_SIZE, INTERMEDIATE_SIZE,
    KEY_VALUE_PROJECTION_SIZE, QUERY_PROJECTION_SIZE, QWEN35_4B_CONV_KERNEL_SIZE,
    QWEN35_4B_HEAD_DIMENSION, QWEN35_4B_KEY_HEADS, QWEN35_4B_MAX_POSITION_EMBEDDINGS,
    QWEN35_4B_VALUE_HEADS, Qwen35Config, Qwen35DecoderLayer, Qwen35FullAttentionBlock,
    Qwen35FullAttentionTensorNames, Qwen35FullAttentionWeights, Qwen35LinearAttentionBlock,
    Qwen35LinearAttentionTensorNames, Qwen35LinearAttentionWeights, Qwen35Mlp,
    Qwen35ProjectionMatrix, Qwen35TensorSpec, Qwen35TensorStream, Qwen35TextDecoder,
    Qwen35TokenMixer, Qwen35WeightIndex, SAGE_CONTEXT_LIMIT, VOCABULARY_SIZE,
};

const SHARD_ONE: &str = "model.safetensors-00001-of-00002.safetensors";
const SHARD_TWO: &str = "model.safetensors-00002-of-00002.safetensors";
const INDEX_NAME: &str = "model.safetensors.index.json";
const CONFIG_NAME: &str = "config.json";
const TOKENIZER_NAME: &str = "tokenizer.json";
const EMBEDDING_NAME: &str = "model.language_model.embed_tokens.weight";
const FINAL_NORM_NAME: &str = "model.language_model.norm.weight";
const VISION_PATCH_WEIGHT: &str = "model.visual.patch_embed.proj.weight";
const VISION_PATCH_BIAS: &str = "model.visual.patch_embed.proj.bias";
const VISION_POSITION_WEIGHT: &str = "model.visual.pos_embed.weight";
const VISION_HIDDEN_SIZE: usize = 1_024;
const VISION_INTERMEDIATE_SIZE: usize = 4_096;
const VISION_OUTPUT_SIZE: usize = 2_560;
const VISION_HEAD_COUNT: usize = 16;
const VISION_BLOCK_COUNT: usize = 24;
const MAX_Q4_MODEL_ELEMENTS: usize = 700_000_000;
const MAX_MULTIMODAL_IMAGES: usize = 8;
const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct Qwen35CandidateLoadOptions {
    pub q4_group_size: usize,
    pub maximum_context: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen35LayerMixerTensorNames {
    Linear(Qwen35LinearAttentionTensorNames),
    Full(Qwen35FullAttentionTensorNames),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35DecoderLayerTensorNames {
    pub input_norm_weight: String,
    pub post_attention_norm_weight: String,
    pub mlp_gate_projection: String,
    pub mlp_up_projection: String,
    pub mlp_down_projection: String,
    pub mixer: Qwen35LayerMixerTensorNames,
}

/// A parsed index whose bytes were authenticated by one specific signed Sage
/// package manifest. Its private fields prevent callers from constructing an
/// index trust marker from arbitrary JSON.
pub struct VerifiedQwen35WeightIndex {
    manifest_sha256: String,
    index: Qwen35WeightIndex,
}

/// A pinned, parsed model configuration whose exact source bytes were covered
/// by the same signed package as the weights.
pub struct VerifiedQwen35Config {
    manifest_sha256: String,
    config: Qwen35Config,
}

/// Sage's own tokenizer parsed from the small artifact covered by the package.
pub struct VerifiedQwen35Tokenizer {
    manifest_sha256: String,
    tokenizer: Qwen35Tokenizer,
}

/// A parsed image processor bound to the exact signed package used by weights.
pub struct VerifiedQwen35ImageProcessor {
    manifest_sha256: String,
    processor: Qwen35VisionProcessor,
}

/// A signed-package-bound text weight source. It validates both complete shard
/// headers before exposing any matrix import operation.
pub struct VerifiedQwen35TextWeights<R1, R2> {
    manifest_sha256: String,
    index: VerifiedQwen35WeightIndex,
    first: SafeTensorReader<R1>,
    second: SafeTensorReader<R2>,
    use_metal_backend: bool,
}

/// Resident, unadmitted text candidate for numerical and task evaluation. The
/// resource reservation is retained for its lifetime; it is cooperative
/// admission, not an OS memory limit. Product generation stays unavailable.
pub struct Qwen35CandidateModel {
    decoder: Qwen35TextDecoder,
    tokenizer: Qwen35Tokenizer,
    image_processor: Qwen35VisionProcessor,
    vision_encoder: Qwen35VisionEncoder,
    image_token_id: u32,
    _reservation: Reservation,
}

impl Qwen35CandidateModel {
    /// Prepare an RGB image using processor metadata bound to this signed package.
    pub fn prepare_vision_image(
        &self,
        image: Qwen35RgbImage<'_>,
        maximum_vision_tokens: usize,
    ) -> Qwen35Result<PreparedQwen35Image> {
        self.image_processor
            .prepare_rgb(image, maximum_vision_tokens)
    }

    /// Encode a prepared image with the package-bound vision encoder.
    pub fn encode_prepared_vision_image(
        &self,
        image: &PreparedQwen35Image,
    ) -> Qwen35Result<Qwen35VisionEncoding> {
        self.vision_encoder.encode_prepared_image(image)
    }

    /// Encode an image through this candidate's verified processor and encoder.
    pub fn encode_vision_image(
        &self,
        image: Qwen35RgbImage<'_>,
        maximum_vision_tokens: usize,
    ) -> Qwen35Result<Qwen35VisionEncoding> {
        let prepared = self.prepare_vision_image(image, maximum_vision_tokens)?;
        self.encode_prepared_vision_image(&prepared)
    }

    /// Read-only tokenization for the Core planning adapter or isolated worker.
    pub fn tokenizer(&self) -> &Qwen35Tokenizer {
        &self.tokenizer
    }

    /// Mutable decoder access is limited to the single inference owner.
    pub fn decoder_mut(&mut self) -> &mut Qwen35TextDecoder {
        &mut self.decoder
    }

    /// Borrow the two generation-only components without exposing model or
    /// resource-admission internals to the integration layer.
    pub fn decoder_and_tokenizer_mut(&mut self) -> (&mut Qwen35TextDecoder, &Qwen35Tokenizer) {
        (&mut self.decoder, &self.tokenizer)
    }

    pub fn image_token_id(&self) -> u32 {
        self.image_token_id
    }
}

pub fn multimodal_prompt_positions(
    token_count: usize,
    image_spans: &[EncodedImageSpan],
    merged_grids: &[(usize, usize)],
) -> Qwen35Result<Vec<[u64; 3]>> {
    if token_count == 0
        || token_count > SAGE_CONTEXT_LIMIT as usize
        || image_spans.len() != merged_grids.len()
        || image_spans.len() > MAX_MULTIMODAL_IMAGES
    {
        return Err(Qwen35Error::Model(
            "Qwen multimodal prompt geometry is outside its bound".into(),
        ));
    }
    let mut positions = vec![[0_u64; 3]; token_count];
    let mut text_position = 0_u64;
    let mut token_index = 0usize;
    for (span, (rows, columns)) in image_spans.iter().zip(merged_grids) {
        let expected_pads = rows.checked_mul(*columns);
        if *rows == 0
            || *columns == 0
            || span.vision_start < token_index
            || span.token_start != span.vision_start + 1
            || expected_pads != Some(span.token_count)
            || span.token_start.checked_add(span.token_count) != Some(span.vision_end)
            || span.vision_end >= token_count
        {
            return Err(Qwen35Error::Model(
                "Qwen image token span does not match its merged grid".into(),
            ));
        }
        while token_index < span.vision_start {
            positions[token_index] = [text_position; 3];
            text_position = next_mrope_position(text_position)?;
            token_index += 1;
        }

        positions[span.vision_start] = [text_position; 3];
        text_position = next_mrope_position(text_position)?;
        let image_position_offset = text_position;
        for row in 0..*rows {
            for column in 0..*columns {
                let relative = row * columns + column;
                let index = span.token_start + relative;
                positions[index] = [
                    image_position_offset,
                    image_position_offset + row as u64,
                    image_position_offset + column as u64,
                ];
            }
        }
        text_position = text_position
            .checked_add((*rows).max(*columns) as u64)
            .ok_or_else(|| Qwen35Error::Model("Qwen image RoPE position overflow".into()))?;
        positions[span.vision_end] = [text_position; 3];
        text_position = next_mrope_position(text_position)?;
        token_index = span.vision_end + 1;
    }
    while token_index < token_count {
        positions[token_index] = [text_position; 3];
        text_position = next_mrope_position(text_position)?;
        token_index += 1;
    }
    if positions
        .iter()
        .flatten()
        .any(|position| *position >= QWEN35_4B_MAX_POSITION_EMBEDDINGS)
    {
        return Err(Qwen35Error::Model(
            "Qwen multimodal prompt exceeds the checkpoint RoPE range".into(),
        ));
    }
    Ok(positions)
}

fn next_mrope_position(position: u64) -> Qwen35Result<u64> {
    position
        .checked_add(1)
        .filter(|next| *next < QWEN35_4B_MAX_POSITION_EMBEDDINGS)
        .ok_or_else(|| Qwen35Error::Model("Qwen multimodal position limit exceeded".into()))
}

pub trait Qwen35PackageIndexExt {
    fn parse_verified(
        package: &VerifiedQwen35Package,
        artifact: VerifiedPackageArtifact<Cursor<Vec<u8>>>,
    ) -> Qwen35Result<VerifiedQwen35WeightIndex>;
}

impl Qwen35PackageIndexExt for Qwen35WeightIndex {
    /// Parse only index bytes read and retained from the exact artifact covered
    /// by `package`; arbitrary unverified JSON cannot create this marker.
    fn parse_verified(
        package: &VerifiedQwen35Package,
        artifact: VerifiedPackageArtifact<Cursor<Vec<u8>>>,
    ) -> Qwen35Result<VerifiedQwen35WeightIndex> {
        package.require_owned_receipt(INDEX_NAME, &artifact)?;
        let manifest_sha256 = package.manifest_sha256().to_owned();
        let bytes = artifact.into_verified_bytes();
        Ok(VerifiedQwen35WeightIndex {
            manifest_sha256,
            index: Self::parse(&bytes)?,
        })
    }
}

impl VerifiedQwen35Config {
    pub fn parse(
        package: &VerifiedQwen35Package,
        artifact: VerifiedPackageArtifact<Cursor<Vec<u8>>>,
    ) -> Qwen35Result<Self> {
        package.require_owned_receipt(CONFIG_NAME, &artifact)?;
        let manifest_sha256 = package.manifest_sha256().to_owned();
        let bytes = artifact.into_verified_bytes();
        Ok(Self {
            manifest_sha256,
            config: Qwen35Config::parse(&bytes)?,
        })
    }
}

impl VerifiedQwen35Tokenizer {
    pub fn parse(
        package: &VerifiedQwen35Package,
        artifact: VerifiedPackageArtifact<Cursor<Vec<u8>>>,
    ) -> Qwen35Result<Self> {
        package.require_owned_receipt(TOKENIZER_NAME, &artifact)?;
        let manifest_sha256 = package.manifest_sha256().to_owned();
        let bytes = artifact.into_verified_bytes();
        Ok(Self {
            manifest_sha256,
            tokenizer: Qwen35Tokenizer::from_json(&bytes)?,
        })
    }
}

impl VerifiedQwen35ImageProcessor {
    pub fn parse(
        package: &VerifiedQwen35Package,
        artifact: VerifiedPackageArtifact<Cursor<Vec<u8>>>,
    ) -> Qwen35Result<Self> {
        let processor = Qwen35VisionProcessor::parse_verified(package, artifact)?;
        Ok(Self {
            manifest_sha256: processor.manifest_sha256().to_owned(),
            processor,
        })
    }
}

impl VerifiedQwen35WeightIndex {
    pub fn total_size(&self) -> u64 {
        self.index.total_size()
    }

    /// Estimate steady-state text weights, recurrent/KV state, decode
    /// activations and runtime overhead before allocating the model.
    pub fn memory_envelope(
        &self,
        q4_group_size: usize,
        maximum_context: usize,
    ) -> Qwen35Result<MemoryEnvelope> {
        if !(16..=4096).contains(&q4_group_size)
            || maximum_context == 0
            || maximum_context > SAGE_CONTEXT_LIMIT as usize
        {
            return Err(Qwen35Error::Model(
                "Qwen memory estimate exceeds Sage's quantization or context bounds".into(),
            ));
        }
        let mut names = text_weight_names(&self.index)?;
        names.extend(vision_weight_names(&self.index)?);
        let weights = names.iter().try_fold(0u64, |total, name| {
            let spec = self.index.tensor_spec(name).ok_or_else(|| {
                Qwen35Error::Model("Qwen text weight is absent from the pinned index".into())
            })?;
            let elements = tensor_elements(spec)?;
            let is_q4_matrix = spec.shape().len() == 2
                && name != VISION_PATCH_WEIGHT
                && name != VISION_POSITION_WEIGHT;
            let bytes = if is_q4_matrix {
                let packed = elements.div_ceil(2);
                let groups = elements.div_ceil(q4_group_size as u64);
                packed
                    .checked_add(
                        groups
                            .checked_mul(std::mem::size_of::<f32>() as u64)
                            .ok_or_else(|| {
                                Qwen35Error::Model("Qwen quantized scale estimate overflow".into())
                            })?,
                    )
                    .ok_or_else(|| {
                        Qwen35Error::Model("Qwen quantized weight estimate overflow".into())
                    })?
            } else {
                elements
                    .checked_mul(std::mem::size_of::<f32>() as u64)
                    .ok_or_else(|| {
                        Qwen35Error::Model("Qwen vector weight estimate overflow".into())
                    })?
            };
            total
                .checked_add(bytes)
                .ok_or_else(|| Qwen35Error::Model("Qwen text weight estimate overflow".into()))
        })?;
        let q4_projection_workspace_bytes = q4_metal_workspace_bytes(&self.index, &names)?;

        // Eight full-attention layers retain K and V for four KV heads at 256
        // values per head in binary16. Twenty-four linear layers retain one
        // 32x128x128 f32 recurrent matrix plus three convolution history taps.
        let full_attention_state_elements = 8_u64
            .checked_mul(2)
            .and_then(|value| value.checked_mul(maximum_context as u64))
            .and_then(|value| value.checked_mul(4 * 256))
            .ok_or_else(|| Qwen35Error::Model("Qwen KV estimate overflow".into()))?;
        let linear_recurrent_elements = 24_u64 * 32 * 128 * 128;
        let linear_convolution_channels =
            2_u64 * QWEN35_4B_KEY_HEADS as u64 * QWEN35_4B_HEAD_DIMENSION as u64
                + QWEN35_4B_VALUE_HEADS as u64 * QWEN35_4B_HEAD_DIMENSION as u64;
        let linear_convolution_elements = 24 * linear_convolution_channels * 3;
        // Conservatively account for the generic four-query tile and 2K score
        // block. The measured Apple profile uses a 1-query, 4K block whose
        // score product is half this envelope; block-output scratch is also
        // no larger than the generic profile.
        let attention_scratch_bytes = 8_u64
            .checked_mul(
                (maximum_context.min(sage_inference_math::ATTENTION_SCORE_BLOCK_SIZE) as u64)
                    .checked_mul(4)
                    .and_then(|positions| positions.checked_mul(std::mem::size_of::<f64>() as u64))
                    .and_then(|scores| {
                        scores.checked_add((16 * 256 + 4 * 256) * std::mem::size_of::<f32>() as u64)
                    })
                    .ok_or_else(|| {
                        Qwen35Error::Model("Qwen attention scratch estimate overflow".into())
                    })?,
            )
            .ok_or_else(|| Qwen35Error::Model("Qwen attention scratch estimate overflow".into()))?;
        // Each full-attention layer retains zeroizing Q/K/V projections and
        // normalized Q/K plus gate buffers so decode does not allocate those
        // vectors for every token.
        let projection_elements_per_layer = (QUERY_PROJECTION_SIZE as u64)
            .checked_add((3 * KEY_VALUE_PROJECTION_SIZE) as u64)
            .and_then(|elements| elements.checked_add((2 * ATTENTION_OUTPUT_SIZE) as u64))
            .ok_or_else(|| {
                Qwen35Error::Model(
                    "Qwen full-attention projection scratch estimate overflow".into(),
                )
            })?;
        let full_attention_projection_scratch_bytes = 8_u64
            .checked_mul(projection_elements_per_layer)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| {
                Qwen35Error::Model(
                    "Qwen full-attention projection scratch estimate overflow".into(),
                )
            })?;
        let linear_state_elements = linear_recurrent_elements
            .checked_add(linear_convolution_elements)
            .ok_or_else(|| Qwen35Error::Model("Qwen state estimate overflow".into()))?;
        let full_attention_cache_bytes = full_attention_state_elements
            .checked_mul(std::mem::size_of::<u16>() as u64)
            .ok_or_else(|| Qwen35Error::Model("Qwen KV byte estimate overflow".into()))?;
        let linear_state_bytes = linear_state_elements
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| {
                Qwen35Error::Model("Qwen recurrent-state byte estimate overflow".into())
            })?;
        let state = full_attention_cache_bytes
            .checked_add(linear_state_bytes)
            .and_then(|bytes| bytes.checked_add(attention_scratch_bytes))
            .and_then(|bytes| bytes.checked_add(full_attention_projection_scratch_bytes))
            .ok_or_else(|| Qwen35Error::Model("Qwen state byte estimate overflow".into()))?;
        // Each decoder layer retains gate/up MLP buffers, its feed-forward
        // output, and one RMS buffer reused before and after attention.
        let decoder_layer_scratch = 32_u64
            .checked_mul(
                u64::try_from(
                    INTERMEDIATE_SIZE
                        .checked_mul(2)
                        .and_then(|elements| {
                            HIDDEN_SIZE
                                .checked_mul(2)
                                .and_then(|layer_buffers| elements.checked_add(layer_buffers))
                        })
                        .ok_or_else(|| {
                            Qwen35Error::Model("Qwen decoder scratch estimate overflow".into())
                        })?,
                )
                .map_err(|_| Qwen35Error::Model("Qwen decoder scratch estimate overflow".into()))?
                .checked_mul(std::mem::size_of::<f32>() as u64)
                .ok_or_else(|| {
                    Qwen35Error::Model("Qwen decoder scratch estimate overflow".into())
                })?,
            )
            .ok_or_else(|| Qwen35Error::Model("Qwen decoder scratch estimate overflow".into()))?;
        // Each of the 24 linear-attention layers retains projection outputs,
        // convolution/head-expansion workspaces, recurrent output, and gated
        // normalization output so decoding does not allocate per token.
        let linear_attention_scratch_elements = (DELTA_QKV_SIZE as u64)
            .checked_mul(2)
            .and_then(|elements| elements.checked_add(5 * DELTA_VALUE_SIZE as u64))
            .and_then(|elements| elements.checked_add(4 * QWEN35_4B_VALUE_HEADS as u64))
            .and_then(|elements| elements.checked_mul(24))
            .ok_or_else(|| {
                Qwen35Error::Model("Qwen linear-attention scratch estimate overflow".into())
            })?;
        let linear_attention_scratch = linear_attention_scratch_elements
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| {
                Qwen35Error::Model("Qwen linear-attention scratch estimate overflow".into())
            })?;
        // The text decoder owns two hidden-width zeroizing buffers for token
        // lookup, layer ping-pong, and final normalization.
        let decoder_hidden_scratch = (2 * HIDDEN_SIZE as u64)
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| Qwen35Error::Model("Qwen hidden scratch estimate overflow".into()))?;
        let retained_decoder_scratch = decoder_layer_scratch
            .checked_add(linear_attention_scratch)
            .and_then(|bytes| bytes.checked_add(decoder_hidden_scratch))
            .ok_or_else(|| Qwen35Error::Model("Qwen decoder scratch estimate overflow".into()))?;
        let envelope = MemoryEnvelope {
            weights,
            state,
            // Includes text decoding and a maximum-size vision pass: prepared
            // RGB patches, 24-block activations, merger scratch and output.
            activations: 1_024 * MIB,
            runtime: (512 * MIB)
                .checked_add(retained_decoder_scratch)
                .and_then(|bytes| bytes.checked_add(q4_projection_workspace_bytes))
                .ok_or_else(|| Qwen35Error::Model("Qwen runtime estimate overflow".into()))?,
        };
        envelope.total()?;
        Ok(envelope)
    }
}

trait Qwen35LayerIndexExt {
    fn decoder_layer_tensor_names(
        &self,
        layer_index: usize,
    ) -> Qwen35Result<Qwen35DecoderLayerTensorNames>;
}

impl Qwen35LayerIndexExt for Qwen35WeightIndex {
    fn decoder_layer_tensor_names(
        &self,
        layer_index: usize,
    ) -> Qwen35Result<Qwen35DecoderLayerTensorNames> {
        if layer_index >= 32 {
            return Err(Qwen35Error::Model(
                "Qwen decoder layer is outside the pinned 4B profile".into(),
            ));
        }
        let prefix = format!("model.language_model.layers.{layer_index}.");
        let names = Qwen35DecoderLayerTensorNames {
            input_norm_weight: format!("{prefix}input_layernorm.weight"),
            post_attention_norm_weight: format!("{prefix}post_attention_layernorm.weight"),
            mlp_gate_projection: format!("{prefix}mlp.gate_proj.weight"),
            mlp_up_projection: format!("{prefix}mlp.up_proj.weight"),
            mlp_down_projection: format!("{prefix}mlp.down_proj.weight"),
            mixer: if layer_index % 4 == 3 {
                Qwen35LayerMixerTensorNames::Full(self.full_attention_tensor_names(layer_index)?)
            } else {
                Qwen35LayerMixerTensorNames::Linear(
                    self.linear_attention_tensor_names(layer_index)?,
                )
            },
        };
        use sage_model_package::safetensors::TensorDType::BF16;
        require_spec(self, &names.input_norm_weight, BF16, &[HIDDEN_SIZE])?;
        require_spec(
            self,
            &names.post_attention_norm_weight,
            BF16,
            &[HIDDEN_SIZE],
        )?;
        require_spec(
            self,
            &names.mlp_gate_projection,
            BF16,
            &[INTERMEDIATE_SIZE, HIDDEN_SIZE],
        )?;
        require_spec(
            self,
            &names.mlp_up_projection,
            BF16,
            &[INTERMEDIATE_SIZE, HIDDEN_SIZE],
        )?;
        require_spec(
            self,
            &names.mlp_down_projection,
            BF16,
            &[HIDDEN_SIZE, INTERMEDIATE_SIZE],
        )?;
        Ok(names)
    }
}

impl<R1: Read + Seek, R2: Read + Seek> VerifiedQwen35TextWeights<R1, R2> {
    pub fn open(
        package: &VerifiedQwen35Package,
        index: VerifiedQwen35WeightIndex,
        first_artifact: VerifiedPackageArtifact<R1>,
        second_artifact: VerifiedPackageArtifact<R2>,
    ) -> Qwen35Result<Self> {
        if index.manifest_sha256 != package.manifest_sha256() {
            return Err(Qwen35Error::Model(
                "Qwen tensor index and shards came from different signed packages".into(),
            ));
        }
        package.require_owned_receipt(SHARD_ONE, &first_artifact)?;
        package.require_owned_receipt(SHARD_TWO, &second_artifact)?;
        let first = SafeTensorReader::new(first_artifact.into_verified_reader())?;
        let second = SafeTensorReader::new(second_artifact.into_verified_reader())?;
        index.index.validate_shard(SHARD_ONE, &first)?;
        index.index.validate_shard(SHARD_TWO, &second)?;
        Ok(Self {
            manifest_sha256: package.manifest_sha256().to_owned(),
            index,
            first,
            second,
            use_metal_backend: false,
        })
    }

    /// Opt this evaluation candidate into Sage's Metal grouped-Q4 projection
    /// kernel. Package verification and model admission remain separate.
    pub fn with_metal_backend(mut self) -> Qwen35Result<Self> {
        sage_metal::MetalContext::shared().map_err(|error| {
            Qwen35Error::Model(format!("Sage Metal backend is unavailable: {error}"))
        })?;
        self.use_metal_backend = true;
        Ok(self)
    }

    /// Import every text tensor into Sage's Q4/f32 structures, then hash both
    /// original open shard handles again before returning a usable decoder.
    /// A modified shard discards the partially built model.
    pub fn load_candidate_model(
        mut self,
        package: &VerifiedQwen35Package,
        config: VerifiedQwen35Config,
        tokenizer: VerifiedQwen35Tokenizer,
        image_processor: VerifiedQwen35ImageProcessor,
        governor: &ResourceGovernor,
        options: Qwen35CandidateLoadOptions,
    ) -> Qwen35Result<Qwen35CandidateModel> {
        if self.manifest_sha256 != package.manifest_sha256()
            || config.manifest_sha256 != self.manifest_sha256
            || tokenizer.manifest_sha256 != self.manifest_sha256
            || image_processor.manifest_sha256 != self.manifest_sha256
            || options.maximum_context == 0
            || options.maximum_context > config.config.admitted_context() as usize
        {
            return Err(Qwen35Error::Model(
                "Qwen text assets do not share one signed package or admitted context".into(),
            ));
        }
        let expected_multimodal_tokens = config.config.multimodal_token_ids();
        let observed_multimodal_tokens = (
            tokenizer.tokenizer.token_id("<|vision_start|>"),
            tokenizer.tokenizer.token_id("<|image_pad|>"),
            tokenizer.tokenizer.token_id("<|vision_end|>"),
        );
        if observed_multimodal_tokens
            != (
                Some(expected_multimodal_tokens.0),
                Some(expected_multimodal_tokens.1),
                Some(expected_multimodal_tokens.2),
            )
            || [
                expected_multimodal_tokens.0,
                expected_multimodal_tokens.1,
                expected_multimodal_tokens.2,
            ]
            .into_iter()
            .any(|token| tokenizer.tokenizer.text_token_bytes(token).is_some())
        {
            return Err(Qwen35Error::Model(
                "Qwen vision framing tokens do not match the signed model config".into(),
            ));
        }
        let envelope = self
            .index
            .memory_envelope(options.q4_group_size, options.maximum_context)?;
        let reservation = governor.reserve_current(envelope.total()?, true)?;

        let embeddings = Qwen35ProjectionMatrix::from(self.load_q4_projection(
            EMBEDDING_NAME,
            &[VOCABULARY_SIZE, HIDDEN_SIZE],
            options.q4_group_size,
        )?);
        let final_norm = self.load_f32_vector(FINAL_NORM_NAME, &[HIDDEN_SIZE])?;
        let mut layers = Vec::with_capacity(32);
        for layer_index in 0..32 {
            layers.push(self.load_decoder_layer(
                layer_index,
                config.config.rms_norm_epsilon(),
                options.maximum_context,
                options.q4_group_size,
            )?);
        }
        let decoder = Qwen35TextDecoder::for_qwen35_4b(
            embeddings,
            final_norm,
            layers,
            config.config.rms_norm_epsilon(),
            options.maximum_context,
        )?;
        let vision_encoder = self.load_vision_encoder(options.q4_group_size)?;
        package.verify_artifact(SHARD_ONE, self.first.source_mut())?;
        package.verify_artifact(SHARD_TWO, self.second.source_mut())?;

        Ok(Qwen35CandidateModel {
            decoder,
            tokenizer: tokenizer.tokenizer,
            image_processor: image_processor.processor,
            vision_encoder,
            image_token_id: expected_multimodal_tokens.1,
            _reservation: reservation,
        })
    }

    fn load_vision_encoder(&mut self, q4_group_size: usize) -> Qwen35Result<Qwen35VisionEncoder> {
        let patch_weight_values =
            self.load_f32_vector(VISION_PATCH_WEIGHT, &[VISION_HIDDEN_SIZE, 3, 2, 16, 16])?;
        let patch_weight_matrix =
            CpuMatrix::new(VISION_HIDDEN_SIZE, 3 * 2 * 16 * 16, patch_weight_values)?;
        let patch_bias = self.load_f32_vector(VISION_PATCH_BIAS, &[VISION_HIDDEN_SIZE])?;
        let patch_embed = crate::qwen35_vision::Qwen35PatchEmbed::new(
            Qwen35ProjectionMatrix::from(patch_weight_matrix),
            patch_bias,
        )?;

        let position_values =
            self.load_f32_vector(VISION_POSITION_WEIGHT, &[2304, VISION_HIDDEN_SIZE])?;
        let position_table = CpuMatrix::new(2304, VISION_HIDDEN_SIZE, position_values)?;
        let position_embedding = Qwen35VisionPositionEmbedding::new(position_table)?;

        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(VISION_BLOCK_COUNT)
            .map_err(|_| Qwen35Error::Model("Qwen vision block allocation was denied".into()))?;
        for block in 0..VISION_BLOCK_COUNT {
            let prefix = format!("model.visual.blocks.{block}.");
            let weights = Qwen35VisionBlockWeights {
                norm1_weight: self
                    .load_f32_vector(&format!("{prefix}norm1.weight"), &[VISION_HIDDEN_SIZE])?,
                norm1_bias: self
                    .load_f32_vector(&format!("{prefix}norm1.bias"), &[VISION_HIDDEN_SIZE])?,
                qkv_projection: self.q4_projection(
                    &format!("{prefix}attn.qkv.weight"),
                    &[VISION_HIDDEN_SIZE * 3, VISION_HIDDEN_SIZE],
                    q4_group_size,
                )?,
                qkv_bias: self.load_f32_vector(
                    &format!("{prefix}attn.qkv.bias"),
                    &[VISION_HIDDEN_SIZE * 3],
                )?,
                attention_projection: self.q4_projection(
                    &format!("{prefix}attn.proj.weight"),
                    &[VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE],
                    q4_group_size,
                )?,
                attention_bias: self
                    .load_f32_vector(&format!("{prefix}attn.proj.bias"), &[VISION_HIDDEN_SIZE])?,
                norm2_weight: self
                    .load_f32_vector(&format!("{prefix}norm2.weight"), &[VISION_HIDDEN_SIZE])?,
                norm2_bias: self
                    .load_f32_vector(&format!("{prefix}norm2.bias"), &[VISION_HIDDEN_SIZE])?,
                mlp_input_projection: self.q4_projection(
                    &format!("{prefix}mlp.linear_fc1.weight"),
                    &[VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE],
                    q4_group_size,
                )?,
                mlp_input_bias: self.load_f32_vector(
                    &format!("{prefix}mlp.linear_fc1.bias"),
                    &[VISION_INTERMEDIATE_SIZE],
                )?,
                mlp_output_projection: self.q4_projection(
                    &format!("{prefix}mlp.linear_fc2.weight"),
                    &[VISION_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE],
                    q4_group_size,
                )?,
                mlp_output_bias: self.load_f32_vector(
                    &format!("{prefix}mlp.linear_fc2.bias"),
                    &[VISION_HIDDEN_SIZE],
                )?,
            };
            blocks.push(Qwen35VisionBlock::new(VISION_HEAD_COUNT, weights)?);
        }
        let block_stack = Qwen35VisionBlockStack::new(blocks)?;

        let merger = Qwen35VisionPatchMerger::new(
            VISION_HIDDEN_SIZE,
            VISION_OUTPUT_SIZE,
            Qwen35VisionPatchMergerWeights {
                norm_weight: self
                    .load_f32_vector("model.visual.merger.norm.weight", &[VISION_HIDDEN_SIZE])?,
                norm_bias: self
                    .load_f32_vector("model.visual.merger.norm.bias", &[VISION_HIDDEN_SIZE])?,
                input_projection: self.q4_projection(
                    "model.visual.merger.linear_fc1.weight",
                    &[VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE * 4],
                    q4_group_size,
                )?,
                input_bias: self.load_f32_vector(
                    "model.visual.merger.linear_fc1.bias",
                    &[VISION_INTERMEDIATE_SIZE],
                )?,
                output_projection: self.q4_projection(
                    "model.visual.merger.linear_fc2.weight",
                    &[VISION_OUTPUT_SIZE, VISION_INTERMEDIATE_SIZE],
                    q4_group_size,
                )?,
                output_bias: self.load_f32_vector(
                    "model.visual.merger.linear_fc2.bias",
                    &[VISION_OUTPUT_SIZE],
                )?,
            },
        )?;

        Qwen35VisionEncoder::new(patch_embed, position_embedding, block_stack, merger)
    }

    fn load_decoder_layer(
        &mut self,
        layer_index: usize,
        norm_epsilon: f32,
        maximum_context: usize,
        q4_group_size: usize,
    ) -> Qwen35Result<Qwen35DecoderLayer> {
        let names = self.index.index.decoder_layer_tensor_names(layer_index)?;
        let input_norm = self.load_f32_vector(&names.input_norm_weight, &[HIDDEN_SIZE])?;
        let post_attention_norm =
            self.load_f32_vector(&names.post_attention_norm_weight, &[HIDDEN_SIZE])?;
        let mlp = Qwen35Mlp::new(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            Qwen35ProjectionMatrix::from(self.load_q4_projection(
                &names.mlp_gate_projection,
                &[INTERMEDIATE_SIZE, HIDDEN_SIZE],
                q4_group_size,
            )?),
            Qwen35ProjectionMatrix::from(self.load_q4_projection(
                &names.mlp_up_projection,
                &[INTERMEDIATE_SIZE, HIDDEN_SIZE],
                q4_group_size,
            )?),
            Qwen35ProjectionMatrix::from(self.load_q4_projection(
                &names.mlp_down_projection,
                &[HIDDEN_SIZE, INTERMEDIATE_SIZE],
                q4_group_size,
            )?),
        )?;
        let mixer = match names.mixer {
            Qwen35LayerMixerTensorNames::Linear(tensors) => {
                Qwen35TokenMixer::Linear(Qwen35LinearAttentionBlock::for_qwen35_4b(
                    norm_epsilon,
                    Qwen35LinearAttentionWeights {
                        qkv_projection: self.q4_projection(
                            &tensors.qkv_projection,
                            &[DELTA_QKV_SIZE, HIDDEN_SIZE],
                            q4_group_size,
                        )?,
                        gate_projection: self.q4_projection(
                            &tensors.gate_projection,
                            &[DELTA_VALUE_SIZE, HIDDEN_SIZE],
                            q4_group_size,
                        )?,
                        decay_projection: self.q4_projection(
                            &tensors.decay_projection,
                            &[QWEN35_4B_VALUE_HEADS, HIDDEN_SIZE],
                            q4_group_size,
                        )?,
                        beta_projection: self.q4_projection(
                            &tensors.beta_projection,
                            &[QWEN35_4B_VALUE_HEADS, HIDDEN_SIZE],
                            q4_group_size,
                        )?,
                        convolution: self.load_f32_vector(
                            &tensors.convolution,
                            &[DELTA_QKV_SIZE, 1, QWEN35_4B_CONV_KERNEL_SIZE],
                        )?,
                        a_log: self.load_f32_vector(&tensors.a_log, &[QWEN35_4B_VALUE_HEADS])?,
                        dt_bias: self
                            .load_f32_vector(&tensors.dt_bias, &[QWEN35_4B_VALUE_HEADS])?,
                        norm_weight: self
                            .load_f32_vector(&tensors.norm_weight, &[QWEN35_4B_HEAD_DIMENSION])?,
                        output_projection: self.q4_projection(
                            &tensors.output_projection,
                            &[HIDDEN_SIZE, DELTA_VALUE_SIZE],
                            q4_group_size,
                        )?,
                    },
                )?)
            }
            Qwen35LayerMixerTensorNames::Full(tensors) => {
                Qwen35TokenMixer::Full(Qwen35FullAttentionBlock::for_qwen35_4b(
                    norm_epsilon,
                    maximum_context,
                    Qwen35FullAttentionWeights {
                        query_gate_projection: self.q4_projection(
                            &tensors.query_gate_projection,
                            &[QUERY_PROJECTION_SIZE, HIDDEN_SIZE],
                            q4_group_size,
                        )?,
                        key_projection: self.q4_projection(
                            &tensors.key_projection,
                            &[KEY_VALUE_PROJECTION_SIZE, HIDDEN_SIZE],
                            q4_group_size,
                        )?,
                        value_projection: self.q4_projection(
                            &tensors.value_projection,
                            &[KEY_VALUE_PROJECTION_SIZE, HIDDEN_SIZE],
                            q4_group_size,
                        )?,
                        output_projection: self.q4_projection(
                            &tensors.output_projection,
                            &[HIDDEN_SIZE, ATTENTION_OUTPUT_SIZE],
                            q4_group_size,
                        )?,
                        query_norm_weight: self
                            .load_f32_vector(&tensors.query_norm_weight, &[256])?,
                        key_norm_weight: self.load_f32_vector(&tensors.key_norm_weight, &[256])?,
                    },
                )?)
            }
        };
        Qwen35DecoderLayer::new(
            HIDDEN_SIZE,
            norm_epsilon,
            input_norm,
            post_attention_norm,
            mixer,
            mlp,
        )
    }

    fn q4_projection(
        &mut self,
        name: &str,
        expected_shape: &[usize],
        group_size: usize,
    ) -> Qwen35Result<Qwen35ProjectionMatrix> {
        Ok(Qwen35ProjectionMatrix::from(self.load_q4_projection(
            name,
            expected_shape,
            group_size,
        )?))
    }

    fn load_q4_projection(
        &mut self,
        name: &str,
        expected_shape: &[usize],
        group_size: usize,
    ) -> Qwen35Result<sage_inference_math::QuantizedQ4Matrix> {
        let shard = self.index.index.shard_for(name).ok_or_else(|| {
            Qwen35Error::Model("Qwen projection is missing from the signed tensor index".into())
        })?;
        let spec = self.index.index.tensor_spec(name).ok_or_else(|| {
            Qwen35Error::Model("Qwen projection has no signed tensor specification".into())
        })?;
        if spec.shape() != expected_shape || expected_shape.len() != 2 {
            return Err(Qwen35Error::Model(
                "Qwen projection geometry differs from the decoder contract".into(),
            ));
        }
        let index = &self.index.index;
        let mut matrix = match shard {
            SHARD_ONE => {
                let stream = tensor_stream(index, SHARD_ONE, &mut self.first, name)?;
                stream.into_q4_matrix(group_size, MAX_Q4_MODEL_ELEMENTS)
            }
            SHARD_TWO => {
                let stream = tensor_stream(index, SHARD_TWO, &mut self.second, name)?;
                stream.into_q4_matrix(group_size, MAX_Q4_MODEL_ELEMENTS)
            }
            _ => Err(Qwen35Error::Model(
                "Qwen projection resolves outside the two signed shards".into(),
            )),
        }?;
        if should_place_q4_matrix_on_metal(name, self.use_metal_backend) {
            matrix.move_to_metal()?;
        }
        Ok(matrix)
    }

    fn load_f32_vector(&mut self, name: &str, expected_shape: &[usize]) -> Qwen35Result<Vec<f32>> {
        let shard = self.index.index.shard_for(name).ok_or_else(|| {
            Qwen35Error::Model("Qwen vector is missing from the signed tensor index".into())
        })?;
        let spec = self.index.index.tensor_spec(name).ok_or_else(|| {
            Qwen35Error::Model("Qwen vector has no signed tensor specification".into())
        })?;
        if spec.shape() != expected_shape {
            return Err(Qwen35Error::Model(
                "Qwen vector geometry differs from the decoder contract".into(),
            ));
        }
        let maximum_elements = expected_shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| Qwen35Error::Model("Qwen vector size overflow".into()))?;
        let index = &self.index.index;
        match shard {
            SHARD_ONE => tensor_stream(index, SHARD_ONE, &mut self.first, name)?
                .into_f32_vector(maximum_elements),
            SHARD_TWO => tensor_stream(index, SHARD_TWO, &mut self.second, name)?
                .into_f32_vector(maximum_elements),
            _ => Err(Qwen35Error::Model(
                "Qwen vector resolves outside the two signed shards".into(),
            )),
        }
    }
}

fn should_place_q4_matrix_on_metal(name: &str, metal_evaluation_requested: bool) -> bool {
    // The tied token embedding is also the output head. On the measured Apple
    // M3 shape, CPU fused argmax is 23.7x faster than Metal projection plus
    // host reduction, so retain this matrix in CPU-readable Q4 storage even
    // when other candidate projections use the evaluation-only Metal route.
    metal_evaluation_requested && name != EMBEDDING_NAME
}

fn tensor_stream<'a, R: Read + Seek>(
    index: &'a Qwen35WeightIndex,
    shard: &'a str,
    reader: &'a mut SafeTensorReader<R>,
    name: &str,
) -> Qwen35Result<Qwen35TensorStream<'a, R>> {
    index.stream_tensor(shard, reader, name)
}

fn require_spec(
    index: &Qwen35WeightIndex,
    name: &str,
    dtype: sage_model_package::safetensors::TensorDType,
    shape: &[usize],
) -> Qwen35Result<()> {
    if index.shard_for(name).is_none()
        || index
            .tensor_spec(name)
            .is_none_or(|spec| spec.dtype() != dtype || spec.shape() != shape)
    {
        return Err(Qwen35Error::Model(
            "Qwen decoder layer tensor map differs from the pinned profile".into(),
        ));
    }
    Ok(())
}

fn text_weight_names(index: &Qwen35WeightIndex) -> Qwen35Result<BTreeSet<String>> {
    let mut names = BTreeSet::from([EMBEDDING_NAME.to_owned(), FINAL_NORM_NAME.to_owned()]);
    for layer in 0..32 {
        let layer = index.decoder_layer_tensor_names(layer)?;
        names.extend([
            layer.input_norm_weight,
            layer.post_attention_norm_weight,
            layer.mlp_gate_projection,
            layer.mlp_up_projection,
            layer.mlp_down_projection,
        ]);
        match layer.mixer {
            Qwen35LayerMixerTensorNames::Linear(tensors) => names.extend([
                tensors.a_log,
                tensors.convolution,
                tensors.dt_bias,
                tensors.decay_projection,
                tensors.beta_projection,
                tensors.qkv_projection,
                tensors.gate_projection,
                tensors.norm_weight,
                tensors.output_projection,
            ]),
            Qwen35LayerMixerTensorNames::Full(tensors) => names.extend([
                tensors.query_gate_projection,
                tensors.key_projection,
                tensors.value_projection,
                tensors.output_projection,
                tensors.query_norm_weight,
                tensors.key_norm_weight,
            ]),
        };
    }
    for name in &names {
        if index.tensor_spec(name).is_none() {
            return Err(Qwen35Error::Model(
                "Qwen text tensor is absent from the signed profile".into(),
            ));
        }
    }
    Ok(names)
}

fn vision_weight_names(index: &Qwen35WeightIndex) -> Qwen35Result<BTreeSet<String>> {
    let mut names = BTreeSet::from([
        VISION_PATCH_WEIGHT.to_owned(),
        VISION_PATCH_BIAS.to_owned(),
        VISION_POSITION_WEIGHT.to_owned(),
        "model.visual.merger.linear_fc1.bias".to_owned(),
        "model.visual.merger.linear_fc1.weight".to_owned(),
        "model.visual.merger.linear_fc2.bias".to_owned(),
        "model.visual.merger.linear_fc2.weight".to_owned(),
        "model.visual.merger.norm.bias".to_owned(),
        "model.visual.merger.norm.weight".to_owned(),
    ]);
    for block in 0..VISION_BLOCK_COUNT {
        let prefix = format!("model.visual.blocks.{block}.");
        for suffix in [
            "attn.proj.bias",
            "attn.proj.weight",
            "attn.qkv.bias",
            "attn.qkv.weight",
            "mlp.linear_fc1.bias",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc2.bias",
            "mlp.linear_fc2.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
        ] {
            names.insert(format!("{prefix}{suffix}"));
        }
    }
    for name in &names {
        if index.tensor_spec(name).is_none() {
            return Err(Qwen35Error::Model(
                "Qwen vision tensor is absent from the signed profile".into(),
            ));
        }
    }
    Ok(names)
}

fn tensor_elements(spec: &Qwen35TensorSpec) -> Qwen35Result<u64> {
    spec.shape()
        .iter()
        .try_fold(1_u64, |count, dimension| {
            count.checked_mul(u64::try_from(*dimension).ok()?)
        })
        .filter(|count| *count > 0)
        .ok_or_else(|| Qwen35Error::Model("Qwen tensor element count overflow".into()))
}

fn q4_metal_workspace_bytes(
    index: &Qwen35WeightIndex,
    names: &BTreeSet<String>,
) -> Qwen35Result<u64> {
    names.iter().try_fold(0_u64, |total, name| {
        let spec = index.tensor_spec(name).ok_or_else(|| {
            Qwen35Error::Model("Qwen projection is absent from the signed tensor index".into())
        })?;
        let shape = spec.shape();
        if shape.len() != 2 || name == VISION_PATCH_WEIGHT || name == VISION_POSITION_WEIGHT {
            return Ok(total);
        }
        let activation_elements = shape[0]
            .checked_add(shape[1])
            .and_then(|elements| u64::try_from(elements).ok())
            .ok_or_else(|| {
                Qwen35Error::Model("Qwen Metal projection workspace estimate overflow".into())
            })?;
        let workspace_bytes = activation_elements
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| {
                Qwen35Error::Model("Qwen Metal projection workspace estimate overflow".into())
            })?;
        total.checked_add(workspace_bytes).ok_or_else(|| {
            Qwen35Error::Model("Qwen Metal projection workspace estimate overflow".into())
        })
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs::File;
    use std::io::Cursor;
    use std::path::Path;

    use ed25519_dalek::{Signer, SigningKey};
    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::*;
    use sage_model_package::{QWEN35_CHECKPOINT_REVISION, Qwen35PackageManifest};

    fn index_with_pinned_specs() -> Qwen35WeightIndex {
        crate::qwen35::core_test_index_with_pinned_specs()
    }

    #[test]
    fn metal_evaluation_keeps_the_tied_output_head_on_its_faster_cpu_route() {
        assert!(!should_place_q4_matrix_on_metal(EMBEDDING_NAME, true));
        assert!(should_place_q4_matrix_on_metal(
            "model.layers.0.q_proj.weight",
            true
        ));
        assert!(!should_place_q4_matrix_on_metal(
            "model.layers.0.q_proj.weight",
            false
        ));
    }

    #[test]
    fn decoder_layer_maps_include_all_pinned_mixer_and_mlp_weights() {
        let index = index_with_pinned_specs();
        for layer_index in 0..32 {
            let names = index.decoder_layer_tensor_names(layer_index).unwrap();
            assert!(names.input_norm_weight.ends_with("input_layernorm.weight"));
            assert!(
                names
                    .post_attention_norm_weight
                    .ends_with("post_attention_layernorm.weight")
            );
            assert!(names.mlp_gate_projection.ends_with("mlp.gate_proj.weight"));
            assert!(names.mlp_up_projection.ends_with("mlp.up_proj.weight"));
            assert!(names.mlp_down_projection.ends_with("mlp.down_proj.weight"));
            assert_eq!(
                matches!(names.mixer, Qwen35LayerMixerTensorNames::Full(_)),
                layer_index % 4 == 3
            );
        }
        assert!(index.decoder_layer_tensor_names(32).is_err());
    }

    #[test]
    fn model_memory_estimate_accounts_for_vision_weights_q4_groups_and_context_state() {
        let verified = VerifiedQwen35WeightIndex {
            manifest_sha256: "a".repeat(64),
            index: index_with_pinned_specs(),
        };
        let small = verified.memory_envelope(128, 1024).unwrap();
        let longer = verified.memory_envelope(128, 8192).unwrap();
        let larger_groups = verified.memory_envelope(256, 1024).unwrap();
        let mut weight_names = text_weight_names(&verified.index).unwrap();
        weight_names.extend(vision_weight_names(&verified.index).unwrap());
        let q4_projection_workspace_bytes =
            q4_metal_workspace_bytes(&verified.index, &weight_names).unwrap();
        let scores_at_1k = 1024.min(sage_inference_math::ATTENTION_SCORE_BLOCK_SIZE) as u64;
        let scores_at_8k = 8192.min(sage_inference_math::ATTENTION_SCORE_BLOCK_SIZE) as u64;
        let scratch_at_1k_context = 8
            * (scores_at_1k * 4 * std::mem::size_of::<f64>() as u64
                + (16 * 256 + 4 * 256) * std::mem::size_of::<f32>() as u64);
        assert!(small.weights > 1_000_000_000);
        assert_eq!(small.state, 86_245_376 + scratch_at_1k_context + 622_592);
        assert_eq!(
            longer.state - small.state,
            234_881_024 + (scores_at_8k - scores_at_1k) * 4 * 8 * std::mem::size_of::<f64>() as u64
        );
        assert!(longer.state > small.state);
        assert!(larger_groups.weights < small.weights);
        assert!(small.total().unwrap() > small.weights);
        assert!(q4_projection_workspace_bytes > 0);
        assert_eq!(
            longer.runtime - 512 * 1024 * 1024,
            6_586_368 + q4_projection_workspace_bytes
        );
        assert_eq!(
            longer.total().unwrap(),
            4_369_238_016 + q4_projection_workspace_bytes
        );
        assert_eq!(small.activations, 1_024 * MIB);
        assert!(verified.memory_envelope(8, 1024).is_err());
        assert!(verified.memory_envelope(128, 8193).is_err());
    }

    #[test]
    fn vision_tensor_map_covers_all_24_blocks_and_the_full_image_path() {
        let index = index_with_pinned_specs();
        let names = super::vision_weight_names(&index).unwrap();
        assert_eq!(names.len(), 297);
        assert!(names.contains("model.visual.patch_embed.proj.weight"));
        assert!(names.contains("model.visual.pos_embed.weight"));
        assert!(names.contains("model.visual.blocks.0.attn.qkv.weight"));
        assert!(names.contains("model.visual.blocks.23.mlp.linear_fc2.weight"));
        assert!(names.contains("model.visual.merger.linear_fc2.weight"));
    }

    #[test]
    fn multimodal_prompt_rope_positions_resume_after_each_spatial_image_grid() {
        let spans = [
            EncodedImageSpan {
                vision_start: 1,
                token_start: 2,
                token_count: 2,
                vision_end: 4,
            },
            EncodedImageSpan {
                vision_start: 7,
                token_start: 8,
                token_count: 1,
                vision_end: 9,
            },
        ];
        let positions = multimodal_prompt_positions(11, &spans, &[(1, 2), (1, 1)])
            .expect("bounded two-image position sequence");
        assert_eq!(positions[0], [0, 0, 0]);
        assert_eq!(positions[1], [1, 1, 1]);
        assert_eq!(positions[2], [2, 2, 2]);
        assert_eq!(positions[3], [2, 2, 3]);
        assert_eq!(positions[4], [4, 4, 4]);
        assert_eq!(positions[5], [5, 5, 5]);
        assert_eq!(positions[6], [6, 6, 6]);
        assert_eq!(positions[7], [7, 7, 7]);
        assert_eq!(positions[8], [8, 8, 8]);
        assert_eq!(positions[9], [9, 9, 9]);
        assert_eq!(positions[10], [10, 10, 10]);
        assert!(multimodal_prompt_positions(4, &spans[..1], &[(2, 2)]).is_err());
    }

    const VISION_CONFIG: &[u8] = br#"{"size":{"longest_edge":16777216,"shortest_edge":65536},"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"processor_class":"Qwen3VLProcessor","image_processor_type":"Qwen2VLImageProcessorFast"}"#;

    fn signed_package(seed: u8, key_id: &str) -> (VerifiedQwen35Package, Vec<u8>) {
        let source_index = index_with_pinned_specs();
        let weight_map = source_index
            .tensor_names()
            .map(|name| {
                (
                    name.to_owned(),
                    json!(source_index.shard_for(name).expect("indexed tensor shard")),
                )
            })
            .collect::<serde_json::Map<String, serde_json::Value>>();
        let index_bytes = serde_json::to_vec(&json!({
            "metadata": { "total_size": source_index.total_size() },
            "weight_map": weight_map,
        }))
        .unwrap();
        let package = signed_package_for_index(seed, key_id, &index_bytes);
        (package, index_bytes)
    }

    fn signed_package_for_index(
        seed: u8,
        key_id: &str,
        index_bytes: &[u8],
    ) -> VerifiedQwen35Package {
        signed_package_for_metadata(seed, key_id, index_bytes, VISION_CONFIG)
    }

    fn signed_package_for_metadata(
        seed: u8,
        key_id: &str,
        index_bytes: &[u8],
        preprocessor_bytes: &[u8],
    ) -> VerifiedQwen35Package {
        let contents = BTreeMap::from([
            (CONFIG_NAME, b"pinned config placeholder".as_slice()),
            (SHARD_ONE, b"first shard placeholder".as_slice()),
            (SHARD_TWO, b"second shard placeholder".as_slice()),
            (INDEX_NAME, index_bytes),
            (
                crate::qwen35_vision::PREPROCESSOR_CONFIG_NAME,
                preprocessor_bytes,
            ),
            (TOKENIZER_NAME, b"pinned tokenizer placeholder".as_slice()),
        ]);
        let artifacts = contents
            .iter()
            .map(|(name, bytes)| ((*name).to_owned(), format!("{:x}", Sha256::digest(*bytes))))
            .collect();
        let key = SigningKey::from_bytes(&[seed; 32]);
        let mut manifest = Qwen35PackageManifest {
            schema_version: 2,
            checkpoint_revision: QWEN35_CHECKPOINT_REVISION.to_owned(),
            key_id: key_id.to_owned(),
            artifacts,
            signature_hex: String::new(),
        };
        manifest.signature_hex = manifest
            .signing_bytes()
            .map(|bytes| key.sign(&bytes))
            .map(|signature| {
                signature
                    .to_bytes()
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect()
            })
            .unwrap();
        manifest
            .verify_signature(&BTreeMap::from([(key_id.to_owned(), key.verifying_key())]))
            .unwrap()
    }

    struct PinnedCandidateEvaluationAssets {
        package: VerifiedQwen35Package,
        config: VerifiedQwen35Config,
        tokenizer: VerifiedQwen35Tokenizer,
        image_processor: VerifiedQwen35ImageProcessor,
        weights: VerifiedQwen35TextWeights<File, File>,
    }

    fn pinned_candidate_evaluation_assets(root: &Path) -> PinnedCandidateEvaluationAssets {
        let artifact_names = [
            CONFIG_NAME,
            SHARD_ONE,
            SHARD_TWO,
            INDEX_NAME,
            crate::qwen35_vision::PREPROCESSOR_CONFIG_NAME,
            TOKENIZER_NAME,
        ];
        let candidate_metadata_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../evals/models/qwen3.5-4b-candidate.json");
        let candidate_metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(candidate_metadata_path).expect("read pinned candidate metadata"),
        )
        .expect("parse pinned candidate metadata");
        assert_eq!(
            candidate_metadata["checkpoint_revision"],
            QWEN35_CHECKPOINT_REVISION
        );
        assert_eq!(candidate_metadata["status"], "candidate_not_admitted");
        assert_eq!(candidate_metadata["external_runtime_policy"], "prohibited");
        let source_artifacts = candidate_metadata["source_artifacts"]
            .as_object()
            .expect("candidate metadata pins every source artifact");
        assert_eq!(source_artifacts.len(), artifact_names.len());
        let mut opened_artifacts = BTreeMap::new();
        let artifacts = artifact_names
            .into_iter()
            .map(|name| {
                let source = source_artifacts
                    .get(name)
                    .expect("pinned candidate metadata contains the artifact");
                let file = File::open(root.join(name)).expect("open candidate model artifact");
                let actual_bytes = file
                    .metadata()
                    .expect("read opened artifact metadata")
                    .len();
                let expected_bytes = source["bytes"].as_u64().expect("pinned artifact size");
                assert!(expected_bytes > 0, "{name} is nonempty");
                assert_eq!(actual_bytes, expected_bytes, "{name} size");
                opened_artifacts.insert(name.to_owned(), file);
                (
                    name.to_owned(),
                    source["sha256"]
                        .as_str()
                        .expect("pinned artifact digest")
                        .to_owned(),
                )
            })
            .collect();
        // This deterministic signer exists only in the ignored evaluation
        // harness; it is not a product trust root or model admission key.
        let signing_key = SigningKey::from_bytes(&[0x53; 32]);
        let key_id = "local-checkpoint-evaluation";
        let mut manifest = Qwen35PackageManifest {
            schema_version: 2,
            checkpoint_revision: QWEN35_CHECKPOINT_REVISION.to_owned(),
            key_id: key_id.to_owned(),
            artifacts,
            signature_hex: String::new(),
        };
        manifest.signature_hex = signing_key
            .sign(
                &manifest
                    .signing_bytes()
                    .expect("encode evaluation manifest"),
            )
            .to_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let package = manifest
            .verify_signature(&BTreeMap::from([(
                key_id.to_owned(),
                signing_key.verifying_key(),
            )]))
            .expect("verify temporary evaluation-only package signature");

        let config = {
            VerifiedQwen35Config::parse(
                &package,
                package
                    .read_verified_small_artifact(
                        CONFIG_NAME,
                        opened_artifacts
                            .get_mut(CONFIG_NAME)
                            .expect("open candidate config"),
                    )
                    .expect("verify candidate config"),
            )
            .expect("parse package-bound candidate config")
        };
        let index = {
            Qwen35WeightIndex::parse_verified(
                &package,
                package
                    .read_verified_small_artifact(
                        INDEX_NAME,
                        opened_artifacts
                            .get_mut(INDEX_NAME)
                            .expect("open candidate index"),
                    )
                    .expect("verify candidate index"),
            )
            .expect("parse package-bound candidate index")
        };
        let tokenizer = {
            VerifiedQwen35Tokenizer::parse(
                &package,
                package
                    .read_verified_small_artifact(
                        TOKENIZER_NAME,
                        opened_artifacts
                            .get_mut(TOKENIZER_NAME)
                            .expect("open candidate tokenizer"),
                    )
                    .expect("verify candidate tokenizer"),
            )
            .expect("parse package-bound candidate tokenizer")
        };
        let image_processor = {
            let name = crate::qwen35_vision::PREPROCESSOR_CONFIG_NAME;
            VerifiedQwen35ImageProcessor::parse(
                &package,
                package
                    .read_verified_small_artifact(
                        name,
                        opened_artifacts
                            .get_mut(name)
                            .expect("open candidate image preprocessor"),
                    )
                    .expect("verify candidate image preprocessor"),
            )
            .expect("parse package-bound candidate image preprocessor")
        };
        let first = package
            .verify_owned_artifact(
                SHARD_ONE,
                opened_artifacts
                    .remove(SHARD_ONE)
                    .expect("open first candidate shard"),
            )
            .expect("verify first candidate shard");
        let second = package
            .verify_owned_artifact(
                SHARD_TWO,
                opened_artifacts
                    .remove(SHARD_TWO)
                    .expect("open second candidate shard"),
            )
            .expect("verify second candidate shard");
        let weights = VerifiedQwen35TextWeights::open(&package, index, first, second)
            .expect("validate both candidate safetensors headers and tensor maps");

        PinnedCandidateEvaluationAssets {
            package,
            config,
            tokenizer,
            image_processor,
            weights,
        }
    }

    #[test]
    #[ignore = "hashes and validates both full pinned safetensors shards without importing model tensors"]
    fn pinned_checkpoint_shard_headers_match_the_first_party_tensor_map() {
        let root = std::env::var_os("SAGE_QWEN35_PACKAGE_DIR")
            .map(std::path::PathBuf::from)
            .expect("set SAGE_QWEN35_PACKAGE_DIR to the exact pinned candidate directory");
        let assets = pinned_candidate_evaluation_assets(&root);
        assert_eq!(assets.weights.index.total_size(), 9_319_737_856);
        assert_eq!(assets.weights.index.index.tensor_names().count(), 738);
        println!(
            "pinned-qwen35-shard-headers tensors={} tensor_bytes={} config={} tokenizer={} processor={}",
            assets.weights.index.index.tensor_names().count(),
            assets.weights.index.total_size(),
            assets.config.manifest_sha256,
            assets.tokenizer.manifest_sha256,
            assets.image_processor.manifest_sha256,
        );
    }

    #[test]
    #[ignore = "imports the full pinned checkpoint and generates one constrained token; requires model files and available memory"]
    fn pinned_checkpoint_imports_and_generates_with_package_bound_candidate() {
        let root = std::env::var_os("SAGE_QWEN35_PACKAGE_DIR")
            .map(std::path::PathBuf::from)
            .expect("set SAGE_QWEN35_PACKAGE_DIR to the exact pinned candidate directory");
        let PinnedCandidateEvaluationAssets {
            package,
            config,
            tokenizer,
            image_processor,
            weights,
        } = pinned_candidate_evaluation_assets(&root);
        let mut candidate = weights
            .load_candidate_model(
                &package,
                config,
                tokenizer,
                image_processor,
                &ResourceGovernor::default(),
                Qwen35CandidateLoadOptions {
                    q4_group_size: 128,
                    maximum_context: SAGE_CONTEXT_LIMIT as usize,
                },
            )
            .expect("load the exact signed-package-bound text and vision candidate");
        assert_eq!(
            candidate.decoder_mut().maximum_context(),
            SAGE_CONTEXT_LIMIT as usize
        );
        assert_eq!(
            candidate.image_token_id(),
            candidate.tokenizer().token_id("<|image_pad|>").unwrap()
        );

        // Exercise the real loaded checkpoint through prompt prefill, the
        // vocabulary projection, and greedy token selection. The single-token
        // allow-list makes this a deterministic execution smoke test; it is
        // not a planner-quality or unconstrained-generation evaluation.
        let prompt = candidate
            .tokenizer()
            .encode_text("Sage can help me")
            .expect("encode a bounded generation prompt");
        let forced_token = candidate
            .tokenizer()
            .encode_text(" ready")
            .expect("encode one allowed output token")
            .into_iter()
            .next()
            .expect("ordinary text produces a token");
        let end_of_turn = candidate
            .tokenizer()
            .token_id("<|im_end|>")
            .expect("Qwen end-of-turn token exists");
        let started = std::time::Instant::now();
        let generated = crate::qwen35::generate_greedy(
            candidate.decoder_mut(),
            &prompt,
            &BTreeSet::from([end_of_turn]),
            1,
            |_, allowed| {
                allowed.clear();
                allowed.push(forced_token);
                Ok(true)
            },
            || false,
        )
        .expect("run one real-checkpoint constrained decode step");
        assert_eq!(generated, vec![forced_token]);
        println!(
            "pinned-qwen35-generation prompt_tokens={} output_tokens={} elapsed_ms={}",
            prompt.len(),
            generated.len(),
            started.elapsed().as_millis(),
        );
    }

    #[test]
    fn parsed_index_is_bound_to_the_manifest_that_signed_its_exact_bytes() {
        let (package, index_bytes) = signed_package(17, "test-key-a");
        let artifact = package
            .read_verified_small_artifact(INDEX_NAME, &mut Cursor::new(index_bytes.clone()))
            .unwrap();
        let index = Qwen35WeightIndex::parse_verified(&package, artifact).unwrap();
        assert_eq!(index.total_size(), 9_319_737_856);
        assert_eq!(index.manifest_sha256, package.manifest_sha256());

        let (different_package, _) = signed_package(18, "test-key-b");
        let first = different_package
            .verify_owned_artifact(SHARD_ONE, Cursor::new(b"first shard placeholder".to_vec()))
            .unwrap();
        let second = different_package
            .verify_owned_artifact(SHARD_TWO, Cursor::new(b"second shard placeholder".to_vec()))
            .unwrap();
        assert!(VerifiedQwen35TextWeights::open(&different_package, index, first, second).is_err());

        let mut changed = index_bytes;
        changed[0] ^= 1;
        assert!(
            package
                .read_verified_small_artifact(INDEX_NAME, &mut Cursor::new(changed))
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires model.safetensors.index.json from the pinned checkpoint"]
    fn pinned_checkpoint_memory_envelope_fits_the_initial_16_gib_tier() {
        let path = std::env::var_os("SAGE_QWEN35_INDEX_JSON")
            .expect("set SAGE_QWEN35_INDEX_JSON to the pinned model index");
        let index_bytes = std::fs::read(path).expect("read pinned Qwen tensor index");
        assert_eq!(
            format!("{:x}", Sha256::digest(&index_bytes)),
            "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
            "tensor index must come from Sage's exact pinned model revision"
        );

        let path = std::env::var_os("SAGE_QWEN35_PREPROCESSOR_JSON")
            .expect("set SAGE_QWEN35_PREPROCESSOR_JSON to the pinned image preprocessor");
        let preprocessor_bytes = std::fs::read(path).expect("read pinned image preprocessor");
        assert_eq!(
            format!("{:x}", Sha256::digest(&preprocessor_bytes)),
            "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "image preprocessor must come from Sage's exact pinned model revision"
        );

        let package =
            signed_package_for_metadata(31, "evaluation-only", &index_bytes, &preprocessor_bytes);
        let artifact = package
            .read_verified_small_artifact(INDEX_NAME, &mut Cursor::new(index_bytes))
            .expect("verify pinned index bytes against the local test manifest");
        let index = Qwen35WeightIndex::parse_verified(&package, artifact)
            .expect("parse package-bound pinned index");
        let artifact = package
            .read_verified_small_artifact(
                crate::qwen35_vision::PREPROCESSOR_CONFIG_NAME,
                &mut Cursor::new(preprocessor_bytes),
            )
            .expect("verify pinned preprocessor bytes against the local test manifest");
        let processor = VerifiedQwen35ImageProcessor::parse(&package, artifact)
            .expect("parse package-bound pinned image preprocessor");
        assert_eq!(processor.manifest_sha256, package.manifest_sha256());
        let pixels = vec![128_u8; 64 * 64 * 3];
        let prepared = processor
            .processor
            .prepare_rgb(
                Qwen35RgbImage {
                    width: 64,
                    height: 64,
                    pixels: &pixels,
                },
                64,
            )
            .expect("preprocess a bounded RGB image with the pinned profile");
        assert_eq!((prepared.width(), prepared.height()), (256, 256));
        assert_eq!(prepared.patch_grid(), (16, 16));
        assert_eq!(prepared.merged_tokens(), 64);
        let normalized_gray = 128.0_f32 / 127.5 - 1.0;
        assert!(
            prepared
                .patch_values()
                .iter()
                .all(|value| (value - normalized_gray).abs() < 1e-6)
        );
        let envelope = index
            .memory_envelope(128, SAGE_CONTEXT_LIMIT as usize)
            .expect("estimate the actual pinned candidate memory");
        let total = envelope.total().expect("total candidate memory");
        println!(
            "pinned-qwen35-memory-envelope weights={} state={} activations={} runtime={} total={}",
            envelope.weights, envelope.state, envelope.activations, envelope.runtime, total
        );
        assert!(
            total <= crate::resource::LOCAL_COMPUTE_BUDGET,
            "candidate estimate {total} exceeds the initial 8 GiB compute budget: {envelope:?}"
        );
    }

    #[test]
    fn image_processor_metadata_must_be_owned_by_the_same_signed_package() {
        let (package, _) = signed_package(23, "test-vision-key");
        let artifact = package
            .read_verified_small_artifact(
                crate::qwen35_vision::PREPROCESSOR_CONFIG_NAME,
                &mut Cursor::new(VISION_CONFIG.to_vec()),
            )
            .unwrap();
        let processor = VerifiedQwen35ImageProcessor::parse(&package, artifact).unwrap();
        assert_eq!(processor.manifest_sha256, package.manifest_sha256());
        assert_eq!(
            processor.processor.manifest_sha256(),
            package.manifest_sha256()
        );
    }
}
