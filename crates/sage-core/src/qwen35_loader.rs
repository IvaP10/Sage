//! Core's model-turn adapter over the standalone, package-bound Qwen runtime.
//!
//! Weight parsing, shard validation, quantization, memory admission, and model
//! assembly live in `sage-qwen35-runtime`. This module only adapts its bounded
//! candidate to Core's untrusted planner request and result contracts.

use std::io::{Read, Seek};

use crate::{CoreError, CoreResult};
use sage_model_package::{VerifiedPackageArtifact, VerifiedQwen35Package};
use sage_qwen35_runtime::{
    Qwen35Error,
    loader::{
        Qwen35CandidateModel as RuntimeCandidateModel,
        VerifiedQwen35TextWeights as RuntimeTextWeights,
    },
    qwen35::{Qwen35EmbeddedPrompt, Qwen35EmbeddedSpan},
    qwen35_vision::{PreparedQwen35Image, Qwen35RgbImage, Qwen35VisionEncoding},
    resource::ResourceGovernor,
};

pub use sage_qwen35_runtime::loader::{
    Qwen35CandidateLoadOptions, Qwen35DecoderLayerTensorNames, Qwen35LayerMixerTensorNames,
    Qwen35PackageIndexExt, VerifiedQwen35Config, VerifiedQwen35ImageProcessor,
    VerifiedQwen35Tokenizer, VerifiedQwen35WeightIndex, multimodal_prompt_positions,
};

const MAX_MULTIMODAL_IMAGES: usize = 8;
const MAX_IMAGE_TOKENS_PER_ATTACHMENT: usize = 64;
const MAX_TOTAL_IMAGE_PATCHES: usize = 256;

fn map_decode_error(error: sage_constrained_generation::DecodeError) -> CoreError {
    match error {
        sage_constrained_generation::DecodeError::Model(message) => CoreError::Model(message),
        sage_constrained_generation::DecodeError::Cancelled => CoreError::Cancelled,
    }
}

fn is_valid_turn_json(output: &str) -> bool {
    crate::model::validate_turn_json(output).is_ok()
}

fn discard_answer_preview(_: &str) {}

/// Core's planner-facing adapter. Its inner candidate owns the decoder and
/// memory reservation; output remains an untrusted proposal for Core.
pub struct Qwen35CandidateModel {
    inner: RuntimeCandidateModel,
}

impl Qwen35CandidateModel {
    pub fn prepare_vision_image(
        &self,
        image: Qwen35RgbImage<'_>,
        maximum_vision_tokens: usize,
    ) -> CoreResult<PreparedQwen35Image> {
        self.inner
            .prepare_vision_image(image, maximum_vision_tokens)
            .map_err(CoreError::from)
    }

    pub fn encode_vision_image(
        &self,
        image: Qwen35RgbImage<'_>,
        maximum_vision_tokens: usize,
    ) -> CoreResult<Qwen35VisionEncoding> {
        self.inner
            .encode_vision_image(image, maximum_vision_tokens)
            .map_err(CoreError::from)
    }

    pub fn generate_turn<C>(
        &mut self,
        context: &crate::model::TurnContext,
        maximum_new_tokens: usize,
        cancelled: C,
    ) -> CoreResult<crate::model::ModelTurn>
    where
        C: FnMut() -> bool,
    {
        self.generate_turn_with_answer_updates(context, maximum_new_tokens, |_| {}, cancelled)
    }

    pub fn generate_turn_with_answer_updates<C, U>(
        &mut self,
        context: &crate::model::TurnContext,
        maximum_new_tokens: usize,
        answer_update: U,
        cancelled: C,
    ) -> CoreResult<crate::model::ModelTurn>
    where
        C: FnMut() -> bool,
        U: FnMut(&str),
    {
        let maximum_context = self.inner.decoder_mut().maximum_context();
        let prompt = crate::qwen_prompt::encode_turn_prompt(
            self.inner.tokenizer(),
            context,
            maximum_new_tokens,
            maximum_context,
        )?;
        let end_of_turn_token_id = self
            .inner
            .tokenizer()
            .token_id("<|im_end|>")
            .ok_or_else(|| CoreError::Model("Qwen end-of-turn token is missing".into()))?;
        let (decoder, tokenizer) = self.inner.decoder_and_tokenizer_mut();
        let json = sage_constrained_generation::generation::generate_schema_greedy(
            decoder,
            tokenizer,
            &prompt.token_ids,
            sage_constrained_generation::generation::SchemaGenerationOptions {
                schema: &prompt.output_schema,
                independent_read_kinds: crate::model::INDEPENDENT_READ_TOOLS,
                end_of_turn_token_id,
                maximum_new_tokens,
                validate_complete: is_valid_turn_json,
                answer_update,
                cancelled,
            },
        )
        .map_err(map_decode_error)?;
        crate::model::parse_turn(&json, context.planning.task_id)
    }

    pub fn generate_turn_with_images<C>(
        &mut self,
        context: &crate::model::TurnContext,
        images: &[Qwen35RgbImage<'_>],
        maximum_vision_tokens_per_image: usize,
        maximum_new_tokens: usize,
        cancelled: C,
    ) -> CoreResult<crate::model::ModelTurn>
    where
        C: FnMut() -> bool,
    {
        if images.is_empty()
            || images.len() > MAX_MULTIMODAL_IMAGES
            || maximum_vision_tokens_per_image == 0
        {
            return Err(CoreError::Model(
                "Qwen multimodal evaluation requires one to eight bounded RGB images".into(),
            ));
        }

        let mut prepared_images = Vec::with_capacity(images.len());
        let mut total_patches = 0usize;
        for image in images {
            let prepared = self.inner.prepare_vision_image(
                *image,
                maximum_vision_tokens_per_image.min(MAX_IMAGE_TOKENS_PER_ATTACHMENT),
            )?;
            let (patches_high, patches_wide) = prepared.patch_grid();
            let patch_count = patches_high
                .checked_mul(patches_wide)
                .filter(|count| *count > 0 && *count <= MAX_TOTAL_IMAGE_PATCHES)
                .ok_or_else(|| {
                    CoreError::Model("Qwen image patch count exceeds its limit".into())
                })?;
            total_patches = total_patches
                .checked_add(patch_count)
                .filter(|total| *total <= MAX_TOTAL_IMAGE_PATCHES)
                .ok_or_else(|| {
                    CoreError::Model("Qwen multimodal patch budget was exceeded".into())
                })?;
            prepared_images.push(prepared);
        }

        let token_counts = prepared_images
            .iter()
            .map(PreparedQwen35Image::merged_tokens)
            .collect::<Vec<_>>();
        let maximum_context = self.inner.decoder_mut().maximum_context();
        let planner_prompt = crate::qwen_prompt::encode_turn_prompt_with_images(
            self.inner.tokenizer(),
            context,
            &token_counts,
            maximum_new_tokens,
            maximum_context,
        )?;
        let encodings = prepared_images
            .iter()
            .map(|image| self.inner.encode_prepared_vision_image(image))
            .collect::<Result<Vec<_>, Qwen35Error>>()?;
        if planner_prompt.image_spans.len() != prepared_images.len()
            || planner_prompt
                .image_spans
                .iter()
                .zip(&encodings)
                .any(|(span, encoding)| span.token_count != encoding.token_count())
        {
            return Err(CoreError::Model(
                "Qwen visual embeddings do not match their prompt image slots".into(),
            ));
        }

        let image_grids = prepared_images
            .iter()
            .map(|image| {
                let (patches_high, patches_wide) = image.patch_grid();
                (patches_high / 2, patches_wide / 2)
            })
            .collect::<Vec<_>>();
        let positions = multimodal_prompt_positions(
            planner_prompt.token_ids.len(),
            &planner_prompt.image_spans,
            &image_grids,
        )?;
        let embedded_spans = planner_prompt
            .image_spans
            .iter()
            .zip(&encodings)
            .map(|(span, encoding)| Qwen35EmbeddedSpan {
                token_start: span.token_start,
                token_id: self.inner.image_token_id(),
                embeddings: encoding.values(),
                positions: &positions[span.token_start..span.token_start + span.token_count],
            })
            .collect::<Vec<_>>();
        let embedded_prompt = Qwen35EmbeddedPrompt {
            token_ids: &planner_prompt.token_ids,
            positions: &positions,
            spans: &embedded_spans,
        };
        let end_of_turn_token_id = self
            .inner
            .tokenizer()
            .token_id("<|im_end|>")
            .ok_or_else(|| CoreError::Model("Qwen end-of-turn token is missing".into()))?;
        let (decoder, tokenizer) = self.inner.decoder_and_tokenizer_mut();
        let json = sage_constrained_generation::generation::generate_schema_with_embedded_prompt(
            decoder,
            tokenizer,
            &embedded_prompt,
            sage_constrained_generation::generation::SchemaGenerationOptions {
                schema: &planner_prompt.output_schema,
                independent_read_kinds: crate::model::INDEPENDENT_READ_TOOLS,
                end_of_turn_token_id,
                maximum_new_tokens,
                validate_complete: is_valid_turn_json,
                answer_update: discard_answer_preview,
                cancelled,
            },
        )
        .map_err(map_decode_error)?;
        crate::model::parse_turn(&json, context.planning.task_id)
    }
}

/// Core-local wrapper preserves the evaluation API while the actual signed
/// artifact parsing and model assembly remain in the standalone runtime.
pub struct VerifiedQwen35TextWeights<R1, R2> {
    inner: RuntimeTextWeights<R1, R2>,
}

impl<R1: Read + Seek, R2: Read + Seek> VerifiedQwen35TextWeights<R1, R2> {
    pub fn open(
        package: &VerifiedQwen35Package,
        index: VerifiedQwen35WeightIndex,
        first: VerifiedPackageArtifact<R1>,
        second: VerifiedPackageArtifact<R2>,
    ) -> CoreResult<Self> {
        Ok(Self {
            inner: RuntimeTextWeights::open(package, index, first, second)?,
        })
    }

    pub fn with_metal_backend(self) -> CoreResult<Self> {
        Ok(Self {
            inner: self.inner.with_metal_backend()?,
        })
    }

    pub fn load_candidate_model(
        self,
        package: &VerifiedQwen35Package,
        config: VerifiedQwen35Config,
        tokenizer: VerifiedQwen35Tokenizer,
        image_processor: VerifiedQwen35ImageProcessor,
        governor: &ResourceGovernor,
        options: Qwen35CandidateLoadOptions,
    ) -> CoreResult<Qwen35CandidateModel> {
        Ok(Qwen35CandidateModel {
            inner: self.inner.load_candidate_model(
                package,
                config,
                tokenizer,
                image_processor,
                governor,
                options,
            )?,
        })
    }
}
