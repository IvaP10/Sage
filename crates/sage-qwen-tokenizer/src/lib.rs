//! Sage-owned text tokenizer for the pinned Qwen3.5-4B BPE vocabulary.
//!
//! This module intentionally implements the tokenizer rather than loading a
//! tokenizer runtime. It accepts only the pinned NFC + Split + ByteLevel + BPE
//! layout, uses Sage's generated Unicode 15 tables, and keeps message framing
//! separate from user text so literal token-looking text cannot alter a chat
//! header or end a message.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::TokenizerResult as CoreResult;

mod unicode15;

/// Error returned when the pinned tokenizer artifact or a token stream is invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerError(String);

impl TokenizerError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for TokenizerError {}

/// Result type for the standalone Sage tokenizer.
pub type TokenizerResult<T> = Result<T, TokenizerError>;

const MAX_TOKENIZER_JSON_BYTES: usize = 64 * 1024 * 1024;
const MAX_TEXT_BYTES: usize = 256 * 1024;
const MAX_CHAT_MESSAGES: usize = 128;
const BASE_VOCAB_SIZE: usize = 248_044;
const ADDED_TOKEN_COUNT: usize = 26;
const TOKEN_COUNT: usize = BASE_VOCAB_SIZE + ADDED_TOKEN_COUNT;
const EXPECTED_MERGE_COUNT: usize = 247_587;
const MAX_TEXT_PIECE_BYTE_LENGTH: usize = 4 * 1024;
const MAX_TEXT_PIECE_TABLE_BYTES: usize = 64 * 1024 * 1024;
const MAX_MULTIMODAL_IMAGES: usize = 8;
const NO_TOKEN_NODE: u32 = u32::MAX;
const PRETOKEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChatMessage<'a> {
    pub role: ChatRole,
    pub content: &'a str,
}

/// Location of the image-pad token run emitted by the trusted multimodal
/// formatter. Callers replace exactly this run with package-bound vision
/// embeddings before decoder prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodedImageSpan {
    pub vision_start: usize,
    pub token_start: usize,
    pub token_count: usize,
    pub vision_end: usize,
}

#[derive(Debug, Clone, Copy)]
struct SpecialTokens {
    message_start: u32,
    message_end: u32,
    think: u32,
}

pub struct Qwen35Tokenizer {
    tokens: Vec<String>,
    token_ids: HashMap<String, u32>,
    merge_rules: HashMap<(u32, u32), MergeRule>,
    byte_token_ids: [u32; 256],
    character_to_byte: HashMap<u32, u8>,
    text_pieces: TextPieceTable,
    text_token_ids_by_first_byte: [Vec<u32>; 256],
    special: SpecialTokens,
}

struct TextPieceTable {
    ranges: Vec<Option<(usize, usize)>>,
    bytes: Vec<u8>,
}

/// Reusable scratch for vocabulary-ordered traversal of allowed first-byte
/// buckets. The heap and cursor array are retained across generated tokens.
#[doc(hidden)]
pub struct TextPieceCandidateWorkspace {
    cursors: [usize; 256],
    heap: BinaryHeap<Reverse<(u32, u8)>>,
}

impl Default for TextPieceCandidateWorkspace {
    fn default() -> Self {
        Self {
            cursors: [0; 256],
            heap: BinaryHeap::with_capacity(256),
        }
    }
}

impl TextPieceCandidateWorkspace {
    #[doc(hidden)]
    pub fn iter<'a>(
        &'a mut self,
        tokenizer: &'a Qwen35Tokenizer,
        permitted_first_bytes: &[bool; 256],
    ) -> TextPieceCandidateIter<'a> {
        self.cursors.fill(0);
        self.heap.clear();
        for (byte, permitted) in permitted_first_bytes.iter().copied().enumerate() {
            if permitted
                && let Some(token_id) = tokenizer.text_token_ids_by_first_byte[byte].first()
            {
                self.heap.push(Reverse((*token_id, byte as u8)));
            }
        }
        TextPieceCandidateIter {
            tokenizer,
            cursors: &mut self.cursors,
            heap: &mut self.heap,
        }
    }
}

/// Vocabulary-ordered merge over the tokenizer's sorted first-byte buckets.
#[doc(hidden)]
pub struct TextPieceCandidateIter<'a> {
    tokenizer: &'a Qwen35Tokenizer,
    cursors: &'a mut [usize; 256],
    heap: &'a mut BinaryHeap<Reverse<(u32, u8)>>,
}

impl<'a> Iterator for TextPieceCandidateIter<'a> {
    type Item = (u32, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let Reverse((token_id, first_byte)) = self.heap.pop()?;
        let bucket = &self.tokenizer.text_token_ids_by_first_byte[first_byte as usize];
        let cursor = &mut self.cursors[first_byte as usize];
        *cursor += 1;
        if let Some(next_token) = bucket.get(*cursor) {
            self.heap.push(Reverse((*next_token, first_byte)));
        }
        self.tokenizer
            .text_token_bytes(token_id)
            .map(|piece| (token_id, piece))
    }
}

#[derive(Debug, Clone, Copy)]
struct MergeRule {
    rank: u32,
    result: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
struct MergeCandidate {
    rank: u32,
    left: u32,
    right: u32,
    left_generation: u32,
    right_generation: u32,
}

#[derive(Debug)]
#[repr(C)]
struct TokenNode {
    token: u32,
    previous: u32,
    next: u32,
    generation: u32,
    alive: bool,
}

/// Scratch is shared across all message fragments in one prompt encoding.
/// Node links and heap indices are u32 because every accepted input is bounded
/// to 256 KiB; this cuts the link/queue working set and retains allocations
/// across short BPE pieces in the same prompt.
#[derive(Default)]
struct TokenizerWorkspace {
    characters: Vec<(usize, char)>,
    ranges: Vec<std::ops::Range<usize>>,
    nodes: Vec<TokenNode>,
    merge_candidates: BinaryHeap<Reverse<MergeCandidate>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTokenizer {
    version: String,
    truncation: (),
    padding: (),
    normalizer: RawNormalizer,
    pre_tokenizer: RawPreTokenizer,
    post_processor: RawByteLevel,
    decoder: RawByteLevel,
    model: RawBpeModel,
    added_tokens: Vec<RawAddedToken>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawNormalizer {
    #[serde(rename = "type")]
    normalizer_type: String,
}

#[derive(Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
enum RawPreTokenizer {
    Sequence {
        pretokenizers: Vec<RawPreTokenizer>,
    },
    Split {
        pattern: RawPattern,
        behavior: String,
        invert: bool,
    },
    ByteLevel {
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPattern {
    #[serde(rename = "Regex")]
    regex: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawByteLevel {
    #[serde(rename = "type")]
    decoder_type: String,
    add_prefix_space: bool,
    trim_offsets: bool,
    use_regex: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBpeModel {
    #[serde(rename = "type")]
    model_type: String,
    #[serde(rename = "dropout")]
    _dropout: (),
    #[serde(rename = "unk_token")]
    _unk_token: (),
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: Option<bool>,
    byte_fallback: Option<bool>,
    ignore_merges: Option<bool>,
    vocab: RawVocabulary,
    merges: Vec<RawMerge>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawMerge {
    Pair([String; 2]),
    Text(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAddedToken {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

struct RawVocabulary(Vec<(String, u32)>);

impl<'de> Deserialize<'de> for RawVocabulary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct VocabularyVisitor;

        impl<'de> Visitor<'de> for VocabularyVisitor {
            type Value = RawVocabulary;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string-to-token-id vocabulary object")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut entries = Vec::with_capacity(BASE_VOCAB_SIZE);
                let mut names = HashSet::with_capacity(BASE_VOCAB_SIZE);
                while let Some((name, id)) = map.next_entry::<String, u32>()? {
                    if !names.insert(name.clone()) {
                        return Err(serde::de::Error::custom("duplicate vocabulary token"));
                    }
                    entries.push((name, id));
                    if entries.len() > BASE_VOCAB_SIZE {
                        return Err(serde::de::Error::custom("vocabulary exceeds pinned size"));
                    }
                }
                Ok(RawVocabulary(entries))
            }
        }

        deserializer.deserialize_map(VocabularyVisitor)
    }
}

impl Qwen35Tokenizer {
    /// Parse and validate the tokenizer artifact pinned for Sage's Qwen3.5-4B
    /// candidate. A future tokenizer revision needs a deliberate profile update.
    pub fn from_json(bytes: &[u8]) -> CoreResult<Self> {
        if bytes.is_empty() || bytes.len() > MAX_TOKENIZER_JSON_BYTES {
            return Err(model_error("tokenizer artifact is empty or exceeds 64 MiB"));
        }
        let raw: RawTokenizer = serde_json::from_slice(bytes)
            .map_err(|_| model_error("tokenizer artifact is malformed or unsupported"))?;
        validate_tokenizer_shape(&raw)?;

        let mut tokens: Vec<Option<String>> = vec![None; TOKEN_COUNT];
        let mut token_ids = HashMap::with_capacity(TOKEN_COUNT);
        for (token, id) in raw.model.vocab.0 {
            let index = id as usize;
            if index >= BASE_VOCAB_SIZE || tokens[index].is_some() {
                return Err(model_error(
                    "base vocabulary ids are not unique and contiguous",
                ));
            }
            tokens[index] = Some(token.clone());
            token_ids.insert(token, id);
        }
        if token_ids.len() != BASE_VOCAB_SIZE
            || tokens[..BASE_VOCAB_SIZE].iter().any(Option::is_none)
        {
            return Err(model_error(
                "base vocabulary does not cover the pinned id range",
            ));
        }

        for added in &raw.added_tokens {
            let index = added.id as usize;
            if !(BASE_VOCAB_SIZE..TOKEN_COUNT).contains(&index)
                || tokens[index].is_some()
                || token_ids.contains_key(&added.content)
            {
                return Err(model_error("added-token ids or contents are not unique"));
            }
            tokens[index] = Some(added.content.clone());
            token_ids.insert(added.content.clone(), added.id);
        }
        if tokens.iter().any(Option::is_none) || token_ids.len() != TOKEN_COUNT {
            return Err(model_error("added tokens do not cover the pinned id range"));
        }
        let tokens = tokens.into_iter().map(Option::unwrap).collect::<Vec<_>>();

        let mut merge_rules = HashMap::with_capacity(EXPECTED_MERGE_COUNT);
        for (rank, merge) in raw.model.merges.iter().enumerate() {
            let (left, right) = merge
                .pieces()
                .ok_or_else(|| model_error("merge entry has an invalid representation"))?;
            let left_id = *token_ids
                .get(left)
                .ok_or_else(|| model_error("merge refers to an unknown left token"))?;
            let right_id = *token_ids
                .get(right)
                .ok_or_else(|| model_error("merge refers to an unknown right token"))?;
            let mut joined = String::with_capacity(left.len() + right.len());
            joined.push_str(left);
            joined.push_str(right);
            let result = *token_ids
                .get(&joined)
                .ok_or_else(|| model_error("merge result is missing from the vocabulary"))?;
            if merge_rules
                .insert(
                    (left_id, right_id),
                    MergeRule {
                        rank: rank as u32,
                        result,
                    },
                )
                .is_some()
            {
                return Err(model_error("merge table contains a duplicate pair"));
            }
        }

        let (byte_to_character, character_to_byte) = byte_level_alphabet();
        for token in &tokens[..BASE_VOCAB_SIZE] {
            if !token
                .chars()
                .all(|character| character_to_byte.contains_key(&(character as u32)))
            {
                return Err(model_error(
                    "base vocabulary contains a non-byte-level symbol",
                ));
            }
        }
        let mut byte_token_ids = [0u32; 256];
        for (byte, character) in byte_to_character.iter().enumerate() {
            byte_token_ids[byte] = *token_ids
                .get(&character.to_string())
                .ok_or_else(|| model_error("base vocabulary is missing a byte symbol"))?;
        }

        let special = SpecialTokens {
            message_start: required_token(&token_ids, "<|im_start|>", 248_045)?,
            message_end: required_token(&token_ids, "<|im_end|>", 248_046)?,
            think: required_token(&token_ids, "<think>", 248_068)?,
        };

        // Store the byte-level spelling for each ordinary BPE token once.
        // Constrained generation asks about many vocabulary IDs per output
        // token, so rebuilding temporary vectors there would create hundreds
        // of thousands of short-lived allocations on every turn.
        let text_pieces = build_text_piece_table(&tokens, BASE_VOCAB_SIZE, &character_to_byte)?;
        let text_token_ids_by_first_byte = build_text_token_first_byte_index(&text_pieces)?;

        Ok(Self {
            tokens,
            token_ids,
            merge_rules,
            byte_token_ids,
            character_to_byte,
            text_pieces,
            text_token_ids_by_first_byte,
            special,
        })
    }

    /// Encode ordinary text. Added-token spellings are deliberately treated as
    /// literal text; only the trusted chat formatter below can emit framing IDs.
    pub fn encode_text(&self, text: &str) -> CoreResult<Vec<u32>> {
        let mut encoded = Vec::new();
        let mut workspace = TokenizerWorkspace::default();
        self.encode_text_into(text, &mut encoded, 8_192, &mut workspace)?;
        Ok(encoded)
    }

    /// Format text-only Qwen chat turns with structural IDs emitted separately
    /// from message content. Image, video, tool-routing, and audio formats are
    /// not admitted by this first text profile.
    pub fn encode_chat(
        &self,
        messages: &[ChatMessage<'_>],
        add_generation_prompt: bool,
    ) -> CoreResult<Vec<u32>> {
        if messages.len() > MAX_CHAT_MESSAGES {
            return Err(model_error("chat exceeds Sage's 128-message bound"));
        }
        let total_bytes = messages.iter().try_fold(0usize, |total, message| {
            total.checked_add(message.content.len())
        });
        if total_bytes.is_none_or(|total| total > MAX_TEXT_BYTES) {
            return Err(model_error("chat text exceeds Sage's 256 KiB bound"));
        }

        let mut output = Vec::new();
        let mut workspace = TokenizerWorkspace::default();
        for message in messages {
            output.push(self.special.message_start);
            self.encode_text_into(message.role.as_str(), &mut output, 8_192, &mut workspace)?;
            self.encode_text_into("\n", &mut output, 8_192, &mut workspace)?;
            self.encode_text_into(message.content, &mut output, 8_192, &mut workspace)?;
            output.push(self.special.message_end);
            self.encode_text_into("\n", &mut output, 8_192, &mut workspace)?;
            if output.len() > 8_192 {
                return Err(model_error("chat exceeds Sage's 8,192-token context bound"));
            }
        }
        if add_generation_prompt {
            output.push(self.special.message_start);
            self.encode_text_into("assistant\n", &mut output, 8_192, &mut workspace)?;
            output.push(self.special.think);
            self.encode_text_into("\n", &mut output, 8_192, &mut workspace)?;
        }
        if output.len() > 8_192 {
            return Err(model_error("chat exceeds Sage's 8,192-token context bound"));
        }
        Ok(output)
    }

    /// Encode a planner chat with trusted image slots appended to its single
    /// user message. Literal special-token spellings in message text remain
    /// ordinary bytes; only this formatter emits vision framing IDs.
    pub fn encode_chat_with_images(
        &self,
        messages: &[ChatMessage<'_>],
        merged_token_counts: &[usize],
        add_generation_prompt: bool,
    ) -> CoreResult<(Vec<u32>, Vec<EncodedImageSpan>)> {
        if messages.len() > MAX_CHAT_MESSAGES
            || merged_token_counts.len() > MAX_MULTIMODAL_IMAGES
            || merged_token_counts.contains(&0)
            || messages
                .iter()
                .filter(|message| message.role == ChatRole::User)
                .count()
                != 1
        {
            return Err(model_error(
                "multimodal chat requires one user message and at most eight nonempty images",
            ));
        }
        let image_token_budget = merged_token_counts
            .iter()
            .try_fold(0usize, |total, count| total.checked_add(*count));
        if image_token_budget.is_none_or(|total| total > 8_192) {
            return Err(model_error(
                "multimodal image slots exceed Sage's 8,192-token context bound",
            ));
        }
        let total_bytes = messages.iter().try_fold(0usize, |total, message| {
            total.checked_add(message.content.len())
        });
        if total_bytes.is_none_or(|total| total > MAX_TEXT_BYTES) {
            return Err(model_error("chat text exceeds Sage's 256 KiB bound"));
        }

        let mut image_tokens = None;
        if !merged_token_counts.is_empty() {
            let vision_start = self
                .token_id("<|vision_start|>")
                .filter(|id| self.text_token_bytes(*id).is_none())
                .ok_or_else(|| model_error("Qwen vision-start token is missing"))?;
            let image_pad = self
                .token_id("<|image_pad|>")
                .filter(|id| self.text_token_bytes(*id).is_none())
                .ok_or_else(|| model_error("Qwen image-pad token is missing"))?;
            let vision_end = self
                .token_id("<|vision_end|>")
                .filter(|id| self.text_token_bytes(*id).is_none())
                .ok_or_else(|| model_error("Qwen vision-end token is missing"))?;
            image_tokens = Some((vision_start, image_pad, vision_end));
        }

        let mut output = Vec::new();
        let mut image_spans = Vec::with_capacity(merged_token_counts.len());
        let mut image_index = 0usize;
        let mut workspace = TokenizerWorkspace::default();
        for message in messages {
            output.push(self.special.message_start);
            self.encode_text_into(message.role.as_str(), &mut output, 8_192, &mut workspace)?;
            self.encode_text_into("\n", &mut output, 8_192, &mut workspace)?;
            self.encode_text_into(message.content, &mut output, 8_192, &mut workspace)?;
            if message.role == ChatRole::User {
                for token_count in merged_token_counts {
                    self.encode_text_into(
                        &format!(
                            "\n\nAttached image {} (visual evidence):\n",
                            image_index + 1
                        ),
                        &mut output,
                        8_192,
                        &mut workspace,
                    )?;
                    let (vision_start, image_pad, vision_end) = image_tokens.ok_or_else(|| {
                        model_error("Qwen image framing tokens were not initialized")
                    })?;
                    let start_marker = output.len();
                    output.push(vision_start);
                    let token_start = output.len();
                    for _ in 0..*token_count {
                        output.push(image_pad);
                    }
                    let vision_end_index = output.len();
                    output.push(vision_end);
                    image_spans.push(EncodedImageSpan {
                        vision_start: start_marker,
                        token_start,
                        token_count: *token_count,
                        vision_end: vision_end_index,
                    });
                    image_index += 1;
                    if output.len() > 8_192 {
                        return Err(model_error(
                            "multimodal chat exceeds Sage's 8,192-token context bound",
                        ));
                    }
                }
            }
            output.push(self.special.message_end);
            self.encode_text_into("\n", &mut output, 8_192, &mut workspace)?;
            if output.len() > 8_192 {
                return Err(model_error(
                    "multimodal chat exceeds Sage's 8,192-token context bound",
                ));
            }
        }
        if image_index != merged_token_counts.len() {
            return Err(model_error(
                "multimodal images were not bound to the user message",
            ));
        }
        if add_generation_prompt {
            output.push(self.special.message_start);
            self.encode_text_into("assistant\n", &mut output, 8_192, &mut workspace)?;
            output.push(self.special.think);
            self.encode_text_into("\n", &mut output, 8_192, &mut workspace)?;
        }
        if output.len() > 8_192 {
            return Err(model_error(
                "multimodal chat exceeds Sage's 8,192-token context bound",
            ));
        }
        Ok((output, image_spans))
    }

    /// Decode byte-level text and render added tokens by their visible spelling.
    pub fn decode(&self, ids: &[u32]) -> CoreResult<String> {
        if ids.len() > 8_192 {
            return Err(model_error(
                "decode exceeds Sage's 8,192-token context bound",
            ));
        }
        let mut bytes = Vec::with_capacity(ids.len().saturating_mul(2));
        for id in ids {
            let token = self
                .tokens
                .get(*id as usize)
                .ok_or_else(|| model_error("token id is outside the pinned tokenizer range"))?;
            if (*id as usize) < BASE_VOCAB_SIZE {
                for character in token.chars() {
                    let byte = self
                        .character_to_byte
                        .get(&(character as u32))
                        .copied()
                        .ok_or_else(|| model_error("base token has an invalid byte symbol"))?;
                    bytes.push(byte);
                }
            } else {
                bytes.extend_from_slice(token.as_bytes());
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn token_id(&self, spelling: &str) -> Option<u32> {
        self.token_ids.get(spelling).copied()
    }

    /// Return byte-level output for one ordinary BPE token. Added control and
    /// framing tokens are kept out of constrained JSON text generation.
    #[doc(hidden)]
    pub fn text_token_bytes(&self, id: u32) -> Option<&[u8]> {
        let (start, length) = self
            .text_pieces
            .ranges
            .get(id as usize)
            .copied()
            .flatten()?;
        self.text_pieces.bytes.get(start..start + length)
    }

    #[doc(hidden)]
    pub fn vocabulary_size(&self) -> usize {
        self.tokens.len()
    }

    #[cfg(feature = "core-test-support")]
    #[doc(hidden)]
    pub fn for_test_text_bytes(bytes: &[u8]) -> Self {
        let text_pieces = TextPieceTable {
            ranges: (0..bytes.len()).map(|offset| Some((offset, 1))).collect(),
            bytes: bytes.to_vec(),
        };
        let text_token_ids_by_first_byte =
            build_text_token_first_byte_index(&text_pieces).expect("valid test byte pieces");
        Self {
            tokens: vec![String::new(); bytes.len()],
            token_ids: HashMap::new(),
            merge_rules: HashMap::new(),
            byte_token_ids: [0; 256],
            character_to_byte: HashMap::new(),
            text_pieces,
            text_token_ids_by_first_byte,
            special: SpecialTokens {
                message_start: 0,
                message_end: 0,
                think: 0,
            },
        }
    }

    #[cfg(feature = "core-test-support")]
    #[doc(hidden)]
    pub fn for_test_text_pieces(pieces: &[&[u8]]) -> Self {
        let total_bytes = pieces.iter().map(|piece| piece.len()).sum();
        let mut bytes = Vec::with_capacity(total_bytes);
        let mut ranges = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let start = bytes.len();
            bytes.extend_from_slice(piece);
            ranges.push(Some((start, piece.len())));
        }
        let text_pieces = TextPieceTable { ranges, bytes };
        let text_token_ids_by_first_byte =
            build_text_token_first_byte_index(&text_pieces).expect("valid test text pieces");
        Self {
            tokens: vec![String::new(); pieces.len()],
            token_ids: HashMap::new(),
            merge_rules: HashMap::new(),
            byte_token_ids: [0; 256],
            character_to_byte: HashMap::new(),
            text_pieces,
            text_token_ids_by_first_byte,
            special: SpecialTokens {
                message_start: 0,
                message_end: 0,
                think: 0,
            },
        }
    }

    pub fn unicode_data_version(&self) -> &'static str {
        unicode15::data_version()
    }

    fn encode_text_into(
        &self,
        text: &str,
        output: &mut Vec<u32>,
        max_tokens: usize,
        workspace: &mut TokenizerWorkspace,
    ) -> CoreResult<()> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(model_error("text exceeds Sage's 256 KiB tokenizer bound"));
        }
        // ASCII is already NFC. Borrow it directly so ordinary English prompts
        // do not allocate the normalizer's decomposition and output buffers.
        // Every non-ASCII input still goes through Sage's pinned Unicode path.
        let normalized = if text.is_ascii() {
            Cow::Borrowed(text)
        } else {
            Cow::Owned(unicode15::normalize_nfc(text)?)
        };
        self.encode_normalized_into(normalized.as_ref(), output, max_tokens, workspace)
    }

    fn encode_normalized_into(
        &self,
        normalized: &str,
        output: &mut Vec<u32>,
        max_tokens: usize,
        workspace: &mut TokenizerWorkspace,
    ) -> CoreResult<()> {
        let range_budget = max_tokens.saturating_sub(output.len());
        pretoken_ranges_into(
            normalized,
            &mut workspace.characters,
            &mut workspace.ranges,
            range_budget,
        )?;
        workspace.characters.clear();
        for index in 0..workspace.ranges.len() {
            let start = workspace.ranges[index].start;
            let end = workspace.ranges[index].end;
            self.encode_piece(&normalized[start..end], output, max_tokens, workspace)?;
        }
        if output.len() > max_tokens {
            return Err(model_error("tokenized input exceeds Sage's token bound"));
        }
        Ok(())
    }

    fn encode_piece(
        &self,
        piece: &str,
        output: &mut Vec<u32>,
        max_tokens: usize,
        workspace: &mut TokenizerWorkspace,
    ) -> CoreResult<()> {
        if piece.len() >= u32::MAX as usize {
            return Err(model_error("tokenizer piece exceeds its node-index bound"));
        }
        let TokenizerWorkspace {
            nodes,
            merge_candidates,
            ..
        } = workspace;
        nodes.clear();
        if nodes.capacity() < piece.len() {
            nodes.reserve(piece.len());
        }
        merge_candidates.clear();
        for (index, byte) in piece.bytes().enumerate() {
            let token = self.byte_token_ids[byte as usize];
            nodes.push(TokenNode {
                token,
                previous: if index == 0 {
                    NO_TOKEN_NODE
                } else {
                    (index - 1) as u32
                },
                next: NO_TOKEN_NODE,
                generation: 0,
                alive: true,
            });
            if index > 0 {
                nodes[index - 1].next = index as u32;
            }
        }
        if nodes.is_empty() {
            return Ok(());
        }

        for left in 0..nodes.len().saturating_sub(1) {
            push_merge_candidate(nodes, &self.merge_rules, left as u32, merge_candidates);
        }
        while let Some(Reverse(candidate)) = merge_candidates.pop() {
            let left = candidate.left as usize;
            let right = candidate.right as usize;
            if !nodes[left].alive
                || !nodes[right].alive
                || nodes[left].next != candidate.right
                || nodes[left].generation != candidate.left_generation
                || nodes[right].generation != candidate.right_generation
            {
                continue;
            }
            let Some(rule) = self
                .merge_rules
                .get(&(nodes[left].token, nodes[right].token))
                .copied()
            else {
                continue;
            };
            if rule.rank != candidate.rank {
                continue;
            }
            let after = nodes[right].next;
            nodes[left].token = rule.result;
            nodes[left].generation = nodes[left].generation.wrapping_add(1);
            nodes[left].next = after;
            nodes[right].alive = false;
            nodes[right].generation = nodes[right].generation.wrapping_add(1);
            if after != NO_TOKEN_NODE {
                nodes[after as usize].previous = candidate.left;
            }
            let before = nodes[left].previous;
            if before != NO_TOKEN_NODE {
                push_merge_candidate(nodes, &self.merge_rules, before, merge_candidates);
            }
            push_merge_candidate(nodes, &self.merge_rules, candidate.left, merge_candidates);
        }

        let mut cursor = 0u32;
        while cursor != NO_TOKEN_NODE {
            let index = cursor as usize;
            if nodes[index].alive {
                output.push(nodes[index].token);
                if output.len() > max_tokens {
                    return Err(model_error("tokenized input exceeds Sage's token bound"));
                }
            }
            cursor = nodes[index].next;
        }
        Ok(())
    }
}

impl RawMerge {
    fn pieces(&self) -> Option<(&str, &str)> {
        match self {
            Self::Pair([left, right]) => Some((left, right)),
            Self::Text(value) => {
                let (left, right) = value.split_once(' ')?;
                (!left.is_empty() && !right.is_empty() && !right.contains(' '))
                    .then_some((left, right))
            }
        }
    }
}

fn validate_tokenizer_shape(raw: &RawTokenizer) -> CoreResult<()> {
    let _ = (&raw.truncation, &raw.padding);
    if raw.version != "1.0"
        || raw.model.model_type != "BPE"
        || raw.model.continuing_subword_prefix.as_deref() != Some("")
        || raw.model.end_of_word_suffix.as_deref() != Some("")
        || raw.model.fuse_unk != Some(false)
        || raw.model.byte_fallback != Some(false)
        || raw.model.ignore_merges != Some(false)
        || raw.model.vocab.0.len() != BASE_VOCAB_SIZE
        || raw.model.merges.len() != EXPECTED_MERGE_COUNT
        || raw.added_tokens.len() != ADDED_TOKEN_COUNT
    {
        return Err(model_error(
            "tokenizer artifact does not match Sage's pinned BPE profile",
        ));
    }

    if raw.normalizer.normalizer_type != "NFC" {
        return Err(model_error("tokenizer normalizer is not pinned NFC"));
    }
    if !is_bytelevel_config(&raw.decoder) || !is_bytelevel_config(&raw.post_processor) {
        return Err(model_error(
            "tokenizer decoder or post-processor differs from pinned ByteLevel settings",
        ));
    }

    let RawPreTokenizer::Sequence { pretokenizers } = &raw.pre_tokenizer else {
        return Err(model_error(
            "tokenizer pre-tokenizer sequence is unsupported",
        ));
    };
    let [
        RawPreTokenizer::Split {
            pattern,
            behavior,
            invert,
        },
        RawPreTokenizer::ByteLevel {
            add_prefix_space,
            trim_offsets,
            use_regex,
        },
    ] = pretokenizers.as_slice()
    else {
        return Err(model_error(
            "tokenizer pre-tokenizer sequence is unsupported",
        ));
    };
    if pattern.regex != PRETOKEN_PATTERN || behavior != "Isolated" || *invert {
        return Err(model_error(
            "tokenizer split pattern differs from the pinned profile",
        ));
    }
    if *add_prefix_space || *trim_offsets || *use_regex {
        return Err(model_error(
            "tokenizer ByteLevel settings differ from the pinned profile",
        ));
    }

    let mut added_ids = HashSet::with_capacity(ADDED_TOKEN_COUNT);
    let mut added_spellings = HashSet::with_capacity(ADDED_TOKEN_COUNT);
    for added in &raw.added_tokens {
        if added.content.is_empty()
            || added.content.len() > 256
            || !(BASE_VOCAB_SIZE as u32..TOKEN_COUNT as u32).contains(&added.id)
            || !added_ids.insert(added.id)
            || !added_spellings.insert(added.content.as_str())
        {
            return Err(model_error("added-token entry is invalid or duplicated"));
        }
        // AddedToken behavior flags affect runtime matching. Sage only emits
        // framing IDs directly and encodes all message bodies as plain text.
        let _ = (
            added.single_word,
            added.lstrip,
            added.rstrip,
            added.normalized,
            added.special,
        );
    }
    Ok(())
}

fn required_token(ids: &HashMap<String, u32>, spelling: &str, expected: u32) -> CoreResult<u32> {
    match ids.get(spelling).copied() {
        Some(id) if id == expected => Ok(id),
        _ => Err(model_error("required chat token is absent or moved")),
    }
}

fn is_bytelevel_config(value: &RawByteLevel) -> bool {
    value.decoder_type == "ByteLevel"
        && !value.add_prefix_space
        && !value.trim_offsets
        && !value.use_regex
}

fn byte_level_alphabet() -> ([char; 256], HashMap<u32, u8>) {
    let mut visible = (b'!'..=b'~').collect::<Vec<u8>>();
    visible.extend(161..=172);
    visible.extend(174..=255);
    let mut byte_to_character = ['\u{0000}'; 256];
    let mut character_to_byte = HashMap::with_capacity(256);
    for byte in &visible {
        let character = char::from(*byte);
        byte_to_character[*byte as usize] = character;
        character_to_byte.insert(character as u32, *byte);
    }
    let mut next_codepoint = 256u32;
    for byte in 0u16..=255 {
        let byte = byte as u8;
        if !visible.contains(&byte) {
            let character = char::from_u32(next_codepoint).expect("byte alphabet is valid Unicode");
            byte_to_character[byte as usize] = character;
            character_to_byte.insert(character as u32, byte);
            next_codepoint += 1;
        }
    }
    (byte_to_character, character_to_byte)
}

fn push_merge_candidate(
    nodes: &[TokenNode],
    rules: &HashMap<(u32, u32), MergeRule>,
    left: u32,
    queue: &mut BinaryHeap<Reverse<MergeCandidate>>,
) {
    let left_index = left as usize;
    let right = nodes[left_index].next;
    if right == NO_TOKEN_NODE {
        return;
    }
    let right_index = right as usize;
    if !nodes[left_index].alive || !nodes[right_index].alive {
        return;
    }
    if let Some(rule) = rules.get(&(nodes[left_index].token, nodes[right_index].token)) {
        queue.push(Reverse(MergeCandidate {
            rank: rule.rank,
            left,
            right,
            left_generation: nodes[left_index].generation,
            right_generation: nodes[right_index].generation,
        }));
    }
}

#[cfg(test)]
fn pretoken_ranges(text: &str) -> Vec<std::ops::Range<usize>> {
    let mut characters = Vec::new();
    let mut ranges = Vec::new();
    pretoken_ranges_into(text, &mut characters, &mut ranges, usize::MAX)
        .expect("unbounded test reference has no range limit");
    ranges
}

fn pretoken_ranges_into(
    text: &str,
    characters: &mut Vec<(usize, char)>,
    ranges: &mut Vec<std::ops::Range<usize>>,
    maximum_ranges: usize,
) -> CoreResult<()> {
    characters.clear();
    characters.extend(text.char_indices());
    ranges.clear();
    // Most text pieces are words or punctuation runs. Start with a small
    // bounded reserve and grow only when the observed split count needs it;
    // a 256 KiB single-word prompt should not reserve 256 KiB ranges.
    let initial_ranges = characters.len().min(maximum_ranges).min(256);
    if ranges.capacity() < initial_ranges {
        ranges.reserve(initial_ranges);
    }
    let mut index = 0;
    while index < characters.len() {
        let start = index;
        let current = characters[index].1;

        if let Some(end) = contraction_end(characters, index) {
            index = end;
        } else if unicode15::is_letter(current) || unicode15::is_mark(current) {
            index += 1;
            while index < characters.len()
                && (unicode15::is_letter(characters[index].1)
                    || unicode15::is_mark(characters[index].1))
            {
                index += 1;
            }
        } else if index + 1 < characters.len()
            && current != '\r'
            && current != '\n'
            && !unicode15::is_letter(current)
            && !unicode15::is_number(current)
            && (unicode15::is_letter(characters[index + 1].1)
                || unicode15::is_mark(characters[index + 1].1))
        {
            index += 2;
            while index < characters.len()
                && (unicode15::is_letter(characters[index].1)
                    || unicode15::is_mark(characters[index].1))
            {
                index += 1;
            }
        } else if unicode15::is_number(current) {
            index += 1;
        } else if unicode15::is_whitespace(current) {
            let whitespace_start = index;
            index += 1;
            while index < characters.len() && unicode15::is_whitespace(characters[index].1) {
                index += 1;
            }
            let whitespace_end = index;
            if let Some(last_newline) = (whitespace_start..whitespace_end)
                .rev()
                .find(|position| matches!(characters[*position].1, '\r' | '\n'))
            {
                // The \s*[\r\n]+ arm takes the run through its final newline
                // sequence; horizontal whitespace after that is reconsidered.
                index = last_newline + 1;
            } else if index < characters.len() && whitespace_end - whitespace_start > 1 {
                // The \s+(?!\S) arm leaves one whitespace scalar for the next
                // alternative, which may bind an ASCII space to a word or
                // punctuation run.
                index = whitespace_end - 1;
            }
        } else {
            if current == ' '
                && index + 1 < characters.len()
                && is_punctuation(characters[index + 1].1)
            {
                index += 1;
            }
            while index < characters.len() && is_punctuation(characters[index].1) {
                index += 1;
            }
            while index < characters.len() && matches!(characters[index].1, '\r' | '\n') {
                index += 1;
            }
        }

        let start_byte = characters[start].0;
        let end_byte = characters
            .get(index)
            .map(|(byte, _)| *byte)
            .unwrap_or(text.len());
        // Every nonempty pre-token produces at least one BPE token, because
        // merges combine nonempty pieces. Reject a lower bound that already
        // exceeds the caller's remaining token budget before growing range scratch.
        if ranges.len() >= maximum_ranges {
            return Err(model_error("tokenized input exceeds Sage's token bound"));
        }
        ranges.push(start_byte..end_byte);
    }
    Ok(())
}

fn contraction_end(characters: &[(usize, char)], start: usize) -> Option<usize> {
    if characters.get(start)?.1 != '\'' {
        return None;
    }
    let remaining = characters.len() - start;
    for contraction in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        let count = contraction.chars().count();
        if remaining < count {
            continue;
        }
        if characters[start..start + count]
            .iter()
            .map(|(_, character)| *character)
            .zip(contraction.chars())
            .all(|(left, right)| left.eq_ignore_ascii_case(&right))
        {
            return Some(start + count);
        }
    }
    None
}

fn is_punctuation(character: char) -> bool {
    !unicode15::is_whitespace(character) && !unicode15::is_letter_mark_or_number(character)
}

fn model_error(message: &str) -> TokenizerError {
    TokenizerError::new(message)
}

fn build_text_piece_table(
    tokens: &[String],
    ordinary_token_count: usize,
    character_to_byte: &HashMap<u32, u8>,
) -> CoreResult<TextPieceTable> {
    if ordinary_token_count > tokens.len() || ordinary_token_count > BASE_VOCAB_SIZE {
        return Err(model_error(
            "ordinary token count is outside the pinned vocabulary",
        ));
    }
    let mut ranges = vec![None; tokens.len()];
    let mut bytes = Vec::new();
    for (token_id, token) in tokens.iter().take(ordinary_token_count).enumerate() {
        let piece_length = token.chars().count();
        if piece_length == 0 || piece_length > MAX_TEXT_PIECE_BYTE_LENGTH {
            return Err(model_error(
                "ordinary token byte spelling is empty or exceeds 4 KiB",
            ));
        }
        if bytes
            .len()
            .checked_add(piece_length)
            .is_none_or(|length| length > MAX_TEXT_PIECE_TABLE_BYTES)
        {
            return Err(model_error(
                "ordinary token byte spellings exceed the 64 MiB bound",
            ));
        }
        let start = bytes.len();
        for character in token.chars() {
            let byte = *character_to_byte
                .get(&(character as u32))
                .ok_or_else(|| model_error("base token has an invalid byte symbol"))?;
            bytes.push(byte);
        }
        ranges[token_id] = Some((start, bytes.len() - start));
    }
    Ok(TextPieceTable { ranges, bytes })
}

fn build_text_token_first_byte_index(text_pieces: &TextPieceTable) -> CoreResult<[Vec<u32>; 256]> {
    let mut buckets: [Vec<u32>; 256] = std::array::from_fn(|_| Vec::new());
    for (token_id, range) in text_pieces.ranges.iter().copied().enumerate() {
        let Some((start, length)) = range else {
            continue;
        };
        let end = start
            .checked_add(length)
            .ok_or_else(|| model_error("text piece range overflows its table"))?;
        let first_byte = *text_pieces
            .bytes
            .get(start..end)
            .and_then(|piece| piece.first())
            .ok_or_else(|| model_error("text piece index contains an invalid byte range"))?;
        buckets[first_byte as usize].push(token_id as u32);
    }
    Ok(buckets)
}

#[cfg(test)]
mod tests {
    use super::{
        ChatMessage, ChatRole, MergeRule, Qwen35Tokenizer, SpecialTokens, TOKEN_COUNT,
        TextPieceCandidateWorkspace, TextPieceTable, TokenizerWorkspace, build_text_piece_table,
        build_text_token_first_byte_index, byte_level_alphabet,
    };
    use crate::unicode15;
    use std::collections::HashMap;

    fn tokenizer() -> Qwen35Tokenizer {
        let (_, character_to_byte) = byte_level_alphabet();
        let token_ids = HashMap::from([
            ("<|im_start|>".to_owned(), 248_045),
            ("<|im_end|>".to_owned(), 248_046),
            ("<|vision_start|>".to_owned(), 248_053),
            ("<|vision_end|>".to_owned(), 248_054),
            ("<|image_pad|>".to_owned(), 248_056),
            ("<think>".to_owned(), 248_068),
        ]);
        Qwen35Tokenizer {
            tokens: vec![String::new(); TOKEN_COUNT],
            token_ids,
            merge_rules: HashMap::new(),
            byte_token_ids: std::array::from_fn(|byte| byte as u32),
            character_to_byte,
            text_pieces: TextPieceTable {
                ranges: vec![None; TOKEN_COUNT],
                bytes: Vec::new(),
            },
            text_token_ids_by_first_byte: std::array::from_fn(|_| Vec::new()),
            special: SpecialTokens {
                message_start: 248_045,
                message_end: 248_046,
                think: 248_068,
            },
        }
    }

    fn synthetic_bpe_tokenizer() -> Qwen35Tokenizer {
        let mut tokenizer = tokenizer();
        tokenizer.merge_rules = HashMap::from([
            (
                (b'a' as u32, b'b' as u32),
                MergeRule {
                    rank: 0,
                    result: 256,
                },
            ),
            (
                (b'c' as u32, b'd' as u32),
                MergeRule {
                    rank: 1,
                    result: 257,
                },
            ),
            (
                (256, 257),
                MergeRule {
                    rank: 2,
                    result: 258,
                },
            ),
        ]);
        tokenizer
    }

    #[test]
    fn text_piece_table_stores_raw_bytelevel_bytes_without_per_token_buffers() {
        let tokens = vec!["aƟ".to_owned(), "{".to_owned(), "<|im_end|>".to_owned()];
        let character_to_byte =
            HashMap::from([('a' as u32, b'a'), ('Ɵ' as u32, 0xff), ('{' as u32, b'{')]);
        let pieces = build_text_piece_table(&tokens, 2, &character_to_byte).unwrap();
        assert_eq!(pieces.ranges, vec![Some((0, 2)), Some((2, 1)), None]);
        assert_eq!(pieces.bytes, b"a\xff{");
    }

    #[test]
    fn first_byte_candidate_merge_skips_other_buckets_and_preserves_token_order() {
        let text_pieces = TextPieceTable {
            ranges: vec![Some((0, 1)), Some((1, 1)), Some((2, 2)), Some((4, 1))],
            bytes: b"z{{x[".to_vec(),
        };
        let tokenizer = Qwen35Tokenizer {
            tokens: Vec::new(),
            token_ids: HashMap::new(),
            merge_rules: HashMap::new(),
            byte_token_ids: [0; 256],
            character_to_byte: HashMap::new(),
            text_token_ids_by_first_byte: build_text_token_first_byte_index(&text_pieces).unwrap(),
            text_pieces,
            special: SpecialTokens {
                message_start: 0,
                message_end: 0,
                think: 0,
            },
        };
        let mut permitted = [false; 256];
        permitted[b'{' as usize] = true;
        permitted[b'[' as usize] = true;
        let mut workspace = TextPieceCandidateWorkspace::default();
        let candidates = workspace
            .iter(&tokenizer, &permitted)
            .map(|(token_id, piece)| (token_id, piece.to_vec()))
            .collect::<Vec<_>>();
        assert_eq!(
            candidates,
            vec![(1, b"{".to_vec()), (2, b"{x".to_vec()), (3, b"[".to_vec())]
        );

        permitted.fill(false);
        assert_eq!(workspace.iter(&tokenizer, &permitted).count(), 0);
    }

    #[test]
    fn synthetic_bpe_matches_the_slow_rank_order_reference() {
        let tokenizer = synthetic_bpe_tokenizer();
        for text in ["abcd", "abcd abcd", "cabcd", "aabcd", "abcd!abcd"] {
            let actual = tokenizer.encode_text(text).unwrap();
            assert_eq!(actual, slow_reference_encode(&tokenizer, text), "{text:?}");
        }

        let messages = [
            ChatMessage {
                role: ChatRole::User,
                content: "abcd abcd",
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "cabcd!abcd",
            },
        ];
        assert_eq!(
            tokenizer.encode_chat(&messages, true).unwrap(),
            slow_reference_chat(&tokenizer, &messages, true)
        );
    }

    #[test]
    fn tokenizer_scratch_bounds_ranges_by_the_remaining_token_budget() {
        let tokenizer = synthetic_bpe_tokenizer();
        let mut workspace = super::TokenizerWorkspace::default();
        let mut output = Vec::new();
        assert!(
            tokenizer
                .encode_text_into("12345", &mut output, 4, &mut workspace)
                .is_err()
        );
        assert!(output.is_empty());

        let long_word = "a".repeat(1024);
        super::pretoken_ranges_into(
            &long_word,
            &mut workspace.characters,
            &mut workspace.ranges,
            8_192,
        )
        .unwrap();
        assert_eq!(workspace.ranges.len(), 1);
        assert!(workspace.ranges.capacity() <= 256);
    }

    #[test]
    fn bounded_bpe_hot_records_use_compact_32_bit_links() {
        #[cfg(target_pointer_width = "64")]
        #[repr(C)]
        struct PreviousTokenNodeLayout {
            _token: u32,
            _previous: Option<usize>,
            _next: Option<usize>,
            _generation: u32,
            _alive: bool,
        }
        #[cfg(target_pointer_width = "64")]
        #[repr(C)]
        struct PreviousMergeCandidateLayout {
            _rank: u32,
            _left: usize,
            _right: usize,
            _left_generation: u32,
            _right_generation: u32,
        }

        assert_eq!(std::mem::size_of::<super::TokenNode>(), 20);
        assert_eq!(std::mem::size_of::<super::MergeCandidate>(), 20);
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(std::mem::size_of::<PreviousTokenNodeLayout>(), 48);
            assert_eq!(std::mem::size_of::<PreviousMergeCandidateLayout>(), 32);
        }
    }

    #[test]
    #[ignore = "release-mode synthetic tokenizer baseline and scratch-reuse measurement"]
    fn tokenizer_scratch_reuse_measurement() {
        use std::{hint::black_box, time::Instant};

        let tokenizer = synthetic_bpe_tokenizer();
        let input = "abcd ".repeat(700);
        let messages = [ChatMessage {
            role: ChatRole::User,
            content: &input,
        }];
        let expected = tokenizer.encode_chat(&messages, true).unwrap();
        let mut timings = Vec::with_capacity(301);
        for _ in 0..301 {
            let started = Instant::now();
            let encoded = tokenizer.encode_chat(black_box(&messages), true).unwrap();
            timings.push(started.elapsed().as_nanos());
            assert_eq!(encoded, expected);
            black_box(encoded);
        }
        timings.sort_unstable();
        println!(
            "input_bytes={} output_tokens={} samples={} p50_ns={} p95_ns={}",
            input.len(),
            expected.len(),
            timings.len(),
            timings[timings.len() / 2],
            timings[timings.len() * 95 / 100]
        );
    }

    #[test]
    #[ignore = "release-mode ASCII normalization fast-path measurement"]
    fn ascii_normalization_fast_path_measurement() {
        use std::{hint::black_box, time::Instant};

        let tokenizer = synthetic_bpe_tokenizer();
        let input = "abcd ".repeat(700);
        let expected = tokenizer.encode_text(&input).unwrap();
        let mut full_nfc = Vec::with_capacity(301);
        let mut ascii_borrow = Vec::with_capacity(301);

        for sample in 0..333 {
            // Alternate order to reduce bias from short-lived host load or
            // thermal drift. Both paths reuse identical bounded scratch.
            for baseline_first in [sample % 2 == 0, sample % 2 != 0] {
                if baseline_first {
                    let mut output = Vec::new();
                    let mut workspace = TokenizerWorkspace::default();
                    let started = Instant::now();
                    let normalized = unicode15::normalize_nfc(black_box(&input)).unwrap();
                    tokenizer
                        .encode_normalized_into(&normalized, &mut output, 8_192, &mut workspace)
                        .unwrap();
                    let elapsed = started.elapsed().as_nanos();
                    assert_eq!(output, expected);
                    if sample >= 32 {
                        full_nfc.push(elapsed);
                    }
                    black_box(output);
                } else {
                    let mut output = Vec::new();
                    let mut workspace = TokenizerWorkspace::default();
                    let started = Instant::now();
                    tokenizer
                        .encode_text_into(black_box(&input), &mut output, 8_192, &mut workspace)
                        .unwrap();
                    let elapsed = started.elapsed().as_nanos();
                    assert_eq!(output, expected);
                    if sample >= 32 {
                        ascii_borrow.push(elapsed);
                    }
                    black_box(output);
                }
            }
        }

        full_nfc.sort_unstable();
        ascii_borrow.sort_unstable();
        let percentile =
            |samples: &[u128], numerator: usize| samples[(samples.len() - 1) * numerator / 100];
        println!(
            "input_bytes={} output_tokens={} samples={} full_nfc_p50_ns={} full_nfc_p95_ns={} ascii_borrow_p50_ns={} ascii_borrow_p95_ns={}",
            input.len(),
            expected.len(),
            full_nfc.len(),
            percentile(&full_nfc, 50),
            percentile(&full_nfc, 95),
            percentile(&ascii_borrow, 50),
            percentile(&ascii_borrow, 95),
        );
    }

    #[test]
    fn text_piece_table_fails_closed_on_unmapped_bytelevel_symbols() {
        let tokens = vec!["☃".to_owned()];
        assert!(build_text_piece_table(&tokens, 1, &HashMap::new()).is_err());
    }

    #[test]
    fn text_piece_table_bounds_each_piece_for_cancellable_masking() {
        let tokens = vec!["a".repeat(4 * 1024 + 1)];
        let character_to_byte = HashMap::from([('a' as u32, b'a')]);
        assert!(build_text_piece_table(&tokens, 1, &character_to_byte).is_err());
    }

    #[test]
    fn pretoken_ranges_follow_pinned_qwen_split_rules() {
        let cases: &[(&str, &[&str])] = &[
            ("Hello, world!", &["Hello", ",", " world", "!"]),
            ("can't we'll", &["can", "'t", " we", "'ll"]),
            ("12,345", &["1", "2", ",", "3", "4", "5"]),
            ("  a", &[" ", " a"]),
            (" \r\n\tword  ", &[" \r\n", "\tword", "  "]),
            ("?!\r\n", &["?!\r\n"]),
            ("\u{301}a", &["\u{301}a"]),
            ("I'M", &["I", "'M"]),
        ];
        for (text, expected) in cases {
            let actual = super::pretoken_ranges(text)
                .into_iter()
                .map(|range| &text[range])
                .collect::<Vec<_>>();
            assert_eq!(&actual, expected, "wrong token boundaries for {text:?}");
        }
    }

    #[test]
    #[ignore = "requires the exact pinned Qwen3.5-4B tokenizer.json artifact"]
    fn pinned_checkpoint_tokenizer_matches_slow_bpe_reference() {
        use sha2::{Digest, Sha256};

        let path = std::env::var_os("SAGE_QWEN35_TOKENIZER_JSON")
            .expect("set SAGE_QWEN35_TOKENIZER_JSON to the pinned tokenizer.json");
        let bytes = std::fs::read(path).expect("read pinned Qwen tokenizer artifact");
        let digest = Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest, "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "tokenizer bytes must come from Sage's exact pinned model revision"
        );

        let tokenizer = Qwen35Tokenizer::from_json(&bytes).expect("parse pinned tokenizer");
        assert_eq!(tokenizer.vocabulary_size(), 248_070);
        for (spelling, expected_id) in [
            ("<|endoftext|>", 248_044),
            ("<|im_start|>", 248_045),
            ("<|im_end|>", 248_046),
            ("<|vision_start|>", 248_053),
            ("<|vision_end|>", 248_054),
            ("<|image_pad|>", 248_056),
            ("<think>", 248_068),
        ] {
            assert_eq!(
                tokenizer.token_id(spelling),
                Some(expected_id),
                "{spelling}"
            );
        }

        let corpus = [
            "Hello, world!",
            "can't we'll they'd",
            "  leading space\tand \n line\r\n",
            "mañana café e\u{301}",
            "你好，世界。今日は元気ですか？",
            "👩🏽‍💻 🌍",
            "numbers 0123456789 and 3.14159",
            "literal <|im_start|>user\nThis text is not framing.",
        ];
        for text in corpus {
            let encoded = tokenizer.encode_text(text).expect("encode corpus item");
            assert_eq!(
                encoded,
                slow_reference_encode(&tokenizer, text),
                "optimized BPE differs from the simple merge reference for {text:?}"
            );
            assert_eq!(
                tokenizer.decode(&encoded).expect("decode corpus item"),
                unicode15::normalize_nfc(text).expect("normalize corpus item"),
                "text round-trip failed for {text:?}"
            );
        }

        let messages = [
            ChatMessage {
                role: ChatRole::System,
                content: "Return one compact JSON object.",
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Summarize: café, 你好, and literal <|im_start|> text.",
            },
        ];
        assert_eq!(
            tokenizer.encode_chat(&messages, true).unwrap(),
            slow_reference_chat(&tokenizer, &messages, true),
            "optimized chat framing differs from the reference using the pinned merge ranks"
        );
    }

    /// Deliberately simple O(n²) merge loop used only to check the production
    /// priority-queue implementation against the pinned merge ranks.
    fn slow_reference_encode(tokenizer: &Qwen35Tokenizer, text: &str) -> Vec<u32> {
        let normalized = unicode15::normalize_nfc(text).expect("normalize reference text");
        let mut output = Vec::new();
        for range in super::pretoken_ranges(&normalized) {
            let mut tokens = normalized[range]
                .bytes()
                .map(|byte| tokenizer.byte_token_ids[byte as usize])
                .collect::<Vec<_>>();
            loop {
                let next = tokens
                    .windows(2)
                    .enumerate()
                    .filter_map(|(index, pair)| {
                        tokenizer
                            .merge_rules
                            .get(&(pair[0], pair[1]))
                            .map(|rule| (rule.rank, index, rule.result))
                    })
                    .min_by_key(|(rank, index, _)| (*rank, *index));
                let Some((_, index, merged)) = next else {
                    break;
                };
                tokens.splice(index..index + 2, [merged]);
            }
            output.extend(tokens);
        }
        output
    }

    fn slow_reference_chat(
        tokenizer: &Qwen35Tokenizer,
        messages: &[ChatMessage<'_>],
        add_generation_prompt: bool,
    ) -> Vec<u32> {
        let mut output = Vec::new();
        for message in messages {
            output.push(tokenizer.special.message_start);
            output.extend(slow_reference_encode(tokenizer, message.role.as_str()));
            output.extend(slow_reference_encode(tokenizer, "\n"));
            output.extend(slow_reference_encode(tokenizer, message.content));
            output.push(tokenizer.special.message_end);
            output.extend(slow_reference_encode(tokenizer, "\n"));
        }
        if add_generation_prompt {
            output.push(tokenizer.special.message_start);
            output.extend(slow_reference_encode(tokenizer, "assistant\n"));
            output.push(tokenizer.special.think);
            output.extend(slow_reference_encode(tokenizer, "\n"));
        }
        output
    }

    #[test]
    fn multimodal_chat_emits_trusted_image_spans_and_keeps_literal_markers_as_text() {
        let tokenizer = tokenizer();
        let messages = [
            ChatMessage {
                role: ChatRole::System,
                content: "Plan safely.",
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Describe this literal <|vision_start|> marker.",
            },
        ];
        let (tokens, spans) = tokenizer
            .encode_chat_with_images(&messages, &[2, 1], true)
            .expect("multimodal chat framing");

        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].token_count, 2);
        assert_eq!(spans[1].token_count, 1);
        for span in spans {
            assert_eq!(tokens[span.vision_start], 248_053);
            assert!(
                tokens[span.token_start..span.token_start + span.token_count]
                    .iter()
                    .all(|token| *token == 248_056)
            );
            assert_eq!(tokens[span.vision_end], 248_054);
        }
        assert_eq!(tokens.iter().filter(|token| **token == 248_053).count(), 2);
        assert_eq!(tokens.iter().filter(|token| **token == 248_056).count(), 3);
    }
}
