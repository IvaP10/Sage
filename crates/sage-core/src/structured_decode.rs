//! First-party bounded JSON prefix grammar for constrained local generation.
//!
//! This validates every candidate against RFC 8259 syntax and, for planner
//! turns, an incremental automaton compiled from Sage's closed output schema.
//! Completed turns still pass the independent planner validator; model output
//! remains untrusted and never grants execution authority.

use std::cell::RefCell;
use std::collections::BTreeSet;

use serde_json::Value;

use crate::planner_schema::PlannerSchemaPrefix;
use crate::qwen_tokenizer::{Qwen35Tokenizer, TextPieceCandidateWorkspace};
use crate::qwen35::{
    Qwen35EmbeddedPrompt, Qwen35TextDecoder, generate_greedy, generate_greedy_with_embedded_prompt,
};
use crate::{CoreError, CoreResult};

const MAX_JSON_BYTES: usize = crate::model::MAX_TURN_BYTES;
const MAX_JSON_NESTING: usize = 64;
const TOKEN_MASK_CANCEL_CHECK_INTERVAL: usize = 256;

enum PlannerDecoderPrompt<'a> {
    Text(&'a [u32]),
    Embedded(&'a Qwen35EmbeddedPrompt<'a>),
}

#[derive(Clone, Copy)]
enum CompleteJsonRule {
    AnyJson,
    PlannerTurn,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlannerGenerationOptions {
    pub end_of_turn_token_id: u32,
    pub maximum_new_tokens: usize,
    pub task_id: uuid::Uuid,
}

struct JsonGenerationPolicy {
    end_of_turn_token_id: u32,
    maximum_new_tokens: usize,
    enforce_planner_schema: bool,
    complete_rule: CompleteJsonRule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootState {
    Value,
    AfterValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectState {
    KeyOrEnd,
    Key,
    Colon,
    Value,
    AfterValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayState {
    ValueOrEnd,
    Value,
    AfterValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    Object(ObjectState),
    Array(ArrayState),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberState {
    Sign,
    Zero,
    Integer,
    DecimalPoint,
    Fraction,
    Exponent,
    ExponentSign,
    ExponentDigits,
}

impl NumberState {
    fn can_end(self) -> bool {
        matches!(
            self,
            Self::Zero | Self::Integer | Self::Fraction | Self::ExponentDigits
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StringMode {
    Normal,
    Escape,
    Unicode { remaining: u8, value: u16 },
    NeedLowSlash,
    NeedLowU,
    UnicodeLow { remaining: u8, value: u16 },
}

#[derive(Debug, Clone, Copy)]
struct JsonString {
    is_key: bool,
    mode: StringMode,
    utf8_remaining: u8,
    utf8_next_min: u8,
    utf8_next_max: u8,
}

impl JsonString {
    fn new(is_key: bool) -> Self {
        Self {
            is_key,
            mode: StringMode::Normal,
            utf8_remaining: 0,
            utf8_next_min: 0x80,
            utf8_next_max: 0xbf,
        }
    }

    /// Return true when this byte closes the string. Invalid or overlong UTF-8,
    /// invalid escapes, and unpaired UTF-16 surrogates are rejected in-place.
    fn push(&mut self, byte: u8) -> CoreResult<bool> {
        match self.mode {
            StringMode::Normal => {
                if self.utf8_remaining > 0 {
                    if !(self.utf8_next_min..=self.utf8_next_max).contains(&byte) {
                        return Err(json_error("invalid UTF-8 continuation in JSON string"));
                    }
                    self.utf8_remaining -= 1;
                    self.utf8_next_min = 0x80;
                    self.utf8_next_max = 0xbf;
                    return Ok(false);
                }
                match byte {
                    b'"' => Ok(true),
                    b'\\' => {
                        self.mode = StringMode::Escape;
                        Ok(false)
                    }
                    0x00..=0x1f => Err(json_error("unescaped control byte in JSON string")),
                    0x20..=0x7f => Ok(false),
                    0xc2..=0xdf => {
                        self.utf8_remaining = 1;
                        Ok(false)
                    }
                    0xe0 => {
                        self.utf8_remaining = 2;
                        self.utf8_next_min = 0xa0;
                        Ok(false)
                    }
                    0xe1..=0xec | 0xee..=0xef => {
                        self.utf8_remaining = 2;
                        Ok(false)
                    }
                    0xed => {
                        self.utf8_remaining = 2;
                        self.utf8_next_max = 0x9f;
                        Ok(false)
                    }
                    0xf0 => {
                        self.utf8_remaining = 3;
                        self.utf8_next_min = 0x90;
                        Ok(false)
                    }
                    0xf1..=0xf3 => {
                        self.utf8_remaining = 3;
                        Ok(false)
                    }
                    0xf4 => {
                        self.utf8_remaining = 3;
                        self.utf8_next_max = 0x8f;
                        Ok(false)
                    }
                    _ => Err(json_error("invalid UTF-8 lead byte in JSON string")),
                }
            }
            StringMode::Escape => {
                match byte {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                        self.mode = StringMode::Normal;
                    }
                    b'u' => {
                        self.mode = StringMode::Unicode {
                            remaining: 4,
                            value: 0,
                        }
                    }
                    _ => return Err(json_error("invalid JSON string escape")),
                }
                Ok(false)
            }
            StringMode::Unicode { remaining, value } => {
                let digit = hex_value(byte).ok_or_else(|| json_error("invalid Unicode escape"))?;
                let value = (value << 4) | u16::from(digit);
                if remaining > 1 {
                    self.mode = StringMode::Unicode {
                        remaining: remaining - 1,
                        value,
                    };
                } else if (0xd800..=0xdbff).contains(&value) {
                    self.mode = StringMode::NeedLowSlash;
                } else if (0xdc00..=0xdfff).contains(&value) {
                    return Err(json_error("unpaired low surrogate in JSON string"));
                } else {
                    self.mode = StringMode::Normal;
                }
                Ok(false)
            }
            StringMode::NeedLowSlash => {
                if byte != b'\\' {
                    return Err(json_error("high surrogate is missing its low surrogate"));
                }
                self.mode = StringMode::NeedLowU;
                Ok(false)
            }
            StringMode::NeedLowU => {
                if byte != b'u' {
                    return Err(json_error("high surrogate is missing its Unicode escape"));
                }
                self.mode = StringMode::UnicodeLow {
                    remaining: 4,
                    value: 0,
                };
                Ok(false)
            }
            StringMode::UnicodeLow { remaining, value } => {
                let digit = hex_value(byte).ok_or_else(|| json_error("invalid low surrogate"))?;
                let value = (value << 4) | u16::from(digit);
                if remaining > 1 {
                    self.mode = StringMode::UnicodeLow {
                        remaining: remaining - 1,
                        value,
                    };
                } else if (0xdc00..=0xdfff).contains(&value) {
                    self.mode = StringMode::Normal;
                } else {
                    return Err(json_error("invalid low surrogate code unit"));
                }
                Ok(false)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Lexical {
    String(JsonString),
    Number(NumberState),
    Literal {
        expected: &'static [u8],
        matched: usize,
    },
}

/// Incremental recognizer for any single RFC 8259 JSON value. A non-error
/// prefix is always extendable to valid JSON; completed output is checked by
/// serde_json before an end-of-turn token can be selected.
#[derive(Debug, Clone)]
pub struct JsonPrefixState {
    root: RootState,
    stack: Vec<Container>,
    lexical: Option<Lexical>,
    bytes_seen: usize,
    invalid: bool,
}

impl Default for JsonPrefixState {
    fn default() -> Self {
        Self {
            root: RootState::Value,
            stack: Vec::new(),
            lexical: None,
            bytes_seen: 0,
            invalid: false,
        }
    }
}

impl JsonPrefixState {
    pub fn push(&mut self, bytes: &[u8]) -> CoreResult<()> {
        if self.invalid
            || self
                .bytes_seen
                .checked_add(bytes.len())
                .is_none_or(|length| length > MAX_JSON_BYTES)
        {
            self.invalid = true;
            return Err(json_error("JSON prefix exceeds Sage's 256 KiB bound"));
        }
        self.bytes_seen += bytes.len();
        for &byte in bytes {
            if let Err(error) = self.push_byte(byte) {
                self.invalid = true;
                return Err(error);
            }
        }
        Ok(())
    }

    pub fn is_viable(&self) -> bool {
        !self.invalid
    }

    /// True once one JSON value has been completely recognized. An unfinished
    /// but valid number exponent remains viable but is not a completed value.
    pub fn is_complete(&self) -> bool {
        if self.invalid || !self.stack.is_empty() || self.root != RootState::AfterValue {
            return false;
        }
        match self.lexical {
            None => true,
            Some(Lexical::Number(state)) => state.can_end(),
            Some(Lexical::String(_) | Lexical::Literal { .. }) => false,
        }
    }

    /// Fast rejection for token candidates before cloning the parser state.
    /// This keeps delimiter, key, and value-position masks inexpensive even
    /// when the vocabulary contains hundreds of thousands of pieces.
    fn could_accept_first_byte(&self, byte: u8) -> bool {
        if self.invalid || self.bytes_seen >= MAX_JSON_BYTES {
            return false;
        }
        match self.lexical {
            Some(Lexical::String(mut string)) => string.push(byte).is_ok(),
            Some(Lexical::Literal { expected, matched }) => {
                expected.get(matched).copied() == Some(byte)
            }
            Some(Lexical::Number(mut state)) => match push_number(&mut state, byte) {
                Ok(true) => true,
                Ok(false) => self.normal_byte_may_be_valid(byte),
                Err(_) => false,
            },
            None => self.normal_byte_may_be_valid(byte),
        }
    }

    fn possible_first_bytes(&self) -> [bool; 256] {
        std::array::from_fn(|byte| self.could_accept_first_byte(byte as u8))
    }

    fn normal_byte_may_be_valid(&self, byte: u8) -> bool {
        if is_json_whitespace(byte) {
            return true;
        }
        let starts_value = is_value_start(byte);
        match self.stack.last().copied() {
            None => self.root == RootState::Value && starts_value,
            Some(Container::Object(state)) => match state {
                ObjectState::KeyOrEnd => matches!(byte, b'"' | b'}'),
                ObjectState::Key => byte == b'"',
                ObjectState::Colon => byte == b':',
                ObjectState::Value => starts_value,
                ObjectState::AfterValue => matches!(byte, b',' | b'}'),
            },
            Some(Container::Array(state)) => match state {
                ArrayState::ValueOrEnd => byte == b']' || starts_value,
                ArrayState::Value => starts_value,
                ArrayState::AfterValue => matches!(byte, b',' | b']'),
            },
        }
    }

    fn push_byte(&mut self, byte: u8) -> CoreResult<()> {
        let mut reprocess = true;
        while reprocess {
            reprocess = false;
            match self.lexical.take() {
                Some(Lexical::String(mut string)) => {
                    if string.push(byte)? {
                        if string.is_key {
                            match self.stack.last_mut() {
                                Some(Container::Object(state))
                                    if matches!(
                                        *state,
                                        ObjectState::KeyOrEnd | ObjectState::Key
                                    ) =>
                                {
                                    *state = ObjectState::Colon;
                                }
                                _ => return Err(json_error("JSON object key is out of place")),
                            }
                        }
                    } else {
                        self.lexical = Some(Lexical::String(string));
                    }
                }
                Some(Lexical::Literal { expected, matched }) => {
                    if expected.get(matched).copied() != Some(byte) {
                        return Err(json_error("invalid JSON literal"));
                    }
                    let matched = matched + 1;
                    if matched < expected.len() {
                        self.lexical = Some(Lexical::Literal { expected, matched });
                    }
                }
                Some(Lexical::Number(mut state)) => {
                    if push_number(&mut state, byte)? {
                        self.lexical = Some(Lexical::Number(state));
                    } else {
                        reprocess = true;
                    }
                }
                None => self.push_normal(byte)?,
            }
        }
        Ok(())
    }

    fn push_normal(&mut self, byte: u8) -> CoreResult<()> {
        if is_json_whitespace(byte) {
            return Ok(());
        }
        match self.stack.last().copied() {
            None => match self.root {
                RootState::Value => self.begin_value(byte),
                RootState::AfterValue => Err(json_error("trailing bytes after JSON value")),
            },
            Some(Container::Object(state)) => match state {
                ObjectState::KeyOrEnd => {
                    if byte == b'}' {
                        self.close_container(true)
                    } else if byte == b'"' {
                        self.lexical = Some(Lexical::String(JsonString::new(true)));
                        Ok(())
                    } else {
                        Err(json_error("expected an object key or closing brace"))
                    }
                }
                ObjectState::Key => {
                    if byte == b'"' {
                        self.lexical = Some(Lexical::String(JsonString::new(true)));
                        Ok(())
                    } else {
                        Err(json_error("expected an object key after comma"))
                    }
                }
                ObjectState::Colon => {
                    if byte == b':' {
                        if let Some(Container::Object(state)) = self.stack.last_mut() {
                            *state = ObjectState::Value;
                        }
                        Ok(())
                    } else {
                        Err(json_error("expected a colon after object key"))
                    }
                }
                ObjectState::Value => self.begin_value(byte),
                ObjectState::AfterValue => match byte {
                    b',' => {
                        if let Some(Container::Object(state)) = self.stack.last_mut() {
                            *state = ObjectState::Key;
                        }
                        Ok(())
                    }
                    b'}' => self.close_container(true),
                    _ => Err(json_error("expected comma or closing brace")),
                },
            },
            Some(Container::Array(state)) => match state {
                ArrayState::ValueOrEnd if byte == b']' => self.close_container(false),
                ArrayState::ValueOrEnd | ArrayState::Value => self.begin_value(byte),
                ArrayState::AfterValue => match byte {
                    b',' => {
                        if let Some(Container::Array(state)) = self.stack.last_mut() {
                            *state = ArrayState::Value;
                        }
                        Ok(())
                    }
                    b']' => self.close_container(false),
                    _ => Err(json_error("expected comma or closing bracket")),
                },
            },
        }
    }

    fn begin_value(&mut self, byte: u8) -> CoreResult<()> {
        self.mark_value_started()?;
        match byte {
            b'{' => self.push_container(Container::Object(ObjectState::KeyOrEnd)),
            b'[' => self.push_container(Container::Array(ArrayState::ValueOrEnd)),
            b'"' => {
                self.lexical = Some(Lexical::String(JsonString::new(false)));
                Ok(())
            }
            b't' => {
                self.lexical = Some(Lexical::Literal {
                    expected: b"true",
                    matched: 1,
                });
                Ok(())
            }
            b'f' => {
                self.lexical = Some(Lexical::Literal {
                    expected: b"false",
                    matched: 1,
                });
                Ok(())
            }
            b'n' => {
                self.lexical = Some(Lexical::Literal {
                    expected: b"null",
                    matched: 1,
                });
                Ok(())
            }
            b'-' => {
                self.lexical = Some(Lexical::Number(NumberState::Sign));
                Ok(())
            }
            b'0' => {
                self.lexical = Some(Lexical::Number(NumberState::Zero));
                Ok(())
            }
            b'1'..=b'9' => {
                self.lexical = Some(Lexical::Number(NumberState::Integer));
                Ok(())
            }
            _ => Err(json_error("expected a JSON value")),
        }
    }

    fn mark_value_started(&mut self) -> CoreResult<()> {
        match self.stack.last_mut() {
            Some(Container::Object(state)) if *state == ObjectState::Value => {
                *state = ObjectState::AfterValue;
            }
            Some(Container::Array(state))
                if matches!(*state, ArrayState::Value | ArrayState::ValueOrEnd) =>
            {
                *state = ArrayState::AfterValue;
            }
            Some(_) => return Err(json_error("JSON value is out of place")),
            None if self.root == RootState::Value => self.root = RootState::AfterValue,
            None => return Err(json_error("multiple top-level JSON values")),
        }
        Ok(())
    }

    fn push_container(&mut self, container: Container) -> CoreResult<()> {
        if self.stack.len() >= MAX_JSON_NESTING {
            return Err(json_error("JSON nesting exceeds Sage's 64-level bound"));
        }
        self.stack.push(container);
        Ok(())
    }

    fn close_container(&mut self, object: bool) -> CoreResult<()> {
        match self.stack.pop() {
            Some(Container::Object(_)) if object => Ok(()),
            Some(Container::Array(_)) if !object => Ok(()),
            _ => Err(json_error("mismatched JSON container closure")),
        }
    }
}

fn push_number(state: &mut NumberState, byte: u8) -> CoreResult<bool> {
    match *state {
        NumberState::Sign => match byte {
            b'0' => *state = NumberState::Zero,
            b'1'..=b'9' => *state = NumberState::Integer,
            _ => return Err(json_error("minus sign is not followed by a digit")),
        },
        NumberState::Zero => match byte {
            b'.' => *state = NumberState::DecimalPoint,
            b'e' | b'E' => *state = NumberState::Exponent,
            b'0'..=b'9' => return Err(json_error("leading zero in JSON number")),
            _ => return Ok(false),
        },
        NumberState::Integer => match byte {
            b'0'..=b'9' => {}
            b'.' => *state = NumberState::DecimalPoint,
            b'e' | b'E' => *state = NumberState::Exponent,
            _ => return Ok(false),
        },
        NumberState::DecimalPoint => match byte {
            b'0'..=b'9' => *state = NumberState::Fraction,
            _ => return Err(json_error("decimal point is not followed by a digit")),
        },
        NumberState::Fraction => match byte {
            b'0'..=b'9' => {}
            b'e' | b'E' => *state = NumberState::Exponent,
            _ => return Ok(false),
        },
        NumberState::Exponent => match byte {
            b'+' | b'-' => *state = NumberState::ExponentSign,
            b'0'..=b'9' => *state = NumberState::ExponentDigits,
            _ => return Err(json_error("exponent marker is not followed by digits")),
        },
        NumberState::ExponentSign => match byte {
            b'0'..=b'9' => *state = NumberState::ExponentDigits,
            _ => return Err(json_error("exponent sign is not followed by digits")),
        },
        NumberState::ExponentDigits => match byte {
            b'0'..=b'9' => {}
            _ => return Ok(false),
        },
    }
    Ok(true)
}

fn is_json_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_value_start(byte: u8) -> bool {
    matches!(
        byte,
        b'{' | b'[' | b'"' | b't' | b'f' | b'n' | b'-' | b'0'..=b'9'
    )
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn mask_json_token_pieces<'a, I, C>(
    grammar: &JsonPrefixState,
    pieces: I,
    end_token: Option<u32>,
    cancelled: C,
    allowed: &mut Vec<u32>,
) -> CoreResult<()>
where
    I: IntoIterator<Item = (u32, &'a [u8])>,
    C: FnMut() -> bool,
{
    mask_token_pieces(grammar, None, pieces, end_token, cancelled, allowed)
}

fn mask_planner_token_pieces<'a, I, C>(
    grammar: &JsonPrefixState,
    planner_schema: &PlannerSchemaPrefix,
    pieces: I,
    end_token: Option<u32>,
    cancelled: C,
    allowed: &mut Vec<u32>,
) -> CoreResult<()>
where
    I: IntoIterator<Item = (u32, &'a [u8])>,
    C: FnMut() -> bool,
{
    mask_token_pieces(
        grammar,
        Some(planner_schema),
        pieces,
        end_token,
        cancelled,
        allowed,
    )
}

fn mask_token_pieces<'a, I, C>(
    grammar: &JsonPrefixState,
    planner_schema: Option<&PlannerSchemaPrefix>,
    pieces: I,
    end_token: Option<u32>,
    mut cancelled: C,
    allowed: &mut Vec<u32>,
) -> CoreResult<()>
where
    I: IntoIterator<Item = (u32, &'a [u8])>,
    C: FnMut() -> bool,
{
    allowed.clear();
    let mut previous_token = None;
    for (index, (token_id, piece)) in pieces.into_iter().enumerate() {
        if index.is_multiple_of(TOKEN_MASK_CANCEL_CHECK_INTERVAL) && cancelled() {
            return Err(CoreError::Cancelled);
        }
        if previous_token.is_some_and(|previous| token_id < previous) {
            return Err(json_error("token mask input is not ordered by token ID"));
        }
        previous_token = Some(token_id);
        if piece
            .first()
            .is_none_or(|byte| !grammar.could_accept_first_byte(*byte))
        {
            continue;
        }
        let mut candidate = grammar.clone();
        if candidate.push(piece).is_err() || !candidate.is_viable() {
            continue;
        }
        if let Some(planner_schema) = planner_schema {
            let mut candidate_schema = planner_schema.clone();
            if candidate_schema.push(piece).is_err() || !candidate_schema.is_viable() {
                continue;
            }
        }
        if allowed.last().copied() != Some(token_id) {
            allowed.push(token_id);
        }
    }
    if grammar.is_complete()
        && planner_schema.is_none_or(PlannerSchemaPrefix::is_complete)
        && let Some(end_token) = end_token
    {
        match allowed.binary_search(&end_token) {
            Ok(_) => {}
            Err(index) => allowed.insert(index, end_token),
        }
    }
    if allowed.is_empty() {
        return Err(json_error("Structured decoding has no valid next token"));
    }
    Ok(())
}

/// Generate one bounded JSON value using Sage's scalar Qwen decoder. Only
/// ordinary tokenizer pieces that preserve JSON-prefix validity are eligible;
/// the sole stop token is enabled after a complete, independently parsed JSON
/// value. The caller must still validate its application-specific schema.
pub fn generate_json_greedy<C>(
    decoder: &mut Qwen35TextDecoder,
    tokenizer: &Qwen35Tokenizer,
    prompt_token_ids: &[u32],
    end_of_turn_token_id: u32,
    maximum_new_tokens: usize,
    cancelled: C,
) -> CoreResult<String>
where
    C: FnMut() -> bool,
{
    generate_json_greedy_validated(
        decoder,
        tokenizer,
        PlannerDecoderPrompt::Text(prompt_token_ids),
        JsonGenerationPolicy {
            end_of_turn_token_id,
            maximum_new_tokens,
            enforce_planner_schema: false,
            complete_rule: CompleteJsonRule::AnyJson,
        },
        |_| {},
        cancelled,
    )
}

/// Generate a complete Sage planner turn. The prefix mask prunes tokens that
/// cannot satisfy the compiled planner schema; the end token additionally
/// requires `parse_turn`'s independent schema and graph checks.
pub fn generate_planner_turn_greedy<C>(
    decoder: &mut Qwen35TextDecoder,
    tokenizer: &Qwen35Tokenizer,
    prompt_token_ids: &[u32],
    end_of_turn_token_id: u32,
    maximum_new_tokens: usize,
    task_id: uuid::Uuid,
    cancelled: C,
) -> CoreResult<crate::model::ModelTurn>
where
    C: FnMut() -> bool,
{
    generate_planner_turn_greedy_with_answer_updates(
        decoder,
        tokenizer,
        prompt_token_ids,
        PlannerGenerationOptions {
            end_of_turn_token_id,
            maximum_new_tokens,
            task_id,
        },
        |_| {},
        cancelled,
    )
}

/// Generate a closed-schema planner turn and publish only cumulative root
/// `answer` prefixes. Tool payload fields and malformed/incomplete UTF-8 are
/// never surfaced; every completed turn still passes the ordinary validator.
pub(crate) fn generate_planner_turn_greedy_with_answer_updates<C, U>(
    decoder: &mut Qwen35TextDecoder,
    tokenizer: &Qwen35Tokenizer,
    prompt_token_ids: &[u32],
    options: PlannerGenerationOptions,
    mut answer_update: U,
    cancelled: C,
) -> CoreResult<crate::model::ModelTurn>
where
    C: FnMut() -> bool,
    U: FnMut(&str),
{
    if options.task_id.is_nil() {
        return Err(json_error(
            "Planner generation requires a valid task identity",
        ));
    }
    let json = generate_json_greedy_validated(
        decoder,
        tokenizer,
        PlannerDecoderPrompt::Text(prompt_token_ids),
        JsonGenerationPolicy {
            end_of_turn_token_id: options.end_of_turn_token_id,
            maximum_new_tokens: options.maximum_new_tokens,
            enforce_planner_schema: true,
            complete_rule: CompleteJsonRule::PlannerTurn,
        },
        &mut answer_update,
        cancelled,
    )?;
    crate::model::parse_turn(&json, options.task_id)
}

/// Generate a closed-schema planner turn from text and already encoded image
/// embeddings. The embeddings remain untrusted evidence and do not grant any
/// capability; target resolution, policy, permission and verification stay in
/// the normal execution path.
pub fn generate_planner_turn_with_embedded_prompt<C>(
    decoder: &mut Qwen35TextDecoder,
    tokenizer: &Qwen35Tokenizer,
    prompt: &Qwen35EmbeddedPrompt<'_>,
    end_of_turn_token_id: u32,
    maximum_new_tokens: usize,
    task_id: uuid::Uuid,
    cancelled: C,
) -> CoreResult<crate::model::ModelTurn>
where
    C: FnMut() -> bool,
{
    if task_id.is_nil() {
        return Err(json_error(
            "Planner generation requires a valid task identity",
        ));
    }
    let json = generate_json_greedy_validated(
        decoder,
        tokenizer,
        PlannerDecoderPrompt::Embedded(prompt),
        JsonGenerationPolicy {
            end_of_turn_token_id,
            maximum_new_tokens,
            enforce_planner_schema: true,
            complete_rule: CompleteJsonRule::PlannerTurn,
        },
        |_| {},
        cancelled,
    )?;
    crate::model::parse_turn(&json, task_id)
}

fn generate_json_greedy_validated<C, U>(
    decoder: &mut Qwen35TextDecoder,
    tokenizer: &Qwen35Tokenizer,
    prompt: PlannerDecoderPrompt<'_>,
    policy: JsonGenerationPolicy,
    mut answer_update: U,
    cancelled: C,
) -> CoreResult<String>
where
    C: FnMut() -> bool,
    U: FnMut(&str),
{
    if decoder.vocabulary_size() != tokenizer.vocabulary_size()
        || policy.end_of_turn_token_id as usize >= tokenizer.vocabulary_size()
        || tokenizer
            .text_token_bytes(policy.end_of_turn_token_id)
            .is_some()
        || tokenizer.token_id("<|im_end|>") != Some(policy.end_of_turn_token_id)
        || policy.maximum_new_tokens == 0
    {
        return Err(json_error(
            "JSON generation profile does not match tokenizer",
        ));
    }

    let stop_ids = BTreeSet::from([policy.end_of_turn_token_id]);
    let mut grammar = JsonPrefixState::default();
    let mut planner_schema = policy
        .enforce_planner_schema
        .then(PlannerSchemaPrefix::new)
        .transpose()?;
    let mut consumed = 0usize;
    let mut output_bytes = Vec::new();
    let mut last_preview = String::new();
    let mut last_preview_scan_bytes = 0usize;
    let cancellation = RefCell::new(cancelled);
    let mut candidate_workspace = TextPieceCandidateWorkspace::default();

    let mut constrain_prefix = |generated: &[u32], allowed: &mut Vec<u32>| {
        while consumed < generated.len() {
            let token_id = generated[consumed];
            let piece = tokenizer
                .text_token_bytes(token_id)
                .ok_or_else(|| json_error("model emitted a non-text token in JSON mode"))?;
            grammar.push(piece)?;
            if let Some(schema) = &mut planner_schema {
                schema.push(piece)?;
            }
            output_bytes.extend_from_slice(piece);
            consumed += 1;
        }

        if let Some(answer) = crate::streaming::next_answer_preview(
            &output_bytes,
            &last_preview,
            grammar.is_complete(),
            &mut last_preview_scan_bytes,
        ) {
            answer_update(&answer);
            last_preview = answer;
        }

        let schema_complete = planner_schema
            .as_ref()
            .is_none_or(PlannerSchemaPrefix::is_complete);
        let terminal = (schema_complete
            && complete_json_is_accepted(&grammar, &output_bytes, policy.complete_rule))
        .then_some(policy.end_of_turn_token_id);
        let possible_first_bytes = grammar.possible_first_bytes();
        let pieces = candidate_workspace.iter(tokenizer, &possible_first_bytes);
        let mut check_cancelled = || (cancellation.borrow_mut())();
        let mask = if let Some(schema) = &planner_schema {
            mask_planner_token_pieces(
                &grammar,
                schema,
                pieces,
                terminal,
                &mut check_cancelled,
                allowed,
            )
        } else {
            mask_json_token_pieces(&grammar, pieces, terminal, &mut check_cancelled, allowed)
        };
        mask.map(|()| true)
    };
    let is_cancelled = || (cancellation.borrow_mut())();
    let generated = match prompt {
        PlannerDecoderPrompt::Embedded(embedded_prompt) => generate_greedy_with_embedded_prompt(
            decoder,
            embedded_prompt,
            &stop_ids,
            policy.maximum_new_tokens,
            &mut constrain_prefix,
            &is_cancelled,
        )?,
        PlannerDecoderPrompt::Text(prompt_token_ids) => generate_greedy(
            decoder,
            prompt_token_ids,
            &stop_ids,
            policy.maximum_new_tokens,
            &mut constrain_prefix,
            &is_cancelled,
        )?,
    };

    if generated.len() >= policy.maximum_new_tokens
        || !grammar.is_complete()
        || serde_json::from_slice::<Value>(&output_bytes).is_err()
    {
        return Err(json_error("generation ended before a complete JSON value"));
    }
    String::from_utf8(output_bytes).map_err(|_| json_error("generated JSON is not UTF-8"))
}

fn complete_json_is_accepted(
    grammar: &JsonPrefixState,
    bytes: &[u8],
    complete_rule: CompleteJsonRule,
) -> bool {
    if !grammar.is_complete() || serde_json::from_slice::<Value>(bytes).is_err() {
        return false;
    }
    std::str::from_utf8(bytes).is_ok_and(|json| match complete_rule {
        CompleteJsonRule::AnyJson => true,
        CompleteJsonRule::PlannerTurn => crate::model::validate_turn_json(json).is_ok(),
    })
}

fn json_error(message: &str) -> CoreError {
    CoreError::Model(message.into())
}

#[cfg(test)]
mod tests {
    use super::{
        CompleteJsonRule, JsonPrefixState, complete_json_is_accepted, mask_json_token_pieces,
        mask_planner_token_pieces,
    };
    use crate::CoreError;
    use crate::planner_schema::PlannerSchemaPrefix;
    use crate::qwen_tokenizer::{Qwen35Tokenizer, TextPieceCandidateWorkspace};
    use serde_json::Value;

    fn state_for(value: &[u8]) -> JsonPrefixState {
        let mut state = JsonPrefixState::default();
        state.push(value).unwrap();
        state
    }

    #[test]
    fn accepts_incomplete_but_extendable_json_prefixes() {
        for prefix in [
            b"".as_slice(),
            b"{".as_slice(),
            b"{\"goal\"".as_slice(),
            b"{\"goal\":".as_slice(),
            b"{\"goal\":\"text\\u12".as_slice(),
            b"{\"items\":[1,".as_slice(),
            b"-".as_slice(),
            b"1.".as_slice(),
            b"2.3e+".as_slice(),
            b"true".as_slice(),
        ] {
            assert!(state_for(prefix).is_viable(), "prefix: {prefix:?}");
        }
        assert!(state_for(b"true").is_complete());
        assert!(!state_for(b"2.3e+").is_complete());
    }

    #[test]
    fn rejects_invalid_json_prefixes_at_the_byte_that_breaks_them() {
        for invalid in [
            b"[1,]".as_slice(),
            b"{\"a\" 1}".as_slice(),
            b"01".as_slice(),
            b"truex".as_slice(),
            b"\"\\q\"".as_slice(),
            b"\"\\uDC00\"".as_slice(),
            b"\"\\uD800x\"".as_slice(),
            b"{}{}".as_slice(),
        ] {
            let mut state = JsonPrefixState::default();
            assert!(state.push(invalid).is_err(), "invalid input: {invalid:?}");
        }
    }

    #[test]
    fn token_mask_checks_cancellation_during_vocabulary_scan() {
        let grammar = JsonPrefixState::default();
        let pieces = (0..768).map(|token| (token, b"{".as_slice()));
        let mut checks = 0;
        let mut allowed = Vec::new();
        let result = mask_json_token_pieces(
            &grammar,
            pieces,
            None,
            || {
                checks += 1;
                checks == 3
            },
            &mut allowed,
        );
        assert!(matches!(result, Err(CoreError::Cancelled)));
        assert_eq!(checks, 3);
    }

    #[test]
    #[ignore = "release-only first-byte indexed constrained-mask comparison"]
    fn indexed_constrained_mask_measurement() {
        use std::collections::BTreeSet;
        use std::hint::black_box;
        use std::time::Instant;

        const TEXT_VOCABULARY_SIZE: usize = 248_044;

        fn previous_tree_mask(
            grammar: &JsonPrefixState,
            tokenizer: &Qwen35Tokenizer,
        ) -> BTreeSet<u32> {
            let mut allowed = BTreeSet::new();
            for token_id in 0..tokenizer.vocabulary_size() {
                let Some(piece) = tokenizer.text_token_bytes(token_id as u32) else {
                    continue;
                };
                if piece
                    .first()
                    .is_none_or(|byte| !grammar.could_accept_first_byte(*byte))
                {
                    continue;
                }
                let mut candidate = grammar.clone();
                if candidate.push(piece).is_err() || !candidate.is_viable() {
                    continue;
                }
                allowed.insert(token_id as u32);
            }
            allowed
        }

        let grammar = JsonPrefixState::default();
        let synthetic_pieces = (0..TEXT_VOCABULARY_SIZE)
            .map(|token_id| {
                if token_id.is_multiple_of(10) {
                    b'1'
                } else {
                    b'x'
                }
            })
            .collect::<Vec<_>>();
        let tokenizer = Qwen35Tokenizer::for_test_text_bytes(&synthetic_pieces);
        let mut candidate_workspace = TextPieceCandidateWorkspace::default();
        let mut fill_indexed_mask = |state: &JsonPrefixState, output: &mut Vec<u32>| {
            let first_bytes = state.possible_first_bytes();
            let candidates = candidate_workspace.iter(&tokenizer, &first_bytes);
            mask_json_token_pieces(state, candidates, None, || false, output)
        };
        let mut reusable_mask = Vec::with_capacity(TEXT_VOCABULARY_SIZE);
        fill_indexed_mask(&grammar, &mut reusable_mask).expect("warm indexed token mask");
        let expected = previous_tree_mask(&grammar, &tokenizer);
        assert_eq!(reusable_mask.len(), TEXT_VOCABULARY_SIZE / 10 + 1);
        assert_eq!(reusable_mask, expected.into_iter().collect::<Vec<_>>());

        let mut tree_times = Vec::with_capacity(31);
        let mut indexed_times = Vec::with_capacity(31);
        for sample in 0_usize..31 {
            if sample.is_multiple_of(2) {
                let start = Instant::now();
                black_box(previous_tree_mask(
                    black_box(&grammar),
                    black_box(&tokenizer),
                ));
                tree_times.push(start.elapsed());

                let start = Instant::now();
                fill_indexed_mask(black_box(&grammar), black_box(&mut reusable_mask))
                    .expect("indexed mask into reusable sorted buffer");
                black_box(&reusable_mask);
                indexed_times.push(start.elapsed());
            } else {
                let start = Instant::now();
                fill_indexed_mask(black_box(&grammar), black_box(&mut reusable_mask))
                    .expect("indexed mask into reusable sorted buffer");
                black_box(&reusable_mask);
                indexed_times.push(start.elapsed());

                let start = Instant::now();
                black_box(previous_tree_mask(
                    black_box(&grammar),
                    black_box(&tokenizer),
                ));
                tree_times.push(start.elapsed());
            }
        }
        tree_times.sort_unstable();
        indexed_times.sort_unstable();
        eprintln!(
            "json-constrained-mask vocab={TEXT_VOCABULARY_SIZE} accepted={} samples=31 tree_p50_us={} tree_p95_us={} indexed_reused_p50_us={} indexed_reused_p95_us={}",
            reusable_mask.len(),
            tree_times[15].as_nanos() / 1_000,
            tree_times[29].as_nanos() / 1_000,
            indexed_times[15].as_nanos() / 1_000,
            indexed_times[29].as_nanos() / 1_000,
        );
    }

    #[test]
    fn token_mask_adds_end_token_only_for_complete_json() {
        let incomplete = state_for(b"{");
        let mut incomplete_mask = Vec::new();
        mask_json_token_pieces(
            &incomplete,
            [(0, b"\"".as_slice())],
            Some(248_046),
            || false,
            &mut incomplete_mask,
        )
        .unwrap();
        assert!(!incomplete_mask.contains(&248_046));

        let complete = state_for(b"{}");
        mask_json_token_pieces(
            &complete,
            [(0, b" ".as_slice())],
            Some(248_046),
            || false,
            &mut incomplete_mask,
        )
        .unwrap();
        assert!(incomplete_mask.contains(&248_046));
        assert_eq!(incomplete_mask, [0, 248_046]);
    }

    #[test]
    fn planner_mask_prunes_schema_invalid_keys_before_json_does() {
        let syntax = state_for(b"{");
        let mut schema = PlannerSchemaPrefix::new().unwrap();
        schema.push(b"{").unwrap();
        let mut mask = Vec::new();
        mask_planner_token_pieces(
            &syntax,
            &schema,
            [(0, b"\"goal\"".as_slice()), (1, b"\"actions\"".as_slice())],
            None,
            || false,
            &mut mask,
        )
        .unwrap();
        assert!(!mask.contains(&0));
        assert!(mask.contains(&1));
    }

    #[test]
    fn planner_terminal_gate_uses_the_closed_turn_validator() {
        let valid = br#"{"goal":"Answer","answer":"Ready","actions":[]}"#;
        let valid_state = state_for(valid);
        assert!(complete_json_is_accepted(
            &valid_state,
            valid,
            CompleteJsonRule::PlannerTurn
        ));

        let invalid = br#"{"goal":"Answer","answer":"Ready","actions":[],"target":"/tmp"}"#;
        let invalid_state = state_for(invalid);
        assert!(complete_json_is_accepted(
            &invalid_state,
            invalid,
            CompleteJsonRule::AnyJson
        ));
        assert!(!complete_json_is_accepted(
            &invalid_state,
            invalid,
            CompleteJsonRule::PlannerTurn
        ));

        let partial = state_for(b"{\"goal\":\"Answer\"");
        assert!(!complete_json_is_accepted(
            &partial,
            b"{\"goal\":\"Answer\"",
            CompleteJsonRule::AnyJson
        ));
    }

    #[test]
    fn first_byte_fast_filter_matches_full_prefix_validation() {
        for prefix in [
            b"".as_slice(),
            b"{\"x\":".as_slice(),
            b"[true,".as_slice(),
            b"\"text".as_slice(),
            b"\"text\\".as_slice(),
            b"1".as_slice(),
            b"1e".as_slice(),
        ] {
            let state = state_for(prefix);
            for byte in 0..=u8::MAX {
                let mut candidate = state.clone();
                assert_eq!(
                    state.could_accept_first_byte(byte),
                    candidate.push(&[byte]).is_ok(),
                    "prefix {prefix:?}, byte {byte:#x}"
                );
            }
        }
    }

    #[test]
    fn handles_utf8_and_surrogate_pairs_split_across_token_boundaries() {
        let mut state = JsonPrefixState::default();
        state.push(b"{\"x\":\"\xf0\x9f").unwrap();
        assert!(state.is_viable());
        state.push(b"\x98\x80\\uD834\\uDD1E\"}").unwrap();
        assert!(state.is_complete());
        let parsed: Value =
            serde_json::from_slice(b"{\"x\":\"\xf0\x9f\x98\x80\\uD834\\uDD1E\"}").unwrap();
        assert!(parsed["x"].as_str().is_some());
    }

    #[test]
    fn requires_complete_syntax_before_end_of_turn() {
        assert!(state_for(b" {\"x\":[true, null, -1.25e3]} \t").is_complete());
        assert!(!state_for(b"{\"x\":[true, null, -1.25e3]").is_complete());
    }
}
