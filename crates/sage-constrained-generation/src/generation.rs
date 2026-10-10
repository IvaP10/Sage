//! First-party bounded JSON prefix grammar for constrained local generation.
//!
//! This validates every candidate against RFC 8259 syntax and, for planner
//! turns, an incremental automaton compiled from Sage's closed output schema.
//! Completed turns still pass the independent planner validator; model output
//! remains untrusted and never grants execution authority.

use std::cell::RefCell;
use std::collections::BTreeSet;

use sage_qwen_tokenizer::{Qwen35Tokenizer, TextPieceCandidateWorkspace};
use sage_qwen35_runtime::qwen35::{
    Qwen35EmbeddedPrompt, Qwen35TextDecoder, generate_greedy, generate_greedy_with_embedded_prompt,
};
use serde_json::Value;

use crate::schema::PlannerSchemaPrefix;
use crate::{DecodeError, DecodeResult, MAX_STRUCTURED_JSON_BYTES};

const MAX_JSON_BYTES: usize = MAX_STRUCTURED_JSON_BYTES;
const MAX_JSON_NESTING: usize = 64;
const TOKEN_MASK_CANCEL_CHECK_INTERVAL: usize = 256;

enum PlannerDecoderPrompt<'a> {
    Text(&'a [u32]),
    Embedded(&'a Qwen35EmbeddedPrompt<'a>),
}

struct JsonGenerationPolicy<'a> {
    end_of_turn_token_id: u32,
    maximum_new_tokens: usize,
    planner_schema: Option<(&'a Value, &'a [&'a str])>,
}

/// Caller-owned policy and observation hooks for schema-constrained decoding.
/// Keeping these together makes the model boundary explicit at call sites.
pub struct SchemaGenerationOptions<'a, V, U, C> {
    pub schema: &'a Value,
    pub independent_read_kinds: &'a [&'a str],
    pub end_of_turn_token_id: u32,
    pub maximum_new_tokens: usize,
    pub validate_complete: V,
    pub answer_update: U,
    pub cancelled: C,
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
    fn push(&mut self, byte: u8) -> DecodeResult<bool> {
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
#[derive(Debug)]
pub struct JsonPrefixState {
    root: RootState,
    stack: Vec<Container>,
    lexical: Option<Lexical>,
    bytes_seen: usize,
    invalid: bool,
}

// Token-mask probes copy the parser state thousands of times per generated
// token. Reuse the bounded nesting stack instead of allocating a new Vec for
// every vocabulary candidate.
impl Clone for JsonPrefixState {
    fn clone(&self) -> Self {
        Self {
            root: self.root,
            stack: self.stack.clone(),
            lexical: self.lexical,
            bytes_seen: self.bytes_seen,
            invalid: self.invalid,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.root = source.root;
        self.stack.clone_from(&source.stack);
        self.lexical = source.lexical;
        self.bytes_seen = source.bytes_seen;
        self.invalid = source.invalid;
    }
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
    pub fn push(&mut self, bytes: &[u8]) -> DecodeResult<()> {
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

    fn push_byte(&mut self, byte: u8) -> DecodeResult<()> {
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

    fn push_normal(&mut self, byte: u8) -> DecodeResult<()> {
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

    fn begin_value(&mut self, byte: u8) -> DecodeResult<()> {
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

    fn mark_value_started(&mut self) -> DecodeResult<()> {
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

    fn push_container(&mut self, container: Container) -> DecodeResult<()> {
        if self.stack.len() >= MAX_JSON_NESTING {
            return Err(json_error("JSON nesting exceeds Sage's 64-level bound"));
        }
        self.stack.push(container);
        Ok(())
    }

    fn close_container(&mut self, object: bool) -> DecodeResult<()> {
        match self.stack.pop() {
            Some(Container::Object(_)) if object => Ok(()),
            Some(Container::Array(_)) if !object => Ok(()),
            _ => Err(json_error("mismatched JSON container closure")),
        }
    }
}

fn push_number(state: &mut NumberState, byte: u8) -> DecodeResult<bool> {
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

#[cfg(test)]
fn mask_json_token_pieces<'a, I, C>(
    grammar: &JsonPrefixState,
    pieces: I,
    end_token: Option<u32>,
    cancelled: C,
    allowed: &mut Vec<u32>,
) -> DecodeResult<()>
where
    I: IntoIterator<Item = (u32, &'a [u8])>,
    C: FnMut() -> bool,
{
    mask_token_pieces(grammar, None, false, pieces, end_token, cancelled, allowed)
}

fn mask_json_token_pieces_prefiltered<'a, I, C>(
    grammar: &JsonPrefixState,
    pieces: I,
    end_token: Option<u32>,
    cancelled: C,
    allowed: &mut Vec<u32>,
) -> DecodeResult<()>
where
    I: IntoIterator<Item = (u32, &'a [u8])>,
    C: FnMut() -> bool,
{
    mask_token_pieces(grammar, None, true, pieces, end_token, cancelled, allowed)
}

#[cfg(test)]
fn mask_planner_token_pieces<'a, I, C>(
    grammar: &JsonPrefixState,
    planner_schema: &PlannerSchemaPrefix,
    pieces: I,
    end_token: Option<u32>,
    cancelled: C,
    allowed: &mut Vec<u32>,
) -> DecodeResult<()>
where
    I: IntoIterator<Item = (u32, &'a [u8])>,
    C: FnMut() -> bool,
{
    mask_token_pieces(
        grammar,
        Some(planner_schema),
        false,
        pieces,
        end_token,
        cancelled,
        allowed,
    )
}

fn mask_prefiltered_planner_token_pieces<'a, I, C>(
    grammar: &JsonPrefixState,
    planner_schema: &PlannerSchemaPrefix,
    pieces: I,
    end_token: Option<u32>,
    cancelled: C,
    allowed: &mut Vec<u32>,
) -> DecodeResult<()>
where
    I: IntoIterator<Item = (u32, &'a [u8])>,
    C: FnMut() -> bool,
{
    mask_token_pieces(
        grammar,
        Some(planner_schema),
        true,
        pieces,
        end_token,
        cancelled,
        allowed,
    )
}

fn possible_planner_first_bytes(
    grammar: &JsonPrefixState,
    planner_schema: &PlannerSchemaPrefix,
) -> [bool; 256] {
    let mut possible = grammar.possible_first_bytes();
    for (index, allowed) in possible.iter_mut().enumerate() {
        if *allowed && !planner_schema.could_accept_first_byte(index as u8) {
            *allowed = false;
        }
    }
    possible
}

fn mask_token_pieces<'a, I, C>(
    grammar: &JsonPrefixState,
    planner_schema: Option<&PlannerSchemaPrefix>,
    first_byte_prefiltered: bool,
    pieces: I,
    end_token: Option<u32>,
    mut cancelled: C,
    allowed: &mut Vec<u32>,
) -> DecodeResult<()>
where
    I: IntoIterator<Item = (u32, &'a [u8])>,
    C: FnMut() -> bool,
{
    allowed.clear();
    let mut previous_token = None;
    let mut candidate = None;
    let mut candidate_schema = None;
    for (index, (token_id, piece)) in pieces.into_iter().enumerate() {
        if index.is_multiple_of(TOKEN_MASK_CANCEL_CHECK_INTERVAL) && cancelled() {
            return Err(DecodeError::Cancelled);
        }
        if previous_token.is_some_and(|previous| token_id < previous) {
            return Err(json_error("token mask input is not ordered by token ID"));
        }
        previous_token = Some(token_id);
        let Some(first_byte) = piece.first() else {
            continue;
        };
        if !first_byte_prefiltered
            && (!grammar.could_accept_first_byte(*first_byte)
                || planner_schema
                    .is_some_and(|schema| !schema.could_accept_first_byte(*first_byte)))
        {
            continue;
        }
        let candidate = candidate.get_or_insert_with(|| grammar.clone());
        candidate.clone_from(grammar);
        if candidate.push(piece).is_err() || !candidate.is_viable() {
            continue;
        }
        if let Some(planner_schema) = planner_schema {
            let candidate_schema = candidate_schema.get_or_insert_with(|| planner_schema.clone());
            candidate_schema.clone_from(planner_schema);
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

/// Generate one bounded JSON value. Only ordinary tokenizer pieces that
/// preserve JSON-prefix validity are eligible; the stop token is enabled only
/// after a complete value accepted by the caller's independent validator.
pub fn generate_json_greedy<C, V>(
    decoder: &mut Qwen35TextDecoder,
    tokenizer: &Qwen35Tokenizer,
    prompt_token_ids: &[u32],
    end_of_turn_token_id: u32,
    maximum_new_tokens: usize,
    validate_complete: V,
    cancelled: C,
) -> DecodeResult<String>
where
    V: Fn(&str) -> bool,
    C: FnMut() -> bool,
{
    generate_json_greedy_validated(
        decoder,
        tokenizer,
        PlannerDecoderPrompt::Text(prompt_token_ids),
        JsonGenerationPolicy {
            end_of_turn_token_id,
            maximum_new_tokens,
            planner_schema: None,
        },
        validate_complete,
        |_| {},
        cancelled,
    )
}

/// Generate a JSON value constrained by a caller-supplied closed schema. The
/// schema controls token admission only; the caller must still validate the
/// completed output independently before using it.
pub fn generate_schema_greedy<C, U, V>(
    decoder: &mut Qwen35TextDecoder,
    tokenizer: &Qwen35Tokenizer,
    prompt_token_ids: &[u32],
    options: SchemaGenerationOptions<'_, V, U, C>,
) -> DecodeResult<String>
where
    V: Fn(&str) -> bool,
    U: FnMut(&str),
    C: FnMut() -> bool,
{
    generate_json_greedy_validated(
        decoder,
        tokenizer,
        PlannerDecoderPrompt::Text(prompt_token_ids),
        JsonGenerationPolicy {
            end_of_turn_token_id: options.end_of_turn_token_id,
            maximum_new_tokens: options.maximum_new_tokens,
            planner_schema: Some((options.schema, options.independent_read_kinds)),
        },
        options.validate_complete,
        options.answer_update,
        options.cancelled,
    )
}

/// Generate a schema-constrained JSON value from a prompt containing embedded
/// image-token spans. The embeddings remain untrusted evidence; this crate has
/// no authority to resolve or execute a proposed action.
pub fn generate_schema_with_embedded_prompt<C, U, V>(
    decoder: &mut Qwen35TextDecoder,
    tokenizer: &Qwen35Tokenizer,
    prompt: &Qwen35EmbeddedPrompt<'_>,
    options: SchemaGenerationOptions<'_, V, U, C>,
) -> DecodeResult<String>
where
    V: Fn(&str) -> bool,
    U: FnMut(&str),
    C: FnMut() -> bool,
{
    generate_json_greedy_validated(
        decoder,
        tokenizer,
        PlannerDecoderPrompt::Embedded(prompt),
        JsonGenerationPolicy {
            end_of_turn_token_id: options.end_of_turn_token_id,
            maximum_new_tokens: options.maximum_new_tokens,
            planner_schema: Some((options.schema, options.independent_read_kinds)),
        },
        options.validate_complete,
        options.answer_update,
        options.cancelled,
    )
}

fn generate_json_greedy_validated<C, U, V>(
    decoder: &mut Qwen35TextDecoder,
    tokenizer: &Qwen35Tokenizer,
    prompt: PlannerDecoderPrompt<'_>,
    policy: JsonGenerationPolicy<'_>,
    validate_complete: V,
    mut answer_update: U,
    cancelled: C,
) -> DecodeResult<String>
where
    V: Fn(&str) -> bool,
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
        .planner_schema
        .map(|(schema, independent_reads)| PlannerSchemaPrefix::new(schema, independent_reads))
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

        if let Some(answer) = crate::preview::next_answer_preview(
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
            && complete_json_is_accepted(&grammar, &output_bytes, &validate_complete))
        .then_some(policy.end_of_turn_token_id);
        let possible_first_bytes = match planner_schema.as_ref() {
            Some(schema) => possible_planner_first_bytes(&grammar, schema),
            None => grammar.possible_first_bytes(),
        };
        let pieces = candidate_workspace.iter(tokenizer, &possible_first_bytes);
        let mut check_cancelled = || (cancellation.borrow_mut())();
        let mask = if let Some(schema) = &planner_schema {
            mask_prefiltered_planner_token_pieces(
                &grammar,
                schema,
                pieces,
                terminal,
                &mut check_cancelled,
                allowed,
            )
        } else {
            mask_json_token_pieces_prefiltered(
                &grammar,
                pieces,
                terminal,
                &mut check_cancelled,
                allowed,
            )
        };
        mask.map(|()| true)
    };
    let is_cancelled = || (cancellation.borrow_mut())();
    let mut runtime_constraint = |generated: &[u32], allowed: &mut Vec<u32>| {
        constrain_prefix(generated, allowed).map_err(|error| match error {
            DecodeError::Cancelled => sage_qwen35_runtime::Qwen35Error::Cancelled,
            DecodeError::Model(message) => sage_qwen35_runtime::Qwen35Error::Model(message),
        })
    };
    let generated = match prompt {
        PlannerDecoderPrompt::Embedded(embedded_prompt) => generate_greedy_with_embedded_prompt(
            decoder,
            embedded_prompt,
            &stop_ids,
            policy.maximum_new_tokens,
            &mut runtime_constraint,
            &is_cancelled,
        )?,
        PlannerDecoderPrompt::Text(prompt_token_ids) => generate_greedy(
            decoder,
            prompt_token_ids,
            &stop_ids,
            policy.maximum_new_tokens,
            &mut runtime_constraint,
            &is_cancelled,
        )?,
    };

    if generated.len() >= policy.maximum_new_tokens
        || !grammar.is_complete()
        || serde_json::from_slice::<Value>(&output_bytes).is_err()
        || !std::str::from_utf8(&output_bytes).is_ok_and(&validate_complete)
    {
        return Err(json_error("generation ended before a complete JSON value"));
    }
    String::from_utf8(output_bytes).map_err(|_| json_error("generated JSON is not UTF-8"))
}

fn complete_json_is_accepted<V>(
    grammar: &JsonPrefixState,
    bytes: &[u8],
    validate_complete: &V,
) -> bool
where
    V: Fn(&str) -> bool,
{
    if !grammar.is_complete() || serde_json::from_slice::<Value>(bytes).is_err() {
        return false;
    }
    std::str::from_utf8(bytes).is_ok_and(validate_complete)
}

fn json_error(message: &str) -> DecodeError {
    DecodeError::Model(message.into())
}

#[cfg(test)]
mod tests {
    use super::{
        JsonPrefixState, complete_json_is_accepted, mask_json_token_pieces,
        mask_planner_token_pieces, mask_prefiltered_planner_token_pieces,
        possible_planner_first_bytes,
    };
    use crate::DecodeError;
    use crate::schema::{PlannerSchemaPrefix, TEST_INDEPENDENT_READ_KINDS, test_schema};
    use sage_qwen_tokenizer::{Qwen35Tokenizer, TextPieceCandidateWorkspace};

    #[test]
    fn json_prefix_recognizer_accepts_only_extendable_bounded_syntax() {
        for prefix in [
            b"".as_slice(),
            b"{".as_slice(),
            b"{\"goal\"".as_slice(),
            b"{\"goal\":".as_slice(),
            b"\"\\uD83D\\uDE00".as_slice(),
            b"[1,".as_slice(),
            b"2.3e+".as_slice(),
        ] {
            let mut state = JsonPrefixState::default();
            assert!(state.push(prefix).is_ok(), "{prefix:?}");
            assert!(state.is_viable(), "{prefix:?}");
        }

        for invalid in [
            b"[1,]".as_slice(),
            b"{\"a\" 1}".as_slice(),
            b"01".as_slice(),
            b"truex".as_slice(),
            b"\"\\q\"".as_slice(),
            b"{}{}".as_slice(),
        ] {
            let mut state = JsonPrefixState::default();
            assert!(state.push(invalid).is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn token_mask_admits_only_viable_pieces_and_delays_the_stop_token() {
        let mut state = JsonPrefixState::default();
        let mut allowed = Vec::new();
        mask_json_token_pieces(
            &state,
            [
                (0, b"{".as_slice()),
                (1, b"x".as_slice()),
                (2, b"1".as_slice()),
            ],
            Some(9),
            || false,
            &mut allowed,
        )
        .unwrap();
        assert_eq!(allowed, [0, 2]);

        state.push(b"true").unwrap();
        assert!(state.is_complete());
        mask_json_token_pieces(&state, [], Some(9), || false, &mut allowed).unwrap();
        assert_eq!(allowed, [9]);
    }

    #[test]
    fn tokenizer_prefiltered_mask_matches_checked_candidate_validation() {
        let pieces: [&[u8]; 9] = [
            b"\"actions\"",
            b"\"answer\"",
            b"\"goal\"",
            b"\"unknown\"",
            b"\"actions\":",
            b" ",
            b"}",
            b"0",
            b"{",
        ];
        let tokenizer = Qwen35Tokenizer::for_test_text_pieces(&pieces);
        let grammar = JsonPrefixState::default();
        let schema = PlannerSchemaPrefix::new(&test_schema(), TEST_INDEPENDENT_READ_KINDS).unwrap();
        let unrestricted = [true; 256];
        let prefiltered = possible_planner_first_bytes(&grammar, &schema);
        let mut full_workspace = TextPieceCandidateWorkspace::default();
        let mut filtered_workspace = TextPieceCandidateWorkspace::default();
        let mut expected = Vec::new();
        let mut observed = Vec::new();

        mask_planner_token_pieces(
            &grammar,
            &schema,
            full_workspace.iter(&tokenizer, &unrestricted),
            None,
            || false,
            &mut expected,
        )
        .unwrap();
        mask_prefiltered_planner_token_pieces(
            &grammar,
            &schema,
            filtered_workspace.iter(&tokenizer, &prefiltered),
            None,
            || false,
            &mut observed,
        )
        .unwrap();

        assert_eq!(observed, expected);
        assert_eq!(observed, [8]);
    }

    #[test]
    fn vocabulary_mask_observes_cancellation_during_candidate_scan() {
        let state = JsonPrefixState::default();
        let mut allowed = Vec::new();
        let result = mask_json_token_pieces(
            &state,
            [(0, b"{".as_slice()), (1, b"1".as_slice())],
            None,
            || true,
            &mut allowed,
        );
        assert!(matches!(result, Err(DecodeError::Cancelled)));
    }

    #[test]
    fn stop_token_gate_requires_syntax_and_an_independent_schema_validator() {
        let valid_syntax = br#"{"actions":[],"answer":"ready","goal":"answer"}"#;
        let mut complete = JsonPrefixState::default();
        complete.push(valid_syntax).unwrap();
        assert!(complete_json_is_accepted(
            &complete,
            valid_syntax,
            &|text| { text.contains("\"goal\"") }
        ));
        assert!(!complete_json_is_accepted(&complete, valid_syntax, &|_| {
            false
        }));

        let invalid_syntax = br#"{"actions":[],}"#;
        let mut incomplete = JsonPrefixState::default();
        assert!(incomplete.push(invalid_syntax).is_err());
        assert!(!complete_json_is_accepted(
            &incomplete,
            invalid_syntax,
            &|_| true
        ));
    }

    #[test]
    fn intersected_first_byte_mask_preserves_extendable_json_schema_transitions() {
        let prefixes: [&[u8]; 6] = [
            b"",
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes"#,
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":64,"path":"#,
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":64,"path":"/tmp/"#,
            br#"{"actions":[],"answer":"Ready","goal":"#,
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":64,"path":"/tmp/a"}}],"answer":"","goal":"#,
        ];

        for prefix in prefixes {
            let mut grammar = JsonPrefixState::default();
            grammar.push(prefix).unwrap();
            let mut schema =
                PlannerSchemaPrefix::new(&test_schema(), TEST_INDEPENDENT_READ_KINDS).unwrap();
            schema.push(prefix).unwrap();
            let permitted = possible_planner_first_bytes(&grammar, &schema);

            for byte in u8::MIN..=u8::MAX {
                let mut next_grammar = grammar.clone();
                let grammar_can_extend =
                    next_grammar.push(&[byte]).is_ok() && next_grammar.is_viable();
                let mut next_schema = schema.clone();
                let schema_can_extend =
                    next_schema.push(&[byte]).is_ok() && next_schema.is_viable();
                assert!(
                    !grammar_can_extend || !schema_can_extend || permitted[byte as usize],
                    "combined prefilter rejected byte {byte:#04x} after {:?}",
                    String::from_utf8_lossy(prefix)
                );
            }
        }
    }

    #[test]
    #[ignore = "release-only tokenizer bucket and schema-first candidate-mask measurement"]
    fn schema_first_candidate_prefilter_measurement() {
        use std::hint::black_box;
        use std::time::Instant;

        const CANDIDATE_COUNT: usize = 32_768;
        const PIECES: [&[u8]; 9] = [
            b"\"actions\"",
            b"\"answer\"",
            b"\"goal\"",
            b"\"unknown\"",
            b"\"actions\":",
            b" ",
            b"}",
            b"0",
            b"{",
        ];

        fn allocating_reference<'a>(
            grammar: &JsonPrefixState,
            schema: &PlannerSchemaPrefix,
            candidates: impl IntoIterator<Item = (u32, &'a [u8])>,
            allowed: &mut Vec<u32>,
        ) {
            allowed.clear();
            for (token_id, piece) in candidates {
                if piece
                    .first()
                    .is_none_or(|byte| !grammar.could_accept_first_byte(*byte))
                {
                    continue;
                }
                let mut candidate_grammar = grammar.clone();
                if candidate_grammar.push(piece).is_err() || !candidate_grammar.is_viable() {
                    continue;
                }
                let mut candidate_schema = schema.clone();
                if candidate_schema.push(piece).is_err() || !candidate_schema.is_viable() {
                    continue;
                }
                allowed.push(token_id);
            }
        }

        let grammar = JsonPrefixState::default();
        let schema = PlannerSchemaPrefix::new(&test_schema(), TEST_INDEPENDENT_READ_KINDS).unwrap();
        let synthetic_pieces = (0..CANDIDATE_COUNT)
            .map(|index| PIECES[index % PIECES.len()])
            .collect::<Vec<_>>();
        let tokenizer = Qwen35Tokenizer::for_test_text_pieces(&synthetic_pieces);
        let json_first_bytes = grammar.possible_first_bytes();
        let schema_first_bytes = possible_planner_first_bytes(&grammar, &schema);
        let mut expected = Vec::new();
        let mut observed = Vec::new();
        let mut reference_workspace = TextPieceCandidateWorkspace::default();
        let mut optimized_workspace = TextPieceCandidateWorkspace::default();
        allocating_reference(
            &grammar,
            &schema,
            reference_workspace.iter(&tokenizer, &json_first_bytes),
            &mut expected,
        );
        mask_prefiltered_planner_token_pieces(
            &grammar,
            &schema,
            optimized_workspace.iter(&tokenizer, &schema_first_bytes),
            None,
            || false,
            &mut observed,
        )
        .unwrap();
        assert_eq!(observed, expected);

        let mut allocating = Vec::with_capacity(51);
        let mut schema_first = Vec::with_capacity(51);
        for sample in 0_usize..51 {
            if sample.is_multiple_of(2) {
                let start = Instant::now();
                allocating_reference(
                    black_box(&grammar),
                    black_box(&schema),
                    reference_workspace.iter(&tokenizer, &json_first_bytes),
                    &mut expected,
                );
                black_box(&expected);
                allocating.push(start.elapsed());

                let start = Instant::now();
                mask_prefiltered_planner_token_pieces(
                    black_box(&grammar),
                    black_box(&schema),
                    optimized_workspace.iter(&tokenizer, &schema_first_bytes),
                    None,
                    || false,
                    &mut observed,
                )
                .unwrap();
                black_box(&observed);
                schema_first.push(start.elapsed());
            } else {
                let start = Instant::now();
                mask_prefiltered_planner_token_pieces(
                    black_box(&grammar),
                    black_box(&schema),
                    optimized_workspace.iter(&tokenizer, &schema_first_bytes),
                    None,
                    || false,
                    &mut observed,
                )
                .unwrap();
                black_box(&observed);
                schema_first.push(start.elapsed());

                let start = Instant::now();
                allocating_reference(
                    black_box(&grammar),
                    black_box(&schema),
                    reference_workspace.iter(&tokenizer, &json_first_bytes),
                    &mut expected,
                );
                black_box(&expected);
                allocating.push(start.elapsed());
            }
        }
        allocating.sort_unstable();
        schema_first.sort_unstable();
        eprintln!(
            "planner-schema-mask vocabulary={CANDIDATE_COUNT} json_candidates={} schema_candidates={} accepted={} samples=51 allocating_p50_us={} allocating_p95_us={} schema_first_p50_us={} schema_first_p95_us={}",
            reference_workspace
                .iter(&tokenizer, &json_first_bytes)
                .count(),
            optimized_workspace
                .iter(&tokenizer, &schema_first_bytes)
                .count(),
            observed.len(),
            allocating[25].as_nanos() / 1_000,
            allocating[48].as_nanos() / 1_000,
            schema_first[25].as_nanos() / 1_000,
            schema_first[48].as_nanos() / 1_000,
        );
    }
}
