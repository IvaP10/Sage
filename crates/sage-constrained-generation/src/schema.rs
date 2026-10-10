//! Incremental prefix automaton compiled from Sage's closed planner schema.
//!
//! This intersects canonical JSON serialization with the planner contract. The
//! RFC 8259 parser handles general JSON syntax; final planner validation stays
//! an independent backstop.

use std::sync::Arc;

use serde_json::Value;

use crate::{DecodeError, DecodeResult};

const MAX_SCHEMA_DEPTH: usize = 16;
const MAX_ACTIVE_CONFIGURATIONS: usize = 128;
const MAX_PREFIX_BYTES: usize = crate::MAX_STRUCTURED_JSON_BYTES;
const MAX_PLANNER_ACTIONS: usize = 8;

#[derive(Clone)]
enum Grammar {
    Bytes(Arc<[u8]>),
    String { minimum: usize, maximum: usize },
    Integer { minimum: u64, maximum: u64 },
    Number,
    Sequence(Vec<Arc<Grammar>>),
    Choice(Vec<Arc<Grammar>>),
}

enum Terminal {
    Bytes { bytes: Arc<[u8]>, offset: usize },
    String(SchemaString),
    Integer(SchemaInteger),
    Number(SchemaNumber),
}

impl Clone for Terminal {
    fn clone(&self) -> Self {
        match self {
            Self::Bytes { bytes, offset } => Self::Bytes {
                bytes: Arc::clone(bytes),
                offset: *offset,
            },
            Self::String(state) => Self::String(state.clone()),
            Self::Integer(state) => Self::Integer(*state),
            Self::Number(state) => Self::Number(state.clone()),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        match (self, source) {
            (
                Self::Bytes {
                    bytes: target_bytes,
                    offset: target_offset,
                },
                Self::Bytes {
                    bytes: source_bytes,
                    offset: source_offset,
                },
            ) => {
                *target_bytes = Arc::clone(source_bytes);
                *target_offset = *source_offset;
            }
            (Self::String(target), Self::String(source)) => target.clone_from(source),
            (Self::Integer(target), Self::Integer(source)) => *target = *source,
            (Self::Number(target), Self::Number(source)) => target.clone_from(source),
            (target, source) => *target = source.clone(),
        }
    }
}

struct Configuration {
    remaining: Vec<Arc<Grammar>>,
    terminal: Option<Terminal>,
}

impl Clone for Configuration {
    fn clone(&self) -> Self {
        Self {
            remaining: self.remaining.clone(),
            terminal: self.terminal.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.remaining.clone_from(&source.remaining);
        self.terminal.clone_from(&source.terminal);
    }
}

#[derive(Clone)]
struct SchemaString {
    started: bool,
    mode: StringMode,
    length: usize,
    minimum: usize,
    maximum: usize,
}

#[derive(Clone, Copy)]
enum StringMode {
    Normal,
    Escape,
    Unicode { remaining: u8, value: u16 },
    NeedLowSlash,
    NeedLowU,
    UnicodeLow { remaining: u8, value: u16 },
}

#[derive(Clone, Copy)]
struct SchemaInteger {
    started: bool,
    digits: usize,
    value: u64,
    minimum: u64,
    maximum: u64,
}

enum IntegerStep {
    Continue(SchemaInteger),
    End,
    Invalid,
}

struct SchemaNumber {
    raw: String,
    state: NumberState,
}

impl Clone for SchemaNumber {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            state: self.state,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.raw.clone_from(&source.raw);
        self.state = source.state;
    }
}

#[derive(Clone, Copy)]
enum NumberState {
    Start,
    NeedInteger,
    Zero,
    Integer,
    NeedFraction,
    Fraction,
    NeedExponent,
    ExponentSign,
    Exponent,
}

enum NumberStep {
    Continue(SchemaNumber),
    End,
    Invalid,
}

/// Bounded recognizer for outputs satisfying Sage's planner schema. It accepts
/// compact JSON with alphabetically ordered object keys, a canonical subset
/// stated in the system prompt.
pub(crate) struct PlannerSchemaPrefix {
    configurations: Vec<Configuration>,
    bytes_seen: usize,
}

impl Clone for PlannerSchemaPrefix {
    fn clone(&self) -> Self {
        Self {
            configurations: self.configurations.clone(),
            bytes_seen: self.bytes_seen,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.configurations.clone_from(&source.configurations);
        self.bytes_seen = source.bytes_seen;
    }
}

impl PlannerSchemaPrefix {
    pub fn new(schema: &Value, independent_read_kinds: &[&str]) -> DecodeResult<Self> {
        let grammar = planner_grammar(schema, independent_read_kinds)?;
        let configurations = normalize_many(vec![Configuration {
            remaining: vec![grammar],
            terminal: None,
        }])?;
        Ok(Self {
            configurations,
            bytes_seen: 0,
        })
    }

    pub fn push(&mut self, bytes: &[u8]) -> DecodeResult<()> {
        let next_length = self
            .bytes_seen
            .checked_add(bytes.len())
            .filter(|length| *length <= MAX_PREFIX_BYTES)
            .ok_or_else(|| schema_error("planner schema prefix exceeds its byte bound"))?;
        let mut configurations = self.configurations.clone();
        for &byte in bytes {
            let mut next = Vec::new();
            for configuration in configurations {
                next.extend(consume(configuration, byte)?);
                if next.len() > MAX_ACTIVE_CONFIGURATIONS {
                    return Err(schema_error(
                        "planner schema prefix exceeds its active-state bound",
                    ));
                }
            }
            configurations = normalize_many(next)?;
            if configurations.is_empty() {
                return Err(schema_error(
                    "planner output cannot satisfy Sage's closed schema",
                ));
            }
        }
        self.configurations = configurations;
        self.bytes_seen = next_length;
        Ok(())
    }

    pub(crate) fn is_viable(&self) -> bool {
        !self.configurations.is_empty()
    }

    /// Reject impossible token starts using the active schema terminals
    /// before copying candidate parser and automaton state. This is a
    /// conservative prefilter: `true` means the full prefix check is needed.
    pub(crate) fn could_accept_first_byte(&self, byte: u8) -> bool {
        if self.bytes_seen >= MAX_PREFIX_BYTES {
            return false;
        }
        self.configurations
            .iter()
            .any(|configuration| match configuration.terminal.as_ref() {
                Some(Terminal::Bytes { bytes, offset }) => {
                    bytes.get(*offset).copied() == Some(byte)
                }
                Some(Terminal::String(state)) => {
                    let mut candidate = state.clone();
                    candidate.push(byte).is_some()
                }
                Some(Terminal::Integer(state)) => {
                    let mut candidate = *state;
                    !matches!(candidate.push(byte), IntegerStep::Invalid)
                }
                // Number termination is schema-context dependent. Keep this
                // branch permissive so the exact automaton remains the sole
                // authority for the delimiter and continuation transition.
                Some(Terminal::Number(_)) => true,
                None => false,
            })
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.configurations.iter().any(|configuration| {
            configuration.remaining.is_empty()
                && match configuration.terminal.as_ref() {
                    None => true,
                    Some(Terminal::Number(number)) => number.is_complete(),
                    Some(_) => false,
                }
        })
    }
}

fn planner_grammar(schema: &Value, independent_read_kinds: &[&str]) -> DecodeResult<Arc<Grammar>> {
    if schema.get("type").and_then(Value::as_str) != Some("object")
        || schema.get("additionalProperties") != Some(&Value::Bool(false))
        || schema
            .get("required")
            .and_then(Value::as_array)
            .is_none_or(|required| {
                required.len() != 3
                    || ["actions", "answer", "goal"]
                        .iter()
                        .any(|name| !required.iter().any(|entry| entry.as_str() == Some(name)))
            })
    {
        return Err(schema_error("planner root schema changed unexpectedly"));
    }
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| schema_error("planner schema has no root properties"))?;
    if properties.len() != 3 {
        return Err(schema_error("planner root schema changed unexpectedly"));
    }
    let goal = nonempty_string(
        properties
            .get("goal")
            .ok_or_else(|| schema_error("planner schema is missing the goal property"))?,
    )?;
    let answer_schema = properties
        .get("answer")
        .ok_or_else(|| schema_error("planner schema is missing the answer property"))?;
    let answer = compile_schema(answer_schema, 0)?;
    if !matches!(answer.as_ref(), Grammar::String { .. }) {
        return Err(schema_error("planner answer must remain a string"));
    }
    let nonempty_answer = nonempty_string(answer_schema)?;
    let actions_schema = properties
        .get("actions")
        .ok_or_else(|| schema_error("planner schema is missing the actions property"))?;
    let maximum_actions = actions_schema
        .get("maxItems")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| (1..=MAX_PLANNER_ACTIONS).contains(value))
        .ok_or_else(|| schema_error("planner action batch bound is invalid"))?;
    let variants = actions_schema
        .get("items")
        .and_then(|items| items.get("anyOf"))
        .and_then(Value::as_array)
        .filter(|variants| !variants.is_empty() && variants.len() <= 32)
        .ok_or_else(|| schema_error("planner actions must retain their closed variants"))?;

    let mut all_actions = Vec::with_capacity(variants.len());
    let mut independent_reads = Vec::new();
    for variant in variants {
        let kind = variant
            .get("properties")
            .and_then(|properties| properties.get("kind"))
            .and_then(|kind| kind.get("enum"))
            .and_then(Value::as_array)
            .filter(|values| values.len() == 1)
            .and_then(|values| values[0].as_str())
            .ok_or_else(|| schema_error("planner action kind must have one closed value"))?;
        let action = compile_schema(variant, 0)?;
        all_actions.push(action.clone());
        if independent_read_kinds.contains(&kind) {
            independent_reads.push(action);
        }
    }
    if all_actions.is_empty() || independent_reads.is_empty() {
        return Err(schema_error("planner action variants are incomplete"));
    }

    // A turn is either a nonempty answer with no actions, or a nonempty plan
    // with an empty answer. Multi-action turns share the validator's read set.
    let answer_turn = sequence(vec![
        bytes(b"{\"actions\":[],\"answer\":"),
        nonempty_answer,
        bytes(b",\"goal\":"),
        goal.clone(),
        bytes(b"}"),
    ]);
    let all_actions = choice(all_actions);
    let independent_reads = choice(independent_reads);
    let mut action_counts = Vec::with_capacity(maximum_actions);
    for count in 1..=maximum_actions {
        let item = if count == 1 {
            all_actions.clone()
        } else {
            independent_reads.clone()
        };
        let mut array = Vec::with_capacity(count * 2 + 2);
        array.push(bytes(b"["));
        for index in 0..count {
            if index > 0 {
                array.push(bytes(b","));
            }
            array.push(item.clone());
        }
        array.push(bytes(b"]"));
        action_counts.push(sequence(array));
    }
    let action_turn = sequence(vec![
        bytes(b"{\"actions\":"),
        choice(action_counts),
        bytes(b",\"answer\":\"\",\"goal\":"),
        goal,
        bytes(b"}"),
    ]);
    Ok(choice(vec![answer_turn, action_turn]))
}

fn nonempty_string(schema: &Value) -> DecodeResult<Arc<Grammar>> {
    match compile_schema(schema, 0)?.as_ref() {
        Grammar::String { maximum, .. } => Ok(Arc::new(Grammar::String {
            minimum: 1,
            maximum: *maximum,
        })),
        _ => Err(schema_error("planner text field must remain a string")),
    }
}

fn compile_schema(schema: &Value, depth: usize) -> DecodeResult<Arc<Grammar>> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(schema_error("planner schema exceeds its nesting bound"));
    }
    if let Some(variants) = schema.get("anyOf").and_then(Value::as_array) {
        if variants.is_empty() || variants.len() > 32 {
            return Err(schema_error("planner schema has an invalid anyOf bound"));
        }
        return variants
            .iter()
            .map(|variant| compile_schema(variant, depth + 1))
            .collect::<DecodeResult<Vec<_>>>()
            .map(choice);
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        if values.is_empty() || values.len() > 256 {
            return Err(schema_error("planner schema has an invalid enum bound"));
        }
        let literals = values
            .iter()
            .map(|value| {
                serde_json::to_vec(value)
                    .map(bytes)
                    .map_err(|_| schema_error("planner enum could not be encoded"))
            })
            .collect::<DecodeResult<Vec<_>>>()?;
        return Ok(choice(literals));
    }
    let kind = schema
        .get("type")
        .ok_or_else(|| schema_error("planner schema node has no type"))?;
    if let Some(kinds) = kind.as_array() {
        if kinds.is_empty() || kinds.len() > 4 {
            return Err(schema_error("planner union has an invalid type bound"));
        }
        return kinds
            .iter()
            .map(|kind| {
                let mut variant = schema.clone();
                let object = variant
                    .as_object_mut()
                    .ok_or_else(|| schema_error("planner schema node is not an object"))?;
                object.insert("type".into(), kind.clone());
                compile_schema(&variant, depth + 1)
            })
            .collect::<DecodeResult<Vec<_>>>()
            .map(choice);
    }
    match kind.as_str() {
        Some("string") => {
            let minimum = bounded_usize(schema.get("minLength"), 0)?;
            let maximum =
                bounded_usize(schema.get("maxLength"), MAX_PREFIX_BYTES)?.min(MAX_PREFIX_BYTES);
            if minimum > maximum {
                return Err(schema_error("planner string bounds are inconsistent"));
            }
            Ok(Arc::new(Grammar::String { minimum, maximum }))
        }
        Some("integer") => {
            let minimum = schema
                .get("minimum")
                .and_then(Value::as_u64)
                .ok_or_else(|| schema_error("planner integer needs a nonnegative minimum"))?;
            let maximum = schema
                .get("maximum")
                .and_then(Value::as_u64)
                .ok_or_else(|| schema_error("planner integer needs a bounded maximum"))?;
            if minimum > maximum {
                return Err(schema_error("planner integer bounds are inconsistent"));
            }
            Ok(Arc::new(Grammar::Integer { minimum, maximum }))
        }
        Some("number") => Ok(Arc::new(Grammar::Number)),
        Some("boolean") => Ok(choice(vec![bytes(b"true"), bytes(b"false")])),
        Some("null") => Ok(bytes(b"null")),
        Some("object") => compile_object(schema, depth),
        Some("array") => compile_array(schema, depth),
        _ => Err(schema_error("planner schema uses an unsupported type")),
    }
}

fn compile_object(schema: &Value, depth: usize) -> DecodeResult<Arc<Grammar>> {
    if schema.get("additionalProperties") != Some(&Value::Bool(false)) {
        return Err(schema_error(
            "planner objects must reject additional properties",
        ));
    }
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .filter(|properties| !properties.is_empty() && properties.len() <= 32)
        .ok_or_else(|| schema_error("planner object properties are invalid"))?;
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .ok_or_else(|| schema_error("planner object required list is missing"))?;
    if required.len() != properties.len()
        || properties
            .keys()
            .any(|name| !required.iter().any(|entry| entry.as_str() == Some(name)))
    {
        return Err(schema_error(
            "planner decoder requires every closed object property",
        ));
    }
    let mut names = properties.keys().collect::<Vec<_>>();
    names.sort_unstable();
    let mut result = Vec::with_capacity(names.len() * 3 + 2);
    result.push(bytes(b"{"));
    for (index, name) in names.into_iter().enumerate() {
        if index > 0 {
            result.push(bytes(b","));
        }
        result.push(
            serde_json::to_vec(name)
                .map(bytes)
                .map_err(|_| schema_error("planner property name could not be encoded"))?,
        );
        result.push(bytes(b":"));
        result.push(compile_schema(
            properties
                .get(name)
                .ok_or_else(|| schema_error("planner property disappeared during compilation"))?,
            depth + 1,
        )?);
    }
    result.push(bytes(b"}"));
    Ok(sequence(result))
}

fn compile_array(schema: &Value, depth: usize) -> DecodeResult<Arc<Grammar>> {
    let minimum = bounded_usize(schema.get("minItems"), 0)?;
    let maximum = bounded_usize(schema.get("maxItems"), 0)?;
    if minimum > maximum || maximum > MAX_PLANNER_ACTIONS {
        return Err(schema_error("planner array bounds are invalid"));
    }
    let item = compile_schema(
        schema
            .get("items")
            .ok_or_else(|| schema_error("planner array item schema is missing"))?,
        depth + 1,
    )?;
    let alternatives = (minimum..=maximum)
        .map(|count| array_with_count(item.clone(), count))
        .collect();
    Ok(choice(alternatives))
}

fn array_with_count(item: Arc<Grammar>, count: usize) -> Arc<Grammar> {
    let mut children = Vec::with_capacity(count * 2 + 2);
    children.push(bytes(b"["));
    for index in 0..count {
        if index > 0 {
            children.push(bytes(b","));
        }
        children.push(item.clone());
    }
    children.push(bytes(b"]"));
    sequence(children)
}

fn bounded_usize(value: Option<&Value>, default: usize) -> DecodeResult<usize> {
    match value.and_then(Value::as_u64) {
        Some(value) => usize::try_from(value)
            .map_err(|_| schema_error("planner schema bound overflows this platform")),
        None => Ok(default),
    }
}

fn bytes(value: impl AsRef<[u8]>) -> Arc<Grammar> {
    Arc::new(Grammar::Bytes(Arc::from(value.as_ref())))
}

fn sequence(children: Vec<Arc<Grammar>>) -> Arc<Grammar> {
    Arc::new(Grammar::Sequence(children))
}

fn choice(children: Vec<Arc<Grammar>>) -> Arc<Grammar> {
    if children.len() == 1 {
        children
            .into_iter()
            .next()
            .expect("one grammar alternative")
    } else {
        Arc::new(Grammar::Choice(children))
    }
}

fn normalize_many(initial: Vec<Configuration>) -> DecodeResult<Vec<Configuration>> {
    let mut pending = initial;
    let mut ready = Vec::new();
    while let Some(mut configuration) = pending.pop() {
        if configuration.terminal.is_some() {
            ready.push(configuration);
        } else {
            match configuration.remaining.pop() {
                None => ready.push(configuration),
                Some(grammar) => match grammar.as_ref() {
                    Grammar::Bytes(bytes) if bytes.is_empty() => pending.push(configuration),
                    Grammar::Bytes(bytes) => {
                        configuration.terminal = Some(Terminal::Bytes {
                            bytes: Arc::clone(bytes),
                            offset: 0,
                        });
                        ready.push(configuration);
                    }
                    Grammar::String { minimum, maximum } => {
                        configuration.terminal = Some(Terminal::String(SchemaString {
                            started: false,
                            mode: StringMode::Normal,
                            length: 0,
                            minimum: *minimum,
                            maximum: *maximum,
                        }));
                        ready.push(configuration);
                    }
                    Grammar::Integer { minimum, maximum } => {
                        configuration.terminal = Some(Terminal::Integer(SchemaInteger {
                            started: false,
                            digits: 0,
                            value: 0,
                            minimum: *minimum,
                            maximum: *maximum,
                        }));
                        ready.push(configuration);
                    }
                    Grammar::Number => {
                        configuration.terminal = Some(Terminal::Number(SchemaNumber {
                            raw: String::new(),
                            state: NumberState::Start,
                        }));
                        ready.push(configuration);
                    }
                    Grammar::Sequence(children) => {
                        configuration
                            .remaining
                            .extend(children.iter().rev().cloned());
                        pending.push(configuration);
                    }
                    Grammar::Choice(children) => {
                        for child in children {
                            let mut branch = configuration.clone();
                            branch.remaining.push(Arc::clone(child));
                            pending.push(branch);
                        }
                    }
                },
            }
        }
        if pending.len() + ready.len() > MAX_ACTIVE_CONFIGURATIONS {
            return Err(schema_error(
                "planner schema expands beyond its active-state bound",
            ));
        }
    }
    Ok(ready)
}

fn consume(mut configuration: Configuration, byte: u8) -> DecodeResult<Vec<Configuration>> {
    let Some(terminal) = configuration.terminal.take() else {
        return Ok(Vec::new());
    };
    match terminal {
        Terminal::Bytes { bytes, offset } => {
            if bytes.get(offset).copied() != Some(byte) {
                return Ok(Vec::new());
            }
            if offset + 1 < bytes.len() {
                configuration.terminal = Some(Terminal::Bytes {
                    bytes,
                    offset: offset + 1,
                });
                Ok(vec![configuration])
            } else {
                normalize_many(vec![configuration])
            }
        }
        Terminal::String(mut string) => match string.push(byte) {
            None => Ok(Vec::new()),
            Some(false) => {
                configuration.terminal = Some(Terminal::String(string));
                Ok(vec![configuration])
            }
            Some(true) => normalize_many(vec![configuration]),
        },
        Terminal::Integer(mut integer) => match integer.push(byte) {
            IntegerStep::Invalid => Ok(Vec::new()),
            IntegerStep::Continue(integer) => {
                configuration.terminal = Some(Terminal::Integer(integer));
                Ok(vec![configuration])
            }
            IntegerStep::End => {
                let normalized = normalize_many(vec![configuration])?;
                let mut branches = Vec::new();
                for next in normalized {
                    branches.extend(consume(next, byte)?);
                }
                Ok(branches)
            }
        },
        Terminal::Number(mut number) => match number.push(byte) {
            NumberStep::Invalid => Ok(Vec::new()),
            NumberStep::Continue(number) => {
                configuration.terminal = Some(Terminal::Number(number));
                Ok(vec![configuration])
            }
            NumberStep::End => {
                let normalized = normalize_many(vec![configuration])?;
                let mut branches = Vec::new();
                for next in normalized {
                    branches.extend(consume(next, byte)?);
                }
                Ok(branches)
            }
        },
    }
}

impl SchemaString {
    /// A true result closes the string; None rejects the byte.
    fn push(&mut self, byte: u8) -> Option<bool> {
        if !self.started {
            if byte != b'"' {
                return None;
            }
            self.started = true;
            return Some(false);
        }
        match self.mode {
            StringMode::Normal => match byte {
                b'"' => (self.length >= self.minimum).then_some(true),
                b'\\' => {
                    self.mode = StringMode::Escape;
                    Some(false)
                }
                0x00..=0x1f => None,
                // JSON Schema string lengths count Unicode scalar values,
                // not UTF-8 bytes. The JSON syntax recognizer independently
                // validates continuation bytes and rejects invalid UTF-8.
                _ => self
                    .add_length(usize::from(byte & 0xc0 != 0x80))
                    .then_some(false),
            },
            StringMode::Escape => match byte {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    self.mode = StringMode::Normal;
                    self.add_length(1).then_some(false)
                }
                b'u' => {
                    self.mode = StringMode::Unicode {
                        remaining: 4,
                        value: 0,
                    };
                    Some(false)
                }
                _ => None,
            },
            StringMode::Unicode { remaining, value } => {
                let value = (value << 4) | u16::from(hex_value(byte)?);
                if remaining > 1 {
                    self.mode = StringMode::Unicode {
                        remaining: remaining - 1,
                        value,
                    };
                    Some(false)
                } else if (0xd800..=0xdbff).contains(&value) {
                    self.mode = StringMode::NeedLowSlash;
                    Some(false)
                } else if (0xdc00..=0xdfff).contains(&value) {
                    None
                } else {
                    self.mode = StringMode::Normal;
                    char::from_u32(u32::from(value))?;
                    self.add_length(1).then_some(false)
                }
            }
            StringMode::NeedLowSlash => {
                if byte != b'\\' {
                    return None;
                }
                self.mode = StringMode::NeedLowU;
                Some(false)
            }
            StringMode::NeedLowU => {
                if byte != b'u' {
                    return None;
                }
                self.mode = StringMode::UnicodeLow {
                    remaining: 4,
                    value: 0,
                };
                Some(false)
            }
            StringMode::UnicodeLow { remaining, value } => {
                let value = (value << 4) | u16::from(hex_value(byte)?);
                if remaining > 1 {
                    self.mode = StringMode::UnicodeLow {
                        remaining: remaining - 1,
                        value,
                    };
                    Some(false)
                } else if (0xdc00..=0xdfff).contains(&value) {
                    self.mode = StringMode::Normal;
                    self.add_length(1).then_some(false)
                } else {
                    None
                }
            }
        }
    }

    fn add_length(&mut self, amount: usize) -> bool {
        if let Some(length) = self.length.checked_add(amount)
            && length <= self.maximum
        {
            self.length = length;
            true
        } else {
            false
        }
    }
}

impl SchemaInteger {
    fn push(&mut self, byte: u8) -> IntegerStep {
        if byte.is_ascii_digit() {
            if self.started && self.value == 0 {
                return IntegerStep::Invalid;
            }
            let Some(value) = self
                .value
                .checked_mul(10)
                .and_then(|value| value.checked_add(u64::from(byte - b'0')))
            else {
                return IntegerStep::Invalid;
            };
            let digits = self.digits + 1;
            if !integer_prefix_can_match(value, digits, self.minimum, self.maximum) {
                return IntegerStep::Invalid;
            }
            self.started = true;
            self.digits = digits;
            self.value = value;
            IntegerStep::Continue(*self)
        } else if self.started && (self.minimum..=self.maximum).contains(&self.value) {
            IntegerStep::End
        } else {
            IntegerStep::Invalid
        }
    }
}

impl SchemaNumber {
    fn push(&mut self, byte: u8) -> NumberStep {
        let next = match (self.state, byte) {
            (NumberState::Start, b'-') => NumberState::NeedInteger,
            (NumberState::Start | NumberState::NeedInteger, b'0') => NumberState::Zero,
            (NumberState::Start | NumberState::NeedInteger, b'1'..=b'9') => NumberState::Integer,
            (NumberState::Zero | NumberState::Integer, b'.') => NumberState::NeedFraction,
            (NumberState::Zero | NumberState::Integer, b'e' | b'E') => NumberState::NeedExponent,
            (NumberState::Integer, b'0'..=b'9') => NumberState::Integer,
            (NumberState::NeedFraction, b'0'..=b'9') => NumberState::Fraction,
            (NumberState::Fraction, b'0'..=b'9') => NumberState::Fraction,
            (NumberState::Fraction, b'e' | b'E') => NumberState::NeedExponent,
            (NumberState::NeedExponent, b'+' | b'-') => NumberState::ExponentSign,
            (NumberState::NeedExponent, b'0'..=b'9') => NumberState::Exponent,
            (NumberState::ExponentSign, b'0'..=b'9') => NumberState::Exponent,
            (NumberState::Exponent, b'0'..=b'9') => NumberState::Exponent,
            (
                NumberState::Zero
                | NumberState::Integer
                | NumberState::Fraction
                | NumberState::Exponent,
                _,
            ) if self.is_complete() => return NumberStep::End,
            _ => return NumberStep::Invalid,
        };
        if self.raw.len() >= 64 {
            return NumberStep::Invalid;
        }
        self.raw.push(char::from(byte));
        self.state = next;
        NumberStep::Continue(self.clone())
    }

    fn is_complete(&self) -> bool {
        matches!(
            self.state,
            NumberState::Zero
                | NumberState::Integer
                | NumberState::Fraction
                | NumberState::Exponent
        ) && self.raw.parse::<f64>().is_ok_and(f64::is_finite)
    }
}

fn integer_prefix_can_match(prefix: u64, digits: usize, minimum: u64, maximum: u64) -> bool {
    if digits == 0 {
        return true;
    }
    if prefix == 0 {
        return digits == 1 && minimum == 0;
    }
    let maximum_digits = maximum.to_string().len();
    if digits > maximum_digits {
        return false;
    }
    (0..=maximum_digits - digits).any(|suffix_digits| {
        let Some(scale) = 10_u64.checked_pow(suffix_digits as u32) else {
            return false;
        };
        let Some(low) = prefix.checked_mul(scale) else {
            return false;
        };
        let Some(high) = low.checked_add(scale - 1) else {
            return false;
        };
        low <= maximum && high >= minimum
    })
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn schema_error(message: &str) -> DecodeError {
    DecodeError::Model(message.into())
}

#[cfg(test)]
pub(crate) const TEST_INDEPENDENT_READ_KINDS: &[&str] = &["read_file", "fetch_public"];

#[cfg(test)]
pub(crate) fn test_schema() -> Value {
    serde_json::json!({
        "type": "object", "additionalProperties": false,
        "required": ["goal", "answer", "actions"],
        "properties": {
            "goal": {"type": "string", "minLength": 1, "maxLength": 4096},
            "answer": {"type": "string", "maxLength": 65536},
            "actions": {"type": "array", "maxItems": 8, "items": {"anyOf": [
                {"type": "object", "additionalProperties": false, "required": ["kind", "payload"],
                 "properties": {"kind": {"type": "string", "enum": ["read_file"]},
                                "payload": {"type": "object", "additionalProperties": false,
                                  "required": ["max_bytes", "path"], "properties": {
                                    "max_bytes": {"type": "integer", "minimum": 1, "maximum": 65536},
                                    "path": {"type": "string", "minLength": 1, "maxLength": 4096}}}}},
                {"type": "object", "additionalProperties": false, "required": ["kind", "payload"],
                 "properties": {"kind": {"type": "string", "enum": ["fetch_public"]},
                                "payload": {"type": "object", "additionalProperties": false,
                                  "required": ["max_bytes", "url"], "properties": {
                                    "max_bytes": {"type": "integer", "minimum": 1, "maximum": 65536},
                                    "url": {"type": "string", "minLength": 1, "maxLength": 4096}}}}}
            ]}}
        }
    })
}

#[cfg(test)]
mod tests {
    use super::PlannerSchemaPrefix;
    use serde_json::Value;

    fn new_prefix() -> crate::DecodeResult<PlannerSchemaPrefix> {
        PlannerSchemaPrefix::new(&super::test_schema(), super::TEST_INDEPENDENT_READ_KINDS)
    }

    fn viable(input: &[u8]) -> bool {
        let Ok(mut state) = new_prefix() else {
            return false;
        };
        state.push(input).is_ok() && state.is_viable()
    }

    fn complete(input: &[u8]) -> bool {
        let Ok(mut state) = new_prefix() else {
            return false;
        };
        state.push(input).is_ok() && state.is_complete()
    }

    #[test]
    fn canonical_answer_and_single_action_match_the_shared_schema() {
        let answer = br#"{"actions":[],"answer":"Ready","goal":"Explain"}"#;
        let action = br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":64,"path":"/tmp/input.txt"}}],"answer":"","goal":"Read"}"#;
        for output in [answer.as_slice(), action.as_slice()] {
            let mut state = new_prefix().unwrap();
            for byte in output {
                state.push(&[*byte]).unwrap();
                assert!(state.is_viable());
            }
            assert!(state.is_complete());
            assert!(serde_json::from_slice::<Value>(output).is_ok());
        }
    }

    #[test]
    fn bounded_independent_read_batches_are_generated_from_shared_tool_set() {
        let output = br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":64,"path":"/tmp/a"}},{"kind":"fetch_public","payload":{"max_bytes":128,"url":"https://example.com"}}],"answer":"","goal":"Read and fetch"}"#;
        assert!(complete(output));
        assert!(serde_json::from_slice::<Value>(output).is_ok());
    }

    #[test]
    fn impossible_keys_kinds_and_numeric_prefixes_are_pruned_early() {
        assert!(!viable(br#"{"goal""#));
        assert!(!viable(br#"{"actions":[{"kind":"shell""#));
        assert!(!viable(
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":0"#
        ));
        assert!(!viable(
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":64,"target""#
        ));
    }

    #[test]
    fn multiple_mutations_cannot_form_a_planner_batch() {
        let prefix = br#"{"actions":[{"kind":"write_file","payload":{"content":"x","overwrite":false,"path":"/tmp/out"}},"#;
        assert!(!viable(prefix));
    }

    #[test]
    fn answer_and_action_completion_modes_cannot_be_mixed() {
        assert!(!complete(
            br#"{"actions":[],"answer":"","goal":"No answer"}"#
        ));
        assert!(!complete(
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":64,"path":"/tmp/a"}}],"answer":"also answer","goal":"Read"}"#
        ));
    }

    #[test]
    fn schema_string_bounds_count_unicode_scalars() {
        let mut literal = super::SchemaString {
            started: false,
            mode: super::StringMode::Normal,
            length: 0,
            minimum: 1,
            maximum: 1,
        };
        for byte in b"\"".iter().chain("😀".as_bytes()) {
            assert_eq!(literal.push(*byte), Some(false));
        }
        assert_eq!(literal.length, 1);
        assert_eq!(literal.push(b'"'), Some(true));

        let mut escaped = super::SchemaString {
            started: false,
            mode: super::StringMode::Normal,
            length: 0,
            minimum: 1,
            maximum: 1,
        };
        for byte in br#""\uD83D\uDE00""# {
            let result = escaped.push(*byte);
            assert!(result.is_some());
        }
        assert_eq!(escaped.length, 1);
    }

    #[test]
    fn first_byte_prefilter_never_rejects_an_extendable_schema_transition() {
        let prefixes: [&[u8]; 6] = [
            b"",
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":"#,
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":64,"path":"#,
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":64,"path":"/tmp/"#,
            br#"{"actions":[],"answer":"Ready","goal":"#,
            br#"{"actions":[{"kind":"read_file","payload":{"max_bytes":64,"path":"/tmp/a"}}],"answer":"","goal":"#,
        ];

        for prefix in prefixes {
            let mut state = new_prefix().unwrap();
            state.push(prefix).unwrap();
            for byte in u8::MIN..=u8::MAX {
                let mut candidate = state.clone();
                let extendable = candidate.push(&[byte]).is_ok();
                assert!(
                    !extendable || state.could_accept_first_byte(byte),
                    "prefilter rejected byte {byte:#04x} after {:?}",
                    String::from_utf8_lossy(prefix)
                );
            }
        }
    }
}
