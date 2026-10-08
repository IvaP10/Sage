//! Sage-owned Unicode 15.0.0 normalization and property lookups for Qwen BPE.
//! The data tables are generated from the Python standard-library UCD at
//! development time; no Unicode or tokenizer runtime crate is used.

use crate::{TokenizerError, TokenizerResult};

#[path = "unicode15_tables.rs"]
mod tables;

pub(crate) fn data_version() -> &'static str {
    tables::UNICODE_DATA_VERSION
}

pub(crate) fn is_letter_mark_or_number(character: char) -> bool {
    in_ranges(character as u32, tables::LETTER_MARK_NUMBER_RANGES)
}

pub(crate) fn is_mark(character: char) -> bool {
    in_ranges(character as u32, tables::MARK_RANGES)
}

pub(crate) fn is_letter(character: char) -> bool {
    in_ranges(character as u32, tables::LETTER_RANGES)
}

pub(crate) fn is_number(character: char) -> bool {
    in_ranges(character as u32, tables::NUMBER_RANGES)
}

pub(crate) fn is_whitespace(character: char) -> bool {
    in_ranges(character as u32, tables::WHITE_SPACE_RANGES)
}

fn in_ranges(codepoint: u32, ranges: &[(u32, u32)]) -> bool {
    ranges
        .binary_search_by(|(start, end)| {
            if codepoint < *start {
                std::cmp::Ordering::Greater
            } else if codepoint > *end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

pub(crate) fn normalize_nfc(input: &str) -> TokenizerResult<String> {
    if input.len() > 256 * 1024 {
        return Err(TokenizerError::new(
            "Tokenizer input exceeds Sage's 256 KiB text bound",
        ));
    }
    let mut decomposed = Vec::with_capacity(input.chars().count());
    for character in input.chars() {
        decompose(character as u32, &mut decomposed);
        if decomposed.len() > 1024 * 1024 {
            return Err(TokenizerError::new(
                "Unicode normalization exceeds Sage's scalar bound",
            ));
        }
    }
    canonical_order(&mut decomposed);
    let composed = canonical_compose(&decomposed);
    let mut output = String::with_capacity(input.len());
    for codepoint in composed {
        output.push(char::from_u32(codepoint).ok_or_else(|| {
            TokenizerError::new("Unicode normalization produced an invalid scalar")
        })?);
    }
    Ok(output)
}

fn decompose(codepoint: u32, output: &mut Vec<u32>) {
    const S_BASE: u32 = 0xAC00;
    const L_BASE: u32 = 0x1100;
    const V_BASE: u32 = 0x1161;
    const T_BASE: u32 = 0x11A7;
    const L_COUNT: u32 = 19;
    const V_COUNT: u32 = 21;
    const T_COUNT: u32 = 28;
    const N_COUNT: u32 = V_COUNT * T_COUNT;
    const S_COUNT: u32 = L_COUNT * N_COUNT;

    if (S_BASE..S_BASE + S_COUNT).contains(&codepoint) {
        let syllable = codepoint - S_BASE;
        output.push(L_BASE + syllable / N_COUNT);
        output.push(V_BASE + (syllable % N_COUNT) / T_COUNT);
        let trailing = syllable % T_COUNT;
        if trailing != 0 {
            output.push(T_BASE + trailing);
        }
        return;
    }

    if let Ok(index) = tables::CANONICAL_DECOMPOSITION_INDEX
        .binary_search_by_key(&codepoint, |(character, _, _)| *character)
    {
        let (_, offset, length) = tables::CANONICAL_DECOMPOSITION_INDEX[index];
        for component in &tables::CANONICAL_DECOMPOSITION_VALUES
            [offset as usize..offset as usize + length as usize]
        {
            decompose(*component, output);
        }
    } else {
        output.push(codepoint);
    }
}

fn canonical_order(values: &mut [u32]) {
    let mut marks_start = 0;
    for index in 0..=values.len() {
        if index == values.len() || combining_class(values[index]) == 0 {
            values[marks_start..index].sort_by_key(|codepoint| combining_class(*codepoint));
            marks_start = index + 1;
        }
    }
}

fn canonical_compose(values: &[u32]) -> Vec<u32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut output: Vec<u32> = Vec::with_capacity(values.len());
    let mut starter_position = None;
    let mut starter = 0;
    let mut last_class = 0;

    for codepoint in values.iter().copied() {
        let current_class = combining_class(codepoint);
        if let Some(position) = starter_position
            && (last_class < current_class || last_class == 0)
            && let Some(composite) = compose_pair(starter, codepoint)
        {
            output[position] = composite;
            starter = composite;
            continue;
        }
        if current_class == 0 {
            starter_position = Some(output.len());
            starter = codepoint;
        }
        last_class = current_class;
        output.push(codepoint);
    }
    output
}

fn combining_class(codepoint: u32) -> u8 {
    tables::COMBINING_CLASSES
        .binary_search_by_key(&codepoint, |(character, _)| *character)
        .map(|index| tables::COMBINING_CLASSES[index].1)
        .unwrap_or(0)
}

fn compose_pair(left: u32, right: u32) -> Option<u32> {
    const S_BASE: u32 = 0xAC00;
    const L_BASE: u32 = 0x1100;
    const V_BASE: u32 = 0x1161;
    const T_BASE: u32 = 0x11A7;
    const L_COUNT: u32 = 19;
    const V_COUNT: u32 = 21;
    const T_COUNT: u32 = 28;
    const N_COUNT: u32 = V_COUNT * T_COUNT;
    const S_COUNT: u32 = L_COUNT * N_COUNT;

    if (L_BASE..L_BASE + L_COUNT).contains(&left) && (V_BASE..V_BASE + V_COUNT).contains(&right) {
        return Some(S_BASE + ((left - L_BASE) * V_COUNT + right - V_BASE) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&left)
        && (left - S_BASE).is_multiple_of(T_COUNT)
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&right)
    {
        return Some(left + right - T_BASE);
    }

    let key = (u64::from(left) << 21) | u64::from(right);
    tables::COMPOSITION_PAIRS
        .binary_search_by_key(&key, |(pair, _)| *pair)
        .ok()
        .map(|index| tables::COMPOSITION_PAIRS[index].1)
}
