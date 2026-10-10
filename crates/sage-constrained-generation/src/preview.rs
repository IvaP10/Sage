//! Safe previews for incomplete Sage planner output.

const PREVIEW_SCAN_INTERVAL_BYTES: usize = 16;

/// Return a larger cumulative answer preview only after a bounded scan interval
/// or a complete JSON value. Generated planner output is append-only, but the
/// previous value is still checked so a malformed prefix cannot retract text.
pub(crate) fn next_answer_preview(
    input: &[u8],
    previous: &str,
    complete: bool,
    last_scan_bytes: &mut usize,
) -> Option<String> {
    if input.len() > crate::MAX_STRUCTURED_JSON_BYTES
        || (!complete && input.len().saturating_sub(*last_scan_bytes) < PREVIEW_SCAN_INTERVAL_BYTES)
    {
        return None;
    }
    *last_scan_bytes = input.len();
    let input = std::str::from_utf8(input).ok()?;
    let answer = answer_prefix(input)?;
    (answer.len() > previous.len() && answer.starts_with(previous)).then_some(answer)
}

/// Find only a root-level answer string, never text inside a tool payload.
pub fn answer_prefix(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut depth = 0i32;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            b'"' => {
                let start = i;
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'"' {
                        break;
                    }
                    i += 1;
                }
                if i >= bytes.len() {
                    return None;
                }
                if depth == 1 && &input[start..=i] == "\"answer\"" {
                    i += 1;
                    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                        i += 1;
                    }
                    if bytes.get(i) != Some(&b':') {
                        continue;
                    }
                    i += 1;
                    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                        i += 1;
                    }
                    if bytes.get(i) != Some(&b'"') {
                        return None;
                    }
                    let value_start = i;
                    i += 1;
                    let mut valid_end = i;
                    while i < bytes.len() {
                        if bytes[i] == b'"' {
                            return serde_json::from_str(&input[value_start..=i]).ok();
                        }
                        if bytes[i] == b'\\' {
                            let count = if bytes.get(i + 1) == Some(&b'u') {
                                6
                            } else {
                                2
                            };
                            if i + count > bytes.len() {
                                break;
                            }
                            i += count;
                        } else {
                            let character = input[i..].chars().next()?;
                            i += character.len_utf8();
                        }
                        valid_end = i;
                    }
                    // A trailing surrogate or incomplete escape is withheld
                    // until its next fragment arrives.
                    for _ in 0..8 {
                        if let Ok(text) = serde_json::from_str::<String>(&format!(
                            "{}\"",
                            &input[value_start..valid_end]
                        )) {
                            return Some(text);
                        }
                        if valid_end <= value_start + 1 {
                            return None;
                        }
                        valid_end -= 1;
                        while !input.is_char_boundary(valid_end) {
                            valid_end -= 1;
                        }
                    }
                    return None;
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_answer_does_not_extract_tool_or_escaped_content() {
        assert_eq!(answer_prefix(r#"{"answer":"hel"#).as_deref(), Some("hel"));
        assert_eq!(
            answer_prefix(r#"{"actions":[{"payload":{"answer":"bad"}}]}"#),
            None
        );
        assert_eq!(answer_prefix(r#"{"goal":"\"answer\":\"bad\""#), None);
    }

    #[test]
    fn cumulative_previews_advance_only_on_monotonic_root_answer_text() {
        let mut last_scan_bytes = 0;
        let first = br#"{"goal":"Read the report","answer":"The report says"#;
        let preview = next_answer_preview(first, "", false, &mut last_scan_bytes).unwrap();
        assert_eq!(preview, "The report says");

        let extended = br#"{"goal":"Read the report","answer":"The report says the first section covers local inference and memory"#;
        let next =
            next_answer_preview(&extended[..], &preview, false, &mut last_scan_bytes).unwrap();
        assert!(next.starts_with(&preview));
        assert!(next.len() > preview.len());

        let retracted = br#"{"goal":"Read the report","answer":"Changed"#;
        assert!(next_answer_preview(&retracted[..], &next, true, &mut last_scan_bytes).is_none());
    }

    #[test]
    fn preview_waits_for_split_utf8_and_never_reads_nested_tool_text() {
        let mut last_scan_bytes = 0;
        let mut split = br#"{"goal":"Summarize","answer":"Cafe "#.to_vec();
        split.extend_from_slice(&[0xe2, 0x82]);
        assert!(next_answer_preview(&split, "", false, &mut last_scan_bytes).is_none());
        split.push(0xac);
        let preview = next_answer_preview(&split, "", true, &mut last_scan_bytes).unwrap();
        assert_eq!(preview, "Cafe €");

        let nested = br#"{"goal":"Inspect","answer":"","actions":[{"kind":"read_file","payload":{"answer":"private tool payload"}}]}"#;
        assert!(next_answer_preview(nested, "", true, &mut last_scan_bytes).is_none());
    }

    #[test]
    fn preview_rescans_are_bounded_by_input_and_scan_interval() {
        let mut last_scan_bytes = 0;
        let short = br#"{"answer":"x"#;
        assert!(next_answer_preview(short, "", false, &mut last_scan_bytes).is_none());
        assert_eq!(last_scan_bytes, 0);

        let oversized = vec![b' '; crate::MAX_STRUCTURED_JSON_BYTES + 1];
        assert!(next_answer_preview(&oversized, "", true, &mut last_scan_bytes).is_none());
        assert_eq!(last_scan_bytes, 0);
    }
}
