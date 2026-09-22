//! Bounded SSE decoding. Partial text is presentation only; tools are parsed
//! and validated only after a normally completed response.
use crate::{CoreError, CoreResult};

#[derive(Default)]
pub struct CompletionStream {
    buffer: Vec<u8>,
    content: String,
    received: usize,
    stopped: bool,
    done: bool,
}
impl CompletionStream {
    pub fn push(&mut self, bytes: &[u8]) -> CoreResult<Option<String>> {
        self.received = self.received.saturating_add(bytes.len());
        if self.received > 1024 * 1024 {
            return Err(error("Provider stream exceeded one MiB"));
        }
        self.buffer.extend_from_slice(bytes);
        while let Some(end) = self.buffer.iter().position(|b| *b == b'\n') {
            let line = self.buffer.drain(..=end).collect::<Vec<_>>();
            let line = std::str::from_utf8(&line)
                .map_err(|_| error("Invalid stream encoding"))?
                .trim();
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if self.done {
                return Err(error("Content followed a completed stream"));
            }
            if data == "[DONE]" {
                self.done = true;
                continue;
            }
            let value: serde_json::Value =
                serde_json::from_str(data).map_err(|_| error("Malformed provider stream"))?;
            if value.get("error").is_some() {
                return Err(error("Provider reported a streaming error"));
            }
            if let Some(choices) = value["choices"].as_array() {
                for choice in choices {
                    if choice["index"].as_u64().unwrap_or(0) != 0 {
                        return Err(error("Multiple response choices are unsupported"));
                    }
                    if let Some(text) = choice["delta"]["content"].as_str() {
                        self.content.push_str(text);
                    }
                    if choice["delta"].get("tool_calls").is_some() {
                        return Err(error("Unstructured tool calls are unsupported"));
                    }
                    if let Some(reason) = choice["finish_reason"].as_str() {
                        if reason != "stop" {
                            return Err(error("Provider response did not finish normally"));
                        }
                        self.stopped = true;
                    }
                }
            }
        }
        Ok(answer_prefix(&self.content))
    }
    pub fn finish(self) -> CoreResult<String> {
        if !self.done || !self.stopped || !self.buffer.iter().all(u8::is_ascii_whitespace) {
            return Err(error("Provider stream ended before completion"));
        }
        Ok(self.content)
    }
}

fn error(message: &str) -> CoreError {
    CoreError::Model(message.into())
}

/// Find only a root-level answer string, never text inside a tool payload.
pub(crate) fn answer_prefix(input: &str) -> Option<String> {
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
    fn fragmented_unicode_stream_requires_normal_completion() {
        let content = r#"{"goal":"hello","answer":"Hi 🌿","actions":[]}"#;
        let events = format!(
            "data: {}\n\ndata: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n",
            serde_json::json!({"choices":[{"index":0,"delta":{"content":content}}]})
        );
        let mut stream = CompletionStream::default();
        let mut preview = None;
        for byte in events.bytes() {
            if let Some(text) = stream.push(&[byte]).unwrap() {
                preview = Some(text);
            }
        }
        assert_eq!(preview.as_deref(), Some("Hi 🌿"));
        assert_eq!(stream.finish().unwrap(), content);
        assert!(CompletionStream::default().finish().is_err());
    }
    #[test]
    fn partial_answer_does_not_extract_tool_or_escaped_content() {
        assert_eq!(answer_prefix(r#"{"answer":"hel"#).as_deref(), Some("hel"));
        assert_eq!(
            answer_prefix(r#"{"actions":[{"payload":{"answer":"bad"}}]}"#),
            None
        );
        assert_eq!(answer_prefix(r#"{"goal":"\"answer\":\"bad\""#), None);
    }
}
