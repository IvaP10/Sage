const SECRET_PREFIXES: &[&str] = &["sk-", "ghp_", "github_pat_", "xoxb-", "xoxp-", "AKIA"];
const SECRET_LABELS: &[&str] = &[
    "api_key",
    "apikey",
    "api key",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
];

pub fn redact_for_persistence(input: &str) -> String {
    input
        .split_inclusive('\n')
        .map(redact_line)
        .collect::<Vec<_>>()
        .concat()
}

fn redact_line(line: &str) -> String {
    let body = line.trim_end_matches(['\r', '\n']);
    let ending = &line[body.len()..];
    for separator in ['=', ':'] {
        if let Some((name, value)) = body.split_once(separator) {
            // A mention of "token" in prose is not a secret assignment.
            let label = name
                .trim()
                .trim_start_matches("export ")
                .trim_matches(['\'', '"'])
                .to_ascii_lowercase();
            if SECRET_LABELS.contains(&label.as_str()) {
                let spacing = &value[..value.len() - value.trim_start().len()];
                return format!("{name}{separator}{spacing}[REDACTED]{ending}");
            }
        }
    }
    let mut output = String::with_capacity(line.len());
    let mut copied = 0;
    for (start, _) in body.char_indices() {
        if start < copied {
            continue;
        }
        let tail = &body[start..];
        let boundary = start == 0
            || body[..start]
                .chars()
                .next_back()
                .is_some_and(|previous| !previous.is_alphanumeric() && previous != '_');
        if !boundary {
            continue;
        }
        if let Some(prefix) = SECRET_PREFIXES
            .iter()
            .find(|prefix| tail.starts_with(**prefix))
        {
            let end = tail
                .find(|c: char| {
                    c.is_whitespace()
                        || matches!(c, '\'' | '"' | '<' | '>' | ')' | ']' | '}' | ',' | ';')
                })
                .unwrap_or(tail.len());
            if end > prefix.len() + 8 {
                output.push_str(&body[copied..start]);
                output.push_str("[REDACTED]");
                copied = start + end;
            }
        }
    }
    output.push_str(&body[copied..]);
    output.push_str(ending);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_labeled_and_prefixed_secrets() {
        assert_eq!(
            redact_for_persistence("api_key=abc123"),
            "api_key=[REDACTED]"
        );
        assert_eq!(
            redact_for_persistence("use sk-123456789012345 now"),
            "use [REDACTED] now"
        );
    }

    #[test]
    fn nonsecret_code_unicode_spacing_and_line_endings_are_preserved() {
        let original =
            "  def render():\r\n\treturn 'α  β'\r\n\nExplain tokens: these are pieces of text.\n";
        assert_eq!(redact_for_persistence(original), original);
    }

    #[test]
    fn removes_secret_spans_without_flattening_surrounding_content() {
        assert_eq!(
            redact_for_persistence("  call(\"sk-123456789012345\")\n\tvalue  =  42\n"),
            "  call(\"[REDACTED]\")\n\tvalue  =  42\n"
        );
        assert_eq!(
            redact_for_persistence("\tpassword:  private\r\n"),
            "\tpassword:  [REDACTED]\r\n"
        );
    }
}
