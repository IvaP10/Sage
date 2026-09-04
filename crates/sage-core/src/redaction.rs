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
        .lines()
        .map(redact_line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn redact_line(line: &str) -> String {
    let lower = line.to_ascii_lowercase();
    if SECRET_LABELS.iter().any(|label| lower.contains(label)) {
        if let Some((name, _)) = line.split_once('=') {
            return format!("{}=[REDACTED]", name.trim_end());
        }
        if let Some((name, _)) = line.split_once(':') {
            return format!("{}: [REDACTED]", name.trim_end());
        }
    }

    let mut words = line
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for word in &mut words {
        if SECRET_PREFIXES
            .iter()
            .any(|prefix| word.starts_with(prefix) && word.len() > prefix.len() + 8)
        {
            *word = "[REDACTED]".into();
        }
    }
    words.join(" ")
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
}
