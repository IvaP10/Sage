//! Bounded, model-free intent compilation. This grammar proposes ordinary broker
//! actions; it never resolves files, grants authority, or executes an effect.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::contracts::ResourceScope;
use crate::domain::{
    Action, ActionGraph, ActionNode, ActionProposal, ApplicationControlValue, ExpectedOutcome,
    Provenance,
};

pub const MAX_INTENT_BYTES: usize = 4096;
pub const MAX_INTENT_STEPS: usize = 8;

/// Parse one literal assignment to a previously learned control. This grammar
/// only produces typed data; the caller must resolve one current stored
/// capability and submit the result through the normal broker path.
pub(crate) fn parse_control_assignment(text: &str) -> Option<(String, ApplicationControlValue)> {
    if text.len() > MAX_INTENT_BYTES || text.chars().any(char::is_control) {
        return None;
    }
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.len() < 4 || !words[0].eq_ignore_ascii_case("set") {
        return None;
    }
    let value_index = words
        .iter()
        .rposition(|word| word.eq_ignore_ascii_case("to"))?;
    if value_index < 2 || value_index + 2 != words.len() {
        return None;
    }
    let label = words[1..value_index].join(" ");
    if label.trim().is_empty() || label.len() > 128 {
        return None;
    }
    let raw_value = words[value_index + 1];
    let value = if raw_value.eq_ignore_ascii_case("on") || raw_value.eq_ignore_ascii_case("true") {
        ApplicationControlValue::Boolean(true)
    } else if raw_value.eq_ignore_ascii_case("off") || raw_value.eq_ignore_ascii_case("false") {
        ApplicationControlValue::Boolean(false)
    } else {
        ApplicationControlValue::Number(raw_value.parse::<f64>().ok()?)
    };
    value.validate().ok()?;
    Some((label, value))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reflex {
    Stop,
    Hold,
    Correct,
}

/// Only an utterance-leading control phrase is a reflex. Quoted text, a file
/// name, "don't stop", and mentions such as "read stop.txt" are never controls.
pub fn reflex(text: &str) -> Option<Reflex> {
    if text.len() > MAX_INTENT_BYTES {
        return None;
    }
    let text = text.trim_start();
    let text = text.strip_prefix("Sage, ").unwrap_or(text);
    let word_end = text
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(text.len());
    let word = &text[..word_end];
    // Reject a partial word, e.g. "stopwatch", and a quoted/literal prefix.
    if word.eq_ignore_ascii_case("stop") {
        return Some(Reflex::Stop);
    }
    if word.eq_ignore_ascii_case("wait") {
        return Some(Reflex::Hold);
    }
    if word.eq_ignore_ascii_case("no") || word.eq_ignore_ascii_case("actually") {
        return Some(Reflex::Correct);
    }
    let mut words = text.split_whitespace();
    if words.next().is_some_and(|w| w.eq_ignore_ascii_case("hold"))
        && words.next().is_some_and(|w| {
            w.trim_end_matches([',', '.', '!'])
                .eq_ignore_ascii_case("on")
        })
    {
        return Some(Reflex::Hold);
    }
    None
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentStep {
    pub action: Action,
    /// A sequential clause waits for every preceding clause. Parallel clauses
    /// retain the same preceding barrier; admission still checks effect safety.
    pub dependencies: BTreeSet<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledIntent {
    pub steps: Vec<IntentStep>,
}

/// Broker-owned bindings preserve source intent separately from resource
/// canonicalization and native identity metadata added during preparation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentBinding {
    pub action_id: Uuid,
    pub action: Action,
    pub dependencies: BTreeSet<Uuid>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentState {
    pub revision: u64,
    pub steps: Vec<IntentBinding>,
    pub retired: BTreeSet<Uuid>,
    pub change_summary: String,
}

impl CompiledIntent {
    pub fn bind(&self, task_id: Uuid, goal: String) -> (ActionGraph, IntentState) {
        let graph = self.graph(task_id, goal);
        let state = IntentState {
            revision: 1,
            steps: graph
                .nodes
                .iter()
                .map(|node| IntentBinding {
                    action_id: node.proposal.id,
                    action: node.proposal.action.clone(),
                    dependencies: node.depends_on.clone(),
                })
                .collect(),
            retired: BTreeSet::new(),
            change_summary: String::new(),
        };
        (graph, state)
    }
    pub fn graph(&self, task_id: Uuid, goal: String) -> ActionGraph {
        let ids: Vec<_> = self.steps.iter().map(|_| Uuid::new_v4()).collect();
        ActionGraph {
            goal,
            nodes: self
                .steps
                .iter()
                .enumerate()
                .map(|(index, step)| ActionNode {
                    proposal: ActionProposal {
                        id: ids[index],
                        task_id,
                        action: step.action.clone(),
                        // The resource resolver/verifier installs the real outcome.
                        expected_outcome: ExpectedOutcome::UserAnswered,
                        target_resource: String::new(),
                        provenance: Provenance::user(),
                        metadata: BTreeMap::from([("intent_compiler".into(), "1".into())]),
                    },
                    depends_on: step.dependencies.iter().map(|i| ids[*i]).collect(),
                })
                .collect(),
        }
    }

    pub fn summaries(&self) -> Vec<String> {
        self.steps
            .iter()
            .map(|step| summary(&step.action))
            .collect()
    }
}

pub fn summary(action: &Action) -> String {
    match action {
        Action::OpenApplication { application } => format!("Open {application}"),
        Action::ReadFile { path, .. } => format!("Read {}", path.display()),
        Action::ListDirectory { path, .. } => format!("List files in {}", path.display()),
        Action::CreateFolder { path } => format!("Create folder {}", path.display()),
        _ => action.redacted_summary(),
    }
}

/// All-or-nothing compilation is deliberate. An unrecognized suffix such as
/// "tomorrow", "if…", "without…" or "don't…" cannot disappear into a fast path.
pub fn compile(text: &str, scopes: &[ResourceScope]) -> Option<CompiledIntent> {
    if text.len() > MAX_INTENT_BYTES || text.chars().any(char::is_control) {
        return None;
    }
    let text = text.trim();
    if reflex(text).is_some() {
        return None;
    }
    let text = strip_prefix_ascii(text, "please ").unwrap_or(text);
    let clauses = clauses(text)?;
    let mut steps = Vec::with_capacity(clauses.len());
    let mut barrier = BTreeSet::new();
    for (clause, sequential) in clauses {
        if sequential {
            barrier = (0..steps.len()).collect();
        }
        steps.push(IntentStep {
            action: compile_clause(clause.trim(), scopes)?,
            dependencies: barrier.clone(),
        });
    }
    if steps.is_empty() {
        None
    } else {
        Some(CompiledIntent { steps })
    }
}

/// A preview never reads a target. Complete leading clauses may be prepared,
/// even while the final clause is unfinished. They never authorize execution.
pub fn preview(text: &str) -> Vec<String> {
    preparation(text, &[]).map_or_else(Vec::new, |intent| intent.summaries())
}

pub(crate) fn preparation(text: &str, scopes: &[ResourceScope]) -> Option<CompiledIntent> {
    if let Some(intent) = compile(text, scopes) {
        return Some(intent);
    }
    if text.len() > MAX_INTENT_BYTES || text.chars().any(char::is_control) {
        return None;
    }
    let text = text.trim_end();
    let mut quoted = false;
    let mut result = None;
    let mut boundaries = 0;
    let mut next_boundary = 0;
    for (index, character) in text.char_indices() {
        if character == '"' {
            quoted = !quoted;
        }
        if character != ' ' || quoted || index < next_boundary {
            continue;
        }
        if let Some(delimiter) = [" and then", " then", " and"].iter().find(|delimiter| {
            strip_prefix_ascii(&text[index..], delimiter)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with(' '))
        }) {
            next_boundary = index + delimiter.len();
            boundaries += 1;
            if boundaries >= MAX_INTENT_STEPS {
                return None;
            }
            if let Some(intent) = compile(&text[..index], scopes) {
                result = Some(intent);
            }
        }
    }
    result
}

/// Returns one explicit, local prefix eligible to start while native speech is
/// still arriving. Only app opening and a single-file read are supported here;
/// all other actions wait for the complete request.
pub(crate) fn streamed_action_prefix(
    text: &str,
    scopes: &[ResourceScope],
) -> Option<(CompiledIntent, String)> {
    let text = text.trim_end();
    let prefix_length = text.len().checked_sub(" and then".len())?;
    if !text.is_char_boundary(prefix_length)
        || !text[prefix_length..].eq_ignore_ascii_case(" and then")
    {
        return None;
    }
    let prefix = text[..prefix_length].trim();
    let compiled = compile(prefix, scopes)?;
    if compiled.steps.len() != 1
        || !matches!(
            compiled.steps.first()?.action,
            Action::OpenApplication { .. } | Action::ReadFile { .. }
        )
    {
        return None;
    }
    Some((compiled, prefix.to_string()))
}

fn compile_clause(text: &str, scopes: &[ResourceScope]) -> Option<Action> {
    if let Some(name) =
        strip_prefix_ascii(text, "open ").or_else(|| strip_prefix_ascii(text, "launch "))
    {
        let application = application_name(name.trim())?;
        return Some(Action::OpenApplication {
            application: application.into(),
        });
    }
    if text.eq_ignore_ascii_case("list files") || text.eq_ignore_ascii_case("show files") {
        let [scope] = scopes else {
            return None;
        };
        return Some(Action::ListDirectory {
            path: scope.root.clone(),
            page_size: 32,
            cursor: None,
        });
    }
    for verb in [
        "list files in ",
        "list ",
        "read file ",
        "read ",
        "create folder ",
    ] {
        if let Some(value) = strip_prefix_ascii(text, verb) {
            let path = literal_path(value)?;
            return Some(if verb.starts_with("list") {
                Action::ListDirectory {
                    path,
                    page_size: 32,
                    cursor: None,
                }
            } else if verb.starts_with("read") {
                Action::ReadFile {
                    path,
                    max_bytes: 64 * 1024,
                }
            } else {
                Action::CreateFolder { path }
            });
        }
    }
    None
}

fn literal_path(text: &str) -> Option<PathBuf> {
    // Quotes are mandatory: no guessed boundaries, relative roots, variables,
    // tilde expansion, or string-to-shell conversion.
    let text = text.trim();
    let path = text.strip_prefix('"')?.strip_suffix('"')?;
    if path.is_empty() || path.contains('"') || path.contains('\0') {
        return None;
    }
    let path = PathBuf::from(path);
    path.is_absolute().then_some(path)
}

fn application_name(name: &str) -> Option<&'static str> {
    let name = name.trim_end_matches(['.', '!']);
    [
        ("chrome", "Google Chrome"),
        ("google chrome", "Google Chrome"),
        ("firefox", "Firefox"),
        ("safari", "Safari"),
        ("vs code", "Visual Studio Code"),
        ("vscode", "Visual Studio Code"),
        ("visual studio code", "Visual Studio Code"),
        ("finder", "Finder"),
        ("terminal", "Terminal"),
        ("notes", "Notes"),
        ("mail", "Mail"),
        ("calendar", "Calendar"),
        ("calculator", "Calculator"),
        ("system settings", "System Settings"),
    ]
    .into_iter()
    .find_map(|(alias, app)| name.eq_ignore_ascii_case(alias).then_some(app))
}

fn strip_prefix_ascii<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    text.get(..prefix.len())
        .filter(|s| s.eq_ignore_ascii_case(prefix))?;
    text.get(prefix.len()..)
}

fn clauses(text: &str) -> Option<Vec<(&str, bool)>> {
    let bytes = text.as_bytes();
    let mut quoted = false;
    let mut start = 0;
    let mut i = 0;
    let mut sequential = false;
    let mut result = Vec::new();
    while i < bytes.len() {
        if bytes[i] == b'"' {
            quoted = !quoted;
        }
        if !quoted && bytes[i] == b' ' {
            let tail = &text[i..];
            if let Some((delimiter, next_sequential)) =
                [(" and then ", true), (" then ", true), (" and ", false)]
                    .into_iter()
                    .find(|(delimiter, _)| strip_prefix_ascii(tail, delimiter).is_some())
            {
                result.push((&text[start..i], sequential));
                if result.len() >= MAX_INTENT_STEPS {
                    return None;
                }
                i += delimiter.len();
                start = i;
                sequential = next_sequential;
                continue;
            }
        }
        i += 1;
    }
    if quoted {
        return None;
    }
    result.push((&text[start..], sequential));
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_known_commands_need_no_reasoner() {
        let intent = compile(
            "Please open Chrome and open Firefox then launch VS Code",
            &[],
        )
        .unwrap();
        assert_eq!(intent.steps.len(), 3);
        assert!(intent.steps[0].dependencies.is_empty());
        assert!(intent.steps[1].dependencies.is_empty());
        assert_eq!(intent.steps[2].dependencies, BTreeSet::from([0, 1]));
        let id = Uuid::new_v4();
        intent.graph(id, "Open apps".into()).validate(id).unwrap();
    }

    #[test]
    fn incomplete_ambiguous_and_qualified_commands_never_execute() {
        for text in [
            "open Chrome and",
            "open Chrome and then",
            "open Chrome tomorrow",
            "don't open Chrome",
            "open Chrome unless I say stop",
            "open Chrome and delete files",
            "read /tmp/test",
            "read \"relative\"",
            "open Chrome\nopen Firefox",
            "open Chrome; open Firefox",
            "open Chrome if it is installed",
        ] {
            assert!(compile(text, &[]).is_none(), "{text}");
        }
        assert_eq!(preview("open Chrome and then"), vec!["Open Google Chrome"]);
    }

    #[test]
    fn quoted_paths_do_not_turn_into_instructions() {
        let parsed = compile(
            "read \"/tmp/stop and then open Chrome\" and list \"/tmp/ä files\"",
            &[],
        )
        .unwrap();
        assert_eq!(parsed.steps.len(), 2);
        assert_eq!(
            parsed.steps[0].action,
            Action::ReadFile {
                path: "/tmp/stop and then open Chrome".into(),
                max_bytes: 65536
            }
        );
        assert!(compile("read \"/tmp/file\" tomorrow", &[]).is_none());
        assert!(compile("read \"/tmp/unterminated", &[]).is_none());
    }

    #[test]
    fn preparation_retains_only_complete_leading_clauses() {
        for text in [
            "open Chrome and then read",
            "OPEN CHROME AND THEN read \"/tmp/unfinished",
            "open Chrome and then open the",
        ] {
            assert!(compile(text, &[]).is_none(), "{text}");
            assert_eq!(preview(text), ["Open Google Chrome"], "{text}");
        }
        assert_eq!(
            preview("open Chrome and open Firefox and then read"),
            ["Open Google Chrome", "Open Firefox"]
        );
        assert_eq!(preview("open Notes and then open Notes and then open Notes and then open Notes and then read").len(), 4);
        for text in [
            "don't open Chrome and then read",
            "open Chrome tomorrow and then read",
            "open Chrome android",
            "read \"/tmp/and then open Chrome",
            "open Chrome\nand then read",
        ] {
            assert!(preparation(text, &[]).is_none(), "{text}");
        }
        let scope = ResourceScope {
            root: "/tmp/selected".into(),
            effects: BTreeSet::from([crate::contracts::Effect::Read]),
        };
        assert_eq!(
            preparation("list files and then read", &[scope])
                .unwrap()
                .summaries(),
            ["List files in /tmp/selected"]
        );
    }

    #[test]
    fn streamed_prefix_requires_one_supported_action_and_an_explicit_boundary() {
        let (intent, prefix) = streamed_action_prefix("Open Notes and then", &[]).unwrap();
        assert_eq!(prefix, "Open Notes");
        assert_eq!(intent.summaries(), ["Open Notes"]);
        for text in [
            "open Notes",
            "open Notes then",
            "open Notes and then read",
            "open Notes and open Safari and then",
            "open Notesx and then",
            "open Notes and then tomorrow",
        ] {
            assert!(streamed_action_prefix(text, &[]).is_none(), "{text}");
        }
    }

    #[test]
    fn streamed_prefix_accepts_one_exact_file_read_but_no_mutation_or_directory_listing() {
        let path = std::env::temp_dir().join("sage-streamed-read.txt");
        let request = format!("Read \"{}\" and then", path.display());
        let (intent, prefix) = streamed_action_prefix(&request, &[]).unwrap();
        assert_eq!(prefix, format!("Read \"{}\"", path.display()));
        assert!(matches!(intent.steps[0].action, Action::ReadFile { .. }));

        for action in [
            format!("Create folder \"{}\" and then", path.display()),
            format!("List files in \"{}\" and then", path.display()),
            format!(
                "Read \"{}\" and then read \"{}\" and then",
                path.display(),
                path.display()
            ),
        ] {
            assert!(streamed_action_prefix(&action, &[]).is_none(), "{action}");
        }
    }

    #[test]
    fn reflex_uses_word_boundaries_and_never_literal_mentions() {
        for text in [
            "Stop",
            "stop!",
            "wait",
            "hold on",
            "No, open Firefox",
            "Actually open Notes",
        ] {
            assert!(reflex(text).is_some(), "{text}");
        }
        for text in [
            "don't stop",
            "stopwatch",
            "nobody",
            "read stop.txt",
            "\"stop\"",
            "open Notes",
            "hold only",
        ] {
            assert!(reflex(text).is_none(), "{text}");
        }
    }

    #[test]
    fn parser_is_bounded_and_utf8_safe() {
        assert!(compile(&"x".repeat(MAX_INTENT_BYTES + 1), &[]).is_none());
        assert!(compile(&["open Notes"; 9].join(" and "), &[]).is_none());
        assert!(compile("🦀 and αβγ", &[]).is_none());
        assert!(compile("list files", &[]).is_none());
    }

    #[test]
    fn learned_control_assignment_is_one_bounded_typed_literal() {
        assert_eq!(
            parse_control_assignment("Set Master Volume to 0.5"),
            Some(("Master Volume".into(), ApplicationControlValue::Number(0.5)))
        );
        assert_eq!(
            parse_control_assignment("set Enable Sound to on"),
            Some((
                "Enable Sound".into(),
                ApplicationControlValue::Boolean(true)
            ))
        );
        assert_eq!(
            parse_control_assignment("SET Enable Sound TO off"),
            Some((
                "Enable Sound".into(),
                ApplicationControlValue::Boolean(false)
            ))
        );
        for text in [
            "set Volume to 0.5 tomorrow",
            "set Volume to NaN",
            "set Volume to inf",
            "set Volume to 0.5 and launch Safari",
            "set Volume to 0.5; delete the document",
            "set Volume to -0.5\nthen publish",
            "set Volume",
            "set Volume to true extra",
        ] {
            assert!(parse_control_assignment(text).is_none(), "{text}");
        }
    }
}
