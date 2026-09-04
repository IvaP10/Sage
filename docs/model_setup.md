# Model setup

Sage has independent provider roles:

- reasoning and structured planning;
- vision;
- speech recognition;
- speech synthesis;
- embeddings.

The shared interfaces are in **crates/sage-core/src/model.rs**. A provider descriptor declares its ID, display name, local/cloud placement, and supported roles.

## Current behavior

The default sage-core executable installs the provider-neutral `OpenAICompatibleProvider`.
It supports the OpenAI preset and OpenAI-compatible `/v1/chat/completions`
endpoints, including exact loopback HTTP addresses for local runtimes such as
Ollama. No provider is active until a user selects it in Settings.

The adapter always requests JSON Schema structured output. A response that is
not valid Sage draft JSON is rejected; it never falls back to prose or shell
text. The model supplies only a draft. Sage assigns action IDs and provenance,
then runs the normal graph, policy, capability, approval, observation, and
verification checks.

## Provider requirements

A reasoning adapter must:

1. accept PlanningContext;
2. keep trusted constraints separate from untrusted context;
3. return one strict ActionGraph;
4. use the core-provided task ID;
5. generate unique action IDs;
6. declare dependencies;
7. provide ExpectedOutcome for every action;
8. treat tool descriptors as schemas, not authority;
9. return errors rather than prose when structured output is invalid;
10. implement bounded replan.

The core validates the result again. Provider validation is not the security boundary.

## Local providers

A local provider may run as a separate model runtime process. It should expose a typed adapter rather than link inference into the native UI. Model crashes and restarts must not corrupt SQLite task state.

Local does not mean trusted. A compromised or hallucinating local model still passes through policy and capabilities.

## Cloud providers

A cloud provider adapter must:

- use HTTPS for public endpoints;
- disable unrelated redirects;
- enforce the configured origin;
- retrieve credentials from SecretStore only for the request;
- avoid logging headers or response bodies containing secrets;
- label all provider output as model provenance;
- apply timeouts and response-size limits;
- support cancellation.

Provider credentials never belong in product.toml, SQLite, source files, environment examples, or command arguments.

## Native settings

macOS and Windows expose matching Provider, Model, Endpoint, API key, Save, and
Test controls. Provider records are stored without secrets; credentials remain
in Keychain or Windows Credential Manager. The Test action sends a temporary
credential to the core and requires the same structured-output contract without
saving it. Anthropic and Google are intentionally not offered in this preview;
legacy records remain visible as unconfigured until reselected.

Provider calls stay in Rust. Do not reintroduce an Electron settings bridge or
put provider requests in Swift/C#.
