# Sage native and Rust architecture

This document defines the replacement architecture. It is the design contract for the macOS app, Windows app, Rust core, isolated workers, and any future model or tool integration.

## 1. Non-negotiable separation

Sage is split into native presentation processes and a shared local control plane.

~~~text
┌───────────────────────────────┐       ┌───────────────────────────────┐
│ macOS                         │       │ Windows                       │
│ SwiftUI + AppKit              │       │ C# + WinUI 3                 │
│                               │       │                               │
│ windows, overlay, menu bar,   │       │ windows, overlay, tray,      │
│ shortcuts, microphone UI,     │       │ shortcuts, microphone UI,    │
│ permissions, native auth      │       │ permissions, Windows Hello   │
└───────────────┬───────────────┘       └───────────────┬───────────────┘
                │ Unix Domain Socket                    │ Named Pipe
                └──────────────────┬────────────────────┘
                                   │ protobuf frames + HMAC authentication
                          ┌────────▼─────────┐
                          │ sage-core (Rust) │
                          │ local daemon     │
                          └────────┬─────────┘
                                   │
                ┌──────────────────┼───────────────────┐
                │                  │                   │
         browser worker      sandbox worker    privileged helper
~~~

The native clients may present an approval, run a platform authentication prompt, request a native permission, and render core events. They do not plan agent work, mint capabilities, dispatch tools, mutate task state, or execute a model-proposed action.

The Rust core owns:

- intent and model-provider coordination;
- structured task and action state;
- action compilation;
- provenance and trust classification;
- resource resolution;
- risk, policy, and approval binding;
- permissions and capabilities;
- executor selection;
- observation and verification;
- recovery and undo metadata;
- local persistence and audit records;
- UI/core state synchronization.

## 2. End-to-end task flow

~~~text
User request
   │
   ▼
Native UI sends SubmitTask
   │
   ▼
Authenticated IPC command
   │
   ▼
Core creates durable task record
   │
   ▼
Model provider receives bounded planning context
   │
   ▼
Strict ActionGraph
   │
   ├── schema and dependency validation
   ├── provenance retained
   └── no execution authority
   │
   ▼
Action Compiler creates ordered implementation candidates
   │
   ▼
Resource Resolver canonicalizes exact resources
   │
   ▼
Policy evaluates this action
   │
   ├── deny
   ├── require exact, single-use approval
   └── allow
   │
   ▼
Capability Broker grants one action one resource scope
   │
   ▼
Execution Broker selects native/browser/sandbox/privileged domain
   │
   ▼
Executor consumes capability and acts
   │
   ▼
Observer obtains fresh state
   │
   ▼
Verifier compares state with ExpectedOutcome
   │
   ├── verified: mark action complete, continue
   └── failed: bounded replan, ask user, or fail
~~~

Dispatch is not success. A task can finish only after the verifier accepts the independently observed result of every active action.

## 3. IPC protocol and lifecycle

The canonical schema is **proto/sage/ipc/v1/sage.proto**.

Every frame contains:

- protocol version;
- monotonically increasing connection-local sequence;
- one typed payload.

Payloads cover:

- server challenge and client authentication;
- task submission and control;
- approval decisions;
- user answers;
- permission updates;
- undo;
- snapshots;
- task and agent events;
- model response deltas;
- approval, question, error, notification, and permission requests;
- ping/pong.

### Authentication

1. The core generates a 32-byte server nonce.
2. The UI generates a 32-byte client nonce.
3. The UI computes HMAC-SHA256 over a domain separator, both nonces, protocol version, client kind, and client version.
4. The core checks the proof in constant time.
5. Application messages are rejected before this succeeds.

The installation key is 256 random bits. On macOS the native UI keeps it in an owner-only Application Support file with mode `0600`; this makes IPC authentication available at startup without opening Keychain or showing a permission dialog. On Windows the UI keeps it in Credential Manager. When launching a new core, the UI sends the key through an anonymous stdin pipe and closes that pipe. It is never placed in process arguments, environment variables, SQLite, or logs. Provider credentials remain in Keychain or Credential Manager and are retrieved only for the action that needs them.

On macOS the socket is created with mode 0600. An existing live socket is never deleted; a second core exits. On Windows the core uses a Named Pipe and the first-pipe-instance guard. Authentication remains mandatory even when the transport has local-user ACLs.

The UI may disconnect and reconnect while the core remains alive. On core restart, incomplete tasks become interrupted and do not auto-resume.

## 4. Model boundary

The model interface is **crates/sage-core/src/model.rs**. Reasoning, vision, speech recognition, speech synthesis, and embeddings are independent roles.

The reasoning provider receives:

- the original user request after persistence redaction;
- the core-assigned task ID;
- explicit trusted constraints;
- current observations selected for this task;
- typed tool descriptors;
- untrusted context in a separate field.

It returns an ActionGraph, not executable code. Every action contains:

- a core task ID;
- a fresh action ID;
- a typed Action variant;
- an ExpectedOutcome;
- a target resource;
- provenance;
- metadata.

The interface is vendor-neutral. A provider can be local or cloud. Provider availability does not alter policy. When no provider is configured, the core returns an explicit model error and performs no action.

## 5. Planning and task state

Task state is defined in **domain/task.rs**. A task retains:

- request and goal;
- status;
- action states and attempt counts;
- dependencies;
- created resources;
- summaries and errors;
- final outcome;
- timestamps.

An ActionGraph must be non-empty, use unique IDs, reference the current task, contain only known dependencies, and be acyclic.

The core chooses only dependency-ready actions. Replanning is bounded by configuration. A replan must use fresh action IDs. The failed route remains in history and is marked skipped only when a replacement plan is accepted.

Supported task statuses are pending, planning, running, waiting for approval, waiting for user, paused, succeeded, failed, cancelled, and interrupted.

## 6. Action Compiler

The compiler is **compiler.rs**. It maps a high-level action to ordered implementation candidates:

~~~text
structured application integration
  → accessibility semantics
  → browser DOM
  → validated keyboard shortcut
  → fresh vision
  → fresh coordinates as the final fallback
~~~

Availability is explicit. No missing safe implementation is replaced with a hidden coordinate or shell fallback.

Browser actions remain in the browser execution domain even when its paired browser worker uses an accessibility or visual fallback. That preserves the origin capability and browser-state verification boundary.

## 7. Policy and risk

The policy engine is **policy.rs**. Its categories are:

- safe;
- sensitive;
- consequential;
- destructive;
- privileged;
- prohibited.

Examples:

- reading a private file is sensitive;
- creating or moving a local resource is consequential;
- upload, send, or submit is consequential;
- overwrite or delete is destructive;
- installation or system-setting mutation is privileged;
- unrestricted shell interpreters and credential-store paths are prohibited.

Policy runs for every action. User approval for one action does not authorize a later action. External content cannot carry user authority. Prompt instructions are never a policy primitive.

An approval digest hashes the complete serialized proposal, including action arguments and expected outcome. The approval response must match approval ID, task ID, action ID, and digest. It is consumed once. Privileged approval additionally requires a successful native device-authentication result.

## 8. Capabilities

The Capability Broker is **capability.rs**. A grant is:

- bound to one task;
- bound to one action;
- bound to one executor domain;
- scoped to an exact file, file pair, application, browser origin, command, setting, or user interaction;
- restricted to declared operations;
- short-lived;
- single-use;
- revocable.

Executors validate the resource again. A filesystem executor will reject a grant for another path even if the model action contains that path.

Capabilities are authority objects inside the trusted core and private worker channel. They are not model-visible bearer tokens and are never persisted as reusable authority.

## 9. Resource resolution

The Resource Resolver is **resources.rs**.

Filesystem actions require absolute paths after native resolution. Parent traversal is rejected. Existing resources are canonicalized. A new resource canonicalizes its existing parent before appending its final name. The result must remain under an authorized root.

Default user roots are Desktop, Documents, and Downloads, plus Sage's local data directory. Future folder-picker grants extend this list for a task; they must not grant the entire home directory implicitly.

Protected credential locations such as SSH, GnuPG, cloud CLI credentials, Keychains, and Windows credential stores are prohibited through normal file tools.

## 10. Execution domains

### Native OS executor

The native executor implements exact file operations and delegates application/window/accessibility behavior through a platform adapter. File reads are bounded. New writes use a sibling temporary file. Overwrites create a recovery copy. Delete moves the file into Sage recovery storage instead of immediately destroying it.

macOS platform adapters use Accessibility, Launch Services/AppKit, native file APIs, screen-capture APIs only when authorized, and Apple Events only for an explicit integration. Windows adapters use UI Automation, Win32, COM, and Windows App SDK APIs.

### Browser executor

The browser worker has an origin-bound capability. It must use structured page state and DOM operations where available. Page content is tagged external and untrusted. The worker rejects actions until a user-authorized browser session is paired; it does not silently fall back to unscoped visual control.

### Sandbox executor

The model cannot send a shell string. RunCommand contains one executable, an argument vector, an optional scoped working directory, a network boolean, and a timeout.

On macOS the worker invokes the program under a deny-by-default sandbox profile, clears the environment, limits output, applies a timeout, and allows network only when the capability includes it. On Windows the worker refuses execution until the AppContainer/job-object backend is installed. Running unsandboxed is not a fallback.

### Privileged helper

The privileged helper understands named operations. Its current allowlist contains the shape of a verified application install; the helper refuses it until a signed platform implementation is installed. There is no execute-as-root-string endpoint. The full core never runs as root or administrator.

## 11. Observation and verification

Observation is **observation.rs** and verification is **verification.rs**.

Expected outcomes include:

- file exists or is absent;
- file content hash;
- application running;
- exact browser URL;
- semantic element present;
- command exit code;
- an external success marker;
- a user answer.

For files, the observer re-reads filesystem state after execution and computes a fresh SHA-256 when requested. It does not trust the executor's “success” string. Unsupported application or browser observation must fail closed until its platform observer is connected.

Vision or a verifier model is a fallback only when deterministic evidence cannot establish the result.

## 12. Persistence, memory, and audit

The LocalStore is **storage.rs**. It creates SQLite in WAL mode with foreign keys and a busy timeout.

Tables cover:

- schema migrations;
- tasks and actions;
- events;
- approvals;
- capabilities;
- rollback plans;
- permissions;
- sessions;
- settings;
- structured memory and memory FTS;
- tool registry;
- audit log.

Temporary task context is not automatically promoted into memory. Screen content, document content, and model output do not become persistent memory without an explicit memory rule and policy check.

The audit log stores redacted payloads. Each record hashes its own canonical fields plus the previous record hash, forming a tamper-evident chain. This detects edits; it is not a substitute for OS file protection.

Secrets never enter these tables.

## 13. Events and native UI

The core broadcasts typed events. The native UI renders those events and requests a snapshot after state-changing events rather than reconstructing task truth itself.

UI-visible “thinking” is limited to high-level state such as planning, proposed action, waiting for approval, executing, observing, verifying, replanning, completed, and failed. Hidden chain-of-thought is neither requested nor displayed.

macOS implements:

- SwiftUI main window and settings;
- AppKit floating task overlay;
- menu-bar controls;
- global shortcut surface;
- on-demand Keychain access for provider credentials;
- LocalAuthentication;
- microphone and Accessibility permission UX.

Windows implements:

- WinUI 3 main window and dialogs;
- Named Pipe client;
- Credential Manager;
- UserConsentVerifier for Windows Hello/device authentication;
- Windows-native packaging surface.

## 14. Speech

Speech roles are separate provider traits:

~~~text
microphone → wake word or shortcut → ASR → SubmitTask
                                          │
                                          ▼
                                      sage-core
                                          │
                                          ▼
                                         TTS
~~~

After transcription, voice and typed commands use the same task pipeline. Audio capture and wake detection never bypass policy or gain direct executor access. Raw audio is transient by default.

## 15. Plugins and tools

ToolDescriptor defines:

- name and version;
- input/output schema;
- risk;
- required capabilities;
- supported platforms;
- confirmation behavior;
- executor;
- timeout;
- verification strategy.

A tool registry entry describes a possibility, not trust. Plugin or MCP content is external unless a separate integration policy elevates a narrowly defined response field. Plugins receive only their action's temporary capabilities and required credential handles.

## 16. Recovery and undo

Reversible actions produce a RollbackPlan with expiry. Plans are persisted so a UI crash does not erase them. Current rollback operations cover moving a file back, restoring a backup, and removing an empty folder.

Undo checks that the recovery source still exists and that an inverse destination is safe. It fails instead of overwriting unrelated data.

External messages, remote deletion, purchases, form submission, and publication are not represented as generally reversible. Their irreversibility must be communicated before approval.

## 17. Performance and process isolation

The UI has no Chromium runtime. Model inference, browser work, code execution, indexing, and file processing remain outside UI processes.

The Rust core uses Tokio, event subscriptions, broadcast channels, and bounded timeouts. It avoids screenshot polling. Stable metadata may be cached, but action targets must be refreshed before consequential execution.

A browser or sandbox worker crash returns a structured action failure. It does not crash the UI or corrupt task state.

## 18. Verification boundary

Source inspection and macOS builds can establish:

- Rust type and module correctness on macOS;
- protobuf generation;
- the SwiftUI/AppKit client build;
- core unit tests;
- the end-to-end file action/approval/capability/verification/undo seam.

They do not establish:

- Windows compilation or native runtime without a Windows host;
- UI Automation behavior in third-party apps;
- Windows Hello on a physical Windows device;
- the incomplete Windows AppContainer backend;
- browser pairing against a real browser;
- provider compatibility against a configured model service;
- Developer ID signing, notarization, Authenticode, or public update delivery.

Those are separate acceptance gates and must remain explicit in release claims.
