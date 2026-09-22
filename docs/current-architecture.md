# Current source architecture

Source status: Sage v2 migration, 2026-09-22. See [implementation status](v2/implementation-status.md) for tested and missing behavior. The broker, context builder, network/provider adapter, retrieval and scheduler currently share `sage-core`; the full target process separation is not implemented.

| Source | Responsibility |
|---|---|
| `crates/sage-core/src/main.rs`, `config.rs` | Native launch, non-interactive IPC bootstrap, configured protected paths and optional signed model profile |
| `engine.rs`, `domain/task.rs` | Durable task state, iterative conversation, tool results, scope, cancellation, recovery and budget continuation |
| `contracts.rs`, `journal.rs` | Versioned run/intent/preparation/result/verification contracts and encrypted dispatch journal |
| `features.rs`, `compiler.rs` | Closed feature schemas, supported operation discovery and installed implementation selection |
| `authorization.rs`, `policies/dispatch.cedar`, `policy.rs` | Cedar dispatch decision, independent effect classification, exact previews and approval digests |
| `capability.rs` | Exact action/policy/worker-session binding, single use, expiry and cancellation revocation |
| `resources.rs`, `execution/files.rs` | Explicit roots, protected internal assets, lexical validation and handle-based file identity/operations |
| `execution/native.rs`, `execution/bridge.rs` | Files, public fetch, narrow native service and paired browser dispatch |
| `browser_target.rs`, `integrations/browser/background.js` | Prepared browser document identity and navigation enforcement |
| `observation.rs`, `verification.rs` | Installed operation-specific success conditions and fresh evidence |
| `model.rs`, `streaming.rs`, `network.rs` | Compatible provider requests, bounded answer streaming and constrained egress |
| `inference.rs` | Signed profile validation, memory admission and a restricted on-demand macOS CPU prototype |
| `storage.rs`, `vault.rs`, `audit.rs` | SQLCipher, staged encrypted migration, artifacts, audit-chain verification and OS checkpoints |
| `knowledge.rs`, `context.rs` | Conversations, FTS, scoped context, provenance/lineage and forgetting |
| `workflows.rs` | Reviewed skill drafts, workflow composition, durable schedules and coalesced folder watches |
| `ipc/auth.rs`, `ipc/codec.rs`, `ipc/server.rs` | Mutual role authentication, bounded versioned frames, replay protection and native command/event mapping |
| `proto/sage/ipc/v2/sage.proto`, `crates/sage-protocol` | Canonical protobuf contracts and generated Rust types |
| `apps/macos/Sources/SageMac` | SwiftUI/AppKit presentation, scoped folder selection, protected history unlock, approvals and current platform adapter |
| `apps/windows/Sage.Windows` | WinUI presentation, matching v2 transport/scopes, approvals and current platform adapter |
| `crates/sage-browser-worker` | Role-separated native browser host and typed grant/document handoff |
| `crates/sage-sandbox-worker` | Explicit refusal: the former host sandbox path is removed; VM execution is not installed |
| `crates/sage-privileged-helper` | Refusal until individual signed privileged implementations are installed and validated |

## Available operations

The closed registry exposes `read_file`, `write_file`, `create_folder`, `fetch_public`, `ask_user`, plus `open_application` and `navigate_url` only when the matching adapter is connected. Compilation independently validates the specific registered operation. A connection does not enable generic click/type/upload/submit, command execution or installation.

File grants begin empty. Native UI can select a folder for reading; typed core contracts also support non-overwriting creation grants. Other resources and effects require a prepared-action approval. All externally managed inference endpoints require per-call permission for the exact context/destination, including loopback HTTP servers.

## Persistence and recovery

macOS stores state under `~/Library/Application Support/Sage`; Windows uses `%LOCALAPPDATA%/Sage`. SQLCipher history opens only on an explicit protected-data operation, preserving non-interactive startup. Provider/database/audit secrets use OS storage. macOS transport role credentials are separate mode-0600 v2 files; signed process identity is still a production gate.

Task state and prepared/dispatch records precede effects. Results are confirmed, failed, cancelled or uncertain. Interrupted external effects are re-observed before retry. File reads retain a content hash and encrypted artifact; recovery refuses stale source contents. Undo consumes its recovery dispatch before mutation and validates current identity/content. Old grants never survive a core restart.

Native build success and Rust/JavaScript fixtures are separate from actual provider inference, Chrome pairing, native accessibility, Windows, VM, microphone and distribution acceptance. Consult the [release gates](../evals/release-gates.json).
