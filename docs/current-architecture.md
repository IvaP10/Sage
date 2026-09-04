# Current source architecture

This document maps the code that exists in this checkout. It replaces the former Electron architecture document.

## Composition roots

- **crates/sage-core/src/main.rs** parses native launch options, obtains the IPC key, constructs the core, and serves the platform endpoint.
- **crates/sage-core/src/engine.rs** is the trusted orchestration composition root.
- **apps/macos/Sources/SageMac/SageApp.swift** is the macOS presentation composition root.
- **apps/windows/Sage.Windows/App.xaml.cs** is the Windows presentation composition root.

There is no Node.js main process, preload bridge, renderer, Vite build, Electron BrowserWindow, or Electron packaging layer.

## Shared Rust workspace

### sage-protocol

**crates/sage-protocol/build.rs** compiles the canonical protobuf schema with prost. **src/lib.rs** exposes generated types, protocol version 1, and the four-megabyte frame ceiling.

### sage-core domain

- **domain/action.rs** contains structured actions, conditions, expected outcomes, executor domains, proposals, nodes, and action graphs.
- **domain/task.rs** contains task/action states, dependency-ready scheduling, initial plan installation, bounded replan installation, and completion rules.
- **domain/provenance.rs** defines sources and separates user authority, trusted components, observations, and untrusted external content.

### Safety and authority

- **policy.rs** validates action limits, classifies risk, prohibits general-purpose shells and protected credential paths, and creates approval digests.
- **resources.rs** canonicalizes file resources and enforces authorized roots.
- **capability.rs** issues and consumes exact, expiring, single-use capabilities.
- **compiler.rs** orders structured integration, accessibility, DOM, shortcut, vision, and coordinate implementations.
- **redaction.rs** removes common credential shapes before persistence.
- **secrets.rs** wraps on-demand provider/standalone credentials through Keychain or Windows Credential Manager and zeroizes in-memory secret containers.

### Orchestration

**engine.rs** implements:

1. request validation and redacted persistence;
2. task creation;
3. planning context construction;
4. ActionGraph validation;
5. dependency scheduling;
6. resource resolution;
7. policy evaluation;
8. approval or question waits;
9. compilation and executor selection;
10. capability issuance;
11. action dispatch;
12. rollback persistence;
13. fresh observation;
14. verification;
15. bounded replan or terminal failure;
16. capability revocation and final outcome.

The module also implements pause, resume, cancel, exact approval resolution, user answers, permission state, snapshots, and Undo.

### Execution

- **execution/mod.rs** contains the executor trait, broker, receipts, and rollback types.
- **execution/native.rs** implements bounded file reads, writes, moves, recoverable deletes, folder creation, and the platform-controller interface.
- **execution/worker.rs** launches one isolated worker request with a cleared environment, framed input/output, response limits, and a timeout.
- **sage-browser-worker** validates origin and capability bindings and currently stops when no structured browser session is paired.
- **sage-sandbox-worker** validates command capability bindings and uses the macOS deny-by-default sandbox backend. Windows stops until AppContainer is implemented.
- **sage-privileged-helper** accepts only a named allowlist shape and stops until its signed platform implementation exists.

### Observation and verification

- **observation.rs** obtains deterministic evidence for file state/hash and structured worker results.
- **verification.rs** compares that evidence with the declared ExpectedOutcome.

Application, browser, and semantic element observation are explicit extension points. No current code treats an unavailable observer as success.

### State and events

- **storage.rs** owns SQLite migrations, WAL configuration, tasks/actions, redacted events, permissions, rollback plans, memory/FTS, tool registry, and the audit chain.
- **events.rs** defines the internal event stream and state snapshot.
- **ipc/auth.rs** defines the HMAC proof.
- **ipc/codec.rs** implements bounded protobuf framing.
- **ipc/server.rs** owns Unix Socket/Named Pipe lifecycle, authentication, command dispatch, event mapping, and lag recovery through snapshots.

### Models and speech

**model.rs** defines vendor-neutral traits for reasoning, ASR, TTS, and vision. The default executable installs UnconfiguredModelProvider, which fails closed until provider configuration is connected. Model absence never weakens policy or enables a direct command path.

## macOS application

- **SageApp.swift** defines the SwiftUI application, main window, menu bar, and commands.
- **AppModel.swift** owns presentation state and translates protobuf events into UI state.
- **SageCoreClient.swift** implements Unix Socket transport, protobuf framing, HMAC authentication, command sending, and streaming receive.
- **CoreSupervisor.swift** locates/launches the bundled core and supplies bootstrap authentication through stdin.
- **IPCSecretStore.swift** stores only the IPC installation key as an owner-only `0600` file, avoiding Keychain UI at startup.
- **NativeAuthentication.swift** uses LocalAuthentication for privileged approval.
- **MainView.swift** renders task history, timeline, composer, approval, question, cancel, and Undo surfaces.
- **SettingsView.swift** renders core, model, microphone, Accessibility, and local-data state.
- **OverlayWindowController.swift** creates the AppKit floating task overlay.
- **GlobalShortcutController.swift** owns the macOS global shortcut surface.
- **Generated/sage/ipc/v1/sage.pb.swift** is generated from the shared proto.

The macOS client does not import or link Rust orchestration code. It only launches and speaks to the separate core process.

## Windows application

- **App.xaml / App.xaml.cs** define the WinUI 3 application.
- **MainWindow.xaml / MainWindow.xaml.cs** render tasks, events, composer, approval, question, and settings.
- **SageCoreClient.cs** implements the Named Pipe, generated protobuf types, HMAC authentication, commands, and event stream.
- **CoreSupervisor.cs** launches the sibling core and supplies bootstrap authentication through stdin.
- **IpcSecretStore.cs** uses Win32 Credential Manager with the same target convention as the Rust keyring backend.
- **NativeAuthentication.cs** uses UserConsentVerifier rather than a Sage-created password or PIN dialog.
- **Sage.Windows.csproj** pins the Windows App SDK, protobuf runtime/generator, architecture targets, and shared proto.

Windows-generated protobuf C# is build output and is not checked in.

## Persistent locations

macOS:

~~~text
~/Library/Application Support/Sage/
  sage.db
  sage.db-wal
  sage.db-shm
  recovery/
  ipc-auth.key
  sage-core.sock
~~~

Windows:

~~~text
%LOCALAPPDATA%\Sage\
  sage.db
  sage.db-wal
  sage.db-shm
  recovery\
~~~

On macOS `ipc-auth.key` is in this directory and is restricted to the current user. On Windows the IPC key is in Credential Manager. Provider secrets remain outside these directories in Keychain or Credential Manager and are loaded only when needed.

## Current task trace

~~~text
MainView / MainWindow
  → SageCoreClient SubmitTask
  → protobuf Frame
  → authenticated socket or pipe
  → ipc/server handle_command
  → SageCore.submit_task
  → ModelProvider.create_plan
  → Task.install_plan
  → ResourceResolver
  → PolicyEngine
  → ApprovalRequested event if necessary
  → native authentication when privileged
  → CapabilityBroker.issue
  → ExecutionBroker
  → executor
  → Observer
  → Verifier
  → SQLite + audit
  → CoreEvent
  → native timeline/snapshot
~~~

## Current tested seam

The Rust end-to-end test **approval_capability_execution_verification_and_undo_are_end_to_end** constructs a real temporary database and core, installs a scripted model plan, receives and resolves the exact approval, creates a folder through the native executor, observes it from the filesystem, verifies the condition, reaches TaskCompleted, loads the succeeded task, invokes Undo, and verifies that the folder is gone.

This test proves the shared control-plane seam. It does not prove a third-party desktop application, live browser, configured LLM, or Windows-native behavior.

## Deliberate fail-closed gaps

- No reasoning provider is configured by default.
- Browser execution requires a paired structured session.
- The privileged helper has no installed signed operation.
- The Windows sandbox has no AppContainer backend.
- Platform application/accessibility adapters are not yet connected to NativeExecutor.
- Application/browser observation is not synthesized when unavailable.

These gaps are visible and return errors. They are not replaced with the former Electron adapters, arbitrary commands, raw coordinates, or prompt-based permission.
