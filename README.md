# Sage

Sage is a native desktop AI agent for macOS and Windows. The product is no longer an Electron application: macOS uses SwiftUI and AppKit, Windows uses C# and WinUI 3, and both clients connect to one long-running Rust control plane named **sage-core**.

The language model is an untrusted planner. It never receives an operating-system handle, a shell, a browser debugging connection, or a permanent filesystem grant. Every proposed action travels through resource resolution, policy, a short-lived capability, a domain-specific executor, a fresh observation, and verification.

~~~text
macOS: SwiftUI + AppKit ─┐
                         ├── authenticated local IPC ── sage-core (Rust)
Windows: C# + WinUI 3 ───┘
                                                     │
                       model proposal ──> action compiler
                                                     │
                        policy ──> capability broker ─┤
                                                     │
                   native / browser / sandbox / privileged executor
                                                     │
                              observation ──> verification
~~~

## Repository layout

~~~text
apps/
  macos/                         SwiftUI/AppKit client
  windows/                       WinUI 3 client
crates/
  sage-core/                     orchestration and security control plane
  sage-protocol/                 generated Rust protobuf contract
  sage-worker-common/            private framed worker contract
  sage-browser-worker/           isolated structured-browser boundary
  sage-sandbox-worker/           isolated command/code boundary
  sage-privileged-helper/        narrow privileged-operation allowlist
proto/sage/ipc/v1/sage.proto     canonical UI/core protocol
scripts/                         generation, verification, and packaging
docs/                            architecture, security, and development guides
~~~

The native clients contain presentation, window and overlay behavior, native approval/authentication prompts, shortcuts, menu-bar or tray surfaces, and operating-system permission UX. Planning, action state, policy, capabilities, execution routing, persistence, audit records, recovery, and verification live in **sage-core**.

## Implemented control-plane guarantees

- Protobuf-framed IPC over a mode-0600 Unix Domain Socket on macOS and a Named Pipe on Windows
- Challenge-response HMAC authentication using a 256-bit installation key held in an owner-only macOS file or Windows Credential Manager
- A version field on every frame and an explicit protocol compatibility handshake
- Strict structured actions instead of arbitrary shell strings
- Action graphs with dependencies, explicit state, bounded replanning, pause/cancel, approvals, and final outcomes
- Provenance and trust classes that keep user authority separate from websites, documents, messages, terminal output, and other external data
- Deterministic risk classification and per-action policy evaluation below the model
- Approval digests bound to the exact task, action, arguments, expected outcome, and resource
- Single-use, task-bound, action-bound, executor-bound capabilities with expiration and revocation
- Native, browser, sandbox, privileged, and user-interaction execution domains
- Canonical filesystem resolution with authorized-root and symlink-escape checks
- Recoverable local deletion, rollback metadata, and an explicit Undo path
- Fresh post-action observation and deterministic verification
- SQLite WAL storage, FTS-backed local memory tables, interrupted-task marking, and a tamper-evident audit hash chain
- OS-backed secret storage; secrets are never written to SQLite or plaintext configuration
- Isolated browser, sandbox, and privileged worker processes that fail closed when a safe backend or paired session is unavailable

The current browser worker intentionally refuses to act until an authenticated structured browser session is paired. The privileged helper intentionally refuses installation until a signed platform implementation is installed. The Windows sandbox intentionally refuses command execution until its AppContainer backend is present. These are security gates, not coordinate or unsandboxed fallbacks.

## Requirements

Shared:

- Rust 1.85.1
- Protocol Buffers compiler (protoc)

macOS:

- macOS 14 or newer
- Xcode/Swift 6.1 or newer
- protoc-gen-swift 1.38.1 only when regenerating the checked-in Swift binding

Windows:

- Windows 10 19041 or newer
- Visual Studio 2022 with the Windows application development workload
- .NET 8 SDK
- Windows App SDK 2.4

## Build and verify

The first command checks formatting, Clippy, Rust tests, Rust targets, the native macOS client, generated protocol state, Electron-removal guards, and common credential patterns:

~~~bash
make verify
~~~

Build the Rust daemon and isolated workers:

~~~bash
cargo build --workspace
~~~

Build the macOS native client:

~~~bash
swift build --package-path apps/macos
~~~

Run the native macOS app from source:

~~~bash
cargo build --workspace
make run-macos
~~~

The run-macos target gives the Swift client the development sage-core location. The client supplies the IPC installation key through an anonymous stdin pipe when it launches a new core; it never places the key in arguments, environment variables, or SQLite. On macOS the transport key is a 32-byte owner-only file with mode `0600`, so opening Sage never presents Keychain UI. If a core is already serving the authenticated socket, the UI reconnects without replacing it.

Build the Windows client from a Windows developer shell:

~~~powershell
dotnet build apps/windows/Sage.Windows/Sage.Windows.csproj -c Release -p:Platform=x64
~~~

## Protocol generation

The file [sage.proto](proto/sage/ipc/v1/sage.proto) is canonical. Rust bindings are generated by prost during Cargo builds, C# bindings are generated by Grpc.Tools during the WinUI build, and the Swift binding is checked in so a source build does not require the generator.

To regenerate Swift after changing the schema:

~~~bash
PROTOC_GEN_SWIFT=/absolute/path/to/protoc-gen-swift make protocol
~~~

Use version 1.38.1 of protoc-gen-swift, matching the pinned SwiftProtobuf runtime.

## Runtime data

Sage remains local-first:

- macOS: ~/Library/Application Support/Sage/
- Windows: %LOCALAPPDATA%\Sage\

The sage.db file contains structured task state, redacted events, settings, permission state, audit records, rollback metadata, and local memory. SQLite uses WAL mode and FTS. On macOS the IPC key is `ipc-auth.key` with mode `0600`; on Windows it remains in Credential Manager. Provider credentials use Keychain or Credential Manager only when the configured provider is invoked.

The UI may restart without killing active core tasks. A core restart marks unfinished tasks **interrupted**; it does not silently resume privileged work.

## Packaging

Create an explicitly unsigned native macOS preview application and DMG:

~~~bash
make package-macos
~~~

Without `SAGE_MACOS_SIGN_IDENTITY`, the package is ad-hoc signed and
unnotarized. Gatekeeper may reject that preview. A stable public build requires
a Developer ID identity and notarization:

~~~bash
SAGE_MACOS_SIGN_IDENTITY="Developer ID Application: …" \
SAGE_NOTARY_PROFILE="sage-notary" \
make package-macos
~~~

Create the Windows x64 preview installer on Windows (Inno Setup must be installed):

~~~powershell
pwsh -File scripts/package-windows.ps1
~~~

The preview is an unsigned EXE and SmartScreen may warn. Stable Windows
distribution still requires Authenticode, a Windows-native packaging run, and
real install/runtime acceptance. macOS source/tests do not prove Windows UI
Automation, Windows Hello, AppContainer, signing, or installation behavior.

## Architecture and security

- [Architecture](docs/architecture.md)
- [Current source map](docs/current-architecture.md)
- [Security model](docs/security.md)
- [Model setup](docs/model_setup.md)
- [Provider development](docs/provider_development.md)
- [Troubleshooting](docs/troubleshooting.md)

The central invariant is:

~~~text
AI decides WHAT to propose.
Policy decides WHETHER it may happen.
Capabilities decide WHICH resources it may access.
The execution broker decides HOW it may happen.
The platform executor performs the operation.
The observer determines WHAT actually happened.
The verifier decides WHETHER the expected result exists.
~~~

Those responsibilities are separate types and modules. A prompt, model response, tool description, or external page can never collapse them into one unrestricted agent loop.
