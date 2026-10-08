# Sage

Sage is a local-first desktop agent with a SwiftUI/AppKit Mac client, a WinUI Windows client and a Rust authority broker. **The native runtime rebuild is in progress; this checkout is not a qualified production release.** The [implementation ledger](docs/v3/implementation-ledger.md) tracks current work and evidence.

Every executable request follows the same authority path. Model inference remains unavailable in product startup until Sage's first-party model package and isolated worker pass their admission gates:

```text
Native intent and explicit scope
  → permission-filtered context → supported local intent or admitted local planner
  → typed action proposal → prepared target and preview → policy/approval
  → single-use execution grant → execution → independent observation
  → verification and encrypted journal → next conversation turn
```

Models are untrusted planners. They cannot grant access, install policies, supply their own success test or silently send local data to a provider.

## Current capabilities

- Bounded model-free local intent paths, iterative verified tool results, cancellation and task continuation.
- Explicit file read/create scopes, bounded directory discovery, protected internal assets, handle-based operations, atomic replacement and guarded Undo.
- Public HTTPS fetch with bounded bodies, public-address validation and DNS pinning.
- User-paired browser navigation bound to the origin, tab, frame and document generation.
- Mac application launches bound to a prepared bundle path and code signature. Windows application launches remain unavailable until a signed-identity adapter is implemented.
- SQLCipher history and recovery artifacts, scoped FTS memory, lineage-aware forgetting, reviewed skill drafts and scoped durable schedules.
- Protocol v2 with role-separated mutual authentication, exact approvals, worker-bound grants, first-party fail-closed authorization and protected audit checkpoints.

Product startup selects `UnconfiguredModelProvider`, so model-generated answers and plans are unavailable. The Qwen3.5 implementation is an offline evaluation candidate: no real checkpoint has been admitted, and its persistent generation lane is still in-process rather than an isolated worker. Hosted model APIs and external inference runtimes are disabled. Arbitrary code execution, generic UI mutations, privileged operations, WASM/MCP tools and full voice qualification remain unavailable; broker/model/network/native process isolation is unfinished.

## Build and run

Shared requirements: Rust **1.89.0**, `protoc`, Clang/libclang for SQLCipher pre-update bindings, Python 3 and Node.js for validation. Mac builds require macOS 14+ and Swift 6.1+; XCTest requires a full Xcode installation. The Windows client needs Windows 10 build 19041+, LLVM with `libclang.dll`, .NET 8 and the Visual Studio Windows application workload; exact package versions live in its project file. If libclang is outside the toolchain's search paths, set `LIBCLANG_PATH` to its containing directory.

```sh
cargo build --workspace --locked
make run-macos
```

The development launcher supplies the built core path. It passes IPC bootstrap material over an anonymous pipe. Startup does not prompt for Keychain access; protected history is unlocked when explicitly requested. Sage's first-party model generator is still under development; [model setup](docs/model_setup.md) records the current availability and migration behavior.

On Windows, from a developer shell:

```powershell
cargo build --workspace --locked
dotnet build apps/windows/Sage.Windows/Sage.Windows.csproj -c Release -p:Platform=x64
```

## Checks and evidence

With full Xcode:

```sh
make verify
```

Individual checks:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
swift build --package-path apps/macos
swift test --package-path apps/macos
node --test integrations/browser/background.test.mjs
python3 -m unittest discover -s scripts -p 'test_*.py'
./scripts/check-repository.sh
python3 scripts/check-v2-release.py
```

On a Mac with Command Line Tools but no XCTest, `python3 scripts/check-macos-signatures.py` exercises read-only signing checks against the compiled production adapter. It does not launch applications or qualify the full native flow. Full XCTest and platform acceptance remain separate requirements.

The fixed [100-workflow specification](evals/workflows.json) requires five measured runs per workflow. `scripts/evaluate-v2-report.py` validates measured reports; it does not fabricate missing results. A passing build or fixture test does not establish actual model inference, Chrome pairing, Windows operation, microphone behavior or production security.

## Source map

| Path | Role |
|---|---|
| `apps/macos`, `apps/windows` | Native clients and current platform adapters |
| `crates/sage-core` | Conversation, authority, context, storage, scheduling and execution routing |
| `crates/sage-protocol`, `proto/sage/ipc/v2/sage.proto` | Canonical protobuf v2 contract |
| `crates/sage-browser-worker`, `integrations/browser` | Role-separated browser host and paired extension |
| `docs`, `evals`, `scripts` | Architecture, implementation records, evaluation and release checks |

VM execution and privileged installation remain unavailable; Sage does not package placeholder helper binaries for those capabilities.

Rust and C# bindings are build-generated. The matching Swift binding is checked in. Regenerate it with protoc-gen-swift **1.38.1**:

```sh
PROTOC_GEN_SWIFT=/absolute/path/to/protoc-gen-swift make protocol
```

The v2 schema is the only protocol source used by the current build. Clients and core must ship together; a protocol mismatch must refuse execution.

## Local state and release

State lives under `~/Library/Application Support/Sage/` on Mac and `%LOCALAPPDATA%\Sage\` on Windows. SQLCipher stores history, memory, task state, audit records and private recovery artifacts. Keys use OS secret storage; legacy provider credentials may remain there but product inference does not read them. Mac transport credentials are separate owner-only `ipc-auth-v2.key` and `browser-ipc-v2.key` files; signed IPC component identity is still a production gate.

Interrupted effects are reconciled before retry. Old grants are revoked on restart. Audit checkpoints detect anchored tampering and truncation; they do not protect against a compromised OS or administrator.

Local preview packages can be built with `make package-macos` or `pwsh -File scripts/package-windows.ps1`. Signing, notarization, installer acceptance and safe updates are separate requirements. No current test result authorizes publication:

```sh
make release-ready
```

This intentionally fails while the [qualification gates](evals/release-gates.json) remain pending. Public tag releases require current evidence; manual development packaging cannot invoke the publication job.

- [Sage architecture and source map](docs/architecture.md)
- [Implementation ledger](docs/v3/implementation-ledger.md)
- [Threat model](docs/v2/threat-model.md)
- [Migration and evaluation](docs/v2/migration-and-evaluation.md)
- [Model setup](docs/model_setup.md)
- [Troubleshooting](docs/troubleshooting.md)
