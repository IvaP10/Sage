# Troubleshooting

## macOS app says sage-core is missing

For source development:

~~~bash
cargo build --workspace
make run-macos
~~~

The packaged app expects sage-core and its workers under Contents/Helpers.

## Local core connected never appears

Check:

~~~bash
ls -l "$HOME/Library/Application Support/Sage/sage-core.sock"
ps aux | rg '[s]age-core'
~~~

The core refuses to replace a live socket. A stale socket is removed only after connection fails with a stale-socket error.

Do not delete the socket while a core is running.

## IPC authentication failed

The UI and core must use the same installation key. On macOS the native UI owns:

~~~text
~/Library/Application Support/Sage/ipc-auth.key
type: regular file owned by the current user
permissions: 0600
value: raw 32-byte key
~~~

On Windows the key remains in Credential Manager under target `local-ipc-v1.com.ivanpadeliya.sage`. For normal native launch, the UI passes the decoded key through stdin. macOS startup does not query Keychain; provider credentials are queried only when a provider action needs them.

Never work around authentication by disabling HMAC or using a fixed key.

## Task fails with no reasoning model configured

This is the current fail-closed default. Read [Model setup](model_setup.md). Do not add direct command parsing as a temporary model substitute.

## Browser action says no session is paired

The browser worker currently enforces the pairing gate. A paired structured browser adapter must be installed before browser actions can run. It must retain exact-origin and verification rules.

## Windows command action refuses AppContainer

The Windows sandbox backend is intentionally incomplete. RunCommand does not fall back to an unsandboxed Process or PowerShell. Implement and validate AppContainer/job-object constraints on a Windows host.

## Privileged install is unavailable

The helper has the operation shape but no signed platform implementation. Do not run the whole app as administrator/root and do not add an arbitrary elevated command endpoint.

## Rust toolchain is missing

Install the pinned toolchain:

~~~bash
rustup toolchain install 1.85.1 --component rustfmt --component clippy
~~~

Then:

~~~bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
~~~

## Swift protobuf regeneration

Source builds use the checked-in generated file. Regeneration needs protoc-gen-swift 1.38.1:

~~~bash
PROTOC_GEN_SWIFT=/absolute/path/to/protoc-gen-swift ./scripts/generate-protocol.sh
~~~

## Windows client cannot build on macOS

WinUI 3 and the Windows App SDK require a Windows build host. Validate on x64 and ARM64 Windows as separate targets. A C# source review is not a Windows build result.

## SQLite reports an interrupted task

At startup the store marks pending, planning, running, waiting, or paused tasks interrupted. This prevents silent resumption after a crash or sleep/restart boundary. Start a new task after reviewing the prior outcome.

## Undo refuses to run

Undo fails when:

- no unconsumed rollback exists;
- rollback expired;
- a recovery source is missing;
- the inverse destination already exists;
- a created folder is no longer empty.

This protects newer user data. Resolve the conflict manually rather than forcing overwrite.

## Release package is blocked

Local macOS packaging can be ad-hoc signed. Public delivery requires Developer ID and notarization. Windows requires Authenticode and a Windows-native package run.

Do not describe a local DMG/ZIP as a public production release.
