# Troubleshooting

## Build and protocol

Use Rust 1.89.0 or newer and the committed lockfile. Build the core with `cargo build --workspace --locked`; build the Mac client with `swift build --package-path apps/macos`. Windows requires its Windows SDK/.NET/WinUI environment. A Mac build does not establish Windows runtime acceptance.

Protocol v2 is required on both ends. Regenerate Swift using `PROTOC_GEN_SWIFT=/absolute/path/to/protoc-gen-swift ./scripts/generate-protocol.sh`; C# and Rust generate from `proto/sage/ipc/v2/sage.proto`. Rebuild/re-pair matching clients and browser hosts. Never downgrade authentication after a mismatch.

## Protected storage and credentials

Startup uses an empty volatile view until an explicit protected-data operation unlocks SQLCipher through the OS key store. If the database key or audit checkpoint is unavailable, restore the matching OS credential; do not delete encrypted history or initialize a replacement key over it. Migration preserves an encrypted rollback database. Key recovery/export and legacy recovery-file migration are still production work.

The Mac UI uses `ipc-auth-v2.key`; the browser uses a separate `browser-ipc-v2.key` under the same Sage data directory. A browser host credential cannot authenticate as the UI. If pairing fails, check that the host and native app use the same data directory and protocol version, then explicitly pair the intended tab again.

Provider keys belong to exact normalized destinations. After the v2 migration or an endpoint change, re-enter the credential for the intended route. A connection test is separate from permitting actual task context to leave the device. Private network, metadata and redirect targets are rejected.

## Unavailable operations

Only the installed feature registry is callable. Generic desktop clicks/typing/uploads/submission, host commands and privileged operations are unavailable. Code execution requires the future qualified VM boundary; there is no host sandbox or PowerShell fallback. Model profiles require actual signed assets and supported isolation, not just a file with a candidate model name.

## Interrupted work

Review the last verified evidence. A timeout after dispatch can leave an uncertain external effect; confirm actual state before retrying. File recovery rejects changed contents/identities. Continue at the tool budget creates a fresh scoped run without reusing old capabilities. Expired background authorization must be reviewed; schedules with interrupted claims stay disabled to avoid duplicates.

Folder events are coalesced and persisted through cooldown/overlap. Lost watch access disables the schedule with a reason. A changed symlink/canonical folder target needs fresh authorization. Editing or deleting a schedule cancels its active firing.

Use [implementation status](v2/implementation-status.md) and [release gates](../evals/release-gates.json) to distinguish a configuration error from a feature that is not implemented or qualified yet.
