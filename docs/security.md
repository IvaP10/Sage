# Sage security model

## Trust boundary

The Rust core is the policy and authority boundary. Models, websites, documents, messages, terminal output, application text, browser DOM, plugins, and tool responses are untrusted inputs.

The native clients are authenticated presentation clients. They may report the result of an OS-owned permission or authentication prompt, but they do not mint executor authority.

~~~text
untrusted model proposal
  → schema validation
  → canonical resource resolution
  → provenance/trust analysis
  → risk classification
  → policy
  → permission
  → exact approval when required
  → single-use capability
  → restricted executor
  → audit
  → observation
  → verification
~~~

## Local IPC

- Unix Domain Socket on macOS, mode 0600.
- Named Pipe on Windows.
- Protobuf frames with a four-megabyte ceiling.
- Protocol version on every frame.
- 32-byte server and client nonces.
- HMAC-SHA256 challenge response with a domain separator.
- Constant-time proof comparison.
- Client kind and version bound into the proof.
- No application command before authentication.

On macOS the installation key is a 32-byte owner-only file under Sage Application Support with mode `0600`. It is transport authentication material, not a provider credential, so startup performs no Keychain access and can never present Keychain UI. On Windows it lives in Credential Manager, whose lookup is non-interactive. At process launch it travels through an anonymous stdin pipe. It is never put on a command line, in an environment variable, in SQLite, or in logs.

## Model containment

The model receives schemas and context, not OS handles. It returns typed Action values. General shell interpreters are prohibited as normal RunCommand programs. The model cannot choose a reusable capability, approval digest, executor process, or native authentication result.

Model provenance is retained. A model proposal can be considered for execution but cannot authorize itself.

## Prompt injection

ProvenanceSource and TrustClass distinguish:

- direct user authority;
- trusted Sage components;
- observations;
- untrusted external content.

Text from a page or document remains external content even when it contains first-person commands, claims to be a system message, or asks Sage to weaken policy. External content cannot directly propose an executable action and cannot satisfy approval.

When external data is supplied to a planner, it occupies the untrusted_context field, separate from trusted constraints.

## Policy categories

- **Safe:** bounded local state operations without private-data or external consequences.
- **Sensitive:** reading private data, semantic UI inspection, or controlled local commands.
- **Consequential:** writes, moves, uploads, messages, submissions, downloads, and networked commands.
- **Destructive:** overwrite and delete.
- **Privileged:** install and system setting mutation.
- **Prohibited:** general-purpose shell dispatch, direct credential-store file access, malformed resources, and policy-evasion surfaces.

Risk is deterministic Rust code. The model cannot lower it with metadata or natural-language explanation.

## Approvals

Approvals are:

- action-specific;
- task-specific;
- digest-bound;
- expiring;
- single-use;
- invalid after task/action mismatch;
- not reused for later actions.

Privileged actions require native device authentication in addition to the approval choice. macOS uses LocalAuthentication. Windows uses UserConsentVerifier. Sage never asks the user to enter a password, PIN, biometric sample, or recovery key.

## Capabilities

A grant names an exact resource and allowed operations. It also contains task ID, action ID, domain, issuance/expiry time, remaining-use count, and revocation state.

The broker consumes a grant before invoking an executor. The executor re-checks its resource. Task completion, failure, or cancellation revokes remaining grants.

Capabilities are not persisted as reusable authority and are not sent to the model or UI.

## Filesystem

- Absolute paths only after native resolution.
- Parent traversal rejected.
- Existing path canonicalization.
- New path parent canonicalization.
- Authorized-root enforcement.
- Symlink escapes rejected by canonical scope checks.
- Protected credential directories prohibited.
- No implicit overwrite.
- Bounded reads and writes.
- New content staged through a sibling temporary file.
- Overwrites retain a recovery copy.
- Delete moves into private recovery storage.
- Undo refuses to overwrite an existing inverse destination.

Folder access is intended to be extended with native folder-picker grants. A request like “clean Downloads” is not a blanket delete authorization.

## Sandbox

RunCommand is structured as executable plus arguments, working directory, network boolean, and timeout.

The worker:

- validates task/action/domain/resource bindings;
- receives one request over inherited stdio;
- starts with a cleared environment;
- restricts PATH;
- limits output;
- applies a timeout;
- is killed when the broker drops it.

The macOS backend uses a deny-by-default sandbox profile. The Windows worker refuses to run until AppContainer and job-object limits are implemented. Sage never runs the same command unsandboxed because the sandbox backend is unavailable.

## Browser

Browser capabilities bind the exact origin. Credential-bearing URLs are rejected. The browser worker must pair with a structured browser session and verify page state after mutation. Web content remains untrusted.

The current worker deliberately refuses action until pairing exists. Raw visual control is not an implicit substitute.

## Privilege

The full UI and core run as the user. The privileged helper is a separate binary with an operation allowlist. It has no command-string API. Its current installation shape refuses execution until a signed platform operation exists.

Never add an execute_root_command, RunAs arbitrary string, sudo shell, or equivalent interface.

## Secrets

Provider SecretStore uses Keychain on macOS and Credential Manager/DPAPI-backed storage on Windows through the keyring backend. It is called only for a configured provider request or another action that needs that credential, never during ordinary app startup. SecretBytes zeroizes its buffer on drop.

SQLite stores only secret handles or presence/configuration metadata. Audit events and UI events must contain redacted summaries, not raw typed content, message bodies, file bodies, stdout credentials, or provider keys.

Protected PDFs must be opened in the native PDF application. Sage does not collect or persist PDF passwords.

## Persistence and audit

SQLite uses WAL, foreign keys, and a busy timeout. Incomplete tasks are marked interrupted at startup.

Audit records contain redacted payloads and a SHA-256 chain:

~~~text
record_hash = SHA256(
  record id | task id | action id | event type |
  redacted payload | previous hash | timestamp
)
~~~

The chain detects database edits but does not make the file immutable. OS account protection and encrypted storage remain separate controls.

## Privacy defaults

- Local state by default.
- No server database.
- No Redis/Postgres requirement.
- No telemetry in the current source.
- No raw audio persistence.
- No automatic screen-to-memory promotion.
- No automatic credential distribution to plugins.
- No cloud call without an explicitly configured provider and its network policy.

## Release gates

A source build is not a public security proof. Release acceptance separately requires:

- macOS Developer ID signing and notarization;
- stable helper and keychain signing identities;
- Windows Authenticode;
- Windows-native UI, pipe ACL, Windows Hello, and AppContainer tests;
- clean-machine installation;
- real browser pairing tests;
- configured provider tests;
- dependency and credential scans.
