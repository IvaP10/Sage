# Sage v2 implementation status

Updated 2026-09-22. The migration is in progress. This checkout does **not** implement or qualify the complete production architecture yet. The original modified files were preserved; see [migration and evaluation](migration-and-evaluation.md).

## Implemented in the working tree

| Area | Concrete implementation | Evidence boundary |
|---|---|---|
| Conversation | Answer-only and streaming answers; iterative typed tool results; bounded repairs; interruption; fresh run continuation at the tool budget | Rust provider fixtures, including a real temporary-file read–reason–write flow; no real-model qualification |
| Authority | Versioned run scopes, closed feature schemas, untrusted model provenance, Cedar dispatch policy, exact single-use worker-bound grants and revocation | Deterministic regressions; model/runtime and native services still share more authority than the target design |
| Prepared execution | Typed intent/target/effect contracts; encrypted prepared-action/dispatch/verification journal; ambiguous effects cannot be re-dispatched automatically | Journal replay tests and core integration tests; full crash fault matrix pending |
| Data release | Exact serialized context and destination preview for each external inference call; no loopback exemption; endpoint-bound credential accounts | Denied disclosure never invokes fixture provider; live provider and native credential-service acceptance pending |
| Files | Explicit read/create folder grants, protected state/model/runtime paths, directory handles, no-follow traversal, hardlink rejection, bounded reads, exclusive creation and atomic replacement | Mac filesystem regressions; residual noncooperative writer race and Windows acceptance remain |
| Undo | Encrypted backups, current content/identity checks, mutation serialization and pre-dispatch journal consumption | Temporary-file integration tests; uncertain Undo requires review |
| Research | Public HTTPS fetch with DNS pinning, no credentials/cookies/proxies/redirects, public IPs only, bounded UTF-8 text/HTML/JSON and content hashes | Validation tests; isolated rich-document parsing and search connectors are not implemented |
| Browser | Role-separated pairing; tab/window/frame/document/generation/origin binding; same-origin navigation; stale/replayed grant rejection | Node Chrome-API fixtures; actual Chrome pairing/races not accepted yet |
| Storage | SQLCipher, in-memory temporary storage, staged encrypted migration and rollback, encrypted artifacts with expiry | Cipher canary/migration tests; legacy plaintext recovery files and full backup lifecycle remain |
| Audit | Hash-chain verification plus a separately stored OS checkpoint before effects, after verification and on unlock | Tamper, truncation and missing-checkpoint tests using the secret-store fixture; OS credential-service timing/availability pending |
| Memory | Permission filtering before FTS ranking, bounded context, source lineage, derived-memory deletion and active-run cancellation on forget | Scope and lineage regressions; embeddings/vector backend/tokenizer allocation remain |
| Skills | Verified traces become disabled drafts; exact digest review; fresh identities and current policy on reuse | Source and core checks; parameterized procedures and promotion pipeline remain |
| Schedules | Explicit typed scope, expiry/run budget, transactional claims, persisted coalesced events, overlap prevention, cancellation on edits/deletion, interrupted-claim review | Claim/budget/changed-folder tests; full watcher and crash acceptance remain |
| IPC and clients | Protocol v2, mutual HMAC proof, role-separated credentials, peer UID on Unix, replay rejection, cancellation-safe bounded reader, matching Swift/C# bindings and new native surfaces | Rust transport tests and Mac source build; signed identity, Windows ACL/peer and physical native acceptance pending |
| Local inference | Signed profile verifier, exact asset digests, shared admission budget and restricted macOS CPU process prototype | Signature/admission code and budget tests; **no real model run**, qualified assets or warm worker |
| Unsupported operations | Host command execution removed; generic UI mutation, privileged installation, VM, WASM/MCP and unqualified platforms remain unavailable | Discovery/compiler/adapter refusal paths; no automatic binary-based enablement |

## Required work before production

1. Extract agent/model/network/native/credential authority into the planned processes. Replace native-authentication Booleans and installation file credentials with signed endpoint identities and service-owned authentication. Complete typed authority messages and result validation across all boundaries.
2. Install reproducibly converted, evaluated and signed model/runtime profiles. Complete tokenizer accounting, warm inference, measured RSS enforcement, pressure/battery scheduling, embedding/vector retrieval and the 16 GB device matrix. The development Mac has **8 GiB RAM** and is not a reference 16 GB device.
3. Implement Apple Virtualization and QEMU/WHPX backends with signed guest assets, mandatory guest restrictions, broker-only network services and validated artifact/patch transfer. Implement narrow native operations with independently observed effects.
4. Complete local keyword/VAD/ASR/TTS and microphone lifecycle acceptance, WASM/MCP/connector isolation, parameterized skills and fully typed background-management contracts.
5. Complete legacy recovery migration, retention/export/key-recovery UX, filesystem race/power-loss validation, full crash recovery and physical platform acceptance.
6. Run the fixed workflow/model/security/performance evaluations. Implement and verify signed updates/TUF, SBOM/advisory/provenance checks, app/helper/model/VM signing, notarization and installer acceptance.

## Verification records

The integrated suite currently has 42 Rust tests and three browser contract tests. Rust 1.89.0 was tested during this implementation (41 tests before the final journal additions); the integrated journal suite was tested using the installed Rust 1.98.0 compiler. A final verification record will distinguish each toolchain and exact source digest. No test count is a real-model, native browser, Windows, VM, microphone or distribution acceptance result.

Run `cargo test --workspace --locked`, `node --test integrations/browser/background.test.mjs`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, and the relevant native build. Use the [release gate manifest](../../evals/release-gates.json) and `python3 scripts/check-v2-release.py --require-ready` before publication. Pending gates are intentional; do not replace missing measurements with estimated values or a build result.
