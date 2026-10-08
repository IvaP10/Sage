# Incremental audit verification

Implemented 30 September 2026 in `crates/sage-core/src/audit.rs`, with connection ownership in `storage.rs`. This replaces the repeated full-history scan previously performed on each checkpoint. It does not close the broader performance or storage-isolation gaps.

## Verification contract

The v2 hash-chain format and OS checkpoint account are unchanged. A persisted head inside the database is not accepted as proof. The first audit checkpoint on each newly opened connection verifies all records, contiguous sequence numbers, hash links, record contents, database identity and the existing protected OS checkpoint. Missing or incompatible expected checkpoints still stop execution.

After successful verification and checkpoint persistence, the process keeps a connection-local verified head. Reuse requires the same protected checkpoint, database identity, schema version, external-change counter and local audit-mutation revision. These cursors are not persisted. Clones share a checkpoint lock; replacing the deferred connection during unlock installs a new observer and discards its old cursor.

On an ordinary checkpoint:

1. Read the OS checkpoint. A secret-store error or malformed record retires the cursor.
2. Reserve the SQLite writer while reading change counters, metadata and the chain. This prevents another connection from committing between the version check and the database snapshot.
3. Reuse the verified prefix only if all its conditions still hold. Otherwise scan from genesis. A reused nonempty prefix also requires its recorded head row to remain present and match.
4. Verify every new row using an indexed sequence range. A gap, invalid sequence, wrong link or changed body fails verification.
5. On the incremental path, independently scrub at most 128 historical records. The scrub walks a fixed prefix, preserving its expected link between batches and checking its final hash against the previously verified head. It then starts another cycle, including the newer prefix.
6. Commit the verification transaction, release the database mutex and advance the OS checkpoint if its value changed. An identical checkpoint still reads the protected credential but does not rewrite it. The first checkpoint records its expected-anchor flag after the OS write.
7. Recheck connection counters and database identity after credential persistence. Unexpected changes during that interval prevent acceptance. A valid append through the same store can remain an unanchored tail for the next checkpoint.
8. Publish the new in-memory cursor only after every required step succeeds. A failed or ambiguous OS write, failed metadata write or changed database forces a full verification on retry.

The OS credential and SQLite are separate durability domains. These steps do not create an atomic transaction across them, and they do not prove that an already dispatched external operation was prevented. The effect/Undo reducers retain their existing conservative accounting.

## Change detection and its window

SQLite's `data_version` reports commits from other connections when compared on the same connection. Commits on that connection do not change it, so the implementation also observes local row changes. [SQLite data-version semantics](https://www.sqlite.org/pragma.html#pragma_data_version).

The SQLCipher build now enables SQLite's pre-update hook. It invalidates the cursor on audit updates/deletes and insertion at or below the verified sequence. The callback uses only atomics; it does not execute SQL or acquire application locks. A rolled-back mutation can conservatively cause an extra full scan. A saturated mutation revision permanently disables prefix reuse. The tests explicitly exercise delete-all, reuse of a prepared delete, and a unique-key replacement that removes an old row while adding a valid-looking new tail. [SQLite pre-update API](https://sqlite.org/c3ref/preupdate_blobwrite.html).

An observed row/schema change or commit from another connection causes full verification at the next checkpoint. The bounded scrub also checks historical records independently of those notifications. One fixed-prefix cycle needs `ceil(prefix_rows / 128)` successful incremental checkpoints. There is no wall-clock scrub deadline while the application is idle. A test deliberately disables notifications and changes record 500 in a 512-record log; the fourth bounded scrub rejects it. That fault injection does not establish detection timing for arbitrary physical disk corruption, page-cache behavior or a malicious OS.

This changes the old detection window: an unreported historical change is checked when its scrub batch is visited, rather than rehashing all old rows on each action. Startup and invalidation still require a full scan. No protection is claimed against a compromised process that can remove observers/change verification code, or an actor that can rewrite both the encrypted database and its protected OS anchor. Multiple independent core processes are not a coordinated checkpoint-writer protocol; the supported writer ownership is one core with shared `LocalStore` clones.

## Measured local probe

The [checkpoint evidence](evidence/2026-09-30-incremental-audit/manifest.json) contains the complete command logs and source hashes. The isolated probe used an Apple M3, arm64 macOS 27.0.1, Rust 1.98.0, a debug build, temporary SQLCipher databases, small synthetic audit records and an in-memory secret store. Each incremental column averages five samples. Seeding records and reopening the database are outside those incremental timings.

| Existing records | Initial full scan, ms | Unchanged checkpoint, mean ms | One new record plus scrub, mean ms |
| --- | ---: | ---: | ---: |
| 100 | 1.360 | 1.053 | 1.080 |
| 1,000 | 9.711 | 1.300 | 1.269 |
| 10,000 | 96.309 | 1.319 | 1.346 |

The stronger regression assertion is the work bound: zero or one new record plus at most 128 historical records in these cases, independent of total history size. Reopening verifies every record again. These timings exclude OS keyring latency, installed-client behavior, production build optimization, disk contention and end-to-end request latency. They are not a service-level guarantee or a comparison against another agent product.

## Validation and remaining work

The workspace passes 130 core tests plus one browser relay test; Clippy passes with warnings denied. Those tests and lint checks used Rust 1.98.0. A separate workspace/all-targets check passes with both Cargo and rustc explicitly set to the CI's 1.89.0 toolchain. Eight added tests cover bounded work/restart, local mutation variants, external commits/schema changes, mutations during credential writes, failed and ambiguous writes, bounded scrubbing, concurrent store clones, deferred unlock and revision saturation. Existing finalization, receipt, Stop, continuation and Undo failure matrices continue to pass. This milestone adds no new physical-power-loss or abrupt-process-exit fixture for the OS credential store.

The `rusqlite` pre-update feature compiles SQLCipher with `SQLITE_ENABLE_PREUPDATE_HOOK` and generates bindings through libclang. The lockfile records the added build dependencies. Native Swift/C#/protocol sources are unchanged. Windows CI and release jobs now explicitly locate and check `libclang.dll`; their YAML was parsed locally, but their PowerShell, compiler and runtime execution were not tested on this Mac. The hosted image's documented LLVM installation is an input to those jobs, not evidence that Sage builds or runs there. [GitHub Windows runner software](https://github.com/actions/runner-images/blob/main/images/windows/Windows2025-Readme.md).

Remaining work includes moving synchronous SQL and OS credential operations off control/runtime threads, coordinating exclusive core writer ownership, bounding total history and artifact retention, persisted segment/compaction policy, idle scrub scheduling, and measuring full stage latency with the actual OS secret store. A full scan after startup or unexpected changes remains proportional to history size and currently occupies the calling thread. Full macOS UI, Windows, installed provider/browser flows and all three requested real-task groups remain unqualified.
