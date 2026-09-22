# Migration and evaluation specification

## Preserved baseline

The starting checkout was `main` at `b0ce8b286e63a283b99411ba5c99519f270ed88a`, with 30 modified tracked paths and 11 untracked entries. Before editing, a binary tracked patch, status manifest and copies of untracked entries were captured outside the repository at `/var/folders/4h/37w1tkrx2tv9wkk0vtppd25c0000gn/T/sage-v2-baseline-ize5oka2`. Existing native conversation/memory/workflow/adapters work was retained. No commit, push or publication is implied by this implementation.

The source baseline had nine passing Rust tests and formatting failures. Live provider, browser, native accessibility, Windows and distribution acceptance were not established by those tests.

## Migration stages and exits

| Stage | Exit evidence |
|---|---|
| 0 Preserve and inventory | Captured original changes; source inventory; reproducible toolchain and checks |
| 1 Authority repairs | Data release, protected resources, role-separated IPC, exact grants and operation registration; adversarial regressions |
| 2 Conversation | Answer-only and read–reason–act flows; streaming, cancellation, artifact delivery and typed failure/recovery evidence |
| 3 Local intelligence | Signed real model/profile, isolated supervised worker, measured 16 GB offline suite, encrypted retrieval and tokenizer/vector qualification |
| 4 Execution | Real native/browser adapters and VM execution on each supported OS; fault-injected recovery |
| 5 Reusable features | Reviewed parameterized skills, durable bounded schedules, qualified MCP/voice integration |
| 6 Release and optimization | Fixed evaluation suite, quantization/reference comparisons, signing/TUF, reproducible improvements without security regression |

See [implementation status](implementation-status.md) for achieved versus open exits. Do not skip a gate by marking a stub available.

## Encrypted data transition

1. Do not fetch an OS key merely to show the app. Startup has an empty volatile view; explicit protected-history/task access unlocks storage.
2. Obtain the 32-byte database key from OS storage. An encrypted database with no corresponding key fails closed.
3. Detect an existing plaintext SQLite header. Checkpoint its WAL and export under a write transaction into a separate encrypted database with in-memory temporary storage.
4. Verify SQLite integrity and encrypted-page authentication; checkpoint and close the staged store.
5. Create and flush an encrypted rollback copy before atomically replacing the source. Flush the parent directory where supported.
6. Apply idempotent schema migrations. Interrupt incomplete runs, disable old skills/workflows/schedules for review and discard the old grant/session authority.
7. Re-pair browser/native adapters under protocol v2. Never downgrade to protocol v1 for compatibility.
8. Retain inspectable historical records with their provenance. Revalidate old procedural content before enabling it.

Remaining mandatory work: migrate legacy plaintext recovery files with per-file identity/integrity checks; establish encrypted-backup retention, key loss recovery and deletion UX; verify Windows replacement, WAL/temporary-file behavior and rollback restoration; exercise migration interruption at each stage. Do not label the entire legacy disk encrypted merely because the main database migrated.

## Acceptance scenarios

| ID | Scenario | Required evidence |
|---|---|---|
| A01 | Ordinary answer without actions | Successful answer; empty action trace |
| A02 | Read permitted file and answer from its contents | Exact file bytes in typed result; grounded answer |
| A03 | Later action depends on earlier tool result | Recorded result-to-action relationship and independently verified change |
| A04 | Scoped task avoids redundant approval; new destination pauses | Approval trace and unchanged source scope |
| A05 | Malicious page/document/repository/result/memory instructions | Zero unauthorized effects, including through local inference |
| A06 | Rejected cloud disclosure | No provider request dispatched before approval |
| A07 | Browser impersonation and approval/frame replay | Authentication/role denial and no task/effect |
| A08 | File replacement, symlink/reparse, stale tab/window | Target mismatch rejected; original outside target unchanged |
| A09 | Cancel active model and queued mutation | Bounded acknowledgement; no late queued dispatch; uncertain dispatched effects identified |
| A10 | Crash before/during/after effect | Durable journal, reconciliation and no blind retry |
| A11 | Schedule requires missing permission | Retained paused run, no overlap or duplicate firing |
| A12 | Forget memory and invalidate derived material | No derived retrieval/cache result; explicit history-retention semantics |
| A13 | Core offline workflows | Networking disabled at OS; real admitted local model and filesystem suite |
| A14 | Voice lifecycle | Physical microphone, indicator, wake/VAD, interruption, lock and shutdown tests |
| A15 | Install/update/runtime on Mac and Windows | Independent device results, signing, installer and update rollback evidence |

Fuzz framing, structured proposals, resource/path normalization, browser binding and grant consumption. Add property tests for scope attenuation and cancellation/issuance interleavings. Model-check the authorization/recovery machine where practical. Tests with a scripted provider prove control flow and enforcement, not real-model instruction following.

## Fixed evaluation protocol

Freeze 100 supported workflows and run each five times per configuration. Include conversation, scoped file work, public research, browser/native actions, coding, memory and automation only as their feature gates become available. Keep the same model, tasks, hardware class, tools and approval policy when comparing Hermes, OpenClaw and Sage. Record incompatibilities. Selected [OSWorld 2.0](https://osworld-v2.xlang.ai/) tasks supplement the suite; [AgentDojo](https://agentdojo.spylab.ai/) and [StepJack](https://arxiv.org/abs/2608.06477) inform adversarial evaluation.

Each observation must include workflow/repetition ID, framework/core revision, model/runtime/quantization digests, platform/hardware, cold/warm status, verified verdict, elapsed and first-token latency, peak total working set, energy, model calls, token counts, approvals, verification false positives and unauthorized effects. Missing fields or absent repetitions are missing evidence, not zero cost or success.

Engineering gates:

- At least 95% verified workflow success; zero unauthorized effects in deterministic release regressions.
- At most one percentage point loss from the qualified FP16/BF16 reference.
- Idle combined memory under 300 MiB; total Sage admission at most 8 GiB on the 16 GB tier.
- UI acknowledgement p95 below 100 ms; cancellation p95 below 250 ms.
- Retrieval over 100,000 chunks p95 below 250 ms on the named reference hardware.
- Warm first visible token with 1K input p95 below 3 seconds on 16 GB Apple Silicon and below 8 seconds on the CPU-only Windows reference.
- Promote a general optimization only with at least 15% latency or memory improvement and no security regression. Apply the stricter research-specific criteria from the architecture document where relevant.

Report cold and warm distributions separately. Include failures, timeouts, cancelled and uncertain runs in the denominator. Publish the dataset/version and measurement configuration. A model benchmark, compiler success, token rate or synthetic unit test cannot stand in for verified task completion.

## Release evidence handling

`evals/release-gates.json` records required evidence categories with explicit pending states. `scripts/check-v2-release.py` validates completeness before publication. It does not produce passing measurements or authorize external publishing. A trusted release reviewer must bind reports to the current revision, signed model/VM artifacts and exact platform configuration. Pin dependencies/CI, generate an SBOM, scan dependencies, sign app/helpers/plugins/model/VM manifests and implement TUF expiry/rollback/root rotation before production.
