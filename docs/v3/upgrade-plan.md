# Native runtime rebuild plan

This checklist tracks the runtime rebuild against the current [Sage architecture](../architecture.md), including migration gates and sixteen acceptance capabilities. It describes the target and current evidence; it does not claim production qualification. Implementation progress belongs in the [ledger](implementation-ledger.md).

## Checkpoint: 3 October

The first implementation pass improves native presentation and several execution bottlenecks. It does **not** complete the target architecture or qualify the product for production.

- macOS: searchable pinned/recent chats, separate conversation and collapsible activity, streamed replies, factual action progress, connection/review visibility, setup guidance, and a debounced single-flight refresh lane. Streaming renders at bounded intervals. Final stored replies take precedence over unfinished prefixes.
- Windows: matching conversation/activity layout, stable row identities, search and keyboard controls, selected-folder indication/removal, connection/model state, batched streaming updates and single-flight snapshot refresh. These changes still require a Windows build and live acceptance.
- IPC: a single outbound transport owner, FIFO sequences, separate regular/control admission budgets, byte/count accounting including in-flight writes, and a five-second queue/write deadline. Socket backpressure does not block inbound Stop admission. A timed-out/overloaded peer is disconnected; this is not proof that an already dispatched OS effect stopped.
- Reasoning: one answer, one ordinary action, or up to eight independent read actions per model turn. Local and compatible providers use the same parser rules. Mutations, questions, app actions and dependent reads cannot share a batch. Execution remains sequential and each action passes its own authorization and verification. This saves model turns; it is not parallel OS execution or a measured end-to-end latency claim.
- Task graphs: later turns retain dependencies on previous verified actions so captured successful traces preserve causal ordering. This does not add result bindings or turn literal traces into the proposed procedure IR.
- Retrieval: direct conversation identity lookup survives the sidebar's 200-row limit. Preference filtering occurs before retrieval limits, newest applicable values win, and only consumed preferences add context lineage.
- Schedules: indexed, bounded due pages reach schedules beyond the 200-row UI list. Skill/workflow/schedule identity operations use direct lookups. Folder dirty roots survive a failed persistence attempt and clear only after pending triggers are saved or revoked. Watcher process-crash catch-up and OS lifecycle policies remain open.

## Migration requirements

| Review item | Current progress | Evidence still required for completion |
|---|---|---|
| P0.1 evidence baseline | Focused regression fixtures and source-bound checkpoint evidence | Installed stage spans, bytes/resources, matched warm/cold task distributions and privacy checks |
| P0.2 trust boundary | Existing capability/role enforcement retained | Service-owned secrets; signed native peer/session attestation; isolated inference, hostile parsing and egress; forged-worker negatives on both OSes |
| P0.3 control/durability | Bounded independent outbound writer; existing transactional transitions/receipts/Undo retained | Durable reconnect cursors/outbox; Stop under DB/AX/COM stalls; real external-effect settlement and crash matrix |
| P0.4 effect/privacy semantics | Existing prepared file handles and journaled compensation retained | Executable/startup/sync effect classes; forward-effect recovery registration; noncooperative writer cases; artifact purposes/retention/forget boundaries |
| P1.1 inference | Bounded read batches reduce serial model turns | Persistent contained worker, exact whole-prompt token budgets, artifact slicing, endpoint-bound pools and measured 16 GiB pressure policy |
| P1.2 procedures | Cross-turn ordering preserved | Typed value bindings, obligation ledger, resource conflicts/leases, replay on changed data, safe independent parallel execution |
| P1.3 perception | Scoped directory discovery retained | Isolated scoped AX/UIA/DOM observations and semantic effects; freshness/target epochs, real native tasks and wrong-window tests |
| P1.4 coding environments | Unsupported host execution remains disabled | Qualified isolated task environment, brokered import/export, quotas, persistent artifacts and real edit/build/debug flows |
| P1.5 voice/UI | Native presentation and refresh changes; Mac build and limited live UI checks | Hardware voice/Stop/device-change tests; Windows voice and native build; paged history, large sessions and energy measurements |
| P2.1 background | Due queue and transient dirty-root retry | Durable watcher catch-up, bounded watch admission, lock/sleep/restart/clock/DST policy and live delivery |
| P2.2 memory | Lookup/filter/collision fixes | Scoped hybrid retrieval at 100k records, evidence graph, purpose retention and reviewed portable procedure migration |
| P2.3 adaptive services | Not implemented | Bounded speculation, calibrated routing/observation and attenuated specialists; quality/energy/tail-latency ablations |
| P3 qualification | Gates still reject readiness | Signed installers/updates, user/accessibility studies, adversarial fault matrix and all real-task/platform suites |

## Product capabilities

These are acceptance items, not enabled feature announcements. Each must meet the security, privacy and latency constraints in the [architecture guide](../architecture.md).

| # | Capability | Required evidence / current boundary |
|---|---|---|
| 1 | Scoped situational awareness | Granted apps/windows only; source invalidation, lock/revocation and bounded capture. Directory scope alone is insufficient. |
| 2 | Event-driven assistance | Durable event catch-up and interruption budget. Scheduler fixes are partial groundwork. |
| 3 | Cross-application handoff | Typed source/destination objects, transformation lineage and verified native destination effect. Open. |
| 4 | Reviewable autonomous workflows | Staged preparation, effect diff and explicit commit frontier with compensation. Existing approvals/Undo are partial. |
| 5 | Intent-level routines | Procedure parameters, predicates and result bindings working on new inputs. Literal skill capture is insufficient. |
| 6 | Bounded self-healing | Repair obligations, scoped alternatives and independently verified recovery under changed state. Existing step retries are partial. |
| 7 | Personal local knowledge | Inspectable evidence, calibrated conflict handling, scoped retrieval and deletion across derivatives. Lookup fixes are partial. |
| 8 | Collaborative specialists | Attenuated contexts/artifacts, one accountable effect executor and resource-budget enforcement. Open. |
| 9 | Anticipatory preparation | Epoch-bound preparation without unapproved reads/egress/effects; negation/cancellation and p95/energy ablations. Open. |
| 10 | Transferable learned workflows | Parameterized/versioned procedures, review on import, provenance and no transferred authority. Open. |
| 11 | Temporary tools | Contained generated functions, typed contracts/tests, quotas and expiration. Open. |
| 12 | Generated micro-apps | Isolated read-only artifact views with brokered action proposals; no ambient network/filesystem bridge. Open. |
| 13 | Persistent task environments | Quota-bound environments and resumable artifacts with current import/export checks. Open. |
| 14 | Multimodal spatial work | Scoped regional capture, semantic/geometry binding and calibrated cross-frame identity. Open. |
| 15 | Outcome contracts/review | Machine-checkable obligations and counterfactual effect preview. Factual action counts are partial groundwork. |
| 16 | Continuation across time | Revalidated epochs, receipts, assumptions and renewed scope after environmental changes. Existing bounded continuation is partial. |

## Verification and local build

See the [macOS accessibility capture](evidence/2026-10-03-native-runtime-upgrade/macos-live-accessibility.txt), [macOS preview image](evidence/2026-10-03-native-runtime-upgrade/macos-live-preview.png) and [Windows source check](evidence/2026-10-03-native-runtime-upgrade/windows-source-check.txt). Tests, source checks, native compilation and live interactions are separate evidence layers.

The installed macOS 27 SDK requires a SwiftUI macro plugin absent from this machine's Command Line Tools. The available 26.5 SDK builds the current application:

```sh
swift build --package-path apps/macos \
  --sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  --build-system native
```

Full Swift package tests require XCTest from Xcode; the standalone transport/cancellation/presentation checks run with `sh scripts/test-native-control.sh`. The development preview is an ad-hoc-signed local build, not a signed distribution release. No provider inference, microphone, Windows UI or installed browser acceptance is implied by the preview checks. Context/artifact budgeting remains a known limitation for large read batches.
