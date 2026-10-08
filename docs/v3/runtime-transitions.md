# Runtime transition contract

Implementation checkpoint: 29 September 2026. This describes current source behavior in the rebuild, not the final architecture or product qualification.

## Authority and verified actions

The planner can propose an action. It cannot choose its verification verdict, change a prepared journal identity or publish authoritative success. The installed tool binds the required outcome; the observer supplies current evidence; `Verifier` checks it. `VerifiedAction::from_observation` then checks that the proposal identifies the current task/action and that the action has reached an execution or recovery state.

`LocalStore::commit_verified_action` performs the following as one database transaction:

1. Check whether returned-content retention was revoked for the run. Remove the newly returned body and artifact when it was.
2. Match run ID, action ID and prepared action digest in the journal. An execution may advance dispatched/uncertain evidence. Recovery may refresh an existing confirmed observation. Ordinary execution replay cannot replace confirmed evidence.
3. Store the bounded returned-content artifact, when permitted, and its reference in the result.
4. Write the task and action projection with `Succeeded` and its corresponding `Confirmed` tool result.
5. Insert the success event and linked audit record.
6. Commit, then update the in-memory projection and publish the event.

The task's aggregate status is preserved. A completed effect may be independently verified after Stop; that does not restart the task or change its cancellation into overall task success. A worker's raw success field remains a receipt, not this verification evidence.

The protected audit checkpoint is an external OS credential-store write and cannot share the SQL transaction. A failure after database commit stops further work while preserving the committed verification. The next checkpoint verifies the unanchored audit tail. This does not make the audit immutable against an attacker controlling both stores.

## Recovery

Recovery independently observes supported postconditions and uses the same verified transition, including a matching tool result. Read recovery also checks the latest confirmed content digest. It cannot recover unsupported outcomes by guessing or replaying an uncertain mutation. Browser/native recovery still depends on the required target identity and observer being available; this checkpoint does not qualify every native effect.

Startup interrupts unfinished runs and does not execute them automatically. The effect journal and the completion reducer retain the distinction between an action that may have run and one with durable verification.

## Preparation, dispatch and interruption

Preparation commits the exact prepared proposal, journal identity, task/action projection, proposal event and audit link together. Dispatch rechecks the proposal digest, task status and grant binding, then commits its journal intent, running action, start event and audit link before invoking an executor. A cancellation or pause that wins the task lock prevents dispatch. A dispatch intent still means an effect may have occurred after a crash; it does not prove that the external worker received it.

Interruption reads the journal inside its transaction. Dispatched, uncertain or incompletely projected confirmed effects remain uncertain and cannot become automatically retryable merely because a cached action status says Failed. Prepared or broker-proven unsent work can fail before execution. The journal transition, resulting tool record, task/action projection, event and audit link commit together. If that commit fails, the planner is not allowed to retry. Error finalization preserves an interrupted state when the projection still contains a possibly executed action, and revokes capabilities before attempting its storage update.

Every persisted task now has a monotonic revision. The common transactional writer compares the input revision with the database, writes the next revision and returns it only after commit. Failed transactions do not advance the caller's revision. Legacy rows without a revision start at zero; startup interruption advances the revision. A stale result cannot overwrite a newer Stop even if that result passed verification before Stop was committed. The revision is currently internal; public event cursors and paged client projections remain separate work.

## Runtime Stop and execution ownership

Each accepted run reserves one bounded runtime lease before acceptance commits. Recovery must acquire a fresh lease for the same task identity; an earlier execution retains ownership through cleanup. The registry admits at most 64 execution reservations and unsaved Stop records. A local watch signal belongs to each execution generation and is retained by its capability grants. A separate watch signal belongs to its control scope. Stop cancels the scope; releasing a lease retires only that execution's grants. Reopening a task cannot revive an earlier generation. Dropping an unaccepted reservation does not leave a tombstone for a task that never existed.

The authenticated native command dispatcher signals Stop before queueing its save request. The signal needs no task-cache, database or credential-store lock, and still reaches a run when the control and response queues are saturated. The Stop handler skips storage unlock. The owned future selects cancellation across preparation, planning, waiting, execution, observation and recovery, then drops that future before cleanup. Existing worker request guards transmit cancellation when dropped.

Persistence is a separate obligation. Stop revokes authority, closes in-memory decisions and saves the cancelled task while preserving verified and uncertain effects. Each unsettled action uses its interruption transaction; the whole task cleanup is not yet one atomic transaction. If a write fails, an in-memory Stop record remains after the execution exits. Snapshots show cancellation with an explicit pending-save message and retain the durable revision. Resume is refused until the Stop is saved. A process restart loses that overlay but interrupts persisted unfinished work instead of automatically replaying it.

This does not interrupt synchronous code already executing, prove an OS call stopped, or undo an effect. Blocking database/filesystem work can still occupy runtime threads, and a slow peer write can delay the session before a command reaches admission. Those boundaries need dedicated workers, bounded transport writes and effect reconciliation.

## Continuation handoff

A continuation receives a fresh task ID, action budget and capability generation while retaining the original `control_scope_id`. The source holds its execution reservation until the child is accepted or rejected. The child's control watch is shared with that source; retiring the source closes its local generation without cancelling the child. Stop at the original control identity therefore reaches the current execution after any number of handoffs, without retaining a runtime alias for every historical task.

Acceptance commits the source's `continued_by` link and revised projection together with the child task, message, conversation link, both events and audit link. Task revisions and an indexed existing-child check prevent a stale or duplicate Continue from forking another execution. The cache lock is acquired before the acceptance transaction. After commit, cache replacement, lease acceptance and spawning the owned future have no intervening await.

Stop saves an idempotent `control_scopes.stopped_at` record and audit link before projecting individual tasks. Continuation acceptance checks that record inside its transaction; recovery checks it before reserving execution. Startup projects unfinished members of a saved stopped scope as cancelled. This survives a failure to update an individual task. An unsaved Stop still depends on the runtime signal; a process crash cannot make it durable retroactively.

Both native clients receive the stable control ID and the continuation link. Their Stop commands use the stable ID, and source cards with an accepted child no longer offer another Resume. Older task rows without these fields retain their own task ID as their control scope; historical continuation relationships are not retroactively regrouped. Undo checks all cached members and outstanding worker effects in a scope before proceeding. Its compensation protocol is described below.

## Worker receipt reconciliation

The bridge saves bounded, correlated worker receipts before waking a waiter. `receipts.rs` tracks whether each `(request_id, kind)` has been projected. Reconciliation reads the saved metadata, never a worker's raw success claim, and commits the task/action state, transport result, journal uncertainty, event, audit link and projection marker together. Publishing and replacing the in-memory task happen only after that commit. A repeated callback is a no-op.

Startup interrupts unfinished runs, then reconciles saved receipts in batches of at most 128 before loading the task cache. Resume and Undo also reconcile pending receipts before reading effect state. A new preparation cannot overwrite an action with unresolved receipts. These batches bound returned rows; historical scans and overall startup latency still require optimization.

A reply leaves an unverified effect uncertain. A cancellation acknowledgement proves only receipt of the signal. Broker evidence that the operation was never sent can make it a cancelled, unexecuted result. Existing independent verification is preserved. The verified-action transaction marks earlier receipts for its exact prepared identity as covered, so a later restart does not downgrade that outcome. Reconciliation does not re-execute the operation or independently verify its postcondition.

Removing a reply's private artifact now nulls its receipt reference instead of deleting the receipt. The migration preserves existing accounting rows. This retains opaque effect history and pending reconciliation after content expiry or forgetting; it does not settle the broader artifact-purpose, Undo and retention design.

## Evidence and limits

Five injected database write failures demonstrate transactional rollback. Six separate child-process exits, without running transaction destructors, cover the journal, artifact, task, event, audit and committed boundaries. Reopening the encrypted database finds either the prior unverified state or the complete verified state. The after-commit case also covers losing the process before event publication. This does not test physical power loss, disk firmware behavior or installed application recovery.

Additional integration fixtures verify an actual temporary-directory effect during recovery and an audit-checkpoint failure after that effect's verification. Fixtures for cancellation and browser messaging are recorded separately in the implementation ledger.

Preparation, dispatch and interruption also have injected final-audit-write failures. These leave their earlier journal and projection writes uncommitted. Revision fixtures reject an old snapshot after Stop and a pre-restart decision projection. A deliberately misleading Failed action projection cannot make a dispatched journal retryable. Abrupt process-exit coverage currently targets the verified transition; it is not a complete crash matrix for every phase.

Runtime control fixtures cover an early signal before a waiter exists, stale grants after Stop and normal completion, a failed Stop database write and later retry, cancellation while the task-cache lock is held, overlapping Resume rejection, and saturated control/response queues. A real temporary-directory creation followed by a stalled observer remains uncertain when Stop drops observation; it is not reported as verified or reversed. These fixtures do not qualify installed native UI or real OS cancellation.

Receipt fixtures inject failures at task, event, audit and projection-marker writes. Separate child processes exit after receipt storage, during task projection, after the marker write and after commit. Raw database inspection before startup repair proves all-or-none projection; reopening then produces one result/event/audit link. Verification-marker failure also rolls back the verified transition. Further checks cover acknowledgement semantics, unsent retry barriers, legacy foreign-key migration, content removal, changed action identity and preservation of verified effects. The live core Resume fixture consumes a saved receipt, independently re-observes an existing temporary folder and finishes without recreating the effect.

Continuation fixtures force Stop before acceptance by holding the storage-unlock lane, then exercise Stop after handoff to a stalled child provider. A failed task-projection fixture preserves the saved scope Stop across reopening. Runtime checks cover 128 further handoffs with one current scope member and independent grant retirement. Transaction failures at message, event and audit insertion roll back both tasks. Four child-process exits cover source projection, child projection, event/audit records and commit; reopening finds either no child or one correctly linked interrupted child. Duplicate Continue and overlapping Undo are refused.

## Remaining work

Run finalization now commits the projections described below in one transaction. Scope Stop intent, external effects and the OS audit checkpoint remain separate boundaries. Saving a rollback plan before observation protects it from a later verifier failure but does not make the external filesystem effect and its recovery metadata atomic. The compensation journal addresses interrupted Undo; forward creation still has gaps before all recovery metadata is saved. Separate artifact purposes, backup retention and full metadata restoration remain open. Durable receipt projection does not establish when an external worker or OS operation has finally stopped; independent postcondition settlement remains required. Continuation, completion and Undo controls still need full native UI and installed-platform qualification.

The task cache still serializes updates under a lock; SQL work is synchronous and historical projections are not yet bounded/paged. Durable cursors, incremental audit validation, storage worker isolation, bounded session writes, explicit stopping/settled states and the broader native authority design remain required.

Source: [`transitions.rs`](../../crates/sage-core/src/transitions.rs), [`runtime.rs`](../../crates/sage-core/src/runtime.rs), [`receipts.rs`](../../crates/sage-core/src/receipts.rs), [`undo.rs`](../../crates/sage-core/src/undo.rs), [`finalization.rs`](../../crates/sage-core/src/finalization.rs), [`engine.rs`](../../crates/sage-core/src/engine.rs), [`storage.rs`](../../crates/sage-core/src/storage.rs), [`journal.rs`](../../crates/sage-core/src/journal.rs), [`dispatch.rs`](../../crates/sage-core/src/ipc/dispatch.rs).

## Journaled compensation, 30 September

Undo is a separate effect, with a durable identity tied to the original action. Native clients send that action ID with every Undo or Check Undo command. Repeating an already verified inverse returns its recorded completion and rechecks the audit anchor; it cannot select an older action. An older client without the identity is rejected before dispatch. Startup repairs available Undo identities in batches of 128 without changing the filesystem or rearming legacy consumed plans.

| Saved phase | Meaning | Explicit retry behavior |
| --- | --- | --- |
| Prepared | Target identity, parent identity, plan digest and desired condition are saved; no inverse was dispatched | Revalidate the same plan, target and backup, then permit dispatch |
| Dispatched | The inverse may have run, including when the audit anchor failed before its syscall | Observe only; never repeat the inverse |
| Uncertain | The desired condition was not established | Observe only; retain the unresolved record and its Check Undo control |
| Verified | A fresh read established the desired current condition | A duplicate command is idempotent; a separate action ID is required for an older Undo |

Preparation, dispatch and verification each commit the compensation journal, task revision, event and audit link together. Only verified compensation consumes the rollback plan, in that same transaction. Events are published after commit. An audit checkpoint in the OS credential store remains a separate operation: failure before the filesystem change prevents execution, and failure after verified commit does not erase the verified database state.

The inverse must match its recorded service-prepared file operation: creating a file permits removal of that file, overwriting permits restoration of its preimage, and creating a folder permits removing the same empty folder. Preparation checks the current content or folder identity, the saved file precondition, task ownership of the backup and its expiry/integrity. Later prepared retries also require the same full file identity and parent identity. Nonempty folders are rejected before dispatch. Verification opens the path independently and checks the pinned parent's identity plus restored content or target absence. Digest evidence survives backup deletion so an already dispatched restore can still be checked. Verification establishes the current postcondition; it does not establish exclusive causation or guarantee that another program will not change it later.

Undo admission holds the submission and mutation lanes, refuses active continuations and outstanding worker effects, and serializes backup use with forgetting. Once a scope has a compensation record, Resume cannot reuse its old forward execution; a fresh request starts from the current state. Current execution facts count undone actions separately and treat unverified inverse effects as uncertain. The old forward action/result remains historical evidence. Dispatch removes the task's assistant response from future context, clears its conversation summary/working memory and disables its episodic memory and recorded memory descendants. A stale outcome finalizer cannot recreate that task's success context. This is not a complete causal invalidation model for every previously derived artifact or copied continuation result.

The [Undo checkpoint](evidence/2026-09-30-undo-journal/manifest.json) records 111 Rust core tests, the browser relay test, Clippy, the macOS runtime typecheck, MainView syntax parsing and repository checks. Fourteen additional test entries include the process-exit helper. The new fault matrix has 21 actual child-process exits: seven boundaries for each of file restoration, created-file removal and empty-folder removal. Raw encrypted-database inspection precedes startup repair. Sixteen injected transaction-write failures cover preparation, dispatch and verification. Other fixtures cover command deduplication across two Undo candidates, stale revisions, changed targets and parents, foreign backup ownership, content expiry, later user edits, nonempty folders, audit-store unavailability, context invalidation and legacy control repair. The real core folder workflow also exercises the new Undo command path.

Remaining compensation limits include synchronous filesystem/database calls, external rename/write races between revalidation and a syscall, original permissions/ACLs/xattrs not being captured by byte-only backups, unclassified artifact purposes, and no inverse for unsupported action types. Legacy plans without a service-prepared file identity require manual review; previously consumed legacy plans are not silently replayed. Ambiguous inverses may need manual resolution when observation cannot establish the intended state. These checks do not prove physical power-loss recovery, installed native UI behavior, Windows runtime behavior or the complete real-task acceptance matrix.

## Atomic run finalization, 30 September

`finalization.rs` owns closure for an answer, failure, Stop or interruption, including budget exhaustion and failed recovery. Before committing it reconciles each action against its durable journal. Current success requires a matching confirmed result, prepared digest and verification record; a Succeeded projection on its own is insufficient. Potentially dispatched effects remain uncertain, and verified effects survive Stop or later reporting failures.

One transaction commits the task/action revisions and cleanup results, journal changes, closure of pending decisions and stored authority, the assistant message, bounded conversation summary, working memory, eligible episodic memory and lineage, final events, audit link and finalization marker. Models do not choose the resulting task status. Events are published only after commit. Runtime grants and waiters are retired separately before the write, and the OS credential checkpoint remains separate afterward. A failure of that checkpoint cannot downgrade an already committed result.

Each explicit recovery admission advances a persisted execution-attempt counter. The finalization marker is bound to the task, attempt and kind. Repeated completion does not append another response, and an older attempt cannot replace a newer result. A resumed run updates its existing assistant message and summary entry rather than retaining a stale failure response. A later unverified worker receipt also updates the closed run's conversation in its projection transaction. Content retired through forgetting or compensation is not recreated as a new answer or success memory.

If the database write fails while the process remains alive, the run registry retains the redacted closure request within its existing 64-slot control limit. The execution owner exits and retires its grants; a read projection shows Interrupted with a persistence explanation. Resume retries that same closure without another model call. Stop can replace the pending result. Persistence retry does not advance the execution-attempt counter or redispatch a tool. The pending response is volatile: if the process exits before any successful commit, it may be lost, and startup reports interruption rather than inventing it.

After the knowledge schema is ready, startup closes unfinalized interrupted/cancelled runs in batches of at most 128. It normalizes effect evidence, closes durable prompts/authority and repairs the conversation projections without invoking a tool or model. Previously finalized runs are idempotent, and runs with an Undo record retain their compensation projection. Scope Stop intent is still saved before these per-run transactions, so failure to save one member cannot reopen a stopped continuation scope. This is not a single SQL transaction over every member of a multi-run scope.

The [finalization checkpoint](evidence/2026-09-30-run-finalization/manifest.json) records 122 Rust core tests plus the browser relay test and Clippy. Eleven added test entries include the process-exit helper. Sixteen injected writes cover action journal, decisions, approvals, capabilities, legacy prompt settings, tasks/actions, messages, working memory, conversation summary, memory/lineage, events, audit and marker boundaries. Eighteen child processes exit at nine boundaries for completion and Stop. Raw database inspection precedes startup repair. A live core fixture forces answer-storage failure, proves Resume saves the same answer with one provider call, proves Stop can replace it, and proves an audit failure after commit preserves the answer. Other fixtures cover old-attempt rejection, invalid verification claims, late receipts, privacy retirement, repeated completion and all 64 pending control slots.

These checks do not establish immediate interruption of synchronous OS/SQL calls, disk power-loss behavior, complete causal invalidation of derived content, bounded total history, or installed native/provider behavior. Native sources and protocol were unchanged in this checkpoint; their earlier typecheck does not become new UI/runtime acceptance. Routine task-state changes, legacy migration, all external effect protocols and their cross-component failure matrices still require broader review. The overall architecture and all three real-task acceptance groups remain open.
