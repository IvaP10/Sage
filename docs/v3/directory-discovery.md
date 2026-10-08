# Scoped directory discovery

The planner previously received neither the selected folder paths nor a tool to discover their contents. A request such as “read the text document in this folder” therefore depended on a filename supplied outside the actual tool flow. `list_directory` now supplies bounded names and types through the existing native authorization and verification path. It is nonrecursive and reads no child file contents.

## Request and result

The closed tool schema accepts an absolute `path`, `page_size` from 1 to 64, and `cursor` (null for the first page). The resolver canonicalizes the target. A current Read scope can cover the operation; an unscoped operation still needs an exact action approval. A Create scope does not imply Read. Every page receives its own prepared action and single-use file-read grant.

The result contains:

| Field | Meaning |
| --- | --- |
| `version`, `path`, `directory_identity` | Format version 1, canonical target and native directory identity |
| `snapshot_sha256` | Digest of directory identity and the sorted raw names/types, not child file contents |
| `total_entries`, `offset` | Number of immediate entries and the zero-based start of this page |
| `entries` | Names, their encoding, and file/directory/symlink/other classification |
| `next_cursor` | Token for the next page; null when no entries remain after this page |
| `complete` | True only when this single page contains the entire directory, including an empty directory |

The last page of a multipage listing has `next_cursor: null` and `complete: false`. Consumers accumulate pages using their offsets and snapshot digest. A returned page can contain fewer entries than requested because of its byte limit.

Cursors encode a version, target digest, snapshot digest and offset. They confer no access and do not bypass scope, policy or grant checks. A cursor for another directory, an invalid range or a changed listing fails with an instruction to start again from null. Changing the requested page size is permitted. The service reports the actual offset; a caller choosing a later offset cannot obtain a page claiming it included the skipped entries.

## Native execution and evidence

`FileBroker` prepares the target using directory handles and records its native identity. `directory.rs` opens the final component without following a link, validates the handle before and after enumeration, sorts native name bytes and builds the bounded page. Entry classification does not traverse links or inspect their targets. A selected path may already have been canonicalized by the resolver; policy and scope apply to that resolved target.

The deterministic observer independently opens and enumerates the directory again. It requires the fresh page to equal the returned page and match the prepared identity. Verification binds the expected target, page size and cursor. The transaction reducer binds the returned page digest to that same evidence item before committing Confirmed output; evidence for a different page cannot supply its digest. Existing action/result/event/audit transactions and retention revocation apply.

Recovery can confirm a retained result only by repeating this independent observation. Changed or retired results cannot be reconstructed as successful pages from a summary. A listing is an observation of names and types at that time: it is not a durable filesystem snapshot or permission to read or mutate a discovered child. Subsequent file reads prepare their own identities and undergo their own checks.

## Work and cancellation limits

| Limit | Current value |
| --- | --- |
| Requested entries per page | 1–64 |
| Serialized page | At most 12 KiB |
| Immediate directory entries admitted | At most 16,384 |
| Raw filename bytes admitted | At most 4 MiB; allocations and encoded representations add overhead |
| Cooperative inspection time | 3 seconds per enumeration |
| Concurrent blocking readers | 4 across execution and observation |

The reader acquires a slot before spawning blocking work. The worker retains that slot until it exits, including when the run is cancelled. Dropping the awaiting future sets a cancellation flag checked during enumeration, hashing/page construction boundaries and before work starts. Grant expiry is checked when the execution worker starts after waiting for capacity. These checks cannot interrupt a blocked filesystem syscall.

Each page currently enumerates and sorts the whole bounded directory, and independent observation repeats that work. This trades repeated work for a simple, current identity check; it is not an indexed large-directory implementation. Exceeding an admission or time limit returns an error, not a partial listing marked complete. Filesystem preparation, other file tools, database work and credential-store access are not moved off the control runtime by this change.

## Planner context and privacy

`context.rs` supplies a bounded `authorized_file_roots` data item from the current validated run contract. It includes exact paths, allowed effects, expiry and the number of omitted roots. Missing or expired contracts produce an unavailable view. Nonabsolute, non-Unicode or redaction-changing paths are omitted. This data stays separate from trusted constraints and cannot grant authority. It is placed first so the existing context clipping does not split a valid scope view.

Ordinary Unicode names are preserved. Recognized secret-like names are replaced with a redacted marker. Native names that cannot be represented as Unicode are explicitly labelled `native_base64`; known secret patterns are checked before encoding. The codec is not a general secret detector, and encoded or redacted names cannot yet be used as JSON path arguments by the existing file tools. No lossy replacement filename is presented as a valid target. The local macOS filesystem rejects the invalid-byte fixture, so its encoding/redaction boundary is tested directly; the Linux-only filesystem fixture is not claimed as locally executed.

Directory results carry the existing private task label. External provider release still follows the current disclosure/approval path. Filenames are untrusted data, never instructions. Listing a selected parent may reveal the name of a protected internal child, but entering that child remains denied. Credential-path policy remains active. Directory pages are retained in task results rather than separate content artifacts; existing retention revocation removes their output when applicable. Full privacy deletion and precise whole-prompt budgeting remain separate open work.

## Evidence and remaining acceptance

The [30 September checkpoint](evidence/2026-09-30-directory-discovery/manifest.json) records the current test and source hashes. The new fixtures cover pagination and byte limits; empty, changed and oversized directories; invalid and cross-target cursors; links and replaced directories; Unicode/redacted/encoded names; fresh observation and substituted output; reader cancellation and capacity; scope views and protected children.

A core integration fixture uses a scripted provider that receives only the selected folder, discovers a randomly named text document through listing, reads its body and answers. Its 35-page case crosses the 32-action budget, continues with fresh run authority and consumes the retained cursor. The read-only scope causes no repeated action-approval prompt. This demonstrates an actual temporary-filesystem/core flow, not natural-language model quality or native UI acceptance.

Native UI, Windows execution, actual provider disclosure and installed-browser flows remain unqualified. Opaque handles for non-Unicode names, retrieval of retained content beyond previews, indexed larger-directory discovery, exact context budgets and the three requested real-task acceptance groups remain open. This checkpoint does not close the overall rebuild.
