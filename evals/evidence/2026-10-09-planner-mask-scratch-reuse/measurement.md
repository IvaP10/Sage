# Constrained-decoder schema-first candidate pruning

Original measurement: 9 October 2026; candidate-bucket measurement: 10 October 2026  
Host: Apple M3 (Mac15,3), arm64, 8 GiB RAM  
Toolchain: rustc 1.98.0, optimized Rust release test binary

## Current reproduction command

```sh
cargo test --locked --offline --release -p sage-constrained-generation --lib generation::tests::schema_first_candidate_prefilter_measurement -- --ignored --exact --nocapture --test-threads=1
```

## Tokenizer-bucket measurement

This fixture exercises Sage's actual `TextPieceCandidateWorkspace` bucket merge using a synthetic table of 32,768 token IDs that repeats nine text-piece shapes. The allocating reference asks the workspace for JSON-grammar-permitted first-byte buckets. The schema-first route intersects that same JSON mask with the active planner-schema mask before the workspace selects buckets. Both iterators preserve token-ID order, reuse their workspace across samples, and return identical allow-lists. Each process collects 51 alternating samples per route; each row is an independent release-test process.

| Process | Allocating p50 / p95 (μs) | Schema-first p50 / p95 (μs) | JSON candidates | Schema candidates |
|---|---:|---:|---:|---:|
| 1 | 5,837 / 7,763 | 720 / 926 | 29,127 | 3,640 |
| 2 | 5,698 / 5,841 | 721 / 734 | 29,127 | 3,640 |
| 3 | 5,967 / 6,124 | 730 / 760 | 29,127 | 3,640 |
| Median run | 5,837 / 6,124 | 721 / 760 | 29,127 | 3,640 |

The schema-first route reduced median run-level p50 and p95 by 87.6% and 87.6%. It reduced the number of candidates yielded to the mask by 87.5%, with all 3,640 admitted IDs preserved. A regression checks every possible next byte after six representative prefixes and confirms the intersected mask never rejects a transition both automata can extend.

## Earlier per-candidate measurements

Before schema filtering, the allocating reference and reusable `clone_from` route each scanned all 32,768 synthetic candidates:

| Process | Allocating p50 / p95 (μs) | Reused p50 / p95 (μs) |
|---|---:|---:|
| 1 | 5,520 / 6,592 | 4,120 / 4,734 |
| 2 | 5,555 / 5,595 | 4,199 / 4,225 |
| 3 | 5,469 / 5,551 | 4,069 / 4,117 |
| Median run | 5,520 / 5,595 | 4,120 / 4,225 |

A later schema-first per-candidate check, also over the full list rather than tokenizer buckets, measured median p50/p95 of 5,303/5,425 μs for the allocating reference and 830/869 μs for the reusable route. The current bucket measurement supersedes those timings for the active candidate-selection path; the earlier values remain useful only as historical isolated comparisons.

## Limits

The synthetic table repeats nine pieces rather than using the pinned tokenizer's full vocabulary distribution. It excludes model logits, complete generation, task quality, model loading, and end-to-end latency. The first-byte intersection is a conservative necessary condition; the full JSON parser, planner-schema automaton, and independent Core validator remain authoritative. Schema transition state still allocates. This isolated result does not establish full-generation speed or satisfy 16 GiB hardware qualification.
