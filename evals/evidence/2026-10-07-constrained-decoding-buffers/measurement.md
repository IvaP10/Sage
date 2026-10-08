# Reusable constrained-decoding buffers

Date: 7 October 2026  
Host: Mac15,3, 8 GiB unified memory  
Build: Rust release profile, `qwen35-evaluation` enabled

## Method

The decoder previously built a `BTreeSet<u32>` for every generation step and visited all 248,044 text vocabulary entries to build a mask. It now indexes text token IDs into 256 first-byte buckets, merges only grammar-permitted buckets in token-ID order with a reusable heap/cursor workspace, and writes valid IDs into one reusable `Vec<u32>`. Dense and sparse constrained selection walk only the ordered allow-list, preserving the lower-ID tie break. The release probes compare these paths with local references to the previous tree algorithms.

Each case alternated both implementations over 31 samples per path in three independent release runs and reported the median of run-level p50 and p95 values. The selection probe used 248,320 finite synthetic logits and three sorted allow-lists larger than half the vocabulary. The mask probe used 248,044 synthetic one-byte pieces: 10% started with the root grammar's permitted digit `1`, while the remainder started with an invalid `x`; both paths ran the same parser clone/check for each viable candidate. The indexed path reused its vocabulary-merge workspace and allow-list buffer. These are algorithm microbenchmarks, not measurements of the pinned tokenizer, checkpoint, or full generation.

Commands:

```sh
cargo test -p sage-core --features qwen35-evaluation --lib indexed_constrained_mask_measurement --release -- --ignored --nocapture
cargo test -p sage-core --features qwen35-evaluation --lib constrained_token_selection_buffer_measurement --release -- --ignored --nocapture
```

## Results

### Mask construction

| Runs × samples | Full-scan tree p50 | Full-scan tree p95 | Indexed reusable p50 | Indexed reusable p95 |
|---:|---:|---:|---:|---:|
| 3 × 31 | 2,071 μs | 2,916 μs | 323 μs | 943 μs |

First-byte indexing and scratch reuse measured 6.4× faster at p50 and 3.1× at p95 in this synthetic root-JSON mask. Run-level medians were tree/indexed p50/p95 of 2,025/2,916 vs. 307/943 μs, 2,078/3,056 vs. 323/464 μs, and 2,071/2,613 vs. 692/1,412 μs.

### Dense constrained selection

| Allowed IDs | Tree p50 | Tree p95 | Sorted-slice p50 | Sorted-slice p95 |
|---:|---:|---:|---:|---:|
| 131,072 | 7,271 μs | 8,429 μs | 144 μs | 157 μs |
| 180,000 | 7,970 μs | 9,410 μs | 198 μs | 216 μs |
| 223,488 | 7,612 μs | 8,027 μs | 244 μs | 261 μs |

The sorted-slice path measured 31–51× faster at p50 and 31–54× at p95 in these dense constrained-selection cases. It also checks finiteness only for eligible logits, since disallowed outputs cannot influence the selected token. Exact candidate generation, prompt/decode latency, loaded weights, model quality, and memory pressure were not measured here. The current 8 GiB host remains below the plan's 16 GiB hardware qualification target.
