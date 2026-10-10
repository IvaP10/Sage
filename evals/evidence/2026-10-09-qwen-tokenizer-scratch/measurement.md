# Qwen tokenizer prompt-scoped scratch and compact BPE indices

Date: 2026-10-09  
Host: Apple M3 (Mac15,3), arm64, 8 GiB RAM  
Toolchain: rustc 1.98.0, optimized Rust release test binary

## Change

Prompt encoding now reuses its character-boundary, pre-token-range, BPE-node, and merge-heap buffers across message fragments and pieces. The BPE node links and heap indices use `u32` plus a sentinel because input is bounded to 256 KiB. On 64-bit targets, the prior `Option<usize>` node layout occupied 48 bytes and the prior merge-candidate layout occupied 32 bytes; the new records occupy 20 bytes each. Pre-token range storage starts at 256 entries maximum and grows only when observed splits need more. A proved lower bound rejects a split count that already exceeds the remaining token budget.

The release fixture encodes 3,500 bytes (`"abcd "` repeated 700 times) to 1,421 tokens using three synthetic BPE merge rules. Each process records 301 samples and verifies each result against that process's expected token sequence. Three processes were measured before the change and three after it.

## Command

```sh
cargo test --locked --offline -p sage-qwen-tokenizer --lib --release tokenizer_scratch_reuse_measurement -- --ignored --nocapture
```

## Results

| Process | Previous p50 / p95 | Reused p50 / p95 |
|---|---:|---:|
| 1 | 259.875 / 510.791 μs | 254.084 / 501.708 μs |
| 2 | 204.666 / 218.459 μs | 191.500 / 208.917 μs |
| 3 | 195.583 / 213.250 μs | 191.084 / 205.750 μs |
| Median run | 204.666 / 218.459 μs | 191.500 / 208.917 μs |

The median run-level p50 and p95 fell 6.4% and 4.4%. The first process in each group had substantial scheduling outliers; the median-of-run comparison is the reported result. This fixture shows a modest encoding-time improvement alongside the deterministic working-record size reduction.

## Limits

The fixture uses synthetic merges and does not reproduce the pinned Qwen tokenizer's merge distribution. The exact-checkpoint tokenizer parity test remains ignored because the pinned `tokenizer.json` is unavailable in this checkout. These measurements exclude Core prompt construction, decoder prefill, model generation, task quality, and the 16 GiB hardware gate. The development host has 8 GiB RAM.
