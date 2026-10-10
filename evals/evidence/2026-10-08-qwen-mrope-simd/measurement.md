# Rejected Qwen MRoPE batch-SIMD experiment

Date: 2026-10-08

Host: Apple M3, Mac15,3, arm64, 8 GiB RAM

Toolchain: rustc 1.98.0 (Homebrew), optimized Rust release test binary

## Method

The experiment applied one prepared table to the pinned Qwen attention shape: 16 query heads, four KV heads, 256 values per head, and 64 rotary dimensions. The scalar baseline called Sage's existing validated per-head implementation; the candidate batched all 20 heads through a first-party four-lane NEON kernel. Each of 101 samples performed 32 alternating iterations and reset both paths from the same input inside the timed region. Exact-output parity passed before the experiment was removed.

Command:

```sh
cargo test --locked -p sage-inference-math --release --lib \
  tests::qwen_mrope_simd_head_batch_latency_measurement \
  -- --ignored --exact --nocapture
```

## Results

| Release process | Scalar p50 / p95 (ns) | Batch NEON p50 / p95 (ns) | Scalar / NEON p50 |
|---|---:|---:|---:|
| 1 | 2,485 / 2,546 | 3,359 / 3,436 | 0.740× |
| 2 | 4,574 / 6,455 | 5,774 / 8,554 | 0.792× |
| 3 | 2,434 / 2,566 | 3,089 / 3,312 | 0.788× |

The median process-level scalar/NEON ratio was 0.788× at p50 and 0.755× at p95: the candidate was about 27% slower at p50 and 33% slower at p95. The candidate was slower in all three runs, so it was removed from the runtime and kernel crates. The existing scalar path remains active.

## Limits

This isolated operation benchmark does not measure transformer-token latency or a real checkpoint. It establishes that this particular batch-SIMD formulation regressed on the current Apple M3; it does not rule out a better angle layout or a different hardware-specific implementation.
