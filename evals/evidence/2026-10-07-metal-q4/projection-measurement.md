# First-party Q4 projection measurement

Date: 7 October 2026

## Environment

- Hardware: Apple M3, 10-core GPU, 8 GiB unified memory.
- macOS: 27.0.1.
- Rust: 1.98.0.
- Kernels: Sage's four-lane AArch64 NEON grouped-Q4 CPU projection with SIMD nibble unpacking, plus the runtime-compiled Metal grouped-Q4 matrix-vector projection.
- Weight format: Sage symmetric signed Q4, 128 values per scale group, 2,560 output rows by 2,560 input columns.

## Command

```sh
cargo test --release -p sage-core --features qwen35-evaluation --lib inference_cpu::tests::qwen_hidden_q4_projection_latency_measurement -- --ignored --exact --nocapture
```

The test compares both fast paths with Sage's scalar f64-accumulation Q4 reference, warms all paths twice, then alternates 21 measurements per path. CPU samples include projection and output allocation. Each Metal sample includes input/output buffer and command-buffer creation, kernel submission, and completion wait; Q4 weights and scales are uploaded once into shared Metal storage before timing. The NEON kernel expands eight packed bytes into sixteen signed nibbles per vector block, applies four independent accumulators, and handles odd row/group boundaries with bounded scalar edges.

## Results

| Path | p50 | p95 |
|---|---:|---:|
| Scalar f64 reference, run 1 | 5.011 ms | 5.856 ms |
| NEON CPU, scalar Q4 unpack, run 1 | 3.275 ms | 4.188 ms |
| Metal SIMD-row kernel, run 1 | 3.351 ms | 7.185 ms |
| Scalar f64 reference, run 2 | 4.920 ms | 4.972 ms |
| NEON CPU, SIMD Q4 unpack, run 1 | 0.714 ms | 0.730 ms |
| Metal SIMD-row kernel, run 2 | 3.197 ms | 5.535 ms |
| Scalar f64 reference, run 3 | 4.923 ms | 5.023 ms |
| NEON CPU, SIMD Q4 unpack, run 2 | 0.659 ms | 0.681 ms |
| Metal SIMD-row kernel, run 3 | 3.148 ms | 4.864 ms |

All runs passed numerical parity at relative tolerance `1e-3` plus an absolute tolerance of `1e-3`; separate kernel tests cover odd rows and crossing groups at tighter tolerance. The latest SIMD-unpack run measured NEON p50/p95 about 7.5× lower than the scalar reference. The first scalar-unpack NEON version was only about 1.5× faster, so it was replaced after measurement. The current CPU fixture also outperformed this Metal dispatch path, whose p95 included command submission and I/O buffer setup.

The reproducible command is `cargo test --release -p sage-core --features qwen35-evaluation --lib inference_cpu::tests::qwen_hidden_q4_projection_latency_measurement -- --ignored --exact --nocapture --test-threads=1`. An earlier fused-f32 scalar experiment measured slower than the f64 reference and was removed. These microbenchmarks are short and synthetic; they do not establish a full-model speedup.

## Attention kernels

The full-attention implementation uses Sage's `dot_rows` NEON kernel for batched Q·K head scores and `weighted_sum_rows` for position-major value accumulation. The scalar f64 implementations remain the comparison reference. Separate release tests use 4,096 rows and report 21 alternating samples per path:

| Fixture | Scalar p50/p95 | NEON p50/p95 |
|---|---:|---:|
| Q·K, row stride 512, head width 256 | 0.785 / 0.899 ms | 0.123 / 0.178 ms |
| Weighted values, row stride 512, head width 128 | 0.270 / 0.545 ms | 0.055 / 0.114 ms |

Both kernels matched independent scalar results within `2e-5` absolute plus relative tolerance in odd-width/head-offset tests. These fixtures do not include softmax, rotary embeddings, Qwen projection layers, end-to-end context growth, or a real model checkpoint.

The NEON weighted-value path now uses fixed stack accumulators for heads up to 512 values and supports writing directly into caller-owned output. Qwen attention uses this `weighted_sum_rows_into` path to fill its final attention buffer, avoiding per-head accumulator and result-vector allocations; wider generic heads retain the scalar path. A post-change release repeat on the same fixture measured scalar p50/p95 at 0.265/0.274 ms, the allocating wrapper around NEON at 0.042/0.045 ms, and the reused-output NEON path at 0.042/0.042 ms. The small fixture does not show a measurable latency difference between the allocating wrapper and reused-output form, but the latter removes per-head allocations in the decoder. These are synthetic CPU measurements.

Reproduce the attention fixtures with:

```sh
cargo test --release -p sage-kernels --locked --offline tests::attention_query_key_kernel_latency_measurement -- --ignored --exact --nocapture --test-threads=1
cargo test --release -p sage-kernels --locked --offline tests::attention_value_kernel_latency_measurement -- --ignored --exact --nocapture --test-threads=1
```

## Gated-delta recurrence

`GatedDeltaState::step` routes each AArch64 value head through Sage's fused row-major NEON recurrence update. It computes the key-state read across contiguous value columns, forms the delta correction, updates state in place, and accumulates the query read in the same pass. The f64 scalar implementation remains active on other architectures. Five-token state/output parity tests cover multiple heads and dimensions with scalar tails. A separate release fixture processes 32 heads with 128×128 state matrices for 21 alternating samples per path:

| Run | Scalar p50/p95 | NEON p50/p95 |
|---|---:|---:|
| 1 | 1.234 / 1.877 ms | 0.241 / 0.348 ms |
| 2 | 0.783 / 0.927 ms | 0.145 / 0.240 ms |

The synthetic fixture shows a repeated p50 improvement of roughly 5×. It excludes Q/K/V projections, convolution, normalization, full model execution, and real-checkpoint quality tests.

Reproduce the recurrence fixture with `cargo test --release -p sage-kernels --locked --offline tests::gated_delta_layer_latency_measurement -- --ignored --exact --nocapture --test-threads=1`.

The NEON implementation now keeps its 128-block accumulator and 512-value correction scratch in fixed stack arrays. `GatedDeltaState::step` likewise uses zeroizing fixed stack arrays for normalized key/query scratch. This removes 66 temporary heap-vector allocations per 32-head token (two scratch vectors per head in the kernel, plus two normalization vectors for the layer call) while keeping the returned output allocation and recurrent state unchanged. The combined maximum scratch array footprint is 14 KiB at the validated 512-wide dimensions. Three release repeats after the change measured scalar/NEON p50/p95 at 1.130/1.389 ms vs 0.205/0.313 ms, 0.779/0.827 ms vs 0.139/0.229 ms, and 0.789/0.878 ms vs 0.143/0.239 ms. The repeat median is 0.143/0.239 ms for NEON, essentially unchanged from the prior 0.145/0.240 ms run; this change reduces allocator traffic but does not establish a further latency gain at this fixture size.

## KV allocation behavior

`KvCache::new` now leaves key and value backing vectors empty. The cache reserves the validated prompt prefix before text or mixed image/text prefill, then doubles capacity as generated tokens exceed the existing prefix. Reallocation copies into a bounded replacement and zeroes the old allocation before releasing it. The focused test `inference_cpu::tests::kv_cache_reserves_only_observed_context_and_grows_geometrically` verifies zero constructor capacity, the 16-position initial allocation, doubling, context-limit rejection, and clear behavior.

For the pinned geometry, the former eager allocation at the full 8K bound was 8192 positions × 4 KV heads × 256 values × 2 (K and V) × 4 bytes × 8 full-attention layers = 512 MiB. Decoder construction now allocates zero KV bytes. Each observed prompt position reserves 64 KiB across those eight layers, so prompts of 1–15 tokens reserve 64–960 KiB; larger prompts reserve their exact validated prefix. This is an allocation-bound calculation confirmed by capacity tests, not an OS resident-memory measurement. The maximum-context governor estimate remains conservative, and an actual checkpoint run is still required to evaluate peak memory and latency.

These tests do not measure checkpoint loading, full Qwen text or vision generation, end-to-end latency, task quality, quantization quality, peak candidate memory, or the 16 GiB acceptance tier. The device has 8 GiB; no real checkpoint shard was loaded. Other normalization and vision operators and GPU batching remain unmeasured or unimplemented. Results are kernel-level development measurements, not product performance claims.
