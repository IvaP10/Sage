# Q4 batched vision patch projection

## Result

Three alternating optimized Rust release runs on Apple Mac15,3 (arm64, 8 GiB RAM) measured a synthetic grouped-Q4 projection with the pinned Qwen patch-embedding shape: 1,024 output rows, 1,536 input columns, group size 128, and 256 patches. Each run contains 51 single-patch baseline samples and 50 batched samples.

| Run | Baseline p50 | Baseline p95 | Batch p50 | Batch p95 | p50 speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 36.844 ms | 37.478 ms | 8.058 ms | 9.825 ms | 4.57× |
| 2 | 35.689 ms | 37.165 ms | 9.322 ms | 10.316 ms | 3.83× |
| 3 | 35.667 ms | 36.935 ms | 9.671 ms | 12.047 ms | 3.69× |

Across-run median p50/p95 was 35.689/37.165 ms for the baseline and 9.322/10.316 ms for batching, a 3.83× median p50 gain. Maximum absolute output difference was `9.69e-8` on this fixture.

## Method

The baseline projects each of 256 row-major patch vectors independently through one Q4 matrix. The batched path transposes activations in 32-patch tiles, reuses each decoded weight across the tile, uses four-lane AArch64 NEON operations, and dispatches independent tiles to at most eight bounded workers. Each invocation alternates the two paths for 101 samples (51 baseline and 50 batched); p95 uses the nearest-rank sample. The matrix values and activation values are deterministic synthetic fixtures; no Qwen checkpoint was loaded. The measurement is one patch-embedding projection only, not end-to-end image encoding, model generation, or task quality.

The retained version scans each weight once per 32-patch tile.

## Reproduction

```sh
cargo test --offline -p sage-kernels --release --lib q4_batch_vision_projection_latency_measurement -- --ignored --nocapture --test-threads=1
```

Each test invocation alternates baseline and batched calls for 101 samples (51 baseline and 50 batched), reports the p50 and nearest-rank p95 for each path, and checks the final output difference. The host has 8 GiB RAM and cannot satisfy Sage's 16 GiB qualification requirement.
