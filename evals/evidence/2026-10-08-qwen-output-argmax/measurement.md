# Fused Qwen output-head greedy argmax

Date: 8 October 2026  
Host: Apple M3 (Mac15,3), 8 GiB RAM  
Compiler: Rust 1.98.0

## Measurement

The synthetic fixture uses the pinned Qwen3.5 vocabulary row count (151,936), hidden width (2,560), and grouped-Q4 output weights (group size 128). It compares the existing dense route—parallel Q4 projection into a reusable 607,744-byte f32 vector followed by the dense greedy finite check and argmax scan—with a fused route that computes the same per-row scalar/NEON projections and reduces each worker's winning `(token, logit)` pair. Both routes use the same deterministic synthetic weight and activation values. Each process runs one warmup per route, then 51 dense and 50 fused measurements in alternating order.

| Run | Dense p50 / p95 (µs) | Fused p50 / p95 (µs) | p50 speedup |
|---|---:|---:|---:|
| 1 | 7,793 / 8,727 | 7,638 / 8,521 | 1.020× |
| 2 | 9,680 / 11,254 | 9,570 / 11,094 | 1.011× |
| 3 | 7,697 / 9,639 | 7,584 / 10,077 | 1.015× |

The median paired p50 speedup is 1.015× (about 1.5%). p95 improved in two runs and regressed in one, so this evidence does not support a consistent tail-latency claim. The fused route also avoids the 607,744-byte logits scratch and a second vocabulary traversal on unconstrained CPU decoding.

Reproduce with:

```sh
cargo test --locked --release -p sage-kernels --lib tests::qwen_output_head_fused_argmax_latency_measurement -- --ignored --exact --nocapture --test-threads=1
```

This is a synthetic kernel measurement on the development Mac, which has 8 GiB RAM. It does not use the pinned checkpoint, measure end-to-end model generation, establish task quality, or satisfy 16 GiB hardware acceptance. The evaluation-only Metal output path retains its prior projection-and-selection route.


The release-only benchmark remains available in source and currently compares dense projection plus scan with scoped fused argmax. The separate persistent output-head pool was retired after the pinned-toolchain reevaluation recorded in [the historical pool experiment](../2026-10-08-q4-persistent-output-workers/measurement.md).

## Pinned Rust 1.89 benchmark revalidation

After restoring the dense-versus-scoped-fused release benchmark, three independent processes on the same Apple M3 host collected 51 alternating samples per route. The fixture uses deterministic Q4 weights, scales and activations. Correctness assertions are outside the timed region, and every dense winner matched fused argmax exactly.

| Run | Dense p50 / p95 (μs) | Fused p50 / p95 (μs) |
|---|---:|---:|
| 1 | 7,510 / 9,451 | 7,390 / 8,816 |
| 2 | 8,154 / 16,197 | 7,728 / 12,409 |
| 3 | 8,819 / 14,321 | 8,408 / 13,007 |

The median run-level p50 was 8,154 μs dense and 7,728 μs fused (about 4.9% lower using the paired per-run speedup median). The median p95 was 14,321 μs dense and 12,409 μs fused, with substantial run-to-run variation. This is kernel-only evidence on the 8 GiB development Mac, not full-model generation or 16 GiB acceptance.
