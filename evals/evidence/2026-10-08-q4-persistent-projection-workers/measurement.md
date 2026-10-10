# Persistent Q4 layer-projection workers and paired-row activation reuse

Date: 8 October 2026  
Host: Apple M3 (Mac15,3), 8 GiB RAM  
Compiler: Rust 1.89.0 (workspace-pinned toolchain)

## Initial row-wise pool measurement (historical)

The original fixture used Qwen's representative MLP projection geometry: 11,008 output rows, 2,560 input values, and grouped-Q4 weights with group size 128. Each independent release process collected 60 alternating samples after warmup:

| Run | Scoped p50 / p95 (μs) | Pooled p50 / p95 (μs) | First pooled call (ms) |
|---|---:|---:|---:|
| 1 | 902 / 1,240 | 825 / 1,154 | 1.096 |
| 2 | 1,194 / 2,376 | 1,091 / 2,018 | 1.062 |
| 3 | 754 / 1,271 | 710 / 1,054 | 0.707 |

The original row-wise implementation measured median p50/p95 reductions of 8.5%/9.2% versus temporary scoped workers, with exact output parity. This historical code path was later replaced by the paired-row NEON tile described below; the original samples are retained for experiment history.

## Paired-row NEON tile

For Qwen's aligned grouped-Q4 matrices, the AArch64 kernel now computes two adjacent output rows together. It loads each 16-value activation tile once and reuses it across both rows, while preserving independent accumulators, per-row scales, the existing reduction order and exact scalar fallbacks. The specialized path is limited to group sizes divisible by 16 and row widths aligned to their group size; odd worker boundaries and other shapes use the reference row kernel.

Three pinned-Rust 1.89 release processes compared the former row kernel with the paired tile on an 11,008×2,560, group-128 fixture. Both routes used the same eight bounded scoped workers and 61 alternating samples per route. Each output matched bit for bit.

| Run | Row kernel p50 / p95 (μs) | Paired tile p50 / p95 (μs) |
|---|---:|---:|
| 1 | 588 / 825 | 585 / 822 |
| 2 | 586 / 703 | 573 / 714 |
| 3 | 872 / 953 | 858 / 927 |

The median paired per-run p50 speedup was 1.6%; p95 was noisy and did not improve in every run. This is a small isolated kernel gain, not a full decoder result.

## Current persistent-pool comparison

After enabling the paired tile, three release processes compared current scoped and persistent-pool routes using the same representative MLP geometry and 60 alternating samples per route. Numerical outputs matched exactly.

| Run | Scoped p50 / p95 (μs) | Pooled p50 / p95 (μs) | First pooled call (ms) |
|---|---:|---:|---:|
| 1 | 1,045 / 2,855 | 1,024 / 2,215 | 2.070 |
| 2 | 940 / 1,334 | 842 / 1,220 | 1.144 |
| 3 | 889 / 1,486 | 831 / 1,283 | 0.920 |

The median run-level p50 was 940 μs scoped and 842 μs pooled (10.4% lower); median p95 was 1,486 μs scoped and 1,283 μs pooled (13.7% lower). Both current routes include the paired-row kernel. The first pooled call includes worker initialization and one projection; median startup plus projection was 1.144 ms.

The persistent pool owns a bounded set of first-party workers, shares immutable Arc-backed weights, and writes disjoint output ranges directly into the caller's buffer. Its synchronous contract waits for every submitted range before returning. If pool admission fails, Sage waits for submitted work and recomputes through the scoped implementation.

Reproduce the paired-kernel comparison with:

```sh
cargo test --offline --locked --release -p sage-kernels --lib tests::qwen_mlp_q4_two_row_activation_reuse_latency_measurement -- --ignored --exact --nocapture --test-threads=1
```

Reproduce the current scoped-versus-pool comparison with:

```sh
cargo test --offline --locked --release -p sage-kernels --lib tests::qwen_mlp_q4_projection_worker_pool_latency_measurement -- --ignored --exact --nocapture --test-threads=1
```

These synthetic layer-kernel measurements do not establish pinned-checkpoint parity, end-to-end generation speed, model quality, Windows runtime performance, or 16 GiB hardware acceptance.
