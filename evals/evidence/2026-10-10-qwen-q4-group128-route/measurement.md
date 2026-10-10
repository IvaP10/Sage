# Qwen Q4 group-128 kernel and batch-route measurements

Date: 10 October 2026. Host: Apple Mac15,3 (arm64, 8 GiB physical memory). Rust: pinned 1.89.0. All runs used the optimized release profile. Fixtures contain deterministic synthetic Q4 weights and inputs, not checkpoint tensors.

## Kernel change

The Metal shader previously performed two runtime group-index divisions and loaded two scales for every packed Q4 byte pair. For the pinned Qwen group size of 128, columns are aligned to the 128-value group boundary and the low/high nibbles share one scale. The guarded fast path now computes the row's scale base once, uses a 7-bit shift for the group index, and loads that scale once per pair. Other group sizes and unaligned geometries retain the original generic path. Complexity remains linear in the projection size; this reduces inner-loop indexing and memory operations.

## Single-input shape sweep

Six Qwen-shaped projections were tested: `256×2,560`, `1,280×2,560`, `2,560×2,560`, `5,120×2,560`, `10,240×2,560`, and `2,560×10,240`. Each run alternated CPU and Metal for 31 calls per shape, checked every output within `1e-3 + 1e-3 × abs(reference)`, and emitted sorted nanosecond samples. The table reports medians of five independent process-level p50/p95 values.

| Rows × columns | CPU p50 / p95 | Metal p50 / p95 | Metal / CPU p50 |
| --- | ---: | ---: | ---: |
| 256 × 2,560 | 52 / 62 μs | 341 / 830 μs | 6.56× |
| 1,280 × 2,560 | 99 / 128 μs | 321 / 1,904 μs | 3.24× |
| 2,560 × 2,560 | 175 / 231 μs | 336 / 1,903 μs | 1.92× |
| 5,120 × 2,560 | 365 / 467 μs | 660 / 3,215 μs | 1.81× |
| 10,240 × 2,560 | 551 / 661 μs | 1,020 / 2,994 μs | 1.85× |
| 2,560 × 10,240 | 566 / 601 μs | 1,066 / 2,569 μs | 1.88× |

The prior group-index path measured about 11.2 ms p50 for `10,240×2,560`; the specialized route now measures 1.02 ms across process medians. That comparison is directional because the prior baseline was one process, while the updated result uses five.

Raw samples from all five optimized runs are in [`shape-run-1.txt`](../2026-10-10-q4-metal-route-sweep/shape-run-1.txt), [`shape-run-2.txt`](../2026-10-10-q4-metal-route-sweep/shape-run-2.txt), [`shape-run-3.txt`](../2026-10-10-q4-metal-route-sweep/shape-run-3.txt), [`shape-run-4.txt`](../2026-10-10-q4-metal-route-sweep/shape-run-4.txt), and [`shape-run-5.txt`](../2026-10-10-q4-metal-route-sweep/shape-run-5.txt).

## Matrix-level prefill batch route

`QuantizedQ4Matrix::project_batch_into` previously looped over batch members and submitted one synchronous Metal projection per input. It now calls the Metal batch kernel once for the whole row-major input batch. Batch sizes above one receive a bounded, exact-size activation workspace for that call; the batch-one path reuses its existing workspace. The measurement below exercises that matrix-level API, so Metal timings include workspace allocation and release.

The Qwen MLP-shaped `10,240×2,560` fixture was tested at batch sizes 1, 4, 16, 64 and 256. Three independent release processes each collected 31 alternating CPU/Metal calls at each size. Every output matched the CPU reference within the same Q4 tolerance. Values are medians of process-level p50/p95 measurements.

| Batch | CPU p50 / p95 | Metal p50 / p95 | Metal / CPU p50 |
| ---: | ---: | ---: | ---: |
| 1 | 729 / 841 μs | 1,291 / 4,136 μs | 1.77× |
| 4 | 4,851 / 5,720 μs | 1,697 / 3,620 μs | 0.35× |
| 16 | 23,021 / 26,705 μs | 6,594 / 8,275 μs | 0.29× |
| 64 | 52,963 / 56,468 μs | 19,776 / 21,489 μs | 0.37× |
| 256 | 126,384 / 137,722 μs | 81,580 / 82,733 μs | 0.64× |

The matrix-level Metal path wins from batch 4 onward for this Qwen shape, including per-call workspace cost; CPU remains faster for one input. Raw logs are in [`qwen-batch-run-1.txt`](../2026-10-10-q4-metal-route-sweep/qwen-batch-run-1.txt), [`qwen-batch-run-2.txt`](../2026-10-10-q4-metal-route-sweep/qwen-batch-run-2.txt), and [`qwen-batch-run-3.txt`](../2026-10-10-q4-metal-route-sweep/qwen-batch-run-3.txt).

## Scope

These results support the group-128 kernel specialization and correct batch submission; they do not enable the Metal route by default or implement automatic route selection. Single-input Metal remains slower on this fixture, other matrix shapes and devices need their own evidence, and complete decoder execution may change the tradeoff. The work does not establish real-checkpoint parity, task quality, whole-model latency, energy use, 16 GiB acceptance, or product-worker generation.
