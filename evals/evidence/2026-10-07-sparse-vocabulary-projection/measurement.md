# Sparse constrained vocabulary projection

Date: 7 October 2026  
Host: Apple M3, 8 GiB unified memory  
Build: Rust release profile, evaluation feature enabled

## Method

The microbenchmark compares the full Q4 output-head projection with projection of 64 allowed rows. The synthetic packed matrix uses the pinned Qwen3.5-4B text geometry: 248,320 vocabulary rows, 2,560 columns, and groups of 128 values. The fixture is generated directly in packed Q4 form so setup does not allocate a multi-gigabyte f32 tensor. Input and output buffers are reused. Each run alternates execution order over 101 samples per path, checks every selected output against the corresponding dense output, and reports median and p95 latency.

Command:

```sh
cargo test -p sage-core --features qwen35-evaluation qwen_sparse_output_head_latency_measurement --release -- --ignored --nocapture
```

## Results

| Run | Dense p50 | Dense p95 | 64-row p50 | 64-row p95 |
|---|---:|---:|---:|---:|
| 1 | 63,328 µs | 66,251 µs | 31 µs | 62 µs |
| 2 | 58,386 µs | 62,198 µs | 29 µs | 56 µs |
| 3 | 56,810 µs | 59,824 µs | 29 µs | 63 µs |
| Across-run median | 58,386 µs | 62,198 µs | 29 µs | 62 µs |

For this constrained synthetic workload, projecting only the allowed rows was about 2,000× faster at p50 and 1,000× faster at p95 than projecting the whole output vocabulary. The AArch64 path uses Sage's NEON Q4 row kernel for selected and dense projection. The timed fixture is CPU-only; it does not measure Metal dispatch, full model generation, checkpoint loading, or task quality.

## Sparse/dense crossover

A second benchmark swept larger allowed sets over three independent release runs, with 31 alternating samples per path and size. The table reports the median of the three run-level medians; large-set p95 varies with host scheduling.

```sh
cargo test -p sage-core --features qwen35-evaluation qwen_sparse_output_head_crossover_measurement --release -- --ignored --nocapture
```

| Allowed rows | Dense p50 | Selected p50 | Dense p95 | Selected p95 |
|---:|---:|---:|---:|---:|
| 4,096 | 58,423 µs | 1,754 µs | 66,474 µs | 2,289 µs |
| 16,384 | 58,434 µs | 7,267 µs | 60,747 µs | 7,710 µs |
| 32,768 | 54,843 µs | 13,085 µs | 60,741 µs | 14,963 µs |
| 65,536 | 57,663 µs | 24,985 µs | 67,723 µs | 31,827 µs |
| 131,072 | 55,176 µs | 31,966 µs | 75,794 µs | 54,211 µs |

The decoder uses selected projection through half of its vocabulary and dense projection above that. The sweep found selected projection faster even at 131,072 rows (about 53% of the 248,320-token vocabulary), so the one-half cutoff leaves a small safety margin for the measured shape.

## Limits

This measures the output projection only. It does not establish end-to-end generation speed or numerical parity with the pinned Qwen checkpoint. The current 8 GiB development Mac cannot close the plan's 16 GiB qualification gate. The constrained route is used only when the live grammar provides a valid allow-list no larger than half the vocabulary; unconstrained and larger allow-lists retain dense projection. Selection preserves sorted token order and the dense path's lower-token-ID tie break.
