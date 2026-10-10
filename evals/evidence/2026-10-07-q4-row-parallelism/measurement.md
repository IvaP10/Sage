# Q4 row-parallel projection

Date: 7 October 2026  
Host: Mac15,3, 8 GiB unified memory  
Build: Rust release profile, `qwen35-evaluation` enabled

## Method

The first-party Q4 kernel now splits large independent output-row ranges across up to eight scoped CPU workers. It stays single-threaded below 256 rows or 2,000,000 matrix elements, retains the AArch64 NEON row kernel, and falls back to the caller thread if the operating system refuses worker creation. Selected-row projection uses the same bounded worker policy while preserving the caller's requested row order.

The existing Qwen hidden-size benchmark uses a synthetic 2,560×2,560 Q4 matrix, verifies CPU and Metal results against Sage's scalar reference, warms each route, and alternates allocating and reused-output calls over 101 samples per route. The recorded pre-change sample and three post-change samples use the same fixture and release profile. This measures one projection only; it does not measure loaded weights, repeated layer scheduling, full generation, thermal behavior, or task quality.

Command:

```sh
cargo test --offline -p sage-inference-math --release --lib qwen_hidden_q4_projection_latency_measurement -- --ignored --nocapture --test-threads=1
```

## Results

Values are p50/p95 in microseconds for the reusable CPU output buffer.

| Run | Single-worker SIMD | Row-parallel SIMD |
|---:|---:|---:|
| Baseline | 612 / 1,003 | — |
| 1 | — | 326 / 597 |
| 2 | — | 280 / 391 |
| 3 | — | 306 / 813 |
| Median after change | 612 / 1,003 | 306 / 597 |

The post-change run medians show 2.0× lower p50 and 1.68× lower p95 latency for this matrix. One run had a noisy p95, so the run-level median is used. The optional Metal measurements were 2,790–2,861 μs p50, so the CPU kernel is the faster measured route on this host and fixture. Sage leaves Metal opt-in for evaluation.

The large full-row and selected-row parity tests compare results to an independent f64 scalar reference and include cross-row quantization groups and selected-row reordering. The Mac run verifies AArch64 NEON parallel execution. These results qualify only this kernel path; the pinned checkpoint has not completed numerical parity or full-model latency evaluation, and the 8 GiB host does not meet the 16 GiB hardware acceptance gate.
