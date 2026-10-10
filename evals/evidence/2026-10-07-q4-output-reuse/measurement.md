# Qwen Q4 projection output reuse

Date: 7 October 2026  
Host: Apple M3, 8 GiB unified memory  
Build: Rust release test binary, local first-party CPU NEON kernel  
Command: `cargo test --offline -p sage-inference-math --lib qwen_hidden_q4_projection_latency_measurement --release -- --ignored --nocapture`

The fixture uses a 2,560 × 2,560 grouped-Q4 matrix with 128-element quantization groups and 101 samples per path. Allocating `project` and caller-owned `project_into` were alternated within each run. Every output was checked against the same f64 scalar reference before timing.

| Run | CPU allocating p50 / p95 | CPU reused p50 / p95 |
|---|---:|---:|
| 1 | 584 / 803 μs | 582 / 735 μs |
| 2 | 592 / 1,139 μs | 593 / 1,219 μs |
| 3 | 587 / 1,088 μs | 588 / 1,053 μs |
| Across-run median | 587 / 1,088 μs | 588 / 1,053 μs |

Output reuse removed the result-vector allocation but did not improve median projection latency in this fixture. The across-run p95 difference is small and inconsistent between runs, so it is inconclusive. The purpose of the API is to remove repeated allocator traffic when a decoder can retain its output buffers; this microbenchmark does not measure a full MLP or model generation.

At the 32-layer Qwen profile, the CPU MLP now reuses its gate, up, and feed-forward projection outputs, structurally removing three result-vector allocations per layer per token (96 total). The zeroizing scratch adds 2,686,976 bytes, which is included in the candidate software memory estimate. Metal's evaluation bridge still returns temporary host vectors and is not covered by that allocation count.

The run uses synthetic weights and does not qualify checkpoint parity, model task quality, end-to-end generation speed, the 16 GiB acceptance tier, or product inference. The actual checkpoint has not been imported into the candidate loader.
