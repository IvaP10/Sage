# Metal Q4 shared-storage row parity and CPU/GPU projection comparison

Date: 9 October 2026  
Host: Apple M3, macOS 27.0.1, 8 GiB RAM  
Build: optimized Rust release profile for the projection measurement

## Correctness finding

The exact Metal parity test previously reached the embedding-row check and found that `row_into` returned incorrect values after Q4 weights moved to shared Metal storage. Full projection and selected-row projection checks in the same test had already passed. The failure used a 7×31 matrix with group size 16, so row 5 begins inside a quantization group. Metal storage preserved every packed byte and every scale. The defect was that the row reader rebased the group index but then used that relative index against the full scale buffer. It now reads the absolute group offset. The parity test compares the transferred packed bytes and all scales, and verifies the Metal-backed embedding row against the CPU reference.

The exact odd-shape parity test and a concurrent independent-projection parity test both pass on the M3 GPU:

```sh
cargo test --offline -p sage-inference-math --lib tests::metal_q4_projection_matches_the_scalar_reference_and_reuses_shared_weights -- --ignored --exact --nocapture
cargo test --offline -p sage-inference-math --lib tests::concurrent_metal_q4_projections_match_independent_scalar_results -- --ignored --exact --nocapture
```

## Qwen-sized projection measurements

The release test ran 101 alternating samples per path on a synthetic 2,560×2,560 grouped-Q4 matrix. Values below are the three separate process outputs in microseconds; each cell is p50/p95 within that process.

| Path | Run 1 | Run 2 | Run 3 | Median of run p50/p95 |
| --- | ---: | ---: | ---: | ---: |
| CPU, caller-reused output | 169/196 | 172/234 | 168/196 | 169/196 |
| Metal, retained workspace and caller-owned output | 2576/3920 | 2433/3993 | 2437/4098 | 2437/3993 |

The retained Metal route was about 14.4× slower at median p50 and 20.4× slower at median p95 than the CPU caller-reused path on this fixture. This supports keeping Metal evaluation-only on the measured M3 workload. These figures do not generalize to other devices or operators and do not include model loading, full generation, or task quality. The Metal bridge still copies activation input and output through host-visible buffers.

Reproduction command:

```sh
cargo test --offline --release -p sage-inference-math --lib tests::qwen_hidden_q4_projection_latency_measurement -- --ignored --exact --nocapture
```
