# Q4 CPU and Metal projection route sweep

Date: 10 October 2026. Host: Apple Mac15,3 (arm64, 8 GiB physical memory). Rust: pinned 1.89.0. The tests used the optimized release profile and a Metal-capable Apple GPU.

> Historical pre-specialization baseline. The group-128 shader fast path and corrected matrix-level batch submission were measured afterward; use [`the updated Qwen route measurements`](../2026-10-10-qwen-q4-group128-route/measurement.md) for the current comparison. This record is retained to show the before/after evidence.

## Question

The Qwen candidate loader exposes an opt-in Metal route for grouped-Q4 projections. Before expanding that route, this measurement compares the current CPU and Metal implementations at six decoder-like shapes and across a range of batched projections.

## Method

The single-input test measures shapes `256×2,560`, `1,280×2,560`, `2,560×2,560`, `5,120×2,560`, `10,240×2,560`, and `2,560×10,240`. The batched test measures one `4,096×1,024` matrix at batch sizes 1, 4, 16, 64 and 256. All use deterministic synthetic packed Q4 values, 128-value groups, and synthetic scales. They are shape fixtures, not checkpoint tensors.

Each CPU matrix is cloned and transferred to Metal before timing. Input/output buffers and the Metal workspace are reused. Both routes are warmed twice, then receive 31 alternating calls per shape. Before timing, each output is compared using tolerance `1e-3 + 1e-3 × abs(reference)`. The timings include the Metal dispatch and result access performed by the projection API; they exclude weight upload and workspace creation. The harness records sorted per-sample nanoseconds in [`raw-run.txt`](raw-run.txt).

Command:

```sh
rustup run 1.89.0-aarch64-apple-darwin cargo test --offline --locked --release -p sage-inference-math --lib q4_cpu_metal -- --ignored --nocapture --test-threads=1
```

## Results

Single-input projections:

| Rows × columns | CPU p50 / p95 | Metal p50 / p95 | Metal / CPU p50 |
| --- | ---: | ---: | ---: |
| 256 × 2,560 | 57 / 61 μs | 399 / 586 μs | 7.0× |
| 1,280 × 2,560 | 109 / 148 μs | 1,293 / 2,769 μs | 11.9× |
| 2,560 × 2,560 | 173 / 207 μs | 2,433 / 3,875 μs | 14.1× |
| 5,120 × 2,560 | 303 / 395 μs | 5,926 / 6,402 μs | 19.6× |
| 10,240 × 2,560 | 558 / 592 μs | 11,652 / 12,881 μs | 20.9× |
| 2,560 × 10,240 | 557 / 616 μs | 11,737 / 13,133 μs | 21.1× |

Batched projections (`4,096 × 1,024`):

| Batch | CPU p50 / p95 | Metal p50 / p95 | Metal / CPU p50 |
| ---: | ---: | ---: | ---: |
| 1 | 245 / 292 μs | 1,894 / 3,646 μs | 7.7× |
| 4 | 1,350 / 1,723 μs | 1,892 / 3,423 μs | 1.4× |
| 16 | 3,744 / 3,766 μs | 7,298 / 7,582 μs | 1.9× |
| 64 | 8,265 / 9,406 μs | 26,245 / 27,387 μs | 3.2× |
| 256 | 21,504 / 22,567 μs | 102,590 / 108,953 μs | 4.8× |

## Interpretation and limits

The CPU route had lower median latency in every measured case. These results do not support moving Q4 projections to Metal on this machine for the measured API boundary. They do not prove that the shader arithmetic is intrinsically slower: host/device synchronization and returning each projection result may dominate. Keeping activations on the device or fusing decoder operations could change the comparison.

This is one 8 GiB Apple M3 system, synthetic data, and isolated projection calls. The tests do not measure a real checkpoint, generation quality, whole-decoder prefill/decode, energy, other Apple GPU generations, or the 16 GiB acceptance target. The opt-in Metal route remains an evaluation facility; it is not a production backend recommendation.
