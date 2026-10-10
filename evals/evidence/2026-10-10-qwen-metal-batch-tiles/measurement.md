# Qwen Q4 Metal batch-tile comparison

Date: 10 October 2026. Host: Apple Mac15,3 (arm64, 8 GiB physical memory). Rust: pinned 1.89.0. All three runs used the optimized release profile and the same Metal-capable Apple GPU.

## Question and method

The first-party Metal kernel reuses each decoded Q4 weight across a fixed number of row-major activation inputs in one threadgroup. The production batch workspace used tile size 4. This comparison tests tile sizes 1, 2 and 4 to see whether less per-threadgroup accumulator pressure improves the same work.

The fixture is deterministic synthetic Q4 data shaped like a Qwen MLP projection (`10,240×2,560`, group size 128), not checkpoint tensors. Batch sizes are 1, 4, 16, 64 and 256. Each tile gets a workspace allocated before timing; timed calls include command submission, GPU execution, result readback and completion wait, and reuse the same weights, activation buffers and output buffers. Routes alternate ascending and descending tile order across 31 samples. Every route's full batch output is compared against the first-party CPU result using `1e-3 + 1e-3 × abs(reference)` tolerance before timing. Each listed value is the median of three independent process-level p50/p95 measurements.

Command:

```sh
rustup run 1.89.0-aarch64-apple-darwin cargo test --offline --locked --release -p sage-inference-math --lib qwen_q4_metal_batch_tile_measurement -- --ignored --nocapture --test-threads=1
```

## Results

| Batch | Tile 1 p50 / p95 | Tile 2 p50 / p95 | Tile 4 p50 / p95 | Tile 2 / tile 4 p50 | Tile 1 / tile 4 p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 886 / 2,840 μs | 862 / 2,607 μs | 891 / 2,456 μs | 0.97× | 0.99× |
| 4 | 2,248 / 3,998 μs | 1,587 / 3,302 μs | 1,246 / 2,915 μs | 1.27× | 1.80× |
| 16 | 10,021 / 11,521 μs | 7,116 / 7,287 μs | 5,494 / 5,888 μs | 1.30× | 1.82× |
| 64 | 41,034 / 41,354 μs | 26,751 / 26,923 μs | 19,185 / 20,636 μs | 1.39× | 2.14× |
| 256 | 159,217 / 161,831 μs | 105,544 / 108,566 μs | 76,412 / 78,101 μs | 1.38× | 2.08× |

Tile size 4 is the measured choice for batches of 4 or more on this device and fixture. Tile sizes have no meaningful p50 difference at batch 1, where CPU remains the faster route. The default remains 4; no device-independent or whole-model claim follows. Raw logs with all 31 samples for each route are [`run-1.txt`](run-1.txt), [`run-2.txt`](run-2.txt), and [`run-3.txt`](run-3.txt). [`validation.txt`](validation.txt) is a follow-up run after adding an explicit rejection check for the unmeasured tile size 3.

## Scope

This result tunes a single Metal Q4 projection boundary. It does not measure weight upload, per-call workspace allocation, real checkpoint activations, complete decoder prefill, token generation, energy, or another GPU. The matrix-level CPU/Metal route results including workspace allocation are recorded in [the group-128 route study](../2026-10-10-qwen-q4-group128-route/measurement.md). Metal stays opt-in pending route policy and complete-model qualification.
