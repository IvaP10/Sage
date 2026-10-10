# Four-lane peer histogram measurement

## Question

Does a four-lane histogram counter improve the closed byte-histogram worker while preserving its exact output and bounded cancellation checks?

## Method

- Baseline: the previous single `u64[256]` counter loop, including its per-element `index % 16_384 == 0` branch and cancellation callback every 16 KiB.
- Candidate: four independent `u32[256]` counters updated from each four-byte group, aggregated to the same `u64[256]` result. Cancellation is checked once per 16 KiB block. The 16 MiB input cap bounds each lane well below `u32::MAX`.
- Each workload is 8 MiB. `skewed` contains only `0xa5`; `mixed` is generated deterministically with xorshift32. Before timing, the benchmark asserts full encoded output equality.
- Each process runs seven paired samples with four calls per sample, alternating which implementation runs first. Reported values are the median per-call time across those seven samples. Five release-test processes were run sequentially under the repository-pinned Rust 1.89 toolchain (`rustup run 1.89.0 cargo test ...`).
- Timed code includes counting, bounded output allocation/encoding and zeroizing output destruction. Input generation and setup are excluded.

## Results

Apple M3, 8 GiB RAM, macOS arm64, Rust 1.89.0 release profile:

| Workload | Scalar p50 | Four-lane p50 | Speedup |
| --- | ---: | ---: | ---: |
| Repeated byte (`0xa5`) | 9.613 ms | 1.698 ms | 5.663× |
| Deterministic mixed bytes | 2.393 ms | 1.745 ms | 1.371× |

Values are medians of the five process-level p50 results. Exact process outputs are retained in the task tool history; the ignored benchmark is `worker::tests::four_lane_histogram_benchmark` in `crates/sage-peer/src/worker.rs`.

A five-process Homebrew Rust 1.98.0 cross-check measured 2.449× on repeated bytes and 1.338× on mixed bytes. It is secondary evidence; the pinned Rust 1.89 results above govern this repository's optimization decision.

## Limits

This is a single-device synthetic microbenchmark. It does not establish distributed throughput, energy use, other CPU architectures, or end-to-end peer offload performance. The Mandelbrot loop restructuring only removes its per-pixel checkpoint-condition branch; it has not been separately benchmarked. The controlled cancellation callback remains unconnected to Core's Stop event.
