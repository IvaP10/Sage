# Q4 per-group scale-hoisting measurement

Date: 7 October 2026

## Environment and fixture

- Hardware: Apple M3, 8 GiB unified memory.
- Architecture: arm64 / AArch64 NEON.
- Rust: 1.98.0.
- Matrix: 2,560 output rows × 2,560 input columns, Sage signed grouped-Q4, 128 values per scale.
- Measurement: release-mode scalar f64 reference, NEON path, and optional Metal path; 101 alternating samples per path, warm-up included, with output parity checked before timing.

The NEON kernel previously multiplied each vector of decoded Q4 values by the group's shared scale before accumulating. It now accumulates decoded signed values and multiplies the reduced group sum by its scale once. This preserves the same group boundaries, four independent vector accumulators, and scalar edge handling.

## Results

| Run | Scalar p50 / p95 | NEON p50 / p95 | Metal p50 / p95 |
|---:|---:|---:|---:|
| 1 | 5.087 / 5.377 ms | 0.570 / 0.642 ms | 5.610 / 7.610 ms |
| 2 | 5.107 / 5.320 ms | 0.566 / 0.653 ms | 5.501 / 7.814 ms |
| 3 | 5.099 / 5.526 ms | 0.567 / 0.626 ms | 5.569 / 7.521 ms |
| Across-run median | 5.099 / 5.377 ms | 0.567 / 0.642 ms | 5.569 / 7.610 ms |

The prior SIMD implementation's recorded 21-sample run measured 0.659 / 0.681 ms p50/p95. Relative to that run, the new three-run medians are about 14% lower at p50 and 6% lower at p95. Because old and new sample counts differ and the runs were not interleaved between code revisions, treat this as an indicative improvement, not a controlled A/B result.

The three release runs used:

```sh
cargo test --release -p sage-core --features qwen35-evaluation --lib --locked --offline inference_cpu::tests::qwen_hidden_q4_projection_latency_measurement -- --ignored --exact --nocapture --test-threads=1
```

The benchmark checks NEON and Metal outputs against the independent f64 Q4 reference within `1e-3` absolute plus relative tolerance. The separate odd-shape/cross-group kernel test passes its tighter `2e-5` tolerance. The current Metal single-projection path measured substantially slower than NEON on this fixture; it remains an opt-in evaluation backend, with batching/fusion required before treating it as a fast path. These synthetic projection measurements do not establish checkpoint parity, quantization quality, full-model latency, peak resident memory, or 16 GiB hardware acceptance.
