# Qwen zero-centered RMS and scratch reuse

## Setup

- Date: 8 October 2026
- Device: Apple Mac15,3, arm64, 8 GiB RAM
- Toolchain: rustc 1.98.0, cargo 1.98.0
- Command: `cargo test -p sage-kernels --release --locked qwen_zero_centered_rms_latency_measurement -- --ignored --nocapture`
- Each timed sample normalizes 20 independent 256-value heads. The allocating scalar path performs the f64 reference reduction and scaling, checks finite inputs/results, allocates and clears one output vector per head. The reusable path performs the same f64 arithmetic through the first-party AArch64 NEON kernel, uses one caller-owned output buffer, checks finite inputs/results, and clears that buffer between heads.
- Each row below is a fresh test invocation with 101 alternating samples. Values are p50/p95 microseconds for all 20 heads in one sample.

## Results

| Run | Allocating scalar p50 / p95 | Reusable NEON p50 / p95 |
|---|---:|---:|
| 1 | 10.583 / 10.666 μs | 7.375 / 7.416 μs |
| 2 | 10.625 / 20.250 μs | 7.375 / 16.292 μs |
| 3 | 10.625 / 20.708 μs | 7.375 / 8.625 μs |
| Across-run median | 10.625 / 20.250 μs | 7.375 / 8.625 μs |

The across-run median is 30.6% lower at p50 and 57.4% lower at p95. The high tail in the first two runs indicates host scheduling noise; p95 is especially sensitive at this short duration.

## Limits

This measures only the Qwen attention-head normalization kernel on an 8 GiB development Mac. It does not measure full-layer latency, end-to-end token generation, checkpoint parity, task quality, sustained thermals, or the 16 GiB acceptance device. The kernel is checked against an independent f64 formula for dimensions 1, 2, 3, 4, 5, 255, 256, and 257, including scalar tails, invalid-input clearing, and non-finite rejection. Full model admission remains disabled.
