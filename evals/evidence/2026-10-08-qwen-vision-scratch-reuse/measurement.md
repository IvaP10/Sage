# Reusable Qwen vision-stack activation workspace

Date: 8 October 2026  
Host: Mac15,3, Apple M3, arm64, 8 GiB RAM, macOS 27.0.1

## Measurement

The release fixture contains 24 synthetic Q4 vision blocks, 128 tokens, hidden width 32, four heads, and intermediate width 64. Its deterministic nonzero projection weights are Sage-quantized Q4 values; it is not the pinned model's actual vision weights or geometry. Before timing, the full-stack output is compared with the per-block allocating route and every value must agree within `1e-6`.

Each invocation performs three warmups and 101 alternating samples per route. The results below are the median of each percentile across three separate release-test processes.

| Run | Per-block scratch p50 / p95 | Shared-stack scratch p50 / p95 |
|---|---:|---:|
| 1 | 23,922 / 43,712 μs | 22,709 / 42,713 μs |
| 2 | 23,810 / 24,177 μs | 22,611 / 22,974 μs |
| 3 | 25,203 / 26,708 μs | 22,600 / 25,799 μs |
| Median of runs | 23,922 / 26,708 μs | 22,611 / 25,799 μs |

The shared workspace improved the median run-level p50 by 5.5% and p95 by 3.4% on this synthetic stack. It deterministically avoids 138 block-activation `Vec` allocations per 24-layer call: six per-block vectors across 24 layers are replaced by five reusable scratch vectors and two ping-pong state buffers. Scratch is zeroized between layers; both state buffers use zeroizing storage.

Reproduce with:

```sh
cargo test -p sage-qwen35-runtime --release --locked qwen35_vision::tests::reusable_vision_stack_scratch_latency_measurement -- --ignored --exact --nocapture --test-threads=1
```

This is a small synthetic CPU measurement. It does not establish performance for the real Qwen3.5-4B vision geometry, full image preprocessing/encoding, real-checkpoint parity, image task quality, end-to-end model latency, or 16 GiB hardware acceptance.
