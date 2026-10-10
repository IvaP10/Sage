# Qwen MRoPE angle reuse microbenchmark

Date: 2026-10-08  
Host: Apple M3, Mac15,3, 8 GiB RAM  
Toolchain: rustc 1.98.0, optimized Rust release test binary

## Command

```sh
cargo test --offline -p sage-inference-math --release --lib qwen_mrope_prepared_angle_latency_measurement -- --ignored --nocapture
```

The benchmark uses the pinned full-attention head shape: 16 query heads, four key heads, 256 values per head, and 64 rotary dimensions. The MRoPE triplet is `[123, 47, 89]`. Each trial has 101 alternating baseline/optimized samples.

The baseline recomputes `powf` and `sin_cos` for all 32 rotary pairs on every head. The optimized path precomputes frequency denominators once, prepares one sine/cosine table per position, and reuses it for all 20 heads. Both paths reuse preallocated head vectors, reset them from the same inputs inside the timed region, and allocate no scratch during the measurement.

## Results

| Run | Repeated p50 / p95 | Reused-angle p50 / p95 | p50 reduction | p95 reduction |
|---|---:|---:|---:|---:|
| 1 | 7 / 8 μs | 2 / 2 μs | 64.4% | 64.0% |
| 2 | 7 / 9 μs | 2 / 3 μs | 63.6% | 64.7% |
| 3 | 8 / 9 μs | 2 / 3 μs | 64.4% | 64.1% |

The median of the three run-level reductions was 64.4% at p50 and 64.1% at p95. Latencies are rounded to whole microseconds while reductions use the underlying nanosecond measurements. A separate exact-output test confirms bit-identical f32 results for the direct and prepared formulas across 20 heads and all three MRoPE axes.

## Limits

This measures one CPU RoPE kernel fixture. It does not include the rest of the transformer, model loading, Metal, checkpoint parity, token throughput, task quality, or 16 GiB device qualification. These numbers must not be presented as full-model latency.
