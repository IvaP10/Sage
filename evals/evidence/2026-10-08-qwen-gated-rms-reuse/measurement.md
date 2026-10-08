# Qwen gated-RMS scratch reuse measurement

This microbenchmark compares the prior per-head allocating scalar operation with Sage's caller-owned gated-RMS kernel for one Qwen linear-attention layer's 32 value heads, each 128 values wide. The AArch64 implementation uses NEON for the f64 sum-of-squares reduction and retains scalar f32 SiLU operations. Both implementations use the same f64 accumulation and output operation order; the optimized path reuses one output buffer instead of allocating one vector per head.

Three optimized Rust release runs each used 1,001 alternating samples. Across-run medians were:

| Path | p50 | p95 |
|---|---:|---:|
| Allocating scalar reference | 10 μs | 12 μs |
| Reused AArch64 kernel | 9 μs | 11 μs |

The measured p50 was about 10% lower and p95 about 8% lower. The test validates every output against an independent wide-precision reference and covers lengths around the NEON vector width, malformed inputs, and non-finite results. The benchmark's test name is `qwen_silu_gated_rms_latency_measurement` in `crates/sage-kernels/src/lib.rs`.

These are synthetic, normalization-only numbers on the 8 GiB Apple M3 development Mac. They cover 32 heads, not model projections, recurrent updates, decoder layers, checkpoint generation, or user-visible latency. They do not establish model quality or satisfy the 16 GiB hardware gate. Run with:

```sh
cargo test -p sage-kernels --release qwen_silu_gated_rms_latency_measurement -- --ignored --nocapture
```
