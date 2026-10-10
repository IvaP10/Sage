# Qwen KV-cache fused validation and FP16 conversion

Date: 8 October 2026  
Host: Apple M3, arm64, 8 GiB RAM  
Compiler: Rust 1.98.0  
Runtime FP16 support: `hw.optional.arm.FEAT_FP16=1`

## Measurement

The fixture matches the pinned Qwen3.5-4B full-attention cache shape: eight full-attention layers, four KV heads per layer, 256 values per head, and separate key/value rows. One generated token appends 16,384 f32 values across those caches. The baseline reproduces the former append path: a full finite/range scan followed by the ordinary AArch64 converter, which scans each four-value group before issuing `FCVTN`. The new path fuses range checks and conversion from each loaded NEON vector; subnormals and tails retain the exact scalar conversion. Both routes use identical cache shape, reserve, zeroization, and exact binary16 output checks. Each process performs ten warmups and 101 alternating measured samples.

| Run | Former path p50 / p95 | Fused path p50 / p95 |
|---|---:|---:|
| 1 | 14,791 / 14,959 ns | 3,125 / 3,209 ns |
| 2 | 15,041 / 15,167 ns | 3,125 / 3,167 ns |
| 3 | 15,000 / 15,125 ns | 3,125 / 3,125 ns |
| 4 | 13,833 / 17,334 ns | 2,875 / 3,625 ns |
| 5 | 13,583 / 15,167 ns | 2,833 / 3,166 ns |
| Median of runs | 14,791 / 15,167 ns | 3,125 / 3,167 ns |

Across five independent release-test processes, median append p50 and p95 fell by about 79% (4.73× and 4.79× throughput). Cache contents were bit-identical. The benchmark ran on this 8 GiB Apple M3 with native FP16 support; it does not satisfy the separate 16 GiB acceptance gate.

Reproduce with:

```sh
cargo test -p sage-inference-math --release tests::qwen_kv_cache_validation_fusion_latency_measurement -- --ignored --exact --nocapture --test-threads=1
```

This isolates validated KV append across the pinned cache geometry. It excludes attention, full decoder layers, prompt prefill, complete generation, real-checkpoint quality, and peak model memory. The isolated append improvement does not imply an equal end-to-end generation speedup.
