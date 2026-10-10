# Qwen KV-cache binary16 append conversion

Date: 8 October 2026  
Host: Mac15,3, Apple M3, arm64, 8 GiB RAM, macOS 27.0.1  
Compiler: Rust 1.98.0  
Runtime FP16 support: `hw.optional.arm.FEAT_FP16=1`

## Measurement

The fixture matches the pinned Qwen3.5-4B full-attention cache shape: eight full-attention layers, four KV heads per layer, 256 values per head, and separate key/value rows. One generated token appends 16,384 f32 values across those caches. The benchmark reserves each cache row before timing, compares the former per-head scalar conversion loop with `KvCache::append`, gives both routes the same context, shape, finiteness, and range checks, alternates the routes, performs ten warmups and 101 measured samples per release-test process, and checks exact binary16 output equality before timing. Cache clearing and its zeroization are outside the timed interval.

| Run | Scalar p50 / p95 | Native-path p50 / p95 |
|---|---:|---:|
| 1 | 65,416 / 90,375 ns | 30,083 / 37,583 ns |
| 2 | 31,792 / 34,917 ns | 14,542 / 15,583 ns |
| 3 | 31,041 / 31,833 ns | 14,250 / 14,667 ns |
| 4 | 31,709 / 31,875 ns | 14,541 / 14,667 ns |
| 5 | 33,708 / 33,875 ns | 15,375 / 15,542 ns |
| Median of runs | 31,792 / 33,875 ns | 14,542 / 15,542 ns |

Across five runs, the median append p50 and p95 improved by about 54% (2.19× and 2.18× throughput, respectively). The benchmark ran on an AArch64 Mac reporting native FP16 support. `sage-kernels` dispatches normal finite values through four-lane `FCVTN`; values near binary16 underflow, non-finite values, and values outside the finite binary16 range stay on Sage's scalar reference conversion. `KvCache::append` still validates inputs, reserves capacity, writes directly to preallocated per-head banks, and zeroizes any partial row if conversion returns an error.

This is the historical measurement from the earlier scalar-versus-native conversion benchmark. That benchmark was replaced when append validation and conversion were fused. For a reproducible comparison against the immediately previous validated SIMD path, use the [current validation-fusion measurement](../2026-10-08-qwen-kv-append-validation/measurement.md).

This isolates KV append conversion over the Qwen cache shape. It does not measure attention, full decoder layers, prompt prefill, complete generation, real-checkpoint quality, peak model memory, or the 16 GiB hardware gate; the append speedup does not establish the same speedup for end-to-end generation.
