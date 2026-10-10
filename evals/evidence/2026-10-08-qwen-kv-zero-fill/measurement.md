# Qwen KV-cache append without redundant row zero-fill

Date: 8 October 2026  
Host: Apple M3, arm64, 8 GiB RAM  
Compiler: Rust 1.98.0  
Runtime FP16 support: `hw.optional.arm.FEAT_FP16=1`

## Method

The fixture matches the pinned Qwen3.5-4B full-attention cache shape: eight layers, four KV heads per layer, 256 values per head, and separate key/value rows. A token appends 16,384 f32 values across those caches. Each release-test process performs ten warmups and 101 alternating samples of the prevalidated reference and the checked production append path.

Before the change, production append resized each pre-reserved `Vec<u16>` row with zeroes, then overwrote that row with converted key/value data. After the change, Sage reserves the same row capacity and converts directly into `Vec::spare_capacity_mut()`. The safe vector length advances only after every value is initialized. A failed conversion zeros the entire attempted append range before returning. The SIMD conversion, cache geometry, zeroization on cache clear, reserve strategy, and exact binary16 output checks are unchanged.

## Results

| Run | Before append p50 / p95 | After append p50 / p95 |
|---|---:|---:|
| 1 | 2,875 / 3,583 ns | 2,917 / 3,000 ns |
| 2 | 3,041 / 3,083 ns | 2,875 / 2,917 ns |
| 3 | 3,041 / 3,083 ns | 2,875 / 3,000 ns |
| 4 | 3,125 / 3,167 ns | 2,875 / 2,958 ns |
| 5 | 3,167 / 26,542 ns | 2,875 / 2,875 ns |
| Median of runs | 3,041 / 3,167 ns | 2,875 / 2,958 ns |

Across five independent release-test processes, median append p50 fell 5.5% and p95 fell 6.6%. The older-path p95 has a visible scheduling outlier; the per-run p50s show the more stable comparison. The kernel tests verify exact output bits, preservation of existing vector prefixes, unchanged vector length on failure, zeroed partial writes, and successful reuse after rejection.

Reproduce with:

```sh
cargo test -p sage-inference-math --release tests::qwen_kv_cache_validation_fusion_latency_measurement -- --ignored --exact --nocapture --test-threads=1
```

This measures only KV append over a synthetic pinned shape. It excludes attention, full decoder layers, prefill, generation, checkpoint-backed task quality, peak model memory, and the separate 16 GiB acceptance gate. It does not imply a matching end-to-end generation speedup.
