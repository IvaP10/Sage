# Streamed grouped-Q4 import

Host: Apple M3, 8 GiB unified memory  
Build: Rust release test binary, Sage AArch64 NEON quantizer  
Input: 16,777,216 deterministic finite f32 values (64 MiB), group size 128, streamed in 1,048,576-element chunks  
Command:

```sh
cargo test --offline --release -p sage-inference-math --lib tests::streamed_q4_import_latency_measurement -- --ignored --exact --nocapture --test-threads=1
```

The benchmark alternates the former scalar per-element streaming importer against the optimized builder, with 11 samples per path in each invocation. The scalar path preserves the previous implementation's full-chunk finite check, per-value group buffering, scalar maximum/division/rounding, and nibble packing. Before timing, each invocation compares the optimized scale and packed-value arrays byte-for-byte with that scalar reference.

| Run | Scalar p50 (ms) | Scalar p95 (ms) | Optimized p50 (ms) | Optimized p95 (ms) | p50 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29.839 | 30.042 | 14.326 | 14.435 | 2.083× |
| 2 | 31.351 | 32.929 | 15.014 | 15.271 | 2.088× |
| 3 | 31.483 | 94.715 | 15.071 | 36.056 | 2.089× |
| 4 | 33.113 | 51.998 | 14.847 | 15.977 | 2.230× |
| 5 | 30.493 | 31.433 | 14.527 | 15.548 | 2.099× |

Across-run medians are 31.351/32.929 ms scalar p50/p95 and 14.847/15.548 ms optimized p50/p95, approximately 2.11× lower median p50 and 2.12× lower median p95. The optimized p50 corresponds to about 1.13 billion imported elements per second for this fixture. Per-run tail outliers are retained in the table rather than hidden.

The fast route forms full groups directly from validated input chunks, eliminating per-weight group-buffer copies and per-value Vec pushes. One reusable signed-group scratch holds quantized values; AArch64 NEON reduces absolute maxima, divides by the group scale, and rounds ties away from zero, then paired nibbles are packed with a simple scalar loop. Partial groups, nibble carries, underflowing subnormal scales, and non-AArch64 builds retain explicit scalar handling.

This is a synthetic in-memory quantization/import comparison. It excludes safetensors reads, BF16 conversion, package hashing, real model weights, allocator pressure from a complete checkpoint, worker startup, generation, model quality, and the 16 GiB acceptance test. x86-64 macOS and Windows test targets cross-compile, but the scalar route has not been run on those hardware targets as part of this measurement.
