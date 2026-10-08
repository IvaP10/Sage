# Head-major KV layout measurement, 8 October 2026

## Method

Host: Apple M3, arm64, 8 GB unified memory. This release-only microbenchmark uses the pinned Qwen3.5-4B full-attention geometry: 16 query heads, four KV heads, and 256 elements per head. It fills Sage's binary16 KV banks, then constructs a position-interleaved copy from the exact same cached bits. Both layouts run the same grouped AArch64 NEON Q·K and weighted-value kernels over all four KV heads with identical queries and normalized weights. The test warms both paths, alternates their order, and collects 101 samples in each of five independent release invocations. Score parity covers all 16 query heads, and output parity covers every output element. It measures KV layout and kernels only; it excludes softmax, decoder layers, model weights, and generation.

Command:

```sh
cargo test -p sage-core --features qwen35-evaluation --release --lib --locked --offline grouped_query_kv_layout_latency_measurement -- --ignored --nocapture --test-threads=1
```

## Results

| Context | Position-interleaved p50 / p95 | Head-major p50 / p95 | Maximum score error | Maximum output error |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 3,224 / 4,842 μs | 1,224 / 1,586 μs | 0 | 0 |
| 8,192 | 20,269 / 21,753 μs | 5,236 / 5,630 μs | 0 | 0 |

Values are medians of the five run-level p50 and p95 measurements. Head-major storage reduced the 8K synthetic kernel p50 and p95 by 74%; at 2K, p50 improved by 62%. Each KV head now occupies one contiguous binary16 bank, which avoids striding over the other heads while reading a long context. The storage format and byte count remain unchanged from the binary16 cache. The test records each query head's scores in separate output ranges, so both layouts write and compare the same full score set. Absolute latency varied between invocations; the five-run medians reduce sensitivity to that variation.

These are kernel-only measurements on an 8 GB development Mac. They do not establish full-model generation latency, checkpoint parity, task quality, or the plan's 16 GB hardware acceptance gate.
