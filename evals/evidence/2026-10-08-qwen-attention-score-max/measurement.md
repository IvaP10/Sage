# Qwen grouped-query attention score reduction, 8 October 2026

## Method

Host: Apple M3, arm64, 8 GiB unified memory. Rust 1.98.0. The release-only benchmark uses Qwen3.5-4B full-attention geometry: 16 query heads, four KV heads, 256 values per head, 8,192 context positions, and Sage's 256-position attention blocks. It compares the previous grouped Q·K kernel followed by separate score scaling and maximum scans with the fused grouped Q·K-scale-maximum kernel. A third route fuses scaling and maximum in one scalar slice pass after the old Q·K kernel. Each path processes the same deterministic binary16 key bits and queries. The benchmark warms each route, rotates route order, records 51 paired samples per process, and checks exact scaled-score and per-block-maximum equality.

Command:

```sh
cargo test --locked --offline -p sage-kernels --release qwen_grouped_qk_scale_max_fusion_latency_measurement -- --ignored --nocapture --test-threads=1
```

## Results

| Run | Previous grouped kernel + separate scans p50 / p95 | Scalar scale/max fusion p50 / p95 | NEON kernel fusion p50 / p95 |
|---:|---:|---:|---:|
| 1 | 2,143 / 2,186 μs | 2,134 / 2,177 μs | 2,014 / 2,045 μs |
| 2 | 2,212 / 2,272 μs | 2,185 / 2,290 μs | 2,040 / 2,166 μs |
| 3 | 2,145 / 2,230 μs | 2,141 / 2,215 μs | 2,021 / 2,092 μs |
| 4 | 2,155 / 2,254 μs | 2,146 / 2,230 μs | 2,028 / 2,102 μs |
| 5 | 2,142 / 2,224 μs | 2,132 / 2,211 μs | 2,017 / 2,094 μs |
| Median of runs | 2,145 / 2,230 μs | 2,141 / 2,215 μs | 2,021 / 2,094 μs |

At this synthetic 8K attention-kernel workload, the fused NEON route lowered median p50 by 5.8% and p95 by 6.1% versus the previous implementation. The scalar one-pass reduction produced less than a 1% improvement, so the product uses the measured kernel fusion. Scaled scores and block maxima matched exactly in all five runs.

This measures grouped Q·K scoring, score scaling, and block-maximum collection. It excludes exponentiation/softmax, weighted-value attention, decoder layers, model weights, full generation, checkpoint parity, and task quality. The 8 GiB development Mac does not satisfy the separate 16 GiB acceptance gate.
