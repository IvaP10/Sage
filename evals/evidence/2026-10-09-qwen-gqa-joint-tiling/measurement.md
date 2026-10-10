# Qwen grouped-query attention tiling sweep

## Result

The pinned Qwen3.5-4B full-attention geometry is **16 query heads, four KV heads, and 256 values per head**. An earlier tile-width note incorrectly attributed a 20-head synthetic fixture to this model; the runtime selector and this record now use the validated 16/4/256 geometry.

A joint sweep on Apple M3 compared query-tile and score-block sizes. The measured Apple Silicon profile is now one query head per tile and a 4,096-position score block. At 8K context it reduced median p50/p95 by 8.2%/7.7% against the four-query, 2,048-position baseline. The 8K score block had similar latency but twice the score workspace; the 4K block had the lowest median p50 and p95 among the measured one-query profiles across four independent processes. At 2K, score blocks are capped by context and the same profile reduced median p50/p95 by 9.6%/7.2% against baseline. Other geometries retain the generic four-query, 2,048-position profile.

Across four independent release-test processes, median-of-process p50/p95 latency was:

| Context | Query tile | Score block | p50 (µs) | p95 (µs) | Across-process range, p50 / p95 (µs) |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 1 | 2,048 | 1,163.5 | 1,179.5 | 1,125–1,167 / 1,167–1,183 |
| 2,048 | 1 | 4,096 | 1,134.5 | 1,176.5 | 1,086–1,166 / 1,152–1,184 |
| 2,048 | 1 | 8,192 | 1,152 | 1,176 | 1,112–1,167 / 1,162–1,187 |
| 2,048 | 3 | 8,192 | 1,159 | 1,171.5 | 1,155–1,161 / 1,161–1,178 |
| 2,048 | 4 | 2,048 | 1,254.5 | 1,268 | 1,246–1,262 / 1,264–1,270 |
| 8,192 | 1 | 2,048 | 4,606.5 | 4,696.5 | 4,590–4,674 / 4,672–4,712 |
| 8,192 | 1 | 4,096 | 4,589 | 4,668.5 | 4,568–4,655 / 4,653–4,694 |
| 8,192 | 1 | 8,192 | 4,596.5 | 4,674.5 | 4,565–4,646 / 4,647–4,687 |
| 8,192 | 3 | 8,192 | 4,627.5 | 4,696 | 4,612–4,658 / 4,692–4,787 |
| 8,192 | 4 | 2,048 | 4,998 | 5,057.5 | 4,972–5,047 / 5,014–5,129 |

Every profile stayed within `5e-5` maximum output difference from the baseline. The 4K block halves the score storage versus the 2K/four-query baseline for the eight full-attention layers while retaining the measured latency improvement. These are synthetic attention-kernel results; counters, energy, checkpoint parity, generation quality, and full-model latency were not measured.

## Method

- Host: Apple Mac15,3, arm64, 8 GiB RAM, macOS 27.0.1; release test compiled with Rust 1.89.0.
- Synthetic binary16 KV cache and deterministic query vectors with the validated 16/4/256 full-attention geometry, at 2K and 8K contexts.
- Each process warms each route, then measures 101 alternating samples for five joint profiles. Order reverses every sample to reduce ordering bias. The table reports the median and full range across four separate processes.
- The comparison profiles were query tile / score block `(1, 2,048)`, `(1, 4,096)`, `(1, 8,192)`, `(3, 8,192)`, and `(4, 2,048)`.
- Reproduce with:

```sh
rustup run 1.89.0 cargo test -p sage-inference-math --release grouped_query_attention_qwen_joint_tiling_latency_measurement -- --ignored --nocapture
```

This is an attention-only synthetic benchmark. It does not load Qwen weights or measure full-model generation, task quality, power, or the required 16 GiB acceptance system.
