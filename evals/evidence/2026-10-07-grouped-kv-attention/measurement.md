# Grouped KV attention measurement, 7 October 2026

## Method

Host: Apple M3, arm64, 8 GB unified memory. The release-only microbenchmark alternates 101 samples per path over synthetic binary16 KV caches. Its fixture uses the pinned Qwen3.5-4B full-attention geometry: 16 query heads, four KV heads, and 256 values per head. It compares the separate per-query-head full-score reference with Sage's grouped, blockwise online-softmax path. The latter loads each selected KV row once for a tile of four query heads. Output error is checked before latency is reported. Two independent invocations are recorded because host frequency changed the absolute times.

Command:

```sh
cargo test -p sage-core --features qwen35-evaluation --release --lib --locked --offline grouped_query_attention_blockwise_latency_measurement -- --ignored --nocapture --test-threads=1
```

## Results

| Context | Run | Separate p50 / p95 | Grouped p50 / p95 | Maximum output difference |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1 | 4,806 / 7,607 μs | 3,541 / 5,554 μs | 2.6e-7 |
| 2,048 | 2 | 3,097 / 3,569 μs | 1,925 / 2,607 μs | 2.6e-7 |
| 8,192 | 1 | 42,114 / 83,965 μs | 16,916 / 36,489 μs | 1.8e-7 |
| 8,192 | 2 | 36,169 / 37,911 μs | 14,534 / 15,227 μs | 1.8e-7 |

Grouped attention reduced this synthetic kernel fixture's median latency by 26–38% at 2K and about 60% at 8K across the two invocations. It reduced p95 by about 27% at 2K and 57–60% at 8K. The score scratch remains fixed at four query heads times 2,048 positions. For the pinned geometry, all eight full-attention layers use 672 KiB of bounded attention scratch, compared with the prior 264 KiB estimate.

These are kernel-only measurements on an 8 GB development Mac. They do not establish full-model generation latency, checkpoint parity, task quality, end-to-end performance, or the plan's 16 GB hardware acceptance gate. Sage keeps the scalar fallback for non-AArch64 systems and unsupported generic shapes.
