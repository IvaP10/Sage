# Reusable grouped-query attention workspace

Date: 2026-10-07  
Hardware: Apple M3  
Command: `cargo test -p sage-core --features qwen35-evaluation --lib --release grouped_query_attention_workspace_latency_measurement -- --ignored --nocapture --test-threads=1`

The release microbenchmark used a synthetic 2,048-position KV cache, 16 query heads, four KV heads, and 64 values per head. It compared allocating and zeroizing a fresh score/output workspace on each call with clearing and reusing one preallocated workspace. Both paths executed the same attention kernels; the test warmed each path ten times, alternated path order, and collected 101 samples.

```text
qwen-grouped-query-workspace context=2048 query_heads=16 head_dim=64 samples=101 fresh_p50_us=608 fresh_p95_us=760 reused_p50_us=608 reused_p95_us=923
```

This run shows no measurable median latency change; the p95 is noisy and higher for reuse, so no speedup claim is warranted. The code does remove two allocator calls per full-attention layer per token. At the configured 8K context, those buffers are up to 80 KiB per layer, or 640 KiB of allocation requests across eight layers for each generated token. The buffers are included in Sage's memory admission estimate and zeroized after use. This remains a synthetic attention microbenchmark, not an end-to-end decoder or real-checkpoint result.
