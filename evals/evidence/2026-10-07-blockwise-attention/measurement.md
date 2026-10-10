# Blockwise attention measurement, 7 October 2026

Command:

```sh
cargo test --offline -p sage-inference-math --release --lib grouped_query_attention_blockwise_latency_measurement -- --ignored --nocapture --test-threads=1
```

Host: Apple M3, macOS. The ignored release-only microbenchmark alternates 101 samples per path over deterministic synthetic binary16 key/value caches, 16 query heads, and 64-element head dimensions. It compares the fixed-2,048-score-block implementation against a full-score reference using identical cached values and validates output error before reporting latency.

```text
qwen-blockwise-attention context=2048 query_heads=16 head_dim=64 score_block=2048 samples=101 full_p50_us=710 full_p95_us=731 blockwise_p50_us=707 blockwise_p95_us=735 max_output_error=0.00000020
qwen-blockwise-attention context=8192 query_heads=16 head_dim=64 score_block=2048 samples=101 full_p50_us=2838 full_p95_us=2970 blockwise_p50_us=2846 blockwise_p95_us=3023 max_output_error=0.00000017
```

The 8K comparison shows near-parity on this CPU kernel fixture while limiting the score vector to 2,048 positions. The memory-envelope test estimates 270,336 bytes (264 KiB) across eight attention layers for score blocks, output vectors, and one head-sized temporary, down 376 KiB from the earlier 640 KiB context-sized score workspace. This is an estimate of reserved scratch, not measured process memory.

This evidence covers synthetic kernel outputs only. The Qwen checkpoint has not been admitted after this change; these measurements do not establish checkpoint parity, task quality, model-generation latency, 16 GiB hardware acceptance, or product readiness.

## Block-size tuning check

A separate release run changed the compile-time score block to 4,096 positions and repeated the same 101-sample alternating full-score/blockwise benchmark. It reported:

```text
qwen-blockwise-attention context=2048 query_heads=16 head_dim=64 score_block=4096 samples=101 full_p50_us=722 full_p95_us=844 blockwise_p50_us=724 blockwise_p95_us=899 max_output_error=0.00000020
qwen-blockwise-attention context=8192 query_heads=16 head_dim=64 score_block=4096 samples=101 full_p50_us=2994 full_p95_us=3301 blockwise_p50_us=3014 blockwise_p95_us=3441 max_output_error=0.00000017
```

The 4,096-position variant used 384 KiB of eight-layer scratch, versus 264 KiB for 2,048, and did not improve latency in that run. The 2,048 setting remains selected. These were separate processes rather than a single cross-configuration randomized trial, so scheduler and frequency variation limit the strength of the performance comparison; the lower scratch use is deterministic from the checked envelope.
