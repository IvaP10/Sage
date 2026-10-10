# Historical persistent Q4 output-head worker experiment

Date: 8 October 2026  
Host: Apple M3 (Mac15,3), 8 GiB RAM  
Compiler: Rust 1.98.0

## Measurement

The synthetic fixture uses the pinned Qwen3.5 vocabulary row count (151,936), hidden width (2,560), and grouped-Q4 weights (group size 128). Three release processes each took 51 alternating samples per route:

- **Dense:** project into a reusable f32 vocabulary buffer, then scan for the greedy winner.
- **Scoped:** fused Q4 argmax, creating and joining bounded OS threads for each projection.
- **Pooled:** the same fused argmax through Sage's bounded persistent workers, sharing immutable Arc-backed weights and a zeroizing activation copy.

| Run | Dense p50 / p95 (µs) | Scoped p50 / p95 (µs) | Pooled p50 / p95 (µs) | First pooled call (ms) |
|---|---:|---:|---:|---:|
| 1 | 7,727 / 10,535 | 7,636 / 9,343 | 7,550 / 9,206 | 7.350 |
| 2 | 7,700 / 9,980 | 7,570 / 9,365 | 7,522 / 9,420 | 10.469 |
| 3 | 7,710 / 9,587 | 7,569 / 9,422 | 7,545 / 9,483 | 9.370 |

The median run-level p50 was 7,710 µs dense, 7,570 µs scoped, and 7,545 µs pooled. The median of paired per-run p50 speedup ratios was 0.64% over the existing scoped fused route and 2.34% over the dense route. The pooled p95 median was 9,420 µs versus 9,365 µs scoped; p95 improved in one run and regressed in two. The first pooled call, which includes worker creation and projection, took 7.350–10.469 ms. This is a modest warm-path gain and not a tail-latency improvement; the main architectural effect is removing per-token OS-thread creation from the long-lived Q4 output head.

The worker count is bounded by detected CPU parallelism and Sage's eight-worker ceiling. The queue is bounded to twice that count. Each call waits for all row ranges before returning; activation copies are zeroized when the last worker releases them. CPU Q4 model clones retain their previous deep-copy semantics. Metal-backed matrices do not use the pooled CPU path.

Historical reproduction command (the persistent pool route has since been removed):

```sh
cargo test --offline --locked --release -p sage-kernels --lib tests::qwen_output_head_fused_argmax_latency_measurement -- --ignored --exact --nocapture --test-threads=1
```

This is a synthetic output-head kernel measurement. It is not backed by the pinned checkpoint and does not establish end-to-end generation speed, model quality, worker integration, memory pressure, or 16 GiB hardware acceptance.


## Pinned Rust 1.89 reevaluation and retirement

The output-head pool was rerun after removing its activation copy. The test compared dense projection plus scan, scoped fused argmax, and pooled fused argmax over the same Qwen-shaped fixture. These three reruns all showed the pool slower than scoped fused argmax:

| Run | Dense p50 / p95 (μs) | Scoped p50 / p95 (μs) | Pooled p50 / p95 (μs) |
|---|---:|---:|---:|
| 1 | 9,371 / 28,457 | 9,107 / 29,118 | 9,453 / 32,763 |
| 2 | 14,770 / 29,138 | 13,579 / 24,732 | 14,412 / 26,305 |
| 3 | 18,086 / 43,320 | 18,993 / 34,856 | 19,732 / 44,729 |

The median run-level p50 was 14,412 μs pooled versus 13,579 μs scoped (6.1% slower); median p95 was 32,763 μs versus 29,118 μs (12.5% slower). One pooled first-call startup measured 68.357 ms. Because the result regressed across all runs, Sage removed the output-head pool route, its worker message type, and its pool-specific concurrency test. Fused output-head argmax remains active through scoped workers. The independent layer-projection pool remains active and has its own measurement record.

The figures above are historical measurements from the now-removed pooled implementation; the command above therefore no longer reproduces that three-route comparison. The current source retains a release-only dense-versus-scoped-fused benchmark in the same test module. None of these synthetic measurements establish full-model generation speed, checkpoint parity, model quality, or 16 GiB acceptance.
