# First-party Qwen vision-attention kernels

## Result

Three alternating release runs on Apple Mac15,3 (arm64, 8 GiB RAM) compared the former scalar vision-attention arithmetic with Sage's f64 NEON/scalar kernels. Each run used 101 alternating samples (51 scalar and 50 kernel) on a deterministic synthetic 128-token QKV fixture with the Qwen vision geometry: 1,024 hidden values, 16 heads, and 64 values per head.

| Run | Scalar p50 | Scalar p95 | Kernel p50 | Kernel p95 | p50 speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.832 ms | 21.246 ms | 7.724 ms | 8.031 ms | 2.57× |
| 2 | 19.497 ms | 20.086 ms | 7.725 ms | 7.805 ms | 2.52× |
| 3 | 19.529 ms | 19.984 ms | 7.724 ms | 7.845 ms | 2.53× |

Across-run median p50/p95 was 19.529/20.086 ms for the scalar path and 7.724/7.845 ms for the kernels, a 2.53× median p50 speedup. The maximum absolute output difference was `8.24e-10`.

## Method and scope

The scalar baseline reproduces the former vision path: f64 query-key dot products, f32 logits and normalized probabilities, and f64 value accumulation. The optimized path uses first-party AArch64 NEON f64-accumulating dot and weighted-value kernels, stable f64 softmax weights, and caller-owned output buffers. Both paths operate on the same precomputed QKV rows; this measures attention math only. Q4 matrix projection, image preprocessing, the rest of the 24 vision blocks, a real checkpoint, task quality and end-to-end image-encoder latency are excluded. The host has 8 GiB RAM and cannot satisfy the 16 GiB acceptance gate.

## Reproduction

```sh
cargo test --offline -p sage-qwen35-runtime --release --lib qwen35_vision::tests::vision_attention_kernel_latency_measurement -- --ignored --nocapture --test-threads=1
```

Each invocation alternates the two paths for 101 samples, reports p50 and nearest-rank p95, and checks maximum absolute difference below `1e-7`.

## Production bounded-worker path

The current production attention path partitions independent query rows across Sage's bounded CPU worker count. This measurement includes scoped worker creation and joining, per-worker scratch allocation/zeroization, and output writes. The scalar baseline remains the former single-thread attention arithmetic. Three alternating release runs on the same 8 GiB Apple Mac15,3 host measured:

| Run | Scalar p50 | Scalar p95 | Bounded workers p50 | Bounded workers p95 | p50 speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 21.371 ms | 22.059 ms | 2.005 ms | 2.228 ms | 10.66× |
| 2 | 21.122 ms | 22.053 ms | 2.008 ms | 2.299 ms | 10.52× |
| 3 | 21.054 ms | 22.022 ms | 2.026 ms | 2.396 ms | 10.39× |

Across-run median p50/p95 was 21.122/22.053 ms for the scalar baseline and 2.008/2.299 ms for the bounded-worker path, a 10.52× median p50 speedup. Maximum absolute output difference was `8.15e-10`. This is a synthetic attention-only benchmark at 128 tokens, hidden width 1,024, 16 heads and 64 values per head; it includes thread startup but excludes the full vision block, real checkpoint, image preprocessing, model quality and end-to-end encoding. It does not satisfy the 16 GiB hardware acceptance gate.

The normal vision test suite also runs a two-worker 64-token fixture and compares its output against the single-range first-party kernel at an absolute tolerance of `1e-7`.
