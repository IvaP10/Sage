# Q4 Metal shared and private weight-storage comparison

Date: 10 October 2026. Host: Apple Mac15,3 (arm64, 8 GiB physical memory). Rust: pinned 1.89.0. Release build; one Qwen-shaped `10,240 × 2,560` grouped-Q4 projection, group size 128.

## Question and method

The Q4 Metal path used CPU-visible shared-storage buffers for immutable weights. This measurement asks whether uploading the packed weights and scales once into GPU-private storage improves repeated projection latency. The private upload uses a shared staging allocation, a Metal blit, waits for completion, and zeroes the staging bytes before returning. The private buffer is intentionally not CPU-readable.

The fixture uses deterministic synthetic packed weights and scales, not checkpoint tensor values. CPU, shared-Metal, private-Metal, and a four-row shared-input Metal kernel each use retained outputs and workspaces. Routes are warmed twice, then sampled 31 times in rotating order. CPU/Metal outputs are checked with the tolerance `1e-3 + 1e-3 × abs(reference)`. GPU execution intervals come from completed Metal command-buffer timestamps. Five independent release processes passed parity; their raw sample vectors are in [`run-1.txt`](run-1.txt), [`run-2.txt`](run-2.txt), [`run-3.txt`](run-3.txt), [`run-4.txt`](run-4.txt), and [`run-5.txt`](run-5.txt).

Command:

```sh
rustup run 1.89.0-aarch64-apple-darwin cargo test --offline --locked --release -p sage-inference-math --lib tests::q4_shared_private_metal_projection_measurement -- --ignored --exact --nocapture --test-threads=1
```

## Result

| Route | p50 | p95 |
| --- | ---: | ---: |
| CPU NEON Q4 | 575 μs | 873 μs |
| Metal, shared weights | 11,317 μs | 12,802 μs |
| Metal, private weights | 11,436 μs | 12,820 μs |
| Metal, four rows per threadgroup | 11,259 μs | 12,721 μs |

The Metal command buffer also reports its GPU execution interval after completion:

| Metal storage | GPU execution p50 / p95 |
| --- | ---: |
| Shared | 10,967 / 12,553 μs |
| Private | 11,097 / 12,524 μs |
| Four-row tile, shared weights | 10,917 / 12,371 μs |

Values are medians of five independent process-level p50/p95 measurements. The private/shared wall p50 ratio was 1.011 and the private/shared GPU-time p50 ratio was 1.008. The four-row tile's median p50 ratio against the shared one-row kernel was 1.000; its p95 difference was under 1%. Neither technique showed a reliable speedup. GPU execution accounted for about 97% of shared-route median wall time, so host submission and completion waiting are not the main source of the measured gap. The shared Metal route was about 19.7× slower than CPU at median p50.

## Conclusion and limits

Changing immutable weight storage alone does not explain the Metal performance gap at this API boundary. Tiling four output rows to reuse one input vector also failed to produce a reliable speedup; it remains a measurement-only kernel and is not selected by the Qwen loader. Since the kernel itself dominates wall time, future GPU work should focus on substantially different Q4 decoding or fused device-resident operations before considering that route.

This is one synthetic shape on one 8 GiB M3. It does not measure real checkpoint values, a full decoder, energy, other GPU generations, or end-to-end generation. No production backend route was changed from this experiment.
