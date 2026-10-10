# Qwen tied-output-head CPU and Metal route measurement

Date: 10 October 2026. Host: Apple Mac15,3 (arm64, 8 GiB physical memory). Rust: pinned 1.89.0. The run used the optimized release profile and a Metal-capable Apple GPU.

## Question

The CPU Q4 route selects an unconstrained token with a fused projection/argmax kernel. For a Metal-backed candidate, `Qwen35ProjectionMatrix::project_argmax` returns no fused result; generation then projects all vocabulary logits and scans them. This benchmark compares those actual routes at the pinned Qwen3.5-4B tied-embedding geometry before changing the Metal evaluation loader.

## Method

The fixture uses the pinned 248,320-row vocabulary, 2,560 hidden values, and 128-value Q4 groups. It constructs deterministic synthetic packed Q4 weights using approximately 337 MiB for the matrix and a deterministic finite hidden vector. These are shape-faithful synthetic weights, not checkpoint tensor values.

CPU timing is `project_argmax_cpu`, which fuses Q4 projection and argmax and returns the selected row/value. Metal timing includes the retained-workspace Q4 projection into the full vocabulary vector followed by the same deterministic host-side max scan used by generation. Both routes were warmed three times, then sampled 31 times in alternating order. The test separately computes all CPU and Metal logits, checks every value against `1e-3 + 1e-3 * abs(reference)`, and records whether the selected row agrees.

Command:

```sh
rustup run 1.89.0-aarch64-apple-darwin cargo test --offline --locked --release -p sage-inference-math --lib tests::qwen_metal_output_head_route_measurement -- --ignored --exact --nocapture --test-threads=1
```

## Result

| Route | p50 | p95 |
| --- | ---: | ---: |
| CPU fused Q4 argmax | 12,634 μs | 13,961 μs |
| Metal Q4 projection + host argmax | 299,084 μs | 341,161 μs |

At this shape, Metal took 23.7× the CPU p50 and 24.4× the CPU p95. Maximum absolute logit difference was `0.00000034`, within the tested tolerance, and both routes selected the same row. Timings exclude model loading, Metal upload, full decoder layers, prompt prefill, energy and task quality.

CPU samples in microseconds (sorted):

```text
[11909, 12028, 12066, 12125, 12138, 12169, 12221, 12231, 12332, 12374, 12384, 12422, 12438, 12586, 12628, 12634, 12648, 12659, 12687, 12825, 12907, 12934, 13049, 13160, 13339, 13365, 13447, 13532, 13575, 13961, 14047]
```

Metal samples in microseconds (sorted):

```text
[290703, 291751, 292284, 292662, 294959, 295559, 295621, 295692, 295738, 296052, 296096, 296137, 297462, 298456, 298861, 299084, 299143, 299172, 299550, 299617, 299658, 299732, 299759, 299770, 300912, 301353, 301962, 304812, 305141, 341161, 341991]
```

## Implementation consequence and limits

The evaluation loader now leaves `model.language_model.embed_tokens.weight` in CPU Q4 storage even when other projections opt into the Metal evaluation path. This preserves CPU embedding lookup and the fused CPU output-head argmax instead of routing the tied output head through the measured slower Metal path. A loader policy test covers the selection. This is a shape-specific route decision supported by one M3 fixture; it does not establish the best Metal/CPU policy for other shapes, devices, actual checkpoint activations, or product generation. The model remains an unadmitted candidate and the worker remains disconnected from generation.
