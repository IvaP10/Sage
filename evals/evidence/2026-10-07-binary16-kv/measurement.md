# Binary16 full-attention cache and kernel measurement

Date: 2026-10-07  
Hardware: Apple M3  
Command: `cargo test -p sage-kernels --release --locked --offline binary16_attention_kernel_latency_measurement -- --ignored --nocapture --test-threads=1`

Sage now stores full-attention keys and values as IEEE binary16 and widens them to f32 inside its own QK and weighted-value kernels. On AArch64, a runtime-checked native `FCVTL` instruction widens four lanes when FP16 is supported; older AArch64 targets use the Sage-written integer-vector conversion with a scalar fallback for half subnormals. The resource envelope uses two bytes per cached element, and the append path rejects non-finite or out-of-range f32 values before mutating cache state.

The synthetic benchmark uses 2,048 positions, 64 query/value columns, a 256-element row stride, normalized weights, and 101 alternating samples per invocation. The f32 baseline contains the exact f32 dequantizations of the same binary16 fixture, so the kernel comparison isolates the storage format and conversion path. Each invocation includes both QK score and weighted-value kernels. Five post-optimization release invocations reported:

| Run | f32 p50 / p95 | binary16 p50 / p95 | Maximum score/value difference |
|---|---:|---:|---:|
| 1 | 33 / 34 μs | 35 / 36 μs | 0 / 0 |
| 2 | 20 / 21 μs | 21 / 21 μs | 0 / 0 |
| 3 | 19 / 21 μs | 20 / 21 μs | 0 / 0 |
| 4 | 19 / 21 μs | 20 / 20 μs | 0 / 0 |
| 5 | 30 / 34 μs | 31 / 36 μs | 0 / 0 |

Across the five invocations, the median of run-level p50 values is 20 μs for f32 and 21 μs for binary16; the median of run-level p95 values is 21 μs for both. Per-process values vary with host scheduling/frequency, so the result supports near-parity on this fixture, not a general speedup claim. A preceding run before hoisting runtime feature detection out of the vector loop measured 154 μs for binary16 versus 28 μs for f32; that result exposed and drove removal of repeated feature-lock checks from each vector conversion. After native widening but before hoisting the feature check, binary16 measured 81 μs versus 33 μs; hoisting made the check happen once per kernel call.

At the Qwen3.5-4B 8K geometry, the eight full-attention layers' theoretical KV storage falls from 512 MiB to 256 MiB, saving 256 MiB. The updated profile envelope fixture reports 4,361,996,288 bytes with this representation. That figure is still a software estimate; the real checkpoint has not been loaded after this change. Focused kernel and cache tests cover exhaustive finite binary16 round trips, ties-to-even, NEON parity, subnormal lanes, cache range rejection, zeroizing growth/clear/drop, and synthetic full-precision attention error below 5e-4. Checkpoint parity, long-context task quality, end-to-end latency, and 16 GiB hardware acceptance remain open.
