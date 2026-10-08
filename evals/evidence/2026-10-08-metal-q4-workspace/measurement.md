# Metal Q4 workspace measurement

This measurement evaluates one first-party Q4 Metal projection optimization. The matrix is 2,560×2,560 with group size 128 and Qwen-shaped input. Each line is a separate release-test process with 101 samples per path. P50 and p95 are computed within each process; the table reports the median of each percentile across three runs.

Host: Mac15,3, Apple M3, arm64, 8 GB RAM, macOS 27.0. This is the available development host, not the plan's 16 GB acceptance device.

Command:

```sh
cargo test -p sage-core --features qwen35-evaluation qwen_hidden_q4_projection_latency_measurement --release --locked -- --ignored --nocapture --test-threads=1
```

| Route | p50 | p95 |
|---|---:|---:|
| CPU NEON, allocating result | 212 μs | 246 μs |
| CPU NEON, caller-owned result | 211 μs | 265 μs |
| Metal, fresh shared input/output buffers each call | 2,574 μs | 2,989 μs |
| Metal, retained workspace, allocating result | 2,563 μs | 3,035 μs |
| Metal, retained workspace and caller-owned result | 2,553 μs | 3,013 μs |

Retained buffers remove two Metal buffer-object allocations from each projection and avoid the caller-result allocation when the caller supplies output storage. Across these runs the p50 difference is below 1%; p95 did not improve consistently. Metal remains about 12× slower than CPU NEON for this matrix on this host, so this result does not justify routing the measured projection to Metal. The packed-pair nibble kernel passed numeric tests, but this benchmark does not isolate its speed against the previous shader.

Raw runs:

```text
run 1: cpu=212/246 us cpu_reused=209/241 us metal_fresh=2569/2963 us metal_retained_allocating=2563/2963 us metal_retained_into=2553/2908 us
run 2: cpu=211/238 us cpu_reused=211/265 us metal_fresh=2574/2989 us metal_retained_allocating=2563/3035 us metal_retained_into=2549/3013 us
run 3: cpu=228/297 us cpu_reused=228/286 us metal_fresh=2671/3174 us metal_retained_allocating=2615/3307 us metal_retained_into=2599/3175 us
```

The Qwen memory envelope reserves 10,752,000 bytes for the fixed Metal activation buffers across the indexed Q4 projection matrices. This is a software estimate, not measured resident memory. No real checkpoint generation, 16 GB acceptance, or end-to-end model timing is claimed.
