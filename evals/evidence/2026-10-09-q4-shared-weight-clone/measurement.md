# Shared Q4 weight-clone measurement

Date: 9 October 2026  
Host: Apple M3 (Mac15,3), 8 GiB RAM, macOS 27.0.1  
Compiler: Rust 1.98.0, `aarch64-apple-darwin`, optimized release profile

## Hypothesis and implementation

`QuantizedQ4Matrix` contains immutable CPU scale and packed-nibble arrays. Its previous manual `Clone` allocated two new arrays and copied their contents for each matrix clone. The implementation now derives `Clone` over the existing `Arc<Vec<_>>` fields, so cloned model layers share the CPU payload. Matrix metadata and backend state remain per clone.

When a clone moves its weights to Metal, it transfers the selected tensor into Metal storage. It zeroizes the CPU payload if it is the sole owner; when other CPU clones remain, it detaches itself from their shared payload and leaves their data intact. The Mac Metal parity test exercises this split on an M3 device.

## Method

The fixture is Qwen's tied-output-head shape: 248,320 rows × 2,560 columns, group size 128. It is generated directly in packed Q4 form and contains no checkpoint values. Scale and packed arrays total 337,715,200 bytes.

Each release-test process takes 21 paired samples. For the shared route, it clones the matrix into a preallocated vector 16,384 times, retains those clones until after timing, and divides elapsed time by the number of clones. This includes the vector pointer store and keeps every Arc clone live during the sample. The deep-copy comparison reproduces the former implementation, including both vector allocations and copies, once per sample. Each route checks shared identity or exact value equality. Reported p50 and p95 are sample indices 10 and 19 after sorting the 21 samples. Five separate test processes were run; raw paired times below are `(shared clone ns, deep-copy ns)`.

## Results

| Process | Shared clone p50 / p95 (ns) | Deep copy p50 / p95 (μs) |
|---|---:|---:|
| 1 | 3 / 6 | 8,655 / 8,868 |
| 2 | 3 / 5 | 8,715 / 10,746 |
| 3 | 3 / 6 | 8,758 / 9,501 |
| 4 | 3 / 4 | 8,728 / 8,939 |
| 5 | 3 / 4 | 8,772 / 9,289 |

Across-process medians were 3 ns p50 and 5 ns p95 for shared clones, versus 8,728 μs and 9,289 μs for deep copies. The confirmed improvement is that cloning no longer allocates or copies the 337,715,200-byte weight payload. The nanosecond timings measure a preallocated metadata-container clone in this microbenchmark; they are not model-load, generation, multi-agent, or whole-application latency results.

## Raw paired samples

```text
1: (7,48100084) (6,8868125) (3,8599459) (3,8685041) (3,8633542) (3,8652500) (3,8644209) (3,8635375) (3,8659583) (3,8640041) (3,8655750) (3,8727917) (3,8606584) (3,8663708) (3,8637459) (3,8657666) (3,8672500) (3,8684875) (3,8655125) (3,8648000) (3,8698666)
2: (6,24993209) (5,8734792) (3,8712083) (3,8737959) (3,8803167) (3,8663792) (3,8774584) (3,8696625) (3,8689917) (3,8711833) (3,8743542) (3,8682041) (3,8726625) (3,8715541) (3,10746750) (3,8713167) (3,8986375) (3,8703708) (3,8756542) (3,8700291) (3,8679625)
3: (7,24612709) (6,9501083) (3,8843625) (3,8736625) (3,8799834) (3,8727875) (3,8760917) (3,8840875) (3,8729833) (3,8787541) (3,8750250) (3,8758292) (3,8768500) (3,8782291) (3,8704750) (3,8728709) (3,8724334) (3,8739625) (3,8771917) (3,8746500) (3,8693250)
4: (6,27413416) (3,8746625) (3,8939459) (3,8755417) (3,8720833) (3,8713167) (3,8714416) (3,8724166) (3,8739584) (3,8728667) (3,8721792) (3,8756667) (3,8816083) (4,8703125) (3,8711333) (3,8794000) (3,8717042) (3,8877541) (3,8757542) (3,8709167) (3,8718750)
5: (5,27352167) (4,8707500) (3,8834583) (3,8913625) (3,9204875) (3,9020750) (3,9289750) (3,8727792) (3,8707583) (3,8738208) (3,8716417) (3,8748875) (3,8787792) (3,8723750) (3,8722042) (3,8754166) (3,8889417) (3,8772500) (3,8750250) (3,8775959) (4,8784833)
```

The exact command for each process was:

```sh
cargo test -p sage-inference-math --locked --release tests::qwen_q4_weight_clone_latency_measurement -- --ignored --exact --nocapture
```

## Limits

This is a synthetic ownership benchmark. No real Qwen shard, product model provider, or multi-agent task was loaded or exercised. Sharing also intentionally extends the host-array lifetime until the final clone drops. The Metal test passed on this host; cross-platform runtime acceptance remains separate.
