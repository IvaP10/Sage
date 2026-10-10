# Grouped-Q4 CPU row scheduling and batched Metal comparison

Date: 9 October 2026  
Host: Apple M3, macOS 27.0.1, 8 GiB RAM  
Build: optimized Rust release profile

## CPU scheduling result

When a Q4 batch fit in one activation tile, the CPU scheduler previously assigned the entire output matrix to one worker. The new path transposes the bounded activation tile once, divides output rows among available CPU workers, and shares the immutable tile. On AArch64 it keeps the NEON kernel's four-input weight reuse. Each worker writes distinct row ranges in every batch result; the API's batch-major layout and scratch zeroization are preserved.

Three processes each collected 31 alternating CPU and Metal samples for a synthetic 4,096×1,024 grouped-Q4 matrix (group size 128), using batches of 1, 4, 16, 64, and 256 inputs. Values are p50/p95 microseconds within each process; the summary takes the median of each percentile across processes.

| Batch | CPU run 1 | CPU run 2 | CPU run 3 | CPU median p50/p95 | Metal run 1 | Metal run 2 | Metal run 3 | Metal median p50/p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 190/227 | 185/228 | 190/235 | 190/228 | 2069/2199 | 2088/2338 | 2138/2439 | 2088/2338 |
| 4 | 1080/1172 | 883/1142 | 1126/1404 | 1080/1172 | 2283/2427 | 2189/2379 | 2416/2739 | 2283/2427 |
| 16 | 6178/9223 | 3755/4574 | 6220/6661 | 6178/6661 | 5994/7326 | 6011/7886 | 6068/8359 | 6011/7886 |
| 64 | 7757/7915 | 7871/9159 | 8418/9194 | 7871/9159 | 22422/22612 | 22456/22658 | 22381/22650 | 22422/22650 |
| 256 | 20496/23752 | 20319/22831 | 20824/22817 | 20496/22831 | 88028/88183 | 88032/88649 | 88056/88218 | 88032/88218 |

For batch size 4, the previous single-worker CPU route measured a three-process median of 3,197/3,411 μs. Row scheduling now measures 1,080/1,172 μs, a 66.2% p50 and 65.6% p95 reduction. CPU is faster than Metal at batches 1, 4, 64, and 256. Batch 16 is near parity: Metal p50 is 6,011 μs versus CPU at 6,178 μs, while Metal p95 is slower (7,886 μs versus 6,661 μs). At batch 256 the CPU p50 is about 4.3× faster. The measured Metal call includes host-visible activation/output handling, command submission, completion wait, and workspace scrubbing. Weight upload is outside the timed region. The batch-16 CPU spread shows sensitivity to scheduling or host load; this synthetic comparison does not establish a production route choice.

This experiment supports row-parallel CPU scheduling for small multi-input batches. It does not qualify the Metal kernel for product inference, establish GPU performance on other devices or operators, or measure a real checkpoint, full vision encoding, generation, or task quality. Keep the Metal batch implementation as an explicit evaluation baseline only.

Reproduction command:

```sh
cargo test --offline --release -p sage-inference-math --lib tests::q4_cpu_metal_batched_projection_measurement -- --ignored --exact --nocapture
```

## Portability checks

The kernel, Metal stub, inference math, and Qwen runtime also compile with the pinned Rust 1.89 toolchain for Intel macOS and Windows GNU. These are compile checks only; the cross-target binaries were not run on those operating systems.

```sh
RUSTC="$(rustup which --toolchain 1.89.0 rustc)" rustup run 1.89.0 cargo check --offline -p sage-kernels -p sage-metal -p sage-inference-math -p sage-qwen35-runtime --target x86_64-apple-darwin
RUSTC="$(rustup which --toolchain 1.89.0 rustc)" rustup run 1.89.0 cargo check --offline -p sage-kernels -p sage-metal -p sage-inference-math -p sage-qwen35-runtime --target x86_64-pc-windows-gnu
```
