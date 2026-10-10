# Qwen tokenizer ASCII NFC bypass

Date: 2026-10-09  
Host: Apple M3, arm64, 8 GiB RAM  
Toolchain: rustc 1.98.0, optimized Rust release test binary

## Change

`encode_text_into` now borrows ASCII input instead of passing it through NFC normalization. ASCII is already normalized, so this avoids the normalizer's decomposition/composition vectors and normalized output string for the common ASCII path. Non-ASCII input still uses the exact Sage-owned Unicode 15 NFC implementation.

## Method

The fixture encodes 3,500 ASCII bytes (`"abcd "` repeated 700 times) to 1,400 tokens with the synthetic BPE vocabulary already used by the tokenizer tests. The baseline explicitly runs the existing NFC normalizer and sends its output through the same BPE/pre-tokenization implementation. The candidate calls the production entry point. Both use fresh output and tokenizer scratch buffers per encode; each process alternates baseline/candidate order, discards 32 warm-up pairs, and checks every output against the same expected token sequence. Each process records 301 paired samples.

Command:

```sh
cargo test --locked --offline -p sage-qwen-tokenizer --lib --release ascii_normalization_fast_path_measurement -- --ignored --nocapture
```

## Results

| Process | Full NFC p50 / p95 | ASCII borrow p50 / p95 |
|---|---:|---:|
| 1 | 209.542 / 233.000 μs | 155.375 / 167.041 μs |
| 2 | 210.416 / 222.250 μs | 154.875 / 168.292 μs |
| 3 | 211.083 / 222.417 μs | 156.417 / 167.708 μs |
| Median run | 210.416 / 222.417 μs | 155.375 / 167.708 μs |

The median run-level p50 and p95 fell 26.2% and 24.6%. The output tokens matched in all samples. The mechanism also removes the Unicode normalizer's temporary allocations for this ASCII input; allocator counts and peak memory were not measured.

## Limits

The synthetic merge table does not represent the pinned Qwen tokenizer's actual merge distribution. The result measures one 3.5 KiB ASCII text encode, not full chat formatting, Core prompt construction, decoder prefill, generation, task quality, or the 16 GiB hardware gate. Unicode normalization behavior is unchanged for all non-ASCII input; the existing Unicode and slow-reference tokenizer tests remain the correctness evidence for that path.
