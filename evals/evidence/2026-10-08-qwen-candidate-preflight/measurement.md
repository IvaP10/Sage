# Pinned Qwen candidate preflight

Date: 8 October 2026. Host: Apple Mac15,3, arm64, 8 GiB physical memory.

## First-party tokenizer check

Sage loaded the pinned `tokenizer.json` (12,807,982 bytes; SHA-256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`) at checkpoint revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The ignored release test passed in 0.24 seconds:

```sh
SAGE_QWEN35_TOKENIZER_JSON=<pinned-candidate>/tokenizer.json cargo test --offline -p sage-qwen-tokenizer --release --lib tests::pinned_checkpoint_tokenizer_matches_slow_bpe_reference -- --ignored --exact --nocapture --test-threads=1
```

The test checks the pinned digest, vocabulary size, seven structural token IDs, and eight short multilingual, punctuation, whitespace, numeric, emoji and literal-marker strings. Sage's optimized BPE merge path matched its deliberately slow merge reference for each string, and decoding matched Sage's NFC-normalized text.

This is narrow tokenizer evidence: both paths share Sage's Unicode normalization and pre-token segmentation, and the fixture is not the full official tokenizer test corpus. Full reference-vector coverage and integration with an admitted model remain open. The same pinned artifact test was rerun against the locally available candidate on 8 October and passed in 0.26 seconds.

## Candidate memory envelope

The ignored release preflight using the pinned index and image preprocessor passed with a 128-value Q4 group and the 8K context limit:

```sh
SAGE_QWEN35_INDEX_JSON=<pinned-candidate>/model.safetensors.index.json SAGE_QWEN35_PREPROCESSOR_JSON=<pinned-candidate>/preprocessor_config.json cargo test --offline -p sage-qwen35-runtime --release --lib loader::tests::pinned_checkpoint_memory_envelope_fits_the_initial_16_gib_tier -- --ignored --exact --nocapture --test-threads=1
```

The 8 October rerun estimated 4,379,990,016 bytes: 2,429,601,792 weights, 322,437,120 state, 1,073,741,824 activations, and 554,209,280 runtime. The envelope fits Sage's 8 GiB per-job ceiling for the 16 GiB desktop tier, but that estimate is not an OS-enforced limit. No tensor values were imported, and no generation or 16 GiB acceptance was tested. This development host has 8 GiB physical memory, so the full candidate load was not attempted.

## Pinned shard validation rerun

On 8 October, the ignored release test `loader::tests::pinned_checkpoint_shard_headers_match_the_first_party_tensor_map` passed against the locally available candidate in 22.62 seconds. It streamed and checked both pinned shard files, then validated all 738 indexed tensor headers covering 9,319,737,856 weight bytes. The reported config, tokenizer, and processor manifest digests were identical. The run did not import tensor values or establish model numerical parity.

The current ignored candidate-import test validates model assembly and context configuration but does not generate tokens. Persistent-worker generation and its real-checkpoint acceptance remain open. Synthetic Metal parity and performance are separate kernel checks.

## 10 October metadata revalidation

Using the currently cached files for revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, pinned Rust 1.89 reran three ignored checks successfully: closed metadata/profile validation (1 test), the pinned tokenizer-versus-slow-BPE fixture (1 test), and the Q4 8K memory/image-preprocessor envelope (1 test). The latter remained 4,379,990,016 bytes: 2,429,601,792 weights, 322,437,120 state, 1,073,741,824 activations, and 554,209,280 runtime. These checks consume only the small metadata/tokenizer assets; they do not inspect model tensor values or establish numerical parity.

At the same inspection, the two live shard downloads contained 1,776,399,846 and 1,906,160,833 bytes, respectively, versus pinned sizes 5,329,398,688 and 3,990,429,408. The current partial files were not suitable for header validation or model loading. This is a point-in-time download snapshot, not a completion claim. Full-shard digest/header validation, real-checkpoint generation, product-worker integration, and 16 GiB hardware acceptance remain open.

## 10 October complete-shard revalidation

The pinned cache now contains both exact shard sizes. The release-only full-shard test was rerun against revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and passed in 19.36 seconds. It re-hashed both artifacts against the evaluation manifest and validated all 738 tensor headers and 9,319,737,856 indexed weight bytes. Config, tokenizer, and processor remained bound to the same manifest. This verifies the checkpoint files and tensor map; it does not import the tensors or prove numerical parity.

The ignored import-and-one-constrained-token smoke test was also attempted. The 8 GiB host's `ResourceGovernor` refused admission before tensor import because the measured 4,379,990,016-byte model envelope could not be reserved while preserving its OS headroom. This is an expected resource-gate refusal, not a loader correctness result. Do not relax admission to force the test on this machine; full import, worker generation, numerical/task quality, and 16 GiB acceptance remain open.
