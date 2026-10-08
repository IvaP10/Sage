# Pinned Qwen candidate preflight

Date: 8 October 2026. Host: Apple Mac15,3, arm64, 8 GiB physical memory.

## First-party tokenizer check

Sage loaded the pinned `tokenizer.json` (12,807,982 bytes; SHA-256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`) at checkpoint revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The ignored release test passed in 0.24 seconds:

```sh
SAGE_QWEN35_TOKENIZER_JSON=<pinned-candidate>/tokenizer.json cargo test --offline -p sage-core --release --features qwen35-evaluation --lib qwen_tokenizer::tests::pinned_checkpoint_tokenizer_matches_slow_bpe_reference -- --ignored --exact --nocapture --test-threads=1
```

The test checks the pinned digest, vocabulary size, seven structural token IDs, and eight short multilingual, punctuation, whitespace, numeric, emoji and literal-marker strings. Sage's optimized BPE merge path matched its deliberately slow merge reference for each string, and decoding matched Sage's NFC-normalized text.

This is narrow tokenizer evidence: both paths share Sage's Unicode normalization and pre-token segmentation, and the fixture is not the full official tokenizer test corpus. Full reference-vector coverage and integration with an admitted model remain open.

## Candidate memory envelope

The ignored release preflight using the pinned index and image preprocessor passed with a 128-value Q4 group and the 8K context limit:

```sh
SAGE_QWEN35_INDEX_JSON=<pinned-candidate>/model.safetensors.index.json SAGE_QWEN35_PREPROCESSOR_JSON=<pinned-candidate>/preprocessor_config.json cargo test --offline -p sage-core --release --features qwen35-evaluation --lib qwen35::weight_loader::tests::pinned_checkpoint_memory_envelope_fits_the_initial_16_gib_tier -- --ignored --exact --nocapture --test-threads=1
```

It estimated 4,379,969,536 bytes: 2,429,601,792 weights, 322,437,120 state, 1,073,741,824 activations, and 554,188,800 runtime. The envelope fits Sage's 8 GiB per-job ceiling for the 16 GiB desktop tier, but that estimate is not an OS-enforced limit. No tensor values were imported, and no generation or 16 GiB acceptance was tested. The live host reported 48% system-wide free memory during this preflight, so the full candidate load was not retried on this 8 GiB machine.

The real-checkpoint generation test now uses Sage's CPU Q4 path by default and no longer initializes Metal as a prerequisite. Synthetic Metal parity and performance remain separate kernel checks.
