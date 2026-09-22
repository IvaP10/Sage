# Model setup

The executable has two routes: an explicitly configured compatible provider, or the supervised local CPU prototype selected with a signed profile. No model weights or qualified inference runtime are bundled. See [candidate metadata](../evals/models/qwen3.5-4b-candidate.json) and [qualification status](v2/implementation-status.md).

## Configured provider

Open Sage, unlock protected storage when requested, and configure the reasoning provider in Settings. Choose the OpenAI preset or an OpenAI-compatible service, enter the actual model name and endpoint, and supply any required credential through the native settings form. Use the connection test; this sends a small synthetic test request, not conversation history.

Provider endpoints require HTTPS. Exact `localhost`, `127.0.0.1` and `[::1]` HTTP endpoints are allowed as explicitly configured external routes. URLs cannot contain credentials, query parameters or fragments. Requests use pinned validated DNS addresses, no proxy and no redirect following. Compatible services must support Sage's bounded structured response contract; unsupported capability errors are surfaced.

A saved endpoint or credential is not permission to disclose task data. Every inference call needs approval for the displayed exact context and destination. Changing an endpoint/path/port does not reuse its old credential. Legacy accounts are preserved in the OS store but disconnected during migration; re-enter a credential for the intended endpoint. There is no silent cloud fallback.

## Managed local prototype

A profile must be signed by an explicitly supplied Ed25519 public key and contain verified model/runtime/dependency digests, revision/expiry, license, evaluation digest, context/output bounds and total memory envelope. Only the current macOS CPU isolation prototype is accepted; unsupported platforms/backends refuse execution.

```sh
cargo run --locked -p sage-core -- --model-profile /absolute/path/to/signed-profile.json --model-trust-key /absolute/path/to/ed25519-public-key.bin
```

The key file is a **public** verification key of exactly 32 bytes. Keep signing private keys outside Sage and the repository. This development route does not install a production trust root, model catalog, downloader or qualification result. Do not generate a signed production profile from estimated memory or made-up evaluation hashes.

The target baseline is a reproducible Qwen3.5-4B Q4_K_M conversion for llama.cpp, 8K context and one active generation. Quantization, backend compatibility, warm latency, peak memory and task quality remain unmeasured here. The current development Mac has 8 GiB RAM; release qualification requires the specified 16 GB hardware matrix.

The prototype clears the environment, restricts file/network/process access, bounds input/output, checks asset digests at admission and stops on cancellation/timeout. It reloads per request and lacks the target warm server, tokenizer allocation, platform RSS enforcement and fully separated native service. It is not the final production inference runtime.
