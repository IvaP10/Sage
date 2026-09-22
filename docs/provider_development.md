# Provider development

Implement the `ModelProvider` contract in `crates/sage-core/src/model.rs`. The normal path is `next_turn_stream(TurnContext, updates)`, which returns `ModelTurn::Answer` or a bounded `ActionGraph`. Results from completed actions re-enter the next turn. The model does not issue grants, select weaker success checks, install policy or treat recalled content as instructions.

Use the closed feature descriptors supplied by the broker and the JSON schema produced by `draft_schema`. A valid response contains an answer and no actions, or supported actions and no answer. Sage assigns action identity and untrusted provenance, validates arguments independently, and binds installed verification requirements. Streaming may expose an answer prefix; it must not dispatch a partial action.

Externally managed providers must return their exact data destination, including model identity, and verify it still matches the route authorized for this call. Use the network broker, credentials bound to the normalized request endpoint, bounded bodies, no redirects/proxies and generic credential-safe errors. Test requests contain synthetic data. Settings are not conversation disclosure consent.

The no-destination exemption is only for a broker-supervised local worker. A provider descriptor, localhost URL, arbitrary plugin or a claimed local flag is not evidence of local processing. Third-party executable providers are not currently loadable; future adapters need isolated workers and the full feature conformance contract.

Extend deterministic tests for malformed/oversized output, secret-safe errors, changed destinations, denied context release, cancellation, schema violations, result-dependent planning and unsupported provider capabilities. Keep real model, hardware, prompt-injection and performance evidence separate from fixtures. See [migration and evaluation](v2/migration-and-evaluation.md).
