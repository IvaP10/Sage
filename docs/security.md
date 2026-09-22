# Sage security

The detailed [v2 threat model](v2/threat-model.md) defines the assets, adversaries, enforced controls, recovery state machine and remaining production gates. The [architecture](v2/architecture.md) specifies the target process boundaries.

Current enforcement includes empty default file roots, protected Sage/model/credential paths, typed closed operations, exact preparation and approvals, Cedar dispatch policy, single-use worker-bound grants, cancellation revocation, DNS-pinned egress without redirects/proxies, explicit context release, role-separated mutual IPC authentication, SQLCipher and protected audit checkpoints.

Model output, tool results, documents, webpages, repository text and memory are untrusted. Recalled material cannot authorize an operation or transmission. Credentials stay in the credential store; derived context remains private. A model's structural JSON validity, successful dispatch or process exit cannot establish arbitrary task success.

Arbitrary host code execution is removed. Generic UI mutation and privileged operations are unavailable. The VM, WASM/MCP boundaries, signed native/credential services, production update trust and hardware acceptance remain required work. Same-user file credentials and UI-provided native-authentication Booleans do not meet the target production identity contract.

Hash-chain verification checks a separately stored OS checkpoint on unlock and around effects. Anchored truncation, changed records and missing anchors fail closed. An unanchored tail between checkpoints remains a recovery limitation. The OS, administrator and trusted-service compromise are outside application-only guarantees; the log is not immutable.

Do not publish a v2 production build until `python3 scripts/check-v2-release.py --require-ready` succeeds with current, attributable evidence for every gate. Source tests do not replace physical device, attack, signing or installation evidence.
