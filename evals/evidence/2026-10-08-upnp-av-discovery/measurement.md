# Passive UPnP-AV renderer discovery

Date: 8 October 2026. Validation host: Apple arm64 Mac with 8 GiB physical memory.

## Scope

Sage now has a first-party Rust discovery crate at `crates/sage-upnp-av`. It sends one IPv4 SSDP `M-SEARCH` to the standard multicast group with TTL 2. Core requests a two-second search; the crate rejects configured windows above six seconds, accepts at most 16 unique device candidates, and performs at most four description fetches concurrently. HTTP requests have a two-second timeout, no proxy, and no redirects. A description is limited to 64 KiB and the in-house XML parser bounds the document to 4,096 nodes and depth 32 while rejecting DTD declarations and custom entities.

Only private, link-local, or loopback responders are considered. The SSDP `LOCATION` and any `URLBase`/AVTransport control URL must remain HTTP on the same responding IP. Device identity in the description must match the SSDP UDN. The returned record includes the observed identity, descriptive fields, AVTransport endpoint candidate, timestamp, and SHA-256 digest of the bounded description.

These values are untrusted, ephemeral observations. Sage does not persist them, pair with the device, establish a peer lease, or send AVTransport commands. The UI labels candidates as untrusted and unpaired. The negotiated `lan_device_discovery_v1` IPC feature gates the route; a capacity-one observation queue isolates discovery from regular and control commands. The macOS bundle explains that local-network permission is requested only when the user starts a scan.

## Validation

- `cargo test --locked -p sage-upnp-av`: 5 passed. Four parser/security tests cover private responder and same-host checks, device/UDN matching, URL-base and control URL origin restrictions, credentials/query/fragment rejection, XML bounds, DTD/custom-entity rejection, and nested identity-field rejection. A loopback integration test exercises the bounded SSDP request/response and HTTP description fetch end to end.
- The focused Core dispatcher test passed, confirming that only `discover_upnp_media_renderers` is assigned to the isolated observation lane.
- The Qwen-enabled Sage Core library suite passed: 317 passed, 8 ignored. This broader run preceded the dispatcher-only test addition; the new route test passed separately afterward.
- Strict all-target Clippy passed for `sage-core` with `qwen35-evaluation` and `sage-upnp-av`, including the final URLBase fixture and dispatcher routing test.
- `cargo check --locked -p sage-upnp-av --target x86_64-apple-darwin` passed. Windows GNU cross-check could not complete because `x86_64-w64-mingw32-gcc` is not installed; the dependency `ring` needs that compiler in this environment.
- Formatting, `git diff --check`, repository architecture/credential checks, plist lint, and Swift parser checks passed. The full macOS package build remains blocked because this host has only Command Line Tools and cannot find the `SwiftUIMacros` plugin.

No multicast LAN scan or TV/device interaction was performed. The loopback integration fixture validates Sage's request/parse/fetch path, not interoperability with real renderer firmware. macOS installed-app behavior, local-network permission presentation, real renderer interoperability, pairing, Matter control, playback, and Windows runtime remain unqualified.
