# One-shot current-reference context, 6 October 2026

This checkpoint adds a deterministic, one-shot path for a foreground task that explicitly names its current selection, page, file, window or application. It is not a general background observer. The raw snapshot is transient, untrusted planner context; only a consumed marker is durable. A task restart does not silently inspect a new foreground item.

The Rust path adds no model, inference API, downloaded weights, provider or crate dependency. It can place the explicitly requested snapshot into the existing planner turn. Sage's existing external data-release gate presents the request and exact selected context before an external destination can receive it. The current reference does not grant a capability or authorize a tool action.

The bridge benchmark uses in-process fake adapters and 1,000 samples. Its measured overhead is p50 **22,417 ns**, p95 **26,459 ns** and p99 **31,458 ns**. These numbers exclude sockets, live native/browser response time, planner work, UI rendering and end-to-end latency.

Evidence files:

- `manifest.json`: validation summary and hashes for the source snapshot.
- `workspace-tests.txt`: locked Rust workspace tests, including current-reference and recovery fixtures.
- `browser-tests.txt`: browser extension test suite, including paired-tab identity and bounded/private-filtered capture.
- `clippy.txt`: workspace/all-target Clippy with warnings denied.
- `format-check.txt` and `diff-check.txt`: Rust formatting and whitespace validation.
- `macos-build.txt`: Swift package build using the installed Command Line Tools macOS 26.5 SDK. `macos-sdk27-default.txt` records the failed default SDK 27 selection because its SwiftUI macro plugin is unavailable in this Command Line Tools installation.
- `adapter-bridge-latency.txt`: optimized in-process bridge benchmark with fake native and browser adapters.
- `windows-validation.txt`: .NET availability and source-level validation boundary; Windows compilation and live UI remain unverified on this host.
- `macos-sdk27-default.txt`: default SDK selection failure and successful 26.5 override.

No live selection, browser page or Windows interaction was exercised for this checkpoint. The native reference adapters still depend on user-granted Accessibility access and an installed, authenticated paired browser extension.
