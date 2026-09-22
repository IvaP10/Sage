# Sage architecture

The current design is [Sage v2: a secure, efficient desktop agent](v2/architecture.md). It retains the SwiftUI/AppKit and WinUI clients and the Rust foundation while rebuilding the conversation, authority, model and feature boundaries.

Read the [current source map](current-architecture.md), [implementation status](v2/implementation-status.md), [threat model](v2/threat-model.md), and [migration/evaluation contract](v2/migration-and-evaluation.md) together. They distinguish implemented modules from planned process boundaries and measured acceptance from untested targets.

The execution flow is user scope → approved context → model answer or typed intent → prepared target/effect/verification contract → policy/approval → durable dispatch intent → single-use grant → execution → observation/verification → tool result → next turn or final answer.

Protocol v2 is canonical. Earlier generic host-command and coarse adapter-availability behavior has been retired; unsupported features are unavailable until their concrete contracts pass qualification. Production publication remains gated by `evals/release-gates.json`.
