# Streamed voice actions, 6 October 2026

The local Mac voice path accepts one complete app-opening action or one exact file read only after the transcript contains the explicit sequential boundary `and then`. App opening uses normal approval and native-verification gates. File reads require a selected read folder and recheck canonical path scope at preparation and admission. Both create normal receipted tasks, wait after the first verified action, and accept a local continuation on the same task without a planner call.

On Apple M3 / macOS 27.0.1, the release app benchmark warms native app identity and times engine admission to the `ApprovalRequested` event over 100 runs: p50 **1.519 ms**, p95 **1.803 ms**, p99 **2.335 ms**. The file benchmark times admission through the verified `ActionSucceeded` event for a 37-byte local fixture over 100 runs: p50 **2.954 ms**, p95 **3.796 ms**, p99 **5.874 ms**. These are core-path timings, not end-to-end latency; speech recognition, socket transport, Swift UI and user interaction are excluded. The app benchmark also excludes user approval and activation on the real host.

Evidence files:

- `workspace-tests.txt`: `cargo test --workspace --locked` — 187 core tests and one browser-worker test pass; five release-only tests are ignored by the normal run.
- `clippy.txt`: workspace/all-target Clippy with warnings denied.
- `macos-build.txt`: Swift package build using Command Line Tools SDK 26.5. The build passes with existing SDK search-path warnings.
- `macos-tests.txt`: `swift test` cannot resolve XCTest with the installed Command Line Tools SDK, so the native XCTest target could not run here.
- `streamed-prefix-approval-latency.txt`: `cargo test --release -p sage-core streamed_prefix_approval_latency -- --ignored --nocapture` output.
- `streamed-file-read-latency.txt`: `cargo test --release -p sage-core streamed_file_read_latency -- --ignored --nocapture` output.
- `diff-check.txt`: clean `git diff --check` output.

The tests use a fixture native adapter. They do not establish live microphone behavior, the interactive approval sheet, or real app activation. Windows does not opt into streamed execution.
