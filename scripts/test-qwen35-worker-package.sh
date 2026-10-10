#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep the test trust root and resulting binaries isolated from normal builds.
export CARGO_TARGET_DIR="$repo_root/target/qwen35-package-verification-test"
export SAGE_QWEN35_PACKAGE_TRUSTED_KEYS="sage-test-2026=2e9761f2c9fa954b0a8757f55e4680196b8bf241f567bb5a9c68666f62e2640d"
export SAGE_INFERENCE_WORKER_EXECUTABLE="$CARGO_TARGET_DIR/debug/sage-inference-worker"

cd "$repo_root"
cargo build --offline --locked -p sage-inference-worker --features qwen35-model-generate
cargo test --offline --locked -p sage-core --lib \
  --features qwen35-worker-generate \
  core_verifies_a_signed_package_in_the_worker_and_retains_its_handles \
  -- --ignored --nocapture
