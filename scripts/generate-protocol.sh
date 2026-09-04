#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
proto_file="$repository_root/proto/sage/ipc/v1/sage.proto"
swift_output="$repository_root/apps/macos/Sources/SageMac/Generated"

command -v protoc >/dev/null 2>&1 || {
  echo "protoc is required" >&2
  exit 1
}

if command -v protoc-gen-swift >/dev/null 2>&1; then
  swift_plugin=$(command -v protoc-gen-swift)
elif [ -n "${PROTOC_GEN_SWIFT:-}" ] && [ -x "$PROTOC_GEN_SWIFT" ]; then
  swift_plugin="$PROTOC_GEN_SWIFT"
else
  generated_file="$swift_output/sage/ipc/v1/sage.pb.swift"
  if [ -f "$generated_file" ]; then
    echo "Using checked-in Swift protobuf binding (set PROTOC_GEN_SWIFT to regenerate)."
    exit 0
  fi
  echo "protoc-gen-swift 1.38.1 is required to generate the Swift binding" >&2
  exit 1
fi

mkdir -p "$swift_output"
protoc \
  -I "$repository_root/proto" \
  --plugin="protoc-gen-swift=$swift_plugin" \
  --swift_out="Visibility=Internal:$swift_output" \
  "$proto_file"
