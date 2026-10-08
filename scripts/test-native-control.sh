#!/bin/sh
set -eu
repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/sage-native-control.XXXXXX")
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
swiftc -swift-version 6 -parse-as-library \
  "$repository_root/apps/macos/Sources/SageMac/OrderedFrameWriter.swift" \
  "$repository_root/apps/macos/Sources/SageMac/PendingSubmissions.swift" \
  "$repository_root/scripts/ipc-writer-smoke.swift" -o "$scratch/writer"
"$scratch/writer"
swiftc -swift-version 6 -parse-as-library \
  "$repository_root/apps/macos/Sources/SageMac/AdapterOperations.swift" \
  "$repository_root/scripts/adapter-cancellation-smoke.swift" -o "$scratch/cancellation"
"$scratch/cancellation"
swiftc -swift-version 6 -parse-as-library \
  "$repository_root/apps/macos/Sources/SageMac/PresentationRefreshCoordinator.swift" \
  "$repository_root/scripts/presentation-refresh-smoke.swift" -o "$scratch/presentation"
"$scratch/presentation"
