#!/usr/bin/env python3
"""Compile read-only signature checks with production sources, without XCTest."""
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    if sys.platform != "darwin":
        raise RuntimeError("This check needs a macOS device")
    package = str(ROOT / "apps/macos")
    subprocess.run(["swift", "build", "--package-path", package], check=True, cwd=ROOT)
    build = pathlib.Path(subprocess.check_output(["swift", "build", "--package-path", package, "--show-bin-path"], text=True, cwd=ROOT).strip())
    # SwiftBuild and the older native builder lay out dependency artifacts
    # differently. Compile the exercised production sources directly so
    # unrelated app entry points, UI and optional voice libraries are not linked.
    modules = build if (build / "SwiftProtobuf.swiftmodule").exists() else build / "Modules"
    objects = [build / "SwiftProtobuf.o"] if (build / "SwiftProtobuf.o").exists() else list((build / "SwiftProtobuf.build").glob("*.o"))
    if not objects or not (modules / "SwiftProtobuf.swiftmodule").exists():
        raise RuntimeError("SwiftProtobuf debug artifacts are unavailable for native checks")
    sources = [ROOT / "apps/macos/Sources/SageMac" / path for path in (
        "Generated/sage/ipc/v2/sage.pb.swift", "MachOSlices.swift",
        "SignedApplicationIdentity.swift", "PlatformAdapter.swift", "SageClientError.swift",
    )]
    with tempfile.TemporaryDirectory(prefix="sage-signature-check-") as temporary:
        executable = str(pathlib.Path(temporary) / "signature-check")
        subprocess.run(["swiftc", "-parse-as-library", "-I", str(modules),
                        str(ROOT / "scripts/macos-signature-smoke.swift"), *map(str, sources), *map(str, sorted(objects)),
                        "-o", executable], check=True, cwd=ROOT)
        subprocess.run([executable], check=True, cwd=ROOT)

if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
