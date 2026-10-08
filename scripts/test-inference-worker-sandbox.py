#!/usr/bin/env python3
"""Exercise the packaged inference helper's macOS App Sandbox boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import socket
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "target" / "release" / "sage-inference-worker"
ENTITLEMENTS = ROOT / "apps" / "macos" / "SageInferenceWorker.entitlements"
INFO_PLIST = ROOT / "apps" / "macos" / "SageInferenceWorker-Info.plist"
EXPECTED = {
    "read_path": "denied",
    "write_path": "denied",
    "network": "denied",
    "inherited_fd": "readable",
}


def run() -> None:
    if os.uname().sysname != "Darwin":
        raise SystemExit("This sandbox check must run on macOS")

    build = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "-p",
            "sage-inference-worker",
            "--bin",
            "sage-inference-worker",
            "--features",
            "sandbox-probe",
        ],
        cwd=ROOT,
        check=False,
    )
    if build.returncode:
        raise SystemExit(build.returncode)

    with tempfile.TemporaryDirectory(prefix="sage-inference-sandbox-") as directory:
        root = Path(directory)
        app = root / "SageInferenceWorker.app"
        contents = app / "Contents"
        executable_dir = contents / "MacOS"
        executable_dir.mkdir(parents=True)
        helper_binary = executable_dir / "sage-inference-worker"
        helper_binary.write_bytes(WORKER.read_bytes())
        helper_binary.chmod(0o755)
        (contents / "Info.plist").write_bytes(INFO_PLIST.read_bytes())

        subprocess.run(
            ["codesign", "--force", "--sign", "-", "--entitlements", str(ENTITLEMENTS), str(app)],
            check=True,
        )
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)

        challenge = "0123456789abcdef" * 4
        request = json.dumps(
            {"protocol_version": 1, "kind": "hello", "challenge": challenge},
            separators=(",", ":"),
        ).encode("utf-8")
        handshake = subprocess.run(
            [str(helper_binary)],
            input=struct.pack(">I", len(request)) + request,
            check=False,
            capture_output=True,
            timeout=10,
        )
        if handshake.returncode != 0 or len(handshake.stdout) < 4:
            raise SystemExit(
                f"FAIL: worker handshake failed: {handshake.returncode}; {handshake.stderr!r}"
            )
        frame_size = struct.unpack(">I", handshake.stdout[:4])[0]
        if frame_size != len(handshake.stdout) - 4:
            raise SystemExit("FAIL: worker handshake returned a malformed frame")
        response = json.loads(handshake.stdout[4:])
        if (
            response.get("protocol_version") != 1
            or response.get("kind") != "hello"
            or response.get("challenge") != challenge
            or response.get("readiness") != "model_not_admitted"
            or not isinstance(response.get("process_id"), int)
            or response["process_id"] <= 0
        ):
            raise SystemExit(f"FAIL: worker handshake response was invalid: {response!r}")

        supervisor_environment = os.environ.copy()
        supervisor_environment["SAGE_TEST_INFERENCE_WORKER_EXECUTABLE"] = str(helper_binary)
        supervised = subprocess.run(
            [
                "cargo",
                "test",
                "--offline",
                "-p",
                "sage-core",
                "--bin",
                "sage-core",
                "inference_worker_process::tests::launches_the_configured_worker_and_completes_the_bounded_handshake",
                "--",
                "--ignored",
                "--exact",
                "--test-threads=1",
            ],
            cwd=ROOT,
            env=supervisor_environment,
            check=False,
        )
        if supervised.returncode:
            raise SystemExit(supervised.returncode)

        protected_file = root / "host-private.txt"
        protected_file.write_text("host-private-data\n", encoding="utf-8")
        denied_write = root / "worker-write.txt"
        inherited_file = root / "inherited-data.txt"
        inherited_file.write_text("sage-inherited-read-only-descriptor\n", encoding="utf-8")
        inherited_fd = os.open(inherited_file, os.O_RDONLY)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                listener.settimeout(0.8)
                address = f"{listener.getsockname()[0]}:{listener.getsockname()[1]}"
                result = subprocess.run(
                    [
                        str(helper_binary),
                        "--sandbox-probe",
                        "--read-path",
                        str(protected_file),
                        "--write-path",
                        str(denied_write),
                        "--connect",
                        address,
                        "--read-fd",
                        str(inherited_fd),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    pass_fds=(inherited_fd,),
                )
                try:
                    accepted, _ = listener.accept()
                except TimeoutError:
                    accepted = None
                if accepted is not None:
                    accepted.close()
                    raise SystemExit("FAIL: sandboxed worker connected to the local listener")
        finally:
            os.close(inherited_fd)

        if result.returncode != 0:
            raise SystemExit(
                "FAIL: sandbox probe exited "
                f"{result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        try:
            observed = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise SystemExit(f"FAIL: invalid probe response: {result.stdout!r}") from error
        if observed != EXPECTED:
            raise SystemExit(f"FAIL: unexpected sandbox probe response: {observed!r}")
        if denied_write.exists():
            raise SystemExit("FAIL: sandboxed worker created its denied output file")

    print("PASS: versioned worker handshake; App Sandbox denied host-file reads, writes, and network; inherited read-only descriptor remained usable")


if __name__ == "__main__":
    run()
