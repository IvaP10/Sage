#!/usr/bin/env python3
"""Fail closed on missing, stale or mismatched release evidence.

This validates recorded evidence; it does not replace the device, model,
penetration or signing checks that produce that evidence.
"""
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED = {
    "core_regressions": "automated_tests",
    "typed_authority": "security_review",
    "process_isolation": "security_review",
    "native_macos": "device_acceptance",
    "native_windows": "device_acceptance",
    "browser_live": "device_acceptance",
    "encrypted_migration": "fault_injection",
    "file_races": "fault_injection",
    "local_model_offline": "model_evaluation",
    "quantization_quality": "model_evaluation",
    "retrieval_100k": "performance",
    "vm_macos": "device_acceptance",
    "vm_windows": "device_acceptance",
    "voice_devices": "device_acceptance",
    "connectors_isolation": "security_review",
    "recovery_crashes": "fault_injection",
    "memory_forget": "automated_tests",
    "workflow_suite": "workflow_evaluation",
    "adversarial_evaluation": "security_evaluation",
    "performance_16gb": "performance",
    "signed_distribution": "distribution",
    "updates_tuf": "distribution",
    "supply_chain": "security_review",
}

def load(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value: {value}")))

def source_digest(root=ROOT):
    # Include tracked and new source files; omit build output, evidence and docs.
    names = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root).decode().split("\0")
    digest = hashlib.sha256()
    prefixes = ("crates/", "apps/", "proto/", "policies/", "integrations/", "scripts/", ".github/")
    files = {name for name in names if name and (name.startswith(prefixes) or name in {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "product.toml", "Makefile", "evals/workflows.json"})}
    for name in sorted(files):
        path = root / name
        digest.update(name.encode() + b"\0")
        if path.is_symlink():
            raise ValueError(f"Source symlink requires review: {name}")
        digest.update(hashlib.sha256(path.read_bytes()).digest() if path.is_file() else b"DELETED")
    return digest.hexdigest()

def evidence_path(value, root=ROOT):
    path = pathlib.Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[:2] != ("evals", "evidence"):
        raise ValueError("Evidence must be a repository-local file in evals/evidence")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to((root / "evals/evidence").resolve()):
        raise ValueError("Evidence escaped its directory")
    return resolved

def validate(manifest, require_ready=False, root=ROOT):
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported release gate schema")
    gates = manifest.get("gates", [])
    ids = [gate.get("id") for gate in gates]
    if len(ids) != len(set(ids)) or set(ids) != set(REQUIRED):
        raise ValueError("Required release gates were removed, duplicated or renamed")
    current = source_digest(root)
    pending = []
    for gate in gates:
        if gate.get("status") not in {"pending", "blocked", "passed"} or gate.get("proof_type") != REQUIRED[gate["id"]]:
            raise ValueError(f"Invalid gate contract: {gate['id']}")
        if gate["status"] != "passed":
            pending.append(gate["id"])
            if not gate.get("reason"):
                raise ValueError("Unfinished gates need a concrete reason")
            continue
        path = evidence_path(gate["evidence"], root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != gate.get("sha256"):
            raise ValueError(f"Evidence digest mismatch: {gate['id']}")
        proof = load(path)
        if proof.get("schema_version") != 2 or proof.get("outcome") != "passed" or proof.get("source_digest") != current or proof.get("proof_type") != REQUIRED[gate["id"]] or proof.get("gate") != gate["id"]:
            raise ValueError(f"Evidence does not establish this gate for current source: {gate['id']}")
        at = dt.datetime.fromisoformat(proof["recorded_at"].replace("Z", "+00:00"))
        now = dt.datetime.now(dt.timezone.utc)
        if at.tzinfo is None or not dt.timedelta(0) <= now - at <= dt.timedelta(days=30):
            raise ValueError(f"Evidence is stale or from the future: {gate['id']}")
        artifacts = proof.get("artifacts", [])
        if not artifacts:
            raise ValueError(f"Evidence has no underlying artifacts: {gate['id']}")
        for artifact in artifacts:
            data = evidence_path(artifact["path"], root).read_bytes()
            if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
                raise ValueError(f"Underlying artifact changed: {gate['id']}")
    if require_ready and pending:
        raise ValueError("Production release is blocked by: " + ", ".join(pending))
    return pending

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--source-digest", action="store_true")
    args = parser.parse_args()
    if args.source_digest:
        print(source_digest())
        return
    pending = validate(load(ROOT / "evals/release-gates.json"), args.require_ready)
    print(f"Release gate definitions valid; {len(pending)} qualification gates remain unfinished.")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
