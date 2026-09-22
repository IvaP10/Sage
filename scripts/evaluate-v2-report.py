#!/usr/bin/env python3
"""Validate and summarize measured workflow results; never synthesize results."""
import argparse
import hashlib
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Missing, negative or nonfinite measurement: {name}")
    return value

def p95(values):
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]

def evaluate(report, suite, enforce=False):
    if report.get("schema_version") != 2 or report.get("suite_id") != suite["suite_id"]:
        raise ValueError("Report does not identify this versioned workflow suite")
    cases = {case["id"] for case in suite["cases"]}
    if len(cases) != 100 or suite["repeats"] != 5:
        raise ValueError("The qualified suite requires 100 distinct workflows and five repeats")
    expected = {(case, repeat) for case in cases for repeat in range(1, 6)}
    results = report.get("results", [])
    keys = [(result["workflow_id"], result["repeat"]) for result in results]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Every workflow needs five results; missing, duplicated or unavailable work cannot disappear from the denominator")
    for field in ["model_sha256", "runtime_sha256", "source_digest", "fixtures_sha256"]:
        value = report.get(field, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing reproducibility identity: {field}")
    durations, peaks, confirmed = [], [], 0
    for result in results:
        if result.get("outcome") not in {"confirmed", "failed", "cancelled", "uncertain", "unavailable"}:
            raise ValueError("Invalid verification outcome")
        durations.append(finite(result.get("duration_ms"), "duration_ms"))
        peaks.append(finite(result.get("peak_sage_working_set_bytes"), "peak_sage_working_set_bytes"))
        for field in ["model_calls", "tokens_in", "tokens_out", "approval_count"]:
            value = finite(result.get(field), field)
            if not isinstance(value, int):
                raise ValueError(f"Counter must be an integer: {field}")
        finite(result.get("energy_wh"), "energy_wh")
        if result["outcome"] == "confirmed":
            artifact = result.get("verification_artifact", {})
            path = pathlib.Path(artifact.get("path", ""))
            if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("evals", "evidence"):
                raise ValueError("Confirmed results need scoped independent verification artifacts")
            resolved = (ROOT / path).resolve(strict=True)
            if not resolved.is_relative_to((ROOT / "evals/evidence").resolve()) or hashlib.sha256(resolved.read_bytes()).hexdigest() != artifact.get("sha256"):
                raise ValueError("Verification artifact digest/path mismatch")
            confirmed += 1
    summary = {"measured_runs": len(results), "verified_success_percent": confirmed * 100 / len(results),
               "task_duration_p95_ms": p95(durations), "peak_sage_working_set_bytes": max(peaks)}
    if enforce:
        reference = report.get("reference", {})
        if reference.get("ram_gib") != 16 or reference.get("platform") not in {"macos-apple-silicon", "windows-cpu"}:
            raise ValueError("This report is not from a required 16 GiB reference configuration")
        limits = {"idle_working_set_mib": 300, "ui_ack_p95_ms": 100,
                  "cancellation_ack_p95_ms": 250, "retrieval_100k_p95_ms": 250,
                  "warm_first_token_1k_p95_ms": 3000 if reference["platform"] == "macos-apple-silicon" else 8000}
        measured = report.get("measurements", {})
        failed = [name for name, limit in limits.items() if finite(measured.get(name), name) >= limit]
        if finite(measured.get("unauthorized_effects"), "unauthorized_effects") != 0:
            failed.append("unauthorized_effects")
        if finite(measured.get("quantization_loss_percentage_points"), "quantization_loss_percentage_points") > 1:
            failed.append("quantization_loss_percentage_points")
        if confirmed < 475:
            failed.append("verified_success_percent")
        if max(peaks) > 8 * 1024**3:
            failed.append("working_set_budget")
        if failed:
            raise ValueError("Measured release targets failed: " + ", ".join(failed))
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument("--enforce-targets", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(json.loads(args.report.read_text()), json.loads((ROOT / "evals/workflows.json").read_text()), args.enforce_targets), indent=2))

if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
