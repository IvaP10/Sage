import copy
import importlib.util
import json
import math
import pathlib
import unittest
import sys
sys.dont_write_bytecode = True
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

gate = module("check-v2-release")
evaluation = module("evaluate-v2-report")

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "evals/release-gates.json").read_text())

    def test_pending_work_cannot_be_published(self):
        with patch.object(gate, "source_digest", return_value="a" * 64):
            self.assertTrue(gate.validate(self.manifest))
            with self.assertRaisesRegex(ValueError, "Production release is blocked"):
                gate.validate(self.manifest, require_ready=True)

    def test_a_required_gate_cannot_be_dropped_or_marked_passed_without_evidence(self):
        with patch.object(gate, "source_digest", return_value="a" * 64):
            missing = copy.deepcopy(self.manifest)
            missing["gates"].pop()
            with self.assertRaises(ValueError):
                gate.validate(missing)
            fake = copy.deepcopy(self.manifest)
            fake["gates"][0]["status"] = "passed"
            with self.assertRaises((ValueError, KeyError)):
                gate.validate(fake, require_ready=True)

    def test_evidence_cannot_escape_the_evidence_directory(self):
        for path in ["/tmp/proof", "evals/evidence/../../Cargo.toml", "Cargo.toml"]:
            with self.assertRaises(ValueError):
                gate.evidence_path(path)

    def test_report_cannot_omit_runs_or_treat_unavailable_work_as_success(self):
        suite = json.loads((ROOT / "evals/workflows.json").read_text())
        report = {"schema_version": 2, "suite_id": suite["suite_id"], "model_sha256": "a"*64,
                  "runtime_sha256": "b"*64, "source_digest": "c"*64, "fixtures_sha256": "d"*64,
                  "results": [{"workflow_id": case["id"], "repeat": repeat, "outcome": "unavailable",
                    "duration_ms": 0, "peak_sage_working_set_bytes": 0, "model_calls": 0,
                    "tokens_in": 0, "tokens_out": 0, "approval_count": 0, "energy_wh": 0}
                    for case in suite["cases"] for repeat in range(1, 6)]}
        summary = evaluation.evaluate(report, suite)
        self.assertEqual(summary["verified_success_percent"], 0)
        with self.assertRaises(ValueError):
            evaluation.evaluate(report, suite, enforce=True)
        for repeat in [True, 1.0, 0, 6]:
            report["results"][0]["repeat"] = repeat
            with self.assertRaisesRegex(ValueError, "integer repeat"):
                evaluation.evaluate(report, suite)
        report["results"][0]["repeat"] = 1
        report["results"].pop()
        with self.assertRaisesRegex(ValueError, "Every workflow"):
            evaluation.evaluate(report, suite)

    def test_nonfinite_or_boolean_measurements_are_rejected(self):
        for value in [None, True, math.nan, math.inf, -1]:
            with self.assertRaises(ValueError):
                evaluation.finite(value, "latency")

if __name__ == "__main__":
    unittest.main()
