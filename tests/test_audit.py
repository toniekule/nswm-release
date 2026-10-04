import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from nswm.data import read_events
from nswm.evaluation import evaluate_events, report
from nswm.planning import Cost, Prediction
from nswm.schema import Certificate, Query, Verdict, commitment_from_dict
from nswm.targets import parse_certificate, serialize_target
from nswm.train import media_identity, read_config

HEAVY = all(importlib.util.find_spec(p) for p in ("torch", "transformers", "mujoco"))


def write_marker(path):
    Path(path).write_text("deserialized")
    return None


class AuditCoreTests(unittest.TestCase):
    def test_isolation_groups_duplicate_clips_and_detects_population_copy(self):
        from nswm.isolation import audit_isolation, group_splits
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frame.png").write_bytes(b"same observation")
            rows = [{"id": family + "/legal", "family_id": family, "split": split,
                     "observations": {mode: [{"path": "frame.png"}] for mode in ("gate", "critic")}}
                    for family, split in (("a", "train"), ("b", "test"))]
            self.assertFalse(audit_isolation([("id", rows, root)])["isolated"])
            groups = group_splits(rows, root, 0, 0.8, 0.1)
            self.assertEqual(groups["groups"], 1)
            self.assertEqual(rows[0]["split"], rows[1]["split"])
            self.assertTrue(audit_isolation([("id", rows, root)])["isolated"])
            copied = {**rows[0], "id": "ood/legal", "family_id": "ood"}
            failed = audit_isolation([("id", rows, root), ("ood", [copied], root)])
            self.assertFalse(failed["isolated"])
            self.assertEqual(failed["family_leaks"], {})
            self.assertTrue(failed["complete_clip_leaks"])

    def test_minimal_alias_rejected_at_every_entry(self):
        from nswm.encoding import encode_event
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"events": "e", "model": "m", "processor": "p", "output": "o", "supervision": "minimal"}))
            with self.assertRaisesRegex(ValueError, "not a distinct"):
                read_config(path)
            with self.assertRaises(ValueError):
                serialize_target({"judgment": "valid"}, "minimal")
            with self.assertRaisesRegex(ValueError, "not a distinct"):
                encode_event({}, "gate", tmp, None, {"supervision": "minimal"})

    def test_media_escape_rejected_before_hashing_or_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "manifest"
            root.mkdir()
            outside = Path(tmp) / "outside.png"
            outside.write_bytes(b"outside")
            (root / "linked.png").symlink_to(outside)
            for name in ("../outside.png", str(outside), "linked.png"):
                row = {"split": "test", "observations": {mode: [{"path": name, "time": 0}] for mode in ("gate", "critic")}}
                with patch("nswm.train.sha256", side_effect=AssertionError("must not hash")):
                    with self.assertRaisesRegex(ValueError, "leaves manifest"):
                        media_identity([row], root / "events.jsonl")
                    with self.assertRaisesRegex(ValueError, "leaves manifest"):
                        evaluate_events([row], None, root / "evaluation", root=root)
                self.assertFalse((root / "evaluation").exists())


@unittest.skipUnless(HEAVY, "train and physics extras required")
class AuditPhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nswm.sdg import build_corpus
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        checkout = Path(__file__).resolve().parents[1]
        cls.populations = build_corpus(checkout / "configs/scenes.json", cls.root / "corpus", 20, 0,
                                      checkout / "configs/splits.json")
        cls.manifest = Path(cls.populations["eval-ood"]["manifest"])
        cls.rows = read_events(cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_history_and_future_sources_are_independent(self):
        from nswm.evidence import FutureOracle, HistoryOracle, load_future
        from nswm.sdg import load_context
        row = next(r for r in self.rows if r["category"] == "contact_solidity" and r["variant"] == "legal")
        context = load_context(row, self.manifest.parent)
        query, z = Query(**row["query"]), commitment_from_dict(row["commitment"])
        future = load_future(row, self.manifest.parent, context)
        gate, critic = HistoryOracle(query, row["category"]), FutureOracle(context, z, future)
        self.assertEqual(gate(query, z), Verdict.VALID)
        self.assertEqual(critic(query, z), Verdict.VALID)
        changed = copy.deepcopy(future)
        changed["positions"][-1][0] = changed["positions"][-1][1]
        changed_critic = FutureOracle(context, z, changed)
        self.assertEqual(changed_critic(query, z), Verdict.INVALID)
        self.assertIsNone(changed_critic.checks(query, z)[0].evaluate().residual)
        self.assertLess(gate.checks(query, z)[0].evaluate().residual.upper, 0)
        self.assertEqual(gate(query, z), Verdict.VALID)
        missing = json.loads(json.dumps(asdict(query)))
        missing["premises"].pop("radii")
        partial = Query(**missing)
        partial_gate = HistoryOracle(partial, row["category"])
        self.assertEqual(partial_gate(partial, z), Verdict.UNKNOWN)
        self.assertFalse(partial_gate.checks(partial, z)[0].evaluate().premises_valid)
        rebound = {**context, "query_binding": partial.binding}
        self.assertEqual(FutureOracle(rebound, z, future)(partial, z), Verdict.VALID)
        self.assertEqual(row["annotation"]["gate"]["label_source"], "permitted_history")
        self.assertEqual(row["annotation"]["critic"]["label_source"], "measured_candidate_future")
        self.assertNotEqual(row["annotation"]["gate"]["source_binding"], row["annotation"]["critic"]["source_binding"])
        self.assertNotIn("positions", gate.context)
        self.assertNotIn("legal_commitment", gate.context)
        changed_path = self.manifest.parent / row["observations"]["critic"][-1]["path"]
        from nswm.train import sha256
        with patch("nswm.train.sha256", side_effect=lambda path: "0" * 64 if Path(path) == changed_path else sha256(path)):
            with self.assertRaisesRegex(ValueError, "rendered media"):
                load_future(row, self.manifest.parent, context)
        tampered = copy.deepcopy(row)
        tampered["construction"]["future_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest differs"):
            load_future(tampered, self.manifest.parent, context)

    def test_model_metrics_and_inventory_diagnostics_are_separated(self):
        class Predictor:
            def __init__(self, correct):
                self.correct = correct
            def predict_event(self, row, mode):
                certificate = (parse_certificate(json.dumps(row["targets"][mode])) if self.correct else
                               Certificate(Verdict.VALID if row["targets"][mode]["judgment"] == "invalid" else Verdict.INVALID))
                return Prediction(certificate, 0.5, Cost())
        paths = []
        for correct, arm in ((False, "wrong"), (True, "correct")):
            out = self.root / arm
            evaluate_events(self.rows, Predictor(correct), out, root=self.manifest.parent, seed=0, arm=arm)
            paths.append(out / "predictions.jsonl")
        a, b = report(paths[0]), report(paths[1])
        self.assertEqual(a["groups"]["wrong/gate"]["overall"]["balanced_accuracy"], 0)
        self.assertEqual(b["groups"]["correct/gate"]["overall"]["balanced_accuracy"], 1)
        self.assertEqual(a["inventory_diagnostics"], b["inventory_diagnostics"])
        combined = report(paths, compare=("wrong", "correct"))
        self.assertEqual(combined["inventory_diagnostics"]["groups"]["gate"]["overall"]["events"], 100)
        self.assertTrue(combined["inventory_diagnostics"]["arm_invariant"])
        for group in combined["groups"].values():
            self.assertTrue({"FR", "MR", "resolution"}.isdisjoint(group["overall"]))
        modified = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
        modified[-1]["assessment"]["cancelled"] = not modified[-1]["assessment"]["cancelled"]
        bad = self.root / "inconsistent.jsonl"
        bad.write_text("".join(json.dumps(r) + "\n" for r in modified))
        with self.assertRaisesRegex(ValueError, "diagnostics disagree"):
            report(bad)

    def test_repaired_variant_records_applied_parent_edits(self):
        for row in self.rows:
            if row["variant"] != "repaired":
                continue
            from nswm.schema import Edit, Interval
            parent = row["construction"]["repair_parent"]
            edits = tuple(Edit(e["slot"], Interval(**e["value"])) for e in parent["applied_edits"])
            repaired = commitment_from_dict(parent["commitment"]).edited(edits)
            self.assertEqual(repaired, commitment_from_dict(row["commitment"]))
            self.assertTrue(edits)
            self.assertEqual(row["targets"]["gate"]["judgment"], "valid")

    def test_unversioned_physical_manifest_has_no_new_assessment_provenance(self):
        from nswm.assessment import assess_prediction
        row = copy.deepcopy(self.rows[0])
        row["construction"].pop("evidence_version")
        result = assess_prediction(row, "gate", Certificate(), self.manifest.parent)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("no separate history and future sources", result["reason"])
        class Predictor:
            def predict_event(self, row, mode):
                return Prediction(Certificate(), 0.5, Cost())
        out = self.root / "unversioned_evaluation"
        with patch("nswm.sdg.load_context", side_effect=AssertionError("unversioned context must not acquire current provenance")):
            evaluate_events([row], Predictor(), out, root=self.manifest.parent)
        summary = report(out / "predictions.jsonl")
        self.assertEqual(summary["inventory_diagnostics"]["groups"], {})
        self.assertIsNone(summary["groups"]["model/gate"]["overall"]["J_cert"])

    def test_checkpoint_payload_cannot_execute_before_identity_checks(self):
        import torch
        from nswm.predict import SharedPredictor
        from nswm.tiny import make_tiny
        from nswm.train import run_training
        root = self.root / "unsafe_checkpoint"
        make_tiny(root)
        config = json.loads((root / "train.json").read_text())
        marker = root / "marker"
        class Payload:
            def __reduce__(self):
                return write_marker, (str(marker),)
        checkpoint = root / "unsafe.pt"
        torch.save({"config": config, "payload": Payload()}, checkpoint)
        for load in (lambda: SharedPredictor(checkpoint), lambda: run_training(config, resume=checkpoint)):
            with self.assertRaises(pickle.UnpicklingError):
                load()
            self.assertFalse(marker.exists())
        self.assertFalse(Path(config["output"]).exists())
