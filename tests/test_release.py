import copy
import importlib.util
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import unittest

from nswm.data import FAMILIES, read_events
from nswm.evaluation import evaluate_events, report
from nswm.planning import Cost, Prediction
from nswm.schema import Certificate, Verdict
from nswm.targets import parse_certificate

HEAVY = all(importlib.util.find_spec(p) for p in ("torch", "transformers", "mujoco"))


class ReportTests(unittest.TestCase):
    def readings(self):
        rows = []
        for arm in ("a", "b"):
            for seed in range(3):
                for family in range(2):
                    for truth in ("valid", "invalid"):
                        rows.append({"id": f"{family}/{truth}", "family_id": str(family), "category": "c",
                            "mode": "gate", "query_binding": str(family), "truth": truth,
                            "prediction": truth if arm == "a" else "unknown", "arm": arm, "seed": seed,
                            "cost": {"compute": 1, "wall_seconds": 2, "unit": "cpu_seconds"}})
        return rows

    def test_paired_seed_intervals_and_unavailable_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in self.readings()))
            result = report(path, compare=("a", "b"), repeats=100)
            self.assertEqual(result["paired"]["gate"]["metrics"]["balanced_accuracy"]["mean"], 1)
            self.assertIsNone(result["groups"]["a/gate"]["overall"]["J_cert"])
            self.assertEqual(result["paired"]["gate"]["unit"], "paired_training_seed")
            rows = self.readings()[:-1]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError):
                report(path, compare=("a", "b"), repeats=100)

    def test_duplicate_readings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            row = json.dumps(self.readings()[0]) + "\n"
            path.write_text(row * 2)
            with self.assertRaises(ValueError):
                report(path)


@unittest.skipUnless(HEAVY, "train and physics extras required")
class ReleasePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from nswm.tiny import make_tiny
        from nswm.sdg import build_corpus
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.checkout = Path(__file__).resolve().parents[1]
        make_tiny(cls.root / "tiny")
        cls.populations = build_corpus(cls.checkout / "configs/scenes.json", cls.root / "corpus", 20, 0,
                                      cls.checkout / "configs/splits.json")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        from nswm.train import base_identity, processor_identity
        self.asset_snapshot = (base_identity(self.root / "tiny/model"), processor_identity(self.root / "tiny/processor"))

    def tearDown(self):
        from nswm.train import base_identity, processor_identity
        current = (base_identity(self.root / "tiny/model"), processor_identity(self.root / "tiny/processor"))
        self.assertEqual(current, self.asset_snapshot)

    def test_corpus_constraints_replay_repairs_and_no_leakage(self):
        from nswm.annotation import check_repair
        from nswm.data import model_input
        from nswm.sdg import load_context, ReplayOracle
        from nswm.schema import Edit, Interval, Query, commitment_from_dict
        manifest = Path(self.populations["eval-ood"]["manifest"])
        rows = read_events(manifest)
        id_manifest = Path(self.populations["corpus-train"]["manifest"])
        id_rows = read_events(id_manifest)
        self.assertTrue({r["family_id"] for r in rows}.isdisjoint(r["family_id"] for r in id_rows))
        def clips(population, root, mode):
            return {hashlib.sha256(b"".join(bytes.fromhex(hashlib.sha256((root / f["path"]).read_bytes()).hexdigest())
                    for f in row["observations"][mode])).hexdigest() for row in population}
        for mode in ("gate", "critic"):
            self.assertTrue(clips(rows, manifest.parent, mode).isdisjoint(clips(id_rows, id_manifest.parent, mode)))
            for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
                a, b = [r for r in id_rows if r["split"] == left], [r for r in id_rows if r["split"] == right]
                self.assertTrue({r["family_id"] for r in a}.isdisjoint(r["family_id"] for r in b))
                self.assertTrue(clips(a, id_manifest.parent, mode).isdisjoint(clips(b, id_manifest.parent, mode)))
        self.assertEqual(len(rows), 100)
        self.assertEqual({r["ood_axis"] for r in rows}, {"geometry", "dynamics", "appearance", "composition"})
        self.assertEqual({r["split"] for r in rows}, {"test"})
        for row in rows:
            if row["variant"] != "violating":
                continue
            context = load_context(row, manifest.parent)
            oracle = ReplayOracle(context)
            z = commitment_from_dict(row["commitment"])
            query = Query(**row["query"])
            self.assertEqual(oracle(query, z), Verdict.INVALID)
            repair = tuple(Edit(e["slot"], Interval(**e["value"])) for e in row["targets"]["gate"]["repair"])
            self.assertEqual(check_repair(query, z, repair, oracle), "verified_minimal")
            self.assertLessEqual(len(repair), 6)
            payload, paths = model_input(row, "gate", manifest.parent)
            self.assertNotIn("targets", payload)
            self.assertNotIn("positions", payload)
            self.assertEqual(len(paths), 5)
        card = json.loads((manifest.parent / "data_card.json").read_text())
        self.assertEqual(card["category_quotas"], {c: 4 for c in FAMILIES})
        self.assertEqual(card["axis_quotas"], {c: 5 for c in ("geometry", "dynamics", "appearance", "composition")})
        row = rows[1]
        changed = copy.deepcopy(row)
        changed["query"]["premises"]["gravity"] = [0, 0, -1]
        with self.assertRaises(ValueError):
            load_context(changed, manifest.parent)

    def test_physical_quality_is_recomputed(self):
        manifest = Path(self.populations["eval-ood"]["manifest"])
        class Predictor:
            def predict_event(self, row, mode):
                return Prediction(parse_certificate(json.dumps(row["targets"][mode])), 0, Cost())
        out = self.root / "truth_contract_evaluation"
        evaluate_events(read_events(manifest), Predictor(), out, root=manifest.parent, seed=0, arm="contract")
        result = report(out / "predictions.jsonl")
        gate = result["groups"]["contract/gate"]["overall"]
        self.assertEqual(gate["J_cert"], 1)
        diagnostics = result["inventory_diagnostics"]["groups"]["gate"]["overall"]
        self.assertEqual(diagnostics["FR"], 0)
        self.assertEqual(diagnostics["MR"], 0)
        self.assertEqual(diagnostics["resolution"], 1)
        self.assertNotIn("FR", gate)
        from nswm.assessment import assess_prediction
        row = next(r for r in read_events(manifest) if r["variant"] == "violating")
        bad = copy.deepcopy(row["targets"]["gate"])
        bad["witness"]["residual"] = {"lo": "999", "hi": "999"}
        checks = assess_prediction(row, "gate", parse_certificate(json.dumps(bad)), manifest.parent)
        self.assertFalse(checks["witness_valid"])

    def test_real_cli_multimodal_training_best_and_resume(self):
        import torch
        config = json.loads((self.root / "tiny/train.json").read_text())
        config["events"] = self.populations["corpus-train"]["manifest"]
        config["output"] = str(self.root / "cli_training")
        config["max_output_tokens"] = 8192
        config["max_text_tokens"] = 16384
        path = self.root / "train.json"
        path.write_text(json.dumps(config))
        run = subprocess.run([sys.executable, "-m", "nswm.cli", "train", str(path)], text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        checkpoint = Path(config["output"]) / "latest.pt"
        self.assertTrue(checkpoint.with_name("best.pt").is_file())
        first = torch.load(checkpoint, weights_only=True)
        self.assertEqual(first["next_step"], 2)
        config["max_steps"] = 4
        path.write_text(json.dumps(config))
        run = subprocess.run([sys.executable, "-m", "nswm.cli", "train", str(path), "--resume", str(checkpoint)],
                             text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        second = torch.load(checkpoint, weights_only=True)
        self.assertEqual(second["next_step"], 4)
        self.assertTrue(any(not torch.equal(first["adapter"][k], second["adapter"][k]) for k in first["adapter"]))
        from nswm.predict import SharedPredictor
        predictor = SharedPredictor(checkpoint, root=Path(config["events"]).parent)
        prediction = predictor.predict_event(read_events(config["events"])[0], "gate", "judgment")
        self.assertGreater(prediction.cost.compute, 0)
        evaluation = self.root / "cli_evaluation"
        run = subprocess.run([sys.executable, "-m", "nswm.cli", "evaluate", config["events"], str(checkpoint),
                             str(evaluation), "--split", "dev", "--modes", "gate", "--arm", "tiny"],
                             text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        run = subprocess.run([sys.executable, "-m", "nswm.cli", "report", str(evaluation / "predictions.jsonl"),
                              "--out", str(self.root / "report.json")], text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads((self.root / "report.json").read_text())
        self.assertGreater(result["groups"]["tiny/gate"]["overall"]["assessed_events"], 0)
        from nswm.resources import inspect_resources
        self.assertTrue(inspect_resources(config)["ready"])

    def test_streaming_conversion_requires_every_tensor(self):
        from nswm.assets import convert_base
        from safetensors import safe_open
        from safetensors.torch import save_file
        source = self.root / "synthetic_mot"
        (source / "transformer").mkdir(parents=True)
        (source / "vision_encoder").mkdir()
        text, vision, expected = {}, {}, {}
        with safe_open(str(self.root / "tiny/model/model.safetensors"), framework="pt") as stream:
            for key in stream.keys():
                expected[key] = stream.get_tensor(key)
                if key.startswith("model.visual."):
                    vision[key[len("model.visual."):]] = expected[key]
                elif key.startswith("model.language_model."):
                    text[key[len("model.language_model."):]] = expected[key]
                else:
                    text[key] = expected[key]
        save_file(text, source / "transformer/weights.safetensors")
        save_file(vision, source / "vision_encoder/weights.safetensors")
        processor = self.root / "conversion_processor"
        shutil.copytree(self.root / "tiny/processor", processor)
        processor.joinpath("config.json").write_bytes((self.root / "tiny/model/config.json").read_bytes())
        destination = self.root / "converted"
        result = convert_base(source, processor, destination, 1)
        self.assertEqual(result["mapped_tensors"], len(expected))
        index = json.loads((destination / "model.safetensors.index.json").read_text())["weight_map"]
        import torch
        for key, shard in index.items():
            with safe_open(str(destination / shard), framework="pt") as stream:
                torch.testing.assert_close(stream.get_tensor(key), expected[key], rtol=0, atol=0)
        text.pop(next(iter(text)))
        save_file(text, source / "transformer/weights.safetensors")
        with self.assertRaises(ValueError):
            convert_base(source, processor, self.root / "incomplete", 1)

    def test_builder_seed_reproduces_artifacts(self):
        from nswm.sdg import build_corpus
        import hashlib
        a, b = self.root / "repeat_a", self.root / "repeat_b"
        for path in (a, b):
            build_corpus(self.checkout / "configs/scenes.json", path, 5, 17)
        files_a = {str(p.relative_to(a)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in a.rglob("*") if p.is_file()}
        files_b = {str(p.relative_to(b)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in b.rglob("*") if p.is_file()}
        self.assertEqual(files_a, files_b)

    def test_split_spec_reproduces_every_artifact(self):
        from nswm.sdg import build_corpus
        a, b = self.root / "split_repeat_a", self.root / "split_repeat_b"
        for path in (a, b):
            build_corpus(self.checkout / "configs/scenes.json", path, 20, 17, self.checkout / "configs/splits.json")
        files_a = {str(p.relative_to(a)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in a.rglob("*") if p.is_file()}
        files_b = {str(p.relative_to(b)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in b.rglob("*") if p.is_file()}
        self.assertEqual(files_a, files_b)
        self.assertEqual({p.parts[0] for p in map(Path, files_a)}, {"corpus-train", "eval-ood", "isolation.json"})

    def test_tiny_assets_are_seed_reproducible(self):
        from nswm.tiny import make_tiny
        from nswm.train import base_identity, processor_identity
        path = self.root / "repeat_tiny"
        make_tiny(path)
        self.assertEqual(base_identity(path / "model"), base_identity(self.root / "tiny/model"))
        left = processor_identity(path / "processor")
        right = processor_identity(self.root / "tiny/processor")
        self.assertEqual(left, right)
