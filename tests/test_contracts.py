import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nswm.data import FAMILIES
from nswm.repair import search_repairs
from nswm.schema import Commitment, Edit, Interval, Query, Verdict, fingerprint


CHECKOUT = Path(__file__).resolve().parents[1]


class PlanContractTests(unittest.TestCase):
    def test_population_subsets_keep_families_and_require_declared_proofs(self):
        from nswm.data import select_population
        rows = []
        for family, available in (("two", True), ("six", False)):
            for label in ("valid", "invalid"):
                rows.append({"family_id": family, "targets": {m: {"judgment": label} for m in ("gate", "critic")},
                    "population_membership": {m: {"certificate": label == "invalid", "four_target": label == "invalid" and available}
                                              for m in ("gate", "critic")}})
        self.assertEqual(len(select_population(rows, "certificate")), 4)
        selected = select_population(rows, "four_target")
        self.assertEqual(len(selected), 2)
        self.assertEqual({r["family_id"] for r in selected}, {"two"})
        self.assertEqual(select_population([{k: v for k, v in r.items() if k != "population_membership"} for r in rows], "certificate"), [])
        partial = copy.deepcopy(rows[:2])
        partial[1]["population_membership"]["critic"]["four_target"] = False
        self.assertEqual(select_population(partial, "four_target"), [])
        self.assertEqual(len(select_population(partial, "four_target", ("gate",))), 2)

    def test_complete_quotas_and_split_plan_are_deterministic_without_engine(self):
        from nswm.corpus_v3 import corpus_plan
        from nswm.scenes import read_spec
        spec = read_spec(CHECKOUT / "configs/mechanisms.json")
        split = json.loads((CHECKOUT / "configs/splits.json").read_text())
        with patch("nswm.mechanisms.simulate", side_effect=AssertionError("plan must not simulate")):
            a, b = corpus_plan(spec, 240, 19, split), corpus_plan(spec, 240, 19, split)
        self.assertEqual(a, b)
        roots = []
        for name, declarations in a["populations"].items():
            axes = {"id"} if name == "corpus-train" else {"geometry", "dynamics", "appearance", "composition"}
            self.assertEqual({r["scene"]["shift"] for r in declarations}, axes)
            for axis in axes:
                cells = [(r["scene"]["category"], r["requested_cardinality"], r["boundary_sampling"])
                         for r in declarations if r["scene"]["shift"] == axis]
                self.assertEqual(set(cells), {(c, k, level) for c in FAMILIES for k in range(1, 7) for level in ("regular", "near")})
            if name == "eval-ood":
                self.assertEqual({r["split"] for r in declarations}, {"test"})
            roots.append({r["root_family_id"] for r in declarations})
        self.assertFalse(roots[0] & roots[1])
        for families in (0, 50, 60, 241):
            with self.assertRaises(ValueError):
                corpus_plan(spec, families, 19, split)

    def test_minimum_cardinality_requires_all_smaller_catalogue_choices(self):
        query = Query("q", ((0,),), {}, 1)
        z = Commitment({k: Interval.point(0) for k in "xyz"})
        edits = tuple(Edit(k, Interval.point(1)) for k in "xyz")
        def oracle(q, candidate):
            selected = {k for k, value in candidate.slots.items() if value.lower == 1}
            if selected == {"z"}:
                return Verdict.UNKNOWN
            return Verdict.VALID if selected == {"x", "y"} else Verdict.INVALID
        result = search_repairs(query, z, edits, oracle)
        self.assertEqual(result.status, "verified_minimal")
        self.assertTrue(result.exhaustive)
        self.assertIsNone(result.minimum_cardinality)
        self.assertIn(1, result.unresolved_cardinalities)
        one = search_repairs(query, z, edits, lambda q, c: Verdict.VALID if c.slots["x"].lower == 1 else Verdict.INVALID, budget=2)
        self.assertFalse(one.exhaustive)
        self.assertEqual(one.minimum_cardinality, 1)

    def test_family_repair_budget_and_source_cache_are_explicit(self):
        from nswm.annotation import annotate_event
        query = Query("q", ((0,),), {}, 1)
        negative = Commitment({"x": Interval.point(2)})
        event = {"id": "q/violating", "query": asdict(query), "commitment": asdict(negative)}
        calls = []
        def oracle(q, c):
            calls.append(c)
            return Verdict.VALID if c.slots["x"].lower == 0 else Verdict.INVALID
        cache = {}
        args = dict(gate_oracle=oracle, repair_cache=cache, oracle_bindings={"gate": "history", "critic": "future"})
        first = annotate_event(event, oracle, (Edit("x", Interval.point(0)),), **args)
        count = len(calls)
        second = annotate_event(event, oracle, (Edit("x", Interval.point(0)),), **args)
        self.assertEqual(first, second)
        self.assertEqual(len(calls) - count, 2)
        self.assertEqual(len(cache), 2)
        with self.assertRaises(ValueError):
            annotate_event(event, oracle, (), repair_cache={}, gate_oracle=oracle)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "physics extra required")
class MechanismTests(unittest.TestCase):
    def test_evidence_assembly_and_parent_audit_with_fixture_media(self):
        from nswm.corpus_v3 import build_family, audit_families, load_evidence
        from nswm.mechanisms import MechanismScene
        from nswm.scenes import read_spec
        spec = read_spec(CHECKOUT / "configs/mechanisms.json")
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / name for name in ("one", "two")]
            manifests = []
            for root in roots:
                root.mkdir()
                for style, rgb in ((0, b"\x10\x20\x30"), (1, b"\x30\x20\x10")):
                    (root / f"fixture-{style}.ppm").write_bytes(b"P6\n1 1\n255\n" + rgb)
                def fixture_media(scene, trace, destination, width, height, style, backend):
                    return [root / f"fixture-{style}.ppm"] * len(trace["positions"])
                declaration = {"scene": asdict(MechanismScene("assembly", "contact_solidity", 0, lanes=2)),
                               "split": "train", "boundary_sampling": "regular"}
                with patch("nswm.render.render_mechanism", side_effect=fixture_media):
                    rows = build_family(root, declaration, spec, "software")
                self.assertEqual(len(rows), 6)
                self.assertEqual(audit_families(rows, root)["appearance_pairs"], 3)
                for row in rows:
                    load_evidence(row, root)
                    self.assertLessEqual(row["construction"]["family_repair_queries"], 4096)
                manifests.append(rows)
                broken = copy.deepcopy(rows)
                repaired = next(r for r in broken if r["variant"] == "repaired")
                repaired["construction"]["repair_parent"]["commitment"] = rows[0]["commitment"]
                with self.assertRaisesRegex(ValueError, "parent"):
                    audit_families(broken, root)
                appearance = next(r for r in rows if r["variant"] == "appearance_violating")
                altered = copy.deepcopy(appearance)
                altered["construction"]["future_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "digest"):
                    load_evidence(altered, root)
                incomplete = [r for r in rows if r["variant"] != "appearance_legal"]
                with self.assertRaisesRegex(ValueError, "appearance"):
                    audit_families(incomplete, root)
            self.assertEqual(manifests[0], manifests[1])
            files = [{str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()} for root in roots]
            self.assertEqual(files[0], files[1])

    def test_five_real_mechanisms_have_independent_two_edit_repairs(self):
        from nswm.corpus_v3 import prepare_family
        from nswm.mechanisms import MechanismScene
        from nswm.counterfactual import MechanismFutureOracle, candidate_trace, physical_identity
        from nswm.annotation import check_repair
        from nswm.verify import verify
        from nswm.schema import Request
        for category in FAMILIES:
            with self.subTest(category=category):
                scene = MechanismScene("test-" + category, category, 0, lanes=2)
                context, plan = prepare_family(scene)
                query = Query(**context["query"])
                self.assertEqual(plan["difficulty"], 2)
                self.assertEqual(plan["search"].minimum_cardinality, 2)
                negative = next(v for v in plan["variants"] if v.name == "violating")
                repaired = next(v for v in plan["variants"] if v.name == "repaired")
                self.assertEqual(negative.commitment.edited(repaired.applied_edits), repaired.commitment)
                future = candidate_trace(context, negative.commitment)
                critic = MechanismFutureOracle(context, negative.commitment, future)
                self.assertEqual(critic(query, negative.commitment), Verdict.INVALID)
                self.assertEqual(critic(query, repaired.commitment), Verdict.VALID)
                self.assertEqual(check_repair(query, negative.commitment, repaired.applied_edits, critic), "verified_minimal")
                self.assertEqual(physical_identity(candidate_trace(context, repaired.commitment)), physical_identity(context["trace"]))
                request = Request("r", query, negative.commitment, negative.commitment)
                for oracle in (plan["gate_oracle"], critic):
                    self.assertTrue(verify(request, oracle.checks(query, negative.commitment), None).cancelled)
                if category in {"contact_solidity", "conservation", "causal_ordering"}:
                    self.assertTrue(context["trace"]["events"])
                    self.assertTrue(all(e["normal_force"] > 0 for e in context["trace"]["events"]))
                if category == "permanence":
                    self.assertTrue(all(all(row) for row in context["trace"]["exists"]))
                    self.assertFalse(all(all(row) for row in context["trace"]["visible"]))

    def test_six_edit_proof_covers_every_category_without_training(self):
        from nswm.mechanisms import MechanismScene
        from nswm.corpus_v3 import prepare_family
        for category in FAMILIES:
            with self.subTest(category=category):
                context, plan = prepare_family(MechanismScene("six-" + category, category, 1, lanes=6))
                self.assertEqual(plan["difficulty"], 6)
                self.assertEqual(plan["difficulty_bin"], "3+")
                self.assertTrue(plan["search"].exhaustive)
                self.assertLessEqual(plan["search"].queries * 2, 4096)
                self.assertEqual(len(plan["search"].repairs), 64)

    def test_future_mutation_does_not_change_history_truth(self):
        from nswm.mechanisms import MechanismScene
        from nswm.corpus_v3 import prepare_family
        from nswm.counterfactual import MechanismHistoryOracle, MechanismFutureOracle, candidate_trace
        context, plan = prepare_family(MechanismScene("independent", "conservation", 0))
        query = Query(**context["query"])
        legal = next(v.commitment for v in plan["variants"] if v.name == "legal")
        future = candidate_trace(context, legal)
        future["velocities"][-1][0][0] += 10
        self.assertEqual(MechanismFutureOracle(context, legal, future)(query, legal), Verdict.INVALID)
        self.assertEqual(plan["gate_oracle"](query, legal), Verdict.VALID)
        premises = json.loads(json.dumps(asdict(query)))["premises"]
        premises.pop("initial_qvel")
        missing = Query(query.history_id, query.actions, premises, query.horizon)
        self.assertEqual(MechanismHistoryOracle(missing)(missing, legal), Verdict.UNKNOWN)

    def test_appearance_pairs_preserve_states_and_visibility_on_both_labels(self):
        from nswm.mechanisms import MechanismScene
        from nswm.corpus_v3 import prepare_family
        from nswm.counterfactual import candidate_trace
        context, plan = prepare_family(MechanismScene("paired", "permanence", 0))
        roles = {v.name: v for v in plan["variants"]}
        for name in ("legal", "violating", "repaired"):
            base, appearance = roles[name], roles["appearance_" + name]
            self.assertNotEqual(base.style, appearance.style)
            self.assertEqual(base.commitment, appearance.commitment)
            self.assertEqual(candidate_trace(context, base.commitment), candidate_trace(context, appearance.commitment))

    def test_near_boundary_changes_the_physical_context(self):
        from nswm.mechanisms import MechanismScene
        from nswm.corpus_v3 import prepare_family
        for category in ("contact_solidity", "conservation", "causal_ordering", "resource_reachability"):
            regular, near = (MechanismScene("margin", category, 0, drive_fraction=0.65,
                                            impact_fraction=2 if category == "contact_solidity" else 0),
                             MechanismScene("margin", category, 0, drive_fraction=0.995,
                                            gap=0.8 / 120 if category == "causal_ordering" else 0.12))
            a, pa = prepare_family(regular)
            b, pb = prepare_family(near)
            self.assertNotEqual(a["trace"], b["trace"])
            self.assertLess(min(pb["boundary"]["margins"].values()), min(pa["boundary"]["margins"].values()))
        _, discrete = prepare_family(MechanismScene("discrete", "permanence", 0))
        self.assertFalse(discrete["boundary"]["continuous"])

    def test_near_causal_edit_changes_the_visible_state_ledger(self):
        from nswm.mechanisms import MechanismScene
        from nswm.corpus_v3 import prepare_family
        from nswm.counterfactual import candidate_trace
        scene = MechanismScene("early", "causal_ordering", 0, gap=0.8 / 120)
        context, plan = prepare_family(scene)
        negative = next(v.commitment for v in plan["variants"] if v.name == "violating")
        self.assertNotEqual(candidate_trace(context, negative)["positions"], context["trace"]["positions"])

    def test_composition_shift_changes_actual_topology(self):
        from nswm.mechanisms import MechanismScene
        from nswm.corpus_v3 import prepare_family
        for category in FAMILIES:
            a, _ = prepare_family(MechanismScene("base", category, 0))
            b, _ = prepare_family(MechanismScene("composed", category, 0, shift="composition"))
            self.assertGreater(len(b["trace"]["names"]), len(a["trace"]["names"]))
        self.assertGreater(len(b["initial_qpos"]), len(a["initial_qpos"]))

    def test_replay_includes_solver_state_and_is_reproducible(self):
        from nswm.mechanisms import MechanismScene
        from nswm.counterfactual import reference_context
        scene = MechanismScene("state", "causal_ordering", 0)
        a, b = reference_context(scene), reference_context(scene)
        self.assertEqual(fingerprint(a), fingerprint(b))
        self.assertEqual(len(a["trace"]["integration_states"]), 17)
        self.assertGreater(len(a["trace"]["integration_states"][0]), len(a["initial_qpos"]))


class ExposureTests(unittest.TestCase):
    def test_evaluation_reuses_identical_predictions_and_charges_work_once(self):
        from nswm.evaluation import evaluate_events
        from nswm.planning import Cost, Prediction
        from nswm.schema import Certificate
        class Predictor:
            calls = 0
            def predict_event(self, row, mode):
                self.calls += 1
                return Prediction(Certificate(Verdict.VALID), 0.2, Cost(2, 1))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.rows(root)[0]
            clone = copy.deepcopy(original)
            original["split"], clone["split"], clone["id"] = "test", "test", "clone"
            predictor = Predictor()
            summary = evaluate_events([original, clone], predictor, root / "evaluation", modes=("gate",), root=root)
            self.assertEqual(predictor.calls, 1)
            self.assertEqual(summary["gate"]["compute"], 2)
            records = [json.loads(line) for line in (root / "evaluation/predictions.jsonl").read_text().splitlines()]
            self.assertEqual([r["prediction_reused"] for r in records], [False, True])
            self.assertEqual(records[0]["input_identity"], records[1]["input_identity"])

    def rows(self, root):
        (root / "a.ppm").write_bytes(b"P6\n1 1\n255\n\x10\x20\x30")
        (root / "b.ppm").write_bytes(b"P6\n1 1\n255\n\x30\x20\x10")
        result = []
        for category, families in (("contact_solidity", ("a", "b")), ("conservation", ("c",))):
            for family in families:
                for label, views in (("valid", 3), ("invalid", 1)):
                    for view in range(views):
                        query = Query(family, ((0,),), {"family": family}, 1)
                        result.append({"id": f"{family}/{label}/{view}", "family_id": family, "category": category,
                            "query": asdict(query), "commitment": asdict(Commitment({"x": Interval.point(1 if label == "valid" else 2)})),
                            "observations": {mode: [{"path": "a.ppm" if view % 2 == 0 else "b.ppm", "time": 0}] for mode in ("gate", "critic")},
                            "targets": {mode: {"judgment": label} for mode in ("gate", "critic")}})
        return result

    def test_weight_totals_balance_categories_families_and_labels_after_dedup(self):
        from nswm.weighting import weighted_events
        from collections import defaultdict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.rows(root)
            weighted = weighted_events(rows, "gate", root)
            self.assertEqual(len(weighted), 9)
            self.assertAlmostEqual(sum(w for _, w in weighted), 1)
            categories, families, labels = defaultdict(float), defaultdict(float), defaultdict(float)
            for row, weight in weighted:
                categories[row["category"]] += weight
                families[row["family_id"]] += weight
                labels[row["family_id"], row["targets"]["gate"]["judgment"]] += weight
            self.assertEqual(dict(categories), {"contact_solidity": 0.5, "conservation": 0.5})
            self.assertEqual(dict(families), {"a": 0.25, "b": 0.25, "c": 0.5})
            for family in families:
                self.assertEqual(labels[family, "valid"], labels[family, "invalid"])

    def test_identity_uses_text_pixels_timestamps_and_preprocessing(self):
        from nswm.weighting import input_identity, weighted_events
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.rows(root)[0]
            same = copy.deepcopy(original)
            (root / "copy.ppm").write_bytes((root / "a.ppm").read_bytes())
            same["observations"]["gate"][0]["path"] = "copy.ppm"
            self.assertEqual(input_identity(original, "gate", root), input_identity(same, "gate", root))
            same["commitment"] = asdict(Commitment({"x": Interval.point(3)}))
            self.assertNotEqual(input_identity(original, "gate", root), input_identity(same, "gate", root))
            self.assertNotEqual(input_identity(original, "gate", root, {"longest_side": 448}),
                                input_identity(original, "gate", root, {"longest_side": 224}))
            bad = copy.deepcopy(original)
            bad["id"], bad["targets"]["gate"]["judgment"] = "opposite", "invalid"
            with self.assertRaisesRegex(ValueError, "incompatible"):
                weighted_events([original, bad], "gate", root)
            other = copy.deepcopy(original)
            other["id"], other["family_id"] = "other", "other"
            with self.assertRaisesRegex(ValueError, "root families"):
                weighted_events([original, other], "gate", root)

    def test_redundancy_unavailability_is_rejected(self):
        from nswm.weighting import weighted_events
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.rows(root)[0]
            row["redundant_available"] = {"gate": False}
            with self.assertRaisesRegex(ValueError, "six edits"):
                weighted_events([row], "gate", root, {"supervision": "redundant"})

    def test_reports_deduplicate_inputs_and_average_labels_within_family(self):
        from nswm.evaluation import grouped_mean
        from nswm.metrics import balanced_accuracy
        rows = [{"id": str(i), "category": "contact_solidity", "family_id": "f", "truth": label,
                 "prediction": "valid", "input_identity": label + str(i % 2)}
                for i, label in enumerate(("valid", "valid", "valid", "invalid"))]
        self.assertEqual(balanced_accuracy(rows), 0.5)
        self.assertEqual(grouped_mean(rows, lambda r: r["prediction"] == r["truth"]), 0.5)
        inconsistent = copy.deepcopy(rows[0])
        inconsistent["prediction"] = "invalid"
        with self.assertRaises(ValueError):
            balanced_accuracy([*rows, inconsistent])


class RobotAssetTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "physics extra required")
    def test_robot_position_task_checks_goals_and_fixed_context(self):
        import numpy as np
        import mujoco
        from nswm.robots import RobotPositionTask, PositionGoal
        from nswm.physics import PositionAssertion
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "model.xml"
            xml.write_text((CHECKOUT / "examples/free_body.xml").read_text())
            model = mujoco.MjModel.from_xml_path(str(xml))
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1)
            task = RobotPositionTask({"xml": str(xml), "identity": "fixture"}, model.qpos0, np.zeros(model.nv),
                np.zeros((4, model.nu)), (PositionAssertion("pose.x", body, 4, 0),),
                (PositionGoal(body, 4, 0, Interval("-10", "10")),))
            reference = task.reference()
            self.assertEqual(task.oracle(task.query, reference), Verdict.VALID)
            changed = Query(task.query.history_id, task.query.actions, {**task.query.premises, "asset_identity": "other"}, 4)
            self.assertEqual(task.oracle(changed, reference), Verdict.UNKNOWN)
            failed = RobotPositionTask({"xml": str(xml), "identity": "fixture"}, model.qpos0, np.zeros(model.nv),
                np.zeros((4, model.nu)), (PositionAssertion("pose.x", body, 4, 0),),
                (PositionGoal(body, 4, 0, Interval("100", "101")),))
            self.assertEqual(failed.oracle(failed.query, failed.reference()), Verdict.INVALID)

    def test_local_robot_asset_receipt_and_xml_confinement(self):
        from nswm.robots import register_robot_assets, inspect_robot_assets
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            (root / "scene.xml").write_text('<mujoco><worldbody/></mujoco>')
            (root / "LICENSE").write_text("Test fixture license")
            licenses = [{"model": "fixture", "spdx": "CC0-1.0", "file": "LICENSE"}]
            receipt = Path(tmp) / "receipt.json"
            registered = register_robot_assets(root, "scene.xml", "https://example.org/fixture", "a" * 40, licenses, receipt)
            self.assertEqual(inspect_robot_assets(root, receipt)["identity"], registered["identity"])
            (root / "LICENSE").write_text("changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                inspect_robot_assets(root, receipt)
            (root / "scene.xml").write_text('<mujoco><include file="../outside.xml"/></mujoco>')
            other = Path(tmp) / "bad.json"
            with self.assertRaisesRegex(ValueError, "external"):
                register_robot_assets(root, "scene.xml", "fixture", "a" * 40, licenses, other)
            self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
