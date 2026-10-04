import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from nswm.data import build_fixtures, read_events, model_input, split_family
from nswm.demo import planning_demo
from nswm.metrics import balanced_accuracy, paired_interval, rejection_metrics
from nswm.perception import joint_error_radius, pose_box
from nswm.repair import search_repairs
from nswm.schema import Certificate, Commitment, Edit, Interval, Query, Request, Verdict, scope_contains
from nswm.verify import Bound, Check, scalar_inventory, verify


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.query = Query("h", ((0.0,),), {"capacity": "1"}, 16)
        self.z = Commitment({"x": Interval.point(2)})
        self.request = Request("r", self.query, self.z, self.z)

    def test_immutable_identity(self):
        with self.assertRaises(TypeError):
            self.query.premises["capacity"] = "9"
        with self.assertRaises(TypeError):
            self.z.slots["x"] = Interval.point(0)
        self.assertNotEqual(self.query.binding, Query("h2", ((0,),), {"capacity": "1"}, 16).binding)

    def test_interval_containment_missing_dimension(self):
        self.assertFalse(scope_contains(self.z, Commitment({"y": Interval.point(2)})))
        self.assertTrue(scope_contains(Commitment({"x": Interval(1, 3)}), self.z))
        a = Interval("1.00000000000000000000000000000000000000000000000000001", "2")
        result = a + Interval.point("0.00000000000000000000000000000000000000000000000000001")
        self.assertLessEqual(result.lower, a.lower)

    def test_prediction_cannot_authorize_exclusion(self):
        prediction = Certificate(Verdict.INVALID, constraint="contact")
        self.assertEqual(verify(self.request, (), prediction).status, "N")
        unknown = Check("missing", "contact", (), (0, 1), lambda: Bound(None, "m", False, None, self.query.binding))
        self.assertFalse(verify(self.request, (unknown,), prediction).cancelled)

    def test_stale_proof_and_partial_scope(self):
        stale = Check("stale", "contact", (), (0, 1), lambda: Bound(Interval.point(1), "m", True, self.z, "wrong"))
        self.assertEqual(verify(self.request, (stale,), None).status, "X")
        narrow = Commitment({"x": Interval.point("2.1")})
        partial = Check("partial", "contact", (), (0, 1), lambda: Bound(Interval.point(1), "m", True, narrow, self.query.binding))
        self.assertEqual(verify(self.request, (partial,), None).status, "S")

    def test_ordering_budget_and_unknown(self):
        safe = Check("safe", "other", (), (0, 1), lambda: Bound(Interval.point(-1), "m", True, self.z, self.query.binding))
        bad = Check("bad", "contact", (), (0, 1), lambda: Bound(Interval.point(1), "m", True, self.z, self.query.binding))
        self.assertEqual(verify(self.request, (safe, bad), None, 1).status, "X")
        self.assertEqual(verify(self.request, (safe, bad), Certificate(constraint="contact"), 1).status, "V")

    def test_nonmonotone_minimality(self):
        z = Commitment({k: Interval.point(0) for k in "abc"})
        edits = tuple(Edit(k, Interval.point(1)) for k in "abc")
        def oracle(q, commitment):
            n = sum(v.lower == 1 for v in commitment.slots.values())
            return Verdict.VALID if n in {1, 3} else Verdict.INVALID
        result = search_repairs(self.query, z, edits, oracle)
        self.assertEqual([len(r) for r in result.repairs], [1, 1, 1])
        self.assertEqual(result.queries, 8)

    def test_many_alternative_edits_do_not_enumerate_same_slot_sets(self):
        edits = tuple(Edit("x", Interval.point(i)) for i in range(200))
        result = search_repairs(self.query, self.z, edits,
                                lambda q, z: Verdict.VALID if z.slots["x"].lower == 0 else Verdict.INVALID,
                                budget=300)
        self.assertEqual(result.repairs, ((edits[0],),))
        self.assertTrue(result.exhaustive)

    def test_malformed_prediction_is_unknown(self):
        from nswm.targets import parse_certificate
        for text in ['{"judgment":"invalid","time":[0]}',
                     '{"judgment":"invalid","entities":[["A"]]}',
                     '{"judgment":"invalid","time":[0,NaN]}',
                     '{"judgment":"invalid","scope":[]}']:
            self.assertEqual(parse_certificate(text).judgment, Verdict.UNKNOWN)

    def test_unknown_subset_and_budget(self):
        z = Commitment({k: Interval.point(0) for k in "ab"})
        edits = tuple(Edit(k, Interval.point(1)) for k in "ab")
        def oracle(q, commitment):
            n = sum(v.lower == 1 for v in commitment.slots.values())
            return [Verdict.INVALID, Verdict.UNKNOWN, Verdict.VALID][n]
        result = search_repairs(self.query, z, edits, oracle)
        self.assertEqual(result.repairs, ())
        self.assertEqual(result.status, "unknown")
        self.assertFalse(search_repairs(self.query, z, edits, oracle, budget=1).exhaustive)

    def test_family_split_and_future_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixtures"
            manifest = build_fixtures(root, 5)
            rows = read_events(manifest)
            self.assertEqual(len(rows), 25)
            for row in rows:
                payload, paths = model_input(row, "gate", root)
                self.assertNotIn("targets", payload)
                self.assertNotIn("measurements", payload)
                self.assertTrue(all(t <= 0 for t in payload["timestamps"]))
                self.assertFalse(set(paths) & {str((root/f["path"]).resolve()) for f in row["observations"]["critic"]})
            damaged = dict(rows[0])
            damaged["observations"] = {**rows[0]["observations"], "gate": [{"path": rows[0]["observations"]["gate"][0]["path"], "time": 1}]}
            from nswm.data import validate_event
            with self.assertRaises(ValueError):
                validate_event(damaged, root)

    def test_planning_cancels_before_dispatch_and_preserves_residual(self):
        cert = planning_demo()
        full = planning_demo("full")
        self.assertEqual(cert["status"], "success")
        generation = [e for e in cert["ledger"]["entries"] if e["stage"] == "generation"]
        self.assertTrue(all("crossing" not in e["request"] for e in generation))
        self.assertTrue(any("residual" in e["request"] for e in generation))
        self.assertLess(cert["ledger"]["nfe"], full["ledger"]["nfe"])
        self.assertEqual(planning_demo(budget=0)["status"], "timeout")

    def test_conformal_insufficient_calibration_is_unknown(self):
        self.assertIsNone(joint_error_radius([[0.1]]))
        self.assertEqual(joint_error_radius([[i/1000] for i in range(100)]), 0.099)
        self.assertIsNone(pose_box("A", 0, [1, 2, 3], None, self.query).position)

    def test_grouped_metrics_and_paired_units(self):
        rows = [{"category": "contact", "family_id": "a", "truth": label, "prediction": label} for label in ["valid", "invalid"]]
        self.assertEqual(balanced_accuracy(rows * 5), 1)
        interval = paired_interval([(0.6, 0.4), (0.7, 0.5)], repeats=100)
        self.assertAlmostEqual(interval["mean"], 0.2)


if __name__ == "__main__":
    unittest.main()
