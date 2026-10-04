from dataclasses import asdict
import importlib.util
from pathlib import Path
import tempfile
import unittest

from nswm.annotation import annotate_event, check_repair
from nswm.evaluation import evaluate_events
from nswm.planning import BackendFailure, Cost, Future, Prediction, run_episode
from nswm.schema import Certificate, Commitment, Edit, Interval, Query, Request, Verdict
from nswm.verify import scalar_inventory, scalar_oracle


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.query = Query("h", ((0.0,),), {"complete_scalar_domain": True}, 1)
        self.z = Commitment({"x": Interval.point(2)})
        self.request = Request("r", self.query, self.z, self.z)

    def episode(self, generator, unit="work_units"):
        return run_episode(0, lambda h: ([self.request], Cost(unit=unit)), None,
            lambda r: (), generator, lambda r, f: (1, True, Cost(unit=unit)),
            lambda h, a, f: (1, Cost(unit=unit)), lambda h: "success" if h else "running",
            policy="full", budget=3, unit=unit)

    def test_failed_work_is_charged_and_unknown_cost_is_incomplete(self):
        def failed(request):
            raise BackendFailure("denoiser stopped", Cost(1.2, 0.2, 3), job_id="job")
        episode = self.episode(failed)
        self.assertEqual(episode.status, "timeout")
        self.assertEqual(episode.ledger.compute, 1.2)
        self.assertEqual(episode.ledger.nfe, 3)
        self.assertTrue(episode.ledger.accounting_complete)
        def unmeasured(request):
            raise BackendFailure("process lost", wall_seconds=0.1, job_id="job")
        episode = self.episode(unmeasured)
        self.assertEqual(episode.status, "backend_error")
        self.assertFalse(episode.ledger.accounting_complete)

    def test_process_identity_error_is_recorded(self):
        import sys
        from nswm.backends import JsonProcessGenerator
        backend = JsonProcessGenerator([sys.executable, "-c", 'print("{}")'], unit="work_units")
        episode = self.episode(backend)
        self.assertEqual(episode.status, "backend_error")
        self.assertFalse(episode.ledger.entries[-1]["compute_known"])

    def test_actual_output_acceptance_and_failed_future(self):
        wrong = Commitment({"x": Interval.point(0)})
        episode = self.episode(lambda r: Future("j", r.binding, wrong, None, Cost(1)))
        self.assertEqual(episode.status, "failure")
        episode = self.episode(lambda r: Future("j", r.binding, None, None, Cost(1), False))
        self.assertEqual(episode.status, "timeout")

    def test_annotation_does_not_copy_critic_proof_into_gate(self):
        rules = [{"id": "x-bound", "slot": "x", "limit": "cap", "constraint": "reachability", "premises_valid": True}]
        measurement = {"cap": Interval.point(1)}
        oracle = lambda q, z: scalar_oracle(q, z, measurement, rules)
        checks = scalar_inventory(self.query, self.z, measurement, rules)
        event = {"id": "r", "query": asdict(self.query), "commitment": asdict(self.z)}
        edit = Edit("x", Interval.point(0))
        annotated = annotate_event(event, oracle, (edit,), critic_checks=checks)
        self.assertEqual(annotated["targets"]["critic"]["judgment"], "invalid")
        self.assertEqual(annotated["targets"]["gate"]["judgment"], "unknown")
        self.assertEqual(annotated["annotation"]["critic"]["repair_status"], "verified_minimal")
        self.assertEqual(check_repair(self.query, self.z, (edit,), oracle), "verified_minimal")
        scoped = annotate_event(event, oracle, (edit,), gate_checks=checks)
        self.assertEqual(scoped["targets"]["gate"]["judgment"], "invalid")
        self.assertIsNone(scoped["targets"]["gate"]["repair"])

    def test_evaluation_preserves_family_units(self):
        from nswm.data import build_fixtures, read_events
        class Predictor:
            def predict_event(self, row, mode):
                return Prediction(Certificate(Verdict(row["targets"][mode]["judgment"])), 0, Cost())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = read_events(build_fixtures(root / "data", 5))
            for row in rows:
                row["split"] = "test"
            summary = evaluate_events(rows, Predictor(), root / "eval")
            self.assertEqual(summary["gate"]["balanced_accuracy"], 1)
            self.assertEqual(summary["critic"]["families"], 5)
            self.assertEqual(len((root / "eval/predictions.jsonl").read_text().splitlines()), 50)
            with self.assertRaises(FileExistsError):
                evaluate_events(rows, Predictor(), root / "eval")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class VisionContractTests(unittest.TestCase):
    def test_segmentation_rejects_future_before_initialization(self):
        from nswm.vision import SAM2HistorySegmenter
        class Predictor:
            def init_state(self, **kw):
                raise AssertionError("future reached predictor")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "00000.jpg").touch()
            with self.assertRaises(ValueError):
                SAM2HistorySegmenter(Predictor()).segment(root, [1.0], [{"entity": "A"}])

    def test_pose_source_requires_metric_depth_and_joint_calibration(self):
        import numpy as np
        from nswm.vision import FoundationPoseHistoryEstimator, calibrated_pose_box
        class Estimator:
            def register(self, **kw):
                return np.eye(4)
            def track_one(self, **kw):
                value = np.eye(4)
                value[0, 3] = 0.1
                return value
        rgb = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        depth = np.ones((2, 4, 4))
        masks = np.ones_like(depth, dtype=bool)
        wrapper = FoundationPoseHistoryEstimator(Estimator())
        poses = wrapper.estimate(rgb, depth, np.eye(3), masks, [-1, 0])
        q = Query("h", ((0,),), {}, 1)
        pose = calibrated_pose_box("A", 0, poses[-1], np.eye(4), None, q)
        self.assertIsNone(pose.position)
        pose = calibrated_pose_box("A", 0, poses[-1], np.eye(4), 0.01, q)
        self.assertTrue(pose.position[0].contains(Interval.point("0.1")))
        with self.assertRaises(ValueError):
            wrapper.estimate(rgb, depth * 0, np.eye(3), masks, [-1, 0])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class TrackerContractTests(unittest.TestCase):
    def test_online_tracker_consumes_only_the_observed_prefix(self):
        import numpy as np
        import torch
        from nswm.vision import CoTracker3HistoryTracker
        class Predictor:
            step = 2
            def __call__(self, video_chunk, is_first_step, **kw):
                if is_first_step:
                    self.total = video_chunk.shape[1]
                    return None, None
                return torch.zeros(1, self.total, 1, 2), torch.ones(1, self.total, 1, dtype=torch.bool)
        tracker = CoTracker3HistoryTracker(Predictor(), "cpu")
        tracks, visibility = tracker.track(np.zeros((5, 8, 8, 3)), [-4, -3, -2, -1, 0], [[0, 2, 2]])
        self.assertEqual(tracks.shape, (5, 1, 2))
        self.assertTrue(visibility.all())
        with self.assertRaises(ValueError):
            tracker.track(np.zeros((5, 8, 8, 3)), [-4, -3, -2, -1, 1], [[0, 2, 2]])
