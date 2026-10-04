import importlib.util
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is unavailable")
class PhysicsTests(unittest.TestCase):
    def test_fixed_replay_and_minimal_repair(self):
        import numpy as np
        from nswm.physics import MuJoCoReplay, PositionAssertion
        from nswm.schema import Commitment, Edit, Interval, Query, Verdict
        from nswm.repair import search_repairs
        xml = Path(__file__).parents[1] / "examples/free_body.xml"
        engine = MuJoCoReplay(xml, [0, 0, 1, 1, 0, 0, 0], [0] * 6, [[]] * 4,
                              [PositionAssertion("x", "object_A", 4, 0)])
        query = Query("observed", ((),) * 4,
                      {"replay_identity": engine.identity, "complete_position_domain": True}, 4)
        trajectory = engine.replay().copy()
        np.testing.assert_array_equal(engine.replay(), trajectory)
        self.assertLess(trajectory[-1, 1, 2], trajectory[0, 1, 2])
        wrong = Commitment({"x": Interval(1, 2)})
        self.assertEqual(engine.oracle(query, wrong), Verdict.INVALID)
        edit = Edit("x", Interval("-0.01", "0.01"))
        result = search_repairs(query, wrong, (edit,), engine.oracle)
        self.assertEqual(result.repairs, ((edit,),))
        stale = Query("other", ((1,),) * 4, query.premises, 4)
        self.assertEqual(engine.oracle(stale, wrong), Verdict.UNKNOWN)
