import unittest
from nswm.schema import Commitment, Interval, Query, Request
from nswm.verify import Check, verify
from nswm.primitives import conservation, persistence, precedence, reachable_box, sphere_penetration


class PrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.query = Query("h", ((0,),), {}, 16)

    def test_overlap_under_full_position_uncertainty(self):
        z = Commitment({f"{entity}.{axis}": Interval("-0.001", "0.001") for entity in "AB" for axis in "xyz"})
        b = sphere_penetration(self.query, z, "A", "B", Interval.point(0.1), Interval.point(0.1), premises_valid=True)
        self.assertGreater(b.residual.lower, 0)
        self.assertIsNone(sphere_penetration(self.query, z, "A", "B", Interval.point(0.1), Interval.point(0.1)).residual)

    def test_separated_and_uncertain_spheres(self):
        z = Commitment({f"{name}.{axis}": Interval.point(1 if name == "B" and axis == "x" else 0)
                        for name in "AB" for axis in "xyz"})
        def outcome(candidate):
            bound = sphere_penetration(self.query, candidate, "A", "B", Interval.point("0.1"),
                                       Interval.point("0.1"), premises_valid=True)
            result = verify(Request("sphere", self.query, candidate, candidate),
                            (Check("contact", "contact_solidity", ("A", "B"), (0, 1), lambda: bound),), None)
            return bound, result
        bound, result = outcome(z)
        self.assertLess(bound.residual.upper, 0)
        self.assertEqual(result.status, "N")
        uncertain = Commitment({**z.slots, "B.x": Interval("0.19", "0.21")})
        bound, result = outcome(uncertain)
        self.assertLessEqual(bound.residual.lower, 0)
        self.assertGreater(bound.residual.upper, 0)
        self.assertEqual(result.status, "X")
        overlap = Commitment({**z.slots, "B.x": Interval.point("0.1")})
        self.assertEqual(outcome(overlap)[1].status, "V")

    def test_exchange_uncertainty_prevents_exclusion(self):
        z = Commitment({"momentum": Interval.point(2)})
        impossible = conservation(self.query, z, "momentum", Interval.point(0), Interval(0, 1), premises_valid=True)
        ambiguous = conservation(self.query, z, "momentum", Interval.point(0), Interval(0, 3), premises_valid=True)
        self.assertGreater(impossible.residual.lower, 0)
        self.assertLessEqual(ambiguous.residual.lower, 0)

    def test_reachability_and_identity(self):
        z = Commitment({"x": Interval(3, 4), "identity_count": Interval.point(0)})
        self.assertGreater(reachable_box(self.query, z, {"x": Interval(0, 1)}, premises_valid=True).residual.lower, 0)
        self.assertGreater(persistence(self.query, z, "identity_count", premises_valid=True).residual.lower, 0)

    def test_precedence_deadline_and_positive_cycle(self):
        z = Commitment({"a": Interval(0, 1), "b": Interval(0, 1)})
        bad = precedence(self.query, z, [("a", "b", Interval.point(2))], True)
        legal = precedence(self.query, z, [("a", "b", Interval.point("0.5"))], True)
        cycle = precedence(self.query, z, [("a", "b", Interval.point("0.1")), ("b", "a", Interval.point("0.1"))], True)
        self.assertGreater(bad.residual.lower, 0)
        self.assertEqual(legal.residual.lower, 0)
        self.assertGreater(cycle.residual.lower, 0)
