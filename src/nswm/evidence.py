"""History-only Gate evidence and measured candidate-future Critic evidence."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .schema import Commitment, Interval, Verdict, commitment_from_dict, fingerprint
from .verify import Bound, Check


class HistoryOracle:
    def __init__(self, query, category):
        self.binding = query.binding
        self.context = None
        self.reference = None
        try:
            k = query.premises
            state = k["initial_state"]
            hz = k["control_hz"]
            if (k["motion_model"] != "constant_velocity" or k["category"] != category
                    or tuple(k["gravity"]) != (0, 0, 0) or any(k["external_impulse"])
                    or any(query.actions) or hz <= 0):
                return
            if any(not math.isfinite(float(v)) for body in state.values()
                   for vector in (body["position"], body["velocity"]) for v in vector):
                return
            from .sdg import around
            duration = query.horizon / hz
            slots = {"A.initial.x": Interval.point(state["A"]["position"][0])}
            context = {"scene": {"category": category}, "query_binding": query.binding}
            if category == "contact_solidity":
                if k["radii"]["A"] != k["radii"]["B"]:
                    return
                context["scene"]["radius"] = k["radii"]["A"]
                slots.update({f"{name}.{axis}": Interval.point(state[name]["position"][j]
                    + state[name]["velocity"][j] * duration) for name in "AB" for j, axis in enumerate("xyz")})
            elif category == "conservation":
                momentum = sum(k["masses"][name] * body["velocity"][0] for name, body in state.items())
                slots["momentum.x"] = Interval.point(momentum)
                context["initial_momentum"] = asdict(around(momentum))
            elif category == "permanence":
                if k["persistent_identity"] != "A":
                    return
                slots["A.count.final"] = Interval.point(1)
            elif category == "causal_ordering":
                dependency = k["event_dependency"]
                if dependency["minimum_delay"] != 2 / 15:
                    return
                for name, key, plane in (("A", "cause.time", "cause_plane_x"), ("B", "effect.time", "effect_plane_x")):
                    slots[key] = Interval.point((dependency[plane] - state[name]["position"][0]) / state[name]["velocity"][0])
            elif category == "resource_reachability":
                position = state["A"]["position"][0]
                reach = k["velocity_bound"] * duration
                context["reachable_x"] = asdict(Interval(str(position - reach - 1e-6), str(position + reach + 1e-6)))
                slots["A.x"] = Interval.point(position + state["A"]["velocity"][0] * duration)
            else:
                return
            self.context, self.reference = context, Commitment(slots)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return

    def checks(self, query, commitment):
        if self.context is None or query.binding != self.binding:
            return (Check("history_missing", "history_coverage", (), (0, 0),
                          lambda: Bound(None, "unknown", False, None, query.binding, "permitted history premises missing")),)
        from .sdg import checks_for
        return checks_for(self.context, query, commitment)

    def __call__(self, query, commitment):
        if self.reference is None or query.binding != self.binding or set(commitment.slots) != set(self.reference.slots):
            return Verdict.UNKNOWN
        return Verdict.VALID if all(commitment.slots[k].contains(v) for k, v in self.reference.slots.items()) else Verdict.INVALID


def measure_future(context, future):
    import numpy as np
    from .sdg import around
    poses = np.asarray(future["positions"], dtype=float)
    visible = np.asarray(future["visible"], dtype=bool)
    scene = context["scene"]
    if (poses.shape != (scene["horizon"] + 1, scene["bodies"], 3)
            or visible.shape != poses.shape[:2] or not np.isfinite(poses).all()):
        raise ValueError("candidate evidence has invalid poses or visibility")
    slots = {"A.initial.x": around(poses[0, 0, 0])}
    category = scene["category"]
    if category == "contact_solidity":
        slots.update({f"{name}.{axis}": around(poses[-1, i, j]) for i, name in enumerate("AB") for j, axis in enumerate("xyz")})
    elif category == "conservation":
        momentum = scene["mass"] * sum(poses[-1, :, 0] - poses[0, :, 0]) * scene["control_hz"] / scene["horizon"]
        slots["momentum.x"] = around(momentum)
    elif category == "permanence":
        slots["A.count.final"] = Interval.point(int(visible[-1, 0]))
    elif category == "causal_ordering":
        dependency = context["query"]["premises"]["event_dependency"]
        for i, key, plane in ((0, "cause.time", "cause_plane_x"), (1, "effect.time", "effect_plane_x")):
            values = poses[:, i, 0]
            crossing = next((t for t in range(1, len(values)) if values[t - 1] <= dependency[plane] <= values[t]), None)
            if crossing is None or values[crossing] == values[crossing - 1]:
                return None
            fraction = (dependency[plane] - values[crossing - 1]) / (values[crossing] - values[crossing - 1])
            slots[key] = around((crossing - 1 + fraction) / scene["control_hz"])
    else:
        slots["A.x"] = around(poses[-1, 0, 0])
    return Commitment(slots)


class FutureOracle:
    def __init__(self, context, commitment, future):
        self.context, self.original, self.future = context, commitment, future
        self.reference = commitment_from_dict(context["legal_commitment"])

    def measurement(self, commitment):
        if commitment == self.original:
            future = self.future
        else:
            from .sdg import candidate_poses
            poses, visible = candidate_poses(self.context, commitment)
            future = {"positions": poses.tolist(), "visible": visible.tolist()}
        return measure_future(self.context, future)

    def checks(self, query, commitment):
        from .sdg import checks_for
        measured = self.measurement(commitment)
        covered = (measured is not None and set(measured.slots) == set(commitment.slots)
                   and all(commitment.slots[k].lower - Interval.point("0.000001").lower <= (v.lower + v.upper) / 2 <= commitment.slots[k].upper + Interval.point("0.000001").upper
                           for k, v in measured.slots.items()))
        if not covered:
            return (Check("future_mismatch", "candidate_coverage", (), (0, query.horizon / 15),
                          lambda: Bound(None, "unknown", False, None, query.binding, "candidate evidence does not cover commitment")),)
        return checks_for(self.context, query, commitment)

    def __call__(self, query, commitment):
        if query.binding != self.context["query_binding"] or set(commitment.slots) != set(self.reference.slots):
            return Verdict.UNKNOWN
        measured = self.measurement(commitment)
        if measured is None:
            return Verdict.UNKNOWN
        for key, ref in self.reference.slots.items():
            point = Interval.point((ref.lower + ref.upper) / 2)
            if not measured.slots[key].contains(point):
                return Verdict.INVALID
        return Verdict.VALID


def write_future(path, poses, visible, observations, root):
    from .train import confined_media_path, sha256
    future = {"positions": poses.tolist(), "visible": visible.tolist(),
              "media": {f["path"]: sha256(confined_media_path(root, f["path"])) for f in observations}}
    path.write_text(json.dumps(future, sort_keys=True) + "\n")
    return {"future": str(path.relative_to(root)), "future_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_future(row, root, context):
    from .sdg import candidate_poses
    from .train import confined_media_path, sha256
    path = confined_media_path(root, row["construction"]["future"])
    if sha256(path) != row["construction"]["future_sha256"]:
        raise ValueError("candidate evidence digest differs")
    future = json.loads(path.read_text())
    media = {f["path"]: sha256(confined_media_path(root, f["path"])) for f in row["observations"]["critic"]}
    poses, visible = candidate_poses(context, commitment_from_dict(row["commitment"]))
    expected = {"positions": poses.tolist(), "visible": visible.tolist(), "media": media}
    if fingerprint(future) != fingerprint(expected):
        raise ValueError("candidate evidence differs from its edit replay or rendered media")
    return future
