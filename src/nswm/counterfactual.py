"""Fixed-context candidates, independent evidence sources and verified variants."""
from copy import deepcopy
from dataclasses import asdict, dataclass

from .mechanisms import MechanismScene, measure, physical_limits, quantity_slot, simulate, slot_schema
from .repair import search_repairs
from .primitives import conservation, minimum_separation, persistence, precedence, reachable_box
from .schema import Commitment, Edit, Interval, Query, Verdict, commitment_from_dict, fingerprint
from .verify import Bound, Check


def enclosed(value, tolerance=1e-6):
    return Interval(str(float(value) - tolerance), str(float(value) + tolerance))


def reference_context(scene):
    replay = simulate(scene)
    if replay["trace"]["max_penetration"] > scene.tolerance:
        raise ValueError("reference penetration exceeds the registered tolerance")
    values = measure(scene, replay["trace"])
    legal = Commitment({key: Interval.point(value) if scene.category == "permanence" and key != "initial.x"
                        else enclosed(value) for key, value in values.items()})
    query = Query(scene.family_id, tuple(tuple(a) for a in replay["actions"]), {
        "domain_version": 3, "mechanism": asdict(scene), "engine": replay["engine"],
        "initial_qpos": replay["initial_qpos"], "initial_qvel": replay["initial_qvel"],
        "slot_schema": slot_schema(scene),
        "identity_persistence": True, "closed_mechanism": True}, scene.horizon)
    return {**replay, "version": 3, "query": asdict(query), "query_binding": query.binding,
            "legal_commitment": asdict(legal), "root_family_id": scene.family_id}


def candidate_trace(context, commitment):
    import numpy as np
    scene = MechanismScene(**context["scene"])
    reference = measure(scene, context["trace"])
    if set(commitment.slots) != set(reference):
        raise ValueError("candidate commitment slot schema differs")
    trace = deepcopy({k: context["trace"][k] for k in ("names", "positions", "velocities", "exists", "visible", "events")})
    poses, velocities = np.asarray(trace["positions"]), np.asarray(trace["velocities"])
    exists, visible = np.asarray(trace["exists"]), np.asarray(trace["visible"])
    for lane in range(scene.lanes):
        key, start = quantity_slot(scene, lane), lane * scene.lane_size
        interval = commitment.slots[key]
        if interval.contains(Interval.point(reference[key])):
            continue
        value = float((interval.lower + interval.upper) / 2)
        ramp = np.linspace(0, 1, 5)
        if scene.category == "contact_solidity":
            if value < 0:
                raise ValueError("distance cannot be negative")
            delta = poses[-1, start + 1] - poses[-1, start]
            length = np.linalg.norm(delta)
            direction = delta / length if length else np.array([1.0, 0, 0])
            offset = poses[-1, start] + direction * value - poses[-1, start + 1]
            poses[-5:, start + 1] += ramp[:, None] * offset
            velocities[-4:, start + 1] = np.diff(poses[-5:, start + 1], axis=0) * scene.control_hz
        elif scene.category == "conservation":
            delta_v = (value - reference[key]) / scene.mass
            velocities[-5:, start, 0] += ramp * delta_v
            poses[-4:, start, 0] += np.cumsum(ramp[1:] * delta_v / scene.control_hz)
        elif scene.category == "permanence":
            if value not in {0, 1}:
                raise ValueError("identity existence must be binary")
            exists[-5:, start] = int(value)
            visible[-5:, start] &= bool(value)
        elif scene.category == "causal_ordering":
            cause = next(e["time"] for e in trace["events"] if tuple(e["entities"]) == (f"L{lane}.A", f"L{lane}.B"))
            last = "ABCD"[scene.lane_size - 2:scene.lane_size]
            effect = next(e for e in trace["events"] if tuple(e["entities"]) == (f"L{lane}.{last[0]}", f"L{lane}.{last[1]}"))
            target = cause + value
            if not 0 <= target <= scene.duration:
                raise ValueError("edited contact event falls outside the horizon")
            old_frame = min(scene.horizon, max(1, round(effect["time"] * scene.control_hz)))
            new_frame = min(scene.horizon, max(1, round(target * scene.control_hz)))
            advance = max(1, old_frame - new_frame)
            for field in (poses, velocities):
                first = start + scene.lane_size - 2
                before = field[:, first:start + scene.lane_size].copy()
                indices = np.minimum(np.arange(len(field)) + advance, scene.horizon)
                field[new_frame:, first:start + scene.lane_size] = before[indices[new_frame:]]
            effect["time"] = target
            effect["source"] = "controlled_event_edit"
        else:
            offset = value - reference[key]
            poses[-5:, start, 0] += ramp * offset
            velocities[-4:, start, 0] = np.diff(poses[-5:, start, 0]) * scene.control_hz
    initial = commitment.slots["initial.x"]
    if not initial.contains(Interval.point(reference["initial.x"])):
        poses[0, 0, 0] = float((initial.lower + initial.upper) / 2)
    trace.update(positions=poses.tolist(), velocities=velocities.tolist(), exists=exists.tolist(), visible=visible.tolist())
    return trace


def physical_identity(trace):
    return fingerprint({k: trace[k] for k in ("names", "positions", "velocities", "exists", "events")})


def inventory(scene, query, commitment, covered=True):
    limits, units = physical_limits(scene)
    result = []
    for key, permitted in limits.items():
        lane = int(key.split('.')[0][1:])
        entities = tuple(scene.names[lane * scene.lane_size:(lane + 1) * scene.lane_size])
        def evaluate(key=key, permitted=permitted):
            value = commitment.slots.get(key)
            if not covered or value is None:
                return Bound(None, units[key], False, None, query.binding, "candidate or history coverage missing")
            if scene.category == "contact_solidity":
                return minimum_separation(query, commitment, key, Interval.point(scene.radius),
                    Interval.point(scene.radius), Interval.point(scene.tolerance), True)
            if scene.category == "conservation":
                initial = Interval.point(scene.mass * scene.speed)
                return conservation(query, commitment, key, initial, permitted - initial, premises_valid=True)
            if scene.category == "permanence":
                return persistence(query, commitment, key, premises_valid=True)
            if scene.category == "causal_ordering":
                local = Commitment({"origin": Interval.point(0), key: value})
                bound = precedence(query, local, [("origin", key, Interval.point(0))], True)
                return Bound(bound.residual, bound.unit, bound.premises_valid, Commitment({key: value}), query.binding, bound.reason)
            return reachable_box(query, commitment, {key: permitted}, premises_valid=True)
        result.append(Check(key, scene.category, entities, (0, scene.duration), evaluate))
    return tuple(result)


class MechanismHistoryOracle:
    def __init__(self, query):
        self.binding, self.scene, self.reference = query.binding, None, None
        try:
            k = query.premises
            if k["domain_version"] != 3 or not k["closed_mechanism"] or not k["identity_persistence"]:
                return
            scene = MechanismScene(**k["mechanism"])
            if query.horizon != scene.horizon or query.history_id != scene.family_id:
                return
            if fingerprint(k["slot_schema"]) != fingerprint(slot_schema(scene)):
                return
            replay = simulate(scene, query.actions)
            if (fingerprint(replay["engine"]) != fingerprint(k["engine"])
                    or fingerprint(replay["initial_qpos"]) != fingerprint(k["initial_qpos"])
                    or fingerprint(replay["initial_qvel"]) != fingerprint(k["initial_qvel"])
                    or replay["trace"]["max_penetration"] > scene.tolerance):
                return
            values = measure(scene, replay["trace"])
            self.scene, self.reference = scene, Commitment({key: Interval.point(v) for key, v in values.items()})
        except (KeyError, TypeError, ValueError):
            return

    def __call__(self, query, commitment):
        if self.reference is None or query.binding != self.binding or set(commitment.slots) != set(self.reference.slots):
            return Verdict.UNKNOWN
        return Verdict.VALID if all(commitment.slots[k].contains(v) for k, v in self.reference.slots.items()) else Verdict.INVALID

    def checks(self, query, commitment):
        if self.scene is not None and query.binding == self.binding:
            return inventory(self.scene, query, commitment)
        return (Check("history_coverage", "history_coverage", (), (0, 0),
                      lambda: Bound(None, "unknown", False, None, query.binding, "permitted mechanism history incomplete")),)


class MechanismFutureOracle:
    def __init__(self, context, original, future):
        self.context, self.original, self.future = context, original, future
        self.scene = MechanismScene(**context["scene"])

    def measurement(self, commitment):
        trace = self.future if commitment == self.original else candidate_trace(self.context, commitment)
        return trace, measure(self.scene, trace)

    def __call__(self, query, commitment):
        if query.binding != self.context["query_binding"]:
            return Verdict.UNKNOWN
        try:
            trace, values = self.measurement(commitment)
        except (KeyError, TypeError, ValueError):
            return Verdict.UNKNOWN
        if set(values) != set(commitment.slots):
            return Verdict.UNKNOWN
        covered = all(commitment.slots[k].contains(Interval.point(v)) for k, v in values.items())
        return Verdict.VALID if covered and physical_identity(trace) == physical_identity(self.context["trace"]) else Verdict.INVALID

    def checks(self, query, commitment):
        try:
            _, measured = self.measurement(commitment)
            covered = (query.binding == self.context["query_binding"] and set(measured) == set(commitment.slots)
                       and all(commitment.slots[k].contains(Interval.point(v)) for k, v in measured.items()))
        except (KeyError, TypeError, ValueError):
            covered = False
        return inventory(self.scene, query, commitment, covered)


@dataclass(frozen=True)
class Variant:
    name: str
    roles: tuple[str, ...]
    commitment: Commitment
    style: int
    parent: str | None = None
    applied_edits: tuple[Edit, ...] = ()


def variant_plan(context, budget=4096):
    scene = MechanismScene(**context["scene"])
    query = Query(**context["query"])
    legal = commitment_from_dict(context["legal_commitment"])
    limits, _ = physical_limits(scene)
    injection = []
    for key, limit in limits.items():
        if scene.category == "contact_solidity":
            value = Interval.point(scene.radius)
        elif scene.category == "permanence":
            value = Interval.point(0)
        elif scene.category == "causal_ordering":
            value = Interval.point(-1 / scene.physics_hz)
        else:
            value = Interval.point(float(limit.upper) + max(0.1, scene.tolerance * 10))
        injection.append(Edit(key, value if scene.category == "permanence" else enclosed(float(value.lower))))
    negative = legal.edited(tuple(injection))
    catalogue = tuple(Edit(e.slot, legal.slots[e.slot]) for e in injection)
    catalogue += tuple(Edit(e.slot, e.value) for e in injection)
    catalogue += tuple(Edit(e.slot, legal.slots[e.slot] + Interval("-0.0001", "0.0001")) for e in injection)
    catalogue += (Edit("initial.x", legal.slots["initial.x"] + Interval("-0.1", "0.1")),)
    gate = MechanismHistoryOracle(query)
    search = search_repairs(query, negative, catalogue, gate, budget=budget)
    difficulty = search.minimum_cardinality
    repaired = negative.edited(search.repairs[0]) if search.repairs else None
    variants = [Variant("legal", ("legal",), legal, 0), Variant("violating", ("violating",), negative, 0),
                Variant("appearance_legal", ("legal", "appearance"), legal, 1),
                Variant("appearance_violating", ("violating", "appearance"), negative, 1)]
    if repaired is not None:
        variants.extend([Variant("repaired", ("repaired",), repaired, 0, "violating", search.repairs[0]),
                         Variant("appearance_repaired", ("repaired", "appearance"), repaired, 1,
                                 "appearance_violating", search.repairs[0])])
    margins = {key: float(min(legal.slots[key].lower - limit.lower, limit.upper - legal.slots[key].upper))
               for key, limit in limits.items()}
    boundary = {"source": "physical_state_and_action_sampling", "margins": margins,
                "units": physical_limits(scene)[1], "continuous": scene.category != "permanence"}
    return {"variants": tuple(variants), "injection": tuple(injection), "catalogue": catalogue,
            "search": search, "difficulty": difficulty,
            "difficulty_bin": "unknown" if difficulty is None else str(difficulty) if difficulty < 3 else "3+",
            "boundary": boundary, "gate_oracle": gate}
