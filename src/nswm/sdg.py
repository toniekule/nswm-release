"""MuJoCo event construction with replay-bound constraints and repair records."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from .annotation import annotate_event, check_repair
from .data import FAMILIES, read_events, split_family
from .primitives import conservation, persistence, precedence, reachable_box, sphere_penetration
from .schema import Commitment, Edit, Interval, Query, Verdict, commitment_from_dict, fingerprint
from .scenes import Scene, read_spec, sample_scene
from .verify import Check


def around(value, epsilon=0.000001):
    return Interval(str(float(value) - epsilon), str(float(value) + epsilon))


def checks_for(context, query, commitment):
    category = context["scene"]["category"]
    valid = query.binding == context["query_binding"]
    def bound():
        if category == "contact_solidity":
            return sphere_penetration(query, commitment, "A", "B", Interval.point(context["scene"]["radius"]),
                Interval.point(context["scene"]["radius"]), premises_valid=valid)
        if category == "conservation":
            return conservation(query, commitment, "momentum.x", Interval(**context["initial_momentum"]),
                                Interval.point(0), premises_valid=valid)
        if category == "permanence":
            return persistence(query, commitment, "A.count.final", premises_valid=valid)
        if category == "causal_ordering":
            return precedence(query, commitment, [("cause.time", "effect.time", Interval.point(2 / 15))], valid)
        return reachable_box(query, commitment, {"A.x": Interval(**context["reachable_x"])}, premises_valid=valid)
    entities = ("A", "B") if category in {"contact_solidity", "causal_ordering"} else ("A",)
    time = (15 / 15, 16 / 15)
    if category == "permanence":
        time = (12 / 15, 16 / 15)
    elif category == "causal_ordering":
        time = (float(min(commitment.slots[k].lower for k in ("cause.time", "effect.time"))),
                float(max(commitment.slots[k].upper for k in ("cause.time", "effect.time"))))
    return (Check(category, category, entities, time, bound),)


class ReplayOracle:
    def __init__(self, context):
        self.context = context
        self.reference = commitment_from_dict(context["legal_commitment"])

    def __call__(self, query, commitment):
        if (query.binding != self.context["query_binding"] or set(commitment.slots) != set(self.reference.slots)):
            return Verdict.UNKNOWN
        for check in checks_for(self.context, query, commitment):
            bound = check.evaluate()
            if bound.premises_valid and bound.residual is not None and bound.residual.lower > 0:
                return Verdict.INVALID
        return Verdict.VALID if all(commitment.slots[k].contains(v) for k, v in self.reference.slots.items()) else Verdict.INVALID


def reference_context(scene, replay, xml_path):
    import mujoco
    import numpy as np
    positions = replay.positions[:, [mujoco.mj_name2id(replay.model, mujoco.mjtObj.mjOBJ_BODY, name)
                                     for name in "ABC"[:scene.bodies]], :]
    positions = np.round(positions, 9)
    slots = {"A.initial.x": around(positions[0, 0, 0])}
    if scene.category == "contact_solidity":
        slots.update({f"{name}.{axis}": around(positions[-1, i, j])
                      for i, name in enumerate("AB") for j, axis in enumerate("xyz")})
    elif scene.category == "conservation":
        slots["momentum.x"] = around(scene.mass * sum(replay.qvel[6 * i] for i in range(scene.bodies)))
    elif scene.category == "permanence":
        slots["A.count.final"] = Interval.point(1)
    elif scene.category == "causal_ordering":
        slots.update({"cause.time": around(6 / 15), "effect.time": around(10 / 15)})
    else:
        slots["A.x"] = around(positions[-1, 0, 0])
    premises = {"replay_identity": replay.identity,
        "initial_state": {name: {"position": positions[0, i].tolist(), "velocity": replay.qvel[i * 6:i * 6 + 3].tolist()}
                          for i, name in enumerate("ABC"[:scene.bodies])},
        "gravity": [0, 0, 0], "external_impulse": [0, 0], "control_hz": 15,
        "motion_model": "constant_velocity", "category": scene.category}
    if scene.category == "contact_solidity":
        premises["radii"] = {name: scene.radius for name in "AB"}
    elif scene.category == "conservation":
        premises["masses"] = {name: scene.mass for name in "ABC"[:scene.bodies]}
    elif scene.category == "permanence":
        premises["persistent_identity"] = "A"
    elif scene.category == "causal_ordering":
        premises["event_dependency"] = {"cause": "A.crossing", "effect": "B.crossing", "minimum_delay": 2 / 15,
            "cause_plane_x": float(positions[0, 0, 0] + replay.qvel[0] * 6 / 15),
            "effect_plane_x": float(positions[0, 1, 0] + replay.qvel[6] * 10 / 15),
            "threshold_tolerance": 0.000001}
    else:
        premises["velocity_bound"] = scene.velocity
    query = Query(scene.family_id, tuple(tuple(a) for a in replay.actions), premises, scene.horizon)
    return {"scene": asdict(scene), "xml": str(xml_path.name), "replay_identity": replay.identity,
        "initial_qpos": replay.qpos.tolist(), "initial_qvel": replay.qvel.tolist(),
        "actions": replay.actions.tolist(), "query": asdict(query), "query_binding": query.binding,
        "positions": positions.tolist(), "legal_commitment": asdict(Commitment(slots)),
        "numeric_precision": {"position_decimals": 9, "assertion_tolerance": 0.000001},
        "initial_momentum": asdict(around(scene.mass * sum(replay.qvel[6 * i] for i in range(scene.bodies)))),
        "reachable_x": asdict(Interval(str(positions[0, 0, 0] - scene.velocity * 16 / 15 - 1e-6),
                                      str(positions[0, 0, 0] + scene.velocity * 16 / 15 + 1e-6)))}


def injected_commitment(context):
    legal = commitment_from_dict(context["legal_commitment"])
    category = context["scene"]["category"]
    if category == "contact_solidity":
        edits = (Edit("A.x", legal.slots["B.x"]),)
        if int(context["scene"]["family_id"][-1], 16) % 2:
            edits += (Edit("B.x", legal.slots["B.x"] - Interval.point("0.05")),)
    elif category == "conservation":
        edits = (Edit("momentum.x", legal.slots["momentum.x"] + Interval.point(0.7)),)
    elif category == "permanence":
        edits = (Edit("A.count.final", Interval.point(0)),)
    elif category == "causal_ordering":
        edits = (Edit("effect.time", around(4 / 15)),)
    else:
        edits = (Edit("A.x", Interval.point(float(context["reachable_x"]["hi"]) + 0.3)),)
    return legal.edited(edits), edits


def candidate_poses(context, commitment):
    import numpy as np
    scene = Scene(**context["scene"])
    poses = np.array(context["positions"], dtype=float)
    visible = np.ones((len(poses), scene.bodies), dtype=bool)
    category = scene.category
    if category in {"contact_solidity", "resource_reachability"}:
        for i, name in enumerate("AB" if category == "contact_solidity" else "A"):
            for j, axis in enumerate("xyz"):
                key = f"{name}.{axis}"
                if key in commitment.slots:
                    value = float((commitment.slots[key].lower + commitment.slots[key].upper) / 2)
                    poses[:, i, j] += np.linspace(0, value - poses[-1, i, j], len(poses))
    elif category == "conservation":
        original = (float(context["initial_momentum"]["lo"]) + float(context["initial_momentum"]["hi"])) / 2
        quantity = float((commitment.slots["momentum.x"].lower + commitment.slots["momentum.x"].upper) / 2)
        poses[:, 0, 0] += np.linspace(0, (quantity - original) / scene.mass * 16 / 15, len(poses))
    elif category == "permanence":
        if commitment.slots["A.count.final"].upper < 1:
            visible[-5:, 0] = False
    else:
        for i, key, crossing in ((0, "cause.time", 6), (1, "effect.time", 10)):
            arrival = float((commitment.slots[key].lower + commitment.slots[key].upper) / 2)
            if arrival <= 0:
                raise ValueError("event timestamps must be positive")
            ratio = crossing / (15 * arrival)
            poses[:, i, 0] = poses[0, i, 0] + (poses[:, i, 0] - poses[0, i, 0]) * ratio
    return poses, visible


def load_context(row, root, replay=True):
    if row["construction"].get("evidence_version") == 3:
        from .corpus_v3 import load_evidence
        return load_evidence(row, root, replay)[0]
    path = (Path(root) / row["construction"]["context"]).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("context path leaves manifest root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["construction"]["context_sha256"]:
        raise ValueError("context digest differs")
    context = json.loads(path.read_text())
    if Query(**row["query"]).binding != context["query_binding"]:
        raise ValueError("event query differs from replay context")
    if replay:
        scene = Scene(**context["scene"])
        xml = path.parent / context["xml"]
        if not xml.resolve().is_relative_to(path.parent):
            raise ValueError("XML path leaves context directory")
        if xml.read_text() != scene.xml():
            raise ValueError("scene specification differs from replay XML")
        engine = scene.replay(xml)
        if engine.identity != context["replay_identity"]:
            raise ValueError("replay source identity differs")
        rebuilt = reference_context(scene, engine, xml)
        if fingerprint(rebuilt) != fingerprint(context):
            raise ValueError("replayed physical evidence differs")
    return context


def build_family(root, scene, split, width, height, renderer):
    from .render import render_clip
    from .evidence import FutureOracle, HistoryOracle, write_future
    from .repair import search_repairs
    directory = root / "contexts" / scene.family_id
    directory.mkdir(parents=True)
    xml = directory / "scene.xml"
    xml.write_text(scene.xml())
    engine = scene.replay(xml)
    context = reference_context(scene, engine, xml)
    context_path = directory / "replay.json"
    context_path.write_text(json.dumps(context, sort_keys=True) + "\n")
    legal = commitment_from_dict(context["legal_commitment"])
    negative, injection = injected_commitment(context)
    query = Query(**context["query"])
    gate_oracle = HistoryOracle(query, scene.category)
    catalogue = tuple(Edit(e.slot, legal.slots[e.slot]) for e in injection)
    slack = legal.slots["A.initial.x"]
    catalogue += (Edit("A.initial.x", slack + Interval("-0.1", "0.1")),)
    repair_search = search_repairs(query, negative, catalogue, gate_oracle)
    if not repair_search.repairs:
        raise ValueError("violating parent has no verified history repair")
    applied_repair = repair_search.repairs[0]
    history_poses = [context["positions"][0] for _ in range(5)]
    for i in range(5):
        for body in range(scene.bodies):
            history_poses[i] = [list(v) for v in history_poses[i]]
            history_poses[i][body][0] += context["initial_qvel"][body * 6] * (i - 4) / 15
    history_paths = render_clip(scene, history_poses, [[True] * scene.bodies] * 5,
        root / "media" / scene.family_id / "history", width, height, scene.tint, renderer, xml)
    history = [{"path": str(p.relative_to(root)), "time": (i - 4) / 15} for i, p in enumerate(history_paths)]
    rows = []
    for variant in ("legal", "violating", "repaired", "appearance", "boundary_legal"):
        z = negative if variant in {"violating", "appearance"} else legal
        if variant == "repaired":
            z = negative.edited(applied_repair)
        if variant == "boundary_legal":
            z = Commitment({k: v + Interval("-0.000001", "0") for k, v in legal.slots.items()})
        poses, visible = candidate_poses(context, z)
        paths = render_clip(scene, poses, visible, root / "media" / scene.family_id / variant,
            width, height, scene.tint + (35 if variant == "appearance" else 0), renderer, xml)
        event = {"id": f"{scene.family_id}/{variant}", "family_id": scene.family_id,
            "category": scene.category, "variant": variant, "split": split,
            "domain": "id" if scene.shift == "id" else "ood", "ood_axis": scene.shift,
            "difficulty": len(injection), "query": asdict(query), "commitment": asdict(z),
            "observations": {"gate": history, "critic": [{"path": str(p.relative_to(root)), "time": i / 15}
                                                        for i, p in enumerate(paths)]},
            "evidence_kind": "mujoco_controlled_domain", "construction": {
                "evidence_version": 2,
                "context": str(context_path.relative_to(root)),
                "context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
                "injected_edits": [asdict(e) for e in injection], "edit_catalogue": [asdict(e) for e in catalogue],
                "max_edits": 6, "render_backend": renderer}}
        event["construction"].update(write_future(directory / (variant + ".json"), poses, visible,
                                                  event["observations"]["critic"], root))
        if variant == "repaired":
            event["construction"]["repair_parent"] = {"id": f"{scene.family_id}/violating",
                "commitment": asdict(negative), "applied_edits": [asdict(e) for e in applied_repair]}
        critic_oracle = FutureOracle(context, z, {"positions": poses.tolist(), "visible": visible.tolist()})
        gate_checks, critic_checks = gate_oracle.checks(query, z), critic_oracle.checks(query, z)
        row = annotate_event(event, critic_oracle, catalogue, critic_checks, gate_checks, gate_oracle=gate_oracle)
        row["redundant_targets"] = {}
        for mode in ("gate", "critic"):
            oracle = gate_oracle if mode == "gate" else critic_oracle
            checks = gate_checks if mode == "gate" else critic_checks
            row["annotation"][mode]["label_source"] = "permitted_history" if mode == "gate" else "measured_candidate_future"
            row["annotation"][mode]["source_binding"] = (fingerprint(dict(query.premises)) if mode == "gate"
                                                         else event["construction"]["future_sha256"])
            target = row["targets"][mode]
            bounds = [check.evaluate().residual for check in checks]
            target["scalar_residual"] = str(max(b.lower for b in bounds if b is not None))
            redundant = dict(target)
            if target["judgment"] == "invalid":
                if row["annotation"][mode]["verification"] != "V" or not target["repair"]:
                    raise ValueError("injected violation lacks an independent proof or replay-valid repair")
                minimum = tuple(Edit(e["slot"], Interval(**e["value"])) for e in target["repair"])
                slack = legal.slots["A.initial.x"]
                extended = (*minimum, Edit("A.initial.x", slack + Interval("-0.1", "0.1")))
                if check_repair(query, z, extended, oracle) != "redundant":
                    raise ValueError("redundant target failed subset replay")
                redundant["repair"] = [asdict(e) for e in extended]
            row["redundant_targets"][mode] = redundant
        rows.append(row)
    return rows


def builder_identity():
    source = Path(__file__).resolve().parent
    paths = sorted(source.glob("*.py"))
    hashes = {"nswm/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, stderr=subprocess.DEVNULL, text=True).strip()
        tracked = bool(subprocess.check_output(["git", "ls-files", "--", str(source)], cwd=source,
                                               stderr=subprocess.DEVNULL, text=True).strip())
        dirty = not tracked or bool(subprocess.check_output(["git", "status", "--porcelain", "--", str(source)], cwd=source,
                                             stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"git_head": commit, "source_dirty": dirty, "source_sha256": fingerprint(hashes), "files": hashes}


def build_corpus(spec_path, destination, families=50, seed=0, split_spec=None, renderer="software"):
    from collections import Counter
    import mujoco
    from .isolation import audit_isolation, group_splits
    spec = read_spec(spec_path)
    if spec["version"] == 3:
        from .corpus_v3 import build_corpus as build_mechanisms
        split = json.loads(Path(split_spec).read_text()) if split_spec else None
        return build_mechanisms(spec, destination, families, seed, split, renderer)
    split = json.loads(Path(split_spec).read_text()) if split_spec else {"version": 1, "train": 0.8, "dev": 0.1}
    if split.get("version") != 1 or not 0 < split["train"] < 1 or not 0 <= split["dev"] < 1 - split["train"]:
        raise ValueError("invalid split specification")
    if split_spec and set(split.get("ood", {})) != {"geometry", "dynamics", "appearance", "composition"}:
        raise ValueError("all four OOD axes required")
    if families < (20 if split_spec else 5) or renderer not in {"software", "mujoco"}:
        raise ValueError("at least five ID families or twenty ID/OOD families and a supported renderer required")
    root = Path(destination).resolve()
    if root.exists():
        raise FileExistsError("corpus destination exists")
    root.mkdir(parents=True)
    marker = root / "BUILD_INCOMPLETE"
    marker.write_text("construction in progress\n")
    populations = {}
    audit_inputs = []
    for name in (["corpus-train", "eval-ood"] if split_spec else ["corpus"]):
        out = root / name if split_spec else root
        out.mkdir(exist_ok=True)
        rows = []
        for i in range(families):
            axis = ("geometry", "dynamics", "appearance", "composition")[(i // 5) % 4] if name == "eval-ood" else "id"
            scene = sample_scene(spec, i, seed, axis, split if axis != "id" else None)
            label = "test" if name == "eval-ood" else split_family(scene.family_id, seed, split["train"], split["dev"])
            rows.extend(build_family(out, scene, label, spec.get("width", 128), spec.get("height", 96), renderer))
        split_groups = group_splits(rows, out, seed, split["train"], split["dev"]) if name != "eval-ood" else None
        manifest = out / "events.jsonl"
        manifest.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
        read_events(manifest)
        members = sorted(p for p in out.rglob("*") if p.is_file() and p != marker)
        card = {"version": 2, "families": families, "events": len(rows), "seed": seed,
            "population": name, "category_quotas": dict(Counter(r["category"] for r in rows if r["variant"] == "legal")),
            "splits": dict(Counter(r["split"] for r in rows if r["variant"] == "legal")),
            "axis_quotas": dict(Counter(r["ood_axis"] for r in rows if r["variant"] == "legal")),
            "split_basis": split_groups if name != "eval-ood" else "independent shifted families; test only",
            "spec": spec, "split_spec": split, "builder": builder_identity(), "mujoco_version": mujoco.__version__,
            "renderer": renderer, "max_edits": 6, "repair_budget": 4096,
            "label_sources": {"gate": "permitted_history", "critic": "measured_candidate_future"},
            "label_counts": {mode: dict(Counter(r["targets"][mode]["judgment"] for r in rows)) for mode in ("gate", "critic")},
            "difficulty_by_category": {category: dict(Counter(str(r["difficulty"]) for r in rows
                if r["category"] == category and r["variant"] == "violating")) for category in FAMILIES},
            "variant_semantics": "five event roles; repaired replay may coincide with the legal reference",
            "truth_domain": "fixed initial state, actions, zero gravity and bounded exchanges",
            "files": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in members}}
        (out / "data_card.json").write_text(json.dumps(card, indent=2) + "\n")
        populations[name] = {"manifest": str(manifest), "data_card": str(out / "data_card.json"),
                             "families": families, "events": len(rows)}
        audit_inputs.append((name, rows, out))
    isolation = audit_isolation(audit_inputs)
    (root / "isolation.json").write_text(json.dumps(isolation, indent=2) + "\n")
    for population in populations.values():
        path = Path(population["data_card"])
        card = json.loads(path.read_text())
        card["isolation"] = isolation
        if not split_spec:
            card["files"]["isolation.json"] = hashlib.sha256((root / "isolation.json").read_bytes()).hexdigest()
        path.write_text(json.dumps(card, indent=2) + "\n")
    if not isolation["isolated"]:
        raise ValueError("corpus family or complete observation content crosses population/split boundaries")
    marker.unlink()
    return populations
