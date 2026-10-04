"""Versioned mechanism corpus planning, evidence assembly and integrity checks."""
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

from .annotation import annotate_event, check_repair, repair_key
from .counterfactual import (MechanismFutureOracle, candidate_trace, physical_identity,
                            reference_context, variant_plan)
from .data import FAMILIES, read_events, split_family
from .mechanisms import MechanismScene
from .schema import Edit, Interval, Query, commitment_from_dict, fingerprint


AXES = ("geometry", "dynamics", "appearance", "composition")


def validate_spec(spec):
    if spec.get("version") != 3 or set(spec.get("categories", ())) != set(FAMILIES) or len(spec["categories"]) != 5:
        raise ValueError("mechanism spec version 3 requires all five categories")
    if spec.get("repair_cardinalities") != [1, 2, 3, 4, 5, 6] or spec.get("boundary_levels") != ["regular", "near"]:
        raise ValueError("six repair cardinalities and regular/near physical sampling are required")
    for key in ("radius", "mass", "speed", "gap", "force_limit"):
        pair = spec.get(key)
        if (not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in pair)
                or not 0 < pair[0] <= pair[1]):
            raise ValueError("invalid mechanism parameter range: " + key)
    if not 32 <= spec.get("width", 128) <= 448 or not 32 <= spec.get("height", 96) <= 256:
        raise ValueError("invalid mechanism image dimensions")
    if spec["gap"][1] / spec["speed"][0] > 0.35:
        raise ValueError("mechanism parameter ranges exceed contact timing coverage")
    if spec.get("repair_budget", 4096) < 1 or not 0 < spec.get("tolerance", 0.005) < spec["radius"][0]:
        raise ValueError("invalid query budget or penetration tolerance")
    thresholds = spec.get("boundary_thresholds", {})
    if set(thresholds) != set(FAMILIES) - {"permanence"} or any(not math.isfinite(v) or v <= 0 for v in thresholds.values()):
        raise ValueError("continuous mechanism boundary thresholds must be declared")
    return spec


def corpus_plan(spec, families, seed=0, split=None):
    validate_spec(spec)
    cell_count = 5 * 6 * 2
    minimum = cell_count * (4 if split else 1)
    if families < minimum or families % minimum:
        raise ValueError(f"families must be a positive multiple of {minimum} for complete mechanism quotas")
    if split is not None and (split.get("version") != 1 or set(split.get("ood", {})) != set(AXES)
            or not 0 < split.get("train", 0) < 1 or not 0 <= split.get("dev", -1) < 1 - split["train"]):
        raise ValueError("invalid split specification")
    populations = {}
    for population in ("corpus-train", "eval-ood") if split else ("corpus",):
        scenes = []
        for index in range(families):
            category = spec["categories"][index % 5]
            lanes, level = 1 + (index // 5) % 6, spec["boundary_levels"][(index // 30) % 2]
            axis = AXES[(index // cell_count) % 4] if population == "eval-ood" else "id"
            identity = fingerprint({"spec": spec, "index": index, "seed": seed, "axis": axis, "split": split})
            rng = random.Random(int(identity[:16], 16))
            values = {k: round(rng.uniform(*spec[k]), 8) for k in ("radius", "mass", "speed", "gap", "force_limit")}
            if axis != "id":
                setting = split["ood"][axis]
                for key in values:
                    alias_key = {"speed": "velocity", "gap": "spacing"}.get(key, key)
                    values[key] *= setting.get(key + "_scale", setting.get(alias_key + "_scale", 1))
            drive = 0.995 if level == "near" else 0.65
            if category == "causal_ordering" and level == "near":
                values["gap"] = values["speed"] / 120
            scene = MechanismScene(f"{axis}-{category}-{identity[:16]}", category, seed, lanes=lanes,
                drive_fraction=drive, tint=rng.randrange(20) + (80 if axis == "appearance" else 0),
                impact_fraction=2.0 if category == "contact_solidity" and level == "regular" else 0.0,
                tolerance=spec.get("tolerance", 0.005), shift=axis, **values)
            fraction = split or {"train": 0.8, "dev": 0.1}
            label = "test" if axis != "id" else split_family(scene.family_id, seed, fraction["train"], fraction["dev"])
            scenes.append({"scene": asdict(scene), "split": label, "requested_cardinality": lanes,
                           "boundary_sampling": level, "root_family_id": scene.family_id})
        populations[population] = scenes
    return {"version": 3, "seed": seed, "spec_sha256": fingerprint(spec), "populations": populations,
            "split_basis": "root family SHA256 before replay, annotation and rendering",
            "sampling_cells": ["category", "requested_cardinality", "boundary_sampling", "ood_axis"]}


def prepare_family(scene, budget=4096):
    context = reference_context(scene)
    plan = variant_plan(context, budget)
    if plan["difficulty"] != scene.lanes:
        raise ValueError("requested repair cardinality was not proved by the bounded catalogue")
    return context, plan


def history_trace(scene, context):
    from copy import deepcopy
    trace = deepcopy({k: context["trace"][k] for k in ("names", "positions", "velocities", "exists", "visible", "events")})
    for key in ("positions", "velocities", "exists", "visible"):
        trace[key] = [deepcopy(trace[key][0]) for _ in range(5)]
    for frame in range(5):
        for body in range(scene.bodies):
            for axis in range(3):
                trace["positions"][frame][body][axis] += trace["velocities"][frame][body][axis] * (frame - 4) / scene.control_hz
    trace["events"] = []
    return trace


def build_family(root, declaration, spec, renderer):
    from .render import render_mechanism
    from .train import sha256
    scene = MechanismScene(**declaration["scene"])
    context, plan = prepare_family(scene, spec.get("repair_budget", 4096))
    query, legal = Query(**context["query"]), commitment_from_dict(context["legal_commitment"])
    directory = root / "contexts" / scene.family_id
    directory.mkdir(parents=True)
    (directory / "scene.xml").write_text(scene.xml())
    context_path = directory / "replay.json"
    context_path.write_text(json.dumps(context, sort_keys=True) + "\n")
    histories, trajectories, clips = {}, {}, {}
    negative = next(v.commitment for v in plan["variants"] if v.name == "violating")
    gate_binding = fingerprint(query.premises)
    repair_cache = {repair_key(query, negative, plan["catalogue"], "gate", gate_binding): plan["search"]}
    rows = []
    for variant in plan["variants"]:
        if variant.style not in histories:
            paths = render_mechanism(scene, history_trace(scene, context),
                root / "media" / scene.family_id / f"history-{variant.style}", spec["width"], spec["height"], variant.style, renderer)
            histories[variant.style] = [{"path": str(p.relative_to(root)), "time": (i - 4) / 15} for i, p in enumerate(paths)]
        trace = candidate_trace(context, variant.commitment)
        trajectory_key = fingerprint(trace)
        if trajectory_key not in trajectories:
            future_path = directory / (trajectory_key + ".json")
            future_path.write_text(json.dumps(trace, sort_keys=True) + "\n")
            trajectories[trajectory_key] = future_path
        clip_key = fingerprint({"trace": trajectory_key, "style": variant.style, "renderer": renderer,
                                "width": spec["width"], "height": spec["height"]})
        if clip_key not in clips:
            clips[clip_key] = render_mechanism(scene, trace, root / "media" / scene.family_id / clip_key,
                spec["width"], spec["height"], variant.style, renderer)
        future_path = trajectories[trajectory_key]
        observations = {"gate": histories[variant.style], "critic": [
            {"path": str(p.relative_to(root)), "time": i / 15} for i, p in enumerate(clips[clip_key])]}
        media = {frame["path"]: sha256(root / frame["path"]) for view in observations.values() for frame in view}
        construction = {"evidence_version": 3, "context": str(context_path.relative_to(root)),
            "context_sha256": sha256(context_path), "future": str(future_path.relative_to(root)),
            "future_sha256": sha256(future_path), "media": media, "physical_identity": physical_identity(trace),
            "edit_catalogue": [asdict(e) for e in plan["catalogue"]], "injected_edits": [asdict(e) for e in plan["injection"]],
            "repair_budget": spec.get("repair_budget", 4096), "max_edits": 6, "render_backend": renderer}
        if variant.parent:
            parent = next(v for v in plan["variants"] if v.name == variant.parent)
            construction["repair_parent"] = {"id": f"{scene.family_id}/{parent.name}",
                "commitment": asdict(parent.commitment), "applied_edits": [asdict(e) for e in variant.applied_edits]}
        boundary = dict(plan["boundary"])
        threshold = spec["boundary_thresholds"].get(scene.category)
        boundary.update(sampling_level=declaration["boundary_sampling"], threshold=threshold,
            near=threshold is not None and all(0 <= v <= threshold for v in boundary["margins"].values()))
        roles = list(variant.roles)
        if boundary["near"] and "legal" in roles:
            roles.append("boundary_legal")
        event = {"id": f"{scene.family_id}/{variant.name}", "family_id": scene.family_id,
            "root_family_id": scene.family_id, "category": scene.category, "variant": variant.name, "roles": roles,
            "split": declaration["split"], "domain": "id" if scene.shift == "id" else "ood", "ood_axis": scene.shift,
            "difficulty": plan["difficulty"], "difficulty_bin": plan["difficulty_bin"],
            "boundary": boundary, "appearance": {"style": variant.style, "physics_unchanged": True},
            "query": asdict(query), "commitment": asdict(variant.commitment), "observations": observations,
            "evidence_kind": "mujoco_mechanisms_v3", "construction": construction}
        critic = MechanismFutureOracle(context, variant.commitment, trace)
        gate = plan["gate_oracle"]
        row = annotate_event(event, critic, plan["catalogue"], critic.checks(query, variant.commitment),
            gate.checks(query, variant.commitment), gate_oracle=gate, repair_budget=spec.get("repair_budget", 4096),
            repair_cache=repair_cache, oracle_bindings={"gate": gate_binding, "critic": physical_identity(trace)})
        row["redundant_targets"], row["redundant_available"], row["population_membership"] = {}, {}, {}
        for mode, oracle in (("gate", gate), ("critic", critic)):
            row["annotation"][mode].update(label_source="permitted_history" if mode == "gate" else "measured_candidate_future",
                source_binding=fingerprint(query.premises) if mode == "gate" else construction["future_sha256"])
            bounds = [c.evaluate().residual for c in oracle.checks(query, variant.commitment)]
            target = row["targets"][mode]
            target["scalar_residual"] = str(max(b.lower for b in bounds if b is not None)) if any(b is not None for b in bounds) else None
            redundant = dict(target)
            available = True
            if target["judgment"] == "invalid":
                if row["annotation"][mode]["verification"] != "V" or row["annotation"][mode]["minimum_cardinality"] != scene.lanes:
                    raise ValueError("independent label source lacks a witness or cardinality proof")
                minimum = tuple(Edit(e["slot"], Interval(**e["value"])) for e in target["repair"])
                extended = (*minimum, Edit("initial.x", legal.slots["initial.x"] + Interval("-0.1", "0.1")))
                available = len(extended) <= 6 and check_repair(query, variant.commitment, extended, oracle) == "redundant"
                redundant["repair"] = [asdict(e) for e in extended] if available else None
            row["redundant_targets"][mode], row["redundant_available"][mode] = redundant, available
            eligible = (target["judgment"] == "invalid" and row["annotation"][mode]["verification"] == "V"
                        and row["annotation"][mode]["repair_status"] == "verified_minimal")
            row["population_membership"][mode] = {"certificate": eligible, "four_target": eligible and available}
        rows.append(row)
    for row in rows:
        row["construction"]["family_repair_queries"] = sum(r.queries for r in repair_cache.values())
        row["construction"]["family_repair_sources"] = len(repair_cache)
    return rows


def load_evidence(row, root, replay=True):
    from .train import confined_media_path, sha256
    construction = row["construction"]
    context_path = confined_media_path(root, construction["context"])
    if sha256(context_path) != construction["context_sha256"]:
        raise ValueError("mechanism context digest differs")
    context = json.loads(context_path.read_text())
    query = Query(**row["query"])
    if query.binding != context["query_binding"] or query.binding != Query(**context["query"]).binding:
        raise ValueError("query differs from frozen mechanism context")
    if replay:
        rebuilt = reference_context(MechanismScene(**context["scene"]))
        if fingerprint(rebuilt) != fingerprint(context):
            raise ValueError("fixed-context mechanism replay differs")
        xml = context_path.parent / "scene.xml"
        if not xml.resolve().is_relative_to(context_path.parent) or xml.read_text() != MechanismScene(**context["scene"]).xml():
            raise ValueError("mechanism XML differs")
    parent = construction.get("repair_parent")
    if parent:
        edits = tuple(Edit(e["slot"], Interval(**e["value"])) for e in parent["applied_edits"])
        if commitment_from_dict(parent["commitment"]).edited(edits) != commitment_from_dict(row["commitment"]):
            raise ValueError("repair parent edits differ from repaired commitment")
    future_path = confined_media_path(root, construction["future"])
    if sha256(future_path) != construction["future_sha256"]:
        raise ValueError("mechanism future digest differs")
    trace = json.loads(future_path.read_text())
    if fingerprint(candidate_trace(context, commitment_from_dict(row["commitment"]))) != fingerprint(trace):
        raise ValueError("mechanism future differs from controlled edits")
    if physical_identity(trace) != construction["physical_identity"]:
        raise ValueError("physical trajectory identity differs")
    media = {f["path"]: sha256(confined_media_path(root, f["path"])) for view in row["observations"].values() for f in view}
    if media != construction["media"]:
        raise ValueError("mechanism observed media differs")
    return context, trace


def audit_families(rows, root):
    from collections import defaultdict
    from .weighting import input_identity, weighted_events
    by_id = {r["id"]: r for r in rows}
    groups = defaultdict(list)
    cache = {}
    for row in rows:
        groups[row["family_id"]].append(row)
        if row["root_family_id"] != row["family_id"] or Query(**row["query"]).history_id != row["family_id"]:
            raise ValueError("variant root lineage differs from the query family")
        parent = row["construction"].get("repair_parent")
        if parent:
            source = by_id.get(parent["id"])
            if source is None or source["family_id"] != row["family_id"] or source["commitment"] != parent["commitment"]:
                raise ValueError("repair parent is missing or differs from the recorded parent")
            edits = tuple(Edit(e["slot"], Interval(**e["value"])) for e in parent["applied_edits"])
            if commitment_from_dict(source["commitment"]).edited(edits) != commitment_from_dict(row["commitment"]):
                raise ValueError("repair application differs from the repaired commitment")
        if row["variant"] in {"violating", "appearance_violating"}:
            for mode in ("gate", "critic"):
                if row["targets"][mode]["judgment"] != "invalid" or row["annotation"][mode]["minimum_cardinality"] != row["difficulty"]:
                    raise ValueError("difficulty is not supported by both independent repair proofs")
    pairs = 0
    for family, variants in groups.items():
        if len({r["split"] for r in variants}) != 1 or len({Query(**r["query"]).binding for r in variants}) != 1:
            raise ValueError("family context or split differs between variants")
        roles = {r["variant"]: r for r in variants}
        for name in ("legal", "violating", "repaired"):
            if name not in roles or "appearance_" + name not in roles:
                raise ValueError("both physical polarities and repair require appearance pairs")
            base, styled = roles[name], roles["appearance_" + name]
            if (base["commitment"] != styled["commitment"] or base["appearance"]["style"] == styled["appearance"]["style"]
                    or base["construction"]["future_sha256"] != styled["construction"]["future_sha256"]):
                raise ValueError("appearance pair changed a commitment, state or visibility")
            for mode in ("gate", "critic"):
                if base["targets"][mode] != styled["targets"][mode]:
                    raise ValueError("appearance pair changed its physical target")
                if input_identity(base, mode, root, cache=cache) == input_identity(styled, mode, root, cache=cache):
                    raise ValueError("appearance pair did not change the observed input")
            pairs += 1
    for mode in ("gate", "critic"):
        weighted_events(rows, mode, root, cache=cache)
    return {"families": len(groups), "appearance_pairs": pairs, "paired_polarities": ["valid", "invalid"],
            "repair_parents_verified": sum(bool(r["construction"].get("repair_parent")) for r in rows)}


def build_corpus(spec, destination, families, seed=0, split=None, renderer="software"):
    import os
    import platform
    import mujoco
    import numpy as np
    import PIL
    from .isolation import audit_isolation
    from .sdg import builder_identity
    from .train import sha256
    from .weighting import input_identity
    plan = corpus_plan(spec, families, seed, split)
    root = Path(destination).resolve()
    if root.exists():
        raise FileExistsError("corpus destination exists")
    if renderer not in {"software", "mujoco"}:
        raise ValueError("unsupported mechanism renderer")
    root.mkdir(parents=True)
    marker = root / "BUILD_INCOMPLETE"
    marker.write_text("construction in progress\n")
    populations, audit_inputs = {}, []
    for name, declarations in plan["populations"].items():
        out = root / name if split else root
        out.mkdir(exist_ok=True)
        rows = [row for declaration in declarations for row in build_family(out, declaration, spec, renderer)]
        manifest = out / "events.jsonl"
        manifest.write_text(''.join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
        read_events(manifest)
        family_audit = audit_families(rows, out)
        cache = {}
        independent = {mode: len({input_identity(row, mode, out, cache=cache) for row in rows}) for mode in ("gate", "critic")}
        card = {"version": 3, "spec": spec, "split_spec": split, "seed": seed, "population": name,
            "runtime": {"mujoco": mujoco.__version__, "numpy": np.__version__, "Pillow": PIL.__version__,
                        "platform": platform.platform(), "gl_backend": os.environ.get("MUJOCO_GL", "platform_default")},
            "builder": builder_identity(), "families": families, "root_families": families, "events": len(rows),
            "split_basis": plan["split_basis"], "renderer": renderer, "unique_inputs": independent,
            "physical_trajectories": len({r["construction"]["physical_identity"] for r in rows}),
            "category_quotas": dict(Counter(d["scene"]["category"] for d in declarations)),
            "splits": dict(Counter(d["split"] for d in declarations)),
            "axis_quotas": dict(Counter(d["scene"]["shift"] for d in declarations)),
            "difficulty_by_category": {c: dict(Counter(str(r["difficulty"]) for r in rows if r["category"] == c and r["variant"] == "violating")) for c in FAMILIES},
            "category_label_difficulty": {mode: dict(Counter('/'.join((r["category"], r["targets"][mode]["judgment"], str(r["difficulty"]))) for r in rows)) for mode in ("gate", "critic")},
            "eligible_violations": {mode: {subset: sum(r["population_membership"][mode][subset] for r in rows)
                for subset in ("certificate", "four_target")} for mode in ("gate", "critic")},
            "continuous_boundary_coverage": dict(Counter(r["category"] for r in rows if r["variant"] == "legal" and r["boundary"]["near"])),
            "roles": dict(Counter(role for r in rows for role in r["roles"])),
            "family_audit": family_audit, "max_edits": 6, "repair_budget": spec.get("repair_budget", 4096),
            "weighting": "unique complete inputs; same-label views, labels, families, categories, modes",
            "files": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p != marker}}
        (out / "data_card.json").write_text(json.dumps(card, indent=2) + "\n")
        populations[name] = {"manifest": str(manifest), "data_card": str(out / "data_card.json"), "families": families, "events": len(rows)}
        audit_inputs.append((name, rows, out))
    isolation = audit_isolation(audit_inputs)
    (root / "isolation.json").write_text(json.dumps(isolation, indent=2) + "\n")
    for population in populations.values():
        path = Path(population["data_card"])
        card = json.loads(path.read_text())
        card["isolation"] = isolation
        if not split:
            card["files"]["isolation.json"] = sha256(root / "isolation.json")
        path.write_text(json.dumps(card, indent=2) + "\n")
    if not isolation["isolated"]:
        raise ValueError("mechanism family or complete media content crosses a population/split boundary")
    marker.unlink()
    return populations
