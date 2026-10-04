"""Recognition evaluation with family grouping and preserved raw predictions."""
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from .metrics import balanced_accuracy, paired_interval
from .schema import Query, canonical, fingerprint
from .targets import FIELDS


def evaluate_events(records, predictor, output, split="test", modes=("gate", "critic"),
                    root=None, seed=None, arm="model"):
    output = Path(output)
    if output.exists():
        raise FileExistsError("evaluation output exists")
    rows = [r for r in records if r["split"] == split]
    if not rows or any(mode not in {"gate", "critic"} for mode in modes):
        raise ValueError("nonempty evaluation population and permitted modes required")
    if root is not None:
        from .train import confined_media_path, sha256
        paths = {frame["path"] for row in rows for view in row["observations"].values() for frame in view}
        resolved = {path: confined_media_path(root, path) for path in paths}
        media_hashes = {path: sha256(file) for path, file in resolved.items()}
    else:
        media_hashes = {}
    output.mkdir(parents=True)
    readings = []
    contexts = {}
    input_cache, prediction_cache = {}, {}
    with (output / "predictions.jsonl").open("w") as stream:
        for row in rows:
            for mode in modes:
                if root is not None:
                    from .weighting import input_identity
                    identity = input_identity(row, mode, root, getattr(predictor, "config", {}), input_cache)
                else:
                    identity = None
                cache_key = (mode, row["family_id"], identity or row["id"])
                reused = cache_key in prediction_cache
                prediction = prediction_cache.get(cache_key)
                if prediction is None:
                    prediction = predictor.predict_event(row, mode)
                    prediction_cache[cache_key] = prediction
                if not math.isfinite(prediction.risk) or not 0 <= prediction.risk <= 1:
                    raise ValueError("prediction risk must be a finite probability")
                assessment_start = time.perf_counter()
                record = {"id": row["id"], "family_id": row["family_id"], "category": row["category"],
                    "mode": mode, "query_binding": Query(**row["query"]).binding,
                    "truth": row["targets"][mode]["judgment"], "prediction": prediction.certificate.judgment.value,
                    "risk": prediction.risk, "target": row["targets"][mode],
                    "certificate": prediction.certificate.target(), "cost": asdict(prediction.cost)}
                record["input_identity"], record["prediction_reused"] = identity, reused
                if reused:
                    record["cost"] = {**record["cost"], "compute": 0.0, "wall_seconds": 0.0, "nfe": 0}
                record.update({"seed": seed if seed is not None else getattr(predictor, "config", {}).get("seed"),
                    "arm": arm, "split": split, "domain": row.get("domain", "unspecified"),
                    "ood_axis": row.get("ood_axis", "unspecified"), "variant": row.get("variant"),
                    "difficulty": row.get("difficulty"), "evidence_kind": row.get("evidence_kind")})
                record["event_binding"] = fingerprint({"query": row["query"], "commitment": row["commitment"],
                    "observations": row["observations"], "targets": row["targets"],
                    "media": {f["path"]: media_hashes.get(f["path"]) for view in row["observations"].values() for f in view}})
                if root is not None and row.get("construction"):
                    from .sdg import load_context
                    from .assessment import assess_prediction
                    if row["construction"].get("evidence_version") in {2, 3} and row["family_id"] not in contexts:
                        contexts[row["family_id"]] = load_context(row, root)
                    record["assessment"] = assess_prediction(row, mode, prediction.certificate, root, contexts.get(row["family_id"]))
                    if record["assessment"]["status"] == "assessed" and record["assessment"]["truth"] != record["truth"]:
                        raise ValueError("replay truth differs from the target judgment")
                record["posthoc_wall_seconds"] = time.perf_counter() - assessment_start
                readings.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
    summary = {}
    for mode in modes:
        group = [r for r in readings if r["mode"] == mode]
        try:
            accuracy = balanced_accuracy(group)
            unavailable = None
        except ValueError as error:
            accuracy, unavailable = None, str(error)
        fields = {field: grouped_mean([r for r in group if r["target"].get(field) is not None],
            lambda r, field=field: canonical(r["target"][field]) == canonical(r["certificate"].get(field))) for field in FIELDS}
        summary[mode] = {"events": len(group), "families": len({r["family_id"] for r in group}),
            "balanced_accuracy": accuracy, "balanced_accuracy_unavailable": unavailable,
            "unknown_targets": sum(r["truth"] == "unknown" for r in group),
            "unknown_predictions": sum(r["prediction"] == "unknown" for r in group),
            "field_exact_match": fields,
            "compute": sum(r["cost"]["compute"] for r in group),
            "wall_seconds": sum(r["cost"]["wall_seconds"] for r in group)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def grouped_mean(rows, value):
    from .weighting import unique_readings
    rows = unique_readings(rows)
    families = defaultdict(list)
    for row in rows:
        families[row["category"], row["family_id"], row["truth"]].append(float(value(row)))
    label_means = defaultdict(list)
    for (category, family, truth), values in families.items():
        label_means[category, family].append(sum(values) / len(values))
    categories = defaultdict(list)
    for (category, family), values in label_means.items():
        categories[category].append(sum(values) / len(values))
    return sum(sum(v) / len(v) for v in categories.values()) / len(categories) if categories else None


def summarize(rows):
    from .data import FAMILIES
    try:
        accuracy, unavailable = balanced_accuracy(rows), None
    except ValueError as error:
        accuracy, unavailable = None, str(error)
    assessed = [r for r in rows if r.get("assessment", {}).get("status") == "assessed"]
    eligible = [r for r in assessed if r["truth"] == "invalid" and r["assessment"]["eligible"]]
    invalid = [r for r in rows if r["truth"] == "invalid"]
    minimal_units = [r for r in eligible if r["assessment"]["repair_valid"]]
    units = defaultdict(lambda: {"compute": 0.0, "wall_seconds": 0.0})
    for row in rows:
        cost = row.get("cost", {})
        unit = units[cost.get("unit", "unspecified")]
        unit["compute"] += cost.get("compute", 0.0)
        unit["wall_seconds"] += cost.get("wall_seconds", 0.0)
    joint = lambda r: r["prediction"] == "invalid" and r["assessment"]["witness_valid"] and r["assessment"]["repair_valid"]
    fields = {f: grouped_mean([r for r in rows if r.get("target", {}).get(f) is not None],
              lambda r, f=f: canonical(r["target"][f]) == canonical(r.get("certificate", {}).get(f))) for f in FIELDS}
    return {"events": len(rows), "families": len({r["family_id"] for r in rows}),
        "categories": sorted({r["category"] for r in rows}),
        "paper_categories_complete": set(r["category"] for r in rows) == set(FAMILIES),
        "balanced_accuracy": accuracy, "balanced_accuracy_unavailable": unavailable,
        "J_valid": grouped_mean(eligible, joint),
        "J_cert": grouped_mean(eligible, lambda r: joint(r) and r["assessment"]["minimal"]),
        "repair": grouped_mean(eligible, lambda r: r["prediction"] == "invalid" and r["assessment"]["repair_valid"]),
        "minimality_given_valid_repair": grouped_mean(minimal_units, lambda r: r["assessment"]["minimal"]),
        "eligible_violations": len(eligible), "complete_truth_violations": len(invalid),
        "assessed_events": len(assessed), "quality_unavailable_events": len(rows) - len(assessed),
        "unknown_targets": sum(r["truth"] == "unknown" for r in rows),
        "unknown_predictions": sum(r["prediction"] == "unknown" for r in rows),
        "posthoc_wall_seconds": sum(r.get("posthoc_wall_seconds", 0) for r in rows),
        "field_exact_match": fields, "compute_by_unit": dict(units)}


def seed_summary(rows):
    result = summarize(rows)
    axes = {"geometry", "dynamics", "appearance", "composition"}
    ood = [r for r in rows if r.get("domain") == "ood"]
    slices = [summarize([r for r in ood if r.get("ood_axis") == axis]) for axis in sorted(axes)]
    for field in ("balanced_accuracy", "J_valid", "J_cert"):
        values = [s[field] for s in slices]
        result["OOD_" + field] = sum(values) / 4 if all(v is not None for v in values) else None
    return result


def inventory_diagnostics(rows):
    independent = [r for r in rows if r.get("assessment", {}).get("verification_role") == "arm_invariant_inventory_diagnostic"]
    unique = {}
    for row in independent:
        key = (row["mode"], row.get("seed"), row["id"], row.get("event_binding"))
        assessment = row["assessment"]
        signature = (row["family_id"], row["category"], row["truth"], row["query_binding"],
                     assessment["verification"], assessment["cancelled"], assessment["verification_budget"],
                     assessment["label_source"])
        if key in unique and signature != unique[key][1]:
            raise ValueError("arm-invariant inventory diagnostics disagree across arms")
        unique[key] = row, signature
    def summary(population):
        valid = [r for r in population if r["truth"] == "valid"]
        bad = [r for r in population if r["truth"] == "invalid"]
        return {"events": len(population),
            "FR": grouped_mean(valid, lambda r: r["assessment"]["cancelled"]),
            "MR": grouped_mean(bad, lambda r: not r["assessment"]["cancelled"]),
            "resolution": grouped_mean(population, lambda r: r["assessment"]["verification"] in {"V", "S", "N"})}
    groups = {}
    for mode in sorted({r["mode"] for r, _ in unique.values()}):
        population = [r for r, _ in unique.values() if r["mode"] == mode]
        per_seed = {str(seed): summary([r for r in population if r.get("seed") == seed])
                    for seed in sorted({r.get("seed") for r in population}, key=str)}
        overall = summary(population)
        for metric in ("FR", "MR", "resolution"):
            values = [s[metric] for s in per_seed.values()]
            overall[metric] = sum(values) / len(values) if all(v is not None for v in values) else None
        slices = {name: {str(value): summary([r for r in population if r.get(name) == value])
                        for value in sorted({r.get(name) for r in population}, key=str)}
                  for name in ("domain", "ood_axis", "category", "difficulty", "variant")}
        groups[mode] = {"overall": overall, "per_seed": per_seed, "slices": slices}
    return {"arm_invariant": True, "rule": "certificate-free fixed-inventory checks; deduplicated across arms",
            "unavailable_readings": len(rows) - len(independent), "groups": groups}


def report(predictions, output=None, compare=None, repeats=10000, bootstrap_seed=0):
    paths = [predictions] if isinstance(predictions, (str, Path)) else list(predictions)
    rows = [json.loads(line) for path in paths for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("prediction population is empty")
    seen = set()
    for row in rows:
        for key in ("id", "family_id", "category", "mode", "truth", "prediction", "query_binding"):
            if not row.get(key):
                raise ValueError("prediction record missing " + key)
        row.setdefault("arm", "model")
        key = (row["arm"], row.get("seed"), row["mode"], row["id"])
        if key in seen:
            raise ValueError("duplicate arm/seed/mode/event reading")
        seen.add(key)
        if row["truth"] not in {"valid", "invalid", "unknown"} or row["prediction"] not in {"valid", "invalid", "unknown"}:
            raise ValueError("invalid judgment reading")
        cost = row.get("cost", {})
        if any(not math.isfinite(float(cost.get(k, 0))) or float(cost.get(k, 0)) < 0 for k in ("compute", "wall_seconds")):
            raise ValueError("invalid compute reading")
    groups = defaultdict(list)
    for row in rows:
        groups[row["arm"], row["mode"]].append(row)
    summary = {}
    for (arm, mode), values in sorted(groups.items()):
        seeds = sorted({r.get("seed") for r in values}, key=str)
        per_seed = {str(seed): seed_summary([r for r in values if r.get("seed") == seed]) for seed in seeds}
        slices = {}
        for name in ("domain", "ood_axis", "category", "difficulty", "variant"):
            slices[name] = {str(v): summarize([r for r in values if r.get(name) == v])
                            for v in sorted({r.get(name) for r in values}, key=str)}
        overall = summarize(values)
        for metric in ("balanced_accuracy", "J_valid", "J_cert",
                       "OOD_balanced_accuracy", "OOD_J_valid", "OOD_J_cert"):
            scores = [s[metric] for s in per_seed.values()]
            overall[metric] = sum(scores) / len(scores) if all(s is not None for s in scores) else None
        summary[f"{arm}/{mode}"] = {"overall": overall, "per_seed": per_seed, "slices": slices}
    contrasts = {}
    if compare:
        left, right = compare
        if left == right or left not in {r["arm"] for r in rows} or right not in {r["arm"] for r in rows}:
            raise ValueError("two distinct observed arms required")
        for mode in sorted({r["mode"] for r in rows}):
            a = [r for r in rows if r["arm"] == left and r["mode"] == mode]
            b = [r for r in rows if r["arm"] == right and r["mode"] == mode]
            identity = lambda r: (r.get("seed"), r["id"], r["family_id"], r["category"], r["truth"], r["query_binding"],
                                  r.get("domain"), r.get("ood_axis"), r.get("event_binding"))
            if {identity(r) for r in a} != {identity(r) for r in b} or not a:
                raise ValueError("paired arms differ in seed or evaluation population")
            seeds = sorted({r.get("seed") for r in a}, key=str)
            if None in seeds or len(seeds) < 2:
                contrasts[mode] = {"unavailable": "at least two identified paired training seeds required"}
                continue
            metrics = {}
            for metric in ("balanced_accuracy", "J_valid", "J_cert",
                           "OOD_balanced_accuracy", "OOD_J_valid", "OOD_J_cert"):
                pairs = [(seed_summary([r for r in a if r.get("seed") == seed])[metric],
                          seed_summary([r for r in b if r.get("seed") == seed])[metric]) for seed in seeds]
                metrics[metric] = paired_interval(pairs, repeats, bootstrap_seed) if all(x is not None and y is not None for x, y in pairs) else None
            contrasts[mode] = {"unit": "paired_training_seed", "arms": [left, right], "metrics": metrics}
    result = {"version": 3, "inputs": [str(p) for p in paths], "groups": summary, "paired": contrasts,
              "aggregation": "unique input; same-label views, labels, families, categories, paired seeds",
              "inventory_diagnostics": inventory_diagnostics(rows),
              "quality_rule": "independent physical witness and fixed-context repair subsets; unavailable without assessment"}
    if output:
        path = Path(output)
        if path.exists():
            raise FileExistsError("report destination exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
