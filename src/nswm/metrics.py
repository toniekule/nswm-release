"""Grouped recognition metrics and paired family/seed resampling."""
from collections import defaultdict
import math
import random


def balanced_accuracy(rows):
    from .weighting import unique_readings
    rows = unique_readings(rows)
    groups = defaultdict(list)
    for row in rows:
        if row["truth"] in {"valid", "invalid"}:
            groups[(row["category"], row["family_id"], row["truth"])].append(row["prediction"] == row["truth"])
    cells = defaultdict(list)
    for (category, family, label), values in groups.items():
        cells[(category, label)].append(sum(values) / len(values))
    categories = sorted({k[0] for k in cells})
    if not categories or any((c, label) not in cells for c in categories for label in ["valid", "invalid"]):
        raise ValueError("balanced accuracy requires both labels in every category")
    scores = [sum(sum(cells[c, label]) / len(cells[c, label]) for label in ["valid", "invalid"]) / 2 for c in categories]
    return sum(scores) / len(scores)


def paired_interval(pairs, repeats=10000, seed=0):
    if len(pairs) < 2 or repeats < 100:
        raise ValueError("paired intervals require at least two independent units")
    differences = [float(a) - float(b) for a, b in pairs]
    if not all(math.isfinite(x) for x in differences):
        raise ValueError("nonfinite paired readings")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(repeats))
    return {"mean": sum(differences) / len(differences), "lower": means[int(repeats * 0.025)],
            "upper": means[min(repeats - 1, int(repeats * 0.975))], "units": len(pairs), "repeats": repeats}


def rejection_metrics(rows):
    feasible = [r for r in rows if r["truth"] == "valid"]
    infeasible = [r for r in rows if r["truth"] == "invalid"]
    return {"feasible_rejection": sum(r["cancelled"] for r in feasible) / len(feasible) if feasible else None,
            "infeasible_retention": sum(not r["cancelled"] for r in infeasible) / len(infeasible) if infeasible else None,
            "unresolved": sum(r["status"] == "X" for r in rows),
            "oracle_unknown": sum(r["truth"] == "unknown" for r in rows)}
