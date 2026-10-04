"""Quota-matched training-family selection."""
from collections import defaultdict
import random


def select_families(records, quotas, strategy="random", seed=0, scores=None):
    if strategy not in {"random", "score"}:
        raise ValueError("unknown selection strategy")
    groups = defaultdict(dict)
    for row in records:
        if row["split"] != "train":
            continue
        key = (row.get("source", "default"), row["category"])
        groups[key][row["family_id"]] = row
    rng = random.Random(seed)
    selected = set()
    for key, quota in sorted(quotas.items()):
        families = sorted(groups[key])
        if quota < 0 or quota > len(families):
            raise ValueError("quota exceeds training population")
        if strategy == "score":
            if scores is None or any(f not in scores for f in families):
                raise ValueError("selection scores do not cover the population")
            families.sort(key=lambda f: (-scores[f], f))
            chosen = families[:quota]
        else:
            chosen = rng.sample(families, quota)
        selected.update(chosen)
    return [row for row in records if row["family_id"] in selected]
