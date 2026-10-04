"""Complete-input identities and category/family/label-balanced exposure."""
from collections import defaultdict
from pathlib import Path

from .data import model_input
from .schema import fingerprint


def input_identity(row, mode, root, config=None, cache=None):
    import hashlib
    config, cache = config or {}, cache if cache is not None else {}
    payload, paths = model_input(row, mode, root, config.get("restricted_critic", False) and mode == "critic")
    hashes = []
    for name in paths:
        file = Path(name)
        if file not in cache:
            try:
                from PIL import Image
                with Image.open(file) as image:
                    rgb = image.convert("RGB")
                    cache[file] = fingerprint({"size": rgb.size, "rgb": hashlib.sha256(rgb.tobytes()).hexdigest()})
            except ImportError:
                cache[file] = hashlib.sha256(file.read_bytes()).hexdigest()
        hashes.append(cache[file])
    return fingerprint({"payload": payload, "frames": hashes,
        "preprocessing": {key: config.get(key) for key in
                          ("longest_side", "fps", "processor", "restricted_critic")}})


def weighted_events(rows, mode, root, config=None, cache=None):
    config, cache = config or {}, cache if cache is not None else {}
    unique, family_categories, roots = {}, {}, {}
    for row in sorted(rows, key=lambda r: r["id"]):
        family, category = row["family_id"], row["category"]
        if family in family_categories and family_categories[family] != category:
            raise ValueError("a family crosses categories")
        family_categories[family] = category
        identity = input_identity(row, mode, root, config, cache)
        if identity in roots and roots[identity] != family:
            raise ValueError("identical model input crosses root families; merge its lineage before splitting")
        roots[identity] = family
        key = family, identity
        target = row["targets"][mode]
        if config.get("supervision") == "redundant" and not row.get("redundant_available", {}).get(mode, True):
            raise ValueError("no admissible redundant repair within six edits; use a registered compatible population")
        if key in unique:
            previous = unique[key]
            if fingerprint(previous["targets"][mode]) != fingerprint(target):
                raise ValueError("identical model input has incompatible canonical targets")
            continue
        unique[key] = row
    families = defaultdict(lambda: defaultdict(list))
    categories = defaultdict(set)
    for (family, _), row in unique.items():
        families[family][row["targets"][mode]["judgment"]].append(row)
        categories[row["category"]].add(family)
    result = []
    for family, labels in sorted(families.items()):
        category = family_categories[family]
        family_weight = 1 / (len(categories) * len(categories[category]))
        for label, views in sorted(labels.items()):
            for row in views:
                result.append((row, family_weight / (len(labels) * len(views))))
    return result


def unique_readings(rows):
    unique, result = {}, []
    for row in rows:
        identity = row.get("input_identity")
        if identity is None:
            result.append(row)
            continue
        key = row.get("arm"), row.get("seed"), row.get("mode"), row["family_id"], identity
        if key in unique:
            if row["truth"] != unique[key]["truth"] or row["prediction"] != unique[key]["prediction"]:
                raise ValueError("identical evaluated input has inconsistent truth or prediction")
            continue
        unique[key] = row
        result.append(row)
    return result
