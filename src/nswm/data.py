"""Family splits, observation isolation, and analytic integration fixtures."""
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import random

from .repair import search_repairs
from .schema import Certificate, Commitment, Edit, Interval, Query, Verdict, commitment_from_dict
from .verify import scalar_oracle


FAMILIES = ("contact_solidity", "conservation", "permanence", "causal_ordering", "resource_reachability")


def split_family(family_id, seed=0, train=0.8, dev=0.1):
    if not (0 < train < 1 and 0 <= dev < 1 and train + dev < 1):
        raise ValueError("invalid split fractions")
    value = int(hashlib.sha256(f"{seed}:{family_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < train else "dev" if value < train + dev else "test"


def read_events(path):
    path = Path(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    family_splits = {}
    ids = set()
    for row in records:
        validate_event(row, path.parent)
        if row["id"] in ids:
            raise ValueError("duplicate event ID")
        ids.add(row["id"])
        if row["family_id"] in family_splits and family_splits[row["family_id"]] != row["split"]:
            raise ValueError("event variants cross data splits")
        family_splits[row["family_id"]] = row["split"]
    return records


def select_population(rows, subset="all", modes=("gate", "critic")):
    if subset not in {"all", "certificate", "four_target"}:
        raise ValueError("unknown registered population subset")
    if subset == "all":
        return rows
    families = {}
    for row in rows:
        families.setdefault(row["family_id"], []).append(row)
    selected = []
    for variants in families.values():
        violations = [r for r in variants if any(r["targets"][m]["judgment"] == "invalid" for m in modes)]
        if violations and all(r.get("population_membership", {}).get(m, {}).get(subset, False)
                              for r in violations for m in modes):
            selected.extend(variants)
    return selected


def validate_event(row, root):
    for key in ["id", "family_id", "split", "query", "commitment", "observations", "targets"]:
        if key not in row:
            raise ValueError(f"missing event key {key}")
    if row["split"] not in {"train", "dev", "test"}:
        raise ValueError("unknown data split")
    if any(not isinstance(row.get(key), str) or not row[key] for key in ("id", "family_id", "category")):
        raise ValueError("event, family and category IDs must be nonempty strings")
    Query(**row["query"])
    commitment_from_dict(row["commitment"])
    for mode in ["gate", "critic"]:
        view = row["observations"][mode]
        if not view or len(view) > (5 if mode == "gate" else 17):
            raise ValueError("invalid observation count")
        timestamps = [frame["time"] for frame in view]
        if any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in timestamps) or timestamps != sorted(timestamps):
            raise ValueError("observations must be time ordered")
        if mode == "gate" and any(t > 0 for t in timestamps):
            raise ValueError("Gate contains candidate future frames")
        for frame in view:
            p = (root / frame["path"]).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file():
                raise ValueError("missing or external observation file")
        target = row["targets"][mode]
        if target["judgment"] not in {v.value for v in Verdict}:
            raise ValueError("invalid target judgment")


def model_input(row, mode, root, restricted_critic=False):
    if mode not in {"critic", "gate"}:
        raise ValueError("unknown learning mode")
    observations = row["observations"]["gate" if restricted_critic else mode]
    times = [f["time"] for f in observations]
    count_limit = 5 if mode == "gate" or restricted_critic else 17
    if (not observations or len(observations) > count_limit
            or any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in times)
            or times != sorted(times) or ((mode == "gate" or restricted_critic) and any(t > 0 for t in times))):
        raise ValueError("invalid permitted observation window")
    root = Path(root).resolve()
    for frame in observations:
        if not (root / frame["path"]).resolve().is_relative_to(root):
            raise ValueError("observation path leaves the permitted root")
    query = Query(**row["query"])
    payload = {"mode": mode, "actions": query.actions, "premises": query.premises,
               "horizon": query.horizon, "commitment": row["commitment"],
               "timestamps": times, "time_origin": observations[0]["time"]}
    return payload, [str((Path(root) / f["path"]).resolve()) for f in observations]


def write_ppm(path, value=0.0, tint=0, size=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = bytearray()
    width = min(size - 4, max(3, int(abs(value) * 12) + 3))
    for y in range(size):
        for x in range(size):
            color = (210, 110 + tint % 50, 40) if 20 <= y < 44 and 3 <= x < width else (20, 30, 45)
            pixels.extend(color)
    path.write_bytes(f"P6\n{size} {size}\n255\n".encode() + pixels)


def build_fixtures(destination, count=50, seed=0):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("fixture destination already exists")
    destination.mkdir(parents=True)
    rng = random.Random(seed)
    records = []
    for i in range(count):
        category = FAMILIES[i % len(FAMILIES)]
        family_id = f"{category}-{i:06d}"
        limit = Decimal(str(round(rng.uniform(0.8, 1.2), 4)))
        measurements = {"capacity": Interval.point(limit)}
        slots = ["demand", "secondary_demand"] if i % 3 == 0 else ["demand"]
        rules = [{"id": f"{category}:{slot}", "slot": slot, "limit": "capacity",
                  "constraint": category, "entities": ["object_A", "object_B"],
                  "time": [0.0, 1.0], "unit": "normalized", "premises_valid": True} for slot in slots]
        query = Query(family_id, ((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),) * 16,
                      {"complete_scalar_domain": True, "capacity": str(limit), "domain": category}, 16)
        history = []
        for f in range(5):
            path = Path("media") / family_id / f"history_{f}.ppm"
            write_ppm(destination / path, float(limit), f)
            history.append({"path": str(path), "time": (f - 4) / 8})
        negative = Commitment({**{s: Interval.point(limit + Decimal("0.3")) for s in slots}, "slack": Interval.point(0)})
        catalogue = tuple(Edit(s, Interval.point(limit - Decimal("0.2"))) for s in slots)
        repair = search_repairs(query, negative, catalogue,
                                lambda q, z: scalar_oracle(q, z, measurements, rules))
        for variant in ["legal", "violating", "repaired", "appearance", "boundary_legal"]:
            z = negative if variant in {"violating", "appearance"} else negative.edited(catalogue)
            if variant == "boundary_legal":
                z = Commitment({**{s: Interval.point(limit - Decimal("0.0001")) for s in slots}, "slack": Interval.point(0)})
            verdict = scalar_oracle(query, z, measurements, rules)
            if verdict == Verdict.INVALID:
                cert = Certificate(verdict, dict(query.premises), category, ("object_A", "object_B"),
                                   (0.0, 1.0), {"residual": "0.3", "unit": "normalized"},
                                   z, repair.repairs[0])
            else:
                cert = Certificate(verdict, dict(query.premises), repair=())
            video = []
            for f in range(17):
                path = Path("media") / family_id / variant / f"frame_{f}.ppm"
                write_ppm(destination / path, float(next(iter(z.slots.values())).lower) * f / 16,
                          30 if variant == "appearance" else 0)
                video.append({"path": str(path), "time": f / 16})
            target = cert.target()
            target["scalar_residual"] = str(next(iter(z.slots.values())).lower - limit)
            redundant = dict(target)
            if verdict == Verdict.INVALID:
                redundant["repair"] = [asdict(e) for e in (*repair.repairs[0], Edit("slack", Interval.point("0.1")))]
            records.append({"id": f"{family_id}/{variant}", "family_id": family_id,
                            "category": category, "variant": variant,
                            "split": split_family(family_id, seed), "query": asdict(query),
                            "commitment": asdict(z), "observations": {"gate": history, "critic": video},
                            "targets": {"gate": target, "critic": target},
                            "redundant_targets": {"gate": redundant, "critic": redundant},
                            "evidence_kind": "analytic_fixture", "rules": rules,
                            "measurements": {k: asdict(v) for k, v in measurements.items()}})
    manifest = destination / "events.jsonl"
    manifest.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records))
    read_events(manifest)
    return manifest
