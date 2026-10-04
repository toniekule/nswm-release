"""Content-based observation grouping and cross-population isolation audits."""
from collections import defaultdict
import hashlib

from .data import split_family
from .schema import fingerprint
from .train import confined_media_path, sha256


def observation_hashes(rows, root):
    cache = {}
    clips = {}
    for row in rows:
        for mode, frames in row["observations"].items():
            hashes = []
            for frame in frames:
                file = confined_media_path(root, frame["path"])
                if file not in cache:
                    cache[file] = sha256(file)
                hashes.append(cache[file])
            clips[row["id"], mode] = hashlib.sha256(b"".join(bytes.fromhex(h) for h in hashes)).hexdigest()
    return clips, set(cache.values())


def group_splits(rows, root, seed, train, dev):
    clips, _ = observation_hashes(rows, root)
    parent = {row["family_id"]: row["family_id"] for row in rows}
    def find(family):
        while parent[family] != family:
            parent[family] = parent[parent[family]]
            family = parent[family]
        return family
    seen = {}
    for row in rows:
        for mode in row["observations"]:
            key = mode, clips[row["id"], mode]
            family = find(row["family_id"])
            if key in seen:
                other = find(seen[key])
                parent[max(family, other)] = min(family, other)
            seen[key] = row["family_id"]
    for row in rows:
        group = find(row["family_id"])
        row["split_group"] = group
        row["split"] = split_family(group, seed, train, dev)
    return {"groups": len({find(f) for f in parent}), "families": len(parent),
            "rule": "connected families sharing a complete Gate or Critic clip; SHA256(seed:minimum_family_id)"}


def audit_isolation(populations):
    families, media, frames = defaultdict(set), defaultdict(set), defaultdict(set)
    for name, rows, root in populations:
        clips, individual = observation_hashes(rows, root)
        for row in rows:
            partition = name, row["split"]
            families[row["family_id"]].add(partition)
            for mode in row["observations"]:
                media[mode, clips[row["id"], mode]].add(partition)
        for digest in individual:
            frames[digest].add(name)
    family_leaks = {family: sorted(parts) for family, parts in families.items() if len(parts) > 1}
    clip_leaks = {fingerprint(key): sorted(parts) for key, parts in media.items() if len(parts) > 1}
    return {"isolated": not family_leaks and not clip_leaks, "family_leaks": family_leaks,
            "complete_clip_leaks": clip_leaks, "shared_individual_frames_across_populations": sum(len(p) > 1 for p in frames.values()),
            "criterion": "family IDs and complete ordered observation content stay within one population/split"}
