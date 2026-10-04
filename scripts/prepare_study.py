"""Materialize paired-seed training configurations without starting jobs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--checkpoint-root", default="checkpoints/study")
    parser.add_argument("--population-subset", choices=["all", "certificate", "four_target"], default="all")
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError("positive seed count required")
    root = Path(args.output)
    if root.exists():
        raise FileExistsError("study destination exists")
    base = json.loads(Path(args.config).read_text())
    arms = {"shared_certificate": {}, "gate_certificate": {"modes": ["gate"]},
        "shared_binary": {"supervision": "binary"}, "shared_scalar": {"supervision": "scalar"},
        "shared_free_form": {"supervision": "free_form"}, "shared_multi_head": {"supervision": "multi_head"},
        "shared_redundant": {"supervision": "redundant"}, "shared_restricted_critic": {"restricted_critic": True}}
    for field in ("premises", "constraint", "entities", "time", "witness", "scope", "repair"):
        arms[f"shared_remove_{field}"] = {"removed_fields": [field]}
    root.mkdir(parents=True)
    runs = []
    for arm, change in arms.items():
        for seed in range(args.seeds):
            tag = f"{arm}-seed{seed}"
            config = {**base, "modes": ["gate", "critic"], "supervision": "certificate", "removed_fields": [],
                      "restricted_critic": False, **change, "seed": seed,
                      "population_subset": args.population_subset,
                      "population_modes": ["gate", "critic"],
                      "output": str(Path(args.checkpoint_root) / tag)}
            path = root / f"{tag}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            runs.append({"arm": arm, "seed": seed, "config": str(path), "output": config["output"]})
    (root / "manifest.json").write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    print(json.dumps({"arms": len(arms), "seeds": args.seeds, "runs": len(runs), "root": str(root)}))


if __name__ == "__main__":
    main()
