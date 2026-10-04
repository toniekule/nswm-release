"""NSWM command line."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="nswm")
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixtures")
    fixture.add_argument("output")
    fixture.add_argument("--families", type=int, default=50)
    fixture.add_argument("--seed", type=int, default=0)
    audit = commands.add_parser("audit-data")
    audit.add_argument("events")
    train = commands.add_parser("train")
    train.add_argument("config")
    train.add_argument("--plan", action="store_true")
    train.add_argument("--resume")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("events")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("output")
    evaluate.add_argument("--split", choices=["dev", "test"], default="test")
    evaluate.add_argument("--modes", nargs="+", choices=["gate", "critic"], default=["gate", "critic"])
    evaluate.add_argument("--arm", default="model")
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--population-subset", choices=["all", "certificate", "four_target"], default="all")
    report_cmd = commands.add_parser("report")
    report_cmd.add_argument("predictions", nargs="+")
    report_cmd.add_argument("--out")
    report_cmd.add_argument("--compare", nargs=2)
    report_cmd.add_argument("--repeats", type=int, default=10000)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--role", choices=["cosmos", "processor", "reasoner"], required=True)
    convert = commands.add_parser("convert-base")
    convert.add_argument("--source", required=True)
    convert.add_argument("--processor", required=True)
    convert.add_argument("--out", required=True)
    convert.add_argument("--shard-mib", type=int, default=128)
    plan = commands.add_parser("plan-demo")
    plan.add_argument("--policy", default="certificate")
    plan.add_argument("--budget", type=float, default=10.0)
    resources = commands.add_parser("resources")
    resources.add_argument("config")
    resources.add_argument("--probe-device", action="store_true")
    tiny = commands.add_parser("make-tiny")
    tiny.add_argument("output")
    tiny.add_argument("--seed", type=int, default=0)
    tiny.add_argument("--families", type=int, default=20)
    corpus = commands.add_parser("corpus")
    corpus.add_argument("--spec", required=True)
    corpus.add_argument("--out")
    corpus.add_argument("--plan", action="store_true")
    corpus.add_argument("--families", type=int, default=50)
    corpus.add_argument("--seed", type=int, default=0)
    corpus.add_argument("--split-spec")
    corpus.add_argument("--renderer", choices=["software", "mujoco"], default="software")
    robot_register = commands.add_parser("register-robot")
    robot_register.add_argument("--assets", required=True)
    robot_register.add_argument("--xml", required=True)
    robot_register.add_argument("--repository", required=True)
    robot_register.add_argument("--revision", required=True)
    robot_register.add_argument("--licenses", required=True)
    robot_register.add_argument("--out", required=True)
    robot_check = commands.add_parser("robot-check")
    robot_check.add_argument("--assets", required=True)
    robot_check.add_argument("--receipt", required=True)
    robot_check.add_argument("--spec", required=True)
    args = parser.parse_args()
    if args.command == "register-robot":
        from .robots import register_robot_assets
        result = register_robot_assets(args.assets, args.xml, args.repository, args.revision,
            json.loads(Path(args.licenses).read_text()), args.out)
    elif args.command == "robot-check":
        from .robots import RobotPositionTask
        task = RobotPositionTask.from_spec(args.spec, args.assets, args.receipt)
        result = {"query_binding": task.query.binding, "asset_identity": task.assets["identity"],
                  "task_status": task.goal_status.value, "reference_status": task.oracle(task.query, task.reference()).value}
    elif args.command == "fetch":
        from .assets import fetch_assets
        result = fetch_assets(args.repo, args.revision, args.out, args.role)
    elif args.command == "convert-base":
        from .assets import convert_base
        result = convert_base(args.source, args.processor, args.out, args.shard_mib)
    elif args.command == "report":
        from .evaluation import report
        result = report(args.predictions, args.out, args.compare, args.repeats)
    elif args.command == "corpus":
        if args.plan:
            from .scenes import read_spec
            from .corpus_v3 import corpus_plan
            spec = read_spec(args.spec)
            if spec["version"] != 3:
                parser.error("corpus --plan requires a version 3 mechanism spec")
            split = json.loads(Path(args.split_spec).read_text()) if args.split_spec else None
            result = corpus_plan(spec, args.families, args.seed, split)
        else:
            if not args.out:
                parser.error("corpus construction requires --out")
            from .sdg import build_corpus
            result = build_corpus(args.spec, args.out, args.families, args.seed, args.split_spec, args.renderer)
    elif args.command == "make-tiny":
        from .tiny import make_tiny
        result = make_tiny(args.output, args.seed, args.families)
    elif args.command == "fixtures":
        from .data import build_fixtures
        result = {"manifest": str(build_fixtures(args.output, args.families, args.seed))}
    elif args.command == "audit-data":
        from .data import read_events
        rows = read_events(args.events)
        result = {"events": len(rows), "families": len({r["family_id"] for r in rows}), "status": "pass"}
    elif args.command == "train":
        from .train import read_config, run_training, training_plan
        cfg = read_config(args.config)
        result = training_plan(cfg) if args.plan else run_training(cfg, args.resume)
    elif args.command == "evaluate":
        from .data import read_events, select_population
        from .evaluation import evaluate_events
        from .predict import SharedPredictor
        predictor = SharedPredictor(args.checkpoint, root=Path(args.events).parent)
        result = evaluate_events(select_population(read_events(args.events), args.population_subset, predictor.config.get("population_modes", ["gate", "critic"])), predictor, args.output, args.split, args.modes,
                                 Path(args.events).parent, args.seed, args.arm)
    elif args.command == "resources":
        from .resources import inspect_resources
        from .train import read_config
        result = inspect_resources(read_config(args.config), args.probe_device)
    else:
        from .demo import planning_demo
        result = planning_demo(args.policy, args.budget)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
