"""Shared adapter training with deterministic family exposure and resumable state."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .data import read_events, select_population
from .schema import fingerprint


EXPOSURE_VERSION = "category-family-label-input-v3"


def read_config(path):
    config = json.loads(Path(path).read_text())
    for key in ["events", "model", "processor", "output"]:
        if not config.get(key):
            raise ValueError(f"missing configuration key {key}")
    if config.get("modes", ["gate", "critic"]) not in [["gate"], ["gate", "critic"]]:
        raise ValueError("modes must be Gate-only or shared Gate/Critic")
    if config.get("effective_batch", 16) < 1 or config.get("epochs", 2) < 1:
        raise ValueError("positive family batch and epochs required")
    if config.get("max_steps") is not None and config["max_steps"] < 1:
        raise ValueError("max_steps must be positive")
    if config.get("population_subset", "all") not in {"all", "certificate", "four_target"}:
        raise ValueError("unknown population subset")
    if config.get("population_modes", ["gate", "critic"]) not in [["gate"], ["gate", "critic"]]:
        raise ValueError("population eligibility modes must be Gate-only or joint")
    if config.get("validation_interval", 1) < 1:
        raise ValueError("validation interval must be positive")
    if config.get("validation_families", 1) < 1:
        raise ValueError("validation family limit must be positive")
    if config.get("supervision") == "minimal":
        raise ValueError("minimal is not a distinct supervision format; use certificate")
    if config.get("supervision", "certificate") not in {"certificate", "redundant", "binary", "scalar", "free_form", "multi_head"}:
        raise ValueError("unknown supervision format")
    if config.get("rank", 64) < 1 or config.get("alpha", 128) <= 0 or not 0 <= config.get("dropout", 0.05) < 1:
        raise ValueError("invalid adapter configuration")
    from .targets import FIELDS
    if any(f not in FIELDS or f == "judgment" for f in config.get("removed_fields", [])):
        raise ValueError("invalid removed field")
    if config.get("removed_fields") and config.get("supervision", "certificate") not in {"certificate", "redundant"}:
        raise ValueError("field masking requires certificate supervision")
    return config


def family_schedule(rows, config):
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            groups[row["family_id"]].append(row)
    if not groups:
        raise ValueError("training population is empty")
    batches = []
    size = config.get("effective_batch", 16)
    for epoch in range(config.get("epochs", 2)):
        families = sorted(groups)
        random.Random(config.get("seed", 0) + epoch).shuffle(families)
        for start in range(0, len(families), size):
            for mode in config.get("modes", ["gate", "critic"]):
                batches.append((epoch, mode, [groups[k] for k in families[start:start + size]]))
    return batches


def training_plan(config):
    from .weighting import weighted_events
    rows = select_population(read_events(config["events"]), config.get("population_subset", "all"), config.get("population_modes", ["gate", "critic"]))
    schedule = family_schedule(rows, config)
    root, cache = Path(config["events"]).parent, {}
    training = [r for r in rows if r["split"] == "train"]
    exposure = {mode: weighted_events(training, mode, root, config, cache) for mode in config.get("modes", ["gate", "critic"])}
    return {"events": len(rows), "training_families": len({r["family_id"] for r in rows if r["split"] == "train"}),
            "mode_updates": len(schedule), "modes": config.get("modes", ["gate", "critic"]),
            "evidence_kinds": sorted({r.get("evidence_kind", "unspecified") for r in rows}),
            "model_exists": Path(config["model"]).is_dir(),
            "processor_exists": Path(config["processor"]).is_dir(),
            "exposure_version": EXPOSURE_VERSION,
            "unique_training_inputs": {mode: len(values) for mode, values in exposure.items()},
            "weight_totals": {mode: sum(w for _, w in values) for mode, values in exposure.items()}}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def base_identity(path):
    root = Path(path)
    files = [root / "config.json"]
    index = root / "model.safetensors.index.json"
    if index.exists():
        files.append(index)
        files += [root / name for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
    else:
        files += list(root.glob("*.safetensors"))
    if len(files) < 2:
        raise ValueError("model config or safetensors weights missing")
    if any(not p.resolve().is_relative_to(root.resolve()) for p in files):
        raise ValueError("weight index leaves the model directory")
    return {str(p.relative_to(root)): sha256(p) for p in files}


def processor_identity(path):
    root = Path(path)
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in {".json", ".txt", ".model", ".jinja"})
    if not files:
        raise ValueError("processor metadata is missing")
    return {p.name: sha256(p) for p in files}


def media_identity(rows, manifest):
    root = Path(manifest).parent.resolve()
    paths = sorted({frame["path"] for row in rows for mode in ("gate", "critic") for frame in row["observations"][mode]})
    resolved = {path: confined_media_path(root, path) for path in paths}
    return fingerprint({path: sha256(file) for path, file in resolved.items()})


def confined_media_path(root, path):
    root = Path(root).resolve()
    file = (root / path).resolve()
    if not file.is_relative_to(root):
        raise ValueError("media path leaves manifest root")
    return file


def move_batch(batch, device, dtype):
    return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
            for k, v in batch.items()}


def load_reasoner(path, device, dtype):
    from transformers import Qwen3VLForConditionalGeneration
    placement = {"device_map": {"": device}, "low_cpu_mem_usage": True} if device.startswith("cuda") else {}
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, local_files_only=True,
            dtype=dtype, attn_implementation="sdpa", **placement)
    return model if placement else model.to(device)


def run_training(config, resume=None):
    import torch
    from transformers import AutoProcessor
    from .encoding import encode_training_views
    from .lora import inject_lora, adapter_state, load_adapter_state
    from .loss import certificate_loss
    from .weighting import weighted_events
    rows = select_population(read_events(config["events"]), config.get("population_subset", "all"), config.get("population_modes", ["gate", "critic"]))
    schedule = family_schedule(rows, config)
    root = Path(config["events"]).parent
    modes = config.get("modes", ["gate", "critic"])
    identity_cache = {}
    training_rows = [r for r in rows if r["split"] == "train"]
    exposure = {mode: {r["id"]: weight for r, weight in weighted_events(training_rows, mode, root, config, identity_cache)}
                for mode in modes}
    exposure_identity = fingerprint(exposure)
    family_count = len({r["family_id"] for r in training_rows})
    dev_rows = [r for r in rows if r["split"] == "dev"]
    dev_ids = sorted({r["family_id"] for r in dev_rows})
    selected = [f for f in dev_ids if int(hashlib.sha256(f.encode()).hexdigest()[:2], 16) % 2 == 0] or dev_ids[:1]
    selected = selected[:config.get("validation_families", len(selected))]
    dev = [r for r in dev_rows if r["family_id"] in selected]
    development = {mode: weighted_events(dev, mode, root, config, identity_cache) for mode in modes}
    device = config.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --plan or the CPU smoke suite")
    seed = config.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    identity = base_identity(config["model"])
    processor_hashes = processor_identity(config["processor"])
    media_binding = media_identity(rows, config["events"])
    config_identity = {k: v for k, v in config.items() if k not in {"output", "max_steps"}}
    binding = fingerprint({"config": config_identity, "events": sha256(config["events"]), "base": identity,
                           "processor": processor_hashes, "media": media_binding,
                           "exposure_version": EXPOSURE_VERSION, "exposure_identity": exposure_identity})
    state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if state and state["binding"] != binding:
        raise ValueError("resume model, data, processor, or configuration identity changed")
    output = Path(config["output"])
    if output.exists() and resume is None:
        raise FileExistsError("output exists; provide an explicit resume checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(config["processor"], local_files_only=True,
                                            use_fast=True)
    model = load_reasoner(config["model"], device, dtype)
    targets = inject_lora(model, config.get("rank", 64), config.get("alpha", 128), config.get("dropout", 0.05))
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                 lr=config.get("learning_rate", 1e-4), betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=config.get("weight_decay", 0.01))
    start = 0
    best = math.inf
    if resume:
        load_adapter_state(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["next_step"], state["best_dev_loss"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    total = len(schedule)
    stop = min(total, config.get("max_steps") or total)
    validation_interval = config.get("validation_interval", max(8, total // 4))
    warmup = max(1, math.ceil(total * 0.05))

    def checkpoint(step, name):
        state = {"binding": binding, "adapter": adapter_state(model), "optimizer": optimizer.state_dict(),
                 "next_step": step, "best_dev_loss": best, "torch_rng": torch.get_rng_state(),
                 "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                 "base_identity": identity, "processor_identity": processor_hashes, "media_identity": media_binding,
                 "targets": targets, "config": config,
                 "exposure_version": EXPOSURE_VERSION, "exposure_identity": exposure_identity}
        temp = output / f".{name}.tmp"
        torch.save(state, temp)
        temp.replace(output / name)

    completed = start
    for step in range(start, total):
        if config.get("max_steps") is not None and step >= config["max_steps"]:
            break
        epoch, mode, families = schedule[step]
        factor = (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
        for group in optimizer.param_groups:
            group["lr"] = config.get("learning_rate", 1e-4) * factor
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        t0 = time.perf_counter()
        for family in families:
            for row in family:
                row_weight = exposure[mode].get(row["id"], 0)
                if not row_weight:
                    continue
                views = encode_training_views(row, mode, root, processor, config)
                counts = [int(view["labels"][:, 1:].ne(-100).sum()) for view in views]
                for view, count in zip(views, counts):
                    batch = move_batch(view, device, dtype)
                    labels = batch.pop("labels")
                    logits = model(**batch).logits
                    loss = certificate_loss(logits, labels) * count / sum(counts) * row_weight * math.ceil(family_count / config.get("effective_batch", 16)) / len(modes)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    loss.backward()
                    loss_value += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite adapter gradient")
        optimizer.step()
        completed = step + 1
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        record = {"step": step + 1, "epoch": epoch, "mode": mode, "loss": loss_value,
                  "family_ids": [f[0]["family_id"] for f in families], "seconds": time.perf_counter() - t0,
                  "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": float(norm)}
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if ((step + 1) % validation_interval == 0 or step + 1 == stop) and dev:
            model.eval()
            score = 0.0
            with torch.no_grad():
                for view in modes:
                    for row, weight in development[view]:
                        encoded = encode_training_views(row, view, root, processor, config)
                        counts = [int(b["labels"][:, 1:].ne(-100).sum()) for b in encoded]
                        value = 0.0
                        for batch, count in zip(encoded, counts):
                            batch = move_batch(batch, device, dtype)
                            labels = batch.pop("labels")
                            value += float(certificate_loss(model(**batch).logits, labels)) * count / sum(counts)
                        score += value * weight / len(modes)
            if not math.isfinite(score):
                raise FloatingPointError("nonfinite development loss")
            with (output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step + 1, "dev_loss": score,
                    "families": selected, "modes": modes}) + "\n")
            if score < best:
                best = score
                checkpoint(step + 1, "best.pt")
        checkpoint(step + 1, "latest.pt")
    return {"completed_updates": completed, "output": str(output),
            "best_checkpoint": str(output / "best.pt") if (output / "best.pt").exists() else None,
            "validation_status": "available" if dev else "no_development_families",
            "validation_interval": validation_interval,
            "binding": binding, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
