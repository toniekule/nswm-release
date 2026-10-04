"""Explicit, revision-pinned downloads and strict streaming reasoner conversion."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil

from .train import base_identity, processor_identity, sha256


def fetch_assets(repo, revision, destination, role):
    from huggingface_hub import snapshot_download
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or role not in {"cosmos", "processor", "reasoner"}:
        raise ValueError("a full immutable Hub commit and a supported asset role are required")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("asset destination exists")
    patterns = ["*.json", "*.txt", "*.model", "*.jinja", "README.md", "LICENSE*"]
    if role == "cosmos":
        patterns += ["transformer/*.json", "transformer/*.safetensors", "vision_encoder/*.json", "vision_encoder/*.safetensors"]
    elif role == "reasoner":
        patterns += ["*.safetensors"]
    snapshot_download(repo_id=repo, revision=revision, local_dir=destination, allow_patterns=patterns)
    files = {str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob("*"))
             if p.is_file() and ".cache" not in p.parts}
    receipt = {"repo": repo, "revision": revision, "role": role, "files": files}
    (destination / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"path": str(destination), "repo": repo, "revision": revision, "files": len(files)}


def target_key(key, relative_path):
    if relative_path.startswith("vision_encoder/"):
        for prefix in ("model.visual.", "vision_encoder.", "visual."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return "model.visual." + key
    for prefix in ("transformer.", "model.net.", "net.", "language_model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    if any(part in key for part in ("_moe_gen", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                                    "norm_added_q", "norm_added_k", "k_norm_und_for_gen")):
        return None
    for old, new in (("to_q", "q_proj"), ("to_k", "k_proj"), ("to_v", "v_proj"),
                     ("to_out", "o_proj"), ("norm_q", "q_norm"), ("norm_k", "k_norm")):
        key = key.replace(".self_attn." + old + ".", ".self_attn." + new + ".")
    if key.startswith("model.language_model.") or key.startswith("model.visual.") or key.startswith("lm_head."):
        return key
    if key.startswith("model.lm_head."):
        return key[len("model."):]
    for prefix in ("model.", ""):
        inner = key[len(prefix):] if key.startswith(prefix) else ""
        if inner.startswith(("embed_tokens.", "layers.", "norm.")):
            return "model.language_model." + inner
    return None


def convert_base(source, processor, destination, shard_mib=128):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, AutoProcessor
    source, processor, destination = map(lambda p: Path(p).resolve(), (source, processor, destination))
    if destination.exists():
        raise FileExistsError("converted base destination exists")
    if source == processor or not source.is_dir() or not processor.is_dir() or shard_mib < 1:
        raise ValueError("separate local source and processor directories and positive shard size required")
    config = Qwen3VLConfig.from_pretrained(processor, local_files_only=True)
    if json.loads((processor / "config.json").read_text()).get("model_type") != "qwen3_vl":
        raise ValueError("processor architecture must be qwen3_vl")
    AutoProcessor.from_pretrained(processor, local_files_only=True, use_fast=True)
    with torch.device("meta"):
        expected = {k: tuple(v.shape) for k, v in Qwen3VLForConditionalGeneration(config).state_dict().items()}
    mapping, sources = {}, {}
    for path in sorted(source.rglob("*.safetensors")):
        if ".cache" in path.parts:
            continue
        relative = str(path.relative_to(source))
        if not path.resolve().is_relative_to(source):
            raise ValueError("source weight path leaves the source directory")
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                mapped = target_key(key, relative)
                if mapped is None:
                    continue
                if mapped not in expected:
                    raise ValueError("unrecognized understanding tensor: " + mapped)
                if mapped in mapping:
                    raise ValueError("duplicate understanding tensor: " + mapped)
                if tuple(stream.get_slice(key).get_shape()) != expected[mapped]:
                    raise ValueError("understanding tensor shape differs: " + mapped)
                mapping[mapped] = (path, key)
                sources[relative] = path
    missing = sorted(set(expected) - set(mapping))
    if missing:
        raise ValueError(f"incomplete Cosmos understanding/vision tower: {len(missing)} missing tensors; {missing[:5]}")
    destination.mkdir(parents=True)
    marker = destination / "CONVERSION_INCOMPLETE"
    marker.write_text("conversion in progress\n")
    config.save_pretrained(destination)
    for path in processor.iterdir():
        if path.is_file() and path.suffix in {".json", ".txt", ".model", ".jinja"} and path.name != "config.json" and "index" not in path.name:
            shutil.copyfile(path, destination / path.name)
    pending, size, total, shard, output_map = {}, 0, 0, 0, {}
    def flush():
        nonlocal pending, size, shard
        if pending:
            shard += 1
            name = f"model-{shard:05d}.safetensors"
            save_file(pending, destination / name, metadata={"format": "pt"})
            output_map.update({key: name for key in pending})
            pending, size = {}, 0
    for key in sorted(mapping):
        path, original = mapping[key]
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            tensor = stream.get_tensor(original).contiguous().clone()
        count = tensor.numel() * tensor.element_size()
        if pending and size + count > shard_mib * 1024**2:
            flush()
        pending[key], size, total = tensor, size + count, total + count
    flush()
    (destination / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
        "weight_map": output_map}, indent=2) + "\n")
    receipt = {"source_files": {name: sha256(path) for name, path in sources.items()},
        "processor_identity": processor_identity(processor), "output_identity": base_identity(destination),
        "mapped_tensors": len(mapping), "missing_tensors": [], "unexpected_tensors": [],
        "mapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tensor_map": {key: {"file": str(path.relative_to(source)), "key": original} for key, (path, original) in mapping.items()}}
    (destination / "conversion_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    marker.unlink()
    return {"path": str(destination), "mapped_tensors": len(mapping), "shards": shard, "bytes": total,
            "base_identity": receipt["output_identity"]}
