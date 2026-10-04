"""Base model and processor resource checks.

Training loads both with ``local_files_only``, so a run can only start when the
configured paths already hold the required files. This module reports what is
missing and which files define the identity that checkpoints bind to.
"""
from pathlib import Path

from .train import base_identity, processor_identity


def validate_weight_headers(path, identity):
    import json
    import math
    import struct
    widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
              "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    for name in identity:
        if not name.endswith(".safetensors"):
            continue
        file = Path(path) / name
        with file.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors header: " + name)
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= min(64 << 20, file.stat().st_size - 8):
                raise ValueError("invalid safetensors header size: " + name)
            header = json.loads(stream.read(length))
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        if not tensors:
            raise ValueError("empty weight shard: " + name)
        for value in tensors.values():
            start, end = value["data_offsets"]
            expected = math.prod(value["shape"]) * widths[value["dtype"]]
            if start < 0 or end - start != expected or end > file.stat().st_size - 8 - length:
                raise ValueError("invalid tensor data offsets: " + name)


def inspect_resources(config, probe_device=False):
    """Report the state of the configured base and processor paths."""
    import importlib.util
    has_torch = importlib.util.find_spec("torch") is not None
    cuda = None
    if has_torch and probe_device:
        import torch
        cuda = torch.cuda.is_available()
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "transformers", "accelerate", "torchvision", "PIL", "numpy")}
    device_ready = (cuda if config.get("device", "cuda").startswith("cuda") else has_torch)
    report = {"ready": False, "files_ready": True, "torch": has_torch, "cuda": cuda,
              "device_ready": device_ready, "dependencies": dependencies, "model": {}, "processor": {}}
    for key, identity in (("model", base_identity), ("processor", processor_identity)):
        raw = config.get(key)
        path = Path(raw) if raw else None
        entry = {"path": raw, "exists": bool(path and path.is_dir())}
        if not entry["exists"]:
            entry["error"] = f"{key} path is not a directory; weights are never downloaded implicitly"
            report["files_ready"] = False
        else:
            try:
                import json
                if key == "model":
                    metadata = json.loads((path / "config.json").read_text())
                    if metadata.get("model_type") != "qwen3_vl":
                        raise ValueError("model must be a converted qwen3_vl reasoner")
                else:
                    required = ["tokenizer_config.json", "preprocessor_config.json", "video_preprocessor_config.json"]
                    missing = [name for name in required if not (path / name).is_file()]
                    if missing:
                        raise ValueError("processor components missing: " + ", ".join(missing))
                entry["identity"] = identity(path)
                if key == "model":
                    validate_weight_headers(path, entry["identity"])
                entry["files"] = len(entry["identity"])
            except (ValueError, OSError, KeyError, TypeError) as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                report["files_ready"] = False
        report[key] = entry
    report["data_ready"] = bool(config.get("events") and Path(config["events"]).is_file())
    report["ready"] = report["files_ready"] and report["data_ready"] and all(dependencies.values()) and device_ready is True
    report["device_probe_pending"] = device_ready is None
    return report
