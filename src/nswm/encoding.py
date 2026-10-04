"""Multimodal prompts and target-only teacher forcing."""
import json
from pathlib import Path

from .data import model_input
from .targets import FIELDS, serialize_target


def target_segments(target, removed=()):
    values = {field: target.get(field) for field in FIELDS}
    text, spans = "{", []
    for i, (key, value) in enumerate(values.items()):
        prefix = ("," if i else "") + json.dumps(key) + ":"
        start = len(text)
        text += prefix + json.dumps(value, separators=(",", ":"))
        if key in removed:
            spans.append((start, len(text)))
    text += "}"
    # Suppress repeated textual identities in later teacher-forced fields.
    for key in removed:
        spans.extend(tuple(span) for span in target.get("field_dependencies", {}).get(key, ()))
        value = values[key]
        tokens = value if isinstance(value, (list, tuple)) else [value]
        for token in tokens:
            if isinstance(token, str):
                encoded = json.dumps(token)
                pos = 0
                while (pos := text.find(encoded, pos)) >= 0:
                    spans.append((pos, pos + len(encoded)))
                    pos += len(encoded)
    return text, spans


def encode_input(row, mode, root, processor, config):
    import numpy as np
    import torch
    from PIL import Image
    payload, paths = model_input(row, mode, root, config.get("restricted_critic", False) and mode == "critic")
    longest = config.get("longest_side", 448)
    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        scale = longest / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        width, height = image.size
        padded = Image.new("RGB", ((width + 31) // 32 * 32, (height + 31) // 32 * 32))
        padded.paste(image, (0, 0))
        frames.append(np.asarray(padded))
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("one observation clip requires a consistent spatial shape")
    valid_count = len(frames)
    frame_count = 17 if mode == "critic" else 5
    frames += [np.zeros_like(frames[0]) for _ in range(frame_count - valid_count)]
    payload["frame_valid"] = [True] * valid_count + [False] * (frame_count - valid_count)
    prompt = ("Return the structured physical judgment and certificate. Unavailable fields use null.\n"
              + json.dumps(payload, separators=(",", ":")))
    if config.get("output_field"):
        prompt += "\nReturn only the field: " + config["output_field"]
    if len(processor.tokenizer.encode(prompt, add_special_tokens=False)) > config.get("max_text_tokens", 4096):
        raise ValueError("text context exceeds limit; commitments cannot be truncated")
    messages = [{"role": "user", "content": [{"type": "video", "video": paths},
                                                 {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    times = payload["timestamps"]
    fps = 1 / (times[1] - times[0]) if len(times) > 1 and times[1] > times[0] else config.get("fps", 8.0)
    from transformers.video_utils import VideoMetadata
    metadata = VideoMetadata(total_num_frames=len(frames), fps=fps, frames_indices=list(range(len(frames))))
    batch = processor(text=[rendered], videos=[np.stack(frames)], video_metadata=[metadata],
                      do_sample_frames=False, do_resize=False, return_tensors="pt")
    batch.pop("token_type_ids", None)
    return batch


def encode_event(row, mode, root, processor, config):
    if config.get("supervision") == "minimal":
        raise ValueError("minimal is not a distinct supervision format; use certificate")
    import torch
    batch = encode_input(row, mode, root, processor, config)
    removed = tuple(config.get("removed_fields", ()))
    supervision = config.get("supervision", "certificate")
    target = row["targets"][mode]
    if supervision == "redundant":
        target = row["redundant_targets"][mode]
        supervision = "certificate"
    if config.get("output_field"):
        field = config["output_field"]
        answer, masked_spans = json.dumps({field: target.get(field)}, separators=(",", ":")), []
    elif supervision == "certificate":
        answer, masked_spans = target_segments(target, removed)
    else:
        if removed:
            raise ValueError("field ablations require certificate supervision")
        answer, masked_spans = serialize_target(target, supervision), []
    answer_encoding = processor.tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    target_ids = list(answer_encoding["input_ids"])
    labels = list(target_ids)
    pad = processor.tokenizer.pad_token_id
    if pad is None:
        pad = processor.tokenizer.eos_token_id
    for i, (lo, hi) in enumerate(answer_encoding["offset_mapping"]):
        if any(lo < end and hi > start for start, end in masked_spans):
            labels[i] = -100
            target_ids[i] = pad
    eos = processor.tokenizer.eos_token_id
    if eos is None or pad is None:
        raise ValueError("tokenizer must define EOS or padding")
    target_ids.append(eos)
    labels.append(eos)
    if len(target_ids) > config.get("max_output_tokens", 512):
        raise ValueError("certificate exceeds output limit; targets cannot be truncated")
    prefix = batch["input_ids"].shape[1]
    batch["input_ids"] = torch.cat([batch["input_ids"], torch.tensor([target_ids])], dim=1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones(1, len(target_ids), dtype=torch.long)], dim=1)
    batch["labels"] = torch.tensor([[-100] * prefix + labels])
    return batch


def encode_training_views(row, mode, root, processor, config):
    if config.get("supervision") != "multi_head":
        return [encode_event(row, mode, root, processor, config)]
    if config.get("removed_fields"):
        raise ValueError("multi-head field ablations require separate registered queries")
    return [encode_event(row, mode, root, processor, {**config, "supervision": "certificate", "output_field": field})
            for field in FIELDS]
