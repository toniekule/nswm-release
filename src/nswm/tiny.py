"""Create a local Qwen3-VL model and processor for CPU integration runs."""
import json
from pathlib import Path


def make_tiny(destination, seed=0, families=20):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import (Qwen2TokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                              Qwen3VLProcessor, Qwen2VLImageProcessor, Qwen3VLVideoProcessor)
    from .data import build_fixtures
    from .train import base_identity, processor_identity
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("tiny model destination exists")
    if families < 10:
        raise ValueError("at least ten families required")
    destination.mkdir(parents=True)
    special = ["<|pad|>", "<|im_end|>", "<|im_start|>", "<|vision_start|>", "<|vision_end|>",
               "<|image_pad|>", "<|video_pad|>", "<|unk|>"]
    vocabulary = {token: i for i, token in enumerate(special)}
    vocabulary.update({token: i + len(special) for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))})
    tokens = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="<|unk|>"))
    tokens.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokens.decoder = decoders.ByteLevel()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=tokens, pad_token="<|pad|>",
                eos_token="<|im_end|>", unk_token="<|unk|>", additional_special_tokens=special[2:7])
    template = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{% for item in message['content'] %}{% if item['type'] == 'video' %}"
        "{{ '<|vision_start|><|video_pad|><|vision_end|>' }}{% elif item['type'] == 'text' %}"
        "{{ item['text'] }}{% endif %}{% endfor %}{{ '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    tokenizer.chat_template = template
    processor = Qwen3VLProcessor(tokenizer=tokenizer, chat_template=template,
        image_processor=Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2))
    processor.save_pretrained(destination / "processor")
    config = Qwen3VLConfig(text_config={"vocab_size": len(tokenizer), "hidden_size": 32,
        "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "out_hidden_size": 32, "patch_size": 16, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "deepstack_visual_indexes": []},
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    torch.manual_seed(seed)
    Qwen3VLForConditionalGeneration(config).save_pretrained(destination / "model")
    events = build_fixtures(destination / "data", families, seed)
    cfg = {"events": str(events), "model": str(destination / "model"),
        "processor": str(destination / "processor"), "output": str(destination / "training"),
        "device": "cpu", "seed": seed, "modes": ["gate", "critic"], "effective_batch": 1,
        "epochs": 1, "max_steps": 2, "rank": 2, "alpha": 4, "dropout": 0.05,
        "longest_side": 32, "max_text_tokens": 8192, "max_output_tokens": 4096, "decode_max_tokens": 32,
        "gradient_checkpointing": False, "validation_families": 1}
    (destination / "train.json").write_text(json.dumps(cfg, indent=2) + "\n")
    receipt = {"seed": seed, "model_kind": "random_initialization", "config": str(destination / "train.json"),
               "base_identity": base_identity(cfg["model"]), "processor_identity": processor_identity(cfg["processor"])}
    (destination / "resources.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
