import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"), "training dependencies are unavailable")
class ProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = os.environ.get("NSWM_PROCESSOR_PATH")
        if not cls.path:
            from nswm.tiny import make_tiny
            make_tiny(Path(cls.temporary.name) / "tiny")
            cls.path = str(Path(cls.temporary.name) / "tiny/processor")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_supervision_formats_and_restricted_temporal_shape(self):
        from transformers import AutoProcessor
        from nswm.data import build_fixtures, read_events
        from nswm.encoding import encode_training_views
        processor = AutoProcessor.from_pretrained(self.path, local_files_only=True, use_fast=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            rows = read_events(build_fixtures(root, 1))
            for supervision in ["certificate", "redundant", "binary", "scalar", "free_form", "multi_head"]:
                config = {"longest_side": 64, "max_output_tokens": 4096, "supervision": supervision, "restricted_critic": True}
                batches = encode_training_views(rows[1], "critic", root, processor, config)
                self.assertEqual(len(batches), 8 if supervision == "multi_head" else 1)
                for batch in batches:
                    self.assertEqual(int(batch["video_grid_thw"][0, 0]), 9)
                    self.assertGreater(int(batch["labels"].ne(-100).sum()), 0)

    def test_actual_processor_and_tiny_multimodal_update(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
        from nswm.data import build_fixtures, read_events
        from nswm.encoding import encode_event
        from nswm.lora import inject_lora
        from nswm.loss import certificate_loss
        torch.set_num_threads(2)
        processor = AutoProcessor.from_pretrained(self.path, local_files_only=True, use_fast=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            rows = read_events(build_fixtures(root, 1))
            config = {"longest_side": 64, "max_output_tokens": 4096, "removed_fields": ["entities"]}
            gate = encode_event(rows[1], "gate", root, processor, config)
            critic = encode_event(rows[1], "critic", root, processor, config)
            intact = encode_event(rows[1], "gate", root, processor, {**config, "removed_fields": []})
            self.assertEqual(gate["input_ids"].shape, intact["input_ids"].shape)
            self.assertEqual(int(gate["video_grid_thw"][0, 0]), 3)
            self.assertEqual(int(critic["video_grid_thw"][0, 0]), 9)
            self.assertTrue((gate["labels"] == -100).any())
            cfg = Qwen3VLConfig(image_token_id=126, video_token_id=127, vision_start_token_id=125, vision_end_token_id=124,
                text_config={"vocab_size": 128, "hidden_size": 32,
                       "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
                       "num_key_value_heads": 2, "head_dim": 8,
                       "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True}},
                vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                       "out_hidden_size": 32, "patch_size": 16, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "deepstack_visual_indexes": []})
            model = Qwen3VLForConditionalGeneration(cfg)
            inject_lora(model, rank=2, alpha=4)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
            for batch in [gate, critic]:
                ids = batch["input_ids"]
                mapped = ids.remainder(120)
                special = [processor.tokenizer.convert_tokens_to_ids(t) for t in
                           ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]]
                for original, compact in zip(special, [126, 127, 125, 124]):
                    mapped[ids == original] = compact
                batch["input_ids"] = mapped
                supervised = batch["labels"] != -100
                batch["labels"][supervised] = batch["labels"][supervised].remainder(120)
                optimizer.zero_grad()
                labels = batch.pop("labels")
                loss = certificate_loss(model(**batch).logits, labels)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()
