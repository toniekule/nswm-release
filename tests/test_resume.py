import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "training dependencies are unavailable")
class ResumeTests(unittest.TestCase):
    def test_full_controller_resume_matches_uninterrupted(self):
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        from nswm.data import build_fixtures
        from nswm.train import run_training
        torch.set_num_threads(2)
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1
            def encode(self, text, **kwargs):
                return [ord(c) % 120 + 2 for c in text]
            def __call__(self, text, **kwargs):
                return {"input_ids": self.encode(text), "offset_mapping": [(i, i+1) for i in range(len(text))]}
        class Processor:
            tokenizer = Tokenizer()
            def apply_chat_template(self, messages, **kwargs):
                return "User Assistant"
            def __call__(self, **kwargs):
                return {"input_ids": torch.tensor([[2, 3]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(3)
            cfg = Qwen3VLConfig(text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                       "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                       "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True}},
                  vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                       "out_hidden_size": 32, "patch_size": 16, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "deepstack_visual_indexes": []})
            Qwen3VLForConditionalGeneration(cfg).save_pretrained(root / "model")
            events = build_fixtures(root / "events", 2)
            config = {"events": str(events), "model": str(root / "model"), "processor": str(root / "model"),
                      "output": str(root / "interrupted"), "device": "cpu", "modes": ["gate", "critic"],
                      "effective_batch": 2, "epochs": 2, "max_steps": 2, "rank": 2, "alpha": 4,
                      "dropout": 0.05, "longest_side": 32, "max_output_tokens": 2048,
                      "gradient_checkpointing": False, "validation_interval": 50}
            with patch("transformers.AutoProcessor.from_pretrained", return_value=Processor()):
                run_training(config)
                config["max_steps"] = 4
                run_training(config, root / "interrupted/latest.pt")
                config["output"] = str(root / "uninterrupted")
                run_training(config)
            a = torch.load(root / "interrupted/latest.pt", weights_only=True)
            b = torch.load(root / "uninterrupted/latest.pt", weights_only=True)
            self.assertEqual(a["next_step"], b["next_step"])
            for key in a["adapter"]:
                torch.testing.assert_close(a["adapter"][key], b["adapter"][key], rtol=0, atol=0)
            from nswm.data import read_events
            from nswm.predict import SharedPredictor
            with patch("transformers.AutoProcessor.from_pretrained", return_value=Processor()):
                predictor = SharedPredictor(root / "uninterrupted/latest.pt", root=root / "events")
                prediction = predictor.predict_event(read_events(events)[0], "gate", "judgment")
                self.assertTrue(0 <= prediction.risk <= 1)
                self.assertGreater(prediction.cost.wall_seconds, 0)
            changed = {**config, "rank": 3}
            with patch("transformers.AutoProcessor.from_pretrained", return_value=Processor()):
                with self.assertRaises(ValueError):
                    run_training(changed, root / "uninterrupted/latest.pt")
            processor_metadata = root / "model/processor_config.json"
            processor_metadata.write_text('{"version":2}')
            with self.assertRaises(ValueError):
                run_training(config, root / "uninterrupted/latest.pt")
