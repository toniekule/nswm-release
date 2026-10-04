from pathlib import Path
import tempfile
import unittest
import importlib.util

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is unavailable")
class TrainingTests(unittest.TestCase):
    def test_real_cpu_updates_and_adapter_reload(self):
        import torch
        from torch import nn
        from nswm.lora import inject_lora, adapter_state, load_adapter_state
        from nswm.loss import certificate_loss
        torch.manual_seed(7)
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(16, 8)
                self.q_proj = nn.Linear(8, 8)
                self.head = nn.Linear(8, 16)
            def forward(self, ids):
                return self.head(self.q_proj(self.embedding(ids)))
        model = Tiny()
        inject_lora(model, rank=2, alpha=4, dropout=0)
        base = {k: p.clone() for k, p in model.named_parameters() if not p.requires_grad}
        before = {k: v.clone() for k, v in adapter_state(model).items()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, -100, 3, 4]])
        for _ in range(3):
            optimizer.zero_grad()
            loss = certificate_loss(model(ids), labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in adapter_state(model).items()))
        self.assertTrue(all(torch.equal(v, dict(model.named_parameters())[k]) for k, v in base.items()))
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "adapter.pt"
            torch.save(adapter_state(model), file)
            prediction = model(ids).detach().clone()
            for p in model.parameters():
                if p.requires_grad:
                    p.data.zero_()
            load_adapter_state(model, torch.load(file, weights_only=True))
            torch.testing.assert_close(model(ids), prediction)

    def test_sequence_normalization_and_prompt_mask(self):
        import torch
        from nswm.loss import certificate_loss
        logits = torch.randn(2, 5, 8, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, -100, -100, -100, 3]])
        combined = certificate_loss(logits, labels)
        separate = (certificate_loss(logits[:1], labels[:1]) + certificate_loss(logits[1:], labels[1:])) / 2
        torch.testing.assert_close(combined, separate)
        combined.backward()
        self.assertEqual(float(logits.grad[:, 0].abs().sum()), 0)
        with self.assertRaises(ValueError):
            certificate_loss(logits, torch.full_like(labels, -100))

    def test_tiny_qwen3_vl_forward_backward(self):
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("Transformers is unavailable")
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        from nswm.lora import inject_lora
        from nswm.loss import certificate_loss
        cfg = Qwen3VLConfig(text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                   "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                   "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True}},
              vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                   "out_hidden_size": 32, "patch_size": 16, "spatial_merge_size": 2,
                   "temporal_patch_size": 2, "deepstack_visual_indexes": []})
        model = Qwen3VLForConditionalGeneration(cfg)
        targets = inject_lora(model, rank=2, alpha=4)
        self.assertEqual(len(targets), 4)
        ids = torch.tensor([[1, 2, 3, 4]])
        loss = certificate_loss(model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits, ids)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))
