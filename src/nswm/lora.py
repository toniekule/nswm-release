"""Language attention adapters and adapter-only checkpoints."""
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=64, alpha=128, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError("LoRA requires a linear layer and positive rank")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        residual = torch.nn.functional.linear(self.dropout(x.float()), self.lora_A)
        residual = torch.nn.functional.linear(residual, self.lora_B) * (self.alpha / self.rank)
        return self.base(x) + residual.to(x.dtype)


def inject_lora(model, rank=64, alpha=128, dropout=0.05,
                targets=("q_proj", "k_proj", "v_proj", "o_proj")):
    for p in model.parameters():
        p.requires_grad_(False)
    paths = []
    for name, module in list(model.named_modules()):
        if (isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets
                and "visual" not in name and "vision" not in name):
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, leaf, LoRALinear(module, rank, alpha, dropout))
            paths.append(name)
    if not paths:
        raise ValueError("no language attention targets found")
    return paths


def adapter_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith("lora_A") or k.endswith("lora_B")}


def load_adapter_state(model, state):
    live = adapter_state(model)
    if set(live) != set(state) or any(live[k].shape != state[k].shape for k in live):
        raise ValueError("adapter keys or shapes do not match")
    model.load_state_dict(state, strict=False)
