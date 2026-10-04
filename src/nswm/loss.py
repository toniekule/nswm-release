"""Per-sequence supervised-token normalization."""
import torch
import torch.nn.functional as F


def certificate_loss(logits, labels):
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("each sequence must have supervised tokens")
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                         shifted_labels.reshape(-1), reduction="none", ignore_index=-100)
    per_sequence = (ce.reshape_as(shifted_labels) * mask).sum(dim=1) / counts
    return per_sequence.mean()
