"""Shared decoder inference from permitted observations."""
import json
import math
from pathlib import Path
import time

from .planning import Cost, Prediction
from .schema import Certificate, Verdict
from .targets import FIELDS, parse_certificate


class SharedPredictor:
    def __init__(self, checkpoint, observe=None, root="."):
        import torch
        from transformers import AutoProcessor
        from .lora import inject_lora, load_adapter_state
        from .train import base_identity, processor_identity, load_reasoner
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = state["config"]
        if base_identity(self.config["model"]) != state["base_identity"]:
            raise ValueError("checkpoint base identity differs from current weights")
        if processor_identity(self.config["processor"]) != state["processor_identity"]:
            raise ValueError("checkpoint processor identity differs from current metadata")
        self.device = self.config.get("device", "cuda")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for this checkpoint configuration")
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.config["processor"], local_files_only=True,
                                                       use_fast=True)
        self.model = load_reasoner(self.config["model"], self.device, self.dtype)
        inject_lora(self.model, self.config.get("rank", 64), self.config.get("alpha", 128), 0)
        load_adapter_state(self.model, state["adapter"])
        self.model.eval()
        self.observe, self.root = observe, Path(root)

    def __call__(self, request, prefix="complete"):
        if self.observe is None:
            raise ValueError("planner inference requires an observation provider")
        observations = self.observe(request.query.history_id)
        row = {"query": {"history_id": request.query.history_id, "actions": request.query.actions,
                          "premises": request.query.premises, "horizon": request.query.horizon},
               "commitment": {"slots": {k: {"lo": v.lo, "hi": v.hi} for k, v in request.commitment.slots.items()}},
               "observations": {"gate": observations}}
        return self.predict_event(row, "gate", prefix)

    def predict_event(self, row, mode, prefix="complete"):
        import torch
        from .encoding import encode_input
        from .train import move_batch
        if prefix not in {"complete", "judgment"}:
            raise ValueError("unknown inference prefix")
        supervision = self.config.get("supervision", "certificate")
        cuda = self.device.startswith("cuda")
        start, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if cuda else (None, None)
        t0 = time.perf_counter()
        if cuda:
            start.record()

        def prepare(field=None):
            config = {**self.config, "output_field": field}
            return move_batch(encode_input(row, mode, self.root, self.processor, config), self.device, self.dtype)

        def decode(batch):
            output = self.model.generate(**batch, max_new_tokens=self.config.get("decode_max_tokens", self.config.get("max_output_tokens", 512)), do_sample=False,
                                         eos_token_id=self.processor.tokenizer.eos_token_id)
            return self.processor.tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)

        with torch.no_grad():
            batch = prepare("judgment" if supervision == "multi_head" else None)
            if supervision == "scalar":
                try:
                    residual = float(json.loads(decode(batch))["residual"])
                    threshold = self.config.get("scalar_threshold", 0.0)
                    temperature = self.config.get("scalar_temperature", 1.0)
                    if not math.isfinite(residual) or temperature <= 0:
                        raise ValueError("invalid scalar prediction")
                    risk = 1 / (1 + math.exp(max(-700, min(700, -(residual - threshold) / temperature))))
                    cert = Certificate(Verdict.INVALID if residual > threshold else Verdict.VALID)
                except (ValueError, KeyError, TypeError):
                    risk, cert = 0.0, Certificate()
            else:
                risk, cert = self._decode_judgment(batch, decode, prepare, supervision, prefix)
        if cuda:
            end.record()
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        return Prediction(cert, risk, Cost(start.elapsed_time(end) / 1000 if cuda else wall, wall,
                                           unit="gpu_seconds" if cuda else "cpu_seconds"))

    def _decode_judgment(self, batch, decode, prepare, supervision, prefix):
        import torch
        with torch.no_grad():
            log_probabilities = []
            for judgment in ["valid", "invalid", "unknown"]:
                text = ('judgment: ' if supervision == "free_form" else '{"judgment":') + json.dumps(judgment)
                ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
                candidate = torch.tensor([ids], device=self.device)
                inputs = {**batch, "input_ids": torch.cat([batch["input_ids"], candidate], dim=1),
                          "attention_mask": torch.cat([batch["attention_mask"], torch.ones_like(candidate)], dim=1)}
                logits = self.model(**inputs).logits[:, batch["input_ids"].shape[1] - 1:-1].float()
                probabilities = logits.log_softmax(-1).gather(-1, candidate.unsqueeze(-1)).squeeze(-1)
                log_probabilities.append(probabilities.sum())
            risk = float(torch.stack(log_probabilities).softmax(0)[1])
            if prefix == "judgment":
                cert = Certificate([Verdict.VALID, Verdict.INVALID, Verdict.UNKNOWN][int(torch.stack(log_probabilities).argmax())])
            elif supervision == "multi_head":
                fields = {}
                for field in FIELDS:
                    try:
                        fields[field] = json.loads(decode(batch if field == "judgment" else prepare(field)))[field]
                    except (ValueError, KeyError, TypeError):
                        fields[field] = None
                cert = parse_certificate(json.dumps(fields))
            elif supervision == "free_form":
                fields = {}
                for line in decode(batch).splitlines():
                    key, sep, value = line.partition(":")
                    if sep and key.strip() in FIELDS:
                        try:
                            fields[key.strip()] = json.loads(value)
                        except ValueError:
                            fields[key.strip()] = None
                cert = parse_certificate(json.dumps(fields))
            else:
                cert = parse_certificate(decode(batch))
        return risk, cert
