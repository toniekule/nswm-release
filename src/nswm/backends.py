"""Generator adapters with explicit request/job binding."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import uuid

from .planning import BackendFailure, Cost, Future
from .schema import Commitment, Interval, commitment_from_dict


class AnalyticGenerator:
    def __call__(self, request):
        cap = request.query.premises["capacity"]
        state = Commitment({k: Interval.point(min(v.lower, Interval.point(cap).lower))
                            for k, v in request.commitment.slots.items()})
        return Future(str(uuid.uuid4()), request.binding, state, {"state": asdict(state)},
                      Cost(1.0, 0.0, 20, "work_units"))


class JsonProcessGenerator:
    def __init__(self, argv, timeout=300, unit="gpu_seconds"):
        if not argv or not isinstance(argv, list):
            raise ValueError("generator argv must be a nonempty list")
        self.argv, self.timeout, self.unit = argv, timeout, unit

    def __call__(self, request):
        job_id = str(uuid.uuid4())
        payload = {"job_id": job_id, "request_binding": request.binding, "request": asdict(request)}
        t0 = time.perf_counter()
        try:
            result = subprocess.run(self.argv, input=json.dumps(payload), capture_output=True, text=True,
                                    timeout=self.timeout, check=True)
            data = json.loads(result.stdout)
            if data["job_id"] != job_id or data["request_binding"] != request.binding:
                raise ValueError("backend response identity mismatch")
            return Future(job_id, request.binding, commitment_from_dict(data["commitment"]) if data.get("commitment") else None,
                          data.get("payload"), Cost(data["compute"], time.perf_counter() - t0,
                                                   data["nfe"], self.unit), data.get("completed", True))
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as error:
            raise BackendFailure(f"process generator failed: {type(error).__name__}",
                                 wall_seconds=time.perf_counter() - t0, job_id=job_id) from error


class CosmosActionGenerator:
    def __init__(self, model_path, observe, measure, output_root, domain="umi", fps=20,
                 resolution=480, steps=20, seed=0):
        import torch
        from diffusers import Cosmos3OmniPipeline
        if not torch.cuda.is_available():
            raise RuntimeError("Cosmos generation requires CUDA")
        self.pipe = Cosmos3OmniPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                    local_files_only=True).to("cuda")
        self.observe, self.measure, self.output = observe, measure, Path(output_root)
        self.output.mkdir(parents=True, exist_ok=True)
        self.domain, self.fps, self.resolution, self.steps, self.seed = domain, fps, resolution, steps, seed

    def __call__(self, request):
        import torch
        from diffusers import CosmosActionCondition
        from diffusers.utils import export_to_video
        job_id = str(uuid.uuid4())
        actions = torch.as_tensor(request.query.actions, dtype=torch.float32)
        if len(actions) != request.query.horizon:
            raise ValueError("action count differs from the generation horizon")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        nfe = [0]
        transformer = getattr(self.pipe, "transformer", None)
        if transformer is None:
            raise RuntimeError("pipeline does not expose its denoiser for NFE accounting")
        handle = transformer.register_forward_hook(lambda *args: nfe.__setitem__(0, nfe[0] + 1))
        t0 = time.perf_counter()
        start.record()
        try:
            image = self.observe(request.query.history_id)
            result = self.pipe(prompt="Predict the action-conditioned future observations.",
                      action=CosmosActionCondition(mode="forward_dynamics", chunk_size=len(actions),
                              domain_name=self.domain, resolution_tier=self.resolution,
                              raw_actions=actions, image=image, view_point="ego_view"),
                      fps=self.fps, num_inference_steps=self.steps, guidance_scale=1.0,
                      use_system_prompt=False, generator=torch.Generator(device="cuda").manual_seed(self.seed))
            video_path = self.output / f"{job_id}.mp4"
            export_to_video(result.video, str(video_path), fps=self.fps, macro_block_size=1)
            measured = self.measure(result.video, request)
            end.record()
            torch.cuda.synchronize()
            cost = Cost(start.elapsed_time(end) / 1000, time.perf_counter() - t0, nfe[0], "gpu_seconds")
        except Exception as error:
            cost = None
            try:
                end.record()
                torch.cuda.synchronize()
                cost = Cost(start.elapsed_time(end) / 1000, time.perf_counter() - t0, nfe[0], "gpu_seconds")
            except RuntimeError:
                pass
            raise BackendFailure(f"Cosmos generation failed: {type(error).__name__}", cost,
                                 time.perf_counter() - t0, job_id) from error
        finally:
            handle.remove()
        return Future(job_id, request.binding, measured, {"video": str(video_path)}, cost)
