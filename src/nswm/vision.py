"""Local perception adapters and observation-window validation."""
from pathlib import Path
import math

from .perception import pose_box


def observed_times(timestamps, count):
    if count < 1 or len(timestamps) != count or any(not math.isfinite(t) or t > 0 for t in timestamps):
        raise ValueError("observed history must have finite past timestamps")
    if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("history timestamps must increase strictly")


def array(value):
    import numpy as np
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


class SAM2HistorySegmenter:
    def __init__(self, predictor):
        self.predictor = predictor

    @classmethod
    def from_local(cls, config, checkpoint, device="cuda"):
        from sam2.build_sam import build_sam2_video_predictor
        checkpoint = Path(checkpoint).resolve(strict=True)
        return cls(build_sam2_video_predictor(config, str(checkpoint), device=device))

    def segment(self, jpeg_directory, timestamps, prompts):
        root = Path(jpeg_directory).resolve(strict=True)
        files = sorted(root.glob("*.jpg")) + sorted(root.glob("*.jpeg"))
        observed_times(timestamps, len(files))
        if len(prompts) == 0 or len({p["entity"] for p in prompts}) != len(prompts):
            raise ValueError("segmentation needs distinct entity prompts")
        state = self.predictor.init_state(video_path=str(root))
        identities = {}
        for obj_id, prompt in enumerate(prompts, 1):
            frame = prompt.get("frame", 0)
            if not isinstance(frame, int) or not 0 <= frame < len(files):
                raise ValueError("segmentation prompt frame is outside history")
            identities[obj_id] = prompt["entity"]
            self.predictor.add_new_points_or_box(inference_state=state, frame_idx=frame, obj_id=obj_id,
                points=prompt.get("points"), labels=prompt.get("labels"), box=prompt.get("box"))
        masks = {}
        for frame, obj_ids, logits in self.predictor.propagate_in_video(state):
            if not 0 <= frame < len(files):
                raise ValueError("segmenter returned a frame outside history")
            masks[frame] = {identities[int(obj_id)]: array(logits[i]) > 0 for i, obj_id in enumerate(obj_ids)}
        return masks


class CoTracker3HistoryTracker:
    def __init__(self, predictor, device="cuda"):
        self.predictor, self.device = predictor, device

    @classmethod
    def from_local(cls, checkpoint, device="cuda"):
        from cotracker.predictor import CoTrackerOnlinePredictor
        checkpoint = Path(checkpoint).resolve(strict=True)
        predictor = CoTrackerOnlinePredictor(checkpoint=str(checkpoint), v2=False, window_len=16).to(device).eval()
        return cls(predictor, device)

    def track(self, rgb_frames, timestamps, queries):
        import numpy as np
        import torch
        frames = np.asarray(rgb_frames)
        observed_times(timestamps, len(frames))
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("tracking frames must be T,H,W,3 RGB")
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != 3 or not np.isfinite(queries).all():
            raise ValueError("track queries must be finite frame,x,y triples")
        if (queries[:, 0] < 0).any() or (queries[:, 0] >= len(frames)).any() or (queries[:, 0] % 1 != 0).any():
            raise ValueError("track query frame is outside history")
        video = torch.as_tensor(frames, device=self.device).permute(0, 3, 1, 2).float()[None]
        points = torch.as_tensor(queries, device=self.device)[None]
        with torch.inference_mode():
            self.predictor(video_chunk=video, is_first_step=True, queries=points, grid_size=0)
            step = self.predictor.step
            if step < 1:
                raise ValueError("tracker step must be positive")
            tracks = visible = None
            for start in range(0, len(frames), step):
                stop = min(len(frames), start + 2 * step)
                tracks, visible = self.predictor(video_chunk=video[:, start:stop], is_first_step=False)
                if stop == len(frames):
                    break
        tracks, visible = array(tracks), array(visible)
        if tracks.shape[1] < len(frames) or visible.shape[1] < len(frames):
            raise ValueError("tracker omitted observed frames")
        return tracks[0, :len(frames)], visible[0, :len(frames)]


def rigid_pose(value):
    import numpy as np
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("pose must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0, 0, 0, 1]) or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError("pose is not rigid")
    if not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-3):
        raise ValueError("pose rotation must preserve orientation")
    return value


class FoundationPoseHistoryEstimator:
    def __init__(self, estimator, register_iterations=5, track_iterations=2):
        self.estimator = estimator
        self.register_iterations, self.track_iterations = register_iterations, track_iterations

    def estimate(self, rgb_frames, depth_meters, intrinsics, masks, timestamps):
        import numpy as np
        observed_times(timestamps, len(rgb_frames))
        depth = np.asarray(depth_meters, dtype=np.float32)
        rgb = np.asarray(rgb_frames)
        k = np.asarray(intrinsics, dtype=float)
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or depth.shape != rgb.shape[:3] or len(masks) != len(rgb):
            raise ValueError("RGB, metric depth and masks must share the observation window")
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("invalid calibrated camera intrinsics")
        poses = []
        for i, (color, distance, mask) in enumerate(zip(rgb, depth, masks)):
            mask = np.asarray(mask, dtype=bool).reshape(distance.shape)
            usable = mask & np.isfinite(distance) & (distance > 0)
            if not usable.any():
                raise ValueError("metric pose requires observed positive depth")
            distance = np.where(np.isfinite(distance) & (distance > 0), distance, 0)
            pose = self.estimator.register(K=k, rgb=color, depth=distance, ob_mask=usable,
                       iteration=self.register_iterations) if i == 0 else self.estimator.track_one(
                       rgb=color, depth=distance, K=k, iteration=self.track_iterations)
            poses.append(rigid_pose(pose))
        return np.stack(poses)


def calibrated_pose_box(entity, timestamp, camera_pose, camera_to_world, radius, query):
    world = rigid_pose(camera_to_world) @ rigid_pose(camera_pose)
    return pose_box(entity, timestamp, world[:3, 3], radius, query, "calibrated_pose")


def nominal_fcl_distance(geometry_a, pose_a, geometry_b, pose_b):
    import fcl
    a, b = rigid_pose(pose_a), rigid_pose(pose_b)
    first = fcl.CollisionObject(geometry_a, fcl.Transform(a[:3, :3], a[:3, 3]))
    second = fcl.CollisionObject(geometry_b, fcl.Transform(b[:3, :3], b[:3, 3]))
    result = fcl.DistanceResult()
    return fcl.distance(first, second, fcl.DistanceRequest(enable_signed_distance=True), result)
