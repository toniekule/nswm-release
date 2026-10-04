"""Fixed-context MuJoCo replay and position commitment predicates."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .schema import Commitment, Interval, Query, Verdict


@dataclass(frozen=True)
class PositionAssertion:
    slot: str
    body: str
    frame: int
    axis: int


class MuJoCoReplay:
    def __init__(self, xml_path, qpos, qvel, actions, assertions, physics_hz=240, control_hz=15,
                 seed=0, maximum_penetration=0.001):
        import mujoco
        import numpy as np
        if physics_hz <= 0 or control_hz <= 0 or physics_hz % control_hz or maximum_penetration < 0:
            raise ValueError("physics rate must be divisible by control rate")
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = 1 / physics_hz
        self.qpos, self.qvel = np.array(qpos, dtype=float), np.array(qvel, dtype=float)
        self.actions = np.array(actions, dtype=float)
        if self.qpos.shape != (self.model.nq,) or self.qvel.shape != (self.model.nv,):
            raise ValueError("initial state dimensions differ from model")
        if self.actions.ndim != 2 or self.actions.shape[1] != self.model.nu:
            raise ValueError("action dimension differs from actuator count")
        if not len(self.actions) or not all(np.isfinite(v).all() for v in (self.qpos, self.qvel, self.actions)):
            raise ValueError("replay state and actions must be finite and nonempty")
        self.assertions = tuple(assertions)
        self.substeps, self.seed, self.max_pen = physics_hz // control_hz, seed, maximum_penetration
        self.identity = hashlib.sha256(self.xml_path.read_bytes() + self.qpos.tobytes()
                     + self.qvel.tobytes() + self.actions.tobytes() + json.dumps({"seed": seed,
                         "physics_hz": physics_hz, "control_hz": control_hz, "max_pen": maximum_penetration,
                         "assertions": [asdict(a) for a in assertions]}, sort_keys=True).encode()).hexdigest()
        self.positions = None
        self.penetrations = None

    def replay(self):
        import mujoco
        import numpy as np
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel
        mujoco.mj_forward(self.model, data)
        self.initial_penetration = max([max(0.0, -c.dist) for c in data.contact] or [0.0])
        positions, penetration = [data.xpos.copy()], []
        self.qpos_frames = [data.qpos.copy()]
        for action in self.actions:
            data.ctrl[:] = action
            maximum = 0.0
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, data)
                maximum = max(maximum, max([max(0.0, -c.dist) for c in data.contact] or [0.0]))
            mujoco.mj_forward(self.model, data)
            positions.append(data.xpos.copy())
            self.qpos_frames.append(data.qpos.copy())
            penetration.append(maximum)
        self.positions = np.array(positions)
        self.penetrations = np.array(penetration)
        return self.positions

    def oracle(self, query: Query, commitment: Commitment):
        import mujoco
        import numpy as np
        if query.premises.get("replay_identity") != self.identity:
            return Verdict.UNKNOWN
        if query.horizon != len(self.actions) or not np.array_equal(np.array(query.actions), self.actions):
            return Verdict.UNKNOWN
        if self.positions is None:
            self.replay()
        if not query.premises.get("complete_position_domain", False):
            return Verdict.UNKNOWN
        if max(self.initial_penetration, self.penetrations.max(initial=0)) > self.max_pen:
            return Verdict.UNKNOWN
        if set(commitment.slots) != {assertion.slot for assertion in self.assertions}:
            return Verdict.UNKNOWN
        for assertion in self.assertions:
            if assertion.slot not in commitment.slots or assertion.axis not in {0, 1, 2}:
                return Verdict.UNKNOWN
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, assertion.body)
            if body < 0 or not 0 <= assertion.frame < len(self.positions):
                return Verdict.UNKNOWN
            actual = Interval.point(float(self.positions[assertion.frame, body, assertion.axis]))
            if not commitment.slots[assertion.slot].contains(actual):
                return Verdict.INVALID
        return Verdict.VALID

    def render_frames(self, destination, width=448, height=256, camera=-1):
        import mujoco
        from PIL import Image
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError("render destination exists")
        destination.mkdir(parents=True)
        if self.positions is None:
            self.replay()
        data = mujoco.MjData(self.model)
        paths = []
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            for i, qpos in enumerate(self.qpos_frames):
                data.qpos[:] = qpos
                mujoco.mj_forward(self.model, data)
                renderer.update_scene(data, camera=camera)
                path = destination / f"{i:05d}.png"
                Image.fromarray(renderer.render()).save(path)
                paths.append(path)
        return paths
