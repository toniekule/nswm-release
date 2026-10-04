"""Deterministically sampled, named-body MuJoCo scene specifications."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random

from .data import FAMILIES
from .schema import fingerprint


@dataclass(frozen=True)
class Scene:
    family_id: str
    category: str
    seed: int
    radius: float
    mass: float
    velocity: float
    spacing: float
    tint: int
    bodies: int = 2
    horizon: int = 16
    control_hz: int = 15
    physics_hz: int = 240
    shift: str = "id"

    def __post_init__(self):
        if self.category not in FAMILIES or self.bodies not in {2, 3} or (self.horizon, self.control_hz, self.physics_hz) != (16, 15, 240):
            raise ValueError("unsupported scene category, body count or replay clock")
        if any(not math.isfinite(v) or v <= 0 for v in (self.radius, self.mass, self.velocity, self.spacing)):
            raise ValueError("scene parameters must be finite and positive")

    def xml(self):
        bodies = []
        for i, name in enumerate("ABC"[:self.bodies]):
            x, y = (-0.7, 0.0) if i == 0 else (self.spacing, 0.0 if self.category == "contact_solidity" and i == 1 else 0.7 * i)
            bodies.append(f'<body name="{name}" pos="{x} {y} 0.2"><freejoint name="{name}.free"/>'
                f'<geom name="{name}.sphere" type="sphere" size="{self.radius}" mass="{self.mass}"'
                f' rgba="{0.8 if i == 0 else 0.2} 0.5 0.7 1"/></body>')
        return ('<mujoco model="nswm"><option gravity="0 0 0" timestep="0.004166666666666667"/>'
            '<visual><global offwidth="448" offheight="256"/></visual><worldbody>'
            '<light pos="0 0 4"/><camera name="overview" pos="0 -3 3" xyaxes="1 0 0 0 1 1"/>'
            + "".join(bodies) + '</worldbody></mujoco>')

    def replay(self, xml_path):
        import mujoco
        import numpy as np
        from .physics import MuJoCoReplay
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        qpos = model.qpos0.copy()
        qvel = np.zeros(model.nv)
        qvel[0] = self.velocity
        if self.category == "causal_ordering":
            qvel[6] = self.velocity * 0.6
        replay = MuJoCoReplay(xml_path, qpos, qvel, np.zeros((self.horizon, model.nu)), (),
                              self.physics_hz, self.control_hz, self.seed)
        replay.replay()
        if max(replay.initial_penetration, replay.penetrations.max(initial=0)) > replay.max_pen:
            raise ValueError("legal scene has excessive contact penetration")
        return replay


def read_spec(path):
    spec = json.loads(Path(path).read_text())
    if spec.get("version") == 3:
        from .corpus_v3 import validate_spec
        return validate_spec(spec)
    if spec.get("version") != 1 or len(spec.get("categories", [])) != 5 or set(spec["categories"]) != set(FAMILIES):
        raise ValueError("scene spec version 1 and all five categories required")
    for key in ("radius", "mass", "velocity", "spacing"):
        value = spec.get(key)
        if (not isinstance(value, list) or len(value) != 2 or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)
                or not 0 < value[0] <= value[1]):
            raise ValueError(f"invalid sampling range: {key}")
    if spec["velocity"][1] * 16 / 15 - 0.7 + 2 * spec["radius"][1] >= spec["spacing"][0]:
        raise ValueError("sampling ranges permit a collision in the legal reference")
    if not 32 <= spec.get("width", 128) <= 448 or not 32 <= spec.get("height", 96) <= 256:
        raise ValueError("render size exceeds supported limits")
    return spec


def sample_scene(spec, index, seed, shift="id", shift_spec=None):
    category = spec["categories"][index % len(FAMILIES)]
    identity = fingerprint({"spec": spec, "index": index, "seed": seed, "shift": shift,
                            "shift_spec": shift_spec})
    rng = random.Random(int(identity[:16], 16))
    values = {k: round(rng.uniform(*spec[k]), 6) for k in ("radius", "mass", "velocity", "spacing")}
    tint, bodies = rng.randrange(20), 2
    if shift != "id":
        if shift not in {"geometry", "dynamics", "appearance", "composition"}:
            raise ValueError("unknown OOD axis")
        settings = shift_spec["ood"][shift]
        for key in values:
            values[key] *= settings.get(key + "_scale", 1.0)
        tint = settings.get("tint", tint)
        bodies = settings.get("bodies", bodies)
    if bodies not in {2, 3} or values["radius"] <= 0 or values["mass"] <= 0:
        raise ValueError("invalid shifted scene parameters")
    return Scene(f"{shift}-{category}-{identity[:16]}", category, seed, tint=tint, bodies=bodies,
                 shift=shift, **values)
