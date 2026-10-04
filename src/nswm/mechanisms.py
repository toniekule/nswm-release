"""Independent lanes of fixed-context MuJoCo mechanisms and state ledgers."""
from dataclasses import asdict, dataclass
import math

from .data import FAMILIES
from .schema import fingerprint


def quantity_slot(scene, lane):
    suffix = {"contact_solidity": "minimum_separation", "conservation": "momentum.x.final",
              "permanence": "A.exists.final", "causal_ordering": "contact_chain.delay",
              "resource_reachability": "A.x.final"}[scene.category]
    return f"L{lane}.{suffix}"


def slot_schema(scene):
    _, units = physical_limits(scene)
    return {**{key: {"type": scene.category, "unit": unit, "time": [0, scene.duration],
                    "entities": list(scene.names[lane * scene.lane_size:(lane + 1) * scene.lane_size]),
                    "sample_hz": scene.control_hz}
               for lane, (key, unit) in enumerate(units.items())},
            "initial.x": {"type": "position", "unit": "m", "time": [0, 0], "entities": ["L0.A"]}}


@dataclass(frozen=True)
class MechanismScene:
    family_id: str
    category: str
    seed: int
    lanes: int = 1
    radius: float = 0.08
    mass: float = 1.0
    speed: float = 0.8
    gap: float = 0.12
    force_limit: float = 0.5
    drive_fraction: float = 0.8
    horizon: int = 16
    control_hz: int = 15
    physics_hz: int = 240
    tolerance: float = 0.005
    contact_margin: float = 0.002
    impact_fraction: float = 0.0
    shift: str = "id"
    tint: int = 0

    def __post_init__(self):
        if (not self.family_id or self.category not in FAMILIES or not 1 <= self.lanes <= 6
                or (self.horizon, self.control_hz, self.physics_hz) != (16, 15, 240)):
            raise ValueError("unsupported mechanism, lane count or observation clock")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.radius, self.mass, self.speed, self.gap, self.force_limit, self.tolerance)):
            raise ValueError("mechanism parameters must be finite and positive")
        if (not 0 < self.drive_fraction <= 1 or self.gap / self.speed > 0.35
                or not math.isfinite(self.contact_margin) or not 0 <= self.contact_margin < self.radius
                or not math.isfinite(self.impact_fraction) or not 0 <= self.impact_fraction <= 2):
            raise ValueError("invalid drive fraction or collision timing")
        if self.shift not in {"id", "geometry", "dynamics", "appearance", "composition"}:
            raise ValueError("unknown mechanism shift")

    @property
    def lane_size(self):
        base = 3 if self.category == "causal_ordering" else 2 if self.category in {"contact_solidity", "conservation"} else 1
        return base + int(self.shift == "composition")

    @property
    def names(self):
        return tuple(f"L{lane}.{name}" for lane in range(self.lanes) for name in "ABCD"[:self.lane_size])

    @property
    def bodies(self):
        return len(self.names)

    @property
    def duration(self):
        return self.horizon / self.control_hz

    @property
    def lane_pitch(self):
        return max(0.7, 10 * self.radius) if self.category == "contact_solidity" else max(0.55, 5 * self.radius)

    def body_position(self, lane, member):
        x = -0.5 + member * (2 * self.radius + self.gap)
        if self.category == "permanence":
            x = -self.speed * self.duration / 2 if member == 0 else x
        y = (lane - (self.lanes - 1) / 2) * self.lane_pitch
        if self.category == "contact_solidity" and member:
            y += self.impact_fraction * 2 * self.radius
        return (x, y, 0.2)

    def xml(self):
        bodies, motors = [], []
        for lane in range(self.lanes):
            for member, suffix in enumerate("ABCD"[:self.lane_size]):
                name = f"L{lane}.{suffix}"
                x, y, z = self.body_position(lane, member)
                joint = (f'<joint name="{name}.slide" type="slide" axis="1 0 0" damping="0"/>'
                         if self.category == "resource_reachability" else f'<freejoint name="{name}.free"/>')
                collision = 'contype="0" conaffinity="0"' if self.category in {"permanence", "resource_reachability"} else ''
                bodies.append(f'<body name="{name}" pos="{x} {y} {z}">{joint}'
                    f'<geom name="{name}.sphere" type="sphere" size="{self.radius}" mass="{self.mass}" '
                    f'friction="0 0 0" margin="{self.contact_margin}" solref="0.002 1" {collision}/></body>')
            if self.category in {"conservation", "resource_reachability"}:
                name = f"L{lane}.A"
                joint_name = name + (".slide" if self.category == "resource_reachability" else ".free")
                gear = "1" if self.category == "resource_reachability" else "1 0 0 0 0 0"
                motors.append(f'<motor name="L{lane}.drive" joint="{joint_name}" gear="{gear}" '
                    f'ctrllimited="true" ctrlrange="{-self.force_limit} {self.force_limit}"/>')
            if self.category == "permanence":
                y = self.body_position(lane, 0)[1]
                bodies.append(f'<geom name="L{lane}.occluder" type="box" pos="0 {y - 0.15} 0.25" '
                    f'size="0.13 0.025 0.25" contype="0" conaffinity="0" rgba="0.3 0.35 0.4 1"/>')
        return ('<mujoco model="nswm-mechanisms"><option gravity="0 0 0" integrator="RK4" '
            f'timestep="{1 / self.physics_hz}"/><visual><global offwidth="448" offheight="256"/></visual>'
            '<worldbody><light pos="0 -1 4"/><camera name="overview" pos="0 -5 4" xyaxes="1 0 0 0 1 1.25"/>'
            + ''.join(bodies) + '</worldbody><actuator>' + ''.join(motors) + '</actuator></mujoco>')

    def initial(self, model):
        import mujoco
        import numpy as np
        qpos, qvel = model.qpos0.copy(), np.zeros(model.nv)
        if self.category != "resource_reachability":
            for lane in range(self.lanes):
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"L{lane}.A.free")
                qvel[model.jnt_dofadr[joint]] = self.speed
        drive = self.force_limit * self.drive_fraction
        return qpos, qvel, np.full((self.horizon, model.nu), drive)


def simulate(scene, actions=None, initial_qpos=None, initial_qvel=None):
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_string(scene.xml())
    qpos, qvel, controls = scene.initial(model)
    if initial_qpos is not None:
        qpos = np.asarray(initial_qpos, dtype=float)
    if initial_qvel is not None:
        qvel = np.asarray(initial_qvel, dtype=float)
    if actions is not None:
        controls = np.asarray(actions, dtype=float)
    if (qpos.shape != (model.nq,) or qvel.shape != (model.nv,) or controls.shape != (scene.horizon, model.nu)
            or any(not np.isfinite(a).all() for a in (qpos, qvel, controls))
            or (model.nu and np.abs(controls).max() > scene.force_limit)):
        raise ValueError("replay state or bounded actions differ from mechanism dimensions")
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, data)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in scene.names]
    positions, velocities, qpositions, states, events = [], [], [], [], []
    seen = set()
    max_penetration = 0.0
    state_kind = mujoco.mjtState.mjSTATE_INTEGRATION

    def capture():
        positions.append(data.xpos[ids].copy().tolist())
        linear = []
        for body in ids:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            linear.append(velocity[3:].tolist())
        velocities.append(linear)
        qpositions.append(data.qpos.copy().tolist())
        state = np.empty(mujoco.mj_stateSize(model, state_kind))
        mujoco.mj_getState(model, data, state, state_kind)
        states.append(state.tolist())

    capture()
    for control in controls:
        data.ctrl[:] = control
        for _ in range(scene.physics_hz // scene.control_hz):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            for contact_index, contact in enumerate(data.contact):
                max_penetration = max(max_penetration, -float(contact.dist))
                names = tuple(sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[g])) for g in (contact.geom1, contact.geom2)))
                if names not in seen:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    events.append({"entities": names, "time": float(data.time), "normal_force": float(force[0])})
                    seen.add(names)
        capture()
    poses = np.asarray(positions)
    visible = np.ones(poses.shape[:2], dtype=bool)
    if scene.category == "permanence":
        visible = np.abs(poses[:, :, 0]) > 0.13 + scene.radius
    trace = {"names": list(scene.names), "positions": positions, "velocities": velocities,
        "exists": np.ones(poses.shape[:2], dtype=int).tolist(), "visible": visible.tolist(),
        "events": events, "qpos": qpositions, "integration_states": states,
        "state_kind": int(state_kind), "max_penetration": max_penetration}
    return {"scene": asdict(scene), "initial_qpos": qpos.tolist(), "initial_qvel": qvel.tolist(),
        "actions": controls.tolist(), "engine": {"name": "mujoco", "version": mujoco.__version__,
        "xml_sha256": fingerprint(scene.xml())}, "trace": trace}


def measure(scene, trace):
    import numpy as np
    poses, velocities = np.asarray(trace["positions"]), np.asarray(trace["velocities"])
    exists, visible = np.asarray(trace["exists"]), np.asarray(trace["visible"])
    if (trace["names"] != list(scene.names) or poses.shape != (scene.horizon + 1, scene.bodies, 3)
            or velocities.shape != poses.shape or exists.shape != poses.shape[:2] or visible.shape != exists.shape
            or not np.isfinite(poses).all() or not np.isfinite(velocities).all()
            or not np.isin(exists, (0, 1)).all() or np.any(visible & (exists == 0))):
        raise ValueError("candidate state ledger has invalid shape, identity or values")
    result = {}
    for lane in range(scene.lanes):
        start = lane * scene.lane_size
        key = quantity_slot(scene, lane)
        if scene.category == "contact_solidity":
            result[key] = min(float(np.linalg.norm(poses[:, start + a] - poses[:, start + b], axis=1).min())
                              for a in range(scene.lane_size) for b in range(a + 1, scene.lane_size))
        elif scene.category == "conservation":
            result[key] = float(scene.mass * velocities[-1, start:start + scene.lane_size, 0].sum())
        elif scene.category == "permanence":
            result[key] = int(exists[-1, start])
        elif scene.category == "causal_ordering":
            times = {}
            last = "ABCD"[scene.lane_size - 2:scene.lane_size]
            for suffix, pair in (("cause", (f"L{lane}.A", f"L{lane}.B")),
                                 ("effect", (f"L{lane}.{last[0]}", f"L{lane}.{last[1]}"))):
                matches = [float(e["time"]) for e in trace["events"] if tuple(sorted(e["entities"])) == pair]
                if not matches or not all(math.isfinite(v) and 0 <= v <= scene.duration for v in matches):
                    raise ValueError("causal contact event coverage missing")
                times[suffix] = min(matches)
            result[key] = times["effect"] - times["cause"]
        else:
            result[key] = float(poses[-1, start, 0])
    result["initial.x"] = float(poses[0, 0, 0])
    return result


def physical_limits(scene):
    from .schema import Interval
    limits, units = {}, {}
    for lane in range(scene.lanes):
        key = quantity_slot(scene, lane)
        if scene.category == "contact_solidity":
            outer = (scene.lane_size - 1) * (2 * scene.radius + scene.gap) + 2 * scene.speed * scene.duration
            limits[key], units[key] = Interval(str(2 * scene.radius - scene.tolerance), str(outer)), "m"
        elif scene.category == "conservation":
            initial = scene.mass * scene.speed
            exchange = scene.force_limit * scene.duration + scene.tolerance
            limits[key], units[key] = Interval(str(initial - exchange), str(initial + exchange)), "kg*m/s"
        elif scene.category == "permanence":
            limits[key], units[key] = Interval.point(1), "count"
        elif scene.category == "causal_ordering":
            limits[key], units[key] = Interval.point(0) + Interval("0", str(scene.duration)), "s"
        else:
            start = scene.body_position(lane, 0)[0]
            reach = scene.force_limit / scene.mass * scene.duration ** 2 / 2 + scene.tolerance
            limits[key], units[key] = Interval(str(start - reach), str(start + reach)), "m"
    return limits, units
