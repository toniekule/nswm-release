"""Local, revision-pinned robot assets and fixed-action position tasks."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .physics import MuJoCoReplay, PositionAssertion
from .schema import Commitment, Interval, Query, Verdict, fingerprint


def inspect_robot_assets(root, receipt_path):
    from .train import confined_media_path, sha256
    root = Path(root).resolve()
    receipt = json.loads(Path(receipt_path).read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", receipt.get("revision", "")) or not receipt.get("repository"):
        raise ValueError("robot receipt needs an immutable repository revision")
    files, licenses = receipt.get("files", {}), receipt.get("licenses", [])
    if not files or not licenses:
        raise ValueError("robot receipt requires asset hashes and per-model licenses")
    for name, expected in files.items():
        if sha256(confined_media_path(root, name)) != expected:
            raise ValueError("robot asset digest differs: " + name)
    for license in licenses:
        if not license.get("spdx") or license.get("file") not in files:
            raise ValueError("robot license is not covered by the asset receipt")
    xml = confined_media_path(root, receipt["xml"])
    if receipt["xml"] not in files:
        raise ValueError("robot XML is not covered by the asset receipt")
    documents = []
    directories = {kind: "" for kind in ("meshdir", "texturedir", "assetdir")}
    for name in files:
        if not name.endswith(".xml"):
            continue
        source = confined_media_path(root, name)
        if source.parent != xml.parent:
            raise ValueError("robot XML files must share the registered model directory")
        document = ET.parse(source).getroot()
        compiler = document.find("compiler")
        if compiler is not None:
            for key in directories:
                value = compiler.get(key)
                if value is not None:
                    if directories[key] and directories[key] != value:
                        raise ValueError("robot compiler asset directories disagree")
                    directories[key] = value
        documents.append((source, document))
    for source, document in documents:
        for element in document.iter():
            if "file" not in element.attrib:
                continue
            prefix = directories["meshdir"] if element.tag == "mesh" else directories["texturedir"] if element.tag == "texture" else ""
            if element.tag in {"mesh", "texture", "hfield"}:
                prefix = prefix or directories["assetdir"]
            target = (source.parent / prefix / element.attrib["file"]).resolve()
            if not target.is_relative_to(root) or str(target.relative_to(root)) not in files:
                raise ValueError("robot XML references an unregistered or external asset")
    return {"root": str(root), "xml": str(xml), "identity": fingerprint(receipt), "receipt": receipt}


def register_robot_assets(root, xml, repository, revision, licenses, output):
    from .train import confined_media_path, sha256
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("robot receipt destination exists")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not repository or not licenses:
        raise ValueError("robot registration requires repository, full revision and per-model licenses")
    confined_media_path(root, xml)
    members = sorted(p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts)
    files = {str(p.relative_to(root)): sha256(confined_media_path(root, str(p.relative_to(root)))) for p in members}
    receipt = {"version": 1, "repository": repository, "revision": revision, "xml": xml,
               "licenses": licenses, "files": files}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        inspect_robot_assets(root, output)
    except Exception:
        output.unlink()
        raise
    return {"receipt": str(output), "identity": fingerprint(receipt), "files": len(files)}


@dataclass(frozen=True)
class PositionGoal:
    body: str
    frame: int
    axis: int
    interval: Interval


class RobotPositionTask:
    def __init__(self, assets, qpos, qvel, actions, assertions, goals, seed=0, maximum_penetration=0.001):
        if not goals or not assertions:
            raise ValueError("robot task requires explicit position assertions and task goals")
        self.assets, self.goals = assets, tuple(goals)
        self.replay = MuJoCoReplay(assets["xml"], qpos, qvel, actions, tuple(assertions),
                                   seed=seed, maximum_penetration=maximum_penetration)
        self.positions = self.replay.replay()
        self.query = Query("robot-" + self.replay.identity, tuple(tuple(a) for a in self.replay.actions), {
            "replay_identity": self.replay.identity, "asset_identity": assets["identity"],
            "complete_position_domain": True, "position_goals": [asdict(g) for g in self.goals],
            "position_assertions": [asdict(a) for a in self.replay.assertions]}, len(actions))
        self.goal_status = self._check_goals()

    @classmethod
    def from_spec(cls, path, asset_root, receipt_path):
        spec = json.loads(Path(path).read_text())
        if spec.get("version") != 1:
            raise ValueError("unsupported robot task specification")
        assets = inspect_robot_assets(asset_root, receipt_path)
        goals = tuple(PositionGoal(g["body"], g["frame"], g["axis"], Interval(**g["interval"])) for g in spec["goals"])
        assertions = tuple(PositionAssertion(**a) for a in spec["assertions"])
        return cls(assets, spec["initial_qpos"], spec["initial_qvel"], spec["actions"], assertions, goals,
                   spec.get("seed", 0), spec.get("maximum_penetration", 0.001))

    def _check_goals(self):
        import mujoco
        for goal in self.goals:
            body = mujoco.mj_name2id(self.replay.model, mujoco.mjtObj.mjOBJ_BODY, goal.body)
            if body < 0 or goal.axis not in {0, 1, 2} or not 0 <= goal.frame < len(self.positions):
                return Verdict.UNKNOWN
            if not goal.interval.contains(Interval.point(float(self.positions[goal.frame, body, goal.axis]))):
                return Verdict.INVALID
        return Verdict.VALID

    def oracle(self, query, commitment):
        if (query.binding != self.query.binding or self.assets["identity"] != query.premises["asset_identity"]
                or fingerprint([asdict(g) for g in self.goals]) != fingerprint(query.premises["position_goals"])
                or fingerprint([asdict(a) for a in self.replay.assertions]) != fingerprint(query.premises["position_assertions"])):
            return Verdict.UNKNOWN
        replay_verdict = self.replay.oracle(query, commitment)
        if replay_verdict == Verdict.UNKNOWN or self.goal_status == Verdict.UNKNOWN:
            return Verdict.UNKNOWN
        return Verdict.INVALID if self.goal_status == Verdict.INVALID else replay_verdict

    def reference(self, tolerance=1e-6):
        import mujoco
        slots = {}
        for assertion in self.replay.assertions:
            body = mujoco.mj_name2id(self.replay.model, mujoco.mjtObj.mjOBJ_BODY, assertion.body)
            if body < 0 or assertion.axis not in {0, 1, 2} or not 0 <= assertion.frame < len(self.positions):
                raise ValueError("robot position assertion is outside the replay domain")
            value = float(self.positions[assertion.frame, body, assertion.axis])
            slots[assertion.slot] = Interval(str(value - tolerance), str(value + tolerance))
        return Commitment(slots)
