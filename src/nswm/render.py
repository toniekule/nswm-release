"""Headless orthographic rendering and optional MuJoCo offscreen rendering."""
from pathlib import Path


def background_texture(scene, style, width, height):
    import numpy as np
    from PIL import Image
    from .schema import fingerprint
    seed = int(fingerprint({"root": scene.family_id, "style": style})[:16], 16)
    pattern = np.random.default_rng(seed).integers(0, 13, size=(8, 8, 3), dtype=np.uint8)
    pattern += np.asarray([25, 30 + style * 8, 40], dtype=np.uint8)
    return Image.fromarray(pattern).resize((width, height), Image.Resampling.BILINEAR)


def render_clip(scene, poses, visible, destination, width=128, height=96,
                tint=0, backend="software", xml_path=None):
    import numpy as np
    from PIL import Image, ImageDraw
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("render output exists")
    if backend not in {"software", "mujoco"} or len(poses) != len(visible):
        raise ValueError("invalid renderer or frame visibility")
    destination.mkdir(parents=True)
    paths = []
    model = renderer = data = None
    if backend == "mujoco":
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        renderer = mujoco.Renderer(model, height=height, width=width)
        data = mujoco.MjData(model)
    try:
        for frame, (positions, mask) in enumerate(zip(poses, visible)):
            if renderer:
                for i, name in enumerate("ABC"[:scene.bodies]):
                    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name + ".free")
                    data.qpos[model.jnt_qposadr[joint]:model.jnt_qposadr[joint] + 3] = positions[i]
                    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name + ".sphere")
                    model.geom_rgba[geom, :3] = np.array((0.8, 0.3 + (tint % 40) / 100, 0.3)) if i == 0 else (0.2, 0.5, 0.9)
                    model.geom_rgba[geom, 3] = float(mask[i])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="overview")
                image = Image.fromarray(renderer.render())
            else:
                image = Image.new("RGB", (width, height), (22 + tint % 20, 28, 38))
                draw = ImageDraw.Draw(image)
                scale = min(width / 4.8, height / 2.5)
                for i in sorted(range(scene.bodies), key=lambda k: positions[k][2]):
                    if not mask[i]:
                        continue
                    x, y, z = positions[i]
                    cx, cy = width / 2 + x * scale, height * 0.7 - (y + z * 0.5) * scale
                    radius = scene.radius * scale
                    color = (215, 100 + tint % 60, 55) if i == 0 else (60, 150, 220)
                    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
                    draw.ellipse((cx - radius * 0.6, cy - radius * 0.6, cx, cy), fill=tuple(min(255, c + 25) for c in color))
            path = destination / f"{frame:05d}.png"
            image.save(path)
            paths.append(path)
    finally:
        if renderer:
            renderer.close()
    return paths


def render_mechanism(scene, trace, destination, width=128, height=96, style=0, backend="software"):
    import numpy as np
    from PIL import Image, ImageDraw
    poses = np.asarray(trace["positions"], dtype=float)
    visible, exists = np.asarray(trace["visible"]), np.asarray(trace["exists"])
    if (not 1 <= len(poses) <= 17 or poses.shape[1:] != (scene.bodies, 3)
            or not np.isfinite(poses).all() or visible.shape != poses.shape[:2]
            or exists.shape != visible.shape or np.any(visible & (exists == 0))):
        raise ValueError("invalid rendering state or identity visibility")
    if backend not in {"software", "mujoco"} or style not in {0, 1} or width < 32 or height < 32:
        raise ValueError("invalid mechanism rendering configuration")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("render output exists")
    destination.mkdir(parents=True)
    model = renderer = data = None
    if backend == "mujoco":
        import mujoco
        model = mujoco.MjModel.from_xml_string(scene.xml())
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
    paths = []
    tint = scene.tint + 37 * style
    xspan = max(2.5, 1 + scene.speed * scene.duration + scene.force_limit / scene.mass * scene.duration ** 2)
    yspan = max(1.0, scene.lanes * scene.lane_pitch + 4 * scene.radius)
    scale = min(width / (2 * xspan), height / (yspan + 0.4))
    background = background_texture(scene, style, width, height)
    try:
        for frame, positions in enumerate(trace["positions"]):
            mask = np.asarray(trace["visible"][frame], dtype=bool) & np.asarray(trace["exists"][frame], dtype=bool)
            if renderer is not None:
                data.qpos[:] = model.qpos0
                for i, name in enumerate(scene.names):
                    suffix = ".slide" if scene.category == "resource_reachability" else ".free"
                    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name + suffix)
                    address = model.jnt_qposadr[joint]
                    if suffix == ".slide":
                        data.qpos[address] = positions[i][0] - scene.body_position(i, 0)[0]
                    else:
                        data.qpos[address:address + 3] = positions[i]
                    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name + ".sphere")
                    model.geom_rgba[geom] = ((0.7, 0.25 + tint % 50 / 100, 0.25, float(mask[i]))
                                            if i % scene.lane_size == 0 else (0.2, 0.55, 0.85, float(mask[i])))
                mujoco.mj_forward(model, data)
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = (0, 0, 0.2)
                camera.distance, camera.azimuth, camera.elevation = max(4, yspan * 1.8), 90, -65
                renderer.update_scene(data, camera=camera)
                pixels = renderer.render().copy()
                renderer.enable_segmentation_rendering()
                try:
                    segments = renderer.render()
                finally:
                    renderer.disable_segmentation_rendering()
                sky = segments[:, :, 0] < 0
                pixels[sky] = np.asarray(background)[sky]
                image = Image.fromarray(pixels)
            else:
                image = background.copy()
                draw = ImageDraw.Draw(image)
                for i, (x, y, _) in enumerate(positions):
                    if not mask[i]:
                        continue
                    cx, cy, radius = width / 2 + x * scale, height / 2 - y * scale, max(1, scene.radius * scale)
                    color = (215, 95 + tint % 60, 55) if i % scene.lane_size == 0 else (65, 150, 220)
                    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
                if scene.category == "permanence":
                    for lane in range(scene.lanes):
                        y = scene.body_position(lane, 0)[1]
                        cy = height / 2 - y * scale
                        draw.rectangle((width / 2 - 0.13 * scale, cy - 0.15 * scale,
                                        width / 2 + 0.13 * scale, cy + 0.15 * scale), fill=(90, 100, 110))
            path = destination / f"{frame:05d}.png"
            image.save(path)
            paths.append(path)
    finally:
        if renderer is not None:
            renderer.close()
    return paths
