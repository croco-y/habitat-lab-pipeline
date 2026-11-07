#!/usr/bin/env python3
"""Dynamically spawn Kenney furniture, rebuild NavMesh, and validate clearance.

Usage:
    /home/ymq/anaconda3/envs/habitat/bin/python \\
        /home/ymq/Habitat/scripts/spawn_furniture_and_navmesh.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import magnum as mn
import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.settings import built_with_bullet

ROOT = Path("/home/ymq/Habitat")
LAB = ROOT / "data/my_lab"
FURNITURE_DIR = ROOT / "data/assets/furniture/kenney"
STAGE = LAB / "room_base_8x6.stage_config.json"
DATASET = LAB / "lab.scene_dataset_config.json"
CATALOG = FURNITURE_DIR / "catalog.json"
OUTPUT_NAVMESH = LAB / "room_base_8x6_with_furniture.navmesh"
OUTPUT_DIR = LAB / "furniture_nav_test"
LAYOUT_OUT = LAB / "scenes" / "lab_layout_furniture.scene_instance.json"


# Default lab layout: positions are on the floor plane (y will be set to half-height
# only if an asset origin were centered; our normalized assets sit on y=0).
DEFAULT_LAYOUT = [
    # name, x, z, yaw_deg
    ("desk", -2.2, -1.5, 0.0),
    ("chairDesk", -2.2, -0.6, 180.0),
    ("loungeSofa", 1.8, -1.6, -30.0),
    ("tableCoffee", 1.5, -0.3, 0.0),
    ("bookcaseOpen", -3.2, 0.0, 90.0),
    ("lampRoundFloor", 2.6, 0.8, 0.0),
    ("pottedPlant", -1.0, 2.0, 0.0),
    ("sideTable", 2.8, -1.6, 0.0),
    ("televisionModern", 0.0, -2.4, 0.0),
    ("trashcan", 3.0, 2.0, 0.0),
]


def quat_yaw(yaw_deg: float) -> mn.Quaternion:
    yaw = math.radians(yaw_deg)
    return mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0.0, 1.0, 0.0))


def load_catalog() -> dict:
    if not CATALOG.is_file():
        raise FileNotFoundError(
            f"Missing {CATALOG}. Run prepare_kenney_furniture.py first."
        )
    return json.loads(CATALOG.read_text())


def make_simulator() -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(STAGE)
    sim_cfg.enable_physics = built_with_bullet
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb.resolution = [480, 640]
    rgb.position = [0.0, 1.3, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def load_templates(sim: habitat_sim.Simulator, names: list[str]) -> dict[str, str]:
    """Load object configs and return {stem: template_handle}."""
    otm = sim.get_object_template_manager()
    handles = {}
    for name in names:
        cfg = FURNITURE_DIR / f"{name}.object_config.json"
        if not cfg.is_file():
            print(f"[WARN] skip missing config: {cfg}")
            continue
        ids = otm.load_configs(str(cfg))
        if not ids:
            print(f"[WARN] failed to load: {cfg}")
            continue
        handle = otm.get_template_handle_by_id(ids[0])
        handles[name] = handle
        print(f"[OK] template {name} -> {handle}")
    return handles


def spawn_layout(sim: habitat_sim.Simulator, handles: dict[str, str], catalog: dict):
    rom = sim.get_rigid_object_manager()
    spawned = []
    for name, x, z, yaw in DEFAULT_LAYOUT:
        if name not in handles:
            print(f"[WARN] layout item missing template: {name}")
            continue
        obj = rom.add_object_by_template_handle(handles[name])
        if obj is None:
            print(f"[WARN] failed to spawn: {name}")
            continue

        meta = catalog["assets"].get(name, {})
        # Habitat translation is the COM. Normalized meshes have bottom at y=0,
        # so COM sits near half-height; place COM at half-height to sit on floor.
        half_h = float(meta.get("half_height_m", 0.4))
        y = half_h
        obj.translation = mn.Vector3(float(x), y, float(z))
        obj.rotation = quat_yaw(yaw)

        if built_with_bullet:
            obj.motion_type = habitat_sim.physics.MotionType.STATIC
            obj.collidable = True

        spawned.append(
            {
                "name": name,
                "object_id": int(obj.object_id),
                "translation": [float(x), y, float(z)],
                "yaw_deg": float(yaw),
                "size": meta.get("normalized_size"),
                "half_height_m": half_h,
                "motion_type": "STATIC",
            }
        )
        print(f"[SPAWN] {name} at ({x:.2f}, {y:.2f}, {z:.2f}) yaw={yaw}")
    return spawned


def recompute_navmesh(sim: habitat_sim.Simulator) -> bool:
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = 0.25
    settings.agent_height = 1.4
    settings.include_static_objects = True
    ok = sim.recompute_navmesh(sim.pathfinder, settings)
    print("NavMesh generated:", ok)
    return ok


def validate_navigation(sim: habitat_sim.Simulator, spawned: list[dict]) -> dict:
    pf = sim.pathfinder
    report = {
        "loaded": bool(pf.is_loaded),
        "num_islands": int(pf.num_islands) if pf.is_loaded else 0,
        "navigable_area": float(pf.navigable_area) if pf.is_loaded else 0.0,
        "free_center_navigable": False,
        "furniture_blocked": [],
        "path_found": False,
        "geodesic_distance": None,
        "path_longer_than_straight": False,
    }

    # NavMesh samples are slightly above the floor (often ~0.2 m).
    center = mn.Vector3(0.0, 0.2, 1.5)
    report["free_center_navigable"] = bool(pf.is_navigable(center, max_y_delta=0.3))
    print("Open area navigable:", report["free_center_navigable"], center)

    for item in spawned:
        x, _, z = item["translation"]
        p = mn.Vector3(x, 0.2, z)
        navigable = bool(pf.is_navigable(p, max_y_delta=0.3))
        snap = pf.snap_point(p)
        # Treat as blocked if not navigable, or snap moves away horizontally.
        shift = ((snap.x - x) ** 2 + (snap.z - z) ** 2) ** 0.5
        blocked = (not navigable) or shift > 0.20
        report["furniture_blocked"].append(
            {
                "name": item["name"],
                "position": [x, 0.2, z],
                "navigable": navigable,
                "snap_shift_m": float(shift),
                "blocked": bool(blocked),
            }
        )
        print(
            f"Furniture footprint {item['name']}: "
            f"blocked={blocked} navigable={navigable} snap_shift={shift:.3f}"
        )

    # Cross the furnished side of the room so the path must detour.
    start = pf.snap_point(mn.Vector3(-3.0, 0.2, -0.5))
    goal = pf.snap_point(mn.Vector3(3.0, 0.2, -0.5))
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = goal
    found = pf.find_path(path)
    report["path_found"] = bool(found)
    if found:
        straight = float((goal - start).length())
        geo = float(path.geodesic_distance)
        report["geodesic_distance"] = geo
        report["straight_distance"] = straight
        report["path_longer_than_straight"] = geo > straight + 0.05
        report["start"] = [float(start.x), float(start.y), float(start.z)]
        report["goal"] = [float(goal.x), float(goal.y), float(goal.z)]
        print(
            f"Path around furniture: found={found}, "
            f"geodesic={geo:.3f} m, straight={straight:.3f} m, "
            f"points={len(path.points)}"
        )
    else:
        print("Path around furniture: NOT FOUND")

    return report


def save_scene_instance(spawned: list[dict]) -> None:
    """Write a reusable scene_instance for the furniture layout."""
    LAYOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    instance = {
        "stage_instance": {"template_name": "lab_stage"},
        "object_instances": [],
    }
    for item in spawned:
        yaw = math.radians(item["yaw_deg"])
        # wxyz identity rotated about Y
        qw = math.cos(yaw / 2.0)
        qy = math.sin(yaw / 2.0)
        instance["object_instances"].append(
            {
                "template_name": item["name"],
                "translation": item["translation"],  # COM placement
                "rotation": [qw, 0.0, qy, 0.0],
                "motion_type": "STATIC",
            }
        )
    LAYOUT_OUT.write_text(json.dumps(instance, indent=2) + "\n")
    print("Wrote scene instance:", LAYOUT_OUT)


def capture_preview(sim: habitat_sim.Simulator) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array([0.0, 0.0, 2.4], dtype=np.float32)
    # Look toward -Z (default)
    agent.set_state(state)
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])[:, :, :3]
    out = OUTPUT_DIR / "furniture_overview.png"
    Image.fromarray(rgb).save(out)
    print("Saved preview:", out)


def update_dataset_config(template_names: list[str]) -> None:
    """Ensure lab.scene_dataset_config.json can find furniture configs."""
    data = json.loads(DATASET.read_text())
    obj = data.setdefault("objects", {})
    paths = obj.setdefault("paths", {})
    json_paths = paths.setdefault(".json", [])
    furniture_rel = "../assets/furniture/kenney"
    if furniture_rel not in json_paths:
        json_paths.append(furniture_rel)

    # Keep existing box entry; append furniture handles if missing.
    configs = obj.setdefault("configs", [])
    existing = {c.get("template_handle") for c in configs}
    for name in template_names:
        if name in existing:
            continue
        configs.append(
            {
                "original_file": f"{name}.object_config.json",
                "template_handle": name,
                "attributes": {},
            }
        )
    DATASET.write_text(json.dumps(data, indent=2) + "\n")
    print("Updated dataset config:", DATASET)


def main() -> None:
    for p in (STAGE, FURNITURE_DIR):
        if not p.exists():
            raise FileNotFoundError(p)

    catalog = load_catalog()
    wanted = [row[0] for row in DEFAULT_LAYOUT]

    print("Bullet:", built_with_bullet)
    sim = make_simulator()
    try:
        handles = load_templates(sim, wanted)
        if not handles:
            raise RuntimeError("No furniture templates loaded")

        spawned = spawn_layout(sim, handles, catalog)
        if not spawned:
            raise RuntimeError("No furniture spawned")

        if not recompute_navmesh(sim):
            raise RuntimeError("NavMesh generation failed")

        saved = sim.pathfinder.save_nav_mesh(str(OUTPUT_NAVMESH))
        print("Saved NavMesh:", OUTPUT_NAVMESH, "| success:", saved)

        report = validate_navigation(sim, spawned)
        capture_preview(sim)
        save_scene_instance(spawned)
        update_dataset_config([s["name"] for s in spawned])

        report_path = OUTPUT_DIR / "nav_validation.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("Validation report:", report_path)

        if not report["path_found"]:
            print(
                "\n[RESULT] Furniture spawned, but no corridor path was found. "
                "Try adjusting DEFAULT_LAYOUT spacing."
            )
            sys.exit(2)

        print("\n[RESULT] Furniture spawned + NavMesh OK, robot can path around.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
