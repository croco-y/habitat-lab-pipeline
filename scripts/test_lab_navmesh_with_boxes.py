from pathlib import Path

import habitat_sim
import magnum as mn

from habitat_sim.utils.settings import built_with_bullet


# ============================================================
# 1. 路径
# ============================================================

ROOT = Path("/home/ymq/Habitat/data/my_lab")

DATASET_CONFIG = ROOT / "lab.scene_dataset_config.json"

SCENE_INSTANCE = ROOT / "scenes/lab_layout_01.scene_instance.json"

OUTPUT_NAVMESH = ROOT / "room_base_8x6_with_boxes.navmesh"


# ============================================================
# 2. 检查文件
# ============================================================

for p in [DATASET_CONFIG, SCENE_INSTANCE]:

    if not p.exists():
        raise FileNotFoundError(p)

print("Dataset:", DATASET_CONFIG)
print("Scene:", SCENE_INSTANCE)
print("Bullet:", built_with_bullet)


# ============================================================
# 3. 初始化 Habitat-Sim
# ============================================================

sim_cfg = habitat_sim.SimulatorConfiguration()

sim_cfg.scene_dataset_config_file = str(DATASET_CONFIG)

sim_cfg.scene_id = str(SCENE_INSTANCE)

sim_cfg.enable_physics = built_with_bullet

sim_cfg.override_scene_light_defaults = True

sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

agent_cfg = habitat_sim.agent.AgentConfiguration()

cfg = habitat_sim.Configuration(
    sim_cfg,
    [agent_cfg]
)

sim = habitat_sim.Simulator(cfg)


try:

    # ========================================================
    # 4. 检查场景中的物体
    # ========================================================

    print("\n========== OBJECT INSPECTION ==========")

    rigid_mgr = sim.get_rigid_object_manager()

    handles = rigid_mgr.get_object_handles()

    print("Object count:", len(handles))

    for handle in handles:

        obj = rigid_mgr.get_object_by_handle(handle)

        print("\nHandle:", handle)
        print("Object ID:", obj.object_id)
        print("Position:", obj.translation)
        print("Motion type:", obj.motion_type)

    if len(handles) != 3:

        raise RuntimeError(
            f"Expected 3 boxes, but found {len(handles)} objects."
        )


    # ========================================================
    # 5. 配置 NavMesh
    # ========================================================

    print("\n========== NAVMESH SETTINGS ==========")

    nav_settings = habitat_sim.NavMeshSettings()

    nav_settings.set_defaults()

    # 纳入 STATIC 类型的物体
    nav_settings.include_static_objects = True

    print("Agent radius:", nav_settings.agent_radius)

    print("Agent height:", nav_settings.agent_height)

    print(
        "Include static objects:",
        nav_settings.include_static_objects
    )


    # ========================================================
    # 6. 重新生成 NavMesh
    # ========================================================

    print("\n========== RECOMPUTE NAVMESH ==========")

    success = sim.recompute_navmesh(
        sim.pathfinder,
        nav_settings
    )

    print("NavMesh generated:", success)

    if not success:

        raise RuntimeError("NavMesh generation failed!")


    # ========================================================
    # 7. 检查三个 Box 所在位置
    # ========================================================

    print("\n========== NAVIGATION TEST ==========")

    test_points = {
        "Box_01": (-1.5, 0.2, 0.0),
        "Box_02": (0.0, 0.2, -1.0),
        "Box_03": (1.5, 0.2, 0.0),

        "Free_Area": (0.0, 0.2, 2.0)
    }

    for name, xyz in test_points.items():

        point = mn.Vector3(*xyz)

        navigable = sim.pathfinder.is_navigable(point)

        snapped = sim.pathfinder.snap_point(point)

        print(f"\n{name}")

        print("Position:", xyz)

        print("Navigable:", navigable)

        print("Nearest navigable point:", snapped)


    # ========================================================
    # 8. 保存新 NavMesh
    # ========================================================

    print("\n========== SAVE NAVMESH ==========")

    saved = sim.pathfinder.save_nav_mesh(
        str(OUTPUT_NAVMESH)
    )

    print("Saved:", saved)

    print("Output:", OUTPUT_NAVMESH)

    print("\n========== FINISHED ==========")


finally:

    sim.close()
