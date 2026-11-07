import os
from pathlib import Path

import habitat_sim
import magnum as mn
from habitat_sim.utils.settings import built_with_bullet
from PIL import Image


# ==========================================================
# 1. 路径配置
# ==========================================================

LAB_DIR = Path("/home/ymq/Habitat/data/my_lab")

OBJECT_DIR = Path("/home/ymq/Habitat/data/assets/objects")

STAGE_CONFIG = LAB_DIR / "room_base_8x6.stage_config.json"

BOX_CONFIG = OBJECT_DIR / "box.object_config.json"

OUTPUT_DIR = LAB_DIR / "box_test_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. 检查文件
# ==========================================================

print("\n========== CHECK FILES ==========")

for p in [STAGE_CONFIG, BOX_CONFIG]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    print("[OK]", p)

print("Bullet available:", built_with_bullet)


# ==========================================================
# 3. 配置 Habitat-Sim
# ==========================================================

sim_cfg = habitat_sim.SimulatorConfiguration()

sim_cfg.scene_id = str(STAGE_CONFIG)

# 只有编译时支持 Bullet，才开启物理仿真
sim_cfg.enable_physics = built_with_bullet

sim_cfg.override_scene_light_defaults = True

sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY


# ==========================================================
# 4. 配置 RGB 相机
# ==========================================================

rgb_spec = habitat_sim.CameraSensorSpec()

rgb_spec.uuid = "rgb"

rgb_spec.sensor_type = habitat_sim.SensorType.COLOR

rgb_spec.resolution = [720, 960]

rgb_spec.position = [0.0, 1.3, 0.0]

rgb_spec.orientation = [0.0, 0.0, 0.0]

rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE


# ==========================================================
# 5. 创建 Agent
# ==========================================================

agent_cfg = habitat_sim.agent.AgentConfiguration()

agent_cfg.sensor_specifications = [rgb_spec]

cfg = habitat_sim.Configuration(
    sim_cfg,
    [agent_cfg]
)


# ==========================================================
# 6. 初始化仿真器
# ==========================================================

print("\n========== INITIALIZE SIMULATOR ==========")

sim = habitat_sim.Simulator(cfg)

try:

    print("[OK] Simulator initialized")


    # ======================================================
    # 7. 加载物体模板
    # ======================================================

    print("\n========== LOAD BOX TEMPLATE ==========")

    obj_template_mgr = sim.get_object_template_manager()

    template_ids = obj_template_mgr.load_configs(
        str(BOX_CONFIG)
    )

    if len(template_ids) == 0:
        raise RuntimeError("No box template loaded!")

    template_id = template_ids[0]

    print("Template IDs:", template_ids)

    print(
        "Template handle:",
        obj_template_mgr.get_template_handle_by_id(template_id)
    )


    # ======================================================
    # 8. 创建物体实例
    # ======================================================

    print("\n========== ADD BOX TO SCENE ==========")

    rigid_obj_mgr = sim.get_rigid_object_manager()

    box = rigid_obj_mgr.add_object_by_template_id(
        template_id
    )

    if box is None:
        raise RuntimeError("Failed to create box object!")

    # 使用 KINEMATIC，方便后续直接修改位置
    if built_with_bullet:
        box.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    # Blender 立方体尺寸为 0.5 x 0.5 x 0.5 m
    # 地板上表面为 Y=0
    # 因此立方体中心高度为 0.25 m

    box.translation = mn.Vector3(0.0, 0.25, 0.0)

    print("Object ID:", box.object_id)

    print("Initial position:", box.translation)

    print("Object handles:", rigid_obj_mgr.get_object_handles())


    # ======================================================
    # 9. 设置相机位置
    # ======================================================

    print("\n========== SET CAMERA ==========")

    agent = sim.get_agent(0)

    state = habitat_sim.AgentState()

    # 相机位于房间后侧，朝向房间中央
    state.position = [0.0, 0.0, 2.5]

    # 默认朝向 -Z
    agent.set_state(state)

    print("Camera agent position:", state.position)


    # ======================================================
    # 10. 获取 RGB 图像
    # ======================================================

    print("\n========== RENDER BOX ==========")

    obs = sim.get_sensor_observations()

    rgb = obs["rgb"]

    Image.fromarray(rgb[:, :, :3]).save(
        OUTPUT_DIR / "box_center.png"
    )

    print(
        "[OK] Saved:",
        OUTPUT_DIR / "box_center.png"
    )


    # ======================================================
    # 11. 修改物体位置
    # ======================================================

    print("\n========== MOVE BOX ==========")

    # 将立方体沿 X 轴移动 0.8 m

    box.translation = mn.Vector3(0.8, 0.25, 0.0)

    print("New position:", box.translation)

    obs = sim.get_sensor_observations()

    rgb = obs["rgb"]

    Image.fromarray(rgb[:, :, :3]).save(
        OUTPUT_DIR / "box_moved.png"
    )

    print(
        "[OK] Saved:",
        OUTPUT_DIR / "box_moved.png"
    )


    # ======================================================
    # 12. 删除物体测试
    # ======================================================

    print("\n========== REMOVE BOX ==========")

    # 先不删除，保留物体直到截图生成完成
    # 实际删除时使用：
    #
    # rigid_obj_mgr.remove_object_by_id(box.object_id)

    print("Box remains in simulation.")

    print("\n========== ALL TESTS FINISHED ==========")

finally:

    sim.close()
