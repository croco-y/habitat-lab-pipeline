from pathlib import Path
import math

import habitat_sim
import numpy as np
import quaternion
import magnum as mn
from PIL import Image


# ==================================================
# 1. 路径配置
# ==================================================

ROOT = Path("/home/ymq/Habitat/data/my_lab")

SCENE = ROOT / "room_base_8x6.stage_config.json"
NAVMESH = ROOT / "room_base_8x6.navmesh"
OUTPUT = ROOT / "debug_views"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. 配置模拟器
# ==================================================

sim_cfg = habitat_sim.SimulatorConfiguration()

sim_cfg.scene_id = str(SCENE)

sim_cfg.enable_physics = False

sim_cfg.override_scene_light_defaults = True
sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY


# ==================================================
# 3. RGB 相机
# ==================================================

rgb = habitat_sim.CameraSensorSpec()

rgb.uuid = "rgb"
rgb.sensor_type = habitat_sim.SensorType.COLOR
rgb.resolution = [480, 640]

# 相对于 Agent 的高度
rgb.position = [0.0, 1.3, 0.0]


# ==================================================
# 4. Depth 相机
# ==================================================

depth = habitat_sim.CameraSensorSpec()

depth.uuid = "depth"
depth.sensor_type = habitat_sim.SensorType.DEPTH
depth.resolution = [480, 640]

depth.position = [0.0, 1.3, 0.0]


# ==================================================
# 5. 创建 Agent
# ==================================================

agent_cfg = habitat_sim.agent.AgentConfiguration()

agent_cfg.sensor_specifications = [rgb, depth]

config = habitat_sim.Configuration(
    sim_cfg,
    [agent_cfg]
)

sim = habitat_sim.Simulator(config)


# ==================================================
# 6. 加载 NavMesh
# ==================================================

if not sim.pathfinder.load_nav_mesh(str(NAVMESH)):
    sim.close()
    raise RuntimeError("Failed to load NavMesh")

print("NavMesh loaded:", sim.pathfinder.is_loaded)


# ==================================================
# 7. 固定机器人位置
# ==================================================

# 使用已经验证过的地板中心
floor_center = mn.Vector3(0.0, 0.0, 0.0)

floor_point = sim.pathfinder.snap_point(floor_center)

print("Floor point:", floor_point)
print("Floor island:", sim.pathfinder.get_island(floor_point))

# Agent 在地板上，RGB 和 Depth 相机高出 1.3m
agent_position = np.array(
    [floor_point.x, floor_point.y, floor_point.z],
    dtype=np.float32
)

agent = sim.get_agent(0)


# ==================================================
# 8. 拍摄四个方向
# ==================================================

directions = {
    "front": 0,
    "right": 90,
    "back": 180,
    "left": 270,
}

try:
    for name, angle_deg in directions.items():

        angle = math.radians(angle_deg)

        state = habitat_sim.AgentState()

        state.position = agent_position

        # 绕世界 Y 轴旋转
        state.rotation = np.quaternion(
            math.cos(angle / 2),
            0,
            math.sin(angle / 2),
            0
        )

        agent.set_state(state)

        observations = sim.get_sensor_observations()

        rgb_image = observations["rgb"][:, :, :3]

        depth_image = observations["depth"]

        image_path = OUTPUT / f"{name}.png"

        Image.fromarray(rgb_image).save(image_path)

        # 检查图像中心的深度
        h, w = depth_image.shape

        center_depth = float(
            depth_image[h // 2, w // 2]
        )

        print(f"\nDirection: {name}")
        print(f"Camera yaw: {angle_deg} degrees")
        print(f"Center depth: {center_depth:.3f} m")
        print(f"Image saved: {image_path}")

finally:
    sim.close()

print("\n========== FINISHED ==========")
