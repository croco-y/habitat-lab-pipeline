from pathlib import Path
import math
import csv
import json

import numpy as np
import quaternion
import magnum as mn

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from PIL import Image


# ============================================================
# 1. 路径配置
# ============================================================

ROOT = Path("/home/ymq/Habitat/data/my_lab")

DATASET_CONFIG = ROOT / "lab.scene_dataset_config.json"

SCENE_INSTANCE = ROOT / "scenes/lab_layout_01.scene_instance.json"

NAVMESH = ROOT / "room_base_8x6_with_boxes.navmesh"

OUTPUT = ROOT / "robot_path_dataset"

RGB_DIR = OUTPUT / "rgb"

DEPTH_DIR = OUTPUT / "depth"

DEPTH_PREVIEW_DIR = OUTPUT / "depth_preview"

for directory in [RGB_DIR, DEPTH_DIR, DEPTH_PREVIEW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 采集参数
# ============================================================

WIDTH = 640
HEIGHT = 480

HFOV_DEG = 90.0

# Agent 基准位置在地板 NavMesh 上
# 相机相对 Agent 的高度
CAMERA_HEIGHT = 1.3

# 路径插值间隔
PATH_STEP = 0.05

# 设定的名义运动速度
SPEED = 0.5

# 深度预览图的显示范围
MAX_DEPTH = 8.0


# ============================================================
# 3. 检查文件
# ============================================================

for p in [DATASET_CONFIG, SCENE_INSTANCE, NAVMESH]:

    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

print("Dataset:", DATASET_CONFIG)
print("Scene:", SCENE_INSTANCE)
print("NavMesh:", NAVMESH)


# ============================================================
# 4. 配置 RGB 相机
# ============================================================

rgb_spec = habitat_sim.CameraSensorSpec()

rgb_spec.uuid = "rgb"

rgb_spec.sensor_type = habitat_sim.SensorType.COLOR

rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

rgb_spec.resolution = [HEIGHT, WIDTH]

rgb_spec.position = [0.0, CAMERA_HEIGHT, 0.0]

rgb_spec.orientation = [0.0, 0.0, 0.0]

rgb_spec.hfov = mn.Deg(HFOV_DEG)


# ============================================================
# 5. 配置 Depth 相机
# ============================================================

depth_spec = habitat_sim.CameraSensorSpec()

depth_spec.uuid = "depth"

depth_spec.sensor_type = habitat_sim.SensorType.DEPTH

depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

depth_spec.resolution = [HEIGHT, WIDTH]

depth_spec.position = [0.0, CAMERA_HEIGHT, 0.0]

depth_spec.orientation = [0.0, 0.0, 0.0]

depth_spec.channels = 1

depth_spec.hfov = mn.Deg(HFOV_DEG)


# ============================================================
# 6. 初始化 Habitat-Sim
# ============================================================

print("\n========== INITIALIZE SIMULATOR ==========")

sim_cfg = habitat_sim.SimulatorConfiguration()

sim_cfg.scene_dataset_config_file = str(DATASET_CONFIG)

sim_cfg.scene_id = str(SCENE_INSTANCE)

sim_cfg.enable_physics = True

sim_cfg.override_scene_light_defaults = True

sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY


agent_cfg = habitat_sim.agent.AgentConfiguration()

agent_cfg.sensor_specifications = [
    rgb_spec,
    depth_spec
]

cfg = habitat_sim.Configuration(
    sim_cfg,
    [agent_cfg]
)

sim = habitat_sim.Simulator(cfg)


try:

    # ========================================================
    # 7. 加载 NavMesh
    # ========================================================

    print("\n========== LOAD NAVMESH ==========")

    if not sim.pathfinder.load_nav_mesh(str(NAVMESH)):
        raise RuntimeError("Failed to load NavMesh!")

    print("NavMesh loaded:", sim.pathfinder.is_loaded)

    print("NavMesh islands:", sim.pathfinder.num_islands)


    # ========================================================
    # 8. 计算路径
    # ========================================================

    print("\n========== COMPUTE PATH ==========")

    start = mn.Vector3(-3.0, 0.2, 0.0)

    goal = mn.Vector3(3.0, 0.2, 0.0)

    path = habitat_sim.ShortestPath()

    path.requested_start = start

    path.requested_end = goal

    found = sim.pathfinder.find_path(path)

    if not found:
        raise RuntimeError("Path planning failed!")

    print("Path found:", found)

    print("Geodesic distance:", path.geodesic_distance)

    path_points = [
        np.array([p[0], p[1], p[2]], dtype=np.float32)
        for p in path.points
    ]


    # ========================================================
    # 9. 对路径进行插值
    # ========================================================

    print("\n========== INTERPOLATE PATH ==========")

    samples = [path_points[0]]

    distances = [0.0]

    total_distance = 0.0

    for i in range(len(path_points) - 1):

        a = path_points[i]

        b = path_points[i + 1]

        segment_length = float(np.linalg.norm(b - a))

        if segment_length < 1e-8:
            continue

        n = max(1, math.ceil(segment_length / PATH_STEP))

        for j in range(1, n + 1):

            t = j / n

            p = a + (b - a) * t

            samples.append(p)

            distances.append(
                total_distance + segment_length * t
            )

        total_distance += segment_length

    print("Original path points:", len(path_points))

    print("Interpolated frames:", len(samples))

    print("Total distance:", total_distance)

    print("Nominal speed:", SPEED, "m/s")


    # ========================================================
    # 10. 配置 Agent
    # ========================================================

    print("\n========== CONFIGURE AGENT ==========")

    agent = sim.get_agent(0)


    # ========================================================
    # 11. 记录 GT 位姿
    # ========================================================

    csv_path = OUTPUT / "trajectory_gt.csv"

    rows = []

    print("\n========== START PATH CAPTURE ==========")

    for i, position in enumerate(samples):

        # ----------------------------------------------------
        # 11.1 计算当前运动方向
        # ----------------------------------------------------

        if i < len(samples) - 1:

            direction = samples[i + 1] - position

        else:

            direction = position - samples[i - 1]

        dx = float(direction[0])

        dz = float(direction[2])

        yaw = math.atan2(-dx, -dz)

        rotation = quat_from_angle_axis(
            yaw,
            np.array([0.0, 1.0, 0.0])
        )


        # ----------------------------------------------------
        # 11.2 设置 Agent 位姿
        # ----------------------------------------------------

        state = habitat_sim.AgentState()

        state.position = position.astype(np.float32)

        state.rotation = rotation

        agent.set_state(state)


        # ----------------------------------------------------
        # 11.3 获取 RGB-D
        # ----------------------------------------------------

        obs = sim.get_sensor_observations()

        rgb = np.asarray(obs["rgb"])

        depth = np.asarray(
            obs["depth"],
            dtype=np.float32
        )

        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]

        rgb = rgb.astype(np.uint8)

        # 保存 RGB
        rgb_path = RGB_DIR / f"{i:06d}.png"

        Image.fromarray(rgb).save(rgb_path)

        # 保存原始深度
        depth_path = DEPTH_DIR / f"{i:06d}.npy"

        np.save(depth_path, depth)

        # 保存深度预览
        depth_safe = np.nan_to_num(
            depth,
            nan=MAX_DEPTH,
            posinf=MAX_DEPTH,
            neginf=0.0
        )

        depth_vis = (
            np.clip(depth_safe, 0.0, MAX_DEPTH)
            / MAX_DEPTH
            * 255.0
        ).astype(np.uint8)

        Image.fromarray(depth_vis).save(
            DEPTH_PREVIEW_DIR / f"{i:06d}.png"
        )


        # ----------------------------------------------------
        # 11.4 获取真实 Agent / Camera 位姿
        # ----------------------------------------------------

        actual_state = agent.get_state()

        camera_state = actual_state.sensor_states["rgb"]

        agent_pos = np.asarray(
            actual_state.position,
            dtype=np.float64
        )

        camera_pos = np.asarray(
            camera_state.position,
            dtype=np.float64
        )

        agent_q = actual_state.rotation

        camera_q = camera_state.rotation

        agent_quat = [
            float(agent_q.w),
            float(agent_q.x),
            float(agent_q.y),
            float(agent_q.z)
        ]

        camera_quat = [
            float(camera_q.w),
            float(camera_q.x),
            float(camera_q.y),
            float(camera_q.z)
        ]

        timestamp = float(distances[i] / SPEED)

        rows.append({

            "frame_idx": i,

            "timestamp_s": timestamp,

            "path_distance_m": float(distances[i]),

            "agent_x": float(agent_pos[0]),
            "agent_y": float(agent_pos[1]),
            "agent_z": float(agent_pos[2]),

            "agent_qw": agent_quat[0],
            "agent_qx": agent_quat[1],
            "agent_qy": agent_quat[2],
            "agent_qz": agent_quat[3],

            "camera_x": float(camera_pos[0]),
            "camera_y": float(camera_pos[1]),
            "camera_z": float(camera_pos[2]),

            "camera_qw": camera_quat[0],
            "camera_qx": camera_quat[1],
            "camera_qy": camera_quat[2],
            "camera_qz": camera_quat[3]

        })

        if i % 10 == 0:

            print(
                f"Frame {i:04d}/{len(samples)} | "
                f"Position: {agent_pos} | "
                f"Time: {timestamp:.2f}s"
            )


    # ========================================================
    # 12. 保存轨迹 CSV
    # ========================================================

    print("\n========== SAVE TRAJECTORY ==========")

    with open(csv_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()

        writer.writerows(rows)

    print("GT trajectory:", csv_path)


    # ========================================================
    # 13. 保存相机内参
    # ========================================================

    fx = WIDTH / (
        2.0 * math.tan(math.radians(HFOV_DEG) / 2.0)
    )

    fy = fx

    cx = (WIDTH - 1) / 2.0

    cy = (HEIGHT - 1) / 2.0

    K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]

    intrinsics = {

        "width": WIDTH,

        "height": HEIGHT,

        "model": "PINHOLE",

        "hfov_deg": HFOV_DEG,

        "fx": fx,

        "fy": fy,

        "cx": cx,

        "cy": cy,

        "K": K

    }

    with open(OUTPUT / "camera_intrinsics.json", "w") as f:

        json.dump(intrinsics, f, indent=4)


    # ========================================================
    # 14. 保存实验信息
    # ========================================================

    metadata = {

        "scene": str(SCENE_INSTANCE),

        "navmesh": str(NAVMESH),

        "start": [float(start[0]), float(start[1]), float(start[2])],

        "goal": [float(goal[0]), float(goal[1]), float(goal[2])],

        "geodesic_distance_m": float(path.geodesic_distance),

        "nominal_speed_mps": SPEED,

        "path_step_m": PATH_STEP,

        "frame_count": len(samples),

        "nominal_duration_s": float(total_distance / SPEED),

        "rgb_resolution": [WIDTH, HEIGHT],

        "depth_resolution": [WIDTH, HEIGHT]

    }

    with open(OUTPUT / "metadata.json", "w") as f:

        json.dump(metadata, f, indent=4)

    print("\n========== CAPTURE COMPLETE ==========")

    print("Output directory:", OUTPUT)

    print("Frames:", len(samples))

    print("RGB:", RGB_DIR)

    print("Depth:", DEPTH_DIR)

    print("Depth preview:", DEPTH_PREVIEW_DIR)

    print("GT:", csv_path)

finally:

    sim.close()
