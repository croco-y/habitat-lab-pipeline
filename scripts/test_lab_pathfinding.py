from pathlib import Path
import json
import math

import habitat_sim
import magnum as mn


# ============================================================
# 1. 路径
# ============================================================

ROOT = Path("/home/ymq/Habitat/data/my_lab")

DATASET_CONFIG = ROOT / "lab.scene_dataset_config.json"

SCENE_INSTANCE = ROOT / "scenes/lab_layout_01.scene_instance.json"

NAVMESH = ROOT / "room_base_8x6_with_boxes.navmesh"

OUTPUT = ROOT / "pathfinding_result.json"


# ============================================================
# 2. 初始化仿真器
# ============================================================

sim_cfg = habitat_sim.SimulatorConfiguration()

sim_cfg.scene_dataset_config_file = str(DATASET_CONFIG)

sim_cfg.scene_id = str(SCENE_INSTANCE)

sim_cfg.enable_physics = True

agent_cfg = habitat_sim.agent.AgentConfiguration()

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

sim = habitat_sim.Simulator(cfg)


try:

    # ========================================================
    # 3. 加载已经生成的 NavMesh
    # ========================================================

    print("\n========== LOAD NAVMESH ==========")

    success = sim.pathfinder.load_nav_mesh(str(NAVMESH))

    if not success:
        raise RuntimeError("Failed to load NavMesh!")

    print("NavMesh loaded:", sim.pathfinder.is_loaded)

    print("Connected islands:", sim.pathfinder.num_islands)


    # ========================================================
    # 4. 设置起点和终点
    # ========================================================

    start_raw = mn.Vector3(-3.0, 0.2, 0.0)

    goal_raw = mn.Vector3(3.0, 0.2, 0.0)

    print("\n========== START / GOAL ==========")

    print("Start navigable:",
          sim.pathfinder.is_navigable(start_raw))

    print("Goal navigable:",
          sim.pathfinder.is_navigable(goal_raw))

    # 如果不在 NavMesh 上，吸附到最近的可导航位置
    start = sim.pathfinder.snap_point(start_raw)

    goal = sim.pathfinder.snap_point(goal_raw)

    if not sim.pathfinder.is_navigable(start):
        raise RuntimeError("Invalid start point!")

    if not sim.pathfinder.is_navigable(goal):
        raise RuntimeError("Invalid goal point!")

    print("Actual start:", start)

    print("Actual goal:", goal)


    # ========================================================
    # 5. 计算最短路径
    # ========================================================

    print("\n========== PATHFINDING ==========")

    path = habitat_sim.ShortestPath()

    path.requested_start = start

    path.requested_end = goal

    found = sim.pathfinder.find_path(path)

    print("Path found:", found)

    if not found:
        raise RuntimeError("No navigable path between start and goal!")

    print("Geodesic distance:",
          path.geodesic_distance)

    direct_distance = (goal - start).length()

    print("Euclidean distance:",
          direct_distance)

    print("Path points:", len(path.points))

    print("\n========== PATH COORDINATES ==========")

    points = []

    for i, p in enumerate(path.points):

        xyz = [
            float(p[0]),
            float(p[1]),
            float(p[2])
        ]

        points.append(xyz)

        print(f"Point {i}: {xyz}")


    # ========================================================
    # 6. 保存路径
    # ========================================================

    result = {
        "start": [float(v) for v in start],
        "goal": [float(v) for v in goal],
        "path_found": bool(found),
        "geodesic_distance": float(path.geodesic_distance),
        "euclidean_distance": float(direct_distance),
        "path_points": points
    }

    with open(OUTPUT, "w") as f:

        json.dump(result, f, indent=4)

    print("\nSaved:", OUTPUT)

    print("\n========== FINISHED ==========")


finally:

    sim.close()
