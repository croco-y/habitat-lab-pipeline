from pathlib import Path
import json
import csv

import numpy as np
from PIL import Image

ROOT = Path(
    "/home/ymq/Habitat/data/my_lab/robot_path_dataset"
)

RGB_DIR = ROOT / "rgb"
DEPTH_DIR = ROOT / "depth"
PREVIEW_DIR = ROOT / "depth_preview"

# =====================================================
# 1. 检查文件数量
# =====================================================

rgb_files = sorted(RGB_DIR.glob("*.png"))
depth_files = sorted(DEPTH_DIR.glob("*.npy"))
preview_files = sorted(PREVIEW_DIR.glob("*.png"))

print("\n========== FILE COUNTS ==========")

print("RGB:", len(rgb_files))
print("Depth:", len(depth_files))
print("Preview:", len(preview_files))

assert len(rgb_files) == 137
assert len(depth_files) == 137
assert len(preview_files) == 137

# =====================================================
# 2. RGB 图像检查
# =====================================================

print("\n========== RGB INSPECTION ==========")

rgb_black_ratios = []

for p in rgb_files:

    img = np.asarray(Image.open(p).convert("RGB"))

    assert img.shape == (480, 640, 3), (
        p.name, img.shape
    )

    black_ratio = np.mean(np.max(img, axis=2) == 0)

    rgb_black_ratios.append(black_ratio)

print("RGB dimensions: 640x480")

print("Mean black pixel ratio:",
      np.mean(rgb_black_ratios))

print("Maximum black pixel ratio:",
      np.max(rgb_black_ratios))

# =====================================================
# 3. Depth 检查
# =====================================================

print("\n========== DEPTH INSPECTION ==========")

valid_ratios = []
depth_min_values = []
depth_max_values = []

sample_values = []

for p in depth_files:

    depth = np.load(p)

    depth = np.squeeze(depth)

    if depth.shape != (480, 640):
        raise ValueError(
            f"Unexpected depth shape: {p.name}, {depth.shape}"
        )

    valid = np.isfinite(depth) & (depth > 0)

    valid_ratios.append(np.mean(valid))

    if np.any(valid):

        values = depth[valid]

        depth_min_values.append(float(values.min()))

        depth_max_values.append(float(values.max()))

        sample_values.append(values[::100])

all_samples = np.concatenate(sample_values)

print("Depth shape: 480x640")

print("Depth dtype:", depth.dtype)

print("Mean valid ratio:", np.mean(valid_ratios))

print("Minimum valid depth:",
      min(depth_min_values), "m")

print("Maximum valid depth:",
      max(depth_max_values), "m")

print("Depth percentiles:")

for q in [1, 5, 25, 50, 75, 95, 99]:

    print(
        f"P{q}: {np.percentile(all_samples, q):.4f} m"
    )

# =====================================================
# 4. GT 轨迹检查
# =====================================================

print("\n========== GT INSPECTION ==========")

csv_path = ROOT / "trajectory_gt.csv"

with open(csv_path, "r") as f:

    rows = list(csv.DictReader(f))

print("Frames:", len(rows))

def column(name):

    return np.array(
        [float(row[name]) for row in rows]
    )

agent = np.column_stack([
    column("agent_x"),
    column("agent_y"),
    column("agent_z")
])

camera = np.column_stack([
    column("camera_x"),
    column("camera_y"),
    column("camera_z")
])

timestamps = column("timestamp_s")

# 轨迹长度
step_lengths = np.linalg.norm(
    np.diff(agent, axis=0),
    axis=1
)

trajectory_length = np.sum(step_lengths)

print("Start:", agent[0])

print("End:", agent[-1])

print("Trajectory length:", trajectory_length, "m")

print("Minimum step:", step_lengths.min(), "m")

print("Maximum step:", step_lengths.max(), "m")

print("Timestamp start:", timestamps[0])

print("Timestamp end:", timestamps[-1])

print("Timestamp monotonic:",
      np.all(np.diff(timestamps) > 0))

# 相机与 Agent 的距离
camera_offset = np.linalg.norm(
    camera - agent,
    axis=1
)

print("Camera-Agent offset mean:",
      camera_offset.mean(), "m")

print("Camera height mean:",
      camera[:, 1].mean(), "m")

# 四元数检查
for prefix in ["agent", "camera"]:

    quat = np.column_stack([
        column(f"{prefix}_qw"),
        column(f"{prefix}_qx"),
        column(f"{prefix}_qy"),
        column(f"{prefix}_qz")
    ])

    norms = np.linalg.norm(quat, axis=1)

    print(
        prefix,
        "quaternion norm range:",
        norms.min(),
        norms.max()
    )

# =====================================================
# 5. 内参检查
# =====================================================

print("\n========== CAMERA INTRINSICS ==========")

with open(ROOT / "camera_intrinsics.json") as f:

    intrinsics = json.load(f)

print(json.dumps(intrinsics, indent=4))

# =====================================================
# 6. 生成可视化
# =====================================================

print("\n========== VISUALIZATION ==========")

try:

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 顶视轨迹图
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        agent[:, 0],
        agent[:, 2],
        linewidth=2,
        label="Robot trajectory"
    )

    ax.scatter(
        agent[0, 0],
        agent[0, 2],
        marker="o",
        s=100,
        label="Start"
    )

    ax.scatter(
        agent[-1, 0],
        agent[-1, 2],
        marker="*",
        s=150,
        label="Goal"
    )

    # 三个 Box，尺寸假设为 0.5m
    box_centers = [
        (-1.5, 0),
        (0, -1),
        (1.5, 0)
    ]

    from matplotlib.patches import Rectangle

    for x, z in box_centers:

        ax.add_patch(
            Rectangle(
                (x - 0.25, z - 0.25),
                0.5,
                0.5,
                fill=False,
                linewidth=2
            )
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Robot Ground Truth Trajectory")

    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    fig.savefig(
        ROOT / "trajectory_topdown.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # 三维轨迹
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        agent[:, 0],
        agent[:, 2],
        agent[:, 1],
        linewidth=2
    )

    ax.scatter(
        agent[0, 0],
        agent[0, 2],
        agent[0, 1],
        s=80,
        label="Start"
    )

    ax.scatter(
        agent[-1, 0],
        agent[-1, 2],
        agent[-1, 1],
        s=80,
        label="Goal"
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y (m)")

    ax.set_title("3D Ground Truth Trajectory")

    ax.legend()

    fig.savefig(
        ROOT / "trajectory_3d.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved trajectory_topdown.png")
    print("Saved trajectory_3d.png")

except ImportError:

    print("Matplotlib unavailable, skip visualization.")

print("\n========== CHECK COMPLETE ==========")
