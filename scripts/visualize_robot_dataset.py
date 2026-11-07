from pathlib import Path
import csv
import numpy as np

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(
    "/home/ymq/Habitat/data/my_lab/robot_path_dataset"
)

# =====================================================
# 字体
# =====================================================

def get_font(size, bold=False):

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)

    return ImageFont.load_default()


FONT = get_font(24)
FONT_SMALL = get_font(19)
FONT_TITLE = get_font(34, True)


# =====================================================
# 读取 GT
# =====================================================

with open(ROOT / "trajectory_gt.csv") as f:
    rows = list(csv.DictReader(f))

positions = np.array([
    [
        float(r["agent_x"]),
        float(r["agent_y"]),
        float(r["agent_z"])
    ]
    for r in rows
])

x = positions[:, 0]
z = positions[:, 2]


# =====================================================
# 第一张：轨迹俯视图
# =====================================================

W, H = 1600, 1100

img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

draw.text(
    (W // 2, 35),
    "Robot Ground Truth Trajectory",
    fill="black",
    font=FONT_TITLE,
    anchor="mt"
)

# 绘图范围
xmin, xmax = -4.3, 4.3
zmin, zmax = -3.3, 3.3

left, top = 150, 130
right, bottom = 1450, 950

scale = min(
    (right-left)/(xmax-xmin),
    (bottom-top)/(zmax-zmin)
)

def transform(px, pz):

    sx = left + (px-xmin)*scale
    sy = top + (zmax-pz)*scale

    return int(sx), int(sy)


# 网格
for gx in range(-4, 5):

    p1 = transform(gx, -3)
    p2 = transform(gx, 3)

    draw.line([p1, p2], fill="#DDDDDD", width=1)

    draw.text(
        (p1[0], bottom+15),
        str(gx),
        fill="black",
        font=FONT_SMALL,
        anchor="mt"
    )

for gz in range(-3, 4):

    p1 = transform(-4, gz)
    p2 = transform(4, gz)

    draw.line([p1, p2], fill="#DDDDDD", width=1)

    draw.text(
        (left-25, p1[1]),
        str(gz),
        fill="black",
        font=FONT_SMALL,
        anchor="rm"
    )


# 房间边界
room = [
    transform(-4, -3),
    transform(4, -3),
    transform(4, 3),
    transform(-4, 3)
]

draw.line(
    room + [room[0]],
    fill="black",
    width=5
)

# 三个 Box
boxes = [
    (-1.5, 0.0, "Box_01"),
    (0.0, -1.0, "Box_02"),
    (1.5, 0.0, "Box_03")
]

for bx, bz, name in boxes:

    p1 = transform(bx-0.25, bz+0.25)
    p2 = transform(bx+0.25, bz-0.25)

    draw.rectangle(
        [p1, p2],
        outline="#CC3333",
        fill="#F5CACA",
        width=4
    )

    center = transform(bx, bz)

    draw.text(
        (center[0], center[1]-35),
        name,
        fill="#AA2222",
        font=FONT_SMALL,
        anchor="mb"
    )


# 机器人轨迹
path = [
    transform(px, pz)
    for px, pz in zip(x, z)
]

draw.line(
    path,
    fill="#1565C0",
    width=6
)

# 每隔一定帧绘制方向箭头
for i in range(0, len(path)-1, 15):

    x1, y1 = path[i]
    x2, y2 = path[i+1]

    draw.ellipse(
        (x1-5, y1-5, x1+5, y1+5),
        fill="#1565C0"
    )


# 起点
start = path[0]

draw.ellipse(
    (
        start[0]-12,
        start[1]-12,
        start[0]+12,
        start[1]+12
    ),
    fill="#20A050",
    outline="black",
    width=2
)

draw.text(
    (start[0]-10, start[1]-35),
    "START",
    fill="#16803D",
    font=FONT
)

# 终点
goal = path[-1]

draw.ellipse(
    (
        goal[0]-12,
        goal[1]-12,
        goal[0]+12,
        goal[1]+12
    ),
    fill="#E53935",
    outline="black",
    width=2
)

draw.text(
    (goal[0]+15, goal[1]-35),
    "GOAL",
    fill="#C62828",
    font=FONT
)


# 坐标轴标签
draw.text(
    (W//2, 1010),
    "X (meters)",
    fill="black",
    font=FONT
)

draw.text(
    (45, H//2),
    "Z (meters)",
    fill="black",
    font=FONT,
    anchor="mm"
)

# 轨迹信息
draw.text(
    (180, 1030),
    "Frames: 137 | Path length: 6.615 m | Speed: 0.5 m/s",
    fill="black",
    font=FONT_SMALL
)

output = ROOT / "trajectory_topdown.png"

img.save(output)

print("Saved:", output)


# =====================================================
# 第二张：RGB / Depth 对比图
# =====================================================

sample_indices = [0, 34, 68, 102, 136]

thumb_w = 320
thumb_h = 240

margin = 25
label_h = 35

canvas_w = margin + 5*(thumb_w+margin)
canvas_h = 2*(thumb_h+label_h) + 3*margin

sheet = Image.new(
    "RGB",
    (canvas_w, canvas_h),
    "white"
)

d = ImageDraw.Draw(sheet)

d.text(
    (canvas_w//2, 10),
    "RGB and Depth Samples",
    fill="black",
    font=FONT_TITLE,
    anchor="mt"
)

for col, idx in enumerate(sample_indices):

    filename = f"{idx:06d}.png"

    rgb = Image.open(
        ROOT / "rgb" / filename
    ).convert("RGB")

    depth = Image.open(
        ROOT / "depth_preview" / filename
    ).convert("RGB")

    rgb = rgb.resize((thumb_w, thumb_h))
    depth = depth.resize((thumb_w, thumb_h))

    px = margin + col*(thumb_w+margin)

    py1 = 80

    sheet.paste(rgb, (px, py1))

    d.text(
        (px, py1+thumb_h+5),
        f"RGB Frame {idx}",
        fill="black",
        font=FONT_SMALL
    )

    py2 = py1 + thumb_h + label_h + 15

    sheet.paste(depth, (px, py2))

    d.text(
        (px, py2+thumb_h+5),
        f"Depth Frame {idx}",
        fill="black",
        font=FONT_SMALL
    )

output = ROOT / "rgb_depth_samples.png"

sheet.save(output)

print("Saved:", output)

print("\nVisualization completed!")
