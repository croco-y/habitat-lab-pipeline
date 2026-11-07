# Habitat Lab Simulation Pipeline

基于 [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) 的自定义实验室场景：建房间、摆家具、生成 NavMesh、采集 RGB-D 轨迹，用于视觉 SLAM 等算法验证。

## 环境要求

- Ubuntu + conda
- 环境名示例：`habitat`（已安装 `habitat-sim`、`numpy`、`Pillow`）
- GPU（推荐）

```bash
conda activate habitat
python -c "import habitat_sim; print('habitat_sim OK')"
```

> 本仓库 **不包含** `habitat-sim` 源码。请按官方文档单独安装：  
> https://github.com/facebookresearch/habitat-sim

---

## 目录结构

```text
Habitat/
├── README.md
├── check_lab_camera.py              # 空房间四面相机调试截图
├── test_lab_navmesh_stage.py        # 仅房间 NavMesh
├── scripts/
│   ├── build_realistic_lab_room.py  # 给房间贴木地板/墙面材质
│   ├── prepare_kenney_furniture.py  # 整理 Kenney 家具 → object_config
│   ├── kenney_glb_utils.py          # 家具归一化工具
│   ├── spawn_furniture_and_navmesh.py
│   ├── test_add_box.py
│   ├── test_lab_navmesh_with_boxes.py
│   ├── test_lab_pathfinding.py
│   ├── run_path_capture.py          # 沿路径采集 RGB-D + GT
│   ├── check_robot_path_dataset.py
│   └── visualize_robot_dataset.py
└── data/
    ├── my_lab/                      # 自定义实验室
    │   ├── room_base_8x6.glb
    │   ├── room_base_8x6.stage_config.json
    │   ├── lab.scene_dataset_config.json
    │   ├── scenes/
    │   │   ├── lab_layout_01.scene_instance.json      # 旧：3 个箱子
    │   │   └── lab_layout_furniture.scene_instance.json
    │   ├── *.navmesh
    │   └── textures/                # CC0 贴图（ambientCG）
    └── assets/
        ├── objects/box.*            # 测试立方体
        └── furniture/kenney/        # 归一化后的家具 + object_config
```

---

## 完整流程（推荐顺序）

### 0. 准备贴图（若 `data/my_lab/textures/` 为空）

从 [ambientCG](https://ambientcg.com/) 下载并解压到：

- `data/my_lab/textures/wood/` ← WoodFloor051
- `data/my_lab/textures/plaster/` ← Plaster001
- `data/my_lab/textures/concrete/` ← Concrete034

### 1. 生成带材质的房间

```bash
python scripts/build_realistic_lab_room.py
```

输出：`data/my_lab/room_base_8x6.glb`（木地板 + 墙面 + 天花板）

### 2. 整理家具模型

原始 Kenney 家具放在 `data/assets/kenney_furniture/`（可选大包，可 gitignore）。  
处理后写入：

```bash
python scripts/prepare_kenney_furniture.py
```

作用：

1. 统一坐标原点（底面中心）
2. 按类别缩放到接近真实尺寸
3. 生成 `*.object_config.json`（碰撞/质量等）
4. 写出 `data/assets/furniture/kenney/catalog.json`

### 3. 摆放家具 + 重建 NavMesh

```bash
python scripts/spawn_furniture_and_navmesh.py
```

作用：

1. 打开空房间
2. 按 `DEFAULT_LAYOUT` 动态 `add_object`
3. 物体设为 `STATIC`
4. `include_static_objects=True` 重算导航网格
5. 验证家具区域不可走、仍可绕行
6. 写出场景：`scenes/lab_layout_furniture.scene_instance.json`
7. 写出 NavMesh：`room_base_8x6_with_furniture.navmesh`

**改家具布局**：编辑该 `scene_instance.json`，或改脚本里的 `DEFAULT_LAYOUT` 后重跑。  
改完布局后务必重算 NavMesh。

### 4. 交互查看场景

```bash
cd habitat-sim   # 或你的 habitat_sim 安装路径中的 examples
python examples/viewer.py \
  --dataset /home/ymq/Habitat/data/my_lab/lab.scene_dataset_config.json \
  --scene /home/ymq/Habitat/data/my_lab/scenes/lab_layout_furniture.scene_instance.json
```

快捷键（常见）：

- `n`：显示/隐藏 NavMesh
- `WASD` / 鼠标：移动视角

### 5. 路径规划测试

```bash
python scripts/test_lab_pathfinding.py
```

使用 NavMesh 上的 `find_path`（底层 Recast/Detour A*）。

### 6. 采集 RGB-D + 位姿真值（给 SLAM）

```bash
python scripts/run_path_capture.py
```

输出目录：`data/my_lab/robot_path_dataset/`

| 内容 | 路径 |
|------|------|
| RGB | `rgb/*.png` |
| 深度（米） | `depth/*.npy` |
| 深度预览 | `depth_preview/*.png` |
| 位姿 GT | `trajectory_gt.csv`（用 **camera_*** 列） |
| 内参 | `camera_intrinsics.json` |
| 元数据 | `metadata.json` |

用该数据跑 SLAM，再与 `trajectory_gt.csv` 对比 ATE/RPE。

---

## 关键配置文件说明

| 文件 | 作用 |
|------|------|
| `*.stage_config.json` | 房间（stage）怎么加载：GLB、轴向、缩放 |
| `*.object_config.json` | 单个物体：渲染/碰撞模型、质量、摩擦 |
| `lab.scene_dataset_config.json` | 数据集索引：有哪些 stage / object / scene |
| `*.scene_instance.json` | 某一局布局：用哪个房间、物体放在哪 |
| `*.navmesh` | 可走区域网格 |

关系：

```text
dataset_config  ──► 登记资源
scene_instance  ──► 点名用哪些资源、放哪
stage/object config ──► 资源如何加载
navmesh         ──► 机器人怎么走
```

---

## 导航原理（简要）

1. 根据房间 + **STATIC** 家具碰撞体生成 **NavMesh**
2. 指定起点 / 终点
3. Habitat `pathfinder.find_path` 在导航多边形上求最短路（Detour）
4. 本场景地板近似水平（`Y≈0`），路径为平面绕障，不是爬坡/飞越

---

## 上传 GitHub 建议

### 建议提交

- `README.md`、`.gitignore`
- `scripts/`、根目录调试脚本
- `data/my_lab/` 下的配置、`room_base_8x6.glb`、`scenes/`、`*.navmesh`
- `data/assets/furniture/kenney/`（处理后的家具）
- `data/assets/objects/`（测试盒子）
- `data/my_lab/textures/` 下的贴图目录（不含 zip）

### 建议不要提交（已在 `.gitignore`）

- `habitat-sim/` 上游仓库
- `data/versioned_data/` 官方测试场景
- `data/assets/kenney_furniture/` 原始大包
- `data/my_lab/robot_path_dataset/` 采集结果（可本地重跑生成）
- `__pycache__/`、`*.blend1`

### 初始化示例

```bash
cd /home/ymq/Habitat
git init
git add README.md .gitignore scripts check_lab_camera.py test_lab_navmesh_stage.py
git add data/my_lab data/assets/furniture data/assets/objects
git status   # 确认没有巨大目录
git commit -m "Add Habitat lab room, furniture pipeline, and RGB-D capture scripts"
# 在 GitHub 建好空仓库后：
git remote add origin git@github.com:<USER>/<REPO>.git
git branch -M main
git push -u origin main
```

若单文件（如 GLB/贴图）超过 100MB，请用 [Git LFS](https://git-lfs.com/)。

---

## 常见命令速查

```bash
conda activate habitat

# 房间贴图
python scripts/build_realistic_lab_room.py

# 家具入库
python scripts/prepare_kenney_furniture.py

# 摆家具 + NavMesh
python scripts/spawn_furniture_and_navmesh.py

# 采集 SLAM 数据
python scripts/run_path_capture.py
```

---

## License 备注

- Habitat-Sim：按 Meta 官方许可证
- Kenney Furniture：见 `data/assets/kenney_furniture/License.txt`（通常 CC0）
- ambientCG 贴图：按 ambientCG 许可（常见为 CC0）
