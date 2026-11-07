#!/usr/bin/env python3
"""Rebuild room_base_8x6.glb with PBR textures (wood floor, plaster walls, concrete ceiling).

Keeps the same 8x6x3 room proportions and front-door opening so existing layouts
remain roughly compatible.

Usage:
    /home/ymq/anaconda3/envs/habitat/bin/python \\
        /home/ymq/Habitat/scripts/build_realistic_lab_room.py
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/ymq/Habitat")
LAB = ROOT / "data/my_lab"
TEX = LAB / "textures"
OUT_GLB = LAB / "room_base_8x6.glb"
BACKUP_GLB = LAB / "room_base_8x6_untextured.glb"

# Room inner volume (meters), matching previous lab proportions.
ROOM_W = 8.0  # X
ROOM_D = 6.0  # Z
ROOM_H = 3.0  # Y
WALL_T = 0.12
FLOOR_T = 0.10
CEIL_T = 0.10
DOOR_W = 1.0
DOOR_H = 2.1


def _box(cx, cy, cz, sx, sy, sz, uv_scale=1.0):
    """Axis-aligned box centered at c with size s. Returns pos/normal/uv/indices."""
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    # 6 faces, 4 verts each
    faces = [
        # +Y top
        (
            [(-hx, hy, -hz), (hx, hy, -hz), (hx, hy, hz), (-hx, hy, hz)],
            (0, 1, 0),
            sx,
            sz,
        ),
        # -Y bottom
        (
            [(-hx, -hy, hz), (hx, -hy, hz), (hx, -hy, -hz), (-hx, -hy, -hz)],
            (0, -1, 0),
            sx,
            sz,
        ),
        # +Z
        (
            [(-hx, -hy, hz), (-hx, hy, hz), (hx, hy, hz), (hx, -hy, hz)],
            (0, 0, 1),
            sx,
            sy,
        ),
        # -Z
        (
            [(hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz), (-hx, -hy, -hz)],
            (0, 0, -1),
            sx,
            sy,
        ),
        # +X
        (
            [(hx, -hy, hz), (hx, hy, hz), (hx, hy, -hz), (hx, -hy, -hz)],
            (1, 0, 0),
            sz,
            sy,
        ),
        # -X
        (
            [(-hx, -hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (-hx, -hy, hz)],
            (-1, 0, 0),
            sz,
            sy,
        ),
    ]

    positions, normals, uvs, indices = [], [], [], []
    base = 0
    for corners, nrm, u_size, v_size in faces:
        us = max(u_size * uv_scale, 1e-6)
        vs = max(v_size * uv_scale, 1e-6)
        face_uv = [(0, 0), (us, 0), (us, vs), (0, vs)]
        for (x, y, z), (u, v) in zip(corners, face_uv):
            positions.append((cx + x, cy + y, cz + z))
            normals.append(nrm)
            uvs.append((u, v))
        indices.extend(
            [base, base + 1, base + 2, base, base + 2, base + 3]
        )
        base += 4

    return (
        np.asarray(positions, np.float32),
        np.asarray(normals, np.float32),
        np.asarray(uvs, np.float32),
        np.asarray(indices, np.uint32),
    )


def _merge(meshes):
    pos, nrm, uv, idx = [], [], [], []
    offset = 0
    for p, n, u, i in meshes:
        pos.append(p)
        nrm.append(n)
        uv.append(u)
        idx.append(i + offset)
        offset += len(p)
    return (
        np.concatenate(pos),
        np.concatenate(nrm),
        np.concatenate(uv),
        np.concatenate(idx),
    )


def build_room_parts():
    half_w = ROOM_W * 0.5
    half_d = ROOM_D * 0.5

    floor = _box(
        0.0,
        -FLOOR_T * 0.5,
        0.0,
        ROOM_W + 2 * WALL_T,
        FLOOR_T,
        ROOM_D + 2 * WALL_T,
        uv_scale=0.5,
    )
    ceiling = _box(
        0.0,
        ROOM_H + CEIL_T * 0.5,
        0.0,
        ROOM_W + 2 * WALL_T,
        CEIL_T,
        ROOM_D + 2 * WALL_T,
        uv_scale=0.35,
    )

    # Side walls (full)
    wall_left = _box(
        -(half_w + WALL_T * 0.5),
        ROOM_H * 0.5,
        0.0,
        WALL_T,
        ROOM_H,
        ROOM_D + 2 * WALL_T,
        uv_scale=0.4,
    )
    wall_right = _box(
        half_w + WALL_T * 0.5,
        ROOM_H * 0.5,
        0.0,
        WALL_T,
        ROOM_H,
        ROOM_D + 2 * WALL_T,
        uv_scale=0.4,
    )
    # Back wall (full, -Z)
    wall_back = _box(
        0.0,
        ROOM_H * 0.5,
        -(half_d + WALL_T * 0.5),
        ROOM_W,
        ROOM_H,
        WALL_T,
        uv_scale=0.4,
    )

    # Front wall with door opening (+Z)
    side = (ROOM_W - DOOR_W) * 0.5
    wall_front_l = _box(
        -(DOOR_W * 0.5 + side * 0.5),
        ROOM_H * 0.5,
        half_d + WALL_T * 0.5,
        side,
        ROOM_H,
        WALL_T,
        uv_scale=0.4,
    )
    wall_front_r = _box(
        DOOR_W * 0.5 + side * 0.5,
        ROOM_H * 0.5,
        half_d + WALL_T * 0.5,
        side,
        ROOM_H,
        WALL_T,
        uv_scale=0.4,
    )
    lintel_h = ROOM_H - DOOR_H
    wall_front_top = _box(
        0.0,
        DOOR_H + lintel_h * 0.5,
        half_d + WALL_T * 0.5,
        DOOR_W,
        lintel_h,
        WALL_T,
        uv_scale=0.4,
    )

    # Baseboard strips for a bit more realism
    board_h = 0.08
    board_t = 0.02
    boards = [
        _box(0.0, board_h * 0.5, -(half_d - board_t * 0.5), ROOM_W - 0.02, board_h, board_t, 0.8),
        _box(-(half_w - board_t * 0.5), board_h * 0.5, 0.0, board_t, board_h, ROOM_D - 0.02, 0.8),
        _box(half_w - board_t * 0.5, board_h * 0.5, 0.0, board_t, board_h, ROOM_D - 0.02, 0.8),
        # front left/right of door
        _box(
            -(DOOR_W * 0.5 + side * 0.5),
            board_h * 0.5,
            half_d - board_t * 0.5,
            side - 0.02,
            board_h,
            board_t,
            0.8,
        ),
        _box(
            DOOR_W * 0.5 + side * 0.5,
            board_h * 0.5,
            half_d - board_t * 0.5,
            side - 0.02,
            board_h,
            board_t,
            0.8,
        ),
    ]

    return {
        "Floor": (floor, "wood"),
        "Ceiling": (ceiling, "concrete"),
        "Walls": (
            _merge(
                [
                    wall_left,
                    wall_right,
                    wall_back,
                    wall_front_l,
                    wall_front_r,
                    wall_front_top,
                ]
            ),
            "plaster",
        ),
        "Baseboard": (_merge(boards), "wood_dark"),
    }


def _pad4(b: bytes, pad=b"\x00") -> bytes:
    return b + pad * ((4 - (len(b) % 4)) % 4)


def _jpg_bytes(path: Path, max_size=1024, tint=None) -> bytes:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_size:
        img = img.resize((max_size, max_size), Image.Resampling.LANCZOS)
    if tint is not None:
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr * np.array(tint, dtype=np.float32), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def write_textured_glb(out_path: Path) -> None:
    parts = build_room_parts()

    wood_color = TEX / "wood/WoodFloor051_1K-JPG_Color.jpg"
    wood_rough = TEX / "wood/WoodFloor051_1K-JPG_Roughness.jpg"
    plaster_color = TEX / "plaster/Plaster001_1K-JPG_Color.jpg"
    plaster_rough = TEX / "plaster/Plaster001_1K-JPG_Roughness.jpg"
    concrete_color = TEX / "concrete/Concrete034_1K-JPG_Color.jpg"
    concrete_rough = TEX / "concrete/Concrete034_1K-JPG_Roughness.jpg"

    for p in (
        wood_color,
        wood_rough,
        plaster_color,
        plaster_rough,
        concrete_color,
        concrete_rough,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    images = {
        "wood_c": _jpg_bytes(wood_color),
        "wood_r": _jpg_bytes(wood_rough),
        "plaster_c": _jpg_bytes(plaster_color, tint=(1.05, 1.03, 1.0)),
        "plaster_r": _jpg_bytes(plaster_rough),
        "concrete_c": _jpg_bytes(concrete_color, tint=(0.95, 0.95, 0.98)),
        "concrete_r": _jpg_bytes(concrete_rough),
        "wood_dark_c": _jpg_bytes(wood_color, tint=(0.45, 0.32, 0.22)),
        "wood_dark_r": _jpg_bytes(wood_rough),
    }

    # Pack binary blob: images first, then geometry accessors.
    blob = bytearray()
    image_views = {}
    for key, data in images.items():
        data = _pad4(data)
        image_views[key] = (len(blob), len(data))
        blob.extend(data)

    def add_array(arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        raw = _pad4(arr.tobytes())
        offset = len(blob)
        blob.extend(raw)
        return offset, len(arr.tobytes()), arr

    accessors = []
    buffer_views = []
    # image bufferViews
    for key, (off, ln) in image_views.items():
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": off,
                "byteLength": ln,
            }
        )

    img_index = {k: i for i, k in enumerate(image_views.keys())}

    meshes = []
    nodes = []
    materials = []

    mat_defs = {
        "wood": ("wood_c", "wood_r", 0.85, [1, 1, 1, 1]),
        "wood_dark": ("wood_dark_c", "wood_dark_r", 0.75, [1, 1, 1, 1]),
        "plaster": ("plaster_c", "plaster_r", 0.9, [1, 1, 1, 1]),
        "concrete": ("concrete_c", "concrete_r", 0.95, [1, 1, 1, 1]),
    }

    mat_index = {}
    for name, (c_key, r_key, rough, base) in mat_defs.items():
        mat_index[name] = len(materials)
        materials.append(
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": base,
                    "baseColorTexture": {"index": None},  # filled after textures list
                    "metallicFactor": 0.0,
                    "roughnessFactor": rough,
                    "metallicRoughnessTexture": {"index": None},
                },
            }
        )

    # glTF textures reference images; metallicRoughness uses G channel for roughness.
    # We'll store grayscale roughness packed into a simple RGB where G=roughness.
    # Simpler approach: skip MR texture and only use baseColor + roughnessFactor.
    textures = []
    gltf_images = []
    for key in image_views.keys():
        if not key.endswith("_c"):
            continue
        bv_i = img_index[key]
        gltf_images.append({"mimeType": "image/jpeg", "bufferView": bv_i})
        textures.append({"source": len(gltf_images) - 1, "sampler": 0})

    # Map material color keys to texture indices
    color_tex = {
        "wood_c": 0,
        "plaster_c": 1,
        "concrete_c": 2,
        "wood_dark_c": 3,
    }
    # Rebuild materials cleanly without MR textures (Habitat handles this well)
    materials = []
    mat_index = {}
    order = ["wood", "wood_dark", "plaster", "concrete"]
    color_for = {
        "wood": "wood_c",
        "wood_dark": "wood_dark_c",
        "plaster": "plaster_c",
        "concrete": "concrete_c",
    }
    rough_for = {
        "wood": 0.82,
        "wood_dark": 0.78,
        "plaster": 0.92,
        "concrete": 0.95,
    }
    # textures list currently only color images in discovery order of image_views keys
    # Rebuild textures explicitly for stable indices.
    gltf_images = []
    textures = []
    tex_of = {}
    for mat_name in order:
        key = color_for[mat_name]
        bv_i = img_index[key]
        tex_of[mat_name] = len(textures)
        gltf_images.append({"mimeType": "image/jpeg", "bufferView": bv_i})
        textures.append({"source": len(gltf_images) - 1, "sampler": 0})
        mat_index[mat_name] = len(materials)
        materials.append(
            {
                "name": mat_name,
                "doubleSided": False,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "baseColorTexture": {"index": tex_of[mat_name]},
                    "metallicFactor": 0.0,
                    "roughnessFactor": rough_for[mat_name],
                },
            }
        )

    for node_name, (mesh, mat_name) in parts.items():
        pos, nrm, uv, idx = mesh
        # accessors
        p_off, p_len, pos = add_array(pos)
        n_off, n_len, nrm = add_array(nrm)
        u_off, u_len, uv = add_array(uv)
        # indices as uint32
        i_off, i_len, idx = add_array(idx.astype(np.uint32))

        bv_base = len(buffer_views)
        for off, ln, stride in (
            (p_off, p_len, 12),
            (n_off, n_len, 12),
            (u_off, u_len, 8),
            (i_off, i_len, None),
        ):
            view = {"buffer": 0, "byteOffset": off, "byteLength": ln}
            if stride:
                view["byteStride"] = stride
            buffer_views.append(view)

        acc_base = len(accessors)
        accessors.extend(
            [
                {
                    "bufferView": bv_base + 0,
                    "componentType": 5126,
                    "count": len(pos),
                    "type": "VEC3",
                    "max": pos.max(axis=0).tolist(),
                    "min": pos.min(axis=0).tolist(),
                },
                {
                    "bufferView": bv_base + 1,
                    "componentType": 5126,
                    "count": len(nrm),
                    "type": "VEC3",
                },
                {
                    "bufferView": bv_base + 2,
                    "componentType": 5126,
                    "count": len(uv),
                    "type": "VEC2",
                },
                {
                    "bufferView": bv_base + 3,
                    "componentType": 5125,
                    "count": len(idx),
                    "type": "SCALAR",
                },
            ]
        )

        mesh_idx = len(meshes)
        meshes.append(
            {
                "name": node_name,
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": acc_base,
                            "NORMAL": acc_base + 1,
                            "TEXCOORD_0": acc_base + 2,
                        },
                        "indices": acc_base + 3,
                        "material": mat_index[mat_name],
                        "mode": 4,
                    }
                ],
            }
        )
        nodes.append({"name": node_name, "mesh": mesh_idx})

    gltf = {
        "asset": {"version": "2.0", "generator": "build_realistic_lab_room.py"},
        "scenes": [{"name": "Scene", "nodes": list(range(len(nodes)))}],
        "scene": 0,
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": textures,
        "images": gltf_images,
        "samplers": [
            {
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497,
            }
        ],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
    }

    json_bytes = _pad4(
        json.dumps(gltf, separators=(",", ":")).encode("utf-8"), pad=b" "
    )
    bin_bytes = _pad4(bytes(blob))
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        f.write(json_bytes)
        f.write(struct.pack("<I4s", len(bin_bytes), b"BIN\x00"))
        f.write(bin_bytes)


def render_preview() -> Path:
    import habitat_sim
    import magnum as mn
    from habitat_sim.utils.settings import built_with_bullet

    stage = LAB / "room_base_8x6.stage_config.json"
    out = LAB / "furniture_nav_test" / "room_textured_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(stage)
    sim_cfg.enable_physics = built_with_bullet
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [480, 640]
    rgb.position = [0.0, 1.4, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        # Also drop a couple furniture pieces if available for context.
        furn = ROOT / "data/assets/furniture/kenney"
        rom = sim.get_rigid_object_manager()
        otm = sim.get_object_template_manager()
        for name, x, z, yaw in (
            ("desk", -2.2, -1.5, 0.0),
            ("loungeSofa", 1.8, -1.6, -30.0),
            ("tableCoffee", 1.5, -0.3, 0.0),
        ):
            cfg = furn / f"{name}.object_config.json"
            if not cfg.is_file():
                continue
            ids = otm.load_configs(str(cfg))
            if not ids:
                continue
            handle = otm.get_template_handle_by_id(ids[0])
            obj = rom.add_object_by_template_handle(handle)
            if obj is None:
                continue
            # rough half-height
            import json as _json

            cat = _json.loads((furn / "catalog.json").read_text())
            half = float(cat["assets"].get(name, {}).get("half_height_m", 0.4))
            obj.translation = mn.Vector3(x, half, z)
            obj.rotation = mn.Quaternion.rotation(
                mn.Rad(np.deg2rad(yaw)), mn.Vector3(0, 1, 0)
            )
            if built_with_bullet:
                obj.motion_type = habitat_sim.physics.MotionType.STATIC

        agent = sim.get_agent(0)
        st = habitat_sim.AgentState()
        st.position = np.array([0.0, 0.0, 2.3], dtype=np.float32)
        agent.set_state(st)
        obs = sim.get_sensor_observations()
        Image.fromarray(np.asarray(obs["rgb"])[:, :, :3]).save(out)
        print("Preview:", out)
    finally:
        sim.close()
    return out


def main() -> None:
    if OUT_GLB.is_file() and not BACKUP_GLB.is_file():
        shutil.copy2(OUT_GLB, BACKUP_GLB)
        print("Backup:", BACKUP_GLB)

    write_textured_glb(OUT_GLB)
    print("Wrote textured room:", OUT_GLB, "size=", OUT_GLB.stat().st_size)
    render_preview()
    print(
        "\nNote: geometry changed slightly; recompute NavMesh if pathfinding looks off:\n"
        "  python /home/ymq/Habitat/scripts/spawn_furniture_and_navmesh.py"
    )


if __name__ == "__main__":
    main()
