"""Utilities for reading/writing Kenney furniture GLBs and normalizing them."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Structural / non-furniture pieces to skip by default.
SKIP_PREFIXES = (
    "wall",
    "floor",
    "doorway",
    "stairs",
    "ceiling",
    "rug",
    "paneling",
)

# Target height (meters) by keyword. Used to scale stylized Kenney assets
# toward real-world proportions for an 8x6 lab room.
# More specific keys must appear before generic ones (e.g. coffee before table).
TARGET_HEIGHT_M = {
    "coffee": 0.40,
    "sideTable": 0.55,
    "bed": 0.55,
    "sofa": 0.85,
    "lounge": 0.85,
    "chair": 0.95,
    "stool": 0.75,
    "desk": 0.75,
    "table": 0.75,
    "bookcase": 1.80,
    "cabinet": 1.00,
    "fridge": 1.80,
    "washer": 0.95,
    "dryer": 0.95,
    "lamp": 1.50,
    "plant": 0.45,
    "potted": 0.90,
    "television": 0.55,
    "tv": 0.55,
    "bathtub": 0.55,
    "toilet": 0.45,
    "sink": 0.85,
    "bench": 0.45,
    "trashcan": 0.55,
    "default": 0.90,
}

DEFAULT_MASS = {
    "bed": 40.0,
    "sofa": 35.0,
    "lounge": 30.0,
    "chair": 8.0,
    "stool": 5.0,
    "desk": 25.0,
    "table": 20.0,
    "bookcase": 30.0,
    "cabinet": 25.0,
    "fridge": 50.0,
    "washer": 45.0,
    "dryer": 40.0,
    "lamp": 4.0,
    "plant": 3.0,
    "default": 10.0,
}


def is_furniture_asset(name: str) -> bool:
    lower = name.lower()
    return not any(lower.startswith(p) for p in SKIP_PREFIXES)


def category_for(name: str) -> str:
    stem = Path(name).stem
    lower = stem.lower()
    # Preserve camelCase keys like sideTable by checking raw stem too.
    for key in TARGET_HEIGHT_M:
        if key == "default":
            continue
        if key.lower() in lower or key in stem:
            return key
    return "default"


def target_height_for(name: str) -> float:
    return TARGET_HEIGHT_M[category_for(name)]


def mass_for(name: str) -> float:
    return DEFAULT_MASS.get(category_for(name), DEFAULT_MASS["default"])


def load_glb(path: Path) -> Tuple[dict, bytes]:
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"Not a GLB file: {path}")
    offset = 12
    gltf = None
    bin_chunk = b""
    while offset < length:
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == b"JSON":
            gltf = json.loads(chunk.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk
    if gltf is None:
        raise ValueError(f"Missing JSON chunk: {path}")
    return gltf, bin_chunk


def save_glb(path: Path, gltf: dict, bin_chunk: bytes) -> None:
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # Pad JSON to 4-byte boundary with spaces.
    json_padding = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_padding
    bin_padding = (4 - (len(bin_chunk) % 4)) % 4
    bin_bytes = bin_chunk + (b"\x00" * bin_padding)

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        f.write(json_bytes)
        f.write(struct.pack("<I4s", len(bin_bytes), b"BIN\x00"))
        f.write(bin_bytes)


def _mat4_from_node(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")

    t = np.eye(4)
    if "translation" in node:
        t[:3, 3] = node["translation"]

    r = np.eye(4)
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        r[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    s = np.eye(4)
    if "scale" in node:
        sx, sy, sz = node["scale"]
        s[0, 0], s[1, 1], s[2, 2] = sx, sy, sz

    return t @ r @ s


def _accessor_array(gltf: dict, bin_chunk: bytes, accessor_idx: int) -> np.ndarray:
    acc = gltf["accessors"][accessor_idx]
    view = gltf["bufferViews"][acc["bufferView"]]
    comp = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[
        acc["componentType"]
    ]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    raw = bin_chunk[offset : offset + count * ncomp * np.dtype(comp).itemsize]
    return np.frombuffer(raw, dtype=comp).reshape(count, ncomp)


def compute_world_aabb(
    gltf: dict, bin_chunk: bytes
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    nodes = gltf.get("nodes", [])
    if not nodes:
        return None

    children = {i: [] for i in range(len(nodes))}
    for i, node in enumerate(nodes):
        for c in node.get("children", []):
            children[i].append(c)

    scene = gltf["scenes"][gltf.get("scene", 0)]
    worlds: List[Optional[np.ndarray]] = [None] * len(nodes)

    def dfs(i: int, parent: Optional[np.ndarray]) -> None:
        local = _mat4_from_node(nodes[i])
        worlds[i] = local if parent is None else parent @ local
        for c in children[i]:
            dfs(c, worlds[i])

    for root in scene["nodes"]:
        dfs(root, None)

    points = []
    for i, node in enumerate(nodes):
        if "mesh" not in node or worlds[i] is None:
            continue
        mesh = gltf["meshes"][node["mesh"]]
        for prim in mesh["primitives"]:
            pos = _accessor_array(gltf, bin_chunk, prim["attributes"]["POSITION"])
            ones = np.ones((pos.shape[0], 1), dtype=np.float64)
            hom = np.concatenate([pos.astype(np.float64), ones], axis=1)
            world = (worlds[i] @ hom.T).T[:, :3]
            points.append(world)

    if not points:
        return None

    pts = np.concatenate(points, axis=0)
    return pts.min(axis=0), pts.max(axis=0)


def normalize_glb_origin_and_scale(
    src: Path,
    dst: Path,
    target_height: Optional[float] = None,
    min_scale: float = 0.5,
    max_scale: float = 4.0,
) -> Dict:
    """Rewrite GLB so bottom-center is at origin and height matches target."""
    gltf, bin_chunk = load_glb(src)
    aabb = compute_world_aabb(gltf, bin_chunk)
    if aabb is None:
        raise RuntimeError(f"No mesh geometry in {src}")

    mn, mx = aabb
    size = mx - mn
    height = float(size[1])
    if height < 1e-6:
        raise RuntimeError(f"Degenerate height in {src}")

    if target_height is None:
        target_height = target_height_for(src.name)
    scale = float(np.clip(target_height / height, min_scale, max_scale))

    cx = float((mn[0] + mx[0]) * 0.5)
    cz = float((mn[2] + mx[2]) * 0.5)
    min_y = float(mn[1])

    # p' = S * p + T, with bottom-center mapped to origin.
    translation = [-scale * cx, -scale * min_y, -scale * cz]

    original_roots = list(gltf["scenes"][gltf.get("scene", 0)]["nodes"])
    root_node = {
        "name": "habitat_normalized_root",
        "translation": translation,
        "scale": [scale, scale, scale],
        "children": original_roots,
    }
    nodes = gltf.setdefault("nodes", [])
    root_idx = len(nodes)
    nodes.append(root_node)
    gltf["scenes"][gltf.get("scene", 0)]["nodes"] = [root_idx]

    # Ensure buffer length metadata stays consistent if present.
    if "buffers" in gltf and gltf["buffers"]:
        gltf["buffers"][0]["byteLength"] = len(bin_chunk)

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_glb(dst, gltf, bin_chunk)

    out_min = np.array(
        [
            scale * (mn[0] - cx),
            0.0,
            scale * (mn[2] - cz),
        ]
    )
    out_max = np.array(
        [
            scale * (mx[0] - cx),
            scale * (mx[1] - min_y),
            scale * (mx[2] - cz),
        ]
    )
    out_size = out_max - out_min

    return {
        "source": str(src),
        "output": str(dst),
        "category": category_for(src.name),
        "raw_min": mn.tolist(),
        "raw_max": mx.tolist(),
        "raw_size": size.tolist(),
        "scale_applied": scale,
        "target_height_m": target_height,
        "normalized_min": out_min.tolist(),
        "normalized_max": out_max.tolist(),
        "normalized_size": out_size.tolist(),
        "half_height_m": float(out_size[1] * 0.5),
        "footprint_m": [float(out_size[0]), float(out_size[2])],
    }


def make_object_config(
    glb_name: str,
    mass: float,
    *,
    use_bounding_box_for_collision: bool = True,
    margin: float = 0.01,
) -> dict:
    return {
        "render_asset": glb_name,
        "collision_asset": glb_name,
        "up": [0.0, 1.0, 0.0],
        "front": [0.0, 0.0, -1.0],
        "scale": [1.0, 1.0, 1.0],
        "units_to_meters": 1.0,
        "mass": float(mass),
        "margin": float(margin),
        "use_bounding_box_for_collision": bool(use_bounding_box_for_collision),
        "join_collision_meshes": True,
        "friction_coefficient": 0.5,
        "restitution_coefficient": 0.0,
    }
