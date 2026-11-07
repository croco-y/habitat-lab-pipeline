#!/usr/bin/env python3
"""Organize Kenney furniture GLBs for Habitat-Sim.

Steps:
1. Copy / normalize selected Kenney GLBs (origin + scale)
2. Write matching *.object_config.json with collision settings
3. Write catalog.json summarizing sizes / categories

Usage:
    /home/ymq/anaconda3/envs/habitat/bin/python \\
        /home/ymq/Habitat/scripts/prepare_kenney_furniture.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/ymq/Habitat")
sys.path.insert(0, str(ROOT / "scripts"))

from kenney_glb_utils import (  # noqa: E402
    is_furniture_asset,
    make_object_config,
    mass_for,
    normalize_glb_origin_and_scale,
)


DEFAULT_SRC = ROOT / "data/assets/kenney_furniture/Models/GLTF format"
DEFAULT_DST = ROOT / "data/assets/furniture/kenney"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of stems, e.g. chair desk loungeSofa",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N assets (0 = all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src: Path = args.src
    dst: Path = args.dst

    if not src.is_dir():
        raise FileNotFoundError(src)

    dst.mkdir(parents=True, exist_ok=True)

    glbs = sorted(src.glob("*.glb"))
    glbs = [g for g in glbs if is_furniture_asset(g.name)]

    if args.only:
        allow = {s.lower().removesuffix(".glb") for s in args.only}
        glbs = [g for g in glbs if g.stem.lower() in allow]

    if args.limit > 0:
        glbs = glbs[: args.limit]

    if not glbs:
        raise RuntimeError("No furniture GLBs selected")

    catalog = {
        "source_dir": str(src),
        "output_dir": str(dst),
        "count": 0,
        "assets": {},
    }

    print(f"Source: {src}")
    print(f"Output: {dst}")
    print(f"Assets: {len(glbs)}")

    ok = 0
    for i, glb in enumerate(glbs, 1):
        out_glb = dst / glb.name
        cfg_path = dst / f"{glb.stem}.object_config.json"
        try:
            meta = normalize_glb_origin_and_scale(glb, out_glb)
            cfg = make_object_config(glb.name, mass_for(glb.name))
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
            meta["object_config"] = str(cfg_path)
            meta["mass"] = cfg["mass"]
            meta["template_handle"] = glb.stem
            catalog["assets"][glb.stem] = meta
            ok += 1
            print(
                f"[{i:03d}/{len(glbs)}] {glb.stem}: "
                f"scale={meta['scale_applied']:.3f} "
                f"size={[round(x, 3) for x in meta['normalized_size']]}"
            )
        except Exception as exc:  # keep going for batch conversion
            print(f"[{i:03d}/{len(glbs)}] FAIL {glb.name}: {exc}")

    catalog["count"] = ok
    catalog_path = dst / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")

    print("\n========== DONE ==========")
    print(f"Converted: {ok}/{len(glbs)}")
    print(f"Catalog:   {catalog_path}")
    print(f"Configs:   {dst}/*.object_config.json")


if __name__ == "__main__":
    main()
