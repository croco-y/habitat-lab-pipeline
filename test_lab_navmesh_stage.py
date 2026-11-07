"""Build and validate a room NavMesh using its StageAttributes configuration.

Usage in Ubuntu (conda environment with habitat_sim installed):
    python /home/ymq/Habitat/test_lab_navmesh_stage.py

Input:
    /home/ymq/Habitat/data/my_lab/room_base_8x6.stage_config.json
Output:
    /home/ymq/Habitat/data/my_lab/room_base_8x6.navmesh
"""
from pathlib import Path
import math
import habitat_sim
import magnum as mn

LAB = Path('/home/ymq/Habitat/data/my_lab')
STAGE = LAB / 'room_base_8x6.stage_config.json'
GLB = LAB / 'room_base_8x6.glb'
NAVMESH = LAB / 'room_base_8x6.navmesh'
for p in (STAGE, GLB):
    if not p.is_file():
        raise FileNotFoundError(f'Expected input file missing: {p}')

cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = str(STAGE)
agent_cfg = habitat_sim.agent.AgentConfiguration()

sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
try:
    nav = habitat_sim.NavMeshSettings()
    nav.set_defaults()
    nav.agent_radius = 0.25    # 25 cm clearance around navigation center
    nav.agent_height = 1.4    # 1.4 m robot height
    ok = sim.recompute_navmesh(sim.pathfinder, nav)
    if not ok:
        raise RuntimeError('NavMesh calculation failed. Check GLB collision geometry and stage axes.')

    pf = sim.pathfinder
    # Room's floor is at Y=0 in Habitat coordinates.
    center = mn.Vector3(0.0, 0.0, 0.0)
    is_floor = pf.is_navigable(center, max_y_delta=0.20)
    print('NavMesh loaded:', pf.is_loaded)
    print('All connected islands:', pf.num_islands)
    print('Total navigable area (may include ceiling):', round(pf.navigable_area, 3), 'm^2')
    print('Room center at [0, 0, 0] is navigable:', is_floor)
    snap = pf.snap_point(center)
    print('Closest navigable point to floor center:', snap)
    if is_floor:
        floor_island = pf.get_island(center)
        print('FLOOR island ID:', floor_island)
        print('FLOOR island area:', round(pf.island_area(floor_island), 3), 'm^2')
        test = pf.get_random_navigable_point(island_index=floor_island)
        print('Sample point on FLOOR island:', test)
        if not math.isfinite(float(test.y)) or abs(float(test.y)) > 0.4:
            print('WARNING: sampled point height is unexpected. Verify orientation and collision mesh.')
    else:
        print('WARNING: Floor center not found near Y=0. Check GLB/stage orientation or geometry.')

    written = pf.save_nav_mesh(str(NAVMESH))
    print('Saved NavMesh:', NAVMESH, '| success:', written)
    if not written:
        raise RuntimeError('Could not save NavMesh')
    if pf.num_islands > 1:
        print('NOTE: ceiling or other surfaces may form extra navigation islands; use only FLOOR island.')
finally:
    sim.close()
