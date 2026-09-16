"""MuJoCo world construction: fly, terrain, markers, camera."""
import os
import time

import mujoco as mj
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment
from flygym.compose import BlocksTerrainWorld, FlatGroundWorld
from flygym.utils.math import Rotation3D

from . import constants as K
from .math_utils import mat2quat
from flygym_demo.complex_terrain import (
    PreprogrammedSteps,
    get_default_locomotion_dof_order,
    make_tripod_cpg_network,
)
from flygym_demo.complex_terrain.common import make_locomotion_fly


def build_world():
    """Create fly + world + simulation. Returns dict of handles."""
    print("  [flygym] building locomotion fly + world "
          "(canonical POSITION/CPG)...", flush=True)
    t0 = time.perf_counter()
    fly = make_locomotion_fly(name="nmf")
    output_dof_order = get_default_locomotion_dof_order()
    assert len(output_dof_order) == 42, "expected 42 active leg DoFs"

    terrain = os.environ.get("FLYGYM_TERRAIN", "flat").lower()
    if terrain == "blocks":
        world = BlocksTerrainWorld(
            x_range=(-12.0, 20.0), y_range=(-12.0, 12.0),
            block_size=6.0, height_range=(0.3, 0.3), rand_seed=0)
        food_z = K.FOOD_Z_BLOCKS
        spawn = np.array([-3.0, -3.0, 2.5])
    else:
        world = FlatGroundWorld()
        food_z = K.FOOD_Z_FLAT
        spawn = K.SPAWN_POS.copy()

    fb = world.mjcf_root.worldbody.add_body(
        name="berry", pos=[8.0, -4.0, food_z])
    fb.add_geom(name="berry_geom", type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[K.FOOD_R, 0, 0], rgba=[0.95, 0.25, 0.3, 1.0],
                contype=0, conaffinity=0)
    mz = 0.06 if terrain != "blocks" else 0.50
    wb = world.mjcf_root.worldbody.add_body(
        name="warmspot", pos=[K.WARM_X, K.WARM_Y, mz])
    wb.add_geom(name="warmspot_geom", type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=[2.0, 0.05, 0], rgba=[1.0, 0.45, 0.1, 0.6],
                contype=0, conaffinity=0)
    cb = world.mjcf_root.worldbody.add_body(
        name="coolspot", pos=[K.COOL_X, K.COOL_Y, mz])
    cb.add_geom(name="coolspot_geom", type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=[2.0, 0.05, 0], rgba=[0.2, 0.5, 1.0, 0.6],
                contype=0, conaffinity=0)
    hb = world.mjcf_root.worldbody.add_body(
        name="waterdrop", pos=[K.HUMID_X, K.HUMID_Y, 1.0])
    hb.add_geom(name="waterdrop_geom", type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[1.0, 0, 0], rgba=[0.3, 0.6, 1.0, 0.8],
                contype=0, conaffinity=0)
    p = np.array([28.0, -28.0, 24.0])
    d = -p / np.linalg.norm(p)
    z_cam = -d
    x_cam = np.cross(np.array([0., 0., 1.]), z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    world.mjcf_root.worldbody.add_camera(
        name="arena", pos=list(p),
        quat=list(mat2quat(np.column_stack([x_cam, y_cam, z_cam]))))
    world.add_fly(
        fly, spawn_position=spawn,
        spawn_rotation=Rotation3D(format="quat", values=(1.0, 0.0, 0.0, 0.0)))
    sim = Simulation(world)
    sim.reset()

    cpg = make_tripod_cpg_network(sim.timestep)
    cpg.reset(
        init_phases=np.array([K.TRIPOD_INIT[leg] for leg in K.LEG_ORDER]),
        init_magnitudes=np.zeros(6))
    steps = PreprogrammedSteps()

    order = list(fly.get_bodysegs_order())
    try:
        thorax_idx = order.index(BodySegment("c_thorax"))
    except ValueError:
        thorax_idx = 0

    try:
        valid = set(s.name for s in fly.get_bodysegs_order())
    except Exception:
        valid = set()
    touch_segs = []
    for leg in K.LEG_ORDER:
        for link in ("tarsus4", "tarsus5"):
            name = f"{leg}_{link}"
            if not valid or name in valid:
                touch_segs.append(name)

    print(f"  [flygym] world ready in {time.perf_counter() - t0:.1f}s "
          f"({len(output_dof_order)} position actuators, "
          f"terrain={terrain})", flush=True)
    return {
        "fly": fly, "world": world, "sim": sim, "cpg": cpg,
        "steps": steps, "output_dof_order": output_dof_order,
        "thorax_idx": thorax_idx, "touch_segs": touch_segs,
        "food_z": food_z, "terrain": terrain,
    }
