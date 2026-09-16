"""FlyGymBridge facade: composes world/drive/senses/render modules.

Public surface matches the original flygym_bridge.py (``FlyGymBridge``,
``FRAME_EVERY``) so server code and old imports keep working.
"""
import random
import time

import mujoco as mj
import numpy as np

from . import constants as K
from .constants import FRAME_EVERY  # noqa: F401  (public re-export)
from .drive import (healthy, heading_yaw, settle_to_stance, step_cpg,
                    thorax_pos, upright_z)
from .render import render_follow_jpeg
from .senses import read_senses
from .world import build_world

__all__ = ["FlyGymBridge", "FRAME_EVERY"]


class FlyGymBridge:
    def __init__(self):
        t0 = time.perf_counter()
        parts = build_world()
        self.fly = parts["fly"]
        self.world = parts["world"]
        self.sim = parts["sim"]
        self.cpg = parts["cpg"]
        self.steps = parts["steps"]
        self.output_dof_order = parts["output_dof_order"]
        self._thorax_idx = parts["thorax_idx"]
        self._touch_segs = parts["touch_segs"]
        self.food_z = parts["food_z"]
        self._gear = 1
        self._thorax_bodyid = None
        self._tipped_time = 0.0

        settle_to_stance(self.sim, self.steps, self.output_dof_order)

        jd_order = list(self.fly.get_jointdofs_order())
        self._propL_idx = np.array(
            [k for k, jd in enumerate(jd_order)
             if str(jd.child.pos).startswith("l")], dtype=int)
        self._propR_idx = np.array(
            [k for k, jd in enumerate(jd_order)
             if str(jd.child.pos).startswith("r")], dtype=int)
        self._qref = np.asarray(
            self.sim.get_joint_angles("nmf"), dtype=float).copy()
        self.cam_id = mj.mj_name2id(
            self.sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, "arena")
        self.renderer = self.sim.set_renderer(
            cameras="arena", camera_res=K.FLYGYM_CAMERA_RES)
        self.food = np.array([8.0, -4.0, self.food_z])
        self.food_eaten = 0.0
        self._last_pos = None
        self._quiet_until = self.sim.time + 0.2
        self._move_food(*self.food[:2])
        print(f"  [flygym] ready in {time.perf_counter() - t0:.1f}s "
              f"({len(self.output_dof_order)} position actuators, "
              f"terrain={parts['terrain']})", flush=True)

    # ---------- helpers ----------
    def _move_food(self, x, y):
        try:
            gid = mj.mj_name2id(
                self.sim.mj_model, mj.mjtObj.mjOBJ_GEOM, "berry_geom")
            self.sim.mj_model.geom_pos[gid] = np.array([x, y, self.food_z])
            bid = mj.mj_name2id(
                self.sim.mj_model, mj.mjtObj.mjOBJ_BODY, "berry")
            self.sim.mj_data.xpos[bid] = np.array([x, y, self.food_z])
        except Exception as e:
            print(f"  [flygym] food move failed: {e}", flush=True)

    def _thorax_id(self):
        if self._thorax_bodyid is None:
            ids = self.sim._internal_bodyids_by_fly["nmf"]
            self._thorax_bodyid = int(ids[self._thorax_idx])
        return self._thorax_bodyid

    def fly_pos(self):
        return thorax_pos(self.sim, self._thorax_idx)

    def heading(self):
        return heading_yaw(self.sim, self._thorax_id())

    # ---------- per-step drive ----------
    def healthy(self):
        return healthy(self.sim, self._thorax_idx)

    def upright(self):
        return upright_z(self.sim, self._thorax_id())

    def step(self, drive):
        if self.sim.time < self._quiet_until:
            drive = {"ampL": 0.0, "ampR": 0.0, "turn": 0.0, "speed": 0.0}
        if self.upright() < 0.5:
            self._tipped_time += K.PHYS_SUBSTEPS * self.sim.timestep
            if self._tipped_time > 1.0:
                print("  [flygym] tipped over, respawning", flush=True)
                self.reset()
                return
        else:
            self._tipped_time = 0.0
        self._gear = step_cpg(
            self.sim, self.cpg, self.steps, self.output_dof_order,
            drive, self._gear)

    # ---------- senses ----------
    def field_temp(self, x, y):
        from .senses import field_temp
        return field_temp(x, y)

    def field_humid(self, x, y):
        from .senses import field_humid
        return field_humid(x, y)

    def sense(self):
        ctx = {
            "sim": self.sim, "thorax_idx": self._thorax_idx,
            "thorax_bodyid": self._thorax_id(),
            "touch_segs": self._touch_segs,
            "food": self.food, "food_eaten": self.food_eaten,
            "food_z": self.food_z, "last_pos": self._last_pos,
            "propL_idx": self._propL_idx, "propR_idx": self._propR_idx,
            "qref": self._qref, "move_food": self._move_food,
        }
        out = read_senses(ctx)
        self.food = ctx["food"]
        self.food_eaten = ctx["food_eaten"]
        self._last_pos = ctx["last_pos"]
        return out

    # ---------- video ----------
    def render_jpeg(self):
        return render_follow_jpeg(
            self.sim, self.renderer, self.cam_id, self.fly_pos())

    def drop_food(self, x=None, y=None):
        fp = self.fly_pos()
        if x is None:
            ang = random.random() * 2 * math.pi
            r = random.uniform(*K.FOOD_SPAWN_R)
            x, y = fp[0] + math.cos(ang) * r, fp[1] + math.sin(ang) * r
        x, y = float(np.clip(x, -10.0, 18.0)), float(np.clip(y, -10.0, 10.0))
        self.food = np.array([x, y, self.food_z])
        self.food_eaten = 0.0
        self._move_food(x, y)

    def reset(self):
        self.sim.reset()
        self.cpg.reset(
            init_phases=np.array([K.TRIPOD_INIT[leg] for leg in K.LEG_ORDER]),
            init_magnitudes=np.zeros(6))
        self._gear = 1
        self._thorax_bodyid = None
        self._last_pos = None
        self._tipped_time = 0.0
        self._quiet_until = self.sim.time + 0.2
        settle_to_stance(self.sim, self.steps, self.output_dof_order)
        self._qref = np.asarray(
            self.sim.get_joint_angles("nmf"), dtype=float).copy()
        self.drop_food()
