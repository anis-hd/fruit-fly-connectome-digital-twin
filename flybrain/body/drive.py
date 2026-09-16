"""CPG locomotion drive: health, posture, gear-stepped physics."""
import math

import numpy as np
from flygym.compose.fly.base_fly import ActuatorType
from flygym_demo.complex_terrain.common import (
    LocomotionAction,
    apply_locomotion_action,
)

from . import constants as K


def settle_to_stance(sim, steps, output_dof_order, n_steps: int = 300):
    """Hold neutral pose with adhesion so the fly lands balanced."""
    neutral = steps.default_pose_by_dof_order(output_dof_order)
    sim.set_actuator_inputs("nmf", ActuatorType.POSITION, neutral)
    sim.set_leg_adhesion_states("nmf", np.ones(6))
    for _ in range(n_steps):
        sim.step()


def thorax_pos(sim, thorax_idx: int) -> np.ndarray:
    try:
        pos = np.asarray(sim.get_body_positions("nmf"), dtype=float)
        return pos[thorax_idx].copy()
    except Exception:
        return np.zeros(3)


def heading_yaw(sim, thorax_bodyid: int) -> float:
    try:
        R = sim.mj_data.xmat[thorax_bodyid].reshape(3, 3)
        return math.atan2(float(R[1, 0]), float(R[0, 0]))
    except Exception:
        return 0.0


def upright_z(sim, thorax_bodyid: int) -> float:
    try:
        R = sim.mj_data.xmat[thorax_bodyid].reshape(3, 3)
        return float(R[2, 2])
    except Exception:
        return 1.0


def healthy(sim, thorax_idx: int) -> bool:
    try:
        qpos = sim.mj_data.qpos
        if not (bool(np.isfinite(qpos).all())
                and bool(np.isfinite(sim.mj_data.qvel).all())):
            return False
        fp = thorax_pos(sim, thorax_idx)
        if not bool(np.isfinite(fp).all()):
            return False
        return bool(abs(fp[0]) < 200.0 and abs(fp[1]) < 200.0
                    and -5.0 < fp[2] < 50.0)
    except Exception:
        return False


def update_gear(gear: int, speed: float) -> int:
    if gear == 0:
        return 1 if speed > 0.20 else 0
    if gear == 1:
        if speed > 0.50:
            return 2
        if speed < 0.10:
            return 0
        return 1
    return 1 if speed < 0.40 else 2


def step_cpg(sim, cpg, steps, output_dof_order, drive: dict, gear: int):
    """Advance CPG + physics by PHYS_SUBSTEPS. Returns (possibly new) gear."""
    ampL = min(1.0, max(0.0, float(drive.get("ampL", 0.0))))
    ampR = min(1.0, max(0.0, float(drive.get("ampR", 0.0))))
    speed = min(1.0, max(0.0, float(drive.get("speed", 0.0))))
    gear = update_gear(gear, speed)
    cpg.intrinsic_amps = np.array(
        [ampL, ampL, ampL, ampR, ampR, ampR], dtype=float)
    cpg.intrinsic_freqs = np.full(6, K.GEAR_FREQS[gear])
    for _ in range(K.PHYS_SUBSTEPS):
        cpg.step()
        joint_angles = steps.get_joint_angles_by_dof_order(
            cpg.curr_phases, cpg.curr_magnitudes, output_dof_order)
        adhesion = steps.get_adhesion_onoff_by_phase(
            cpg.curr_phases).astype(float)
        apply_locomotion_action(
            sim, "nmf",
            LocomotionAction(joint_angles=joint_angles,
                             adhesion_onoff=adhesion))
        sim.step()
    return gear
