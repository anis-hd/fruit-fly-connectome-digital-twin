"""Sensory readout: contact, food geometry, climate, proprioception."""
import math
import random

import numpy as np

from . import constants as K
from .math_utils import wrap_angle


def field_temp(x: float, y: float) -> float:
    dw2 = (x - K.WARM_X) ** 2 + (y - K.WARM_Y) ** 2
    dc2 = (x - K.COOL_X) ** 2 + (y - K.COOL_Y) ** 2
    return (K.TEMP_AMBIENT
            + (K.WARM_T - K.TEMP_AMBIENT) * math.exp(-dw2 / (2 * K.WARM_SIGMA ** 2))
            + (K.COOL_T - K.TEMP_AMBIENT) * math.exp(-dc2 / (2 * K.COOL_SIGMA ** 2)))


def field_humid(x: float, y: float) -> float:
    dh2 = (x - K.HUMID_X) ** 2 + (y - K.HUMID_Y) ** 2
    return K.HUMID_AMBIENT + (1.0 - K.HUMID_AMBIENT) * math.exp(
        -dh2 / (2 * K.HUMID_SIGMA ** 2))


def leg_contact(sim, touch_segs) -> np.ndarray:
    try:
        F = np.asarray(sim.get_bodysegment_contact_forces(
            "nmf", touch_segs, ground_only=True), dtype=float)
        mags = np.linalg.norm(F, axis=1)
        return mags.reshape(len(K.LEG_ORDER), -1).max(axis=1)
    except Exception:
        return np.zeros(len(K.LEG_ORDER))


def read_senses(ctx: dict) -> dict:
    """Read touch/pose/food geometry. Mutates food state on eat events."""
    from .drive import heading_yaw, thorax_pos

    sim = ctx["sim"]
    fp = thorax_pos(sim, ctx["thorax_idx"])
    h = heading_yaw(sim, ctx["thorax_bodyid"])
    per_leg = leg_contact(sim, ctx["touch_segs"])
    touchL = float(per_leg[:3].max(initial=0.0))
    touchR = float(per_leg[3:].max(initial=0.0))
    dx, dy = ctx["food"][0] - fp[0], ctx["food"][1] - fp[1]
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    lat = math.sin(wrap_angle(bearing - h))
    smell = 1.0 / (1.0 + dist * 0.15)
    touch = min(1.0, max(touchL, touchR) * 2.0)
    if dist < K.EAT_DIST:
        ctx["food_eaten"] = min(1.0, ctx["food_eaten"] + 0.05)
        taste = 1.0
        if ctx["food_eaten"] >= 1.0:
            ang = random.random() * 2 * math.pi
            r = random.uniform(*K.FOOD_SPAWN_R)
            nx = float(np.clip(fp[0] + math.cos(ang) * r, -10.0, 18.0))
            ny = float(np.clip(fp[1] + math.sin(ang) * r, -10.0, 10.0))
            ctx["food"] = np.array([nx, ny, ctx["food_z"]])
            ctx["move_food"](nx, ny)
            ctx["food_eaten"] = 0.0
    else:
        taste = 0.0
        ctx["food_eaten"] = max(0.0, ctx["food_eaten"] - 0.01)
    vel = ((fp - ctx["last_pos"])
           / max(1e-9, K.PHYS_SUBSTEPS * sim.timestep)
           if ctx["last_pos"] is not None else np.zeros(3))
    ctx["last_pos"] = fp.copy()
    speed = min(1.0, float(np.linalg.norm(vel)) / 20.0)
    wind = speed
    light = min(1.0, 0.15 + 0.6 * speed)
    temp_c = field_temp(float(fp[0]), float(fp[1]))
    warm = max(0.0, min(1.0, (temp_c - K.TEMP_AMBIENT) / K.TEMP_RANGE))
    cool = max(0.0, min(1.0, (K.TEMP_AMBIENT - temp_c) / K.TEMP_RANGE))
    humid = max(0.0, min(1.0, field_humid(float(fp[0]), float(fp[1]))))
    try:
        q = np.asarray(sim.get_joint_angles("nmf"), dtype=float)
        qd = np.asarray(sim.get_joint_velocities("nmf"), dtype=float)
        devL = float(np.abs(q[ctx["propL_idx"]]
                            - ctx["qref"][ctx["propL_idx"]]).mean())
        devR = float(np.abs(q[ctx["propR_idx"]]
                            - ctx["qref"][ctx["propR_idx"]]).mean())
        velL = float(np.abs(qd[ctx["propL_idx"]]).mean())
        velR = float(np.abs(qd[ctx["propR_idx"]]).mean())
        propL = min(1.0, 0.6 * min(1.0, devL / K.PROP_DEV_REF)
                    + 0.4 * min(1.0, velL / K.PROP_VEL_REF))
        propR = min(1.0, 0.6 * min(1.0, devR / K.PROP_DEV_REF)
                    + 0.4 * min(1.0, velR / K.PROP_VEL_REF))
    except Exception:
        propL = propR = 0.0
    return {
        "smell": smell, "taste": taste, "touch": touch,
        "touchL": min(1.0, touchL * 2.0), "touchR": min(1.0, touchR * 2.0),
        "left": max(0.0, min(1.0, smell * (0.5 + 0.7 * lat))),
        "right": max(0.0, min(1.0, smell * (0.5 - 0.7 * lat))),
        "wind": wind, "light": light, "warm": warm, "cool": cool,
        "humid": humid, "propL": propL, "propR": propR,
        "speed": speed, "temp_c": round(temp_c, 1),
        "food_dist": dist, "food_eaten": ctx["food_eaten"],
        "fly_xy": [float(fp[0]), float(fp[1])],
    }
