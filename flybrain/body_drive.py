"""Brain -> body -> bridge-channel closed loop. Split from simulation.py."""
import base64
import time
from datetime import datetime

from . import config as C
from .body.constants import FRAME_EVERY


async def step_body(rt, t: int, device: str):
    """Brain spikes -> CPG -> physics -> senses -> bridge channels + video."""
    st, bd, drive = rt.state, rt.brain_data, rt.drive
    flygym = drive.flygym
    try:
        if not flygym.healthy():
            print("  [flygym] unhealthy state, resetting body")
            flygym.reset()
        bm = drive.bridge_motor
        rawL = C.recruit(max(bm.get("descL", 0.0), bm.get("legL", bm["left"])))
        rawR = C.recruit(max(bm.get("descR", 0.0), bm.get("legR", bm["right"])))
        rawTurn = max(-1.0, min(1.0, ((bm.get("descR", 0.0) - bm.get("descL", 0.0))
                                      + (bm["right"] - bm["left"])) * C.TURN_GAIN))
        if abs(rawTurn) < C.TURN_DEADBAND:
            rawTurn = 0.0
        desc = bm.get("desc", 0.0)
        rawSpd = C.recruit(desc if desc > 0 else bm["all"])
        ema = drive.drive_ema
        ema["ampL"] += C.AMP_EMA_ALPHA * (rawL - ema["ampL"])
        ema["ampR"] += C.AMP_EMA_ALPHA * (rawR - ema["ampR"])
        ema["turn"] += C.TURN_EMA_ALPHA * (rawTurn - ema["turn"])
        ema["speed"] += C.SPEED_EMA_ALPHA * (rawSpd - ema["speed"])
        ampL, ampR, turn, spd = ema["ampL"], ema["ampR"], ema["turn"], ema["speed"]
        flygym.step({"ampL": ampL, "ampR": ampR, "turn": turn, "speed": spd})
        fs = flygym.sense()
        now = time.monotonic()
        for _ch in C.BRIDGE_CHANNELS:
            st.bridge[_ch] = max(
                float(fs.get(_ch, 0.0)),
                float(st.bridge.get(_ch, 0.0)
                      if now - drive.bridge_ts.get(_ch, 0.0) <= C.BRIDGE_TIMEOUT_S
                      else 0.0))
            drive.bridge_ts[_ch] = now
        drive.flygym_tele = {
            "food_dist": round(fs["food_dist"], 1),
            "food_eaten": round(fs["food_eaten"], 2),
            "touch": round(fs["touch"], 2),
            "speed": round(fs["speed"], 2),
            "temp_c": fs.get("temp_c", 25.0),
            "humid": round(fs.get("humid", 0.0), 2),
            "prop": [round(fs.get("propL", 0.0), 2),
                     round(fs.get("propR", 0.0), 2)],
            "ampL": round(ampL, 3), "ampR": round(ampR, 3),
            "spd": round(spd, 3),
            "fly_xy": fs["fly_xy"], "step": t}
        if t % 500 == 0:
            print(f"  [flygym] {datetime.now().strftime('%H:%M:%S')} t={t} "
                  f"legL={bm.get('legL', 0.0):.5f} "
                  f"legR={bm.get('legR', 0.0):.5f} "
                  f"desc={bm.get('desc', 0.0):.5f} "
                  f"descL={bm.get('descL', 0.0):.5f} "
                  f"descR={bm.get('descR', 0.0):.5f} "
                  f"all={bm.get('all', 0.0):.5f} "
                  f"-> ampL={ampL:.3f} ampR={ampR:.3f} "
                  f"spd={spd:.3f} turn={turn:+.3f}")
        if t % (FRAME_EVERY * 2) == 0:
            jpg = flygym.render_jpeg()
            if jpg:
                await rt.manager.broadcast({
                    "type": "flygym_frame",
                    "step": t,
                    "jpg": base64.b64encode(jpg).decode(),
                    "tele": drive.flygym_tele,
                })
    except Exception as e:
        print(f"  [flygym] step failed: {e}")
