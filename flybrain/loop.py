"""Async live simulation loop. Split from simulation.py (loop half)."""
import asyncio

import numpy as np
import torch

from .body_drive import step_body
from .config import ServerConfig, resolve_device
from .lif import FlyBrainLIF
from .loader import load_connectome_into
from .stimuli import (apply_bridge_drive, apply_manual_stim,
                      apply_scheduled_stim)


def reset_run(rt):
    st, drive = rt.state, rt.drive
    st.step = 0
    drive.pop_rate_history = []
    drive.mot_rate_history = []
    drive.raster_buffer = []
    drive.run_completed = False
    if rt.brain:
        rt.brain.reset()


def _get_flygym_bridge():
    try:
        from .body.bridge import FlyGymBridge
        return FlyGymBridge, True
    except Exception as e:
        print(f"[flygym] disabled ({e})")
        return None, False


async def simulation_loop(rt, cfg: ServerConfig | None = None):
    cfg = cfg or ServerConfig()
    device = resolve_device()
    st, bd, drive = rt.state, rt.brain_data, rt.drive
    drive.loop = asyncio.get_running_loop()

    try:
        n = load_connectome_into(rt, cfg)
        rt.brain = FlyBrainLIF(
            bd.W, n, dt=cfg.dt, tau_mem=cfg.tau_mem, v_thr=st.v_thr,
            v_reset=cfg.v_reset, refractory=cfg.refractory,
            device=device).to(device)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error initializing simulation: {e}")
        return

    FlyGymBridge, have_flygym = _get_flygym_bridge()
    if have_flygym:
        try:
            drive.flygym = FlyGymBridge()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[flygym] build failed, continuing without body: {e}")
            drive.flygym = None

    mot_idx_t = torch.from_numpy(bd.mot_idx).to(device)
    st.running = True
    st.step = 0
    print(f"Starting simulation loop ({st.T_STEPS} steps)...")

    try:
        while True:
            stim_pending = (st.manual_stim_sensory > 0
                            or st.manual_stim_all > 0)
            if not st.running or st.paused or rt.brain is None:
                if stim_pending and rt.brain is not None:
                    if st.step >= st.T_STEPS:
                        reset_run(rt)
                    st.paused = False
                    st.running = True
                    await rt.manager.broadcast(
                        {"type": "status", "running": True,
                         "paused": False, "step": st.step})
                else:
                    await asyncio.sleep(0.05)
                    continue

            if st.step >= st.T_STEPS:
                if not drive.run_completed:
                    drive.run_completed = True
                    st.running = False
                    print("Simulation complete")
                    kernel = np.ones(20) / 20
                    mot_smooth = np.convolve(
                        drive.mot_rate_history, kernel, mode="same")
                    base = (mot_smooth[:st.stim_windows[0][0]].mean()
                            if len(mot_smooth) > st.stim_windows[0][0] else 0)
                    hi = st.stim_windows[0][1] + 150
                    resp = (mot_smooth[st.stim_windows[0][1]:hi].mean()
                            if len(mot_smooth) > hi else 0)
                    print(f"\nMotor response — baseline {base:.4f} vs "
                          f"post-stim {resp:.4f} "
                          f"(ratio {resp / max(base, 1e-9):.1f}x)")
                    await rt.manager.broadcast({
                        "type": "complete",
                        "pop_rate_history": drive.pop_rate_history,
                        "mot_rate_history": drive.mot_rate_history,
                        "raster": drive.raster_buffer,
                    })
                await asyncio.sleep(0.05)
                continue

            t = st.step
            ext = torch.zeros(n, device=device)
            apply_scheduled_stim(ext, t, st, bd, device, dt=cfg.dt)
            stim_active = apply_manual_stim(ext, st, n, bd, device, dt=cfg.dt)
            if apply_bridge_drive(ext, st, bd, drive, device, dt=cfg.dt):
                stim_active = True

            S = rt.brain.step(ext, noise=st.noise)
            pop_rate = S.mean().item()
            mot_rate = S[mot_idx_t].mean().item()
            drive.bridge_motor.update(
                {"left": S[bd.mot_left_t].mean().item(),
                 "right": S[bd.mot_right_t].mean().item(),
                 "all": mot_rate, "step": t})
            if t % cfg.vis_every == 0:
                ms = S[bd.mon_t].cpu().numpy()
                for _k, _v in bd.leg_pool_t.items():
                    drive.bridge_motor[_k] = (
                        S[_v].mean().item() if _v.numel() > 0 else 0.0)
                drive.bridge_motor["desc"] = (
                    S[bd.desc_t].mean().item()
                    if bd.desc_t.numel() > 0 else 0.0)
                for _k, _v in (("descL", bd.descL_t), ("descR", bd.descR_t),
                               ("abd", bd.abd_t), ("head", bd.head_t)):
                    drive.bridge_motor[_k] = (
                        S[_v].mean().item()
                        if _v is not None and _v.numel() > 0 else 0.0)
                drive.pop_rate_history.append(pop_rate)
                drive.mot_rate_history.append(mot_rate)
                drive.raster_buffer.append(np.flatnonzero(ms).tolist())

            if t % cfg.vis_every == 0:
                fired = np.flatnonzero(ms).tolist()
                bm = drive.bridge_motor
                await rt.manager.broadcast({
                    "type": "spikes",
                    "t": t,
                    "step": t,
                    "spikes": fired,
                    "n": int(len(ms)),
                    "pop_rate": pop_rate,
                    "mot_rate": mot_rate,
                    "motor": {
                        "legL": bm.get("legL", 0.0),
                        "legR": bm.get("legR", 0.0),
                        "desc": bm.get("desc", 0.0),
                        "t1": (bm.get("legT1L", 0.0)
                               + bm.get("legT1R", 0.0)) / 2.0,
                        "t2": (bm.get("legT2L", 0.0)
                               + bm.get("legT2R", 0.0)) / 2.0,
                        "t3": (bm.get("legT3L", 0.0)
                               + bm.get("legT3R", 0.0)) / 2.0,
                    },
                    "bounds": bd.bounds,
                    "n_spikes": len(fired),
                    "stim_active": stim_active,
                })

            if pop_rate > 0.5:
                print(f"  [warn] runaway activity at t={t} (rate={pop_rate:.2f})")
                await rt.manager.broadcast(
                    {"type": "warning",
                     "message": f"Runaway activity at t={t}, rate={pop_rate:.2f}"})

            if drive.flygym is not None:
                await step_body(rt, t, device)

            st.step += 1
            await asyncio.sleep(cfg.dt / 1000.0 / st.speed)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in simulation loop: {e}")
        st.running = False


# Back-compat alias (old private name).
_reset_run = reset_run
