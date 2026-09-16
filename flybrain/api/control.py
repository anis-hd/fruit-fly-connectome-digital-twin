"""Run-control actions: start/pause/reset/stim/params. From server.py."""
import numpy as np

from .. import config as C


def state_dict(rt) -> dict:
    st, bd, drive = rt.state, rt.brain_data, rt.drive
    W = bd.W
    if W is not None and hasattr(W, "_nnz"):
        n_syn = int(W._nnz())
    elif W is not None:
        n_syn = int(W.values().numel())
    else:
        n_syn = 0
    return {
        "running": st.running,
        "paused": st.paused,
        "step": st.step,
        "T_STEPS": st.T_STEPS,
        "speed": st.speed,
        "stim_windows": st.stim_windows,
        "stim_rate_hz": st.stim_rate_hz,
        "stim_fraction": st.stim_fraction,
        "i_ext": st.i_ext,
        "gain": st.gain,
        "noise": st.noise,
        "tau_mem": st.tau_mem,
        "v_thr": st.v_thr,
        "num_neurons": len(bd.universe) if bd.universe is not None else 0,
        "num_sensory": len(bd.sens_idx) if bd.sens_idx is not None else 0,
        "num_motor": len(bd.mot_idx) if bd.mot_idx is not None else 0,
        "num_monitored": len(bd.mon) if bd.mon is not None else 0,
        "num_synapses": n_syn,
        "motor_left_count": bd.mot_left_n,
        "motor_right_count": bd.mot_right_n,
        "flygym": drive.flygym is not None,
        "flygym_tele": dict(drive.flygym_tele),
    }


def _fresh_run(rt):
    st, drive = rt.state, rt.drive
    st.step = 0
    drive.pop_rate_history = []
    drive.mot_rate_history = []
    drive.raster_buffer = []
    drive.run_completed = False
    if rt.brain:
        rt.brain.reset()


async def apply_control(rt, action: dict) -> dict:
    """Apply one control action; returns fresh state dict."""
    st = rt.state
    drive = rt.drive
    act = action.get("action")
    if act == "pause":
        st.paused = True
    elif act == "resume":
        st.paused = False
    elif act == "start":
        if st.step >= st.T_STEPS:
            _fresh_run(rt)
        st.paused = False
        st.running = True
    elif act == "reset":
        st.paused = False
        st.running = False
        _fresh_run(rt)
    elif act == "stop":
        st.running = False
    elif act == "set_speed":
        st.speed = action.get("value", 1.0)
    elif act == "set_param":
        param = action.get("param")
        value = action.get("value")
        updates = {}
        if param is not None and value is not None:
            updates[param] = value
        for key in ("gain", "noise", "tau_mem", "v_thr",
                    "stim_rate_hz", "stim_fraction", "i_ext"):
            if key in action:
                updates[key] = action[key]
        if "gain" in updates:
            st.gain = updates["gain"]
        if "noise" in updates:
            st.noise = updates["noise"]
        if "tau_mem" in updates:
            st.tau_mem = updates["tau_mem"]
            if rt.brain:
                rt.brain.update_params(tau_mem=updates["tau_mem"])
        if "v_thr" in updates:
            st.v_thr = updates["v_thr"]
            if rt.brain:
                rt.brain.update_params(v_thr=updates["v_thr"])
        if "stim_rate_hz" in updates:
            st.stim_rate_hz = updates["stim_rate_hz"]
        if "stim_fraction" in updates:
            st.stim_fraction = updates["stim_fraction"]
        if "i_ext" in updates:
            st.i_ext = updates["i_ext"]
    elif act == "stimulate_sensory":
        st.manual_stim_sensory = C.MANUAL_STIM_DURATION
        if st.step >= st.T_STEPS:
            _fresh_run(rt)
        st.paused = False
        st.running = True
    elif act == "stimulate_all":
        st.manual_stim_all = C.MANUAL_STIM_DURATION
        if st.step >= st.T_STEPS:
            _fresh_run(rt)
        st.paused = False
        st.running = True
    elif act == "flygym_drop_food":
        if drive.flygym is not None:
            drive.flygym.drop_food()
    elif act == "flygym_reset":
        if drive.flygym is not None:
            drive.flygym.reset()
    return state_dict(rt)


def random_neurons(rt, limit: int = 1000) -> dict:
    bd = rt.brain_data
    if bd.universe is None:
        return {"neurons": []}
    idx = np.random.choice(
        len(bd.universe), size=min(limit, len(bd.universe)), replace=False)
    return {
        "neurons": [
            {"id": int(bd.universe[i]), "index": int(i),
             "x": 0, "y": 0, "z": 0, "type": "interneuron"}
            for i in idx
        ]
    }
