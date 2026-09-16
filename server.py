"""Fly Brain Server — thin entry point (was 1124 lines).

Run: ``uvicorn server:app --port 8000``.
Full implementation lives in :mod:`flybrain` (config/connectome/pools/
lif/stimuli/simulation/api/body). This module wires the singletons and
keeps back-compat names (``sim_state``, ``manager``, ...) alive.
"""
from flybrain.api import create_app
from flybrain.api.control import apply_control, state_dict
from flybrain.config import ServerConfig
from flybrain.lif import FlyBrainLIF
from flybrain.logging_utils import setup_file_logging
from flybrain.runtime import Runtime, SimState
from flybrain.simulation import load_connectome_into, simulation_loop

setup_file_logging()

config = ServerConfig()
runtime = Runtime()
app = create_app(runtime, config)

# ---- Back-compat aliases (old module globals -> Runtime fields) ----
sim_state = runtime.state
manager = runtime.manager


def __getattr__(name: str):
    """Proxy brain tensors so `server.W` etc. stay live after load.

    load_connectome_into() mutates runtime.brain_data in place; a static
    snapshot taken at import would go stale. Module __getattr__ (PEP 562)
    reads through to the Runtime on every access.
    """
    bd, drive = runtime.brain_data, runtime.drive
    if name == "brain":
        return runtime.brain
    live = {
        "W": bd.W, "universe": bd.universe, "sens_idx": bd.sens_idx,
        "mot_idx": bd.mot_idx, "mon": bd.mon, "mon_t": bd.mon_t,
        "stim_sub": bd.stim_sub, "stim_sub_t": bd.stim_sub_t,
        "bounds": bd.bounds, "lut": bd.lut, "id_ann": bd.id_ann,
        "cached_annotations": bd.cached_annotations,
        "mon_ids_list": bd.mon_ids_list,
        "bridge_sens_subs": bd.bridge_sens_subs,
        "mot_left_t": bd.mot_left_t, "mot_right_t": bd.mot_right_t,
        "mot_left_n": bd.mot_left_n, "mot_right_n": bd.mot_right_n,
        "leg_pool_t": bd.leg_pool_t, "desc_t": bd.desc_t,
        "descL_t": bd.descL_t, "descR_t": bd.descR_t,
        "abd_t": bd.abd_t, "head_t": bd.head_t,
        "flygym": drive.flygym, "flygym_tele": drive.flygym_tele,
        "pop_rate_history": drive.pop_rate_history,
        "mot_rate_history": drive.mot_rate_history,
        "raster_buffer": drive.raster_buffer,
        "bridge_ts": drive.bridge_ts, "bridge_motor": drive.bridge_motor,
        "drive_ema": drive.drive_ema,
    }
    if name in live:
        return live[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "app", "runtime", "config", "sim_state", "manager",
    "SimState", "FlyBrainLIF", "simulation_loop", "load_connectome_into",
    "apply_control", "state_dict",
]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
