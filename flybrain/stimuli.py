"""External current builders: scheduled / manual / body-bridge drive."""
import time

import torch

from . import config as C


def apply_scheduled_stim(ext, t: int, state, brain_data, device: str,
                       dt: float = 1.0) -> None:
    for (a, b) in state.stim_windows:
        if a <= t < b:
            ev = torch.rand(len(brain_data.stim_sub), device=device) < (
                state.stim_rate_hz * dt / 1000.0)
            ext[brain_data.stim_sub_t] = state.i_ext * ev.float()


def apply_manual_stim(ext, state, n: int, brain_data, device: str,
                      dt: float = 1.0) -> bool:
    """Sustained Poisson bursts. Returns True if any burst step was applied."""
    active = False
    if state.manual_stim_sensory > 0:
        sub, sub_t = brain_data.stim_sub, brain_data.stim_sub_t
        ev = torch.rand(len(sub), device=device) < (
            state.stim_rate_hz * 2.0 * dt / 1000.0)
        if int(ev.sum().item()) == 0:
            ev[torch.randint(len(sub), (min(8, len(sub)),),
                             device=device)] = True
        ext[sub_t] = torch.maximum(ext[sub_t], state.i_ext * 2.0 * ev.float())
        state.manual_stim_sensory -= 1
        active = True
    if state.manual_stim_all > 0:
        ev = torch.rand(n, device=device) < 0.02  # ~2% of all neurons/step
        if int(ev.sum().item()) == 0:
            ev[torch.randint(n, (64,), device=device)] = True
        ext[ev] = torch.maximum(ext[ev], state.i_ext * 1.5)
        state.manual_stim_all -= 1
        active = True
    return active


def apply_bridge_drive(ext, state, brain_data, drive, device: str,
                     dt: float = 1.0) -> bool:
    """Level-based Poisson drive into each channel's sensory subset."""
    active = False
    if not state.bridge:
        return False
    now = time.monotonic()
    for ch, sub_t in brain_data.bridge_sens_subs.items():
        lvl = float(state.bridge.get(ch, 0.0) or 0.0)
        if now - drive.bridge_ts.get(ch, 0.0) > C.BRIDGE_TIMEOUT_S:
            lvl = 0.0
        if lvl > 0.01 and sub_t.numel() > 0:
            ev = torch.rand(sub_t.numel(), device=device) < (
                min(lvl, 1.0) * state.stim_rate_hz * 2.0 * dt / 1000.0)
            if int(ev.sum().item()) == 0 and lvl > 0.25:
                ev[torch.randint(sub_t.numel(), (4,), device=device)] = True
            ext[sub_t] = torch.maximum(ext[sub_t], state.i_ext * 2.0 * ev.float())
            active = True
    return active
