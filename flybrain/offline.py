"""Offline pipeline: feather files -> LIF sim -> raster PNG + .npz.

Functional form of the original app.py (no import-time side effects).
``main()`` preserves exact constants/behaviour via :class:`OfflineConfig`.
"""
import numpy as np
import torch

from . import config as C
from .config import OfflineConfig, resolve_device
from . import connectome as CX
from .lif import FlyBrainLIF
from .pools import build_monitor_set
from .viz import LiveViz


def run_offline(cfg: OfflineConfig | None = None) -> dict:
    cfg = cfg or OfflineConfig()
    device = resolve_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ann, nt, conn = CX.load_dataframes(cfg.ann_path, cfg.nt_path, cfg.conn_path)
    id_ann, id_nt, c_pre, c_post, c_w = CX.resolve_id_columns(ann, nt, conn)
    ann_ids = np.unique(ann[id_ann].dropna().to_numpy())
    pre, post, w = CX.filter_edges(
        conn, c_pre, c_post, c_w, ann_ids, cfg.min_weight)
    universe, lut, pre_i, post_i = CX.build_universe(pre, post)
    n = len(universe)
    sign = CX.build_nt_signs(nt, id_nt, universe)
    sens_idx, mot_idx = CX.select_sensory_motor(ann, id_ann, lut, universe, n)
    sens_idx, mot_idx = CX.apply_degree_fallbacks(
        sens_idx, mot_idx, pre_i, post_i, n)
    W = CX.build_signed_sparse_W(
        pre_i, post_i, w, sign, n, cfg.norm_mode, cfg.gain, cfg.w_syn, device)

    brain = FlyBrainLIF(W, n, dt=cfg.dt, tau_mem=cfg.tau_mem,
                        v_thr=cfg.v_thr, v_reset=cfg.v_reset,
                        refractory=cfg.refractory, device=device)

    mon, bounds = build_monitor_set(sens_idx, mot_idx, n, cfg.monitor_cap)
    viz = LiveViz(len(mon), live=cfg.live)

    stim_sub = np.random.choice(
        sens_idx, size=max(1, int(len(sens_idx) * cfg.stim_fraction)),
        replace=False)
    stim_sub_t = torch.from_numpy(stim_sub).to(device)
    mon_t = torch.from_numpy(mon).to(device)

    raster = np.zeros((cfg.t_steps, len(mon)), dtype=np.uint8)
    pop_rate = np.zeros(cfg.t_steps)
    mot_rate = np.zeros(cfg.t_steps)
    mot_idx_t = torch.from_numpy(mot_idx).to(device)

    print(f"Simulating {cfg.t_steps} steps...")
    for t in range(cfg.t_steps):
        ext = torch.zeros(n, device=device)
        for (a, b) in cfg.stim_windows:
            if a <= t < b:
                ev = torch.rand(len(stim_sub), device=device) < (
                    cfg.stim_rate_hz * cfg.dt / 1000.0)
                ext[stim_sub_t] = cfg.i_ext * ev.float()
        S = brain.step(ext, noise=cfg.background_noise)
        ms = S[mon_t].cpu().numpy()
        raster[t] = ms
        pop_rate[t] = S.mean().item()
        mot_rate[t] = S[mot_idx_t].mean().item()
        if t % cfg.vis_every == 0:
            viz.update(t, ms, pop_rate[t])
        if pop_rate[t] > 0.5:
            print(f"  [warn] runaway activity at t={t} "
                  f"(rate={pop_rate[t]:.2f}) -> lower GAIN")
            break

    kernel = np.ones(20) / 20
    mot_smooth = np.convolve(mot_rate, kernel, mode="same")
    base = mot_smooth[:cfg.stim_windows[0][0]].mean()
    resp = mot_smooth[cfg.stim_windows[0][1]:
                      cfg.stim_windows[0][1] + 150].mean()
    print(f"\nMotor response — baseline {base:.4f} vs post-stim {resp:.4f} "
          f"(ratio {resp / max(base, 1e-9):.1f}x)")

    import matplotlib.pyplot as plt
    plt.ioff()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.imshow(raster.T, aspect="auto", cmap="gray_r", vmin=0, vmax=1,
              interpolation="none")
    for b in bounds:
        a1.axhline(b, color="red", lw=0.7)
    a1.set_ylabel("sensory | interneurons | motor")
    a1.set_title("Spike raster of monitored neurons")
    a2.plot(mot_smooth, label="motor pop. rate (smoothed)")
    for (a, b) in cfg.stim_windows:
        a2.axvspan(a, b, color="orange", alpha=0.3)
    a2.set_xlabel("time (ms)")
    a2.legend()
    fig.savefig("fly_brain_results.png", dpi=150)
    np.savez("fly_brain_spikes.npz", raster=raster, mon_ids=universe[mon],
             pop_rate=pop_rate, mot_rate=mot_rate)
    print("Saved: fly_brain_results.png, fly_brain_spikes.npz")
    return {"raster": raster, "pop_rate": pop_rate, "mot_rate": mot_rate,
            "bounds": bounds}


def main() -> None:
    run_offline()


if __name__ == "__main__":
    main()
