"""Load feather connectome into a Runtime. Split from simulation.py."""
import numpy as np
import torch

from . import config as C
from . import connectome as CX
from .config import ServerConfig, resolve_device
from .pools import (build_annotations_cache, build_bridge_channels,
                    build_monitor_set, build_motor_pools)


def load_connectome_into(rt, cfg: ServerConfig | None = None) -> int:
    cfg = cfg or ServerConfig()
    device = resolve_device()
    bd = rt.brain_data
    st = rt.state

    ann, nt, conn = CX.load_dataframes(cfg.ann_path, cfg.nt_path, cfg.conn_path)
    id_ann, id_nt, c_pre, c_post, c_w = CX.resolve_id_columns(ann, nt, conn)
    bd.id_ann = id_ann

    ann_ids = np.unique(ann[id_ann].dropna().to_numpy())
    pre, post, w = CX.filter_edges(
        conn, c_pre, c_post, c_w, ann_ids, cfg.min_weight)
    universe, lut, pre_i, post_i = CX.build_universe(pre, post)
    bd.universe, bd.lut = universe, lut
    n = len(universe)
    sign = CX.build_nt_signs(nt, id_nt, universe)

    sens_idx, mot_idx = CX.select_sensory_motor(ann, id_ann, lut, universe, n)
    sens_idx, mot_idx = CX.apply_degree_fallbacks(
        sens_idx, mot_idx, pre_i, post_i, n)
    bd.sens_idx, bd.mot_idx = sens_idx, mot_idx

    # Anatomical body wiring (must precede monitor set: reorders mot_idx L/R).
    bd.bridge_sens_subs = build_bridge_channels(
        ann, id_ann, lut, universe, sens_idx, device)
    pools = build_motor_pools(ann, id_ann, lut, universe, mot_idx, device)
    bd.mot_idx = pools["mot_idx"]
    bd.leg_pool_t = pools["leg_pool_t"]
    bd.desc_t, bd.descL_t, bd.descR_t = (
        pools["desc_t"], pools["descL_t"], pools["descR_t"])
    bd.abd_t, bd.head_t = pools["abd_t"], pools["head_t"]
    bd.mot_left_n, bd.mot_right_n = (
        pools["mot_left_n"], pools["mot_right_n"])
    bd.mot_left_t, bd.mot_right_t = (
        pools["mot_left_t"], pools["mot_right_t"])
    print(f"  bridge channels: "
          f"{[(c, bd.bridge_sens_subs[c].numel()) for c in C.BRIDGE_CHANNELS]}"
          f" | motor L/R: {bd.mot_left_t.numel()}/{bd.mot_right_t.numel()}"
          f" | leg pools: {[(k, v.numel()) for k, v in bd.leg_pool_t.items()]}"
          f" | desc: {bd.desc_t.numel()}")

    bd.W = CX.build_signed_sparse_W(
        pre_i, post_i, w, sign, n, cfg.norm_mode, cfg.gain, cfg.w_syn, device)

    bd.mon, bd.bounds = build_monitor_set(
        bd.sens_idx, bd.mot_idx, n, cfg.monitor_cap)
    bd.stim_sub = np.random.choice(
        bd.sens_idx,
        size=max(1, int(len(bd.sens_idx) * st.stim_fraction)), replace=False)
    bd.stim_sub_t = torch.from_numpy(bd.stim_sub).to(device)
    bd.mon_t = torch.from_numpy(bd.mon).to(device)
    bd.mon_ids_list = universe[bd.mon].tolist()
    bd.cached_annotations = build_annotations_cache(
        ann, id_ann, lut, universe, bd.mon)
    return n
