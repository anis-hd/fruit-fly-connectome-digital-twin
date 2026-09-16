"""Anatomical pools + monitor set + annotation cache.

Extracted from server.py load_connectome() (bridge channels, leg/desc/
abdominal/head motor pools, L/R lateralization, monitored subset,
centroid cache). Offline pipeline uses only ``build_monitor_set``.
"""
import numpy as np
import pandas as pd
import torch

from . import config as C


def _pool_u(lut, body_ids, device: str):
    t = torch.from_numpy(
        lut.reindex(np.asarray(body_ids)).to_numpy().astype(np.int64))
    return t.to(device)


def build_bridge_channels(ann, id_ann, lut, universe, sens_idx, device: str):
    """Anatomical sensory subsets per bridge channel (never random-first)."""
    info = ann.set_index(id_ann)[[
        "superclass", "somaSide", "somaNeuromere", "entryNerve",
        "type", "class", "subclass", "rootSide",
    ]]
    sens_bodies = universe[sens_idx]
    sInfo = info.reindex(sens_bodies)

    ch_mask = {
        "smell": (sInfo["entryNerve"] == "AN").to_numpy(),
        "touch": sInfo["entryNerve"].isin(list(C.LEG_NERVES)).to_numpy(),
        "left": (sInfo["somaSide"] == "L").to_numpy(),
        "right": (sInfo["somaSide"] == "R").to_numpy(),
        "taste": (sInfo["entryNerve"] == "MxLbN").to_numpy(),
        "wind": (sInfo["class"].fillna("").astype(str).str.contains(
            "johnston|chordotonal", case=False, regex=True)
            | sInfo["type"].fillna("").astype(str).str.contains(
                "johnston|chordotonal", case=False, regex=True)
            | sInfo["superclass"].fillna("").astype(str).str.contains(
                "johnston|chordotonal", case=False, regex=True)).to_numpy(),
        "light": sInfo["superclass"].isin(list(C.LIGHT_CLASSES)).to_numpy(),
        "warm": sInfo["type"].fillna("").astype(str).str.contains(
            "VP2", regex=False).to_numpy(),
        "cool": sInfo["type"].fillna("").astype(str).str.contains(
            "VP3", regex=False).to_numpy(),
    }
    bridge_sens_subs = {}
    for i, ch in enumerate(C.BRIDGE_CHANNELS):
        if ch not in ch_mask:  # universe-based channels built below
            continue
        part = sens_bodies[ch_mask[ch]]
        if len(part) < 5:  # annotation gap -> random fallback, logged loudly
            part = np.array_split(
                np.random.permutation(sens_bodies), len(C.BRIDGE_CHANNELS))[i]
            print(f"  [warn] bridge channel '{ch}' only "
                  f"{int(ch_mask[ch].sum())} labeled -> random fallback ({len(part)})")
        bridge_sens_subs[ch] = torch.from_numpy(
            np.sort(lut.reindex(part).to_numpy().astype(np.int64))).to(device)

    # Proprioception + humidity: universe-based pools (sens_idx misses some).
    uInfo = info.reindex(universe)
    is_proprio = (
        uInfo["subclass"].fillna("").astype(str).str.contains(
            "chordotonal|campaniform|bristle|hair plate",
            case=False, regex=True).to_numpy()
        | uInfo["class"].fillna("").astype(str).str.contains(
            "propriocept|tactile", case=False, regex=True).to_numpy())
    uRoot = uInfo["rootSide"].fillna("").astype(str).to_numpy()
    prop_all = universe[is_proprio]
    print(f"  proprioceptive afferents in universe: {len(prop_all):,} "
          f"(rootSide L={(is_proprio & (uRoot == 'L')).sum()}, "
          f"R={(is_proprio & (uRoot == 'R')).sum()})")
    for j, (ch, sd) in enumerate((("propL", "L"), ("propR", "R"))):
        part = universe[is_proprio & (uRoot == sd)]
        if len(part) < 5:  # keep it proprioceptive, just unlateralized
            part = np.array_split(np.random.permutation(prop_all), 2)[j]
            print(f"  [warn] bridge channel '{ch}' only labeled on side {sd} "
                  f"-> random proprioceptive fallback ({len(part)})")
        bridge_sens_subs[ch] = torch.from_numpy(
            np.sort(lut.reindex(part).to_numpy().astype(np.int64))).to(device)
    hum_bodies = np.intersect1d(
        ann.loc[ann["class"].fillna("").astype(str).str.contains(
            "hygro", case=False, regex=True), id_ann].to_numpy(), universe)
    if len(hum_bodies) < 5:
        hum_bodies = np.array_split(
            np.random.permutation(sens_bodies), len(C.BRIDGE_CHANNELS))[9]
        print("  [warn] bridge channel 'humid' unlabeled -> random fallback "
              f"({len(hum_bodies)})")
    bridge_sens_subs["humid"] = torch.from_numpy(
        np.sort(lut.reindex(hum_bodies).to_numpy().astype(np.int64))).to(device)
    return bridge_sens_subs


def build_motor_pools(ann, id_ann, lut, universe, mot_idx, device: str):
    """Leg/descending/abdominal/head pools + L/R lateralized motor split."""
    info = ann.set_index(id_ann)[[
        "superclass", "somaSide", "somaNeuromere", "entryNerve",
        "type", "class", "subclass", "rootSide",
    ]]
    mot_bodies = universe[mot_idx]
    mInfo = info.reindex(mot_bodies)
    mSide = mInfo["somaSide"].to_numpy()
    mSup = mInfo["superclass"].to_numpy()
    mNm = mInfo["somaNeuromere"].to_numpy()

    leg_pool_t = {}
    is_leg = (mSup == "vnc_motor")
    for seg in C.LEG_SEGS:
        for sd in ["L", "R"]:
            sel = is_leg & (mNm == seg) & (mSide == sd)
            leg_pool_t[f"leg{seg}{sd}"] = _pool_u(
                lut, mot_bodies[np.asarray(sel)], device)
    leg_pool_t["legL"] = _pool_u(
        lut, mot_bodies[is_leg & (mSide == "L")], device)
    leg_pool_t["legR"] = _pool_u(
        lut, mot_bodies[is_leg & (mSide == "R")], device)
    desc_bodies = np.intersect1d(
        ann.loc[ann["superclass"] == "descending_neuron", id_ann].to_numpy(),
        universe)
    desc_t = _pool_u(lut, desc_bodies, device)
    dSide = info.reindex(desc_bodies)["somaSide"].to_numpy()
    descL_t = _pool_u(lut, desc_bodies[dSide == "L"], device)
    descR_t = _pool_u(lut, desc_bodies[dSide == "R"], device)
    abd_bodies = np.intersect1d(
        ann.loc[(ann["superclass"] == "vnc_motor")
                & ann["somaNeuromere"].fillna("").astype(str).str.match(r"A\d+"),
                id_ann].to_numpy(), universe)
    abd_t = _pool_u(lut, abd_bodies, device)
    head_bodies = np.intersect1d(
        ann.loc[ann["superclass"] == "cb_motor", id_ann].to_numpy(), universe)
    head_t = _pool_u(lut, head_bodies, device)

    oL = np.where(mSide == "L")[0]
    oR = np.where(mSide == "R")[0]
    oM = np.where(~np.isin(mSide, ["L", "R"]))[0]
    if len(oL) < 5 or len(oR) < 5:
        print(f"  [warn] motor somaSide too sparse (L={len(oL)}, R={len(oR)}) "
              f"-> ordered halves")
        h = max(1, len(mot_idx) // 2)
        mot_idx_new = mot_idx
        mot_left_n, mot_right_n = h, len(mot_idx) - h
    else:
        mot_idx_new = mot_idx[np.concatenate([oL, oR, oM])]
        mot_left_n, mot_right_n = len(oL), len(oR)
    mot_left_t = torch.from_numpy(mot_idx_new[:mot_left_n]).to(device)
    mot_right_t = torch.from_numpy(
        mot_idx_new[mot_left_n:mot_left_n + mot_right_n]).to(device)
    pools = {
        "leg_pool_t": leg_pool_t, "desc_t": desc_t,
        "descL_t": descL_t, "descR_t": descR_t,
        "abd_t": abd_t, "head_t": head_t,
        "mot_idx": mot_idx_new, "mot_left_n": mot_left_n,
        "mot_right_n": mot_right_n,
        "mot_left_t": mot_left_t, "mot_right_t": mot_right_t,
    }
    return pools


def build_monitor_set(sens_idx, mot_idx, n: int, monitor_cap: int):
    inter = np.setdiff1d(np.arange(n), np.union1d(sens_idx, mot_idx))
    n_extra = max(0, monitor_cap - len(sens_idx) - len(mot_idx))
    inter_sample = np.random.choice(
        inter, size=min(n_extra, len(inter)), replace=False)
    mon = np.concatenate([sens_idx, inter_sample, mot_idx]).astype(np.int64)
    bounds = [len(sens_idx), len(sens_idx) + len(inter_sample)]
    print(f"  monitoring {len(mon):,} neurons (sensory | inter | motor)")
    return mon, bounds


def build_annotations_cache(ann, id_ann, lut, universe, mon,
                            sample_target: int = 50000):
    """Centroid cache for the 3D UI: monitored + sampled background."""
    print("  Building annotations cache for 3D visualization (using centroids)...")
    mon_map = {int(u_idx): int(m_idx) for m_idx, u_idx in enumerate(mon)}
    ann_sub = ann[ann[id_ann].isin(universe)]
    has_soma = "somaLocation" in ann_sub.columns

    mon_neurons, other_neurons = [], []
    for row in ann_sub.itertuples():
        body_id = getattr(row, id_ann)
        if body_id not in lut:
            continue
        univ_idx = int(lut[body_id])
        mon_idx = mon_map.get(univ_idx, -1)
        if has_soma:
            loc = getattr(row, "somaLocation", None)
            if (loc is None or not hasattr(loc, "__len__")
                    or len(loc) < 3 or pd.isna(loc[0])):
                continue
            x, y, z = float(loc[0]), float(loc[1]), float(loc[2])
        else:
            continue
        neuron = {
            "id": int(body_id),
            "index": mon_idx if mon_idx >= 0 else univ_idx,
            "univ_index": univ_idx,
            "mon_index": mon_idx,
            "x": x, "y": y, "z": z,
        }
        for attr in ("superclass", "class", "type"):
            val = getattr(row, attr, None)
            if val is not None and pd.notna(val):
                neuron[attr] = str(val)
        (mon_neurons if mon_idx >= 0 else other_neurons).append(neuron)

    n_sample = max(0, sample_target - len(mon_neurons))
    if len(other_neurons) > n_sample:
        step = len(other_neurons) / n_sample
        sampled_other = [other_neurons[int(i * step)] for i in range(n_sample)]
    else:
        sampled_other = other_neurons
    for m in sampled_other:
        m["is_background"] = True
    for m in mon_neurons:
        m["is_background"] = False
    neurons_list = mon_neurons + sampled_other
    print(f"  Cached {len(neurons_list):,} neuron centroids "
          f"({len(mon_neurons):,} monitored + {len(sampled_other):,} background)")
    return {"status": "ready", "neurons": neurons_list}
