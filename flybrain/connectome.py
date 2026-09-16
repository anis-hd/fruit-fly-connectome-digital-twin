"""Connectome loading pipeline: feather files -> signed sparse graph.

Pure functions (no module globals) shared by the offline script and the
live server. Anatomical body wiring lives in :mod:`flybrain.pools`.
"""
import numpy as np
import pandas as pd
import torch

from . import config as C
from .datautils import pick_col


def load_dataframes(ann_path: str, nt_path: str, conn_path: str):
    print("Loading feather files...")
    ann = pd.read_feather(ann_path)
    nt = pd.read_feather(nt_path)
    conn = pd.read_feather(conn_path)
    print(f"  annotations: {ann.shape}, "
          f"neurotransmitters: {nt.shape}, connectome: {conn.shape}")
    return ann, nt, conn


def resolve_id_columns(ann, nt, conn):
    id_ann = pick_col(ann, ["bodyId", "body_id", "body", "id"],
                      name="annotation id")
    id_nt = pick_col(nt, ["bodyId", "body_id", "body", "id"], name="nt id")
    c_pre = pick_col(conn, ["body_pre", "pre_body_id", "pre"],
                     contains="pre", name="pre")
    c_post = pick_col(conn, ["body_post", "post_body_id", "post"],
                      contains="post", name="post")
    c_w = pick_col(conn, ["weight", "synapses", "count"],
                   contains="weight", name="weight")
    return id_ann, id_nt, c_pre, c_post, c_w


def filter_edges(conn, c_pre, c_post, c_w, ann_ids, min_weight: int):
    pre = conn[c_pre].to_numpy()
    post = conn[c_post].to_numpy()
    w = conn[c_w].to_numpy().astype(np.float32)
    keep = (w >= min_weight) & (pre != post)
    keep &= np.isin(pre, ann_ids)
    keep &= np.isin(post, ann_ids)
    dropped = int((~keep).sum())
    pre, post, w = pre[keep], post[keep], w[keep]
    print(f"  edges kept: {len(pre):,} "
          f"(dropped {dropped:,} weak/self/unannotated)")
    return pre, post, w


def build_universe(pre, post):
    universe = np.unique(np.concatenate([pre, post]))
    n = len(universe)
    lut = pd.Series(np.arange(n), index=universe)
    pre_i = lut.reindex(pre).to_numpy().astype(np.int64)
    post_i = lut.reindex(post).to_numpy().astype(np.int64)
    print(f"  neurons: {n:,}")
    return universe, lut, pre_i, post_i


def build_nt_signs(nt, id_nt, universe) -> np.ndarray:
    """Per-neuron synaptic sign (+1/-1) from NT predictions."""
    str_nt_col = next(
        (c for c in ["consensus_nt", "predicted_nt", "celltype_predicted_nt"]
         if c in nt.columns),
        None,
    )
    prob_cols = {}
    for c in nt.columns:
        if c == id_nt or not pd.api.types.is_numeric_dtype(nt[c]):
            continue
        lc = c.lower()
        if any(k in lc for k in C.EXC_KEYS):
            prob_cols[c] = 1.0
        elif any(k in lc for k in C.INH_KEYS):
            prob_cols[c] = -1.0
        elif any(k in lc for k in C.MOD_KEYS):
            prob_cols[c] = C.MOD_SIGN

    if str_nt_col is not None:
        print(f"  Using NT classification column: '{str_nt_col}'")

        def nt_label_to_sign(val):
            if not isinstance(val, str):
                return 1.0
            vl = val.lower()
            if any(k in vl for k in C.INH_KEYS):
                return -1.0
            if any(k in vl for k in C.EXC_KEYS):
                return 1.0
            if any(k in vl for k in C.MOD_KEYS):
                return C.MOD_SIGN
            return 1.0

        if str_nt_col == "consensus_nt" and "predicted_nt" in nt.columns:
            series = nt["consensus_nt"].copy()
            unclear_mask = series.isna() | (series.str.lower() == "unclear")
            series[unclear_mask] = nt.loc[unclear_mask, "predicted_nt"]
        else:
            series = nt[str_nt_col]
        signs = series.map(nt_label_to_sign)
        sign_map = pd.Series(signs.to_numpy(dtype=np.float32),
                             index=nt[id_nt].to_numpy())
    elif prob_cols:
        print(f"  Using numeric NT probability columns: {list(prob_cols.keys())}")
        dominant = nt[list(prob_cols.keys())].idxmax(axis=1)
        sign_map = pd.Series([prob_cols[c] for c in dominant],
                             index=nt[id_nt].to_numpy(), dtype=np.float32)
    else:
        raise KeyError(
            f"No neurotransmitter columns recognized. "
            f"Available: {list(nt.columns)}")

    mapped = sign_map.reindex(universe)
    n_known = int(mapped.notna().sum())
    sign = mapped.fillna(1.0).to_numpy().astype(np.float32)
    print(f"  NT coverage: {n_known / len(universe):.1%} | "
          f"inhibitory neurons: {float((sign < 0).mean()):.1%}")
    return sign


def select_sensory_motor(ann, id_ann, lut, universe, n: int):
    """Keyword sensory/motor sets with in/out-degree fallbacks."""
    sens_pattern = "|".join(C.SENS_KW)
    mot_pattern = "|".join(C.MOT_KW)
    sens_mask = np.zeros(len(ann), dtype=bool)
    mot_mask = np.zeros(len(ann), dtype=bool)

    text_cols = [c for c in C.CANDIDATE_TEXT_COLS if c in ann.columns]
    if not text_cols:
        text_cols = [
            c for c in ann.select_dtypes(include=["object", "string"]).columns
            if c not in ["somaLocation", "tosomaLocation"]
        ]
    for c in text_cols:
        s = ann[c].fillna("").astype(str).str.lower()
        sens_mask |= s.str.contains(sens_pattern, regex=True, na=False).to_numpy()
        mot_mask |= s.str.contains(mot_pattern, regex=True, na=False).to_numpy()

    sens_ids = ann.loc[sens_mask, id_ann].to_numpy()
    mot_ids = ann.loc[mot_mask, id_ann].to_numpy()
    sens_idx = np.sort(lut.reindex(
        np.intersect1d(sens_ids, universe)).dropna().to_numpy().astype(np.int64))
    mot_idx = np.sort(lut.reindex(
        np.intersect1d(mot_ids, universe)).dropna().to_numpy().astype(np.int64))
    return sens_idx, mot_idx


def apply_degree_fallbacks(sens_idx, mot_idx, pre_i, post_i, n: int):
    in_deg = np.zeros(n)
    np.add.at(in_deg, post_i, 1)
    out_deg = np.zeros(n)
    np.add.at(out_deg, pre_i, 1)
    if len(sens_idx) < 10:
        sens_idx = np.where(in_deg == 0)[0]
        print("  [warn] sensory keywords failed -> using in-degree==0")
    if len(mot_idx) < 10:
        mot_idx = np.where(out_deg == 0)[0]
        print("  [warn] motor keywords failed -> using out-degree==0")
    print(f"  sensory neurons: {len(sens_idx):,} | motor neurons: {len(mot_idx):,}")
    return sens_idx, mot_idx


def build_signed_sparse_W(pre_i, post_i, w, sign, n: int,
                          norm_mode: str, gain: float, w_syn: float,
                          device: str):
    vals = sign[pre_i] * w
    if norm_mode == "input":
        in_sum = np.zeros(n, dtype=np.float32)
        np.add.at(in_sum, post_i, np.abs(vals))
        vals = vals * (gain / np.clip(in_sum[post_i], 1e-9, None))
    else:
        vals = vals * w_syn
    W = torch.sparse_coo_tensor(
        torch.stack([torch.from_numpy(post_i), torch.from_numpy(pre_i)]),
        torch.from_numpy(vals.astype(np.float32)), (n, n)).coalesce().to(device)
    try:
        W = W.to_sparse_csr()
    except Exception:
        pass
    nnz = int(W._nnz()) if hasattr(W, "_nnz") else int(W.values().numel())
    print(f"  sparse W: {n:,} x {n:,}, nnz={nnz:,} on {device}")
    return W
