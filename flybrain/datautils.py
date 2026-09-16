"""Small dataframe helpers shared by offline + server loaders."""
import pandas as pd


def pick_col(df: pd.DataFrame, candidates, contains=None, name="") -> str:
    """Resolve a column name across schema variants (exact, then contains, then ci)."""
    for c in candidates:
        if c in df.columns:
            return c
    if contains:
        for c in df.columns:
            if contains.lower() in c.lower():
                return c
    cand_lower = {c.lower(): c for c in candidates}
    for c in df.columns:
        if c.lower() in cand_lower:
            return c
    raise KeyError(f"Cannot find {name} column. Available: {list(df.columns)}")
