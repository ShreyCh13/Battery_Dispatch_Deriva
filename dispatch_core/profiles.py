import numpy as np, pandas as pd
from .config import RunConfig

def generate(idx: pd.DatetimeIndex, cfg: RunConfig) -> pd.Series:
    base, m = cfg.load_mw, cfg.load_type
    if m == "24-7":
        return pd.Series(base, index=idx)
    if m == "16-7":
        return pd.Series([base if 7 <= t.hour < 23 else 0 for t in idx], index=idx)
    if m == "random":
        vals = np.random.normal(base, cfg.load_std, len(idx))
        return pd.Series(np.clip(vals, 0, None), index=idx)
    if m == "random_16-7":
        vals = np.random.normal(base, cfg.load_std, len(idx))
        return pd.Series([vals[i] if 7 <= t.hour < 23 else 0
                          for i, t in enumerate(idx)], index=idx)
    raise ValueError(f"[profiles] Unknown load_type: {m}")
