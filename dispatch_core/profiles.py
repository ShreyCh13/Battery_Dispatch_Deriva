"""
profiles.py
-----------
Defines synthetic load profile generation for different load types (24-7, 16-7, random, etc.) used in battery dispatch simulations.
"""

import numpy as np
import pandas as pd
from .config import RunConfig

def generate(idx: pd.DatetimeIndex, cfg: RunConfig) -> pd.Series:
    """
    Generate a synthetic load profile based on the specified type in the configuration.

    Args:
        idx (pd.DatetimeIndex): Time index for the profile.
        cfg (RunConfig): Configuration object specifying load parameters and type.

    Returns:
        pd.Series: Load profile indexed by idx.

    Raises:
        ValueError: If an unknown load_type is specified.
    """
    base, m = cfg.load_mw, cfg.load_type
    if m == "24-7":
        # Constant load, all hours
        return pd.Series(base, index=idx)
    if m == "16-7":
        # Load only from 7:00 to 22:59 each day
        return pd.Series([base if 7 <= t.hour < 23 else 0 for t in idx], index=idx)
    if m == "random":
        # Random load (normal distribution), all hours
        vals = np.random.normal(base, cfg.load_std, len(idx))
        return pd.Series(np.clip(vals, 0, None), index=idx)
    if m == "random_16-7":
        # Random load (normal distribution), but only 7:00 to 22:59 each day
        vals = np.random.normal(base, cfg.load_std, len(idx))
        return pd.Series([vals[i] if 7 <= t.hour < 23 else 0
                          for i, t in enumerate(idx)], index=idx)
    raise ValueError(f"[profiles] Unknown load_type: {m}")
