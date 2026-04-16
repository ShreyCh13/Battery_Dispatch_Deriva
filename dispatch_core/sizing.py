"""
sizing.py
---------
Capacity sweep helpers for natural gas right-sizing recommendations.
"""

from __future__ import annotations

from typing import Iterable
import pandas as pd

from .config import RunConfig
from .optimize import run_lp


def _copy_cfg_with_capacity(cfg: RunConfig, gas_pmax_mw: float) -> RunConfig:
    return cfg.model_copy(
        update={
            "gas_enabled": True,
            "gas_dispatchable": True,
            "gas_pmax_mw": float(gas_pmax_mw),
            "gas_pmin_mw": min(float(cfg.gas_pmin_mw), float(gas_pmax_mw)),
        }
    )


def run_gas_capacity_sweep(
    df: pd.DataFrame,
    cfg: RunConfig,
    capacities_mw: Iterable[float],
    *,
    firmness_target_pct: float | None = None,
    grid_allowed: bool = False,
) -> pd.DataFrame:
    """Run dispatch optimization for each candidate gas capacity."""
    rows: list[dict] = []
    for cap in capacities_mw:
        run_cfg = _copy_cfg_with_capacity(cfg, float(cap))
        res, mets = run_lp(
            df,
            run_cfg,
            mode="grid_on_max_revenue" if grid_allowed else "resilience",
            grid_allowed=grid_allowed,
            blend_lambda=1.0,
        )
        rows.append(
            {
                "gas_capacity_mw": float(cap),
                "firmness_pct": float(mets.get("firmness (%)", 0.0)),
                "merchant_revenue_cost_usd": float(mets.get("Merchant Revenue/Cost", 0.0)),
                "total_served_mwh": float(mets.get("total_served_mwh", 0.0)),
                "total_gen_mwh": float(mets.get("total_gen_mwh", 0.0)),
                "total_clip_mwh": float(mets.get("total_clip_mwh", 0.0)),
                "natgas_total_cost_usd": float(mets.get("natgas_total_cost_$", 0.0)),
                "natgas_generation_mwh": float(res.get("gas_gen", pd.Series(dtype=float)).sum(skipna=True)),
            }
        )

    out = pd.DataFrame(rows).sort_values("gas_capacity_mw").reset_index(drop=True)
    out["meets_target"] = False
    if firmness_target_pct is not None and len(out) > 0:
        out["meets_target"] = out["firmness_pct"] >= float(firmness_target_pct)
    return out


def recommend_gas_capacity(
    sweep_df: pd.DataFrame,
    *,
    firmness_target_pct: float | None = None,
) -> dict:
    """Pick recommendation and an economic knee candidate from sweep results."""
    if sweep_df.empty:
        return {"recommended_capacity_mw": 0.0, "knee_capacity_mw": 0.0, "reason": "No sweep rows"}

    df = sweep_df.sort_values("gas_capacity_mw").reset_index(drop=True)
    recommended = df.iloc[0]
    reason = "Lowest tested capacity."
    if firmness_target_pct is not None:
        meets = df[df["firmness_pct"] >= float(firmness_target_pct)]
        if not meets.empty:
            recommended = meets.iloc[0]
            reason = f"Minimum capacity meeting firmness target ({firmness_target_pct:.2f}%)."

    # Knee by best marginal merchant improvement per MW.
    d_rev = df["merchant_revenue_cost_usd"].diff().fillna(0.0)
    d_cap = df["gas_capacity_mw"].diff().replace(0, pd.NA).fillna(1.0)
    slope = (d_rev / d_cap).fillna(0.0).abs()
    knee_idx = int(slope.idxmax()) if len(slope) else 0
    knee_row = df.iloc[knee_idx]

    return {
        "recommended_capacity_mw": float(recommended["gas_capacity_mw"]),
        "recommended_firmness_pct": float(recommended["firmness_pct"]),
        "recommended_merchant_revenue_cost_usd": float(recommended["merchant_revenue_cost_usd"]),
        "knee_capacity_mw": float(knee_row["gas_capacity_mw"]),
        "knee_firmness_pct": float(knee_row["firmness_pct"]),
        "knee_merchant_revenue_cost_usd": float(knee_row["merchant_revenue_cost_usd"]),
        "reason": reason,
    }
