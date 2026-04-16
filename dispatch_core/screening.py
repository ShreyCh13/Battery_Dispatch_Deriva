"""
screening.py
------------
Multi-resource capacity screening: enumerate combinations of solar / wind /
BESS power / BESS duration / gas capacity, run cost-optimal dispatch on each,
score by reliability + total annualised cost, and return the recommended
configuration plus a Pareto frontier.

Design choices:
- **Structured sweep, not joint LP.** Transparent to the user; trivially
  parallelisable later; produces an explicit trade-off surface.
- **Renewable scaling.** CSV ``Solar (MW)`` / ``Wind (MW)`` columns are
  interpreted as absolute MW from a *reference* installation. Auto-detect
  reference MW from each column's max value. Each candidate scales the
  reference profile by ``candidate_mw / reference_mw``. Renewable shape is
  preserved; only magnitude changes.
- **Gas UC relaxed during sweep** for tractability; the recommended config is
  re-validated with full MILP afterwards via ``validate_with_full_uc``.
- **Grid mode.** Defaults to grid-off (firmness focus). When ``grid_allowed=True``
  the cost objective also subtracts merchant revenue, so the screener can
  evaluate hybrids that sell to the grid.
- **Fallback behaviour.** If no candidate meets the reliability tier, return
  the most reliable one tested with a warning. Never raise infeasible.

This module has zero Streamlit imports; it is callable from the UI, a
notebook, or the CLI (``python -m dispatch_core.screen``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from .config import RunConfig
from .economics import EconAssumptions, annualized_cost, HOURS_PER_YEAR
from .optimize import run_lp


# Hard cap to prevent runaway sweeps from a misclick.
SWEEP_HARD_CAP = 1000
SWEEP_WARN_THRESHOLD = 300


@dataclass
class ResourceRange:
    """Specification for one resource's sweep range.

    Modes:
      - ``pinned``: single value (passed in ``values``).
      - ``multipliers``: ``values`` are multipliers of a reference MW (renewables only).
      - ``absolute``: ``values`` are MW values directly.
    """

    name: Literal["solar", "wind", "bess_power", "bess_duration", "gas"]
    mode: Literal["pinned", "multipliers", "absolute"]
    values: list[float]
    reference_mw: Optional[float] = None  # renewables only; required for multipliers

    def resolved_mw(self) -> list[float]:
        """Return the candidate MW (or hours, for bess_duration) values."""
        if self.mode == "multipliers":
            if self.reference_mw is None or self.reference_mw <= 0:
                # Without a reference, multipliers are meaningless; treat as pinned-zero.
                return [0.0]
            return [float(m) * float(self.reference_mw) for m in self.values]
        return [float(v) for v in self.values]


@dataclass
class ScreeningRanges:
    """Bundle of ResourceRange specs covering all sweepable resources.

    Default behaviour: pin everything at the current RunConfig values and the
    CSV reference MW. Override per resource in the UI / scenario.
    """

    solar: ResourceRange
    wind: ResourceRange
    bess_power: ResourceRange
    bess_duration: ResourceRange
    gas: ResourceRange

    def grid_size(self) -> int:
        n = 1
        for r in (self.solar, self.wind, self.bess_power, self.bess_duration, self.gas):
            n *= max(1, len(r.resolved_mw()))
        return n


@dataclass
class ScreeningResult:
    """Container for the full screening output.

    - ``sweep_df``: every config tested with metrics.
    - ``ranked_df``: scored + Pareto-flagged + reliability-flagged subset.
    - ``recommended``: dict for the chosen config (or warning details).
    - ``validation``: optional full-MILP re-run metrics for the recommended config.
    - ``period_weights``: list of weights for representative periods (V2 hook;
      currently always ``[1.0]`` for the single-period case).
    """

    sweep_df: pd.DataFrame
    ranked_df: pd.DataFrame
    recommended: dict
    validation: Optional[dict] = None
    period_weights: list[float] = field(default_factory=lambda: [1.0])


def detect_reference_mw(df: pd.DataFrame, col: str) -> float:
    """Infer the reference installed MW for a renewable column from its max.

    Heuristic: the maximum hourly output of a real installation is roughly its
    nameplate. We round up to a clean number for display (1 MW resolution).
    """
    if col not in df.columns:
        return 0.0
    series = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    if series.empty:
        return 0.0
    peak = float(series.max())
    if peak <= 0:
        return 0.0
    # Round up to nearest 1 MW for clean display.
    return float(math.ceil(peak))


def scale_renewables(df: pd.DataFrame, *, solar_mw: float, wind_mw: float,
                     ref_solar_mw: float, ref_wind_mw: float) -> pd.DataFrame:
    """Return a copy of df with Solar/Wind columns scaled to the candidate sizes.

    ``solar_mw / ref_solar_mw`` is the multiplier; same for wind. If reference
    is zero (no profile) the column is set to zero. Shape is preserved exactly.
    """
    out = df.copy()
    if "Solar (MW)" in out.columns:
        if ref_solar_mw > 0:
            out["Solar (MW)"] = pd.to_numeric(out["Solar (MW)"], errors="coerce").fillna(0.0) * (solar_mw / ref_solar_mw)
        else:
            out["Solar (MW)"] = 0.0
    if "Wind (MW)" in out.columns:
        if ref_wind_mw > 0:
            out["Wind (MW)"] = pd.to_numeric(out["Wind (MW)"], errors="coerce").fillna(0.0) * (wind_mw / ref_wind_mw)
        else:
            out["Wind (MW)"] = 0.0
    return out


def build_config_grid(ranges: ScreeningRanges) -> list[dict]:
    """Return the Cartesian product of resource sweep values as a list of dicts.

    Raises a clear error if the grid exceeds SWEEP_HARD_CAP.
    """
    candidates = {
        "solar_mw": ranges.solar.resolved_mw(),
        "wind_mw": ranges.wind.resolved_mw(),
        "bess_power_mw": ranges.bess_power.resolved_mw(),
        "bess_duration_h": ranges.bess_duration.resolved_mw(),
        "gas_mw": ranges.gas.resolved_mw(),
    }
    n = 1
    for v in candidates.values():
        n *= max(1, len(v))
    if n > SWEEP_HARD_CAP:
        raise ValueError(
            f"Sweep grid has {n} configurations, exceeding the hard cap of {SWEEP_HARD_CAP}. "
            "Reduce the sweep ranges (fewer values per resource) and try again."
        )
    grid = []
    for combo in itertools.product(*candidates.values()):
        grid.append(dict(zip(candidates.keys(), combo)))
    return grid


def _config_for_candidate(base_cfg: RunConfig, c: dict) -> RunConfig:
    """Return a RunConfig for one candidate, with BESS / gas sizes substituted.

    Renewable sizes are applied to the dataframe (scale_renewables); the
    RunConfig fields for renewables remain unchanged because the LP reads
    them from the dataframe columns.
    """
    bess_p = float(c.get("bess_power_mw", base_cfg.battery_power_mw))
    bess_d = float(c.get("bess_duration_h", base_cfg.battery_duration_h))
    gas_mw = float(c.get("gas_mw", base_cfg.gas_pmax_mw))
    update = {
        "battery_power_mw": bess_p,
        "battery_duration_h": bess_d,
        "gas_pmax_mw": gas_mw,
        # Always enable + dispatchable for the screener so candidates with gas
        # actually use it. Candidates with gas_mw=0 get zero capacity and the
        # solver will simply not run gas.
        "gas_enabled": gas_mw > 0,
        "gas_dispatchable": gas_mw > 0,
        "gas_pmin_mw": min(float(base_cfg.gas_pmin_mw), gas_mw),
    }
    return base_cfg.model_copy(update=update)


def run_config_sweep(
    df: pd.DataFrame,
    base_cfg: RunConfig,
    ranges: ScreeningRanges,
    econ: EconAssumptions,
    *,
    grid_allowed: bool = False,
    relax_uc: bool = True,
    progress_cb: Optional[Callable[[int, int, dict], None]] = None,
) -> pd.DataFrame:
    """Run cost-optimal dispatch for every candidate in the grid.

    Returns one row per candidate with metrics + annualised cost components.

    Args:
        df: Reference time-series with Solar/Wind/Load/Price columns. Renewable
            columns are interpreted as absolute MW from a reference installation;
            ``ranges.solar.reference_mw`` / ``ranges.wind.reference_mw`` provide
            the scaling base.
        base_cfg: Baseline RunConfig (load, RTE, POI, gas cost mode, VOLL, etc.).
            Per-candidate sizes are substituted before each run.
        ranges: Sweep specification.
        econ: Economic assumptions for annualised capex + O&M.
        grid_allowed: If True, allow grid imports/exports during dispatch and
            include merchant revenue in the cost objective.
        relax_uc: If True (default for screening), relax gas commitment binaries
            to LP for tractability. The recommended config can be re-validated
            with full MILP via ``validate_with_full_uc``.
        progress_cb: Optional callback ``(idx, total, candidate_dict)`` invoked
            before each run; useful for UI progress bars.

    Returns:
        DataFrame with columns:
          - candidate sizes (solar_mw, wind_mw, bess_power_mw, bess_duration_h,
            bess_energy_mwh, gas_mw)
          - dispatch metrics (firmness_pct, total_served_mwh, unserved_mwh,
            total_natgas_mwh, co2_tons, total_op_cost_usd_horizon,
            cost_breakdown_*)
          - economics (annualized_*_usd, annualized_total_usd_yr,
            opex_usd_yr, total_annual_cost_usd, lcoe_usd_per_mwh)
    """
    grid = build_config_grid(ranges)
    if not grid:
        return pd.DataFrame()

    ref_solar = float(ranges.solar.reference_mw or 0.0)
    ref_wind = float(ranges.wind.reference_mw or 0.0)
    rows: list[dict] = []

    total = len(grid)
    for idx, candidate in enumerate(grid):
        if progress_cb is not None:
            try:
                progress_cb(idx, total, candidate)
            except Exception:
                pass

        df_scaled = scale_renewables(
            df,
            solar_mw=float(candidate["solar_mw"]),
            wind_mw=float(candidate["wind_mw"]),
            ref_solar_mw=ref_solar,
            ref_wind_mw=ref_wind,
        )
        run_cfg = _config_for_candidate(base_cfg, candidate)

        try:
            res, mets = run_lp(
                df_scaled,
                run_cfg,
                mode="cost_min_gridoff",
                grid_allowed=grid_allowed,
                relax_uc=relax_uc,
            )
            ok = True
            err = ""
        except Exception as e:
            res, mets = None, {}
            ok = False
            err = str(e)

        # Annualised capex + fixed O&M
        ann = annualized_cost(
            run_cfg,
            econ,
            solar_mw=float(candidate["solar_mw"]),
            wind_mw=float(candidate["wind_mw"]),
        )

        horizon_h = float(mets.get("horizon_hours", len(df_scaled) * float(getattr(run_cfg, "dt_hours", 1.0))))
        op_cost_horizon = float(mets.get("total_operating_cost_usd", 0.0))
        # Annualise opex: scale horizon cost up to 8760 h.
        opex_usd_yr = (op_cost_horizon / horizon_h * HOURS_PER_YEAR) if horizon_h > 0 else 0.0
        total_annual = float(ann["annualized_total_usd_yr"]) + opex_usd_yr
        served_mwh_year = float(mets.get("total_served_mwh", 0.0)) / horizon_h * HOURS_PER_YEAR if horizon_h > 0 else 0.0
        lcoe = (total_annual / served_mwh_year) if served_mwh_year > 0 else float("nan")
        cb = mets.get("cost_breakdown", {}) or {}

        bess_energy_mwh = float(candidate["bess_power_mw"]) * float(candidate["bess_duration_h"])
        rows.append(
            {
                "ok": ok,
                "error": err,
                "solar_mw": float(candidate["solar_mw"]),
                "wind_mw": float(candidate["wind_mw"]),
                "bess_power_mw": float(candidate["bess_power_mw"]),
                "bess_duration_h": float(candidate["bess_duration_h"]),
                "bess_energy_mwh": bess_energy_mwh,
                "gas_mw": float(candidate["gas_mw"]),
                "firmness_pct": float(mets.get("firmness (%)", 0.0)),
                "total_served_mwh": float(mets.get("total_served_mwh", 0.0)),
                "unserved_mwh": float(mets.get("unserved_energy_mwh", 0.0)),
                "total_natgas_mwh": float(mets.get("total_natgas_mwh", 0.0)),
                "co2_tons": float(mets.get("co2_tons", 0.0)),
                "merchant_revenue_usd": float(mets.get("Merchant Revenue/Cost", 0.0)),
                "total_op_cost_usd_horizon": op_cost_horizon,
                "cost_unserved_usd": float(cb.get("unserved_voll", 0.0)),
                "cost_gas_var_usd": float(cb.get("gas_variable", 0.0)),
                "cost_gas_no_load_usd": float(cb.get("gas_no_load", 0.0)),
                "cost_gas_startup_usd": float(cb.get("gas_startup", 0.0)),
                "cost_bess_deg_usd": float(cb.get("bess_degradation", 0.0)),
                "cost_carbon_usd": float(cb.get("carbon", 0.0)),
                "annualized_solar_usd": ann["annualized_solar_usd"],
                "annualized_wind_usd": ann["annualized_wind_usd"],
                "annualized_bess_power_usd": ann["annualized_bess_power_usd"],
                "annualized_bess_energy_usd": ann["annualized_bess_energy_usd"],
                "annualized_gas_usd": ann["annualized_gas_usd"],
                "fixed_om_total_usd_yr": (
                    ann["fixed_om_solar_usd_yr"] + ann["fixed_om_wind_usd_yr"]
                    + ann["fixed_om_bess_usd_yr"] + ann["fixed_om_gas_usd_yr"]
                ),
                "annualized_total_usd_yr": ann["annualized_total_usd_yr"],
                "opex_usd_yr": round(opex_usd_yr, 2),
                "total_annual_cost_usd": round(total_annual, 2),
                "lcoe_usd_per_mwh": round(lcoe, 2) if not (lcoe != lcoe) else float("nan"),  # NaN-safe
                "horizon_hours": horizon_h,
            }
        )

    return pd.DataFrame(rows)


def score_and_rank(sweep_df: pd.DataFrame, *, reliability_target_pct: float = 99.0) -> pd.DataFrame:
    """Add ``meets_reliability`` and ``is_pareto`` flags; sort by total annual cost.

    Pareto frontier here is in the (firmness, cost) plane: a config is on the
    frontier if no other tested config has both higher firmness AND lower cost.
    """
    if sweep_df.empty:
        return sweep_df.assign(meets_reliability=False, is_pareto=False)

    df = sweep_df.copy()
    df["meets_reliability"] = df["firmness_pct"] >= float(reliability_target_pct)

    # Pareto frontier: maximize firmness, minimize cost
    df = df.sort_values(["firmness_pct", "total_annual_cost_usd"], ascending=[False, True]).reset_index(drop=True)
    is_pareto = []
    best_cost = float("inf")
    for _, row in df.iterrows():
        if row["total_annual_cost_usd"] < best_cost - 1e-6:
            is_pareto.append(True)
            best_cost = float(row["total_annual_cost_usd"])
        else:
            is_pareto.append(False)
    df["is_pareto"] = is_pareto

    # Final ordering: cost-ascending among reliable, then most-reliable among unreliable
    reliable = df[df["meets_reliability"]].sort_values("total_annual_cost_usd")
    unreliable = df[~df["meets_reliability"]].sort_values("firmness_pct", ascending=False)
    return pd.concat([reliable, unreliable], ignore_index=True)


def recommend_config(ranked_df: pd.DataFrame, *, reliability_target_pct: float = 99.0) -> dict:
    """Return the recommended configuration plus rationale.

    Strategy:
      - Cheapest config that meets the reliability target.
      - If none meets it, return the most reliable config tested with a warning.
    """
    if ranked_df.empty:
        return {
            "found": False,
            "message": "No configurations were tested.",
            "config": None,
        }

    reliable = ranked_df[ranked_df["meets_reliability"]]
    if not reliable.empty:
        best = reliable.sort_values("total_annual_cost_usd").iloc[0]
        return {
            "found": True,
            "meets_reliability": True,
            "message": (
                f"Cheapest configuration meeting {reliability_target_pct:.1f}% reliability: "
                f"${float(best['total_annual_cost_usd']):,.0f}/yr at {float(best['firmness_pct']):.2f}% firmness."
            ),
            "config": best.to_dict(),
        }

    # Fallback: best reliability available
    best = ranked_df.sort_values(["firmness_pct", "total_annual_cost_usd"], ascending=[False, True]).iloc[0]
    return {
        "found": True,
        "meets_reliability": False,
        "message": (
            f"No tested configuration meets {reliability_target_pct:.1f}% reliability. "
            f"Highest available firmness: {float(best['firmness_pct']):.2f}% at "
            f"${float(best['total_annual_cost_usd']):,.0f}/yr. Consider expanding the sweep ranges "
            "(more gas / longer-duration BESS / larger renewables)."
        ),
        "config": best.to_dict(),
    }


def validate_with_full_uc(
    df: pd.DataFrame,
    base_cfg: RunConfig,
    recommended: dict,
    ranges: ScreeningRanges,
    *,
    grid_allowed: bool = False,
) -> dict:
    """Re-run the recommended config with full MILP gas commitment for validation.

    Screening uses LP-relaxed UC for speed. This re-runs once with binaries
    enforced and returns the hardened metrics.
    """
    if not recommended.get("found") or recommended.get("config") is None:
        return {}
    cfg_dict = recommended["config"]
    candidate = {
        "solar_mw": float(cfg_dict["solar_mw"]),
        "wind_mw": float(cfg_dict["wind_mw"]),
        "bess_power_mw": float(cfg_dict["bess_power_mw"]),
        "bess_duration_h": float(cfg_dict["bess_duration_h"]),
        "gas_mw": float(cfg_dict["gas_mw"]),
    }
    df_scaled = scale_renewables(
        df,
        solar_mw=candidate["solar_mw"],
        wind_mw=candidate["wind_mw"],
        ref_solar_mw=float(ranges.solar.reference_mw or 0.0),
        ref_wind_mw=float(ranges.wind.reference_mw or 0.0),
    )
    run_cfg = _config_for_candidate(base_cfg, candidate)
    try:
        res, mets = run_lp(
            df_scaled,
            run_cfg,
            mode="cost_min_gridoff",
            grid_allowed=grid_allowed,
            relax_uc=False,
        )
        return {
            "ok": True,
            "metrics": mets,
            "dispatch_df": res,
            "run_cfg": run_cfg,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def screen(
    df: pd.DataFrame,
    base_cfg: RunConfig,
    ranges: ScreeningRanges,
    econ: EconAssumptions,
    *,
    reliability_target_pct: float = 99.0,
    grid_allowed: bool = False,
    relax_uc: bool = True,
    validate_recommended: bool = True,
    progress_cb: Optional[Callable[[int, int, dict], None]] = None,
) -> ScreeningResult:
    """End-to-end screening: sweep -> rank -> recommend -> optional MILP validation.

    This is the single entrypoint the UI and the CLI both call.
    """
    sweep_df = run_config_sweep(
        df, base_cfg, ranges, econ,
        grid_allowed=grid_allowed,
        relax_uc=relax_uc,
        progress_cb=progress_cb,
    )
    ranked_df = score_and_rank(sweep_df, reliability_target_pct=reliability_target_pct)
    rec = recommend_config(ranked_df, reliability_target_pct=reliability_target_pct)

    validation = None
    if validate_recommended and rec.get("found"):
        validation = validate_with_full_uc(df, base_cfg, rec, ranges, grid_allowed=grid_allowed)

    # period_weights: stub for V2 representative periods. Single period today.
    return ScreeningResult(
        sweep_df=sweep_df,
        ranked_df=ranked_df,
        recommended=rec,
        validation=validation,
        period_weights=[1.0],
    )
