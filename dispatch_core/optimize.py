"""
optimize.py
-----------
Defines the main optimization routines for battery dispatch, including linear programming (LP) models for various operational modes and trade-off analysis between resilience and revenue.

Functions:
    - run_lp: Main LP optimizer for battery dispatch (multiple modes)
    - tradeoff_analysis: Multi-slack trade-off analysis for resilience vs. revenue
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pulp
from typing import Optional
from .config   import RunConfig
from .profiles import generate

KWH_TO_MWH = 1000
BIG_M = 1_000_000


def _build_gas_cap_series(df: pd.DataFrame, cfg: RunConfig) -> pd.Series:
    """Return gas availability cap (MW) for each interval."""
    default_cap = float(cfg.gas_pmax_mw) if cfg.gas_pmax_mw > 0 else 0.0
    if cfg.gas_availability_col in df.columns:
        cap = pd.to_numeric(df[cfg.gas_availability_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if cfg.gas_pmax_mw > 0:
            return cap.clip(upper=float(cfg.gas_pmax_mw))
        return cap
    return pd.Series(default_cap, index=df.index, dtype=float)

def run_lp(
    df: pd.DataFrame,
    cfg: RunConfig,
    *,
    mode: str = "blend",        # blend | revenue | resilience | serve | grid_on_max_revenue
    blend_lambda: float = 0.9,
    grid_allowed: bool = True,
    min_served_mwh: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Linear program for battery dispatch optimization.

    Modes:
      - 'blend': maximize weighted sum of resilience and grid revenue
      - 'revenue': maximize grid revenue
      - 'resilience': maximize load served
      - 'serve': (not used)
      - 'grid_on_max_revenue': (grid ON only) serve 100% of load, maximize net revenue (grid export * price - grid import * price)

    Args:
        df (pd.DataFrame): Input time series data (must include generation, load, price columns).
        cfg (RunConfig): Run configuration object.
        mode (str): Optimization mode (see above).
        blend_lambda (float): Weight for blend mode (0–1).
        grid_allowed (bool): If False, disables grid import/export.

    Returns:
        Tuple[pd.DataFrame, dict]:
            - Dispatch DataFrame (with all time series results)
            - Metrics dictionary (summary statistics, sanity check results)

    Raises:
        AssertionError: If blend_lambda is out of bounds in blend mode.
        RuntimeError: If the LP solver fails to find an optimal solution.
    """
    if mode == "blend":
        assert 0 <= blend_lambda <= 1, "blend_lambda must be between 0 and 1."

    # Prepare input series
    wind = pd.to_numeric(df.get("Wind (MW)", 0.0), errors="coerce").fillna(0.0)
    solar = pd.to_numeric(df.get("Solar (MW)", 0.0), errors="coerce").fillna(0.0)
    natgas_profile = pd.to_numeric(df.get("NatGas (MW)", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    renewable_gen = wind + solar
    if "Load (MW)" in df.columns:
        load = pd.to_numeric(df["Load (MW)"], errors="coerce").fillna(0.0)
    else:
        load = generate(pd.DatetimeIndex(df.index), cfg)
    price = pd.to_numeric(df[cfg.market_price_col], errors="coerce").fillna(0.0)

    # Battery 1 parameters
    Pmax1 = cfg.battery_power_mw
    Emax1 = cfg.battery_energy_mwh
    eta1 = cfg.rte
    # Battery 2 parameters
    Pmax2 = cfg.battery2_power_mw
    Emax2 = cfg.battery2_energy_mwh
    eta2 = cfg.battery2_rte
    T, val = len(df), cfg.value_per_mwh
    POI = cfg.poi_limit_mw if grid_allowed else 0

    gas_dispatchable = bool(cfg.gas_enabled and cfg.gas_dispatchable)
    gas_var_cost = float(cfg.gas_variable_cost_usd_per_mwh)
    gas_cap_series = _build_gas_cap_series(df, cfg)

    # ── LP/MILP model setup ───────────────────────────────────────
    prob = pulp.LpProblem("BatteryDispatch", pulp.LpMaximize)
    # Battery 1 variables
    c1  = pulp.LpVariable.dicts("c1",  range(T), 0, Pmax1)  # Charge
    d1  = pulp.LpVariable.dicts("d1",  range(T), 0, Pmax1)  # Discharge
    s1  = pulp.LpVariable.dicts("s1",  range(T), 0, Emax1)  # State of charge
    # Battery 2 variables
    c2  = pulp.LpVariable.dicts("c2",  range(T), 0, Pmax2)
    d2  = pulp.LpVariable.dicts("d2",  range(T), 0, Pmax2)
    s2  = pulp.LpVariable.dicts("s2",  range(T), 0, Emax2)
    # System variables
    v  = pulp.LpVariable.dicts("v",  range(T), 0)           # Load served
    gi = pulp.LpVariable.dicts("gi", range(T), 0, POI)      # Grid import
    ge = pulp.LpVariable.dicts("ge", range(T), 0, POI)      # Grid export
    cl = pulp.LpVariable.dicts("cl", range(T), 0)           # Clipped energy
    gas_gen = pulp.LpVariable.dicts("gas_gen", range(T), 0)
    gas_on = {}
    gas_start = {}
    gas_stop = {}
    if gas_dispatchable:
        gas_on = pulp.LpVariable.dicts("gas_on", range(T), 0, 1, cat="Binary")
        gas_start = pulp.LpVariable.dicts("gas_start", range(T), 0, 1, cat="Binary")
        gas_stop = pulp.LpVariable.dicts("gas_stop", range(T), 0, 1, cat="Binary")

    # Objective function components
    grid_revenue = pulp.lpSum(price.iloc[t] * (ge[t] - gi[t]) for t in range(T))
    resilience    = pulp.lpSum(val*v[t] for t in range(T))
    clipped_revenue = pulp.lpSum(price.iloc[t] * cl[t] for t in range(T))  # potential merchant revenue from clipped energy
    gas_energy_cost = pulp.lpSum(gas_var_cost * gas_gen[t] for t in range(T))
    gas_start_cost = pulp.lpSum(float(cfg.gas_startup_cost_usd) * gas_start[t] for t in range(T)) if gas_dispatchable else 0.0
    total_gas_cost = gas_energy_cost + gas_start_cost
    LEX_WEIGHT = 1_000_000  # large weight to enforce lexicographic priority

    # Artificially high revenue for served load in 'resilience_first_blend' mode
    artificial_resilience_revenue = pulp.LpAffineExpression()
    if mode == "resilience_first_blend":
        artificial_price = float(np.max(price))
        artificial_resilience_revenue = pulp.lpSum(artificial_price * v[t] for t in range(T))

    # Set objective based on mode
    if mode == "blend":
        prob += blend_lambda * resilience + (1 - blend_lambda) * grid_revenue - total_gas_cost
    elif mode == "resilience":
        # Lexicographic objective: maximise resilience first, then merchant revenue from clipped energy
        prob += LEX_WEIGHT * resilience + clipped_revenue - total_gas_cost
    elif mode == "revenue":
        prob += grid_revenue - total_gas_cost
    elif mode == "resilience_first_blend":
        prob += blend_lambda * artificial_resilience_revenue + (1 - blend_lambda) * grid_revenue - total_gas_cost
    elif mode == "grid_on_max_revenue":
        # Enforce all load is served, maximize net grid revenue
        for t in range(T):
            prob += v[t] == load.iloc[t]
        prob += grid_revenue - total_gas_cost
    else:
        raise ValueError("mode must be blend/revenue/resilience/resilience_first_blend/grid_on_max_revenue")

    # ── Constraints ──────────────────────────────────────────────
    for t in range(T):
        # Natural gas bounds / commitment dynamics
        if gas_dispatchable:
            cap_t = float(gas_cap_series.iloc[t])
            pmax_t = min(float(cfg.gas_pmax_mw) if cfg.gas_pmax_mw > 0 else cap_t, cap_t)
            pmax_t = max(0.0, pmax_t)
            pmin_t = min(max(0.0, float(cfg.gas_pmin_mw)), pmax_t)
            prob += gas_gen[t] <= pmax_t * gas_on[t]
            prob += gas_gen[t] >= pmin_t * gas_on[t]
            if t == 0:
                prob += gas_on[t] == gas_start[t] - gas_stop[t]
            else:
                prob += gas_on[t] - gas_on[t - 1] == gas_start[t] - gas_stop[t]
                prob += gas_gen[t] - gas_gen[t - 1] <= float(cfg.gas_ramp_up_mw_per_h)
                prob += gas_gen[t - 1] - gas_gen[t] <= float(cfg.gas_ramp_down_mw_per_h)
        else:
            gas_fixed = natgas_profile.iloc[t]
            prob += gas_gen[t] == gas_fixed

        # System power balance
        prob += renewable_gen.iloc[t] + gas_gen[t] + d1[t] + d2[t] + gi[t] == v[t] + c1[t] + c2[t] + ge[t] + cl[t]
        prob += v[t] <= load.iloc[t]  # Cannot serve more than load
        # POI net flow constraint
        prob += ge[t] - gi[t] <= POI
        prob += ge[t] - gi[t] >= -POI
        # Battery 1 SOC
        if t == 0:
            prob += s1[t] == c1[t]*eta1 - d1[t]
            prob += s2[t] == c2[t]*eta2 - d2[t]
        else:
            prob += s1[t] == s1[t-1] + c1[t]*eta1 - d1[t]
            prob += s2[t] == s2[t-1] + c2[t]*eta2 - d2[t]
        # SOC bounds (redundant with variable bounds, but explicit)
        prob += s1[t] >= 0
        prob += s1[t] <= Emax1
        prob += s2[t] >= 0
        prob += s2[t] <= Emax2
        # Discharge cannot exceed available SOC
        if t == 0:
            prob += d1[t] <= 0.5 * Emax1
            prob += d2[t] <= 0.5 * Emax2
        else:
            prob += d1[t] <= s1[t-1]
            prob += d2[t] <= s2[t-1]

    if gas_dispatchable:
        min_up = int(max(0, cfg.gas_min_up_h))
        min_down = int(max(0, cfg.gas_min_down_h))
        for t in range(T):
            if min_up > 0:
                start_ix = max(0, t - min_up + 1)
                prob += pulp.lpSum(gas_start[k] for k in range(start_ix, t + 1)) <= gas_on[t]
            if min_down > 0:
                start_ix = max(0, t - min_down + 1)
                prob += pulp.lpSum(gas_stop[k] for k in range(start_ix, t + 1)) <= 1 - gas_on[t]

    if min_served_mwh is not None:
        prob += pulp.lpSum(v[t] for t in range(T)) >= float(min_served_mwh)

    # ── Solve LP/MILP ────────────────────────────────────────────
    solved_with_relaxation = False
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal" and gas_dispatchable:
        # Safe fallback: LP relaxation keeps app usable if MILP is hard/infeasible.
        prob.solve(pulp.PULP_CBC_CMD(msg=False, mip=False))
        solved_with_relaxation = pulp.LpStatus[prob.status] == "Optimal"
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver failed: {pulp.LpStatus[prob.status]}")

    # ── Collect results ────────────────────────────────────────
    def safe_value(x):
        try:
            v = pulp.value(x)
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan
    arr = lambda var: np.array([safe_value(var[t]) for t in range(T)])
    res = df.copy()
    gas_gen_arr = arr(gas_gen)
    res["generation"] = (renewable_gen.values + gas_gen_arr)
    res["load"]       = load.values
    # Only add 'price' if not already present, and remove any duplicate 'price' columns later
    if "price" not in res.columns:
        res["price"] = price.values
    # Add price in $/kWh for display purposes
    res["price_$/kWh"] = price.values / 1000
    # Battery 1
    res["charge1"]     = arr(c1)
    res["discharge1"]  = arr(d1)
    # Battery 2
    res["charge2"]     = arr(c2)
    res["discharge2"]  = arr(d2)
    # Aggregate
    res["charge"]      = res["charge1"] + res["charge2"]
    res["discharge"]   = res["discharge1"] + res["discharge2"]
    res["serve"]      = arr(v)
    res["grid_imp"]   = arr(gi)
    res["grid_exp"]   = arr(ge)
    res["clipped"]    = arr(cl)
    res["gas_gen"] = gas_gen_arr
    if gas_dispatchable:
        res["gas_on"] = arr(gas_on)
        res["gas_start"] = arr(gas_start)
        res["gas_stop"] = arr(gas_stop)
    else:
        res["gas_on"] = (gas_gen_arr > 1e-6).astype(float)
        starts = np.zeros(T, dtype=float)
        starts[0] = 1.0 if gas_gen_arr[0] > 1e-6 else 0.0
        if T > 1:
            starts[1:] = ((gas_gen_arr[1:] > 1e-6) & (gas_gen_arr[:-1] <= 1e-6)).astype(float)
        res["gas_start"] = starts
        res["gas_stop"] = 0.0
    per_step_cost = gas_gen_arr * gas_var_cost + res["gas_start"].to_numpy(dtype=float) * float(cfg.gas_startup_cost_usd)
    res["gas_cost_$"] = per_step_cost
    # SOC
    res["soc1"] = arr(s1)
    res["soc2"] = arr(s2)
    res["soc"] = res["soc1"] + res["soc2"]
    res["net_to_grid"] = res["grid_exp"] - res["grid_imp"]
    # Remove duplicate 'price' columns and unnamed/empty columns
    res = res.loc[:, ~res.columns.duplicated()]
    res = res.dropna(axis=1, how='all')
    res = res.loc[:, ~res.columns.str.contains('^Unnamed', na=False)]
    # Ensure all columns are numeric after LP extraction
    for col in res.columns:
        res[col] = pd.to_numeric(res[col], errors='coerce')

    # ── Metrics calculation ────────────────────────────────────
    total_load = float(res["load"].sum(skipna=True))
    served = float(res["serve"].sum(skipna=True))
    grid_exp = res["grid_exp"].to_numpy(dtype=float)
    grid_imp = res["grid_imp"].to_numpy(dtype=float)
    price_arr = res["price"].to_numpy(dtype=float)
    revenue_tot = ((grid_exp - grid_imp) * price_arr).sum()  # $ (for hourly data: MW * $/MWh * 1h = $)
    total_wind = float(wind.sum(skipna=True))
    total_solar = float(solar.sum(skipna=True))
    total_natgas = float(res["gas_gen"].sum(skipna=True))
    total_gen = total_wind + total_solar + total_natgas
    green_gen_over_load_pct = (total_gen / total_load * 100) if total_load > 0 else 0.0
    resilience_pct = round(served / total_load * 100, 2) if total_load > 0 else 0.0
    total_charge = float(res["charge"].sum(skipna=True)) if "charge" in res else 0.0
    total_clip = float(res["clipped"].sum(skipna=True)) if "clipped" in res else 0.0
    total_gen_mwh = float(res["generation"].sum(skipna=True)) if "generation" in res else 0.0
    
    capex = (
        cfg.capex_power_usd_per_kw  * cfg.battery_power_mw   * 1000 +
        cfg.capex_energy_usd_per_kwh* cfg.battery_energy_mwh * 1000 +
        cfg.capex_power_usd_per_kw  * cfg.battery2_power_mw  * 1000 +
        cfg.capex_energy_usd_per_kwh* cfg.battery2_energy_mwh* 1000
    )
    # Calculate clipped revenue (only valid for no-grid case, after results are collected)
    clipped_revenue = None
    if not grid_allowed and "clipped" in res and "price" in res:
        clipped_arr = res["clipped"].to_numpy(dtype=float)
        price_arr = res["price"].to_numpy(dtype=float)
        if len(clipped_arr) == len(price_arr):
            clipped_revenue = float((clipped_arr * price_arr).sum())  # $ (for hourly data: MW * $/MWh * 1h = $)
        else:
            clipped_revenue = None
    # Calculate number of cycles for each battery
    discharge1_sum = float(res["discharge1"].to_numpy(dtype=float).sum()) if "discharge1" in res else 0.0
    discharge2_sum = float(res["discharge2"].to_numpy(dtype=float).sum()) if "discharge2" in res else 0.0
    cycles1 = discharge1_sum / float(cfg.battery_energy_mwh) if cfg.battery_energy_mwh > 0 else 0.0
    cycles2 = discharge2_sum / float(cfg.battery2_energy_mwh) if cfg.battery2_energy_mwh > 0 else 0.0
    charge_arr = res["charge"].to_numpy(dtype=float) if "charge" in res else np.zeros(len(res))
    clipped_arr = res["clipped"].to_numpy(dtype=float) if "clipped" in res else np.zeros(len(res))
    gen_arr = res["generation"].to_numpy(dtype=float) if "generation" in res else np.zeros(len(res))
    grid_imp_arr = res["grid_imp"].to_numpy(dtype=float)
    grid_exp_arr = res["grid_exp"].to_numpy(dtype=float)
    gas_start_count = int(np.round(res["gas_start"].to_numpy(dtype=float).sum())) if "gas_start" in res else 0
    gas_cap_ref = float(cfg.gas_pmax_mw) if cfg.gas_pmax_mw > 0 else float(gas_cap_series.max()) if len(gas_cap_series) else 0.0
    gas_cf_pct = (100.0 * total_natgas / (gas_cap_ref * T)) if gas_cap_ref > 0 and T > 0 else 0.0
    mets = {
        "firmness (%)": round(resilience_pct, 2),
        "TotalGen/TotalLoad": round(float(green_gen_over_load_pct), 2),
        "Merchant Revenue/Cost": round(float(revenue_tot - per_step_cost.sum()), 2),
        "total_charge_mwh": round(float(charge_arr.sum()), 2),
        "total_clip_mwh": round(float(clipped_arr.sum()), 2),
        "total_gen_mwh": round(float(gen_arr.sum()), 2),
        "total_load_mwh": round(total_load, 2),
        "total_served_mwh": round(served, 2),
        "grid_imp_mwh":   round(float(grid_imp_arr.sum()), 2),
        "grid_exp_mwh":   round(float(grid_exp_arr.sum()), 2),
        "cycles_battery1": round(float(cycles1), 2),
        "cycles_battery2": round(float(cycles2), 2),
        "total_wind_mwh": round(total_wind, 2),
        "total_solar_mwh": round(total_solar, 2),
        "total_natgas_mwh": round(total_natgas, 2),
        "natgas_start_count": gas_start_count,
        "natgas_capacity_factor_%": round(float(gas_cf_pct), 2),
        "natgas_total_cost_$": round(float(per_step_cost.sum()), 2),
        "solver_relaxed_lp": bool(solved_with_relaxation),
    }
    if clipped_revenue is not None:
        mets["clipped_revenue_$"] = round(clipped_revenue, 2)
        mets["clipped_revenue_explanation"] = (
            "Clipped revenue estimates the potential market revenue if all clipped (spilled) renewable energy "
            "could have been sold at the market price. This is only valid in the no-grid case, and represents "
            "the value of energy that was generated but could not be used or exported."
        )
    mets["ppa_price_per_kwh"] = 0.0 # Removed ppa_price

    # --- Sanity checks ---
    sanity_errors = []
    # 1. SOC within bounds
    if (res["soc1"].min() < -1e-6) or (res["soc1"].max() > cfg.battery_energy_mwh + 1e-6):
        sanity_errors.append("Battery 1 SOC out of bounds")
    if (res["soc2"].min() < -1e-6) or (res["soc2"].max() > cfg.battery2_energy_mwh + 1e-6):
        sanity_errors.append("Battery 2 SOC out of bounds")
    # 2. max_resilience >= actual resilience (removed, now always based on green_gen_over_load_pct)
    # 3. clipped always >= 0
    if (res["clipped"].min() < -1e-6):
        sanity_errors.append("Clipped energy negative")
    # 4. grid_imp/grid_exp zero if grid is off
    if not grid_allowed and mode != "grid_on_max_revenue":
        if (res["grid_imp"].abs().max() > 1e-6) or (res["grid_exp"].abs().max() > 1e-6):
            sanity_errors.append("Grid import/export nonzero when grid is off")
    # 5. serve never exceeds cumulative generation (over period)
    if not grid_allowed and mode != "grid_on_max_revenue":
        if res["serve"].sum() > (wind.clip(lower=0).sum() + solar.clip(lower=0).sum() + res["gas_gen"].clip(lower=0).sum()) + 1e-6:
            sanity_errors.append("Total served exceeds total available generation")
    sanity_check_passed = len(sanity_errors) == 0
    mets["sanity_check_passed"] = sanity_check_passed
    mets["sanity_check_errors"] = sanity_errors
    return res, mets

def tradeoff_analysis(
    df: pd.DataFrame,
    cfg: RunConfig,
    slack_list: Optional[list[float]] = None
) -> tuple[pd.DataFrame, dict]:
    """
    Multi-slack trade-off analysis: for each slack, maximize revenue subject to serve >= (1-slack)*S_max.
    Used as an optional output in resilience (no grid) mode, not as a separate mode.

    Args:
        df (pd.DataFrame): Input time series data.
        cfg (RunConfig): Run configuration object.
        slack_list (list[float], optional): List of slack values (fractional, e.g. 0.01 for 1%).

    Returns:
        Tuple[pd.DataFrame, dict]:
            - DataFrame of trade-off results (resilience %, revenue, etc.)
            - Dictionary of dispatch DataFrames for each slack value

    Raises:
        RuntimeError: If the max resilience problem is infeasible.
    """
    if slack_list is None:
        slack_list = [0.00, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.09]

    # Phase 1: find max served energy using the same run_lp core.
    res_max, _ = run_lp(df, cfg, mode="resilience", blend_lambda=1.0, grid_allowed=False)
    max_served = float(res_max["serve"].sum(skipna=True))
    total_load = float(res_max["load"].sum(skipna=True))

    # Phase 2: apply served-energy floor and optimize secondary economics.
    results = []
    dispatch_dict: dict[float, pd.DataFrame] = {}
    for slack in slack_list:
        served_floor = (1.0 - float(slack)) * max_served
        res_slack, mets_slack = run_lp(
            df,
            cfg,
            mode="resilience",
            blend_lambda=1.0,
            grid_allowed=False,
            min_served_mwh=served_floor,
        )
        served = float(res_slack["serve"].sum(skipna=True))
        firmness_pct = (served / total_load * 100.0) if total_load > 0 else 0.0
        results.append(
            {
                "slack_%": int(float(slack) * 100),
                "firmness (%)": round(firmness_pct, 4),
                "served_MWh": round(served, 4),
                "Merchant Revenue/Cost": float(mets_slack.get("Merchant Revenue/Cost", 0.0)),
            }
        )
        dispatch_dict[float(slack)] = res_slack

    return pd.DataFrame(results), dispatch_dict
