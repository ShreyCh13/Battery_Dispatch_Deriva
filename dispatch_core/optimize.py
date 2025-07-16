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

def run_lp(
    df: pd.DataFrame,
    cfg: RunConfig,
    *,
    mode: str = "blend",        # blend | revenue | resilience | serve | grid_on_max_revenue
    blend_lambda: float = 0.9,
    grid_allowed: bool = True
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
    gen   = df["Wind (MW)"].fillna(0) + df["Solar (MW)"].fillna(0)
    if "Load (MW)" in df.columns:
        load = df["Load (MW)"]
    else:
        load = generate(pd.DatetimeIndex(df.index), cfg)
    price = df[cfg.market_price_col]

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

    # ── LP model setup ─────────────────────────────────────────────
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

    # Objective function components
    grid_revenue = pulp.lpSum(price[t]*(ge[t]-gi[t]) for t in range(T))
    resilience = pulp.lpSum(val*v[t] for t in range(T))

    # Artificially high revenue for served load in 'resilience_first_blend' mode
    artificial_resilience_revenue = pulp.LpAffineExpression()
    if mode == "resilience_first_blend":
        artificial_price = float(np.max(price))
        artificial_resilience_revenue = pulp.lpSum(artificial_price * v[t] for t in range(T))

    # Set objective based on mode
    if mode == "blend":
        prob += blend_lambda*resilience + (1-blend_lambda)*grid_revenue
    elif mode == "resilience":
        prob += resilience
    elif mode == "revenue":
        prob += grid_revenue
    elif mode == "resilience_first_blend":
        prob += blend_lambda*artificial_resilience_revenue + (1-blend_lambda)*grid_revenue
    elif mode == "grid_on_max_revenue":
        # Enforce all load is served, maximize net grid revenue
        for t in range(T):
            prob += v[t] == load.iloc[t]
        prob += grid_revenue
    else:
        raise ValueError("mode must be blend/revenue/resilience/resilience_first_blend/grid_on_max_revenue")

    # ── Constraints ──────────────────────────────────────────────
    for t in range(T):
        # System power balance
        prob += gen.iloc[t] + d1[t] + d2[t] + gi[t] == v[t] + c1[t] + c2[t] + ge[t] + cl[t]
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

    # ── Solve LP ────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
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
    res["generation"] = gen.values
    res["load"]       = load.values
    # Only add 'price' if not already present, and remove any duplicate 'price' columns later
    if "price" not in res.columns:
        res["price"] = price.values
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
    revenue_tot = ((grid_exp - grid_imp) * price_arr).sum() / 1000
    total_wind = float(df["Wind (MW)"].fillna(0).sum(skipna=True))
    total_solar = float(df["Solar (MW)"].fillna(0).sum(skipna=True))
    total_gen = total_wind + total_solar
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
            clipped_revenue = float((clipped_arr * price_arr).sum() / 1000)  # $/MWh to $
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
    mets = {
        "resilience_pct": round(resilience_pct, 2),
        "green_gen_over_load_pct": round(float(green_gen_over_load_pct), 2),
        "revenue_$":      round(float(revenue_tot), 2),
        "total_charge_mwh": round(float(charge_arr.sum()), 2),
        "total_clip_mwh": round(float(clipped_arr.sum()), 2),
        "total_gen_mwh": round(float(gen_arr.sum()), 2),
        "total_load_mwh": round(total_load, 2),
        "total_served_mwh": round(served, 2),
        "grid_imp_mwh":   round(float(grid_imp_arr.sum()), 2),
        "grid_exp_mwh":   round(float(grid_exp_arr.sum()), 2),
        "capex_$":        round(float(capex)),
        "cycles_battery1": round(float(cycles1), 2),
        "cycles_battery2": round(float(cycles2), 2),
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
        if res["serve"].sum() > (df["Wind (MW)"].clip(lower=0).sum() + df["Solar (MW)"].clip(lower=0).sum()) + 1e-6:
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
    # --- Phase 1: Maximize resilience ---
    gen = df["Wind (MW)"].fillna(0) + df["Solar (MW)"].fillna(0)
    load = generate(pd.DatetimeIndex(df.index), cfg)
    price = df[cfg.market_price_col]
    T = len(df)
    Pmax = cfg.battery_power_mw
    Emax = cfg.battery_energy_mwh
    eta = cfg.rte
    POI = cfg.poi_limit_mw if hasattr(cfg, 'poi_limit_mw') else None
    # Max resilience model
    prob1 = pulp.LpProblem("Max_Serve", pulp.LpMaximize)
    chg1 = pulp.LpVariable.dicts("chg1", range(T), 0, Pmax)
    dis1 = pulp.LpVariable.dicts("dis1", range(T), 0, Pmax)
    soc1 = pulp.LpVariable.dicts("soc1", range(T), 0, Emax)
    serve1 = pulp.LpVariable.dicts("serve1", range(T), 0)
    prob1 += pulp.lpSum(serve1[t] for t in range(T))
    for t in range(T):
        prob1 += serve1[t] + chg1[t] <= gen.iloc[t] + dis1[t]
        prob1 += serve1[t] <= load.iloc[t]
        if t == 0:
            prob1 += soc1[0] == chg1[0]*eta
            prob1 += dis1[0] <= 0
        else:
            prob1 += soc1[t] == soc1[t-1] + chg1[t]*eta - dis1[t]
            prob1 += dis1[t] <= soc1[t-1]
        if POI is not None:
            prob1 += gen.iloc[t] + dis1[t] - chg1[t] <= POI
    prob1.solve(pulp.PULP_CBC_CMD(msg=False))
    S_max_raw = pulp.value(prob1.objective)
    if S_max_raw is None or not isinstance(S_max_raw, (int, float)):
        raise RuntimeError("Max resilience problem infeasible: S_max is None or not a number")
    S_max = float(S_max_raw)
    # --- Phase 2: For each slack, maximize revenue with min serve constraint ---
    results = []
    dispatch_dict = {}
    load_arr = np.array(load, dtype=float)
    for slack in slack_list:
        prob2 = pulp.LpProblem(f"Rev_Slack_{int(slack*100)}", pulp.LpMaximize)
        chg2 = pulp.LpVariable.dicts("chg2", range(T), 0, Pmax)
        dis2 = pulp.LpVariable.dicts("dis2", range(T), 0, Pmax)
        soc2 = pulp.LpVariable.dicts("soc2", range(T), 0, Emax)
        serve2 = pulp.LpVariable.dicts("serve2", range(T), 0)
        prob2 += pulp.lpSum(price.iloc[t] * (gen.iloc[t] + dis2[t] - chg2[t]) for t in range(T))
        for t in range(T):
            prob2 += serve2[t] + chg2[t] <= gen.iloc[t] + dis2[t]
            prob2 += serve2[t] <= load.iloc[t]
            if t == 0:
                prob2 += soc2[0] == chg2[0]*eta
                prob2 += dis2[0] <= 0
            else:
                prob2 += soc2[t] == soc2[t-1] + chg2[t]*eta - dis2[t]
                prob2 += dis2[t] <= soc2[t-1]
            if POI is not None:
                prob2 += gen.iloc[t] + dis2[t] - chg2[t] <= POI
        prob2 += pulp.lpSum(serve2[t] for t in range(T)) >= (1 - slack) * S_max
        prob2.solve(pulp.PULP_CBC_CMD(msg=False))
        served = sum(pulp.value(serve2[t]) for t in range(T))
        total_load = float(load_arr.sum())
        revenue = sum(price.iloc[t] * (pulp.value(dis2[t]) - pulp.value(chg2[t]) + gen.iloc[t]) for t in range(T)) / 1000
        resilience_pct = served / total_load * 100.0 if total_load > 0 else 0.0
        results.append({'slack_%': int(slack*100), 'resilience_%': resilience_pct, 'served_MWh': served, 'revenue_$': revenue})
        # Save dispatch for this slack
        dispatch_df = pd.DataFrame({
            'serve': [pulp.value(serve2[t]) for t in range(T)],
            'chg': [pulp.value(chg2[t]) for t in range(T)],
            'dis': [pulp.value(dis2[t]) for t in range(T)],
            'soc': [pulp.value(soc2[t]) for t in range(T)],
            'gen': gen.values,
            'price': price.values,
            'load': load.values,
        }, index=df.index)
        dispatch_dict[slack] = dispatch_df
    results_df = pd.DataFrame(results)
    return results_df, dispatch_dict
