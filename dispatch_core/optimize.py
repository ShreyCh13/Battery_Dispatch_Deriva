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
    """Return per-interval gas availability cap (MW).

    Precedence:
      1. If `gas_pmax_mw > 0`, it is the baseline cap for every interval.
      2. Only when `gas_use_profile_as_cap=True` does the CSV column further
         clip the cap per interval; otherwise the column is ignored so a
         zero-filled column never silently disables the resource.
      3. If `gas_pmax_mw == 0` and `gas_use_profile_as_cap=True`, the column
         alone acts as the time-varying cap (legacy-style behavior).
    """
    pmax = float(cfg.gas_pmax_mw) if cfg.gas_pmax_mw > 0 else 0.0
    has_col = cfg.gas_availability_col in df.columns
    use_profile = bool(cfg.gas_use_profile_as_cap)

    if pmax > 0 and not use_profile:
        return pd.Series(pmax, index=df.index, dtype=float)

    if has_col and use_profile:
        col = pd.to_numeric(df[cfg.gas_availability_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if pmax > 0:
            return col.clip(upper=pmax)
        return col

    return pd.Series(pmax, index=df.index, dtype=float)

def run_lp(
    df: pd.DataFrame,
    cfg: RunConfig,
    *,
    mode: str = "blend",        # blend | revenue | resilience | serve | grid_on_max_revenue | cost_min_gridoff
    blend_lambda: float = 0.9,
    grid_allowed: bool = True,
    min_served_mwh: float | None = None,
    relax_uc: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Linear program for battery dispatch optimization.

    Modes:
      - 'blend': maximize weighted sum of resilience and grid revenue
      - 'revenue': maximize grid revenue
      - 'resilience': maximize load served (legacy lexicographic objective)
      - 'serve': (not used)
      - 'grid_on_max_revenue': (grid ON only) serve 100% of load, maximize net revenue (grid export * price - grid import * price)
      - 'cost_min_gridoff': minimize total economic cost (VOLL * unserved + gas variable + gas no-load + gas startup + battery degradation [+ carbon])
        Reliability is internally priced via VOLL rather than enforced as a hard constraint, which removes the
        "free idle gas" and "phantom clipped revenue" artifacts of the legacy resilience mode and is the scoring
        function used by the configuration screener.

    Args:
        df (pd.DataFrame): Input time series data (must include generation, load, price columns).
        cfg (RunConfig): Run configuration object.
        mode (str): Optimization mode (see above).
        blend_lambda (float): Weight for blend mode (0–1).
        grid_allowed (bool): If False, disables grid import/export.
        relax_uc (bool): If True, gas commitment binaries are relaxed to [0,1] continuous (LP rather than MILP).
            Used by the screener for tractability; the recommended config is then re-validated with full MILP.

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

    # Time step (hours per interval). Default 1.0 preserves legacy hourly behaviour.
    dt = float(getattr(cfg, "dt_hours", 1.0) or 1.0)
    if dt <= 0:
        raise ValueError(f"cfg.dt_hours must be positive, got {dt}")

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
    gas_no_load_cost = float(getattr(cfg, "effective_no_load_cost_usd_per_h", 0.0) or 0.0)
    bess_deg_cost = float(getattr(cfg, "bess_deg_cost_usd_per_mwh", 0.0) or 0.0)
    voll = float(getattr(cfg, "voll_usd_per_mwh", 5000.0) or 0.0)
    carbon_price = float(getattr(cfg, "carbon_price_usd_per_ton", 0.0) or 0.0)
    co2_per_mwh = float(getattr(cfg, "gas_co2_tons_per_mwh", 0.0) or 0.0)
    gas_cap_series = _build_gas_cap_series(df, cfg)

    # cost_min_gridoff is conceptually grid-off; let callers still pass grid_allowed=True
    # if they want to extend to grid-on screening (merchant revenue is added below).
    if mode == "cost_min_gridoff" and not grid_allowed:
        POI = 0
    POI_local = POI

    # ── LP/MILP model setup ───────────────────────────────────────
    # cost_min_gridoff is a minimization; everything else stays maximization.
    if mode == "cost_min_gridoff":
        prob = pulp.LpProblem("BatteryDispatch", pulp.LpMinimize)
    else:
        prob = pulp.LpProblem("BatteryDispatch", pulp.LpMaximize)
    # Battery 1 variables
    c1  = pulp.LpVariable.dicts("c1",  range(T), 0, Pmax1)  # Charge (MW)
    d1  = pulp.LpVariable.dicts("d1",  range(T), 0, Pmax1)  # Discharge (MW)
    s1  = pulp.LpVariable.dicts("s1",  range(T), 0, Emax1)  # State of charge (MWh)
    # Battery 2 variables
    c2  = pulp.LpVariable.dicts("c2",  range(T), 0, Pmax2)
    d2  = pulp.LpVariable.dicts("d2",  range(T), 0, Pmax2)
    s2  = pulp.LpVariable.dicts("s2",  range(T), 0, Emax2)
    # System variables (all power vars are MW; energy = power*dt)
    v  = pulp.LpVariable.dicts("v",  range(T), 0)             # Load served (MW)
    gi = pulp.LpVariable.dicts("gi", range(T), 0, POI_local)  # Grid import (MW)
    ge = pulp.LpVariable.dicts("ge", range(T), 0, POI_local)  # Grid export (MW)
    cl = pulp.LpVariable.dicts("cl", range(T), 0)             # Clipped energy (MW)
    gas_gen = pulp.LpVariable.dicts("gas_gen", range(T), 0)   # Gas generation (MW)
    # Unserved energy auxiliary (only meaningful for cost_min_gridoff but defined always for consistency)
    unserved = pulp.LpVariable.dicts("unserved", range(T), 0)
    gas_on = {}
    gas_start = {}
    gas_stop = {}
    if gas_dispatchable:
        bin_cat = "Continuous" if relax_uc else "Binary"
        gas_on = pulp.LpVariable.dicts("gas_on", range(T), 0, 1, cat=bin_cat)
        gas_start = pulp.LpVariable.dicts("gas_start", range(T), 0, 1, cat=bin_cat)
        gas_stop = pulp.LpVariable.dicts("gas_stop", range(T), 0, 1, cat=bin_cat)

    # Objective function components (energy = power * dt; per-event costs unchanged by dt)
    grid_revenue = pulp.lpSum(price.iloc[t] * (ge[t] - gi[t]) * dt for t in range(T))
    resilience    = pulp.lpSum(val * v[t] * dt for t in range(T))
    clipped_revenue = pulp.lpSum(price.iloc[t] * cl[t] * dt for t in range(T))  # legacy "phantom" merchant from clipped energy
    gas_energy_cost = pulp.lpSum(gas_var_cost * gas_gen[t] * dt for t in range(T))
    gas_start_cost = pulp.lpSum(float(cfg.gas_startup_cost_usd) * gas_start[t] for t in range(T)) if gas_dispatchable else 0.0
    gas_no_load_cost_term = (
        pulp.lpSum(gas_no_load_cost * gas_on[t] * dt for t in range(T)) if gas_dispatchable and gas_no_load_cost > 0 else 0.0
    )
    bess_deg_cost_term = pulp.lpSum(bess_deg_cost * (d1[t] + d2[t]) * dt for t in range(T)) if bess_deg_cost > 0 else 0.0
    co2_cost_term = pulp.lpSum(carbon_price * co2_per_mwh * gas_gen[t] * dt for t in range(T)) if (carbon_price > 0 and co2_per_mwh > 0) else 0.0
    unserved_cost_term = pulp.lpSum(voll * unserved[t] * dt for t in range(T))

    total_gas_cost = gas_energy_cost + gas_start_cost
    LEX_WEIGHT = 1_000_000  # large weight to enforce lexicographic priority

    # Artificially high revenue for served load in 'resilience_first_blend' mode
    artificial_resilience_revenue = pulp.LpAffineExpression()
    if mode == "resilience_first_blend":
        artificial_price = float(np.max(price))
        artificial_resilience_revenue = pulp.lpSum(artificial_price * v[t] * dt for t in range(T))

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
    elif mode == "cost_min_gridoff":
        # Cost-optimal dispatch. v[t] + unserved[t] = load[t] is added in the constraints loop
        # so the optimizer trades the marginal cost of generation (gas, degradation) against
        # the value of lost load (VOLL). When grid_allowed is True we subtract merchant revenue
        # so the screener can also evaluate hybrids that sell to the grid; the term collapses
        # to zero when imports/exports are pinned by POI=0.
        prob += (
            unserved_cost_term
            + gas_energy_cost
            + gas_no_load_cost_term
            + gas_start_cost
            + bess_deg_cost_term
            + co2_cost_term
            - grid_revenue
        )
    else:
        raise ValueError("mode must be blend/revenue/resilience/resilience_first_blend/grid_on_max_revenue/cost_min_gridoff")

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
                # Ramp limits scale with the time step (MW/h * h_per_interval).
                prob += gas_gen[t] - gas_gen[t - 1] <= float(cfg.gas_ramp_up_mw_per_h) * dt
                prob += gas_gen[t - 1] - gas_gen[t] <= float(cfg.gas_ramp_down_mw_per_h) * dt
        else:
            # Non-dispatchable gas: must-run style.
            # When gas_enabled and a static pmax is set, honor it instead of
            # silently trusting a possibly all-zero CSV column.
            if cfg.gas_enabled:
                profile_val = float(natgas_profile.iloc[t])
                pmax_static = float(cfg.gas_pmax_mw) if cfg.gas_pmax_mw > 0 else 0.0
                col_all_zero = bool(natgas_profile.sum() <= 1e-9)
                if pmax_static > 0 and col_all_zero:
                    gas_fixed = pmax_static
                elif pmax_static > 0:
                    gas_fixed = min(profile_val, pmax_static)
                else:
                    gas_fixed = profile_val
            else:
                gas_fixed = float(natgas_profile.iloc[t])
            prob += gas_gen[t] == gas_fixed

        # System power balance (per-interval; instantaneous power balance is dt-invariant)
        prob += renewable_gen.iloc[t] + gas_gen[t] + d1[t] + d2[t] + gi[t] == v[t] + c1[t] + c2[t] + ge[t] + cl[t]
        # Load served + unserved == load (cost_min_gridoff uses unserved as a slack with VOLL price;
        # other modes effectively pin unserved=0 because there is no incentive to leave load unserved).
        if mode == "cost_min_gridoff":
            prob += v[t] + unserved[t] == load.iloc[t]
        else:
            prob += v[t] <= load.iloc[t]
            prob += unserved[t] == load.iloc[t] - v[t]
        # POI net flow constraint
        prob += ge[t] - gi[t] <= POI_local
        prob += ge[t] - gi[t] >= -POI_local
        # Battery 1 SOC (energy = power * dt; eta applied to charging energy)
        if t == 0:
            prob += s1[t] == (c1[t] * eta1 - d1[t]) * dt
            prob += s2[t] == (c2[t] * eta2 - d2[t]) * dt
        else:
            prob += s1[t] == s1[t-1] + (c1[t] * eta1 - d1[t]) * dt
            prob += s2[t] == s2[t-1] + (c2[t] * eta2 - d2[t]) * dt
        # SOC bounds (redundant with variable bounds, but explicit)
        prob += s1[t] >= 0
        prob += s1[t] <= Emax1
        prob += s2[t] >= 0
        prob += s2[t] <= Emax2
        # Discharge cannot exceed available SOC for this interval (energy conservation: d*dt <= soc_prev)
        if t == 0:
            prob += d1[t] * dt <= 0.5 * Emax1
            prob += d2[t] * dt <= 0.5 * Emax2
        else:
            prob += d1[t] * dt <= s1[t-1]
            prob += d2[t] * dt <= s2[t-1]

    if gas_dispatchable:
        # Convert hour-based min up/down to interval count via dt.
        # ceil() makes the constraint slightly stricter at sub-hourly resolution, which is the safe direction.
        min_up_h = float(max(0.0, cfg.gas_min_up_h))
        min_down_h = float(max(0.0, cfg.gas_min_down_h))
        min_up = int(np.ceil(min_up_h / dt)) if min_up_h > 0 else 0
        min_down = int(np.ceil(min_down_h / dt)) if min_down_h > 0 else 0
        for t in range(T):
            if min_up > 0:
                start_ix = max(0, t - min_up + 1)
                prob += pulp.lpSum(gas_start[k] for k in range(start_ix, t + 1)) <= gas_on[t]
            if min_down > 0:
                start_ix = max(0, t - min_down + 1)
                prob += pulp.lpSum(gas_stop[k] for k in range(start_ix, t + 1)) <= 1 - gas_on[t]

    if min_served_mwh is not None:
        prob += pulp.lpSum(v[t] * dt for t in range(T)) >= float(min_served_mwh)

    # ── Solve LP/MILP ────────────────────────────────────────────
    solved_with_relaxation = bool(relax_uc)
    if relax_uc:
        prob.solve(pulp.PULP_CBC_CMD(msg=False, mip=False))
    else:
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
    res["unserved"] = arr(unserved)
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
    # Energy-weighted gas cost (gas_var_cost is $/MWh, gas_gen is MW, so multiply by dt to get $).
    # Startup cost is a per-event $ amount and doesn't scale with dt.
    per_step_cost = (
        gas_gen_arr * gas_var_cost * dt
        + res["gas_start"].to_numpy(dtype=float) * float(cfg.gas_startup_cost_usd)
        + res["gas_on"].to_numpy(dtype=float) * gas_no_load_cost * dt
    )
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
    # Energy = power * dt (MW * h = MWh). All metrics use dt for sub-hourly correctness.
    total_load = float(res["load"].sum(skipna=True)) * dt
    served = float(res["serve"].sum(skipna=True)) * dt
    unserved_arr = res["unserved"].to_numpy(dtype=float)
    unserved_mwh = float(unserved_arr.sum()) * dt
    grid_exp = res["grid_exp"].to_numpy(dtype=float)
    grid_imp = res["grid_imp"].to_numpy(dtype=float)
    price_arr = res["price"].to_numpy(dtype=float)
    revenue_tot = ((grid_exp - grid_imp) * price_arr).sum() * dt  # $ (MW * $/MWh * h = $)
    total_wind = float(wind.sum(skipna=True)) * dt
    total_solar = float(solar.sum(skipna=True)) * dt
    total_natgas = float(res["gas_gen"].sum(skipna=True)) * dt
    total_gen = total_wind + total_solar + total_natgas
    green_gen_over_load_pct = (total_gen / total_load * 100) if total_load > 0 else 0.0
    resilience_pct = round(served / total_load * 100, 2) if total_load > 0 else 0.0
    total_charge = float(res["charge"].sum(skipna=True)) * dt if "charge" in res else 0.0
    total_clip = float(res["clipped"].sum(skipna=True)) * dt if "clipped" in res else 0.0
    total_gen_mwh = float(res["generation"].sum(skipna=True)) * dt if "generation" in res else 0.0
    
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
            clipped_revenue = float((clipped_arr * price_arr).sum()) * dt  # $ (MW * $/MWh * h = $)
        else:
            clipped_revenue = None
    # Calculate number of cycles for each battery (energy throughput / capacity)
    discharge1_sum = float(res["discharge1"].to_numpy(dtype=float).sum()) * dt if "discharge1" in res else 0.0
    discharge2_sum = float(res["discharge2"].to_numpy(dtype=float).sum()) * dt if "discharge2" in res else 0.0
    cycles1 = discharge1_sum / float(cfg.battery_energy_mwh) if cfg.battery_energy_mwh > 0 else 0.0
    cycles2 = discharge2_sum / float(cfg.battery2_energy_mwh) if cfg.battery2_energy_mwh > 0 else 0.0
    charge_arr = res["charge"].to_numpy(dtype=float) if "charge" in res else np.zeros(len(res))
    clipped_arr = res["clipped"].to_numpy(dtype=float) if "clipped" in res else np.zeros(len(res))
    gen_arr = res["generation"].to_numpy(dtype=float) if "generation" in res else np.zeros(len(res))
    grid_imp_arr = res["grid_imp"].to_numpy(dtype=float)
    grid_exp_arr = res["grid_exp"].to_numpy(dtype=float)
    discharge1_arr = res["discharge1"].to_numpy(dtype=float) if "discharge1" in res else np.zeros(len(res))
    discharge2_arr = res["discharge2"].to_numpy(dtype=float) if "discharge2" in res else np.zeros(len(res))
    gas_start_count = int(np.round(res["gas_start"].to_numpy(dtype=float).sum())) if "gas_start" in res else 0
    gas_cap_ref = float(cfg.gas_pmax_mw) if cfg.gas_pmax_mw > 0 else float(gas_cap_series.max()) if len(gas_cap_series) else 0.0
    horizon_hours = float(T) * dt
    gas_cf_pct = (100.0 * total_natgas / (gas_cap_ref * horizon_hours)) if gas_cap_ref > 0 and horizon_hours > 0 else 0.0
    renewable_gen_over_load_pct = (
        ((total_wind + total_solar) / total_load * 100.0) if total_load > 0 else 0.0
    )
    # Cost decomposition (always computed; zero terms are zero rather than missing).
    cost_unserved = voll * unserved_mwh
    cost_gas_var = gas_var_cost * total_natgas
    cost_gas_no_load = gas_no_load_cost * float(res["gas_on"].to_numpy(dtype=float).sum()) * dt
    cost_gas_startup = float(cfg.gas_startup_cost_usd) * gas_start_count
    cost_bess_deg = bess_deg_cost * (discharge1_sum + discharge2_sum)
    co2_tons_total = co2_per_mwh * total_natgas
    cost_carbon = carbon_price * co2_tons_total
    total_op_cost = cost_unserved + cost_gas_var + cost_gas_no_load + cost_gas_startup + cost_bess_deg + cost_carbon

    mets = {
        "firmness (%)": round(resilience_pct, 2),
        "TotalGen/TotalLoad": round(float(green_gen_over_load_pct), 2),
        "renewable_gen_over_load_%": round(float(renewable_gen_over_load_pct), 2),
        "Merchant Revenue/Cost": round(float(revenue_tot - per_step_cost.sum()), 2),
        "total_charge_mwh": round(float(charge_arr.sum()) * dt, 2),
        "total_clip_mwh": round(float(clipped_arr.sum()) * dt, 2),
        "total_gen_mwh": round(float(gen_arr.sum()) * dt, 2),
        "total_load_mwh": round(total_load, 2),
        "total_served_mwh": round(served, 2),
        "unserved_energy_mwh": round(unserved_mwh, 4),
        "grid_imp_mwh":   round(float(grid_imp_arr.sum()) * dt, 2),
        "grid_exp_mwh":   round(float(grid_exp_arr.sum()) * dt, 2),
        "cycles_battery1": round(float(cycles1), 2),
        "cycles_battery2": round(float(cycles2), 2),
        "total_wind_mwh": round(total_wind, 2),
        "total_solar_mwh": round(total_solar, 2),
        "total_natgas_mwh": round(total_natgas, 2),
        "natgas_start_count": gas_start_count,
        "natgas_capacity_factor_%": round(float(gas_cf_pct), 2),
        "natgas_total_cost_$": round(float(per_step_cost.sum()), 2),
        "co2_tons": round(co2_tons_total, 3),
        "total_operating_cost_usd": round(total_op_cost, 2),
        "cost_of_unserved_usd": round(cost_unserved, 2),
        "cost_breakdown": {
            "unserved_voll": round(cost_unserved, 2),
            "gas_variable": round(cost_gas_var, 2),
            "gas_no_load": round(cost_gas_no_load, 2),
            "gas_startup": round(cost_gas_startup, 2),
            "bess_degradation": round(cost_bess_deg, 2),
            "carbon": round(cost_carbon, 2),
        },
        "horizon_hours": round(horizon_hours, 4),
        "dt_hours": dt,
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
    # Use power sums rather than energy here; the inequality is dt-invariant since both sides scale equally.
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
