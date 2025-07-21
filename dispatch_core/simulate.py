"""
simulate.py
-----------
Simulates battery operation using a user-supplied fixed charge/discharge schedule (no optimization).
Handles round-trip efficiency (RTE), SOC, and all battery constraints.
"""

import numpy as np
import pandas as pd
from .config import RunConfig

def simulate_fixed_schedule(df: pd.DataFrame, cfg: RunConfig, schedule_df: pd.DataFrame, grid_on: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Simulate battery operation using a fixed charge/discharge schedule.
    Args:
        df: DataFrame with time series data (must include 'Wind (MW)', 'Solar (MW)', 'Load (MW)', price col).
        cfg: RunConfig object.
        schedule_df: DataFrame with same index as df, column 'action' with values 'C', 'D', or '' (idle).
        grid_on: If True, allow charging from grid (subject to POI limit). If False, charging only from generation.
    Returns:
        (results DataFrame, metrics dict)
    """
    # --- Parameters ---
    Pmax = cfg.battery_power_mw
    Emax = cfg.battery_energy_mwh
    eta = cfg.rte
    price_col = cfg.market_price_col
    T = len(df)
    POI = getattr(cfg, 'poi_limit_mw', 1e6)  # Use a very high value if not set
    
    # --- Prepare input series ---
    gen = df["Wind (MW)"].fillna(0) + df["Solar (MW)"].fillna(0) + df["NatGas (MW)"].fillna(0)
    load = df["Load (MW)"]
    price = df[price_col]
    
    # --- Initialize outputs ---
    soc = np.zeros(T)
    charge = np.zeros(T)
    discharge = np.zeros(T)
    serve = np.zeros(T)
    grid_imp = np.zeros(T)
    grid_exp = np.zeros(T)
    clipped = np.zeros(T)
    action = schedule_df["action"].values if "action" in schedule_df.columns else np.array([None]*T)
    
    # --- Simulation loop ---
    for t in range(T):
        # Previous SOC
        soc_prev = soc[t-1] if t > 0 else 0.0
        act = str(action[t]).strip().upper() if action[t] is not None else ''
        # --- Charge ---
        if act == 'C':
            max_charge_soc = (Emax - soc_prev) / eta if eta > 0 else 0
            if not grid_on:
                # Only allow charging from generation
                max_charge_gen = gen.iloc[t]
                max_charge = min(Pmax, max_charge_soc, max_charge_gen)
                charge[t] = max(0, max_charge)
                grid_charge = 0.0
            else:
                # Allow charging from both generation and grid, limited by POI
                max_charge = min(Pmax, max_charge_soc, POI)
                charge[t] = max(0, max_charge)
                grid_charge = max(0, charge[t] - gen.iloc[t])
            discharge[t] = 0.0
            soc[t] = soc_prev + charge[t] * eta
        # --- Discharge ---
        elif act == 'D':
            # Max possible discharge (limited by power, available SOC, and POI minus generation)
            max_discharge_power = min(Pmax, soc_prev)
            max_discharge_poi = max(0, POI - gen.iloc[t])
            max_discharge = min(max_discharge_power, max_discharge_poi)
            discharge[t] = max(0, max_discharge)
            charge[t] = 0.0
            soc[t] = soc_prev - discharge[t]
        # --- Idle ---
        else:
            charge[t] = 0.0
            discharge[t] = 0.0
            soc[t] = soc_prev
        # --- Serve load (renewables + discharge) ---
        serve[t] = min(load.iloc[t], gen.iloc[t] + discharge[t])
        # --- Grid flows ---
        # Physically correct: grid export = max(0, generation + discharge - load), grid import = max(0, load - (generation + discharge))
        # Enforce POI on both
        grid_exp[t] = min(max(0, gen.iloc[t] + discharge[t] - load.iloc[t]), POI)
        if grid_on:
            grid_imp[t] = min(max(0, load.iloc[t] - (gen.iloc[t] + discharge[t])), POI)
        else:
            grid_imp[t] = 0.0  # No grid import allowed in fixed schedule mode
        # --- Clipped (unused renewables only) ---
        clipped[t] = max(0, gen.iloc[t] - (serve[t] + charge[t]))
        # Enforce SOC bounds
        soc[t] = min(max(soc[t], 0.0), Emax)
    # --- Results DataFrame ---
    res = df.copy()
    res["generation"] = gen.values
    res["load"] = load.values
    res["price"] = price.values
    res["charge1"] = charge
    res["discharge1"] = discharge
    res["charge2"] = 0.0
    res["discharge2"] = 0.0
    res["charge"] = charge
    res["discharge"] = discharge
    res["serve"] = serve
    res["grid_imp"] = grid_imp
    res["grid_exp"] = grid_exp
    res["clipped"] = clipped
    res["soc1"] = soc
    res["soc2"] = 0.0
    res["soc"] = soc
    res["net_to_grid"] = grid_exp - grid_imp
    res["schedule_action"] = action
    # --- Metrics ---
    total_load = float(res["load"].sum(skipna=True))
    served = float(res["serve"].sum(skipna=True))
    grid_exp_sum = float(res["grid_exp"].sum(skipna=True))
    grid_imp_sum = float(res["grid_imp"].sum(skipna=True))
    price_arr = res["price"].to_numpy(dtype=float)
    revenue_tot = ((res["grid_exp"] - res["grid_imp"]) * price_arr).sum() / 1000
    total_charge = float(res["charge"].sum(skipna=True))
    total_discharge = float(res["discharge"].sum(skipna=True))
    cycles = total_discharge / Emax if Emax > 0 else 0.0
    # Calculate cycles per month
    cycles_per_month = {}
    if Emax > 0:
        res_monthly = res.copy()
        res_monthly['month'] = res_monthly.index.to_series().dt.to_period('M')
        for month, group in res_monthly.groupby('month'):
            month_discharge = float(group['discharge'].sum(skipna=True))
            cycles_per_month[str(month)] = round(month_discharge / Emax, 2)
    total_gen = float(res["generation"].sum(skipna=True)) if "generation" in res else 0.0
    shift_pct = 100 * total_charge / total_gen if total_gen > 0 else 0.0
    total_wastage = total_charge * (1 - eta)
    mets = {
        "total_charge_mwh": round(total_charge, 2),
        "total_discharge_mwh": round(total_discharge, 2),
        "cycles_battery1": round(float(cycles), 2),
        "cycles_per_month": cycles_per_month,
        "shift_pct": round(float(shift_pct), 2),
        "total_wastage_mwh": round(float(total_wastage), 2),
        "total_load_mwh": round(total_load, 2),
        "total_served_mwh": round(served, 2),
        "grid_imp_mwh": round(grid_imp_sum, 2),
        "grid_exp_mwh": round(grid_exp_sum, 2),
        "mode": "fixed_schedule",
        "note": "This run used a fixed charge/discharge schedule. Battery RTE and all constraints were enforced. No optimization was performed."
    }
    return res, mets 