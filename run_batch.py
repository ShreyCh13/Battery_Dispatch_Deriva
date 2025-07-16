"""
run_batch.py
------------
Batch runner for battery dispatch simulations.
Runs the optimizer for multiple combinations of load, battery power, and stack count, and saves results to CSV.
"""

from itertools import product
import pandas as pd
from pathlib import Path
from dispatch_core import config, data_io, optimize

# TODO: Update EXCEL path to your actual data file
EXCEL = Path(r"C:\path\to\your.xlsx")   # <- change this to your data file

# Base configuration for all runs
cfg0 = config.RunConfig(
    path=EXCEL,
    sheet_name="Combined",
    start_date="2015-01-01",
    end_date="2015-01-31",
    load_mw=140,
    load_type="24-7",
    battery_power_mw=200,
    battery_duration_h=4,
    battery_count=1,
    rte=0.86,
    poi_limit_mw=250,
)

# Load the full dataset once
df_full = data_io.load_data(cfg0)
records = []

# Iterate over all combinations of load, battery power, and stack count
for load, power, stacks in product([100, 140], [0, 100, 200], [1, 2]):
    # Update configuration for this run
    cfg = cfg0.copy(update=dict(
        load_mw=load,
        battery_power_mw=power,
        battery_count=stacks
    ))
    # Only use supported modes for run_lp (e.g., 'blend', 'revenue', 'resilience')
    _, m = optimize.run_lp(df_full, cfg, mode="blend", blend_lambda=0.9)
    m.update(load=load, batt_power=power, stacks=stacks)
    records.append(m)

# Save all results to CSV
pd.DataFrame(records).to_csv("batch_results.csv", index=False)
print("batch_results.csv written")
