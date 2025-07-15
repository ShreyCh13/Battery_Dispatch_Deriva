from itertools import product
import pandas as pd
from pathlib import Path
from dispatch_core import config, data_io, optimize

EXCEL = Path(r"C:\path\to\your.xlsx")   # <- change once

cfg0 = config.RunConfig(
    path=EXCEL, sheet_name="Combined",
    start_date="2015-01-01", end_date="2015-01-31",
    load_mw=140, load_type="24-7",
    battery_power_mw=200, battery_duration_h=4,
    battery_count=1, rte=0.86, poi_limit_mw=250,
)

df_full = data_io.load_data(cfg0)
records = []
for load, power, stacks in product([100, 140], [0, 100, 200], [1, 2]):
    cfg = cfg0.copy(update=dict(load_mw=load,
                                battery_power_mw=power,
                                battery_count=stacks))
    # Only use supported modes for run_lp (resilience, blend, revenue, etc.)
    _, m = optimize.run_lp(df_full, cfg, mode="blend", blend_lambda=0.9)
    m.update(load=load, batt_power=power, stacks=stacks)
    records.append(m)
pd.DataFrame(records).to_csv("batch_results.csv", index=False)
print("batch_results.csv written")
