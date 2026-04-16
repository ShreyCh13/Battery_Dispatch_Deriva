# Scenario library

Each JSON in this folder is a `ScenarioBundle`:

- `run_config` — RunConfig (load, battery, gas, VOLL, dt_hours, ...).
- `econ` — EconAssumptions (capex / fixed O&M / discount rate / life / carbon price).
- `screening_ranges` — per-resource sweep spec (solar/wind/BESS/gas MW ranges).
- `reliability_target_pct` — threshold for "meets reliability" ranking.
- `grid_allowed` — whether exports/imports are part of the cost objective.

Loaded from the UI's **Scenario** dropdown on the Configuration Recommendation page,
or from the CLI via `python -m dispatch_core.screen --scenario scenarios/<file>.json --data <data.csv> --out results/`.

The 3 preloaded cases cover the common ends of the design space:
- `mission_critical_microgrid.json` — data center / hospital, ~100% firmness required, high VOLL.
- `commercial_c_and_i.json` — C&I site, medium reliability, cost-driven.
- `remote_industrial.json` — off-grid, diesel/gas-dominant, mid-range firmness.
