"""
dispatch_core.screen
--------------------
CLI entrypoint for the capacity screening workflow. Runs end-to-end with no UI.

Usage:
    python -m dispatch_core.screen \\
        --scenario scenarios/mission_critical_microgrid.json \\
        --data template_may2015.csv \\
        --out results/run_2026_04_16/

Writes to the output directory:
    - sweep.csv         : every configuration tested + metrics
    - ranked.csv        : sorted / Pareto-flagged / reliability-flagged
    - recommended.json  : chosen configuration + rationale
    - validation.json   : full-MILP re-run metrics for the recommended config
    - summary.txt       : one-page human-readable summary

Enables batch runs, scheduled jobs, server execution, and CI smoke tests
without launching Streamlit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data_io import load_data
from .profiles import generate as generate_load_profile
from .scenarios import ScenarioBundle
from .screening import screen


def _json_default(o):
    """JSON encoder fallback for pandas / numpy scalars and DataFrames."""
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable")


def _progress(idx: int, total: int, candidate: dict) -> None:
    # Simple stderr progress; avoids external deps.
    if total <= 0:
        return
    pct = 100.0 * (idx + 1) / total
    sys.stderr.write(f"\r[{idx+1:4d}/{total:4d}]  {pct:5.1f}%   "
                     f"solar={candidate.get('solar_mw',0):.0f} wind={candidate.get('wind_mw',0):.0f} "
                     f"bess={candidate.get('bess_power_mw',0):.0f}x{candidate.get('bess_duration_h',0):.0f}h "
                     f"gas={candidate.get('gas_mw',0):.0f}  ")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dispatch_core.screen",
        description="Run a capacity screening from a scenario JSON + data CSV.",
    )
    parser.add_argument("--scenario", required=True, type=Path,
                        help="Path to a ScenarioBundle JSON file (see scenarios/).")
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to the time-series CSV (Datetime, Load (MW), Solar (MW), Wind (MW), NatGas (MW), Market Price ($/MWh)).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory. Will be created if missing.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the MILP re-run on the recommended config.")
    parser.add_argument("--reliability-pct", type=float, default=None,
                        help="Override the scenario's reliability target (percent).")
    args = parser.parse_args(argv)

    # Load scenario
    if not args.scenario.exists():
        parser.error(f"Scenario file not found: {args.scenario}")
    bundle = ScenarioBundle.load(args.scenario)

    # Point the RunConfig at the user-supplied data path (scenarios are data-agnostic).
    run_cfg = bundle.run_config.model_copy(update={"path": args.data})

    # Load + filter data
    df = load_data(run_cfg)
    if df.empty:
        parser.error(f"No rows in {args.data} within {run_cfg.start_date}..{run_cfg.end_date}.")

    # Auto-synthesise Load if the CSV column is missing or all-zero so shipped
    # scenarios work out-of-the-box with the generic template. The user can
    # always override by providing a real Load column in their data file.
    if "Load (MW)" not in df.columns or pd.to_numeric(df["Load (MW)"], errors="coerce").fillna(0.0).abs().sum() < 1e-9:
        sys.stderr.write("[screen] No non-zero Load (MW) in data; synthesising from load_mw/load_type.\n")
        df["Load (MW)"] = generate_load_profile(pd.DatetimeIndex(df.index), run_cfg)

    if bundle.screening_ranges is None:
        parser.error("Scenario has no screening_ranges; nothing to sweep.")
    ranges = bundle.screening_ranges.to_ranges()

    reliability = float(args.reliability_pct if args.reliability_pct is not None else bundle.reliability_target_pct)

    sys.stderr.write(f"Loaded scenario: {bundle.name}\n")
    sys.stderr.write(f"Grid size: {ranges.grid_size()} candidates\n")
    sys.stderr.write(f"Reliability target: {reliability}%\n")
    sys.stderr.write(f"Grid allowed: {bundle.grid_allowed}\n\n")

    result = screen(
        df, run_cfg, ranges, bundle.econ,
        reliability_target_pct=reliability,
        grid_allowed=bundle.grid_allowed,
        relax_uc=True,
        validate_recommended=not args.no_validate,
        progress_cb=_progress,
    )
    sys.stderr.write("\n\n")

    # Write outputs
    args.out.mkdir(parents=True, exist_ok=True)
    result.sweep_df.to_csv(args.out / "sweep.csv", index=False)
    result.ranked_df.to_csv(args.out / "ranked.csv", index=False)
    (args.out / "recommended.json").write_text(json.dumps(result.recommended, indent=2, default=_json_default))

    if result.validation is not None:
        v = dict(result.validation)
        # Dispatch DF is large; save separately as CSV rather than embed in JSON.
        dispatch_df = v.pop("dispatch_df", None)
        run_cfg_obj = v.pop("run_cfg", None)
        if dispatch_df is not None:
            dispatch_df.to_csv(args.out / "validation_dispatch.csv")
        if run_cfg_obj is not None:
            v["run_cfg"] = run_cfg_obj.model_dump(mode="json") if hasattr(run_cfg_obj, "model_dump") else None
        (args.out / "validation.json").write_text(json.dumps(v, indent=2, default=_json_default))

    # Human-readable summary
    rec = result.recommended or {}
    cfg = rec.get("config") or {}
    summary_lines = [
        f"Scenario: {bundle.name}",
        f"Description: {bundle.description}",
        f"Data: {args.data}",
        f"Reliability target: {reliability}%",
        f"Grid allowed: {bundle.grid_allowed}",
        f"Grid size tested: {len(result.sweep_df)} configurations",
        "",
        "Recommendation:",
        f"  {rec.get('message','(no recommendation)')}",
        "",
    ]
    if cfg:
        summary_lines += [
            f"  Solar: {float(cfg.get('solar_mw',0)):.1f} MW",
            f"  Wind:  {float(cfg.get('wind_mw',0)):.1f} MW",
            f"  BESS:  {float(cfg.get('bess_power_mw',0)):.1f} MW / {float(cfg.get('bess_duration_h',0)):.1f} h  ({float(cfg.get('bess_energy_mwh',0)):.1f} MWh)",
            f"  Gas:   {float(cfg.get('gas_mw',0)):.1f} MW",
            f"  Firmness: {float(cfg.get('firmness_pct',0)):.2f}%",
            f"  Total annual cost: ${float(cfg.get('total_annual_cost_usd',0)):,.0f}/yr",
            f"  CO2 tons (horizon): {float(cfg.get('co2_tons',0)):.1f}",
        ]

    (args.out / "summary.txt").write_text("\n".join(summary_lines))
    sys.stderr.write(f"Wrote outputs to {args.out.resolve()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
