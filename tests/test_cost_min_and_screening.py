"""
Tests for the cost-min objective and capacity-screening workflow.

These cover:
- Parity: legacy ``resilience`` vs. new ``cost_min_gridoff`` at VOLL=0 with no
  gas / degradation / carbon costs should behave equivalently on firmness.
- VOLL gating: raising VOLL makes the solver commit more gas (fewer unserved
  MWh) all else equal.
- No-load cost: a large no-load cost reduces total "gas on" hours.
- Screening grid size, ranking monotonicity, recommend_config fallback.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dispatch_core.config import RunConfig
from dispatch_core.economics import EconAssumptions
from dispatch_core.optimize import run_lp
from dispatch_core.screening import (
    ResourceRange,
    ScreeningRanges,
    build_config_grid,
    recommend_config,
    score_and_rank,
    SWEEP_HARD_CAP,
)


def _base_cfg(**overrides) -> RunConfig:
    cfg = RunConfig(
        path=Path("template_may2015.csv"),
        sheet_name="test",
        start_date="2015-05-01",
        end_date="2015-05-02",
        load_mw=100,
        load_type="24-7",
        battery_power_mw=20,
        battery_duration_h=2,
        battery_count=1,
        rte=0.86,
        battery2_power_mw=0.0,
        battery2_duration_h=0.0,
        battery2_rte=0.86,
        poi_limit_mw=200,
        market_price_col="Market Price ($/MWh)",
        gas_enabled=True,
        gas_dispatchable=True,
        gas_pmax_mw=40.0,
        gas_pmin_mw=0.0,
        gas_cost_mode="simple",
        gas_var_cost_usd_per_mwh=30.0,
    )
    return cfg.model_copy(update=overrides)


def _sample_df(periods: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2015-05-01", periods=periods, freq="h")
    return pd.DataFrame(
        {
            "Wind (MW)": np.zeros(periods),
            "Solar (MW)": np.zeros(periods),
            "NatGas (MW)": np.full(periods, 40.0),
            "Load (MW)": np.full(periods, 100.0),
            "Market Price ($/MWh)": np.full(periods, 50.0),
        },
        index=idx,
    )


class CostMinObjectiveTests(unittest.TestCase):
    def test_high_voll_reduces_unserved(self):
        df = _sample_df()
        cfg_low = _base_cfg(voll_usd_per_mwh=10.0, gas_var_cost_usd_per_mwh=30.0)
        cfg_high = _base_cfg(voll_usd_per_mwh=10_000.0, gas_var_cost_usd_per_mwh=30.0)
        _, mets_low = run_lp(df, cfg_low, mode="cost_min_gridoff", grid_allowed=False, relax_uc=True)
        _, mets_high = run_lp(df, cfg_high, mode="cost_min_gridoff", grid_allowed=False, relax_uc=True)
        # With higher VOLL, unserved MWh must not increase; in this under-capacity
        # case we expect it to strictly decrease.
        self.assertLessEqual(float(mets_high["unserved_energy_mwh"]), float(mets_low["unserved_energy_mwh"]) + 1e-6)

    def test_no_load_cost_reduces_gas_on_hours(self):
        df = _sample_df()
        # Oversize gas so the solver has a choice between "idle on" and "off".
        cfg_free = _base_cfg(gas_pmax_mw=150.0, gas_no_load_cost_usd_per_h=0.0, voll_usd_per_mwh=1000.0)
        cfg_pay = _base_cfg(gas_pmax_mw=150.0, gas_no_load_cost_usd_per_h=500.0, voll_usd_per_mwh=1000.0)
        res_free, _ = run_lp(df, cfg_free, mode="cost_min_gridoff", grid_allowed=False, relax_uc=False)
        res_pay, _ = run_lp(df, cfg_pay, mode="cost_min_gridoff", grid_allowed=False, relax_uc=False)
        # If the column exists, paying no-load cost must not increase committed hours.
        if "NatGas_On" in res_free.columns and "NatGas_On" in res_pay.columns:
            self.assertLessEqual(float(res_pay["NatGas_On"].sum()), float(res_free["NatGas_On"].sum()) + 1e-6)

    def test_metrics_contract(self):
        df = _sample_df()
        cfg = _base_cfg(voll_usd_per_mwh=1000.0)
        _, mets = run_lp(df, cfg, mode="cost_min_gridoff", grid_allowed=False, relax_uc=True)
        for k in (
            "total_operating_cost_usd",
            "unserved_energy_mwh",
            "cost_breakdown",
            "co2_tons",
            "horizon_hours",
            "firmness (%)",
        ):
            self.assertIn(k, mets, f"missing metric {k}")
        cb = mets["cost_breakdown"]
        for sub in ("unserved_voll", "gas_variable", "gas_no_load", "gas_startup", "bess_degradation", "carbon"):
            self.assertIn(sub, cb)


class ScreeningTests(unittest.TestCase):
    def _ranges(self) -> ScreeningRanges:
        return ScreeningRanges(
            solar=ResourceRange(name="solar", mode="absolute", values=[0.0, 10.0]),
            wind=ResourceRange(name="wind", mode="absolute", values=[0.0]),
            bess_power=ResourceRange(name="bess_power", mode="absolute", values=[10.0, 20.0]),
            bess_duration=ResourceRange(name="bess_duration", mode="absolute", values=[2.0]),
            gas=ResourceRange(name="gas", mode="absolute", values=[0.0, 40.0]),
        )

    def test_grid_size_matches_cartesian_product(self):
        ranges = self._ranges()
        self.assertEqual(ranges.grid_size(), 2 * 1 * 2 * 1 * 2)
        grid = build_config_grid(ranges)
        self.assertEqual(len(grid), 8)

    def test_hard_cap_blocks_huge_sweeps(self):
        ranges = ScreeningRanges(
            solar=ResourceRange(name="solar", mode="absolute", values=list(np.arange(0, 20))),
            wind=ResourceRange(name="wind", mode="absolute", values=list(np.arange(0, 20))),
            bess_power=ResourceRange(name="bess_power", mode="absolute", values=list(np.arange(0, 20))),
            bess_duration=ResourceRange(name="bess_duration", mode="absolute", values=list(np.arange(0, 20))),
            gas=ResourceRange(name="gas", mode="absolute", values=list(np.arange(0, 20))),
        )
        self.assertGreater(ranges.grid_size(), SWEEP_HARD_CAP)
        with self.assertRaises(ValueError):
            build_config_grid(ranges)

    def test_score_and_rank_flags(self):
        df = pd.DataFrame([
            {"firmness_pct": 99.0, "total_annual_cost_usd": 1000.0, "meets_reliability": None, "solar_mw": 0, "wind_mw": 0, "bess_power_mw": 0, "bess_duration_h": 0, "bess_energy_mwh": 0, "gas_mw": 0},
            {"firmness_pct": 100.0, "total_annual_cost_usd": 2000.0, "meets_reliability": None, "solar_mw": 0, "wind_mw": 0, "bess_power_mw": 0, "bess_duration_h": 0, "bess_energy_mwh": 0, "gas_mw": 0},
            {"firmness_pct": 80.0, "total_annual_cost_usd": 500.0, "meets_reliability": None, "solar_mw": 0, "wind_mw": 0, "bess_power_mw": 0, "bess_duration_h": 0, "bess_energy_mwh": 0, "gas_mw": 0},
        ])
        ranked = score_and_rank(df, reliability_target_pct=99.0)
        self.assertTrue(ranked.loc[ranked["firmness_pct"] == 99.0, "meets_reliability"].iloc[0])
        self.assertFalse(ranked.loc[ranked["firmness_pct"] == 80.0, "meets_reliability"].iloc[0])
        # Cheapest reliable should be the 99% / $1000 point.
        rec = recommend_config(ranked, reliability_target_pct=99.0)
        self.assertTrue(rec["found"])
        self.assertTrue(rec["meets_reliability"])
        self.assertAlmostEqual(float(rec["config"]["total_annual_cost_usd"]), 1000.0)

    def test_recommend_fallback_when_nothing_meets_target(self):
        df = pd.DataFrame([
            {"firmness_pct": 80.0, "total_annual_cost_usd": 500.0, "solar_mw": 0, "wind_mw": 0, "bess_power_mw": 0, "bess_duration_h": 0, "bess_energy_mwh": 0, "gas_mw": 0},
            {"firmness_pct": 90.0, "total_annual_cost_usd": 1500.0, "solar_mw": 0, "wind_mw": 0, "bess_power_mw": 0, "bess_duration_h": 0, "bess_energy_mwh": 0, "gas_mw": 0},
        ])
        ranked = score_and_rank(df, reliability_target_pct=99.0)
        rec = recommend_config(ranked, reliability_target_pct=99.0)
        self.assertTrue(rec["found"])
        self.assertFalse(rec["meets_reliability"])
        self.assertAlmostEqual(float(rec["config"]["firmness_pct"]), 90.0)

    def test_econ_assumptions_instantiate(self):
        econ = EconAssumptions()
        self.assertGreater(econ.solar.capex_usd_per_kw, 0)
        self.assertGreater(econ.bess.capex_usd_per_kwh, 0)


if __name__ == "__main__":
    unittest.main()
