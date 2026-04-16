import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dispatch_core.config import RunConfig
from dispatch_core.optimize import run_lp
from dispatch_core.sizing import run_gas_capacity_sweep, recommend_gas_capacity


def _base_cfg() -> RunConfig:
    return RunConfig(
        path=Path("template_may2015.csv"),
        sheet_name="test",
        start_date="2015-05-01",
        end_date="2015-05-02",
        load_mw=100,
        load_type="24-7",
        battery_power_mw=50,
        battery_duration_h=2,
        battery_count=1,
        rte=0.86,
        battery2_power_mw=0.0,
        battery2_duration_h=0.0,
        battery2_rte=0.86,
        poi_limit_mw=200,
        market_price_col="Market Price ($/MWh)",
    )


def _sample_df(periods: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2015-05-01", periods=periods, freq="h")
    return pd.DataFrame(
        {
            "Wind (MW)": np.linspace(20, 60, periods),
            "Solar (MW)": np.maximum(0, 80 * np.sin(np.linspace(-1.5, 1.5, periods))),
            "NatGas (MW)": np.full(periods, 70.0),
            "Load (MW)": np.full(periods, 120.0),
            "Market Price ($/MWh)": np.linspace(30, 90, periods),
        },
        index=idx,
    )


class NatGasIntegrationTests(unittest.TestCase):
    def test_legacy_path_matches_when_dispatchable_off(self):
        df = _sample_df()
        cfg_legacy = _base_cfg().model_copy(update={"gas_enabled": False, "gas_dispatchable": False})
        cfg_profile = _base_cfg().model_copy(
            update={
                "gas_enabled": True,
                "gas_dispatchable": False,
                "gas_var_cost_usd_per_mwh": 0.0,
                "gas_startup_cost_usd": 0.0,
            }
        )
        _, m1 = run_lp(df, cfg_legacy, mode="resilience", grid_allowed=False)
        _, m2 = run_lp(df, cfg_profile, mode="resilience", grid_allowed=False)

        self.assertAlmostEqual(m1["firmness (%)"], m2["firmness (%)"], places=6)
        self.assertAlmostEqual(m1["total_served_mwh"], m2["total_served_mwh"], places=6)
        self.assertAlmostEqual(m1["total_gen_mwh"], m2["total_gen_mwh"], places=6)

    def test_simple_and_advanced_cost_match_when_equivalent(self):
        df = _sample_df()
        base = {
            "gas_enabled": True,
            "gas_dispatchable": True,
            "gas_pmax_mw": 70.0,
            "gas_pmin_mw": 10.0,
            "gas_ramp_up_mw_per_h": 1000.0,
            "gas_ramp_down_mw_per_h": 1000.0,
            "gas_var_cost_usd_per_mwh": 28.0,
            "gas_startup_cost_usd": 0.0,
        }
        cfg_simple = _base_cfg().model_copy(update={**base, "gas_cost_mode": "simple"})
        cfg_adv = _base_cfg().model_copy(
            update={
                **base,
                "gas_cost_mode": "advanced",
                "gas_heat_rate_mmbtu_per_mwh": 7.0,
                "gas_fuel_price_usd_per_mmbtu": 4.0,
                "gas_vom_usd_per_mwh": 0.0,
            }
        )
        _, m_simple = run_lp(df, cfg_simple, mode="grid_on_max_revenue", grid_allowed=True)
        _, m_adv = run_lp(df, cfg_adv, mode="grid_on_max_revenue", grid_allowed=True)
        self.assertAlmostEqual(m_simple["natgas_total_cost_$"], m_adv["natgas_total_cost_$"], places=4)

    def test_uc_ramp_limits_respected(self):
        df = _sample_df(12)
        cfg = _base_cfg().model_copy(
            update={
                "gas_enabled": True,
                "gas_dispatchable": True,
                "gas_pmax_mw": 100.0,
                "gas_pmin_mw": 0.0,
                "gas_ramp_up_mw_per_h": 5.0,
                "gas_ramp_down_mw_per_h": 5.0,
            }
        )
        res, _ = run_lp(df, cfg, mode="resilience", grid_allowed=False)
        ramps = np.abs(np.diff(res["gas_gen"].to_numpy(dtype=float)))
        self.assertTrue((ramps <= 5.0001).all())

    def test_gas_runs_when_csv_natgas_column_is_zero(self):
        """Regression: a zero-filled NatGas (MW) column must not silently
        override a user-configured gas_pmax_mw."""
        df = _sample_df()
        df["NatGas (MW)"] = 0.0
        cfg = _base_cfg().model_copy(
            update={
                "gas_enabled": True,
                "gas_dispatchable": True,
                "gas_pmax_mw": 50.0,
                "gas_pmin_mw": 0.0,
                "gas_var_cost_usd_per_mwh": 5.0,
                "gas_use_profile_as_cap": False,
            }
        )
        res, mets = run_lp(df, cfg, mode="resilience", grid_allowed=False)
        self.assertGreater(float(res["gas_gen"].sum()), 0.0)
        self.assertGreater(float(mets["total_natgas_mwh"]), 0.0)

    def test_profile_cap_still_works_when_opted_in(self):
        df = _sample_df()
        df["NatGas (MW)"] = 0.0
        cfg = _base_cfg().model_copy(
            update={
                "gas_enabled": True,
                "gas_dispatchable": True,
                "gas_pmax_mw": 50.0,
                "gas_use_profile_as_cap": True,
            }
        )
        res, _ = run_lp(df, cfg, mode="resilience", grid_allowed=False)
        self.assertEqual(float(res["gas_gen"].sum()), 0.0)

    def test_renewable_only_kpi_is_present(self):
        df = _sample_df()
        cfg = _base_cfg().model_copy(update={"gas_enabled": False})
        _, mets = run_lp(df, cfg, mode="resilience", grid_allowed=False)
        self.assertIn("renewable_gen_over_load_%", mets)
        self.assertIn("TotalGen/TotalLoad", mets)

    def test_sizing_recommendation_has_target_solution(self):
        df = _sample_df()
        cfg = _base_cfg().model_copy(
            update={
                "gas_enabled": True,
                "gas_dispatchable": True,
                "gas_pmax_mw": 0.0,
                "gas_var_cost_usd_per_mwh": 50.0,
            }
        )
        sweep = run_gas_capacity_sweep(df, cfg, capacities_mw=[0, 25, 50, 75, 100], firmness_target_pct=90.0, grid_allowed=False)
        rec = recommend_gas_capacity(sweep, firmness_target_pct=90.0)
        self.assertIn("recommended_capacity_mw", rec)
        self.assertGreaterEqual(rec["recommended_capacity_mw"], 0.0)
        self.assertFalse(sweep.empty)


if __name__ == "__main__":
    unittest.main()
