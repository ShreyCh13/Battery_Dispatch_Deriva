"""
economics.py
------------
Economic assumptions and annualized cost helpers for the configuration
screening / capacity-planning workflow.

This module is intentionally thin: it owns nothing about dispatch, only the
financial layer used to compare resource builds. The defaults are
*screening-grade* numbers calibrated against publicly available estimates
(NREL ATB, EIA AEO, BNEF) for 2024-2026 vintages and are meant to be a
starting point, not a substitute for project-specific economic modelling.

Design choices:
- All capex is given in $/kW (power) or $/kWh (energy, BESS only).
- Fixed O&M is $/kW-year. Variable O&M for gas is handled inside RunConfig.
- Discount rate and life are per-resource; CRF derived as r/(1-(1+r)^-N).
- annualized_cost() returns a clear breakdown in $/year prorated to the
  simulation horizon length so dispatch-period opex (from run_lp) and
  annualised capex/O&M can be summed apples-to-apples.

Units: USD throughout. Energy in MWh, power in MW.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from .config import RunConfig


HOURS_PER_YEAR = 8760.0


class ResourceEcon(BaseModel):
    """Per-resource economic parameters."""

    capex_usd_per_kw: float = 0.0
    capex_usd_per_kwh: float = 0.0  # BESS energy capex; ignored for non-storage resources
    fixed_om_usd_per_kw_yr: float = 0.0
    life_years: float = 25.0
    discount_rate: float = 0.07


class EconAssumptions(BaseModel):
    """Bundle of per-resource economic assumptions plus optional carbon price.

    Defaults are 2024-2026 screening-grade values for the U.S. market.
    Override any field via the UI or a saved scenario bundle.
    """

    solar: ResourceEcon = Field(default_factory=lambda: ResourceEcon(
        capex_usd_per_kw=1000.0,
        fixed_om_usd_per_kw_yr=15.0,
        life_years=30.0,
        discount_rate=0.07,
    ))
    wind: ResourceEcon = Field(default_factory=lambda: ResourceEcon(
        capex_usd_per_kw=1500.0,
        fixed_om_usd_per_kw_yr=35.0,
        life_years=25.0,
        discount_rate=0.07,
    ))
    bess: ResourceEcon = Field(default_factory=lambda: ResourceEcon(
        capex_usd_per_kw=300.0,
        capex_usd_per_kwh=200.0,
        fixed_om_usd_per_kw_yr=10.0,
        life_years=15.0,
        discount_rate=0.07,
    ))
    gas: ResourceEcon = Field(default_factory=lambda: ResourceEcon(
        capex_usd_per_kw=800.0,
        fixed_om_usd_per_kw_yr=20.0,
        life_years=25.0,
        discount_rate=0.07,
    ))

    # Optional carbon price; mirrored into RunConfig.carbon_price_usd_per_ton at run time.
    carbon_price_usd_per_ton: float = 0.0

    # ── Stubs for deferred features (do not compute today; reserved for V2) ──
    # Ancillary services: revenue per MW per year for BESS reg-up/reg-down/spin.
    # Placeholder so the schema is forward-compatible with future scenario JSONs.
    bess_as_revenue_usd_per_kw_yr: float = 0.0
    # Representative-period weighting (list of weights summing to 1.0). With one
    # period today this is implicitly [1.0]. Multi-week/year support will iterate
    # over a list of (df, weight) tuples.
    period_weights: Optional[list[float]] = None
    # Multi-year stochastic: list of weather years to evaluate (defer implementation).
    weather_years: Optional[list[int]] = None


def crf(rate: float, life_years: float) -> float:
    """Capital recovery factor: annualises lump-sum capex over life at given rate.

    Returns 1/life when rate=0 (avoid divide-by-zero) and a sensible value for
    typical rates. CRF=r/(1-(1+r)^-N).
    """
    if life_years <= 0:
        return 0.0
    if abs(rate) < 1e-9:
        return 1.0 / float(life_years)
    return rate / (1.0 - (1.0 + rate) ** (-float(life_years)))


def annualized_cost(cfg: RunConfig, econ: EconAssumptions, *,
                    solar_mw: Optional[float] = None,
                    wind_mw: Optional[float] = None) -> dict:
    """Return per-resource annualised capex + fixed O&M and prorated values for the run horizon.

    Args:
        cfg: Run configuration providing battery / gas sizes and dt_hours.
        econ: Economic assumptions per resource.
        solar_mw: Override for installed solar capacity (MW). If None, the
            screener interprets the CSV column as already representing the
            installed MW and the caller must pass the chosen value.
        wind_mw: Same as ``solar_mw`` for wind.

    Returns:
        dict with keys:
          - ``annualized_*_usd``: annualised cost per resource ($/yr)
          - ``fixed_om_*_usd_yr``: fixed O&M per resource ($/yr)
          - ``annualized_total_usd_yr``: full annual fixed cost
          - ``horizon_hours``: simulation horizon in hours (dt_hours * T)
          - ``period_capex_usd``: capex+O&M prorated to the horizon
            (so it can be summed with operating cost from run_lp directly).

    The screener uses ``annualized_total_usd_yr + (operating_cost / horizon * 8760)``
    when reporting "total annual cost"; this function provides the capex side.
    """
    if solar_mw is None:
        solar_mw = 0.0
    if wind_mw is None:
        wind_mw = 0.0

    # Annualised capex via CRF
    def _ann_capex_kw(rec: ResourceEcon, mw: float) -> float:
        return crf(rec.discount_rate, rec.life_years) * rec.capex_usd_per_kw * mw * 1000.0

    def _ann_capex_kwh(rec: ResourceEcon, mwh: float) -> float:
        return crf(rec.discount_rate, rec.life_years) * rec.capex_usd_per_kwh * mwh * 1000.0

    solar_ann = _ann_capex_kw(econ.solar, float(solar_mw))
    wind_ann = _ann_capex_kw(econ.wind, float(wind_mw))
    bess_power_mw = float(cfg.battery_power_mw) + float(cfg.battery2_power_mw)
    bess_energy_mwh = float(cfg.battery_energy_mwh) + float(cfg.battery2_energy_mwh)
    bess_pwr_ann = _ann_capex_kw(econ.bess, bess_power_mw)
    bess_eng_ann = _ann_capex_kwh(econ.bess, bess_energy_mwh)
    gas_ann = _ann_capex_kw(econ.gas, float(cfg.gas_pmax_mw))

    # Fixed O&M
    solar_fom = econ.solar.fixed_om_usd_per_kw_yr * float(solar_mw) * 1000.0
    wind_fom = econ.wind.fixed_om_usd_per_kw_yr * float(wind_mw) * 1000.0
    bess_fom = econ.bess.fixed_om_usd_per_kw_yr * bess_power_mw * 1000.0
    gas_fom = econ.gas.fixed_om_usd_per_kw_yr * float(cfg.gas_pmax_mw) * 1000.0

    annual_total = (
        solar_ann + wind_ann
        + bess_pwr_ann + bess_eng_ann
        + gas_ann
        + solar_fom + wind_fom + bess_fom + gas_fom
    )

    # Prorate to the simulation horizon so dispatch opex (which covers the horizon)
    # and capex (which is annual) can be summed without unit confusion.
    dt_h = float(getattr(cfg, "dt_hours", 1.0) or 1.0)
    # Caller is expected to pass T via the screener; we cannot infer it from cfg alone.
    # We expose horizon_hours=None here; screening.py handles the proration.
    return {
        "annualized_solar_usd": round(solar_ann, 2),
        "annualized_wind_usd": round(wind_ann, 2),
        "annualized_bess_power_usd": round(bess_pwr_ann, 2),
        "annualized_bess_energy_usd": round(bess_eng_ann, 2),
        "annualized_gas_usd": round(gas_ann, 2),
        "fixed_om_solar_usd_yr": round(solar_fom, 2),
        "fixed_om_wind_usd_yr": round(wind_fom, 2),
        "fixed_om_bess_usd_yr": round(bess_fom, 2),
        "fixed_om_gas_usd_yr": round(gas_fom, 2),
        "annualized_total_usd_yr": round(annual_total, 2),
        "dt_hours": dt_h,
    }
