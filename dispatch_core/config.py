"""
config.py
---------
Defines the RunConfig class for storing all configuration parameters for a battery dispatch simulation run.
Uses Pydantic for validation and type safety.
"""

from pathlib import Path
from typing   import Literal
from pydantic import BaseModel, Field

class RunConfig(BaseModel):
    """
    Configuration for a single battery dispatch simulation run.

    Attributes:
        path (Path): Path to the input data file (CSV or Excel).
        sheet_name (str): Sheet name for Excel files (ignored for CSV).
        start_date (str): Start date for the simulation window (YYYY-MM-DD).
        end_date (str): End date for the simulation window (YYYY-MM-DD).
        load_mw (float): Average load in MW.
        load_std (float): Standard deviation for random load profiles.
        load_type (Literal): Type of load profile ('24-7', '16-7', 'random', 'random_16-7').
        battery_power_mw (float): Power rating of battery 1 (MW).
        battery_duration_h (float): Duration of battery 1 (hours).
        battery_count (int): Number of battery stacks (default 1).
        rte (float): Round-trip efficiency of battery 1 (0 < rte < 1).
        battery2_power_mw (float): Power rating of battery 2 (MW).
        battery2_duration_h (float): Duration of battery 2 (hours).
        battery2_rte (float): Round-trip efficiency of battery 2.
        poi_limit_mw (float): Point of interconnection (POI) limit (MW).
        value_per_mwh (float): Value per MWh for resilience (default 100).
        market_price_col (str): Name of the market price column in the data.
        capex_power_usd_per_kw (float): Capex per kW of power.
        capex_energy_usd_per_kwh (float): Capex per kWh of energy.
        discount_years (int): Discount period in years.
    """
    # ── File window ─────────────────────────────────────────
    path: Path
    sheet_name: str = "Combined"
    start_date: str = "2015-01-01"
    end_date:   str = "2015-02-01"

    # ── Load profile ────────────────────────────────────────
    load_mw: float
    load_std: float = 15
    load_type: Literal["24-7", "16-7", "random", "random_16-7"]

    # ── Battery spec per stack ──────────────────────────────
    battery_power_mw:   float
    battery_duration_h: float
    battery_count:      int   = 1
    rte:                float = Field(..., gt=0, lt=1)
    # Battery 2
    battery2_power_mw: float = 0.0
    battery2_duration_h: float = 0.0
    battery2_rte: float = 0.86

    # ── Inverter / POI limit ───────────────────────────────
    poi_limit_mw: float

    # ── Economics ─────────────────────────────────────────--
    value_per_mwh:            float = 100.0
    market_price_col:         str   = "Market Price ($/MWh)"
    capex_power_usd_per_kw:   float = 200
    capex_energy_usd_per_kwh: float = 150
    discount_years:           int   = 20

    # Helper properties for battery energy (MWh)
    @property
    def battery_energy_mwh(self) -> float:
        """Total energy capacity of battery 1 (MWh)."""
        return self.battery_power_mw * self.battery_duration_h

    @property
    def battery2_energy_mwh(self) -> float:
        """Total energy capacity of battery 2 (MWh)."""
        return self.battery2_power_mw * self.battery2_duration_h
