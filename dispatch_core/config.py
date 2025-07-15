from pathlib import Path
from typing   import Literal
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    # ── File window ─────────────────────────────────────────────────────────
    path: Path
    sheet_name: str = "Combined"
    start_date: str = "2015-01-01"
    end_date:   str = "2015-02-01"

    # ── Load profile ────────────────────────────────────────────────────────
    load_mw: float
    load_std: float = 15
    load_type: Literal["24-7", "16-7", "random", "random_16-7"]

    # ── Battery spec per stack ──────────────────────────────────────────────
    battery_power_mw:   float
    battery_duration_h: float
    battery_count:      int   = 1
    rte:                float = Field(..., gt=0, lt=1)
    # Battery 2
    battery2_power_mw: float = 0.0
    battery2_duration_h: float = 0.0
    battery2_rte: float = 0.86

    # ── Inverter / POI limit ────────────────────────────────────────────────
    poi_limit_mw: float

    # ── Economics ───────────────────────────────────────────────────────────
    value_per_mwh:            float = 100.0
    market_price_col:         str   = "Market Price ($/kwh)"
    capex_power_usd_per_kw:   float = 200
    capex_energy_usd_per_kwh: float = 150
    discount_years:           int   = 20

    # helper -----------------------------------------------------------------
    @property
    def battery_energy_mwh(self) -> float:
        return self.battery_power_mw * self.battery_duration_h
    @property
    def battery2_energy_mwh(self) -> float:
        return self.battery2_power_mw * self.battery2_duration_h
