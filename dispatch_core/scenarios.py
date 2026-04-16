"""
scenarios.py
------------
Scenario bundle: serializable snapshot of RunConfig + EconAssumptions +
ScreeningRanges, for save/load on both Streamlit pages and for the CLI.

A scenario is a single JSON file that any of the following can consume:
- Streamlit Configuration Recommendation page ("Scenario" dropdown).
- Streamlit Dispatch Analysis page (loads the RunConfig piece).
- ``python -m dispatch_core.screen --scenario my_case.json --data data.csv``.

Future-proofing: adding a new field (e.g. ``weather_years`` for multi-year
runs or ``period_weights`` for representative periods) is a Pydantic model
extension with sensible defaults. Older scenario files remain loadable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .config import RunConfig
from .economics import EconAssumptions
from .screening import ResourceRange, ScreeningRanges


class ResourceRangeSpec(BaseModel):
    """Serializable mirror of the ResourceRange dataclass."""

    name: str
    mode: str
    values: list[float]
    reference_mw: Optional[float] = None

    def to_range(self) -> ResourceRange:
        return ResourceRange(
            name=self.name,  # type: ignore[arg-type]
            mode=self.mode,  # type: ignore[arg-type]
            values=list(self.values),
            reference_mw=self.reference_mw,
        )

    @classmethod
    def from_range(cls, r: ResourceRange) -> "ResourceRangeSpec":
        return cls(name=r.name, mode=r.mode, values=list(r.values), reference_mw=r.reference_mw)


class ScreeningRangesSpec(BaseModel):
    solar: ResourceRangeSpec
    wind: ResourceRangeSpec
    bess_power: ResourceRangeSpec
    bess_duration: ResourceRangeSpec
    gas: ResourceRangeSpec

    def to_ranges(self) -> ScreeningRanges:
        return ScreeningRanges(
            solar=self.solar.to_range(),
            wind=self.wind.to_range(),
            bess_power=self.bess_power.to_range(),
            bess_duration=self.bess_duration.to_range(),
            gas=self.gas.to_range(),
        )

    @classmethod
    def from_ranges(cls, r: ScreeningRanges) -> "ScreeningRangesSpec":
        return cls(
            solar=ResourceRangeSpec.from_range(r.solar),
            wind=ResourceRangeSpec.from_range(r.wind),
            bess_power=ResourceRangeSpec.from_range(r.bess_power),
            bess_duration=ResourceRangeSpec.from_range(r.bess_duration),
            gas=ResourceRangeSpec.from_range(r.gas),
        )


class ScenarioBundle(BaseModel):
    """Single JSON scenario holding everything a run/screen call needs except the data.

    The CSV/Excel path is referenced separately so the same scenario can be run
    against different data files (different sites, different years).
    """

    name: str = "unnamed"
    description: str = ""
    run_config: RunConfig
    econ: EconAssumptions = Field(default_factory=EconAssumptions)
    screening_ranges: Optional[ScreeningRangesSpec] = None
    reliability_target_pct: float = 99.0
    grid_allowed: bool = False

    # V2 hooks (reserved; do not affect current runs)
    weather_years: Optional[list[int]] = None
    period_weights: Optional[list[float]] = None

    def save(self, path: str | Path) -> None:
        """Write this scenario to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioBundle":
        """Load a scenario from a JSON file. Raises FileNotFoundError / ValidationError."""
        p = Path(path)
        data = json.loads(p.read_text())
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
