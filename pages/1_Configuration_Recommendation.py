"""
Configuration Recommendation (Streamlit page)
============================================
Multi-resource capacity-planning screener. Given a load and a reference time
series, this page sweeps configurations of solar, wind, BESS (power +
duration), and natural gas, runs a cost-optimal dispatch on each, ranks them
on (firmness vs. total annual cost), and recommends a starting-point build.

Packaged for non-experts:
- Three input panels  -> Serving, Economics, Build limits.
- Load a Scenario JSON (from ``scenarios/``) to pre-fill everything.
- Three output tabs   -> Recommendation + Pareto, All configs, Dispatch drill-down.

This page is Streamlit-only glue; all math lives in ``dispatch_core``.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover - plotly is an optional dep at import time
    px = None

from dispatch_core.config import RunConfig
from dispatch_core.data_io import load_data
from dispatch_core.economics import EconAssumptions, ResourceEcon
from dispatch_core.profiles import generate as generate_load_profile
from dispatch_core.scenarios import ScenarioBundle, ScreeningRangesSpec
from dispatch_core.screening import (
    ResourceRange,
    ScreeningRanges,
    detect_reference_mw,
    screen,
)
from dispatch_core import reporting


st.set_page_config(page_title="Configuration Recommendation", layout="wide")
st.title("Configuration Recommendation")
st.caption(
    "Find a build (solar / wind / BESS / gas) that reliably serves your load at the "
    "lowest total annual cost. Pick a scenario or tune the three panels below."
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "template_may2015.csv"


def _list_scenarios() -> list[Path]:
    if not SCENARIO_DIR.exists():
        return []
    return sorted(p for p in SCENARIO_DIR.glob("*.json"))


def _parse_values(text: str, default: list[float]) -> list[float]:
    """Turn comma / space separated text into a list of floats; fall back to default."""
    if not text or not text.strip():
        return list(default)
    parts = [p.strip() for p in text.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out or list(default)


def _load_dataframe(cfg: RunConfig) -> pd.DataFrame:
    df = load_data(cfg)
    if "Load (MW)" not in df.columns or pd.to_numeric(df["Load (MW)"], errors="coerce").fillna(0.0).abs().sum() < 1e-9:
        df["Load (MW)"] = generate_load_profile(pd.DatetimeIndex(df.index), cfg)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 0. Scenario selector
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("0 · Scenario")
    scenarios = _list_scenarios()
    scenario_labels = ["(start blank)"] + [p.stem.replace("_", " ").title() for p in scenarios]
    chosen = st.selectbox(
        "Preloaded scenario",
        scenario_labels,
        index=0,
        help="Pre-fills every input from a saved scenario JSON in ``scenarios/``.",
    )
    uploaded_scenario = st.file_uploader("…or upload your own scenario JSON", type=["json"])

    bundle: Optional[ScenarioBundle] = None
    if uploaded_scenario is not None:
        try:
            bundle = ScenarioBundle.model_validate_json(uploaded_scenario.read().decode("utf-8"))
            st.success(f"Loaded scenario: {bundle.name}")
        except Exception as e:  # pragma: no cover - UI feedback path
            st.error(f"Could not parse scenario JSON: {e}")
    elif chosen != "(start blank)":
        sel = scenarios[scenario_labels.index(chosen) - 1]
        try:
            bundle = ScenarioBundle.load(sel)
        except Exception as e:  # pragma: no cover
            st.error(f"Could not load {sel.name}: {e}")

# Seed defaults from the bundle (or fall back to blanks)
seed_cfg: RunConfig = bundle.run_config if bundle else RunConfig(
    path=str(DEFAULT_DATA_PATH),
    start_date="2015-05-01",
    end_date="2015-05-31",
    load_mw=100.0,
    load_type="24-7",
    battery_power_mw=50.0,
    battery_duration_h=4.0,
    gas_enabled=True,
    gas_dispatchable=True,
    gas_pmax_mw=50.0,
    voll_usd_per_mwh=5000.0,
)
seed_econ: EconAssumptions = bundle.econ if bundle else EconAssumptions()
seed_ranges: Optional[ScreeningRanges] = bundle.screening_ranges.to_ranges() if (bundle and bundle.screening_ranges) else None
seed_reliability: float = float(bundle.reliability_target_pct) if bundle else 99.0
seed_grid_allowed: bool = bool(bundle.grid_allowed) if bundle else False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data panel
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("1 · Data")
    data_choice = st.radio(
        "Data source",
        ["Template (May 2015)", "Upload CSV"],
        index=0,
        horizontal=False,
    )
    if data_choice == "Upload CSV":
        up = st.file_uploader("CSV with Datetime + Solar/Wind/Price columns", type=["csv"])
        if up is not None:
            tmp_path = Path(st.session_state.get("_tmp_dir", ".")) / "uploaded_screen.csv"
            tmp_path.write_bytes(up.read())
            data_path = tmp_path
        else:
            data_path = DEFAULT_DATA_PATH
    else:
        data_path = DEFAULT_DATA_PATH

    start_date = st.date_input("Start date", value=pd.to_datetime(seed_cfg.start_date).date())
    end_date = st.date_input("End date", value=pd.to_datetime(seed_cfg.end_date).date())


# Build the base RunConfig (everything except the swept sizes)
base_cfg = seed_cfg.model_copy(update={
    "path": str(data_path),
    "start_date": str(start_date),
    "end_date": str(end_date),
})

# Load the data once so reference MW can be auto-detected for the ranges panel.
with st.spinner("Loading time-series…"):
    try:
        df = _load_dataframe(base_cfg)
    except SystemExit as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # pragma: no cover
        st.error(f"Data load failed: {e}")
        st.stop()

ref_solar_auto = detect_reference_mw(df, "Solar (MW)")
ref_wind_auto = detect_reference_mw(df, "Wind (MW)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Three input panels (main area)
# ─────────────────────────────────────────────────────────────────────────────

col_serve, col_econ, col_build = st.columns(3)

# ── Panel A: Serving ────────────────────────────────────────────────────────
with col_serve:
    st.subheader("A · Serving")
    load_mw = st.number_input(
        "Average load (MW)",
        min_value=0.1, max_value=5000.0,
        value=float(base_cfg.load_mw), step=1.0,
        help="Used only when the CSV has no Load column; synthetic load is generated from this.",
    )
    load_type = st.selectbox(
        "Load pattern",
        ["24-7", "16-7", "random", "random_16-7"],
        index=["24-7", "16-7", "random", "random_16-7"].index(base_cfg.load_type),
    )
    reliability_target_pct = st.slider(
        "Reliability target (% load served)",
        min_value=80.0, max_value=100.0,
        value=float(seed_reliability), step=0.1,
        help="The screener recommends the cheapest configuration whose firmness is at least this high.",
    )
    grid_allowed = st.checkbox(
        "Allow grid imports / exports",
        value=bool(seed_grid_allowed),
        help="When on, the cost objective also subtracts merchant revenue (POI-limited).",
    )
    voll = st.number_input(
        "VOLL ($/MWh of unserved)",
        min_value=0.0, max_value=100_000.0,
        value=float(base_cfg.voll_usd_per_mwh), step=100.0,
        help="Value of Lost Load. Higher VOLL ⇒ the solver works harder (gas / batteries) to keep load served.",
    )

# ── Panel B: Economics ──────────────────────────────────────────────────────
with col_econ:
    st.subheader("B · Economics")
    with st.expander("Capex & O&M (screening-grade defaults)", expanded=False):
        def _econ_inputs(label: str, rec: ResourceEcon, *, include_energy: bool = False) -> ResourceEcon:
            st.markdown(f"**{label}**")
            c1, c2 = st.columns(2)
            capex_kw = c1.number_input(f"{label} capex ($/kW)", 0.0, 10_000.0, float(rec.capex_usd_per_kw), 10.0, key=f"{label}_capex_kw")
            fom = c2.number_input(f"{label} fixed O&M ($/kW-yr)", 0.0, 500.0, float(rec.fixed_om_usd_per_kw_yr), 1.0, key=f"{label}_fom")
            c3, c4 = st.columns(2)
            life = c3.number_input(f"{label} life (yrs)", 1.0, 60.0, float(rec.life_years), 1.0, key=f"{label}_life")
            rate = c4.number_input(f"{label} discount rate", 0.0, 0.5, float(rec.discount_rate), 0.005, format="%.3f", key=f"{label}_rate")
            capex_kwh = 0.0
            if include_energy:
                capex_kwh = st.number_input(f"{label} energy capex ($/kWh)", 0.0, 2_000.0, float(rec.capex_usd_per_kwh), 5.0, key=f"{label}_capex_kwh")
            return ResourceEcon(
                capex_usd_per_kw=capex_kw,
                capex_usd_per_kwh=capex_kwh,
                fixed_om_usd_per_kw_yr=fom,
                life_years=life,
                discount_rate=rate,
            )

        solar_econ = _econ_inputs("Solar", seed_econ.solar)
        wind_econ = _econ_inputs("Wind", seed_econ.wind)
        bess_econ = _econ_inputs("BESS", seed_econ.bess, include_energy=True)
        gas_econ = _econ_inputs("Gas", seed_econ.gas)
        carbon_price = st.number_input(
            "Carbon price ($/ton CO2)", 0.0, 1_000.0,
            float(seed_econ.carbon_price_usd_per_ton), 5.0,
        )
    with st.expander("Dispatch economics (VOLL, no-load, degradation)", expanded=False):
        gas_no_load = st.number_input(
            "Gas no-load cost ($/h committed)", 0.0, 100_000.0,
            float(base_cfg.gas_no_load_cost_usd_per_h), 10.0,
            help="Prevents the solver from keeping gas committed at zero output for free.",
        )
        bess_deg = st.number_input(
            "BESS degradation cost ($/MWh discharged)", 0.0, 500.0,
            float(base_cfg.bess_deg_cost_usd_per_mwh), 1.0,
        )
        gas_var_cost = st.number_input(
            "Gas variable cost ($/MWh)", 0.0, 1_000.0,
            float(base_cfg.gas_var_cost_usd_per_mwh), 1.0,
            help="Simple mode: $/MWh. Leave advanced heat-rate inputs on the Dispatch page.",
        )
    economics = EconAssumptions(
        solar=solar_econ, wind=wind_econ, bess=bess_econ, gas=gas_econ,
        carbon_price_usd_per_ton=carbon_price,
    )

# ── Panel C: Build ranges ───────────────────────────────────────────────────
with col_build:
    st.subheader("C · Build ranges")
    st.caption(
        f"CSV renewables are interpreted as absolute MW from a reference farm. "
        f"Detected reference sizes — solar: {ref_solar_auto:.0f} MW, wind: {ref_wind_auto:.0f} MW."
    )

    def _range_inputs(
        label: str,
        default_values: list[float],
        *,
        units: str = "MW",
        reference_auto: float = 0.0,
        allow_multipliers: bool = False,
    ) -> ResourceRange:
        st.markdown(f"**{label}**")
        mode_options = ["absolute", "pinned"] + (["multipliers"] if allow_multipliers else [])
        mode = st.selectbox(
            f"{label} sweep mode",
            mode_options,
            index=0,
            key=f"{label}_mode",
            help=(
                "absolute: comma-separated MW values.\n"
                "pinned: single MW value, no sweep.\n"
                + ("multipliers: fraction of reference farm (renewables).\n" if allow_multipliers else "")
            ),
        )
        if mode == "pinned":
            val = st.number_input(f"{label} value ({units})", 0.0, 5000.0, float(default_values[0] if default_values else 0.0), 1.0, key=f"{label}_pin")
            values = [float(val)]
        else:
            text = st.text_input(
                f"{label} values ({units})",
                value=", ".join(f"{v:g}" for v in default_values),
                key=f"{label}_text",
                help="Comma or space separated. Each value is one sweep point.",
            )
            values = _parse_values(text, default_values)
        reference_mw: Optional[float] = None
        if allow_multipliers and mode == "multipliers":
            reference_mw = st.number_input(
                f"{label} reference MW",
                0.0, 5000.0, float(reference_auto), 1.0,
                key=f"{label}_ref",
            )
        elif allow_multipliers:
            reference_mw = float(reference_auto)
        return ResourceRange(
            name={  # type: ignore[arg-type]
                "Solar": "solar", "Wind": "wind",
                "BESS power": "bess_power", "BESS duration": "bess_duration",
                "Gas": "gas",
            }[label],
            mode=mode,  # type: ignore[arg-type]
            values=values,
            reference_mw=reference_mw,
        )

    # Pre-fill from seed_ranges if present, else sensible defaults.
    def _seed_vals(name: str, fallback: list[float]) -> list[float]:
        if seed_ranges is None:
            return fallback
        r = getattr(seed_ranges, name)
        return [float(v) for v in (r.values or fallback)]

    solar_range = _range_inputs(
        "Solar",
        _seed_vals("solar", [0.0, max(50.0, ref_solar_auto), max(100.0, 2 * ref_solar_auto)]),
        reference_auto=ref_solar_auto, allow_multipliers=True,
    )
    wind_range = _range_inputs(
        "Wind",
        _seed_vals("wind", [0.0, max(50.0, ref_wind_auto), max(100.0, 2 * ref_wind_auto)]),
        reference_auto=ref_wind_auto, allow_multipliers=True,
    )
    bess_power_range = _range_inputs(
        "BESS power",
        _seed_vals("bess_power", [25.0, 50.0, 100.0]),
    )
    bess_duration_range = _range_inputs(
        "BESS duration",
        _seed_vals("bess_duration", [2.0, 4.0, 8.0]),
        units="hours",
    )
    gas_range = _range_inputs(
        "Gas",
        _seed_vals("gas", [0.0, 25.0, 50.0]),
    )

ranges = ScreeningRanges(
    solar=solar_range,
    wind=wind_range,
    bess_power=bess_power_range,
    bess_duration=bess_duration_range,
    gas=gas_range,
)

# Finalise the base RunConfig with the user's dispatch-economics overrides
base_cfg = base_cfg.model_copy(update={
    "load_mw": load_mw,
    "load_type": load_type,
    "voll_usd_per_mwh": voll,
    "gas_no_load_cost_usd_per_h": gas_no_load,
    "bess_deg_cost_usd_per_mwh": bess_deg,
    "gas_var_cost_usd_per_mwh": gas_var_cost,
    "carbon_price_usd_per_ton": carbon_price,
})

# Grid-size guardrail ─ warn above a threshold, block above the hard cap.
grid_size = ranges.grid_size()
if grid_size > 300:
    st.warning(
        f"You are about to sweep **{grid_size}** configurations. "
        "Each config runs one LP-relaxed dispatch; consider reducing the sweep values "
        "per resource to keep runtime reasonable."
    )
st.info(f"Grid size: **{grid_size}** configurations.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run + Save
# ─────────────────────────────────────────────────────────────────────────────

run_col, save_col = st.columns([1, 1])
with run_col:
    run_btn = st.button("🚀 Run screening", type="primary", use_container_width=True)
    validate_milp = st.checkbox("Validate recommendation with full MILP", value=True)
with save_col:
    if st.button("💾 Save current scenario", use_container_width=True):
        out_bundle = ScenarioBundle(
            name="user_scenario",
            description="Saved from the Configuration Recommendation page.",
            run_config=base_cfg,
            econ=economics,
            screening_ranges=ScreeningRangesSpec.from_ranges(ranges),
            reliability_target_pct=reliability_target_pct,
            grid_allowed=grid_allowed,
        )
        st.download_button(
            "⬇️ Download scenario JSON",
            data=out_bundle.model_dump_json(indent=2).encode(),
            file_name="user_scenario.json",
            mime="application/json",
        )

# ─────────────────────────────────────────────────────────────────────────────
# 4. Screening execution + output tabs
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    progress = st.progress(0.0, text="Starting sweep…")

    def _progress_cb(idx: int, total: int, candidate: dict) -> None:
        if total <= 0:
            return
        pct = (idx + 1) / total
        progress.progress(
            min(1.0, pct),
            text=(
                f"Config {idx+1}/{total}  "
                f"solar={candidate.get('solar_mw',0):.0f} wind={candidate.get('wind_mw',0):.0f} "
                f"bess={candidate.get('bess_power_mw',0):.0f}×{candidate.get('bess_duration_h',0):.0f}h "
                f"gas={candidate.get('gas_mw',0):.0f}"
            ),
        )

    t0 = time.time()
    try:
        result = screen(
            df, base_cfg, ranges, economics,
            reliability_target_pct=reliability_target_pct,
            grid_allowed=grid_allowed,
            relax_uc=True,
            validate_recommended=validate_milp,
            progress_cb=_progress_cb,
        )
    except Exception as e:
        st.error(f"Screening failed: {e}")
        st.stop()
    progress.empty()
    st.success(f"Done. Tested {len(result.sweep_df)} configurations in {time.time()-t0:.1f}s.")

    # Persist last run to session for post-run drill-downs.
    st.session_state["_screen_result"] = result

# If we have a result (either just run or held in session), render the tabs.
result = st.session_state.get("_screen_result")
if result is not None:
    tab_rec, tab_all, tab_drill = st.tabs(["Recommendation", "All configurations", "Dispatch drill-down"])

    # ── Tab: Recommendation + Pareto ──────────────────────────────────────
    with tab_rec:
        rec = result.recommended or {}
        cfg = rec.get("config") or {}
        if cfg:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Firmness", f"{float(cfg.get('firmness_pct',0)):.2f}%")
            c2.metric("Total annual cost", f"${float(cfg.get('total_annual_cost_usd',0)):,.0f}/yr")
            c3.metric("LCOE", f"${float(cfg.get('lcoe_usd_per_mwh',0)):.0f}/MWh")
            c4.metric("CO2 (horizon tons)", f"{float(cfg.get('co2_tons',0)):.1f}")

            st.markdown(f"**{rec.get('message','')}**")
            st.write({
                "Solar MW": float(cfg.get("solar_mw", 0)),
                "Wind MW": float(cfg.get("wind_mw", 0)),
                "BESS MW": float(cfg.get("bess_power_mw", 0)),
                "BESS hours": float(cfg.get("bess_duration_h", 0)),
                "BESS MWh": float(cfg.get("bess_energy_mwh", 0)),
                "Gas MW": float(cfg.get("gas_mw", 0)),
            })

            if not rec.get("meets_reliability", False):
                st.warning(
                    "No tested configuration met the reliability target. "
                    "Widen the sweep ranges (more gas / longer-duration BESS / larger renewables) and re-run."
                )

            if result.validation is not None and result.validation.get("ok"):
                v_mets = result.validation.get("metrics", {}) or {}
                st.caption(
                    f"Full-MILP validation: firmness {float(v_mets.get('firmness (%)',0)):.2f}%, "
                    f"total op cost ${float(v_mets.get('total_operating_cost_usd',0)):,.0f}, "
                    f"CO2 {float(v_mets.get('co2_tons',0)):.1f} tons."
                )

        # Pareto plot (firmness vs. total annual cost)
        if px is not None and not result.ranked_df.empty:
            fig = px.scatter(
                result.ranked_df,
                x="firmness_pct", y="total_annual_cost_usd",
                color="is_pareto", symbol="meets_reliability",
                hover_data=["solar_mw", "wind_mw", "bess_power_mw", "bess_duration_h", "gas_mw", "lcoe_usd_per_mwh"],
                title="Firmness vs. total annual cost",
            )
            fig.update_layout(yaxis_title="Total annual cost ($/yr)", xaxis_title="Firmness (%)")
            st.plotly_chart(fig, use_container_width=True)
        elif not result.ranked_df.empty:
            st.line_chart(result.ranked_df.set_index("firmness_pct")["total_annual_cost_usd"])

    # ── Tab: All configurations ──────────────────────────────────────────
    with tab_all:
        st.dataframe(result.ranked_df, use_container_width=True)
        st.download_button(
            "⬇️ Ranked configs CSV",
            result.ranked_df.to_csv(index=False).encode(),
            "screening_ranked.csv",
        )
        st.download_button(
            "⬇️ Full sweep CSV",
            result.sweep_df.to_csv(index=False).encode(),
            "screening_sweep.csv",
        )

    # ── Tab: Dispatch drill-down ─────────────────────────────────────────
    with tab_drill:
        if result.validation is not None and result.validation.get("ok"):
            dispatch_df = result.validation.get("dispatch_df")
            v_cfg = result.validation.get("run_cfg")
            if dispatch_df is not None and v_cfg is not None:
                st.subheader("Recommended configuration — full MILP dispatch")
                try:
                    fig = reporting.plot_dispatch(dispatch_df, v_cfg, title="")
                    st.pyplot(fig)
                    fig = reporting.plot_soc(dispatch_df, v_cfg, title="Battery State of Charge")
                    st.pyplot(fig)
                except Exception as e:  # pragma: no cover
                    st.warning(f"Could not render dispatch plot: {e}")
                st.dataframe(dispatch_df.head(200), use_container_width=True)
                st.download_button(
                    "⬇️ Dispatch CSV",
                    dispatch_df.to_csv().encode(),
                    "recommended_dispatch.csv",
                )
            else:
                st.info("Validation ran but no dispatch dataframe was returned.")
        else:
            st.info(
                "Enable *Validate recommendation with full MILP* before running "
                "to see a dispatch drill-down for the recommended configuration."
            )
else:
    st.info("Choose a scenario (or tune the three panels) and click **Run screening**.")
