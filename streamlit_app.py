"""
streamlit_app.py
----------------
Streamlit web application for interactive battery dispatch optimization.
Allows users to upload data, configure battery and grid parameters, run optimizations, and visualize results.
"""

from pathlib import Path
import io, zipfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# from dispatch_core.optimize import tradeoff_analysis  # No longer needed as direct import

# Import core modules in safe order: config FIRST
import dispatch_core.config    as cfg
import dispatch_core.data_io   as data_io
import dispatch_core.optimize  as optimize
import dispatch_core.reporting as reporting

import datetime
from typing import cast, Literal
import time

# Helper to robustly extract date from index min/max
def get_date_safe(val):
    import pandas as pd
    import datetime
    if pd.isnull(val) or str(val) == 'NaT':
        return datetime.date(2015, 1, 1)  # fallback
    if isinstance(val, pd.Timestamp):
        return val.date()
    if isinstance(val, datetime.datetime):
        return val.date()
    if isinstance(val, datetime.date):
        return val
    try:
        d = pd.Timestamp(val)
        if pd.isnull(d) or str(d) == 'NaT':
            return datetime.date(2015, 1, 1)
        return d.date()
    except Exception:
        return datetime.date(2015, 1, 1)

# ── Page set-up ───────────────────────────────────────────────
st.set_page_config(page_title="Battery Dispatch", layout="wide")
st.title("Battery Dispatch Optimiser")

# ── Sidebar: Data source ─────────────────────────────────────
st.sidebar.header("1 · Data source (upload or synthetic)")
# Downloadable template (static file)
template_path = "template_may2015.csv"
with open(template_path, "rb") as f:
    st.download_button(
        label="Download example/template file (May 2015)",
        data=f,
        file_name="template_may2015.csv",
        mime="text/csv"
    )

uploaded_file = st.sidebar.file_uploader(
    "Upload your time series data (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="Required columns: Datetime, Load (MW), Market Price ($/MWh). Optional: Wind (MW), Solar (MW), NatGas (MW). Download the template above."
)

# Help section for data format
with st.sidebar.expander("? Data format help"):
    st.markdown("""
    **Required columns:**
    - `Datetime` (parseable, e.g. 2023-01-01 00:00)
    - `Load (MW)`
    - `Market Price ($/MWh)`
    **Optional columns:**
    - `Wind (MW)`, `Solar (MW)`, `NatGas (MW)` (filled with zeros if missing)
    **File types:** CSV, Excel (.xlsx)
    **How missing data is handled:**
    - Missing required columns: error, must fix
    - Missing optional columns: filled with zeros
    - Missing values in required columns: error
    - Missing values in optional columns: filled with zeros
    - Unparseable datetimes: row dropped, warning shown
    """)

# Data handling logic
user_df = None
user_data_source = "synthetic"
d_from, d_to = None, None  # Will be set after date selection
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_up = pd.read_csv(uploaded_file)
        else:
            df_up = pd.read_excel(uploaded_file, engine="openpyxl")
        # Validate required columns
        required_cols = ["Datetime", "Load (MW)", "Market Price ($/MWh)"]
        optional_cols = ["Wind (MW)", "Solar (MW)", "NatGas (MW)"]
        missing_required = [col for col in required_cols if col not in df_up.columns]
        if missing_required:
            st.error(f"Missing required column(s): {missing_required}. Please check your file.")
        else:
            # Parse datetime and drop unparseable rows
            df_up["Datetime"] = pd.to_datetime(df_up["Datetime"], errors="coerce")
            n_before = len(df_up)
            df_up = df_up.dropna(subset=["Datetime"])
            n_after = len(df_up)
            if n_after < n_before:
                st.warning(f"Dropped {n_before - n_after} rows with unparseable datetimes.")
            df_up = df_up.set_index("Datetime")
            # Ensure index is DatetimeIndex
            if not isinstance(df_up.index, pd.DatetimeIndex):
                df_up.index = pd.to_datetime(df_up.index)
            # Fill missing optional columns with zeros
            filled_cols = []
            for col in optional_cols:
                if col not in df_up.columns:
                    df_up[col] = 0.0
                    filled_cols.append(col)
            for col in optional_cols:
                df_up[col] = df_up[col].fillna(0.0)
            # Check for missing values in required columns (excluding Datetime)
            req_cols_no_dt = [col for col in required_cols if col != "Datetime"]
            missing_vals = df_up[req_cols_no_dt].isnull().any()
            if isinstance(missing_vals, bool):
                missing_cols = req_cols_no_dt if missing_vals else []
            else:
                import pandas as pd
                missing_vals = pd.Series(missing_vals, index=req_cols_no_dt)
                missing_cols = list(missing_vals[missing_vals].index)
            if len(missing_cols) > 0:
                st.error(f"Missing values in required columns: {missing_cols}. Please fix your file.")
            else:
                import pandas as pd
                user_df = pd.DataFrame(df_up)
                user_data_source = "uploaded"
                # Ensure index is DatetimeIndex
                if not isinstance(user_df.index, pd.DatetimeIndex):
                    user_df.index = pd.to_datetime(user_df.index)
                min_date = get_date_safe(user_df.index.min())
                max_date = get_date_safe(user_df.index.max())
                # Ensure min_date and max_date are datetime.date
                import datetime
                if not isinstance(min_date, datetime.date):
                    min_date = datetime.date(2015, 1, 1)
                if not isinstance(max_date, datetime.date):
                    max_date = datetime.date(2015, 1, 1)
                selected_dates = st.sidebar.date_input(
                    "Date window for analysis",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                    d_from, d_to = selected_dates
                else:
                    st.error("Please select a valid start and end date.")
                    st.stop()
                mask = (user_df.index >= pd.to_datetime(d_from)) & (user_df.index <= pd.to_datetime(d_to))
                user_df = user_df.loc[mask]
                st.success("File uploaded and parsed successfully.")
                st.dataframe(user_df.head(), use_container_width=True)
                if filled_cols:
                    st.info(f"Filled missing optional columns with zeros: {filled_cols}")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    # If no file uploaded, use the static template as default data
    template_path = "template_may2015.csv"
    user_df = pd.read_csv(template_path, parse_dates=["Datetime"])
    user_df = pd.DataFrame(user_df)
    user_df = user_df.set_index("Datetime")
    # Ensure index is DatetimeIndex
    if not isinstance(user_df.index, pd.DatetimeIndex):
        user_df.index = pd.to_datetime(user_df.index)
    user_data_source = "template"
    min_date = get_date_safe(user_df.index.min())
    max_date = get_date_safe(user_df.index.max())
    # Ensure min_date and max_date are datetime.date
    import datetime
    if not isinstance(min_date, datetime.date):
        min_date = datetime.date(2015, 1, 1)
    if not isinstance(max_date, datetime.date):
        max_date = datetime.date(2015, 1, 1)
    selected_dates = st.sidebar.date_input(
        "Date window for analysis",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        d_from, d_to = selected_dates
    else:
        st.error("Please select a valid start and end date.")
        st.stop()
    mask = (user_df.index >= pd.to_datetime(d_from)) & (user_df.index <= pd.to_datetime(d_to))
    user_df = user_df.loc[mask]

# Warn if market price values look like $/kWh
if user_df is not None and "Market Price ($/MWh)" in user_df.columns:
    if user_df["Market Price ($/MWh)"].max() < 10:
        st.warning("Market price values look very low. Are you using $/kWh instead of $/MWh? All calculations assume $/MWh.")

# Helper for data and source
def get_data_and_source():
    """Return the user DataFrame and data source type."""
    if user_df is not None:
        return user_df, "uploaded"
    else:
        return None, "synthetic"

# Sidebar: load profile selection
if uploaded_file is not None:
    load_source_default = 0  # Default to 'Uploaded file' if user uploaded
else:
    load_source_default = 1  # Default to 'Synthetic profile' if using template

if user_df is not None:
    load_source = st.sidebar.radio(
        "Select load data source",
        options=["Uploaded file", "Synthetic profile"],
        index=load_source_default,
        help="Choose whether to use the load curve from your uploaded file or generate a synthetic load profile."
    )
else:
    st.sidebar.info("No file uploaded. Using synthetic load profile.")
    load_source = "Synthetic profile"

# ── Sidebar: load profile ────────────────────────────────────
st.sidebar.header("2 · Load profile")
load_mw   = st.sidebar.number_input("Average load\u00a0MW", 0.0, 2000.0, 150.0, 10.0)
load_type_options = [
    ("24-7", "24-7"),
    ("16-7", "16-7"),
    ("random", "random - 24-7"),
    ("random_16-7", "random - 16-7")
]
load_type_labels = [label for _, label in load_type_options]
load_type_values = [value for value, _ in load_type_options]
load_type = st.sidebar.selectbox(
    "Shape",
    load_type_labels,
    index=0,
    help="Random modes use 'Load–std' as \u03c3"
)
load_type_value = load_type_values[load_type_labels.index(load_type)]

if load_type_value.startswith("random"):
    load_std  = st.sidebar.number_input("Load–std (MW) for random modes", 0.0, 500.0, 15.0, 1.0)
else:
    load_std = 15.0

# ── Sidebar: battery specs ───────────────────────────────────
st.sidebar.header("3 · Battery stacks")
batt_power = st.sidebar.number_input("Battery 1 Power MW", 0.0, 2000.0, 200.0, 10.0)
batt_dur   = st.sidebar.number_input("Battery 1 Duration h", 0.5, 1000.0, 4.0, 0.5)
batt_rte   = st.sidebar.number_input("Battery 1 RTE", 0.5, 1.0, 0.86, 0.01)
batt2_power = st.sidebar.number_input("Battery 2 Power MW", 0.0, 2000.0, 0.0, 10.0)
batt2_dur   = st.sidebar.number_input("Battery 2 Duration h", 0.0, 1000.0, 0.0, 0.5)
batt2_rte   = st.sidebar.number_input("Battery 2 RTE", 0.5, 1.0, 0.86, 0.01)

# ── Sidebar: POI and grid flag ───────────────────────────────
st.sidebar.header("4 · POI / Grid")
poi_limit  = st.sidebar.number_input("POI limit MW", 1.0, 5000.0, 250.0, 10.0)

grid_on = st.sidebar.toggle(
    "Allow grid import/export",
    value=False,
    help="If ON, all load is served and the optimizer maximizes net revenue. If OFF, the optimizer maximizes load served (resilience)."
)

# ── Sidebar: optimisation objective ──────────────────────────
st.sidebar.header("5 · Optimisation")
if grid_on:
    st.sidebar.info("**Objective:** Maximize net revenue (all load is served)")
else:
    st.sidebar.info("**Objective:** Maximize load served (Resilience)")

tradeoff_toggle = False
if not grid_on:
    tradeoff_toggle = st.sidebar.checkbox(
        "Show trade-off analysis (resilience vs. revenue)",
        value=False,
        help="Explore the trade-off between resilience and revenue (only available when grid is OFF)."
    )

run = st.sidebar.button("🚀 Run optimisation")

# If using synthetic data, keep date window selection
if user_data_source == "synthetic":
    _date_default = (datetime.date(2015, 5, 1), datetime.date(2015, 5, 31))
    min_date = _date_default[0]
    max_date = _date_default[1]
    selected_dates = st.sidebar.date_input(
        "Date window for analysis",
        value=_date_default,
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        d_from, d_to = selected_dates
    else:
        st.error("Please select a valid start and end date.")
        st.stop()
    # Hardcoded Excel path and sheet name for synthetic fallback
    excel_path = r"C:\Users\1010013\OneDrive - Deriva Energy\Documents\python_file_100.xlxs.xlsx"
    sheet = "Combined"
else:
    # d_from and d_to already set above for uploaded/template
    pass

# Always build config and load data before main workflow
try:
    if user_data_source in ["uploaded", "template"] and user_df is not None:
        df = user_df.copy()
        # Ensure df is a pandas DataFrame before accessing .index
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        idx = pd.to_datetime(df.index)
        d_from = idx.min()
        d_to = idx.max()
        if not isinstance(d_from, pd.Timestamp):
            d_from = pd.Timestamp(str(d_from))
        if not isinstance(d_to, pd.Timestamp):
            d_to = pd.Timestamp(str(d_to))
        d_from = d_from.date()
        d_to = d_to.date()
        from pathlib import Path
        if load_source.strip().lower() == "synthetic profile":
            # Use sidebar values for synthetic config
            run_cfg = cfg.RunConfig(
                path           = Path("template_may2015.csv" if user_data_source == "template" else "uploaded.csv"),
                sheet_name     = user_data_source,
                start_date     = str(d_from),
                end_date       = str(d_to),
                load_mw        = load_mw,
                load_std       = load_std,
                load_type      = cast(Literal['24-7', '16-7', 'random', 'random_16-7'], load_type_value),
                battery_power_mw   = batt_power,
                battery_duration_h = batt_dur,
                battery_count      = 1,
                rte                = batt_rte,
                battery2_power_mw     = batt2_power,
                battery2_duration_h   = batt2_dur,
                battery2_rte          = batt2_rte,
                poi_limit_mw       = poi_limit,
                market_price_col   = "Market Price ($/MWh)",
            )
            from dispatch_core.profiles import generate
            valid_types = ["24-7", "16-7", "random", "random_16-7"]
            if run_cfg.load_type not in valid_types:
                st.error(f"Invalid load_type for synthetic profile: {run_cfg.load_type}")
            else:
                df["Load (MW)"] = generate(pd.DatetimeIndex(df.index), run_cfg)
                user_df["Load (MW)"] = df["Load (MW)"]
        else:
            run_cfg = cfg.RunConfig(
                path           = Path("template_may2015.csv" if user_data_source == "template" else "uploaded.csv"),
                sheet_name     = user_data_source,
                start_date     = str(d_from),
                end_date       = str(d_to),
                load_mw        = float(df["Load (MW)"].mean()),
                load_std       = 0.0,
                load_type      = cast(Literal['24-7', '16-7', 'random', 'random_16-7'], load_type_value),
                battery_power_mw   = batt_power,
                battery_duration_h = batt_dur,
                battery_count      = 1,
                rte                = batt_rte,
                battery2_power_mw     = batt2_power,
                battery2_duration_h   = batt2_dur,
                battery2_rte          = batt2_rte,
                poi_limit_mw       = poi_limit,
                market_price_col   = "Market Price ($/MWh)",
            )
        # === FIX: Use user_df directly for uploaded files, otherwise load from disk ===
        if user_data_source == "uploaded":
            df = user_df.copy()
        else:
            df = data_io.load_data(run_cfg)
        # Only override with synthetic load if user selects 'Synthetic profile'
        from dispatch_core.profiles import generate
        valid_types = ["24-7", "16-7", "random", "random_16-7"]
        if load_source.strip().lower() == "synthetic profile":
            if run_cfg.load_type not in valid_types:
                st.error(f"Invalid load_type for synthetic profile: {run_cfg.load_type}")
            else:
                df["Load (MW)"] = generate(pd.DatetimeIndex(df.index), run_cfg)
                run_cfg.load_mw = float(df["Load (MW)"].mean())
except Exception as e:
    st.error(f"⚠️ Config or data load failed: {e}")
    run_cfg = None
    df = None

# ── Main workflow ────────────────────────────────────────────
if run and df is not None and run_cfg is not None:
    if grid_on:
        st.subheader("Grid ON: Maximize Revenue with 100% Load Served")
        start_time = time.time()
        st.markdown(r"""
        **All load is served (100% resilience).**
        The optimizer maximizes net revenue:
        $$
        \text{Revenue} = \sum_t \left( \text{Grid Export}_t \times \text{Price}_t - \text{Grid Import}_t \times \text{Price}_t \right)
        $$
        (Units: $)
        """)
        try:
            res, mets = optimize.run_lp(
                df,
                run_cfg,
                mode          = "grid_on_max_revenue",
                blend_lambda  = 1.0,  # not used
                grid_allowed  = True
            )
        except Exception as e:
            st.error(f"⚠️ Optimiser crashed: {e}")
            st.stop()
        elapsed = time.time() - start_time
        st.markdown(f"**Optimization time:** {elapsed:.2f} seconds", unsafe_allow_html=True)
        st.dataframe(pd.Series(mets).to_frame("Value").T, use_container_width=True)
        st.subheader("Dispatch overview")
        fig = reporting.plot_dispatch(res, run_cfg, title="Dispatch (Grid ON)")
        st.pyplot(fig)
        st.subheader("Interactive Dispatch Dashboard")
        fig_plotly = reporting.plot_dispatch_plotly(res, run_cfg, title="Dispatch (All Metrics)")
        st.plotly_chart(fig_plotly, use_container_width=True)
        st.subheader("Grid Import/Export and Revenue Time Series")
        revenue_ts = (res["grid_exp"] * res["price"] - res["grid_imp"] * res["price"]) / 1000  # $/h
        res["revenue_t"] = revenue_ts
        st.line_chart(revenue_ts, use_container_width=True)
        st.markdown(f"**Total Revenue:** ${revenue_ts.sum():,.2f}")
        st.subheader("Battery State of Charge")
        fig = reporting.plot_soc(res, run_cfg, title="Battery State of Charge")
        st.pyplot(fig)
        st.subheader("Clipped Energy")
        fig = reporting.plot_clipped(res, run_cfg, title="Clipped Energy")
        st.pyplot(fig)
        st.subheader("Cumulative Revenue")
        fig = reporting.plot_revenue(res, run_cfg, title="Revenue Over Time")
        st.pyplot(fig)
        st.markdown(f"**Cumulative Revenue (final):** ${revenue_ts.cumsum().iloc[-1]:,.2f}")
        with st.expander("Show entire dispatch DataFrame"):
            st.dataframe(res, height=400, use_container_width=True)
        case_label = f"dispatch_{'gridON' if grid_on else 'gridOFF'}_{str(d_from)}_{str(d_to)}.csv"
        csv = res.to_csv().encode()
        st.download_button("⬇️ Time‑series CSV", csv, case_label)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metrics.csv", pd.Series(mets).to_csv(header=False))
            zf.writestr("dispatch.csv", csv)
        st.download_button("⬇️ Everything (ZIP)", buf.getvalue(), "dispatch_results.zip")
    else:
        st.subheader("Resilience Mode: Maximize Load Served")
        start_time = time.time()
        try:
            res, mets = optimize.run_lp(
                df,
                run_cfg,
                mode          = "resilience",
                blend_lambda  = 1.0,
                grid_allowed  = grid_on
            )
        except Exception as e:
            st.error(f"⚠️ Optimiser crashed: {e}")
            st.stop()
        elapsed = time.time() - start_time
        st.markdown(f"**Optimization time:** {elapsed:.2f} seconds", unsafe_allow_html=True)
        st.dataframe(pd.Series(mets).to_frame("Value").T, use_container_width=True)
        st.subheader("Dispatch overview")
        fig = reporting.plot_dispatch(res, run_cfg, title="")
        st.pyplot(fig)
        st.subheader("Battery State of Charge")
        fig = reporting.plot_soc(res, run_cfg, title="Battery State of Charge")
        st.pyplot(fig)
        st.subheader("Clipped Energy")
        fig = reporting.plot_clipped(res, run_cfg, title="Clipped Energy")
        st.pyplot(fig)
        if grid_on:
            st.subheader("Grid Charging/Discharging")
            fig = reporting.plot_grid(res, run_cfg, title="Grid Charging/Discharging")
            st.pyplot(fig)
        st.subheader("Cumulative Revenue")
        fig = reporting.plot_revenue(res, run_cfg, title="Revenue Over Time")
        st.pyplot(fig)
        with st.expander("Show entire dispatch DataFrame"):
            st.dataframe(res, height=400, use_container_width=True)
        case_label = f"dispatch_{'gridON' if grid_on else 'gridOFF'}_{str(d_from)}_{str(d_to)}.csv"
        csv = res.to_csv().encode()
        st.download_button("⬇️ Time‑series CSV", csv, case_label)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metrics.csv", pd.Series(mets).to_csv(header=False))
            zf.writestr("dispatch.csv", csv)
        st.download_button("⬇️ Everything (ZIP)", buf.getvalue(), "dispatch_results.zip")

        # Trade-off analysis (only if grid is OFF and toggle is enabled)
        if not grid_on and tradeoff_toggle:
            st.subheader("Resilience vs Revenue Trade-off Analysis")
            with st.spinner("Running trade-off analysis..."):
                results_df, dispatch_dict = optimize.tradeoff_analysis(df, run_cfg)
            st.dataframe(results_df, use_container_width=True)
            fig, ax = plt.subplots()
            ax.plot(results_df['resilience_%'], results_df['revenue_$'], marker='o', label='Trade-off Curve')
            ax.set_xlabel('Resilience (%)')
            ax.set_ylabel('Revenue ($)')
            ax.set_title('Resilience vs Revenue for various slack levels')
            ax.grid(True)
            # Highlight the knee of the curve (max revenue increase per resilience loss)
            diffs = results_df['revenue_$'].diff().fillna(0) / results_df['resilience_%'].diff().fillna(1)
            knee_idx = diffs.abs().idxmax()
            ax.plot(results_df['resilience_%'][knee_idx], results_df['revenue_$'][knee_idx], 'ro', label='Knee (max trade-off)')
            ax.legend()
            st.pyplot(fig)
            st.markdown("**Knee of the curve:** Where a small loss in resilience gives a big gain in revenue.")
            st.write(f"Knee at slack = {results_df['slack_%'][knee_idx]}%: Resilience = {results_df['resilience_%'][knee_idx]:.2f}%, Revenue = ${results_df['revenue_$'][knee_idx]:,.2f}")
            st.markdown("""
            **How is revenue calculated?**  
            For each slack value, the optimizer maximizes the sum over all timesteps of:
            $\text{Revenue} = \sum_t \text{Market Price}_t \times (\text{Generation}_t + \text{Discharge}_t - \text{Charge}_t)$
            This represents the net export to the grid (positive for export, negative for import) times the market price at each timestep.
            """)
            st.download_button("Download trade-off table (CSV)", results_df.to_csv(index=False), "tradeoff_table.csv")
            slack_options = results_df['slack_%'].tolist()
            selected_slack = st.selectbox("Select slack % to view dispatch", slack_options, index=0, help="Choose a slack value to see the dispatch profile for that scenario.")
            st.subheader(f"Dispatch for slack = {selected_slack}%")
            if selected_slack is not None:
                dispatch_df = dispatch_dict[float(selected_slack)/100.0]
                st.dataframe(dispatch_df, use_container_width=True)
                st.download_button(f"Download dispatch for slack {selected_slack}% (CSV)", dispatch_df.to_csv(), f"dispatch_slack_{selected_slack}.csv")
