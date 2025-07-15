import matplotlib.pyplot as plt
import pandas as pd
from .config import RunConfig
import plotly.graph_objs as go
import numpy as np

def plot_dispatch(res: pd.DataFrame, cfg: RunConfig, title="Dispatch"):
    t = res.index
    # Extract individual sources
    solar = res["Solar (MW)"] if "Solar (MW)" in res else pd.Series(0, index=t)
    wind = res["Wind (MW)"] if "Wind (MW)" in res else pd.Series(0, index=t)
    battery1 = res["discharge1"] - res["charge1"] if "discharge1" in res and "charge1" in res else pd.Series(0, index=t)
    battery2 = res["discharge2"] - res["charge2"] if "discharge2" in res and "charge2" in res else pd.Series(0, index=t)
    grid = res["grid_exp"] - res["grid_imp"] if "grid_exp" in res and "grid_imp" in res else pd.Series(0, index=t)
    POI = getattr(cfg, 'poi_limit_mw', None)
    # Stacked area plot for renewables and batteries only
    fig, ax = plt.subplots(figsize=(14,5))
    ax.stackplot(
        t,
        solar,
        wind,
        battery1,
        battery2,
        labels=["Solar", "Wind", "Battery 1", "Battery 2"],
        colors=["yellow", "orange", "blue", "green"],
        alpha=0.7
    )
    # Plot grid import/export as a line
    ax.plot(t, grid, color="brown", label="Grid import/export", linewidth=1.5)
    # Plot POI limits if available
    if POI is not None and POI > 0:
        ax.axhline(POI, color='red', linestyle='--', label='POI limit')
        ax.axhline(-POI, color='red', linestyle='--')
    ax.plot(t, res["load"], color="black", linewidth=1, label="Load")
    ax.legend(loc="upper left")
    ax.set_xlabel("Time"); ax.set_ylabel("Power (MW)")
    ax.set_title(title); fig.tight_layout()
    return fig

def plot_soc(res: pd.DataFrame, cfg: RunConfig, title="Battery State of Charge"):
    t = res.index
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(t, res["soc1"], label="Battery 1 SOC", color="blue")
    ax.plot(t, res["soc2"], label="Battery 2 SOC", color="green")
    ax.set_xlabel("Time"); ax.set_ylabel("State of Charge (MWh)")
    ax.set_title(title)
    ax.legend(); fig.tight_layout()
    return fig

def plot_clipped(res: pd.DataFrame, cfg: RunConfig, title="Clipped Energy"):
    t = res.index
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(t, res["clipped"], label="Clipped Energy", color="red")
    ax.set_xlabel("Time"); ax.set_ylabel("Clipped (MW)")
    ax.set_title(title)
    ax.legend(); fig.tight_layout()
    return fig

def plot_grid(res: pd.DataFrame, cfg: RunConfig, title="Grid Charging/Discharging"):
    t = res.index
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(t, res["grid_imp"], label="Grid Import (Charging)", color="brown")
    ax.plot(t, res["grid_exp"], label="Grid Export (Discharging)", color="orange")
    ax.set_xlabel("Time"); ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.legend(); fig.tight_layout()
    return fig

def plot_revenue(res: pd.DataFrame, cfg: RunConfig, title="Revenue Over Time"):
    t = res.index
    if "price" in res:
        revenue = (res["grid_exp"] - res["grid_imp"]) * res["price"]  # $/h
        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(t, revenue.cumsum(), label="Cumulative Revenue ($)", color="purple")
        ax.set_xlabel("Time"); ax.set_ylabel("Cumulative Revenue ($)")
        ax.set_title(title)
        # Overlay market price on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(t, res["price"], color="orange", label="Market Price ($/MWh)", linewidth=1.2, alpha=0.7)
        ax2.set_ylabel("Market Price ($/MWh)", color="orange")
        ax2.tick_params(axis='y', labelcolor="orange")
        # Add legends for both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper left")
        fig.tight_layout()
        return fig

def plot_dispatch_plotly(res, cfg, title="Dispatch (All Metrics)"):
    t = res.index
    fig = go.Figure()
    # Add traces for each available metric
    if "Solar (MW)" in res:
        fig.add_trace(go.Scatter(x=t, y=res["Solar (MW)"], mode='lines', name='Solar', line=dict(color='yellow')))
    if "Wind (MW)" in res:
        fig.add_trace(go.Scatter(x=t, y=res["Wind (MW)"], mode='lines', name='Wind', line=dict(color='orange')))
    if "discharge1" in res and "charge1" in res:
        fig.add_trace(go.Scatter(x=t, y=res["discharge1"] - res["charge1"], mode='lines', name='Battery 1', line=dict(color='blue')))
    if "discharge2" in res and "charge2" in res:
        fig.add_trace(go.Scatter(x=t, y=res["discharge2"] - res["charge2"], mode='lines', name='Battery 2', line=dict(color='green')))
    if "grid_exp" in res and "grid_imp" in res:
        fig.add_trace(go.Scatter(x=t, y=res["grid_exp"] - res["grid_imp"], mode='lines', name='Grid import/export', line=dict(color='brown')))
    if "load" in res:
        fig.add_trace(go.Scatter(x=t, y=res["load"], mode='lines', name='Load', line=dict(color='black')))
    if "price" in res:
        fig.add_trace(go.Scatter(x=t, y=res["price"], mode='lines', name='Market Price ($/MWh)', line=dict(color='purple', dash='dot')))
    if "revenue_t" in res:
        fig.add_trace(go.Scatter(x=t, y=res["revenue_t"].cumsum(), mode='lines', name='Cumulative Revenue ($)', line=dict(color='magenta', dash='dash')))
    if "clipped" in res:
        fig.add_trace(go.Scatter(x=t, y=res["clipped"], mode='lines', name='Clipped', line=dict(color='red', dash='dot')))
    # Add POI limits if available
    POI = getattr(cfg, 'poi_limit_mw', None)
    if POI is not None and POI > 0:
        fig.add_trace(go.Scatter(x=t, y=[POI]*len(t), mode='lines', name='POI limit', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=t, y=[-POI]*len(t), mode='lines', name='POI limit', line=dict(color='red', dash='dash')))
    # Add any other time series columns
    for col in res.columns:
        if col not in ["Solar (MW)", "Wind (MW)", "discharge1", "charge1", "discharge2", "charge2", "grid_exp", "grid_imp", "load", "price", "revenue_t", "clipped"]:
            if np.issubdtype(res[col].dtype, np.number):
                fig.add_trace(go.Scatter(x=t, y=res[col], mode='lines', name=col))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value (various units)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
