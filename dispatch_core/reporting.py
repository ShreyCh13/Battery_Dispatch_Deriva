"""
reporting.py
------------
Provides plotting and reporting utilities for battery dispatch simulation results.
Includes matplotlib and plotly visualizations for dispatch, state of charge, clipped energy, grid flows, and revenue.
"""

import matplotlib.pyplot as plt
import pandas as pd
from .config import RunConfig
import plotly.graph_objs as go
import numpy as np
from matplotlib.figure import Figure

def plot_dispatch(res: pd.DataFrame, cfg: RunConfig, title: str = "Dispatch", show_grid: bool = True) -> Figure:
    """
    Plot stacked area chart of dispatch (solar, wind, battery 1, battery 2) and grid import/export.

    Args:
        res (pd.DataFrame): Dispatch results DataFrame.
        cfg (RunConfig): Run configuration object.
        title (str): Plot title.
        show_grid (bool): If True, plot grid import/export line. If False, omit it (for fixed schedule mode).

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    t = res.index
    # Extract individual sources
    solar = res["Solar (MW)"] if "Solar (MW)" in res else pd.Series(0, index=t)
    wind = res["Wind (MW)"] if "Wind (MW)" in res else pd.Series(0, index=t)
    natgas = res["NatGas (MW)"] if "NatGas (MW)" in res else pd.Series(0, index=t)
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
        natgas,
        battery1,
        battery2,
        labels=["Solar", "Wind", "NatGas", "Battery 1", "Battery 2"],
        colors=["yellow", "orange", "grey", "lightblue", "lightgreen"],
        alpha=0.6
    )
    # Plot POI limits if available
    if POI is not None and POI > 0:
        ax.axhline(POI, color='red', linestyle='--', label='POI limit')
    # Plot grid import/export as a line (always show)
    # ax.plot(t, grid, color="brown", label="Grid import/export", linewidth=1.5)
    ax.plot(t, res["load"], color="black", linewidth=1, label="Load")
    # Place legend above the plot, outside the axes
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax.set_xlabel("Time"); ax.set_ylabel("Power (MW)")
    ax.set_title(title); fig.tight_layout()
    # Set y-axis lower limit to 0 in fixed schedule mode
    if not show_grid:
        ax.set_ylim(bottom=0)
    return fig

def plot_soc(res: pd.DataFrame, cfg: RunConfig, title: str = "Battery State of Charge") -> Figure:
    """
    Plot state of charge (SOC) for both batteries over time.

    Args:
        res (pd.DataFrame): Dispatch results DataFrame.
        cfg (RunConfig): Run configuration object.
        title (str): Plot title.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    t = res.index
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(t, res["soc1"], label="Battery 1 SOC", color="lightblue", alpha=0.5)
    ax.plot(t, res["soc2"], label="Battery 2 SOC", color="lightgreen", alpha=0.5)
    ax.set_xlabel("Time"); ax.set_ylabel("State of Charge (MWh)")
    ax.set_title(title)
    ax.legend(); fig.tight_layout()
    return fig

def plot_clipped(res: pd.DataFrame, cfg: RunConfig, title: str = "Clipped Energy") -> Figure:
    """
    Plot clipped (unused/curtailed) energy over time.

    Args:
        res (pd.DataFrame): Dispatch results DataFrame.
        cfg (RunConfig): Run configuration object.
        title (str): Plot title.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    t = res.index
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(t, res["clipped"], label="Clipped Energy", color="red")
    ax.set_xlabel("Time"); ax.set_ylabel("Clipped (MW)")
    ax.set_title(title)
    ax.legend(); fig.tight_layout()
    return fig

def plot_grid(res: pd.DataFrame, cfg: RunConfig, title: str = "Grid Charging/Discharging") -> Figure:
    """
    Plot grid import (charging) and export (discharging) over time.

    Args:
        res (pd.DataFrame): Dispatch results DataFrame.
        cfg (RunConfig): Run configuration object.
        title (str): Plot title.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    t = res.index
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(t, res["grid_imp"], label="Grid Import (Charging)", color="brown")
    ax.plot(t, res["grid_exp"], label="Grid Export (Discharging)", color="orange")
    ax.set_xlabel("Time"); ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.legend(); fig.tight_layout()
    return fig

def plot_revenue(res: pd.DataFrame, cfg: RunConfig, title: str = "Revenue Over Time") -> Figure:
    """
    Plot cumulative revenue over time, with market price overlay.

    Args:
        res (pd.DataFrame): Dispatch results DataFrame.
        cfg (RunConfig): Run configuration object.
        title (str): Plot title.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    t = res.index
    if "price" in res:
        revenue = ((res["grid_exp"] - res["grid_imp"]) * res["price"]) / 1000  # $/h
        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(t, revenue.cumsum(), label="Cumulative Revenue ($)", color="purple")
        ax.set_xlabel("Time"); ax.set_ylabel("Cumulative Revenue ($)")
        ax.set_title(title)
        # Overlay market price on secondary y-axis
        ax2 = ax.twinx()
        if "price_$/kWh" in res:
            ax2.plot(t, res["price_$/kWh"], color="orange", label="Market Price ($/kWh)", linewidth=1.2, alpha=0.7)
            ax2.set_ylabel("Market Price ($/kWh)", color="orange")
        else:
            ax2.plot(t, res["price"], color="orange", label="Market Price ($/MWh)", linewidth=1.2, alpha=0.7)
            ax2.set_ylabel("Market Price ($/MWh)", color="orange")
        ax2.tick_params(axis='y', labelcolor="orange")
        # Add legends for both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper left")
        fig.tight_layout()
        return fig
    else:
        fig, ax = plt.subplots(figsize=(14,5))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No price data available", ha="center", va="center")
        return fig

def plot_dispatch_plotly(res: pd.DataFrame, cfg: RunConfig, title: str = "Dispatch (All Metrics)") -> go.Figure:
    """
    Create an interactive Plotly figure for all available dispatch metrics.

    Args:
        res (pd.DataFrame): Dispatch results DataFrame.
        cfg (RunConfig): Run configuration object.
        title (str): Plot title.

    Returns:
        go.Figure: Plotly figure object.
    """
    t = res.index
    fig = go.Figure()
    
    # Default visible traces - only show net_to_grid and load by default
    # Add net_to_grid (most important trace, shown by default)
    if "grid_exp" in res and "grid_imp" in res:
        fig.add_trace(go.Scatter(x=t, y=res["grid_exp"] - res["grid_imp"], mode='lines', name='Net to Grid', line=dict(color='brown'), visible=True))
    
    # Add load for reference (shown by default)
    if "load" in res:
        fig.add_trace(go.Scatter(x=t, y=res["load"], mode='lines', name='Load', line=dict(color='black'), visible=True))
    
    # Add market price in $/kWh (shown by default)
    if "price_$/kWh" in res:
        fig.add_trace(go.Scatter(x=t, y=res["price_$/kWh"], mode='lines', name='Market Price ($/kWh)', line=dict(color='purple', dash='dot'), visible=True))
    
    # All other traces are hidden by default - user can enable them
    if "Solar (MW)" in res:
        fig.add_trace(go.Scatter(x=t, y=res["Solar (MW)"], mode='lines', name='Solar', line=dict(color='yellow'), visible='legendonly'))
    if "Wind (MW)" in res:
        fig.add_trace(go.Scatter(x=t, y=res["Wind (MW)"], mode='lines', name='Wind', line=dict(color='orange'), visible='legendonly'))
    if "NatGas (MW)" in res:
        fig.add_trace(go.Scatter(x=t, y=res["NatGas (MW)"], mode='lines', name='NatGas', line=dict(color='grey'), visible='legendonly'))
    if "discharge1" in res and "charge1" in res:
        fig.add_trace(go.Scatter(x=t, y=res["discharge1"] - res["charge1"], mode='lines', name='Battery 1', line=dict(color='lightblue'), opacity=0.5, visible='legendonly'))
    if "discharge2" in res and "charge2" in res:
        fig.add_trace(go.Scatter(x=t, y=res["discharge2"] - res["charge2"], mode='lines', name='Battery 2', line=dict(color='lightgreen'), opacity=0.5, visible='legendonly'))
    if "revenue_t" in res:
        fig.add_trace(go.Scatter(x=t, y=res["revenue_t"].cumsum(), mode='lines', name='Cumulative Revenue ($)', line=dict(color='magenta', dash='dash'), visible='legendonly'))
    if "clipped" in res:
        fig.add_trace(go.Scatter(x=t, y=res["clipped"], mode='lines', name='Clipped', line=dict(color='red', dash='dot'), visible='legendonly'))
    # Add POI limits if available
    POI = getattr(cfg, 'poi_limit_mw', None)
    if POI is not None and POI > 0:
        fig.add_trace(go.Scatter(x=t, y=[POI]*len(t), mode='lines', name='POI limit', line=dict(color='red', dash='dash'), visible='legendonly'))
    
    # Add any other time series columns (hidden by default, avoid duplications)
    for col in res.columns:
        if col not in ["Solar (MW)", "Wind (MW)", "NatGas (MW)", "discharge1", "charge1", "discharge2", "charge2", "grid_exp", "grid_imp", "load", "Load (MW)", "price", "price_$/kWh", "Market Price ($/MWh)", "revenue_t", "clipped", "net_to_grid"]:
            # Use dtype.type to avoid ExtensionDtype issues
            if np.issubdtype(res[col].dtype.type, np.number):
                fig.add_trace(go.Scatter(x=t, y=res[col], mode='lines', name=col, visible='legendonly'))
    
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value (various units)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
