"""
data_io.py
------------
Handles loading and preprocessing of time series data for battery dispatch optimization.
Supports CSV and Excel files, ensures required columns are present, and fills missing optional columns.
"""

import os
import sys
import pandas as pd
from .config import RunConfig

def load_data(cfg: RunConfig) -> pd.DataFrame:
    """
    Loads time series data from a CSV or Excel file, validates required columns, and fills missing optional columns.

    Parameters:
        cfg (RunConfig): Configuration object specifying file path, date window, and expected columns.

    Returns:
        pd.DataFrame: DataFrame indexed by Datetime, filtered by date window, with required and optional columns.

    Raises:
        SystemExit: If the file is not found or required columns are missing.
    """
    # Check if the file exists
    if not os.path.exists(cfg.path):
        sys.exit(f"[data_io] File not found: {cfg.path}. Please check the path and try again.")

    # Load data based on file extension
    if str(cfg.path).lower().endswith('.csv'):
        df = pd.read_csv(cfg.path, parse_dates=["Datetime"])
        if "Datetime" in df.columns:
            df = df.set_index("Datetime")
    else:
        # For Excel, parse combined Date and Time columns if present
        df = pd.read_excel(
            cfg.path,
            sheet_name=cfg.sheet_name,
            engine="openpyxl",
            parse_dates={"Datetime": ["Date", "Time"]},
        ).set_index("Datetime")

    # Filter by date window
    df = df.loc[cfg.start_date : cfg.end_date]

    # Ensure optional columns exist; fill with zeros if missing
    for col in ["Wind (MW)", "Solar (MW)"]:
        if col not in df.columns:
            df[col] = 0.0  # Fill missing optional columns with zeros

    # Check for required columns
    needed = ["Wind (MW)", "Solar (MW)", cfg.market_price_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"[data_io] Missing required columns: {missing}. Please ensure your data file includes these columns.")

    return df
