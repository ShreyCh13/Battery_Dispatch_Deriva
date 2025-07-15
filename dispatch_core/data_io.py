import os, sys
import pandas as pd
from .config import RunConfig

def load_data(cfg: RunConfig) -> pd.DataFrame:
    if not os.path.exists(cfg.path):
        sys.exit(f"[data_io] File not found: {cfg.path}")

    df = pd.read_excel(
        cfg.path,
        sheet_name=cfg.sheet_name,
        engine="openpyxl",
        parse_dates={"Datetime": ["Date", "Time"]},
    ).set_index("Datetime")

    df = df.loc[cfg.start_date : cfg.end_date]

    # --- FILL BEFORE CHECKING ---
    for col in ["Wind (MW)", "Solar (MW)"]:
        if col not in df.columns:
            df[col] = 0.0

    needed = ["Wind (MW)", "Solar (MW)", cfg.market_price_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"[data_io] Missing columns: {missing}")

    return df
