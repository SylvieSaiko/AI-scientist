"""
AI Scientist — data loader scaffold.

Replace the body of load_data() with your own loading logic.
Supports CSV, SPSS (.sav via R/haven), or any other format.
The pipeline expects load_data() to return a pandas DataFrame
and preprocess() to return a cleaned DataFrame.
"""

import os
import pandas as pd
import numpy as np
from config import DATA_PATH, WORK_DIR

CACHE_PATH = os.path.join(WORK_DIR, "data", "dataset.csv")


def load_data(force_reload: bool = False) -> pd.DataFrame:
    """Load dataset from DATA_PATH (or from cache if available)."""
    if os.path.exists(CACHE_PATH) and not force_reload:
        print(f"[data_loader] Loading from cache: {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH, low_memory=False)

    print(f"[data_loader] Loading from source: {DATA_PATH}")
    # --- Replace the line below with your actual loading code ---
    # e.g. for CSV:   df = pd.read_csv(DATA_PATH, low_memory=False)
    # e.g. for Excel: df = pd.read_excel(DATA_PATH)
    # e.g. for .sav (requires R + haven):
    #   import subprocess
    #   subprocess.run(["Rscript", "data/_load_sav.R"], check=True)
    #   df = pd.read_csv(CACHE_PATH, low_memory=False)
    raise NotImplementedError("Replace with your dataset loading logic in data_loader.py")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH, index=False, encoding="utf-8")
    print(f"[data_loader] Cached: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning — adapt to your dataset.
    - Drop PII columns
    - Coerce numeric columns
    - Engineer any derived features (e.g. log-transform skewed labs)
    """
    # Drop PII — replace with your own PII column names
    pii_cols = []
    df = df.drop(columns=[c for c in pii_cols if c in df.columns], errors="ignore")

    # Example: coerce numeric columns
    # for col in ["age", "lvef", "egfr"]:
    #     if col in df.columns:
    #         df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_survival_df(df: pd.DataFrame, event_col: str, time_col: str,
                    max_days: float = None) -> pd.DataFrame:
    """Return a dataframe ready for survival analysis (event + time columns)."""
    sdf = df.copy()
    sdf["event"] = pd.to_numeric(sdf[event_col], errors="coerce").fillna(0).astype(int)
    sdf["time"]  = pd.to_numeric(sdf[time_col],  errors="coerce")
    sdf = sdf[sdf["time"] > 0].copy()
    sdf["time"] = sdf["time"].clip(lower=0.5)
    if max_days:
        sdf.loc[sdf["time"] > max_days, "event"] = 0
        sdf["time"] = sdf["time"].clip(upper=max_days)
    return sdf
