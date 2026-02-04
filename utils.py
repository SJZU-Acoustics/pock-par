"""
Utility functions for noise attenuation analysis in pocket parks.

This module provides common data loading, preprocessing, and output utilities
used across all analysis scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Path Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]  # applsci folder
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "measurements.csv"

OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

# Matplotlib config directory (to avoid permission issues)
MPLCONFIGDIR = ROOT / "code" / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


# =============================================================================
# Park Metadata Mappings
# =============================================================================

PARK_TYPE_LABEL = {
    # Balanced within-sample size groups (4 parks per category) used in the analysis
    "Large": "Large (6400-9500 m²)",
    "Medium": "Medium (3900-4800 m²)",
    "Small": "Small (2000-2800 m²)",
}


# =============================================================================
# Directory Management
# =============================================================================


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for d in [OUT_DIR, FIG_DIR, TABLE_DIR, MPLCONFIGDIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_measurements() -> pd.DataFrame:
    """
    Load and preprocess the measurement data from CSV.

    Returns
    -------
    pd.DataFrame
        Preprocessed measurement data with derived columns for time periods, etc.
    """
    df = pd.read_csv(DATA_PATH)
    df = df.copy()

    # Add park type label
    df["park_type_label"] = df["park_type"].map(PARK_TYPE_LABEL)

    # Parse monitoring time to extract start hour
    df["time_start"] = df["time_period"].str.extract(r"^\s*(\d{1,2}:\d{1,2})")[0]
    start_h = pd.to_numeric(df["time_start"].str.split(":").str[0], errors="coerce")
    df["start_hour"] = start_h
    df["period"] = np.where(start_h < 12, "morning", "evening")

    return df


def attach_baseline(
    meas: pd.DataFrame,
    pm_cols: list[str] | None = None,
    noise_col: str = "LAeq",
) -> pd.DataFrame:
    """
    Attach P1 baseline measurements and compute relative metrics.

    For each interior measurement point (P2-P7), this function finds the
    corresponding P1 baseline measurement and computes:
    - Log-ratio for PM indicators: log(PM_inside / PM_P1)
    - Delta for noise: LAeq_inside - LAeq_P1

    Baseline matching priority:
    1. Same sample, round, and monitoring time
    2. Same sample and round (averaged)

    Parameters
    ----------
    meas : pd.DataFrame
        Raw measurement data
    pm_cols : list[str], optional
        PM indicator columns, default ["PM1", "PM2.5", "PM10", "PM_resp", "TSP"]
    noise_col : str
        Noise indicator column, default "LAeq"

    Returns
    -------
    pd.DataFrame
        Interior measurements with baseline-relative metrics attached
    """
    if pm_cols is None:
        pm_cols = ["PM1", "PM2.5", "PM10", "PM_resp", "TSP"]

    base_cols = pm_cols + [noise_col]

    # Extract P1 baseline measurements
    p1 = meas[meas["point_id"].eq("P1")][
        ["sample_id", "round", "time_period"] + base_cols
    ].copy()

    # Time-matched baseline
    p1_time = (
        p1.groupby(["sample_id", "round", "time_period"], as_index=False)[base_cols]
        .mean()
        .rename(columns={c: f"{c}_P1_time" for c in base_cols})
    )

    # Round-averaged baseline (fallback)
    p1_round = (
        p1.groupby(["sample_id", "round"], as_index=False)[base_cols]
        .mean()
        .rename(columns={c: f"{c}_P1_round" for c in base_cols})
    )

    # Get interior points only
    inside = meas[meas["point_id"].ne("P1")].copy()
    inside = inside.merge(p1_time, on=["sample_id", "round", "time_period"], how="left")
    inside = inside.merge(p1_round, on=["sample_id", "round"], how="left")

    # Use time-matched if available, otherwise round-averaged
    for c in base_cols:
        inside[f"{c}_P1"] = inside[f"{c}_P1_time"].where(
            inside[f"{c}_P1_time"].notna(), inside[f"{c}_P1_round"]
        )

    inside["baseline_source"] = np.where(
        inside[f"{noise_col}_P1_time"].notna(), "time", "round"
    )

    # Compute relative metrics
    for c in pm_cols:
        inside[f"logratio_{c}"] = np.log(inside[c] / inside[f"{c}_P1"])
        inside[f"pct_change_{c}"] = (inside[c] / inside[f"{c}_P1"] - 1) * 100

    inside["delta_LAeq"] = inside[noise_col] - inside[f"{noise_col}_P1"]
    inside["sample_round"] = (
        inside["sample_id"].astype(str) + "_" + inside["round"].astype(str)
    )

    # Add distance10 for convenience (distance in units of 10m)
    inside["distance10"] = inside["distance_m"] / 10.0

    return inside


def park_metadata(meas: pd.DataFrame) -> pd.DataFrame:
    """
    Extract park-level metadata from measurement data.

    Parameters
    ----------
    meas : pd.DataFrame
        Measurement data

    Returns
    -------
    pd.DataFrame
        One row per park with attributes and summary statistics
    """
    meta = (
        meas.groupby("sample_id")
        .agg(
            sample_name=("sample_name", "first"),
            park_area=("park_area_m2", "first"),
            park_type=("park_type", "first"),
            park_type_label=("park_type_label", "first"),
            street_sides=("street_sides", "first"),
            openness=("openness", "first"),
            n_rounds=("round", "nunique"),
            n_points=("point_id", "nunique"),
            max_distance=("distance_m", "max"),
        )
        .reset_index()
        .sort_values("sample_id")
    )
    return meta


# =============================================================================
# Output Functions
# =============================================================================


def save_table(df: pd.DataFrame, filename: str) -> Path:
    """Save DataFrame to CSV in the tables output directory."""
    ensure_dirs()
    path = TABLE_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def save_figure(fig, filename: str, dpi: int = 300) -> Path:
    """Save matplotlib figure to the figures output directory.

    By default, most scripts call this with a `.png` filename. For manuscript
    convenience, we also save a matching `.pdf` when the input ends with `.png`.
    """
    ensure_dirs()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")

    # Also save a vector-friendly PDF for manuscript submission.
    if str(path).lower().endswith(".png"):
        pdf_path = Path(str(path)[:-4] + ".pdf")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {pdf_path}")

    return path
