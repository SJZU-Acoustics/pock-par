"""
01_core_attenuation_model.py
============================

Core analysis: Noise attenuation as a function of distance from road.

This script fits OLS regression models with cluster-robust standard errors
to estimate the distance-attenuation relationship for traffic noise in
pocket parks.

Model specification:
    delta_LAeq ~ distance_m / 10

where delta_LAeq = LAeq(inside) - LAeq(P1 baseline)

Clustering: by sample_round (park × measurement round) to account for
within-cluster correlation.

Output:
    - tables/core_model_coefficients.csv
    - tables/per_park_slopes.csv
    - figures/delta_LAeq_by_distance_scatter.png
    - figures/delta_LAeq_by_distance_box.png
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from utils import (
    attach_baseline,
    ensure_dirs,
    load_measurements,
    save_figure,
    save_table,
)


def fit_ols_cluster(formula: str, data: pd.DataFrame, cluster: str):
    """
    Fit OLS model with cluster-robust standard errors.

    Parameters
    ----------
    formula : str
        Patsy formula
    data : pd.DataFrame
        Data for fitting
    cluster : str
        Column name for clustering

    Returns
    -------
    statsmodels RegressionResults
    """
    model = smf.ols(formula, data=data)
    return model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})


def extract_coefficients(result, model_name: str) -> pd.DataFrame:
    """Extract coefficient table from regression results."""
    return pd.DataFrame({
        "model": model_name,
        "term": result.params.index,
        "coef": result.params.values,
        "se": result.bse.values,
        "t": result.tvalues.values,
        "p": result.pvalues.values,
        "ci_low": result.conf_int()[0].values,
        "ci_high": result.conf_int()[1].values,
    })


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    # Load and prepare data
    meas = load_measurements()
    inside = attach_baseline(meas)

    print(f"Total observations (interior points): {len(inside)}")
    print(f"Unique parks: {inside['sample_id'].nunique()}")
    print(f"Unique sample_rounds (clusters): {inside['sample_round'].nunique()}")

    # =========================================================================
    # Model 1: Simple distance model (no park fixed effects)
    # =========================================================================
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = fit_ols_cluster(
            "delta_LAeq ~ distance10",
            inside.dropna(subset=["delta_LAeq", "distance10"]),
            cluster="sample_round"
        )

    # =========================================================================
    # Model 2: With park fixed effects
    # =========================================================================
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2 = fit_ols_cluster(
            "delta_LAeq ~ distance10 + C(sample_id)",
            inside.dropna(subset=["delta_LAeq", "distance10"]),
            cluster="sample_round"
        )

    # Combine coefficient tables
    coef_m1 = extract_coefficients(m1, "m1_simple")
    coef_m2 = extract_coefficients(m2, "m2_park_fe")
    coef_table = pd.concat([coef_m1, coef_m2], ignore_index=True)
    save_table(coef_table, "core_model_coefficients.csv")

    # =========================================================================
    # Per-park simple slopes (exploratory)
    # =========================================================================
    slopes = []
    for sid, g in inside.groupby("sample_id"):
        g = g.dropna(subset=["distance10", "delta_LAeq"])
        if g["distance10"].nunique() < 2:
            continue
        r = smf.ols("delta_LAeq ~ distance10", data=g).fit()
        slopes.append({
            "sample_id": int(sid),
            "sample_name": g["sample_name"].iloc[0],
            "park_type": g["park_type"].iloc[0],
            "n_obs": len(g),
            "slope_dB_per10m": r.params.get("distance10", np.nan),
            "intercept": r.params.get("Intercept", np.nan),
            "r_squared": r.rsquared,
        })
    slopes_df = pd.DataFrame(slopes).sort_values("sample_id")
    save_table(slopes_df, "per_park_slopes.csv")

    # =========================================================================
    # Key results summary
    # =========================================================================
    slope_m1 = float(m1.params["distance10"])
    se_m1 = float(m1.bse["distance10"])
    ci_low = slope_m1 - 1.96 * se_m1
    ci_high = slope_m1 + 1.96 * se_m1

    print("\n" + "=" * 60)
    print("CORE RESULT: Distance-Attenuation Effect")
    print("=" * 60)
    print(f"Model: delta_LAeq ~ distance/10 (clustered by sample_round)")
    print(f"Slope: {slope_m1:.3f} dB per 10m")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"SE (cluster-robust): {se_m1:.3f}")
    print(f"p-value: {m1.pvalues['distance10']:.4f}")
    print(f"R²: {m1.rsquared:.3f}")
    print("=" * 60)

    # =========================================================================
    # Visualization: Scatter plot with regression line
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        inside["distance_m"], inside["delta_LAeq"],
        s=25, alpha=0.4, c="#4C78A8", label="Observations"
    )

    # Add regression line
    x_line = np.linspace(inside["distance_m"].min(), inside["distance_m"].max(), 100)
    y_line = m1.params["Intercept"] + m1.params["distance10"] * (x_line / 10)
    ax.plot(x_line, y_line, c="#E45756", lw=2, label=f"Fitted (slope={slope_m1:.2f} dB/10m)")

    ax.axhline(0, lw=1, c="black", alpha=0.5, ls="--")
    ax.set_xlabel("Distance from road boundary (m)")
    ax.set_ylabel("ΔLAeq (dB, inside - P1 baseline)")
    ax.set_title("Noise Attenuation by Distance in Pocket Parks")
    ax.legend()

    save_figure(fig, "delta_LAeq_by_distance_scatter.png")
    plt.close(fig)

    # =========================================================================
    # Visualization: Box plot by distance category
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    distance_order = sorted(inside["distance_m"].dropna().unique())
    sns.boxplot(
        data=inside, x="distance_m", y="delta_LAeq",
        order=distance_order, color="#4C78A8", ax=ax
    )
    ax.axhline(0, lw=1, c="black", alpha=0.5, ls="--")
    ax.set_xlabel("Distance from road boundary (m)")
    ax.set_ylabel("ΔLAeq (dB, inside - P1 baseline)")
    ax.set_title("Distribution of Noise Attenuation by Distance")

    save_figure(fig, "delta_LAeq_by_distance_box.png")
    plt.close(fig)

    print("\nDone! Output files saved to applsci/outputs/")


if __name__ == "__main__":
    main()
