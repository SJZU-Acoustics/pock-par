"""
02_linearity_test.py
====================

Model specification tests: Linear vs. nonlinear distance-response.

This script compares linear, logarithmic, and nonlinear (spline) models to
verify that the linear specification is appropriate for the observed distance
range. It also performs piecewise regression with bootstrap model selection
to test for breakpoints.

Theoretical background:
- Point source in free field: inverse square law, -6 dB per doubling (20*log10(r))
- Line source (road): cylindrical spreading, -3 dB per doubling (10*log10(r))
- Since dB is already logarithmic, L_dB ~ log(distance) is the theoretically
  correct form. However, for narrow distance ranges, linear approximation
  may be equally valid.

Tests performed:
1. Linear vs. Log vs. Natural Spline: AIC comparison
2. Piecewise regression: Bootstrap selection of linear vs. piecewise models

Output:
    - tables/linearity_model_comparison.csv
    - tables/spline_predictions.csv
    - tables/piecewise_bootstrap_selection.csv
    - figures/spline_vs_linear_delta_LAeq.png
    - figures/piecewise_bootstrap_breakpoints.png
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
import statsmodels.formula.api as smf

from utils import (
    attach_baseline,
    ensure_dirs,
    load_measurements,
    save_figure,
    save_table,
)


def predict_with_ci(result, new_data: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions with confidence intervals."""
    design_info = result.model.data.design_info
    x_new = patsy.build_design_matrices(
        [design_info], new_data, return_type="dataframe"
    )[0]
    pred = x_new.to_numpy() @ result.params.to_numpy()
    cov = result.cov_params().to_numpy()
    var = np.einsum("ij,jk,ik->i", x_new.to_numpy(), cov, x_new.to_numpy())
    se = np.sqrt(np.maximum(var, 0))
    out = new_data.copy()
    out["pred"] = pred
    out["se"] = se
    out["ci_low"] = pred - 1.96 * se
    out["ci_high"] = pred + 1.96 * se
    return out


def fit_linear(df: pd.DataFrame) -> dict:
    """Fit linear model with cluster-robust SE."""
    res = smf.ols("delta_LAeq ~ distance10", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["sample_round"]}
    )
    return {
        "model": "linear",
        "breakpoint_m": np.nan,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "r2": float(res.rsquared),
        "slope_per10m": float(res.params["distance10"]),
        "res": res,
    }


def fit_piecewise(df: pd.DataFrame, breakpoint: float) -> dict:
    """Fit piecewise linear model with hinge at specified breakpoint."""
    dd = df.copy()
    dd["hinge10"] = np.maximum(0.0, dd["distance_m"] - float(breakpoint)) / 10.0
    res = smf.ols("delta_LAeq ~ distance10 + hinge10", data=dd).fit(
        cov_type="cluster", cov_kwds={"groups": dd["sample_round"]}
    )
    slope_pre = float(res.params["distance10"])
    slope_post = slope_pre + float(res.params["hinge10"])
    return {
        "model": f"piecewise@{int(breakpoint)}m",
        "breakpoint_m": float(breakpoint),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "r2": float(res.rsquared),
        "slope_pre_per10m": slope_pre,
        "slope_post_per10m": slope_post,
        "p_hinge": float(res.pvalues["hinge10"]),
        "res": res,
    }


def cluster_bootstrap_model_selection(
    df: pd.DataFrame,
    candidates: list[float],
    n_boot: int = 1000,
    seed: int = 20250118,
) -> pd.DataFrame:
    """
    Bootstrap model selection between linear and piecewise models.

    Parameters
    ----------
    df : pd.DataFrame
        Data with delta_LAeq, distance10, distance_m, sample_round
    candidates : list[float]
        Candidate breakpoint distances in meters
    n_boot : int
        Number of bootstrap iterations
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Bootstrap results with selected model for each iteration
    """
    rng = np.random.default_rng(seed)
    clusters = df["sample_round"].dropna().unique().tolist()
    n_clusters = len(clusters)

    rows = []
    for b in range(n_boot):
        sampled = rng.choice(clusters, size=n_clusters, replace=True)
        boot = pd.concat(
            [df[df["sample_round"] == c] for c in sampled], ignore_index=True
        )

        # Fit all models
        fits = [fit_linear(boot)]
        for c in candidates:
            try:
                fits.append(fit_piecewise(boot, c))
            except Exception:
                continue

        # Select by AIC
        best = min(fits, key=lambda x: x["aic"])
        rows.append(
            {
                "boot": b,
                "best_model": best["model"],
                "breakpoint_m": best.get("breakpoint_m", np.nan),
                "aic": best["aic"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    # Load and prepare data
    meas = load_measurements()
    inside = attach_baseline(meas).copy()

    # =========================================================================
    # Part 1: Linear vs. Log vs. Spline comparison (AIC)
    # =========================================================================
    print("Fitting linear vs. log vs. spline models...")

    # Prepare log-transformed distance
    inside["log_distance"] = np.log(inside["distance_m"])

    linear_res = smf.ols("delta_LAeq ~ distance10", data=inside).fit(
        cov_type="cluster", cov_kwds={"groups": inside["sample_round"]}
    )
    log_res = smf.ols("delta_LAeq ~ log_distance", data=inside).fit(
        cov_type="cluster", cov_kwds={"groups": inside["sample_round"]}
    )
    spline_res = smf.ols(
        "delta_LAeq ~ bs(distance_m, df=4, degree=3)", data=inside
    ).fit(cov_type="cluster", cov_kwds={"groups": inside["sample_round"]})

    # Calculate implied dB per doubling for log model
    log_coef = float(log_res.params["log_distance"])
    db_per_doubling = log_coef * np.log(2)

    comparison = pd.DataFrame(
        [
            {
                "model": "linear",
                "formula": "delta_LAeq ~ distance/10",
                "n": int(linear_res.nobs),
                "df_model": float(linear_res.df_model),
                "aic": float(linear_res.aic),
                "bic": float(linear_res.bic),
                "r2": float(linear_res.rsquared),
                "coef": float(linear_res.params["distance10"]),
                "coef_interpretation": "dB per 10m",
            },
            {
                "model": "log",
                "formula": "delta_LAeq ~ ln(distance)",
                "n": int(log_res.nobs),
                "df_model": float(log_res.df_model),
                "aic": float(log_res.aic),
                "bic": float(log_res.bic),
                "r2": float(log_res.rsquared),
                "coef": log_coef,
                "coef_interpretation": f"dB per ln(m); implies {db_per_doubling:.2f} dB per doubling",
            },
            {
                "model": "spline",
                "formula": "delta_LAeq ~ bs(distance, df=4)",
                "n": int(spline_res.nobs),
                "df_model": float(spline_res.df_model),
                "aic": float(spline_res.aic),
                "bic": float(spline_res.bic),
                "r2": float(spline_res.rsquared),
                "coef": np.nan,
                "coef_interpretation": "nonlinear",
            },
        ]
    )
    save_table(comparison, "linearity_model_comparison.csv")

    print("\nModel Comparison (AIC):")
    print(comparison[["model", "aic", "bic", "r2"]].to_string(index=False))

    # Report theoretical comparison
    print(f"\nTheoretical comparison:")
    print(f"  Log model coefficient: {log_coef:.2f} dB per ln(m)")
    print(f"  Implied dB per doubling: {db_per_doubling:.2f} dB")
    print(f"  Theory (line source): -3.0 dB per doubling")
    print(f"  Theory (point source): -6.0 dB per doubling")
    print(f"\n  AIC difference (Linear - Log): {linear_res.aic - log_res.aic:.2f}")
    if abs(linear_res.aic - log_res.aic) < 2:
        print("  → Models essentially equivalent (|ΔAIC| < 2)")

    # Generate predictions for plotting
    grid = pd.DataFrame(
        {"distance_m": np.linspace(5, float(inside["distance_m"].max()), 200)}
    )
    grid["distance10"] = grid["distance_m"] / 10.0
    grid["log_distance"] = np.log(grid["distance_m"])

    pred_linear = predict_with_ci(linear_res, grid)
    pred_log = predict_with_ci(log_res, grid)
    pred_spline = predict_with_ci(spline_res, grid)
    save_table(pred_spline, "spline_predictions.csv")

    # Plot comparison (now including log model)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(
        inside["distance_m"],
        inside["delta_LAeq"],
        s=18,
        alpha=0.35,
        c="gray",
        label="Observations",
    )
    ax.plot(
        pred_linear["distance_m"],
        pred_linear["pred"],
        lw=2,
        c="#4C78A8",
        label=f"Linear (AIC={linear_res.aic:.1f})",
    )
    ax.plot(
        pred_log["distance_m"],
        pred_log["pred"],
        lw=2,
        c="#59A14F",
        ls="--",
        label=f"Log (AIC={log_res.aic:.1f})",
    )
    ax.plot(
        pred_spline["distance_m"],
        pred_spline["pred"],
        lw=2,
        c="#E45756",
        ls=":",
        label=f"Spline (AIC={spline_res.aic:.1f})",
    )
    ax.fill_between(
        pred_linear["distance_m"],
        pred_linear["ci_low"],
        pred_linear["ci_high"],
        alpha=0.12,
        color="#4C78A8",
    )
    ax.axhline(0, lw=1, c="black", alpha=0.5, ls="--")
    ax.set_xlabel("Distance from road boundary (m)")
    ax.set_ylabel("ΔLAeq (dB)")
    ax.set_title("Model Comparison: Linear vs. Log vs. Spline")
    ax.legend(loc="lower left")

    # Add annotation about model equivalence
    if abs(linear_res.aic - log_res.aic) < 2:
        ax.annotate(
            f"Linear ≈ Log (|ΔAIC| = {abs(linear_res.aic - log_res.aic):.1f} < 2)",
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            style="italic",
            color="gray",
        )

    save_figure(fig, "spline_vs_linear_delta_LAeq.png")
    plt.close(fig)

    # =========================================================================
    # Part 2: Piecewise regression with bootstrap model selection
    # =========================================================================
    print("\nRunning piecewise regression bootstrap (1000 iterations)...")

    # Define candidate breakpoints (distances with sufficient observations)
    dist_counts = inside["distance_m"].value_counts().sort_index()
    dmin = float(dist_counts.index.min())
    dmax = float(dist_counts.index.max())
    candidates = [
        float(d)
        for d, n in dist_counts.items()
        if float(d) not in {dmin, dmax} and int(n) >= 30
    ]

    print(f"Candidate breakpoints: {candidates}")

    # Run bootstrap
    boot_results = cluster_bootstrap_model_selection(
        inside, candidates, n_boot=1000, seed=20250118
    )
    save_table(boot_results, "piecewise_bootstrap_results.csv")

    # Summarize selection frequencies
    selection_freq = (
        boot_results["best_model"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
        .rename(columns={"index": "model"})
    )
    save_table(selection_freq, "piecewise_bootstrap_selection.csv")

    print("\nBootstrap Model Selection Results:")
    print(selection_freq.to_string(index=False))

    linear_pct = selection_freq.loc[
        selection_freq["best_model"] == "linear", "proportion"
    ]
    if len(linear_pct) > 0:
        linear_pct = float(linear_pct.iloc[0]) * 100
        print(f"\nLinear model selected in {linear_pct:.1f}% of bootstrap samples")

    # Plot bootstrap breakpoint distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    bn = boot_results.copy()
    bn["breakpoint_m"] = bn["breakpoint_m"].fillna(-1)  # -1 indicates linear
    sns.countplot(data=bn, x="breakpoint_m", ax=ax, color="#4C78A8")
    ax.set_title("Bootstrap Model Selection: Breakpoint Distribution")
    ax.set_xlabel("Breakpoint (m); -1 = linear model selected")
    ax.set_ylabel("Count (out of 500)")

    save_figure(fig, "piecewise_bootstrap_breakpoints.png")
    plt.close(fig)

    print("\nDone! Output files saved to applsci/outputs/")


if __name__ == "__main__":
    main()
