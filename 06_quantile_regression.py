"""
06_quantile_regression.py
=========================

Quantile regression to examine distance effects across the distribution.

Standard OLS estimates the conditional mean, but for planning purposes we
often care about worst-case scenarios (upper quantiles). This script fits
quantile regression models at multiple quantiles (0.1, 0.25, 0.5, 0.75, 0.9)
with cluster bootstrap confidence intervals.

Key questions:
- Is the distance effect consistent across quantiles?
- Is the buffering reliable for worst-case noise levels?

Output:
    - tables/quantreg_point_estimates.csv
    - tables/quantreg_bootstrap_summary.csv
    - figures/quantreg_slopes_delta_LAeq.png
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
import seaborn as sns
import statsmodels.formula.api as smf

from figure_style import setup_style
from utils import (
    attach_baseline,
    ensure_dirs,
    load_measurements,
    save_figure,
    save_table,
)


def cluster_bootstrap_quantreg(
    df: pd.DataFrame,
    quantiles: list[float],
    n_boot: int = 1000,
    seed: int = 20250118,
) -> pd.DataFrame:
    """
    Cluster bootstrap for quantile regression slopes.

    Parameters
    ----------
    df : pd.DataFrame
        Data with delta_LAeq, distance10, sample_round
    quantiles : list[float]
        Quantiles to estimate
    n_boot : int
        Number of bootstrap iterations
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Bootstrap draws with quantile and slope columns
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
        for q in quantiles:
            try:
                res = smf.quantreg("delta_LAeq ~ distance10", data=boot).fit(
                    q=q, max_iter=4000
                )
                slope = float(res.params["distance10"])
            except Exception:
                slope = np.nan
            rows.append({"boot": b, "quantile": q, "slope": slope})

    return pd.DataFrame(rows)


def summarize_bootstrap(boot_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize bootstrap distribution by quantile."""
    rows = []
    for q in sorted(boot_df["quantile"].unique()):
        s = boot_df.loc[boot_df["quantile"] == q, "slope"].dropna().to_numpy()
        if len(s) == 0:
            rows.append({
                "quantile": float(q),
                "slope_mean": np.nan,
                "slope_median": np.nan,
                "ci_2_5": np.nan,
                "ci_97_5": np.nan,
                "n_boot_ok": 0,
            })
            continue
        rows.append({
            "quantile": float(q),
            "slope_mean": float(np.mean(s)),
            "slope_median": float(np.median(s)),
            "ci_2_5": float(np.quantile(s, 0.025)),
            "ci_97_5": float(np.quantile(s, 0.975)),
            "n_boot_ok": len(s),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="white")
    setup_style()

    # Load and prepare data
    meas = load_measurements()
    inside = attach_baseline(meas).copy()

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    # =========================================================================
    # Point estimates (standard quantile regression)
    # =========================================================================
    print("Fitting quantile regression models...")

    point_rows = []
    for q in quantiles:
        res = smf.quantreg("delta_LAeq ~ distance10", data=inside).fit(
            q=q, max_iter=4000
        )
        point_rows.append({
            "quantile": q,
            "slope_dB_per10m": float(res.params["distance10"]),
            "intercept": float(res.params["Intercept"]),
            "pseudo_r2": float(res.prsquared),
        })

    point_df = pd.DataFrame(point_rows)
    save_table(point_df, "quantreg_point_estimates.csv")

    print("\nPoint Estimates:")
    print(point_df.to_string(index=False))

    # =========================================================================
    # Cluster bootstrap for confidence intervals
    # =========================================================================
    print(f"\nRunning cluster bootstrap ({1000} iterations)...")

    boot_df = cluster_bootstrap_quantreg(inside, quantiles, n_boot=1000, seed=20250118)
    save_table(boot_df, "quantreg_bootstrap_draws.csv")

    # Summarize
    summary_df = summarize_bootstrap(boot_df)
    save_table(summary_df, "quantreg_bootstrap_summary.csv")

    print("\nBootstrap Summary:")
    print(summary_df.to_string(index=False))

    # =========================================================================
    # Key findings
    # =========================================================================
    print("\n" + "=" * 60)
    print("QUANTILE REGRESSION SUMMARY")
    print("=" * 60)

    median_slope = summary_df.loc[summary_df["quantile"] == 0.5, "slope_median"].iloc[0]
    q90_slope = summary_df.loc[summary_df["quantile"] == 0.9, "slope_median"].iloc[0]
    q90_ci = summary_df.loc[summary_df["quantile"] == 0.9, ["ci_2_5", "ci_97_5"]].iloc[0]

    print(f"\nMedian (q=0.5) slope: {median_slope:.3f} dB/10m")
    print(f"  - Robust estimate of typical attenuation")

    print(f"\nUpper tail (q=0.9) slope: {q90_slope:.3f} dB/10m")
    print(f"  - 95% CI: [{q90_ci['ci_2_5']:.3f}, {q90_ci['ci_97_5']:.3f}]")

    if q90_ci["ci_2_5"] < 0 < q90_ci["ci_97_5"]:
        print("  - CI spans zero: UNRELIABLE for worst-case planning")
    elif q90_slope > 0:
        print("  - POSITIVE slope at upper tail: distance may NOT help worst cases")

    # =========================================================================
    # Visualization
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot point estimates with bootstrap CIs
    ax.plot(
        summary_df["quantile"], summary_df["slope_median"],
        marker="o", lw=2, color="#4C78A8", label="Bootstrap median"
    )
    ax.fill_between(
        summary_df["quantile"],
        summary_df["ci_2_5"], summary_df["ci_97_5"],
        alpha=0.2, color="#4C78A8", label="95% CI"
    )
    ax.axhline(0, lw=1, c="black", alpha=0.5, ls="--")

    # Mark the median (q=0.5) slope for reference
    ax.scatter([0.5], [median_slope], s=100, c="#E45756", zorder=5, marker="s",
               label=f"Median slope = {median_slope:.2f}")

    ax.set_xlabel("Quantile")
    ax.set_ylabel("Slope (dB per 10m)")
    ax.legend()
    ax.set_xlim(0.05, 0.95)
    ax.grid(False)

    # Add quantile labels
    for _, row in summary_df.iterrows():
        ax.annotate(
            f"{row['slope_median']:.2f}",
            xy=(row["quantile"], row["slope_median"]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=9
        )

    plt.tight_layout()
    save_figure(fig, "quantreg_slopes_delta_LAeq.png")
    plt.close(fig)

    # =========================================================================
    # Interpretation guide
    # =========================================================================
    print("\nINTERPRETATION:")
    print("-" * 40)
    print("q=0.10: Effect on quietest 10% of observations")
    print("q=0.50: Effect on median (typical) observations")
    print("q=0.90: Effect on noisiest 10% (worst-case)")
    print("\nFor planning, consider:")
    print("- If q=0.9 slope ≈ 0: distance doesn't help worst cases")
    print("- If q=0.9 CI includes 0: buffering is unreliable")
    print("- Robust planning should use upper quantile estimates")

    print("\nDone! Output files saved to applsci/outputs/")


if __name__ == "__main__":
    main()
