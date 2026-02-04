"""
07_roc_threshold.py
===================

ROC analysis for optimal planning threshold determination.

This script uses logistic regression and ROC curve analysis to:
1. Evaluate distance as a predictor of noise compliance (LAeq ≤ 55 dB)
2. Find the optimal setback distance using Youden's J statistic
3. Compare predictive performance with/without park type

The 55 dB threshold is based on typical park noise standards (e.g., China's
GB 3096 Class 1 standard for residential/park areas).

Output:
    - tables/roc_compliance_by_distance.csv
    - tables/roc_analysis_summary.csv
    - tables/roc_sensitivity_thresholds.csv
    - figures/roc_threshold_analysis.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from figure_style import add_subplot_label, setup_style
from utils import (
    attach_baseline,
    ensure_dirs,
    load_measurements,
    save_figure,
    save_table,
)


def find_optimal_threshold(fpr, tpr, thresholds):
    """Find optimal threshold using Youden's J statistic (J = TPR - FPR)."""
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx], j_scores[optimal_idx]


def cluster_bootstrap_roc_distance(
    data: pd.DataFrame,
    outcome_col: str,
    cluster_col: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster bootstrap AUC and optimal distance for distance-only ROC."""
    rng = np.random.default_rng(seed)
    clusters = data[cluster_col].dropna().unique()
    draws: list[dict] = []

    for b in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        df_b = pd.concat(
            [data.loc[data[cluster_col].eq(c)] for c in sampled],
            ignore_index=True,
        )

        y_b = df_b[outcome_col].to_numpy()
        if y_b.min() == y_b.max():
            continue

        X_b = df_b[["distance_m"]].to_numpy()
        lr_b = LogisticRegression(solver="lbfgs", max_iter=1000)
        try:
            lr_b.fit(X_b, y_b)
        except Exception:
            continue

        y_prob_b = lr_b.predict_proba(X_b)[:, 1]
        fpr_b, tpr_b, thresholds_b = roc_curve(y_b, y_prob_b)
        auc_b = auc(fpr_b, tpr_b)

        opt_prob_b, j_stat_b = find_optimal_threshold(fpr_b, tpr_b, thresholds_b)
        b0_b, b1_b = float(lr_b.intercept_[0]), float(lr_b.coef_[0, 0])

        if 0 < opt_prob_b < 1 and b1_b != 0:
            opt_dist_b = (np.log(opt_prob_b / (1 - opt_prob_b)) - b0_b) / b1_b
        else:
            opt_dist_b = np.nan

        draws.append(
            {
                "draw": b,
                "AUC": float(auc_b),
                "optimal_distance_m": float(opt_dist_b),
                "youden_J": float(j_stat_b),
            }
        )

    return pd.DataFrame(draws)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="white")
    setup_style()

    # Load and prepare data
    meas = load_measurements()
    inside = attach_baseline(meas).copy()

    # Binary outcomes: compliance with noise standards
    inside["comply_55"] = (inside["LAeq"] <= 55).astype(int)
    inside["comply_60"] = (inside["LAeq"] <= 60).astype(int)

    results = []

    # =========================================================================
    # Descriptive: Compliance rates by distance
    # =========================================================================
    comply_by_dist = inside.groupby("distance_m").agg(
        n=("comply_55", "count"),
        comply_55_pct=("comply_55", lambda x: x.mean() * 100),
        comply_60_pct=("comply_60", lambda x: x.mean() * 100),
        LAeq_mean=("LAeq", "mean"),
        LAeq_std=("LAeq", "std"),
    ).reset_index()

    save_table(comply_by_dist, "roc_compliance_by_distance.csv")

    print("Compliance Rates by Distance:")
    print(comply_by_dist.to_string(index=False))

    # =========================================================================
    # ROC Analysis 1: Distance → LAeq ≤ 55 dB
    # =========================================================================
    print("\n" + "=" * 60)
    print("ROC Analysis: Distance predicting LAeq ≤ 55 dB")
    print("=" * 60)

    X = inside[["distance_m"]].values
    y = inside["comply_55"].values

    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(X, y)
    y_prob = lr.predict_proba(X)[:, 1]

    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    optimal_prob, j_stat = find_optimal_threshold(fpr, tpr, thresholds)

    # Convert probability threshold back to distance
    b0, b1 = lr.intercept_[0], lr.coef_[0, 0]
    if 0 < optimal_prob < 1:
        optimal_distance = (np.log(optimal_prob / (1 - optimal_prob)) - b0) / b1
    else:
        optimal_distance = np.nan

    results.append({
        "outcome": "LAeq ≤ 55",
        "predictor": "distance only",
        "AUC": roc_auc,
        "AUC_boot_ci_low": np.nan,
        "AUC_boot_ci_high": np.nan,
        "optimal_distance_m": optimal_distance,
        "optimal_distance_boot_ci_low": np.nan,
        "optimal_distance_boot_ci_high": np.nan,
        "youden_J": j_stat,
        "logit_intercept": b0,
        "logit_slope": b1,
        "odds_ratio_per_m": np.nan,
        "or_ci_low_per_m": np.nan,
        "or_ci_high_per_m": np.nan,
        "p_distance_cluster": np.nan,
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logit55 = smf.logit("comply_55 ~ distance_m", data=inside).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": inside["sample_round"]},
        )

    coef_m = float(logit55.params["distance_m"])
    se_m = float(logit55.bse["distance_m"])
    or_m = float(np.exp(coef_m))
    or_m_ci_low = float(np.exp(coef_m - 1.96 * se_m))
    or_m_ci_high = float(np.exp(coef_m + 1.96 * se_m))
    p_m = float(logit55.pvalues["distance_m"])

    save_table(
        pd.DataFrame(
            [
                {
                    "term": "distance_m",
                    "coef": coef_m,
                    "se_cluster": se_m,
                    "odds_ratio": or_m,
                    "ci_low": or_m_ci_low,
                    "ci_high": or_m_ci_high,
                    "p": p_m,
                }
            ]
        ),
        "roc_logit_odds_ratio_55.csv",
    )

    print("\nLogistic Regression (cluster-robust):")
    print(f"  Distance OR per 1m: {or_m:.3f} (95% CI: {or_m_ci_low:.3f}, {or_m_ci_high:.3f})")
    print(f"  p (clustered by park×round): {p_m:.4f}")

    print(f"\nROC Performance:")
    print(f"  AUC = {roc_auc:.3f}")
    print(f"  Optimal distance threshold ≈ {optimal_distance:.1f} m")
    print(f"  Youden's J = {j_stat:.3f}")

    # Cluster bootstrap (by park) for uncertainty on AUC and optimal threshold
    boot55 = cluster_bootstrap_roc_distance(
        inside, outcome_col="comply_55", cluster_col="sample_id", n_boot=1000, seed=42
    )
    save_table(boot55, "roc_bootstrap_draws_55_by_park.csv")

    auc_ci_low = float(boot55["AUC"].quantile(0.025))
    auc_ci_high = float(boot55["AUC"].quantile(0.975))
    dist_ci_low = float(boot55["optimal_distance_m"].quantile(0.025))
    dist_ci_high = float(boot55["optimal_distance_m"].quantile(0.975))

    boot_summary_55 = pd.DataFrame(
        [
            {
                "cluster_unit": "park (sample_id)",
                "n_boot": 1000,
                "n_success": int(len(boot55)),
                "AUC_median": float(boot55["AUC"].median()),
                "AUC_ci_low": auc_ci_low,
                "AUC_ci_high": auc_ci_high,
                "optimal_distance_median": float(boot55["optimal_distance_m"].median()),
                "optimal_distance_ci_low": dist_ci_low,
                "optimal_distance_ci_high": dist_ci_high,
            }
        ]
    )
    save_table(boot_summary_55, "roc_bootstrap_summary_55_by_park.csv")

    # Attach bootstrap CI + OR results to summary table row 0
    results[0].update(
        {
            "AUC_boot_ci_low": auc_ci_low,
            "AUC_boot_ci_high": auc_ci_high,
            "optimal_distance_boot_ci_low": dist_ci_low,
            "optimal_distance_boot_ci_high": dist_ci_high,
            "odds_ratio_per_m": or_m,
            "or_ci_low_per_m": or_m_ci_low,
            "or_ci_high_per_m": or_m_ci_high,
            "p_distance_cluster": p_m,
        }
    )

    # =========================================================================
    # ROC Analysis 2: Distance → LAeq ≤ 60 dB
    # =========================================================================
    y60 = inside["comply_60"].values
    lr60 = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr60.fit(X, y60)
    y_prob60 = lr60.predict_proba(X)[:, 1]

    fpr60, tpr60, thresholds60 = roc_curve(y60, y_prob60)
    roc_auc60 = auc(fpr60, tpr60)
    optimal_prob60, j_stat60 = find_optimal_threshold(fpr60, tpr60, thresholds60)

    b0_60, b1_60 = lr60.intercept_[0], lr60.coef_[0, 0]
    if 0 < optimal_prob60 < 1:
        optimal_distance60 = (np.log(optimal_prob60 / (1 - optimal_prob60)) - b0_60) / b1_60
    else:
        optimal_distance60 = np.nan

    results.append({
        "outcome": "LAeq ≤ 60",
        "predictor": "distance only",
        "AUC": roc_auc60,
        "AUC_boot_ci_low": np.nan,
        "AUC_boot_ci_high": np.nan,
        "optimal_distance_m": optimal_distance60,
        "optimal_distance_boot_ci_low": np.nan,
        "optimal_distance_boot_ci_high": np.nan,
        "youden_J": j_stat60,
        "logit_intercept": b0_60,
        "logit_slope": b1_60,
        "odds_ratio_per_m": np.nan,
        "or_ci_low_per_m": np.nan,
        "or_ci_high_per_m": np.nan,
        "p_distance_cluster": np.nan,
    })

    print(f"\nLAeq ≤ 60 dB threshold:")
    print(f"  AUC = {roc_auc60:.3f}")
    print(f"  Optimal distance ≈ {optimal_distance60:.1f} m")

    # =========================================================================
    # ROC Analysis 3: Distance + Park Type → LAeq ≤ 55 dB
    # =========================================================================
    print("\n" + "-" * 40)
    print("Adding Park Type to the model...")

    inside["ParkType_Medium"] = (inside["park_type"] == "Medium").astype(int)
    inside["ParkType_Small"] = (inside["park_type"] == "Small").astype(int)

    X_full = inside[["distance_m", "ParkType_Medium", "ParkType_Small"]].values
    lr_full = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr_full.fit(X_full, y)
    y_prob_full = lr_full.predict_proba(X_full)[:, 1]

    fpr_full, tpr_full, _ = roc_curve(y, y_prob_full)
    roc_auc_full = auc(fpr_full, tpr_full)

    results.append({
        "outcome": "LAeq ≤ 55",
        "predictor": "distance + park_type",
        "AUC": roc_auc_full,
        "AUC_boot_ci_low": np.nan,
        "AUC_boot_ci_high": np.nan,
        "optimal_distance_m": np.nan,
        "optimal_distance_boot_ci_low": np.nan,
        "optimal_distance_boot_ci_high": np.nan,
        "youden_J": np.nan,
        "logit_intercept": lr_full.intercept_[0],
        "logit_slope": lr_full.coef_[0, 0],
        "odds_ratio_per_m": np.nan,
        "or_ci_low_per_m": np.nan,
        "or_ci_high_per_m": np.nan,
        "p_distance_cluster": np.nan,
    })

    print(f"  AUC (distance only): {roc_auc:.3f}")
    print(f"  AUC (distance + park type): {roc_auc_full:.3f}")
    print(f"  AUC improvement: {(roc_auc_full - roc_auc):.3f}")

    # Save results
    results_df = pd.DataFrame(results)
    save_table(results_df, "roc_analysis_summary.csv")

    # =========================================================================
    # Sensitivity Analysis: Different thresholds
    # =========================================================================
    print("\n" + "-" * 40)
    print("Sensitivity Analysis: Different noise thresholds")

    thresholds_test = [50, 55, 60, 65]
    sens_results = []

    for thresh in thresholds_test:
        y_thresh = (inside["LAeq"] <= thresh).astype(int)
        comply_rate = y_thresh.mean() * 100

        if y_thresh.sum() > 10 and (1 - y_thresh).sum() > 10:
            lr_t = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr_t.fit(X, y_thresh)
            y_prob_t = lr_t.predict_proba(X)[:, 1]
            fpr_t, tpr_t, _ = roc_curve(y_thresh, y_prob_t)
            auc_t = auc(fpr_t, tpr_t)
        else:
            auc_t = np.nan

        sens_results.append({
            "threshold_dB": thresh,
            "comply_rate_pct": comply_rate,
            "AUC": auc_t,
        })

    sens_df = pd.DataFrame(sens_results)
    save_table(sens_df, "roc_sensitivity_thresholds.csv")

    print(sens_df.to_string(index=False))

    # =========================================================================
    # Visualization
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: ROC curves comparison
    ax = axes[0]
    ax.plot(fpr, tpr, color="#4C78A8", lw=2, label=f"LAeq≤55, AUC={roc_auc:.3f}")
    ax.plot(fpr60, tpr60, color="#F58518", lw=2, label=f"LAeq≤60, AUC={roc_auc60:.3f}")
    ax.plot(fpr_full, tpr_full, color="#E45756", lw=2, ls="--",
            label=f"+ParkType, AUC={roc_auc_full:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(False)
    add_subplot_label(ax, "a")

    # Plot 2: Compliance rate by distance
    ax = axes[1]
    ax.plot(comply_by_dist["distance_m"], comply_by_dist["comply_55_pct"],
            "o-", color="#4C78A8", lw=2, markersize=8, label="LAeq≤55")
    ax.plot(comply_by_dist["distance_m"], comply_by_dist["comply_60_pct"],
            "s-", color="#F58518", lw=2, markersize=8, label="LAeq≤60")
    if not np.isnan(optimal_distance):
        ax.axvline(optimal_distance, ls="--", color="#E45756", lw=2,
                   label=f"Optimal={optimal_distance:.0f}m")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Compliance Rate (%)")
    ax.legend()
    ax.set_ylim([0, 100])
    ax.grid(False)
    add_subplot_label(ax, "b")

    # Plot 3: Predicted probability curve
    ax = axes[2]
    dist_range = np.linspace(0, 50, 100).reshape(-1, 1)
    prob_55 = lr.predict_proba(dist_range)[:, 1]
    prob_60 = lr60.predict_proba(dist_range)[:, 1]
    ax.plot(dist_range, prob_55 * 100, color="#4C78A8", lw=2, label="P(LAeq≤55)")
    ax.plot(dist_range, prob_60 * 100, color="#F58518", lw=2, label="P(LAeq≤60)")
    ax.axhline(50, ls="--", color="gray", alpha=0.5)
    if not np.isnan(optimal_distance):
        ax.axvline(optimal_distance, ls="--", color="#E45756", lw=1.5, alpha=0.7)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Predicted Compliance Probability (%)")
    ax.legend()
    ax.set_ylim([0, 100])
    ax.grid(False)
    add_subplot_label(ax, "c")

    plt.tight_layout()
    save_figure(fig, "roc_threshold_analysis.png")
    plt.close(fig)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("PLANNING THRESHOLD RECOMMENDATIONS")
    print("=" * 60)
    print(f"\nFor LAeq ≤ 55 dB compliance:")
    print(f"  Optimal setback distance: ≈ {optimal_distance:.0f} m")
    print(f"  Discriminative ability: AUC = {roc_auc:.3f} (moderate)")
    print(f"  Note: At 50m, compliance rate is only {comply_by_dist.loc[comply_by_dist['distance_m']==50, 'comply_55_pct'].values[0]:.1f}%")

    print(f"\nCaveats:")
    print(f"  - AUC < 0.8 indicates limited predictive power")
    print(f"  - Distance alone is insufficient; park design matters")
    print(f"  - Adding park type improves AUC by only {(roc_auc_full - roc_auc):.3f}")
    print(f"  - Consider combining distance with area/vegetation requirements")

    print("\nDone! Output files saved to applsci/outputs/")


if __name__ == "__main__":
    main()
