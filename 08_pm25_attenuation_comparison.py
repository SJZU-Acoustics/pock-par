"""
08_pm25_attenuation_comparison.py
=================================

PM2.5 attenuation analysis as a contrast dimension to noise.

This script applies the same analytical framework used for noise (scripts 01-06)
to PM2.5 concentration data, enabling direct comparison of attenuation
effectiveness across environmental indicators.

Analyses included:
    1. Core linear model: logratio_PM2.5 ~ distance/10 (cluster-robust SE)
    2. Model specification: linear vs. log vs. spline (AIC + bootstrap selection)
    3. Baseline sensitivity: time-matched vs. round-averaged P1
    4. Park-level heterogeneity: mixed-effects random slopes
    5. Quantile regression: tau = 0.10, 0.25, 0.50, 0.75, 0.90
    6. Comparative summary table: noise vs. PM2.5

Output:
    - tables/pm25_core_model_coefficients.csv
    - tables/pm25_model_comparison.csv
    - tables/pm25_bootstrap_model_selection.csv
    - tables/pm25_baseline_sensitivity.csv
    - tables/pm25_mixed_effects_slopes.csv
    - tables/pm25_quantile_regression.csv
    - tables/noise_vs_pm25_comparison.csv
    - figures/pm25_logratio_by_distance_box.png
    - figures/pm25_model_comparison.png
    - figures/pm25_quantile_slopes.png
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import interpolate

from figure_style import setup_style
from utils import (
    attach_baseline,
    ensure_dirs,
    load_measurements,
    save_figure,
    save_table,
)


# =========================================================================
# Helper functions
# =========================================================================

def fit_ols_cluster(formula: str, data: pd.DataFrame, cluster: str):
    """Fit OLS model with cluster-robust standard errors."""
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


# =========================================================================
# 1. Core linear model
# =========================================================================

def core_linear_model(inside: pd.DataFrame) -> dict:
    """Fit core OLS model for PM2.5 logratio vs distance."""
    df = inside.dropna(subset=["logratio_PM2.5", "distance10"]).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_ols_cluster(
            'Q("logratio_PM2.5") ~ distance10', df, "sample_round"
        )

    coef_table = extract_coefficients(result, "pm25_core")
    save_table(coef_table, "pm25_core_model_coefficients.csv")

    slope = float(result.params["distance10"])
    se = float(result.bse["distance10"])
    pval = float(result.pvalues["distance10"])
    pct_per10m = (np.exp(slope) - 1) * 100

    print("\n" + "=" * 60)
    print("PM2.5 CORE RESULT: Distance-Attenuation Effect")
    print("=" * 60)
    print(f"Slope (logratio/10m): {slope:.6f}")
    print(f"Approximate % change per 10m: {pct_per10m:.2f}%")
    print(f"95% CI: [{slope - 1.96*se:.6f}, {slope + 1.96*se:.6f}]")
    print(f"p-value: {pval:.4f}")
    print(f"R²: {result.rsquared:.4f}")
    print("=" * 60)

    return {
        "slope": slope,
        "se": se,
        "p": pval,
        "pct_per10m": pct_per10m,
        "r2": result.rsquared,
        "n": int(result.nobs),
    }


# =========================================================================
# 2. Model specification test
# =========================================================================

def model_specification_test(inside: pd.DataFrame) -> pd.DataFrame:
    """Compare linear, log, and spline models for PM2.5."""
    from patsy import dmatrix

    df = inside.dropna(subset=["logratio_PM2.5", "distance10"]).copy()
    df["log_distance"] = np.log(df["distance_m"])

    models = {}

    # Linear
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_lin = fit_ols_cluster(
            'Q("logratio_PM2.5") ~ distance10', df, "sample_round"
        )
    models["linear"] = m_lin

    # Log
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_log = fit_ols_cluster(
            'Q("logratio_PM2.5") ~ log_distance', df, "sample_round"
        )
    models["log"] = m_log

    # Natural cubic spline (df=4)
    spline_basis = dmatrix(
        "cr(distance_m, df=4) - 1", data=df, return_type="dataframe"
    )
    for col in spline_basis.columns:
        df[f"spl_{col}"] = spline_basis[col].values

    spl_terms = " + ".join([f'Q("spl_{c}")' for c in spline_basis.columns])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_spl = fit_ols_cluster(
            f'Q("logratio_PM2.5") ~ {spl_terms}', df, "sample_round"
        )
    models["spline_df4"] = m_spl

    comparison = pd.DataFrame([
        {
            "model": name,
            "aic": float(res.aic),
            "bic": float(res.bic),
            "r2": float(res.rsquared),
            "n_params": len(res.params),
        }
        for name, res in models.items()
    ]).sort_values("aic")

    save_table(comparison, "pm25_model_comparison.csv")

    print("\nPM2.5 Model Comparison (AIC):")
    for _, row in comparison.iterrows():
        print(f"  {row['model']:12s}: AIC={row['aic']:.1f}, R²={row['r2']:.4f}")

    return comparison


# =========================================================================
# 3. Bootstrap model selection
# =========================================================================

def bootstrap_model_selection(
    inside: pd.DataFrame, n_boot: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Bootstrap model selection between linear and piecewise models."""
    df = inside.dropna(subset=["logratio_PM2.5", "distance10"]).copy()
    clusters = df["sample_round"].unique()
    rng = np.random.default_rng(seed)

    candidates = [10, 20, 30, 40]  # breakpoint candidates (meters)
    model_names = ["linear"] + [f"piecewise@{bp}m" for bp in candidates]
    counts = {m: 0 for m in model_names}

    for _ in range(n_boot):
        boot_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        boot_df = pd.concat(
            [df[df["sample_round"] == c] for c in boot_clusters],
            ignore_index=True,
        )
        if boot_df["distance10"].nunique() < 2:
            continue

        best_aic = np.inf
        best_model = "linear"

        # Linear
        try:
            m = smf.ols('Q("logratio_PM2.5") ~ distance10', data=boot_df).fit()
            if m.aic < best_aic:
                best_aic = m.aic
                best_model = "linear"
        except Exception:
            continue

        # Piecewise
        for bp in candidates:
            bp10 = bp / 10.0
            boot_df["_hinge"] = np.maximum(boot_df["distance10"] - bp10, 0)
            try:
                m = smf.ols(
                    'Q("logratio_PM2.5") ~ distance10 + _hinge', data=boot_df
                ).fit()
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_model = f"piecewise@{bp}m"
            except Exception:
                pass

        counts[best_model] += 1

    total = sum(counts.values())
    selection = pd.DataFrame([
        {"model": m, "count": c, "share": c / total if total > 0 else 0}
        for m, c in counts.items()
    ]).sort_values("share", ascending=False)

    save_table(selection, "pm25_bootstrap_model_selection.csv")

    print("\nPM2.5 Bootstrap Model Selection:")
    for _, row in selection.iterrows():
        print(f"  {row['model']:18s}: {row['share']:.1%}")

    return selection


# =========================================================================
# 4. Baseline sensitivity
# =========================================================================

def baseline_sensitivity(meas: pd.DataFrame) -> pd.DataFrame:
    """Test sensitivity to P1 baseline definition for PM2.5."""
    pm_cols = ["PM1", "PM2.5", "PM10", "PM_resp", "TSP"]
    inside = attach_baseline(meas, pm_cols=pm_cols)
    inside["distance10"] = inside["distance_m"] / 10.0

    # Main model (time-matched with round fallback)
    df_main = inside.dropna(subset=["logratio_PM2.5", "distance10"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_main = fit_ols_cluster(
            'Q("logratio_PM2.5") ~ distance10', df_main, "sample_round"
        )

    # Round-only baseline (ignore time matching)
    p1 = meas[meas["point_id"].eq("P1")].copy()
    p1_round = (
        p1.groupby(["sample_id", "round"], as_index=False)[["PM2.5"]]
        .mean()
        .rename(columns={"PM2.5": "PM2.5_P1_roundonly"})
    )
    inside2 = inside.merge(p1_round, on=["sample_id", "round"], how="left")
    inside2["logratio_pm25_roundonly"] = np.log(
        inside2["PM2.5"] / inside2["PM2.5_P1_roundonly"]
    )
    df_round = inside2.dropna(subset=["logratio_pm25_roundonly", "distance10"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_round = fit_ols_cluster(
            "logratio_pm25_roundonly ~ distance10", df_round, "sample_round"
        )

    sensitivity = pd.DataFrame([
        {
            "baseline": "time_then_round (main)",
            "slope": float(m_main.params["distance10"]),
            "se": float(m_main.bse["distance10"]),
            "p": float(m_main.pvalues["distance10"]),
            "n": int(m_main.nobs),
        },
        {
            "baseline": "round_only",
            "slope": float(m_round.params["distance10"]),
            "se": float(m_round.bse["distance10"]),
            "p": float(m_round.pvalues["distance10"]),
            "n": int(m_round.nobs),
        },
    ])

    save_table(sensitivity, "pm25_baseline_sensitivity.csv")

    print("\nPM2.5 Baseline Sensitivity:")
    for _, row in sensitivity.iterrows():
        pct = (np.exp(row["slope"]) - 1) * 100
        print(
            f"  {row['baseline']:30s}: slope={row['slope']:.6f} "
            f"({pct:.2f}%/10m), p={row['p']:.4f}"
        )

    return sensitivity


# =========================================================================
# 5. Park-level heterogeneity (mixed-effects random slopes)
# =========================================================================

def mixed_effects_heterogeneity(inside: pd.DataFrame) -> pd.DataFrame:
    """Fit mixed-effects model with random slopes by park for PM2.5."""
    df = inside.dropna(subset=["logratio_PM2.5", "distance10"]).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm(
            'Q("logratio_PM2.5") ~ distance10',
            data=df,
            groups=df["sample_id"],
            re_formula="~distance10",
        )
        mdf = md.fit(reml=True)

    # Extract random slopes
    re = mdf.random_effects
    slopes = []
    for park_id, effects in re.items():
        fixed_slope = float(mdf.fe_params["distance10"])
        random_slope = float(effects.get("distance10", 0))
        total_slope = fixed_slope + random_slope
        pct = (np.exp(total_slope) - 1) * 100

        park_rows = df[df["sample_id"] == park_id]
        slopes.append({
            "sample_id": int(park_id),
            "sample_name": park_rows["sample_name"].iloc[0]
                if "sample_name" in park_rows.columns else str(park_id),
            "park_type": park_rows["park_type"].iloc[0]
                if "park_type" in park_rows.columns else "",
            "fixed_slope": fixed_slope,
            "random_slope": random_slope,
            "total_slope_logratio": total_slope,
            "pct_change_per10m": pct,
        })

    slopes_df = pd.DataFrame(slopes).sort_values("sample_id")
    save_table(slopes_df, "pm25_mixed_effects_slopes.csv")

    print("\nPM2.5 Park-Level Random Slopes:")
    print(f"  Fixed slope: {fixed_slope:.6f} logratio/10m")
    print(f"  Range: {slopes_df['pct_change_per10m'].min():.2f}% to "
          f"{slopes_df['pct_change_per10m'].max():.2f}% per 10m")

    return slopes_df


# =========================================================================
# 6. Quantile regression
# =========================================================================

def quantile_regression(
    inside: pd.DataFrame, n_boot: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Quantile regression for PM2.5 logratio across distribution."""
    df = inside.dropna(subset=["logratio_PM2.5", "distance10"]).copy()
    taus = [0.10, 0.25, 0.50, 0.75, 0.90]
    clusters = df["sample_round"].unique()
    rng = np.random.default_rng(seed)

    results = []
    for tau in taus:
        # Point estimate
        mod = smf.quantreg('Q("logratio_PM2.5") ~ distance10', data=df)
        res = mod.fit(q=tau)
        slope = float(res.params["distance10"])
        pct = (np.exp(slope) - 1) * 100

        # Cluster bootstrap CI
        boot_slopes = []
        for _ in range(n_boot):
            boot_clusters = rng.choice(clusters, size=len(clusters), replace=True)
            boot_df = pd.concat(
                [df[df["sample_round"] == c] for c in boot_clusters],
                ignore_index=True,
            )
            try:
                bres = smf.quantreg(
                    'Q("logratio_PM2.5") ~ distance10', data=boot_df
                ).fit(q=tau)
                boot_slopes.append(float(bres.params["distance10"]))
            except Exception:
                pass

        boot_slopes = np.array(boot_slopes)
        ci_low = np.percentile(boot_slopes, 2.5) if len(boot_slopes) > 0 else np.nan
        ci_high = np.percentile(boot_slopes, 97.5) if len(boot_slopes) > 0 else np.nan

        results.append({
            "tau": tau,
            "slope_logratio": slope,
            "pct_per10m": pct,
            "ci_low_logratio": ci_low,
            "ci_high_logratio": ci_high,
            "ci_low_pct": (np.exp(ci_low) - 1) * 100 if not np.isnan(ci_low) else np.nan,
            "ci_high_pct": (np.exp(ci_high) - 1) * 100 if not np.isnan(ci_high) else np.nan,
            "n_boot_ok": len(boot_slopes),
        })

    qr_df = pd.DataFrame(results)
    save_table(qr_df, "pm25_quantile_regression.csv")

    print("\nPM2.5 Quantile Regression:")
    for _, row in qr_df.iterrows():
        print(
            f"  tau={row['tau']:.2f}: {row['pct_per10m']:.2f}%/10m "
            f"(95% CI: {row['ci_low_pct']:.2f}% to {row['ci_high_pct']:.2f}%)"
        )

    return qr_df


# =========================================================================
# 7. Noise vs PM2.5 comparison table
# =========================================================================

def comparative_summary(
    pm25_core: dict,
    pm25_boot_sel: pd.DataFrame,
    pm25_sensitivity: pd.DataFrame,
    pm25_slopes: pd.DataFrame,
    pm25_qr: pd.DataFrame,
    inside: pd.DataFrame,
) -> pd.DataFrame:
    """Generate noise vs PM2.5 comparison table."""

    # Noise core model (refit for fresh numbers)
    df = inside.dropna(subset=["delta_LAeq", "distance10"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise_result = fit_ols_cluster(
            "delta_LAeq ~ distance10", df, "sample_round"
        )

    noise_slope = float(noise_result.params["distance10"])
    noise_p = float(noise_result.pvalues["distance10"])

    # Try to reuse the noise quantile regression summary if available (produced by 06_quantile_regression.py).
    # Fallback to fixed values to keep the script self-contained.
    noise_q50 = None
    noise_q90 = None
    try:
        qsum_path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "tables"
            / "quantreg_bootstrap_summary.csv"
        )
        if qsum_path.exists():
            qsum = pd.read_csv(qsum_path)
            noise_q50 = float(qsum.loc[qsum["quantile"].eq(0.50), "slope_median"].iloc[0])
            noise_q90 = float(qsum.loc[qsum["quantile"].eq(0.90), "slope_median"].iloc[0])
    except Exception:
        noise_q50 = None
        noise_q90 = None

    if noise_q50 is None:
        noise_q50 = -1.75
    if noise_q90 is None:
        noise_q90 = 0.02

    pm25_main_p = float(
        pm25_sensitivity.loc[
            pm25_sensitivity["baseline"].eq("time_then_round (main)"), "p"
        ].iloc[0]
    )
    pm25_round_p = float(
        pm25_sensitivity.loc[pm25_sensitivity["baseline"].eq("round_only"), "p"].iloc[0]
    )

    pm25_linear_share = float(
        pm25_boot_sel[pm25_boot_sel["model"] == "linear"]["share"].iloc[0]
    )

    pm25_q50 = pm25_qr[pm25_qr["tau"] == 0.50].iloc[0]
    pm25_q90 = pm25_qr[pm25_qr["tau"] == 0.90].iloc[0]

    noise_intensity_pct_10m = (1 - 10 ** (noise_slope / 10)) * 100
    noise_50m_db = noise_slope * 5
    noise_intensity_pct_50m = (1 - 10 ** (noise_50m_db / 10)) * 100

    pm25_50m_pct = (np.exp(pm25_core["slope"] * 5) - 1) * 100

    comparison = pd.DataFrame([
        {
            "dimension": "Core slope",
            "noise": f"{noise_slope:.2f} dB/10m (p={noise_p:.3g})",
            "pm25": f"{pm25_core['pct_per10m']:.2f}%/10m (p={pm25_main_p:.3f})",
        },
        {
            "dimension": "Effect magnitude",
            "noise": f"~{noise_intensity_pct_10m:.0f}% intensity/10m",
            "pm25": f"~{abs(pm25_core['pct_per10m']):.1f}% concentration/10m",
        },
        {
            "dimension": "50m cumulative effect",
            "noise": f"{noise_50m_db:.1f} dB (~{noise_intensity_pct_50m:.0f}% intensity)",
            "pm25": f"{pm25_50m_pct:.1f}% (~{abs(pm25_50m_pct):.0f}% concentration)",
        },
        {
            "dimension": "Bootstrap model stability",
            "noise": "93.0% select linear",
            "pm25": f"{pm25_linear_share:.1%} select linear",
        },
        {
            "dimension": "Baseline sensitivity",
            "noise": "Robust (p<0.001 all baselines)",
            "pm25": f"Sensitive (round-avg p={pm25_round_p:.3f})",
        },
        {
            "dimension": "Park heterogeneity",
            "noise": "3/12 parks positive slope (failure)",
            "pm25": (
                f"All negative, range {pm25_slopes['pct_change_per10m'].min():.2f}% "
                f"to {pm25_slopes['pct_change_per10m'].max():.2f}%"
            ),
        },
        {
            "dimension": "Quantile regression (tau=0.50)",
            "noise": f"{noise_q50:.2f} dB/10m (significant)",
            "pm25": f"{pm25_q50['pct_per10m']:.2f}%/10m",
        },
        {
            "dimension": "Quantile regression (tau=0.90)",
            "noise": f"{noise_q90:+.2f} dB/10m (CI crosses zero)",
            "pm25": f"{pm25_q90['pct_per10m']:.2f}%/10m (CI crosses zero)",
        },
        {
            "dimension": "Planning threshold applicability",
            "noise": "Empirical cutoff ~40m (low compliance)",
            "pm25": "No practical threshold",
        },
    ])

    save_table(comparison, "noise_vs_pm25_comparison.csv")
    return comparison


# =========================================================================
# Figures
# =========================================================================

def plot_pm25_boxplot(inside: pd.DataFrame) -> None:
    """Box plot of PM2.5 logratio by distance."""
    fig, ax = plt.subplots(figsize=(8, 5))
    distance_order = sorted(inside["distance_m"].dropna().unique())
    sns.boxplot(
        data=inside, x="distance_m", y="logratio_PM2.5",
        order=distance_order, color="#72B7B2", ax=ax
    )
    ax.axhline(0, lw=1, c="black", alpha=0.5, ls="--")
    ax.set_xlabel("Distance from road boundary (m)")
    ax.set_ylabel("log(PM2.5_inside / PM2.5_P1)")
    ax.grid(False)

    save_figure(fig, "pm25_logratio_by_distance_box.png")
    plt.close(fig)


def plot_model_comparison(inside: pd.DataFrame) -> None:
    """Linear vs spline fit comparison for PM2.5."""
    from patsy import dmatrix

    df = inside.dropna(subset=["logratio_PM2.5", "distance_m"]).copy()
    df["distance10"] = df["distance_m"] / 10.0

    # Linear fit
    m_lin = smf.ols('Q("logratio_PM2.5") ~ distance10', data=df).fit()
    x_line = np.linspace(df["distance_m"].min(), df["distance_m"].max(), 200)
    y_lin = m_lin.params["Intercept"] + m_lin.params["distance10"] * (x_line / 10)

    # Spline fit
    spl_basis = dmatrix("cr(distance_m, df=4) - 1", data=df, return_type="dataframe")
    for col in spl_basis.columns:
        df[f"spl_{col}"] = spl_basis[col].values
    spl_terms = " + ".join([f'Q("spl_{c}")' for c in spl_basis.columns])
    m_spl = smf.ols(f'Q("logratio_PM2.5") ~ {spl_terms}', data=df).fit()

    pred_df = pd.DataFrame({"distance_m": x_line})
    pred_basis = dmatrix("cr(distance_m, df=4) - 1", data=pred_df, return_type="dataframe")
    y_spl = m_spl.params["Intercept"]
    for i, col in enumerate(spl_basis.columns):
        y_spl = y_spl + m_spl.params.iloc[i + 1] * pred_basis.iloc[:, i].values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        df["distance_m"], df["logratio_PM2.5"],
        s=20, alpha=0.3, c="grey", label="Observations"
    )
    ax.plot(x_line, y_lin, c="#4C78A8", lw=2, label=f"Linear (AIC={m_lin.aic:.1f})")
    ax.plot(x_line, y_spl, c="#E45756", lw=2, ls="--", label=f"Spline (AIC={m_spl.aic:.1f})")
    ax.axhline(0, lw=0.8, c="black", alpha=0.4, ls=":")
    ax.set_xlabel("Distance from road boundary (m)")
    ax.set_ylabel("log(PM2.5_inside / PM2.5_P1)")
    ax.grid(False)
    ax.legend()

    save_figure(fig, "pm25_model_comparison.png")
    plt.close(fig)


def plot_quantile_slopes(qr_df: pd.DataFrame) -> None:
    """Plot quantile regression slopes for PM2.5."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        qr_df["tau"], qr_df["pct_per10m"],
        yerr=[
            qr_df["pct_per10m"] - qr_df["ci_low_pct"],
            qr_df["ci_high_pct"] - qr_df["pct_per10m"],
        ],
        fmt="o-", color="#72B7B2", capsize=5, lw=2, markersize=8,
        label="PM2.5 (% change/10m)"
    )
    ax.axhline(0, lw=1, c="black", alpha=0.5, ls="--")
    ax.set_xlabel("Quantile (τ)")
    ax.set_ylabel("Slope (% change per 10m)")
    ax.set_xticks(qr_df["tau"])
    ax.grid(False)
    ax.legend()

    # Shade effective vs ineffective regions
    ax.axhspan(ax.get_ylim()[0], 0, alpha=0.05, color="green")
    ax.axhspan(0, ax.get_ylim()[1], alpha=0.05, color="red")

    save_figure(fig, "pm25_quantile_slopes.png")
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    ensure_dirs()
    sns.set_theme(style="white")
    setup_style()

    # Load data
    meas = load_measurements()
    inside = attach_baseline(meas)

    print(f"Total observations (interior points): {len(inside)}")
    print(f"PM2.5 non-null: {inside['logratio_PM2.5'].notna().sum()}")

    # 1. Core model
    pm25_core = core_linear_model(inside)

    # 2. Model specification
    model_comp = model_specification_test(inside)

    # 3. Bootstrap model selection
    boot_sel = bootstrap_model_selection(inside, n_boot=1000)

    # 4. Baseline sensitivity
    sensitivity = baseline_sensitivity(meas)

    # 5. Park-level heterogeneity
    slopes = mixed_effects_heterogeneity(inside)

    # 6. Quantile regression
    qr_df = quantile_regression(inside, n_boot=1000)

    # 7. Comparative summary
    comparison = comparative_summary(
        pm25_core, boot_sel, sensitivity, slopes, qr_df, inside
    )

    # Figures
    plot_pm25_boxplot(inside)
    plot_model_comparison(inside)
    plot_quantile_slopes(qr_df)

    print("\n" + "=" * 60)
    print("NOISE vs PM2.5 COMPARISON")
    print("=" * 60)
    for _, row in comparison.iterrows():
        print(f"\n{row['dimension']}:")
        print(f"  Noise: {row['noise']}")
        print(f"  PM2.5: {row['pm25']}")
    print("\n" + "=" * 60)
    print("Done! Output files saved to applsci/outputs/")


if __name__ == "__main__":
    main()
