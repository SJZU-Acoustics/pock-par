"""
generate_dual_panel_figures.py
==============================

Generate dual-panel comparison figures for noise vs PM2.5 attenuation.

Figure 3: Box plots — (a) ΔLAeq by distance, (b) logratio_PM2.5 by distance
Figure 4: Model comparison — (a) Noise linear vs spline, (b) PM2.5 linear vs spline

Uses utils.py for data loading and figure_style.py for consistent styling.
"""

from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from patsy import dmatrix

from figure_style import (
    COLORS,
    FIG_SIZES,
    MATH_DELTA_L_AEQ,
    MATH_LOG_PM25_RATIO,
    add_subplot_label,
    add_reference_line,
    setup_style,
)
from utils import (
    attach_baseline,
    ensure_dirs,
    load_measurements,
    FIG_DIR,
)

setup_style()


# ============================================================================
# FIGURE 3: Dual-panel box plot
# ============================================================================

def figure3_dual_boxplot(inside: pd.DataFrame) -> plt.Figure:
    """
    Figure 3: Dual-panel box plots comparing noise and PM2.5 by distance.
    (a) ΔLAeq by distance — clear negative trend
    (b) logratio_PM2.5 by distance — no clear trend
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.8))

    # Shared distance groups (exclude rare distances 15, 25)
    main_distances = [5, 10, 20, 30, 40, 50]
    df = inside[inside["distance_m"].isin(main_distances)].copy()

    # ---- Panel (a): Noise ΔLAeq ----
    noise_data = [
        df[df["distance_m"] == d]["delta_LAeq"].dropna().values for d in main_distances
    ]

    bp_a = ax_a.boxplot(
        noise_data,
        positions=range(len(main_distances)),
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": COLORS["light_gray"],
            "markeredgecolor": COLORS["secondary"],
        },
    )
    for patch in bp_a["boxes"]:
        patch.set_facecolor(COLORS["medium"])
        patch.set_edgecolor(COLORS["primary"])
        patch.set_alpha(0.7)
    for w in bp_a["whiskers"]:
        w.set_color(COLORS["primary"])
        w.set_linewidth(1)
    for c in bp_a["caps"]:
        c.set_color(COLORS["primary"])
        c.set_linewidth(1)
    for m in bp_a["medians"]:
        m.set_color(COLORS["highlight"])
        m.set_linewidth(2)

    # Mean trend line
    means_a = [np.nanmean(d) for d in noise_data]
    ax_a.plot(
        range(len(main_distances)),
        means_a,
        "o-",
        color=COLORS["large"],
        linewidth=2,
        markersize=5,
        label="Mean",
        zorder=5,
    )

    add_reference_line(
        ax_a, 0, "h", color=COLORS["highlight"], linestyle="--", linewidth=1, alpha=0.7
    )

    ax_a.set_xticks(range(len(main_distances)))
    ax_a.set_xticklabels([str(d) for d in main_distances])
    ax_a.set_xlabel("Distance from road boundary (m)")
    ax_a.set_ylabel(f"Relative noise level {MATH_DELTA_L_AEQ} (dB)")
    ax_a.set_ylim(-25, 15)
    ax_a.grid(False)
    ax_a.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Sample size annotations
    for i, d in enumerate(main_distances):
        n = len(noise_data[i])
        ax_a.text(
            i, ax_a.get_ylim()[0] + 0.8, f"n={n}",
            ha="center", va="bottom", fontsize=7, color=COLORS["secondary"],
        )

    add_subplot_label(ax_a, "a")

    # ---- Panel (b): PM2.5 logratio ----
    # Remove extreme outliers beyond Q1/Q3 ± 3×IQR before plotting
    pm_data_raw = [
        df[df["distance_m"] == d]["logratio_PM2.5"].dropna().values for d in main_distances
    ]
    pm_data = []
    for arr in pm_data_raw:
        if len(arr) == 0:
            pm_data.append(arr)
            continue
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        pm_data.append(arr[(arr >= q1 - 3 * iqr) & (arr <= q3 + 3 * iqr)])

    bp_b = ax_b.boxplot(
        pm_data,
        positions=range(len(main_distances)),
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": COLORS["light_gray"],
            "markeredgecolor": COLORS["secondary"],
        },
    )
    for patch in bp_b["boxes"]:
        patch.set_facecolor("#72B7B2")
        patch.set_edgecolor(COLORS["primary"])
        patch.set_alpha(0.7)
    for w in bp_b["whiskers"]:
        w.set_color(COLORS["primary"])
        w.set_linewidth(1)
    for c in bp_b["caps"]:
        c.set_color(COLORS["primary"])
        c.set_linewidth(1)
    for m in bp_b["medians"]:
        m.set_color(COLORS["highlight"])
        m.set_linewidth(2)

    # Mean trend line
    means_b = [np.nanmean(d) if len(d) > 0 else np.nan for d in pm_data]
    ax_b.plot(
        range(len(main_distances)),
        means_b,
        "o-",
        color="#2E8B7F",
        linewidth=2,
        markersize=5,
        label="Mean",
        zorder=5,
    )

    add_reference_line(
        ax_b, 0, "h", color=COLORS["highlight"], linestyle="--", linewidth=1, alpha=0.7
    )

    ax_b.set_xticks(range(len(main_distances)))
    ax_b.set_xticklabels([str(d) for d in main_distances])
    ax_b.set_xlabel("Distance from road boundary (m)")
    ax_b.set_ylabel(MATH_LOG_PM25_RATIO)
    ax_b.grid(False)
    ax_b.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Sample size annotations
    for i, d in enumerate(main_distances):
        n = len(pm_data[i])
        ax_b.text(
            i, ax_b.get_ylim()[0] + (ax_b.get_ylim()[1] - ax_b.get_ylim()[0]) * 0.02,
            f"n={n}",
            ha="center", va="bottom", fontsize=7, color=COLORS["secondary"],
        )

    add_subplot_label(ax_b, "b")

    fig.tight_layout(w_pad=3.0)
    return fig


# ============================================================================
# FIGURE 4: Dual-panel model comparison
# ============================================================================

def figure4_dual_model_comparison(inside: pd.DataFrame) -> plt.Figure:
    """
    Figure 4: Dual-panel linear vs spline model comparison.
    (a) Noise: linear, log, and spline fits — linear preferred
    (b) PM2.5: linear, log, and spline fits — no clear preference
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.8))

    df = inside.dropna(subset=["delta_LAeq", "logratio_PM2.5", "distance_m"]).copy()

    # Remove extreme PM2.5 outliers beyond Q1/Q3 ± 3×IQR (for panel b only)
    pm_vals = df["logratio_PM2.5"]
    q1, q3 = pm_vals.quantile([0.25, 0.75])
    iqr = q3 - q1
    df_pm = df[(pm_vals >= q1 - 3 * iqr) & (pm_vals <= q3 + 3 * iqr)].copy()

    df["distance10"] = df["distance_m"] / 10.0
    df["log_distance"] = np.log(df["distance_m"].clip(lower=1))
    df_pm["distance10"] = df_pm["distance_m"] / 10.0
    df_pm["log_distance"] = np.log(df_pm["distance_m"].clip(lower=1))

    x_pred = np.linspace(df["distance_m"].min(), df["distance_m"].max(), 200)
    x_pred10 = x_pred / 10.0
    x_pred_log = np.log(np.clip(x_pred, 1, None))

    # ---- Panel (a): Noise ----
    ax_a.scatter(
        df["distance_m"], df["delta_LAeq"],
        alpha=0.25, s=15, c=COLORS["secondary"], edgecolors="none", label="Observations",
    )

    # Linear
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_lin = smf.ols("delta_LAeq ~ distance10", data=df).fit()
    y_lin = m_lin.params["Intercept"] + m_lin.params["distance10"] * x_pred10
    ax_a.plot(x_pred, y_lin, "-", color=COLORS["large"], lw=2,
              label=f"Linear (AIC={m_lin.aic:.1f})")

    # CI band for linear
    n = len(df)
    se_fit = np.sqrt(np.sum(m_lin.resid ** 2) / (n - 2))
    x_mean = df["distance10"].mean()
    ci = 1.96 * se_fit * np.sqrt(
        1 / n + (x_pred10 - x_mean) ** 2 / np.sum((df["distance10"] - x_mean) ** 2)
    )
    ax_a.fill_between(x_pred, y_lin - ci, y_lin + ci, color=COLORS["large"], alpha=0.12)

    # Log model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_log = smf.ols("delta_LAeq ~ log_distance", data=df).fit()
    y_log = m_log.params["Intercept"] + m_log.params["log_distance"] * x_pred_log
    ax_a.plot(x_pred, y_log, "--", color=COLORS["success"], lw=1.8,
              label=f"Log (AIC={m_log.aic:.1f})")

    # Spline
    spl_basis = dmatrix("cr(distance_m, df=4) - 1", data=df, return_type="dataframe")
    for col in spl_basis.columns:
        df[f"_spl_n_{col}"] = spl_basis[col].values
    spl_terms = " + ".join([f'Q("_spl_n_{c}")' for c in spl_basis.columns])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_spl = smf.ols(f"delta_LAeq ~ {spl_terms}", data=df).fit()
    pred_basis = dmatrix("cr(distance_m, df=4) - 1",
                         data=pd.DataFrame({"distance_m": x_pred}),
                         return_type="dataframe")
    y_spl = m_spl.params["Intercept"] + np.zeros(len(x_pred))
    for i, col in enumerate(spl_basis.columns):
        y_spl += m_spl.params.iloc[i + 1] * pred_basis.iloc[:, i].values
    ax_a.plot(x_pred, y_spl, "-.", color=COLORS["small"], lw=1.8,
              label=f"Spline (AIC={m_spl.aic:.1f})")

    add_reference_line(ax_a, 0, "h", color=COLORS["highlight"], linestyle=":", linewidth=1, alpha=0.6)

    ax_a.set_xlabel("Distance from road boundary (m)")
    ax_a.set_ylabel(f"Relative noise level {MATH_DELTA_L_AEQ} (dB)")
    ax_a.set_xlim(-2, 55)
    ax_a.set_ylim(-25, 15)
    ax_a.grid(False)
    ax_a.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax_a.text(
        0.97, 0.97, "Linear preferred\n(|ΔAIC| < 2; 93.0% bootstrap)",
        transform=ax_a.transAxes, ha="right", va="top",
        fontsize=7.5, style="italic", color=COLORS["secondary"],
    )
    add_subplot_label(ax_a, "a")

    # ---- Panel (b): PM2.5 (using df_pm with extreme outliers removed) ----
    ax_b.scatter(
        df_pm["distance_m"], df_pm["logratio_PM2.5"],
        alpha=0.25, s=15, c=COLORS["secondary"], edgecolors="none", label="Observations",
    )

    # Linear
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_lin_pm = smf.ols('Q("logratio_PM2.5") ~ distance10', data=df_pm).fit()
    y_lin_pm = m_lin_pm.params["Intercept"] + m_lin_pm.params["distance10"] * x_pred10
    ax_b.plot(x_pred, y_lin_pm, "-", color="#2E8B7F", lw=2,
              label=f"Linear (AIC={m_lin_pm.aic:.1f})")

    # CI band for linear
    n_pm = len(df_pm)
    se_fit_pm = np.sqrt(np.sum(m_lin_pm.resid ** 2) / (n_pm - 2))
    x_mean_pm = df_pm["distance10"].mean()
    ci_pm = 1.96 * se_fit_pm * np.sqrt(
        1 / n_pm + (x_pred10 - x_mean_pm) ** 2
        / np.sum((df_pm["distance10"] - x_mean_pm) ** 2)
    )
    ax_b.fill_between(x_pred, y_lin_pm - ci_pm, y_lin_pm + ci_pm,
                      color="#2E8B7F", alpha=0.12)

    # Spline
    spl_basis_pm = dmatrix("cr(distance_m, df=4) - 1", data=df_pm, return_type="dataframe")
    for col in spl_basis_pm.columns:
        df_pm[f"_spl_p_{col}"] = spl_basis_pm[col].values
    spl_terms_pm = " + ".join([f'Q("_spl_p_{c}")' for c in spl_basis_pm.columns])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_spl_pm = smf.ols(f'Q("logratio_PM2.5") ~ {spl_terms_pm}', data=df_pm).fit()
    y_spl_pm = m_spl_pm.params["Intercept"] + np.zeros(len(x_pred))
    for i, col in enumerate(spl_basis_pm.columns):
        y_spl_pm += m_spl_pm.params.iloc[i + 1] * pred_basis.iloc[:, i].values
    ax_b.plot(x_pred, y_spl_pm, "-.", color=COLORS["small"], lw=1.8,
              label=f"Spline (AIC={m_spl_pm.aic:.1f})")

    add_reference_line(ax_b, 0, "h", color=COLORS["highlight"], linestyle=":", linewidth=1, alpha=0.6)

    ax_b.set_xlabel("Distance from road boundary (m)")
    ax_b.set_ylabel(MATH_LOG_PM25_RATIO)
    ax_b.set_xlim(-2, 55)
    ax_b.grid(False)
    ax_b.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax_b.text(
        0.97, 0.97, "No preferred model\n(36.8% bootstrap linear)",
        transform=ax_b.transAxes, ha="right", va="top",
        fontsize=7.5, style="italic", color=COLORS["secondary"],
    )
    add_subplot_label(ax_b, "b")

    fig.tight_layout(w_pad=3.0)
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    ensure_dirs()

    print("Loading data...")
    meas = load_measurements()
    inside = attach_baseline(meas)
    print(f"Interior observations: {len(inside)}")

    # Figure 3: Dual-panel box plot
    print("\nGenerating Figure 3: Dual-panel box plot (noise vs PM2.5)...")
    fig3 = figure3_dual_boxplot(inside)
    out3 = FIG_DIR / "figure3_boxplot_noise_pm25.png"
    fig3.savefig(out3, dpi=300, bbox_inches="tight")
    fig3.savefig(str(out3).replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out3}")

    # Figure 4: Dual-panel model comparison
    print("\nGenerating Figure 4: Dual-panel model comparison (noise vs PM2.5)...")
    fig4 = figure4_dual_model_comparison(inside)
    out4 = FIG_DIR / "figure4_model_comparison_noise_pm25.png"
    fig4.savefig(out4, dpi=300, bbox_inches="tight")
    fig4.savefig(str(out4).replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: {out4}")

    print("\nDone! Dual-panel figures saved.")


if __name__ == "__main__":
    main()
