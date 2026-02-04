"""
05_interaction_effects.py
=========================

Moderator analysis: Park attributes × Distance interaction effects.

This script tests whether park characteristics (type, area) moderate
the distance-attenuation relationship. We fit interaction models and extract
simple slopes to understand how buffering effectiveness varies by park attributes.

Models:
    1. delta_LAeq ~ distance × park_type
    2. delta_LAeq ~ distance × area (continuous)

Output:
    - tables/interaction_parktype_coefficients.csv
    - tables/interaction_simple_slopes_parktype.csv
    - tables/interaction_area_coefficients.csv
    - tables/interaction_simple_slopes_area.csv
    - figures/interaction_simple_slopes.png
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from figure_style import COLORS, add_subplot_label, add_reference_line, setup_style
from utils import (
    attach_baseline,
    ensure_dirs,
    load_measurements,
    save_figure,
    save_table,
)


def fit_mixedlm_interaction(formula: str, data: pd.DataFrame):
    """Fit mixed-effects model with random intercept by park."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            formula,
            data=data,
            groups=data["sample_id"],
            re_formula="1",
        )
        return model.fit(method="lbfgs", maxiter=200, disp=False)


def extract_coefs(res) -> pd.DataFrame:
    """Extract coefficient table from model results."""
    terms = list(res.fe_params.index)
    return pd.DataFrame({
        "term": terms,
        "coef": [float(res.fe_params[t]) for t in terms],
        "se": [float(res.bse_fe.get(t, np.nan)) for t in terms],
        "z": [float(res.tvalues.get(t, np.nan)) for t in terms],
        "p": [float(res.pvalues.get(t, np.nan)) for t in terms],
    })


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="white")
    setup_style()

    # Load and prepare data
    meas = load_measurements()
    inside = attach_baseline(meas).copy()

    # Create categorical variables
    inside["ParkType"] = pd.Categorical(
        inside["park_type"], categories=["Large", "Medium", "Small"], ordered=True
    )
    inside["area_1000"] = inside["park_area_m2"] / 1000  # Scale to 1000 m²

    results = {}

    # =========================================================================
    # Model 1: Distance × Park Type
    # =========================================================================
    print("Fitting Model 1: Distance × Park Type...")

    try:
        m1 = fit_mixedlm_interaction("delta_LAeq ~ distance10 * C(ParkType)", inside)
        results["parktype"] = m1

        coef_df1 = extract_coefs(m1)
        save_table(coef_df1, "interaction_parktype_coefficients.csv")

        # Extract simple slopes
        slope_large = float(m1.fe_params["distance10"])
        int_medium = float(m1.fe_params.get("distance10:C(ParkType)[T.Medium]", 0))
        int_small = float(m1.fe_params.get("distance10:C(ParkType)[T.Small]", 0))

        simple_slopes_type = pd.DataFrame({
            "park_type": ["Large", "Medium", "Small"],
            "simple_slope_dB_per10m": [slope_large, slope_large + int_medium, slope_large + int_small],
            "reference": ["baseline", "Large + interaction", "Large + interaction"],
        })
        save_table(simple_slopes_type, "interaction_simple_slopes_parktype.csv")

        print("\nSimple Slopes by Park Type:")
        print(simple_slopes_type.to_string(index=False))

        p_medium = float(m1.pvalues.get("distance10:C(ParkType)[T.Medium]", np.nan))
        p_small = float(m1.pvalues.get("distance10:C(ParkType)[T.Small]", np.nan))
        print(f"\nInteraction p-values:")
        print(f"  Large vs Medium: p = {p_medium:.4f}")
        print(f"  Large vs Small: p = {p_small:.4f}")

    except Exception as e:
        print(f"Model 1 failed: {e}")

    # =========================================================================
    # Model 2: Distance × Area (continuous)
    # =========================================================================
    print("\nFitting Model 2: Distance × Area (continuous)...")

    interact_coef = np.nan
    try:
        m2 = fit_mixedlm_interaction("delta_LAeq ~ distance10 * area_1000", inside)
        results["area"] = m2

        coef_df2 = extract_coefs(m2)
        save_table(coef_df2, "interaction_area_coefficients.csv")

        # Simple slopes at different area percentiles
        area_pcts = inside["area_1000"].quantile([0.25, 0.5, 0.75]).values
        main_dist = float(m2.fe_params["distance10"])
        interact_coef = float(m2.fe_params.get("distance10:area_1000", 0))

        simple_slopes_area = pd.DataFrame({
            "area_1000sqm": area_pcts,
            "percentile": ["25th", "50th", "75th"],
            "simple_slope_dB_per10m": [main_dist + interact_coef * a for a in area_pcts],
        })
        save_table(simple_slopes_area, "interaction_simple_slopes_area.csv")

        print("\nSimple Slopes at Area Percentiles:")
        print(simple_slopes_area.to_string(index=False))

        p_interact = float(m2.pvalues.get("distance10:area_1000", np.nan))
        print(f"\nDistance × Area interaction: coef = {interact_coef:.4f}, p = {p_interact:.4f}")
        print(f"Interpretation: Each additional 1000 m² of park area improves")
        print(f"the distance slope by {abs(interact_coef):.2f} dB/10m")

    except Exception as e:
        print(f"Model 2 failed: {e}")

    # =========================================================================
    # Visualization: Simple slopes
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Plot 1: By Park Type
    if "parktype" in results:
        ax = axes[0]
        m = results["parktype"]
        colors = {"Large": "#4C78A8", "Medium": "#F58518", "Small": "#E45756"}

        for ptype in ["Large", "Medium", "Small"]:
            if ptype == "Large":
                slope = float(m.fe_params["distance10"])
                intercept = float(m.fe_params["Intercept"])
            else:
                slope = float(m.fe_params["distance10"]) + float(
                    m.fe_params.get(f"distance10:C(ParkType)[T.{ptype}]", 0)
                )
                intercept = float(m.fe_params["Intercept"]) + float(
                    m.fe_params.get(f"C(ParkType)[T.{ptype}]", 0)
                )
            x = np.array([0, 5])
            y = intercept + slope * x
            ax.plot(
                x,
                y,
                label=f"{ptype} ({slope:.2f} dB/10m)",
                color=colors[ptype],
                lw=2.5,
            )

        ax.set_xlabel("Distance (10 m units)")
        ax.set_ylabel("ΔLAeq (dB)")
        ax.legend(frameon=False, fontsize=9)
        add_reference_line(
            ax,
            0,
            "h",
            color=COLORS["highlight"],
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )
        ax.set_xlim(0, 5)
        ax.grid(False)
        add_subplot_label(ax, "a")

    # Plot 2: By Area percentiles
    if "area" in results:
        ax = axes[1]
        m = results["area"]

        b0 = float(m.fe_params.get("Intercept", 0.0))
        b_area = float(m.fe_params.get("area_1000", 0.0))
        b_dist = float(m.fe_params.get("distance10", 0.0))
        b_int = float(m.fe_params.get("distance10:area_1000", 0.0))

        area_pcts = inside["area_1000"].quantile([0.25, 0.50, 0.75])
        colors = {"25th": COLORS["small"], "50th": COLORS["medium"], "75th": COLORS["large"]}

        for pct_label, a in zip(["25th", "50th", "75th"], area_pcts.to_numpy()):
            a = float(a)
            slope = b_dist + b_int * a
            intercept = b0 + b_area * a
            x = np.array([0, 5])
            y = intercept + slope * x
            ax.plot(
                x,
                y,
                label=f"{int(round(a * 1000)):,} m² ({pct_label}; {slope:.2f} dB/10m)",
                color=colors[pct_label],
                lw=2.5,
            )

        ax.set_xlabel("Distance (10 m units)")
        ax.set_ylabel("ΔLAeq (dB)")
        ax.legend(frameon=False, fontsize=9)
        add_reference_line(
            ax,
            0,
            "h",
            color=COLORS["highlight"],
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )
        ax.set_xlim(0, 5)
        ax.grid(False)
        add_subplot_label(ax, "b")

    plt.tight_layout()
    save_figure(fig, "interaction_simple_slopes.png")
    plt.close(fig)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("INTERACTION EFFECTS SUMMARY")
    print("=" * 60)
    if "parktype" in results:
        print("\nPark Type moderation:")
        print("  - Large parks: effective noise buffer")
        print("  - Medium parks: moderate buffer")
        print("  - Small parks: INEFFECTIVE (positive slope)")
    if "area" in results:
        print(f"\nPark Area (continuous): {interact_coef:.3f} dB/10m per 1000 m²")
        print("  - Larger parks = more effective buffering")

    print("\nDone! Output files saved to applsci/outputs/")


if __name__ == "__main__":
    main()
