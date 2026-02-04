"""
03_mixed_effects_model.py
=========================

Mixed-effects model with random slopes to quantify park-level heterogeneity
in noise attenuation effectiveness.

Model specification:
    delta_LAeq ~ distance10 + (1 + distance10 | park) + (1 | sample_round)

This model allows each park to have its own distance-attenuation slope,
while accounting for within-round correlation.

Output:
    - tables/mixedlm_fixed_effects.csv
    - tables/mixedlm_random_slopes.csv
    - figures/mixedlm_park_slopes.png
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
    park_metadata,
    save_figure,
    save_table,
)


def fit_random_slope_model(data: pd.DataFrame):
    """
    Fit mixed-effects model with random intercept and slope by park.

    Parameters
    ----------
    data : pd.DataFrame
        Data with delta_LAeq, distance10, sample_id, sample_round

    Returns
    -------
    MixedLMResults
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            "delta_LAeq ~ distance10",
            data=data,
            groups=data["sample_id"],
            re_formula="1 + distance10",
            vc_formula={"sample_round": "0 + C(sample_round)"},
        )
        return model.fit(method="lbfgs", maxiter=200, disp=False)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    # Load and prepare data
    meas = load_measurements()
    inside = attach_baseline(meas).copy()
    meta = park_metadata(meas)

    print(f"Total observations: {len(inside)}")
    print(f"Number of parks: {inside['sample_id'].nunique()}")

    # =========================================================================
    # Fit random-slope mixed-effects model
    # =========================================================================
    print("\nFitting mixed-effects model with random slopes...")

    res = fit_random_slope_model(inside)

    # Extract fixed effects
    fe_df = pd.DataFrame({
        "term": res.fe_params.index,
        "coef": res.fe_params.values,
        "se": [res.bse_fe.get(t, np.nan) for t in res.fe_params.index],
        "z": [res.tvalues.get(t, np.nan) for t in res.fe_params.index],
        "p": [res.pvalues.get(t, np.nan) for t in res.fe_params.index],
    })
    save_table(fe_df, "mixedlm_fixed_effects.csv")

    print("\nFixed Effects:")
    print(fe_df.to_string(index=False))

    # =========================================================================
    # Extract random slopes (BLUP) for each park
    # =========================================================================
    re = pd.DataFrame.from_dict(res.random_effects, orient="index")
    re.index.name = "sample_id"
    re = re.reset_index()

    # Rename columns based on what's available
    if "Group" in re.columns:
        re = re.rename(columns={"Group": "re_intercept"})
    if "distance10" in re.columns:
        re = re.rename(columns={"distance10": "re_slope"})
    else:
        re["re_slope"] = 0.0

    re["sample_id"] = re["sample_id"].astype(int)

    # Compute total slope = fixed + random
    fixed_slope = float(res.fe_params["distance10"])
    if "re_slope" in re.columns:
        re["total_slope_per10m"] = fixed_slope + re["re_slope"]
    else:
        re["total_slope_per10m"] = fixed_slope

    # Merge with metadata
    re = meta[["sample_id", "sample_name", "park_type", "park_area"]].merge(
        re, on="sample_id", how="right"
    )
    re = re.sort_values("sample_id")
    save_table(
        re[["sample_id", "sample_name", "park_type", "park_area", "total_slope_per10m"]],
        "mixedlm_random_slopes.csv"
    )

    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("RANDOM SLOPES SUMMARY")
    print("=" * 60)
    print(f"Fixed slope (population average): {fixed_slope:.3f} dB/10m")
    print(f"Park-specific slope range: [{re['total_slope_per10m'].min():.3f}, "
          f"{re['total_slope_per10m'].max():.3f}]")

    # Identify parks with positive slopes (ineffective buffer)
    positive_slope_parks = re[re["total_slope_per10m"] > 0]
    if len(positive_slope_parks) > 0:
        print(f"\nParks with POSITIVE slope (ineffective buffer):")
        for _, row in positive_slope_parks.iterrows():
            print(f"  - {row['sample_name']}: {row['total_slope_per10m']:.3f} dB/10m")

    # =========================================================================
    # Visualization: Park-specific slopes
    # =========================================================================
    plot_data = re.sort_values("total_slope_per10m")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plot_data["total_slope_per10m"].apply(
        lambda x: "#E45756" if x > 0 else "#4C78A8"
    )
    bars = ax.barh(
        range(len(plot_data)),
        plot_data["total_slope_per10m"],
        color=colors
    )
    ax.axvline(fixed_slope, lw=2, ls="--", c="black", alpha=0.7,
               label=f"Fixed slope = {fixed_slope:.2f}")
    ax.axvline(0, lw=1, c="gray", alpha=0.5)

    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data["sample_name"])
    ax.set_xlabel("Slope (dB per 10m)")
    ax.set_title("Park-Specific Distance-Attenuation Slopes\n(Mixed-Effects Random Slopes)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    save_figure(fig, "mixedlm_park_slopes.png")
    plt.close(fig)

    # =========================================================================
    # Model fit summary
    # =========================================================================
    print(f"\nModel AIC: {res.aic:.1f}")
    print(f"Model BIC: {res.bic:.1f}")

    print("\nDone! Output files saved to applsci/outputs/")


if __name__ == "__main__":
    main()
