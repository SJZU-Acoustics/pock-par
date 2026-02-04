"""
09_robustness_checks.py
=======================

Targeted robustness checks requested during internal review.

Currently implemented:
  - Leave-one-park-out (LOPO) re-estimation for the core distance slope
    for both noise (delta_LAeq) and PM2.5 (logratio_PM2.5).

Outputs:
  - tables/lopo_noise_core_slope.csv
  - tables/lopo_pm25_core_slope.csv
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import attach_baseline, ensure_dirs, load_measurements, save_table

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))


def fit_ols_cluster(formula: str, data: pd.DataFrame, cluster_col: str):
    model = smf.ols(formula, data=data)
    return model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})


def lopo_core_slope(
    inside: pd.DataFrame,
    outcome: str,
    formula: str,
    cluster_col: str = "sample_round",
) -> pd.DataFrame:
    rows: list[dict] = []

    for sid in sorted(inside["sample_id"].unique()):
        df = inside.loc[inside["sample_id"].ne(sid)].copy()
        df = df.dropna(subset=[outcome, "distance10", cluster_col])

        if df["distance10"].nunique() < 2 or df[cluster_col].nunique() < 2:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_ols_cluster(formula, df, cluster_col)

        coef = float(res.params["distance10"])
        se = float(res.bse["distance10"])
        ci_low, ci_high = (coef - 1.96 * se, coef + 1.96 * se)

        rows.append(
            {
                "leave_out_sample_id": int(sid),
                "n_obs": int(res.nobs),
                "n_clusters": int(df[cluster_col].nunique()),
                "slope": coef,
                "se_cluster": se,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p": float(res.pvalues["distance10"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()

    meas = load_measurements()
    inside = attach_baseline(meas).copy()

    # Noise LOPO
    noise = inside.dropna(subset=["delta_LAeq", "distance10", "sample_round"])
    noise_lopo = lopo_core_slope(
        noise,
        outcome="delta_LAeq",
        formula="delta_LAeq ~ distance10",
        cluster_col="sample_round",
    )
    save_table(noise_lopo, "lopo_noise_core_slope.csv")

    # PM2.5 LOPO
    pm = inside.dropna(subset=["logratio_PM2.5", "distance10", "sample_round"])
    pm_lopo = lopo_core_slope(
        pm,
        outcome="logratio_PM2.5",
        formula='Q("logratio_PM2.5") ~ distance10',
        cluster_col="sample_round",
    )
    save_table(pm_lopo, "lopo_pm25_core_slope.csv")

    # Console summary
    print("\nLOPO summary (core slope):")
    if len(noise_lopo) > 0:
        print(
            "  Noise slope range: "
            f"[{noise_lopo['slope'].min():.3f}, {noise_lopo['slope'].max():.3f}] dB/10m; "
            f"p range [{noise_lopo['p'].min():.4f}, {noise_lopo['p'].max():.4f}]"
        )
    if len(pm_lopo) > 0:
        pct = (np.exp(pm_lopo["slope"]) - 1) * 100
        print(
            "  PM2.5 slope range: "
            f"[{pct.min():.2f}%, {pct.max():.2f}%]/10m; "
            f"p range [{pm_lopo['p'].min():.4f}, {pm_lopo['p'].max():.4f}]"
        )

    print("\nDone! Output files saved to applsci/outputs/")


if __name__ == "__main__":
    main()

