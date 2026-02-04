"""
04_bayesian_hierarchical.py
===========================

Bayesian hierarchical model for noise attenuation with park-specific slopes.

This script implements a fully Bayesian random-slope model using Gibbs sampling,
providing posterior distributions for:
- Global (population-average) distance effect
- Park-specific distance slopes
- Probability of "buffer failure" (P(slope > 0)) for each park

Model:
    y_ijk = beta_0 + beta_1 * x_ijk + b_0j + b_1j * x_ijk + u_k + epsilon_ijk

where:
    - y_ijk = delta_LAeq for observation i in park j, round k
    - x_ijk = distance/10 (centered)
    - (b_0j, b_1j) ~ N(0, D) - park random effects
    - u_k ~ N(0, tau^2) - round random effects
    - epsilon ~ N(0, sigma^2)

Prior specification:
    - beta ~ N(0, V0) with V0 = diag(10^2, 5^2)
    - sigma^2 ~ InvGamma(2, 50)
    - tau^2 ~ InvGamma(2, 25)
    - D ~ InvWishart(4, S0) with S0 = diag(10^2, 3^2)

Sampling: 2 chains × 10,000 iterations, burn-in = 2,000, thin = 4

Output:
    - tables/bayes_global_posterior_summary.csv
    - tables/bayes_park_slope_posterior.csv
    - tables/bayes_convergence_diagnostics.csv
    - figures/bayes_global_slope_trace_density.png
    - figures/bayes_park_slopes_forest.png
"""

from __future__ import annotations

import argparse
import math
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
from scipy.stats import invgamma, invwishart

from figure_style import COLORS, PARK_TYPE_COLORS, setup_style
from utils import (
    attach_baseline,
    ensure_dirs,
    load_measurements,
    park_metadata,
    save_figure,
    save_table,
    TABLE_DIR,
)


def summarize_posterior(x: np.ndarray) -> dict[str, float]:
    """Compute posterior summary statistics."""
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "p2_5": float(np.quantile(x, 0.025)),
        "p50": float(np.quantile(x, 0.5)),
        "p97_5": float(np.quantile(x, 0.975)),
    }


def compute_rhat(chains: list[np.ndarray]) -> float:
    """Compute Gelman-Rubin R-hat diagnostic."""
    xs = [np.asarray(c, dtype=float) for c in chains]
    m = len(xs)
    if m < 2:
        return float("nan")
    n = min(len(c) for c in xs)
    if n < 5:
        return float("nan")
    xs = [c[:n] for c in xs]
    means = np.array([c.mean() for c in xs])
    variances = np.array([c.var(ddof=1) for c in xs])
    w = variances.mean()
    if w <= 0:
        return float("nan")
    b = n * means.var(ddof=1)
    var_plus = (n - 1) / n * w + b / n
    return float(math.sqrt(var_plus / w))


def plot_park_slopes_forest(
    park_sum: pd.DataFrame,
    global_slope_mean: float | None = None,
) -> None:
    """Generate the manuscript forest plot for park-specific distance slopes."""
    park_plot = park_sum.sort_values("mean").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(park_plot))

    # Colors by park size category
    colors = [
        PARK_TYPE_COLORS.get(str(pt), COLORS["posterior"]) for pt in park_plot["park_type"]
    ]

    ax.hlines(
        y=y_pos,
        xmin=park_plot["p2_5"],
        xmax=park_plot["p97_5"],
        colors=colors,
        lw=2,
    )
    ax.scatter(park_plot["mean"], y_pos, c=colors, s=50, zorder=3)

    # Highlighted reference lines (no background gridlines)
    ax.axvline(0, color=COLORS["highlight"], linestyle="--", linewidth=1.2, alpha=0.9)
    if global_slope_mean is not None and np.isfinite(global_slope_mean):
        ax.axvline(
            global_slope_mean,
            color=COLORS["secondary"],
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(park_plot["sample_name"])
    ax.set_xlabel("Slope of ΔLAeq per 10m (posterior mean and 95% CrI)")
    ax.set_ylabel("Park")
    ax.grid(False)

    # Add P(slope > 0) annotations on the right
    x_min = float(np.nanmin(park_plot["p2_5"]))
    x_max = float(np.nanmax(park_plot["p97_5"]))
    x_pad = 0.08 * (x_max - x_min) if x_max > x_min else 0.5
    ax.set_xlim(x_min - x_pad, x_max + 3.5 * x_pad)

    for i, row in park_plot.iterrows():
        p_gt0 = float(row.get("prob_slope_gt0", np.nan))
        if not np.isfinite(p_gt0):
            continue
        color = COLORS["highlight"] if p_gt0 >= 0.9 else COLORS["secondary"]
        ax.text(
            x_max + 2.2 * x_pad,
            i,
            f"P(>0)={p_gt0:.0%}",
            va="center",
            ha="left",
            fontsize=8,
            color=color,
        )

    plt.tight_layout()
    save_figure(fig, "bayes_park_slopes_forest.png")
    plt.close(fig)


def gibbs_random_slope(
    y: np.ndarray,
    x: np.ndarray,
    park_idx: np.ndarray,
    round_idx: np.ndarray,
    *,
    n_iter: int = 10_000,
    burn: int = 2_000,
    thin: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Gibbs sampler for Bayesian hierarchical random-slope model.

    Parameters
    ----------
    y : np.ndarray
        Response (delta_LAeq)
    x : np.ndarray
        Predictor (distance/10)
    park_idx : np.ndarray
        Park index (0-indexed)
    round_idx : np.ndarray
        Round index (0-indexed)
    n_iter : int
        Total iterations
    burn : int
        Burn-in iterations
    thin : int
        Thinning interval
    seed : int
        Random seed

    Returns
    -------
    dict[str, np.ndarray]
        Posterior draws for parameters
    """
    rng = np.random.default_rng(seed)

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    park_idx = np.asarray(park_idx, dtype=int)
    round_idx = np.asarray(round_idx, dtype=int)

    n = len(y)
    j = int(park_idx.max() + 1)  # number of parks
    k = int(round_idx.max() + 1)  # number of rounds

    # Center predictor
    x_mean = float(x.mean())
    x_c = x - x_mean
    X = np.column_stack([np.ones(n), x_c])

    # Observation indices by group
    park_obs = [np.flatnonzero(park_idx == jj) for jj in range(j)]
    round_obs = [np.flatnonzero(round_idx == kk) for kk in range(k)]

    # Weakly informative priors (conjugate)
    beta0 = np.zeros(2)
    V0 = np.diag([10.0**2, 5.0**2])
    V0_inv = np.linalg.inv(V0)

    a_sigma, b_sigma = 2.0, 50.0
    a_tau, b_tau = 2.0, 25.0

    nu0 = 4
    S0 = np.diag([10.0**2, 3.0**2])

    # Initialize parameters
    beta = np.array([np.mean(y), 0.0])
    b = np.zeros((j, 2))
    u = np.zeros(k)
    sigma2 = float(np.var(y, ddof=1))
    tau2 = float(np.var(y, ddof=1) / 4)
    D = np.diag([10.0, 1.0])

    # Storage
    n_keep = max(0, (n_iter - burn) // thin)
    draws_beta = np.zeros((n_keep, 2))
    draws_sigma2 = np.zeros(n_keep)
    draws_tau2 = np.zeros(n_keep)
    draws_slope_by_park = np.zeros((n_keep, j))

    keep = 0
    for it in range(n_iter):
        # --- Sample beta | rest ---
        b_effect = b[park_idx, 0] + b[park_idx, 1] * x_c
        y_star = y - b_effect - u[round_idx]
        XtX = X.T @ X
        Vn = np.linalg.inv(XtX / sigma2 + V0_inv)
        mn = Vn @ (X.T @ y_star / sigma2 + V0_inv @ beta0)
        beta = rng.multivariate_normal(mean=mn, cov=Vn)

        # --- Sample b_j | rest ---
        D_inv = np.linalg.inv(D)
        for jj in range(j):
            idx = park_obs[jj]
            if len(idx) == 0:
                continue
            Z = np.column_stack([np.ones(len(idx)), x_c[idx]])
            r = y[idx] - X[idx] @ beta - u[round_idx[idx]]
            Vj = np.linalg.inv(Z.T @ Z / sigma2 + D_inv)
            mj = Vj @ (Z.T @ r / sigma2)
            b[jj] = rng.multivariate_normal(mean=mj, cov=Vj)

        # --- Sample u_k | rest ---
        for kk in range(k):
            idx = round_obs[kk]
            if len(idx) == 0:
                continue
            b_eff = b[park_idx[idx], 0] + b[park_idx[idx], 1] * x_c[idx]
            r = y[idx] - X[idx] @ beta - b_eff
            vk = 1.0 / (len(idx) / sigma2 + 1.0 / tau2)
            mk = vk * (r.sum() / sigma2)
            u[kk] = rng.normal(loc=mk, scale=math.sqrt(vk))

        # --- Sample sigma2 | rest ---
        resid = y - (X @ beta) - (b[park_idx, 0] + b[park_idx, 1] * x_c) - u[round_idx]
        sse = float(resid @ resid)
        sigma2 = float(invgamma.rvs(
            a_sigma + n / 2.0, scale=b_sigma + 0.5 * sse, random_state=rng
        ))

        # --- Sample tau2 | rest ---
        tau2 = float(invgamma.rvs(
            a_tau + k / 2.0, scale=b_tau + 0.5 * float(u @ u), random_state=rng
        ))

        # --- Sample D | rest ---
        S = b.T @ b
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            D = np.asarray(
                invwishart.rvs(df=nu0 + j, scale=S0 + S, random_state=rng),
                dtype=float
            )

        # Store draws
        if it >= burn and (it - burn) % thin == 0:
            draws_beta[keep] = beta
            draws_sigma2[keep] = sigma2
            draws_tau2[keep] = tau2
            draws_slope_by_park[keep] = beta[1] + b[:, 1]
            keep += 1

    return {
        "beta": draws_beta,
        "sigma2": draws_sigma2,
        "tau2_round": draws_tau2,
        "slope_by_park": draws_slope_by_park,
        "x_mean": np.array([x_mean]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian hierarchical random-slope model for noise attenuation."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate manuscript figures from existing output tables "
        "without rerunning the Gibbs sampler.",
    )
    args = parser.parse_args()

    ensure_dirs()
    sns.set_theme(style="white")
    setup_style()

    if args.plot_only:
        park_path = TABLE_DIR / "bayes_park_slope_posterior.csv"
        if not park_path.exists():
            raise SystemExit(f"Missing required table: {park_path}")
        park_sum = pd.read_csv(park_path)

        global_mean = None
        global_path = TABLE_DIR / "bayes_global_posterior_summary.csv"
        if global_path.exists():
            global_sum = pd.read_csv(global_path)
            row = global_sum.loc[global_sum["param"] == "beta1_distance10", "mean"]
            if not row.empty:
                global_mean = float(row.iloc[0])

        plot_park_slopes_forest(park_sum, global_slope_mean=global_mean)
        print("\nPlot-only mode: regenerated bayes_park_slopes_forest.png")
        return

    # Load and prepare data
    meas = load_measurements()
    inside = attach_baseline(meas).copy()
    meta = park_metadata(meas)

    df = inside[["sample_id", "sample_name", "sample_round", "distance10", "delta_LAeq"]].dropna().copy()
    park_cat = pd.Categorical(df["sample_id"].astype(int))
    round_cat = pd.Categorical(df["sample_round"].astype(str))
    df["park_idx"] = park_cat.codes
    df["round_idx"] = round_cat.codes

    y = df["delta_LAeq"].to_numpy()
    x = df["distance10"].to_numpy()
    park_idx = df["park_idx"].to_numpy()
    round_idx = df["round_idx"].to_numpy()

    print(f"Observations: {len(y)}")
    print(f"Parks: {park_cat.categories.size}")
    print(f"Rounds: {round_cat.categories.size}")

    # =========================================================================
    # Run two chains for convergence assessment
    # =========================================================================
    print("\nRunning Gibbs sampler (2 chains × 10,000 iterations)...")

    chain1 = gibbs_random_slope(y, x, park_idx, round_idx, seed=0)
    chain2 = gibbs_random_slope(y, x, park_idx, round_idx, seed=1)

    # Combine chains
    beta1 = np.concatenate([chain1["beta"][:, 1], chain2["beta"][:, 1]])
    beta0 = np.concatenate([chain1["beta"][:, 0], chain2["beta"][:, 0]])
    sigma2 = np.concatenate([chain1["sigma2"], chain2["sigma2"]])
    tau2 = np.concatenate([chain1["tau2_round"], chain2["tau2_round"]])
    slopes = np.concatenate([chain1["slope_by_park"], chain2["slope_by_park"]], axis=0)

    # =========================================================================
    # Convergence diagnostics
    # =========================================================================
    rhat_tbl = pd.DataFrame([
        {"param": "beta0", "rhat": compute_rhat([chain1["beta"][:, 0], chain2["beta"][:, 0]])},
        {"param": "beta1_distance10", "rhat": compute_rhat([chain1["beta"][:, 1], chain2["beta"][:, 1]])},
        {"param": "sigma2", "rhat": compute_rhat([chain1["sigma2"], chain2["sigma2"]])},
        {"param": "tau2_round", "rhat": compute_rhat([chain1["tau2_round"], chain2["tau2_round"]])},
    ])
    save_table(rhat_tbl, "bayes_convergence_diagnostics.csv")

    print("\nConvergence Diagnostics (R-hat):")
    print(rhat_tbl.to_string(index=False))

    # =========================================================================
    # Global parameter summaries
    # =========================================================================
    global_rows = []
    for name, arr in [("beta0", beta0), ("beta1_distance10", beta1),
                      ("sigma2", sigma2), ("tau2_round", tau2)]:
        row = {"param": name, **summarize_posterior(arr)}
        rhat_val = rhat_tbl.loc[rhat_tbl["param"] == name, "rhat"]
        row["rhat"] = float(rhat_val.iloc[0]) if len(rhat_val) > 0 else np.nan
        global_rows.append(row)
    global_sum = pd.DataFrame(global_rows)
    save_table(global_sum, "bayes_global_posterior_summary.csv")

    print("\nGlobal Parameter Posteriors:")
    print(global_sum.to_string(index=False))

    # =========================================================================
    # Park-specific slope posteriors
    # =========================================================================
    park_ids = park_cat.categories.astype(int).to_numpy()
    rows = []
    for j, sid in enumerate(park_ids):
        s = slopes[:, j]
        summ = summarize_posterior(s)
        rows.append({
            "sample_id": int(sid),
            **summ,
            "prob_slope_lt0": float(np.mean(s < 0)),
            "prob_slope_gt0": float(np.mean(s > 0)),
        })
    park_sum = pd.DataFrame(rows)
    park_sum = meta[["sample_id", "sample_name", "park_type"]].merge(
        park_sum, on="sample_id", how="right"
    ).sort_values("sample_id")
    save_table(park_sum, "bayes_park_slope_posterior.csv")

    print("\nPark-Specific Slopes (sorted by P(slope > 0)):")
    top_fail = park_sum.sort_values("prob_slope_gt0", ascending=False).head(5)
    print(top_fail[["sample_name", "mean", "p2_5", "p97_5", "prob_slope_gt0"]].to_string(index=False))

    # =========================================================================
    # Visualization: Trace and density of global slope
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=False)

    axes[0].plot(chain1["beta"][:, 1], lw=0.8, alpha=0.8, label="Chain 1")
    axes[0].plot(chain2["beta"][:, 1], lw=0.8, alpha=0.8, label="Chain 2")
    axes[0].set_xlabel("Iteration (post burn-in, thinned)")
    axes[0].set_ylabel("dB / 10m")
    axes[0].legend()
    axes[0].grid(False)

    sns.kdeplot(beta1, ax=axes[1], fill=True, color="#4C78A8")
    ci = np.quantile(beta1, [0.025, 0.975])
    axes[1].axvline(ci[0], ls="--", c="black", lw=1)
    axes[1].axvline(ci[1], ls="--", c="black", lw=1)
    axes[1].set_xlabel("dB / 10m")
    axes[1].set_ylabel("Density")
    axes[1].grid(False)

    plt.tight_layout()
    save_figure(fig, "bayes_global_slope_trace_density.png")
    plt.close(fig)

    # =========================================================================
    # Visualization: Forest plot of park slopes
    # =========================================================================
    global_row = global_sum.loc[global_sum["param"] == "beta1_distance10", "mean"]
    global_mean = float(global_row.iloc[0]) if not global_row.empty else None
    plot_park_slopes_forest(park_sum, global_slope_mean=global_mean)

    print("\nDone! Output files saved to applsci/outputs/")


if __name__ == "__main__":
    main()
