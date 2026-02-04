# Analysis Code for "Spatial Attenuation of Noise and PM2.5 in Roadside Pocket Parks"

This repository contains the Python code for reproducing the statistical analyses presented in the paper.

## Overview

We analyse noise and PM2.5 attenuation patterns in 12 urban pocket parks adjacent to traffic roads, using 544 synchronised field measurements at distances ranging from 0 to 50 metres from the road boundary (with a small number of intermediate points such as 15/25 m or transects ending before 50 m). The analyses reveal that pocket parks are selective environmental barriers: effective for noise attenuation but marginal for PM2.5 filtration.

## Repository Structure

```
README.md                         # This file
utils.py                          # Common utilities and data loading functions
figure_style.py                   # Consistent figure styling
01_core_attenuation_model.py      # Core distance-attenuation regression
02_linearity_test.py              # Linear vs. nonlinear model comparison
03_mixed_effects_model.py         # Mixed-effects random slopes by park
04_bayesian_hierarchical.py       # Bayesian hierarchical model
05_interaction_effects.py         # Park attribute × distance interactions
06_quantile_regression.py         # Distributional analysis
07_roc_threshold.py               # Optimal planning threshold analysis
08_pm25_attenuation_comparison.py # PM2.5 parallel analysis & comparison
09_robustness_checks.py           # LOPO robustness checks
generate_dual_panel_figures.py    # Generate dual-panel comparison figures
```

## Requirements

- Python 3.9+
- pandas >= 1.4.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- patsy >= 0.5.2

Install dependencies:
```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn patsy
```

## Data Requirements

The analyses read from `../data/measurements.csv`, which contains 544 observations with the following columns:

| Column | Description |
|--------|-------------|
| sample_id | Park identifier (1-12) |
| sample_name | Anonymised park label (Park A through Park L) |
| point_id | Measurement point (P1=roadside baseline, P2-P7=interior) |
| round | Measurement round (1-4) |
| time_period | Monitoring time window |
| distance_m | Distance from road boundary (metres) |
| LAeq | A-weighted equivalent continuous sound level (dB) |
| PM1 | PM1 concentration (μg/m³) |
| PM2.5 | PM2.5 concentration (μg/m³) |
| PM10 | PM10 concentration (μg/m³) |
| PM_resp | Respirable particulate matter (μg/m³) |
| TSP | Total suspended particulates (μg/m³) |
| park_type | Size category (balanced within-sample; 4 parks per category) |
| park_area_m2 | Park area (m²) |
| openness | Spatial configuration (Open, Enclosed) |
| street_sides | Number of sides facing streets |
| temperature_C | Ambient temperature (°C) |
| humidity_pct | Relative humidity (%) |

Data is uploaded separately onto Mendeley Data. See Data availability statement of the paper for the link.

## Output

Results are saved to `../outputs/`:
- `tables/` - CSV files with statistical results
- `figures/` - PNG figures for the manuscript

## Analysis Descriptions

### 01_core_attenuation_model.py
Fits OLS regression with cluster-robust standard errors (clustered by park × round) to estimate the population-average distance-attenuation effect.

**Key output:** Core slope estimate of -1.22 dB per 10m (95% CI: -1.56, -0.79)

### 02_linearity_test.py
Compares linear vs. natural cubic spline models using AIC, and performs bootstrap model selection between linear and piecewise regression models to test for breakpoints.

**Key output:** Linear model is preferred (AIC linear < AIC spline; 92.6% bootstrap selection)

### 03_mixed_effects_model.py
Fits mixed-effects model with random intercepts and slopes by park to quantify between-park heterogeneity in buffering effectiveness.

**Key output:** Park-specific slopes range from -2.29 to +0.48 dB/10m

### 04_bayesian_hierarchical.py
Implements a fully Bayesian random-slope model using Gibbs sampling to provide posterior probability estimates for park-level "buffer failure" (positive slope).

**Key output:** Two parks show >90% probability of ineffective buffering

### 05_interaction_effects.py
Tests whether park attributes (type, area) moderate the distance-attenuation relationship using interaction models.

**Key output:** Small parks show reversed effect (+0.21 dB/10m); area interaction = -0.21 dB/10m per 1000 m²

### 06_quantile_regression.py
Examines distance effects across the conditional distribution using quantile regression with cluster bootstrap confidence intervals.

**Key output:** Median slope = -1.75 dB/10m; upper tail (q=0.9) slope ≈ 0 (unreliable)

### 07_roc_threshold.py
Uses logistic regression and ROC analysis to estimate an empirical distance cutoff for short-term quietness (LAeq ≤ 55 dB(A)) and quantify uncertainty via park-level bootstrap.

**Key output:** Empirical cutoff ≈ 40m (Youden's J; bootstrap mainly 30-40m); AUC = 0.785 (note: compliance at 40-50m remains low)

### 08_pm25_attenuation_comparison.py
Applies the noise analysis framework (core model, model specification, mixed-effects, quantile regression) to PM2.5 data. Tests baseline sensitivity and generates a noise vs. PM2.5 comparison table.

**Key output:** PM2.5 slope = -1.40%/10m (p=0.019), but model selection unstable (36.8% linear) and baseline-sensitive (round-averaged p=0.274). Pocket parks are selective barriers.

### 09_robustness_checks.py
Runs targeted robustness checks requested during internal review.

**Key output:** Leave-one-park-out (LOPO) ranges for the core distance slopes for both noise and PM2.5
