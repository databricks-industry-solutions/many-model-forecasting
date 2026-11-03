# Pooling validation: does ABC x XYZ pooling help beyond M5?

## Question

`calibrate_intervals` pools the conformal scores over all series by default (`pool="marginal"`), with each series'
tail quantiles shrunk toward the pool's. `pool="group", group_col="auto"` pools per ABC x XYZ class instead: ABC ranks series by
volume (A = top 80% of the total), XYZ by volatility (terciles of the coefficient of variation). On M5 the grouping
looked useful, but M5 is one retail dataset, and ABC x XYZ is a planning convention, not a property of forecast
errors. Before recommending it, or making it a default for some kinds of data, we need evidence on other datasets.

## Hypotheses

1. **H1, marginal coverage.** The default reaches its nominal coverage on held-out data, on every dataset. Pooling per
   group should not change marginal coverage much: both are conformal.
2. **H2, conditional coverage.** On volume data (sales, counts, demand), ABC x XYZ pooling reduces the coverage gap of
   the worst classes (typically C-Z: low volume, erratic), at equal or better interval score.
3. **H3, where it breaks.** On targets that are not volumes (temperatures, rates, prices, mixed macro series), ABC x
   XYZ is arbitrary: it should not help, and can hurt (small, heterogeneous pools).
4. **H4, alternatives.** Groups the user already has (the dataset's natural keys: category, state, purpose), the
   Syntetos-Boylan class (`group_col="sb_class"`), or the series' scale do as well as ABC x XYZ, without its volume
   assumption.

## Configurations

All on the same forecasts (one `run_forecast` per dataset, native intervals on), calibrated with `calibrate_intervals`:

| Name | Call | Notes |
|---|---|---|
| `native` | none (raw bands of `scoring_output`) | Reference for the native models |
| `default` | `pool="marginal"` | One pool per step, series tail quantiles shrunk toward it |
| `abc_xyz` | `pool="group", group_col="auto"` | ABC x XYZ classes computed by MMF |
| `sb_class` | `pool="group", group_col="sb_class"` | Syntetos-Boylan: smooth, erratic, intermittent, lumpy |
| `keys` | `pool="group", group_col=[natural keys]` | Only for datasets with keys (table below) |
| `series_scale` | `settings={"coverage": {"target": "series_scale"}}` | One pool, each series' scale shrunk instead of its tails |

`min_scores_per_group` stays at its default (100), so small groups are merged into one "other" pool; report how many
series end up there.

## Metrics

Computed on the held-out horizon (the last horizon of each series, never seen by the backtest or the calibration),
per dataset, model and level (80, 95):

- **Marginal coverage**, and its gap to nominal.
- **Conditional coverage** per ABC x XYZ class, per series-scale decile, per Syntetos-Boylan class, and per natural key
  when there is one. Summarised as the **worst-group gap** (largest shortfall below nominal over groups with at least
  30 series) and the share of series whose own coverage is below nominal by more than 10 points.
- **MSIS** (scaled by each series' seasonal-naive error), **width**, and **WIS** over both levels: sharpness at equal
  coverage.
- **Leave-one-window-out** coverage from `_interval_metrics`, next to the held-out one: is the backtest estimate honest?
- **Runtime** of each calibration.

Uncertainty: a bootstrap over series (1,000 resamples) for every coverage and score, and 3 random samples of series per
dataset. Report differences between configurations as paired differences on the same series.

## Decision rule

- **Recommend ABC x XYZ pooling for volume data** (in the docs and Skill 4 Step 3f) if, on at least 3 of the volume
  datasets below, it reduces the worst-group gap by 3 points or more without moving marginal coverage by more than
  1 point or worsening WIS by more than 2%, beyond the bootstrap noise.
- **Make it the default for volume data** only if that holds on all of them and it never loses on the others: the
  default must be safe without knowing the data.
- **Keep it opt-in only** otherwise, documented with the datasets where it helped and where it did not.
- If `keys` or `sb_class` match ABC x XYZ (H4), recommend them instead: they need no volume semantics.

## Datasets

Chosen to vary the domain, the frequency, the intermittency and whether the target is a volume. Sample at most 3,000
series per dataset. Horizons and windows follow the frequency (3 backtest windows, and 6 where series are long enough,
to see how the comparison moves with more calibration data).

### Volume targets (where ABC x XYZ makes sense)

| Dataset | Domain | Freq | Series | Horizon | Why | Natural keys |
|---|---|---|---|---|---|---|
| M5 (Walmart) | Retail sales | Daily | 30,490 (sample) | 28 | Baseline; intermittent; already run | store, category, department |
| Car Parts | Spare parts sales | Monthly | 2,674 | 12 | Very intermittent and lumpy: stress test for C-Z classes; only 51 months | none |
| Hospital | Patient counts (medical products) | Monthly | 767 | 12 | Counts outside retail, smooth to erratic | none |
| PBS (Australia) | Prescription counts | Monthly | 336 | 12 | Health-care counts with a hierarchy | concession, type, ATC1, ATC2 |
| `aus_retail` (Australia) | Retail turnover | Monthly | 152 | 12 | Retail but aggregate, smooth, not intermittent | state, industry |
| Tourism (Australia) | Overnight trips | Quarterly | 304 | 8 | Non-retail volumes with a hierarchy | region, state, purpose |
| Electricity (UCI, 321 clients) | Energy consumption | Hourly | 321 | 48 | Smooth, high frequency, very different scales | none |
| Pedestrian counts (Melbourne) | Sensor counts | Hourly | 66 | 48 | Few series, strong daily seasonality; tests small pools | none |

### Targets that are not volumes (negative controls)

| Dataset | Domain | Freq | Series | Horizon | Why |
|---|---|---|---|---|---|
| M4 monthly | Macro, micro, finance, industry, demographic | Monthly | 48,000 (sample) | 18 | Mixed domains; already run |
| Weather (Australian BoM) | Rainfall, min/max temperature, solar radiation | Daily | 3,010 | 28 | Temperatures can be negative: ABC is meaningless; rainfall is zero-inflated |
| Traffic (California occupancy) | Occupancy rates in [0, 1] | Hourly | 862 | 48 | A rate, not a volume |

### Sources and loaders (to verify before running)

- **M4, M5, Electricity, Traffic, Tourism (hierarchical)**: the `datasetsforecast` package (already a dependency of
  the examples) has M4 and M5, and loaders for long-horizon and hierarchical benchmarks that should include
  Electricity, Traffic and the Australian tourism hierarchy.
- **Car Parts, Hospital, Pedestrian counts, Weather**: the Monash Time Series Forecasting Repository
  (forecastingdata.org), as `.tsf` files on Zenodo; the repository provides a Python loader for the format.
- **PBS, `aus_retail`**: the R packages `tsibbledata` / `fpp3` (data from the Australian Bureau of Statistics and the
  Department of Health); export them to CSV once, or read them from an R-datasets mirror.
- **Licenses**: check each source's license before publishing results in the repository (the Monash datasets are on
  Zenodo with their own licenses; the ABS data is CC BY).

## Models

Per dataset, two models with native intervals (`StatsForecastAutoETS`, `StatsForecastBaselineSeasonalNaive`, so CQR)
and one without (`StatsForecastCrostonSBA` for intermittent data, `StatsForecastBaselineWindowAverage` otherwise, so
conformal). Optionally `ChronosBoltSmall` on GPU for one volume and one non-volume dataset, since foundation models'
errors may group differently.

## Implementation

1. **Extend `pi_calibration_coverage.py`**:
   - a dataset registry (loader, frequency, horizon, stride, models, natural keys);
   - the `sb_class` and `keys` configurations;
   - the worst-group gap, per-series shortfall and bootstrap intervals;
   - the seed and window-count widgets.
2. **Add a Monash `.tsf` loader** in the notebook (download from Zenodo, parse to long format with dates), and a
   CSV loader for the `tsibbledata` exports.
3. **One job per dataset**, in parallel, on CPU clusters sized like the current run (4 workers; 8 for M5 and
   Weather). Results to `tests/validation/results/pi_pooling_<dataset>.json`.
4. **A summary notebook** that reads every result file and produces one table per hypothesis, plus the decision.

Rough cost: the statsforecast backtests dominate (about 15 minutes per 2,000 daily series on 4 workers); the
calibrations take a few minutes each. Eleven datasets, 3 seeds, 3 and 6 windows: about a day of cluster time,
run in parallel.

## Risks and caveats

- **Short series** (Car Parts: 51 months) leave few backtest windows after the held-out horizon; use 3 windows only.
- **Small datasets** (Pedestrian counts, `aus_retail`) have classes too small for their own pool, so most series
  fall into the merged "other" pool: report the merge rate, and read their results as tests of the merging.
- **ABC classes depend on the sample**: they rank the sampled series. Use the same samples for every configuration.
- **Multiple comparisons**: 11 datasets x 3 models x 2 levels x 5 configurations. Judge on the paired, bootstrapped
  differences and on consistency across datasets, not on single cells.
- **Negative targets**: check that the features (seasonal-naive scale, ABC, XYZ) behave on Weather's negative
  temperatures; ABC on a negative "volume" is the failure mode H3 expects, and must not crash.

## Inputs from the first run (M5 and M4 monthly)

`pi_calibration_coverage.py` on 2,000 series per dataset, 3 backtest windows, commit 81e5c9b (results in
`results/pi_calibration_coverage_*.json`). Held-out coverage / MSIS; the worst ABC x XYZ class is over classes with at
least 30 series.

**M5 (daily retail, h = 28)**

| Model | Level | native | default | abc_xyz | former |
|---|---|---|---|---|---|
| AutoETS | 80 | 0.868 / 8.11 | 0.794 / 7.13 | 0.800 / 7.21 | 0.800 / 7.21 |
| AutoETS | 95 | 0.942 / 15.49 | 0.949 / 12.89 | 0.949 / 12.77 | 0.949 / 12.77 |
| SeasonalNaive | 80 | 0.913 / 12.64 | 0.792 / 8.70 | 0.804 / 8.68 | 0.804 / 8.68 |
| CrostonSBA (conformal) | 80 | — | 0.802 / 7.50 | 0.804 / 7.38 | 0.794 / 8.38 |
| CrostonSBA (conformal) | 95 | — | 0.941 / 13.32 | 0.945 / 12.27 | 0.953 / 17.37 |

- Marginal coverage is near nominal for every configuration (H1 holds on M5).
- **Conditional coverage favours ABC x XYZ (H2, on M5).** With the default, AX series (high volume, stable) are
  under-covered at 80% (0.736-0.749 across the three models) and the top decile of series scale too (0.75-0.78, against
  0.87-0.88 for the bottom decile). ABC x XYZ pooling raises the worst class to 0.776-0.783 and the top decile to
  0.81-0.83, at the same or lower MSIS.

**M4 monthly (non-retail, h = 18)**

| Model | Level | native | default | abc_xyz | former |
|---|---|---|---|---|---|
| AutoETS | 80 | 0.748 / 7.28 | 0.773 / 7.06 | 0.776 / 7.03 | 0.776 / 7.03 |
| AutoETS | 95 | 0.883 / 13.20 | 0.935 / 11.96 | 0.933 / 11.97 | 0.933 / 11.97 |
| SeasonalNaive | 95 | 0.923 / 14.77 | 0.939 / 14.04 | 0.935 / 14.08 | 0.935 / 14.08 |
| WindowAverage (conformal) | 80 | — | **0.707** / 9.18 | 0.706 / 9.17 | 0.769 / 9.32 |
| WindowAverage (conformal) | 95 | — | **0.850** / 17.91 | 0.851 / 17.98 | 0.933 / 16.63 |

- The CQR models are 1-2 points under nominal on the held-out horizon (their backtest estimate is on target): the last
  horizon of M4 series is harder than the backtest windows.
- **The default conformal target under-covers on M4** (WindowAverage: 0.707 and 0.850, against 0.769 and 0.933 for the
  former tail-quantile shrinkage; the backtest estimate already shows 0.778 and 0.929). A window average on trending
  series has errors that grow over time, and a per-series mean absolute error shrunk mostly toward the series' own
  (weight W x H / (W x H + kappa) = 0.84 here) does not capture the tail. **This must be fixed before the pooling study**:
  candidates are a smaller weight (n_i = W windows instead of W x H scores), a robust tail-aware scale (e.g. a high
  quantile of |e| rather than its mean), or the former target for conformal models.
- ABC x XYZ changes little on M4 (abc_xyz is identical to former for the CQR models, which pool the same way).

**Scale** (`results/pi_calibration_scale_*.json`): `calibrate_intervals` on 1M series x 2 models (h = 7, 3 windows)
took 7.8 minutes on 4 x i3.2xlarge, including the calibrated scoring intervals; 100k took 3 minutes. Row counts are
exact and no interval is NULL. The synthetic CrostonSBA backtest coverage is 0.788 / 0.936 (former: 0.827 / 0.944),
another sign that the default conformal target runs narrow.

**Implications for the plan.** Fix the default conformal target first and re-run these two datasets; then run the
other datasets. Add the series-scale decile and the AX class to the conditional-coverage report, since that is where
the default and ABC x XYZ differ on M5.
