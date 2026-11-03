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

| Registry name | Dataset | Domain | Freq | Series | Horizon | Why | Natural keys |
|---|---|---|---|---|---|---|---|
| `m5` | M5 (Walmart) | Retail sales, US | Daily | 2,000 sampled | 28 | Baseline; intermittent | store, category |
| `car_parts` | Car Parts | Spare parts sales, US | Monthly | 2,674 | 6 | 76% zeros: stress test for C-Z classes; only 51 months | none |
| `hospital` | Hospital | Patient counts (medical products), US | Monthly | 767 | 12 | Counts outside retail | none |
| `web_traffic` | Kaggle Web Traffic | Wikipedia page views | Daily | 2,000 of 145k | 28 | Heavy tails and spikes, huge scale range | none |
| `london_smart_meters` | London Smart Meters | Household electricity, UK | Daily (summed from half-hourly) | 2,000 of 5,560 | 28 | Many small, noisy series | none |
| `electricity` | Electricity (UCI) | Electricity of 321 clients, Portugal | Hourly | 321 | 24 | Smooth, high frequency, very different scales | none |
| `online_retail` | UCI Online Retail II | E-commerce product sales, UK | Daily | products with 60+ sale days | 14 | Intermittent online demand | none |
| `tourism` | TourismLarge | Overnight trips, Australia | Monthly | 304 (region x purpose) | 12 | Volumes with a hierarchy | state, purpose |

### Targets that are not volumes (negative controls)

| Registry name | Dataset | Domain | Freq | Series | Horizon | Why | Natural keys |
|---|---|---|---|---|---|---|---|
| `m4_monthly` | M4 monthly | Macro, micro, finance, industry, demographic | Monthly | 2,000 sampled | 18 | Mixed domains | none |
| `traffic` | Traffic (PeMS) | Road occupancy rates in [0, 1], California | Hourly | 862 | 24 | A rate, not a volume | none |
| `epf_prices` | EPF day-ahead prices (NP, PJM, BE, FR, DE) | Electricity prices; DE goes negative | Daily, one series per market x hour | 120 | 7 | Prices, negative values; BE and FR are coupled | market, hour |
| `fred_md` | FRED-MD | US macro levels | Monthly | ~125 | 12 | Indices and rates; reachability from Databricks unverified | none |

### Sources and loaders

All are in the registry of `pi_calibration_coverage.py`:

- **Monash repository** (Zenodo, CC BY 4.0): Car Parts (record 4656021, without missing values), Hospital (4656014), Kaggle Web Traffic (4656075, zeros for missing), London Smart Meters (4656091), Electricity (4656140), Traffic (4656132). `.tsf` files, parsed by `read_tsf`. Monthly series are moved to month ends; hourly ones start at 00:00:01 and are floored to the hour.
- **UCI Online Retail II** (CC BY 4.0): transactions aggregated to daily quantity per product (returns and non-product codes dropped), zeros on days without sales.
- **EPF** (Lago et al. 2021, Zenodo 4624805): hourly prices, one daily series per market and hour.
- **TourismLarge** (`datasetsforecast.hierarchical`): the 304 bottom series; state and purpose come from the positional ids.
- **FRED-MD** (St. Louis Fed `current.csv`): raw levels; the URL could not be reached from a laptop.
- Dropped from the first plan: PBS, `aus_retail`, Pedestrian counts and BoM weather (too many Australian sets), and NYC mobility (no public count dataset on Monash; its rideshare set is Boston prices).

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
other datasets.

**Done:** `pi_scale_candidates.py` compared scale statistics and weights on the same backtests, and the per-series
tail shrinkage over one pool (`coverage.target: series`) won; it is now the default. The rerun with it
(`results/pi_calibration_coverage_*_series_default.json`) gives, held-out at 80% / 95%: M4 monthly WindowAverage
0.763 / 0.935 (was 0.707 / 0.850), CrostonSBA on M5 0.797 / 0.952, and the CQR models unchanged. ABC x XYZ is now the
same method pooled per class, so the pooling study compares `default` and `abc_xyz` directly. Add the series-scale decile and the AX class to the conditional-coverage report, since that is where
the default and ABC x XYZ differ on M5.

## First pass (seed 7, 3 windows)

Job run 783821921772758 on commit 9074796; results in `results/pi_pooling_<dataset>_s7_w3.json`. 11 of 12 datasets ran;
FRED-MD failed with HTTP 403 (the download now sends a User-Agent; retry in the next pass). Paired differences are
against `default`, with 95% bootstrap intervals over series.

**H1, marginal coverage of the default.** Within 2-3 points of nominal on 9 datasets. Exceptions: Electricity (0.70-0.73
at 80%, 0.85-0.87 at 95%, native bands included: the held-out day is harder than the backtest days) and Traffic
SeasonalNaive (0.736 / 0.882). On the zero-heavy datasets (Car Parts, London, Online Retail) 80% intervals over-cover
(0.84-0.87): ties at zero.

**H2, ABC x XYZ on volume data.** Cells meeting the decision rule (worst ABC x XYZ gap down by 3+ points with the
interval excluding 0, marginal coverage within 1 point, WIS not worse by more than 2%):

| Dataset | Cells | Where |
|---|---|---|
| M5 | 2 | SeasonalNaive 80, CrostonSBA 95 |
| Web Traffic | 2 | SeasonalNaive 95, WindowAverage 80 (but SeasonalNaive 80 loses 1.9 points of coverage) |
| Car Parts | 1 | CrostonSBA 95 (CrostonSBA 80: gap -13.6 points, WIS -6%, but coverage +2.6) |
| Online Retail | 1 | AutoETS 80 |
| Hospital, London, Tourism, Electricity | 0 | Neutral; Electricity slightly worse (WIS +1%, coverage -0.3 to -1.1) |

The gains concentrate on the conformal models (CrostonSBA, WindowAverage) and on intermittent data; for the CQR models
the differences are mostly within 1-2 points. ABC x XYZ also lowers WIS on Car Parts, M5 CrostonSBA, Online Retail and
Web Traffic (1-6%).

**H3, not volumes.** Confirmed. On Traffic, ABC x XYZ loses 3-10 points of marginal coverage (AutoETS 80: 0.851 to
0.788; SeasonalNaive 80: 0.736 to 0.641) and raises WIS by 1.5-5.6%. On EPF prices it loses 1-4 points of coverage
and SeasonalNaive's WIS rises 6.9%. M4 monthly is neutral (one winning cell).

**H4, alternatives.** `sb_class` is nearly identical to the default almost everywhere (most series are "smooth", so
one class holds most of them); it wins once on Online Retail CrostonSBA and once on Traffic AutoETS. `keys` is neutral
on M5 (store x category). On Tourism (state x purpose) and EPF (market x hour) every group was below
`min_scores_per_group` (34 series at 3 windows), so everything merged into one pool and `keys` equals the default: the
next pass uses `purpose` and `market`.

**Caveats.**
- The worst-group gap of the decision rule is measured on the ABC x XYZ classes, the grouping `abc_xyz` pools on, which
  favours it. The series-scale decile gap is the fairer check: it agrees on M5 CrostonSBA (-5.2 points), Car Parts
  CrostonSBA (-13.4) and M4 WindowAverage (-4.5), and is near zero for most other winning cells.
- One seed, 3 windows: 11 datasets x 3 models x 2 levels are many comparisons; a few winning cells are expected by chance.
- Web Traffic AutoETS has a few exploding forecasts, so its mean WIS (about 3e13) is meaningless; the paired
  differences are not affected.

**Reading against the decision rule (provisional).** Four volume datasets have at least one winning cell, so the
"recommend for volume data" branch is met on a lenient count, but only M5 and Web Traffic win on more than one cell, and
the wins are mostly for conformal models on intermittent data. It loses clearly on Traffic and EPF, so it cannot be a
default. Provisional conclusion: keep `pool="marginal"` as the default; document ABC x XYZ as an opt-in for intermittent
volume data (mainly for models without native intervals), never for rates or prices. To confirm with seeds 11 and 23,
6 windows where series allow, and with the series-scale decile gap as the primary conditional metric.

## Second pass (seeds 11 and 23 at 3 windows, seed 7 at 6 windows) and decision

Run 973983404032747 on commit 6e56c66 (with the calibration fixes) and the notebook of 521696e, which adds the paired
bootstrap of the series-scale decile gap; results in `results/pi_pooling_<dataset>_s<seed>_w<windows>.json`. 32 of 35
tasks ran; FRED-MD still returns HTTP 403 from Databricks and is dropped. Car Parts (51 months) has no 6-window run.
Hospital, Electricity, Tourism, Traffic and EPF have fewer series than the sample size, so their seeds use the same
series and give the same numbers: the seed replication concerns M5, Car Parts, Web Traffic, London, Online Retail and
M4 monthly.

**ABC x XYZ against the default, per dataset** (paired differences, mean over the 3- and 6-window runs; "gap" is the
worst group's shortfall below nominal, on ABC x XYZ classes and on series-scale deciles):

| Dataset | Volume | Verdict | Evidence |
|---|---|---|---|
| M5 | yes | **helps** (every run) | CrostonSBA 95%: decile gap -4.6 to -8.2 points, class gap -3.8 to -8.7, WIS -1.0 to -2.0%; CQR models: class gaps -1 to -3, coverage +0.2 to +0.8 |
| Car Parts | yes | **helps** (every run) | CrostonSBA 80%: decile gap -25 to -27 points, WIS -4.9 to -6.3% (coverage +2.2 to +2.8, from 0.84 toward over-coverage); AutoETS, SeasonalNaive WIS -1 to -2% |
| Online Retail | yes | **helps** (every run) | CrostonSBA: decile gap -10.5 to -13.2 points, WIS -2.0 to -6.7%; SeasonalNaive 80%: decile gap -3.1 to -3.9 |
| Web Traffic | yes | mixed | WIS -0.3 to -5.1%, AutoETS 80% decile gap -9.5 to -13.6, but SeasonalNaive 80% loses 1.9 to 2.7 points of coverage |
| Hospital, London, Tourism | yes | neutral | All differences within 1 point and 1% |
| Electricity | yes | slightly worse | SeasonalNaive WIS +1.3 to +4.1%, decile gap +1.6 to +3.3 |
| M4 monthly | no | neutral | WindowAverage 80% class gap -3, otherwise within 1 point |
| Traffic | no | **hurts** | Coverage -3 to -10 points at 3 windows (SeasonalNaive 80%: -9.5), -0.5 to -5 at 6 windows; WIS +0.4 to +5.6% |
| EPF prices | no | **hurts** | Coverage -1 to -4 points; SeasonalNaive WIS +1.2 to +6.9% |

- **H1** holds: the default is within 2-3 points of nominal everywhere except Electricity and Traffic SeasonalNaive,
  where the native bands miss too (the held-out horizon is harder than the backtest windows).
- **H2** holds for intermittent demand, not for volumes in general: the gains are on M5, Car Parts and Online Retail,
  in every run and under both gap definitions, and mostly for the models without native intervals (CrostonSBA).
  Smooth volumes (Hospital, London, Tourism, Electricity) gain nothing.
- **H3** holds: on rates and prices ABC x XYZ loses coverage, less with 6 windows (larger pools per class).
- **H4**: `sb_class` is close to the default (most series fall in one class); it helps on Online Retail (WIS -1.8 to
  -6.0%) and slightly on Traffic. `keys` is neutral on M5 (store x category) and Tourism (purpose); EPF by market only
  has pools large enough at 6 windows, and loses 4 points of coverage there. Neither replaces ABC x XYZ.

**Decision.** The rule's first branch is met: on 3 volume datasets (M5, Car Parts, Online Retail) ABC x XYZ lowers the
worst-group gap by 3 points or more beyond the bootstrap noise, with marginal coverage within 1 point (Car Parts
CrostonSBA aside, where it rises toward over-coverage) and WIS lower. It is not met on all volume datasets and loses on
Traffic and EPF, so it does not become a default. The docs and Skill 4 Step 3f now recommend `pool="group",
group_col="auto"` for intermittent demand data, and never for targets that are not volumes; one pool stays the default.
