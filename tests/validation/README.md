# Prediction-interval validation (Databricks notebooks)

End-to-end validation of the native and calibrated prediction intervals on Databricks. These are Databricks notebooks (source
format), run as one-off jobs on a workspace; they install `mmf_sa` from the branch named in their requirements. They
are not part of the unit test suite.

| Notebook | What it checks |
|---|---|
| `pi_native_models.py` | Each model of the `models` widget, one at a time, with its real library and weights on an M5 sample: the native quantiles in the backtest (keys, levels outside the native range left out, finite, non-crossing, median), their raw out-of-sample coverage on the backtest, that the point forecast equals a run without intervals, the scoring interval columns (filled or NULL, non-crossing) with their held-out coverage, that no calibration tables are written, and the warnings. Models without native intervals (e.g. `StatsForecastCrostonSBA`) are checked for NULL columns. |
| `pi_calibration_coverage.py` | `calibrate_intervals` on a dataset of its registry whose last horizon is held out (12 datasets, volumes and not: M5, Car Parts, Hospital, Web Traffic, London smart meters, Electricity, Online Retail, TourismLarge, M4 monthly, Traffic, EPF prices, FRED-MD), after `run_forecast` with native intervals, in several poolings: the default (one pool), ABC x XYZ, Syntetos-Boylan classes and the dataset's natural keys. Held-out coverage, MSIS, width and WIS per model and level, worst-group gaps per grouping, merge rates, and bootstrap intervals of the paired differences with the default, next to the raw native bands. Plan and results: `POOLING_VALIDATION.md`. |
| `pi_scale_candidates.py` | On the backtests of `pi_calibration_coverage.py` (M5, M4 monthly), held-out coverage of candidate per-series targets for the conformal intervals (scale statistics and weights, per-series tail quantiles over one pool): the comparison that chose the default `coverage.target` "series". |
| `pi_scale_calibration.py` | `calibrate_intervals` on synthetic backtest rows and forecasts for 100k and 1M series (h = 7, 3 windows), with timings and row counts. Skips the local-model backtest, which fits one statsforecast model per series and window. |

