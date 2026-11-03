# Prediction-interval validation (Databricks notebooks)

End-to-end validation of the native prediction intervals on Databricks. These are Databricks notebooks (source
format), run as one-off jobs on a workspace; they install `mmf_sa` from the branch named in their requirements. They
are not part of the unit test suite.

| Notebook | What it checks |
|---|---|
| `pi_native_models.py` | Each model of the `models` widget, one at a time, with its real library and weights on an M5 sample: the native quantiles in the backtest (keys, levels outside the native range left out, finite, non-crossing, median), their raw out-of-sample coverage on the backtest, that the point forecast equals a run without intervals, the scoring interval columns (filled or NULL, non-crossing) with their held-out coverage, that no calibration tables are written, and the warnings. Models without native intervals (e.g. `StatsForecastCrostonSBA`) are checked for NULL columns. |
| `pi_calibration_coverage.py` | `calibrate_intervals` on a dataset whose last horizon is held out (`m5`: daily retail sales; `m4_monthly`: M4 monthly, non-retail), after `run_forecast` with native intervals, in three configurations: the default (one pool, each series' scale shrunk toward it), ABC x XYZ pools, and the former settings. Held-out coverage, MSIS and width per model and level, overall, by series-scale decile and by ABC x XYZ class, next to the raw native bands. |
| `pi_scale_calibration.py` | `calibrate_intervals` on synthetic backtest rows and forecasts for 100k and 1M series (h = 7, 3 windows), with timings and row counts. Skips the local-model backtest, which fits one statsforecast model per series and window. |
| `pi_validation_m5.py` | Equality of the Spark calibration with the numpy reference on MMF's own M5 backtest, and held-out coverage; written for the calibration inside `run_forecast` and kept with its results until it is moved to `calibrate_intervals`. |

