# Prediction-interval validation (Databricks notebooks)

End-to-end validation of the native prediction intervals on Databricks. These are Databricks notebooks (source
format), run as one-off jobs on a workspace; they install `mmf_sa` from the branch named in their requirements. They
are not part of the unit test suite.

| Notebook | What it checks |
|---|---|
| `pi_native_models.py` | Each model of the `models` widget, one at a time, with its real library and weights on an M5 sample: the native quantiles in the backtest (keys, levels outside the native range left out, finite, non-crossing, median), their raw out-of-sample coverage on the backtest, that the point forecast equals a run without intervals, the scoring interval columns (filled or NULL, non-crossing) with their held-out coverage, that no calibration tables are written, and the warnings. Models without native intervals (e.g. `StatsForecastCrostonSBA`) are checked for NULL columns. |
