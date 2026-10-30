# Prediction-interval validation (Databricks notebooks)

Task 7 of the prediction-interval plan (`docs/specs/confidence_interval.md`, section 11.5). These are Databricks
notebooks (source format), run as one-off jobs on a workspace; they install `mmf_sa` from the branch named in
their first cell. They are not part of the unit test suite.

| Notebook | What it checks | Result |
|---|---|---|
| `pi_validation_m5.py` | MMF on the experiment's M5 sample (seed 7, 4,966 series, 3 backtest windows): coverage of the held-out 28 days, and equality of the Spark calibration with the numpy reference on MMF's own backtest. | `results/pi_validation_m5_s7_w3.json` |
| `pi_scale_calibration.py` | Calibration and scoring on synthetic backtest rows for 100k and 1M series (h = 7, 3 windows), with timings and row counts. Skips the local-model backtest, which fits one statsforecast model per series and window. | `results/pi_scale_calibration_100k.json`, `results/pi_scale_calibration_1m.json` |
