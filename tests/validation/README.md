# Prediction-interval validation (Databricks notebooks)

End-to-end validation of the prediction intervals on Databricks. These are Databricks
notebooks (source format), run as one-off jobs on a workspace; they install `mmf_sa` from the branch named in
their first cell. They are not part of the unit test suite.

| Notebook | What it checks | Result |
|---|---|---|
| `pi_validation_m5.py` | MMF on the experiment's M5 sample (seed 7, 4,966 series, 3 backtest windows): coverage of the held-out 28 days, and equality of the Spark calibration with the numpy reference on MMF's own backtest. | `results/pi_validation_m5_s7_w3.json` |
| `pi_scale_calibration.py` | Calibration and scoring on synthetic backtest rows for 100k and 1M series (h = 7, 3 windows), with timings and row counts. Skips the local-model backtest, which fits one statsforecast model per series and window. | `results/pi_scale_calibration_100k.json`, `results/pi_scale_calibration_1m.json` |
| `pi_chronos_gpu.py` | MMF with `ChronosBoltSmall` and `Chronos2` on 200 M5 series on a GPU cluster (g5.2xlarge, 1 GPU), through the distributed (Pandas UDF) and the driver-only predict paths: native quantiles in the backtest, CQR calibration, leave-one-window-out and held-out coverage (Chronos-Bolt's native 95% band covers 82%, 95% after CQR), and the creation of a missing experiment folder. | `results/pi_chronos_gpu_200.json` |
