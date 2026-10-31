# Prediction-interval validation (Databricks notebooks)

End-to-end validation of the prediction intervals on Databricks. These are Databricks
notebooks (source format), run as one-off jobs on a workspace; they install `mmf_sa` from the branch named in
their first cell. They are not part of the unit test suite. `pi_moirai_local.py` is a local script instead (see its
docstring).

| Notebook | What it checks | Result |
|---|---|---|
| `pi_validation_m5.py` | MMF on the experiment's M5 sample (seed 7, 4,966 series, 3 backtest windows): coverage of the held-out 28 days, and equality of the Spark calibration with the numpy reference on MMF's own backtest. | `results/pi_validation_m5_s7_w3.json` |
| `pi_scale_calibration.py` | Calibration and scoring on synthetic backtest rows for 100k and 1M series (h = 7, 3 windows), with timings and row counts. Skips the local-model backtest, which fits one statsforecast model per series and window. | `results/pi_scale_calibration_100k.json`, `results/pi_scale_calibration_1m.json` |
| `pi_chronos_gpu.py` | MMF with `ChronosBoltSmall` and `Chronos2` on 200 M5 series on a GPU cluster (g5.2xlarge, 1 GPU), through the distributed (Pandas UDF) and the driver-only predict paths: native quantiles in the backtest, CQR calibration, leave-one-window-out and held-out coverage (Chronos-Bolt's native 95% band covers 82%, 95% after CQR), and the creation of a missing experiment folder. | `results/pi_chronos_gpu_200.json` |
| `pi_native_models.py` | Each model with native quantiles, one at a time, with its real library and weights on 200 M5 series: SKTimeProphet (CPU), TimesFM_2_5_200m (distributed and driver-only paths) and the 19 `FF*` models (g5.2xlarge, DBR 18 ML). Checks the backtest quantiles (keys, finite, non-crossing, median, 95/80 width ratio, which is 1 when the model clamps to 10-90%), that the point forecast equals a run without intervals, CQR calibration, held-out coverage and the warnings. Run on commit c26458d, before the FFChronos2 and FFFlowState range fixes: the results show the false clamp warning of the Chronos-2 models and FlowState's missing one. | `results/pi_native_models_200.json`; after the refactor into `mmf_sa/intervals` and the nesting fix (commit 83dccca: AutoETS, CrostonSBA, SKTimeProphet, MLForecastLGBM, ChronosBoltSmall, Chronos2, TimesFM, FFChronos2, FFFlowState, FFToto, with no crossing quantiles): `results/pi_native_models_200_post_refactor.json` |
| `pi_moirai_local.py` | MoiraiSmall and MoiraiMoESmall (uni2ts 2.0.0, CPU) on the same sample, with 10 and 100 sample paths: the sample shapes MMF's predict expects, the quantile packing round trip, CQR calibration and held-out coverage. | `results/pi_moirai_local.json` |
