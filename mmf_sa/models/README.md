# Supported Models

Model hyperparameters can be modified under [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml).

## Local


| model                                      | source                                                                                                                          | covariate support | recommended compute                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---------------------------------------------- |
| StatsForecastBaselineWindowAverage         | [Statsforecast Window Average](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#windowaverage)                  |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastBaselineSeasonalWindowAverage | [Statsforecast Seasonal Window Average](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#seasonalwindowaverage) |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastBaselineNaive                 | [Statsforecast Naive](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#naive)                                   |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastBaselineSeasonalNaive         | [Statsforecast Seasonal Naive](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#seasonalnaive)                  |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastAutoArima                     | [Statsforecast AutoARIMA](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoarima)                           | ✅                 | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastAutoETS                       | [Statsforecast AutoETS](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoets)                               |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastAutoCES                       | [Statsforecast AutoCES](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoces)                               |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastAutoTheta                     | [Statsforecast AutoTheta](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotheta)                           |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastAutoTbats                     | [Statsforecast AutoTBATS](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotbats)                           |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastAutoMfles                     | [Statsforecast AutoMFLES](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#automfles)                           | ✅                 | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastTSB                           | [Statsforecast TSB](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#tsb)                                       |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastADIDA                         | [Statsforecast ADIDA](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#adida)                                   |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastIMAPA                         | [Statsforecast IMAPA](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#imapa)                                   |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastCrostonClassic                | [Statsforecast Croston Classic](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#crostonclassic)                |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastCrostonOptimized              | [Statsforecast Croston Optimized](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#crostonoptimized)            |                   | DBR 18 ML; single-node or multi-node CPU |
| StatsForecastCrostonSBA                    | [Statsforecast Croston SBA](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#crostonsba)                        |                   | DBR 18 ML; single-node or multi-node CPU |
| SKTimeProphet                              | [sktime Prophet](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.fbprophet.Prophet.html)       |                   | DBR 18 ML; single-node or multi-node CPU |


## Global


| model                      | source                                                                                                                                | covariate support | recommended compute                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------ |
| MLForecastLGBM             | [MLForecast + LightGBM](https://nixtlaverse.nixtla.io/mlforecast/index.html)                                                          | ✅                 | DBR 18 ML; single-node CPU Spark cluster                                 |
| MLForecastAutoLGBM         | [MLForecast AutoMLForecast + AutoModel](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/hyperparameter_optimization.html) | ✅                 | DBR 18 ML; single-node CPU Spark cluster                                 |
| NeuralForecastRNN          | [NeuralForecast RNN](https://nixtlaverse.nixtla.io/neuralforecast/models.rnn.html)                                                    | ✅                 | DBR 18 ML; A10G GPU; single-node multi-GPU recommended, multi-node supported |
| NeuralForecastLSTM         | [NeuralForecast LSTM](https://nixtlaverse.nixtla.io/neuralforecast/models.lstm.html)                                                  | ✅                 | DBR 18 ML; A10G GPU; single-node multi-GPU recommended, multi-node supported |
| NeuralForecastNBEATSx      | [NeuralForecast NBEATSx](https://nixtlaverse.nixtla.io/neuralforecast/models.nbeatsx.html)                                            | ✅                 | DBR 18 ML; A10G GPU; single-node multi-GPU recommended, multi-node supported |
| NeuralForecastNHITS        | [NeuralForecast NHITS](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html)                                                | ✅                 | DBR 18 ML; A10G GPU; single-node multi-GPU recommended, multi-node supported |
| NeuralForecastAutoRNN      | [NeuralForecast AutoRNN](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autornn)                                            | ✅                 | DBR 18 ML; A10G GPU; single-node multi-GPU recommended, multi-node supported |
| NeuralForecastAutoLSTM     | [NeuralForecast AutoLSTM](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autolstm)                                          | ✅                 | DBR 18 ML; A10G GPU; single-node multi-GPU recommended, multi-node supported |
| NeuralForecastAutoNBEATSx  | [NeuralForecast AutoNBEATSx](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autonbeatsx)                                    | ✅                 | DBR 18 ML; A10G GPU; single-node multi-GPU recommended, multi-node supported |
| NeuralForecastAutoNHITS    | [NeuralForecast AutoNHITS](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autonhits)                                        | ✅                 | DBR 18 ML; A10G GPU; single-node multi-GPU recommended, multi-node supported |
| NeuralForecastAutoTiDE     | [NeuralForecast AutoTiDE](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autotide)                                          | ✅                 | DBR 18 ML; A10G GPU; single-node multi-GPU recommended, multi-node supported |
| NeuralForecastAutoPatchTST | [NeuralForecast AutoPatchTST](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autopatchtst)                                  |                   | DBR 18 ML; A10G GPU; single-node multi-GPU recommended, multi-node supported |


## Foundation


| model              | source                                                                                            | covariate support | recommended compute                                 |
| ------------------ | ------------------------------------------------------------------------------------------------- | ----------------- | --------------------------------------------------- |
| ChronosBoltTiny    | [amazon/chronos-bolt-tiny](https://huggingface.co/amazon/chronos-bolt-tiny)                       |                   | DBR 18 ML; single-node A10G GPU                   |
| ChronosBoltMini    | [amazon/chronos-bolt-mini](https://huggingface.co/amazon/chronos-bolt-mini)                       |                   | DBR 18 ML; single-node A10G GPU                   |
| ChronosBoltSmall   | [amazon/chronos-bolt-small](https://huggingface.co/amazon/chronos-bolt-small)                     |                   | DBR 18 ML; single-node A10G GPU                   |
| ChronosBoltBase    | [amazon/chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base)                       |                   | DBR 18 ML; single-node A10G GPU                   |
| Chronos2           | [amazon/chronos-2](https://huggingface.co/amazon/chronos-2)                                       | ✅                 | DBR 18 ML; single-node A10G GPU or serverless GPU |
| Chronos2Small      | [autogluon/chronos-2-small](https://huggingface.co/autogluon/chronos-2-small)                     | ✅                 | DBR 18 ML; single-node A10G GPU or serverless GPU |
| Chronos2Synth      | [autogluon/chronos-2-synth](https://huggingface.co/autogluon/chronos-2-synth)                     | ✅                 | DBR 18 ML; single-node A10G GPU or serverless GPU |
| TimesFM_2_5_200m   | [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch)         | ✅                 | DBR 18 ML; single-node A10G GPU or serverless GPU |
| ~~MoiraiSmall~~    | ~~[Salesforce/moirai-1.1-R-small](https://huggingface.co/Salesforce/moirai-1.1-R-small)~~         |                   | Temporarily disabled                                |
| ~~MoiraiBase~~     | ~~[Salesforce/moirai-1.1-R-base](https://huggingface.co/Salesforce/moirai-1.1-R-base)~~           |                   | Temporarily disabled                                |
| ~~MoiraiLarge~~    | ~~[Salesforce/moirai-1.1-R-large](https://huggingface.co/Salesforce/moirai-1.1-R-large)~~         |                   | Temporarily disabled                                |
| ~~MoiraiMoESmall~~ | ~~[Salesforce/moirai-moe-1.0-R-small](https://huggingface.co/Salesforce/moirai-moe-1.0-R-small)~~ |                   | Temporarily disabled                                |
| ~~MoiraiMoEBase~~  | ~~[Salesforce/moirai-moe-1.0-R-base](https://huggingface.co/Salesforce/moirai-moe-1.0-R-base)~~   |                   | Temporarily disabled                                |
| ~~MoiraiMoELarge~~ | ~~[Salesforce/moirai-moe-1.0-R-large](https://huggingface.co/Salesforce/moirai-moe-1.0-R-large)~~ |                   | Temporarily disabled                                |

### FoundationForecast

These models run through [foundationforecast](https://pypi.org/project/foundationforecast/) 0.1.10 (TimeCopilot) and use `framework: FoundationForecast`. Install them with [requirements-foundationforecast.txt](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/requirements-foundationforecast.txt) or the `mmf_sa[foundationforecast]` extra, not with `requirements-foundation.txt`. Installing both sets in one environment works, but it isn't a supported setup.

| model              | source                                                                                                                      | license                             | covariate support | recommended compute                                 |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ----------------- | --------------------------------------------------- |
| FFChronos2         | [amazon/chronos-2](https://huggingface.co/amazon/chronos-2)                                                                 | apache-2.0                          |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFChronos2Small    | [autogluon/chronos-2-small](https://huggingface.co/autogluon/chronos-2-small)                                               | apache-2.0                          |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFTimesFM_2_5_200m | [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch)                                   | apache-2.0                          |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFTimesFM_3_0      | [google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch)                                             | timesfm-non-commercial-license-v1.0 |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFTiRex2           | [NX-AI/TiRex-2](https://huggingface.co/NX-AI/TiRex-2)                                                                       | apache-2.0                          |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFToto             | [Datadog/Toto-Open-Base-1.0](https://huggingface.co/Datadog/Toto-Open-Base-1.0)                                             | apache-2.0                          |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFToto2_22m        | [Datadog/Toto-2.0-22m](https://huggingface.co/Datadog/Toto-2.0-22m)                                                         | apache-2.0                          |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFToto2_313m       | [Datadog/Toto-2.0-313m](https://huggingface.co/Datadog/Toto-2.0-313m)                                                       | apache-2.0                          |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFToto2_1B         | [Datadog/Toto-2.0-1B](https://huggingface.co/Datadog/Toto-2.0-1B)                                                           | apache-2.0                          |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFFlowState        | [ibm-research/flowstate](https://huggingface.co/ibm-research/flowstate)                                                     | apache-2.0                          |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFPatchTSTFM_R2    | [ibm-granite/granite-timeseries-patchtst-fm-r2](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2)       | openmdw-1.0                         |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFPatchTSTFM_R1    | [ibm-research/patchtst-fm-r1](https://huggingface.co/ibm-research/patchtst-fm-r1)                                           | cc-by-nc-sa-4.0                     |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFTafsutBase       | [Tafsut-FM/tafsut-univariate-base](https://huggingface.co/Tafsut-FM/tafsut-univariate-base)                                 | mit                                 |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |
| FFT0Beta           | [theforecastingcompany/t0-beta](https://huggingface.co/theforecastingcompany/t0-beta)                                       | apache-2.0                          |                   | DBR 18 ML; single-node A10G GPU or serverless GPU |

Things to know before using them:

- **Licenses.** `FFTimesFM_3_0` and `FFPatchTSTFM_R1` are released under non-commercial licenses. MMF logs a warning the first time each one is used. Check the model card before using any checkpoint commercially.
- **Univariate only.** Covariate columns passed to `run_forecast` are ignored, with a one-time warning per model.
- **Driver GPU.** Inference always runs on the driver's GPU, on classic clusters and on serverless GPU alike, so a single GPU processes all series. `serverless=True` is still required on serverless GPU for the rest of the MMF pipeline.
- **Monthly data.** With `freq="M"`, every timestamp must be a month end; otherwise data preparation fails with an error instead of shifting the forecast dates.
- **Hugging Face access.** None of these checkpoints is gated. If you hit Hugging Face rate limits, set `HF_TOKEN` from a Databricks secret before calling `run_forecast` (see the [example notebook](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundationforecast/foundationforecast_daily.ipynb)). Never put the token in a config file.
- **Memory.** MMF frees the previous checkpoint's weights when it switches to another model, so normally only one model's weights occupy the GPU. On classic clusters, up to about 4.4 GB from `FFToto2_1B` can stay allocated for a few more models before it is released. `FFToto2_1B` is the largest model: about 4.4 GB of GPU memory and 11 GB of host memory at peak.


## Configurable Hyperparameters

All model defaults live in `mmf_sa/models/models_conf.yaml`. Run-level values from `forecasting_conf_*.yaml` are promoted into each active model when present.

### Promoted Run Settings

- `prediction_length`: Forecast horizon, in periods of `freq`.
- `group_id`: Column that identifies each time series.
- `date_col`: Timestamp column used as the time index.
- `target`: Numeric column being forecast.
- `metric`: Evaluation metric, such as `smape`, `mape`, `mae`, `mse`, or `rmse`.
- `freq`: Time frequency, such as hourly, daily, weekly, or monthly.
- `temp_path`: Temporary storage path used by distributed or partitioned model code.
- `accelerator`: Compute type requested by supported models, typically `cpu` or `gpu`.
- `num_nodes`: Number of cluster nodes used by supported distributed paths.
- `backtest_length`: Number of historical periods reserved for backtesting.
- `stride`: Step size between rolling backtest windows.
- `static_features`: Columns that do not vary over time within a series.
- `dynamic_future_numerical`: Numeric covariates known for future timestamps.
- `dynamic_future_categorical`: Categorical covariates known for future timestamps.
- `dynamic_historical_numerical`: Numeric covariates available only in historical data.
- `dynamic_historical_categorical`: Categorical covariates available only in historical data.

### Local Model Settings

StatsForecast local models use the nested `model_spec` block.

- `window_size`: Number of recent observations used by window-average baseline models.
- `season_length`: Number of periods in one seasonal cycle, such as `7` for weekly seasonality in daily data or `12` for yearly seasonality in monthly data.
- `approximation`: Enables faster approximate search in `StatsForecastAutoArima`.
- `model`: Model structure code used by automatic ETS or CES variants.
- `decomposition_type`: Seasonal decomposition mode, typically `additive` or `multiplicative`.
- `use_boxcox`: Enables Box-Cox transformation in TBATS.
- `bc_lower_bound`: Lower bound for the Box-Cox lambda search.
- `bc_upper_bound`: Upper bound for the Box-Cox lambda search.
- `use_trend`: Enables a trend component in TBATS.
- `use_damped_trend`: Enables trend damping in TBATS.
- `use_arma_errors`: Enables ARMA error modeling in TBATS.
- `alpha_d`: Smoothing parameter for intermittent-demand occurrence in TSB.
- `alpha_p`: Smoothing parameter for intermittent-demand size in TSB.
- `enable_gcv`: Enables grid or cross-validation behavior for the SKTime Prophet wrapper.
- `growth`: Prophet trend type, such as `linear`.
- `yearly_seasonality`: Prophet yearly seasonality mode or value.
- `weekly_seasonality`: Prophet weekly seasonality mode or value.
- `daily_seasonality`: Prophet daily seasonality mode or value.
- `seasonality_mode`: Prophet seasonality interaction mode, usually `additive` or `multiplicative`.

`StatsForecastBaselineNaive`, `StatsForecastADIDA`, `StatsForecastIMAPA`, `StatsForecastCrostonClassic`, `StatsForecastCrostonOptimized`, and `StatsForecastCrostonSBA` define no model-specific hyperparameters in the default config.

### NeuralForecast Settings

- `max_steps`: Maximum training steps for the neural model.
- `num_samples`: Number of hyperparameter samples for Auto NeuralForecast models.
- `input_size_factor`: Multiplier used to derive historical input window size from the forecast horizon.
- `input_size`: Explicit historical input window length.
- `loss`: Training loss or optimization metric used by the model.
- `learning_rate`: Step size used by the optimizer during training.
- `batch_size`: Number of training examples per optimization batch.
- `dropout_prob_theta`: Dropout probability used in supported NBEATS-style components.
- `encoder_n_layers`: Number of recurrent encoder layers.
- `encoder_hidden_size`: Hidden-state width of recurrent encoder layers.
- `encoder_activation`: Activation function used by the encoder.
- `decoder_hidden_size`: Hidden-layer width of decoder layers.
- `decoder_layers`: Number of decoder layers.
- `n_harmonics`: Number of harmonic terms used by NBEATSx seasonality components.
- `n_polynomials`: Number of polynomial terms used by NBEATSx trend components.
- `stack_types`: NHITS stack type sequence.
- `n_blocks`: Number of blocks per NHITS stack.
- `n_pool_kernel_size`: Pooling kernel sizes used by NHITS stacks.
- `n_freq_downsample`: Frequency downsampling factors used by NHITS stacks.
- `interpolation_mode`: Interpolation method used by NHITS.
- `pooling_mode`: Pooling method used by NHITS.
- `scaler_type`: Input scaling strategy, such as `robust` or `standard`.
- `hidden_size`: Hidden-layer width for TiDE or PatchTST search spaces.
- `decoder_output_dim`: TiDE decoder output dimension.
- `temporal_decoder_dim`: TiDE temporal decoder width.
- `num_encoder_layers`: Number of TiDE encoder layers.
- `num_decoder_layers`: Number of TiDE decoder layers.
- `temporal_width`: Width of TiDE temporal features.
- `dropout`: Dropout probability for supported Auto models.
- `layernorm`: Whether to use layer normalization in supported Auto models.
- `n_heads`: Number of attention heads in PatchTST.
- `patch_len`: Length of each temporal patch in PatchTST.
- `revin`: Whether to use reversible instance normalization in PatchTST.

Auto NeuralForecast values written as YAML lists are candidate values for HPO.

### MLForecast Settings

- `num_threads`: Number of CPU threads used by MLForecast and LightGBM.
- `num_samples`: Number of Optuna trials used by `MLForecastAutoLGBM`.
- `num_windows`: Number of rolling-origin cross-validation windows used during `MLForecastAutoLGBM` tuning.
- `season_length`: Seasonal period used by AutoMLForecast when no explicit `feature_space` is provided.
- `model_params`: Fixed keyword arguments passed to `lightgbm.LGBMRegressor`.
- `learning_rate`: LightGBM shrinkage rate; lower values learn more slowly and often need more trees.
- `num_leaves`: Maximum number of leaves per LightGBM tree.
- `n_estimators`: Number of boosting trees.
- `feature_fraction`: Fraction of features sampled for each LightGBM tree.
- `bagging_fraction`: Fraction of rows sampled for each LightGBM bagging iteration.
- `min_child_samples`: Minimum rows required in a LightGBM leaf.
- `features`: Fixed MLForecast feature configuration for `MLForecastLGBM`.
- `features.lags`: Lagged target values used as autoregressive features.
- `features.date_features`: Calendar-derived features, such as `dayofweek`, `week`, `month`, or `quarter`.
- `features.target_transforms`: Target transformations such as differencing or local standard scaling.
- `features.lag_transforms`: Rolling or exponentially weighted transforms applied to selected lags.
- `fit_params`: Fixed keyword arguments passed to `MLForecast.fit`.
- `fit_params.use_static_features`: Adapter flag for exposing configured static feature columns to the model.
- `fit_params.dropna`: Whether MLForecast drops rows with null feature values during training.
- `reuse_cv_splits`: Whether AutoMLForecast reuses the same cross-validation splits across trials.
- `model_hp_space`: LightGBM hyperparameter ranges passed through `AutoModel(config=...)`.
- `feature_space`: Candidate MLForecast feature configurations passed through `AutoMLForecast(init_config=...)`.
- `feature_space.lags`: Candidate lag lists for feature tuning.
- `feature_space.date_features`: Candidate calendar-feature lists for feature tuning.
- `feature_space.target_transforms`: Candidate target transformations for feature tuning.
- `feature_space.lag_transforms`: Candidate lag transform identifiers for feature tuning.
- `fit_space`: Candidate fit-time settings passed through `AutoMLForecast(fit_config=...)`.
- `fit_space.use_static_features`: Candidate values for static feature handling during HPO.
- `fit_space.dropna`: Candidate values for null-row dropping during HPO.

Supported MLForecast transform identifiers are `rolling_mean_<window>`, `rolling_std_<window>`, `ewm_alpha_<float>`, `differences_<lag>[_<lag>...]`, `local_standard_scaler`, and `none`.

### Foundation Model Settings

- `num_samples`: Number of sampled forecast paths for probabilistic foundation-model forecasts.
- `batch_size`: Number of series or windows scored per inference batch.
- `patch_size`: Patch length used by Moirai models.

`TimesFM_2_5_200m` defines no model-specific hyperparameters in `models_conf.yaml`.

### FoundationForecast Settings

Each `FF*` model forwards only the keys listed in its YAML entry to foundationforecast. A key set to `null` is not passed, so the library default applies.

- `context_length`: Maximum number of recent observations passed to the model. TimesFM models default to `512`; on an A10G this is about three times faster than `2048`. Raise it for long series with long seasonal patterns.
- `batch_size`: Number of series per inference batch.
- `dtype`: Weight precision for Chronos-2 models: `float32`, `bfloat16`, or `float16`.
- `per_core_batch_size`: Batch size per device for `FFTimesFM_2_5_200m`. Leaving it at the library default of `1` makes inference roughly 20 times slower. `FFTimesFM_3_0` does not accept this key.
- `num_samples`: Number of sampled paths for `FFToto` (Toto 1.0). The default is `32`; in our tests, `128` took four times as long with almost no change in point accuracy.
- `samples_per_batch`: Number of sample paths per batch for `FFToto`; it must divide `num_samples`.
- `decode_block_size`: Decoding block size for Toto-2 models.
- `scale_factor`: FlowState time-scale factor. When `null`, foundationforecast derives it from `freq`.

The `license` key in each entry is metadata used for the non-commercial warning; it is not passed to the model.