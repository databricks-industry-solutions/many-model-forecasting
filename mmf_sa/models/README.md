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