# Supported Models

Model hyperparameters can be modified under [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml).

## Local
| model | source                                                                                                                                                              | covariate support |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| StatsForecastBaselineWindowAverage | [Statsforecast Window Average](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#windowaverage)                                                      |  | 
| StatsForecastBaselineSeasonalWindowAverage | [Statsforecast Seasonal Window Average](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#seasonalwindowaverage)                                     |  | 
| StatsForecastBaselineNaive | [Statsforecast Naive](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#naive)                                                                       |  | 
| StatsForecastBaselineSeasonalNaive | [Statsforecast Seasonal Naive](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#seasonalnaive)                                                      |  | 
| StatsForecastAutoArima | [Statsforecast AutoARIMA](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoarima)                                                               | ✅ | 
| StatsForecastAutoETS | [Statsforecast AutoETS](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoets)                                                                   |  | 
| StatsForecastAutoCES | [Statsforecast AutoCES](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoces)                                                                   |  | 
| StatsForecastAutoTheta | [Statsforecast AutoTheta](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotheta)                                                               |  | 
| StatsForecastAutoTbats | [Statsforecast AutoTBATS](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotbats) |  | 
| StatsForecastAutoMfles | [Statsforecast AutoMFLES](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#automfles) | ✅ | 
| StatsForecastTSB | [Statsforecast TSB](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#tsb)                                                                           |  | 
| StatsForecastADIDA | [Statsforecast ADIDA](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#adida)                                                                       |  | 
| StatsForecastIMAPA | [Statsforecast IMAPA](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#imapa)                                                                       |  | 
| StatsForecastCrostonClassic | [Statsforecast Croston Classic](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#crostonclassic)                                                    |  | 
| StatsForecastCrostonOptimized | [Statsforecast Croston Optimized](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#crostonoptimized)                                                |  | 
| StatsForecastCrostonSBA | [Statsforecast Croston SBA](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#crostonsba)                                                            |  | 
| SKTimeTBats | [sktime TBATS](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.tbats.TBATS.html)                                                   |  | 
| SKTimeProphet | [sktime Prophet](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.fbprophet.Prophet.html)                                           |  | 
| SKTimeLgbmDsDt | [SKTimeLgbmDsDt](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/sktime/SKTimeForecastingPipeline.py)               |  | 

## Global
| model | source | covariate support |
|----------------------------------------|-------------------------|------------|
| NeuralForecastRNN | [NeuralForecast RNN](https://nixtlaverse.nixtla.io/neuralforecast/models.rnn.html) | ✅ | 
| NeuralForecastLSTM | [NeuralForecast LSTM](https://nixtlaverse.nixtla.io/neuralforecast/models.lstm.html) | ✅ | 
| NeuralForecastNBEATSx | [NeuralForecast NBEATSx](https://nixtlaverse.nixtla.io/neuralforecast/models.nbeatsx.html) | ✅ | 
| NeuralForecastNHITS | [NeuralForecast NHITS](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html) | ✅ | 
| NeuralForecastAutoRNN | [NeuralForecast AutoRNN](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autornn) | ✅ | 
| NeuralForecastAutoLSTM | [NeuralForecast AutoLSTM](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autolstm) | ✅ | 
| NeuralForecastAutoNBEATSx | [NeuralForecast AutoNBEATSx](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autonbeatsx) | ✅ | 
| NeuralForecastAutoNHITS | [NeuralForecast AutoNHITS](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autonhits) | ✅ | 
| NeuralForecastAutoTiDE | [NeuralForecast AutoTiDE](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autotide) | ✅ | 
| NeuralForecastAutoPatchTST | [NeuralForecast AutoPatchTST](https://nixtlaverse.nixtla.io/neuralforecast/models.html#autopatchtst) |  | 

## Foundation
| model | source | covariate support |
|----------------------------------------|-------------------------|------------|
| ChronosT5Tiny | [amazon/chronos-t5-tiny](https://huggingface.co/amazon/chronos-t5-tiny) |  | 
| ChronosT5Mini | [amazon/chronos-t5-mini](https://huggingface.co/amazon/chronos-t5-mini) |  | 
| ChronosT5Small | [amazon/chronos-t5-small](https://huggingface.co/amazon/chronos-t5-small) |  | 
| ChronosT5Base | [amazon/chronos-t5-base](https://huggingface.co/amazon/chronos-t5-base) |  | 
| ChronosT5Large | [amazon/chronos-t5-large](https://huggingface.co/amazon/chronos-t5-large) |  | 
| ChronosBoltTiny | [amazon/chronos-bolt-tiny](https://huggingface.co/amazon/chronos-bolt-tiny) |  | 
| ChronosBoltMini | [amazon/chronos-bolt-mini](https://huggingface.co/amazon/chronos-bolt-mini) |  | 
| ChronosBoltSmall | [amazon/chronos-bolt-small](https://huggingface.co/amazon/chronos-bolt-small) |  | 
| ChronosBoltBase | [amazon/chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base) |  | 
| MoiraiSmall | [Salesforce/moirai-1.1-R-small](https://huggingface.co/Salesforce/moirai-1.1-R-small) |  | 
| MoiraiBase | [Salesforce/moirai-1.1-R-base](https://huggingface.co/Salesforce/moirai-1.1-R-base) |  | 
| MoiraiLarge | [Salesforce/moirai-1.1-R-large](https://huggingface.co/Salesforce/moirai-1.1-R-large) |  | 
| MoiraiMoESmall | [Salesforce/moirai-moe-1.0-R-small](https://huggingface.co/Salesforce/moirai-moe-1.0-R-small) |  | 
| MoiraiMoEBase | [Salesforce/moirai-moe-1.0-R-base](https://huggingface.co/Salesforce/moirai-moe-1.0-R-base) |  | 
| TimesFM_1_0_200m | [google/timesfm-1.0-200m-pytorch](https://huggingface.co/google/timesfm-1.0-200m-pytorch) | ✅ | 
| TimesFM_2_0_500m | [google/timesfm-2.0-500m-pytorch](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | ✅ |
