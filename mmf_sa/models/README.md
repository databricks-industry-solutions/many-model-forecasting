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
| SKTimeProphet | [sktime Prophet](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.fbprophet.Prophet.html)                                           |  | 

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
| ChronosBoltTiny | [amazon/chronos-bolt-tiny](https://huggingface.co/amazon/chronos-bolt-tiny) |  | 
| ChronosBoltMini | [amazon/chronos-bolt-mini](https://huggingface.co/amazon/chronos-bolt-mini) |  | 
| ChronosBoltSmall | [amazon/chronos-bolt-small](https://huggingface.co/amazon/chronos-bolt-small) |  | 
| ChronosBoltBase | [amazon/chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base) |  | 
| Chronos2 | [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) | ✅ | 
| Chronos2Small | [autogluon/chronos-2-small](https://huggingface.co/autogluon/chronos-2-small) | ✅ | 
| Chronos2Synth | [autogluon/chronos-2-synth](https://huggingface.co/autogluon/chronos-2-synth) | ✅ | 
| TimesFM_2_5_200m | [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | ✅ |
| ~~MoiraiSmall~~ | ~~[Salesforce/moirai-1.1-R-small](https://huggingface.co/Salesforce/moirai-1.1-R-small)~~ | | *Temporarily disabled: `uni2ts` requires `torch<2.5`, incompatible with DBR ML 18.0* |
| ~~MoiraiBase~~ | ~~[Salesforce/moirai-1.1-R-base](https://huggingface.co/Salesforce/moirai-1.1-R-base)~~ | | *Temporarily disabled* |
| ~~MoiraiLarge~~ | ~~[Salesforce/moirai-1.1-R-large](https://huggingface.co/Salesforce/moirai-1.1-R-large)~~ | | *Temporarily disabled* |
| ~~MoiraiMoESmall~~ | ~~[Salesforce/moirai-moe-1.0-R-small](https://huggingface.co/Salesforce/moirai-moe-1.0-R-small)~~ | | *Temporarily disabled* |
| ~~MoiraiMoEBase~~ | ~~[Salesforce/moirai-moe-1.0-R-base](https://huggingface.co/Salesforce/moirai-moe-1.0-R-base)~~ | | *Temporarily disabled* |