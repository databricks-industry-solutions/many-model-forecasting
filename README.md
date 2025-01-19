# Many Model Forecasting by Databricks

## Introduction

Bootstrap your large-scale forecasting solutions on Databricks with the Many Models Forecasting (MMF) Solution Accelerator.

MMF accelerates the development of sales and demand forecasting solutions on Databricks, including critical phases of data preparation, training, backtesting, cross-validation, scoring, and deployment. Adopting a configuration-over-code approach, MMF minimizes the need for extensive coding. But with its extensible architecture, MMF allows technically proficient users to incorporate new models and algorithms. We recommend users to read through the source code, and modify it to their specific requirements.

MMF integrates a variety of well-established and cutting-edge algorithms, including [local statistical models](https://github.com/databricks-industry-solutions/many-model-forecasting?tab=readme-ov-file#local-models), [global deep learning models](https://github.com/databricks-industry-solutions/many-model-forecasting?tab=readme-ov-file#global-models), and [foundation time series models](https://github.com/databricks-industry-solutions/many-model-forecasting?tab=readme-ov-file#foundation-models). MMF enables parallel modeling of hundreds or thousands of time series leveraging Spark's distributed compute. Users can apply multiple models at once and select the best performing one for each time series based on their custom metrics.

Get started now!

## What's New

- Jan 2025: [TimesFM](https://github.com/google-research/timesfm) is available for univariate forecasting. Try the [notebook](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation_daily.py).
- Jan 2025: [Chronos Bolt](https://github.com/amazon-science/chronos-forecasting) models are available for univariate forecasting. Try the [notebook](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation_daily.py).
- Jan 2025: [Moirai MoE](https://github.com/SalesforceAIResearch/uni2ts) models are available for univariate forecasting. Try the [notebook](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation_daily.py).

## Getting started

To run this solution on a public [M4](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset) dataset, clone this MMF repo into your [Databricks Repos](https://www.databricks.com/product/repos).

### Local Models

Local models are used to model individual time series. They could be advantageous over other types of model for their capabilities to tailor fit to individual series, offer greater interpretability, and require lower data requirements. We support models from [statsforecast](https://github.com/Nixtla/statsforecast), [r fable](https://cran.r-project.org/web/packages/fable/vignettes/fable.html) and [sktime](https://www.sktime.net/en/stable/). Covariates (i.e. exogenous regressors) are currently only supported for some models from statsforecast. 

To get started, attach the [examples/local_univariate_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/local_univariate_daily.py) notebook to a cluster running [DBR 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or later versions. The cluster can be either a single-node or multi-node CPU cluster. Make sure to set the following [Spark configurations](https://spark.apache.org/docs/latest/configuration.html) on the cluster before you start using MMF: ```spark.sql.execution.arrow.enabled true``` and ```spark.sql.adaptive.enabled false``` (more detailed explanation to follow). 

In this notebook, we will apply 20+ models to 100 time series. You can specify the models to use in a list:

```python
active_models = [
    "StatsForecastBaselineWindowAverage",
    "StatsForecastBaselineSeasonalWindowAverage",
    "StatsForecastBaselineNaive",
    "StatsForecastBaselineSeasonalNaive",
    "StatsForecastAutoArima",
    "StatsForecastAutoETS",
    "StatsForecastAutoCES",
    "StatsForecastAutoTheta",
    "StatsForecastTSB",
    "StatsForecastADIDA",
    "StatsForecastIMAPA",
    "StatsForecastCrostonClassic",
    "StatsForecastCrostonOptimized",
    "StatsForecastCrostonSBA",
    "RFableArima",
    "RFableETS",
    "RFableNNETAR",
    "RFableEnsemble",
    "RDynamicHarmonicRegression",
    "SKTimeTBats",
    "SKTimeLgbmDsDt",
]
```

A comprehensive list of local models currently supported by MMF is available in the [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml). 

Now, run the forecasting using ```run_forecast``` function with the ```active_models``` list specified above:

```python

catalog = "your_catalog_name"
db = "your_db_name"

run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.m4_daily_train",
    scoring_data=f"{catalog}.{db}.m4_daily_train",
    scoring_output=f"{catalog}.{db}.daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.daily_evaluation_output",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=10,
    backtest_months=1,
    stride=10,
    metric="smape",
    train_predict_ratio=2,
    data_quality_check=True,
    resample=False,
    active_models=active_models,
    experiment_path="/Shared/mmf_experiment",
    use_case_name="m4_daily",
)
```

#### Parameters description:

- ```train_data``` is a delta table name that stores the input dataset.
- ```scoring_data``` is a delta table name that stores the [dynamic future regressors](https://nixtlaverse.nixtla.io/neuralforecast/examples/exogenous_variables.html#3-training-with-exogenous-variables). If not provided or if the same name as ```train_data``` is provided, the models will ignore the future dynamical regressors. 
- ```scoring_output``` is a delta table where you write your forecasting output. This table will be created if does not exist
- ```evaluation_output``` is a delta table where you write the evaluation results from all backtesting trials from all time series and all models. This table will be created if does not exist.
- ```group_id``` is a column storing the unique id that groups your dataset to each time series.
- ```date_col``` is your time column name.
- ```target``` is your target column name.
- ```freq``` is your prediction frequency. Currently, "D" for daily and "M" for monthly are supported. Note that ```freq``` supported is as per the model basis, hence check the model documentation carefully. Monthly forecasting expects the timestamp column in ```train_data``` and ```scoring_output``` to be the last day of the month.
- ```prediction_length``` is your forecasting horizon in the number of steps.
- ```backtest_months``` specifies how many previous months you use for backtesting. 
- ```stride``` is the number of steps in which you update your backtesting trial start date when going from one trial to the next.
- ```metric``` is the metric to log in the evaluation table and MLFlow. Supported metrics are mae, mse, rmse, mape and smape. Default is smape.
- ```train_predict_ratio``` specifies the minimum length required for your training dataset with respect to ```prediction_length```. If ```train_predict_ratio```=2, you need to have training dataset that is at least twice as long as ```prediciton_length```.
- ```data_quality_check``` checks the quality of the input data if set to True (default False). See [data_quality_checks.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/data_quality_checks.py) for the full details of the checks. 
- ```resample``` backfills skipped entries with 0 if set to True. Only relevant when data_quality_check is True. Default is False. If data_quality_check is True and resample is False, the check removes all time series with skipped dates.
- ```active_models``` is a list of models you want to use.
- ```experiment_path``` to keep metrics under the MLFlow.
- ```use_case_name``` a new column will be created under the delta Table, in case you save multiple trials under 1 table.
  
To modify the model hyperparameters, change the values in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml) or overwrite these values in [mmf_sa/forecasting_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/forecasting_conf.yaml). 

MMF is fully integrated with MLflow, so once the training kicks off, the experiments will be visible in the MLflow Tracking UI with the corresponding metrics and parameters (note that we do not log all local models in MLFlow, but we store the binaries in the tables ```evaluation_output``` and ```scoring_output```). The metric you see in the MLflow Tracking UI is a simple mean over backtesting trials over all time series.

We encourage you to read through [examples/local_univariate_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/local_univariate_daily.py) notebook to better understand how local models can be applied to your time series using MMF. Other example notebooks for monthly forecasting and forecasting with exogenous regressors can be found in [examples/local_univariate_monthly.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/local_univariate_monthly.py) and [examples/local_univariate_external_regressors_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/local_univariate_external_regressors_daily.py).

### Global Models

Global models leverage patterns across multiple time series, enabling shared learning and improved predictions for each series. You would typically train one big model for many or all time series. They can often deliver better performance and robustness for forecasting large and similar datasets. We support deep learning based models from [neuralforecast](https://nixtlaverse.nixtla.io/neuralforecast/index.html). Covariates (i.e. exogenous regressors) and hyperparameter tuning are both supported for some models. 

To get started, attach the [examples/global_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/global_daily.py) notebook to a cluster running [DBR 14.3LTS for ML](https://docs.databricks.com/en/release-notes/runtime/index.html) or later version. We recommend using a single-node cluster with multiple GPU instances such as [g4dn.12xlarge [T4]](https://aws.amazon.com/ec2/instance-types/g4/) on AWS or [Standard_NC64as_T4_v3](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series) on Azure. Multi-node setting is currently not supported.

You can choose the models to train and put them in a list:

```python
active_models = [
    "NeuralForecastRNN",
    "NeuralForecastLSTM",
    "NeuralForecastNBEATSx",
    "NeuralForecastNHITS",
    "NeuralForecastAutoRNN",
    "NeuralForecastAutoLSTM",
    "NeuralForecastAutoNBEATSx",
    "NeuralForecastAutoNHITS",
    "NeuralForecastAutoTiDE",
    "NeuralForecastAutoPatchTST",
]
```

The models prefixed with "Auto" perform hyperparameter optimization within a specified range (see below for more detail). A comprehensive list of models currently supported by MMF is available in the [models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml).

Now, with the following command, we run the [examples/run_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_daily.py) notebook that will in turn call ```run_forecast``` function and loop through the ```active_models``` list . 

```python
for model in active_models:
  dbutils.notebook.run(
    "run_daily",
    timeout_seconds=0, 
    arguments={"catalog": catalog, "db": db, "model": model, "run_id": run_id})
```

Inside the [examples/run_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_daily.py), we have the ```run_forecast``` function specified as:

```python
run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.m4_daily_train",
    scoring_data=f"{catalog}.{db}.m4_daily_train",
    scoring_output=f"{catalog}.{db}.daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.daily_evaluation_output",
    model_output=f"{catalog}.{db}",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=10,
    backtest_months=1,
    stride=10,
    metric="smape",
    train_predict_ratio=2,
    data_quality_check=True,
    resample=False,
    active_models=[model],
    experiment_path="/Shared/mmf_experiment",
    use_case_name="m4_daily",
    run_id=run_id,
    accelerator="gpu",
)
```

#### Parameters description:

The parameters are all the same except:
-  ```model_output``` is where you store your model.
-  ```use_case_name``` will be used to suffix the model name when registered to Unity Catalog.
-  ```accelerator``` tells MMF to use GPU instead of CPU.
  
To modify the model hyperparameters or reset the range of the hyperparameter search, change the values in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml) or overwrite these values in [mmf_sa/forecasting_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/forecasting_conf.yaml). 

MMF is fully integrated with MLflow and so once the training kicks off, the experiments will be visible in the MLflow Tracking UI with the corresponding metrics and parameters. Once the training is complete the models will be logged to MLFlow and registered to Unity Catalog. 

We encourage you to read through [examples/global_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/global_daily.py) notebook to better understand how global models can be applied to your time series using MMF. Other example notebooks for monthly forecasting and forecasting with exogenous regressors can be found in [examples/global_monthly.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/global_monthly.py) and [examples/global_external_regressors_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/global_external_regressors_daily.py) respectively.

### Foundation Models

Foundation time series models are mostly transformer based models pretrained on millions or billions of time points. These models can perform analysis (i.e. forecasting, anomaly detection, classification) on a previously unseen time series without training or tuning. We support open source models from multiple sources: [chronos](https://github.com/amazon-science/chronos-forecasting), [timesfm](https://github.com/google-research/timesfm), and [moirai](https://blog.salesforceairesearch.com/moirai/). Covariates (i.e. exogenous regressors) and fine-tuning are currently not yet supported. This is a rapidly changing field, and we are working on updating the supported models and new features as the field evolves.

To get started, attach the [examples/foundation_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation_daily.py) notebook to a cluster running [DBR 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/index.html) or later versions. We recommend using a single-node cluster with multiple GPU instances such as [g4dn.12xlarge [T4]](https://aws.amazon.com/ec2/instance-types/g4/) on AWS or [Standard_NC64as_T4_v3](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series) on Azure. Multi-node setup is currently not supported. 

You can choose the models you want to evaluate and forecast by specifying them in a list:

```python
active_models = [
    "ChronosT5Tiny",
    "ChronosT5Mini",
    "ChronosT5Small",
    "ChronosT5Base",
    "ChronosT5Large",
    "ChronosBoltTiny",
    "ChronosBoltMini",
    "ChronosBoltSmall",
    "ChronosBoltBase",
    "MoiraiSmall",
    "MoiraiBase",
    "MoiraiLarge",
    "MoiraiMoESmall",
    "MoiraiMoEBase",
    "TimesFM_1_0_200m",
    "TimesFM_2_0_500m",
]
```

A comprehensive list of models currently supported by MMF is available in the [models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml). 

Now, with the following command, we run [examples/run_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_daily.py) notebook that will in turn run ```run_forecast``` function. We loop through the ```active_models``` list for the same reason mentioned above (see the global model section).

```python
for model in active_models:
  dbutils.notebook.run(
    "run_daily",
    timeout_seconds=0, 
    arguments={"catalog": catalog, "db": db, "model": model, "run_id": run_id})
```

Inside the [examples/run_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_daily.py), we have the same ```run_forecast``` function as above. 
  
To modify the model hyperparameters, change the values in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml) or overwrite these values in [mmf_sa/forecasting_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/forecasting_conf.yaml). 

MMF is fully integrated with MLflow and so once the training kicks off, the experiments will be visible in the MLflow Tracking UI with the corresponding metrics and parameters. During the evaluation, the models are logged and registered to Unity Catalog.

We encourage you to read through [examples/foundation_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation_daily.py) notebook to better understand how foundation models can be applied to your time series using MMF. An example notebook for monthly forecasting can be found in [examples/foundation_monthly.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation_monthly.py).

#### Using Time Series Foundation Models on Databricks

If you want to try out time series foundation models on Databricks without MMF, you can find example notebooks in [databricks-industry-solutions/transformer_forecasting](https://github.com/databricks-industry-solutions/transformer_forecasting). These notebooks will show you how you can load, distribute the inference, fine-tune, register, deploy a model and generate online forecasts on it. We have notebooks for [TimeGPT](https://docs.nixtla.io/), [Chronos](https://github.com/amazon-science/chronos-forecasting), [Moirai](https://github.com/SalesforceAIResearch/uni2ts), [Moment](https://github.com/moment-timeseries-foundation-model/moment), and [TimesFM](https://github.com/google-research/timesfm).

## [Vector Lab](https://www.youtube.com/@VectorLab) - Many Model Forecasting

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/wYeuPxtap-8/0.jpg)](https://www.youtube.com/watch?v=wYeuPxtap-8)

## Project support
Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| rpy2 | Python interface to the R language (embedded R) | GNU General Public License v2 or later | https://pypi.org/project/rpy2/
| kaleido | Static image export for web-based visualization libraries with zero dependencies | MIT | https://pypi.org/project/kaleido/
| fugue | An abstraction layer for distributed computation | Apache 2.0 | https://pypi.org/project/fugue/
| Jinja2 | A very fast and expressive template engine | BSD | https://pypi.org/project/Jinja2/
| omegaconf | A flexible configuration library | BSD | https://pypi.org/project/omegaconf/
| missingno | Missing data visualization module for Python | MIT | https://pypi.org/project/missingno/
| datasetsforecast | Datasets for Time series forecasting | MIT | https://pypi.org/project/datasetsforecast/
| statsforecast | Time series forecasting suite using statistical models | Apache 2.0 | https://pypi.org/project/statsforecast/
| neuralforecast | Time series forecasting suite using deep learning models | Apache 2.0 | https://pypi.org/project/neuralforecast/
| fable | Forecasting Models for Tidy Time Series | GPL-3 | https://cran.r-project.org/web/packages/fable/index.html
| fabletools | Core Tools for Packages in the 'fable' Framework | GPL-3 | https://cran.r-project.org/web/packages/fabletools/index.html
| feasts | Feature Extraction and Statistics for Time Series | GPL-3 | https://cran.r-project.org/web/packages/feasts/index.html
| lazyeval | Lazy (Non-Standard) Evaluation | GPL-3 | https://cran.r-project.org/web/packages/lazyeval/index.html
| tsibble | Tidy Temporal Data Frames and Tools | GPL-3 | https://cran.r-project.org/web/packages/tsibble/index.html
| urca | Unit Root and Cointegration Tests for Time Series Data | GPL-3 | https://cran.r-project.org/web/packages/urca/index.html
| sktime | A unified framework for machine learning with time series | BSD 3-Clause | https://pypi.org/project/sktime/
| tbats | BATS and TBATS for time series forecasting | MIT | https://pypi.org/project/tbats/
| lightgbm | LightGBM Python Package | MIT | https://pypi.org/project/lightgbm/
| Chronos | Pretrained (Language) Models for Probabilistic Time Series Forecasting | Apache 2.0 | https://github.com/amazon-science/chronos-forecasting
| Moirai | Unified Training of Universal Time Series Forecasting Transformers | Apache 2.0 | https://github.com/SalesforceAIResearch/uni2ts
| Moment | A Family of Open Time-series Foundation Models | MIT | https://github.com/moment-timeseries-foundation-model/moment
| TimesFM | A pretrained time-series foundation model developed by Google Research for time-series forecasting | Apache 2.0 | https://github.com/google-research/timesfm
