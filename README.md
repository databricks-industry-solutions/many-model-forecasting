# Many Model Forecasting by Databricks

## Introduction

Bootstrap your large-scale forecasting solutions on Databricks with the Many Models Forecasting (MMF) Solution Accelerator.

MMF expedites the development of sales and demand forecasting solutions on Databricks, including all critical phases: data preparation, training, backtesting, cross-validation, scoring, and deployment. Adopting a configuration-over-code approach, it minimizes the need for extensive coding.

MMF integrates a variety of well-established and cutting-edge algorithms, including **local statistical models**, **machine learning models**, **global deep learning models**, and **foundation time series models**. MMF enables parallel modeling of hundreds or thousands of time series leveraging Spark's distributed computing power. Users can apply multiple models at once and select the best performing one for each time series based on their custom metrics.

With its extensible architecture, MMF allows technically proficient users to incorporate new models and new algorithms. We recommend reading though the source code and modify it to your specific requirements. 

Get started now!

## Getting started

To run this solution on a public [M4](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset) dataset, clone this MMF repo into [Databricks Repos](https://www.databricks.com/product/repos).

### Local Models

Local models are used to model individual time series. We support models from [statsforecast](https://github.com/Nixtla/statsforecast), [r fable](https://cran.r-project.org/web/packages/fable/vignettes/fable.html) and [sktime](https://www.sktime.net/en/stable/). Covariates (i.e. exogenous regressors) are currently only supported for some statsforecast models. 

To get started, attach the [notebooks/demo_local_univariate_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/demo_local_univariate_daily.py) notebook to a cluster running [DBR 14.3 ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or later runtime. The cluster can be either a single-node or multi-node CPU cluster. Make sure to set the following [Spark configurations](https://spark.apache.org/docs/latest/configuration.html) on the cluster before you start using MMF: ```spark.sql.execution.arrow.enabled true``` and ```spark.sql.adaptive.enabled false``` (more detailed explanation to follow). 

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
    train_predict_ratio=2,
    data_quality_check=True,
    resample=False,
    ensemble=True,
    ensemble_metric="smape",
    ensemble_metric_avg=0.3,
    ensemble_metric_max=0.5,
    ensemble_scoring_output=f"{catalog}.{db}.daily_ensemble_output",
    active_models=active_models,
    experiment_path=f"/Shared/mmf_experiment",
    use_case_name="m4_daily",
)
```

#### Parameters description:

-  ```train_data``` is a delta table name that stores the input dataset.
-  ```scoring_data``` is a delta table name that stores the [future dynamical regressors](https://nixtlaverse.nixtla.io/neuralforecast/examples/exogenous_variables.html#3-training-with-exogenous-variables). If not provided or if the same name as ```train_data``` is provided, the models will ignore the future dynamical regressors. 
-  ```scoring_output``` is a delta table where you write your forecasting output. This table will be created if does not exist
-  ```evaluation_output``` is a delta table where you write the evalution results from all backtesting trials from all time series and all models. This table will be created if does not exist.
-  ```group_id``` is a column storing the unique id that groups your dataset to each time series.
-  ```date_col``` is your time column name.
-  ```target``` is your target column name.
-  ```freq``` is your prediction frequency. Currently, "D" for daily and "M" for monthly are supported. Note that ```freq``` supported is as per the model basis, hence check the model documentation carefully.
-  ```prediction_length``` is your forecasting horizon in the number of steps.
-  ```backtest_months``` specifies how many previous months you use for backtesting. 
-  ```stride``` is the number of steps in which you update your backtesting trial start date when going from one trial to the next.
-  ```train_predict_ratio``` specifies the minimum length required for your training dataset with respect to ```prediction_length```. If ```train_predict_ratio```=2, you need to have training dataset that is at least twice as long as ```prediciton_length```.
-  ```data_quality_check``` checks the quality of the input data if set to True. See [data_quality_checks.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/data_quality_checks.py) for the full details of the checks. 
-  ```resample``` backfills empty entries with 0 if set to True.
-  ```ensemble```: if you have forecasts from multiple models, you can take a simple mean, min and max of these values and generate an ensemble forecast. 
-  ```ensemble_metric``` is smape (symmetric mean absolute percentage error) by default. You can or add your own metrics at the main core of the forecasting_sa package or you simply can use ```evaluation_output``` to calculate any metric of your choice.
-  ```ensemble_metric_avg``` sets the maximum for the avg smape from each model, above which we exclude from ensembling.
-  ```ensemble_metric_max``` sets the maximum for the smape of each model, above which we exclude from ensembling.
-  ```ensemble_scoring_output``` is a delta table where you write the ensembled forecasts. 
-  ```active_models``` is a list of models you want to use.
-  ```experiment_path``` to keep metrics under the MLFlow.
-  ```use_case_name``` a new column will be created under the delta Table, in case you save multiple trials under 1 table.
  
To modify the model hyperparameters, directly change the values in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml) or overwrite these values in [mmf_sa/base_forecasting_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/base_forecasting_conf.yaml). 

MMF is fully integrated with MLflow, so once the training kicks off, the experiments will be visible in the MLflow Tracking UI with the corresponding metrics and parameters (note that we do not log all local models in MLFlow but we store the binary in the tables ```evaluation_output``` and ```scoring_output```). The metric you see in the MLflow Tracking UI is a simple mean over backtesting trials over all time series.

Other example notebooks for monthly forecasting and forecasting with exogenous regressors can be found in [notebooks/demo_local_univariate_monthly.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/demo_local_univariate_monthly.py) and [notebooks/demo_local_univariate_external_regressors_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/demo_local_univariate_external_regressors_daily.py).

### Global Models

Global models leverage patterns across multiple time series, enabling shared learning and improved predictions for each series. You typically train one big model for many or all time series. We support deep learning based models from [neuralforecast](https://nixtlaverse.nixtla.io/neuralforecast/index.html). Covariates (i.e. exogenous regressors) and hyperparameter tuning are both supported. 

To get started, attach the [notebooks/demo_global_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/demo_global_daily.py) notebook to a cluster running [DBR 14.3 ML](https://docs.databricks.com/en/release-notes/runtime/index.html) or later runtime. We recommend using a single-node cluster with multiple GPU instances such as [g4dn.12xlarge [T4]](https://aws.amazon.com/ec2/instance-types/g4/) on AWS or [Standard_NC64as_T4_v3](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series) on Azure. Multi-node setting is currently not supported.

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

Now, with the following command, we run the [notebooks/run_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/run_daily.py) that will run the ```run_forecast``` function and loop through the ```active_models``` list . The reason why we iterate through the models this way is because once a neuralforecast model is loaded to the memory, we need to restart the python kernel to use another model. 

```python
for model in active_models:
  dbutils.notebook.run(
    "run_daily",
    timeout_seconds=0, 
    arguments={"catalog": catalog, "db": db, "model": model})
```

Inside the [notebooks/run_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/run_daily.py), we have the ```run_forecast``` function specified as:

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
    train_predict_ratio=2,
    data_quality_check=True,
    resample=False,
    ensemble=True,
    ensemble_metric="smape",
    ensemble_metric_avg=0.3,
    ensemble_metric_max=0.5,
    ensemble_scoring_output=f"{catalog}.{db}.daily_ensemble_output",
    active_models=[model],
    experiment_path=f"/Shared/mmf_experiment",
    use_case_name="m4_daily",
    accelerator="gpu",
)
```

#### Parameters description:

The parameters are all the same except:
-  ```model_output``` is where you store your model.
-  ```use_case_name``` will be used to suffix the model name when registered to Unity Catalog.
-  ```accelerator``` tells MMF to use GPU instead of CPU.
  
To modify the model hyperparameters or reset the range of the hyperparameter optimization, directly change the values in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml) or overwrite these values in [mmf_sa/base_forecasting_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/base_forecasting_conf.yaml). 

MMF is fully integrated with MLflow and so once the training kicks off, the experiments will be visible in the MLflow Tracking UI with the corresponding metrics and parameters. Once the training is complete the models will be logged to MLFlow and registered to Unity Catalog. 

Other example notebooks for monthly forecasting and forecasting with exogenous regressors can be found in [notebooks/demo_global_monthly.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/demo_global_monthly.py) and [notebooks/demo_global_external_regressors_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/demo_global_external_regressors_daily.py) respectively.

### Foundation Models

Foundation time series models are large transformer based models pretrained on millions or billions of time series. These models can produce analysis (i.e. forecasting, anomaly detection, classfication) on an unforeseen time series without training or tuning. We support open source models from multiple sources: [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting), [moirai](https://blog.salesforceairesearch.com/moirai/), and [moment](https://github.com/moment-timeseries-foundation-model/moment). Covariates (i.e. exogenous regressors) and fine-tuning are currently not yet supported. This is a rapidly changing field, and we are working on updating the supported models and features as the field evolves.

To get started, attach the [notebooks/demo_foundation_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/demo_foundation_daily.py) notebook to a cluster running [DBR 14.3 ML](https://docs.databricks.com/en/release-notes/runtime/index.html) or later runtime. We recommend using a single-node cluster with multiple GPU instances such as [g4dn.12xlarge [T4]](https://aws.amazon.com/ec2/instance-types/g4/) on AWS or [Standard_NC64as_T4_v3](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series) on Azure. Multi-node setup is currently not supported. 

You can choose the models you want to evaluate and forecast by specifying them in a list:

```python
active_models = [
    "ChronosT5Tiny",
    "ChronosT5Mini",
    "ChronosT5Small",
    "ChronosT5Base",
    "ChronosT5Large",
    "MoiraiSmall",
    "MoiraiBase",
    "MoiraiLarge",
    "Moment1Large",
]
```

A comprehensive list of models currently supported by MMF is available in the [models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml). 

Now, with the following command, we run the [notebooks/run_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/run_daily.py) that will run the ```run_forecast``` function. We loop through the ```active_models``` list for the same reason mentioned above.

```python
for model in active_models:
  dbutils.notebook.run(
    "run_daily",
    timeout_seconds=0, 
    arguments={"catalog": catalog, "db": db, "model": model})
```

Inside the [notebooks/run_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/run_daily.py), we have the same ```run_forecast``` function as above. 
  
To modify the model hyperparameters, directly change the values in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml) or overwrite these values in [mmf_sa/base_forecasting_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/base_forecasting_conf.yaml). 

MMF is fully integrated with MLflow and so once the training kicks off, the experiments will be visible in the MLflow Tracking UI with the corresponding metrics and parameters. However, note that foundation models are currently not logged in MLFlow or registered to Unity Catalog. 

An example notebook for monthly forecasting can be found in [notebooks/demo_foundation_monthly.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/notebooks/demo_foundation_monthly.py).

## Project support
Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.
