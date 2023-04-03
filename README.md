# Many Model Forecasting by Databricks

## Introduction

Bootstrap your large scale forecasting solution on Databricks with Many Models Forecasting (MMF) 
Project.

MMF accelerates the development of Sales / Demand Forecasting solution by automating all phases of the ML lifecycle. 
This includes data exploration, preparation, training, backtesting, scoring, and deployment.
It follows the configuration over code approach and requires no coding.
MMF can model hundreds or thousands of time series at the same time leveraging Spark's powerful compute.
It brings together many well-known and state-of-the-art algorithms (local statistical models, global deep learning models, etc.) 
and libraries. 
The Project has an extensible architecture and allows users with technical skills to bring in their models and 
algorithms. 
Get started now!

## Getting started

To run this solution using a public [M4](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset) 
dataset, clone this MMF repo into Databricks Repos.

Attach the [01_mm_forecasting_demo.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/01_mm_forecasting_demo.py) 
notebook to any cluster running a DBR 11.2 ML or later runtime. 
You can choose the models to train on your input data by specifying them in a list:
```python
active_models = [
    "StatsForecastArima",
    "StatsForecastETS",
    "RDynamicHarmonicRegression",
    "RFableEnsemble",
    "GluonTSTorchDeepAR",
    "GluonTSNBEATS",
    "GluonTSProphet",
    "SKTimeLgbmDsDt",
    "SKTimeTBats",
]
```
A comprehensive list of models currently supported by MMF is available in the [models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/forecasting_sa/models/models_conf.yaml). 
Now, run the forecasting using ```run_forecast``` function with the ```active_models``` list specified above:
```python
run_forecast(
    spark=spark,
    train_data="train",
    scoring_data="train",
    scoring_output="forecast_scoring_out",
    metrics_output="metrics",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    data_quality_check=False,
    ensemble=True,
    ensemble_metric="smape",
    ensemble_metric_avg=0.3,
    ensemble_metric_max=0.5,
    ensemble_scoring_output="forecast_ensemble_out",
    train_predict_ratio=2,
    active_models=active_models,
    experiment_path=f"/Shared/fsa_cicd_pr_experiment",
    use_case_name="fsa",
)
```
See the notebook for the detail descriptions of the parameters. But in a nutshell, ```train_data``` is the input data set, 
```scoring_output``` is a delta table where you want to write your forecasting output into, ```metrics_output``` is a 
delta table where you want to write metrics from all backtest windows from all models into, and ```active_models``` is a 
list of models you want to use. 
To configure the model parameters, change the values in [base_forecasting_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/forecasting_sa/base_forecasting_conf.yaml). 

MMF is fully integrated with MLflow and so once the training kicks off, the experiments will be visible in the MLflow UI. 

## Project support
Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.
