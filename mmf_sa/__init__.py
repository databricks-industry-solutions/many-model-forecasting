__version__ = "0.0.1"
import pathlib
import sys
from typing import Union, Any, Dict, List
import importlib.resources as pkg_resources
import pandas as pd
import yaml
from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer
from pyspark.sql import SparkSession, DataFrame
from mmf_sa.Forecaster import Forecaster


def run_forecast(
    spark: SparkSession,
    train_data: Union[str, pd.DataFrame, DataFrame],
    group_id: str,
    date_col: str,
    target: str,
    freq: str,
    prediction_length: int,
    backtest_length: int,
    stride: int,
    metric: str = "smape",
    scoring_data: Union[str, pd.DataFrame, DataFrame] = None,
    scoring_output: str = None,
    evaluation_output: str = None,
    model_output: str = None,
    use_case_name: str = None,
    static_features: List[str] = None,
    dynamic_future_numerical: List[str] = None,
    dynamic_future_categorical: List[str] = None,
    dynamic_historical_numerical: List[str] = None,
    dynamic_historical_categorical: List[str] = None,
    active_models: List[str] = None,
    accelerator: str = "cpu",
    backtest_retrain: bool = None,
    train_predict_ratio: int = None,
    data_quality_check: bool = False,
    resample: bool = False,
    experiment_path: str = None,
    run_id: str = None,
    conf: Union[str, Dict[str, Any], OmegaConf] = None,
) -> str:

    """
    The function creates a Forecaster object with the provided configuration.
    And then calls the evaluate_score method to perform the evaluation and forecasting.
    The function returns the run id.

    Parameters:
        spark (SparkSession): A SparkSession object.
        train_data (Union[str, pd.DataFrame, DataFrame]): Training data as a string of delta table name, pandas DataFrame, or Spark DataFrame.
        group_id (str): A string specifying the column name that groups the training data into individual time series.
        date_col (str): A string specifying the column name that stores the date variable.
        target (str): A string specifying the column name of the target variable.
        freq (str): A string specifying the frequency. "H" for hourly, "D" for daily, "W" for weekly and "M" for monthly are supported.
        prediction_length (int): An integer specifying the prediction length: i.e. forecasting horizon.
        backtest_length (int): An integer specifying the number of time points to be used for backtesting.
        stride (int): An integer specifying the stride length.
        metric (str): A string specifying the metric to use for evaluation. Supported metrics are mae, mse, rmse, mape and smape. Default is smape.
        scoring_data (Union[str, pd.DataFrame, DataFrame]): Scoring data as a string of delta table name, pandas DataFrame, or Spark DataFrame.
        scoring_output (str): A string specifying the output table name for scoring.
        evaluation_output (str): A string specifying the output table name for evaluation.
        model_output (str): A string specifying the output path for the model.
        use_case_name (str): A string specifying the use case name.
        static_features (List[str]): A list of strings specifying the static features.
        dynamic_future_numerical (List[str]): A list of strings specifying the dynamic future features that are numerical.
        dynamic_future_categorical (List[str]): A list of strings specifying the dynamic future features that are categorical.
        dynamic_historical_numerical (List[str]): A list of strings specifying the dynamic historical features that are numerical.
        dynamic_historical_categorical (List[str]): A list of strings specifying the dynamic historical features that are categorical.
        active_models (List[str]): A list of strings specifying the active models.
        accelerator (str): A string specifying the accelerator to use: cpu or gpu. Default is cpu.
        backtest_retrain (bool): A boolean specifying whether to retrain the model during backtesting. Currently, not supported.
        train_predict_ratio (int): An integer specifying the train predict ratio.
        data_quality_check (bool): A boolean specifying whether to check the data quality. Default is False.
        resample (bool): A boolean specifying whether to back-fill skipped entries with 0. Only relevant when data_quality_check is True. Default is False.
        experiment_path (str): A string specifying the experiment path.
        run_id (str): A string specifying the run id. If not provided a random string is generated and assigned to each run.
        conf (Union[str, Dict[str, Any], OmegaConf]): A configuration object.

    Returns:
    Dict[str, Union[int, str]]: A dictionary with an integer and a string as values.
    """

    if isinstance(conf, dict):
        _conf = OmegaConf.create(conf)
    elif isinstance(conf, str):
        _yaml_conf = yaml.safe_load(pathlib.Path(conf).read_text())
        _conf = OmegaConf.create(_yaml_conf)
    elif isinstance(conf, BaseContainer):
        _conf = conf
    else:
        _conf = OmegaConf.create()

    base_conf = OmegaConf.create(
        pkg_resources.read_text(sys.modules[__name__], "forecasting_conf.yaml")
    )
    _conf = OmegaConf.merge(base_conf, _conf)

    _data_conf = {}
    if train_data is not None and (isinstance(train_data, pd.DataFrame) or isinstance(train_data, DataFrame)):
        _data_conf["train_data"] = train_data
    else:
        _conf["train_data"] = train_data
    _conf["group_id"] = group_id
    _conf["date_col"] = date_col
    _conf["target"] = target
    _conf["freq"] = freq
    _conf["prediction_length"] = prediction_length
    _conf["backtest_length"] = backtest_length
    _conf["stride"] = stride
    _conf["metric"] = metric
    _conf["resample"] = resample
    run_evaluation = True
    run_scoring = False
    if scoring_data is not None and scoring_output is not None:
        run_scoring = True
        _conf["scoring_output"] = scoring_output
        if scoring_data is not None and (isinstance(scoring_data, pd.DataFrame) or isinstance(scoring_data, DataFrame)):
            _data_conf["scoring_data"] = scoring_data
        else:
            _conf["scoring_data"] = scoring_data
    if use_case_name is not None:
        _conf["use_case_name"] = use_case_name
    if active_models is not None:
        _conf["active_models"] = active_models
    if accelerator is not None:
        _conf["accelerator"] = accelerator
    if backtest_retrain is not None:
        _conf["backtest_retrain"] = backtest_retrain
    if train_predict_ratio is not None:
        _conf["train_predict_ratio"] = train_predict_ratio
    if experiment_path is not None:
        _conf["experiment_path"] = experiment_path
    if evaluation_output is not None:
        _conf["evaluation_output"] = evaluation_output
    if model_output is not None:
        _conf["model_output"] = model_output
    if data_quality_check is not None:
        _conf["data_quality_check"] = data_quality_check
    if static_features is not None:
        _conf["static_features"] = static_features
    if dynamic_future_numerical is not None:
        _conf["dynamic_future_numerical"] = dynamic_future_numerical
    if dynamic_future_categorical is not None:
        _conf["dynamic_future_categorical"] = dynamic_future_categorical
    if dynamic_historical_numerical is not None:
        _conf["dynamic_historical_numerical"] = dynamic_historical_numerical
    if dynamic_historical_categorical is not None:
        _conf["dynamic_historical_categorical"] = dynamic_historical_categorical
    if run_id is not None:
        _conf["run_id"] = run_id

    f = Forecaster(
        conf=_conf,
        data_conf=_data_conf,
        spark=spark,
        run_id=run_id,
    )

    run_id = f.evaluate_score(evaluate=run_evaluation, score=run_scoring)

    return run_id


__all__ = ["run_forecast", "Forecaster"]
