__version__ = "0.0.1"

import pathlib
import sys
from typing import Union, Any, Dict, List

import importlib.resources as pkg_resources
import yaml
from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer
from pyspark.sql import SparkSession

from forecasting_sa.Forecaster import Forecaster


def run_forecast(
    spark: SparkSession,
    train_data: str,
    group_id: str = "unique_id",
    date_col: str = "date",
    target: str = "y",
    freq: str = "D",
    prediction_length: int = 10,
    backtest_months: int = 1,
    stride: int = 10,
    scoring_data: str = None,
    scoring_output: str = None,
    metrics_output: str = None,
    ensemble: bool = None,
    ensemble_metric: str = None,
    ensemble_metric_avg: float = None,
    ensemble_metric_max: float = None,
    ensemble_scoring_output: str = None,
    use_case_name: str = None,
    active_models: List[str] = None,
    accelerator: str = None,
    scoring_model_stage: str = None,
    selection_metric: str = None,
    backtest_retrain: bool = None,
    train_predict_ratio: int = None,
    data_quality_check: bool = None,
    experiment_path: str = None,
    conf: Union[str, Dict[str, Any], OmegaConf] = None,
) -> str:
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
        pkg_resources.read_text(sys.modules[__name__], "base_forecasting_conf.yaml")
    )
    _conf = OmegaConf.merge(base_conf, _conf)

    _conf["train_data"] = train_data
    _conf["group_id"] = group_id
    _conf["date_col"] = date_col
    _conf["target"] = target
    _conf["freq"] = freq
    _conf["prediction_length"] = prediction_length
    _conf["backtest_months"] = backtest_months
    _conf["stride"] = stride

    run_scoring = False
    if scoring_data is not None and scoring_output is not None:
        run_scoring = True
        _conf["scoring_data"] = scoring_data
        _conf["scoring_output"] = scoring_output

    if use_case_name is not None:
        _conf["use_case_name"] = use_case_name
    if active_models is not None:
        _conf["active_models"] = active_models
    if accelerator is not None:
        _conf["accelerator"] = accelerator
    if scoring_model_stage is not None:
        _conf["scoring_model_stage"] = scoring_model_stage
    if selection_metric is not None:
        _conf["selection_metric"] = selection_metric
    if backtest_retrain is not None:
        _conf["backtest_retrain"] = backtest_retrain
    if train_predict_ratio is not None:
        _conf["train_predict_ratio"] = train_predict_ratio
    if experiment_path is not None:
        _conf["experiment_path"] = experiment_path
    if metrics_output is not None:
        _conf["metrics_output"] = metrics_output
    if ensemble is not None:
        _conf["ensemble"] = ensemble
    if ensemble_metric is not None:
        _conf["ensemble_metric"] = ensemble_metric
    if ensemble_metric_avg is not None:
        _conf["ensemble_metric_avg"] = ensemble_metric_avg
    if ensemble_metric_max is not None:
        _conf["ensemble_metric_max"] = ensemble_metric_max
    if ensemble_scoring_output is not None:
        _conf["ensemble_scoring_output"] = ensemble_scoring_output
    if data_quality_check is not None:
        _conf["data_quality_check"] = data_quality_check

    f = Forecaster(conf=_conf, spark=spark)
    run_id = f.train_eval_score(export_metrics=False, scoring=run_scoring)
    return run_id


__all__ = ["run_forecast", "Forecaster"]
