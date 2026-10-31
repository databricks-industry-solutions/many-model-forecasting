import os
import functools
import logging
import pathlib
import posixpath
from datetime import datetime
import uuid
import yaml
from typing import Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType,
    DoubleType,
    TimestampType,
    BinaryType,
    ArrayType,
    IntegerType,
    BooleanType,
)
from pyspark.sql.functions import lit, col, posexplode, to_date, countDistinct
from mmf_sa.models.abstract_model import ForecastingRegressor, collect_sorted
from mmf_sa.models import ModelRegistry
from mmf_sa.data_quality_checks import DataQualityChecks
from mmf_sa.intervals.calibrator import IntervalCalibrator
from mmf_sa.intervals.config import check_supported, resolve_prediction_intervals_conf
from mmf_sa.intervals.quantiles import quantile_columns, quantile_maps
from mmf_sa.exceptions import (
    ConfigurationError,
    ExperimentError,
    ModelError,
    EvaluationError,
    ScoringError,
    ModelRegistryError,
    SparkError,
    DataError,
    UnsupportedFrequencyError
)
_logger = logging.getLogger(__name__)
os.environ['NIXTLA_ID_AS_COL'] = '1'
mlflow.set_registry_uri("databricks-uc")


def ensure_workspace_parent_dir(path: str) -> None:
    """
    Creates the workspace folder that will hold an MLflow experiment, e.g. /Users/me/mmf for /Users/me/mmf/exp.
    Uses the Workspace API (databricks-sdk, a dependency of mlflow), which works on classic and serverless compute;
    os.makedirs on /Workspace does not create workspace folders on serverless. Falls back to os.makedirs.
    """
    parent = posixpath.dirname(path[len("/Workspace"):] if path.startswith("/Workspace/") else path)
    if parent in ("", "/"):
        return
    try:
        from databricks.sdk import WorkspaceClient
        WorkspaceClient().workspace.mkdirs(parent)
        return
    except Exception as err:
        _logger.debug(f"Could not create {parent} with the Workspace API ({err}); trying /Workspace{parent}.")
    try:
        os.makedirs(f"/Workspace{parent}", exist_ok=True)
    except OSError as err:
        _logger.warning(f"Could not create the experiment folder {parent} ({err}); mlflow.set_experiment may fail "
                        "if it does not exist.")


class Forecaster:
    def __init__(
        self,
        conf: Union[str, Dict[str, Any]],
        data_conf: Dict[str, Any],
        spark: SparkSession,
        experiment_id: str = None,
        run_id: str = None,
    ):
        if isinstance(conf, BaseContainer):
            self.conf = conf
        elif isinstance(conf, dict):
            self.conf = OmegaConf.create(conf)
        elif isinstance(conf, str):
            _yaml_conf = yaml.safe_load(pathlib.Path(conf).read_text())
            self.conf = OmegaConf.create(_yaml_conf)
        else:
            raise ConfigurationError("No configuration provided!")
        if run_id:
            self.run_id = run_id
        else:
            self.run_id = str(uuid.uuid4())
        self.data_conf = data_conf
        self.conf["prediction_intervals"] = resolve_prediction_intervals_conf(
            self.conf.get("prediction_intervals"))
        if self.intervals_enabled:
            # Fail before the backtest for settings that are validated but not implemented yet.
            check_supported(self.conf.prediction_intervals)
        if self.intervals_enabled and not self.conf.get("evaluation_output"):
            raise ConfigurationError(
                "prediction_intervals.enabled requires evaluation_output: intervals are "
                "calibrated on the backtest results stored there."
            )
        self.model_registry = ModelRegistry(self.conf)
        self.spark = spark
        self.serverless = bool(self.conf.get("serverless", False))
        if experiment_id:
            self.experiment_id = experiment_id
        elif self.conf.get("experiment_path"):
            self.experiment_id = self.set_mlflow_experiment()
        else:
            raise ExperimentError(
                "Set 'experiment_path' parameter in the configuration file!"
            )
        if self.conf["freq"] == "H":
            self.backtest_offset = pd.DateOffset(hours=self.conf["backtest_length"])
        elif self.conf["freq"] == "D":
            self.backtest_offset = pd.DateOffset(days=self.conf["backtest_length"])
        elif self.conf["freq"] == "W":
            self.backtest_offset = pd.DateOffset(weeks=self.conf["backtest_length"])
        elif self.conf["freq"] == "M":
            self.backtest_offset = pd.DateOffset(months=self.conf["backtest_length"])
        else:
            self.backtest_offset =  None
        self.run_date = datetime.now()
        self.calibrator = IntervalCalibrator(self)

    def set_mlflow_experiment(self):
        """
        Sets the MLflow experiment using the provided experiment path and returns the experiment id.
        Parameters: self (Forecaster): A Forecaster object.
        Returns: experiment_id (str): A string specifying the experiment id.
        """
        ensure_workspace_parent_dir(self.conf["experiment_path"])
        mlflow.set_experiment(self.conf["experiment_path"])
        experiment_id = (
            MlflowClient()
            .get_experiment_by_name(self.conf["experiment_path"])
            .experiment_id
        )
        return experiment_id

    def resolve_source(self, key: str) -> DataFrame:
        """
        Resolve a data source using the provided key and return Spark DataFrame.
        Parameters: self (Forecaster): A Forecaster object. key (str): A string specifying the key.
        Returns: DataFrame: A Spark DataFrame object.
        """
        # A source is a DataFrame in data_conf, or a table name in conf; the two kinds can be mixed
        # (e.g. train_data as a DataFrame and scoring_data as a table).
        df_val = (self.data_conf or {}).get(key)
        if isinstance(df_val, pd.DataFrame):
            return self.spark.createDataFrame(df_val)
        if isinstance(df_val, DataFrame):
            return df_val
        return self.spark.read.table(self.conf[key])

    def data_source(self, key: str):
        """The DataFrame (data_conf) or table name (conf) given for a data source, or None."""
        value = (self.data_conf or {}).get(key)
        return value if value is not None else self.conf.get(key)

    def has_separate_scoring_data(self) -> bool:
        """Whether scoring_data is given and is not the training data itself (same table or same DataFrame)."""
        scoring, train = self.data_source("scoring_data"), self.data_source("train_data")
        if scoring is None:
            return False
        if isinstance(scoring, str) and isinstance(train, str):
            return scoring != train
        if scoring is train:
            return False
        if isinstance(scoring, DataFrame) and isinstance(train, DataFrame):
            return not scoring.sameSemantics(train)
        if isinstance(scoring, pd.DataFrame) and isinstance(train, pd.DataFrame):
            return not scoring.equals(train)
        return True

    def prepare_data_for_global_model(self, mode: str = None):
        """
        Prepares data for a global model by resolving the training data source, performing data quality checks,
        optionally merging scoring data, and converting the DataFrame to a pandas DataFrame.
        Parameters:
            self (Forecaster): A Forecaster object.
            mode (str, optional): A string specifying the mode. Default is None.
        Returns:
            src_df (pd.DataFrame): A pandas DataFrame.
            removed (List[str]): A list of strings specifying the removed groups.
        """
        src_df = self.resolve_source("train_data")
        src_df, removed = DataQualityChecks(src_df, self.conf, self.spark).run()

        # This block runs when preparing data for scoring
        if (mode == "scoring") and self.has_separate_scoring_data():
            score_df = self.resolve_source("scoring_data")
            score_df = score_df.where(~col(self.conf["group_id"]).isin(removed))
            src_df = src_df.unionByName(score_df, allowMissingColumns=True)

        src_df = src_df.toPandas()
        return src_df, removed

    def split_df_train_val(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits df into train and validation data, based on backtest months and prediction length.
        Data before backtest will be "train", and data after the backtest (at most prediction length days) will be "val".
        Parameters:
            self (Forecaster): A Forecaster object.
            df (pd.DataFrame): A pandas DataFrame.
        Returns:
            train_df (pd.DataFrame): A pandas DataFrame.
            val_df (pd.DataFrame): A pandas DataFrame.
        """
        # Train with data before the backtest months in conf
        train_df = df[
            df[self.conf["date_col"]]
            <= df[self.conf["date_col"]].max() - self.backtest_offset
        ]
        # Validate with data after the backtest months cutoff...
        val_df = df[
            df[self.conf["date_col"]]
            > df[self.conf["date_col"]].max() - self.backtest_offset
        ]
        return train_df, val_df

    def evaluate_score(
            self, evaluate: bool = True, score: bool = True) -> str:
        """
        Evaluates and scores the models using the provided configuration.
        Parameters:
            self (Forecaster): A Forecaster object.
            evaluate (bool, optional): A boolean specifying whether to evaluate the models. Default is True.
            score (bool, optional): A boolean specifying whether to score the models. Default is True.
        Returns: run_id (str): A string specifying the run id.
        """
        print("Run quality checks")
        src_df = self.resolve_source("train_data")
        clean_df, removed = DataQualityChecks(src_df, self.conf, self.spark).run(verbose=True)
        print("Finished quality checks")
        print("Starting evaluate_score")
        failed_models = set()
        if evaluate:
            failed_models = self.evaluate_models()
        if evaluate and self.intervals_enabled:
            # Without evaluation, scoring reuses the latest calibration stored for this run_id.
            self.run_calibration()
        if score:
            self.score_models(failed_models)
        print("Finished evaluate_score")
        return self.run_id

    @property
    def intervals_enabled(self) -> bool:
        return bool(self.conf.prediction_intervals.enabled)

    def native_quantiles(self, model_conf) -> list:
        """Delegates to IntervalCalibrator.native_quantiles."""
        return self.calibrator.native_quantiles(model_conf)

    def get_model_with_native_quantiles(self, model_conf) -> ForecastingRegressor:
        """Delegates to IntervalCalibrator.get_model_with_native_quantiles."""
        return self.calibrator.get_model_with_native_quantiles(model_conf)

    def evaluation_writer(self, sdf: DataFrame):
        """Delegates to IntervalCalibrator.evaluation_writer."""
        return self.calibrator.evaluation_writer(sdf)

    def run_calibration(self):
        """Delegates to IntervalCalibrator.run_calibration."""
        self.calibrator.run_calibration()

    def evaluate_models(self) -> set:
        """
        Trains and evaluates all models from the active models list.
        Parameters: self (Forecaster): A Forecaster object.
        Returns: failed_models (set): A set of model names that failed evaluation.
            Only global and foundation models are tracked because they register
            to the model registry during evaluation; scoring would fail if the
            registered model does not exist.
        """
        print("Starting evaluate_models")
        failed_models = set()
        for model_name in self.model_registry.get_active_model_keys():
            print(f"Started evaluating {model_name}")
            try:
                model_conf = self.model_registry.get_model_conf(model_name)
                if model_conf["model_type"] == "local":
                    self.evaluate_local_model(model_conf)
                elif model_conf["model_type"] == "global":
                    self.evaluate_global_model(model_conf)
                elif model_conf["model_type"] == "foundation":
                    self.evaluate_foundation_model(model_conf)
            except (ModelError, EvaluationError, DataError, ConfigurationError) as err:
                _logger.error(
                    f"Error has occurred while evaluating model {model_name}: {repr(err)}",
                    exc_info=err,
                )
                if model_conf["model_type"] in ("global", "foundation"):
                    failed_models.add(model_name)
            except Exception as err:
                _logger.error(
                    f"Unexpected error while evaluating model {model_name}: {repr(err)}",
                    exc_info=err,
                )
                if model_conf["model_type"] in ("global", "foundation"):
                    failed_models.add(model_name)
            print(f"Finished evaluating {model_name}")
        print("Finished evaluate_models")
        return failed_models

    def evaluate_local_model(self, model_conf):
        """
        Evaluates a local model using the provided model configuration. It applies the Pandas UDF to the training data.
        It then logs the aggregated metrics and a few tags to MLflow.
        Parameters:
            self (Forecaster): A Forecaster object.
            model_conf (dict): A dictionary specifying the model configuration.
        """
        with mlflow.start_run(experiment_id=self.experiment_id):
            src_df = self.resolve_source("train_data")
            src_df, removed = DataQualityChecks(src_df, self.conf, self.spark).run()

            # Specifying the output schema for Pandas UDF
            output_schema = StructType(
                [
                    StructField(
                        self.conf["group_id"], src_df.schema[self.conf["group_id"]].dataType
                    ),
                    StructField("backtest_window_start_date", TimestampType()),
                    StructField("metric_name", StringType()),
                    StructField("metric_value", DoubleType()),
                    StructField("forecast", ArrayType(DoubleType())),
                    StructField("actual", ArrayType(DoubleType())),
                    StructField("model_pickle", BinaryType()),
                ]
                + self.calibrator.evaluation_quantile_fields()
            )

            model = self.get_model_with_native_quantiles(model_conf)

            # Use Pandas UDF to forecast individual group
            evaluate_one_local_model_fn = functools.partial(
                Forecaster.evaluate_one_local_model, model=model,
                with_quantiles=self.intervals_enabled,
            )

            res_sdf = (
                src_df.groupby(self.conf["group_id"])
                .applyInPandas(evaluate_one_local_model_fn, schema=output_schema)
            )
        
            # Write evaluation result to a delta table
            if self.conf.get("evaluation_output", None) is not None:
                self.evaluation_writer(
                    res_sdf.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
                    .withColumn("run_id", lit(self.run_id))
                    .withColumn("run_date", lit(self.run_date))
                    .withColumn("model", lit(model_conf["name"]))
                    .withColumn("use_case", lit(self.conf["use_case_name"]))
                    .withColumn("model_uri", lit(""))
                ).saveAsTable(self.conf.get("evaluation_output"))

            # Compute aggregated metrics
            res_df = (
                res_sdf.groupby(["metric_name"])
                .mean("metric_value")
                .withColumnRenamed("avg(metric_value)", "metric_value")
                .toPandas()
            )
            # Print out aggregated metrics
            print(res_df)

            # Log aggregated metrics to MLflow
            for rec in res_df.values:
                metric_name, metric_value = rec
                mlflow.log_metric(metric_name, metric_value)
                mlflow.set_tag("model_name", model_conf["name"])
                mlflow.set_tag("run_id", self.run_id)

    @staticmethod
    def evaluate_one_local_model(
        pdf: pd.DataFrame, model: ForecastingRegressor, with_quantiles: bool = False
    ) -> pd.DataFrame:
        """
        A static method that evaluates a single local model using the provided pandas DataFrame and model.
        If the evaluation for a single group fails, it returns an empty DataFrame without failing the entire process.
        Parameters:
            pdf (pd.DataFrame): A pandas DataFrame.
            model (ForecastingRegressor): A ForecastingRegressor object.
            with_quantiles (bool): Whether to return the forecast_quantiles column (prediction intervals enabled).
        Returns: metrics_df (pd.DataFrame): A pandas DataFrame.
        """
        pdf[model.params["date_col"]] = pd.to_datetime(pdf[model.params["date_col"]])
        pdf.sort_values(by=model.params["date_col"], inplace=True)
        if model.params["freq"] == "H":
            backtest_offset = pd.DateOffset(hours=model.params["backtest_length"])
        elif model.params["freq"] == "D":
            backtest_offset = pd.DateOffset(days=model.params["backtest_length"])
        elif model.params["freq"] == "W":
            backtest_offset = pd.DateOffset(weeks=model.params["backtest_length"])
        elif model.params["freq"] == "M":
            backtest_offset = pd.DateOffset(months=model.params["backtest_length"])
        else:
            backtest_offset = None
        split_date = pdf[model.params["date_col"]].max() - backtest_offset
        group_id = pdf[model.params["group_id"]].iloc[0]
        empty_result = pd.DataFrame(
            columns=[
                model.params["group_id"],
                "backtest_window_start_date",
                "metric_name",
                "metric_value",
                "forecast",
                "actual",
                "model_pickle",
            ] + (["forecast_quantiles"] if with_quantiles else [])
        )
        try:
            pdf = pdf.fillna(0)
            pdf[model.params["target"]] = pdf[model.params["target"]].clip(0)
            metrics_df = model.backtest(pdf, start=split_date, group_id=group_id)
            if not with_quantiles:
                metrics_df = metrics_df.drop(columns="forecast_quantiles")
            return metrics_df
        except (ModelError, EvaluationError, DataError) as err:
            _logger.error(
                f"Error evaluating group {group_id} using model {repr(model)}: {err}",
                exc_info=err,
                stack_info=True,
            )
            return empty_result
        except Exception as err:
            _logger.error(
                f"Unexpected error evaluating group {group_id} using model {repr(model)}: {err}",
                exc_info=err,
                stack_info=True,
            )
            return empty_result

    def evaluate_global_model(self, model_conf):
        """
        Evaluate a global model using the provided model configuration. Trains and registers the model.
        Parameters:
            self (Forecaster): A Forecaster object.
            model_conf (dict): A dictionary specifying the model configuration.
        """
        mlflow.autolog(disable=True)
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            model_name = model_conf["name"]
            hist_df, removed = self.prepare_data_for_global_model("evaluating")
            train_df, val_df = self.split_df_train_val(hist_df)

            # First, we train the model on the entire history (train_df, val_df).
            # Then, we register this model as our final model in Unity Catalog.
            final_model = self.model_registry.get_model(model_name)
            final_model.fit(pd.concat([train_df, val_df]))
            input_example = train_df[train_df[self.conf['group_id']] == train_df[self.conf['group_id']] \
                        .unique()[0]].sort_values(by=[self.conf['date_col']])
            model_info = final_model.register(
                model=final_model,
                registered_model_name=f"{self.conf['model_output']}.{model_conf['name']}_{self.conf['use_case_name']}",
                input_example=input_example,
            )
            # Next, we train the model only with train_df and run detailed backtesting
            model = self.model_registry.get_model(model_name)
            model.fit(pd.concat([train_df]))
            metrics = self.backtest_global_model(
                model=model,
                train_df=train_df,
                val_df=val_df,
                model_uri=model_info.model_uri,  # This model_uri is from the final model
                write=True,
            )
            mlflow.set_tag("run_id", self.run_id)
            mlflow.set_tag("model_name", model.params["name"])

    def backtest_global_model(
        self,
        model: ForecastingRegressor,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_uri: str,
        write: bool = True,
        serverless_predict: bool = False,
    ):
        """
        Performs detailed backtesting of a global model using the provided model, training DataFrame,
        validation DataFrame, model URI, and write parameter.
        Parameters:
            self (Forecaster): A Forecaster object.
            model (ForecastingRegressor): A ForecastingRegressor object.
            train_df (pd.DataFrame): A pandas DataFrame.
            val_df (pd.DataFrame): A pandas DataFrame.
            model_uri (str): A string specifying the model URI.
            write (bool, optional): A boolean specifying whether to write the results to a table. Default is True.
            serverless_predict (bool, optional): When True, dispatch model.backtest with spark=None so the
                model's driver-only predict path is used. There is no multi-GPU data parallelism across Spark workers. 
                Required on serverless GPU compute, where Pandas UDF workers are CPU-only. Default is False.
        Returns: metric_value (float): A float specifying the mean metric value.
        """
        model_spark = None if serverless_predict else self.spark
        res_pdf = (
            model.backtest(
                pd.concat([train_df, val_df]),
                start=train_df[self.conf["date_col"]].max(),
                spark=model_spark,
                # backtest_retrain=self.conf["backtest_retrain"],
            ))
        group_id_dtype = IntegerType() \
            if train_df[self.conf["group_id"]].dtype == 'int' else StringType()

        schema = StructType(
            [
                StructField(self.conf["group_id"], group_id_dtype),
                StructField("backtest_window_start_date", TimestampType()),
                StructField("metric_name", StringType()),
                StructField("metric_value", DoubleType()),
                StructField("forecast", ArrayType(DoubleType())),
                StructField("actual", ArrayType(DoubleType())),
                StructField("model_pickle", BinaryType()),
            ]
            + self.calibrator.evaluation_quantile_fields()
        )
        # Covert to Python-native types before converting to pyspark dataframe
        res_pdf['forecast'] = res_pdf['forecast'].apply(lambda x: [float(i) for i in x])
        res_pdf['actual'] = res_pdf['actual'].apply(lambda x: [float(i) for i in x])
        if self.intervals_enabled:
            res_pdf['forecast_quantiles'] = res_pdf['forecast_quantiles'].apply(
                lambda m: None if m is None else {k: [float(i) for i in v] for k, v in m.items()})
        else:
            res_pdf = res_pdf.drop(columns="forecast_quantiles")
        res_sdf = self.spark.createDataFrame(res_pdf, schema)
        # Write evaluation results to a delta table
        if write:
            if self.conf.get("evaluation_output", None):
                self.evaluation_writer(
                    res_sdf.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
                    .withColumn("run_id", lit(self.run_id))
                    .withColumn("run_date", lit(self.run_date))
                    .withColumn("model", lit(model.params.name))
                    .withColumn("use_case", lit(self.conf["use_case_name"]))
                    .withColumn("model_uri", lit(model_uri))
                ).saveAsTable(self.conf.get("evaluation_output"))
        # Compute aggregated metrics
        res_df = (
            res_sdf.groupby(["metric_name"])
            .mean("metric_value")
            .withColumnRenamed("avg(metric_value)", "metric_value")
            .toPandas()
        )
        metric_name = None
        metric_value = None

        # Log metrics to MLFlow
        for rec in res_df.values:
            metric_name, metric_value = rec
            if write:
                mlflow.log_metric(metric_name, metric_value)
                print(res_df)
        return metric_value

    def evaluate_foundation_model(self, model_conf):
        """
        Evaluates a foundation model using the provided model configuration. Registers the model.
        Parameters:
            self (Forecaster): A Forecaster object.
            model_conf (dict): A dictionary specifying the model configuration.
        """
        if self.serverless and model_conf.get("framework") == "Moirai":
            raise ModelError(
                "Moirai does not support serverless=True because it has no driver-only "
                "single-GPU predict path. Use Chronos, TimesFM or FoundationForecast models on serverless GPU, "
                "or run Moirai on a classic GPU cluster."
            )
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            model_name = model_conf["name"]
            model = self.get_model_with_native_quantiles(model_conf)
            hist_df, removed = self.prepare_data_for_global_model("evaluating")  # Reuse the same as global
            train_df, val_df = self.split_df_train_val(hist_df)
            input_example = train_df[train_df[self.conf['group_id']] == train_df[self.conf['group_id']] \
                .unique()[0]].sort_values(by=[self.conf['date_col']])
            registered_model_name = f"{self.conf['model_output']}.{model_conf['name']}_{self.conf['use_case_name']}"
            if model_conf["framework"] == "FoundationForecast":
                model.register(registered_model_name=registered_model_name, input_example=input_example)
            elif model_conf["framework"] in ["Chronos", "Moirai", "TimesFM"]:
                model.register(registered_model_name=registered_model_name)
            model_uri = f"runs:/{run.info.run_id}/model"
            metrics = self.backtest_global_model(  # Reuse the same as global
                model=model,
                train_df=train_df,
                val_df=val_df,
                model_uri=model_uri,
                write=True,
                serverless_predict=self.serverless,
            )
            mlflow.log_metric(self.conf["metric"], metrics)
            mlflow.set_tag("model_name", model.params["name"])
            mlflow.set_tag("run_id", self.run_id)
            mlflow.log_params(model.get_params())

    def score_models(self, failed_models: set = None):
        """
        Scores the models using the provided model configuration.
        Models whose evaluation failed are skipped to avoid "resource does not
        exist" errors when retrieving from the model registry.
        Parameters:
            self (Forecaster): A Forecaster object.
            failed_models (set, optional): Model names that failed evaluation and
                should be skipped during scoring. Default is None.
        """
        print("Starting run_scoring")
        if failed_models is None:
            failed_models = set()
        for model_name in self.model_registry.get_active_model_keys():
            if model_name in failed_models:
                _logger.warning(f"Skipping scoring for {model_name}: evaluation failed")
                print(f"Skipping scoring for {model_name}: evaluation failed")
                continue
            model_conf = self.model_registry.get_model_conf(model_name)
            print(f"Started scoring with {model_name}")
            if model_conf["model_type"] == "global":
                self.score_global_model(model_conf)
            elif model_conf["model_type"] == "local":
                self.score_local_model(model_conf)
            elif model_conf["model_type"] == "foundation":
                self.score_foundation_model(model_conf)
            print(f"Finished scoring with {model_name}")
        print("Finished run_scoring")

    def score_local_model(self, model_conf):
        """
        Scores a local model using the provided model configuration and writes the results to a delta table.
        Parameters:
            self (Forecaster): A Forecaster object.
            model_conf (dict): A dictionary specifying the model configuration.
        """
        src_df = self.resolve_source("train_data")
        src_df, removed = DataQualityChecks(src_df, self.conf, self.spark).run()
        # Check if external regressors are provided and if the framework is statsforecast.
        # External regressors are supported only with statsforecast models.
        if self.has_separate_scoring_data() and (model_conf["framework"] == "StatsForecast"):
            score_df = self.resolve_source("scoring_data")
            score_df = score_df.where(~col(self.conf["group_id"]).isin(removed))
            src_df = src_df.unionByName(score_df, allowMissingColumns=True)

        # Specify output schema for Pandas UDF
        output_schema = StructType(
            [
                StructField(
                    self.conf["group_id"], src_df.schema[self.conf["group_id"]].dataType
                ),
                StructField(self.conf["date_col"], ArrayType(TimestampType())),
                StructField(self.conf["target"], ArrayType(DoubleType())),
                StructField("model_pickle", BinaryType()),
            ]
            + self.calibrator.evaluation_quantile_fields()
        )
        model = self.get_model_with_native_quantiles(model_conf)

        # Use Pandas UDF to distribute scoring
        score_one_local_model_fn = functools.partial(
            Forecaster.score_one_local_model, model=model, with_quantiles=self.intervals_enabled)
        res_sdf = (
            src_df.groupby(self.conf["group_id"])
            .applyInPandas(score_one_local_model_fn, schema=output_schema)
        )

        # Write the results to a delta table
        self.write_scoring_output(
            res_sdf.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
            .withColumn("run_id", lit(self.run_id))
            .withColumn("run_date", lit(self.run_date))
            .withColumn("use_case", lit(self.conf["use_case_name"]))
            .withColumn("model", lit(model_conf["name"]))
            .withColumn("model_uri", lit("")),
            model_conf["name"],
        )

    @staticmethod
    def score_one_local_model(
        pdf: pd.DataFrame, model: ForecastingRegressor, with_quantiles: bool = False
    ) -> pd.DataFrame:
        """
        A static method that scores a single local model using the provided pandas DataFrame and model.
        If the scoring for one time series fails, it returns an empty DataFrame instead of failing the
        entire process.
        Parameters:
            pdf (pd.DataFrame): A pandas DataFrame.
            model (ForecastingRegressor): A ForecastingRegressor object.
            with_quantiles (bool): Whether to return the forecast_quantiles column (prediction intervals enabled).
        Returns: res_df (pd.DataFrame): A pandas DataFrame.
        """
        pdf[model.params["date_col"]] = pd.to_datetime(pdf[model.params["date_col"]])
        pdf.sort_values(by=model.params["date_col"], inplace=True)
        group_id = pdf[model.params["group_id"]].iloc[0]
        res_df, model_fitted = model.forecast(pdf)
        try:
            data = [
                group_id,
                res_df[model.params["date_col"]].to_numpy(),
                res_df[model.params["target"]].to_numpy(),
                cloudpickle.dumps(model_fitted)]
            if with_quantiles:
                data.append(ForecastingRegressor.forecast_quantiles(res_df))
        except (ModelError, ScoringError, DataError) as err:
            _logger.error(
                f"Error scoring group {group_id} using model {repr(model)}: {err}",
                exc_info=err,
                stack_info=True,
            )
            data = [group_id, None, None, None] + ([None] if with_quantiles else [])
        except Exception as err:
            _logger.error(
                f"Unexpected error scoring group {group_id} using model {repr(model)}: {err}",
                exc_info=err,
                stack_info=True,
            )
            data = [group_id, None, None, None] + ([None] if with_quantiles else [])
        res_df = pd.DataFrame(
            columns=[
                model.params["group_id"],
                model.params["date_col"],
                model.params["target"],
                "model_pickle"] + (["forecast_quantiles"] if with_quantiles else []), data=[data]
        )
        return res_df

    def score_global_model(self, model_conf):
        print(f"Running scoring for {model_conf['name']}...")
        model, model_uri = self.get_model_for_scoring(model_conf)
        score_df, removed = self.prepare_data_for_global_model("scoring")
        prediction_df = model.predict(score_df)
        if prediction_df[self.conf["date_col"]].dtype.type != np.datetime64:
            prediction_df[self.conf["date_col"]] = prediction_df[
                self.conf["date_col"]
            ].dt.to_timestamp()
        sdf = self.spark.createDataFrame(prediction_df)\
            .drop('index')\
            .withColumn(self.conf["target"], col(self.conf["target"]).cast(DoubleType()))
        sdf = collect_sorted(sdf, self.conf["group_id"], self.conf["date_col"], [self.conf["target"]])
        # Defensive: align with score_local_model's explicit ArrayType(TimestampType())
        # so the scoring_output table schema stays consistent across all writers,
        # even if a future global model wrapper changes its predict() return shape.
        sdf = sdf.withColumn(
            self.conf["date_col"],
            sdf[self.conf["date_col"]].cast(ArrayType(TimestampType())),
        )
        self.write_scoring_output(
            sdf.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
            .withColumn("model", lit(model_conf["name"]))
            .withColumn("run_id", lit(self.run_id))
            .withColumn("run_date", lit(self.run_date))
            .withColumn("use_case", lit(self.conf["use_case_name"]))
            .withColumn("model_pickle", lit(b""))
            .withColumn("model_uri", lit(model_uri)),
            model_conf["name"],
        )

    def score_foundation_model(self, model_conf):
        """
        Scores a global model using the provided model configuration. Writes the results to a delta table.
        Parameters:
            self (Forecaster): A Forecaster object.
            model_conf (dict): A dictionary specifying the model configuration.
        """
        if self.serverless and model_conf.get("framework") == "Moirai":
            raise ModelError(
                "Moirai does not support serverless=True because it has no driver-only "
                "single-GPU predict path. Use Chronos, TimesFM or FoundationForecast models on serverless GPU, "
                "or run Moirai on a classic GPU cluster."
            )
        print(f"Running scoring for {model_conf['name']}...")
        model_name = model_conf["name"]
        _, model_uri = self.get_model_for_scoring(model_conf)
        model = self.get_model_with_native_quantiles(model_conf)
        score_df, removed = self.prepare_data_for_global_model("scoring")
        forecast_spark = None if self.serverless else self.spark
        prediction_df, model_pretrained = model.forecast(score_df, spark=forecast_spark)
        q_cols = list(quantile_columns(prediction_df).values())
        if self.intervals_enabled:
            maps = quantile_maps(prediction_df, self.conf["group_id"])
            prediction_df = prediction_df.drop(columns=q_cols)
            prediction_df["forecast_quantiles"] = [maps.get(k) for k in prediction_df[self.conf["group_id"]]]
            prediction_df[self.conf["group_id"]] = prediction_df[self.conf["group_id"]].astype(str)
        elif q_cols:
            prediction_df = prediction_df.drop(columns=q_cols)
        # Driver-only single-GPU predict paths build ds arrays as numpy
        # datetime64[ns]; Spark Connect (serverless) Arrow rejects nanosecond
        # timestamps. Coerce to microsecond precision before createDataFrame.
        date_col = self.conf["date_col"]
        if date_col in prediction_df.columns and len(prediction_df) > 0:
            sample = prediction_df[date_col].iloc[0]
            if isinstance(sample, np.ndarray) and np.issubdtype(sample.dtype, np.datetime64):
                prediction_df = prediction_df.copy()
                prediction_df[date_col] = prediction_df[date_col].apply(
                    lambda arr: np.asarray(arr, dtype="datetime64[us]")
                )
        sdf = self.spark.createDataFrame(
            prediction_df,
            schema=self.calibrator.foundation_scoring_schema(prediction_df) if self.intervals_enabled else None,
        ).drop('index')
        # Normalize the date column to TIMESTAMP (LTZ) so the schema matches the
        # table created by score_local_model (which uses ArrayType(TimestampType())).
        # No-op on classic compute (already LTZ); applies NTZ -> LTZ on Spark Connect,
        # where Arrow inference of object-dtype columns containing numpy datetime64
        # arrays produces array<timestamp_ntz> and would otherwise fail Delta merge.
        sdf = sdf.withColumn(date_col, sdf[date_col].cast(ArrayType(TimestampType())))
        self.write_scoring_output(
            sdf.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
            .withColumn("model", lit(model_conf["name"]))
            .withColumn("run_id", lit(self.run_id))
            .withColumn("run_date", lit(self.run_date))
            .withColumn("use_case", lit(self.conf["use_case_name"]))
            .withColumn("model_pickle", lit(b""))
            .withColumn("model_uri", lit(model_uri)),
            model_conf["name"],
        )

    def write_scoring_output(self, sdf: DataFrame, model_name: str):
        """
        Appends one model's forecasts to scoring_output. With prediction intervals enabled, the calibration
        of this run is applied first and the interval columns are added (output_format).
        """
        if not self.intervals_enabled:
            sdf.write.mode("append").saveAsTable(self.conf["scoring_output"])
            return
        # The forecasts are used twice (missing-calibration count, then the write): compute them once.
        materializer = self.calibrator.materializer()
        sdf = materializer.cache(sdf)
        try:
            (
                self.calibrator.add_prediction_intervals(sdf, model_name)
                .write.mode("append").option("mergeSchema", "true")
                .saveAsTable(self.conf["scoring_output"])
            )
        finally:
            materializer.close()

    def calibration_rows_for(self, model_name: str):
        """Delegates to IntervalCalibrator.calibration_rows_for."""
        return self.calibrator.calibration_rows_for(model_name)

    def wide_corrections(self, cal: DataFrame, levels: list) -> DataFrame:
        """Delegates to IntervalCalibrator.wide_corrections."""
        return self.calibrator.wide_corrections(cal, levels)

    def get_model_for_scoring(self, model_conf):
        """
        Gets a model for scoring using the provided model configuration.
        Parameters:
            self (Forecaster): A Forecaster object.
            model_conf (dict): A dictionary specifying the model configuration.
        Returns: model (Model): A model object. model_uri (str): A string specifying the model URI.
        """
        client = MlflowClient()
        registered_name = f"{self.conf['model_output']}.{model_conf['name']}_{self.conf['use_case_name']}"
        model_info = self.get_latest_model_info(client, registered_name)
        model_version = model_info.version
        model_uri = f"runs:/{model_info.run_id}/model"
        if model_conf.get("model_type", None) == "global":
            model = mlflow.pyfunc.load_model(f"models:/{registered_name}/{model_version}")
            return model, model_uri
        elif model_conf.get("model_type", None) == "foundation":
            return None, model_uri
        else:
            return self.model_registry.get_model(model_conf["name"]), None

    @staticmethod
    def get_latest_model_info(client, registered_name):
        """
         Gets the latest model info using the provided MLflow client and registered name.
        Parameters:
            client (MlflowClient): An MLflowClient object.
            registered_name (str): A string specifying the registered name.
        Returns: model_info (ModelVersion): A ModelVersion object.
        """
        latest_version = 1
        for mv in client.search_model_versions(f"name='{registered_name}'"):
            version_int = int(mv.version)
            if version_int > latest_version:
                latest_version = version_int
        return client.get_model_version(registered_name, str(latest_version))
