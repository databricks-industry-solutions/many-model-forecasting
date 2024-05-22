import os
import functools
import logging
import pathlib
from datetime import datetime
import uuid
import yaml
from typing import Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
import cloudpickle
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from omegaconf import OmegaConf, DictConfig
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
)
from pyspark.sql.functions import lit, avg, min, max, col, posexplode, collect_list, to_date
from mmf_sa.models.abstract_model import ForecastingRegressor
from mmf_sa.models import ModelRegistry
from mmf_sa.data_quality_checks import DataQualityChecks
_logger = logging.getLogger(__name__)
os.environ['NIXTLA_ID_AS_COL'] = '1'
mlflow.set_registry_uri("databricks-uc")


class Forecaster:
    def __init__(
        self,
        conf: Union[str, Dict[str, Any]],
        data_conf: Dict[str, Any],
        spark: SparkSession,
        experiment_id: str = None,
    ):
        if isinstance(conf, BaseContainer):
            self.conf = conf
        elif isinstance(conf, dict):
            self.conf = OmegaConf.create(conf)
        elif isinstance(conf, str):
            _yaml_conf = yaml.safe_load(pathlib.Path(conf).read_text())
            self.conf = OmegaConf.create(_yaml_conf)
        else:
            raise Exception("No configuration provided!")

        self.run_id = str(uuid.uuid4())
        self.data_conf = data_conf
        self.model_registry = ModelRegistry(self.conf)
        self.spark = spark
        if experiment_id:
            self.experiment_id = experiment_id
        elif self.conf.get("experiment_path"):
            self.experiment_id = self.set_mlflow_experiment()
        else:
            raise Exception(
                "Please set 'experiment_path' parameter in the configuration file!"
            )
        self.selection_metric = self.conf["selection_metric"]
        self.run_date = datetime.now()

    def set_mlflow_experiment(self):
        mlflow.set_experiment(self.conf["experiment_path"])
        experiment_id = (
            MlflowClient()
            .get_experiment_by_name(self.conf["experiment_path"])
            .experiment_id
        )
        return experiment_id

    def resolve_source(self, key: str) -> DataFrame:
        if self.data_conf:
            df_val = self.data_conf.get(key)
            if df_val is not None and isinstance(df_val, pd.DataFrame):
                return self.spark.createDataFrame(df_val)
            elif df_val is not None and isinstance(df_val, DataFrame):
                return df_val
        else:
            return self.spark.read.table(self.conf[key])

    def train_eval_score(self, export_metrics=False, scoring=True) -> str:
        print("Starting train_evaluate_models")
        self.train_models()
        self.evaluate_models()
        if scoring:
            self.score_models()
            self.ensemble()
        if export_metrics:
            self.update_metrics()
        print("Finished train_evaluate_models")
        return self.run_id

    def ensemble(self):
        if self.conf.get("ensemble") and self.conf["ensemble_scoring_output"]:
            metrics_df = (
                self.spark.table(self.conf["evaluation_output"])
                .where(col("run_id").eqNullSafe(lit(self.run_id)))
                .where(
                    col("metric_name").eqNullSafe(
                        lit(self.conf.get("ensemble_metric", "smape"))
                    )
                )
            )
            models_df = (
                metrics_df.groupby(self.conf["group_id"], "model")
                .agg(
                    avg("metric_value").alias("metric_avg"),
                    min("metric_value").alias("metric_min"),
                    max("metric_value").alias("metric_max"),
                )
                .where(
                    col("metric_avg") < lit(self.conf.get("ensemble_metric_avg", 0.2))
                )
                .where(
                    col("metric_max") < lit(self.conf.get("ensemble_metric_max", 0.5))
                )
                .where(col("metric_min") > lit(0.01))
            )
            df = (
                self.spark.table(self.conf["scoring_output"])
                .where(col("run_id").eqNullSafe(lit(self.run_id)))
                .join(
                    models_df.select(self.conf["group_id"], "model"),
                    on=[self.conf["group_id"], "model"],
                )
            )

            left = df.select(
                self.conf["group_id"], "run_id", "run_date", "use_case", "model",
                posexplode(self.conf["date_col"])
            ).withColumnRenamed('col', self.conf["date_col"])

            right = df.select(
                self.conf["group_id"], "run_id", "run_date", "use_case", "model",
                posexplode(self.conf["target"])
            ).withColumnRenamed('col', self.conf["target"])

            merged = left.join(right, [
                self.conf["group_id"], 'run_id', 'run_date', 'use_case', 'model',
                'pos'], 'inner').drop("pos")

            aggregated_df = merged.groupby(
                self.conf["group_id"], self.conf["date_col"]
            ).agg(
                avg(self.conf["target"]).alias(self.conf["target"] + "_avg"),
                min(self.conf["target"]).alias(self.conf["target"] + "_min"),
                max(self.conf["target"]).alias(self.conf["target"] + "_max"),
            )

            aggregated_df = aggregated_df.orderBy(self.conf["group_id"], self.conf["date_col"])\
                .groupBy(self.conf["group_id"]).agg(
                collect_list(self.conf["date_col"]).alias(self.conf["date_col"]),
                collect_list(self.conf["target"] + "_avg").alias(self.conf["target"] + "_avg"),
                collect_list(self.conf["target"] + "_min").alias(self.conf["target"] + "_min"),
                collect_list(self.conf["target"] + "_max").alias(self.conf["target"] + "_max")
            )

            (
                aggregated_df.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
                .withColumn("run_id", lit(self.run_id))
                .withColumn("run_date", lit(self.run_date))
                .withColumn("use_case", lit(self.conf["use_case_name"]))
                .withColumn("model", lit("ensemble"))
                .write.format("delta")
                .mode("append")
                .saveAsTable(self.conf["ensemble_scoring_output"])
            )

    def prepare_data_for_global_model(self, mode: str):
        src_df = self.resolve_source("train_data")
        src_df, removed = DataQualityChecks(src_df, self.conf, self.spark).run()
        if (mode == "scoring") \
                and (self.conf["scoring_data"]) \
                and (self.conf["scoring_data"] != self.conf["train_data"]):
            score_df = self.resolve_source("scoring_data")
            score_df = score_df.where(~col(self.conf["group_id"]).isin(removed))
            src_df = src_df.unionByName(score_df, allowMissingColumns=True)
        src_df = src_df.toPandas()
        return src_df, removed

    def split_df_train_val(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits df into train and data, based on backtest months and prediction length.
        Data before backtest will be train, and data from backtest (at most prediction length days) will be val data."""
        # Train with data before the backtest months in conf
        train_df = df[
            df[self.conf["date_col"]]
            <= df[self.conf["date_col"]].max()
            - pd.DateOffset(months=self.conf["backtest_months"])
        ]
        # Validate with data after the backtest months cutoff...
        val_true_df = df[
            df[self.conf["date_col"]]
            > df[self.conf["date_col"]].max()
            - pd.DateOffset(months=self.conf["backtest_months"])
        ]
        return train_df, val_true_df

    def train_models(self):
        """Trains and evaluates all models from the configuration file with the configuration's training data.
        Then evaluates the current best model with the configuration's training data.
        Saves the params, models and metrics, so the runs will all have evaluation data
        """
        print("Starting train_models")
        for model_name in self.model_registry.get_active_model_keys():
            model_conf = self.model_registry.get_model_conf(model_name)
            if (
                model_conf.get("trainable", False)
                and model_conf.get("model_type", None) == "global"
            ):
                with mlflow.start_run(experiment_id=self.experiment_id) as run:
                    try:
                        model = self.model_registry.get_model(model_name)
                        # Get training and scoring data
                        hist_df, removed = self.prepare_data_for_global_model("training")
                        train_df, val_train_df = self.split_df_train_val(hist_df)
                        # Train and evaluate new models - results are saved to MLFlow
                        print(f"Training model: {model}")
                        self.train_global_model(train_df, val_train_df, model_conf, model)
                    except Exception as err:
                        _logger.error(
                            f"Error has occurred while training model {model_name}: {repr(err)}",
                            exc_info=err,
                        )
                        raise err
        print("Finished train_models")

    def train_global_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_conf: DictConfig,
        model: ForecastingRegressor,
    ):
        print(f"Started training {model_conf['name']}")
        model.fit(pd.concat([train_df, val_df]))
        # TODO fix
        signature = infer_signature(
            model_input=train_df,
            model_output=train_df,
        )
        input_example = train_df
        model_info = mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=f"{self.conf['model_output']}.{model_conf['name']}_{self.conf['use_case_name']}",
            input_example=input_example,
            signature=signature,
            pip_requirements=[],
        )
        print(f"Model registered: {self.conf['model_output']}.{model_conf['name']}_{self.conf['use_case_name']}")
        try:
            mlflow.log_params(model.get_params())
        except MlflowException:
            # MLflow log_params has a parameter length limit of 500
            # When using ensemble models parameters consist of
            # nested parameter dictionaries which are flattened here
            mlflow.log_params(
                flatten_nested_parameters(OmegaConf.to_object(model.get_params()))
            )
        metrics = self.backtest_global_model(
            model=model,
            train_df=train_df,
            val_df=val_df,
            model_uri=model_info.model_uri,
            write=True,
        )
        mlflow.set_tag("action", "train")
        mlflow.set_tag("candidate", "true")
        mlflow.set_tag("model_name", model.params["name"])
        print(f"Finished training {model_conf.get('name')}")

    def backtest_global_model(
        self,
        model: ForecastingRegressor,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_uri: str,
        write: bool = True,
    ):
        res_pdf = (
            model.backtest(
                pd.concat([train_df, val_df]),
                start=train_df[self.conf["date_col"]].max(),
                retrain=self.conf["backtest_retrain"],
            ))

        group_id_dtype = IntegerType() \
            if train_df[self.conf["group_id"]].dtype == 'int' else StringType()

        schema = StructType(
            [
                StructField(self.conf["group_id"], group_id_dtype),
                StructField("backtest_window_start_date", DateType()),
                StructField("metric_name", StringType()),
                StructField("metric_value", DoubleType()),
                StructField("forecast", ArrayType(DoubleType())),
                StructField("actual", ArrayType(DoubleType())),
                StructField("model_pickle", BinaryType()),
            ]
        )
        res_sdf = self.spark.createDataFrame(res_pdf, schema)

        if write:
            if self.conf.get("evaluation_output", None):
                (
                    res_sdf.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
                    .withColumn("run_id", lit(self.run_id))
                    .withColumn("run_date", lit(self.run_date))
                    .withColumn("model", lit(model.params.name))
                    .withColumn("use_case", lit(self.conf["use_case_name"]))
                    .withColumn("model_uri", lit(model_uri))
                    .write.mode("append")
                    .saveAsTable(self.conf.get("evaluation_output"))
                )

        res_df = (
            res_sdf.groupby(["metric_name"])
            .mean("metric_value")
            .withColumnRenamed("avg(metric_value)", "metric_value")
            .toPandas()
        )

        metric_name = None
        metric_value = None

        for rec in res_df.values:
            metric_name, metric_value = rec
            if write:
                mlflow.log_metric(metric_name, metric_value)
                print(res_df)
        return metric_value

    def evaluate_models(self):
        print("Starting evaluate_models")
        for model_name in self.model_registry.get_active_model_keys():
            print(f"Started evaluating {model_name}")
            try:
                model_conf = self.model_registry.get_model_conf(model_name)
                if model_conf["model_type"] == "global":
                    self.evaluate_global_model(model_conf)
                elif model_conf["model_type"] == "local":
                    self.evaluate_local_model(model_conf)
                elif model_conf["model_type"] == "foundation":
                    self.evaluate_foundation_model(model_conf)
            except Exception as err:
                _logger.error(
                    f"Error has occurred while training model {model_name}: {repr(err)}",
                    exc_info=err,
                )
            print(f"Finished evaluating {model_name}")
        print("Finished evaluate_models")

    def evaluate_local_model(self, model_conf):
        src_df = self.resolve_source("train_data")
        src_df, removed = DataQualityChecks(src_df, self.conf, self.spark).run()
        output_schema = StructType(
            [
                StructField(
                    self.conf["group_id"], src_df.schema[self.conf["group_id"]].dataType
                ),
                StructField("backtest_window_start_date", DateType()),
                StructField("metric_name", StringType()),
                StructField("metric_value", DoubleType()),
                StructField("forecast", ArrayType(DoubleType())),
                StructField("actual", ArrayType(DoubleType())),
                StructField("model_pickle", BinaryType()),
            ]
        )
        model = self.model_registry.get_model(model_conf["name"])

        # Use Pandas UDF to forecast
        evaluate_one_model_fn = functools.partial(
            Forecaster.evaluate_one_model, model=model
        )
        res_sdf = (
            src_df.groupby(self.conf["group_id"])
            .applyInPandas(evaluate_one_model_fn, schema=output_schema)
        )
        if self.conf.get("evaluation_output", None) is not None:
            (
                res_sdf.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
                .withColumn("run_id", lit(self.run_id))
                .withColumn("run_date", lit(self.run_date))
                .withColumn("model", lit(model_conf["name"]))
                .withColumn("use_case", lit(self.conf["use_case_name"]))
                .withColumn("model_uri", lit(""))
                .write.mode("append")
                .saveAsTable(self.conf.get("evaluation_output"))
            )
        res_df = (
            res_sdf.groupby(["metric_name"])
            .mean("metric_value")
            .withColumnRenamed("avg(metric_value)", "metric_value")
            .toPandas()
        )
        # Print out aggregated metrics
        print(res_df)

        # Log aggregated metrics to MLflow
        with mlflow.start_run(experiment_id=self.experiment_id):
            for rec in res_df.values:
                metric_name, metric_value = rec
                mlflow.log_metric(metric_name, metric_value)
                mlflow.set_tag("model_name", model_conf["name"])
                mlflow.set_tag("run_id", self.run_id)

    @staticmethod
    def evaluate_one_model(
        pdf: pd.DataFrame, model: ForecastingRegressor
    ) -> pd.DataFrame:
        pdf[model.params["date_col"]] = pd.to_datetime(pdf[model.params["date_col"]])
        pdf.sort_values(by=model.params["date_col"], inplace=True)
        split_date = pdf[model.params["date_col"]].max() - pd.DateOffset(
            months=model.params["backtest_months"]
        )
        group_id = pdf[model.params["group_id"]].iloc[0]
        try:
            pdf = pdf.fillna(0.1)
            # Fix here
            pdf[model.params["target"]] = pdf[model.params["target"]].clip(0.1)
            metrics_df = model.backtest(pdf, start=split_date, group_id=group_id, retrain=False)
            return metrics_df
        except Exception as err:
            _logger.error(
                f"Error evaluating group {group_id} using model {repr(model)}: {err}",
                exc_info=err,
                stack_info=True,
            )
            # raise Exception(f"Error evaluating group {group_id}: {err}")
            return pd.DataFrame(
                columns=[
                    model.params["group_id"],
                    "backtest_window_start_date",
                    "metric_name",
                    "metric_value",
                    "forecast",
                    "actual",
                    "model_pickle",
                ]
            )

    def evaluate_global_model(self, model_conf):
        mlflow_client = mlflow.tracking.MlflowClient()
        hist_df, removed = self.prepare_data_for_global_model("evaluating")
        train_df, val_df = self.split_df_train_val(hist_df)
        model_name = model_conf["name"]
        mlflow_model_name = f"{self.conf['model_output']}.{model_name}_{self.conf['use_case_name']}"
        try:
            champion = mlflow_client.get_model_version_by_alias(mlflow_model_name, "champion")
            champion_version = champion.version
            champion_run_id = f"runs:/{champion.run_id}/model"
            champion_model = mlflow.sklearn.load_model(
                f"models:/{mlflow_model_name}/{champion_version}"
            )
            champion_metrics = self.backtest_global_model(
                model=champion_model,
                train_df=train_df,
                val_df=val_df,
                model_uri=champion_run_id,
                write=False,
            )
        except:
            print(f"No deployed model yet available for model: {mlflow_model_name}")
            champion_model = None

        new_runs = mlflow_client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.candidate='true' and tags.model_name='{model_name}'",
            order_by=["start_time DESC"],
            max_results=10,
        )
        if len(new_runs) == 0:
            print(
                f"No candidate models found for model {model_name}! Nothing to deploy! Exiting.."
            )
            return
        new_run = new_runs[0]
        new_model_uri = f"runs:/{new_run.info.run_uuid}/model"
        new_model = mlflow.sklearn.load_model(new_model_uri)
        new_metrics = self.backtest_global_model(
            model=new_model,
            train_df=train_df,
            val_df=val_df,
            model_uri=new_model_uri,
            write=False,
        )

        if (champion_model is None) or (new_metrics <= champion_metrics):
            model_details = mlflow.register_model(
                model_uri=new_model_uri, name=mlflow_model_name)
            # wait_until_ready(model_details.name, model_details.version)
            # TODO: Add description, version, metadata in general
            mlflow_client.set_registered_model_alias(
                mlflow_model_name,
                "champion",
                model_details.version)
            print(f"Champion alias assigned to the new model")

    def evaluate_foundation_model(self, model_conf):
        model_name = model_conf["name"]
        model = self.model_registry.get_model(model_name)
        hist_df, removed = self.prepare_data_for_global_model("evaluating")
        train_df, val_df = self.split_df_train_val(hist_df)
        metrics = self.backtest_global_model(
            model=model,
            train_df=train_df,
            val_df=val_df,
            model_uri="",
            write=True,
        )
        mlflow.set_tag("action", "train")
        mlflow.set_tag("candidate", "true")
        mlflow.set_tag("model_name", model.params["name"])
        print(f"Finished training {model_conf.get('name')}")

    def score_models(self):
        print("Starting run_scoring")
        for model_name in self.model_registry.get_active_model_keys():
            model_conf = self.model_registry.get_model_conf(model_name)
            print(f"Started scoring with {model_name}")
            if model_conf["model_type"] == "global":
                self.score_global_model(model_conf)
            elif model_conf["model_type"] == "local":
                self.score_local_model(model_conf)
            print(f"Finished scoring with {model_name}")
        print("Finished run_scoring")

    def score_local_model(self, model_conf):
        src_df = self.resolve_source("train_data")
        src_df, removed = DataQualityChecks(src_df, self.conf, self.spark).run()
        # Check if external regressors are provided and framework is statsforecast
        # External regressors are supported only with statsforecast and neuralforecast models
        if (self.conf["scoring_data"])\
                and (self.conf["train_data"] != self.conf["scoring_data"])\
                and (model_conf["framework"] == "StatsForecast"):
            score_df = self.resolve_source("scoring_data")
            score_df = score_df.where(~col(self.conf["group_id"]).isin(removed))
            src_df = src_df.unionByName(score_df, allowMissingColumns=True)
        output_schema = StructType(
            [
                StructField(
                    self.conf["group_id"], src_df.schema[self.conf["group_id"]].dataType
                ),
                StructField(self.conf["date_col"], ArrayType(TimestampType())),
                StructField(self.conf["target"], ArrayType(DoubleType())),
                StructField("model_pickle", BinaryType()),
            ]
        )
        model = self.model_registry.get_model(model_conf["name"])
        score_one_model_fn = functools.partial(Forecaster.score_one_model, model=model)
        res_sdf = (
            src_df.groupby(self.conf["group_id"])
            .applyInPandas(score_one_model_fn, schema=output_schema)
        )
        (
            res_sdf.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
            .withColumn("run_id", lit(self.run_id))
            .withColumn("run_date", lit(self.run_date))
            .withColumn("use_case", lit(self.conf["use_case_name"]))
            .withColumn("model", lit(model_conf["name"]))
            .withColumn("model_uri", lit(""))
            .write.mode("append")
            .saveAsTable(self.conf["scoring_output"])
        )

    @staticmethod
    def score_one_model(
        pdf: pd.DataFrame, model: ForecastingRegressor
    ) -> pd.DataFrame:
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
        except:
            data = [group_id, None, None, None]
        res_df = pd.DataFrame(
            columns=[
                model.params["group_id"],
                model.params["date_col"],
                model.params["target"],
                "model_pickle"], data=[data]
        )
        return res_df

    def score_global_model(self, model_conf):
        print(f"Running scoring for {model_conf['name']}...")
        champion_model, champion_model_uri = self.get_model_for_scoring(model_conf)
        score_df, removed = self.prepare_data_for_global_model("scoring")
        prediction_df, model_fitted = champion_model.forecast(score_df)
        if prediction_df[self.conf["date_col"]].dtype.type != np.datetime64:
            prediction_df[self.conf["date_col"]] = prediction_df[
                self.conf["date_col"]
            ].dt.to_timestamp()
        sdf = self.spark.createDataFrame(prediction_df)\
            .drop('index')\
            .withColumn(self.conf["target"], col(self.conf["target"]).cast(DoubleType()))\
            .orderBy(self.conf["group_id"], self.conf["date_col"])\
            .groupBy(self.conf["group_id"])\
            .agg(
                collect_list(self.conf["date_col"]).alias(self.conf["date_col"]),
                collect_list(self.conf["target"]).alias(self.conf["target"]))
        (
            sdf.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
            .withColumn("model", lit(model_conf["name"]))
            .withColumn("run_id", lit(self.run_id))
            .withColumn("run_date", lit(self.run_date))
            .withColumn("use_case", lit(self.conf["use_case_name"]))
            .withColumn("model_pickle", lit(b""))
            .withColumn("model_uri", lit(champion_model_uri))
            .write.mode("append")
            .saveAsTable(self.conf["scoring_output"])
        )

    def get_latest_model_version(self, mlflow_client, registered_name):
        latest_version = 1
        for mv in mlflow_client.search_model_versions(f"name='{registered_name}'"):
            version_int = int(mv.version)
            if version_int > latest_version:
                latest_version = version_int
        return latest_version

    def get_model_for_scoring(self, model_conf):
        mlflow_client = MlflowClient()
        if model_conf["trainable"]:
            mlflow_model_name = f"{self.conf['model_output']}.{model_conf['name']}_{self.conf['use_case_name']}"
            champion = mlflow_client.get_model_version_by_alias(mlflow_model_name, "champion")
            champion_version = champion.version
            champion_model_uri = f"runs:/{champion.run_id}/model"
            champion_model = mlflow.sklearn.load_model(
                f"models:/{mlflow_model_name}/{champion_version}"
            )
            return champion_model, champion_model_uri
        else:
            return self.model_registry.get_model(model_conf["name"]), None

def flatten_nested_parameters(d):
    out = {}
    for key, val in d.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list) and all(isinstance(item, dict) for item in val):
            for subdict in val:
                deeper = flatten_nested_parameters(subdict).items()
                out.update({key + "_" + key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out
