import functools
import logging
import pathlib
import shutil
from datetime import datetime
import uuid

import yaml
from hyperopt import fmin, tpe, SparkTrials, STATUS_OK, rand, atpe
from typing import Dict, Any, Tuple, List, Union
import pandas as pd
import numpy as np
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf, DictConfig
from omegaconf.basecontainer import BaseContainer
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DateType,
    DoubleType,
    TimestampType,
)
from pyspark.sql.functions import lit, avg, min, max, col

from forecasting_sa.models.abstract_model import ForecastingSARegressor
from forecasting_sa.models import ModelRegistry
from forecasting_sa.data_quality_checks import DataQualityChecks

_logger = logging.getLogger(__name__)


class Forecaster:
    def __init__(
        self,
        conf: Union[str, Dict[str, Any]],
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
        # But just until prediction_length
        # val_true_df = val_true_df[
        #    val_true_df[self.conf['date_col']] < val_true_df[self.conf['date_col']].min() + pd.DateOffset(
        #        days=self.conf['prediction_length'])]
        return train_df, val_true_df

    def prepare_data(self, model_conf: DictConfig, path: str) -> pd.DataFrame:

        df = self.spark.read.table(path)

        if model_conf.get("data_prep", "none") == "pivot":
            df = (
                df.groupby([self.conf["date_col"]])
                .pivot(self.conf["group_id"])
                .sum(self.conf["target"])
            )

        df = df.toPandas()
        df = DataQualityChecks(df, self.conf).run()
        if model_conf.get("data_prep", "none") == "none":
            df[self.conf["group_id"]] = df[self.conf["group_id"]].astype(str)

        print(df)
        return df

    ##### Put Ensamble Here | Add output for past mb?
    def train_eval_score(self, export_metrics=False, scoring=True) -> str:
        self.run_id = str(uuid.uuid4())
        self.train_models()
        self.evaluate_models()
        if scoring:
            self.run_scoring()
            self.ensemble()
        # if export_metrics:
        #    self.update_metrics()
        return self.run_id

    def ensemble(self):
        if self.conf.get("ensemble") and self.conf["ensemble_scoring_output"]:
            metrics_df = (
                self.spark.table(self.conf["metrics_output"])
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
            aggregated_df = df.groupby(
                self.conf["group_id"], self.conf["date_col"]
            ).agg(
                avg(self.conf["target"]).alias(self.conf["target"] + "_avg"),
                min(self.conf["target"]).alias(self.conf["target"] + "_min"),
                max(self.conf["target"]).alias(self.conf["target"] + "_max"),
            )
            aggregated_df.show(100, False)
            (
                aggregated_df.withColumn("run_id", lit(self.run_id))
                .withColumn("run_date", lit(self.run_date))
                .withColumn("use_case", lit(self.conf["use_case_name"]))
                .withColumn("model", lit("ensemble"))
                .write.format("delta")
                .option("mergeSchema", "true")
                .mode("append")
                .saveAsTable(self.conf["ensemble_scoring_output"])
            )

    def train_models(self):
        """Trains and evaluates all models from the configuration file with the configuration's training data.
        Then evaluates the current best model with the configuration's training data.
        Saves the params, models and metrics, so the runs will all have evaluation data
        """
        print("Starting train_evaluate_models")
        for model_name in self.model_registry.get_active_model_keys():
            model_conf = self.model_registry.get_model_conf(model_name)
            if (
                model_conf.get("trainable", False)
                and model_conf.get("model_type", None) == "global"
            ):
                with mlflow.start_run(experiment_id=self.experiment_id) as run:
                    try:
                        # Get training and scoring data
                        hist_df = self.prepare_data(model_conf, self.conf["train_data"])
                        train_df, val_train_df = self.split_df_train_val(hist_df)
                        # Train and evaluate new models - results are saved to MLFlow
                        model = self.model_registry.get_model(model_name)
                        print("--------------------\nTraining model:")
                        print(model)
                        # Trains and evaluates the model - logs results to experiment
                        self.train_one_model(train_df, val_train_df, model_conf, model)
                    except Exception as err:
                        _logger.error(
                            f"Error has occurred while training model {model_name}: {repr(err)}",
                            exc_info=err,
                        )
                        raise err

        print("Finished train_models")

    def train_one_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_conf: DictConfig,
        model: ForecastingSARegressor,
    ):
        print("starting train")
        tuned_model, tuned_params = self.tune_model(model_conf, model, train_df, val_df)
        model_info = mlflow.sklearn.log_model(tuned_model, "model")
        try:
            # TODO Decide if we should flatten in general
            mlflow.log_params(tuned_params)
        except MlflowException:
            # MLflow log_params has a parameter length limit of 500
            # When using ensemble models parameters consist of
            # nested parameter dictionaries which are flattened here
            mlflow.log_params(
                flatten_nested_parameters(OmegaConf.to_object(tuned_params))
            )
        self.backtest_and_log_metrics(tuned_model, train_df, val_df, "train")
        mlflow.set_tag("action", "train")
        mlflow.set_tag("candidate", "true")
        mlflow.set_tag("model_name", model.params["name"])
        print("Finished train")
        return model_info

    def tune_model(
        self,
        model_conf: DictConfig,
        model: ForecastingSARegressor,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ):
        def objective(params, train_df_path, val_df_path):
            _conf = dict(model_conf)
            _conf.update(params)
            _model = model.__class__(_conf)
            train_df = pd.read_parquet(train_df_path)
            val_df = pd.read_parquet(val_df_path)
            _model.fit(train_df)
            _metrics = _model.backtest(
                pd.concat([train_df, val_df]),
                start=val_df[_date_col].min(),
                retrain=_tuning_retrain,
            )

            return {"loss": _metrics["smape"], "status": STATUS_OK}

        if (
            self.conf.get("tuning_enabled", False)
            and model.supports_tuning()
            and model.params.get("tuning", True)
        ):
            print(f'Tuning model: {model_conf["name"]}')
            # with mlflow.start_run():
            spark_trials = None
            if self.conf["tuning_distributed"]:
                spark_trials = SparkTrials(
                    spark_session=self.spark,
                    parallelism=int(self.conf["tuning_parallelism"]),
                )
            # print(model.search_space())
            temp_prefix = self.conf.get("temp_path", "/tmp")
            run_id = uuid.uuid4().hex
            final_prefix = f"{temp_prefix}/{run_id}/"
            pathlib.Path(final_prefix).mkdir(parents=True, exist_ok=True)
            train_path = f"{final_prefix}/train.parquet"
            val_path = f"{final_prefix}/val.parquet"
            train_df.to_parquet(f"{temp_prefix}/{run_id}/train.parquet")
            val_df.to_parquet(f"/{temp_prefix}/{run_id}/val.parquet")
            _objective_bound = functools.partial(
                objective,
                train_df_path=train_path,
                val_df_path=val_path,
            )
            _date_col = self.conf["date_col"]
            _tuning_retrain = self.conf["tuning_retrain"]
            best_params = fmin(
                fn=_objective_bound,
                space=model.search_space(),
                algo=tpe.suggest,
                max_evals=int(self.conf["tuning_max_trials"]),
                trials=spark_trials,
            )
            shutil.rmtree(final_prefix, ignore_errors=True)
            print(best_params)
            _conf = dict(model_conf)
            _conf.update(best_params)
            for k in _conf.keys():
                if type(_conf[k]) == np.float64:
                    _conf[k] = float(_conf[k])
                elif type(_conf[k]) == np.int64:
                    _conf[k] = int(_conf[k])
            best_model = self.model_registry.get_model(
                model_conf["name"], OmegaConf.create(_conf)
            )
            # TODO final train should be configurable
            # check if we should use all the data for final retrain
            best_model.fit(train_df)
            return best_model, best_params
        else:
            print(f'Fitting model: {model_conf["name"]}')
            model.fit(train_df)
            return model, model_conf

    def backtest_and_log_metrics(
        self,
        model: ForecastingSARegressor,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        prefix: str,
    ):
        metrics_df = (
            model.backtest(
                pd.concat([train_df, val_df]),
                start=val_df[self.conf["date_col"]].min(),
                retrain=self.conf["backtest_retrain"],
            )
            .groupby("metric_name")
            .mean()
        )
        metrics = {
            f"{k}_{prefix}": v
            for x in metrics_df.to_dict().values()
            for k, v in x.items()
        }
        mlflow.log_metrics(next(iter(metrics_df.to_dict().values())))
        return metrics

    def evaluate_models(self):
        print("Starting evaluate_models")
        for model_name in self.model_registry.get_active_model_keys():
            try:
                model_conf = self.model_registry.get_model_conf(model_name)
                if model_conf["model_type"] == "global":
                    self.evaluate_global_model(model_conf)
                elif model_conf["model_type"] == "local":
                    self.evaluate_local_model(model_conf)
            except Exception as err:
                _logger.error(
                    f"Error has occurred while training model {model_name}: {repr(err)}",
                    exc_info=err,
                )
        print("Finished evaluate_models")

    """
    @staticmethod
    def _prep_data_one_ts(
        pdf: pd.DataFrame, model: ForecastingSARegressor
    ) -> pd.DataFrame:
        pdf[model.params["date_col"]] = pd.to_datetime(pdf[model.params["date_col"]])
        date_idx = pd.date_range(
            start=pdf[model.params["date_col"]].min(),
            end=pdf[model.params["date_col"]].max(),
            freq=model.params["freq"],
            name=model.params["date_col"],
        )
        pdf.set_index(model.params["date_col"], inplace=True)
        pdf.sort_index(inplace=True)
        pdf = pdf.reindex(date_idx, method="backfill").reset_index()
        return pdf
    """

    @staticmethod
    def evaluate_one_model(
        pdf: pd.DataFrame, model: ForecastingSARegressor
    ) -> pd.DataFrame:

        pdf[model.params["date_col"]] = pd.to_datetime(pdf[model.params["date_col"]])
        pdf.sort_values(by=model.params["date_col"], inplace=True)
        split_date = pdf[model.params["date_col"]].max() - pd.DateOffset(
            months=model.params["backtest_months"]
        )

        group_id = pdf[model.params["group_id"]].iloc[0]
        print(group_id)
        try:
            pdf = pdf.fillna(0.1)
            pdf[model.params["target"]] = pdf[model.params["target"]].clip(0.1)
            metrics_df = model.backtest(pdf, start=split_date, retrain=True)
            print(metrics_df)
            metrics_df[model.params["group_id"]] = group_id
            return metrics_df
        except Exception as err:
            _logger.error(
                f"Error evaluating group {group_id} using mode {repr(model)}: {err}",
                exc_info=err,
                stack_info=True,
            )
            # raise Exception(f"Error evaluating group {group_id}: {e}", e)
            return pd.DataFrame(
                columns=[
                    model.params["group_id"],
                    "backtest_window_start_date",
                    "metric_name",
                    "metric_value",
                ]
            )

    def evaluate_local_model(self, model_conf):
        src_df = self.spark.read.table(self.conf["train_data"])
        src_df = DataQualityChecks(src_df, self.conf, self.spark).run()
        output_schema = StructType(
            [
                StructField(
                    self.conf["group_id"], src_df.schema[self.conf["group_id"]].dataType
                ),
                StructField("backtest_window_start_date", DateType()),
                StructField("metric_name", StringType()),
                StructField("metric_value", DoubleType()),
            ]
        )
        model = self.model_registry.get_model(model_conf["name"])
        evaluate_one_model_fn = functools.partial(
            Forecaster.evaluate_one_model, model=model
        )
        res_sdf = (
            src_df.repartition(100)
            .groupby(self.conf["group_id"])
            .applyInPandas(evaluate_one_model_fn, schema=output_schema)
        )
        if self.conf.get("metrics_output", None) is not None:
            (
                res_sdf.withColumn("run_id", lit(self.run_id))
                .withColumn("run_date", lit(self.run_date))
                .withColumn("model", lit(model_conf["name"]))
                .withColumn("use_case", lit(self.conf["use_case_name"]))
                .write.mode("append")
                .saveAsTable(self.conf.get("metrics_output"))
            )
        res_df = (
            res_sdf.groupby(["metric_name"])
            .mean("metric_value")
            .withColumnRenamed("avg(metric_value)", "metric_value")
            .toPandas()
        )

        print(res_df)
        with mlflow.start_run(experiment_id=self.experiment_id):
            for rec in res_df.values:
                metric_name, metric_value = rec
                mlflow.log_metric(metric_name, metric_value)
                mlflow.set_tag("model_name", model_conf["name"])
                mlflow.set_tag("forecast_run_id", self.run_id)

    def evaluate_global_model(self, model_conf):
        mlflow_client = mlflow.tracking.MlflowClient()
        with mlflow.start_run(experiment_id=self.experiment_id):
            hist_df = self.prepare_data(model_conf, self.conf["train_data"])
            train_df, val_df = self.split_df_train_val(hist_df)
            model_name = model_conf["name"]
            mlflow.set_tag("model_name", model_conf["name"])
            mlflow_model_name = f"{self.conf['use_case_name']}_{model_name}"
            try:
                deployed_model = mlflow.sklearn.load_model(
                    f"models:/{mlflow_model_name}/{self.conf['scoring_model_stage']}"
                )
                deployed_metrics = self.backtest_and_log_metrics(
                    deployed_model, train_df, val_df, "deployed"
                )
            except:
                print(
                    "No deployed model yet available for model: ",
                    mlflow_model_name,
                )
                deployed_model = None

            new_runs = mlflow_client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.candidate='true' and tags.model_name='{model_name}'",
                order_by=[f"metrics.{self.conf['selection_metric']}"],
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
            new_metrics = self.backtest_and_log_metrics(
                new_model, train_df, val_df, "new"
            )

            if (
                deployed_model is None
                or new_metrics["smape_new"] <= deployed_metrics["smape_deployed"]
            ):
                model_details = mlflow.register_model(
                    model_uri=new_model_uri, name=mlflow_model_name
                )
                # wait_until_ready(model_details.name, model_details.version)
                # TODO: Add description, version, metadata in general
                mlflow_client.transition_model_version_stage(
                    name=model_details.name,
                    version=model_details.version,
                    stage=self.conf["scoring_model_stage"],
                )
                print("Model promoted to production:")
                print(model_details)

    def run_scoring(self):
        print("starting run_scoring")
        for model_name in self.model_registry.get_active_model_keys():
            model_conf = self.model_registry.get_model_conf(model_name)
            if model_conf["model_type"] == "global":
                self.run_scoring_for_global_model(model_conf)
            elif model_conf["model_type"] == "local":
                self.run_scoring_for_local_model(model_conf)
            print("finished run_scoring")

    @staticmethod
    def score_one_model(
        pdf: pd.DataFrame, model: ForecastingSARegressor
    ) -> pd.DataFrame:

        pdf[model.params["date_col"]] = pd.to_datetime(pdf[model.params["date_col"]])
        pdf.sort_values(by=model.params["date_col"], inplace=True)
        group_id = pdf[model.params["group_id"]].iloc[0]
        print(group_id)
        try:
            model.fit(pdf)
            res_df = model.predict(pdf)
            res_df[model.params["group_id"]] = group_id
            return res_df
        except Exception as e:
            # raise Exception(f"Error scroing group {group_key}: {e}")
            # todo add logging for failed time series
            return pd.DataFrame(
                columns=[
                    model.params["date_col"],
                    model.params["group_id"],
                    model.params["target"],
                ]
            )

    def run_scoring_for_local_model(self, model_conf):
        src_df = self.spark.read.table(self.conf["scoring_data"])
        src_df = DataQualityChecks(src_df, self.conf, self.spark).run()
        output_schema = StructType(
            [
                StructField(self.conf["date_col"], TimestampType()),
                StructField(
                    self.conf["group_id"], src_df.schema[self.conf["group_id"]].dataType
                ),
                StructField(self.conf["target"], DoubleType()),
            ]
        )
        model = self.model_registry.get_model(model_conf["name"])
        score_one_model_fn = functools.partial(Forecaster.score_one_model, model=model)

        res_sdf = (
            src_df.repartition(100)
            .groupby(self.conf["group_id"])
            .applyInPandas(score_one_model_fn, schema=output_schema)
        )
        if not isinstance(res_sdf.schema[self.conf["group_id"]].dataType, StringType):
            res_sdf = res_sdf.withColumn(
                self.conf["group_id"], col(self.conf["group_id"]).cast(StringType())
            )
        (
            res_sdf.withColumn("run_id", lit(self.run_id))
            .withColumn("run_date", lit(self.run_date))
            .withColumn("use_case", lit(self.conf["use_case_name"]))
            .withColumn("model", lit(model_conf["name"]))
            .write.mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(self.conf["scoring_output"])
        )

    def run_scoring_for_global_model(self, model_conf):
        print(f"Running scoring for {model_conf['name']}...")
        best_model = self.get_model_for_scoring(model_conf)
        score_df = self.prepare_data(model_conf, self.conf["scoring_data"])
        prediction_df = best_model.predict(score_df)
        # check if we need to unpivot data
        if model_conf.get("data_prep", "none") == "pivot":
            prediction_df = (
                prediction_df.reset_index()
                .melt(id_vars=["index"])
                .rename(
                    columns={
                        "index": self.conf["date_col"],
                        "variable": self.conf["group_id"],
                        "value": self.conf["target"],
                    }
                )
            )
        if prediction_df[self.conf["date_col"]].dtype.type != np.datetime64:
            prediction_df[self.conf["date_col"]] = prediction_df[
                self.conf["date_col"]
            ].dt.to_timestamp()

        print(f"prediction generated, saving to {self.conf['scoring_output']}")
        spark_df = (
            self.spark.createDataFrame(prediction_df)
            .withColumn("model", lit(model_conf["name"]))
            .withColumn(
                self.conf["target"], col(self.conf["target"]).cast(DoubleType())
            )
        )
        (
            spark_df.withColumn("run_id", lit(self.run_id))
            .withColumn("run_date", lit(self.run_date))
            .withColumn("use_case", lit(self.conf["use_case_name"]))
            .withColumn("model", lit(model_conf["name"]))
            .write.mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(self.conf["scoring_output"])
        )
        print(f"Finished scoring for {model_conf['name']}...")

    def get_model_for_scoring(self, model_conf):
        if model_conf["trainable"]:
            mlflow_model_name = f"{self.conf['use_case_name']}_{model_conf['name']}"
            best_model = mlflow.sklearn.load_model(
                f"models:/{mlflow_model_name}/{self.conf['scoring_model_stage']}"
            )
            return best_model
        else:
            return self.model_registry.get_model(model_conf["name"])

    # def update_metrics(self):
    #     df = self.spark.read.format("mlflow-experiment").load(str(self.experiment_id))
    #     df = df.where("tags.candidate='true'").selectExpr(
    #         "run_id",
    #         "end_time as time",
    #         "tags.model_name",
    #         "metrics.smape_train as smape",
    #     )
    #     df.write.format("delta").mode("overwrite").saveAsTable(
    #         "forecastingsa.model_metrics"
    #     )


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
