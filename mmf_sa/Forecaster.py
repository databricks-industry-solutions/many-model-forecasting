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
from mlflow.tracking import MlflowClient
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec
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
)
from pyspark.sql.functions import lit, avg, min, max, col, posexplode, collect_list, to_date, countDistinct
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
            raise Exception("No configuration provided!")
        if run_id:
            self.run_id = run_id
        else:
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
        self.run_date = datetime.now()

    def set_mlflow_experiment(self):
        """
        Sets the MLflow experiment using the provided experiment path and returns the experiment id.
        Parameters: self (Forecaster): A Forecaster object.
        Returns: experiment_id (str): A string specifying the experiment id.
        """
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
        if self.data_conf:
            df_val = self.data_conf.get(key)
            if df_val is not None and isinstance(df_val, pd.DataFrame):
                return self.spark.createDataFrame(df_val)
            elif df_val is not None and isinstance(df_val, DataFrame):
                return df_val
        else:
            return self.spark.read.table(self.conf[key])

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
        if (mode == "scoring") \
                and (self.conf["scoring_data"]) \
                and (self.conf["scoring_data"] != self.conf["train_data"]):
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
            <= df[self.conf["date_col"]].max()
            - pd.DateOffset(months=self.conf["backtest_months"])
        ]
        # Validate with data after the backtest months cutoff...
        val_df = df[
            df[self.conf["date_col"]]
            > df[self.conf["date_col"]].max()
            - pd.DateOffset(months=self.conf["backtest_months"])
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
        print("Starting evaluate_score")
        if evaluate:
            self.evaluate_models()
        if score:
            self.score_models()
        print("Finished evaluate_score")
        return self.run_id

    def evaluate_models(self):
        """
        Trains and evaluates all models from the active models list.
        Parameters: self (Forecaster): A Forecaster object.
        """
        print("Starting evaluate_models")
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
            except Exception as err:
                _logger.error(
                    f"Error has occurred while evaluating model {model_name}: {repr(err)}",
                    exc_info=err,
                )
            print(f"Finished evaluating {model_name}")
        print("Finished evaluate_models")

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
                    StructField("backtest_window_start_date", DateType()),
                    StructField("metric_name", StringType()),
                    StructField("metric_value", DoubleType()),
                    StructField("forecast", ArrayType(DoubleType())),
                    StructField("actual", ArrayType(DoubleType())),
                    StructField("model_pickle", BinaryType()),
                ]
            )

            model = self.model_registry.get_model(model_conf["name"])

            # Use Pandas UDF to forecast individual group
            evaluate_one_local_model_fn = functools.partial(
                Forecaster.evaluate_one_local_model, model=model
            )

            res_sdf = (
                src_df.groupby(self.conf["group_id"])
                .applyInPandas(evaluate_one_local_model_fn, schema=output_schema)
            )
        
            # Write evaluation result to a delta table
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
        pdf: pd.DataFrame, model: ForecastingRegressor
    ) -> pd.DataFrame:
        """
        A static method that evaluates a single local model using the provided pandas DataFrame and model.
        If the evaluation for a single group fails, it returns an empty DataFrame without failing the entire process.
        Parameters:
            pdf (pd.DataFrame): A pandas DataFrame.
            model (ForecastingRegressor): A ForecastingRegressor object.
        Returns: metrics_df (pd.DataFrame): A pandas DataFrame.
        """
        pdf[model.params["date_col"]] = pd.to_datetime(pdf[model.params["date_col"]])
        pdf.sort_values(by=model.params["date_col"], inplace=True)
        split_date = pdf[model.params["date_col"]].max() - pd.DateOffset(
            months=model.params["backtest_months"]
        )
        group_id = pdf[model.params["group_id"]].iloc[0]
        try:
            pdf = pdf.fillna(0)
            # Fix here
            pdf[model.params["target"]] = pdf[model.params["target"]].clip(0)
            metrics_df = model.backtest(pdf, start=split_date, group_id=group_id)
            return metrics_df
        except Exception as err:
            _logger.error(
                f"Error evaluating group {group_id} using model {repr(model)}: {err}",
                exc_info=err,
                stack_info=True,
            )
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
        """
        Evaluate a global model using the provided model configuration. Trains and registers the model.
        Parameters:
            self (Forecaster): A Forecaster object.
            model_conf (dict): A dictionary specifying the model configuration.
        """
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            model_name = model_conf["name"]
            hist_df, removed = self.prepare_data_for_global_model("evaluating")
            train_df, val_df = self.split_df_train_val(hist_df)

            # First, we train the model on the entire history (train_df, val_df).
            # Then we register this model as our final model in Unity Catalog.
            final_model = self.model_registry.get_model(model_name)
            final_model.fit(pd.concat([train_df, val_df]))
            input_example = train_df[train_df[self.conf['group_id']] == train_df[self.conf['group_id']] \
                .unique()[0]].sort_values(by=[self.conf['date_col']])

            # Prepare model signature for model registry
            input_schema = infer_signature(model_input=input_example).inputs
            output_schema = Schema(
                [
                    ColSpec("integer", "index"),
                    ColSpec("string", self.conf['group_id']),
                    ColSpec("datetime", self.conf['date_col']),
                    ColSpec("float", self.conf['target']),
                ]
            )
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # Register the model
            model_info = mlflow.sklearn.log_model(
                final_model,
                "model",
                registered_model_name=f"{self.conf['model_output']}.{model_conf['name']}_{self.conf['use_case_name']}",
                input_example=input_example,
                signature=signature,
                pip_requirements=[],
            )
            mlflow.log_params(final_model.get_params())
            mlflow.set_tag("run_id", self.run_id)
            mlflow.set_tag("model_name", final_model.params["name"])
            print(f"Model registered: {self.conf['model_output']}.{model_conf['name']}_{self.conf['use_case_name']}")

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

    def backtest_global_model(
        self,
        model: ForecastingRegressor,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_uri: str,
        write: bool = True,
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
        Returns: metric_value (float): A float specifying the mean metric value.
        """
        res_pdf = (
            model.backtest(
                pd.concat([train_df, val_df]),
                start=train_df[self.conf["date_col"]].max(),
                spark=self.spark,
                # backtest_retrain=self.conf["backtest_retrain"],
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

        # Write evaluation results to a delta table
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
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            model_name = model_conf["name"]
            model = self.model_registry.get_model(model_name)
            # For now, only support registering chronos, moirai and moment models
            if model_conf["framework"] in ["Chronos", "Moirai", "Moment", "TimesFM"]:
                model.register(
                    registered_model_name=f"{self.conf['model_output']}.{model_conf['name']}_{self.conf['use_case_name']}"
                )
            hist_df, removed = self.prepare_data_for_global_model("evaluating")  # Reuse the same as global
            train_df, val_df = self.split_df_train_val(hist_df)
            model_uri = f"runs:/{run.info.run_id}/model"
            metrics = self.backtest_global_model(  # Reuse the same as global
                model=model,
                train_df=train_df,
                val_df=val_df,
                model_uri=model_uri,
                write=True,
            )
            mlflow.log_metric(self.conf["metric"], metrics)
            mlflow.set_tag("model_name", model.params["name"])
            mlflow.set_tag("run_id", self.run_id)
            mlflow.log_params(model.get_params())

    def score_models(self):
        """
        Scores the models using the provided model configuration.
        Parameters: self (Forecaster): A Forecaster object.
        """
        print("Starting run_scoring")
        for model_name in self.model_registry.get_active_model_keys():
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
        if (self.conf["scoring_data"])\
                and (self.conf["train_data"] != self.conf["scoring_data"])\
                and (model_conf["framework"] == "StatsForecast"):
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
        )
        model = self.model_registry.get_model(model_conf["name"])

        # Use Pandas UDF to distribute scoring
        score_one_local_model_fn = functools.partial(Forecaster.score_one_local_model, model=model)
        res_sdf = (
            src_df.groupby(self.conf["group_id"])
            .applyInPandas(score_one_local_model_fn, schema=output_schema)
        )

        # Write the results to a delta table
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
    def score_one_local_model(
        pdf: pd.DataFrame, model: ForecastingRegressor
    ) -> pd.DataFrame:
        """
        A static method that scores a single local model using the provided pandas DataFrame and model.
        If the scoring for one time series fails, it returns an empty DataFrame instead of failing the
        entire process.
        Parameters:
            pdf (pd.DataFrame): A pandas DataFrame.
            model (ForecastingRegressor): A ForecastingRegressor object.
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
        model, model_uri = self.get_model_for_scoring(model_conf)
        score_df, removed = self.prepare_data_for_global_model("scoring")
        prediction_df, model_fitted = model.forecast(score_df)
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
            .withColumn("model_uri", lit(model_uri))
            .write.mode("append")
            .saveAsTable(self.conf["scoring_output"])
        )

    def score_foundation_model(self, model_conf):
        """
        Scores a global model using the provided model configuration. Writes the results to a delta table.
        Parameters:
            self (Forecaster): A Forecaster object.
            model_conf (dict): A dictionary specifying the model configuration.
        """
        print(f"Running scoring for {model_conf['name']}...")
        model_name = model_conf["name"]
        _, model_uri = self.get_model_for_scoring(model_conf)
        model = self.model_registry.get_model(model_name)
        hist_df, removed = self.prepare_data_for_global_model()
        prediction_df, model_pretrained = model.forecast(hist_df, spark=self.spark)
        sdf = self.spark.createDataFrame(prediction_df).drop('index')
        (
            sdf.withColumn(self.conf["group_id"], col(self.conf["group_id"]).cast(StringType()))
            .withColumn("model", lit(model_conf["name"]))
            .withColumn("run_id", lit(self.run_id))
            .withColumn("run_date", lit(self.run_date))
            .withColumn("use_case", lit(self.conf["use_case_name"]))
            .withColumn("model_pickle", lit(b""))
            .withColumn("model_uri", lit(model_uri))
            .write.mode("append")
            .saveAsTable(self.conf["scoring_output"])
        )

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
            model = mlflow.sklearn.load_model(f"models:/{registered_name}/{model_version}")
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

