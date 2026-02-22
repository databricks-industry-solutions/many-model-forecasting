import pandas as pd
import numpy as np
import logging
import torch
import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
from typing import Iterator, Tuple
from pyspark.sql.functions import collect_list, pandas_udf
from pyspark.sql import DataFrame
from mmf_sa.models.abstract_model import ForecastingRegressor, MODEL_PIP_REQUIREMENTS
from mmf_sa.exceptions import (
    MissingFeatureError,
    UnsupportedMetricError,
    ModelPredictionError,
    DataPreparationError,
)

_logger = logging.getLogger(__name__)


class ChronosForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = None
        self.model = None

    def register(self, registered_model_name: str):
        pipeline = ChronosModel(
            self.repo,
            self.params["prediction_length"],
        )
        input_schema = Schema([TensorSpec(np.dtype(np.double), (-1, -1))])
        output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, -1, -1))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = np.random.rand(1, 52)
        mlflow.pyfunc.log_model(
            "model",
            python_model=pipeline,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=MODEL_PIP_REQUIREMENTS["chronos"],
        )

    def create_horizon_timestamps_udf(self):
        @pandas_udf('array<timestamp>')
        def horizon_timestamps_udf(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            import numpy as np
            batch_horizon_timestamps = []
            for batch in batch_iterator:
                for series in batch:
                    last = series.max()
                    horizon_timestamps = []
                    for i in range(self.params["prediction_length"]):
                        last = last + self.one_ts_offset
                        horizon_timestamps.append(last.to_numpy())
                    batch_horizon_timestamps.append(np.array(horizon_timestamps))    
            yield pd.Series(batch_horizon_timestamps)
        return horizon_timestamps_udf

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> DataFrame:
        df = spark.createDataFrame(df)
        df = (
            df.groupBy(self.params.group_id)
            .agg(
                collect_list(self.params.date_col).alias('ds'),
                collect_list(self.params.target).alias('y'),
            )).withColumnRenamed(self.params.group_id, "unique_id")

        return df

    def predict(self,
                hist_df: pd.DataFrame,
                val_df: pd.DataFrame = None,
                curr_date=None,
                spark=None):
        hist_df = self.prepare_data(hist_df, spark=spark)
        horizon_timestamps_udf = self.create_horizon_timestamps_udf()
        device_count = torch.cuda.device_count()
        if device_count == 0:
            device_count = 1
        forecast_udf = self.create_predict_udf(device_count)
        forecast_df = (
            hist_df.repartition(device_count, self.params.group_id)
            .select(
                hist_df.unique_id,
                horizon_timestamps_udf(hist_df.ds).alias("ds"),
                forecast_udf(hist_df.y).alias("y"))
        ).toPandas()
        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                "y": self.params.target,
            }
        )
        # Todo
        # forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame, spark=None):
        return self.predict(df, spark=spark)

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> list:
        pred_df, model_pretrained = self.predict(hist_df, val_df, curr_date, spark)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        metric_name = self.params["metric"]
        if metric_name not in ("smape", "mape", "mae", "mse", "rmse"):
            raise UnsupportedMetricError(f"Metric {self.params['metric']} not supported!")
        for key in keys:
            actual = val_df[val_df[self.params["group_id"]] == key][self.params["target"]].to_numpy()
            forecast = pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].to_numpy()[0]
            # Mapping metric names to their respective classes
            metric_classes = {
                "smape": MeanAbsolutePercentageError(symmetric=True),
                "mape": MeanAbsolutePercentageError(symmetric=False),
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(square_root=False),
                "rmse": MeanSquaredError(square_root=True),
            }
            try:
                if metric_name in metric_classes:
                    metric_function = metric_classes[metric_name]
                    metric_value = metric_function(actual, forecast)
                metrics.extend(
                    [(
                        key,
                        curr_date,
                        metric_name,
                        metric_value,
                        forecast,
                        actual,
                        b'',
                    )])
            except (ModelPredictionError, DataPreparationError) as err:
                _logger.warning(f"Failed to calculate metric for key {key}: {err}")
            except Exception as err:
                _logger.warning(f"Unexpected error calculating metric for key {key}: {err}")
        return metrics

    def create_predict_udf(self, device_count):
        repo = self.repo
        num_devices = device_count

        @pandas_udf('array<double>')
        def predict_udf(bulk_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            # initialization step
            import torch
            import numpy as np
            import pandas as pd
            from pyspark import TaskContext

            # Assign this worker to a specific GPU based on partition ID
            ctx = TaskContext.get()
            partition_id = ctx.partitionId() if ctx else 0
            gpu_id = partition_id % num_devices
            torch.cuda.set_device(gpu_id)

            # Initialize the ChronosPipeline with a pretrained model from the specified repository
            from chronos import BaseChronosPipeline
            pipeline = BaseChronosPipeline.from_pretrained(
                repo,
                device_map=f"cuda:{gpu_id}",
                dtype=torch.bfloat16,
            )

            # inference
            median = []
            for bulk in bulk_iterator:
                for i in range(0, len(bulk), self.params["batch_size"]):
                    batch = bulk[i:i+self.params["batch_size"]]
                    contexts = [torch.tensor(list(series)) for series in batch]
                    forecasts = pipeline.predict(
                        inputs=contexts,
                        prediction_length=self.params["prediction_length"],
                    )
                    median.extend([np.median(forecast, axis=0) for forecast in forecasts])
            yield pd.Series(median)
        return predict_udf


class ChronosBoltTiny(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-bolt-tiny"


class ChronosBoltMini(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-bolt-mini"


class ChronosBoltSmall(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-bolt-small"


class ChronosBoltBase(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-bolt-base"


class Chronos2Forecaster(ChronosForecaster):
    """Forecaster for Chronos-2 models with native covariate support.

    Chronos-2 natively supports past-only and known-future covariates via
    a list-of-dicts input format:
      {"target": ..., "past_covariates": {...}, "future_covariates": {...}}

    Output from predict() is a list of tensors, each shaped
    (n_variates, n_quantiles, prediction_length) with default quantiles
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
    """

    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def register(self, registered_model_name: str):
        pipeline = Chronos2MLflowModel(
            self.repo,
            self.params["prediction_length"],
        )
        input_schema = Schema([TensorSpec(np.dtype(np.double), (-1, -1))])
        output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, -1, -1))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = np.random.rand(1, 52)
        mlflow.pyfunc.log_model(
            "model",
            python_model=pipeline,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=MODEL_PIP_REQUIREMENTS["chronos"],
        )

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> pd.DataFrame:
        """Select and rename columns from a pandas DataFrame.

        When *future* is False the target column is included; when True only
        the identifier, timestamp, and covariate columns are kept.
        """
        if not future:
            features = [self.params.group_id, self.params.date_col, self.params.target]
            if "dynamic_future_numerical" in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_numerical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future numerical missing: {e}")
            if "dynamic_future_categorical" in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_categorical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future categorical missing: {e}")
            if "static_features" in self.params.keys():
                try:
                    features = features + self.params.static_features
                except Exception as e:
                    raise MissingFeatureError(f"Static features missing: {e}")
            _df = df[features]
            _df = _df.rename(
                columns={
                    self.params.group_id: "unique_id",
                    self.params.date_col: "ds",
                    self.params.target: "y",
                }
            )
        else:
            features = [self.params.group_id, self.params.date_col]
            if "dynamic_future_numerical" in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_numerical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future numerical missing: {e}")
            if "dynamic_future_categorical" in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_categorical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future categorical missing: {e}")
            _df = df[features]
            _df = _df.rename(
                columns={
                    self.params.group_id: "unique_id",
                    self.params.date_col: "ds",
                }
            )
        return _df.sort_values(by=["unique_id", "ds"])

    def prepare_data_for_spark(self, hist_df: pd.DataFrame, spark,
                               val_df: pd.DataFrame = None) -> DataFrame:
        """Convert pandas DataFrames to a grouped Spark DataFrame.

        The resulting Spark DataFrame has one row per unique_id with columns:
          - ``ds``: array of historical timestamps
          - ``y``: array of historical target values
          - one array column per dynamic covariate (spanning history + future)
          - one scalar column per static feature
        """
        dyn_num_vars = list(self.params.get("dynamic_future_numerical", []))
        dyn_cat_vars = list(self.params.get("dynamic_future_categorical", []))
        static_vars = list(self.params.get("static_features", []))
        dynamic_vars = dyn_num_vars + dyn_cat_vars

        hist_pdf = self.prepare_data(hist_df)

        core_pdf = hist_pdf[["unique_id", "ds", "y"]].sort_values(
            by=["unique_id", "ds"]
        )
        core_sdf = spark.createDataFrame(core_pdf)
        core_sdf = core_sdf.groupBy("unique_id").agg(
            collect_list("ds").alias("ds"),
            collect_list("y").alias("y"),
        )

        if dynamic_vars:
            if val_df is not None:
                future_pdf = self.prepare_data(val_df, future=True)
                union_pdf = pd.concat(
                    [hist_pdf, future_pdf], axis=0, join="outer", ignore_index=True
                ).sort_values(by=["unique_id", "ds"])
            else:
                union_pdf = hist_pdf.sort_values(by=["unique_id", "ds"])

            cov_pdf = union_pdf[["unique_id"] + dynamic_vars]
            cov_sdf = spark.createDataFrame(cov_pdf)
            cov_agg = [collect_list(v).alias(v) for v in dynamic_vars]
            cov_sdf = cov_sdf.groupBy("unique_id").agg(*cov_agg)
            core_sdf = core_sdf.join(cov_sdf, on="unique_id", how="left")

        if static_vars:
            static_pdf = hist_df[
                [self.params.group_id] + static_vars
            ].drop_duplicates(subset=[self.params.group_id])
            static_pdf = static_pdf.rename(
                columns={self.params.group_id: "unique_id"}
            )
            static_sdf = spark.createDataFrame(static_pdf)
            core_sdf = core_sdf.join(static_sdf, on="unique_id", how="left")

        return core_sdf

    def predict(self,
                hist_df: pd.DataFrame,
                val_df: pd.DataFrame = None,
                curr_date=None,
                spark=None):
        if spark is not None:
            return self._predict_distributed(hist_df, spark, val_df)
        else:
            return self._predict_single(hist_df, val_df)

    def _predict_distributed(self, hist_df: pd.DataFrame, spark,
                             val_df: pd.DataFrame = None):
        """Distributed prediction using Spark pandas UDFs across GPUs."""
        dyn_num_vars = list(self.params.get("dynamic_future_numerical", []))
        dyn_cat_vars = list(self.params.get("dynamic_future_categorical", []))
        static_vars = list(self.params.get("static_features", []))

        stat_num_vars = []
        stat_cat_vars = []
        for var in static_vars:
            if pd.api.types.is_numeric_dtype(hist_df[var]):
                stat_num_vars.append(var)
            else:
                stat_cat_vars.append(var)

        col_map = []
        for var in dyn_num_vars:
            col_map.append(("dyn_num", var))
        for var in dyn_cat_vars:
            col_map.append(("dyn_cat", var))
        for var in stat_num_vars:
            col_map.append(("stat_num", var))
        for var in stat_cat_vars:
            col_map.append(("stat_cat", var))

        hist_sdf = self.prepare_data_for_spark(hist_df, spark, val_df)
        horizon_timestamps_udf = self.create_horizon_timestamps_udf()
        device_count = torch.cuda.device_count()
        if device_count == 0:
            device_count = 1
        forecast_udf = self.create_predict_udf(device_count, col_map)

        udf_columns = [hist_sdf.y]
        for _, var in col_map:
            udf_columns.append(hist_sdf[var])

        forecast_df = (
            hist_sdf.repartition(device_count, "unique_id")
            .select(
                hist_sdf.unique_id,
                horizon_timestamps_udf(hist_sdf.ds).alias("ds"),
                forecast_udf(*udf_columns).alias("y"),
            )
        ).toPandas()

        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                "y": self.params.target,
            }
        )
        return forecast_df, self.model

    def _predict_single(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Single-GPU prediction path that builds Chronos-2 list-of-dicts inputs."""
        from chronos import BaseChronosPipeline

        pipeline = BaseChronosPipeline.from_pretrained(
            self.repo,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
        )
        quantile_idx = pipeline.quantiles.index(0.5)

        hist_pdf = self.prepare_data(hist_df)
        future_pdf = self.prepare_data(val_df, future=True) if val_df is not None else None

        dyn_num_vars = list(self.params.get("dynamic_future_numerical", []))
        dyn_cat_vars = list(self.params.get("dynamic_future_categorical", []))
        static_vars = list(self.params.get("static_features", []))
        dynamic_vars = dyn_num_vars + dyn_cat_vars
        has_covariates = bool(dynamic_vars or static_vars)

        if has_covariates and future_pdf is not None:
            union_pdf = pd.concat(
                [hist_pdf, future_pdf], axis=0, join="outer", ignore_index=True
            ).sort_values(by=["unique_id", "ds"])
        else:
            union_pdf = hist_pdf.sort_values(by=["unique_id", "ds"])

        unique_ids = hist_pdf["unique_id"].unique()
        inputs = []
        for uid in unique_ids:
            hist_group = hist_pdf[hist_pdf["unique_id"] == uid].sort_values("ds")
            target_arr = hist_group["y"].values.astype(np.float64)

            if not has_covariates:
                inputs.append(torch.tensor(target_arr))
                continue

            past_covariates = {}
            future_covariates = {}

            union_group = union_pdf[union_pdf["unique_id"] == uid].sort_values("ds")
            hist_len = len(hist_group)

            for var in dynamic_vars:
                full_arr = union_group[var].values
                past_covariates[var] = full_arr[:hist_len]
                if future_pdf is not None and len(full_arr) > hist_len:
                    future_covariates[var] = full_arr[hist_len:]

            for var in static_vars:
                val = hist_group[var].iloc[0]
                past_covariates[var] = np.full(hist_len, val)

            entry = {"target": target_arr, "past_covariates": past_covariates}
            if future_covariates:
                entry["future_covariates"] = future_covariates
            inputs.append(entry)

        forecasts = pipeline.predict(
            inputs=inputs,
            prediction_length=self.params["prediction_length"],
        )

        from utilsforecast.processing import make_future_dataframe
        timestamps = make_future_dataframe(
            uids=unique_ids,
            last_times=hist_pdf.groupby("unique_id")["ds"].tail(1),
            h=self.params["prediction_length"],
            freq=self.params["freq"],
        )
        timestamps = [
            group["ds"].values
            for _, group in timestamps.groupby("unique_id")
        ]

        medians = []
        for forecast in forecasts:
            medians.append(forecast[0, quantile_idx, :].numpy().astype(np.float64))

        forecast_df = pd.DataFrame({
            self.params.group_id: unique_ids,
            self.params.date_col: timestamps,
            self.params.target: medians,
        })
        return forecast_df, None

    def forecast(self, df: pd.DataFrame, spark=None):
        hist_df = df[df[self.params.target].notnull()]
        last_date = hist_df[self.params.date_col].max()
        future_df = df[
            (df[self.params.date_col] > np.datetime64(last_date))
            & (df[self.params.date_col]
               <= np.datetime64(last_date + self.prediction_length_offset))
        ]
        forecast_df, model = self.predict(
            hist_df=hist_df, val_df=future_df, spark=spark
        )
        return forecast_df, model

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> list:
        pred_df, model_pretrained = self.predict(hist_df, val_df, spark=spark)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        metric_name = self.params["metric"]
        if metric_name not in ("smape", "mape", "mae", "mse", "rmse"):
            raise UnsupportedMetricError(f"Metric {self.params['metric']} not supported!")
        for key in keys:
            actual = val_df[val_df[self.params["group_id"]] == key][self.params["target"]].to_numpy()
            forecast = np.array(
                pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].iloc[0]
            )
            metric_classes = {
                "smape": MeanAbsolutePercentageError(symmetric=True),
                "mape": MeanAbsolutePercentageError(symmetric=False),
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(square_root=False),
                "rmse": MeanSquaredError(square_root=True),
            }
            try:
                if metric_name in metric_classes:
                    metric_function = metric_classes[metric_name]
                    metric_value = metric_function(actual, forecast)
                metrics.extend(
                    [(
                        key,
                        curr_date,
                        metric_name,
                        metric_value,
                        forecast,
                        actual,
                        b'',
                    )])
            except (ModelPredictionError, DataPreparationError) as err:
                _logger.warning(f"Failed to calculate metric for key {key}: {err}")
            except Exception as err:
                _logger.warning(f"Unexpected error calculating metric for key {key}: {err}")
        return metrics

    def create_predict_udf(self, device_count, col_map=None):
        """Create a pandas UDF for distributed Chronos-2 inference.

        Parameters
        ----------
        device_count : int
            Number of GPUs available.
        col_map : list[tuple[str, str]] or None
            Ordered mapping of covariate columns after the leading ``y``
            column.  Each entry is ``(category, variable_name)`` where
            *category* is ``"dyn_num"``, ``"dyn_cat"``, ``"stat_num"``,
            or ``"stat_cat"``.
        """
        repo = self.repo
        prediction_length = self.params["prediction_length"]
        batch_size = self.params["batch_size"]
        num_devices = device_count
        _col_map = list(col_map) if col_map else []
        _has_covariates = len(_col_map) > 0

        @pandas_udf('array<double>')
        def predict_udf(
            bulk_iterator: Iterator[Tuple[pd.Series, ...]]
        ) -> Iterator[pd.Series]:
            import torch
            import numpy as np
            import pandas as pd
            from pyspark import TaskContext

            ctx = TaskContext.get()
            partition_id = ctx.partitionId() if ctx else 0
            gpu_id = partition_id % num_devices
            torch.cuda.set_device(gpu_id)

            from chronos import BaseChronosPipeline
            pipeline = BaseChronosPipeline.from_pretrained(
                repo,
                device_map=f"cuda:{gpu_id}",
                dtype=torch.bfloat16,
            )
            quantile_idx = pipeline.quantiles.index(0.5)

            for batch_tuple in bulk_iterator:
                if not isinstance(batch_tuple, tuple):
                    batch_tuple = (batch_tuple,)

                y_batch = batch_tuple[0]
                n = len(y_batch)
                median = []

                for i in range(0, n, batch_size):
                    end = min(i + batch_size, n)

                    if not _has_covariates:
                        contexts = [
                            torch.tensor(list(y_batch.iloc[j]))
                            for j in range(i, end)
                        ]
                        forecasts = pipeline.predict(
                            inputs=contexts,
                            prediction_length=prediction_length,
                        )
                    else:
                        inputs = []
                        for j in range(i, end):
                            target_arr = np.array(
                                list(y_batch.iloc[j]), dtype=np.float64
                            )
                            hist_len = len(target_arr)
                            past_cov = {}
                            future_cov = {}

                            for idx, (category, name) in enumerate(_col_map):
                                col_batch = batch_tuple[1 + idx]
                                raw = list(col_batch.iloc[j])

                                if category in ("dyn_num", "dyn_cat"):
                                    full_arr = np.array(raw)
                                    if category == "dyn_num":
                                        full_arr = full_arr.astype(np.float64)
                                    past_cov[name] = full_arr[:hist_len]
                                    if len(full_arr) > hist_len:
                                        future_cov[name] = full_arr[hist_len:]
                                elif category in ("stat_num", "stat_cat"):
                                    val = raw
                                    if category == "stat_num":
                                        past_cov[name] = np.full(
                                            hist_len, float(val), dtype=np.float64
                                        )
                                    else:
                                        past_cov[name] = np.full(
                                            hist_len, str(val)
                                        )

                            entry = {
                                "target": target_arr,
                                "past_covariates": past_cov,
                            }
                            if future_cov:
                                entry["future_covariates"] = future_cov
                            inputs.append(entry)

                        forecasts = pipeline.predict(
                            inputs=inputs,
                            prediction_length=prediction_length,
                        )

                    for forecast in forecasts:
                        median.append(
                            forecast[0, quantile_idx, :].numpy().astype(np.float64)
                        )

                yield pd.Series(median)

        return predict_udf


class Chronos2(Chronos2Forecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-2"


class Chronos2Small(Chronos2Forecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "autogluon/chronos-2-small"


class Chronos2Synth(Chronos2Forecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "autogluon/chronos-2-synth"


class ChronosModel(mlflow.pyfunc.PythonModel):
    def __init__(self, repo, prediction_length):
        import torch
        self.repo = repo
        self.prediction_length = prediction_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from chronos import BaseChronosPipeline
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.repo,
            device_map='cuda',
            dtype=torch.bfloat16,
        )

    def predict(self, context, input_data, params=None):
        history = [torch.tensor(list(series)) for series in input_data]
        forecast = self.pipeline.predict(
            inputs=history,
            prediction_length=self.prediction_length,
        )
        return forecast.numpy()


class Chronos2MLflowModel(mlflow.pyfunc.PythonModel):
    def __init__(self, repo, prediction_length):
        import torch
        self.repo = repo
        self.prediction_length = prediction_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from chronos import BaseChronosPipeline
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.repo,
            device_map='cuda',
            dtype=torch.bfloat16,
        )

    def predict(self, context, input_data, params=None):
        history = [torch.tensor(list(series)) for series in input_data]
        forecasts = self.pipeline.predict(
            inputs=history,
            prediction_length=self.prediction_length,
        )
        # forecasts is a list of tensors, each of shape
        # (n_variates, n_quantiles, prediction_length)
        # Extract median (0.5) quantile for univariate forecasting
        quantile_idx = self.pipeline.quantiles.index(0.5)
        results = np.stack([f[0, quantile_idx, :].numpy() for f in forecasts])
        return results
