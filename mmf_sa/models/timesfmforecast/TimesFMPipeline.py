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
from typing import Iterator
from pyspark.sql.functions import collect_list, pandas_udf
from pyspark.sql import DataFrame
from utilsforecast.processing import make_future_dataframe
from mmf_sa.models.abstract_model import ForecastingRegressor
from mmf_sa.exceptions import MissingFeatureError, UnsupportedMetricError, ModelPredictionError, DataPreparationError

_logger = logging.getLogger(__name__)


class TimesFMForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = None
        self.model = None
        self.repo = None
        self.model_hparams = {}

    def _get_or_create_model(self):
        """Lazy-load the TimesFM model on the driver (for single-GPU fallback with covariates)."""
        if self.model is None:
            import timesfm
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                    horizon_len=self.params.prediction_length,
                    **self.model_hparams,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.repo,
                ),
            )
        return self.model

    def register(self, registered_model_name: str):
        pipeline = TimesFMModel(self.params, self.repo)
        input_schema = Schema([TensorSpec(np.dtype(np.double), (-1, -1))])
        output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, -1))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pyfunc.log_model(
            "model",
            python_model=pipeline,
            registered_model_name=registered_model_name,
            signature=signature,
            #input_example=input_example,
            pip_requirements=[
                "timesfm[torch]==1.2.6",
                "git+https://github.com/databricks-industry-solutions/many-model-forecasting.git",
                "pyspark==3.5.0",
            ],
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

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> pd.DataFrame:
        """Prepare data as pandas DataFrame (column selection and renaming).
        Used by the single-GPU fallback path for covariate handling."""
        if not future:
            # Prepare historical dataframe with or without exogenous regressors for training
            features = [self.params.group_id, self.params.date_col, self.params.target]
            if 'dynamic_future_numerical' in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_numerical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future numerical missing: {e}")
            if 'dynamic_future_categorical' in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_categorical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future categorical missing: {e}")
            _df = df[features]
            _df = (
                _df.rename(
                    columns={
                        self.params.group_id: "unique_id",
                        self.params.date_col: "ds",
                        self.params.target: "y",
                    }
                )
            )
        else:
            # Prepare future dataframe with exogenous regressors for forecasting
            features = [self.params.group_id, self.params.date_col]
            if 'dynamic_future_numerical' in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_numerical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future numerical missing: {e}")
            if 'dynamic_future_categorical' in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_categorical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future categorical missing: {e}")
            _df = df[features]
            _df = (
                _df.rename(
                    columns={
                        self.params.group_id: "unique_id",
                        self.params.date_col: "ds",
                    }
                )
            )
        return _df.sort_values(by=["unique_id", "ds"])

    def prepare_data_for_spark(self, df: pd.DataFrame, spark) -> DataFrame:
        """Convert pandas DataFrame to Spark DataFrame grouped by unique_id with collected lists.
        Used by the distributed multi-GPU prediction path."""
        features = [self.params.group_id, self.params.date_col, self.params.target]
        _df = df[features].rename(columns={
            self.params.group_id: "unique_id",
            self.params.date_col: "ds",
            self.params.target: "y",
        }).sort_values(by=["unique_id", "ds"])

        sdf = spark.createDataFrame(_df)
        sdf = (
            sdf.groupBy("unique_id")
            .agg(
                collect_list("ds").alias("ds"),
                collect_list("y").alias("y"),
            )
        )
        return sdf

    def create_predict_udf(self, device_count):
        """Create a pandas UDF that loads the TimesFM model inside each Spark worker
        and runs distributed inference across multiple GPUs."""
        repo = self.repo
        prediction_length = self.params["prediction_length"]
        freq = self.params["freq"]
        model_hparams = self.model_hparams
        num_devices = device_count

        @pandas_udf('array<double>')
        def predict_udf(bulk_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            import os
            import torch
            import timesfm
            import numpy as np
            import pandas as pd
            from pyspark import TaskContext

            # Assign this worker to a specific GPU based on partition ID
            ctx = TaskContext.get()
            partition_id = ctx.partitionId() if ctx else 0
            gpu_id = partition_id % num_devices
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)  # Device 0 within the visible set

            # Load the model on the assigned GPU
            model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                    horizon_len=prediction_length,
                    **model_hparams,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=repo,
                ),
            )

            freq_index = 0 if freq in ("H", "D") else 1

            results = []
            for bulk in bulk_iterator:
                inputs = [np.array(list(series), dtype=np.float64) for series in bulk]
                forecasts, _ = model.forecast(
                    inputs=inputs,
                    freq=[freq_index] * len(inputs),
                )
                results.extend([arr.astype(np.float64) for arr in forecasts])
            yield pd.Series(results)

        return predict_udf

    def predict(self,
                hist_df: pd.DataFrame,
                val_df: pd.DataFrame = None,
                spark=None):
        """Route to distributed or single-GPU prediction based on available context."""
        has_covariates = (
            'dynamic_future_numerical' in self.params.keys()
            or 'dynamic_future_categorical' in self.params.keys()
            or 'static_features' in self.params.keys()
        )

        if spark is not None and not has_covariates:
            return self._predict_distributed(hist_df, spark)
        else:
            return self._predict_single(hist_df, val_df)

    def _predict_distributed(self, hist_df: pd.DataFrame, spark):
        """Distributed prediction using Spark pandas UDFs across multiple GPUs.
        Each partition explicitly targets a specific GPU via CUDA_VISIBLE_DEVICES."""
        hist_sdf = self.prepare_data_for_spark(hist_df, spark)
        horizon_timestamps_udf = self.create_horizon_timestamps_udf()
        device_count = torch.cuda.device_count()
        if device_count == 0:
            device_count = 1
        forecast_udf = self.create_predict_udf(device_count)

        forecast_df = (
            hist_sdf.repartition(device_count, "unique_id")
            .select(
                hist_sdf.unique_id,
                horizon_timestamps_udf(hist_sdf.ds).alias("ds"),
                forecast_udf(hist_sdf.y).alias("y"),
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
        """Single-GPU prediction (fallback for covariate case or when spark is unavailable)."""
        model = self._get_or_create_model()
        df = self.prepare_data(hist_df)
        dynamic_covariates = self.prepare_data(val_df, future=True) if val_df is not None else None

        if dynamic_covariates is not None:
            df_union = pd.concat([df, dynamic_covariates], axis=0, join='outer', ignore_index=True)
        else:
            df_union = df

        forecast_input = [group['y'].values for _, group in df.groupby('unique_id')]
        freq_index = 0 if self.params.freq in ("H", "D") else 1
        dynamic_numerical_covariates = {}
        if 'dynamic_future_numerical' in self.params.keys():
            for var in self.params.dynamic_future_numerical:
                dynamic_numerical_covariates[var] = [group[var].values for _, group in df_union.groupby('unique_id')]
        dynamic_categorical_covariates = {}
        if 'dynamic_future_categorical' in self.params.keys():
            for var in self.params.dynamic_future_categorical:
                dynamic_categorical_covariates[var] = [group[var].values for _, group in df_union.groupby('unique_id')]
        static_numerical_covariates = {}
        static_categorical_covariates = {}
        if 'static_features' in self.params.keys():
            for var in self.params.static_features:
                if pd.api.types.is_numeric_dtype(df[var]):
                    static_numerical_covariates[var] = [group[var].iloc[0] for _, group in df.groupby('unique_id')]
                else:
                    static_categorical_covariates[var] = [group[var].iloc[0] for _, group in df.groupby('unique_id')]
        if not dynamic_numerical_covariates \
            and not dynamic_categorical_covariates \
            and not static_numerical_covariates \
            and not static_categorical_covariates:
            forecasts, _ = model.forecast(
                inputs=forecast_input,
                freq=[freq_index] * len(forecast_input)
            )
        else:
            forecasts, _ = model.forecast_with_covariates(
                inputs=forecast_input,
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                dynamic_categorical_covariates=dynamic_categorical_covariates,
                static_numerical_covariates=static_numerical_covariates,
                static_categorical_covariates=static_categorical_covariates,
                freq=[freq_index] * len(forecast_input),
                xreg_mode="xreg + timesfm",  # default
                ridge=0.0,
                force_on_cpu=False,
                normalize_xreg_target_per_input=True,  # default
            )

        unique_ids = np.array(df["unique_id"].unique())
        timestamps = make_future_dataframe(
            uids=unique_ids,
            last_times=df.groupby("unique_id")["ds"].tail(1),
            h=self.params.prediction_length,
            freq=self.params.freq,
        )
        timestamps = [group['ds'].values for _, group in timestamps.groupby('unique_id')]
        forecast_df = pd.DataFrame({
            self.params.group_id: unique_ids,
            self.params.date_col: timestamps,
            self.params.target: [arr.astype(np.float64) for arr in forecasts],
        })

        # Todo
        # forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df, model

    def forecast(self, df: pd.DataFrame, spark=None):
        hist_df = df[df[self.params.target].notnull()]
        last_date = hist_df[self.params.date_col].max()
        future_df = df[
            (df[self.params.date_col] > np.datetime64(last_date))
            & (df[self.params.date_col]
               <= np.datetime64(last_date + self.prediction_length_offset))
            ]
        forecast_df, model = self.predict(hist_df=hist_df, val_df=future_df, spark=spark)
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
            forecast = np.array(pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].iloc[0])
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


class TimesFM_1_0_200m(TimesFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "google/timesfm-1.0-200m-pytorch"
        self.model_hparams = {}


class TimesFM_2_0_500m(TimesFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "google/timesfm-2.0-500m-pytorch"
        self.model_hparams = {
            "num_layers": 50,
            "use_positional_embedding": False,
            "context_len": 2048,
        }


class TimesFMModel(mlflow.pyfunc.PythonModel):
    def __init__(self, params, repo):
        import timesfm
        self.params = params
        self.repo = repo
        #self.backend = "gpu" if torch.cuda.is_available() else "cpu"
        if self.repo == "google/timesfm-1.0-200m-pytorch":
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                    horizon_len=self.params.prediction_length,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.repo,
                ),
            )
        else:
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                    horizon_len=self.params.prediction_length,
                    num_layers=50,
                    use_positional_embedding=False,
                    context_len=2048,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.repo
                ),
        )

    def predict(self, context, input_df, params=None):
        # Generate forecasts on the input DataFrame
        forecast_df = self.model.forecast_on_df(
            inputs=input_df,  # Input DataFrame containing the time series data.
            freq=self.params.freq,  # Frequency of the time series data, set to daily.
            value_name=self.params.target,  # Column name in the DataFrame containing the values to forecast.
            num_jobs=-1,  # Number of parallel jobs to run, set to -1 to use all available processors.
        )
        return forecast_df  # Return the forecast DataFrame
