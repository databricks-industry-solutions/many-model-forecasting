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
from mmf_sa.models.abstract_model import ForecastingRegressor
from mmf_sa.exceptions import UnsupportedMetricError, ModelPredictionError, DataPreparationError

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
            pip_requirements=[  # List of pip requirements
                "torch==2.3.1",
                "torchvision==0.18.1",
                "transformers==4.41.2",
                "cloudpickle==2.2.1",
                "chronos-forecasting==1.4.1",
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
        forecast_udf = self.create_predict_udf()
        device_count = torch.cuda.device_count()
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

        # Use pure numpy metric implementations to avoid sktime/sklearn version incompatibilities.
        def _to_1d(a):
            return np.asarray(a, dtype=float).reshape(-1)

        def _smape(y_true, y_pred, eps=1e-8):
            denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
            return float(np.mean(200.0 * np.abs(y_pred - y_true) / denom))

        def _mape(y_true, y_pred, eps=1e-8):
            denom = np.maximum(np.abs(y_true), eps)
            return float(np.mean(100.0 * np.abs((y_true - y_pred) / denom)))

        def _mae(y_true, y_pred):
            return float(np.mean(np.abs(y_true - y_pred)))

        def _mse(y_true, y_pred):
            err = y_true - y_pred
            return float(np.mean(err * err))

        def _rmse(y_true, y_pred):
            return float(np.sqrt(_mse(y_true, y_pred)))

        for key in keys:
            # Align actuals to the forecast horizon. Some calling paths provide a longer val_df
            # than prediction_length, which breaks metric computation.
            _val = val_df[val_df[self.params["group_id"]] == key]
            if self.params.get("date_col") in _val.columns:
                _val = _val.sort_values(by=[self.params["date_col"]])
            actual = _val[self.params["target"]].to_numpy()

            forecast = pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].to_numpy()[0]
            forecast = np.asarray(forecast, dtype=float).reshape(-1)

            if len(actual) > len(forecast):
                actual = actual[-len(forecast):]
            elif len(forecast) > len(actual):
                forecast = forecast[-len(actual):]

            actual_1d = _to_1d(actual)
            forecast_1d = _to_1d(forecast)
            try:
                if metric_name == "smape":
                    metric_value = _smape(actual_1d, forecast_1d)
                elif metric_name == "mape":
                    metric_value = _mape(actual_1d, forecast_1d)
                elif metric_name == "mae":
                    metric_value = _mae(actual_1d, forecast_1d)
                elif metric_name == "mse":
                    metric_value = _mse(actual_1d, forecast_1d)
                elif metric_name == "rmse":
                    metric_value = _rmse(actual_1d, forecast_1d)
                else:
                    raise UnsupportedMetricError(f"Metric {metric_name} not supported!")
                metrics.extend(
                    [(
                        key,
                        curr_date,
                        metric_name,
                        metric_value,
                        forecast_1d,
                        actual_1d,
                        b'',
                    )])
            except (ModelPredictionError, DataPreparationError) as err:
                _logger.warning(f"Failed to calculate metric for key {key}: {err}")
            except Exception as err:
                _logger.warning(f"Unexpected error calculating metric for key {key}: {err}")
        return metrics

    def create_predict_udf(self):
        @pandas_udf('array<double>')
        def predict_udf(bulk_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            # initialization step
            import torch
            import numpy as np
            import pandas as pd
            # Initialize the ChronosPipeline with a pretrained model from the specified repository
            from chronos import BaseChronosPipeline
            pipeline = BaseChronosPipeline.from_pretrained(
                self.repo,
                device_map='cuda',
                torch_dtype=torch.bfloat16,
            )

            # inference
            median = []
            for bulk in bulk_iterator:
                for i in range(0, len(bulk), self.params["batch_size"]):
                    batch = bulk[i:i+self.params["batch_size"]]
                    contexts = [torch.tensor(list(series)) for series in batch]
                    forecasts = pipeline.predict(
                        context=contexts,
                        prediction_length=self.params["prediction_length"],
                    )
                    median.extend([np.median(forecast, axis=0) for forecast in forecasts])
            yield pd.Series(median)
        return predict_udf


class ChronosT5Tiny(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-t5-tiny"


class ChronosT5Mini(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-t5-mini"


class ChronosT5Small(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-t5-small"


class ChronosT5Base(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-t5-base"


class ChronosT5Large(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-t5-large"


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


class Chronos2(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-2"
    
    def register(self, registered_model_name: str):
        """Override register to use Chronos2Model with correct API."""
        pipeline = Chronos2Model(
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
            pip_requirements=[
                "torch==2.3.1",
                "torchvision==0.18.1",
                "transformers==4.41.2",
                "cloudpickle==2.2.1",
                "chronos-forecasting>=2.0",  # Chronos-2 requires v2.0+
                "git+https://github.com/databricks-industry-solutions/many-model-forecasting.git",
                "pyspark==3.5.0",
            ],
        )
    
    def create_predict_udf(self):
        @pandas_udf('array<double>')
        def predict_udf(bulk_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            # initialization step
            import torch
            import numpy as np
            import pandas as pd
            from chronos import Chronos2Pipeline
            
            # Initialize Chronos2Pipeline (per official docs: https://huggingface.co/amazon/chronos-2)
            pipeline = Chronos2Pipeline.from_pretrained(
                self.repo,
                device_map='cuda',
            )

            # inference
            median = []
            for bulk in bulk_iterator:
                for i in range(0, len(bulk), self.params["batch_size"]):
                    batch = bulk[i:i+self.params["batch_size"]]
                    contexts = [torch.tensor(list(series)) for series in batch]
                    # Chronos2 uses 'inputs' parameter (not 'context')
                    forecasts = pipeline.predict(
                        inputs=contexts,
                        prediction_length=self.params["prediction_length"],
                    )
                    # Arrow is extremely picky here: each row must be a *flat* list[python-float]
                    # (no numpy arrays, no numpy scalars, no nested lists).
                    for f in forecasts:
                        # Convert tensor-like outputs to numpy
                        if hasattr(f, "detach"):
                            f = f.detach()
                        if hasattr(f, "cpu"):
                            f = f.cpu()
                        if hasattr(f, "numpy"):
                            f = f.numpy()

                        arr = np.asarray(f)
                        # Common shapes:
                        # - (num_samples, prediction_length) -> take median over samples axis
                        # - (prediction_length,) -> already 1D
                        # - (q, prediction_length) -> median over q axis (still gives prediction_length)
                        if arr.ndim >= 2:
                            arr = np.median(arr, axis=0)
                        arr = np.asarray(arr, dtype=np.float64).reshape(-1)

                        row = arr.tolist()
                        # If tolist() still produced nested lists for any reason, flatten once.
                        while len(row) > 0 and isinstance(row[0], list):
                            row = row[0]
                        # Ensure pure Python floats (not numpy scalar types).
                        row = [float(x) for x in row]
                        median.append(row)
            yield pd.Series(median)
        return predict_udf


class ChronosModel(mlflow.pyfunc.PythonModel):
    """MLflow model for Chronos 1.x (T5 and Bolt models)."""
    def __init__(self, repo, prediction_length):
        import torch
        self.repo = repo
        self.prediction_length = prediction_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize the BaseChronosPipeline with a pretrained model from the specified repository
        from chronos import BaseChronosPipeline
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.repo,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
        )

    def predict(self, context, model_input, params=None):
        # MLflow pyfunc prefers signature: (context, model_input, params=None)
        input_data = model_input
        history = [torch.tensor(list(series)) for series in input_data]
        forecast = self.pipeline.predict(
            context=history,
            prediction_length=self.prediction_length,
        )
        return forecast.numpy()


class Chronos2Model(mlflow.pyfunc.PythonModel):
    """MLflow model for Chronos-2 (uses Chronos2Pipeline with 'inputs' parameter)."""
    def __init__(self, repo, prediction_length):
        import torch
        self.repo = repo
        self.prediction_length = prediction_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize Chronos2Pipeline (per official docs: https://huggingface.co/amazon/chronos-2)
        from chronos import Chronos2Pipeline
        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.repo,
            device_map='cuda',
        )

    def predict(self, context, model_input, params=None):
        """
        MLflow pyfunc predict signature should use 'model_input' (not 'input_data') to avoid warnings.
        We accept model_input as array-like [n_series, context_length] and return numpy array.
        """
        import numpy as np
        import torch

        input_data = model_input
        history = [torch.tensor(list(series)) for series in input_data]
        # Chronos2 uses 'inputs' parameter instead of 'context'
        forecast = self.pipeline.predict(
            inputs=history,
            prediction_length=self.prediction_length,
        )

        # forecast can be tensor / numpy array / list depending on chronos-forecasting version
        if hasattr(forecast, "detach"):
            forecast = forecast.detach()
        if hasattr(forecast, "cpu"):
            forecast = forecast.cpu()
        if hasattr(forecast, "numpy"):
            forecast = forecast.numpy()
        return np.asarray(forecast)
