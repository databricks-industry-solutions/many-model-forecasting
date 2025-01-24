import pandas as pd
import numpy as np
import torch
import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
from utilsforecast.processing import make_future_dataframe
from mmf_sa.models.abstract_model import ForecastingRegressor


class TimesFMForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = None
        self.model = None
        self.repo = None

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
                "timesfm[torch]==1.2.7",
                "git+https://github.com/databricks-industry-solutions/many-model-forecasting.git",
                "pyspark==3.5.0",
            ],
        )

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> pd.DataFrame:
        df = df[[self.params.group_id, self.params.date_col, self.params.target]]
        df = (
            df.rename(
                columns={
                    self.params.group_id: "unique_id",
                    self.params.date_col: "ds",
                    self.params.target: "y",
                }
            )
        )
        return df.sort_values(by=["unique_id", "ds"])

    def predict(self,
                hist_df: pd.DataFrame,
                val_df: pd.DataFrame = None,
                curr_date=None,
                spark=None):

        hist_df = self.prepare_data(hist_df)

        # Need if-else here
        forecast_input = [group['y'].values for _, group in hist_df.groupby('unique_id')]
        unique_ids = np.array(hist_df["unique_id"].unique())
        timestamps = make_future_dataframe(
            uids=unique_ids,
            last_times=hist_df.groupby("unique_id")["ds"].tail(1),
            h=self.params.prediction_length,
            freq=self.params.freq,
        )
        timestamps = [group['ds'].values for _, group in timestamps.groupby('unique_id')]
        freq_index = 0 if self.params.freq in ("H", "D", "B", "U") else 1
        forecasts, _ = self.model.forecast(forecast_input, freq=[freq_index] * len(forecast_input))
        forecast_df = pd.DataFrame({
            self.params.group_id: unique_ids,
            self.params.date_col: timestamps,
            self.params.target: list(forecasts.astype(np.float64)),
        })
        # Todo
        # forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame, spark=None):
        return self.predict(df, spark=spark)

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> list:
        pred_df, model_pretrained = self.predict(hist_df, val_df, curr_date)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        metric_name = self.params["metric"]
        if metric_name not in ("smape", "mape", "mae", "mse", "rmse"):
            raise Exception(f"Metric {self.params['metric']} not supported!")
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
            except:
                pass
        return metrics


class TimesFM_1_0_200m(TimesFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        import timesfm
        self.params = params
        #self.backend = "gpu" if torch.cuda.is_available() else "cpu"
        self.repo = "google/timesfm-1.0-200m-pytorch"
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=self.params.prediction_length,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.repo
            ),
        )

class TimesFM_2_0_500m(TimesFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        import timesfm
        self.params = params
        #self.backend = "gpu" if torch.cuda.is_available() else "cpu"
        self.repo = "google/timesfm-2.0-500m-pytorch"
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