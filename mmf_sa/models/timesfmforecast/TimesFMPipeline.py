from abc import ABC
import subprocess
import sys
import pandas as pd
import torch
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
from mmf_sa.models.abstract_model import ForecastingRegressor


class TimesFMForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = None
        self.model = None
        self.install("git+https://github.com/google-research/timesfm.git")

    def install(self, package: str):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

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
        return df

    def predict(self,
                hist_df: pd.DataFrame,
                val_df: pd.DataFrame = None,
                curr_date=None,
                spark=None):

        hist_df = self.prepare_data(hist_df)
        self.model.load_from_checkpoint(repo_id=self.repo)
        forecast_df = self.model.forecast_on_df(
            inputs=hist_df,
            freq=self.params["freq"],
            value_name="y",
            num_jobs=-1,
        )
        forecast_df = forecast_df[["unique_id", "ds", "timesfm"]]
        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                "timesfm": self.params.target,
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
        pred_df, model_pretrained = self.predict(hist_df, val_df, curr_date)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        if self.params["metric"] == "smape":
            metric_name = "smape"
        elif self.params["metric"] == "mape":
            metric_name = "mape"
        elif self.params["metric"] == "mae":
            metric_name = "mae"
        elif self.params["metric"] == "mse":
            metric_name = "mse"
        elif self.params["metric"] == "rmse":
            metric_name = "rmse"
        else:
            raise Exception(f"Metric {self.params['metric']} not supported!")
        for key in keys:
            actual = val_df[val_df[self.params["group_id"]] == key][self.params["target"]].to_numpy()
            forecast = pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].to_numpy()[0]
            try:
                if metric_name == "smape":
                    smape = MeanAbsolutePercentageError(symmetric=True)
                    metric_value = smape(actual, forecast)
                elif metric_name == "mape":
                    mape = MeanAbsolutePercentageError(symmetric=False)
                    metric_value = mape(actual, forecast)
                elif metric_name == "mae":
                    mae = MeanAbsoluteError()
                    metric_value = mae(actual, forecast)
                elif metric_name == "mse":
                    mse = MeanSquaredError(square_root=False)
                    metric_value = mse(actual, forecast)
                elif metric_name == "rmse":
                    rmse = MeanSquaredError(square_root=True)
                    metric_value = rmse(actual, forecast)
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
        self.params = params
        self.backend = "gpu" if torch.cuda.is_available() else "cpu"
        self.repo = "google/timesfm-1.0-200m"
        import timesfm
        self.model = timesfm.TimesFm(
            context_len=512,
            horizon_len=10,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=self.backend,
        )

