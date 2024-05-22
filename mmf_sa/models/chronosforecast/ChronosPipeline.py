import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
from sktime.performance_metrics.forecasting import
from typing import Iterator
from pyspark.sql.functions import collect_list, pandas_udf
from mmf_sa.models.abstract_model import ForecastingRegressor


class ChronosForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = None
        self.model = None

    def prepare_data(self, df: pd.DataFrame, future: bool = False) -> pd.DataFrame:
        context = (
            df.rename(
                columns={
                    self.params.group_id: "unique_id",
                    self.params.date_col: "ds",
                    self.params.target: "y",
                }
            )
        )

        torch.tensor(_df[self.params.target])

        return context

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        # context must be either a 1D tensor, a list of 1D tensors,
        # or a left-padded 2D tensor with batch as the first dimension
        # forecast shape: [num_series, num_samples, prediction_length]
        hist_df = self.spark
        context = self.prepare_data(hist_df)
        forecast_df = self.model.predict(
            context=context,
            prediction_length=self.params["prediction_length"],
            num_samples=self.params["num_samples"],
        )

        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                target: self.params.target,
            }
        )
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)

        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame):
        return self.predict(df)

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date
    ) -> list:

        print(f"hist_df: {hist_df}")
        print(f"val_df: {val_df}")
        pred_df, model_fitted = self.predict(hist_df, val_df)

        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        if self.params["metric"] == "smape":
            metric_name = "smape"
        else:
            raise Exception(f"Metric {self.params['metric']} not supported!")
        for key in keys:
            actual = val_df[val_df[self.params["group_id"]] == key][self.params["target"]]
            forecast = pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].\
                         iloc[-self.params["prediction_length"]:]
            try:
                if metric_name == "smape":
                    metric_value = mean_absolute_percentage_error(actual, forecast, symmetric=True)
                metrics.extend(
                    [(
                        key,
                        curr_date,
                        metric_name,
                        metric_value,
                        actual.to_numpy(),
                        forecast.to_numpy(),
                        b'',
                    )])
            except:
                pass
        return metrics


class ChronosT5Large(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map=self.device,  # use "cuda" for GPU and "cpu" for CPU inference
            torch_dtype=torch.bfloat16,
        )
