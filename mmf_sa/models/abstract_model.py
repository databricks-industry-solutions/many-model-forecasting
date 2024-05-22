from abc import abstractmethod
import numpy as np
import pandas as pd
import cloudpickle
from typing import Dict, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


class ForecastingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, params):
        self.params = params
        self.freq = params["freq"].upper()[0]
        self.one_ts_offset = (
            pd.offsets.MonthEnd(1) if self.freq == "M" else pd.DateOffset(days=1)
        )
        self.prediction_length_offset = (
            pd.offsets.MonthEnd(params["prediction_length"])
            if self.freq == "M"
            else pd.DateOffset(days=params["prediction_length"])
        )

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @abstractmethod
    def fit(self, x, y=None):
        pass

    @abstractmethod
    def predict(self, x):
        # TODO Shouldn't X be optional if we have a trainable model and provide a prediction length
        pass

    @abstractmethod
    def forecast(self, x):
        # TODO Shouldn't X be optional if we have a trainable model and provide a prediction length
        pass

    def backtest(
            self,
            df: pd.DataFrame,
            start: pd.Timestamp,
            group_id: Union[str, int] = None,
            stride: int = None,
            retrain: bool = True,
    ) -> pd.DataFrame:
        if stride is None:
            stride = int(self.params.get("stride", 7))
        stride_offset = (
            pd.offsets.MonthEnd(stride)
            if self.freq == "M"
            else pd.DateOffset(days=stride)
        )
        df = df.copy().sort_values(by=[self.params["date_col"]])
        end_date = df[self.params["date_col"]].max()
        curr_date = start + self.one_ts_offset
        #print("end_date = ", end_date)

        results = []

        while curr_date + self.prediction_length_offset <= end_date + self.one_ts_offset:
            #print("start_date = ", curr_date)
            _df = df[df[self.params["date_col"]] < np.datetime64(curr_date)]
            actuals_df = df[
                (df[self.params["date_col"]] >= np.datetime64(curr_date))
                & (
                        df[self.params["date_col"]]
                        < np.datetime64(curr_date + self.prediction_length_offset)
                )]

            if retrain:
                self.fit(_df)

            metrics = self.calculate_metrics(_df, actuals_df, curr_date)

            if isinstance(metrics, dict):
                evaluation_results = [
                    (
                        group_id,
                        metrics["curr_date"],
                        metrics["metric_name"],
                        metrics["metric_value"],
                        metrics["forecast"],
                        metrics["actual"],
                        metrics["model_pickle"],
                    )
                ]
                results.extend(evaluation_results)
            elif isinstance(metrics, list):
                results.extend(metrics)

            curr_date += stride_offset

        res_df = pd.DataFrame(
            results,
            columns=[self.params["group_id"],
                     "backtest_window_start_date",
                     "metric_name",
                     "metric_value",
                     "forecast",
                     "actual",
                     "model_pickle"],
        )
        return res_df

    def calculate_metrics(
            self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date
    ) -> Dict[str, Union[str, float, bytes]]:
        pred_df, model_fitted = self.predict(hist_df, val_df)
        smape = mean_absolute_percentage_error(
            val_df[self.params["target"]],
            pred_df[self.params["target"]],
            symmetric=True,
        )
        if self.params["metric"] == "smape":
            metric_value = smape
        else:
            raise Exception(f"Metric {self.params['metric']} not supported!")

        return {
            "curr_date": curr_date,
            "metric_name": self.params["metric"],
            "metric_value": metric_value,
            "forecast": pred_df[self.params["target"]].to_numpy("float"),
            "actual": val_df[self.params["target"]].to_numpy(),
            "model_pickle": cloudpickle.dumps(model_fitted)}
