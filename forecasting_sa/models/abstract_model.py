from abc import abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


class ForecastingSARegressor(BaseEstimator, RegressorMixin):
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

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X):
        # TODO Shouldn't X be optional if we have a trainable model and provide a prediction length
        pass

    def supports_tuning(self) -> bool:
        return False

    @abstractmethod
    def search_space(self):
        pass

    @abstractmethod
    def calculate_metrics(
            self, hist_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, Union[str, float, bytes]]:
        pass

    def backtest(
            self,
            df: pd.DataFrame,
            start: pd.Timestamp,
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
        print("end_date = ", end_date)

        results = []

        while curr_date + self.prediction_length_offset <= end_date + self.one_ts_offset:
            print("start_date = ", curr_date)
            _df = df[df[self.params["date_col"]] < np.datetime64(curr_date)]
            actuals_df = df[
                (df[self.params["date_col"]] >= np.datetime64(curr_date))
                & (
                        df[self.params["date_col"]]
                        < np.datetime64(curr_date + self.prediction_length_offset)
                )
                ]
            if retrain:
                self.fit(_df)

            metrics = self.calculate_metrics(_df, actuals_df)
            metrics_and_date = [
                (
                    curr_date,
                    metrics["metric_name"],
                    metrics["metric_value"],
                    metrics["forecast"],
                    metrics["actual"],
                )
            ]
            results.extend(metrics_and_date)
            curr_date += stride_offset

        res_df = pd.DataFrame(
            results,
            columns=["backtest_window_start_date",
                     "metric_name",
                     "metric_value",
                     "forecast",
                     "actual"],
        )
        return res_df


class ForecastingSAPivotRegressor(ForecastingSARegressor):
    def calculate_metrics(
            self, hist_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, Union[str, float, bytes, None]]:
        print("start calculate_metrics_pivot for model: ", self.params["name"])
        pred_df = self.predict(hist_df)
        pred_cols = [c for c in pred_df.columns if c not in [self.params["date_col"]]]
        smape = mean_absolute_percentage_error(
            val_df[pred_cols],
            pred_df[pred_cols],
            symmetric=True,
        )
        # metrics = []
        # for c in pred_df.columns:
        #     if c not in [self.params["date_col"]]:
        #         smape = mean_absolute_percentage_error(
        #             val_df[c].values, pred_df[c].values, symmetric=True
        #         )
        #         metrics.append(smape)
        # smape = sum(metrics) / len(metrics)
        print("finished calculate_metrics")
        return {"metric_name": self.params["metric"],
                "metric_value": smape,
                "forecast": None,
                "actual": None}


class ForecastingSAVerticalizedDataRegressor(ForecastingSARegressor):
    def calculate_metrics(
            self, hist_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, Union[str, float, bytes, None]]:
        print("starting calculate_metrics")
        to_pred_df = val_df.copy()
        to_pred_df[self.params["target"]] = np.nan
        to_pred_df = pd.concat([hist_df, to_pred_df]).reset_index(drop=True)
        pred_df = self.predict(to_pred_df)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        # Compared predicted with val
        for key in keys:
            try:
                smape = mean_absolute_percentage_error(
                    val_df[val_df[self.params["group_id"]] == key][self.params["target"]],
                    pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]]
                    .iloc[-self.params["prediction_length"]:],
                    symmetric=True,
                )
                metrics.append(smape)
            except:
                pass
        smape = sum(metrics) / len(metrics)
        print("finished calculate_metrics")
        if self.params["metric"] == "smape":
            metric_value = smape
        else:
            raise Exception(f"Metric {self.params['metric']} not supported!")
        return {"metric_name": self.params["metric"],
                "metric_value": metric_value,
                "forecast": None,
                "actual": None}
