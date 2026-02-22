from abc import abstractmethod
import numpy as np
import pandas as pd
import cloudpickle
from typing import Dict, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
import mlflow
from mmf_sa.exceptions import UnsupportedMetricError
mlflow.set_registry_uri("databricks-uc")

MMF_PACKAGE = "git+https://github.com/databricks-industry-solutions/many-model-forecasting.git"

MODEL_PIP_REQUIREMENTS = {
    "neuralforecast": [
        "cloudpickle==2.2.1",
        "neuralforecast==3.1.4",
        "ray[tune]==2.5.0",
        MMF_PACKAGE,
    ],
    "chronos": [
        "torch>=2.3.1",
        "transformers>=4.41.2",
        "chronos-forecasting==2.2.2",
        MMF_PACKAGE,
    ],
    "timesfm": [
        "timesfm[torch] @ git+https://github.com/google-research/timesfm.git@2dcc66fbfe2155adba1af66aa4d564a0ee52f61e",
        MMF_PACKAGE,
    ],
    "moirai": [
        "uni2ts==2.0.0",
        MMF_PACKAGE,
    ],
}


class ForecastingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, params):
        self.params = params
        self.freq = params["freq"].upper()[0]
        self.one_ts_offset = (
            pd.offsets.MonthEnd(1) if self.freq == "M" else
            pd.DateOffset(weeks=1) if self.freq == "W" else
            pd.DateOffset(days=1) if self.freq == "D" else
            pd.DateOffset(hours=1) if self.freq == "H" else
            None
        )
        self.prediction_length_offset = (
            pd.offsets.MonthEnd(params["prediction_length"]) if self.freq == "M" else
            pd.DateOffset(weeks=params["prediction_length"]) if self.freq == "W" else
            pd.DateOffset(days=params["prediction_length"]) if self.freq == "D" else
            pd.DateOffset(hours=params["prediction_length"]) if self.freq == "H" else
            None
        )

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @abstractmethod
    def fit(self, x, y=None):
        pass

    @abstractmethod
    def predict(self, x, y=None):
        pass

    @abstractmethod
    def forecast(self, x, spark=None):
        pass

    def backtest(
            self,
            df: pd.DataFrame,
            start: pd.Timestamp,
            group_id: Union[str, int] = None,
            # backtest_retrain: bool = False,
            spark=None,
    ) -> pd.DataFrame:
        """
        Performs backtesting using the provided pandas DataFrame, start timestamp, group id, stride and SparkSession.
        Parameters:
            self (Forecaster): A Forecaster object.
            df (pd.DataFrame): A pandas DataFrame.
            start (pd.Timestamp): A pandas Timestamp object.
            group_id (Union[str, int], optional): A string or an integer specifying the group id. Default is None.
            spark (SparkSession, optional): A SparkSession object. Default is None.
        Returns: res_df (pd.DataFrame): A pandas DataFrame.
        """
        stride = int(self.params["stride"]) # Read in stride
        stride_offset = (
            pd.offsets.MonthEnd(stride) if self.freq == "M" else
            pd.DateOffset(weeks=stride) if self.freq == "W" else
            pd.DateOffset(days=stride) if self.freq == "D" else
            pd.DateOffset(hours=stride) if self.freq == "H" else
            None
        )
        df = df.copy().sort_values(by=[self.params["date_col"]])
        end_date = df[self.params["date_col"]].max() # Last date from the training data
        # Offsets the timestamp: e.g. if it's in the middle of the month for a monthly time series, makes it the end of the month
        curr_date = start + self.one_ts_offset
        # print("end_date = ", end_date)

        results = []

        while curr_date + self.prediction_length_offset <= end_date + self.one_ts_offset:
            # print("start_date = ", curr_date)
            _df = df[df[self.params["date_col"]] < np.datetime64(curr_date)]
            actuals_df = df[
                (df[self.params["date_col"]] >= np.datetime64(curr_date))
                & (
                        df[self.params["date_col"]]
                        < np.datetime64(curr_date + self.prediction_length_offset)
                )]

            # backtest_retrain for global models is currently not supported
            # if backtest_retrain and self.params["model_type"] == "global":
            #    self.fit(_df)

            metrics = self.calculate_metrics(_df, actuals_df, curr_date, spark)

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
            self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> Dict[str, Union[str, float, bytes]]:
        """
        Calculates the metrics using the provided historical DataFrame, validation DataFrame, current date, and SparkSession.
        Parameters:
            self (Forecaster): A Forecaster object.
            hist_df (pd.DataFrame): A pandas DataFrame.
            val_df (pd.DataFrame): A pandas DataFrame.
            curr_date: A pandas Timestamp object.
            spark (SparkSession, optional): A SparkSession object. Default is None.
        Returns: metrics (Dict[str, Union[str, float, bytes]]): A dictionary specifying the metrics.
        """
        pred_df, model_fitted = self.predict(hist_df, val_df)
        
        actual = val_df[self.params["target"]].to_numpy()
        forecast = pred_df[self.params["target"]].to_numpy()

        if self.params["metric"] == "smape":
            smape = MeanAbsolutePercentageError(symmetric=True)
            metric_value = smape(actual, forecast)
        elif self.params["metric"] == "mape":
            mape = MeanAbsolutePercentageError(symmetric=False)
            metric_value = mape(actual, forecast)
        elif self.params["metric"] == "mae":
            mae = MeanAbsoluteError()
            metric_value = mae(actual, forecast)
        elif self.params["metric"] == "mse":
            mse = MeanSquaredError(square_root=False)
            metric_value = mse(actual, forecast)
        elif self.params["metric"] == "rmse":
            rmse = MeanSquaredError(square_root=True)
            metric_value = rmse(actual, forecast)
        else:
            raise UnsupportedMetricError(f"Metric {self.params['metric']} not supported!")

        return {
            "curr_date": curr_date,
            "metric_name": self.params["metric"],
            "metric_value": metric_value,
            "forecast": pred_df[self.params["target"]].to_numpy("float"),
            "actual": val_df[self.params["target"]].to_numpy(),
            "model_pickle": cloudpickle.dumps(model_fitted)}
