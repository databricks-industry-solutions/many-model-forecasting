import logging
from abc import abstractmethod
import numpy as np
import pandas as pd
import cloudpickle
from typing import Any, Dict, List, NamedTuple, Optional, Union
from pyspark.sql import DataFrame
from pyspark.sql.functions import array_sort, collect_list, struct
from sklearn.base import BaseEstimator, RegressorMixin
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
import mlflow
from mmf_sa.exceptions import DataPreparationError, ModelPredictionError, UnsupportedMetricError
from mmf_sa.calibration import quantile_columns, quantile_maps
mlflow.set_registry_uri("databricks-uc")

_logger = logging.getLogger(__name__)

# Backtest metrics, by the name used in the `metric` parameter.
METRICS = {
    "smape": lambda: MeanAbsolutePercentageError(symmetric=True),
    "mape": lambda: MeanAbsolutePercentageError(symmetric=False),
    "mae": lambda: MeanAbsoluteError(),
    "mse": lambda: MeanSquaredError(square_root=False),
    "rmse": lambda: MeanSquaredError(square_root=True),
}


def metric_function(metric_name: str):
    """The sktime metric for a `metric` parameter value; raises UnsupportedMetricError for unknown names."""
    if metric_name not in METRICS:
        raise UnsupportedMetricError(f"Metric {metric_name} not supported!")
    return METRICS[metric_name]()


class EvaluationRow(NamedTuple):
    """One series' backtest result for one window: a row of evaluation_output (before run metadata)."""
    key: Any
    backtest_window_start_date: Any
    metric_name: str
    metric_value: float
    forecast: np.ndarray
    actual: np.ndarray
    model_pickle: bytes = b""
    forecast_quantiles: Optional[Dict[str, Any]] = None

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
        "timesfm[torch,xreg] @ https://github.com/google-research/timesfm/archive/a83dbf3f163cf15993dac6a45bbb5dcb160e14e8.tar.gz",
        MMF_PACKAGE,
    ],
    "moirai": [
        "uni2ts==2.0.0",
        MMF_PACKAGE,
    ],
    "foundationforecast": [
        "--extra-index-url https://download.pytorch.org/whl/cu128",
        "foundationforecast==0.1.15",
        "torch==2.11.0+cu128",
        "torchvision==0.26.0+cu128",
        "torchaudio==2.11.0+cu128",
        "transformers==5.12.1",
        "huggingface-hub==1.33.0",
        "granite-tsfm==0.3.10",
        "timecopilot-chronos-forecasting==0.2.2",
        "timecopilot-timesfm==0.4.0",
        "timecopilot-timestral==0.1.0",
        "timecopilot-tirex2==0.1.0",
        "timecopilot-toto==0.1.7",
        "timecopilot-toto-2==0.1.1",
        "tafsut==0.1.0",
        "tfc-t0[torch]==0.6.0",
        # Upper bounds of fev and nixtla; without them pip backtracks for hours in a fresh environment.
        "datasets>=4,<5",
        "pyarrow<24",
        MMF_PACKAGE,
    ],
    "mlforecast": [
        "cloudpickle==3.0.0",
        "mlforecast==1.0.31",
        "lightgbm==4.6.0",
        "optuna==3.6.1",
        MMF_PACKAGE,
    ],
}


def collect_sorted(sdf: DataFrame, group_col: str, order_col: str, value_cols: List[str]) -> DataFrame:
    """
    Groups a Spark DataFrame into one row per group, with one array per column sorted by order_col.
    collect_list alone does not guarantee the order of the collected values after a shuffle, so the
    arrays are built from (order_col, *value_cols) structs sorted with array_sort (structs compare
    field by field, order_col first), then split back into one array per column.
    Parameters:
        sdf (DataFrame): A Spark DataFrame.
        group_col (str): The column to group by.
        order_col (str): The column that orders the values inside each group (e.g. the date column).
        value_cols (List[str]): The other columns to collect.
    Returns: DataFrame: group_col, order_col and value_cols, each value column as a sorted array.
    """
    rows = sdf.groupBy(group_col).agg(
        array_sort(collect_list(struct(order_col, *value_cols))).alias("_rows")
    )
    return rows.select(group_col, *[rows["_rows"].getField(c).alias(c) for c in [order_col, *value_cols]])


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
                results.append(EvaluationRow(
                    group_id,
                    metrics["curr_date"],
                    metrics["metric_name"],
                    metrics["metric_value"],
                    metrics["forecast"],
                    metrics["actual"],
                    metrics["model_pickle"],
                    metrics.get("forecast_quantiles"),
                ))
            elif isinstance(metrics, list):
                # Plain tuples are accepted too: model_pickle and forecast_quantiles default to b"" and None.
                results.extend(EvaluationRow(*m) for m in metrics)

            curr_date += stride_offset

        res_df = pd.DataFrame(results, columns=[self.params["group_id"], *EvaluationRow._fields[1:]])

        return res_df

    def series_metrics(
            self, pred_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, long_format: bool = False
    ) -> List[EvaluationRow]:
        """
        Backtest rows of a multi-series forecast, one per series of pred_df: the shared body of the
        calculate_metrics of global and foundation models.
        Parameters:
            pred_df (pd.DataFrame): The forecasts. One row per series whose target cell holds the horizon array
                (with native quantile columns, if any), or with long_format one row per series and step, of which
                the last prediction_length rows of each series are used.
            val_df (pd.DataFrame): The actuals, one row per series and step.
            curr_date: The start of the backtest window.
        Returns: List[EvaluationRow]. Series whose metric cannot be computed are logged and skipped.
        """
        metric_name = self.params["metric"]
        metric = metric_function(metric_name)
        group_col = self.params["group_id"]
        target_col = self.params["target"]
        actuals = {k: np.asarray(v, dtype=float) for k, v in val_df.groupby(group_col, sort=False)[target_col]}
        if long_format:
            prediction_length = int(self.params["prediction_length"])
            forecasts = {k: v.iloc[-prediction_length:].to_numpy()
                         for k, v in pred_df.groupby(group_col, sort=False)[target_col]}
            quantiles = {}
        else:
            forecasts = dict(zip(pred_df[group_col].to_numpy(), pred_df[target_col].to_numpy()))
            quantiles = quantile_maps(pred_df, group_col)
        rows = []
        for key in pred_df[group_col].unique():
            try:
                actual = actuals[key]
                forecast = np.asarray(forecasts[key], dtype=float)
                rows.append(EvaluationRow(key, curr_date, metric_name, metric(actual, forecast), forecast, actual,
                                          b"", quantiles.get(key)))
            except (ModelPredictionError, DataPreparationError) as err:
                _logger.warning(f"Failed to calculate metric for key {key}: {err}")
            except Exception as err:
                _logger.warning(f"Unexpected error calculating metric for key {key}: {err}")
        return rows

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
        metric_value = metric_function(self.params["metric"])(actual, forecast)

        return {
            "curr_date": curr_date,
            "metric_name": self.params["metric"],
            "metric_value": metric_value,
            "forecast": pred_df[self.params["target"]].to_numpy("float"),
            "actual": val_df[self.params["target"]].to_numpy(),
            "model_pickle": cloudpickle.dumps(model_fitted),
            "forecast_quantiles": self.forecast_quantiles(pred_df)}

    @staticmethod
    def forecast_quantiles(pred_df: pd.DataFrame):
        """
        Native quantiles of a one-series forecast frame, from its QUANTILE_COLUMN_PREFIX columns.
        Returns: dict of quantile key -> array of length prediction_length, or None without such columns.
        """
        cols = quantile_columns(pred_df)
        if not cols:
            return None
        return {key: [float(x) for x in pred_df[c]] for key, c in cols.items()}
