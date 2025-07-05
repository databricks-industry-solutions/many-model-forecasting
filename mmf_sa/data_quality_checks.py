import functools
from typing import Dict, Any, Union
from omegaconf import DictConfig
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from mmf_sa.exceptions import (
    DataQualityError,
    ParameterValidationError,
    EmptyDatasetError,
    InvalidConfigurationError
)


class DataQualityChecks:
    """
    Class to run the data quality checks.
    """
    def __init__(
        self,
        df: pyspark.sql.DataFrame,
        conf: DictConfig,
        spark: SparkSession = None,
    ):
        self.df = df.toPandas()
        self.conf = conf
        self.spark = spark

    def _backtest_length_check(self):
        """
        Checks if backtest_length contains at least one prediction_length.
        Mandatory check regardless of data_quality_check set to True or False.
        Parameters: self (Forecaster): A Forecaster object.
        """
        if self.conf["backtest_length"] < self.conf["prediction_length"]:
            raise ParameterValidationError(f"Backtest length is shorter than prediction length!")

    def _external_regressors_check(self):
        """
        Checks if the resampling is turned off when an exogenous regressor is given.
        Mandatory check irrespective of data_quality_check set to True or False.
        Parameters: self (Forecaster): A Forecaster object.
        """
        if (
            self.conf.get("static_features", None)
            or self.conf.get("dynamic_future_numerical", None)
            or self.conf.get("dynamic_future_categorical", None)
            or self.conf.get("dynamic_historical_numerical", None)
            or self.conf.get("dynamic_historical_categorical", None)
        ):
            if self.conf.get("resample"):
                raise InvalidConfigurationError(
                    f"Disable resampling when an external regressor is given!"
                )

    @staticmethod
    def _multiple_checks(
        _df: pd.DataFrame, conf: Dict[str, Any], max_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Runs 4 checks on the subset dataset grouped by group_id.
        These optional checks run only when data_quality_check is True.
        1. Checks if any of external regressor provided contains null. If it does, this time series is removed.
        2. Checks if the training period is longer than the requirement (train_predict_ratio).
        3. Checks for missing entries. If the time series has a missing entry and the resampling is disabled,
        it is removed. If the time series has too many missing entries (more than 0.2 of the
        entire duration), it is removed even when resampling is enabled.
        4. If the time series has too many negative entries (more than 0.2 of the entire duration), it is removed.

        Parameters:
            _df (pd.DataFrame): A pandas DataFrame.
            conf (Dict[str, Any]): A dictionary specifying the configuration.
            max_date (pd.Timestamp, optional): A pandas Timestamp object.

        Returns: _df (pd.DataFrame): A pandas DataFrame after time series not meeting the requirement removed.
        """

        group_id = _df[conf["group_id"]].iloc[0]

        # 1. Checking for nulls in external regressors
        static_features = conf.get("static_features", None)
        dynamic_future_numerical = conf.get("dynamic_future_numerical", None)
        dynamic_future_categorical = conf.get("dynamic_future_categorical", None)
        dynamic_historical_numerical = conf.get("dynamic_historical_numerical", None)
        dynamic_historical_categorical = conf.get("dynamic_historical_categorical", None)
        if static_features:
            if _df[static_features].isnull().values.any():
                # Removing: null in static categorical
                return pd.DataFrame()
        if dynamic_future_numerical:
            if _df[dynamic_future_numerical].isnull().values.any():
                # Removing: null in dynamic future numerical
                return pd.DataFrame()
        if dynamic_future_categorical:
            if _df[dynamic_future_categorical].isnull().values.any():
                # Removing: null in dynamic future categorical
                return pd.DataFrame()
        if dynamic_historical_numerical:
            if _df[dynamic_historical_numerical].isnull().values.any():
                # Removing: null in dynamic historical numerical
                return pd.DataFrame()
        if dynamic_historical_categorical:
            if _df[dynamic_historical_categorical].isnull().values.any():
                # Removing: null in dynamic historical categorical
                return pd.DataFrame()

        # 2. Checking for training period length
        temp_df = _df[_df[conf["target"]] > 0]
        if conf["freq"] == "H":
            backtest_offset = pd.DateOffset(hours=conf["backtest_length"])
        elif conf["freq"] == "D":
            backtest_offset = pd.DateOffset(days=conf["backtest_length"])
        elif conf["freq"] == "W":
            backtest_offset = pd.DateOffset(weeks=conf["backtest_length"])
        elif conf["freq"] == "M":
            backtest_offset = pd.DateOffset(months=conf["backtest_length"])
        else:
            backtest_offset = None
        split_date = temp_df[conf["date_col"]].max() - backtest_offset
        if (
            temp_df[temp_df[conf["date_col"]] < split_date].count().iloc[0]
            <= conf["train_predict_ratio"] * conf["prediction_length"]
        ):
            # Removing: train_predict_ratio requirement violated
            return pd.DataFrame()

        # 3. Checking for missing entries
        if max_date is None:
            max_date = _df[conf["date_col"]].max()
        
        _resampled = _df.set_index(conf["date_col"])
        date_idx = pd.date_range(
            start=_df[conf["date_col"]].min(),
            end=max_date,
            freq=conf["freq"],
            name=conf["date_col"],
        )
        _resampled = (
            _resampled.reindex(date_idx)
            .reset_index()
            .fillna(value=0)
        )
        
        if len(_resampled) > len(_df):
            if conf.get("resample"):
                if (len(_resampled) - len(_df)) / len(_resampled) > 0.2:
                    # Removing: missing rate over 0.2
                    return pd.DataFrame()
                else:
                    _df = _resampled
            else:
                # Removing: missing entry and resampling disabled
                return pd.DataFrame()

        # 4. Checking for negative entries
        _positive = _resampled[_resampled[conf["target"]] >= 0]
        if (len(_resampled) - len(_positive)) / len(_resampled) > 0.2:
            # Removing: negative entries over 0.2
            return pd.DataFrame()
        else:
            _df = _resampled
        return _df

    def run(self) -> tuple[Union[pd.DataFrame, pyspark.sql.DataFrame], list]:
        """
        Runs the main method of the job.
        Parameters: self (Forecaster): A Forecaster object.
        Returns:
            clean_df (Union[pd.DataFrame, pyspark.sql.DataFrame]): A pandas DataFrame or a PySpark DataFrame.
            removed (list): A list of group ids that are removed.
        """
        print(f"Running data quality checks...")
        self.df[self.conf["date_col"]] = pd.to_datetime(self.df[self.conf["date_col"]])
        self.df.sort_values(by=self.conf["date_col"], inplace=True)
        self._external_regressors_check()
        self._backtest_length_check()
        removed = []

        # If data_quality_check is None (not provided), we don't run the optional checks
        if self.conf.get("data_quality_check", False):
            _multiple_checks_func = functools.partial(
                self._multiple_checks,
                conf=self.conf,
                max_date=self.df[self.conf["date_col"]].max(),
            )
            clean_df = self.df.groupby(self.conf["group_id"]).apply(
                _multiple_checks_func
            )
            if isinstance(clean_df.index, pd.MultiIndex):
                clean_df = clean_df.drop(
                    columns=[self.conf["group_id"]], errors="ignore"
                ).reset_index()
                clean_df = clean_df[
                    clean_df.columns.drop(list(clean_df.filter(regex="level")))
                ]
            else:
                clean_df = clean_df.reset_index()
                clean_df = clean_df[
                    clean_df.columns.drop(list(clean_df.filter(regex="index")))
                ]
            before = set(self.df[self.conf['group_id']].unique())
            after = set(clean_df[self.conf['group_id']].unique())
            removed = sorted(list(before - after))
            print(f"Following {self.conf['group_id']} "
                  f"have been removed: {removed}")
        else:
            clean_df = self.df

        if clean_df.empty:
            raise EmptyDatasetError("None of the time series passed the data quality checks.")
        print(f"Finished data quality checks...")
        clean_df = self.spark.createDataFrame(clean_df)
        return clean_df, removed
