import functools
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from logging import Logger
from typing import Dict, Any, Union
import yaml
import pathlib
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
import sys


class DataQualityChecks:
    """
    Class to run the data quality checks.
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, pyspark.sql.DataFrame],
        conf: DictConfig,
        spark: SparkSession = None,
    ):
        if isinstance(df, pd.DataFrame):
            self.type = "pandas"
            self.df = df
        else:
            self.type = "spark"
            self.df = df.toPandas()
        self.conf = conf
        self.spark = spark

    def _backtest_length_check(self):
        """
        Checks if the backtest interval contains at least one prediction length.
        """
        backtest_days = self.conf["backtest_months"] * 30
        prediction_length_days = (
            self.conf["prediction_length"]
            if self.conf["freq"] == "D"
            else self.conf["prediction_length"] * 30
        )
        if backtest_days < prediction_length_days:
            raise Exception(f"Backtesting interval is shorter than prediction length!")

    def _external_regressors_check(self):
        """
        Checks if the resampling is turned off when an external regressor is given.
        """
        if (
            self.conf.get("static_categoricals", None)
            or self.conf.get("dynamic_categoricals", None)
            or self.conf.get("dynamic_reals", None)
        ):
            if self.conf.get("resample"):
                raise Exception(
                    f"Disable resampling when an external regressor is given!"
                )

    @staticmethod
    def _multiple_checks(
        _df: pd.DataFrame, conf: Dict[str, Any], max_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Run 3 checks on the subset dataset grouped by group_id.
        1. Check if any of external regressor provided contains null. If it does, this time series is removed.
        2. Check if the training period is longer than the requirement (train_predict_ratio).
        3. Check for missing entries. If the time series has a missing entry and the resampling is disabled,
        this time series is removed. If the time series has too many missing entries (more than 0.2 of the
        entire duration), it is removed even when resampling is enabled.
        4. If the time series has too many negative entries (more than 0.2 of the entire duration), it is removed.
        :return:
        pandas DataFrame with time series not meeting the requirement removed.
        """

        group_id = _df[conf["group_id"]].iloc[0]

        # Checking for nulls in external regressors
        static_features = conf.get("static_features", None)
        dynamic_reals = conf.get("dynamic_reals", None)
        if static_features:
            if _df[static_features].isnull().values.any():
                print(
                    f"Removing {conf['group_id']} {group_id} since static categoricals provided contain null."
                )
                return pd.DataFrame()
        if dynamic_reals:
            if _df[dynamic_reals].isnull().values.any():
                print(
                    f"Removing {conf['group_id']} {group_id} since dynamic reals provided contain null."
                )
                return pd.DataFrame()

        # Checking for training period length
        temp_df = _df[_df[conf["target"]] > 0]
        split_date = temp_df[conf["date_col"]].max() - pd.DateOffset(
            months=conf["backtest_months"]
        )
        if (
            temp_df[temp_df[conf["date_col"]] < split_date].count()[0]
            <= conf["train_predict_ratio"] * conf["prediction_length"]
        ):
            print(
                f"Removing {conf['group_id']} {group_id} as it does not meet train_predict_ratio requirement."
            )
            return pd.DataFrame()

        # Checking for missing entries
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
            # _resampled.resample(self.conf["freq"]).sum().reset_index()
            _resampled.reindex(date_idx)
            .reset_index()
            .fillna(value=0)
        )
        if len(_df) != len(_resampled):
            if conf.get("resample"):
                if (len(_resampled) - len(_df)) / len(_resampled) > 0.2:
                    print(
                        f"Removing {conf['group_id']} {group_id} as it contains too many missing "
                        f"entries (over 0.2) even though resampling is enabled."
                    )
                    return pd.DataFrame()
                else:
                    _df = _resampled
            else:
                print(
                    f"Removing {conf['group_id']} {group_id} as it contains missing entry and "
                    f"resampling is disabled."
                )
                return pd.DataFrame()

        _positive = _resampled[_resampled[conf["target"]] > 0]
        if (len(_resampled) - len(_positive)) / len(_resampled) > 0.2:
            print(
                f"Removing {conf['group_id']} {group_id} as it contains too many zero or negative "
                f"entries (over 0.2)."
            )
            return pd.DataFrame()
        else:
            _df = _resampled
        return _df

    def run(self) -> Union[pd.DataFrame, pyspark.sql.DataFrame]:
        """
        Main method of the job.
        :return:
        """

        print(f"Running data quality checks...")
        self.df[self.conf["date_col"]] = pd.to_datetime(self.df[self.conf["date_col"]])
        self.df.sort_values(by=self.conf["date_col"], inplace=True)
        self._external_regressors_check()
        self._backtest_length_check()
        if self.conf.get("data_quality_check", True):
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
        else:
            clean_df = self.df

        if clean_df.empty:
            raise Exception("None of the time series passed the data quality checks.")
        print(f"Finished data quality checks...")

        if self.type == "spark":
            clean_df = self.spark.createDataFrame(clean_df)

        return clean_df
