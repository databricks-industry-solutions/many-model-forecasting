import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from neuralforecast import NeuralForecast
from forecasting_sa.models.abstract_model import ForecastingRegressor


class NeuralFcForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model_spec = None
        self.model = None

    def prepare_data(self, df: pd.DataFrame, future: bool = False) -> pd.DataFrame:
        if future:
            # Prepare future dataframe with exogenous regressors for forecasting
            if 'dynamic_future' in self.params.keys():
                try:
                    _df = (
                        df[[self.params.group_id, self.params.date_col]
                           + self.params.dynamic_future]
                    )
                except Exception as e:
                    raise Exception(f"Dynamic future regressors missing: {e}")
            else:
                _df = df[[self.params.group_id, self.params.date_col]]

            _df = (
                _df.rename(
                    columns={
                        self.params.group_id: "unique_id",
                        self.params.date_col: "ds",
                    }
                )
            )
        else:
            # Prepare historical dataframe with or without exogenous regressors for training
            df[self.params.target] = df[self.params.target].clip(0.1)
            if 'dynamic_future' in self.params.keys():
                try:
                    _df = (
                        df[[self.params.group_id, self.params.date_col, self.params.target]
                           + self.params.dynamic_future]
                    )
                except Exception as e:
                    raise Exception(f"Dynamic future regressor columns missing from "
                                    f"the training dataset: {e}")
            elif 'dynamic_historical' in self.params.keys():
                try:
                    _df = (
                        df[[self.params.group_id, self.params.date_col, self.params.target]
                           + self.params.dynamic_historical]
                    )
                except Exception as e:
                    raise Exception(f"Dynamic historical regressor columns missing from "
                                    f"the training dataset: {e}")
            else:
                _df = df[[self.params.group_id, self.params.date_col, self.params.target]]

            _df = (
                _df.rename(
                    columns={
                        self.params.group_id: "unique_id",
                        self.params.date_col: "ds",
                        self.params.target: "y",
                    }
                )
            )
        return _df

    def prepare_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'static_features' in self.params.keys():
            static_features = df[[self.params.group_id] + self.params.static_features].drop_duplicates()
            static_features = (
                static_features.rename(
                    columns={
                        self.params.group_id: "unique_id",
                    }
                )
            )
            # categorical encoding of columns that are in object
            for col in static_features.columns:
                if (col != "unique_id") and (static_features[col].dtype == object):
                    static_features[col] = static_features[col].astype('category')
                    static_features[col] = static_features[col].cat.codes
            return static_features
        else:
            return None

    def fit(self, X, y=None):
        if isinstance(self.model, NeuralForecast):
            _df = self.prepare_data(X)
            _static_df = self.prepare_static_features(X)
            self.model.fit(
                df=_df,
                static_df=_static_df,
            )
        return self

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        _df = self.prepare_data(hist_df)
        _dynamic_future = self.prepare_data(val_df, future=True)
        _dynamic_future = None if _dynamic_future.empty else _dynamic_future
        _static_df = self.prepare_static_features(hist_df)
        forecast_df = self.model.predict(
            df=_df,
            static_df=_static_df,
            futr_df=_dynamic_future
        )
        first_model = [col for col in forecast_df.columns.to_list() if col != "ds"][0]
        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                first_model: self.params.target,
            }
        )
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df

    def forecast(self, df: pd.DataFrame):
        _df = df[df[self.params.target].notnull()]
        _df = self.prepare_data(_df)
        _last_date = _df["ds"].max()
        _future_df = df[
            (df[self.params["date_col"]] > np.datetime64(_last_date))
            & (df[self.params["date_col"]]
               <= np.datetime64(_last_date + self.prediction_length_offset))
        ]

        _dynamic_future = self.prepare_data(_future_df, future=True)
        _dynamic_future = None if _dynamic_future.empty else _dynamic_future
        _static_df = self.prepare_static_features(_future_df)

        # Check if dynamic futures for all unique_id are provided.
        # If not, drop unique_id without dynamic futures from scoring.
        if (_dynamic_future is not None) and \
                (not set(_df["unique_id"].unique().flatten()) \
                        .issubset(set(_dynamic_future["unique_id"].unique().flatten()))):
            _df = _df[_df["unique_id"].isin(list(_dynamic_future["unique_id"].unique()))]

        forecast_df = self.model.predict(
            df=_df,
            static_df=_static_df,
            futr_df=_dynamic_future
        )
        first_model = [col for col in forecast_df.columns.to_list() if col != "ds"][0]
        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                first_model: self.params.target,
            }
        )
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, Union[str, float, bytes]]:
        pred_df = self.predict(hist_df, val_df)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        for key in keys:
            forecast = val_df[val_df[self.params["group_id"]] == key][self.params["target"]]
            actual = pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].\
                         iloc[-self.params["prediction_length"]:]
            try:
                smape = mean_absolute_percentage_error(
                    actual,
                    forecast,
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