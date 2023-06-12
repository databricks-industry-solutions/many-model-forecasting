import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from neuralforecast import NeuralForecast
from forecasting_sa.models.abstract_model import ForecastingSAVerticalizedDataRegressor


class NeuralFcForecaster(ForecastingSAVerticalizedDataRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model_spec = None
        self.model = None

    def prepare_data(self, df: pd.DataFrame, future: bool = False) -> pd.DataFrame:
        if future:
            # Prepare future dataframe with exogenous regressors for forecasting
            if 'dynamic_reals' in self.params.keys():
                try:
                    _df = (
                        df[[self.params.group_id, self.params.date_col]
                           + self.params.dynamic_reals]
                    )
                except Exception as e:
                    raise Exception(f"Exogenous regressors missing: {e}")
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
            if 'dynamic_reals' in self.params.keys():
                try:
                    _df = (
                        df[[self.params.group_id, self.params.date_col, self.params.target]
                           + self.params.dynamic_reals]
                    )
                except Exception as e:
                    raise Exception(f"Exogenous regressor columns missing from "
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
        _df = _df.set_index("unique_id")
        return _df

    def fit(self, X, y=None):
        if isinstance(self.model, NeuralForecast):
            _df = self.prepare_data(X)
            self.model.fit(
                _df,
                ##### From here
                static_df=self.params["static_features"],
            )
        return self

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        _df = self.prepare_data(hist_df)
        forecast_df = self.model.predict(_df)
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
        model = NeuralForecast(models=[self.model_spec], freq=self.freq, n_jobs=-1)
        model.fit(_df)
        if 'dynamic_reals' in self.params.keys():
            _last_date = _df["ds"].max()
            _future_df = df[
                (df[self.params["date_col"]] > np.datetime64(_last_date))
                & (df[self.params["date_col"]]
                   <= np.datetime64(_last_date + self.prediction_length_offset))
            ]
            _future_exogenous = self.prepare_data(_future_df, future=True)
            try:
                forecast_df = model.predict(self.params["prediction_length"], _future_exogenous)
            except Exception as e:
                print(
                    f"Removing group_id {df[self.params.group_id][0]} as future exogenous "
                    f"regressors are not provided.")
                return pd.DataFrame(
                    columns=[self.params.date_col, self.params.target]
                )
        else:
            forecast_df = model.predict(self.params["prediction_length"])

        first_model = [col for col in forecast_df.columns.to_list() if col != "ds"][0]
        forecast_df = forecast_df.reset_index(drop=True).rename(
            columns={
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