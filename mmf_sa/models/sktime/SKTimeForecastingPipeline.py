from abc import abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from sktime.forecasting.model_selection import (
    SlidingWindowSplitter,
    ForecastingGridSearchCV,
)
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from mmf_sa.models.abstract_model import ForecastingRegressor
from mmf_sa.intervals.config import central_levels
from mmf_sa.intervals.quantiles import QUANTILE_COLUMN_PREFIX, quantile_key


class SKTimeForecastingPipeline(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model_spec = self.params.model_spec
        self.model = None
        self.param_grid = self.create_param_grid()

    @abstractmethod
    def create_model(self) -> BaseForecaster:
        pass

    def create_param_grid(self) -> Dict[str, Any]:
        return {}

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().fillna(0)
        df[self.params.target] = df[self.params.target].clip(0)
        freq = self.params.freq
        if freq == "H":
            freq = "h"
        elif freq == "M":
            freq = "ME"
        date_idx = pd.date_range(
            start=df[self.params.date_col].min(),
            end=df[self.params.date_col].max(),
            freq=freq,
            name=self.params.date_col,
        )
        df = df.set_index(self.params.date_col)
        df = df.reindex(date_idx, method="backfill")
        df = df.sort_index()
        df = pd.DataFrame({"y": df[self.params.target].values}, index=df.index.to_period(self.params.freq))
        return df

    def fit(self, x, y=None):
        if (self.params.get("enable_gcv", False)
                and self.model is None
                and self.param_grid):
            _model = self.create_model()
            cv = SlidingWindowSplitter(
                initial_window=int(len(x) - self.params.prediction_length * 4),
                window_length=self.params.prediction_length * 10,
                step_length=int(self.params.prediction_length * 1.5),
            )
            gscv = ForecastingGridSearchCV(_model, cv=cv, param_grid=self.param_grid, n_jobs=-1)
            gscv.fit(x)
            self.model = gscv.best_forecaster_
        else:
            self.model = self.create_model()
            self.model.fit(x)

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        _df = self.prepare_data(hist_df)
        self.fit(_df)
        fh = ForecastingHorizon(np.arange(1, self.params.prediction_length + 1))
        pred_df = self.model.predict(fh)
        freq = self.params.freq
        if freq == "H":
            freq = "h"
        elif freq == "M":
            freq = "ME"
        date_idx = pd.date_range(
            _df.index.max().to_timestamp(freq=self.params.freq) + self.one_ts_offset,
            _df.index.max().to_timestamp(freq=self.params.freq) + self.prediction_length_offset,
            freq=freq,
            name=self.params.date_col,
        )
        forecast_df = pd.DataFrame(data=[], index=date_idx).reset_index()
        forecast_df[self.params.target] = pred_df.y.values
        quantiles = self.params.get("native_quantiles")
        if quantiles:
            forecast_df = self.native_quantile_columns(forecast_df, quantiles, fh)
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0)
        return forecast_df, self.model

    def native_quantile_columns(self, forecast_df: pd.DataFrame, quantiles, fh: ForecastingHorizon) -> pd.DataFrame:
        """
        Adds one QUANTILE_COLUMN_PREFIX column per requested quantile, read from the central intervals of
        sktime's predict_interval: coverage c gives the bounds at the quantiles (1 - c) / 2 and (1 + c) / 2.
        The median is the point forecast, as for statsforecast. Quantiles are the model's raw band: unlike
        the point forecast, they are not clipped here; calibration applies prediction_intervals.clip_lower.
        The model is fitted once by predict, so each coverage only costs a prediction.
        """
        levels = central_levels(quantiles)
        bands = self.model.predict_interval(fh, coverage=[level / 100 for level in levels]) if levels else None
        for q in quantiles:
            if q == 0.5:
                values = forecast_df[self.params.target]
            else:
                (level,) = central_levels([q])
                values = bands[(bands.columns[0][0], level / 100, "lower" if q < 0.5 else "upper")]
            forecast_df[QUANTILE_COLUMN_PREFIX + quantile_key(q)] = values.to_numpy(dtype=float)
        return forecast_df

    def forecast(self, x, spark=None):
        return self.predict(x)


class SKTimeProphet(SKTimeForecastingPipeline):
    native_quantiles_supported = True  # bands from Prophet's predict_interval (native_quantile_columns)

    def __init__(self, params):
        super().__init__(params)

    def create_model(self) -> BaseForecaster:
        model = Prophet(
            freq=self.params.freq,
            growth = self.model_spec.get("growth"),
            yearly_seasonality=self.model_spec.get("yearly_seasonality"),
            weekly_seasonality=self.model_spec.get("weekly_seasonality"),
            daily_seasonality=self.model_spec.get("daily_seasonality"),
            seasonality_mode=self.model_spec.get("seasonality_mode"),
        )
        return model

    def create_param_grid(self):
        return {
            "growth": ['linear', 'logarithmic'],
            "seasonality_mode": ['additive', 'multiplicative'],
        }
