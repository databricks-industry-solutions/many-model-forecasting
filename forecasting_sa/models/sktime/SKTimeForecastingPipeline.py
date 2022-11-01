from abc import abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Detrender, ConditionalDeseasonalizer
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster

from forecasting_sa.models.abstract_model import ForecastingSAVerticalizedDataRegressor


class SKTimeForecastingPipeline(ForecastingSAVerticalizedDataRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model_spec = None

    def fit(self, X, y=None):
        pass

    @abstractmethod
    def create_model(self) -> BaseForecaster:
        pass

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.params.target] = df[self.params.target].clip(0.1)
        date_idx = pd.date_range(
            start=df[self.params.date_col].min(),
            end=df[self.params.date_col].max(),
            freq=self.params.freq,
            name=self.params.date_col,
        )
        df = df.set_index(self.params.date_col)
        df = df.reindex(date_idx, method="backfill")

        return df

    def predict(self, X):
        df = self.prepare_data(X)

        model = self.create_model()

        _df = pd.DataFrame(
            {"y": df[self.params.target].values},
            index=df.index.to_period(self.params.freq),
        )
        model.fit(_df)

        pred_df = model.predict(
            ForecastingHorizon(np.arange(1, self.params.prediction_length + 1))
        )
        date_idx = pd.date_range(
            df.index.max() + pd.DateOffset(days=1),
            df.index.max() + pd.DateOffset(days=self.params.prediction_length),
            freq=self.params.freq,
            name=self.params.date_col,
        )
        forecast_df = pd.DataFrame(data=[], index=date_idx).reset_index()
        forecast_df[self.params.target] = pred_df.y.values
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, float]:
        pred_df = self.predict(hist_df)
        smape = mean_absolute_percentage_error(
            val_df[self.params["target"]],
            pred_df[self.params["target"]],
            symmetric=True,
        )
        return {"smape": smape}


class SKTimeLgbmDsDt(SKTimeForecastingPipeline):
    def __init__(self, params):
        super().__init__(params)

    def create_model(self) -> BaseForecaster:
        model = TransformedTargetForecaster(
            [
                (
                    "deseasonalise",
                    ConditionalDeseasonalizer(
                        model=self.params.get("deseasonalise_model", "additive"), sp=7
                    ),
                ),
                (
                    "detrend",
                    Detrender(
                        forecaster=PolynomialTrendForecaster(
                            degree=int(self.params.get("detrend_poly_degree", 1))
                        )
                    ),
                ),
                (
                    "forecast",
                    make_reduction(
                        estimator=LGBMRegressor(),
                        scitype="tabular-regressor",
                        window_length=int(
                            self.params.get(
                                "window_size", self.params.prediction_length
                            )
                        ),
                        strategy="recursive",
                    ),
                ),
            ]
        )
        return model
