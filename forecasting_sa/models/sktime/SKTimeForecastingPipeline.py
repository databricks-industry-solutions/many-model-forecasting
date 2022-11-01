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

    def create_model(self) -> BaseForecaster:
        pass

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.params.target] = df[self.params.target].clip(0.1)
        _df = pd.DataFrame(
            {"y": df[self.params.target].values},
            index=df.index.to_period(self.params.freq),
        )
        return _df

    def predict(self, X):
        _df = self.prepare_data(X)

        model = self.create_model()

        model.fit(_df)

        y_pred = model.predict(
            ForecastingHorizon(
                np.arange(1, self.params["prediction_length"] + 1))
        )
        date_idx = pd.date_range(
            _df.index.max() + pd.DateOffset(days=1),
            _df.index.max() +
            pd.DateOffset(days=self.params["prediction_length"]),
            freq=self.params["freq"],
            name=self.params["date_col"],
        )
        forecast_df = pd.DataFrame(
            data={self.params["prediction_length"]: y_pred.values}, index=date_idx
        ).reset_index()
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(
            0.01)
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

    def create_model(self):
        model = TransformedTargetForecaster(
            [
                (
                    "deseasonalise",
                    ConditionalDeseasonalizer(
                        model=self.params.get("deseasonalise_model", "additive"), sp=7),
                ),
                (
                    "detrend",
                    Detrender(forecaster=PolynomialTrendForecaster(
                        degree=int(self.params.get("detrend_poly_degree", 1)))),
                ),
                (
                    "forecast",
                    make_reduction(
                        estimator=LGBMRegressor(),
                        scitype="tabular-regressor",
                        window_length=int(self.params.get(
                            "window_size", self.params.prediction_length)),
                        strategy="recursive",
                    ),
                ),
            ]
        )
        return model
