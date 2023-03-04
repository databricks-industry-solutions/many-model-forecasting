from abc import abstractmethod
from typing import Dict, Any

import numpy as np
import pandas as pd
import cloudpickle
from typing import Dict, Any, Union
from lightgbm import LGBMRegressor
from sktime.forecasting.model_selection import (
    SlidingWindowSplitter,
    ForecastingGridSearchCV,
)
from sktime.forecasting.tbats import TBATS
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
        self.model_spec = self.params.model_spec
        self.model = None
        self.param_grid = self.create_param_grid()

    def fit(self, X, y=None):
        print("fit called")
        if self.params.get("enable_gcv", False) and self.model is None and self.param_grid:
            print("tuning..")
            _model = self.create_model()
            df = self.prepare_data(X)
            cv = SlidingWindowSplitter(
                initial_window=int(len(df) - self.params.prediction_length * 4),
                window_length=self.params.prediction_length * 10,
                step_length=int(self.params.prediction_length * 1.5),
            )
            gscv = ForecastingGridSearchCV(
                _model, cv=cv, param_grid=self.param_grid, n_jobs=1
            )
            _df = pd.DataFrame(
                {"y": df[self.params.target].values},
                index=df.index.to_period(self.params.freq),
            )
            gscv.fit(_df)
            self.model = gscv.best_forecaster_
            print(gscv.best_params_)

    @abstractmethod
    def create_model(self) -> BaseForecaster:
        pass

    def create_param_grid(self) -> Dict[str, Any]:
        return {}

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().fillna(0.1)
        df[self.params.target] = df[self.params.target].clip(0.1)
        date_idx = pd.date_range(
            start=df[self.params.date_col].min(),
            end=df[self.params.date_col].max(),
            freq=self.params.freq,
            name=self.params.date_col,
        )
        df = df.set_index(self.params.date_col)
        df = df.reindex(date_idx, method="backfill")
        df = df.sort_index()
        return df

    def predict(self, X):
        print("predict called")
        df = self.prepare_data(X)

        if not self.model:
            self.model = self.create_model()

        _df = pd.DataFrame(
            {"y": df[self.params.target].values},
            index=df.index.to_period(self.params.freq),
        )
        self.model.fit(_df)

        pred_df = self.model.predict(
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
    ) -> Dict[str, Union[str, float, bytes]]:
        pred_df = self.predict(hist_df)
        smape = mean_absolute_percentage_error(
            val_df[self.params["target"]].values,
            pred_df[self.params["target"]].values,
            symmetric=True,
        )
        if self.params["metric"] == "smape":
            metric_value = smape
        else:
            raise Exception(f"Metric {self.params['metric']} not supported!")
        return {"metric_name": self.params["metric"],
                "metric_value": metric_value,
                "forecast": cloudpickle.dumps(pred_df),
                "actual": cloudpickle.dumps(val_df),
                }


class SKTimeLgbmDsDt(SKTimeForecastingPipeline):
    def __init__(self, params):
        super().__init__(params)

    def create_model(self) -> BaseForecaster:
        model = TransformedTargetForecaster(
            [
                (
                    "deseasonalise",
                    ConditionalDeseasonalizer(
                        model=self.model_spec.get("deseasonalise_model", "additive"),
                        sp=int(self.model_spec.get("season_length", 1)),
                    ),
                ),
                (
                    "detrend",
                    Detrender(
                        forecaster=PolynomialTrendForecaster(
                            degree=int(self.model_spec.get("detrend_poly_degree", 1))
                        )
                    ),
                ),
                (
                    "forecast",
                    make_reduction(
                        estimator=LGBMRegressor(random_state=42),
                        scitype="tabular-regressor",
                        window_length=int(
                            self.model_spec.get(
                                "window_size", self.params.prediction_length
                            )
                        ),
                        strategy="recursive",
                    ),
                ),
            ]
        )
        return model

    def create_param_grid(self):
        return {
            "deseasonalise__model": ["additive", "multiplicative"],
            "deseasonalise__sp": [1, 7, 14],
            "detrend__forecaster__degree": [1, 2, 3],
            # "forecast__estimator__learning_rate": [0.1, 0.01, 0.001],
            "forecast__window_length": [
                self.params.prediction_length,
                self.params.prediction_length * 2,
            ],
        }


class SKTimeTBats(SKTimeForecastingPipeline):
    def __init__(self, params):
        super().__init__(params)

    def create_model(self) -> BaseForecaster:
        model = TBATS(
            sp=int(self.model_spec.get("season_length", 1)),
            use_trend=self.model_spec.get("use_trend", True),
            use_box_cox=self.model_spec.get("box_cox", True),
        )
        return model

    def create_param_grid(self):
        return {
            "use_trend": [True, False],
            "use_box_cox": [True, False],
            "sp": [1, 7, 14],
        }
