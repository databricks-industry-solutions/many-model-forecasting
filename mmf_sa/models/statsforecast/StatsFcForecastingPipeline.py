import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoETS,
    AutoARIMA,
    ADIDA,
    IMAPA,
    TSB,
    AutoCES,
    AutoTheta,
    AutoTBATS,
    AutoMFLES,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    WindowAverage,
    SeasonalWindowAverage,
    Naive,
    SeasonalNaive,
)
from mmf_sa.models.abstract_model import ForecastingRegressor
from mmf_sa.exceptions import MissingFeatureError, DataPreparationError
from mmf_sa.intervals.config import central_levels
from mmf_sa.intervals.quantiles import QUANTILE_COLUMN_PREFIX, quantile_key


class StatsFcForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model_spec = None
        self.model = None

    def prepare_data(self, df: pd.DataFrame, future: bool = False) -> pd.DataFrame:
        if not future:
            # Prepare historical dataframe with/out exogenous regressors for training
            # Fix here
            df[self.params.target] = df[self.params.target].clip(0)
            features = [self.params.group_id, self.params.date_col, self.params.target]
            if 'dynamic_future_numerical' in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_numerical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future numerical missing: {e}")
            if 'dynamic_future_categorical' in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_categorical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future categorical missing: {e}")
            _df = df[features]
            _df = (
                _df.rename(
                    columns={
                        self.params.group_id: "unique_id",
                        self.params.date_col: "ds",
                        self.params.target: "y",
                    }
                )
            )
        else:
            # Prepare future dataframe with/out exogenous regressors for forecasting
            features = [self.params.group_id, self.params.date_col]
            if 'dynamic_future_numerical' in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_numerical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future numerical missing: {e}")
            if 'dynamic_future_categorical' in self.params.keys():
                try:
                    features = features + self.params.dynamic_future_categorical
                except Exception as e:
                    raise MissingFeatureError(f"Dynamic future categorical missing: {e}")
            _df = df[features]
            _df = (
                _df.rename(
                    columns={
                        self.params.group_id: "unique_id",
                        self.params.date_col: "ds",
                    }
                )
            )
        return _df

    def fit(self, x, y=None):
        self.model = StatsForecast(models=[self.model_spec], freq=self.freq, n_jobs=-1)
        self.model.fit(x)

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        _df = self.prepare_data(hist_df)
        _exogenous = self.prepare_data(val_df, future=True)
        self.fit(_df)
        quantiles = self.params.get("native_quantiles")
        level_kwargs = {"level": central_levels(quantiles)} if quantiles else {}
        if len(_exogenous.columns) == 2:
            forecast_df = self.model.predict(self.params["prediction_length"], **level_kwargs)
        else:
            forecast_df = self.model.predict(self.params["prediction_length"], _exogenous, **level_kwargs)
        target = [col for col in forecast_df.columns.to_list()
                  if col not in ["unique_id", "ds"] and "-lo-" not in col and "-hi-" not in col][0]
        if quantiles:
            forecast_df = self.native_quantile_columns(forecast_df, target, quantiles)
        forecast_df = forecast_df.reset_index(drop=True).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                target: self.params.target,
            }
        )
        # Fix here
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0)
        return forecast_df, self.model

    @staticmethod
    def native_quantile_columns(forecast_df: pd.DataFrame, target: str, quantiles) -> pd.DataFrame:
        """
        Replaces statsforecast's <model>-lo-<level> / <model>-hi-<level> columns with one
        QUANTILE_COLUMN_PREFIX column per requested quantile. The median is the point forecast.
        Quantiles are the model's raw band: unlike the point forecast, they are not clipped at 0
        here; calibration applies prediction_intervals.clip_lower to the bounds.
        """
        out = forecast_df[[c for c in forecast_df.columns if "-lo-" not in c and "-hi-" not in c]].copy()
        for q in quantiles:
            if q == 0.5:
                values = forecast_df[target]
            else:
                (level,) = central_levels([q])
                values = forecast_df[f"{target}-{'lo' if q < 0.5 else 'hi'}-{level}"]
            out[QUANTILE_COLUMN_PREFIX + quantile_key(q)] = values.to_numpy(dtype=float)
        return out

    def forecast(self, df: pd.DataFrame, spark=None):
        _df = df[df[self.params.target].notnull()]
        _df = self.prepare_data(_df)
        self.fit(_df)
        quantiles = self.params.get("native_quantiles")
        level_kwargs = {"level": central_levels(quantiles)} if quantiles else {}
        if 'dynamic_future_numerical' in self.params.keys() or 'dynamic_future_categorical' in self.params.keys():
            _last_date = _df["ds"].max()
            _future_df = df[
                (df[self.params["date_col"]] > np.datetime64(_last_date))
                & (df[self.params["date_col"]]
                   <= np.datetime64(_last_date + self.prediction_length_offset))
            ]
            _future_exogenous = self.prepare_data(_future_df, future=True)
            try:
                forecast_df = self.model.predict(self.params["prediction_length"], _future_exogenous,
                                                 **level_kwargs)
            except Exception as e:
                print(
                    f"Removing group_id {df[self.params.group_id][0]} as future exogenous "
                    f"regressors are not provided.")
                return pd.DataFrame(
                    columns=[self.params.date_col, self.params.target]
                )
        else:
            forecast_df = self.model.predict(self.params["prediction_length"], **level_kwargs)

        target = [col for col in forecast_df.columns.to_list()
                  if col not in ["unique_id", "ds"] and "-lo-" not in col and "-hi-" not in col][0]
        if quantiles:
            forecast_df = self.native_quantile_columns(forecast_df, target, quantiles)
        forecast_df = forecast_df.reset_index(drop=True).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                target: self.params.target,
            }
        )
        # Fix here
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0)
        return forecast_df, self.model


class StatsFcBaselineWindowAverage(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = WindowAverage(
            window_size=self.params.model_spec.window_size,
        )


class StatsFcBaselineSeasonalWindowAverage(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = SeasonalWindowAverage(
            season_length=self.params.model_spec.season_length,
            window_size=self.params.model_spec.window_size,
        )


class StatsFcBaselineNaive(StatsFcForecaster):
    native_quantiles_supported = True

    def __init__(self, params):
        super().__init__(params)
        self.model_spec = Naive()


class StatsFcBaselineSeasonalNaive(StatsFcForecaster):
    native_quantiles_supported = True

    def __init__(self, params):
        super().__init__(params)
        self.model_spec = SeasonalNaive(
            season_length=self.params.model_spec.season_length,
        )


class StatsFcAutoArima(StatsFcForecaster):
    native_quantiles_supported = True

    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoARIMA(
            season_length=self.params.model_spec.season_length,
            approximation=self.params.model_spec.approximation,
        )


class StatsFcAutoETS(StatsFcForecaster):
    native_quantiles_supported = True

    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoETS(
            season_length=self.params.model_spec.season_length,
            model=self.params.model_spec.model,
        )


class StatsFcAutoCES(StatsFcForecaster):
    native_quantiles_supported = True

    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoCES(
            season_length=self.params.model_spec.season_length,
            model=self.params.model_spec.model,
        )


class StatsFcAutoTheta(StatsFcForecaster):
    native_quantiles_supported = True

    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoTheta(
            season_length=self.params.model_spec.season_length,
            decomposition_type=self.params.model_spec.decomposition_type,
        )


class StatsFcAutoTbats(StatsFcForecaster):
    native_quantiles_supported = True

    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoTBATS(
            season_length=self.params.model_spec.season_length,
            use_boxcox=self.params.model_spec.use_boxcox,
            bc_lower_bound=self.params.model_spec.bc_lower_bound,
            bc_upper_bound=self.params.model_spec.bc_upper_bound,
            use_trend=self.params.model_spec.use_trend,
            use_damped_trend=self.params.model_spec.use_damped_trend,
            use_arma_errors=self.params.model_spec.use_arma_errors,
        )


class StatsFcAutoMfles(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoMFLES(
            test_size=self.params.prediction_length,
            season_length=self.params.model_spec.season_length,
        )


class StatsFcTSB(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = TSB(
            alpha_d=self.params.model_spec.alpha_d,
            alpha_p=self.params.model_spec.alpha_p,
        )


class StatsFcADIDA(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = ADIDA()


class StatsFcIMAPA(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = IMAPA()


class StatsFcCrostonClassic(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = CrostonClassic()


class StatsFcCrostonOptimized(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = CrostonOptimized()


class StatsFcCrostonSBA(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = CrostonSBA()

