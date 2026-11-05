import inspect
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
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
from mmf_sa.exceptions import MissingFeatureError, DataPreparationError, ConfigurationError


class StatsFcForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model_spec = None
        self.model = None

    def build_model_spec(self, model_cls, **defaults):
        """
        Instantiates a statsforecast model with every model_spec key that its constructor accepts, so any
        constructor argument (e.g. AutoARIMA's max_p, stepwise, ic, blambda) can be set from the config.
        Keys are passed as given, None included: for some arguments None has a meaning of its own
        (AutoARIMA's approximation=None decides from the series length and season length).
        Parameters:
            model_cls: The statsforecast model class.
            defaults: Arguments the wrapper sets unless model_spec sets them.
        Returns: The statsforecast model.
        Raises: ConfigurationError if model_spec sets a key the constructor does not accept. Keys left
            empty (None) are ignored, as the frequency configs keep AutoArima's unused level: key.
        """
        spec = self.params.get("model_spec")
        spec = OmegaConf.to_container(spec, resolve=True) if OmegaConf.is_config(spec) else dict(spec or {})
        accepted = set(inspect.signature(model_cls.__init__).parameters) - {"self"}
        unknown = sorted(k for k, v in spec.items() if k not in accepted and v is not None)
        if unknown:
            raise ConfigurationError(
                f"{self.params.get('name', model_cls.__name__)}: model_spec has keys that {model_cls.__name__} "
                f"does not accept: {', '.join(unknown)}. Accepted keys: {', '.join(sorted(accepted))}"
            )
        return model_cls(**{**defaults, **{k: v for k, v in spec.items() if k in accepted}})

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
        if len(_exogenous.columns) == 2:
            forecast_df = self.model.predict(self.params["prediction_length"])
        else:
            forecast_df = self.model.predict(self.params["prediction_length"], _exogenous)
        target = [col for col in forecast_df.columns.to_list()
                       if col not in ["unique_id", "ds"]][0]
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

    def forecast(self, df: pd.DataFrame, spark=None):
        _df = df[df[self.params.target].notnull()]
        _df = self.prepare_data(_df)
        self.fit(_df)
        if 'dynamic_future_numerical' in self.params.keys() or 'dynamic_future_categorical' in self.params.keys():
            _last_date = _df["ds"].max()
            _future_df = df[
                (df[self.params["date_col"]] > np.datetime64(_last_date))
                & (df[self.params["date_col"]]
                   <= np.datetime64(_last_date + self.prediction_length_offset))
            ]
            _future_exogenous = self.prepare_data(_future_df, future=True)
            try:
                forecast_df = self.model.predict(self.params["prediction_length"], _future_exogenous)
            except Exception as e:
                print(
                    f"Removing group_id {df[self.params.group_id][0]} as future exogenous "
                    f"regressors are not provided.")
                return pd.DataFrame(
                    columns=[self.params.date_col, self.params.target]
                )
        else:
            forecast_df = self.model.predict(self.params["prediction_length"])

        target = [col for col in forecast_df.columns.to_list() if col not in ["unique_id", "ds"]][0]
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
        self.model_spec = self.build_model_spec(WindowAverage)


class StatsFcBaselineSeasonalWindowAverage(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(SeasonalWindowAverage)


class StatsFcBaselineNaive(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(Naive)


class StatsFcBaselineSeasonalNaive(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(SeasonalNaive)


class StatsFcAutoArima(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(AutoARIMA)


class StatsFcAutoETS(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(AutoETS)


class StatsFcAutoCES(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(AutoCES)


class StatsFcAutoTheta(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(AutoTheta)


class StatsFcAutoTbats(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(AutoTBATS)


class StatsFcAutoMfles(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(AutoMFLES, test_size=self.params.prediction_length)


class StatsFcTSB(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(TSB)


class StatsFcADIDA(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(ADIDA)


class StatsFcIMAPA(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(IMAPA)


class StatsFcCrostonClassic(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(CrostonClassic)


class StatsFcCrostonOptimized(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(CrostonOptimized)


class StatsFcCrostonSBA(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = self.build_model_spec(CrostonSBA)

