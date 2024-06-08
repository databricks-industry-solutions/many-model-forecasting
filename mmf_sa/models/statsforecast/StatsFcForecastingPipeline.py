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
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    WindowAverage,
    SeasonalWindowAverage,
    Naive,
    SeasonalNaive,
)
from mmf_sa.models.abstract_model import ForecastingRegressor


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
            df[self.params.target] = df[self.params.target].clip(0.1)
            if 'dynamic_future' in self.params.keys():
                try:
                    df_statsfc = (
                        df[[self.params.group_id, self.params.date_col, self.params.target]
                           + self.params.dynamic_future]
                    )
                except Exception as e:
                    raise Exception(f"Exogenous regressors missing: {e}")
            else:
                df_statsfc = df[[self.params.group_id, self.params.date_col, self.params.target]]

            df_statsfc = (
                df_statsfc.rename(
                    columns={
                        self.params.group_id: "unique_id",
                        self.params.date_col: "ds",
                        self.params.target: "y",
                    }
                )
            )
        else:
            # Prepare future dataframe with/out exogenous regressors for forecasting
            if 'dynamic_future' in self.params.keys():
                try:
                    df_statsfc = (
                        df[[self.params.group_id, self.params.date_col]
                           + self.params.dynamic_future]
                    )
                except Exception as e:
                    raise Exception(f"Exogenous regressors missing: {e}")
            else:
                df_statsfc = df[[self.params.group_id, self.params.date_col]]

            df_statsfc = (
                df_statsfc.rename(
                    columns={
                        self.params.group_id: "unique_id",
                        self.params.date_col: "ds",
                    }
                )
            )
        return df_statsfc

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
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame, spark=None):
        _df = df[df[self.params.target].notnull()]
        _df = self.prepare_data(_df)
        self.fit(_df)
        if 'dynamic_future' in self.params.keys():
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
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
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
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = Naive()


class StatsFcBaselineSeasonalNaive(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = SeasonalNaive(
            season_length=self.params.model_spec.season_length,
        )


class StatsFcAutoArima(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoARIMA(
            season_length=self.params.model_spec.season_length,
            approximation=self.params.model_spec.approximation,
        )


class StatsFcAutoETS(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoETS(
            season_length=self.params.model_spec.season_length,
            model=self.params.model_spec.model,
        )


class StatsFcAutoCES(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoCES(
            season_length=self.params.model_spec.season_length,
            model=self.params.model_spec.model,
        )


class StatsFcAutoTheta(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoTheta(
            season_length=self.params.model_spec.season_length,
            decomposition_type=self.params.model_spec.decomposition_type,
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

