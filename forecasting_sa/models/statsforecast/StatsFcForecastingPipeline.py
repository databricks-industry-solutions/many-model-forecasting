from typing import Dict

import pandas as pd
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
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

from forecasting_sa.models.abstract_model import ForecastingSAVerticalizedDataRegressor


class StatsFcForecaster(ForecastingSAVerticalizedDataRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model_spec = None

    def fit(self, X, y=None):
        pass

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.params.target] = df[self.params.target].clip(0.1)
        df_statsfc = (
            df[[self.params.group_id, self.params.date_col, self.params.target]]
            .rename(
                columns={
                    self.params.group_id: "unique_id",
                    self.params.date_col: "ds",
                    self.params.target: "y",
                }
            )
            .set_index("unique_id")
        )
        return df_statsfc

    def predict(self, X):
        _df = self.prepare_data(X)
        model = StatsForecast(models=[self.model_spec], freq=self.freq, n_jobs=-1)
        model.fit(_df)
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
    ) -> Dict[str, float]:
        pred_df = self.predict(hist_df)
        smape = mean_absolute_percentage_error(
            val_df[self.params["target"]],
            pred_df[self.params["target"]],
            symmetric=True,
        )
        return {"smape": smape}


class StatsFcAutoArima(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoARIMA(
            season_length=self.params.model_spec.season_length
            if self.params.model_spec.season_length
            else 1,
            approximation=self.params.model_spec.approximation
            if self.params.model_spec.approximation
            else False,
        )


class StatsFcAutoETS(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoETS(
            season_length=self.params.model_spec.season_length
            if self.params.model_spec.season_length
            else 1
        )


class StatsFcAutoCES(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoCES(
            season_length=self.params.model_spec.season_length
            if self.params.model_spec.season_length
            else 1,
            model="Z",
        )


class StatsFcAutoTheta(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = AutoTheta(
            season_length=self.params.model_spec.season_length
            if self.params.model_spec.season_length
            else 1,
            decomposition_type="multiplicative",
        )


class StatsFcADIDA(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = ADIDA()


class StatsFcIMAPA(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = IMAPA()


class StatsFcTSB(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = TSB(
            alpha_d=float(params.get("alpha_d", 0.2)),
            alpha_p=float(params.get("alpha_p", 0.2)),
        )


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


class StatsFcBaselineWindowAverage(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = WindowAverage(window_size=int(params.get("window_size", 7)))


class StatsFcBaselineSeasonalWindowAverage(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = SeasonalWindowAverage(
            season_length=int(params.get("season_length", 7)),
            window_size=int(params.get("window_size", 7)),
        )


class StatsFcBaselineNaive(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = Naive()


class StatsFcBaselineSeasonalNaive(StatsFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.model_spec = SeasonalNaive(
            season_length=int(params.get("season_length", 7))
        )
