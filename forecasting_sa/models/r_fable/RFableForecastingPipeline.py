from typing import Dict, Optional
import warnings
import numpy as np
import pandas as pd
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.lib.dplyr import DataFrame
from rpy2.robjects import rl

from rpy2.robjects.conversion import localconverter

from forecasting_sa.models.abstract_model import (
    ForecastingSARegressor,
    ForecastingSAVerticalizedDataRegressor,
)

# make sure R and the fable package are installed on your system
# TODO: Above comment should be moved to README at some point
base = importr("base")
tsibble = importr("tsibble")
fabletools = importr("fabletools")
distributional = importr("distributional")


class RFableModel(ForecastingSAVerticalizedDataRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.r_model: Optional[ro.vectors.DataFrame] = None

    def fit(self, X, y=None):
        rts = self.prepare_training_data(
            X,
        )

        self.r_model = fabletools.model(
            rts,
            rl(self._get_model_definition()()),
        )

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        # initialize as R's NULL object
        xreg_rts = ro.NULL
        h = ro.NULL
        if "xreg" in self.params.model_spec and len(self.params.model_spec.xreg) > 0:
            xreg_rts = self.prepare_forecast_data(X)
            if "prediction_length" in self.params and self.params.prediction_length:
                warnings.warn(
                    "Both prediction length and external regressors specified. External regressors will take precedence"
                )
        else:
            h = self.params.prediction_length

        r_forecast = fabletools.forecast(
            self.r_model, h=h, new_data=xreg_rts, simulate=False, times=0
        )
        forecast_pdf = self.conv_r_forecast_to_pandas(
            r_forecast,
            date_col=self.params.date_col,
            target=self.params.target,
            group_id=self.params.group_id,
        )
        return forecast_pdf

    def _get_model_definition(self):
        """Implements Fable model factory

        Returns:

        """
        return get_model_definition(self.params)

    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "xreg" in self.params.model_spec:
            regressor_variables = [xreg for xreg in self.params.model_spec.xreg]
        else:
            regressor_variables = []

        df_rfable = df[
            [self.params.group_id, self.params.date_col, self.params.target]
            + regressor_variables
        ].rename(
            columns={
                self.params.group_id: "unique_id",
                self.params.date_col: "ds",
                self.params.target: "y",
            }
        )
        rts = self.prepare_data(df_rfable)

        return rts

    def prepare_forecast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO Combine the prepare_data method to one single method
        df_rfable = df[
            [self.params.group_id, self.params.date_col]
            + [xreg for xreg in self.params.model_spec.xreg]
        ].rename(
            columns={
                self.params.group_id: "unique_id",
                self.params.date_col: "ds",
            }
        )
        rts = self.prepare_data(df_rfable)
        return rts

    def prepare_data(self, df_rfable: pd.DataFrame) -> pd.DataFrame:
        with localconverter(ro.default_converter + pandas2ri.converter):
            rdf = ro.conversion.py2rpy(df_rfable)

        rts = tsibble.as_tsibble(
            DataFrame(rdf).mutate(ds=rl(self._get_datetime_conversion(self.freq))),
            index="ds",
            key="unique_id",
        )

        return rts

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, float]:
        to_pred_df = val_df.copy()
        to_pred_df[self.params["target"]] = np.nan
        pred_df = self.predict(to_pred_df)
        smape = mean_absolute_percentage_error(
            val_df[self.params["target"]],
            pred_df[self.params["target"]],
            symmetric=True,
        )
        return {"smape": smape}

    @staticmethod
    def _get_datetime_conversion(freq: str):
        if freq == "D":
            r_conversion_func = "lubridate::ymd(ds)"
        elif freq == "M":
            r_conversion_func = "tsibble::yearmonth(ds)"
        elif freq == "Q":
            r_conversion_func = "tsibble::yearquarter(ds)"
        elif freq == "W":
            r_conversion_func = "tsibble::yearweek(ds)"
        elif freq == "H":
            r_conversion_func = "lubridate::ymd_hms(ds)"
        else:
            raise ValueError(f"Frequency {freq} is not available for RFable models")

        return r_conversion_func

    @staticmethod
    def conv_r_forecast_to_pandas(
        r_forecast, date_col: str, target: str, group_id: str
    ):
        # get 95% prediction interval
        fcst_dist = distributional.hilo(r_forecast, level=95.0)
        #fcst_dist = fabletools.unpack_hilo(fcst_dist, "95%")

        # convert back to pandas
        fcst_r_df = base.as_data_frame(
            DataFrame(fcst_dist).select(
                "unique_id", "ds", ".mean"
            )
        )
        # R has a date type while pandas only has datetimes so we need
        # to convert to datetime to make sure conversion to pandas works
        fcst_r_df = DataFrame(fcst_r_df).mutate(ds=rl("lubridate::as_datetime(ds)"))

        with localconverter(ro.default_converter + pandas2ri.converter):
            forecast_pdf = ro.conversion.rpy2py(fcst_r_df)

        forecast_pdf = forecast_pdf.rename(
            columns={
                "ds": date_col,
                ".mean": target,
                "unique_id": group_id,
            }
        )

        # TODO: Right now the scoring pipeline uses applyinpandas and
        #  expects the output schema to just include group_id, data_col
        #  and target so we remove the intervals for now.
        #forecast_pdf = forecast_pdf.drop(columns=["95%_lower", "95%_upper"])
        forecast_pdf = forecast_pdf[[date_col, group_id, target]]
        return forecast_pdf


class RFableArima:
    def __init__(self, params):
        self.params = params

    def __call__(self):

        if self.params.model_spec.season_length is None:
            self.params.model_spec.season_length = "NULL"

        model_string = (
            f"fable::ARIMA(y ~ PDQ(period={self.params.model_spec.season_length}))"
        )

        model_string = _add_xreg_to_model_string(self.params.model_spec, model_string)

        return model_string


class RFableETS:
    def __init__(self, params):
        self.params = params

    def __call__(self):

        if self.params.model_spec.season_length is None:
            self.params.model_spec.season_length = "NULL"

        return f"fable::ETS(y ~ season(method = c('N', 'A', 'M'), period={self.params.model_spec.season_length}))"


class RFableNNETAR:
    def __init__(self, params):
        self.params = params

    def __call__(self):

        if self.params.model_spec.season_length is None:
            self.params.model_spec.season_length = "NULL"

        model_string = (
            f"fable::NNETAR(y ~ AR(P=1, period={self.params.model_spec.season_length}))"
        )

        model_string = _add_xreg_to_model_string(self.params.model_spec, model_string)

        return model_string


class RFableEnsemble:
    def __init__(self, params):
        self.params = params

    def __call__(self):
        model_list = [
            get_model_definition(model["model"])()
            for model in self.params.model_spec.models
        ]
        model_comb = "(" + " + ".join(model_list) + f")/{len(model_list)}"
        return model_comb


class RDynamicHarmonicRegression:
    def __init__(self, params):
        self.params = params

    def __call__(self):

        if self.params.model_spec.fourier_terms is None:
            raise Warning("0 Fourier terms specified. Use non-seasonal Arima")
            rhs = "PDQ(0, 0, 0)"
        else:
            rhs = (
                " + ".join(
                    [
                        f"fourier({value.season_length}, K={value.fourier_order})"
                        for dict in self.params.model_spec.fourier_terms
                        for key, value in dict.items()
                    ]
                )
                + " + PDQ(0, 0, 0)"
            )

        return f"fable::ARIMA(y ~ {rhs})"


def get_model_definition(params):
    """Implements Fable model factory

    Returns:

    """
    if params.name == "RFableArima":
        return RFableArima(params)
    elif params.name == "RDynamicHarmonicRegression":
        return RDynamicHarmonicRegression(params)
    elif params.name == "RFableETS":
        return RFableETS(params)
    elif params.name == "RFableNNETAR":
        return RFableNNETAR(params)
    elif params.name == "RFableEnsemble":
        return RFableEnsemble(params)
    else:
        raise ValueError(f"Model {params.name} is not available")


def _add_xreg_to_model_string(model_spec: Dict, model_string: str) -> str:
    # check if external regressors were provided
    if "xreg" in model_spec and len(model_spec["xreg"]) > 0:
        model_string = (
            model_string[:-1]
            + " + "
            + " + ".join(model_spec["xreg"])
            + model_string[-1]
        )
    return model_string
