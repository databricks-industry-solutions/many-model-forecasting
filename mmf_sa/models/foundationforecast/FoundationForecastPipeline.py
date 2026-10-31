import gc
import logging
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import mlflow
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types import ColSpec, Schema

from mmf_sa.intervals.quantiles import QUANTILE_COLUMN_PREFIX, quantile_key
from mmf_sa.models.abstract_model import ForecastingRegressor, MODEL_PIP_REQUIREMENTS, warn_once as _warn_once
from mmf_sa.models.foundationforecast._runtime import ensure_runtime
from mmf_sa.exceptions import (
    DataPreparationError,
    MissingFeatureError,
    ModelPredictionError,
)

_logger = logging.getLogger(__name__)

# MMF frequency codes -> pandas aliases accepted by foundationforecast.
FF_FREQ = {"H": "h", "D": "D", "W": "W", "M": "ME"}

NON_COMMERCIAL_LICENSES = frozenset({
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "timesfm-non-commercial-license-v1.0",
})

# Licenses that require a commercial agreement above a company-size threshold.
REVENUE_CAPPED_LICENSES = frozenset({
    "mistral-modified-mit",
})

COVARIATE_PARAMS = (
    "static_features",
    "dynamic_future_numerical",
    "dynamic_future_categorical",
    "dynamic_historical_numerical",
    "dynamic_historical_categorical",
)
# foundationforecast only accepts covariates whose values are known over the forecast horizon.
FUTURE_COVARIATE_PARAMS = ("dynamic_future_numerical", "dynamic_future_categorical")

_FF_ALIAS = "ff_forecast"
_current_checkpoint = None


def ff_level(q: float) -> float:
    """The level requested from foundationforecast for quantile q: q rounded outward (away from the median) to 3
    decimals, within [0.001, 0.999], the levels foundationforecast accepts."""
    milli = (math.floor if q < 0.5 else math.ceil)(round(q * 1000, 6))
    return min(max(milli, 1), 999) / 1000


def ff_quantile_column(q: float) -> str:
    """Output column foundationforecast uses for quantile ``q``, e.g. ``ff_forecast-q-2.5`` for 0.025.

    Mirrors foundationforecast.core.quantiles.quantile_column_name, which this module does not import
    so that it loads without the library.
    """
    whole, frac = divmod(round(float(q) * 1000), 10)
    return f"{_FF_ALIAS}-q-{whole}" if frac == 0 else f"{_FF_ALIAS}-q-{whole}.{frac}"


def _step_offset(freq: str):
    return {
        "H": pd.DateOffset(hours=1),
        "D": pd.DateOffset(days=1),
        "W": pd.DateOffset(weeks=1),
        "M": pd.offsets.MonthEnd(1),
    }[freq]


def _resolve_dtype(value):
    import torch
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if value not in dtypes:
        raise ValueError(f"Unsupported dtype {value!r}; expected one of {sorted(dtypes)}.")
    return dtypes[value]


def _release_previous_checkpoint(checkpoint: Tuple[str, str]) -> None:
    """Free the weights of the previously used checkpoint when switching models.

    foundationforecast's weight cache evicts the old model, but its modules sit in
    reference cycles and keep GPU memory until the cyclic garbage collector runs.
    """
    global _current_checkpoint
    if _current_checkpoint is not None and _current_checkpoint != checkpoint:
        from foundationforecast.core.model_weight_cache import get_model_weight_cache
        get_model_weight_cache().clear()
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _current_checkpoint = checkpoint


def build_ff_model(ff_class: str, repo: str, ff_kwargs: dict):
    """Instantiate a foundationforecast model. ``ff_kwargs`` holds raw YAML values."""
    ensure_runtime()
    _release_previous_checkpoint((ff_class, repo))
    from foundationforecast import models as ff_models

    kwargs = dict(ff_kwargs)
    if "dtype" in kwargs:
        kwargs["dtype"] = _resolve_dtype(kwargs["dtype"])
    return getattr(ff_models, ff_class)(repo_id=repo, alias=_FF_ALIAS, **kwargs)


def _weekly_alias(df: pd.DataFrame) -> str:
    """Anchored weekly alias (e.g. "W-MON") of a frame whose ``ds`` share one weekday."""
    days = df["ds"].dt.day_name().str[:3].str.upper().unique()
    if len(days) != 1:
        raise DataPreparationError(
            f"Weekly series must all fall on the same weekday to use covariates; found {sorted(days)}."
        )
    return f"W-{days[0]}"


def horizon_dates(last: pd.Series, prediction_length: int, freq: str) -> pd.DataFrame:
    """``unique_id, ds`` rows for the ``prediction_length`` steps after each series' last date."""
    offset = _step_offset(freq)
    return pd.DataFrame({
        "unique_id": np.repeat(last.index.to_numpy(), prediction_length),
        "ds": [ts + offset * step for ts in last for step in range(1, prediction_length + 1)],
    })


def _warn_no_future_covariates(name: str) -> None:
    _warn_once(
        f"no_future_covariates:{name}",
        f"{name}: no future covariate values were provided (for example, no "
        "scoring_data); forecasting without covariates.",
    )


def check_covariates(pdf: pd.DataFrame, covariates, accepts_strings: bool, name: str) -> None:
    """Reject missing or unusable covariate values; cast numeric ones to float64 in place."""
    for col in covariates:
        if pdf[col].isna().any():
            bad = pdf.loc[pdf[col].isna(), "unique_id"].iloc[0]
            raise DataPreparationError(
                f"Covariate '{col}' contains missing values (e.g. series {bad!r})."
            )
        if pd.api.types.is_numeric_dtype(pdf[col]):
            pdf[col] = pdf[col].astype(np.float64)
        elif not accepts_strings:
            raise DataPreparationError(
                f"{name}: covariate '{col}' has non-numeric dtype {pdf[col].dtype}. "
                "Label-encode it to numbers first, or use FFChronos2 or FFChronos2Small, "
                "which accept categorical values."
            )


def build_horizon(
    hist: pd.DataFrame, future: pd.DataFrame, prediction_length: int, freq: str,
    covariates, accepts_strings: bool, name: str,
) -> pd.DataFrame:
    """Covariate rows for the ``prediction_length`` steps after each series in ``hist``.

    Both frames use string ``unique_id``s and datetime ``ds``; ``future`` may hold rows
    outside the horizon, which are ignored.
    """
    last = hist.groupby("unique_id", sort=True)["ds"].max()
    expected = horizon_dates(last, prediction_length, freq)
    horizon = expected.merge(
        future[["unique_id", "ds", *covariates]], on=["unique_id", "ds"], how="left", indicator=True
    )
    if len(horizon) != len(expected):
        bad = horizon.loc[horizon.duplicated(["unique_id", "ds"]), "unique_id"].iloc[0]
        raise DataPreparationError(
            f"Future covariates contain duplicate dates (e.g. series {bad!r})."
        )
    missing = horizon.loc[horizon["_merge"] == "left_only", "unique_id"].unique()
    if len(missing):
        raise DataPreparationError(
            f"Covariates must be known for all {prediction_length} forecast steps of every "
            f"series; {len(missing)} series are missing future rows (e.g. {missing[0]!r})."
        )
    horizon = horizon.drop(columns="_merge")
    check_covariates(horizon, covariates, accepts_strings, name)
    return horizon


def forecast_series(
    ff_model, df: pd.DataFrame, prediction_length: int, freq: str, X_df: pd.DataFrame = None, quantiles=None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray]]]:
    """Forecast a long ``unique_id, ds, y`` frame, plus covariate columns when ``X_df`` is given.

    ``X_df`` holds ``unique_id, ds`` and the covariate values for the forecast horizon.
    ``quantiles`` (levels in (0, 1)) asks foundationforecast for native quantiles.
    Returns ``{unique_id: (dates, values, quantile_values)}``, where ``quantile_values`` maps each
    level the library returned to its values (empty without ``quantiles``). Dates are rebuilt from
    each series' last observation with MMF's step offset, because pandas' "W" alias anchors
    foundationforecast's future dates to Sundays.
    """
    kwargs = {}
    ff_freq = FF_FREQ[freq]
    if X_df is not None:
        kwargs = {"X_df": X_df, "futr_exog_list": [c for c in X_df.columns if c not in ("unique_id", "ds")]}
        # Chronos-2 rejects horizon timestamps that differ from the ones it derives from freq.
        if freq == "W":
            ff_freq = _weekly_alias(df)
    if quantiles:
        kwargs["quantiles"] = list(quantiles)
    fcst = ff_model.forecast(df, h=prediction_length, freq=ff_freq, **kwargs)
    fcst = fcst.sort_values(["unique_id", "ds"], kind="stable")
    values = {
        uid: grp[_FF_ALIAS].to_numpy(dtype=np.float64)
        for uid, grp in fcst.groupby("unique_id", sort=False)
    }
    # The library names quantile columns by level; levels it did not return are left out.
    columns = {level: ff_quantile_column(level) for level in quantiles or () if ff_quantile_column(level) in fcst}
    native = {
        uid: {level: grp[col].to_numpy(dtype=np.float64) for level, col in columns.items()}
        for uid, grp in fcst.groupby("unique_id", sort=False)
    }
    offset = _step_offset(freq)
    out = {}
    for uid, last in df.groupby("unique_id", sort=False)["ds"].max().items():
        forecast = values.get(uid)
        if forecast is None or len(forecast) != prediction_length:
            raise ModelPredictionError(
                f"foundationforecast returned {0 if forecast is None else len(forecast)} "
                f"forecast steps for series {uid!r}; expected {prediction_length}."
            )
        quantile_values = native.get(uid, {})
        for level, quantile in quantile_values.items():
            if len(quantile) != prediction_length:
                raise ModelPredictionError(
                    f"foundationforecast returned {len(quantile)} steps for quantile {level} of series {uid!r}; "
                    f"expected {prediction_length}."
                )
        dates = np.array(
            [(last + offset * step).to_datetime64() for step in range(1, prediction_length + 1)],
            dtype="datetime64[ns]",
        )
        out[uid] = (dates, forecast, quantile_values)
    return out


class FoundationForecastForecaster(ForecastingRegressor):
    """Base class for models served through the foundationforecast library.

    Family subclasses set ``ff_class`` (the foundationforecast model class) and
    ``hparams`` (the YAML keys forwarded to its constructor). Checkpoint leaves
    set ``self.repo``. Prediction runs on the driver's GPU only.
    """

    ff_class: str = None
    hparams: Tuple[str, ...] = ()
    # Must match foundationforecast's supports_native_futr_exog() for the checkpoint.
    supports_future_covariates: bool = False
    # Chronos-2 encodes non-numeric covariates itself; the other models cast them to float32.
    accepts_categorical_strings: bool = False
    # Every adapter returns quantiles; native_quantile_range is set per family below (None: any level).
    native_quantiles_supported = True

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = None
        self.model = None
        self._warn_unsupported_config()

    def _warn_unsupported_config(self):
        name = self.params.get("name", type(self).__name__)
        configured = [k for k in COVARIATE_PARAMS if self.params.get(k)]
        if self.supports_future_covariates:
            ignored = [k for k in configured if k not in FUTURE_COVARIATE_PARAMS]
            reason = "only known-future covariates are supported"
        else:
            ignored = configured
            reason = "this model does not support covariates"
        if ignored:
            _warn_once(
                f"covariates:{name}",
                f"{name}: {reason}; ignoring {', '.join(ignored)}.",
            )
        license_id = self.params.get("license")
        if license_id in NON_COMMERCIAL_LICENSES:
            _warn_once(
                f"license:{name}",
                f"{name} is released under the non-commercial license '{license_id}'. "
                "Check the license terms before using it.",
            )
        elif license_id in REVENUE_CAPPED_LICENSES:
            _warn_once(
                f"license:{name}",
                f"{name} is released under the license '{license_id}', which requires a "
                "commercial license from the rights holder for companies above a revenue "
                "threshold. Check the license terms before using it.",
            )

    def ff_kwargs(self) -> dict:
        kwargs = {}
        for key in self.hparams:
            value = self.params.get(key)
            if value is not None:
                kwargs[key] = value
        return kwargs

    def _get_ff_model(self):
        if self.model is None:
            self.model = build_ff_model(self.ff_class, self.repo, self.ff_kwargs())
        return self.model

    def future_covariates(self) -> list:
        """Configured known-future covariates this model uses, numerical first."""
        if not self.supports_future_covariates:
            return []
        columns = [c for key in FUTURE_COVARIATE_PARAMS for c in (self.params.get(key) or [])]
        return list(dict.fromkeys(columns))

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> pd.DataFrame:
        """Rename MMF columns to foundationforecast's ``unique_id, ds, y`` and validate.

        Keeps the model's future covariates. With ``future=True`` the target is dropped and
        only the columns are checked; ``_horizon_frame`` validates the rows it selects.
        """
        group_col = self.params.group_id
        date_col = self.params.date_col
        target_col = self.params.target
        covariates = self.future_covariates()
        missing = [c for c in covariates if c not in df.columns]
        if missing:
            raise MissingFeatureError(f"Covariate columns missing from the data: {missing}.")
        columns = [group_col, date_col] + ([] if future else [target_col]) + covariates
        pdf = df[columns].rename(
            columns={group_col: "unique_id", date_col: "ds", target_col: "y"}
        )
        pdf["ds"] = pd.to_datetime(pdf["ds"])
        if future:
            return pdf
        if pdf["y"].isna().any():
            bad = pdf.loc[pdf["y"].isna(), "unique_id"].iloc[0]
            raise DataPreparationError(
                f"Column '{target_col}' contains missing values (e.g. series {bad!r}); "
                "foundationforecast models require complete histories."
            )
        if self.freq == "M" and not pdf["ds"].dt.is_month_end.all():
            bad = pdf.loc[~pdf["ds"].dt.is_month_end, "ds"].iloc[0]
            raise DataPreparationError(
                f"freq='M' requires month-end dates in '{date_col}', but found {bad.date()}. "
                "Align the dates to the last day of each month."
            )
        check_covariates(pdf, covariates, self.accepts_categorical_strings, self._name())
        pdf["y"] = pdf["y"].astype(np.float64)
        return pdf.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def _name(self) -> str:
        return self.params.get("name", type(self).__name__)

    def _horizon_frame(self, pdf: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        """Covariate values for the ``prediction_length`` steps after each series in ``pdf``.

        ``pdf`` is the prepared history with string ``unique_id``s.
        """
        future = self.prepare_data(val_df, future=True)
        future["unique_id"] = future["unique_id"].astype(str)
        return build_horizon(
            pdf, future, self.params["prediction_length"], self.freq,
            self.future_covariates(), self.accepts_categorical_strings, self._name(),
        )

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None, curr_date=None, spark=None):
        # Distributed prediction is not implemented; every call uses the driver GPU.
        return self._predict_single(hist_df, val_df)

    def _predict_distributed(self, hist_df: pd.DataFrame, spark):
        """Multi-GPU prediction across Spark workers. Not implemented.

        Intended design: repartition the series into one partition per GPU and run
        foundationforecast inside a Pandas UDF, as ``ChronosForecaster._predict_distributed``
        does for Chronos.
        """
        raise NotImplementedError(
            "Distributed prediction is not supported for foundationforecast models yet; "
            "they run on the driver's GPU."
        )

    def _predict_single(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Driver-only single-GPU prediction path.

        ``val_df`` supplies the future covariates: the backtest window's actuals, or the
        future rows of the scoring data.
        """
        pdf = self.prepare_data(hist_df)
        original_ids = {}
        for uid in pdf["unique_id"].unique():
            original_ids.setdefault(str(uid), uid)
        if len(original_ids) != pdf["unique_id"].nunique():
            raise DataPreparationError(
                f"Values of '{self.params.group_id}' are not unique once converted to strings."
            )
        pdf["unique_id"] = pdf["unique_id"].astype(str)

        X_df = None
        if self.future_covariates():
            if val_df is None or val_df.empty:
                _warn_no_future_covariates(self._name())
                pdf = pdf[["unique_id", "ds", "y"]]
            else:
                X_df = self._horizon_frame(pdf, val_df)

        levels = self._native_levels()
        series = forecast_series(
            self._get_ff_model(), pdf, self.params["prediction_length"], self.freq, X_df=X_df,
            quantiles=sorted(set(levels.values())),
        )
        data = {
            self.params.group_id: [original_ids[uid] for uid in series],
            self.params.date_col: [dates for dates, _, _ in series.values()],
            self.params.target: [values for _, values, _ in series.values()],
        }
        returned = [level for level, requested in levels.items()
                    if all(requested in qs for _, _, qs in series.values())]
        if len(returned) < len(levels):
            _warn_once(
                f"native_quantiles_missing:{self._name()}",
                f"{self._name()}: foundationforecast did not return the quantiles "
                f"{sorted(set(levels) - set(returned))}; their columns are left out, so intervals at those levels "
                "are not calibrated from native bands.",
            )
        for level in returned:
            data[QUANTILE_COLUMN_PREFIX + quantile_key(level)] = [qs[levels[level]] for _, _, qs in series.values()]
        forecast_df = pd.DataFrame(data)
        return forecast_df, self.model

    def _native_levels(self) -> dict:
        """
        Quantile levels for native intervals (native_levels), mapped to the level requested from foundationforecast
        for each, or {} when none are needed.
        """
        # foundationforecast accepts levels with at most 3 decimals and raises otherwise (e.g. 0.0125 for a 97.5%
        # interval): such levels are rounded outward, so the band is a little wider, and CQR calibrates it.
        levels = {level: ff_level(level) for level in self.native_levels()}
        rounded = [level for level, ff in levels.items() if ff != level]
        if rounded:
            _warn_once(
                f"native_quantiles_precision:{self._name()}",
                f"{self._name()}: foundationforecast only accepts quantile levels with at most 3 decimals, so "
                f"{rounded} are requested as {[levels[level] for level in rounded]}.",
            )
        return levels

    def forecast(self, df: pd.DataFrame, spark=None):
        observed = df[self.params.target].notnull()
        return self.predict(df[observed], val_df=df[~observed], spark=spark)

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> list:
        pred_df, _ = self.predict(hist_df, val_df, curr_date, spark)
        return self.series_metrics(pred_df, val_df, curr_date)

    def register(self, registered_model_name: str, input_example: pd.DataFrame = None):
        """Log and register the model.

        ``input_example`` is training data in MMF's column names. With covariates, the
        registered input schema takes the covariate types from it, so pass real data.
        """
        covariates = self.future_covariates()
        model = FoundationForecastMLflowModel(
            ff_class=self.ff_class,
            repo=self.repo,
            ff_kwargs=self.ff_kwargs(),
            prediction_length=int(self.params["prediction_length"]),
            freq=self.freq,
            futr_exog_list=covariates,
            accepts_categorical_strings=self.accepts_categorical_strings,
            name=self._name(),
        )
        output_schema = Schema([
            ColSpec("string", "unique_id"),
            ColSpec("datetime", "ds"),
            ColSpec("double", "y"),
        ])
        if covariates:
            input_example = self._covariate_input_example(input_example)
            input_schema = infer_signature(input_example).inputs
        else:
            input_schema = output_schema
            dates = pd.date_range(end="2024-12-31", periods=52, freq=FF_FREQ[self.freq])
            input_example = pd.DataFrame({
                "unique_id": "series_0",
                "ds": dates,
                "y": np.random.rand(len(dates)),
            })
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=MODEL_PIP_REQUIREMENTS["foundationforecast"],
        )

    def _covariate_input_example(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """One series in the registered model's input layout: history rows, then
        ``prediction_length`` future rows whose ``y`` is null."""
        h = int(self.params["prediction_length"])
        covariates = self.future_covariates()
        group_col, date_col, target_col = self.params.group_id, self.params.date_col, self.params.target
        if data is not None and set(covariates) <= set(data.columns) and len(data) > h:
            first = data[group_col].iloc[0]
            example = (
                data.loc[data[group_col] == first, [group_col, date_col, target_col, *covariates]]
                .rename(columns={group_col: "unique_id", date_col: "ds", target_col: "y"})
                .sort_values("ds")
                .reset_index(drop=True)
            )
            example["unique_id"] = example["unique_id"].astype(str)
            example["ds"] = pd.to_datetime(example["ds"])
        else:
            dates = pd.date_range(end="2024-12-31", periods=52 + h, freq=FF_FREQ[self.freq])
            example = pd.DataFrame({"unique_id": "series_0", "ds": dates, "y": np.random.rand(len(dates))})
            for col in covariates:
                example[col] = 0.0
        example["y"] = example["y"].astype(np.float64)
        example.loc[example.index[-h:], "y"] = np.nan
        return example


class FoundationForecastMLflowModel(mlflow.pyfunc.PythonModel):
    """Takes a long ``unique_id, ds, y`` frame and returns the forecast in the same layout.

    With covariates, the frame also holds the covariate columns, and rows whose ``y`` is
    null supply the covariate values for the forecast horizon.
    """

    def __init__(
        self, ff_class: str, repo: str, ff_kwargs: dict, prediction_length: int, freq: str,
        futr_exog_list=(), accepts_categorical_strings: bool = False, name: str = None,
    ):
        self.ff_class = ff_class
        self.repo = repo
        self.ff_kwargs = dict(ff_kwargs)
        self.prediction_length = prediction_length
        self.freq = freq
        self.futr_exog_list = list(futr_exog_list)
        self.accepts_categorical_strings = accepts_categorical_strings
        self.name = name or ff_class
        self._model = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if self._model is None:
            self._model = build_ff_model(self.ff_class, self.repo, self.ff_kwargs)
        # Models registered before covariate support have no futr_exog_list.
        covariates = list(getattr(self, "futr_exog_list", ()))
        df = model_input[["unique_id", "ds", "y", *covariates]].copy()
        df["unique_id"] = df["unique_id"].astype(str)
        df["ds"] = pd.to_datetime(df["ds"])
        X_df = None
        if covariates:
            accepts_strings = getattr(self, "accepts_categorical_strings", False)
            name = getattr(self, "name", self.ff_class)
            observed = df["y"].notna()
            future = df[~observed]
            df = df[observed].sort_values(["unique_id", "ds"]).reset_index(drop=True)
            if future.empty:
                _warn_no_future_covariates(name)
                df = df[["unique_id", "ds", "y"]]
            else:
                check_covariates(df, covariates, accepts_strings, name)
                X_df = build_horizon(
                    df, future, self.prediction_length, self.freq, covariates, accepts_strings, name
                )
        df["y"] = df["y"].astype(np.float64)
        series = forecast_series(self._model, df, self.prediction_length, self.freq, X_df=X_df)
        return pd.DataFrame({
            "unique_id": np.repeat(list(series), self.prediction_length),
            "ds": np.concatenate([dates for dates, _, _ in series.values()]),
            "y": np.concatenate([values for _, values, _ in series.values()]),
        })


class FFChronosForecaster(FoundationForecastForecaster):
    ff_class = "Chronos"
    hparams = ("batch_size", "dtype")
    supports_future_covariates = True
    accepts_categorical_strings = True
    # The quantiles of the chronos-2 and chronos-2-small checkpoints (chronos_config.quantiles).
    native_quantile_range = (0.01, 0.99)


class FFTimesFMForecaster(FoundationForecastForecaster):
    ff_class = "TimesFM"
    hparams = ("context_length", "batch_size")
    native_quantile_range = (0.1, 0.9)


class FFTiRexForecaster(FoundationForecastForecaster):
    ff_class = "TiRex"
    hparams = ("batch_size",)
    supports_future_covariates = True
    native_quantile_range = (0.1, 0.9)


class FFTotoForecaster(FoundationForecastForecaster):
    ff_class = "Toto"
    hparams = ("context_length", "batch_size")


class FFToto2Forecaster(FFTotoForecaster):
    hparams = FFTotoForecaster.hparams + ("decode_block_size",)
    # Toto 2.0 predicts the knots 0.1 to 0.9. Toto 1.0 (FFToto) takes sample quantiles at any level.
    native_quantile_range = (0.1, 0.9)


class FFFlowStateForecaster(FoundationForecastForecaster):
    ff_class = "FlowState"
    hparams = ("context_length", "batch_size", "scale_factor")
    # The quantiles of the flowstate checkpoint (config.quantiles).
    native_quantile_range = (0.1, 0.9)


class FFPatchTSTFMForecaster(FoundationForecastForecaster):
    ff_class = "PatchTSTFM"
    hparams = ("context_length", "batch_size")
    native_quantile_range = (0.01, 0.99)


class FFTafsutForecaster(FoundationForecastForecaster):
    ff_class = "Tafsut"
    hparams = ("context_length", "batch_size")
    native_quantile_range = (0.1, 0.9)


class FFT0Forecaster(FoundationForecastForecaster):
    ff_class = "T0"
    hparams = (
        "context_length",
        "batch_size",
        "trim_leading_missing",
        "symmetric_averaging",
        "clip_nonnegative",
        "snap_integers",
        "cross_inference",
    )
    supports_future_covariates = True
    # t0-beta and t0-gamma predict 0.01 to 0.99; t0-alpha (0.1 to 0.9) is not exposed by MMF.
    native_quantile_range = (0.01, 0.99)


class FFTimestralForecaster(FoundationForecastForecaster):
    ff_class = "Timestral"
    hparams = ("context_length", "batch_size", "symmetric_averaging", "clip_nonnegative")
    supports_future_covariates = True


class FFChronos2(FFChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "amazon/chronos-2"


class FFChronos2Small(FFChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "autogluon/chronos-2-small"


class FFTimesFM_2_5_200m(FFTimesFMForecaster):
    # TimesFM 2.5 defaults to one series per forward pass unless per_core_batch_size is set.
    # TimesFM 3.0 derives it from batch_size and rejects it as an extra argument.
    hparams = FFTimesFMForecaster.hparams + ("per_core_batch_size",)

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "google/timesfm-2.5-200m-pytorch"


class FFTimesFM_3_0(FFTimesFMForecaster):
    supports_future_covariates = True

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "google/timesfm-3.0-pytorch"


class FFTiRex2(FFTiRexForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "NX-AI/TiRex-2"


class FFToto(FFTotoForecaster):
    hparams = FFTotoForecaster.hparams + ("num_samples", "samples_per_batch")
    samples_hparam = "num_samples"

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Datadog/Toto-Open-Base-1.0"


class FFToto2_22m(FFToto2Forecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Datadog/Toto-2.0-22m"


class FFToto2_313m(FFToto2Forecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Datadog/Toto-2.0-313m"


class FFToto2_1B(FFToto2Forecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Datadog/Toto-2.0-1B"


class FFFlowState(FFFlowStateForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "ibm-research/flowstate"


class FFPatchTSTFM_R2(FFPatchTSTFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "ibm-granite/granite-timeseries-patchtst-fm-r2"


class FFPatchTSTFM_R1(FFPatchTSTFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "ibm-research/patchtst-fm-r1"


class FFTafsutBase(FFTafsutForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Tafsut-FM/tafsut-univariate-base"


class FFT0Beta(FFT0Forecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "theforecastingcompany/t0-beta"


class FFT0Gamma(FFT0Forecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "theforecastingcompany/t0-gamma"


class FFTimestral_5M(FFTimestralForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "mistralai/Timestral-5M"


class FFTimestral_28M(FFTimestralForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "mistralai/Timestral-28M"


class FFTimestral_123M(FFTimestralForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "mistralai/Timestral-123M"


class FFTimestral_209M(FFTimestralForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "mistralai/Timestral-209M"
