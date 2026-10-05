import gc
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)

from mmf_sa.models.abstract_model import ForecastingRegressor, MODEL_PIP_REQUIREMENTS
from mmf_sa.models.foundationforecast._runtime import ensure_runtime
from mmf_sa.exceptions import (
    DataPreparationError,
    ModelPredictionError,
    UnsupportedMetricError,
)

_logger = logging.getLogger(__name__)

# MMF frequency codes -> pandas aliases accepted by foundationforecast.
FF_FREQ = {"H": "h", "D": "D", "W": "W", "M": "ME"}

NON_COMMERCIAL_LICENSES = frozenset({
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "timesfm-non-commercial-license-v1.0",
})

COVARIATE_PARAMS = (
    "static_features",
    "dynamic_future_numerical",
    "dynamic_future_categorical",
    "dynamic_historical_numerical",
    "dynamic_historical_categorical",
)

_FF_ALIAS = "ff_forecast"
_warned = set()
_current_checkpoint = None


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        _logger.warning(message)


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


def forecast_series(
    ff_model, df: pd.DataFrame, prediction_length: int, freq: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Forecast a long ``unique_id, ds, y`` frame.

    Returns ``{unique_id: (dates, values)}``. Dates are rebuilt from each series'
    last observation with MMF's step offset, because pandas' "W" alias anchors
    foundationforecast's future dates to Sundays.
    """
    fcst = ff_model.forecast(df, h=prediction_length, freq=FF_FREQ[freq])
    fcst = fcst.sort_values(["unique_id", "ds"], kind="stable")
    values = {
        uid: grp[_FF_ALIAS].to_numpy(dtype=np.float64)
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
        dates = np.array(
            [(last + offset * step).to_datetime64() for step in range(1, prediction_length + 1)],
            dtype="datetime64[ns]",
        )
        out[uid] = (dates, forecast)
    return out


class FoundationForecastForecaster(ForecastingRegressor):
    """Base class for models served through the foundationforecast library.

    Family subclasses set ``ff_class`` (the foundationforecast model class) and
    ``hparams`` (the YAML keys forwarded to its constructor). Checkpoint leaves
    set ``self.repo``. Prediction runs on the driver's GPU only.
    """

    ff_class: str = None
    hparams: Tuple[str, ...] = ()

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = None
        self.model = None
        self._warn_unsupported_config()

    def _warn_unsupported_config(self):
        name = self.params.get("name", type(self).__name__)
        configured = [k for k in COVARIATE_PARAMS if self.params.get(k)]
        if configured:
            _warn_once(
                f"covariates:{name}",
                f"{name}: foundationforecast models are univariate; ignoring "
                f"{', '.join(configured)}.",
            )
        license_id = self.params.get("license")
        if license_id in NON_COMMERCIAL_LICENSES:
            _warn_once(
                f"license:{name}",
                f"{name} is released under the non-commercial license '{license_id}'. "
                "Check the license terms before using it.",
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

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> pd.DataFrame:
        """Rename MMF columns to foundationforecast's ``unique_id, ds, y`` and validate."""
        group_col = self.params.group_id
        date_col = self.params.date_col
        target_col = self.params.target
        pdf = df[[group_col, date_col, target_col]].rename(
            columns={group_col: "unique_id", date_col: "ds", target_col: "y"}
        )
        pdf["ds"] = pd.to_datetime(pdf["ds"])
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
        pdf["y"] = pdf["y"].astype(np.float64)
        return pdf.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None, curr_date=None, spark=None):
        # Distributed prediction is not implemented; every call uses the driver GPU.
        return self._predict_single(hist_df)

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

    def _predict_single(self, hist_df: pd.DataFrame):
        """Driver-only single-GPU prediction path."""
        pdf = self.prepare_data(hist_df)
        original_ids = {}
        for uid in pdf["unique_id"].unique():
            original_ids.setdefault(str(uid), uid)
        if len(original_ids) != pdf["unique_id"].nunique():
            raise DataPreparationError(
                f"Values of '{self.params.group_id}' are not unique once converted to strings."
            )
        pdf["unique_id"] = pdf["unique_id"].astype(str)

        series = forecast_series(
            self._get_ff_model(), pdf, self.params["prediction_length"], self.freq
        )
        forecast_df = pd.DataFrame({
            self.params.group_id: [original_ids[uid] for uid in series],
            self.params.date_col: [dates for dates, _ in series.values()],
            self.params.target: [values for _, values in series.values()],
        })
        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame, spark=None):
        hist_df = df[df[self.params.target].notnull()]
        return self.predict(hist_df, spark=spark)

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> list:
        pred_df, _ = self.predict(hist_df, val_df, curr_date, spark)
        metric_name = self.params["metric"]
        metric_classes = {
            "smape": MeanAbsolutePercentageError(symmetric=True),
            "mape": MeanAbsolutePercentageError(symmetric=False),
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(square_root=False),
            "rmse": MeanSquaredError(square_root=True),
        }
        if metric_name not in metric_classes:
            raise UnsupportedMetricError(f"Metric {metric_name} not supported!")
        metric_function = metric_classes[metric_name]

        group_col = self.params["group_id"]
        target_col = self.params["target"]
        actuals_map = {
            k: np.asarray(v) for k, v in val_df.groupby(group_col, sort=False)[target_col]
        }
        forecasts_map = dict(zip(pred_df[group_col].to_numpy(), pred_df[target_col].to_numpy()))

        metrics = []
        for key, forecast in forecasts_map.items():
            try:
                actual = actuals_map[key]
                forecast = np.asarray(forecast)
                metric_value = metric_function(actual, forecast)
                metrics.append(
                    (key, curr_date, metric_name, metric_value, forecast, actual, b"")
                )
            except Exception as err:
                _logger.warning(f"Failed to calculate metric for key {key}: {err}")
        return metrics

    def register(self, registered_model_name: str):
        model = FoundationForecastMLflowModel(
            ff_class=self.ff_class,
            repo=self.repo,
            ff_kwargs=self.ff_kwargs(),
            prediction_length=int(self.params["prediction_length"]),
            freq=self.freq,
        )
        input_schema = Schema([
            ColSpec("string", "unique_id"),
            ColSpec("datetime", "ds"),
            ColSpec("double", "y"),
        ])
        signature = ModelSignature(inputs=input_schema, outputs=input_schema)
        dates = pd.date_range(end="2024-12-31", periods=52, freq=FF_FREQ[self.freq])
        input_example = pd.DataFrame({
            "unique_id": "series_0",
            "ds": dates,
            "y": np.random.rand(len(dates)),
        })
        mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=MODEL_PIP_REQUIREMENTS["foundationforecast"],
        )


class FoundationForecastMLflowModel(mlflow.pyfunc.PythonModel):
    """Takes a long ``unique_id, ds, y`` frame and returns the forecast in the same layout."""

    def __init__(self, ff_class: str, repo: str, ff_kwargs: dict, prediction_length: int, freq: str):
        self.ff_class = ff_class
        self.repo = repo
        self.ff_kwargs = dict(ff_kwargs)
        self.prediction_length = prediction_length
        self.freq = freq
        self._model = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if self._model is None:
            self._model = build_ff_model(self.ff_class, self.repo, self.ff_kwargs)
        df = model_input[["unique_id", "ds", "y"]].copy()
        df["unique_id"] = df["unique_id"].astype(str)
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = df["y"].astype(np.float64)
        series = forecast_series(self._model, df, self.prediction_length, self.freq)
        return pd.DataFrame({
            "unique_id": np.repeat(list(series), self.prediction_length),
            "ds": np.concatenate([dates for dates, _ in series.values()]),
            "y": np.concatenate([values for _, values in series.values()]),
        })


class FFChronosForecaster(FoundationForecastForecaster):
    ff_class = "Chronos"
    hparams = ("batch_size", "dtype")


class FFTimesFMForecaster(FoundationForecastForecaster):
    ff_class = "TimesFM"
    hparams = ("context_length", "batch_size")


class FFTiRexForecaster(FoundationForecastForecaster):
    ff_class = "TiRex"
    hparams = ("batch_size",)


class FFTotoForecaster(FoundationForecastForecaster):
    ff_class = "Toto"
    hparams = ("context_length", "batch_size")


class FFToto2Forecaster(FFTotoForecaster):
    hparams = FFTotoForecaster.hparams + ("decode_block_size",)


class FFFlowStateForecaster(FoundationForecastForecaster):
    ff_class = "FlowState"
    hparams = ("context_length", "batch_size", "scale_factor")


class FFPatchTSTFMForecaster(FoundationForecastForecaster):
    ff_class = "PatchTSTFM"
    hparams = ("context_length", "batch_size")


class FFTafsutForecaster(FoundationForecastForecaster):
    ff_class = "Tafsut"
    hparams = ("context_length", "batch_size")


class FFT0Forecaster(FoundationForecastForecaster):
    ff_class = "T0"
    hparams = ("context_length", "batch_size")


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
