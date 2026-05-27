"""MLForecast-based global model integrations for MMF.

Provides two LightGBM models that share the same adapter base class:

- ``MLForecastLGBM``     — fixed hyperparameters (no HPO).
- ``MLForecastAutoLGBM`` — hyperparameter optimization plus optional feature
  and fit-config tuning, driven by Optuna under MLForecast's ``AutoMLForecast``.

Both models plug into MMF's existing global-model contract via
``ForecastingRegressor`` and are configured entirely from the YAML files
(``models_conf.yaml`` and ``forecasting_conf_*.yaml``) exactly the way the
existing ``NeuralForecastPipeline`` models are. No edits to the core
``Forecaster`` or ``ModelRegistry`` are required.

The implementation is designed to run on a single-node CPU Databricks cluster.
Multi-node CPU Spark scale-out via ``mlforecast.distributed.DistributedMLForecast``
is documented in the implementation plan and gated by future YAML flags.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cloudpickle
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import ColSpec, Schema
from mlforecast import MLForecast
from mlforecast.auto import AutoMLForecast, AutoModel
from mlforecast.lag_transforms import (
    ExponentiallyWeightedMean,
    RollingMean,
    RollingStd,
)
from mlforecast.target_transforms import Differences, LocalStandardScaler
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)

from mmf_sa.exceptions import (
    ConfigurationError,
    DataPreparationError,
    MissingFeatureError,
    ModelPredictionError,
    UnsupportedMetricError,
)
from mmf_sa.models.abstract_model import MODEL_PIP_REQUIREMENTS, ForecastingRegressor

_logger = logging.getLogger(__name__)


# -- Resolvers: YAML strings -> MLForecast callables --------------------------

# Recognized identifiers for ``target_transforms`` entries in YAML.
# A user can write things like ``"differences_1"`` or ``"differences_1_7"``
# (a chain of differences applied in order).
_TARGET_TRANSFORM_PATTERN = re.compile(r"^differences_(?P<lags>\d+(?:_\d+)*)$")

# Recognized identifiers for ``lag_transforms`` entries. Examples:
#   ``rolling_mean_7``, ``rolling_std_28``, ``ewm_alpha_0.3``
_ROLLING_MEAN_PATTERN = re.compile(r"^rolling_mean_(?P<window>\d+)$")
_ROLLING_STD_PATTERN = re.compile(r"^rolling_std_(?P<window>\d+)$")
_EWM_PATTERN = re.compile(r"^ewm_alpha_(?P<alpha>0?\.\d+)$")

_VALID_DATE_FEATURES = {
    "year", "quarter", "month", "day",
    "dayofweek", "dayofyear", "week", "hour",
}


def _resolve_target_transforms(spec: Optional[Iterable[str]]):
    """Translate a list of YAML strings into MLForecast target-transform objects.

    ``"none"`` (case-insensitive) is the no-op sentinel — callers should map it
    to ``None`` upstream. Returns ``None`` when ``spec`` itself is empty/None
    so MLForecast skips target transformation entirely.
    """
    if not spec:
        return None
    out = []
    for item in spec:
        if item is None:
            continue
        s = str(item).strip()
        if s.lower() == "none":
            continue
        m = _TARGET_TRANSFORM_PATTERN.match(s)
        if m:
            lags = [int(x) for x in m.group("lags").split("_")]
            out.append(Differences(lags))
            continue
        if s.lower() == "local_standard_scaler":
            out.append(LocalStandardScaler())
            continue
        raise ConfigurationError(
            f"Unrecognized target_transform identifier '{s}'. "
            f"Supported: 'none', 'differences_<lag>[_<lag>...]', 'local_standard_scaler'."
        )
    return out or None


def _resolve_one_lag_transform(s: str):
    """Map a single lag-transform string to a MLForecast lag-transform callable."""
    s = str(s).strip()
    m = _ROLLING_MEAN_PATTERN.match(s)
    if m:
        return RollingMean(window_size=int(m.group("window")), min_samples=1)
    m = _ROLLING_STD_PATTERN.match(s)
    if m:
        return RollingStd(window_size=int(m.group("window")), min_samples=2)
    m = _EWM_PATTERN.match(s)
    if m:
        return ExponentiallyWeightedMean(alpha=float(m.group("alpha")))
    raise ConfigurationError(
        f"Unrecognized lag_transform identifier '{s}'. "
        f"Supported: 'none', 'rolling_mean_<window>', 'rolling_std_<window>', 'ewm_alpha_<float>'."
    )


def _resolve_lag_transforms(spec):
    """Translate ``lag_transforms`` from YAML into the dict MLForecast expects.

    Two YAML shapes are supported:

    1. Dict-of-lists keyed by lag (used by the fixed-hyperparam model)::

           lag_transforms:
             1: ["rolling_mean_7", "rolling_mean_28"]
             7: ["rolling_mean_28"]

    2. A single string (used inside an HPO ``init_config`` after a categorical
       choice has been drawn, e.g. ``"rolling_mean_7"`` or ``"none"``). In that
       case we attach the transform to lag 1 by convention.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec.strip().lower() == "none":
            return None
        return {1: [_resolve_one_lag_transform(spec)]}
    # Use ``Mapping`` (not ``dict``) so OmegaConf's DictConfig is accepted.
    if isinstance(spec, Mapping):
        out: Dict[int, List[Any]] = {}
        for lag, transforms in spec.items():
            resolved = []
            for t in transforms or []:
                if str(t).strip().lower() == "none":
                    continue
                resolved.append(_resolve_one_lag_transform(t))
            if resolved:
                out[int(lag)] = resolved
        return out or None
    raise ConfigurationError(
        f"lag_transforms must be a Mapping or string; got {type(spec).__name__}."
    )


def _resolve_date_features(spec: Optional[Iterable[str]]) -> Optional[List[str]]:
    """Validate date-feature names. MLForecast accepts the strings directly."""
    if not spec:
        return None
    out = []
    for item in spec:
        s = str(item).strip()
        if s.lower() == "none":
            continue
        if s not in _VALID_DATE_FEATURES:
            raise ConfigurationError(
                f"Unrecognized date_feature '{s}'. "
                f"Supported: {sorted(_VALID_DATE_FEATURES)}."
            )
        out.append(s)
    return out or None


def _to_python_list(maybe_listing) -> List[Any]:
    """Best-effort conversion of an OmegaConf ListConfig (or list/tuple) to list."""
    if maybe_listing is None:
        return []
    if isinstance(maybe_listing, (list, tuple)):
        return list(maybe_listing)
    if isinstance(maybe_listing, Mapping):
        # Defensive: never silently coerce a Mapping into ``list(keys)``.
        raise ConfigurationError(
            f"Expected a list-like value, got Mapping: {dict(maybe_listing)!r}"
        )
    try:
        return list(maybe_listing)
    except TypeError:
        return [maybe_listing]


# Map MMF's freq codes to pandas-2.x-compatible offset aliases used by
# MLForecast / utilsforecast. MMF passes through legacy single-letter codes
# (``"M"``, ``"H"``) but pandas 2.x deprecated some of them.
_FREQ_MAP = {"M": "ME", "H": "h"}


def _mlforecast_freq(freq: str) -> str:
    return _FREQ_MAP.get(freq, freq)


# -- Base adapter -------------------------------------------------------------


class MLForecastBase(ForecastingRegressor):
    """Shared adapter logic for all MLForecast-based models in MMF.

    Subclasses are responsible for setting ``self.model`` (an ``MLForecast`` or
    ``AutoMLForecast`` instance) in their ``__init__``. They may override
    ``fit`` to dispatch to the right ``self.model.fit`` signature.
    """

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model = None  # set by subclass

    # -- Data shaping --

    def _exog_columns(self) -> List[str]:
        """Return all exogenous columns declared on the run-level config."""
        cols: List[str] = []
        for key in (
            "dynamic_future_numerical",
            "dynamic_future_categorical",
            "dynamic_historical_numerical",
            "dynamic_historical_categorical",
        ):
            try:
                cols += list(self.params.get(key, []) or [])
            except Exception as exc:  # pragma: no cover — defensive
                raise MissingFeatureError(f"Failed to read {key}: {exc}")
        return cols

    def prepare_data(self, df: pd.DataFrame, future: bool = False) -> pd.DataFrame:
        """Rename to MLForecast's ``unique_id`` / ``ds`` / ``y`` schema.

        When ``future=True`` we drop the target column because the future frame
        is an exogenous-feature table (``X_df``) for prediction.

        Static feature columns are included in the historical (training) frame
        so MLForecast can mark them as static via the ``static_features``
        argument of ``MLForecast.fit``.
        """
        if not future:
            df = df.copy()
            df[self.params["target"]] = df[self.params["target"]].clip(0)
            features = [self.params["group_id"], self.params["date_col"], self.params["target"]]
            features += self._exog_columns()
            statics = list(self.params.get("static_features", []) or [])
            features += [c for c in statics if c not in features]
            _df = df[features].rename(
                columns={
                    self.params["group_id"]: "unique_id",
                    self.params["date_col"]: "ds",
                    self.params["target"]: "y",
                }
            )
        else:
            features = [self.params["group_id"], self.params["date_col"]]
            future_only = (
                list(self.params.get("dynamic_future_numerical", []) or [])
                + list(self.params.get("dynamic_future_categorical", []) or [])
            )
            features += future_only
            _df = df[features].rename(
                columns={
                    self.params["group_id"]: "unique_id",
                    self.params["date_col"]: "ds",
                }
            )
        _df["ds"] = pd.to_datetime(_df["ds"])
        return _df

    def _has_future_covariates(self) -> bool:
        return bool(
            self.params.get("dynamic_future_numerical")
            or self.params.get("dynamic_future_categorical")
        )

    def _static_features_kwarg(self) -> List[str]:
        """Map MMF's ``static_features`` config to MLForecast's static columns.

        Returns the explicit static feature column names (excluding
        ``unique_id``). The unique identifier stays out of this list — passing
        it would cause MLForecast to feed the string id into LightGBM as a
        feature, which fails with ``bad pandas dtypes: unique_id: object``.

        Always returns a list (possibly empty). Callers should pass the result
        directly to MLForecast's ``static_features=`` kwarg. Returning an empty
        list (rather than ``None``) is important because MLForecast's
        ``static_features=None`` default would otherwise mark **all** non-id,
        non-ds, non-y columns as static — incorrectly pinning the time-varying
        ``future_*`` covariates.
        """
        return list(self.params.get("static_features", []) or [])

    # -- Predict / forecast --

    def _model_predict(
        self,
        h: int,
        X_df: Optional[pd.DataFrame] = None,
        new_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Unified prediction call for both ``MLForecast`` and
        ``AutoMLForecast``.

        ``new_df`` re-aligns each series' future window to that series' last
        observed date — required for MMF's rolling-origin backtest where
        ``hist_df`` advances per iteration. ``AutoMLForecast.predict`` does
        not accept ``new_df``, so we reach through to its fitted inner
        ``MLForecast`` instance (``self.model.models_['lgb']``) when needed.
        """
        kwargs: Dict[str, Any] = {"h": h}
        if X_df is not None:
            kwargs["X_df"] = X_df

        if isinstance(self.model, AutoMLForecast):
            if new_df is not None:
                kwargs["new_df"] = new_df
                fitted = getattr(self.model, "models_", None) or {}
                if not fitted:
                    raise ModelPredictionError(
                        "AutoMLForecast has no fitted models; call fit() first."
                    )
                # Single model 'lgb' is registered; predict via the inner MLForecast.
                inner = next(iter(fitted.values()))
                return inner.predict(**kwargs)
            return self.model.predict(**kwargs)

        if new_df is not None:
            kwargs["new_df"] = new_df
        return self.model.predict(**kwargs)

    def _build_x_df(
        self,
        new_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        h: int,
    ) -> Optional[pd.DataFrame]:
        """Build a complete ``X_df`` for MLForecast.predict.

        Always returns an X_df when **any** exogenous features are configured
        (dynamic future covariates OR static features) — both must appear as
        columns in ``X_df`` for ``AutoMLForecast``'s inner predict path, which
        does not pull statics from internal state the way plain ``MLForecast``
        does.

        Generates one row per ``(unique_id, future_ds)`` based on each series'
        last date in ``new_df``, left-merges future covariates from
        ``val_df``, and replicates static features from ``new_df``. Guarantees
        MLForecast's per-id row-count check passes even when the source
        ``val_df`` is ragged (missing months for some series).
        """
        has_future = self._has_future_covariates()
        statics = list(self.params.get("static_features", []) or [])
        statics = [c for c in statics if c in new_df.columns]
        if not (has_future or statics):
            return None

        # Build the per-series future skeleton from ``new_df``'s last dates.
        freq = _mlforecast_freq(self.params["freq"])
        last_dates = new_df.groupby("unique_id", sort=False)["ds"].max()
        skeletons = []
        for uid, last_ds in last_dates.items():
            future_dates = pd.date_range(
                start=pd.Timestamp(last_ds), periods=h + 1, freq=freq
            )[1:]
            skeletons.append(pd.DataFrame({"unique_id": uid, "ds": future_dates}))
        skeleton = pd.concat(skeletons, ignore_index=True)

        # Left-merge dynamic future covariates from val_df when available.
        cov_cols: List[str] = []
        if has_future and val_df is not None and not val_df.empty:
            futr_pdf = self.prepare_data(val_df, future=True)
            if not futr_pdf.empty:
                cov_cols = [c for c in futr_pdf.columns if c not in ("unique_id", "ds")]
                skeleton = skeleton.merge(
                    futr_pdf[["unique_id", "ds"] + cov_cols],
                    on=["unique_id", "ds"],
                    how="left",
                )

        # Replicate static features (constant per unique_id) from new_df.
        if statics:
            statics_df = (
                new_df.groupby("unique_id", sort=False)[statics]
                .last()
                .reset_index()
            )
            skeleton = skeleton.merge(statics_df, on="unique_id", how="left")
            cov_cols += statics

        if not cov_cols:
            return None

        # Forward/back-fill ragged covariates within each series, then zero-fill
        # any boundary NaNs (safe for LightGBM, which tolerates NaN but cleaner
        # results when feature engineering pipeline runs on densely-filled data).
        skeleton = (
            skeleton.sort_values(["unique_id", "ds"])
            .groupby("unique_id", group_keys=False, sort=False)
            .apply(lambda g: g.ffill().bfill())
            .reset_index(drop=True)
        )
        skeleton[cov_cols] = skeleton[cov_cols].fillna(0)
        return skeleton

    def _rename_predictions(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Map MLForecast's wide-format prediction back to MMF's schema.

        MLForecast returns columns like ``unique_id, ds, lgb`` (one column per
        registered model). We always have a single model named ``lgb``, so we
        rename it to the configured target column.
        """
        # Pick the single non-id column as the prediction column.
        pred_cols = [c for c in forecast_df.columns if c not in {"unique_id", "ds"}]
        if not pred_cols:
            raise ModelPredictionError(
                f"MLForecast returned no prediction column. Got: {forecast_df.columns.tolist()}"
            )
        # Use the first prediction column even if multiple exist.
        pred_col = pred_cols[0]
        out = forecast_df.reset_index(drop=True).rename(
            columns={
                "unique_id": self.params["group_id"],
                "ds": self.params["date_col"],
                pred_col: self.params["target"],
            }
        )
        # Drop any extra prediction columns to keep MMF's schema clean.
        for extra in pred_cols[1:]:
            if extra in out.columns:
                out = out.drop(columns=[extra])
        out[self.params["target"]] = out[self.params["target"]].clip(0)
        return out

    def predict(
        self,
        hist_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Any]:
        """Backtest-time prediction. Called by the base ``backtest()`` loop.

        ``hist_df`` advances each iteration so we pass it as MLForecast's
        ``new_df`` to re-align the per-series prediction window. ``X_df`` is
        constructed via ``_build_x_df`` to guarantee row coverage.
        """
        h = int(self.params["prediction_length"])
        new_df = self.prepare_data(hist_df)
        X_df = self._build_x_df(new_df=new_df, val_df=val_df, h=h)
        forecast_df = self._model_predict(h=h, X_df=X_df, new_df=new_df)
        return self._rename_predictions(forecast_df), self.model

    def forecast(self, df: pd.DataFrame, spark=None) -> Tuple[pd.DataFrame, Any]:
        """Scoring-time prediction.

        ``df`` contains both the history (target populated) and a future window
        (target null). The future rows supply exogenous covariates for ``X_df``.
        """
        hist_df = df[df[self.params["target"]].notnull()]
        hist_pdf = self.prepare_data(hist_df)
        last_date = hist_pdf["ds"].max()
        future_df = df[
            (df[self.params["date_col"]] > np.datetime64(last_date))
            & (
                df[self.params["date_col"]]
                <= np.datetime64(last_date + self.prediction_length_offset)
            )
        ]
        h = int(self.params["prediction_length"])
        X_df = self._build_x_df(new_df=hist_pdf, val_df=future_df, h=h)
        forecast_df = self._model_predict(h=h, X_df=X_df, new_df=hist_pdf)
        return self._rename_predictions(forecast_df), self.model

    # -- Custom backtest metrics (per-key, long-format) --

    def calculate_metrics(
        self,
        hist_df: pd.DataFrame,
        val_df: pd.DataFrame,
        curr_date,
        spark=None,
    ) -> list:
        """Per-key metric calculation, shaped like ``NeuralFcForecaster``."""
        pred_df, _ = self.predict(hist_df, val_df)
        keys = pred_df[self.params["group_id"]].unique()
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
        prediction_length = int(self.params["prediction_length"])
        actuals_map = {
            k: v.to_numpy()
            for k, v in val_df.groupby(group_col, sort=False)[target_col]
        }
        forecasts_map = {
            k: v.iloc[-prediction_length:].to_numpy()
            for k, v in pred_df.groupby(group_col, sort=False)[target_col]
        }

        metrics = []
        for key in keys:
            try:
                actual = actuals_map[key]
                forecast = forecasts_map[key]
                metric_value = metric_function(actual, forecast)
                metrics.append(
                    (
                        key,
                        curr_date,
                        metric_name,
                        metric_value,
                        forecast,
                        actual,
                        b"",
                    )
                )
            except (ModelPredictionError, DataPreparationError) as err:
                _logger.warning(f"Failed to calculate metric for key {key}: {err}")
            except Exception as err:  # pragma: no cover — defensive
                _logger.warning(f"Unexpected error calculating metric for key {key}: {err}")
        return metrics

    # -- MLflow registration --

    def register(self, model, registered_model_name: str, input_example):
        """Log this pipeline as an MLflow pyfunc and register it to UC."""
        pipeline = MLForecastModel(model)
        input_schema = infer_signature(model_input=input_example).inputs
        output_schema = Schema(
            [
                ColSpec("string", self.params["group_id"]),
                ColSpec("datetime", self.params["date_col"]),
                ColSpec("float", self.params["target"]),
            ]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=pipeline,
            registered_model_name=registered_model_name,
            signature=signature,
            pip_requirements=MODEL_PIP_REQUIREMENTS["mlforecast"],
        )
        mlflow.log_params(self._loggable_params())
        print(f"Model registered: {registered_model_name}")
        return model_info

    def _loggable_params(self) -> Dict[str, Any]:
        """Flatten the OmegaConf params into something MLflow can log."""
        flat: Dict[str, Any] = {}
        for k, v in dict(self.params).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                flat[k] = v
        return flat


# -- Concrete fixed-hyperparam model -----------------------------------------


class MLForecastLGBM(MLForecastBase):
    """LightGBM forecaster with fixed hyperparameters and a fixed feature
    pipeline. Driven entirely by the YAML keys ``model_params``, ``features``,
    and ``fit_params`` under the model entry."""

    def __init__(self, params):
        super().__init__(params)

        model_params = dict(self.params.get("model_params", {}) or {})
        model_params.setdefault("verbosity", -1)
        model_params.setdefault("n_jobs", int(self.params.get("num_threads", -1)))

        feats = self.params.get("features", {}) or {}
        mlf_kwargs = self._build_init_kwargs(feats)

        self.model = MLForecast(
            models={"lgb": lgb.LGBMRegressor(**model_params)},
            freq=_mlforecast_freq(self.params["freq"]),
            num_threads=int(self.params.get("num_threads", -1)),
            **mlf_kwargs,
        )

        self._fit_kwargs = self._build_fit_kwargs(self.params.get("fit_params", {}) or {})

    # -- Helpers --

    def _build_init_kwargs(self, feats) -> Dict[str, Any]:
        """Resolve ``features`` into kwargs for ``MLForecast.__init__``."""
        kwargs: Dict[str, Any] = {}
        lags = _to_python_list(feats.get("lags"))
        if lags:
            kwargs["lags"] = [int(x) for x in lags]
        date_feats = _resolve_date_features(_to_python_list(feats.get("date_features")))
        if date_feats:
            kwargs["date_features"] = date_feats
        target_t = _resolve_target_transforms(_to_python_list(feats.get("target_transforms")))
        if target_t:
            kwargs["target_transforms"] = target_t
        lag_t = _resolve_lag_transforms(feats.get("lag_transforms"))
        if lag_t:
            kwargs["lag_transforms"] = lag_t
        return kwargs

    def _build_fit_kwargs(self, fit_params) -> Dict[str, Any]:
        """Resolve ``fit_params`` into kwargs for ``MLForecast.fit``.

        Always passes ``static_features=[]`` so MLForecast treats every
        configured exogenous column (both ``static_features`` and
        ``dynamic_*``) uniformly as time-varying features. This avoids an
        asymmetry between plain ``MLForecast.predict`` (which rejects static
        columns in ``X_df``) and ``AutoMLForecast``'s inner predict path
        (which requires them in ``X_df``). LightGBM's tree splits do not
        depend on the static/dynamic label, so functionally this is
        equivalent — the ``static_features`` YAML key still controls *which*
        columns are exposed to the model via ``prepare_data``.
        """
        kwargs: Dict[str, Any] = {"static_features": []}
        if "dropna" in fit_params:
            kwargs["dropna"] = bool(fit_params["dropna"])
        return kwargs

    def fit(self, x, y=None):
        pdf = self.prepare_data(x)
        self.model.fit(df=pdf, **self._fit_kwargs)


# -- Concrete HPO model -------------------------------------------------------


class MLForecastAutoLGBM(MLForecastBase):
    """LightGBM forecaster with Optuna-driven hyperparameter and feature
    tuning, via MLForecast's ``AutoMLForecast`` + ``AutoModel``.

    Driven entirely by the YAML keys ``model_hp_space``, ``feature_space``, and
    ``fit_space`` under the model entry. Each ``*_space`` value is a list of
    candidate choices (or a 2-element ``[low, high]`` range for continuous
    hyperparameters under ``model_hp_space``).
    """

    def __init__(self, params):
        super().__init__(params)

        # Suppress Optuna's per-trial INFO chatter; keep WARN+.
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except Exception:  # pragma: no cover
            pass

        hp = self.params.get("model_hp_space", {}) or {}
        model_config_fn = self._build_model_config_fn(hp)

        fs = self.params.get("feature_space", {}) or {}
        init_config_fn = self._build_init_config_fn(fs)

        fts = self.params.get("fit_space", {}) or {}
        fit_config_fn = self._build_fit_config_fn(fts)

        kwargs: Dict[str, Any] = dict(
            models={"lgb": AutoModel(model=lgb.LGBMRegressor(), config=model_config_fn)},
            freq=_mlforecast_freq(self.params["freq"]),
            num_threads=int(self.params.get("num_threads", -1)),
            reuse_cv_splits=bool(self.params.get("reuse_cv_splits", True)),
        )
        if init_config_fn is not None:
            kwargs["init_config"] = init_config_fn
        else:
            kwargs["season_length"] = int(self.params.get("season_length", 1))
        if fit_config_fn is not None:
            kwargs["fit_config"] = fit_config_fn

        self.model = AutoMLForecast(**kwargs)

    # -- Helpers --

    @staticmethod
    def _build_model_config_fn(hp) -> Callable[[optuna.Trial], Dict[str, Any]]:
        """Return an Optuna config callable for the LightGBM hyperparameter space."""
        def _range(name, lo_hi, kind="float", log=False):
            lo, hi = _to_python_list(lo_hi)
            return name, kind, lo, hi, log

        spec = []
        if "learning_rate" in hp:
            spec.append(_range("learning_rate", hp.learning_rate, "float", log=True))
        if "num_leaves" in hp:
            spec.append(_range("num_leaves", hp.num_leaves, "int"))
        if "n_estimators" in hp:
            spec.append(_range("n_estimators", hp.n_estimators, "int"))
        if "feature_fraction" in hp:
            spec.append(_range("feature_fraction", hp.feature_fraction, "float"))
        if "bagging_fraction" in hp:
            spec.append(_range("bagging_fraction", hp.bagging_fraction, "float"))
        if "min_child_samples" in hp:
            spec.append(_range("min_child_samples", hp.min_child_samples, "int"))

        n_jobs_default = -1

        def model_config(trial: optuna.Trial) -> Dict[str, Any]:
            cfg: Dict[str, Any] = {"verbosity": -1, "n_jobs": n_jobs_default}
            for name, kind, lo, hi, log in spec:
                if kind == "int":
                    cfg[name] = trial.suggest_int(name, int(lo), int(hi))
                else:
                    cfg[name] = trial.suggest_float(name, float(lo), float(hi), log=log)
            return cfg

        return model_config

    def _build_init_config_fn(self, fs):
        """Return an Optuna ``init_config`` callable for feature tuning, or None.

        When ``feature_space`` is empty/missing we return ``None`` and let
        ``AutoMLForecast`` use its default ``season_length``-based features.
        """
        if not fs:
            return None

        # Pre-compute candidate lists so the closure has no OmegaConf coupling.
        lags_choices = [_to_python_list(x) for x in _to_python_list(fs.get("lags"))]
        date_features_choices = [
            _to_python_list(x) for x in _to_python_list(fs.get("date_features"))
        ]
        target_transforms_choices = _to_python_list(fs.get("target_transforms"))
        lag_transforms_choices = _to_python_list(fs.get("lag_transforms"))

        # If every list is empty, there's nothing to tune.
        if not any(
            [lags_choices, date_features_choices, target_transforms_choices, lag_transforms_choices]
        ):
            return None

        def init_config(trial: optuna.Trial) -> Dict[str, Any]:
            cfg: Dict[str, Any] = {}
            if lags_choices:
                idx = trial.suggest_categorical(
                    "lags_choice", list(range(len(lags_choices)))
                )
                cfg["lags"] = [int(x) for x in lags_choices[idx]]
            if date_features_choices:
                idx = trial.suggest_categorical(
                    "date_features_choice", list(range(len(date_features_choices)))
                )
                resolved = _resolve_date_features(date_features_choices[idx])
                if resolved:
                    cfg["date_features"] = resolved
            if target_transforms_choices:
                choice = trial.suggest_categorical(
                    "target_transforms_choice", target_transforms_choices
                )
                resolved = _resolve_target_transforms([choice])
                if resolved:
                    cfg["target_transforms"] = resolved
            if lag_transforms_choices:
                choice = trial.suggest_categorical(
                    "lag_transforms_choice", lag_transforms_choices
                )
                resolved = _resolve_lag_transforms(choice)
                if resolved:
                    cfg["lag_transforms"] = resolved
            return cfg

        return init_config

    def _build_fit_config_fn(self, fts):
        """Return an Optuna ``fit_config`` callable for fit-arg tuning, or None."""
        if not fts:
            return None

        use_static_choices = _to_python_list(fts.get("use_static_features"))
        dropna_choices = _to_python_list(fts.get("dropna"))

        if not (use_static_choices or dropna_choices):
            return None

        def fit_config(trial: optuna.Trial) -> Dict[str, Any]:
            # Always pass ``static_features=[]`` for the same reason as
            # ``_build_fit_kwargs`` — see that method's docstring. The
            # ``use_static_features`` knob is preserved as an Optuna
            # categorical so the same YAML schema works as before, but it no
            # longer affects MLForecast's static-vs-dynamic handling; static
            # feature columns are now always treated as dynamic.
            cfg: Dict[str, Any] = {"static_features": []}
            if use_static_choices:
                trial.suggest_categorical("use_static_features", use_static_choices)
            if dropna_choices:
                cfg["dropna"] = trial.suggest_categorical("dropna", dropna_choices)
            return cfg

        return fit_config

    def fit(self, x, y=None):
        pdf = self.prepare_data(x)
        self.model.fit(
            df=pdf,
            n_windows=int(self.params.get("num_windows", 3)),
            h=int(self.params["prediction_length"]),
            num_samples=int(self.params.get("num_samples", 20)),
        )


# -- MLflow pyfunc wrapper for the registered model --------------------------


class MLForecastModel(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper that delegates to the pipeline's ``forecast()`` so the
    ``Forecaster.score_global_model`` path works unchanged. Used by both
    ``MLForecastLGBM`` and ``MLForecastAutoLGBM``."""

    def __init__(self, pipeline: MLForecastBase):
        self.pipeline = pipeline

    def predict(self, context, model_input, params=None):
        forecast, _ = self.pipeline.forecast(model_input)
        return forecast
