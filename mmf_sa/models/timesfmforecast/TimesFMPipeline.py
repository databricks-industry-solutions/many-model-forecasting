import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
from utilsforecast.processing import make_future_dataframe
from mmf_sa.models.abstract_model import ForecastingRegressor
from mmf_sa.exceptions import MissingFeatureError, UnsupportedMetricError, ModelPredictionError, DataPreparationError

_logger = logging.getLogger(__name__)


class TimesFMForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = None
        self.model = None
        self.repo = None

    def register(self, registered_model_name: str):
        pipeline = TimesFMModel(self.params, self.repo)
        input_schema = Schema([TensorSpec(np.dtype(np.double), (-1, -1))])
        output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, -1))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pyfunc.log_model(
            "model",
            python_model=pipeline,
            registered_model_name=registered_model_name,
            signature=signature,
            #input_example=input_example,
            pip_requirements=[
                "timesfm[torch]>=2.5.0",
                "git+https://github.com/databricks-industry-solutions/many-model-forecasting.git",
                "pyspark==3.5.0",
            ],
        )

    def _prepare_forecast_inputs(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        df = self.prepare_data(hist_df)
        dynamic_covariates = None
        if val_df is not None:
            dynamic_covariates = self.prepare_data(val_df, future=True)
        df_union = (
            pd.concat([df, dynamic_covariates], axis=0, join='outer', ignore_index=True)
            if dynamic_covariates is not None
            else df
        )
        forecast_input = [group['y'].values for _, group in df.groupby('unique_id')]
        freq_index = 0 if self.params.freq in ("H", "D") else 1
        dynamic_numerical_covariates = {}
        if 'dynamic_future_numerical' in self.params.keys():
            for var in self.params.dynamic_future_numerical:
                dynamic_numerical_covariates[var] = [group[var].values for _, group in df_union.groupby('unique_id')]
        dynamic_categorical_covariates = {}
        if 'dynamic_future_categorical' in self.params.keys():
            for var in self.params.dynamic_future_categorical:
                dynamic_categorical_covariates[var] = [group[var].values for _, group in df_union.groupby('unique_id')]
        static_numerical_covariates = {}
        static_categorical_covariates = {}
        if 'static_features' in self.params.keys():
            for var in self.params.static_features:
                if pd.api.types.is_numeric_dtype(df[var]):
                    static_numerical_covariates[var] = [group[var].iloc[0] for _, group in df.groupby('unique_id')]
                else:
                    static_categorical_covariates[var] = [group[var].iloc[0] for _, group in df.groupby('unique_id')]
        unique_ids = np.array(df["unique_id"].unique())
        timestamps = make_future_dataframe(
            uids=unique_ids,
            last_times=df.groupby("unique_id")["ds"].tail(1),
            h=self.params.prediction_length,
            freq=self.params.freq,
        )
        timestamps = [group['ds'].values for _, group in timestamps.groupby('unique_id')]
        return (
            forecast_input,
            freq_index,
            dynamic_numerical_covariates,
            dynamic_categorical_covariates,
            static_numerical_covariates,
            static_categorical_covariates,
            unique_ids,
            timestamps,
        )

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> pd.DataFrame:
        if not future:
            # Prepare historical dataframe with or without exogenous regressors for training
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
            # Prepare future dataframe with exogenous regressors for forecasting
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
        return _df.sort_values(by=["unique_id", "ds"])

    def predict(self,
                hist_df: pd.DataFrame,
                val_df: pd.DataFrame = None,
                spark=None):
        (
            forecast_input,
            freq_index,
            dynamic_numerical_covariates,
            dynamic_categorical_covariates,
            static_numerical_covariates,
            static_categorical_covariates,
            unique_ids,
            timestamps,
        ) = self._prepare_forecast_inputs(hist_df, val_df)
        if not dynamic_numerical_covariates \
            and not dynamic_categorical_covariates \
            and not static_numerical_covariates \
            and not static_categorical_covariates:
            forecasts, _ = self.model.forecast(
                inputs=forecast_input,
                freq=[freq_index] * len(forecast_input)
            )
        else:
            if not hasattr(self.model, "forecast_with_covariates"):
                raise MissingFeatureError(
                    "This TimesFM model does not support covariates. "
                    "Use a TimesFM variant with xreg support."
                )
            forecasts, _ = self.model.forecast_with_covariates(
                inputs=forecast_input,
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                dynamic_categorical_covariates=dynamic_categorical_covariates,
                static_numerical_covariates=static_numerical_covariates,
                static_categorical_covariates=static_categorical_covariates,
                freq=[freq_index] * len(forecast_input),
                xreg_mode="xreg + timesfm",  # default
                ridge=0.0,
                force_on_cpu=False,
                normalize_xreg_target_per_input=True,  # default
            )
        forecast_df = pd.DataFrame({
            self.params.group_id: unique_ids,
            self.params.date_col: timestamps,
            self.params.target: [arr.astype(np.float64) for arr in forecasts],
        })

        # Todo
        # forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame, spark=None):
        hist_df = df[df[self.params.target].notnull()]
        last_date = hist_df[self.params.date_col].max()
        future_df = df[
            (df[self.params.date_col] > np.datetime64(last_date))
            & (df[self.params.date_col]
               <= np.datetime64(last_date + self.prediction_length_offset))
            ]
        forecast_df, model = self.predict(hist_df=hist_df, val_df=future_df)
        return forecast_df, model

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> list:
        pred_df, model_pretrained = self.predict(hist_df, val_df)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        metric_name = self.params["metric"]
        if metric_name not in ("smape", "mape", "mae", "mse", "rmse"):
            raise UnsupportedMetricError(f"Metric {self.params['metric']} not supported!")
        for key in keys:
            actual = val_df[val_df[self.params["group_id"]] == key][self.params["target"]].to_numpy()
            forecast = np.array(pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].iloc[0])
            if np.isnan(actual).any() or np.isnan(forecast).any():
                _logger.warning(f"Skipping metric for key {key}: NaN in actual/forecast.")
                continue
            # Mapping metric names to their respective classes
            metric_classes = {
                "smape": MeanAbsolutePercentageError(symmetric=True),
                "mape": MeanAbsolutePercentageError(symmetric=False),
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(square_root=False),
                "rmse": MeanSquaredError(square_root=True),
            }
            try:
                if metric_name in metric_classes:
                    metric_function = metric_classes[metric_name]
                    metric_value = metric_function(actual, forecast)
                metrics.extend(
                    [(
                        key,
                        curr_date,
                        metric_name,
                        metric_value,
                        forecast,
                        actual,
                        b'',
                    )])
            except (ModelPredictionError, DataPreparationError) as err:
                _logger.warning(f"Failed to calculate metric for key {key}: {err}")
            except Exception as err:
                _logger.warning(f"Unexpected error calculating metric for key {key}: {err}")
        return metrics


class TimesFM_1_0_200m(TimesFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        import timesfm
        self.params = params
        #self.backend = "gpu" if torch.cuda.is_available() else "cpu"
        self.repo = "google/timesfm-1.0-200m-pytorch"
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=self.params.prediction_length,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.repo
            ),
        )

class TimesFM_2_0_500m(TimesFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        import timesfm
        self.params = params
        #self.backend = "gpu" if torch.cuda.is_available() else "cpu"
        self.repo = "google/timesfm-2.0-500m-pytorch"
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=self.params.prediction_length,
                num_layers=50,
                use_positional_embedding=False,
                context_len=2048,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.repo
            ),
        )

class TimesFM_2_5_200m(TimesFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        import timesfm
        self.params = params
        self.repo = "google/timesfm-2.5-200m-pytorch"
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.repo, torch_compile=True
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.params.get("max_context", 2048),
                max_horizon=self.params.prediction_length,
                normalize_inputs=False,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )
    
    def register(self, registered_model_name: str):
        pipeline = TimesFM2p5Model(self.params, self.repo)
        input_schema = Schema([TensorSpec(np.dtype(np.double), (-1, -1))])
        output_schema = Schema([TensorSpec(np.dtype(np.double), (-1, -1))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pyfunc.log_model(
            "model",
            python_model=pipeline,
            registered_model_name=registered_model_name,
            signature=signature,
            pip_requirements=[
                "timesfm @ git+https://github.com/google-research/timesfm.git",
                "git+https://github.com/databricks-industry-solutions/many-model-forecasting.git",
                "pyspark==3.5.0",
            ],
        )

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None, spark=None):
        allow_fallback = True
        pred_len = int(self.params.get("prediction_length", 1))
        (
            forecast_input,
            _,
            dynamic_numerical_covariates,
            dynamic_categorical_covariates,
            static_numerical_covariates,
            static_categorical_covariates,
            unique_ids,
            timestamps,
        ) = self._prepare_forecast_inputs(hist_df, val_df)
        max_context = int(self.params.get("max_context", 2048))

        def _sanitize_series(values: np.ndarray) -> tuple[np.ndarray, float, bool, bool, float]:
            series = np.asarray(values, dtype=np.float32)
            if series.size == 0:
                return series, 1.0, False, False, 0.0
            if not np.isfinite(series).all():
                series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            if np.isnan(series).any():
                # Forward-fill NaNs; if all NaN, fallback to zeros
                s = pd.Series(series)
                s = s.fillna(method="ffill").fillna(0.0)
                series = s.to_numpy(dtype=np.float32)
            if series.size > max_context:
                series = series[-max_context:]
            last_val = float(series[-1]) if series.size else 0.0
            use_log = bool(series.min() >= 0.0)
            if use_log:
                series = np.log1p(series)
            scale = float(np.nanpercentile(np.abs(series), 90))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            series = series / scale
            # Add a tiny ramp for near-constant series to avoid zero-variance issues
            degenerate = float(np.std(series)) < 1e-3
            if degenerate:
                series = series + np.linspace(0.0, 1e-3, series.size, dtype=np.float32)
            return series, scale, use_log, degenerate, last_val

        sanitized = [_sanitize_series(arr) for arr in forecast_input]
        forecast_input = [s[0] for s in sanitized]
        scales = [s[1] for s in sanitized]
        use_logs = [s[2] for s in sanitized]
        degenerates = [s[3] for s in sanitized]
        last_values = [s[4] for s in sanitized]
        use_covariates = (
            dynamic_numerical_covariates
            or dynamic_categorical_covariates
            or static_numerical_covariates
            or static_categorical_covariates
        )
        if use_covariates and hasattr(self.model, "forecast_with_covariates"):
            forecasts, _ = self.model.forecast_with_covariates(
                inputs=forecast_input,
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                dynamic_categorical_covariates=dynamic_categorical_covariates,
                static_numerical_covariates=static_numerical_covariates,
                static_categorical_covariates=static_categorical_covariates,
                freq=[0 if self.params.freq in ("H", "D") else 1] * len(forecast_input),
                xreg_mode="xreg + timesfm",
                ridge=0.0,
                force_on_cpu=False,
                normalize_xreg_target_per_input=True,
            )
        elif use_covariates:
            raise MissingFeatureError(
                "TimesFM 2.5 covariates require a timesfm build with xreg support."
            )
        else:
            point_forecast, _ = self.model.forecast(
                horizon=self.params.prediction_length,
                inputs=forecast_input,
            )
            forecasts = point_forecast
        used_fallback = False
        if np.isnan(forecasts).any():
            def _series_stats(series: np.ndarray) -> str:
                if series.size == 0:
                    return "empty"
                return (
                    f"len={series.size} min={float(series.min()):.4f} "
                    f"max={float(series.max()):.4f} mean={float(series.mean()):.4f} "
                    f"std={float(series.std()):.4f}"
                )

            bad_idx = np.where(np.isnan(np.asarray(forecasts)).any(axis=1))[0][:5]
            for idx in bad_idx:
                _logger.warning(
                    "TimesFM 2.5 NaN forecast for index %s; input stats: %s (scale=%.4f, log=%s, degenerate=%s)",
                    idx,
                    _series_stats(forecast_input[idx]),
                    scales[idx] if idx < len(scales) else 1.0,
                    use_logs[idx] if idx < len(use_logs) else False,
                    degenerates[idx] if idx < len(degenerates) else False,
                )
            if allow_fallback:
                _logger.warning(
                    "TimesFM 2.5 produced NaNs; applying last-value fallback."
                )
                forecasts = np.asarray(
                    [np.full(pred_len, v, dtype=np.float64) for v in last_values]
                )
                used_fallback = True
            else:
                raise ModelPredictionError(
                    "TimesFM 2.5 produced NaNs in forecast output."
                )
        # Rescale forecasts back to original scale
        forecasts = np.asarray(forecasts, dtype=np.float64)
        if not used_fallback and len(scales) == forecasts.shape[0]:
            forecasts = forecasts * np.asarray(scales, dtype=np.float64)[:, None]
        if not used_fallback and len(use_logs) == forecasts.shape[0] and any(use_logs):
            for i, use_log in enumerate(use_logs):
                if use_log:
                    forecasts[i] = np.expm1(forecasts[i])

        forecast_df = pd.DataFrame({
            self.params.group_id: unique_ids,
            self.params.date_col: timestamps,
            self.params.target: [np.asarray(arr, dtype=np.float64) for arr in forecasts],
        })
        return forecast_df, self.model

class TimesFMModel(mlflow.pyfunc.PythonModel):
    def __init__(self, params, repo):
        import timesfm
        self.params = params
        self.repo = repo
        #self.backend = "gpu" if torch.cuda.is_available() else "cpu"
        if self.repo == "google/timesfm-1.0-200m-pytorch":
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                    horizon_len=self.params.prediction_length,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.repo,
                ),
            )
        else:
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                    horizon_len=self.params.prediction_length,
                    num_layers=50,
                    use_positional_embedding=False,
                    context_len=2048,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.repo
                ),
        )

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        # Generate forecasts on the input DataFrame
        if hasattr(self.model, "forecast_on_df"):
            forecast_df = self.model.forecast_on_df(
                inputs=model_input,  # Input DataFrame containing the time series data.
                freq=self.params.freq,  # Frequency of the time series data, set to daily.
                value_name=self.params.target,  # Column name in the DataFrame containing the values to forecast.
                num_jobs=-1,  # Number of parallel jobs to run, set to -1 to use all available processors.
            )
            return forecast_df
        raise ModelPredictionError("timesfm model does not support forecast_on_df")


class TimesFM2p5Model(mlflow.pyfunc.PythonModel):
    def __init__(self, params, repo):
        import timesfm
        self.params = params
        self.repo = repo
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.repo, torch_compile=True
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.params.get("max_context", 2048),
                max_horizon=self.params.prediction_length,
                normalize_inputs=False,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        allow_fallback = True
        pred_len = int(self.params.get("prediction_length", 1))
        if not isinstance(model_input, pd.DataFrame):
            raise ModelPredictionError("TimesFM 2.5 expects a pandas DataFrame input.")
        if "unique_id" not in model_input.columns:
            raise ModelPredictionError("Input DataFrame must include a 'unique_id' column.")
        if "y" not in model_input.columns:
            raise ModelPredictionError("Input DataFrame must include a 'y' column.")
        
        max_context = int(self.params.get("max_context", 2048))

        def _sanitize_series(values: np.ndarray) -> tuple[np.ndarray, float, bool, bool, float]:
            series = np.asarray(values, dtype=np.float32)
            if series.size == 0:
                return series, 1.0, False, False, 0.0
            if not np.isfinite(series).all():
                series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            if np.isnan(series).any():
                s = pd.Series(series)
                s = s.fillna(method="ffill").fillna(0.0)
                series = s.to_numpy(dtype=np.float32)
            if series.size > max_context:
                series = series[-max_context:]
            last_val = float(series[-1]) if series.size else 0.0
            use_log = bool(series.min() >= 0.0)
            if use_log:
                series = np.log1p(series)
            scale = float(np.nanpercentile(np.abs(series), 90))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            series = series / scale
            degenerate = float(np.std(series)) < 1e-3
            if degenerate:
                series = series + np.linspace(0.0, 1e-3, series.size, dtype=np.float32)
            return series, scale, use_log, degenerate, last_val

        sanitized = [_sanitize_series(group["y"].values) for _, group in model_input.groupby("unique_id")]
        inputs = [s[0] for s in sanitized]
        scales = [s[1] for s in sanitized]
        use_logs = [s[2] for s in sanitized]
        degenerates = [s[3] for s in sanitized]
        last_values = [s[4] for s in sanitized]
        
        try:
            point_forecast, _ = self.model.forecast(
                horizon=self.params.prediction_length,
                inputs=inputs,
            )
        except Exception as e:
            raise ModelPredictionError(f"TimesFM 2.5 forecast failed: {e}") from e

        used_fallback = False
        if np.isnan(point_forecast).any():
            def _series_stats(series: np.ndarray) -> str:
                if series.size == 0:
                    return "empty"
                return (
                    f"len={series.size} min={float(series.min()):.4f} "
                    f"max={float(series.max()):.4f} mean={float(series.mean()):.4f} "
                    f"std={float(series.std()):.4f}"
                )

            bad_idx = np.where(np.isnan(np.asarray(point_forecast)).any(axis=1))[0][:5]
            for idx in bad_idx:
                _logger.warning(
                    "TimesFM 2.5 NaN forecast for index %s; input stats: %s (scale=%.4f, log=%s, degenerate=%s)",
                    idx,
                    _series_stats(inputs[idx]),
                    scales[idx] if idx < len(scales) else 1.0,
                    use_logs[idx] if idx < len(use_logs) else False,
                    degenerates[idx] if idx < len(degenerates) else False,
                )
            if allow_fallback:
                _logger.warning(
                    "TimesFM 2.5 produced NaNs; applying last-value fallback."
                )
                point_forecast = np.asarray(
                    [np.full(pred_len, v, dtype=np.float64) for v in last_values]
                )
                used_fallback = True
            else:
                raise ModelPredictionError(
                    "TimesFM 2.5 produced NaNs in forecast output."
                )

        point_forecast = np.asarray(point_forecast, dtype=np.float64)
        if not used_fallback and len(scales) == point_forecast.shape[0]:
            point_forecast = point_forecast * np.asarray(scales, dtype=np.float64)[:, None]
        if not used_fallback and len(use_logs) == point_forecast.shape[0] and any(use_logs):
            for i, use_log in enumerate(use_logs):
                if use_log:
                    point_forecast[i] = np.expm1(point_forecast[i])

        return point_forecast
