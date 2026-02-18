import pandas as pd
import numpy as np
import logging
import torch
import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
from typing import Iterator, Tuple
from pyspark.sql.functions import collect_list, pandas_udf
from pyspark.sql import DataFrame
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
        self.model_hparams = {}

    def _get_or_create_model(self):
        """Lazy-load the TimesFM model on the driver (for single-GPU fallback with covariates)."""
        if self.model is None:
            import timesfm
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo)
            forecast_config = timesfm.ForecastConfig(
                max_context=self.model_hparams.get("context_len", 512),
                max_horizon=max(self.params.prediction_length, 128),
                per_core_batch_size=32,
            )
            self.model.compile(forecast_config)
        return self.model

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
            pip_requirements=[
                "timesfm[torch] @ git+https://github.com/google-research/timesfm.git@2dcc66fbfe2155adba1af66aa4d564a0ee52f61e",
                "git+https://github.com/databricks-industry-solutions/many-model-forecasting.git",
            ],
        )

    def create_horizon_timestamps_udf(self):
        @pandas_udf('array<timestamp>')
        def horizon_timestamps_udf(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            import numpy as np
            batch_horizon_timestamps = []
            for batch in batch_iterator:
                for series in batch:
                    last = series.max()
                    horizon_timestamps = []
                    for i in range(self.params["prediction_length"]):
                        last = last + self.one_ts_offset
                        horizon_timestamps.append(last.to_numpy())
                    batch_horizon_timestamps.append(np.array(horizon_timestamps))
            yield pd.Series(batch_horizon_timestamps)
        return horizon_timestamps_udf

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> pd.DataFrame:
        """Prepare data as pandas DataFrame (column selection and renaming).
        Used by the single-GPU fallback path for covariate handling."""
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
            if 'static_features' in self.params.keys():
                try:
                    features = features + self.params.static_features
                except Exception as e:
                    raise MissingFeatureError(f"Static features missing: {e}")
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

    def prepare_data_for_spark(self, hist_df: pd.DataFrame, spark,
                               val_df: pd.DataFrame = None) -> DataFrame:
        """Convert pandas DataFrame to Spark DataFrame grouped by unique_id with collected lists.
        Used by the distributed multi-GPU prediction path. Supports covariates.

        The resulting Spark DataFrame has one row per unique_id with columns:
          - ``ds``: array of historical timestamps
          - ``y``: array of historical target values
          - one array column per dynamic covariate (spanning history + future)
          - one scalar column per static feature
        """
        dyn_num_vars = list(self.params.get("dynamic_future_numerical", []))
        dyn_cat_vars = list(self.params.get("dynamic_future_categorical", []))
        static_vars = list(self.params.get("static_features", []))
        dynamic_vars = dyn_num_vars + dyn_cat_vars

        hist_pdf = self.prepare_data(hist_df)

        # Core Spark DataFrame: y and ds from historical data only
        core_pdf = hist_pdf[["unique_id", "ds", "y"]].sort_values(
            by=["unique_id", "ds"]
        )
        core_sdf = spark.createDataFrame(core_pdf)
        core_sdf = core_sdf.groupBy("unique_id").agg(
            collect_list("ds").alias("ds"),
            collect_list("y").alias("y"),
        )

        # Dynamic covariate columns (from union of historical + future data)
        if dynamic_vars:
            if val_df is not None:
                future_pdf = self.prepare_data(val_df, future=True)
                union_pdf = pd.concat(
                    [hist_pdf, future_pdf], axis=0, join="outer", ignore_index=True
                ).sort_values(by=["unique_id", "ds"])
            else:
                union_pdf = hist_pdf.sort_values(by=["unique_id", "ds"])

            cov_pdf = union_pdf[["unique_id"] + dynamic_vars]
            cov_sdf = spark.createDataFrame(cov_pdf)
            cov_agg = [collect_list(v).alias(v) for v in dynamic_vars]
            cov_sdf = cov_sdf.groupBy("unique_id").agg(*cov_agg)
            core_sdf = core_sdf.join(cov_sdf, on="unique_id", how="left")

        # Static feature columns (scalar per unique_id, from original data)
        if static_vars:
            static_pdf = hist_df[
                [self.params.group_id] + static_vars
            ].drop_duplicates(subset=[self.params.group_id])
            static_pdf = static_pdf.rename(
                columns={self.params.group_id: "unique_id"}
            )
            static_sdf = spark.createDataFrame(static_pdf)
            core_sdf = core_sdf.join(static_sdf, on="unique_id", how="left")

        return core_sdf

    def create_predict_udf(self, device_count, col_map=None):
        """Create a pandas UDF that loads the TimesFM model inside each Spark worker
        and runs distributed inference across multiple GPUs.

        Parameters
        ----------
        device_count : int
            Number of GPUs available (used for partition-to-GPU mapping).
        col_map : list[tuple[str, str]] or None
            Ordered mapping of covariate columns that the UDF will receive
            after the leading ``y`` column.  Each entry is
            ``(category, variable_name)`` where *category* is one of
            ``"dyn_num"``, ``"dyn_cat"``, ``"stat_num"``, ``"stat_cat"``.
            When *None* or empty the UDF performs plain forecasting without
            covariates.
        """
        repo = self.repo
        prediction_length = self.params["prediction_length"]
        model_hparams = self.model_hparams
        num_devices = device_count
        _col_map = list(col_map) if col_map else []
        _has_covariates = len(_col_map) > 0

        @pandas_udf('array<double>')
        def predict_udf(
            bulk_iterator: Iterator[Tuple[pd.Series, ...]]
        ) -> Iterator[pd.Series]:
            import torch
            import timesfm
            import numpy as np
            import pandas as pd
            from pyspark import TaskContext

            ctx = TaskContext.get()
            partition_id = ctx.partitionId() if ctx else 0
            gpu_id = partition_id % num_devices
            torch.cuda.set_device(gpu_id)

            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo)
            inner_module = object.__getattribute__(model, 'model')
            inner_module.device = torch.device(f"cuda:{gpu_id}")
            inner_module.to(inner_module.device)

            forecast_config = timesfm.ForecastConfig(
                max_context=model_hparams.get("context_len", 512),
                max_horizon=max(prediction_length, 128),
                per_core_batch_size=32,
                return_backcast=_has_covariates,
            )
            model.compile(forecast_config)

            for batch_tuple in bulk_iterator:
                # When a single column is passed, PySpark delivers a bare
                # pd.Series instead of a one-element tuple.  Normalise so
                # the rest of the code can always index by position.
                if not isinstance(batch_tuple, tuple):
                    batch_tuple = (batch_tuple,)

                y_batch = batch_tuple[0]
                inputs = [
                    np.array(list(series), dtype=np.float64)
                    for series in y_batch
                ]

                if not _has_covariates:
                    forecasts, _ = model.forecast(
                        horizon=prediction_length,
                        inputs=inputs,
                    )
                else:
                    dyn_num_dict = {}
                    dyn_cat_dict = {}
                    stat_num_dict = {}
                    stat_cat_dict = {}

                    for idx, (category, name) in enumerate(_col_map):
                        col_batch = batch_tuple[1 + idx]
                        if category == "dyn_num":
                            dyn_num_dict[name] = [
                                np.array(list(s), dtype=np.float64)
                                for s in col_batch
                            ]
                        elif category == "dyn_cat":
                            dyn_cat_dict[name] = [
                                np.array(list(s)) for s in col_batch
                            ]
                        elif category == "stat_num":
                            stat_num_dict[name] = [
                                float(s) for s in col_batch
                            ]
                        elif category == "stat_cat":
                            stat_cat_dict[name] = [
                                str(s) for s in col_batch
                            ]

                    forecasts, _ = model.forecast_with_covariates(
                        inputs=inputs,
                        dynamic_numerical_covariates=dyn_num_dict,
                        dynamic_categorical_covariates=dyn_cat_dict,
                        static_numerical_covariates=stat_num_dict,
                        static_categorical_covariates=stat_cat_dict,
                        xreg_mode="xreg + timesfm",
                        ridge=0.0,
                        force_on_cpu=True,
                        normalize_xreg_target_per_input=True,
                    )

                yield pd.Series(
                    [arr.astype(np.float64) for arr in forecasts]
                )

        return predict_udf

    def predict(self,
                hist_df: pd.DataFrame,
                val_df: pd.DataFrame = None,
                spark=None):
        """Route to distributed or single-GPU prediction based on available context."""
        if spark is not None:
            return self._predict_distributed(hist_df, spark, val_df)
        else:
            return self._predict_single(hist_df, val_df)

    def _predict_distributed(self, hist_df: pd.DataFrame, spark,
                             val_df: pd.DataFrame = None):
        """Distributed prediction using Spark pandas UDFs across multiple GPUs.
        Each partition explicitly targets a specific GPU via torch.cuda.set_device.
        Supports dynamic and static covariates."""
        # Classify covariate columns
        dyn_num_vars = list(self.params.get("dynamic_future_numerical", []))
        dyn_cat_vars = list(self.params.get("dynamic_future_categorical", []))
        static_vars = list(self.params.get("static_features", []))

        stat_num_vars = []
        stat_cat_vars = []
        for var in static_vars:
            if pd.api.types.is_numeric_dtype(hist_df[var]):
                stat_num_vars.append(var)
            else:
                stat_cat_vars.append(var)

        # Build ordered column map consumed by the UDF closure.
        # Position 0 in the UDF tuple is always ``y``; covariate columns
        # follow in the order recorded here.
        col_map = []
        for var in dyn_num_vars:
            col_map.append(("dyn_num", var))
        for var in dyn_cat_vars:
            col_map.append(("dyn_cat", var))
        for var in stat_num_vars:
            col_map.append(("stat_num", var))
        for var in stat_cat_vars:
            col_map.append(("stat_cat", var))

        hist_sdf = self.prepare_data_for_spark(hist_df, spark, val_df)
        horizon_timestamps_udf = self.create_horizon_timestamps_udf()
        device_count = torch.cuda.device_count()
        if device_count == 0:
            device_count = 1
        forecast_udf = self.create_predict_udf(device_count, col_map)

        # Build UDF column list: y first, then covariates in col_map order
        udf_columns = [hist_sdf.y]
        for _, var in col_map:
            udf_columns.append(hist_sdf[var])

        forecast_df = (
            hist_sdf.repartition(device_count, "unique_id")
            .select(
                hist_sdf.unique_id,
                horizon_timestamps_udf(hist_sdf.ds).alias("ds"),
                forecast_udf(*udf_columns).alias("y"),
            )
        ).toPandas()

        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                "y": self.params.target,
            }
        )
        return forecast_df, self.model

    def _predict_single(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Single-GPU prediction (used when Spark session is unavailable)."""
        model = self._get_or_create_model()
        df = self.prepare_data(hist_df)
        dynamic_covariates = self.prepare_data(val_df, future=True) if val_df is not None else None

        if dynamic_covariates is not None:
            df_union = pd.concat([df, dynamic_covariates], axis=0, join='outer', ignore_index=True)
        else:
            df_union = df

        forecast_input = [group['y'].values for _, group in df.groupby('unique_id')]
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
        if not dynamic_numerical_covariates \
            and not dynamic_categorical_covariates \
            and not static_numerical_covariates \
            and not static_categorical_covariates:
            forecasts, _ = model.forecast(
                horizon=self.params.prediction_length,
                inputs=forecast_input,
            )
        else:
            # Recompile with return_backcast=True required by forecast_with_covariates
            import timesfm
            xreg_config = timesfm.ForecastConfig(
                max_context=self.model_hparams.get("context_len", 512),
                max_horizon=max(self.params.prediction_length, 128),
                per_core_batch_size=32,
                return_backcast=True,
            )
            model.compile(xreg_config)
            forecasts, _ = model.forecast_with_covariates(
                inputs=forecast_input,
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                dynamic_categorical_covariates=dynamic_categorical_covariates,
                static_numerical_covariates=static_numerical_covariates,
                static_categorical_covariates=static_categorical_covariates,
                xreg_mode="xreg + timesfm",  # default
                ridge=0.0,
                force_on_cpu=False,
                normalize_xreg_target_per_input=True,  # default
            )

        unique_ids = np.array(df["unique_id"].unique())
        timestamps = make_future_dataframe(
            uids=unique_ids,
            last_times=df.groupby("unique_id")["ds"].tail(1),
            h=self.params.prediction_length,
            freq=self.params.freq,
        )
        timestamps = [group['ds'].values for _, group in timestamps.groupby('unique_id')]
        forecast_df = pd.DataFrame({
            self.params.group_id: unique_ids,
            self.params.date_col: timestamps,
            self.params.target: [arr.astype(np.float64) for arr in forecasts],
        })

        # Todo
        # forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df, model

    def forecast(self, df: pd.DataFrame, spark=None):
        hist_df = df[df[self.params.target].notnull()]
        last_date = hist_df[self.params.date_col].max()
        future_df = df[
            (df[self.params.date_col] > np.datetime64(last_date))
            & (df[self.params.date_col]
               <= np.datetime64(last_date + self.prediction_length_offset))
            ]
        forecast_df, model = self.predict(hist_df=hist_df, val_df=future_df, spark=spark)
        return forecast_df, model

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> list:
        pred_df, model_pretrained = self.predict(hist_df, val_df, spark=spark)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        metric_name = self.params["metric"]
        if metric_name not in ("smape", "mape", "mae", "mse", "rmse"):
            raise UnsupportedMetricError(f"Metric {self.params['metric']} not supported!")
        for key in keys:
            actual = val_df[val_df[self.params["group_id"]] == key][self.params["target"]].to_numpy()
            forecast = np.array(pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].iloc[0])
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


class TimesFM_2_5_200m(TimesFMForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "google/timesfm-2.5-200m-pytorch"
        self.model_hparams = {}


class TimesFMModel(mlflow.pyfunc.PythonModel):
    def __init__(self, params, repo):
        # Store only simple Python types for pickling compatibility.
        # The heavy model is loaded lazily in predict().
        self.prediction_length = int(params["prediction_length"])
        self.repo = repo
        self.model = None

    def _load_model(self):
        if self.model is None:
            import timesfm
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo)
            forecast_config = timesfm.ForecastConfig(
                max_context=512,
                max_horizon=max(self.prediction_length, 128),
                per_core_batch_size=32,
            )
            self.model.compile(forecast_config)

    def predict(self, context, input_data, params=None):
        import numpy as np
        self._load_model()
        # input_data is expected to be a numpy array of shape (batch, context)
        if hasattr(input_data, 'values'):
            input_data = input_data.values
        inputs = [input_data[i] for i in range(input_data.shape[0])]
        point_forecast, _ = self.model.forecast(
            horizon=self.prediction_length,
            inputs=inputs,
        )
        return point_forecast
