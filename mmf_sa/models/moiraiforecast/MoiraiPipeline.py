import pandas as pd
import numpy as np
import logging
import torch
import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
from typing import Iterator
from pyspark.sql.functions import pandas_udf
from pyspark.sql import DataFrame
from mmf_sa.models.abstract_model import ForecastingRegressor, MODEL_PIP_REQUIREMENTS, collect_sorted
from mmf_sa.calibration import decode_quantile_forecasts, encode_quantile_forecasts, forecast_levels

_logger = logging.getLogger(__name__)


def sample_quantile_forecasts(samples: np.ndarray, quantiles):
    """
    Median and native quantiles of one series from its Moirai sample paths: empirical quantiles per step.
    Parameters:
        samples (np.ndarray): Sample paths, shape (num_samples, prediction_length).
        quantiles: Requested quantile levels; the median is added if missing.
    Returns: (np.ndarray, np.ndarray, list): the median (H,), the quantiles (Q, H) and the Q levels.
    """
    levels = forecast_levels(quantiles)
    median = np.median(samples, axis=0)
    qs = np.quantile(np.asarray(samples, dtype=np.float64), levels, axis=0)
    return median.astype(np.float64), qs, levels


def pack_sample_forecasts(samples: np.ndarray, quantiles):
    """
    Packs one series' sample paths into the predict UDF output. Without quantiles this is the median, as before;
    with quantiles it is the median and the quantiles, packed by encode_quantile_forecasts.
    """
    if not quantiles:
        return np.median(samples, axis=0)
    medians, qs, _ = sample_quantile_forecasts(samples, quantiles)
    return encode_quantile_forecasts(medians[np.newaxis], qs[np.newaxis])[0]


class MoiraiForecaster(ForecastingRegressor):
    # Native quantiles are empirical quantiles of the sample paths: with few samples the tails are poorly estimated.
    native_quantiles_supported = True
    samples_hparam = "num_samples"

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = None
        self.model = None
        self.repo = None

    def register(self, registered_model_name: str):
        pipeline = MoiraiModel(
            self.repo,
            self.params["prediction_length"],
            self.params["patch_size"],
            self.params["num_samples"],
        )
        input_schema = Schema([TensorSpec(np.dtype(np.double), (-1,))])
        output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1,))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = np.random.rand(52)
        mlflow.pyfunc.log_model(
            "model",
            python_model=pipeline,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=MODEL_PIP_REQUIREMENTS["moirai"],
        )

    def create_horizon_timestamps_udf(self):
        @pandas_udf('array<timestamp>')
        def horizon_timestamps_udf(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
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

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> DataFrame:
        df = collect_sorted(
            spark.createDataFrame(df), self.params.group_id, self.params.date_col, [self.params.target]
        )
        return df.select(
            df[self.params.group_id].alias("unique_id"),
            df[self.params.date_col].alias("ds"),
            df[self.params.target].alias("y"),
        )

    def predict(self,
                hist_df: pd.DataFrame,
                val_df: pd.DataFrame = None,
                curr_date=None,
                spark=None):
        hist_df = self.prepare_data(hist_df, spark=spark)
        horizon_timestamps_udf = self.create_horizon_timestamps_udf()
        device_count = torch.cuda.device_count()
        if device_count == 0:
            device_count = 1
        forecast_udf = self.create_predict_udf(device_count)
        forecast_df = (
            hist_df.repartition(device_count, self.params.group_id)
            .select(
                hist_df.unique_id,
                horizon_timestamps_udf(hist_df.ds).alias("ds"),
                forecast_udf(hist_df.y).alias("y"))
        ).toPandas()

        if self.params.get("native_quantiles"):
            forecast_df = decode_quantile_forecasts(
                forecast_df, "y", forecast_levels(self.params["native_quantiles"]), self.params["prediction_length"])

        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                "y": self.params.target,
            }
        )

        # Todo
        # forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame, spark=None):
        hist_df = df[df[self.params.target].notnull()]
        return self.predict(hist_df, spark=spark)

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> list:
        pred_df, _ = self.predict(hist_df, val_df, curr_date, spark)
        return self.series_metrics(pred_df, val_df, curr_date)

    def create_predict_udf(self, device_count):
        repo = self.repo
        num_devices = device_count
        prediction_length = self.params["prediction_length"]
        patch_size = self.params.get("patch_size", 16)
        num_samples = self.params["num_samples"]
        quantiles = [float(q) for q in self.params.get("native_quantiles") or []]
        self.native_levels()  # warns on the driver when num_samples is too small for the tails

        @pandas_udf('array<double>')
        def predict_udf(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            # initialization step
            import torch
            import numpy as np
            import pandas as pd
            from pyspark import TaskContext
            from einops import rearrange

            # Assign this worker to a specific GPU based on partition ID
            ctx = TaskContext.get()
            partition_id = ctx.partitionId() if ctx else 0
            gpu_id = partition_id % num_devices
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)

            from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
            from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
            if 'moe' in repo:
                module = MoiraiMoEModule.from_pretrained(repo).to(device)
            else:
                module = MoiraiModule.from_pretrained(repo).to(device)
            # inference
            median = []
            for batch in batch_iterator:
                for series in batch:
                    if 'moe' in repo:
                        model = MoiraiMoEForecast(
                            module=module,
                            prediction_length=prediction_length,
                            context_length=len(series),
                            patch_size=16,
                            num_samples=num_samples,
                            target_dim=1,
                            feat_dynamic_real_dim=0,
                            past_feat_dynamic_real_dim=0,
                        )
                    else:
                        model = MoiraiForecast(
                            module=module,
                            prediction_length=prediction_length,
                            context_length=len(series),
                            patch_size=patch_size,
                            num_samples=num_samples,
                            target_dim=1,
                            feat_dynamic_real_dim=0,
                            past_feat_dynamic_real_dim=0,
                        )

                    # Time series values. Shape: (batch, time, variate)
                    past_target = rearrange(
                        torch.as_tensor(series, dtype=torch.float32), "t -> 1 t 1"
                    ).to(device)
                    # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
                    past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
                    # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
                    past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

                    forecast = model(
                        past_target=past_target,
                        past_observed_target=past_observed_target,
                        past_is_pad=past_is_pad,
                    )
                    median.append(pack_sample_forecasts(forecast[0].cpu().numpy(), quantiles))
            yield pd.Series(median)
        return predict_udf


class MoiraiSmall(MoiraiForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Salesforce/moirai-1.1-R-small"


class MoiraiBase(MoiraiForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Salesforce/moirai-1.1-R-base"


class MoiraiLarge(MoiraiForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Salesforce/moirai-1.1-R-large"

class MoiraiMoESmall(MoiraiForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Salesforce/moirai-moe-1.0-R-small"

class MoiraiMoEBase(MoiraiForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Salesforce/moirai-moe-1.0-R-base"

class MoiraiMoELarge(MoiraiForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Salesforce/moirai-moe-1.0-R-large"

class MoiraiModel(mlflow.pyfunc.PythonModel):
    def __init__(self, repo, prediction_length, patch_size, num_samples):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
        self.repo = repo
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        if 'moe' in self.repo:
            self.module = MoiraiMoEModule.from_pretrained(self.repo)
        else:
            self.module = MoiraiModule.from_pretrained(self.repo)
        self.pipeline = None

    def predict(self, context, input_data, params=None):
        from einops import rearrange
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
        if 'moe' in self.repo:
            self.pipeline = MoiraiMoEForecast(
                module=self.module,
                prediction_length=self.prediction_length,
                context_length=len(input_data),
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        else:
            self.pipeline = MoiraiForecast(
                module=self.module,
                prediction_length=self.prediction_length,
                context_length=len(input_data),
                patch_size=16,
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        # Time series values. Shape: (batch, time, variate)
        past_target = rearrange(torch.as_tensor(input_data, dtype=torch.float32), "t -> 1 t 1")
        # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
        forecast = self.pipeline(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        forecast_np = forecast[0].cpu().numpy() if forecast[0].is_cuda else forecast[0].numpy()
        return np.median(forecast_np, axis=0)
