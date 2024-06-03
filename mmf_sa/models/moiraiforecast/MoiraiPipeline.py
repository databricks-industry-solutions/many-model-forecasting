from abc import ABC
import subprocess
import sys
import pandas as pd
import numpy as np
import torch
import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from typing import Iterator
from pyspark.sql.functions import collect_list, pandas_udf
from pyspark.sql import DataFrame
from mmf_sa.models.abstract_model import ForecastingRegressor


class MoiraiForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = None
        self.model = None
        self.install("git+https://github.com/SalesforceAIResearch/uni2ts.git")

    @staticmethod
    def install(package: str):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

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
            pip_requirements=[
                "git+https://github.com/SalesforceAIResearch/uni2ts.git",
                "git+https://github.com/databricks-industry-solutions/many-model-forecasting.git",
                "pyspark==3.5.0",
            ],
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
                        horizon_timestamps.append(last)
                    batch_horizon_timestamps.append(np.array(horizon_timestamps))
            yield pd.Series(batch_horizon_timestamps)
        return horizon_timestamps_udf

    def prepare_data(self, df: pd.DataFrame, future: bool = False, spark=None) -> DataFrame:
        df = spark.createDataFrame(df)
        df = (
            df.groupBy(self.params.group_id)
            .agg(
                collect_list(self.params.date_col).alias('ds'),
                collect_list(self.params.target).alias('y'),
            ))
        return df

    def predict(self,
                hist_df: pd.DataFrame,
                val_df: pd.DataFrame = None,
                curr_date=None,
                spark=None):
        hist_df = self.prepare_data(hist_df, spark=spark)
        horizon_timestamps_udf = self.create_horizon_timestamps_udf()
        forecast_udf = self.create_predict_udf()
        device_count = torch.cuda.device_count()
        forecast_df = (
            hist_df.repartition(device_count, self.params.group_id)
            .select(
                hist_df.unique_id,
                horizon_timestamps_udf(hist_df.ds).alias("ds"),
                forecast_udf(hist_df.y).alias("y"))
        ).toPandas()

        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                "y": self.params.target,
            }
        )

        # Todo
        #forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame, spark=None):
        return self.predict(df, spark=spark)

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> list:
        pred_df, model_pretrained = self.predict(hist_df, val_df, curr_date, spark)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        if self.params["metric"] == "smape":
            metric_name = "smape"
        else:
            raise Exception(f"Metric {self.params['metric']} not supported!")
        for key in keys:
            actual = val_df[val_df[self.params["group_id"]] == key][self.params["target"]].to_numpy()
            forecast = pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].to_numpy()[0]
            try:
                if metric_name == "smape":
                    metric_value = mean_absolute_percentage_error(actual, forecast, symmetric=True)
                metrics.extend(
                    [(
                        key,
                        curr_date,
                        metric_name,
                        metric_value,
                        actual,
                        forecast,
                        b'',
                    )])
            except:
                pass
        return metrics

    def create_predict_udf(self):
        @pandas_udf('array<double>')
        def predict_udf(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            # initialization step
            import torch
            import numpy as np
            import pandas as pd
            from einops import rearrange
            from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
            module = MoiraiModule.from_pretrained(self.repo)
            # inference
            for batch in batch_iterator:
                median = []
                for series in batch:
                    model = MoiraiForecast(
                        module=module,
                        prediction_length=self.params["prediction_length"],
                        context_length=len(series),
                        patch_size=self.params["patch_size"],
                        num_samples=self.params["num_samples"],
                        target_dim=1,
                        feat_dynamic_real_dim=0,
                        past_feat_dynamic_real_dim=0,
                    )

                    # Time series values. Shape: (batch, time, variate)
                    past_target = rearrange(
                        torch.as_tensor(series, dtype=torch.float32), "t -> 1 t 1"
                    )
                    # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
                    past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
                    # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
                    past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

                    forecast = model(
                        past_target=past_target,
                        past_observed_target=past_observed_target,
                        past_is_pad=past_is_pad,
                    )
                    median.append(np.median(forecast[0], axis=0))
            yield pd.Series(median)
        return predict_udf


class MoiraiSmall(MoiraiForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Salesforce/moirai-1.0-R-small"


class MoiraiBase(MoiraiForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Salesforce/moirai-1.0-R-base"


class MoiraiLarge(MoiraiForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.repo = "Salesforce/moirai-1.0-R-base"


class MoiraiModel(mlflow.pyfunc.PythonModel):
    def __init__(self, repository, prediction_length, patch_size, num_samples):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        self.repository = repository
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.module = MoiraiModule.from_pretrained(self.repository)

    def predict(self, context, input_data, params=None):
        from einops import rearrange
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        model = MoiraiForecast(
            module=self.module,
            prediction_length=self.prediction_length,
            context_length=len(input_data),
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        # Time series values. Shape: (batch, time, variate)
        past_target = rearrange(
            torch.as_tensor(input_data, dtype=torch.float32), "t -> 1 t 1"
        )
        # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
        forecast = model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        return np.median(forecast[0], axis=0)
