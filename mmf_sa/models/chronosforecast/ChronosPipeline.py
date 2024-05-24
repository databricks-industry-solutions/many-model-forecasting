from abc import ABC
import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from typing import Iterator
from pyspark.sql.functions import collect_list, pandas_udf
from pyspark.sql import DataFrame
import mlflow
from mmf_sa.models.abstract_model import ForecastingRegressor
mlflow.set_registry_uri("databricks-uc")


class ChronosForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = None
        self.model = None

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
        forecast_udf = create_predict_udf(
            prediction_length=self.params["prediction_length"],
            num_samples=self.params["num_samples"]
        )

        horizon_timestamps_udf = self.create_horizon_timestamps_udf()

        # Todo figure out the distribution
        forecast_df = (
            hist_df.repartition(4)
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


class ChronosT5Tiny(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map=self.device,  # use "cuda" for GPU and "cpu" for CPU inference
            torch_dtype=torch.bfloat16,
        )


class ChronosT5Mini(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-mini",
            device_map=self.device,  # use "cuda" for GPU and "cpu" for CPU inference
            torch_dtype=torch.bfloat16,
        )


class ChronosT5Small(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map=self.device,  # use "cuda" for GPU and "cpu" for CPU inference
            torch_dtype=torch.bfloat16,
        )


class ChronosT5Base(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map=self.device,  # use "cuda" for GPU and "cpu" for CPU inference
            torch_dtype=torch.bfloat16,
        )


class ChronosT5Large(ChronosForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map=self.device,  # use "cuda" for GPU and "cpu" for CPU inference
            torch_dtype=torch.bfloat16,
        )


def create_predict_udf(prediction_length: int, num_samples: int):

    @pandas_udf('array<double>')
    def predict_udf(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        # initialization step
        import torch
        import numpy as np
        import pandas as pd
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        # inference
        for batch in batch_iterator:
            median = []
            for series in batch:
                context = torch.tensor(list(series))
                forecast = pipeline.predict(
                    context=context,
                    prediction_length=prediction_length,
                    num_samples=num_samples
                )
                median.append(np.quantile(forecast[0].numpy(), [0.5], axis=0)[0])
        yield pd.Series(median)

    return predict_udf
