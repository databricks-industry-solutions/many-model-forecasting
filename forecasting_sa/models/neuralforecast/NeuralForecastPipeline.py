import pandas as pd
import numpy as np
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from neuralforecast import NeuralForecast, DistributedConfig
from forecasting_sa.models.abstract_model import ForecastingRegressor
from neuralforecast.auto import (
    RNN,
    LSTM,
    NBEATSx,
    NHITS,
    AutoRNN,
    AutoLSTM,
    AutoNBEATSx,
    AutoNHITS,

)
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.losses.pytorch import (
    MAE, MSE, RMSE, MAPE, SMAPE, MASE,
)


class NeuralFcForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model_spec = None
        self.model = None

    def prepare_data(self, df: pd.DataFrame, future: bool = False) -> pd.DataFrame:
        if future:
            # Prepare future dataframe with exogenous regressors for forecasting
            if 'dynamic_future' in self.params.keys():
                try:
                    _df = (
                        df[[self.params.group_id, self.params.date_col]
                           + self.params.dynamic_future]
                    )
                except Exception as e:
                    raise Exception(f"Dynamic future regressors missing: {e}")
            else:
                _df = df[[self.params.group_id, self.params.date_col]]
            _df = (
                _df.rename(
                    columns={
                        self.params.group_id: "unique_id",
                        self.params.date_col: "ds",
                    }
                )
            )
        else:
            # Prepare historical dataframe with or without exogenous regressors for training
            df[self.params.target] = df[self.params.target].clip(0.1)
            if 'dynamic_future' in self.params.keys():
                try:
                    _df = (
                        df[[self.params.group_id, self.params.date_col, self.params.target]
                           + self.params.dynamic_future]
                    )
                except Exception as e:
                    raise Exception(f"Dynamic future regressor columns missing from "
                                    f"the training dataset: {e}")
            elif 'dynamic_historical' in self.params.keys():
                try:
                    _df = (
                        df[[self.params.group_id, self.params.date_col, self.params.target]
                           + self.params.dynamic_historical]
                    )
                except Exception as e:
                    raise Exception(f"Dynamic historical regressor columns missing from "
                                    f"the training dataset: {e}")
            else:
                _df = df[[self.params.group_id, self.params.date_col, self.params.target]]

            _df = (
                _df.rename(
                    columns={
                        self.params.group_id: "unique_id",
                        self.params.date_col: "ds",
                        self.params.target: "y",
                    }
                )
            )
        return _df

    def prepare_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'static_features' in self.params.keys():
            static_features = df[[self.params.group_id] + self.params.static_features].drop_duplicates()
            static_features = (
                static_features.rename(
                    columns={
                        self.params.group_id: "unique_id",
                    }
                )
            )
            # categorical encoding of columns that are in object
            for col in static_features.columns:
                if (col != "unique_id") and (static_features[col].dtype == object):
                    static_features[col] = static_features[col].astype('category')
                    static_features[col] = static_features[col].cat.codes
            return static_features
        else:
            return None

    def fit(self, x, y=None):
        #self.spark = spark
        if isinstance(self.model, NeuralForecast):
            pdf = self.prepare_data(x)
            static_pdf = self.prepare_static_features(x)

            #self.model.fit(
            #    df=self.spark.createDataFrame(pdf),
            #    static_df=self.spark.createDataFrame(static_pdf) if static_pdf else None,
            #    distributed_config=self.dist_cfg
            #    )

            self.model.fit(df=pdf, static_df=static_pdf)

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        _df = self.prepare_data(hist_df)
        _dynamic_future = self.prepare_data(val_df, future=True)
        _dynamic_future = None if _dynamic_future.empty else _dynamic_future
        _static_df = self.prepare_static_features(hist_df)

        #forecast_df = self.model.predict(
        #    df=self.spark.createDataFrame(_df),
        #    static_df=self.spark.createDataFrame(_static_df) if _static_df else None,
        #    futr_df=self.spark.createDataFrame(_dynamic_future) if _dynamic_future else None
        #).toPandas()

        forecast_df = self.model.predict(df=_df, static_df=_static_df, futr_df=_dynamic_future)
        target = [col for col in forecast_df.columns.to_list()
                  if col not in ["unique_id", "ds"]][0]
        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                target: self.params.target,
            }
        )
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)

        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame):
        _df = df[df[self.params.target].notnull()]
        _df = self.prepare_data(_df)
        _last_date = _df["ds"].max()
        _future_df = df[
            (df[self.params["date_col"]] > np.datetime64(_last_date))
            & (df[self.params["date_col"]]
               <= np.datetime64(_last_date + self.prediction_length_offset))
        ]

        _dynamic_future = self.prepare_data(_future_df, future=True)
        _dynamic_future = None if _dynamic_future.empty else _dynamic_future
        _static_df = self.prepare_static_features(_future_df)

        # Check if dynamic futures for all unique_id are provided.
        # If not, drop unique_id without dynamic futures from scoring.
        if (_dynamic_future is not None) and \
                (not set(_df["unique_id"].unique().flatten())
                        .issubset(set(_dynamic_future["unique_id"].unique().flatten()))):
            _df = _df[_df["unique_id"].isin(list(_dynamic_future["unique_id"].unique()))]

        #forecast_df = self.model.predict(
        #    df=self.spark.createDataFrame(_df),
        #    static_df=self.spark.createDataFrame(_static_df),
        #    futr_df=self.spark.createDataFrame(_dynamic_future)
        #).toPandas()

        forecast_df = self.model.predict(df=_df, static_df=_static_df, futr_df=_dynamic_future)

        target = [col for col in forecast_df.columns.to_list()
                  if col not in ["unique_id", "ds"]][0]
        forecast_df = forecast_df.reset_index(drop=False).rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                target: self.params.target,
            }
        )
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0.01)
        return forecast_df, self.model

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date
    ) -> list:
        pred_df, model_fitted = self.predict(hist_df, val_df)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        if self.params["metric"] == "smape":
            metric_name = "smape"
        else:
            raise Exception(f"Metric {self.params['metric']} not supported!")
        for key in keys:
            actual = val_df[val_df[self.params["group_id"]] == key][self.params["target"]]
            forecast = pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].\
                         iloc[-self.params["prediction_length"]:]
            try:
                if metric_name == "smape":
                    metric_value = mean_absolute_percentage_error(actual, forecast, symmetric=True)
                metrics.extend(
                    [(
                        key,
                        curr_date,
                        metric_name,
                        metric_value,
                        actual.to_numpy(),
                        forecast.to_numpy(),
                        b'',
                    )])
            except:
                pass
        return metrics


class NeuralFcRNN(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.accelerator = 'gpu' if self.params.accelerator == 'gpu' else 'cpu'
        self.devices = -1 if self.params.accelerator == 'gpu' else 1
        self.model = NeuralForecast(
            models=[
                RNN(
                    h=self.params.prediction_length,
                    input_size=self.params.input_size_factor*self.params.prediction_length,
                    loss=self.loss,
                    max_steps=self.params.max_steps,
                    batch_size=self.params.batch_size,
                    encoder_n_layers=self.params.encoder_n_layers,
                    encoder_hidden_size=self.params.encoder_hidden_size,
                    encoder_activation=self.params.encoder_activation,
                    context_size=self.params.context_size,
                    decoder_hidden_size=self.params.decoder_hidden_size,
                    decoder_layers=self.params.decoder_layers,
                    learning_rate=self.params.learning_rate,
                    stat_exog_list=list(self.params.get("static_features", [])),
                    futr_exog_list=list(self.params.get("dynamic_future", [])),
                    hist_exog_list=list(self.params.get("dynamic_historical", [])),
                    scaler_type='robust',
                    accelerator=self.params.accelerator,
                    devices=self.devices,
                ),
            ],
            freq=self.params["freq"]
        )

    def supports_tuning(self) -> bool:
        return False


class NeuralFcLSTM(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.accelerator = 'gpu' if self.params.accelerator == 'gpu' else 'cpu'
        self.devices = -1 if self.params.accelerator == 'gpu' else 1
        self.model = NeuralForecast(
            models=[
                LSTM(
                    h=self.params.prediction_length,
                    input_size=self.params.input_size_factor*self.params.prediction_length,
                    loss=self.loss,
                    max_steps=self.params.max_steps,
                    batch_size=self.params.batch_size,
                    encoder_n_layers=self.params.encoder_n_layers,
                    encoder_hidden_size=self.params.encoder_hidden_size,
                    context_size=self.params.context_size,
                    decoder_hidden_size=self.params.decoder_hidden_size,
                    decoder_layers=self.params.decoder_layers,
                    learning_rate=self.params.learning_rate,
                    stat_exog_list=list(self.params.get("static_features", [])),
                    futr_exog_list=list(self.params.get("dynamic_future", [])),
                    hist_exog_list=list(self.params.get("dynamic_historical", [])),
                    scaler_type='robust',
                    accelerator=self.params.accelerator,
                    devices=self.devices,
                ),
            ],
            freq=self.params["freq"]
        )

    def supports_tuning(self) -> bool:
        return False


class NeuralFcNBEATSx(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.accelerator = 'gpu' if self.params.accelerator == 'gpu' else 'cpu'
        self.devices = -1 if self.params.accelerator == 'gpu' else 1
        self.model = NeuralForecast(
            models=[
                NBEATSx(
                    h=self.params.prediction_length,
                    input_size=self.params.input_size_factor*self.params.prediction_length,
                    loss=self.loss,
                    max_steps=self.params.max_steps,
                    batch_size=self.params.batch_size,
                    n_harmonics=self.params.n_harmonics,
                    n_polynomials=self.params.n_polynomials,
                    dropout_prob_theta=self.params.dropout_prob_theta,
                    stat_exog_list=list(self.params.get("static_features", [])),
                    futr_exog_list=list(self.params.get("dynamic_future", [])),
                    hist_exog_list=list(self.params.get("dynamic_historical", [])),
                    scaler_type='robust',
                    accelerator=self.params.accelerator,
                    devices=self.devices,
                ),
            ],
            freq=self.params["freq"]
        )

    def supports_tuning(self) -> bool:
        return False


class NeuralFcNHITS(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.accelerator = 'gpu' if self.params.accelerator == 'gpu' else 'cpu'
        self.devices = -1 if self.params.accelerator == 'gpu' else 1
        self.model = NeuralForecast(
            models=[
                NHITS(
                    h=self.params.prediction_length,
                    input_size=self.params.input_size_factor*self.params.prediction_length,
                    loss=self.loss,
                    max_steps=self.params.max_steps,
                    batch_size=self.params.batch_size,
                    dropout_prob_theta=self.params.dropout_prob_theta,
                    stack_types=list(self.params.stack_types),
                    n_blocks=list(self.params.n_blocks),
                    n_pool_kernel_size=list(self.params.n_pool_kernel_size),
                    n_freq_downsample=list(self.params.n_freq_downsample),
                    interpolation_mode=self.params.interpolation_mode,
                    pooling_mode=self.params.pooling_mode,
                    stat_exog_list=list(self.params.get("static_features", [])),
                    futr_exog_list=list(self.params.get("dynamic_future", [])),
                    hist_exog_list=list(self.params.get("dynamic_historical", [])),
                    scaler_type='robust',
                    accelerator=self.params.accelerator,
                    devices=self.devices,
                ),
            ],
            freq=self.params["freq"]
        )

    def supports_tuning(self) -> bool:
        return False


class NeuralFcAutoRNN(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.cpus = 0 if self.params.accelerator == 'gpu' else -1
        self.gpus = -1 if self.params.accelerator == 'gpu' else 0
        self.config = dict(
            encoder_n_layers=self.params.encoder_n_layers,
            encoder_hidden_size=self.params.encoder_hidden_size,
            encoder_activation=self.params.encoder_activation,
            context_size=self.params.context_size,
            decoder_hidden_size=self.params.decoder_hidden_size,
            decoder_layers=self.params.decoder_layers,
            max_steps=self.params.max_steps,
            stat_exog_list=list(self.params.get("static_features", [])),
            futr_exog_list=list(self.params.get("dynamic_future", [])),
            hist_exog_list=list(self.params.get("dynamic_historical", [])),
            scaler_type='robust',
            learning_rate=tune.loguniform(1e-5, 1e-1),
            batch_size=tune.choice([16, 32]),

        )
        self.model = NeuralForecast(
            models=[
                AutoRNN(
                    h=int(self.params["prediction_length"]),
                    loss=self.loss,
                    config=self.config,
                    #cpus=self.cpus,
                    gpus=self.gpus,
                    search_alg=HyperOptSearch(),
                    num_samples=int(self.params["num_samples"]),
                ),
            ],
            freq=self.params["freq"]
        )

    def supports_tuning(self) -> bool:
        return True


class NeuralFcAutoLSTM(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.cpus = 0 if self.params.accelerator == 'gpu' else -1
        self.gpus = -1 if self.params.accelerator == 'gpu' else 0
        self.config = dict(
            encoder_n_layers=self.params.encoder_n_layers,
            encoder_hidden_size=self.params.encoder_hidden_size,
            encoder_activation=self.params.encoder_activation,
            context_size=self.params.context_size,
            decoder_hidden_size=self.params.decoder_hidden_size,
            decoder_layers=self.params.decoder_layers,
            max_steps=self.params.max_steps,
            stat_exog_list=list(self.params.get("static_features", [])),
            futr_exog_list=list(self.params.get("dynamic_future", [])),
            hist_exog_list=list(self.params.get("dynamic_historical", [])),
            scaler_type='robust',
            learning_rate=tune.loguniform(1e-5, 1e-1),
            batch_size=tune.choice([16, 32]),
        )
        self.model = NeuralForecast(
            models=[
                AutoLSTM(
                    h=int(self.params["prediction_length"]),
                    loss=self.loss,
                    config=self.config,
                    #cpus=self.cpus,
                    gpus=self.gpus,
                    search_alg=HyperOptSearch(),
                    num_samples=int(self.params["num_samples"]),
                ),
            ],
            freq=self.params["freq"]
        )

    def supports_tuning(self) -> bool:
        return True


class NeuralFcAutoNBEATSx(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.cpus = 0 if self.params.accelerator == 'gpu' else -1
        self.gpus = -1 if self.params.accelerator == 'gpu' else 0
        self.config = dict(
            input_size=self.params.input_size_factor * self.params.prediction_length,
            n_harmonics=self.params.n_harmonics,
            n_polynomials=self.params.n_polynomials,
            dropout_prob_theta=self.params.dropout_prob_theta,
            max_steps=self.params.max_steps,
            stat_exog_list=list(self.params.get("static_features", [])),
            futr_exog_list=list(self.params.get("dynamic_future", [])),
            hist_exog_list=list(self.params.get("dynamic_historical", [])),
            scaler_type='robust',
            learning_rate=tune.loguniform(1e-5, 1e-1),
            batch_size=tune.choice([16, 32]),
        )
        #self.dist_cfg = DistributedConfig(
        #    partitions_path=f'{self.params.get("temp_path")}',  # path where the partitions will be saved
        #    num_nodes=1,  # number of nodes to use during training (machines)
        #    devices=4,  # number of GPUs in each machine
        #)
        # pytorch lightning configuration
        # the executors don't have permission to write on the filesystem, so we disable saving artifacts
        #self.distributed_kwargs = dict(
        #    accelerator='gpu',
        #    enable_progress_bar=False,
        #    logger=False,
        #    enable_checkpointing=False,
        #)
        # exogenous features
        #self.exogs = {
        #    'futr_exog_list': list(self.params.get("dynamic_future", [])),
        #    'stat_exog_list': list(self.params.get("static_features", [])),
        #}
        #def config(trial):
        #    return dict(
        #        input_size=self.params.input_size_factor * self.params.prediction_length,
        #        max_steps=self.params.max_steps,
        #        learning_rate=tune.loguniform(1e-5, 1e-1),
        #        **self.exogs,
        #        **self.distributed_kwargs,
        #    )

        self.model = NeuralForecast(
            models=[
                AutoNBEATSx(
                    h=int(self.params["prediction_length"]),
                    loss=self.loss,
                    config=self.config,
                    #cpus=self.cpus,
                    gpus=self.gpus,
                    search_alg=HyperOptSearch(),
                    num_samples=int(self.params["num_samples"]),
                ),
                #AutoNBEATSx(
                #    h=int(self.params["prediction_length"]),
                #    loss=self.loss,
                #    config=config,
                #    backend='optuna',
                #    num_samples=int(self.params["num_samples"]),
                #),
            ],
            freq=self.params["freq"]
        )

    def supports_tuning(self) -> bool:
        return True


class NeuralFcAutoNHITS(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.cpus = 0 if self.params.accelerator == 'gpu' else -1
        self.gpus = -1 if self.params.accelerator == 'gpu' else 0
        self.config = dict(
            input_size=self.params.input_size_factor * self.params.prediction_length,
            dropout_prob_theta=self.params.dropout_prob_theta,
            stack_types=list(self.params.stack_types),
            n_blocks=list(self.params.n_blocks),
            n_pool_kernel_size=list(self.params.n_pool_kernel_size),
            n_freq_downsample=list(self.params.n_freq_downsample),
            interpolation_mode=self.params.interpolation_mode,
            pooling_mode=self.params.pooling_mode,
            max_steps=self.params.max_steps,
            stat_exog_list=list(self.params.get("static_features", [])),
            futr_exog_list=list(self.params.get("dynamic_future", [])),
            hist_exog_list=list(self.params.get("dynamic_historical", [])),
            scaler_type='robust',
            learning_rate=tune.loguniform(1e-5, 1e-1),
            batch_size=tune.choice([16, 32]),
        )
        self.model = NeuralForecast(
            models=[
                AutoNHITS(
                    h=int(self.params["prediction_length"]),
                    loss=self.loss,
                    config=self.config,
                    #cpus=self.cpus,
                    gpus=self.gpus,
                    search_alg=HyperOptSearch(),
                    num_samples=int(self.params["num_samples"]),
                ),
            ],
            freq=self.params["freq"]
        )

    def supports_tuning(self) -> bool:
        return True


def get_loss_function(loss):
    if loss == "smape":
        return SMAPE()
    elif loss == "mae":
        return MAE()
    elif loss == "mse":
        return MSE()
    elif loss == "rmse":
        return RMSE()
    elif loss == "mape":
        return MAPE()
    elif loss == "mase":
        return MASE()
    else:
        raise Exception(
            f"Provided loss {loss} not supported!"
        )
