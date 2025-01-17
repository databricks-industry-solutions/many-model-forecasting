import pandas as pd
import numpy as np
import mlflow
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
    MeanAbsolutePercentageError,
)
from neuralforecast import NeuralForecast
from mmf_sa.models.abstract_model import ForecastingRegressor
from neuralforecast.auto import (
    RNN,
    LSTM,
    NBEATSx,
    NHITS,
    AutoRNN,
    AutoLSTM,
    AutoNBEATSx,
    AutoNHITS,
    AutoTiDE,
    AutoPatchTST,
)
from neuralforecast.losses.pytorch import (
    MAE, MSE, RMSE, MAPE, SMAPE, MASE,
)


class NeuralFcForecaster(ForecastingRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model = None

    def register(self, model, registered_model_name: str, input_example, run_id):
        pipeline = NeuralForecastModel(model)
        # Prepare model signature for model registry
        input_schema = infer_signature(model_input=input_example).inputs
        output_schema = Schema(
            [
                ColSpec("integer", "index"),
                ColSpec("string", self.params.group_id),
                ColSpec("datetime", self.params.date_col),
                ColSpec("float", self.params.target),
            ]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model=pipeline,
            registered_model_name=registered_model_name,
            #input_example=input_example,
            signature=signature,
            pip_requirements=[
                "cloudpickle==2.2.1",
                "neuralforecast==1.7.2",
                "git+https://github.com/databricks-industry-solutions/many-model-forecasting.git",
                "pyspark==3.5.0",
            ],
        )
        mlflow.log_params(model.get_params())
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("model_name", model.params["name"])
        print(f"Model registered: {registered_model_name}")
        return model_info

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
            df[self.params.target] = df[self.params.target].clip(0)
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
        if isinstance(self.model, NeuralForecast):
            pdf = self.prepare_data(x)
            static_pdf = self.prepare_static_features(x)
            self.model.fit(df=pdf, static_df=static_pdf)

    def predict(self, hist_df: pd.DataFrame, val_df: pd.DataFrame = None):
        _df = self.prepare_data(hist_df)
        _dynamic_future = self.prepare_data(val_df, future=True)
        _dynamic_future = None if _dynamic_future.empty else _dynamic_future
        _static_df = self.prepare_static_features(hist_df)
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
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0)

        return forecast_df, self.model

    def forecast(self, df: pd.DataFrame, spark=None):
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
        forecast_df[self.params.target] = forecast_df[self.params.target].clip(0)
        return forecast_df, self.model

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame, curr_date, spark=None
    ) -> list:
        pred_df, model_fitted = self.predict(hist_df, val_df)
        keys = pred_df[self.params["group_id"]].unique()
        metrics = []
        if self.params["metric"] == "smape":
            metric_name = "smape"
        elif self.params["metric"] == "mape":
            metric_name = "mape"
        elif self.params["metric"] == "mae":
            metric_name = "mae"
        elif self.params["metric"] == "mse":
            metric_name = "mse"
        elif self.params["metric"] == "rmse":
            metric_name = "rmse"
        else:
            raise Exception(f"Metric {self.params['metric']} not supported!")
        for key in keys:
            actual = val_df[val_df[self.params["group_id"]] == key][self.params["target"]]
            forecast = pred_df[pred_df[self.params["group_id"]] == key][self.params["target"]].\
                         iloc[-self.params["prediction_length"]:]
            try:
                if metric_name == "smape":
                    smape = MeanAbsolutePercentageError(symmetric=True)
                    metric_value = smape(actual, forecast)
                elif metric_name == "mape":
                    mape = MeanAbsolutePercentageError(symmetric=False)
                    metric_value = mape(actual, forecast)
                elif metric_name == "mae":
                    mae = MeanAbsoluteError()
                    metric_value = mae(actual, forecast)
                elif metric_name == "mse":
                    mse = MeanSquaredError(square_root=False)
                    metric_value = mse(actual, forecast)
                elif metric_name == "rmse":
                    rmse = MeanSquaredError(square_root=True)
                    metric_value = rmse(actual, forecast)
                metrics.extend(
                    [(
                        key,
                        curr_date,
                        metric_name,
                        metric_value,
                        forecast.to_numpy(),
                        actual.to_numpy(),
                        b'',
                    )])
            except:
                pass
        return metrics


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
                    scaler_type='robust',
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
                    accelerator=self.params.accelerator,
                    devices=self.devices,
                ),
            ],
            freq=self.params["freq"]
        )


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
                    scaler_type='robust',
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
                    accelerator=self.params.accelerator,
                    devices=self.devices,
                ),
            ],
            freq=self.params["freq"]
        )


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
                    scaler_type='robust',
                    batch_size=self.params.batch_size,
                    n_harmonics=self.params.n_harmonics,
                    n_polynomials=self.params.n_polynomials,
                    dropout_prob_theta=self.params.dropout_prob_theta,
                    stat_exog_list=list(self.params.get("static_features", [])),
                    futr_exog_list=list(self.params.get("dynamic_future", [])),
                    hist_exog_list=list(self.params.get("dynamic_historical", [])),
                    accelerator=self.params.accelerator,
                    devices=self.devices,
                ),
            ],
            freq=self.params["freq"]
        )


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
                    scaler_type='robust',
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
                    accelerator=self.params.accelerator,
                    devices=self.devices,
                ),
            ],
            freq=self.params["freq"]
        )


class NeuralFcAutoRNN(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.cpus = 0 if self.params.accelerator == 'gpu' else -1
        self.gpus = -1 if self.params.accelerator == 'gpu' else 0
        self.distributed_kwargs = dict(
            accelerator=self.params.accelerator,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        self.exogs = {
            'stat_exog_list': list(self.params.get("static_features", [])),
            'futr_exog_list': list(self.params.get("dynamic_future", [])),
            'hist_exog_list': list(self.params.get("dynamic_historical", [])),
        }

        def config(trial):
            return dict(
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                batch_size=trial.suggest_int("batch_size", 16, 32, step=8),
                max_steps=self.params.max_steps,
                scaler_type='robust',
                encoder_hidden_size=trial.suggest_categorical(
                    'encoder_hidden_size', list(self.params.encoder_hidden_size)),
                encoder_n_layers=trial.suggest_categorical(
                    'encoder_n_layers', list(self.params.encoder_n_layers)),
                context_size=trial.suggest_categorical(
                    'context_size', list(self.params.context_size)),
                decoder_hidden_size=trial.suggest_categorical(
                    'decoder_hidden_size', list(self.params.decoder_hidden_size)),
                **self.exogs,
                **self.distributed_kwargs,
            )

        self.model = NeuralForecast(
            models=[
                AutoRNN(
                    h=int(self.params["prediction_length"]),
                    loss=self.loss,
                    config=config,
                    backend='optuna',
                    cpus=self.cpus,
                    gpus=self.gpus,
                    num_samples=int(self.params["num_samples"]),
                ),
            ],
            freq=self.params["freq"]
        )


class NeuralFcAutoLSTM(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.cpus = 0 if self.params.accelerator == 'gpu' else -1
        self.gpus = -1 if self.params.accelerator == 'gpu' else 0
        self.distributed_kwargs = dict(
            accelerator=self.params.accelerator,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        self.exogs = {
            'stat_exog_list': list(self.params.get("static_features", [])),
            'futr_exog_list': list(self.params.get("dynamic_future", [])),
            'hist_exog_list': list(self.params.get("dynamic_historical", [])),
        }

        def config(trial):
            return dict(
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                batch_size=trial.suggest_int("batch_size", 16, 32, step=8),
                max_steps=self.params.max_steps,
                encoder_hidden_size=trial.suggest_categorical(
                    'encoder_hidden_size', list(self.params.encoder_hidden_size)),
                encoder_n_layers=trial.suggest_categorical(
                    'encoder_n_layers', list(self.params.encoder_n_layers)),
                context_size=trial.suggest_categorical(
                    'context_size', list(self.params.context_size)),
                decoder_hidden_size=trial.suggest_categorical(
                    'decoder_hidden_size', list(self.params.decoder_hidden_size)),
                **self.exogs,
                **self.distributed_kwargs,
            )

        self.model = NeuralForecast(
            models=[
                AutoLSTM(
                    h=int(self.params["prediction_length"]),
                    loss=self.loss,
                    config=config,
                    backend='optuna',
                    cpus=self.cpus,
                    gpus=self.gpus,
                    num_samples=int(self.params["num_samples"]),
                ),
            ],
            freq=self.params["freq"]
        )


class NeuralFcAutoNBEATSx(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.cpus = 0 if self.params.accelerator == 'gpu' else -1
        self.gpus = -1 if self.params.accelerator == 'gpu' else 0
        self.distributed_kwargs = dict(
            accelerator=self.params.accelerator,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        self.exogs = {
            'stat_exog_list': list(self.params.get("static_features", [])),
            'futr_exog_list': list(self.params.get("dynamic_future", [])),
            'hist_exog_list': list(self.params.get("dynamic_historical", [])),
        }

        def config(trial):
            return dict(
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                batch_size=trial.suggest_int("batch_size", 16, 32, step=8),
                max_steps=self.params.max_steps,
                input_size=self.params.input_size,
                scaler_type=trial.suggest_categorical(
                    'scaler_type', list(self.params.scaler_type)),
                **self.exogs,
                **self.distributed_kwargs,
            )

        self.model = NeuralForecast(
            models=[
                AutoNBEATSx(
                    h=int(self.params["prediction_length"]),
                    config=config,
                    loss=self.loss,
                    backend='optuna',
                    cpus=self.cpus,
                    gpus=self.gpus,
                    num_samples=int(self.params["num_samples"]),
                ),
            ],
            freq=self.params["freq"]
        )


class NeuralFcAutoNHITS(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.cpus = 0 if self.params.accelerator == 'gpu' else -1
        self.gpus = -1 if self.params.accelerator == 'gpu' else 0
        self.distributed_kwargs = dict(
            accelerator=self.params.accelerator,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        self.exogs = {
            'stat_exog_list': list(self.params.get("static_features", [])),
            'futr_exog_list': list(self.params.get("dynamic_future", [])),
            'hist_exog_list': list(self.params.get("dynamic_historical", [])),
        }

        def config(trial):
            return dict(
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                batch_size=trial.suggest_int("batch_size", 16, 32, step=8),
                max_steps=self.params.max_steps,
                input_size=self.params.input_size,
                n_pool_kernel_size=trial.suggest_categorical(
                    'n_pool_kernel_size', list(self.params.n_pool_kernel_size)),
                n_freq_downsample=trial.suggest_categorical(
                    'n_freq_downsample', list(self.params.n_freq_downsample)),
                scaler_type=trial.suggest_categorical(
                    'scaler_type', list(self.params.scaler_type)),
                **self.exogs,
                **self.distributed_kwargs,
            )
        self.model = NeuralForecast(
            models=[
                AutoNHITS(
                    h=int(self.params["prediction_length"]),
                    config=config,
                    loss=self.loss,
                    backend='optuna',
                    cpus=self.cpus,
                    gpus=self.gpus,
                    num_samples=int(self.params["num_samples"]),
                ),
            ],
            freq=self.params["freq"]
        )


class NeuralFcAutoTiDE(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.cpus = 0 if self.params.accelerator == 'gpu' else -1
        self.gpus = -1 if self.params.accelerator == 'gpu' else 0
        self.distributed_kwargs = dict(
            accelerator=self.params.accelerator,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        self.exogs = {
            'stat_exog_list': list(self.params.get("static_features", [])),
            'futr_exog_list': list(self.params.get("dynamic_future", [])),
            'hist_exog_list': list(self.params.get("dynamic_historical", [])),
        }

        def config(trial):
            return dict(
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                batch_size=trial.suggest_int("batch_size", 16, 32, step=8),
                max_steps=self.params.max_steps,
                scaler_type='robust',
                input_size=self.params.input_size,
                hidden_size=trial.suggest_categorical(
                    'hidden_size', list(self.params.hidden_size)),
                decoder_output_dim=trial.suggest_categorical(
                    'decoder_output_dim', list(self.params.decoder_output_dim)),
                temporal_decoder_dim=trial.suggest_categorical(
                    'temporal_decoder_dim', list(self.params.temporal_decoder_dim)),
                num_encoder_layers=trial.suggest_categorical(
                    'num_encoder_layers', list(self.params.num_encoder_layers)),
                num_decoder_layers=trial.suggest_categorical(
                    'num_decoder_layers', list(self.params.num_decoder_layers)),
                temporal_width=trial.suggest_categorical(
                    'temporal_width', list(self.params.temporal_width)),
                dropout=trial.suggest_categorical(
                    'dropout', list(self.params.dropout)),
                layernorm=trial.suggest_categorical(
                    'layernorm', list(self.params.layernorm)),
                #**self.exogs, #exogenous regressors not yet supported
                **self.distributed_kwargs,
            )
        self.model = NeuralForecast(
            models=[
                AutoTiDE(
                    h=int(self.params["prediction_length"]),
                    config=config,
                    loss=self.loss,
                    backend='optuna',
                    cpus=self.cpus,
                    gpus=self.gpus,
                    num_samples=int(self.params["num_samples"]),
                ),
            ],
            freq=self.params["freq"]
        )


class NeuralFcAutoPatchTST(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.loss = get_loss_function(self.params.loss)
        self.cpus = 0 if self.params.accelerator == 'gpu' else -1
        self.gpus = -1 if self.params.accelerator == 'gpu' else 0
        self.distributed_kwargs = dict(
            accelerator=self.params.accelerator,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        self.exogs = {
            'stat_exog_list': list(self.params.get("static_features", [])),
            'futr_exog_list': list(self.params.get("dynamic_future", [])),
            'hist_exog_list': list(self.params.get("dynamic_historical", [])),
        }

        def config(trial):
            return dict(
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                batch_size=trial.suggest_int("batch_size", 16, 32, step=8),
                max_steps=self.params.max_steps,
                input_size=self.params.input_size,
                hidden_size=trial.suggest_categorical(
                    'hidden_size', list(self.params.hidden_size)),
                n_heads=trial.suggest_categorical(
                    'n_heads', list(self.params.n_heads)),
                patch_len=trial.suggest_categorical(
                    'patch_len', list(self.params.patch_len)),
                scaler_type=trial.suggest_categorical(
                    'scaler_type', list(self.params.scaler_type)),
                revin=trial.suggest_categorical(
                    'revin', list(self.params.revin)),
                #**self.exogs, #exogenous regressors not yet supported
                **self.distributed_kwargs,
            )
        self.model = NeuralForecast(
            models=[
                AutoPatchTST(
                    h=int(self.params["prediction_length"]),
                    config=config,
                    loss=self.loss,
                    backend='optuna',
                    cpus=self.cpus,
                    gpus=self.gpus,
                    num_samples=int(self.params["num_samples"]),
                ),
            ],
            freq=self.params["freq"]
        )


class NeuralForecastModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, input_data, params=None):
        forecast, model = self.pipeline.forecast(input_data)
        return forecast
