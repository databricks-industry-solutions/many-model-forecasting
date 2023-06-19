from neuralforecast import NeuralForecast
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

from forecasting_sa.models.neuralforecast.NeuralForecastPipeline import (
    NeuralFcForecaster,
)


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
        self.model = NeuralForecast(
            models=[
                AutoNBEATSx(
                    h=int(self.params["prediction_length"]),
                    loss=self.loss,
                    config=self.config,
                    gpus=self.gpus,
                    search_alg=HyperOptSearch(),
                    num_samples=int(self.params["num_samples"]),
                ),
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
                    gpus=self.gpus,
                    search_alg=HyperOptSearch(),
                    num_samples=int(self.params["num_samples"]),
                ),
            ],
            freq=self.params["freq"]
        )

    def supports_tuning(self) -> bool:
        return True
