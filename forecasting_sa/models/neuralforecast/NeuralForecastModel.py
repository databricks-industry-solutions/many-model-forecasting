from typing import Dict, List
from sklearn.base import BaseEstimator, RegressorMixin
import torch as t
import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.auto import (
    AutoRNN,
    AutoLSTM,
    AutoNBEATS,
    AutoNBEATSx,
)
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
from neuralforecast.losses.pytorch import (
    MAE, MSE, RMSE, MAPE, SMAPE, MASE,
)

from forecasting_sa.models.neuralforecast.NeuralForecastPipeline import (
    NeuralFcForecaster,
)


class NeuralFcAutoRNN(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        if self.params.loss == "smape":
            self.loss = SMAPE()
        elif self.params.loss == "mae":
            self.loss = MAE()
        elif self.params.loss == "mse":
            self.loss = MSE()
        elif self.params.loss == "rmse":
            self.loss = RMSE()
        elif self.params.loss == "mape":
            self.loss = MAPE()
        elif self.params.loss == "mase":
            self.loss = MASE()
        else:
            raise Exception(
                f"Provided loss {self.params.loss} not supported!"
            )
        self.config = dict(
            encoder_n_layers=self.params.encoder_n_layers,
            encoder_hidden_size=self.params.encoder_hidden_size,
            encoder_activation=self.params.encoder_activation,
            context_size=self.params.context_size,
            decoder_hidden_size=self.params.decoder_hidden_size,
            decoder_layers=self.params.decoder_layers,

            max_steps=self.params.max_steps,
            learning_rate=0.001,
            batch_size=32,

            #learning_rate=tune.loguniform(1e-5, 1e-1),
            #batch_size=tune.quniform(32, 256, 8),
        )
        self.model = NeuralForecast(
            models=[
                AutoRNN(
                    h=int(self.params["prediction_length"]),
                    loss=self.loss,
                    config=self.config,
                    #search_alg=HyperOptSearch(),
                    #num_samples=int(self.params["num_samples"]),
                    stat_exog_list=self.params["static_features"],
                    futr_exog_list=self.params["dynamic_future"],
                    hist_exog_list=self.params["dynamic_historical"],
                    scaler_type='robust',
                ),
            ],
            freq=self.params["freq"]
        )

        self._search_space = {
            "epochs": hp.quniform(
                "epochs", 1, int(self.params["tuning_max_epochs"]), 1
            ),
            "context_length": hp.quniform(
                "context_length", 20, int(self.params["tuning_max_context_len"]), 5
            ),
            "batch_size": hp.quniform("batch_size", 32, 256, 8),
            "num_cells": hp.quniform("num_cells", 32, 192, 8),
            "num_layers": hp.quniform("num_layers", 1, 8, 1),
            "dropout_rate": hp.uniform("dropout_rate", 0.000001, 0.5),
        }

    def search_space(self):
        return self._search_space

    def supports_tuning(self) -> bool:
        return True

