from typing import Dict, List
from sklearn.base import BaseEstimator, RegressorMixin
import torch as t
import numpy as np
import pandas as pd

from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.auto import (
    AutoRNN,
    AutoLSTM,
    AutoNBEATS,
    AutoNBEATSx,
)

from forecasting_sa.models.neuralforecast.NeuralForecastPipeline import (
    NeuralFcForecaster,
)

class NeuralFcAutoRNN(NeuralFcForecaster):
    def __init__(self, params):
        super().__init__(params)
        self.config = dict(max_steps=self.params.max_steps,
                           val_check_steps=self.params.val_check_steps,
                           input_size=self.params.input_size,
                           encoder_hidden_size=self.params.encoder_hidden_size)
        self.model = AutoRNN(h=int(self.params["prediction_length"]),
                             config=self.config,
                             search_alg=HyperOptSearch(),
                             num_samples=1,
                             cpus=-1)

