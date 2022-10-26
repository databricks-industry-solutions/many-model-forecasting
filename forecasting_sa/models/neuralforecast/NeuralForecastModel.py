from typing import Dict, List
from sklearn.base import BaseEstimator, RegressorMixin
import torch as t
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from neuralforecast.data.tsdataset import WindowsDataset
from neuralforecast.data.tsloader import TimeSeriesLoader
from neuralforecast.experiments.utils import get_mask_dfs
from neuralforecast.models.mqnhits.mqnhits import MQNHITS
from neuralforecast.losses.numpy import mqloss


class NeuralForecastRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.predictor = None
        self.model = MQNHITS(
            n_time_in=3 * self.params.prediction_length,  # number of input lags
            n_time_out=self.params.prediction_length,  # forecasting horizon
            quantiles=[5, 50, 95],  # quantiles for MQ-LOSS
            shared_weights=False,  # shared parameters between blocks in each stack
            initialization="lecun_normal",  # initialization
            activation="ReLU",  # activation function
            stack_types=3
            * ["identity"],  # list of stack types (only 'identity' type its supported)
            n_blocks=3 * [1],  # number of blocks in each stack
            n_layers=3 * [2],  # number of layers in MLP of each block
            n_mlp_units=3 * [2 * [256]],  # number of units (nodes) in each layer
            n_pool_kernel_size=3
            * [1],  # Pooling kernel size for input downsampling for each stack
            n_freq_downsample=[12, 4, 1],
            # Inverse of expresivity ratio. Output size of stack i is (H/n_freq_downsample_i)
            pooling_mode="max",  # Pooling mode
            interpolation_mode="linear",  # Interpolation mode
            batch_normalization=False,  # Batch normalization in MLP
            dropout_prob_theta=0,  # Dropout probability
            learning_rate=0.0001,
            lr_decay=1.0,
            lr_decay_step_size=100_000,
            weight_decay=0.0,
            loss_train="MQ",
            loss_valid="MQ",
            frequency=self.freq,
            n_x=0,
            n_s=0,
            n_x_hidden=0,
            n_s_hidden=0,
            loss_hypar=0.5,
            random_seed=1,
        )

    def prepare_data(self, df):
        return df.rename(
            columns={
                self.params.group_id: "unique_id",
                self.params.date_col: "ds",
                self.params.target: "y",
            }
        )

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        _df = self.prepare_data(X)
        train_mask_df, valid_mask_df, _ = get_mask_dfs(
            Y_df=_df, ds_in_val=self.params["prediction_length"], ds_in_test=0
        )
        train_dataset = WindowsDataset(
            Y_df=_df,
            X_df=None,
            S_df=None,
            mask_df=train_mask_df,
            f_cols=[],
            input_size=3 * self.params["prediction_length"],
            output_size=self.params["prediction_length"],
            sample_freq=1,
            complete_windows=True,
            verbose=False,
        )

        valid_dataset = WindowsDataset(
            Y_df=_df,
            X_df=None,
            S_df=None,
            mask_df=valid_mask_df,
            f_cols=[],
            input_size=3 * self.params["prediction_length"],
            output_size=self.params["prediction_length"],
            sample_freq=1,
            complete_windows=True,
            verbose=False,
        )

        train_loader = TimeSeriesLoader(
            dataset=train_dataset, batch_size=32, n_windows=1024, shuffle=True
        )

        valid_loader = TimeSeriesLoader(
            dataset=valid_dataset, batch_size=1, shuffle=False
        )

        gpus = -1 if t.cuda.is_available() else 0
        trainer = pl.Trainer(
            max_epochs=None,
            max_steps=5000,
            gradient_clip_val=1.0,
            check_val_every_n_epoch=50,
            gpus=gpus,
            log_every_n_steps=1,
        )

        trainer.fit(self.model, train_loader, valid_loader)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        _df = self.prepare_data(X)
        pred_df = self.model.forecast(_df)
        return pred_df.rename(
            columns={
                "unique_id": self.params.group_id,
                "ds": self.params.date_col,
                "y_50": self.params.target,
            }
        )

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
