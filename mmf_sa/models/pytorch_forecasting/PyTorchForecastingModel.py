import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


class PyTorchForecastingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, params):
        self.params = params

    def create_training_dataset(self, X, y=None):
        return TimeSeriesDataSet(
            X,
            time_idx="time_idx",
            target=self.params["target"],
            group_ids=self.params["static_categoricals"],
            min_encoder_length=self.params["min_encoder_length"],
            max_encoder_length=self.params["max_encoder_length"],
            min_prediction_length=self.params["min_prediction_length"],
            max_prediction_length=self.params["max_prediction_length"],
            static_categoricals=self.params["static_categoricals"],
            static_reals=self.params["static_reals"],
            time_varying_known_categoricals=self.params[
                "time_varying_known_categoricals"
            ],
            time_varying_known_reals=self.params["time_varying_known_reals"],
            time_varying_unknown_categoricals=self.params[
                "time_varying_unknown_categoricals"
            ],
            time_varying_unknown_reals=self.params["time_varying_unknown_reals"],
            target_normalizer=GroupNormalizer(
                groups=self.params["static_categoricals"],
                method=self.params["target_normalizer_method"],
                transformation=self.params["target_normalizer_transformation"],
                center=self.params["target_normalizer_center"],
            ),  # use softplus with beta=1.0 and normalize by group
            add_relative_time_idx=self.params["add_relative_time_idx"],
            add_target_scales=self.params["add_target_scales"],
            add_encoder_length=self.params["add_encoder_length"],
            allow_missing_timesteps=self.params["allow_missing_timesteps"],
            # categorical_encoders={"weekofyear": NaNLabelEncoder(add_nan=True).fit(df.ts_key)},
        )

    def create_validation_dataset(self, X, training_dataset, y=None):
        return TimeSeriesDataSet.from_dataset(
            training_dataset, X, predict=True, stop_randomization=True
        )

    def train_validation_split(self, X, y=None):
        _train = X[
            X.Date <= X.Date.max() - pd.DateOffset(months=self.params["val_month"])
        ]
        print(_train.columns)
        training_dataset = self.create_training_dataset(_train)
        validation_dataset = self.create_validation_dataset(X, training_dataset)
        return training_dataset, validation_dataset

    def create_model(self, training_dataset):
        callbacks_list = []
        logger = TensorBoardLogger(
            "/tmp/lightning_logs"
        )  # logging results to a tensorboard

        if self.params["early_stopping"]:
            early_stop_callback = EarlyStopping(
                monitor=self.params["early_stop_monitor"],
                min_delta=self.params["early_stop_min_delta"],
                patience=self.params["early_stop_patience"],
                verbose=True,
                mode=self.params["early_stop_mode"],
            )
            lr_logger = LearningRateMonitor()  # log the learning rate
            callbacks_list = [lr_logger, early_stop_callback]

        trainer = pl.Trainer(
            max_epochs=self.params["max_epochs"],
            gpus=self.params["gpus"],
            weights_summary=self.params["weights_summary"],
            gradient_clip_val=self.params["gradient_clip_val"],
            # limit_train_batches=self.params['limit_train_batches'],
            fast_dev_run=self.params["fast_dev_run"],
            callbacks=callbacks_list,
            logger=logger,
        )

        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self.params["learning_rate"],
            hidden_size=self.params["hidden_size"],
            attention_head_size=self.params["attention_head_size"],
            dropout=self.params["dropout"],  # between 0.1 and 0.3 are good values
            hidden_continuous_size=self.params[
                "hidden_continuous_size"
            ],  # set to <= hidden_size
            output_size=self.params["output_size"],
            loss=QuantileLoss(),
            reduce_on_plateau_patience=self.params["reduce_on_plateau_patience"],
        )
        return trainer, tft

    def prepare_data(self, df):
        df[self.params["date_col"]] = pd.to_datetime(df[self.params["date_col"]])
        df = df.sort_values(
            [self.params["group_id"], self.params["date_col"]], ascending=True
        )
        df["time_idx"] = (
            df[self.params["date_col"]].dt.year * 12 * 31
            + df[self.params["date_col"]].dt.month * 31
            + df[self.params["date_col"]].dt.day
        )
        df["time_idx"] -= df["time_idx"].min()
        return df

    def fit(self, X, y=None):
        _df = self.prepare_data(X)
        training_dataset, validation_dataset = self.train_validation_split(_df)
        train_dataloader = training_dataset.to_dataloader(
            train=True,
            batch_size=self.params["batch_size"],
            num_workers=self.params["num_workers"],
        )
        val_dataloader = validation_dataset.to_dataloader(
            train=False,
            batch_size=self.params["batch_size"],
            num_workers=self.params["num_workers"],
        )

        _trainer, _tft = self.create_model(training_dataset)
        self.trainer = _trainer
        self.tft = _tft

        self.trainer.fit(
            self.tft,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        predictions = self.tft.predict(val_dataloader)
        self.smape_train = mean_absolute_percentage_error(
            actuals.numpy(), predictions.numpy(), symmetric=True
        )
        return self

    def fillin_prediction_values(self, df):
        keys = df[self.params["group_id"]].unique()
        dfs = []
        for key in keys:
            k_df = df[df.Store == key]
            train_df = k_df[df.Sales.notnull()]
            pred_df = k_df[df.Sales.isnull()].sort_values(by=self.params["date_col"])
            pred_df = pred_df[: self.params["prediction_length"]]
            for col in self.params["time_varying_unknown_reals"]:
                _max = train_df.loc[train_df.Date.idxmax(), col]
                pred_df[col] = _max
            dfs.append(pd.concat([train_df, pred_df]))
        new_df = pd.concat(dfs)
        return new_df

    def predict(self, X, y=None):
        df = self.prepare_data(X)
        prediction_start_date = df[df[self.params["target"]].isnull()][
            self.params["date_col"]
        ].min()
        df = self.fillin_prediction_values(df)
        self.tft.eval()
        prediction, idx_df = self.tft.predict(df, mode="prediction", return_index=True)
        dfs = []
        for i in range(len(prediction)):
            date_idx = pd.date_range(
                prediction_start_date, periods=self.params["prediction_length"]
            )
            _df = pd.DataFrame(
                data=prediction[i], index=date_idx, columns=[self.params["target"]]
            )
            for col in idx_df.columns:
                if col not in ["time_idx"]:
                    _df[col] = idx_df[col].iloc[i]
            dfs.append(_df)
        res_df = pd.concat(dfs)
        res_df.reset_index(inplace=True)
        res_df.rename(columns={"index": self.params["date_col"]}, inplace=True)

        return res_df
