from typing import Union, Dict

import pandas as pd
from gluonts.dataset import common
from gluonts.dataset.pandas import PandasDataset

from gluonts.model.estimator import Estimator
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.torch.model.deepar import DeepAREstimator
from hyperopt import hp

from forecasting_sa.models.abstract_model import (
    ForecastingSARegressor,
    ForecastingSAPivotRegressor,
    ForecastingSAVerticalizedDataRegressor,
)


class GluonTSRegressor(ForecastingSAVerticalizedDataRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model = None
        self.predictor = None

    def create_gluon_dataset(self, df: pd.DataFrame) -> PandasDataset:

        static_categoricals = self.params.get("static_categoricals", [])
        if static_categoricals:
            for static_categorical in static_categoricals:
                if df[static_categorical].dtype != int:
                    df[static_categorical] = (
                        df[static_categorical].astype("category").cat.codes
                    )
        dataset = PandasDataset.from_long_dataframe(
            df,
            item_id=self.params["group_id"],
            target=self.params["target"],
            timestamp=self.params["date_col"],
            freq=self.freq,
            feat_dynamic_cat=self.params.get("dynamic_categoricals", []),
            feat_dynamic_real=self.params.get("dynamic_reals", []),
            feat_static_cat=static_categoricals,
            feat_static_real=[],
        )
        return dataset

    def fit(self, X, y=None):
        if isinstance(self.model, Estimator):
            _df = self.prepare_data(X)
            gluon_dataset = self.create_gluon_dataset(_df)
            self.predictor = self.model.train(gluon_dataset)
        return self

    def predict_gluon_ds(self, ds):
        sampleforecast_list = list(self.predictor.predict(ds))
        res_df = pd.DataFrame([f.mean_ts for f in sampleforecast_list]).transpose()
        return res_df

    def predict(self, X):
        _df = self.prepare_data(X)
        gluon_dataset = self.create_gluon_dataset(_df)
        df_idx = dict([(i, data["item_id"]) for i, data in enumerate(gluon_dataset)])
        res_df = pd.concat([self.predict_gluon_ds(gluon_dataset) for i in range(10)])
        res_df = res_df.reset_index().groupby(by="index").mean()
        res_df = res_df.rename(columns=df_idx)
        res_df = (
            res_df.reset_index()
            .melt(id_vars=["index"])
            .rename(
                columns={
                    "index": self.params["date_col"],
                    "variable": self.params["group_id"],
                    "value": self.params["target"],
                }
            )
        )
        res_df[self.params["target"]] = res_df[self.params["target"]].fillna(value=0)
        res_df[self.params["target"]] = res_df[self.params["target"]].clip(0)
        return res_df


class GluonTSPivotRegressor(ForecastingSAPivotRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model = None
        self.predictor = None

    def create_gluon_dataset(self, df: pd.DataFrame) -> common.ListDataset:
        data = [
            {
                "start": df[self.params["date_col"]].iloc[0],
                "target": df[c],
                "col_name": c,
            }
            for c in df.columns
            if c not in [self.params["date_col"]]
        ]
        dataset = common.ListDataset(data, freq=self.params["freq"])
        return dataset

    def fit(self, X, y=None):
        if isinstance(self.model, Estimator):
            _df = self.prepare_data(X)
            gluon_dataset = self.create_gluon_dataset(_df)
            self.predictor = self.model.train(gluon_dataset)

    def predict_gluon_ds(self, ds):
        sampleforecast_list = list(self.predictor.predict(ds))
        res_df = pd.DataFrame([f.mean_ts for f in sampleforecast_list]).transpose()
        return res_df

    def predict(self, X):
        _df = self.prepare_data(X)
        gluon_dataset = self.create_gluon_dataset(_df)
        df_idx = dict([(i, data["col_name"]) for i, data in enumerate(gluon_dataset)])
        res_df = pd.concat([self.predict_gluon_ds(gluon_dataset) for i in range(10)])
        res_df = res_df.reset_index().groupby(by="index").mean()
        res_df = res_df.rename(columns=df_idx)
        return res_df


class GluonTSSeasonalNaiveRegressor(GluonTSRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model = SeasonalNaivePredictor(
            freq=self.params["freq"],
            prediction_length=int(self.params["prediction_length"]),
        )
        self.predictor = self.model


class GluonTSProphetRegressor(GluonTSRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model = ProphetPredictor(
            prediction_length=int(self.params["prediction_length"]),
            prophet_params={
                "growth": self.params["model_spec"]["growth"],
                "changepoints": self.params["model_spec"]["changepoints"],
                "n_changepoints": self.params["model_spec"]["n_changepoints"],
                "yearly_seasonality": self.params["model_spec"]["yearly_seasonality"],
                "weekly_seasonality": self.params["model_spec"]["weekly_seasonality"],
                "daily_seasonality": self.params["model_spec"]["daily_seasonality"],
                "seasonality_mode": self.params["model_spec"]["seasonality_mode"],
            },
        )
        self.predictor = self.model
        self._search_space = {
            "yearly_seasonality": hp.quniform("yearly_seasonality", 10, 20, 5),
        }

    def search_space(self):
        return self._search_space

    def supports_tuning(self) -> bool:
        return True


class GluonTSTorchDeepARRegressor(GluonTSRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.model = DeepAREstimator(
            freq=self.freq,
            prediction_length=int(self.params["prediction_length"]),
            context_length=int(self.params["context_length"]),
            batch_size=int(self.params["batch_size"]),
            hidden_size=int(self.params["hidden_size"]),
            num_layers=int(self.params["num_layers"]),
            dropout_rate=float(self.params["dropout_rate"]),
            trainer_kwargs={
                "logger": False,
                "max_epochs": int(self.params["epochs"]),
                "accelerator": self.params.get("accelerator", "auto"),
                "default_root_dir": self.params.get("temp_path", "/tmp"),
            },
        )
        self.predictor = self.model
        self._search_space = {
            "epochs": hp.quniform(
                "epochs", 1, int(self.params["tuning_max_epochs"]), 1
            ),
            "context_length": hp.quniform(
                "context_length", 20, int(self.params["tuning_max_context_len"]), 5
            ),
            "batch_size": hp.quniform("batch_size", 32, 256, 8),
            "hidden_size": hp.quniform("hidden_size", 32, 192, 8),
            "num_layers": hp.quniform("num_layers", 1, 8, 1),
            "dropout_rate": hp.uniform("dropout_rate", 0.000001, 0.5),
        }

    def search_space(self):
        return self._search_space

    def supports_tuning(self) -> bool:
        return True


# from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator
# class GluonTSTorchMQF2Regressor(GluonTSRegressor):
#     def __init__(self, params):
#         super().__init__(params)
#         self.params = params
#         self.model = MQF2MultiHorizonEstimator(freq=self.params['freq'],
#                                      prediction_length=int(self.params['prediction_length']),
#                                      context_length=40,
#                                      trainer_kwargs={
#                                          "max_epochs": 10,
#                                      })
#         self.predictor = self.model
