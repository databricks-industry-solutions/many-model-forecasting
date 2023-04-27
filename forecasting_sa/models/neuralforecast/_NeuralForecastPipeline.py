from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from forecasting_sa.models.abstract_model import ForecastingSAVerticalizedDataRegressor
from forecasting_sa.models.neuralforecast._NeuralForecastModel import (
    NeuralForecastRegressor,
)


class NeuralForecastPipelineRegressor(ForecastingSAVerticalizedDataRegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.pipeline = self.create_model(params)

    def create_model(self, params):
        pipeline = Pipeline(
            steps=[
                # ("date_transformer", DateTransformer(params["date_col"])),
                ("std_scaler", StandardScaler(params)),
                # ("convert_to_cat", ConvertToCategorical(params["categorical_columns"])),
                # ("imputation", FillNaN(params["columns_to_fillna"])),
                # ("ts_features_generator", FeatureGenerator(params["date_col"])),
                ("neuralforecast", NeuralForecastRegressor(params)),
            ]
        )
        return pipeline

    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    def predict(self, X):
        df = self.pipeline.predict(X)
        df = self.pipeline.inverse_transform(df)
        df[self.params["target"]] = df[self.params["target"]].clip(0)
        return df

    def calculate_metrics(
        self, hist_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, float]:
        pred_df = self.predict(hist_df)
        smape = mean_absolute_percentage_error(
            val_df[self.params["target"]],
            pred_df[self.params["target"]],
            symmetric=True,
        )
        return {"smape": smape}


class StandardScaler(BaseEstimator, TransformerMixin):
    """This class helps to standardize a dataframe with multiple time series."""

    def __init__(self, params):
        self.params = params
        self.norm: pd.DataFrame

    def fit(self, X: pd.DataFrame, y=None) -> "StandardScaler":
        self.columns = [
            col
            for col in X.columns
            if col not in [self.params["group_id"], self.params["date_col"]]
        ]
        self.norm = X.groupby(self.params["group_id"]).agg(
            {self.params["target"]: [np.mean, np.std]}
        )
        self.norm = self.norm.droplevel(0, 1).reset_index()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        transformed = X.merge(self.norm, how="left", on=[self.params["group_id"]])
        transformed[self.params["target"]] = (
            transformed[self.params["target"]] - transformed["mean"]
        ) / transformed["std"]
        return transformed[
            [self.params["group_id"], self.params["date_col"], self.params["target"]]
        ]

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.merge(self.norm, how="left", on=[self.params["group_id"]])
        _cols = []
        for col in self.columns:
            if col in X.columns:
                _cols.append(col)
                transformed[col] = (
                    transformed[col] * transformed["std"] + transformed["mean"]
                )
        return transformed[[self.params["group_id"], self.params["date_col"]] + _cols]

    def get_feature_names_out(self, input_features=None):
        return self.columns


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["Date"] = pd.to_datetime(X["Date"])
        X["year"] = X.Date.dt.year.astype(str).astype("category")
        X["month"] = X.Date.dt.month.astype(str).astype("category")
        return X


class ConvertToCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.columns] = X[self.columns].astype(str).astype("category")
        return X


class FillNaN(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.columns] = X[self.columns].fillna(value=-1)
        return X


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, data_column):
        self.data_column = data_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["Week"] = X[self.data_column].dt.week
        X["YearDay"] = X[self.data_column].dt.dayofyear
        X["MonthDay"] = X[self.data_column].dt.day
        return X
