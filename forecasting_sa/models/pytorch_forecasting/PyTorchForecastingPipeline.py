import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from forecasting_sa.models.abstract_model import ForecastingSARegressor
from forecasting_sa.models.pytorch_forecasting.PyTorchForecastingModel import (
    PyTorchForecastingRegressor,
)


class PyTorchForecastingPipelineRegressor(ForecastingSARegressor):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.pipeline = self.create_model(params)

    def create_model(self, params):
        pipeline = Pipeline(
            steps=[
                ("date_transformer", DateTransformer(params["date_col"])),
                ("convert_to_cat", ConvertToCategorical(params["categorical_columns"])),
                ("imputation", FillNaN(params["columns_to_fillna"])),
                ("ts_features_generator", FeatureGenerator(params["date_col"])),
                ("tft_forecast", PyTorchForecastingRegressor(params)),
            ]
        )
        return pipeline

    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)


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
