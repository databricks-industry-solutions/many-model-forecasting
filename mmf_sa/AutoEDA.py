import uuid
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pyspark.pandas as pd
from typing import Dict, Any, List
from pyspark.sql import SparkSession, DataFrame

import mlflow
import missingno as msno

# data visualization
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns  # advanced vizs

# statistics
from statsmodels.distributions.empirical_distribution import ECDF

# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# report generation
import sys
import importlib.resources as pkg_resources

from jinja2 import Environment, BaseLoader
import base64
import pathlib
import os
import logging

_logger = logging.getLogger(__name__)


class AutoEDA:
    def __init__(
        self,
        spark: SparkSession,
        df: DataFrame = None,
        sql: str = None,
        table: str = None,
        date_col: str = "Date",
        target: str = "y",
        group_columns: List[str] = None,
        trend_granularity: str = "month",
    ):
        self.conf = {}
        self.spark = spark
        # self.experiment_id = experiment_id
        self.c = "#386B7F"
        self.df = df
        self.sql = sql
        self.table = table
        self.conf["group_columns"] = group_columns
        self.conf["date_col"] = date_col
        self.conf["target"] = target
        self.conf["trend_granularity"] = trend_granularity
        self.temp_path = f"/tmp/{str(uuid.uuid4())}"

    def read_data(self) -> pd.DataFrame:
        if self.df is not None:
            _df = self.df
        elif self.sql is not None:
            _df = self.spark.sql(self.sql)
        else:
            _df = self.spark.read.table(self.table)
        _df = _df.toPandas()
        _df[self.conf["date_col"]] = pd.to_datetime(_df[self.conf["date_col"]])
        return _df

    def discriptive_stats(self, df):
        df.describe().T.to_html(f"{self.temp_path}/descriptive_stats.html")
        mlflow.log_artifact(f"{self.temp_path}/descriptive_stats.html")
        return

    def sample_data(self, df):
        df.head(25).to_html(f"{self.temp_path}/data_sample.html")
        mlflow.log_artifact(f"{self.temp_path}/data_sample.html")
        return

    def missing_values(self, df):
        msno.matrix(df)
        plt.savefig(f"{self.temp_path}/missing_values.png")
        mlflow.log_artifact(f"{self.temp_path}/missing_values.png")
        return

    def variable_stats(self, df):
        table_cat = ff.create_table(
            df.describe(include=["O"]).T, index=True, index_title="Categorical columns"
        )
        fig = go.Figure(table_cat)
        fig.write_image(f"{self.temp_path}/categorical_variables.png")
        # mlflow.log_artifact("categorical_variables.png")

        table_num = ff.create_table(
            df.describe(include=["float", "integer"]).T,
            index=True,
            index_title="Numerical columns",
        )
        fig = go.Figure(table_num)
        fig.write_image(f"{self.temp_path}/numerical_variables.png")
        mlflow.log_artifact(f"{self.temp_path}/numerical_variables.png")
        return

    def build_hist(self, df):
        data = [
            go.Histogram(x=df[self.conf["target"]], nbinsx=50, name=self.conf["target"])
        ]
        fig = go.Figure(data)
        fig.write_image(f"{self.temp_path}/hist.png")
        mlflow.log_artifact(f"{self.temp_path}/f.png")
        return

    def distrubutions(self, df):

        plt.figure(figsize=(12, 6))
        cdf = ECDF(df[self.conf["target"]])
        plt.plot(cdf.x, cdf.y, label="statmodels", color=self.c)
        plt.xlabel(self.conf["target"])
        plt.ylabel("ECDF")

        plt.savefig(f"{self.temp_path}/{self.conf['target']}_ecdf.png")
        mlflow.log_artifact(
            f"{self.temp_path}/{self.conf['target']}_ecdf.png",
            artifact_path="distributions",
        )
        return

    def correlation(self, df):
        # Compute the correlation matrix
        corr_all = df.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr_all, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            corr_all, mask=mask, square=True, linewidths=0.5, ax=ax, cmap="BuPu"
        )
        plt.savefig(f"{self.temp_path}/correlation_matrix.png")
        mlflow.log_artifact(f"{self.temp_path}/correlation_matrix.png")
        return

    def trend_extraction(self, df):
        if self.conf["trend_granularity"] == "year":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.year
            resampling = "AS"

        if self.conf["trend_granularity"] == "month":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.month
            resampling = "M"

        if self.conf["trend_granularity"] == "day":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.day
            resampling = "D"

        if self.conf["trend_granularity"] == "hour":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.hour
            resampling = "H"

        if self.conf["trend_granularity"] == "minute":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.minute
            resampling = "60S"

        if self.conf["trend_granularity"] == "second":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.second
            resampling = "1S"

        if self.conf["trend_granularity"] == "microsecond":
            df[self.conf["trend_granularity"]] = df[
                self.conf["date_col"]
            ].dt.microsecond
            resampling = "U"

        if self.conf["trend_granularity"] == "week":
            df[self.conf["trend_granularity"]] = (
                df[self.conf["date_col"]].dt.isocalendar().week
            )
            resampling = "W"

        def group_stats(df):
            print("Calculating Group Statistics")
            for group in self.conf["group_columns"]:
                df_html = ""
                style = (
                    df.groupby(group)[self.conf["target"]]
                    .describe()
                    .style.background_gradient(cmap="Blues")
                    .set_properties(**{"font-size": "12px"})
                )
                df_html = style.render()

                with open(
                    f"{self.temp_path}/{group}_vs_{self.conf['target']}_stats.html", "w"
                ) as file:
                    file.seek(0)
                    file.write(df_html)
                    file.truncate()

                mlflow.log_artifact(
                    f"{self.temp_path}/{group}_vs_{self.conf['target']}_stats.html",
                    artifact_path="group_stats",
                )

                plt.figure(figsize=(7, 7))
                df.groupby(group)[self.conf["target"]].plot.kde(
                    legend=True, bw_method=0.5
                )
                plt.savefig(f"{self.temp_path}/{group}_{self.conf['target']}_kde.png")
                mlflow.log_artifact(
                    f"{self.temp_path}/{group}_{self.conf['target']}_kde.png",
                    artifact_path="distributions",
                )

                plt.figure(figsize=(45, 10))
                sns.factorplot(
                    data=df,
                    x=self.conf["trend_granularity"],
                    y=self.conf["target"],
                    col=group,  # per store type in cols
                    col_wrap=3,
                    palette="plasma",
                    hue=group,
                    color=self.c,
                )
                plt.savefig(f"{self.temp_path}/{group}_{self.conf['target']}_trend.png")
                mlflow.log_artifact(
                    f"{self.temp_path}/{group}_{self.conf['target']}_trend.png",
                    artifact_path="trends",
                )
            return

        def extract_time_components(df, resampling):
            # Time Components
            print("Extracting Time Series Components")

            # f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(15, 20))
            df[self.conf["target"]] = df[self.conf["target"]] * 1.0

            plt.figure(figsize=(15, 4))
            df.groupby(self.conf["date_col"])[self.conf["target"]].mean().resample(
                resampling
            ).mean().sort_index().plot(title="Mean")
            plt.savefig(f"{self.temp_path}/mean.png")

            decomposition = seasonal_decompose(
                df.groupby(self.conf["date_col"])[self.conf["target"]].mean(),
                model="additive",
                period=self.conf["seasonal_decompose_period"],
            )

            plt.figure(figsize=(15, 4))
            decomposition.trend.plot(color=self.c, title="Trend")
            plt.savefig(f"{self.temp_path}/trend.png")

            plt.figure(figsize=(15, 4))
            decomposition.seasonal.plot(color=self.c, title="Seasonality")
            plt.savefig(f"{self.temp_path}/seasonality.png")

            plt.figure(figsize=(15, 4))
            decomposition.resid.plot(color=self.c, title="Residuals")
            plt.savefig(f"{self.temp_path}/residuals.png")

            plt.figure(figsize=(15, 4))
            plot_acf(
                df.groupby(self.conf["date_col"])[self.conf["target"]].mean(),
                lags=self.conf["seasonal_decompose_period"],
                color=self.c,
            )
            plt.savefig(f"{self.temp_path}/acf.png")

            plt.figure(figsize=(15, 4))
            plot_pacf(
                df.groupby(self.conf["date_col"])[self.conf["target"]].mean(),
                lags=self.conf["seasonal_decompose_period"],
                color=self.c,
            )
            plt.savefig(f"{self.temp_path}/pacf.png")

            mlflow.log_artifact(f"{self.temp_path}/trend.png")
            mlflow.log_artifact(f"{self.temp_path}/seasonality.png")
            mlflow.log_artifact(f"{self.temp_path}/residuals.png")
            mlflow.log_artifact(f"{self.temp_path}/acf.png")
            mlflow.log_artifact(f"{self.temp_path}/pacf.png")
            mlflow.log_artifact(f"{self.temp_path}/mean.png")
            return

        group_stats(df)
        extract_time_components(df, resampling)
        return

    def explore(self):
        pathlib.Path(self.temp_path).mkdir(parents=True)
        sns.set(style="ticks")  # to format into seaborn
        # c = '#386B7F'  # basic color for plots
        print("Reading Data")
        df = self.read_data()
        print("Collecting Discriptive Statietics")
        self.discriptive_stats(df)
        print("Logging Data Sample")
        self.sample_data(df)
        print("Missing Value Analysis")
        self.missing_values(df)
        print("Collecting Variable Statietics")
        self.variable_stats(df)
        print("Building Histograms")
        self.build_hist(df)
        print("Building Distributions")
        self.distrubutions(df)
        print("Corelation Analysis")
        self.correlation(df)
        self.trend_extraction(df)
        html = self.report_generation_new()
        self.display_html(html)

        return

    def display_html(self, html: str):
        from IPython.display import display as ip_display, HTML
        import IPython.core.display as icd

        orig_display = icd.display
        icd.display = display  # pylint: disable=undefined-variable
        ip_display(HTML(data=html))  #  , filename=html_file_path
        icd.display = orig_display

    def auto_exploration(self):

        sns.set(style="ticks")  # to format into seaborn
        # c = '#386B7F'  # basic color for plots
        print("Reading Data")
        df = self.read_data(self.conf["train_data"])
        print("Collecting Discriptive Statietics")
        self.discriptive_stats(df)
        print("Logging Data Sample")
        self.sample_data(df)
        print("Missing Value Analysis")
        self.missing_values(df)
        print("Collecting Variable Statietics")
        self.variable_stats(df)
        print("Building Histograms")
        self.build_hist(df)
        print("Building Distributions")
        self.distrubutions(df)
        print("Corelation Analysis")
        self.correlation(df)
        self.trend_extraction(df)
        self.report_generation_new()
        return

    def add_image(
        self, image_file_path: str, width: int = None, height: int = None
    ) -> None:
        if not os.path.exists(image_file_path):
            _logger.warning(f"Unable to locate image file {image_file_path} to render.")
            return

        with open(image_file_path, "rb") as f:
            base64_str = base64.b64encode(f.read()).decode("utf-8")

        image_type = pathlib.Path(image_file_path).suffix[1:]

        width_style = f'width="{width}"' if width else ""
        height_style = f'height="{height}"' if height else ""
        img_html = (
            f'<img src="data:image/{image_type};base64, {base64_str}" '
            f"{width_style} {height_style} />"
        )
        return img_html

    def add_title_and_describtion(self, html_object: str, title: "", describtion: ""):
        html_string = (
            "<h2>" + title + "</h2>" + html_object + "<p>" + describtion + "</p>"
        )
        return html_string

    def report_generation_new(self):
        def load_html():
            import mmf_sa.html_templates

            html_template = pkg_resources.read_text(
                sys.modules["mmf_sa.html_templates"], "eda_template_v2.html"
            )
            return html_template

        eda_template = Environment(loader=BaseLoader).from_string(load_html())

        page_id = str(uuid.uuid4())

        tab_list = []

        # Descriptive Statistics
        with open(f"{self.temp_path}/descriptive_stats.html") as f:
            descriptive_stats = f.readlines()
        descriptive_stats = "".join(descriptive_stats)
        tab_list.append(
            [
                self.conf["target"] + " Descriptive Statistics",
                self.add_title_and_describtion(
                    descriptive_stats,
                    "Descriptive Statistics",
                    "Overall statistics of the numerical variables in the provided data.",
                ),
            ]
        )

        # Data Sample
        with open(f"{self.temp_path}/data_sample.html") as f:
            data_sample = f.readlines()
        data_sample = "".join(data_sample)
        tab_list.append(
            [
                "Data Sample",
                self.add_title_and_describtion(
                    data_sample, "Random Sample of Provided Data", ""
                ),
            ]
        )

        # Categorical Variables
        tab_list.append(
            [
                "Categorical Variables Stats",
                self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/categorical_variables.png"),
                    "Categorical Variables Statistics",
                    "",
                ),
            ]
        )

        # Numerical Variables
        tab_list.append(
            [
                "Numerical Variables",
                self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/numerical_variables.png"),
                    "Numerical Variables Statistics",
                    "",
                ),
            ]
        )

        # Missing Values
        tab_list.append(
            [
                "Missing Values Analysis",
                self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/missing_values.png", 1000, 500),
                    "Analysis of Missing Values",
                    "The nullity matrix is a data-dense display which is showing the distribution of missing values.",
                ),
            ]
        )

        # Correlation
        tab_list.append(
            [
                "Correlation Analysis",
                self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/correlation_matrix.png"),
                    "Correlation Matrix",
                    "Heatmap of correlated variables. Darker collors stand for higher correlation.",
                ),
            ]
        )

        # Distribution of the Target Variable
        tab_list.append(
            [
                "Distribution of " + self.conf["target"],
                self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/hist.png"),
                    "Histogram of " + self.conf["target"],
                    "",
                )
                + self.add_title_and_describtion(
                    self.add_image(f'{self.temp_path}/{self.conf["target"]}_ecdf.png'),
                    "Cumulative Distribution Function of" + self.conf["target"],
                    "How to read: What percentage (y-axes) of "
                    + self.conf["target"]
                    + " has value lower than X on x-axes.",
                ),
            ]
        )

        # Overall Time Series Decomposition
        tab_list.append(
            [
                "General Time Series Decomposition of " + self.conf["target"],
                self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/mean.png"),
                    "Average " + self.conf["target"] + " across all groups",
                    "",
                )
                + self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/trend.png"),
                    "Trend extracted from overall "
                    + self.conf["target"]
                    + " on a level of "
                    + self.conf["trend_granularity"],
                    "The data are not stationary in case of an up or down going trend. Additional ADF or KPSS tess might be needed to check for stationarity.",
                )
                + self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/seasonality.png"),
                    "General seasonality extracted from "
                    + self.conf["target"]
                    + " on a level of "
                    + self.conf["trend_granularity"],
                    "The data are not stationary in case of clear cyclic pattern. Differentiation techniques should be applied before statistical modelling.",
                )
                + self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/residuals.png"),
                    "Residual Plot",
                    "Variance in "
                    + self.conf["target"]
                    + " that can not be explained by linear combination of trend, seasonality, and level.",
                )
                + self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/acf.png"),
                    "Autocorrelation Function",
                    "Autocorrelation represents the degree of similarity between a given time series and a lagged version of itself over successive time intervals. Statistically significant correlation is observed on the lag values that go over above the significant threshold levels (blue area).",
                )
                + self.add_title_and_describtion(
                    self.add_image(f"{self.temp_path}/pacf.png"),
                    "Partial Autocorrelation Function",
                    "Partial autocorrelation function explains the partial correlation between the series and lags itself. Finds correlation of the residuals which remains after removing the effects which are already explained by the earlier lag(s)) with the next lag value. Statistically significant correlation is observed on the lag values that go over above the significant threshold levels (blue area).",
                ),
            ]
        )

        # Generate Group Tabs
        for group in self.conf["group_columns"]:
            with open(
                f'{self.temp_path}/{group}_vs_{self.conf["target"]}_stats.html'
            ) as f:
                group_stats = f.readlines()
            group_stats = "".join(group_stats)

            tab_list.append(
                [
                    group + " vs " + self.conf["target"] + " Statistics",
                    self.add_title_and_describtion(
                        group_stats,
                        "How "
                        + self.conf["target"]
                        + " behaves between different "
                        + group
                        + "s",
                        "",
                    )
                    + self.add_title_and_describtion(
                        self.add_image(
                            f'{self.temp_path}/{group}_{self.conf["target"]}_trend.png'
                        ),
                        self.conf["target"] + " Trend between different " + group + "s",
                        "",
                    )
                    + self.add_title_and_describtion(
                        self.add_image(
                            f'{self.temp_path}/{group}_{self.conf["target"]}_kde.png'
                        ),
                        "Kernel Density Estimation of "
                        + self.conf["target"]
                        + " in different "
                        + group
                        + "s",
                        "Estimate the probablilty density function of"
                        + self.conf["target"]
                        + " in different "
                        + group
                        + "s",
                    ),
                ]
            )
            group_stats = ""

        template_dict = {}
        template_dict["HEADER_TITLE"] = "Exploratory Data Analysis"
        template_dict["TABLINK"] = f"tablink-{page_id}"
        template_dict["CONTENT"] = f"content-{page_id}"
        template_dict["BUTTON_CONTAINER"] = f"button-container-{page_id}"
        template_dict["tab_list"] = tab_list

        data = eda_template.render(template_dict)

        with open(f"{self.temp_path}/html_output_v2.html", "w") as file:
            file.seek(0)
            file.write(data)
            file.truncate()
        mlflow.log_artifact(f"{self.temp_path}/html_output_v2.html")

        return data
