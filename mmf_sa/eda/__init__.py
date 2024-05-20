# Main entry point method
import base64
import sys
import uuid
import io
import importlib.resources as pkg_resources
from abc import abstractmethod
from typing import Dict, Any, List, Union
from jinja2 import Environment, BaseLoader
from omegaconf import OmegaConf
from pyspark.sql import DataFrame
import pandas as pd
import numpy as np
from PIL import Image

# data visualization
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns  # advanced vizs

# statistics
from statsmodels.distributions.empirical_distribution import ECDF
import missingno as msno
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


def display_html(html: str):
    from IPython.display import display as ip_display, HTML
    import IPython.core.display as icd

    orig_display = icd.display
    icd.display = display  # pylint: disable=undefined-variable
    ip_display(HTML(data=html))  # , filename=html_file_path
    icd.display = orig_display


def perform_auto_eda(
    df: DataFrame,
    conf: Union[Dict[str, Any], OmegaConf] = None,
    date_col: str = "Date",
    target: str = "y",
    group_columns: List[str] = None,
    trend_granularity: str = "month",
    seasonal_decompose_period: int = 356,
) -> None:
    if conf is None:
        conf = {
            "group_columns": group_columns,
            "date_col": date_col,
            "target": target,
            "trend_granularity": trend_granularity,
            "seasonal_decompose_period": seasonal_decompose_period,
        }

    report = EDAReport(conf)
    display_html(report.render(df))


__all__ = ["perform_auto_eda"]


class EDAReport:
    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf
        self.tabs = [
            DescriptiveStatisticsItem(conf),
            CorrelationItem(conf),
            SampleDataItem(conf),
            MissingValuesItem(conf),
            CategoricalStatsItem(conf),
            NumericalStatsItem(conf),
            DistItem(conf),
            TimeComponentsItem(conf),
        ]
        self.group_tabs = [GroupStatsItem(conf)]

    def render(self, df: DataFrame):
        _df = df.toPandas()
        # print(HistItem(conf))
        # print(type(HistItem(conf)))
        # distribution_tab = [f"{HistItem(conf)} \n {ECDFItem(conf)}"]
        # _tabs = self.tabs + distribution_tab
        html_template = pkg_resources.read_text(
            sys.modules[__name__], "eda_template_v2.html"
        )
        eda_template = Environment(loader=BaseLoader).from_string(html_template)
        page_id = str(uuid.uuid4())
        tab_list = [(tab.header, tab.render(_df)) for tab in self.tabs]

        # try:
        for group_tab in self.group_tabs:
            tab_list += [
                (f"{group} vs Target", group_tab.render(_df, group))
                for group in self.conf["group_columns"]
                if len(_df[group].unique()) < 50
            ]
        #         except:
        #             print("no groups will be used")

        template_dict = {
            "HEADER_TITLE": "Exploratory Data Analysis",
            "TABLINK": f"tablink-{page_id}",
            "CONTENT": f"content-{page_id}",
            "BUTTON_CONTAINER": f"button-container-{page_id}",
            "tab_list": tab_list,
        }
        html = eda_template.render(template_dict)
        return html


class EDAReportItem:
    def __init__(self, conf: Dict[str, Any], header: str, title: str, description: str):
        self.conf = conf
        self.header = header
        self.title = title
        self.description = description

    def render_image(self, bytes, image_type):
        base64_str = base64.b64encode(bytes).decode("utf-8")

        width, height = Image.open(io.BytesIO(bytes)).size
        html = f'<img src="data:image/{image_type};base64, {base64_str}" height={height} width={width} />'
        return self.render_simple_report_item(html)

    def render_simple_report_item(self, html: str):
        return f"<h2>{self.title}</h2>" f"{html}" f"<p>{self.description}</p>"

    @abstractmethod
    def render(self, df: DataFrame) -> str:
        pass


class DescriptiveStatisticsItem(EDAReportItem):
    def __init__(self, conf: Dict[str, Any]):
        super().__init__(
            conf,
            "Descriptive Statistics",
            "Statistics of Numerical Data",
            "Overall statistics of the numerical variables in the provided data.",
        )

    def render(self, df: pd.DataFrame) -> str:
        return self.render_simple_report_item(df.describe().T.to_html())


class SampleDataItem(EDAReportItem):
    def __init__(self, conf: Dict[str, Any]):
        super().__init__(conf, "Data Sample", "Random Sample of Provided Data", "")

    def render(self, df: pd.DataFrame) -> str:
        return self.render_simple_report_item(df.head(25).to_html())


class MissingValuesItem(EDAReportItem):
    def __init__(self, conf: Dict[str, Any]):
        super().__init__(
            conf,
            "Missing Values Analysis",
            "Analysis of Missing Values",
            "The nullity matrix is a data-dense display which is showing the distribution of missing values.",
        )

    def render(self, df: pd.DataFrame) -> str:
        msno.matrix(df)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")


class CorrelationItem(EDAReportItem):
    def __init__(self, conf: Dict[str, Any]):
        super().__init__(
            conf,
            "Correlation Analysis",
            "Correlation Matrix",
            "Heatmap of correlated variables. Darker collors stand for higher correlation.",
        )

    def render(self, df: pd.DataFrame) -> str:
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
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")


class CategoricalStatsItem(EDAReportItem):
    def __init__(self, conf: Dict[str, Any]):
        super().__init__(
            conf, "Categorical Variables Stats", "Categorical Variables Statistics", ""
        )

    def render(self, df: pd.DataFrame) -> str:
        table_cat = ff.create_table(
            df.describe(include=["O"]).T, index=True, index_title="Categorical columns"
        )
        fig = go.Figure(table_cat)
        img = fig.to_image(format="png")
        return self.render_image(img, "png")


class NumericalStatsItem(EDAReportItem):
    def __init__(self, conf: Dict[str, Any]):
        super().__init__(
            conf, "Numerical Variables", "Numerical Variables Statistics", ""
        )

    def render(self, df: pd.DataFrame) -> str:
        table_num = ff.create_table(
            df.describe(include=["float", "integer"]).T,
            index=True,
            index_title="Numerical columns",
        )
        fig = go.Figure(table_num)
        img = fig.to_image(format="png")
        return self.render_image(img, "png")


class DistItem(EDAReportItem):
    def __init__(self, conf: Dict[str, Any]):
        super().__init__(conf, "Distribution of " + conf["target"], "", "")

    def render(self, df: pd.DataFrame) -> str:
        return f"{ECDFItem.render(self, df)} \n {HistItem.render(self, df)}"


class ECDFItem(EDAReportItem):
    def render(self, df: pd.DataFrame) -> str:
        self.title = "Cumulative Distribution Function of" + self.conf["target"]
        self.description = (
            "How to read: What percentage (y-axes) of "
            + self.conf["target"]
            + " has value lower than X on x-axes."
        )

        plt.figure(figsize=(12, 6))
        cdf = ECDF(df[self.conf["target"]])
        plt.plot(cdf.x, cdf.y, label="statmodels")
        plt.xlabel(self.conf["target"])
        plt.ylabel("ECDF")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")


class HistItem(EDAReportItem):
    def render(self, df: pd.DataFrame) -> str:
        self.title = "Histogram of " + self.conf["target"]
        self.description = ""

        data = [
            go.Histogram(x=df[self.conf["target"]], nbinsx=50, name=self.conf["target"])
        ]
        fig = go.Figure(data)
        img = fig.to_image(format="png")
        return self.render_image(img, "png")


class GroupStatsItem(EDAReportItem):
    def __init__(self, conf: Dict[str, Any]):
        super().__init__(conf, "", "", "")

    def render(self, df: pd.DataFrame, group: str) -> str:
        return f"{GroupDescItem.render(self, df, group)} \n {GroupKdeItem.render(self, df, group)} \n {GroupFactorPlotItem.render(self, df, group)}"


class GroupDescItem(EDAReportItem):
    def render(self, df: pd.DataFrame, group: str) -> str:
        self.title = f'How {self.conf["target"]} behaves between different {group}s'
        self.description = ""

        style = (
            df.groupby(group)[self.conf["target"]]
            .describe()
            .style.background_gradient(cmap="Blues")
            .set_properties(**{"font-size": "12px"})
        )
        df_html = style.render()
        return self.render_simple_report_item(df_html)


class GroupKdeItem(EDAReportItem):
    def render(self, df: pd.DataFrame, group: str) -> str:
        self.title = (
            f'Kernel Density Estimation of {self.conf["target"]} in different {group}s'
        )
        self.description = f'Estimate the probablilty density function of {self.conf["target"]} in different {group}s'

        plt.figure(figsize=(7, 7))
        df.groupby(group)[self.conf["target"]].plot.kde(legend=True, bw_method=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")


class GroupFactorPlotItem(EDAReportItem):
    def render(self, df: pd.DataFrame, group: str) -> str:
        self.title = f'{self.conf["target"]} trend between different {group}s'
        self.description = ""

        if self.conf["trend_granularity"] == "year":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.year

        if self.conf["trend_granularity"] == "month":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.month

        if self.conf["trend_granularity"] == "day":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.day

        if self.conf["trend_granularity"] == "hour":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.hour

        if self.conf["trend_granularity"] == "minute":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.minute

        if self.conf["trend_granularity"] == "second":
            df[self.conf["trend_granularity"]] = df[self.conf["date_col"]].dt.second

        if self.conf["trend_granularity"] == "microsecond":
            df[self.conf["trend_granularity"]] = df[
                self.conf["date_col"]
            ].dt.microsecond

        if self.conf["trend_granularity"] == "week":
            df[self.conf["trend_granularity"]] = (
                df[self.conf["date_col"]].dt.isocalendar().week
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
            color="#386B7F",
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")


class TimeComponentsItem(EDAReportItem):
    def __init__(self, conf: Dict[str, Any]):
        super().__init__(
            conf, f'General Time Series Decomposition of {conf["target"]}', "", ""
        )

    def render(self, df: pd.DataFrame) -> str:
        df[self.conf["target"]] = df[self.conf["target"]] * 1.0

        decomposition = seasonal_decompose(
            df.groupby(self.conf["date_col"])[self.conf["target"]].mean(),
            model="additive",
            period=self.conf["seasonal_decompose_period"],
        )

        return (
            f"{MeanTimeComponentItem.render(self, df)} \n {TrendTimeComponentItem.render(self, df, decomposition)} "
            f"\n {SeasonalityTimeComponentItem.render(self, df, decomposition)} "
            f"\n {ResidualTimeComponentItem.render(self, df, decomposition)} "
            f"\n {ACFTimeComponentItem.render(self, df, decomposition)} "
            f"\n {PACFTimeComponentItem.render(self, df, decomposition)}"
        )


class MeanTimeComponentItem(EDAReportItem):
    def render(self, df: pd.DataFrame) -> str:
        self.title = f'Average {self.conf["target"]} across all groups'
        self.description = ""

        if self.conf["trend_granularity"] == "year":
            resampling = "AS"

        if self.conf["trend_granularity"] == "month":
            resampling = "M"

        if self.conf["trend_granularity"] == "day":
            resampling = "D"

        if self.conf["trend_granularity"] == "hour":
            resampling = "H"

        if self.conf["trend_granularity"] == "minute":
            resampling = "60S"

        if self.conf["trend_granularity"] == "second":
            resampling = "1S"

        if self.conf["trend_granularity"] == "microsecond":
            resampling = "U"

        if self.conf["trend_granularity"] == "week":
            resampling = "W"

        df[self.conf["target"]] = df[self.conf["target"]] * 1.0

        plt.figure(figsize=(15, 4))
        df.groupby(self.conf["date_col"])[self.conf["target"]].mean().resample(
            resampling
        ).mean().sort_index().plot(color="#386B7F", title="Mean")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")


class TrendTimeComponentItem(EDAReportItem):
    def render(self, df: pd.DataFrame, decomposition: str) -> str:
        self.title = f'Trend extracted from overall {self.conf["target"]} on a level of {self.conf["trend_granularity"]}'
        self.description = (
            "The data are not stationary in case of an up or down going trend. Additional ADF or KPSS "
            "test might be needed to check for stationarity."
        )

        plt.figure(figsize=(15, 4))
        decomposition.trend.plot(color="#386B7F", title="Trend")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")


class SeasonalityTimeComponentItem(EDAReportItem):
    def render(self, df: pd.DataFrame, decomposition: str) -> str:
        self.title = (
            f'General seasonality extracted from {self.conf["target"]} on a level of '
            f'{self.conf["trend_granularity"]}'
        )
        self.description = (
            "The data are not stationary in case of clear cyclic pattern. Differentiation techniques "
            "should be applied before statistical modelling."
        )

        plt.figure(figsize=(15, 4))
        decomposition.seasonal.plot(color="#386B7F", title="Seasonality")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")


class ResidualTimeComponentItem(EDAReportItem):
    def render(self, df: pd.DataFrame, decomposition: str) -> str:
        self.title = f"Residual Plot"
        self.description = (
            f'Variance in {self.conf["target"]} that can not be explained by linear combination of trend '
            f"seasonality, and level."
        )

        plt.figure(figsize=(15, 4))
        decomposition.resid.plot(color="#386B7F", title="Residuals")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")


class ACFTimeComponentItem(EDAReportItem):
    def render(self, df: pd.DataFrame, decomposition: str) -> str:
        self.title = f"Autocorrelation Function"
        self.description = (
            f"Autocorrelation represents the degree of similarity between a given time series and a "
            f"lagged version of itself over successive time intervals. Statistically significant correlation is observed on the lag values that go over above the significant threshold levels (blue area)."
        )

        plt.figure(figsize=(15, 4))
        plot_acf(
            df.groupby(self.conf["date_col"])[self.conf["target"]].mean(),
            lags=self.conf["seasonal_decompose_period"],
            color="#386B7F",
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")


class PACFTimeComponentItem(EDAReportItem):
    def render(self, df: pd.DataFrame, decomposition: str) -> str:
        self.title = f"Partial Autocorrelation Function"
        self.description = (
            f"Partial autocorrelation function explains the partial correlation between the series and "
            f"lags itself. Finds correlation of the residuals which remains after removing the effects "
            f"which are already explained by the earlier lag(s)) with the next lag value. Statistically "
            f"significant correlation is observed on the lag values that go over above the significant "
            f"threshold levels (blue area)."
        )

        plt.figure(figsize=(15, 4))
        plot_pacf(
            df.groupby(self.conf["date_col"])[self.conf["target"]].mean(),
            lags=self.conf["seasonal_decompose_period"],
            color="#386B7F",
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return self.render_image(buf.read(), "png")
