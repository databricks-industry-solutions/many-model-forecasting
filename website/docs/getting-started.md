---
id: getting-started
title: Getting Started
sidebar_position: 2
---

# Getting Started

To run this solution on a public [M4](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset) dataset, clone the MMF repo into your [Databricks Repos](https://www.databricks.com/product/repos).

## Installing `mmf_sa` without cloning the repository

If you want to use `mmf_sa` as a package without cloning the entire repository, install it directly from GitHub using `pip`:

```bash
pip install "mmf_sa @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git"
```

MMF provides optional dependency groups for different model types. Install them as needed:

```bash
# Local statistical models (statsforecast, prophet)
pip install "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git"

# Global deep learning models (neuralforecast)
pip install "mmf_sa[global] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git"

# Foundation models (chronos, timesfm)
pip install "mmf_sa[foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git"
```

To pin to a specific version, use a commit hash or a tag:

```bash
pip install "mmf_sa @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.5"
```

On Databricks, use `%pip` in a notebook cell:

```python
%pip install "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git" --quiet
dbutils.library.restartPython()
```

## The `run_forecast` function

Forecasting in MMF is driven by a single `run_forecast` call:

```python
catalog = "your_catalog_name"
db = "your_db_name"

run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.m4_daily_train",
    scoring_data=f"{catalog}.{db}.m4_daily_train",
    scoring_output=f"{catalog}.{db}.daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.daily_evaluation_output",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=10,
    backtest_length=30,
    stride=10,
    metric="smape",
    train_predict_ratio=2,
    data_quality_check=True,
    resample=False,
    active_models=active_models,
    experiment_path="/Shared/mmf_experiment",
    use_case_name="m4_daily",
)
```

### Parameters

- `train_data` — a delta table name that stores the input dataset.
- `scoring_data` — a delta table name that stores the [dynamic future regressors](https://nixtlaverse.nixtla.io/statsforecast/docs/how-to-guides/exogenous.html). If not provided, or if the same name as `train_data` is provided, the models ignore future dynamic regressors.
- `scoring_output` — a delta table where forecasting output is written (created if it does not exist).
- `evaluation_output` — a delta table where evaluation results from all backtesting trials are written (created if it does not exist).
- `group_id` — the column storing the unique id that groups your dataset into each time series.
- `date_col` — your time column name.
- `target` — your target column name.
- `freq` — your prediction frequency. `H` (hourly), `D` (daily), `W` (weekly), and `M` (monthly) are supported. Support is per-model, so check the model documentation.
- `prediction_length` — your forecasting horizon in number of steps.
- `backtest_length` — how many historical time points to use for backtesting.
- `stride` — the number of steps to advance the backtesting trial start date between trials.
- `metric` — the metric logged in the evaluation table and MLflow. Supported: `mae`, `mse`, `rmse`, `mape`, `smape`. Default is `smape`.
- `train_predict_ratio` — the minimum training length required relative to `prediction_length`. If set to `2`, training data must be at least twice as long as `prediction_length`.
- `data_quality_check` — runs quality checks on the input data when `True` (default `False`).
- `resample` — backfills skipped entries with `0` when `True`. Only relevant when `data_quality_check` is `True`. Default `False`.
- `active_models` — a list of models to use.
- `experiment_path` — where to keep metrics under MLflow.
- `use_case_name` — a column created in the delta table, in case you save multiple trials under one table.

## Timestamp Alignment Requirements

The `ds` (timestamp) column in `train_data` and `scoring_data` **must** be aligned to specific boundary dates depending on the frequency. Misaligned timestamps will produce incorrect backtesting windows and forecasts.

| Frequency     | Timestamp requirement        | Example                    |
| ------------- | ---------------------------- | -------------------------- |
| `H` (hourly)  | Any valid timestamp          | `2024-01-15 08:00:00`      |
| `D` (daily)   | Any valid date               | `2024-01-15`               |
| `W` (weekly)  | **Sunday** (end of ISO week) | `2024-01-14` (a Sunday)    |
| `M` (monthly) | **Last day of the month**    | `2024-01-31`, `2024-02-29` |

This is required because the backtesting engine uses `pd.offsets.MonthEnd` for monthly offsets and `pd.DateOffset(weeks=...)` for weekly offsets. If your source data uses different conventions, align the dates during data preparation:

```sql
-- Weekly: align to Sunday (end of ISO week)
CAST(DATE_TRUNC('week', date_col) + INTERVAL 6 DAY AS TIMESTAMP) AS ds

-- Monthly: align to month-end
CAST(LAST_DAY(date_col) AS TIMESTAMP) AS ds
```

## MLflow Integration

MMF is fully integrated with MLflow. Once training starts, experiments are visible in the MLflow Tracking UI with the corresponding metrics and parameters. The metric shown in the MLflow UI is a simple mean over backtesting trials across all time series. Refer to the [post-evaluation analysis notebook](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/post-evaluation-analysis.ipynb) for guidance on fine-grained model selection after running `run_forecast`.

To modify model hyperparameters, change the values in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml) or override them in a frequency-specific config such as [mmf_sa/forecasting_conf_daily.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/forecasting_conf_daily.yaml).
