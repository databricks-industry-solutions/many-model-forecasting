"""calibrate_intervals on the tables of a local Spark run_forecast with native intervals (prediction intervals).

Tests cover, with StatsForecastAutoETS (native 80% and 95% bands), StatsForecastCrostonSBA (no native intervals) and
a copy of AutoETS without its 95% quantiles, as run_forecast writes them for a model whose native range is 10-90%:
- the method of each model and level: CQR where the backtest has the native band, conformal otherwise
- calibration_output, interval metrics (with wis over all levels) and scoring_output_intervals, with the run's ids
- calibrated scoring intervals filled for every level, clipped and nested
- the tables of run_forecast are not modified, and a re-run appends with a new calibration_date
- models and run_ids select what is calibrated

These tests need Spark Python workers (PySpark >= 3.5 on Python <= 3.11 locally); they are skipped otherwise.
"""
import mlflow
import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from mmf_sa import calibrate_intervals, run_forecast

H = 7
NATIVE, NO_NATIVE, NARROW = "StatsForecastAutoETS", "StatsForecastCrostonSBA", "NarrowNative"
PREFIX = "default.pi_cal"


@pytest.fixture(scope="module")
def spark(spark_session):
    spark_session.conf.set("spark.sql.shuffle.partitions", "4")
    try:
        spark_session.createDataFrame(pd.DataFrame({"a": [1]})).groupBy("a").applyInPandas(
            lambda p: p, "a long").collect()
    except Exception as err:  # PySpark 3.4 workers cannot start on Python 3.12
        pytest.skip(f"Spark Python workers unavailable: {type(err).__name__}")
    return spark_session


def _train(n_series=10, n_days=98, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    frames = []
    for i in range(n_series):
        y = np.maximum(5 + i + 2 * np.sin(np.arange(n_days) * 2 * np.pi / 7) + rng.normal(0, 1, n_days), 0)
        frames.append(pd.DataFrame({"unique_id": f"s{i}", "ds": dates, "y": y}))
    return pd.concat(frames, ignore_index=True)


def _without_95(sdf, map_col):
    """A model's rows renamed NARROW, without the 95% quantiles (as for a model whose native range is 10-90%)."""
    return (sdf.withColumn("model", F.lit(NARROW))
            .withColumn(map_col, F.expr(f"map_filter({map_col}, (k, v) -> k NOT IN ('0.025', '0.975'))")))


@pytest.fixture(scope="module")
def forecast(spark, tmp_path_factory):
    path = tmp_path_factory.mktemp("pi_cal")
    mlflow.set_tracking_uri(f"sqlite:///{path}/mlruns.db")
    spark.createDataFrame(_train()).createOrReplaceTempView("pi_cal_train")
    run_id = run_forecast(
        spark=spark, train_data="pi_cal_train", scoring_data="pi_cal_train", scoring_output=f"{PREFIX}_scoring_output",
        evaluation_output=f"{PREFIX}_evaluation_output", group_id="unique_id", date_col="ds", target="y", freq="D",
        prediction_length=H, backtest_length=21, stride=7, train_predict_ratio=1, active_models=[NATIVE, NO_NATIVE],
        experiment_path=f"{path}/exp", use_case_name="pi_cal",
        prediction_intervals={"enabled": True, "output_format": "quantile_map"},
    )
    for table, map_col in (("evaluation_output", "forecast_quantiles"), ("scoring_output", "y_quantiles")):
        rows = spark.table(f"{PREFIX}_{table}").filter(F.col("model") == NATIVE)
        _without_95(rows, map_col).write.mode("append").saveAsTable(f"{PREFIX}_{table}")
    return run_id


def _snapshot(spark):
    return {t: (spark.table(f"{PREFIX}_{t}").count(), spark.table(f"{PREFIX}_{t}").columns)
            for t in ("evaluation_output", "scoring_output")}


@pytest.fixture(scope="module")
def calibrated(spark, forecast):
    before = _snapshot(spark)
    out = calibrate_intervals(spark, train_data="pi_cal_train", evaluation_output=f"{PREFIX}_evaluation_output",
                              scoring_output=f"{PREFIX}_scoring_output", freq="D", output_format="both")
    return out, before


def test_methods_per_model_and_level(calibrated):
    out, _ = calibrated
    methods = {m: s["methods"] for m, s in out["models"].items()}
    cqr, conformal = "cqr/asymmetric/marginal", "conformal/signed_normalized/series"
    assert methods[NATIVE] == {"80": cqr, "95": cqr}
    assert methods[NO_NATIVE] == {"80": conformal, "95": conformal}
    assert methods[NARROW] == {"80": cqr, "95": conformal}
    assert out["tables"] == {"calibration": f"{PREFIX}_evaluation_output_calibration",
                             "interval_metrics": f"{PREFIX}_evaluation_output_interval_metrics",
                             "scoring_intervals": f"{PREFIX}_scoring_output_intervals"}


def test_calibration_and_metrics_carry_the_forecast_run(spark, forecast, calibrated):
    out, _ = calibrated
    cal = spark.table(out["tables"]["calibration"]).toPandas()
    assert set(cal["run_id"]) == {forecast} and cal["calibration_date"].notna().all()
    narrow = cal[cal["model"] == NARROW].groupby("level")["method"].first().to_dict()
    assert narrow == {80.0: "cqr", 95.0: "conformal"}
    metrics = spark.table(out["tables"]["interval_metrics"]).toPandas()
    names = metrics.groupby("model")["metric_name"].unique().map(set)
    assert {"wis", "scaled_ae", "coverage_80", "coverage_95"} <= names[NARROW]
    assert "native_wis" in names[NATIVE] and "native_wis" not in names[NARROW]
    assert not metrics.duplicated(["unique_id", "backtest_window_start_date", "model", "metric_name"]).any()


def test_scoring_intervals_are_filled_clipped_and_nested(spark, calibrated):
    out, _ = calibrated
    score = spark.table(out["tables"]["scoring_intervals"]).toPandas()
    assert score.groupby("model").size().to_dict() == {NARROW: 10, NO_NATIVE: 10, NATIVE: 10}
    for _, r in score.iterrows():
        lo80, hi80, lo95, hi95 = (np.asarray(r[c]) for c in ("y_lo_80", "y_hi_80", "y_lo_95", "y_hi_95"))
        assert all(len(a) == H for a in (lo80, hi80, lo95, hi95)), r["model"]
        assert (lo95 >= 0).all() and (lo95 <= lo80 + 1e-9).all() and (lo80 <= hi80).all() and (hi80 <= hi95 + 1e-9).all()
        np.testing.assert_allclose(dict(r["y_quantiles"])["0.975"], hi95)


def test_the_tables_of_run_forecast_are_not_modified(spark, calibrated):
    _, before = calibrated
    assert _snapshot(spark) == before


def test_a_rerun_appends_with_a_new_calibration_date_and_models_select(spark, calibrated):
    out, _ = calibrated
    again = calibrate_intervals(spark, train_data="pi_cal_train", evaluation_output=f"{PREFIX}_evaluation_output",
                                scoring_output=f"{PREFIX}_scoring_output", freq="D", models=[NO_NATIVE],
                                method="conformal")
    assert list(again["models"]) == [NO_NATIVE]
    score = spark.table(out["tables"]["scoring_intervals"]).filter(F.col("model") == NO_NATIVE)
    assert score.select("calibration_date").distinct().count() == 2 and score.count() == 20


def test_run_ids_that_do_not_exist_calibrate_nothing(spark, calibrated, caplog):
    with caplog.at_level("WARNING"):
        out = calibrate_intervals(spark, train_data="pi_cal_train", evaluation_output=f"{PREFIX}_evaluation_output",
                                  scoring_output=f"{PREFIX}_scoring_output", freq="D", run_ids={NATIVE: "nope"},
                                  models=[NATIVE])
    assert out["models"] == {} and "No backtest rows" in caplog.text


def test_pool_and_group_col_go_together(spark):
    from mmf_sa.exceptions import ConfigurationError
    kwargs = dict(train_data="pi_cal_train", evaluation_output=f"{PREFIX}_evaluation_output",
                  scoring_output=f"{PREFIX}_scoring_output")
    with pytest.raises(ConfigurationError, match="pool='group' needs group_col"):
        calibrate_intervals(spark, pool="group", **kwargs)
    with pytest.raises(ConfigurationError, match="pool='group' needs group_col"):
        calibrate_intervals(spark, group_col="auto", **kwargs)
    with pytest.raises(ConfigurationError, match="pool must be"):
        calibrate_intervals(spark, pool="series", **kwargs)


def test_pool_group_pools_per_abc_xyz_class(spark, forecast):
    out = calibrate_intervals(spark, train_data="pi_cal_train", evaluation_output=f"{PREFIX}_evaluation_output",
                              scoring_output=f"{PREFIX}_scoring_output", output_prefix="default.pi_cal_grouped",
                              models=[NATIVE, NO_NATIVE], pool="group", group_col="auto")
    assert out["models"][NATIVE]["methods"]["80"] == "cqr/asymmetric/group"
    assert out["models"][NO_NATIVE]["methods"]["80"] == "conformal/signed_normalized/series"
    pools = spark.table("default.pi_cal_grouped_evaluation_output_calibration").select("pool_key").distinct().count()
    assert pools >= 1

