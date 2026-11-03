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
import uuid
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from pyspark.sql import functions as F

from mmf_sa import calibrate_intervals, run_forecast
from mmf_sa.exceptions import ConfigurationError
from mmf_sa.intervals import spark as calibration_spark
from mmf_sa.intervals.calibration_config import resolve_prediction_intervals_conf
from mmf_sa.intervals.calibrator import IntervalCalibrator

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


def _calibrator(spark, levels=(80,)):
    pi = resolve_prediction_intervals_conf({"enabled": True, "levels": list(levels)})
    return IntervalCalibrator(spark, OmegaConf.create({"target": "y", "prediction_intervals": pi}),
                              train_data=pd.DataFrame())


def test_native_map_falls_back_to_the_bounds_row_by_row(spark_session):
    """An all-NULL y_quantiles column (pre-created, or left by an earlier quantile_map run) must not hide the bounds."""
    schema = "uid string, y_quantiles map<string,array<double>>, y_lo_80 array<double>, y_hi_80 array<double>"
    sdf = spark_session.createDataFrame([
        ("map", {"0.1": [0.5], "0.9": [4.0]}, [1.0], [3.0]),
        ("nulls", None, [1.0], [3.0]),
        ("nothing", None, None, None),
    ], schema)
    native = _calibrator(spark_session).native_scoring_map(sdf)
    got = {r["uid"]: {k: v for k, v in (r["native"] or {}).items() if v is not None}
           for r in sdf.select("uid", native.alias("native")).collect()}
    assert got == {"map": {"0.1": [0.5], "0.9": [4.0]}, "nulls": {"0.1": [1.0], "0.9": [3.0]}, "nothing": {}}


def _model_output(spark, model):
    """One calibration row and one metric row of a model, as calibrate_model_spark returns them."""
    schema = calibration_spark.output_schema("uid")
    cal = {"kind": "calibration", "model": model, "method": "conformal", "score": "signed", "coverage_target": "series",
           "uid": "s0", "pool_key": "__all__", "step": 1, "level": 80.0, "q_lo": -1.0, "q_hi": 1.0, "n_windows": 2,
           "shrinkage_weight": 0.5, "infinite_bound": False, "clip_lower": 0.0, "sigma": 1.0, "abc_class": "A",
           "xyz_class": "X", "sb_class": "smooth"}
    metric = {"kind": "metric", "model": model, "uid": "s0", "window_date": datetime(2024, 1, 1),
              "metric_name": "scaled_ae", "metric_value": 0.5}
    return spark.createDataFrame([tuple(r.get(f.name) for f in schema.fields) for r in (cal, metric)], schema)


def test_a_model_whose_write_fails_does_not_stop_the_others(spark_session, monkeypatch, caplog):
    """
    A failing write of one model is logged and recorded in its summary; the other model is still written. The
    scoring intervals are stubbed (their inputs are a full run_forecast), the calibration and metric tables are real.
    """
    prefix = f"default.pi_bug3_{uuid.uuid4().hex[:8]}"
    runs = {"A": {"run_id": "run-a", "run_date": datetime(2024, 1, 2), "metric_name": "rmse"},
            "B": {"run_id": "run-b", "run_date": datetime(2024, 1, 2), "metric_name": "rmse"}}
    backtest = spark_session.createDataFrame([("s0", datetime(2024, 1, 1))],
                                             "uid string, backtest_window_start_date timestamp")
    monkeypatch.setattr(IntervalCalibrator, "latest_runs", lambda self: runs)
    monkeypatch.setattr(IntervalCalibrator, "backtest_rows", lambda self, model, run: backtest)
    monkeypatch.setattr(IntervalCalibrator, "calibration_features", lambda self, first_window, materializer: (
        backtest, 0.0))
    monkeypatch.setattr(calibration_spark, "calibrate_model_spark", lambda spark, rows, features, gid, model, pi,
                        clip_lower, materializer: (_model_output(spark, model), {"method": "conformal"}))
    monkeypatch.setattr(IntervalCalibrator, "write_scoring_intervals", lambda self, model, run, cal: 1)
    real_write = IntervalCalibrator.write_interval_metrics

    def failing_write(self, rows, run):
        if run["run_id"] == "run-a":
            raise OSError("metrics table is locked")
        return real_write(self, rows, run)

    monkeypatch.setattr(IntervalCalibrator, "write_interval_metrics", failing_write)
    conf = OmegaConf.create({
        "evaluation_output": "unused", "scoring_output": "unused", "train_data": None, "output_prefix": prefix,
        "group_id": "uid", "date_col": "ds", "target": "y", "freq": "D", "models": None, "run_ids": None,
        "prediction_intervals": resolve_prediction_intervals_conf({"enabled": True}),
    })
    with caplog.at_level("ERROR"):
        out = IntervalCalibrator(spark_session, conf, train_data=pd.DataFrame()).run()

    assert "metrics table is locked" in out["models"]["A"]["error"]
    assert "error" not in out["models"]["B"] and out["models"]["B"]["run_id"] == "run-b"
    assert "Calibration of A failed" in caplog.text
    written = spark_session.table(out["tables"]["interval_metrics"]).select("run_id").distinct().collect()
    assert [r["run_id"] for r in written] == ["run-b"]
    assert spark_session.table(out["tables"]["calibration"]).filter(F.col("model") == "B").count() == 1


@pytest.fixture
def configured(monkeypatch):
    """calibrate_intervals up to the calibration: returns the resolved prediction_intervals block, no Spark work."""
    seen = {}

    def run(self):
        seen["pi"] = self.settings.prediction_intervals
        return {}

    monkeypatch.setattr(IntervalCalibrator, "run", run)

    def resolve(**kwargs):
        calibrate_intervals(None, train_data=None, evaluation_output=f"{PREFIX}_evaluation_output",
                            scoring_output=f"{PREFIX}_scoring_output", **kwargs)
        return seen["pi"]

    return resolve


def test_settings_are_used_when_the_arguments_are_left_at_their_defaults(configured):
    pi = configured(settings={"method": "cqr", "output_format": "lo_hi", "method_by_model": {NO_NATIVE: "conformal"},
                              "cqr_pool": "series", "coverage": {"shrinkage_kappa": 20, "group_col": None}})
    assert (pi.method, pi.output_format, pi.cqr_pool) == ("cqr", "lo_hi", "series")
    assert pi.method_by_model[NO_NATIVE] == "conformal"
    assert pi.coverage.shrinkage_kappa == 20


def test_levels_in_settings_are_used_without_an_argument(configured):
    assert configured(settings={"levels": [90]}).levels == [90]


def test_the_argument_wins_over_a_setting_only_when_they_agree(configured):
    assert configured(method="cqr", settings={"method": "cqr"}).method == "cqr"
    assert configured(output_format="lo_hi", settings={"output_format": "lo_hi"}).output_format == "lo_hi"


@pytest.mark.parametrize("kwargs, key", [
    ({"method": "conformal", "settings": {"method": "cqr"}}, "method"),
    ({"output_format": "both", "settings": {"output_format": "lo_hi"}}, "output_format"),
    ({"method_by_model": {NO_NATIVE: "cqr"}, "settings": {"method_by_model": {NO_NATIVE: "conformal"}}},
     "method_by_model"),
    ({"levels": [80], "settings": {"levels": [90]}}, "levels"),
    ({"pool": "group", "group_col": "auto", "settings": {"cqr_pool": "marginal"}}, "cqr_pool"),
    ({"shrinkage_kappa": 5, "settings": {"coverage": {"shrinkage_kappa": 20}}}, "shrinkage_kappa"),
])
def test_arguments_and_settings_that_disagree_raise(configured, kwargs, key):
    with pytest.raises(ConfigurationError, match=key):
        configured(**kwargs)


def test_levels_and_quantiles_together_raise(configured):
    with pytest.raises(ConfigurationError, match="levels or quantiles, not both"):
        configured(levels=[80], quantiles=[0.1, 0.5, 0.9])
    with pytest.raises(ConfigurationError, match="levels or quantiles, not both"):
        configured(quantiles=[0.1, 0.5, 0.9], settings={"levels": [80]})


def test_use_case_calibrates_only_the_rows_of_that_use_case(spark_session, monkeypatch):
    """Two use cases in one evaluation_output and scoring_output: use_case picks the rows (and the run) of one."""
    uid = uuid.uuid4().hex[:8]
    ev_view, sc_view = f"pi_uc_ev_{uid}", f"pi_uc_sc_{uid}"
    ev_schema = ("model string, run_id string, run_date timestamp, metric_name string, unique_id string, "
                 "backtest_window_start_date timestamp, forecast array<double>, actual array<double>, use_case string")
    ds = datetime(2024, 1, 1)
    ev = [("M", "run-a", datetime(2024, 1, 2), "rmse", f"s{i}", ds, [1.0], [1.0], "A") for i in range(2)]
    ev += [("M", "run-b", datetime(2024, 2, 2), "rmse", f"s{i}", ds, [1.0], [1.0], "B") for i in range(3)]
    sc_schema = "unique_id string, ds timestamp, y double, model string, run_id string, run_date timestamp, use_case string"
    sc = [("s0", ds, 1.0, "M", "run-a", datetime(2024, 1, 2), "A") for _ in range(2)]
    sc += [("s0", ds, 1.0, "M", "run-b", datetime(2024, 2, 2), "B") for _ in range(3)]
    spark_session.createDataFrame(ev, ev_schema).createOrReplaceTempView(ev_view)
    spark_session.createDataFrame(sc, sc_schema).createOrReplaceTempView(sc_view)
    seen = []

    def record(self, model, run, backtest, features, clip_lower, materializer):
        seen.append((model, run["run_id"], backtest.count()))
        return {"run_id": run["run_id"], "methods": {}}

    monkeypatch.setattr(IntervalCalibrator, "calibration_features", lambda self, first_window, materializer: (None, 0.0))
    monkeypatch.setattr(IntervalCalibrator, "calibrate_one", record)
    kwargs = dict(train_data=pd.DataFrame(), evaluation_output=ev_view, scoring_output=sc_view,
                  output_prefix=f"default.pi_uc_{uid}")
    try:
        calibrate_intervals(spark_session, use_case="A", **kwargs)
        assert seen == [("M", "run-a", 2)]
        seen.clear()
        calibrate_intervals(spark_session, **kwargs)  # no use_case: the latest run of the table, of either use case
        assert seen == [("M", "run-b", 3)]

        conf = OmegaConf.create({"evaluation_output": ev_view, "scoring_output": sc_view, "group_id": "unique_id",
                                 "date_col": "ds", "target": "y", "use_case": "A",
                                 "prediction_intervals": resolve_prediction_intervals_conf({"enabled": True})})
        run_a = {"run_id": "run-a", "run_date": datetime(2024, 1, 2)}
        assert IntervalCalibrator(spark_session, conf, train_data=pd.DataFrame()).scoring_rows("M", run_a).count() == 2
        conf.use_case = "B"
        assert IntervalCalibrator(spark_session, conf, train_data=pd.DataFrame()).scoring_rows("M", run_a).count() == 0
    finally:
        spark_session.catalog.dropTempView(ev_view)
        spark_session.catalog.dropTempView(sc_view)
