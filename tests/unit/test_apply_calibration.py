"""Unit tests for applying the calibration at scoring time (prediction intervals).

Tests cover:
- Bounds per method: residual conformal (signed_normalized and signed), CQR asymmetric and
  scaled, native; clipping and uncrossing
- Fallback for series without calibration rows, and null bounds without a native band
- output_format columns (quantile_map, lo_hi, both) and their names
- The local scoring UDF with native quantiles, and statsforecast forecast() quantiles
"""
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mmf_sa import Forecaster
from mmf_sa.intervals.quantiles import QUANTILE_COLUMN_PREFIX
from mmf_sa.intervals.scoring import (
    CORRECTION_FIELDS, apply_calibration, apply_joined_corrections, correction_column, interval_columns,
    median_corrections,
)
from mmf_sa.models.statsforecast.StatsFcForecastingPipeline import StatsFcAutoETS

H = 3


def _cal_rows(method, score, series=("a", "b"), levels=(80,), q_lo=-1.0, q_hi=2.0, sigma=0.5, clip=0.0):
    rows = []
    for sid in series:
        for lv in levels:
            for step in range(1, H + 1):
                rows.append({"uid": sid, "level": float(lv), "step": step, "q_lo": q_lo * step, "q_hi": q_hi * step,
                             "sigma": sigma, "method": method, "score": score, "clip_lower": clip})
    return pd.DataFrame(rows)


def _forecasts(native=None):
    return pd.DataFrame({
        "uid": ["a", "b"],
        "ds": [np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")] * 2,
        "y": [np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.2, 0.2])],
        "forecast_quantiles": native if native is not None else [None, None],
    })


def test_conformal_bounds():
    out, stats = apply_calibration(_forecasts(), _cal_rows("conformal", "signed_normalized"), "uid", "y", [80])
    qa = out.loc[0, "y_quantiles"]
    # lo = yhat + 0.5 * (-1, -2, -3), clipped at 0; hi = yhat + 0.5 * (2, 4, 6)
    np.testing.assert_allclose(qa["0.1"], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(qa["0.9"], [2.0, 4.0, 6.0])
    np.testing.assert_allclose(out.loc[1, "y_quantiles"]["0.1"], [0.0, 0.0, 0.0])  # clipped
    assert "forecast_quantiles" not in out.columns
    assert stats == {"fallback": 0, "null": 0}


def test_signed_score_is_not_scaled():
    out, _ = apply_calibration(_forecasts(), _cal_rows("conformal", "signed"), "uid", "y", [80])
    np.testing.assert_allclose(out.loc[0, "y_quantiles"]["0.9"], [3.0, 6.0, 9.0])


def _native():
    band = {"0.1": [0.5, 1.0, 2.0], "0.9": [2.0, 3.0, 4.0], "0.5": [1.0, 2.0, 3.0],
            "0.025": [-1.0, 0.0, 1.0], "0.975": [3.0, 4.0, 5.0]}
    return [band, [(k, v) for k, v in band.items()]]  # dict, and (key, value) pairs as some Arrow versions return


def test_cqr_asymmetric_bounds():
    out, _ = apply_calibration(_forecasts(_native()), _cal_rows("cqr", "asymmetric", q_lo=0.0, q_hi=1.0),
                                   "uid", "y", [80])
    qa = out.loc[0, "y_quantiles"]
    np.testing.assert_allclose(qa["0.1"], [0.5, 1.0, 2.0])        # lo - sigma * 0
    np.testing.assert_allclose(qa["0.9"], [2.5, 4.0, 5.5])        # hi + 0.5 * (1, 2, 3)
    np.testing.assert_allclose(qa["0.5"], [1.0, 2.0, 3.0])        # native median is kept
    assert list(qa) == ["0.1", "0.5", "0.9"]


def test_cqr_clips_the_native_band_first():
    out, _ = apply_calibration(_forecasts(_native()), _cal_rows("cqr", "asymmetric", levels=(95,), q_lo=0.0, q_hi=0.0),
                                   "uid", "y", [95])
    np.testing.assert_allclose(out.loc[0, "y_quantiles"]["0.025"], [0.0, 0.0, 1.0])


def test_cqr_scaled_bounds():
    out, _ = apply_calibration(_forecasts(_native()), _cal_rows("cqr", "scaled", q_lo=1.0, q_hi=1.0),
                                   "uid", "y", [80])
    half = np.array([0.75, 1.0, 1.0])
    yhat = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(out.loc[0, "y_quantiles"]["0.9"], yhat + half * [1, 2, 3])


def test_native_method_passes_the_band():
    out, _ = apply_calibration(_forecasts(_native()), _cal_rows("native", "native", q_lo=0, q_hi=0),
                                   "uid", "y", [80])
    np.testing.assert_allclose(out.loc[1, "y_quantiles"]["0.9"], [2.0, 3.0, 4.0])


def test_crossed_bounds_are_uncrossed():
    rows = _cal_rows("conformal", "signed_normalized", q_lo=-1.0, q_hi=-1.0, sigma=1.0, clip=None)
    rows["clip_lower"] = np.nan
    out, _ = apply_calibration(_forecasts(), rows, "uid", "y", [80])
    qa = out.loc[1, "y_quantiles"]
    assert (np.array(qa["0.9"]) >= np.array(qa["0.1"])).all()
    assert min(qa["0.1"]) < 0  # no clipping without clip_lower


def test_missing_series_uses_the_fallback():
    out, stats = apply_calibration(_forecasts(), _cal_rows("conformal", "signed_normalized", series=("a",)),
                                       "uid", "y", [80])
    assert stats == {"fallback": 1, "null": 0}
    np.testing.assert_allclose(out.loc[1, "y_quantiles"]["0.9"], np.array([0.2] * 3) + 0.5 * np.array([2, 4, 6]))


def test_cqr_without_native_band_gives_null_bounds():
    out, stats = apply_calibration(_forecasts(), _cal_rows("cqr", "asymmetric"), "uid", "y", [80], "both")
    assert stats["null"] == 2
    assert out["y_quantiles"].isna().all() and out["y_lo_80"].isna().all()


def test_no_calibration_gives_null_columns():
    out, _ = apply_calibration(_forecasts(), pd.DataFrame(), "uid", "y", [80], "both")
    assert out["y_quantiles"].isna().all() and out["y_hi_80"].isna().all()


@pytest.mark.parametrize("fmt, columns", [
    ("quantile_map", ["y_quantiles"]),
    ("lo_hi", ["y_lo_80", "y_hi_80", "y_lo_97_5", "y_hi_97_5"]),
    ("both", ["y_quantiles", "y_lo_80", "y_hi_80", "y_lo_97_5", "y_hi_97_5"]),
])
def test_output_formats(fmt, columns):
    assert interval_columns("y", [80, 97.5], fmt) == columns
    rows = _cal_rows("conformal", "signed_normalized", levels=(80, 97.5))
    out, _ = apply_calibration(_forecasts(), rows, "uid", "y", [80, 97.5], fmt)
    assert [c for c in out.columns if c.startswith("y_")] == columns
    if fmt != "quantile_map":
        np.testing.assert_allclose(out.loc[0, "y_hi_80"], out.loc[0, "y_hi_80"])
        assert len(out.loc[0, "y_lo_97_5"]) == H


def _ets(quantiles=None):
    params = dict(group_id="uid", date_col="date", target="y", freq="D", prediction_length=7, metric="smape",
                  stride=7, backtest_length=14, model_spec={"season_length": 7, "model": "ZNZ"})
    if quantiles:
        params["native_quantiles"] = quantiles
    return StatsFcAutoETS(OmegaConf.create(params))


def _series(n=120):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"uid": "a", "date": pd.date_range("2023-01-01", periods=n),
                         "y": rng.poisson(3, n).astype(float)})


def test_statsforecast_forecast_quantiles():
    plain, _ = _ets().forecast(_series())
    with_q, _ = _ets([0.1, 0.5, 0.9]).forecast(_series())
    np.testing.assert_array_equal(plain["y"], with_q["y"])
    assert [c for c in with_q.columns if c.startswith(QUANTILE_COLUMN_PREFIX)] == ["__q_0.1", "__q_0.5", "__q_0.9"]


@pytest.mark.parametrize("with_quantiles", [False, True])
def test_local_scoring_udf(with_quantiles):
    out = Forecaster.score_one_local_model(_series(), _ets([0.1, 0.5, 0.9]), with_quantiles=with_quantiles)
    assert ("forecast_quantiles" in out.columns) == with_quantiles
    if with_quantiles:
        fq = out["forecast_quantiles"].iloc[0]
        assert list(fq) == ["0.1", "0.5", "0.9"] and len(fq["0.9"]) == 7


def _joined(cal_rows, forecasts, levels):
    """The wide correction columns that Forecaster.wide_corrections joins to the forecasts."""
    out = forecasts.copy()
    for j, lv in enumerate(sorted(levels)):
        for field in CORRECTION_FIELDS:
            values = []
            for sid in out["uid"]:
                g = cal_rows[(cal_rows["uid"] == sid) & (cal_rows["level"] == float(lv))].sort_values("step")
                if g.empty:
                    values.append(None)
                elif field in ("q_lo", "q_hi"):
                    values.append(g[field].to_numpy())
                else:
                    values.append(g[field].iloc[0])
            out[correction_column(j, field)] = values
    return out


@pytest.mark.parametrize("method, score, native", [
    ("conformal", "signed_normalized", None),
    ("cqr", "asymmetric", "native"),
    ("cqr", "scaled", "native"),
])
def test_joined_corrections_match_reference(method, score, native):
    rows = _cal_rows(method, score, series=("a",), levels=(80, 95), q_lo=0.5, q_hi=1.0)
    forecasts = _forecasts(_native() if native else None)
    expected, stats = apply_calibration(forecasts, rows, "uid", "y", [80, 95], "both")
    stats_rows = []
    for lv, g in rows.groupby("level"):
        for step, gs in g.groupby("step"):
            stats_rows.append({"level": lv, "step": step, "q_lo": gs["q_lo"].median(), "q_hi": gs["q_hi"].median(),
                               "sigma": g["sigma"].median(), "method": method, "score": score})
    fallback = median_corrections(pd.DataFrame(stats_rows))
    got, _ = apply_joined_corrections(_joined(rows, forecasts, [80, 95]), "uid", "y", [80, 95], "both",
                                          0.0, fallback)
    assert stats == {"fallback": 1, "null": 0}  # series "b" has no rows
    assert list(got.columns) == list(expected.columns)
    for c in ("y_lo_80", "y_hi_95"):
        for a, b in zip(got[c], expected[c]):
            np.testing.assert_allclose(a, b)
    for a, b in zip(got["y_quantiles"], expected["y_quantiles"]):
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_allclose(a[k], b[k])
