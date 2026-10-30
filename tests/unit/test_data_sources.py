"""Unit tests for the Forecaster's data sources (train_data / scoring_data as tables or DataFrames).

Tests cover:
- resolve_source with a DataFrame, a table name, and the two mixed
- has_separate_scoring_data for every combination, including the same table or DataFrame
"""
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mmf_sa import Forecaster


class _FakeReader:
    def table(self, name):
        return f"table:{name}"


class _FakeSpark:
    read = _FakeReader()

    def createDataFrame(self, pdf):
        return ("spark", id(pdf))


TRAIN = pd.DataFrame({"y": [1.0]})
SCORE = pd.DataFrame({"y": [2.0]})


def _forecaster(conf=None, data_conf=None):
    base = {"freq": "D", "backtest_length": 7, "active_models": []}
    base.update(conf or {})
    return Forecaster(conf=OmegaConf.create(base), data_conf=data_conf or {}, spark=_FakeSpark(), experiment_id="e")


def test_resolve_source_tables_dataframes_and_mixed():
    f = _forecaster({"scoring_data": "cat.db.score"}, {"train_data": TRAIN})
    assert f.resolve_source("train_data") == ("spark", id(TRAIN))
    assert f.resolve_source("scoring_data") == "table:cat.db.score"
    g = _forecaster({"train_data": "cat.db.train", "scoring_data": "cat.db.score"})
    assert g.resolve_source("train_data") == "table:cat.db.train"


@pytest.mark.parametrize("conf, data_conf, expected", [
    ({}, {"train_data": TRAIN}, False),                                              # no scoring data
    ({"train_data": "t"}, {}, False),
    ({"train_data": "t", "scoring_data": "t"}, {}, False),                          # same table
    ({"train_data": "t", "scoring_data": "s"}, {}, True),
    ({}, {"train_data": TRAIN, "scoring_data": TRAIN}, False),                      # same DataFrame
    ({}, {"train_data": TRAIN, "scoring_data": SCORE}, True),
    ({"scoring_data": "s"}, {"train_data": TRAIN}, True),                           # mixed
    ({"train_data": "t"}, {"scoring_data": SCORE}, True),
])
def test_has_separate_scoring_data(conf, data_conf, expected):
    assert _forecaster(conf, data_conf).has_separate_scoring_data() is expected
