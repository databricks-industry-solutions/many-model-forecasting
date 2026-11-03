"""Builds tests/unit/data/calibration_golden.npz from the reference experiment notebook.

The experiment notebook (the research that chose the prediction-interval defaults) is not part of
this repository; pass its path as the only argument.

The fixture holds the backtest cubes of a small M5 sample and the metrics that the notebook
computed from them, so that test_calibration.py can check mmf_sa.intervals.conformal against the
notebook without statsforecast or the M5 data.

Usage (needs statsforecast==2.0.0 and datasetsforecast==0.0.8; downloads M5 to /tmp/m5):
    python tests/unit/data/make_calibration_golden.py <path to the experiment notebook>

MIN_SCORES_PER_GROUP is set to 1, so that no group is merged into "__other__": with 30
series every group would be merged otherwise, and the notebook's per-series step profile
uses the unmerged labels (mmf_sa.intervals.conformal uses the merged ones).
"""
import json
import multiprocessing
import pathlib
import sys

import numpy as np

N_SERIES = 30
SEED = 7
OUT = pathlib.Path(__file__).with_name("calibration_golden.npz")


class _Widgets:
    def __init__(self, values):
        self.values = values

    def text(self, key, default):
        self.values.setdefault(key, default)

    def get(self, key):
        return self.values[key]


class _Notebook:
    def exit(self, value):
        raise SystemExit(0)


class _DBUtils:
    widgets = _Widgets({"n_series": str(N_SERIES), "seed": str(SEED),
                        "models": "SeasonalNaive,CrostonSBA,AutoETS", "foundation": ""})
    notebook = _Notebook()


class _GroupedFrame:
    """Just enough of a Spark DataFrame for the notebook's groupBy().applyInPandas()."""

    def __init__(self, pdf, col=None):
        self.pdf, self.col = pdf, col

    def groupBy(self, col):
        return _GroupedFrame(self.pdf, col)

    def applyInPandas(self, fn, schema):
        import pandas as pd
        return _GroupedFrame(pd.concat([fn(g.copy()) for _, g in self.pdf.groupby(self.col)], ignore_index=True))

    def toPandas(self):
        return self.pdf


class _Spark:
    def createDataFrame(self, pdf):
        return _GroupedFrame(pdf)


def main(notebook_path):
    multiprocessing.set_start_method("fork", force=True)
    src = pathlib.Path(notebook_path).read_text()
    src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("# MAGIC"))
    src = src.replace("dbutils.library.restartPython()", "pass")
    assert src.count("MIN_SCORES_PER_GROUP = 100") == 1
    src = src.replace("MIN_SCORES_PER_GROUP = 100", "MIN_SCORES_PER_GROUP = 1")
    g = {"spark": _Spark(), "dbutils": _DBUtils(), "display": lambda df: None, "__name__": "__main__"}
    try:
        exec(compile(src, notebook_path, "exec"), g)
    except SystemExit:
        pass

    ids, F, H = g["ids"], g["F"], g["H"]
    hist = g["hist"]
    hist_y = np.stack([hist.loc[hist["unique_id"] == u, "y"].to_numpy() for u in ids])
    arrays = {
        "hist_y": hist_y,
        "Y_cal": g["Y_cal"],
        "Y_test": g["Y_test"],
        "sigma": g["sigma"],
        "sb": F.loc[ids, "sb"].to_numpy().astype(str),
        "abc": F.loc[ids, "abc"].to_numpy().astype(str),
        "xyz": F.loc[ids, "xyz"].to_numpy().astype(str),
        "abc_xyz": F.loc[ids, "abc_xyz"].to_numpy().astype(str),
    }
    to_cube, cal_cv, test_cv = g["to_cube"], g["cal_cv"], g["test_cv"]
    native_cv, native_cal_cv = g["native_cv"], g["native_cal_cv"]
    for model in ("CrostonSBA", "AutoETS"):
        arrays[f"{model}_cal"] = to_cube(cal_cv, model, g["cal_cutoffs"])
        arrays[f"{model}_test"] = to_cube(test_cv, model, g["test_cutoffs"])
    for level in g["LEVELS"]:
        for side in ("lo", "hi"):
            arrays[f"AutoETS_{side}_{level}_cal"] = to_cube(native_cal_cv, f"AutoETS-{side}-{level}", g["cal_cutoffs"])
            arrays[f"AutoETS_{side}_{level}_test"] = to_cube(native_cv, f"AutoETS-{side}-{level}", g["test_cutoffs"])
    keep = ("method", "model", "W", "level", "coverage", "msis", "scaled_width")
    rows = [{k: r[k] for k in keep} for r in g["results"] if r["model"] in ("CrostonSBA", "AutoETS")]
    arrays["expected"] = np.array(json.dumps(rows))
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB, {len(ids)} series, {len(rows)} expected rows, H={H})")


if __name__ == "__main__":
    main(sys.argv[1])
