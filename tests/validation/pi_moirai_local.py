"""Moirai native quantiles with the real uni2ts 2.0.0 models, on CPU, on the 200-series M5 sample of
pi_native_models.py.

Moirai needs torch<2.5, so it cannot run on DBR 18 ML, and on DBR 15.4 ML the mmf_sa install breaks the runtime
(see results/pi_native_models_200.json). MMF's Moirai predict is a Spark Pandas UDF that requires CUDA, so this
script calls the models exactly as that UDF does (same constructor arguments, context = whole history), packs and
decodes the outputs with MMF's own functions (pack_sample_forecasts, decode_quantile_forecasts), and calibrates with
conformal.calibrate_model, the numpy reference that the Spark calibration matches to 1e-12.

Usage, in a Python 3.11 environment with `uni2ts==2.0.0` and mmf_sa (pip install -e .), with M5 in /tmp/m5
(datasetsforecast's M5.download):
    python tests/validation/pi_moirai_local.py tests/validation/results/pi_moirai_local.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

from mmf_sa.intervals.config import level_quantiles, requested_quantiles, resolve_prediction_intervals_conf
from mmf_sa.intervals.quantiles import QUANTILE_COLUMN_PREFIX, decode_quantile_forecasts, forecast_levels, quantile_key
from mmf_sa.intervals.conformal import calibrate_model, cqr_bounds, finalize_bounds
from mmf_sa.intervals.features import classify_series, positive_sigma, series_features
from mmf_sa.models.moiraiforecast import MoiraiPipeline as mp

M5 = "/tmp/m5/m5/datasets"
N_SERIES, H, STRIDE, W, TEST_WINDOWS = 200, 28, 14, 3, 4
QUANTILES = requested_quantiles(resolve_prediction_intervals_conf({"enabled": True}))
LEVELS = forecast_levels(QUANTILES)
MODELS = {"MoiraiSmall": ("Salesforce/moirai-1.1-R-small", 32), "MoiraiMoESmall": ("Salesforce/moirai-moe-1.0-R-small", 16)}

calendar = pd.read_csv(f"{M5}/calendar.csv", usecols=["date"], parse_dates=["date"])
calendar["d"] = [f"d_{i + 1}" for i in range(len(calendar))]
wide = pd.read_csv(f"{M5}/sales_train_evaluation.csv")
wide["unique_id"] = wide["item_id"] + "_" + wide["store_id"]
wide = wide.sample(n=N_SERIES, random_state=7)
day_cols = [c for c in wide.columns if c.startswith("d_")]
Y = wide.melt(id_vars=["unique_id"], value_vars=day_cols, var_name="d", value_name="y")
Y = Y.merge(calendar, on="d").rename(columns={"date": "ds"}).drop(columns="d")
Y["y"] = Y["y"].astype(float)
Y = Y.sort_values(["unique_id", "ds"])
Y = Y[Y.groupby("unique_id")["y"].cumsum() > 0].reset_index(drop=True)
test_cutoff = Y["ds"].max() - pd.Timedelta(days=TEST_WINDOWS * H)
series = {u: g.set_index("ds")["y"] for u, g in Y.groupby("unique_id")}
ids = sorted(series)
# Backtest cutoffs (oldest first), then the scoring cutoff, whose next 28 days are held out.
cutoffs = [test_cutoff - pd.Timedelta(days=H + k * STRIDE) for k in reversed(range(W))] + [test_cutoff]


def forecast_samples(module, repo, patch_size, num_samples, context):
    cls = MoiraiMoEForecast if "moe" in repo else MoiraiForecast
    model = cls(module=module, prediction_length=H, context_length=len(context),
                patch_size=16 if "moe" in repo else patch_size, num_samples=num_samples,
                target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    past_target = rearrange(torch.as_tensor(context, dtype=torch.float32), "t -> 1 t 1")
    with torch.no_grad():
        out = model(past_target=past_target, past_observed_target=torch.ones_like(past_target, dtype=torch.bool),
                    past_is_pad=torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1))
    return out[0].cpu().numpy()  # what the MMF UDF passes to pack_sample_forecasts


def run(name, num_samples):
    repo, patch_size = MODELS[name]
    module = (MoiraiMoEModule if "moe" in repo else MoiraiModule).from_pretrained(repo)
    torch.manual_seed(0)
    t0 = time.time()
    point, bands, actual, shapes, crossing, roundtrip = [], [], [], set(), [], []
    for cutoff in cutoffs:
        rows = []
        for u in ids:
            s = series[u]
            hist, fut = s[s.index <= cutoff].to_numpy(), s[s.index > cutoff].to_numpy()[:H]
            samples = forecast_samples(module, repo, patch_size, num_samples, hist)
            shapes.add(samples.shape)
            packed = mp.pack_sample_forecasts(samples, QUANTILES)
            plain = mp.pack_sample_forecasts(samples, None)  # the point forecast without native quantiles
            rows.append({"y": packed, "plain": plain, "actual": fut})
        df = decode_quantile_forecasts(pd.DataFrame(rows), "y", LEVELS, H)
        roundtrip.append(max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(df["y"], df["plain"])))
        q = np.stack([np.stack(df[QUANTILE_COLUMN_PREFIX + quantile_key(lv)]) for lv in LEVELS], 1)  # N,Q,H
        crossing.append(float((np.diff(q, axis=1) < -1e-9).any(axis=1).mean()))
        point.append(np.stack(df["y"]))
        bands.append({lv: (q[:, LEVELS.index(level_quantiles(lv)[0])], q[:, LEVELS.index(level_quantiles(lv)[1])])
                      for lv in (80, 95)})
        actual.append(np.stack(df["actual"]))
    runtime = round(time.time() - t0, 1)

    # Calibration on the backtest windows, as run_calibration does (series features before the first window).
    feats = pd.DataFrame([series_features(np.nan_to_num(series[u][series[u].index <= cutoffs[0]].to_numpy())[-30:],
                                              season_length=7) for u in ids], index=ids)
    classes = classify_series(feats)
    sigma, _ = positive_sigma(classes["sigma"].to_numpy())
    yb, fb = np.stack(actual[:W], 1), np.stack(point[:W], 1)  # N, W, H
    native = {lv: (np.maximum(np.stack([b[lv][0] for b in bands[:W]], 1), 0.0), np.stack([b[lv][1] for b in bands[:W]], 1))
              for lv in (80, 95)}
    pi = resolve_prediction_intervals_conf({"enabled": True})
    res = calibrate_model(yb, fb, sigma, pi, labels=classes["abc_xyz"].to_numpy(), native=native, clip_lower=0.0,
                              model=name)
    out = {"num_samples": num_samples, "runtime_s": runtime, "sample_shapes": sorted(map(str, shapes)),
           "decode_roundtrip_max_diff": max(roundtrip), "crossing_share": round(float(np.mean(crossing)), 4)}
    yt, ft = actual[W][:, None, :], point[W][:, None, :]
    for lv in (80, 95):
        r = res[lv]
        lo_t, hi_t = np.maximum(bands[W][lv][0], 0.0)[:, None, :], bands[W][lv][1][:, None, :]
        lo, hi = finalize_bounds(*cqr_bounds(ft, lo_t, hi_t, sigma, r["q_lo"], r["q_hi"], r["score"]), 0.0)
        width_native = float((native[lv][1] - native[lv][0]).mean())
        out[str(lv)] = {
            "method": f"{r['method']}/{r['score']}/{r['target']}",
            "lowo_native_coverage": round(float(r["native_metrics"]["coverage"].mean()), 3),
            "lowo_coverage": round(float(r["metrics"]["coverage"].mean()), 3),
            "heldout_native_coverage": round(float(((yt >= lo_t) & (yt <= hi_t)).mean()), 3),
            "heldout_coverage": round(float(((yt >= lo) & (yt <= hi)).mean()), 3),
            "native_width": round(width_native, 3),
        }
    out["width_ratio_95_80"] = round(out["95"]["native_width"] / out["80"]["native_width"], 3)
    return out


results = {}
for name, num_samples in [("MoiraiSmall", 10), ("MoiraiMoESmall", 10), ("MoiraiSmall", 100)]:
    key = f"{name}_s{num_samples}"
    results[key] = run(name, num_samples)
    print(key, json.dumps(results[key]), flush=True)
json.dump({"series": len(ids), "quantiles": QUANTILES, "torch": torch.__version__, "results": results},
          open(sys.argv[1], "w"), indent=1)
