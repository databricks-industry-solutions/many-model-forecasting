# Spec: A Spark-free test layer for `mmf_sa`

Status: draft / analysis
Author: lbruand
Date: 2026-06-10

## Why are we asking?

Running `uv run --extra tests pytest` today fails immediately because every
unit test transitively requires a running JVM. Even tests that are
algorithmically pure pandas — e.g. `TestDataQualityThresholds.test_default_values`
(`tests/unit/test_data_quality_checks.py:69`) — error out before collection
because `tests/conftest.py:10` and `tests/unit/fixtures.py:24` build a
`SparkSession` at module import. The Spark session is `scope="module",
autouse=True`, so it gets pulled in for any test that lives in that module
tree, including pure unit tests.

Operational consequences:

- Local dev requires a working Java toolchain (JDK 8/11/17, depending on PySpark
  version). Today the repo pins `pyspark==3.3.2`, which is incompatible with
  recent macOS Java releases (`Java gateway process exited before sending its
  port number` is the symptom we observed).
- CI must install a JDK, download Spark JARs, and start a JVM per test
  module — for tests that don't need any of it.
- A run that should take ~1 second per test currently takes 5–30 seconds of
  JVM warmup before a single assertion fires.

The question is: how much of the test suite can be reasonably made
Spark-independent, what work is required, and is the payoff worth it?

## Current state of the test suite

Inventory (`tests/`, ~1130 lines):

| File | Tests | Spark? | Comment |
| --- | --- | --- | --- |
| `tests/integration_test.py` | n/a | yes (Databricks) | uses `dbutils`, only runnable on a Databricks cluster — already broken under local pytest |
| `tests/unit/test_data_quality_checks.py` | 42 | **mostly no** | 30+ tests assert pure-Python logic (`ValidationResult`, `DataQualityThresholds`, `DataQualityMetrics`, `SupportedFrequencies`, `ExternalRegressorTypes`, `DateOffsetUtility`, `ExternalRegressorValidator`). 12 tests instantiate `DataQualityChecks` and pass a `Mock()` whose `.toPandas()` returns a pandas DataFrame — Spark is only used as a *type slot*, not a runtime dependency. |
| `tests/unit/test_exogenous_regressors_pipeline.py` | 1 | no, but autouse fixture forces it | calls `ModelRegistry(...).get_model("StatsForecastAutoArima")` and runs `fit/predict/backtest` over a pandas DataFrame. No Spark in the body. |
| `tests/unit/test_sktime_pipeline.py` | 2 | no, but autouse fixture forces it | same shape — pure pandas, two sktime model wrappers. |
| `tests/unit/test_pipelines.py` | 1 | **yes** | `run_forecast` end-to-end through `Forecaster`, which calls `applyInPandas`, `createDataFrame`, and writes to temp views. Genuinely Spark-bound. |

Of 46 collectible unit tests, **45 are algorithmically Spark-free**. Only
`test_api_func` in `test_pipelines.py` actually exercises Spark code paths.

## What's blocking the no-Spark path

Three root causes, in order of impact:

1. **`tests/conftest.py:10` imports `pyspark`** and re-exports the
   `spark_session` fixture as `autouse=True` at module scope. This means any
   test under `tests/` triggers JVM startup.
2. **`mmf_sa.data_quality_checks.DataQualityChecks.__init__`** declares
   `df: pyspark.sql.DataFrame` in its signature and calls `df.toPandas()` on
   line 225 — and on line 535 calls `self.spark.createDataFrame(clean_df)`
   before returning. Both ends of the function require a `SparkSession`.
   Existing unit tests already work around this with `Mock()` for input, but
   the output path still calls into Spark on the way out (the tests avoid
   that code path by calling private methods directly).
3. **`mmf_sa/__init__.py:11`** imports `pyspark.sql.SparkSession` at module
   import, so `from mmf_sa import run_forecast` (or anything that pulls the
   package) hard-requires PySpark to be installed. The data-quality and
   model-registry submodules are reachable without `mmf_sa.__init__` running,
   but right now Python imports the package on the way to any
   `from mmf_sa.X import Y`.

Worth noting: `mmf_sa/Forecaster.py` and the foundation-model pipelines
(`neuralforecast`, `chronos`, `timesfm`, `moirai`) **are** genuinely
Spark-coupled — they use `applyInPandas`, `TorchDistributor`, and `pandas_udf`
to fan out work across executors. Those are not in scope for the no-Spark
layer; they need to stay tested through the integration test on Databricks.

## Proposal

Split the test suite into two clearly-named directories with separate pytest
configurations:

```
tests/
  no_spark/          # JVM not required, runs in <5s, no Java install
    conftest.py      # no Spark fixtures, just pandas/OmegaConf builders
    test_data_quality_pure.py
    test_validators.py
    test_date_offset.py
    test_sktime_models.py
    test_statsforecast_models.py
  spark/             # JVM required, kept for integration coverage
    conftest.py      # current Spark session fixture, scope=session
    test_forecaster_pipeline.py    # current test_pipelines.py, renamed
    test_data_quality_with_spark.py # the handful that genuinely need Spark
  integration/       # Databricks-only (dbutils), already exists
    integration_test.py
```

Then in `pyproject.toml`:

```toml
[project.optional-dependencies]
tests-no-spark = ["pytest", "pytest-cov", "pyyaml"]
tests = [...current list..., "pyspark==3.3.2", ...]
```

so contributors who only touch data-quality / model-wrapper code can run
`uv run --extra tests-no-spark pytest tests/no_spark` with no JDK installed.

### Concrete work items, in dependency order

| # | Item | Effort | Risk |
| --- | --- | --- | --- |
| 1 | Move `pyspark` import in `mmf_sa/__init__.py` behind `TYPE_CHECKING` / lazy-import inside `run_forecast` | S (~30 min) | low — `run_forecast` is the only thing that actually instantiates Spark types |
| 2 | Split `DataQualityChecks.run` so the pandas pipeline (`_run_group_checks` and friends) is callable on a pandas-in / pandas-out boundary. The Spark→pandas conversion at the top and pandas→Spark conversion at the bottom become thin adapters. | M (~2–4 hrs) | low — refactor is mechanical; covered by existing tests |
| 3 | Reorganise `tests/` into `no_spark/`, `spark/`, `integration/`. Move the 30+ pure tests from `test_data_quality_checks.py` to `tests/no_spark/`, leave the 12 mock-Spark tests where they are (or rewrite them against the new pandas API). Move `test_sktime_pipeline.py` and `test_exogenous_regressors_pipeline.py` to `tests/no_spark/` (they don't use Spark; the fixture currently does). | M (~1–2 hrs) | low |
| 4 | Add `tests-no-spark` extra in `pyproject.toml`, add a `pytest.ini` per-folder or markers (`-m "not spark"`) to scope. | S (~30 min) | low |
| 5 | Update CI to run `tests/no_spark` on every PR (fast lane), `tests/spark` on PRs touching `Forecaster.py` or foundation-model code, and `tests/integration` only on a scheduled Databricks job. | S–M depending on CI shape | medium — depends on whether GH Actions has the right runners |

**Total effort estimate**: ~1 engineer-day for the refactor + reorganisation,
plus whatever CI work is appropriate.

## Does it make sense?

Yes — for three reasons:

1. **The dependency is mostly accidental.** 45 of 46 unit tests don't
   exercise Spark; they're held hostage by a single autouse fixture and one
   ambient import. Removing it doesn't reduce coverage, it surfaces the
   coverage that's already there.
2. **It unblocks the most common contributor change.** Most PRs to this
   repo touch model wrappers (`mmf_sa/models/*`) or data quality logic, not
   distributed orchestration. Today those PRs can't run any tests locally
   without provisioning Java + Spark.
3. **It clarifies what the integration test is actually for.** Right now
   `test_pipelines.py` and the integration test overlap conceptually — both
   are end-to-end. After the split, `tests/spark/` is the
   single-JVM smoke test, and `tests/integration/` is the Databricks-only
   distributed test. Each has a clear role.

## Will it be faster?

Yes, by a meaningful margin:

- **Module import**: the autouse Spark session takes ~3–8 s of JVM warmup
  per test module. Removing it brings test module load to <100 ms.
- **Whole-suite latency on a clean checkout**: ~30 s today (JVM bootstrap
  dominates over the few hundred ms of actual assertions). Expected
  <2 s after the split for `tests/no_spark`.
- **Dependency install**: `pyspark==3.3.2` is ~250 MB and pulls a native
  build. `tests-no-spark` shrinks the install footprint to roughly
  `pytest + pyyaml + omegaconf + pandas + sktime + statsforecast` — which
  is already required by the package itself.
- **CI**: ~10x speedup on the fast lane (no JDK provisioning, no Spark
  download).

## How much effort?

About **1 engineer-day** for the core refactor, plus an additional
half-day for CI plumbing if it needs new workflows. Breakdown:

- 30 min: lazy-import Spark in `mmf_sa/__init__.py`
- 2–4 hrs: split `DataQualityChecks` pandas/spark boundaries
- 1–2 hrs: reorganise tests into `no_spark/spark/integration` layout
- 30 min: `pyproject.toml` extras + pytest markers
- 2–4 hrs: CI workflow updates

## Risks and caveats

- **The 12 mock-Spark tests in `test_data_quality_checks.py`** all rely on
  the `spark_session` fixture being injected, but only as a sentinel —
  they pass it to `DataQualityChecks(...)` so the constructor's
  `_validate_configuration` runs. If item 2 (pandas/Spark boundary split)
  is done first, those tests can be migrated to call the new pandas API
  directly with no Spark session at all.
- **`mmf_sa.Forecaster` is genuinely Spark-coupled** via `applyInPandas`.
  We are explicitly not trying to make it testable without Spark — that's
  what the `tests/spark/` lane is for. Don't be tempted to mock `applyInPandas`;
  it would test the mock, not the code.
- **`integration_test.py` uses `dbutils.library.restartPython()` at import
  time** (`tests/integration_test.py:8`), which makes it uncollectable
  outside Databricks. The reorganisation should at minimum add a
  `pytest.importorskip("dbutils")` or move the import into a fixture, so
  pytest collection on the integration folder doesn't error out.
- **The `m4_df` fixture downloads M4 data into `~`** via
  `datasetsforecast.m4.M4.load`. That's an external network dependency in
  what claims to be a unit test. Worth flagging for a follow-up, but
  orthogonal to the Spark split.

## Recommendation

Do it. The effort is bounded (~1 day), the payoff (local dev unblocked,
~10x CI speedup on the fast lane, clearer test-pyramid story) is high, and
the refactor is mechanical with no functional change to the production
code path. Start with item 2 (the pandas/Spark boundary split in
`data_quality_checks.py`) since it's the only piece that needs design
thought; the rest follows.
