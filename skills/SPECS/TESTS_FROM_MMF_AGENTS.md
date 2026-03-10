# Integrating mmf-agent Tests into mmf-dev-kit

## Context

The `mmf-agent` repository contains a comprehensive tiered test pyramid for the
Many-Model Forecasting skill. This document analyses its test infrastructure and
proposes a plan for integrating it into `mmf-dev-kit`.

---

## mmf-agent Test Inventory

### Tier 0 — Unit Tests (~40 tests, zero infrastructure)

**Location:** `mmf-agent/test/unit/`

| File | Classes | Tests | What it covers |
|------|---------|-------|----------------|
| `test_generators.py` | 11 | ~30 | Base series generation, missing-value injection, outliers, level shifts, irregular timestamps, duplicates, wrong types, mixed-frequency data |
| `test_scenarios.py` | 1 | ~5 | Validates 11 named data scenarios (clean_baseline, moderate_noise, heavy_noise, boundary_missing, over_threshold, all_negative, intermittent, duplicates, wrong_types, short_history, mixed_frequencies) |
| `test_templates.py` | 1 | parametrized | Python syntax validation of all notebook templates |
| `test_skill_frontmatter.py` | 1 | parametrized | YAML frontmatter validation of SKILL.md files |

**Supporting modules:**

- `test/scenarios/generators.py` — Synthetic time-series data generators
- `test/scenarios/scenarios.py` — 11 scenario configurations

**Dependencies:** `numpy`, `pandas`, `sqlglot>=28.0`, `pytest`, `pyyaml`

### Tier 1 — Skill Logic Tests (~14 tests, mock tools)

**Location:** `mmf-agent/test/tier1/`

| File | Classes | Tests | What it covers |
|------|---------|-------|----------------|
| `test_explore_data.py` | 4 | ~12 | Table discovery (SHOW TABLES, DESCRIBE), data profiling, quality checks, frequency-dependent date alignment |
| `test_setup_cluster.py` | 2 | 2 | CPU and GPU cluster recommendations |
| `test_run_mmf.py` | 1 | 3 | Notebook parameter generation for daily, weekly, monthly frequencies |

**Supporting modules:**

- `conftest.py` — Session-scoped and function-scoped fixtures
- `agent_runner.py` — Multi-turn agent loop orchestration (OpenAI client)
- `mock_tools.py` — OpenAI-format tool definitions with handlers
- `duckdb_backend.py` — In-memory DuckDB database; transpiles Databricks SQL to DuckDB SQL via SQLGlot
- `fixtures/databricks_responses.py` — Pre-canned API responses

**Dependencies:** `duckdb`, `openai`, `pytz`, `pytest`, `python-dotenv`

**Infrastructure required:** `DATABRICKS_TOKEN` (calls Claude Sonnet for the agent loop)

### Tier 2 — Workspace Integration Tests (~6 tests, real Databricks)

**Location:** `mmf-agent/test/tier2/`

| File | Classes | Tests | What it covers |
|------|---------|-------|----------------|
| `test_explore_data.py` | 1 | ~4 | End-to-end data exploration against a real Databricks workspace with synthetic data |
| `test_run_mmf.py` | 1 | 1 | End-to-end job submission and execution |

**Supporting modules:**

- `conftest.py` — Session-scoped fixtures (Databricks client, warehouse, catalog/schema lifecycle)
- `databricks_client.py` — REST API client with Bearer token auth
- `real_tools.py` — Real Databricks tool handlers
- Schema isolation: auto-creates `mmf_test_{YYYYMMDD}_{short_uuid}` schemas

**Dependencies:** `openai`, `requests`, `pytest`, `python-dotenv`

**Infrastructure required:** `DATABRICKS_TOKEN`, live SQL warehouse

---

## Comparison of Testing Approaches

| Aspect | mmf-agent | mmf-dev-kit |
|--------|-----------|-------------|
| Test style | Classic pytest (assert-based) | Scorer/evaluation framework |
| Data generation | Synthetic generators + 11 scenarios | Ground-truth YAML |
| Mock infrastructure | DuckDB + SQLGlot transpilation | None (real tools or LLM judges) |
| Agent loop | OpenAI client, multi-turn | MLflow-based trace evaluation |
| Dependencies | numpy, pandas, sqlglot, duckdb, openai | mlflow, pyyaml, databricks-sdk |
| Package layout | `src/mmf_agent/` | `.test/src/skill_test/` |

---

## Integration Plan

### Option A — Port Tier 0 unit tests (low risk, immediate value)

The unit tests are self-contained and require no Databricks infrastructure.

**Steps:**

1. Copy scenario modules into the test package:
   - `mmf-agent/test/scenarios/generators.py` → `.test/src/skill_test/scenarios/generators.py`
   - `mmf-agent/test/scenarios/scenarios.py` → `.test/src/skill_test/scenarios/scenarios.py`

2. Copy unit tests:
   - `mmf-agent/test/unit/test_generators.py` → `.test/tests/unit/test_generators.py`
   - `mmf-agent/test/unit/test_scenarios.py` → `.test/tests/unit/test_scenarios.py`
   - `mmf-agent/test/unit/test_templates.py` → `.test/tests/unit/test_templates.py`
   - `mmf-agent/test/unit/test_skill_frontmatter.py` → `.test/tests/unit/test_skill_frontmatter.py`

3. Add dependencies to `.test/pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   dev = [
       "pytest",
       "pytest-asyncio",
       "numpy",
       "pandas",
       "sqlglot>=28.0",
       "pyyaml",
   ]
   ```

4. Adjust imports in copied files to use the new module paths.

5. Update CI (`.github/workflows/ci.yml`) to run the new tests:
   ```yaml
   - name: Run unit tests
     run: |
       cd .test
       uv sync --extra dev
       uv run pytest tests/ -v -m "not tier1 and not tier2"
   ```

**Result:** ~40 fast, zero-cost tests validating data generators and notebook templates.

### Option B — Port Tier 1 mock infrastructure (high value, moderate effort)

The DuckDB mock backend is the most valuable piece from mmf-agent — it enables
testing skill logic without hitting Databricks.

**Steps:**

1. Copy mock infrastructure into a new subpackage:
   - `duckdb_backend.py` → `.test/src/skill_test/mock/duckdb_backend.py`
   - `mock_tools.py` → `.test/src/skill_test/mock/mock_tools.py`
   - `agent_runner.py` → `.test/src/skill_test/mock/agent_runner.py`
   - `fixtures/databricks_responses.py` → `.test/src/skill_test/mock/fixtures.py`

2. Add a `tier1` optional dependency group to `.test/pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   tier1 = [
       "duckdb",
       "openai",
       "pytz",
       "pytest",
       "python-dotenv",
   ]
   ```

3. Port tier 1 tests:
   - `mmf-agent/test/tier1/test_explore_data.py` → `.test/tests/tier1/test_explore_data.py`
   - `mmf-agent/test/tier1/test_setup_cluster.py` → `.test/tests/tier1/test_setup_cluster.py`
   - `mmf-agent/test/tier1/test_run_mmf.py` → `.test/tests/tier1/test_run_mmf.py`
   - `mmf-agent/test/tier1/conftest.py` → `.test/tests/tier1/conftest.py`

4. Wire the mock DuckDB executor into the existing `skill_test` framework as an
   alternative execution backend (alongside the real Databricks executor). Make sure
   it is easy to merge this part to maintain the branch as it is going 

5. No integration of the tier 1 tests in the CICD, as they would cost too much to run.

**Result:** ~14 skill-logic tests that verify SQL generation, cluster configs,
and notebook params using an in-memory DuckDB backend.

### Option C — Unify evaluation approaches (long-term)

Merge the two testing philosophies:

- Use mmf-dev-kit's **scorer framework** (pattern adherence, fact checking,
  hallucination detection) as the evaluation layer.
- Use mmf-agent's **mock DuckDB backend** as the execution layer.
- Feed mmf-agent's **11 scenario configurations** into `ground_truth.yaml` as
  parametrized test cases, so each scenario gets scored by both classic asserts
  and LLM judges.

This gives the best of both worlds: deterministic unit-level checks plus
LLM-based qualitative evaluation, all runnable without a live Databricks
workspace.

---

## Recommended Sequence

1. **Start with Option A.** Low risk, no refactoring of existing code, adds ~40
   tests to CI immediately.
2. **Follow with Option B.** The DuckDB mock backend fills the biggest gap in
   mmf-dev-kit (no way to test skill logic without real Databricks today).
3. **Pursue Option C** once both test suites coexist, unifying them into a
   single evaluation pipeline.

---

## Scenario Configurations Reference

The 11 scenarios from mmf-agent cover the full spectrum of data-quality edge
cases that the MMF skill must handle:

| Scenario | Missing % | Outliers % | Negatives % | Other characteristics |
|----------|-----------|------------|-------------|----------------------|
| `clean_baseline` | 0 | 0 | 0 | No issues |
| `moderate_noise` | 5 | 1 | 5 | Typical production data |
| `heavy_noise` | 15 | 5 | 15 | Irregular timestamps |
| `boundary_missing` | 19.9 | 0 | 0 | Near threshold |
| `over_threshold` | 25 | 0 | 0 | Above threshold |
| `all_negative` | 0 | 0 | 100 | Revenue-loss patterns |
| `intermittent` | 0 | 0 | 0 | 85% zeros |
| `duplicates` | 0 | 0 | 0 | 5% duplicate rows |
| `wrong_types` | 0 | 0 | 0 | String target, mixed date formats |
| `short_history` | 0 | 0 | 0 | Only 5 data points |
| `mixed_frequencies` | 0 | 0 | 0 | Daily + weekly + monthly series |