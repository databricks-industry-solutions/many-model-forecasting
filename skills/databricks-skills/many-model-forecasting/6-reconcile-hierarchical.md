# Hierarchical Reconciliation

> ⛔ **MANDATORY:** If you have not read [SKILL.md](SKILL.md) yet, read it now before proceeding. Do NOT take any action until you have read both SKILL.md and this file in full.

**Slash command:** `/reconcile-hierarchical`

Applies hierarchical reconciliation to MMF forecasts, making them coherent across all levels of a hierarchy (e.g., SKU → Category → Total). Reads the best-model forecasts from Skill 5 and produces a reconciled output table.

**This skill is optional — only run it if the use case has a meaningful hierarchy.**

## Preconditions

> ⛔ **Verify before starting this skill.** If preconditions are missing, do NOT improvise — route the user back.

| Precondition | How to verify | If missing |
|---|---|---|
| `{catalog}.{schema}.{use_case}_best_models` exists and is populated | `SELECT COUNT(*) FROM ...` | Go back to **Skill 5 (`/post-process-and-evaluate`)** |
| `{catalog}.{schema}.{use_case}_evaluation_output` exists and is populated | `SELECT COUNT(*) FROM ...` | Go back to **Skill 4 (`/execute-mmf-forecast`)** — `evaluation_output` is required for all methods |
| `{catalog}.{schema}.{use_case}_hierarchy_S` exists | `SELECT COUNT(*) FROM ...` | Skill 1 hierarchical prep step was not run — go back to **Skill 1** |
| `{catalog}.{schema}.{use_case}_hierarchy_tags` exists | `SELECT COUNT(*) FROM ...` | Same as above |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catalog` | From prior skills | Unity Catalog name |
| `schema` | From prior skills | Schema name |
| `use_case` | From Skill 1 | Use case name (prefixes all table names) |
| `freq` | From Skill 1 | Frequency code — H \| D \| W \| M |
| `date_col` | `ds` | Date column name |
| `target` | `y` | Target column name |
| `reconciliation_method` | `MinTrace` | Reconciliation method — see Step 2 |

## Steps

### ⛔ STOP GATE — Step 0: Confirm catalog, schema, use case

If not already known from prior skills, ask:

```
AskUserQuestion:
  "Which catalog, schema and use case name contain the forecast outputs?
   • Catalog:
   • Schema:
   • Use case name:"
```

**Do NOT proceed until confirmed.**

---

### Step 1: Verify forecast inputs

Check that the forecast tables exist and are populated:

```sql
SELECT COUNT(*) AS n_forecast FROM {catalog}.{schema}.{use_case}_forecast_table
```
```sql
SELECT COUNT(*) AS n_evaluation FROM {catalog}.{schema}.{use_case}_evaluation_output
```

If either is missing or empty, route back to the appropriate skill (Skill 4 or Skill 5).

---

### Step 1a: Ensure hierarchy metadata (preprocessing)

Check if `_hierarchy_S` and `_hierarchy_tags` exist:

```sql
SELECT COUNT(*) AS n_S FROM {catalog}.{schema}.{use_case}_hierarchy_S
```
```sql
SELECT COUNT(*) AS n_tags FROM {catalog}.{schema}.{use_case}_hierarchy_tags
```

**If both exist:** continue to Step 2.

**If either is missing:** derive them now from `{use_case}_train_data` unique_ids — do NOT ask the user to go back to Skill 1.

Generate and execute a notebook on serverless compute with:

```python
%pip install /Workspace/Repos/{full_email}/many-model-forecasting[hierarchical] --quiet
```
```python
dbutils.library.restartPython()
```
```python
import sys
sys.path.insert(0, "/Workspace/Repos/{full_email}/many-model-forecasting")
from mmf_sa import derive_hierarchy_from_unique_ids

derive_hierarchy_from_unique_ids(
    spark=spark,
    train_table="{catalog}.{schema}.{use_case}_train_data",
    hierarchy_s_table="{catalog}.{schema}.{use_case}_hierarchy_S",
    hierarchy_tags_table="{catalog}.{schema}.{use_case}_hierarchy_tags",
)
print("✓ Hierarchy metadata derived")
```

> ⛔ **Use `/Workspace/Repos/{full_email}/many-model-forecasting` as install path — NOT a GitHub URL.**

After execution, verify:

```sql
SELECT DISTINCT level_name FROM {catalog}.{schema}.{use_case}_hierarchy_tags ORDER BY level_name
```

Show the user the detected hierarchy levels.

---

### Step 2: Propose reconciliation method

### Step 2: Propose reconciliation method

Do NOT ask "which method do you want?" — propose one with reasoning:

> "I recommend **MinTrace** (`mint_shrink`) as it minimizes forecast error variance by estimating the error covariance matrix across all hierarchy levels — this is the statistically optimal approach.
>
> Alternatives:
> - **BottomUp** — aggregates leaf forecasts upward. Simple, reliable baseline.
> - **TopDown** — distributes top-level forecast downward. Works well when top-level demand is more reliable.
> - **MiddleOut** — anchors at a middle level. Useful when middle aggregations are most trustworthy.
> - **ERM** — learns an optimal reconciliation matrix. Similar requirements to MinTrace.
>
> Unless you have a reason to prefer otherwise, I'll use MinTrace."

Wait for user confirmation or correction.

---

### Step 3: Generate reconciliation notebook

Generate `{notebook_base_path}/run_reconciliation.ipynb` from the template `mmf_reconciliation_notebook_template.ipynb`, filling in:

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | confirmed catalog |
| `{schema}` | confirmed schema |
| `{use_case}` | confirmed use case |
| `{freq}` | frequency from Skill 1 — H \| D \| W \| M |
| `{date_col}` | `ds` (or user-specified) |
| `{target}` | `y` (or user-specified) |
| `{reconciliation_method}` | method confirmed in Step 2 |

---

### Step 4: Run on serverless compute

Reconciliation is a matrix operation — no GPU needed. Run on **serverless** compute.

Tell the user:
> "This notebook runs on serverless compute — no cluster setup needed."

Run the notebook. The output table will be `{catalog}.{schema}.{use_case}_reconciliation_output`.

---

### Step 5: Validate coherence and summarize

After the notebook completes, run a coherence spot-check:

```sql
SELECT
  hierarchy_level,
  reconciliation_method,
  COUNT(DISTINCT unique_id) AS n_series,
  COUNT(*) AS n_rows,
  ROUND(AVG(ABS(y_base - y_reconciled)), 4) AS avg_adjustment
FROM {catalog}.{schema}.{use_case}_reconciliation_output
GROUP BY hierarchy_level, reconciliation_method
ORDER BY hierarchy_level
```

Present a summary to the user:
- How many series were reconciled at each level
- Average adjustment magnitude (how much forecasts changed)
- Confirm coherence is achieved

Example narrative:
> "Reconciliation complete. The forecasts at the `store` level were adjusted by an average of X units to ensure they sum correctly to the `region` and `country` levels. All {n} hierarchy relationships are now coherent."

---

### Step 6: Generate reproducibility notebook

Generate a reproducibility notebook at `{notebook_base_path}/run_reconciliation_repro.ipynb` — identical to the run notebook, allowing the user to re-run reconciliation independently.

---

## Output Tables

| Table | Description |
|-------|-------------|
| `{use_case}_reconciliation_output` | Reconciled forecasts across all hierarchy levels |

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | string | Series identifier (encodes hierarchy level, e.g. `USA/California/Store1`) |
| `ds` | timestamp | Forecast date |
| `y_base` | double | Original forecast before reconciliation |
| `y_reconciled` | double | Coherent forecast after reconciliation |
| `hierarchy_level` | string | Which hierarchy level this series belongs to |
| `reconciliation_method` | string | Method used (e.g. `MinTrace`) |

---

## Key Concepts

**Why reconcile?** Without reconciliation, the sum of store-level forecasts will not equal the region forecast, which will not equal the country forecast. This creates inconsistencies when different teams use forecasts at different levels for planning.

**What `aggregate()` did in Skill 1:** When Skill 1 detected a hierarchy and called `aggregate()`, it created series at all levels (`USA`, `USA/California`, `USA/California/Store1`) and saved the summation matrix (`_hierarchy_S`) and level metadata (`_hierarchy_tags`). Skill 6 uses those to apply the reconciliation.

**MinTrace uses backtest residuals:** MinTrace estimates how correlated the forecast errors are across the hierarchy to find the optimal adjustment weights. It uses out-of-sample backtest residuals from `_evaluation_output` — this is statistically better than in-sample fits because it avoids optimism bias, especially important when MMF selects different models per series. All methods supported by Skill 6 use `_evaluation_output`.
