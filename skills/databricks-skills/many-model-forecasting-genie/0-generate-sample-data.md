# Generate Sample Data

> ⚠️ **For demos and testing only.** If you have real time series data, start at Step 1 (Prep and Clean Data) instead. Output of this step is compatible with the MMF pipeline — you can skip Step 1 and proceed directly to Step 2 (Profile and Classify) or Step 3 (Provision Resources).

Generates a synthetic time series dataset with diverse patterns (seasonal, trending, intermittent, noisy) and writes it as `{use_case}_train_data` — the same format expected by the MMF pipeline.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catalog` | Ask user | Unity Catalog name |
| `schema` | Ask user | Schema name |
| `use_case` | Ask user | Use case name — prefixes all output tables |
| `n_series` | Ask user | Number of synthetic time series to generate |
| `freq` | Ask user | Time series frequency (`D`, `W`, `M`, `H`) |
| `history_length` | Ask user | Number of historical periods per series |

## Steps

### ⛔ STOP GATE — Step 0: Collect catalog, schema, and use case name

**Always ask. Do NOT assume or reuse values.**

Ask the user:

> "Which catalog and schema should I use for the sample data?
> - Catalog: (e.g., main, ml_dev)
> - Schema: (e.g., default, forecasting)
> - Use case name: (e.g., demo, synthetic, test_run)
>
> All generated assets will be prefixed with the use case name, e.g. `{use_case}_train_data`."

**Do NOT proceed until the user provides all three.**

### ⛔ STOP GATE — Step 1: Configure data generation

Ask the user:

> "Configure the sample dataset:
>
> **Number of series:** how many time series to generate?
> (Suggested: 100 for quick tests · 500 for realistic scale · 1000+ for stress testing)
>
> **Frequency:**
> (a) Daily (D) — recommended default
> (b) Weekly (W)
> (c) Monthly (M)
> (d) Hourly (H)
>
> **History length:** how many periods of history per series?
> (Suggested defaults: Daily → 365 · Weekly → 104 · Monthly → 36 · Hourly → 720)
>
> The dataset will include a realistic mix of patterns automatically:
> 40% seasonal · 20% trending · 20% mixed (trend + season) · 10% intermittent · 10% noisy."

**Do NOT proceed until the user confirms all three values.**

### Step 2: Show generation plan

Before generating, present a summary to the user and ask for confirmation:

> "I'll generate the following dataset:
> - Series: {n_series}
> - Frequency: {freq}
> - History: {history_length} periods per series
> - Total rows: ~{n_series × history_length}
> - Output table: `{catalog}.{schema}.{use_case}_train_data`
>
> Shall I proceed?"

**Wait for confirmation before generating.**

### Step 3: Generate and upload notebook

**CRITICAL: Use the template at `notebooks/mmf_generate_data_notebook_template.ipynb` (in this skill folder). Copy it verbatim, only replacing the `{placeholder}` tokens. Do NOT add, remove, or modify any other code.**

Replace these placeholders:
- `{catalog}` → user's catalog
- `{schema}` → user's schema
- `{use_case}` → use case name
- `{n_series}` → number of series (integer, no quotes)
- `{freq}` → frequency string (`D`, `W`, `M`, or `H`)
- `{history_length}` → number of periods (integer, no quotes)

Upload the generated notebook to the Databricks workspace at:
```
{home_path}/mmf-skills-test/notebooks/{use_case}/00_generate_sample_data
```

Where `{home_path}` = `/Workspace/Users/{current_user_email}`.

### Step 4: Execute notebook

Execute the uploaded notebook directly using serverless compute. **Do NOT create a job** — data generation is a single lightweight step, not a recurring pipeline.

### ⛔ STOP GATE — Step 5: Confirm output and next step

After the notebook completes, verify the output:

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT unique_id) AS n_series,
  MIN(ds) AS start_date,
  MAX(ds) AS end_date
FROM {catalog}.{schema}.{use_case}_train_data
```

Present the summary and ask:

> "✅ Sample dataset generated for use case '{use_case}'.
>
> Summary:
> - Table: `{catalog}.{schema}.{use_case}_train_data`
> - Series: {n_series}
> - Date range: {start_date} → {end_date}
> - Total rows: {total_rows}
> - Pattern mix: seasonal · trending · mixed · intermittent · noisy
>
> Your data is already in MMF format — you can skip Step 1 (Prep and Clean Data).
>
> Would you like to:
> (a) Proceed to Step 2 — Profile and classify series (recommended — estimates forecastability and suggests models)
> (b) Proceed to Step 3 — Provision resources and select models directly
> (c) Stop here — I'll come back later"

**Do NOT proceed until the user responds.**

## Outputs

- Delta table `{catalog}.{schema}.{use_case}_train_data` with columns:
  - `unique_id` (STRING) — series identifier, e.g. `seasonal_001`, `trending_003`
  - `ds` (DATE for D/W/M · TIMESTAMP for H) — timestamp
  - `y` (DOUBLE) — target value (non-negative)
- Notebook `{home_path}/mmf-skills-test/notebooks/{use_case}/00_generate_sample_data` — reproducibility notebook
