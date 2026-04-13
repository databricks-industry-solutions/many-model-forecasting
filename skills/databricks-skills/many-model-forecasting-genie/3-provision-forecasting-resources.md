# Provision Forecasting Resources

Asks the user which models to run, determines required cluster types, asks the
user which clusters to start and confirms configuration. GPU clusters always
use **single-node** (0 workers).

**Note:** Clusters are created as **ephemeral job clusters** within the Databricks Workflow (see Step 4). This step validates the configuration and lets the user customize it before the job is created. No long-lived clusters are provisioned.

## Steps

### Step 1: Determine model classes from profiling (if available)

Check if `{use_case}_series_profile` exists to auto-detect model requirements:

```sql
SELECT DISTINCT model_types_needed
FROM {catalog}.{schema}.{use_case}_series_profile
WHERE forecastability_class = 'high_confidence'
```

If the profile table exists, extract the recommended models as suggestions. If it does not exist, skip to the model selection step without suggestions.

### Step 1a: Read non-forecastable strategy from pipeline config

Check if `{use_case}_pipeline_config` exists (created by Step 2):

```sql
SELECT non_forecastable_strategy, non_forecastable_models, n_forecastable, n_non_forecastable
FROM {catalog}.{schema}.{use_case}_pipeline_config
WHERE use_case = '{use_case}'
```

If the table does NOT exist (Step 2 was skipped), treat strategy as `include`.

### ⛔ STOP GATE — Step 2: Ask user which models to use

**Always ask the user to select models. Do NOT proceed until the user confirms their selection.**

Ask the user:

> "{if profile exists: 'Based on series profiling, suggested models: {recommended_models}'}
>
> Which models do you want to run? Select one or more from each class:
>
> LOCAL MODELS (CPU cluster — statistical, fast):
> [ ] StatsForecastBaselineWindowAverage / StatsForecastBaselineSeasonalWindowAverage
> [ ] StatsForecastBaselineNaive / StatsForecastBaselineSeasonalNaive
> [ ] StatsForecastAutoArima / StatsForecastAutoETS / StatsForecastAutoCES
> [ ] StatsForecastAutoTheta / StatsForecastAutoTbats / StatsForecastAutoMfles
> [ ] StatsForecastTSB / StatsForecastADIDA / StatsForecastIMAPA
> [ ] StatsForecastCrostonClassic / StatsForecastCrostonOptimized / StatsForecastCrostonSBA
> [ ] SKTimeProphet
>
> GLOBAL MODELS (GPU cluster — neural network, learns across series):
> [ ] NeuralForecastRNN / NeuralForecastLSTM / NeuralForecastNBEATSx / NeuralForecastNHITS
> [ ] NeuralForecastAutoRNN / NeuralForecastAutoLSTM / NeuralForecastAutoNBEATSx
> [ ] NeuralForecastAutoNHITS / NeuralForecastAutoTiDE / NeuralForecastAutoPatchTST
>
> FOUNDATION MODELS (GPU cluster — pretrained, zero-shot):
> [ ] ChronosBoltTiny / ChronosBoltMini / ChronosBoltSmall / ChronosBoltBase
> [ ] Chronos2 / Chronos2Small / Chronos2Synth / TimesFM_2_5_200m
>
> You can select any combination (e.g., local + foundation).
> List the model names you want to run."

**WAIT for the user to respond. Do NOT proceed until the user has selected their models.**

Store the selected models and determine which model classes are needed:
- **Local**: any `StatsForecast*` or `SKTime*` model selected
- **Global**: any `NeuralForecast*` model selected
- **Foundation**: any `Chronos*` or `TimesFM*` model selected

### Step 2a: Confirm non-forecastable models (if `separate_job` strategy)

If `non_forecastable_strategy == 'separate_job'`, present the models selected in Step 2 for confirmation and allow changes.

Determine which model classes the non-forecastable models need:
- **Local (CPU)**: any `StatsForecast*` or `SKTime*` model
- **GPU**: any `NeuralForecast*`, `Chronos*`, or `TimesFM*` model

If `non_forecastable_strategy` is `include` or `fallback`, skip this step entirely.

### Step 3: Determine cloud provider

Ask the user which cloud provider the workspace runs on: **AWS**, **Azure**, or **GCP**.
This determines the specific node types.

### Step 4: Skip cluster listing

> ⚠️ Do NOT attempt to list all workspace clusters — the result is too large and rarely useful since clusters are ephemeral job clusters created per Workflow run.
> Proceed directly to cluster configuration.

### Step 5: Select cluster configuration

Based on the model types and cloud provider, select from these configurations:

#### CPU Cluster (`{use_case}_cpu_cluster`) — for local models

| Setting | Value |
|---------|-------|
| **Runtime** | `17.3.x-cpu-ml-scala2.13` |
| **Node type (AWS)** | `i3.xlarge` |
| **Node type (Azure)** | `Standard_DS3_v2` |
| **Node type (GCP)** | `n1-standard-4` |
| **Workers** | Dynamic — see sizing logic below |
| **Spark config** | `spark.sql.execution.arrow.enabled=true`, `spark.sql.adaptive.enabled=false`, `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |

**CPU worker sizing logic:**

```sql
-- If strategy is 'fallback' or 'separate_job', use the forecastable-only table
SELECT COUNT(DISTINCT unique_id) AS n_series
FROM {catalog}.{schema}.{use_case}_train_data_forecastable

-- If strategy is 'include' or pipeline_config does not exist, use the full table
SELECT COUNT(DISTINCT unique_id) AS n_series
FROM {catalog}.{schema}.{use_case}_train_data
```

| Series count | Suggested workers |
|-------------|-------------------|
| < 100 | 0 (single-node) |
| 100 – 1,000 | 4 |
| 1,000 – 10,000 | 6 |
| 10,000 – 100,000 | 8 |
| > 100,000 | 10 |

#### GPU Cluster (`{use_case}_gpu_cluster`) — for global and foundation models

**GPU clusters MUST always be single-node (0 workers).** This is a hard requirement — do NOT configure multi-node GPU clusters.

| Setting | Value |
|---------|-------|
| **Runtime** | `18.0.x-gpu-ml-scala2.13` |
| **Node type** | User-selectable — see GPU instance options below |
| **Workers** | **0 (single-node) — ALWAYS** |
| **data_security_mode** | `SINGLE_USER` |
| **Spark config** | `spark.master=local[*]`, `spark.databricks.cluster.profile=singleNode`, `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |
| **custom_tags** | `{"ResourceClass": "SingleNode"}` |

#### Non-Forecastable CPU Cluster (`{use_case}_nf_cpu_cluster`)

Only created when `non_forecastable_strategy == 'separate_job'` and NF models include local models. Same node type as main CPU cluster, workers sized to NF series count:

```sql
SELECT COUNT(DISTINCT unique_id) AS n_nf_series
FROM {catalog}.{schema}.{use_case}_train_data_non_forecastable
```

#### Non-Forecastable GPU Cluster (`{use_case}_nf_gpu_cluster`)

Only created when `non_forecastable_strategy == 'separate_job'` and NF models include global or foundation models. Same single-node config as main GPU cluster.

### ⛔ STOP GATE — Step 6: Ask user which clusters to start

**Always present the cluster configuration and ask the user to confirm. Do NOT proceed until the user approves.**

Ask the user:

> "Here is the proposed cluster configuration for your selected models:
>
> ── MAIN PIPELINE (forecastable series: {n_forecastable}) ──
>
> {if local models selected:
> CPU Cluster ({use_case}_cpu_cluster):
>   - Runtime: 17.3.x-cpu-ml-scala2.13
>   - Node type: {cpu_node_type}
>   - Workers: {suggested_workers} (dataset has {n_series} forecastable series)
> }
>
> {if global or foundation models selected:
> GPU Cluster ({use_case}_gpu_cluster) — SINGLE NODE:
>   - Runtime: 18.0.x-gpu-ml-scala2.13
>   - Node type: (select one)
>
>   AWS: (a) g5.xlarge — 1×A10G/24GB  (b) g5.2xlarge  (c) g5.12xlarge — 4×A10G/96GB (recommended)  (d) g5.48xlarge — 8×A10G/192GB
>   Azure: (a) Standard_NC4as_T4_v3  (b) Standard_NV36ads_A10_v5 — 1×A10/24GB (recommended)  (c) Standard_NV72ads_A10_v5  (d) Standard_NC24ads_A100_v4
>   GCP: (a) g2-standard-4 — 1×L4/24GB  (b) g2-standard-48 — 4×L4/96GB (recommended)  (c) a2-highgpu-1g — 1×A100/40GB
> }
>
> Would you like to:
> (1) Accept the proposed configuration
> (2) Change the CPU worker count
> (3) Select a different GPU instance type
> (4) Change both"

**WAIT for the user to respond. Do NOT create any clusters until the user confirms.**

### Decision logic

**Main pipeline:**
- **Local models only** → CPU cluster only
- **Global or foundation models** → GPU cluster (single-node)
- **Local + global/foundation** → Both CPU and GPU clusters
- **Global + Foundation** → One GPU cluster (shared)

**Non-forecastable pipeline (separate_job only):**
- **Local-only NF models** → `nf_cpu_cluster` only
- **GPU NF models** → `nf_gpu_cluster` (single-node)

### Step 7: Unity Catalog enablement verification

All clusters MUST use `SINGLE_USER` data security mode — ML runtimes reject `USER_ISOLATION`:

```json
{
  "data_security_mode": "SINGLE_USER",
  "spark_conf": {
    "spark.databricks.unityCatalog.enabled": "true"
  }
}
```

### Step 8: Present and validate

Present the complete cluster configuration including: cluster key names, runtime versions, node types, worker counts (always 0 for GPU), Spark config, and selected models grouped by class.

### Step 9: Save configuration

Store the cluster configuration and selected models so Step 4 can use them when creating Workflow jobs.

### ⛔ STOP GATE — Step 10: Confirm before proceeding

Ask the user:

> "✅ Cluster configuration complete.
>
> Main pipeline (forecastable series):
> - Selected models: {model_list_summary}
> - Model classes: {classes_summary}
> {if cpu: '- CPU cluster: {cpu_node_type}, {workers} workers'}
> {if gpu: '- GPU cluster: {gpu_node_type}, single-node'}
>
> Would you like to proceed to execute the forecast?
> (a) Yes, proceed
> (b) No, stop here — I'll come back later"

**Do NOT proceed until the user responds.**

## MMF Installation

Each notebook installs MMF at the start:

- **Local models** (`run_local` notebook): `%pip install "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- **GPU models** (`run_gpu` notebook): Uses `subprocess.check_call` with model-specific install logic:
  - **Global (NeuralForecast)**: `mmf_sa[global]`
  - **Foundation (Chronos)**: base `mmf_sa` + `chronos-forecasting==2.2.2` + `utilsforecast==0.2.15`
  - **Foundation (TimesFM)**: base `mmf_sa` + `timesfm[torch]` from tarball URL + `utilsforecast==0.2.15`

> **Why subprocess for GPU?** `%pip` does not interpolate Python variables when the notebook is called via `dbutils.notebook.run()`. The `run_gpu` template uses `subprocess.check_call` to bypass this issue.

## Outputs

- Validated cluster configuration(s) with UC enablement
- Configuration JSON for ephemeral job clusters
- Selected models list, grouped by class (local, global, foundation)
- Non-forecastable cluster configuration (if `separate_job` strategy)
