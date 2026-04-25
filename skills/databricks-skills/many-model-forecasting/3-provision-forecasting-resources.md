# Provision Forecasting Resources

> ⛔ **MANDATORY:** If you have not read [SKILL.md](SKILL.md) yet, read it now before proceeding. Do NOT take any action until you have read both SKILL.md and this file in full.

**Slash command:** `/provision-forecasting-resources`

Asks the user which models to run, determines required cluster types, asks the
user which clusters to start and confirms configuration. GPU clusters always
use **single-node** (0 workers).

**Note:** Clusters are created as **ephemeral job clusters** within the Databricks Workflow (see Skill 4). This skill validates the configuration and lets the user customize it before the job is created. No long-lived clusters are provisioned.

## Preconditions

> ⛔ **Verify before starting this skill.** If preconditions are missing, do NOT improvise — route the user back to the earliest unmet skill.

| Precondition | How to verify | If missing |
|---|---|---|
| `{catalog}.{schema}.{use_case}_train_data` exists | `get_table` or `SELECT 1 FROM ... LIMIT 1` | Go back to **Skill 1 (`/prep-and-clean-data`)** |
| `{forecast_problem_brief}` is in conversation context | Check prior turns | Reconfirm with the user |
| *(Optional)* `{use_case}_series_profile` and `{use_case}_pipeline_config` | Check existence | Treat strategy as `include` if missing (Skill 2 was skipped) — this is allowed |

**On completion this skill produces** (used as preconditions by Skill 4):
- `active_models` selection (grouped by class: local, global, foundation)
- Confirmed cluster configuration(s)

## Steps

**Forecast problem brief (`{forecast_problem_brief}`):** Carry forward from Skill 1 (or Skill 2 if captured there). Any **optional research** or extended rationale when helping the user choose models must be **scoped** to the brief (domain, meaning of `y`, horizon intent, intermittency / exogenous emphasis). When summarizing the user's model selection after Step 2, restate **one line** from `{forecast_problem_brief}` so choices stay traceable to the problem.

### Step 1: Determine model classes from profiling (if available)

First, check if `{use_case}_series_profile` exists to auto-detect model requirements:

```sql
SELECT DISTINCT model_types_needed
FROM {catalog}.{schema}.{use_case}_series_profile
WHERE forecastability_class = 'high_confidence'
```

If the profile table exists, extract the recommended models as suggestions. If it does not exist, skip to the model selection step without suggestions.

### Step 1a: Read non-forecastable strategy from pipeline config

Check if `{use_case}_pipeline_config` exists (created by Skill 2):

```sql
SELECT non_forecastable_strategy, non_forecastable_models, n_forecastable, n_non_forecastable
FROM {catalog}.{schema}.{use_case}_pipeline_config
WHERE use_case = '{use_case}'
```

If the table exists, read:
- `non_forecastable_strategy`: `include`, `fallback`, or `separate_job`
- `non_forecastable_models`: comma-separated model names (only for `separate_job`)
- `n_forecastable` / `n_non_forecastable`: series counts for cluster sizing

If the table does NOT exist (Skill 2 was skipped), treat strategy as `include` (default — all series forecasted together).

### ⛔ STOP GATE — Step 2: Ask user which models to use

**Always ask the user to select models. Do NOT proceed until the user confirms their selection.**

Present the full model catalog and let the user select specific models:

```
AskUserQuestion:
  "Which models do you want to run? Select one or more from each class:

   {if profile exists: '📊 Based on series profiling, suggested models: {recommended_models}'}

   LOCAL MODELS (CPU cluster — statistical, fast):
   [ ] StatsForecastBaselineWindowAverage
   [ ] StatsForecastBaselineSeasonalWindowAverage
   [ ] StatsForecastBaselineNaive
   [ ] StatsForecastBaselineSeasonalNaive
   [ ] StatsForecastAutoArima
   [ ] StatsForecastAutoETS
   [ ] StatsForecastAutoCES
   [ ] StatsForecastAutoTheta
   [ ] StatsForecastAutoTbats
   [ ] StatsForecastAutoMfles
   [ ] StatsForecastTSB
   [ ] StatsForecastADIDA
   [ ] StatsForecastIMAPA
   [ ] StatsForecastCrostonClassic
   [ ] StatsForecastCrostonOptimized
   [ ] StatsForecastCrostonSBA
   [ ] SKTimeProphet

   GLOBAL MODELS (GPU cluster — neural network, learns across series):
   [ ] NeuralForecastRNN
   [ ] NeuralForecastLSTM
   [ ] NeuralForecastNBEATSx
   [ ] NeuralForecastNHITS
   [ ] NeuralForecastAutoRNN
   [ ] NeuralForecastAutoLSTM
   [ ] NeuralForecastAutoNBEATSx
   [ ] NeuralForecastAutoNHITS
   [ ] NeuralForecastAutoTiDE
   [ ] NeuralForecastAutoPatchTST

   FOUNDATION MODELS (GPU cluster — pretrained, zero-shot):
   [ ] ChronosBoltTiny
   [ ] ChronosBoltMini
   [ ] ChronosBoltSmall
   [ ] ChronosBoltBase
   [ ] Chronos2
   [ ] Chronos2Small
   [ ] Chronos2Synth
   [ ] TimesFM_2_5_200m

   You can select any combination (e.g., local + foundation).
   List the model names you want to run."
```

**WAIT for the user to respond. Do NOT proceed until the user has selected their models.**

Store the selected models and determine which model classes are needed:
- **Local**: any `StatsForecast*` or `SKTime*` model selected
- **Global**: any `NeuralForecast*` model selected
- **Foundation**: any `Chronos*` or `TimesFM*` model selected

### Step 2a: Confirm non-forecastable models (if `separate_job` strategy)

If `non_forecastable_strategy == 'separate_job'` (from Step 1a), the user already selected non-forecastable models in Skill 2. Present them for confirmation and allow changes:

```
AskUserQuestion:
  "For the {n_non_forecastable} non-forecastable series, you selected these models in
   the profiling step:
   • {non_forecastable_models}

   Would you like to:
   (1) Keep these models
   (2) Change the model selection

   {if (2): present the same model catalog as Step 2, pre-filled with the Skill 2 selections}"
```

Determine which model classes the non-forecastable models need:
- **Local (CPU)**: any `StatsForecast*` or `SKTime*` model
- **GPU**: any `NeuralForecast*`, `Chronos*`, or `TimesFM*` model

If `non_forecastable_strategy` is `include` or `fallback`, skip this step entirely.

### Step 3: Determine cloud provider

Ask the user which cloud provider the workspace runs on: **AWS**, **Azure**, or **GCP**.
This determines the specific node types.

<!-- 
  NOTE — Step 4 (Check existing clusters) removed.
  MMF jobs always use ephemeral job clusters defined in the Workflow spec (Skill 4).
  There is no scenario in the normal flow where an existing all-purpose cluster needs to be listed or reused.
  `list_clusters` was previously called here but returns 300k+ chars in shared workspaces with no actionable result.
  PENDING: validate with the team whether there is a legitimate reuse scenario that requires this step.
-->

### Step 4: Select cluster configuration

> ⛔ **MANDATORY runtime pinning — DO NOT SUBSTITUTE.**
> Each cluster class has exactly one allowed `spark_version`. They are **not interchangeable**:
>
> | Cluster class | Required `spark_version` | Used by |
> |---|---|---|
> | CPU (local models) | `17.3.x-cpu-ml-scala2.13` | `{use_case}_cpu_cluster`, `{use_case}_nf_cpu_cluster` |
> | GPU (global & foundation models) | `18.0.x-gpu-ml-scala2.13` | `{use_case}_gpu_cluster`, `{use_case}_nf_gpu_cluster` |
>
> The agent is FORBIDDEN from:
> - Using `17.3.x-cpu-ml-scala2.13` on a GPU cluster (it has no GPU drivers — global/foundation models will fail or silently run on CPU).
> - Using `17.3.x-gpu-ml-scala2.13` on the GPU cluster because it "looks like LTS." The GPU pipeline is pinned to **18.0** and tested against it.
> - Using `18.0.x-cpu-ml-scala2.13` on the CPU cluster.
> - Copying the CPU job's `spark_version` into the GPU job templates in Skill 4. Each job class has its own pinned runtime.
>
> Skill 4 Step 5c reads back the `spark_version` of every job cluster after creation and aborts if it doesn't match this table.

Based on the model types and cloud provider, select from these configurations:

#### CPU Cluster (`{use_case}_cpu_cluster`) — for local models

| Setting | Value |
|---------|-------|
| **Runtime** | `17.3.x-cpu-ml-scala2.13` |
| **Node type** | User-selectable — see CPU instance options below (default: 16 vCPU) |
| **Workers** | **Always ask the user** — present sizing guideline as recommendation |
| **Availability** | **On-demand only** — do NOT use spot/preemptible instances (see note below) |
| **Spark config** | `spark.sql.execution.arrow.enabled=true`, `spark.sql.adaptive.enabled=false`, `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |

**CPU instance options by cloud provider:**

**AWS:**
| Option | Instance | vCPUs | Memory | Notes |
|--------|----------|-------|--------|-------|
| **(a) recommended** | `i3.4xlarge` | 16 | 122 GB | Good balance of CPU and memory for most workloads |
| (b) | `i3.8xlarge` | 32 | 244 GB | High-parallelism series fitting or very wide feature sets |

**Azure:**
| Option | Instance | vCPUs | Memory | Notes |
|--------|----------|-------|--------|-------|
| **(a) recommended** | `Standard_DS5_v2` | 16 | 56 GB | Good balance of CPU and memory for most workloads |
| (b) | `Standard_D32ds_v5` | 32 | 128 GB | High-parallelism series fitting or very wide feature sets |

**GCP:**
| Option | Instance | vCPUs | Memory | Notes |
|--------|----------|-------|--------|-------|
| **(a) recommended** | `n1-standard-16` | 16 | 60 GB | Good balance of CPU and memory for most workloads |
| (b) | `n1-standard-32` | 32 | 120 GB | High-parallelism series fitting or very wide feature sets |

**CPU worker sizing — always ask the user:**

Query the number of distinct series from the **correct** training table based on the non-forecastable strategy:
```sql
-- If strategy is 'fallback' or 'separate_job', use the forecastable-only table
SELECT COUNT(DISTINCT unique_id) AS n_series
FROM {catalog}.{schema}.{use_case}_train_data_forecastable

-- If strategy is 'include' or pipeline_config does not exist, use the full table
SELECT COUNT(DISTINCT unique_id) AS n_series
FROM {catalog}.{schema}.{use_case}_train_data
```

Present the series count and the sizing guideline to the user, then **ask them to choose** the number of workers:

| Series count | Recommended workers | Rationale |
|-------------|-------------------|-----------|
| < 100 | 0 (single-node) | No parallelism needed |
| 100 – 1,000 | 4 | Moderate parallelism |
| 1,000 – 10,000 | 6 | Each worker handles ~1,500 series |
| 10,000 – 100,000 | 8 | High parallelism for large-scale |
| > 100,000 | 10 | Maximum recommended; beyond this consider partitioning |

**⛔ Do NOT auto-apply the recommended value.** Always ask the user how many workers they want (showing the recommendation as guidance).

#### Non-Forecastable CPU Cluster (`{use_case}_nf_cpu_cluster`) — for separate_job strategy, local models only

Only created when `non_forecastable_strategy == 'separate_job'` and the non-forecastable models include local (CPU) models.

Uses the same node type and instance selection as the main CPU cluster. Query the non-forecastable series count:

```sql
SELECT COUNT(DISTINCT unique_id) AS n_nf_series
FROM {catalog}.{schema}.{use_case}_train_data_non_forecastable
```

Present the series count and the sizing guideline to the user, then **ask them to choose** the number of workers:

| Non-forecastable series count | Recommended workers | Rationale |
|------|-------------------|-----------|
| < 100 | 0 (single-node) | No parallelism needed |
| 100 – 1,000 | 2 | Light parallelism |
| 1,000 – 10,000 | 4 | Moderate parallelism |
| > 10,000 | 6 | Higher parallelism |

**⛔ Do NOT auto-apply the recommended value.** Always ask the user how many workers they want (showing the recommendation as guidance).

---

#### GPU Cluster (`{use_case}_gpu_cluster`) — for global and foundation models

**GPU clusters MUST always be single-node (0 workers).** This is a hard requirement — do NOT configure multi-node GPU clusters. Single-node mode requires explicit Spark config and custom tags beyond just setting `num_workers: 0`.

| Setting | Value |
|---------|-------|
| **Runtime** | `18.0.x-gpu-ml-scala2.13` |
| **Node type** | User-selectable — see GPU instance options below |
| **Workers** | **0 (single-node) — ALWAYS** |
| **Availability** | **On-demand only** — do NOT use spot/preemptible instances (see note below) |
| **data_security_mode** | `SINGLE_USER` (ML runtimes do not support `USER_ISOLATION`) |
| **Spark config** | `spark.master=local[*]`, `spark.databricks.cluster.profile=singleNode`, `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |
| **custom_tags** | `{"ResourceClass": "SingleNode"}` |

**GPU instance options by cloud provider:**

**AWS:**
| Option | Instance | GPUs | GPU Memory | Notes |
|--------|----------|------|------------|-------|
| (a) | `g5.xlarge` | 1× A10G | 24 GB | Small foundation models |
| (b) | `g5.2xlarge` | 1× A10G | 24 GB | More CPU/RAM |
| **(c) recommended** | `g5.12xlarge` | 4× A10G | 96 GB | Global + foundation |
| (d) | `g5.48xlarge` | 8× A10G | 192 GB | Large-scale training |

**Azure:**
| Option | Instance | GPUs | GPU Memory | Notes |
|--------|----------|------|------------|-------|
| (a) | `Standard_NC4as_T4_v3` | 1× T4 | 16 GB | Budget option |
| (b) | `Standard_NC8as_T4_v3` | 1× T4 | 16 GB | More CPU/RAM |
| **(c) recommended** | `Standard_NV36ads_A10_v5` | 1× A10 | 24 GB | Good balance |
| (d) | `Standard_NV72ads_A10_v5` | 2× A10 | 48 GB | Global + foundation |
| (e) | `Standard_NC24ads_A100_v4` | 1× A100 | 80 GB | Large models |

**GCP:**
| Option | Instance | GPUs | GPU Memory | Notes |
|--------|----------|------|------------|-------|
| (a) | `g2-standard-4` | 1× L4 | 24 GB | Small foundation models |
| (b) | `g2-standard-8` | 1× L4 | 24 GB | More CPU/RAM |
| **(c) recommended** | `g2-standard-48` | 4× L4 | 96 GB | Global + foundation |
| (d) | `a2-highgpu-1g` | 1× A100 | 40 GB | Large models |

#### On-demand instance requirement (all clusters)

**All clusters MUST use on-demand (non-spot) instances for both driver and worker nodes.** Spot/preemptible instances can be reclaimed mid-run, causing long-running forecasting jobs to fail partway through with no automatic recovery.

When generating the job cluster JSON, include the cloud-specific availability settings:

**AWS:**
```json
{
  "aws_attributes": {
    "first_on_demand": 100,
    "availability": "ON_DEMAND"
  }
}
```

**Azure:**
```json
{
  "azure_attributes": {
    "first_on_demand": 100,
    "availability": "ON_DEMAND_AZURE"
  }
}
```

**GCP:**
```json
{
  "gcp_attributes": {
    "availability": "ON_DEMAND_GCP"
  }
}
```

`first_on_demand: 100` ensures all nodes (driver + up to 99 workers) are on-demand. Apply these attributes to **every** cluster configuration — CPU, GPU, and non-forecastable clusters alike.

#### Non-Forecastable GPU Cluster (`{use_case}_nf_gpu_cluster`) — for separate_job strategy, GPU models only

Only created when `non_forecastable_strategy == 'separate_job'` and the non-forecastable models include global or foundation models. Same single-node config as the main GPU cluster. A smaller GPU instance is usually sufficient since non-forecastable series are typically fewer and simpler.

### ⛔ STOP GATE — Step 6: Ask user which clusters to start

**Always present the cluster configuration and ask the user to confirm. Do NOT proceed until the user approves.**

Ask about each cluster one at a time. Only ask about cluster types the user's selected models require.

#### Step 6a: CPU cluster configuration (if local models selected)

```
AskUserQuestion:
  "CPU Cluster ({use_case}_cpu_cluster) — for local models:
     • Runtime: 17.3.x-cpu-ml-scala2.13

   Which instance type?

   {cloud-specific options for the user's provider only:}

   AWS:
   (a) i3.4xlarge   — 16 vCPUs, 122 GB  (recommended)
   (b) i3.8xlarge   — 32 vCPUs, 244 GB  (high-parallelism or wide features)

   Azure:
   (a) Standard_DS5_v2    — 16 vCPUs, 56 GB  (recommended)
   (b) Standard_D32ds_v5  — 32 vCPUs, 128 GB (high-parallelism or wide features)

   GCP:
   (a) n1-standard-16 — 16 vCPUs, 60 GB  (recommended)
   (b) n1-standard-32 — 32 vCPUs, 120 GB (high-parallelism or wide features)"
```

**WAIT for the user to respond.**

Then ask about worker count separately:

```
AskUserQuestion:
  "How many workers for the CPU cluster?

   • Forecastable series: {n_series}
   • Recommended workers: {recommended_workers}

   Sizing guide:
     < 100 series → 0 (single-node)
     100–1,000    → 4
     1,000–10,000 → 6
     10,000–100,000 → 8
     > 100,000    → 10

   Enter the number of workers:"
```

**WAIT for the user to respond.**

{if non_forecastable_strategy == 'separate_job' and nf_local_models, ask in a separate question:}

```
AskUserQuestion:
  "How many workers for the NF CPU cluster ({use_case}_nf_cpu_cluster)?

   • Non-forecastable series: {n_nf_series}
   • Recommended workers: {nf_recommended_workers}
   • Instance type: same as main CPU cluster ({cpu_node_type})

   Enter the number of workers:"
```

**WAIT for the user to respond.**

#### Step 6b: GPU cluster configuration (if global or foundation models selected)

```
AskUserQuestion:
  "GPU Cluster ({use_case}_gpu_cluster) — single-node, for global & foundation models:
     • Runtime: 18.0.x-gpu-ml-scala2.13
     • Workers: 0 (always single-node)

   Which instance type?

   {cloud-specific options for the user's provider only:}

   AWS:
   (a) g5.xlarge    — 1× A10G GPU, 24 GB  (small foundation models)
   (b) g5.2xlarge   — 1× A10G GPU, 24 GB, more CPU/RAM
   (c) g5.12xlarge  — 4× A10G GPUs, 96 GB  (recommended for global + foundation)
   (d) g5.48xlarge  — 8× A10G GPUs, 192 GB (large-scale training)

   Azure:
   (a) Standard_NC4as_T4_v3    — 1× T4 GPU, 16 GB
   (b) Standard_NC8as_T4_v3    — 1× T4 GPU, 16 GB, more CPU/RAM
   (c) Standard_NV36ads_A10_v5 — 1× A10 GPU, 24 GB  (recommended)
   (d) Standard_NV72ads_A10_v5 — 2× A10 GPUs, 48 GB (global + foundation)
   (e) Standard_NC24ads_A100_v4 — 1× A100 GPU, 80 GB (large models)

   GCP:
   (a) g2-standard-4   — 1× L4 GPU, 24 GB
   (b) g2-standard-8   — 1× L4 GPU, 24 GB, more CPU/RAM
   (c) g2-standard-48  — 4× L4 GPUs, 96 GB  (recommended)
   (d) a2-highgpu-1g   — 1× A100 GPU, 40 GB (large models)"
```

**WAIT for the user to respond.**

{if non_forecastable_strategy == 'separate_job' and nf_gpu_models, ask in a separate question:}

```
AskUserQuestion:
  "Which GPU instance for the NF GPU cluster ({use_case}_nf_gpu_cluster)?
   A smaller instance is usually sufficient for non-forecastable series.

   (select from the same GPU options above)"
```

**WAIT for the user to respond.**

### Decision logic

**Main pipeline (forecastable series):**
- **Local models only** → CPU cluster only, no GPU cluster needed
- **Global models** → GPU cluster required (single-node)
- **Foundation models** → GPU cluster required (single-node)
- **Local + global/foundation** → Both CPU and GPU clusters
- **Global + Foundation** → GPU cluster (single-node) — both model classes share the same GPU cluster

**Non-forecastable pipeline (separate_job strategy only):**
- **Local-only non-forecastable models** → `{use_case}_nf_cpu_cluster` only
- **GPU non-forecastable models** → `{use_case}_nf_gpu_cluster` (single-node)
- **Mixed** → Both `nf_cpu_cluster` and `nf_gpu_cluster`

Only include clusters that are needed for the selected model types. If `non_forecastable_strategy` is `include` or `fallback`, no non-forecastable clusters are needed.

### Step 7: Unity Catalog enablement verification

All clusters MUST have UC enabled. **All ML runtimes** (both CPU-ML and GPU-ML) require `SINGLE_USER`:

**CPU clusters (ML Runtime):**
```json
{
  "data_security_mode": "SINGLE_USER",
  "spark_conf": {
    "spark.databricks.unityCatalog.enabled": "true"
  }
}
```

**GPU clusters (ML Runtime):**
```json
{
  "data_security_mode": "SINGLE_USER",
  "spark_conf": {
    "spark.databricks.unityCatalog.enabled": "true"
  }
}
```

> **Why `SINGLE_USER` for all ML runtimes?** ML runtimes (both `*-cpu-ml-*` and `*-gpu-ml-*`) reject `USER_ISOLATION` with "Spark version does not support Table Access Control". `SINGLE_USER` still provides UC access for the job owner.

If cluster doesn't have UC → warn user, suggest adding required Spark config.

### Step 8: Present and validate

Present the complete cluster configuration to the user, including:
- Cluster key name(s) (prefixed with `{use_case}`)
- Runtime version(s)
- Node type(s) for their cloud
- Number of workers (always 0 for GPU)
- All Spark configuration parameters
- Selected models grouped by class (local, global, foundation)

### Step 9: Save configuration

Save the cluster configuration and selected models locally so `/execute-mmf-forecast` can use it when creating the Workflow jobs.

### ⛔ STOP GATE — Step 10: Confirm before proceeding to next skill

```
AskUserQuestion:
  "✅ Cluster configuration complete.

   Main pipeline (forecastable series):
   • Selected models: {model_list_summary}
   • Model classes: {classes_summary}
   {if cpu: '• CPU cluster: {cpu_node_type}, {workers} workers'}
   {if gpu: '• GPU cluster: {gpu_node_type}, single-node'}

   {if non_forecastable_strategy == 'separate_job':
   Non-forecastable pipeline ({n_non_forecastable} series):
   • Selected models: {nf_model_list_summary}
   {if nf_cpu: '• NF CPU cluster: {cpu_node_type}, {nf_workers} workers'}
   {if nf_gpu: '• NF GPU cluster: {nf_gpu_node_type}, single-node'}
   }

   {if non_forecastable_strategy == 'fallback':
   Non-forecastable series ({n_non_forecastable}): handled by {fallback_method} fallback (no cluster needed)
   }

   {if non_forecastable_strategy == 'include':
   Non-forecastable series: included in main pipeline (no separate handling)
   }

   Would you like to proceed to execute the forecast?
   (a) Yes, proceed to /execute-mmf-forecast
   (b) No, stop here — I'll come back later"
```

**Do NOT proceed until the user responds.**

## MMF Installation

Each notebook installs MMF at the start:

- **Local models** (`run_local` notebook): Uses `%pip install "mmf_sa[local] @ git+https://...@main"`
- **GPU models** (`run_gpu` notebook): Uses `subprocess.check_call` with model-specific install logic:
  - **Global (NeuralForecast)**: `mmf_sa[global] @ git+https://...@main`
  - **Foundation (Chronos)**: base `mmf_sa` + `chronos-forecasting==2.2.2` + `utilsforecast==0.2.15`
  - **Foundation (TimesFM)**: base `mmf_sa` + `timesfm[torch,xreg]` from tarball URL + `utilsforecast==0.2.15`

> **Why subprocess for GPU?** `%pip` does not interpolate Python variables when the notebook is called via `dbutils.notebook.run()`. Additionally, `mmf_sa[foundation]` includes a transitive `timesfm @ git+https://...@commit` dependency whose `git checkout` fails on GPU clusters. The `run_gpu` template uses `subprocess.check_call` and installs `timesfm[torch,xreg]` from a GitHub tarball URL to bypass both issues.

## Outputs

- Validated cluster configuration(s) with UC enablement
- Cluster IDs if existing clusters are reused
- Configuration JSON for ephemeral job clusters
- Configuration details: cluster key, runtime, node type, workers (always 0 for GPU), Spark config
- Selected models list, grouped by class (local, global, foundation)
- Non-forecastable cluster configuration (if `separate_job` strategy): cluster key, runtime, node type, workers
- Non-forecastable models list (if `separate_job` strategy)

## ⛔ Step-transition gate — Ask the user before moving on

After model and cluster selections are confirmed, the agent MUST stop and ask before starting Skill 4. **Do NOT auto-advance to executing the forecast.**

```
AskUserQuestion:
  "Skill 3 (Provision Forecasting Resources) is complete.

  Selected models:
    • Local (CPU): {local_models}
    • Global (GPU): {global_models}
    • Foundation (GPU): {foundation_models}
  Cluster config: {cluster_summary}

  Ready to proceed to Skill 4 (Execute MMF Forecast) — generate notebooks, create jobs, and run the forecast?
    (a) Yes, continue to Skill 4
    (b) Stop here for now"
  Options: [a, b]
```
