# Provision Forecasting Resources

**Slash command:** `/provision-forecasting-resources`

Asks the user which models to run, determines required cluster types, asks the
user which clusters to start and confirms configuration. GPU clusters always
use **single-node** (0 workers).

**Note:** Clusters are created as **ephemeral job clusters** within the Databricks Workflow (see Skill 4). This skill validates the configuration and lets the user customize it before the job is created. No long-lived clusters are provisioned.

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

### Step 3: Determine cloud provider

Ask the user which cloud provider the workspace runs on: **AWS**, **Azure**, or **GCP**.
This determines the specific node types.

### Step 4: Check existing clusters

Use MCP `list_clusters` to find clusters matching the required configurations:
- Match by runtime version (`17.3.x-cpu-ml-scala2.13` for CPU, `18.0.x-gpu-ml-scala2.13` for GPU)
- Match by node type for the target cloud provider

Decision logic:
```
if matching_cluster.state == "RUNNING":
    → "Found running cluster {name}. Reuse it?"
elif matching_cluster.state == "TERMINATED":
    → "Found terminated cluster {name}. Start it?"
    → Use start_cluster MCP tool if confirmed
else:
    → Proceed to generate ephemeral job cluster config
```

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

Query the number of distinct series:
```sql
SELECT COUNT(DISTINCT unique_id) AS n_series
FROM {catalog}.{schema}.{use_case}_train_data
```

Suggest workers based on series count:

| Series count | Suggested workers | Rationale |
|-------------|-------------------|-----------|
| < 100 | 0 (single-node) | No parallelism needed |
| 100 – 1,000 | 4 | Moderate parallelism |
| 1,000 – 10,000 | 6 | Each worker handles ~1,500 series |
| 10,000 – 100,000 | 8 | High parallelism for large-scale |
| > 100,000 | 10 | Maximum recommended; beyond this consider partitioning |

#### GPU Cluster (`{use_case}_gpu_cluster`) — for global and foundation models

**GPU clusters MUST always be single-node (0 workers).** This is a hard requirement — do NOT configure multi-node GPU clusters. Single-node mode requires explicit Spark config and custom tags beyond just setting `num_workers: 0`.

| Setting | Value |
|---------|-------|
| **Runtime** | `18.0.x-gpu-ml-scala2.13` |
| **Node type** | User-selectable — see GPU instance options below |
| **Workers** | **0 (single-node) — ALWAYS** |
| **data_security_mode** | `SINGLE_USER` (ML runtimes do not support `USER_ISOLATION`) |
| **Spark config** | `spark.master=local[*]`, `spark.databricks.cluster.profile=singleNode`, `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |
| **custom_tags** | `{"ResourceClass": "SingleNode"}` |

### ⛔ STOP GATE — Step 6: Ask user which clusters to start

**Always present the cluster configuration and ask the user to confirm. Do NOT proceed until the user approves.**

Present the computed configuration and let the user customize:

```
AskUserQuestion:
  "Here is the proposed cluster configuration for your selected models:

   {if local models selected:
   CPU Cluster ({use_case}_cpu_cluster):
     • Runtime: 17.3.x-cpu-ml-scala2.13
     • Node type: {cpu_node_type}
     • Workers: {suggested_workers} (dataset has {n_series} series)
   }

   {if global or foundation models selected:
   GPU Cluster ({use_case}_gpu_cluster) — SINGLE NODE:
     • Runtime: 18.0.x-gpu-ml-scala2.13
     • Node type: (select one)

     AWS:
     (a) g5.xlarge    — 1× A10G GPU, 24 GB  (small foundation models)
     (b) g5.2xlarge   — 1× A10G GPU, 24 GB, more CPU/RAM
     (c) g5.12xlarge  — 4× A10G GPUs, 96 GB  (recommended for global + foundation)
     (d) g5.48xlarge  — 8× A10G GPUs, 192 GB (large-scale training)

     Azure:
     (a) Standard_NC4as_T4_v3    — 1× T4 GPU, 16 GB
     (b) Standard_NC8as_T4_v3    — 1× T4 GPU, 16 GB, more CPU/RAM
     (c) Standard_NV36ads_A10_v5 — 2× A10 GPUs, 48 GB  (recommended)
     (d) Standard_NC24ads_A100_v4 — 1× A100 GPU, 80 GB (large models)

     GCP:
     (a) g2-standard-4   — 1× L4 GPU, 24 GB
     (b) g2-standard-8   — 1× L4 GPU, 24 GB, more CPU/RAM
     (c) g2-standard-48  — 4× L4 GPUs, 96 GB  (recommended)
     (d) a2-highgpu-1g   — 1× A100 GPU, 40 GB (large models)
   }

   Would you like to:
   (1) Accept the proposed configuration
   (2) Change the CPU worker count (currently {suggested_workers})
   (3) Select a different GPU instance type
   (4) Change both"
```

**WAIT for the user to respond. Do NOT create any clusters until the user confirms.**

### Decision logic

- **Local models only** → CPU cluster only, no GPU cluster needed
- **Global models** → GPU cluster required (single-node)
- **Foundation models** → GPU cluster required (single-node)
- **Local + global/foundation** → Both CPU and GPU clusters
- **Global + Foundation** → GPU cluster (single-node) — both model classes share the same GPU cluster

Only include clusters that are needed for the selected model types.

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

   Summary:
   • Selected models: {model_list_summary}
   • Model classes: {classes_summary}
   {if cpu: '• CPU cluster: {cpu_node_type}, {workers} workers'}
   {if gpu: '• GPU cluster: {gpu_node_type}, single-node'}

   Would you like to proceed to execute the forecast?
   (a) Yes, proceed to /execute-mmf-forecast
   (b) No, stop here — I'll come back later"
```

**Do NOT proceed until the user responds.**

## MMF Installation

Each notebook installs MMF at the start:

- **Local models** (`run_local` notebook): Uses `%pip install "mmf_sa[local] @ git+https://...@v0.1.2"`
- **GPU models** (`run_gpu` notebook): Uses `subprocess.check_call` with model-specific install logic:
  - **Global (NeuralForecast)**: `mmf_sa[global] @ git+https://...@v0.1.2`
  - **Foundation (Chronos)**: base `mmf_sa` + `chronos-forecasting==2.2.2` + `utilsforecast==0.2.15`
  - **Foundation (TimesFM)**: base `mmf_sa` + `timesfm[torch]` from tarball URL + `utilsforecast==0.2.15`

> **Why subprocess for GPU?** `%pip` does not interpolate Python variables when the notebook is called via `dbutils.notebook.run()`. Additionally, `mmf_sa[foundation]` includes a transitive `timesfm @ git+https://...@commit` dependency whose `git checkout` fails on GPU clusters. The `run_gpu` template uses `subprocess.check_call` and installs `timesfm[torch]` from a GitHub tarball URL to bypass both issues.

## Outputs

- Validated cluster configuration(s) with UC enablement
- Cluster IDs if existing clusters are reused
- Configuration JSON for ephemeral job clusters
- Configuration details: cluster key, runtime, node type, workers (always 0 for GPU), Spark config
- Selected models list, grouped by class (local, global, foundation)
