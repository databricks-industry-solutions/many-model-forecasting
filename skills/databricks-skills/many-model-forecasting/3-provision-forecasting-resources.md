# Provision Forecasting Resources

**Slash command:** `/provision-forecasting-resources`

Determines required cluster types (from profiling or user input), configures
clusters with the correct specs, verifies Unity Catalog enablement, and
optionally reuses or restarts existing clusters.

**Note:** Clusters are created as **ephemeral job clusters** within the Databricks Workflow (see Skill 4). This skill validates the configuration and lets the user customize it before the job is created. No long-lived clusters are provisioned.

## Steps

### Step 1: Determine model classes

First, check if `{use_case}_series_profile` exists to auto-detect model requirements:

```sql
SELECT DISTINCT model_types_needed
FROM {catalog}.{schema}.{use_case}_series_profile
WHERE forecastability_class = 'high_confidence'
```

In both cases (profile exists or not), present the full model catalog and let the user select one or more model classes:

```
AskUserQuestion:
  "Which model classes do you want to run? Select one or more:

   [ ] Local models (CPU cluster)
       StatsForecastBaselineWindowAverage, StatsForecastBaselineSeasonalWindowAverage,
       StatsForecastBaselineNaive, StatsForecastBaselineSeasonalNaive,
       StatsForecastAutoArima, StatsForecastAutoETS, StatsForecastAutoCES,
       StatsForecastAutoTheta, StatsForecastAutoTbats, StatsForecastAutoMfles,
       StatsForecastTSB, StatsForecastADIDA, StatsForecastIMAPA,
       StatsForecastCrostonClassic, StatsForecastCrostonOptimized,
       StatsForecastCrostonSBA, SKTimeProphet

   [ ] Global models (GPU cluster)
       NeuralForecastRNN, NeuralForecastLSTM, NeuralForecastNBEATSx,
       NeuralForecastNHITS, NeuralForecastAutoRNN, NeuralForecastAutoLSTM,
       NeuralForecastAutoNBEATSx, NeuralForecastAutoNHITS,
       NeuralForecastAutoTiDE, NeuralForecastAutoPatchTST

   [ ] Foundation models (GPU cluster)
       ChronosBoltTiny, ChronosBoltMini, ChronosBoltSmall, ChronosBoltBase,
       Chronos2, Chronos2Small, Chronos2Synth, TimesFM_2_5_200m

   {if profile exists: 'Based on series profiling, suggested classes: {auto_detected_classes}'}
   You can select any combination (e.g., local + foundation)."
  Options: [local, global, foundation] (multi-select)
```

The user's selection determines which clusters are provisioned:
- **Local only** → CPU cluster
- **Global only** → GPU cluster
- **Foundation only** → GPU cluster
- **Local + Global** → CPU + GPU clusters
- **Local + Foundation** → CPU + GPU clusters
- **Global + Foundation** → GPU cluster
- **All three** → CPU + GPU clusters

### Step 2: Determine cloud provider

Ask the user which cloud provider the workspace runs on: **AWS**, **Azure**, or **GCP**.
This determines the specific node types.

### Step 3: Check existing clusters

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

### Step 4: Select cluster configuration

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

Present the suggestion and let the user override:

```
AskUserQuestion:
  "Your dataset has {n_series} distinct time series.
   Suggested CPU workers: {suggested_workers}

   How many workers would you like?
   (a) Use suggested: {suggested_workers} workers
   (b) Custom: enter a number (0 for single-node, max 64)"
```

#### GPU Cluster (`{use_case}_gpu_cluster`) — for global and foundation models

| Setting | Value |
|---------|-------|
| **Runtime** | `18.0.x-gpu-ml-scala2.13` |
| **Node type** | User-selectable — see GPU instance options below |
| **Workers** | 0 (single-node) |
| **Spark config** | `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |

**GPU instance selection:**

The cluster runs as single-node (0 workers). The user chooses the GPU instance type, which determines the number of GPUs available:

```
AskUserQuestion:
  "Select a GPU instance type:

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
   (d) a2-highgpu-1g   — 1× A100 GPU, 40 GB (large models)"
```

### Decision logic

- **Local models only** → CPU cluster only, no GPU cluster needed
- **Global models** → GPU cluster required
- **Foundation models** → GPU cluster required
- **Local + global/foundation** → Both CPU and GPU clusters

Only include clusters that are needed for the selected model types.

### Step 5: Unity Catalog enablement verification

All clusters MUST have UC enabled. Verify:
```json
{
  "data_security_mode": "USER_ISOLATION",
  "spark_conf": {
    "spark.databricks.unityCatalog.enabled": "true"
  }
}
```

If cluster doesn't have UC → warn user, suggest adding required Spark config.

### Step 6: Present and validate

Present the complete cluster configuration to the user, including:
- Cluster key name(s) (prefixed with `{use_case}`)
- Runtime version(s)
- Node type(s) for their cloud
- Number of workers
- All Spark configuration parameters

Use `AskUserQuestion` to confirm the configuration.

### Step 7: Save configuration

Save the cluster configuration locally so `/execute-mmf-forecast` can use it when creating the Workflow job.

## MMF Installation

Each notebook installs MMF at the start via `%pip`:
- Local models: `pip install "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Global models: `pip install "mmf_sa[global] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Foundation models: `pip install "mmf_sa[foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`

## Outputs

- Validated cluster configuration(s) with UC enablement
- Cluster IDs if existing clusters are reused
- Configuration JSON for ephemeral job clusters
- Configuration details: cluster key, runtime, node type, workers, Spark config
