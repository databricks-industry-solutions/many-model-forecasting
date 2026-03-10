# Setup the MMF Cluster

**Slash command:** `/setup-cluster <model-types>`

Recommends and configures a Databricks cluster appropriate for the selected
model types (local CPU, global GPU, foundation models).

**Note:** Clusters are created as **ephemeral job clusters** within the Databricks Workflow (see Skill 3). This skill validates the configuration and lets the user customize it before the job is created. No long-lived clusters are provisioned.

## Steps

### Step 1: Determine model types

Ask the user which model types they plan to run:
- **Local models only** (StatsForecast: AutoArima, AutoETS, AutoCES, AutoTheta, SKTimeProphet) — CPU is sufficient
- **Global models** (NeuralForecast: NHITS, PatchTST) — GPU recommended
- **Foundation models** (Chronos, Moirai, TimesFM) — GPU required

### Step 2: Determine cloud provider

Ask the user which cloud provider the workspace runs on: **AWS**, **Azure**, or **GCP**.
This determines the specific node types.

### Step 3: Select cluster configuration

Based on the model types and cloud provider, select from these configurations:

#### CPU Cluster (`mmf_cpu_cluster`) — for local models

| Setting | Value |
|---------|-------|
| **Runtime** | `17.3.x-cpu-ml-scala2.13` |
| **Node type (AWS)** | `i3.xlarge` |
| **Node type (Azure)** | `Standard_DS3_v2` |
| **Node type (GCP)** | `n1-standard-4` |
| **Workers** | 2 |
| **Spark config** | `spark.sql.execution.arrow.enabled=true`, `spark.sql.adaptive.enabled=false`, `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |

#### GPU Cluster (`mmf_gpu_cluster`) — for global and foundation models

| Setting | Value |
|---------|-------|
| **Runtime** | `18.0.x-gpu-ml-scala2.13` |
| **Node type (AWS)** | `g5.12xlarge` |
| **Node type (Azure)** | `Standard_NV36ads_A10_v5` |
| **Node type (GCP)** | `g2-standard-48` |
| **Workers** | 0 (single-node) |
| **Spark config** | `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |

### Decision logic

- **Local models only** → CPU cluster only, no GPU cluster needed
- **Global models** → GPU cluster required
- **Foundation models** → GPU cluster required
- **Local + global/foundation** → Both CPU and GPU clusters

Only include clusters that are needed for the selected model types.

### Step 4: Present and validate

Present the complete cluster configuration to the user, including:
- Cluster key name(s)
- Runtime version(s)
- Node type(s) for their cloud
- Number of workers
- All Spark configuration parameters

Use `AskUserQuestion` to confirm the configuration.

### Step 5: Save configuration

Save the cluster configuration locally so `/run-mmf` can use it when creating the Workflow job.

## MMF Installation

Each notebook installs MMF at the start via `%pip`:
- Local models: `pip install "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Global models: `pip install "mmf_sa[global] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Foundation models: `pip install "mmf_sa[foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`

## Outputs

- Validated cluster configuration(s) ready for use in `/run-mmf`
- Configuration details: cluster key, runtime, node type, workers, Spark config