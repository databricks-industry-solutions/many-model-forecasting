# TODO

## Test Datasets

Time series datasets to test the MMF skill on, organized by complexity.

### Small / Quick Tests

| Dataset | Series | Frequency | Source |
|---------|--------|-----------|--------|
| M4 Daily | 4,227 | Daily | `datasetsforecast.m4` or [Kaggle](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition) |
| M4 Weekly | 359 | Weekly | Same |
| M4 Monthly | 48,000 | Monthly | Same |
| Australian Tourism | 304 | Quarterly | `datasetsforecast.hierarchical` |
| ETTh1 / ETTm1 | 7 (multivariate) | Hourly/15min | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |

### Medium / Realistic

| Dataset | Series | Frequency | Source |
|---------|--------|-----------|--------|
| Walmart M5 | ~30,490 | Daily | [Kaggle M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy) |
| Favorita (store sales) | ~1,782 | Daily | [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) |
| UCI Electricity | 370 | Hourly | [UCI ML Repo](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) |
| Wikipedia Web Traffic | 145,063 | Daily | [Kaggle](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting) |

### On Databricks (no download needed)

| Dataset | How to access |
|---------|---------------|
| `samples.tpch.lineitem` | Built-in — use `l_shipdate` as `ds`, `l_quantity` as `y`, group by `l_partkey` |
| `samples.nyctaxi.trips` | Built-in — use `pickup_datetime` as `ds`, `fare_amount` as `y`, group by `pickup_zip` |
| `databricks-datasets/COVID` | `/databricks-datasets/COVID/covid-19-data/` — daily cases by country |

### Easiest to Load via Python

```python
# pip install datasetsforecast
from datasetsforecast.m4 import M4
# Returns df with columns: unique_id, ds, y — exactly the MMF schema
df, *_ = M4.load("data", group="Daily")
```

**Recommended starting points:**
- **M4 Daily** — small, fast, already in MMF schema
- **samples.nyctaxi.trips** — zero setup on Databricks
- **Favorita** or **M5** — realistic stress tests
