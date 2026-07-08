# FreshRetailNet — Serverless MMF example

An end-to-end example that runs Many-Model Forecasting (MMF) on the public
**FreshRetailNet** dataset on **Databricks Serverless GPU**, then makes the
forecast output consumable in natural language via a Databricks Genie Space.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_fresh_retail_net_data_prep.ipynb` | Loads the public FreshRetailNet dataset (commercial-permissive license) and prepares it for MMF. Runs on standard Serverless. |
| `02_fresh_retail_net_mmf_forecast.ipynb` | Runs MMF foundation models on the prepared dataset. Requires **A10 Serverless GPU** compute — see the setup section at the top of the notebook. |
| `03_build_product_location_dims.ipynb` | Builds illustrative product and location dimension tables so FreshRetailNet's anonymized integer IDs (`product_id`, `city_id`, category IDs) map to friendly labels, making the forecast queryable in natural language. |
| `04_genie_views_setup.ipynb` | Explodes MMF's `scoring_output` / `evaluation_output` 7-element `ARRAY` columns into flat, one-row-per-day views so Databricks Genie (natural-language-to-SQL) can reason over the forecast output. |

Notebooks `01` and `02` are the core MMF pipeline; `03` and `04` are
complementary and self-contained — they build on the forecast output for a
downstream Genie / BI demo and do not modify `01`/`02`.

## Compute

All notebooks are written for the **Serverless GPU (Spark Connect)** environment
that `examples/serverless/` targets. `02` requires an A10 GPU instance
(environment version 5); the others run on standard Serverless.

## Data note

FreshRetailNet anonymizes all entities to integer IDs — there are no product
names or regions in the source data. The product and location labels created in
`03_build_product_location_dims.ipynb` (e.g. "Skim Milk," "Northeast") are
**illustrative only** and are not real data from any customer.

## Credits

- Core FreshRetailNet MMF notebooks (`01`, `02`): Ryuta Yoshimatsu (Databricks).
- Serverless-GPU data-prep fixes in `01` (guarded catalog creation; removal of
  `.cache()` on Spark Connect) and the complementary notebooks (`03`, `04`):
  Venkatavaradhan Viswanathan (AWS, [@venkatavaradhanv](https://github.com/venkatavaradhanv)).

This example accompanies the joint Databricks × AWS blog on multi-modal
supply-chain forecasting with MMF, Databricks Genie, and Amazon Quick.
