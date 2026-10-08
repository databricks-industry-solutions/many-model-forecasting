# Databricks notebook source
# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. 
# MAGIC
# MAGIC The sources in all notebooks in this directory and the sub-directories are provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | omegaconf | A flexible configuration library | BSD | https://pypi.org/project/omegaconf/
# MAGIC | datasetsforecast | Datasets for Time series forecasting | MIT | https://pypi.org/project/datasetsforecast/
# MAGIC | statsforecast | Time series forecasting suite using statistical models | Apache 2.0 | https://pypi.org/project/statsforecast/
# MAGIC | mlforecast | Time series forecasting suite using machine learning models | Apache 2.0 | https://pypi.org/project/mlforecast/
# MAGIC | neuralforecast | Time series forecasting suite using deep learning models | Apache 2.0 | https://pypi.org/project/neuralforecast/
# MAGIC | sktime | A unified framework for machine learning with time series | BSD 3-Clause | https://pypi.org/project/sktime/
# MAGIC | tbats | BATS and TBATS for time series forecasting | MIT | https://pypi.org/project/tbats/
# MAGIC | lightgbm | LightGBM Python Package | MIT | https://pypi.org/project/lightgbm/
# MAGIC | Chronos | Pretrained (Language) Models for Probabilistic Time Series Forecasting | Apache 2.0 | https://github.com/amazon-science/chronos-forecasting
# MAGIC | Moirai | Unified Training of Universal Time Series Forecasting Transformers | Apache 2.0 | https://github.com/SalesforceAIResearch/uni2ts
# MAGIC | TimesFM | A pretrained time-series foundation model developed by Google Research for time-series forecasting | Apache 2.0 | https://github.com/google-research/timesfm
# MAGIC | hierarchicalforecast | Hierarchical forecast reconciliation methods | Apache 2.0 | https://pypi.org/project/hierarchicalforecast/
# MAGIC | polars | Fast multi-threaded DataFrame library | MIT | https://pypi.org/project/polars/
# MAGIC | foundationforecast | Foundation time series forecasting models (TimeCopilot) | Apache 2.0 | https://pypi.org/project/foundationforecast/
# MAGIC | granite-tsfm | IBM time series foundation model utilities (FlowState, PatchTST-FM) | Apache 2.0 | https://pypi.org/project/granite-tsfm/
# MAGIC | tfc-t0 | T0 time series foundation model from The Forecasting Company | Apache 2.0 | https://github.com/theforecastingcompany/tfc-t0
# MAGIC
# MAGIC The FoundationForecast models (`FF*`) download the following checkpoints from Hugging Face. Each checkpoint is subject to the license on its model card.
# MAGIC
# MAGIC | model | checkpoint | license | source |
# MAGIC |-------|------------|---------|--------|
# MAGIC | FFChronos2 | amazon/chronos-2 | Apache 2.0 | https://huggingface.co/amazon/chronos-2
# MAGIC | FFChronos2Small | autogluon/chronos-2-small | Apache 2.0 | https://huggingface.co/autogluon/chronos-2-small
# MAGIC | FFTimesFM_2_5_200m | google/timesfm-2.5-200m-pytorch | Apache 2.0 | https://huggingface.co/google/timesfm-2.5-200m-pytorch
# MAGIC | FFTimesFM_3_0 | google/timesfm-3.0-pytorch | TimesFM Non-Commercial License v1.0 | https://huggingface.co/google/timesfm-3.0-pytorch
# MAGIC | FFTiRex2 | NX-AI/TiRex-2 | Apache 2.0 | https://huggingface.co/NX-AI/TiRex-2
# MAGIC | FFToto | Datadog/Toto-Open-Base-1.0 | Apache 2.0 | https://huggingface.co/Datadog/Toto-Open-Base-1.0
# MAGIC | FFToto2_22m | Datadog/Toto-2.0-22m | Apache 2.0 | https://huggingface.co/Datadog/Toto-2.0-22m
# MAGIC | FFToto2_313m | Datadog/Toto-2.0-313m | Apache 2.0 | https://huggingface.co/Datadog/Toto-2.0-313m
# MAGIC | FFToto2_1B | Datadog/Toto-2.0-1B | Apache 2.0 | https://huggingface.co/Datadog/Toto-2.0-1B
# MAGIC | FFFlowState | ibm-research/flowstate | Apache 2.0 | https://huggingface.co/ibm-research/flowstate
# MAGIC | FFPatchTSTFM_R2 | ibm-granite/granite-timeseries-patchtst-fm-r2 | OpenMDW 1.0 | https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2
# MAGIC | FFPatchTSTFM_R1 | ibm-research/patchtst-fm-r1 | CC BY-NC-SA 4.0 | https://huggingface.co/ibm-research/patchtst-fm-r1
# MAGIC | FFTafsutBase | Tafsut-FM/tafsut-univariate-base | MIT | https://huggingface.co/Tafsut-FM/tafsut-univariate-base
# MAGIC | FFT0Beta | theforecastingcompany/t0-beta | Apache 2.0 | https://huggingface.co/theforecastingcompany/t0-beta

# COMMAND ----------
