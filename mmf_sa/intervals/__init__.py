"""Prediction intervals: configuration, the quantile contract with the models, calibration and scoring.

The modules are config, quantiles, conformal, features, scoring, spark and calibrator. Importing this package loads
nothing; import the module that you need. Models import only intervals.quantiles (and intervals.config for the level
helpers). mmf_sa.calibration and mmf_sa.calibration_spark re-export the names of this package.
"""
