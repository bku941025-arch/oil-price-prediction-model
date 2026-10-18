# Notebooks — original exploration (archived)

These notebooks are how the project started: building the Montréal daily dataset,
exploring features, and comparing Ridge, XGBoost and LightGBM. They are kept for
reference and are **not** used by the live site.

The production code in [`gasprix/`](../gasprix) supersedes them. It covers all
Kalibrate cities with one pooled model, uses market data only as it was published at
the time, removes tax step changes, and calibrates its ranges out of sample. Its results
are reproducible with `python -m gasprix.backtest`.

To rerun the notebooks, run them in order. Each writes the CSV the next one reads
(`Data/montreal_gas_features_daily.csv`, then `Data/montreal_gas_ml_ready.csv`):

1. `montreal_gas_daily_build.ipynb` (set `FRED_API_KEY` in your environment)
2. `montreal_gas_feature_engineering.ipynb`
3. `montreal_gas_modelling.ipynb`
4. `montreal_gas_inference.ipynb`
