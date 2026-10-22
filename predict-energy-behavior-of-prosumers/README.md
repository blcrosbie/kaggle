# Enefit Prosumer Forecasting

This is a revived version of an old Kaggle notebook project for **Enefit - Predict Energy Behavior of Prosumers**. The original work explored weather, calendar, customer, lag, and solar features for predicting Estonian prosumer electricity consumption and production. The revamp turns that notebook work into a reusable Python package with a CLI, tests, and a clearer story.

## Why This Is Interesting

Energy forecasting is a practical grid operations problem: prosumers both consume and produce electricity, and bad forecasts create imbalance costs. This project focuses on the patterns that matter operationally:

- Hourly and seasonal demand cycles.
- Production versus consumption behavior.
- County, business, and product-type segments.
- Weather station rollups at national and local levels.
- Lagged target features over 2 to 14 day windows.
- Estonian holidays and customer installed capacity.
- Solar-position utilities for future production feature work.

## Project Layout

```text
src/enefit_prosumer/
  data.py       # competition CSV loader and API-style incremental updates
  features.py   # feature pipeline extracted from notebooks
  model.py      # XGBoost MAE training helper
  solar.py      # solar geometry utilities
  cli.py        # terminal entrypoint
tests/          # lightweight regression tests
*.ipynb         # original exploration and Kaggle submission notebooks
```

## Setup

Create an environment and install the package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,notebooks]"
```

Download the Kaggle competition CSV files and place them in `input/`:

```text
input/
  train.csv
  client.csv
  gas_prices.csv
  electricity_prices.csv
  forecast_weather.csv
  historical_weather.csv
  weather_station_to_county_mapping.csv
```

## Train

```powershell
enefit-prosumer train --input-dir input --output models/xgb.joblib
```

The command builds the feature matrix, trains an XGBoost model with categorical support, and saves both the model and training column order.

## Revamp Notes

The old notebooks are still preserved as research history, but the main path is now code-first:

- `DataStorage` owns file loading and live API-style updates.
- `FeatureEngineer` owns feature generation.
- `fit_model` owns model construction and training defaults.
- Tests cover fragile date and solar behavior.

The next high-value additions are time-series validation reports, feature importance plots, and a small dashboard or article-ready notebook that explains the operational story behind the model.
