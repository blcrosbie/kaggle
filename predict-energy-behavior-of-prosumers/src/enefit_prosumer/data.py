from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl


@dataclass(frozen=True)
class CompetitionFiles:
    train: str = "train.csv"
    client: str = "client.csv"
    gas_prices: str = "gas_prices.csv"
    electricity_prices: str = "electricity_prices.csv"
    forecast_weather: str = "forecast_weather.csv"
    historical_weather: str = "historical_weather.csv"
    weather_station_to_county_mapping: str = "weather_station_to_county_mapping.csv"


class DataStorage:
    """Loads and incrementally updates Kaggle competition tables."""

    data_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
        "row_id",
    ]
    client_cols = [
        "product_type",
        "county",
        "eic_count",
        "installed_capacity",
        "is_business",
        "date",
    ]
    gas_prices_cols = ["forecast_date", "lowest_price_per_mwh", "highest_price_per_mwh"]
    electricity_prices_cols = ["forecast_date", "euros_per_mwh"]
    forecast_weather_cols = [
        "latitude",
        "longitude",
        "hours_ahead",
        "temperature",
        "dewpoint",
        "cloudcover_high",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_total",
        "10_metre_u_wind_component",
        "10_metre_v_wind_component",
        "forecast_datetime",
        "direct_solar_radiation",
        "surface_solar_radiation_downwards",
        "snowfall",
        "total_precipitation",
    ]
    historical_weather_cols = [
        "datetime",
        "temperature",
        "dewpoint",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloudcover_total",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",
        "direct_solar_radiation",
        "diffuse_radiation",
        "latitude",
        "longitude",
    ]
    location_cols = ["longitude", "latitude", "county"]
    target_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
    ]

    def __init__(
        self,
        root: str | Path = "input",
        files: CompetitionFiles | None = None,
        start_date: str | None = "2022-01-01",
    ) -> None:
        self.root = Path(root)
        self.files = files or CompetitionFiles()
        self.start_date = start_date

        self.df_data = self._read_csv(self.files.train, self.data_cols)
        self.df_client = self._read_csv(self.files.client, self.client_cols)
        self.df_gas_prices = self._read_csv(self.files.gas_prices, self.gas_prices_cols)
        self.df_electricity_prices = self._read_csv(
            self.files.electricity_prices, self.electricity_prices_cols
        ).drop_nulls()
        self.df_forecast_weather = self._read_csv(
            self.files.forecast_weather, self.forecast_weather_cols
        )
        self.df_historical_weather = self._read_csv(
            self.files.historical_weather, self.historical_weather_cols
        )
        self.df_weather_station_to_county_mapping = self._read_csv(
            self.files.weather_station_to_county_mapping, self.location_cols
        ).with_columns(
            pl.col("latitude").cast(pl.Float32),
            pl.col("longitude").cast(pl.Float32),
        )

        if start_date:
            self.df_data = self.df_data.filter(pl.col("datetime") >= pd.to_datetime(start_date))

        self.df_target = self.df_data.select(self.target_cols)
        self._capture_schemas()

    def _read_csv(self, filename: str, columns: list[str]) -> pl.DataFrame:
        path = self.root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download the Kaggle competition files into {self.root}."
            )
        return pl.read_csv(path, columns=columns, try_parse_dates=True)

    def _capture_schemas(self) -> None:
        self.schema_data = self.df_data.schema
        self.schema_client = self.df_client.schema
        self.schema_gas_prices = self.df_gas_prices.schema
        self.schema_electricity_prices = self.df_electricity_prices.schema
        self.schema_forecast_weather = self.df_forecast_weather.schema
        self.schema_historical_weather = self.df_historical_weather.schema
        self.schema_target = self.df_target.schema

    def update_with_new_data(
        self,
        df_new_client: pd.DataFrame,
        df_new_gas_prices: pd.DataFrame,
        df_new_electricity_prices: pd.DataFrame,
        df_new_forecast_weather: pd.DataFrame,
        df_new_historical_weather: pd.DataFrame,
        df_new_target: pd.DataFrame,
    ) -> None:
        """Append live API batches while preserving the latest unique rows."""

        new_client = pl.from_pandas(
            df_new_client[self.client_cols], schema_overrides=self.schema_client
        )
        new_gas = pl.from_pandas(
            df_new_gas_prices[self.gas_prices_cols],
            schema_overrides=self.schema_gas_prices,
        )
        new_electricity = pl.from_pandas(
            df_new_electricity_prices[self.electricity_prices_cols],
            schema_overrides=self.schema_electricity_prices,
        )
        new_forecast = pl.from_pandas(
            df_new_forecast_weather[self.forecast_weather_cols],
            schema_overrides=self.schema_forecast_weather,
        )
        new_historical = pl.from_pandas(
            df_new_historical_weather[self.historical_weather_cols],
            schema_overrides=self.schema_historical_weather,
        )
        new_target = pl.from_pandas(
            df_new_target[self.target_cols], schema_overrides=self.schema_target
        )

        self.df_client = pl.concat([self.df_client, new_client]).unique(
            ["date", "county", "is_business", "product_type"], keep="last"
        )
        self.df_gas_prices = pl.concat([self.df_gas_prices, new_gas]).unique(
            ["forecast_date"], keep="last"
        )
        self.df_electricity_prices = pl.concat(
            [self.df_electricity_prices, new_electricity]
        ).unique(["forecast_date"], keep="last")
        self.df_forecast_weather = pl.concat(
            [self.df_forecast_weather, new_forecast]
        ).unique(["forecast_datetime", "latitude", "longitude", "hours_ahead"], keep="last")
        self.df_historical_weather = pl.concat(
            [self.df_historical_weather, new_historical]
        ).unique(["datetime", "latitude", "longitude"], keep="last")
        self.df_target = pl.concat([self.df_target, new_target]).unique(
            ["datetime", "county", "is_business", "product_type", "is_consumption"],
            keep="last",
        )

    def preprocess_test(self, df_test: pd.DataFrame) -> pl.DataFrame:
        """Convert Kaggle API test rows into the same shape as training rows."""

        renamed = df_test.rename(columns={"prediction_datetime": "datetime"})
        return pl.from_pandas(renamed[self.data_cols[1:]], schema_overrides=self.schema_data)
