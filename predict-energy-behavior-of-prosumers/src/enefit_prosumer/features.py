from __future__ import annotations

import datetime as dt

import holidays
import numpy as np
import pandas as pd
import polars as pl


class FeatureEngineer:
    """Feature pipeline extracted from the Kaggle notebooks."""

    categorical_columns = [
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "segment",
    ]

    def __init__(self, data_storage, holiday_years: range = range(2021, 2026)) -> None:
        self.data = data_storage
        self.estonian_holidays = set(holidays.country_holidays("EE", years=holiday_years).keys())

    def generate_features(self, df_prediction_items: pl.DataFrame) -> pd.DataFrame:
        if "target" in df_prediction_items.columns:
            df_prediction_items, y = (
                df_prediction_items.drop("target"),
                df_prediction_items.select("target"),
            )
        else:
            y = None

        df_features = df_prediction_items.with_columns(
            pl.col("datetime").cast(pl.Date).alias("date")
        )

        for add_features in [
            self._add_general_features,
            self._add_client_features,
            self._add_forecast_weather_features,
            self._add_historical_weather_features,
            self._add_target_features,
            self._add_holidays_features,
            self._reduce_memory_usage,
            self._drop_columns,
        ]:
            df_features = add_features(df_features)

        return self._to_pandas(df_features, y)

    def _add_general_features(self, df_features: pl.DataFrame) -> pl.DataFrame:
        return (
            df_features.with_columns(
                pl.col("datetime").dt.ordinal_day().alias("dayofyear"),
                pl.col("datetime").dt.hour().alias("hour"),
                pl.col("datetime").dt.day().alias("day"),
                pl.col("datetime").dt.weekday().alias("weekday"),
                pl.col("datetime").dt.month().alias("month"),
                pl.col("datetime").dt.year().alias("year"),
            )
            .with_columns(
                pl.concat_str(
                    "county",
                    "is_business",
                    "product_type",
                    "is_consumption",
                    separator="_",
                ).alias("segment")
            )
            .with_columns(
                (np.pi * pl.col("dayofyear") / 183).sin().alias("sin_dayofyear"),
                (np.pi * pl.col("dayofyear") / 183).cos().alias("cos_dayofyear"),
                (np.pi * pl.col("hour") / 12).sin().alias("sin_hour"),
                (np.pi * pl.col("hour") / 12).cos().alias("cos_hour"),
            )
        )

    def _add_client_features(self, df_features: pl.DataFrame) -> pl.DataFrame:
        df_client = self.data.df_client.with_columns(
            (pl.col("date") + pl.duration(days=2)).cast(pl.Date)
        )
        return df_features.join(
            df_client,
            on=["county", "is_business", "product_type", "date"],
            how="left",
        )

    def _add_holidays_features(self, df_features: pl.DataFrame) -> pl.DataFrame:
        return df_features.with_columns(
            pl.struct(["year", "month", "day"])
            .map_elements(self._is_country_holiday, return_dtype=pl.Boolean)
            .alias("is_country_holiday")
        )

    def _is_country_holiday(self, row: dict[str, int]) -> bool:
        return dt.date(row["year"], row["month"], row["day"]) in self.estonian_holidays

    def _add_forecast_weather_features(self, df_features: pl.DataFrame) -> pl.DataFrame:
        df_forecast_weather = (
            self.data.df_forecast_weather.rename({"forecast_datetime": "datetime"})
            .filter((pl.col("hours_ahead") >= 22) & (pl.col("hours_ahead") <= 45))
            .drop("hours_ahead")
            .with_columns(
                pl.col("latitude").cast(pl.Float32),
                pl.col("longitude").cast(pl.Float32),
            )
            .join(
                self.data.df_weather_station_to_county_mapping,
                how="left",
                on=["longitude", "latitude"],
            )
            .drop("longitude", "latitude")
        )
        return self._join_weather_rollups(df_features, df_forecast_weather, [0, 7 * 24], "forecast")

    def _add_historical_weather_features(self, df_features: pl.DataFrame) -> pl.DataFrame:
        df_historical_weather = (
            self.data.df_historical_weather.with_columns(
                pl.col("latitude").cast(pl.Float32),
                pl.col("longitude").cast(pl.Float32),
            )
            .join(
                self.data.df_weather_station_to_county_mapping,
                how="left",
                on=["longitude", "latitude"],
            )
            .drop("longitude", "latitude")
        )
        df_features = self._join_weather_rollups(
            df_features, df_historical_weather, [2 * 24, 7 * 24], "historical"
        )

        morning_history = (
            self._weather_by_datetime(df_historical_weather)
            .with_columns(
                pl.col("datetime") + pl.duration(hours=24),
                pl.col("datetime").dt.hour().alias("hour"),
            )
            .filter(pl.col("hour") <= 10)
            .drop("hour")
        )
        return df_features.join(
            morning_history,
            on="datetime",
            how="left",
            suffix="_historical_24h",
        )

    def _join_weather_rollups(
        self,
        df_features: pl.DataFrame,
        weather: pl.DataFrame,
        hour_lags: list[int],
        name: str,
    ) -> pl.DataFrame:
        weather_by_datetime = self._weather_by_datetime(weather)
        weather_by_county = self._weather_by_county_datetime(weather)

        for hours_lag in hour_lags:
            df_features = df_features.join(
                weather_by_datetime.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)),
                on="datetime",
                how="left",
                suffix=f"_{name}_{hours_lag}h",
            )
            df_features = df_features.join(
                weather_by_county.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)),
                on=["county", "datetime"],
                how="left",
                suffix=f"_{name}_local_{hours_lag}h",
            )

        return df_features

    @staticmethod
    def _weather_by_datetime(weather: pl.DataFrame) -> pl.DataFrame:
        return weather.group_by("datetime").mean().drop("county")

    @staticmethod
    def _weather_by_county_datetime(weather: pl.DataFrame) -> pl.DataFrame:
        return weather.filter(pl.col("county").is_not_null()).group_by("county", "datetime").mean()

    def _add_target_features(self, df_features: pl.DataFrame) -> pl.DataFrame:
        df_target = self.data.df_target
        target_all_type_sum = (
            df_target.group_by(["datetime", "county", "is_business", "is_consumption"])
            .sum()
            .drop("product_type")
        )
        target_all_county_type_sum = (
            df_target.group_by(["datetime", "is_business", "is_consumption"])
            .sum()
            .drop("product_type", "county")
        )

        hours_list = [i * 24 for i in range(2, 15)]
        for hours_lag in hours_list:
            df_features = df_features.join(
                df_target.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)).rename(
                    {"target": f"target_{hours_lag}h"}
                ),
                on=["county", "is_business", "product_type", "is_consumption", "datetime"],
                how="left",
            )

        for hours_lag in [2 * 24, 3 * 24, 7 * 24, 14 * 24]:
            df_features = df_features.join(
                target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)).rename(
                    {"target": f"target_all_type_sum_{hours_lag}h"}
                ),
                on=["county", "is_business", "is_consumption", "datetime"],
                how="left",
            )
            df_features = df_features.join(
                target_all_county_type_sum.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ).rename({"target": f"target_all_county_type_sum_{hours_lag}h"}),
                on=["is_business", "is_consumption", "datetime"],
                how="left",
                suffix=f"_all_county_type_sum_{hours_lag}h",
            )

        stats_cols = [f"target_{hours_lag}h" for hours_lag in hours_list[:4]]
        df_features = df_features.with_columns(
            pl.mean_horizontal(stats_cols).alias("target_mean"),
            pl.concat_list(stats_cols).list.std().alias("target_std"),
        )

        for target_prefix, numerator, denominator in [
            ("target", 24 * 7, 24 * 14),
            ("target", 24 * 2, 24 * 9),
            ("target", 24 * 3, 24 * 10),
            ("target", 24 * 2, 24 * 3),
            ("target_all_type_sum", 24 * 2, 24 * 3),
            ("target_all_type_sum", 24 * 7, 24 * 14),
            ("target_all_county_type_sum", 24 * 2, 24 * 3),
            ("target_all_county_type_sum", 24 * 7, 24 * 14),
        ]:
            df_features = df_features.with_columns(
                (pl.col(f"{target_prefix}_{numerator}h") / (pl.col(f"{target_prefix}_{denominator}h") + 1e-3))
                .alias(f"{target_prefix}_ratio_{numerator}_{denominator}")
            )

        return df_features

    @staticmethod
    def _reduce_memory_usage(df_features: pl.DataFrame) -> pl.DataFrame:
        return df_features.with_columns(pl.col(pl.Float64).cast(pl.Float32))

    @staticmethod
    def _drop_columns(df_features: pl.DataFrame) -> pl.DataFrame:
        return df_features.drop("date", "datetime", "hour", "dayofyear")

    def _to_pandas(self, df_features: pl.DataFrame, y: pl.DataFrame | None) -> pd.DataFrame:
        if y is not None:
            df_pandas = pd.concat([df_features.to_pandas(), y.to_pandas()], axis=1)
        else:
            df_pandas = df_features.to_pandas()

        df_pandas = df_pandas.set_index("row_id")
        existing_cat_cols = [col for col in self.categorical_columns if col in df_pandas.columns]
        df_pandas[existing_cat_cols] = df_pandas[existing_cat_cols].astype("category")
        return df_pandas
