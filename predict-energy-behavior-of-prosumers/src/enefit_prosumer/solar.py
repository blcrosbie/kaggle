from __future__ import annotations

import datetime as dt
import math


def calculate_declination(day_of_year: int = 1, cycle_start: str | None = None) -> float:
    """Approximate solar declination angle in degrees."""

    offset = {"spring": -81, "autumn": 284}.get(cycle_start, 10)
    relative_angle = ((day_of_year + offset) / 365.25) * 360
    return -23.45 * math.cos(math.radians(relative_angle))


def local_standard_time_meridian(utc_offset: float) -> float:
    return 15 * utc_offset


def equation_of_time(day_of_year: int, cycle_start: str = "spring") -> float:
    offset = {"spring": -81, "autumn": 284}.get(cycle_start, 10)
    b = 2 * math.pi * (day_of_year + offset) / 365.25
    return 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def time_correction_factor(day_of_year: int, longitude: float, utc_offset: float | None = None) -> float:
    if utc_offset is None:
        utc_offset = round(longitude / 15)
    return 4 * (longitude - local_standard_time_meridian(utc_offset)) + equation_of_time(day_of_year)


def local_solar_time(
    local_time: dt.datetime,
    day_of_year: int,
    longitude: float,
    utc_offset: float | None = None,
) -> dt.datetime:
    correction_minutes = time_correction_factor(day_of_year, longitude, utc_offset)
    return local_time + dt.timedelta(minutes=correction_minutes)


def hour_angle(
    local_time: dt.datetime,
    day_of_year: int,
    longitude: float,
    utc_offset: float | None = None,
) -> float:
    solar_time = local_solar_time(local_time, day_of_year, longitude, utc_offset)
    decimal_hour = solar_time.hour + solar_time.minute / 60 + solar_time.second / 3600
    return 15 * (decimal_hour - 12)


def calculate_elevation_angle(
    local_time: dt.datetime,
    day_of_year: int,
    longitude: float,
    latitude: float,
    utc_offset: float | None = None,
) -> float:
    hra = hour_angle(local_time, day_of_year, longitude, utc_offset)
    declination = calculate_declination(day_of_year)

    value = (
        math.sin(math.radians(declination)) * math.sin(math.radians(latitude))
        + math.cos(math.radians(declination))
        * math.cos(math.radians(latitude))
        * math.cos(math.radians(hra))
    )
    return math.degrees(math.asin(max(-1, min(1, value))))


def convert_int_to_time(hour_value: float) -> dt.time:
    hour = math.floor(hour_value)
    minute_value = (hour_value - hour) * 60
    minute = math.floor(minute_value)
    second = round((minute_value - minute) * 60)
    return dt.time(hour=hour % 24, minute=minute, second=min(second, 59))


def calculate_irradiation_on_surface(
    elevation: float,
    tilt: float,
    direct_irradiation: float,
    albedo: float = 0,
) -> float:
    del albedo
    incident_irradiation = direct_irradiation * math.sin(math.radians(elevation))
    return incident_irradiation * math.sin(math.radians(tilt + elevation))
