import datetime as dt

from enefit_prosumer.solar import (
    calculate_elevation_angle,
    convert_int_to_time,
    equation_of_time,
)


def test_equation_of_time_is_reasonable_near_spring_equinox():
    assert -15 < equation_of_time(81) < 15


def test_midday_summer_sun_is_high_in_estonia():
    elevation = calculate_elevation_angle(
        dt.datetime(2022, 6, 21, 12),
        day_of_year=172,
        longitude=25.0,
        latitude=59.0,
        utc_offset=3,
    )
    assert 45 < elevation < 60
