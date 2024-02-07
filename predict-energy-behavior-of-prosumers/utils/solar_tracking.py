#!/usr/bin/env python
# coding: utf-8

import math
import datetime

# Solar Tracking Helpers


def calculate_declination(day_of_year: int = 1, cycle_start = None):
    """
    declination angle is at most 23.45 degrees magnitude
    """

    if cycle_start == 'spring':
        offset = -81
    elif cycle_start == 'autumn':
        offset = 284
    else:
        offset = 10
        
    min_declination_angle = -23.45
    total_days_in_year = 365.25
    total_degrees = 360
    
    relative_day = (day_of_year + offset) / total_days_in_year
    relative_angle = relative_day * total_degrees
    
    deca = min_declination_angle * math.cos(math.radians(relative_angle))
    
    return deca


def local_standard_time_meridian(utc_offset):
    """ 1 minute = 15 degrees
    """
    return 15 * utc_offset


def equation_of_time(day_of_year, cycle_start = 'spring'):
    
    if cycle_start == 'spring':
        offset = -81
    elif cycle_start == 'autumn':
        offset = 284
    else:
        offset = 10
        
    total_days_in_year = 365.25
    total_degrees = 360
    total_radians = 2 * math.pi
    
    B = total_radians * (day_of_year + offset) / total_days_in_year
    EoT = 9.87 * math.sin(2*B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    return EoT


def time_correction_factor(day_of_year, longitude, utc_offset=None):
    
    if utc_offset is None:
        # Approximate the UTC offset based on rounding by the nearest 15 degree
        utc_offset = round(longitude / 15)
        
    LSTM = local_standard_time_meridian(utc_offset)
    
    EoT = equation_of_time(day_of_year)
    
    TC = 4 * (longitude - LSTM) + EoT
    return TC
    

def local_solar_time(local_time, day_of_year, longitude, utc_offset=None):
    TC = time_correction_factor(day_of_year, longitude, utc_offset)
    LST = local_time + datetime.timedelta(hours=(TC/60))
    return LST


def hour_angle(local_time, day_of_year, longitude, utc_offset=None):
    LST = local_solar_time(local_time, day_of_year, longitude, utc_offset)    
    LST_value = LST.hour + LST.minute/60 + LST.second/3600
    HRA = 15 * (LST_value - 12)
    return HRA


def calculate_elevation_angle(local_time, day_of_year, longitude, latitude, utc_offset=None):
    HRA = hour_angle(local_time, day_of_year, longitude, utc_offset)
    deca = calculate_declination(day_of_year)
    
    sines = math.sin(math.radians(deca)) * math.sin(math.radians(latitude))
    cosines = math.cos(math.radians(latitude)) * math.cos(math.radians(latitude)) * math.cos(math.radians(HRA))
    try:
        elevation = math.degrees(math.asin(sines + cosines))
    except Exception as e:
        print(e)
        print(local_time, day_of_year, longitude, latitude, utc_offset)
        print(sines)
        print(cosines)
        print(deca)
        print(HRA)

    return elevation


def calculate_angles(local_time, day_of_year, longitude, latitude, utc_offset=None):
    HRA = hour_angle(local_time, day_of_year, longitude, utc_offset)
    deca = calculate_declination(day_of_year)
    elevation = calculate_elevation_angle(local_time, day_of_year, longitude, latitude)
    
    first_term = math.sin(math.radians(deca)) * math.cos(math.radians(latitude))
    second_term = math.cos(math.radians(deca)) * math.sin(math.radians(latitude)) * math.cos(math.radians(HRA))
    
    denom = math.cos(math.radians(elevation))
    
    azimuth = math.degrees(math.acos((first_term-second_term)/denom))
    return azimuth, elevation


def convert_int_to_time(hour_int_val):
    
    actual_hour = math.floor(hour_int_val)
    minute_value = (hour_int_val - actual_hour) * 60
    actual_minute = math.floor(minute_value)
    second_value = (minute_value - actual_minute) * 60
    actual_second = max(math.floor(second_value), 59)
    
    return datetime.time(hour=actual_hour, minute=actual_minute, second=actual_second)
    

def when_is_daylight(day_of_year, longitude, latitude, utc_offset=None):
    
    deca = calculate_declination(day_of_year)
    sines_numerator = -1 * math.sin(math.radians(latitude)) * math.sin(math.radians(deca))
    cosines_denom = math.cos(math.radians(latitude)) * math.cos(math.radians(deca))
    res = math.degrees(math.acos(sines_numerator/cosines_denom))
    TC = time_correction_factor(day_of_year, longitude, utc_offset)
    
    start_value = 12 - res/15 - TC/60
    end_value = 12 + res/15 - TC/60
    
    sunrise = convert_int_to_time(start_value)
    sunset = convert_int_to_time(end_value)
    
    return sunrise, sunset
    

def calculate_irradiation_on_surface(elevation, tilt, direct_irradiation, albedo = 0):
    incident_irradiation = direct_irradiation * math.sin(math.radians(elevation))
    panel_irradiation = incident_irradiation * math.sin(math.radians(tilt + elevation))
    return panel_irradiation
