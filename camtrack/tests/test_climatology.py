#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kara Hartig

Unit tests for climatology.py
"""
# Standard imports
import numpy as np
from netCDF4 import Dataset
import os

# Testing imports
from numpy.testing import assert_raises, assert_allclose

# camtrack imports
from camtrack.climatology import anomaly_DJF, find_coldest, OLD_find_coldest
from camtrack.data import subset_nc

# Make simple "data" dictionaries to work as inputs
SAMPLE_DATA = np.array([ [[2, 0], [0, 2]], [[1, 1], [1, 1]] ])
TILED_DATA = np.tile(SAMPLE_DATA, (2, 1, 1))  # extended along 'time' axis
SAMPLE_DICT_1 = {'data': SAMPLE_DATA, 'time': [0, 1], 'lat': [0.5, 1.5], 'lon': [0.5, 1.5]}
SAMPLE_DICT_2 = {'data': TILED_DATA, 'time': [5, 6, 7, 8], 'lat': [0.5, 1.5], 'lon': [0.5, 1.5]}
BAD_LAT_DICT = {'data': TILED_DATA, 'time': [5, 6, 7, 8], 'lat': [0.5, 1.5, 2.0], 'lon': [0.5, 1.5]}
BAD_LON_DICT = {'data': TILED_DATA, 'time': [5, 6, 7, 8], 'lat': [0.5, 1.5], 'lon': [0.5, 1.5, 2.0]}

# Make simple climatology dictionaries (like output of anomaly_DFJ)
SAMPLE_TEMP_ANOMALY = np.array([ [[5, 4], [3, 2]], [[-5, -4], [-3, np.nan]] ])
COORDS_COLDEST_TEMP_ANOMALY = (1, 0.5, 0.5) # in the order (time, lat, lon)
SAMPLE_CLIMATOLOGY_DICT = {'diff': SAMPLE_TEMP_ANOMALY, 'time': [0, 1], 'lat': [0.5, 1.5], 'lon': [0.5, 1.5]}

# Sample netCDF file from CAM4
# time range: 0008-12-01 00:00:00 to 0008-12-07 00:00:00 (YYYY-MM-DD HH:MM:SS)
# latitude range: 80.52631579 to 90.0
# longitude range: 320.0 to 330.0 (on 0-360 scale)
NC_SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_CAM4_for_nosetests.nc')
NC_SAMPLE_FILE = Dataset(NC_SAMPLE_PATH)
WINTER_IDX = 1 # index 1 corresponds to 08-09 winter
COLDEST_LAND_TEMPERATURE = 236.67903 # using np.nanmin(temperature masked by landfraction)
THIRD_COLDEST_LAND_TEMPERATURE = 236.76225
FOURTH_COLDEST_LAND_TEMPERATURE = 236.81648

# Absolute method: value - mean
def test_absolute_method():
    dict_list = [SAMPLE_DICT_1, SAMPLE_DICT_2]
    expected_diff = np.tile(0.5*np.array([ [[1, -1], [-1, 1]], [[-1, 1], [1, -1]] ]), (3, 1, 1))
    expected_time = [0, 1, 5, 6, 7, 8]
    expected_lat = [0.5, 1.5]
    expected_lon = [0.5, 1.5]
    diff_dict = anomaly_DJF(dict_list, 'absolute')
    assert_allclose(diff_dict['diff'], expected_diff)
    assert_allclose(diff_dict['time'], expected_time)
    assert_allclose(diff_dict['lat'], expected_lat)
    assert_allclose(diff_dict['lon'], expected_lon)

# Scaled method: (value - mean) / stdev
def test_scaled_method():
    dict_list = [SAMPLE_DICT_1, SAMPLE_DICT_2]
    expected_diff = np.tile(np.array([ [[1, -1], [-1, 1]], [[-1, 1], [1, -1]] ]), (3, 1, 1))
    expected_time = [0, 1, 5, 6, 7, 8]
    expected_lat = [0.5, 1.5]
    expected_lon = [0.5, 1.5]
    diff_dict = anomaly_DJF(dict_list, 'scaled')
    assert_allclose(diff_dict['diff'], expected_diff)
    assert_allclose(diff_dict['time'], expected_time)
    assert_allclose(diff_dict['lat'], expected_lat)
    assert_allclose(diff_dict['lon'], expected_lon)

# Raised Errors
def test_different_lat():
    bad_dict_list = [SAMPLE_DICT_1, BAD_LAT_DICT]
    assert_raises(ValueError, anomaly_DJF, bad_dict_list, 'absolute')

def test_different_lon():
    bad_dict_list = [SAMPLE_DICT_1, BAD_LON_DICT]
    assert_raises(ValueError, anomaly_DJF, bad_dict_list, 'absolute')

def test_invalid_method():
    bad_method = 'null'
    dict_list = [SAMPLE_DICT_1, SAMPLE_DICT_2]
    assert_raises(ValueError, anomaly_DJF, dict_list, bad_method)

# tests for find_coldest
def test_sort_find_coldest():
    null_distinct = {'min time separation': 0.01, 'min lat separation': 0.01, 'min lon separation': 0.01}
    cold_events = find_coldest(SAMPLE_CLIMATOLOGY_DICT, 1, null_distinct)
    assert_allclose(cold_events.shape, (1, 5))
    assert_allclose(cold_events['temp anomaly'][0], np.nanmin(SAMPLE_TEMP_ANOMALY))
    assert_allclose((cold_events['time'][0], cold_events['lat'][0], cold_events['lon'][0]), COORDS_COLDEST_TEMP_ANOMALY)

# tests for OLD_find_coldest:
def test_find_coldest():
    data_dict = subset_nc(NC_SAMPLE_PATH, WINTER_IDX, 'TREFHT', -np.inf, (-np.inf, np.inf), testing=True)
    null_distinct = {'min time separation': 0.01, 'min lat separation': 0.01, 'min lon separation': 0.01}
    cold_events = OLD_find_coldest(data_dict, WINTER_IDX, 1, null_distinct)
    assert_allclose(cold_events.shape, (1, 5))
    assert_allclose(cold_events['2m temp'][0], COLDEST_LAND_TEMPERATURE)

def test_find_distinct_in_time():
    # mark coldest two as indistinct using time separation
    data_dict = subset_nc(NC_SAMPLE_PATH, WINTER_IDX, 'TREFHT', -np.inf, (-np.inf, np.inf), testing=True)
    time_distinct = {'min time separation': 0.1, 'min lat separation': 10.0, 'min lon separation': 10.0} # 1st and 2nd coldest indistinct, 3rd is distinct from first two in time
    cold_events = OLD_find_coldest(data_dict, WINTER_IDX, 2, time_distinct)
    assert_allclose(cold_events['2m temp'][1], THIRD_COLDEST_LAND_TEMPERATURE)

def test_find_distinct_in_latlon():
    # mark coldest three as indistinct using lat/lon separation
    data_dict = subset_nc(NC_SAMPLE_PATH, WINTER_IDX, 'TREFHT', -np.inf, (-np.inf, np.inf), testing=True)
    lon_distinct = {'min time separation': 3.0, 'min lat separation': 10.0, 'min lon separation': 3.0} # 1st-2nd-3rd coldest indistinct, 4th is distinct from first three in longitude
    cold_events = OLD_find_coldest(data_dict, WINTER_IDX, 2, lon_distinct)
    assert_allclose(cold_events['2m temp'][1], FOURTH_COLDEST_LAND_TEMPERATURE)