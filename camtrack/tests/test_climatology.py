#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kara Hartig

Unit tests for climatology.py
"""
# Standard imports
import numpy as np

# Testing imports
from numpy.testing import assert_raises, assert_allclose

# camtrack imports
from camtrack.climatology import compare_to_climatology, sort_and_find_coldest

# Make simple "data" dictionaries to work as inputs
SAMPLE_DATA = np.array([ [[2, 0], [0, 2]], [[1, 1], [1, 1]] ])
TILED_DATA = np.tile(SAMPLE_DATA, (2, 1, 1))  # extended along 'time' axis
SAMPLE_DICT_1 = {'data': SAMPLE_DATA, 'time': [0, 1], 'lat': [0.5, 1.5], 'lon': [0.5, 1.5]}
SAMPLE_DICT_2 = {'data': TILED_DATA, 'time': [5, 6, 7, 8], 'lat': [0.5, 1.5], 'lon': [0.5, 1.5]}
BAD_LAT_DICT = {'data': TILED_DATA, 'time': [5, 6, 7, 8], 'lat': [0.5, 1.5, 2.0], 'lon': [0.5, 1.5]}
BAD_LON_DICT = {'data': TILED_DATA, 'time': [5, 6, 7, 8], 'lat': [0.5, 1.5], 'lon': [0.5, 1.5, 2.0]}

# Make simple climatology dictionaries (like output of compare_to_climatology)
# Make sample climatology dict
SAMPLE_TEMP_ANOMALY = np.array([ [[5, 4], [3, 2]], [[-5, -4], [-3, np.nan]] ])
COORDS_COLDEST_TEMP_ANOMALY = (1, 0.5, 0.5) # in the order (time, lat, lon)
SAMPLE_CLIMATOLOGY_DICT = {'diff': SAMPLE_TEMP_ANOMALY, 'time': [0, 1], 'lat': [0.5, 1.5], 'lon': [0.5, 1.5]}

# Absolute method: value - mean
def test_absolute_method():
    dict_list = [SAMPLE_DICT_1, SAMPLE_DICT_2]
    expected_diff = np.tile(0.5*np.array([ [[1, -1], [-1, 1]], [[-1, 1], [1, -1]] ]), (3, 1, 1))
    expected_time = [0, 1, 5, 6, 7, 8]
    expected_lat = [0.5, 1.5]
    expected_lon = [0.5, 1.5]
    diff_dict = compare_to_climatology(dict_list, 'absolute')
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
    diff_dict = compare_to_climatology(dict_list, 'scaled')
    assert_allclose(diff_dict['diff'], expected_diff)
    assert_allclose(diff_dict['time'], expected_time)
    assert_allclose(diff_dict['lat'], expected_lat)
    assert_allclose(diff_dict['lon'], expected_lon)

# Raised Errors
def test_different_lat():
    bad_dict_list = [SAMPLE_DICT_1, BAD_LAT_DICT]
    assert_raises(ValueError, compare_to_climatology, bad_dict_list, 'absolute')

def test_different_lon():
    bad_dict_list = [SAMPLE_DICT_1, BAD_LON_DICT]
    assert_raises(ValueError, compare_to_climatology, bad_dict_list, 'absolute')

def test_invalid_method():
    bad_method = 'null'
    dict_list = [SAMPLE_DICT_1, SAMPLE_DICT_2]
    assert_raises(ValueError, compare_to_climatology, dict_list, bad_method)

# tests for sort_and_find_coldest
def test_sort_find_coldest():
    null_distinct = {'min time separation': 0.01, 'min lat separation': 0.01, 'min lon separation': 0.01}
    cold_events = sort_and_find_coldest(SAMPLE_CLIMATOLOGY_DICT, 1, null_distinct)
    assert_allclose(cold_events.shape, (1, 5))
    assert_allclose(cold_events['temp anomaly'][0], np.nanmin(SAMPLE_TEMP_ANOMALY))
    assert_allclose((cold_events['time'][0], cold_events['lat'][0], cold_events['lon'][0]), COORDS_COLDEST_TEMP_ANOMALY)