#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kara Hartig

Unit tests for climatology.py
"""
# Standard imports
import numpy as np
import xarray as xr
import cftime
import os

# Testing imports
from numpy.testing import assert_raises, assert_allclose

# camtrack imports
from camtrack.climatology import anomaly_DJF, sample_coldtail, find_coldest


#####################################
##    Anomaly from DJF Details     ##
#####################################
# Inputs
ANOMALY_VALUES_1 = np.arange(0, 8).reshape((2, 2, 2)).astype(float)
ANOMALY_VALUES_1[0, 1, 1] = np.nan
ANOMALY_VALUES_1[1, 1, 1] = np.nan
ANOMALY_VALUES_2 = 2*ANOMALY_VALUES_1
ANOMALY_TIME_1 = [0, 1.5]
ANOMALY_TIME_2 = [10, 11.5]
ANOMALY_LAT = [2, 4]
ANOMALY_LON = [0, 2.5]
ANOMALY_DATA_1 = xr.DataArray(ANOMALY_VALUES_1, [ ('time', ANOMALY_TIME_1), ('lat', ANOMALY_LAT), ('lon', ANOMALY_LON) ])
ANOMALY_DATA_2 = xr.DataArray(ANOMALY_VALUES_2, [ ('time', ANOMALY_TIME_2), ('lat', ANOMALY_LAT), ('lon', ANOMALY_LON) ])

# Outputs
ANOMALY_MEAN = np.array([[3., 4.5], [6., np.nan]])
ANOMALY_DIFF_ABSOLUTE = np.array([ [[-3., -3.5],[-4., np.nan]], [[1., 0.5],[0., np.nan]], [[-3., -2.5],[-2., np.nan]], [[5., 5.5],[6., np.nan]]])
ANOMALY_STD = np.std(np.concatenate((ANOMALY_VALUES_1, ANOMALY_VALUES_2), axis=0), axis=0)
ANOMALY_DIFF_SCALED = ANOMALY_DIFF_ABSOLUTE/ANOMALY_STD

#####################################
##    Sample Coldtail Details      ##
#####################################
# Inputs
TIME_UNITS = 'days since 0001-01-01 00:00:00'
TIME_CALENDAR = 'noleap'
SAMPLE_TIME = cftime.num2date(ANOMALY_TIME_1 + ANOMALY_TIME_2, TIME_UNITS, TIME_CALENDAR)
SAMPLE_INPUT = {'diff': xr.DataArray(ANOMALY_DIFF_ABSOLUTE, [ ('time', SAMPLE_TIME), ('lat', ANOMALY_LAT), ('lon', ANOMALY_LON) ])}
TOTAL_EVENTS = np.prod(ANOMALY_DIFF_ABSOLUTE.shape)
SAMPLE_PERCENTILE_LOW = (0, 25)
SAMPLE_PERCENTILE_MID = (100./3, 200./3)
SAMPLE_PERCENTILE_HIGH = (75, 100)
SAMPLE_PERCENTILE_INBETWEEN = (30, 90) # percentiles that do not divide SAMPLE_INPUT cleanly
SAMPLE_SEED = 10

# Outputs
SAMPLE_LOW_VALUES = np.array([-4., -3.5, -3.])
SAMPLE_MID_VALUES = np.array([-2.5, -2., 0., 0.5])
SAMPLE_HIGH_VALUES = np.array([5., 5.5, 6.])
SAMPLE_INBETWEEN_VALUES = np.array([-2.5, -2., 0., 0.5, 1., 5.])
SAMPLED_VALUES_WITH_SEED = [-3., 0.] # sampling 2 events from SAMPLE_INPUT full range with random seed SAMPLE_SEED
SAMPLE_SINGLE_EVENT = {'time': 0, 'lat': 2, 'lon': 0, 'temp anomaly': -3.}  # location with value -3. (unraveled index 2) in SAMPLE_INPUT

#####################################
##      Find Coldest Details       ##
#####################################
# Inputs
DISTINCT_LAX = {'delta t': 0, 'delta lat': 0, 'delta lon': 0}
DISTINCT_TIME = {'delta t': 11, 'delta lat': 0.5, 'delta lon': 0.5} # strict in time, lax in space
DISTINCT_SPACE = {'delta t': 1, 'delta lat': 3, 'delta lon': 3} # lax in space, strict in time
DISTINCT_STRICT = {'delta t': 12, 'delta lat': 5, 'delta lon': 5}
FIND_NUM_EVENTS = 4

# Outputs
FIND_LAX_ANOMALIES = np.array(np.sort(ANOMALY_DIFF_ABSOLUTE, axis=None)[:FIND_NUM_EVENTS])
FIND_TIME_EXCLUDED_ANOMALIES = np.array([-4., -3.5, -3., 5.])
FIND_SPACE_EXCLUDED_ANOMALIES = np.array([-4., -3., 0., 5.])
FIND_STRICT_ANOMALIES = np.array([-4.])

#####################################
##      TESTS: anomaly_DJF         ##
#####################################

# Absolute method: value - mean
def test_absolute_method():
    input_list = [ANOMALY_DATA_1, ANOMALY_DATA_2]
    expected_time = ANOMALY_TIME_1 + ANOMALY_TIME_2
    diff_dict = anomaly_DJF(input_list, 'absolute')
    assert_allclose(diff_dict['diff'].values, ANOMALY_DIFF_ABSOLUTE)
    assert_allclose(diff_dict['diff'].time.values, expected_time)
    assert_allclose(diff_dict['mean'].values, ANOMALY_MEAN)

# Scaled method: (value - mean) / stdev
def test_scaled_method():
    input_list = [ANOMALY_DATA_1, ANOMALY_DATA_2]
    expected_time = ANOMALY_TIME_1 + ANOMALY_TIME_2
    diff_dict = anomaly_DJF(input_list, 'scaled')
    assert_allclose(diff_dict['diff'].values, ANOMALY_DIFF_SCALED)
    assert_allclose(diff_dict['diff'].time.values, expected_time)
    assert_allclose(diff_dict['mean'].values, ANOMALY_MEAN)

# Raised Errors
def test_invalid_method():
    bad_method = 'null'
    input_list = [ANOMALY_DATA_1, ANOMALY_DATA_2]
    assert_raises(ValueError, anomaly_DJF, input_list, bad_method)

# def test_different_lat():
#     bad_dict_list = [SAMPLE_DICT_1, BAD_LAT_DICT]
#     assert_raises(ValueError, anomaly_DJF, bad_dict_list, 'absolute')

# def test_different_lon():
#     bad_dict_list = [SAMPLE_DICT_1, BAD_LON_DICT]
#     assert_raises(ValueError, anomaly_DJF, bad_dict_list, 'absolute')

#####################################
##     TESTS: sample_coldtail      ##
#####################################
# Values for various percentile ranges
def test_sample_percentile_low():
    # NOTE: random seed is unecessary since we are sampling all events in the percentile range
    num_events = len(SAMPLE_LOW_VALUES)
    cold_events = sample_coldtail(SAMPLE_INPUT, num_events, SAMPLE_PERCENTILE_LOW)
    sampled_values = np.sort(cold_events['temp anomaly'].to_numpy())
    assert_allclose(sampled_values, SAMPLE_LOW_VALUES)

def test_sample_percentile_mid():
    # NOTE: random seed is unecessary since we are sampling all events in the percentile range
    num_events = len(SAMPLE_MID_VALUES)
    cold_events = sample_coldtail(SAMPLE_INPUT, num_events, SAMPLE_PERCENTILE_MID)
    sampled_values = np.sort(cold_events['temp anomaly'].to_numpy())
    assert_allclose(sampled_values, SAMPLE_MID_VALUES)

def test_sample_percentile_high():
    # NOTE: random seed is unecessary since we are sampling all events in the percentile range
    num_events = len(SAMPLE_HIGH_VALUES)
    cold_events = sample_coldtail(SAMPLE_INPUT, num_events, SAMPLE_PERCENTILE_HIGH)
    sampled_values = np.sort(cold_events['temp anomaly'].to_numpy())
    assert_allclose(sampled_values, SAMPLE_HIGH_VALUES)

def test_sample_percentile_inbetween():
    # NOTE: random seed is unecessary since we are sampling all events in the percentile range
    # the bounds of this percentile range fall between elements
    #  looking for sample_coldtail to pull from a conservative range (next above lower end, next below higher end)
    num_events = len(SAMPLE_INBETWEEN_VALUES)
    cold_events = sample_coldtail(SAMPLE_INPUT, num_events, SAMPLE_PERCENTILE_INBETWEEN)
    sampled_values = np.sort(cold_events['temp anomaly'].to_numpy())
    assert_allclose(sampled_values, SAMPLE_INBETWEEN_VALUES)

# Random seed initialization
def test_sample_seed():
    cold_events = sample_coldtail(SAMPLE_INPUT, 2, (0, 100), seed=SAMPLE_SEED)
    sampled_values = cold_events['temp anomaly'].to_numpy()
    assert_allclose(sampled_values, SAMPLED_VALUES_WITH_SEED)

# Correspondance of time, lat, lon, and value for a sample
def test_sample_location_corresponds():
    cold_events = sample_coldtail(SAMPLE_INPUT, 2, (0, 100), seed=SAMPLE_SEED)
    single_event = cold_events.loc[0]
    assert_allclose(single_event['time'], SAMPLE_SINGLE_EVENT['time'])
    assert_allclose(single_event['lat'], SAMPLE_SINGLE_EVENT['lat'])
    assert_allclose(single_event['lon'], SAMPLE_SINGLE_EVENT['lon'])
    assert_allclose(single_event['temp anomaly'], SAMPLE_SINGLE_EVENT['temp anomaly'])

#####################################
##     TESTS: find_coldest      ##
#####################################
def test_find_lax():
    cold_events = find_coldest(SAMPLE_INPUT, FIND_NUM_EVENTS, DISTINCT_LAX)
    assert_allclose(cold_events['temp anomaly'].to_numpy(), FIND_LAX_ANOMALIES)

def test_find_exclude_with_time():
    cold_events = find_coldest(SAMPLE_INPUT, FIND_NUM_EVENTS, DISTINCT_TIME)
    assert_allclose(cold_events['temp anomaly'].to_numpy(), FIND_TIME_EXCLUDED_ANOMALIES)

def test_find_exclude_with_space():
    cold_events = find_coldest(SAMPLE_INPUT, FIND_NUM_EVENTS, DISTINCT_SPACE)
    assert_allclose(cold_events['temp anomaly'].to_numpy(), FIND_SPACE_EXCLUDED_ANOMALIES)

def test_find_strict():
    cold_events = find_coldest(SAMPLE_INPUT, FIND_NUM_EVENTS, DISTINCT_STRICT)
    assert_allclose(cold_events['temp anomaly'].to_numpy(), FIND_STRICT_ANOMALIES)