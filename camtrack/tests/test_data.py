#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kara Hartig

Unit tests for data.py
"""
# Standard imports
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os

# Testing imports
from numpy.testing import assert_raises, assert_equal, assert_allclose

# camtrack imports
from camtrack.data import TrajectoryFile, WinterCAM

#####################################
##   Sample Trajectory Details     ##
#####################################

# File path of sample .traj file
#    based on HYSPLIT trajectory for Jan 8, year 9 at lat 65.4, lon -72.5
#SAMPLE_TRAJ_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_traj.traj')
#SAMPLE_TRAJ = TrajectoryFile(SAMPLE_TRAJ_FILE_PATH)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_TRAJ = TrajectoryFile(os.path.join(TEST_DIR, 'sample_traj_for_CAM.traj'))

# trajectory grid header info
SAMPLE_GRIDS_NGRIDS = 1
SAMPLE_GRIDS_MODEL = 'CAM4'

# trajectory initial condition header info
SAMPLE_NTRAJ = 2
SAMPLE_DIRECTION = 'BACKWARD'
SAMPLE_TRAJ1_START_HEIGHT = 10.0  # starting height of trajectory 1 in sample file

# some data points from sample trajectory file
SAMPLE_TRAJ1_PRESSURE1 = 996.8  # 1st listed pressure for traj 1
SAMPLE_TRAJ2_HEIGHT3 = 502.2  # 3rd listed height for traj 2
SAMPLE_TRAJ2_LAT4 = 85.541  # 4th listed latitude for traj 2

# trajectory winter years
SAMPLE_TRAJ_WINTER = {'first': '08', 'first-second': '08-09', 'firstsecond': '0809'}

#####################################
##      Sample CAM4 Details        ##
#####################################
# time range: 0008-12-01 00:00:00 to 0008-12-07 00:00:00 (YYYY-MM-DD HH:MM:SS)
# latitude range: 80.52631579 to 90.0
# longitude range: 320.0 to 330.0 (on 0-360 scale)
# NC_SAMPLE_PATH = os.path.join(TEST_DIR, 'sample_CAM4_for_nosetests.nc')
# PS_SUBSET_FIRST_VALUE = 101194.05 # value of 'PS' for time=2889., lat=86.2, lon=322.5
# PS_SUBSET_LAST_VALUE = 99774.55 # value of 'PS' for time=2895., lat=90, lon=327.5
# T2M_LAST_LAND_VALUE = 240.78333 # value of 'TREFHT' for time=2895., lat=82.4, lon=330.0 (last "on land" point in TREFHT data)
# GRIDPOINTS_ON_LAND = 4  # number of gridpoints with landfraction >= 90 for a single timestep in lat=(82.4-90), lon=(322.5, 330)
# LENGTH_OF_TIMES_DIMENSION = 49


#####################################
##     TESTS: TrajectoryFile       ##
#####################################
# Trajectory headers
def test_grid_header():
    assert_equal(len(SAMPLE_TRAJ.grids.index), SAMPLE_GRIDS_NGRIDS)
    assert_equal(SAMPLE_TRAJ.grids.iloc[0]['model'], SAMPLE_GRIDS_MODEL)

def test_traj_header():
    assert_equal(len(SAMPLE_TRAJ.traj_start.index), SAMPLE_NTRAJ)
    assert_equal(SAMPLE_TRAJ.direction, SAMPLE_DIRECTION)
    assert_equal(SAMPLE_TRAJ.traj_start.iloc[0]['height'], SAMPLE_TRAJ1_START_HEIGHT)

# Trajectory data
def test_traj_data():
    assert_equal(SAMPLE_TRAJ.data_1h.loc[1].loc[0]['PRESSURE'], SAMPLE_TRAJ1_PRESSURE1)
    assert_equal(SAMPLE_TRAJ.data_1h.loc[2].loc[-2]['height (m)'], SAMPLE_TRAJ2_HEIGHT3)
    assert_equal(SAMPLE_TRAJ.data_1h.loc[2].loc[-3]['lat'], SAMPLE_TRAJ2_LAT4)

# Trajectory winter method
def test_traj_winter_method_asis():
    for key,value in SAMPLE_TRAJ_WINTER.items():
        assert_equal(SAMPLE_TRAJ.winter(out_format=key), value)

# def test_traj_winter_method_allDec():
#     TEMP_TRAJ = TrajectoryFile(SAMPLE_TRAJ_FILE_PATH)
#     TEMP_TRAJ.data.replace({'year': 9}, 10, inplace=True) # trajectory now in Jan '10
#     TEMP_TRAJ.data.replace({'month': 1}, 12, inplace=True) # trajectory now in Dec '10
#     EXPECTED_WINTER = {'first': '10', 'first-second': '10-11', 'firstsecond': '1011'}
#     for key,value in EXPECTED_WINTER.items():
#         assert_equal(TEMP_TRAJ.winter(out_format=key), value)

# def test_traj_winter_method_DectoJan():    
#     TEMP_TRAJ = TrajectoryFile(SAMPLE_TRAJ_FILE_PATH)
#     TEMP_TRAJ.data.replace({'month': 1}, 12, inplace=True) # trajectory now in Dec of '09
#     # add row with a Jan '10 entry:
#     Jan10_row = pd.Series({'traj #': 1, 'grid #': 1, 'year': 10, 'month': 1,
#                            'day': 1, 'hour': 1, 'minute': 0, 'fhour': 99,
#                            'traj age': -6.0, 'lat': 65.0, 'lon': -72.5,
#                            'height (m)': 10.0, 'PRESSURE': 900.0})
#     TEMP_TRAJ.data.append(Jan10_row, ignore_index=True)
#     EXPECTED_WINTER = {'first': '09', 'first-second': '09-10', 'firstsecond': '0910'}
#     for key,value in EXPECTED_WINTER.items():
#         assert_equal(TEMP_TRAJ.winter(out_format=key), value)

def test_traj_winter_badformat():
    bad_format = 'null'
    assert_raises(ValueError, SAMPLE_TRAJ.winter, bad_format)


# #####################################
# ##       TESTS: subset_nc          ##
# #####################################
# def test_subset_corners():
#     var_key = 'PS'
#     lat_bounds = (85, 90) # should pull (86.2, 88.1, 90)
#     lon_bounds = (322, 328) # should pull (322.5, 325 , 327.5)
#     output = subset_nc(NC_SAMPLE_PATH, WINTER_IDX, var_key, lat_bounds, lon_bounds, testing=True)
#     assert_allclose(output['unmasked_data'][0,0,0], PS_SUBSET_FIRST_VALUE)
#     assert_allclose(output['unmasked_data'][-1,-1,-1], PS_SUBSET_LAST_VALUE)

# def test_subset_dimension_slices():
#     var_key = 'PS'
#     lat_bounds = (84, 90)
#     lon_bounds = (323, 328)
#     expected_lats = np.array([84.31578947, 86.21052632, 88.10526316, 90])
#     expected_lons = np.array([325. , 327.5])
#     output = subset_nc(NC_SAMPLE_PATH, WINTER_IDX, var_key, lat_bounds, lon_bounds, testing=True)
#     assert_allclose(output['lat'], expected_lats)
#     assert_allclose(output['lon'], expected_lons)
#     assert_allclose(output['unmasked_data'].shape, [len(output['time']), len(output['lat']), len(output['lon'])])

# def test_landfrac_masking():
#     var_key = 'TREFHT'
#     lat_bounds = (81, 90)
#     lon_bounds = (321, 330)
#     output = subset_nc(NC_SAMPLE_PATH, WINTER_IDX, var_key, lat_bounds, lon_bounds, testing=True)
#     assert_equal(np.sum(~np.isnan(output['data'])), GRIDPOINTS_ON_LAND*LENGTH_OF_TIMES_DIMENSION)
#     last_unmasked_value = output['data'][np.where(~np.isnan(output['data']))][-1]
#     assert_allclose(last_unmasked_value, T2M_LAST_LAND_VALUE)

