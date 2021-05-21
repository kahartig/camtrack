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
from camtrack.data import TrajectoryFile, WinterCAM, subset_and_mask

#####################################
##   Sample Trajectory Details     ##
#####################################
# Inputs
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_TRAJ_PATH = os.path.join(TEST_DIR, 'sample_traj.traj')
SAMPLE_TRAJ = TrajectoryFile(SAMPLE_TRAJ_PATH)

# Outputs
#  trajectory grid header info
TRAJ_GRIDS_NGRIDS = 1
TRAJ_GRIDS_MODEL = 'CAM4'
#  trajectory initial condition header info
NTRAJ = 2
TRAJ_DIRECTION = 'BACKWARD'
TRAJ1_START_HEIGHT = 100.0  # starting height of trajectory 1 in sample file
DIAG_VARS = ['PRESSURE', 'AIR_TEMP', 'TERR_MSL']
#  some data points from sample trajectory file
TRAJ1_PRESSURE1 = 97280.  # 1st listed pressure for traj 1, in Pa
TRAJ2_HEIGHT3 = 513.9  # 3rd listed height for traj 2
TRAJ2_LAT4 = 84.504 # 4th listed latitude for traj 2
TRAJ1_2HOURLY_PRESSURES = np.array([95890., 96840., 97150.]) # in Pa
TRAJ1_2AGE_PRESSURES = np.array([96490., 97090., 97280.]) # in Pa
#  trajectory winter years
TRAJ_WINTER_DICT = {'first': '08', 'first-second': '08-09', 'firstsecond': '0809'}

#####################################
##      Sample CAM4 Details        ##
#####################################
# Inputs
# time range: 0008-12-01 00:00:00 to 0008-12-07 00:00:00 (YYYY-MM-DD HH:MM:SS)
# latitude range: 80.52631579 to 90.0
# longitude range: 320.0 to 345.0 (on 0-360 scale)
SAMPLE_CAM_PATH = os.path.join(TEST_DIR, 'sample_CAM4.nc')
SAMPLE_CAM = WinterCAM(SAMPLE_CAM_PATH)
ABOVE_EXTRAP_THRESHOLD = 1e30 # fill value used by vinth2p when extrapolation is turned off
VARS_WITH_SURFACE = ['T'] # variables with a surface-level added before interpolation
TIME_SUBSET = ('0008-12-01T00:00:00', '0008-12-03T21:00:00')
#  exact bounds equal specific lat/lon coord values
LAT_SUBSET_EXACT = (84.31578947, 90.)
LON_SUBSET_EXACT = (322.5, 327.5)
#  approx bounds are between lat/lon coord values
LAT_SUBSET_APPROX = (84., 90.)
LON_SUBSET_APPROX = (322., 327.)
#  less than minimum TREFHT value, so masking will not eliminate any values
TREFHT_THRESH = 200. # preserves all lat/lon points
LANDFRAC_THRESH = 0.05 # preserves a single lat/lon point in exact subset

# Outputs
#  time units
CAM_TIME_UNITS = 'days since 0001-01-01 00:00:00'
CAM_TIME_CAL = 'noleap'
#  interpolation onto pressure levels
ASCENDING_PRESSURES = np.linspace(1e4, 1.1e5, 30)  # 1,100 to 100 hPa
DESCENDING_PRESSURES = np.linspace(1.1e5, 1e4, 30)  # 1,100 to 100 hPa
#     after interp U and T+TREFHT onto (ascending) pressure levels,
#     this pressure level has fewer NaN values for T than U because
#     of addition of surface-level data:
PIDX_FEWER_NAN_WITH_SURFACE_DATA = -4
#  data values
PSURF_FIRST_VALUE = 85657.61 # no subsetting
PS_EXACT_FIRST_VALUE = 100528.586 # time=0008-12-01 00:00:00, lat=84.32, lon=322.5
PS_EXACT_LAST_VALUE = 99947.15 # time=0008-12-03 21:00:00, lat=90., lon=327.5
PS_APPROX_FIRST_VALUE = 100528.586 # time=0008-12-01 00:00:00, lat=84.32, lon=322.5
PS_APPROX_LAST_VALUE = 99947.15 # time=0008-12-03 21:00:00, lat=90., lon=325.
#  subsetting and masking
CAM_TIME_SUBSET_LENGTH = 24


#####################################
##     TESTS: TrajectoryFile       ##
#####################################
# Trajectory headers
def test_grid_header():
    assert_equal(len(SAMPLE_TRAJ.grids.index), TRAJ_GRIDS_NGRIDS)
    assert_equal(SAMPLE_TRAJ.grids.iloc[0]['model'], TRAJ_GRIDS_MODEL)

def test_traj_header():
    assert_equal(SAMPLE_TRAJ.ntraj, NTRAJ)
    assert_equal(SAMPLE_TRAJ.direction, TRAJ_DIRECTION)
    assert_equal(SAMPLE_TRAJ.traj_start.iloc[0]['height'], TRAJ1_START_HEIGHT)
    assert_equal(SAMPLE_TRAJ.diag_var_names, DIAG_VARS)

# Trajectory data
def test_traj_data():
    assert_allclose(SAMPLE_TRAJ.data_1h.loc[1].loc[0]['PRESSURE'], TRAJ1_PRESSURE1, atol=1)
    assert_allclose(SAMPLE_TRAJ.data_1h.loc[2].loc[-2]['height (m)'], TRAJ2_HEIGHT3, atol=0.1)
    assert_allclose(SAMPLE_TRAJ.data_1h.loc[2].loc[-3]['lat'], TRAJ2_LAT4, atol=0.001)

# Get trajectory method
def test_traj_get_both():
    good_hourly = 2
    good_age = 5
    assert_raises(ValueError, SAMPLE_TRAJ.get_trajectory, 1, good_hourly, good_age)

def test_traj_get_neither():
    assert_raises(ValueError, SAMPLE_TRAJ.get_trajectory, 1)

def test_traj_get_hourly():
    good_hourly = 2
    pressures = SAMPLE_TRAJ.get_trajectory(1, hourly_interval=good_hourly)['PRESSURE'].values
    assert_allclose(pressures, TRAJ1_2HOURLY_PRESSURES)

def test_traj_get_age():
    good_age = 2
    pressures = SAMPLE_TRAJ.get_trajectory(1, age_interval=good_age)['PRESSURE'].values
    assert_allclose(pressures, TRAJ1_2AGE_PRESSURES)

# Winter method
def test_traj_winter_method_asis():
    for key,value in TRAJ_WINTER_DICT.items():
        assert_equal(SAMPLE_TRAJ.winter(out_format=key), value)

def test_traj_winter_badformat():
    bad_format = 'null'
    assert_raises(ValueError, SAMPLE_TRAJ.winter, bad_format)


#####################################
##       TESTS: WinterCAM          ##
#####################################
# WinterCAM init
def test_CAM_init_values():
    psurf_first = SAMPLE_CAM.dataset['PS'][0, 0, 0]
    assert_allclose(psurf_first.values, PSURF_FIRST_VALUE)

def test_CAM_time_attributes():
    assert_equal(SAMPLE_CAM.time_units, CAM_TIME_UNITS)
    assert_equal(SAMPLE_CAM.time_calendar, CAM_TIME_CAL)

# Variable method
def test_CAM_retrieve_variable():
    psurf_direct = SAMPLE_CAM.dataset['PS']
    psurf_with_method = SAMPLE_CAM.variable('PS')
    assert_allclose(psurf_direct.values, psurf_with_method.values)

# Interpolate onto pressure levels
def test_pinterp_missing_vars():
    CAM_copy = WinterCAM(SAMPLE_CAM_PATH)
    CAM_copy.dataset = CAM_copy.dataset.drop('PS')
    good_da = CAM_copy.variable('T')
    assert_raises(KeyError, CAM_copy.interpolate, good_da, ASCENDING_PRESSURES)

def test_pinterp_wrong_dims():
    bad_da = SAMPLE_CAM.variable('LANDFRAC')
    assert_raises(ValueError, SAMPLE_CAM.interpolate, bad_da, ASCENDING_PRESSURES)

def test_pinterp_bad_method():
    bad_interp_method = 'log-linear'
    good_da = SAMPLE_CAM.variable('U')
    assert_raises(ValueError, SAMPLE_CAM.interpolate, good_da, ASCENDING_PRESSURES, interpolation=bad_interp_method)

def test_pinterp_exceed_noextrap_threshold():
    extrap = False
    uwind = SAMPLE_CAM.variable('U')
    uwind_large_vals = uwind.where(uwind.lat > 89, ABOVE_EXTRAP_THRESHOLD)
    assert_raises(ValueError, SAMPLE_CAM.interpolate, uwind_large_vals, ASCENDING_PRESSURES, extrapolate=extrap)

def test_pinterp_asc_matches_desc():
    # ascending pressure gives same results (with pres array flipped) as descending
    uwind = SAMPLE_CAM.variable('U').isel(time=slice(None,2))
    interpolated_asc = SAMPLE_CAM.interpolate(uwind, ASCENDING_PRESSURES).values
    interpolated_des = SAMPLE_CAM.interpolate(uwind, DESCENDING_PRESSURES).values
    flipped_interpolated_des = np.flip(interpolated_des, axis=1)
    assert_allclose(interpolated_asc, flipped_interpolated_des)

def test_pinterp_vars_with_surf_more_notNaN():
    # addition of surface-level values to variables like T means that there should always
    # be same number or more non-NaN values on any given pressure level than there are
    # for other variables
    subset = SAMPLE_CAM.dataset.isel(time=slice(None,1))
    uwind = subset['U']
    u_interpolated = SAMPLE_CAM.interpolate(uwind, ASCENDING_PRESSURES).values
    u_num_nan = np.sum(np.isnan(u_interpolated[0, PIDX_FEWER_NAN_WITH_SURFACE_DATA, :, :]))
    for key in VARS_WITH_SURFACE:
        var_interpolated = SAMPLE_CAM.interpolate(subset[key], ASCENDING_PRESSURES).values
        var_num_nan = np.sum(np.isnan(var_interpolated[0, PIDX_FEWER_NAN_WITH_SURFACE_DATA, :, :]))
        diff_in_NaN = u_num_nan > var_num_nan
        assert_equal(diff_in_NaN, True)

# interp with rolled lon


#####################################
##    TESTS: subset_and_mask       ##
#####################################
def test_subset_exact_corners():
    var_key = 'PS'
    mask_key = 'TREFHT'
    psurf = subset_and_mask(SAMPLE_CAM, var_key, TIME_SUBSET, LAT_SUBSET_EXACT, LON_SUBSET_EXACT, mask_key, TREFHT_THRESH)
    assert_allclose(psurf[0,0,0], PS_EXACT_FIRST_VALUE)
    assert_allclose(psurf[-1,-1,-1], PS_EXACT_LAST_VALUE)

def test_subset_approx_corners():
    # bounds that are inbetween lat/lon coords should be strictly inclusive
    #  subset min coord is strictly greater than (or equal to) lower bound
    #  subset max coord is strictly less than (or equal to) upper bound
    var_key = 'PS'
    mask_key = 'TREFHT'
    psurf = subset_and_mask(SAMPLE_CAM, var_key, TIME_SUBSET, LAT_SUBSET_APPROX, LON_SUBSET_APPROX, mask_key, TREFHT_THRESH)
    assert_allclose(psurf[0,0,0], PS_APPROX_FIRST_VALUE)
    assert_allclose(psurf[-1,-1,-1], PS_APPROX_LAST_VALUE)

def test_masking_threshold():
    var_key = 'TREFHT'
    mask_key = 'LANDFRAC'
    masked_var = subset_and_mask(SAMPLE_CAM, var_key, TIME_SUBSET, LAT_SUBSET_EXACT, LON_SUBSET_EXACT, mask_key, LANDFRAC_THRESH)
    assert_equal(np.sum(~np.isnan(masked_var.values)), CAM_TIME_SUBSET_LENGTH)

def test_masking_values():
    var_key = 'TREFHT'
    mask_key = 'TREFHT'
    mask_thresh = 238.
    masked_var = subset_and_mask(SAMPLE_CAM, var_key, TIME_SUBSET, LAT_SUBSET_EXACT, LON_SUBSET_EXACT, mask_key, mask_thresh)
    # to compare: subset TREFHT w/o masking, use np.where to replace values below threshold, then compare arrays
    subset_only = subset_and_mask(SAMPLE_CAM, var_key, TIME_SUBSET, LAT_SUBSET_EXACT, LON_SUBSET_EXACT, mask_key, TREFHT_THRESH)
    masked_by_hand = np.where(subset_only > mask_thresh, subset_only, np.nan)
    assert_allclose(masked_var.values, masked_by_hand)

