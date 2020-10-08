#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kara Hartig

Unit tests for parceltrack.py
"""
# Standard imports
import numpy as np
import os

# Testing imports
from numpy.testing import assert_raises, assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

# camtrack imports
from camtrack.parceltrack import ClimateAlongTrajectory
from camtrack.data import TrajectoryFile, WinterCAM

#########################################
##  Sample Trajectory for CAM Details  ##
#########################################
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_TRAJ = TrajectoryFile(os.path.join(TEST_DIR, 'sample_traj.traj'))
TRAJ_NUMBER = 1

#####################################
##        Sample CAM Details       ##
#####################################
SAMPLE_CAM = WinterCAM(os.path.join(TEST_DIR, 'sample_CAM4.nc'))
VARIABLES_2D = ['PS']
VARIABLES_3D = ['U']
VARIABLES_3Dto1D = ['U_1D']
VARIABLES_HARDCODE = ['THETA_hc'] # also LWP_hc, but SAMPLE_CAM is missing required variable 'Q'
CAM_OUTPUT_CADENCE = 3 # hours

#####################################
## ClimateAlongTrajectory Details  ##
#####################################
# Inputs
VALID_INTERP_METHODS = ['linear', 'nearest']
ASCENDING_PRESSURES = np.linspace(1e4, 1.1e5, 30)  # 1,100 to 100 hPa
DESCENDING_PRESSURES = np.linspace(1.1e5, 1e4, 30)  # 1,100 to 100 hPa

# Sample ClimateAlongTrajectory instances
CAT_2D_NEAREST = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_2D, 'nearest')
CAT_3D_NEAREST_ASC = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_3D, 'nearest', ASCENDING_PRESSURES)

# Outputs
#    correspond to traj 1, age = -3 and age = 0 of SAMPLE_TRAJ
#    unless specified, assume ASCENDING_PRESSURES
PSURF_NEAREST = np.array([98434.875, 98783.21])
PSURF_LINEAR = np.array([98447.28329206, 98692.00421007])
UWIND_NEAREST_ASC = np.array([
    [9.25380865, 9.22444147, 9.10755424, 8.9660217,  8.71305806, 8.36038698, 8.06076424, 7.53487811, 7.1987233,  7.2536702,  7.4353998,  7.74787241, 7.99421444, 8.09324479, 8.19227515, 8.46202295, 8.79105296, 9.12008297, 9.43044186, 9.74076036, 9.88033614, 9.77052806, 9.63172328, 9.3166928,  9.23678227, 8.52498717, np.nan, np.nan, np.nan, np.nan], 
    [9.05710137, 9.20520922, 9.10250115, 8.69541636, 8.12081756, 7.51344471, 6.96502016, 6.42076322, 6.05987739, 6.09564285, 6.29882855, 6.70392884, 7.05974953, 7.27705818, 7.49436682, 7.80537604, 8.15948343, 8.51359082, 8.85979815, 9.20541062, 9.37381564, 9.18008749, 8.96230104, 8.32897139, 7.80192629, 7.20642263, np.nan, np.nan, np.nan, np.nan]
    ])
UWIND_NEAREST_DES = np.flip(UWIND_NEAREST_ASC, axis=1)
UWIND_LINEAR_ASC = np.array([
    [9.20695813, 9.19247124, 9.07204931, 8.92273465, 8.66395395, 8.30904572, 8.00633651, 7.49036272, 7.16256918, 7.22323298, 7.40569543, 7.71425062, 7.9579541,  8.05600083, 8.15404756, 8.42499919, 8.75670567, 9.08841694, 9.39859206, 9.70859021, 9.84652626, 9.72989902, 9.58338368, 9.2486444,  9.14007495, 8.42203269, np.nan, np.nan, np.nan, np.nan],
    [8.71545702, 9.09818587, 9.20359496, 9.20831293, 9.07600804, 8.77140672, 8.39166261, 7.85658459, 7.42912571, 7.26357226, 7.29102819, 7.54075826, 7.76716319, 7.92593826, 8.08471333, 8.29148978, 8.52070261, 8.74991545, 9.00148329, 9.25357613, 9.35508924, 9.16801523, 8.95278062, 8.38174335, 7.93366477, 7.13464004, np.nan, np.nan, np.nan, np.nan]
    ])
UWIND_LINEAR_DES = np.flip(UWIND_LINEAR_ASC, axis=1)
UWIND_NEAREST_3DTO1D = np.array([8.52498717, 7.20642263])
UWIND_LINEAR_3DTO1D = np.array([np.nan, np.nan]) # traj path is between two pressure levels: higher level ('nearest', 9.62e4) is filled, lower (9.97e4) is NaN
THETA_NEAREST = np.array([254.55040291, 253.96924332])
THETA_LINEAR = np.array([np.nan, np.nan]) # traj path is between two pressure levels: higher level ('nearest', 9.62e4) is filled, lower (9.97e4) is NaN

#####################################
##  TESTS: ClimateAlongTrajectory  ##
#####################################
# Raised errors: general
def test_bad_CAM_variable_dims():
    bad_variable_dim = VARIABLES_2D + ['P0']
    for good_method in VALID_INTERP_METHODS:
        assert_raises(ValueError, ClimateAlongTrajectory, SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, bad_variable_dim, good_method)

def test_bad_method():
    bad_method = 'null'
    assert_raises(ValueError, ClimateAlongTrajectory, SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_2D, bad_method)

def test_no_plevels_on_init():
    for good_method in VALID_INTERP_METHODS:
        assert_raises(NameError, ClimateAlongTrajectory, SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_3D, good_method)

def test_no_plevels_on_add():
    assert_raises(NameError, CAT_2D_NEAREST.add_variable, VARIABLES_3D[0])

def test_no_plevels_on_setup_pinterp():
    assert_raises(NameError, CAT_2D_NEAREST.setup_pinterp, None)

def test_invalid_variable_prefix():
    invalid_prefix = 'NULL_1D'
    assert_raises(ValueError, CAT_2D_NEAREST.add_variable, invalid_prefix)

def test_invalid_variable_suffix():
    invalid_suffix = 'OMEGA_null'
    assert_raises(ValueError, CAT_2D_NEAREST.add_variable, invalid_suffix)

# Raised errors: variable exists in CAM files
def test_missing_CAM_variable_name():
    bad_variable_name = 'NULL'
    assert_raises(ValueError, CAT_2D_NEAREST.check_variable_exists, bad_variable_name)

def test_missing_CAM_variable_to1D():
    missing_variable = 'CLOUD_1D'
    assert_raises(ValueError, CAT_2D_NEAREST.check_variable_exists, missing_variable)

def test_missing_CAM_variable_hardcode():
    missing_variable = 'LWP_hc'
    assert_raises(ValueError, CAT_2D_NEAREST.check_variable_exists, missing_variable)


# Value check: 2-D variables
def test_2Dvar_values_nearest():
    traj_interp_method = 'nearest'
    psurf_along_traj = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_2D, traj_interp_method).data['PS']
    assert_allclose(psurf_along_traj.values, PSURF_NEAREST)

def test_2Dvar_values_linear():
    traj_interp_method = 'linear'
    psurf_along_traj = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_2D, traj_interp_method).data['PS']
    assert_allclose(psurf_along_traj.values, PSURF_LINEAR)


# Value check: 3-D variables
def test_3Dvar_values_nearest_ascending():
    traj_interp_method = 'nearest'
    uwind_along_traj = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_3D, traj_interp_method, ASCENDING_PRESSURES).data['U']
    assert_allclose(uwind_along_traj.values, UWIND_NEAREST_ASC)

def test_3Dvar_values_nearest_descending():
    traj_interp_method = 'nearest'
    uwind_along_traj = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_3D, traj_interp_method, DESCENDING_PRESSURES).data['U']
    assert_allclose(uwind_along_traj.values, UWIND_NEAREST_DES)

def test_3Dvar_values_linear_ascending():
    traj_interp_method = 'linear'
    uwind_along_traj = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_3D, traj_interp_method, ASCENDING_PRESSURES).data['U']
    assert_allclose(uwind_along_traj.values, UWIND_LINEAR_ASC)

def test_3Dvar_values_linear_descending():
    traj_interp_method = 'linear'
    uwind_along_traj = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_3D, traj_interp_method, DESCENDING_PRESSURES).data['U']
    assert_allclose(uwind_along_traj.values, UWIND_LINEAR_DES)


# Value check: 3-D -> 1-D variables
def test_3Dto1D_values_nearest():
    traj_interp_method = 'nearest'
    cat = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_3Dto1D, traj_interp_method, ASCENDING_PRESSURES)
    uwind_along_traj = cat.data['U_1D']
    assert_allclose(uwind_along_traj.values, UWIND_NEAREST_3DTO1D)

def test_3Dto1D_values_linear():
    traj_interp_method = 'linear'
    cat = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_3Dto1D, traj_interp_method, ASCENDING_PRESSURES)
    uwind_along_traj = cat.data['U_1D']
    assert_allclose(uwind_along_traj.values, UWIND_LINEAR_3DTO1D)


# Value check: hard-coded variables (THETA: potential temperature)
def test_THETA_values_nearest():
    traj_interp_method = 'nearest'
    cat = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_HARDCODE, traj_interp_method, ASCENDING_PRESSURES)
    theta_along_traj = cat.data['THETA']
    assert_allclose(theta_along_traj.values, THETA_NEAREST)

def test_THETA_values_linear():
    traj_interp_method = 'linear'
    cat = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, VARIABLES_HARDCODE, traj_interp_method, ASCENDING_PRESSURES)
    theta_along_traj = cat.data['THETA']
    assert_allclose(theta_along_traj.values, THETA_LINEAR)


# Check attributes
def test_traj_values():
    traj_everyXhours = SAMPLE_TRAJ.get_trajectory(TRAJ_NUMBER, CAM_OUTPUT_CADENCE)
    #assert_allclose(CAT_2D_NEAREST.traj_time.values, traj_everyXhours['cftime date'].values)
    assert_allclose(CAT_2D_NEAREST.traj_lat.values, traj_everyXhours['lat'].values)
    assert_allclose(CAT_2D_NEAREST.traj_lon.values, traj_everyXhours['lon'].values)

def test_subset_contains_traj():
    traj_everyXhours = SAMPLE_TRAJ.get_trajectory(TRAJ_NUMBER, CAM_OUTPUT_CADENCE)
    traj_time = traj_everyXhours['cftime date'].values
    traj_lat = traj_everyXhours['lat'].values
    traj_lon = traj_everyXhours['lon'].values
    all_true = np.full(len(traj_everyXhours.index), True, dtype=bool)
    assert_array_equal((traj_time >= CAT_2D_NEAREST.subset_time.start) & (traj_time <= CAT_2D_NEAREST.subset_time.stop), all_true)
    assert_array_equal((traj_lat >= CAT_2D_NEAREST.subset_lat.start) & (traj_lat <= CAT_2D_NEAREST.subset_lat.stop), all_true)
    assert_array_equal((traj_lon >= CAT_2D_NEAREST.subset_lon.start) & (traj_lon <= CAT_2D_NEAREST.subset_lon.stop), all_true)

def test_trajectory_attr():
    traj_everyXhours = SAMPLE_TRAJ.get_trajectory(TRAJ_NUMBER, CAM_OUTPUT_CADENCE)
    assert_frame_equal(CAT_2D_NEAREST.trajectory, traj_everyXhours)

