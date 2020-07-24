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
SAMPLE_TRAJ = TrajectoryFile(os.path.join(TEST_DIR, 'sample_traj_for_CAM.traj'))
TRAJ_NUMBER = 1

#####################################
##        Sample CAM Details       ##
#####################################
SAMPLE_CAM = WinterCAM(os.path.join(TEST_DIR, 'sample_CAM4_for_nosetests.nc'))
VARIABLES_2D = ['PS']
VARIABLES_3D = ['U']
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
PSURF_NEAREST = np.array([97791.79, 84785.91])
PSURF_LINEAR = np.array([96472.89378072, 84859.42708281])
UWIND_NEAREST_ASC = np.array([
    [10.43023922, 10.02690618,  9.92227166,  9.75004783,  9.5248333, 9.26438415,  9.00442739,  8.45571027,  8.03413,     7.81050418, 7.90901281,  8.25621577,  8.61533068,  8.99251134,  9.36969199, 9.71098004, 10.04583621, 10.37089508, 10.62618473, 10.88147437, 11.19758911, 11.56344531, 11.90968008, 12.21037261, 12.27593719, 10.02478109, np.nan, np.nan, np.nan, np.nan], 
    [ 9.86537773,  9.44583627,  8.08912645,  6.81425586,  5.2707048, 4.82852328,  3.59853404,  3.08969499,  3.11929866,  3.74333177, 4.35414374,  4.89929581,  5.44444788,  5.83499913,  6.22358004, 6.33994146,  6.3142298,   6.87861663,  8.14521847,  9.09696402,  9.89970177, 16.26494813, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
UWIND_NEAREST_DES = np.flip(UWIND_NEAREST_ASC, axis=1)
UWIND_LINEAR_ASC = np.array([
    [10.3476441,   9.95376095,  9.65743452,  9.26977096,  8.82144278, 8.4492893,   8.16838821,  7.51185551,  7.22575491,  7.17502271, 7.54244316,  8.07365977,  8.50206856,  8.92360336,  9.31405017, 9.64395651,  9.97046127, 10.16566347, 10.29077666, 10.56213324, 11.15155597, 11.77067191, 12.3818393, 13.04354122, 12.49212827, np.nan, np.nan, np.nan, np.nan, np.nan],
    [ 9.86227242,  9.4414325 ,  8.09042234,  6.822293,    5.2846976,  4.82911624,  3.59132831,  3.06603278,  3.08925859,  3.70339255, 4.30942843,  4.85912737,  5.40852038,  5.8034791,   6.19448654, 6.32136866,  6.30715881,  6.86046303,  8.10287125,  9.03993266, 9.8172146,   16.12436708, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
UWIND_LINEAR_DES = np.flip(UWIND_LINEAR_ASC, axis=1)

#####################################
##  TESTS: ClimateAlongTrajectory  ##
#####################################
# Check for raised errors
def test_bad_CAM_variable_name():
    bad_variable_name = VARIABLES_2D + ['NULL']
    for good_method in VALID_INTERP_METHODS:
        assert_raises(ValueError, ClimateAlongTrajectory, SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, bad_variable_name, good_method)

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


# Interpolation onto pressure levels
def test_default_to_init_plevels():
    traj_interp_method = 'nearest'
    init_variable = ['V']
    added_variable = 'U'
    cat_ascending = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ, TRAJ_NUMBER, init_variable, traj_interp_method, ASCENDING_PRESSURES)
    cat_ascending.add_variable(added_variable, pressure_levels=DESCENDING_PRESSURES)
    uwind_along_traj = cat_ascending.data[added_variable]
    assert_allclose(uwind_along_traj.values, UWIND_NEAREST_ASC)

