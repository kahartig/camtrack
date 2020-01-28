#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kara Hartig

Unit tests for plot_along_trajectory.py
"""
# Standard imports
import numpy as np
import pandas as pd
import os

# Testing imports
from numpy.testing import assert_equal, assert_raises, assert_allclose

# camtrack imports
from camtrack.plot_along_trajectory import TrajectoryFile, WinterCAM, ClimateAlongTrajectory


#####################################
##   Sample Trajectory Details     ##
#####################################

# File path of sample .traj file
#    based on HYSPLIT trajectory for Jan 8, year 9 at lat 65.4, lon -72.5
SAMPLE_TRAJ_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_traj.traj')
SAMPLE_TRAJ = TrajectoryFile(SAMPLE_TRAJ_FILE_PATH)

# trajectory grid header info
SAMPLE_GRIDS_NGRIDS = 1
SAMPLE_GRIDS_MODEL = 'CAM4'

# trajectory initial condition header info
SAMPLE_NTRAJ = 2
SAMPLE_DIRECTION = 'BACKWARD'
SAMPLE_TRAJ1_START_HEIGHT = 10.0  # starting longitude of trajectory 1 in sample file

# some data points from sample trajectory file
SAMPLE_TRAJ1_PRESSURE1 = 996.8  # 1st listed pressure for traj 1
SAMPLE_TRAJ2_HEIGHT3 = 502.2  # 3rd listed height for traj 2
SAMPLE_TRAJ2_LAT4 = 64.533  # 4th listed latitude for traj 2

# trajectory winter years
SAMPLE_TRAJ_WINTER = {'first': '08', 'first-second': '08-09', 'firstsecond': '0809'}

#####################################
##        Sample CAM Details       ##
#####################################
SAMPLE_CAM_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
SAMPLE_TRAJ_FOR_CAM = TrajectoryFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_traj_for_CAM.traj'))
SAMPLE_CAM = WinterCAM(SAMPLE_CAM_FILE_PATH, SAMPLE_TRAJ_FOR_CAM, nosetest=True)
NUM_LEVELS = 26

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
    assert_equal(SAMPLE_TRAJ.data.loc[1, 0]['PRESSURE'], SAMPLE_TRAJ1_PRESSURE1)
    assert_equal(SAMPLE_TRAJ.data.loc[2, -2]['height (m)'], SAMPLE_TRAJ2_HEIGHT3)
    assert_equal(SAMPLE_TRAJ.data.loc[2, -3]['lat'], SAMPLE_TRAJ2_LAT4)

# Trajectory winter method
def test_traj_winter_method_asis():
    for key,value in SAMPLE_TRAJ_WINTER.items():
        assert_equal(SAMPLE_TRAJ.winter(out_format=key), value)

def test_traj_winter_method_allDec():
    TEMP_TRAJ = TrajectoryFile(SAMPLE_TRAJ_FILE_PATH)
    TEMP_TRAJ.data.replace({'year': 9}, 10, inplace=True) # trajectory now in Jan '10
    TEMP_TRAJ.data.replace({'month': 1}, 12, inplace=True) # trajectory now in Dec '10
    EXPECTED_WINTER = {'first': '10', 'first-second': '10-11', 'firstsecond': '1011'}
    for key,value in EXPECTED_WINTER.items():
        assert_equal(TEMP_TRAJ.winter(out_format=key), value)

def test_traj_winter_method_DectoJan():    
    TEMP_TRAJ = TrajectoryFile(SAMPLE_TRAJ_FILE_PATH)
    TEMP_TRAJ.data.replace({'month': 1}, 12, inplace=True) # trajectory now in Dec of '09
    # add row with a Jan '10 entry:
    Jan10_row = pd.Series({'traj #': 1, 'grid #': 1, 'year': 10, 'month': 1,
                           'day': 1, 'hour': 1, 'minute': 0, 'fhour': 99,
                           'traj age': -6.0, 'lat': 65.0, 'lon': -72.5,
                           'height (m)': 10.0, 'PRESSURE': 900.0})
    TEMP_TRAJ.data.append(Jan10_row, ignore_index=True)
    EXPECTED_WINTER = {'first': '09', 'first-second': '09-10', 'firstsecond': '0910'}
    for key,value in EXPECTED_WINTER.items():
        assert_equal(TEMP_TRAJ.winter(out_format=key), value)

def test_traj_winter_badformat():
    bad_format = 'null'
    assert_raises(ValueError, SAMPLE_TRAJ.winter, bad_format)

#####################################
##  TESTS: ClimateAlongTrajectory  ##
#####################################
def test_bad_CAM_variable():
    bad_plot_var = ['NULL']
    assert_raises(ValueError, ClimateAlongTrajectory, SAMPLE_CAM, SAMPLE_TRAJ_FOR_CAM, 1, bad_plot_var)

def test_every3hours_trajectory():
    expected_hours = np.array([12, 15])
    good_plot_var = ['OMEGA', 'LANDFRAC'] # these must be in sample CAM file
    trajectory_3h = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ_FOR_CAM, 1, good_plot_var).trajectory
    assert_allclose(trajectory_3h['hour'].values, expected_hours)
    
def test_2Dvar_along_trajectory():
    expected_surface_pressures = np.array([97791.79, 84785.91])
    good_plot_var = ['PS']
    psurf_along_traj_3h = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ_FOR_CAM, 1, good_plot_var).data['PS']
    assert_allclose(psurf_along_traj_3h.values, expected_surface_pressures)

def test_3Dvar_along_trajectory():
    expected_temperatures_length = NUM_LEVELS
    expected_temperatures_atfirstlevel = np.array([206.81966, 207.17694])
    expected_temperatures_atlastlevel = np.array([252.60793, 247.17357])
    good_plot_var = ['T']
    temp_along_traj_3h = ClimateAlongTrajectory(SAMPLE_CAM, SAMPLE_TRAJ_FOR_CAM, 1, good_plot_var).data['T']
    assert_equal(temp_along_traj_3h.sizes['lev'], expected_temperatures_length)
    assert_allclose(temp_along_traj_3h.isel(lev=0).values, expected_temperatures_atfirstlevel)
    assert_allclose(temp_along_traj_3h.isel(lev=NUM_LEVELS-1).values, expected_temperatures_atlastlevel)