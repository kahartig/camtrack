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
from numpy.testing import assert_raises, assert_equal, assert_allclose

# camtrack imports
from camtrack.parceltrack import ClimateAlongTrajectory
from camtrack.data import TrajectoryFile, WinterCAM

#########################################
##  Sample Trajectory for CAM Details  ##
#########################################
SAMPLE_TRAJ_FOR_CAM = TrajectoryFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_traj_for_CAM.traj'))

#####################################
##        Sample CAM Details       ##
#####################################
SAMPLE_CAM = WinterCAM('NULL', SAMPLE_TRAJ_FOR_CAM, nosetest=True)
NUM_LEVELS = 26


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