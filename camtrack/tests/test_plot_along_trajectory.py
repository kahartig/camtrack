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
from camtrack.plot_along_trajectory import TrajectoryFile

# File path of sample .traj file
#    based on HYSPLIT trajectory for Jan 8, year 9 at lat 65.4, lon -72.5
SAMPLE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_traj.traj')
SAMPLE_TRAJ = TrajectoryFile(SAMPLE_FILE_PATH)
SAMPLE_GRIDS_NGRIDS = 1
SAMPLE_GRIDS_MODEL = 'CAM4'
SAMPLE_NTRAJ = 2
SAMPLE_TRAJ1_START_HEIGHT = 10.0  # starting longitude of trajectory 1 in sample file
#SAMPLE_DIAG_NAMES = ['PRESSURE']
#SAMPLE_DIRECTION = 'BACKWARD'

# Some data points from sample trajectory file
SAMPLE_TRAJ1_PRESSURE1 = 996.8  # 1st listed pressure for traj 1
SAMPLE_TRAJ2_HEIGHT3 = 502.2  # 3rd listed height for traj 2
SAMPLE_TRAJ2_LAT4 = 64.533  # 4th listed latitude for traj 2

# Check reading of headers
def test_grid_header():
    assert_equal(len(SAMPLE_TRAJ.grids.index), SAMPLE_GRIDS_NGRIDS)
    assert_equal(SAMPLE_TRAJ.grids.iloc[0]['model'], SAMPLE_GRIDS_MODEL)

def test_traj_start():
    assert_equal(SAMPLE_TRAJ.traj_start.iloc[0]['height'], SAMPLE_TRAJ1_START_HEIGHT)
    assert_equal(len(SAMPLE_TRAJ.traj_start.index), SAMPLE_NTRAJ)

def test_traj_data():
    assert_equal(SAMPLE_TRAJ.data.loc[1, 0]['PRESSURE'], SAMPLE_TRAJ1_PRESSURE1)
    assert_equal(SAMPLE_TRAJ.data.loc[2, -2]['height (m)'], SAMPLE_TRAJ2_HEIGHT3)
    assert_equal(SAMPLE_TRAJ.data.loc[2, -3]['lat'], SAMPLE_TRAJ2_LAT4)
    