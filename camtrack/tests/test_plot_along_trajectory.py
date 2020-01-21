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
from camtrack.plot_along_trajectory import DataFromTrajectoryFile

# File path of sample .traj file
#    based on HYSPLIT trajectory for Jan 8, year 9 at lat 65.4, lon -72.5
SAMPLE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_traj.traj')
SAMPLE_TRAJ = DataFromTrajectoryFile(SAMPLE_FILE_PATH)
SAMPLE_GRIDS_NGRIDS = 1
SAMPLE_GRIDS_MODEL = 'CAM4'
SAMPLE_TRAJ1_START_LON = -72.5  # starting longitude of trajectory 1 in sample file
SAMPLE_DIAG_NAMES = ['PRESSURE']
SAMPLE_NTRAJ = 2
SAMPLE_DIRECTION = 'BACKWARD'

# Check reading of headers
def test_grid_header():
    assert_equal(len(SAMPLE_TRAJ.grids.index), SAMPLE_GRIDS_NGRIDS)
    assert_equal(SAMPLE_TRAJ.grids.iloc[0]['model'], SAMPLE_GRIDS_MODEL)

def test_traj_start():
    assert_equal(SAMPLE_TRAJ.traj_start.iloc[0]['lon'], SAMPLE_TRAJ1_START_LON)
    assert_equal(len(SAMPLE_TRAJ.traj_start.index), SAMPLE_NTRAJ)

def test_misc_headers():
    assert_allclose(SAMPLE_TRAJ.diag_var_names, SAMPLE_DIAG_NAMES)
    assert_equal(SAMPLE_TRAJ.direction, SAMPLE_DIRECTION)
    