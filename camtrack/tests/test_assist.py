#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kara Hartig

Unit tests for assist.py
"""
# Standard imports
import numpy as np
import os

# Testing imports
from numpy.testing import assert_allclose#, assert_raises, assert_array_equal

# camtrack imports
from camtrack.assist import roll_longitude
from camtrack.data import WinterCAM

#####################################
##        Sample CAM Details       ##
#####################################
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CAM_ROLL = WinterCAM(os.path.join(TEST_DIR, 'sample_CAM4_for_roll_lon.nc'))
VARIABLES_2D = ['PS']
VARIABLES_3D = ['U']

#####################################
##      TESTS: roll_longitude      ##
#####################################
def test_2D_lon0_equals_lon360():
    variable = SAMPLE_CAM_ROLL.variable(VARIABLES_2D[0])
    rolled_variable = roll_longitude(variable)
    assert_allclose(rolled_variable.sel(lon=0).values, rolled_variable.sel(lon=360).values)

def test_3D_lon0_equals_lon360():
    variable = SAMPLE_CAM_ROLL.variable(VARIABLES_3D[0])
    rolled_variable = roll_longitude(variable)
    assert_allclose(rolled_variable.sel(lon=0).values, rolled_variable.sel(lon=360).values)