"""
Author: Kara Hartig

Utility functions to assist in short operations throughout camtrack

Functions:
    roll_longitude: duplicate lon=0 values as lon=360
"""

# Standard imports
import numpy as np
import xarray as xr
import pandas as pd
import os
import cftime
import calendar


def roll_longitude(variable):
    '''
    DOC
    '''
    lon_0 = variable.sel(lon=0.)
    lon_360 = lon_0.assign_coords(lon=360.)
    rolled_variable = xr.concat([variable, lon_360], dim='lon')
    return rolled_variable