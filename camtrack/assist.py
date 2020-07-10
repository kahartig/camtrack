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
    Duplicate the lon=0 values at the other end of the longitude dimension as
    lon=360

    Allows data to mimic periodic longitude behavior. For example, interoplation
    will recognize that lon=359.9 is between lon=0 and max(lon)
    Slow to run on large files, so consider subsetting the variable along other
    dimensions before providing to this function

    Parameters
    ----------
    variable: xarray.DataArray
        DataArray of climate variable values
        Must have a globe-spanning longitude dimension

    Returns
    -------
    rolled_variable: xarray.DataArray
        same as input variable, but with an extra set of values at the end of
        the longitude dimension (lon=360) that are a duplicate of lon=0 values
    '''
    lon_0 = variable.sel(lon=0.)
    lon_360 = lon_0.assign_coords(lon=360.)
    rolled_variable = xr.concat([variable, lon_360], dim='lon')
    return rolled_variable