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

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from math import radians, cos, sin, asin, sqrt
from math import atan2


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

def set_fontsize(titles=24, labels=20, legend=20, other=16):
    '''
    Set matplotlib default fontsizes with plt.rc()

    Parameters
    ----------
    titles: integer
        font size applied to axes titles and figure titles
        Default is 24
    labels: integer
        font size applied to x- and y-axis labels
        Defult is 20
    legend: integer
        font size applied to legend
        Defult is 20
    other: integer
        font size applied to x- and y-ticks and all other text
        Default is 16
    '''
    plt.rc('font', size=other)           # controls default text sizes
    plt.rc('axes', titlesize=titles)     # fontsize of the axes title
    plt.rc('axes', labelsize=labels)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=other)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=other)    # fontsize of the tick labels
    plt.rc('legend', fontsize=legend)    # legend fontsize
    plt.rc('figure', titlesize=titles)   # fontsize of the figure title