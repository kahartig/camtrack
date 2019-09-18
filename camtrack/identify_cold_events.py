"""
Author: Kara Hartig

Contains functions to:
	load a winter netCDF from CAM
	subset by time, lat, lon, and landfraction
	identify coldest events
	print corresponding CONTROL files for HYSPLIT

PROGRESS NOTES:
need to write tests for subset_by_timelatlon
	both nosetests and check functions here
	how can I check that bounds are reasonable?
check variable names and dimensions for landfrac, 2m temperature, etc


"""

# Standard Imports
import numpy as np
from netCDF4 import Dataset
import cftime
import os
import sys


def subset_by_timelatlon(filename, winter_idx, lat_bounds, lon_bounds, landfrac_min):
	"""
	filename: path to concatenated netCDF file
	winter_idx: index of winter under study
	lat_bounds: list-like of [min_lat, max_lat]
	lon_bounds: list-like of [min_lon, max_lon]
	landfrac_min: minumum landfraction threshold to consider "over land"
	"""
	nc_file = Dataset(filename)
	latitudes = nc_file.variables['lat'][:].data
	longitudes = nc_file.variables['lon'][:].data
	time_object = nc_file.variables['time']
	times = time_object[:].data

	# time subset: define winter as Dec-Jan-Feb
	min_time = cftime.date2num(cftime.datetime(7 + winter_idx, 12, 1), time_object.units, calendar=time_object.calendar) 
	max_time = cftime.date2num(cftime.datetime(8 + winter_idx, 3, 1), time_object.units, calendar=time_object.calendar) 
	time_subset = np.where(np.logical_and(times >= min_time, times < max_time))

	# latitude subset:
	min_lat = lat_bounds[0]
	max_lat = lat_bounds[1]
	lat_subset = np.where(np.logical_and(latitudes >= min_lat, latitudes <= max_lat))

	# longitude subset:
	min_lon = lon_bounds[0]
	max_lon = lon_bounds[1]
	lon_subset = np.where(np.logical_and(longitudes >= min_lon, longitudes <= max_lon))

	# subset temperature and landfraction data by time, lat, and lon
	# IS 2M TEMP TIME, LAT, LON?
	# CONFIRM THIS BOOLEAN INDEXING
	temp_2m_subset = nc_file.variables['TREFHT'][time_subset, lat_subset, lon_subset].data
	landfrac_subset = nc_file.variables['LANDFRAC'][time_subset, lat_subset, lon_subset].data

	# landfrac subset:
	landfrac_threshold = np.where(landfrac >= landfrac_min)

	# take temperatures over land
	#temp_2m_land = 

def check_time_dimension(file, winter_idx):
	"""
	Check that time dimension matches up with expected start and end times of netCDF file in ordinal format
	"""
	time_object = file.variables['time']
	time_list = time_object[:].data
	expected_start = cftime.date2num(cftime.datetime(7 + winter_idx, 11, 1), time_object.units, calendar=time_object.calendar) 
	expected_end = cftime.date2num(cftime.datetime(8 + winter_idx, 3, 1), time_object.units, calendar=time_object.calendar) 
	tol = 1e-4  # tolerance for "is equal"
	if (abs(time_list[0] - expected_start) > tol) or (abs(time_list[-1] - expected_end) > tol):
		raise ValueError('Time dimension of netCDF input file does not match expected time bounds, from 00{:02.0f}-11-01 to 00{:02.0f}-03-01'.format(7 + winter_idx, 8 + winter_idx))
