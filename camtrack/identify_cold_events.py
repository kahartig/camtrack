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

	ARE bounds inclusive or exclusive? [] VS (). Implement in call to slice_from_bounds and in slice_from_bounds function
	WHICH temperature from CAM should I use?
	WHAT is the best way to sort temperature? fully unravel index and use reshape to go back?
	"""
	nc_file = Dataset(filename)
	time_object = nc_file.variables['time']

	# time subset: define winter as Dec-Jan-Feb
	min_time = cftime.date2num(cftime.datetime(7 + winter_idx, 12, 1), time_object.units, calendar=time_object.calendar) 
	max_time = cftime.date2num(cftime.datetime(8 + winter_idx, 3, 1), time_object.units, calendar=time_object.calendar)

	# index slices for time, lat, and lon
	time_subset = slice_from_bounds(nc_file, 'time', min_time, max_time, contains_bounds=(True,True))
	lat_subset = slice_from_bounds(nc_file, 'lat', lat_bounds[0], lat_bounds[1], contains_bounds=(True,True))
	lon_subset = slice_from_bounds(nc_file, 'lon', lon_bounds[0], lon_bounds[1], contains_bounds=(True,True))

	# subset temperature and landfraction data by time, lat, and lon
	# WHICH TEMP SHOULD I USE?
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

def slice_from_bounds(file, dimension_key, low_bound, high_bound=np.inf, contains_bounds=(True, False)):
	"""
	file: Dataset instance [SHOULD I WRITE AN ISINSTANCE CHECK?]
	selects the interval [low_bound, high_bound) [HOW SHOULD I HANDLE INCLUSIVE HIGH BOUNDS, LIKE FOR LATITUDE? OPTION IN SUBSET_BY_TIMELATLON TO GIVE ONLY LOWER BOUND?]
	contains_bounds: if True for (low_bound, high_bound), check that low_bound and/or high_bound is contained in the range of dimension
	"""
	dimension = file.variables[dimension_key][:].data
	increasing = np.all(np.diff(dimension) > 0) # True if dimension is monotonically increasing
	if not increasing:
		raise ValueError("NetCDF dimension '{}' must be monotonically increasing to produce valid index slices.".format(dimension_key))
	if contains_bounds[0]:
		if not ((low_bound >= dimension[0]) and (low_bound < dimension[-1])):
			raise ValueError("Lower bound on index slice ({:.2f}) should be within the range of dimension '{}' ".format(low_bound, dimension_key))
	if contains_bounds[1]:
		if not ((high_bound > dimension[0]) and (high_bound <= dimension[-1])):
			raise ValueError("Upper bound on index slice ({:.2f}) should be within the range of dimension '{}' ".format(high_bound, dimension_key))

	slice_idx_list = np.squeeze(np.where(np.logical_and(dimension >= low_bound, dimension < high_bound)))
	return slice(slice_idx_list[0], slice_idx_list[-1]+1)


