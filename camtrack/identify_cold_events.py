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


def subset_by_timelatlon(filename, winter_idx, desired_variable_key, lat_lower_bound, lon_bounds, testing=False):
	"""
	filename: path to concatenated netCDF file
	winter_idx: index of winter under study
	desired_variable_key: string; key of variable to be subset (must be 3-D: [time, lat, lon])
	lat_lower_bound: integer or float giving lower bound of latitude; upper bound will be maximum lat value (N pole steroegraphic)
	lon_bounds: list-like of [min_lon, max_lon]
	if testing=True, uses a different bound for max_time to accomodate sample CAM4 file for nosetests
	"""
	nc_file = Dataset(filename)
	time_object = nc_file.variables['time']
	latitude_object = nc_file.variables['lat']
	longitude_object = nc_file.variables['lon']

	# time subset: define winter as Dec-Jan-Feb
	min_time = cftime.date2num(cftime.datetime(7 + winter_idx, 12, 1), time_object.units, calendar=time_object.calendar)
	if testing:
		max_time = cftime.date2num(cftime.datetime(7 + winter_idx, 12, 7), time_object.units, calendar=time_object.calendar)
	else:
		max_time = cftime.date2num(cftime.datetime(8 + winter_idx, 2, 28), time_object.units, calendar=time_object.calendar)

	# index slices for time, lat, and lon
	time_subset = slice_from_bounds(nc_file, 'time', min_time, max_time)
	lat_subset = slice_from_bounds(nc_file, 'lat', lat_lower_bound)
	if lon_bounds[0] >= lon_bounds[1]:
		raise ValueError('Longitude bounds must be in the order [lower bound, upper bound], where lower bound < upper bound; given {}'.format(lon_bounds))
	else:
		lon_subset = slice_from_bounds(nc_file, 'lon', lon_bounds[0], lon_bounds[1])

	# subset data by time, lat, and lon
	variable_object = nc_file.variables[desired_variable_key]
	if variable_object.dimensions != ('time', 'lat', 'lon'):
		raise ValueError("Variable {} has dimensions {}; expecting dimensions ['time', 'lat', 'lon']".format(desired_variable_key, variable_object.dimensions))
	else:
		return variable_object[time_subset, lat_subset, lon_subset].data


#def check_time_dimension(file, winter_idx):
#	"""
#	Check that time dimension matches up with expected start and end times of netCDF file in ordinal format
#	"""
#	time_object = file.variables['time']
#	time_list = time_object[:].data
#	expected_start = cftime.date2num(cftime.datetime(7 + winter_idx, 11, 1), time_object.units, calendar=time_object.calendar) 
#	expected_end = cftime.date2num(cftime.datetime(8 + winter_idx, 3, 1), time_object.units, calendar=time_object.calendar) 
#	tol = 1e-4  # tolerance for "is equal"
#	if (abs(time_list[0] - expected_start) > tol) or (abs(time_list[-1] - expected_end) > tol):
#		raise ValueError('Time dimension of netCDF input file does not match expected time bounds, from 00{:02.0f}-11-01 to 00{:02.0f}-03-01'.format(7 + winter_idx, 8 + winter_idx))

def slice_from_bounds(file, dimension_key, low_bound, upper_bound=np.inf):
	"""
	file: Dataset instance
	creates a slice for the interval [low_bound, upper_bound]
	raises an error if any non-infinite bounds are outside the range of the dimension
	"""
	if not isinstance(file, Dataset):
		raise TypeError('File argument must be an instance of the netCDF4 Dataset class; given type {}'.format(type(file)))
	else:
		dimension = file.variables[dimension_key][:].data

	increasing = np.all(np.diff(dimension) > 0) # True if dimension is monotonically increasing
	if not increasing:
		raise ValueError("NetCDF dimension '{}' must be monotonically increasing to produce valid index slices.".format(dimension_key))
	if not np.isinf(low_bound) and not ((low_bound >= dimension[0]) and (low_bound <= dimension[-1])):
		raise ValueError("Lower bound on index slice ({:.2f}) should be within the range of dimension '{}', which runs from {:.4f} to {:.4f} ".format(low_bound, dimension_key, dimension[0], dimension[-1]))
	if not np.isinf(upper_bound) and not ((upper_bound >= dimension[0]) and (upper_bound <= dimension[-1])):
		raise ValueError("Upper bound on index slice ({:.2f}) should be within the range of dimension '{}', which runs from {:.4f} to {:.4f} ".format(upper_bound, dimension_key, dimension[0], dimension[-1]))
	slice_idx_list = np.squeeze(np.where(np.logical_and(dimension >= low_bound, dimension <= upper_bound)))
	return slice(slice_idx_list[0], slice_idx_list[-1]+1)


