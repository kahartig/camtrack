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


def subset_by_timelatlon(filename, winter_idx, desired_variable_key, lat_lower_bound, lon_bounds, landfrac_min=0.9, testing=False):
	"""
	filename: path to concatenated netCDF file
	winter_idx: index of winter under study
	desired_variable_key: string; key of variable to be subset (must be 3-D: [time, lat, lon])
	lat_lower_bound: integer or float giving lower bound of latitude; upper bound will be maximum lat value (N pole steroegraphic)
	lon_bounds: list-like of [min_lon, max_lon]
	landfrac_min: minimum value of landfraction categorizing a grid point as "on land" 
	if testing=True, uses a different bound for max_time to accomodate sample CAM4 file for nosetests and includes unmasked data in output
	"""
	nc_file = Dataset(filename)
	variable_object = nc_file.variables[desired_variable_key]
	if variable_object.dimensions != ('time', 'lat', 'lon'):
		raise ValueError("Variable {} has dimensions {}; expecting dimensions ('time', 'lat', 'lon')".format(desired_variable_key, variable_object.dimensions))
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
	lon_subset = slice_from_bounds(nc_file, 'lon', lon_bounds[0], lon_bounds[1])

	# subset data by time, lat, and lon
	variable_subset = variable_object[time_subset, lat_subset, lon_subset].data

	# mask by landfraction
	#   replace any value in variable_subset where landfraction < landfrac_min with np.nan
	landfrac_subset = nc_file.variables['LANDFRAC'][time_subset, lat_subset, lon_subset].data
	masked_variable = np.where(landfrac_subset >= landfrac_min, variable_subset, np.nan)
	subset_dict = {'data': masked_variable, 'time': time_object[time_subset].data, 'lat': latitude_object[lat_subset].data, 'lon': longitude_object[lon_subset].data}
	if testing:
		subset_dict['unmasked_data'] = variable_subset
	return subset_dict


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

	# dimension is monotonically increasing:
	increasing = np.all(np.diff(dimension) > 0) # True if dimension is monotonically increasing
	if not increasing:
		raise ValueError("NetCDF dimension '{}' must be monotonically increasing to produce valid index slices.".format(dimension_key))

	# low_bound < upper_bound:
	if not (low_bound < upper_bound):
		raise ValueError("Dimension slicing by index error for dimension {}:\n   lower bound on index slice ({:.4f}) must be less than upper bound ({:.4f})".format(dimension_key, low_bound, upper_bound))

	# bounds are within the dimension range for non-infinite bounds:
	if not np.isinf(low_bound) and not ((low_bound >= dimension[0]) and (low_bound <= dimension[-1])):
		raise ValueError("Dimension slicing by index error for dimension {}:\n   lower bound on index slice ({:.2f}) should be within the range of the dimension, from {:.4f} to {:.4f} ".format(dimension_key, low_bound, dimension[0], dimension[-1]))
	if not np.isinf(upper_bound) and not ((upper_bound >= dimension[0]) and (upper_bound <= dimension[-1])):
		raise ValueError("Dimension slicing by index error for dimension {}:\n   upper bound on index slice ({:.2f}) should be within the range of the dimension, from {:.4f} to {:.4f} ".format(dimension_key, upper_bound, dimension[0], dimension[-1]))
	
	slice_idx_list = np.squeeze(np.where(np.logical_and(dimension >= low_bound, dimension <= upper_bound)))
	return slice(slice_idx_list[0], slice_idx_list[-1]+1)

#def find_overall_cold_events(data_dict, number_of_events, distinct_conditions):
#"""
#data_dict: dictionary with 'data', 'time', 'lat', and 'lon' keys
#distinct_conditions = dictionary with conditions that determine whether events are distinct
#	includes: 'time_separation', 'lat_separation', 'lon_separation'
#"""
#t2m_on_land = data_dict['data']
#times = data_dict['time']
#latitudes = data_dict['lat']
#longitudes = data_dict['lon']

# sort 2m temperatures from lowest to highest
# sorted_idx is a tuple of index arrays; for ex, times[sorted_idx[0][0]] is the time value of the very coldest point in t2m_on_land
#idx_sorted_by_temp = np.unravel_index(t2m_on_land.argsort(axis=None), t2m_on_land.shape)
