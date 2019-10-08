"""
Author: Kara Hartig
	adapted from code written by Pratap Singh

Contains functions to:
	load a winter netCDF from CAM
	subset by time, lat, lon, and landfraction
	identify coldest events
	print corresponding CONTROL files for HYSPLIT
"""

# Standard Imports
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import cftime
import os


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


def find_overall_cold_events(data_dict, winter_idx, number_of_events, distinct_conditions):
	"""
	two events are "indistinct" only if they violate all three distinct_conditions
	data_dict: dictionary with 'data', 'time', 'lat', and 'lon' keys
		data must have dimensions data[time, lat, lon]
	distinct_conditions = dictionary with conditions that determine whether events are distinct
		includes: 'min time separation' (fractional days), 'min lat separation' (degrees), 'min lon separation' (degrees)
	"""
	t2m_on_land = data_dict['data']
	times = data_dict['time']
	latitudes = data_dict['lat']
	longitudes = data_dict['lon']

	# sort 2m temperatures from lowest to highest
	# sorted_idx is a tuple of index arrays; for ex, times[sorted_idx[0][0]] is the time value of the very coldest point in t2m_on_land, latitudes[sorted_idx[1][0]] is the corresponding latitude, etc.
	sorted_idx = np.unravel_index(t2m_on_land.argsort(axis=None), t2m_on_land.shape)

	# initialize DataFrame of cold events
	cold_events = pd.DataFrame(index=range(number_of_events), columns=['winter index', 'time index', 'time', 'lat index', 'lat', 'lon index', 'lon', '2m temp'])

	# store first (coldest) event
	cold_events.loc[0] = [winter_idx, sorted_idx[0][0], times[sorted_idx[0][0]], sorted_idx[1][0], latitudes[sorted_idx[1][0]], sorted_idx[2][0], longitudes[sorted_idx[2][0]], t2m_on_land[sorted_idx[0][0], sorted_idx[1][0], sorted_idx[2][0]]]

	# find distinct cold events, starting with second-coldest
	idx = 1
	num_found = 1
	while num_found < number_of_events:
		print('Start loop for idx {}'.format(idx))
		time_idx = sorted_idx[0][idx]
		lat_idx = sorted_idx[1][idx]
		lon_idx = sorted_idx[2][idx]
		time = times[time_idx]
		lat = latitudes[lat_idx]
		lon = longitudes[lon_idx]
		# check distinctness conditions:
		distinct = True
		for found_idx, found_row in cold_events.iterrows():
			# NOTE: iterates over empty, pre-allocated rows as well, but should not affect distinct flag
			if ((abs(time - found_row['time']) < distinct_conditions['min time separation'])
				and (abs(lat - found_row['lat']) < distinct_conditions['min lat separation'])
				and (abs(lon - found_row['lon']) < distinct_conditions['min lon separation'])):
				# indistinct only if overlapping another event simultaneously in time and location
				distinct = False
		if distinct:
			cold_events.loc[num_found]['winter index'] = winter_idx
			cold_events.loc[num_found]['time index'] = time_idx
			cold_events.loc[num_found]['time'] = time
			cold_events.loc[num_found]['lat index'] = lat_idx
			cold_events.loc[num_found]['lat'] = lat
			cold_events.loc[num_found]['lon index'] = lon_idx
			cold_events.loc[num_found]['lon'] = lon
			cold_events.loc[num_found]['2m temp'] = t2m_on_land[time_idx, lat_idx, lon_idx]
			num_found = num_found + 1
		idx = idx + 1
	return cold_events



def print_CONTROL_files(events, traj_heights, backtrack_time, output_dir, traj_dir, data_file_path):
    """
    Prints out the CONTROL files that HYSPLIT uses to set up a backtracking run using the entries in events.
    Creates CONTROL files in output_dir. Also creates a subdirectory called traj_dir, which must be in the
    HYSPLIT working directory, in which the trajectory file will be placed when HYSPLIT is run with these
    CONTROL files.

    CONTROL files will be named CONTROL_winter<winter idx>_event<events idx>
    	winter idx: taken from 'winter idx' key in events; 0 corresponds to 07-08 year, 1 to 08-09, etc.
    	events idx: the DataFrame index of each event in the events DataFrame
    trajectory files will be named traj_winter<winter idx>_event<events idx>

    Parameters:
    events: Pandas DataFrame containing details of the particular cold events for which the CONTROL files
    are being generated. events must contain at least the following:
        winter idx = index of the winter during which the event takes place (07-08 is 0, 08-09 is 1 etc)
        time = the time of the event as a real number of days since 0001-01-01 00:00:00 (fractional part for hours)
        lat = latitude coordinate of the event as a real number of degrees on -90 to 90 scale
        lon = longitude coordinate of the event as a real number of degrees on 0 to 360 scale
    traj_heights: array-like of starting height(s) in meters for each event 
    backtrack_time: hours back in time to follow trajectories; must be positive integer
    output_dir: output directory for CONTROL files
    traj_dir: HYSPLIT directory to output trajectory files
    data_file_path: full path and filename of CAM4 data file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)
    for ev_idx, ev_row in events.iterrows():
    	control_path = os.path.join(output_dir, 'CONTROL_winter{:d}_event{:02d}'.format(ev_row['winter idx'], ev_idx))
    	with open(control_path, 'w') as f:
    		data_path, data_filename = os.path.split(data_file_path)
    		t = cftime.num2date(ev_row['time'], 'days since 0001-01-01 00:00:00', calendar='noleap')
    		lat = ev_row['lat']
    		if ev_row['lon'] > 180:
    			# HYSPLIT requires longitude on a -180 to 180 scale
    			lon = ev_row['lon'] - 360
    		else:
    			lon = ev_row['lon']
    		# Print CONTROL file
    		# Start time:
    		f.write('{:02d} {:02d} {:02d} {:02d}\n'.format(t.year, t.month, t.day, t.hour))
    		# Number of starting positions:
    		f.write('{:d}\n'.format(len(traj_heights)))
    		# Starting positions:
    		for ht in traj_heights:
    			f.write('{:.1f} {:.1f} {:.1f}\n'.format(lat, lon, ht))
    		# Duration of backtrack in hours:
    		f.write('-{:d}\n'.format(backtrack_time))
    		# Vertical motion option:
    		f.write('0\n') # 0 to use data's vertical velocity fields
    		# Top of model:
    		f.write('10000.0\n')  # in meters above ground level; trajectories terminate when they reach this level
    		# Number of input files:
    		f.write('1\n')
    		# Input file path:
    		f.write(data_path + '\n')
    		# Input file name:
    		f.write(data_filename + '\n')
    		# Output trajectory file path:
    		f.write(traj_dir + '\n')
    		# Output trajectory file name:
    		f.write('traj_winter{}_event{}'.format(ev_row['winter idx'], ev_idx))