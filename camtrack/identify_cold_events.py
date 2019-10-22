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
	Take a subset in time and lat/lon of the variable specified by
	desired_variable_key from filename and mask by landfraction. The time subset
	runs from December 1st through February 28th of the following year and the
	spatial extent must include the north pole, requiring only a lower bound for
	latitude.

	Parameters
	----------
	filename: string
		path to netCDF file of CAM4 output covering at least Dec through Feb of
		a given year
	winter_idx: integer
		index of winter under study. 0 is 07-08 year, 1 is 08-09, etc.
	desired_variable_key: string
		key of variable to be subset by time, latitude, and longitude
		the corresponding data must have the dimensions data[time, lat, lon]
	lat_lower_bound: integer or float
		lower bound of latitude range for the subset; upper bound is North Pole
		latitude subset = [lat_lower_bound, 90.0]
	lon_bounds: array-like of floats
		lower and upper bounds of longitude range for the subset
		must be in the order (lower bound, upper bound)
		longitude subset = [lon_bounds[0], lon_bounds[1]]
	landfrac_min: float between 0 and 1
		minimum value of landfraction for which a gridpoint will be considered
		"on land"
	testing: boolean
		if testing=True, activates special conditions on time bounds and output
		for nosetests
		default is False

	Returns
	-------
	subset_dict: dictionary
		'data': a subset from time=Dec 1st - Feb 28th,
			latitude=[lat_lower_bound, 90.0],
			longitude=[lon_bounds[0], lon_bounds[1]] of the variable
			desired_variable_key, masked by landfraction so that any points with
			landfraction < landfrac_min have a value of np.nan
		'time': array of ordinal time values corresponding to subsetted time
			dimension
		'lat': array of latitudes corresponding to subsetted lat dimension
		'lon': array of longitudes corresponding to subsetted lon dimension
		if testing=True, also includes:
			'unmasked_data': same as 'data' but without the landfraction masking
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
	datetime_min = cftime.num2date(min_time, time_object.units, calendar=time_object.calendar)
	datetime_max = cftime.num2date(max_time, time_object.units, calendar=time_object.calendar)
	print('Taking a subset in time and location of variable {}:'.format(desired_variable_key))
	print('    time: {:04d}-{:02d}-{:02d} to {:04d}-{:02d}-{:02d}'.format(datetime_min.year, datetime_min.month, datetime_min.day, datetime_max.year, datetime_max.month, datetime_max.day))
	print('    latitude: {:+.1f} to {:+.1f}'.format(lat_lower_bound, 90.0))
	print('    longitude: {:+.1f} to {:+.1f}'.format(lon_bounds[0], lon_bounds[1]))
	variable_subset = variable_object[time_subset, lat_subset, lon_subset].data

	# mask by landfraction
	#   replace any value in variable_subset where landfraction < landfrac_min with np.nan
	print('Masking {} by landfraction: np.nan anywhere landfraction < {:.2f}'.format(desired_variable_key, landfrac_min))
	landfrac_subset = nc_file.variables['LANDFRAC'][time_subset, lat_subset, lon_subset].data
	masked_variable = np.where(landfrac_subset >= landfrac_min, variable_subset, np.nan)
	subset_dict = {'data': masked_variable, 'time': time_object[time_subset].data, 'lat': latitude_object[lat_subset].data, 'lon': longitude_object[lon_subset].data}
	if testing:
		subset_dict['unmasked_data'] = variable_subset
	return subset_dict


def slice_from_bounds(file, dimension_key, low_bound, upper_bound=np.inf):
	"""
	Given closed bounds [low_bound, upper_bound], return a slice object of the
	given dimension that spans the range low_bound <= dimension <= upper_bound.

	For example, if dim is the array of values in the dimension, then dim[slice]
	will return those values of dim in the closed interval
	[low_bound, upper_bound]. If var is a variable with the corresponding
	dimension, var[dimension], then var[slice] will return the values of var at
	locations where dim is in the closed interval [low_bound, upper_bound].

	Raises an error if any of the non-infinite bounds are outside the range of
	the dimension.

	Parameters
	----------
	file: instance of netCDF4 Dataset
		netCDF file containing the dimension to be sliced
	dimension_key: string
		name of dimension to be sliced
	low_bound: float or -np.inf
		lower bound of closed dimension slice
		if -np.inf, lower bound will be the lowest value in the dimension
	upper_bound: float or np.inf
		upper bound of closed dimension slice
		if np.inf, upper bound will be the highest value in the dimension

	Returns
	-------
	slice object spanning the closed interval [low_bound, upper_bound] of the
	dimension
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
	Identify the x coldest distinct events for the temperature data given in
	data_dict, where x=number_of_events and the conditions for distinctness are
	given in distinct_conditions.

	Two events, where 'events' are separate entries in the temperature data
	contained in data_dict, are considered distinct if the absolute
	difference in the times, latitudes, and longitudes of the two events are
	greater than or equal to the minimum separations given in
	distinct_conditions. Two events are indistinct only if all three
	distinctness conditions are violated.

	Parameters
	----------
	data_dict: dictionary
		'data': array of temperature data with dimensions data[time, lat, lon]
		'time': array of ordinal time values corresponding time dimension
		'lat': array of latitudes corresponding to lat dimension
		'lon': array of longitudes corresponding to lon dimension
	winter_idx: integer
		index of winter under study. 0 is 07-08 year, 1 is 08-09, etc.
	number_of_events: integer
		number of distinct cold events to identify, starting with the coldest
	distinct_conditions: dictionary
		'min time separation': fractional days; minimum separation between two
			events in time to be considered time-distinct
		'min lat separation': degrees; minimum separation between two events in
			latitude to be considered latitude-distinct
		'min lon separation': degrees; minimum separation between two events in
			longitude to be considered longitude-distinct

	Returns
	-------
	cold_events: Pandas DataFrame
		index: integer; 0 for coldest event, 1 for second-coldest, etc.
		columns:
			'winter index': winter_idx
			'time': time of event in days since 0001-01-01 on 'noleap' calendar
			'lat': latitude of event in degrees on -90 to 90 scale
			'lon': longitude of event in degrees on 0 to 360 scale
			'2m temp': temperature (in K) at 2 meters of the event 
	"""
	t2m_on_land = data_dict['data']
	times = data_dict['time']
	latitudes = data_dict['lat']
	longitudes = data_dict['lon']

	# Sort 2m temperatures from lowest to highest
	# sorted_idx is a tuple of index arrays
		# first index is dimension: 0=time, 1=lat, 2=lon
		# second index is sorted order: coldest value occurs at time
		#     times[sorted_idx[0][0]], lat latitudes[sorted_idx[1][0]], etc.
	sorted_idx = np.unravel_index(t2m_on_land.argsort(axis=None), t2m_on_land.shape)

	# Initialize DataFrame of cold events
	cold_events = pd.DataFrame(index=range(number_of_events), columns=['winter index', 'time index', 'time', 'lat index', 'lat', 'lon index', 'lon', '2m temp'])

	# Store first (coldest) event
	cold_events.loc[0] = [winter_idx, sorted_idx[0][0], times[sorted_idx[0][0]], sorted_idx[1][0], latitudes[sorted_idx[1][0]], sorted_idx[2][0], longitudes[sorted_idx[2][0]], t2m_on_land[sorted_idx[0][0], sorted_idx[1][0], sorted_idx[2][0]]]

	# Find distinct cold events, starting with second-coldest
	idx = 1
	num_found = 1
	print('Starting loop over sorted temperatures to identify cold events...')
	while num_found < number_of_events:
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
			cold_events.loc[num_found]['time'] = time
			cold_events.loc[num_found]['lat'] = lat
			cold_events.loc[num_found]['lon'] = lon
			cold_events.loc[num_found]['2m temp'] = t2m_on_land[time_idx, lat_idx, lon_idx]
			num_found = num_found + 1
		idx = idx + 1
	print('Found {:d} distinct cold events out of the {:d} coldest datapoints in sorted temperature array'.format(number_of_events, idx-1))
	return cold_events



def print_CONTROL_files(events, traj_heights, backtrack_time, output_dir, traj_dir, data_file_path):
	"""
	Generate the CONTROL files that HYSPLIT uses to set up a backtracking run
	based on the entries in 'events'.

	Also creates two directories:
		output_dir: where CONTROL files are placed after creation
		traj_dir: where trajectory files will be placed when HYSPLIT is run with
			these CONTROL files

	CONTROL files will be named CONTROL_winter<winter idx>_event<events idx>
		winter idx: taken from 'winter idx' key in events
			0 corresponds to 07-08 year, 1 to 08-09, etc.
		events idx: the DataFrame index of each event in the events DataFrame
	trajectory files will be named traj_winter<winter idx>_event<events idx>

	Parameters
	----------
	events: Pandas DataFrame
		each index/row corresponds to a distinct cold event for which a CONTROL
		file will be generated. Columns must include:
			'winter index': index of winter during which the event takes place
				(07-08 is 0, 08-09 is 1 etc)
			'time': time of event in days since 0001-01-01 on 'noleap' calendar
			'lat': latitude of event in degrees on -90 to 90 scale
			'lon': longitude of event in degrees on 0 to 360 scale
	traj_heights: array-like
		starting heights in meters for each trajectory
	backtrack_time: integer
		number of hours to follow each trajectory back in time
	output_dir: string
		output directory for CONTROL files
	traj_dir: string
		HYSPLIT directory to output trajectory files
	data_file_path: string
		full path and filename of CAM4 data file from which to calculate
		trajectories
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	if not os.path.exists(traj_dir):
		os.makedirs(traj_dir)
	for ev_idx, ev_row in events.iterrows():
		control_path = os.path.join(output_dir, 'CONTROL_winter{:d}_event{:02d}'.format(ev_row['winter index'], ev_idx))
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
			f.write('traj_winter{}_event{}'.format(ev_row['winter index'], ev_idx))