"""
Author: Kara Hartig

Contains functions to:
	calculate the difference from climatology for a list of data dictionaries
"""

# Standard Imports
import numpy as np
import pandas as pd
import cftime


def compare_to_climatology(data_dict_list, method):
	"""
	DOC
	method: 'absolute' or 'scaled'
	"""
	# Check that all data arrays have the same lat and lon dimensions
	same_lat = np.allclose(np.concatenate([d['lat'] for d in data_dict_list]), np.tile(data_dict_list[0]['lat'], len(data_dict_list)))
	same_lon = np.allclose(np.concatenate([d['lon'] for d in data_dict_list]), np.tile(data_dict_list[0]['lon'], len(data_dict_list)))
	if same_lat:
		latitudes = data_dict_list[0]['lat']
	else:
		raise ValueError('Latitude dimensions of data in data_dict_list do not match')
	if same_lon:
		longitudes = data_dict_list[0]['lon']
	else:
		raise ValueError('Longitude dimensions of data in data_dict_list do not match')

	data_all_winters = np.concatenate([d['data'] for d in data_dict_list], axis=0)
	mean_all_winters = np.nanmean(data_all_winters, axis=0)
	stdev_all_winters = np.nanstd(data_all_winters, axis=0)
	if method == 'absolute':
		difference = data_all_winters - mean_all_winters  # CHECK NUMPY BROADCASTING
	elif method == 'scaled':
		difference = (data_all_winters - mean_all_winters)/stdev_all_winters
	else:
		raise ValueError("Method must be 'absolute' or 'scaled'")

	# make dictionary to hold difference array and dimensions
	time_all_winters = np.concatenate([d['time'] for d in data_dict_list])
	diff_dict = {'diff': difference, 'mean': mean_all_winters, 'time': time_all_winters, 'lat': latitudes, 'lon': longitudes}
	return diff_dict


def sort_and_find_coldest(climatology_dict, number_of_events, distinct_conditions):
	"""
	DOC
	"""
	temperature_anomaly = climatology_dict['diff']
	times = climatology_dict['time']
	latitudes = climatology_dict['lat']
	longitudes = climatology_dict['lon']

	# Sort by temperature anomaly, from coldest (most negative) to warmest (most positive and NaN)
	sorted_idx = np.unravel_index(temperature_anomaly.argsort(axis=None), temperature_anomaly.shape)

	# Initialize cold_events DataFrame
	cold_events = pd.DataFrame(index=range(number_of_events), columns=['date', 'time', 'lat', 'lon', 'temp anomaly'])
	date_coldest = cftime.num2date(times[sorted_idx[0][0]], 'days since 0001-01-01 00:00:00', calendar='noleap')
	cold_events.loc[0] = ['{:04d}-{:02d}-{:02d}'.format(date_coldest.year, date_coldest.month, date_coldest.day), times[sorted_idx[0][0]], latitudes[sorted_idx[1][0]], longitudes[sorted_idx[2][0]], temperature_anomaly[sorted_idx[0][0], sorted_idx[1][0], sorted_idx[2][0]]]

	# Find distinct cold events, starting with second-coldest
	idx = 1
	num_found = 1
	print('Starting loop over sorted temperature anomalies to identify cold events...')
	while ((num_found < number_of_events) and (idx < len(times))):
		time_idx = sorted_idx[0][idx]
		lat_idx = sorted_idx[1][idx]
		lon_idx = sorted_idx[2][idx]
		time = times[time_idx]
		date = cftime.num2date(time, 'days since 0001-01-01 00:00:00', calendar='noleap')
		date_string = '{:04d}-{:02d}-{:02d}'.format(date.year, date.month, date.day)
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
			cold_events.loc[num_found]['date'] = date_string
			cold_events.loc[num_found]['time'] = time
			cold_events.loc[num_found]['lat'] = lat
			cold_events.loc[num_found]['lon'] = lon
			cold_events.loc[num_found]['temp anomaly'] = temperature_anomaly[time_idx, lat_idx, lon_idx]
			num_found = num_found + 1
		idx = idx + 1
	print('Found {:d} distinct cold events out of the {:d} coldest datapoints in sorted temperature anomaly array'.format(number_of_events, idx-1))
	return cold_events