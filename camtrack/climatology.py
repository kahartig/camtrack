"""
Author: Kara Hartig

Find distinct cold events by comparing 3-hourly temperatures to winter climatology

Functions:
	anomaly_DJF:  produce an array of temperature anomalies from climatological average
	find_coldest:  sort anomaly temperature array and identify X coldest distinct events
	OLD_find_coldest:  an earlier version of find_coldest; probably redundant?
"""

# Standard Imports
import numpy as np
import pandas as pd
import cftime


def anomaly_DJF(data_dict_list, method):
	"""
	Calculate the anomaly from the DJF mean of the yearly data in data_dict_list

	Parameters
	----------
	data_dict_list: list of dictionaries
		each element in the list is a dictionary containing one winter (DJF) of
		data, a [time, lat, lon] array accessed with 'data' key
	method: string
		must be either 'absolute' or 'scaled'
		if 'absolute', output is just data - DJF mean
		if 'scaled', output is (data - DJF mean)/stdev

	Returns
	-------
	diff_dict: dictionary
		'diff': a [time, lat, lon] array of anomalies from the DJF mean,
			calculated according to method argument. Arrays from each year of
			the input have been concatenated along the first (time) axis;
			corresponding time coordinates are stored in 'time'
		'mean': a [lat, lon] array of the DJF mean across all years provided
		'time': coordinate array for time; formed by concatenating along all
			years provided in data_dict_list
		'lat': coordinate array for latitude
		'lon': coordinate array for longitude
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


def find_coldest(climatology_dict, number_of_events, distinct_conditions):
	"""
	Sort data from lowest to highest and select the number_of_events distinct
	events with the lowest values

	Two events, where 'events' are separate entries in the temperature data
	contained in climatology_dict, are considered distinct if the absolute
	difference in the times, latitudes, and longitudes of the two events are
	greater than or equal to the minimum separations given in
	distinct_conditions. Two events are indistinct only if all three
	distinctness conditions are violated.

	Parameters
	----------
	climatology_dict: dictionary
		the output of climatology.anomaly_DJF
		must include the keys 'diff' (data to be sorted), 'time', 'lat', 'lon'
	number_of_events: integer
		number of distinct events to select and return from climatology_dict
	distinct_conditions: dictionary
		'delta t': integer or float; minimum separation in days for two events
			to be considered distinct
		'delta lat': integer or float; minimum separation in degrees latitude
		'delta lon':integer or float; minimum separation in degrees longitude

	Returns
	-------
	cold_events: pandas DataFrame
		contains time, location, and associated anomaly temperature of the
		distinct events discovered, indexed by an integer ranking the events
		from coldest (0) to warmest (number_of_events - 1)
		'date': date-time string of event
		'time': time of event in days since 0001-01-01 00:00:00
		'lat': latitude of event in degrees
		'lon': longitude of event in degrees
		'temp anomaly': temperature anomaly associated with the event; taken
			from climatology_dict['diff'] at the time, lat, and lon listed
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


def OLD_find_coldest(data_dict, winter_idx, number_of_events, distinct_conditions):
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
	cold_events = pd.DataFrame(index=range(number_of_events), columns=['winter index', 'time', 'lat', 'lon', '2m temp'])

	# Store first (coldest) event
	cold_events.loc[0] = [winter_idx, times[sorted_idx[0][0]], latitudes[sorted_idx[1][0]], longitudes[sorted_idx[2][0]], t2m_on_land[sorted_idx[0][0], sorted_idx[1][0], sorted_idx[2][0]]]

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
