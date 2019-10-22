"""
Author: Kara Hartig

Contains functions to:
	calculate the difference from climatology for a list of data dictionaries
"""

# Standard Imports
import numpy as np


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
	diff_dict = {'diff': difference, 'time': time_all_winters, 'lat': latitudes, 'lon': longitudes}
	return diff_dict


