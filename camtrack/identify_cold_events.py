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
import os
import datetime
import sys

#def check_valid_bounds(bound_type, bounds):
#	"""
#	"""
#	lower = bounds[0]
#	upper = bounds[1]

def subset_by_timelatlon(filename, winter_idx, lat_bounds, lon_bounds, landfrac_min):
	"""
	filename: os.path to concatenated netCDF file
	winter_idx: index of winter under study
	lat_bounds: list-like of [min_lat, max_lat]
	lon_bounds: list-like of [min_lon, max_lon]
	landfrac_min: minumum landfraction threshold to consider "over land"
	"""
	nc_file = Dataset(filename)
	latitudes = nc_file.variables['lat'][:]  # WHY NO .DATA FOR LAT AND LON??????
	longitudes = nc_file.variables['lon'][:]
	times = nc_file.variables['time'][:].data

	# time subset: define winter as Dec-Jan-Feb
	min_time = datetime.datetime.toordinal(datetime.datetime(year = 7 + winter_idx, month = 12, day = 1))
	max_time = datetime.datetime.toordinal(datetime.datetime(year = 8 + winter_idx, month = 3, day = 1))
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