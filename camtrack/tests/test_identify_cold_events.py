"""
Author: Kara Hartig

Unit tests for identify_cold_events.py
"""
# Standard imports
import numpy as np
import os
from netCDF4 import Dataset

# Testing imports
from numpy.testing import assert_raises, assert_equal, assert_allclose, assert_array_equal

# camtrack imports
from camtrack.identify_cold_events import subset_by_timelatlon, slice_from_bounds

# Sample netCDF file from CAM4
# time range: 0008-12-01 00:00:00 to 0008-12-07 00:00:00 (YYYY-MM-DD HH:MM:SS)
# latitude range: 80.52631579 to 90.0
# longitude range: 320.0 to 330.0 (on 0-360 scale)
NC_SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_CAM4_for_nosetests.nc')
NC_SAMPLE_FILE = Dataset(NC_SAMPLE_PATH)
WINTER_IDX = 1 # index 1 corresponds to 08-09 winter
PS_SUBSET_FIRST_VALUE = 101194.05 # value of 'PS' for time=2889., lat=86.2, lon=322.5
PS_SUBSET_LAST_VALUE = 99774.55 # value of 'PS' for time=2895., lat=90, lon=327.5
T2M_LAST_LAND_VALUE = 240.78333 # value of 'TREFHT' for time=2895., lat=82.4, lon=330.0 (last "on land" point in TREFHT data)
GRIDPOINTS_ON_LAND = 4  # number of gridpoints with landfraction >= 90 for a single timestep in lat=(82.4-90), lon=(322.5, 330)
LENGTH_OF_TIMES_DIMENSION = 49

# tests for slice_from_bounds:
def test_slice_file_type():
    invalid_file_type = 'null string'
    assert_raises(TypeError, slice_from_bounds, invalid_file_type, 'time', 0)

# cannot createDimension without opening file in write mode
#def test_slice_nonmonotonic_dimension():
#    file = NC_SAMPLE_FILE
#    bad_dimension = file.createDimension('null', 3)
#    bad_dimension_var = file.createVariable('null', np.int, ('null',))
#    bad_dimension_var[:] = np.array([0, 2, 1])
#    assert_raises(ValueError, slice_from_bounds, file, 'null', 1)

def test_slice_wrong_bound_order():
    bad_lowbound = 2891
    bad_upperbound = 2890
    assert_raises(ValueError, slice_from_bounds, NC_SAMPLE_FILE, 'time', bad_lowbound, bad_upperbound)

def test_slice_lowbound_out_of_range():
    bad_lowbound = -100
    assert_raises(ValueError, slice_from_bounds, NC_SAMPLE_FILE, 'lat', bad_lowbound)

def test_slice_upperbound_out_of_range():
    good_lowbound = 100
    bad_upperbound = 400
    assert_raises(ValueError, slice_from_bounds, NC_SAMPLE_FILE, 'lon', good_lowbound, bad_upperbound)

def test_slice_good_timeslice():
    # from ordinal time 2890 (index 8) through 2891 (index 16)
    expected_timelist = [2890., 2890.125, 2890.25, 2890.375, 2890.5, 2890.625,
                         2890.75, 2890.875, 2891.]
    time_slice = slice_from_bounds(NC_SAMPLE_FILE, 'time', 2890, 2891)
    sliced_time = NC_SAMPLE_FILE.variables['time'][time_slice].data
    assert_array_equal(expected_timelist, sliced_time)

# tests for subset_timelatlon:
def test_subset_corners():
    var_key = 'PS'
    lat_min = 85 # should pull (86.2, 88.1, 90)
    lon_bounds = (322, 328) # should pull (322.5, 325 , 327.5)
    output = subset_by_timelatlon(NC_SAMPLE_PATH, WINTER_IDX, var_key, lat_min, lon_bounds, testing=True)
    assert_allclose(output['unmasked_data'][0,0,0], PS_SUBSET_FIRST_VALUE)
    assert_allclose(output['unmasked_data'][-1,-1,-1], PS_SUBSET_LAST_VALUE)

def test_subset_dimension_slices():
    var_key = 'PS'
    lat_min = 84
    lon_bounds = (323, 328)
    expected_lats = np.array([84.31578947, 86.21052632, 88.10526316, 90])
    expected_lons = np.array([325. , 327.5])
    output = subset_by_timelatlon(NC_SAMPLE_PATH, WINTER_IDX, var_key, lat_min, lon_bounds, testing=True)
    assert_allclose(output['lat'], expected_lats)
    assert_allclose(output['lon'], expected_lons)
    assert_allclose(output['unmasked_data'].shape, [len(output['time']), len(output['lat']), len(output['lon'])])

def test_landfrac_masking():
    var_key = 'TREFHT'
    lat_min = 81
    lon_bounds = (321, 330)
    output = subset_by_timelatlon(NC_SAMPLE_PATH, WINTER_IDX, var_key, lat_min, lon_bounds, testing=True)
    assert_equal(np.sum(~np.isnan(output['data'])), GRIDPOINTS_ON_LAND*LENGTH_OF_TIMES_DIMENSION)
    last_unmasked_value = output['data'][np.where(~np.isnan(output['data']))][-1]
    assert_allclose(last_unmasked_value, T2M_LAST_LAND_VALUE)