"""
Author: Kara Hartig

Unit tests for identify_cold_events.py
"""
# Standard imports
import numpy as np
import os
from netCDF4 import Dataset

# Testing imports
from numpy.testing import assert_raises, assert_equal, assert_almost_equal, assert_array_equal

# camtrack imports
from camtrack.identify_cold_events import subset_by_timelatlon, slice_from_bounds

# Globals
NC_SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_CAM4_for_nosetests.nc')
NC_SAMPLE_FILE = Dataset(NC_SAMPLE_PATH)


# subset by time, lat, and lon
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

