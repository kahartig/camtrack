"""
Author: Kara Hartig

Unit tests for identify_cold_events.py
"""
import numpy as np
import os
from numpy.testing import assert_raises, assert_equal, assert_almost_equal, assert_array_equal

from camtrack.identify_cold_events import subset_by_timelatlon

NC_SAMPLE_PATH = os.path.join(os.curdir, 'sample_CAM4_for_nosetests.py')