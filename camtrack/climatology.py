"""
Author: Kara Hartig

Find distinct cold events by comparing 3-hourly temperatures to winter climatology

Functions:
    anomaly_DJF:  produce an array of temperature anomalies from climatological average
    sample_coldtail:  randomly sample events from specified percentile range of an array of temperature anomalies
    find_coldest:  identify X coldest distinct events from an array of temperature anomalies
"""

# Standard Imports
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import math

# Misc imports
from numpy import random


def anomaly_DJF(data_list, method):
    """
    Calculate the anomaly from the DJF mean of the yearly data in data_dict_list

    Parameters
    ----------
    data_dict_list: list of DataArrays
        each element in the list is a DataArray containing one winter (DJF) of
        (time, lat, lon) data
    method: string
        must be either 'absolute' or 'scaled'
        if 'absolute', output is data - DJF mean
        if 'scaled', output is (data - DJF mean)/stdev

    Returns
    -------
    diff_dict: dictionary
        'diff': a (time, lat, lon) DataArray of anomalies from the DJF mean,
            calculated according to 'method' argument. Arrays from each year of
            the input have been concatenated along the time axis
        'mean': a (lat, lon) DataArray of the DJF mean across all years provided
    """
    data_all_winters = xr.concat(data_list, dim='time')
    mean_all_winters = data_all_winters.mean(dim='time')
    stdev_all_winters = data_all_winters.std(dim='time')
    if method == 'absolute':
        difference = data_all_winters - mean_all_winters
    elif method == 'scaled':
        difference = (data_all_winters - mean_all_winters)/stdev_all_winters
    else:
        raise ValueError("Method must be 'absolute' or 'scaled'")

    # make dictionary to hold mean and anomaly arrays
    diff_dict = {'diff': difference, 'mean': mean_all_winters}
    return diff_dict


def sample_coldtail(climatology_dict, number_of_events, percentile_range, seed=None):
    """
    Sort data from lowest to highest and randomly sample number_of_events from a
    specified percentile range of the distribution

    Parameters
    ----------
    climatology_dict: dict
        dictionary such as the output of climatology.anomaly_DJF(), containing
        'diff', a (time, lat, lon) DataArray of anomalies from a mean which will
        be arranged from lowest to highest and randomly sampled
    number_of_events: int
        number of events to sample from the given percentile range of the
        anomaly array 
    percentile_range: list-like
        (minimum, maximum) of percentile range to sample from
        each element is a number between 0 and 100
    seed: int
        if seed is not None, it is passed to randint
        Default is None

    Returns
    -------
    cold_events: pandas.DataFrame
        list of events sampled from climatology_dict['diff']
        contains columns:
            'time': ordinal time of event
            'date': date string of event YYYY-MM-DD HH:MM:SS
            'cftime date': cftime.DatetimeNoLeap() instances
            'lat': latitude
            'lon': longitude
            'temp anomaly': anomaly value of event from climatology_dict['diff']
    """
    def samples2values(samples_idx, sorted_coord_idx, coordinates):
        # sorted_coord_idx = sorted_idx[index of desired coordinate]
        coordinate_idx = [sorted_coord_idx[idx] for idx in samples_idx]
        coordinate_values = [coordinates[c_idx] for c_idx in coordinate_idx]
        return coordinate_values, coordinate_idx

    temperature_anomaly = climatology_dict['diff'].values
    times = climatology_dict['diff'].time.values
    latitudes = climatology_dict['diff'].lat.values
    longitudes = climatology_dict['diff'].lon.values

    # Sort by temperature anomaly, from coldest (most negative) to warmest (most positive and NaN)
    sorted_idx = np.unravel_index(temperature_anomaly.argsort(axis=None), temperature_anomaly.shape)
        # first index: 0-index time array, 1-index lat array, 2-index lon array
        # second index runs from coldest to warmest

    # Randomly sample events from given percentile range
    num_notnan = np.count_nonzero(~np.isnan(temperature_anomaly))
    idx_lower = int(math.ceil( (percentile_range[0]/100.)*num_notnan ))
    idx_upper = int(math.floor( (percentile_range[1]/100.)*num_notnan )) - 1
    if seed is not None:
        random.seed(seed)
    idx_in_percentile = np.arange(idx_lower, idx_upper + 1)
    samples = random.choice(idx_in_percentile, number_of_events, replace=False)
    #samples = random.randint(idx_lower, idx_upper, number_of_events) # samples is a list of indices for the second index of sorted_idx

    # Pull coordinate values cooresponding to samples
    sample_times, sample_times_idx = samples2values(samples, sorted_idx[0], times)
    sample_lat, sample_lat_idx = samples2values(samples, sorted_idx[1], latitudes)
    sample_lon, sample_lon_idx = samples2values(samples, sorted_idx[2], longitudes)
    sample_temp_anomaly = [temperature_anomaly[t, la, lo] for t,la,lo in zip(sample_times_idx, sample_lat_idx, sample_lon_idx)]

    # Convert sampled times to ordinal time and datestring
    ordinal_times = cftime.date2num(sample_times, 'days since 0001-01-01 00:00:00', calendar='noleap')
    datestrings = [t.strftime() for t in sample_times]

    # Construct cold_events DataFrame
    cold_events = pd.DataFrame.from_dict({'time': ordinal_times, 'date': datestrings, 'cftime date': sample_times, 'lat': sample_lat, 'lon': sample_lon, 'temp anomaly': sample_temp_anomaly})
    print("Randomly sampled {} events from the {}-{} percentile range of temperature anomalies from DJF mean".format(number_of_events, percentile_range[0], percentile_range[1]))
    return cold_events


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
    def convert_times(cf_datetime):
        """cf_datetime is a list of cftime.DatetimeNoLeap instance(s) """
        ordinal_time = cftime.date2num(cf_datetime, 'days since 0001-01-01 00:00:00', calendar='noleap')
        datestring = [t.strftime() for t in cf_datetime]
        return ordinal_time, datestring

    def samples2values(samples_idx, sorted_coord_idx, coordinates):
        # sorted_coord_idx = sorted_idx[index of desired coordinate][:]
        coordinate_idx = [sorted_coord_idx[idx] for idx in samples_idx]
        coordinate_values = [coordinates[c_idx] for c_idx in coordinate_idx]
        return coordinate_values, coordinate_idx

    temperature_anomaly = climatology_dict['diff'].values
    times = climatology_dict['diff'].time.values
    latitudes = climatology_dict['diff'].lat.values
    longitudes = climatology_dict['diff'].lon.values

    # Sort by temperature anomaly, from coldest (most negative) to warmest (most positive and NaN)
    sorted_idx = np.unravel_index(temperature_anomaly.argsort(axis=None), temperature_anomaly.shape)

    # Find distinct cold events, starting with second-coldest
    event_idx = [0] # store all distinct event indices, starting with coldest
    idx = 1
    num_found = 1
    print('Starting loop over sorted temperature anomalies to identify cold events...')
    while ((num_found < number_of_events) and (idx < len(times))):
        time = times[sorted_idx[0][idx]]
        lat = latitudes[sorted_idx[1][idx]]
        lon = longitudes[sorted_idx[2][idx]]
        [ord_time], [datestring] = convert_times([time])
        # check distinctness conditions:
        distinct = True
        found_time, _ = samples2values(event_idx, sorted_idx[0], times)
        found_lat, _ = samples2values(event_idx, sorted_idx[1], latitudes)
        found_lon, _ = samples2values(event_idx, sorted_idx[2], longitudes)
        found_ord, found_datestring = convert_times(found_time)
        for f_time, f_lat, f_lon in zip(found_ord, found_lat, found_lon):
            if ((abs(ord_time - f_time) < distinct_conditions['delta t'])
                and (abs(lat - f_lat) < distinct_conditions['delta lat'])
                and (abs(lon - f_lon) < distinct_conditions['delta lon'])):
                # indistinct only if overlapping another event simultaneously in time and location
                distinct = False
        if distinct:
            event_idx.append(idx)
            num_found = num_found + 1
        idx = idx + 1
    print('Found {:d} distinct cold events out of the {:d} coldest datapoints in sorted temperature anomaly array'.format(num_found, idx-1))
    
    # Pull coordinate values cooresponding to samples
    sample_times, sample_times_idx = samples2values(event_idx, sorted_idx[0], times)
    sample_lat, sample_lat_idx = samples2values(event_idx, sorted_idx[1], latitudes)
    sample_lon, sample_lon_idx = samples2values(event_idx, sorted_idx[2], longitudes)
    sample_temp_anomaly = [temperature_anomaly[t, la, lo] for t,la,lo in zip(sample_times_idx, sample_lat_idx, sample_lon_idx)]

    # Convert sampled times to ordinal time and datestring
    ordinal_times, datestrings = convert_times(sample_times)

    # Generate cold events dataframe
    cold_events = pd.DataFrame.from_dict({'time': ordinal_times, 'date': datestrings, 'cftime date': sample_times, 'lat': sample_lat, 'lon': sample_lon, 'temp anomaly': sample_temp_anomaly})
    return cold_events
