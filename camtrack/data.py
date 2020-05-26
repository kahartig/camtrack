"""
Author: Kara Hartig

Handle data from HYSPLIT trajectories and CAM model output

Classes:
    TrajectoryFile:  read in and store data from HYSPLIT .traj file
    WinterCAM:  read in and store data from CAM4 winter (Nov-Feb) file
Functions:
    subset_nc:  subset a netCDF file by time, latitude, and longitude
    slice_dim:  given coordinate bounds, generate index slice for corresponding dimension
    make_CONTROL:  print a CONTROL file for running HYSPLIT trajectories
    winter_string:  return years corresponding to the winter in which a given time falls

ASSUMPTIONS:
    CAM time units are days since 0001-01-01 00:00:00 and calendar is 'noleap'
    all trajectories cover same time window; none terminate early
    available CAM variables are the same as those listed by Zeyuan (see WinterCAM.name_to_h)
    CONTROL file input data is stored in directory structure winter_YY-YY/pi_3h_YYYY.arl
"""

# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
import os
import cftime
import calendar
import Ngl


class TrajectoryFile:
    '''
    Store of all data read in from a .traj file, an ASCII file
    output from HYSPLIT. These .traj files contain lat/lon locations of
    an air mass every hour along a trajectory.

    Methods
    -------
    winter
        return year(s) corresponding to this trajectory in user-specified format

    Attributes
    ----------
    grids: pandas DataFrame
        contains all grids used: model, year, month, day, hour, and fhour
    traj_start: pandas DataFrame
        contains initialization information for all trajectories: date, time,
        and location at which trajectory was started
    diag_var_names: list
        list of names of diagnostic output variables. Corresponding values along
        trajectories are in the trailing columns of the .traj file
    ntraj: int
        number of distinct trajectories stored in .traj file
    direction: string
        direction of trajectory calculation: 'FORWARD' or 'BACKWARD'
    data: pandas DataFrame
        trajectory data every 3 hours
        uses a MultiIndex:
            top level: int from 1 to ntraj
                trajectory number 'traj #'
            second level: float from 0.0 to -(total length of trajectory)
                age of trajectory 'traj age'
            EX: to access trajectory 3 at time -5 hours, use data.loc[3, -5]
        columns: grid #, year, month, day, hour, minute, fhour, lat, lon,
            height (m), <any diagnostic variables...>, datetime, cftime date,
            numerical time
        note that there are three equivalent representations of time:
            datetime: str; 'YYYY-MM-DD HH:MM:SS'
            cftime date: cftime.DatetimeNoLeap()
            numerical time: float; days since 0001-01-01 00:00:00
    data_1h: pandas DataFrame
        same as data, but every hour
    data_12h: pandas DataFrame
        same as data, but every 12 hours
    data_24h: pandas DataFrame
        same as data, but every 24 hours
    '''

    def __init__(self, filepath):
        '''
        Parameters
        ----------
        filepath:  string
            file path to HYSPLIT trajectory file
        '''
        # open the .traj file
        file = open(filepath, 'r')

        # Header 1
        #    number of meteorological grids used
        header_1 = file.readline()
        header_1 = (header_1.strip()).split()
        ngrids = int(header_1[0])

        # read in list of grids used
        h1_columns = ['model', 'year', 'month', 'day', 'hour', 'fhour']
        h1_dtypes = ['str', 'int32', 'int32', 'int32', 'int32', 'int32']

        # loop over each grid
        grids_list = []
        for i in range(ngrids):
            line = file.readline().strip().split()
            grids_list.append(line)
        grids_df = pd.DataFrame(grids_list, columns=h1_columns)
        self.grids = grids_df.astype(dict(zip(h1_columns, h1_dtypes)))

        # Header 2
        #    col 0: number of different trajectories in file
        #    col 1: direction of trajectory calculation (FORWARD, BACKWARD)
        #    col 2: vertical motion calculation method (OMEGA, THETA, ...)
        header_2 = file.readline()
        header_2 = (header_2.strip()).split()
        ntraj = int(header_2[0])          # number of trajectories
        direction = header_2[1]           # direction of trajectories
        vert_motion = header_2[2]         # vertical motion calculation method
        self.ntraj = ntraj
        self.direction = direction

        # read in list of trajectories
        h2_columns = ['year', 'month', 'day', 'hour', 'lat', 'lon', 'height']
        h2_dtypes = ['int32', 'int32', 'int32',
                     'int32', 'float32', 'float32', 'float32']

        # loop over each trajectory
        traj_start_list = []
        for i in range(ntraj):
            line = file.readline().strip().split()
            traj_start_list.append(line)
        traj_df = pd.DataFrame(traj_start_list, columns=h2_columns)
        self.traj_start = traj_df.astype(dict(zip(h2_columns, h2_dtypes)))

        # Header 3
        #    col 0 - number (n) of diagnostic output variables
        #    col 1+ - label identification of each of n variables (PRESSURE,
        #             AIR_TEMP, ...)
        header_3 = file.readline()
        header_3 = header_3.strip().split()
        nvars = int(header_3[0])  # number of diagnostic variables
        self.diag_var_names = header_3[1:]

        file.close()

        # Trajectories
        #    0 - trajectory number
        #    1 - grid number
        #    2 - year [calendar date, year part only]
        #    3 - month [calendar date, month part only]
        #    4 - day [calendar date, day part only]
        #    5 - hour [calendar time, hour part only]
        #    6 - minute [calendar time, minute part only]
        #    7 - forecast hour [if .arl file had times appended one-by-one, this will always be 0]
        #    8 - age [hours since start of trajectory; negative for backwards trajectory]
        #    9 - lats
        #    10 - lons
        #    11 - height
        #    ... and then any additional diagnostic output variables
        traj_columns = ['traj #', 'grid #', 'year', 'month', 'day', 'hour',
                        'minute', 'fhour', 'traj age', 'lat', 'lon', 'height (m)']
        traj_dtypes = {'traj #': int, 'grid #': int, 'year': int, 'month': int, 'day': int, 'hour': int,
                       'minute': int, 'fhour': int, 'traj age': int, 'lat': float, 'lon': float, 'height (m)': float}
        for var in self.diag_var_names:
            traj_columns.append(var)
            traj_dtypes[var] = float
        # skip over header; length depends on number of grids and trajectories
        traj_skiprow = 1 + ngrids + 1 + ntraj + 1

        # read in file as csv
        trajectories = pd.read_csv(filepath, delim_whitespace=True, header=None, names=traj_columns, index_col=[
                                   0, 8], dtype=traj_dtypes, skiprows=traj_skiprow)
        trajectories.sort_index(inplace=True)

        # convert longitudes from -180 to 180 to 0 to 360 scale for consistency with CAM files
        trajectories['lon'].mask(trajectories['lon'] < 0, trajectories['lon'] + 360, inplace=True)

        # new column: datetime string
        def traj_datetime(row):
            return '00{:02.0f}-{:02.0f}-{:02.0f} {:02.0f}:{:02.0f}:00'.format(row['year'], row['month'], row['day'], row['hour'], row['minute'])
        trajectories['datetime'] = trajectories.apply(traj_datetime, axis=1)

        # new column: cftime Datetime objects
        def traj_cftimedate(row):
            return cftime.DatetimeNoLeap(row['year'], row['month'], row['day'], row['hour'], row['minute'])
        trajectories['cftime date'] = trajectories.apply(
            traj_cftimedate, axis=1)

        # new column: numerical time (days since 0001-01-01 00:00:00)
        # min_time = cftime.date2num(cftime.datetime(7 + winter_idx, 12, 1),
        # time_object.units, calendar=time_object.calendar)
        def traj_numtime(row):
            return cftime.date2num(row['cftime date'], units='days since 0001-01-01 00:00:00', calendar='noleap')
        trajectories['numerical time'] = trajectories.apply(
            traj_numtime, axis=1)

        # Store trajectories in increments of 1 hour, 3 hours, 12, and 24
        # default self.data will be every 3 hours to match CAM output frequency
        self.data_1h = trajectories
        self.data = trajectories[trajectories['hour'] % 3 == 0] # every 3 hours
        self.data_12h = trajectories[trajectories['hour'] % 12 == 0]
        self.data_24h = trajectories[trajectories['hour'] % 24 == 0]

    def col2da(self, trajectory_number, data_column, include_coords=None):
        '''
        Convert any trajectory data column into an xarray.DataArray with
        additional coordinate(s) given by other column(s)

        Dimension of resulting DataArray will be highest-order index in 
        trajectory data, self.data. If additional columns are requested in
        include_coords, they will share the same dimension.

        Parameters
        ----------
        trajectory_number: integer
            number corresponding to specific trajectory of interest
            Trajectory data is retrieved with self.data.loc[trajectory_number]
        data_column: string
            key of data column to be converted to DataArray
        include_coords: string or list of strings
            keys for other columns to be included as additional coordinates in
            the DataArray produced
            Default is None: no additional coordinates
        '''
        trajectory = self.data.loc[trajectory_number]
        column_da = xr.DataArray.from_series(trajectory[data_column])

        # Retrieve name of dimension
        if len(column_da.dims) == 1:
            indexer_name = column_da.dims[0]
        else:
            raise ValueError("Trajectory data has too many dimensions: {}.\
                Expecting only 1 indexer like 'traj age' on a single\
                trajectory".format(column_da.dims))

        # Add extra columns as coordinates
        if isinstance(include_coords, str):
            include_coords = (include_coords, ) # make string an iterable

        if include_coords is not None:
            for column in include_coords:
                column_da = column_da.assign_coords({column: (indexer_name, trajectory[column])})
        return column_da

    def winter(self, out_format):
        '''
        Return year(s) corresponding to the winter in which these trajectories occurred

        Parameters
        ----------
        out_format: string
            output format; must be 'first' or 'first-second' or 'firstsecond'
            for the winter of 0009-0010:
                'first' -> '09'
                'first-second' -> '09-10'
                'firstsecond' -> '0910'

        Returns
        -------
        output: string
            year(s) corresponding to the winter in which the trajectories occurred
            format specified by out_format argument
        '''
        if any(self.data['month'] > 10):
            start_year = min(self.data['year'])
        else:
            start_year = min(self.data['year']) - 1
        end_year = start_year + 1

        if out_format == 'first':
            output = '{:02d}'.format(start_year)
        elif out_format == 'first-second':
            output = '{:02d}-{:02d}'.format(start_year, end_year)
        elif out_format == 'firstsecond':
            output = '{:02d}{:02d}'.format(start_year, end_year)
        else:
            raise ValueError(
                "Invalid out_format {}. Must be 'first' or 'first-second' or 'firstsecond'".format(out_format))
        return output


class WinterCAM:
    '''
    Store CAM4 output data corresponding to a set of trajectories

    Methods
    -------
    (None)

    Attributes
    ----------
    name_to_h (class attribute): dictionary
        maps CAM variable name string to corresponding file, denoted by string
        'h1', 'h2', 'h3', or 'h4'
    h_to_d: dictionary
        maps 'h1' through 'h4' to xarray Dataset containing the corresponding
        CAM data
    lat: numpy array of latitude coordinates
    lon: numpy array of longitude coordinates

    '''

    # Map CAM variable names to h1 through h4 files
    # name_to_h = dict.fromkeys(['CLDHGH', 'CLDICE', 'CLDLIQ', 'CLDLOW', 'CLDMED',
    #                            'CLDTOT', 'CLOUD', 'CONCLD', 'FICE', 'ICIMR', 'ICLDIWP', 'ICLDTWP', 'ICWMR',
    #                            'Q', 'QFLX', 'QREFHT', 'RELHUM', 'SFCLDICE', 'SFCLDLIQ', 'TGCLDCWP',
    #                            'TGCLDIWP', 'TGCLDLWP', 'TMQ'], 'h1')
    # name_to_h.update(dict.fromkeys(['FLDS', 'FLDSC', 'FLNS', 'FLNSC', 'FLNT',
    #                                 'FLNTC', 'FLUT', 'FLUTC', 'FSDS', 'FSDSC', 'FSDTOA', 'FSNS', 'FSNSC', 'FSNT',
    #                                 'FSNTC', 'FSNTOA', 'FSNTOAC', 'FSUTOA', 'LHFLX', 'LWCF', 'QRL', 'QRS', 'SHFLX',
    #                                 'SOLIN', 'SWCF'], 'h2'))
    # name_to_h.update(dict.fromkeys(['OMEGA', 'OMEGAT', 'PBLH', 'PHIS', 'PRECC',
    #                                 'PRECL', 'PRECT', 'PS', 'PSL', 'SNOWHICE', 'SNOWHLND', 'TAUX', 'TAUY', 'U',
    #                                 'U10', 'UU', 'V', 'VQ', 'VT', 'VU', 'VV', 'Z3'], 'h3'))
    # name_to_h.update(dict.fromkeys(['ICEFRAC', 'LANDFRAC', 'OCNFRAC', 'T', 'T200',
    #                                 'T500', 'T850', 'TREFHT', 'TREFHTMN', 'TREFHTMX', 'TS', 'TSMN', 'TSMX'], 'h4'))

    def __init__(self, file_dir, trajectories=None):
        '''
        Parameters
        ----------
        file_dir: string
            if trajectories is None:
                full path to a netCDF file
            if trajectories is not None:
                path to directory containing CAM files
                assumes CAM file names within file_dir are in the format
                'pi_3h_YYYY_h?.nc' where e.g. YYYY=0910 for the 0009-0010 winter
        trajectories: TrajectoryFile instance
            if None, then use file_dir as full path of netCDF file to load
            if not None, must be a family of trajectories that start at the same
                place and time. CAM files are loaded that correspond to the
                year(s) associated with these trajectories
        '''
        # Open the CAM files with xarray
        if trajectories is not None:
            # Read in h1, h2, h3, and h4 for the winter corresponding to trajectories
            winter_str = trajectories.winter(out_format='firstsecond')
            nc_file_path = os.path.join(
                file_dir, 'pi_3h_' + winter_str + '_h1.nc')
            ds1 = xr.open_dataset(nc_file_path)
            nc_file_path = os.path.join(
                file_dir, 'pi_3h_' + winter_str + '_h2.nc')
            ds2 = xr.open_dataset(nc_file_path)
            nc_file_path = os.path.join(
                file_dir, 'pi_3h_' + winter_str + '_h3.nc')
            ds3 = xr.open_dataset(nc_file_path)
            nc_file_path = os.path.join(
                file_dir, 'pi_3h_' + winter_str + '_h4.nc')
            ds4 = xr.open_dataset(nc_file_path)

            # Remove variables that should conflict between files before merging
            dropped_vars = ['time_written']
            ds1 = ds1.drop_vars(dropped_vars, errors='ignore')
            ds2 = ds2.drop_vars(dropped_vars, errors='ignore')
            ds3 = ds3.drop_vars(dropped_vars, errors='ignore')
            ds4 = ds4.drop_vars(dropped_vars, errors='ignore')

            dataset = xr.merge([ds1, ds2, ds3, ds4], join='exact')
        else:
            # Read in a single netCDF file
            dataset = xr.open_dataset(file_dir)

        # Add numerical time coordinate
        # NOTE: time units are an assumption; cannot retrieve the units
        # directly from the cftime.DatetimeNoLeap object
        self.time_units = 'days since 0001-01-01 00:00:00' # by assumption
        self.time_calendar = dataset.time.item(0).calendar
        numerical_times = cftime.date2num(dataset['time'], units=self.time_units, calendar=self.time_calendar)
        self.dataset = dataset.assign_coords({'numerical_time': ('time', numerical_times)})

    def variable(self, key):
        '''
        Return DataArray for the variable named 'key'
        '''
        return self.dataset[key]

    def interpolate(self, variable_da, pressure_levels, interpolation='linear', extrapolate=False, fill_value=np.nan):
        '''
        Interpolate data from CAM hybrid levels onto pressure levels

        Parameters
        ----------
        variable_da: xarray.DataArray
            data in three spatial dimensions + time, where the vertical spatial
            dimension 'lev' is CAM hybrid levels and runs from top-of-atmosphere
            to surface
        pressure_levels: array-like
            one-dimensional array of pressure levels, in Pa, to interpolate onto
            Can be either ascending or descending
        interpolation: string
            interpolation method. Options are 'linear', 'log', and 'log-log'.
            Default is 'linear'
        extrapolate: boolean
            if True, extrapolation is permitted even when a pressure level is
            beyond the range of the surface pressure.
            Default is False
        fill_value: float or NaN
            if extrapolate=True, replaces values outside of extrapolation range
            (where pressure_levels is beyond the range allowed by surface
            pressure)

        Returns
        -------
        variable_da_on_p_levels: xarray.DataArray
            variable data interpolated from hybrid levels onto pressure levels
            using PyNGL's vinth2p interpolation function.
            Has dimensions ('time', 'pres', 'lat', 'lon') where the coordinates
            of 'pres' were given in pressure_levels

        '''
        # Check nc files for required variables
        required_variables = ['P0', 'hyam', 'hybm', 'PS']
        for key in required_variables:
            if key not in self.dataset.data_vars:
                raise KeyError('WinterCAM file is missing variable {}, which is\
                    required for interpolation.'.format(key))

        # Check that variable has 3 spatial dimensions
        required_dims = ('time', 'lev', 'lat', 'lon')
        if variable_da.dims != required_dims:
            raise ValueError('Variable DataArray has dimensions {}; must have \
                dimensions {} for interpolation onto pressure levels'.format(
                    variable_da.dims, required_dims))

        # Determine range of variable data
        time_slice = variable_da.time
        lat_slice = variable_da.lat
        lon_slice = variable_da.lon

        # Extract hybrid level information from CAM files
        p_0_mb = self.variable('P0').values.item()/100 # p0 must be in mb
        hyam = self.variable('hyam').values
        hybm = self.variable('hybm').values
        if len(pressure_levels) < len(hyam):
            print('WARNING:    The number of pressure levels given, {}, is less \
                than the number of hybrid levels in the CAM data, {}. There may \
                be unnecessary loss of information when interpolating to \
                pressure levels as a result.'.format(len(pressure_levels), len(hyam)))

        # Set up interpolation arguments
        pressure_levels_mb = pressure_levels/100 # Ngl.vinth2p requires pressure levels in mb
        if interpolation == 'linear':
            interp_flag = 1
        elif interpolation == 'log':
            interp_flag = 2
        elif interpolation == 'log-log':
            interp_flag = 3
        else:
            raise ValueError('Invalid interpolation method {}. Interpolation must be linear, log, or log-log'.format(interpolation))
        p_surf = self.variable('PS').sel(time=time_slice, lat=lat_slice, lon=lon_slice).values
        
        # Interpolate onto pressure levels
        replacement_threshold = 1e29
        if (not extrapolate) and (np.any(variable_da.values >= replacement_threshold)):
            raise ValueError('Some values in the data array are larger than the\
                threshold used to identify and replace extrapolated data ({}).\
                Either set extrapolate=True or scale down the input array by a\
                few order of magnitude before interpolating'.format(
                    replacement_threshold))

        on_p_levels = Ngl.vinth2p(variable_da.values, hyam, hybm, pressure_levels_mb, p_surf, interp_flag, p_0_mb, 1, extrapolate)
        
        # If extrapolation is False, replace default fill value
        # (1e30 > replacement_threshold) with designated fill_value
        if not extrapolate:
            on_p_levels = np.where(on_p_levels >= replacement_threshold, fill_value, on_p_levels)
        
        # Bundle into a DataArray
        variable_da_on_p_levels = xr.DataArray(on_p_levels,
            name=variable_da.name,
            dims=('time', 'pres', 'lat', 'lon'),
            coords={'time': variable_da['time'].values,
                    'pres': pressure_levels,
                    'lat': variable_da['lat'].values,
                    'lon': variable_da['lon'].values},
            attrs=variable_da.attrs)
        return variable_da_on_p_levels


def subset_nc(filename, winter_idx, desired_variable_key, lat_bounds, lon_bounds, landfrac_min=0.9, testing=False):
    '''
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
    lat_bounds: array-like of floats
        lower and upper bounds of latitude range for the subset
        must be in the order (lower bound, upper bound)
        latitude subset = [lat_bounds[0], lat_bounds[1]]
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
            latitude=[lat_bounds[0], lat_bounds[1]],
            longitude=[lon_bounds[0], lon_bounds[1]] of the variable
            desired_variable_key, masked by landfraction so that any points with
            landfraction < landfrac_min have a value of np.nan
        'time': array of ordinal time values corresponding to subsetted time
            dimension
        'lat': array of latitudes corresponding to subsetted lat dimension
        'lon': array of longitudes corresponding to subsetted lon dimension
        if testing=True, also includes:
            'unmasked_data': same as 'data' but without the landfraction masking
    '''
    nc_file = Dataset(filename)
    variable_object = nc_file.variables[desired_variable_key]
    if variable_object.dimensions != ('time', 'lat', 'lon'):
        raise ValueError("Variable {} has dimensions {}; expecting dimensions ('time', 'lat', 'lon')".format(
            desired_variable_key, variable_object.dimensions))
    time_object = nc_file.variables['time']
    latitude_object = nc_file.variables['lat']
    longitude_object = nc_file.variables['lon']

    # time subset: define winter as Dec-Jan-Feb
    min_time = cftime.date2num(cftime.datetime(
        7 + winter_idx, 12, 1), time_object.units, calendar=time_object.calendar)
    if testing:
        max_time = cftime.date2num(cftime.datetime(
            7 + winter_idx, 12, 7), time_object.units, calendar=time_object.calendar)
    else:
        max_time = cftime.date2num(cftime.datetime(
            8 + winter_idx, 2, 28), time_object.units, calendar=time_object.calendar)

    # index slices for time, lat, and lon
    time_subset = slice_dim(nc_file, 'time', min_time, max_time)
    lat_subset = slice_dim(nc_file, 'lat', lat_bounds[0], lat_bounds[1])
    lon_subset = slice_dim(nc_file, 'lon', lon_bounds[0], lon_bounds[1])

    # subset data by time, lat, and lon
    datetime_min = cftime.num2date(
        min_time, time_object.units, calendar=time_object.calendar)
    datetime_max = cftime.num2date(
        max_time, time_object.units, calendar=time_object.calendar)
    print('Taking a subset in time and location of variable {}:'.format(
        desired_variable_key))
    print('    time: {:04d}-{:02d}-{:02d} to {:04d}-{:02d}-{:02d}'.format(datetime_min.year,
          datetime_min.month, datetime_min.day, datetime_max.year, datetime_max.month, datetime_max.day))
    print('    latitude: {:+.1f} to {:+.1f}'.format(lat_bounds[0], lat_bounds[1]))
    print(
        '    longitude: {:+.1f} to {:+.1f}'.format(lon_bounds[0], lon_bounds[1]))
    variable_subset = variable_object[time_subset, lat_subset, lon_subset].data

    # mask by landfraction
    # replace any value in variable_subset where landfraction < landfrac_min
    # with np.nan
    print('Masking {} by landfraction: np.nan anywhere landfraction < {:.2f}'.format(
        desired_variable_key, landfrac_min))
    landfrac_subset = nc_file.variables['LANDFRAC'][
        time_subset, lat_subset, lon_subset].data
    masked_variable = np.where(
        landfrac_subset >= landfrac_min, variable_subset, np.nan)
    subset_dict = {'data': masked_variable, 'time': time_object[time_subset].data, 'lat': latitude_object[
        lat_subset].data, 'lon': longitude_object[lon_subset].data}
    if testing:
        subset_dict['unmasked_data'] = variable_subset
    return subset_dict


def slice_dim(file, dimension_key, low_bound, upper_bound=np.inf, allow_reset=False):
    '''
    Given closed bounds [low_bound, upper_bound], return a slice object of the
    given dimension that spans the range low_bound <= dimension <= upper_bound.

    For example, if dim is the array of values in the dimension, then dim[slice]
    will return those values of dim in the closed interval
    [low_bound, upper_bound]. If var is a variable with the corresponding
    dimension, var[dimension], then var[slice] will return the values of var at
    locations where dim is in the closed interval [low_bound, upper_bound].

    If allow_reset is False, raises an error if any of the non-infinite bounds
    are outside the range of the dimension.

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
    allow_reset: boolean
        if True and lower [upper] bound is out of range, then replace lower
        [upper] bound with minimum [maximum] value of dimension instead
        if False and lower or upper bound is out of range (but not infinity),
        raise an error
        Default is False.

    Returns
    -------
    slice object spanning the closed interval [low_bound, upper_bound] of the
    dimension
    '''
    if not isinstance(file, Dataset):
        raise TypeError(
            'File argument must be an instance of the netCDF4 Dataset class; given type {}'.format(type(file)))
    else:
        dimension = file.variables[dimension_key][:].data

    # dimension is monotonically increasing:
    # True if dimension is monotonically increasing
    increasing = np.all(np.diff(dimension) > 0)
    if not increasing:
        raise ValueError(
            "NetCDF dimension '{}' must be monotonically increasing to produce valid index slices.".format(dimension_key))

    # low_bound < upper_bound:
    if not (low_bound < upper_bound):
        raise ValueError("Dimension slicing by index error for dimension {}:\n   lower bound on index slice ({:.4f}) must be less than upper bound ({:.4f})".format(
            dimension_key, low_bound, upper_bound))

    # bounds are within the dimension range for non-infinite bounds:
    if not np.isinf(low_bound) and not ((low_bound >= dimension[0]) and (low_bound <= dimension[-1])):
        if allow_reset:
            print("WARNING: Lower bound {:.2f} is out of range of dimension '{}'. Re-setting lower bound to minimum value of dimension, {:.2f}".format(
                low_bound, dimension_key, dimension[0]))
            low_bound = dimension[0]
        else:
            raise ValueError("Dimension slicing by index error for dimension {}:\n   lower bound on index slice ({:.2f}) should be within the range of the dimension, from {:.4f} to {:.4f} ".format(
                dimension_key, low_bound, dimension[0], dimension[-1]))
    if not np.isinf(upper_bound) and not ((upper_bound >= dimension[0]) and (upper_bound <= dimension[-1])):
        if allow_reset:
            print("WARNING: Upper bound {:.2f} is out of range of dimension '{}'. Re-setting upper bound to maximum value of dimension, {:.2f}".format(
                low_bound, dimension_key, dimension[-1]))
            upper_bound = dimension[-1]
        else:
            raise ValueError("Dimension slicing by index error for dimension {}:\n   upper bound on index slice ({:.2f}) should be within the range of the dimension, from {:.4f} to {:.4f} ".format(
                dimension_key, upper_bound, dimension[0], dimension[-1]))

    slice_idx_list = np.squeeze(np.where(np.logical_and(
        dimension >= low_bound, dimension <= upper_bound)))
    return slice(slice_idx_list[0], slice_idx_list[-1] + 1)


def make_CONTROL(event, event_ID, traj_heights, backtrack_time, output_dir, traj_dir, data_dir):
    '''
    Generate the CONTROL file that HYSPLIT uses to set up a backtracking run
    based on a time and location stored in 'event'

    Also creates two directories:
        output_dir: where CONTROL files are placed after creation
        traj_dir: where trajectory files will be placed when HYSPLIT is run with
            these CONTROL files

    CONTROL files will be named CONTROL_<event_ID>
    trajectory files will be named traj_event<event_ID>.traj

    Parameters
    ----------
    event: Pandas DataSeries
        contains time and lat/lon information for trajectory initialization.
        Entries must include:
            'time': time of event in days since 0001-01-01 on 'noleap' calendar
            'lat': latitude of event in degrees on -90 to 90 scale
            'lon': longitude of event in degrees on 0 to 360 scale
    event_ID: integer
        unique identifier for the event, used to make CONTROL file name:
        CONTROL_<event ID>
    traj_heights: array-like
        starting heights in meters for each trajectory
    backtrack_time: integer
        number of hours to follow each trajectory back in time
    output_dir: string
        output directory for CONTROL files
    traj_dir: string
        HYSPLIT directory to output trajectory files, which will be named:
        traj_event<event_ID>.traj
    data_dir: string
        path name of parent directory containing the data folder winter_YY-YY/
        and data file pi_3h_YYYY.arl
    '''
    event_ID = str(event_ID)
    # Set up file paths
    data_path = os.path.join(data_dir, 'winter_' + winter_string(event['time'], 'first-second'), '')  # with trailing slash
    data_filename = 'pi_3h_' + winter_string(event['time'], 'firstsecond') + '.arl'
    traj_dir = os.path.join(traj_dir, '') # add trailing slash if not already there
    control_path = os.path.join(output_dir, 'CONTROL_' + event_ID)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)

    # Write CONTROL file
    with open(control_path, 'w') as f:
        t = cftime.num2date(event['time'], 'days since 0001-01-01 00:00:00', calendar='noleap')
        lat = event['lat']
        if event['lon'] > 180:
            # HYSPLIT requires longitude on a -180 to 180 scale
            lon = event['lon'] - 360
        else:
            lon = event['lon']
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
        f.write('traj_event{:02d}.traj\n'.format(event_ID))

def winter_string(numerical_time, out_format):
    '''
    Return year(s) corresponding to the winter spanning the given time

    Parameters
    ----------
    numerical_time: float
        number of days since 0001-01-01 00:00:00 on a noleap calendar
    out_format: string
        output format; must be 'first' or 'first-second' or 'firstsecond'
        for the winter of 0009-0010:
            'first' -> '09'
            'first-second' -> '09-10'
            'firstsecond' -> '0910'

    Returns
    -------
    output: string
        year(s) corresponding to the winter that spans the given time. Output
        format specified by out_format argument
    '''
    date_time = cftime.num2date(numerical_time, 'days since 0001-01-01 00:00:00', calendar='noleap')
    if date_time.month > 6:
        start_year = date_time.year
    else:
        start_year = date_time.year - 1
    end_year = start_year + 1

    if out_format == 'first':
        output = '{:02d}'.format(start_year)
    elif out_format == 'first-second':
        output = '{:02d}-{:02d}'.format(start_year, end_year)
    elif out_format == 'firstsecond':
        output = '{:02d}{:02d}'.format(start_year, end_year)
    else:
        raise ValueError(
            "Invalid out_format {}. Must be 'first' or 'first-second' or 'firstsecond'".format(out_format))
    return output
