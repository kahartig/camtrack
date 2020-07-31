"""
Author: Kara Hartig

Handle data from HYSPLIT trajectories and CAM model output

Classes:
    TrajectoryFile:  read in and store data from HYSPLIT .traj file
    WinterCAM:  read in and store data from CAM4 winter (Nov-Feb) file
    
Functions:
    subset_and_mask:  subset a WinterCAM variable by time, lat, lon and mask by landfraction
    make_CONTROL:  print a CONTROL file for running HYSPLIT trajectories
    winter_string:  return years corresponding to the winter in which a given time falls

ASSUMPTIONS:
    CAM time units are days since 0001-01-01 00:00:00 and calendar is 'noleap'
"""

# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import cftime
import calendar
import Ngl
import scipy.interpolate as interpolate

# camtrack imports
import camtrack.assist as assist


class TrajectoryFile:
    '''
    Store of all data read in from a .traj file, an ASCII file
    output from HYSPLIT. These .traj files contain lat/lon locations of
    an air mass every hour along a trajectory.

    Methods
    -------
    get_trajectory
        return DataFrame of trajectory data for a specified trajectory number
        time interval (e.g. every 3 hours)
    col2da
        convert any data column into an xarray DataArray
    height2pressure
        convert height column to pressure, based on the calculation performed by
        HYSPLIT in prfecm.f
    winter
        return year(s) corresponding to this trajectory in user-specified format

    Attributes
    ----------
    grids: pandas DataFrame
        contains all grids used: model, year, month, day, hour, and fhour
    traj_start: pandas DataFrame
        contains initialization information for all trajectories: date, time,
        location, and height at which trajectory was started
    diag_var_names: list
        list of names of diagnostic output variables. Corresponding values along
        trajectories are in the trailing columns of the .traj file
    ntraj: int
        number of distinct trajectories stored in .traj file
    direction: string
        direction of trajectory calculation: 'FORWARD' or 'BACKWARD'
    hit_ground: dict
        trajectory number:boolean pairs indicating whether that trajectory
        contained any NaN values
            NOTE that hitting the ground does not always produce NaN values, so
            hit_ground is a very conservative measure
    data: pandas DataFrame
        trajectory data every 3 hours
        uses a MultiIndex:
            top level: int from 1 to ntraj
                trajectory number 'traj #'
            second level: int from 0 to -(total length of trajectory)
                age of trajectory 'traj age'
            EX: to access trajectory 3 at time -5 hours, use data.loc[3, -5]
        columns: grid #, year, month, day, hour, minute, fhour, lat, lon,
            height (m), <any diagnostic variables...>, datetime, cftime date,
            ordinal time
        note that there are three equivalent representations of time:
            datetime: str; 'YYYY-MM-DD HH:MM:SS'
            cftime date: cftime.DatetimeNoLeap()
            ordinal time: float; days since 0001-01-01 00:00:00
    data_1h: pandas DataFrame
        same as data, but every hour
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

        # skip over header; length depends on number of grids and trajectories
        traj_skiprow = 1 + ngrids + 1 + ntraj + 1

        # set up column names, dtype, widths
        traj_columns = ['traj #', 'grid #', 'year', 'month', 'day', 'hour',
                        'minute', 'fhour', 'traj age', 'lat', 'lon', 'height (m)']
        traj_dtypes = {'traj #': int, 'grid #': int, 'year': int, 'month': int, 'day': int, 'hour': int,
                       'minute': int, 'fhour': int, 'traj age': int, 'lat': str, 'lon': str, 'height (m)': str}
        col_widths = [6, 6, 6, 6, 6, 6, 6, 6, 8, 9, 9, 9]
        for var in self.diag_var_names:
            col_widths.append(9)
            traj_columns.append(var)
            traj_dtypes[var] = str

        # read in file in fixed-width format
        trajectories = pd.read_fwf(filepath, widths=col_widths, names=traj_columns, dtype=traj_dtypes,
                           skiprows=traj_skiprow).set_index(['traj #', 'traj age'])
        trajectories.sort_index(inplace=True)

        # if trajectories hit ground (starting with row of ***** or NaN):
        #    identify row(s) where trajectory hits the ground
        idx_grounded = trajectories[trajectories.lat == '*********'].index
        #    replace values in grounded row with np.nan
        float_columns = ['lat', 'lon', 'height (m)']
        for var in self.diag_var_names:
            float_columns.append(var)
        #    for each column that should become a float, replace value with NaN
        for col in float_columns:
            trajectories.loc[idx_grounded, col] = np.nan
            traj_dtypes[col] = float
        #    remove columns that have become indices before changing dtype
        del traj_dtypes['traj #']
        del traj_dtypes['traj age']
        trajectories = trajectories.astype(traj_dtypes)
        #    store record of which trajectories have NaN values/become grounded
        is_grounded = {}
        for traj_idx in range(1, self.ntraj + 1):
            is_grounded[traj_idx] = trajectories.loc[traj_idx].isnull().values.any()
        self.hit_ground = is_grounded
        # drop any rows with NaN (after trajectory becomes grounded)
        trajectories = trajectories.dropna('index', 'any')

        # convert longitudes from -180 to 180 to 0 to 360 scale for consistency with CAM files
        trajectories['lon'].mask(trajectories['lon'] < 0, trajectories['lon'] + 360, inplace=True)

        # convert units of PRESSURE from hPa to Pa
        if 'PRESSURE' in self.diag_var_names:
            trajectories['PRESSURE'] = 100. * trajectories['PRESSURE']

        # new column: datetime string
        def traj_datetime(row):
            return '00{:02.0f}-{:02.0f}-{:02.0f} {:02.0f}:{:02.0f}:00'.format(row['year'], row['month'], row['day'], row['hour'], row['minute'])
        trajectories['datetime'] = trajectories.apply(traj_datetime, axis=1)

        # new column: cftime Datetime objects
        def traj_cftimedate(row):
            return cftime.DatetimeNoLeap(row['year'], row['month'], row['day'], row['hour'], row['minute'])
        trajectories['cftime date'] = trajectories.apply(
            traj_cftimedate, axis=1)

        # new column: ordinal time (days since 0001-01-01 00:00:00)
        # min_time = cftime.date2num(cftime.datetime(7 + winter_idx, 12, 1),
        # time_object.units, calendar=time_object.calendar)
        def traj_numtime(row):
            return cftime.date2num(row['cftime date'], units='days since 0001-01-01 00:00:00', calendar='noleap')
        trajectories['ordinal time'] = trajectories.apply(
            traj_numtime, axis=1)

        # Store trajectories in increments of 1 hour and 3 hours
        # default self.data will be every 3 hours to match CAM output frequency
        self.data_1h = trajectories
        self.data = trajectories[trajectories['hour'] % 3 == 0] # every 3 hours

    def get_trajectory(self, trajectory_number, hourly_interval=None, age_interval=None):
        '''
        Return data from every hourly_interval hours or age_interval age for a
        single trajectory

        Must choose between hourly_interval and age_interval, as they are
        mutually exclusive methods for selecting trajectory data

        Parameters
        ----------
        trajectory_number: int
            which trajectory to retrieve
        hourly_interval: int or float
            retrieve trajectory values every hourly_interval hours, based on
            'hour' data column
        age_interval: int or float
            retrieve trajectory values every age_interval hours, based on
            'traj age' index

        Returns
        -------
        pandas DataFrame of trajectory data every X hours, where X is set by
        either hourly_interval or age_interval
        '''
        single_trajectory = self.data_1h.loc[trajectory_number]

        if (hourly_interval is not None) and (age_interval is None):
            interval_col = single_trajectory['hour']
            interval = hourly_interval
        elif (hourly_interval is None) and (age_interval is not None):
            interval_col = single_trajectory.index
            interval = age_interval
        else:
            raise ValueError("Must provide interval for retrieval either by hour-of-day or by trajectory age in hours")
        
        iseveryxhours = interval_col % interval == 0
        return single_trajectory[iseveryxhours]

    def col2da(self, trajectory_number, data_column, include_coords=None):
        '''
        Convert any trajectory data column into an xarray.DataArray with
        additional coordinate(s) given by other column(s)

        Dimension of resulting DataArray will be highest-order index in 
        trajectory data, self.data. If additional columns are requested in
        include_coords, they will share the same dimension.

        Parameters
        ----------
        trajectory_number: int
            number corresponding to specific trajectory of interest
            Trajectory data is retrieved with self.data.loc[trajectory_number]
        data_column: string
            key of data column to be converted to DataArray
        include_coords: string or list of strings
            keys for other columns to be included as additional coordinates in
            the DataArray produced
            Default is None: no additional coordinates

        Returns
        -------
        column_da: xarray.DataArray
            trajectory data as a DataArray. The indexing dimension is the
            highest-order index in TrajectoryFile.data, and additional
            coordinates can be added with include_coords
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

    def height2pressure(self, cam_dir, trajectory_number, height_key='height (m)'):
        '''
        Return 3-hourly trajectory with new column 'pressure' of pressure levels
        along trajectory path

        Based on the calculation performed by HYSPLIT in prfecm.f
        NOTE this function is not exact; it should depend on which variables
        were provided to HYSPLIT when calculating trajectory, model height, etc.

        Parameters
        ----------
        cam_dir: string or path-like
            path to netCDF files containing climate variables used to calculate
            this trajectory in HYSPLIT
                required variables: 'PS', 'T', 'Q'
        trajectory_number: int
            number corresponding to specific trajectory of interest
        height_key: string
            name of column containing trajectory heights in meters

        Returns
        -------
        trajectory: pandas.DataFrame
            a copy of the requested trajectory data every 3 hours, but with new
            column 'pressure' of corresponding pressure level in Pa
        '''
        print("DEPRECATED height2pressure:  Use 'PRESSURE' column of trajectory data instead")
        
        winter_file = WinterCAM(cam_dir, self)
        trajectory = self.get_trajectory(trajectory_number, 3)
        pressures_from_h = []
        for age,point in trajectory.iterrows():
            t_time = point['cftime date']
            t_lat = point['lat']
            t_lon = point['lon']
            
            # Retrieve surface pressure and temperature at point
            p_surf_col = winter_file.variable('PS').sel(time=t_time, lat=t_lat, lon=t_lon, method='nearest').values.item()
            temp_surf_col = winter_file.variable('TREFHT').sel(time=t_time, lat=t_lat, lon=t_lon, method='nearest').values.item()
            
            # Retrieve vertical column of temperature, specific humidity at point
            T_col = winter_file.variable('T').sel(time=t_time, lat=t_lat, lon=t_lon, method='nearest')
            Q_col = winter_file.variable('Q').sel(time=t_time, lat=t_lat, lon=t_lon, method='nearest')
            
            # Replace hybrid level coordinate 'lev' with pressure levels 'pres'
            p0 = winter_file.variable('P0').values.item()
            hyam = winter_file.variable('hyam').values
            hybm = winter_file.variable('hybm').values
            column_data = xr.Dataset({'T': T_col, 'Q': Q_col}).assign_coords({'pres': ('lev', p0*hyam + p_surf_col*hybm)}).swap_dims({'lev': 'pres'})
            column_data = column_data.sortby('pres', ascending=False) # sort surface -> TOA

            # Re-create HYSPLIT's method for converting from pressure to height
            # REF: prfecm.f
            # initialize DataArray of heights of each data level
            level_heights = xr.full_like(column_data.pres, 0)
            # constants
            RDRY = 287.04
            GRAV = 9.80616
            ZMDL = 10000 # model height in m; provided to HYSPLIT in CONTROL file
            # at bottom of column:
            ZBOT = 0
            P0 = p_surf_col # surface pressure
            TBOT = temp_surf_col # surface temp; should be TREFHT, if provided to HYSPLIT
            TVBOT = TBOT * (1 + 0.61*column_data['Q'][0])
            PBOT = np.log(P0)
            # loop over data values:
            for KZ,p in enumerate(column_data.pres):
                # set values for top of layer
                PTOP = np.log(p)
                TTOP = column_data['T'][KZ]
                TVTOP = TTOP * (1.0 + 0.61*column_data['Q'][KZ])
                # get height at top of layer
                TVBAR = 0.5 * (TVTOP + TVBOT)
                DELZ = (PBOT - PTOP) * RDRY * TVBAR / GRAV
                ZTOP = ZBOT + DELZ
                level_heights[KZ] = ZTOP
                # re-set bottom values to current top values
                PBOT=PTOP
                ZBOT=ZTOP
                TBOT=TTOP
                TVBOT=TVTOP
            # add surface values to pressure and height mapping arrays
            h_surf_da = xr.DataArray([0], [('pres', [p_surf_col])])
            heights = xr.concat([h_surf_da, level_heights.drop_vars('lev')], dim='pres')
            pressures = heights.pres

            # Generate function to map from height to pressure
            h2p = interpolate.interp1d(heights, pressures)
            pressures_from_h.append(h2p(point[height_key]).item())
        trajectory['pressure'] = pressures_from_h
        return trajectory

    def winter(self, out_format):
        '''
        Return year(s) corresponding to the winter in which the trajectories occurred

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
    variable
        retrieve a variable by name from WinterCAM.dataset
    interpolate
        interpolate the given DataArray onto pressure levels

    Attributes
    ----------
    directory: string
        if a TrajectoryFile instance or winter is given during init:
            path to directory containing CAM files
        else:
            full path to a netCDF file
    time_units: string
        units for CAM file time dimension (ordinal days)
        'days since 0001-01-01 00:00:00'
    time_calendar: string
        calendar type for CAM file time dimension
        retrieved directly from dataset
    dataset: xarray.Dataset
        contains all variables from CAM file(s) indicated in init
        if trajectories or winter is not None, merges h1, h2, h3, and h4 files
        into a single dataset
    '''

    def __init__(self, file_dir, trajectories=None, winter=None):
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
            mutually exclusive with winter
        winter: string
            indicates which winter to pull h1 through h4 files for
                e.g. for 0009-0010 winter, winter='0910'
            mutually exclusive with trajectories
        '''
        # Store source directory
        self.directory = file_dir
        
        # Open the CAM files with xarray
        if (trajectories is None) and (winter is None):
            # Read in a single netCDF file
            dataset = xr.open_dataset(file_dir)
        else:
            if trajectories is not None:
                winter_str = trajectories.winter(out_format='firstsecond')
            elif winter is not None:
                winter_str = winter
            else:
                raise ValueError('Trajectories and winter arguments are mutually exclusive, only one or the other can be provided')
            # Read in h1, h2, h3, and h4 for the winter corresponding to trajectories
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
            dropped_vars = ['time_written', 'date_written']
            ds1 = ds1.drop_vars(dropped_vars, errors='ignore')
            ds2 = ds2.drop_vars(dropped_vars, errors='ignore')
            ds3 = ds3.drop_vars(dropped_vars, errors='ignore')
            ds4 = ds4.drop_vars(dropped_vars, errors='ignore')
            dataset = xr.merge([ds1, ds2, ds3, ds4], join='exact')

        # Add ordinal time coordinate
        # NOTE: time units are an assumption; cannot retrieve the units
        # directly from the cftime.DatetimeNoLeap object
        self.time_units = 'days since 0001-01-01 00:00:00' # by assumption
        self.time_calendar = dataset.time.item(0).calendar
        ordinal_times = cftime.date2num(dataset['time'], units=self.time_units, calendar=self.time_calendar)
        self.dataset = dataset.assign_coords({'ordinal_time': ('time', ordinal_times)})

    def variable(self, key):
        '''
        Return DataArray for the variable named 'key'

        To see the list of all available variable keys, use
        self.dataset.data_vars
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

        # Determine whether input variable duplicated lon=0 as lon=360
        roll_lon = max(lon_slice) > max(self.variable('lon').values)

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
        if roll_lon:
            reduced_p_surf = self.variable('PS').sel(time=time_slice)
            rolled_p_surf = assist.roll_longitude(reduced_p_surf)
            p_surf = rolled_p_surf.sel(lat=lat_slice, lon=lon_slice).values
        else:
            p_surf = self.variable('PS').sel(time=time_slice, lat=lat_slice, lon=lon_slice).values
        
        # Interpolate onto pressure levels
        replacement_threshold = 1e29
        if (not extrapolate) and (np.any(variable_da.values >= replacement_threshold)):
            raise ValueError('Some values in the data array are larger than the\
                threshold used to identify and replace extrapolated data ({}).\
                Either set extrapolate=True or scale down the input array by a\
                few order of magnitude before interpolating'.format(
                    replacement_threshold))

        # If interpolating temperature, add on surface temperatures
        variable_values = variable_da.values
        if variable_da.name == 'T':
            hyam_surf = 0.
            hybm_surf = 0.999
            hyam = np.append(self.variable('hyam').values, hyam_surf)
            hybm = np.append(self.variable('hybm').values, hybm_surf)
            # Add singleton 'lev' dimension to TREFHT (surface temperature)
            if roll_lon:
                reduced_T_surf = self.variable('TREFHT').sel(time=time_slice)
                rolled_T_surf = assist.roll_longitude(reduced_T_surf)
                T_surf = rolled_T_surf.sel(lat=lat_slice, lon=lon_slice)
            else:
                T_surf = self.variable('TREFHT').sel(time=time_slice, lat=lat_slice, lon=lon_slice)
            T_surf_newcoord = T_surf.assign_coords(lev=1000*(hyam_surf + hybm_surf))
            T_surf_expanded = T_surf_newcoord.expand_dims('lev')
            # Concatenate TREFHT onto T
            variable_values = xr.concat([variable_da, T_surf_expanded], dim='lev').values

        on_p_levels = Ngl.vinth2p(variable_values, hyam, hybm, pressure_levels_mb, p_surf, interp_flag, p_0_mb, 1, extrapolate)
        
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


def subset_and_mask(winter_file, variable_key, time_bounds, lat_bounds, lon_bounds, mask_by='LANDFRAC', mask_threshold=0.9):
    '''
    Return variable after subsetting by time, latitude, and longitude and
    masking by another variable.

    When masking, values from variable_key are replaced by np.nan wherever the
    mask_by variable is less than the mask_threshold

    Parameters
    ----------
    winter_file: WinterCAM instance
        the CAM data file from which variable_key and mask_by will be pulled
    variable_key: strinng
        name of variable to subset and mask
    time_bounds: list-like
        (minimum, maximum) times to subset to
        the bounds are inclusive
        for compatibility with xarray.DataArray.sel, must be in one of the
        following formats:
            'YYYY-MM-DD' (will default to HH:MM:SS = '00:00:00')
            cftime.DatetimeNoLeap()
    lat_bounds: list-like
        (minimum, maximum) latitudes to subset to
        the bounds are inclusive
    lon_bounds: list-like
        (minimum, maximum) longitudes to subset to
        the bounds are inclusive
    mask_by: string
        name of variable to mask variable_key by
        Default is 'LANDFRAC'
            landfrac = 1.0 is fully land-covered
            landfrac = 0.0 is fully ocean-covered
    mask_threshold: float
        wherever the mask_by variable values are less than mask_threshold,
        replace values in variable_key by np.nan

    Returns
    -------
    masked_variable: xarray.DataArray
        variable_key from winter_file, subset onto a region defined by time,
        latitude, and longitude bounds and masked by the variable mask_by
    '''
    time_slice = slice(time_bounds[0], time_bounds[1])
    lat_slice = slice(lat_bounds[0], lat_bounds[1])
    lon_slice = slice(lon_bounds[0], lon_bounds[1])
    subset_variable = winter_file.variable(variable_key).sel(time=time_slice, lat=lat_slice, lon=lon_slice)
    subset_mask = winter_file.variable(mask_by).sel(time=time_slice, lat=lat_slice, lon=lon_slice)
    masked_variable = subset_variable.where(subset_mask > mask_threshold, np.nan)
    return masked_variable


def make_CONTROL(event, event_ID, traj_heights, backtrack_time, output_dir, traj_dir, data_dir, case_name):
    '''
    Generate the CONTROL file that HYSPLIT uses to set up a backtracking run
    based on a time and location stored in 'event'

    Also creates two directories:
        output_dir: where CONTROL files are placed after creation
        traj_dir: where trajectory files will be placed when HYSPLIT is run with
            these CONTROL files

    CONTROL files will be named CONTROL_<event_ID>
    trajectory files will be named traj_event<event_ID>.traj
    Assumes that input data files are named case_name+'_YYYY.arl' where e.g.
        YYYY=0910 for the 0009-0010 winter

    Parameters
    ----------
    event: pandas.DataSeries
        contains time and lat/lon information for trajectory initialization.
        Entries must include:
            'time': time of event in days since 0001-01-01 on 'noleap' calendar
            'lat': latitude of event in degrees on -90 to 90 scale
            'lon': longitude of event in degrees on 0 to 360 scale
    event_ID: int
        unique identifier for the event, used to make CONTROL file name:
        CONTROL_<event ID>
    traj_heights: array-like
        starting heights in meters for each trajectory
    backtrack_time: int
        number of hours to follow each trajectory back in time
    output_dir: string
        output directory for CONTROL files
    traj_dir: string
        HYSPLIT directory to output trajectory files, which will be named:
        traj_event<event_ID>.traj
    data_dir: string
        path name of parent directory containing the binary format data file
        case_name+'_YYYY.arl'
    case_name: string
        case name of CESM run and prefix of .nc and .arl data files
    '''
    # Set up file paths
    data_path = os.path.join(data_dir, '')  # with trailing slash
    data_filename = case_name + '_' + winter_string(event['time'], 'firstsecond') + '.arl'
    traj_dir = os.path.join(traj_dir, '') # add trailing slash if not already there
    control_path = os.path.join(output_dir, 'CONTROL_' + str(event_ID))
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

def winter_string(ordinal_time, out_format):
    '''
    Return year(s) corresponding to the winter spanning the given time

    Parameters
    ----------
    ordinal_time: float
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
    date_time = cftime.num2date(ordinal_time, 'days since 0001-01-01 00:00:00', calendar='noleap')
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
