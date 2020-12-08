"""
Author: Kara Hartig

Extract climate variables from CAM along trajectories from HYSPLIT

Classes:
    ClimateAlongTrajectory: store CAM variables interpolated along trajectory paths

"""

# Standard imports
import numpy as np
import xarray as xr
import pandas as pd
import os
import cftime
import calendar

# NCL/NGL imports
import Ngl

# Scipy imports
from scipy.interpolate import griddata

# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# camtrack imports
import camtrack.assist as assist


class ClimateAlongTrajectory:
    '''
    Store climate variables along given air parcel trajectory

    Methods
    -------
    add_variable
        interpolate a new variable along trajectory path and add to self.data
    setup_pinterp
        one-time set up of interpolation onto pressure levels
    check_variable_exists
        raise error if a required variable does not exist in WinterCAM instance

    Attributes
    ----------
    winter_file: WinterCAM instance
        CAM file(s) used in init
    traj_file: TrajectoryFile instance
        trajectorie file used in init
    traj_interpolation: string
        method of interpolation onto trajectory path
    traj_number: int
        number corresponding to trajectory stored
    traj_time: xarray.DataArray
        trajectory times every 3 hours, indexed by dimension 'time'
    traj_lat: xarray.DataArray
        trajectory latitudes every 3 hours, indexed by dimension 'time'
    traj_lon: xarray.DataArray
        trajectory longitudes every 3 hours, indexed by dimension 'time'
    subset_time: slice
        slice used to subset variables onto range of trajectory times
    subset_lat: slice
        slice used to subset variables onto range of trajectory latitudes
    subset_lon: slice
        slice used to subset variables onto range of trajectory longitudes
    direction: string
        direction of trajectory calculation: 'FORWARD' or 'BACKWARD'
    trajectory: pandas.DataFrame
        3-hourly points along the trajectory, indexed by trajectory age
        output frequency matches that of CAM file
    data: xarray Dataset
        values of 2-D and 3-D variables along the trajectory. Dimensions are
        'time' and possibly 'pres' (if there are any 3-D variables). 2-D
        variables have 'time' dimension and 3-D have 'time' and 'pres'. Lat and
        lon are included as coordinate arrays with dimension 'time' and reflect
        trajectory path, or nearest-neighbor equivalent on CAM coordinates
    '''

    def __init__(self, winter_file, trajectories, trajectory_number, variables, traj_interpolation):
        '''
        Parameters
        ----------
        winter_file: WinterCAM instance
            winter CAM file corresponding to trajectories
        trajectories: TrajectoryFile instance
            contains trajectory of interest
        trajectory_number: integer
            number corresponding to trajectory of interest; the trajectory data
            is retrieved with trajectories.data.loc[trajectory_number]
        variables: list of strings
            list of variables that will be stored
            Must be a CAM variable name (field in all caps) OR one of two types
            of special variables:
                ends in '_1D': 3-D+time variable to be interpolated directly
                    onto trajectory path
                ends in '_hc': hard-coded variable with special handling
                    Available options:
                    'LWP_hc': liquid water path (vertical integral of 'Q')
                    'THETA_hc': potential temperature
        traj_interpolation: 'nearest' or 'linear'
            interpolation method for matching trajectory lat-lon to CAM variables
        pressure_levels: array-like of floats
            pressure levels, in Pa, to interpolate onto for variables with a vertical level coordinate
            Not required if none of the variables have a vertical dimension
            Default is None
        '''
        # Store initial attributes
        self.winter_file = winter_file
        self.traj_file = trajectories
        if (traj_interpolation == 'nearest') or (traj_interpolation == 'linear'):
            self.traj_interpolation = traj_interpolation
        else:
            raise ValueError("Invalid interpolation method onto trajectory '{}'. Must be 'nearest' or 'linear'".format(traj_interpolation))
        self.traj_number = trajectory_number

        # Check that all required variables are present in CAM file
        for key in variables:
            self.check_variable_exists(key)

        # Select a single trajectory
        self.trajectory = trajectories.get_trajectory(trajectory_number, 3)
        self.direction = trajectories.direction

        # Retrieve time, lat, lon along trajectory
        time_coord = {'time': self.trajectory['cftime date'].values}
        self.traj_time = xr.DataArray(time_coord['time'], dims=('time'), coords=time_coord)
        self.traj_lat = xr.DataArray(self.trajectory['lat'].values, dims=('time'), coords=time_coord)
        self.traj_lon = xr.DataArray(self.trajectory['lon'].values, dims=('time'), coords=time_coord)
        self.traj_pres = xr.DataArray(self.trajectory['PRESSURE'].values, dims=('time'), coords=time_coord)
        
        # Store height and diagnostic output variables
        diagnostic_variables = []
        height_attrs = {'units': 'm above ground level', 'long_name': 'Parcel height above ground level'}
        diagnostic_variables.append(xr.DataArray(self.trajectory['height (m)'].values, name='HEIGHT', attrs=height_attrs, dims=('time'), coords=time_coord))
        for key in trajectories.diag_var_names:
            if key == 'PRESSURE':
                key_attrs = {'units': 'Pa', 'long_name': 'pressure along trajectory (HYSPLIT)'}
            elif key == 'AIR_TEMP':
                key_attrs = {'units': 'K', 'long_name': 'air temperature from along trajectory (HYSPLIT)'}
            elif key == 'TERR_MSL':
                key_attrs = {'units': 'm', 'long_name': 'underlying terrain height relative to mean sea level'}
            else:
                key_attrs = {'units': 'unknown', 'long_name': key + ' from HYSPLIT diagnostic variables'}
            diagnostic_variables.append(xr.DataArray(self.trajectory[key].values, name=key, attrs=key_attrs, dims=('time'), coords=time_coord))
        self.data = xr.merge(diagnostic_variables)

        # Status of setup for pressure interpolation
        self.ready_pinterp = False

        # Store requested climate variables
        for key in variables:
            self.add_variable(key)


    def add_variable(self, variable_key):
        '''
        Interpolate a new variable onto trajectory path

        The new variable is automatically added to self.data
        For 3-D+time variables:
            to interpolate onto time, lat, and lon but retain entire vertical
              column, provide variable name e.g. 'OMEGA'
            to interpolate onto pressure as well as time, lat, and lon
              (collapse vertical dimension), add '_1D' to end of variable name
              e.g. 'OMEGA_1D'

        Parameters
        ----------
        variable: string
            name of variable to be interpolated onto trajectory path
            Must be a CAM variable name (field in all caps) OR one of two types
            of special variables:
                ends in '_1D': 3-D+time variable to be interpolated directly
                    onto trajectory path, e.g. 'T_1D' or 'OMEGA_1D'
                ends in '_hc': hard-coded variable with special handling
                    Available options:
                        'LWP_hc': liquid water path (vertical integral of 'Q')
                        'THETA_hc': potential temperature
                        'THETADEV_hc': potential temperature anomaly from
                                       time-average
        pressure_levels: array-like
            pressure levels, in Pa, to interpolate onto, if a 3-D+time variable
            is requested
            Set to None if variable is 2-D+time
            If self.pressure_levels already exists (for example, was set during
            init), the pressure_levels argument is ignored and the existing
            attribute used instead
            Default is None
        '''
        # Check if required variables are present in CAM file
        self.check_variable_exists(variable_key)

        # Identify if variable requires special handling
        to_1D = False
        hardcoded = False
        if '_' in variable_key:
            prefix, suffix = variable_key.split('_', 1)
            if suffix == '1D':
                to_1D = True
                variable = prefix
            elif suffix == 'hc':
                hardcoded = True
                if prefix == 'LWP':
                    #variable = 'Q'
                    raise NotImplementedError('LWP no longer supported')
                elif prefix == 'THETA':
                    variable = 'PS' # filler; unused
                elif prefix == 'THETADEV':
                    variable = 'PS' # filler; unused
                else:
                    raise ValueError('Invalid variable key for a hard-coded variable {}. Check docs for ClimateAlongTrajectory for a list of valid hard-coded variables'.format(variable_key))
            else:
                raise ValueError("Invalid suffix {} for variable key {}; must be '1D' or 'hc'".format(suffix, variable_key))
        else:
            variable = variable_key

        # Load requested variable
        raw_data = self.winter_file.variable(variable)

        if hardcoded:
            if prefix == 'THETA':
                p_0 = 1e5 # reference pressure 1,000 hPa
                kappa = 2./7. # Poisson constant
                T_values = self.traj_file.col2da(self.traj_number, 'AIR_TEMP', include_coords='cftime date').swap_dims({'traj age': 'cftime date'}).rename({'cftime date': 'time'})
                p_values = self.traj_pres
                values = T_values * (p_0 / p_values)**kappa
                values.name = variable_key
                values = values.assign_attrs({'units': 'K', 'long_name': 'Potential temperature'})
                variable_name = variable_key
            elif prefix == 'THETADEV':
                p_0 = 1e5 # reference pressure 1,000 hPa
                kappa = 2./7. # Poisson constant
                T_values = self.traj_file.col2da(self.traj_number, 'AIR_TEMP', include_coords='cftime date').swap_dims({'traj age': 'cftime date'}).rename({'cftime date': 'time'})
                p_values = self.traj_pres
                values = T_values * (p_0 / p_values)**kappa
                values = values - values.mean(dim='time')
                values.name = variable_key
                values = values.assign_attrs({'units': 'K', 'long_name': 'Potential temperature anomaly from time-avg'})
                variable_name = variable_key

        # Two-dimensional climate variables
        elif raw_data.dims == ('time', 'lat', 'lon'):
            variable_name = variable
            values = np.zeros(len(self.trajectory))
            
            t_idx = 0
            for age, point in self.trajectory.iterrows():
                time = point['cftime date']
                time_array = np.array([time, ])
                variable_at_time = raw_data.sel(time=time)

                # Account for periodicity in lon by duplicating lon=0 as lon=360, if necessary
                if point['lon'] > max(raw_data['lon'].values):
                    variable_at_time = assist.roll_longitude(variable_at_time)
                
                values[t_idx] = variable_at_time.interp(lat=point['lat'], lon=point['lon'], method=self.traj_interpolation, kwargs={'bounds_error': True})
                t_idx = t_idx + 1
            # Bundle into DataArray
            xr.DataArray(values, name = variable,
                dims=('time',),
                coords={'time': self.traj_time, 'lat': self.traj_lat, 'lon': self.traj_lon},
                attrs=raw_data.attrs)

        # Three-dimensional climate variables
        elif raw_data.dims == ('time', 'lev', 'lat', 'lon'):
            if not to_1D:
                raise NotImplementedError("Interpolating 3-D variables only onto time, lat, and lon has not been implemented; must add '_1D' to end of variable name and interpolate onto pressure as well")
            
            variable_name = variable
            values = np.zeros(len(self.trajectory))
            
            t_idx = 0
            for age, point in self.trajectory.iterrows():
                time = point['cftime date']
                time_array = np.array([time, ])
                variable_at_time = raw_data.sel(time=time)

                # Account for periodicity in lon by duplicating lon=0 as lon=360, if necessary
                if point['lon'] > max(raw_data['lon'].values):
                    variable_at_time = assist.roll_longitude(variable_at_time)

                # Set up for vertical interpolation if it has never been done before
                if not self.ready_pinterp:
                    self.setup_pinterp()

                # Pressure level
                pressure_array = np.array([point['PRESSURE'], ])
                if point['PRESSURE'] > self.lowest_model_pressure.sel(time=time):
                    # point is below lowest data level (higher pressure)
                    values[t_idx] = np.nan
                else:
                    # Lat/lon subset
                    subset_lat = slice(point['lat'] - self.lat_pad, point['lat'] + self.lat_pad)
                    if (this_lon < self.lon_pad) or (this_lon > (360 - self.lon_pad)):
                        # trajectory path crosses meridian -> include full longitude range
                        subset_lon = slice(0, 360)
                    else:
                        subset_lon = slice(point['lon'] - self.lon_pad, point['lon'] + self.lon_pad)
                    # Interpolate onto pressure level
                    raw_local_data = variable_at_time.sel(lat=subset_lat, lon=subset_lon).expand_dims({'time': time_array})
                    local_data = self.winter_file.interpolate(raw_local_data, pressure_array, interpolation=self.pres_interpolation, extrapolate=self.pres_extrapolate, fill_value=self.fill_value)
                    variable_data = local_data.squeeze('time').squeeze('pres')
                    # Mask NaNs before interpolation
                    variable_masked = np.ma.masked_invalid(variable_data.values)
                    xx, yy = np.meshgrid(variable_data.lon.values, variable_data.lat.values)
                    x1 = xx[~variable_masked.mask]
                    y1 = yy[~variable_masked.mask]
                    variable_valid = variable_masked[~variable_masked.mask]
                    # Interpolate onto traj lat/lon
                    values[t_idx] = griddata((x1, y1), variable_valid.ravel(), (point['lon'], point['lat']), method=self.traj_interpolation)
                t_idx = t_idx + 1
            # Bundle into DataArray
            xr.DataArray(values, name = variable,
                dims=('time',),
                coords={'time': self.traj_time, 'pres': self.traj_pres, 'lat': self.traj_lat, 'lon': self.traj_lon},
                attrs=raw_data.attrs)

        else:
            raise ValueError('The requested variable {} has unexpected dimensions {}. Dimensions must be (time, lat, lon) or (time, lev, lat, lon)'.format(variable, variable_data.dims))
        
        # Update Dataset with new DataArray
        self.data[variable_name] = values

    def setup_pinterp(self):
        '''
        One-time setup of interpolation onto pressure levels

        If 3-D+time variables were not requested on init, this function will be
        called by add_variable the first time a 3-D+time variable is added
        '''
        # Subset pad in lat and lon
        self.lat_pad = 7 * max(abs(np.diff(self.winter_file.variable('lat'))))
        self.lon_pad = 7 * max(abs(np.diff(self.winter_file.variable('lon'))))

        # Store pressure of surface and lowest model level
        self.add_variable('PS') # always store surface pressure
        P0 = self.winter_file.variable('P0').item()
        hyam = self.winter_file.variable('hyam')
        hybm = self.winter_file.variable('hybm')
        self.lowest_model_pressure = (hyam * P0 + hybm * self.data['PS']).isel(lev=-1)

        # Set arguments for Ngl.vinth2p
        self.pres_interpolation = self.traj_interpolation  # for Ngl.vinth2p; options=linear, log, log-log
        self.pres_extrapolate = False  # for Ngl.vinth2p
        self.fill_value = np.nan  # for winter_file.interpolate

        # Store status: ready to interpolate in pressure
        self.ready_pinterp = True
        
    def check_variable_exists(self, variable_key):
        '''
        Raise error if CAM variable corresponding to variable_key does not exist
        in WinterCAM file

        If variable_key corresponds to a hard-coded variable, check that the
        variable(s) required to calculate variable_key are present in WinterCAM

        Parameters
        ----------
        variable_key: string
            variable name to be checked against self.winter_file data variables
        '''
        # Map hard-coded variable names to the CAM variables they require
        #  note that THETA and THETADEV don't actually require PS, but including
        #  check simplifies handling in add_variable
        hc_requires = {'LWP_hc': 'Q', 'THETA_hc': 'PS', 'THETADEV_hc': 'PS'}

        # Standard CAM variable
        if '_' not in variable_key:
            if variable_key not in self.winter_file.dataset.data_vars:
                raise ValueError('CAM output files are missing a requested variable {}'.format(variable_key))
        else:
            # Interpolated to 1-D
            if '_1D' in variable_key:
                prefix, suffix = variable_key.split('_', 1)
                if prefix not in self.winter_file.dataset.data_vars:
                    raise ValueError("CAM output files are missing a variable to be interpolated to 1D {}".format(variable_key))
            # Hard-coded
            elif variable_key in hc_requires.keys():
                if hc_requires[variable_key] not in self.winter_file.dataset.data_vars:
                    raise ValueError("CAM output files are missing the variable {} which is required by the requested hard-coded variable {}".format(hc_requires[variable_key], variable_key))
            # Invalid
            else:
                raise ValueError('Invalid variable key {}. See WinterCAM.dataset.data_vars for list of allowed CAM variables and ClimateAlongTrajectory docs for valid hard-coded variables'.format(variable_key))
