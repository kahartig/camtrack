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

    def __init__(self, winter_file, trajectories, trajectory_number, variables, traj_interpolation, pressure_levels=None):
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
                ends in '_hc': variable with special handling that has been
                    hard-coded in camtrack. Available options:
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

        # Set up subset to trajectory path        
        lat_pad = 1.5 * max(abs(np.diff(self.winter_file.variable('lat'))))
        lon_pad = 1.5 * max(abs(np.diff(self.winter_file.variable('lon'))))
        self.subset_lat = slice(min(self.traj_lat.values) - lat_pad, max(self.traj_lat.values) + lat_pad)
        self.subset_time = slice(min(self.traj_time.values), max(self.traj_time.values))
        #   special instructions for longitude because it is periodic:
        if (any(self.traj_lon < 10)) and (any(self.traj_lon > 350)):
            # trajectory path crosses meridian -> include full longitude range
            self.subset_lon = slice(0, 360)
        else:
            self.subset_lon = slice(min(self.traj_lon.values) - lon_pad, max(self.traj_lon.values) + lon_pad)
        
        # Store height and diagnostic output variables
        diagnostic_variables = []
        height_attrs = {'units': 'm above ground level', 'long_name': 'Parcel height above ground level'}
        diagnostic_variables.append(xr.DataArray(self.trajectory['height (m)'].values, name='HEIGHT', attrs=height_attrs, dims=('time'), coords=time_coord))
        for key in trajectories.diag_var_names:
            key_attrs = {'units': 'unknown', 'long_name': key + ' from HYSPLIT diagnostic variables'}
            diagnostic_variables.append(xr.DataArray(self.trajectory[key].values, name=key, attrs=key_attrs, dims=('time'), coords=time_coord))
        self.data = xr.merge(diagnostic_variables)        

        # Store requested climate variables
        for key in variables:
            self.add_variable(key, pressure_levels=pressure_levels)


    def add_variable(self, variable_key, pressure_levels=None):
        '''
        Interpolate a new variable onto trajectory path

        The new variable is automatically added to self.data
        If the variable is 3-D + time (time, lev, lat, lon), there are two
        interpolation options:
            if to_1D=False, retrieve full air column along trajectory path
                output dimensions are (time, pres)
            if to_1D=True, also interpolate onto the trajectory pressure
                output dimensions are (time)

        Special treatment for liquid water path:
            if variable='LWP', return vertical integral of Q along trajectory

        Parameters
        ----------
        variable: string
            name of variable to be interpolated onto trajectory path
        to_1D: boolean
            determines whether a 3-D+time variable is interpolated onto
            trajectory height/pressure as well as latitude and longitude
            If False:
                retrieve full air column along trajectory path
                For 2-D+time variables, output dimensions are (time)
                For 3-D+time variables, output dimensions are (time, pres)
            If True and variable has dimensions (time, lev, lat, lon):
                interpolate onto pressure as well as the usual time, lat, lon of
                trajectory path
                Output dimensions are (time)
            Default is False
        pressure_levels: array-like
            pressure levels, in Pa, to interpolate onto, if a 3-D+time variable
            is requested
            Set to None if variable is 2-D+time
            Only required if no 3-D+time variables and pressure levels were
            provided on init
            If pressure levels were provided on init, those values will be used
            instead
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
                    variable = 'Q'
                elif prefix == 'THETA':
                    variable = 'T'
                else:
                    raise ValueError('Invalid Variable key for a hard-coded variable {}. Check docs for ClimateAlongTrajectory for a list of valid hard-coded variables'.format(variable_key))
            else:
                raise ValueError("Invalid suffix {} for variable key {}; must be '1D' or 'hc'".format(suffix, variable_key))
        else:
            variable = variable_key
        raw_data = self.winter_file.variable(variable).sel(time=self.subset_time)

        # Account for periodicity in lon by duplicating lon=0 as lon=360, if necessary
        if any(self.traj_lon.values > max(raw_data['lon'].values)):
            variable_data = assist.roll_longitude(raw_data)
        else:
            variable_data = raw_data

        # Two-dimensional climate variables
        if variable_data.dims == ('time', 'lat', 'lon'):
            values = variable_data.interp(time=self.traj_time, lat=self.traj_lat, lon=self.traj_lon, method=self.traj_interpolation, kwargs={'bounds_error': True})
            variable_name = variable

        # Three-dimensional climate variables
        elif variable_data.dims == ('time', 'lev', 'lat', 'lon'):
            # Set up for vertical interpolation if it has never been done before
            if not hasattr(self, 'pressure_levels'):
                self.setup_pinterp(pressure_levels)
            # Subset first to reduce interpolation time
            subset = variable_data.sel(lat=self.subset_lat, lon=self.subset_lon)
            da_on_pressure_levels = self.winter_file.interpolate(subset, self.pressure_levels, interpolation=self.pres_interpolation, extrapolate=self.pres_extrapolate, fill_value=self.fill_value)
            if to_1D:
                # Interpolate onto trajectory pressure to collapse vertical dimension
                values = da_on_pressure_levels.interp(time=self.traj_time, pres=self.traj_pres, lat=self.traj_lat, lon=self.traj_lon, method=self.traj_interpolation, kwargs={'bounds_error': True})
                variable_name = variable_key
            elif hardcoded and (prefix == 'LWP'):
                unit_conversion = 1/9.81 # LWP = Q*dp/g
                along_traj = da_on_pressure_levels.interp(time=self.traj_time, lat=self.traj_lat, lon=self.traj_lon, method=self.traj_interpolation, kwargs={'bounds_error': True})
                along_traj_nan0 = along_traj.where(~np.isnan(along_traj.values), other=0.) # convert NaN to 0 so they don't contribute to integral
                values = unit_conversion * along_traj_nan0.sortby('pres').integrate('pres') # sortby pressure so that answer is positive definite
                values.name = 'LWP'
                variable_name = 'LWP'
            elif hardcoded and (prefix == 'THETA'):
                p_0 = 1e5 # reference pressure 1,000 hPa
                kappa = 2./7. # Poisson constant
                T_values = da_on_pressure_levels.interp(time=self.traj_time, pres=self.traj_pres, lat=self.traj_lat, lon=self.traj_lon, method=self.traj_interpolation, kwargs={'bounds_error': True})
                p_values = self.traj_pres
                values = T_values * (p_0 / p_values)**kappa
                values.name = 'THETA'
                variable_name = 'THETA'
            else:
                values = da_on_pressure_levels.interp(time=self.traj_time, lat=self.traj_lat, lon=self.traj_lon, method=self.traj_interpolation, kwargs={'bounds_error': True})
                variable_name = variable

        else:
            raise ValueError('The requested variable {} has unexpected dimensions {}. Dimensions must be (time, lat, lon) or (time, lev, lat, lon)'.format(variable, variable_data.dims))
        
        # Update Dataset with new DataArray
        self.data[variable_name] = values

    def setup_pinterp(self, pressure_levels):
        '''
        One-time setup of interpolation onto pressure levels

        If 3-D+time variables were not requested on init, this function will be
        called by add_variable the first time a 3-D+time variable is added

        Parameters
        ----------
        pressure_levels: array-like
            pressure levels, in Pa, to interpolate onto
        '''
        if pressure_levels is not None:
            self.pressure_levels = pressure_levels
            #    inputs for vertical interpolation function
            self.pres_interpolation = 'linear'  # for Ngl.vinth2p; options=linear, log, log-log
            self.pres_extrapolate = False  # for Ngl.vinth2p
            self.fill_value = np.nan  # for winter_file.interpolate
        else:
            raise NameError('pressure_levels has not been defined, please provide an array of pressure values in Pa to interpolate 3-D variables onto')

    def check_variable_exists(self, variable_key):
        '''
        Raise error if CAM variable corresponding to variable_key does not exist
        in WinterCAM file

        If variable_key corresponds to a hard-coded variable, check that the
        variables required to calculate variable_key are present in WinterCAM

        Parameters
        ----------
        variable_key: string
            variable name to be checked against self.winter_file data variables
        '''
        # Map hard-coded variable names to the CAM variables they require
        hc_requires = {'LWP_hc': 'Q', 'THETA_hc': 'T'}

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
