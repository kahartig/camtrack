"""
Author: Kara Hartig

Extract climate variables from CAM along trajectories from HYSPLIT

Classes:
    ClimateAlongTrajectory: store CAM variables at times and locations
        corresponding to trajectory paths

NOTES on line_plots:
    time axis is numerical days; could add option to set to some other
        trajectory column instead, like datetime string
    QUESTION: how should I dynamically set fig height based on number of plots?
    QUESTION: how/where should I check that variables_to_plot are valid names? new attribute for list of stored variable names?
    QUESTION: add option to give list of variables to plot? one per plot

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


class ClimateAlongTrajectory:
    '''
    Store climate variables along given air parcel trajectory

    NOTE: uses nearest-neighbor lat and lon points from CAM file; no
    interpolation yet

    Methods
    -------
    trajectory_plot
        plot map of trajectory path in North Polar Stereo
    line_plots
        line plots of selected 2-D variables along trajectory
    contour_plots
        contour plots of selected 3-D variables along trajectory

    Attributes
    ----------
    direction: string
        direction of trajectory calculation: 'FORWARD' or 'BACKWARD'
    trajectory: pandas DataFrame
        3-hourly points along the trajectory, indexed by trajectory age; output
        frequency matches that of CAM file
    data: xarray Dataset
        values of 2-D and 3-D variables along the trajectory. Dimensions are
        'time' and possibly 'lev' (if there are any 3-D variables). 2-D
        variables have 'time' dimension and 3-D have 'time' and 'lev'. Lat and
        lon are included as coordinate arrays with dimension 'time' and reflect
        trajectory path or nearest-neighbor equivalent using CAM coordinates
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
            list of CAM variables (field names in all caps) that will be stored
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
        self.traj_interpolation = traj_interpolation
        self.traj_number = trajectory_number

        # Check that all requested variables exist in CAM files
        if not all(key in winter_file.dataset.data_vars for key in variables):
            missing_keys = [
                key for key in variables if key not in winter_file.dataset.data_vars]
            raise ValueError(
                'One or more variable names provided is not present in CAM output files. Invalid name(s): {}'.format(missing_keys))

        # Check that pressure levels are provided if any 3-D variables are requested
        has_3d_vars = False # True if there are any 3-D variables requested
        if any(winter_file.variable(key).dims == ('time', 'lev', 'lat', 'lon') for key in variables):
            has_3d_vars = True
            list_3d_vars = [key for key in variables if winter_file.variable(key).dims == ('time', 'lev', 'lat', 'lon')]
            if pressure_levels is None:
                raise ValueError('One or more requested variables has 3 spatial dimensions ({}), so pressure_levels must be provided for vertical interpolation'.format(list_3d_vars))
    
        # Select a single trajectory
        self.trajectory = trajectories.get_trajectory(trajectory_number, 3)
        self.direction = trajectories.direction

        # Retrieve time, lat, lon along trajectory
        time_coord = {'time': self.trajectory['cftime date'].values}
        self.traj_time = xr.DataArray(time_coord['time'], dims=('time'), coords=time_coord)
        self.traj_lat = xr.DataArray(self.trajectory['lat'].values, dims=('time'), coords=time_coord)
        self.traj_lon = xr.DataArray(self.trajectory['lon'].values, dims=('time'), coords=time_coord)

        # Set up subset to trajectory path        
        lat_pad = 1.5 * max(abs(np.diff(self.winter_file.variable('lat'))))
        lon_pad = 1.5 * max(abs(np.diff(self.winter_file.variable('lon'))))
        self.subset_lat = slice(min(self.traj_lat.values) - lat_pad, max(self.traj_lat.values) + lat_pad)
        self.subset_lon = slice(min(self.traj_lon.values) - lon_pad, max(self.traj_lon.values) + lon_pad)
        self.subset_time = slice(min(self.traj_time.values), max(self.traj_time.values))

        # Set up interpolation to pressure levels for 3-D variables:
        if has_3d_vars:
            self.setup_pinterp(pressure_levels)
        
        # Store height and diagnostic output variables
        list_of_variables = []
        height_attrs = {'units': 'm above ground level', 'long_name': 'Parcel height above ground level'}
        list_of_variables.append(xr.DataArray(self.trajectory['height (m)'].values, name='HEIGHT', attrs=height_attrs, dims=('time'), coords=time_coord))
        for key in trajectories.diag_var_names:
            key_attrs = {'units': 'unknown', 'long_name': key + ' from HYSPLIT diagnostic variables'}
            list_of_variables.append(xr.DataArray(self.trajectory[key].values, name=key, attrs=key_attrs, dims=('time'), coords=time_coord))
        self.data = xr.merge(list_of_variables)        

        # Store requested climate variables
        for key in variables:
            self.add_variable(key)


    def add_variable(self, variable, to_1D=False, pressure_levels=None):
        '''
        DOC

        if variable is 3-D+time and to_1D is True, will also interp onto trajectory pressure to collapse vertical dimension
        '''
        variable_data = self.winter_file.variable(variable)

        # if interpolating 3-D onto 1-D, find pressures along trajectory
        if to_1D:
            pressures = self.traj_file.height2pressure(self.winter_file.directory, self.traj_number)['pressure']
            time_coord = self.traj_time.coords
            traj_pres = xr.DataArray(pressures.values, dims='time', coords=time_coord)

        # Two-dimensional climate variables
        if variable_data.dims == ('time', 'lat', 'lon'):
            if self.traj_interpolation == 'nearest':
                values = variable_data.sel(time=self.traj_time, lat=self.traj_lat, lon=self.traj_lon, method='nearest')
            elif self.traj_interpolation == 'linear':
                values = variable_data.interp(time=self.traj_time, lat=self.traj_lat, lon=self.traj_lon, method='linear', kwargs={'bounds_error': True})
            else:
                raise ValueError("Invalid interpolation method onto trajectory '{}'. Must be 'nearest' or 'linear'".format(self.traj_interpolation))

        # Three-dimensional climate variables
        elif variable_data.dims == ('time', 'lev', 'lat', 'lon'):
            # Set up for vertical interpolation if it has never been done before
            if not hasattr(self, 'pressure_levels'):
                self.setup_pinterp(pressure_levels)
            # Subset first to reduce interpolation time
            subset = variable_data.sel(time=self.subset_time, lat=self.subset_lat, lon=self.subset_lon)
            da_on_pressure_levels = self.winter_file.interpolate(subset, self.pressure_levels, interpolation=self.pres_interpolation, extrapolate=self.pres_extrapolate, fill_value=self.fill_value)
            if to_1D:
                # Interpolate onto trajectory pressure to collapse vertical dimension
                if self.traj_interpolation == 'nearest':
                    values = da_on_pressure_levels.sel(time=self.traj_time, pres=traj_pres, lat=self.traj_lat, lon=self.traj_lon, method='nearest')
                elif self.traj_interpolation == 'linear':
                    values = da_on_pressure_levels.interp(time=self.traj_time, pres=traj_pres, lat=self.traj_lat, lon=self.traj_lon, method='linear', kwargs={'bounds_error': True})
                else:
                    raise ValueError("Invalid interpolation method onto trajectory '{}'. Must be 'nearest' or 'linear'".format(self.traj_interpolation))
                variable = variable + '_1D'
            else:
                if self.traj_interpolation == 'nearest':
                    values = da_on_pressure_levels.sel(time=self.traj_time, lat=self.traj_lat, lon=self.traj_lon, method='nearest')
                elif self.traj_interpolation == 'linear':
                    values = da_on_pressure_levels.interp(time=self.traj_time, lat=self.traj_lat, lon=self.traj_lon, method='linear', kwargs={'bounds_error': True})
                else:
                    raise ValueError("Invalid interpolation method onto trajectory '{}'. Must be 'nearest' or 'linear'".format(self.traj_interpolation))
        else:
            raise ValueError('The requested variable {} has unexpected dimensions {}. Dimensions must be (time, lat, lon) or (time, lev, lat, lon)'.format(variable, variable_data.dims))
        
        # Update Dataset with new DataArray
        self.data[variable] = values

    def setup_pinterp(self, pressure_levels):
        '''
        DOC
        '''
        if pressure_levels is not None:
            self.pressure_levels = pressure_levels
            #    inputs for vertical interpolation function
            self.pres_interpolation = 'linear'  # for Ngl.vinth2p; options=linear, log, log-log
            self.pres_extrapolate = False  # for Ngl.vinth2p
            self.fill_value = np.nan  # for winter_file.interpolate
        else:
            raise NameError('pressure_levels has not been defined, please provide an array of pressure values in Pa to interpolate onto')
