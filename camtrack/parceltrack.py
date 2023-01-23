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
import datetime
import calendar
import scipy # for qhull error

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

    def __init__(self, winter_file, trajectories, trajectory_number, variables, traj_interpolation, below_LML='NaN'):
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
                ends in ':1D': 3-D+time variable to be interpolated directly
                    onto trajectory path
                ends in ':hc': hard-coded variable with special handling
                    Available options:
                    'LWP:hc': liquid water path (vertical integral of 'Q')
                    'THETA:hc': potential temperature
        traj_interpolation: 'nearest' or 'linear'
            interpolation method for matching trajectory lat-lon to CAM variables
        below_LML: string
            retrieval method for interpolation when trajectory is below lowest model level
            if 'NaN': return np.nan
            if 'LML': return value at lowest model level (above parcel)
            Default is 'NaN'.

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
        cam_cadence = int(winter_file.time_step/3600) # CAM timestep in hours; anything < 1 -> 0
        traj_cadence = cam_cadence if cam_cadence >= 1 else 1 # match CAM, lowest allowed value is 1 hour
        self.trajectory = trajectories.get_trajectory(trajectory_number, traj_cadence)
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
            self.add_variable(key, below_LML)


    def add_variable(self, variable_key, below_LML='NaN'):
        '''
        Interpolate a new variable onto trajectory path

        The new variable is automatically added to self.data
        For 2-D+time variables:
            to interpolate onto time, lat, and lon, simply provide the variable
              name e.g. 'TS'
        For 3-D+time variables:
            to interpolate onto pressure as well as time, lat, and lon
              (collapse vertical dimension), add ':1D' to end of variable name
              e.g. 'OMEGA:1D'
            interpolating onto time, lat, and lon and retaining the vertical
              dimension is not currently supported

        Parameters
        ----------
        variable_key: string
            name of variable to be interpolated onto trajectory path
            Must be a CAM variable name (field in all caps) OR one of two types
            of special variables:
                ends in ':1D': 3-D+time variable to be interpolated directly
                    onto trajectory path, e.g. 'T:1D'
                ends in ':hc': hard-coded variable with special handling
                    See self.hc_requires for valid hard-coded variables
        below_LML: string
            retrieval method for interpolation of 3-D+time variables when
              trajectory is below lowest model level
            if 'NaN': return np.nan
            if 'LML': return value at lowest model level (above parcel)
            Default is 'NaN'.
        '''
        # Identify if variable requires special handling
        to_1D = False
        hardcoded = False
        if ':' in variable_key:
            variable, tag = variable_key.split(':', 1)
            if tag == '1D':
                to_1D = True
            elif tag == 'hc':
                hardcoded = True
            else:
                raise ValueError("Invalid tag {} on variable key {}; must be '1D' or 'hc'".format(tag, variable_key))
        else:
            variable = variable_key

        if hardcoded:
            # Hard-coded variable
            self.add_hardcoded_variable(variable, below_LML)

        else:
            # Load variable dimensions
            data_dims = self.winter_file.variable(variable).dims

            # Two-dimensional climate variable
            if data_dims == ('time', 'lat', 'lon'):
                self.add_2D_variable(variable)

            # Three-dimensional climate variable
            elif data_dims == ('time', 'lev', 'lat', 'lon'):
                if to_1D:
                    self.add_3Dto1D_variable(variable, below_LML)
                else:
                    raise NotImplementedError("Interpolating 3-D variables only onto time, lat, and lon has not been implemented; must add ':1D' to end of variable name and interpolate onto pressure as well")

            # Error: invalid/unexpected dimensions
            else:
                raise ValueError('The requested variable {} has unexpected dimensions {}. Dimensions must be (time, lat, lon) or (time, lev, lat, lon)'.format(variable, data_dims))


    def add_hardcoded_variable(self, variable, below_LML):
        '''
        Add a hard-coded variable along trajectory path

        See self.hc_requires for the set of valid hard-coded variables
        '''
        variable_key = variable + ':hc'

        # Checks
        # is this a valid hard-coded variable?
        if variable_key not in self.hc_requires.keys():
            raise ValueError('Invalid variable key for a hard-coded variable {}. Check self.hc_requires for a list of valid hard-coded variables'.format(variable_key))
        # are required variables present in CAM file?
        self.check_variable_exists(variable_key)
        # do required variables already exist in self.data?
        required = self.hc_requires[variable_key]
        if required is not None:
            for var in required:
                if var not in self.data.data_vars:
                    self.add_variable(var, below_LML)

        if variable == 'THETA':
            p_0 = 1e5 # reference pressure 1,000 hPa
            kappa = 2./7. # Poisson constant
            T_values = self.traj_file.col2da(self.traj_number, 'AIR_TEMP', include_coords='cftime date').swap_dims({'traj age': 'cftime date'}).rename({'cftime date': 'time'})
            p_values = self.traj_pres
            values = T_values * (p_0 / p_values)**kappa
            values.name = variable
            values = values.assign_attrs({'units': 'K', 'long_name': 'Potential temperature'})
            variable_name = variable_key
        elif variable == 'DSE':
            # Specific heat
            c_p_dry = 1005.7 # specific heat of dry air, J/kg*K
            c_p_vapor = 1859 # specific heat of water vapor, J/kg*K
            try:
                # Use specific humidity to calculate c_p of moist air
                if 'Q:1D' not in self.data.data_vars:
                    self.add_variable('Q:1D', below_LML)
                c_p = c_p_dry + self.data['Q:1D'] * c_p_vapor
            except ValueError:
                # ValueError raised when calling self.add_variable();
                # specific humidity missing from CAM file
                c_p = c_p_dry
            g = 9.81 # m/s2
            values = self.data['T:1D'] + (g/c_p)*self.data['Z3:1D']
            values.name = variable
            values = values.assign_attrs({'units': 'K', 'long_name': 'Dry static energy'})
            variable_name = variable_key
        else:
            raise ValueError('Invalid variable key for a hard-coded variable {}. Check self.hc_requires for a list of valid hard-coded variables'.format(variable_key))

        # Update Dataset with new DataArray
        self.data[variable_name] = values


    def add_2D_variable(self, variable):
        '''
        Interpolate (time, lat, lon) variable onto trajectory path
        '''
        # Checks
        # is variable present in CAM file?
        self.check_variable_exists(variable)
        # does variable have the correct dimensions?
        variable_data = self.winter_file.variable(variable)
        if variable_data.dims != ('time', 'lat', 'lon'):
            raise ValueError("The requested variable {} has unexpected dimensions {}. Dimensions must be (time, lat, lon)".format(variable, variable_data.dims))

        values = np.zeros(len(self.trajectory))
        t_idx = 0
        for age, point in self.trajectory.iterrows():
            time = point['cftime date']
            variable_at_time = variable_data.sel(time=time, method='nearest', tolerance=self.dt_tol)

            # Account for periodicity in lon by duplicating lon=0 as lon=360, if necessary
            if point['lon'] > max(variable_data['lon'].values):
                variable_at_time = assist.roll_longitude(variable_at_time)
            
            values[t_idx] = variable_at_time.interp(lat=point['lat'], lon=point['lon'], method=self.traj_interpolation, kwargs={'bounds_error': True})
            t_idx = t_idx + 1
        # Bundle into DataArray
        variable_name = variable
        values = xr.DataArray(values, name=variable_name,
                              dims=('time',),
                              coords={'time': self.traj_time, 'lat': self.traj_lat, 'lon': self.traj_lon},
                              attrs=variable_data.attrs)

        # Update Dataset with new DataArray
        self.data[variable_name] = values


    def add_3Dto1D_variable(self, variable, below_LML):
        '''
        Interpolate (time, lev, lat, lon) variable onto trajectory path
        '''
        # Checks
        # is variable present in CAM file?
        self.check_variable_exists(variable)
        # does variable have the correct dimensions?
        variable_data = self.winter_file.variable(variable)
        if variable_data.dims != ('time', 'lev', 'lat', 'lon'):
            raise ValueError("The requested variable {} has unexpected dimensions {}. Dimensions must be (time, lev, lat, lon)".format(variable, variable_data.dims))
        # set up for vertical interpolation, if it has never been done before
        if not self.ready_pinterp:
            self.setup_pinterp()

        # Identify a corresponding surface-level variable, if any
        surface_counterpart = {'T': 'TREFHT', 'Q': 'QREFHT', 'RELHUM': 'RHREFHT', 'Z3': 'PHIS'} # map upper-level variables to surface-level counterparts
        if (variable in surface_counterpart.keys()) and (surface_counterpart[variable] in self.winter_file.dataset.data_vars):
            surf_available = True
            if surface_counterpart[variable] == 'PHIS':
                surf_scale = 1/9.8 # convert m2/s2 to m to match Z3
            else:
                surf_scale = 1.0
        else:
            surf_available = False


        values = np.zeros(len(self.trajectory))
        t_idx = 0
        for age, point in self.trajectory.iterrows():
            time = point['cftime date']
            variable_at_time = variable_data.sel(time=time, method='nearest', tolerance=self.dt_tol)

            # If needed, load surface-level data
            if surf_available and (point['PRESSURE'] > self.lowest_model_pressure.sel(time=time)):
                add_surf = True
                surf_at_time = self.winter_file.variable(surface_counterpart[variable]).sel(time=time, method='nearest', tolerance=self.dt_tol)
            else:
                add_surf = False

            # Account for periodicity in lon by duplicating lon=0 as lon=360, if necessary
            if point['lon'] > max(variable_data['lon'].values):
                variable_at_time = assist.roll_longitude(variable_at_time)
                if add_surf:
                    surf_at_time = assist.roll_longitude(surf_at_time)

            if (point['PRESSURE'] > self.lowest_model_pressure.sel(time=time)) and (not surf_available):
                # point is below lowest data level (higher pressure)
                if below_LML == 'NaN':
                    values[t_idx] = np.nan
                elif below_LML == 'LML':
                    values[t_idx] = variable_at_time.interp(lat=point['lat'], lon=point['lon'], method=self.traj_interpolation).isel(lev=-1).values
                else:
                    raise ValueError("Invalid retrieval method requested for trajectory points below lowest model level, below_LML={}. Must be 'NaN' or 'LML'".format(below_LML))
            else:
                vertical_profile = variable_at_time.interp(lat=point['lat'], lon=point['lon'], method=self.traj_interpolation)
                # switch to pressure levels
                P_surf = self.data['PS'].sel(time=time, method='nearest', tolerance=self.dt_tol).item()
                vertical_profile = vertical_profile.assign_coords(pressure=("lev", self.P0*self.hyam.values + P_surf*self.hybm.values))
                vertical_profile = vertical_profile.swap_dims({"lev": "pressure"})
                if add_surf:
                    # append surface-level value below lowest model level
                    surface_value = surf_scale * surf_at_time.interp(lat=point['lat'], lon=point['lon'], method=self.traj_interpolation)
                    surface_value = surface_value.assign_coords({'pressure': P_surf}) # assign surface pressure to surface-level value
                    vertical_profile = xr.concat([vertical_profile.reset_coords('lev', drop=True), surface_value], dim='pressure')
                # interpolate onto traj pressure
                values[t_idx] = vertical_profile.interp(pressure=point['PRESSURE'], method=self.traj_interpolation)
            t_idx = t_idx + 1
        # Bundle into DataArray
        variable_name = variable + ':1D'
        values = xr.DataArray(values, name=variable_name,
                              dims=('time',),
                              coords={'time': self.traj_time, 'pres': self.traj_pres, 'lat': self.traj_lat, 'lon': self.traj_lon},
                              attrs=variable_data.attrs)

        # Update Dataset with new DataArray
        self.data[variable_name] = values


    def setup_pinterp(self):
        '''
        One-time setup of interpolation onto pressure levels

        If 3-D+time variables were not requested on init, this function will be
        called by add_variable the first time a 3-D+time variable is added
        '''
        # Store pressure of surface and lowest model level
        self.add_variable('PS')
        self.P0 = self.winter_file.variable('P0').item()
        self.hyam = self.winter_file.variable('hyam')
        self.hybm = self.winter_file.variable('hybm')
        self.lowest_model_pressure = (self.hyam * self.P0 + self.hybm * self.data['PS']).isel(lev=-1)

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
        valid_variables = self.winter_file.dataset.data_vars

        if ':' in variable_key:
            variable, tag = variable_key.split(':', 1)
            if tag == 'hc':
                # Hard-coded
                required = self.hc_requires[variable_key]
                if required is not None:
                    for var in required:
                        req_var, suffix = var.split(':', 1)
                        if req_var not in valid_variables:
                            raise ValueError("CAM output files are missing the variable {} which is required by the requested hard-coded variable {}".format(self.hc_requires[variable_key], variable_key))
            else:
                # Check variable name preceding the tag
                if variable not in valid_variables:
                    raise ValueError("CAM output files are missing a requested variable: {}".format(variable))
        else:
            # Standard CAM variable
            if variable_key not in valid_variables:
                raise ValueError("CAM output files are missing a requested variable: {}".format(variable_key))

    @property
    def hc_requires(self):
        # List of CAM variables required by each hard-coded variable
        return {'THETA:hc': None, 'DSE:hc': ['T:1D', 'Z3:1D']}

    @property
    def dt_tol(self):
        # Set tolerance for 'nearest' time
        # if timestep is off by a few miliseconds, select within half a timestep
        return datetime.timedelta(seconds=self.winter_file.time_step/2)
    
    
