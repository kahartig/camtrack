# ##############################################################################
# Plot various climate variables along air parcel trajectories
# Authors: Kara Hartig
#     based on code by Naomi Wharton
#
# search for 'QUESTION' to find sections I am unsure about
# ##############################################################################

# Standard imports
import numpy as np
#from netCDF4 import Dataset
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import cftime
import calendar
#import math
#import sys

# Matplotlib imports
import matplotlib.path as mpath
#from matplotlib.pyplot import figure
from matplotlib.cm import get_cmap
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Scipy imports
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator as interpolator

# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#from cartopy.feature import NaturalEarthFeature
#from cartopy import config
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Misc imports
from numpy import nanmean
from mpl_toolkits.axes_grid1 import make_axes_locatable


class TrajectoryFile:
    '''
        Class for storage of all data read in from a .traj file, an ASCII file
    output from HYSPLIT. These .traj files contain lat/lon locations of
    an air mass every hour along a trajectory.

    init Input Parameters
    ----------
    filepath: string
        path to .traj file

    Attributes
    ----------
    grids: pandas DataFrame
        contains information for all grids used: model, year, month, day, hour,
        and fhour
    traj_start: pandas DataFrame
        contains initialization information for all trajectories: date, time,
        and location at which trajectory was started
    diag_var_names: list
        list of names of diagnostic output variables. Corresponding values are
        presented in the trailing columns of the .traj files, following the
        current height
    number_of_trajectories: int
        number of distinct trajectories stored in .traj file
    direction: string
        direction of trajectory calculation: 'FORWARD' or 'BACKWARD'
    data: pandas DataFrame
        trajectory data
        uses a MultiIndex:
            top level: int from 1 to number_of_trajectories
                trajectory number 'traj #'
            second level: float from 0.0 to -(total length of trajectory)
                age of trajectory 'traj age'
            EX: to access trajectory 3 at time -5 hours, use data.loc[3, -5]
        columns: grid #, year, month, day, hour, minute, fhour, lat, lon,
            height (m), <any diagnostic variables...>, datetime, cftime date
        note that cftime date is a cftime.DatetimeNoLeap object
                useful for indexing netCDF files by time dimension
    '''

    def __init__(self, filepath):
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
        self.number_of_trajectories = ntraj
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
        # col 1+ - label identification of each of n variables (PRESSURE,
        # THETA, ...)
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
        #min_time = cftime.date2num(cftime.datetime(7 + winter_idx, 12, 1), time_object.units, calendar=time_object.calendar)
        def traj_numtime(row):
            return cftime.date2num(row['cftime date'], units='days since 0001-01-01 00:00:00', calendar='noleap')
        trajectories['numerical time'] = trajectories.apply(
            traj_numtime, axis=1)

        self.data = trajectories

    def winter(self, out_format):
        '''
        Return year(s) corresponding to the winter in which this trajectory occurred

        Should there be an option to look at a single trajectory? if some stop early

        out_format='first' or 'first-second' or 'firstsecond'
            for the winter of 0009-0010:
                'first' -> '09'
                'first-second' -> '09-10'
                'firstsecond' -> '0910'
        returns a string
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
    Class for storing CAM4 output data along a set of trajectories
    '''

    # Map CAM variable names to h1 through h4 files
    # QUESTION: should this be pulled from the CAM files themselves? something
    # to consider...
    name_to_h = dict.fromkeys(['CLDHGH', 'CLDICE', 'CLDLIQ', 'CLDLOW', 'CLDMED',
                               'CLDTOT', 'CLOUD', 'CONCLD', 'FICE', 'ICIMR', 'ICLDIWP', 'ICLDTWP', 'ICWMR',
                               'Q', 'QFLX', 'QREFHT', 'RELHUM', 'SFCLDICE', 'SFCLDLIQ', 'TGCLDCWP',
                               'TGCLDIWP', 'TGCLDLWP', 'TMQ'], 'h1')
    name_to_h.update(dict.fromkeys(['FLDS', 'FLDSC', 'FLNS', 'FLNSC', 'FLNT',
                                    'FLNTC', 'FLUT', 'FLUTC', 'FSDS', 'FSDSC', 'FSDTOA', 'FSNS', 'FSNSC', 'FSNT',
                                    'FSNTC', 'FSNTOA', 'FSNTOAC', 'FSUTOA', 'LHFLX', 'LWCF', 'QRL', 'QRS', 'SHFLX',
                                    'SOLIN', 'SWCF'], 'h2'))
    name_to_h.update(dict.fromkeys(['OMEGA', 'OMEGAT', 'PBLH', 'PHIS', 'PRECC',
                                    'PRECL', 'PRECT', 'PS', 'PSL', 'SNOWHICE', 'SNOWHLND', 'TAUX', 'TAUY', 'U',
                                    'U10', 'UU', 'V', 'VQ', 'VT', 'VU', 'VV', 'Z3'], 'h3'))
    name_to_h.update(dict.fromkeys(['ICEFRAC', 'LANDFRAC', 'OCNFRAC', 'T', 'T200',
                                    'T500', 'T850', 'TREFHT', 'TREFHTMN', 'TREFHTMX', 'TS', 'TSMN', 'TSMX'], 'h4'))

    def __init__(self, file_dir, trajectories, nosetest=False):
        '''
        file_dir is parent directory of CAM files
        assuming file name format pi_3h_<yr range>_h?.nc
        trajectories is an instance of TrajectoryFile
            contains a family of trajectories with same start point and time
        assumes all trajectories run over the same time window
                takes overall min and max of times listed in trajectories
        nosetest only True if running nosetests
        '''
        # Open the CAM files with xarray
        if not nosetest:
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

            # Map h1 through h4 to ds1 through ds4
            self.h_to_d = {'h1': d1, 'h2': d2, 'h3': d3, 'h4': d4}
        else:
            # Only activated when running nosetests
            nc_file_path = os.path.join(os.path.dirname(os.path.abspath(
                __file__)), 'tests', 'sample_CAM4_for_nosetests.nc')
            ds1 = xr.open_dataset(nc_file_path)
            self.h_to_d = dict.fromkeys(['h1', 'h2', 'h3', 'h4'], ds1)

        # Lists of coordinate variables
        #time = np.array(ds1['time'][:])
        self.lat = np.array(ds1['lat'][:])
        self.lon = np.array(ds1['lon'][:])
        #self.lev = np.array(ds1['lev'][:])


class ClimateAlongTrajectory:
    '''
    Stores climate variables along given air parcel trajectory
    '''

    def __init__(self, winter_file, trajectories, trajectory_number, variables_to_plot):
        '''
        DOC
        trajectory file is TrajectoryFile object
        trajectory number is index corresponding to desired traj
        variables_to_plot: list of CAM variable names (string in all caps, like 'LANDFRAC') for plotting
        note that self.trajectory is a Pandas DataFrame while self.data is an xarray Dataset

        NEAREST NEIGHBOR lat and lon only (for now)
        '''
        # Check that all requested variables exist in CAM files
        # QUESTION: should this check against the list of actual variables in
        # the CAM file instead of just the h-file mapping?
        if not all(key in winter_file.name_to_h for key in variables_to_plot):
            missing_keys = [
                key for key in variables_to_plot if key not in winter_file.name_to_h]
            raise ValueError(
                'One or more variable names provided is not present in CAM output files. Invalid name(s): {}'.format(missing_keys))

        # Select a single trajectory
        single_trajectory = trajectories.data.loc[trajectory_number].copy()
        self.direction = trajectories.direction
        #print('Starting position for trajectory {}:'.format(trajectory_number))
        #print('    {:.2f}N lat, {:.2f}E lon, {:.0f} m above ground'.format(trajectories.loc[(trajectory_number,0)]['lon'],trajectories.loc[(trajectory_number, 0)]['lat'],trajectories.loc[(trajectory_number, 0)]['height (m)']))

        # Extract data every 3 hours as default
        #    CAM only provides output in 3-hourly intervals
        is_every3hours = single_trajectory['hour'] % 3 == 0
        self.trajectory = single_trajectory[is_every3hours]

        # Also save every 1, 12, 24 hours for plotting
        self.trajectory_1h = single_trajectory  # output every 1 hour
        self.trajectory_12h = single_trajectory[
            single_trajectory['hour'] % 12 == 0]  # every 12 hours
        self.trajectory_24h = single_trajectory[
            single_trajectory['hour'] % 24 == 0]  # every 24 hours

        # Convert trajectory lat and lon to DataArrays with dimension 'time'
        # using the cftime date as 'time' for direct comparison to CAM 'time'
        # dimension
        time_coord = {'time': self.trajectory['cftime date'].values}
        traj_lat = xr.DataArray(
            self.trajectory['lat'].values, dims=('time'), coords=time_coord)
        traj_lon = xr.DataArray(
            self.trajectory['lon'].values, dims=('time'), coords=time_coord)

        # Map trajectory times to climate variables
        # xarray supports vectorized indexing across multiple dimensions
        # (pulling a series of points by matching up lists of each coordinate)
        # as long as DataArrays are used for indexing. If lists were provided
        # instead, xarray would use orthogonal indexing
        list_of_variables = []
        for key in variables_to_plot:
            ds = winter_file.h_to_d[winter_file.name_to_h[key]]
            variable = ds[key]
            if (variable.dims == ('time', 'lat', 'lon')) or (variable.dims == ('time', 'lev', 'lat', 'lon')):
                values = variable.sel(lat=traj_lat, lon=traj_lon,
                                      time=traj_lat['time'], method='nearest')
                list_of_variables.append(values)
            else:
                raise ValueError('The requested variable {} has unexpected dimensions {}. Dimensions must be (time, lat, lon) for line plot or (time, lev, lat, lon) for contour plot'.format(
                    key, variable.dims))
        self.data = xr.merge(list_of_variables)

    def plot_trajectory_path(self, save_file_path):
        '''
        DOC
        save_file_path: directory path and name for saving file
            file name suffix will determine saved file format
        '''
        # Initialize plot
        plt.clf()
        plt.rcParams.update({'font.size': 14})  # set overall font size
        plt.title('{} trajectory starting at {:.0f} m'.format(
            self.direction, self.trajectory.loc[0]['height (m)']), fontsize=24)
        cm = plt.get_cmap('inferno')  # colormap for trajectory age

        # Set map projection
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        ax.set_global()
        min_plot_lat = 50 if all(self.trajectory_1h['lat'].values > 50) else min(
            self.trajectory_1h['lat'].values) - 5
        ax.set_extent([-180, 180, min_plot_lat, 90], crs=ccrs.PlateCarree())

        # Add features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.gridlines(color='black', linestyle='dotted')

        # Plot trajectory path as black line
        plt.plot(self.trajectory_1h['lon'].values,
                 self.trajectory_1h['lat'].values,
                 color='black',
                 transform=ccrs.Geodetic(),
                 zorder=1)

        # Plot points every 12 hours, shaded by trajectory age
        plt.scatter(self.trajectory_12h['lon'].values,
                    self.trajectory_12h['lat'].values,
                    transform=ccrs.Geodetic(),
                    c=self.trajectory_12h.index.values,
                    vmin=min(self.trajectory_12h.index.values),
                    vmax=max(self.trajectory_12h.index.values),
                    cmap=cm,
                    s=100,
                    zorder=2,
                    edgecolor='black',
                    linewidth=0.8)

        # Add labels at start of each day
        for idx, row in self.trajectory_24h.iterrows():
            month = calendar.month_abbr(row['month'])
            at_lat, at_lon = ax.projection.transform_point(
                x=row['lon'], y=row['lat'], src_crs=ccrs.Geodetic())
            plt.annotate('{} {:02d}'.format(month, row['day']),
                         xy=(at_lat, at_lon),
                         xytext=(25, 15),
                         textcoords='offset points',
                         color='black',
                         backgroundcolor='xkcd:silver',
                         size=14,
                         zorder=3,
                         bbox=dict(boxstyle='round', alpha=0.9,
                                   fc='xkcd:silver', ec='xkcd:silver'),
                         arrowprops=dict(arrowstyle='wedge,tail_width=0.5',
                                         alpha=0.9, fc='xkcd:silver',
                                         ec='xkcd:silver'))

        # Make colorbar
        cbar = plt.colorbar(ax=ax, shrink=0.7, pad=0.05,
                            label='Trajectory Age (hours)')

        # Set circular outer boundary
        # from https://scitools.org.uk/cartopy/docs/v0.15/examples/always_circular_stereo.html
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        ax.figure.set_size_inches(10, 10)

        # Save as pdf
        # QUESTION: keep transparent=True as default, or set as argument?
        # h=plt.gcf()
        # h.savefig(map_filename, transparent=True)

    def make_line_plots(self, save_file_path, variables_to_plot=None):
        '''
        DOC
        variables_to_plot: dict or None
            if dict, then each key, value pair is a graph title and the set of
                variable names to plot on that graph
                ex: {'Cloud Cover': ['CLDTOT', 'CLDLOW', 'CLDHGH']}
                each variable in the list must have the same units
            if None, plot each line variable in the data set on its own figure
        time axis is numerical days; could add option to set to some other
            trajectory column instead, like datetime string
        must give save_file_name? no good way to display multiple plots w/o saving

        QUESTION: how should I dynamically set fig height based on number of plots?
        QUESTION: how/where should I check that variables_to_plot are valid names? new attribute for list of stored variable names?
        '''
        line_data = self.data.drop_dims('lev')  # remove all 3-D variables
        time = self.trajectory['numerical time'].values

        # Initialize figure
        plt.clf()
        fig = plt.figure()
        fig.set_figheight(14)
        fig.set_figwidth(8)

        if isinstance(variables_to_plot, dict):
            num_plots = len(variables_to_plot.keys())
            for idx, (title, variable_list) in enumerate(variables_to_plot.items()):
                # Check that all variables have same units
                unit_list = [line_data[var].units for var in variable_list]
                if unit_list.count(unit_list[0]) != len(unit_list):
                    raise ValueError('Units do not match for the variables {}, cannot be plot together. Corresponding units for the given variables are {}'.format(
                        variable_list, unit_list))

                # Initialize plot
                ax = fig.add_subplot(num_plots, 1, idx + 1)
                ax.set_xlabel('Days')
                ax.set_ylabel(unit_list[0])

                # Plot all variables
                for variable in variable_list:
                    plt.plot(time, line_data[variable].values,
                             '-o', label=line_data[variable].long_name)

                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.title(title)
            plt.tight_layout(h_pad=3.0)
        elif variables_to_plot is None:
            num_plots = len(line_data.keys())

            # Plot all variables
            for idx, variable in enumerate(line_data.keys()):
                ax = fig.add_subplot(num_plots, 1, idx + 1)
                ax.set_xlabel('Days')
                ax.set_ylabel(line_data[variable].units)
                plt.plot(time, line_data[variable].values, '-o')
                plt.title(line_data[variable].long_name)
        else:
            raise ValueError('Invalid first argument; must be a dictionary or None, but given type {}'.format(
                type(variables_to_plot)))

        # Save as PDF
        #h = plt.gcf()
        # h.savefig(save_file_name)

    def make_contour_plots(self, save_file_path, variables_to_plot=None):
        '''
        DOC
        '''