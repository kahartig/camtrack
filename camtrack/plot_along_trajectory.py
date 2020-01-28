# ##############################################################################
# Plot various climate variables along air parcel trajectories
# Authors: Kara Hartig
#     based on code by Naomi Wharton
#
# search for 'QUESTION' to find sections I am unsure about
# ##############################################################################

# Standard imports
import numpy as np
from netCDF4 import Dataset    # QUESTION: do I still use this?
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import cftime
import math    # QUESTION: do I still use this?
import sys    # QUESTION: do I still use this?

# Matplotlib imports
import matplotlib.path as mpath
from matplotlib.pyplot import figure
from matplotlib.cm import get_cmap
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Scipy imports
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator as interpolator

# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from cartopy import config
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

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
        
        # construct column with datetime string
        def traj_datetime(row):
            return '00{:02.0f}-{:02.0f}-{:02.0f} {:02.0f}:{:02.0f}:00'.format(row['year'], row['month'], row['day'], row['hour'], row['minute'])
        trajectories['datetime'] = trajectories.apply(traj_datetime, axis=1)

        # construct column with cftime Datetime objects
        def traj_cftimedate(row):
        	return cftime.DatetimeNoLeap(row['year'], row['month'], row['day'], row['hour'], row['minute'])
        trajectories['cftime date'] = trajectories.apply(traj_cftimedate, axis=1)
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
            raise ValueError("Invalid out_format {}. Must be 'first' or 'first-second' or 'firstsecond'".format(out_format))
        return output


class WinterCAM:
    '''
    Class for storing CAM4 output data along a set of trajectories
    '''

    # Map CAM variable names to h1 through h4 files
    # QUESTION: should this be pulled from the CAM files themselves? something to consider...
    name_to_h = dict.fromkeys(['CLDHGH','CLDICE','CLDLIQ','CLDLOW','CLDMED',
    	'CLDTOT','CLOUD','CONCLD','FICE','ICIMR','ICLDIWP','ICLDTWP','ICWMR',
    	'Q','QFLX','QREFHT','RELHUM','SFCLDICE','SFCLDLIQ','TGCLDCWP',
    	'TGCLDIWP','TGCLDLWP','TMQ'], 'h1') 
    name_to_h.update(dict.fromkeys(['FLDS','FLDSC','FLNS','FLNSC','FLNT',
    	'FLNTC','FLUT','FLUTC','FSDS','FSDSC','FSDTOA','FSNS','FSNSC','FSNT',
    	'FSNTC','FSNTOA','FSNTOAC','FSUTOA','LHFLX','LWCF','QRL','QRS','SHFLX',
    	'SOLIN','SWCF'], 'h2'))
    name_to_h.update(dict.fromkeys(['OMEGA','OMEGAT','PBLH','PHIS','PRECC',
    	'PRECL','PRECT','PS','PSL','SNOWHICE','SNOWHLND','TAUX','TAUY','U',
    	'U10','UU','V','VQ','VT','VU','VV','Z3'], 'h3'))
    name_to_h.update(dict.fromkeys(['ICEFRAC','LANDFRAC','OCNFRAC','T', 'T200',
    	'T500','T850','TREFHT','TREFHTMN','TREFHTMX','TS','TSMN','TSMX'], 'h4'))

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
            nc_file_path = os.path.join(file_dir, 'pi_3h_' + winter_str + '_h1.nc')
            ds1 = xr.open_dataset(nc_file_path)
            nc_file_path = os.path.join(file_dir, 'pi_3h_' + winter_str + '_h2.nc')
            ds2 = xr.open_dataset(nc_file_path)
            nc_file_path = os.path.join(file_dir, 'pi_3h_' + winter_str + '_h3.nc')
            ds3 = xr.open_dataset(nc_file_path)
            nc_file_path = os.path.join(file_dir, 'pi_3h_' + winter_str + '_h4.nc')
            ds4 = xr.open_dataset(nc_file_path)

            # Map h1 through h4 to ds1 through ds4
            self.h_to_d = {'h1': d1, 'h2': d2, 'h3': d3, 'h4': d4}
        else:
            # Only activated when running nosetests
            nc_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests', 'sample_CAM4_for_nosetests.nc')
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
        # QUESTION: should this check against the list of actual variables in the CAM file instead of just the h-file mapping?
        if not all(key in winter_file.name_to_h for key in variables_to_plot):
            missing_keys = [key for key in variables_to_plot if key not in winter_file.name_to_h]
            raise ValueError('One or more variable names provided is not present in CAM output files. Invalid name(s): {}'.format(missing_keys))
        
        # Select a single trajectory
        single_trajectory = trajectories.data.loc[trajectory_number].copy()
        #print('Starting position for trajectory {}:'.format(trajectory_number))
        #print('    {:.2f}N lat, {:.2f}E lon, {:.0f} m above ground'.format(trajectories.loc[(trajectory_number,0)]['lon'],trajectories.loc[(trajectory_number, 0)]['lat'],trajectories.loc[(trajectory_number, 0)]['height (m)']))

        # Extract data every 3 hours
        #    CAM only provides output in 3-hourly intervals
        is_every3hours = single_trajectory['hour'] % 3 == 0
        self.trajectory = single_trajectory[is_every3hours]

        # Convert trajectory lat and lon to DataArrays with dimension 'time'
        #    using the cftime date as 'time' for direct comparison to CAM 'time' dimension
        time_coord = {'time': self.trajectory['cftime date'].values}
        traj_lat = xr.DataArray(self.trajectory['lat'].values, dims=('time'), coords=time_coord)
        traj_lon = xr.DataArray(self.trajectory['lon'].values, dims=('time'), coords=time_coord)

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
                values = variable.sel(lat=traj_lat, lon=traj_lon, time=traj_lat['time'], method='nearest')
                list_of_variables.append(values)
            else:
                raise ValueError('The requested variable {} has unexpected dimensions {}. Dimensions must be (time, lat, lon) for line plot or (time, lev, lat, lon) for contour plot'.format(key, variable.dims))
        self.data = xr.merge(list_of_variables)




def make_plots(plotting_data, trajectory):
    ##########################################################################
    '''
    Plot values of various quantities along the given lat/lon trajectory
    over time. Produces contour plots, line plots, and a map of the 
    overall trajectory.

    Parameters: 
    plotting_data: a DataForPlotting() object containing various
        quantities along a trajectory that we want to plot

    trajectory: a SingleTrajectory() object containing lat/lon values 
        along a single trajectory

    Output: 
    - a PDF containing contour plots along the trajectory of (currently)
    four variables
    - a PDF containing line plots along the trajectory of (currently)
    eleven variables over four plots
    - a PDF containing a map showing the lat/lon coordinates of the
    trajectory with the days labeled

    '''

    every3hours = trajectory.indices['every3hours']
    every24hours = trajectory.indices['every24hours']

    time = plotting_data.time

    # set up time/level grid
    TIME, LEV = np.meshgrid(plotting_data.time, plotting_data.lev)

    # make date labels for x-axis
    idays = np.squeeze(np.where(time[:] % 1 == 0))  # indices

    locs = time[idays]  # locations for x-axis labels

    days = plotting_data.stime[idays]  # dates as time objects
    labels = []
    for day in days:
        labels.append(str(day)[0:10].lstrip('0'))

    # mark minor x-axis tick marks every 3 hours
    minorLocator = MultipleLocator(.125)

    # ------------------------------------------------------------
    # Contour plots: temperature, specific humidity, relative
    #                humidity, and cloud cover along trajectory
    # ------------------------------------------------------------
    if make_contour_plots:

        # create figure
        plt.clf()
        fig = plt.figure()
        fig.set_figheight(12)
        fig.set_figwidth(11)

        # ------------------------------------------------------------
        # plot temperature
        # ------------------------------------------------------------

        # create axes
        ax = fig.add_subplot(4, 1, 1)
        ax.set_ylim([1000, 0])
        # ax.set_xlabel("Days")
        ax.set_ylabel("Pressure")

        # make the contours:
        contour = plt.contourf(TIME, LEV, plotting_data.T, 15,
                               cmap=plt.cm.jet,
                               vmin=190,               # minimum contour value
                               vmax=280)               # maximum contour value

        # add a color bar
        plt.colorbar(ax=ax, shrink=.62, pad=0.02, label='temperature (K)')

        # create minor ticks every three hours
        ax.xaxis.set_minor_locator(minorLocator)

        # set x-axis labels every day
        plt.xticks(locs, labels)

        # add gridlines
        ax.grid(color="grey", linestyle="dotted")

        # title
        plt.title('Temperature along Trajectory, height 10m')

        # add black line showing pressure of air mass
        plt.plot(time, trajectory.pressure[every3hours], color='black')

        # ------------------------------------------------------------
        # plot specific humidity
        # ------------------------------------------------------------

        # create axes
        ax = fig.add_subplot(4, 1, 2)
        ax.set_ylim([1000, 0])
        # ax.set_xlabel("Days")
        ax.set_ylabel("Pressure")

        # make the contours:
        plt.contourf(TIME, LEV, plotting_data.Q, 15,
                     cmap=plt.cm.jet,
                     vmin=0,
                     vmax=.0032)

        # Add a color bar
        plt.colorbar(ax=ax, shrink=.62, pad=0.02,
                     label='specific humidity (kg/kg)')

        # create minor ticks every three hours
        ax.xaxis.set_minor_locator(minorLocator)

        # set x-axis labels every day
        plt.xticks(locs, labels)

        # add gridlines
        ax.grid(color="grey", linestyle="dotted")

        # title
        plt.title('Specific Humidity along Trajectory')

        # add black line showing pressure of air mass
        plt.plot(time, trajectory.pressure[every3hours], color='black')

        # ------------------------------------------------------------
        # plot relative humidity
        # ------------------------------------------------------------

        ax = fig.add_subplot(4, 1, 3)
        ax.set_ylim([1000, 0])
        ax.set_xlabel("Days")
        ax.set_ylabel("Pressure")

        # make the contours:
        plt.contourf(TIME, LEV, plotting_data.RELHUM, 15,
                     cmap=plt.cm.jet,
                     vmin=0,
                     vmax=100)

        # add a color bar
        plt.colorbar(ax=ax, shrink=.62, pad=0.02, label='percent')

        # create minor ticks every three hours
        ax.xaxis.set_minor_locator(minorLocator)

        # set x-axis labels every day
        plt.xticks(locs, labels)

        # add gridlines
        ax.grid(color="grey", linestyle="dotted")

        # title
        plt.title('Relative Humidity along Trajectory')

        # add black line showing pressure of air mass
        plt.plot(time, trajectory.pressure[every3hours], color='black')

        # ------------------------------------------------------------
        # plot cloud fraction
        # ------------------------------------------------------------

        ax = fig.add_subplot(4, 1, 4)
        ax.set_ylim([1000, 0])
        ax.set_xlabel("Days")
        ax.set_ylabel("Pressure")

        # make the contours:
        plt.contourf(TIME, LEV, plotting_data.CLOUD, 15,
                     cmap=plt.cm.jet,
                     vmin=0,
                     vmax=1)

        # add a color bar
        plt.colorbar(ax=ax, shrink=.62, pad=0.02, label='cloud fraction')

        # create minor ticks every three hours
        ax.xaxis.set_minor_locator(minorLocator)

        # set x-axis labels every day
        plt.xticks(locs, labels)

        # add gridlines
        ax.grid(color="grey", linestyle="dotted")

        # title
        plt.title('Cloud Fraction along Trajectory')

        # add black line showing pressure of air mass
        plt.plot(time, trajectory.pressure[every3hours], color='black')

        # add padding between subplots
        plt.tight_layout(h_pad=3.0)

        # save as pdf
        if save_contour_plots_as_pdf:
            h = plt.gcf()
            h.savefig(contour_filename)

        plt.close()

    # ------------------------------------------------------------
    # Line plots: LW/SW flux, heat flux, cloud water path,
    #             cloud forcing
    # ------------------------------------------------------------
    if make_line_plots:

        # create figure
        plt.clf()
        fig = plt.figure()
        fig.set_figheight(14)
        fig.set_figwidth(8)

        # ------------------------------------------------------------
        # heat fluxes (sensible, latent)
        # ------------------------------------------------------------

        # create axes
        ax = fig.add_subplot(4, 1, 1)

        units = plotting_data.variable_units['lhflx']
        ax.set_xlabel("Days")
        ax.set_ylabel(units)

        # points
        # surface latent heat flux
        plt.scatter(time, plotting_data.lhflx, s=5)
        # surface sensible heat flux
        plt.scatter(time, plotting_data.shflx, s=5)

        # lines
        plt.plot(time, plotting_data.lhflx,
                 label=plotting_data.variable_descriptions['lhflx'])
        plt.plot(time, plotting_data.shflx,
                 label=plotting_data.variable_descriptions['shflx'])

        # create minor ticks every three hours
        ax.xaxis.set_minor_locator(minorLocator)

        # set x-axis labels every day
        plt.xticks(locs, labels)

        # add gridlines
        ax.grid(color="grey", linestyle="dotted")

        # shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # title
        plt.title('Heat Flux at Surface')

        # ------------------------------------------------------------
        # cloud forcing (LW, SW)
        # ------------------------------------------------------------

        # create axes
        ax = fig.add_subplot(4, 1, 2)

        units = plotting_data.variable_units['lwcf']
        ax.set_xlabel("Days")
        ax.set_ylabel(units)

        # points
        plt.scatter(time, plotting_data.lwcf, s=5)     # longwave cloud forcing
        # shortwave cloud forcing
        plt.scatter(time, plotting_data.swcf, s=5)
        plt.scatter(time, plotting_data.swcf +
                    plotting_data.lwcf, s=5)  # net cloud forcing

        # line
        plt.plot(time, plotting_data.lwcf,
                 label=plotting_data.variable_descriptions['lwcf'])
        plt.plot(time, plotting_data.swcf,
                 label=plotting_data.variable_descriptions['swcf'])
        plt.plot(time, plotting_data.swcf +
                 plotting_data.lwcf, label='Net cloud forcing')

        # create minor ticks every three hours
        ax.xaxis.set_minor_locator(minorLocator)

        # set x-axis labels every day
        plt.xticks(locs, labels)

        # add gridlines
        ax.grid(color="grey", linestyle="dotted")

        # shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # title
        plt.title('Cloud Forcing at Surface')

        # ------------------------------------------------------------
        # water path (liquid, ice, total)
        # ------------------------------------------------------------

        # create axes
        ax = fig.add_subplot(4, 1, 3)

        units = plotting_data.variable_units['icepath']
        ax.set_ylabel(units)
        ax.set_xlabel("Days")

        # points
        # total grid-box cloud ice water path
        plt.scatter(time, plotting_data.icepath, s=5)
        # total grid-box cloud liquid water path
        plt.scatter(time, plotting_data.liqpath, s=5)
        # total grid-box cloud water path (liq + ice)
        plt.scatter(time, plotting_data.waterpath, s=5)

        # lines
        plt.plot(time, plotting_data.icepath,
                 label=plotting_data.variable_descriptions['icepath'])
        plt.plot(time, plotting_data.liqpath,
                 label=plotting_data.variable_descriptions['liqpath'])
        plt.plot(time, plotting_data.waterpath,
                 label=plotting_data.variable_descriptions['waterpath'])

        # create minor ticks every three hours
        ax.xaxis.set_minor_locator(minorLocator)

        # set x-axis labels every day
        plt.xticks(locs, labels)

        # add gridlines
        ax.grid(color="grey", linestyle="dotted")

        # shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # title
        plt.title('Cloud Water Path at Surface')

        # ------------------------------------------------------------
        # Flux at surface
        # ------------------------------------------------------------

        # create axes
        ax = fig.add_subplot(4, 1, 4)

        units = plotting_data.variable_units['flds']
        ax.set_ylabel(units)
        ax.set_xlabel("Days")

        # points
        # downwelling long wave flux at surface
        plt.scatter(time, plotting_data.flds, s=5)
        plt.scatter(time, plotting_data.flns - plotting_data.flds,
                    s=5)  # net long wave flux at surface
        # downwelling solar flux at surface
        plt.scatter(time, plotting_data.fsds, s=5)
        plt.scatter(time, plotting_data.fsns - plotting_data.fsds,
                    s=5)   # net solar flux at surface

        # lines
        plt.plot(time, plotting_data.flds,
                 label=plotting_data.variable_descriptions['flds'])
        plt.plot(time, plotting_data.flns - plotting_data.flds,
                 label='Upwelling longwave flux at surface')
        plt.plot(time, plotting_data.fsds,
                 label=plotting_data.variable_descriptions['fsds'])
        plt.plot(time, plotting_data.fsns - plotting_data.fsds,
                 label='Upwelling solar flux at surface')

        # create minor ticks every three hours
        ax.xaxis.set_minor_locator(minorLocator)

        # set x-axis labels every day
        plt.xticks(locs, labels)

        # add gridlines
        ax.grid(color="grey", linestyle="dotted")

        # shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # title
        plt.title('Flux at Surface')

        # add padding between subplots
        plt.tight_layout(h_pad=3.0)

        # save as pdf
        if save_line_plots_as_pdf:
            h = plt.gcf()
            h.savefig(line_plots_filename, bbox_inches="tight")

        plt.close()

    # ------------------------------------------------------------
    # Plot lat/lon trajectory on map
    # ------------------------------------------------------------
    if plot_map:
        # create figure
        plt.clf()
        fig = plt.figure(1, figsize=(8, 8))

        # create axes
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        ax.set_global()
        ax.set_extent([0.000001, 360, 55, 90], crs=ccrs.PlateCarree())

        # convert longitude from -180-180 scale to to 360 degree
        # for i in np.where(lons[:] > 0):
        #    lons[i]=lons[i]-360

        plt.plot(trajectory.lons[every3hours], trajectory.lats[every3hours],
                 color='black',
                 transform=ccrs.PlateCarree(),
                 # alpha=.8,          # make slightly transparent
                 zorder=5)          # put below scatter points

        # add land + coasts
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.stock_img()

        # add gridlines
        ax.gridlines(color="black", linestyle="dotted")

        # add labels for the start of each day
        for ind, label in zip(every24hours, labels):

            lat = trajectory.lats[ind]
            lon = trajectory.lons[ind]

            # plt.annotate does not have a transform parameter, so first we find
            # xy coordinates for the points
            at_x, at_y = ax.projection.transform_point(
                lon, lat, src_crs=ccrs.PlateCarree())

            # annotate each day with date
            plt.annotate(label,
                         # location of point to be labeled
                         xy=(at_x, at_y),
                         # offset distance from point for label
                         xytext=(15, -10),
                         textcoords='offset points',
                         color='black',
                         backgroundcolor='white',
                         size=6,
                         arrowprops=dict(arrowstyle='-', color='white', linewidth=1.0))

        # plot all points on map
        plt.scatter(trajectory.lons[trajectory.indices['every12hours']], trajectory.lats[trajectory.indices['every12hours']],
                    transform=ccrs.PlateCarree(),
                    # set color of point based on pressure
                    c=trajectory.pressure[trajectory.indices['every12hours']],
                    cmap='jet_r',
                    s=25,                           # label text size
                    zorder=10,                      # put above plot line
                    edgecolor='black',
                    linewidth=.8)
        # title
        plt.title('Trajectory for air mass starting at height 10m')

        # make colorbar
        cbar = plt.colorbar(ax=ax, shrink=.5, pad=0.1,
                            label='pressure of air mass (hPa)')
        cbar.ax.invert_yaxis()

        # save as pdf
        if save_map_as_pdf:
            h = plt.gcf()
            h.savefig(map_filename)

        plt.close()
