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
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import cftime
import calendar

# Matplotlib imports
import matplotlib.path as mpath
from matplotlib.cm import get_cmap

# Scipy imports
#from scipy.interpolate import griddata
#from scipy.interpolate import RegularGridInterpolator as interpolator

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
    trajectory_1h: pandas DataFrame
        same as trajectory but output is every hour; the source format from
        TrajectoryFile
    trajectory_12h: pandas DataFrame
        same as trajectory but output is every 12 hours
    trajectory_24h: pandas DataFrame
        same as trajectory but output is every 24 hours
    data: xarray Dataset
        values of 2-D and 3-D variables along the trajectory. Dimensions are
        'time' and possibly 'lev' (if there are any 3-D variables). 2-D
        variables have 'time' dimension and 3-D have 'time' and 'lev'. Lat and
        lon are included as coordinate arrays with dimension 'time' and reflect
        trajectory path or nearest-neighbor equivalent using CAM coordinates
    '''

    def __init__(self, winter_file, trajectories, trajectory_number, variables):
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
        '''
        # Check that all requested variables exist in CAM files
        # QUESTION: should this check against the list of actual variables in
        # the CAM file instead of just the h-file mapping?
        if not all(key in winter_file.name_to_h for key in variables):
            missing_keys = [
                key for key in variables if key not in winter_file.name_to_h]
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
        for key in variables:
            ds = winter_file.h_to_d[winter_file.name_to_h[key]]
            variable_data = ds[key]
            if (variable_data.dims == ('time', 'lat', 'lon')) or (variable_data.dims == ('time', 'lev', 'lat', 'lon')):
                values = variable_data.sel(lat=traj_lat, lon=traj_lon,
                                           time=traj_lat['time'], method='nearest')
                list_of_variables.append(values)
            else:
                raise ValueError('The requested variable {} has unexpected dimensions {}. Dimensions must be (time, lat, lon) for line plot or (time, lev, lat, lon) for contour plot'.format(
                    key, variable_data.dims))
        self.data = xr.merge(list_of_variables)

    def trajectory_plot(self, save_file_path=None):
        '''
        Either saves or displays a map of trajectory path

        Parameters
        ----------
        save_file_path: path-like or None
            if None, plot is displayed to screen with plt.show()
            if path-like, must be directory path and name for saving figure
                file format assumed from suffix
        '''
        # Initialize plot
        plt.clf()
        plt.rcParams.update({'font.size': 14})  # set overall font size
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
                 linestyle='dashed',
                 transform=ccrs.Geodetic(),
                 zorder=1)

        # Plot start point as a cyan X
        traj_start = self.trajectory.loc[0]
        plt.scatter(traj_start['lon'], traj_start['lat'],
                    transform=ccrs.Geodetic(), c='tab:cyan', marker='X', s=100, zorder=2)

        # Plot points every 12 hours, shaded by trajectory age
        plt.scatter(self.trajectory_12h['lon'].values,
                    self.trajectory_12h['lat'].values,
                    transform=ccrs.Geodetic(),
                    c=self.trajectory_12h.index.values,
                    vmin=min(self.trajectory_12h.index.values),
                    vmax=max(self.trajectory_12h.index.values),
                    cmap=cm,
                    s=100,
                    zorder=3,
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
                         zorder=4,
                         bbox=dict(boxstyle='round', alpha=0.9,
                                   fc='xkcd:silver', ec='xkcd:silver'),
                         arrowprops=dict(arrowstyle='wedge,tail_width=0.5',
                                         alpha=0.9, fc='xkcd:silver',
                                         ec='xkcd:silver'))

        # Make colorbar
        cbar = plt.colorbar(ax=ax, shrink=0.7, pad=0.05,
                            label='Trajectory Age (hours)')

        # Set circular outer boundary
        # from
        # https://scitools.org.uk/cartopy/docs/v0.15/examples/always_circular_stereo.html
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        # Resize and add title
        ax.figure.set_size_inches(10, 10)
        deg = u'\N{DEGREE SIGN}'
        plt.title('{} trajectory from {:.01f}{}N, {:.01f}{}E, {:.0f} m'.format(
            self.direction, traj_start['lat'], deg, traj_start['lon'], deg,
            self.trajectory.loc[0]['height (m)']), fontsize=22)

        if save_file_path is not None:
            # QUESTION: use transparent=True as default, or set as argument?
            plt.savefig(save_file_path)  # , transparent=True)
            plt.close()
        else:
            plt.show()

    def line_plots(self, save_file_path=None, variables_to_plot=None):
        '''
        Either saves or displays a series of line plots of 2-D variables along
        trajectory

        Parameters
        ----------
        save_file_path: path-like or None
            if None, plot is displayed to screen with plt.show()
            if path-like, must be directory path and name for saving figure
                file format assumed from suffix
        variables_to_plot: dict or None
            if None, plot every 2-D variable present in this instance on its
                own plot
            if dict, then each key, value pair is a graph title and the set of
                variable names to plot on that graph
                ex: {'Cloud Cover': ['CLDTOT', 'CLDLOW', 'CLDHGH']}
                each variable in the list must have the same units

        variables_to_plot: dict or None
            if dict, then each key, value pair is a graph title and the set of
                variable names to plot on that graph
                ex: {'Cloud Cover': ['CLDTOT', 'CLDLOW', 'CLDHGH']}
                each variable in the list must have the same units
            if None, plot each line variable in the data set on its own figure
        '''
        line_data = self.data.drop_dims(
            'lev', errors='ignore')  # remove all 3-D variables, if present
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
        elif variables_to_plot is None:
            num_plots = len(line_data.keys())

            # Plot all variables
            for idx, variable in enumerate(line_data.keys()):
                ax = fig.add_subplot(num_plots, 1, idx + 1)
                ax.set_xlabel('Days')
                ax.set_ylabel(line_data[variable].units)
                plt.plot(time, line_data[variable].values, '-o')
                plt.title(variable + ': ' + line_data[variable].long_name)
        else:
            raise ValueError('Invalid first argument; must be a dictionary or None, but given type {}'.format(
                type(variables_to_plot)))

        plt.tight_layout(h_pad=2.0)
        if save_file_path is not None:
            plt.savefig(save_file_path)
            plt.close()
        else:
            plt.show()

    def contour_plots(self, save_file_path=None, variables_to_plot=None):
        '''
        Either saves or displays a series of line plots of 2-D variables along
        trajectory

        NOTE: vertical coordinate is the hybrid level index. 0 is
        top-of-atmosphere

        Parameters
        ----------
        save_file_path: path-like or None
            if None, plot is displayed to screen with plt.show()
            if path-like, must be directory path and name for saving figure
                file format assumed from suffix
        variables_to_plot: list of strings
            list of CAM 3-D variables names to plot
            NOTE: no special behavior if None; should make a default
        '''
        # Set up plot coordinates
        time = cftime.date2num(
            self.data.time.values, units='days since 0001-01-01 00:00:00', calendar='noleap')
        lev_idx = range(0, self.data.lev.size)
        mesh_time, mesh_lev = np.meshgrid(time, lev_idx)

        # Initialize figure
        plt.clf()
        fig = plt.figure()
        fig.set_figheight(12)
        fig.set_figwidth(11)
        cm = plt.get_cmap('viridis')
        num_plots = len(variables_to_plot)
        num_contours = 15

        # Set y-axis limits and direction
        if self.data.lev.positive == 'down':
            y_bot, y_top = (max(lev_idx), min(lev_idx))
        elif self.data.lev.positive == 'up':
            y_bot, y_top = (min(lev_idx), max(lev_idx))
        else:
            raise ValueError("Unexpected direction for lev dimension {}. Expecting 'up' or 'down'".format(
                self.data.lev.positive))

        for idx, variable in enumerate(variables_to_plot):
            var_label = '{} ({})'.format(
                self.data[variable].long_name, self.data[variable].units)
            ax = fig.add_subplot(num_plots, 1, idx + 1)
            ax.set_xlabel('Days')
            ax.set_ylabel('Vertical Level Index')
            ax.set_ylim(y_bot, y_top)
            # (time, lev) -> (lev, time) since lev must be y-axis
            data = np.transpose(self.data[variable].values)
            contour = plt.contourf(mesh_time, mesh_lev, data, num_contours,
                                   cmap=cm)
            plt.colorbar(ax=ax, shrink=0.6, pad=0.02, label=var_label)
            plt.title('{} along Trajectory'.format(variable))

        plt.tight_layout(h_pad=2.0)
        if save_file_path is not None:
            plt.savefig(save_file_path)
            plt.close()
        else:
            plt.show()