"""
Author: Kara Hartig

Plot trajectory paths or climate variables by event or height

Functions:
    trajectory_path_plots: for each event, plot all trajectory paths in North Polar Stereo, colored by initial height
    line_plots_by_event: for each event, plot 2-D climate variables sampled along trajectory paths at all heights
    line_plots_by_trajectory: for each initial trajectory height, plot 2-D climate variables sampled along trajectory path for all events
    contour_plots:  for each event, plot contours of 3-D climate variables interpolated onto the path of a given trajectory
    generate_trajlist: generate a list of paths to trajectory files
    generate_traj2save: generate a dictionary mapping between trajectory file paths and save file paths
    generate_tnum2save: generates a dictionary mapping between integer trajectory numbers and save file paths
"""
# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import os

# Matplotlib imports
import matplotlib.path as mpath
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection

# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# camtrack import
import camtrack as ct

def trajectory_path_plots(trajectory_paths):
    '''
    For each event index from 0 to num_events-1, save North Polar Stereo plot of
    all trajectories in the corresponding trajectory file
    
    Parameters
    ----------
    trajectory_paths: list or dictionary
        if list:
            list of paths to all .traj files to be plot
            all plots will be printed to the screen
        if dict:
            mapping between paths to .traj files and paths to save files
            all plots will be saved at designated save file locations
    '''
    # Parse input argument
    if isinstance(trajectory_paths, dict):
        saving = True
        path_list = list(trajectory_paths.keys())
    elif isinstance(trajectory_paths, list):
        saving = False
        path_list = trajectory_paths
    else:
        raise TypeError('trajectory_paths must be either a dictionary of trajectory : savefile path \
            pairs or a list of trajectory paths, not {}'.format(type(trajectory_paths)))

    # Initialize plot
    if saving:
        # a single figure will be saved and then over-written for each loop
        fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    plt.rcParams.update({'font.size': 14})  # set overall font size
    cm_height = plt.get_cmap('inferno') # colormap for trajectory height
    deg = u'\N{DEGREE SIGN}'

    for traj_path in path_list:
        if saving:
            save_file_path = trajectory_paths[traj_path]
        else:
            # a new figure for each loop to be displayed
            fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw={'projection': ccrs.NorthPolarStereo()})

        trajfile = ct.TrajectoryFile(traj_path)

        # Set map projection
        ax.set_global()
        min_plot_lat = 50 if all(trajfile.data['lat'].values > 50) else min(
            trajfile.data['lat'].values) - 5
        ax.set_extent([-180, 180, min_plot_lat, 90], crs=ccrs.PlateCarree())

        # Add features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.gridlines(color='black', linestyle='dotted')

        # Set circular outer boundary
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        # Plot start point as a blue X
        ax.scatter(trajfile.traj_start['lon'], trajfile.traj_start['lat'],
            s=150, c='tab:blue', marker='X', transform=ccrs.Geodetic(), zorder=4)

        # Set up coloring by height
        n_heights = len(trajfile.traj_start['height'])
        c_height = cm_height(np.linspace(0.2, 0.8, n_heights))

        # Loop over trajectory heights
        for traj_idx,df in trajfile.data.groupby(level=0):
            trajectory = trajfile.get_trajectory(traj_idx, 3)
            max_age = min(trajectory.index.values)
            # Plot trajectory path, colored by initial height
            initial_height = trajectory.loc[0]['height (m)']
            ax.plot(trajectory['lon'].values, trajectory['lat'].values,
                color=c_height[traj_idx-1], label='{:.0f} m'.format(initial_height),
                linewidth=2.5, transform=ccrs.Geodetic(), zorder=1)
            # Mark end point of trajectory
            ax.scatter(trajectory.loc[max_age]['lon'], trajectory.loc[max_age]['lat'],
                s=75, c='tab:gray', marker='o', transform=ccrs.Geodetic(), zorder=3)
            # Mark every 24 hours in age
            daily_traj = trajfile.get_trajectory(traj_idx, age_interval=24)
            ax.scatter(daily_traj['lon'].values, daily_traj['lat'].values,
                s=60, c='tab:blue', marker='+', transform=ccrs.Geodetic(), zorder=2)

        # Add title
        traj_file_name = os.path.basename(traj_path)
        date_string = '{:04.0f}-{:02.0f}-{:02.0f}'.format(trajfile.traj_start.loc[0]['year'], trajfile.traj_start.loc[0]['month'], trajfile.traj_start.loc[0]['day'])
        ax.set(title='{} on {} starting at {:.01f}{}N, {:.01f}{}E'.format(traj_file_name, date_string, trajfile.traj_start.loc[0]['lat'], deg, trajfile.traj_start.loc[0]['lon'], deg))
        ax.legend(loc='upper right')

        if saving:
            fig.savefig(save_file_path)
            print('Finished saving path for {}...'.format(traj_file_name))
        else:
            plt.show()
        ax.clear()
    plt.close()


def trajectory_path_with_wind(trajectory_paths, traj_number, cam_dir, color=None):
    '''
    For the given trajectory paths and trajectory number, generate a plot of the
    trajectory path overlaid by instantaneous horizontal wind vectors and,
    optionally, colored by parcel height

    color = None (single color by initial height) or 'HEIGHT'
    '''
    # Parse input argument
    if isinstance(trajectory_paths, dict):
        saving = True
        path_list = list(trajectory_paths.keys())
    elif isinstance(trajectory_paths, list):
        saving = False
        path_list = trajectory_paths
    else:
        raise TypeError('trajectory_paths must be either a dictionary of trajectory : savefile path \
            pairs or a list of trajectory paths, not {}'.format(type(trajectory_paths)))

    # Initialize plot
    plt.rcParams.update({'font.size': 14})  # set overall font size
    deg = u'\N{DEGREE SIGN}'

    for traj_path in path_list:
        if saving:
            save_file_path = trajectory_paths[traj_path]
        fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw={'projection': ccrs.NorthPolarStereo()})

        trajfile = ct.TrajectoryFile(traj_path)

        # Set map projection
        ax.set_global()
        min_plot_lat = 50 if all(trajfile.data['lat'].values > 50) else min(
            trajfile.data['lat'].values) - 5
        ax.set_extent([-180, 180, min_plot_lat, 90], crs=ccrs.PlateCarree())

        # Add features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.gridlines(color='black', linestyle='dotted')

        # Set circular outer boundary
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Interpolate wind onto path
        winter_file = ct.WinterCAM(cam_dir, trajfile)
        pressure_levels = np.linspace(1.2e5, 300, 40)
        cat = ct.ClimateAlongTrajectory(winter_file, trajfile, traj_number, ['TREFHT'], 'linear')
        cat.add_variable('U', True, pressure_levels)
        cat.add_variable('V', True)
        lats = cat.data['lat'].values
        lons = cat.data['lon'].values
        u_wind = cat.data['U_1D'].values
        v_wind = cat.data['V_1D'].values

        # Plot start point as a black X
        ax.scatter(trajfile.traj_start['lon'], trajfile.traj_start['lat'],
            s=150, c='k', marker='X', transform=ccrs.Geodetic(), zorder=4)

        # Set up coloring scheme
        if color is None:
            cm_height = plt.get_cmap('inferno') # colormap for trajectory height
            n_heights = len(trajfile.traj_start['height'])
            c_height = cm_height(np.linspace(0.2, 0.8, n_heights))
        elif color == 'HEIGHT':
            heights = cat.data['HEIGHT'].values
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(min(heights), max(heights))
        else:
            raise ValueError('color must be None (default; single color based on initial height) or HEIGHT; given {}'.format(color))

        # Loop over trajectory heights
        trajectory = trajfile.get_trajectory(traj_number, 3)
        max_age = min(trajectory.index.values)
        initial_height = trajectory.loc[0]['height (m)']
        # Plot trajectory path
        if color is None:
            ax.plot(trajectory['lon'].values, trajectory['lat'].values,
                    color=c_height[traj_number-1],
                    label='{:.0f} m'.format(initial_height),
                    linewidth=2.5, transform=ccrs.Geodetic(), zorder=1)
        elif color == 'HEIGHT':
            lc = LineCollection(segments, cmap='plasma', norm=norm)
            lc.set(array=heights, label='{:.0f} m'.format(initial_height),
                   linewidth=5, transform=ccrs.Geodetic(), zorder=5)
            line = ax.add_collection(lc)
            cb = fig.colorbar(line, ax=ax, shrink=0.6, pad=0.02)
        # Mark end point of trajectory
        ax.scatter(trajectory.loc[max_age]['lon'], trajectory.loc[max_age]['lat'],
                   s=75, c='tab:gray', marker='o', transform=ccrs.Geodetic(),
                   zorder=3)
        # Add wind vectors
        ax.quiver(lons[::5], lats[::5], u_wind[::5], v_wind[::5],
                  transform=ccrs.PlateCarree(), zorder=2)

        # Add title
        traj_file_name = os.path.basename(traj_path)
        date_string = '{:04.0f}-{:02.0f}-{:02.0f}'.format(trajfile.traj_start.loc[0]['year'], trajfile.traj_start.loc[0]['month'], trajfile.traj_start.loc[0]['day'])
        ax.set(title='{} on {} starting at {:.01f}{}N, {:.01f}{}E'.format(traj_file_name, date_string, trajfile.traj_start.loc[0]['lat'], deg, trajfile.traj_start.loc[0]['lon'], deg))
        ax.legend(loc='upper right')

        if saving:
            fig.savefig(save_file_path)
            print('Finished saving path for {}...'.format(traj_file_name))
        else:
            plt.show()
        plt.close()


def line_plots_by_event(trajectory_paths, cam_variables, other_variables, traj_interp_method, cam_dir, pressure_levels=None):
    '''
    For each event index from 0 to num_events-1, generate line plots of climate
    variables along all trajectories

    Saves 1 .png figure per event. Each figure is a column of subplots, each
    subplot corresponding to a different variable in cam_variables,
    traj_variables, and custom_variables. Lines on each plot are colored by
    initial trajectory height.

    Parameters
    ----------
    num_events: integer
        number of events to generate line plots for
        assuming each .traj file is named 'traj_event<event idx>.traj':
            event index of 2 -> 'traj_event02.traj'
    num_traj: integer
        number of trajectories per .traj file
    cam_variables: list-like of strings
        list of CAM variables to plot
        must correpond to 2-D variables with dimensions (time, lat, lon)
    traj_variables: list-like of strings
        list of variables to plot from trajectory file
        'HEIGHT' is always available; other diagnostic output variables are only
        available if corresponding SETUP.CFG variable in HYSPLIT is set to 1
        when trajectories were generated
    custom_variables: list-like of strings
        list of variables to plot with hard-coded special plotting instructions
        currently supported: 'Net cloud forcing'
    traj_interp_method: 'nearest' or 'linear'
        interpolation method for matching trajectory lat-lon to CAM variables
    traj_dir: path-like or string
        path to directory where trajectory files are stored
    cam_dir: path-like or string
        path to directory where winter CAM files are stored
    save_dir: path-like or string
        path to directory where trajectory plots will be saved
        output file name format is 'traj_plot_event<event idx>.png'
    pressure_levels: array-like of floats
        pressure levels, in Pa, to interpolate onto for variables with a vertical level coordinate
        Only used if requesting 3-D variables interpolated directly onto trajectory path
        Default is None
    '''
    # Parse input argument
    if isinstance(trajectory_paths, dict):
        saving = True
        path_list = list(trajectory_paths.keys())
    elif isinstance(trajectory_paths, list):
        saving = False
        path_list = trajectory_paths
    else:
        raise TypeError('trajectory_paths must be either a dictionary of trajectory : savefile path \
            pairs or a list of trajectory paths, not {}'.format(type(trajectory_paths)))

    # Initialize plot
    var_to_plot = cam_variables + other_variables
    num_plots = len(var_to_plot)
    if saving:
        # a single figure will be saved and then over-written for each loop
        fig, axs = plt.subplots(num_plots, 1, figsize=(8,4*num_plots))
    #plt.rcParams.update({'font.size': 14})  # set overall font size
    cm_height = plt.get_cmap('inferno') # colormap for trajectory height

    for traj_path in path_list:
        if saving:
            save_file_path = trajectory_paths[traj_path]
        else:
            # a new figure for each loop to be displayed
            fig, axs = plt.subplots(num_plots, 1, figsize=(8,4*num_plots))
        traj_file_name = os.path.basename(traj_path)
        print('Starting event {}'.format(traj_file_name))

        # Load all trajectories for the event
        all_trajectories = []
        trajfile = ct.TrajectoryFile(traj_path)
        camfile = ct.WinterCAM(cam_dir, trajfile)
        for traj_idx,df in trajfile.data.groupby(level=0):
            print('  Loading trajectory {}...'.format(traj_idx))
            all_trajectories.append(ct.ClimateAlongTrajectory(camfile, trajfile, traj_idx, cam_variables, traj_interp_method))

        # Set up coloring by height
        n_heights = len(trajfile.traj_start['height'])
        c_height = cm_height(np.linspace(0.2, 0.8, n_heights))

        # Plot all variables
        for var_idx, variable in enumerate(var_to_plot):
            axs[var_idx].set_xlabel('Trajectory Age (hours)')
            if variable == 'Net cloud forcing':
                axs[var_idx].set_ylabel(all_trajectories[0].data['LWCF'].units)
                axs[var_idx].set_title('LWCF + SWCF: Net cloud forcing')
                for traj_idx,traj in enumerate(all_trajectories):
                    time = traj.trajectory.index.values
                    axs[var_idx].plot(time, traj.data['LWCF'].values + traj.data['SWCF'].values, '-o', linewidth=2, c=c_height[traj_idx])
            else:
                if variable[-3:] == '_1D':
                    variable_key = variable[:-3]
                    sample_data = camfile.variable(variable_key)
                else:
                    sample_data = all_trajectories[0].data[variable]
                axs[var_idx].set_ylabel(sample_data.units)
                axs[var_idx].set_title(variable + ': ' + sample_data.long_name)
                for traj_idx,traj in enumerate(all_trajectories):
                    time = traj.trajectory.index.values
                    if variable[-3:] == '_1D':
                        traj.add_variable(variable_key, to_1D=True, pressure_levels=pressure_levels)
                    axs[var_idx].plot(time, traj.data[variable].values, '-o', linewidth=2, c=c_height[traj_idx])
        plt.tight_layout(h_pad=2.0)
        if saving:
            fig.savefig(save_file_path)  ## NEW ##
            print('Finished saving line plots for {}...'.format(traj_file_name))  ## NEW ##
        else:
            plt.show()
        for ax in axs:
            ax.clear()
    plt.close()

def line_plots_by_trajectory(trajectory_list, traj_numbers, cam_variables, other_variables, traj_interp_method, cam_dir, pressure_levels=None):
    '''
    For each trajectory number from 1 to num_traj, generate line plots of
    climate variables across all events.

    Saves 1 .png figure per trajectory number. Each figure is a column of
    subplots, each subplot corresponding to a different variable in
    cam_variables, traj_variables, and custom_variables. Thin lines are from
    individual trajectories, thick lines are averages across all events.

    Parameters
    ----------
    trajectory_list: list-like of path-likes
        list of paths to all .traj files to be included
    traj_numbers: list or dict
        to print plots to screen:
            list of trajectory numbers to plot
        to save plots to file:
            dict mapping trajectory numbers to save file paths
    cam_variables: list-like of strings
        list of CAM variables to plot
        must correpond to 2-D variables with dimensions (time, lat, lon)
    other_variables: list-like of strings
        list of non-CAM variables to plot
        includes custom variables and HYSPLIT diagnostic output variables
        supported custom variables:
            'Net cloud forcing'
        HYSPLIT diagnostic output variables:
            'HEIGHT' is always available
            other diagnostic output variables are only available if
            corresponding SETUP.CFG variable in HYSPLIT was set to 1 when
            trajectories were generated
    traj_interp_method: 'nearest' or 'linear'
        interpolation method for matching trajectory lat-lon to CAM variables
    cam_dir: path-like or string
        path to directory where winter CAM files are stored
    pressure_levels: array-like of floats
        pressure levels, in Pa, to interpolate onto for variables with a vertical level coordinate
        Only used if requesting 3-D variables interpolated directly onto trajectory path
        Default is None
    '''
    # Parse input arguments
    if isinstance(traj_numbers, dict):
        saving = True
        traj_number_list = list(traj_numbers.keys())
    elif isinstance(traj_numbers, list):
        saving = False
        traj_number_list = traj_numbers
    else:
        raise TypeError('traj_numbers must be either a dictionary of trajectory number : savefile path \
            pairs or a list of trajectory numbers, not {}'.format(type(traj_numbers)))

    # Initialize plot
    var_to_plot = cam_variables + other_variables
    num_plots = len(var_to_plot)
    if saving:
        # a single figure will be saved and then over-written for each loop
        fig, axs = plt.subplots(num_plots, 1, figsize=(8,4*num_plots))
    #plt.rcParams.update({'font.size': 14})  # set overall font size
    cm_height = plt.get_cmap('inferno') # colormap for trajectory height

    for traj_number in traj_number_list:
        if saving:
            save_file_path = traj_numbers[traj_number]
        else:
            # a new figure for each loop to be displayed
            fig, axs = plt.subplots(num_plots, 1, figsize=(8,4*num_plots))
        print('Starting trajectory number {}'.format(traj_number))

        # Save all events at same height
        all_events = []
        all_ages = []
        for traj_path in trajectory_list:
            trajfile = ct.TrajectoryFile(traj_path)
            camfile = ct.WinterCAM(cam_dir, trajfile)
            cat = ct.ClimateAlongTrajectory(camfile, trajfile, traj_number, cam_variables, traj_interp_method)
            all_events.append(cat)
            all_ages.append(cat.trajectory.index.values)
        max_age = max(len(age) for age in all_ages)
        
        # Plot all variables
        for var_idx, variable in enumerate(var_to_plot):
            axs[var_idx].set_xlabel('Trajectory Age (hours)')
            sum_all_events = np.ma.empty((len(all_events), max_age))
            sum_all_events.mask = True
            if variable == 'Net cloud forcing':
                sample_data = all_events[0].data['LWCF']
                axs[var_idx].set_ylabel(sample_data.units)
                axs[var_idx].set_title('LWCF + SWCF: Net cloud forcing')
                for ev_idx, ev in enumerate(all_events):
                    time = all_ages[ev_idx]
                    plot_data = ev.data['LWCF'].values + ev.data['SWCF'].values
                    sum_all_events[ev_idx, -len(time):] = plot_data
                    axs[var_idx].plot(time, plot_data, '-', linewidth=0.5, c='lightsteelblue')
            elif variable[-3:] == '_1D':
                variable_key = variable[:-3]
                for ev_idx, ev in enumerate(all_events):
                    time = all_ages[ev_idx]
                    ev.add_variable(variable_key, to_1D=True, pressure_levels=pressure_levels)
                    plot_data = ev.data[variable].values
                    sum_all_events[ev_idx, -len(time):] = plot_data
                    axs[var_idx].plot(time, plot_data, '-', linewidth=0.5, c='lightsteelblue')
                sample_data = ev.data[variable]
                axs[var_idx].set_ylabel(sample_data.units)
                axs[var_idx].set_title(variable + ': ' + sample_data.long_name)
            else:
                sample_data = all_events[0].data[variable]
                axs[var_idx].set_ylabel(sample_data.units)
                axs[var_idx].set_title(variable + ': ' + sample_data.long_name)
                for ev_idx, ev in enumerate(all_events):
                    time = all_ages[ev_idx]
                    plot_data = ev.data[variable].values
                    sum_all_events[ev_idx, -len(time):] = plot_data
                    axs[var_idx].plot(time, plot_data, '-', linewidth=0.5, c='lightsteelblue')
            avg_all_events = sum_all_events.mean(axis=0)
            axs[var_idx].plot(max(all_ages, key=len), avg_all_events, '-', linewidth=2., c='steelblue')

        plt.tight_layout(h_pad=2.0)
        if saving:
            fig.savefig(save_file_path)
            print('Finished saving line plots for trajectory {}...'.format(traj_number))
        else:
            plt.show()
        for ax in axs:
            ax.clear()
    plt.close()


def contour_plots(trajectory_paths, traj_number, cam_variables, pressure_levels, traj_interp_method, cam_dir):
    '''
    For each event, generate contour plots in time and pressure of climate
    variables for a specific trajectory.

    Saves 1 .png figure per event. Each figure is a column of subplots, each
    subplot corresponding to a different variable in cam_variables along the
    trajectory specified by traj_number.

    Parameters
    ----------
    trajectory_paths: list or dictionary
        to print to screen:
            list of paths to all .traj files to be plot
            all plots will be printed to the screen
        to save to file:
            dict mapping between paths to .traj files and paths to save files
            all plots will be saved at designated save file locations
    traj_number: integer
        specifies which trajectory to plot from each event
    cam_variables: list-like of strings
        list of CAM variables to plot
        must correpond to 2-D variables with dimensions (time, lat, lon)
    pressure_levels: array-like of floats
        list of pressures, in Pa, to interpolate onto. Note that the length of
        pressure_levels should be at least equal to the number of hybrid levels
        in CAM, otherwise there will be unnecessary loss of information during
        the interpolation
    traj_interp_method: 'nearest' or 'linear'
        interpolation method for matching trajectory lat-lon to CAM variables
    cam_dir: path-like or string
        path to directory where winter CAM files are stored
    '''
    # Parse input argument
    if isinstance(trajectory_paths, dict):
        saving = True
        path_list = list(trajectory_paths.keys())
    elif isinstance(trajectory_paths, list):
        saving = False
        path_list = trajectory_paths
    else:
        raise TypeError('trajectory_paths must be either a dictionary of trajectory : savefile path \
            pairs or a list of trajectory paths, not {}'.format(type(trajectory_paths)))

    # Initialize plot
    num_plots = len(cam_variables)
    cm = plt.get_cmap('viridis')
    num_contours = 15

    for traj_path in path_list:
        if saving:
            save_file_path = trajectory_paths[traj_path]
        fig, axs = plt.subplots(num_plots, 1, figsize=(8, 6*num_plots))
        cbars = [0] * len(axs)
        traj_file_name = os.path.basename(traj_path)
        print('Starting event {}'.format(traj_file_name))

        # Load trajectory for the event
        trajfile = ct.TrajectoryFile(traj_path)
        camfile = ct.WinterCAM(cam_dir, trajfile)
        cat = ct.ClimateAlongTrajectory(camfile, trajfile, traj_number, cam_variables, traj_interp_method, pressure_levels)
        time = cat.trajectory.index.values
        pres = cat.data.pres.values
        mesh_time, mesh_pres = np.meshgrid(time, pres)

        # Load parcel height in pressure coordinates
        traj_w_height = trajfile.height2pressure(cam_dir, traj_number)
        heights = traj_w_height['pressure']

        for var_idx, variable in enumerate(cam_variables):
            var_label = '{} ({})'.format(cat.data[variable].long_name, cat.data[variable].units)
            axs[var_idx].set(title='{} along Trajectory'.format(variable), xlabel='Trajectory Age (hours)', ylabel='Pressure (Pa)', ylim=(max(pres), min(pres)))
            contour_data = np.transpose(cat.data[variable].values)
            contour = axs[var_idx].contourf(mesh_time, mesh_pres, contour_data, num_contours, cmap=cm)
            axs[var_idx].plot(heights.index.values, heights.values, '-', lw=2, c='black')
            cbars[var_idx] = fig.colorbar(contour, ax=axs[var_idx], shrink=0.6, pad=0.02, label=var_label)

        plt.tight_layout(h_pad=2.0)
        if saving:
            fig.savefig(save_file_path)
            print('Finished saving contour plot for {}...'.format(traj_file_name))
        else:
            plt.show()
        for ax in axs:
            ax.clear()
        for cb in cbars:
            cb.remove()
    plt.close()

def generate_trajlist(num_events, traj_dir):
    '''
    Generate a list of paths to trajectory files numbered 0 to num_events-1

    Assumes trajectory file name format 'traj_eventXX.traj' where XX runs from 00 to num_events-1
    '''
    trajectory_list = []
    for event_ID in range(num_events):
        trajectory_list.append(os.path.join(traj_dir, 'traj_event{:02d}.traj'.format(event_ID)))
    return trajectory_list

def generate_traj2save(num_events, traj_dir, save_dir, save_prefix):
    '''
    Generate a dictionary mapping between trajectory file paths and save file paths

    Assumes trajectory file name format 'traj_eventXX.traj' where XX runs from 00 to num_events-1
    Assumes save file name format save_prefix + '_eventXX.png' where XX runs from 00 to num_events-1
    '''
    traj2save_dict = {}
    for event_ID in range(num_events):
        traj_path = os.path.join(traj_dir, 'traj_event{:02d}.traj'.format(event_ID))
        save_path = os.path.join(save_dir, save_prefix + '_event{:02d}.png'.format(event_ID)) ########
        traj2save_dict[traj_path] = save_path
    return traj2save_dict

def generate_tnum2save(num_traj, save_dir, save_prefix):
    '''
    Generate a dictionary mapping between integer trajectory numbers and save file paths

    Trajectory numbers run from 1 to num_traj
    Assumes save file name format save_prefix + '_trajXX.png' where XX runs from 01 to num_traj
    '''
    tnum2save_dict = {}
    for traj_number in range(1, num_traj + 1):
        save_path = os.path.join(save_dir, save_prefix + '_traj{:02d}.png'.format(traj_number))
        tnum2save_dict[traj_number] = save_path
    return tnum2save_dict
