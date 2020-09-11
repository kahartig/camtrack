"""
Author: Kara Hartig

Perform and analyze cluster analysis of trajectories

Classes:
  
Functions:
    shift_origin:  shift start point of a group of traj files onto a common origin
    cluster_paths:  plot trajectory paths by cluster
    cluster_line_plots:  interpolate climate variables onto trajectory paths and plot by cluster
"""

# Standard imports
import numpy as np
import pandas as pd
import os

# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.path as mpath

# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# camtrack import
import camtrack as ct


def shift_origin(trajectory_paths, common_origin, output_dir):
    '''
    Shift all trajectories to a common origin and write out new, shifted
    trajectory files

    Parameters
    ----------
    trajectory_paths: list
        list of paths to all .traj files that must be shifted
    common_origin: dict
        contains 'lat' and 'lon' of common origin onto which
        trajectories will be shifted
    output_dir: str
        directory to write the shifted trajectory files to
        Adds "shifted_" prefix to traj file names before writing to distinguish
        from input files
    '''
    # Convert common origin to lon -180 to +180 if necessary
    if common_origin['lon'] > 180:
        common_origin['lon'] = common_origin['lon'] - 360

    # Read in header
    for traj_name in trajectory_paths:
        # Set up read-in of traj files
        #   by hand because TrajectoryFile sets index, sorts, etc
        traj_columns = ['traj #', 'grid #', 'year', 'month', 'day', 'hour',
                        'minute', 'fhour', 'traj age', 'lat', 'lon', 'height (m)']
        traj_dtypes = {'traj #': int, 'grid #': int, 'year': int, 'month': int, 'day': int, 'hour': int,
                       'minute': int, 'fhour': int, 'traj age': int, 'lat': float, 'lon': float, 'height (m)': float}
        col_widths = [6, 6, 6, 6, 6, 6, 6, 6, 8, 9, 9, 9]
        data_fmt = '%6.d%6.d%6.d%6.d%6.d%6.d%6.d%6.d%8.1f%9.3f%9.3f%9.1f'
        
        # Read through header
        full_header = []
        with open(traj_name, 'r') as file:
            # Header 1
            #   number of meteorological grids used
            header_1 = file.readline()
            full_header.append(header_1)
            header_1 = header_1.strip().split()
            ngrids = int(header_1[0])
            #   loop over each grid
            for i in range(ngrids):
                grid_line = file.readline()
                full_header.append(grid_line)

            # Header 2
            #    col 0: number of different trajectories in file
            #    col 1: direction of trajectory calculation (FORWARD, BACKWARD)
            #    col 2: vertical motion calculation method (OMEGA, THETA, ...)
            header_2 = file.readline()
            full_header.append(header_2)
            header_2 = (header_2.strip()).split()
            ntraj = int(header_2[0])          # number of trajectories
            #   loop over each trajectory
            for i in range(ntraj):
                traj_line = file.readline()
                # change start position (col 5 and 6) to common_origin
                len_before_co = 4*6
                len_end_co = len_before_co + 9*2
                shifted_traj_line = traj_line[:len_before_co] + '{:9.3f}{:9.3f}'.format(common_origin['lat'], common_origin['lon']) + traj_line[len_end_co:]
                full_header.append(shifted_traj_line)

            # Header 3
            #    col 0 - number (n) of diagnostic output variables
            #    col 1+ - label identification of each of n variables (PRESSURE,
            #             AIR_TEMP, ...)
            header_3 = file.readline()
            full_header.append(header_3)
            header_3 = header_3.strip().split()
            diag_var_names = header_3[1:]

        # Adjust expected columns for diagnostic output variables
        for var in diag_var_names:
            col_widths.append(9)
            traj_columns.append(var)
            traj_dtypes[var] = float
            data_fmt = data_fmt + '%9.1f'

        # Read in file in fixed-width format
        #   DO NOT convert longitudes since written file will be input to HYSPLIT
        header_lines = 1 + ngrids + 1 + ntraj + 1
        trajectory = pd.read_fwf(traj_name, widths=col_widths, names=traj_columns, dtype=traj_dtypes, skiprows=header_lines)
        
        # Calculate and apply shift to trajectory lat/lon
        origin_shift = {'lat': common_origin['lat'] - trajectory.loc[0]['lat'],
                        'lon': common_origin['lon'] - trajectory.loc[0]['lon']}
        trajectory['lat'] = trajectory['lat'] + origin_shift['lat']
        trajectory['lon'] = trajectory['lon'] + origin_shift['lon']
        
        # Write to output file
        traj_dir, traj_filename = os.path.split(traj_name)
        new_file = os.path.join(output_dir, 'shifted_' + traj_filename)
        #   write header
        with open(new_file, 'w') as newf:
            newf.writelines(full_header)
        #   write data
        with open(new_file, 'a') as newf:
            np.savetxt(newf, trajectory.values, fmt=data_fmt)
    print('Finished shifting trajs to common origin lat {:.2f}, lon {:.2f}'.format(common_origin['lat'], common_origin['lon']))

def cluster_paths(num_clusters, cluster_dir, save_file_path=None):
    '''
    Plot trajectory paths by cluster

    Parameters
    ----------
    num_clusters: int
        number of clusters, used to load CLUSLIST_? file
    cluster_dir: str
        path to CLUSLIST_? and shifted trajectory files
    save_file_path: str
        if None, print plots to screen
        if str, save plots to full file path provided
    '''
    cluslist_file = os.path.join(cluster_dir, 'CLUSLIST_{}'.format(num_clusters))
    cluslist_cols = ['cluster', 'num in cluster', 'start year', 'start month', 'start day', 'start hour', 'idx+1', 'traj file']
    cluslist_col_widths = [5, 6, 5, 3, 3, 3, 6, 100]
    clusters = pd.read_fwf(cluslist_file, widths=cluslist_col_widths, names=cluslist_cols)

    # Initialize plots
    num_plots = num_clusters + 1
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 10*num_plots), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-100)})
    plt.rcParams.update({'font.size': 14})  # set overall font size

    # Set up circular outer boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    # All trajectories
    axs[0].set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
    # Add features
    axs[0].add_feature(cfeature.LAND)
    axs[0].add_feature(cfeature.COASTLINE)
    axs[0].gridlines(color='black', linestyle='dotted')
    axs[0].set(title='All trajectories')
    axs[0].set_boundary(circle, transform=axs[0].transAxes)
    for traj_name in clusters['traj file']:
        trajfile = ct.TrajectoryFile(os.path.join(cluster_dir, traj_name))
        trajectory = trajfile.get_trajectory(1, 3)
        axs[0].plot(trajectory['lon'].values, trajectory['lat'].values, c='black', linewidth=1., transform=ccrs.Geodetic())

    # By cluster
    for cluster_number in range(1, num_clusters + 1):
        idx_ax = cluster_number
        axs[idx_ax].set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
        axs[idx_ax].add_feature(cfeature.LAND)
        axs[idx_ax].add_feature(cfeature.COASTLINE)
        axs[idx_ax].gridlines(color='black', linestyle='dotted')
        axs[idx_ax].set_boundary(circle, transform=axs[idx_ax].transAxes)
        axs[idx_ax].set(title='Cluster {}'.format(cluster_number))
        # plot all trajectories in cluster:
        for traj_name in clusters.loc[clusters['cluster'] == cluster_number]['traj file']:
            trajfile = ct.TrajectoryFile(os.path.join(cluster_dir, traj_name))
            trajectory = trajfile.get_trajectory(1, 3)
            traj_label = 'Event ' + traj_name[-7:-5]
            axs[idx_ax].plot(trajectory['lon'].values, trajectory['lat'].values, label=traj_label, linewidth=1.5, transform=ccrs.Geodetic(), zorder=1)
        # plot cluster mean:
        cluster_mean_file = os.path.join(cluster_dir, 'C{}mean.tdump'.format(cluster_number))
        cluster_mean = ct.TrajectoryFile(cluster_mean_file)
        cluster_mean_trajectory = cluster_mean.get_trajectory(1, 3)
        axs[idx_ax].plot(cluster_mean_trajectory['lon'].values, cluster_mean_trajectory['lat'].values, '--', label='Cluster mean', c='black', linewidth=3., transform=ccrs.Geodetic(), zorder=2)
        axs[idx_ax].legend(loc='upper right', fontsize=11)

    # Save or print to screen
    if save_file_path is None:
        plt.show()
    else:
        fig.savefig(save_file_path)
    plt.close()

def cluster_line_plots(cluslist, cam_variables, other_variables, traj_interp_method, traj_dir, cam_dir, case_name, save_dir=None, pressure_levels=None):
    '''
    Interpolate climate variables onto trajectory paths and plot by cluster

    To print plots to screen, leave save_dir to default (None).
    To save plots to file, provide path-like to save_dir and plots will be saved
    as 'cluster{}_line_plots.png'.format(cluster_number).
    Each figure is a column of subplots, each subplot corresponding to a
    different variable in cam_variables and other_variables. Colored lines are
    from individual trajectories, black lines are averages across all events in
    cluster.

    Parameters
    ----------
    cluslist: path-like or string
        full path to CLUSLIST_{} file (produced by HYSPLIT) containing all
        trajectory file names and their assigned cluster number
    cam_variables: list-like of strings
        list of CAM variables to plot
        must correpond to 2-D variables with dimensions (time, lat, lon)
    other_variables: list-like of strings
        list of non-CAM variables to plot
        includes custom variables and HYSPLIT diagnostic output variables
        supported custom variables:
            'Net cloud forcing'
            'LWP'
            '<3D>_1D' where <3D> is the variable name of some 3-D variable to
                interpolate directly onto the trajectory path
                Must provide pressure_levels
        HYSPLIT diagnostic output variables:
            'HEIGHT' is always available
            other diagnostic output variables are only available if
            corresponding SETUP.CFG variable in HYSPLIT was set to 1 when
            trajectories were generated
    traj_interp_method: 'nearest' or 'linear'
        interpolation method for matching trajectory lat-lon to CAM variables
    traj_dir: path-like or string
        path to directory where .traj files are stored
    cam_dir: path-like or string
        path to directory where winter CAM files are stored
    case_name: string
        case name of CESM run and prefix of .nc and .arl data files
    save_dir: path-like or string
        if None, print plots to screen
        if path-like or string, directory in which to save plots
        Default is None
    pressure_levels: array-like of floats
        pressure levels, in Pa, to interpolate onto for variables with a
        vertical level coordinate
        Only used if requesting 3-D variables interpolated directly onto
        trajectory path in other_variables
        Default is None
    '''
    var_to_plot = cam_variables + other_variables
    num_plots = len(var_to_plot)

    # Retrieve total number of clusters from CLUSLIST file name
    cluslist_path, cluslist_filename = os.path.split(cluslist)
    total_clusters = int(cluslist_filename.split('_')[1])

    # Traj file info
    shifted_prefix = 'shifted_'
    traj_number = 1

    # Load CLUSLIST file
    total_clusters = cluslist[-1]
    cluslist_cols = ['cluster', 'num in cluster', 'start year', 'start month', 'start day', 'start hour', 'event+1', 'traj file']
    cluslist_col_widths = [5, 6, 5, 3, 3, 3, 6, 100] # final col for .traj file name
    clusters = pd.read_fwf(cluslist, widths=cluslist_col_widths, names=cluslist_cols)

    # Plot by cluster
    for cluster_number in range(1, total_clusters + 1):
        fig, axs = plt.subplots(num_plots, 1, figsize=(8,4*num_plots))

        # Load all trajectories in cluster
        cluster_cats = []
        cluster_names = []
        cluster_ages = []
        for shifted_traj_name in clusters.loc[clusters['cluster'] == cluster_number]['traj file']:
            traj_name = shifted_traj_name.replace(shifted_prefix, '') # strip prefix from traj file name
            trajfile = ct.TrajectoryFile(os.path.join(traj_dir, traj_name))
            camfile = ct.WinterCAM(cam_dir, trajfile, case_name=case_name)
            cat = ct.ClimateAlongTrajectory(camfile, trajfile, traj_number, cam_variables, traj_interp_method)
            cluster_cats.append(cat)
            cluster_names.append(traj_name)
            cluster_ages.append(cat.trajectory.index.values)
        max_age = max(len(age) for age in cluster_ages)

        # Plot all variables
        for var_idx, variable in enumerate(var_to_plot):
            axs[var_idx].set_xlabel('Trajectory Age (hours)')
            sum_all_events = np.ma.empty((len(cluster_cats), max_age))
            sum_all_events.mask = True
            if variable == 'Net cloud forcing':
                for ev_idx, ev in enumerate(cluster_cats):
                    time = cluster_ages[ev_idx]
                    plot_data = ev.data['LWCF'].values + ev.data['SWCF'].values
                    axs[var_idx].plot(time, plot_data, '-', linewidth=2., label=cluster_names[ev_idx])
                    sum_all_events[ev_idx, -len(time):] = plot_data
                sample_data = ev.data['LWCF']
                axs[var_idx].set_ylabel(sample_data.units)
                axs[var_idx].set_title('LWCF + SWCF: Net cloud forcing')
            elif variable[-3:] == '_1D':
                variable_key = variable[:-3]
                for ev_idx, ev in enumerate(cluster_cats):
                    time = cluster_ages[ev_idx]
                    ev.add_variable(variable_key, to_1D=True, pressure_levels=pressure_levels)
                    plot_data = ev.data[variable].values
                    axs[var_idx].plot(time, plot_data, '-', linewidth=2., label=cluster_names[ev_idx])
                    sum_all_events[ev_idx, -len(time):] = plot_data
                sample_data = ev.data[variable]
                axs[var_idx].set_ylabel(sample_data.units)
                axs[var_idx].set_title(variable + ': ' + sample_data.long_name)
            elif variable == 'LWP':
                for ev_idx, ev in enumerate(cluster_cats):
                    time = cluster_ages[ev_idx]
                    ev.add_variable(variable, pressure_levels=pressure_levels)
                    plot_data = ev.data[variable].values
                    sum_all_events[ev_idx, -len(time):] = plot_data
                    axs[var_idx].plot(time, plot_data, '-', linewidth=2., label=cluster_names[ev_idx])
                axs[var_idx].set_ylabel('kg/m^2')
                axs[var_idx].set_title('LWP: Liquid water path (integral of Q)')
            else:
                for ev_idx, ev in enumerate(cluster_cats):
                    time = cluster_ages[ev_idx]
                    plot_data = ev.data[variable].values
                    axs[var_idx].plot(time, plot_data, '-', linewidth=2., label=cluster_names[ev_idx])
                    sum_all_events[ev_idx, -len(time):] = plot_data
                sample_data = ev.data[variable]
                axs[var_idx].set_ylabel(sample_data.units)
                axs[var_idx].set_title(variable + ': ' + sample_data.long_name)
            avg_all_events = sum_all_events.mean(axis=0)
            axs[var_idx].plot(max(cluster_ages, key=len), avg_all_events, '--', linewidth=2., c='black', label='mean of cluster')
            axs[var_idx].legend()

        plt.tight_layout(h_pad=2.0)
        if save_dir is None:
            plt.show()
        else:
            save_file_path = os.path.join(save_dir, 'cluster{}_line_plots.png'.format(cluster_number))
            fig.savefig(save_file_path)
            print('Finished saving line plots for cluster {}...'.format(cluster_number))
        plt.close()