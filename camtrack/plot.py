"""
Author: Kara Hartig

Plot trajectory paths or climate variables by event or height

Functions:
    trajectory_path_plots: for each event, plot all trajectory paths in North Polar Stereo, colored by initial height
    line_plots_by_event: for each event, plot 2-D climate variables sampled along trajectory paths at all heights
    line_plots_by_trajectory: for each initial trajectory height, plot 2-D climate variables sampled along trajectory path for all events
    contour_plots:  for each event, plot contours of 3-D climate variables interpolated onto the path of a given trajectory
"""

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
        ax.set(title='{} on {} starting at {:.01f}{}N, {:.01f}{}E'.format(traj_file_name, date_string, trajfile.traj_start.loc[0]['lat'], deg, trajfile.traj_start.loc[0]['lon'], deg))NEW ##
        ax.legend(loc='upper right')

        if saving:
            fig.savefig(save_file_path)
            print('Finished saving path for {}...'.format(traj_file_name))
        else:
            plt.show()
        ax.clear()
    plt.close()


def line_plots_by_event(trajectory_paths, cam_variables, other_variables, traj_interp_method, cam_dir):
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
                sample_data = all_trajectories[0].data[variable]
                axs[var_idx].set_ylabel(sample_data.units)
                axs[var_idx].set_title(variable + ': ' + sample_data.long_name)
                for traj_idx,traj in enumerate(all_trajectories):
                    time = traj.trajectory.index.values
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

def line_plots_by_trajectory(trajectory_list, traj_numbers, cam_variables, other_variables, traj_interp_method, cam_dir):
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
        for traj_path in trajectory_list:
            trajfile = ct.TrajectoryFile(traj_path)
            camfile = ct.WinterCAM(cam_dir, trajfile)
            all_events.append(ct.ClimateAlongTrajectory(camfile, trajfile, traj_number, cam_variables, traj_interp_method))
        
        # Plot all variables
        for var_idx, variable in enumerate(var_to_plot):
            axs[var_idx].set_xlabel('Trajectory Age (hours)')
            if variable == 'Net cloud forcing':
                sample_data = all_events[0].data['LWCF']
                axs[var_idx].set_ylabel(sample_data.units)
                axs[var_idx].set_title('LWCF + SWCF: Net cloud forcing')
                for ev_idx,ev in enumerate(all_events):
                    time = ev.trajectory.index.values
                    axs[var_idx].plot(time, ev.data['LWCF'].values + ev.data['SWCF'].values, '-', linewidth=0.5, c='lightsteelblue')
            else:
                sample_data = all_events[0].data[variable]
                axs[var_idx].set_ylabel(sample_data.units)
                axs[var_idx].set_title(variable + ': ' + sample_data.long_name)
                for ev in all_events:
                    time = ev.trajectory.index.values
                    axs[var_idx].plot(time, ev.data[variable].values, '-', linewidth=0.5, c='lightsteelblue')

        plt.tight_layout(h_pad=2.0)
        if saving:
            fig.savefig(save_file_path)
            print('Finished saving line plots for trajectory {}...'.format(traj_number))
        else:
            plt.show()
        for ax in axs:
            ax.clear()
    plt.close()


def contour_plots(num_events, traj_number, cam_variables, pressure_levels, interp_method, traj_dir, cam_dir, save_dir):
    '''
    For each event, generate contour plots in time and pressure of climate
    variables for a specific trajectory.

    Saves 1 .png figure per event. Each figure is a column of subplots, each
    subplot corresponding to a different variable in cam_variables along the
    trajectory specified by traj_number.

    Parameters
    ----------
    num_events: integer
        number of events to generate line plots for
        assuming each .traj file is named 'traj_event<event idx>.traj':
            event index of 2 -> 'traj_event02.traj'
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
    interp_method: 'nearest' or 'linear'
        interpolation method for matching trajectory lat-lon to CAM variables
    traj_dir: path-like or string
        path to directory where trajectory files are stored
    cam_dir: path-like or string
        path to directory where winter CAM files are stored
    save_dir: path-like or string
        path to directory where trajectory plots will be saved
        output file name format is 'traj_plot_event<event idx>.png'
    '''
    for event_ID in range(0, num_events):
        print('Starting event {}'.format(event_ID))
        # Generate save file path
        save_file_path = os.path.join(save_dir, 'contour_plot_event{:02d}.png'.format(event_ID))

        # Load trajectory for the event
        traj_path = os.path.join(traj_dir, 'traj_event{:d}.traj'.format(event_ID))
        trajfile = ct.TrajectoryFile(traj_path)
        camfile = ct.WinterCAM(cam_dir, trajfile)
        cat = ct.ClimateAlongTrajectory(camfile, trajfile, traj_number, cam_variables, pressure_levels, interp_method)
        time = cat.trajectory.index.values
        pres = cat.data.pres.values
        mesh_time, mesh_pres = np.meshgrid(time, pres)

        # Initialize figure
        num_plots = len(cam_variables)
        fig = plt.figure()
        fig.set_figheight(6 * num_plots)
        fig.set_figwidth(8)
        cm = plt.get_cmap('viridis')
        num_contours = 15

        for idx, variable in enumerate(cam_variables):
            var_label = '{} ({})'.format(cat.data[variable].long_name, cat.data[variable].units)
            ax = fig.add_subplot(num_plots, 1, idx + 1)
            ax.set_xlabel('Trajectory Age (hours)')
            ax.set_ylabel('Pressure (Pa)')
            ax.set_ylim(max(pres), min(pres))
            contour_data = np.transpose(cat.data[variable].values)
            contour = plt.contourf(mesh_time, mesh_pres, contour_data, num_contours, cmap=cm)
            plt.colorbar(ax=ax, shrink=0.6, pad=0.02, label=var_label)
            plt.title('{} along Trajectory'.format(variable))

        plt.tight_layout(h_pad=2.0)
        plt.savefig(save_file_path)
        plt.close()

        print('Finished event {}\n'.format(event_ID))