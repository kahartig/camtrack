"""
Author: Kara Hartig

Perform and analyze cluster analysis of trajectories

Classes:
  
Functions:
    shift_origin: shift start point of a group of traj files onto a common origin
"""

# Standard imports
import numpy as np
import pandas as pd
import os

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