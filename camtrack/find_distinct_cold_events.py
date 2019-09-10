# Reads CAM output in netcdf format and finds distinct cold events.
# Writes a HYSPLIT CONTROL file for each cold event
# To run, run python3 find_distinct_cold_events.py WIDX where WIDX is
# the index of the winter of interest (07-08 is 0, 08-09 is 1, etc)

import numpy as np
from netCDF4 import Dataset
import os
import datetime
import sys

#########################################################
# PARAMETERS

# FILE SYSTEM PARAMETERS - set as appropriate for your file system
# location of netCDF data files
netcdf_data_file_path = "/n/tzipermanfs2/CAM-output-for-Arctic-air-suppression/PI"

# names of netCDF data files
netcdf_data_file_names = ['pi_3h.cam.h4.0007-01-01-10800.nc',
                          'pi_3h.cam.h4.0008-01-01-10800.nc',
                          'pi_3h.cam.h4.0009-01-01-10800.nc',
                          'pi_3h.cam.h4.0010-01-01-10800.nc',
                          'pi_3h.cam.h4.0011-01-01-10800.nc',
                          'pi_3h.cam.h4.0012-01-01-10800.nc',
                          'pi_3h.cam.h4.0013-01-01-10800.nc',
                          'pi_3h.cam.h4.0014-01-01-10800.nc',
                          'pi_3h.cam.h4.0015-01-01-10800.nc',
                          'pi_3h.cam.h4.0016-01-01-10800.nc']

# location of ARL data files
arl_data_file_path = '/n/tzipermanfs2/psingh/converted_cam_data/'

# names of ARL data files
arl_data_file_names = ['pi_3h_07.arl','pi_3h_08.arl','pi_3h_09.arl','pi_3h_10.arl','pi_3h_11.arl',
                       'pi_3h_12.arl','pi_3h_13.arl','pi_3h_14.arl','pi_3h_15.arl','pi_3h_16.arl']

# location of HYSPLIT working directory
hysplit_working_directory_path = '/n/tzipermanfs2/psingh/hysplit-924/working/'


# COLD EVENT PARAMTERS - used during search for cold events
# Lowest latitude for cold events in degrees - look for cold events at all latitudes above this
minimum_latitude_for_cold_events = 45

# Bounds on longitude of region in which to search for cold events, in degrees on 0 to 360 scale
minimum_longitude_for_cold_events = 180 
maximum_longitude_for_cold_events = 310

# minimum land fraction to consider
minimum_land_fraction_for_cold_events = 0.9

# minimum separation requirements for cold events. All three separation criteria must be met.
# Time separation in days, lat/lon separations in degrees
minimum_time_separation_of_cold_events = 5
minimum_latitude_separation_of_cold_events = 5
minimum_longitude_separation_of_cold_events = 5

# the number of cold events to find and print CONTROL files for. The coldest distinct events are chosen
number_of_cold_events_to_find = 5

# HYSPLIT parameters - used to write the CONTROL file for HYSPLIT
# heights at which to begin backtracking trajectories in meters
trajectory_start_heights = [10, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

# duration of backtracking required in hours
backtrack_duration = 336
#########################################################

# which winter to search (07-08 is 0, 08-09 is 1 etc), taken from command-line argument
index_of_winter_to_search = int(sys.argv[1])

# A unique identifying string that is put into the CONTROL files output from this run of the program,
# taken from command-line argument
unique_id_for_this_run = sys.argv[1]

################################################################
def print_CONTROL_file_for_HYSPLIT(uid, ev):
################################################################
    """
    Prints out the CONTROL file that HYSPLIT uses to set up a backtracking run. Creates a subdirectory 
    under the current folder called control_files in which it writes the CONTROL file. Also creates a 
    subdirectory under the HYSPLIT working directory called trajs in which the trajectory file will be
    placed when HYSPLIT is run with this CONTROL file.

    Parameters:
    uid: A unique identifier string for the particular CONTROL file. The output filename will be 
    CONTROL_uid. This will need to be changed to CONTROL and the file be placed in the HYSPLIT 
    working directory prior to running HYSPLIT.
    ev: A Python dict containing details of the particular cold event for which the CONTROL file
    is being generated. ev must contain at least the following:
        winter_idx = index of the winter during which the event takes place (07-08 is 0, 08-09 is 1 etc)
        time = the time of the event as a real number of days since 0001-00-00 (fractional part for hours)
        lat = latitude coordinate of the event as a real number of degrees on -90 to 90 scale
        lon = longitude coordinate of the event as a real number of degrees on 0 to 360 scale
    """
    if not os.path.exists("./control_files"):
        os.makedirs("./control_files")
    if not os.path.exists("/n/tzipermanfs2/psingh/hysplit-924/working/trajs"):
        os.makedirs("/n/tzipermanfs2/psingh/hysplit-924/working/trajs")
    with open('control_files/CONTROL_' + str(uid), 'w') as f:
        day = ev['time']
        la = ev['lat']
        lo = ev['lon']
        t = datetime.datetime.combine(datetime.datetime.fromordinal(int(day // 1)), datetime.time(int((day % 1)*24)))                
        if lo > 180:
            lo = lo - 360
        # Print the CONTROL file:
        # Start time:
        f.write("%02d %02d %02d %02d\n" % (t.year, t.month, t.day, t.hour))
        # No. of start positions:
        f.write(str(len(trajectory_start_heights)) + "\n")
        # Start positions:
        for ht in trajectory_start_heights:
            f.write("%.1f %.1f %.1f\n" % (la, lo, ht))
        # Duration of backtrack in hours:
        f.write(str(-1 * backtrack_duration) + "\n")
        # Vertical motion option:
        f.write("0\n")
        # Top of model:
        f.write("10000.0\n")
        # No. of input file sources:
        f.write("2\n")
        # Input file 1 path:
        f.write(arl_data_file_path + "\n")
        # Input file 1 name:
        f.write(arl_data_file_names[ev['winter_idx']] + "\n")
        # Input file 2 path:
        f.write(arl_data_file_path + "\n")
        # Input file 2 name:
        f.write(arl_data_file_names[ev['winter_idx'] + 1] + "\n")
        # Output trajectory file path:
        f.write("./trajs/\n")
        # Output trajectory file name:
        f.write("traj_" + str(uid) + "\n")
    
# get two input files - use December of first and Jan/Feb of second
nc0 = Dataset(os.path.join(netcdf_data_file_path, netcdf_data_file_names[index_of_winter_to_search]))
nc1 = Dataset(os.path.join(netcdf_data_file_path, netcdf_data_file_names[index_of_winter_to_search + 1]))

# get lat-lon grids
latT = nc0.variables['lat'][:]
lonT = nc0.variables['lon'][:]

# combine data from two files along the time axis
time_l      = np.append(nc0.variables["time"][:].data, 
                        nc1.variables["time"][:].data, axis = 0)
temp_2m_l   = np.append(nc0.variables["TREFHT"][:].data, 
                        nc1.variables["TREFHT"][:].data, axis = 0)
landfrac_l  = np.append(nc0.variables["LANDFRAC"][:].data, 
                        nc1.variables["LANDFRAC"][:].data, axis = 0)
del nc0
del nc1
print("\tSubset the data", flush = True)

# define North America as between 50 and 180 degrees West
min_lon_idx = min(np.where(np.logical_and(lonT >= minimum_longitude_for_cold_events, 
                                          lonT <= maximum_longitude_for_cold_events))[0])
max_lon_idx = max(np.where(np.logical_and(lonT >= minimum_longitude_for_cold_events, 
                                          lonT <= maximum_longitude_for_cold_events))[0])

min_lat_idx = min(np.where(latT >= minimum_latitude_for_cold_events)[0])
max_lat_idx = max(np.where(latT >= minimum_latitude_for_cold_events)[0])

# define winter as Dec-Jan-Feb 
min_time = datetime.datetime.toordinal(datetime.datetime(year = 7 + index_of_winter_to_search, 
                                                         month = 12, day = 1))
max_time = datetime.datetime.toordinal(datetime.datetime(year = 8 + index_of_winter_to_search, 
                                                         month = 3, day = 1))
min_time_idx = min(np.where(np.logical_and(time_l >= min_time, time_l < max_time))[0])
max_time_idx = max(np.where(np.logical_and(time_l >= min_time, time_l < max_time))[0])
print("\tFound regions of interest", flush = True)

# subset the data
time     = time_l[min_time_idx:max_time_idx]
lat      = latT[min_lat_idx:max_lat_idx]
lon      = lonT[min_lon_idx:max_lon_idx]
temp_2m  = temp_2m_l[min_time_idx:max_time_idx, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx]
landfrac = landfrac_l[min_time_idx:max_time_idx, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx]
print("\tSubset the data", flush = True)

# only consider areas greater than land_threshold land fraction:
temp_2m_land = np.where(landfrac < minimum_land_fraction_for_cold_events, np.nan, temp_2m)

# order all points by temperature
sorted_idx = np.unravel_index(temp_2m_land.argsort(axis=None), temp_2m_land.shape)
print("\tOrdered the data by temperature", flush = True)


cold_events = []
# {winter_idx, time_idx, time, lat_idx, lat, lon_idx, lon, temp}

# find the coldest distinct events
curidx = 0;
numfound = 0;
while numfound < number_of_cold_events_to_find:
    day_i = time[sorted_idx[0][curidx]]
    lat_i = lat[sorted_idx[1][curidx]]
    lon_i = lon[sorted_idx[2][curidx]]
    distinct = True
    for found in cold_events:
        if abs(day_i - found['time']) < minimum_time_separation_of_cold_events
        or abs(lat_i - found['lat']) < minimum_latitude_separation_of_cold_events 
        or abs(lon_i - found['lon']) < minimum_longitude_separation_of_cold_events:
            distinct = False
    if not distinct:
        curidx = curidx + 1
        continue
    cold_events.append(dict(winter_idx = index_of_winter_to_search,
                            time_idx = sorted_idx[0][curidx],
                            time = day_i,
                            lat_idx = sorted_idx[1][curidx],
                            lat = lat_i,
                            lon_idx = sorted_idx[2][curidx],
                            lon = lon_i,
                            temp = temp_2m_land[sorted_idx][curidx]))
    curidx = curidx + 1
    numfound = numfound + 1

# print the CONTROL files
num_printed = 0     
for ev in cold_events:
    print_CONTROL_file_for_HYSPLIT(unique_id_for_this_run + '_' + str(num_printed), ev)
    num_printed = num_printed + 1

