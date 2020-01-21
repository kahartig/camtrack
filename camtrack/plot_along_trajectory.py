# ##############################################################################
# Plot various climate variables along air parcel trajectories
# Authors: Kara Hartig
#     based on code by Naomi Wharton
# ##############################################################################

# Standard imports
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
#import xarray as xr
import pandas as pd
import math
import sys

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

class DataFromTrajectoryFile:
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
    grids: dictionary
        contains information for all grids used: model, year, month, day, hour,
        and fhour
    traj_start: dictionary
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
    data: dictionary
    	trajectory data
    	columns: traj #, grid #, year, month, day, hour, minute, fhour,
    	         traj age, lat, lon, height (m)
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
		# columns: model, year, month, day, hour, forecast_hour
		h1_columns = ['model', 'year', 'month', 'day', 'hour', 'fhour']
		
		# loop over each grid
		grids_list = []
		for i in range(ngrids):
			line = file.readline().strip().split()
			grids_list.append(line)
		self.grids = pd.DataFrame(grids_list, columns=h1_columns)

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
		# columns: year, month, day, hour, lat, lon, height
		h2_columns = ['year', 'month', 'day', 'hour', 'lat', 'lon', 'height']

		# loop over each trajectory
		traj_start_list = []
		for i in range(ntraj):
			line = file.readline().strip().split()
			traj_start_list.append(line)
		self.traj_start = pd.DataFrame(traj_start_list, columns=h2_columns)

		# Header 3
		#    col 0 - number (n) of diagnostic output variables
		#    col 1+ - label identification of each of n variables (PRESSURE, THETA, ...)
		header_3 = file.readline()
		header_3 = header_3.strip().split()
		nvars = int(header_3[0]) # number of diagnostic variables
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
		traj_columns = ['traj #', 'grid #', 'year', 'month', 'day', 'hour', 'minute', 'fhour', 'traj age', 'lat', 'lon', 'height (m)']
		for var in diag_output_vars:
			traj_columns.append(var)
		traj_dtypes = {'traj #': int, 'grid #': int, 'year': int, 'month': int, 'day': int, 'hour': int, 'minute': int, 'fhour': int, 'traj age': int, 'lat': float, 'lon': float, 'height (m)': float}
		traj_skiprow = 1 + ngrids + 1 + ntraj + 1  # skip over header; length depends on number of grids and trajectories

		# read in file as csv
		trajectories = pd.read_csv(path, delim_whitespace=True, header=None, names=traj_columns, index_col=[0,8], dtype=traj_dtypes, skiprows=traj_skiprow)
		trajectories.sort_index(inplace=True)
		print('Starting position for trajectory {}:'.format(trajectory_number))
		print('    {:.2f}N lat, {:.2f}E lon, {:.0f} m above ground'.format(trajectories.loc[(trajectory_number,0)]['lon'],trajectories.loc[(trajectory_number, 0)]['lat'],trajectories.loc[(trajectory_number, 0)]['height (m)']))
		# construct datetime string
		# for whatever reason, specific values pulled from trajectories register as floats rather than int...
		def traj_datetime(row):
			return '00{:02.0f}-{:02.0f}-{:02.0f} {:02.0f}:{:02.0f}:00'.format(row['year'], row['month'], row['day'], row['hour'], row['minute'])
		trajectories['datetime'] = trajectories.apply(traj_datetime, axis=1)
		self.data = trajectories


class SingleTrajectory(object):
    '''
    Class for storage of data from a single trajectory in a .traj file.  
    '''
    def __init__(self, *args):

        # list of longitude values along the trajectory, moving forward
        # in time
        self.lons = args[0]

        # list of latitude values along the trajectory, moving forward
        # in time
        self.lats = args[1]

        # list of pressure values along the trajectory, moving forward
        # in time
        self.pressure = args[2]

        # dict containing lists of indices of the trajectory at several 
        # time intervals
        # format: "everyNhours", where n = 3, 6, 9, or 12
        # usage: lons[indices['every3hours']] gives the longitude values
        #        every three hours along the trajectory
        self.indices = args[3]

class DataForPlotting(object):
    '''
    Class for storage of netcdf data along a trajectory for plotting. 
    '''

    def __init__(self, *args):

        # list of all time values along trajectory as pseudo date objects
        self.stime = args[0]

        # list of all pressure values along trajectory
        self.lev = args[1]

        # list of all time values along trajectory as floats
        self.time = args[2]

        # lists of variable values along trajectory
        self.T = args[3]            # temperature
        self.Q = args[4]            # specific humidity
        self.RELHUM = args[5]       # relative humidity
        self.CLOUD = args[6]        # cloud fraction

        self.lhflx = args[7]        # surface latent heat flux
        self.shflx = args[8]        # surface sensible heat flux

        self.lwcf = args[9]         # longwave cloud forcing
        self.swcf = args[10]        # shortwave cloud forcing

        self.icepath =  args[11]    # total grid-box cloud ice water path
        self.liqpath =  args[12]    # total grid-box cloud liquid water path
        self.waterpath = args[13]   # total grid-box cloud water path (liq + ice)

        self.flds = args[14]        # downwelling long wave flux at surface
        self.flns = args[15]        # net long wave flux at surface
        self.fsds = args[16]        # downwelling solar flux at surface
        self.fsns = args[17]        # net solar flux at surface

        # dict containing descriptions of each variable as given in the netcdf file
        self.variable_descriptions = args[18]

        # dict containing units of each variable as given in the netcdf file
        self.variable_units = args[19]

##########################################################################
def select_trajectory(trajectory_number, all_trajectories):
##########################################################################
    '''
    Pull out data for a single trajectory from a DataFromTrajectoryFile() 
    object.

    Parameters:
    trajectory_number (int): number of desired trajectory within the 
        DataFromTrajectoryFile() object. 

    all_trajectories: a DataFromTrajectoryFile() object containing 
        data from multiple trajectories

    Output:
    a SingleTrajectory() object containing lons, lats, and pressures
        along a single trajectory as well as a means of indexing
        them over time intervals
    '''
    
    # make sure there are enough trajectories in the data
    if (trajectory_number >= all_trajectories.number_of_trajectories):
        sys.exit("Trajectory number inputted is larger than total \
                    number of trajectories. ")

    # select all rows that are from the trajectory of interest
    traj = all_trajectories.data[np.where(all_trajectories.data[:, 0] == trajectory_number)]

    # pull out age, lat, lon, and pressure columns of trajectory
    hours = traj[:, 5]  
    lats = traj[:, 9]
    lons = traj[:, 10]
    pressure = traj[:, 12]

    # reverse data if trajectory direction is backwards
    if all_trajectories.direction == 'BACKWARD':
        hours = hours[::-1]
        lats = lats[::-1]
        lons = lons[::-1]
        pressure = pressure[::-1]
        
    # convert lons to 0 < lon < 360 format (from -180 < lon < 180)
    for negative_lon in np.where(lons[:] < 0):

        # add 360 to any lon <0
        lons[negative_lon] = lons[negative_lon] + 360
    

    # store the indices of traj data at intervals of 3, 6, 12, and 24 hours
    indices = {
        'every3hours': np.squeeze(np.where(hours[:] % 3 == 0)),
        'every6hours': np.squeeze(np.where(hours[:] % 6 == 0)),
        'every12hours': np.squeeze(np.where(hours[:] % 12 == 0)),
        'every24hours': np.squeeze(np.where(hours[:] % 24 == 0))
    }

    return SingleTrajectory(lons, lats, pressure, indices)

##########################################################################
def read_netcdf_files(trajectory, trajectory_number, all_trajectories):
##########################################################################
    '''
    Pull out values of various variables (temp, humidity, etc) at the 
    time and lat/lon coordinates of a single trajectory from several
    netcdf files.

    Parameters:
    trajectory: a SingleTrajectory() object containing lat/lon values 
        along a single trajectory

    trajectory_number (int): number of desired trajectory within the 
        DataFromTrajectoryFile() object. 

    all_trajectories: a DataFromTrajectoryFile() object containing 
        data from multiple trajectories

    Output:
    a DataforPlotting() object containing values of variables 
    along the trajectory 

    '''

    # find date on which trajectory starts
    year = all_trajectories.list_of_trajectories['year'][trajectory_number-1]
    month = all_trajectories.list_of_trajectories['month'][trajectory_number-1]
    day = all_trajectories.list_of_trajectories['day'][trajectory_number-1]
    hour = all_trajectories.list_of_trajectories['hour'][trajectory_number-1]

    # For each year of netcdf data, there are four files (h1, h2, h3, h4) 
    # containing different quantities. We are interested in quantities from
    # all four file types, so we will open them all

    # open h1 file
    nc_f='../CAM-output-for-Arctic-air-suppression/PI/pi_3h.cam.h1.00' + "%02d" % (year) + '-01-01-10800.nc'
    ds1 = xr.open_dataset(nc_f)

    # open h2 file
    nc_f='../CAM-output-for-Arctic-air-suppression/PI/pi_3h.cam.h2.00' + "%02d" % (year) + '-01-01-10800.nc'
    ds2 = xr.open_dataset(nc_f)

    # open h3 file
    nc_f='../CAM-output-for-Arctic-air-suppression/PI/pi_3h.cam.h3.00' + "%02d" % (year) + '-01-01-10800.nc'
    ds3 = xr.open_dataset(nc_f)

    # open h4 file
    nc_f='../CAM-output-for-Arctic-air-suppression/PI/pi_3h.cam.h4.00' + "%02d" % (year) + '-01-01-10800.nc'
    ds4 = xr.open_dataset(nc_f)

    # list of all pressure levels
    lev = np.array(ds1['lev'][:])

    # number of levels
    nlev = len(lev)

    # list of *all* time values
    time = np.array(ds1['time'][:])

    # ------------------------------------------------------------
    # find time indices 
    # - location of trajectory's timeline within larger netcdf data
    # ------------------------------------------------------------
    dates = np.array(ds1['date'][:])

    # string containing starting date of trajectory (eg. '80214')
    date = int(str(int(year))+ "%02d" % (month) + str(int(day))) 

    # find starting index for trajectory, then add to find end index
    t2 = int((int(np.array(np.where(dates[:]==date))[0,0])) + int(hour)/3)
    t1 = int((t2-len(trajectory.lons)/3))

    # list of all time values along trajectory - to be used when plotting
    stime = time[t1:t2]

    # list of all time *indices* along trajectory
    itime = np.array(list(range(t2-t1))) + t1

    # all lat/lon values in the netcdf files
    latT = np.array(ds1['lat'][:])
    lonT = np.array(ds1['lon'][:])

    # ------------------------------------------------------------
    # select variables to pull out of netcdf files
    # ------------------------------------------------------------

    # contour plot variables
    temp = ds4['T']             # temperature
    shumidity = ds1['Q']        # specific humidity
    rhumidity = ds1['RELHUM']   # relative humidity
    cloud = ds1['CLOUD']        # cloud fraction

    contour_vars = [temp, shumidity, rhumidity, cloud]


    # line plot variables
    lhflx = ds2['LHFLX']        # surface latent heat flux
    shflx = ds2['SHFLX']        # surface sensible heat flux
    lwcf = ds2['LWCF']          # longwave cloud forcing
    swcf = ds2['SWCF']          # shortwave cloud forcing
    icepath = ds1['TGCLDIWP']       # total grid-box cloud ice water path
    liqpath = ds1['TGCLDLWP']       # total grid-box cloud liquid water path
    waterpath = ds1['TGCLDCWP']     # total grid-box cloud water path (liq + ice)
    flds = ds2['FLDS']          # downwelling long wave flux at surface
    flns = ds2['FLNS']          # net long wave flux at surface
    fsds = ds2['FSDS']          # downwelling solar flux at surface
    fsns = ds2['FSNS']          # net solar flux at surface

    line_vars = [lhflx, shflx, lwcf, swcf, icepath, liqpath, waterpath, 
                flds, flns, fsds, fsns]

    # dict containing descriptions of each variable
    variable_descriptions= {
        "temp": temp.long_name,
        "shumidity": shumidity.long_name,
        "rhumidity": rhumidity.long_name,
        "cloud": cloud.long_name,
        "lhflx": lhflx.long_name,
        "shflx": shflx.long_name,
        "lwcf": lwcf.long_name,
        "swcf": swcf.long_name,
        "icepath": icepath.long_name,
        "liqpath": liqpath.long_name,
        "waterpath": waterpath.long_name,
        "flds": flds.long_name,
        "flns": flns.long_name,
        "fsds": fsds.long_name,
        "fsns": fsns.long_name
    }

    # dict containing units of each variable
    variable_units= {
        "temp": temp.units,
        "shumidity": shumidity.units,
        "rhumidity": rhumidity.units,
        "cloud": cloud.units,
        "lhflx": lhflx.units,
        "shflx": shflx.units,
        "lwcf": lwcf.units,
        "swcf": swcf.units,
        "icepath": icepath.units,
        "liqpath": liqpath.units,
        "waterpath": waterpath.units,
        "flds": flds.units,
        "flns": flns.units,
        "fsds": fsds.units,
        "fsns": fsns.units
    }

    # number of variables to contour plot
    ncont = len(contour_vars)

    # number of variables to line plot
    nline = len(line_vars)

    # netcdf files only have points every 3 hours, so we 
    # will pull out the trajectory values only every 3 hours as well
    every3hours = trajectory.indices['every3hours']
    number_of_time_intervals = len(every3hours)

    # arrays (currently empty) to contain data for plotting
    contour_data = np.empty(shape=( ncont, 
                                    nlev, 
                                    number_of_time_intervals))

    line_data = np.empty(shape=(nline, 
                                number_of_time_intervals))

    # ------------------------------------------------------------
    # linear interpolation
    # ------------------------------------------------------------
    if linear_interpolation:

        # all lat/lon points in netcdf data
        points = (latT, lonT)

        for t, lat, lon, i in zip(itime, trajectory.lats[every3hours], trajectory.lons[every3hours], range(number_of_time_intervals)):

            # convert longitude format to 0-360 E
            if lon < 0:
                lon = lon + 360

            # interpolate values for line plots
            for var, k in zip(line_vars, range(nline)):
                # pull out all values at particular time
                values = np.array(var[t, :, :])
                
                # create interpolator function
                interp_func = interpolator(points, values, method='linear')
                
                # calculate interpolated data point
                out = float(interp_func((lat, lon)))

                # store data point
                line_data[k, i] = out

            # interpolate at each of the 26 pressure levels for contour plots
            for level in range(len(lev)):

                for var, j in zip(contour_vars, range(ncont)):
                    # pull out all values at particular time and level
                    values = np.array(var[t, level, :, :])

                    # create interpolator function
                    interp_func = interpolator(points, values, method='linear')  
                    
                    # calculate interpolated data point
                    out = float(interp_func((lat, lon)))     # output value
                    
                    # store data point
                    contour_data[j, level, i] = out

    # ------------------------------------------------------------
    # data from closest lat/lon value - don't use if using linear interpolation
    # currently set to run if linear interpolation is turned off
    # ------------------------------------------------------------
    if not linear_interpolation:

        # find indices for lat/lon to use to find matching data in T
        ilat = []
        ilon = []

        for lat,lon in zip(trajectory.lats[every3hours],trajectory.lons[every3hours]):
            
            # index of closest lat value
            i = (np.abs(latT - lat)).argmin()        
            
            # index of closest lon value
            j = (np.abs(lonT - lon)).argmin()        
            
            ilat.append(i)
            ilon.append(j)


        # this contains the indices of the data (time, lat, lon) along the trajectory
        indices = np.transpose([itime,ilat,ilon])

        # store in arrays
        for index, j in zip(indices, range(len(indices))):

            # contour plot variables
            for var, i in zip(contour_vars, range(ncont)):
                contour_data[i, :, j] = var[index[0],:,index[1],index[2]]

            # line plot variables
            for var, i in zip(line_vars, range(nline)):
                line_data[i, j] = var[index[0],index[1],index[2]]

    # final plotting data
    T_data = contour_data[0]
    Q_data = contour_data[1]
    RELHUM_data = contour_data[2]
    CLOUD_data = contour_data[3]

    lhflx_data = line_data[0]       # surface latent heat flux
    shflx_data = line_data[1]       # surface sensible heat flux

    lwcf_data = line_data[2]        # longwave cloud forcing
    swcf_data = line_data[3]        # shortwave cloud forcing

    icepath_data =  line_data[4]    # total grid-box cloud ice water path
    liqpath_data =  line_data[5]    # total grid-box cloud liquid water path
    waterpath_data = line_data[6]   # total grid-box cloud water path (liq + ice)

    flds_data = line_data[7]        # downwelling long wave flux at surface
    flns_data = line_data[8]        # net long wave flux at surface
    fsds_data = line_data[9]        # downwelling solar flux at surface
    fsns_data = line_data[10]       # net solar flux at surface


    # time data pulled from xarray is a cftime object, but this method
    #   stores time data as floats, which is more useful for plotting
    year = all_trajectories.list_of_trajectories['year'][trajectory_number-1] 
 
    nc_f='../CAM-output-for-Arctic-air-suppression/PI/pi_3h.cam.h1.00' + "%02d" % (year) + '-01-01-10800.nc'
    ncfile = Dataset(nc_f, 'r');

    # times along trajectory as floats
    time = ncfile.variables['time'][t1:t2]



    return DataForPlotting(
            stime,
            lev,
            time,
            T_data,
            Q_data,
            RELHUM_data,
            CLOUD_data,
            lhflx_data,
            shflx_data,
            lwcf_data,
            swcf_data,
            icepath_data,
            liqpath_data,
            waterpath_data,
            flds_data,
            flns_data,
            fsds_data,
            fsns_data,
            variable_descriptions,
            variable_units
            )

##########################################################################
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
    TIME,LEV = np.meshgrid(plotting_data.time, plotting_data.lev)

    # make date labels for x-axis
    idays = np.squeeze(np.where(time[:] % 1 == 0))  # indices

    locs = time[idays]  # locations for x-axis labels

    days = plotting_data.stime[idays] # dates as time objects 
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
        ax = fig.add_subplot(4,1,1)
        ax.set_ylim([1000,0])
        #ax.set_xlabel("Days")
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
        ax = fig.add_subplot(4,1,2)
        ax.set_ylim([1000,0])
        #ax.set_xlabel("Days")
        ax.set_ylabel("Pressure")

        # make the contours:
        plt.contourf(TIME, LEV, plotting_data.Q, 15, 
                        cmap=plt.cm.jet,
                        vmin=0,
                        vmax=.0032)

        # Add a color bar
        plt.colorbar(ax=ax, shrink=.62, pad=0.02, label='specific humidity (kg/kg)')

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

        ax = fig.add_subplot(4,1,3)
        ax.set_ylim([1000,0])
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

        ax = fig.add_subplot(4,1,4)
        ax.set_ylim([1000,0])
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
            h=plt.gcf();
            h.savefig(contour_filename);

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
        ax = fig.add_subplot(4,1,1)
        
        units = plotting_data.variable_units['lhflx']
        ax.set_xlabel("Days")
        ax.set_ylabel(units)

        # points
        plt.scatter(time, plotting_data.lhflx, s = 5)    # surface latent heat flux
        plt.scatter(time, plotting_data.shflx, s = 5)    # surface sensible heat flux

        # lines
        plt.plot(time, plotting_data.lhflx, label=plotting_data.variable_descriptions['lhflx'])
        plt.plot(time, plotting_data.shflx, label=plotting_data.variable_descriptions['shflx'])

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
        ax = fig.add_subplot(4,1,2)
        
        units = plotting_data.variable_units['lwcf']
        ax.set_xlabel("Days")
        ax.set_ylabel(units)

        # points
        plt.scatter(time, plotting_data.lwcf, s = 5)     # longwave cloud forcing
        plt.scatter(time, plotting_data.swcf, s = 5)     # shortwave cloud forcing
        plt.scatter(time, plotting_data.swcf + plotting_data.lwcf, s = 5) # net cloud forcing

        # line
        plt.plot(time, plotting_data.lwcf, label=plotting_data.variable_descriptions['lwcf'])
        plt.plot(time, plotting_data.swcf, label=plotting_data.variable_descriptions['swcf'])
        plt.plot(time, plotting_data.swcf + plotting_data.lwcf, label = 'Net cloud forcing')

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
        ax = fig.add_subplot(4,1,3)

        units = plotting_data.variable_units['icepath']
        ax.set_ylabel(units)
        ax.set_xlabel("Days")

        # points
        plt.scatter(time, plotting_data.icepath, s = 5)      # total grid-box cloud ice water path
        plt.scatter(time, plotting_data.liqpath, s = 5)      # total grid-box cloud liquid water path
        plt.scatter(time, plotting_data.waterpath, s = 5)    # total grid-box cloud water path (liq + ice)

        # lines
        plt.plot(time, plotting_data.icepath, label=plotting_data.variable_descriptions['icepath'])
        plt.plot(time, plotting_data.liqpath, label=plotting_data.variable_descriptions['liqpath'])
        plt.plot(time, plotting_data.waterpath, label=plotting_data.variable_descriptions['waterpath'])

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
        ax = fig.add_subplot(4,1,4)
        
        units = plotting_data.variable_units['flds']
        ax.set_ylabel(units)
        ax.set_xlabel("Days")       

        # points
        plt.scatter(time, plotting_data.flds, s = 5)             # downwelling long wave flux at surface
        plt.scatter(time, plotting_data.flns - plotting_data.flds, s = 5) # net long wave flux at surface
        plt.scatter(time, plotting_data.fsds, s = 5)             # downwelling solar flux at surface
        plt.scatter(time, plotting_data.fsns - plotting_data.fsds, s = 5)   # net solar flux at surface

        # lines
        plt.plot(time, plotting_data.flds, label=plotting_data.variable_descriptions['flds'])
        plt.plot(time, plotting_data.flns - plotting_data.flds, label='Upwelling longwave flux at surface')
        plt.plot(time, plotting_data.fsds, label=plotting_data.variable_descriptions['fsds'])
        plt.plot(time, plotting_data.fsns - plotting_data.fsds, label='Upwelling solar flux at surface')

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
            h=plt.gcf();
            h.savefig(line_plots_filename, bbox_inches="tight");

        plt.close()

    # ------------------------------------------------------------
    # Plot lat/lon trajectory on map
    # ------------------------------------------------------------
    if plot_map:
        # create figure
        plt.clf()
        fig = plt.figure(1, figsize=(8,8))

        # create axes
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        ax.set_global()
        ax.set_extent([0.000001, 360, 55, 90], crs=ccrs.PlateCarree())

        # convert longitude from -180-180 scale to to 360 degree
        #for i in np.where(lons[:] > 0):
        #    lons[i]=lons[i]-360

        plt.plot(trajectory.lons[every3hours],trajectory.lats[every3hours], 
                    color='black', 
                    transform=ccrs.PlateCarree(),
                    #alpha=.8,          # make slightly transparent
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
            at_x, at_y = ax.projection.transform_point(lon, lat, src_crs=ccrs.PlateCarree())

            # annotate each day with date
            plt.annotate(label, 
                        xy=(at_x, at_y),                # location of point to be labeled
                        xytext=(15, -10),               # offset distance from point for label
                        textcoords='offset points',
                        color='black', 
                        backgroundcolor='white', 
                        size=6, 
                        arrowprops=dict(arrowstyle='-', color='white', linewidth=1.0))

        # plot all points on map
        plt.scatter(trajectory.lons[trajectory.indices['every12hours']],trajectory.lats[trajectory.indices['every12hours']], 
                    transform=ccrs.PlateCarree(), 
                    c=trajectory.pressure[trajectory.indices['every12hours']],       # set color of point based on pressure
                    cmap='jet_r',
                    s=25,                           # label text size
                    zorder=10,                      # put above plot line
                    edgecolor='black',
                    linewidth=.8)
        # title
        plt.title('Trajectory for air mass starting at height 10m')

        # make colorbar
        cbar = plt.colorbar(ax=ax, shrink=.5, pad=0.1, label='pressure of air mass (hPa)')
        cbar.ax.invert_yaxis() 

        # save as pdf
        if save_map_as_pdf:
            h=plt.gcf();
            h.savefig(map_filename);


        plt.close()


# path for .traj file
trajectory_filepath = 'Data/sample_traj.traj'


# read in all data from the .traj file and store it
all_trajectories_from_traj_file = read_in_trajectory_file(trajectory_filepath)

# number of desired trajectory within the .traj file
trajectory_number = 1

# lat/lon data for a single trajectory
single_trajectory = select_trajectory(trajectory_number, 
                                        all_trajectories_from_traj_file)

# pull out values of various quantities along the lat/lon/time path 
# of the desired trajectory
data_ready_for_plotting = read_netcdf_files(single_trajectory, 
                                            trajectory_number, 
                                            all_trajectories_from_traj_file)

# plot data for a single trajectory
# produces contour plots, line plots, and a map of overall lat/lon trajectory
make_plots(data_ready_for_plotting, single_trajectory)