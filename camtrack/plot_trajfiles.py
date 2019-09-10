# Reads in and plots a trajectory file
# Based on read_trajectory.py by Naomi Wharton and rotate_trajfiles.py by Pratap Singh
#
# Last modified by Kara Hartig on 26 July 2019

# Standard Imports
import numpy as np
from netCDF4 import Dataset
import os
#import datetime
import sys
#import time
#import math
#import xarray as xr

# matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.path as mpath

# cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature

# pandas imports
import pandas as pd

##########################################################################
##################              PARAMETERS               #################
##########################################################################
if len(sys.argv) != 4:
    print('Usage: >>python3 plot_trajfiles.py [trajectory number] [in-file path] [out-file path]')
    print('    trajectory number: numerical identifier of trajectory to plot')
    print('    in-file path: path and filename for input .traj file from HYSPLIT')
    print('    out-file path: path and filename for .pdf of trajectory plot')
    sys.exit()
else:
    trajectory_number = int(sys.argv[1])
    path = str(sys.argv[2])
    map_filename = str(sys.argv[3])
#path = '/Users/karahartig/Documents/altering_Pratap_scripts/no_rotation/traj_0708_test7'
#save_map_as_pdf = True
#trajectory_number = 3
#map_filename = '/Users/karahartig/Documents/altering_Pratap_scripts/no_rotation/trajplot_test7_t' + str(trajectory_number) + '.pdf'

##########################################################################
##################                HEADERS                #################
##########################################################################
f = open(path, 'r')

# ------------------------------------------------------------
# HEADER 1: 
# Number of meteorological grids used in calculation
# ------------------------------------------------------------
h1 = f.readline()
h1 = (h1.strip()).split()
ngrids = int(h1[0])         # number of grids

# read in list of grids used
# columns: model, year, month, day, hour, forecast_hour
h1_columns = ['model', 'year', 'month', 'day', 'hour', 'fhour']
h1_dtypes = [str, int, int, int, int, int]

# loop over each grid
for i in range(ngrids):
    line = f.readline().strip().split()
    if i < 1:
        grids = pd.DataFrame(dict(zip(h1_columns, line)), index=[1])
    else:
        grids.append(dict(zip(h1_columns, line)), index=[i+1])
grids = grids.astype(dtype=dict(zip(h1_columns, h1_dtypes)))

# ------------------------------------------------------------
# HEADER 2: 
# 0 - number of different trajectories in file
# 1 - direction of trajectory calculation (FORWARD, BACKWARD) 
# 2 - vertical motion calculation method (OMEGA, THETA, ...)  
# ------------------------------------------------------------

h2 = f.readline()
h2 = (h2.strip()).split()
ntraj = int(h2[0])          # number of trajectories
direction = h2[1]           # direction of trajectories
vert_motion = h2[2]         # vertical motion calculation method

# read in list of trajectories
# columns: year, month, day, hour, lat, lon, height
traj_list = np.array([], dtype=[('year',int),
                                ('month',int),
                                ('day',int),
                                ('hour',int),
                                ('lat',float),
                                ('lon',float),
                                ('height',float)])
# loop over each trajectory
for i in range(ntraj):
        line = f.readline().strip().split()
        line = np.array([tuple(line)], dtype=traj_list.dtype)
        traj_list = np.append(traj_list, line)

# ------------------------------------------------------------
# HEADER 3: 
# 0 - number (n) of diagnostic output variables
# n - label identification of each variable (PRESSURE, THETA, ...)
# ------------------------------------------------------------
h3 = f.readline()
h3 = h3.strip().split()
nvars = int(h3[0])      # number of extra variables
diag_output_vars = h3[1:] # list of diagnostic output variable names

f.close()


##########################################################################
##################              TRAJECTORIES             #################
##########################################################################
# data columns:
# 0 trajectory number, 
# 1 grid number
# 2 year [calendar date, year part only]
# 3 month [calendar date, month part only]
# 4 day [calendar date, day part only]
# 5 hour [calendar time, hour part only]
# 6 minute [calendar time, minute part only]
# 7 forecast hour [if .arl file had times appended one-by-one, this will always be 0]
# 8 age [hours since start of trajectory; negative for backwards trajectory] 
# 9 lats 
# 10 lons
# 11 height
# ... and then any additional diagnostic output variables
# ------------------------------------------------------------

# trajectory file information
traj_columns = ['traj #', 'grid #', 'year', 'month', 'day', 'hour', 'minute', 'fhour', 'traj age', 'lat',
                'lon', 'height (m)']
for var in diag_output_vars:
    traj_columns.append(var)
traj_dtypes = {'traj #': int, 'grid #': int, 'year': int, 'month': int, 'day': int, 'hour': int, 'minute': int,
               'fhour': int, 'traj age': int, 'lat': float, 'lon': float, 'height (m)': float}
traj_skiprow = 1 + ngrids + 1 + ntraj + 1  # skip over header; length depends on number of grids and trajectories


# read in file as csv
trajectories = pd.read_csv(path, delim_whitespace=True, header=None, names=traj_columns, index_col=[0,8],
                               dtype=traj_dtypes, skiprows=traj_skiprow)
trajectories.sort_index(inplace=True)

print('Starting position for trajectory {}:'.format(trajectory_number))
print('    {:.2f}N lat, {:.2f}E lon, {:.0f} m above ground'.format(trajectories.loc[(trajectory_number,0)]['lon'],trajectories.loc[(trajectory_number, 0)]['lat'],trajectories.loc[(trajectory_number, 0)]['height (m)']))

# Construct datetime string
# for whatever reason, specific values pulled from trajectories register as floats rather than int...
def traj_datetime(row):
    return '00{:02.0f}-{:02.0f}-{:02.0f} {:02.0f}:{:02.0f}:00'.format(row['year'], row['month'], row['day'],
                                                                      row['hour'], row['minute'])
trajectories['datetime'] = trajectories.apply(traj_datetime, axis=1)


##########################################################################
##################                PLOTTING               #################
##########################################################################
plt.clf()

# Choose trajectory
traj = trajectories.loc[trajectory_number]
# Subsets every __ hours
every3hours = traj.iloc[traj.index % 3 == 0]
every12hours = traj.iloc[traj.index % 12 == 0]
every24hours = traj.iloc[traj.index % 24 == 0]

# set map projection
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_global()
ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
#ax.set_extent([0.000001, 360, 50, 90], crs=ccrs.PlateCarree())

# add continents and coasts
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
#ax.stock_img()

# add gridlines
ax.gridlines(color="black", linestyle="dotted")

# title
plt.title('Trajectory for air mass starting at height {:.0f}m'.format(traj.loc[0]['height (m)']))

# create colorscale for trajectory scatter plot
cm = plt.get_cmap('inferno') 

# plot trajectory path in black
plt.plot(every3hours['lon'],every3hours['lat'], 
         color='black', 
         transform=ccrs.Geodetic(),
         #alpha=.8,          # make slightly transparent
         zorder=1)          # put below scatter points

# plot all points on map
plt.scatter(every12hours['lon'],every12hours['lat'], 
            transform=ccrs.Geodetic(), 
            c=every12hours.index,       # set color of point based on hours since start
            vmin=min(every12hours.index),
            vmax=max(every12hours.index),
            cmap=cm,
            s=25,                          # label text size
            zorder=3,                      # put above plot line
            edgecolor='black',
            linewidth=.8)

# add labels for the start of each day
for ind, row in every24hours.iterrows():
    at_x, at_y = ax.projection.transform_point(x=row['lon'], y=row['lat'], src_crs=ccrs.Geodetic())
    plt.annotate(row['datetime'][:10], 
                 xy=(at_x, at_y),                # location of point to be labeled
                 xytext=(15, -10),               # offset distance from point for label
                 textcoords='offset points',
                 color='black', 
                 backgroundcolor='white', 
                 size=6,
                 zorder=2,
                 arrowprops=dict(arrowstyle='-', color='white', linewidth=1.0))

# make colorbar
cbar = plt.colorbar(ax=ax, shrink=.5, pad=0.1, label='Hours back along trajectory')
cbar.ax.invert_yaxis() 

# circular boundary 
# from https://scitools.org.uk/cartopy/docs/v0.15/examples/always_circular_stereo.html
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

ax.figure.set_size_inches(10,10)
# save as pdf
h=plt.gcf();
h.savefig(map_filename);