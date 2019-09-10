# convert_cam_to_arl.py
# by Kara Hartig
# last modified: August 7, 2019
# based on:
#     convert_cam_to_arl.py by Pratap Singh
#     wrfout2arl.html by Yaqiang Wang
#
# Convert a netcdf output file from the Community Atmosphere Model, version 4, to
# .arl input format for HYSPLIT backtracking code
# Run with the MeteoInfo software suite (http://www.meteothinker.com/index.html) by
# Yaqiang Wang.
# As it runs, the code produces and deletes a file called cam_to_arl_temp.arl in the output 
# directory.  This file must not be tampered with while the code is running.
# To run within MeteoInfo main directory:
#    on Unix: >>./milab.sh convert_cam_to_arl.py
#    on Mac: >>./milab_mac.sh convert_cam_to_arl.py


# Set data folders
indatadir = '/n/tzipermanfs2/khartig/arctic_air_suppression/no_rotation/full_workflow_TEST_1/'
outdatadir = '/n/tzipermanfs2/khartig/arctic_air_suppression/no_rotation/full_workflow_TEST_1/'

# Read the input netCDF data files
in_file = addfile(os.path.join(indatadir, 'pi_3h_0809.nc'))
print 'opened netcdf' 

# Touch the output file without writing to it:
out_filename = os.path.join(outdatadir, "pi_3h_0809.arl")
open(out_filename, 'a').close();

# input variable names in netcdf  # NOT USED
#nc_varnames_2d = ['PS']
#nc_varnames_3d = ['T','U','V','OMEGA']

# corresponding output variable names in arl
arl_varnames_2d = ['PRSS']
arl_varnames_3d = ['TEMP','UWND','VWND','WWND']

# dimensions of (3-D) input netCDF variables
sample_variable = in_file['U']  # sample variable for calculating dimensions and levels
lon_index = 3
lat_index = 2
level_index = 1
#time_index = 0
nx = sample_variable.dimlen(lon_index)  # longitude
ny = sample_variable.dimlen(lat_index)  # latitude
nz = sample_variable.dimlen(level_index)  # vertical level
#levels = sample_variable.dimvalue(level_index)

# convert vertical level coefficients to ECMWF-format levels
hyam = in_file['hyam'][:]
hybm = in_file['hybm'][:]
P_0 = in_file.read('P0').getFloat(0)  # retrieves value for P_0 from file object
P_0_units = str(in_file['P0'].attrvalue('units')[0])
if P_0_units != 'Pa':
	raise ValueError("Reference pressure P0 is not in units of Pa. Change definition of 'levels' so that integer part 'a' is in hPa.")
a = hyam[::-1] * P_0 / 100.0  # reverse index order; factor of 100 to convert to hPa
b = hybm[::-1]  # reverse index order; dimensionless, so no need to convert units
levels = [int(round(a[ind])) + b[ind] for ind in range(len(a))]

# loop over times, append each to end of ARL file
num_times = in_file.timenum()
fhour = 0  # Offset for initial time; incremented for each timestamp
for t in range(0, num_times):
    # Set temporary output ARL data file
    temp_filename = os.path.join(outdatadir, 'cam_to_arl_temp.arl')
    temp_file = addfile(temp_filename, 'c', dtype='arl')
    temp_file.setlevels(levels)
    temp_file.set2dvar(arl_varnames_2d)
    for l in levels:
        temp_file.set3dvar(arl_varnames_3d)

    # Write ARL data file
    temp_file.setx(sample_variable.dimvalue(lon_index))
    temp_file.sety(sample_variable.dimvalue(lat_index))
    timestamp = in_file.gettime(t)
    print 'Time stamp: ' + str(timestamp)
    # In getdatahead below, integer is "vertical coordinate system flag":
    #     1 (sigma fraction), 2 (pressure mb), 3 (terrain fraction), 4 (hybrid sigma-pressure)
    data_header = temp_file.getdatahead(in_file.proj, 'CAM4', 4, fhour)
    # modify data header grid parameter(s)
    #    for lat-lon grid, Pole Longitude must be maximum longitude value
    data_header.POLE_LON = max(sample_variable.dimvalue(lon_index))
    # Pre-write index record without data record - will be over-written later
    temp_file.writeindexrec(timestamp, data_header)

    # List of data records for this timestamp
    full_data_record = []

    # Write 2-D Variables: surface pressure
    surface_data_record = []
    psurf = in_file['PS'][t,:,:] * 0.01  # unit conversion from Pa to hPa
    single_data_record = temp_file.writedatarec(timestamp, 0, 'PRSS', fhour, 99, psurf)  # write surface level (index 0)
    surface_data_record.append(single_data_record)
    full_data_record.append(surface_data_record)

    # Write 3-D Variables
    for lidx in range(0, nz):
        level_data_record = []
        temp = in_file['T'][t,lidx,:,:]  # input K, output K
        uwnd = in_file['U'][t,lidx,:,:] #input m/s, output m/s
        vwnd = in_file['V'][t,lidx,:,:] # input m/s, output m/s
        wwnd = in_file['OMEGA'][t,lidx,:,:] * 0.01 # input Pa/s, output hPa/s
        single_data_record = temp_file.writedatarec(timestamp, lidx + 1, 'TEMP', fhour, 99, temp)
        level_data_record.append(single_data_record)
        single_data_record = temp_file.writedatarec(timestamp, lidx + 1, 'UWND', fhour, 99, uwnd)
        level_data_record.append(single_data_record)
        single_data_record = temp_file.writedatarec(timestamp, lidx + 1, 'VWND', fhour, 99, vwnd)
        level_data_record.append(single_data_record)
        single_data_record = temp_file.writedatarec(timestamp, lidx + 1, 'WWND', fhour, 99, wwnd)
        level_data_record.append(single_data_record)
        full_data_record.append(level_data_record)
    # Re-write index record with data record
    temp_file.writeindexrec(timestamp, data_header, full_data_record)
    temp_file.close()
    # Append the temp output file to the main output file
    os.system("cat " + temp_filename + " >> " + out_filename)
    # Delete the temp output file
    os.system("rm " + temp_filename)
    fhour += 1
    if fhour >= 99:
        fhour = 99
print 'Finished!'
