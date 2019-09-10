#!/bin/bash

# bash script to concatenate Nov-Dec and Jan-Feb CAM4 files along time
# dimension for the following variables:
#     from h3 file:  U, V, OMEGA, PS, P0, hyam, hybm
#     from h4 file:  T
# USAGE: >>./concat_winter.sh <winter index> <output path>
#     winter index is 0 for 07-08 winter, 1 for 08-09, etc.

# Load modules
module load gcc/7.1.0-fasrc01 nco/4.7.4-fasrc01

# Define list of variables to save
VAR_LIST_H3=U,V,OMEGA,PS,P0,hyam,hybm
VAR_LIST_H4=T
echo "Storing variables: $VAR_LIST_H3 (from h3), $VAR_LIST_H4 (from h4)"

# Read winter index and output path from command line argument
# index 0 is 07-08 winter, index 1 is 08-09, etc.
if [ $# -lt 2 ]; then
  echo "Not enough arguments; must provide winter index and path to output file"
  exit 2
else
  WIDX=$1
  OUT_DIR=$2
fi
printf -v YR1 "%02d" $((7+$WIDX))
printf -v YR2 "%02d" $((8+$WIDX))
echo "for winter ${YR1}-${YR2}"

# Generate time bound strings for concatenation
TIME1=("00${YR1}-11-01 00:00:00")  # start Nov 1st
TIME2=("00${YR2}-03-01 00:00:00")  # end March 1st
echo "from time $TIME1 to $TIME2"
echo

# Define file locations and names
IN_DIR=/n/tzipermanfs2/CAM-output-for-Arctic-air-suppression/PI
IN_FILE_H3_1=${IN_DIR}/pi_3h.cam.h3.00${YR1}-01-01-10800.nc
IN_FILE_H3_2=${IN_DIR}/pi_3h.cam.h3.00${YR2}-01-01-10800.nc
IN_FILE_H4_1=${IN_DIR}/pi_3h.cam.h4.00${YR1}-01-01-10800.nc
IN_FILE_H4_2=${IN_DIR}/pi_3h.cam.h4.00${YR2}-01-01-10800.nc
OUT_FILE=${OUT_DIR}/pi_3h_${YR1}${YR2}.nc
TEMP_FILE=${OUT_DIR}/pi_3h_${YR1}${YR2}_h3only.nc
echo "Input files:"
echo "    " $IN_FILE_H3_1
echo "    " $IN_FILE_H3_2
echo "    " $IN_FILE_H4_1
echo "    " $IN_FILE_H4_2
echo
echo "Output file:"
echo "    " $OUT_FILE
echo

# If output files already exist, delete them
rm -f $OUT_FILE
rm -f $TEMP_FILE

# Use ncrcat to concatenate files across desired times and variables
echo "Concatenating h3 variables..."
ncrcat -d time,"${TIME1[@]}","${TIME2[@]}" -v $VAR_LIST_H3 $IN_FILE_H3_1 $IN_FILE_H3_2 $TEMP_FILE
echo "Concatenating h4 variables..."
ncrcat -d time,"${TIME1[@]}","${TIME2[@]}" -v $VAR_LIST_H4 $IN_FILE_H4_1 $IN_FILE_H4_2 $OUT_FILE
echo "Combining h3 and h4 files..."
ncks -A $TEMP_FILE $OUT_FILE
rm $TEMP_FILE
echo "Finished"
