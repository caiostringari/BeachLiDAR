# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : LiDAR2nc.py
# POURPOSE : Convert LSM5X exported CSV files to netcdf
#
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 21/03/2018 [Caio Stringari]
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import sys

import subprocess

import xarray as xr



def main():

    print("\nResampling data, please wait...\n")

    if len (sys.argv) < 3:
    	raise IOError("Usage: python resample.py input.nc offset_string")
    else:
    	# IO names
    	input_file = sys.argv[1]
    	frequency = sys.argv[2]
    	output_file = sys.argv[1].replace(".nc", "_resampled.nc")

    # open dataset
    ds = xr.open_dataset(input_file)  #.isel(time=slice(0, 500))

    # resample
    dsr = ds.resample(time=frequency).mean()


    print(" --> Resampled dataset:")
    print(dsr)

    # dump to file
    dsr.to_netcdf(output_file)

    print("\nMy work is done!\n")

if __name__ == '__main__':
    
    # main
    main()
