# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : slice.py
# POURPOSE : cut X minutes of data from raw files
#
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 15/02/2019 [Caio Stringari]
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import sys
import argparse
import numpy as np
import xarray as xr

import pytz
import datetime
from pytz import timezone
from pandas import to_datetime
from matplotlib.dates import date2num


def main():
    inp = args.input[0]
    out = args.output[0]
    start = args.start[0]
    dt = float(args.dt[0])

    t1 = datetime.datetime.strptime(start, "%Y%m%d-%H:%M:%S")
    t2 = t1+datetime.timedelta(minutes=dt)

    ds = xr.open_dataset(inp)

    times = to_datetime(ds["time"].values).to_pydatetime()

    i1 = np.argmin(np.abs(date2num(times) - date2num(EST.localize(t1))))
    i2 = np.argmin(np.abs(date2num(times) - date2num(EST.localize(t2))))
    dsc = ds.isel(time=slice(i1, i2))

    dsc.to_netcdf(out)

    print("My work is done!\n")


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument('--input', '-i',
                        nargs=1,
                        action='store',
                        dest='input',
                        help="Input netCDF file from LiDAR2nc.py.",
                        required=True)
    # output data
    parser.add_argument('--output', '-o',
                        nargs=1,
                        action='store',
                        dest='output',
                        help="Output netCDF.",
                        required=True)

    # dt data
    parser.add_argument('--duration', '-dt',
                        nargs=1,
                        action='store',
                        dest='dt',
                        help="Duration in minutes. Default is 10m.",
                        required=False,
                        default=[10])

    # start time
    parser.add_argument('--start', '-t1',
                        nargs=1,
                        action='store',
                        dest='start',
                        help="Start time. Format is YYYYMMDD-HH:MM:SS.",
                        required=True)

    args = parser.parse_args()

    EST = timezone('Australia/Sydney')
    UTC = timezone("UTC")

    print("\nSlicing timestack, please wait...")

    main()
