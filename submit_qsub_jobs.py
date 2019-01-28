# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : submit_qsub_jobs.py
# POURPOSE : extract a timestack from lidar data using multiple qsub calls.
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 19/10/2018 [Caio Stringari]
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# System
import os
import sys
import datetime
import subprocess

# Dates
import pytz
from pytz import timezone

# Data
import xarray as xr
import pandas as pd

# Arguments
import argparse


def template(start, end, inp, profile, out, jobname, cut):
    """Write the bash template to be executed using qsub."""
    #
    start = start.strftime("%Y%m%d-%H:%M:%S")
    end = end.strftime("%Y%m%d-%H:%M:%S")
    #
    TEMPLATE_SERIAL = """
#!/bin/bash
#PBS -l select=1:ncpus=1:mem=12gb
#PBS -l walltime=01:00:00
#PBS -k oe
#PBS -q xeon5q

# Fix CPU usage going beyond 100%
export OMP_NUM_THREADS=1

source /etc/profile.d/modules.sh
module load opencv/3.4.1.15-python3.6
module load scikit-image/0.14.0-python3.6
module load scikit-learn/1.19.1-python3.6

cd {}

python lidar_timestack.py -i {} -o {} -p {} -t1 {} -t2 {} --cut {} {}

""".format(WORK, inp,  out, profile, start, end, cut[0], cut[1])

    f = open("{}".format(jobname), "w")
    f.write(TEMPLATE_SERIAL)
    f.close()


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
    # input data
    parser.add_argument('--profile', '-p',
                        nargs=1,
                        action='store',
                        dest='profile',
                        help="An input profile.",
                        required=True)
    # analysis start
    parser.add_argument('--start', '-t1',
                        nargs=1,
                        action='store',
                        dest='start',
                        help="Start time. Format is YYYYMMDD-HH:MM:SS.",
                        required=True)
    # time delta
    parser.add_argument('--time-delta', '-dt',
                        nargs=1,
                        action='store',
                        dest='dt',
                        help="Analysis duration. Default is 10 min.",
                        default=[10],
                        required=False)
    parser.add_argument('--cut', '-cut',
                         nargs=2,
                         action='store',
                         dest='xcut',
                         default=[-20, 20],
                         help="Analysis limits in the x-direction.",
                         required=False)
    # time delta
    parser.add_argument('--base-name', '-o',
                        nargs=1,
                        action='store',
                        dest='output',
                        help="Timestack basename.",
                        required=True)
    args = parser.parse_args()

    # timezones
    EST = timezone('Australia/Sydney')
    UTC = timezone("UTC")

    # inputs
    fname = args.input[0]
    prof = args.profile[0]
    start = datetime.datetime.strptime(args.start[0], "%Y%m%d-%H:%M:%S")
    start_utc = EST.localize(start).astimezone(UTC)
    dt = int(args.dt[0])
    basename = args.output[0]
    cut = [float(args.xcut[0]), float(args.xcut[1])] 

    # work folder
    WORK = os.path.dirname(os.path.realpath(__file__))

    # open file
    ds = xr.open_dataset(fname)

    # get times
    times = pd.to_datetime(ds["time"].values).to_pydatetime()

    # utc times
    for t, time in enumerate(times):
        time = UTC.localize(time)
        if time == start_utc:
            start_utc = time
    end_utc = time

    # analysis loop
    t1 = start_utc
    while t1 <= end_utc:

        # times
        t2 = t1+datetime.timedelta(minutes=dt)
        aut1 = t1.astimezone(EST)
        aut2 = t2.astimezone(EST)

        # output
        out = basename + \
            aut1.strftime("_%Y%m%d_%H%M")+"_"+aut2.strftime("%H%M")+".nc"

        # write template
        script = "lidar_timestack_{}.sh".format(
            aut1.strftime("%Y%m%d_%H%M"))
        template(aut1, aut2, fname, prof, out, script, cut)

        subprocess.call("qsub {}".format(script), shell=True)

        # update
        t1 = t2

        break
