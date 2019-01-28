# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : get_lidar_boundary.py
# POURPOSE : get boundary conditions from LiDAR data
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.1     : 28/06/2018 [Caio Stringari]
# v1.2     : 15/10/2018 [Caio Stringari]
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# System
import os
import sys
import subprocess
import warnings

# # Dates
import datetime
import pytz
from pytz import timezone
from matplotlib.dates import date2num

import xarray as xr
import pandas as pd

# Arguments
import argparse

# Numpy and scipy
import numpy as np
from scipy import interpolate
from scipy.stats import binned_statistic

# Progress bar
from tqdm import tqdm

# Plotting
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2


def rotation_transform(theta):
    """
    Build a 2-D rotation matrix transformation given an input angle.

    ---------
    Args:
        theta [Mandatory (float)] : rotation angle in degrees.

    ---------
    Return:
        R [Mandatory (np.array)] : rotation matrix
    """
    theta = np.radians(theta)
    R = [[np.math.cos(theta), -np.math.sin(theta)],
         [np.math.sin(theta), np.math.cos(theta)]]

    return np.array(R)

# def main():
    # """Call the main processing algorithm."""
    # read the profile

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
    parser.add_argument('--profile', '-p',
                        nargs=1,
                        action='store',
                        dest='profile',
                        help="An input profile.",
                        required=True)
    parser.add_argument('--lidar-coords', '-coords', '-xy',
                        nargs=2,
                        action='store',
                        dest='coords',
                        default=[0, 0],
                        help="LiDAR surveyed coordinates. Default is 0, 0",
                        required=False)
    parser.add_argument('--xlim', '-xlim',
                        nargs=2,
                        action='store',
                        dest='xlim',
                        default=[0.75, 1.25],
                        help="Detection boundary in the x-direction.",
                        required=False)
    parser.add_argument('--ylim', '-ylim',
                        nargs=2,
                        action='store',
                        dest='ylim',
                        default=[0.75, 1.25],
                        help="Detection boundary in the y-direction.",
                        required=False)
    parser.add_argument('--cut', '-cut',
                        nargs=2,
                        action='store',
                        dest='xcut',
                        default=[-20, 20],
                        help="Analysis limits in the x-direction.",
                        required=False)
    parser.add_argument('--ycut', '-ycut',
                        nargs=2,
                        action='store',
                        dest='ycut',
                        default=[-1, 2],
                        help="Analysis limits in the y-direction.",
                        required=False)
    parser.add_argument('--theta', '-theta',
                        nargs=1,
                        action='store',
                        dest='theta',
                        default=[180],
                        help="Rotation angle in degrees.",
                        required=False)
    parser.add_argument('--start', '-t1',
                        nargs=1,
                        action='store',
                        dest='start',
                        help="Start time. Format is YYYYMMDD-HH:MM:SS.",
                        required=True)
    parser.add_argument('--end', '-t2',
                        nargs=1,
                        action='store',
                        dest='end',
                        help="End time. Format is YYYYMMDD-HH:MM:SS.",
                        required=True)
    parser.add_argument('--output', '-o',
                        nargs=1,
                        action='store',
                        dest='output',
                        help="Output file name (.csv).",
                        required=True)

    args = parser.parse_args()

    print("\nGetting boundary conditions, please wait...\n")

    EST = timezone('Australia/Sydney')
    UTC = timezone("UTC")

    XLIMS = [float(args.xlim[0]), float(args.xlim[1])]
    YLIMS = [float(args.ylim[0]), float(args.ylim[1])]
    XCUT = [float(args.xcut[0]), float(args.xcut[1])]
    YCUT = [float(args.ycut[0]), float(args.ycut[1])]
    THETA = float(args.theta[0])

    # main call
    # main()

    # read the profile
    profile = pd.read_csv(args.profile[0])
    xprof = profile["x"].values
    yprof = profile["y"].values

    # open file
    ds = xr.open_dataset(args.input[0])

    # get times
    times = pd.to_datetime(ds["time"].values).to_pydatetime()
    fmt = "%Y%m%d-%H:%M:%S"
    end = datetime.datetime.strptime(args.end[0], fmt)
    start = datetime.datetime.strptime(args.start[0], fmt)

    for t, time in enumerate(times):
        time = UTC.localize(time)
        if time == EST.localize(start):
            i1 = t
        if time == EST.localize(end):
            i2 = t
    ds = ds.isel(time=slice(i1, i2))
    times = pd.to_datetime(ds["time"].values).to_pydatetime()

    # get the lidar height
    height = np.mean(
        ds["y"].values[:, int(np.ceil(len(ds["points"].values)/2))])

    # rotation matrix
    R = rotation_transform(THETA)

    # sample frequency
    sf = int(1/(times[1]-times[0]).total_seconds())

    # define the bins
    dx = np.diff(xprof).mean()
    bins = np.arange(xprof[0], xprof[-1]+dx, dx)

    # timeloop
    t = 0
    Eta = []
    Times = []
    with tqdm(total=len(ds.time.values)) as pbar:
        for time in times:

            # localize time
            now = UTC.localize(time)
            now = now.astimezone(EST)

            xraw = ds.isel(time=t)["x"].values
            yraw = ds.isel(time=t)["y"].values
            XY = np.dot(R, np.vstack([xraw, yraw])).T

            # bin the data
            ybin, bin_edges, _ = binned_statistic(XY[:, 0],
                                                  XY[:, 1], bins=bins)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width/2

            # fix according to the profile
            idx = np.where(ybin < yprof)[0]
            ybin[idx] = yprof[idx]

            # get nearest value to the boundary
            Z = []
            for i, z in enumerate(ybin[::-1]):
                if not np.isnan(z):
                    Z.append(yprof[iz])
                    break

            # output
            Eta.append(np.abs(Z))
            Times.append(now)
            pbar.update()
            t += 1

    Eta = np.array(Eta)
    Times = np.array(Times)

    # plot
    fif, ax = plt.subplots(figsize=(12, 6))
    ax.plot(Times, Eta, ls="-", lw=3, color="k")
    ax.set_ylabel("Surface Elevation [m]")
    ax.grid(color="w", lw=2, ls="-")
    sns.despine(ax=ax)
    plt.show()


    print("\nMy work is done!\n")
