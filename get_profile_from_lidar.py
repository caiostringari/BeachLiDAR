# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : get_lidar_profile.py
# POURPOSE : create a beach profile based on LiDAR data
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.1     : 27/04/2018 [Caio Stringari]
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')

# System
import os
import sys
import subprocess

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
from scipy import linalg
from scipy.stats import binned_statistic


# Outliers
from scipy.spatial import ConvexHull
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (RANSACRegressor,
                                  LinearRegression,
                                  TheilSenRegressor)
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# Personal tools
from pywavelearn.utils import ellapsedseconds

# Progress bar
from tqdm import tqdm

# Plotting
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2


def rotation_transform(theta):
    """
    Get a 2-D rotation matrix transformation.

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


def main():

    # open file
    ds = xr.open_dataset(args.input[0])

    # get times
    times = pd.to_datetime(ds["time"].values).to_pydatetime()
    fmt = "%Y%m%d-%H:%M:%S"
    start = datetime.datetime.strptime(args.start[0], fmt)
    end = start+datetime.timedelta(minutes=int(args.dt[0]))

    # print(start, end)

    for t, time in enumerate(times):
        time = UTC.localize(time)
        if time == EST.localize(start):
            i1 = t
        if time == EST.localize(end):
            i2 = t
    ds = ds.isel(time=slice(i1, i2))
    times = pd.to_datetime(ds["time"].values).to_pydatetime()

    # output path
    out = args.output[0]

    # timeloop
    k = 0
    X = []
    Y = []
    with tqdm(total=len(ds.time.values)) as pbar:
        for t, time in enumerate(times):

            # localize time
            now = UTC.localize(time)
            now = now.astimezone(EST)

            # slice ds in time
            tds = ds.isel(time=t)

            # get x and y coordinates
            x = tds["x"].values
            y = tds["y"].values

            # sort the data
            sorts = np.argsort(x)[::-1]
            x = x[sorts]
            y = y[sorts]

            # rotate
            R = rotation_transform(THETA)

            # apply the rotation transform
            XYr = np.dot(R, np.vstack([x, y]))

            # extract coords
            x = XYr[0, :]
            y = XYr[1, :]

            # cut to the limits
            i1 = np.argmin(np.abs(x-XCUT[0]))
            i2 = np.argmin(np.abs(x-XCUT[1]))
            xf = x[i1:i2]
            yf = y[i1:i2]

            # try to bin the data
            try:
                ybin, bin_edges, _ = binned_statistic(xf, yf, bins=XBINS)
                bin_width = (bin_edges[1] - bin_edges[0])
                bin_centers = bin_edges[1:] - bin_width/2
                # append
                X.append(bin_centers)
                Y.append(ybin)

            # if not possible, just pass
            except Exception:
                pass

            pbar.update()

    # get medians
    X = np.array(X)
    Y = np.array(Y)
    xm = bin_centers
    ym = np.nanmin(Y, axis=0)

    # get rid of nans
    # nans = np.isnan(ym)
    xm = xm[np.logical_not(np.isnan(ym))]
    ym = ym[np.logical_not(np.isnan(ym))]
    # ym = ym[!=nans]
    # xm = 
    # print(ym)

    # remove nans
    ym = np.squeeze(pd.Series(ym).interpolate().values)

    # fit
    model = Pipeline([('poly',
                       PolynomialFeatures(degree=ORDER)),
                      ('ols',
                       RANSACRegressor())])
    model.fit(xm.reshape(-1, 1), ym)

    xpred = bin_centers
    ypred = model.predict(xpred.reshape(-1, 1))

    # save
    df = pd.DataFrame()
    df["x"] = xpred
    df["y"] = ypred
    df.to_csv(out)

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#E9E9F1")
    ax.scatter(X.flatten(), Y.flatten(), 50,
               marker="+", color="k", alpha=0.35)
    ax.plot(xm, ym, ls="-", lw=2, color="r")
    ax.plot(xpred, ypred, ls="-", lw=2, color="dodgerblue")
    ax.grid(color="w")
    ax.set_ylim(ypred.min(), ypred.max())
    ax.set_xlim(xpred.min(), xpred.max())
    ax.set_aspect(10)
    sns.despine()
    plt.savefig(out.replace(".csv", ".png"))
    plt.close()
    # plt.show()

    print("\nMy work is done!\n")


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
    parser.add_argument('--dt', '-dt',
                        nargs=1,
                        action='store',
                        dest='dt',
                        help="Time delta in seconds.",
                        required=True)
    parser.add_argument('--order', '-order',
                        nargs=1,
                        action='store',
                        dest='order',
                        help="Order for the RANSAC regressor.",
                        default=[5],
                        required=False)
    parser.add_argument('--dx', '-dx',
                        nargs=1,
                        action='store',
                        dest='dx',
                        help="Spacing in the x-direction.",
                        default=[0.05],
                        required=False)
    parser.add_argument('--output', '-o',
                        nargs=1,
                        action='store',
                        dest='output',
                        help="Output file name (.csv).",
                        required=True)

    args = parser.parse_args()

    print("\nAproximating profile, please wait...\n")

    EST = timezone('Australia/Sydney')
    UTC = timezone("UTC")

    XLIMS = [float(args.xlim[0]), float(args.xlim[1])]
    YLIMS = [float(args.ylim[0]), float(args.ylim[1])]
    XCUT = [float(args.xcut[0]), float(args.xcut[1])]
    THETA = float(args.theta[0])
    XBINS = np.arange(XCUT[0], XCUT[1], float(args.dx[0]))
    ORDER = int(args.order[0])

    main()
