# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : lidar_timestack.py
# POURPOSE : extract a timestack from lidar data.
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.1     : 28/06/2018 [Caio Stringari]
# v1.2     : 19/10/2018 [Caio Stringari]
# v2.0     : 14/02/2019 [Caio Stringari] -- Fix outlier detection and
#                                           beach profile definitions.
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
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic

from scipy.spatial import KDTree
from scipy.stats import binned_statistic

# Outliers
from scipy.spatial import ConvexHull
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

# Plotting
# import seaborn as sns
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.0})
# sns.set_style("ticks", {'axes.linewidth': 2,
#                         'legend.frameon': True,
#                         'axes.facecolor': "#E9E9F1",
#                         'grid.color': "w"})
# mpl.rcParams['axes.linewidth'] = 2

# np.where warnings
warnings.filterwarnings("ignore")


def profile(ds, xprof, yprof, R, lidar_height, dx=0.01, plot=False):
    """
    Aproximate a beach profile based on measured RTK and LiDAR data.

    ---------
    Args:
        ds [Mandatory (xarray.Dataset)] : LiDAR dataset.

        xprof, yprof [Mandatory (np.array)]: Measuered profile coordinates

        R [Mandatory (array)]: Rotation transform

        lidar_height [Mandatory (float)]: Measured LiDAR height.

        dx [Optinal (float)]: Interpolation grid resolution.

    ---------
    Return:
        xprof, yprof [Mandatory (np.array)]: Processed profile coordinates
    """

    # height diferences
    hprof = yprof[int(np.ceil(len(yprof)/2))]
    height_diff = np.abs(lidar_height-hprof)

    # apply the rotation transform
    x = ds.isel(time=0)["x"].values.flatten()
    y = ds.isel(time=0)["y"].values.flatten()

    # print(x.shape, y.shape)
    XYr = np.dot(R, np.vstack([x, y]))

    # extract coords
    x = XYr[0, :]
    y = XYr[1, :]+lidar_height

    df = pd.DataFrame(np.vstack([x, y]).T, columns=["x", "y"])
    df = df.dropna()

    # aproximate prifile from lidar data
    model = Pipeline([('poly',
                       PolynomialFeatures(degree=1)),
                      ('ols',
                       RANSACRegressor())])
    model.fit(df["x"].values.reshape(-1, 1), df["y"].values)
    xpred = np.arange(df["x"].min(), df["x"].max(), dx)
    ypred = np.squeeze(model.predict(xpred.reshape(-1, 1)))

    # aproximate diferences in profile
    h1 = yprof[np.argmin(np.abs(xprof))]
    h2 = ypred[np.argmin(np.abs(xpred))]
    hd = h1-h2

    # fix
    yprof = yprof-hd

    # interpolate
    xnew = np.arange(xprof.min(), xprof.max(), dx)
    f = interp1d(xprof, yprof, kind="linear")
    ynew = f(xnew)

    #

    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(x, y, marker="+", color="k")
        ax.plot(xnew, ynew, color="r")
        ax.scatter(xpred, ypred)
        plt.show()

    return xnew, ynew-lidar_height


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


def get_inliers(x, y, xprof, return_boundary=False,
                algorithm="convex_hull", interpolation="NN"):
    """
    Search for out and inliers.

    ---------
    Args:
        x, y [Mandatory (1D np.ndarray)] : LiDAR x and y coordinates.

        xprof [Mandatory (1D np.ndarray)] : x-coordinates of the beach
                                            profile.

        return_boundary [Optional (bool)] : return search boundaries

        algorithm [Optional (bool)] : algorithm to be used. Default is
                                      to use the convex-hull method.

    ---------
    Return:
        xf, yf [Mandatory (1D np.ndarray)] : inlier LiDAR x and y
                                             coordinates.

        if return_boundary:

        xb, yb [Mandatory (1D np.ndarray)] : boundary  x and y
                                             coordinates.
    """
    # define the bins
    dx = np.diff(xprof).mean()
    bins = np.arange(xprof[0], xprof[-1]+dx, dx)

    # build the ransac model
    model = Pipeline([('poly',
                       PolynomialFeatures(degree=1)),
                      ('ols',
                       RANSACRegressor())])
    model.fit(x.reshape(-1, 1), y)
    xpred = xprof
    ypred = model.predict(xpred.reshape(-1, 1))

    # detect using a convex hull approach
    if algorithm == "R+H":
        xcv = np.hstack([XLIMS[0]*xpred, XLIMS[1]*xpred])
        ycv = np.hstack([YLIMS[0]*ypred, YLIMS[1]*ypred])
        points = np.vstack([xcv, ycv]).T
        hull = ConvexHull(points)

        xf = []
        yf = []
        for _x, _y in zip(x, y):
            inside = point_in_hull((_x, _y), hull)
            if inside:
                xf.append(_x)
                yf.append(_y)

        if interpolation == "NN":
            # nearest neighbour look up
            Tree = KDTree(np.vstack([xf, np.ones(len(xf))]).T)
            _, idx = Tree.query(np.vstack([bins, np.ones(len(bins))]).T, k=1)
            yf = np.array(yf)[idx]
            xf = bins
            # interploate
            f = interp1d(xf, yf, fill_value="extrapolate", kind="linear")
            yf = f(xprof)
            xf = xprof

        elif interpolation == "BS":
            # bin the data
            ybin, bin_edges, _ = binned_statistic(xf, yf, bins=bins)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width/2
            xf = bin_centers
            yf = ybin
            # interploate
            f = interp1d(xf, yf, fill_value="extrapolate", kind="linear")
            yf = f(xprof)
            xf = xprof
        else:
            raise ValueError("Interpolation method unknown")
    else:
        raise NotImplementedError()

    if return_boundary:
        return xf, yf, xpred, ypred
    else:
        return xf, yf


def point_in_hull(point, hull, tolerance=1e-12):
    """
    Return true if point is inside a convex hull.

    ---------
    Args:
        point [Mandatory (tuple)] : tuple of x and y coordinates.

        hull [Mandatory (scipy.spatial.convex_hull)] : convex hull

    ---------
    Return:
        true [ (bool)] : true if point is inside hull.
    """
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def ellapsedseconds(times):
    """
    Count how many (fractions) of seconds have passed from the begining.

    Round to 6 decimal places no matter what.

    ----------
    Args:
        time [Mandatory (pandas.to_datetime(array)]: array of timestamps

    ----------
    Return:
        seconds [Mandatory (np.ndarray)]: array of ellapsed seconds.
    """
    times = pd.to_datetime(times).to_pydatetime()
    seconds = []
    for t in range(len(times)):
        dt = (times[t]-times[0]).total_seconds()
        seconds.append(round(dt, 6))

    return np.array(seconds)


def main():
    """Call the main processing algorithm."""

    # read the profile
    df = pd.read_csv(args.profile[0])
    xprof = df["x"].values
    yprof = df["y"].values

    # open file
    ds = xr.open_dataset(args.input[0])

    # get times
    times = pd.to_datetime(ds["time"].values).to_pydatetime()
    fmt = "%Y%m%d-%H:%M:%S"
    start = datetime.datetime.strptime(args.start[0], fmt)
    end = start+datetime.timedelta(minutes=DT)

    # get the lidar height
    lidar_height = np.nanmean(
        ds["y"].values[:, int(np.ceil(len(ds["points"].values)/2))])

    i1 = np.argmin(np.abs(date2num(times) - date2num(EST.localize(start))))
    i2 = np.argmin(np.abs(date2num(times) - date2num(EST.localize(end))))
    ds = ds.isel(time=slice(i1, i2))
    times = pd.to_datetime(ds["time"].values).to_pydatetime()

    # rotation matrix
    R = rotation_transform(THETA)

    # sample frequency
    sf = int(1/(times[1]-times[0]).total_seconds())

    # plot profile()
    plot = False
    if args.debug_profile:
        plot = True
    xprof, yprof = profile(ds, xprof, yprof, R, lidar_height, dx=DX,
                           plot=plot)
    if plot:
        sys.exit()

    # cut to XCUT
    idx1 = np.argmin(np.abs(xprof-XCUT[0]))
    idx2 = np.argmin(np.abs(xprof-XCUT[1]))
    xprof = xprof[idx1:idx2]
    yprof = yprof[idx1:idx2]
    bins = np.arange(xprof.min(), xprof.max()+DX, DX)

    # timeloop
    t = 0
    Z = []
    H = []
    autimes = []
    for time in times:

        print("  --- Processing step {} of {}".format(t+1, len(times)),
              end="\r")

        # localize time
        now = UTC.localize(time)
        now = now.astimezone(EST)

        # plot raw data
        xraw = ds.isel(time=t)["x"].values
        yraw = ds.isel(time=t)["y"].values
        XY = np.dot(R, np.vstack([xraw, yraw])).T

        # slice ds in time
        cds = ds.isel(time=t)  # current

        # current x and y coordinates
        xcurr = cds["x"].values
        ycurr = cds["y"].values

        # sort the data
        currsorts = np.argsort(xcurr)[::-1]  # current

        XYcurr = np.dot(R, np.vstack([xcurr[currsorts],
                                      ycurr[currsorts]])).T  # current

        # try to apply the decision boundaries
        try:
            xcurr, ycurr, xbnd, ybnd = get_inliers(XYcurr[:, 0],
                                                   XYcurr[:, 1], xprof,
                                                   return_boundary=True,
                                                   algorithm=outliers,
                                                   interpolation=interp)
            # final coordinates
            xf = xcurr
            yf = ycurr
        except Exception:
            xf = xprof
            yf = yprof

        # append raw heights
        H.append(yf)

        # bore heights
        idx = np.where(yf < yprof)[0]
        yf[idx] = yprof[idx]

        # calculate the difference between the profile and the
        # current water level
        z = np.abs(yprof-yf)
        Z.append(z)

        # # make sure plot
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.plot(xprof, yprof, lw=2, color="r")
        # ax.scatter(xf, yf, color="k", marker="+")
        # plt.show()
        # sys.exit()

        t += 1

    # reshape a
    Z = np.array(Z)
    Z[np.isnan(Z)] = 0
    # secs = ellapsedseconds(pd.to_datetime(times).to_pydatetime())

    # build the netcdf file
    ds = xr.Dataset()
    # write surface elevation
    ds['eta'] = (('time', 'points'),  Z)  # surface elevation
    ds['height'] = (('time', 'points'),  H)  # measured heights
    # write distance
    ds['distance'] = (('points'),  xf)  # surface elevation
    ds["profile"] = (('points'),  yf)   # beach profile
    # write coordinates
    ds.coords['time'] = pd.to_datetime(times)
    ds.coords["points"] = np.arange(0, len(xf), 1)
    # write to file
    units = 'days since 2000-01-01 00:00:00'
    calendar = 'gregorian'
    encoding = dict(time=dict(units=units, calendar=calendar))
    ds.to_netcdf(args.output[0], encoding=encoding)
    ds.close()


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
    parser.add_argument('--outlier-detection', '-outliers',
                        nargs=2,
                        action='store',
                        dest='outliers',
                        default=["R+H"],
                        help="Outlier detection method."
                             "Defualt is RANSAC + Convex Hull",
                        required=False)
    parser.add_argument('--interpolation', '-interp',
                        nargs=1,
                        action='store',
                        dest='interp',
                        default=["NN"],
                        help="Interpolation method."
                             "Either Nearest neighbour (NN) or"
                             "Binned Stats (BS). Default is NN.",
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
                        default=[0.80, 1.25],
                        help="Detection boundary in the y-direction.",
                        required=False)
    parser.add_argument('--cut', '-cut',
                        nargs=2,
                        action='store',
                        dest='xcut',
                        default=[-20, 20],
                        help="Analysis limits in the x-direction.",
                        required=False)
    # parser.add_argument('--ycut', '-ycut',
    #                     nargs=2,
    #                     action='store',
    #                     dest='ycut',
    #                     default=[-1, 2],
    #                     help="Analysis limits in the y-direction.",
    #                     required=False)
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
    parser.add_argument('-dt', '--time-delta',
                        nargs=1,
                        action='store',
                        dest='dt',
                        help="Analysis duration in minutes. Default is 10m.",
                        default=[10],
                        required=False)
    parser.add_argument('-dx', '--horizontal-resolution',
                        nargs=1,
                        action='store',
                        dest='dx',
                        help="Horizontal resolution. Default is 1cm.",
                        required=False,
                        default=[0.01],)
    parser.add_argument('--output', '-o',
                        nargs=1,
                        action='store',
                        dest='output',
                        help="Output file name (.nc).",
                        required=True)
    parser.add_argument('--debug-profile',
                        action='store_true',
                        dest='debug_profile',
                        help="If parsed, will plot the profile.")

    args = parser.parse_args()

    print("\nExtracting timestack, please wait...\n")

    EST = timezone('Australia/Sydney')
    UTC = timezone("UTC")

    # sort out some options here
    XLIMS = [float(args.xlim[0]), float(args.xlim[1])]
    YLIMS = [float(args.ylim[0]), float(args.ylim[1])]
    XCUT = [float(args.xcut[0]), float(args.xcut[1])]
    # YCUT = [float(args.ycut[0]), float(args.ycut[1])]
    THETA = float(args.theta[0])
    DT = float(args.dt[0])
    DX = float(args.dx[0])

    outliers = "R+H"
    if args.outliers[0] != "R+H":
        print("Setting outlier detection to RANSAC + Convex Hull."
              "Other algorithms do not work yet, sorry!")

    # interpolation method
    interp = args.interp[0]

    # the main calll
    main()

    print("\n\nMy work is done!\n")
