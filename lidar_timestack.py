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
from pywavelearn.utils import ellapsedseconds

import xarray as xr
import pandas as pd

# Arguments
import argparse

# Numpy and scipy
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic

# Outliers
from scipy.spatial import ConvexHull
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

# np.where warnings
warnings.filterwarnings("ignore")


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
                algorithm="convex_hull"):
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
    if algorithm == "convex_hull":
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

        # bin the data
        ybin, bin_edges, _ = binned_statistic(xf, yf, bins=bins)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        xf = bin_centers
        yf = ybin

    # detect using the upper boundary line
    elif algorithm == "upper_boundary":

        # bin the data
        ybin, bin_edges, _ = binned_statistic(x, y, bins=bins)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2

        # you gotta love linear algera! lol

        # get slope and intersect
        dx = np.diff(xpred).mean()
        dy = np.diff(ypred*YLIMS[0]).mean()
        coef = dy/dx
        intercept = (ypred*YLIMS[0])[int(len(xpred)/2)]
        line = (coef*xprof)+intercept

        # use the slope to find if the point is above or below
        # the line
        i = 0
        idxs = []
        for x, y in zip(bin_centers, ybin):
            yp = (coef*x)+intercept
            delta_y = y-yp
            if delta_y > 0:
                idxs.append(i)
            i += 1
        xf = bin_centers
        yf = ybin
        yf[idxs] = np.nan

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


def main():
    """Call the main processing algorithm."""

    # read the profile
    profile = pd.read_csv(args.profile[0])
    xprof = profile["x"].values
    yprof = profile["y"].values

    # define the bins
    dx = np.diff(xprof).mean()
    bins = np.arange(xprof[0], xprof[-1]+dx, dx)

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

    # rotation matrix
    R = rotation_transform(THETA)

    # sample frequency
    sf = int(1/(times[1]-times[0]).total_seconds())

    # timeloop
    t = 0
    Z = []
    autimes = []
    for time in times:

        print("  --- Processing {} {}".format(t+1, len(times)), end="\r")

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
                                                   algorithm=OD)
            # final coordinates
            xf_ = xcurr
            yf_ = ycurr

            # interpolate to same as xprof
            f = interp1d(xf_, yf_, fill_value="extrapolate", kind="linear")
            yf = f(xprof)
            xf = xprof

        except Exception:
            xf = xprof
            yf = yprof

        # print(xprof)
        idx = np.where(yf < yprof)[0]
        yf[idx] = yprof[idx]

        # calculate the difference between the profile and the
        # current water level
        z = np.abs(yprof-yf)

        # append
        Z.append(z)

        t += 1

    # reshape and plot
    Z = np.array(Z)
    Z[np.isnan(Z)] = 0
    secs = ellapsedseconds(pd.to_datetime(times).to_pydatetime())

    # build the netcdf file
    ds = xr.Dataset()
    # write surface elevation
    ds['eta'] = (('time', 'points'),  Z)  # surface elevation
    # write distance
    ds['distance'] = (('points'),  xf)  # surface elevation
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
                        help="Output file name (.nc).",
                        required=True)

    args = parser.parse_args()

    print("\nExtracting timestack, please wait...\n")

    EST = timezone('Australia/Sydney')
    UTC = timezone("UTC")

    XLIMS = [float(args.xlim[0]), float(args.xlim[1])]
    YLIMS = [float(args.ylim[0]), float(args.ylim[1])]
    XCUT = [float(args.xcut[0]), float(args.xcut[1])]
    YCUT = [float(args.ycut[0]), float(args.ycut[1])]
    THETA = float(args.theta[0])
    OD = "convex_hull"  # other algorithms not working properly yet

    # the main calll
    main()

    print("\n\nMy work is done!\n")
