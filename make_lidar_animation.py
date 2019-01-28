# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : plot_LiDAR
# POURPOSE : plot lidar data in the same reference frame as the LiDAR.
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


# Outliers
from scipy.spatial import ConvexHull
from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

import string
import random

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

# ignore NaNs
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


def random_string(n):
    """Return a random string of lenght n."""
    return ''.join(random.choice(string.ascii_letters) for m in range(n))


def main():
    """Call the main processing algorithm."""
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

    # create an temporary folder
    opath = random_string(6)
    if not os.path.isdir(opath):
        subprocess.call("mkdir {}".format(opath), shell=True)
    else:
        subprocess.call("rm -rf {}".format(opath), shell=True)
        subprocess.call("mkdir {}".format(opath), shell=True)

    # rotation matrix
    R = rotation_transform(THETA)

    # sample frequency
    sf = int(1/(times[1]-times[0]).total_seconds())

    # timeloop
    t = 0
    with tqdm(total=len(ds.time.values)) as pbar:
        for time in times:

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

            # apply the decision boundaries
            xcurr, ycurr, xbnd, ybnd = get_inliers(XYcurr[:, 0],
                                                   XYcurr[:, 1], xprof,
                                                   return_boundary=True,
                                                   algorithm=OD)

            # final coordinates
            xf = xcurr
            yf = ycurr

            # fix according to the profile
            idx = np.where(yf < yprof)[0]
            yf[idx] = yprof[idx]

            # interpolate, if asked to
            if args.interpolate:
                yn = yf[~np.isnan(yf)]
                xn = xf[~np.isnan(yf)]
                f = interpolate.interp1d(xn, yn, kind="linear",
                                         fill_value='extrapolate')
                yfi = f(xf)
                idx1 = np.argmin(np.abs(xf-xn.min()))
                idx2 = np.argmin(np.abs(xf-xn.max()))
                xi = xn[idx1:idx2]
                yi = yn[idx1:idx2]

            # cut to the plot limits
            idx1 = np.argmin(np.abs(XCUT[0]-xf))
            idx2 = np.argmin(np.abs(XCUT[1]-xf))
            xf = xf[idx1:idx2]
            yf = yf[idx1:idx2]
            xfill = xprof[idx1:idx2]
            yfill = yprof[idx1:idx2]

            # open a new figure
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_zorder(100)

            # plot raw data
            ax.scatter(XY[:, 0], XY[:, 1]+height,
                       60, facecolor="k",
                       marker="+", alpha=0.25, zorder=19,
                       label="LiDAR measurement [m]")

            # plot detection boundary
            ax.plot(xbnd, (ybnd*YLIMS[0])+height, ls="--",
                    color="r", zorder=100, label="Outlier boundary")
            ax.plot(xbnd, (ybnd*YLIMS[1])+height, ls="--",
                    color="dodgerblue", zorder=100,
                    label="Outlier boundary")

            # plot processed
            ax.scatter(xf, yf+height, 70,
                       facecolor="dodgerblue",
                       edgecolor="navy", alpha=0.5, zorder=20,
                       label="Water elevation [m]")
            if args.interpolate:
                ax.plot(xi, yi+height, color="r", ls='-', lw=1, zorder=30,
                        label="Interpolation")

            # plot profile
            ax.plot(xprof, yprof+height, lw=3, color="k", zorder=10)

            # fill water
            ax.fill_between(xfill, yfill+height, yf+height,
                            color="dodgerblue", alpha=0.5)

            # fill sand
            ax.fill_between(xprof+0.1, YCUT[0]+0.01, yprof+height,
                            interpolate=True,
                            color='#ded6c4', zorder=6, )

            # timestamp
            tb = ax.text(0.025, 0.95,
                         now.strftime("%d/%m/%Y %H:%M:%S.%f"),
                         transform=ax.transAxes,
                         color="k", ha="left",
                         va="top")
            tb.set_bbox(dict(facecolor="w", edgecolor="k", alpha=0.85))

            # vertical scale
            tb = ax.text(0.02, 0.15,
                         "Vertical Exageration: {}x".format(ASPECT),
                         transform=ax.transAxes,
                         color="k", ha="left",
                         va="top", zorder=100)
            tb.set_bbox(dict(facecolor="w", edgecolor="k", alpha=0.85))

            # legend
            lg = ax.legend(loc=1)
            lg.get_frame().set_color("w")

            ax.grid(ls="-", color="w", lw=2)
            #
            ax.set_xlim(xprof.min(), xprof.max())
            ax.set_ylim(YCUT[0], YCUT[1])

            ax.set_xlabel("Distance from LiDAR [m]")
            ax.set_ylabel("Elevation [m]")

            ax.set_aspect(ASPECT)

            sns.despine(ax=ax)
            fig.tight_layout()
            fname = time.strftime("%Y%m%d_%H%M%S.%f")
            plt.savefig(opath+"/"+fname+".png", dpi=120,
                        bbox_inches='tight', pad_inches=0.2)

            plt.close()
            pbar.update()
            # sys.exit()

            t += 1

    # animate
    cmdl1 = "ffmpeg -y -framerate 8 -pattern_type glob"
    cmdl2 = " -i \'{}/*.png\'".format(opath)
    cmdl3 = " -vf pad=\'width=ceil(iw/2)*2:height=ceil(ih/2)*2\'"
    cmdl4 = " -c:v libx264 -pix_fmt yuv420p {}".format(args.output[0])
    p1 = subprocess.Popen(cmdl1+cmdl2+cmdl3+cmdl4, shell=True)
    p1.wait()
    subprocess.call("rm -rf {}".format(opath), shell=True)

    pbar.close()


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
    parser.add_argument('--aspect-ratio', '-aspect',
                        nargs=1,
                        action='store',
                        dest='aspect',
                        help="Aspect ratio for plotting.",
                        default=[5],
                        required=False)
    parser.add_argument('--interpolate', '-inter',
                        action='store_true',
                        dest='interpolate',
                        help="Interpolate water level.")
    parser.add_argument('--output', '-o',
                        nargs=1,
                        action='store',
                        dest='output',
                        help="Output file name (.csv).",
                        required=True)

    args = parser.parse_args()

    print("\nCreating animation, please wait...\n")

    EST = timezone('Australia/Sydney')
    UTC = timezone("UTC")

    XLIMS = [float(args.xlim[0]), float(args.xlim[1])]
    YLIMS = [float(args.ylim[0]), float(args.ylim[1])]
    XCUT = [float(args.xcut[0]), float(args.xcut[1])]
    YCUT = [float(args.ycut[0]), float(args.ycut[1])]
    THETA = float(args.theta[0])
    ASPECT = int(args.aspect[0])
    OD = "convex_hull"   # other algorithms not working properly yet

    # main call
    main()

    print("\nMy work is done!\n")
