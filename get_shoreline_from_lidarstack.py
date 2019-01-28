# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : get_shoreline_from_lidar.py
# POURPOSE : extract the time evolution of the shoreline from a lidar
#            timestack
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.1     : 28/06/2018 [Caio Stringari]
# v1.2     : 19/10/2018 [Caio Stringari]
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# Arguments
import argparse

import cv2

import xarray as xr
import pandas as pd

import numpy as np

from pywavelearn.utils import ellapsedseconds

from sklearn.preprocessing import MinMaxScaler

from skimage import measure
from skimage.filters import sobel, sobel_h, sobel_v


from scipy.signal import find_peaks

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


def get_analysis_locations(dist, start, end, step=1):
    """
    Select analysis locations based on the analysis domain and user defined
    step.

    ----------
    Args:
        stk_dist (Mandatory [np.array]): Space array [m].

        start, end (Mandatory [float]): start and end of the spatial domain,
        use get_analysis_domain() to obtain this values.

        step (Optional [int]): skip step. Default is not to skip anything.

    ----------
    Return:
        Y (Mandatory [np.array]): Cross-shore locations in meters.

        Idx (Mandatory [np.array]): Cross-shore locations indexes.
    """
    space_step = step
    Y = []
    Idx = []
    for y in dist[::step]:
        idx = np.argmin(np.abs(dist - y))
        # append only values in the surf zone
        if y > start and y < end:
            Y.append(y)
            Idx.append(idx)
    Y = Y[::-1]
    Idx = Idx[::-1]

    return Y, Idx


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
                        help="Input profile.",
                        required=True)

    args = parser.parse_args()

    MIN = 0.01

    print("\nLearning shoreline, please wait...\n")



    # load the dataset
    ds = xr.open_dataset(args.input[0])

    # extract variables
    secs = ellapsedseconds(pd.to_datetime(
        ds["time"].values).to_pydatetime())
    x = ds["distance"].values
    Z = ds["eta"].values

    # apply the first threshold
    Z[Z < MIN] = 0

    # Find contours at a constant value of 0.8
    # contours = measure.find_contours(Z, 0.02, fully_connected="high")

    # plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    m = ax.pcolormesh(secs, x, Z.T, cmap="Greys_r")
    ax.contour(secs, x, Z.T, np.arange(0.02, 0.06, 0.01), cmap="Reds_r")
    cb = plt.colorbar(m, aspect=15)
    cb.set_label("Surface elevation [m]")
    # ax.grid(ls="--", lw=1, color="w")
    ax.set_ylabel("Distance [m]")
    ax.set_xlabel("Time [m]")
    sns.despine(ax=ax)
    fig.tight_layout()

    plt.show()
    # plt.savefig(args.input[0].replace(".nc", ".png"))
    # plt.close()

    # main()

    print("\n\nMy work is done!\n")
