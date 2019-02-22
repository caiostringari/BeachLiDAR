# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : lidarstack_to_qgis.py
# POURPOSE : Convert files to qgis format.
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.1     : 31/01/2019 [Caio Stringari]
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import argparse

import os
import sys

import subprocess

import xarray as xr
import pandas as pd

import numpy as np

from skimage import measure

from pywavelearn.utils import ellapsedseconds

import matplotlib.pyplot as plt


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()
    # input data
    parser.add_argument('--input', '-i',
                        nargs=1,
                        action='store',
                        dest='input',
                        help="Input netCDF file from lidarstack.py.",
                        required=True)
    parser.add_argument('--path', '-p',
                        nargs=1,
                        action='store',
                        dest='path',
                        help="Output file path.",
                        required=True)
    parser.add_argument('--variable', '-var',
                        nargs=1,
                        action='store',
                        dest='var',
                        help="Which variable to rasterize.",
                        required=False,
                        default=["eta"])
    # parser.add_argument('--mask', '-m',
    #                     action='store_true',
    #                     dest='mask',
    #                     help="Mask no changing values if parsed.",
    #                     required=False)
    # parser.add_argument('--tolerance', '-tol',
    #                     nargs=1,
    #                     action='store',
    #                     dest='tol',
    #                     help="Tolerance for masking. Default is 1cm.",
    #                     required=False,
    #                     default=[0.01])

    print("\nRasterizing, please wait...")

    args = parser.parse_args()

    # load the dataset
    inp = args.input[0]
    ds = xr.open_dataset(inp)
    var = args.var[0]
    # tol = float(args.tol[0])

    # output locations
    path = args.path[0]
    subprocess.call("mkdir -p {}".format(path), shell=True)

    # get raster
    eta = ds[var].values

    # get coordinates
    time = ellapsedseconds(pd.to_datetime(ds["time"].values).to_pydatetime())
    dist = ds["distance"].values

    istr = inp.split("/")[-1]
    fname = os.path.join(path, istr.strip("nc").strip("/").strip(".")+".jpeg")

    # dump stack to a jpeg
    fig = plt.figure(frameon=False)
    fig.set_size_inches(len(time)/100, len(dist)/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.pcolormesh(time, dist, ds[var].values.T, cmap="inferno")
    # plt.show()
    fig.savefig(fname, dpi=100)
    plt.close("all")

    print("My work is done!\n")
