# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : vectorize.py
# POURPOSE : vectorize lidar stacks (.svg files).
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

import subprocess

import xarray as xr
import pandas as pd

import numpy as np

from svgpathtools import svg2paths, wsvg

from pywavelearn.utils import ellapsedseconds

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from scipy.spatial import KDTree

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


def read_wavepaths(fname):
    """Read wave paths from svg file."""
    # read paths
    paths, attributes = svg2paths(fname)

    X = []
    Y = []
    Z = []
    for i, path in enumerate(paths):
        for line in path:
            # point = (line.real, line.imag)
            x1 = line.start.real
            y1 = line.start.imag
            x2 = line.end.real
            y2 = line.end.imag
            X.append(x1)
            X.append(x2)
            Y.append(y1)
            Y.append(y2)
            Z.append(i)
            Z.append(i)
    df = pd.DataFrame()
    df["x"] = X
    df["y"] = Y
    df["label"] = Z

    return df


def main():
    """Call the main script."""

    # mAHD lookup table - You will need to build yours!
    mAHD = pd.read_csv("/mnt/doc/DataHub/SMB-2018/mAHD.csv")

    # read wave paths
    df = read_wavepaths(args.svg[0])

    # read timestack
    ds = xr.open_dataset(args.netcdf[0])
    eta = ds["eta"].values
    time = ellapsedseconds(pd.to_datetime(ds["time"].values).to_pydatetime())
    dist = ds["distance"].values

    # Tree searsch for coordinates
    I, J = np.meshgrid(np.arange(0, eta.shape[0], 1),
                       np.arange(0, eta.shape[1], 1))
    # X, Y = np.meshgrid(time, dist)
    Tree = Tree = KDTree(np.vstack([I.flatten(), J.flatten()]).T)

    coords = df[["x", "y"]].values

    _, idx = Tree.query(coords, k=1)

    # unravel
    i = np.unravel_index(idx, I.shape)[0]
    j = np.unravel_index(idx, J.shape)[1]

    # get real-world coordinates
    df["time"] = time[j]
    df["dist"] = dist[::-1][i]

    # interpolate
    dt = 0.25
    Itime = []
    Idist = []
    Ilblb = []
    for l, ldf in df.groupby("label"):

        # new time
        tpred = np.arange(ldf["time"].min(), ldf["time"].max()+dt, dt)

        # OLS model
        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                          ('ols', LinearRegression(fit_intercept=False))])
        model.fit(ldf["time"].values.reshape(-1, 1), ldf["dist"].values)

        # predict
        ypred = model.predict(tpred.reshape(-1, 1))

        # cut at maximun runup
        idx = np.argmin(ypred)

        for t, y in zip(tpred[:idx], ypred[:idx]):
            Itime.append(t)
            Idist.append(y)
            Ilblb.append(l)

    # get a runup height in mAHD
    ImAH = []
    xmAHD = mAHD["distance"].values
    zmAHD = mAHD["height"].values
    for x in Idist:
        idx = np.argmin(np.abs(xmAHD-x))
        ImAH.append(zmAHD[idx])

    # final dataframe
    df = pd.DataFrame()
    df["time"] = Itime
    df["dist"] = Idist
    df["height"] = ImAH
    df["label"] = Ilblb
    df.to_csv(args.output[0])

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.pcolormesh(time, dist, eta.T)
    for _, gdf in df.groupby("label"):
        ax.plot(gdf["time"], gdf["dist"], ls="-", lw=2)
    sns.despine()
    ax.set_xlabel("Time [s")
    ax.set_ylabel("Distance [m]")
    ax.grid(color="w", ls="-", lw=2)
    fig.tight_layout()
    plt.savefig(args.output[0].replace(".csv", ".png"))
    plt.close()
    # plt.show()


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()
    # input data
    parser.add_argument('--netcdf', '-nc',
                        nargs=1,
                        action='store',
                        dest='netcdf',
                        help="Input netCDF file from lidarstack.py.",
                        required=True)
    parser.add_argument('--svg', '-svg',
                        nargs=1,
                        action='store',
                        dest='svg',
                        help="Digitized SVG file.",
                        required=True)
    parser.add_argument('--output', '-o',
                        nargs=1,
                        action='store',
                        dest='output',
                        help="Output file (.csv).",
                        required=True)

    args = parser.parse_args()

    main()



