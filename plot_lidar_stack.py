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

import xarray as xr
import pandas as pd

# Arguments
import argparse

from pywavelearn.utils import ellapsedseconds

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


def main():
    """Plot the timestack."""
    # load the dataset
    ds = xr.open_dataset(args.input[0])

    # extract variables
    secs = ellapsedseconds(pd.to_datetime(
        ds["time"].values).to_pydatetime())
    x = ds["distance"].values
    z = ds["eta"].values

    # plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_axisbelow(False)
    m = ax.pcolormesh(secs, x, z.T, vmin=0, vmax=0.3)
    cb = plt.colorbar(m, aspect=15)
    cb.set_label("Surface elevation [m]")
    ax.grid(ls="--", lw=1, color="w")
    ax.set_ylabel("Distance [m]")
    ax.set_xlabel("Time [m]")
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.savefig(args.input[0].replace(".nc", ".png"))
    plt.close()


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

    args = parser.parse_args()

    print("\nPlotting timestack, please wait...\n")

    main()

    print("\n\nMy work is done!\n")
