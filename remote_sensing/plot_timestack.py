# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : plot_timestack.py
# pourpose : plot a timestack example
# author   : caio eadi stringari
# email    : caio.eadistringari@uon.edu.au
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
import warnings

# data I/O
import numpy as np
import pandas as pd
import xarray as xr
from simpledbf import Dbf5

from pywavelearn.utils import ellapsedseconds
from pywavelearn.tracking import *

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2

# quite skimage warningss
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # main()

    lim = 0.02

    # files
    f = "data/stacks/20180614_A_20180614_1900_1910.nc"

    # process timestack
    ds = xr.open_dataset(f)
    time = ellapsedseconds(ds["time"].values)
    distance = ds["distance"].values
    dx = distance.min()
    eta = ds["eta"].values

    # process wavepaths
    f = "data/tracking/SMB/20180614_A_20180614_1900_1910.dbf"
    wp = Dbf5(f, codec='utf-8').to_dataframe()
    if "newtime" in wp.columns.values:
        wp = wp[["newtime", "newspace", "wave"]]
        wp.columns = ["t", "x", "wave"]

    # unique colours
    colors = plt.cm.get_cmap("tab20", len(np.unique(wp["wave"].values)))

    # track
    T, X, L = optimal_wavepaths(wp, order=2, min_wave_period=1, N=100,
                                project=False, t_weights=1, min_slope=0)

    # rundown
    Trnd, Xrnd = project_rundown(T, X, slope=0.037, timestep=0.1, k=3)
    Tm, Xm = merge_runup_rundown(T, X, Trnd, Xrnd)
    Tsws, Xsws = swash_edge(time, distance-dx, Tm, Xm, sigma=0.1, shift=1)

    # mask land

    # pass 1
    # eta[eta < lim] = np.nan
    # pass 2
    eta_m = np.zeros(eta.shape)
    for t, x in zip(Tsws, Xsws):
        # time search
        tidx = np.argmin(np.abs(time-t))
        strip = eta[tidx, :]
        # space search
        xidx = np.argmin(np.abs(x-(distance-dx)))
        strip[:xidx] = lim
        # update
        eta_m[tidx, :] = strip
    eta_m[eta_m <= lim] = np.nan

    # open a new figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#E9E9F1")
    ax.set_axisbelow(False)

    # plot stack
    m = ax.pcolormesh(time, distance-dx, eta_m.T, cmap="Greys_r",
                      vmin=lim, vmax=0.5)
    cb = plt.colorbar(m, pad=0.01, extend='max')
    cb.set_label(r"Bore Height $[m]$")

    # plot wavepaths
    k = 0
    for t, x in zip(T, X):
        ax.plot(t, x, color=colors(k), lw=2, zorder=10, ls="--")
        ax.plot(t+1, x, color="k", lw=1, ls="--", zorder=10)
        ax.plot(t-1, x, color="k", lw=1, ls="--", zorder=10)
        k += 1
    ax.plot(Tsws, Xsws, color="r", lw=3, ls="-", zorder=10, label="Shoreline")

    ax.grid(color="w", lw=2, ls="-")

    ax.set_ylim(5, 28)
    ax.set_xlim(0, 300)

    ax.set_xlabel(r"Time $[s]$")
    ax.set_ylabel(r"Cross-shore Distance $[m]$")

    fig.tight_layout()

    sns.despine(ax=ax)
    for _, spine in ax.spines.items():
        spine.set_zorder(300)


    fig.tight_layout()
    plt.savefig("../Figures/Figure_04b.png",
                dpi=120, bbox_inches='tight', pad_inches=0.2)
    plt.close()
