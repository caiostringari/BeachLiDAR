# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   :
# pourpose :
# author   : caio eadi stringari
# email    : caio.eadistringari@uon.edu.au
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
import sys

import datetime
import warnings

# data I/O
import numpy as np
import pandas as pd

from scipy import stats

from scipy.spatial.distance import *

from matplotlib.dates import date2num

from sklearn.mixture import GaussianMixture

from string import ascii_lowercase

from scipy.interpolate import Rbf

import statsmodels.nonparametric.api as sm

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2
warnings.filterwarnings("ignore")


def statsmodels_univariate_kde(data, kernel, bw, gridsize, cut, clip,
                               cumulative=False):
    """Compute a univariate kernel density estimate using statsmodels."""
    fft = kernel == "gau"
    kde = sm.KDEUnivariate(data)
    kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip)
    if cumulative:
        grid, y = kde.support, kde.cdf
    else:
        grid, y = kde.support, kde.density
    return grid, y


def gaussian_mixture(X, N=4):
    """Fit gaussian mixture to data."""

    bic = np.zeros(N)
    n = np.arange(1, N + 1)
    models = []

    # loop through each number of Gaussians and compute the BIC,
    # and save the model
    for i, j in enumerate(n):
        # create mixture model with j components
        gmm = GaussianMixture(n_components=j)
        # fit it to the data
        gmm.fit(X)
        # compute the BIC for this model
        bic[i] = gmm.bic(X)
        # add the best-fit model with j components to the list of models
        models.append(gmm)

    # best model
    i = np.argmin(bic)

    return models[i]


if __name__ == '__main__':
    print("Analysing data, please wait...\n")

    # create cblind friendly a colormap
    colors = ['#1b9e77', '#d95f02', '#7570b3']
    ev_cmap = mpl.colors.ListedColormap(colors)

    # data examples
    IDX = [2, 12, 39]

    # QB
    QB = 0.9

    # KDE parameters
    CLIP = (-np.inf, np.inf)
    CUT = 4
    KERNEL = 'gau'
    BW = 'scott'
    GRIDSIZE = 100
    DX = 0.05
    GRID = np.arange(-4, 4 + DX, DX)
    CUMULATIVE = False
    p0 = [1, 1, 0.5]

    # plotting
    plot_transform = False
    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)

    # read timeseries
    D = np.load("data/swash_timeseries.npy", allow_pickle=True).item()
    dates = D["date"]

    # read divergence data
    div = pd.read_csv("data/divergences.csv")

    # read offhsore data
    df = pd.read_csv("data/offshore_data.csv")
    of = df.copy()  # all offshore data
    time = pd.to_datetime(np.squeeze(of["time"].values)).to_pydatetime()
    idx2 = []
    for t in dates:
        idx2.append(np.argmin(np.abs(date2num(time) - date2num(t))))
    of = of.iloc[idx2]
    of = of[["Hm0", "Tm01"]].reset_index()

    # loop over timeseries and resample
    PDFs = []
    NCT = []
    timeseries = []

    for time, series, date in zip(D["time"], D["series"], D["date"]):

        # build a DataFrame and resample to 1s
        fs = np.round(time[1] - time[0], 3)
        drange = pd.date_range(start=date,
                               periods=len(series),
                               freq="{}S".format(fs))
        df = pd.DataFrame(series, index=drange).resample("1S").mean()
        # df = df.iloc[0:LEN]
        x = np.squeeze(df.values)

        # normalize
        mean = x.mean()
        std = x.std()
        x = (x - mean) / std
        timeseries.append(x)

        # compute PDF
        g, y = statsmodels_univariate_kde(x,
                                          kernel=KERNEL,
                                          clip=CLIP,
                                          bw=BW,
                                          gridsize=GRIDSIZE,
                                          cut=CUT,
                                          cumulative=CUMULATIVE)
        # interp to the same grid
        f = Rbf(g, y, function="multiquadric")
        ikde = f(GRID)
        PDFs.append(ikde)
    PDFs = np.array(PDFs)

    # --- Cluster offshore data ---

    M = GaussianMixture(3, n_init=200, init_params="random", random_state=42,
                        verbose=0, covariance_type="full")
    M.fit(of[["Hm0", "Tm01"]])
    labels = M.predict(of[["Hm0", "Tm01"]])
    C = []
    for lb in labels:
        if lb == 0:
            C.append(colors[0])
        elif lb == 1:
            C.append(colors[1])
        else:
            C.append(colors[2])

    # --- Plot ---
    # fig, ax = plt.subplots(2, 4, figsize=(12, 8))

    gs = gridspec.GridSpec(3, 4)

    ax0 = plt.subplot(gs[1, 0])  # offshore data

    ax1 = plt.subplot(gs[0, 1])  # cluster a
    ax2 = plt.subplot(gs[0, 2])  # cluster b
    ax3 = plt.subplot(gs[0, 3])  # cluster c

    ax4 = plt.subplot(gs[1, 1])  # ex cluster a
    ax5 = plt.subplot(gs[1, 2])  # ex cluster b
    ax6 = plt.subplot(gs[1, 3])  # ex cluster c

    ax7 = plt.subplot(gs[2, :2])  # divergence
    ax8 = plt.subplot(gs[2, 2:])  # divergence

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    axs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]

    # enviromental data
    axs[0].scatter(of["Hm0"], of["Tm01"], 120, c=C,
                   edgecolor="w", linewidth=1, zorder=20, alpha=0.8,)
    # plot the decision boundary
    # for that, we will assign a color to each
    x_min, x_max = of["Hm0"].min() - 2, of["Hm0"].max() + 2
    y_min, y_max = of["Tm01"].min() - 2, of["Tm01"].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = M.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axs[0].imshow(Z, interpolation='quadric',
                  extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                  cmap=ev_cmap,
                  aspect='auto', origin='lower', alpha=0.5)
    axs[0].axvline(1.56, color="k", lw=3, ls="--")
    axs[0].axhline(7.08, color="k", lw=3, ls="--")
    # set limits
    axs[0].set_xlim(0, 3)
    axs[0].set_ylim(2, 10)
    # set labels
    axs[0].set_xlabel(r"$H_{m_{0\infty}}$ $[m]$")
    axs[0].set_ylabel(r"$T_{m_{01\infty}}$ $[s]$")

    # PDFs
    # plot PDFs
    mean_1 = []
    mean_2 = []
    mean_3 = []
    for z, pdf in zip(labels, PDFs):
        if z == 0:
            axs[1].plot(GRID, pdf,  color=colors[0], alpha=0.5)
            mean_1.append(pdf)
        elif z == 1:
            axs[2].plot(GRID, pdf, color=colors[1], alpha=0.5)
            mean_2.append(pdf)
        else:
            axs[3].plot(GRID, pdf, color=colors[2], alpha=0.5)
            mean_3.append(pdf)

    # means
    axs[1].plot(GRID, np.mean(mean_1, axis=0), color="k", lw=3, label="Mean")
    axs[2].plot(GRID, np.mean(mean_2, axis=0), color="k", lw=3, label="Mean")
    axs[3].plot(GRID, np.mean(mean_3, axis=0), color="k", lw=3, label="Mean")
    # gaussians
    axs[1].plot(GRID, stats.norm.pdf(GRID, 0, 1), color="k", ls="--",
                lw=3, label=r"$\mathcal{N}(0, 1)$")
    axs[2].plot(GRID, stats.norm.pdf(GRID, 0, 1), color="k", ls="--",
                lw=3, label=r"$\mathcal{N}(0, 1)$")
    axs[3].plot(GRID, stats.norm.pdf(GRID, 0, 1), color="k", ls="--",
                lw=3, label=r"$\mathcal{N}(0, 1)$")

    # legend and limits
    for ax in [axs[1], axs[2], axs[3]]:
        lg = ax.legend(loc=1, fontsize=10)
        lg.get_frame().set_color("w")
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.75)

    axs[1].set_title("Cluster A")
    axs[2].set_title("Cluster B")
    axs[3].set_title("Cluster C")

    # plot examples
    k = 0
    M = [np.mean(mean_1, axis=0),
         np.mean(mean_2, axis=0),
         np.mean(mean_3, axis=0)]
    for i, ax, mean in zip(IDX, [axs[4], axs[5], axs[6]], M):
        sns.distplot(timeseries[i], bins=None, ax=ax, label="Data", kde=False,
                     color="k", norm_hist=True)
        sns.kdeplot(timeseries[i], ax=ax, label="KDE",
                    color="k", legend=False, lw=3)

        # fit Student's t
        params = stats.nct.fit(timeseries[i])
        y_nct = stats.nct.pdf(GRID, *params)
        ax.plot(GRID, y_nct, color="dodgerblue", label="NCT", lw=3)

        # FIT GM
        gmm = gaussian_mixture(timeseries[i].reshape(-1, 1))
        gmm_y = np.exp(gmm.score_samples(GRID.reshape(-1, 1)))
        ax.plot(GRID, gmm_y, ls="-", lw=3,
                color="orangered", label="GMM")

        # ax.plot(GRID, mean, color=colors[k], Label="C. Avg", lw=3)

        # set limits
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.75)

        # label
        ax.set_xlabel(r"($\zeta-\mu)/\sigma$ $[-]$")

        # legend
        lg = ax.legend(fontsize=10, loc=1)
        lg.get_frame().set_color("w")
        k += 1

    axs[4].set_title("Example - Cluster A")
    axs[5].set_title("Example - Cluster B")
    axs[6].set_title("Example - Cluster C")

    # letters
    for ax in axs[:-1]:
        ax.grid(color="w", lw=2, ls="-")
        sns.despine(ax=ax)
    k = 0
    for ax in axs[:-1]:
        ax.text(0.05, 0.9, ascii_lowercase[k]+")",
                transform=ax.transAxes, ha="left",
                va="center", bbox=bbox, zorder=100)
        k += 1

    axs[1].set_ylabel(r"$p((\zeta-\mu)/\sigma)$ $[-]$")
    axs[4].set_ylabel(r"$p((\zeta-\mu)/\sigma)$ $[-]$")

    # plot divergences
    df = div.loc[div["factor"] == "$p(\zeta)$"]
    sns.boxplot(data=df, x="cluster", y="divergence", ax=ax7,
                linewidth=2, color="deepskyblue",
                width=0.4)
    ax7.set_title(r"$p(\zeta)$")
    ax7.set_ylim(0, 1.0)
    ax7.set_xlabel(" ")
    sns.despine(ax=ax7)
    ax7.grid(color="w", lw=2)
    ax7.set_ylabel(r"KL-Div $[-]$")
    ax7.text(1-0.025, 0.9, "h)",
             transform=ax7.transAxes, ha="right",
             va="center", bbox=bbox, zorder=100)
    # change box colors
    for box in [0, 1, 2, 3, 4, 5]:
        mybox = ax7.artists[box]
        mybox.set_edgecolor('black')
        mybox.set_linewidth(2)

    df = div.loc[div["factor"] == "$p((\zeta-\mu)/\sigma)$"]
    sns.boxplot(data=df, x="cluster", y="divergence", ax=ax8,
                linewidth=2, color="deepskyblue",
                width=0.4)
    ax8.set_title("$p((\zeta-\mu)/\sigma)$")
    ax8.set_ylim(0, 0.25)
    ax8.set_xlabel(" ")
    sns.despine(ax=ax8)
    ax8.grid(color="w", lw=2)
    ax8.set_ylabel(r" ")
    ax8.text(1-0.025, 0.9, "i)",
             transform=ax8.transAxes, ha="right",
             va="center", bbox=bbox, zorder=100)
    for box in [0, 1, 2, 3, 4, 5]:
        mybox = ax8.artists[box]
        mybox.set_edgecolor('black')
        mybox.set_linewidth(2)

    # finalize
    fig.tight_layout()
    pos1 = ax0.get_position()  # get the original position
    pos2 = [pos1.x0, pos1.y0+0.2,  pos1.width, pos1.height]
    ax0.set_position(pos2)  # set a new position
    plt.savefig("figures/data_clustering_final.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0.2)
    plt.savefig("figures/data_clustering_final.png", dpi=300,
                bbox_inches='tight', pad_inches=0.2)
    plt.show()

    print("My work is done!\n")
