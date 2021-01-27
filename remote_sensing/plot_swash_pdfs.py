# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : plot_swash_pdfs.py
# pourpose : plot a measured swash PDFs
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

from scipy import stats
from scipy.stats import gaussian_kde

from sklearn.mixture import GaussianMixture

from seaborn.distributions import _freedman_diaconis_bins as nbins

# from scipy.stats import norm
# import statsmodels.nonparametric.api as smnp

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2
warnings.filterwarnings("ignore")


def gaussian_mixture(X, N=16):

    bic = np.zeros(N)
    n = np.arange(1, N+1)
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


def aliases(fname="/home/stringari/DocHub/scipy_pdfs_aliases.csv"):
    """Read PDFs names."""
    df = pd.read_csv(fname)
    return df


def ismultimodal(x, xgrid, bandwidth=0.1, threshold=0.1, plot=False, **kwargs):
    """
    Compute if sample data is unimodal using gaussian kernel density funcions.

    ----------
    Args:
        x [Mandatory (np.array)]: Array with water levels

        xgrid [Mandatory (np.array)]: Array of values in which the KDE
                                      is computed.

        bandwidth [Mandatory (np.array)]: KDE bandwith. Note that scipy weights
                                          its bandwidth by the covariance of
                                          the input data. To make the results
                                          comparable to the other methods,
                                          we divide the bandwidth by the sample
                                          standard deviation.

        threshold [Optional (float)]: Threshold for peak detection.
                                      Default is 0.1

        plot [Optional (bool)]: If True, plot the results. Default is false

        **kwargs [Optional (dict)]: scipy.stats.gausian_kde kwargs

    ----------
    Returns:
        multimodal [Mandatory (bool): If True, distribution is multimodal.

        npeaks [Mandatory (bool)]: number of peaks in the distribution.
    """
    #

    # start multimodal as false
    multimodal = 0

    # compute gaussian KDE
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1),
                       **kwargs).evaluate(xgrid)

    # compute how many peaks in the distribution
    root_mean_square = np.sqrt(np.sum(np.square(kde) / len(kde)))

    # compute peak to average ratios
    ratios = np.array([pow(x / root_mean_square, 2) for x in kde])

    # apply first order logic
    peaks = (
        ratios > np.roll(ratios, 1)) & (ratios > np.roll(
            ratios, -1)) & (ratios > threshold)

    # optional: return peak indices
    peak_indexes = []
    for i in range(0, len(peaks)):
        if peaks[i]:
            peak_indexes.append(i)
    npeaks = len(peak_indexes)

    # if more than one peak, distribution is multimodal
    if npeaks > 1:
        multimodal = 1

    if plot:
        fig, ax = plt.subplots()
        plt.plot(xgrid, kde, "-k", lw=3, zorder=10)
        plt.scatter(xgrid[peaks], kde[peaks], 120, "r", zorder=11)
        plt.hist(x, bins=20, normed=True, alpha=0.5)
        ax.grid()
        plt.show()

    return multimodal, npeaks


if __name__ == '__main__':
    print("Analysing data, please wait...\n")

    # read PDFs names aliases
    alias = aliases()

    D = np.load("data/swash_timeseries.npy", allow_pickle=True).item()
    # df = pd.read_csv("data/offshore_and_slope_data.csv")
    timeseries = D["series"]
    dates = D["date"]

    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)
    x = np.arange(-4, 4, 0.01)
    fig, axs = plt.subplots(10, 4, figsize=(8.27, 11.69),
                            sharex=True, sharey=True)
    k = 0
    axs = axs.flatten()
    Multimodal = []
    for ax, ts, date in zip(axs, timeseries, dates):

        # normalize
        ts = np.array(ts)
        series = (ts-ts.mean())/ts.std()

        print("  - Plotting timeseries {} of {}".format(k+1, len(axs)),
              end="\r")
        sns.distplot(series, bins=None, ax=ax, label="Histogram", kde=False,
                     color="k", norm_hist=True)
        sns.kdeplot(series, ax=ax, label="KDE", color="k", legend=False)

        # fit Student's t
        params = stats.nct.fit(series)
        y_nct = stats.nct.pdf(x, *params)
        ax.plot(x, y_nct, color="dodgerblue", label="Non Centered Student's T")

        # fit GMM
        gmm = gaussian_mixture(series.reshape(-1, 1))

        # plot GMM
        gmm_y = np.exp(gmm.score_samples(x.reshape(-1, 1)))
        ax.plot(x, gmm_y, ls="--",
                color="orangered", label="GMM")

        ax.grid(color="w", lw=2, ls="-")

        # add date
        txt = pd.to_datetime(date).to_pydatetime().strftime("%d/%m %H:%M")
        ax.text(0.95, 0.85, txt,
                transform=ax.transAxes, ha="right",
                va="center", bbox=bbox, zorder=100,
                fontsize=10)

        sns.despine(ax=ax)

        # verify if multimodal
        multimodal, npeaks = ismultimodal(series, x)
        Multimodal.append(multimodal)

        k += 1

    # limits
    axs[0].set_xlim(-4, 4)
    axs[0].set_ylim(0, 0.75)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    lg = fig.legend(handles, labels, loc="lower center",
                    ncol=4,)
    lg.get_frame().set_color("w")
    lg.get_frame().set_edgecolor("k")

    for ax in axs[np.arange(0, 40, 4)]:
        ax.set_ylabel(r"$p(\zeta)$")
    for ax in axs[-4:]:
        ax.set_xlabel(r"($\zeta-\mu)/\sigma$")

    md = np.where(np.array(Multimodal) == 1)[0]
    pmd = 100*(len(md)/len(Multimodal))
    print("\n\n {}% of the PDFs are multimodal".format(pmd))

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.070, wspace=None, hspace=None)
    plt.savefig("figures/pdf_fits.pdf", dpi=300, bbox_inches='tight',
                pad_inches=0.2)
    # plt.show()
    plt.close()

    print("\nMy work is done!\n")
