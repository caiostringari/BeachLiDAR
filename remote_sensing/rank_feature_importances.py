# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : predict_pdf_params.py
# POURPOSE : predict swash PDF using kown variables
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 06/09/2018 [Caio Stringari]
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import warnings
from matplotlib.dates import date2num

import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture

from sklearn.tree import DecisionTreeRegressor

# plotting
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from string import ascii_lowercase
sns.set_context("paper", font_scale=1.8, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2

# quite scipy warningss
warnings.filterwarnings("ignore")


def gaussian_mixture(X, N=20):
    """Fit gaussian mixture to data."""

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


if __name__ == '__main__':
    print("Analysing data, please wait...\n")

    # load surf zone data
    sdf = pd.read_csv("data/surfzone_data.csv")
    Qb_tartgets = [0.05, 0.5, 0.95]

    # load swsh zone data
    D = np.load("data/swash_timeseries.npy", allow_pickle=True).item()

    # load measured offshore data
    odf = pd.read_csv("data/offshore_data.csv")
    offshore_time = pd.to_datetime(np.squeeze(odf["time"].values)).to_pydatetime()

    # output variables
    hs_offshore = []
    tp_offshore = []
    hs_surf_05 = []
    tp_surf_05 = []
    hs_surf_50 = []
    tp_surf_50 = []
    hs_surf_95 = []
    tp_surf_95 = []
    mixtures = []

    for time, series, date in zip(D["time"], D["series"], D["date"]):

        # offshore data
        idx = np.argmin(np.abs(date2num(date) - date2num(offshore_time)))
        hs_offshore.append(odf.iloc[idx]["Hm0"])
        tp_offshore.append(odf.iloc[idx]["Tm01"])

        # surf zone data
        key = date.strftime("%Y%m%d-%H%M")
        _sdf = sdf.loc[sdf["Run"] == key]

        idx_05 = np.argmin(np.abs(_sdf["Qb"].values - 0.05))
        idx_50 = np.argmin(np.abs(_sdf["Qb"].values - 0.50))
        idx_95 = np.argmin(np.abs(_sdf["Qb"].values - 0.95))

        hs_surf_05.append(_sdf.iloc[idx_05]["Hs"])
        tp_surf_05.append(_sdf.iloc[idx_05]["Tp"])
        hs_surf_50.append(_sdf.iloc[idx_50]["Hs"])
        tp_surf_50.append(_sdf.iloc[idx_05]["Tp"])
        hs_surf_95.append(_sdf.iloc[idx_95]["Hs"])
        tp_surf_95.append(_sdf.iloc[idx_05]["Tp"])

        # compute number of mixtures
        series = np.array(series)
        series = (series-series.mean())/series.std()
        gmm = gaussian_mixture(series.reshape(-1, 1))
        mixtures.append(gmm.n_components)

        # break

    # prepare data for classification
    X = np.vstack([hs_offshore, tp_offshore,
                   hs_surf_05, tp_surf_05,
                   hs_surf_50, tp_surf_50,
                   hs_surf_95, tp_surf_95]).T
    columns = [r"$H_{m0_{\infty}}$", r"$T_{m01_{\infty}}$",
               r"$H_{m0_{5\%}}$", r"$T_{m01_{5\%}}$",
               r"$H_{m0_{50\%}}$", r"$T_{m01_{50\%}}$",
               r"$H_{m0_{95\%}}$", r"$T_{m01_{95\%}}$"]
    df = pd.DataFrame(X, columns=columns)
    y = np.array(mixtures)

    dfs = []
    for i in range(100):
        print("Fitting model ", i, "of 100", end="\r")

        # build the regressor - this is intentionally overfit
        M = DecisionTreeRegressor(max_depth=32)
        M.fit(df, y)
        pred = M.predict(df)

        # build a dataframe
        fi = pd.DataFrame(M.feature_importances_*100, columns=["feature_importance"])
        fi["feature"] = columns
        fi = fi.sort_values(by="feature_importance", ascending=False)
        fi["run"] = i
        dfs.append(fi)
    # merge
    FI = pd.concat(dfs)
    order = FI.groupby("feature").mean().sort_values(by="feature_importance", ascending=False).index.values

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=FI, x="feature", y="feature_importance", ax=ax,
                palette="Blues_d", saturation=0.9, errcolor='r', capsize=0.1,
                order=order)

    sns.despine(ax=ax)

    labels = []
    for l in ax.get_xticklabels():
        labels.append(l.get_text())
    ax.set_xticklabels(labels, rotation=90, fontsize=18)

    ax.grid(color="w", ls="-", lw=2)

    ax.set_ylim(0, 35)

    ax.set_ylabel(r"Feature importance $[\%]$")
    ax.set_xlabel(r"")

    fig.tight_layout()
    plt.savefig("figures/feature_importances_data_only.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.close()
    print("My work is done!")
