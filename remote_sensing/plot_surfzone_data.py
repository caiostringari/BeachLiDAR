# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : plot_pdfs_and_qb.py
# pourpose : plot measured swash PDFs and Qb curves
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

from scipy.optimize import curve_fit

from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import summary_table

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sns.set_context("paper", font_scale=1.35, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2
warnings.filterwarnings("ignore")


def pl(x, a, b):
    """Fit a power law."""
    y = a*x**b
    return y


def lm(x, a):
    """Linear model without intercep."""
    y = a*x
    return y


def logit_4p(x, a, b, c, d):
    """Fit a 4 parameter logistic curve to the data."""
    y = d + ((a - d) / ((1 + (x / c)**b)))
    return y

if __name__ == '__main__':
    print("Analysing data, please wait...\n")

    QB_1 = 0.1
    QB_2 = 0.5
    QB_3 = 0.9

    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)

    # read nearshore data
    Dsurf = np.load("data/surf_zonedata.npy", allow_pickle=True).item()
    Dswash = np.load("data/swash_timeseries.npy", allow_pickle=True).item()
    timeseries = Dswash["series"]
    dates = Dswash["date"]

    # read offhsore data
    doff = pd.read_csv("data/offshore_data.csv")

    # parse dates
    odates = pd.to_datetime(doff["time"].values).to_pydatetime()
    Run = []
    for i in odates:
        Run.append(i.strftime("%Y%m%d-%H%M"))
    doff["Run"] = Run

    df_surf = pd.DataFrame(Dsurf)

    Hs_o = []
    Tp_o = []
    Hs_s = []
    Ho_a = []
    To_a = []
    Hs_inner = []
    Hs_middle = []
    Hs_outer = []
    Qb_inner = []
    Qb_middle = []
    Qb_outer = []
    Gamma = []
    Hbar = []
    Hbar_l = []
    Qb = []
    Qb_l = []
    Hue = []
    for g, gdf in df_surf.groupby("Run"):

        # get offshore data
        hs = doff.loc[doff["Run"] == g]["Hm0"].values[0]
        tp = doff.loc[doff["Run"] == g]["Tm01"].values[0]
        Hs_o.append(hs)
        Tp_o.append(tp)

        # get variables
        h = gdf["Hbar"].values
        qb = gdf["Qb"].values
        Hbar_l.append(h)
        Qb_l.append(qb)
        for i in h:
            Hbar.append(i)
            Ho_a.append(hs)
            To_a.append(tp)
            Hue.append(g)
        for i in qb:
            Qb.append(i)
        for i in gdf["Hs"].values:
            Hs_s.append(i)
        for i in gdf["Hs"].values/h:
            Gamma.append(i)

        # wave height in the surf zone
        idx = np.argmin(np.abs(qb - QB_3))
        Hs_inner.append(gdf["Hs"].values[idx])
        Qb_inner.append(qb[idx])

        idx = np.argmin(np.abs(qb - QB_2))
        Hs_middle.append(gdf["Hs"].values[idx])
        Qb_middle.append(qb[idx])

        idx = np.argmin(np.abs(qb - QB_1))
        Hs_outer.append(gdf["Hs"].values[idx])
        Qb_outer.append(qb[idx])

    Hs_o = np.array(Hs_o)
    Tp_o = np.array(Tp_o)

    # sort
    sorts = np.argsort(Hs_o)
    Hs_o = Hs_o[sorts]
    Tp_o = Tp_o[sorts]
    Hs_inner = np.array(Hs_inner)[sorts]
    Hs_middle = np.array(Hs_middle)[sorts]
    Hs_outer = np.array(Hs_outer)[sorts]

    # compute equations
    M = LinearRegression(fit_intercept=False)

    # offshore wave height
    coef = []
    intercept = []
    r1 = []
    p1 = []
    Hs_pred = []
    CI_low_1 = []
    CI_upp_1 = []
    xpred = np.arange(0, 3, 0.1)
    for H in [Hs_inner, Hs_middle, Hs_outer]:

        # fit
        M.fit(Hs_o.reshape(-1, 1), H)

        # calculate CI using bootstrap
        preds = []
        for i in range(1000):
            boot_idx = np.random.choice(len(Hs_o),
                                        replace=True, size=len(Hs_o))
            popt, pcov = curve_fit(lm, Hs_o[boot_idx], H[boot_idx],
                                   maxfev=500)
            yfit_p = lm(xpred, *popt)
            preds.append(yfit_p)
            # break
        B = np.array(preds)
        up_ci_pl = np.percentile(B, 97.5, axis=0)
        lw_ci_pl = np.percentile(B, 2.5, axis=0)

        # predict
        Hpred = M.predict(Hs_o.reshape(-1, 1))
        Hs_pred.append(M.predict(xpred.reshape(-1, 1)))

        # evaluate
        r, p = pearsonr(Hpred, H)

        # append
        intercept.append(M.intercept_)
        coef.append(M.coef_[0])
        r1.append(r)
        p1.append(p)
        CI_low_1.append(lw_ci_pl)
        CI_upp_1.append(up_ci_pl)

    # -- --  -- Plot -- -- --
    fig, ((ax_qb, ax2, ax6, ax5), (axd, ax4, ax1, ax3)) = plt.subplots(
        2, 4, figsize=(12, 7))
    scatter_kws = {"s": 80, "color": "k", "facecolor": "k", "edgecolor": "w",
                   "linewidths": 1, "alpha": 0.75}

    # ---- Plot Qb Example ----
    colors = sns.color_palette("colorblind", 3).as_hex()[::-1]

    x = np.array(Hbar_l[-2])
    y = np.array(Qb_l[-2])

    # regression
    qb_xpred = np.arange(0, 5., 0.05)
    popt, pcov = curve_fit(logit_4p, x, y,
                           maxfev=500)
    qb_ypred = logit_4p(qb_xpred, *popt)
    qb_ypred[qb_ypred <= 0] = 0

    # segment Qb
    idx_inner = np.where(x <= 1)[0]
    idx_mid = np.where(x > 1)[0]
    idx_outer = np.where(x >= 1.75)[0]

    ax_qb.plot(qb_xpred, qb_ypred, lw=3, color="k")
    ax_qb.scatter(x[idx_inner], y[idx_inner], 100, facecolor=colors[0],
                  edgecolor="w", linewidths=1, alpha=1, zorder=20,
                  label="Inner Surf Zone")
    ax_qb.scatter(x[idx_mid], y[idx_mid], 100, facecolor=colors[1],
                  edgecolor="w", linewidths=1, alpha=1, zorder=20,
                  label="Mid Surf Zone")
    ax_qb.scatter(x[idx_outer], y[idx_outer], 100, facecolor=colors[2],
                  edgecolor="w", linewidths=1, alpha=1, zorder=20,
                  label="Inner Surf Zone")

    lg = ax_qb.legend(loc=3, fontsize=10)
    lg.get_frame().set_color("w")

    ax_qb.set_xlabel(r"$h$ $[-]$")
    ax_qb.set_ylabel(r"$Q_b$ $[-]$")

    ax_qb.set_xlim(0, 2.5)
    ax_qb.set_ylim(0, 1)
    ax_qb.grid(lw=2, color="w", ls="-")
    sns.despine(ax=ax_qb)

    ax_qb.text(0.95, 0.9, "a)",
               transform=ax_qb.transAxes, ha="right",
               va="center", bbox=bbox, zorder=100)

    # ---- Plot general trends ----

    # ------------------------------------------------------------------------
    # Hs x Tp x Hb

    ax1.scatter(doff["Hm0"], doff["Tm01"], 90, color="k", marker="+",
                linewidth=3, alpha=0.5, label="Offshore data")
    m = ax1.scatter(Hs_o, Tp_o, 120, c=Hs_outer, vmin=0, vmax=1,
                    cmap="viridis", edgecolor="w", linewidth=1,
                    label="Collocated data", alpha=0.9)

    # draw a colorbar
    cax = inset_axes(ax1, width="40%", height="4%", loc=4,
                     bbox_to_anchor=(0.0, 0.1, 1, 1),
                     bbox_transform=ax1.transAxes)
    cb = plt.colorbar(mappable=m, ax=ax1, cax=cax, orientation="horizontal",
                      )
    cb.set_ticks([1, 0.5, 0, -0.5, -1])
    cb.set_ticklabels([1, 0.5, 0, -0.5, -1])
    cb.ax.tick_params(labelsize=12)
    cb.set_label(r"$H_b$ $[m]$")
    cb.ax.xaxis.set_label_position('top')

    ax1.set_xlim(0, 3)
    ax1.set_ylim(2, 10)

    ax1.axvline(1.56, color="k", lw=2, ls="--", label="L.T. Means", zorder=2)
    ax1.axhline(7.08, color="k", lw=2, ls="--", zorder=3)

    lg = ax1.legend(loc=1)
    lg.get_frame().set_color("w")
    ax1.set_xlabel(r"$H_{m0_{\infty}}$ $[m]$")
    ax1.set_ylabel(r"$T_{m01_{\infty}}$ $[s]$")
    ax1.text(0.05, 0.9, "f)",
             transform=ax1.transAxes, ha="left",
             va="center", bbox=bbox, zorder=100)

    # ------------------------------------------------------------------------

    # Qb Histograms
    colors = sns.color_palette("colorblind", 3).as_hex()
    sns.distplot(Qb_outer, color=colors[0], ax=ax2, label=r"Outer Surf Zone")
    sns.distplot(Qb_middle, color=colors[1], ax=ax2, label=r"Mid Surf Zone")
    sns.distplot(Qb_inner, color=colors[2], ax=ax2, label=r"Inner Surf Zone")
    lg = ax2.legend()
    lg.get_frame().set_color("w")

    ax2.set_xlim(0, 1)
    ax2.set_xlabel(r"$Q_b$ $[-]$")
    ax2.set_ylabel(r"$p(Q_b$) $[-]$")
    ax2.text(0.05, 0.1, "b)",
             transform=ax2.transAxes, ha="left",
             va="center", bbox=bbox, zorder=100)

    # ------------------------------------------------------------------------

    # Ho/h x gamma
    x = np.array(Ho_a)/np.array(Hbar)
    y = np.array(Gamma)
    xfit = np.linspace(0, 4, 250)

    # fit power law
    popt, pcov = curve_fit(pl, x, y,
                           maxfev=500)
    yfit_p = pl(xfit, *popt)
    ypreds = pl(x, *popt)
    residuals = y - pl(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    R = 1 - (ss_res / ss_tot)

    # compute confidence interval
    preds = []
    for i in range(1000):
        boot_idx = np.random.choice(len(x), replace=True, size=len(x))
        popt, pcov = curve_fit(pl, x[boot_idx], y[boot_idx],
                               maxfev=500)
        yfit_p = pl(xfit, *popt)
        preds.append(yfit_p)
        # break
    B = np.array(preds)
    up_ci_pl = np.percentile(B, 97.5, axis=0)
    lw_ci_pl = np.percentile(B, 2.5, axis=0)

    # fit linear model
    M = LinearRegression(fit_intercept=True)
    M.fit(x.reshape(-1, 1), y)
    yfit_l = M.predict(xfit.reshape(-1, 1))
    r, p = pearsonr(y, M.predict(x.reshape(-1, 1)))

    ax3.scatter(x, y, 80, facecolor="k", edgecolor="w",
                linewidths=1, alpha=0.75)
    ax3.plot(xfit, yfit_l, color="dodgerblue", lw=3,
             label=r"$\gamma_{sig}=0.15(H_{m0_{\infty}}/\overline{h})+0.17$")
    sns.regplot(x, y, ax=ax3, color="dodgerblue", scatter_kws={"alpha": 0})

    lb = r"$\gamma_{sig}=0.37(H_{m0_{\infty}}/\overline{h})^{0.49}$"
    ax3.plot(xfit, yfit_p, lw=3, color="orangered", ls="-", label=lb,
             zorder=10)
    ax3.fill_between(xfit,  up_ci_pl, lw_ci_pl, facecolor="r",
                     alpha=0.25, zorder=9, edgecolor="none")

    ax3.text(0.95, 0.35, r"$R^2=0.60$", color="orangered",
             transform=ax3.transAxes, ha="right",
             va="center", bbox=bbox, zorder=100, fontsize=12)

    ax3.text(0.95, 0.25, r"$r_{xy}=0.65, p \ll 0.05$", color="dodgerblue",
             transform=ax3.transAxes, ha="right",
             va="center", bbox=bbox, zorder=100, fontsize=12)

    lg = ax3.legend(fontsize=10)
    lg.get_frame().set_color("w")

    ax3.set_xlim(0, 4)
    ax3.set_ylim(0, 1)

    ax3.set_ylabel(r"$\gamma_{sig}$ $[-]$")
    ax3.set_xlabel(r"$H_{m0_{\infty}}/\overline{h}$")
    ax3.text(0.95, 0.1, "g)",
             transform=ax3.transAxes, ha="right",
             va="center", bbox=bbox, zorder=100)

    k = 0
    for ax in [ax1, ax2, ax3]:
        ax.grid(color="w", lw=2, ls="-")
        sns.despine(ax=ax)

        k += 1

    # ------------------------------------------------------------------------
    # Hs Regressions

    ax4.plot(xpred, Hs_pred[0], color="r", lw=3)
    ax4.fill_between(xpred,  CI_upp_1[0], CI_low_1[0], facecolor="r",
                     alpha=0.25, zorder=5, edgecolor="none")
    ax4.scatter(Hs_o, Hs_inner, 80, facecolor="k", edgecolor="w",
                linewidths=1, alpha=0.75)

    ax5.plot(xpred, Hs_pred[1], color="r", lw=3)
    ax5.fill_between(xpred,  CI_upp_1[1], CI_low_1[1], facecolor="r",
                     alpha=0.25, zorder=5, edgecolor="none")
    ax5.scatter(Hs_o, Hs_middle, 80, facecolor="k", edgecolor="w",
                linewidths=1, alpha=0.75)

    ax6.plot(xpred, Hs_pred[2], color="r", lw=3)
    ax6.fill_between(xpred,  CI_upp_1[2], CI_low_1[2], facecolor="r",
                     alpha=0.25, zorder=5, edgecolor="none")
    ax6.scatter(Hs_o, Hs_outer, 80, facecolor="k", edgecolor="w",
                linewidths=1, alpha=0.75)

    # add equations and letters
    letters = ["e)", "d)", "c)"]
    for k, ax in enumerate([ax4, ax5, ax6]):

        txt = r"$H_{m0}$ = " + \
            r"${0:.2f}$".format(coef[k]) + r"$H_{m0_{\infty}}$"
        ax.text(0.05, 0.9, txt,
                transform=ax.transAxes, ha="left",
                va="center", bbox=bbox, zorder=100, fontsize=12)

        if p1[k] < 0.05:
            txt = r"$r_{xy}=$" + r"{0:.2}".format(r1[k]) + r", $p \ll 0.05$"
        else:
            txt = r"$r_{xy}=$" + \
                r"{0:.2}".format(r1[k]) + r", $p=$" + "{0:.3f}".format(p1[k])
        ax.text(0.05, 0.775, txt,
                transform=ax.transAxes, ha="left",
                va="center", bbox=bbox, zorder=100, fontsize=12)

        ax.text(0.95, 0.1, letters[k],
                transform=ax.transAxes, ha="right",
                va="center", bbox=bbox, zorder=100)

    ax4.set_title(r"Inner Surf Zone $(Q_b=0.95)$")
    ax5.set_title(r"Mid Surf Zone $(Q_b=0.50)$")
    ax6.set_title(r"Outer Surf Zone $(Q_b=0.05)$")

    for ax in [ax4, ax5, ax6]:

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)

        ax.set_xlabel(r"$H_{m0_{\infty}}$")
        ax.set_ylabel(r"$H_{m0}$")

        ax.grid(color="w", lw=2, ls="-")
        sns.despine(ax=ax)

    fig.delaxes(axd)
    fig.tight_layout()
    # plt.subplots_adjust(hspace=0.3)
    l, b, w, h = ax_qb.get_position().bounds
    ax_qb.set_position([l, b-0.25, w, h], "both")
    plt.savefig("figures/saturation_analysis.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0.2)
    plt.savefig("figures/saturation_analysis.png", dpi=300,
                bbox_inches='tight', pad_inches=0.2)
    plt.show()

    print("\nMy work is done!\n")
