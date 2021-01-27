# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : plot_swash-by-swash_analysis.py
# pourpose : plot swash by swash stuff
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

import pycwt as wavelet

from scipy.interpolate import interp1d

from seaborn.distributions import _freedman_diaconis_bins as nbins

from scipy import stats
from scipy.interpolate import Rbf
import statsmodels.nonparametric.api as sm


from pywavelearn.stats import *
from pywavelearn.utils import peaklocalextremas
from pywavelearn.spectral import power_spectrum_density

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2

# quite skimage warningss
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


def wavelet_transform(dat, mother, s0, dj, J, dt, lims=[20, 120], t0=0):
    """
    Plot the continous wavelet transform for a given signal.

    Make sure to detrend and normalize the data before calling this funcion.

    This is a function wrapper around the pycwt simple_sample example with
    some modifications.

    ----------
    Args:
        dat (Mandatory [array like]): input signal data.

        mother (Mandatory [str]): the wavelet mother name.

        s0 (Mandatory [float]): starting scale.

        dj (Mandatory [float]): number of sub-octaves per octaves.

        j (Mandatory [float]):  powers of two with dj sub-octaves.

        dt (Mandatory [float]): same frequency in the same unit as the input.

        lims (Mandatory [list]): Period interval to integrate the local
                                 power spectrum.

        label (Mandatory [str]): the plot y-label.

        title (Mandatory [str]): the plot title.
    ----------
    Return:
        No
    """

    # also create a time array in years.
    N = dat.size
    t = np.arange(0, N) * dt + t0

    # write the following code to detrend and normalize the input data by its
    # standard deviation. Sometimes detrending is not necessary and simply
    # removing the mean value is good enough. However, if your dataset has a
    # well defined trend, such as the Mauna Loa CO\ :sub:`2` dataset available
    # in the above mentioned website, it is strongly advised to perform
    # detrending. Here, we fit a one-degree polynomial function and then
    # subtract it from the
    # original data.
    p = np.polyfit(t - t0, dat, 1)
    dat_notrend = dat - np.polyval(p, t - t0)
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset
    alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

    # the following routines perform the wavelet transform and inverse wavelet
    # transform using the parameters defined above. Since we have normalized
    # our input time-series, we multiply the inverse transform by the standard
    # deviation.
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        dat_norm, dt, dj, s0, J, mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    # calculate the normalized wavelet and Fourier power spectra, as well as
    # the Fourier equivalent periods for each wavelet scale.
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    # inverse transform but only considering lims
    idx1 = np.argmin(np.abs(period - LIMS[0]))
    idx2 = np.argmin(np.abs(period - LIMS[1]))
    _wave = wave.copy()
    _wave[0:idx1, :] = 0
    igwave = wavelet.icwt(_wave, scales, dt, dj, mother) * std

    # could stop at this point and plot our results. However we are also
    # interested in the power spectra significance test. The power is
    # significant where the ratio ``power / sig95 > 1``.
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    # calculate the global wavelet spectrum and determine its
    # significance level.
    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof,
                                            wavelet=mother)

    return t, dt, power, period, coi, sig95, iwave, igwave


def get_surfzone_data(D):
    """Compute surf zone parameters."""
    # compute Qb and get the energy ratios at Qb=0.9
    df_surf = pd.DataFrame(Dsurf)

    k = 0
    Mask = []
    Mask_time = []
    Surf_Runs = []
    Hs = []
    Tp = []
    for g, gdf in df_surf.groupby("Run"):

        # get variables
        h = gdf["Hbar"].values
        qb = gdf["Qb"].values

        # find lowest Qb
        idx = np.argmin(np.abs(qb - Qb_TRX))

        # get timeseries
        time = gdf.iloc[idx]["Time"]
        series = gdf.iloc[idx]["Series"]

        # interpolate to 1Hz
        it = np.arange(0, 600, 1)
        f = interp1d(time, series)
        eta = f(it)
        # print(time[2]-time[1])

        # compute the wavelet transform
        t, dt, power, period, coi, sig95, ieta, igeta = wavelet_transform(
            eta, MOTHER, S0, DJ, J, DT, lims=LIMS)

        # integrate and normalize the local spectrum
        idx = np.argmin(np.abs(period - LIMS[0]))
        sw = np.trapz(power[0:idx, :], dx=dt, axis=0) / power.max()
        ig = np.trapz(power[idx::, :], dx=dt, axis=0) / power.max()

        # create mask
        z = 0
        mask = np.ones(len(t))
        for i, j in zip(sw, ig):
            if i >= j:
                mask[z] = 1  # swell
            else:
                mask[z] = -1  # infragravity
            z += 1

        # append
        Mask.append(mask)
        Mask_time.append(t)
        Surf_Runs.append(g)

        # get offshore data
        hs = doff.loc[doff["Run"] == g]["Hm0"].values[0]
        tp = doff.loc[doff["Run"] == g]["Tm01"].values[0]
        Hs.append(hs)
        Tp.append(tp)

        k += 1

    return Surf_Runs, Mask, Mask_time, Hs, Tp


def get_swash_zone_data(Dswash):
    """Compute swash zone data."""
    df_swash = pd.DataFrame(Dswash)

    k = 0
    Swash_Runs = []
    Runup = []
    Runup_time = []
    for g, gdf in df_swash.groupby("date"):

        # get data
        time = np.array(gdf["time"].values[0])
        series = np.array(gdf["series"].values[0])

        # local extrema analysis
        fs = 1 / (time[1] - time[0])
        f, psd = power_spectrum_density(series, fs)
        mins, maxs = peaklocalextremas(series, lookahead=int(fs * 2),
                                       delta=0.1 * Hm0(f, psd))

        # make sure to always start and end with a local minima
        if time[mins[0]] > time[maxs[0]]:
            maxs = np.delete(maxs, 0)
        if time[mins[-1]] < time[maxs[-1]]:
            maxs = np.delete(maxs, -1)

        # compute runup heights
        runup = []
        runup_time = []
        for i in range(len(mins) - 1):

            # get time and height for each wave
            wtime = time[mins[i]:mins[i + 1]]
            wseries = series[mins[i]:mins[i + 1]]

            # compute runup height
            runup.append(wseries.max() - wseries[0])
            runup_time.append(wtime[np.argmax(wseries)])

        # append
        Runup.append(np.array(runup))
        Runup_time.append(np.array(runup_time))
        Swash_Runs.append(g.strftime("%Y%m%d-%H%M"))

    return Swash_Runs, Runup_time, Runup


if __name__ == '__main__':

    print("\nProcessing swash-by-swash data, please wait...")

    # Constants
    Qb_TRX = 0.9
    MOTHER = wavelet.MexicanHat()
    DT = 1  # 1 second
    S0 = 0.25 * DT  # starting scale, in this case 0.25*1 = 0.25 seconds
    DJ = 1 / 12  # twelve sub-octaves per octaves
    J = 8 / DJ  # eight powers of two with dj sub-octaves
    LIMS = [25, 250]

    # Labels
    labels = np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

    # KDE parameters
    CLIP = (-np.inf, np.inf)
    CUT = 4
    KERNEL = 'gau'
    BW = 'scott'
    GRIDSIZE = 100
    DX = 0.01
    NGRID = np.arange(-4, 4 + DX, DX)
    RGRID = np.arange(0, 2 + DX, DX)
    CUMULATIVE = False
    p0 = [1, 1, 0.5]

    # data examples
    # IDX = [2, 12, 39]

    # Load data
    Dsurf = np.load("data/surf_zonedata.npy", allow_pickle=True).item()
    Dswash = np.load("data/swash_timeseries.npy", allow_pickle=True).item()

    # read offhsore data
    doff = pd.read_csv("data/offshore_data.csv")

    # parse dates
    dates = pd.to_datetime(doff["time"].values).to_pydatetime()
    Run = []
    for i in dates:
        Run.append(i.strftime("%Y%m%d-%H%M"))
    doff["Run"] = Run

    print("\n  - Computing surf zone information, please wait...")
    Surf_Runs, Mask, Mask_time, Hs, Tp = get_surfzone_data(Dsurf)

    # Compute Runup heights
    print("\n  - Computing swash zone information")
    Swash_Runs, Runup_time, Runup = get_swash_zone_data(Dswash)

    # Compute PDFs at each freq
    print("\n  - Combining information")
    SW = []
    IG = []
    Ig_Cluster_0_norm = []
    Ig_Cluster_1_norm = []
    Ig_Cluster_2_norm = []
    Sw_Cluster_0_norm = []
    Sw_Cluster_1_norm = []
    Sw_Cluster_2_norm = []
    Ig_Cluster_0_raw = []
    Ig_Cluster_1_raw = []
    Ig_Cluster_2_raw = []
    Sw_Cluster_0_raw = []
    Sw_Cluster_1_raw = []
    Sw_Cluster_2_raw = []
    Rs_Ig = []
    Rs_Sw = []
    R_All = []
    n = 0
    for trnp, rnp, tmask, mask, hs, lb in zip(Runup_time,
                                              Runup,
                                              Mask_time,
                                              Mask, Hs, labels):

        # loop over runups
        Ig = []
        Sw = []
        for t, r in zip(trnp, rnp):

            # find nearest value in time
            idx = np.argmin(np.abs(t - tmask.astype(np.float)))

            if mask[idx] == 1:
                Sw.append(r)
            else:
                Ig.append(r)

            R_All.append(r)
            n += 1

        ig_norm = (Ig - np.mean(Ig)) / np.std(Ig)
        sw_norm = (Sw - np.mean(Sw)) / np.std(Sw)
        ig_raw = Ig
        sw_raw = Sw
        IG.append(ig_norm)
        SW.append(sw_norm)
        Rs_Ig.append(significant_wave_height(Ig))
        Rs_Sw.append(significant_wave_height(Sw))

        if len(sw_norm) > 1:
            # compute KDE and interpolate normalized data
            g_ig_n, y_ig_n = statsmodels_univariate_kde(ig_norm,
                                                        kernel=KERNEL,
                                                        clip=CLIP,
                                                        bw=BW,
                                                        gridsize=GRIDSIZE,
                                                        cut=CUT,
                                                        cumulative=CUMULATIVE)
            # interp to the same grid
            f = Rbf(g_ig_n, y_ig_n, function="multiquadric")
            kde_ig_n = f(NGRID)
            g_sw_n, y_sw_n = statsmodels_univariate_kde(sw_norm,
                                                        kernel=KERNEL,
                                                        clip=CLIP,
                                                        bw=BW,
                                                        gridsize=GRIDSIZE,
                                                        cut=CUT,
                                                        cumulative=CUMULATIVE)
            # interp to the same grid
            f = Rbf(g_ig_n, y_sw_n, function="multiquadric")
            kde_sw_n = f(NGRID)

            # compute KDE and interpolate normalized data
            g_ig_r, y_ig_r = statsmodels_univariate_kde(ig_raw,
                                                        kernel=KERNEL,
                                                        clip=CLIP,
                                                        bw=BW,
                                                        gridsize=GRIDSIZE,
                                                        cut=CUT,
                                                        cumulative=CUMULATIVE)
            # interp to the same grid
            f = Rbf(g_ig_r, y_ig_r, function="multiquadric")
            kde_ig_r = f(RGRID)
            g_sw_r, y_sw_r = statsmodels_univariate_kde(sw_raw,
                                                        kernel=KERNEL,
                                                        clip=CLIP,
                                                        bw=BW,
                                                        gridsize=GRIDSIZE,
                                                        cut=CUT,
                                                        cumulative=CUMULATIVE)
            # interp to the same grid
            f = Rbf(g_ig_r, y_sw_r, function="multiquadric")
            kde_sw_r = f(RGRID)

            # append based on clusters
            if lb == 0:
                Ig_Cluster_0_norm.append(kde_ig_n)
                Sw_Cluster_0_norm.append(kde_sw_n)
                Ig_Cluster_0_raw.append(kde_ig_r)
                Sw_Cluster_0_raw.append(kde_sw_r)
            elif lb == 1:
                Ig_Cluster_1_norm.append(kde_ig_n)
                Sw_Cluster_1_norm.append(kde_sw_n)
                Ig_Cluster_1_raw.append(kde_ig_r)
                Sw_Cluster_1_raw.append(kde_sw_r)
            elif lb == 2:
                Ig_Cluster_2_norm.append(kde_ig_n)
                Sw_Cluster_2_norm.append(kde_sw_n)
                Ig_Cluster_2_raw.append(kde_ig_r)
                Sw_Cluster_2_raw.append(kde_sw_r)
            else:
                print("Huston, we've got a problem")

        # break

    # compute cluster means
    Ig_Cluster_0_Mean_norm = np.mean(Ig_Cluster_0_norm, axis=0)
    Ig_Cluster_1_Mean_norm = np.mean(Ig_Cluster_1_norm, axis=0)
    Ig_Cluster_2_Mean_norm = np.mean(Ig_Cluster_2_norm, axis=0)
    Sw_Cluster_0_Mean_norm = np.mean(Sw_Cluster_0_norm, axis=0)
    Sw_Cluster_1_Mean_norm = np.mean(Sw_Cluster_1_norm, axis=0)
    Sw_Cluster_2_Mean_norm = np.mean(Sw_Cluster_2_norm, axis=0)

    Ig_Cluster_0_Mean_raw = np.mean(Ig_Cluster_0_raw, axis=0)
    Ig_Cluster_1_Mean_raw = np.mean(Ig_Cluster_1_raw, axis=0)
    Ig_Cluster_2_Mean_raw = np.mean(Ig_Cluster_2_raw, axis=0)
    Sw_Cluster_0_Mean_raw = np.mean(Sw_Cluster_0_raw, axis=0)
    Sw_Cluster_1_Mean_raw = np.mean(Sw_Cluster_1_raw, axis=0)
    Sw_Cluster_2_Mean_raw = np.mean(Sw_Cluster_2_raw, axis=0)

    # concatenate all
    IG_all = []
    for i in IG:
        for v in i:
            IG_all.append(v)
    SW_all = []
    for i in SW:
        for v in i:
            SW_all.append(v)
    IG_all = np.array(IG_all)
    SW_all = np.array(SW_all)
    IG_all[np.isnan(IG_all)] = 0
    SW_all[np.isnan(SW_all)] = 0

    # Fit distributions
    xgrid = NGRID

    # NCT
    params = stats.nct.fit(SW_all)
    sw_rf = stats.nct.pdf(xgrid, *params)
    params = stats.nct.fit(IG_all)
    ig_rf = stats.nct.pdf(xgrid, *params)

    # normal
    params = stats.norm.fit(SW_all)
    sw_wb = stats.norm.pdf(xgrid, *params)
    params = stats.norm.fit(IG_all)
    ig_wb = stats.norm.pdf(xgrid, *params)

    # kumaraswamy
    params = stats.beta.fit(SW_all)
    sw_km = stats.beta.pdf(xgrid, *params)
    params = stats.beta.fit(IG_all)
    ig_km = stats.beta.pdf(xgrid, *params)

    # plot the results

    print("\n  - Plotting the results")

    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)
    fig, axs = plt.subplots(3, 3, figsize=(12, 8))
    # raw PDF
    ax1 = axs[0][0]
    ax2 = axs[0][1]
    ax3 = axs[0][2]

    # normalized PDFs
    ax4 = axs[1][0]
    ax5 = axs[1][1]
    ax6 = axs[1][2]

    # Other plots
    ax7 = axs[2][0]
    ax8 = axs[2][1]
    ax9 = axs[2][2]

    # plot raw data

    # cluster 0
    for pdf in Ig_Cluster_0_raw:
        ax1.plot(RGRID, pdf, color="orangered", lw=2, alpha=0.7)
    for pdf in Sw_Cluster_0_raw:
        ax1.plot(RGRID, pdf, color="dodgerblue", lw=2, alpha=0.7)
    # fake legend plot
    ax1.plot(RGRID - 10, Sw_Cluster_0_raw[0], color="dodgerblue", lw=2,
             alpha=0.7, label="Sea-swell (SW)")
    ax1.plot(RGRID - 10, Ig_Cluster_0_raw[0], color="orangered", lw=2,
             alpha=0.7, label="Infragravity (IG)")
    # means
    ax1.plot(RGRID, Ig_Cluster_0_Mean_raw, color="k", lw=3, ls="--",
             label="Mean IG", zorder=20)
    ax1.plot(RGRID, Sw_Cluster_0_Mean_raw, color="k", lw=3, ls="-",
             label="Mean SW", zorder=20)
    lg = ax1.legend(loc=2, fontsize=9)
    lg.get_frame().set_color("w")

    # cluster 2
    for pdf in Ig_Cluster_1_raw:
        ax2.plot(RGRID, pdf, color="orangered", lw=2, alpha=0.3)
    for pdf in Sw_Cluster_1_raw:
        ax2.plot(RGRID, pdf, color="dodgerblue", lw=2, alpha=0.3)
    # fake legend plot
    ax2.plot(RGRID - 10, Sw_Cluster_0_raw[0], color="dodgerblue", lw=2,
             alpha=0.7, label="Sea-swell (SW)")
    ax2.plot(RGRID - 10, Ig_Cluster_0_raw[0], color="orangered", lw=2,
             alpha=0.7, label="Infragravity (IG)")
    # means
    ax2.plot(RGRID, Ig_Cluster_1_Mean_raw, color="k", lw=3, ls="--",
             label="Mean IG", zorder=20)
    ax2.plot(RGRID, Sw_Cluster_1_Mean_raw, color="k", lw=3, ls="-",
             label="Mean SW", zorder=20)
    lg = ax2.legend(loc=4, fontsize=9)
    lg.get_frame().set_color("w")

    # cluster 3
    for pdf in Ig_Cluster_2_raw:
        ax3.plot(RGRID, pdf, color="orangered", lw=2, alpha=0.7)
    for pdf in Sw_Cluster_2_raw:
        ax3.plot(RGRID, pdf, color="dodgerblue", lw=2, alpha=0.7)
    # fake legend plot
    ax3.plot(RGRID - 10, Sw_Cluster_0_raw[0], color="dodgerblue", lw=2,
             alpha=0.7, label="Sea-swell (SW)")
    ax3.plot(RGRID - 10, Ig_Cluster_0_raw[0], color="orangered", lw=2,
             alpha=0.7, label="Infragravity (IG)")
    # means
    ax3.plot(RGRID, Ig_Cluster_2_Mean_raw, color="k", lw=3, ls="--",
             label="Mean IG", zorder=20)
    ax3.plot(RGRID, Sw_Cluster_2_Mean_raw, color="k", lw=3, ls="-",
             label="Mean SW", zorder=20)
    lg = ax3.legend(loc=2, fontsize=9)
    lg.get_frame().set_color("w")

    k = 0
    letters = ["a)", "b)", "c)"]
    titles = ["Cluster A", "Cluster B", "Cluster C"]
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 12)

        ax.grid(color="w", lw=2, ls="-")

        sns.despine(ax=ax)

        ax.set_title(titles[k])

        ax.set_xlabel(r"$\varrho$ $[m]$")

        ax.text(0.95, 0.85, letters[k],
                transform=ax.transAxes, ha="right",
                va="center", bbox=bbox, zorder=100)
        k += 1
    ax1.set_ylabel(r"$p(\varrho)$ $[-]$")

    # plot normalized data

    # cluster 0
    for pdf in Ig_Cluster_0_norm:
        ax4.plot(NGRID, pdf, color="orangered", lw=2, alpha=0.7)
    for pdf in Sw_Cluster_0_norm:
        ax4.plot(NGRID, pdf, color="dodgerblue", lw=2, alpha=0.7)
    # fake legend plot
    ax4.plot(NGRID - 10, Sw_Cluster_0_norm[0], color="dodgerblue", lw=2,
             alpha=0.7, label="SW")
    ax4.plot(NGRID - 10, Ig_Cluster_0_norm[0], color="orangered", lw=2,
             alpha=0.7, label="IG")
    # means
    ax4.plot(NGRID, Ig_Cluster_0_Mean_norm, color="k", lw=3, ls="--",
             label="Mean IG", zorder=20)
    ax4.plot(NGRID, Sw_Cluster_0_Mean_norm, color="k", lw=3, ls="-",
             label="Mean SW", zorder=20)
    lg = ax4.legend(loc=2, fontsize=9)
    lg.get_frame().set_color("w")

    # cluster 2
    for pdf in Ig_Cluster_1_norm:
        ax5.plot(NGRID, pdf, color="orangered", lw=2, alpha=0.3)
    for pdf in Sw_Cluster_1_norm:
        ax5.plot(NGRID, pdf, color="dodgerblue", lw=2, alpha=0.3)
    # fake legend plot
    ax5.plot(NGRID - 10, Sw_Cluster_0_norm[0], color="dodgerblue", lw=2,
             alpha=0.7, label="SW")
    ax5.plot(NGRID - 10, Ig_Cluster_0_norm[0], color="orangered", lw=2,
             alpha=0.7, label="IG")
    # means
    ax5.plot(NGRID, Ig_Cluster_1_Mean_norm, color="k", lw=3, ls="--",
             label="Mean IG", zorder=20)
    ax5.plot(NGRID, Sw_Cluster_1_Mean_norm, color="k", lw=3, ls="-",
             label="Mean SW", zorder=20)
    lg = ax5.legend(loc=2, fontsize=9)
    lg.get_frame().set_color("w")

    # cluster 3
    for pdf in Ig_Cluster_2_norm:
        ax6.plot(NGRID, pdf, color="orangered", lw=2, alpha=0.7)
    for pdf in Sw_Cluster_2_norm:
        ax6.plot(NGRID, pdf, color="dodgerblue", lw=2, alpha=0.7)
    # fake legend plot
    ax6.plot(NGRID - 10, Sw_Cluster_0_norm[0], color="dodgerblue", lw=2,
             alpha=0.7, label="SW")
    ax6.plot(NGRID - 10, Ig_Cluster_0_norm[0], color="orangered", lw=2,
             alpha=0.7, label="IG")
    # means
    ax6.plot(NGRID, Ig_Cluster_2_Mean_norm, color="k", lw=3, ls="--",
             label="Mean IG", zorder=20)
    ax6.plot(NGRID, Sw_Cluster_2_Mean_norm, color="k", lw=3, ls="-",
             label="Mean SW", zorder=20)
    lg = ax6.legend(loc=2, fontsize=9)
    lg.get_frame().set_color("w")

    k = 0
    letters = ["d)", "e)", "f)"]
    titles = ["Cluster A", "Cluster B", "Cluster C"]
    for ax in [ax4, ax5, ax6]:
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.75)

        ax.grid(color="w", lw=2, ls="-")

        sns.despine(ax=ax)

        ax.set_title(titles[k])

        ax.set_xlabel(r"$(\varrho-\mu)/\sigma$ $[-]$")

        ax.text(0.95, 0.85, letters[k],
                transform=ax.transAxes, ha="right",
                va="center", bbox=bbox, zorder=100)
        k += 1
    ax4.set_ylabel(r"$p((\varrho-\mu)/\sigma)$ $[-]$")

    # other plots

    ax7.hist(SW_all, bins=nbins(SW_all), density=True, facecolor="k",
             edgecolor="k", alpha=0.5, label="All data", zorder=15)
    sns.kdeplot(IG_all, color="k", lw=3, label="KDE", ax=ax7, zorder=20)
    ax7.plot(xgrid, sw_rf, color="r", lw=3, ls="--", label="NCT Fit",
             zorder=21)
    ax7.plot(xgrid, sw_wb, color="indigo", lw=3,
             label="Gaussian Fit", zorder=15)
    ax7.plot(xgrid, sw_km, color="g", lw=3, label="Beta Fit", zorder=15)

    ax8.hist(IG_all, bins=nbins(IG_all), density=True, facecolor="k",
             edgecolor="k", alpha=0.5, label="All data", zorder=15)
    sns.kdeplot(IG_all, color="k", lw=3, label="KDE", ax=ax8, zorder=20)
    ax8.plot(xgrid, ig_rf, color="r", lw=3, ls="--", label="NCT Fit",
             zorder=21)
    ax8.plot(xgrid, ig_wb, color="indigo", lw=3,
             label="Gaussian Fit", zorder=15)
    ax8.plot(xgrid, ig_km, color="g", lw=3, label="Beta Fit", zorder=15)

    # same as TG1982
    sns.regplot(Hs, Rs_Ig, color="orangered", ax=ax9, label="Infragravity",
                robust=False)
    sns.regplot(Hs, Rs_Sw, color="dodgerblue", ax=ax9, label="Sea-swell",
                robust=False)

    # labels
    ax7.set_xlabel(r"$(\varrho-\mu)/\sigma$ $[-]$")
    ax7.set_ylabel(r"$p(\varrho-\mu)/\sigma)$ $[-]$")
    ax8.set_xlabel(r"$(\varrho-\mu)/\sigma$ $[-]$")
    # ax8.set_ylabel(r"$p(\varrho-\mu)/\sigma)$ $[-]$")
    ax9.set_xlabel(r"$H_{m_{0\infty}}$ $[m]$")
    ax9.set_ylabel(r"$\varrho_{sig}$ $[m]$")

    ax7.set_title("Sea-swell Band")
    ax8.set_title("Infragravity Band")
    ax9.set_title("GT1982 Comparision")

    ax7.set_xlim(-4, 4)
    ax7.set_ylim(0, 0.75)
    ax8.set_xlim(-4, 4)
    ax8.set_ylim(0, 0.75)
    ax9.set_xlim(0.05, 2)
    ax9.set_ylim(0.05, 1.5)

    letters = ["g)", "h)", "i)"]
    for k, ax in enumerate([ax7, ax8, ax9]):
        # legend
        lg = ax.legend(loc=2, fontsize=9)
        lg.get_frame().set_color("w")
        # grids
        ax.grid(lw=2, ls="-", color="w")
        sns.despine(ax=ax)
        # letters
        ax.text(0.90, 0.85, letters[k],
                transform=ax.transAxes, ha="left",
                va="center", bbox=bbox, zorder=100)
    #
    fig.tight_layout()
    plt.savefig("figures/swash_by_swash.png",
                dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.savefig("figures/swash_by_swash.pdf",
                dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()

    print("\nMy work is done!\n")
