"""
Nicholas M. Boffi

This file contains code to plot the rank and expected fitness increment
as a function of time or fitness for several different curves as we vary the
epistasis parameter $\beta$.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import typing
import sys
from typing import Tuple
from common import *
import pickle
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import sem
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import quad, odeint
from seaborn import cubehelix_palette
from math import isclose
import glob


sns.set_style("white")
mpl.rcParams['axes.grid']  = True
mpl.rcParams['axes.grid.which']  = 'both'
mpl.rcParams['xtick.minor.visible']  = True
mpl.rcParams['ytick.minor.visible']  = True
mpl.rcParams['xtick.minor.visible']  = True
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['grid.color'] = '0.8'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['figure.titlesize'] = 30
mpl.rcParams['font.size'] = 27.5
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.dpi'] = 125


def downsample(times: list,
               fits: list,
               replicate_mean_incs: list,
               replicate_sem_incs: list,
               replicate_mean_ranks: list,
               replicate_sem_ranks: list,
               start_index: int,
               plot_skip: int):
    """ Downsample the trajectory data so the plots take up less memory.
    Preserve the initial datapoints so that plotting on a logarithmic
    scale still seems reasonable."""

    def single_downsample(arr: np.ndarray):
        return np.concatenate((arr[:start_index], arr[start_index::plot_skip]))

    for ii in range(len(times)):
        times[ii] = single_downsample(times[ii])
        fits[ii] = single_downsample(fits[ii])
        replicate_mean_incs[ii] = single_downsample(replicate_mean_incs[ii])
        replicate_sem_incs[ii] = single_downsample(replicate_sem_incs[ii])
        replicate_mean_ranks[ii] = single_downsample(replicate_mean_ranks[ii])
        replicate_sem_ranks[ii] = single_downsample(replicate_sem_ranks[ii])

    return times, fits, replicate_mean_incs, replicate_sem_incs, \
        replicate_mean_ranks, replicate_sem_ranks


def get_data(load_dat: bool,
             data_folders: list,
             inner_folder: str,
             bac_str: str,
             mut_str: str,
             nexps: int):
    """ Either load the pre-computed data from a .pickle file
    or compute the data. """
    # load in the fitness data for each experiment.
    replicate_mean_incs, replicate_sem_incs, replicate_mean_ranks, \
        replicate_sem_ranks, fits, times = [], [], [], [], [], []
    if not load_dat:
        for beta_ind, data_folder in enumerate(data_folders):
            # load in the bacteria and mutation data
            bac_files = get_files([data_folder], inner_folder, bac_str, nexps)
            mut_files = get_files([data_folder], inner_folder, mut_str, nexps)

            # compute the time points for this beta value
            ntimes = min([len(bac_files_list) for bac_files_list in bac_files])
            dt = float(bac_files[0][1].split('.')[-2])
            times.append(np.arange(ntimes)*dt)

            # load in the mean increments and ranks
            mean_ranks, mean_incs, _, _, _, _, _ = \
                get_rank_eff_and_select_info(nexps, 1, L, mut_files, bac_files)

            # load in the fitnessses
            mean_fits, _, _= get_fit_data(nexps, 1, ntimes, bac_files)
            fits.append(np.mean(mean_fits, axis=0))

            # compute the mean and SEM for the increment
            replicate_mean_incs.append(np.mean(mean_incs, axis=0))
            replicate_sem_incs.append(sem(mean_incs, axis=0))

            # and for the ranks
            replicate_mean_ranks.append(np.mean(mean_ranks, axis=0))
            replicate_sem_ranks.append(sem(mean_ranks, axis=0))

            # print diagnostics
            print('Finished loading mechanism on epistasis value %d/%d' \
                  % (beta_ind+1, len(data_folders)))

        data_dict = {}
        data_dict['fits'] = fits
        data_dict['mean_incs'] = replicate_mean_incs
        data_dict['sem_incs'] = replicate_sem_incs
        data_dict['mean_ranks'] = replicate_mean_ranks
        data_dict['sem_ranks'] = replicate_sem_ranks
        data_dict['times'] = times
        pickle.dump(data_dict, open("%s/figs/mechanism.pickle" \
                                    % base_folder, "wb"))

    else:
        data_dict = pickle.load(open("%s/figs/mechanism.pickle" \
                                     % base_folder, "rb"))
        replicate_mean_incs = data_dict['mean_incs']
        replicate_mean_ranks = data_dict['mean_ranks']
        replicate_sem_incs = data_dict['sem_incs']
        replicate_sem_ranks = data_dict['sem_ranks']
        fits = data_dict['fits']
        times = data_dict['times']

    return times, fits, replicate_mean_incs, replicate_mean_ranks, \
        replicate_sem_incs, replicate_sem_ranks


def plot_increment(times: list,
                   replicate_mean_incs: list,
                   replicate_sem_incs: list,
                   xlabel: str):
    """ Visualize the expected beneficial increment as a function of time. """
    fig, ax = plt.subplots()

    for ii, mu_val in enumerate(mu_vals):
        # plot the main figure
        inds = replicate_mean_incs[ii] >= 0
        power = int(np.log(mu_val) / np.log(10) - 0.5)
        label_str = r"$\mu=10^{%d}$" % power
        plt.plot(times[ii][inds], replicate_mean_incs[ii][inds],
                 color=cmap_fits[ii+2], label=label_str, lw=2.5)
        ax.fill_between(times[ii][inds],
                        replicate_mean_incs[ii][inds] + replicate_sem_incs[ii][inds],
                        replicate_mean_incs[ii][inds] - replicate_sem_incs[ii][inds],
                        alpha=0.3, color=cmap_fits[ii+2])

        # plot the inset
        if xlabel == 'days':
            if ii == 0:
                axins = ax.inset_axes([0.525, 0.275, .425, .425])

            axins.plot(times[ii], replicate_mean_incs[ii], color=cmap_fits[ii+2], 
                       lw=1.25)
            axins.fill_between(times[ii], 
                               replicate_mean_incs[ii] + replicate_sem_incs[ii],
                               replicate_mean_incs[ii] - replicate_sem_incs[ii], 
                               alpha=0.3, color=cmap_fits[ii+2])

    plt.legend(ncol=3)
    plt.xlabel(xlabel)
    plt.ylabel(r"$\langle \Delta F_b\rangle$")

    if xlabel == 'days':
        ax.set_xscale('log')
        axins.set_xlim([5e7, 5e8])
        axins.set_ylim([0.0, 0.00075])
        axins.set_xscale('log')
        axins.set_xticklabels([], [], fontsize=0, fontweight=0)
        axins.set_yticklabels([], [], fontsize=0, fontweight=0)
        axins.tick_params(which='both', labelbottom=False, labelleft=False)
        ax.indicate_inset_zoom(axins, edgecolor='black')

    plt.tight_layout()
    if savefig:
        plt.savefig("%s/epi_comp_increment_%s.pdf" % (output_folder, xlabel),
                    dpi=300, transparent=True, bbox_inches='tight')


def plot_rank(times: list,
              replicate_mean_ranks: list,
              replicate_sem_ranks: list,
              xlabel: str):
    """ Plot the rank as a function of time. """
    fig, ax = plt.subplots()
    for ii, mu_val in enumerate(mu_vals):
        # plot the main figure
        inds = replicate_mean_ranks[ii] >= 0
        power = int(np.log(mu_val) / np.log(10) - 0.5)
        label_str = r"$\mu=10^{%d}$" % power
        plt.plot(times[ii][inds], replicate_mean_ranks[ii][inds],
                 color=cmap_fits[ii+2], label=label_str, lw=3.0)
        ax.fill_between(times[ii][inds],
                        replicate_mean_ranks[ii][inds] + replicate_sem_ranks[ii][inds],
                        replicate_mean_ranks[ii][inds] - replicate_sem_ranks[ii][inds],
                        alpha=0.3, color=cmap_fits[ii+2])

        # plot the inset
        if xlabel == 'days':
            if ii == 0:
                axins = ax.inset_axes([0.6, 0.35, .35, .2])
            axins.plot(times[ii], replicate_mean_ranks[ii], color=cmap_fits[ii+2], 
                       label=r"$\beta=%0.2f$" % mu_val, lw=1.25)
            axins.fill_between(times[ii], 
                               replicate_mean_ranks[ii] + replicate_sem_ranks[ii],
                               replicate_mean_ranks[ii] - replicate_sem_ranks[ii], 
                               alpha=0.3, color=cmap_fits[ii+2])

    plt.legend(ncol=2, framealpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('rank')

    if xlabel == 'days':
        ax.set_xscale('log')
        axins.set_xscale('log')
        axins.set_xlim([5e7, 1e8])
        axins.set_ylim([1, 5])
        axins.set_xticklabels('', fontsize=0, fontweight=0)
        axins.set_yticklabels('', fontsize=0, fontweight=0)
        axins.tick_params(which='both', labelbottom=False, labelleft=False)
        ax.indicate_inset_zoom(axins, edgecolor='black')

    plt.tight_layout()

    if savefig:
        plt.savefig("%s/epi_comp_rank_%s.pdf" % (output_folder, xlabel), dpi=300,
                    transparent=True, bbox_inches='tight')


def plot_effective(times: list,
                   fits: list,
                   replicate_mean_incs: list,
                   replicate_mean_ranks: list,
                   L: int):
    """ Visualize the expected beneficial increment as a function of time. """
    ## First construct the effective figure
    ## and compute the predicted fitness values
    eff_fig, ax = plt.subplots()
    effectives = np.zeros_like(replicate_mean_incs)
    predicted_fitnesses = np.zeros_like(replicate_mean_incs)
    for ii, mu_val in enumerate(mu_vals):
        # make effective plot
        effectives[ii, :] = (replicate_mean_incs[ii] / fits[ii]) \
            * replicate_mean_incs[ii] * replicate_mean_ranks[ii] / L
        effective = effectives[ii, :]
        plt.plot(times[ii], effective/effective[0],
                 color=cmap_fits[ii+2], label=r"$\beta=%0.2f$" % mu_val, lw=2.5)

        # make fitness and approximate fitness plot
        interp_effective = interp1d(times[ii], effective, kind='cubic')

        # right-hand side of the effective fitness dynamics
        def rhs(y, t):
            # linear extrapolation in case the ODE solver goes outside the range
            if t > times[ii][-1]:
                return effective[-1] + \
                    (effective[-1] - effective[-2])*(t - times[ii][-1]) \
                    / (times[ii][-1] - times[ii][-2])
            else:
                return interp_effective(t)

        # compute and scale the approximate fitness, as well as the true fitness.
        F_approx = odeint(rhs, 1, times[ii]).ravel()
        F_approx /= F_approx[-1] - F_approx[0]
        F_approx += 1 - F_approx[0]
        predicted_fitnesses[ii, :] = F_approx
        fits[ii] /= fits[ii][-1] - fits[ii][0]
        fits[ii] += 1-fits[ii][0]

    # effective figure parameters
    plt.xlabel('days')
    plt.ylabel(r"$\langle\dot{F}(t)\rangle/\langle\dot{F}(0)\rangle$")
    plt.ylim([1e-8, 1.0])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.tight_layout()

    ## now construct the predicted fitness figure
    fit_fig, ax = plt.subplots()
    for ii, mu_val in enumerate(mu_vals):
        plt.plot(times[ii], fits[ii], color=cmap_fits[ii+2],
                    label=r"$\beta=%0.2f$" % mu_val, lw=2.5)
        plt.plot(times[ii], predicted_fitnesses[ii], color=cmap_fits[ii+2],
                    lw=1.5, linestyle='-.')

    # fitness figure parameters
    ax.set_xlabel('days')
    ax.set_ylabel("fitness")
    ax.set_xscale('log')
    plt.tight_layout()
    plt.legend(loc='best')

    if savefig:
        eff_fig.savefig("%s/epi_comp_effective.pdf" % output_folder,
                        dpi=300, transparent=True, bbox_inches='tight')
        fit_fig.savefig("%s/fit_approx_effective.pdf" % output_folder,
                        dpi=300, transparent=True, bbox_inches='tight')

    plt.show()


def plot_delta_rank(fits: list,
                    replicate_mean_ranks: list,
                    replicate_sem_ranks: list,
                    replicate_mean_incs: list,
                    replicate_sem_incs: list) -> None:
    """ Plot the effective change in rank per mutation event given by dR/dF * dF. """
    fig, ax = plt.subplots()
    fit_grid = np.linspace(1.75, 1.99, 1000)
    dRs = np.zeros((len(mu_vals), 3))
    for ii, mu_val in enumerate(mu_vals):
        curr_ranks = replicate_mean_ranks[ii]
        inds = curr_ranks > 0
        curr_ranks = curr_ranks[inds]
        curr_fits = fits[ii][inds]
        curr_sem_ranks = replicate_sem_ranks[ii][inds]
        curr_incs, curr_sem_incs = replicate_mean_incs[ii][inds], replicate_sem_incs[ii][inds]

        # near the fitness peak, the fitness can become extremely mildly non-monotonic.
        # drop the non-monotonicity to improve the derivative calculation
        monotonic_inds = [0]
        curr_max = curr_fits[0]
        for kk in range(1, curr_fits.size):
            curr_val = curr_fits[kk]
            if curr_val > curr_max:
                monotonic_inds.append(kk)
            curr_max = max(curr_val, curr_max)

        monotonic_inds = np.array(monotonic_inds)
        monotonic_fits = curr_fits[monotonic_inds]
        monotonic_ranks, monotonic_rank_sems = \
                curr_ranks[monotonic_inds], curr_sem_ranks[monotonic_inds]
        monotonic_incs, monotonic_inc_sems = \
                curr_incs[monotonic_inds], curr_sem_incs[monotonic_inds]

        interped_ranks = UnivariateSpline(x=monotonic_fits, y=monotonic_ranks,
                                          w=1.0/monotonic_rank_sems, k=3)
        interped_incs = UnivariateSpline(x=monotonic_fits, y=monotonic_incs,
                                          w=1.0/monotonic_inc_sems, k=3)

        dR_dF = interped_ranks.derivative()
        dR = dR_dF(fit_grid) * interped_incs(fit_grid)
        dRs[ii, 0] = np.quantile(dR, q=0.20)
        dRs[ii, 1] = np.median(dR)
        dRs[ii, 2] = np.quantile(dR, q=0.80)

    plt.plot(mu_vals, dRs[:, 1], color=cmap_fits[ii+2], lw=3)
    ax.fill_between(mu_vals, dRs[:, 0], dRs[:, 2], color=cmap_fits[ii+2], alpha=0.35)

    plt.xlabel('beta')
    plt.ylabel('expected rank decrease')
    plt.tight_layout()

    if savefig:
       plt.savefig("%s/rank_decrease.pdf" % output_folder,
                   dpi=300, transparent=True, bbox_inches='tight')
       np.save("data_proc/paper_figures/paper_figure_data/sswm_expected_rank.npy", dRs)


def make_epistasis_plot(data_folders: list,
                        nexps: int,
                        inner_folder: str,
                        bac_str: str,
                        mut_str: str,
                        mu_vals: np.ndarray,
                        L: int,
                        base_folder: str,
                        start_index: int,
                        plot_skip: int,
                        rescale: bool,
                        load_dat: bool,
                        savefig: bool) -> None:
    """ Plot the mean fitness increment and the mean rank over time
    while varying the epistasis parameter \beta. """
    # obtain the data we need for plotting
    times, fits, replicate_mean_incs, replicate_mean_ranks, \
        replicate_sem_incs, replicate_sem_ranks = get_data(load_dat,
                                                           data_folders,
                                                           inner_folder,
                                                           bac_str,
                                                           mut_str,
                                                           nexps)

    if rescale:
        for ii, fit in enumerate(fits):
            scale = fit[-1] - fit[0]
            fits[ii] /= scale
            shift = 1 - fits[ii][0]
            fits[ii] += shift

    # downsample the data to avoid plotting too much
    times, fits, replicate_mean_incs, replicate_sem_incs, \
            replicate_mean_ranks, replicate_sem_ranks = \
            downsample(times, fits, replicate_mean_incs, replicate_sem_incs,
                       replicate_mean_ranks, replicate_sem_ranks, start_index, plot_skip)

    # make the plots
    plot_increment(fits, replicate_mean_incs, replicate_sem_incs, xlabel='fitness')
    # plot_increment(times, replicate_mean_incs, replicate_sem_incs, xlabel='days')

    plot_rank(fits, replicate_mean_ranks, replicate_sem_ranks, xlabel='fitness')
    # plot_rank(times, replicate_mean_ranks, replicate_sem_ranks, xlabel='days')

    # plot_delta_rank(fits, replicate_mean_ranks, replicate_sem_ranks,
                    # replicate_mean_incs, replicate_sem_incs)

    # plot_effective(times, fits, replicate_mean_incs, replicate_mean_ranks, L)


if __name__ == '__main__':
    nexps          = 20
    bac_str        = 'bac_data*'
    mut_str        = 'mut_data*'
    skip           = 1
    L              = 1000
    base_folder    = '/home2/nick/research/lenski/mu_scan_3_20_22_epi_same_landscape_noreset'
    data_folders   = sorted(glob.glob('%s/L1e3*' % base_folder))[::skip]
    mu_vals        = np.sort(
        np.array([float(folder.split('mu')[-1]) \
                  for folder in data_folders])
    )
    print(mu_vals)
    inner_folder   = 'replicate'
    output_folder  = '%s/figs' % base_folder
    L              = int(1e3)
    savefig        = True
    load_dat       = True
    plot_skip      = 1
    start_index    = 0
    rescale        = True

    cmap_fits = cubehelix_palette(n_colors=len(mu_vals)+2, start=.5,
                                  rot=-.75, light=.85, dark=.1, hue=1,
                                  gamma=.95)

    make_epistasis_plot(data_folders, nexps, inner_folder, bac_str, mut_str,
                        mu_vals, L, base_folder, start_index, plot_skip, 
                        rescale, load_dat, savefig)

    plt.show()
