"""
Nicholas M. Boffi
2/6/22

This file contains code to produce Figure 1B.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import typing
from typing import Tuple
from common import *
import pickle
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import sem
from seaborn import cubehelix_palette


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
mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['figure.titlesize'] = 30
mpl.rcParams['font.size'] = 32
mpl.rcParams['legend.fontsize'] = 25
mpl.rcParams['figure.dpi'] = 125


def asymp_pow_traj(t, a, b, c, d=1):
    """ 1/t^2 relaxation expected for the no-epistatic case. """
    return b + (d-b)/(a*t + 1)**c


def asymp_pow_jac(t, a, b, c, d=1):
    """ Jacobian for the above. """
    return np.array([(b - d)*c*t*(1 + a*t)**(-1-c),
                     1 - (1 + a*t)**(-c),
                     (b - d)*(1 + a*t)**(-c)*np.log(1 + a*t)]).T


def make_fitness_plot(
        times: np.array,
        total_n_mutants: np.ndarray,
        fit_data: np.array,
        asymptotic_vals: np.ndarray,
        skip: int,
        tf_index: int,
        nexps: int,
        n_bootstraps: int,
        log_plot: bool,
        log_lin_plot: bool,
        lin_plot: bool,
        fig_title: str,
        save_title: str) -> None:
    """ Make a single plot of fit_data over time. """
    # clip the data
    times = times[:tf_index]
    fit_data = fit_data[:, :tf_index]
    total_n_mutants = total_n_mutants[:, :tf_index]

    # plot the individual replicate trajectories in the background.
    fig, ax = plt.subplots()
    for replicate in range(nexps):
        plt.plot(times, fit_data[replicate, :], color=cmap[replicate],
                 lw=3.0, alpha=.35)

    # plot the replicate mean with error bars.
    replicate_mean = np.mean(fit_data, axis=0)
    replicate_sem = sem(fit_data, axis=0, ddof=1)
    plt.errorbar(times, replicate_mean,  yerr=replicate_sem, ls='',
                 marker='o', markerfacecolor='none', markersize=7.5,
                 color='k', alpha=1.0, errorevery=skip, markevery=skip)


    # bootstrap estimation
    a_vals    = np.zeros(n_bootstraps)
    b_vals    = np.zeros(n_bootstraps)
    exponents = np.zeros(n_bootstraps)
    all_data = fit_data.ravel()
    all_times = total_n_mutants.ravel()
    for curr_estimate in range(n_bootstraps):
        sample = np.random.randint(nexps, size=nexps)
        traj_est_sem = sem(fit_data[sample, :], axis=0)
        sigma = traj_est_sem[traj_est_sem > 0]
        traj_est_data = np.mean(fit_data[sample, :], axis=0)[traj_est_sem > 0]
        traj_est_times = times[traj_est_sem > 0]
        b_estimate = np.mean(fit_data[sample, -1])

        asymp_pow_params, asymp_pow_cov \
                = curve_fit(asymp_pow_traj,
                            traj_est_times,
                            traj_est_data,
                            sigma=sigma,
                            jac=asymp_pow_jac,
                            p0=[1e-7, b_estimate, 2.0],
                            absolute_sigma=True,
                            bounds=([0.0, .95*b_estimate, 0.0], \
                                    [1.0, 1.05*b_estimate, 4.0]),
                            tr_solver='exact',
                            x_scale='jac',
                            maxfev=25000,
                            method='dogbox',
                            loss='linear',
                            ftol=2.5e-14,
                            xtol=2.5e-14,
                            gtol=2.5e-14)

        print(f'Parameters for bootstrap {curr_estimate+1}/{n_bootstraps}:{asymp_pow_params}')
        a_vals[curr_estimate] = asymp_pow_params[0]
        b_vals[curr_estimate] = asymp_pow_params[1]
        exponents[curr_estimate] = asymp_pow_params[2]

    median = np.median(exponents)
    lb = np.quantile(exponents, 0.025)
    ub = np.quantile(exponents, 0.975)
    print(f'Bootstrap interval: {lb} / {median} / {ub}.')

    # evaluate and plot the fit
    asymp_a, asymp_b, asymp_c \
            = np.median(a_vals), np.median(b_vals), np.median(exponents)
    asymp_pow_fit = lambda t: asymp_pow_traj(t, asymp_a, asymp_b, asymp_c)
    asymp_pow_dat = asymp_pow_fit(times)
    asymp_pow_rsq = compute_rsq(replicate_mean, asymp_pow_dat)
    plt.plot(times, asymp_pow_dat,
             color=cmap[-2], lw=3, alpha=1.0)

    # axis labels
    plt.xlabel("days")
    plt.ylabel("fitness")
    plt.title(fig_title)
    plt.legend(loc='best', framealpha=0, ncol=1)

    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.tight_layout()

    axins = ax.inset_axes([0.35, 0.2, .45, .3])
    axins.plot(np.log(1 + asymp_a*times),
               np.log(asymp_b - replicate_mean),
               color=cmap[-6], ls='--')
    axins.plot(np.log(1 + asymp_a*times),
               np.log(asymp_b - 1) - asymp_c*np.log(1 + asymp_a*times),
               color=cmap[-6], linestyle='-')
    axins.set_xlabel(r"$\log(1 + at)$", fontsize=25)
    axins.set_ylabel(r"$\log(F_{\infty} - F(t))$", fontsize=25)
    axins.tick_params(axis='both', which='both', labelsize=25)

    if savefig:
        plt.savefig("%s/%s.pdf" \
                % (output_folder, save_title), dpi=300, transparent=True)


    # exponent histogram of the bootstrap
    plt.figure()
    plt.hist(exponents)
    plt.tight_layout()


def make_plots(
        bac_files: list,
        nexps: int,
        n_top_strains: int,
        savefig: bool,
        skip: int,
        data_folder: str,
        inner_folder: str,
        tf_index: int,
        n_bootstraps: int,
        log_plot: bool,
        load_dat: bool) -> None:
    """ Plot the mean fitness and dominant fitness over time of
    the dominant strains. Also fit the replicate-avergaed mean fitness
    and dominant fitnesses to specific functional forms.
    """
    # gather the data.
    ntimes = max([len(bac_files_list) for bac_files_list in bac_files])
    if not load_dat:
        mean_fits, max_fits, dominant_fits \
                = get_fit_data(nexps, n_top_strains, ntimes, bac_files)
        total_n_mutants = get_n_mutants(data_folder, inner_folder,
                                        ntimes, nexps)
        # extract the time of the first output day.
        dt = float(bac_files[0][1].split('.')[-2])
        times = np.arange(ntimes)*dt

        data_dict = {}
        data_dict['mean_fits'] = mean_fits
        data_dict['times'] = times
        data_dict['max_fits'] = max_fits
        data_dict['dominant_fits'] = dominant_fits
        data_dict['total_n_mutants'] = total_n_mutants
        pickle.dump(data_dict,
                    open("%s/figs/fit_data.pickle" % data_folder, "wb"))
    else:
        data_dict = pickle.load(
                open("%s/figs/fit_data.pickle" % data_folder, "rb")
                )
        mean_fits = data_dict['mean_fits']
        max_fits = data_dict['max_fits']
        dominant_fits = data_dict['dominant_fits']
        total_n_mutants = data_dict['total_n_mutants']
        times = data_dict['times']

    if rescale:
        scale = mean_fits[:, -1] - mean_fits[:, 0]
        mean_fits[:, :] /= scale[:, None]
        shift = 1 - mean_fits[:, 0]
        mean_fits[:, :] += shift[:, None]


    print("Making fitness plot for the mean fitness.")
    make_fitness_plot(times, total_n_mutants, mean_fits, mean_fits[:, -1],
                      skip, tf_index, nexps, n_bootstraps,
                      log_plot, log_lin_plot, lin_plot, '', "mean_fits")


if __name__ == '__main__':
    data_folders \
            = ['/scratch3/nick/lenski/clonal_extended_more_reps_diff_init/L1e3_r1e2_m2e-4_b0.500000']
    output_folder = '%s/figs' % data_folders[0]
    inner_folder = 'replicate'
    nexps = 20
    bac_str = 'bac_data*'
    n_top_strains = 1
    L = int(1e3)
    tf_index = -1
    savefig  = True
    load_dat = True
    rescale = False
    n_bootstraps = 200
    skip = 90

    cmap = cubehelix_palette(n_colors=nexps, start=.5, rot=-.75, light=.85,
                             dark=.1, hue=1, gamma=.95)
    bac_files = get_files(data_folders, inner_folder, bac_str, nexps)
    make_plots(bac_files, nexps*len(data_folders), n_top_strains, savefig,
               skip, data_folders[0], inner_folder, tf_index, n_bootstraps,
               log_plot, load_dat)

    plt.show()
