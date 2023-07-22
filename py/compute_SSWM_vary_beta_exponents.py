"""
Nicholas M. Boffi

This file contains code to plot the exponents of fitness relaxation as a function of \beta.
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
from scipy.stats.distributions import t
from seaborn import cubehelix_palette
from math import isclose
import glob
from plot_fit import asymp_pow_traj, asymp_pow_jac


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
mpl.rcParams['figure.figsize'] = (5, 5)
mpl.rcParams['figure.titlesize'] = 30
mpl.rcParams['font.size'] = 27.5
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.dpi'] = 125


def make_epistasis_exponent_plot(data_folders: list, 
                                 nexps: int, 
                                 L: int,
                                 inner_folder: str, 
                                 bac_str: str,
                                 vary_str: str,
                                 variable_str: str,
                                 vary_vals: np.ndarray, 
                                 n_bootstraps: int,
                                 output_folder: str, 
                                 log_scale: bool,
                                 savefig: bool, 
                                 cmap: list,
                                 load_dat: bool, 
                                 plot_fitness: bool) -> None:
    """ Plots a histogram of exponents, final values, and slopes
    as a function of \beta or \mu. """
    increments = np.zeros((nexps, vary_vals.size))
    finals = np.zeros((nexps, vary_vals.size))
    a_vals = np.zeros((n_bootstraps, vary_vals.size))
    b_vals = np.zeros((n_bootstraps, vary_vals.size))
    exponents = np.zeros((n_bootstraps, vary_vals.size))

    # for each value of beta, compute the initial increment, 
    # final value, and fit the exponent.
    for ii, vary_val_folder in enumerate(data_folders):
        # load in the mean fitness over replicates.
        if plot_fitness:
            if not load_dat:
                # load in the bacteria files and set up the times array
                bac_files = get_files([vary_val_folder], inner_folder,
                                      bac_str, nexps)
                ntimes = max([len(bac_files_list) for bac_files_list\
                              in bac_files])
                dt = float(bac_files[0][1].split('.')[-2])
                times = np.arange(ntimes)*dt

                # load in the fitness over replicates
                traj_data, max_fits, dominant_fits = get_fit_data(nexps, 1,
                                                                  ntimes,
                                                                  bac_files)

                # initial fitness increment and value that we converge to.
                increments[:, ii] = traj_data[:, 1] - traj_data[:, 0]
                finals[:, ii] = traj_data[:, -1]

                # save the data
                data_dict = {}
                data_dict['mean_fits'] = traj_data
                data_dict['times'] = times
                data_dict['max_fits'] = max_fits
                data_dict['dominant_fits'] = dominant_fits
                pickle.dump(data_dict, open("%s/figs/fit_data.pickle" % \
                                            vary_val_folder, "wb"))

            else:
                data_dict = pickle.load(open("%s/figs/fit_data.pickle" % \
                                             vary_val_folder, "rb"))
                traj_data = data_dict['mean_fits']
                times = data_dict['times']
                finals[:, ii] = traj_data[:nexps, -1]
                increments[:, ii] = traj_data[:nexps, 1] - traj_data[:nexps, 0]
        else:
            # only load substitution trajectory for now.
            data_dict = pickle.load(open("%s/figs/subst_data.pickle" % \
                                         vary_val_folder, "rb"))
            traj_data = data_dict['subst_traj']
            finals[:, ii] = traj_data[:, -1]

        # bootstrap estimation
        all_data = traj_data.ravel()
        for curr_estimate in range(n_bootstraps):
            # bootstrap over means
            sample = np.random.randint(nexps, size=nexps)
            traj_est_sem = sem(traj_data[sample, :], axis=0)
            sigma = traj_est_sem[traj_est_sem > 0]
            traj_est_data = \
                np.mean(traj_data[sample, :], axis=0)[traj_est_sem > 0]
            traj_est_times = times[traj_est_sem > 0]
            b_estimate = np.mean(traj_data[sample, -1])
            asymp_pow_params, asymp_pow_cov = curve_fit(asymp_pow_traj,
                                                        traj_est_times,
                                                        traj_est_data,
                                                        sigma=sigma,
                                                        jac=asymp_pow_jac,
                                                        p0=[1e-6, 
                                                            b_estimate,
                                                            2.0],
                                                        absolute_sigma=True,
                                                        bounds=([0.0,
                                                                 .95*b_estimate,
                                                                 0.0],
                                                                [1.0,
                                                                 1.05*b_estimate,
                                                                 np.inf]),
                                                        tr_solver='exact',
                                                        x_scale='jac',
                                                        maxfev=25000,
                                                        method='dogbox',
                                                        loss='linear',
                                                        ftol=2.5e-12,
                                                        xtol=2.5e-12,
                                                        gtol=2.5e-12)

            print("Finished fitting on bootstrap %d/%d for %s=%e, \
                  value %d/%d. Exponent: %g" % \
                  (curr_estimate+1, n_bootstraps, vary_str, vary_vals[ii], 
                      ii+1, len(data_folders), asymp_pow_params[2]))
            a_vals[curr_estimate, ii]    = asymp_pow_params[0]
            b_vals[curr_estimate, ii]    = asymp_pow_params[1]
            exponents[curr_estimate, ii] = asymp_pow_params[2]

    # plot the exponents
    fig, ax = plt.subplots()
    medians = np.median(exponents, axis=0)
    lbs = np.quantile(exponents, axis=0, q=0.025)
    ubs = np.quantile(exponents, axis=0, q=0.975)
    plt.plot(vary_vals, medians, color=cmap[1], lw=4.0, marker='o', ms=10.0)
    ax.fill_between(vary_vals, lbs, ubs, alpha=0.3, color=cmap[1])

    if log_scale:
        ax.set_xscale('log')

    # add relevant axis labels and save the figure
    # hard-code to avoid weird latex bug
    if vary_str == 'beta':
        plt.xlabel(r"$\beta$")
    else:
        plt.xlabel(r"$%s$" % variable_str)
    plt.ylabel('exponent')
    plt.tight_layout()
    if savefig:
        plt.savefig('%s/%s_exponents_vary_%s_bootstrap.pdf' % \
                    (output_folder, 'fitness' if plot_fitness else 'subst',
                     vary_str),
                    dpi=300, bbox_inches='tight')


    save_exponent_info(output_folder, plot_fitness, exponents, 
                       vary_vals, vary_str)


def save_exponent_info(output_folder: str,
                       plot_fitness: bool,
                       exponents: np.ndarray,
                       vary_vals: np.ndarray,
                       vary_str: str) -> None:
    """ Construct and output a dictionary with exponent information. """
    exponent_data_dict = {}
    exponent_data_dict['exponents'] = exponents
    exponent_data_dict['%s_vals' % vary_str] = vary_vals
    data_str = 'fitness' if plot_fitness else 'subst'

    pickle.dump(
            exponent_data_dict, 
            open("%s/exponent_data_%s_vary_%s.pickle" \
                            % (output_folder, data_str, vary_str), "wb")
                )


if __name__ == '__main__':
    nexps          = 25
    bac_str        = 'bac_data*'
    skip           = 1
    base_folder    = 'sswm_extended_more_reps_diff_init_rerun_p75_1p0'
    data_folders   = sorted(glob.glob('%s/L1e3*' % base_folder))[::skip]
    break_str      = 'b'
    vary_vals      = np.sort(
        np.array(
            [float(folder.split('L')[-1].split(break_str)[-1]) \
             for folder in data_folders])
    )
    print(f'Vary vals: {vary_vals}')
    inner_folder   = 'replicate'
    output_folder  = '%s/figs' % base_folder
    L              = int(1e3)
    savefig        = False
    vary_str       = 'beta'
    variable_str   = '\beta'
    plot_fitness   = True
    data_str       = 'fitness' if plot_fitness else 'subst'
    save_title     = '%s_exponents_vary_%s_bootstrap' % (data_str, vary_str)
    load_dat       = True
    log_scale      = False
    n_bootstraps   = 200

    ## colorschemes
    cmap = cubehelix_palette(n_colors=3, start=.5, rot=-.75, light=.85,
                             dark=.1, hue=1, gamma=.95)
    cmap_replicates = cubehelix_palette(n_colors=nexps, start=.5, rot=-.75,
                                        light=.85, dark=.1, hue=1, gamma=.95)

    make_epistasis_exponent_plot(data_folders, nexps, L, inner_folder,
                                 bac_str, vary_str, variable_str, vary_vals, 
                                 n_bootstraps, output_folder, log_scale,
                                 savefig, cmap, load_dat, plot_fitness)
    plt.show()
