"""
Nicholas M. Boffi

This file contains code to visualize the fitness trajectory as a function
of the mutation rate in the presence of epistasis.
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


def downsample(arr: np.ndarray,
               init_skip: int,
               plot_skip: int) -> np.ndarray:
    """ Downsample an array by plot_skip while preserving the first init_skip
    entries so that plotting on a logarithmic scale still looks reasonable. """

    return np.concatenate((arr[:init_skip], arr[init_skip::plot_skip]))


def make_epistasis_plot(
    data_folders: list,
    nexps: int,
    inner_folder: str,
    bac_str: str,
    vary_vals: np.ndarray,
    vary_str: str,
    variable_str: str,
    save_title: str,
    savefig: bool,
    plot_skip: int,
    load_dat: bool,
    log_plot: bool,
    scale_trajs: bool,
    x_label: str,
    extend_trajs: bool,
    vary_scale: bool,
    slope_scale: bool,
    log_diff_plot: bool
) -> None:
    """ Plot the mean fitness and dominant fitness over time and
    vary the epistasis parameter \beta. """

    if not load_dat:
        # load in the fitness data for each experiment.
        for vary_ind, data_folder in enumerate(data_folders):
            bac_files = get_files([data_folder], inner_folder, bac_str, nexps)
            ntimes = max([len(bac_files_list) for bac_files_list in bac_files])

            # allocate arrays if we haven't yet
            if (vary_ind == 0):
                replicate_mean_fits_array = np.zeros((vary_vals.size, ntimes))
                replicate_sem_fits_array = np.zeros((vary_vals.size, ntimes))
                times = np.zeros((vary_vals.size, ntimes))

            # set up time
            dt = float(bac_files[0][1].split('.')[-2])
            times[vary_ind, :] = np.arange(ntimes)*dt

            # load in the mean fitness
            mean_fits, _, _ = get_fit_data(nexps, 1, ntimes, bac_files)

            # print diagnostics
            print('Finished loading fitness on epistasis value %d/%d' \
                  % (vary_ind+1, len(data_folders)))

            replicate_mean_fits_array[vary_ind, :] = np.mean(mean_fits, axis=0)
            replicate_sem_fits_array[vary_ind, :]  = sem(mean_fits, axis=0)

        data_dict = {}
        data_dict['replicate_mean_fits_array'] = replicate_mean_fits_array
        data_dict['replicate_sem_fits_array'] = replicate_sem_fits_array
        data_dict['times'] = times


        pickle.dump(data_dict, open("%s/%s_comp_fit_data.pickle"
                                    % (output_folder, vary_str), "wb"))

    else:
        data_dict = pickle.load(open("%s/%s_comp_fit_data.pickle" \
                                     % (output_folder, vary_str), "rb"))
        replicate_mean_fits_array = data_dict['replicate_mean_fits_array']
        replicate_sem_fits_array = data_dict['replicate_sem_fits_array']
        times = data_dict['times']
        print(replicate_mean_fits_array[:, -1])


    # plot in re-normalized time
    if vary_str == 'mu':
        # normalize by the mutation rate
        if vary_scale:
            times *= vary_vals[:, None]
        elif slope_scale:
            # normalize by the initial time derivative / increment
            initial_increments = replicate_mean_fits_array[:, 1] \
                    - replicate_mean_fits_array[:, 0]
            dts = times[:, 1] - times[:, 0]
            init_derivs = initial_increments / dts
            times *= init_derivs[:, None]

            after_derivs = initial_increments / (times[:, 1] - times[:, 0])
            print(f'init_derivs: {init_derivs}')
            print(f'after_derivs: {after_derivs}')


    # scale the trajectories for a qualitative comparison
    if scale_trajs:
        for vary_ind in np.arange(vary_vals.size):
            scale = replicate_mean_fits_array[vary_ind, -1] \
                - replicate_mean_fits_array[vary_ind, 0]
            replicate_mean_fits_array[vary_ind, :] /= scale
            replicate_sem_fits_array[vary_ind, :] /= scale
            shift = 1 - replicate_mean_fits_array[vary_ind, 0]
            replicate_mean_fits_array[vary_ind, :] += shift

    # extend the trajectories to the length of the longest (assuming that
    # all have been run sufficiently long to converge)
    if extend_trajs:
        max_t = max([time_arr[-1] for time_arr in times])
        print('max t: %g' % max_t)
        npts_extend = 50
        ntimes = times[0].size
        extended_times = np.zeros((vary_vals.size, ntimes + npts_extend))
        extended_means = np.zeros_like(extended_times)
        extended_sems = np.zeros_like(extended_times)
        for vary_ind, _ in enumerate(vary_vals):
            curr_max_t = times[vary_ind, -1]
            extended_times[vary_ind, :ntimes] = times[vary_ind]
            extended_times[vary_ind, ntimes:] = \
                    np.linspace(curr_max_t, max_t, npts_extend)
            extended_means[vary_ind, :ntimes] = \
                    replicate_mean_fits_array[vary_ind]
            extended_means[vary_ind, ntimes:] = \
                    np.repeat(replicate_mean_fits_array[vary_ind, -1], npts_extend)
            extended_sems[vary_ind, :ntimes] = \
                    replicate_sem_fits_array[vary_ind]
            extended_sems[vary_ind, ntimes:] = \
                    np.repeat(replicate_sem_fits_array[vary_ind, -1], npts_extend)

        times = extended_times
        replicate_mean_fits_array = extended_means
        replicate_sem_fits_array = extended_sems


    # plot the replicate mean with error bars.
    fig, ax = plt.subplots()

    for ii, vary_val in enumerate(vary_vals):
        if vary_str == 'mu':
            power = int(np.log(vary_val) / np.log(10) - 0.5)
            label_str = r"$%s=10^{%d}$" % (variable_str, power)
        else:
            # hard-code to avoid weird latex bug
            label_str = r"$\beta=%0.3f$" % (vary_val)

        plot_times = downsample(times[ii, :], 5*plot_skip, plot_skip)
        plot_means = downsample(replicate_mean_fits_array[ii, :], 5*plot_skip, plot_skip)
        plot_sems = downsample(replicate_sem_fits_array[ii, :], 5*plot_skip, plot_skip)


        if log_diff_plot:
            max_val = plot_means[-1]
            log_vals = -np.log(max_val - plot_means)
            print(max_val - plot_means)
            plt.plot(plot_times, log_vals, color=cmap_fits[ii+2], label=label_str, lw=3.5)
        else:
            plt.plot(plot_times, plot_means, color=cmap_fits[ii+2], label=label_str, lw=3.5)
            ax.fill_between(plot_times, plot_means + plot_sems, plot_means - plot_sems,
                            alpha=0.3, color=cmap_fits[ii+2])

    plt.legend(ncol=1, framealpha=1.0, loc='lower right')
    plt.xlabel(x_label)
    plt.ylabel('fitness')

    if log_plot:
        ax.set_xscale('log')
        plt.tight_layout()

        if savefig:
            plt.savefig("%s/%s_log.pdf" % (output_folder, save_title),
                        dpi=300, transparent=True, bbox_inches='tight')

    else:
        # ax.set_xscale('asinh')
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.tight_layout()

        if savefig:
            plt.savefig("%s/%s.pdf" % (output_folder, save_title),
                        dpi=300, transparent=True, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    nexps          = 20
    bac_str        = 'bac_data*'
    skip           = 1
    base_folder    = 'mu_scan_3_14_22_noepi_same_landscape'
    data_folders   = sorted(glob.glob('%s/L1e3*' % base_folder))[::skip]
    break_str      = 'mu'
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
    log_plot       = False
    scale_trajs    = True
    extend_trajs   = True
    vary_scale     = False
    slope_scale    = True
    log_diff_plot  = True
    vary_str       = 'mu'
    variable_str   = '\mu'
    x_label        = r"days$\times \mu$" if vary_scale \
            else r"days $\times\dot{F}(0)$" if slope_scale else r"days"
    save_title     = '%s_fit_comp%s%s%s' \
            % (vary_str, '_scale' if scale_trajs else '', 
                    '_varyscale' if vary_scale else '',
                    '_slopescale' if slope_scale else '')
    plot_skip      = 2
    load_dat       = True

    cmap_fits = cubehelix_palette(n_colors=len(vary_vals)+2, start=.5,
                                  rot=-.75, light=.85, dark=.1, hue=1,
                                  gamma=.95)
    make_epistasis_plot(data_folders, nexps, inner_folder, bac_str, vary_vals,
                        vary_str, variable_str, save_title, savefig, 
                        plot_skip, load_dat, log_plot, scale_trajs, x_label,
                        extend_trajs, vary_scale, slope_scale, log_diff_plot)
