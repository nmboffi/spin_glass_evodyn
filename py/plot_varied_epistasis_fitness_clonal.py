"""
Nicholas M. Boffi

This file contains code to visualize the fitness trajectory
as a function of the epistasis parameter \beta in the clonal
interference regime
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


def make_epistasis_plot(data_folders: list,
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
                        extend_trajs: bool) -> None:
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

            times[vary_ind, :] = np.arange(ntimes)*dt \
                    * vary_vals[vary_ind] if vary_str == 'mu' else 1.0

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
    if not log_plot:
        axins = ax.inset_axes([0.35, 0.375, .45, .4])

    for ii, vary_val in enumerate(vary_vals):
        # hard-code to avoid weird latex bug
        label_str = r"$\beta=%0.2f$" % (vary_val)

        plt.plot(times[ii, ::plot_skip],
                replicate_mean_fits_array[ii, ::plot_skip],
                 color=cmap_fits[ii+2],
                 label=label_str, lw=4)
        ax.fill_between(times[ii, ::plot_skip],
                        replicate_mean_fits_array[ii, ::plot_skip] \
                            + replicate_sem_fits_array[ii,::plot_skip],
                        replicate_mean_fits_array[ii, ::plot_skip] \
                            - replicate_sem_fits_array[ii, ::plot_skip],
                        alpha=0.3, color=cmap_fits[ii+2])

        if not log_plot:
            axins.plot(times[ii, ::plot_skip],
                       replicate_mean_fits_array[ii, ::plot_skip],
                       color=cmap_fits[ii+2], lw=2.0)
            axins.fill_between(times[ii, ::plot_skip],
                               replicate_mean_fits_array[ii, ::plot_skip] \
                                   + replicate_sem_fits_array[ii, ::plot_skip],
                               replicate_mean_fits_array[ii, ::plot_skip] \
                                   - replicate_sem_fits_array[ii, ::plot_skip],
                               alpha=0.2, color=cmap_fits[ii+2])

    plt.legend(ncol=3)
    plt.xlabel(x_label)
    plt.ylabel('fitness')

    if log_plot:
        ax.set_xscale('log')
        plt.tight_layout()

        if savefig:
            plt.savefig("%s/%s_log.pdf" % (output_folder, save_title),
                        dpi=300, transparent=True, bbox_inches='tight')

    else:
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        axins.set_ylim([1.935, 2.035])
        axins.set_xlim([5.0e5, 8e5])
        axins.set_xticklabels([], [], fontsize=0, fontweight=0)
        axins.set_yticklabels([], [], fontsize=0, fontweight=0)
        axins.tick_params(which='both', labelbottom=False, labelleft=False)
        ax.indicate_inset_zoom(axins, edgecolor='black')
        plt.tight_layout()

        if savefig:
            plt.savefig("%s/%s.pdf" % (output_folder, save_title),
                        dpi=300, transparent=True, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    nexps          = 20
    bac_str        = 'bac_data*'
    skip           = 1
    base_folder    = '/scratch3/nick/lenski/clonal_extended_more_reps_diff_init'
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
    savefig        = True
    log_plot       = False
    scale_trajs    = True
    extend_trajs   = True
    vary_str       = 'beta'
    variable_str   = "\beta"
    x_label        = r"days"
    save_title     = '%s_fit_comp%s' \
            % (vary_str, '_scale' if scale_trajs else '')
    plot_skip      = 25
    load_dat       = True

    cmap_fits = cubehelix_palette(n_colors=len(vary_vals)+2, start=.5,
                                  rot=-.75, light=.85, dark=.1, hue=1,
                                  gamma=.95)
    make_epistasis_plot(data_folders, nexps, inner_folder, bac_str, vary_vals,
                        vary_str, variable_str, save_title, savefig, 
                        plot_skip, load_dat, log_plot, scale_trajs, x_label,
                        extend_trajs)
