"""
Nicholas M. Boffi

This file contains code to visualize the substitution trajectory
as a function of the epistasis parameter \beta in the clonal interference regime.
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


def make_epistasis_plot(base_folder: str,
                        data_folders: list,
                        nexps: int,
                        inner_folder: str,
                        beta_vals: list,
                        plot_skip: int,
                        save_title: str,
                        extend_trajs: bool,
                        savefig: bool) -> None:
    """ Plot the mean substitution trajectory over time and vary
    the epistasis parameter \beta. """
    # load in the substitution trajectory data for each experiment.
    for ii, data_folder in enumerate(data_folders):
        pickle_str = '%s/figs/subst_data.pickle' % data_folder
        data_dict = pickle.load(open(pickle_str, "rb"))
        subst_traj = data_dict['subst_traj']

        # allocate arrays if we haven't yet
        if (ii == 0):
            replicate_mean_subst_array = np.zeros((nexps, subst_traj.shape[1]))
            replicate_sem_subst_array = np.zeros((nexps, subst_traj.shape[1]))
            times = np.zeros((nexps, subst_traj.shape[1]))

        replicate_mean_subst_array[ii, :] = np.mean(subst_traj, axis=0)
        replicate_sem_subst_array[ii, :] = sem(subst_traj, axis=0)
        times[ii, :] = data_dict['times']

    # extend the trajectories to the length of the longest (assuming that
    # all have been run sufficiently long to converge)
    if extend_trajs:
        max_t = np.max(times)
        npts_extend = 50
        ntimes = times.shape[1]
        extended_times = np.zeros((beta_vals.size, ntimes + npts_extend))
        extended_means = np.zeros_like(extended_times)
        extended_sems = np.zeros_like(extended_times)
        for beta_ind, _ in enumerate(beta_vals):
            curr_max_t = times[beta_ind, -1]
            extended_times[beta_ind, :ntimes] = times[beta_ind]
            extended_times[beta_ind, ntimes:] = \
                    np.linspace(curr_max_t, max_t, npts_extend)
            extended_means[beta_ind, :ntimes] = \
                    replicate_mean_subst_array[beta_ind]
            extended_means[beta_ind, ntimes:] = \
                    np.repeat(replicate_mean_subst_array[beta_ind, -1], npts_extend)
            extended_sems[beta_ind, :ntimes] = \
                    replicate_sem_subst_array[beta_ind]
            extended_sems[beta_ind, ntimes:] = \
                    np.repeat(replicate_sem_subst_array[beta_ind, -1], npts_extend)

        times = extended_times
        replicate_mean_subst_array = extended_means
        replicate_sem_subst_array = extended_sems

    # plot the replicate mean with error bars.
    fig, ax = plt.subplots()
    for ii, beta_val in enumerate(beta_vals):
        plt.plot(times[ii, ::plot_skip],
                 replicate_mean_subst_array[ii, ::plot_skip],
                 color=cmap[ii+2],
                 label=r"$\beta=%0.2f$" % beta_val, lw=4)
        ax.fill_between(times[ii, ::plot_skip],
                        replicate_mean_subst_array[ii, ::plot_skip] \
                                + replicate_sem_subst_array[ii, ::plot_skip],
                        replicate_mean_subst_array[ii, ::plot_skip] \
                                - replicate_sem_subst_array[ii, ::plot_skip],
                        alpha=0.3, color=cmap[ii+2])
    plt.legend(ncol=3)
    plt.xlabel('day')
    plt.ylabel('number of fixed mutations')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    if log_plot:
        ax.set_xscale('log')

    plt.tight_layout()

    if savefig:
        plt.savefig("%s/%s%s.pdf" \
                % (output_folder, save_title, '_log' if log_plot else ''),
                    dpi=300, transparent=True)


if __name__ == '__main__':
    inner_folder   = 'replicate'
    nexps          = 20
    skip           = 1
    base_folder    = '/scratch3/nick/lenski/clonal_extended_more_reps_diff_init'
    data_folders   = sorted(glob.glob('%s/L1e3*' % base_folder))[::skip]
    beta_vals      = np.sort(
            np.array(
                [float(folder.split('L')[-1].split('b')[-1]) \
                        for folder in data_folders]
                )
            )
    inner_folder   = 'replicate'
    output_folder  = '%s/figs' % base_folder
    L              = int(1e3)
    savefig        = True
    log_plot       = False
    plot_skip      = 25
    save_title     = 'epi_subst_comp'
    extend_trajs   = True

    cmap = cubehelix_palette(n_colors=len(beta_vals)+2, start=.5, rot=-.75,
                             light=.85, dark=.1, hue=1, gamma=.95)
    make_epistasis_plot(base_folder, data_folders, nexps, inner_folder,
                        beta_vals, plot_skip, save_title, extend_trajs,
                        savefig)

    plt.show()
