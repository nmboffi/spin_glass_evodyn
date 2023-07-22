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
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['figure.titlesize'] = 30
mpl.rcParams['font.size'] = 27.5
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.dpi'] = 125


def comp_subst_traj(mut_files: list, 
                    nexps: int, 
                    nt: int, 
                    L: int) -> np.ndarray:
    """ Comput the substitution trajectory over all replicates. """
    subst_traj = np.zeros((nexps, nt))
    for tt in range(nt):
        print("Finding fixed mutations on day %d/%d" % (tt+1, nt))
        fixed_muts_list, _ = get_fixed_mutations(mut_files, 
                                                 L, 
                                                 calc_true_subst=True, 
                                                 file_index=tt,
                                                 print_info=False)
        subst_traj[:, tt] = np.array(
                [len(fixed_muts_set) for fixed_muts_set in fixed_muts_list]
                )

    return subst_traj


def make_subst_plot(mut_files: list, 
                    nexps: int, 
                    L: int, 
                    load_dat: bool, 
                    output_folder: str,
                    file_str: str, 
                    savefig: bool, 
                    skip: int) -> None:
    """ Plot the substitution trajectories over time. """
    nt = len(mut_files[0])
    dt = float(mut_files[0][1].split('.')[-2])
    times = np.arange(nt)*dt
    if not load_dat:
        # compute the substitution trajectories for each replicate
        subst_traj = comp_subst_traj(mut_files, nexps, nt, L)
        data_dict = {}
        data_dict['subst_traj'] = subst_traj
        data_dict['times'] = times
        pickle.dump(data_dict, open("%s/%s" % (output_folder, file_str), "wb"))
    else:
        data_dict = pickle.load(open("%s/%s" % (output_folder, file_str), "rb"))
        subst_traj = data_dict['subst_traj']

    times = times[:int(nt)]
    subst_traj = subst_traj[:, :int(nt)]
    # plot_mechanism_over_time(nexps, subst_traj, times, cmap_replicates,
                             # cmap_means, 'day', 'number of fixed mutations', '', output_folder, 'subst_traj',
                             # savefig, skip)

    # plot the individual replicate trajectories in the background.
    fig, ax = plt.subplots()
    for replicate in range(nexps):
        plt.plot(times, 
                 subst_traj[replicate, :], 
                 color=cmap_replicates[replicate], lw=3.0, alpha=.35)

    # plot the replicate mean with error bars.
    replicate_mean = np.mean(subst_traj, axis=0)
    replicate_sem = sem(subst_traj, axis=0, ddof=1)
    plt.errorbar(times, replicate_mean,  yerr=replicate_sem, ls='', 
                 marker='o', markerfacecolor='none', markersize=5.5,
                 color=cmap_means[-2], alpha=1.0, errorevery=skip, 
                 markevery=skip)

    ## fit the mean with weighted error.
    b_estimate = np.mean(subst_traj[:, -1])
    asymp_pow_fit_traj = \
            lambda t, a, c: asymp_pow_traj(t, a, b_estimate, c, d=0)
    asymp_pow_fit_jac = \
            lambda t, a, c: asymp_pow_jac(t, a, b_estimate, c, d=0)[:, ::2]
    asymp_pow_params, asymp_pow_cov = curve_fit(asymp_pow_fit_traj, 
                                                times[replicate_sem > 0], 
                                                replicate_mean[replicate_sem > 0], 
                                                sigma=replicate_sem[replicate_sem > 0], 
                                                verbose=2,
                                                p0=[1e-8, 1.0], 
                                                bounds=([0.0, 0.0], [1.0, 7.5]), 
                                                jac=asymp_pow_fit_jac,
                                                tr_solver='exact', 
                                                maxfev=10000, 
                                                method='dogbox', 
                                                x_scale='jac',
                                                loss='huber', 
                                                ftol=2.5e-16, 
                                                xtol=2.5e-16, 
                                                gtol=2.5e-16)

    # evaluate and plot the fit
    asymp_a, asymp_c = asymp_pow_params[0], asymp_pow_params[1]
    print('asymp_pow params:', asymp_a, asymp_c)
    asymp_pow_fit = lambda t: asymp_pow_traj(t, asymp_a, b_estimate, asymp_c)
    asymp_pow_dat = asymp_pow_fit(times)
    asymp_pow_rsq = compute_rsq(replicate_mean, asymp_pow_dat)
    plt.plot(times, asymp_pow_dat, 
             label=r"$b(1-\frac{1}{(at+1)^{c}}), R^2=%0.4f, c=%0.7f$" \
                     % (asymp_pow_rsq, asymp_c),
             color=cmap_means[-2], lw=3, alpha=1.0)


    # axis labels
    plt.xlabel("day")
    plt.ylabel("number of fixed mutations")
    plt.legend(loc='best', framealpha=0.3, ncol=1)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.tight_layout()

    if savefig:
        plt.savefig("%s/subst_traj.pdf" % output_folder, dpi=300, transparent=True)


if __name__ == '__main__':
    data_folders = ['sswm_extended_more_reps_diff_init/L1e3_r1e2_m1e-8_b0.000000']
    inner_folder = 'replicate'
    output_folder = '%s/figs' % data_folders[0]
    nexps = 25
    bac_str = 'bac_data*'
    file_str = 'subst_data.pickle'
    L = int(1e3)
    savefig  = True
    load_dat = False
    skip = 50

    cmap_means = cubehelix_palette(n_colors=6)
    cmap_replicates = cubehelix_palette(n_colors=nexps*len(data_folders), start=.5, 
                                        rot=-.75, light=.85, dark=.1, hue=1, gamma=.95)
    mut_str = 'mut_data*'
    mut_files = get_files(data_folders, inner_folder, mut_str, nexps)
    make_subst_plot(mut_files, nexps, L, load_dat, output_folder, file_str, savefig, skip)
    plt.show()
