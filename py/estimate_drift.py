"""
Nicholas M. Boffi

This file contains code to estimate the drift term over the course of an SSWM relaxation.
"""


import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.stats import sem
import typing
import pickle
import seaborn as sns
from seaborn import cubehelix_palette
from approximate_kernel import draw_disorder, compute_initial_spin_sequence
from common import get_available_beneficial_mutations
from numba import jit, njit
from scipy.stats import linregress


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


@jit
def approximate_drift(n_disorders: int, n_init_states: int, init_ranks: np.ndarray, betas: np.ndarray, L: int, rho: float, Delta: float) -> np.ndarray:
    """ Approximates the drift term by averaging over initial spin flips. """
    init_dFs = np.zeros((betas.size, n_disorders, n_init_states, init_ranks.size, L))
    ddFs = np.zeros((betas.size, n_disorders, n_init_states, init_ranks.size, L))
    ddF_vars = np.zeros((betas.size, n_disorders, n_init_states, init_ranks.size, L))

    for beta_index, beta in enumerate(betas):
        for curr_disorder in range(n_disorders):
            Jijs, his = draw_disorder(beta, L, rho, Delta)
            for curr_init_state in range(n_init_states):
                for rank_index, init_rank in enumerate(init_ranks):
                    spins = compute_initial_spin_sequence(Jijs, his, init_rank, L)
                    dFs, _ = get_available_beneficial_mutations(Jijs, his, spins)
                    init_dFs[beta_index, curr_disorder, curr_init_state, :] = dFs
                    ddF_mat = 4*spins[:, None]*Jijs*spins[None, :] # fancy broadcasting
                    dFs *= dFs > 0
                    pflips = dFs/np.sum(dFs)
                    E_ddFs = ddF_mat @ pflips 
                    E_ddFs_sq = (ddF_mat**2) @ pflips
                    ddFs[beta_index, curr_disorder, curr_init_state, rank_index, :] = E_ddFs
                    ddF_vars[beta_index, curr_disorder, curr_init_state, rank_index, :] = E_ddFs_sq - E_ddFs**2

                    print('Finished computing the expected change for iteration %d/%d on beta value %d/%d' % (1 + rank_index + init_ranks.size*curr_init_state + curr_disorder*n_init_states*init_ranks.size, n_disorders*n_init_states*init_ranks.size, beta_index+1, betas.size))

    return init_dFs, ddFs, ddF_vars


def make_drift_plot(betas: np.ndarray, init_dFs: np.ndarray, ddFs: np.ndarray, ddF_vars: np.ndarray, n_disorders: int, 
                    n_init_states: int, output_folder: str, cmap) -> None:
    """ Plot expected drift <ddF> and its variance as a function of dF over different values of beta """
    ddf_fig, ddf_ax = plt.subplots()
    var_fig, var_ax = plt.subplots()

    slopes, intercepts = np.zeros_like(betas), np.zeros_like(betas)
    slope_errs, intercept_errs = np.zeros_like(betas), np.zeros_like(betas)
    max_ddf, min_ddf, max_var, min_var = -1, 1, -1, 1
    for ii, beta in enumerate(betas):
        # obtain the initial dFs and the average change in dFs for this value of beta
        curr_init_dFs = init_dFs[ii, :, :, :, :].ravel()
        curr_ddFs = ddFs[ii, :, :, :, :].ravel()
        curr_ddF_vars = ddF_vars[ii, :, :, :, :].ravel()

        # bin the initial dFs over all runs
        dF_hist, bin_edges = np.histogram(curr_init_dFs)
        bin_indices = np.searchsorted(bin_edges, curr_init_dFs) - 1

        # now compute the average value for the variance and the mean over all runs
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        curr_avg, curr_avg_var = np.zeros_like(bin_centers), np.zeros_like(bin_centers)

        # compute the average within bins
        np.add.at(curr_avg, bin_indices, curr_ddFs)
        curr_avg /= dF_hist

        # compute the average variance within bins
        np.add.at(curr_avg_var, bin_indices, curr_ddF_vars)
        curr_avg_var /= dF_hist

        # also compute the variance and the standard error of the mean
        # curr_var = np.zeros_like(bin_centers)
        # np.add.at(curr_var, bin_indices, curr_ddFs)
        # curr_var *= -2*curr_avg
        # curr_var += curr_avg**2
        # np.add.at(curr_var, bin_indices, curr_ddFs**2)
        # curr_var[dF_hist > 0] /= (dF_hist - 1)[dF_hist > 0]
        # curr_sem = np.copy(np.sqrt(curr_var))
        # curr_sem[dF_hist > 0] /= np.sqrt(dF_hist[dF_hist > 0])

        # find the drift approximant
        result = linregress(bin_centers[1:-1], curr_avg[1:-1])
        slopes[ii] = result.slope
        intercepts[ii] = result.intercept
        slope_errs[ii] = result.stderr
        print("Finished computing bin statistics for beta value %d/%d" % (ii+1, betas.size))
        print('slope: %g, intercept: %g' % (slopes[ii], intercepts[ii]))

        # update the maxes for axis limits
        max_ddf = max(max_ddf, np.max(curr_avg))
        min_ddf = min(min_ddf, np.min(curr_avg))
        max_var = max(max_var, np.max(curr_avg_var))
        min_var = min(min_var, np.min(curr_avg_var))

        # plot the results
        var_ax.scatter(bin_centers, curr_avg_var, alpha=0.8, label=r"$\beta=%s$" % str(betas[ii]), linewidth=2, color=cmap[ii+1])
        ddf_ax.scatter(bin_centers, curr_avg, alpha=0.8, label=r"$\beta=%s$" % str(betas[ii]), color=cmap[ii+1], s=75)
        # ddf_ax.fill_between(bin_centers, curr_avg - np.sqrt(curr_avg_var), curr_avg + np.sqrt(curr_avg_var), color=cmap[ii+1], alpha=0.25)
        ddf_ax.plot(bin_centers, slopes[ii]*bin_centers + intercepts[ii], linewidth=3, color=cmap[ii+1], alpha=0.35)

    # save the slope and intercept data
    np.save(open('%s/slopes.npz' % output_folder, "wb"), slopes)
    np.save(open('%s/intercepts.npz' % output_folder, "wb"), intercepts)

    ## set up axis labels for the ddF plot
    ddf_ax.set_xlabel(r"$\Delta F$")
    ddf_ax.set_ylabel(r"$\langle \Delta \Delta F\rangle$")
    ddf_ax.set_ylim(auto=True)
    ddf_ax.legend(ncol=2, loc='lower left')
    ddf_ax.set_ylim([-1.0e-4, .6e-4])
    ddf_fig.tight_layout()

    ## set up axis labels for the variance plot
    var_ax.set_xlabel(r"$\Delta F$")
    var_ax.set_ylabel(r"$Var(\Delta \Delta F)$")
    var_ax.set_ylim(auto=True)
    var_ax.set_ylim([-1e-7, 5e-7])
    var_ax.legend(ncol=3)
    var_fig.tight_layout()

    # plot the slopes as a function of beta        
    result = linregress(betas, slopes)
    print('slopes slope: %g, slopes intercept: %g' % (result.slope, result.intercept))
    slope_fig, ax = plt.subplots()
    ax.scatter(betas, slopes)
    ax.plot(betas, result.slope*betas + result.intercept)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("slope")
    ax.set_ylim(auto=True)
    plt.tight_layout()

    # plot the intercepts as a function of beta
    result = linregress(betas, intercepts)
    print('interecepts slope: %g, intercepts intercept: %g' % (result.slope, result.intercept))
    intercept_fig, ax = plt.subplots()
    ax.scatter(betas, intercepts)
    ax.plot(betas, result.slope*betas + result.intercept)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("intercept")
    ax.set_ylim(auto=True)
    plt.tight_layout()

    if save_figs:
        ddf_fig.savefig('%s/ddf.pdf' % output_folder, transparent=True, dpi=300, bbox_inches='tight')
        var_fig.savefig('%s/ddf_var.pdf' % output_folder, transparent=True, dpi=300, bbox_inches='tight')
        slope_fig.savefig('%s/slopes.pdf' % output_folder, transparent=True, dpi=300, bbox_inches='tight')
        intercept_fig.savefig('%s/intercepts.pdf' % output_folder, transparent=True, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # distribution information
    L = 1000
    betas = np.array([0.05, 0.25, 0.5, 0.75, 1.0])
    rho = 0.05
    Delta = 0.0075
    n_disorders = 30
    n_init_states = 30
    init_ranks = np.array([100])
    load_dat = True

    # plotting colors
    cmap = cubehelix_palette(n_colors=len(betas)+1, start=.5, rot=-.75, light=.85, dark=0, hue=.9, gamma=.7, reverse=False)

    # where to save the data
    output_folder = 'estimate_drift_rslts/estimate_drift_R100_small'
    save_figs = True

    if not load_dat:
        init_dFs, ddFs, ddF_vars = approximate_drift(n_disorders, n_init_states, init_ranks, betas, L, rho, Delta)
        data_dict = {}
        data_dict['init_dFs'] = init_dFs
        data_dict['ddFs'] = ddFs
        data_dict['ddF_vars'] = ddF_vars
        pickle.dump(data_dict, open('%s/drift_data.pkl' % output_folder, "wb"))
    else:
        data_dict = pickle.load(open('%s/drift_data.pkl' % output_folder, "rb"))
        init_dFs = data_dict['init_dFs']
        ddFs = data_dict['ddFs']
        ddF_vars = data_dict['ddF_vars']

    # visualize the results
    make_drift_plot(betas, init_dFs, ddFs, ddF_vars, n_disorders, n_init_states, output_folder, cmap)

    plt.show()

