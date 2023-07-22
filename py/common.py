"""
Nicholas M. Boffi
11/5/21

This file contains routines common to multiple python files used for
processing the output of lenski_sim.cc.
"""

import numpy as np
import glob
import typing
from typing import Tuple
import sys
from scipy.stats import sem
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from numba import jit


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


def make_mm_clonal_hist(
        xlabel: str, 
        ylabel: str, 
        bin_centers: np.ndarray, 
        bin_widths: np.ndarray,
        data: np.ndarray, 
        output_folder: str, 
        save_str: str, 
        savefig: bool,
        cmap_avgs, 
        cmap_replicates, 
        log=False) -> None:
    """ Makes a histogram for visualizing the effects of multiple mutations, 
    clonal interference, and fixation probabilities. """
    replicate_means, replicate_sems \
            = np.zeros_like(bin_centers), np.zeros_like(bin_centers)
    compute_bin_averaged_replicate_means(replicate_means, replicate_sems, data)
    nonzero_inds = replicate_means > 0
    fig, ax = plt.subplots()
    ax.bar(bin_centers[nonzero_inds], replicate_means[nonzero_inds], 
           bin_widths[nonzero_inds], yerr=replicate_sems[nonzero_inds], 
           alpha=.8, color=cmap_avgs[1], linewidth=2, edgecolor='k')
    for replicate in range(data.shape[0]):
        ax.scatter(bin_centers[nonzero_inds], 
                   data[replicate, :][nonzero_inds], marker='x',
                   color=cmap_replicates[replicate], alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')
    plt.tight_layout()

    if savefig:
        plt.savefig("%s/%s.pdf" % (output_folder, save_str), dpi=300, 
                    transparent=True)


def merge_bins(
        merge_bin_fac: int, 
        bin_edges: np.ndarray, 
        bin_counts: np.ndarray,
        outlier_num: int, 
        merge_tail: bool) -> Tuple[np.ndarray, np.ndarray]:
    """ Merges merge_bin_fac bins in bin_edges and bin_counts to create 
    smaller bins. Preserves the left-most bin edge (minimum selection 
    coefficient), the right-most bin edge (maximum selection coefficient), 
    and preserves the distinction between beneficial and deleterious mutations.

    If merge_tail is set to True, then we will merge the last outlier_num 
    bins into one and preserve all the others."""

    if ((merge_bin_fac == 1) and merge_tail):
        bin_counts_retained = bin_counts[:, :-outlier_num]
        bin_edges_retained = bin_edges[:-outlier_num]
        bin_counts_final = np.sum(bin_counts[:, -outlier_num:], axis=1)

        bin_counts_new = np.zeros((bin_counts_retained.shape[0], 
                                  bin_counts_retained.shape[1] + 1))
        bin_counts_new[:, :-1] = bin_counts_retained
        bin_counts_new[:, -1] = bin_counts_final

        bin_edges_new = np.zeros(bin_edges_retained.size + 1)
        bin_edges_new[:-1] = bin_edges_retained
        bin_edges_new[-1] = bin_edges[-1]

        return bin_edges_new, bin_counts_new

    if merge_bin_fac == 1:
        return bin_edges, bin_counts

    else:
        # preserve the min, max, and presence of zero. 
        # otherwise, downsample the edges.
        new_bin_edges = [bin_edges[0]]
        merge_counter = 0
        for ii in range(bin_edges.size):
            merge_counter += 1
            if (bin_edges[ii] == 0) and (merge_counter < merge_bin_fac):
                del new_bin_edges[-1]
                new_bin_edges.append(bin_edges[ii])
                merge_counter = 0
            elif (merge_counter == merge_bin_fac):
                new_bin_edges.append(bin_edges[ii])
                merge_counter = 0
        new_bin_edges.append(bin_edges[-1])
        new_bin_edges = np.array(new_bin_edges)

        # because we preserve the min and max, the out-of-bounds counts 
        # are preserved as well.
        new_bin_counts = np.zeros((bin_counts.shape[0], new_bin_edges.size-1))

        # identify each old bin with its center, and find the index 
        # corresponding  to the new bin to update the new bin counts.
        bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2
        for replicate in range(bin_counts.shape[0]):
            for ii, center in enumerate(bin_centers):
                new_ind = get_bin_ind(center, new_bin_edges)
                new_bin_counts[replicate, new_ind] += bin_counts[replicate, ii]

        return new_bin_edges, new_bin_counts


def trim_bins(
        bin_counts: np.ndarray, 
        bin_edges: np.ndarray, 
        merge_bin_fac: int, 
        nexps: int, 
        outlier_num: int, 
        merge_tail: bool,
        drop_zeros: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """ Process the bins coming out of lenski_sim.cc on a replicate basis.
    Throw out the out-of-bounds bins, remove any zero tails, and downsample 
    the bins.
    """

    # throw out the out-of-bounds - if bins were chosen 
    # correctly, this will be negligible.
    bin_counts = bin_counts[:, 1:-1]

    # throw out the zero tails
    if drop_zeros:
        min_ind = min([np.min(np.nonzero(bin_counts[replicate, :])) \
                for replicate in range(nexps)])
        max_ind = max([np.max(np.nonzero(bin_counts[replicate, :])) \
                for replicate in range(nexps)])
        bin_counts = bin_counts[:, min_ind:max_ind+1]

        # normalize bin_edges to correspond to the new bin_counts
        bin_edges = bin_edges[min_ind:max_ind+2]

    # perform any downsampling necessary as post-processing
    bin_edges, bin_counts = merge_bins(merge_bin_fac, bin_edges, bin_counts, 
                                       outlier_num, merge_tail)

    return bin_edges, bin_counts


def load_bins(
        outer_folders: str, 
        inner_folder: str, 
        nexps: int, 
        file_index: int=-1) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the bin edges (consistent across all replicates) and the 
    bin counters for each experiment. """
    folder_name = "%s/%s/" % (outer_folders[0], inner_folder + str(0))
    bin_edges = np.fromfile('%sbin_edges.dat.bin' % folder_name, 
                            dtype=np.float64)
    bin_counts = np.zeros((nexps*len(outer_folders), bin_edges.size+1))

    for replicate in range(nexps):
        for outer_folder in outer_folders:
            folder_name = "%s/%s/" % (outer_folder, \
                    inner_folder + str(replicate))

            bin_counts_list \
                    = glob.glob('%s%s' % (folder_name, 'bin_counts*'))

            bin_counts_list.sort(key = lambda x: float(x.split('.')[-2]))

            bin_counts[replicate, :] \
                    = np.fromfile(bin_counts_list[file_index], dtype=np.int32)

    return bin_edges, bin_counts


def plot_bac_count_over_time(
        max_nbacs: np.ndarray, 
        times: np.ndarray, 
        N0: float, 
        curr_strain: int, 
        skip: int, 
        cmap_replicates: list, 
        cmap_means: list, 
        output_folder: str, 
        savefig: bool) -> None:
    """ Plots the bacteria count as a function of time of 
    the dominant few strains. """
    fig, ax = plt.subplots()
    for replicate in range(max_nbacs.shape[0]):
        output_inds = max_nbacs[replicate, :, curr_strain] >= 0
        plt.plot(times, max_nbacs[replicate, :, curr_strain]/N0, 
                 color=cmap_replicates[replicate], alpha=0.25)
    replicate_mean = np.mean(max_nbacs[:, :, curr_strain]/N0, axis=0)
    replicate_sem = sem(max_nbacs[:, :, curr_strain]/N0, axis=0)
    plt.errorbar(times[output_inds], replicate_mean[output_inds], 
                 yerr=replicate_sem[output_inds], color=cmap_means[5], 
                 linewidth=2.5, alpha=1.0, errorevery=skip)

    plt.xlabel("day")
    plt.ylabel(r"$N_i/N_0$")
    plt.title("bacteria count strain %d" % (curr_strain+1))
    plt.tick_params(axis='both')
    ax.minorticks_on()
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.grid(True, which='both')
    plt.tight_layout()

    if savefig:
        plt.savefig("%s/max_nbacs_strain%d.pdf" \
                % (output_folder, curr_strain+1), dpi=300, transparent=True)


def plot_mechanism_over_time(
        nexps: int, 
        mechanism_data: np.ndarray, 
        times: np.ndarray, 
        cmap_replicates: list, 
        cmap_means: list, 
        xlabel: str, 
        ylabel: str, 
        title: str, 
        output_folder: str, 
        save_title: str, 
        savefig: bool, 
        skip: int) -> None:
    """ Constructs a mechanism plot over time (e.g. rank over time, fitness 
    increment over time, selection coefficient over time). """
    fig, ax = plt.subplots()
    for replicate in range(nexps):
        output_inds = mechanism_data[replicate, :] >= 0
        plt.plot(times[output_inds], 
                 mechanism_data[replicate, :][output_inds], 
                 color=cmap_replicates[replicate],
                 alpha=.80, linewidth=2.25)
    replicate_mean = np.mean(mechanism_data, axis=0)
    replicate_sem = sem(mechanism_data, axis=0)
    # plt.scatter(times[output_inds][::skip], replicate_mean[output_inds][::skip], color=cmap_replicates[7], marker='o', s=20,
    # plt.plot(times[output_inds][::skip], replicate_mean[output_inds][::skip], color=cmap_replicates[8], linewidth=3)
    # plt.errorbar(times[output_inds], replicate_mean[output_inds], yerr=replicate_sem[output_inds], marker='o', ms=0.5,
                 # color=cmap_means[5], errorevery=skip, linewidth=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tick_params(axis='both')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.minorticks_on()
    ax.grid(True, which='both')
    plt.tight_layout()

    if savefig:
        plt.savefig("%s/%s.pdf" % (output_folder, save_title), 
                    dpi=300, transparent=True)


# @jit
def get_rank_eff_and_select_info(
        nexps: int, 
        n_top_strains: int, 
        L: int, 
        mut_files: list, 
        bac_files: list) -> Tuple[np.ndarray, 
                                  np.ndarray, 
                                  np.ndarray, 
                                  np.ndarray, 
                                  np.ndarray, 
                                  np.ndarray]:
    """ Compute the mean rank, average beneficial fitness increment, and 
    average beneficial selection coefficient over time over all strains. 
    Do the same for the dominant few strains, and return the bacteria count 
    over time for the dominant few strains.

    Input:
    -----
    nexps: Number of replicate experiments.
    n_top_strains: Number of dominante strains of interest.
    L: Size of the genome.
    mut_files: A list (over replicates) of output files storing 
    mutation information.
    bac_files: A lit (over replicates) of output files storing bacterial 
    strain information.

    Returns:
    --------
    mean_ranks: Mean (over all bacteria) rank over time.
    mean_incs: Mean (over all bacteria) expected beneficial fitness 
    increments over time.
    mean_selects: Mean (over all bacteria) expected beneficial selection 
    coefficients over time.
    max_ranks: Rank over time for the n_top_strains dominant strains.
    max_incs: Expected beneficial fitness increments over time for the 
    n_top_strains dominant strains.
    max_selects: Expected beneficial selection coefficients over time for 
    the n_top_strains dominant strains.
    """
    # allocate necessary space for all of the data.
    ntimes = max([len(mut_file_list) for mut_file_list in mut_files])
    mean_ranks = np.zeros((nexps, ntimes))
    mean_incs = np.zeros((nexps, ntimes))
    mean_selects  = np.zeros((nexps, ntimes))
    max_ranks = np.zeros((nexps, ntimes, n_top_strains))
    max_incs = np.zeros((nexps, ntimes, n_top_strains))
    max_selects = np.zeros((nexps, ntimes, n_top_strains))
    max_nbacs = np.zeros_like(max_ranks)

    # loop over each day for each replicate
    for replicate, mut_file_list in enumerate(mut_files):
        bac_file_list = bac_files[replicate]
        for jj, mut_file in enumerate(mut_file_list):
            # extract the bacteria data
            bac_file = bac_file_list[jj]
            bac_data = np.fromfile(bac_file, dtype=np.float32)
            fits, nbac = bac_data[1::2], bac_data[::2]
            top_nbac_inds = np.argsort(nbac)[::-1][:n_top_strains]

            # extract the mutation and separation data
            mut_data = np.fromfile(mut_file, dtype=np.float32)
            seps = np.argwhere(mut_data > L).flatten()

            # extract the rank and increment data from the mutation data
            ranks, incs = np.zeros_like(seps, dtype=np.float32), \
                    np.zeros_like(seps, dtype=np.float32)
            ranks[:-1] = mut_data[seps[1:] - 2]; ranks[-1] = mut_data[-2]
            # ranks_test = np.concatenate((mut_data[seps[1:] - 2], np.array([mut_data[-2]])))
            incs[:-1] = mut_data[seps[1:] - 1]; incs[-1] = mut_data[-1]
            # incs_test = np.concatenate((mut_data[seps[1:] - 1], np.array([mut_data[-1]])))


            # compute the mean statistics
            mean_ranks[replicate, jj] = np.sum(nbac*ranks)/np.sum(nbac)
            mean_incs[replicate, jj] = np.sum(nbac*incs)/np.sum(nbac)
            mean_fit = np.sum(nbac*fits)/np.sum(nbac)
            mean_selects[replicate, jj] = mean_incs[replicate, jj]/mean_fit

            # compute the max statistics
            max_ranks[replicate, jj, :len(top_nbac_inds)] \
                    = ranks[top_nbac_inds]
            max_incs[replicate, jj, :len(top_nbac_inds)]  \
                    = incs[top_nbac_inds]
            max_nbacs[replicate, jj, :len(top_nbac_inds)] \
                    = nbac[top_nbac_inds]
            max_selects[replicate, jj, :len(top_nbac_inds)] \
                    = max_incs[replicate, jj, :len(top_nbac_inds)] \
                    / fits[top_nbac_inds]


        print('Finished finding rank, increment, and select info on \
                experiment %d/%d' % (replicate+1, len(mut_files)))

    return mean_ranks, mean_incs, mean_selects, max_ranks, \
            max_incs, max_selects, max_nbacs


def get_n_mutants(
        data_folder: str,
        inner_folder: str,
        ntimes: int,
        nexps: int) -> np.ndarray:
    """ Get the number of mutants that have occurred up to this time. """
    n_mutants = np.zeros((nexps, ntimes))
    for curr_exp in range(nexps):
        n_mutants[curr_exp, :] = np.fromfile("%s/%s%d/nmuts.bin" \
                % (data_folder, inner_folder, curr_exp), dtype=np.int32)

    return n_mutants


# @jit
def get_fit_data(
        nexps: int,
        n_top_strains: int,
        ntimes: int,
        bac_files: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Extract mean and dominant fitness trajectories as a function
    of time for each replicate. """
    mean_fits = np.zeros((nexps, ntimes))
    max_fits = np.zeros((nexps, ntimes))
    dominant_fits = np.zeros((nexps, ntimes, n_top_strains))

    for replicate, bac_list in enumerate(bac_files):
        for jj, bac_file in enumerate(bac_list):
            bac_data = np.fromfile(bac_file, dtype=np.float32)
            nbac = bac_data[::2]
            fits = bac_data[1::2]
            # argsort sorts in ascending order - flip it around.
            top_nbac_inds = np.argsort(nbac)[::-1][:n_top_strains] 

            # average over bacteria, not strains.
            mean_fits[replicate, jj] = np.sum(fits*nbac)/np.sum(nbac)  
            max_fits[replicate, jj] = np.max(fits)
            dominant_fits[replicate, jj, :len(top_nbac_inds)] \
                    = fits[top_nbac_inds]
        print("Finished loading fitness data on replicate %d/%d" \
                % (replicate+1, len(bac_files)))

    return mean_fits, max_fits, dominant_fits


def load_Jijs(Jij_arr: np.ndarray, L: int):
    """ Takes the vector of Jijs as loaded from binary and 
    converts it into a matrix. """
    Jijs = np.zeros((L, L))
    n_elements = 0
    for row in range(L):
        Jijs[row, row+1:] = Jij_arr[n_elements:n_elements + L-row-1]
        n_elements += L-row-1
    return Jijs + Jijs.T


@jit
def get_available_beneficial_mutations(
        Jijs: np.ndarray, 
        his: np.ndarray, 
        spins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the distribution of beneficial mutations. Observe that we 
    do not use fast fitness computation for this - this is because numpy 
    indexing is faster in Python than a for loop + fast fitness computation. """
    Jspin = Jijs @ spins
    dFs = -2*spins*(his + 2*Jspin)
    beneficial_inds = np.nonzero(dFs > 0)[0]
    return dFs, beneficial_inds


@jit
def compute_fitness_fast(
        Jijs: np.ndarray, 
        his: np.ndarray, 
        mut_sites: set, 
        mut_ind: int,
        Jinit: np.ndarray, 
        init_spins: np.ndarray, 
        spins: np.ndarray) -> float:
    """ Performs the fast fitness computation algorithm. """
    Jcross = 0
    if len(mut_sites) > 0:
        existing_mut_inds = np.array(list(mut_sites))
        Jcross = Jijs[mut_ind, existing_mut_inds] \
                @ init_spins[existing_mut_inds]
    dF = -spins[mut_ind]*(2*his[mut_ind] + 4*Jinit[mut_ind] - 8*Jcross)
    return dF


# @jit
def compute_asymptotic_fitness(
        output_folder: str, 
        inner_folder: str, 
        nexps: int, 
        L: int, 
        beta: float):
    """ Compute a potential asymptotic fitness value by relaxing 
    the spin glass over time. """
    # store the asymptotic value over each experiment
    relax_vals = np.zeros(nexps)
    for replicate in range(nexps):
        # load the magnetic fields
        his = np.fromfile("%s/%s%d/his.dat.bin" \
                % (output_folder, inner_folder, replicate), dtype=np.float64)
        spins = np.loadtxt("%s/%s%d/alpha0s.dat" \
                % (output_folder, inner_folder, replicate))

        # check for epistasis
        try:
            Jij_arr = np.fromfile("%s/%s%d/Jijs.dat.bin" \
                    % (output_folder, inner_folder, replicate), 
                    dtype=np.float64)
            Jijs = load_Jijs(Jij_arr, L)
        except:
            print('Could not load Jijs!')
            Jijs = None

        # without epistasis, we don't need this.
        if (Jijs is None) or np.all(Jijs == 0):
            total_mag = np.sum(np.abs(his))
            Foff = 1 - his @ spins
            relax_vals[replicate] = Foff + total_mag
        else:
            # convention, choose initial fitness = 1
            fit = 1

            # for fast fitness computation
            init_spins = spins.copy()
            Jinit = Jijs @ init_spins
            mut_sites = set()

            # keep minimizing to get to nearest local minimum
            not_optimal = True

            # when < .01*L beneficial mutations remain, switch to hill climbing
            hill_climb = True

            # keep track of diagnostics
            nflips = 0
            n_neg_accepted = 0
            n_pos_accepted = 0

            while not_optimal:
                # perform MCMC
                if not hill_climb:
                    # grab a random mutation, compute its fitness, 
                    # and accept with some probability.
                    mut_ind = np.random.randint(low=0, high=L)
                    dF = compute_fitness_fast(Jijs, his, mut_sites, mut_ind, 
                                              Jinit, init_spins, spins)
                    accept_prob = min([1, np.exp(beta*dF)])
                    flip_prob = np.random.uniform()

                # hill climb, randomly selecting from the beneficial mutations
                else:
                    dFs, beneficial_inds \
                            = get_available_beneficial_mutations(Jijs, 
                                                                 his, 
                                                                 spins)

                    # randomly choose among the remaining beneficial mutations
                    if beneficial_inds.size > 0:
                        mut_ind = np.random.choice(beneficial_inds)
                        flip_prob, accept_prob = 0, 1
                        dF = dFs[mut_ind]
                    else:
                        not_optimal = False
                        flip_prob, accept_prob = 1, 0
                        relax_vals[replicate] = fit
                        print('Finished relaxation on replicate %d. \
                                n_pos_accept: %d, n_neg_accept: %d, \
                                fit: %g' % (replicate, n_pos_accepted, \
                                n_neg_accepted, fit))

                # perform the acceptance/rejection step
                if flip_prob < accept_prob:
                    spins[mut_ind] = -spins[mut_ind]
                    fit += dF

                    if dF < 0:
                        n_neg_accepted += 1
                    else:
                        n_pos_accepted += 1

                    if mut_ind in mut_sites:
                        mut_sites.remove(mut_ind)
                    else:
                        mut_sites.add(mut_ind)

                    if (nflips % 500) == 0:
                        dFs, _ = get_available_beneficial_mutations(Jijs, 
                                                                    his, 
                                                                    spins)
                        n_avail = np.sum(dFs > 0)
                        print('Data for replicate %d: fit: %g, n_avail %d, \
                               n_sites: %d, n_pos_accept: %d, \
                               n_neg_accept: %d' % (replicate, fit, n_avail, \
                               len(mut_sites), n_pos_accepted, n_neg_accepted))

                        # switch to hill climbing regime when we 
                        # are sufficiently low
                        if (n_avail < .01*L):
                            hill_climb = True

                    nflips += 1

    return relax_vals


def compute_bin_averaged_replicate_means(
        replicate_means: np.ndarray, 
        replicate_sems: np.ndarray, 
        bin_data: np.ndarray) -> None:
    """Compute bin-averaged replicate means, correctly ensuring we do not 
    average over replicates with zero values."""
    for ii in range(replicate_means.size):
        curr_dat = bin_data[:, ii]
        nonzero_dat = curr_dat[curr_dat > 0]
        replicate_means[ii] = 0 if nonzero_dat.size == 0 \
                else np.mean(nonzero_dat)
        replicate_sems[ii] = 0 \
                if (nonzero_dat.size == 0 or nonzero_dat.size == 1) \
                else sem(nonzero_dat)


def make_binned_scatter_plot(
        bin_centers: np.ndarray, 
        data: list, 
        xlabel: str, 
        ylabel: str, 
        ylim: list, 
        output_folder: str, 
        fig_title: str, 
        savefig: bool, 
        cmap, 
        cmap_replicates) -> None:
    """ Make a scatter plot of data against bin_centers. """
    fig, ax = plt.subplots()
    replicate_means, replicate_sems \
            = np.zeros_like(bin_centers), np.zeros_like(bin_centers)
    compute_bin_averaged_replicate_means(replicate_means, replicate_sems, data)
    nonzero_inds = replicate_means > 0

    plt.scatter(bin_centers[nonzero_inds],  replicate_means[nonzero_inds], 
                s=50, edgecolors=cmap[3], marker='o')
    plt.errorbar(bin_centers[nonzero_inds], replicate_means[nonzero_inds], 
                 yerr=replicate_sems[nonzero_inds], linestyle='None', 
                 color=cmap[3])

    for replicate in range(data.shape[0]):
        plt.scatter(bin_centers[nonzero_inds], 
                    data[replicate, :][nonzero_inds], 
                    color=cmap_replicates[replicate],
                    marker='x', alpha=0.3)

    plt.plot()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    if (ylim[0] is not None) and (ylim[1] is not None):
        plt.ylim(ylim)
    else:
        plt.ylim([ylim[0], np.max(data) + .1*np.max(data)])

    plt.tick_params(axis='both')
    ax.minorticks_on()
    ax.grid(True, which='both')
    plt.tight_layout()
    if savefig:
        plt.savefig("%s/%s.pdf" % (output_folder, fig_title), 
                    dpi=300, transparent=True)


def compute_avg_within_bins(
        fits: list, 
        bin_data: list, 
        L: int, 
        bin_edges: np.ndarray,
        total_counters: np.ndarray, 
        sum_counters: np.ndarray):
    """ Computes average data within each fitness bin for each replicate.
        Used for rank and average beneficial increment calculations.

    Input:
    ------
    fits:           A list (over replicates) of a fitness data for every 
                    strain that has been output.
    bin_data:       A list (over replicates) of data (e.g rank, average 
                    beneficial fitness effect)
                    for every strain that has been output.
    bin_edges:      The edges of the bins we want to count within.
    total_counters: The total number of datapoints that have fit 
                    within each bin.
    sum_counters:   The sum of the datapoints within each bin.

    Returns:
    --------
    avgs: The average of bin_data within each bin.

    """
    avgs = np.zeros_like(total_counters)
    for replicate in range(len(bin_data)):
        curr_data = bin_data[replicate]
        curr_fits = fits[replicate]
        ind = 0
        for data, fit in zip(curr_data, curr_fits):
            # if we did not output rank information on this timestep, 
            # lenski_sim.cc will output -L instead.
            if (data > -L):
                bin_ind = get_bin_ind(fit, bin_edges)
                total_counters[replicate, bin_ind] += 1
                sum_counters[replicate, bin_ind] += data

            ind += 1
            if (ind % int(curr_data.size/10) == 0):
                print('Finished binning data %d/%d on replicate %d/%d' \
                        % (ind, curr_data.size, replicate+1, len(bin_data)))

    total_counters[total_counters == 0] = -1
    return sum_counters / total_counters


def setup_replicate_bins(
        data: list, 
        nbins: int, 
        split_val: float):
    """ Bins the data for each replicate. """
    min_data = min([np.min(curr_data) for curr_data in data])
    max_data = max([np.max(curr_data) for curr_data in data])
    bin_edges, total_counters, sum_counters \
            = setup_bins(min_data, max_data, nbins, split_val)
    bin_widths  = bin_edges[1:] - bin_edges[:-1]
    bin_centers = bin_widths/2  + bin_edges[:-1]
    replicate_total_counters = np.zeros((len(data), total_counters.size))
    replicate_sum_counters = np.zeros_like(replicate_total_counters)

    return bin_edges, replicate_total_counters, replicate_sum_counters, \
            bin_widths, bin_centers


def compute_rsq(input_data: np.ndarray, fit_data: np.ndarray) -> float:
    """ Computes the R^2 value for a nonlinear fit. """
    input_mean = np.mean(input_data)
    SS_tot = np.sum((input_data - input_mean)**2)
    SS_res = np.sum((fit_data - input_data)**2)
    return 1 - SS_res/SS_tot


def get_current_muts_and_selects(
        mut_data: np.ndarray, 
        seps: np.ndarray) -> Tuple[list, list, dict]:
    """ Extracts the sequences of mutations and corresponding selection 
    coefficients from a given day's output of mutation data.

    Input:
    ------
    mut_data: A numpy array holding mutation data for each strain.
    seps: An array of indices into mut_data separating the individual strains.

    Returns:
    --------
    current_muts: A list of mutation sequences stored as numpy arrays.
    current_selects: A list of selection coefficients corresponding to the 
    latest mutation in each sequence.
    sep_inds: A dictionary mapping indices into current_muts and 
    current_selects to indices in the original mut_data array.
    """

    current_muts, current_selects, sep_inds = [], [], {}
    for jj in range(seps.size):
        try:
            end = end_ind(seps, jj)
            cdat = mut_data[seps[jj] + 1:end]
            curr_effs, final_eff = cdat[1:-1:2], cdat[-1]
            select = final_eff/(1 + np.sum(curr_effs))
            current_selects.append(select)
            current_muts.append(cdat[::2])
            sep_inds[len(current_muts) - 1] = jj
        except:
            print("Error: %s" % sys.exc_info()[1])

    assert(len(current_muts) == len(current_selects))
    return current_muts, current_selects, sep_inds


def get_bin_ind(select: float, bin_edges: np.ndarray) -> int:
    return max(0, np.argmax(select <= bin_edges) - 1)


def setup_bins(
        min_val: float, 
        max_val: float, 
        nbins: int, 
        split_val=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Constructs a set of bins over values 
    (selection coefficients, fitness, etc.)

    Input:
    -----
    min_select: Lower bound on the selection coefficient.
    max_select: Upper bound on the selection coefficient.
    nbins:      Total number of bins.
    split_val:  Split the bins around split_val. i.e., 0 for 
    selection coefficients or 1 for fitness.
    """
    bin_edges = np.linspace(min_val, max_val, nbins+1)

    # ensure that deleterious mutations are separated from beneficial 
    # mutations, or fitness > 1 is separated from fitness < 1
    positive_ind = np.argmin(bin_edges < split_val)
    bin_edges[positive_ind] = split_val

    bin_total_counters = np.zeros(nbins)
    bin_yes_counters = np.zeros(nbins)
    return bin_edges, bin_total_counters, bin_yes_counters


def end_ind(seps: list, strain_ind: int):
    """ Computes the final index for a given mutation sequence. """
    return -2 if (strain_ind == (seps.size-1)) else (seps[strain_ind + 1] - 2)


def get_files(
        outer_folders: str, 
        inner_folder: str, 
        file_str: str, 
        nexps: int) -> list:
    """Returns a list of files matching the glob regex expression file_str
    in sorted order by day."""

    files = []
    for curr_exp in range(nexps):
        for outer_folder in outer_folders:
            folder_name = "%s/%s/" \
                    % (outer_folder, inner_folder + str(curr_exp))
            file_list = glob.glob(folder_name + file_str)
            file_list.sort(key = lambda x: float(x.split('.')[-2]))
            files.append(file_list)
            print("Finished loading file %d/%d" % (curr_exp + 1, nexps))

    return files


@jit
def get_ranks_and_avg_beneficial_incs(
        mut_files: list, 
        L: int) -> Tuple[list, list]:
    """ Get the rank over time and average beneficial fitness increment 
    over time data for each replicate."""
    ranks = []
    avg_beneficial_incs = []
    for replicate, mut_list in enumerate(mut_files):
        replicate_ranks = np.array([])
        replicate_avg_beneficial_incs = np.array([])
        for file_ind, mut_file in enumerate(mut_list):
            data = np.fromfile(mut_file, dtype=np.float32)
            seps = np.argwhere(data > L).flatten()
            curr_ranks \
                    = np.concatenate((data[seps[1:] - 2], np.array([data[-2]])))
            curr_avg_beneficial_incs \
                    = np.concatenate((data[seps[1:] - 1], np.array([data[-1]])))

            # if np.all(curr_ranks > -L) and np.all(replicate_avg_beneficial_incs > -L):
            replicate_ranks = np.concatenate((replicate_ranks, curr_ranks))
            replicate_avg_beneficial_incs \
                    = np.concatenate((replicate_avg_beneficial_incs, \
                    curr_avg_beneficial_incs))

        print('Finished getting rank and fitness effect \
                data on replicate %d/%d' % (replicate+1, len(mut_files)))
        ranks.append(replicate_ranks)
        avg_beneficial_incs.append(replicate_avg_beneficial_incs)

    return ranks, avg_beneficial_incs


def get_fits(bac_files: list, L: int) -> list:
    """ Get the fitness over time for each strain in each replicate. """
    fits = []
    for replicate, bac_list in enumerate(bac_files):
        curr_fits = np.array([])
        for bac_file in bac_list:
            bac_dat = np.fromfile(bac_file, dtype=np.float32)
            curr_fits = np.concatenate((curr_fits, bac_dat[1::2]))
        print('Finished getting fitness data \
                on replicate %d/%d' % (replicate+1, len(bac_files)))
        fits.append(curr_fits)

    return fits


def get_fixed_mutations(
        mut_files: list, 
        L: int, 
        calc_true_subst: bool=True, 
        file_index: int=-1,
        print_info=True) -> Tuple[list, list]:
    """ Finds all fixed mutations at output point corresponding to file_index.

    Input:
    -----
    mut_files:       A list (over replicates) of output files containing 
                     mutation information.
    L:               Size of the genome.
    calc_true_subst: If true, calculate the true number of fixed mutations. 
                     If false, just return the minimum number of mutations in 
                     any strain as an approximation to the true number of 
                     fixed mutations.
                     Typically, these differ by just a few mutations, but just 
                     returning the minimum is significantly faster.
    file_index:      The index for the output file we want to find the fixed 
                     mutations at. -1 corresponds to last output.

    Returns:
    ---------
    fixed_muts: A list (over replicates) of sets of the fixed mutations.
    selects:    A list (over replicates) of dictionaries mapping the byte strings for
                the fixed mutations to selection coefficients.
    """
    # first find the minimum mutation sequence for each replicate, as all we need to do is
    # compare all substrings of the minimum sequence to all other mutation sequences.
    min_seqs = [0]*len(mut_files)
    min_lengths = [100*L]*len(mut_files)
    min_seq_effs = [0]*len(mut_files)
    for replicate, mut_list in enumerate(mut_files):
        mut_data = np.fromfile(mut_list[file_index], dtype=np.float32)
        seps = np.argwhere(mut_data > L).flatten()
        for jj in range(seps.size):
            end = end_ind(seps, jj)
            curr_mut_seq = mut_data[seps[jj] + 1:end][::2]
            mut_length = len(curr_mut_seq)
            if mut_length < min_lengths[replicate]:
                min_lengths[replicate] = mut_length
                min_seqs[replicate] = curr_mut_seq
                min_seq_effs[replicate] = mut_data[seps[jj] + 1:end][1::2]

        if print_info:
            print("Found the minimum sequence on \
                    replicate %d/%d" % (replicate+1, len(mut_files)))

    # using the minimum sequences, check for fixed sequences
    selects, fixed_muts = [], []
    if calc_true_subst:
        for replicate, mut_list in enumerate(mut_files):
            # load in the data for this replicate.
            mut_data = np.fromfile(mut_list[file_index], dtype=np.float32)
            seps = np.argwhere(mut_data > L).flatten()
            min_seq, min_length = min_seqs[replicate], min_lengths[replicate]

            # find all fixed mutations by testing successively smaller sequences
            # if we have that one mutation fixed, then all subsequences fixed.
            # hence, start with the entire minimum sequence and work backwards.
            found_fixed_muts = False
            test_seq_length = min_length
            while not found_fixed_muts:
                test_seq = min_seq[:test_seq_length].tobytes()
                mut_fixed = True
                strain_ind = 0
                while ((mut_fixed) and (strain_ind < seps.size)):
                    end = end_ind(seps, strain_ind)
                    seq_sub_bytes = mut_data[seps[strain_ind] + 1:end][::2][:test_seq_length].tobytes()
                    if test_seq != seq_sub_bytes:
                        mut_fixed = False
                    else:
                        strain_ind += 1

                if mut_fixed:
                    found_fixed_muts = True
                else:
                    test_seq_length -= 1

            # now, add all fixed mutations corresponding to this sequence.
            curr_fixed_muts = set()
            curr_selects = {}
            min_seq_eff = min_seq_effs[replicate]
            for ii in range(1, test_seq_length):
                fixed_mut_byte_seq = min_seq[:ii].tobytes()
                curr_fixed_muts.add(fixed_mut_byte_seq)
                curr_select = min_seq_eff[ii]/(1 + np.sum(min_seq_eff[:ii-1]))
                curr_selects[fixed_mut_byte_seq] = curr_select

            assert(len(curr_fixed_muts) <= min_length)
            fixed_muts.append(curr_fixed_muts)
            selects.append(curr_selects)

            if print_info:
                print("Finished finding fixed mutations on \
                        replicate %d/%d" % (replicate+1, len(mut_files)))
                print("Number of fixed mutations: %d, \
                        min_length: %d" % (len(curr_fixed_muts), min_length))
    else:
        for min_seq in min_seqs:
            curr_fixed_muts = set()
            curr_selects = {}
            for ii in range(1, min_seq.size):
                fixed_mut_byte_seq = min_seq[:ii].tobytes()
                curr_fixed_muts.add(fixed_mut_byte_seq)
                curr_select = min_seq_eff[ii]/(1 + np.sum(min_seq_eff[:ii-1]))
                curr_selects[fixed_mut_byte_seq] = curr_select
            fixed_muts.append(curr_fixed_muts)
            selects.append(curr_selects)


    return fixed_muts, selects
