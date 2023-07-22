"""
Nicholas M. Boffi

This file contains code that compute a coarse-grained dynamics (Gillespie simulation) for
the SK spin glass under the SSWM approximation. It tracks the evolution
of the distribution of fitness effects over time.
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from common import get_available_beneficial_mutations, compute_fitness_fast, get_bin_ind

import typing
from typing import Tuple

from scipy.stats import sem
from scipy.interpolate import interp1d

from numba import jit, njit

from approximate_kernel import draw_disorder, compute_initial_spin_sequence

import pickle

import seaborn as sns
from seaborn import cubehelix_palette

import time


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
def average_data(data_dict: dict, 
                 npts: int, 
                 fixed_grid: bool) -> dict:
    """ Averages the simulation output over the different initializations.
    Accounts for the fact that each sample has reactions which occur at 
    different times."""
    # unpack the data from the dictionary
    rhos = data_dict['rhos']
    niters, ns, _ = rhos.shape
    times = data_dict['times']
    fits = data_dict['fitnesses']
    means = data_dict['means']
    nfixed_muts = data_dict['nfixed_muts']

    # construct a time array that is fixed for all runs
    if fixed_grid:
        all_times = np.linspace(0, np.max(times.ravel()), npts)
    else:
        all_times = np.sort(np.array(list(set(times.ravel()))))
        npts = all_times.size

    # save the time array into the dictionary
    data_dict['all_times'] = all_times

    # construct new arrays that has all quantities evaluated at the same time points
    even_rhos = np.zeros((niters, ns, npts))
    even_nfixed_muts = np.zeros((niters, npts))
    even_fits = np.zeros((niters, npts))
    even_means = np.zeros((niters, npts))

    # interpolate the fitness arrays to ensure every fitness value
    # occurs at the same time points despite stochastic event timings
    for curr_iter in range(niters):
        # extract the data
        curr_nfixed_muts = nfixed_muts[curr_iter, :]
        max_nfixed_muts = int(curr_nfixed_muts[-1])
        curr_times = times[curr_iter, :max_nfixed_muts]
        curr_fits = fits[curr_iter, :max_nfixed_muts]
        curr_means = means[curr_iter, :max_nfixed_muts]

        # construct the interpolant
        fits_interp = interp1d(curr_times, curr_fits, kind='previous', 
                               fill_value='extrapolate')
        means_interp = interp1d(curr_times, curr_means, kind='previous', 
                                fill_value='extrapolate')
        nfixed_muts_interp = interp1d(curr_times, 
                                      curr_nfixed_muts[:max_nfixed_muts], 
                                      kind='previous', 
                                      fill_value='extrapolate')

        # evaluate the interpolant at the fixed times
        even_fits[curr_iter, :] = fits_interp(all_times)
        even_means[curr_iter, :] = means_interp(all_times)
        even_nfixed_muts[curr_iter, :] = nfixed_muts_interp(all_times)

        # interpolate the distribution for each s value
        curr_rho = rhos[curr_iter, :, :]
        for s_index in range(curr_rho.shape[0]):
            fixed_s_interp = interp1d(curr_times, curr_rho[s_index, :max_nfixed_muts], 
                                      kind='previous', fill_value='extrapolate')
            even_rhos[curr_iter, s_index, :] = fixed_s_interp(all_times)

    # compute the statistics
    data_dict['mean_rhos'] = np.mean(even_rhos, axis=0)
    data_dict['sem_rhos'] = sem(even_rhos, axis=0)

    data_dict['mean_means'] = np.mean(even_means, axis=0)
    data_dict['sem_means'] = sem(even_means, axis=0)

    data_dict['mean_fits'] = np.mean(even_fits, axis=0)
    data_dict['sem_fits'] = sem(even_fits, axis=0)

    data_dict['mean_nfixed_muts'] = np.mean(even_nfixed_muts, axis=0)
    data_dict['sem_nfixed_muts'] = sem(even_nfixed_muts, axis=0)

    # also save the full data.
    data_dict['even_fits'] = even_fits
    data_dict['even_means'] = even_means
    data_dict['even_rhos'] = even_rhos
    data_dict['even_nfixed_muts'] = even_nfixed_muts

    return data_dict


def plot_distribution(mean_rhos: np.ndarray, 
                      time_skip: int, 
                      bin_plot_values: np.ndarray, 
                      title: str, 
                      cmap: list) -> None:
    """ Plots the distribution of fitness effects over time. 
    Assumes that rhos is of shape niters x nbins x ntime. """
    fig, ax = plt.subplots()
    for color_index, tt in enumerate(range(0, mean_rhos.shape[1], time_skip)):
        plt.plot(bin_plot_values, mean_rhos[:, tt], color=cmap[color_index], 
                 marker='o', linewidth=1.0, ms=3.0)
    plt.xlabel("fitness increment")
    plt.ylabel(r"count")
    plt.title(title)
    plt.tight_layout()


@jit
def approximate_distribution(niters: int, 
                             n_disorders: int, 
                             init_rank: int, 
                             beta: float, 
                             L: int,
                             rho: float, 
                             Delta: float, 
                             fit_bins: np.ndarray, 
                             random_walk: bool, 
                             drift: bool, 
                             rw_spins: bool,
                             diff_init: bool, 
                             output_folder: str, 
                             output_str: str, 
                             total_fixed_muts: int, 
                             dilute_fac: float,
                             drift_slope: float, 
                             drift_intercept: float) -> np.ndarray:
    """ Approximates the distribution of fitness effects over time using the 
    Gillespie algorithm under the SSWM dynamics.
    The code here is very similar to approximate_kernel in approximate_kernel.py. """
    rhos = np.zeros((niters*n_disorders, fit_bins.size-1, total_fixed_muts+1))
    means = np.zeros((niters*n_disorders, total_fixed_muts+1))
    fitnesses = np.zeros((niters*n_disorders, total_fixed_muts+1))
    times = np.zeros((niters*n_disorders, total_fixed_muts+1))
    nfixed_muts = np.zeros((niters*n_disorders, total_fixed_muts+1))
    init_spins = np.zeros(L)

    # define the glass
    for disorder_index in range(n_disorders):
        Jijs, his = draw_disorder(beta, L, rho, Delta)
        sparsity_pattern = Jijs != 0

        # fix the sequence if we are not averaging over initial spins
        if not diff_init:
            init_spins = compute_initial_spin_sequence(Jijs, his, init_rank, L)

        # relax niters initial sequences and find the distributions the whole way down
        for iteration_index in range(niters):
            # set up this initial sequence if we are averaging over initial spins
            if diff_init:
                init_spins = compute_initial_spin_sequence(Jijs, his, init_rank, L)

            # set up this sequence
            iteration = iteration_index + niters*disorder_index
            Jinit = Jijs @ init_spins
            spins = np.copy(init_spins)
            fitness = 1.0
            fitnesses[iteration, 0] = fitness
            times[iteration, 0] = 0.0
            dFs, beneficial_inds = get_available_beneficial_mutations(Jijs, his, spins)

            # compute the initial distributions
            rho_fits, _ = np.histogram(dFs, bins=fit_bins)
            rhos[iteration, :, 0] = rho_fits
            means[iteration, 0] = np.mean(dFs[beneficial_inds])

            # keep going until we have relaxed all the way down to our desired end rank
            for fixation_index in range(total_fixed_muts):
                # draw uniform random numbers that determine the time of the next 
                # reaction and which reaction
                r1, r2 = np.random.random(2)

                # compute the propensity functions (fixation probabilities) 
                # for the spin flips
                props = 2*dFs/fitness*np.log(dilute_fac)/L
                props[props > 1] = 1
                props[props < 0] = 0

                # draw the time of the next mutation
                alpha0 = np.sum(props)
                tau = np.log(1.0/r1)/alpha0
                times[iteration, fixation_index+1] = \
                        times[iteration, fixation_index] + tau
                nfixed_muts[iteration, fixation_index+1] = \
                        nfixed_muts[iteration, fixation_index] + 1

                # find the chosen mutation.
                cumulative_sum_arr = np.cumsum(props)/alpha0
                flip_index = np.argmax(r2 < cumulative_sum_arr)
                dF = dFs[flip_index]
                fitness += dF
                fitnesses[iteration, fixation_index+1] = fitness
                spins[flip_index] *= -1

                # compute new distribution information
                if random_walk:
                    # compute the new distribution by flipping the increment 
                    # for the flipped spin and randomly perturbing all the others.
                    flipped_dF = -dFs[flip_index]
                    perturbations = \
                            2*np.random.randn(dFs.size)*np.sqrt(beta/(L*rho))*Delta
                    perturbations *= spins*spins[flip_index] if rw_spins else 1.0

                    ## true drift
                    perturbations += \
                            (drift_slope*dFs + drift_intercept) if drift else 0.0
                    
                    ## play with drift
                    # perturbations += (-1e-3*beta*dFs - 1.25e-4*beta) if drift else 0.0

                    ## re-draw sparsity every time
                    # sparsity_pattern = np.random.random(size=dFs.size) < rho
                    # perturbations *= sparsity_pattern

                    ## fixed sparsity pattern
                    perturbations *= sparsity_pattern[flip_index, :]

                    dFs += perturbations
                    dFs[flip_index] = flipped_dF
                    beneficial_inds = np.nonzero(dFs > 0)[0]
                elif drift:
                    flipped_dF = -dFs[flip_index]
                    perturbations = (drift_slope*dFs + drift_intercept)
                    perturbations *= sparsity_pattern[flip_index, :]
                    dFs += perturbations
                    dFs[flip_index] = flipped_dF
                    beneficial_inds = np.nonzero(dFs > 0)[0]
                else:
                    dFs, beneficial_inds = get_available_beneficial_mutations(Jijs, 
                                                                              his, 
                                                                              spins)

                rho_fits, _ = np.histogram(dFs, bins=fit_bins)
                rhos[iteration, :, fixation_index+1] = rho_fits
                if beneficial_inds.size > 0:
                    means[iteration, fixation_index+1] = np.mean(dFs[beneficial_inds])
                print('Fixed mutation %d/%d on iteration %d/%d.' \
                        % (fixation_index+1, total_fixed_muts, 
                            iteration+1, niters*n_disorders))

                # cancel the loop if we ran out of beneficial mutations before the upper bound
                if (beneficial_inds.size == 0):
                    # the dynamics dictate that the distributions, means, and 
                    # fitnesses stay the same over time.
                    rhos[iteration, :, fixation_index+2:] = rho_fits[:, None]
                    means[iteration, fixation_index+2:] = \
                            means[iteration, fixation_index+1]
                    fitnesses[iteration, fixation_index+2:] = fitness
                    times[iteration, fixation_index+2:] = \
                            times[iteration, fixation_index+1]
                    nfixed_muts[iteration, fixation_index+2:] = \
                            nfixed_muts[iteration, fixation_index+1]
                    break


    # save and return the result of the computation.
    data_dict = {}
    data_dict['rhos'] = rhos
    data_dict['means'] = means
    data_dict['bins'] = fit_bins
    data_dict['fitnesses'] = fitnesses
    data_dict['times'] = times
    data_dict['dilute_fac'] = dilute_fac
    data_dict['nfixed_muts'] = nfixed_muts
    data_dict['fit_bins'] = fit_bins
    return data_dict


if __name__ == '__main__':
    # distribution information
    L = 1000
    # betas = np.array([0.05, 0.25, 0.5, 0.75, 1.0])
    betas = np.array([0.05])
    rho = 0.05
    Delta = 0.0075
    dilute_fac  = 100
    npts = int(5e3)

    # taken from fits
    drift_folder = 'estimate_drift_rslts/estimate_drift_R100_small_fixed_disorder'
    slopes = np.load('%s/slopes.npz' % drift_folder)
    intercepts = np.load('%s/intercepts.npz' % drift_folder)

    # relaxation information
    n_disorders = 1
    niters = 100
    init_rank = int(L/10)
    total_fixed_muts = int(10*init_rank)

    # binning information
    smin, smax = -0.1, 0.1
    nbins = 50
    deleterious_bins = np.linspace(smin, 0, nbins)
    beneficial_bins = np.linspace(0, smax, nbins)
    fit_bins = np.concatenate((deleterious_bins[:-1], beneficial_bins))
    bin_centers = np.diff(fit_bins)/2 + fit_bins[:-1]
    bin_widths = .75*np.diff(fit_bins)

    # plotting colors
    cmap = cubehelix_palette(n_colors=npts, start=.5, rot=-.75, light=.85, dark=0, 
                             hue=.9, gamma=.7, reverse=False)

    output_folder = 'dist_dat/bootstrap_data/diffuse_drift'
    output_str = ''
    random_walk = True
    rw_spins = False
    diff_init = True
    drift = True

    # do the computation and visualize
    for ii, beta in enumerate(betas):
        data_dict = approximate_distribution(niters, n_disorders, init_rank, beta, L, 
                                             rho, Delta, fit_bins, random_walk,
                                             drift, rw_spins, diff_init, output_folder, 
                                             output_str, total_fixed_muts, dilute_fac,
                                             0 if not drift else slopes[ii], 
                                             0 if not drift else intercepts[ii])
        print('Average fixed mutations on beta=%g: %d' \
                % (beta, np.mean(data_dict['nfixed_muts'][:, -1])))

        # save the averaged data to the dictionary and output the dictionary
        data_dict = average_data(data_dict, npts, fixed_grid=(True if beta==0 else False))
        pickle.dump(data_dict, 
                open('%s/%s_b=%s.pkl' \
                        % (output_folder, output_str, str(beta)), "wb"), protocol=4)

    print("Finished computing distributions over time.")
    # plot_distribution(data_dict['mean_rhos'], time_skip, fit_bins[:-1], '', cmap)
    # plt.show()
