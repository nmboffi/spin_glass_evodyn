"""
Nicholas M. Boffi

This file contains code the plot the fixation probability as a function of 
selection coefficient data from lenski_sim.cc. This code will also fit the 
fixation probability to a linear or functional form, and provide count data 
demonstrating how the fixation probability computed.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import typing
import sys
from typing import Tuple
from common import *
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.stats import sem
import pickle
import seaborn as sns
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
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['figure.titlesize'] = 30
mpl.rcParams['font.size'] = 27.5
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.dpi'] = 125


def fill_bins(
        fixed_mut_selects: list, 
        bin_edges: np.ndarray, 
        bin_fix_counters: np.ndarray,
        print_info: bool=True) -> None:
    """ Computes the number of selection coefficients that lie 
    in each bin for the fixed mutants. """
    for replicate, current_selects in enumerate(fixed_mut_selects):
        select_values = np.array(list(current_selects.values())).ravel()
        bin_fix_counters[replicate, :], _ \
                = np.histogram(select_values, bins=bin_edges)

        if print_info:
            print('Finished binning on experiment %d/%d' \
                    % (replicate+1, len(fixed_mut_selects)))


def make_hists(
        bin_centers: np.ndarray, 
        bin_widths: np.ndarray, 
        probs: np.ndarray, 
        bin_fix_counters: np.ndarray,
        bin_counts: np.ndarray, 
        savefig: bool, 
        output_folder: str) -> None:

    ### fixation probability histogram
    make_mm_clonal_hist('selection coefficient', 'probability of fixation', 
                        bin_centers, bin_widths, probs, output_folder,
                        'fix_prob_hist', savefig, cmap_fits, cmap_replicates)

    ### total count histogram
    make_mm_clonal_hist('selection coefficient', 'total number of mutations', 
                        bin_centers, bin_widths, bin_counts, output_folder,
                        'fix_prob_total_count_log', savefig, cmap_fits, 
                        cmap_replicates, log=True)
    make_mm_clonal_hist('selection coefficient', 'total number of mutations', 
                        bin_centers, bin_widths, bin_counts, output_folder,
                        'fix_prob_total_count', savefig, cmap_fits, 
                        cmap_replicates, log=False)

    ### fixation count histogram
    make_mm_clonal_hist('selection coefficient', 'number of fixations', 
                        bin_centers, bin_widths, bin_fix_counters, 
                        output_folder, 'fix_prob_fix_count', savefig, 
                        cmap_fits, cmap_replicates, log=False)


def cut_and_get_beneficial(
        bin_centers: np.ndarray, 
        probs: np.ndarray, 
        outlier_num: int,
        drop_zeros: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Returns the centers and probabilities of beneficial mutations 
    after trimming outliers. """
    # extract only the places where we have data for beneficial 
    # mutations in each replicate
    beneficial_inds = bin_centers > 0

    # remove anywhere we have zero count data
    if drop_zeros:
        for ii in range(bin_centers.size):
            if beneficial_inds[ii] == True:
                if np.sum(probs[:, ii]) == 0:
                    beneficial_inds[ii] = False

    # only look at beneficial indices
    beneficial_probs = probs[:, beneficial_inds]
    beneficial_centers = bin_centers[beneficial_inds]

    # cut off some outliers at the tail if we didnt already merge 
    # them into one observation.
    if (outlier_num > 0):
       beneficial_probs = beneficial_probs[:, :-outlier_num]
       beneficial_centers = beneficial_centers[:-outlier_num]

    return beneficial_centers, beneficial_probs, beneficial_inds


def fit_and_make_scatter(
        bin_centers: np.ndarray, 
        probs: np.ndarray, 
        bin_counts: np.ndarray, 
        nexps: int, 
        outlier_num: int,
        plot_replicates: bool, 
        merge_tail: bool, 
        savefig: bool, 
        output_folder: str) -> None:
    """ Fits the fixation probability data as a function of selection 
    coefficient to linear and nonlinear functional forms and make 
    a scatter plot. """

    # only look at beneficial mutations and cut off outliers
    beneficial_centers, beneficial_probs, beneficial_inds \
            = cut_and_get_beneficial(bin_centers, probs, outlier_num)

    # compute the bin averages and bin SEMs, but dont average over 
    # replicates who had zero in the bin.
    replicate_means = np.zeros(beneficial_centers.size)
    replicate_sems = np.zeros(beneficial_centers.size)
    compute_bin_averaged_replicate_means(replicate_means, 
                                         replicate_sems, 
                                         beneficial_probs)

    # functional forms we will fit the data to
    def linear_fit(s, k):
        return k*s

    def nonlinear_fit(s, k, B):
        return k*s*np.exp(-B/s)

    # plot the replicates in the background
    fig, ax = plt.subplots()
    if plot_replicates:
        for replicate in range(nexps):
            plt.scatter(bin_centers[beneficial_inds], 
                        probs[replicate, beneficial_inds], marker='x',
                        color=cmap_replicates[replicate], alpha=0.4)

    # plot the fit against the grand mean
    plt.scatter(beneficial_centers, replicate_means, marker='o', 
                color=cmap_fits[3], facecolors='none')
    plt.errorbar(beneficial_centers, replicate_means, yerr=replicate_sems, 
                 linestyle='None', color=cmap_fits[3], alpha=1.0)

    # linear fit
    print('replicate_means:', replicate_means)
    print('replicate_sems:', replicate_sems)
    print('beneficial_centers:', beneficial_centers)
    fit_inds = replicate_sems > 0
    slope, cov = curve_fit(linear_fit, 
                           beneficial_centers[fit_inds], 
                           replicate_means[fit_inds], 
                           sigma=replicate_sems[fit_inds])
    slope = slope[0]
    linear_data = linear_fit(beneficial_centers, slope)
    linear_rsq = compute_rsq(replicate_means, linear_data)
    print('linear slope:', slope)

    # nonlinear fit
    nlin_params, cov = curve_fit(nonlinear_fit, 
                                 beneficial_centers[fit_inds], 
                                 replicate_means[fit_inds], 
                                 sigma=replicate_sems[fit_inds], 
                                 p0=[.435, .1], 
                                 bounds=([0.434, 0], [0.436, np.inf]), 
                                 loss='cauchy')
    k, B = nlin_params[0], nlin_params[1]
    nonlinear_data = nonlinear_fit(beneficial_centers, k, B)
    nonlinear_rsq = compute_rsq(replicate_means, nonlinear_data)
    print('nonlinear slope: %g, nonlinear B: %g' % (k, B))

    # plot the fit data
    plt.plot(beneficial_centers, 
            linear_fit(beneficial_centers, slope), 
             color=cmap_fits[1], 
             label=r"$p(s) = ks: k=%0.3f, R^2=%0.4f$" % (slope, linear_rsq))
    plt.plot(beneficial_centers, 
             nonlinear_fit(beneficial_centers, k, B), 
             color=cmap_fits[2],
             label=r"$p(s) = kse^{-B/s}: k=%0.3f, B=%0.5f, R^2=%0.4f$" \
                     % (k, B, nonlinear_rsq))

    plt.xlabel("selection coefficient")
    plt.ylabel("probability of fixation")
    plt.tick_params(axis='both')
    plt.legend(framealpha=0.3)
    plt.xlim([-.0001, np.max(beneficial_centers) + .0005])
    plt.ylim([-.0001, np.max(replicate_means) + .0005])
    plt.tight_layout()

    if savefig:
        plt.savefig("%s/fix_prob.pdf" % (output_folder), 
                    dpi=300, transparent=True)


def plot_fix_prob_vs_selec_coeff(
        mut_files: list, 
        L: int, 
        merge_bin_fac: int, 
        bin_edges: np.ndarray,
        bin_counts: np.ndarray, 
        nexps: int, 
        outlier_num: int, 
        merge_tail: bool, 
        load_data: bool,
        output_folder: str, 
        plot_replicates: bool, 
        file_str: str, 
        savefig: bool) -> None:
    """ Plot and fit the fixation probability as a function of 
    selection coefficient. """

    # perform some processing to simplify the bins.
    bin_edges, bin_counts = trim_bins(bin_counts, bin_edges, merge_bin_fac, 
                                      nexps, outlier_num, merge_tail)

    if not load_data:
        _, fixed_mut_selects = get_fixed_mutations(mut_files, L)
        data_dict = {}
        data_dict['fixed_mut_selects'] = fixed_mut_selects
        data_dict['bin_counts'] = bin_counts
        pickle.dump(data_dict, 
                    open("%s/%s" % (output_folder, file_str), "wb"))
    else:
        data_dict = pickle.load(
                open("%s/%s" % (output_folder, file_str), "rb")
                )
        fixed_mut_selects = data_dict['fixed_mut_selects']
        bin_counts = data_dict['bin_counts']

    # tally up the counts for each bin
    bin_fix_counters = np.zeros_like(bin_counts)
    fill_bins(fixed_mut_selects, bin_edges, bin_fix_counters)

    # ensure no division by zero when computing probabilities
    bin_counts[bin_counts == 0] = -1
    probs = bin_fix_counters / bin_counts
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_centers = bin_widths/2 + bin_edges[:-1]
    bin_widths = .8*bin_widths

    make_hists(bin_centers, bin_widths, probs, bin_fix_counters, 
               bin_counts, savefig, output_folder)
    fit_and_make_scatter(bin_centers, probs, bin_counts, nexps, outlier_num, 
                         plot_replicates, merge_tail, savefig, output_folder)


if __name__ == '__main__':
    ## config settings
    nexps = 10
    merge_bin_fac = 1
    L = int(1e3)
    outlier_num = 9


    # TODO: I don't think merge_tail is working correctly right now.
    merge_tail = False    
    mut_str = 'mut_data*'
    bac_str = 'bac_data*'
    load_dat = False


    # data_folders = ['/scratch/nick/lenski_data/experiment_data/epistasis_scan_bp9comp/clonal/L1e3_r1e2_m2e-4_bp75']
    # data_folders = ['/scratch/nick/lenski_data/experiment_data/epistasis_scan_bp9comp/sswm/L1e3_r1e2_m1e-8_b1']
    # data_folders = ['/scratch/nick/lenski_data/experiment_data/tunnel_test/clonal/L1e3_r5_m1e-3']
    # data_folders = ['/scratch/nick/lenski_data/experiment_data/tsq_test_L1e3_output']
    # data_folders = ['/scratch/nick/lenski_data/experiment_data/tsq_test_L1e3_100rep']
    # data_folders = ['/scratch/nick/lenski_data/experiment_data/tsq_test_L1e4']
    # data_folders  = ['/scratch/nick/lenski_data/experiment_data/epistasis_scan_6_15_21/sswm_distributions/L1e3_r1e2_m1e-8_b0.000000']
    # data_folders = ['clonal_vepi_extended_8_3_21/L1e3_r1e2_m2e-4_b0.5']
    data_folders= \
            ['/scratch/nick/lenski/mu_scan_10_4_21_epi_track_nmuts_fixed/L1e3_r1e2_b0p25_m0.0001000000']
    inner_folder = 'replicate'
    output_folder = '%s/figs' % data_folders[0]
    file_str = 'fix_prob_data.pickle'
    savefig = True
    plot_replicates = True


    ## colorschemes
    cmap_replicates = cubehelix_palette(n_colors=nexps*len(data_folders))
    cmap_fits = cubehelix_palette(n_colors=4, start=.5, rot=-.75, light=.85, 
                                  dark=.1, hue=1, gamma=.95)


    ## do the computation
    mut_files = get_files(data_folders, inner_folder, mut_str, nexps)
    bin_edges, bin_counts = load_bins(data_folders, inner_folder, nexps)
    plot_fix_prob_vs_selec_coeff(mut_files, L, merge_bin_fac, bin_edges, 
                                 bin_counts, nexps, outlier_num, merge_tail, 
                                 load_dat, output_folder, plot_replicates, 
                                 file_str, savefig)

    plt.show()
