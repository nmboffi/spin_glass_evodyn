"""
Nicholas M. Boffi
11/6/21

This file contains code to plot the fixation probability as a 
function of the mutation rate.
"""


import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from common import *
import pickle
import seaborn as sns
from seaborn import cubehelix_palette
from scipy.integrate import cumtrapz
import glob
from plot_fix_prob import cut_and_get_beneficial, fill_bins


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


def count_average_fixed_mutations(mut_files: list, L: int, file_index: int):
    """ Get the average number of fixed mutations (across replicates) at a specific
    time index. """

    _, fixed_mut_selects = get_fixed_mutations(mut_files, L, 
                                               calc_true_subst=True,
                                               file_index=file_index,
                                               print_info=False)
    
    navg_muts = np.mean(
            np.array([len(fixed_muts_dict) for fixed_muts_dict in fixed_mut_selects])
            )

    return navg_muts, fixed_mut_selects


def compute_fixation_probability_trajectory_horizon(
        data_folders: list,
        mu_vals: np.ndarray,
        mut_str: str,
        inner_folder: str,
        L: int,
        file_indices: np.ndarray,
        bin_edges: np.ndarray,
        nexps: int,
        output_folder: str,
        file_str: str) -> dict:
        """ Compute the fixation probability as a function of time 
        over replicates and different values of mu. Do so by computing
        the probability that a mutation has fixed by this time.
        """

        # store data as a dictionary of dictionaries
        data_dict = {}
        for ii, mu_val_folder in enumerate(data_folders):
            # load in mutation and increment file information
            curr_data = {}
            mut_files = get_files([mu_val_folder], inner_folder, mut_str, nexps)
            inc_files = get_files([mu_val_folder], inner_folder, 'inc_dist*', nexps)

            # load in fitness information
            bac_files = get_files([mu_val_folder], inner_folder, 'bac_data*', nexps)
            _, _, dominant_fits \
                    = get_fit_data(nexps, n_top_strains=1, 
                                   ntimes=file_indices.size, bac_files=bac_files)
            dominant_fits = dominant_fits[:, :, 0].T


            # save time and fitness information
            dt = float(mut_files[0][1].split('.')[-2])
            curr_data['time'] = file_indices*dt
            curr_data['dominant_fits'] = dominant_fits

            # compute the fixation probability data for this value of mu 
            # over all replicates and over all files of interest
            for jj, file_index in enumerate(file_indices):
                ### compute fixation probability information.
                _, fixed_mut_selects = \
                        get_fixed_mutations(mut_files, L, 
                                            calc_true_subst=True, 
                                            file_index=file_index,
                                            print_info=False)

                _, bin_counts = load_bins([mu_val_folder], inner_folder, 
                                           nexps, file_index=file_index)
                bin_counts = bin_counts[:, 1:-1]  # cut out out-of-bounds bins.
                bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

                bin_fix_counters = np.zeros_like(bin_counts)
                fill_bins(fixed_mut_selects, bin_edges, bin_fix_counters, 
                          print_info=False)
                bin_counts[bin_counts == 0] = -1
                curr_probs = bin_fix_counters / bin_counts

                beneficial_centers, beneficial_probs, beneficial_inds \
                        = cut_and_get_beneficial(bin_centers, curr_probs,
                                                 outlier_num=0, drop_zeros=False)

                if jj == 0:
                    probs = np.zeros((len(file_indices), *beneficial_probs.shape))
                    total_counts = np.zeros_like(probs)
                    fix_counts = np.zeros_like(probs)
                    s_dists = np.zeros_like(probs)

                probs[jj, :, :] = beneficial_probs
                total_counts[jj, :, :] = bin_counts[:, beneficial_inds]
                fix_counts[jj, :, :] = bin_fix_counters[:, beneficial_inds]

                ### get the increment distribution
                for replicate_ind, inc_list in enumerate(inc_files):
                    # load the distribution of increments (for the dominant strain)
                    curr_inc_dist = np.fromfile(inc_list[file_index], dtype=np.float64)
                    curr_s_dist = curr_inc_dist / dominant_fits[jj, replicate_ind]
                    binned_s_dist, _ = np.histogram(curr_s_dist, bin_edges)
                    s_dists[jj, replicate_ind, :] = binned_s_dist[beneficial_inds]


                print(f'Finished file {jj+1}/{len(file_indices)} on mu value '\
                        + f'{ii+1}/{len(data_folders)}')

            # save the data in the overall dictionary with the numerical 
            # value of the current mu value as the key
            curr_data['probs'] = probs
            curr_data['total_counts'] = total_counts
            curr_data['fix_counts'] = fix_counts
            curr_data['s_dists'] = s_dists
            data_dict[mu_vals[ii]] = curr_data

        # save information common to all mu values.
        data_dict['file_indices'] = file_indices
        data_dict['beneficial_centers'] = beneficial_centers

        # save the data to the output
        pickle.dump(data_dict, 
                    open("%s/%s" % (output_folder, file_str), "wb"))

        return data_dict


def compute_fixation_probability_trajectory_fixed_windows(
        data_folders: list,
        mu_vals: np.ndarray,
        mut_str: str,
        inner_folder: str,
        L: int,
        n_time_points: int,
        bin_edges: np.ndarray,
        nexps: int,
        output_folder: str,
        file_str: str) -> dict:
        """ Compute the fixation probability as a function of time 
        over replicates and different values of mu. """
        bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

        # store data as a dictionary of dictionaries
        data_dict = {}
        for ii, mu_val_folder in enumerate(data_folders):
            curr_data = {}
            mut_files = get_files([mu_val_folder],
                                  inner_folder, mut_str, nexps)
            dt = float(mut_files[0][1].split('.')[-2])

            if ii == 0:
                indices = np.logspace(1, np.log10(len(mut_files[0])), 
                                      num=n_time_points,
                                      base=10, 
                                      dtype=np.int64)
                indices[-1] = len(mut_files[0])-1

            time = np.array(indices * dt)

            _, final_fixed_mut_selects= \
                    count_average_fixed_mutations(mut_files, L, 
                                                  file_index=-1)


            # compute the fixation probability across each window.
            # TODO: speed up with a binary search.
            for tt, file_index in enumerate(indices):
                last_window_index = 0 if tt == 0 else indices[tt-1]

                _, last_bin_counts = load_bins([mu_val_folder], 
                                                inner_folder, 
                                                nexps, 
                                                file_index=last_window_index)

                last_fixed_muts, last_fixed_mut_selects = \
                        count_average_fixed_mutations(mut_files, L, 
                                                      file_index=last_window_index)

                print(f'file_index:{file_index}, {len(mut_files[0])}')
                total_fixed_muts, fixed_mut_selects = \
                        count_average_fixed_mutations(mut_files, L, file_index)

                # subtract off last window to get current window
                curr_window_fixed_muts = total_fixed_muts - last_fixed_muts

                print(f'Number of fixed mutations={curr_window_fixed_muts}' \
                        + f' on tt={tt}')

                # compute the bincounters up to this time point
                _, total_bin_counts = load_bins([mu_val_folder], inner_folder, 
                                                nexps, file_index=file_index)

                # subtract off up to last time point to get the current
                # time window.
                curr_window_bin_counts = total_bin_counts - last_bin_counts
                curr_window_bin_counts = curr_window_bin_counts[:, 1:-1]
                bin_fix_counters = np.zeros_like(curr_window_bin_counts)

                # only count the fixed mutations over each replicate that
                # were not in the last time window for that replicate.
                curr_window_fixed_mut_selects = [
                        dict(
                            # set(fixed_mut_selects[kk].items()) \
                            set(final_fixed_mut_selects[kk].items()) \
                                    - set(last_fixed_mut_selects[kk].items())
                                    )
                        for kk in range(len(fixed_mut_selects))
                        ]

                # compute the fixation probabilities over this window
                fill_bins(curr_window_fixed_mut_selects, 
                          bin_edges, 
                          bin_fix_counters, 
                          print_info=False)
                curr_window_bin_counts[curr_window_bin_counts == 0] = -1
                curr_window_probs = bin_fix_counters / curr_window_bin_counts
                beneficial_centers, beneficial_probs, beneficial_inds \
                        = cut_and_get_beneficial(bin_centers, 
                                                 curr_window_probs,
                                                 outlier_num=0, 
                                                 drop_zeros=False)

                # save the probabilities
                if tt == 0:
                    probs = np.zeros((n_time_points, *beneficial_probs.shape))
                    total_counts = np.zeros_like(probs)
                    fix_counts = np.zeros_like(probs)

                probs[tt, :, :] = beneficial_probs
                total_counts[tt, :, :] = curr_window_bin_counts[:, beneficial_inds]
                fix_counts[tt, :, :] = bin_fix_counters[:, beneficial_inds]

                print(f'Finished time point {tt+1}/{n_time_points} on mu value '\
                        + f'{ii+1}/{len(data_folders)}')

            # save the data in the overall dictionary with the numerical 
            # value of the current mu value as the key
            curr_data['time'] = np.array(time)
            curr_data['probs'] = probs
            curr_data['total_counts'] = total_counts
            curr_data['fix_counts'] = fix_counts
            data_dict[mu_vals[ii]] = curr_data


        data_dict['n_time_points'] = n_time_points
        data_dict['beneficial_centers'] = beneficial_centers


        # save the data to the output
        pickle.dump(data_dict, 
                    open("%s/%s" % (output_folder, file_str), "wb"))


        return data_dict



def compute_fixation_probability_trajectory_adaptive_windows(
        data_folders: list,
        mu_vals: np.ndarray,
        mut_str: str,
        inner_folder: str,
        L: int,
        n_time_points: int,
        bin_edges: np.ndarray,
        nexps: int,
        output_folder: str,
        file_str: str) -> dict:
        """ Compute the fixation probability as a function of time 
        over replicates and different values of mu. 
        Do so by computing time windows so that roughly the same
        number of new mutations fix in each bin. """
        bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

        # store data as a dictionary of dictionaries
        data_dict = {}
        for ii, mu_val_folder in enumerate(data_folders):
            curr_data = {}
            mut_files = get_files([mu_val_folder],
                                  inner_folder, mut_str, nexps)
            dt = float(mut_files[0][1].split('.')[-2])
            time = []

            # find the number of fixed mutations at the final time
            average_fixed_muts, final_fixed_mut_selects = \
                    count_average_fixed_mutations(mut_files, L, file_index=-1)
            window_fixed_muts = int(average_fixed_muts/n_time_points)
            print(f'Window fixed muts: {window_fixed_muts}')

            # initialize variables for the window search
            start_index = 1
            _, last_bin_counts = load_bins([mu_val_folder], 
                                            inner_folder, 
                                            nexps, 
                                            file_index=0)
            last_fixed_muts, last_fixed_mut_selects = \
                    count_average_fixed_mutations(mut_files, L, file_index=0)

            # compute the fixation probability across each window.
            # TODO: speed up with a binary search.
            for tt in range(n_time_points):
                curr_window_fixed_muts = 0
                file_index = start_index if tt < n_time_points-1 else -1 
                final_t_flag = False

                while (curr_window_fixed_muts < window_fixed_muts) \
                        and not final_t_flag:
                    # get total number of muttaions up to this file index
                    total_fixed_muts, fixed_mut_selects = \
                            count_average_fixed_mutations(mut_files, L, file_index)

                    # subtract off last window to get current window
                    curr_window_fixed_muts = total_fixed_muts - last_fixed_muts

                    if (curr_window_fixed_muts >= window_fixed_muts) \
                            or (tt == n_time_points-1):

                        if (tt == n_time_points-1):
                            final_t_flag = True

                        print(f'Number of fixed mutations={curr_window_fixed_muts}' \
                                + f' on tt={tt}')

                        # compute the bincounters up to this time point
                        _, total_bin_counts = load_bins([mu_val_folder], inner_folder, 
                                                        nexps, file_index=file_index)

                        # subtract off up to last time point to get the current
                        # time window.
                        curr_window_bin_counts = total_bin_counts - last_bin_counts
                        curr_window_bin_counts = curr_window_bin_counts[:, 1:-1]
                        bin_fix_counters = np.zeros_like(curr_window_bin_counts)

                        # only count the fixed mutations over each replicate that
                        # were not in the last time window for that replicate.
                        curr_window_fixed_mut_selects = [
                                dict(
                                    # set(fixed_mut_selects[kk].items()) \
                                    set(final_fixed_mut_selects[kk].items()) \
                                            - set(last_fixed_mut_selects[kk].items())
                                            )
                                for kk in range(len(fixed_mut_selects))
                                ]

                        # compute the fixation probabilities over this window
                        fill_bins(curr_window_fixed_mut_selects, 
                                  bin_edges, 
                                  bin_fix_counters, 
                                  print_info=False)
                        curr_window_bin_counts[curr_window_bin_counts == 0] = -1
                        curr_window_probs = bin_fix_counters / curr_window_bin_counts
                        beneficial_centers, beneficial_probs, beneficial_inds \
                                = cut_and_get_beneficial(bin_centers, 
                                                         curr_window_probs,
                                                         outlier_num=0, 
                                                         drop_zeros=False)

                        # save the probabilities
                        if tt == 0:
                            probs = np.zeros((n_time_points, *beneficial_probs.shape))
                            total_counts = np.zeros_like(probs)
                            fix_counts = np.zeros_like(probs)

                        probs[tt, :, :] = beneficial_probs
                        total_counts[tt, :, :] = curr_window_bin_counts[:, beneficial_inds]
                        fix_counts[tt, :, :] = bin_fix_counters[:, beneficial_inds]

                        # copy over this point as the end of the window
                        last_bin_counts = np.copy(total_bin_counts)
                        last_fixed_muts = total_fixed_muts
                        last_fixed_mut_selects = \
                                [curr_dict.copy() for curr_dict in fixed_mut_selects]
                        time.append(0.5*dt*(file_index + start_index))

                    file_index += 1

                start_index = file_index
                print(f'Finished time point {tt+1}/{n_time_points} on mu value '\
                        + f'{ii+1}/{len(data_folders)}')


            # save the data in the overall dictionary with the numerical 
            # value of the current mu value as the key
            curr_data['time'] = np.array(time)
            curr_data['probs'] = probs
            curr_data['total_counts'] = total_counts
            curr_data['fix_counts'] = fix_counts
            data_dict[mu_vals[ii]] = curr_data


        data_dict['n_time_points'] = n_time_points
        data_dict['beneficial_centers'] = beneficial_centers


        # save the data to the output
        pickle.dump(data_dict, 
                    open("%s/%s" % (output_folder, file_str), "wb"))


        return data_dict


def plot_fix_probs_over_time(
        data_folders: list,
        mu_vals: np.ndarray,
        mut_str: str,
        inner_folder: str,
        L: int,
        n_time_points: int,
        file_indices:  np.ndarray,
        bin_edges: np.ndarray,
        bin_indices: np.ndarray,
        nexps: int,
        load_data: bool,
        output_folder: str,
        file_str: str,
        savefig: bool) -> None:
    """ Compute the fixation probability as a function of time. 

    Args:
        data_folders  - folders holding data for different mutation rate values.
        mu_vals       - values of the mutation rate.
        mut_str       - base string defining the mutation information files.
        inner_folder  - base folder name describing the different replicates.
        L             - size of the genome.
        n_time_points - number of time points to compute the fixation probability at.
        file_indices  - actual time points to compute the fixation probability at.
        bin_indices   - actual bins to use when tracking fixation probability
                        over time. indexed assuming w.r.t. beneficial bins.
        nexps         - number of replicates per mu value.
        load_data     - whether or not to load the fixation information.
        output_folder - where to save the data and figures.
        file_str      - name to use when saving the data.
        savefig       - whether or not to save any generated figures.
    """
    cmap = cubehelix_palette(n_colors=mu_vals.size+1, start=.5, rot=-.75, 
                             light=.85, dark=.1, hue=1, gamma=.95)

    if not load_data:
        # data_dict = compute_fixation_probability_trajectory(data_folders, mu_vals, 
                                                            # mut_str, inner_folder, L, 
                                                            # n_time_points, bin_edges,
                                                            # nexps, output_folder, 
                                                            # file_str)

        # data_dict = compute_fixation_probability_trajectory_fixed_windows(data_folders, 
                                                                          # mu_vals, 
                                                                          # mut_str, 
                                                                          # inner_folder, 
                                                                          # L, 
                                                                          # n_time_points, 
                                                                          # bin_edges,
                                                                          # nexps, 
                                                                          # output_folder, 
                                                                          # file_str)

        data_dict = compute_fixation_probability_trajectory_horizon(data_folders,
                                                                    mu_vals,
                                                                    mut_str,
                                                                    inner_folder,
                                                                    L,
                                                                    file_indices,
                                                                    bin_edges,
                                                                    nexps,
                                                                    output_folder,
                                                                    file_str)

    else:
        data_dict = pickle.load(
                open("%s/%s" % (output_folder, file_str), "rb")
                )


    beneficial_centers = data_dict['beneficial_centers']
    for bin_index in bin_indices:
        s_val = beneficial_centers[bin_index]
        construct_individual_figure(r'time $\times \mu$',
                                    r'fixation probability',
                                    beneficial_centers,
                                    s_val,
                                    bin_index,
                                    mu_vals,
                                    data_dict,
                                    'probs',
                                    output_folder,
                                    'fixation_prob',
                                    cmap,
                                    plot_analytical_curves=True)

        # construct_individual_figure(r'time $\times \mu$',
                                    # r'number of fixations',
                                    # beneficial_centers,
                                    # s_val,
                                    # bin_index,
                                    # mu_vals,
                                    # data_dict,
                                    # 'fix_counts',
                                    # output_folder,
                                    # 'fixation_count',
                                    # cmap)

        # construct_individual_figure(r'time $\times \mu$',
                                    # r'total number of mutations',
                                    # beneficial_centers,
                                    # s_val,
                                    # bin_index,
                                    # mu_vals,
                                    # data_dict,
                                    # 'total_counts',
                                    # output_folder,
                                    # 'mutation_count',
                                    # cmap)


        print(f'Finished making figures for bin ' \
                + f'{bin_index+1-bin_indices[0]}/{len(bin_indices)}')


def construct_individual_figure(xlabel: str,
                                ylabel: str,
                                beneficial_centers: np.ndarray,
                                s_val: float,
                                bin_index: int,
                                mu_vals: list,
                                data_dict: dict,
                                plot_data_key: str,
                                output_folder: str,
                                save_name: str,
                                cmap: list,
                                plot_analytical_curves: bool=False) -> None:
    """ Makes a plot of the data in data_dict corresponding to plot_data_key. 
    plot_analytical_curves is an option to plot the analytically-predicted fixatin
    probability trajectory. """
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    main_curve_alpha = 1.0 if not plot_analytical_curves else 0.7

    max_val = 0
    min_val = 0

    for mu_index, mu_val in enumerate(mu_vals):
        # unpack the data
        curr_data = data_dict[mu_val]
        curr_time = curr_data['time']
        curr_plot_data = curr_data[plot_data_key]

        # the replicate data goes along the first axis, while here we have the replicate
        # data going along the second axis.
        curr_plot_data_trajs = curr_plot_data[:, :, bin_index].T
        replicate_mean_traj = np.zeros(curr_plot_data.shape[0])
        replicate_sem_traj = np.zeros_like(replicate_mean_traj)
        compute_bin_averaged_replicate_means(replicate_mean_traj,
                                             replicate_sem_traj, 
                                             curr_plot_data_trajs)

        # find indices for plotting based on whether or not we've observed enough
        # data yet.
        if plot_analytical_curves:
            fix_count_data = curr_data['fix_counts'][:, :, bin_index].T
            replicate_mean_fix_count = np.zeros(curr_plot_data.shape[0])
            replicate_sem_fix_count = np.zeros_like(replicate_mean_fix_count)
            compute_bin_averaged_replicate_means(replicate_mean_fix_count,
                                                 replicate_sem_fix_count,
                                                 fix_count_data)
            plot_indices = replicate_mean_fix_count > 0.2*np.max(replicate_mean_fix_count)
        else:
            plot_indices = np.arange(curr_time.size)

        # downsample for plotting 
        curr_time = curr_time[plot_indices]
        replicate_mean_traj = replicate_mean_traj[plot_indices]
        replicate_sem_traj = replicate_sem_traj[plot_indices]

        # for yscale in the fixation probability plots
        max_val = max([max_val, np.max(replicate_mean_traj)])
        min_val = min([min_val, np.min(replicate_mean_traj)])

        ax.plot(curr_time*mu_val, replicate_mean_traj, 
                color=cmap[mu_index+1], lw=4.0, alpha=main_curve_alpha,
                label=rf"$\mu={mu_val}$")

        ax.fill_between(curr_time*mu_val, 
                        replicate_mean_traj - replicate_sem_traj,
                        replicate_mean_traj + replicate_sem_traj, alpha=0.3,
                        color=cmap[mu_index+1])

        # compute and plot the analytical curve
        if plot_analytical_curves:
            pfix_over_time = analytical_prediction(curr_data, bin_index, s_val,
                                                   mu_val, beneficial_centers, 
                                                   curr_time)

            ax.plot(curr_time*mu_val, pfix_over_time[plot_indices], 
                    color=cmap[mu_index+1], ls='-.')
            print('pfix:', pfix_over_time)
            max_val = max([max_val, np.max(pfix_over_time)])
            plt.ylim([1e-5, max_val + max_val/10])

    ax.set_xscale('log')
    ax.set_yscale('log')
    # plt.legend(ncol=3, loc='best', framealpha=0.75)
    plt.tight_layout()

    if savefig:
        plt.savefig(f'{output_folder}/{save_name}_s={s_val:0.5g}.pdf', dpi=300, 
                    bbox_inches='tight')

    # plt.close()


def analytical_prediction(curr_data: dict,
                          bin_index: int,
                          s: float,
                          mu_val: float,
                          beneficial_centers: np.ndarray,
                          curr_time: np.ndarray) -> np.ndarray:
    """ Compute the analytically-predicted pfix(s, t) curve. """
    # extract the distribution information
    curr_s_dists = curr_data['s_dists']
    curr_dominant_fits = curr_data['dominant_fits']

    # compute the average distribution over all replicates
    replicate_mean_s_dists = np.mean(curr_s_dists, axis=1)

    # dilution factor constant
    D = 100
    k = 2 * np.log(D)**2 / (D-1)

    # compute the replicate-averaged rank and mean selection coefficient
    # as functions of time.
    Rbar = np.sum(replicate_mean_s_dists, axis=1)
    sbar = replicate_mean_s_dists @ beneficial_centers / Rbar
    sbar[Rbar == 0] = 0

    # compute the fixation probability for this bin as a function of time.
    mu_b = Rbar * mu_val / L
    Nf = 1e8  # TODO: make this not hard-coded
    N0 = 1e6
    dN = Nf - N0
    pfix = k * s * np.exp(-dN*np.log(N0)/np.log(D) * mu_b * k \
            * (1 + sbar/s)*np.exp(-s/sbar))
    pfix[Rbar == 0] = 0

    # compute the integral defining the analytical prediction
    s_count_over_t = replicate_mean_s_dists[:, bin_index]
    denominator = np.cumsum(s_count_over_t)
    numerator = np.cumsum(pfix * s_count_over_t)

    return numerator / denominator


if __name__ == '__main__':
    nexps = 25
    mut_str = 'mut_data*'
    base_folder = 'mu_scan_12_5_21_noepi'
    skip = 1
    data_folders = sorted(glob.glob('%s/L1e3*' % base_folder))[::skip]
    mu_vals = np.array([float(folder.split('mu')[-1].split('_')[0]) \
            for folder in data_folders])
    inner_folder = 'replicate'
    output_folder = '%s/figs' % base_folder
    file_str = 'fix_prob_vary_mu_full_prediction.pickle'
    # file_str = 'fix_prob_vary_mu_full_prediction_m1e-8.pickle'
    n_time_points = 15
    file_indices = np.arange(5001)
    bin_indices = np.arange(1, 10)
    L = int(1e3)
    savefig = True
    load_data = True

    ## colorschemes
    cmap = cubehelix_palette(n_colors=len(mu_vals), start=.5, rot=-.75, 
                             light=.85, dark=.1, hue=1, gamma=.95)

    bin_edges, _ = load_bins([data_folders[0]], inner_folder, nexps)

    plot_fix_probs_over_time(data_folders, mu_vals, mut_str, inner_folder, 
                             L, n_time_points, file_indices, bin_edges, bin_indices, 
                             nexps, load_data, output_folder, file_str, savefig)

    plt.show()
