"""
Nicholas M. Boffi

This file contains code to visualize the distribution of fitness increments
for the dominant strain as a function of \mu, demonstrating that
there is no fixed fitness-parameterized mapping.
"""


import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

import typing
from typing import Tuple

from common import*

from scipy.stats import gaussian_kde
import seaborn as sns
from seaborn import cubehelix_palette
import pandas as pd

import pickle

sns.set_style("white")
mpl.rcParams['axes.grid']  = True
mpl.rcParams['axes.grid.which']  = 'both'
mpl.rcParams['xtick.minor.visible']  = True
mpl.rcParams['ytick.minor.visible']  = True
mpl.rcParams['xtick.minor.visible']  = True
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['grid.color'] = '0.8'
mpl.rcParams['grid.alpha'] = '0.35'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['figure.titlesize'] = 30
mpl.rcParams['font.size'] = 27.5
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.dpi'] = 125


def make_ridge_plot(
    increments: list,
    Fls: list,
    Fus: list
):
    Favgs = 0.5*(np.array(Fls) + np.array(Fus))
    all_increments = np.array([])
    F_labels = np.array([])
    mu_labels = np.array([])
    for mu_index, increment_list in enumerate(increments):
        added_incs = 0
        for Fav, F_incs in zip(Favgs, increment_list):
            F_labels = np.concatenate(
                    (F_labels, np.array([Fav]*F_incs.size))
                )
            all_increments = np.concatenate((all_increments, F_incs))
            added_incs += F_incs.size

        mu_labels = np.concatenate(
                (mu_labels, np.array([mu_vals[mu_index]]*added_incs))
            )

    df = pd.DataFrame(dict(F=F_labels, dF=all_increments, mu=mu_labels))
    g = sns.FacetGrid(df, col='F', hue='mu', palette=pal)
    g.map(sns.kdeplot, "dF",
          bw_adjust=1.5, clip_on=False,
          fill=True, alpha=0.4, linewidth=1.5)
    g.map(sns.kdeplot, "dF",
          bw_adjust=1.5, clip_on=False,
          color='w', linewidth=2)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, "dF")
    g.refline(y=0, linewidth=2, linestyle='-', color=None, clip_on=False)
    # g.figure.subplots_adjust(hspace=-.1, wspace=-.1)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    plt.show()


def plot_kdes(
    kdes: list,
    min_df: float,
    max_df: float,
):
    npts = 1000
    pts = np.linspace(min_df, max_df, npts)
    fig, ax = plt.subplots()
    max_val = 0
    for kk, kde in enumerate(kdes):
        power = int(np.log(mu_vals[kk]) / np.log(10) - 0.5)
        label_str = r"$\mu=10^{%d}$" % power
        kde_vals = kde(pts)
        max_val = max(0, np.max(kde_vals))
        plt.plot(pts, kde_vals, color=cmap[kk+1], lw=2.5, label=label_str)

    plt.xlabel(r'$\Delta F$')
    plt.ylabel(r"$\rho_\mu(\Delta F | F)$")
    plt.xlim([min_df - 1.5e-3, max_df])
    plt.ylim([0.0, 1.10*max_val])
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(f"{output_folder}/dists_vary_mu.pdf",
            dpi=300, transparent=True, bbox_inches='tight')

    plt.show()


def load_increment_data(
    data_folders: list,
    inner_folder: str,
    nexps: int,
    inc_str: str,
    bac_str: str,
    Fls: list,
    Fus: list 
):
    """Find and load the increment data."""
    increments = []
    n_mus = len(data_folders)
    dist_indices = np.zeros((n_mus, nexps))
    fit_vals = np.zeros((n_mus, nexps))


    # loop over all values of \mu
    for mu_index, data_folder in enumerate(data_folders):
        # load files
        inc_files = get_files([data_folder], inner_folder, inc_str, nexps)
        bac_files = get_files([data_folder], inner_folder, bac_str, nexps)


        # extract dominant fitness data
        ntimes = max([len(bac_files_list) for bac_files_list in bac_files])
        _, _, dominant_fits = get_fit_data(nexps, 1, ntimes, bac_files)
        dominant_fits = np.squeeze(dominant_fits)


        # normalize fitness
        scales = dominant_fits[:, -1] - dominant_fits[:, 0]
        dominant_fits /= scales[:, None]
        shifts = 1 - dominant_fits[:, 0]
        dominant_fits += shifts[:, None]


        # find indices for when fitness is in range
        increments.append([])
        for Find, (Fl, Fu) in enumerate(zip(Fls, Fus)):
            increments[mu_index].append(np.array([]))
            for replicate, fit_traj in enumerate(dominant_fits):
                dist_index = np.nonzero(fit_traj >= Fl)[0][0]
                dist_indices[mu_index, replicate] = dist_index
                fit_vals[mu_index, replicate] = fit_traj[dist_index]
                print(Fl, fit_vals[mu_index, replicate], Fu)
                assert(fit_vals[mu_index, replicate] <= Fu)
                curr_increments = np.fromfile(
                        inc_files[replicate][dist_index], dtype=np.float64
                ) / scales[replicate]
                increments[mu_index][Find] = np.concatenate(
                        (increments[mu_index][Find], curr_increments)
                )

        print(f'Finished loading increment data on {mu_index+1}/{n_mus}.')


    return increments, dist_indices, fit_vals


def make_plots(
    data_folders: list,
    inner_folder: str,
    nexps: int,
    inc_str: str,
    bac_str: str,
    Fls: list,
    Fus: list,
    min_df: float,
    max_df: float,
    output_folder: str,
    load_inc_data: bool,
    savefig: bool
) -> None:
    """Make the comparison figure."""
    # store the increment data over different values of \mu

    if load_inc_data:
        data_dict = pickle.load(open(f'{output_folder}/inc_data.npy', 'rb'))
        increments = data_dict['increments']
    else:
        increments, dist_indices, fit_vals = \
                load_increment_data(data_folders, inner_folder, nexps, 
                                    inc_str, bac_str, Fl, Fu)
        # data_dict = {
            # 'increments': increments,
            # 'dist_indices': dist_indices,
            # 'fit_vals': fit_vals
        # }
        # pickle.dump(data_dict, open(f'{output_folder}/inc_data.npy', 'wb'))


    kdes = [gaussian_kde(increment_data, bw_method=bw_method) \
            for increment_data in increments]
    plot_kdes(kdes, min_df, max_df)
    # make_ridge_plot(increments, Fls, Fus)


if __name__ == '__main__':
    # plotting flags
    # nexps = 20
    nexps = 15
    # base_folder = 'mu_scan_3_14_22_noepi_same_landscape'
    base_folder = 'mu_scan_3_30_22_epi_same_landscape'
    data_folders = sorted(glob.glob('%s/L1e3*' % base_folder))
    mu_vals = np.sort(
        np.array([float(folder.split('mu')[-1]) for folder in data_folders])
    )
    inner_folder = 'replicate'
    output_folder = f'{base_folder}/figs'
    inc_str = 'inc_dist*'
    bac_str = 'bac_data*'
    Fls = [1.95]
    Fus = [2.0]


    min_df, max_df = 0, 3e-2
    load_inc_data = False
    savefig = True
    bw_method = 1.0


    # colorschemes for plotting
    pal = cubehelix_palette(n_colors=len(mu_vals) + 2, start=.5,
                             rot=-.75, light=.85, dark=.1, hue=1,
                             gamma=.95)


    # do the main computation
    make_plots(data_folders, inner_folder, nexps, inc_str, bac_str, Fls, Fus, 
               min_df, max_df, output_folder, load_inc_data, savefig)
