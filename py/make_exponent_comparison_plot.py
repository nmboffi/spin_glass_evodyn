"""
Nicholas M. Boffi

This file contains code to plot the exponents of fitness relaxation as a function of 
\beta, comparing the clonal interference and strong-selection weak-mutation limit
directly.

It also plots the exponents as a function of $\mu$, comparing epistasis to no epistasis.
"""


import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import typing
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


def make_sswm_clonal_figure():
    fig, ax = plt.subplots()
    for index, label_str in enumerate(['SSWM', 'clonal']):
        input_str = f'data_proc/paper_figures/paper_figure_data/' \
                + f'exponent_data_fitness_vary_beta_{label_str}.pickle'
        data = pickle.load(open(input_str, 'rb'))
        medians = np.median(data['exponents'], axis=0)
        lbs = np.quantile(data['exponents'], axis=0, q=0.025)
        ubs = np.quantile(data['exponents'], axis=0, q=0.975)
        betas = data['beta_vals']

        print('Exponents for %s: %s' % (label_str, medians))

        plt.plot(betas, medians, color=cmap[index+1], lw=4.0, marker='o', ms=10.0, 
                 label=label_str)
        ax.fill_between(betas, lbs, ubs, alpha=0.3, color=cmap[index+1])

    plt.plot(betas, np.ones_like(betas)*2, ls='--', color='k')
    plt.plot(betas, np.ones_like(betas)*0.5, ls='--', color='k')

    plt.xlabel(r"epistasis parameter $\beta$")
    plt.ylabel('relaxation exponent $c$')
    plt.ylim([0, 3.0])
    plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.tight_layout()
    plt.legend()

    if savefig:
        plt.savefig('%s/clonal_sswm_exponent_comp.pdf' % output_folder,
                    dpi=300, bbox_inches='tight')


def make_epi_noepi_figure():
    fig, ax = plt.subplots()
    for index, label_str in enumerate(['epistasis', 'no epistasis']):
        file_str = 'epi' if label_str == 'epistasis' else 'noepi'
        input_str = f'data_proc/paper_figures/paper_figure_data/' \
                + f'exponent_data_fitness_vary_mu_{file_str}.pickle'
        data = pickle.load(open(input_str, 'rb'))
        medians = np.median(data['exponents'], axis=0)
        lbs = np.quantile(data['exponents'], axis=0, q=0.025)
        ubs = np.quantile(data['exponents'], axis=0, q=0.975)
        mus = data['mu_vals']

        print('Exponents for %s: %s' % (label_str, medians))

        plt.plot(mus, medians, color=cmap[index+1], lw=4.0, marker='o', ms=10.0, 
                 label=label_str)
        ax.fill_between(mus, lbs, ubs, alpha=0.3, color=cmap[index+1])

    plt.xlabel(r"mutation rate $\mu$")
    plt.ylabel('relaxation exponent $c$')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.legend()

    if savefig:
        plt.savefig('%s/epi_noepi_exponent_comp.pdf' % output_folder,
                    dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    savefig = True
    output_folder = 'data_proc/paper_figures/paper_figure_data'
    cmap = cubehelix_palette(n_colors=4, start=.5, rot=-.75, light=.85,
                             dark=.1, hue=1, gamma=.95, reverse=True)

    make_sswm_clonal_figure()
    make_epi_noepi_figure()

    plt.show()
