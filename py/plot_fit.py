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
from scipy.stats.distributions import t
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
mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['figure.titlesize'] = 30
mpl.rcParams['font.size'] = 32
mpl.rcParams['legend.fontsize'] = 25
mpl.rcParams['figure.dpi'] = 125


def log_traj(t, a, b):
    """ Logarithmic fitness trajectory. """
    return 1 + b*np.log(1 + a*t)


def log_jac(t, a, b):
    return np.array([t*b/(1+a*t), np.log(1 + a*t)]).T


def pow_traj(t, a, c):
    """ Power law fitness trajectory from Lenski's 2013 Science paper. """
    return (a*t + 1)**c


def pow_jac(t, a, c):
    return np.array([np.log(a*t + 1)*(a*t + c)**c, c*t*(a*t+1)**(c-1)]).T


def hyp_traj(t, a, b):
    """ Hyperbolic fitness trajectory from Lenski's 2013 Science paper. """
    return 1 + a*t/(t+b)


def hyp_jac(t, a, b):
    return np.array([t/(t+b), -a*t/(t+b)**2]).T


def asymp_pow_traj_log(t, a, c):
    """ logarithm of 1/t^2 relaxation. """
    return c*np.log(1 + a*t)


def asymp_pow_jac_log(t, a, c):
    """ jacobian for logarithm of 1/t^2 relaxation. """
    return np.array([c*t/(1 + a*t), np.log(1 + a*t)]).T


def asymp_pow_traj(t, a, b, c, d=1):
    """ 1/t^2 relaxation expected for the no-epistatic case. """
    return b + (d-b)/(a*t + 1)**c


def asymp_pow_jac(t, a, b, c, d=1):
    """ Jacobian for the above. """
    return np.array([(b - d)*c*t*(1 + a*t)**(-1-c), 
                     1 - (1 + a*t)**(-c), 
                     (b - d)*(1 + a*t)**(-c)*np.log(1 + a*t)]).T


def make_fitness_plot(
        times: np.array, 
        fit_data: np.array, 
        asymptotic_vals: np.ndarray, 
        skip: int, 
        tf_index: int, 
        nexps: int, 
        n_bootstraps: int, 
        log_plot: bool, 
        log_lin_plot: bool, 
        lin_plot: bool, 
        fig_title: str, 
        save_title: str
) -> None:
    """ Make a single plot of fit_data over time. """
    # clip the data
    times = times[:tf_index]
    fit_data = fit_data[:, :tf_index]

    # plot the individual replicate trajectories in the background.
    fig, ax = plt.subplots()
    for replicate in range(nexps):
        plt.plot(times, fit_data[replicate, :], color=cmap[replicate], 
                 lw=3.0, alpha=.35)

    # plot the replicate mean with error bars.
    replicate_mean = np.mean(fit_data, axis=0)
    replicate_sem = sem(fit_data, axis=0, ddof=1)
    plt.errorbar(times, replicate_mean,  yerr=replicate_sem, ls='', 
                 marker='o', markerfacecolor='none', markersize=5.5,
                 color=cmap[-6], alpha=1.0, errorevery=skip, markevery=skip)


    # bootstrap estimation
    a_vals    = np.zeros(n_bootstraps)
    b_vals    = np.zeros(n_bootstraps)
    exponents = np.zeros(n_bootstraps)
    all_data = fit_data.ravel()
    all_times = times.ravel()
    b_estimate = np.mean(asymptotic_vals)
    print('b_estimate:', b_estimate)
    for curr_estimate in range(n_bootstraps):
            # bootstrap over means
            sample = np.random.randint(nexps, size=nexps)
            traj_est_sem = sem(fit_data[sample, :], axis=0)
            sigma = traj_est_sem[traj_est_sem > 0]
            traj_est_data = \
                np.mean(fit_data[sample, :], axis=0)[traj_est_sem > 0]
            traj_est_times = times[traj_est_sem > 0]
            b_estimate = np.mean(fit_data[sample, -1])


            func_form = asymp_pow_traj
            func_jac = asymp_pow_jac
            p0 = [1e-7, b_estimate, 2.0]
            lbs = [0, .95*b_estimate, 1e-5]
            ubs = [1.0, 1.05*b_estimate, 10.0]

            # perform the fit
            asymp_pow_params, asymp_pow_cov = curve_fit(func_form,
                                                        traj_est_times,
                                                        traj_est_data,
                                                        sigma=sigma,
                                                        jac=func_jac,
                                                        p0=p0,
                                                        absolute_sigma=True,
                                                        bounds=(lbs, ubs),
                                                        tr_solver='exact',
                                                        x_scale='jac',
                                                        method='dogbox',
                                                        loss='linear',
                                                        ftol=2.5e-14,
                                                        xtol=2.5e-14,
                                                        gtol=2.5e-14)

            print(f'Parameters for bootstrap {curr_estimate+1}/{n_bootstraps}: {asymp_pow_params}')

            a_vals[curr_estimate]    = asymp_pow_params[0]
            b_vals[curr_estimate]    = asymp_pow_params[1]
            exponents[curr_estimate] = asymp_pow_params[2]

    ## fit the mean with weighted error.
    asymp_pow_params, cov \
            = curve_fit(asymp_pow_traj, 
                        times[replicate_sem > 0], 
                        replicate_mean[replicate_sem > 0],
                        sigma=replicate_sem[replicate_sem > 0], 
                        absolute_sigma=True, 
                        jac=asymp_pow_jac, 
                        p0=[1e-7, b_estimate, 2.0],  
                        bounds=([0.0, 0.99*b_estimate, 0.0], \
                                [1.0, 1.01*b_estimate, 10.0]),
                        tr_solver='exact', 
                        x_scale='jac', 
                        maxfev=25000, 
                        method='trf', 
                        loss='linear', 
                        ftol=2.5e-16, 
                        xtol=2.5e-16, 
                        gtol=2.5e-16)

    print(f"mean fit parameters: {asymp_pow_params}")
    print('Mean fit exponent with bounds: %g, %g, %g' \
            % (asymp_pow_params[2] - np.sqrt(cov[2, 2]), asymp_pow_params[2], \
            asymp_pow_params[2] + np.sqrt(cov[2, 2])))
    print('Mean bootstrap: %g' % np.mean(exponents))
    print('Median bootstrap: %g' % np.median(exponents))
    print('Quantiles: %g, %g' \
            % (np.quantile(exponents, 0.025), np.quantile(exponents, 0.975)))

    # evaluate and plot the fit
    asymp_a, asymp_b, asymp_c \
            = np.median(a_vals), np.median(b_vals), np.median(exponents)
    print('bounds:', asymp_c - sem(exponents), asymp_c + sem(exponents))
    print("asymptotic: %g, true_asymptotic: %g" \
            % (asymp_b, np.mean(asymptotic_vals)))
    asymp_pow_fit = lambda t: asymp_pow_traj(t, asymp_a, asymp_b, asymp_c)
    asymp_pow_dat = asymp_pow_fit(times)
    asymp_pow_rsq = compute_rsq(replicate_mean, asymp_pow_dat)
    plt.plot(times, asymp_pow_dat, 
             label=r"$F_{\infty} + \frac{1-F_{\infty}}{(at+1)^{c}}$" \
                     +r"$R^2=%0.4f, c=%0.7f$" % (asymp_pow_rsq, asymp_c),
             color=cmap[-2], lw=3, alpha=1.0)

    # axis labels
    plt.xlabel("day")
    plt.ylabel("fitness")
    plt.title(fig_title)
    plt.legend(loc='best', framealpha=0, ncol=1)

    if log_plot:
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()

        if savefig:
            plt.savefig("%s/%s_log.pdf" % (output_folder, save_title), 
                        dpi=300, transparent=True)
    if lin_plot:
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.tight_layout()

        axins = ax.inset_axes([0.35, 0.3, .45, .3]) 
        axins.plot(np.log(1 + asymp_a*times), 
                   np.log(asymp_b - replicate_mean), color=cmap[-6])
        axins.plot(np.log(1 + asymp_a*times),
                   -asymp_c*np.log(1 + asymp_a*times), 
                   color=cmap[-6], linestyle='--')
        axins.set_xlabel(r"$\log(1 + at)$", fontsize=17.5)
        axins.set_ylabel(r"$\log(F_{\infty} - F(t))$", fontsize=17.5)
        axins.tick_params(axis='both', which='both', labelsize=17.5)

        if savefig:
            plt.savefig("%s/%s.pdf" \
                    % (output_folder, save_title), dpi=300, transparent=True)
    if log_lin_plot:
        ax.set_xscale('log')
        ax.set_yscale('linear')
        plt.tight_layout()

        if savefig:
            plt.savefig("%s/%s_log_lin.pdf" \
                    % (output_folder, save_title), dpi=300, transparent=True)

    # exponent histogram of the bootstrap
    plt.figure()
    plt.hist(exponents)
    plt.tight_layout()


def make_plots(
        bac_files: list, 
        nexps: int, 
        n_top_strains: int, 
        savefig: bool, 
        skip: int, 
        data_folder: str, 
        inner_folder: str,
        tf_index: int, 
        n_bootstraps: int, 
        log_plot: bool, load_dat: bool) -> None:
    """ Plot the mean fitness and dominant fitness over time of 
    the dominant strains. Also fit the replicate-avergaed mean fitness 
    and dominant fitnesses to specific functional forms.
    """
    # gather the data.
    ntimes = max([len(bac_files_list) for bac_files_list in bac_files])
    if not load_dat:
        mean_fits, max_fits, dominant_fits \
                = get_fit_data(nexps, n_top_strains, ntimes, bac_files)
        # extract the time of the first output day.
        dt = float(bac_files[0][1].split('.')[-2])  
        times = np.arange(ntimes)*dt

        data_dict = {}
        data_dict['mean_fits'] = mean_fits
        data_dict['times'] = times
        data_dict['max_fits'] = max_fits
        data_dict['dominant_fits'] = dominant_fits
        pickle.dump(data_dict, 
                    open("%s/figs/fit_data.pickle" % data_folder, "wb"))
    else:
        data_dict = pickle.load(
                open("%s/figs/fit_data.pickle" % data_folder, "rb")
                )
        mean_fits = data_dict['mean_fits']
        max_fits = data_dict['max_fits']
        dominant_fits = data_dict['dominant_fits']
        times = data_dict['times']
        print('MEAN FITS:', mean_fits[:, -1])

    if rescale:
        scale = mean_fits[:, -1] - mean_fits[:, 0]
        mean_fits[:, :] /= scale[:, None]
        shift = 1 - mean_fits[:, 0]
        mean_fits[:, :] += shift[:, None]


    print("maximum fitness over anything:", np.max(mean_fits))
    print("final fitness over anything:", mean_fits[:, -1])
    print("minimum fitness over anything:", np.min(mean_fits))
    print("initial fitness over anything:", mean_fits[:, 0])

    print("Making fitness plot for the mean fitness.")
    make_fitness_plot(times, mean_fits, mean_fits[:, -1], 
                      skip, tf_index, nexps, n_bootstraps, 
                      log_plot, log_lin_plot, lin_plot, '', "mean_fits")
    print()


if __name__ == '__main__':
    data_folders = \
            ['/home2/nick/research/lenski/mu_scan_3_20_22_epi_same_landscape_noreset/L1e3_r1e2_b0p25_mu0.0000100000']
    # data_folders \
            # = ['mu_scan_3_14_22_noepi_same_landscape/L1e3_r1e2_b0p0_mu0.0000100000']

    # data_folders \
            # = ['beta_scan_4_14_22_clonal/L1e3_r1e2_m1e-4_b0.000000']

    data_folders \
            = ['mu_scan_3_30_22_epi_same_landscape/L1e3_r1e2_b0p25_mu0.0001000000']

    output_folder = '%s/figs' % data_folders[0]
    inner_folder = 'replicate'
    nexps = 20
    bac_str = 'bac_data*'
    n_top_strains = 1
    L = int(1e3)
    tf_index = -1
    savefig  = False
    log_plot = False
    lin_plot = True
    log_lin_plot = False
    load_dat = False
    rescale = False
    n_bootstraps = 50
    skip = 75

    cmap = cubehelix_palette(n_colors=nexps, start=.5, rot=-.75, light=.85, 
                             dark=.1, hue=1, gamma=.95)
    bac_files = get_files(data_folders, inner_folder, bac_str, nexps)
    make_plots(bac_files, nexps*len(data_folders), n_top_strains, savefig, 
               skip, data_folders[0], inner_folder, tf_index, n_bootstraps, 
               log_plot, load_dat)

    plt.show()
