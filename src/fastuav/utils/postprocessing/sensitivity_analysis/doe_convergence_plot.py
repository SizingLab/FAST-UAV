"""
Plots for showing convergence of the design of experiments with increasing numbers of model evaluations.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FixedLocator, PercentFormatter
import numpy as np


def saltelli_eval(x, d, second_order):
    """
    Auxiliary function for converting number of samples to number of model evaluations
    when using Sobol method.

    Parameters
    ----------
    * x: number of samples for Sobol method
    * d: number of uncertain parameters for Saltelli samples
    * second_order: True if second order indices are calculated in Sobol' method.

    Returns
    ----------
    * k: number of model evaluations
    """

    if second_order:
        return x * (2 * d + 2)  # with second order indices calculation
    return x * (d + 2)  # without second order indices calculation


def saltelli_eval_inv(x, d, second_order):
    """
    Auxiliary function for converting number of model evaluations to number of samples
    when using Sobol method.

    Parameters
    ----------
    * x: number of model evaluations
    * d: number of uncertain parameters for Saltelli samples
    * second_order: True if second order indices are calculated in Sobol' method.

    Returns
    ----------
    * k: number of samples for Sobol method
    """

    if second_order:
        return x / (2 * d + 2)  # with second order indices calculation
    return x / (d + 2)  # without second order indices calculation


def mean_convergence(n_array, mu_array, d, second_order: bool = True):
    """
    Create a plot for showing convergence of the output's distribution mean with increasing number of Sobol' samples.

    Parameters
    ----------
    * n_array: array of number of samples for each DoE
    * mu_array: array of resulting means
    * d: number of uncertain parameters for Saltelli samples
    * second_order: True if second order indices are calculated in Sobol' method.

    Returns
    ----------
    * fig, ax : matplotlib fig object
    """

    # Plot total-order indices and confidence intervals as a function of number of samples
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(n_array, mu_array, 'r-x', linewidth=2, label='mean')

    secax = ax.secondary_xaxis(-.25, functions=(lambda x: saltelli_eval(x, d, second_order),
                                                lambda x: saltelli_eval_inv(x, d, second_order)))
    ax.set_xlabel('Number of samples (-)')
    secax.set_xlabel('Number of model evaluations (-)')
    ax.set_ylabel('Mean')
    ax.xaxis.set_major_locator(FixedLocator(np.logspace(np.log2(n_array[0]),
                                                        np.log2(n_array[-1]),
                                                        base=2,
                                                        num=len(n_array))))
    ax.tick_params(axis="x", labelrotation=-30)
    secax.xaxis.set_major_locator(FixedLocator(saltelli_eval(
        np.logspace(np.log2(n_array[0]),
                    np.log2(n_array[-1]),
                    base=2,
                    num=len(n_array)),
        d,
        second_order)))
    secax.tick_params(axis="x", labelrotation=-30)
    plt.grid(alpha=0.8)

    return fig, ax


def std_convergence(n_array, std_array, d, second_order: bool = True):
    """
    Create a plot for showing convergence of the output's distribution standard deviation
    with increasing number of Sobol' samples.

    Parameters
    ----------
    * n_array: array of number of samples for each DoE
    * std_array: array of resulting means
    * d: number of uncertain parameters for Saltelli samples
    * second_order: True if second order indices are calculated in Sobol' method.

    Returns
    ----------
    * fig, ax : matplotlib fig object
    """

    # Plot total-order indices and confidence intervals as a function of number of samples
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(n_array, std_array, 'g-x', linewidth=2, label='standard deviation')

    secax = ax.secondary_xaxis(-.25, functions=(lambda x: saltelli_eval(x, d, second_order),
                                                lambda x: saltelli_eval_inv(x, d, second_order)))
    ax.set_xlabel('Number of samples (-)')
    secax.set_xlabel('Number of model evaluations (-)')
    ax.set_ylabel('Standard deviation')
    ax.xaxis.set_major_locator(FixedLocator(np.logspace(np.log2(n_array[0]),
                                                        np.log2(n_array[-1]),
                                                        base=2,
                                                        num=len(n_array))))
    ax.tick_params(axis="x", labelrotation=-30)
    secax.xaxis.set_major_locator(FixedLocator(saltelli_eval(
        np.logspace(np.log2(n_array[0]),
                    np.log2(n_array[-1]),
                    base=2,
                    num=len(n_array)),
        d,
        second_order)))
    secax.tick_params(axis="x", labelrotation=-30)
    plt.grid(alpha=0.8)

    return fig, ax


def sobol_index_convergence(n_array, ST_array, ST_conf_array, d, second_order: bool = True):
    """
    Create a plot for showing convergence of Sobol total-order index for a given variable.

    Parameters
    ----------
    * n_array: array of number of samples for each DoE
    * ST_array: array of Sobol indices for each DoE
    * ST_array: array of 95% confidence intervals for Sobol indices for each DoE
    * d: number of uncertain parameters for Saltelli samples
    * second_order: True if second order indices are calculated in Sobol' method.

    Returns
    ----------
    * fig, ax : matplotlib fig object
    """

    # Plot total-order indices and confidence intervals as a function of number of samples
    fig = plt.figure()

    # Sobol' index
    ax = fig.add_subplot(111)
    ax.plot(n_array, ST_array, '-x', linewidth=2,  label=r'Total-order index')
    ax.fill_between(n_array,
                    ST_array-ST_conf_array,
                    ST_array+ST_conf_array,
                    alpha=.2,
                    label='95% confidence interval')

    secax = ax.secondary_xaxis(-.25, functions=(lambda x: saltelli_eval(x, d, second_order),
                                                lambda x: saltelli_eval_inv(x, d, second_order)))

    ax.set_xlabel('Number of samples (-)')
    secax.set_xlabel('Number of model evaluations (-)')
    ax.set_ylabel('Total-order index (-)')
    ax.xaxis.set_major_locator(FixedLocator(np.logspace(np.log2(n_array[0]),
                                                        np.log2(n_array[-1]),
                                                        base=2,
                                                        num=len(n_array))))
    ax.tick_params(axis="x", labelrotation=-30)
    secax.xaxis.set_major_locator(FixedLocator(saltelli_eval(
        np.logspace(np.log2(n_array[0]),
                    np.log2(n_array[-1]),
                    base=2,
                    num=len(n_array)),
        d,
        second_order)))
    secax.tick_params(axis="x", labelrotation=-30)
    # ax.yaxis.set_ticks(np.arange(0.1, 1.0, 0.1))
    fig.legend(loc='upper left', bbox_to_anchor=(0.18, 0.85), ncol=1, fancybox=True, shadow=True)
    plt.grid(alpha=0.8)

    return fig, ax


def sobol_index_error(n_array, ST_array, ST_conf_array, d, second_order: bool = True):
    """
    Create a plot for showing convergence of Sobol total-order index for a given variable.

    Parameters
    ----------
    * n_array: array of number of samples for each DoE
    * ST_array: array of Sobol indices for each DoE
    * ST_array: array of 95% confidence intervals for Sobol indices for each DoE
    * d: number of uncertain parameters for Saltelli samples
    * second_order: True if second order indices are calculated in Sobol' method.

    Returns
    ----------
    * fig, ax : matplotlib fig object
    """

    # Plot total-order indices and confidence intervals as a function of number of samples
    fig = plt.figure()

    # Sobol' index
    ax = fig.add_subplot(111)
    ax.plot(n_array, ST_conf_array / ST_array * 100, '-x', linewidth=2, label='Margin of error (95% confidence level)')
    ax.axhline(y=10, color='r', linestyle=":", label='10% criterion')  # 10% threshold

    # Encircle 1024 samples / 16384 simulations point
    # rect = patches.Rectangle((1008, 5.5), 30, 4, linewidth=2, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)

    secax = ax.secondary_xaxis(-.25, functions=(lambda x: saltelli_eval(x, d, second_order),
                                                lambda x: saltelli_eval_inv(x, d, second_order)))

    ax.set_xlabel('Number of samples (-)')
    secax.set_xlabel('Number of model evaluations (-)')
    ax.set_ylabel('Margin of error (%)')
    ax.xaxis.set_major_locator(FixedLocator(np.logspace(np.log2(n_array[0]),
                                                        np.log2(n_array[-1]),
                                                        base=2,
                                                        num=len(n_array))))
    ax.tick_params(axis="x", labelrotation=-30)
    secax.xaxis.set_major_locator(FixedLocator(saltelli_eval(
        np.logspace(np.log2(n_array[0]),
                    np.log2(n_array[-1]),
                    base=2,
                    num=len(n_array)),
        d,
        second_order)))
    secax.tick_params(axis="x", labelrotation=-30)
    xticks = PercentFormatter()
    ax.yaxis.set_major_formatter(xticks)
    fig.legend(loc='upper left', bbox_to_anchor=(0.18, 0.85), ncol=1, fancybox=True, shadow=True)
    plt.grid(alpha=0.8)

    return fig, ax

