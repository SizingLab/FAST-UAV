"""
Plots for displaying the outputs of a Design of Experiments (e.g., Monte Carlo)
"""

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd
import seaborn as sns


def hist_dist_plot(df_output, y):
    """
    Create histogram plot for visualizing distribution of a variable y.

    Parameters
    ----------
    * df_output: pd.DataFrame, of DoE results

    Returns
    ----------
    * fig : matplotlib fig object
    """

    # Get data and fit normal law
    mu, std = norm.fit(df_output[y])

    # Initialize plot
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 5), gridspec_kw={'height_ratios': [1, 2]})
    fig.tight_layout()

    # Plot the histogram.
    q25, q50, q75 = np.percentile(df_output[y], [25, 50, 75])
    bin_width = 2 * (q75 - q25) * len(df_output[y]) ** (-1 / 3)  # Freedman–Diaconis number of bins
    bins = round((df_output[y].max() - df_output[y].min()) / bin_width)
    n, bins, patches = axes[1].hist(
        df_output[y],
        bins=bins,
        density=True,
        edgecolor='black',
        linewidth=0.5,
        label="output distribution",
    )

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axes[1].plot(x, p, "r", linewidth=3, label="normal distribution $\mu$=%.2f, $\sigma$=%.2f" % (mu, std))

    plt.xlabel(y)
    plt.ylabel("Probability")
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                   ncol=1, fancybox=True, shadow=True)

    # top boxplot
    axes[0].boxplot(df_output[y], 0, "", vert=False)
    axes[0].axvline(q50, ymin=0.1, ymax=0.8, color='red', alpha=.6, linewidth=2)
    axes[0].axis('off')
    plt.subplots_adjust(hspace=0)

    return fig


def DoE_plot(df_output_array, x, y):
    """
    Create a plot for visualizing the results of a DoE, along with their distributions.
    The plot is two-axes, i.e. one can visualize two variables at a time.

    Parameters
    ----------
    * df_output_array: array of pd.DataFrame, of DoE results

    Returns
    ----------
    * g : seaborn JointGrid object
    """

    df_concat = pd.concat([df_output_array[i].assign(dataset='DoE %d' %i) for i in range(len(df_output_array))])

    # Plot results and distributions
    g = sns.jointplot(x=x,
                      y=y,
                      data=df_concat,
                      hue='dataset',
                      s=10,
                      # size=df_concat['dataset'],
                      # sizes=[5, 10, 35],
                      style=df_concat['dataset'],
                      edgecolor=None)

    return g