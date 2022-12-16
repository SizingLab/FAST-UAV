"""
Adapted from SALib plotting module
"""

import matplotlib.pyplot as plt
import numpy as np


def _sort_Si(Si, key, sortby="mu_star"):
    return np.array([Si[key][x] for x in np.argsort(Si[sortby])])


def _sort_Si_by_index(Si, key, index):
    return np.array([Si[key][x] for x in index])


def covariance_plot(ax, Si, unit="", legend=None, opts=None):
    """
    Plots mu* against sigma or the 95% confidence interval
    """

    if opts is None:
        opts = {}

    if Si["sigma"] is not None:
        out = []
        x = Si["mu_star"]
        y = Si["sigma"]
        # c = np.random.rand(len(y))
        # c = plt.cm.get_cmap('hsv', len(y))
        # m = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        m = ["d", "v", "s", "*", "^", "d", "v", "s", "*", "^", "p"]
        m = np.resize(m, len(y))

        for xp, yp, mp in zip(x, y, m):
            outp = ax.scatter(xp, yp, marker=mp)
            out.append(outp)
        # out = ax.scatter(Si['mu_star'], y, c=c,
        #                 **opts)
        # out = mscatter(x, y, c=c, m=m, ax=ax, **opts)
        ax.set_ylabel(r"$\sigma$ " + unit)

        ax.set_xlim(
            0,
        )
        ax.set_ylim(
            0,
        )

        x_axis_bounds = np.array(ax.get_xlim())

        (line1,) = ax.plot(x_axis_bounds, x_axis_bounds, "k-")
        (line2,) = ax.plot(x_axis_bounds, 0.5 * x_axis_bounds, "k--")
        (line3,) = ax.plot(x_axis_bounds, 0.1 * x_axis_bounds, "k-.")

        legend_0 = [
            r"$\sigma / \mu^{\star} = 1.0$",
            r"$\sigma / \mu^{\star} = 0.5$",
            r"$\sigma / \mu^{\star} = 0.1$",
        ]
        legend = Si["names"]

        if legend is not None:
            # ax.legend([line1, line2, line3] + out.legend_elements()[0], legend_0 + legend)
            ax.legend(
                [line1, line2, line3] + out,
                legend_0 + legend,
                loc="upper right",
                bbox_to_anchor=(1.0, -0.12),
                fancybox=True,
                shadow=True,
                ncol=3,
            )
        else:
            ax.legend((line1, line2, line3), legend_0, loc="best")

    else:
        y = Si["mu_star_conf"]
        out = ax.scatter(Si["mu_star"], y, c="k", marker="o", **opts)
        ax.set_ylabel(r"$95\% CI$")

    ax.set_xlabel(r"$\mu^\star$ " + unit)
    ax.set_ylim(
        0 - (0.01 * np.array(ax.get_ylim()[1])),
    )

    return out


def horizontal_bar_plot(ax, Si, opts=None, sortby="mu_star", unit="", legend=None):
    """
    Updates a matplotlib axes instance with a horizontal bar plot
    of mu_star, with error bars representing mu_star_conf.
    """
    assert sortby in ["mu_star", "mu_star_conf", "sigma", "mu"]

    if opts is None:
        opts = {}

    # Sort all the plotted elements by mu_star (or optionally another
    # metric)
    names_sorted = _sort_Si(Si, "names", sortby)
    mu_star_sorted = _sort_Si(Si, "mu_star", sortby)
    mu_star_conf_sorted = _sort_Si(Si, "mu_star_conf", sortby)

    # Plot horizontal barchart
    y_pos = np.arange(len(mu_star_sorted))
    plot_names = names_sorted

    out = ax.barh(
        y_pos, mu_star_sorted, xerr=mu_star_conf_sorted, align="center", ecolor="black", **opts
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_names)
    ax.set_xlabel(r"$\mu^\star$" + unit)

    ax.set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    return out


def morris_plot(Si, unit="", opts=None):
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    out_1 = horizontal_bar_plot(ax1, Si, unit=unit)
    out_2 = covariance_plot(ax2, Si, unit=unit)
    return out_1, out_2
