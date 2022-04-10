"""
Adapted from SALib plotting module
"""

import pandas as pd

# magic string indicating DF columns holding conf bound values
CONF_COLUMN = "_conf"


def sobol_plot(Si, ax=None):
    """Create bar chart of results.

    Parameters
    ----------
    * Si_df: pd.DataFrame, of sensitivity results

    Returns
    ----------
    * ax : matplotlib axes object
    """
    if len(Si.to_df()) == 2:
        total, first = Si.to_df()
    else:
        total, first, _ = Si.to_df()
    Si_df = pd.concat([total, first], axis=1)

    conf_cols = Si_df.columns.str.contains(CONF_COLUMN)

    confs = Si_df.loc[:, conf_cols]
    confs.columns = [c.replace(CONF_COLUMN, "") for c in confs.columns]
    confs = confs.rename(columns={"S1": "first-order", "ST": "total-order"})

    Sis = Si_df.loc[:, ~conf_cols]
    Sis = Sis.rename(columns={"S1": "first-order", "ST": "total-order"})

    ax = Sis.plot(kind="bar", yerr=confs, ax=ax, ylabel="Sobol' Index")
    return ax
