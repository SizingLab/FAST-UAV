"""
Plots for life cycle assessments interpretation
"""
from fastoad.io import VariableIO
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fastuav.constants import RESULTS_VARIABLE_KEY, MODEL_KEY, USER_DB
import lca_algebraic as lcalg
import brightway2 as bw
from sympy.parsing.sympy_parser import parse_expr
from pyvis.network import Network


def lca_sun_plot(file_path: str, file_formatter=None):
    """
    Returns sunburst figures with the contributions of each activity and for each impact.
    """
    # file containing variables and their values
    variables = VariableIO(file_path, file_formatter).read()

    # look for method names
    methods = []
    for variable in variables.names():
        if RESULTS_VARIABLE_KEY not in variable:
            continue
        method_name = variable.split(RESULTS_VARIABLE_KEY)[-1].split(":" + MODEL_KEY)[0]
        methods.append(method_name)
    methods = set(methods)  # keep only unique method names

    fig = make_subplots(rows=1,
                        cols=len(methods),
                        specs=[[{"type": "sunburst"} for i in range(len(methods))]],
                        subplot_titles=list(m.split(':')[-1].replace("_", " ") for m in methods))
    for i, method in enumerate(methods):
        labels = []
        parents = []
        values = []
        for variable in variables.names():
            if RESULTS_VARIABLE_KEY not in variable or method not in variable:
                continue
            end = variable.find(":" + MODEL_KEY)
            full_name = variable[end:]
            name_split = full_name.split(":")
            name = name_split[-1]
            parent_name = name_split[-2]
            value = variables[variable].value[0]
            labels.append(name)
            parents.append(parent_name)
            values.append(value)

        trace = go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
        )
        fig.add_trace(trace, row=1, col=i + 1)

    # layout and figure production
    fig.update_layout(margin=dict(t=80.0, l=0, r=0, b=0))

    return fig


def lca_bar_plot(file_path: str, file_formatter=None):
    """
    Returns a bar plot with the contributions of each activity and for each impact.
    """
    # file containing variables and their values
    variables = VariableIO(file_path, file_formatter).read()

    # look for method names
    methods = []
    for variable in variables.names():
        if RESULTS_VARIABLE_KEY not in variable:
            continue
        method_name = variable.split(RESULTS_VARIABLE_KEY)[-1].split(":" + MODEL_KEY)[0]
        methods.append(method_name)
    methods = set(methods)  # keep only unique method names

    df = pd.DataFrame()
    for i, method in enumerate(methods):
        scores_dict = dict()
        for variable in variables.names():
            if RESULTS_VARIABLE_KEY not in variable or method not in variable:
                continue
            end = variable.find(":" + MODEL_KEY)
            full_name = variable[end:]
            # name_split = full_name.split(":")
            # name = name_split[-1]
            value = variables[variable].value[0]
            scores_dict[full_name] = [value]

        df2 = pd.DataFrame(scores_dict, index=[method.split(':')[-1].replace("_", " ")]).transpose()
        df = pd.concat([df, df2], axis=1, ignore_index=False)

    # keep only elementary activities
    for full_name in df.index.values:
        name_split = full_name.rsplit(":", 1)  # split name
        parent = name_split[0]
        name = name_split[-1]
        if parent in df.index.values:
            df = df.drop(parent)

    # plot bar chart
    df = df / df.sum()  # normalize impacts (sum for each impact = 1)
    data = []
    x = df.columns.values
    for name in df.index.values:
        row = df.loc[df.index == name]
        y = [row[method][0] for method in x]

        trace = go.Bar(
            name=name.split(':')[-1],  # short name
            x=x,
            y=y,
        )
        data.append(trace)

    fig = go.Figure(data=data)

    # layout and figure production
    fig.update_layout(barmode='stack',
                      title="Life Cycle Impact Assessment Results",
                      yaxis=dict(tickformat=".0%", title='Normalized Score'),
                      margin=dict(t=80.0, l=0, r=0, b=0),
                      xaxis_tickangle=0)

    return fig


def LCAMonteCarlo(model, methods, n_runs, **params):
    """
    Does Monte Carlo simulations to assess uncertainty on the impact categories.
    Input uncertainties are embedded in EcoInvent activities.
    Parameters used in the parametric study are frozen.
    """

    if not isinstance(methods, list):
        methods = [methods]

    # Freeze params
    db = model[0]  # get database in which model is defined
    if lcalg.helpers._isForeground(db):
        lcalg.freezeParams(db, **params)  # freeze parameters

    # Monte Carlo for each impact category with vanilla brightway
    scores_array = []
    for method in methods:
        mc = bw.MonteCarloLCA({model: 1}, method)  # MC on inventory uncertainties (background db)
        scores = [next(mc) for _ in range(n_runs)]
        scores_array.append(scores)

    return scores_array


def recursive_activities(act):
    """Traverse tree of sub-activities of a given activity, until background database is reached."""
    activities = []
    units = []
    parents = []
    exchanges = []
    levels = []

    def _recursive_activities(act, activities, units, parents, exchanges, levels, parent: str = "", exc: dict = {},
                              level: int = 0):
        # if act.as_dict()['database'] != USER_DB: # uncomment here if you only want foreground activities
        #    return
        name = act.as_dict()['name']
        unit = act.as_dict()['unit']
        loc = act.as_dict()['location']
        if loc != 'GLO':
            name += f'\n({loc})'
        exchange = _getAmountOrFormula(exc)
        activities.append(name)
        units.append(unit)
        parents.append(parent)
        exchanges.append(exchange)
        levels.append(level)
        if act.as_dict()['database'] != USER_DB:  # we stop AFTER reaching the first level of background activities
            return
        for exc in act.technosphere():
            _recursive_activities(exc.input, activities, units, parents, exchanges, levels, parent=name, exc=exc,
                                  level=level + 1)
        return

    def _getAmountOrFormula(ex):
        """ Return either a fixed float value or an expression for the amount of this exchange"""
        if 'formula' in ex:
            return parse_expr(ex['formula'])
        elif 'amount' in ex:
            return ex['amount']
        return ""

    _recursive_activities(act, activities, units, parents, exchanges, levels)
    data = {'activity': activities, 'unit': units, 'parent': parents, 'exchange': exchanges, 'level': levels}
    df = pd.DataFrame(data, index=activities)
    return df


def graph_activities():
    """
    Plots an interactive tree to visualize the activities and exchanges declared in the LCA module.
    """
    model = lcalg.getActByCode(USER_DB, MODEL_KEY)
    df = recursive_activities(model)

    net = Network(notebook=True, directed=True, layout=True)

    activities = df['activity']
    parents = df['parent']
    amounts = df['exchange']
    levels = df['level']

    edge_data = zip(activities, parents, amounts, levels)

    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]
        n = e[3]

        if dst == "":
            continue

        net.add_node(src, src, title=src, level=n + 1, group=n + 1, shape='box')
        net.add_node(dst, dst, title=dst, level=n, group=n, shape='box')
        net.add_edge(src, dst, label=str(w))

    net.set_edge_smooth('vertical')
    net.toggle_physics(False)
    return net
