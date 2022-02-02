"""
Morris Analysis Method
"""
import contextlib
import os
from openmdao_extensions.salib_doe_driver import SalibDOEDriver
import fastoad.api as oad
from fastoad.io.variable_io import DataFile
import openmdao.api as om
import pandas as pd
from ipywidgets import widgets, Layout
import plotly.graph_objects as go
from SALib.analyze import morris
from typing import List
from plotly.validators.scatter.marker import SymbolValidator
import numpy as np


def doe_morris(
    x_dict: dict, y_list: List[str], conf_file: str, nt: int = 1024
) -> pd.DataFrame:
    """
    DoE for Morris Method

    :param x_dict: inputs dictionary {input_name: distribution_law}
    :param y_list: list of outputs
    :param conf_file: configuration file for the problem
    :param nt: number of trajectories to apply morris method

    :return df: dataframe of the monte carlo simulation results
    """

    class SubProbComp(om.ExplicitComponent):
        """
        Sub-problem component for nested optimization to ensure system consistency.
        """

        def initialize(self):
            self.options.declare("conf")
            self.options.declare("x_list")
            self.options.declare("y_list")

        def setup(self):
            # create a sub-problem to use later in the compute
            # sub_conf = oad.FASTOADProblemConfigurator(conf_file)
            conf = self.options["conf"]
            prob = conf.get_problem(read_inputs=True)
            prob.driver.options["disp"] = False
            p = self._prob = prob
            p.setup()

            # set counter for optimization failure
            self._fail_count = 0

            # define the i/o of the component
            x_list = self._x_list = self.options["x_list"]
            y_list = self._y_list = self.options["y_list"]

            for x in x_list:
                self.add_input(x)

            for y in y_list:
                self.add_output(y)

            self.declare_partials("*", "*", method="fd")

        def compute(self, inputs, outputs):
            p = self._prob
            x_list = self._x_list
            y_list = self._y_list

            for x in x_list:
                p[x] = inputs[x]

            with open(os.devnull, "w") as f, contextlib.redirect_stdout(
                f
            ):  # turn off all convergence messages (including failures)
                fail = p.run_driver()

            for y in y_list:
                outputs[y] = p[y]

            if fail:
                self._fail_count += 1

    # Problem configuration
    conf = oad.FASTOADProblemConfigurator(conf_file)
    prob_definition = conf.get_optimization_definition()
    x_list = [x_name for x_name in x_dict.keys()]

    # CASE 1: nested optimization is declared (i.e. optimization problem is defined in configuration file)
    if "objective" in prob_definition.keys():
        nested_optimization = True
        prob = om.Problem()
        prob.model.add_subsystem(
            "sub_prob",
            SubProbComp(conf=conf, x_list=x_list, y_list=y_list),
            promotes=["*"],
        )

    # CASE 2: simple model without optimization
    else:
        nested_optimization = False
        prob = conf.get_problem(read_inputs=True)

    # DoE parameters
    # dists = []
    for x_name, x_value in x_dict.items():
        prob.model.add_design_var(
            x_name, lower=x_value[0], upper=x_value[1]
        )  # add input parameter for DoE
        # dist = x_value[2]  # not used in this version
        # dists.append(dist)

    # Setup driver (Morris method)
    prob.driver = SalibDOEDriver(
        sa_method_name="Morris",
        sa_doe_options={"n_trajs": nt},
    )

    # Attach recorder to the driver
    if os.path.exists("cases.sql"):
        os.remove("cases.sql")
    prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
    prob.driver.recording_options["includes"] = (
        x_list + y_list
    )  # include all variables from the problem

    # Run problem
    prob.setup()
    prob.run_driver()
    prob.cleanup()

    # Get results from recorded cases
    df = pd.DataFrame()
    cr = om.CaseReader("cases.sql")
    cases = cr.list_cases("driver", out_stream=None)
    for case in cases:
        values = cr.get_case(case).outputs
        df = df.append(values, ignore_index=True)

    for i in df.columns:
        df[i] = df[i].apply(lambda x: x[0])

    # Print number of optimization failures
    fail_count = (
        prob.model.sub_prob._fail_count if nested_optimization else 0
    )  # count number of failures
    if fail_count > 0:
        print("%d out of %d optimizations failed." % (fail_count, len(cases)))

    # save to .csv for future use
    df.to_csv("workdir/morris/doe.csv")

    return df


def morris_analysis(conf_file, data_file):
    """
    Interactive interface to define and simulate a Morris sensitivity analysis with Multiple Inputs and a Single Output.
    Plots the Morris results.

    :param conf_file: configuration file of the problem
    :param data_file: output file of the initial design problem, to set up initial values.
    """

    # Get variables data from file
    variables = DataFile(data_file)
    variables.sort(key=lambda var: var.name)
    table = variables.to_dataframe()[
        ["name", "val", "units", "is_input", "desc"]
    ].rename(
        columns={"name": "Name", "val": "Value", "units": "Unit", "desc": "Description"}
    )
    # Uncertain variables table
    x_table = table.loc[table["is_input"]]  # select inputs only
    x_table = x_table.loc[
        x_table["Name"].str.contains("uncertainty:")
    ]  # select uncertain parameters only
    x_list_short = x_table["Name"].apply(get_short_name).unique().tolist()  # short name
    x_list_short.append("")  # add empty name for variable un-selection

    # Outputs table
    y_list = table.loc[~table["is_input"]]["Name"].unique().tolist()

    def input_box():
        # Input box
        inputbox = widgets.Dropdown(
            description="Uncertain parameter:   ",
            options=x_list_short,
            value=None,
            style={"description_width": "initial"},
        )
        # Values boxes
        value_box = widgets.Text(
            value="", description="", continuous_update=False, disabled=True
        )
        var_box = widgets.FloatRangeSlider(
            value=[-0.1, 0.1],
            min=-0.5,
            max=0.5,
            step=0.01,
            description="error interval:",
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format=".0%",
        )
        return widgets.HBox([inputbox, value_box, var_box])

    # "add input" button
    addinput_button = widgets.Button(
        description="add parameter", style={"description_width": "initial"}
    )

    # Number of samples
    samples = widgets.IntSlider(
        value=4,
        min=4,
        max=64,
        step=1,
        description="Trajectories:",
        continuous_update=False,
    )

    # Output box
    outputbox = widgets.Dropdown(
        description="Output of interest:   ",
        options=y_list,
        value=None,
        style={"description_width": "initial"},
    )

    # "Update" button
    update_button = widgets.Button(description="run Morris")

    # Assign empty figures
    # Bar plot
    fig1 = go.FigureWidget(
        layout=go.Layout(
            title=dict(text="Morris results"), yaxis=dict(title=r"$\mu^*$")
        )
    )
    fig1.add_trace(
        go.Bar(name="mu_star", x=[], y=[], error_y=dict(type="data", array=[]))
    )

    # Scatter plot
    fig2 = go.FigureWidget(
        layout=go.Layout(xaxis=dict(title=r"$\mu^*$"), yaxis=dict(title=r"$\sigma$"))
    )
    fig2.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers+text",
            name="parameters",
            text=[],
            textposition="top center",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[0, 1.0],
            y=[0, 1.0],
            mode="lines",
            line=dict(color="black"),
            name=r"$\sigma/\mu^* = 1.0$",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[0, 2.0],
            y=[0, 1.0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name=r"$\sigma/\mu^* = 0.5$",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[0, 10.0],
            y=[0, 1.0],
            mode="lines",
            line=dict(color="black", dash="dot"),
            name=r"$\sigma/\mu^* = 0.1$",
        )
    )

    def add_input(change):
        # add new input row
        new_input = input_box()
        inputs_array.append(new_input)
        widg.children = widg.children[:-1] + (new_input, addinput_button)
        n_input = len(inputs_array) - 1  # input row indice

        def variable_data(change):
            inputbox = inputs_array[n_input].children[
                0
            ]  # variable selected from dropdown
            value_box = inputs_array[n_input].children[1]  # value of the variable
            var_box = inputs_array[n_input].children[
                2
            ]  # variation to apply for the DoE
            x_data = table.loc[
                table["Name"] == get_long_name(inputbox.value)[0]
            ]  # corresponding data from file
            if x_data["Value"].unique().size != 0:  # check data exists
                x_value = x_data["Value"].unique()[0]  # get value
                x_unit = x_data["Unit"].unique()[0]
                value_box.value = "{:10.3f} ".format(x_value) + (
                    x_unit if x_unit is not None else ""
                )
                var_box.value = [-0.1, 0.1]
            else:
                value_box.value = ""
                return False

        # add an observe event to update value according to selected variable
        new_input.children[0].observe(variable_data, names="value")

    def validate(outputbox, x_dict):
        if outputbox.value is None or len(x_dict) == 0:
            return False
        else:
            return True

    def morris_analysis(df, x_dict, y):
        """
        Perform Sobol' analysis on model outputs.
        """

        num_vars = len(x_dict)

        # Screening method of Morris
        problem_morris = {
            "num_vars": num_vars,
            "names": list(x_dict.keys()),
            "bounds": list(x_dict.values()),
        }
        X_morris = df[list(x_dict.keys())].to_numpy()
        Y_morris = df[y].to_numpy()
        Si = morris.analyze(
            problem_morris, X_morris, Y_morris, conf_level=0.95, num_resamples=100
        )

        return Si

    def update_morris(change):
        """
        Based on the simulations,
        perform a Morris analysis and update charts according to the selected output.
        """

        # Perform Morris analysis
        y = outputbox.value  # name of output variable of interest
        Si = morris_analysis(df, x_dict, y)

        # Update figures
        y_data = table.loc[table["Name"] == y]  # corresponding data from file
        y_unit = y_data["Unit"].unique()[0]  # get unit

        with fig1.batch_update():
            fig1.data[0].x = list(get_short_name(x) for x in x_dict.keys())
            fig1.data[0].y = Si["mu_star"]
            fig1.data[0].error_y = dict(type="data", array=Si["mu_star_conf"])
            fig1.update_layout(
                yaxis=dict(
                    title="$\\mu^* \\text{ ("
                    + (y_unit if y_unit is not None else "")
                    + ")}$"
                ),
                xaxis={"categoryorder": "total descending"},
            )

        with fig2.batch_update():
            # data
            fig2.data[0].x = Si["mu_star"]
            fig2.data[0].y = Si["sigma"]
            fig2.data[0].text = list(get_short_name(x) for x in x_dict.keys())

            raw_symbols = SymbolValidator().values
            symbols = [raw_symbols[i] for i in range(0, 12 * len(x_dict.keys()), 12)]
            fig2.data[0].marker = dict(
                symbol=symbols, color=np.sqrt(Si["mu_star"] ** 2 + Si["sigma"] ** 2)
            )

            # reference axes
            fig2.data[1].x = [0, max(Si["mu_star"])]
            fig2.data[1].y = [0, max(Si["mu_star"])]
            fig2.data[2].x = [0, 2 * max(Si["mu_star"])]
            fig2.data[2].y = [0, max(Si["mu_star"])]
            fig2.data[3].x = [0, 10 * max(Si["mu_star"])]
            fig2.data[3].y = [0, max(Si["mu_star"])]

            # scale
            fig2.update_layout(
                xaxis=dict(
                    title="$\\mu^* \\text{ ("
                    + (y_unit if y_unit is not None else "")
                    + ")}$"
                ),
                xaxis_range=[-0.05, 1.2 * max(Si["mu_star"])],
                yaxis_range=[
                    -0.05,
                    1.2 * max(0.1 * max(Si["mu_star"]), max(Si["sigma"])),
                ],
            )

            # export
            fig1.write_html("workdir/morris/results_1.html")
            fig1.write_image("workdir/morris/results_1.pdf")
            fig2.write_html("workdir/morris/results_2.html")
            fig2.write_image("workdir/morris/results_2.pdf")

    def update_all(change):
        """
        Run DoE, Morris analysis and update figures
        """

        # global variables
        global x_dict, df  # needed to modify global copies of x_dict and df

        # Get user inputs
        x_dict = {}
        for inputrow in inputs_array:
            inputbox = inputrow.children[0]
            var_box = inputrow.children[2]
            if inputbox.value is not None:  # not empty input
                a = var_box.value[0]
                b = var_box.value[1]
                bounds = [a, b]
                x_dict[
                    get_long_name(inputbox.value)[1]
                ] = bounds  # bounds for parameter error distribution
        if not validate(outputbox, x_dict):
            return False

        # Design of experiments
        nt = int(samples.value)  # number of trajectories
        df = doe_morris(x_dict, y_list, conf_file, nt)

        # Perform Morris analysis and update charts
        outputbox.observe(
            update_morris, names="value"
        )  # enable to change the output to visualize
        update_morris(0)

    # Set up Figure
    options_panel = widgets.VBox([widgets.HBox([outputbox, samples, update_button])])
    widg = widgets.VBox(
        [widgets.HBox([fig1, fig2]), options_panel, addinput_button],
        layout=Layout(align_items="flex-start"),
    )

    # initialize inputs array and add first input
    inputs_array = []
    add_input(0)

    # add input and update buttons interactions
    addinput_button.on_click(add_input)
    update_button.on_click(
        update_all
    )  # Run the Monte Carlo with provided parameters, and display the charts

    return widg


def get_short_name(long_name):
    """
    Return the short name of a variable.
    The short name is defined as the initial name but without the first and last filters (separated by ':')
    """
    if long_name is None:
        return ""
    short_name = ":".join(long_name.split(":")[1:-1])  # removes first and last filters
    return short_name


def get_long_name(short_name):
    """
    Return the long name of a variable.
    The long name is defined as the short name to which a first filter and last filter are added (with separator ':').
    """
    if short_name is None:
        return "", "", ""
    long_name_1 = "data:" + short_name + ":estimated"
    long_name_2 = "uncertainty:" + short_name + ":rel"
    long_name_3 = "uncertainty:" + short_name + ":abs"
    return long_name_1, long_name_2, long_name_3
