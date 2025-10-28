"""
Uncertainty and Sensitivity Analysis module.
It allows to investigate uncertainty in the model output
that is generated from uncertainty in the model inputs and model parameters.
Two sensitivity analysis methods are available:
    - The method of Morris,
    - The Sobol' Sensitivity Analysis.
For the Sobol' SA, the uncertain inputs are generated using Saltelli's sampling.
"""


import contextlib
import os
import os.path as pth
from venv import create

from fastuav.utils.drivers.salib_doe_driver import SalibDOEDriver
import fastoad.api as oad
from fastoad.io.variable_io import DataFile
import openmdao.api as om
import pandas as pd
import numpy as np
from ipywidgets import widgets, Layout
import plotly.graph_objects as go
from SALib.analyze import sobol, morris
from typing import List
from plotly.validators.scatter.marker import SymbolValidator
import itertools
# from openmdao_drivers.cmaes_driver import CMAESDriver

SA_PATH = pth.join(
    pth.dirname(pth.abspath(__file__)),
    "..",
    "..",
    "..",
    "notebooks",
    "workdir",
    "sensitivity_analysis",
)


def doe_fast(
    method_name: str,
    x_dict: dict,
    y_list: List[str],
    conf_file: str,
    ns: int = 100,
    custom_driver=None,
    calc_second_order: bool = True,
) -> pd.DataFrame:
    """
    DoE function for FAST-UAV problems.
    Various generators are available:
        - List generator that reads cases from a provided list of DOE cases
        - Uniform generator provided by pyDOE2 and included in OpenMDAO
        - Latin Hypercube generator provided by OpenMDAO
        - Generator for Sobol-Saltelli 2002 method provided by SALib
        - Generator for Morris method provided by SALib
    If an optimization problem is declared in the configuration file,
    a nested optimization (sub-problem) is run (e.g. to ensure system optimality and/or consistency at each simulation).

    :param method_name: 'uniform', 'lhs', 'fullfactorial', 'Sobol' or 'Morris'
    :param x_dict: inputs dictionary {input_name: [dist_parameter_1, dist_parameter_2, distribution_type]}
    :param y_list: list of problem outputs to record
    :param conf_file: configuration file for the problem
    :param ns: number of samples (for Uniform, LHS and Sobol) or trajectories (for Morris)
    :param custom_driver: user-defined OpenMDAO driver if method_name is set to "custom"
    :param calc_second_order: calculate second order indices (Sobol)

    :return: dataframe of the design of experiments results
    """

    class SubProbComp(om.ExplicitComponent):
        """
        Sub-problem component for nested optimization.
        """

        def initialize(self):
            self.options.declare("conf")
            self.options.declare("x_list")
            self.options.declare("y_list")

        def setup(self):
            # create a sub-problem to use later in the compute
            # sub_conf = oad.FASTOADProblemConfigurator(conf_file)
            conf = self.options["conf"]
            prob = conf.get_problem(read_inputs=True)  # get conf file (design variables, objective, driver...)

            # UNCOMMENT THESE LINES IF USING CMA-ES Driver for solving sub-problem
            # TODO: automatically detect use of CMA-ES driver
            # driver = prob.driver = CMAESDriver()
            # driver.CMAOptions['tolfunhist'] = 1e-4
            # driver.CMAOptions['popsize'] = 100

            # prob.driver.options['disp'] = False
            p = self._prob = prob
            p.setup()

            # define the i/o of the component
            x_list = self._x_list = self.options["x_list"]
            y_list = self._y_list = self.options["y_list"]

            for x in x_list:
                self.add_input(x)

            for y in y_list:
                self.add_output(y)

            # set counter and output variable for recording optimization failure or success
            self._fail_count = 0
            self.add_output('optim_failed')

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
            outputs['optim_failed'] = fail

    conf = oad.FASTOADProblemConfigurator(conf_file)
    prob_definition = conf.get_optimization_definition()
    x_list = [x_name for x_name in x_dict.keys()]

    # CASE 1: nested optimization is declared (i.e. optimization problem is defined in configuration file)
    if "objective" in prob_definition.keys():
        nested_optimization = True
        prob = om.Problem()
        prob.model.add_subsystem(
            "sub_prob",
            SubProbComp(
                conf=conf,
                x_list=x_list,
                y_list=y_list,
            ),
            promotes=["*"],
        )

    # CASE 2: simple model without optimization
    else:
        nested_optimization = False
        prob = conf.get_problem(read_inputs=True)

    # Setup driver
    if method_name == "list":
        # add input parameters for DoE
        for x_name, x_value in x_dict.items():
            prob.model.add_design_var(
                x_name, lower=x_value.min(), upper=x_value.max()
            )
        # generate all combinations from values in the dict of parameters
        keys, values = zip(*x_dict.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        case_list = [[(key, val) for key, val in permut_dict.items()] for permut_dict in permutations_dicts]
        prob.driver = om.DOEDriver(
            om.ListGenerator(
                data=case_list
            )
        )
    elif method_name in ("uniform", "lhs", "fullfactorial"):
        # add input parameters for DoE
        for x_name, x_value in x_dict.items():
            prob.model.add_design_var(
                x_name, lower=x_value[0], upper=x_value[1]
            )
        # setup driver
        if method_name == "uniform":
            prob.driver = om.DOEDriver(
                om.UniformGenerator(
                    num_samples=ns
                )
            )
        elif method_name == "lhs":
            prob.driver = om.DOEDriver(
                om.LatinHypercubeGenerator(
                    samples=ns
                )
            )
        elif method_name == "fullfactorial":
            prob.driver = om.DOEDriver(
                om.FullFactorialGenerator(
                    levels=ns
                )
            )
    elif method_name in ("Sobol", "Morris"):
        # add input parameters for DoE
        dists = []
        for x_name, x_value in x_dict.items():
            prob.model.add_design_var(
                x_name, lower=x_value[0], upper=x_value[1]
            )
            dist = x_value[2]  # add distribution type ('unif' or 'norm')
            dists.append(dist)
        # setup driver
        if method_name == "Sobol":
            prob.driver = SalibDOEDriver(
                sa_method_name=method_name,
                sa_doe_options={"n_samples": ns, "calc_second_order": calc_second_order},
                distributions=dists,
            )
        elif method_name == "Morris":
            # setup driver
            prob.driver = SalibDOEDriver(
                sa_method_name="Morris",
                sa_doe_options={"n_trajs": ns},
                distributions=dists,
            )
    elif method_name == "custom":
        # add input parameters for DoE
        for x_name, x_value in x_dict.items():
            prob.model.add_design_var(
                x_name, lower=x_value.min(), upper=x_value.max()
            )
        # setup driver
        prob.driver = custom_driver

    # Attach recorder to the driver
    if os.path.exists("cases.sql"):
        os.remove("cases.sql")
    prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
    recorded_variables = x_list + y_list
    if nested_optimization:
        recorded_variables.append("optim_failed")
    prob.driver.recording_options["includes"] = recorded_variables

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
        # df = df.append(values, ignore_index=True)
        df = pd.concat([df, pd.DataFrame(values)], ignore_index=True)

    # for i in df.columns:
    #     df[i] = df[i].apply(lambda x: x[0])

    # Print number of optimization failures
    fail_count = (
        prob.model.sub_prob._fail_count if nested_optimization else 0
    )  # count number of failures for nested optimization
    if fail_count > 0:
        print("%d out of %d optimizations failed." % (fail_count, len(cases)))

    # If SA_PATH does not exist, create it
    if not pth.exists(SA_PATH):
        os.makedirs(SA_PATH)
    
    # Create figures subdirectory if it doesn't exist
    figures_path = pth.join(SA_PATH, "figures")
    if not pth.exists(figures_path):
        os.makedirs(figures_path)

    # Save to .csv for future use
    df.to_csv(pth.join(SA_PATH, "doe_" + method_name + ".csv"))

    return df


def sobol_analysis(conf_file, data_file):
    """
    Interactive interface to define and simulate a Sobol' sensitivity analysis with Multiple Inputs and a Single Output.
    Plots the Sobol' indices.

    :param conf_file: configuration file of the problem
    :param data_file: output file of the initial design problem, to set up initial values.
    """

    # DATA #
    # Get variables data from file
    variables = DataFile(data_file)
    variables.sort(key=lambda var: var.name)
    table = variables.to_dataframe()[["name", "val", "units", "is_input", "desc"]].rename(
        columns={"name": "Name", "val": "Value", "units": "Unit", "desc": "Description"}
    )
    # Remove variables whose shape is different from a single value (i.e., n-dimensional arrays).
    table['type'] = [type(x) for x in table.Value.values]
    table = table[table['type'] == float].drop('type', axis=1)
    # Uncertain variables table
    x_table = table.loc[table["is_input"]]  # select inputs only
    x_table = x_table.loc[
        x_table["Name"].str.contains("uncertainty:")
    ]  # select uncertain parameters only
    x_list_short = (
        x_table["Name"].apply(get_short_name).unique().tolist()
    )  # short name (e.g. removes "uncertainty:" and ":var")
    x_list_short.append("")  # add empty name for variable un-selection

    # outputs
    y_list = table.loc[~table["is_input"]]["Name"].unique().tolist()

    # WIDGETS #
    def input_box():
        """
        Input line layout initialization (widgets)
        """
        # Input box
        inputbox = widgets.Dropdown(
            description="Uncertain parameter:   ",
            options=x_list_short,
            value=None,
            style={"description_width": "initial"},
        )
        # Values boxes
        value_box = widgets.Text(
            value="",
            description="",
            continuous_update=False,
            disabled=True,
            layout={"width": "180px"},
        )
        # Distribution laws
        law_buttons = widgets.ToggleButtons(
            options=["Uniform", "Normal"],
            description="Distribution:",
            disabled=False,
            button_style="",
        )
        # Distribution laws parameters
        var_box_normal = widgets.FloatSlider(
            value=0.1,
            min=0.0,
            max=0.5,
            step=0.01,
            description="error std:",
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format=".0%",
            style={"description_width": "initial"},
        )
        var_box_uniform = widgets.FloatRangeSlider(
            value=[-0.1, 0.1],
            min=-0.5,
            max=0.5,
            step=0.01,
            description="error interval:",
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format=".0%",
            style={"description_width": "initial"},
        )
        # Error type (relative of absolute)
        is_relative_error = widgets.Checkbox(
            value=True, description="relative", disabled=False, indent=False
        )
        return widgets.HBox([inputbox, value_box, law_buttons, var_box_uniform, is_relative_error])

    # "add input" button
    addinput_button = widgets.Button(description="add parameter")

    # Number of samples
    samples = widgets.FloatLogSlider(
        value=128,
        base=2,
        min=3,
        max=14,
        step=1,
        description="Samples:",
        continuous_update=False,
    )

    # Output of interest
    outputbox = widgets.Dropdown(
        description="Output of interest:   ",
        options=y_list,
        value=None,
        style={"description_width": "initial"},
    )

    # "Update" button
    update_button = widgets.Button(description="run simulations")

    # Second order checkbox
    second_order_box = widgets.Checkbox(
        value=True, description="Second order indices", disabled=False, indent=False
    )

    # FIGURES #
    # Bar plot
    fig1 = go.FigureWidget(
        layout=go.Layout(
            title=dict(text="Contributions to the output standard deviation"),
            yaxis=dict(title="Output standard deviation"),
            # yaxis2=dict(title='Output Variance', side='right',overlaying='y'),
        )
    )
    fig1.add_trace(go.Bar(name="Total-effect", x=[], y=[], error_y=dict(type="data", array=[])))
    fig1.add_trace(go.Bar(name="First-order", x=[], y=[], error_y=dict(type="data", array=[])))
    # fig1.add_trace(go.Bar(name='S2', x=[], y=[], error_y=dict(type='data', array=[])))
    fig1.update_layout(barmode="group")

    # Heat map
    fig2 = go.FigureWidget(
        layout=go.Layout(title=dict(text="First-order and Interaction effects")),
        data=go.Heatmap(
            z=[],
            x=[],
            y=[],
            hoverongaps=False,
            colorbar=dict(title="Sobol index", titleside="top"),
        ),
    )

    # Pie chart
    fig3 = go.FigureWidget(
        layout=go.Layout(title=dict(text="Sobol indices")),
        data=go.Sunburst(
            labels=[],
            parents=[],
            values=[],
            textinfo="label+value",
        ),
    )

    # Output distribution
    fig4 = go.FigureWidget(
        data=[go.Histogram(histnorm="probability", autobinx=True)],
        layout=go.Layout(title=dict(text="Output Distribution")),
    )

    # Parallel coordinates plot
    fig5 = go.FigureWidget(
        data=go.Parcoords(labelangle=0, labelside="top"),
        layout=go.Layout(title=dict(text="Parallel Coordinates Plot")),
    )
    fig5.update_layout(width=1000)

    # Inputs distribution (deprecated)
    # fig6 = go.FigureWidget(data=[],
    #                       layout=go.Layout(
    #                           title=dict(
    #                               text='Inputs Errors Distribution',
    #                           ),
    #                           barmode='overlay'
    #                       ))

    def add_input(change):
        """
        Add an new uncertain parameter to the problem
        """
        # add new input row
        new_input = input_box()
        inputs_array.append(new_input)
        widg.children = widg.children[:-1] + (
            new_input,
            addinput_button,
        )  # update display by adding a new row and placing the 'add' button below.
        n_input = len(inputs_array) - 1  # input row indice

        def variable_data(change):
            """
            Get and set data for the uncertain parameter
            """
            # Widgets layout
            inputbox = inputs_array[n_input].children[0]  # variable selected from dropdown
            value_box = inputs_array[n_input].children[1]  # value of the variable
            law_buttons = inputs_array[n_input].children[2]  # distribution law
            is_relative_error = inputs_array[n_input].children[
                4
            ]  # check box for selection relative or absolute error
            x_data = table.loc[
                table["Name"] == get_long_name(inputbox.value)[0]
            ]  # corresponding data from file
            if law_buttons.value == "Normal":
                new_var_box = widgets.FloatSlider(
                    value=0.1,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    description="error std:",
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    readout_format=".0%",
                    style={"description_width": "auto"},
                )
            elif law_buttons.value == "Uniform":
                new_var_box = widgets.FloatRangeSlider(
                    value=[-0.1, 0.1],
                    min=-0.5,
                    max=0.5,
                    step=0.01,
                    description="error interval:",
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    readout_format=".0%",
                    style={"description_width": "auto"},
                )
            # Widgets values
            if x_data["Value"].unique().size != 0:  # check data from selected variable exists
                x_value = x_data["Value"].unique()[0]  # get value from datafile
                x_unit = x_data["Unit"].unique()[0]  # get unit
                value_box.value = "{:10.4f} ".format(x_value) + (
                    x_unit if x_unit is not None else ""
                )  # display value and unit of selected variable
                if law_buttons.value == "Normal":
                    new_var_box.min = 0.0
                    new_var_box.max = 0.5 if is_relative_error.value else (0.5 * x_value)
                    new_var_box.value = 0.1 if is_relative_error.value else (0.1 * x_value)
                    new_var_box.readout_format = ".0%" if is_relative_error.value else ".3g"
                    new_var_box.step = 0.01 if is_relative_error.value else (0.01 * x_value)
                if law_buttons.value == "Uniform":
                    new_var_box.min = -0.5 if is_relative_error.value else (-0.5 * x_value)
                    new_var_box.max = 0.5 if is_relative_error.value else (0.5 * x_value)
                    new_var_box.value = (
                        [-0.1, 0.1] if is_relative_error.value else [-0.1 * x_value, 0.1 * x_value]
                    )
                    new_var_box.readout_format = ".0%" if is_relative_error.value else ".3g"
                    new_var_box.step = 0.01 if is_relative_error.value else (0.01 * x_value)
            else:
                value_box.value = ""
            inputs_array[n_input].children = (
                inputbox,
                value_box,
                law_buttons,
                new_var_box,
                is_relative_error,
            )

        def error_conversion(change):
            """
            Conversion from relative to absolute error, and vice-versa.
            """
            inputbox = inputs_array[n_input].children[0]  # variable selected from dropdown
            law_buttons = inputs_array[n_input].children[2]  # distribution law
            var_box = inputs_array[n_input].children[3]  # variation to apply for the DoE
            is_relative_error = inputs_array[n_input].children[
                4
            ]  # check box for selection relative or absolute error
            x_data = table.loc[
                table["Name"] == get_long_name(inputbox.value)[0]
            ]  # corresponding data from file
            if x_data["Value"].unique().size != 0:  # check data from selected variable exists
                x_value = x_data["Value"].unique()[0]  # get value from datafile
                if law_buttons.value == "Normal":
                    var_box.max = max(0.5, 0.5 * x_value)
                    if is_relative_error.value:
                        var_box.value = var_box.value / x_value
                        var_box.max = 0.5
                        var_box.readout_format = ".0%"
                        var_box.step = 0.01
                    else:
                        var_box.value = var_box.value * x_value
                        var_box.max = 0.5 * x_value
                        var_box.readout_format = ".3g"
                        var_box.step = 0.01 * x_value
                if law_buttons.value == "Uniform":
                    var_box.min = min(-0.5, -0.5 * x_value)
                    var_box.max = max(0.5, 0.5 * x_value)
                    if is_relative_error.value:
                        var_box.value = [
                            var_box.value[0] / x_value,
                            var_box.value[1] / x_value,
                        ]
                        var_box.min = -0.5
                        var_box.max = 0.5
                        var_box.readout_format = ".0%"
                        var_box.step = 0.01
                    else:
                        var_box.value = [
                            var_box.value[0] * x_value,
                            var_box.value[1] * x_value,
                        ]
                        var_box.min = -0.5 * x_value
                        var_box.max = 0.5 * x_value
                        var_box.readout_format = ".0%" if is_relative_error.value else ".3g"
                        var_box.step = 0.01 * x_value

        # add observe events to update values according to variable selection, distribution law and error type
        new_input.children[0].observe(variable_data, names="value")  # variable selection event
        new_input.children[2].observe(variable_data, names="value")  # distribution law event
        new_input.children[4].observe(error_conversion, names="value")  # error type event

    def validate(outputbox, x_dict):
        """
        Check if the problem is well defined (at least one input and an output).
        """
        if outputbox.value is None or len(x_dict) == 0:
            return False
        else:
            return True

    def run_sobol_analysis(df, x_dict, y, second_order):
        """
        Perform Sobol' Analysis on model outputs.
        """
        num_vars = len(x_dict)

        problem_sobol = {
            "num_vars": num_vars,
            "names": list(x_dict.keys()),
            "bounds": list(x_dict.values()),
        }
        Y_sobol = df[y].to_numpy()

        # save problem definition for further use if necessary
        with open(SA_PATH + "/problem_sobol.txt", "w+") as file:
            file.write(str(problem_sobol))
        with open(SA_PATH + "/y_sobol.txt", "w+") as file:
            file.write(y)

        # Run Sobol analysis with SALib library
        y_var = df[y].var()  # output variance
        if y_var > 1e-6:
            Si = sobol.analyze(problem_sobol, Y_sobol, calc_second_order=second_order)
        else:
            Si = {
                "S1": np.zeros(num_vars),
                "S1_conf": np.zeros(num_vars),
                "ST": np.zeros(num_vars),
                "ST_conf": np.zeros(num_vars),
                "S2": np.zeros((num_vars, num_vars)),
                "S2_conf": np.zeros((num_vars, num_vars)),
            }
        for s in Si:
            # Forbid negative values.
            # Sobol' indices cannot be negative, but this may happen for numerical reasons such as too small sample size
            Si[s] = Si[s].clip(0)
        return Si

    def update_sobol(change):
        """
        Based on the Monte Carlo simulations,
        perform a Sobol' analysis and update charts according to the selected output.
        """

        # Perform Sobol' analysis
        y = outputbox.value  # name of output variable of interest
        Si = run_sobol_analysis(df, x_dict, y, second_order)

        # Additional data for plots
        y_data = table.loc[table["Name"] == y]  # output variable of interest
        y_unit = y_data["Unit"].unique()[0]  # unit
        y_var = df[y].var()  # variance

        with fig1.batch_update():  # total effect and first order Sobol' indices
            fig1.data[0].x = list(get_short_name(x) for x in x_dict.keys())
            fig1.data[0].y = np.sqrt(Si["ST"] * y_var)  # contribution to standard deviation
            fig1.data[0].error_y = dict(type="data", array=np.sqrt(Si["ST_conf"] * y_var))
            fig1.data[1].x = list(get_short_name(x) for x in x_dict.keys())
            fig1.data[1].y = np.sqrt(Si["S1"] * y_var)  # contribution to standard deviation
            fig1.data[1].error_y = dict(type="data", array=np.sqrt(Si["S1_conf"] * y_var))
            fig1.update_yaxes(
                title="Standard deviation (" + (y_unit if y_unit is not None else "") + ") <br>" + y
            )
            fig1.update_xaxes(categoryorder="total descending")

        with fig2.batch_update():  # second order indices
            fig2.data[0].x = list(get_short_name(x) for x in x_dict.keys())
            fig2.data[0].y = list(get_short_name(x) for x in x_dict.keys())
            if second_order:
                z_data = Si["S2"]
            else:
                z_data = np.empty((len(x_dict), len(x_dict)))  # second-order indices
                z_data[:] = np.nan
            np.fill_diagonal(z_data, Si["S1"])  # add first order indices on diagonal
            fig2.data[0].z = z_data

        with fig3.batch_update():  # Sobol' total effect indices (pie chart)
            # sunburst title
            labels = ["Total-effect"]
            parents = [""]
            values = list([""])
            # total-effect indices
            labels += list(get_short_name(x) for x in x_dict.keys())
            parents += ["Total-effect" for i in range(len(x_dict))]
            values += list(Si["ST"])
            # First- and second-order indices
            # for i in range(len(x_dict)):
            #    key = list(x_dict.keys())[i]
            #    parents += [key]
            #    labels += ['S1']
            #    values += [Si['S1'][i]]
            #    values += [S2[i]]
            fig3.data[0].labels = labels
            fig3.data[0].parents = parents
            fig3.data[0].values = values

        with fig4.batch_update():  # Output Distribution
            fig4.data[0].x = df[y]
            fig4.layout.xaxis.title = y + " [%s]" % (
                y_data["Unit"].unique()[0] if y_data["Unit"].unique()[0] is not None else "-"
            )

        with fig5.batch_update():  # Parallel coordinate plot
            dimensions = []
            for x in x_dict:
                if x.split(":")[-1] == "rel":
                    dimensions.append(
                        dict(
                            label=get_short_name(x) + ":error",
                            values=df[x],
                            tickformat="%",
                        )
                    )  # add relative error in percentage
                else:
                    dimensions.append(
                        dict(label=get_short_name(x) + ":error", values=df[x])
                    )  # add absolute error with unit
            dimensions.append(dict(label=y, values=df[y]))
            fig5.data[0].dimensions = dimensions
            fig5.data[0].line = dict(color=df[y], colorscale="Viridis")

        # export
        fig1.write_html(pth.join(SA_PATH, "figures", "sobol_indices_hist.html"))
        fig1.write_image(pth.join(SA_PATH, "figures", "sobol_indices_hist.pdf"))
        fig2.write_html(pth.join(SA_PATH, "figures", "sobol_second_order.html"))
        fig2.write_image(pth.join(SA_PATH, "figures", "sobol_second_order.pdf"))
        fig3.write_html(pth.join(SA_PATH, "figures", "sobol_indices_pie.html"))
        fig3.write_image(pth.join(SA_PATH, "figures", "sobol_indices_pie.pdf"))
        fig4.write_html(pth.join(SA_PATH, "figures", "output_dist.html"))
        fig4.write_image(pth.join(SA_PATH, "figures", "output_dist.pdf"))
        fig5.write_html(pth.join(SA_PATH, "figures", "parallel_plot.html"))
        fig5.write_image(pth.join(SA_PATH, "figures", "parallel_plot.pdf"))

        # with fig6.batch_update():  # Inputs Distributions
        #    # DEPRECATED : both relative and absolute errors are ploted on the same chart...
        #    fig6.data = []  # empty traces
        #    for x in x_dict:  # add new traces
        #        fig6.add_trace(go.Histogram(x=df[x], name=get_short_name(x), histnorm='probability', autobinx=True))
        #    fig6.layout.xaxis.title = 'error interval'
        #    fig6.layout.xaxis.tickformat = '%'
        #    fig6.update_traces(opacity=0.75)

    def update_all(change):
        """
        Run Monte carlo simulations, Sobol' analysis and update figures
        """

        # global variables
        global x_dict, second_order, df  # needed to modify global copies of x_dict, second_order and df

        # Get user inputs
        x_dict = {}
        for inputrow in inputs_array:
            inputbox = inputrow.children[0]
            law_buttons = inputrow.children[2]
            var_box = inputrow.children[3]
            is_relative_error = inputrow.children[4]
            # x_data = table.loc[table['Name'] == get_long_name(inputbox.value)[0]]
            if inputbox.value is not None:  # not empty input
                x_var = (
                    var_box.value
                )  # single value if normal law (std), or tuple if uniform law (a,b)
                # x_value = x_data["Value"].unique()[0]
                if law_buttons.value == "Normal":
                    mu = 0
                    sigma = x_var
                    if is_relative_error.value:  # apply to relative error variable
                        x_dict[get_long_name(inputbox.value)[1]] = [
                            mu,
                            sigma,
                            "norm",
                        ]  # add bounds and distribution type
                    else:  # apply to absolute error variable
                        x_dict[get_long_name(inputbox.value)[2]] = [
                            mu,
                            sigma,
                            "norm",
                        ]  # add bounds and distribution type
                elif law_buttons.value == "Uniform":
                    a = x_var[0]
                    b = x_var[1]
                    if is_relative_error.value:  # apply to relative error variable
                        x_dict[get_long_name(inputbox.value)[1]] = [
                            a,
                            b,
                            "unif",
                        ]  # add bounds and distribution type
                    else:  # apply to absolute error variable
                        x_dict[get_long_name(inputbox.value)[2]] = [
                            a,
                            b,
                            "unif",
                        ]  # add bounds and distribution type
        if not validate(outputbox, x_dict):
            return False

        # Monte Carlo with Saltelli's sampling
        ns = int(samples.value)  # number of samples to generate
        second_order = second_order_box.value  # boolean for second order Sobol' indices calculation
        df = doe_fast("Sobol", x_dict, y_list, conf_file, ns, second_order)

        # Perform Sobol' analysis and update charts
        outputbox.observe(update_sobol, names="value")  # enable to change the output to visualize
        update_sobol(0)

    # Set up Figure
    options_panel = widgets.VBox(
        [widgets.HBox([outputbox, samples, second_order_box, update_button])]
    )
    widg = widgets.VBox(
        [
            widgets.HBox([fig1, fig2]),
            widgets.HBox(
                [
                    fig3,
                ]
            ),
            widgets.HBox([fig5, fig4]),
            # widgets.HBox([fig3, fig5]),
            # widgets.HBox([fig6,
            #              fig4]),
            options_panel,
            addinput_button,
        ],
        layout=Layout(align_items="flex-start"),
    )

    # add first input
    inputs_array = []
    add_input(0)

    # add input and update buttons interactions
    addinput_button.on_click(add_input)
    update_button.on_click(update_all)  # Run the Sobol' analysis and display charts

    return widg


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
    table = variables.to_dataframe()[["name", "val", "units", "is_input", "desc"]].rename(
        columns={"name": "Name", "val": "Value", "units": "Unit", "desc": "Description"}
    )
    # Remove variables whose shape is different from a single value (i.e., n-dimensional arrays).
    table['type'] = [type(x) for x in table.Value.values]
    table = table[table['type'] == float].drop('type', axis=1)
    # Uncertain variables table
    x_table = table.loc[table["is_input"]]  # select inputs only
    x_table = x_table.loc[
        x_table["Name"].str.contains("uncertainty:")
    ]  # select uncertain parameters only
    x_list_short = x_table["Name"].apply(get_short_name).unique().tolist()  # short name
    x_list_short.append("")  # add empty name for variable un-selection

    # Outputs table
    y_list = table.loc[~table["is_input"]]["Name"].unique().tolist()

    # WIDGETS #
    def input_box():
        """
        Input line layout initialization (widgets)
        """
        # Input box
        inputbox = widgets.Dropdown(
            description="Uncertain parameter:   ",
            options=x_list_short,
            value=None,
            style={"description_width": "initial"},
        )
        # Values boxes
        value_box = widgets.Text(
            value="",
            description="",
            continuous_update=False,
            disabled=True,
            layout={"width": "180px"},
        )
        # Distribution laws
        law_buttons = widgets.ToggleButtons(
            options=["Uniform"],  # Alternate distributions are not support yet in SALib (https://github.com/SALib/SALib/issues/515)
            description="Distribution:",
            disabled=True,
            button_style="",
        )
        # Distribution laws parameters
        var_box_normal = widgets.FloatSlider(
            value=0.1,
            min=0.0,
            max=0.5,
            step=0.01,
            description="error std:",
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format=".0%",
            style={"description_width": "initial"},
        )
        var_box_uniform = widgets.FloatRangeSlider(
            value=[-0.1, 0.1],
            min=-0.5,
            max=0.5,
            step=0.01,
            description="error interval:",
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format=".0%",
            style={"description_width": "initial"},
        )
        # Error type (relative of absolute)
        is_relative_error = widgets.Checkbox(
            value=True, description="relative", disabled=False, indent=False
        )
        return widgets.HBox([inputbox, value_box, law_buttons, var_box_uniform, is_relative_error])

    # "add input" button
    addinput_button = widgets.Button(description="add parameter")

    # Number of samples
    samples = widgets.IntSlider(
        value=8,
        min=4,
        max=64,
        step=1,
        description="Trajectories:",
        continuous_update=False,
    )

    # Output of interest
    outputbox = widgets.Dropdown(
        description="Output of interest:   ",
        options=y_list,
        value=None,
        style={"description_width": "initial"},
    )

    # "Update" button
    update_button = widgets.Button(description="run Morris")

    # FIGURES #
    # Bar plot
    fig1 = go.FigureWidget(
        layout=go.Layout(
            title=dict(text="Morris results"),
            yaxis=dict(title=r"$\mu^*$"),
            font=dict(size=14),
        )
    )
    fig1.add_trace(go.Bar(name="mu_star", x=[], y=[], error_y=dict(type="data", array=[])))

    # Scatter plot
    fig2 = go.FigureWidget(
        layout=go.Layout(
            xaxis=dict(title=r"$\mu^*$", titlefont=dict(size=20)),
            yaxis=dict(title=r"$\sigma$", titlefont=dict(size=20)),
            legend=dict(title="", orientation="h", bordercolor="black", borderwidth=1),
            font=dict(size=14),
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
        """
        Add an new uncertain parameter to the problem
        """
        # add new input row
        new_input = input_box()
        inputs_array.append(new_input)
        widg.children = widg.children[:-1] + (
            new_input,
            addinput_button,
        )  # update display by adding a new row and placing the 'add' button below.
        n_input = len(inputs_array) - 1  # input row indice

        def variable_data(change):
            """
            Get and set data for the uncertain parameter
            """
            # Widgets layout
            inputbox = inputs_array[n_input].children[0]  # variable selected from dropdown
            value_box = inputs_array[n_input].children[1]  # value of the variable
            law_buttons = inputs_array[n_input].children[2]  # distribution law
            is_relative_error = inputs_array[n_input].children[
                4
            ]  # check box for selection relative or absolute error
            x_data = table.loc[
                table["Name"] == get_long_name(inputbox.value)[0]
            ]  # corresponding data from file
            if law_buttons.value == "Normal":
                new_var_box = widgets.FloatSlider(
                    value=0.1,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    description="error std:",
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    readout_format=".0%",
                    style={"description_width": "auto"},
                )
            elif law_buttons.value == "Uniform":
                new_var_box = widgets.FloatRangeSlider(
                    value=[-0.1, 0.1],
                    min=-0.5,
                    max=0.5,
                    step=0.01,
                    description="error interval:",
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    readout_format=".0%",
                    style={"description_width": "auto"},
                )
            # Widgets values
            if x_data["Value"].unique().size != 0:  # check data from selected variable exists
                x_value = x_data["Value"].unique()[0]  # get value from datafile
                x_unit = x_data["Unit"].unique()[0]  # get unit
                value_box.value = "{:10.4f} ".format(x_value) + (
                    x_unit if x_unit is not None else ""
                )  # display value and unit of selected variable
                if law_buttons.value == "Normal":
                    new_var_box.min = 0.0
                    new_var_box.max = 0.5 if is_relative_error.value else (0.5 * x_value)
                    new_var_box.value = 0.1 if is_relative_error.value else (0.1 * x_value)
                    new_var_box.readout_format = ".0%" if is_relative_error.value else ".3g"
                    new_var_box.step = 0.01 if is_relative_error.value else (0.01 * x_value)
                if law_buttons.value == "Uniform":
                    new_var_box.min = -0.5 if is_relative_error.value else (-0.5 * x_value)
                    new_var_box.max = 0.5 if is_relative_error.value else (0.5 * x_value)
                    new_var_box.value = (
                        [-0.1, 0.1] if is_relative_error.value else [-0.1 * x_value, 0.1 * x_value]
                    )
                    new_var_box.readout_format = ".0%" if is_relative_error.value else ".3g"
                    new_var_box.step = 0.01 if is_relative_error.value else (0.01 * x_value)
            else:
                value_box.value = ""
            inputs_array[n_input].children = (
                inputbox,
                value_box,
                law_buttons,
                new_var_box,
                is_relative_error,
            )

        def error_conversion(change):
            """
            Conversion from relative to absolute error, and vice-versa.
            """
            inputbox = inputs_array[n_input].children[0]  # variable selected from dropdown
            law_buttons = inputs_array[n_input].children[2]  # distribution law
            var_box = inputs_array[n_input].children[3]  # variation to apply for the DoE
            is_relative_error = inputs_array[n_input].children[
                4
            ]  # check box for selection relative or absolute error
            x_data = table.loc[
                table["Name"] == get_long_name(inputbox.value)[0]
            ]  # corresponding data from file
            if x_data["Value"].unique().size != 0:  # check data from selected variable exists
                x_value = x_data["Value"].unique()[0]  # get value from datafile
                if law_buttons.value == "Normal":
                    var_box.max = max(0.5, 0.5 * x_value)
                    if is_relative_error.value:
                        var_box.value = var_box.value / x_value
                        var_box.max = 0.5
                        var_box.readout_format = ".0%"
                        var_box.step = 0.01
                    else:
                        var_box.value = var_box.value * x_value
                        var_box.max = 0.5 * x_value
                        var_box.readout_format = ".3g"
                        var_box.step = 0.01 * x_value
                if law_buttons.value == "Uniform":
                    var_box.min = min(-0.5, -0.5 * x_value)
                    var_box.max = max(0.5, 0.5 * x_value)
                    if is_relative_error.value:
                        var_box.value = [
                            var_box.value[0] / x_value,
                            var_box.value[1] / x_value,
                        ]
                        var_box.min = -0.5
                        var_box.max = 0.5
                        var_box.readout_format = ".0%"
                        var_box.step = 0.01
                    else:
                        var_box.value = [
                            var_box.value[0] * x_value,
                            var_box.value[1] * x_value,
                        ]
                        var_box.min = -0.5 * x_value
                        var_box.max = 0.5 * x_value
                        var_box.readout_format = ".0%" if is_relative_error.value else ".3g"
                        var_box.step = 0.01 * x_value

        # add observe events to update values according to variable selection, distribution law and error type
        new_input.children[0].observe(variable_data, names="value")  # variable selection event
        new_input.children[2].observe(variable_data, names="value")  # distribution law event
        new_input.children[4].observe(error_conversion, names="value")  # error type event

    def validate(outputbox, x_dict):
        """
        Check if the problem is well defined (at least one input and an output).
        """
        if outputbox.value is None or len(x_dict) <= 1:
            return False
        # FIXME: display warning error
        else:
            return True

    def run_morris_analysis(df, x_dict, y):
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

        # save problem definition for further use if necessary
        with open(SA_PATH + "/problem_morris.txt", "w+") as file:
            file.write(str(problem_morris))
        with open(SA_PATH + "/y_morris.txt", "w+") as file:
            file.write(y)
        with open(SA_PATH + "/x_morris.txt", "w+") as file:
            file.write(str(list(x_dict.keys())))

        X_morris = df[list(x_dict.keys())].to_numpy()
        Y_morris = df[y].to_numpy()
        Si = morris.analyze(problem_morris, X_morris, Y_morris, conf_level=0.95, num_resamples=100)

        return Si  # sensitivity indices

    def update_morris(change):
        """
        Based on the simulations,
        perform a Morris analysis and update charts according to the selected output.
        """

        # Perform Morris analysis
        y = outputbox.value  # name of output variable of interest
        Si = run_morris_analysis(df, x_dict, y)  # sensitivity indices

        # Update figures
        y_data = table.loc[table["Name"] == y]  # corresponding data from file
        y_unit = y_data["Unit"].unique()[0]  # get unit

        with fig1.batch_update():
            fig1.data[0].x = list(get_short_name(x) for x in x_dict.keys())
            fig1.data[0].y = Si["mu_star"]
            fig1.data[0].error_y = dict(type="data", array=Si["mu_star_conf"])
            fig1.update_layout(
                yaxis=dict(
                    title="$\\mu^* \\text{ (" + (y_unit if y_unit is not None else "") + ")}$"
                ),
                xaxis=dict(categoryorder="total descending"),
                font=dict(size=14),
            )

        with fig2.batch_update():
            # clear previous data (except reference axes)
            fig2.data = [fig2.data[0], fig2.data[1], fig2.data[2]]

            # update reference axes
            fig2.data[0].x = [0, max(Si["mu_star"])]
            fig2.data[0].y = [0, max(Si["mu_star"])]
            fig2.data[1].x = [0, 2 * max(Si["mu_star"])]
            fig2.data[1].y = [0, max(Si["mu_star"])]
            fig2.data[2].x = [0, 10 * max(Si["mu_star"])]
            fig2.data[2].y = [0, max(Si["mu_star"])]

            # add parameters
            raw_symbols = SymbolValidator().values
            for i in range(len(Si["names"])):
                fig2.add_trace(
                    go.Scatter(
                        x=[Si["mu_star"][i]],
                        y=[Si["sigma"][i]],
                        mode="markers",
                        name=get_short_name(Si["names"][i]),
                        marker=dict(symbol=raw_symbols[12 * i], size=10),
                    )
                )

            # scale
            fig2.update_layout(
                xaxis=dict(
                    title="$\\mu^* \\text{ (" + (y_unit if y_unit is not None else "") + ")}$"
                ),
                yaxis=dict(title="$\\sigma$"),
                xaxis_range=[-0.05, 1.2 * max(Si["mu_star"])],
                yaxis_range=[
                    -0.05,
                    1.2 * max(0.1 * max(Si["mu_star"]), max(Si["sigma"])),
                ],
                legend=dict(title="", orientation="h", bordercolor="black", borderwidth=1),
                font=dict(size=14),
            )

            # export
            fig1.write_html(pth.join(SA_PATH, "figures", "morris_mu.html"), include_mathjax="cdn")
            fig1.write_image(pth.join(SA_PATH, "figures", "morris_mu.pdf"))
            fig2.write_html(pth.join(SA_PATH, "figures", "morris_mu_sigma.html"), include_mathjax="cdn")
            fig2.write_image(pth.join(SA_PATH, "figures", "morris_mu_sigma.pdf"))

    def update_all(change):
        """
        Run method of Morris and update figures
        """

        # global variables
        global x_dict, df  # needed to modify global copies of x_dict, second_order and df

        # Get user inputs
        x_dict = {}
        for inputrow in inputs_array:
            inputbox = inputrow.children[0]
            law_buttons = inputrow.children[2]
            var_box = inputrow.children[3]
            is_relative_error = inputrow.children[4]
            # x_data = table.loc[table['Name'] == get_long_name(inputbox.value)[0]]
            if inputbox.value is not None:  # not empty input
                x_var = (
                    var_box.value
                )  # single value if normal law (std), or tuple if uniform law (a,b)
                # x_value = x_data["Value"].unique()[0]
                if law_buttons.value == "Normal":
                    mu = 0
                    sigma = x_var
                    if is_relative_error.value:  # apply to relative error variable
                        x_dict[get_long_name(inputbox.value)[1]] = [
                            mu,
                            sigma,
                            "norm",
                        ]  # add bounds and distribution type
                    else:  # apply to absolute error variable
                        x_dict[get_long_name(inputbox.value)[2]] = [
                            mu,
                            sigma,
                            "norm",
                        ]  # add bounds and distribution type
                elif law_buttons.value == "Uniform":
                    a = x_var[0]
                    b = x_var[1]
                    if is_relative_error.value:  # apply to relative error variable
                        x_dict[get_long_name(inputbox.value)[1]] = [
                            a,
                            b,
                            "unif",
                        ]  # add bounds and distribution type
                    else:  # apply to absolute error variable
                        x_dict[get_long_name(inputbox.value)[2]] = [
                            a,
                            b,
                            "unif",
                        ]  # add bounds and distribution type
        if not validate(outputbox, x_dict):
            return False

        # Run DoEs
        nt = int(samples.value)  # number of trajectories for morris method
        df = doe_fast("Morris", x_dict, y_list, conf_file, nt)

        # Perform method of Morris on results and update charts
        outputbox.observe(update_morris, names="value")  # enable to change the output to visualize
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
