import contextlib
import os
from utils.drivers.salib_doe_driver import SalibDOEDriver
import fastoad.api as oad
from fastoad.io.variable_io import DataFile
import openmdao.api as om
import pandas as pd
import numpy as np
from ipywidgets import widgets, Layout
import plotly.graph_objects as go
from SALib.analyze import sobol
from typing import List


def doe_sobol(x_dict: dict, y_list: List[str], conf_file: str, ns: int = 1024,
              calc_second_order: bool = True) -> pd.DataFrame:
    """
    DoE for Sobol-Saltelli 2002 Method.
    In this version, a nested optimization (sub-problem) is run to ensure system consistency at each simulation.

    :param x_dict: inputs dictionary {input_name: distribution_law}
    :param y_list: list of problem outputs
    :param conf_file: configuration file for the problem
    :param ns: number of samples for MC simulation
    :param calc_second_order: calculate second order indices (boolean)

    :return: dataframe of the monte carlo simulation results
    """

    class SubProbComp(om.ExplicitComponent):
        """
        Sub-problem component for nested optimization to ensure system consistency.
        """

        def initialize(self):
            self.options.declare('conf')
            self.options.declare('x_list')
            self.options.declare('y_list')

        def setup(self):
            # create a sub-problem to use later in the compute
            # sub_conf = oad.FASTOADProblemConfigurator(conf_file)
            conf = self.options['conf']
            prob = conf.get_problem(read_inputs=True)
            prob.driver.options['disp'] = False
            p = self._prob = prob
            p.setup()

            # set counter for optimization failure
            self._fail_count = 0

            # define the i/o of the component
            x_list = self._x_list = self.options['x_list']
            y_list = self._y_list = self.options['y_list']

            for x in x_list:
                self.add_input(x)

            for y in y_list:
                self.add_output(y)

            self.declare_partials('*', '*', method='fd')

        def compute(self, inputs, outputs):
            p = self._prob
            x_list = self._x_list
            y_list = self._y_list

            for x in x_list:
                p[x] = inputs[x]

            with open(os.devnull, "w") as f, contextlib.redirect_stdout(
                    f):  # turn off all convergence messages (including failures)
                fail = p.run_driver()

            for y in y_list:
                outputs[y] = p[y]

            if fail:
                self._fail_count += 1

    conf = oad.FASTOADProblemConfigurator(conf_file)
    prob_definition = conf.get_optimization_definition()
    x_list = [x_name for x_name in x_dict.keys()]

    # CASE 1: nested optimization is declared (i.e. optimization problem is defined in configuration file)
    if 'objective' in prob_definition.keys():
        nested_optimization = True
        prob = om.Problem()
        prob.model.add_subsystem('sub_prob', SubProbComp(conf=conf, x_list=x_list, y_list=y_list), promotes=['*'])

    # CASE 2: simple model without optimization
    else:
        nested_optimization = False
        prob = conf.get_problem(read_inputs=True)

    # DoE parameters
    dists = []
    for x_name, x_value in x_dict.items():
        prob.model.add_design_var(x_name, lower=x_value[0],
                                  upper=x_value[1])  # add input parameter for DoE
        dist = x_value[2]  # add distribution type ('unif' or 'norm')
        dists.append(dist)

    # Setup driver (Sobol method with Saltelli sample)
    prob.driver = SalibDOEDriver(
        sa_method_name="Sobol",
        sa_doe_options={"n_samples": ns, "calc_second_order": calc_second_order},
        distributions=dists,
    )

    # Attach recorder to the driver
    if os.path.exists("cases.sql"):
        os.remove("cases.sql")
    prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
    prob.driver.recording_options['includes'] = x_list + y_list  # include all variables from the problem

    # Run problem
    prob.setup()
    prob.run_driver()
    prob.cleanup()

    # Get results from recorded cases
    df = pd.DataFrame()
    cr = om.CaseReader("cases.sql")
    cases = cr.list_cases('driver', out_stream=None)
    for case in cases:
        values = cr.get_case(case).outputs
        df = df.append(values, ignore_index=True)

    for i in df.columns:
        df[i] = df[i].apply(lambda x: x[0])

    # Print number of optimization failures
    fail_count = prob.model.sub_prob._fail_count if nested_optimization else 0  # count number of failures for nested optimization
    if fail_count > 0:
        print("%d out of %d optimizations failed." % (fail_count, len(cases)))

    # save to .csv for future use
    df.to_csv('workdir/sobol/doe.csv')

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
    # Uncertain variables table
    x_table = table.loc[table['is_input']]  # select inputs only
    x_table = x_table.loc[x_table["Name"].str.contains('uncertainty:')]  # select uncertain parameters only
    x_list_short = x_table["Name"].apply(
        get_short_name).unique().tolist()  # short name (e.g. removes "uncertainty:" and ":var")
    x_list_short.append("")  # add empty name for variable un-selection

    # outputs
    y_list = table.loc[~table['is_input']]["Name"].unique().tolist()

    # WIDGETS #
    def input_box():
        """
        Input line layout initialization (widgets)
        """
        # Input box
        inputbox = widgets.Dropdown(
            description='Uncertain parameter:   ',
            options=x_list_short,
            value=None,
            style={'description_width': 'initial'}
        )
        # Values boxes
        value_box = widgets.Text(
            value='',
            description='',
            continuous_update=False,
            disabled=True,
            layout={"width": "180px"}
        )
        # Distribution laws
        law_buttons = widgets.ToggleButtons(
            options=['Uniform', 'Normal'],
            description='Distribution:',
            disabled=False,
            button_style='',
        )
        # Distribution laws parameters
        var_box_normal = widgets.FloatSlider(
            value=0.1,
            min=0.0,
            max=0.5,
            step=0.01,
            description='error std:',
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format='.0%',
            style={'description_width': 'initial'}
        )
        var_box_uniform = widgets.FloatRangeSlider(
            value=[-0.1, 0.1],
            min=-0.5,
            max=0.5,
            step=0.01,
            description='error interval:',
            disabled=False,
            continuous_update=False,
            readout=True,
            readout_format='.0%',
            style={'description_width': 'initial'}
        )
        # Error type (relative of absolute)
        is_relative_error = widgets.Checkbox(
            value=True,
            description='relative',
            disabled=False,
            indent=False
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
        description='Samples:',
        continuous_update=False
    )

    # Output of interest
    outputbox = widgets.Dropdown(
        description='Output of interest:   ',
        options=y_list,
        value=None,
        style={'description_width': 'initial'}
    )

    # "Update" button
    update_button = widgets.Button(description="run simulations")

    # Second order checkbox
    second_order_box = widgets.Checkbox(
        value=True,
        description='Second order indices',
        disabled=False,
        indent=False
    )

    # FIGURES #
    # Bar plot
    fig1 = go.FigureWidget(layout=go.Layout(
        title=dict(
            text='Contributions to the output standard deviation'
        ),
        yaxis=dict(title='Output standard deviation'),
        # yaxis2=dict(title='Output Variance', side='right',overlaying='y'),
    ))
    fig1.add_trace(go.Bar(name='Total-effect', x=[], y=[], error_y=dict(type='data', array=[])))
    fig1.add_trace(go.Bar(name='First-order', x=[], y=[], error_y=dict(type='data', array=[])))
    # fig1.add_trace(go.Bar(name='S2', x=[], y=[], error_y=dict(type='data', array=[])))
    fig1.update_layout(barmode='group')

    # Heat map
    fig2 = go.FigureWidget(layout=go.Layout(
        title=dict(
            text='Interaction effects'
        )),
        data=go.Heatmap(
            z=[],
            x=[],
            y=[],
            hoverongaps=False,
            colorbar=dict(title='Sobol index', titleside="top")
        ))

    # Pie chart
    fig3 = go.FigureWidget(layout=go.Layout(
        title=dict(
            text='Sobol indices'
        )),
        data=go.Sunburst(
            labels=[],
            parents=[],
            values=[],
            textinfo='label+value',
        ))

    # Output distribution
    fig4 = go.FigureWidget(data=[go.Histogram(histnorm='probability', autobinx=True)],
                           layout=go.Layout(
                               title=dict(
                                   text='Output Distribution'
                               )
                           ))

    # Parallel coordinates plot
    fig5 = go.FigureWidget(data=go.Parcoords(labelangle=0, labelside='top'),
                           layout=go.Layout(
                               title=dict(
                                   text='Parallel Coordinates Plot'
                               )
                           ))
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
        new_input, addinput_button)  # update display by adding a new row and placing the 'add' button below.
        n_input = len(inputs_array) - 1  # input row indice

        def variable_data(change):
            """
            Get and set data for the uncertain parameter
            """
            # Widgets layout
            inputbox = inputs_array[n_input].children[0]  # variable selected from dropdown
            value_box = inputs_array[n_input].children[1]  # value of the variable
            law_buttons = inputs_array[n_input].children[2]  # distribution law
            is_relative_error = inputs_array[n_input].children[4]  # check box for selection relative or absolute error
            x_data = table.loc[table['Name'] == get_long_name(inputbox.value)[0]]  # corresponding data from file
            if law_buttons.value == "Normal":
                new_var_box = widgets.FloatSlider(
                    value=0.1,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    description='error std:',
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    readout_format='.0%',
                    style={'description_width': 'auto'},
                )
            elif law_buttons.value == "Uniform":
                new_var_box = widgets.FloatRangeSlider(
                    value=[-0.1, 0.1],
                    min=-0.5,
                    max=0.5,
                    step=0.01,
                    description='error interval:',
                    disabled=False,
                    continuous_update=False,
                    readout=True,
                    readout_format='.0%',
                    style={'description_width': 'auto'},
                )
            # Widgets values
            if x_data["Value"].unique().size != 0:  # check data from selected variable exists
                x_value = x_data["Value"].unique()[0]  # get value from datafile
                x_unit = x_data["Unit"].unique()[0]  # get unit
                value_box.value = "{:10.4f} ".format(x_value) + (
                    x_unit if x_unit is not None else '')  # display value and unit of selected variable
                if law_buttons.value == "Normal":
                    new_var_box.min = 0.0
                    new_var_box.max = 0.5 if is_relative_error.value else (0.5 * x_value)
                    new_var_box.value = 0.1 if is_relative_error.value else (0.1 * x_value)
                    new_var_box.readout_format = '.0%' if is_relative_error.value else '.3g'
                    new_var_box.step = 0.01 if is_relative_error.value else (0.01 * x_value)
                if law_buttons.value == "Uniform":
                    new_var_box.min = -0.5 if is_relative_error.value else (-0.5 * x_value)
                    new_var_box.max = 0.5 if is_relative_error.value else (0.5 * x_value)
                    new_var_box.value = [-0.1, 0.1] if is_relative_error.value else [-0.1 * x_value, 0.1 * x_value]
                    new_var_box.readout_format = '.0%' if is_relative_error.value else '.3g'
                    new_var_box.step = 0.01 if is_relative_error.value else (0.01 * x_value)
            else:
                value_box.value = ""
            inputs_array[n_input].children = (inputbox, value_box, law_buttons, new_var_box, is_relative_error)

        def error_conversion(change):
            """
            Conversion from relative to absolute error, and vice-versa.
            """
            inputbox = inputs_array[n_input].children[0]  # variable selected from dropdown
            law_buttons = inputs_array[n_input].children[2]  # distribution law
            var_box = inputs_array[n_input].children[3]  # variation to apply for the DoE
            is_relative_error = inputs_array[n_input].children[4]  # check box for selection relative or absolute error
            x_data = table.loc[table['Name'] == get_long_name(inputbox.value)[0]]  # corresponding data from file
            if x_data["Value"].unique().size != 0:  # check data from selected variable exists
                x_value = x_data["Value"].unique()[0]  # get value from datafile
                if law_buttons.value == "Normal":
                    var_box.max = max(0.5, 0.5 * x_value)
                    if is_relative_error.value:
                        var_box.value = var_box.value / x_value
                        var_box.max = 0.5
                        var_box.readout_format = '.0%'
                        var_box.step = 0.01
                    else:
                        var_box.value = var_box.value * x_value
                        var_box.max = 0.5 * x_value
                        var_box.readout_format = '.3g'
                        var_box.step = 0.01 * x_value
                if law_buttons.value == "Uniform":
                    var_box.min = min(-0.5, -0.5 * x_value)
                    var_box.max = max(0.5, 0.5 * x_value)
                    if is_relative_error.value:
                        var_box.value = [var_box.value[0] / x_value, var_box.value[1] / x_value]
                        var_box.min = -0.5
                        var_box.max = 0.5
                        var_box.readout_format = '.0%'
                        var_box.step = 0.01
                    else:
                        var_box.value = [var_box.value[0] * x_value, var_box.value[1] * x_value]
                        var_box.min = -0.5 * x_value
                        var_box.max = 0.5 * x_value
                        var_box.readout_format = '.0%' if is_relative_error.value else '.3g'
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

    def sobol_analysis(df, x_dict, y, second_order):
        """
        Perform Sobol' Analysis on model outputs.
        """
        num_vars = len(x_dict)

        problem_sobol = {
            'num_vars': num_vars,
            'names': list(x_dict.keys()),
            'bounds': list(x_dict.values())}
        Y_sobol = df[y].to_numpy()

        y_var = df[y].var()  # output variance
        if y_var > 1e-6:
            Si = sobol.analyze(problem_sobol, Y_sobol, calc_second_order=second_order)
        else:
            Si = {'S1': np.zeros(num_vars), 'S1_conf': np.zeros(num_vars), 'ST': np.zeros(num_vars),
                  'ST_conf': np.zeros(num_vars), 'S2': np.zeros((num_vars, num_vars)),
                  'S2_conf': np.zeros((num_vars, num_vars))}
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
        Si = sobol_analysis(df, x_dict, y, second_order)

        # Additional data for plots
        y_data = table.loc[table['Name'] == y]  # output variable of interest
        y_unit = y_data["Unit"].unique()[0]  # unit
        y_var = df[y].var()  # variance

        with fig1.batch_update():  # total effect and first order Sobol' indices
            fig1.data[0].x = list(get_short_name(x) for x in x_dict.keys())
            fig1.data[0].y = np.sqrt(Si['ST'] * y_var)
            fig1.data[0].error_y = dict(type='data', array=np.sqrt(Si['ST_conf'] * y_var))
            fig1.data[1].x = list(get_short_name(x) for x in x_dict.keys())
            fig1.data[1].y = np.sqrt(Si['S1'] * y_var)
            fig1.data[1].error_y = dict(type='data', array=np.sqrt(Si['S1_conf'] * y_var))
            fig1.update_yaxes(title='Standard deviation (' + (y_unit if y_unit is not None else '') + ') <br>' + y)
            fig1.update_xaxes(categoryorder='total descending')

        with fig2.batch_update():  # second order indices
            fig2.data[0].x = list(get_short_name(x) for x in x_dict.keys())
            fig2.data[0].y = list(get_short_name(x) for x in x_dict.keys())
            if second_order:
                z_data = Si['S2']
            else:
                z_data = np.empty((len(x_dict), len(x_dict)))  # second-order indices
                z_data[:] = np.nan
            np.fill_diagonal(z_data, Si['ST'])  # add total order indices on diagonal
            fig2.data[0].z = z_data

        with fig3.batch_update():  # Sobol' total effect indices (pie chart)
            # sunburst title
            labels = ["Total-effect"]
            parents = [""]
            values = list([""])
            # total-effect indices
            labels += list(get_short_name(x) for x in x_dict.keys())
            parents += ["Total-effect" for i in range(len(x_dict))]
            values += list(Si['ST'])
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
            fig4.layout.xaxis.title = y + ' [%s]' % (
                y_data["Unit"].unique()[0] if y_data["Unit"].unique()[0] is not None else "-")

        with fig5.batch_update():  # Parallel coordinate plot
            dimensions = []
            for x in x_dict:
                if x.split(':')[-1] == 'rel':
                    dimensions.append(dict(label=get_short_name(x) + ':error', values=df[x],
                                           tickformat='%'))  # add relative error in percentage
                else:
                    dimensions.append(
                        dict(label=get_short_name(x) + ':error', values=df[x]))  # add absolute error with unit
            dimensions.append(dict(label=y, values=df[y]))
            fig5.data[
                0].dimensions = dimensions
            fig5.data[0].line = dict(color=df[y], colorscale='Viridis')

        # export
        fig1.write_html('workdir/sobol/indices_hist.html')
        fig1.write_image('workdir/sobol/indices_hist.pdf')
        fig2.write_html('workdir/sobol/second_order.html')
        fig2.write_image('workdir/sobol/second_order.pdf')
        fig3.write_html('workdir/sobol/indices_pie.html')
        fig3.write_image('workdir/sobol/indices_pie.pdf')
        fig4.write_html('workdir/sobol/output_dist.html')
        fig4.write_image('workdir/sobol/output_dist.pdf')
        fig5.write_html('workdir/sobol/parallel_plot.html')
        fig5.write_image('workdir/sobol/parallel_plot.pdf')

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
                x_var = var_box.value  # single value if normal law (std), or tuple if uniform law (a,b)
                # x_value = x_data["Value"].unique()[0]
                if law_buttons.value == "Normal":
                    mu = 0
                    sigma = x_var
                    if is_relative_error.value:  # apply to relative error variable
                        x_dict[get_long_name(inputbox.value)[1]] = [mu, sigma,
                                                                    'norm']  # add bounds and distribution type
                    else:  # apply to absolute error variable
                        x_dict[get_long_name(inputbox.value)[2]] = [mu, sigma,
                                                                    'norm']  # add bounds and distribution type
                elif law_buttons.value == "Uniform":
                    a = x_var[0]
                    b = x_var[1]
                    if is_relative_error.value:  # apply to relative error variable
                        x_dict[get_long_name(inputbox.value)[1]] = [a, b, 'unif']  # add bounds and distribution type
                    else:  # apply to absolute error variable
                        x_dict[get_long_name(inputbox.value)[2]] = [a, b, 'unif']  # add bounds and distribution type
        if not validate(outputbox, x_dict):
            return False

        # Monte Carlo with Saltelli's sampling
        ns = int(samples.value)  # number of samples to generate
        second_order = second_order_box.value  # boolean for second order Sobol' indices calculation
        df = doe_sobol(x_dict, y_list, conf_file, ns, second_order)

        # Perform Sobol' analysis and update charts
        outputbox.observe(update_sobol, names="value")  # enable to change the output to visualize
        update_sobol(0)

    # Set up Figure
    options_panel = widgets.VBox([widgets.HBox([outputbox, samples, second_order_box, update_button])])
    widg = widgets.VBox([widgets.HBox([fig1, fig2]),
                         widgets.HBox([fig3, ]),
                         widgets.HBox([fig5, fig4]),
                         # widgets.HBox([fig3, fig5]),
                         # widgets.HBox([fig6,
                         #              fig4]),
                         options_panel,
                         addinput_button,
                         ],
                        layout=Layout(align_items="flex-start"))

    # add first input
    inputs_array = []
    add_input(0)

    # add input and update buttons interactions
    addinput_button.on_click(add_input)
    update_button.on_click(update_all)  # Run the Sobol' analysis and display charts

    return widg


def get_short_name(long_name):
    """
    Return the short name of a variable.
    The short name is defined as the initial name but without the first and last filters (separated by ':')
    """
    if long_name is None:
        return ''
    short_name = ':'.join(long_name.split(':')[1:-1])  # removes first and last filters
    return short_name


def get_long_name(short_name):
    """
    Return the long name of a variable.
    The long name is defined as the short name to which a first filter and last filter are added (with separator ':').
    """
    if short_name is None:
        return '', '', ''
    long_name_1 = 'data:' + short_name + ':estimated'
    long_name_2 = 'uncertainty:' + short_name + ':rel'
    long_name_3 = 'uncertainty:' + short_name + ':abs'
    return long_name_1, long_name_2, long_name_3
